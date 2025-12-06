#include "mesh/avk-mesh.h"

#include "os/avk-log.h"

// TODO: TINYGLTF_ANDROID_LOAD_FROM_ASSETS
// https://github.com/syoyo/tinygltf/tree/release

// tinyGLTF plus STB
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define STBI_NO_SIMD  // TODO on windows clang, somehow doesn't find intrinsics
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

// imageinfo for header inspection
#include <imageinfo.hpp>

// glm library
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// for the length square check
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/epsilon.hpp>
#include <glm/gtx/norm.hpp>

// lib
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Note: The glTF 2.0 specification doesn't specify any standardized name for
// primitive attributes. What we require are the names used by blender's GLTF
// exporter

// Note: Each primitive has only one material

// Note: uri is to be decoded (eg whitespace is %20)

// Note: we ignore the samplers (TODO)

// TODO: basisu glTF Extension for KTX texture support

// TODO: imageinfo reader for AAsset: https://github.com/xiaozhuai/imageinfo

// Steps to filter mesh import
// - For each accessor:
//    - reject glTF file if there are any sparse accessors
// - reject glTF file using any extension (TODO)
// - reject glTF file if any image is not using a uri or there's an invalid uri
// - reject glTF file if any image is not using pixel type byte
// - reject glTF file if there are any skins
// - reject glTF file if there are any animations
// - For each primitive inside a mesh (Rejections):
//    - reject if any primitives morph targets
//    - reject if any primitives whose attribute is not one of POSITION, NORMAL,
//      TANGENT, TEXCOORD_0
//      - if POSITION is missing, reject mesh
//      - if POSITION accessor is not of type VEC3, reject mesh
//      - if POSITION accessor has componentType !=
//           TINYGLTF_COMPONENT_TYPE_FLOAT (5126), reject mesh
//      - if TEXCOORD_0 present, ignore them if accessor type != VEC2, or
//           componentType is not TINYGLTF_COMPONENT_TYPE_FLOAT (5126)
//      - if NORMAL present, ignore them if accessor type != VEC3 or
//           componentType is not TINYGLTF_COMPONENT_TYPE_FLOAT (5126)
//           (TODO octahedral encoding)
//      - if NORMAL present, ignore them if accessor type != VEC3 or
//           componentType is not TINYGLTF_COMPONENT_TYPE_FLOAT (5126)
//           (TODO octahedral encoding)
//    - (TODO) reject if any primitives whose mode is not "triangles" (4) ==
//      TINYGLTF_MODE_TRIANGLES
//    - (TODO) reject if any primitives don't have a valid "indices"
//      or if the used accessor doesn't have componentType unsigned int (5125)
//      TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT and type SCALAR
// - For each primitive inside a mesh (Generation):
//    - if attribute NORMAL, TANGENT, TEXCOORD_0 is missing, generate it
//      - for normal: assume smooth shading for each indexed triangle
//      - for tangent: gram-schmidt
//      - for uv: orthographic projection toward +y

namespace avk {

static char const* SupportedImageFormats[] = {".png", ".jpeg", ".jpg"};

static bool hasExtension(const std::filesystem::path& p,
                         const char* const* exts) {
  std::string ext = p.extension().string();  // includes '.', e.g. ".png"

  for (const char* e = *exts ? exts[0] : nullptr; e; ++exts, e = *exts) {
    // Compare case-insensitively
    if (ext.size() == std::char_traits<char>::length(e) &&
        std::equal(ext.begin(), ext.end(), e,
                   [](char a, char b) { return tolower(a) == tolower(b); })) {
      return true;
    }
  }

  return false;
}

inline static glm::vec4 floatGlm4VecFromDoubleArray(
    std::vector<double> const& vec) {
  assert(vec.size() == 4);
  if (vec.size() != 4) return glm::vec4{1};
  glm::vec4 const ret{vec[0], vec[1], vec[2], vec[3]};
  return ret;
}

inline static glm::vec3 floatGlm3VecFromDoubleArray(
    std::vector<double> const& vec) {
  assert(vec.size() == 3);
  if (vec.size() != 3) return glm::vec3{1};
  glm::vec3 const ret{vec[0], vec[1], vec[2]};
  return ret;
}

// ------------------ Entry Types and bookkeping methods --------------------
/// id is implicitly stored in the map
struct Entry {
  Entry() : refCount(0) {}
  Entry(int count) : refCount(count) {}

  // Move constructor
  Entry(Entry&& other) noexcept {
    refCount.store(other.refCount.load(std::memory_order_relaxed),
                   std::memory_order_relaxed);
    other.refCount.store(0, std::memory_order_relaxed);
  }

  // Move assignment
  Entry& operator=(Entry&& other) noexcept {
    if (this != &other) {
      refCount.store(other.refCount.load(std::memory_order_relaxed),
                     std::memory_order_relaxed);
      other.refCount.store(0, std::memory_order_relaxed);
    }
    return *this;
  }

  std::atomic_int refCount;
};

/// \note supports only byte pixel formats (TODO)
struct TextureEntry : public Entry {
  uint32_t width;
  uint32_t height;
  uint32_t channels;

  inline size_t bytes() const {
    return static_cast<size_t>(width) * height * channels;
  }

  std::vector<unsigned char> data;
};

/// \note 0 id means no texture
/// \note glTF supports a uv slot for each texture. we don't (TODO)
struct MaterialEntry : public Entry {
  // PBR slots
  id_t albedoTex;
  id_t metalRoughnessTex;
  glm::vec4 albedoFactor;
  float roughnessFactor;
  float metallicFactor;

  // other slots
  id_t normalTex;  // if 0 use geometric normals
  id_t emissiveTex;
  glm::vec3 emissiveFactor;
};

/// Utility struct to keep track of processed GLTF textures (ignoring sampler)
struct TexSpec {
  int imageIndex;  // index of image used
  int texCoord;    // index of uv map used
  float scale;     // normal and occlusion use this
  id_t MaterialEntry::* field;
  std::string_view name;  // logging
};

static std::vector<id_t> texturesFromMaterialEntry(
    MaterialEntry const& materialEntry) {
  std::vector<id_t> res;
  res.reserve(16);
  if (0 != materialEntry.albedoTex) res.push_back(materialEntry.albedoTex);
  if (0 != materialEntry.metalRoughnessTex)
    res.push_back(materialEntry.metalRoughnessTex);
  if (0 != materialEntry.normalTex) res.push_back(materialEntry.normalTex);
  if (0 != materialEntry.emissiveTex) res.push_back(materialEntry.emissiveTex);
  return res;
}

// --- Primitive ---

/// \note assuming types for position (TODO)
using PositionBuffer = std::vector<glm::vec3>;
/// \note assuming types for index (TODO)
using IndexBuffer = std::vector<uint32_t>;

struct NormalTangentUV {
  glm::vec3 normal;
  glm::vec3 tangent;
  glm::vec2 uv;
};
using NormalTangentUVBuffer = std::vector<NormalTangentUV>;

/// \note If multiple primitives use the same accessors, data gets
/// duplicated
struct Primitive {
  Primitive(size_t vertCount, size_t indexCount, bool size);

  PositionBuffer positions;
  IndexBuffer indices;
  NormalTangentUVBuffer attrs;
  id_t material;
  SphericalBB bounds;
};

Primitive::Primitive(size_t vertCount, size_t indexCount, bool size)
    : material(0), bounds{} {
  if (size) {
    positions.resize(vertCount);
    indices.resize(indexCount);
    attrs.resize(vertCount);
  } else {
    positions.reserve(vertCount);
    indices.reserve(indexCount);
    attrs.reserve(vertCount);
  }
}

// --- End Primitive ---

struct MeshEntry : public Entry {
  std::vector<Primitive> primitives;
};

static std::vector<id_t> materialsFromMeshEntry(MeshEntry const& meshEntry) {
  std::vector<id_t> res;
  res.reserve(meshEntry.primitives.size());
  for (auto const& prim : meshEntry.primitives) {
    res.push_back(prim.material);
  }
  return res;
}

template <typename EntryType>
static bool removeIfNotUsed(id_t key, std::unordered_map<id_t, EntryType>& map,
                            std::shared_mutex& mutex) {
  {
    std::shared_lock rLock{mutex};
    auto it = map.find(key);
    if (it == map.end() ||
        it->second.refCount.load(std::memory_order_acquire) > 0) {
      return false;
    }
  }
  {
    std::unique_lock wLock{mutex};
    auto it = map.find(key);
    if (it == map.end() ||
        it->second.refCount.load(std::memory_order_acquire) > 0) {
      return false;
    }
    map.erase(it);
  }
  return true;
}

template <typename EntryType>
static bool lookupIncreaseRefCountIfFound(
    id_t key, std::unordered_map<id_t, EntryType>& map,
    std::shared_mutex& mtx) {
  std::shared_lock rLock{mtx};
  auto it = map.find(key);
  if (it == map.cend()) return false;
  it->second.refCount.fetch_add(1, std::memory_order_acq_rel);
  return true;
}

template <typename EntryType>
static bool lookupDecreaseRefCountIfFound(
    id_t key, std::unordered_map<id_t, EntryType>& map,
    std::shared_mutex& mtx) {
  std::shared_lock rLock{mtx};
  auto it = map.find(key);
  if (it == map.cend()) return false;
  it->second.refCount.fetch_sub(1, std::memory_order_acq_rel);
  assert(it->second.refCount.load(std::memory_order_relaxed) >= 0);
  return true;
}

// ------------------------- Material Registry ------------------------------

class MaterialRegistry {
 public:
  MaterialRegistry();

  bool materialLookupIncreaseRefCountIfFound(id_t matId);

  /// \note decreases reference counters cascading to texture entries
  bool materialDecreaseRefCountsCascadingIfPresent(id_t matId);

  /// \note r-value reference to express that, if any dynamically alloated
  /// resources are taken
  bool materialInsert(id_t matId, MaterialEntry&& materialEntry);

  bool materialRemoveIfNotUsedAndDecreaseTexRefCounts(id_t matId);

  void materialRemoveAllUnused();

  /// \warning doesn't check that the contained texture if of the format
  /// the caller expects
  /// TODO add consistency and pixel format check
  bool textureLookupIncreaseRefCountIfFound(id_t texId);

  bool textureDecreaseRefCountIfPresent(id_t texId);

  void textureRemoveAllUnused();

  // TODO Android AAsset
  // TODO specify target channels and pixel format
  bool textureLoad(std::filesystem::path const& gltfPath, id_t texId,
                   tinygltf::Image const& image);

 private:
  /// protects insertions, deletions, rehashes, not element modification
  std::shared_mutex m_matMapMtx;
  std::unordered_map<id_t, MaterialEntry> m_matMap;

  /// protects insertions, deletions, rehashes, not element modification
  std::shared_mutex m_texMapMtx;
  std::unordered_map<id_t, TextureEntry> m_texMap;
};

MaterialRegistry::MaterialRegistry() {
  m_matMap.reserve(128);
  m_texMap.reserve(128);
}

bool MaterialRegistry::materialLookupIncreaseRefCountIfFound(id_t matId) {
  return lookupIncreaseRefCountIfFound(matId, m_matMap, m_matMapMtx);
}

bool MaterialRegistry::materialDecreaseRefCountsCascadingIfPresent(id_t matId) {
  std::shared_lock rLock{m_matMapMtx};
  auto it = m_matMap.find(matId);
  if (it == m_matMap.end()) {
    return false;
  }
  it->second.refCount.fetch_sub(1, std::memory_order_acq_rel);
  assert(it->second.refCount.load(std::memory_order_relaxed) >= 0);
  std::vector<id_t> usedTextures = texturesFromMaterialEntry(it->second);
  for (id_t tex : usedTextures) {
    textureDecreaseRefCountIfPresent(tex);
  }
  return true;
}

bool MaterialRegistry::materialInsert(id_t matId,
                                      MaterialEntry&& materialEntry) {
  std::unique_lock wLock{m_matMapMtx};
  auto const& [it, wasInserted] =
      m_matMap.try_emplace(matId, std::move(materialEntry));
  return wasInserted;
}

bool MaterialRegistry::materialRemoveIfNotUsedAndDecreaseTexRefCounts(
    id_t matId) {
  {
    std::shared_lock rLock{m_matMapMtx};
    auto const it = m_matMap.find(matId);
    if (it == m_matMap.cend() ||
        it->second.refCount.load(std::memory_order_acquire) > 0)
      return false;
  }
  {
    std::lock_guard wLock{m_matMapMtx};
    auto const it = m_matMap.find(matId);
    if (it == m_matMap.cend() ||
        it->second.refCount.load(std::memory_order_acquire) > 0)
      return false;
    std::vector<id_t> const textures = texturesFromMaterialEntry(it->second);
    for (id_t tex : textures) {
      textureDecreaseRefCountIfPresent(tex);
    }
    m_matMap.erase(it);
    return true;
  }
}

void MaterialRegistry::materialRemoveAllUnused() {
  std::lock_guard wLock{m_matMapMtx};
  for (auto it = m_matMap.begin(); it != m_matMap.end(); /*inside*/) {
    if (it->second.refCount.load(std::memory_order_acquire) > 0) {
      ++it;
    } else {
      std::vector<id_t> const textures = texturesFromMaterialEntry(it->second);
      for (id_t tex : textures) {
        textureDecreaseRefCountIfPresent(tex);
      }
      it = m_matMap.erase(it);
    }
  }
}

bool MaterialRegistry::textureLookupIncreaseRefCountIfFound(id_t texId) {
  return lookupIncreaseRefCountIfFound(texId, m_texMap, m_texMapMtx);
}

bool MaterialRegistry::textureDecreaseRefCountIfPresent(id_t texId) {
  return lookupDecreaseRefCountIfFound(texId, m_texMap, m_texMapMtx);
}

void MaterialRegistry::textureRemoveAllUnused() {
  std::lock_guard wLock{m_texMapMtx};
  for (auto it = m_texMap.begin(); it != m_texMap.end(); /*inside*/) {
    if (it->second.refCount.load(std::memory_order_acquire) > 0) {
      ++it;
    } else {
      it = m_texMap.erase(it);
    }
  }
}

// TODO support more file formats and pixel formats
// TODO add "imageinfo" library which reads headers without decoding content
// TODO insted of stb, use libpng for png files
// \warning assumes initial reference count to 1
bool MaterialRegistry::textureLoad(std::filesystem::path const& gltfPath,
                                   id_t texId, tinygltf::Image const& image) {
  // - Check image file path validity
  std::filesystem::path uriComponent{tinygltf::dlib::urldecode(image.uri)};
  if (uriComponent.is_absolute()) {
    LOGE << AVK_LOG_RED
        "[GLTF Loader] glTF file shouldn't contain absolute path (Found '"
         << uriComponent.string() << "'" AVK_LOG_RST << std::endl;
    return false;
  }
  std::error_code err;
  std::filesystem::path imagePath = std::filesystem::weakly_canonical(
      gltfPath.parent_path() / uriComponent, err);
  if (err) {
    LOGE << AVK_LOG_RED "[GLTF Loader] error computing path for image: "
         << err.message() << AVK_LOG_RST << std::endl;
    return false;
  }
  // - Check image extension and headers
  imageinfo::ImageInfo const info =
      imageinfo::parse<imageinfo::FilePathReader>(imagePath.string());
  // TODO support more TODO handle mipmaps
  if (!info.ok()) {
    LOGE << AVK_LOG_RED "[GLTF Loader] Error fetching image '" << imagePath
         << "' metadata: " << info.error_msg() << AVK_LOG_RST << std::endl;
    return false;
  }
  if (info.format() != imageinfo::kFormatPng &&
      info.format() != imageinfo::kFormatJpeg) {
    LOGE << AVK_LOG_RED
        "[GLTF Loader] error: unsupported image file format (PNG or JPEG) for "
        "image '"
         << imagePath << "'" AVK_LOG_RST << std::endl;
    return false;
  }
  // now we know how much memory we need
  TextureEntry textureEntry{};
  textureEntry.refCount.store(1, std::memory_order_relaxed);
  textureEntry.width = static_cast<uint32_t>(info.size().width);
  textureEntry.height = static_cast<uint32_t>(info.size().height);
  {
    int width, height, channels;
    stbi_uc* imageData =
        stbi_load(imagePath.string().c_str(), &width, &height, &channels, 0);
    if (!imageData) {
      LOGE << AVK_LOG_RED "[GLTF Loader] failed to stbi_load image '"
           << imagePath << "'" AVK_LOG_RST << std::endl;
      return false;
    }
    textureEntry.channels = static_cast<uint32_t>(channels);
    textureEntry.data.resize(textureEntry.bytes());
    memcpy(textureEntry.data.data(), imageData, textureEntry.bytes());
    stbi_image_free(imageData);
  }

  {
    std::lock_guard wLock{m_texMapMtx};
    auto const& [it, wasInserted] =
        m_texMap.try_emplace(texId, std::move(textureEntry));

    return wasInserted;
  }
}

// ------------------------- Impl Class -------------------------------------

class MeshSystemImpl {
 public:
  MeshSystemImpl();
  std::unordered_map<std::string, id_t> loadMeshesFromScene(
      const std::filesystem::path& path, const SceneImportOptions& opts,
      MaterialRegistry& matReg);

  bool acquireMesh(id_t meshId);
  bool releaseMesh(id_t meshId);
  bool unloadMeshIfUnusedCascading(id_t meshId, MaterialRegistry& matReg);
  void gcUnusedResources(MaterialRegistry& matReg);

 private:
  std::unordered_map<std::string, id_t> traverseScene(
      tinygltf::Model const& model, const std::filesystem::path& gltfPath,
      SceneImportOptions const& opts, MaterialRegistry& matReg);

  bool insertMesh(id_t meshId, MeshEntry&& meshEntry);

 private:
  tinygltf::TinyGLTF m_loader;

  /// protects insertions, deletions, rehashes, not element modification
  std::shared_mutex m_meshMapMtx;
  std::unordered_map<id_t, MeshEntry> m_meshMap;
};

static std::vector<std::string> anyUnsupportedExtensions(
    tinygltf::Model const& model) {
  return model.extensionsRequired;
}

static std::vector<std::string> anyUnsupportedAccessors(
    tinygltf::Model const& model) {
  using namespace std::string_literals;

  std::vector<std::string> sparseAccessors;
  sparseAccessors.reserve(64);
  size_t index = 0;
  for (auto const& accessor : model.accessors) {
    size_t const current = index++;
    if (accessor.sparse.isSparse) {
      sparseAccessors.push_back("Accessor Index "s + std::to_string(current) +
                                " is sparse");
    }
  }
  return sparseAccessors;
}

static std::vector<std::string> anyPrimitivesMissingOrInvalidFields(
    tinygltf::Mesh const& mesh) {
  std::vector<std::string> msgs;
  msgs.reserve(mesh.primitives.size());

  size_t index = 0;
  for (tinygltf::Primitive const& prim : mesh.primitives) {
    size_t const current = index++;
    // no morph targets
    if (prim.targets.size() > 0) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " has 1+ morph targets");
    }
    // need position and indices
    if (prim.attributes.find("POSITION") == prim.attributes.cend()) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " doesn't have a POSITION attribute");
    }
    if (prim.indices < 0) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " doesn't have an index accessor");
    }
    // TODO support only triangle list mode
    if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " doesn't have 'mode' equal to 4 (triangle list)");
    }
    if (prim.material < 0) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " doesn't have a valid material index");
    }
  }

  return msgs;
}

/// \note we are not checking for attributes which we use but are not there,
/// because it's up to the other functions to check/generate them (TODO)
/// \warning assumes that position is among the attributes
static std::vector<std::string> anyPrimitiveWithInvalidAccessors(
    tinygltf::Mesh const& mesh,
    std::vector<tinygltf::Accessor> const& accessors) {
  std::vector<std::string> msgs;
  msgs.reserve(64);
  size_t index = 0;
  for (tinygltf::Primitive const& prim : mesh.primitives) {
    size_t const current = index++;
    // position accessor should be vec3<float> assuming it's there because
    // we already checked its presence
    if (size_t const idx = prim.attributes.at("POSITION");
        idx >= accessors.size() ||
        (accessors[idx].type != TINYGLTF_TYPE_VEC3 ||
         accessors[idx].componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)) {
      msgs.emplace_back(
          "Primitive index " + std::to_string(current) +
          " has incorrect type/componentType for POSITION attribute accessor");
    }
    // index accessor should be scalar<uint32>
    if (size_t const idx = prim.indices;
        idx >= accessors.size() ||
        (accessors[idx].type != TINYGLTF_TYPE_SCALAR ||
         accessors[idx].componentType !=
             TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)) {
      msgs.emplace_back(
          "Primitive index " + std::to_string(current) +
          " has incorrect type/componentType for indices accessor");
    }
    // (opt) normal should be vec3<float>
    if (auto const it = prim.attributes.find("NORMAL");
        it != prim.attributes.cend()) {
      if (size_t const idx = it->second;
          idx >= accessors.size() ||
          (accessors[idx].type != TINYGLTF_TYPE_VEC3 ||
           accessors[idx].componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)) {
        msgs.emplace_back(
            "Primitive index " + std::to_string(current) +
            " has incorrect type/componentType for NORMAL attribute accessor");
      }
    }
    // (opt) tangent should be vec3<float>
    if (auto const it = prim.attributes.find("TANGENT");
        it != prim.attributes.cend()) {
      if (size_t const idx = it->second;
          idx >= accessors.size() ||
          (accessors[idx].type != TINYGLTF_TYPE_VEC3 ||
           accessors[idx].componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)) {
        msgs.emplace_back(
            "Primitive index " + std::to_string(current) +
            " has incorrect type/componentType for TANGENT attribute accessor");
      }
    }
    // (opt) uv should be vec2<float>
    if (auto const it = prim.attributes.find("TEXCOORD_0");
        it != prim.attributes.cend()) {
      if (size_t const idx = it->second;
          idx >= accessors.size() ||
          (accessors[idx].type != TINYGLTF_TYPE_VEC2 ||
           accessors[idx].componentType != TINYGLTF_COMPONENT_TYPE_FLOAT)) {
        msgs.emplace_back("Primitive index " + std::to_string(current) +
                          " has incorrect type/componentType for TEXCOORD_0 "
                          "attribute accessor");
      }
    }
  }
  return msgs;
}

static std::vector<std::string> anyPrimitiveMissingAttributes(
    tinygltf::Mesh const& mesh) {
  std::vector<std::string> msgs;
  msgs.reserve(64);
  size_t index = 0;
  for (tinygltf::Primitive const& prim : mesh.primitives) {
    size_t const current = index++;
    if (auto const it = prim.attributes.find("NORMAL");
        it == prim.attributes.cend()) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " has no NORMAL attribute (TODO generate it)");
    }
    if (auto const it = prim.attributes.find("TANGENT");
        it == prim.attributes.cend()) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " has no TANGENT attribute (TODO generate it)");
    }
    if (auto const it = prim.attributes.find("TEXCOORD_0");
        it == prim.attributes.cend()) {
      msgs.emplace_back("Primitive index " + std::to_string(current) +
                        " has no TEXCOORD_0 attribute (TODO generate it)");
    }
  }
  return msgs;
}

static void emitWarningIfMultipleTexCoord(tinygltf::Primitive const& prim) {
  size_t const count =
      std::count_if(prim.attributes.cbegin(), prim.attributes.cend(),
                    [](std::pair<std::string, int> const& pair) {
                      return pair.first.find("TEXCOORD_") != std::string::npos;
                    });
  if (count > 1) {
    LOGW << AVK_LOG_YLW
        "[GLTF Loader] Warning: found primitive with multiple TEXCOORD_* "
        "attributes. we currently support only TEXCOORD_0" AVK_LOG_RST
         << std::endl;
  }
}

// TODO better in android with AAsset
static std::vector<std::string> anyUriInvalidOrEmbeddedImages(
    std::filesystem::path const& gltfPath, tinygltf::Model const& model) {
  using namespace std::string_literals;

  std::vector<std::string> invalidImages;
  invalidImages.reserve(64);

  size_t index = 0;
  for (auto const& image : model.images) {
    size_t const current = index++;
    if (image.bufferView != -1) {
      invalidImages.push_back("Image Index "s + std::to_string(current) +
                              " is embedded");
    } else {
      auto const imagePath = std::filesystem::weakly_canonical(
          gltfPath.parent_path() /
          std::filesystem::path(tinygltf::dlib::urldecode(image.uri)));
      if (!std::filesystem::is_regular_file(imagePath)) {
        invalidImages.push_back("Image Index "s + std::to_string(current) +
                                " with uri '" + imagePath.string() +
                                "' is not a valid file path");
      } else if (!hasExtension(imagePath, SupportedImageFormats)) {
        invalidImages.push_back("Image Index "s + std::to_string(current) +
                                " has unsupported extension '" +
                                imagePath.extension().string() + '\'');
      }
    }
  }

  return invalidImages;
}

static bool anySkins(tinygltf::Model const& model) {
  return !model.skins.empty();
}

static bool anyAnimations(tinygltf::Model const& model) {
  return !model.animations.empty();
}

// ----------------------- Helpers ------------------------------------------
/// \warning:Works only if vector element type matches in size the element in
/// buffer (when tightly packed, ie byteStride = 0)
template <typename T,
          typename V = std::enable_if_t<std::is_standard_layout_v<T>, void>>
static bool copyAttributeVector(std::vector<T>& out,
                                tinygltf::Model const& model,
                                tinygltf::Accessor const& accessor) {
  if (accessor.bufferView < 0) return false;

  tinygltf::BufferView const& bufferView =
      model.bufferViews[accessor.bufferView];
  tinygltf::Buffer const& buffer = model.buffers[bufferView.buffer];

  size_t const count = accessor.count;
  out.resize(count);

  // TODO stride may be 0 meaning tightly packed; fallback to sizeof(T) for now
  size_t const stride =
      (bufferView.byteStride == 0) ? sizeof(T) : bufferView.byteStride;
  size_t const baseOffset = bufferView.byteOffset + accessor.byteOffset;

  // bounds check
  if (baseOffset + (count - 1) * stride + sizeof(T) > buffer.data.size())
    return false;

  // copy: not using memcpy as this is more debuggable
  for (size_t i = 0; i < count; ++i) {
    size_t const srcOffset = baseOffset + i * stride;
    out[i] = *reinterpret_cast<T const*>(&buffer.data[srcOffset]);
  }
  return true;
}

static inline bool copyIndices(std::vector<uint32_t>& out,
                               tinygltf::Model const& model,
                               tinygltf::Accessor const& accessor) {
  if (accessor.bufferView < 0) return false;
  assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
  return copyAttributeVector(out, model, accessor);
}

static inline void printWarningList(std::vector<std::string> const& msgs,
                                    std::string_view meshName) {
  LOGW << AVK_LOG_YLW "[GLTF Loader] mesh name " << meshName << ':'
       << std::flush;
  for (auto const& m : msgs) LOGW << "\n\t" << m;
  LOGW << AVK_LOG_RST << std::endl;
}

// texture loader helper that centralizes lookup/loading/logging
// TODO handle Android AAsset
// TODO handle normal texture scale
// 0 -> nothing
inline static id_t loadTextureMaybe(MaterialRegistry& matReg,
                                    std::filesystem::path const& gltfPath,
                                    id_t texId, tinygltf::Image const& image,
                                    std::string_view meshName) {
  if (matReg.textureLookupIncreaseRefCountIfFound(texId)) return texId;
  if (matReg.textureLoad(gltfPath, texId, image)) return texId;
  LOGE << AVK_LOG_RED << "[GLTF Loader] Couldn't load image " << image.uri
       << " for mesh " << meshName << ", skipping it" AVK_LOG_RST << std::endl;
  return 0;
}

/// \warning populates only non-texture fields and refCount to 1
inline static MaterialEntry makeMaterialEntryFromTinyGLTF(
    tinygltf::Material const& material,
    tinygltf::PbrMetallicRoughness const& pbr) {
  // note: we are relying on tinygltf default values for properties
  // which are absent on the glTF file. Example: emissiveFactor is
  // additive, hence tinygltf default it at 0,0,0
  // while, baseColorFactor, which is multiplicative factors, are
  // defaulted to 1 by tinygltf (metallicFactor, roughnessFactor 1 too)
  MaterialEntry me{};
  me.refCount.store(1, std::memory_order_relaxed);
  me.albedoFactor = floatGlm4VecFromDoubleArray(pbr.baseColorFactor);
  me.emissiveFactor = floatGlm3VecFromDoubleArray(material.emissiveFactor);
  me.metallicFactor = pbr.metallicFactor;
  me.roughnessFactor = pbr.roughnessFactor;
  return me;
}

/// RAII rollback for material/texture refCounts on failure
struct MaterialRefJanitor : public NonMoveable {
  MaterialRegistry& matReg;
  std::vector<id_t> materialIds;  // materials to decrement (cascading textures)
  std::vector<id_t> textureIds;   // textures to decrement
  bool active = true;

  MaterialRefJanitor(MaterialRegistry& r, std::vector<id_t> mats,
                     std::vector<id_t> texs)
      : matReg(r), materialIds(std::move(mats)), textureIds(std::move(texs)) {}

  inline void dismiss() { active = false; }

  ~MaterialRefJanitor() noexcept {
    if (!active) return;
    for (id_t mat : materialIds)
      matReg.materialDecreaseRefCountsCascadingIfPresent(mat);
    for (id_t tex : textureIds)  //
      matReg.textureDecreaseRefCountIfPresent(tex);
  }
};

// ----------------------- Mesh System Impl ---------------------------------

MeshSystemImpl::MeshSystemImpl() { m_meshMap.reserve(128); }

bool MeshSystemImpl::insertMesh(id_t meshId, MeshEntry&& meshEntry) {
  std::unique_lock wLock{m_meshMapMtx};
  auto const& [it, wasInserted] =
      m_meshMap.try_emplace(meshId, std::move(meshEntry));
  return wasInserted;
}

// TODO Android AAsset
std::unordered_map<std::string, id_t> MeshSystemImpl::traverseScene(
    const tinygltf::Model& model, const std::filesystem::path& gltfPath,
    const SceneImportOptions& opts, MaterialRegistry& matReg) {
  using namespace std::string_view_literals;

  std::unordered_map<std::string, id_t> insertedMeshes;
  insertedMeshes.reserve(64);

  // we support glTF files with only one scene
  if (model.scenes.size() != 1) {
    LOGE << AVK_LOG_RED "[GLTF Loader] Expected glTF File with 1 scene, got "
         << model.scenes.size() << AVK_LOG_RST << std::endl;
    return insertedMeshes;
  }

  LOGI << "[GLTF Loader] Processing Scene '" << model.scenes[0].name << '\''
       << std::endl;

  if (opts.applySceneTransform) {
    // traverse each scene, and keep a stack of transforms, such that when a
    // mesh is encountered, we can apply the transformation to each of its
    // primitives and process the transformed mesh (add new materials, ...)
    // - for each root node in the scene
    for (int rootNode [[maybe_unused]] : model.scenes[0].nodes) {
      // -------------------------------------------------------------------
      LOGE << AVK_LOG_RED
          "[GLTF Loader] Scene Transform Application not implemented "
          "yet!" AVK_LOG_RST
           << std::endl;
      return insertedMeshes;
    }
  } else {
    // local variables as shortcuts
    auto const& accessors = model.accessors;
    auto const& textures = model.textures;
    auto const& images = model.images;

    // traverse the mesh list and process the original mesh
    for (auto const& mesh : model.meshes) {
      id_t const meshId = fnv1aHash(gltfPath.string() + mesh.name);
      MeshEntry entry{};

      if (mesh.primitives.size() == 0) {
        LOGW << AVK_LOG_YLW "[GLTF Loader] mesh " << mesh.name
             << " doesn't have any primitives, skipping it..." AVK_LOG_RST
             << std::endl;
        continue;
      }

      if (auto msgs = anyPrimitivesMissingOrInvalidFields(mesh);
          !msgs.empty()) {
        printWarningList(msgs, mesh.name);
        continue;
      }
      if (auto msgs = anyPrimitiveWithInvalidAccessors(mesh, model.accessors);
          !msgs.empty()) {
        printWarningList(msgs, mesh.name);
        continue;
      }
      // TODO attribute generation
      if (auto msgs = anyPrimitiveMissingAttributes(mesh); !msgs.empty()) {
        printWarningList(msgs, mesh.name);
        continue;
      }
      // now fill in mesh data
      entry.primitives.reserve(mesh.primitives.size());
      for (tinygltf::Primitive const& prim : mesh.primitives) {
        // TODO
        emitWarningIfMultipleTexCoord(prim);
        // take vertex data accessors
        // (throws if not present, as previous checks should ensure presence)
        int const positionAccessorIndex = prim.attributes.at("POSITION");
        int const normalAccessorIndex = prim.attributes.at("NORMAL");
        int const tangentAccessorIndex = prim.attributes.at("TANGENT");
        int const uvAccessorIndex = prim.attributes.at("TEXCOORD_0");
        int const indicesAccessorIndex = prim.indices;

        auto const& positionAccessor = accessors[positionAccessorIndex];
        auto const& normalAccessor = accessors[normalAccessorIndex];
        auto const& tangentAccessor = accessors[tangentAccessorIndex];
        auto const& uvAccessor = accessors[uvAccessorIndex];
        auto const& indicesAccessor = accessors[indicesAccessorIndex];

        // clang-format off
        assert(positionAccessor.type == TINYGLTF_TYPE_VEC3 && positionAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        assert(normalAccessor.type == TINYGLTF_TYPE_VEC3 && normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        assert(tangentAccessor.type == TINYGLTF_TYPE_VEC3 && tangentAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        assert(uvAccessor.type == TINYGLTF_TYPE_VEC2 && uvAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        assert(indicesAccessor.type == TINYGLTF_TYPE_SCALAR && indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
        assert(positionAccessor.count == normalAccessor.count && normalAccessor.count == tangentAccessor.count && tangentAccessor.count == uvAccessor.count);
        // clang-format on

        // copy index data
        size_t const dstPrimIdx = entry.primitives.size();
        entry.primitives.emplace_back(positionAccessor.count,
                                      indicesAccessor.count, true);
        auto& dstPrim = entry.primitives.back();
        // - positions
        if (!copyAttributeVector(dstPrim.positions, model, positionAccessor)) {
          LOGE << AVK_LOG_RED "[GLTF Loader] Failed to copy positions for mesh "
               << mesh.name << ", primitive " << dstPrimIdx
               << ", skipping primitive" AVK_LOG_RST << std::endl;
          continue;
        }
        // - index (note: no accessor byte offset)
        if (!copyIndices(dstPrim.indices, model, indicesAccessor)) {
          LOGE << AVK_LOG_RED "[GLTF Loader] Failed to copy indices for mesh "
               << mesh.name << ", primitive " << dstPrimIdx
               << ", skipping primitive" AVK_LOG_RST << std::endl;
          continue;
        }
        // - other attributes: normal, tangent, uv
        // for simplicity, first use temporary vectors, then copy into dest
        std::vector<glm::vec3> normals, tangents;
        std::vector<glm::vec2> uvs;
        normals.reserve(normalAccessor.count);
        tangents.reserve(tangentAccessor.count);
        uvs.reserve(uvAccessor.count);
        if (!copyAttributeVector(normals, model, normalAccessor)) {
          LOGE << AVK_LOG_RED "[GLTF Loader] Failed to copy normals for mesh "
               << mesh.name << ", primitive " << dstPrimIdx
               << ", skipping primitive" AVK_LOG_RST << std::endl;
          continue;
        }
        if (!copyAttributeVector(tangents, model, tangentAccessor)) {
          LOGE << AVK_LOG_RED "[GLTF Loader] Failed to copy tangents for mesh "
               << mesh.name << ", primitive " << dstPrimIdx
               << ", skipping primitive" AVK_LOG_RST << std::endl;
          continue;
        }
        if (!copyAttributeVector(uvs, model, uvAccessor)) {
          LOGE << AVK_LOG_RED "[GLTF Loader] Failed to copy uvs for mesh "
               << mesh.name << ", primitive " << dstPrimIdx
               << ", skipping primitive" AVK_LOG_RST << std::endl;
          continue;
        }
        assert(normals.size() == tangents.size() &&
               tangents.size() == uvs.size() &&
               uvs.size() == dstPrim.attrs.size());
        for (size_t i = 0; i < dstPrim.attrs.size(); ++i) {
          dstPrim.attrs[i].uv = uvs[i];
          dstPrim.attrs[i].normal = normals[i];
          dstPrim.attrs[i].tangent = tangents[i];
          // normalization (maybe not necessary)
          auto& n = dstPrim.attrs[i].normal;
          auto& t = dstPrim.attrs[i].tangent;
          if (glm::epsilonNotEqual(glm::length2(n), 1.f, 1e-5f))
            n = glm::normalize(n);
          if (glm::epsilonNotEqual(glm::length2(n), 1.f, 1e-5f))
            t = glm::normalize(t);
        }
        // - create material for the primitive
        tinygltf::Material const& material = model.materials[prim.material];
        id_t const materialId = fnv1aHash(gltfPath.string() + material.name);

        // -- if the material is already present, then skip
        if (!matReg.materialLookupIncreaseRefCountIfFound(materialId)) {
          auto const& pbr = material.pbrMetallicRoughness;
          // note: mesh starts with ref count 1 until component. material starts
          // at 1 because at least this mesh uses it
          auto materialEntry = makeMaterialEntryFromTinyGLTF(material, pbr);
          // load textures with rollback of reference counts on failure
          std::vector<id_t> savedTextureIds;
          savedTextureIds.reserve(8);
          // clang-format off
          std::array<TexSpec, 4> specs{{
              {pbr.baseColorTexture.index,         pbr.baseColorTexture.texCoord,         0.f,                                        &MaterialEntry::albedoTex,         "baseColor"},
              {pbr.metallicRoughnessTexture.index, pbr.metallicRoughnessTexture.texCoord, 0.f,                                        &MaterialEntry::metalRoughnessTex, "metallicRoughness"},
              {material.normalTexture.index,       material.normalTexture.texCoord, static_cast<float>(material.normalTexture.scale), &MaterialEntry::normalTex,         "normal"},
              {material.emissiveTexture.index,     material.emissiveTexture.texCoord,     0.f,                                        &MaterialEntry::emissiveTex,       "emissive"}
          }};
          // clang-format on

          // - albedo (RGBA|RGB|BC|ETC2)
          // - metalRough (RGBA|RGB|BC|ETC2), RG channels matter
          // - normal (RGB|BC|ETC2)
          // - emissive (RGB|BC|ETC2)
          for (auto const& s : specs) {
            if (s.imageIndex < 0) continue;
            tinygltf::Texture const& tex = textures[s.imageIndex];
            tinygltf::Image const& image = images[tex.source];

            id_t const texId = fnv1aHash(
                gltfPath.string() + std::to_string(s.imageIndex) + image.uri);
            if (id_t loaded =
                    loadTextureMaybe(matReg, gltfPath, texId, image, mesh.name);
                loaded == texId) {
              materialEntry.*(s.field) = loaded;
              savedTextureIds.push_back(loaded);
            }
          }

          // finally, insert the material in the registry
          if (!matReg.materialInsert(materialId, std::move(materialEntry))) {
            LOGE << AVK_LOG_RED "[GLTF Loader] Couldn't insert material "
                 << material.name << " inside the registry for mesh "
                 << mesh.name
                 << ", skipping it (might result in visual "
                    "artifacts) (Textures are NOT deallocated)" AVK_LOG_RST
                 << std::endl;
            // tick down ref counters for all textures
            for (id_t tex : savedTextureIds) {
              matReg.textureDecreaseRefCountIfPresent(tex);
            }
          }

          dstPrim.material = materialId;

          // Compute Spherical Bounds
          {
            auto const& acc = model.accessors[prim.attributes.at("POSITION")];
            glm::vec3 const min = floatGlm3VecFromDoubleArray(acc.minValues);
            glm::vec3 const max = floatGlm3VecFromDoubleArray(acc.maxValues);
            dstPrim.bounds.Center = (min + max) * 0.5f;
            dstPrim.bounds.Radius = glm::length(max - dstPrim.bounds.Center);
          }
        }  // end if (!matReg.materialLookupIncreaseRefCountIfFound(materialId))
      }  // for (tinygltf::Primitive const& prim : mesh.primitives)

      // prepare material ids so that, if insertion fails, we can decrease
      // material ref counts and texture ref counts
      std::vector<id_t> materialsVec = materialsFromMeshEntry(entry);

      // insert the current mesh entry
      if (!insertMesh(meshId, std::move(entry))) {
        LOGE << AVK_LOG_RED "[GLTF Loader] Couldn't insert mesh " << mesh.name
             << " inside the Mesh System. Skipping it (textures and materials "
                "are NOT Deallocated)" AVK_LOG_RST
             << std::endl;
        for (id_t mat : materialsVec) {
          matReg.materialDecreaseRefCountsCascadingIfPresent(mat);
        }
      } else {
        LOGI << "[GLTF Loader] mesh " << mesh.name << " inserted in Mesh System"
             << std::endl;
        insertedMeshes.try_emplace(mesh.name, meshId);
      }
    }  // end for (auto const& mesh : model.meshes)
  }

  return insertedMeshes;
}

std::unordered_map<std::string, id_t> MeshSystemImpl::loadMeshesFromScene(
    const std::filesystem::path& path, const SceneImportOptions& opts,
    MaterialRegistry& matReg) {
  if (std::filesystem::exists(path) ||
      !std::filesystem::is_regular_file(path)) {
    LOGE << AVK_LOG_RED "[GLTF Loader] File Path '" << path.string()
         << "' doesn't exist" AVK_LOG_RST << std::endl;
    return {};
  }

  tinygltf::Model model;
  {
    std::string err, warn;
    bool res = m_loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
    if (!warn.empty()) {
      LOGW << AVK_LOG_YLW "Warn: " << warn << AVK_LOG_RST << std::endl;
    }
    if (!err.empty()) {
      LOGE << AVK_LOG_RED "[GLTF Loader] Error: " << err << AVK_LOG_RST
           << std::endl;
    }
    if (!res) {
      LOGE << AVK_LOG_RED
          "[GLTF Loader] Failed to parse GLTF, Returning" AVK_LOG_RST
           << std::endl;
      return {};
    }
  }

  if (auto extensions = anyUnsupportedExtensions(model); !extensions.empty()) {
    LOGE << AVK_LOG_RED "[GLTF Loader] Unsupported GLTF Extensions:";
    for (auto const& ext : extensions) LOGE << "\n\t- " << ext;
    LOGE << AVK_LOG_RST << std::flush;
    return {};
  }

  if (auto msgs = anyUnsupportedAccessors(model); !msgs.empty()) {
    LOGE << AVK_LOG_RED "[GLTF Loader] Unsupported GLTF \"accessors\"";
    for (auto const& msg : msgs) LOGE << "\n\t" << msg;
    LOGE << AVK_LOG_RST << std::flush;
    return {};
  }

  if (auto msgs = anyUriInvalidOrEmbeddedImages(path, model); !msgs.empty()) {
    LOGE << AVK_LOG_RED "[GLTF Loader] Found unsupported \"images\"";
    for (auto const& msg : msgs) LOGE << "\n\t" << msg;
    LOGE << AVK_LOG_RST << std::endl;
    return {};
  }

  if (anySkins(model)) {
    LOGE << AVK_LOG_RED
        "[GLTF Loader] Found \"skins\", which is not supported" AVK_LOG_RST
         << std::endl;
    return {};
  }

  if (anyAnimations(model)) {
    LOGE << AVK_LOG_RED
        "[GLTF Loader] Found \"animations\", which is not "
        "supported" AVK_LOG_RST
         << std::endl;
  }

  return traverseScene(model, path, opts, matReg);
}

// --------------- Mesh System API Implementation ---------------------------

bool MeshSystemImpl::acquireMesh(id_t meshId) {
  return lookupIncreaseRefCountIfFound(meshId, m_meshMap, m_meshMapMtx);
}

bool MeshSystemImpl::releaseMesh(id_t meshId) {
  std::shared_lock rLock{m_meshMapMtx};
  auto it = m_meshMap.find(meshId);
  if (it == m_meshMap.cend()) return false;

  int const old = it->second.refCount.fetch_sub(1, std::memory_order_acq_rel);
  assert(old >= 1);
  // doesn't cascade to materials even if new count is zero, because mesh
  // is still there
  return true;
}

bool MeshSystemImpl::unloadMeshIfUnusedCascading(id_t meshId,
                                                 MaterialRegistry& matReg) {
  {
    std::shared_lock rLock{m_meshMapMtx};
    auto it = m_meshMap.find(meshId);
    if (it == m_meshMap.end() ||
        it->second.refCount.load(std::memory_order_acquire) > 0)
      return false;
  }
  {
    std::unique_lock wLock{m_meshMapMtx};
    auto it = m_meshMap.find(meshId);
    if (it == m_meshMap.end() ||
        it->second.refCount.load(std::memory_order_acquire) > 0)
      return false;

    // note: not removing materials. call GC method for that
    for (auto const& prim : it->second.primitives) {
      matReg.materialDecreaseRefCountsCascadingIfPresent(prim.material);
    }
    m_meshMap.erase(it);
    return true;
  }
}

// mesh first, then materials, then textures, so that reference counters can
// be inspected at their lowest possible value
void MeshSystemImpl::gcUnusedResources(MaterialRegistry& matReg) {
  // cleanup meshes
  {
    std::lock_guard rLock{m_meshMapMtx};
    for (auto it = m_meshMap.begin(); it != m_meshMap.end(); /*inside*/) {
      if (it->second.refCount.load(std::memory_order_acquire) > 0) {
        ++it;
      } else {
        for (auto const& prim : it->second.primitives) {
          matReg.materialDecreaseRefCountsCascadingIfPresent(prim.material);
        }
        it = m_meshMap.erase(it);
      }
    }
  }
  // cleanup materials
  matReg.materialRemoveAllUnused();
  // cleanup textures
  matReg.textureRemoveAllUnused();
}

// ------------------------- Interface --------------------------------------

MeshSystem::MeshSystem()
    : m_impl(std::make_unique<MeshSystemImpl>()),
      m_matReg(std::make_unique<MaterialRegistry>()) {}

std::unordered_map<std::string, id_t> MeshSystem::loadScene(
    const std::filesystem::path& path, const SceneImportOptions& opts) {
  return m_impl->loadMeshesFromScene(path, opts, *m_matReg);
}

bool MeshSystem::acquireMesh(id_t meshId) {
  return m_impl->acquireMesh(meshId);
}

bool MeshSystem::releaseMesh(id_t meshId) {
  return m_impl->releaseMesh(meshId);
}

bool MeshSystem::unloadMeshIfUnusedCascading(id_t meshId) {
  return m_impl->unloadMeshIfUnusedCascading(meshId, *m_matReg);
}

void MeshSystem::gcUnusedResources() {
  return m_impl->gcUnusedResources(*m_matReg);
}

}  // namespace avk