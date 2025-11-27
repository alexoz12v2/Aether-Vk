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

// glm library
#include <glm/glm.hpp>

// lib
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

// Note: The glTF 2.0 specification doesn't specify any standardized name for
// primitive attributes. What we require are the names used by blender's GLTF
// exporter

// Note: Each primitive has only one material

// Note: uri is to be decoded (eg whitespace is %20)

// Note: we ignore the samplers (TODO)

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

// ------------------ Entry Types and bookkeping methods --------------------
/// id is implicitly stored in the map
struct Entry {
  std::atomic_int refCount;
};

/// \note supports only byte pixel formats (TODO)
struct TextureEntry : public Entry {
  uint32_t width;
  uint32_t height;
  uint32_t channels;
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
  float emissiveFactor;
};

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
  PositionBuffer positions;
  IndexBuffer indices;
  NormalTangentUVBuffer attrs;
  id_t material;
};

// --- End Primitive ---

struct MeshEntry : public Entry {
  std::vector<Primitive> primitives;
};

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

// ------------------------- Material Registry ------------------------------

class MaterialRegistry {
 public:
  MaterialRegistry();

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

// ------------------------- Impl Class -------------------------------------

class MeshSystemImpl {
 public:
  MeshSystemImpl();
  std::unordered_map<std::string, id_t> loadMeshesFromScene(
      const std::filesystem::path& path, const SceneImportOptions& opts,
      MaterialRegistry& matReg);

 private:
  std::unordered_map<std::string, id_t> traverseScene(
      tinygltf::Model const& model, SceneImportOptions const& opts,
      MaterialRegistry& matReg);

 private:
  tinygltf::TinyGLTF m_loader;

  /// protects insertions, deletions, rehashes, not element modification
  std::shared_mutex m_meshMapMtx;
  std::unordered_map<id_t, MeshEntry> m_meshMap;
};

static std::vector<std::string> anyUnsupportedExtensions(
    tinygltf::Model const& model) {
  return {};
}

static std::vector<std::string> anyUnsupportedAccessors(
    tinygltf::Model const& model) {
  return {};
}

static std::vector<std::string> anyUriInvalidOrEmbeddedImages(
    tinygltf::Model const& model) {
  return {};
}

static bool anySkins(tinygltf::Model const& model) { return {}; }

static bool anyAnimations(tinygltf::Model const& model) { return {}; }

MeshSystemImpl::MeshSystemImpl() { m_meshMap.reserve(128); }

std::unordered_map<std::string, id_t> MeshSystemImpl::traverseScene(
    const tinygltf::Model& model, const SceneImportOptions& opts,
    MaterialRegistry& matReg) {
  std::unordered_map<std::string, id_t> insertedMeshes;
  insertedMeshes.reserve(64);
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

  if (auto msgs = anyUriInvalidOrEmbeddedImages(model); !msgs.empty()) {
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

  return traverseScene(model, opts, matReg);
}

// ------------------------- Interface --------------------------------------

MeshSystem::MeshSystem()
    : m_impl(std::make_unique<MeshSystemImpl>()),
      m_matReg(std::make_unique<MaterialRegistry>()) {}

std::unordered_map<std::string, id_t> MeshSystem::loadScene(
    const std::filesystem::path& path, const SceneImportOptions& opts) {
  return m_impl->loadMeshesFromScene(path, opts, *m_matReg);
}

}  // namespace avk