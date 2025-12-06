#pragma once

#include <type_traits>

#include "utils/bits.h"
#include "utils/integer.h"
#include "utils/mixins.h"
#include "utils/types.h"

// lib
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace avk {

struct SceneImportOptions {
  bool applySceneTransform;
};

class MeshSystemImpl;
class MaterialRegistry;
class MeshSystem : public NonMoveable {
 public:
  MeshSystem();
  // TODO integrate with Android AAsset
  [[nodiscard]] std::unordered_map<std::string, id_t> loadScene(
      std::filesystem::path const& path, SceneImportOptions const& opts);

  /// increases mesh refCount if found, and returns true if that's the case
  bool acquireMesh(id_t meshId);

  /// decreases refCount of mesh
  /// \warning caller should release only when it actually acquired the mesh
  bool releaseMesh(id_t meshId);

  /// unloads mesh only if refCount == 0 and decrease materials ref count
  bool unloadMeshIfUnusedCascading(id_t meshId);

  /// removes unused mesh/materials/textures
  void gcUnusedResources();

 private:
  std::unique_ptr<MeshSystemImpl> m_impl;
  std::unique_ptr<MaterialRegistry> m_matReg;
};

}  // namespace avk