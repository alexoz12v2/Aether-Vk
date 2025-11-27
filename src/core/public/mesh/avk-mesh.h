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

// TODO possible: Store a importance metric on the rendering
//   required: Bitmask for storing when frustum culling passed,screen space area
// TODO Component Container must track the reference count for each component
struct MeshComponent {
  /// Hash of the container object
  id_t Object;
  /// Hash of the referenced mesh
  /// \warning There can be multiple mesh components having same mesh id
  id_t Mesh;
  /// Bounding Box of this mesh in Model Space.
  /// - Once mesh loaded, this is constant and used to spawn `RenderComponent`
  ///   instance, whose bounds vary with the transform
  /// \warning a copy from the mash asset in the system class
  SphericalBB Bounds;
};
static_assert(std::is_standard_layout_v<MeshComponent>);

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

 private:
  std::unique_ptr<MeshSystemImpl> m_impl;
  std::unique_ptr<MaterialRegistry> m_matReg;
};

}  // namespace avk