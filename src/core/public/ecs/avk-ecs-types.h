#pragma once

#include "utils/bits.h"
#include "utils/integer.h"
#include "utils/mixins.h"
#include "utils/types.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

// std lib
#include <type_traits>

namespace avk {

/// Necessary condition to have a pure data object. The only exception
/// to this constraints is `IScriptComponent`
template <typename T>
struct is_component {
  static bool constexpr value = std::is_standard_layout_v<T> &&
                                std::is_trivially_copyable_v<T> &&
                                std::is_trivially_destructible_v<T>;
};

template <typename T>
inline bool constexpr is_component_v = is_component<T>::value;

// ------------------------ Components ---------------------------------------

struct ComponentBase {
  /// identifier of the Object to which the component belongs to
  id_t Object;
  /// Identifier of
  id_t Id;
};

// TODO possible: Store a importance metric on the rendering
//   required: Bitmask for storing when frustum culling passed,screen space area
// TODO Component Container must track the reference count for each component
struct MeshComponent {
  ComponentBase base;
  /// Hash of the referenced mesh
  /// \warning There can be multiple mesh components having same mesh id
  id_t Mesh;
  /// Bounding Box of this mesh in Model Space.
  /// - Once mesh loaded, this is constant and used to spawn `RenderComponent`
  ///   instance, whose bounds vary with the transform
  /// \warning union of all spherical bb of primitives. so if any of these
  /// change, this is not valid anymore
  SphericalBB Bounds;
};
static_assert(is_component_v<MeshComponent>);

struct TransformComponent {
  ComponentBase Base;
  glm::vec3 Position;
  glm::quat Orientation;
  glm::vec3 Scale;
};
static_assert(is_component_v<TransformComponent>);

/// component to store up to arbitrary 104 Bytes
struct DataComponent104 {
  ComponentBase Base;
  id_t DataId;
  std::aligned_storage_t<104, 8> Storage;
};
static_assert(is_component_v<DataComponent104>);

struct CameraComponent {
  ComponentBase Base;
  union U {
    struct OrthoT {
      float Width;
      float Height;
      float Near;
      float Far;
    } Ortho;
    struct PerspT {
      float Fov;
      float AspectRatio;
      float Near;
      float Far;
    } Persp;
  } Params;
  bool IsPersp;
};
static_assert(is_component_v<DataComponent104>);

class IScriptComponent : NonMoveable {
  virtual ~IScriptComponent() = default;

  virtual void OnUpdate() = 0;
  virtual void OnFixedUpdate() = 0;
  virtual void OnInit() = 0;
  virtual void OnDestroy() = 0;
  virtual void OnEvent() = 0;
};

// ----------------------------- Object --------------------------------------

// --------------------- Object Registry ------------------------------------

}  // namespace avk