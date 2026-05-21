//! Scene graph and Entity-Component-System (ECS) implementation.
//!
//! ## Design
//! - **Backend-agnostic:** The scene representation is independent of the rendering backend.
//! - **Thread-safe:** The main `Scene` struct will be designed for concurrent access (`Send + Sync`).
//! - **Archetype-based ECS:** Inspired by Bevy's architecture for efficient memory layout and querying.
//!   - Entities with the same set of components (an archetype) are stored together in contiguous memory.
//!   - This is a simplified implementation focusing on the core concepts.
//!
//! The ECS is therefore a virtual "SQL-like" table, stored sparsely as a set of archetypes, each of which
//! is a *Sparse-Dense Tombstone Array* `Vec<Option<_>>`
//! 1. O(1) Hole Creation: Removing an entity or component simply calls .take() and pushes the index to a free_slots stack. Nothing shifts.
//! 2. Generational Compaction: The Archetype organically monitors its own fragmentation. When holes exceed a calculated threshold (e.g. >25% and >64 elements), it performs an ultra-fast Two-Pointer defragmentation (compact()) that seamlessly sorts the holes to the end, shifts active rows left, and correctly resyncs your SlotMap.
//! 3. Column Dropping: Safely stripping a component (with or without ThreadPool) becomes virtually instantaneous.

// TODO add the cache class for meshes and billboard data as specified in simulation api

// TODO reduce duplicate code on query

// TODO add tests for new methods

use crate::{
  simulation::comet::Comet,
  types,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{
  math::{
    FloatLike,
    matrix::{Matrix4, mat4::Mat4x4f32},
    quaternion::Quaternion,
    safe_div,
    vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
  },
  os::pool::{ThreadPool, chunked::ThreadPoolChunkedExt, tasklet::ThreadPoolExt},
};
use alloc::{boxed::Box, string::String, vec::Vec};
use core::any::{Any, TypeId};
use hashbrown::{HashMap, HashSet};
use slotmap::{SlotMap, new_key_type};
use spin::RwLock;
use thiserror::Error;

pub mod almanac_planet;
pub mod camera;
pub mod interaction;
pub mod particles;
pub mod script_components;
pub mod text;
pub mod trajectory;
pub mod ui;

pub use almanac_planet::AlmanacPlanet;
pub use particles::{
  GaussianParams, ParticleData, ParticleEmitterComponent, ParticleSystemComponent,
};
pub use ui::{Transform2DComponent, UiComponent};

/// An error that can occur when adding a component.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum AddComponentError {
  #[error("Entity not found.")]
  EntityNotFound,
  #[error("Component already exists for this entity.")]
  ComponentAlreadyExists,
  #[error("Component type not registered.")]
  ComponentNotRegistered,
  #[error("A dependency is not satisfied: '{missing}'")]
  DependencyNotSatisfied { missing: &'static str },
}

impl From<AddComponentError> for types::EngineError {
  fn from(value: AddComponentError) -> Self {
    Self::InvalidOperation(match value {
      AddComponentError::EntityNotFound => "AddComponentError::EntityNotFound",
      AddComponentError::ComponentAlreadyExists => "AddComponentError::ComponentAlreadyExists",
      AddComponentError::ComponentNotRegistered => "AddComponentError::ComponentNotRegistered",
      AddComponentError::DependencyNotSatisfied { .. } => {
        "AddComponentError::DependencyNotSatisfied"
      }
    })
  }
}

// === Core ECS Types ===

// TODO PhysicalMeshComponent should store Option<Arc> as a mesh, because
// TODO AssetCache, when inserting, should evict something if the assets content crosses a given in-memory GB threshold
// TODO and therefore, when asked about its data for rendering or for simulation, it should see whether the Comet is full, and
// TODO if not ask the AssetCache for it. If there's a miss, load from file or generate uv sphere (or any other procedural method we'll do)
// TODO so the asset cache should store an enum for the source (file or a procedural gen method call)
// TODO this means that the `bvh` and other computed properties should be brought out from the `Comet` struct and inside the
// TODO `PhysicalMeshComponent`. Cause these should always stay in memory if the component is alive.
// TODO memory threshold should be given to the `new` function of `AssetCache`, so that during tests we can give a low threshold. If threshold is too low and we are empty, allow to at least hold 1 mesh fully
// TODO then, add unit tests about this.

/// A thread-safe, basic Asset Cache
pub struct AssetCache<T> {
  // A map of file path to the loaded asset, wrapped in Arc to allow sharing
  assets: RwLock<HashMap<String, alloc::sync::Arc<T>>>,
}

impl<T> AssetCache<T> {
  /// TODO: Document this item
  pub fn new() -> Self {
    Self {
      assets: RwLock::new(HashMap::new()),
    }
  }

  /// TODO: Document this item
  pub fn get(&self, path: &str) -> Option<alloc::sync::Arc<T>> {
    self.assets.read().get(path).cloned()
  }

  /// TODO: Document this item
  pub fn insert(&self, path: String, asset: T) -> alloc::sync::Arc<T> {
    let mut map = self.assets.write();
    let arc = alloc::sync::Arc::new(asset);
    map.insert(path, arc.clone());
    arc
  }

  /// TODO: Document this item
  pub fn remove(&self, path: &str) -> Option<alloc::sync::Arc<T>> {
    self.assets.write().remove(path)
  }

  /// TODO: Document this item
  pub fn clear(&self) {
    self.assets.write().clear();
  }
}

new_key_type! {
  /// A unique identifier for an entity in the scene.
  pub struct EntityId;
}

/// A marker trait for all components.
/// Components must be `'static + Send + Sync`' to be used in the ECS.
pub trait Component: 'static + Send + Sync + core::fmt::Debug {
  fn stringify(&self) -> alloc::string::String {
    alloc::string::String::from("Component")
  }
}

/// A trait for components that can be serialized into a stable FFI representation.
pub trait ForeignSerializable: Component {
  /// The associated DTO struct that is #[repr(C)] and Blittable
  type ForeignData: Copy + Send + Sync + 'static;

  /// Stable ID for foreign environments (C#) to identify this component type
  const COMPONENT_ID: u64;

  /// Snapshot the current state into the DTO
  fn to_foreign(&self) -> Self::ForeignData;

  /// Update the component from a foreign DTO
  fn apply_foreign(&mut self, data: &Self::ForeignData);
}

// === Component Definitions ===

/// Defines the position, rotation, and scale of an entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformComponent {
  pub position: Vec3f32,
  /// Stored as a quaternion.
  pub rotation: Quat,
  pub scale: Vec3f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformDTO {
  pub px: f32,
  pub py: f32,
  pub pz: f32,
  pub rw: f32,
  pub rx: f32,
  pub ry: f32,
  pub rz: f32,
  pub sx: f32,
  pub sy: f32,
  pub sz: f32,
}

impl ForeignSerializable for TransformComponent {
  type ForeignData = TransformDTO;
  const COMPONENT_ID: u64 = 1;

  fn to_foreign(&self) -> Self::ForeignData {
    TransformDTO {
      px: self.position.x(),
      py: self.position.y(),
      pz: self.position.z(),
      rw: self.rotation.0.w(),
      rx: self.rotation.0.x(),
      ry: self.rotation.0.y(),
      rz: self.rotation.0.z(),
      sx: self.scale.x(),
      sy: self.scale.y(),
      sz: self.scale.z(),
    }
  }

  fn apply_foreign(&mut self, data: &Self::ForeignData) {
    self.position = Vec3f32::from_components(data.px, data.py, data.pz);
    self.rotation = Quat::from_components(data.rw, data.rx, data.ry, data.rz);
    self.scale = Vec3f32::from_components(data.sx, data.sy, data.sz);
  }
}

impl Default for TransformComponent {
  fn default() -> Self {
    Self {
      position: Vec3f32::zero(),
      rotation: Quat::identity(),
      scale: Vec3f32::one(),
    }
  }
}

impl Component for TransformComponent {}

impl TransformComponent {
  /// Constructs a 4x4 transformation matrix from the component's TRS
  /// (Translation, Rotation, Scale) properties.
  pub fn to_mat4<T>(&self) -> T
  where
    T: Matrix4 + From<Mat4x4f32>,
    T::Vector: Vector4, // Requires Vector4 to use `from_components`
    T::Scalar: FloatLike,
  {
    let p = &self.position;
    let q = &self.rotation;
    let s = &self.scale;

    // Use the Matrix4 trait's custom frame constructor for rotation
    // then apply scaling and translation.
    let rot_mat = T::from(Mat4x4f32::from_quat_custom_frame(*q));

    // Scaling is applied to the basis vectors (columns 0, 1, 2)
    let c0 = unsafe { rot_mat.column_unchecked(0) } * <T::Scalar as FloatLike>::from_f32(s.x());
    let c1 = unsafe { rot_mat.column_unchecked(1) } * <T::Scalar as FloatLike>::from_f32(s.y());
    let c2 = unsafe { rot_mat.column_unchecked(2) } * <T::Scalar as FloatLike>::from_f32(s.z());

    // Translation (column 3)
    let c3 = <T::Vector as Vector4>::from_components(
      <T::Scalar as FloatLike>::from_f32(p.x()),
      <T::Scalar as FloatLike>::from_f32(p.y()),
      <T::Scalar as FloatLike>::from_f32(p.z()),
      <T::Scalar as FloatLike>::from_f32(1.0),
    );

    T::from_columns(c0, c1, c2, c3)
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CameraProjection {
  Perspective {
    fov: f32,
    aspect_ratio: f32,
    near: f32,
    far: f32,
  },
  Orthographic {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
  },
}

impl Default for CameraProjection {
  fn default() -> Self {
    Self::Perspective {
      fov: 45.0_f32.to_radians(),
      aspect_ratio: 800.0 / 600.0,
      near: 0.1,
      far: 10000.0,
    }
  }
}

/// Represents a camera in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CameraComponent {
  pub projection: CameraProjection,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDTO {
  pub is_orthographic: bool,
  pub fov: f32,
  pub aspect: f32,
  pub near: f32,
  pub far: f32,
  pub ortho_left: f32,
  pub ortho_right: f32,
  pub ortho_bottom: f32,
  pub ortho_top: f32,
}

impl ForeignSerializable for CameraComponent {
  type ForeignData = CameraDTO;
  const COMPONENT_ID: u64 = 2;

  fn to_foreign(&self) -> Self::ForeignData {
    match self.projection {
      CameraProjection::Perspective {
        fov,
        aspect_ratio,
        near,
        far,
      } => CameraDTO {
        is_orthographic: false,
        fov: fov.to_degrees(),
        aspect: aspect_ratio,
        near,
        far,
        ortho_left: 0.0,
        ortho_right: 0.0,
        ortho_bottom: 0.0,
        ortho_top: 0.0,
      },
      CameraProjection::Orthographic {
        left,
        right,
        bottom,
        top,
        near,
        far,
      } => CameraDTO {
        is_orthographic: true,
        fov: 0.0,
        aspect: 0.0,
        near,
        far,
        ortho_left: left,
        ortho_right: right,
        ortho_bottom: bottom,
        ortho_top: top,
      },
    }
  }

  fn apply_foreign(&mut self, data: &Self::ForeignData) {
    if data.is_orthographic {
      self.projection = CameraProjection::Orthographic {
        left: data.ortho_left,
        right: data.ortho_right,
        bottom: data.ortho_bottom,
        top: data.ortho_top,
        near: data.near,
        far: data.far,
      };
    } else {
      self.projection = CameraProjection::Perspective {
        fov: data.fov.to_radians(),
        aspect_ratio: data.aspect,
        near: data.near,
        far: data.far,
      };
    }
  }
}

impl Component for CameraComponent {}

impl CameraComponent {
  pub fn new_persp(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
    Self {
      projection: CameraProjection::Perspective {
        fov,
        aspect_ratio,
        near,
        far,
      },
    }
  }

  pub fn new_ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
    Self {
      projection: CameraProjection::Orthographic {
        left,
        right,
        bottom,
        top,
        near,
        far,
      },
    }
  }

  pub fn update_for_extent(&mut self, width: u32, height: u32) {
    let aspect_ratio = if height == 0 {
      1.0
    } else {
      width as f32 / height as f32
    };
    match &mut self.projection {
      CameraProjection::Perspective {
        aspect_ratio: aspect,
        ..
      } => {
        *aspect = aspect_ratio;
      }
      CameraProjection::Orthographic {
        left,
        right,
        bottom,
        top,
        ..
      } => {
        let current_height = *top - *bottom;
        let half_w = (current_height * aspect_ratio) / 2.0;
        let center_x = (*right + *left) / 2.0;
        *left = center_x - half_w;
        *right = center_x + half_w;
      }
    }
  }

  pub fn get_projection_matrix(&self) -> Mat4x4f32 {
    match self.projection {
      CameraProjection::Perspective {
        fov,
        aspect_ratio,
        near,
        far,
      } => Mat4x4f32::perspective_vk(fov, aspect_ratio, near, far),
      CameraProjection::Orthographic {
        left,
        right,
        bottom,
        top,
        near,
        far,
      } => Mat4x4f32::orthographic_vk(left, right, bottom, top, near, far),
    }
  }

  pub fn near_plane(&self) -> f32 {
    match self.projection {
      CameraProjection::Perspective { near, .. } => near,
      CameraProjection::Orthographic { near, .. } => near,
    }
  }

  pub fn far_plane(&self) -> f32 {
    match self.projection {
      CameraProjection::Perspective { far, .. } => far,
      CameraProjection::Orthographic { far, .. } => far,
    }
  }
}

/// A marker component for entities that should be rendered as a cursor.
#[derive(Debug, PartialEq)]
pub struct CursorComponent {}
impl Component for CursorComponent {}

#[derive(Debug, PartialEq, Clone, Copy)]
/// TODO: Document this item
pub struct Marker {
  pub local_pos: [f32; 3],
  pub color: [f32; 3],
  pub size: f32,
}

#[derive(Debug, PartialEq, Clone)]
/// TODO: Document this item
pub struct MarkersComponent {
  pub markers: alloc::vec::Vec<Marker>,
}
impl Component for MarkersComponent {}

/// A physically-based mesh loaded from a glTF file.
#[derive(Debug)]
pub struct PhysicalMeshComponent {
  pub asset_path: alloc::string::String,
  pub mesh: alloc::sync::Arc<Comet>,
  pub emissive_intensity: f32,
  pub emissive_color: [f32; 3],
  pub use_new_path: bool,
  /// 0 = off, 1 = RGB color, 2 = alpha distribution, 3 = spherical grid (see physical_mesh2.frag)
  pub paint_display_mode: u32,
  pub sphere_center: [f32; 3],
  pub sphere_radius: f32,
  pub grid_color: [f32; 3],
  pub grid_density: f32,
}

impl Clone for PhysicalMeshComponent {
  fn clone(&self) -> Self {
    Self {
      asset_path: self.asset_path.clone(),
      mesh: self.mesh.clone(),
      emissive_intensity: self.emissive_intensity,
      emissive_color: self.emissive_color,
      use_new_path: self.use_new_path,
      paint_display_mode: self.paint_display_mode,
      sphere_center: self.sphere_center,
      sphere_radius: self.sphere_radius,
      grid_color: self.grid_color,
      grid_density: self.grid_density,
    }
  }
}

impl PartialEq for PhysicalMeshComponent {
  fn eq(&self, other: &Self) -> bool {
    self.asset_path == other.asset_path
      && self.emissive_intensity == other.emissive_intensity
      && self.emissive_color == other.emissive_color
      && self.paint_display_mode == other.paint_display_mode
      && self.sphere_center == other.sphere_center
      && self.sphere_radius == other.sphere_radius
      && self.grid_color == other.grid_color
      && self.grid_density == other.grid_density
  }
}

impl Component for PhysicalMeshComponent {}

/// Represents a 2D texture billboard.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BillboardType {
  WorldSpace { width: f32, height: f32 },
  ScreenSpace { pct_width: f32, pct_height: f32 },
}

#[derive(Debug)]
/// TODO: Document this item
pub struct ImageBillboardComponent {
  pub texture_id: u64,
  pub billboard_type: BillboardType,
}
impl Component for ImageBillboardComponent {}

/// Tags an entity as a Renderable Sun
#[derive(Clone, Copy, Debug)]
pub struct SunComponent {
  pub resolution: (u32, u32, u32),
  pub radius: f32,
}
impl Component for SunComponent {}

/// A marker component for entities that should be rendered as a background sky.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SkyComponent {}
impl Component for SkyComponent {}

/// A component that stores whether a gizmo should be rendered for this entity
#[derive(Clone, Debug, PartialEq)]
pub struct GizmoComponent {
  pub gizmo_visible: bool,
  pub gizmo_scale: f32,
}
impl Component for GizmoComponent {}

/// A component that renders a wireframe sphere and the 3 local axes (R,G,B) for the local frame.
#[derive(Clone, Debug, PartialEq)]
pub struct SphereGizmoComponent {
  pub radius: f32,
  pub subdivisions: f32,
  pub local_frame: Mat4x4f32,
}
impl Component for SphereGizmoComponent {}

/// A marker component for entities that should be rendered as an infinite grid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GridComponent {}
impl Component for GridComponent {}

/// A marker component for the selected entity.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SelectedComponent {}
impl Component for SelectedComponent {}

/// A marker component for the entity being followed by the camera.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FollowingComponent {}
impl Component for FollowingComponent {}

/// A marker component indicating an entity should be hidden from rendering.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HiddenComponent {}
impl Component for HiddenComponent {}

/// TODO: Deduplicate (in math::physics there's this exact thing)
/// A custom force emitter applying physics on rigid bodies and particles
#[derive(Clone, Copy, Debug)]
pub enum ForceEmitterComponent {
  Gravity {
    mu: f32,
  },
  Planar {
    normal: Vec3f32,
    base_force: f32,
    trunc_distance: f32,
  },
}
impl Component for ForceEmitterComponent {}

/// A component that stores debug render states for BVH nodes
#[derive(Clone, Debug)]
pub struct BvhDebugComponent {
  pub node_render_states: alloc::vec::Vec<bool>,
  pub use_new_path: bool,
}
impl Component for BvhDebugComponent {}

/// A measurement line between two points with an associated distance.
#[derive(Clone, PartialEq, Debug)]
pub struct MeasurementComponent {
  pub pos1: Vec3f32,
  pub pos2: Vec3f32,
  /// Size in PTs for character rendering
  pub points: f32,
  pub significant_digits: u32,
}
impl Component for MeasurementComponent {}

impl Default for MeasurementComponent {
  fn default() -> Self {
    Self {
      pos1: Vec3f32::from_components(0.0, 0.0, 0.0),
      pos2: Vec3f32::from_components(0.0, 0.0, 0.0),
      points: 12.0,
      significant_digits: 3,
    }
  }
}

/// A component that defines a uniform or gradient background color
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct BackgroundComponent {
  pub color_top: [f32; 4],
  pub color_bottom: [f32; 4],
}

impl Component for BackgroundComponent {}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// TODO: Document this item
pub enum ReferenceFrameType {
  Macro = 0,
  Micro = 1,
}

#[repr(C, align(16))]
#[derive(Clone, Debug, PartialEq)]
/// TODO: Document this item
pub struct ReferenceFrameComponent {
  pub frame_type: ReferenceFrameType,
  pub scale: f32,
  pub soi_radius: f32,
  pub _padding: u32,
}

impl ReferenceFrameComponent {
  #[inline(always)]
  /// TODO: Document this item
  pub fn micro_to_macro(
    p_micro: Vec3f32,
    v_micro: Vec3f32,
    c_macro: Vec3f32,
    c_vel_macro: Vec3f32,
    scale: f32,
  ) -> (Vec3f32, Vec3f32) {
    let p_macro = c_macro + (p_micro * scale);
    let v_macro = c_vel_macro + (v_micro * scale);
    (p_macro, v_macro)
  }

  #[inline(always)]
  /// TODO: Document this item
  pub fn macro_to_micro(
    p_macro: Vec3f32,
    v_macro: Vec3f32,
    c_macro: Vec3f32,
    c_vel_macro: Vec3f32,
    scale: f32,
  ) -> (Vec3f32, Vec3f32) {
    let p_micro = (p_macro - c_macro) / scale;
    let v_micro = (v_macro - c_vel_macro) / scale;
    (p_micro, v_micro)
  }
}

impl Component for ReferenceFrameComponent {}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderShape {
  Sphere { radius: f32 },
  OBB { half_extents: Vec3f32 },
}

impl Default for ColliderShape {
  fn default() -> Self {
    Self::Sphere { radius: 1.0 }
  }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ColliderComponent {
  pub shape: ColliderShape,
  pub mass: f32,
  pub restitution: f32,
  pub friction: f32,
}
impl Component for ColliderComponent {}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
/// TODO: Document this item
pub struct KinematicComponent {
  pub velocity: Vec3f32,
  pub angular_velocity: Vec3f32,
}
impl Component for KinematicComponent {}

/// Necessary boilerplate for rendering system
pub enum RenderableDataRef<'a> {
  ImageBillboard(&'a ImageBillboardComponent),
  PhysicalMesh(&'a PhysicalMeshComponent),
  Cursor(&'a CursorComponent),
  Markers(&'a MarkersComponent),
  Measurement(&'a MeasurementComponent),
  Gizmo(&'a GizmoComponent),
  BvhWireframe(
    &'a BvhDebugComponent,
    &'a [crate::math::collision::linear_bvh::LinearBound<f32>],
  ),
  ParticleSystem(
    &'a particles::ParticleSystemComponent,
    &'a particles::ParticleEmitterComponent,
  ),
}

impl<'a> RenderableDataRef<'a> {
  /// TODO: Document this item
  pub fn index_count(&self) -> u32 {
    match self {
      RenderableDataRef::ImageBillboard(_) => 4,
      RenderableDataRef::PhysicalMesh(mesh) => mesh.mesh.indices.len() as u32,
      RenderableDataRef::Cursor(_) => 4, // 4 vertices for the quad cursor
      RenderableDataRef::Markers(m) => (m.markers.len() * 4) as u32,
      RenderableDataRef::Measurement(_) => 6, // 6 vertices for line list
      RenderableDataRef::Gizmo(_) => 6,
      RenderableDataRef::BvhWireframe(_, nodes) => (nodes.len() * 24) as u32, // Approximation
      RenderableDataRef::ParticleSystem(p, _) => (p.particles.read().len() * 4) as u32,
    }
  }
}

/// Information about a component for a specific entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityComponentInfo {
  pub type_name: &'static str,
  /// True if the entity has this component, false otherwise.
  pub is_present: bool,
}

// === ECS Storage ===

/// A trait for type-erased component storage handling Sparse Arrays (Option wrapping).
trait ComponentStorage: Send + Sync + core::fmt::Debug {
  fn as_any(&self) -> &dyn Any;
  fn as_mut_any(&mut self) -> &mut dyn Any;

  /// Appends a hole (None) to the end of the storage.
  fn push_none(&mut self);

  /// Inserts a component dynamically at an exact index, expanding storage if needed.
  fn insert_any(&mut self, index: usize, component: Box<dyn Any + Send + Sync>);

  /// Migrates a component out of this storage into another, explicitly targeting a new slot.
  fn move_to(&mut self, src_index: usize, target_index: usize, other: &mut dyn ComponentStorage);

  /// Leaves a hole at the given index, dropping the component.
  fn remove(&mut self, index: usize);

  /// Swaps two elements. Essential for the high-performance memory defragmenter.
  fn swap_elements(&mut self, a: usize, b: usize);

  /// Truncates the underlying array to chop off empty trailing space.
  fn truncate(&mut self, len: usize);

  fn component_type_id(&self) -> TypeId;
}

impl<T: Component> ComponentStorage for Vec<Option<T>> {
  fn as_any(&self) -> &dyn Any {
    self
  }
  fn as_mut_any(&mut self) -> &mut dyn Any {
    self
  }

  fn push_none(&mut self) {
    self.push(None);
  }

  fn insert_any(&mut self, index: usize, component: Box<dyn Any + Send + Sync>) {
    if let Ok(c) = component.downcast::<T>() {
      if index >= self.len() {
        self.resize_with(index + 1, || None);
      }
      self[index] = Some(*c);
    }
  }

  fn move_to(&mut self, src_index: usize, dest_index: usize, other: &mut dyn ComponentStorage) {
    let item = self[src_index].take();
    if let Some(other_vec) = other.as_mut_any().downcast_mut::<Vec<Option<T>>>() {
      if dest_index >= other_vec.len() {
        other_vec.resize_with(dest_index + 1, || None);
      }
      other_vec[dest_index] = item;
    }
  }

  fn remove(&mut self, index: usize) {
    if index < self.len() {
      self[index] = None;
    }
  }

  fn swap_elements(&mut self, a: usize, b: usize) {
    self.swap(a, b);
  }

  fn truncate(&mut self, len: usize) {
    self.truncate(len);
  }

  fn component_type_id(&self) -> TypeId {
    TypeId::of::<T>()
  }
}

#[derive(Debug)]
struct ComponentMeta {
  dependencies: Vec<TypeId>,
  new_storage: fn() -> RwLock<Box<dyn ComponentStorage>>,
  type_name: &'static str,
}

#[derive(Debug)]
struct Archetype {
  components: HashMap<TypeId, RwLock<Box<dyn ComponentStorage>>>,
  component_types: HashSet<TypeId>,
  entities: Vec<Option<EntityId>>, // Option natively models O(1) allocation holes
  free_slots: Vec<usize>,          // Stack of reusable memory rows
}

impl Archetype {
  fn has_components(&self, component_types: &[TypeId]) -> bool {
    component_types.iter().all(|t| self.component_types.contains(t))
  }

  /// Allocates a new row, recycling a hole if available to keep cache packed.
  fn allocate_slot(&mut self) -> usize {
    if let Some(idx) = self.free_slots.pop() {
      idx
    } else {
      let idx = self.entities.len();
      self.entities.push(None);
      for storage in self.components.values_mut() {
        storage.get_mut().push_none();
      }
      idx
    }
  }

  /// Reclaims a row, erasing the entity and all associated component memory.
  fn free_slot(&mut self, index: usize) {
    self.entities[index] = None;
    for storage in self.components.values_mut() {
      storage.get_mut().remove(index);
    }
    self.free_slots.push(index);
  }

  /// Check if holes exceed a logical fragmentation limit.
  fn needs_compaction(&self) -> bool {
    self.free_slots.len() >= 64 && self.free_slots.len() > self.entities.len() / 4
  }

  /// Generational GC: Sweeps memory efficiently using Two-Pointers when fragmentation exceeds thresholds.
  fn compact(&mut self, entities_map: &mut SlotMap<EntityId, EntityLocation>) {
    if !self.needs_compaction() {
      return;
    }

    let mut left = 0;
    let mut right = self.entities.len();
    let mut comp_guards: Vec<_> = self.components.values().map(|c| c.write()).collect();

    while left < right {
      if self.entities[left].is_none() {
        right -= 1;
        while left < right && self.entities[right].is_none() {
          right -= 1;
        }
        if left < right {
          self.entities.swap(left, right);
          for guard in &mut comp_guards {
            guard.swap_elements(left, right);
          }

          // Correctly update the SlotMap pointer for the trailing Entity that just got dragged forwards
          if let Some(moved_entity) = self.entities[left] {
            if let Some(loc) = entities_map.get_mut(moved_entity) {
              loc.row_index = left;
            }
          }
        }
      }
      left += 1;
    }

    let new_len = right;
    self.entities.truncate(new_len);
    for guard in &mut comp_guards {
      guard.truncate(new_len);
    }
    self.free_slots.clear();
  }
}

/// Stores the location of an entity within the ECS.
#[derive(Clone, Copy, Debug)]
struct EntityLocation {
  archetype_index: usize,
  row_index: usize,
}

/// The main scene struct, containing all entities and their components.
#[derive(Debug)]
pub struct Scene {
  entities: RwLock<SlotMap<EntityId, EntityLocation>>,
  archetypes: RwLock<Vec<Archetype>>,
  component_meta: RwLock<HashMap<TypeId, ComponentMeta>>,
  hierarchy: RwLock<SceneHierarchy>,
  names: RwLock<HashMap<EntityId, String>>,
  pub texture_cache: alloc::sync::Arc<spin::RwLock<crate::simulation::texture_cache::TextureCache>>,
}

#[derive(Default, Debug)]
/// TODO: Document this item
pub struct SceneHierarchy {
  parents: HashMap<EntityId, EntityId>,
  children: HashMap<EntityId, Vec<EntityId>>,
}

impl SceneHierarchy {
  fn set_parent(&mut self, child: EntityId, parent: Option<EntityId>) {
    if let Some(old_parent) = self.parents.remove(&child) {
      if let Some(children) = self.children.get_mut(&old_parent) {
        children.retain(|c| *c != child);
      }
    }
    if let Some(new_parent) = parent {
      self.parents.insert(child, new_parent);
      self.children.entry(new_parent).or_default().push(child);
    }
  }

  fn remove_entity(&mut self, entity: EntityId) {
    if let Some(parent) = self.parents.remove(&entity) {
      if let Some(children) = self.children.get_mut(&parent) {
        children.retain(|c| *c != entity);
      }
    }
    if let Some(children) = self.children.remove(&entity) {
      for child in children {
        self.remove_entity(child);
      }
    }
  }

  /// Returns the root entity of the scene.
  /// If the hierarchy is empty, it returns `None`.
  pub fn get_first_root(&self) -> Option<EntityId> {
    self.children.keys().copied().find(|id| !self.parents.contains_key(id))
  }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// TODO: Document this item
pub enum HasComponentResultEnum {
  EntityHasComponent = 0,
  EntityNotFound = 1,
  ComponentNotFound = 2,
}

impl Into<bool> for HasComponentResultEnum {
  fn into(self) -> bool {
    self == HasComponentResultEnum::EntityHasComponent
  }
}

impl Scene {
  /// TODO: Document this item
  pub fn new(
    texture_cache: alloc::sync::Arc<spin::RwLock<crate::simulation::texture_cache::TextureCache>>,
  ) -> Self {
    let empty_archetype = Archetype {
      components: HashMap::new(),
      component_types: HashSet::new(),
      entities: Vec::new(),
      free_slots: Vec::new(),
    };

    Self {
      entities: RwLock::new(SlotMap::with_key()),
      archetypes: RwLock::new(alloc::vec![empty_archetype]),
      component_meta: RwLock::new(HashMap::new()),
      hierarchy: RwLock::new(SceneHierarchy::default()),
      names: RwLock::new(HashMap::new()),
      texture_cache,
    }
  }

  pub fn get_root(&self) -> Option<EntityId> {
    let hierarchy = self.hierarchy.read();
    hierarchy.get_first_root()
  }

  /// TODO: Document this item
  pub fn entity_count(&self) -> usize {
    self.entities.read().len()
  }

  /// TODO: Document this item
  pub fn hierarchy_depth(&self) -> usize {
    let hierarchy = self.hierarchy.read();
    let mut max_depth = 0;
    for entity in self.entities.read().keys() {
      let mut depth = 0;
      let mut curr = entity;
      while let Some(&parent) = hierarchy.parents.get(&curr) {
        depth += 1;
        curr = parent;
      }
      max_depth = max_depth.max(depth);
    }
    max_depth
  }

  /// TODO: Document this item
  pub fn hierarchy_breadth(&self) -> usize {
    let hierarchy = self.hierarchy.read();
    hierarchy.children.values().map(|children| children.len()).max().unwrap_or(0)
  }

  /// TODO: Document this item
  pub fn should_parallelize(&self) -> bool {
    let size = self.entity_count();
    let _depth = self.hierarchy_depth();
    let breadth = self.hierarchy_breadth();

    // Simple heuristic for parallelization threshold
    // Parallelize if we have a lot of entities, or if the scene graph is very wide
    size > 1000 || breadth > 100
  }

  /// TODO: Document this item
  pub fn add_default_camera<S>(
    &self,
    name: S,
    inital_pos: Vec3f32,
    parent: EntityId,
  ) -> EngineResult<EntityId>
  where
    S: Into<String>,
  {
    let name = name.into();
    if !self.entities.read().contains_key(parent) {
      return Err(EngineError::InvalidOperation(
        "scene:add_camera Parent not present",
      ));
    }
    if self.get_entity_by_name(&name).is_some() {
      return Err(EngineError::InvalidOperation(
        "scene:add_camera name already exists",
      ));
    }

    let camera_entity = self.spawn_entity(&name);
    self.add_component(
      camera_entity,
      TransformComponent {
        position: inital_pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    self.add_component(camera_entity, CameraComponent::default())?;
    self.set_parent(camera_entity, Some(parent));
    Ok(camera_entity)
  }

  /// TODO: Document this item
  pub fn set_parent(&self, child: EntityId, parent: Option<EntityId>) {
    let entities = self.entities.read();
    if !entities.contains_key(child)
      || (parent.is_some() && !entities.contains_key(parent.unwrap()))
    {
      return;
    }
    self.hierarchy.write().set_parent(child, parent);
  }

  /// TODO: Document this item
  pub fn get_parent(&self, entity: EntityId) -> Option<EntityId> {
    let entities = self.entities.read();
    if !entities.contains_key(entity) {
      return None;
    }
    self.hierarchy.read().parents.get(&entity).cloned()
  }

  /// TODO: Document this item
  pub fn get_children(&self, entity: EntityId) -> Option<alloc::vec::Vec<EntityId>> {
    let entities = self.entities.read();
    if !entities.contains_key(entity) {
      return None;
    }
    self.hierarchy.read().children.get(&entity).cloned()
  }

  /// TODO: Document this item
  pub fn get_entity_component_names(&self, entity: EntityId) -> Vec<&'static str> {
    let archetypes = self.archetypes.read();
    let entities = self.entities.read();
    let location = match entities.get(entity) {
      Some(l) => l,
      None => return Vec::new(),
    };
    let archetype = &archetypes[location.archetype_index];
    let component_meta = self.component_meta.read();

    let mut names = Vec::new();
    for &type_id in &archetype.component_types {
      if let Some(meta) = component_meta.get(&type_id) {
        names.push(meta.type_name);
      }
    }
    names
  }

  /// TODO: Document this item
  pub fn spawn_reference_frame(
    &self,
    name: &str,
    parent: Option<EntityId>,
    transform: TransformComponent,
    frame_type: ReferenceFrameType,
    scale: f32,
    soi_radius: f32,
  ) -> EntityId {
    let entity = self.spawn_entity(name);
    if let Some(p) = parent {
      self.set_parent(entity, Some(p));
    }
    let _ = self.add_component(entity, transform);
    let _ = self.add_component(
      entity,
      ReferenceFrameComponent {
        frame_type,
        scale,
        soi_radius,
        _padding: 0,
      },
    );
    entity
  }

  // TODO error report with rollback (ie removal)
  pub fn spawn_camera(
    &self,
    name: &str,
    parent: Option<EntityId>,
    t: TransformComponent,
    c: CameraComponent,
  ) -> EntityId {
    let entity = self.spawn_entity(name);
    if let Some(p) = parent {
      self.set_parent(entity, Some(p));
    }
    let _ = self.add_component(entity, t);
    let _ = self.add_component(entity, c);
    entity
  }

  pub fn spawn_rigidbody(
    &self,
    name: &str,
    parent: Option<EntityId>,
    transform: TransformComponent,
  ) -> EntityId {
    let entity = self.spawn_entity(name);
    if let Some(p) = parent {
      self.set_parent(entity, Some(p));
    }
    let _ = self.add_component(entity, transform);
    // Dynamic physics state would be initialized elsewhere based on this
    entity
  }

  pub fn spawn_entity(&self, name: impl Into<alloc::string::String>) -> EntityId {
    let mut actual_name = name.into();
    {
      let names = self.names.read();
      if names.values().any(|n| *n == actual_name) {
        let mut counter = 1;
        loop {
          let try_name = alloc::format!("{}_{}", actual_name, counter);
          if !names.values().any(|n| *n == try_name) {
            actual_name = try_name;
            break;
          }
          counter += 1;
        }
      }
    }

    let entity_id = {
      let mut archetypes = self.archetypes.write();
      let mut entities = self.entities.write();

      let archetype = &mut archetypes[0];
      let row_index = archetype.allocate_slot();

      let entity_location = EntityLocation {
        archetype_index: 0,
        row_index,
      };
      let entity_id = entities.insert(entity_location);
      archetype.entities[row_index] = Some(entity_id);
      entity_id
    };

    self.names.write().insert(entity_id, actual_name);
    entity_id
  }

  /// TODO: Document this item
  pub fn get_entity_by_name(&self, name: &str) -> Option<EntityId> {
    self.names.read().iter().find(|(_, n)| *n == name).map(|(id, _)| *id)
  }

  /// TODO: Document this item
  pub fn get_name(&self, entity: EntityId) -> Option<alloc::string::String> {
    self.names.read().get(&entity).cloned()
  }

  /// TODO: Document this item
  pub fn set_name(&self, entity: EntityId, name: impl Into<alloc::string::String>) {
    let mut actual_name = name.into();
    {
      let names = self.names.read();
      if names.values().any(|n| *n == actual_name) {
        let mut counter = 1;
        loop {
          let try_name = alloc::format!("{}_{}", actual_name, counter);
          if !names.values().any(|n| *n == try_name) {
            actual_name = try_name;
            break;
          }
          counter += 1;
        }
      }
    }
    self.names.write().insert(entity, actual_name);
  }

  pub fn register_all_crate_components(&self) {
    let transform_type_id = [TypeId::of::<TransformComponent>()];
    let transform_and_mesh = [
      TypeId::of::<TransformComponent>(),
      TypeId::of::<PhysicalMeshComponent>(),
    ];

    // this module
    self.register_component::<TransformComponent>(&[]);
    self.register_component::<CameraComponent>(&transform_type_id);
    self.register_component::<CursorComponent>(&transform_type_id);
    self.register_component::<MarkersComponent>(&transform_type_id);
    self.register_component::<PhysicalMeshComponent>(&transform_type_id);
    self.register_component::<ImageBillboardComponent>(&transform_type_id);
    self.register_component::<SunComponent>(&transform_type_id);
    self.register_component::<SkyComponent>(&[]);
    self.register_component::<BackgroundComponent>(&[]);
    self.register_component::<GizmoComponent>(&transform_type_id);
    self.register_component::<SphereGizmoComponent>(&transform_type_id);
    self.register_component::<GridComponent>(&transform_type_id);
    self.register_component::<SelectedComponent>(&transform_type_id);
    self.register_component::<FollowingComponent>(&transform_type_id);
    self.register_component::<HiddenComponent>(&[]);
    self.register_component::<ForceEmitterComponent>(&transform_type_id);
    self.register_component::<BvhDebugComponent>(&transform_type_id);
    self.register_component::<MeasurementComponent>(&[]);
    self.register_component::<ParticleEmitterComponent>(&transform_and_mesh);
    self.register_component::<particles::ParticleSystemComponent>(&transform_type_id);
    self.register_component::<ReferenceFrameComponent>(&[]);
    self.register_component::<KinematicComponent>(&transform_and_mesh);
    self.register_component::<ColliderComponent>(&transform_and_mesh);

    // ui module
    self.register_component::<ui::Transform2DComponent>(&[]);
    self.register_component::<ui::UiComponent>(&[]);
    self.register_component::<ui::ScreenSpaceTextComponent>(&[TypeId::of::<
      ui::Transform2DComponent,
    >()]);

    // trajectory module
    self.register_component::<trajectory::TrajectoryComponent>(&transform_type_id);

    // script components module
    self.register_component::<script_components::UpdateComponent>(&[]);

    // almanac planet module
    self.register_component::<almanac_planet::AlmanacPlanet>(&transform_and_mesh);
  }

  /// TODO: Document this item
  pub fn register_component<T: Component>(&self, dependencies: &[TypeId]) {
    let mut meta = self.component_meta.write();
    meta.insert(
      TypeId::of::<T>(),
      ComponentMeta {
        dependencies: dependencies.to_vec(),
        new_storage: || RwLock::new(Box::new(Vec::<Option<T>>::new())),
        type_name: core::any::type_name::<T>(),
      },
    );
  }

  /// TODO: Document this item
  pub fn add_component<T: Component>(
    &self,
    entity_id: EntityId,
    component: T,
  ) -> Result<(), AddComponentError> {
    let new_component_type = TypeId::of::<T>();

    let src_location = {
      let entities = self.entities.read();
      *entities.get(entity_id).ok_or(AddComponentError::EntityNotFound)?
    };

    let (target_archetype_index, is_new_archetype) = {
      let archetypes = self.archetypes.read();
      let src_archetype = &archetypes[src_location.archetype_index];

      if src_archetype.component_types.contains(&new_component_type) {
        return Err(AddComponentError::ComponentAlreadyExists);
      }

      let meta = self.component_meta.read();
      let component_meta =
        meta.get(&new_component_type).ok_or(AddComponentError::ComponentNotRegistered)?;

      if !src_archetype.has_components(&component_meta.dependencies) {
        let missing_dep = component_meta
          .dependencies
          .iter()
          .find(|t| !src_archetype.component_types.contains(*t))
          .unwrap();
        let missing_name = meta.get(missing_dep).map_or("Unknown", |m| m.type_name);
        aethervk_oshal_rlib::log!(
          "Missing dependency: {} when adding {}",
          missing_name,
          core::any::type_name::<T>()
        );
        return Err(AddComponentError::DependencyNotSatisfied {
          missing: missing_name,
        });
      }

      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      let found_index =
        archetypes.iter().position(|arch| arch.component_types == target_component_types);
      (found_index, found_index.is_none())
    };

    let target_archetype_index = if is_new_archetype {
      let mut archetypes = self.archetypes.write();
      let meta = self.component_meta.read();

      let src_archetype = &archetypes[src_location.archetype_index];
      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      if let Some(index) =
        archetypes.iter().position(|arch| arch.component_types == target_component_types)
      {
        index
      } else {
        let mut new_arch = Archetype {
          component_types: target_component_types,
          components: HashMap::new(),
          entities: Vec::new(),
          free_slots: Vec::new(),
        };
        for type_id in &new_arch.component_types {
          let storage_fn = meta.get(type_id).unwrap().new_storage;
          new_arch.components.insert(*type_id, storage_fn());
        }
        archetypes.push(new_arch);
        archetypes.len() - 1
      }
    } else {
      target_archetype_index.unwrap()
    };

    if src_location.archetype_index != target_archetype_index {
      let mut archetypes = self.archetypes.write(); // TODO deadlock
      let mut entities = self.entities.write();

      let (src_arch, target_arch) = if src_location.archetype_index < target_archetype_index {
        let (left, right) = archetypes.split_at_mut(target_archetype_index);
        (&mut left[src_location.archetype_index], &mut right[0])
      } else {
        let (left, right) = archetypes.split_at_mut(src_location.archetype_index);
        (&mut right[0], &mut left[target_archetype_index])
      };

      let target_row_index = target_arch.allocate_slot();
      target_arch.entities[target_row_index] = Some(entity_id);

      for (type_id, target_storage_lock) in target_arch.components.iter() {
        if *type_id == new_component_type {
          continue;
        }
        let mut src_storage = src_arch.components[type_id].write();
        let mut target_storage = target_storage_lock.write();
        src_storage.move_to(
          src_location.row_index,
          target_row_index,
          &mut **target_storage,
        );
      }

      target_arch.components[&new_component_type]
        .write()
        .insert_any(target_row_index, Box::new(component));

      // Frees the src slot and drops its trailing components without shifting memory!
      src_arch.free_slot(src_location.row_index);

      *entities.get_mut(entity_id).unwrap() = EntityLocation {
        archetype_index: target_archetype_index,
        row_index: target_row_index,
      };

      src_arch.compact(&mut entities);
    }

    Ok(())
  }

  /// TODO: Document this item
  pub fn remove_entity(&self, entity_id: EntityId) {
    let src_location = {
      let mut entities = self.entities.write();
      if let Some(loc) = entities.remove(entity_id) {
        loc
      } else {
        return;
      }
    };

    self.hierarchy.write().remove_entity(entity_id);
    self.names.write().remove(&entity_id);

    let mut archetypes = self.archetypes.write();
    let mut entities = self.entities.write();
    let src_arch = &mut archetypes[src_location.archetype_index];

    src_arch.free_slot(src_location.row_index);
    src_arch.compact(&mut entities); // Organically triggers when arrays begin resembling swiss-cheese
  }

  /// TODO: Document this item
  pub fn remove_component<T: Component>(&self, entity_id: EntityId) -> Result<(), &'static str> {
    let type_id_to_remove = TypeId::of::<T>();

    let src_location = {
      let entities = self.entities.read();
      *entities.get(entity_id).ok_or("Entity not found")?
    };

    // 1. Identify Target Archetype layout or flag if it needs creation
    let (target_archetype_index, is_new_archetype) = {
      let archetypes = self.archetypes.read();
      let src_archetype = &archetypes[src_location.archetype_index];

      if !src_archetype.component_types.contains(&type_id_to_remove) {
        return Err("Component not found on entity");
      }

      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.remove(&type_id_to_remove);

      let found_index =
        archetypes.iter().position(|arch| arch.component_types == target_component_types);
      (found_index, found_index.is_none())
    };

    // 2. Dynamically create the Target Archetype if it didn't exist!
    let target_archetype_index = if is_new_archetype {
      let mut archetypes = self.archetypes.write();
      let meta = self.component_meta.read();

      let src_archetype = &archetypes[src_location.archetype_index];
      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.remove(&type_id_to_remove);

      // Re-check to prevent race conditions during write lock acquisition
      if let Some(index) =
        archetypes.iter().position(|arch| arch.component_types == target_component_types)
      {
        index
      } else {
        let mut new_arch = Archetype {
          component_types: target_component_types,
          components: HashMap::new(),
          entities: Vec::new(),
          free_slots: Vec::new(), // Threshold tracker initializes at 0
        };

        // Initialize empty storage arrays for the new signature
        for type_id in &new_arch.component_types {
          let storage_fn = meta.get(type_id).unwrap().new_storage;
          new_arch.components.insert(*type_id, storage_fn());
        }
        archetypes.push(new_arch);
        archetypes.len() - 1
      }
    } else {
      target_archetype_index.unwrap()
    };

    // 3. Perform the migration safely
    if src_location.archetype_index != target_archetype_index {
      let mut archetypes = self.archetypes.write();
      let mut entities = self.entities.write();

      let (src_arch, target_arch) = if src_location.archetype_index < target_archetype_index {
        let (left, right) = archetypes.split_at_mut(target_archetype_index);
        (&mut left[src_location.archetype_index], &mut right[0])
      } else {
        let (left, right) = archetypes.split_at_mut(src_location.archetype_index);
        (&mut right[0], &mut left[target_archetype_index])
      };

      let target_row_index = target_arch.allocate_slot();

      for (type_id, target_storage_lock) in target_arch.components.iter() {
        // target_arch's components are a strict mathematical subset of src_arch, so unwrap is infallible
        let mut src_storage = src_arch.components.get(type_id).unwrap().write();
        let mut target_storage = target_storage_lock.write();

        // Move valid components to the new archetype array slot using Option::take()
        src_storage.move_to(
          src_location.row_index,
          target_row_index,
          &mut **target_storage,
        );
      }

      target_arch.entities[target_row_index] = Some(entity_id);

      // PERFECT HOLE CREATION:
      // Frees the source slot. Because `move_to` already migrated the surviving elements,
      // this natively drops the removed `T` component without executing any memory shifts!
      src_arch.free_slot(src_location.row_index);

      // Update global map
      *entities.get_mut(entity_id).unwrap() = EntityLocation {
        archetype_index: target_archetype_index,
        row_index: target_row_index,
      };

      // Compaction will intelligently absorb the hole generated in `src_arch` only if limits are exceeded
      src_arch.compact(&mut entities);
    }

    Ok(())
  }

  /// Removes an entire column (all components of type `T`) natively skipping entity tracking mapping.
  pub fn remove_column<T: Component>(&self) {
    let type_id = TypeId::of::<T>();
    let mut archetypes = self.archetypes.write();
    for arch in archetypes.iter_mut() {
      if arch.component_types.contains(&type_id) {
        arch.component_types.remove(&type_id);
        arch.components.remove(&type_id);
      }
    }
  }

  /// Extinguishes a component selectively from entities passing the target filter algorithm.
  pub fn remove_column_if<T: Component, F>(&self, mut filter: F)
  where
    F: FnMut(EntityId, &T) -> bool,
  {
    let to_remove = self.query1_res(|e, c: &T| if filter(e, c) { Some(e) } else { None });
    for (e, _) in to_remove {
      let _ = self.remove_component::<T>(e);
    }
  }

  /// Parallelized bulk drop - evaluates columns structurally using the ThreadPool filter mapping!
  pub fn remove_column_par<T: Component>(&self, pool: &ThreadPool) {
    let to_remove = self.query1_res_par(pool, |e, _: &T| Some(e));
    for (e, _) in to_remove {
      let _ = self.remove_component::<T>(e);
    }
  }

  /// TODO: Document this item
  pub fn remove_column_if_par<T: Component, F>(&self, pool: &ThreadPool, filter: F)
  where
    F: Fn(EntityId, &T) -> bool + Send + Sync,
  {
    let to_remove = self.query1_res_par(pool, |e, c: &T| if filter(e, c) { Some(e) } else { None });
    for (e, _) in to_remove {
      let _ = self.remove_component::<T>(e);
    }
  }

  /// TODO: Document this item
  pub fn remove_first_component<T: Component>(&self) -> Option<EntityId> {
    let target_entity = self.query1_first_res(|e, _: &T| Some(e));
    if let Some((e, _)) = target_entity {
      if self.remove_component::<T>(e).is_ok() {
        return Some(e);
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn has_component<T: Component>(&self, entity_id: EntityId) -> HasComponentResultEnum {
    let archetypes = self.archetypes.read();
    let archetype =
      archetypes.iter().find(|archetype| archetype.entities.iter().any(|e| *e == Some(entity_id)));
    if archetype.is_none() {
      return HasComponentResultEnum::EntityNotFound;
    }

    let archetype = unsafe { archetype.unwrap_unchecked() };
    if archetype.has_components(&[TypeId::of::<T>()]) {
      HasComponentResultEnum::EntityHasComponent
    } else {
      HasComponentResultEnum::ComponentNotFound
    }
  }

  /// TODO: Document this item
  pub fn with_component<T: Component, F, R>(&self, entity_id: EntityId, f: F) -> Option<R>
  where
    F: FnOnce(&T) -> R,
  {
    let archetypes = self.archetypes.read();
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;
    let archetype = &archetypes[location.archetype_index];

    let components_lock = archetype.components.get(&TypeId::of::<T>())?.read();
    let components = components_lock.as_any().downcast_ref::<Vec<Option<T>>>()?;

    Some(f(components[location.row_index].as_ref()?))
  }

  /// TODO: Document this item
  pub fn with_component_mut<T: Component, F, R>(&self, entity_id: EntityId, f: F) -> Option<R>
  where
    F: FnOnce(&mut T) -> R,
  {
    let archetypes = self.archetypes.read();
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;
    let archetype = &archetypes[location.archetype_index];

    let mut components_lock = archetype.components.get(&TypeId::of::<T>())?.write();
    let components = components_lock.as_mut_any().downcast_mut::<Vec<Option<T>>>()?;

    Some(f(components[location.row_index].as_mut()?))
  }

  // === Sequential Single-Threaded Queries ===

  /// TODO: Document this item
  pub fn query1<T: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T),
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &components[i]) {
              f(*entity_id, comp);
            }
          }
        }
      }
    }
  }

  /// TODO: Document this item
  pub fn query2<T1: Component, T2: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T1, &T2),
  {
    let archetypes = self.archetypes.read();
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let comp_storage1_lock = archetype.components[&type_t1].read();
        let comp_storage2_lock = archetype.components[&type_t2].read();

        let components1 = comp_storage1_lock.as_any().downcast_ref::<Vec<Option<T1>>>().unwrap();
        let components2 = comp_storage2_lock.as_any().downcast_ref::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &components1[i], &components2[i])
          {
            f(*entity_id, c1, c2);
          }
        }
      }
    }
  }

  /// TODO: Document this item
  pub fn query1_mut<T: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &mut T),
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &mut components[i]) {
              f(*entity_id, comp);
            }
          }
        }
      }
    }
  }

  /// TODO: Document this item
  pub fn query2_mut<T1: Component, T2: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &mut T1, &mut T2),
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();

        let components1 =
          comp_storage1_lock.as_mut_any().downcast_mut::<Vec<Option<T1>>>().unwrap();
        let components2 =
          comp_storage2_lock.as_mut_any().downcast_mut::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &mut components1[i], &mut components2[i])
          {
            f(*entity_id, c1, c2);
          }
        }
      }
    }
  }

  /// TODO: Document this item
  pub fn query3_mut<T1: Component, T2: Component, T3: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &mut T1, &mut T2, &mut T3),
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    let type_t3 = TypeId::of::<T3>();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1)
        && archetype.components.contains_key(&type_t2)
        && archetype.components.contains_key(&type_t3)
      {
        let mut comp_storage1_lock = archetype.components.get(&type_t1).unwrap().write();
        let mut comp_storage2_lock = archetype.components.get(&type_t2).unwrap().write();
        let mut comp_storage3_lock = archetype.components.get(&type_t3).unwrap().write();

        let components1 =
          comp_storage1_lock.as_mut_any().downcast_mut::<Vec<Option<T1>>>().unwrap();
        let components2 =
          comp_storage2_lock.as_mut_any().downcast_mut::<Vec<Option<T2>>>().unwrap();
        let components3 =
          comp_storage3_lock.as_mut_any().downcast_mut::<Vec<Option<T3>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2), Some(c3)) = (
            opt_entity,
            &mut components1[i],
            &mut components2[i],
            &mut components3[i],
          ) {
            f(*entity_id, c1, c2, c3);
          }
        }
      }
    }
  }

  pub fn query3_mut_par<T1: Component, T2: Component, T3: Component, F>(
    &self,
    pool: &ThreadPool,
    f: F,
  ) where
    F: Fn(EntityId, &mut T1, &mut T2, &mut T3) + Send + Sync,
    T1: Send,
    T2: Send,
    T3: Send,
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    let type_t3 = TypeId::of::<T3>();
    let archetypes = self.archetypes.read();

    // 1. Completely erase the type and lifetime by casting the thin pointer to a usize.
    // usize is completely detached from the borrow checker and is Send + Sync + 'static.
    let f_ptr = &f as *const F as usize;

    let mut tasks = Vec::new();
    let mut held_locks = Vec::new();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1)
        && archetype.components.contains_key(&type_t2)
        && archetype.components.contains_key(&type_t3)
      {
        let mut comp_storage1_lock = archetype.components.get(&type_t1).unwrap().write();
        let mut comp_storage2_lock = archetype.components.get(&type_t2).unwrap().write();
        let mut comp_storage3_lock = archetype.components.get(&type_t3).unwrap().write();

        let components1 =
          comp_storage1_lock.as_mut_any().downcast_mut::<Vec<Option<T1>>>().unwrap();
        let components2 =
          comp_storage2_lock.as_mut_any().downcast_mut::<Vec<Option<T2>>>().unwrap();
        let components3 =
          comp_storage3_lock.as_mut_any().downcast_mut::<Vec<Option<T3>>>().unwrap();

        struct Ptrs<A, B, C> {
          id: EntityId,
          c1: *mut A,
          c2: *mut B,
          c3: *mut C,
        }
        struct SendWrapped<A, B, C>(Ptrs<A, B, C>);
        unsafe impl<A, B, C> Send for SendWrapped<A, B, C> {}
        unsafe impl<A, B, C> Sync for SendWrapped<A, B, C> {}

        let mut ptrs: Vec<SendWrapped<T1, T2, T3>> = Vec::with_capacity(archetype.entities.len());
        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2), Some(c3)) = (
            opt_entity,
            &mut components1[i],
            &mut components2[i],
            &mut components3[i],
          ) {
            ptrs.push(SendWrapped(Ptrs {
              id: *entity_id,
              c1: c1 as *mut _,
              c2: c2 as *mut _,
              c3: c3 as *mut _,
            }));
          }
        }

        // Copy the usize so it can be moved into the closure
        let f_ptr_clone = f_ptr;

        if !ptrs.is_empty() {
          if let Ok(task) = pool.spawn_tasklet(None, move || {
            // 2. Cast the usize back to a pointer, then to a reference safely inside the thread
            let f_ref = unsafe { &*(f_ptr_clone as *const F) };

            for ptr in ptrs {
              let c1 = unsafe { &mut *ptr.0.c1 };
              let c2 = unsafe { &mut *ptr.0.c2 };
              let c3 = unsafe { &mut *ptr.0.c3 };

              f_ref(ptr.0.id, c1, c2, c3);
            }
          }) {
            tasks.push(task);
            held_locks.push((comp_storage1_lock, comp_storage2_lock, comp_storage3_lock));
          }
        }
      }
    }

    for task in tasks {
      let _ = task.wait();
    }
  }

  pub fn query1_res<T: Component, F, R>(&self, mut f: F) -> Vec<(R, EntityId)>
  where
    F: FnMut(EntityId, &T) -> Option<R>,
  {
    let mut results = Vec::new();
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &components[i]) {
              if let Some(result) = f(*entity_id, comp) {
                results.push((result, *entity_id));
              }
            }
          }
        }
      }
    }
    results
  }

  /// TODO: Document this item
  pub fn query2_res<T1: Component, T2: Component, F, R>(&self, mut f: F) -> Vec<(R, EntityId)>
  where
    F: FnMut(EntityId, &T1, &T2) -> Option<R>,
  {
    let mut results = Vec::new();
    let archetypes = self.archetypes.read();
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let comp_storage1_lock = archetype.components[&type_t1].read();
        let comp_storage2_lock = archetype.components[&type_t2].read();

        let components1 = comp_storage1_lock.as_any().downcast_ref::<Vec<Option<T1>>>().unwrap();
        let components2 = comp_storage2_lock.as_any().downcast_ref::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &components1[i], &components2[i])
          {
            if let Some(result) = f(*entity_id, c1, c2) {
              results.push((result, *entity_id));
            }
          }
        }
      }
    }
    results
  }

  /// TODO: Document this item
  pub fn query1_res_mut<T: Component, F, R>(&self, mut f: F) -> Vec<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T) -> Option<R>,
  {
    let mut results = Vec::new();
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &mut components[i]) {
              if let Some(result) = f(*entity_id, comp) {
                results.push((result, *entity_id));
              }
            }
          }
        }
      }
    }
    results
  }

  /// TODO: Document this item
  pub fn query2_res_mut<T1: Component, T2: Component, F, R>(&self, mut f: F) -> Vec<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T1, &mut T2) -> Option<R>,
  {
    let mut results = Vec::new();
    let archetypes = self.archetypes.read();
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();

        let components1 =
          comp_storage1_lock.as_mut_any().downcast_mut::<Vec<Option<T1>>>().unwrap();
        let components2 =
          comp_storage2_lock.as_mut_any().downcast_mut::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &mut components1[i], &mut components2[i])
          {
            if let Some(result) = f(*entity_id, c1, c2) {
              results.push((result, *entity_id));
            }
          }
        }
      }
    }
    results
  }

  /// TODO: Document this item
  pub fn query1_first_res<T: Component, F, R>(&self, mut f: F) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &components[i]) {
              if let Some(result) = f(*entity_id, comp) {
                return Some((result, *entity_id));
              }
            }
          }
        }
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn query2_first_res<T1: Component, T2: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T1, &T2) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let comp_storage1_lock = archetype.components[&type_t1].read();
        let comp_storage2_lock = archetype.components[&type_t2].read();

        let components1 = comp_storage1_lock.as_any().downcast_ref::<Vec<Option<T1>>>().unwrap();
        let components2 = comp_storage2_lock.as_any().downcast_ref::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &components1[i], &components2[i])
          {
            if let Some(result) = f(*entity_id, c1, c2) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn query1_res_first_mut<T: Component, F, R>(&self, mut f: F) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &mut components[i]) {
              if let Some(result) = f(*entity_id, comp) {
                return Some((result, *entity_id));
              }
            }
          }
        }
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn query2_res_first_mut<T1: Component, T2: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T1, &mut T2) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();

        let components1 =
          comp_storage1_lock.as_mut_any().downcast_mut::<Vec<Option<T1>>>().unwrap();
        let components2 =
          comp_storage2_lock.as_mut_any().downcast_mut::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &mut components1[i], &mut components2[i])
          {
            if let Some(result) = f(*entity_id, c1, c2) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn query1_without<T: Component, U: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T),
  {
    let type_t = TypeId::of::<T>();
    let type_u = TypeId::of::<U>();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap().read();
        if let Some(components) = comp_storage_lock.as_any().downcast_ref::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &components[i]) {
              f(*entity_id, comp);
            }
          }
        }
      }
    }
  }

  /// TODO: Document this item
  pub fn query1_first_res_without<T: Component, U: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T) -> Option<R>,
  {
    let type_t = TypeId::of::<T>();
    let type_u = TypeId::of::<U>();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap().read();
        if let Some(components) = comp_storage_lock.as_any().downcast_ref::<Vec<Option<T>>>() {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity_id), Some(comp)) = (opt_entity, &components[i]) {
              if let Some(result) = f(*entity_id, comp) {
                return Some((result, *entity_id));
              }
            }
          }
        }
      }
    }
    None
  }

  /// TODO: Document this item
  pub fn query2_first_res_without<T1: Component, T2: Component, U: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T1, &T2) -> Option<R>,
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    let type_u = TypeId::of::<U>();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1)
        && archetype.components.contains_key(&type_t2)
        && !archetype.components.contains_key(&type_u)
      {
        let comp_storage1_lock = archetype.components.get(&type_t1).unwrap().read();
        let comp_storage2_lock = archetype.components.get(&type_t2).unwrap().read();

        let components1 = comp_storage1_lock.as_any().downcast_ref::<Vec<Option<T1>>>().unwrap();
        let components2 = comp_storage2_lock.as_any().downcast_ref::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &components1[i], &components2[i])
          {
            if let Some(result) = f(*entity_id, c1, c2) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }
    None
  }

  /// Queries entities that possess both `T1` and `T2` components, but do NOT possess a `U` component.
  pub fn query2_without<T1: Component, T2: Component, U: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T1, &T2),
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    let type_u = TypeId::of::<U>();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1)
        && archetype.components.contains_key(&type_t2)
        && !archetype.components.contains_key(&type_u)
      {
        let comp_storage1_lock = archetype.components.get(&type_t1).unwrap().read();
        let comp_storage2_lock = archetype.components.get(&type_t2).unwrap().read();

        let components1 = comp_storage1_lock.as_any().downcast_ref::<Vec<Option<T1>>>().unwrap();
        let components2 = comp_storage2_lock.as_any().downcast_ref::<Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity_id), Some(c1), Some(c2)) =
            (opt_entity, &components1[i], &components2[i])
          {
            f(*entity_id, c1, c2);
          }
        }
      }
    }
  }

  // === Advanced Scene Traversal Logic ===

  /// TODO: Document this item
  pub fn traverse_dfs_pre_order<A, F, C>(
    &self,
    start_entity: EntityId,
    accumulator: &mut A,
    filter: &F,
    callback: &mut C,
  ) where
    F: Fn(&Scene, EntityId) -> bool,
    C: FnMut(&Scene, EntityId, &mut A) -> bool,
  {
    if !self.entities.read().contains_key(start_entity) {
      return;
    }
    let mut visited = HashSet::new();
    self.traverse_recursive(start_entity, accumulator, filter, callback, &mut visited);
  }

  fn traverse_recursive<A, F, C>(
    &self,
    current_entity: EntityId,
    accumulator: &mut A,
    filter: &F,
    callback: &mut C,
    visited: &mut HashSet<EntityId>,
  ) where
    F: Fn(&Scene, EntityId) -> bool,
    C: FnMut(&Scene, EntityId, &mut A) -> bool,
  {
    if !visited.insert(current_entity) {
      return;
    }
    if !filter(self, current_entity) {
      return;
    }
    if !callback(self, current_entity, accumulator) {
      return;
    }

    let hierarchy = self.hierarchy.read();
    if let Some(children) = hierarchy.children.get(&current_entity) {
      let children_clone = children.clone();
      drop(hierarchy);
      for &child in &children_clone {
        self.traverse_recursive(child, accumulator, filter, callback, visited);
      }
    }
  }

  /// TODO: Document this item
  pub fn traverse_with_hooks<A, Pre, Post, T>(
    &self,
    start_entity: EntityId,
    accumulator: &mut A,
    pre_visit: &mut Pre,
    post_visit: &mut Post,
  ) where
    Pre: FnMut(&mut A, EntityId, Option<TransformComponent>, Option<&T>) -> bool,
    Post: FnMut(&mut A, EntityId),
    T: Component,
  {
    if !self.entities.read().contains_key(start_entity) {
      return;
    }
    let mut visited = HashSet::new();
    self.traverse_with_hooks_recursive(
      start_entity,
      accumulator,
      pre_visit,
      post_visit,
      &mut visited,
    );
  }

  fn traverse_with_hooks_recursive<A, Pre, Post, T>(
    &self,
    current_entity: EntityId,
    accumulator: &mut A,
    pre_visit: &mut Pre,
    post_visit: &mut Post,
    visited: &mut HashSet<EntityId>,
  ) where
    Pre: FnMut(&mut A, EntityId, Option<TransformComponent>, Option<&T>) -> bool,
    Post: FnMut(&mut A, EntityId),
    T: Component,
  {
    if !visited.insert(current_entity) {
      return;
    }

    let transform = self.with_component(current_entity, |c: &TransformComponent| *c);
    let mesh_ptr = self.with_component(current_entity, |c: &T| c as *const _);

    let continue_traversal = unsafe {
      let mesh_ref = if let Some(ptr) = mesh_ptr {
        Some(&*ptr)
      } else {
        None
      };
      pre_visit(accumulator, current_entity, transform, mesh_ref)
    };

    if !continue_traversal {
      post_visit(accumulator, current_entity);
      return;
    }

    let children_clone = {
      let hierarchy = self.hierarchy.read();
      hierarchy.children.get(&current_entity).cloned()
    };

    if let Some(children) = children_clone {
      for &child in &children {
        self.traverse_with_hooks_recursive(child, accumulator, pre_visit, post_visit, visited);
      }
    }

    post_visit(accumulator, current_entity);
  }

  /// TODO: Document this item
  pub fn global_transform(&self, entity_id: EntityId) -> Option<TransformComponent> {
    let mut accumulated_transform = self.with_component(entity_id, |c: &TransformComponent| *c)?;
    let mut current_entity = entity_id;

    loop {
      let parent_opt = {
        let hierarchy = self.hierarchy.read();
        hierarchy.parents.get(&current_entity).copied()
      };

      if let Some(parent_id) = parent_opt {
        if let Some(mut parent_transform) =
          self.with_component(parent_id, |c: &TransformComponent| *c)
        {
          let mut frame_scale = 1.0;
          let _ = self.with_component(parent_id, |c: &ReferenceFrameComponent| {
            frame_scale = c.scale;
          });
          parent_transform.scale = parent_transform.scale * frame_scale;

          accumulated_transform =
            Self::combine_transforms(&parent_transform, &accumulated_transform);
        }
        current_entity = parent_id;
      } else {
        break;
      }
    }

    Some(accumulated_transform)
  }

  /// TODO: Document this item
  pub fn get_relative_transform(
    &self,
    target_entity: EntityId,
    reference_entity: EntityId,
  ) -> Option<TransformComponent> {
    if target_entity == reference_entity {
      return Some(TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: self
          .global_transform(target_entity)
          .map(|t| t.rotation)
          .unwrap_or(Quat::identity()),
        scale: self
          .global_transform(target_entity)
          .map(|t| t.scale)
          .unwrap_or(Vec3f32::from_components(1.0, 1.0, 1.0)),
      });
    }

    let mut target_path = Vec::new();
    let mut curr = target_entity;
    target_path.push(curr);
    while let Some(parent) = self.get_parent(curr) {
      curr = parent;
      target_path.push(curr);
    }

    let mut ref_path = Vec::new();
    let mut curr = reference_entity;
    ref_path.push(curr);
    while let Some(parent) = self.get_parent(curr) {
      curr = parent;
      ref_path.push(curr);
    }

    let mut lca = None;
    let mut t_idx = target_path.len() as isize - 1;
    let mut r_idx = ref_path.len() as isize - 1;

    while t_idx >= 0 && r_idx >= 0 && target_path[t_idx as usize] == ref_path[r_idx as usize] {
      lca = Some(target_path[t_idx as usize]);
      t_idx -= 1;
      r_idx -= 1;
    }

    if lca.is_none() {
      // Fallback to old behavior: absolute global transforms
      let target_global = self.global_transform(target_entity)?;
      let ref_global = self.global_transform(reference_entity)?;

      let safe_div_zero = |a: f32, b: f32| {
        if b > -1e-15_f32 && b < 1e-15_f32 {
          0.0
        } else {
          a / b
        }
      };

      return Some(TransformComponent {
        scale: Vec3f32::from_components(
          safe_div_zero(target_global.scale.x(), ref_global.scale.x()),
          safe_div_zero(target_global.scale.y(), ref_global.scale.y()),
          safe_div_zero(target_global.scale.z(), ref_global.scale.z()),
        ),
        rotation: target_global.rotation, // World rotation
        position: target_global.position - ref_global.position, // RTE World Position
      });
    }

    let mut target_to_lca = TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    };

    if t_idx >= 0 {
      for i in 0..=(t_idx as usize) {
        let node_id = target_path[i];
        let node_transform = self.with_component(node_id, |c: &TransformComponent| *c)?;

        let mut frame_scale = 1.0;
        let _ = self.with_component(node_id, |c: &ReferenceFrameComponent| {
          frame_scale = c.scale;
        });

        let scaled_child_pos = target_to_lca.position * frame_scale;
        let scaled_child_scale = target_to_lca.scale * frame_scale;

        target_to_lca = TransformComponent {
          scale: node_transform.scale * scaled_child_scale,
          rotation: node_transform.rotation * target_to_lca.rotation,
          position: node_transform.position
            + node_transform.rotation.rotate_vector(node_transform.scale * scaled_child_pos),
        };
      }
    }

    let mut ref_to_lca = TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    };

    if r_idx >= 0 {
      for i in 0..=(r_idx as usize) {
        let node_id = ref_path[i];
        let node_transform = self.with_component(node_id, |c: &TransformComponent| *c)?;

        let mut frame_scale = 1.0;
        let _ = self.with_component(node_id, |c: &ReferenceFrameComponent| {
          frame_scale = c.scale;
        });

        let scaled_child_pos = ref_to_lca.position * frame_scale;
        let scaled_child_scale = ref_to_lca.scale * frame_scale;

        ref_to_lca = TransformComponent {
          scale: node_transform.scale * scaled_child_scale,
          rotation: node_transform.rotation * ref_to_lca.rotation,
          position: node_transform.position
            + node_transform.rotation.rotate_vector(node_transform.scale * scaled_child_pos),
        };
      }
    }

    let safe_div_zero = |a: f32, b: f32| {
      if b > -1e-15_f32 && b < 1e-15_f32 {
        0.0
      } else {
        a / b
      }
    };

    let diff_pos = target_to_lca.position - ref_to_lca.position;

    // RTE Approach: Keep the orientation in the world (LCA) space!
    // Do NOT unrotate it by the camera's rotation. The Camera View Matrix will handle rotation.
    Some(TransformComponent {
      scale: Vec3f32::from_components(
        safe_div_zero(target_to_lca.scale.x(), ref_to_lca.scale.x()),
        safe_div_zero(target_to_lca.scale.y(), ref_to_lca.scale.y()),
        safe_div_zero(target_to_lca.scale.z(), ref_to_lca.scale.z()),
      ),
      rotation: target_to_lca.rotation,
      position: Vec3f32::from_components(
        safe_div_zero(diff_pos.x(), ref_to_lca.scale.x()),
        safe_div_zero(diff_pos.y(), ref_to_lca.scale.y()),
        safe_div_zero(diff_pos.z(), ref_to_lca.scale.z()),
      ),
    })
  }

  fn combine_transforms(
    parent: &TransformComponent,
    child: &TransformComponent,
  ) -> TransformComponent {
    TransformComponent {
      scale: parent.scale * child.scale,
      rotation: parent.rotation * child.rotation,
      position: parent.position + (parent.rotation.rotate_vector(parent.scale * child.position)),
    }
  }

  /// TODO: Document this item
  pub fn parent_global_transform(&self, entity_id: EntityId) -> Option<TransformComponent> {
    let mut current_parent = self.get_parent(entity_id);
    while let Some(parent_id) = current_parent {
      if let Some(pg) = self.global_transform(parent_id) {
        return Some(pg);
      }
      current_parent = self.get_parent(parent_id);
    }
    None
  }

  /// TODO: Document this item
  pub fn set_global_transform(
    &self,
    entity_id: EntityId,
    new_global: TransformComponent,
  ) -> EngineResult<()> {
    let parent_global = self.parent_global_transform(entity_id);

    self
      .with_component_mut(entity_id, |t: &mut TransformComponent| {
        if let Some(pg) = parent_global {
          t.scale = Vec3f32::from_components(
            safe_div(new_global.scale.x(), pg.scale.x()),
            safe_div(new_global.scale.y(), pg.scale.y()),
            safe_div(new_global.scale.z(), pg.scale.z()),
          );

          let inv_rot = pg.rotation.inverse();
          t.rotation = inv_rot * new_global.rotation;

          let diff_pos = Vec3f32::from_components(
            new_global.position.x() - pg.position.x(),
            new_global.position.y() - pg.position.y(),
            new_global.position.z() - pg.position.z(),
          );
          let unrotated_diff = inv_rot.rotate_vector(diff_pos);

          t.position = Vec3f32::from_components(
            safe_div(unrotated_diff.x(), pg.scale.x()),
            safe_div(unrotated_diff.y(), pg.scale.y()),
            safe_div(unrotated_diff.z(), pg.scale.z()),
          );
        } else {
          *t = new_global;
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "set_global_transform: Entity missing TransformComponent",
      ))?;

    Ok(())
  }

  /// TODO: Document this item
  pub fn set_global_position_and_rotation(
    &self,
    entity_id: EntityId,
    new_position: Vec3f32,
    new_rotation: Quat,
  ) -> EngineResult<()> {
    let parent_global = self.parent_global_transform(entity_id);

    self
      .with_component_mut(entity_id, |t: &mut TransformComponent| {
        if let Some(pg) = parent_global {
          let safe_div_zero = |a: f32, b: f32| {
            if b > -1e-15_f32 && b < 1e-15_f32 {
              0.0
            } else {
              a / b
            }
          };

          let inv_rot = pg.rotation.inverse();
          t.rotation = inv_rot * new_rotation;

          let diff_pos = Vec3f32::from_components(
            new_position.x() - pg.position.x(),
            new_position.y() - pg.position.y(),
            new_position.z() - pg.position.z(),
          );
          let unrotated_diff = inv_rot.rotate_vector(diff_pos);

          t.position = Vec3f32::from_components(
            safe_div_zero(unrotated_diff.x(), pg.scale.x()),
            safe_div_zero(unrotated_diff.y(), pg.scale.y()),
            safe_div_zero(unrotated_diff.z(), pg.scale.z()),
          );
        } else {
          t.position = new_position;
          t.rotation = new_rotation;
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "set_global_position_and_rotation: Entity missing TransformComponent",
      ))?;

    Ok(())
  }

  /// TODO: Document this item
  pub fn validate(&self) -> EngineResult<()> {
    let archetypes = self.archetypes.read();
    let sun_type = TypeId::of::<SunComponent>();
    let cursor_type = TypeId::of::<CursorComponent>();
    let sky_type = TypeId::of::<SkyComponent>();
    let grid_type = TypeId::of::<GridComponent>();

    let mut sun_count = 0;
    let mut cursor_count = 0;
    let mut sky_count = 0;
    let mut grid_count = 0;

    for arch in archetypes.iter() {
      let alive_count = arch.entities.len() - arch.free_slots.len();
      if alive_count == 0 {
        continue;
      }

      if arch.component_types.contains(&sun_type) {
        sun_count += alive_count;
      }
      if arch.component_types.contains(&cursor_type) {
        cursor_count += alive_count;
      }
      if arch.component_types.contains(&sky_type) {
        sky_count += alive_count;
      }
      if arch.component_types.contains(&grid_type) {
        grid_count += alive_count;
      }
    }

    if sun_count > 1 {
      return Err(EngineError::InvalidOperation("multiple SunComponent"));
    }
    if cursor_count > 1 {
      return Err(EngineError::InvalidOperation("multiple CursorComponent"));
    }
    if sky_count > 1 {
      return Err(EngineError::InvalidOperation("multiple SkyComponent"));
    }
    if grid_count > 1 {
      return Err(EngineError::InvalidOperation("multiple GridComponent"));
    }

    Ok(())
  }

  // === Parallel Load-Balanced Query Execution Engine ===

  #[inline(always)]
  /// TODO: Document this item
  pub fn iter_par_archetypes<F>(&self, pool: &ThreadPool, f: F)
  where
    F: Fn(usize, &Archetype) + Send + Sync,
  {
    let archetypes = self.archetypes.read();
    let num_archetypes = archetypes.len();

    if num_archetypes == 0 {
      return;
    }

    let archetypes_ptr = ErasedPtr::new(archetypes.as_ptr());
    let f_ptr = ErasedPtr::new(&f);

    let handle_res = pool.spawn_chunked(num_archetypes, move |chunk_id| {
      let archetype_slice =
        unsafe { core::slice::from_raw_parts(archetypes_ptr.get::<Archetype>(), num_archetypes) };
      let archetype = &archetype_slice[chunk_id];
      let func = unsafe { &*f_ptr.get::<F>() };
      func(chunk_id, archetype);
    });

    if let Ok(handle) = handle_res {
      handle.wait();
    } else {
      for (i, archetype) in archetypes.iter().enumerate() {
        f(i, archetype);
      }
    }
  }

  /// TODO: Document this item
  pub fn query1_first_res_par<T: Component, F, R>(
    &self,
    pool: &ThreadPool,
    f: F,
  ) -> Option<(R, EntityId)>
  where
    F: Fn(EntityId, &T) -> Option<R> + Send + Sync,
    R: Send,
  {
    let type_t = core::any::TypeId::of::<T>();
    let found = core::sync::atomic::AtomicBool::new(false);
    let result: spin::Mutex<Option<(R, EntityId)>> = spin::Mutex::new(None);

    let found_ptr = ErasedPtr::new(&found);
    let result_ptr = ErasedPtr::new(&result);

    self.iter_par_archetypes(pool, |_, archetype| {
      let found_ref = unsafe { &*found_ptr.get::<core::sync::atomic::AtomicBool>() };
      if found_ref.load(core::sync::atomic::Ordering::Relaxed) {
        return;
      }

      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components_vec) =
          comp_storage.as_any().downcast_ref::<alloc::vec::Vec<Option<T>>>()
        {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if i % 16 == 0 && found_ref.load(core::sync::atomic::Ordering::Relaxed) {
              return;
            }
            if let (Some(entity), Some(comp)) = (opt_entity, &components_vec[i]) {
              if let Some(res) = f(*entity, comp) {
                let result_mut =
                  unsafe { &*result_ptr.get::<spin::Mutex<Option<(R, EntityId)>>>() };
                let mut lock = result_mut.lock();
                if lock.is_none() {
                  *lock = Some((res, *entity));
                  found_ref.store(true, core::sync::atomic::Ordering::Relaxed);
                }
                return;
              }
            }
          }
        }
      }
    });

    result.into_inner()
  }

  /// TODO: Document this item
  pub fn query1_res_par<T: Component, F, R>(
    &self,
    pool: &ThreadPool,
    f: F,
  ) -> alloc::vec::Vec<(R, EntityId)>
  where
    F: Fn(EntityId, &T) -> Option<R> + Send + Sync,
    R: Send,
  {
    let type_t = core::any::TypeId::of::<T>();
    let num_archetypes = self.archetypes.read().len();

    if num_archetypes == 0 {
      return alloc::vec::Vec::new();
    }

    let mut chunk_results: alloc::vec::Vec<alloc::vec::Vec<(R, EntityId)>> =
      core::iter::repeat_with(alloc::vec::Vec::new).take(num_archetypes).collect();

    let results_ptr = ErasedMutPtr::new(chunk_results.as_mut_ptr());

    self.iter_par_archetypes(pool, |arch_id, archetype| {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components_vec) =
          comp_storage.as_any().downcast_ref::<alloc::vec::Vec<Option<T>>>()
        {
          let result_vec =
            unsafe { &mut *results_ptr.get::<alloc::vec::Vec<(R, EntityId)>>().add(arch_id) };
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity), Some(comp)) = (opt_entity, &components_vec[i]) {
              if let Some(res) = f(*entity, comp) {
                result_vec.push((res, *entity));
              }
            }
          }
        }
      }
    });

    let mut final_results = alloc::vec::Vec::new();
    for mut res in chunk_results {
      final_results.append(&mut res);
    }
    final_results
  }

  /// TODO: Document this item
  pub fn query2_res_par<T1: Component, T2: Component, F, R>(
    &self,
    pool: &ThreadPool,
    f: F,
  ) -> alloc::vec::Vec<(R, EntityId)>
  where
    F: Fn(EntityId, &T1, &T2) -> Option<R> + Send + Sync,
    R: Send,
  {
    let type_t1 = core::any::TypeId::of::<T1>();
    let type_t2 = core::any::TypeId::of::<T2>();

    assert_ne!(
      type_t1, type_t2,
      "Cannot query the same component type twice in a single pass."
    );

    let num_archetypes = self.archetypes.read().len();

    if num_archetypes == 0 {
      return alloc::vec::Vec::new();
    }

    let mut chunk_results: alloc::vec::Vec<alloc::vec::Vec<(R, EntityId)>> =
      core::iter::repeat_with(alloc::vec::Vec::new).take(num_archetypes).collect();

    let results_ptr = ErasedMutPtr::new(chunk_results.as_mut_ptr());

    self.iter_par_archetypes(pool, |arch_id, archetype| {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let comp_storage1_lock = archetype.components.get(&type_t1).unwrap();
        let comp_storage2_lock = archetype.components.get(&type_t2).unwrap();

        let comp_storage1 = comp_storage1_lock.read();
        let comp_storage2 = comp_storage2_lock.read();

        if let (Some(components_vec1), Some(components_vec2)) = (
          comp_storage1.as_any().downcast_ref::<alloc::vec::Vec<Option<T1>>>(),
          comp_storage2.as_any().downcast_ref::<alloc::vec::Vec<Option<T2>>>(),
        ) {
          let result_vec =
            unsafe { &mut *results_ptr.get::<alloc::vec::Vec<(R, EntityId)>>().add(arch_id) };

          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity), Some(c1), Some(c2)) =
              (opt_entity, &components_vec1[i], &components_vec2[i])
            {
              if let Some(res) = f(*entity, c1, c2) {
                result_vec.push((res, *entity));
              }
            }
          }
        }
      }
    });

    let mut final_results = alloc::vec::Vec::new();
    for mut res in chunk_results {
      final_results.append(&mut res);
    }
    final_results
  }

  /// TODO: Document this item
  pub fn query1_mut_par<T: Component, F>(&self, pool: &ThreadPool, f: F)
  where
    F: Fn(EntityId, &mut T) + Send + Sync,
  {
    let type_t = core::any::TypeId::of::<T>();

    self.iter_par_archetypes(pool, |_, archetype| {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components_vec) =
          comp_storage.as_mut_any().downcast_mut::<alloc::vec::Vec<Option<T>>>()
        {
          for (i, opt_entity) in archetype.entities.iter().enumerate() {
            if let (Some(entity), Some(comp)) = (opt_entity, &mut components_vec[i]) {
              f(*entity, comp);
            }
          }
        }
      }
    });
  }

  /// TODO: Document this item
  pub fn query2_mut_par<T1: Component, T2: Component, F>(&self, pool: &ThreadPool, f: F)
  where
    F: Fn(EntityId, &mut T1, &mut T2) + Send + Sync,
  {
    let type_t1 = core::any::TypeId::of::<T1>();
    let type_t2 = core::any::TypeId::of::<T2>();
    assert_ne!(
      type_t1, type_t2,
      "Cannot mutably query the same component type twice."
    );

    self.iter_par_archetypes(pool, |_, archetype| {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1 = archetype.components.get(&type_t1).unwrap().write();
        let mut comp_storage2 = archetype.components.get(&type_t2).unwrap().write();

        let components_vec1 =
          comp_storage1.as_mut_any().downcast_mut::<alloc::vec::Vec<Option<T1>>>().unwrap();
        let components_vec2 =
          comp_storage2.as_mut_any().downcast_mut::<alloc::vec::Vec<Option<T2>>>().unwrap();

        for (i, opt_entity) in archetype.entities.iter().enumerate() {
          if let (Some(entity), Some(c1), Some(c2)) =
            (opt_entity, &mut components_vec1[i], &mut components_vec2[i])
          {
            f(*entity, c1, c2);
          }
        }
      }
    });
  }

  // ------------------------------------------------------------------------------------------------

  /// TODO: Document this item
  pub fn query1_res_without_par<T: Component, U: Component, F, R>(
    &self,
    pool: &ThreadPool,
    f: F,
  ) -> alloc::vec::Vec<(R, EntityId)>
  where
    F: Fn(EntityId, &T) -> Option<R> + Send + Sync,
    R: Send,
  {
    let type_t = core::any::TypeId::of::<T>();
    let type_u = core::any::TypeId::of::<U>();
    assert_ne!(
      type_t, type_u,
      "Included and excluded component types must be distinct."
    );

    let num_archetypes = self.archetypes.read().len();

    if num_archetypes == 0 {
      return alloc::vec::Vec::new();
    }

    let mut chunk_results: alloc::vec::Vec<alloc::vec::Vec<(R, EntityId)>> =
      core::iter::repeat_with(alloc::vec::Vec::new).take(num_archetypes).collect();

    let results_ptr = crate::scene::ErasedMutPtr::new(chunk_results.as_mut_ptr());

    self.iter_par_archetypes(pool, |arch_id, archetype| {
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap();
        let comp_storage = comp_storage_lock.read();
        if let Some(components_vec) = comp_storage.as_any().downcast_ref::<alloc::vec::Vec<T>>() {
          let result_vec =
            unsafe { &mut *results_ptr.get::<alloc::vec::Vec<(R, EntityId)>>().add(arch_id) };

          for (i, &entity) in archetype.entities.iter().enumerate() {
            if let Some(ent) = entity {
              if let Some(res) = f(ent, &components_vec[i]) {
                result_vec.push((res, ent));
              }
            }
          }
        }
      }
    });

    let mut final_results = alloc::vec::Vec::new();
    for mut res in chunk_results {
      final_results.append(&mut res);
    }
    final_results
  }

  // === Archetype Iteration Abstractions ===

  /// Sequentially iterates over all archetypes.
  #[inline(always)]
  fn iter_archetypes<F>(&self, mut f: F)
  // TODO refactor sequential query function with this
  where
    F: FnMut(usize, &Archetype),
  {
    let archetypes = self.archetypes.read();
    for (i, archetype) in archetypes.iter().enumerate() {
      f(i, archetype);
    }
  }
}

/// Type-erased safe wrappers to transfer pointer provenance across thread boundaries.
/// By erasing `T` into `()`, we hide non-'static lifetimes from the ThreadPool's strict bounds.
/// Soundness is strictly guaranteed by the synchronous `handle.wait()` barrier.
#[derive(Clone, Copy)]
pub struct ErasedPtr(*const ());
unsafe impl Send for ErasedPtr {}
unsafe impl Sync for ErasedPtr {}

impl ErasedPtr {
  #[inline(always)]
  /// TODO: Document this item
  pub fn new<T>(ptr: *const T) -> Self {
    Self(ptr as *const ())
  }
  #[inline(always)]
  /// TODO: Document this item
  pub fn get<T>(self) -> *const T {
    self.0 as *const T
  }
}

#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct ErasedMutPtr(*mut ());
unsafe impl Send for ErasedMutPtr {}
unsafe impl Sync for ErasedMutPtr {}

impl ErasedMutPtr {
  #[inline(always)]
  /// TODO: Document this item
  pub fn new<T>(ptr: *mut T) -> Self {
    Self(ptr as *mut ())
  }
  #[inline(always)]
  /// TODO: Document this item
  pub fn get<T>(self) -> *mut T {
    self.0 as *mut T
  }
}

#[cfg(test)]
mod tests {
  extern crate std; // Allow std for testing harness environment

  use super::*;
  use alloc::format;
  use core::any::TypeId;

  // Replace with the exact path where your ThreadPool is exposed in your library
  use aethervk_oshal_rlib::os::pool::ThreadPool;

  // === Mock Components ===

  #[derive(Debug, Clone, PartialEq)]
  struct HealthComp {
    hp: i32,
  }
  impl Component for HealthComp {}

  #[derive(Debug, Clone, PartialEq)]
  struct VelocityComp {
    speed: i32,
  }
  impl Component for VelocityComp {}

  #[derive(Debug, Clone, PartialEq)]
  struct ShieldComp;
  impl Component for ShieldComp {}

  // Helper function to build a pre-registered testing scene
  fn setup_scene() -> Scene {
    let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_component::<HealthComp>(&[]);
    scene.register_component::<VelocityComp>(&[]);
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<SunComponent>(&[]);
    scene.register_component::<CursorComponent>(&[]);

    // Dependent components: ShieldComp REQUIRES HealthComp to be present first
    scene.register_component::<ShieldComp>(&[TypeId::of::<HealthComp>()]);

    scene
  }

  // === Tests ===

  #[test]
  fn test_entity_spawn_and_names() {
    let scene = setup_scene();

    let e1 = scene.spawn_entity("Player");
    let e2 = scene.spawn_entity("Player");
    let e3 = scene.spawn_entity("Enemy");

    // Proves string deduplication logic correctly appends index counters
    assert_eq!(scene.get_name(e1).unwrap(), "Player");
    assert_eq!(scene.get_name(e2).unwrap(), "Player_1");
    assert_eq!(scene.get_name(e3).unwrap(), "Enemy");

    // Proves reverse lookup works
    assert_eq!(scene.get_entity_by_name("Player"), Some(e1));
    assert_eq!(scene.get_entity_by_name("Player_1"), Some(e2));

    // Proves string renaming memory safety and old name liberation
    scene.set_name(e1, "Hero");
    assert_eq!(scene.get_name(e1).unwrap(), "Hero");
    assert_eq!(scene.get_entity_by_name("Player"), None);
  }

  #[test]
  fn test_add_and_remove_components() {
    let scene = setup_scene();
    let e = scene.spawn_entity("e1");

    assert_eq!(
      scene.has_component::<HealthComp>(e),
      HasComponentResultEnum::ComponentNotFound
    );

    // Tests Component Insert
    scene.add_component(e, HealthComp { hp: 100 }).unwrap();
    assert_eq!(
      scene.has_component::<HealthComp>(e),
      HasComponentResultEnum::EntityHasComponent
    );

    let val = scene.with_component(e, |h: &HealthComp| h.hp).unwrap();
    assert_eq!(val, 100);

    // Tests single Component Mutate safely without moving Archetypes
    scene.with_component_mut(e, |h: &mut HealthComp| h.hp = 50);
    assert_eq!(scene.with_component(e, |h: &HealthComp| h.hp), Some(50));

    // Adding second component shifts memory correctly into a composite Archetype
    scene.add_component(e, VelocityComp { speed: 10 }).unwrap();
    assert_eq!(
      scene.has_component::<VelocityComp>(e),
      HasComponentResultEnum::EntityHasComponent
    );

    // Removing component cleanly shifts out into a different Archetype map entirely
    scene.remove_component::<HealthComp>(e).unwrap();
    assert_eq!(
      scene.has_component::<HealthComp>(e),
      HasComponentResultEnum::ComponentNotFound
    );
    assert_eq!(
      scene.has_component::<VelocityComp>(e),
      HasComponentResultEnum::EntityHasComponent
    );
  }

  #[test]
  fn test_dependencies() {
    let scene = setup_scene();
    let e = scene.spawn_entity("e");

    // Trying to add `ShieldComp` which demands `HealthComp` first should throw an error.
    let err = scene.add_component(e, ShieldComp);
    assert!(matches!(
      err,
      Err(AddComponentError::DependencyNotSatisfied { .. })
    ));

    // Fullfilling dependency works!
    scene.add_component(e, HealthComp { hp: 100 }).unwrap();
    assert!(scene.add_component(e, ShieldComp).is_ok());
  }

  #[test]
  fn test_entity_removal() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("e1");
    let e2 = scene.spawn_entity("e2");
    let e3 = scene.spawn_entity("e3");

    scene.add_component(e1, HealthComp { hp: 10 }).unwrap();
    scene.add_component(e2, HealthComp { hp: 20 }).unwrap();
    scene.add_component(e3, HealthComp { hp: 30 }).unwrap();

    // Ensures `swap_remove` in inner architecture does not pollute remaining rows locally
    scene.remove_entity(e2);

    assert_eq!(
      scene.has_component::<HealthComp>(e2),
      HasComponentResultEnum::EntityNotFound
    );
    assert_eq!(scene.get_name(e2), None);
    assert_eq!(scene.with_component(e1, |h: &HealthComp| h.hp), Some(10));
    assert_eq!(scene.with_component(e3, |h: &HealthComp| h.hp), Some(30));
  }

  #[test]
  fn test_hierarchy() {
    let scene = setup_scene();
    let parent = scene.spawn_entity("Parent");
    let child = scene.spawn_entity("Child");

    scene.set_parent(child, Some(parent));
    assert_eq!(scene.get_parent(child), Some(parent));

    // Entity removal recursively tears down parent maps mapping
    scene.remove_entity(parent);
    assert_eq!(scene.get_parent(child), None);
  }

  #[test]
  fn test_scene_validation() {
    let scene = setup_scene();
    assert!(scene.validate().is_ok());

    let sun1 = scene.spawn_entity("sun1");
    scene
      .add_component(
        sun1,
        SunComponent {
          resolution: (1, 1, 1),
          radius: 0.6,
        },
      )
      .unwrap();
    assert!(scene.validate().is_ok());

    let sun2 = scene.spawn_entity("sun2");
    scene
      .add_component(
        sun2,
        SunComponent {
          resolution: (1, 1, 1),
          radius: 0.6,
        },
      )
      .unwrap();

    // Scene throws if multiple 1-instance components are found (Renderer restrictions on the scene graph)
    assert!(scene.validate().is_err());
  }

  #[test]
  fn test_sequential_queries() {
    let scene = setup_scene();
    for i in 0..5 {
      let e = scene.spawn_entity(format!("e{}", i));
      scene.add_component(e, HealthComp { hp: 10 }).unwrap();
      if i % 2 == 0 {
        scene.add_component(e, VelocityComp { speed: 5 }).unwrap();
      }
    }

    let mut sum_hp = 0;
    scene.query1(|_e, h: &HealthComp| sum_hp += h.hp);
    assert_eq!(sum_hp, 50);

    let mut count_both = 0;
    scene.query2(|_e, _h: &HealthComp, _v: &VelocityComp| count_both += 1);
    assert_eq!(count_both, 3); // 0, 2, 4

    let mut without_vel = 0;
    scene.query1_without::<HealthComp, VelocityComp, _>(|_e, _h| without_vel += 1);
    assert_eq!(without_vel, 2); // 1, 3

    scene.query1_mut(|_e, h: &mut HealthComp| h.hp += 5);
    let first = scene.query1_first_res(|_e, h: &HealthComp| Some(h.hp)).unwrap();
    assert_eq!(first.0, 15);
  }

  // === Parallel Load-Balancing Execution Tests ===

  #[test]
  fn test_parallel_queries() {
    let scene = setup_scene();
    let pool = ThreadPool::new(4).unwrap();

    // Spawning 5000 entities to guarantee we utilize multiple archetype memory chunks
    for i in 0..5000 {
      let e = scene.spawn_entity(format!("e{}", i));
      scene.add_component(e, HealthComp { hp: 100 }).unwrap();

      // By staggering logic heavily we create entirely isolated memory layouts which tests
      // parallel thread dispersion on `iter_par_archetypes` across hardware queues efficiently!
      if i % 2 == 0 {
        scene.add_component(e, VelocityComp { speed: 10 }).unwrap();
      }
      if i % 3 == 0 {
        scene.add_component(e, ShieldComp).unwrap();
      }
    }

    // 1. Parallel Mutable Map
    scene.query1_mut_par(&pool, |_e, h: &mut HealthComp| {
      h.hp -= 10; // All drop to 90 internally lock-free
    });

    // 2. Parallel Result Gathering (Lock-Free Thread Array Condensing)
    let results = scene.query1_res_par(
      &pool,
      |_e, h: &HealthComp| {
        if h.hp == 90 { Some(h.hp) } else { None }
      },
    );
    assert_eq!(results.len(), 5000);

    // 3. Parallel Dual Mutate Map
    scene.query2_mut_par(&pool, |_e, h: &mut HealthComp, v: &mut VelocityComp| {
      h.hp -= v.speed;
    });

    // Validating OS-managed memory mutations were operated on successfully across workers
    let mut count_80 = 0;
    scene.query1(|_e, h: &HealthComp| {
      if h.hp == 80 {
        count_80 += 1;
      }
    });
    assert_eq!(count_80, 2500); // Only even entities took velocity damage

    // 4. Parallel First Find
    let first_res = scene.query1_first_res_par(&pool, |e, h: &HealthComp| {
      if scene.get_name(e).unwrap() == "e500" {
        Some(h.hp)
      } else {
        None
      }
    });
    assert!(first_res.is_some());
    assert_eq!(first_res.unwrap().0, 80); // e500 is an even number!
  }

  #[test]
  fn test_entity_removal_with_compaction_holes() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("e1");
    let e2 = scene.spawn_entity("e2");
    let e3 = scene.spawn_entity("e3");

    scene.add_component(e1, HealthComp { hp: 10 }).unwrap();
    scene.add_component(e2, HealthComp { hp: 20 }).unwrap();
    scene.add_component(e3, HealthComp { hp: 30 }).unwrap();

    // The core requested fix. Deleting an entity only frees the slot natively,
    // it DOES NOT blindly shift the array anymore!
    scene.remove_entity(e2);

    assert_eq!(
      scene.has_component::<HealthComp>(e2),
      HasComponentResultEnum::EntityNotFound
    );
    assert_eq!(scene.get_name(e2), None);
    assert_eq!(scene.with_component(e1, |h: &HealthComp| h.hp), Some(10));
    assert_eq!(scene.with_component(e3, |h: &HealthComp| h.hp), Some(30)); // 3 remains absolutely perfect!

    // Validate a new entity beautifully slips back into the hole in Memory
    let e4 = scene.spawn_entity("e4");
    scene.add_component(e4, HealthComp { hp: 40 }).unwrap();
    assert_eq!(scene.with_component(e4, |h: &HealthComp| h.hp), Some(40));
  }

  #[test]
  fn test_remove_column_bulk() {
    let scene = setup_scene();
    for _ in 0..100 {
      let e = scene.spawn_entity("Ent");
      scene.add_component(e, HealthComp { hp: 100 }).unwrap();
      scene.add_component(e, VelocityComp { speed: 5 }).unwrap();
    }

    // Instantly delete Health component without moving any Memory/Entities
    scene.remove_column::<HealthComp>();

    let mut health_count = 0;
    scene.query1(|_e, _h: &HealthComp| health_count += 1);
    assert_eq!(health_count, 0);

    let mut vel_count = 0;
    scene.query1(|_e, _v: &VelocityComp| vel_count += 1);
    assert_eq!(vel_count, 100);
  }

  #[test]
  fn test_compaction() {
    let scene = setup_scene();
    let mut ents = Vec::new();

    // Spawn enough to surpass threshold (>64 elements)
    for _ in 0..100 {
      let e = scene.spawn_entity("Comp");
      scene.add_component(e, HealthComp { hp: 1 }).unwrap();
      ents.push(e);
    }

    // Remove 50% randomly, crossing the >25% hole-count Threshold limit
    for i in 0..50 {
      scene.remove_entity(ents[i]);
    }

    // All active entities should still execute properly, compacted sequentially backwards!
    let mut sum = 0;
    scene.query1(|_e, h: &HealthComp| sum += h.hp);
    assert_eq!(sum, 50);
  }

  #[test]
  fn test_relative_transform() {
    let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<ReferenceFrameComponent>(&[]);

    let macro_frame = scene.spawn_entity("macro_frame");
    scene
      .add_component(
        macro_frame,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        macro_frame,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Macro,
          scale: 1.0,
          soi_radius: core::f32::MAX,
          _padding: 0,
        },
      )
      .unwrap();

    let planet = scene.spawn_entity("planet");
    scene
      .add_component(
        planet,
        TransformComponent {
          position: Vec3f32::from_components(1.0, 0.0, 0.0), // 1 AU away
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    // Assuming a smaller scale for test to avoid f32 precision loss
    let au_to_km = 100_000.0_f32;
    scene
      .add_component(
        planet,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 1.0 / au_to_km, // 1 km in AU
          soi_radius: 1000000.0,
          _padding: 0,
        },
      )
      .unwrap();
    scene.set_parent(planet, Some(macro_frame));

    let camera = scene.spawn_entity("camera");
    scene
      .add_component(
        camera,
        TransformComponent {
          position: Vec3f32::from_components(1000.0, 0.0, 0.0), // 1000 km away from planet center
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.set_parent(camera, Some(planet));

    let sun = scene.spawn_entity("sun");
    scene
      .add_component(
        sun,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0), // at origin of macro frame
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.set_parent(sun, Some(macro_frame));

    // Get transform of sun relative to camera
    // Camera is at global AU = 1.0 + (1000.0 / 149597870.7) = 1.00000668458
    // Sun is at global AU = 0.0
    // Relative position should be - (1.0 * au_to_km + 1000.0) km
    let rel_transform = scene.get_relative_transform(sun, camera).unwrap();

    // Test the output position in camera's frame (which is Micro -> km)
    // The relative transform calculates position in the target's coordinate space (camera), wait no,
    // It's the transform of `target_entity` (sun) relative to `reference_entity` (camera).
    // So the coordinate space is the reference entity's coordinate space (km).
    // Let's check diff
    let sun_pos_in_km = rel_transform.position;
    assert!(
      (sun_pos_in_km.x() - (-au_to_km - 1000.0)).abs() < 10.0,
      "Expected {}, got {}",
      -au_to_km - 1000.0,
      sun_pos_in_km.x()
    );
  }

  #[test]
  fn test_hidden_component_subtree() {
    let scene = Scene::new(alloc::sync::Arc::new(spin::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_component::<TransformComponent>(&[]);
    scene.register_component::<HiddenComponent>(&[]);
    let root = scene.spawn_entity("root");
    let child = scene.spawn_entity("child");
    scene.set_parent(child, Some(root));
    scene.add_component(root, HiddenComponent {}).unwrap();

    let mut hidden_set = hashbrown::HashSet::new();
    scene.query1::<HiddenComponent, _>(|id, _| {
      scene.traverse_dfs_pre_order(
        id,
        &mut hidden_set,
        &|_, _| true,
        &mut |_, child_id, set| {
          set.insert(child_id);
          true
        },
      );
    });

    assert!(hidden_set.contains(&child));
  }

  #[test]
  fn test_remove_column_if() {
    let scene = setup_scene();
    for i in 0..10 {
      let e = scene.spawn_entity("Ent");
      scene.add_component(e, HealthComp { hp: i }).unwrap();
    }
    scene.remove_column_if(|_, h: &HealthComp| h.hp % 2 == 0);

    let mut count = 0;
    scene.query1(|_, h: &HealthComp| {
      assert!(h.hp % 2 != 0);
      count += 1;
    });
    assert_eq!(count, 5);
  }

  #[test]
  fn test_remove_column_par() {
    let pool = ThreadPool::new(4).unwrap();
    let scene = setup_scene();
    for _ in 0..100 {
      let e = scene.spawn_entity("Ent");
      scene.add_component(e, HealthComp { hp: 100 }).unwrap();
      scene.add_component(e, VelocityComp { speed: 5 }).unwrap();
    }

    scene.remove_column_par::<HealthComp>(&pool);

    let mut health_count = 0;
    scene.query1(|_e, _h: &HealthComp| health_count += 1);
    assert_eq!(health_count, 0);

    let mut vel_count = 0;
    scene.query1(|_e, _v: &VelocityComp| vel_count += 1);
    assert_eq!(vel_count, 100);
  }

  #[test]
  fn test_remove_column_if_par() {
    let pool = ThreadPool::new(4).unwrap();
    let scene = setup_scene();
    for i in 0..100 {
      let e = scene.spawn_entity("Ent");
      scene.add_component(e, HealthComp { hp: i }).unwrap();
    }

    scene.remove_column_if_par::<HealthComp, _>(&pool, |_, h| h.hp < 50);

    let mut health_count = 0;
    scene.query1(|_e, h: &HealthComp| {
      assert!(h.hp >= 50);
      health_count += 1;
    });
    assert_eq!(health_count, 50);
  }

  #[test]
  fn test_remove_first_component() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("Ent1");
    scene.add_component(e1, HealthComp { hp: 10 }).unwrap();
    let e2 = scene.spawn_entity("Ent2");
    scene.add_component(e2, HealthComp { hp: 20 }).unwrap();

    let removed = scene.remove_first_component::<HealthComp>();
    assert!(removed.is_some());
    let removed_id = removed.unwrap();

    let mut count = 0;
    scene.query1(|_e, _h: &HealthComp| count += 1);
    assert_eq!(count, 1);
    assert_eq!(
      scene.has_component::<HealthComp>(removed_id),
      HasComponentResultEnum::ComponentNotFound
    );
  }

  #[test]
  fn test_has_component() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("Ent1");
    scene.add_component(e1, HealthComp { hp: 10 }).unwrap();

    assert_eq!(
      scene.has_component::<HealthComp>(e1),
      HasComponentResultEnum::EntityHasComponent
    );
    assert_eq!(
      scene.has_component::<VelocityComp>(e1),
      HasComponentResultEnum::ComponentNotFound
    );

    scene.remove_entity(e1);
    assert_eq!(
      scene.has_component::<HealthComp>(e1),
      HasComponentResultEnum::EntityNotFound
    );
  }

  #[test]
  fn test_with_component() {
    let scene = setup_scene();
    let e1 = scene.spawn_entity("Ent1");
    scene.add_component(e1, HealthComp { hp: 42 }).unwrap();

    let val = scene.with_component(e1, |h: &HealthComp| h.hp);
    assert_eq!(val, Some(42));

    let none_val = scene.with_component(e1, |v: &VelocityComp| v.speed);
    assert_eq!(none_val, None);

    let val_mut = scene.with_component_mut(e1, |h: &mut HealthComp| {
      h.hp += 1;
      h.hp
    });
    assert_eq!(val_mut, Some(43));

    let val_after = scene.with_component(e1, |h: &HealthComp| h.hp);
    assert_eq!(val_after, Some(43));
  }
}
