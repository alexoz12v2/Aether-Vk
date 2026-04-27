//! Scene graph and Entity-Component-System (ECS) implementation.
//!
//! ## Design
//! - **Backend-agnostic:** The scene representation is independent of the rendering backend.
//! - **Thread-safe:** The main `Scene` struct will be designed for concurrent access (`Send + Sync`).
//! - **Archetype-based ECS:** Inspired by Bevy's architecture for efficient memory layout and querying.
//!   - Entities with the same set of components (an archetype) are stored together in contiguous memory.
//!   - This is a simplified implementation focusing on the core concepts.

// TODO add the cache class for meshes and billboard data as specified in simulation api

// TODO reduce duplicate code on query

// TODO add tests for new methods

pub mod text;

use crate::simulation::comet::Comet;
use aethervk_oshal_rlib::math::{safe_div, FloatLike};
use aethervk_oshal_rlib::math::matrix::Matrix4;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{Vector3, Vector4};
use aethervk_oshal_rlib::math::{matrix::mat4::Mat4x4f32, vector::vec4::Quat};
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use slotmap::{new_key_type, SlotMap};
use spin::RwLock;
use alloc::boxed::Box;
use alloc::vec::Vec;
use alloc::string::{String};
use core::any::{Any, TypeId};
use hashbrown::{HashMap, HashSet};

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
  pub fn new() -> Self {
    Self {
      assets: RwLock::new(HashMap::new()),
    }
  }

  pub fn get(&self, path: &str) -> Option<alloc::sync::Arc<T>> {
    self.assets.read().get(path).cloned()
  }

  pub fn insert(&self, path: String, asset: T) -> alloc::sync::Arc<T> {
    let mut map = self.assets.write();
    let arc = alloc::sync::Arc::new(asset);
    map.insert(path, arc.clone());
    arc
  }

  pub fn remove(&self, path: &str) -> Option<alloc::sync::Arc<T>> {
    self.assets.write().remove(path)
  }

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

// === Component Definitions ===

/// Defines the position, rotation, and scale of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransformComponent {
  pub position: Vec3f32,
  /// Stored as a quaternion.
  pub rotation: Quat,
  pub scale: Vec3f32,
}
impl Component for TransformComponent {}
impl TransformComponent {
  /// Constructs a 4x4 transformation matrix from the component's TRS
  /// (Translation, Rotation, Scale) properties.
  pub fn to_mat4<T>(&self) -> T
  where
    T: Matrix4,
    T::Vector: Vector4, // Requires Vector4 to use `from_components`
    T::Scalar: FloatLike,
  {
    let p = &self.position;
    let q = &self.rotation;
    let s = &self.scale;

    // Precompute quaternion products to avoid redundant multiplications
    let xx = q.0.x() * q.0.x();
    let yy = q.0.y() * q.0.y();
    let zz = q.0.z() * q.0.z();
    let xy = q.0.x() * q.0.y();
    let xz = q.0.x() * q.0.z();
    let yz = q.0.y() * q.0.z();
    let wx = q.0.w() * q.0.x();
    let wy = q.0.w() * q.0.y();
    let wz = q.0.w() * q.0.z();

    // Column 0 (Rotated & Scaled X-axis)
    let c0 = <T::Vector as Vector4>::from_components(
      <T::Scalar as FloatLike>::from_f32((1.0 - 2.0 * (yy + zz)) * s.x()),
      <T::Scalar as FloatLike>::from_f32((2.0 * (xy + wz)) * s.x()),
      <T::Scalar as FloatLike>::from_f32((2.0 * (xz - wy)) * s.x()),
      <T::Scalar as FloatLike>::from_f32(0.0),
    );

    // Column 1 (Rotated & Scaled Y-axis)
    let c1 = <T::Vector as Vector4>::from_components(
      <T::Scalar as FloatLike>::from_f32((2.0 * (xy - wz)) * s.y()),
      <T::Scalar as FloatLike>::from_f32((1.0 - 2.0 * (xx + zz)) * s.y()),
      <T::Scalar as FloatLike>::from_f32((2.0 * (yz + wx)) * s.y()),
      <T::Scalar as FloatLike>::from_f32(0.0),
    );

    // Column 2 (Rotated & Scaled Z-axis)
    let c2 = <T::Vector as Vector4>::from_components(
      <T::Scalar as FloatLike>::from_f32((2.0 * (xz + wy)) * s.z()),
      <T::Scalar as FloatLike>::from_f32((2.0 * (yz - wx)) * s.z()),
      <T::Scalar as FloatLike>::from_f32((1.0 - 2.0 * (xx + yy)) * s.z()),
      <T::Scalar as FloatLike>::from_f32(0.0),
    );

    // Column 3 (Translation)
    let c3 = <T::Vector as Vector4>::from_components(
      <T::Scalar as FloatLike>::from_f32(p.x()),
      <T::Scalar as FloatLike>::from_f32(p.y()),
      <T::Scalar as FloatLike>::from_f32(p.z()),
      <T::Scalar as FloatLike>::from_f32(1.0),
    );

    T::from_columns(c0, c1, c2, c3)
  }
}

/// Represents a camera in the scene.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CameraComponent {
  pub projection: Mat4x4f32,
  pub near_plane: f32,
  pub far_plane: f32,
}
impl Component for CameraComponent {}

/// A marker component for entities that should be rendered as a cursor.
#[derive(Debug, PartialEq)]
pub struct CursorComponent {}
impl Component for CursorComponent {}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Marker {
  pub local_pos: [f32; 3],
  pub color: [f32; 3],
  pub size: f32,
}

#[derive(Debug, PartialEq, Clone)]
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
}

impl Clone for PhysicalMeshComponent {
  fn clone(&self) -> Self {
    Self {
      asset_path: self.asset_path.clone(),
      mesh: self.mesh.clone(),
      emissive_intensity: self.emissive_intensity,
      emissive_color: self.emissive_color,
    }
  }
}

impl PartialEq for PhysicalMeshComponent {
  fn eq(&self, other: &Self) -> bool {
    self.asset_path == other.asset_path
      && self.emissive_intensity == other.emissive_intensity
      && self.emissive_color == other.emissive_color
  }
}

impl Component for PhysicalMeshComponent {}

/// A marker component for entities that should be rendered.
#[derive(Debug)]
pub struct Renderable;
impl Component for Renderable {}

/// Represents a 2D texture billboard.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BillboardType {
  WorldSpace { width: f32, height: f32 },
  ScreenSpace { pct_width: f32, pct_height: f32 },
}

#[derive(Debug)]
pub struct ImageBillboardComponent {
  pub texture_id: u64,
  pub billboard_type: BillboardType,
}
impl Component for ImageBillboardComponent {}

/// Tags an entity as a Renderable Sun
#[derive(Clone, Copy, Debug)]
pub struct SunComponent {
  pub resolution: (u32, u32, u32),
}
impl Component for SunComponent {}

/// A marker component for entities that should be rendered as a background sky.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SkyComponent {}
impl Component for SkyComponent {}

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

/// A component that stores debug render states for BVH nodes
#[derive(Clone, Debug)]
pub struct BvhDebugComponent {
  pub node_render_states: alloc::vec::Vec<bool>,
}
impl Component for BvhDebugComponent {}

/// A measurement line between two points with an associated distance.
#[derive(Clone, PartialEq, Debug)]
pub struct MeasurementComponent {
  pub pos1: Vec3f32,
  pub pos2: Vec3f32,
  /// Size in PTs for character rendering
  pub points: f32,
}
impl Component for MeasurementComponent {}

/// A particle emitter, defining the properties of particles to be spawned.
#[derive(Clone, Debug)]
pub struct ParticleEmitterComponent {
  /// Number of particles to spawn per second.
  pub rate: f32,
  /// The lifetime of each particle, in seconds.
  pub lifetime: f32,
  /// Initial velocity of spawned particles.
  pub initial_velocity: Vec3f32,
}
impl Component for ParticleEmitterComponent {}

/// State of an individual particle in the simulation.
#[derive(Clone, Debug)]
pub struct ParticleStateComponent {
  /// The simulation time when the particle was created.
  pub created_at: f32,
  /// The total lifetime of this particle.
  pub lifetime: f32,
  /// The current velocity of the particle.
  pub velocity: Vec3f32,
}
impl Component for ParticleStateComponent {}

/// Necessary boilerplate for rendering system
pub enum RenderableDataRef<'a> {
  ImageBillboard(&'a ImageBillboardComponent),
  PhysicalMesh(&'a PhysicalMeshComponent),
  Cursor(&'a CursorComponent),
  Markers(&'a MarkersComponent),
  Measurement(&'a MeasurementComponent),
}

impl<'a> RenderableDataRef<'a> {
  pub fn index_count(&self) -> u32 {
    match self {
      RenderableDataRef::ImageBillboard(_) => 4,
      RenderableDataRef::PhysicalMesh(mesh) => mesh.mesh.indices.len() as u32,
      RenderableDataRef::Cursor(_) => 4, // 4 vertices for the quad cursor
      RenderableDataRef::Markers(m) => (m.markers.len() * 4) as u32,
      RenderableDataRef::Measurement(_) => 6, // 6 vertices for line list
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

use thiserror::Error;
use crate::types;
use crate::types::{EngineError, EngineResult};

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

/// A trait for type-erased component storage.
trait ComponentStorage: Send + Sync + core::fmt::Debug {
  fn as_any(&self) -> &dyn Any;
  fn as_mut_any(&mut self) -> &mut dyn Any;
  /// Moves a component from this storage to another, using swap_remove for efficiency.
  fn swap_remove_and_push_to(&mut self, index: usize, other: &mut dyn ComponentStorage);
  /// Removes a component from this storage, using swap_remove for efficiency.
  fn swap_remove(&mut self, index: usize);
  /// Pushes a type-erased component into this storage.
  fn push_any(&mut self, component: Box<dyn Any + Send + Sync>);
  /// The `TypeId` of the component stored.
  fn component_type_id(&self) -> TypeId;
}

impl<T: Component> ComponentStorage for Vec<T> {
  fn as_any(&self) -> &dyn Any {
    self
  }
  fn as_mut_any(&mut self) -> &mut dyn Any {
    self
  }

  fn swap_remove_and_push_to(&mut self, index: usize, other: &mut dyn ComponentStorage) {
    let removed = self.swap_remove(index);
    if let Some(other_vec) = other.as_mut_any().downcast_mut::<Vec<T>>() {
      other_vec.push(removed);
    }
  }

  fn swap_remove(&mut self, index: usize) {
    self.swap_remove(index);
  }

  fn push_any(&mut self, component: Box<dyn Any + Send + Sync>) {
    if let Ok(c) = component.downcast::<T>() {
      self.push(*c);
    }
  }

  fn component_type_id(&self) -> TypeId {
    TypeId::of::<T>()
  }
}

/// Metadata about a registered component type.
#[derive(Debug)]
struct ComponentMeta {
  dependencies: Vec<TypeId>,
  /// A function pointer to create a new, empty storage for this component type.
  /// TODO: remove double indirection by storing an "inline", Not Sized, StableVector, which uses OS's Virtual Memory system to reserve an enormous amount of space and commit what it needs
  new_storage: fn() -> RwLock<Box<dyn ComponentStorage>>,
  type_name: &'static str,
}

/// An Archetype represents a unique set of component types.
#[derive(Debug)]
struct Archetype {
  components: HashMap<TypeId, RwLock<Box<dyn ComponentStorage>>>,
  component_types: HashSet<TypeId>,
  entities: Vec<EntityId>,
}

impl Archetype {
  fn has_components(&self, component_types: &[TypeId]) -> bool {
    component_types
      .iter()
      .all(|t| self.component_types.contains(t))
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
  // TODO: add a hierarchy of EntityIds. Challenge: consistency with entities SlotMap
  hierarchy: RwLock<SceneHierarchy>,
  names: RwLock<HashMap<EntityId, String>>,
}

#[derive(Default, Debug)]
pub struct SceneHierarchy {
  parents: HashMap<EntityId, EntityId>,
  children: HashMap<EntityId, Vec<EntityId>>,
}

impl SceneHierarchy {
  fn set_parent(&mut self, child: EntityId, parent: Option<EntityId>) {
    // Remove from old parent's children list
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
    // Remove from parent's children list
    if let Some(parent) = self.parents.remove(&entity) {
      if let Some(children) = self.children.get_mut(&parent) {
        children.retain(|c| *c != entity);
      }
    }

    // Remove entity as a parent from its children and recursively remove them
    if let Some(children) = self.children.remove(&entity) {
      for child in children {
        self.remove_entity(child);
      }
    }
  }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
  pub fn new() -> Self {
    let empty_archetype = Archetype {
      components: HashMap::new(),
      component_types: HashSet::new(),
      entities: Vec::new(),
    };

    Self {
      entities: RwLock::new(SlotMap::with_key()),
      archetypes: RwLock::new(alloc::vec![empty_archetype]),
      component_meta: RwLock::new(HashMap::new()),
      hierarchy: RwLock::new(SceneHierarchy::default()),
      names: RwLock::new(HashMap::new()),
    }
  }

  pub fn add_camera<S>(
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
    self.add_component(
      camera_entity,
      CameraComponent {
        // this is to be set every frame in the rendering code. Just give a reasonable default here
        projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 10000.0),
        near_plane: 0.1,
        far_plane: 10000.0,
      },
    )?;
    self.set_parent(camera_entity, Some(parent));
    Ok(camera_entity)
  }

  // TODO: probably return a result
  pub fn set_parent(&self, child: EntityId, parent: Option<EntityId>) {
    // Ensure both entities exist.
    let entities = self.entities.read();
    if !entities.contains_key(child)
      || (parent.is_some() && !entities.contains_key(parent.unwrap()))
    {
      return;
    }

    self.hierarchy.write().set_parent(child, parent);
  }

  pub fn get_parent(&self, entity: EntityId) -> Option<EntityId> {
    let entities = self.entities.read();
    if !entities.contains_key(entity) {
      return None;
    }
    let hierarchy = self.hierarchy.read();
    hierarchy.parents.get(&entity).cloned()
  }

  pub fn get_entity_component_names(&self, entity: EntityId) -> Vec<&'static str> {
    let entities = self.entities.read();
    let location = match entities.get(entity) {
      Some(l) => l,
      None => return Vec::new(),
    };

    let archetypes = self.archetypes.read();
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
      return; // Cycle detected
    }

    if !filter(self, current_entity) {
      return;
    }

    if !callback(self, current_entity, accumulator) {
      return; // Stop traversal down this branch if callback returns false
    }

    let hierarchy = self.hierarchy.read();
    if let Some(children) = hierarchy.children.get(&current_entity) {
      // Must clone children to avoid holding lock during recursive call
      let children_clone = children.clone();
      drop(hierarchy); // Release lock before recursive calls
      for &child in &children_clone {
        self.traverse_recursive(child, accumulator, filter, callback, visited);
      }
    }
  }

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
      return; // Cycle detected
    }

    let transform = self.with_component(current_entity, |c: &TransformComponent| *c);
    // This is tricky. We can't return a reference from with_component due to lifetimes.
    // So we get a pointer, and use it within an unsafe block. This is safe because
    // we are single-threaded here and nothing will deallocate the component.
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
      // If pre_visit returns false, we still need to call post_visit to balance the stack.
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

  /// Computes the global transform of an entity by traversing up the hierarchy.
  /// Returns `None` if the target entity does not have a `TransformComponent`.
  /// Skips any ancestors that do not possess a `TransformComponent`.
  pub fn global_transform(&self, entity_id: EntityId) -> Option<TransformComponent> {
    // 1. Get the entity's local transform. If it doesn't have one, bail out.
    let mut accumulated_transform = self.with_component(entity_id, |c: &TransformComponent| *c)?;

    let mut current_entity = entity_id;

    // 2. Traverse up the hierarchy
    loop {
      // Scope the hierarchy read lock tightly. We do NOT want to hold this
      // while calling `with_component`, as that grabs `entities` and `archetypes`
      // locks, which could lead to deadlocks if other threads lock in a different order.
      let parent_opt = {
        let hierarchy = self.hierarchy.read();
        hierarchy.parents.get(&current_entity).copied()
      };

      if let Some(parent_id) = parent_opt {
        // 3. If the parent has a transform, accumulate it. Otherwise, it simply skips.
        if let Some(parent_transform) = self.with_component(parent_id, |c: &TransformComponent| *c)
        {
          accumulated_transform =
            Self::combine_transforms(&parent_transform, &accumulated_transform);
        }

        // Move up to the next ancestor
        current_entity = parent_id;
      } else {
        // No more parents, we've reached the root of this tree.
        break;
      }
    }

    Some(accumulated_transform)
  }

  /// Helper to combine a parent's transform with a child's transform.
  /// TODO: Move elsewhere if needed
  fn combine_transforms(
    parent: &TransformComponent,
    child: &TransformComponent,
  ) -> TransformComponent {
    // Typical TRS (Translation, Rotation, Scale) combination logic:
    // Global Scale = Parent Scale * Child Scale (Component-wise)
    // Global Rotation = Parent Rotation * Child Rotation
    // Global Position = Parent Position + (Parent Rotation * (Parent Scale * Child Position))

    TransformComponent {
      scale: parent.scale * child.scale,
      rotation: parent.rotation * child.rotation,
      position: parent.position
        + (parent
          .rotation
          .rotate_vector((parent.scale * child.position))),
    }
  }

  /// Registers a component type, its dependencies, and its storage constructor.
  pub fn register_component<T: Component>(&self, dependencies: &[TypeId]) {
    let mut meta = self.component_meta.write();
    meta.insert(
      TypeId::of::<T>(),
      ComponentMeta {
        dependencies: dependencies.to_vec(),
        new_storage: || RwLock::new(Box::new(Vec::<T>::new())),
        type_name: core::any::type_name::<T>(),
      },
    );
  }

  pub fn add_component<T: Component>(
    &self,
    entity_id: EntityId,
    component: T,
  ) -> Result<(), AddComponentError> {
    let new_component_type = TypeId::of::<T>();

    // --- 1. Find entity and source archetype, and perform all checks ---
    let src_location = {
      let entities = self.entities.read();
      *entities
        .get(entity_id)
        .ok_or(AddComponentError::EntityNotFound)?
    };

    let (target_archetype_index, is_new_archetype) = {
      let archetypes = self.archetypes.read();
      let src_archetype = &archetypes[src_location.archetype_index];

      // TODO: Support for checking whether the new component type is present but its place in the entity
      // is with a none-like value. This implies that archetype grouping can be loose to some extent based on some
      // cost factor which determines whether it's worth it to migrate and entity to a new Archetype or not
      if src_archetype.component_types.contains(&new_component_type) {
        return Err(AddComponentError::ComponentAlreadyExists);
      }

      // requested component is a new type, therefore check for presence of its dependencies
      let meta = self.component_meta.read();
      // if we are in a proper state, and the component type has been already registered, then
      // its metadata should alsow be there. Checking regardless
      let component_meta = meta
        .get(&new_component_type)
        .ok_or(AddComponentError::ComponentNotRegistered)?;

      if !src_archetype.has_components(&component_meta.dependencies) {
        let missing_dependency_type_id = component_meta
          .dependencies
          .iter()
          .find(|type_id| !src_archetype.component_types.contains(*type_id))
          .unwrap(); // This is safe because we know a dependency is missing.

        let missing_component_name = meta
          .get(missing_dependency_type_id)
          .map_or("Unknown Component", |meta| meta.type_name);

        return Err(AddComponentError::DependencyNotSatisfied {
          missing: missing_component_name,
        });
      }

      // To check whether we already have an archetype with this new component set,
      // we must iterate through existing archetypes.
      // NOTE: This could be optimized with a bloom filter or by hashing the component set.
      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      // Find a target archetype that matches the new component set
      let mut found_index = None;
      for (i, arch) in archetypes.iter().enumerate() {
        if arch.component_types == target_component_types {
          found_index = Some(i);
          break;
        }
      }
      (found_index, found_index.is_none())
    };

    let target_archetype_index = if is_new_archetype {
      // --- Create a new archetype if none was found ---
      let mut archetypes = self.archetypes.write();
      let meta = self.component_meta.read();

      let src_archetype = &archetypes[src_location.archetype_index];
      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      // Re-check if another thread created it in the meantime
      let mut re_check_index = None;
      for (i, arch) in archetypes.iter().enumerate() {
        if arch.component_types == target_component_types {
          re_check_index = Some(i);
          break;
        }
      }

      if let Some(index) = re_check_index {
        index
      } else {
        let mut new_arch = Archetype {
          component_types: target_component_types,
          components: HashMap::new(),
          entities: Vec::new(),
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

    // --- 3. Move the entity and its components ---
    if src_location.archetype_index != target_archetype_index {
      let mut entities = self.entities.write();
      let mut archetypes = self.archetypes.write();

      // Hacky syntax to get two `&mut` out of a single `&mut Vec`
      let (src_arch, target_arch) = if src_location.archetype_index < target_archetype_index {
        let (left, right) = archetypes.split_at_mut(target_archetype_index);
        (&mut left[src_location.archetype_index], &mut right[0])
      } else {
        let (left, right) = archetypes.split_at_mut(src_location.archetype_index);
        (&mut right[0], &mut left[target_archetype_index])
      };

      let moved_entity_id = src_arch.entities.swap_remove(src_location.row_index);
      let swapped_entity_id_opt = src_arch.entities.get(src_location.row_index).copied();

      for (type_id, target_storage_lock) in target_arch.components.iter() {
        if *type_id == new_component_type {
          continue;
        }
        let mut src_storage = src_arch.components[type_id].write();
        let mut target_storage = target_storage_lock.write();
        src_storage.swap_remove_and_push_to(src_location.row_index, &mut **target_storage);
      }

      target_arch.components[&new_component_type]
        .write()
        .push_any(Box::new(component));
      target_arch.entities.push(moved_entity_id);

      let new_location = EntityLocation {
        archetype_index: target_archetype_index,
        row_index: target_arch.entities.len() - 1,
      };

      // entity location bookkeeping
      *entities.get_mut(moved_entity_id).unwrap() = new_location;
      if let Some(swapped_id) = swapped_entity_id_opt {
        entities.get_mut(swapped_id).unwrap().row_index = src_location.row_index;
      }
    }

    Ok(())
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
      let mut entities = self.entities.write();
      let mut archetypes = self.archetypes.write();

      let archetype = &mut archetypes[0];
      let entity_location = EntityLocation {
        archetype_index: 0,
        row_index: archetype.entities.len(),
      };

      let entity_id = entities.insert(entity_location);
      archetype.entities.push(entity_id);
      entity_id
    };

    self.names.write().insert(entity_id, actual_name);

    entity_id
  }

  pub fn get_entity_by_name(&self, name: &str) -> Option<EntityId> {
    self
      .names
      .read()
      .iter()
      .find(|(_, n)| *n == name)
      .map(|(id, _)| *id)
  }

  pub fn get_name(&self, entity: EntityId) -> Option<alloc::string::String> {
    self.names.read().get(&entity).cloned()
  }

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

  pub fn with_component<T: Component, F, R>(&self, entity_id: EntityId, f: F) -> Option<R>
  where
    F: FnOnce(&T) -> R,
  {
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;

    let archetypes = self.archetypes.read();
    let archetype = &archetypes[location.archetype_index];

    let components_lock = archetype.components.get(&TypeId::of::<T>())?.read();
    let components = components_lock.as_any().downcast_ref::<Vec<T>>()?;

    Some(f(&components[location.row_index]))
  }

  pub fn with_component_mut<T: Component, F, R>(&self, entity_id: EntityId, f: F) -> Option<R>
  where
    F: FnOnce(&mut T) -> R,
  {
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;

    let archetypes = self.archetypes.read();
    let archetype = &archetypes[location.archetype_index];

    let mut components_lock = archetype.components.get(&TypeId::of::<T>())?.write();
    let components = components_lock.as_mut_any().downcast_mut::<Vec<T>>()?;

    Some(f(&mut components[location.row_index]))
  }

  // TODO error or option?
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
    let src_arch = &mut archetypes[src_location.archetype_index];

    let swapped_entity_id_opt = src_arch.entities.get(src_location.row_index).copied();
    src_arch.entities.swap_remove(src_location.row_index);

    for (_, storage_lock) in src_arch.components.iter() {
      let mut storage = storage_lock.write();
      storage.swap_remove(src_location.row_index);
    }

    if let Some(swapped_id) = swapped_entity_id_opt {
      // The entity that was swapped in now has row_index = src_location.row_index
      if let Some(loc) = self.entities.write().get_mut(swapped_id) {
        loc.row_index = src_location.row_index;
      }
    }
  }

  pub fn remove_component<T: Component>(&self, entity_id: EntityId) -> Result<(), &'static str> {
    let type_id_to_remove = TypeId::of::<T>();

    let src_location = {
      let entities = self.entities.read();
      *entities.get(entity_id).ok_or("Entity not found")?
    };

    let target_archetype_index = {
      let archetypes = self.archetypes.read();
      let src_archetype = &archetypes[src_location.archetype_index];

      if !src_archetype.component_types.contains(&type_id_to_remove) {
        return Err("Component not found on entity");
      }

      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.remove(&type_id_to_remove);

      archetypes
        .iter()
        .position(|arch| arch.component_types == target_component_types)
        .ok_or(
          "Target archetype not found. This should not happen if an empty archetype always exists.",
        )?
    };

    if src_location.archetype_index != target_archetype_index {
      let mut entities = self.entities.write();
      let mut archetypes = self.archetypes.write();

      let (src_arch, target_arch) = if src_location.archetype_index < target_archetype_index {
        let (left, right) = archetypes.split_at_mut(target_archetype_index);
        (&mut left[src_location.archetype_index], &mut right[0])
      } else {
        let (left, right) = archetypes.split_at_mut(src_location.archetype_index);
        (&mut right[0], &mut left[target_archetype_index])
      };

      let moved_entity_id = src_arch.entities.swap_remove(src_location.row_index);
      let swapped_entity_id_opt = src_arch.entities.get(src_location.row_index).copied();

      // The component to be removed is handled separately to avoid moving it.
      src_arch
        .components
        .get(&type_id_to_remove)
        .unwrap()
        .write()
        .as_mut_any()
        .downcast_mut::<Vec<T>>()
        .unwrap()
        .swap_remove(src_location.row_index);

      for (type_id, src_storage_lock) in src_arch.components.iter() {
        if *type_id == type_id_to_remove {
          continue;
        }

        if let Some(target_storage_lock) = target_arch.components.get(type_id) {
          let mut src_storage = src_storage_lock.write();
          let mut target_storage = target_storage_lock.write();
          src_storage.swap_remove_and_push_to(src_location.row_index, &mut **target_storage);
        }
      }

      target_arch.entities.push(moved_entity_id);
      let new_location = EntityLocation {
        archetype_index: target_archetype_index,
        row_index: target_arch.entities.len() - 1,
      };

      *entities.get_mut(moved_entity_id).unwrap() = new_location;
      if let Some(swapped_id) = swapped_entity_id_opt {
        entities.get_mut(swapped_id).unwrap().row_index = src_location.row_index;
      }
    }

    Ok(())
  }

  pub fn has_component<T: Component>(&self, entity_id: EntityId) -> HasComponentResultEnum {
    let archetypes = self.archetypes.read();
    // 1. find the archetype which contains the entity. not found means Err
    let archetype = archetypes
      .iter()
      .find(|archetype| archetype.entities.iter().any(|e| *e == entity_id));
    if archetype.is_none() {
      return HasComponentResultEnum::EntityNotFound;
    }
    // 2. Check if archetype has component in question
    let archetype = unsafe { archetype.unwrap_unchecked() };
    if archetype.has_components(&[TypeId::of::<T>()]) {
      HasComponentResultEnum::EntityHasComponent
    } else {
      HasComponentResultEnum::ComponentNotFound
    }
  }

  pub fn query1<T: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T),
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            f(*entity_id, &components[i]);
          }
        }
      }
    }
  }

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
        let components1 = comp_storage1_lock
          .as_any()
          .downcast_ref::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          f(*entity_id, &components1[i], &components2[i]);
        }
      }
    }
  }

  pub fn query1_mut<T: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &mut T),
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            f(*entity_id, &mut components[i]);
          }
        }
      }
    }
  }

  pub fn query2_mut<T1: Component, T2: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &mut T1, &mut T2),
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    assert_ne!(
      type_t1, type_t2,
      "Cannot mutably query the same component type twice."
    );

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();
        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          f(*entity_id, &mut components1[i], &mut components2[i]);
        }
      }
    }
  }

  // TODO: unit tests for _res version of query methods
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
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            // Only keep the result if the closure returns Some(R)
            if let Some(result) = f(*entity_id, &components[i]) {
              results.push((result, *entity_id));
            }
          }
        }
      }
    }

    results
  }

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
        let components1 = comp_storage1_lock
          .as_any()
          .downcast_ref::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          if let Some(result) = f(*entity_id, &components1[i], &components2[i]) {
            results.push((result, *entity_id));
          }
        }
      }
    }

    results
  }

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
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            if let Some(result) = f(*entity_id, &mut components[i]) {
              results.push((result, *entity_id));
            }
          }
        }
      }
    }

    results
  }

  pub fn query2_res_mut<T1: Component, T2: Component, F, R>(&self, mut f: F) -> Vec<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T1, &mut T2) -> Option<R>,
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    assert_ne!(
      type_t1, type_t2,
      "Cannot mutably query the same component type twice."
    );

    let mut results = Vec::new();
    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();
        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          if let Some(result) = f(*entity_id, &mut components1[i], &mut components2[i]) {
            results.push((result, *entity_id));
          }
        }
      }
    }

    results
  }

  pub fn query1_first_res<T: Component, F, R>(&self, mut f: F) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            // Only keep the result if the closure returns Some(R)
            if let Some(result) = f(*entity_id, &components[i]) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }

    None
  }

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
        let components1 = comp_storage1_lock
          .as_any()
          .downcast_ref::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          if let Some(result) = f(*entity_id, &components1[i], &components2[i]) {
            Some((result, *entity_id));
          }
        }
      }
    }

    None
  }

  pub fn query1_res_first_mut<T: Component, F, R>(&self, mut f: F) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T) -> Option<R>,
  {
    let archetypes = self.archetypes.read();
    let type_t = TypeId::of::<T>();

    for archetype in archetypes.iter() {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components) = comp_storage.as_mut_any().downcast_mut::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            if let Some(result) = f(*entity_id, &mut components[i]) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }

    None
  }

  pub fn query2_res_first_mut<T1: Component, T2: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &mut T1, &mut T2) -> Option<R>,
  {
    let type_t1 = TypeId::of::<T1>();
    let type_t2 = TypeId::of::<T2>();
    assert_ne!(
      type_t1, type_t2,
      "Cannot mutably query the same component type twice."
    );

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      if archetype.components.contains_key(&type_t1) && archetype.components.contains_key(&type_t2)
      {
        let mut comp_storage1_lock = archetype.components[&type_t1].write();
        let mut comp_storage2_lock = archetype.components[&type_t2].write();
        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          if let Some(result) = f(*entity_id, &mut components1[i], &mut components2[i]) {
            return Some((result, *entity_id));
          }
        }
      }
    }

    None
  }

  /// Computes the effective parent global transform by walking up the hierarchy
  /// until an ancestor with a `TransformComponent` is found.
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

  /// Efficiently sets the global transform of an entity.
  ///
  /// This dynamically computes the required local transform so that when evaluated against
  /// its parents, it produces the desired global transform without 4x4 matrix inversions.
  pub fn set_global_transform(
    &self,
    entity_id: EntityId,
    new_global: TransformComponent,
  ) -> EngineResult<()> {
    // 1. Traverse upwards to get the effective parent global transform.
    // Executed BEFORE acquiring the component mutation lock to prevent deadlocks.
    let parent_global = self.parent_global_transform(entity_id);

    // 2. Mathematically isolate and apply the new local transform.
    self
      .with_component_mut(entity_id, |t: &mut TransformComponent| {
        if let Some(pg) = parent_global {
          // Local Scale = Target Scale / Parent Scale
          t.scale = Vec3f32::from_components(
            safe_div(new_global.scale.x(), pg.scale.x()),
            safe_div(new_global.scale.y(), pg.scale.y()),
            safe_div(new_global.scale.z(), pg.scale.z()),
          );

          // Local Rotation = Parent Rotation^-1 * Target Rotation
          // Note: Assumes `Quat` has `.inverse()`. If your math lib uses `.conjugate()`
          // for unit quaternions instead, you can safely substitute it here.
          let inv_rot = pg.rotation.inverse();
          t.rotation = inv_rot * new_global.rotation;

          // Local Position = Parent Rotation^-1 * (Target Position - Parent Position) / Parent Scale
          // Extracted component-wise to avoid assuming a `Sub` operator overload on `Vec3f32`.
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
          // No parent hierarchy implies local space perfectly equals global space
          *t = new_global;
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "set_global_transform: Entity not found or missing TransformComponent",
      ))?;

    Ok(())
  }

  /// Sets only the global position and rotation of an entity.
  /// The local scale configuration is preserved securely intact.
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
          let safe_div = |a: f32, b: f32| {
            if b > -1e-6_f32 && b < 1e-6_f32 {
              0.0
            } else {
              a / b
            }
          };

          let inv_rot = pg.rotation.inverse();

          // Local Rotation
          t.rotation = inv_rot * new_rotation;

          // Local Position
          let diff_pos = Vec3f32::from_components(
            new_position.x() - pg.position.x(),
            new_position.y() - pg.position.y(),
            new_position.z() - pg.position.z(),
          );
          let unrotated_diff = inv_rot.rotate_vector(diff_pos);

          t.position = Vec3f32::from_components(
            safe_div(unrotated_diff.x(), pg.scale.x()),
            safe_div(unrotated_diff.y(), pg.scale.y()),
            safe_div(unrotated_diff.z(), pg.scale.z()),
          );
          // Note: `t.scale` is deliberately unmodified!
        } else {
          t.position = new_position;
          t.rotation = new_rotation;
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "set_global_position_and_rotation: Entity not found or missing TransformComponent",
      ))?;

    Ok(())
  }

  /// Searches for the first instance of a component of type `T` and deletes it.
  ///
  /// Returns `Some(EntityId)` containing the affected entity if the component
  /// was successfully found and removed, or `None` if it was not found.
  pub fn remove_first_component<T: Component>(&self) -> Option<EntityId> {
    let type_id = TypeId::of::<T>();

    // 1. Locate the first entity possessing the component.
    // We tightly scope this block to ensure the `archetypes.read()` lock is explicitly
    // dropped BEFORE we call `remove_component`. This prevents thread deadlocks.
    let target_entity = {
      let archetypes = self.archetypes.read();
      archetypes.iter().find_map(|arch| {
        // Check if the archetype has the component.
        // If it does, we try to grab the first entity.
        // If the archetype is empty, `.first()` evaluates to None and find_map continues.
        if arch.component_types.contains(&type_id) {
          arch.entities.first().copied()
        } else {
          None
        }
      })
    };

    // 2. Attempt to remove it using existing archetype migration logic.
    if let Some(entity_id) = target_entity {
      // We check `.is_ok()` to protect against concurrent modifications in the
      // microsecond gap between step 1 and step 2, and to handle potential ECS structural
      // errors (like missing empty target archetypes).
      if self.remove_component::<T>(entity_id).is_ok() {
        return Some(entity_id);
      }
    }

    None
  }

  /// Queries all entities that have component `T` but DO NOT have component `U`.
  pub fn query1_without<T: Component, U: Component, F>(&self, mut f: F)
  where
    F: FnMut(EntityId, &T),
  {
    let type_t = TypeId::of::<T>();
    let type_u = TypeId::of::<U>();
    assert_ne!(
      type_t, type_u,
      "Included and excluded component types must be distinct."
    );

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      // Check that the archetype has T, but crucially, does NOT have U
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap();
        let comp_storage = comp_storage_lock.read();

        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            f(*entity_id, &components[i]);
          }
        }
      }
    }
  }

  /// Queries entities that have component `T` but DO NOT have component `U`,
  /// stopping and returning the first result where the closure returns `Some`.
  pub fn query1_first_res_without<T: Component, U: Component, F, R>(
    &self,
    mut f: F,
  ) -> Option<(R, EntityId)>
  where
    F: FnMut(EntityId, &T) -> Option<R>,
  {
    let type_t = TypeId::of::<T>();
    let type_u = TypeId::of::<U>();
    assert_ne!(
      type_t, type_u,
      "Included and excluded component types must be distinct."
    );

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      // Check that the archetype has T, but crucially, does NOT have U
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap();
        let comp_storage = comp_storage_lock.read();

        if let Some(components) = comp_storage.as_any().downcast_ref::<Vec<T>>() {
          for (i, entity_id) in archetype.entities.iter().enumerate() {
            if let Some(result) = f(*entity_id, &components[i]) {
              return Some((result, *entity_id));
            }
          }
        }
      }
    }

    None
  }

  /// Queries entities that have components `T1` and `T2`, but DO NOT have component `U`.
  /// Stops and returns the first result where the closure returns `Some`.
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

    assert_ne!(
      type_t1, type_t2,
      "Included component types must be distinct."
    );
    assert_ne!(
      type_t1, type_u,
      "Included component T1 and excluded component U must be distinct."
    );
    assert_ne!(
      type_t2, type_u,
      "Included component T2 and excluded component U must be distinct."
    );

    let archetypes = self.archetypes.read();

    for archetype in archetypes.iter() {
      // Check that the archetype has T1 and T2, but crucially, does NOT have U
      if archetype.components.contains_key(&type_t1)
        && archetype.components.contains_key(&type_t2)
        && !archetype.components.contains_key(&type_u)
      {
        let comp_storage1_lock = archetype.components.get(&type_t1).unwrap().read();
        let comp_storage2_lock = archetype.components.get(&type_t2).unwrap().read();

        let components1 = comp_storage1_lock
          .as_any()
          .downcast_ref::<Vec<T1>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<T2>>()
          .unwrap();

        for (i, entity_id) in archetype.entities.iter().enumerate() {
          if let Some(result) = f(*entity_id, &components1[i], &components2[i]) {
            return Some((result, *entity_id));
          }
        }
      }
    }

    None
  }

  /// Validates the scene against specific structural constraints.
  ///
  /// Constraints:
  /// - Maximum of 1 `SunComponent`
  /// - Maximum of 1 `CursorComponent`
  /// - Maximum of 1 `SkyComponent`
  /// - Maximum of 1 `GridComponent`
  /// TODO constraints on following component and selected component
  /// Multiple instances of other components are permitted.
  pub fn validate(&self) -> EngineResult<()> {
    let archetypes = self.archetypes.read();

    // Cache the TypeIds to avoid recalculating them in the loop
    let sun_type = TypeId::of::<SunComponent>();
    let cursor_type = TypeId::of::<CursorComponent>();
    let sky_type = TypeId::of::<SkyComponent>();
    let grid_type = TypeId::of::<GridComponent>();

    let mut sun_count = 0;
    let mut cursor_count = 0;
    let mut sky_count = 0;
    let mut grid_count = 0;

    // Single pass over archetypes avoids multiple lock acquisitions
    for arch in archetypes.iter() {
      let num_entities = arch.entities.len();

      // Skip empty archetypes
      if num_entities == 0 {
        continue;
      }

      if arch.component_types.contains(&sun_type) {
        sun_count += num_entities;
      }
      if arch.component_types.contains(&cursor_type) {
        cursor_count += num_entities;
      }
      if arch.component_types.contains(&sky_type) {
        sky_count += num_entities;
      }
      if arch.component_types.contains(&grid_type) {
        grid_count += num_entities;
      }
    }

    // Enforce constraints (0 or 1 instances)
    if sun_count > 1 {
      return Err(EngineError::InvalidOperation(
        "scene validation failed: multiple SunComponent found (expected 0 or 1)",
      ));
    }
    if cursor_count > 1 {
      return Err(EngineError::InvalidOperation(
        "scene validation failed: multiple CursorComponent found (expected 0 or 1)",
      ));
    }
    if sky_count > 1 {
      return Err(EngineError::InvalidOperation(
        "scene validation failed: multiple SkyComponent found (expected 0 or 1)",
      ));
    }
    if grid_count > 1 {
      return Err(EngineError::InvalidOperation(
        "scene validation failed: multiple GridComponent found (expected 0 or 1)",
      ));
    }

    Ok(())
  }
}
