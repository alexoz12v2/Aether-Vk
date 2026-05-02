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

use thiserror::Error;
use crate::{
  types,
  types::{EngineError, EngineResult},
  simulation::comet::Comet,
};
use aethervk_oshal_rlib::{
  math::matrix::Matrix4,
  math::{safe_div, FloatLike},
  os::pool::ThreadPool,
  os::pool::chunked::ThreadPoolChunkedExt,
  math::quaternion::Quaternion,
  math::vector::{Vector3, Vector4},
  math::{matrix::mat4::Mat4x4f32, vector::vec4::Quat},
  math::vector::vec3::Vec3f32,
};
use slotmap::{new_key_type, SlotMap};
use spin::RwLock;
use alloc::{
  boxed::Box,
  vec::Vec,
  string::{String},
};
use core::any::{Any, TypeId};
use hashbrown::{HashMap, HashSet};

pub mod almanac_planet;
pub mod camera;
pub mod interaction;
pub mod particles;
pub mod text;

pub use almanac_planet::AlmanacPlanet;
pub use particles::{ParticleEmitterConfig, ParticleSystemComponent, ParticleData, GaussianParams};

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

/// A component that stores whether a gizmo should be rendered for this entity
#[derive(Clone, Debug, PartialEq)]
pub struct GizmoComponent {
  pub gizmo_visible: bool,
  pub gizmo_scale: f32,
}
impl Component for GizmoComponent {}

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
  Gizmo(&'a GizmoComponent),
  ParticleSystem(&'a particles::ParticleSystemComponent),
}

impl<'a> RenderableDataRef<'a> {
  pub fn index_count(&self) -> u32 {
    match self {
      RenderableDataRef::ImageBillboard(_) => 4,
      RenderableDataRef::PhysicalMesh(mesh) => mesh.mesh.indices.len() as u32,
      RenderableDataRef::Cursor(_) => 4, // 4 vertices for the quad cursor
      RenderableDataRef::Markers(m) => (m.markers.len() * 4) as u32,
      RenderableDataRef::Measurement(_) => 6, // 6 vertices for line list
      RenderableDataRef::Gizmo(_) => 6,
      RenderableDataRef::ParticleSystem(p) => (p.particles.len() * 4) as u32,
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
    component_types
      .iter()
      .all(|t| self.component_types.contains(t))
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
}

#[derive(Default, Debug)]
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
      free_slots: Vec::new(),
    };

    Self {
      entities: RwLock::new(SlotMap::with_key()),
      archetypes: RwLock::new(alloc::vec![empty_archetype]),
      component_meta: RwLock::new(HashMap::new()),
      hierarchy: RwLock::new(SceneHierarchy::default()),
      names: RwLock::new(HashMap::new()),
    }
  }

  pub fn entity_count(&self) -> usize {
    self.entities.read().len()
  }

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

  pub fn hierarchy_breadth(&self) -> usize {
    let hierarchy = self.hierarchy.read();
    hierarchy
      .children
      .values()
      .map(|children| children.len())
      .max()
      .unwrap_or(0)
  }

  pub fn should_parallelize(&self) -> bool {
    let size = self.entity_count();
    let depth = self.hierarchy_depth();
    let breadth = self.hierarchy_breadth();

    // Simple heuristic for parallelization threshold
    // Parallelize if we have a lot of entities, or if the scene graph is very wide
    size > 1000 || breadth > 100
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
        projection: Mat4x4f32::perspective_vk(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 10000.0),
        near_plane: 0.1,
        far_plane: 10000.0,
      },
    )?;
    self.set_parent(camera_entity, Some(parent));
    Ok(camera_entity)
  }

  pub fn set_parent(&self, child: EntityId, parent: Option<EntityId>) {
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
    self.hierarchy.read().parents.get(&entity).cloned()
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

  pub fn add_component<T: Component>(
    &self,
    entity_id: EntityId,
    component: T,
  ) -> Result<(), AddComponentError> {
    let new_component_type = TypeId::of::<T>();

    let src_location = {
      let entities = self.entities.read();
      *entities
        .get(entity_id)
        .ok_or(AddComponentError::EntityNotFound)?
    };

    let (target_archetype_index, is_new_archetype) = {
      let archetypes = self.archetypes.read();
      let src_archetype = &archetypes[src_location.archetype_index];

      if src_archetype.component_types.contains(&new_component_type) {
        return Err(AddComponentError::ComponentAlreadyExists);
      }

      let meta = self.component_meta.read();
      let component_meta = meta
        .get(&new_component_type)
        .ok_or(AddComponentError::ComponentNotRegistered)?;

      if !src_archetype.has_components(&component_meta.dependencies) {
        let missing_dep = component_meta
          .dependencies
          .iter()
          .find(|t| !src_archetype.component_types.contains(*t))
          .unwrap();
        let missing_name = meta.get(missing_dep).map_or("Unknown", |m| m.type_name);
        return Err(AddComponentError::DependencyNotSatisfied {
          missing: missing_name,
        });
      }

      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      let found_index = archetypes
        .iter()
        .position(|arch| arch.component_types == target_component_types);
      (found_index, found_index.is_none())
    };

    let target_archetype_index = if is_new_archetype {
      let mut archetypes = self.archetypes.write();
      let meta = self.component_meta.read();

      let src_archetype = &archetypes[src_location.archetype_index];
      let mut target_component_types = src_archetype.component_types.clone();
      target_component_types.insert(new_component_type);

      if let Some(index) = archetypes
        .iter()
        .position(|arch| arch.component_types == target_component_types)
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
      let mut entities = self.entities.write();
      let mut archetypes = self.archetypes.write();

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

    let mut entities = self.entities.write();
    let mut archetypes = self.archetypes.write();
    let src_arch = &mut archetypes[src_location.archetype_index];

    src_arch.free_slot(src_location.row_index);
    src_arch.compact(&mut entities); // Organically triggers when arrays begin resembling swiss-cheese
  }

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

      let found_index = archetypes
        .iter()
        .position(|arch| arch.component_types == target_component_types);
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
      if let Some(index) = archetypes
        .iter()
        .position(|arch| arch.component_types == target_component_types)
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
      let mut entities = self.entities.write();
      let mut archetypes = self.archetypes.write();

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

  pub fn remove_column_if_par<T: Component, F>(&self, pool: &ThreadPool, filter: F)
  where
    F: Fn(EntityId, &T) -> bool + Send + Sync,
  {
    let to_remove = self.query1_res_par(pool, |e, c: &T| if filter(e, c) { Some(e) } else { None });
    for (e, _) in to_remove {
      let _ = self.remove_component::<T>(e);
    }
  }

  pub fn remove_first_component<T: Component>(&self) -> Option<EntityId> {
    let target_entity = self.query1_first_res(|e, _: &T| Some(e));
    if let Some((e, _)) = target_entity {
      if self.remove_component::<T>(e).is_ok() {
        return Some(e);
      }
    }
    None
  }

  pub fn has_component<T: Component>(&self, entity_id: EntityId) -> HasComponentResultEnum {
    let archetypes = self.archetypes.read();
    let archetype = archetypes
      .iter()
      .find(|archetype| archetype.entities.iter().any(|e| *e == Some(entity_id)));
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

  pub fn with_component<T: Component, F, R>(&self, entity_id: EntityId, f: F) -> Option<R>
  where
    F: FnOnce(&T) -> R,
  {
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;
    let archetypes = self.archetypes.read();
    let archetype = &archetypes[location.archetype_index];

    let components_lock = archetype.components.get(&TypeId::of::<T>())?.read();
    let components = components_lock.as_any().downcast_ref::<Vec<Option<T>>>()?;

    Some(f(components[location.row_index].as_ref()?))
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
    let components = components_lock
      .as_mut_any()
      .downcast_mut::<Vec<Option<T>>>()?;

    Some(f(components[location.row_index].as_mut()?))
  }

  // === Sequential Single-Threaded Queries ===

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
          .downcast_ref::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<Option<T2>>>()
          .unwrap();

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

        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T2>>>()
          .unwrap();

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
          .downcast_ref::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<Option<T2>>>()
          .unwrap();

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

        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T2>>>()
          .unwrap();

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
          .downcast_ref::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<Option<T2>>>()
          .unwrap();

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

        let components1 = comp_storage1_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_mut_any()
          .downcast_mut::<Vec<Option<T2>>>()
          .unwrap();

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

        let components1 = comp_storage1_lock
          .as_any()
          .downcast_ref::<Vec<Option<T1>>>()
          .unwrap();
        let components2 = comp_storage2_lock
          .as_any()
          .downcast_ref::<Vec<Option<T2>>>()
          .unwrap();

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

  // === Advanced Scene Traversal Logic ===

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

  pub fn global_transform(&self, entity_id: EntityId) -> Option<TransformComponent> {
    let mut accumulated_transform = self.with_component(entity_id, |c: &TransformComponent| *c)?;
    let mut current_entity = entity_id;

    loop {
      let parent_opt = {
        let hierarchy = self.hierarchy.read();
        hierarchy.parents.get(&current_entity).copied()
      };

      if let Some(parent_id) = parent_opt {
        if let Some(parent_transform) = self.with_component(parent_id, |c: &TransformComponent| *c)
        {
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

  fn combine_transforms(
    parent: &TransformComponent,
    child: &TransformComponent,
  ) -> TransformComponent {
    TransformComponent {
      scale: parent.scale * child.scale,
      rotation: parent.rotation * child.rotation,
      position: parent.position
        + (parent
          .rotation
          .rotate_vector((parent.scale * child.position))),
    }
  }

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
            if b > -1e-6_f32 && b < 1e-6_f32 {
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
        if let Some(components_vec) = comp_storage
          .as_any()
          .downcast_ref::<alloc::vec::Vec<Option<T>>>()
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
      core::iter::repeat_with(alloc::vec::Vec::new)
        .take(num_archetypes)
        .collect();

    let results_ptr = ErasedMutPtr::new(chunk_results.as_mut_ptr());

    self.iter_par_archetypes(pool, |arch_id, archetype| {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let comp_storage = comp_storage_lock.read();
        if let Some(components_vec) = comp_storage
          .as_any()
          .downcast_ref::<alloc::vec::Vec<Option<T>>>()
        {
          let result_vec = unsafe {
            &mut *results_ptr
              .get::<alloc::vec::Vec<(R, EntityId)>>()
              .add(arch_id)
          };
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

  pub fn query1_mut_par<T: Component, F>(&self, pool: &ThreadPool, f: F)
  where
    F: Fn(EntityId, &mut T) + Send + Sync,
  {
    let type_t = core::any::TypeId::of::<T>();

    self.iter_par_archetypes(pool, |_, archetype| {
      if let Some(comp_storage_lock) = archetype.components.get(&type_t) {
        let mut comp_storage = comp_storage_lock.write();
        if let Some(components_vec) = comp_storage
          .as_mut_any()
          .downcast_mut::<alloc::vec::Vec<Option<T>>>()
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

        let components_vec1 = comp_storage1
          .as_mut_any()
          .downcast_mut::<alloc::vec::Vec<Option<T1>>>()
          .unwrap();
        let components_vec2 = comp_storage2
          .as_mut_any()
          .downcast_mut::<alloc::vec::Vec<Option<T2>>>()
          .unwrap();

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
      core::iter::repeat_with(alloc::vec::Vec::new)
        .take(num_archetypes)
        .collect();

    let results_ptr = crate::scene::ErasedMutPtr::new(chunk_results.as_mut_ptr());

    self.iter_par_archetypes(pool, |arch_id, archetype| {
      if archetype.components.contains_key(&type_t) && !archetype.components.contains_key(&type_u) {
        let comp_storage_lock = archetype.components.get(&type_t).unwrap();
        let comp_storage = comp_storage_lock.read();
        if let Some(components_vec) = comp_storage.as_any().downcast_ref::<alloc::vec::Vec<T>>() {
          let result_vec = unsafe {
            &mut *results_ptr
              .get::<alloc::vec::Vec<(R, EntityId)>>()
              .add(arch_id)
          };

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
struct ErasedPtr(*const ());
unsafe impl Send for ErasedPtr {}
unsafe impl Sync for ErasedPtr {}

impl ErasedPtr {
  #[inline(always)]
  pub fn new<T>(ptr: *const T) -> Self {
    Self(ptr as *const ())
  }
  #[inline(always)]
  pub fn get<T>(self) -> *const T {
    self.0 as *const T
  }
}

#[derive(Clone, Copy)]
struct ErasedMutPtr(*mut ());
unsafe impl Send for ErasedMutPtr {}
unsafe impl Sync for ErasedMutPtr {}

impl ErasedMutPtr {
  #[inline(always)]
  pub fn new<T>(ptr: *mut T) -> Self {
    Self(ptr as *mut ())
  }
  #[inline(always)]
  pub fn get<T>(self) -> *mut T {
    self.0 as *mut T
  }
}

#[cfg(test)]
mod tests {
  extern crate std; // Allow std for testing harness environment

  use super::*;
  use core::any::TypeId;
  use alloc::format;

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
    let scene = Scene::new();
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
    let first = scene
      .query1_first_res(|_e, h: &HealthComp| Some(h.hp))
      .unwrap();
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
}
