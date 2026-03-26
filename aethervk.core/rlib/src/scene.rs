//! Scene graph and Entity-Component-System (ECS) implementation.
//!
//! ## Design
//! - **Backend-agnostic:** The scene representation is independent of the rendering backend.
//! - **Thread-safe:** The main `Scene` struct will be designed for concurrent access (`Send + Sync`).
//! - **Archetype-based ECS:** Inspired by Bevy's architecture for efficient memory layout and querying.
//!   - Entities with the same set of components (an archetype) are stored together in contiguous memory.
//!   - This is a simplified implementation focusing on the core concepts.

use crate::simulation::comet::Comet;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Vec4f32;
use slotmap::{new_key_type, SlotMap};
use spin::RwLock;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::any::{Any, TypeId};
use hashbrown::{HashMap, HashSet};

// === Core ECS Types ===

new_key_type! {
  /// A unique identifier for an entity in the scene.
  pub struct EntityId;
}

/// A marker trait for all components.
/// Components must be `'static + Send + Sync` to be used in the ECS.
pub trait Component: 'static + Send + Sync {}

// === Component Definitions ===

/// Defines the position, rotation, and scale of an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformComponent {
  pub position: Vec3f32,
  /// Stored as a quaternion.
  pub rotation: Vec4f32,
  pub scale: Vec3f32,
}
impl Component for TransformComponent {}

/// Represents a camera in the scene.
pub struct CameraComponent {
  pub projection: Mat4x4f32,
}
impl Component for CameraComponent {}

/// A physically-based mesh loaded from a glTF file.
#[derive(Debug, PartialEq)]
pub struct PhysicalMeshComponent {
  pub mesh: Comet,
}
impl Component for PhysicalMeshComponent {}

/// A marker component for entities that should be rendered.
pub struct Renderable;
impl Component for Renderable {}

/// Represents a 2D texture billboard.
pub struct ImageBillboardComponent {
  pub texture_id: u64,
  pub width: f32,
  pub height: f32,
}
impl Component for ImageBillboardComponent {}

/// A particle emitter, defining the properties of particles to be spawned.
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
}

impl<'a> RenderableDataRef<'a> {
  pub fn index_count(&self) -> u32 {
    match self {
      RenderableDataRef::ImageBillboard(_) => 4,
      RenderableDataRef::PhysicalMesh(mesh) => mesh.mesh.indices.len() as u32,
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

/// A trait for type-erased component storage.
trait ComponentStorage: Send + Sync {
  fn as_any(&self) -> &dyn Any;
  fn as_mut_any(&mut self) -> &mut dyn Any;
  /// Moves a component from this storage to another, using swap_remove for efficiency.
  fn swap_remove_and_push_to(&mut self, index: usize, other: &mut dyn ComponentStorage);
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
struct ComponentMeta {
  dependencies: Vec<TypeId>,
  /// A function pointer to create a new, empty storage for this component type.
  /// TODO: remove double indirection by storing an "inline", Not Sized, StableVector, which uses OS's Virtual Memory system to reserve an enormous amount of space and commit what it needs
  new_storage: fn() -> RwLock<Box<dyn ComponentStorage>>,
  type_name: &'static str,
}

/// An Archetype represents a unique set of component types.
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
#[derive(Clone, Copy)]
struct EntityLocation {
  archetype_index: usize,
  row_index: usize,
}

/// The main scene struct, containing all entities and their components.
pub struct Scene {
  entities: RwLock<SlotMap<EntityId, EntityLocation>>,
  archetypes: RwLock<Vec<Archetype>>,
  component_meta: RwLock<HashMap<TypeId, ComponentMeta>>,
  // TODO: add a hierarchy of EntityIds. Challenge: consistency with entities SlotMap
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
      archetypes: RwLock::new(vec![empty_archetype]),
      component_meta: RwLock::new(HashMap::new()),
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

  pub fn spawn_entity(&self) -> EntityId {
    let mut archetypes = self.archetypes.write();
    let mut entities = self.entities.write();

    let archetype = &mut archetypes[0];
    let entity_location = EntityLocation {
      archetype_index: 0,
      row_index: archetype.entities.len(),
    };

    let entity_id = entities.insert(entity_location);
    archetype.entities.push(entity_id);

    entity_id
  }

  /// Queries for information about all registered components for a given entity.
  ///
  /// This allows checking which components an entity has (is_present = true) versus
  /// which are "filler" slots that could be added (is_present = false).
  ///
  /// Returns `None` if the `entity_id` is invalid.
  pub fn get_entity_components_info(
    &self,
    entity_id: EntityId,
  ) -> Option<Vec<EntityComponentInfo>> {
    let entities = self.entities.read();
    let location = entities.get(entity_id)?;

    let archetypes = self.archetypes.read();
    let archetype = &archetypes[location.archetype_index];
    let entity_component_types = &archetype.component_types;

    let meta = self.component_meta.read();
    let mut info_list = Vec::with_capacity(meta.len());

    for (type_id, component_meta) in meta.iter() {
      info_list.push(EntityComponentInfo {
        type_name: component_meta.type_name,
        is_present: entity_component_types.contains(type_id),
      });
    }

    // For consistent ordering
    info_list.sort_by_key(|info| info.type_name);

    Some(info_list)
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
    assert!(
      type_t1 != type_t2,
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
}
