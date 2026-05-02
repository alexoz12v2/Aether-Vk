use crate::math::collision::linear_bvh::LinearBVHNode;
use crate::{
  expect_scene, expect_scene_and_entity,
  scene::{
    AddComponentError, BvhDebugComponent, CameraComponent, CursorComponent, EntityId,
    FollowingComponent, GridComponent, HiddenComponent, MarkersComponent, MeasurementComponent,
    PhysicalMeshComponent, Scene, SelectedComponent, SkyComponent, SunComponent,
    TransformComponent,
  },
  simulation_api::structs::{self, SceneContext},
  simulation_api::SimulationContext,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib as oshal;
use alloc::{string::String, sync::Arc, vec::Vec};
use core::{any::TypeId, ffi::c_char};
use oshal::{
  math::quaternion::Quaternion,
  math::vector::vec3::Vec3f32,
  math::vector::vec4::Quat,
  math::vector::{Vector, Vector3},
};
use spin::{RwLock, RwLockReadGuard};

impl SimulationContext {
  pub fn raycast_ndc(
    &self,
    scene_id: u64,
    ndc_x: f32,
    ndc_y: f32,
  ) -> EngineResult<core::num::NonZero<u64>> {
    let mut task_manager = self.task_manager.write();
    let task_id = task_manager.create_task();
    self
      .threads
      .logic_thread
      .tx()
      .try_send(structs::LogicCommand::RaycastNdc {
        task_id: task_id.get(),
        scene_id,
        ndc_x,
        ndc_y,
      })
      .map_err(|_| {
        task_manager.fail_task(
          task_id.get(),
          alloc::string::String::from("logic thread closed"),
        );
        EngineError::InvalidOperation("scene_api: failed to send raycast command")
      })?;
    Ok(task_id)
  }

  pub fn raycast(
    &self,
    scene_id: u64,
    ro: Vec3f32,
    rd: Vec3f32,
  ) -> EngineResult<core::num::NonZero<u64>> {
    let mut task_manager = self.task_manager.write();
    let task_id = task_manager.create_task();
    self
      .threads
      .logic_thread
      .tx()
      .try_send(structs::LogicCommand::Raycast {
        task_id: task_id.get(),
        scene_id,
        ro,
        rd,
      })
      .map_err(|_| {
        task_manager.fail_task(
          task_id.get(),
          alloc::string::String::from("logic thread closed"),
        );
        EngineError::InvalidOperation("scene_api: failed to send raycast command")
      })?;
    Ok(task_id)
  }

  pub fn spawn_entity(&self, scene_id: u64, name: &str) -> EngineResult<u64> {
    let mut scene_data = self.scenes.write();
    let active = expect_scene!(scene_data.get_scene(scene_id), "scene_api:spawn_entity");
    let id = active.write().scene.spawn_entity(name);
    Ok(active.write().register_entity(id))
  }

  pub fn remove_entity(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let mut scene_data = self.scenes.write();
    let (mut active, entity_id) = expect_scene_and_entity!(
      scene_data.get_scene(scene_id),
      entity,
      "scene_api:remove_entity"
    );
    active.write().scene.remove_entity(entity_id);

    if active.write().entity_map.remove(&entity).is_some() {
      Ok(())
    } else {
      Err(EngineError::InvalidOperation(
        "scene_api:remove_entity | entity not found",
      ))
    }
  }

  pub fn set_parent(&self, scene_id: u64, entity: u64, parent: u64) -> EngineResult<()> {
    let (active, entity_id) =
      expect_scene_and_entity!(self.get_scene(scene_id), entity, "scene_api:set_parent");
    let parent = if parent == 0 {
      None
    } else {
      active.read().get_entity(parent)
    }
    .ok_or(EngineError::InvalidOperation(
      "scene_api:set_parent | parent entity not found",
    ))?;
    active.write().scene.set_parent(entity_id, Some(parent));
    Ok(())
  }

  pub fn get_bvh_nodes<FFI: From<LinearBVHNode<f32>> + Copy>(
    &self,
    scene_id: u64,
    entity: u64,
    count: *mut u32,
  ) -> EngineResult<*mut FFI> {
    let (active, entity_id) =
      expect_scene_and_entity!(self.get_scene(scene_id), entity, "scene_api:get_bvh_nodes");
    let mut ffi_nodes = Vec::new();

    active
      .read()
      .scene
      .with_component(entity_id, |mesh: &PhysicalMeshComponent| {
        if let Some(bvh) = &mesh.mesh.bvh {
          for node in &bvh.nodes {
            let ffi_node = node.clone().into();
            ffi_nodes.push(ffi_node);
          }
        }
      });

    if !count.is_null() {
      unsafe {
        *count = ffi_nodes.len() as u32;
      }
    }

    if ffi_nodes.is_empty() {
      return Ok(core::ptr::null_mut());
    }

    let ptr = ffi_nodes.as_mut_ptr();
    // TODO should this method marked unsafe for this?
    core::mem::forget(ffi_nodes);
    Ok(ptr)
  }

  pub fn free_bvh_nodes<FFI: From<LinearBVHNode<f32>> + Copy + Sized>(ptr: *mut FFI, count: u32) {
    if !ptr.is_null() {
      let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
  }

  pub fn get_entity_count(&self, scene_id: u64) -> EngineResult<u32> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_count");
    Ok(scene.read().entity_map.len() as u32)
  }

  /// Return number of entities copied, and number of missing ones
  pub fn get_entity_ids(&self, scene_id: u64, out_ids: &mut [u64]) -> EngineResult<(u32, u32)> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_ids");
    let map = scene.read();
    let entities_num = map.entity_map.len();
    let buffer_len = out_ids.len();
    let missing = if buffer_len >= entities_num {
      0
    } else {
      (entities_num - out_ids.len()) as u32
    };
    let take_len = core::cmp::min(entities_num, buffer_len) as u32;
    for (i, &id) in map.entity_map.keys().enumerate().take(take_len as usize) {
      out_ids[i] = id;
    }
    Ok((take_len, missing))
  }

  /// Returns number of missing characters in the name
  pub fn get_entity_name(
    &self,
    scene_id: u64,
    entity: u64,
    out_name: &mut [c_char],
  ) -> EngineResult<u32> {
    let (scene, internal_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:get_entity_name"
    );
    // unwrap because if the entity is in there, there's the name.
    let name = scene.read().scene.get_name(internal_id).unwrap();
    let bytes: &[c_char] = bytemuck::cast_slice(name.as_bytes());
    let copy_len = core::cmp::min(bytes.len(), out_name.len());
    let missing: u32 = if out_name.len() >= bytes.len() {
      0
    } else {
      bytes.len() - out_name.len()
    } as _;
    out_name[..copy_len].copy_from_slice(&bytes[..copy_len]);
    out_name[copy_len] = 0;
    Ok(missing)
  }

  pub fn get_entity_parent(&self, scene_id: u64, entity: u64) -> EngineResult<u64> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_parent");
    let internal_id = scene
      .read()
      .get_entity(entity)
      .ok_or(EngineError::InvalidOperation(
        "scene_api:get_entity_parent child entity not found",
      ))?;
    let parent_id =
      scene
        .read()
        .scene
        .get_parent(internal_id)
        .ok_or(EngineError::InvalidOperation(
          "scene_api:get_entity_parent parent not found",
        ))?;
    // we don't maintain an inverse mapping, so we need to find it manually
    // precondition for unwrap: if entity exists, then the simulation api has its external id.
    Ok(
      scene
        .read()
        .entity_map
        .iter()
        .find(|&(_, v)| *v == parent_id)
        .map(|(ext, _)| *ext)
        .unwrap(),
    )
  }
  pub fn create_empty_scene(&self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object()?;
    let camera_entity = scene.add_camera("camera", Self::camera_start_pos(), root_entity)?;
    let scene_ctx = Arc::new(RwLock::new(
      SceneContext::new_empty(Arc::new(scene), root_entity)
        .with_active_camera_entity(camera_entity)?,
    ));
    Ok(self.scenes.write().insert_scene(scene_ctx))
  }

  pub fn camera_start_pos() -> Vec3f32 {
    Vec3f32::from_components(0.0, -400.0, 0.0)
  }

  pub fn create_default_scene(&self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object()?;

    let camera_entity = scene.add_camera("camera", Self::camera_start_pos(), root_entity)?;

    let sky_entity = scene.spawn_entity("sky");
    scene.add_component(sky_entity, SkyComponent {})?;
    scene.set_parent(sky_entity, Some(root_entity));

    let cursor_entity = scene.spawn_entity("cursor");
    scene.add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(0.02, 0.02, 0.02),
      },
    )?;
    scene.add_component(cursor_entity, CursorComponent {})?;
    scene.set_parent(cursor_entity, Some(root_entity));

    let sun_entity = scene.spawn_entity("sun");
    scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
      },
    )?;
    scene.set_parent(sun_entity, Some(root_entity));

    let mut sun_radius = 0.0696 * crate::simulation::constants::PLANET_VISUAL_SCALE; // 696000 km / 10,000,000 km
    if let Some(asset_dir) = crate::gpu::ASSET_DIR.read().as_ref() {
      let pck_path = alloc::format!("{}/planets/pck00011.tpc", asset_dir);
      if let Some(radii) = crate::simulation::pck::read_body_radii(&pck_path, 10) {
        sun_radius = (radii[0] / crate::simulation::constants::DISTANCE_SCALE_FACTOR) as f32 * crate::simulation::constants::PLANET_VISUAL_SCALE;
      }
    }

    let sun_sphere = crate::simulation::comet::generate_uv_sphere(sun_radius, 64, 64);
    scene.add_component(
      sun_entity,
      PhysicalMeshComponent {
        asset_path: String::new(),
        mesh: Arc::from(sun_sphere),
        emissive_intensity: 0.9,
        emissive_color: [1.0, 0.35, 0.02],
      },
    )?;

    let grid_entity = scene.spawn_entity("grid");
    scene.add_component(grid_entity, GridComponent {})?;
    scene.set_parent(grid_entity, Some(root_entity));

    let scene_ctx = Arc::new(RwLock::new(
      SceneContext::new_empty(Arc::new(scene), root_entity)
        .with_active_camera_entity(camera_entity)?
        .with_cursor_entity(cursor_entity)?
        .with_sun_entity(sun_entity)?
        .with_grid_entity(grid_entity)?
        .with_sky_entity(sky_entity)?
        .with_physics_scene(),
    ));

    if let Some(tx) = self.threads.render_thread.tx_opt() {
      let _ = tx.try_send(crate::simulation_api::structs::RenderCommand::GenerateSky);
    }

    Ok(self.scenes.write().insert_scene(scene_ctx))
  }

  pub fn set_entity_visibility(
    &self,
    scene_id: u64,
    entity: u64,
    visible: bool,
  ) -> EngineResult<()> {
    let (scene, id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_visibility"
    );
    if visible {
      scene
        .write()
        .scene
        .remove_component::<crate::scene::HiddenComponent>(id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
    } else {
      scene
        .write()
        .scene
        .add_component(id, crate::scene::HiddenComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    }
    Ok(())
  }

  pub fn set_entity_selected(
    &self,
    scene_id: u64,
    entity: u64,
    selected: bool,
  ) -> EngineResult<()> {
    let (scene, id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_selected"
    );
    if selected {
      scene
        .write()
        .scene
        .add_component(id, crate::scene::SelectedComponent {})
        .map_err(|e| <AddComponentError as Into<EngineError>>::into(e.into()))?;
    } else {
      scene
        .write()
        .scene
        .remove_component::<crate::scene::SelectedComponent>(id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
    }
    Ok(())
  }

  pub fn set_entity_following(
    &self,
    scene_id: u64,
    entity: u64,
    following: bool,
  ) -> EngineResult<()> {
    let (active, entity_id) = expect_scene_and_entity!(
      self.get_scene(scene_id),
      entity,
      "scene_api:set_entity_following"
    );
    if following {
      active
        .write()
        .scene
        .add_component(entity_id, crate::scene::FollowingComponent {})?;
    } else {
      active
        .write()
        .scene
        .remove_component::<crate::scene::FollowingComponent>(entity_id)
        .map_err(|s| EngineError::InvalidOperation(s))?;
    }
    Ok(())
  }

  // ------------------------- INTERNAL --------------------------------------
  fn raycast_internal(
    &self,
    _scene: RwLockReadGuard<SceneContext>,
    _ro: Vec3f32,
    _dir: Vec3f32,
    _out_hit_entity: *mut u64,
    _out_px: *mut f32,
    _out_py: *mut f32,
    _out_pz: *mut f32,
  ) -> EngineResult<bool> {
    // This is now redundant as it's implemented in logic_thread.rs via structs.rs
    Ok(false)
  }
}

// TODO probably to move in scene.rs in rlib
pub(crate) fn empty_scene_object() -> EngineResult<(Scene, EntityId)> {
  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SkyComponent>(&[]);
  scene.register_component::<GridComponent>(&[]);
  scene.register_component::<MarkersComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SelectedComponent>(&[]);
  scene.register_component::<FollowingComponent>(&[]);
  scene.register_component::<HiddenComponent>(&[]);
  scene.register_component::<BvhDebugComponent>(&[]);
  scene.register_component::<MeasurementComponent>(&[]);
  scene.register_component::<crate::scene::ImageBillboardComponent>(&[TypeId::of::<
    TransformComponent,
  >()]);
  scene.register_component::<crate::scene::GizmoComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<crate::scene::ParticleStateComponent>(&[TypeId::of::<
    TransformComponent,
  >()]);
  scene.register_component::<crate::scene::AlmanacPlanet>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<crate::scene::ParticleSystemComponent>(&[TypeId::of::<
    TransformComponent,
  >()]);

  let root_entity = scene.spawn_entity("root");
  scene
    .add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

  Ok((scene, root_entity))
}
