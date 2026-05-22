//! scene_api module.

use crate::{
  expect_scene, expect_scene_and_entity,
  math::collision::linear_bvh::LinearBVHNode,
  scene::{
    AddComponentError, ColliderComponent, CursorComponent, EntityId, GridComponent,
    KinematicComponent, ParticleEmitterCirclesComponent, PhysicalMeshComponent, Scene,
    SkyComponent, SphereGizmoComponent, SunComponent, TransformComponent,
  },
  simulation_api::{
    SimulationContext,
    structs::{self, SceneContext},
  },
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib as oshal;
use alloc::{string::String, sync::Arc, vec::Vec};
use core::ffi::c_char;
use oshal::math::{
  matrix::mat4::Mat4x4f32,
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};
use spin::RwLock;

impl SimulationContext {
  /// TODO: Document this item
  pub fn raycast_ndc(
    &self,
    scene_id: u64,
    camera_id: u64,
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
        camera_id,
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

  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn spawn_entity(&self, scene_id: u64, name: &str) -> EngineResult<u64> {
    let scene_data = self.scenes.read();
    let active = expect_scene!(scene_data.get_scene(scene_id), "scene_api:spawn_entity");
    let id = active.write().scene.spawn_entity(name);
    Ok(active.write().register_entity(id))
  }

  /// TODO: Document this item
  pub fn remove_entity(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let scene_data = self.scenes.read();
    let (active, entity_id) = expect_scene_and_entity!(
      scene_data.get_scene(scene_id),
      entity,
      "scene_api:remove_entity"
    );
    let mut write_active = active.write();
    write_active.scene.remove_entity(entity_id);

    if write_active.entity_map.remove(&entity).is_some() {
      Ok(())
    } else {
      Err(EngineError::InvalidOperation(
        "scene_api:remove_entity | entity not found",
      ))
    }
  }

  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn get_bvh_nodes<FFI: From<LinearBVHNode<f32>> + Copy>(
    &self,
    scene_id: u64,
    entity: u64,
    count: *mut u32,
  ) -> EngineResult<*mut FFI> {
    let (active, entity_id) =
      expect_scene_and_entity!(self.get_scene(scene_id), entity, "scene_api:get_bvh_nodes");
    let mut ffi_nodes = Vec::new();

    active.read().scene.with_component(entity_id, |mesh: &PhysicalMeshComponent| {
      if let Some(_bvh) = &mesh.mesh.bvh {
        // Linear BVH is gone; returning empty for now
        // TODO: support FFI export of MultiBVH nodes
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

  /// TODO: Document this item
  pub fn free_bvh_nodes<FFI: From<LinearBVHNode<f32>> + Copy + Sized>(ptr: *mut FFI, count: u32) {
    if !ptr.is_null() {
      let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
    }
  }

  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn get_entity_parent(&self, scene_id: u64, entity: u64) -> EngineResult<u64> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_parent");
    let internal_id = scene.read().get_entity(entity).ok_or(EngineError::InvalidOperation(
      "scene_api:get_entity_parent child entity not found",
    ))?;
    let parent_id = scene.read().scene.get_parent(internal_id).ok_or(
      EngineError::InvalidOperation("scene_api:get_entity_parent parent not found"),
    )?;
    // we don't maintain an inverse mapping, so we need to find it manually
    // precondition for unwrap: if entity exists, then the simulation api has its external id.
    Ok(scene.read().entity_map.iter().find(|&(_, v)| *v == parent_id).map(|(ext, _)| *ext).unwrap())
  }
  pub fn destroy_scene(&self, scene_id: u64) -> EngineResult<()> {
    let mut pes_to_destroy = Vec::new();
    if let Some(scene_ctx) = self.scenes.read().get(&scene_id) {
      let read_scene = scene_ctx.read();
      for (&handle, _) in read_scene.presentation_engines.read().iter() {
        pes_to_destroy.push(handle);
      }
    } else {
      return Err(EngineError::InvalidOperation(
        "scene_api:destroy_scene | scene not found",
      ));
    }

    for pe in pes_to_destroy {
      // Ignore errors if the PE was already destroyed or is invalid
      let _ = self.destroy_presentation_engine(scene_id, pe);
    }

    if self.scenes.write().remove(&scene_id).is_some() {
      Ok(())
    } else {
      Err(EngineError::InvalidOperation(
        "scene_api:destroy_scene | scene not found on remove",
      ))
    }
  }

  /// TODO: Document this item
  pub fn create_empty_scene(&self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object(Arc::clone(&self.texture_cache))?;
    let scene_ctx = Arc::new(RwLock::new(
      SceneContext::new_empty(Arc::new(scene), root_entity).with_physics_scene(),
    ));
    Ok(self.scenes.write().insert_scene(scene_ctx))
  }

  /// TODO: Document this item
  pub fn camera_start_pos() -> Vec3f32 {
    Vec3f32::from_components(0.0, -7.07, 7.07)
  }

  // TODO remove
  /// TODO: Document this item
  pub fn create_default_scene(&self) -> EngineResult<u64> {
    let (scene, root_entity) = empty_scene_object(Arc::clone(&self.texture_cache))?;

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

    let mut sun_radius = 0.0696 * crate::simulation::constants::PLANET_VISUAL_SCALE; // 696000 km / 10,000,000 km
    if let Some(asset_dir) = crate::gpu::ASSET_DIR.read().as_ref() {
      let pck_path = alloc::format!("{}/planets/pck00011.tpc", asset_dir);
      if let Some(radii) = crate::simulation::pck::read_body_radii(&pck_path, 10) {
        sun_radius = (radii[0] / crate::simulation::constants::DISTANCE_SCALE_FACTOR) as f32
          * crate::simulation::constants::PLANET_VISUAL_SCALE;
      }
    }

    let sun_scale = sun_radius / 0.6; // Core is 0.6 of the sun volume cube

    let sun_entity = scene.spawn_entity("sun");
    scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(sun_scale, sun_scale, sun_scale),
      },
    )?;
    scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
        radius: 0.6,
      },
    )?;
    scene.set_parent(sun_entity, Some(root_entity));

    let sun_sphere = crate::simulation::comet::generate_uv_sphere(0.6, 64, 64, 1.989e30_f32);
    scene.add_component(
      sun_entity,
      PhysicalMeshComponent {
        asset_path: String::new(),
        mesh: Arc::from(sun_sphere),
        emissive_intensity: 0.9,
        emissive_color: [1.0, 0.35, 0.02],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )?;

    let grid_entity = scene.spawn_entity("grid");
    scene.add_component(
      grid_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene.add_component(grid_entity, GridComponent {})?;
    scene.set_parent(grid_entity, Some(root_entity));

    let scene_ctx_obj = SceneContext::new_empty(Arc::new(scene), root_entity)
      .with_cursor_entity(cursor_entity)?
      .with_sun_entity(sun_entity)?
      .with_grid_entity(grid_entity)?
      .with_sky_entity(sky_entity)?
      .with_physics_scene();
    let scene_ctx = Arc::new(RwLock::new(scene_ctx_obj));

    if let Some(tx) = self.threads.render_thread.tx_opt() {
      let _ = tx.try_send(crate::simulation_api::structs::RenderCommand::GenerateSky);
    }

    Ok(self.scenes.write().insert_scene(scene_ctx))
  }

  /// TODO: Document this item
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

  /// TODO: Document this item
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

  /// TODO: Document this item
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
      active.write().scene.add_component(entity_id, crate::scene::FollowingComponent {})?;
    } else {
      active
        .write()
        .scene
        .remove_component::<crate::scene::FollowingComponent>(entity_id)
        .map_err(|s| EngineError::InvalidOperation(s))?;
    }
    Ok(())
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Comet spawn helper
// ─────────────────────────────────────────────────────────────────────────────

impl SimulationContext {
  /// Atomically spawns the LCA micro-frame entity and the comet mesh entity
  /// as its child, attaching all required components.
  ///
  /// # Arguments
  /// * `scene_id`     – target scene.
  /// * `model_id`     – id returned by `import_model_from_mesh`.
  /// * `entity_name`  – base name; the micro-frame will be named `<name>_microframe`.
  /// * `pos`          – spawn position **in macro-frame units (AU)**.
  /// * `rotation`     – orientation quaternion for the comet mesh (relative to micro-frame).
  /// * `radius_km`    – nucleus radius in km; used to derive the mesh scale.
  /// * `physics_type` – `0` = Static, `1` = Kinematic, `2` = Dynamic.
  ///
  /// Returns `(lca_frame_ext_id, comet_ext_id)` on success.
  pub fn spawn_comet_internal(
    &self,
    scene_id: u64,
    model_id: u64,
    entity_name: &str,
    pos: Vec3f32,
    rotation: Quat,
    radius_km: f32,
    physics_type: u32,
  ) -> EngineResult<(u64, u64)> {
    use crate::scene::ReferenceFrameComponent;
    use aethervk_oshal_rlib::math::matrix::SquareMatrix;

    // ── Resolve mesh from model registry ────────────────────────────────────
    let (path_str, mesh_arc) = {
      let scenes = self.scenes.read();
      let path_str = scenes
        .model_registry
        .get(&model_id)
        .ok_or(EngineError::InvalidOperation("spawn_comet: model not found"))?
        .clone();
      let mesh_arc = scenes
        .mesh_cache
        .get(&path_str)
        .ok_or(EngineError::InvalidOperation("spawn_comet: mesh not in cache"))?;
      (path_str, mesh_arc)
    };

    let scene_ctx_lock = {
      let scenes = self.scenes.read();
      scenes
        .get_scene(scene_id)
        .ok_or(EngineError::InvalidOperation("spawn_comet: scene not found"))?
    };
    let mut scene_ctx = scene_ctx_lock.write();

    // ── LCA micro-frame entity ───────────────────────────────────────────────
    let lca_name = alloc::format!("{}_microframe", entity_name);
    let lca_id = scene_ctx.scene.spawn_entity(&lca_name);

    // Transform: position is in macro-frame coordinates (AU).
    scene_ctx.scene.add_component(
      lca_id,
      TransformComponent {
        position: pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;

    // SOI radius: capped at 1 AU; also capped from below at 0.01 AU so the
    // micro-frame is never degenerate even for a sun-grazing comet.
    let dist_au =
      (pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z()).sqrt();
    let soi_radius_au = dist_au.min(1.0_f32).max(0.01_f32);

    scene_ctx.scene.add_component(
      lca_id,
      ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: soi_radius_au,
        soi_radius: soi_radius_au,
        _padding: 0,
      },
    )?;

    let root = scene_ctx.root_entity;
    scene_ctx.scene.set_parent(lca_id, Some(root));
    let lca_ext_id = scene_ctx.register_entity(lca_id);

    // ── Comet mesh entity (child of LCA frame) ───────────────────────────────
    let comet_id = scene_ctx.scene.spawn_entity(entity_name);

    // Derive uniform scale so that the mesh bounding sphere == radius_km.
    let bounding_sphere = compute_bounding_sphere_radius(&mesh_arc.vertices);
    let mesh_scale = if bounding_sphere > 0.0 { radius_km / bounding_sphere } else { 1.0 };

    scene_ctx.scene.add_component(
      comet_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation,
        scale: Vec3f32::from_components(mesh_scale, mesh_scale, mesh_scale),
      },
    )?;

    scene_ctx.scene.add_component(
      comet_id,
      PhysicalMeshComponent {
        asset_path: path_str,
        mesh: mesh_arc,
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        // sphere_radius is in micro-frame km — matches radius_km directly.
        sphere_radius: radius_km,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )?;

    scene_ctx.scene.add_component(
      comet_id,
      SphereGizmoComponent {
        radius: radius_km,
        subdivisions: 4.0,
        local_frame: Mat4x4f32::identity(),
        is_visible: true,
      },
    )?;

    scene_ctx.scene.add_component(
      comet_id,
      ParticleEmitterCirclesComponent { circles: alloc::vec::Vec::new() },
    )?;

    // Physics-type specific components
    match physics_type {
      0 => {
        // Static: participates as collider, never moves.
        scene_ctx.scene.add_component(
          comet_id,
          ColliderComponent {
            shape: crate::scene::ColliderShape::Sphere { radius: radius_km },
            mass: 0.0,
            restitution: 0.3,
            friction: 0.6,
          },
        )?;
      }
      1 => {
        // Kinematic: collider + velocity-driven by almanac each tick.
        scene_ctx.scene.add_component(
          comet_id,
          ColliderComponent {
            shape: crate::scene::ColliderShape::Sphere { radius: radius_km },
            mass: 0.0,
            restitution: 0.3,
            friction: 0.6,
          },
        )?;
        scene_ctx.scene.add_component(comet_id, KinematicComponent::default())?;
      }
      2 => {
        // Dynamic: full rigid-body physics.
        // KinematicComponent carries velocity/angular-velocity state.
        scene_ctx.scene.add_component(comet_id, KinematicComponent::default())?;
      }
      _ => {}
    }

    scene_ctx.scene.set_parent(comet_id, Some(lca_id));
    let comet_ext_id = scene_ctx.register_entity(comet_id);

    Ok((lca_ext_id, comet_ext_id))
  }

  /// Atomically spawns the LCA micro-frame entity and the static mesh entity
  /// as its child, attaching all required components including ParticleEmitterCirclesComponent.
  ///
  /// # Arguments
  /// * `scene_id`     – target scene.
  /// * `model_id`     – id returned by `import_model_from_mesh`.
  /// * `entity_name`  – base name; the micro-frame will be named `<name>_microframe`.
  /// * `pos`          – spawn position **in macro-frame units (AU)**.
  /// * `rotation`     – orientation quaternion for the mesh (relative to micro-frame).
  /// * `scale`        – scale vector for the mesh.
  ///
  /// Returns `(lca_frame_ext_id, mesh_ext_id)` on success.
  pub fn spawn_static_mesh_internal(
    &self,
    scene_id: u64,
    model_id: u64,
    entity_name: &str,
    pos: Vec3f32,
    rotation: Quat,
    scale: Vec3f32,
  ) -> EngineResult<(u64, u64)> {
    use crate::scene::ReferenceFrameComponent;

    // ── Resolve mesh from model registry ────────────────────────────────────
    let (path_str, mesh_arc) = {
      let scenes = self.scenes.read();
      let path_str = scenes
        .model_registry
        .get(&model_id)
        .ok_or(EngineError::InvalidOperation("spawn_static_mesh: model not found"))?
        .clone();
      let mesh_arc = scenes
        .mesh_cache
        .get(&path_str)
        .ok_or(EngineError::InvalidOperation("spawn_static_mesh: mesh not in cache"))?;
      (path_str, mesh_arc)
    };

    let scene_ctx_lock = {
      let scenes = self.scenes.read();
      scenes
        .get_scene(scene_id)
        .ok_or(EngineError::InvalidOperation("spawn_static_mesh: scene not found"))?
    };
    let mut scene_ctx = scene_ctx_lock.write();

    // ── LCA micro-frame entity ───────────────────────────────────────────────
    let lca_name = alloc::format!("{}_microframe", entity_name);
    let lca_id = scene_ctx.scene.spawn_entity(&lca_name);

    // Transform: position is in macro-frame coordinates (AU).
    scene_ctx.scene.add_component(
      lca_id,
      TransformComponent {
        position: pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;

    // SOI radius: static meshes might not have a massive SOI, we use a generic scale or compute it
    let dist_au =
      (pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z()).sqrt();
    let soi_radius_au = dist_au.min(1.0_f32).max(0.01_f32);

    scene_ctx.scene.add_component(
      lca_id,
      ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: soi_radius_au,
        soi_radius: soi_radius_au,
        _padding: 0,
      },
    )?;

    let root = scene_ctx.root_entity;
    scene_ctx.scene.set_parent(lca_id, Some(root));
    let lca_ext_id = scene_ctx.register_entity(lca_id);

    // ── Mesh entity (child of LCA frame) ───────────────────────────────
    let mesh_id = scene_ctx.scene.spawn_entity(entity_name);

    let bounding_sphere = compute_bounding_sphere_radius(&mesh_arc.vertices);

    scene_ctx.scene.add_component(
      mesh_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation,
        scale,
      },
    )?;

    scene_ctx.scene.add_component(
      mesh_id,
      PhysicalMeshComponent {
        asset_path: path_str,
        mesh: mesh_arc,
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: bounding_sphere * scale.x().max(scale.y()).max(scale.z()),
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      },
    )?;

    scene_ctx.scene.add_component(
      mesh_id,
      crate::scene::SphericalGizmoComponent {
        is_visible: true,
      },
    )?;

    scene_ctx.scene.add_component(
      mesh_id,
      crate::scene::particles::ParticleEmitterCirclesComponent { circles: alloc::vec::Vec::new() },
    )?;

    scene_ctx.scene.set_parent(mesh_id, Some(lca_id));
    let mesh_ext_id = scene_ctx.register_entity(mesh_id);

    Ok((lca_ext_id, mesh_ext_id))
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the radius of the smallest sphere centred at the mesh origin that
/// encloses all vertices. Result is in the same unit as vertex positions
/// (object-space km for comet meshes).
fn compute_bounding_sphere_radius(vertices: &[crate::simulation::comet::Vertex]) -> f32 {
  vertices
    .iter()
    .map(|v| {
      v.position[0] * v.position[0]
        + v.position[1] * v.position[1]
        + v.position[2] * v.position[2]
    })
    .fold(0.0_f32, f32::max)
    .sqrt()
}

// ------------------------- INTERNAL --------------------------------------

// TODO probably to move in scene.rs in rlib
/// TODO: Document this item
pub(crate) fn empty_scene_object(
  texture_cache: alloc::sync::Arc<spin::RwLock<crate::simulation::texture_cache::TextureCache>>,
) -> EngineResult<(Scene, EntityId)> {
  let scene = Scene::new(texture_cache);
  scene.register_all_crate_components();

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

  scene
    .add_component(
      root_entity,
      crate::scene::ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        _padding: 0,
      },
    )
    .map_err(|e| <AddComponentError as Into<EngineError>>::into(e))?;

  Ok((scene, root_entity))
}
