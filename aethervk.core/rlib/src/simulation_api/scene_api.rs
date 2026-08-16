//! scene_api module.

use crate::{
  expect_scene, expect_scene_and_entity,
  scene::{EntityId, Scene, SphereGizmoComponent, StaticMeshComponent, TransformComponent},
  simulation_api::{
    SimulationContext,
    structs::{self, SceneContext},
  },
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{self as oshal, math::vector::vec3f64::DVec3};
use alloc::{sync::Arc, vec::Vec};
use core::ffi::c_char;
use oshal::math::{
  matrix::mat4::Mat4x4f32,
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};
use parking_lot::RwLock;

pub struct SceneReturn {
  pub scene_id: u64,
  pub comet_body: u64,
  pub earth_body: u64,
}

impl SimulationContext {
  pub fn spawn_entity(&self, scene_id: u64, name: &str) -> EngineResult<u64> {
    let scene_data = self.scenes.read();
    let active = expect_scene!(scene_data.get_scene(scene_id), "scene_api:spawn_entity");
    // Use a single write guard to atomically spawn + register the entity,
    // avoiding the TOCTOU gap that the previous two-write-call pattern had.
    let mut guard = active.write();
    let id = guard.scene.spawn_entity(name);
    Ok(EntityId::as_ffi(&id))
  }

  pub fn remove_entity(&self, scene_id: u64, entity: u64) -> EngineResult<()> {
    let scene_data = self.scenes.read();
    let (active, entity_id) = expect_scene_and_entity!(
      scene_data.get_scene(scene_id),
      entity,
      "scene_api:remove_entity"
    );
    let write_active = active.write();
    write_active.scene.remove_entity(entity_id);

    Ok(())
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

  pub fn get_entity_count(&self, scene_id: u64) -> EngineResult<u32> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_count");
    Ok(scene.read().scene.entity_count() as u32)
  }

  /// Return number of entities copied, and number of missing ones
  pub fn get_entity_ids(&self, scene_id: u64, out_ids: &mut [u64]) -> EngineResult<(u32, u32)> {
    let scene = expect_scene!(self.get_scene(scene_id), "scene_api:get_entity_ids");
    let map = scene.read();
    let entities_num = map.scene.entity_count();
    let buffer_len = out_ids.len();
    let missing = if buffer_len >= entities_num {
      0
    } else {
      (entities_num - out_ids.len()) as u32
    };
    let take_len = core::cmp::min(entities_num, buffer_len) as u32;
    let mut i = 0;
    map.scene.for_each_entity(|id| {
      out_ids[i] = EntityId::as_ffi(&id);
      i += 1;
    });

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
    let internal_id = scene.read().get_entity(entity).ok_or(EngineError::InvalidOperation(
      "scene_api:get_entity_parent child entity not found",
    ))?;
    let parent_id = scene.read().scene.get_parent(internal_id).ok_or(
      EngineError::InvalidOperation("scene_api:get_entity_parent parent not found"),
    )?;

    Ok(EntityId::as_ffi(&parent_id))
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

  pub fn create_empty_scene(
    &self,
    spawn_fallback_camera: bool,
    start_epoch: hifitime::Epoch,
    end_epoch: hifitime::Epoch,
  ) -> EngineResult<u64> {
    self
      .create_empty_scene2(spawn_fallback_camera, start_epoch, end_epoch)
      .map(|r| r.scene_id)
  }

  pub fn create_empty_scene2(
    &self,
    spawn_fallback_camera: bool,
    start_epoch: hifitime::Epoch,
    end_epoch: hifitime::Epoch,
  ) -> EngineResult<SceneReturn> {
    // if start_epoch comes after end or if time interval is less than a day, then refuse
    if start_epoch >= end_epoch || end_epoch - start_epoch < hifitime::Duration::from_days(1 as _) {
      return Err(EngineError::InvalidOperation(
        "end_epoch and start_epoch should be at least 1 day apart",
      ));
    }

    let (scene, root_entity) = {
      let scene = Scene::new(Arc::clone(&self.texture_cache));
      scene.register_all_crate_components();
      let root_entity = scene.spawn_entity("Root");
      (scene, root_entity)
    };

    let create_subtree =
      |name: &str, is_comet: bool, add_gizmo: bool| -> crate::simulation_api::structs::SubtreeEntities {
        const AU_TO_KM: f32 = 149_597_870.7;
        // subtree root (AU frame, child of root)
        let subtree = scene.spawn_entity(&alloc::format!("{}_subtree", name));
        scene.set_parent(subtree, Some(root_entity));
        // will be modified on "forced repositioning" (repositioning implemented in logic_thread
        // per-frame FRAME SHIFT, and explicitly via reposition::force_reposition at scene creation
        // or epoch range change)
        scene.add_component(subtree, crate::scene::TransformComponent::default());
        scene.add_component(
          subtree,
          crate::scene::ReferenceFrameComponent {
            frame_type: crate::scene::ReferenceFrameType::Micro,
            scale: AU_TO_KM,
            soi_radius: 1.0,
            depth_layer: 1,
          },
        );

        // body entity (from here on out everything in km)
        let body = scene.spawn_entity(&alloc::format!("{}_body", name));
        scene.set_parent(body, Some(subtree));
        // note for comets: the local frame represents the axis set by the user theorizing the
        // nucleus model. Movement is expressed in global units (km, heliocentric) in double
        // precision (see logic_thread)
        scene.add_component(body, crate::scene::TransformComponent::default());
        if add_gizmo {
          scene.add_component(
            body,
            crate::scene::SphericalGizmoComponent { is_visible: true },
          );
        }
        // body will see AlmanacPlanet added once the relevant almanac files are confirmed loaded.
        if is_comet {
          scene.add_component(body, crate::scene::CometMarkerComponent {});
        } else {
          scene.add_component(body, crate::scene::PlanetMarkerComponent {});
        }

        // orbit entity — MUST be a child of root_entity (depth_layer=0) so that its AU-scale
        // TrajectoryComponent control points are rendered in the heliocentric frame.
        // If parented to the Micro-frame subtree (depth_layer=1), the renderer's RTE path would
        // multiply AU control points by AU_TO_KM (~1.5e8), placing them far off-screen.
        let orbit = scene.spawn_entity(&alloc::format!("{}_orbit", name));
        scene.set_parent(orbit, Some(root_entity));
        scene.add_component(orbit, crate::scene::TransformComponent::default());

        crate::simulation_api::structs::SubtreeEntities { subtree, body, orbit }
      };

    // 1. Cursor Entity
    let cursor_entity = scene.spawn_entity("cursor");
    scene.add_component(
      cursor_entity,
      crate::scene::HighResTransformComponent {
        position: DVec3::zero(),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(0.02, 0.02, 0.02),
      },
    )?;
    scene.add_component(cursor_entity, crate::scene::CursorComponent {})?;
    scene.set_parent(cursor_entity, Some(root_entity));

    // 2. Sun Entity
    let sun_entity = scene.spawn_entity("sun");
    scene.set_parent(sun_entity, Some(root_entity));
    scene.add_component(
      sun_entity,
      crate::scene::TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(0.0046524726, 0.0046524726, 0.0046524726),
      },
    )?;
    scene.add_component(
      sun_entity,
      crate::scene::SunComponent {
        resolution: (2048, 2048, 1),
        radius: 0.0046524726,
      },
    )?;

    // 3. Camera Entity & 4. Sky Entity
    // purely for testing
    if spawn_fallback_camera {
      let home_position = Vec3f32::from_components(0.0115, 0.0115, 0.0115);
      let camera_entity = scene.spawn_entity("camera");
      scene.set_parent(camera_entity, Some(root_entity));

      scene.add_component(
        camera_entity,
        crate::scene::HighResTransformComponent {
          position: aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
            home_position.x() as f64,
            home_position.y() as f64,
            home_position.z() as f64,
          ),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )?;
      scene.add_component(
        camera_entity,
        crate::scene::CameraComponent {
          projection: crate::scene::CameraProjection::Perspective {
            fov: 60.0_f32.to_radians(),
            aspect_ratio: 16.0 / 9.0,
            near: 0.00001,
            far: 1000.0,
          },
          focus_distance: 1.0,
        },
      )?;
    }

    let sky_entity = scene.spawn_entity("sky");
    scene.set_parent(sky_entity, Some(root_entity));
    scene.add_component(sky_entity, crate::scene::SkyComponent {})?;
    let sky_id = Some(sky_entity);

    let time_state = {
      use aethervk_oshal_rlib::os::time::v2::SimSpeed;
      let scene_data = self.scenes.read();
      let scene_id = scene_data.next_scene_id();
      let time_manager = oshal::os::time::v2::TimeManager::new(
        start_epoch,
        end_epoch,
        SimSpeed::Paused,
        super::MAX_UNSCALED_DELTA_MS,
      );
      let time_state = time_manager.state.clone();
      scene_data.time_managers.insert(scene_id, time_manager);
      time_state
    };

    // ── Create planet Earth and comet subtree hierarchies ────────────────────────────────────
    // Must be called BEFORE Arc::new(scene) on the next line, because the closure borrows `scene`
    // by shared reference and Rust NLL requires the last use of the borrow to precede the move.
    let earth = create_subtree("Earth", false, false);
    let comet = create_subtree("Comet", true, true);
    // Drop the closure explicitly so the borrow of `scene` ends here.
    let _ = create_subtree;

    let mut scene_ctx_obj = SceneContext::new_empty(Arc::new(scene), root_entity, time_state);
    scene_ctx_obj.cursor_entity = Some(cursor_entity);
    scene_ctx_obj.sun_entity = Some(sun_entity);
    if let Some(s) = sky_id {
      scene_ctx_obj.sky_entity = Some(s);
    }

    let scene_ctx = Arc::new(RwLock::new(scene_ctx_obj));

    if let Some(tx) = self.threads.render_thread.tx_opt() {
      let _ = tx.try_send(crate::simulation_api::structs::RenderCommand::GenerateSky);
    }


    // Store entity IDs on SceneContext before insert_scene.
    // We hold a write lock here; insert_scene will assert strong_count == 1, which is satisfied
    // because this write guard does not increment the Arc count — it borrows through it.
    {
      let mut guard = scene_ctx.write();
      guard.earth = Some(earth);
      guard.comet = Some(comet);
    }

    // ── Move insert_scene before Earth init so we have scene_id for trajectory dispatch ──────
    // The user confirmed that holding a write lock on the newly inserted SceneContext during init
    // is safe (no other thread has the scene_id yet, so no concurrent access is possible).
    let scene_id = self.scenes.write().insert_scene(scene_ctx);

    // ── Earth Initialization ──────────────────────────────────────────────────────────────────
    //
    // Earth requires two almanac files to be fully operational:
    //   1. `de442.bsp`  — planetary SPK ephemeris for Earth's heliocentric position (NAIF ID 399)
    //   2. `earth_latest_high_prec.bpc` — high-precision Binary PCK for Earth's rotation
    //
    // avkSimulationContext_startup now loads both synchronously from ASSET_DIR before calling
    // create_empty_scene2, so can_move_earth should be true on the first call.
    //
    // If `can_move_earth` is false (e.g. asset path not set, or files not found), Earth_body
    // is left at identity with no AlmanacPlanet component. The logic_thread will not drive it.
    let can_move_earth = {
      let logic_state = self.logic_state.read();
      let has_de442 = logic_state
        .almanac_data
        .file_names
        .iter()
        .any(|f| f.contains("de442.bsp"));
      let has_earth_bpc = logic_state
        .almanac_data
        .file_names
        .iter()
        .any(|f| f.contains("earth_latest_high_prec.bpc"));
      has_de442 && has_earth_bpc
    };

    if can_move_earth {
      // STEP 1 — Attach AlmanacPlanet. This causes logic_thread's per-frame
      // query3(PlanetMarkerComponent + AlmanacPlanet) sweep to insert earth.body into
      // cartesian_state_cache and begin driving it on the next simulation tick.
      let planet = crate::scene::AlmanacPlanet::new(
        anise::constants::celestial_objects::EARTH, // NAIF ID 399
        86400.0 * 0.99726968_f64,                  // sidereal rotation period (seconds)
        3.986004418e5_f32,                          // Earth GM (km³/s²)
      );
      {
        let scenes_guard = self.scenes.read();
        if let Some(scene_arc) = scenes_guard.get_scene(scene_id) {
          let scene_guard = scene_arc.read();
          let _ = scene_guard.scene.add_component(earth.body, planet);

          // STEP 2 — Forced repositioning at start_epoch.
          // Snaps Earth_subtree (AU) and Earth_body (km residual) to the almanac position.
          {
            let logic_state = self.logic_state.read();
            if let Err(e) = crate::simulation_api::reposition::force_reposition(
              &scene_guard.scene,
              earth.subtree,
              earth.body,
              &logic_state.almanac_data,
              &planet,
              start_epoch,
            ) {
              aethervk_oshal_rlib::log!(
                "[scene_api] Earth force_reposition failed: {}",
                e
              );
            }
          }
        }
      }

      // STEP 3 — Queue trajectory generation for the full calendar year containing start_epoch.
      // UpdateTrajectoryForSpk is an async command processed on the thread pool; earth.orbit
      // will gain a TrajectoryComponent once complete.
      let year = crate::simulation_api::reposition::year_of_epoch(start_epoch);
      let (traj_start_tai, traj_end_tai) =
        crate::simulation_api::reposition::full_year_tai_seconds(year);
      let _ = self.threads.logic_thread.tx().try_send(
        crate::simulation_api::structs::LogicCommand::UpdateTrajectoryForSpk {
          task_id: 0,
          scene_id,
          entity_id: crate::scene::EntityId::as_ffi(&earth.orbit),
          spk_id: anise::constants::celestial_objects::EARTH,
          start_epoch_tai_sec: traj_start_tai,
          end_epoch_tai_sec: traj_end_tai,
          sample_step_days: 1.0,
        },
      );
      // Record the trajectory year so SetEpochRange can skip redundant rebuilds.
      {
        let scenes_guard = self.scenes.read();
        if let Some(scene_arc) = scenes_guard.get_scene(scene_id) {
          scene_arc.write().earth_orbit_year = Some(year);
        }
      }
    } else {
      aethervk_oshal_rlib::log!(
        "[scene_api] Earth almanac data not ready — Earth_body at origin. \
         avkSimulationContext_startup will retry loading de442.bsp and \
         earth_latest_high_prec.bpc from ASSET_DIR before scene creation."
      );
    }

    // ── Comet Default Placement — 1 AU along +X ──────────────────────────────────────────────
    //
    // The comet subtree is placed at (1, 0, 0) AU (heliocentric EclipJ2000). The comet body
    // stays at local origin within the km-frame subtree. This is the "no SPK loaded" default.
    // Once a comet SPK is loaded and validated (via LoadAlmanac → InitComet), force_reposition
    // will snap the subtree to the ephemeris position and AlmanacPlanet will be attached.
    {
      let scenes_guard = self.scenes.read();
      if let Some(scene_arc) = scenes_guard.get_scene(scene_id) {
        let scene_guard = scene_arc.read();
        let _ = scene_guard.scene.with_component_mut(
          comet.subtree,
          |t: &mut crate::scene::TransformComponent| {
            t.position =
              aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 0.0, 0.0);
          },
        );
      }
    }

    Ok(SceneReturn {
      scene_id,
      comet_body: crate::scene::EntityId::as_ffi(&comet.body),
      earth_body: crate::scene::EntityId::as_ffi(&earth.body),
    })
  }


  #[cfg(test)]
  pub fn create_default_scene(&self, spawn_fallback_camera: bool) -> EngineResult<u64> {
    oshal::log!("create_default_scene: START");
    let (scene, root_entity) = empty_scene_object(Arc::clone(&self.texture_cache))?;
    oshal::log!(
      "create_default_scene: empty_scene_object OK, root={:?}",
      root_entity
    );
    // since this is a testing function I can make up some values
    let start_epoch =
      hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC);
    let end_epoch = start_epoch + hifitime::Duration::from_days(2 as _);

    let sun_radius = 0.004652472; // Solar radius in AU
    let sun_scale = sun_radius / 0.6; // UV sphere is radius 0.6, so scale matches extent

    let sun_entity = scene.spawn_entity("sun");
    oshal::log!("create_default_scene: sun spawned={:?}", sun_entity);
    scene.add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(sun_scale, sun_scale, sun_scale),
      },
    )?;
    oshal::log!("create_default_scene: sun TransformComponent OK");
    scene.add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
        radius: 0.6,
      },
    )?;
    oshal::log!("create_default_scene: sun SunComponent OK");
    scene.add_component(
      sun_entity,
      crate::scene::ForceEmitterComponent::Gravity {
        mu: 1.3271244e11_f32,
        beta: 0.0,
      },
    )?;
    oshal::log!("create_default_scene: sun ForceEmitterComponent OK");
    scene.set_parent(sun_entity, Some(root_entity));

    let sun_sphere = crate::simulation::comet::generate_uv_sphere(0.6, 64, 64, 1.989e30_f32, false);
    oshal::log!("create_default_scene: sun_sphere generated, adding PhysicalMeshComponent");
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
        rotational_model: None,
      },
    )?;
    oshal::log!("create_default_scene: sun PhysicalMeshComponent OK");

    let mut camera_id = None;
    let mut sky_id = None;
    if spawn_fallback_camera {
      let camera_entity = scene.spawn_entity("camera");
      // Use exact start position requested by the user
      let pos_x = 0.010429309357456616;
      let pos_y = 0.010962580663326662;
      let pos_z = 0.007890014217773569;

      let rot = aethervk_oshal_rlib::math::vector::vec4::Quat::from_components(
        0.24757917,
        -0.098841526,
        -0.35735834,
        0.8951145,
      );

      scene.add_component(
        camera_entity,
        crate::scene::HighResTransformComponent {
          position: aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
            pos_x, pos_y, pos_z,
          ),
          rotation: rot,
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )?;
      scene.add_component(
        camera_entity,
        crate::scene::CameraComponent::new_persp(45.0, 1.0, 0.00001, 1000.0),
      )?;
      scene.set_parent(camera_entity, Some(root_entity));
      oshal::log!("create_default_scene: camera OK");

      let sky_entity = scene.spawn_entity("sky");
      scene.add_component(sky_entity, SkyComponent {})?;
      scene.add_component(
        sky_entity,
        TransformComponent {
          position: Vec3f32::zero(),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )?;
      scene.set_parent(sky_entity, Some(root_entity));
      oshal::log!("create_default_scene: sky OK");

      camera_id = Some(camera_entity);
      sky_id = Some(sky_entity);
    }

    let grid_entity = scene.spawn_entity("grid");
    scene.add_component(
      grid_entity,
      TransformComponent {
        position: Vec3f32::zero(),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;
    scene.add_component(grid_entity, GridComponent {})?;
    scene.set_parent(grid_entity, Some(root_entity));
    oshal::log!("create_default_scene: grid OK");

    let cursor_entity = scene.spawn_entity("cursor");
    scene.add_component(
      cursor_entity,
      crate::scene::HighResTransformComponent {
        position: DVec3::zero(),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(0.02, 0.02, 0.02),
      },
    )?;
    scene.add_component(cursor_entity, CursorComponent {})?;
    scene.set_parent(cursor_entity, Some(root_entity));
    oshal::log!("create_default_scene: cursor OK");

    let time_scale = {
      use oshal::os::time::v2::{SimSpeed, TimeManager};
      let scene_data = self.scenes.read();
      let time_manager = TimeManager::new(
        start_epoch,
        end_epoch,
        SimSpeed::Paused,
        super::MAX_UNSCALED_DELTA_MS,
      );
      let scene_id = scene_data.next_scene_id();
      let time_scale = time_manager.state.clone();
      scene_data.time_managers.insert(scene_id, time_manager);
      time_scale
    };

    let mut scene_ctx_obj = SceneContext::new_empty(Arc::new(scene), root_entity, time_scale);
    oshal::log!("create_default_scene: with_physics_scene OK, calling with_cursor_entity");
    let mut scene_ctx_obj = scene_ctx_obj.with_cursor_entity(cursor_entity)?;
    oshal::log!("create_default_scene: with_cursor_entity OK");
    let mut scene_ctx_obj = scene_ctx_obj.with_sun_entity(sun_entity)?;
    oshal::log!("create_default_scene: with_sun_entity OK");
    let mut scene_ctx_obj = scene_ctx_obj.with_grid_entity(grid_entity)?;
    oshal::log!("create_default_scene: with_grid_entity OK");
    if let Some(s) = sky_id {
      scene_ctx_obj = scene_ctx_obj.with_sky_entity(s)?;
      oshal::log!("create_default_scene: with_sky_entity OK");
    }
    if let Some(c) = camera_id {
      scene_ctx_obj.register_entity(c);
    }

    let scene_ctx = Arc::new(RwLock::new(scene_ctx_obj));

    if let Some(tx) = self.threads.render_thread.tx_opt() {
      let _ = tx.try_send(crate::simulation_api::structs::RenderCommand::GenerateSky);
    }

    Ok(self.scenes.write().insert_scene(scene_ctx))
  }

  /// Shows or hides an entity **and all of its descendants** (BFS).
  ///
  /// Dispatches a [`LogicCommand::SetEntityVisibility`] to the logic thread instead of
  /// spin-waiting for a write lock. This prevents the logic_thread from hanging when
  /// it holds the scene read lock during a simulation tick while this is called from
  /// an FFI/UI thread. The command is processed by the logic thread between ticks.
  pub fn set_entity_visibility(
    &self,
    scene_id: u64,
    entity: u64,
    visible: bool,
  ) -> EngineResult<()> {
    self
      .threads
      .logic_thread
      .tx()
      .try_send(structs::LogicCommand::SetEntityVisibility {
        scene_id,
        entity,
        visible,
      })
      .map_err(|_| {
        EngineError::InvalidOperation("scene_api:set_entity_visibility | logic thread closed")
      })
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
  // TODO rewrite completely
  pub fn spawn_comet_internal(
    &self,
    scene_id: u64,
    model_id: u64,
    entity_name: &str,
    pos: Vec3f32,
    rotation: Quat,
    radius_km: f32,
    mass_kg: f32,
    physics_type: u32,
    naif_id: i32,
    rotational_model: Option<crate::scene::BodyRotationalModel>,
    angular_velocity: Vec3f32,
  ) -> EngineResult<(u64, u64)> {
    use crate::scene::ReferenceFrameComponent;
    use aethervk_oshal_rlib::math::matrix::SquareMatrix;

    // ── Resolve mesh from model registry ────────────────────────────────────
    let (path_str, mesh_arc) = {
      let scenes = self.scenes.read();
      let path_str = scenes
        .model_registry
        .get(&model_id)
        .ok_or(EngineError::InvalidOperation(
          "spawn_comet: model not found",
        ))?
        .clone();
      let mesh_arc = scenes.mesh_cache.get(&path_str).ok_or(EngineError::InvalidOperation(
        "spawn_comet: mesh not in cache",
      ))?;
      (path_str, mesh_arc)
    };

    let scene_ctx_lock = {
      let scenes = self.scenes.read();
      scenes.get_scene(scene_id).ok_or(EngineError::InvalidOperation(
        "spawn_comet: scene not found",
      ))?
    };
    let mut scene_ctx = scene_ctx_lock.write();

    // ── LCA micro-frame entity ───────────────────────────────────────────────
    let lca_name = alloc::format!("{}_microframe", entity_name);
    let lca_id = scene_ctx.scene.spawn_entity(&lca_name);

    // SOI radius: bounded by the distance from the Sun's surface to the comet.
    // At origin (static spawn): clamped to the Sun's radius.
    // SUN_RADIUS_AU = 0.00465 AU (695,700 km / 149,597,870.7 km·AU⁻¹)
    const SUN_RADIUS_AU: f32 = 0.0046524726_f32;
    const KM_PER_AU: f32 = 149_597_870.7_f32;
    let dist_au = (pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z()).sqrt();
    let soi_radius_au = (dist_au - SUN_RADIUS_AU).max(SUN_RADIUS_AU);

    // Transform: position is in macro-frame coordinates (AU).
    // Scale is (1,1,1) — the frame extent is driven by soi_radius,
    // not by TransformComponent::scale.
    scene_ctx.scene.add_component(
      lca_id,
      TransformComponent {
        position: pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;

    // ReferenceFrameComponent::scale = AU/km (the factor that converts
    // micro-frame km to macro-frame AU).  This is the ONLY correct value
    // for physics: the compute shaders do
    //   r_local_km = (em_pos_AU − center_AU) / scale
    // and micro_to_macro does
    //   p_AU = center_AU + p_km * scale.
    scene_ctx.scene.add_component(
      lca_id,
      ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: 1.0 / KM_PER_AU,
        soi_radius: soi_radius_au,
        depth_layer: 1,
      },
    )?;

    let root = scene_ctx.root_entity;
    scene_ctx.scene.set_parent(lca_id, Some(root));

    // ── Comet mesh entity (child of LCA frame) ───────────────────────────────
    let comet_id = scene_ctx.scene.spawn_entity(entity_name);

    // Derive uniform scale: everything inside the micro-frame is in km.
    // `bounding_sphere` is in the mesh's own vertex units (whatever the GLTF
    // stores — metres, km, etc).  We want the rendered mesh to have radius
    // == `radius_km` in the micro-frame's local km space.
    //
    //   mesh_scale = radius_km / bounding_sphere
    //
    // The rendering pipeline's `get_relative_transform` will then multiply
    // by `frame.scale` (AU/km) to produce the correct macro-frame size.
    // Always use vertex-based bounding sphere for mesh scale so that the rendered
    // size is exactly `radius_km` regardless of whether a BVH is present.
    //
    // DO NOT use bvh.nodes[0].bound.half_extents.length(): the BVH root AABB is
    // axis-aligned and overestimates the sphere by a factor of sqrt(3) for
    // axis-aligned shapes, causing the mesh to appear ~57% smaller than intended.
    let bounding_sphere = compute_bounding_sphere_radius(&mesh_arc.vertices);

    let mesh_scale = if bounding_sphere > 0.0 {
      radius_km / bounding_sphere
    } else {
      1.0
    };

    oshal::log!(
      "spawn_comet: dist_au={:.4} soi_au={:.6} bounding_sphere={:.3} radius_km={:.3} mesh_scale={:.8} frame_scale={:.10e}",
      dist_au,
      soi_radius_au,
      bounding_sphere,
      radius_km,
      mesh_scale,
      1.0 / KM_PER_AU
    );

    let sim_local_rotation = Quat::identity();

    scene_ctx.scene.add_component(
      comet_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: rotation * sim_local_rotation,
        scale: Vec3f32::from_components(mesh_scale, mesh_scale, mesh_scale),
      },
    )?;

    scene_ctx.scene.add_component(
      comet_id,
      StaticMeshComponent {
        asset_path: path_str,
        mesh: mesh_arc,
        emissive_color: [0.0, 0.0, 0.0, 0.0],
        is_visible: true,
      },
    )?;

    scene_ctx.scene.add_component(
      comet_id,
      SphereGizmoComponent {
        // radius is in LOCAL (native mesh vertex) units.  The entity's
        // TransformComponent.scale = mesh_scale = radius_km / bounding_sphere
        // is baked into the model matrix by the gizmo renderer, so the rendered
        // sphere radius = bounding_sphere × mesh_scale = radius_km. ✓
        radius: bounding_sphere,
        subdivisions: 4.0,
        local_frame: Mat4x4f32::identity(),
        is_visible: true,
      },
    )?;

    scene_ctx.scene.set_parent(comet_id, Some(lca_id));

    Ok((EntityId::as_ffi(&lca_id), EntityId::as_ffi(&comet_id)))
  }

  pub fn spawn_trajectory_internal(
    &self,
    scene_id: u64,
    parent_entity: u64,
    entity_name: &str,
    trajectory: crate::gpu::TrajectoryGpu,
    segments: &[crate::gpu::RationalBezierGpu],
  ) -> EngineResult<u64> {
    use crate::scene::{TransformComponent, trajectory::TrajectoryComponent};
    use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat};

    let scene_ctx_lock = {
      let scenes = self.scenes.read();
      scenes.get_scene(scene_id).ok_or(EngineError::InvalidOperation(
        "spawn_trajectory: scene not found",
      ))?
    };
    let scene_ctx = scene_ctx_lock.write();

    let entity_id = scene_ctx.scene.spawn_entity(entity_name);

    // Transform relative to parent
    scene_ctx.scene.add_component(
      entity_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )?;

    // Link parent
    if parent_entity > 0 {
      if let Some(parent_id) = scene_ctx.get_entity(parent_entity) {
        scene_ctx.scene.set_parent(entity_id, Some(parent_id));
      }
    }

    // Build control points array
    let mut control_points = alloc::vec::Vec::with_capacity(segments.len() * 4);
    for seg in segments {
      control_points.push(seg.cp0);
      control_points.push(seg.cp1);
      control_points.push(seg.cp2);
      control_points.push(seg.cp3);
    }

    // Default subdivisions to 64 per segment for smoothness
    let traj_comp = TrajectoryComponent::new(
      control_points,
      trajectory.color,
      trajectory.line_width,
      trajectory.texture_id,
      64,
    );

    scene_ctx.scene.add_component(entity_id, traj_comp)?;

    Ok(EntityId::as_ffi(&entity_id))
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
        .ok_or(EngineError::InvalidOperation(
          "spawn_static_mesh: model not found",
        ))?
        .clone();
      let mesh_arc = scenes.mesh_cache.get(&path_str).ok_or(EngineError::InvalidOperation(
        "spawn_static_mesh: mesh not in cache",
      ))?;
      (path_str, mesh_arc)
    };

    let scene_ctx_lock = {
      let scenes = self.scenes.read();
      scenes.get_scene(scene_id).ok_or(EngineError::InvalidOperation(
        "spawn_static_mesh: scene not found",
      ))?
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
    let dist_au = (pos.x() * pos.x() + pos.y() * pos.y() + pos.z() * pos.z()).sqrt();
    const SUN_RADIUS_AU: f32 = 0.0046524726_f32;
    const KM_PER_AU: f32 = 149_597_870.7_f32;
    let soi_radius_au = (dist_au - SUN_RADIUS_AU).max(SUN_RADIUS_AU);

    // ReferenceFrameComponent::scale = AU/km (unit conversion only).
    // soi_radius is the visual extent, not the scale.
    scene_ctx.scene.add_component(
      lca_id,
      ReferenceFrameComponent {
        frame_type: crate::scene::ReferenceFrameType::Micro,
        scale: 1.0 / KM_PER_AU,
        soi_radius: soi_radius_au,
        depth_layer: 1,
      },
    )?;

    let root = scene_ctx.root_entity;
    scene_ctx.scene.set_parent(lca_id, Some(root));
    let lca_ext_id = EntityId::as_ffi(&lca_id);

    // ── Mesh entity (child of LCA frame) ───────────────────────────────
    let mesh_id = scene_ctx.scene.spawn_entity(entity_name);

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
      StaticMeshComponent {
        asset_path: path_str,
        mesh: mesh_arc,
        emissive_color: [0.0, 0.0, 0.0, 0.0],
        is_visible: true,
      },
    )?;

    scene_ctx.scene.add_component(
      mesh_id,
      crate::scene::SphericalGizmoComponent { is_visible: true },
    )?;

    scene_ctx.scene.set_parent(mesh_id, Some(lca_id));
    let mesh_ext_id = EntityId::as_ffi(&mesh_id);

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
      v.position[0] * v.position[0] + v.position[1] * v.position[1] + v.position[2] * v.position[2]
    })
    .fold(0.0_f32, f32::max)
    .sqrt()
}
