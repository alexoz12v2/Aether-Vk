//! tests module.

extern crate std;

use super::{components_api::CameraParams, *};
use crate::{gpu, scene::Marker};
use alloc::format;
use core::ffi::{CStr, c_char};
use std::{
  println,
  sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
  },
};

fn panic_error_callback(msg: &str) {
  panic!("Vulkan Error: {}", msg);
}

fn get_test_context() -> Option<*mut SimulationContext> {
  // Set asset path before startup so it's available for shaders
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);

  let ctx_ptr = SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback));
  if let Ok(boxed) = ctx_ptr {
    return Some(alloc::boxed::Box::into_raw(boxed));
  }
  println!("Skipping test: Vulkan backend could not be initialized");
  None
}

#[test]
fn test_startup_and_drop() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_time_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      // Default time
      let initial_time = ctx.get_simulation_time(scene_id);

      // Set time
      ctx.set_simulation_time(scene_id, 12345.678);
      let new_time = ctx.get_simulation_time(scene_id);
      assert!((new_time - 12345.678).abs() < 1e-5);

      // Time scale
      ctx.set_time_scale(scene_id, 1); // OneDay
      let scale = ctx.get_scene(scene_id).unwrap().read().time_state.read().current_scale.clone();
      assert!(matches!(
        scale,
        crate::simulation_api::structs::TimeScale::OneDay
      ));

      ctx.set_time_scale(scene_id, 0); // Stopped
      let scale = ctx.get_scene(scene_id).unwrap().read().time_state.read().current_scale.clone();
      assert!(matches!(
        scale,
        crate::simulation_api::structs::TimeScale::Stopped
      ));

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_entity_transform_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx.create_empty_scene(true).unwrap();
      let name = alloc::ffi::CString::new("TestEntity").unwrap();
      let entity_id = ctx.spawn_entity(scene_id, name.to_str().unwrap()).unwrap();

      // Ensure entity exists
      assert!(entity_id > 0);

      // Add transform
      let res = ctx.add_transform_component(
        scene_id,
        entity_id,
        Vec3f32::from_components(1.0, 2.0, 3.0),
        Quat::from_components(1.0, 0.0, 0.0, 0.0),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      );
      assert!(res.is_ok());

      // Get transform
      let mut px = 0.0;
      let mut py = 0.0;
      let mut pz = 0.0;
      let mut rw = 0.0;
      let mut rx = 0.0;
      let mut ry = 0.0;
      let mut rz = 0.0;
      let mut sx = 0.0;
      let mut sy = 0.0;
      let mut sz = 0.0;

      let res = ctx.get_transform_component(
        scene_id, entity_id, &mut px, &mut py, &mut pz, &mut rw, &mut rx, &mut ry, &mut rz,
        &mut sx, &mut sy, &mut sz,
      );
      assert!(res.is_ok());
      assert!((px - 1.0).abs() < 1e-5);
      assert!((py - 2.0).abs() < 1e-5);
      assert!((pz - 3.0).abs() < 1e-5);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_camera_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx.create_empty_scene(true).unwrap();
      let name = alloc::ffi::CString::new("CameraEntity").unwrap();
      let entity_id = ctx.spawn_entity(scene_id, name.to_str().unwrap()).unwrap();

      let _ = ctx.add_transform_component(
        scene_id,
        entity_id,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::from_components(1.0, 0.0, 0.0, 0.0),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      );

      let res = ctx.add_camera_component(
        scene_id,
        entity_id,
        CameraParams::new_perspective(60.0f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0),
      );
      assert!(res.is_ok());

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_scene_entities_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      let initial_count = ctx.get_entity_count(scene_id).unwrap_or(0);

      let name = alloc::ffi::CString::new("TestEntity2").unwrap();
      let entity_id = ctx.spawn_entity(scene_id, name.to_str().unwrap()).unwrap();

      assert_eq!(ctx.get_entity_count(scene_id).unwrap(), initial_count + 1);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_misc_and_models_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let path = String::from("/dummy/path");
      SimulationContext::set_asset_path(&path);

      let task_id = ctx.task_manager.write().create_task().get();
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::ImportModel {
        task_id,
        path: path.clone(),
      });
      let mut attempts = 0;
      while matches!(
        ctx.get_task_status(task_id),
        structs::TaskStatusCode::Pending
      ) && attempts < 20
      {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
        attempts += 1;
      }

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

static LAST_RENDER_TASK_ID: AtomicU64 = AtomicU64::new(0);

extern "C" fn render_callback_impl(scene_id: u64, pe_id: u64, render_generation: u64) {
  LAST_RENDER_TASK_ID.store(render_generation, Ordering::Release);
}

#[test]
fn test_simulation_context_text_rendering() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_default_scene(true).unwrap();

      let width = 256;
      let height = 256;

      let pe_handle = ctx.create_presentation_engine(scene_id, width, height).unwrap();
      ctx
        .add_perspective_camera(scene_id, pe_handle, "camera", 45.0, 0.1, 100.0)
        .unwrap();

      SimulationContext::set_render_callback(Some(render_callback_impl));

      // Let the logic thread tick and call RENDER_CALLBACK
      let _ = ctx
        .threads
        .logic_thread
        .tx()
        .try_send(structs::LogicCommand::PlayScene { scene_id });

      let mut attempts = 0;
      let mut task_id = 0;
      while attempts < 500 {
        task_id = LAST_RENDER_TASK_ID.load(Ordering::Acquire);
        if task_id != 0 {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(20));
        attempts += 1;
      }

      assert!(
        task_id > 0,
        "Render task was never completed / callback not called"
      );

      // Wait for task to be truly ready in renderer
      let mut status = ctx.get_task_status(task_id);
      attempts = 0;
      while matches!(status, structs::TaskStatusCode::Pending) && attempts < 500 {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(20));
        status = ctx.get_task_status(task_id);
        attempts += 1;
      }

      assert!(
        matches!(status, structs::TaskStatusCode::Completed),
        "Render task did not complete successfully"
      );

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      let success = ctx.download_image(task_id, buffer.as_mut_ptr(), buffer.len());
      assert!(success, "Download image failed");

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}
use super::*;
use aethervk_oshal_rlib::math::vector::Vector;
use alloc::vec::Vec;

#[test]
fn test_components_api() {
  if let Some(ctx_ptr) = super::tests::get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();
      let name = alloc::ffi::CString::new("TestEntity").unwrap();
      let entity_id = ctx.spawn_entity(scene_id, name.to_str().unwrap()).unwrap();

      let _ = ctx.add_transform_component(
        scene_id,
        entity_id,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      );

      assert!(ctx.add_sky_component(scene_id, entity_id).is_ok());
      assert!(ctx.add_cursor_component(scene_id, entity_id).is_ok());
      assert!(ctx.add_sun_component(scene_id, entity_id, (128, 128, 128), 0.6).is_ok());
      assert!(ctx.add_grid_component(scene_id, entity_id).is_ok());

      assert!(
        ctx
          .add_measurement_component(
            scene_id,
            entity_id,
            Vec3f32::from_components(0.0, 0.0, 0.0),
            Vec3f32::from_components(1.0, 1.0, 1.0)
          )
          .is_ok()
      );

      assert!(ctx.add_image_billboard_component(scene_id, entity_id, true, 1.0, 1.0).is_ok());

      let entity2 = ctx.spawn_entity(scene_id, "TestEntity2").unwrap();
      let _ = ctx.add_transform_component(
        scene_id,
        entity2,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      );
      assert!(ctx.add_image_billboard_component(scene_id, entity2, false, 1.0, 1.0).is_ok());

      // Markers
      let marker = crate::scene::Marker {
        local_pos: [0.0, 0.0, 0.0],
        color: [1.0, 0.0, 0.0],
        size: 1.0,
      };
      assert!(ctx.set_markers(scene_id, entity_id, &[marker]).is_ok());

      // Physical mesh
      // We need a path
      let path = alloc::format!("{}/../../assets/Comet.glb", env!("CARGO_MANIFEST_DIR"));
      let path_buf = oshal::os::fs::PathBuf::from(&path);
      let mesh_res =
        ctx.add_physical_mesh_component(scene_id, entity_id, &path_buf, 1.0, [1.0, 1.0, 1.0]);
      // Ignore error if path not found in CI, just test it doesn't crash badly

      // BVH node visibility
      if mesh_res.is_ok() {
        let _ = ctx.set_bvh_node_visibility(scene_id, entity_id, 0, false);
      }

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_scene_api() {
  if let Some(ctx_ptr) = super::tests::get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();
      let parent_name = alloc::ffi::CString::new("Parent").unwrap();
      let parent_id = ctx.spawn_entity(scene_id, parent_name.to_str().unwrap()).unwrap();
      ctx
        .add_transform_component(
          scene_id,
          parent_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();

      let child_name = alloc::ffi::CString::new("Child").unwrap();
      let child_id = ctx.spawn_entity(scene_id, child_name.to_str().unwrap()).unwrap();
      ctx
        .add_transform_component(
          scene_id,
          child_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();

      assert!(ctx.set_parent(scene_id, child_id, parent_id).is_ok());

      let fetched_parent = ctx.get_entity_parent(scene_id, child_id).unwrap();
      assert_eq!(fetched_parent, parent_id);

      let mut out_ids = vec![0u64; 10];
      let (count, missing) = ctx.get_entity_ids(scene_id, &mut out_ids).unwrap();
      assert!(count > 0);

      let mut out_name = vec![0i8 as core::ffi::c_char; 100];
      let missing_name = ctx.get_entity_name(scene_id, child_id, &mut out_name).unwrap();
      let name_str = core::ffi::CStr::from_ptr(out_name.as_ptr()).to_str().unwrap();
      assert_eq!(name_str, "Child");

      assert!(ctx.set_entity_visibility(scene_id, child_id, false).is_ok());
      assert!(ctx.set_entity_selected(scene_id, child_id, true).is_ok());
      assert!(ctx.set_entity_following(scene_id, child_id, true).is_ok());

      assert!(ctx.remove_entity(scene_id, child_id).is_ok());

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_core_api() {
  if let Some(ctx_ptr) = super::tests::get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      let cam_name = alloc::ffi::CString::new("Cam").unwrap();
      let cam_id = ctx.spawn_entity(scene_id, cam_name.to_str().unwrap()).unwrap();
      ctx
        .add_transform_component(
          scene_id,
          cam_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();
      ctx
        .add_camera_component(
          scene_id,
          cam_id,
          CameraParams::new_perspective(45.0_f32.to_radians(), 1.0, 0.1, 100.0),
        )
        .unwrap();

      let pe_handle = ctx.create_presentation_engine(scene_id, 800, 600).unwrap();
      assert!(ctx.destroy_presentation_engine(scene_id, pe_handle).is_ok());

      fn dummy_logic(
        _: &crate::simulation_api::structs::LogicThreadContext,
        _: *mut core::ffi::c_void,
      ) -> EngineResult<crate::simulation_api::structs::SimulationTaskResult> {
        Ok(crate::simulation_api::structs::SimulationTaskResult::Bool(
          true,
        ))
      }

      let task_id = ctx.dispatch_logic_command_custom(dummy_logic, None).unwrap();
      let mut attempts = 0;
      while matches!(
        ctx.get_task_status(task_id.get()),
        structs::TaskStatusCode::Pending
      ) && attempts < 20
      {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
        attempts += 1;
      }
      assert!(ctx.get_task_result_bool(task_id.get()));

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_misc_and_models_api_direct() {
  if let Some(ctx_ptr) = super::tests::get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      let mut count = 0;
      let ptr = ctx.get_almanac_loaded_files(&mut count);
      if !ptr.is_null() && count > 0 {
        // Just testing coverage
        let _ = alloc::ffi::CString::from_raw(*ptr);
      }

      let sphere_id = ctx.spawn_procedural_sphere(scene_id, core::ptr::null(), 1.0, 1.0).unwrap();
      assert!(sphere_id > 0);

      // test unload model (model 0 doesn't exist, but it covers the branch)
      ctx.unload_model(0);

      // test misc API
      SimulationContext::set_logger_callback(None);
      SimulationContext::set_breadcrumb_callback(None);
      SimulationContext::set_simulation_callback(None);
      SimulationContext::set_render_callback(None);

      let status = ctx.get_task_status(0);
      assert_eq!(status, structs::TaskStatusCode::Invalid);

      let res_u64 = ctx.get_task_result_u64(0);
      assert_eq!(res_u64, 0);

      let res_bool = ctx.get_task_result_bool(0);
      assert_eq!(res_bool, false);

      let mut out_hit = Some(structs::RayCastHit {
        entity_ext_id: 0,
        p: Vec3f32::zero(),
        uv: [0.0, 0.0],
      });
      let _ = ctx.get_task_result_raycast(0, &mut out_hit);

      let mut out_state = crate::simulation::almanac::KinematicState::default();
      let _ = ctx.get_task_result_kinematic_state(0, &mut out_state);

      let pe_handle = ctx.create_presentation_engine(scene_id, 100, 100).unwrap();
      ctx
        .add_perspective_camera(scene_id, pe_handle, "camera", 45.0, 0.1, 100.0)
        .unwrap();
      let _ = ctx.resize(scene_id, pe_handle, 200, 200);

      let mut buffer = [0u8; 10];
      let _ = ctx.download_image(0, buffer.as_mut_ptr(), 10);

      // Logic thread commands that missed coverage
      let _ = ctx
        .threads
        .logic_thread
        .tx()
        .try_send(structs::LogicCommand::PauseScene { scene_id });
      let _ = ctx
        .threads
        .logic_thread
        .tx()
        .try_send(structs::LogicCommand::SetSceneTimeScale {
          scene_id,
          scale: structs::TimeScale::OneMonth,
        });
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::SetSceneEpoch {
        scene_id,
        epoch_tai_seconds: 0.0,
      });
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::StepScene {
        scene_id,
        step_days: 1.0,
      });
      // TODO this takes internal entity id from slot map, therefore you need to first insert it into
      // TODO the entity mapping. Furthermore, you cannot unfollow an entity without following one
      // let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::UnfollowEntity(
      //   structs::UnfollowEntity {
      //     entity_id: 0,
      //     scene: ctx.get_scene(scene_id).unwrap(),
      //   },
      // ));

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

fn set_asset_directory_for_tests() {
  crate::gpu::set_asset_dir_for_tests();
}

#[test]
fn test_snapshot_and_restore() {
  set_asset_directory_for_tests();
  println!("test_snapshot_and_restore: 1. start");
  let ctx =
    SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback)).unwrap();
  println!("test_snapshot_and_restore: 2. created SimulationContext");
  let scene_id = ctx.create_empty_scene(true).unwrap();
  println!("test_snapshot_and_restore: 3. created empty scene");

  // Spawn an entity and set position
  let entity = ctx.spawn_entity(scene_id, "TestSnapshot").unwrap();
  let entity_internal =
    ctx.get_scene(scene_id).as_ref().unwrap().read().get_entity(entity).unwrap();
  // TODO add a way to get the external id of root
  let root_entity = ctx.get_scene(scene_id).as_ref().unwrap().read().root_entity;
  let initial_pos = <Vec3f32 as Vector3>::from_components(1.0, 2.0, 3.0);
  ctx
    .add_component_to_entity(
      scene_id,
      entity,
      crate::scene::TransformComponent {
        position: initial_pos,
        rotation: <Quat as Quaternion>::identity(),
        scale: <Vec3f32 as Vector3>::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  ctx
    .get_scene(scene_id)
    .as_ref()
    .unwrap()
    .read()
    .scene
    .set_parent(entity_internal, Some(root_entity));

  // Take snapshot
  let _ = ctx
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::SnapshotScene { scene_id });

  // Wait for logic thread to process snapshot
  oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));

  // Move the entity
  ctx
    .set_transform_component(
      scene_id,
      entity,
      Vec3f32::from_components(5.0, 5.0, 5.0),
      Quat::identity(),
      Vec3f32::one(),
    )
    .unwrap();

  // Verify it moved
  let new_pos = ctx.get_transform_component2(scene_id, entity).unwrap().position;
  assert_eq!(new_pos.x(), 5.0);

  // Restore snapshot
  let _ = ctx
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::RestoreSnapshot { scene_id });

  // Wait for logic thread to process restore
  oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));

  // Verify it was restored
  let restored_pos = ctx.get_transform_component2(scene_id, entity).unwrap().position;
  assert_eq!(restored_pos.x(), initial_pos.x());
  assert_eq!(restored_pos.y(), initial_pos.y());
  assert_eq!(restored_pos.z(), initial_pos.z());
}

#[test]
fn test_spawn_comet_internal_bounds_and_hierarchy() {
  set_asset_directory_for_tests();
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      // Register a dummy model
      let model_id = 999;
      let path = "test_comet.glb";
      {
        let mut scenes = ctx.scenes.write();
        scenes.model_registry.insert(model_id, path.into());
        let dummy_mesh = crate::simulation::comet::Comet {
          id: 0,
          vertices: alloc::vec![crate::simulation::comet::Vertex {
            position: [0.0, 0.0, 2.5],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
          }],
          indices: alloc::vec![],
          albedo_map: None,
          normal_map: None,
          roughness_map: None,
          ao_map: None,
          mass_properties: polyhedral_mass_properties::MassProperties::from_contrib_sum(
            polyhedral_mass_properties::TriangleContrib::new(
              [-1.0, 0.0, -1.0],
              [1.0, 0.0, -1.0],
              [0.5, 1.0, 0.0],
            ),
          )
          .unwrap(),
          bvh: None,
          pa_basis_bf: None,
          bf_to_pa: None,
        };
        scenes.mesh_cache.insert(path.into(), dummy_mesh);
      }

      let pos = Vec3f32::from_components(2.0, 0.0, 0.0);
      let rot = Quat::identity();
      let radius_km = 2.5;
      let mass_kg = 1e13;
      let physics_type = 2; // Dynamic

      let (lca_ext_id, comet_ext_id) = ctx
        .spawn_comet_internal(
          scene_id,
          model_id,
          "Halley",
          pos,
          rot,
          radius_km,
          mass_kg,
          physics_type,
          0, // dynamic doesn't use comet id
          None,
          Vec3f32::zero(),
        )
        .expect("spawn_comet_internal should succeed");

      let scenes = ctx.scenes.read();
      let scene_arc = scenes.get_scene(scene_id).unwrap();
      let scene_ctx = scene_arc.read();
      let lca_id = scene_ctx.entity_map.get(&lca_ext_id).unwrap().clone();
      let comet_id = scene_ctx.entity_map.get(&comet_ext_id).unwrap().clone();

      // 1. Assert microframe is there on the specified position (2.0 AU)
      let lca_transform = scene_ctx
        .scene
        .with_component(lca_id, |c: &crate::scene::TransformComponent| *c)
        .unwrap();
      assert_eq!(lca_transform.position.x(), 2.0);
      assert_eq!(lca_transform.position.y(), 0.0);
      assert_eq!(lca_transform.position.z(), 0.0);

      // 2. Assert microframe scale is AU/km unit conversion (not soi_radius).
      let lca_ref = scene_ctx
        .scene
        .with_component(lca_id, |c: &crate::scene::ReferenceFrameComponent| {
          c.clone()
        })
        .unwrap();
      let expected_soi = 2.0_f32 - 0.0046524726_f32; // dist_au - SUN_RADIUS_AU
      let expected_scale = 1.0_f32 / 149_597_870.7_f32; // AU/km
      assert!(
        (lca_ref.scale - expected_scale).abs() < 1e-15,
        "scale should be AU/km, got {}",
        lca_ref.scale
      );
      assert!((lca_ref.soi_radius - expected_soi).abs() < 1e-4);
      let min_x = lca_transform.position.x() - lca_ref.soi_radius;
      assert!(min_x > 0.0, "Microframe bounds overlap with sun at origin!");

      // 3. Assert comet is child of microframe
      assert_eq!(scene_ctx.scene.get_parent(comet_id).unwrap(), lca_id);

      // 4. Assert comet is in the center of the microframe
      let comet_transform = scene_ctx
        .scene
        .with_component(comet_id, |c: &crate::scene::TransformComponent| *c)
        .unwrap();
      assert_eq!(comet_transform.position.x(), 0.0);
      assert_eq!(comet_transform.position.y(), 0.0);
      assert_eq!(comet_transform.position.z(), 0.0);

      // 5. Assert comet occupies specified radius in km.
      // mesh_scale = radius_km / bounding_sphere (since micro-frame units are km)
      // bounding_sphere = 2.5 (from the test mesh: 5 vertices spanning [-5..5])
      let bounding_sphere = 2.5_f32; // half-extent length of the test mesh
      let expected_mesh_scale = 2.5_f32 / bounding_sphere;
      assert!(
        (comet_transform.scale.x() - expected_mesh_scale).abs() < expected_mesh_scale * 0.01,
        "mesh scale: got {} expected {}",
        comet_transform.scale.x(),
        expected_mesh_scale
      );
      assert!((comet_transform.scale.y() - expected_mesh_scale).abs() < expected_mesh_scale * 0.01);
      assert!((comet_transform.scale.z() - expected_mesh_scale).abs() < expected_mesh_scale * 0.01);

      let comet_mesh_radius = scene_ctx
        .scene
        .with_component(comet_id, |c: &crate::scene::PhysicalMeshComponent| {
          c.sphere_radius
        })
        .unwrap();
      assert_eq!(comet_mesh_radius, 2.5);

      drop(scene_ctx);
      drop(scene_arc);
      drop(scenes);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_spawn_comet_multi_scale_layer_separation() {
  use crate::gpu::scene_conversion::SceneConversionExt;

  set_asset_directory_for_tests();
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      // Register a dummy model
      let model_id = 1000;
      let path = "test_comet_layers.glb";
      {
        let mut scenes = ctx.scenes.write();
        scenes.model_registry.insert(model_id, path.into());
        let dummy_mesh = crate::simulation::comet::Comet {
          id: 0,
          vertices: alloc::vec![crate::simulation::comet::Vertex {
            position: [0.0, 0.0, 2.5],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
          }],
          indices: alloc::vec![],
          albedo_map: None,
          normal_map: None,
          roughness_map: None,
          ao_map: None,
          mass_properties: polyhedral_mass_properties::MassProperties::from_contrib_sum(
            polyhedral_mass_properties::TriangleContrib::new(
              [-1.0, 0.0, -1.0],
              [1.0, 0.0, -1.0],
              [0.5, 1.0, 0.0],
            ),
          )
          .unwrap(),
          bvh: None,
          pa_basis_bf: None,
          bf_to_pa: None,
        };
        scenes.mesh_cache.insert(path.into(), dummy_mesh);
      }

      // Spawn comet at 2 AU from origin
      let pos = Vec3f32::from_components(2.0, 0.0, 0.0);
      let rot = Quat::identity();
      let radius_km = 2.5;
      let mass_kg = 1e13;
      let physics_type = 2;

      let (lca_ext_id, _comet_ext_id) = ctx
        .spawn_comet_internal(
          scene_id,
          model_id,
          "TestComet",
          pos,
          rot,
          radius_km,
          mass_kg,
          physics_type,
          0, // dynamic doesn't use comet id
          None,
          Vec3f32::zero(),
        )
        .expect("spawn_comet_internal should succeed");

      // Add a camera looking at the comet position
      let camera_name = alloc::ffi::CString::new("Camera").unwrap();
      let camera_ext = ctx.spawn_entity(scene_id, camera_name.to_str().unwrap()).unwrap();
      ctx
        .add_transform_component(
          scene_id,
          camera_ext,
          Vec3f32::from_components(1.9, 0.0, 0.01),
          Quat::identity(),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();
      let cam_params = CameraParams::Perspective(components_api::PerspectiveCameraParams {
        fov: 45.0f32.to_radians(),
        aspect_ratio: 1.0,
        near_plane: 1e-5,
        far_plane: 1000.0,
      });
      ctx.add_camera_component(scene_id, camera_ext, cam_params).unwrap();

      // Convert scene and check layer separation
      let scenes = ctx.scenes.read();
      let scene_arc = scenes.get_scene(scene_id).unwrap();
      let scene_ctx = scene_arc.read();

      let camera_id = scene_ctx.entity_map.get(&camera_ext).unwrap().clone();
      let result = scene_ctx
        .scene
        .convert_scene(camera_id, false, None, [800, 600], None)
        .expect("convert_scene should succeed");

      // Should have 2 depth layers
      assert!(
        result.depth_layers.len() >= 2,
        "Expected at least 2 depth layers after spawn_comet, got {}",
        result.depth_layers.len()
      );

      // Validate near < far for every layer
      for layer in &result.depth_layers {
        assert!(
          layer.near < layer.far,
          "Layer {} violates near < far: near={}, far={}",
          layer.layer_index,
          layer.near,
          layer.far,
        );
        assert!(
          layer.near > 0.0,
          "Layer {} has non-positive near plane: {}",
          layer.layer_index,
          layer.near,
        );
      }

      // Micro layer should have tight SOI bounds (in frame-local km units)
      // Comet at 2 AU, camera at ~1.9 AU => dist ≈ 0.1 AU ≈ 1.5e7 km
      // SOI can be large; just verify it's finite and bounded.
      if let Some(micro_layer) = result.depth_layers.iter().find(|l| l.layer_index == 1) {
        assert!(
          micro_layer.far < 1e10 && micro_layer.far > 0.0,
          "Micro layer far={} should be bounded in frame-local km space",
          micro_layer.far,
        );
      }

      drop(scene_ctx);
      drop(scene_arc);
      drop(scenes);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_camera_controls() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_empty_scene(true).unwrap();

      let camera_name = alloc::ffi::CString::new("TestCamera").unwrap();
      let camera_ext = ctx.spawn_entity(scene_id, camera_name.to_str().unwrap()).unwrap();

      let initial_pos = Vec3f32::from_components(0.0, 0.0, 10.0);
      let initial_rot = Quat::identity();
      ctx
        .add_transform_component(
          scene_id,
          camera_ext,
          initial_pos,
          initial_rot,
          Vec3f32::one(),
        )
        .unwrap();

      let cam_params = CameraParams::Perspective(components_api::PerspectiveCameraParams {
        fov: 45.0f32.to_radians(),
        aspect_ratio: 1.0,
        near_plane: 0.1,
        far_plane: 1000.0,
      });
      ctx.add_camera_component(scene_id, camera_ext, cam_params).unwrap();

      let camera_id = {
        let scenes = ctx.scenes.read();
        let scene_arc = scenes.get_scene(scene_id).unwrap();
        let scene_ctx = scene_arc.read();
        scene_ctx.entity_map.get(&camera_ext).unwrap().clone()
      };

      let get_cam_transform = || {
        let scenes = ctx.scenes.read();
        let scene_arc = scenes.get_scene(scene_id).unwrap();
        let scene_ctx = scene_arc.read();
        scene_ctx
          .scene
          .with_component(camera_id, |c: &crate::scene::HighResTransformComponent| *c)
          .unwrap()
      };

      let initial_hrt = get_cam_transform();

      // Test Rotate
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::RotateCamera(
        structs::RotateCamera {
          camera_entity: camera_id,
          delta_x: 0.1,
          delta_y: 0.1,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.rotation != initial_hrt.rotation {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }

      let hrt1 = get_cam_transform();
      assert!(hrt1.rotation != initial_hrt.rotation);

      // Test Pan
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCamera(
        structs::PanCamera {
          camera_entity: camera_id,
          delta_x: 1.0,
          delta_y: 1.0,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.position != hrt1.position {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }

      let hrt2 = get_cam_transform();
      assert!(hrt2.position != hrt1.position);
      assert_eq!(hrt2.rotation, hrt1.rotation);

      // Test Zoom
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::ZoomCamera(
        structs::ZoomCamera {
          camera_entity: camera_id,
          amount: 1.0,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.position != hrt2.position {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }

      let hrt3 = get_cam_transform();
      assert!(hrt3.position != hrt2.position);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_camera_controls_microframe() {
  set_asset_directory_for_tests();
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      // Use create_default_scene to also spawn a camera
      let scene_id = ctx.create_default_scene(true).unwrap();

      // Register a dummy model
      let model_id = 4567;
      let path = "test_comet_cam2.glb";
      {
        let mut scenes = ctx.scenes.write();
        scenes.model_registry.insert(model_id, path.into());
        let dummy_mesh = crate::simulation::comet::Comet {
          id: 0,
          vertices: alloc::vec![crate::simulation::comet::Vertex {
            position: [0.0, 0.0, 2.5],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
          }],
          indices: alloc::vec![],
          albedo_map: None,
          normal_map: None,
          roughness_map: None,
          ao_map: None,
          mass_properties: polyhedral_mass_properties::MassProperties::from_contrib_sum(
            polyhedral_mass_properties::TriangleContrib::new(
              [-1.0, 0.0, -1.0],
              [1.0, 0.0, -1.0],
              [0.5, 1.0, 0.0],
            ),
          )
          .unwrap(),
          bvh: None,
          pa_basis_bf: None,
          bf_to_pa: None,
        };
        scenes.mesh_cache.insert(path.into(), dummy_mesh);
      }

      let pos = Vec3f32::from_components(0.01, 0.0, 0.0);
      let rot = Quat::identity();
      // static doesn't use comet id
      let (_lca_ext, _comet_ext) = ctx
        .spawn_comet_internal(
          scene_id,
          model_id,
          "comet_micro",
          pos,
          rot,
          1.0,
          1000.0,
          0,
          0,
          None,
          Vec3f32::zero(),
        )
        .unwrap();

      // Get the camera that create_default_scene already created
      let camera_int = {
        let scene_arc = ctx.scenes.read().get_scene(scene_id).unwrap();
        let scene_read = scene_arc.read();
        let mut found = None;
        for (_ext_id, &int_id) in scene_read.entity_map.iter() {
          if scene_read.scene.has_component::<crate::scene::CameraComponent>(int_id).into() {
            found = Some(int_id);
            break;
          }
        }
        found.expect("No camera found in scene")
      };

      // Set camera local to the microframe
      let cam_pos = Vec3f32::from_components(0.01001, 0.0, 0.0);
      let rot_cam = <Quat as Quaternion>::from_vector_and_scalar(
        Vec3f32::from_components(0.0, 0.0, -std::f32::consts::FRAC_1_SQRT_2),
        std::f32::consts::FRAC_1_SQRT_2,
      );

      {
        let scene_arc = ctx.scenes.read().get_scene(scene_id).unwrap();
        let mut scene_read = scene_arc.write();
        let _ = scene_read.scene.with_component_mut(
          camera_int,
          |t: &mut crate::scene::HighResTransformComponent| {
            t.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
              cam_pos.x() as f64,
              cam_pos.y() as f64,
              cam_pos.z() as f64,
            );
            t.rotation = rot_cam;
          },
        );
      }

      let get_cam_transform = || {
        let scene_arc = ctx.scenes.read().get_scene(scene_id).unwrap();
        scene_arc
          .read()
          .scene
          .with_component(camera_int, |c: &crate::scene::HighResTransformComponent| *c)
          .unwrap()
      };

      let initial_hrt = get_cam_transform();

      // Test Rotate
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::RotateCamera(
        structs::RotateCamera {
          camera_entity: camera_int,
          delta_x: 0.1,
          delta_y: 0.1,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.rotation != initial_hrt.rotation {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }
      let hrt1 = get_cam_transform();
      assert!(
        hrt1.rotation != initial_hrt.rotation,
        "Rotation should change in microframe"
      );

      // Test Pan
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::PanCamera(
        structs::PanCamera {
          camera_entity: camera_int,
          delta_x: 1.0,
          delta_y: 1.0,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.position != hrt1.position {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }
      let hrt2 = get_cam_transform();
      assert!(
        hrt2.position != hrt1.position,
        "Position should change after Pan in microframe"
      );
      assert_eq!(
        hrt2.rotation, hrt1.rotation,
        "Rotation should not change after Pan in microframe"
      );

      // Test Zoom
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::ZoomCamera(
        structs::ZoomCamera {
          camera_entity: camera_int,
          amount: 1.0,
          scene: ctx.get_scene(scene_id).unwrap(),
        },
      ));

      for _ in 0..100 {
        let curr = get_cam_transform();
        if curr.position != hrt2.position {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
      }
      let hrt3 = get_cam_transform();
      assert!(
        hrt3.position != hrt2.position,
        "Position should change after Zoom in microframe"
      );

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}
#[test]
fn test_callbacks_safety() {
  static BREADCRUMB_HIT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

  unsafe extern "C" fn mock_breadcrumb(_status: u32, _msg: *const core::ffi::c_char) {
    BREADCRUMB_HIT.store(true, core::sync::atomic::Ordering::SeqCst);
  }

  // 1. Set the safe callback
  SimulationContext::set_breadcrumb_callback(Some(mock_breadcrumb));

  // 2. Fire the breadcrumb
  crate::simulation_api::emit_breadcrumb(0, "Test Safe Breadcrumb");

  // 3. Verify it was hit
  assert!(BREADCRUMB_HIT.load(core::sync::atomic::Ordering::SeqCst));

  // 4. Remove the callback safely
  SimulationContext::set_breadcrumb_callback(None);
}
