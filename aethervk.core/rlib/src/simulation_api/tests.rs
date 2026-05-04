extern crate std;

use super::components_api::CameraParams;
use super::*;
use crate::gpu;
use crate::scene::Marker;
use alloc::format;
use core::ffi::CStr;
use core::ffi::c_char;
use std::println;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
      let scene_id = ctx.create_empty_scene().unwrap();

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

      let scene_id = ctx.create_empty_scene().unwrap();
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

      let scene_id = ctx.create_empty_scene().unwrap();
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
      let scene_id = ctx.create_empty_scene().unwrap();

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
      let scene_id = ctx.create_default_scene().unwrap();

      // Register text archetype
      ctx.scenes.write().get_mut(&scene_id).unwrap().write().entity_map.clear();

      let width = 256;
      let height = 256;

      let pe_handle = ctx.create_presentation_engine(scene_id, width, height).unwrap();

      SimulationContext::set_render_callback(Some(render_callback_impl));

      // Let the logic thread tick and call RENDER_CALLBACK
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::PlayScene { scene_id });

      let mut attempts = 0;
      let mut task_id = 0;
      while attempts < 100 {
        task_id = LAST_RENDER_TASK_ID.load(Ordering::Acquire);
        if task_id != 0 {
          break;
        }
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
        attempts += 1;
      }

      assert!(
        task_id > 0,
        "Render task was never completed / callback not called"
      );

      // Wait for task to be truly ready in renderer
      let mut status = ctx.get_task_status(task_id);
      attempts = 0;
      while matches!(status, structs::TaskStatusCode::Pending) && attempts < 50 {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(10));
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
      let scene_id = ctx.create_empty_scene().unwrap();
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
      assert!(ctx.add_sun_component(scene_id, entity_id, (128, 128, 128)).is_ok());
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
      let scene_id = ctx.create_empty_scene().unwrap();
      let parent_name = alloc::ffi::CString::new("Parent").unwrap();
      let parent_id = ctx.spawn_entity(scene_id, parent_name.to_str().unwrap()).unwrap();

      let child_name = alloc::ffi::CString::new("Child").unwrap();
      let child_id = ctx.spawn_entity(scene_id, child_name.to_str().unwrap()).unwrap();

      assert!(ctx.set_parent(scene_id, child_id, parent_id).is_ok());

      let fetched_parent = ctx.get_entity_parent(scene_id, child_id).unwrap();
      assert_eq!(fetched_parent, parent_id);

      let mut out_ids = vec![0u64; 10];
      let (count, missing) = ctx.get_entity_ids(scene_id, &mut out_ids).unwrap();
      assert!(count > 0);

      let mut out_name = vec![0i8; 100];
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
      let scene_id = ctx.create_empty_scene().unwrap();

      let cam_name = alloc::ffi::CString::new("Cam").unwrap();
      let cam_id = ctx.spawn_entity(scene_id, cam_name.to_str().unwrap()).unwrap();
      assert!(ctx.set_active_camera(scene_id, cam_id).is_ok());

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
      let scene_id = ctx.create_empty_scene().unwrap();

      let mut count = 0;
      let ptr = ctx.get_almanac_loaded_files(&mut count);
      if !ptr.is_null() && count > 0 {
        // Just testing coverage
        let _ = alloc::ffi::CString::from_raw(*ptr);
      }

      let sphere_id = ctx.spawn_procedural_sphere(scene_id, core::ptr::null(), 1.0).unwrap();
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
      });
      let _ = ctx.get_task_result_raycast(0, &mut out_hit);

      let mut out_state = crate::simulation::almanac::KinematicState::default();
      let _ = ctx.get_task_result_kinematic_state(0, &mut out_state);

      let pe_handle = ctx.create_presentation_engine(scene_id, 100, 100).unwrap();
      let _ = ctx.resize(scene_id, pe_handle, 200, 200);

      let mut buffer = [0u8; 10];
      let _ = ctx.download_image(0, buffer.as_mut_ptr(), 10);

      // Logic thread commands that missed coverage
      let _ =
        ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::PauseScene { scene_id });
      let _ = ctx.threads.logic_thread.tx().try_send(structs::LogicCommand::SetSceneTimeScale {
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
