extern crate std;

use super::*;
use std::println;
use core::ffi::c_char;
use core::ffi::CStr;
use std::sync::Arc;
use alloc::format;
use aethervk_core_rlib::gpu;
use super::components_api::CameraParams;

fn get_test_context() -> Option<*mut SimulationContext> {
  // Set asset path before startup so it's available for shaders
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  let path = alloc::ffi::CString::new(asset_dir).unwrap();
  SimulationContext::set_asset_path(path.as_ptr());

  let ctx_ptr = SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND);
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

      // Default time
      let initial_time = ctx.get_simulation_time();

      // Set time
      ctx.set_simulation_time(12345.678);
      let new_time = ctx.get_simulation_time();
      assert!((new_time - 12345.678).abs() < 1e-5);

      // Time scale
      ctx.set_time_scale(1); // OneDay
      assert_eq!(ctx.logic_state.read().current_scale, crate::structs::TimeScale::OneDay);

      ctx.set_time_scale(0); // Stopped
      assert_eq!(ctx.logic_state.read().current_scale, crate::structs::TimeScale::Stopped);

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
        scene_id, entity_id, 
        Vec3f32::from_components(1.0, 2.0, 3.0), 
        Quat::from_components(1.0, 0.0, 0.0, 0.0), 
        Vec3f32::from_components(1.0, 1.0, 1.0)
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

      // Set transform
      let res = ctx.set_transform_component(
        scene_id, entity_id, 
        Vec3f32::from_components(4.0, 5.0, 6.0), 
        Quat::from_components(1.0, 0.0, 0.0, 0.0), 
        Vec3f32::from_components(2.0, 2.0, 2.0)
      );
      assert!(res.is_ok());

      let res = ctx.get_transform_component(
        scene_id, entity_id, &mut px, &mut py, &mut pz, &mut rw, &mut rx, &mut ry, &mut rz,
        &mut sx, &mut sy, &mut sz,
      );
      assert!(res.is_ok());
      assert!((px - 4.0).abs() < 1e-5);
      assert!((py - 5.0).abs() < 1e-5);

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

      // Transform is required before adding camera
      let _ = ctx.add_transform_component(
        scene_id, entity_id, 
        Vec3f32::from_components(0.0, 0.0, 0.0), 
        Quat::from_components(1.0, 0.0, 0.0, 0.0), 
        Vec3f32::from_components(1.0, 1.0, 1.0)
      );

      let res = ctx.add_camera_component(scene_id, entity_id, CameraParams::new_perspective(60.0f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0));
      assert!(res.is_ok());

      let mut proj = [0.0f32; 16];
      let res = ctx.get_camera_component(scene_id, entity_id, &mut proj);
      assert!(res.is_ok());

      // Set camera
      let res = ctx.set_camera_component(scene_id, entity_id, CameraParams::new_perspective(90.0f32.to_radians(), 1.0, 0.1, 500.0));
      assert!(res.is_ok());

      let res = ctx.set_active_camera(scene_id, entity_id);
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

      let mut out_name = [0i8; 64];
      let _missing = ctx.get_entity_name(scene_id, entity_id, &mut out_name).unwrap();
      let name_str = CStr::from_ptr(out_name.as_ptr() as *const c_char)
        .to_str()
        .unwrap();
      assert_eq!(name_str, "TestEntity2");

      // Visibility and selections
      assert!(
        ctx
          .set_entity_visibility(scene_id, entity_id, false)
          .is_ok()
      );
      assert!(ctx.set_entity_selected(scene_id, entity_id, true).is_ok());
      assert!(ctx.set_entity_following(scene_id, entity_id, true).is_ok());

      let res = ctx.remove_entity(scene_id, entity_id);
      assert!(res.is_ok());
      assert_eq!(ctx.get_entity_count(scene_id).unwrap(), initial_count);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_misc_and_models_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let path = alloc::ffi::CString::new("/dummy/path").unwrap();
      SimulationContext::set_asset_path(path.as_ptr());

      // Models API async dummy test
      let model_id = avkSimulationContext_importModel(ctx_ptr, path.as_ptr());
      assert!(model_id > 0); 
      let mut attempts = 0;
      while ctx.get_task_status(model_id) == 0 && attempts < 20 {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
        attempts += 1;
      }
      assert!(ctx.get_task_status(model_id) != 0);

      // Test almanac file
      let almanac_id = avkSimulationContext_loadAlmanacFile(ctx_ptr, path.as_ptr());
      assert!(almanac_id > 0);
      let mut attempts = 0;
      while ctx.get_task_status(almanac_id) == 0 && attempts < 20 {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
        attempts += 1;
      }
      assert!(ctx.get_task_status(almanac_id) != 0);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_render_tick_and_download() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_default_scene().unwrap();
      let name = alloc::ffi::CString::new("TestSphere").unwrap();
      let sphere_id = ctx.spawn_procedural_sphere(scene_id, name.as_ptr() as *const _, 1.0).unwrap();
      assert!(sphere_id > 0);

      let width = 256;
      let height = 256;
      
      let pe_handle = ctx.create_presentation_engine(width, height).unwrap();

      let task_id_nonzero = ctx.render_tick(pe_handle, scene_id, [width, height]).unwrap();
      let task_id = task_id_nonzero.get();

      // Allow thread to process
      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));

      let mut status = ctx.get_task_status(task_id);
      let mut attempts = 0;
      while status == 0 && attempts < 20 {
        // Pending
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
        status = ctx.get_task_status(task_id);
        attempts += 1;
      }
      // 1 is success, 2 is failure, 0 is pending
      assert!(status == 1 || status == 2, "Render task stuck in pending");

      let mut buffer = vec![0u8; (width * height * 4) as usize];
      let success = ctx.download_image(task_id, buffer.as_mut_ptr(), buffer.len());
      if success {
          let mut attempts = 0;
          while ctx.get_task_status(task_id) == 0 && attempts < 20 {
            oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
            attempts += 1;
          }
      }

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_async_camera_commands() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_default_scene().unwrap();
      let cam_id = ctx.scenes.read().get_scene(scene_id).unwrap().read().active_camera_entity.unwrap();
      let ext_cam_id = ctx.scenes.read().get_scene(scene_id).unwrap().read().entity_map.iter().find(|&(_, &v)| v == cam_id).map(|(&k, _)| k).unwrap();

      let task1 = avkSimulationContext_panCamera(ctx_ptr, scene_id, ext_cam_id, 10.0, -5.0);
      assert!(task1 > 0);

      let task2 = avkSimulationContext_zoomCamera(ctx_ptr, scene_id, ext_cam_id, 5.0);
      assert!(task2 > 0);

      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_multiple_scenes() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let new_scene_id = ctx.create_default_scene().unwrap();
      assert!(new_scene_id > 0);

      let count = avkSimulationContext_getEntityCount(ctx_ptr, new_scene_id);
      assert!(count > 0);

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_all_archetypes() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.create_default_scene().unwrap();

      let name = alloc::ffi::CString::new("TestSphere").unwrap();
      let sphere_id = ctx.spawn_procedural_sphere(scene_id, name.as_ptr() as *const _, 1.0).unwrap();
      assert!(sphere_id > 0);

      let meas_name = alloc::ffi::CString::new("Meas").unwrap();
      let meas_id = ctx.spawn_entity(scene_id, meas_name.to_str().unwrap()).unwrap();
      let _ = ctx.add_measurement_component(scene_id, meas_id, Vec3f32::from_components(0.0, 0.0, 0.0), Vec3f32::from_components(1.0, 1.0, 1.0));

      let bill_name = alloc::ffi::CString::new("Bill").unwrap();
      let bill_id = ctx.spawn_entity(scene_id, bill_name.to_str().unwrap()).unwrap();
      let _ = ctx.add_image_billboard_component(scene_id, bill_id, false, 1.0, 1.0);

      let markers = [
          crate::structs::FfiMarker {
              position: [1.0, 1.0, 1.0],
              color: [1.0, 0.0, 0.0],
              size: 1.0,
          }
      ];
      let _ = ctx.set_markers(
        scene_id,
        sphere_id,
        &markers
      );

      let _ = ctx.set_bvh_node_visibility(scene_id, sphere_id, 0, true);

      let pe_handle = ctx.create_presentation_engine(256, 256).unwrap();

      for _ in 0..3 {
        let task_id = ctx.render_tick(pe_handle, scene_id, [256, 256]).unwrap().get();
        let mut status = ctx.get_task_status(task_id);
        let mut attempts = 0;
        while status == 0 && attempts < 20 {
          // Pending
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
          status = ctx.get_task_status(task_id);
          attempts += 1;
        }
        assert!(status == 1 || status == 2, "Render task stuck in pending");
      }

      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}
