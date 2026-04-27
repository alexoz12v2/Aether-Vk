extern crate std;

use super::*;
use std::println;
use core::ffi::CStr;
use std::sync::Arc;

fn get_test_context() -> Option<*mut SimulationContext> {
  let backend = alloc::ffi::CString::new("Vulkan").unwrap();
  let ctx_ptr = SimulationContext::startup(backend.as_ptr(), 256, 256);
  if let Ok(ptr) = ctx_ptr {
    if ptr.is_null() {
      println!("Skipping test: Vulkan backend could not be initialized");
      return None;
    }
    return Some(ptr);
  }
  println!("Skipping test: Vulkan backend could not be initialized");
  None
}

#[test]
fn test_startup_and_shutdown() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      ctx.shutdown();
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
      assert_eq!(ctx.logic_state.read().current_scale, TimeScale::OneDay);

      ctx.set_time_scale(0); // Stopped
      assert_eq!(ctx.logic_state.read().current_scale, TimeScale::Stopped);

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_entity_transform_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx.active_scene_id;
      let name = alloc::ffi::CString::new("TestEntity").unwrap();
      let entity_id = ctx.spawn_entity(name.as_ptr()).unwrap();

      // Ensure entity exists
      assert!(entity_id > 0);

      // Add transform
      let res = ctx.add_transform_component(
        scene_id, entity_id, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
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
      assert!(res.unwrap());
      assert!((px - 1.0).abs() < 1e-5);
      assert!((py - 2.0).abs() < 1e-5);
      assert!((pz - 3.0).abs() < 1e-5);

      // Set transform
      let res = ctx.set_transform_component(
        scene_id, entity_id, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0,
      );
      assert!(res.is_ok());

      let res = ctx.get_transform_component(
        scene_id, entity_id, &mut px, &mut py, &mut pz, &mut rw, &mut rx, &mut ry, &mut rz,
        &mut sx, &mut sy, &mut sz,
      );
      assert!(res.unwrap());
      assert!((px - 4.0).abs() < 1e-5);
      assert!((py - 5.0).abs() < 1e-5);

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_camera_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx.active_scene_id;
      let name = alloc::ffi::CString::new("CameraEntity").unwrap();
      let entity_id = ctx.spawn_entity(name.as_ptr()).unwrap();

      // Transform is required before adding camera (or not, but good practice)
      let _ = ctx.add_transform_component(
        scene_id, entity_id, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
      );

      let res = ctx.add_camera_component(scene_id, entity_id, 60.0, 16.0 / 9.0, 0.1, 1000.0);
      assert!(res.is_ok());

      let mut proj = [0.0f32; 16];
      let res = ctx.get_camera_component(scene_id, entity_id, proj.as_mut_ptr());
      assert!(res.unwrap());

      // Set camera
      let res = ctx.set_camera_component(scene_id, entity_id, false, 90.0, 1.0, 0.1, 500.0);
      assert!(res.is_ok());

      let res = ctx.set_active_camera(entity_id);
      assert!(res.is_ok());

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_scene_entities_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.active_scene_id;

      let initial_count = ctx.get_entity_count();

      let name = alloc::ffi::CString::new("TestEntity2").unwrap();
      let entity_id = ctx.spawn_entity(name.as_ptr()).unwrap();

      assert_eq!(ctx.get_entity_count(), initial_count + 1);

      let mut out_name = [0i8; 64];
      let has_name = ctx.get_entity_name(entity_id, out_name.as_mut_ptr() as *mut c_char, 64);
      assert!(has_name);
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

      let res = ctx.remove_entity(entity_id);
      assert!(res.unwrap());
      assert_eq!(ctx.get_entity_count(), initial_count);

      ctx.shutdown();
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

      // Models API dummy test (import model will fail with fake path but shouldn't crash)
      let model_res = ctx.import_model(path.as_ptr());
      assert!(model_res.is_ok()); // The API returns Ok(0) on failure to import!
      assert_eq!(model_res.unwrap(), 0); // 0 is invalid model id

      // Test almanac file
      let almanac_res = ctx.load_almanac_file(path.as_ptr());
      assert!(almanac_res.is_ok());
      assert!(!almanac_res.unwrap()); // false because it fails to load

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_render_tick_and_download() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let name = alloc::ffi::CString::new("TestSphere").unwrap();
      let sphere_id = ctx.spawn_procedural_sphere(name.as_ptr(), 1.0).unwrap();
      assert!(sphere_id > 0);

      // Set clear color to green
      ctx.set_clear_color(0.0, 1.0, 0.0, 1.0);

      let task_id = ctx.render_tick();

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

      let width = 256;
      let height = 256;
      let mut buffer = vec![0u8; (width * height * 4) as usize];
      let success = ctx.download_image(buffer.as_mut_ptr(), buffer.len());
      if success {
        let sum: u32 = buffer.iter().map(|&b| b as u32).sum();
        assert!(sum > 0, "Buffer should have some data");
      }

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_process_command() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let cmd = FfiLogicCommand {
        cmd_type: FfiLogicCommandType::PanCamera,
        float_val_1: 10.0,
        float_val_2: -5.0,
        ulong_val: 0,
        bool_val: false,
      };

      avkSimulationContext_processCommand(ctx_ptr, cmd);

      let cmd2 = FfiLogicCommand {
        cmd_type: FfiLogicCommandType::ZoomCamera,
        float_val_1: 5.0,
        float_val_2: 0.0,
        ulong_val: 0,
        bool_val: false,
      };

      avkSimulationContext_processCommand(ctx_ptr, cmd2);

      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));

      avkSimulationContext_shutdown(ctx_ptr);
    }
  }
}

#[test]
fn test_multiple_scenes() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let new_scene_id = ctx.create_default_scene(256, 256).unwrap();
      assert!(new_scene_id > 0);

      let count = avkSimulationContext_getEntityCount(ctx);
      assert!(count > 0);

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}

#[test]
fn test_all_archetypes() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;
      let scene_id = ctx.active_scene_id;

      let name = alloc::ffi::CString::new("TestSphere").unwrap();
      let sphere_id = ctx.spawn_procedural_sphere(name.as_ptr(), 1.0).unwrap();
      assert!(sphere_id > 0);

      let meas_name = alloc::ffi::CString::new("Meas").unwrap();
      let meas_id = ctx.spawn_entity(meas_name.as_ptr()).unwrap();
      let _ = ctx.add_measurement_component(scene_id, meas_id, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

      let bill_name = alloc::ffi::CString::new("Bill").unwrap();
      let bill_id = ctx.spawn_entity(bill_name.as_ptr()).unwrap();
      let _ = ctx.add_image_billboard_component(scene_id, bill_id, false, 1.0, 1.0);

      let px = [1.0];
      let py = [1.0];
      let pz = [1.0];
      let cr = [1.0];
      let cg = [0.0];
      let cb = [0.0];
      let sizes = [1.0];
      let _ = ctx.set_markers(
        scene_id,
        sphere_id,
        1,
        px.as_ptr(),
        py.as_ptr(),
        pz.as_ptr(),
        cr.as_ptr(),
        cg.as_ptr(),
        cb.as_ptr(),
        sizes.as_ptr(),
      );

      let _ = ctx.set_bvh_node_visibility(scene_id, sphere_id, 0, true);

      for _ in 0..3 {
        let task_id = ctx.render_tick();
        let mut status = ctx.get_task_status(task_id);
        let mut attempts = 0;
        while status == 0 && attempts < 20 {
          // Pending
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(50));
          status = ctx.get_task_status(task_id);
          attempts += 1;
        }
      }

      ctx.shutdown();
      let _ = alloc::boxed::Box::from_raw(ctx_ptr);
    }
  }
}
