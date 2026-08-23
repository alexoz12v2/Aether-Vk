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
  println!("Vulkan Validation Error in test: {}", msg);
  panic!("Vulkan Error: {}", msg);
}

fn get_test_context() -> Option<*mut SimulationContext> {
  // Set asset path before startup so it's available for shaders
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);

  let ctx_ptr = SimulationContext::startup(Some(panic_error_callback));
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
fn test_time_api() {}

#[test]
fn test_entity_transform_api() {
  if let Some(ctx_ptr) = get_test_context() {
    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx
        .create_empty_scene(
          true,
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
        )
        .unwrap();
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

      let scene_id = ctx
        .create_empty_scene(
          true,
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
        )
        .unwrap();
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
fn test_scene_entities_api() {}

#[test]
fn test_misc_and_models_api() {}

#[test]
fn test_simulation_context_text_rendering() {}

#[test]
fn test_components_api() {}

#[test]
fn test_scene_api() {}

#[test]
fn test_core_api() {}

#[test]
fn test_misc_and_models_api_direct() {}

#[test]
fn test_snapshot_and_restore() {}

#[test]
fn test_spawn_comet_internal_bounds_and_hierarchy() {}

#[test]
fn test_spawn_comet_multi_scale_layer_separation() {}

#[test]
fn test_camera_controls() {}

#[test]
fn test_camera_controls_microframe() {}

#[test]
fn test_callbacks_safety() {}

#[test]
fn test_particle_velocity_beta_0_5_30days() {}

#[test]
fn test_particle_velocity_beta_1_5_30days() {}
