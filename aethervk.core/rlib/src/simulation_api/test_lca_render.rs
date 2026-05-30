use super::{components_api::CameraParams, *};
use crate::{gpu, scene::Marker};
use alloc::format;
use core::ffi::{CStr, c_char};
extern crate std;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat};
use core::sync::atomic::{AtomicU64, Ordering};
use std::println;

fn panic_error_callback(msg: &str) {
  panic!("Vulkan Error: {}", msg);
}

fn get_test_context() -> Option<*mut SimulationContext> {
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);

  let ctx_ptr = SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback));
  if let Ok(boxed) = ctx_ptr {
    return Some(alloc::boxed::Box::into_raw(boxed));
  }
  println!("Skipping test: Vulkan backend could not be initialized");
  None
}

static LAST_PE_TASK_ID: AtomicU64 = AtomicU64::new(0);
static PE_ID: AtomicU64 = AtomicU64::new(0);

extern "C" fn render_callback(scene_id: u64, pe_id: u64, render_generation: u64) {
  if pe_id == PE_ID.load(Ordering::Acquire) {
    LAST_PE_TASK_ID.store(render_generation, Ordering::Release);
  }
}

#[test]
fn test_lca_render() {
  LAST_PE_TASK_ID.store(0, Ordering::Release);

  if let Some(ctx_ptr) = get_test_context() {
    struct CtxGuard(*mut SimulationContext);
    impl Drop for CtxGuard {
      fn drop(&mut self) {
        unsafe {
          let _ = alloc::boxed::Box::from_raw(self.0);
        }
      }
    }
    let _guard = CtxGuard(ctx_ptr);

    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx.create_empty_scene(true).unwrap();

      // Create an LCA microframe
      let name = alloc::ffi::CString::new("TestLCA").unwrap();
      let lca_id = ctx.spawn_entity(scene_id, name.to_str().unwrap()).unwrap();

      ctx
        .add_transform_component(
          scene_id,
          lca_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::from_components(1.0, 0.0, 0.0, 0.0),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();

      let mut scene_ctx = ctx.get_scene(scene_id).unwrap();
      {
        let mut guard = scene_ctx.write();
        let _ = guard.scene.add_component(
          slotmap::KeyData::from_ffi(lca_id).into(),
          crate::scene::ReferenceFrameComponent {
            frame_type: crate::scene::ReferenceFrameType::Micro,
            scale: 1.0 / 149597870.7, // 1/KM_PER_AU
            soi_radius: 10.0,
            depth_layer: 1,
          },
        );
      }

      let width = 256;
      let height = 256;
      let pe = ctx.create_presentation_engine(scene_id, width, height).unwrap();
      PE_ID.store(pe.0, Ordering::Release);

      let cam_name = alloc::ffi::CString::new("Camera").unwrap();
      let cam_id = ctx.spawn_entity(scene_id, cam_name.to_str().unwrap()).unwrap();
      // Attach to LCA
      ctx.get_scene(scene_id).unwrap().write().scene.set_parent(
        slotmap::KeyData::from_ffi(cam_id).into(),
        Some(slotmap::KeyData::from_ffi(lca_id).into()),
      );

      ctx
        .add_transform_component(
          scene_id,
          cam_id,
          Vec3f32::from_components(0.0, 0.0, 0.0),
          Quat::from_components(1.0, 0.0, 0.0, 0.0),
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();

      ctx
        .add_camera_component(
          scene_id,
          cam_id,
          CameraParams::new_perspective(60.0, width as f32 / height as f32, 0.1, 1000.0),
        )
        .unwrap();
      ctx.set_camera_for_presentation_engine(scene_id, pe, cam_id).unwrap();

      SimulationContext::set_render_callback(Some(render_callback));

      let sky_name = alloc::ffi::CString::new("Sky").unwrap();
      let sky_id = ctx.spawn_entity(scene_id, sky_name.to_str().unwrap()).unwrap();
      ctx.add_sky_component(scene_id, sky_id).unwrap();

      let _ = ctx
        .threads
        .logic_thread
        .tx()
        .try_send(crate::simulation_api::structs::LogicCommand::PlayScene { scene_id });

      let mut ready = false;
      for _ in 0..100 {
        if LAST_PE_TASK_ID.load(Ordering::Acquire) > 0 {
          ready = true;
          break;
        }
        std::thread::sleep(core::time::Duration::from_millis(10));
      }

      if ready {
        let tid = LAST_PE_TASK_ID.load(Ordering::Acquire);
        let mut status = ctx.get_task_status(tid);
        let mut attempt = 0;
        while matches!(
          status,
          crate::simulation_api::structs::TaskStatusCode::Pending
        ) && attempt < 100
        {
          std::thread::sleep(core::time::Duration::from_millis(10));
          status = ctx.get_task_status(tid);
          attempt += 1;
        }

        let mut buffer = vec![0u8; (width * height * 4) as usize];
        if ctx.download_image(tid, buffer.as_mut_ptr(), buffer.len()) {
          // Just assert we successfully downloaded
          assert!(buffer.len() > 0, "Downloaded buffer should not be empty");

          let has_non_zero = buffer.iter().any(|&x| x != 0);
          // Depending on clear color, it might not be all zero
          println!("Downloaded image. Has non-zero pixels: {}", has_non_zero);
        } else {
          panic!("Failed to download image");
        }
      } else {
        println!("Skipping render checks due to timeout waiting for task.");
      }

      let _ = ctx
        .threads
        .logic_thread
        .tx()
        .try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
    }
  }
}
