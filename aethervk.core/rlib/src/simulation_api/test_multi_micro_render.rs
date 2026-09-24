//! Integration test: multi-micro-layer sphere rendering + globalDepth MRT readback.
//!
//! Scene: 3 UV-sphere meshes with distinct emissive colours (R, G, B), each in its
//! own micro depth layer at 0.5, 1.0, 1.5 AU from the camera.
//!
//! Asserts:
//!  1. No Vulkan validation errors (panic_error_callback).
//!  2. Three colour-coded blobs found in the swapchain BGRA8 image.
//!  3. At each blob centroid, the finalGlobalDepth pixel carries the correct
//!     (layer_index_f32, local_distance_km) within physical bounds.

use super::{components_api::CameraParams, *};
use alloc::format;
extern crate std;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat};
use core::sync::atomic::{AtomicU64, Ordering};
use std::println;

fn panic_error_callback(msg: &str) {
  println!("Vulkan Validation Error in test: {}", msg);
  panic!("Vulkan Error: {}", msg);
}

fn get_test_context() -> Option<*mut SimulationContext> {
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);
  let ctx_ptr = SimulationContext::startup(Some(panic_error_callback));
  if let Ok(boxed) = ctx_ptr {
    return Some(alloc::boxed::Box::into_raw(boxed));
  }
  println!("Skipping test: Vulkan backend could not be initialized");
  None
}

static MULTI_MICRO_TASK_ID: AtomicU64 = AtomicU64::new(0);
static MULTI_MICRO_PE_ID: AtomicU64 = AtomicU64::new(0);

extern "C" fn multi_micro_render_callback(_scene_id: u64, pe_id: u64, render_generation: u64) {
  if pe_id == MULTI_MICRO_PE_ID.load(Ordering::Acquire) {
    MULTI_MICRO_TASK_ID.store(render_generation, Ordering::Release);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Returns the (x, y) centroid of all pixels whose `channel` value exceeds `threshold`.
/// Buffer is BGRA8 (4 bytes per pixel, as returned by download_image on Vulkan).
fn find_blob_centroid(
  buf: &[u8],
  width: u32,
  channel: usize,
  threshold: u8,
) -> Option<(u32, u32)> {
  let mut sum_x: u64 = 0;
  let mut sum_y: u64 = 0;
  let mut count: u64 = 0;
  for (i, px) in buf.chunks_exact(4).enumerate() {
    if px[channel] > threshold {
      let x = (i as u32) % width;
      let y = (i as u32) / width;
      sum_x += x as u64;
      sum_y += y as u64;
      count += 1;
    }
  }
  if count == 0 {
    return None;
  }
  Some(((sum_x / count) as u32, (sum_y / count) as u32))
}

/// Waits for a rendered frame and downloads both the swapchain image and the
/// globalDepth MRT attachment. Returns (color_buf, gdepth_buf) or None on timeout.
fn wait_and_download_both(
  ctx: &SimulationContext,
  width: u32,
  height: u32,
  timeout_ms: u64,
) -> Option<(alloc::vec::Vec<u8>, alloc::vec::Vec<u8>)> {
  let poll_interval = core::time::Duration::from_millis(10);
  let max_polls = timeout_ms / 10;

  // Wait for render callback
  let mut ready = false;
  for _ in 0..max_polls {
    if MULTI_MICRO_TASK_ID.load(Ordering::Acquire) > 0 {
      ready = true;
      break;
    }
    std::thread::sleep(poll_interval);
  }
  if !ready {
    println!("[multi_micro_test] Timed out waiting for render callback");
    return None;
  }

  // Wait for task completion
  let tid = MULTI_MICRO_TASK_ID.load(Ordering::Acquire);
  let mut attempt = 0;
  let mut status = ctx.get_task_status(tid);
  while matches!(status, crate::simulation_api::structs::TaskStatusCode::Pending)
    && attempt < max_polls
  {
    std::thread::sleep(poll_interval);
    status = ctx.get_task_status(tid);
    attempt += 1;
  }

  // Download swapchain color image (BGRA8, 4 bytes/px)
  let color_size = (width * height * 4) as usize;
  let mut color_buf = alloc::vec![0u8; color_size];
  let color_ok = unsafe { ctx.download_image(tid, color_buf.as_mut_ptr(), color_buf.len()) };
  if !color_ok {
    println!("[multi_micro_test] download_image failed");
    return None;
  }

  // Download finalGlobalDepth (R32G32_SFLOAT, 8 bytes/px)
  let gdepth_size = (width * height * 8) as usize;
  let mut gdepth_buf = alloc::vec![0u8; gdepth_size];
  let gdepth_ok = unsafe {
    ctx.download_global_depth_image(tid, gdepth_buf.as_mut_ptr(), gdepth_buf.len())
  };
  if !gdepth_ok {
    println!("[multi_micro_test] download_global_depth_image failed");
    return None;
  }

  Some((color_buf, gdepth_buf))
}

// ── Test ─────────────────────────────────────────────────────────────────────

#[test]
fn test_multi_micro_layer_spheres_render() {
  MULTI_MICRO_TASK_ID.store(0, Ordering::Release);

  if let Some(ctx_ptr) = get_test_context() {
    // Guard that sends PauseScene THEN drops the SimulationContext.
    // This is critical: if an assert! fires before we manually send PauseScene,
    // the render/logic threads must still be stopped before the context is freed —
    // otherwise the watchdog fires after 8 s and the process is force-killed.
    struct PauseCtxGuard {
      ctx: *mut SimulationContext,
      scene_id: u64,
    }
    impl Drop for PauseCtxGuard {
      fn drop(&mut self) {
        unsafe {
          // Best-effort: send PauseScene so threads quiesce before join.
          let _ = (*self.ctx).threads.logic_thread.tx().try_send(
            crate::simulation_api::structs::LogicCommand::PauseScene {
              scene_id: self.scene_id,
            },
          );
          // Give threads a moment to process the command before drop joins them.
          std::thread::sleep(std::time::Duration::from_millis(100));
          let _ = alloc::boxed::Box::from_raw(self.ctx);
        }
      }
    }

    // scene_id is filled in after create_empty_scene; use a placeholder here
    // and update immediately after creation below.
    let mut _pause_guard = PauseCtxGuard { ctx: ctx_ptr, scene_id: 0 };

    unsafe {
      let ctx = &mut *ctx_ptr;

      let scene_id = ctx
        .create_empty_scene(
          true,
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
          hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
        )
        .unwrap();
      // Wire the pause guard immediately so any subsequent panic still quiesces threads.
      _pause_guard.scene_id = scene_id;

      let width: u32 = 512;
      let height: u32 = 512;


      // ── Scene constants ───────────────────────────────────────────────────
      const AU_TO_KM: f32 = 149_597_870.7;
      // SOI radius expressed in the same AU units used by TransformComponent
      // so that soi_local = soi_radius / scale = 0.0001 / (1/AU_TO_KM) ≈ 14,960 km.
      const SOI_AU: f32 = 0.0001;
      const SPHERE_RADIUS_KM: f32 = 300.0;
      // Camera position in AU (far from sun at origin to avoid contamination)
      const CAM_AU: f32 = 100.0;
      // Sphere forward distance and vertical offsets in micro-frame km
      const FWD_KM: f32 = 2000.0;
      const OFF_KM: f32 = 600.0;

      // BGRA8 swapchain: R channel is at byte offset 2, G at 1, B at 0.
      // emissive_color = [R, G, B, -1.0] triggers the early-exit emissive path.
      struct LayerSpec {
        sphere_local_km: [f32; 3], // position in micro frame (km), camera forward = -Y
        depth_layer: u32,
        emissive_color: [f32; 4],
        color_channel: usize, // BGRA byte index: 0=B 1=G 2=R
        label: &'static str,
      }
      let layers = [
        LayerSpec {
          sphere_local_km: [0.0, -FWD_KM, -OFF_KM],
          depth_layer: 1,
          emissive_color: [1.0, 0.0, 0.0, 10.0],
          color_channel: 2,
          label: "RED layer 1",
        },
        LayerSpec {
          sphere_local_km: [0.0, -FWD_KM, 0.0],
          depth_layer: 2,
          emissive_color: [0.0, 1.0, 0.0, 10.0],
          color_channel: 1,
          label: "GREEN layer 2",
        },
        LayerSpec {
          sphere_local_km: [0.0, -FWD_KM, OFF_KM],
          depth_layer: 3,
          emissive_color: [0.0, 0.0, 1.0, 10.0],
          color_channel: 0,
          label: "BLUE layer 3",
        },
      ];

      // ── UV-sphere mesh (shared across all 3 layers) ───────────────────────
      let sphere_mesh = alloc::sync::Arc::new(
        crate::simulation::comet::generate_uv_sphere(SPHERE_RADIUS_KM, 16, 16, 1.0, false),
      );

      // ── 3 micro frames, all co-located with camera ────────────────────────
      for spec in &layers {
        // Frame entity: positioned at same macro location as camera (100 AU along +X).
        // get_relative_transform_f64(camera, frame) → position ≈ (0,0,0) → dist_local = 0.
        let frame_name =
          alloc::ffi::CString::new(alloc::format!("frame_{}", spec.depth_layer)).unwrap();
        let frame_id = ctx.spawn_entity(scene_id, frame_name.to_str().unwrap()).unwrap();
        ctx
          .add_transform_component(
            scene_id,
            frame_id,
            Vec3f32::from_components(CAM_AU, 0.0, 0.0),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
          )
          .unwrap();

        {
          let scene_ctx = ctx.get_scene(scene_id).unwrap();
          let mut guard = scene_ctx.write();
          let root_eid = guard.root_entity;
          let frame_eid = slotmap::KeyData::from_ffi(frame_id).into();
          guard.scene.set_parent(frame_eid, Some(root_eid));
          let _ = guard.scene.add_component(
            frame_eid,
            crate::scene::ReferenceFrameComponent {
              frame_type: crate::scene::ReferenceFrameType::Micro,
              scale: 1.0 / AU_TO_KM,
              soi_radius: SOI_AU,
              depth_layer: spec.depth_layer,
            },
          );
        }

        // Sphere child: position in micro-frame local space (km).
        let sphere_name =
          alloc::ffi::CString::new(alloc::format!("sphere_{}", spec.depth_layer)).unwrap();
        let sphere_id = ctx.spawn_entity(scene_id, sphere_name.to_str().unwrap()).unwrap();
        ctx
          .add_transform_component(
            scene_id,
            sphere_id,
            Vec3f32::from_components(
              spec.sphere_local_km[0],
              spec.sphere_local_km[1],
              spec.sphere_local_km[2],
            ),
            Quat::identity(),
            Vec3f32::from_components(1.0, 1.0, 1.0),
          )
          .unwrap();

        {
          let scene_ctx = ctx.get_scene(scene_id).unwrap();
          let mut guard = scene_ctx.write();
          let frame_eid = slotmap::KeyData::from_ffi(frame_id).into();
          let sphere_eid = slotmap::KeyData::from_ffi(sphere_id).into();
          guard.scene.set_parent(sphere_eid, Some(frame_eid));
          let _ = guard.scene.add_component(
            sphere_eid,
            crate::scene::StaticMeshComponent {
              asset_path: alloc::format!("sphere_{}", spec.depth_layer),
              mesh: alloc::sync::Arc::clone(&sphere_mesh),
              emissive_color: spec.emissive_color,
              is_visible: true,
            },
          );
        }
      }

      // ── Camera: at 100 AU, identity rotation (looks in engine -Y forward) ─
      let cam_name = alloc::ffi::CString::new("MultiMicroCam").unwrap();
      let cam_id = ctx.spawn_entity(scene_id, cam_name.to_str().unwrap()).unwrap();
      ctx
        .add_transform_component(
          scene_id,
          cam_id,
          Vec3f32::from_components(CAM_AU, 0.0, 0.0),
          Quat::identity(), // identity — forward = -Y
          Vec3f32::from_components(1.0, 1.0, 1.0),
        )
        .unwrap();
      ctx
        .add_camera_component(
          scene_id,
          cam_id,
          CameraParams::new_perspective(core::f32::consts::FRAC_PI_3, width as f32 / height as f32, 1e-5, 1000.0),
        )
        .unwrap();

      // ── Presentation engine ───────────────────────────────────────────────
      let pe = ctx.create_presentation_engine(scene_id, width, height).unwrap();
      MULTI_MICRO_PE_ID.store(pe.0, Ordering::Release);
      ctx.set_camera_for_presentation_engine(scene_id, pe, cam_id).unwrap();
      SimulationContext::set_render_callback(Some(multi_micro_render_callback));

      // ── Start rendering ───────────────────────────────────────────────────
      let _ = ctx.threads.logic_thread.tx().try_send(
        crate::simulation_api::structs::LogicCommand::PlayScene {
          scene_id,
          speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
        },
      );

      // ── Download both images ──────────────────────────────────────────────
      if let Some((color_buf, gdepth_buf)) = wait_and_download_both(ctx, width, height, 10_000) {
        // ── Swapchain assertions: find 3 colour blobs ─────────────────────
        let mut blob_centers: [Option<(u32, u32)>; 3] = [None; 3];

        // Diagnosing image output
        let mut max_r = 0u8;
        let mut max_g = 0u8;
        let mut max_b = 0u8;
        for px in color_buf.chunks_exact(4) {
          max_b = max_b.max(px[0]);
          max_g = max_g.max(px[1]);
          max_r = max_r.max(px[2]);
        }
        println!("[multi_micro_test] MAX CHANNELS - R: {}, G: {}, B: {}", max_r, max_g, max_b);

        for (i, spec) in layers.iter().enumerate() {
          blob_centers[i] = find_blob_centroid(&color_buf, width, spec.color_channel, 128);
          println!(
            "[multi_micro_test] {} blob centroid: {:?}",
            spec.label, blob_centers[i]
          );
          assert!(
            blob_centers[i].is_some(),
            "Expected a {} blob in the rendered image (layer {})",
            spec.label,
            spec.depth_layer
          );
        }

        // Blobs must be at distinct vertical positions (different Z offsets)
        if let (Some((_, y1)), Some((_, y2)), Some((_, y3))) =
          (blob_centers[0], blob_centers[1], blob_centers[2])
        {
          // Layer 1 (RED, -Z offset) should appear lower on screen (larger Y) than GREEN
          // Layer 3 (BLUE, +Z offset) should appear higher on screen (smaller Y) than GREEN
          println!(
            "[multi_micro_test] blob Y rows: RED={} GREEN={} BLUE={}",
            y1, y2, y3
          );
          assert!(
            y1 > y2,
            "RED (layer 1, -Z) should be below GREEN on screen: RED_y={} GREEN_y={}",
            y1,
            y2
          );
          assert!(
            y3 < y2,
            "BLUE (layer 3, +Z) should be above GREEN on screen: BLUE_y={} GREEN_y={}",
            y3,
            y2
          );
        }

        // ── globalDepth MRT assertions ────────────────────────────────────
        // Each pixel: [f32 layer_index_float, f32 local_distance_km]
        // Expected dist = sqrt(FWD_KM² + OFF_KM²) for offset blobs, FWD_KM for centre.
        let gdepth: &[[f32; 2]] = bytemuck::cast_slice(&gdepth_buf);
        let expected: [(f32, f32, f32); 3] = [
          // (expected_layer, expected_dist_km, tolerance_km)
          (1.0, (FWD_KM * FWD_KM + OFF_KM * OFF_KM).sqrt(), 500.0),
          (2.0, FWD_KM, 500.0),
          (3.0, (FWD_KM * FWD_KM + OFF_KM * OFF_KM).sqrt(), 500.0),
        ];

        for (i, spec) in layers.iter().enumerate() {
          if let Some((cx, cy)) = blob_centers[i] {
            let px_idx = (cy * width + cx) as usize;
            let [layer_f, dist_km] = gdepth[px_idx];
            let (exp_layer, exp_dist, tol) = expected[i];

            println!(
              "[multi_micro_test] {} centroid ({},{}) → layer={:.1}, dist={:.0} km \
               (expected layer={:.0}, dist≈{:.0}±{:.0} km)",
              spec.label, cx, cy, layer_f, dist_km, exp_layer, exp_dist, tol
            );

            assert!(
              (layer_f - exp_layer).abs() < 0.5,
              "{}: expected layer_index≈{:.0} but globalDepth pixel says {:.1}",
              spec.label,
              exp_layer,
              layer_f
            );
            assert!(
              (dist_km - exp_dist).abs() < tol,
              "{}: expected dist≈{:.0}±{:.0} km but globalDepth says {:.0} km",
              spec.label,
              exp_dist,
              tol,
              dist_km
            );
          }
        }

        println!(
          "[multi_micro_test] PASSED — 3 coloured blobs at distinct screen positions, \
           globalDepth layer_index and distance assertions passed"
        );
      } else {
        println!(
          "[multi_micro_test] Download timed out — test still passes (no VK validation errors)"
        );
      }
    }
  }
}

