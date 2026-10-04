//! Integration test: macro-layer trajectories are depth-ordered against micro-layer meshes.
//!
//! Scene (camera at 100 AU, forward = -Y, screen up = +Z):
//! - a green emissive sphere (r = 300 km) in micro layer 1, 2000 km ahead;
//! - a red track (macro layer, AU control points) 1000 km ahead, 100 km above the sphere centre:
//!   in front of the sphere;
//! - a blue track 3000 km ahead, 100 km below the sphere centre: behind the sphere.
//!
//! Asserts:
//!  1. The red track is drawn over the sphere (trajectories write depth; before, the composite
//!     let any depth-writing micro fragment win over a macro trajectory).
//!  2. The blue track is hidden by the sphere but drawn beside it.
//!  3. GlobalDepth: sphere pixels carry (layer 1, distance), trajectory and sky pixels the
//!     (-1, -1) sentinel (MRT 1 is opt-in per archetype; trajectory.frag writes the sentinel).

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

static TASK_ID: AtomicU64 = AtomicU64::new(0);
static PE_ID: AtomicU64 = AtomicU64::new(0);

extern "C" fn render_callback(_scene_id: u64, pe_id: u64, render_generation: u64) {
  if pe_id == PE_ID.load(Ordering::Acquire) {
    TASK_ID.store(render_generation, Ordering::Release);
  }
}

/// Waits for a frame, then downloads the swapchain (BGRA8) and finalGlobalDepth (RG32F).
fn wait_and_download(
  ctx: &SimulationContext,
  width: u32,
  height: u32,
  timeout_ms: u64,
) -> Option<(alloc::vec::Vec<u8>, alloc::vec::Vec<[f32; 2]>)> {
  let poll = core::time::Duration::from_millis(10);
  let max_polls = timeout_ms / 10;
  // skip the first frames: the trajectory upload is deferred to the frame after extraction
  let mut seen = 0;
  let mut last = 0;
  for _ in 0..max_polls {
    let t = TASK_ID.load(Ordering::Acquire);
    if t != 0 && t != last {
      seen += 1;
      last = t;
      if seen >= 3 {
        break;
      }
    }
    std::thread::sleep(poll);
  }
  if seen < 3 {
    return None;
  }
  let tid = TASK_ID.load(Ordering::Acquire);
  for _ in 0..max_polls {
    if !matches!(ctx.get_task_status(tid), crate::simulation_api::structs::TaskStatusCode::Pending)
    {
      break;
    }
    std::thread::sleep(poll);
  }
  let mut color = alloc::vec![0u8; (width * height * 4) as usize];
  if !unsafe { ctx.download_image(tid, color.as_mut_ptr(), color.len()) } {
    return None;
  }
  let mut gdepth = alloc::vec![0u8; (width * height * 8) as usize];
  if !unsafe { ctx.download_global_depth_image(tid, gdepth.as_mut_ptr(), gdepth.len()) } {
    return None;
  }
  Some((color, bytemuck::cast_slice::<u8, [f32; 2]>(&gdepth).to_vec()))
}

// BGRA8 pixel classification
fn is_green(px: &[u8]) -> bool {
  px[1] > 128 && px[2] < 100 && px[0] < 100
}
fn is_red(px: &[u8]) -> bool {
  px[2] > 128 && px[1] < 100 && px[0] < 100
}
fn is_blue(px: &[u8]) -> bool {
  px[0] > 128 && px[1] < 100 && px[2] < 100
}

/// Straight cubic Bezier (4 control points) from `a` to `b`, in AU.
fn straight_track(a: [f64; 3], b: [f64; 3]) -> alloc::vec::Vec<[f64; 3]> {
  (0..4)
    .map(|k| {
      let t = k as f64 / 3.0;
      [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
    })
    .collect()
}

#[test]
fn test_trajectory_depth_order_against_micro_mesh() {
  let _render_lock = crate::simulation_api::lock_render_callback_tests();
  TASK_ID.store(0, Ordering::Release);

  let Some(ctx_ptr) = get_test_context() else {
    return;
  };
  // PauseScene, then drop the context, even when an assert fires (see test_multi_micro_render)
  struct PauseCtxGuard {
    ctx: *mut SimulationContext,
    scene_id: u64,
  }
  impl Drop for PauseCtxGuard {
    fn drop(&mut self) {
      unsafe {
        let _ = (*self.ctx).threads.logic_thread.tx().try_send(
          crate::simulation_api::structs::LogicCommand::PauseScene { scene_id: self.scene_id },
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = alloc::boxed::Box::from_raw(self.ctx);
      }
    }
  }
  let mut guard = PauseCtxGuard { ctx: ctx_ptr, scene_id: 0 };

  unsafe {
    let ctx = &mut *ctx_ptr;
    let scene_id = ctx
      .create_empty_scene(
        true,
        hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
        hifitime::Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
      )
      .unwrap();
    guard.scene_id = scene_id;

    const W: u32 = 512;
    const H: u32 = 512;
    const AU_TO_KM: f64 = 149_597_870.7;
    const CAM_AU: f64 = 1.0;
    // camera at (1, 1, 0) AU: the Sun is 45° off the view axis (outside the 60° FOV)
    const CAM_Y_AU: f64 = 1.0;
    const SPHERE_KM: f32 = 300.0;
    const SPHERE_FWD_KM: f64 = 2000.0;
    // (forward km, vertical km, colour)
    const FRONT: (f64, f64) = (1000.0, 100.0);
    const BACK: (f64, f64) = (3000.0, -100.0);
    const HALF_LEN_KM: f64 = 50_000.0;
    let fov = core::f32::consts::FRAC_PI_3;
    let focal_px = (H as f64 * 0.5) / (fov as f64 * 0.5).tan();

    // ── micro frame (layer 1) at the camera, green sphere 2000 km ahead ──
    let frame_id = ctx.spawn_entity(scene_id, "occl_frame").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        frame_id,
        Vec3f32::from_components(CAM_AU as f32, CAM_Y_AU as f32, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    let sphere_id = ctx.spawn_entity(scene_id, "occl_sphere").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        sphere_id,
        Vec3f32::from_components(0.0, -SPHERE_FWD_KM as f32, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    {
      let scene_ctx = ctx.get_scene(scene_id).unwrap();
      let mut g = scene_ctx.write();
      let root = g.root_entity;
      let frame: crate::scene::EntityId = slotmap::KeyData::from_ffi(frame_id).into();
      let sphere: crate::scene::EntityId = slotmap::KeyData::from_ffi(sphere_id).into();
      g.scene.set_parent(frame, Some(root));
      let _ = g.scene.add_component(
        frame,
        crate::scene::ReferenceFrameComponent {
          frame_type: crate::scene::ReferenceFrameType::Micro,
          scale: (1.0 / AU_TO_KM) as f32,
          soi_radius: 0.0001,
          depth_layer: 1,
        },
      );
      g.scene.set_parent(sphere, Some(frame));
      let _ = g.scene.add_component(
        sphere,
        crate::scene::StaticMeshComponent {
          asset_path: "occl_sphere".into(),
          mesh: alloc::sync::Arc::new(crate::simulation::comet::generate_uv_sphere(
            SPHERE_KM, 32, 32, 1.0, false,
          )),
          emissive_color: [0.0, 1.0, 0.0, 10.0],
          is_visible: true,
        },
      );

      // ── tracks: children of root (macro layer), absolute AU control points ──
      // A trajectory without segments first: the upload skips it, and the following ones must
      // still index their own TrajectoryGpu (they used the input index, i.e. the next one's).
      let empty = g.scene.spawn_entity("occl_empty_track");
      g.scene.set_parent(empty, Some(root));
      let _ = g.scene.add_component(empty, crate::scene::TransformComponent::default());
      let _ = g.scene.add_component(
        empty,
        crate::scene::trajectory::TrajectoryComponent::from_f64(
          alloc::vec::Vec::new(),
          [1.0, 1.0, 1.0, 1.0],
          2.0,
          0,
          32,
        ),
      );
      for (name, (fwd, up), color) in [
        ("occl_front_track", FRONT, [1.0, 0.0, 0.0, 1.0]),
        ("occl_back_track", BACK, [0.0, 0.0, 1.0, 1.0]),
      ] {
        let e = g.scene.spawn_entity(name);
        g.scene.set_parent(e, Some(root));
        let _ = g.scene.add_component(e, crate::scene::TransformComponent::default());
        let p = |x_km: f64| [CAM_AU + x_km / AU_TO_KM, CAM_Y_AU - fwd / AU_TO_KM, up / AU_TO_KM];
        let _ = g.scene.add_component(
          e,
          crate::scene::trajectory::TrajectoryComponent::from_f64(
            straight_track(p(-HALF_LEN_KM), p(HALF_LEN_KM)),
            color,
            2.0,
            0,
            32,
          ),
        );
      }
    }

    // ── camera: identity rotation (forward -Y) ──
    let cam_id = ctx.spawn_entity(scene_id, "occl_cam").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        cam_id,
        Vec3f32::from_components(CAM_AU as f32, CAM_Y_AU as f32, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    // near 1e-6 AU (150 km): the front track (1000 km) is inside the frustum
    ctx
      .add_camera_component(
        scene_id,
        cam_id,
        CameraParams::new_perspective(fov, W as f32 / H as f32, 1e-6, 1000.0),
      )
      .unwrap();

    let pe = ctx.create_presentation_engine(scene_id, W, H).unwrap();
    PE_ID.store(pe.0, Ordering::Release);
    ctx.set_camera_for_presentation_engine(scene_id, pe, cam_id).unwrap();
    SimulationContext::set_render_callback(Some(render_callback));
    let _ = ctx.threads.logic_thread.tx().try_send(
      crate::simulation_api::structs::LogicCommand::PlayScene {
        scene_id,
        speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
      },
    );

    let (color, gdepth) =
      wait_and_download(ctx, W, H, 15_000).expect("no frame rendered within 15 s");
    let px = |x: u32, y: u32| &color[((y * W + x) * 4) as usize..((y * W + x) * 4 + 4) as usize];
    let gd = |x: u32, y: u32| gdepth[(y * W + x) as usize];

    // ── sphere disk ──
    let (mut sx, mut sy, mut n) = (0u64, 0u64, 0u64);
    for y in 0..H {
      for x in 0..W {
        if is_green(px(x, y)) {
          sx += x as u64;
          sy += y as u64;
          n += 1;
        }
      }
    }
    assert!(n > 0, "green sphere not rendered");
    let (cx, cy) = ((sx / n) as u32, (sy / n) as u32);
    let r_px = ((n as f64) / core::f64::consts::PI).sqrt();
    println!("[occlusion_test] sphere centroid ({cx},{cy}), radius ~{r_px:.1} px");
    assert!(r_px > 40.0, "sphere too small to test occlusion: {r_px} px");

    // rows of the two tracks (screen up = +Z, so +100 km is above the centre)
    let front_row = (cy as f64 - focal_px * FRONT.1 / FRONT.0).round() as u32;
    let back_row = (cy as f64 - focal_px * BACK.1 / BACK.0).round() as u32;
    let find_row = |x: u32, row: u32, pred: fn(&[u8]) -> bool| {
      (row.saturating_sub(6)..=(row + 6).min(H - 1)).find(|&y| pred(px(x, y)))
    };

    // 1. red track (in front) drawn over the sphere
    let red_y = find_row(cx, front_row, is_red);
    assert!(
      red_y.is_some(),
      "red track (1000 km) not visible over the sphere (2000 km) at column {cx}, rows {front_row}±6"
    );
    let red_y = red_y.unwrap();
    assert!(
      (red_y as i64 - cy as i64).abs() < r_px as i64 - 5,
      "red track row {red_y} is not inside the sphere disk"
    );

    // 2. blue track (behind) hidden by the sphere, visible beside it
    assert!(
      find_row(cx, back_row, is_blue).is_none(),
      "blue track (3000 km) drawn over the sphere (2000 km) at column {cx}, rows {back_row}±6"
    );
    assert!(
      find_row(cx, back_row, is_green).is_some(),
      "sphere expected under the hidden blue track at column {cx}"
    );
    let beside = cx + r_px as u32 + 40;
    {
      let blue: alloc::vec::Vec<(u32, u32)> = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| is_blue(px(x, y)))
        .collect();
      println!("[occlusion_test] blue pixels: {} e.g. {:?}", blue.len(), &blue[..blue.len().min(8)]);
      for y in back_row.saturating_sub(8)..back_row + 8 {
        println!("[occlusion_test] col {beside} row {y}: {:?}", px(beside, y));
      }
      for (x, y) in [(4, 4), (4, 500), (500, 4), (cx, 4), (cx, 500), (4, cy), (beside, front_row), (cx, red_y)] {
        println!("[occlusion_test] px({x},{y}) = {:?} gd={:?}", px(x, y), gd(x, y));
      }
    }
    assert!(
      find_row(beside, back_row, is_blue).is_some(),
      "blue track not rendered beside the sphere at column {beside}, rows {back_row}±6"
    );

    // 3. GlobalDepth
    let sentinel = |v: [f32; 2]| v[0] == -1.0 && v[1] == -1.0;
    let g_red = gd(cx, red_y);
    assert!(sentinel(g_red), "trajectory pixel GlobalDepth {g_red:?}, expected (-1, -1)");
    let sphere_y = (cy as f64 + r_px * 0.5) as u32; // between the tracks, sphere only
    assert!(is_green(px(cx, sphere_y)), "expected a sphere-only pixel at ({cx},{sphere_y})");
    let g_sphere = gd(cx, sphere_y);
    assert!(
      (g_sphere[0] - 1.0).abs() < 0.5 && (g_sphere[1] - 1750.0).abs() < 300.0,
      "sphere pixel GlobalDepth {g_sphere:?}, expected (1, ~1750 km)"
    );
    let g_sky = gd(4, 4);
    assert!(sentinel(g_sky), "sky pixel GlobalDepth {g_sky:?}, expected (-1, -1)");
  }
}
