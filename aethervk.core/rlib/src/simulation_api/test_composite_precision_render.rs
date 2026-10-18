//! Integration test: macro↔micro depth ordering at Earth-observer distances
//! (`.agents/briefs/04-composite-precision-plan.md` §4.1).
//!
//! Scene (camera at (1, 1, 0) AU looking −Y, a child of root, screen right = world −X, screen
//! up = +Z, Sun hidden):
//! - a micro frame (its own depth layer, 4) D AU ahead, with the app's SOI (`soi_radius = D` AU;
//!   the SOI window D ± D is only the fallback: the depth-fitting pass of `scene_conversion.rs`
//!   refits it to the visible meshes), `scale = 1/AU_TO_KM`; a green emissive sphere r = 2 km at
//!   its origin;
//! - a red macro track (layer 0, AU control points) on the centre row, tilted in depth:
//!   depth = −2 km + 2·lateral relative to the nucleus centre, lateral −12 .. +12 km. It passes the
//!   front pole: on the left half of the disc it is in front of the surface, on the right half
//!   behind it (a constant ordering bias of either sign misorders one of the halves);
//! - FOV = 32 km / D (the nucleus spans 64 px of 512, as a telescope field).
//!
//! An f64 reference per pixel of the two centre rows (the track ribbon's core) says which surface
//! is nearer; pixels with |Δz| < 20 m (the crossing, the tessellated limb) are skipped.
//! Asserts: (1) no misordered pixel; (2) the track's row (off the disc) within 1 px of the centre
//! row; (3) GlobalDepth: a nucleus-only pixel reads (4, its distance ± 2·ulp(D)), a track pixel
//! and a sky pixel the (−1, −1) sentinel. Printed: the counts, the max |Δz| among the misordered.
//! Before the precision fix the ordering floor is ~60 km·(D/AU) (§1.2): the test fails.

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
/// (Copy of `test_trajectory_occlusion_render::wait_and_download`.)
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
    if !matches!(
      ctx.get_task_status(tid),
      crate::simulation_api::structs::TaskStatusCode::Pending
    ) {
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
  Some((
    color,
    bytemuck::cast_slice::<u8, [f32; 2]>(&gdepth).to_vec(),
  ))
}

// BGRA8 pixel classification (as test_trajectory_occlusion_render)
fn is_green(px: &[u8]) -> bool {
  px[1] > 128 && px[2] < 100 && px[0] < 100
}
fn is_red(px: &[u8]) -> bool {
  px[2] > 128 && px[1] < 100 && px[0] < 100
}

/// Straight cubic Bezier (4 control points) from `a` to `b`, in AU.
fn straight_track(a: [f64; 3], b: [f64; 3]) -> alloc::vec::Vec<[f64; 3]> {
  (0..4)
    .map(|k| {
      let t = k as f64 / 3.0;
      [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
      ]
    })
    .collect()
}

/// f32 ulp of `x` (> 0)
fn ulp_f32(x: f64) -> f64 {
  2f64.powi(x.log2().floor() as i32 - 23)
}

const W: u32 = 512;
const H: u32 = 512;
const AU_TO_KM: f64 = 149_597_870.7;
const CAM: [f64; 3] = [1.0, 1.0, 0.0];
/// A depth layer of its own: the empty scene's Earth subtree is layer 1, the Sun 2, the comet 3
/// (`scene_api.rs`: "every micro frame needs its own depth layer").
const LAYER: u32 = 4;
/// Track line width: at AU distances the ribbon is `min(max(w, ~0), 4w)` = w px wide with an
/// opaque core of 0.75 w (`trajectory.vert/.frag`); 2.0 gave a 2 px ribbon at alpha 0.5 (127).
const LINE_WIDTH: f32 = 8.0;
const R_KM: f64 = 2.0;
/// track: depth (km, + = farther, relative to the nucleus centre) = DEPTH0 + SLOPE · lateral
const DEPTH0_KM: f64 = -2.0;
const SLOPE: f64 = 2.0;
const HALF_LEN_KM: f64 = 12.0;
/// pixels whose two surfaces are nearer than this in depth are not classified (§4.1)
const SKIP_KM: f64 = 0.020;

/// Distance offsets from D (km) along the ray of pixel (tx, ty) (tangents of the pixel angle):
/// the sphere's front surface, `None` when the ray misses it. Formed without cancellation
/// (every term is a small number), so f64 resolves micrometres at 5 AU.
fn sphere_offset(d: f64, r: f64, tx: f64, ty: f64) -> Option<f64> {
  let t2 = tx * tx + ty * ty;
  let q = (d * tx) * (d * tx) + (d * ty) * (d * ty); // = d²·t2
  let dd = 1.0 + t2;
  let disc = r * r * dd - q;
  if !(disc > 0.0) {
    return None;
  }
  let sdd = dd.sqrt();
  // distance = (d − sqrt(disc)) / sqrt(dd); d / sqrt(dd) − d = −d·t2 / (sqrt(dd)(1 + sqrt(dd)))
  Some(-(q / d) / (sdd * (1.0 + sdd)) - disc.sqrt() / sdd)
}

/// The track point seen in pixel column `tx`: (lateral km, distance offset from D, km).
/// Lateral s on the line satisfies s = tx · (d + DEPTH0 + SLOPE·s).
fn track_offset(d: f64, tx: f64) -> (f64, f64) {
  let s = tx * (d + DEPTH0_KM) / (1.0 - SLOPE * tx);
  let z = DEPTH0_KM + SLOPE * s;
  let f = d + z;
  // sqrt(s² + f²) − d = z + (sqrt(s² + f²) − f)
  (s, z + s * s / ((s * s + f * f).sqrt() + f))
}

fn run_composite_precision(d_au: f64) {
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
          crate::simulation_api::structs::LogicCommand::PauseScene {
            scene_id: self.scene_id,
          },
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = alloc::boxed::Box::from_raw(self.ctx);
      }
    }
  }
  let mut guard = PauseCtxGuard {
    ctx: ctx_ptr,
    scene_id: 0,
  };

  let tag = format!("[composite_precision D={d_au} AU]");
  // the nucleus, D ahead along −Y (all values exact in f32)
  let nucleus = [CAM[0], CAM[1] - d_au, CAM[2]];
  let scale_f32 = (1.0 / AU_TO_KM) as f32;
  let d_km = d_au * AU_TO_KM;
  // vertical FOV; the engine projects with `fov as f64`
  let fov = (8.0 * (2.0 * R_KM) / d_km) as f32;
  let focal_px = (H as f64 * 0.5) / (fov as f64 * 0.5).tan();
  // sphere radius in AU-consistent km (the micro layer's km is 1/scale_f32 AU, not 1/AU_TO_KM)
  let r_km = R_KM * scale_f32 as f64 * AU_TO_KM;

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

    // ── micro frame (layer 1) with the app's SOI, green sphere at its origin ──
    let frame_id = ctx.spawn_entity(scene_id, "prec_frame").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        frame_id,
        Vec3f32::from_components(nucleus[0] as f32, nucleus[1] as f32, nucleus[2] as f32),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    let sphere_id = ctx.spawn_entity(scene_id, "prec_sphere").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        sphere_id,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    {
      let scene_ctx = ctx.get_scene(scene_id).unwrap();
      let g = scene_ctx.write();
      let root = g.root_entity;
      let frame: crate::scene::EntityId = slotmap::KeyData::from_ffi(frame_id).into();
      let sphere: crate::scene::EntityId = slotmap::KeyData::from_ffi(sphere_id).into();
      g.scene.set_parent(frame, Some(root));
      let _ = g.scene.add_component(
        frame,
        crate::scene::ReferenceFrameComponent {
          frame_type: crate::scene::ReferenceFrameType::Micro,
          scale: scale_f32,
          soi_radius: d_au as f32,
          depth_layer: LAYER,
        },
      );
      g.scene.set_parent(sphere, Some(frame));
      let _ = g.scene.add_component(
        sphere,
        crate::scene::StaticMeshComponent {
          asset_path: "prec_sphere".into(),
          mesh: alloc::sync::Arc::new(crate::simulation::comet::generate_uv_sphere(
            R_KM as f32,
            64,
            64,
            1.0,
            false,
          )),
          emissive_color: [0.0, 1.0, 0.0, 10.0],
          is_visible: true,
        },
      );

      // Sun hidden, as in test_trajectory_occlusion_render
      let mut suns = alloc::vec::Vec::new();
      g.scene.query1(|id, _s: &crate::scene::SunComponent| suns.push(id));
      for id in suns {
        let _ = g.scene.add_component(id, crate::scene::HiddenComponent {});
      }

      // The empty scene's own micro subtrees are hidden too (Earth, layer 1; the comet, layer 3,
      // whose default nucleus was drawn over this test's sphere: the 1 AU "before" run read
      // layer 3 at the nucleus pixel, a dark grey disc, and only 22 green px poked through).
      // A hidden frame drops out of the layer maps, and its subtree out of every extraction.
      let mut others = alloc::vec::Vec::new();
      g.scene.query1(|id, f: &crate::scene::ReferenceFrameComponent| {
        if f.depth_layer > 0 && id != frame {
          others.push((id, f.depth_layer));
        }
      });
      for (id, layer) in others {
        let pos = g.scene.global_transform_f64(id).map(|t| {
          use aethervk_oshal_rlib::math::vector::Vector3;
          [t.position.x(), t.position.y(), t.position.z()]
        });
        println!("[composite_precision D={d_au} AU] hiding the scene's frame on layer {layer} at {pos:?} AU");
        let _ = g.scene.add_component(id, crate::scene::HiddenComponent {});
      }

      // ── the tilted red track: child of root (macro layer), absolute AU control points ──
      // (lateral s, depth z) km → world: x = N.x − s (screen right = −X), y = N.y − z (−Y ahead)
      let p = |s_km: f64| {
        let z_km = DEPTH0_KM + SLOPE * s_km;
        [
          nucleus[0] - s_km / AU_TO_KM,
          nucleus[1] - z_km / AU_TO_KM,
          nucleus[2],
        ]
      };
      let e = g.scene.spawn_entity("prec_track");
      g.scene.set_parent(e, Some(root));
      let _ = g.scene.add_component(e, crate::scene::TransformComponent::default());
      let _ = g.scene.add_component(
        e,
        crate::scene::trajectory::TrajectoryComponent::from_f64(
          straight_track(p(-HALF_LEN_KM), p(HALF_LEN_KM)),
          [1.0, 0.0, 0.0, 1.0],
          LINE_WIDTH,
          0,
          32,
        ),
      );
    }

    // ── camera: identity rotation (forward −Y), Earth-observer near/far ──
    let cam_id = ctx.spawn_entity(scene_id, "prec_cam").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        cam_id,
        Vec3f32::from_components(CAM[0] as f32, CAM[1] as f32, CAM[2] as f32),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    // The camera is a child of root. (Without a common ancestor
    // `get_relative_transform_f64(camera, frame)` used to return the difference of the global
    // positions *not divided by the frame scale* (AU read as km): a frame 1 AU away put the
    // camera 1 km from the nucleus, inside the sphere. Fixed in scene.rs (track 9E), test
    // `scene::tests::relative_transform_f64_without_common_ancestor_is_in_frame_units`.)
    {
      let scene_ctx = ctx.get_scene(scene_id).unwrap();
      let g = scene_ctx.write();
      let root = g.root_entity;
      let cam: crate::scene::EntityId = slotmap::KeyData::from_ffi(cam_id).into();
      g.scene.set_parent(cam, Some(root));
    }
    ctx
      .add_camera_component(
        scene_id,
        cam_id,
        CameraParams::new_perspective(fov, W as f32 / H as f32, 1e-9, 10.0),
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
    let tan_x = |x: u32| (x as f64 + 0.5 - W as f64 * 0.5) / focal_px;
    let tan_y = |y: u32| (H as f64 * 0.5 - (y as f64 + 0.5)) / focal_px;

    // the disc as rendered (information only: the classification uses the f64 geometry)
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
    let r_px_expected = focal_px * r_km / d_km;
    println!(
      "{tag} green pixels {n}, centroid ({:.1}, {:.1}), radius ~{:.1} px (expected {:.1} px at \
       ({:.1}, {:.1}))",
      sx as f64 / n.max(1) as f64,
      sy as f64 / n.max(1) as f64,
      ((n as f64) / core::f64::consts::PI).sqrt(),
      r_px_expected,
      W as f64 * 0.5 - 0.5,
      H as f64 * 0.5 - 0.5
    );

    // ── (1) ordering on the two centre rows (the ribbon core) ──
    let (mut checked, mut checked_red, mut checked_green) = (0u32, 0u32, 0u32);
    let (mut bad, mut bad_red, mut bad_green) = (0u32, 0u32, 0u32);
    let mut bad_max_dz = 0.0f64;
    let mut first_bad = alloc::vec::Vec::new();
    let mut track_pixel_on_disc = None;
    for y in [H / 2 - 1, H / 2] {
      for x in 0..W {
        let (tx, ty) = (tan_x(x), tan_y(y));
        // inside 0.9 r laterally: the 64×64 tessellation is within a few m of the sphere there
        if (tx * d_km).abs() > 0.9 * r_km {
          continue;
        }
        let Some(zs) = sphere_offset(d_km, r_km, tx, ty) else {
          continue;
        };
        let (s, zt) = track_offset(d_km, tx);
        if s.abs() > HALF_LEN_KM {
          continue;
        }
        let dz = zs - zt; // > 0: the track is nearer
        if dz.abs() < SKIP_KM {
          continue;
        }
        let expect_red = dz > 0.0;
        let c = px(x, y);
        let ok = if expect_red { is_red(c) } else { is_green(c) };
        checked += 1;
        if expect_red {
          checked_red += 1;
        } else {
          checked_green += 1;
        }
        if expect_red && is_red(c) && track_pixel_on_disc.is_none() {
          track_pixel_on_disc = Some((x, y));
        }
        if !ok {
          bad += 1;
          if expect_red {
            bad_red += 1;
          } else {
            bad_green += 1;
          }
          bad_max_dz = bad_max_dz.max(dz.abs());
          if first_bad.len() < 6 {
            first_bad.push(format!(
              "({x},{y}) lateral {:.3} km Δz {:+.3} km bgra {:?}",
              tx * d_km,
              dz,
              c
            ));
          }
        }
      }
    }
    println!(
      "{tag} ordering: {bad} misordered of {checked} classified (track in front: {bad_red} of \
       {checked_red} wrong, behind: {bad_green} of {checked_green} wrong), max |Δz| among the \
       misordered {:.1} m",
      bad_max_dz * 1e3
    );
    for line in &first_bad {
      println!("{tag}   misordered {line}");
    }

    // ── (2) lateral precision: the track's row off the disc ──
    let expected_row = H as f64 * 0.5 - 0.5;
    let (mut cols, mut cols_missing, mut row_sum, mut row_err_max) = (0u32, 0u32, 0.0f64, 0.0f64);
    let mut track_pixel_off_disc = None;
    for x in 0..W {
      let lat = (tan_x(x) * d_km).abs();
      if !(3.0..=11.0).contains(&lat) {
        continue;
      }
      cols += 1;
      let (mut ys, mut k) = (0.0f64, 0u32);
      for y in 0..H {
        if is_red(px(x, y)) {
          ys += y as f64;
          k += 1;
        }
      }
      if k == 0 {
        cols_missing += 1;
        continue;
      }
      let row = ys / k as f64;
      if track_pixel_off_disc.is_none() {
        track_pixel_off_disc = Some((x, row.round() as u32));
      }
      row_sum += row;
      row_err_max = row_err_max.max((row - expected_row).abs());
    }
    let found = cols - cols_missing;
    println!(
      "{tag} lateral: track found in {found} of {cols} off-disc columns, mean row {:.2} \
       (expected {expected_row:.1}), max row error {row_err_max:.2} px",
      row_sum / found.max(1) as f64
    );

    // ── (3) GlobalDepth readback ──
    let sentinel = |v: [f32; 2]| v[0] == -1.0 && v[1] == -1.0;
    let (nx, ny) = (W / 2, H / 2 + (r_px_expected * 0.5) as u32); // below the ribbon, nucleus only
    let zs = sphere_offset(d_km, r_km, tan_x(nx), tan_y(ny)).expect("nucleus pixel off the disc");
    // micro layer units: km of 1/scale_f32 AU
    let expected_local = (d_km + zs) / AU_TO_KM / scale_f32 as f64;
    let tol = 2.0 * ulp_f32(expected_local);
    let g_nuc = gd(nx, ny);
    println!(
      "{tag} readback nucleus ({nx},{ny}) bgra {:?}: {g_nuc:?}, expected ({LAYER}, \
       {expected_local:.3} ± {tol} km), error {:.3} km",
      px(nx, ny),
      g_nuc[1] as f64 - expected_local
    );
    let track_px = track_pixel_on_disc.or(track_pixel_off_disc);
    let g_track = track_px.map(|(x, y)| gd(x, y));
    println!(
      "{tag} readback track pixel {track_px:?} (on disc: {}): {g_track:?}",
      track_pixel_on_disc.is_some()
    );
    let g_sky = gd(4, 4);
    println!("{tag} readback sky (4,4): {g_sky:?}");
    // an on-disc pixel where the track is behind: the nucleus wins and must read its own
    // (layer, distance), no macro tag leaking through the precise branch
    let behind_px = (W / 2 + 1..W).find(|&x| {
      let (tx, ty) = (tan_x(x), tan_y(H / 2));
      if (tx * d_km).abs() > 0.9 * r_km {
        return false;
      }
      let Some(zs) = sphere_offset(d_km, r_km, tx, ty) else { return false };
      zs - track_offset(d_km, tx).1 < -0.1 && is_green(px(x, H / 2))
    });
    let behind = behind_px.map(|x| {
      let zs = sphere_offset(d_km, r_km, tan_x(x), tan_y(H / 2)).unwrap();
      (x, gd(x, H / 2), (d_km + zs) / AU_TO_KM / scale_f32 as f64)
    });
    println!("{tag} readback track-behind pixel (x, gd, expected): {behind:?}");

    assert!(
      checked_red > 20 && checked_green > 20,
      "{tag} classification degenerate: {checked_red} front / {checked_green} behind pixels"
    );
    assert_eq!(
      bad, 0,
      "{tag} {bad} misordered pixels of {checked} (max |Δz| {:.1} m)",
      bad_max_dz * 1e3
    );
    assert!(
      cols_missing == 0 && row_err_max <= 1.0,
      "{tag} track row: {cols_missing} of {cols} columns without the track, max row error \
       {row_err_max:.2} px (> 1 px)"
    );
    assert!(
      (g_nuc[0] - LAYER as f32).abs() < 0.5 && (g_nuc[1] as f64 - expected_local).abs() <= tol,
      "{tag} nucleus GlobalDepth {g_nuc:?}, expected ({LAYER}, {expected_local:.3} ± {tol})"
    );
    let g_track = g_track.expect("no track pixel found for the readback check");
    assert!(sentinel(g_track), "{tag} track GlobalDepth {g_track:?}, expected (-1, -1)");
    assert!(sentinel(g_sky), "{tag} sky GlobalDepth {g_sky:?}, expected (-1, -1)");
    let (bx, g_b, e_b) = behind.expect("no green on-disc pixel behind the track");
    assert!(
      (g_b[0] - LAYER as f32).abs() < 0.5 && (g_b[1] as f64 - e_b).abs() <= 2.0 * ulp_f32(e_b),
      "{tag} track-behind pixel ({bx}) GlobalDepth {g_b:?}, expected ({LAYER}, {e_b:.3})"
    );
  }
}

#[test]
fn test_composite_precision_1au() {
  run_composite_precision(1.0);
}

#[test]
fn test_composite_precision_5au() {
  run_composite_precision(5.0);
}
