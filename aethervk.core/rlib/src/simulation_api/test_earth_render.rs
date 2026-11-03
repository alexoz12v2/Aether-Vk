//! GPU render tests of the Earth visual (`earth_visual`), offscreen with swapchain readback (the
//! `test_multi_micro_render` recipe). Skipped when no Vulkan device is available.
//!
//! The scene is the app's (`create_empty_scene`, no almanac: transforms are posed by hand). The Earth
//! sits at (−1, 0, 0) AU, so the Sun (at the origin) lights it from world +X. Cameras sit on the
//! macro layer `VIEW_DISTANCE_R` Earth radii from its centre.
//!
//! Frames are written to `$TMPDIR/avk_earth_render/*.png` for inspection.
//!
//! Guarantees (plan A-3, A-8, A-10 … A-14):
//! - only cameras with `RENDER_LAYER_UPZENITH_ONLY` in their mask draw the globe and its label,
//!   two cameras with different masks in the same scene and frame disagree as required;
//! - the globe is where and as large as the projection says, and the texture faces the right way
//!   (Sahara sandy, Pacific blue, Congo green, night side dark);
//! - the polar axis is dashed with the configured dash count, ends at ±1.5 R, and carries a
//!   label beyond each end, distinct glyphs at the two ends, the +Z one the same in an upside-down
//!   view;
//! - a comet gizmo behind the Earth does not show through it;
//! - a comet-gizmo axis pointing away from the viewer is dimmed and dashed, and the nucleus's lit
//!   side (nucleus-from-Earth shading: dark Lambert, black night side) is where the Sun arrow
//!   points;
//! - an Earth-observer camera (inside the globe's sphere, default mask) renders the same frame with
//!   and without the globe in the scene.

use super::{components_api::CameraParams, *};
use crate::{
  scene::{HiddenComponent, RENDER_LAYER_DEFAULT, RENDER_LAYER_UPZENITH_ONLY},
  simulation::constants::PlanetRadii,
  simulation_api::earth_visual::{AXIS_DASH_COUNT, AXIS_LENGTH_FACTOR},
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix3, mat3::Mat3f32},
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat, vec4f64::Quat64},
};
use alloc::{format, vec::Vec};
use core::sync::atomic::{AtomicU64, Ordering};
extern crate std;
use std::println;

const W: u32 = 512;
const H: u32 = 512;
const FOV_DEG: f64 = 40.0;
const AU_KM: f64 = 149_597_870.7;
const EARTH_AU: [f64; 3] = [-1.0, 0.0, 0.0];
const VIEW_DISTANCE_R: f64 = 5.0;

static PE_IDS: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(u64::MAX)];
static GENS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

extern "C" fn on_render(_scene: u64, pe_id: u64, generation: u64) {
  for slot in 0..2 {
    if PE_IDS[slot].load(Ordering::Acquire) == pe_id {
      GENS[slot].store(generation, Ordering::Release);
    }
  }
}

/// Engine log lines on stdout (shown with `--nocapture`).
extern "C" fn on_log(msg: *const core::ffi::c_char) {
  if !msg.is_null() {
    let s = unsafe { core::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    println!("[engine] {}", s.trim_end());
  }
}

fn on_vk_error(msg: &str) {
  println!("Vulkan validation error in test: {msg}");
  panic!("Vulkan error: {msg}");
}

fn r_km() -> f64 {
  PlanetRadii::EARTH as f64
}

/// The app's scene with two cameras, each on its own presentation engine.
struct Rig {
  ctx: *mut SimulationContext,
  scene_id: u64,
  earth: structs::SubtreeEntities,
  comet: structs::SubtreeEntities,
  earth_indicator: crate::scene::EntityId,
  cams: [u64; 2],
}

impl Drop for Rig {
  fn drop(&mut self) {
    unsafe {
      let _ = (*self.ctx).threads.logic_thread.tx().try_send(structs::LogicCommand::PauseScene {
        scene_id: self.scene_id,
      });
      std::thread::sleep(std::time::Duration::from_millis(100));
      SimulationContext::set_render_callback(None);
      for p in &PE_IDS {
        p.store(u64::MAX, Ordering::Release);
      }
      let _ = alloc::boxed::Box::from_raw(self.ctx);
    }
  }
}

impl Rig {
  fn new() -> Option<Self> {
    let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
    SimulationContext::set_asset_path(&asset_dir);
    SimulationContext::set_logger_callback(Some(on_log));
    let ctx = match SimulationContext::startup(Some(on_vk_error)) {
      Ok(b) => alloc::boxed::Box::into_raw(b),
      Err(_) => {
        println!("Skipping test: Vulkan backend could not be initialized");
        return None;
      }
    };
    let c = unsafe { &mut *ctx };
    let scene_id = c
      .create_empty_scene(
        true,
        hifitime::Epoch::from_gregorian_at_midnight(2026, 3, 20, hifitime::TimeScale::UTC),
        hifitime::Epoch::from_gregorian_at_midnight(2026, 3, 21, hifitime::TimeScale::UTC),
      )
      .unwrap();

    let (earth, comet, earth_indicator) = {
      let s = c.get_scene(scene_id).unwrap();
      let g = s.read();
      let earth = g.earth.unwrap();
      let indicator = g
        .scene
        .query1_res::<crate::scene::ReferentialIndicatorComponent, _, _>(|id, r| {
          (r.target_entity == earth.body).then_some(id)
        })
        .first()
        .map(|(id, _)| *id)
        .expect("earth_indicator");
      // the Sun's label would be drawn over the Earth when the camera looks sunwards
      let sun_labels = g
        .scene
        .query1_res::<crate::scene::indicator::IndicatorComponent, _, _>(|id, _| Some(id));
      for (id, _) in sun_labels {
        let _ = g.scene.add_component(id, HiddenComponent {});
      }


      (earth, g.comet.unwrap(), indicator)
    };

    let mut cams = [0u64; 2];
    for (slot, cam) in cams.iter_mut().enumerate() {
      let id = c.spawn_entity(scene_id, &format!("EarthTestCam{slot}")).unwrap();
      c.add_highres_transform_component(
        scene_id,
        id,
        Vec3f64::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
      c.add_camera_component(
        scene_id,
        id,
        CameraParams::new_perspective(FOV_DEG.to_radians() as f32, 1.0, 1e-6, 10.0),
      )
      .unwrap();
      {
        // under the root like the app's cameras (camera-relative micro transforms walk the tree)
        let sc = c.get_scene(scene_id).unwrap();
        let g = sc.read();
        let e: crate::scene::EntityId = slotmap::KeyData::from_ffi(id).into();
        g.scene.set_parent(e, Some(g.root_entity));
      }
      let pe = c.create_presentation_engine(scene_id, W, H).unwrap();
      PE_IDS[slot].store(pe.0, Ordering::Release);
      c.set_camera_for_presentation_engine(scene_id, pe, id).unwrap();
      *cam = id;
    }
    SimulationContext::set_render_callback(Some(on_render));
    let rig = Rig {
      ctx,
      scene_id,
      earth,
      comet,
      earth_indicator,
      cams,
    };
    rig.place_earth(Quat::identity());
    let _ = c.threads.logic_thread.tx().try_send(structs::LogicCommand::PlayScene {
      scene_id,
      speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
    });
    Some(rig)
  }

  fn ctx(&self) -> &SimulationContext {
    unsafe { &*self.ctx }
  }

  fn with_scene<R>(&self, f: impl FnOnce(&crate::scene::Scene) -> R) -> R {
    let s = self.ctx().get_scene(self.scene_id).unwrap();
    let g = s.read();
    f(&g.scene)
  }

  /// Earth subtree at `EARTH_AU`, body (km residual 0) rotated by `q` (body → world).
  fn place_earth(&self, q: Quat) {
    self.with_scene(|s| {
      let _ = s.with_component_mut(self.earth.subtree, |t: &mut crate::scene::TransformComponent| {
        t.position = Vec3f32::from_components(EARTH_AU[0] as f32, EARTH_AU[1] as f32, EARTH_AU[2] as f32);
      });
      let _ = s.with_component_mut(self.earth.body, |t: &mut crate::scene::TransformComponent| {
        t.position = Vec3f32::from_components(0.0, 0.0, 0.0);
        t.rotation = q;
      });
    });
  }

  /// Camera `slot` at `dir · VIEW_DISTANCE_R · R` from the Earth's centre looking at it, screen up
  /// = `up` (orthogonalised).
  fn view_earth_from(&self, slot: usize, dir: [f64; 3], up: [f64; 3]) {
    self.view_from(slot, EARTH_AU, dir, up, VIEW_DISTANCE_R * r_km() / AU_KM);
  }

  /// Camera `slot` at `center + dir · d` (AU) looking at `center`, screen up = `up`
  /// (orthogonalised).
  fn view_from(&self, slot: usize, center: [f64; 3], dir: [f64; 3], up: [f64; 3], d: f64) {
    let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let dir = [dir[0] / n, dir[1] / n, dir[2] / n];
    let pos = Vec3f64::from_components(
      center[0] + dir[0] * d,
      center[1] + dir[1] * d,
      center[2] + dir[2] * d,
    );
    // camera axes: +Y backward (= dir), +Z up, +X = Y × Z
    let norm = |v: [f64; 3]| {
      let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
      [v[0] / l, v[1] / l, v[2] / l]
    };
    let y = norm(dir);
    let k = up[0] * y[0] + up[1] * y[1] + up[2] * y[2];
    let z = norm([up[0] - k * y[0], up[1] - k * y[1], up[2] - k * y[2]]);
    let x = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]];
    let v3 = |v: [f64; 3]| Vec3f32::from_components(v[0] as f32, v[1] as f32, v[2] as f32);
    let q = Quat::from_rotation_matrix(&Mat3f32::from_columns(v3(x), v3(y), v3(z))).normalize();
    let cam: crate::scene::EntityId = slotmap::KeyData::from_ffi(self.cams[slot]).into();
    self.with_scene(|s| {
      let _ = s.with_component_mut(cam, |t: &mut crate::scene::HighResTransformComponent| {
        t.position = pos;
        t.rotation = Quat64::from_quat(q);
      });
    });
  }

  fn set_mask(&self, slot: usize, mask: u32) {
    assert!(self.ctx().set_camera_render_mask(self.scene_id, self.cams[slot], mask));
  }

  fn set_hidden(&self, e: crate::scene::EntityId, hidden: bool) {
    self.with_scene(|s| {
      if hidden {
        let _ = s.add_component(e, HiddenComponent {});
      } else {
        let _ = s.remove_component::<HiddenComponent>(e);
      }
    });
  }

  /// RGBA8 of camera `slot` after `frames` new render generations (exposure settling).
  fn grab(&self, slot: usize, frames: u32, name: &str) -> Vec<u8> {
    let poll = std::time::Duration::from_millis(5);
    let mut last = GENS[slot].load(Ordering::Acquire);
    let mut seen = 0;
    for _ in 0..4000 {
      let g = GENS[slot].load(Ordering::Acquire);
      if g != 0 && g != u64::MAX && g != last {
        seen += 1;
        last = g;
        if seen >= frames {
          break;
        }
      }
      std::thread::sleep(poll);
    }
    assert!(seen >= frames, "{name}: no fresh frames from camera {slot}");
    for _ in 0..4000 {
      if !matches!(self.ctx().get_task_status(last), structs::TaskStatusCode::Pending) {
        break;
      }
      std::thread::sleep(poll);
    }
    let mut buf = alloc::vec![0u8; (W * H * 4) as usize];
    assert!(
      unsafe { self.ctx().download_image(last, buf.as_mut_ptr(), buf.len()) },
      "{name}: download failed"
    );
    for px in buf.chunks_exact_mut(4) {
      px.swap(0, 2); // BGRA → RGBA
    }
    save_png(&buf, name);
    buf
  }
}

fn save_png(rgba: &[u8], name: &str) {
  let dir = std::env::temp_dir().join("avk_earth_render");
  let _ = std::fs::create_dir_all(&dir);
  if let Some(img) = image::RgbaImage::from_raw(W, H, rgba.to_vec()) {
    let _ = img.save(dir.join(format!("{name}.png")));
  }
}

fn px(img: &[u8], x: i32, y: i32) -> [f64; 3] {
  let i = ((y.clamp(0, H as i32 - 1) as u32 * W + x.clamp(0, W as i32 - 1) as u32) * 4) as usize;
  [img[i] as f64, img[i + 1] as f64, img[i + 2] as f64]
}

fn lum(c: [f64; 3]) -> f64 {
  0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn mean_patch(img: &[u8], cx: i32, cy: i32, k: i32) -> [f64; 3] {
  let mut acc = [0.0; 3];
  let mut n = 0.0;
  for y in cy - k..=cy + k {
    for x in cx - k..=cx + k {
      let p = px(img, x, y);
      for c in 0..3 {
        acc[c] += p[c];
      }
      n += 1.0;
    }
  }
  acc.map(|a| a / n)
}

/// Pixel radius of the Earth's disk seen from `VIEW_DISTANCE_R` radii with the test fov.
fn expected_disk_radius_px() -> f64 {
  let f = (H as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
  let alpha = (1.0 / VIEW_DISTANCE_R).asin();
  f * alpha.tan()
}

/// Pixel distance from the centre of a point `h` km from the Earth's centre along the screen's
/// vertical, perpendicular to the view direction.
fn expected_offset_px(h_km: f64) -> f64 {
  let f = (H as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
  f * h_km / (VIEW_DISTANCE_R * r_km())
}

/// Centroid and equivalent radius of the pixels brighter than `threshold`.
fn bright_blob(img: &[u8], threshold: f64) -> Option<([f64; 2], f64)> {
  let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
  for y in 0..H as i32 {
    for x in 0..W as i32 {
      if lum(px(img, x, y)) > threshold {
        sx += x as f64;
        sy += y as f64;
        n += 1.0;
      }
    }
  }
  (n > 0.0).then(|| ([sx / n, sy / n], (n / core::f64::consts::PI).sqrt()))
}

/// Centre and radius of the Earth's disk from the pixels where `with` (globe drawn) differs from
/// `without` (same pose, globe not drawn): the horizontal chord through the image centre row and
/// the vertical chord 20 px beside the centre column (the polar axis runs down the centre).
fn disk_from_diff(with: &[u8], without: &[u8]) -> Option<([f64; 2], f64)> {
  let differs = |x: i32, y: i32| {
    let (a, b) = (px(with, x, y), px(without, x, y));
    (0..3).any(|i| (a[i] - b[i]).abs() > 1.0)
  };
  let (cx, cy) = (W as i32 / 2, H as i32 / 2);
  let row: Vec<i32> = (0..W as i32).filter(|&x| differs(x, cy)).collect();
  let off = 20;
  let col: Vec<i32> = (0..H as i32).filter(|&y| differs(cx + off, y)).collect();
  let (l, r) = (*row.first()?, *row.last()?);
  let (t, b) = (*col.first()?, *col.last()?);
  let centre = [(l + r) as f64 / 2.0, (t + b) as f64 / 2.0];
  let rx = (r - l + 1) as f64 / 2.0;
  let half = (b - t + 1) as f64 / 2.0;
  let ry = (half * half + (off as f64 - (centre[0] - cx as f64)).powi(2)).sqrt();
  Some((centre, 0.5 * (rx + ry)))
}

/// Body rotation putting geographic (lat, lon) on world +X (towards the Sun and the day camera):
/// `Ry(lat) · Rz(−lon)`.
fn facing_sun(lat_deg: f64, lon_deg: f64) -> Quat {
  let ry = Quat::from_axis_angle(Vec3f32::from_components(0.0, 1.0, 0.0), lat_deg.to_radians() as f32);
  let rz =
    Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), -lon_deg.to_radians() as f32);
  (ry * rz).normalize()
}

const SUNWARD: [f64; 3] = [1.0, 0.0, 0.0];
const ANTISUN: [f64; 3] = [-1.0, 0.0, 0.0];
const UP: [f64; 3] = [0.0, 0.0, 1.0];
const DOWN: [f64; 3] = [0.0, 0.0, -1.0];
const SETTLE: u32 = 8;

// ── A-8, A-10: per-camera layers ──────────────────────────────────────────────────────────────

#[test]
fn earth_is_drawn_only_by_upzenith_cameras() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };

  // the FFI entry refuses non-cameras and unknown scenes
  let body_ffi = slotmap::Key::data(&rig.earth.body).as_ffi();
  assert!(!rig.ctx().set_camera_render_mask(rig.scene_id, body_ffi, RENDER_LAYER_DEFAULT));
  assert!(!rig.ctx().set_camera_render_mask(rig.scene_id + 999, rig.cams[0], RENDER_LAYER_DEFAULT));

  rig.place_earth(facing_sun(0.0, 20.0));
  rig.set_hidden(rig.earth_indicator, true);
  for slot in 0..2 {
    rig.view_earth_from(slot, SUNWARD, UP);
  }
  rig.set_mask(0, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  rig.set_mask(1, RENDER_LAYER_DEFAULT);
  let up = rig.grab(0, SETTLE, "mask_upzenith");
  let other = rig.grab(1, 1, "mask_default");

  let (c, r) = disk_from_diff(&up, &other).expect("an UpZenith camera draws the Earth");
  let want_r = expected_disk_radius_px();
  println!("disk centre {c:?} radius {r:.1} px (expected {want_r:.1})");
  assert!((c[0] - W as f64 / 2.0).abs() < 3.0 && (c[1] - H as f64 / 2.0).abs() < 3.0, "{c:?}");
  assert!((r - want_r).abs() / want_r < 0.05, "radius {r} vs {want_r}");

  // the UpZenith camera draws over (nearly) the whole disk …
  let differing = |p: &[u8], q: &[u8], inside_only: bool| {
    let mut n = 0usize;
    for y in 0..H as i32 {
      for x in 0..W as i32 {
        let d = ((x as f64 - c[0]).powi(2) + (y as f64 - c[1]).powi(2)).sqrt();
        if inside_only && d >= r * 0.95 {
          continue;
        }
        let (a, b) = (px(p, x, y), px(q, x, y));
        if (0..3).any(|i| (a[i] - b[i]).abs() > 2.0) {
          n += 1;
        }
      }
    }
    n
  };
  let disk_area = core::f64::consts::PI * (r * 0.95).powi(2);
  let covered = differing(&up, &other, true) as f64 / disk_area;
  println!("disk pixels drawn by the UpZenith camera only: {:.1} %", covered * 100.0);
  assert!(covered > 0.9, "the UpZenith camera draws the globe");
  // … and the default camera's frame is the same as with the globe hidden for everyone
  let visual = rig.earth.visual.unwrap();
  rig.set_hidden(visual, true);
  let other_hidden = rig.grab(1, SETTLE, "mask_default_globe_hidden");
  rig.set_hidden(visual, false);
  let changed = differing(&other, &other_hidden, false);
  println!("default camera pixels changed by hiding the globe: {changed}");
  assert!(changed < 5, "a default camera must not draw the Earth");

  // the "Earth" label: drawn by the UpZenith camera only
  rig.set_hidden(rig.earth_indicator, false);
  let up_label = rig.grab(0, SETTLE, "mask_upzenith_label");
  let other_label = rig.grab(1, 1, "mask_default_label");
  let changed = |a: &[u8], b: &[u8]| {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(p, q)| {
      (0..3).any(|i| (p[i] as i32 - q[i] as i32).abs() > 24)
    }).count()
  };
  let up_diff = changed(&up, &up_label);
  let other_diff = changed(&other, &other_label);
  println!("label pixels: upzenith {up_diff}, default {other_diff}");
  assert!(up_diff > 50, "the UpZenith camera shows the Earth label");
  assert!(other_diff < 5, "the default camera shows no Earth label");

  // switching the masks swaps who sees the Earth
  rig.set_hidden(rig.earth_indicator, true);
  rig.set_mask(0, RENDER_LAYER_DEFAULT);
  rig.set_mask(1, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  let a = rig.grab(0, SETTLE, "mask_swapped_0");
  let b = rig.grab(1, 1, "mask_swapped_1");
  // camera 1 now shows what camera 0 showed, and camera 0 what camera 1 did
  let (cb, rb) = disk_from_diff(&b, &a).expect("camera 1 now draws the Earth");
  assert!((rb - want_r).abs() / want_r < 0.05 && (cb[0] - c[0]).abs() < 2.0, "{cb:?} {rb}");
  let same = |p: &[u8], q: &[u8]| {
    p.chunks_exact(4).zip(q.chunks_exact(4)).filter(|(x, y)| {
      (0..3).any(|i| (x[i] as i32 - y[i] as i32).abs() > 2)
    }).count()
  };
  assert!(same(&a, &other) < 50, "camera 0 shows no Earth after the swap");
  assert!(same(&b, &up) < 50, "camera 1 shows the Earth as camera 0 did");
}

// ── A-11: texture orientation on screen ───────────────────────────────────────────────────────

#[test]
fn earth_texture_faces_the_right_way() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  rig.set_hidden(rig.earth_indicator, true);
  rig.set_mask(0, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  rig.view_earth_from(0, SUNWARD, UP);
  // sample beside the polar axis, which crosses the disk centre
  let (cx, cy) = (W as i32 / 2 + 10, H as i32 / 2);

  rig.place_earth(facing_sun(23.0, 13.0));
  let sahara = mean_patch(&rig.grab(0, SETTLE, "texture_sahara"), cx, cy, 2);
  rig.place_earth(facing_sun(0.0, -160.0));
  let pacific = mean_patch(&rig.grab(0, SETTLE, "texture_pacific"), cx, cy, 2);
  rig.place_earth(facing_sun(-2.0, 23.0));
  let congo_img = rig.grab(0, SETTLE, "texture_congo");
  let congo = mean_patch(&congo_img, cx, cy, 2);
  println!("Sahara {sahara:?} Pacific {pacific:?} Congo {congo:?}");
  assert!(sahara[0] > sahara[2] * 1.2, "Sahara must be sandy: {sahara:?}");
  assert!(pacific[2] > pacific[0] * 1.2, "the Pacific must be blue: {pacific:?}");
  // a mirrored map would show the Atlantic (blue) here
  assert!(congo[1] > congo[2], "the Congo basin must be green: {congo:?}");

  // night side: the camera on the far side from the Sun
  rig.view_earth_from(0, ANTISUN, UP);
  let night = rig.grab(0, SETTLE, "texture_night");
  let day_lum = bright_blob(&congo_img, 20.0).map(|(c, r)| {
    lum(mean_patch(&congo_img, c[0] as i32 + (r * 0.3) as i32, c[1] as i32, 4))
  });
  let night_lum = lum(mean_patch(&night, cx, cy, 4));
  println!("day {day_lum:?} night {night_lum:.1}");
  assert!(night_lum < 0.1 * day_lum.unwrap(), "the night side must be dark");
}

// ── A-12, A-13: dashed axis and labels ────────────────────────────────────────────────────────

/// Rows (top to bottom) where the column band around `x` has a pixel brighter than `threshold`.
fn lit_rows(img: &[u8], x: i32, threshold: f64) -> Vec<bool> {
  (0..H as i32)
    .map(|y| (x - 2..=x + 2).any(|xx| lum(px(img, xx, y)) > threshold))
    .collect()
}

/// Binary mask of the pixels brighter than `threshold` in the box, cropped to their bounding box.
/// `(width, height, mask, left, top)`.
fn glyph(
  img: &[u8],
  x0: i32,
  y0: i32,
  x1: i32,
  y1: i32,
  threshold: f64,
) -> Option<(usize, usize, Vec<bool>, i32, i32)> {
  let (mut bx0, mut by0, mut bx1, mut by1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
  for y in y0..y1 {
    for x in x0..x1 {
      if lum(px(img, x, y)) > threshold {
        bx0 = bx0.min(x);
        by0 = by0.min(y);
        bx1 = bx1.max(x);
        by1 = by1.max(y);
      }
    }
  }
  if bx0 > bx1 {
    return None;
  }
  let (w, h) = ((bx1 - bx0 + 1) as usize, (by1 - by0 + 1) as usize);
  let mut m = alloc::vec![false; w * h];
  for y in 0..h {
    for x in 0..w {
      m[y * w + x] = lum(px(img, bx0 + x as i32, by0 + y as i32)) > threshold;
    }
  }
  Some((w, h, m, bx0, by0))
}

/// Intersection over union of two glyph masks (sizes must agree within a pixel).
fn glyph_iou(a: &(usize, usize, Vec<bool>, i32, i32), b: &(usize, usize, Vec<bool>, i32, i32)) -> f64 {
  let (w, h) = (a.0.min(b.0), a.1.min(b.1));
  let (mut i, mut u) = (0.0, 0.0);
  for y in 0..h {
    for x in 0..w {
      let (p, q) = (a.2[y * a.0 + x], b.2[y * b.0 + x]);
      if p && q {
        i += 1.0;
      }
      if p || q {
        u += 1.0;
      }
    }
  }
  if u == 0.0 { 0.0 } else { i / u }
}

#[test]
fn polar_axis_is_dashed_and_labelled() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  rig.set_hidden(rig.earth_indicator, true);
  // body Z on world Z: the axis is vertical on screen; the night side keeps the globe dark
  rig.place_earth(Quat::identity());
  rig.set_mask(0, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  rig.set_mask(1, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  rig.view_earth_from(0, ANTISUN, UP);
  rig.view_earth_from(1, ANTISUN, DOWN); // upside down
  let img = rig.grab(0, SETTLE, "axis_upright");
  let flipped = rig.grab(1, 1, "axis_flipped");

  let (cx, cy) = (W as i32 / 2, H as i32 / 2);
  let h_px = expected_offset_px(r_km() * AXIS_LENGTH_FACTOR as f64);
  let (top, bottom) = (cy as f64 - h_px, cy as f64 + h_px);
  let rows = lit_rows(&img, cx, 60.0);
  // the axis: first and last lit rows of the band between the ends
  let in_axis: Vec<usize> = (0..H as usize)
    .filter(|&y| rows[y] && (y as f64) > top - 3.0 && (y as f64) < bottom + 3.0)
    .collect();
  let (first, last) = (*in_axis.first().expect("axis drawn"), *in_axis.last().unwrap());
  println!("axis rows {first}..{last}, expected {top:.1}..{bottom:.1}");
  assert!((first as f64 - top).abs() < 3.0 && (last as f64 - bottom).abs() < 3.0);
  // dashes: lit runs between the ends
  let mut runs = 0;
  for y in first..=last {
    if rows[y] && (y == first || !rows[y - 1]) {
      runs += 1;
    }
  }
  // N + ½ periods (both ends on a dash): N + 1 dashes
  println!("dashes: {runs} (configured {AXIS_DASH_COUNT} + 1)");
  assert!((runs as f32 - (AXIS_DASH_COUNT + 1.0)).abs() <= 2.0, "{runs} dashes");

  // labels just beyond each end (the label gap is 4 px; the line may reach 2 px past its end)
  // windows the size of a 16 pt capital (≈ 10 × 12 px) plus slack, so that stars stay out
  let (bw, bh) = (9, 20);
  let (above, below) = ((top - 2.0) as i32, (bottom + 3.0) as i32);
  let north = glyph(&img, cx - bw, above - bh, cx + bw, above, 60.0)
    .expect("a label above the north end");
  let south = glyph(&img, cx - bw, below, cx + bw, below + bh, 60.0)
    .expect("a label below the south end");
  // upside down, the +Z end is at the bottom and its (screen-aligned) label is the same glyph
  let north_flipped = glyph(&flipped, cx - bw, below, cx + bw, below + bh, 60.0)
    .expect("the +Z label below the end when upside down");
  let south_flipped = glyph(&flipped, cx - bw, above - bh, cx + bw, above, 60.0)
    .expect("the −Z label above the end when upside down");
  // each label is just beyond its axis end: the 4 px gap, plus the room between a capital and
  // the ascent line (below the end) and rasterisation slack
  let gap_above = top - (north.4 + north.1 as i32 - 1) as f64;
  let gap_below = north_flipped.4 as f64 - bottom;
  println!("label gaps: above {gap_above:.1} px, below {gap_below:.1} px");
  assert!((2.0..=9.0).contains(&gap_above) && (2.0..=9.0).contains(&gap_below));
  // classify the upside-down labels against the upright ones: each end keeps its own glyph, and
  // the two glyphs differ
  let n_as_n = glyph_iou(&north_flipped, &north);
  let n_as_s = glyph_iou(&north_flipped, &south);
  let s_as_s = glyph_iou(&south_flipped, &south);
  let s_as_n = glyph_iou(&south_flipped, &north);
  println!(
    "glyph IoU: +Z label vs N {n_as_n:.2} / vs S {n_as_s:.2}; −Z label vs S {s_as_s:.2} / vs N {s_as_n:.2}"
  );
  assert!(n_as_n > n_as_s + 0.15, "the +Z end keeps its label when the view rolls");
  assert!(s_as_s > s_as_n + 0.1, "the −Z end keeps its label when the view rolls");
  assert!(glyph_iou(&north, &south) < 0.8, "the two ends carry different labels");
}

// ── A-14: comet gizmo behind the Earth ────────────────────────────────────────────────────────

#[test]
fn comet_gizmo_behind_the_earth_does_not_show_through() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  rig.set_hidden(rig.earth_indicator, true);
  rig.place_earth(facing_sun(0.0, 20.0));
  rig.set_mask(0, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
  rig.view_earth_from(0, SUNWARD, UP);
  // the comet 3 R behind the Earth, gizmo radius 2 R: its outline overhangs the disk
  let behind = 3.0 * r_km() / AU_KM;
  rig.with_scene(|s| {
    let _ = s.with_component_mut(rig.comet.subtree, |t: &mut crate::scene::TransformComponent| {
      t.position = Vec3f32::from_components((EARTH_AU[0] - behind) as f32, 0.0, 0.0);
    });
    let _ = s.with_component_mut(rig.comet.body, |g: &mut crate::scene::SphereGizmoComponent| {
      g.radius = (2.0 * r_km()) as f32;
      g.is_visible = false;
    });
  });
  let without = rig.grab(0, SETTLE, "gizmo_off");
  rig.with_scene(|s| {
    let _ = s.with_component_mut(rig.comet.body, |g: &mut crate::scene::SphereGizmoComponent| {
      g.is_visible = true;
    });
  });
  let with = rig.grab(0, SETTLE, "gizmo_on");
  let (c, r) = bright_blob(&without, 20.0).expect("disk");
  let (mut diff_in, mut diff_out) = (0, 0);
  for y in 0..H as i32 {
    for x in 0..W as i32 {
      let (dx, dy) = (x as f64 - c[0], y as f64 - c[1]);
      let d = (dx * dx + dy * dy).sqrt();
      let p = px(&without, x, y);
      let q = px(&with, x, y);
      let changed = (0..3).any(|i| (p[i] - q[i]).abs() > 24.0);
      if changed && d < r - 3.0 {
        diff_in += 1;
      } else if changed && d > r + 3.0 {
        diff_out += 1;
      }
    }
  }
  println!("gizmo pixels: inside the disk {diff_in}, outside {diff_out}");
  assert!(diff_out > 100, "the gizmo must be drawn around the Earth");
  assert_eq!(diff_in, 0, "the gizmo must not show through the Earth");
}

// ── the comet gizmo's axes and Sun arrow over the nucleus ─────────────────────────────────────

/// The comet 1.25 AU from the Sun on world +Y (the Sun lights it from −Y), the camera `slot`
/// `COMET_VIEW_KM` from the nucleus, the gizmo with only `parts`, the body unrotated. The subtree
/// position is f32 and the camera aims at the f64 value, so the distance must be exact in f32:
/// 1.2 rounds to 1.2000000477 AU, ~7 km off the aim, ~125 px at 40 km.
const COMET_AU: [f64; 3] = [0.0, 1.25, 0.0];
const COMET_VIEW_KM: f64 = 40.0;

fn pose_comet(rig: &Rig, slot: usize, parts: u32, dir: [f64; 3]) {
  rig.set_hidden(rig.earth_indicator, true);
  rig.set_mask(0, RENDER_LAYER_DEFAULT);
  let cam: crate::scene::EntityId = slotmap::KeyData::from_ffi(rig.cams[slot]).into();
  rig.with_scene(|s| {
    let _ = s.with_component_mut(rig.comet.subtree, |t: &mut crate::scene::TransformComponent| {
      t.position = Vec3f32::from_components(COMET_AU[0] as f32, COMET_AU[1] as f32, COMET_AU[2] as f32);
    });
    let _ = s.with_component_mut(rig.comet.body, |t: &mut crate::scene::TransformComponent| {
      t.position = Vec3f32::from_components(0.0, 0.0, 0.0);
      t.rotation = Quat::identity();
    });
    let parts_component = crate::scene::SphereGizmoPartsComponent { parts, velocity_dir: [0.0; 3] };
    if s
      .with_component_mut(rig.comet.body, |p: &mut crate::scene::SphereGizmoPartsComponent| *p = parts_component)
      .is_none()
    {
      let _ = s.add_component(rig.comet.body, parts_component);
    }
    // a near plane well inside the 40 km view distance
    let _ = s.with_component_mut(cam, |c: &mut crate::scene::CameraComponent| {
      if let crate::scene::CameraProjection::Perspective { near, .. } = &mut c.projection {
        *near = 1e-9;
      }
    });
  });
  rig.view_from(slot, COMET_AU, dir, UP, COMET_VIEW_KM / AU_KM);
}

fn set_gizmo_visible(rig: &Rig, visible: bool) {
  rig.with_scene(|s| {
    let _ = s.with_component_mut(rig.comet.body, |g: &mut crate::scene::SphereGizmoComponent| {
      g.is_visible = visible;
    });
  });
}

/// Pixels where `with` differs from `without` by more than 24 in a channel, with their colour in
/// `with`.
fn changed_pixels(with: &[u8], without: &[u8]) -> Vec<([f64; 2], [f64; 3])> {
  let mut out = Vec::new();
  for y in 0..H as i32 {
    for x in 0..W as i32 {
      let (p, q) = (px(without, x, y), px(with, x, y));
      if (0..3).any(|i| (p[i] - q[i]).abs() > 24.0) {
        out.push(([x as f64, y as f64], q));
      }
    }
  }
  out
}

/// The spin axis pointing away from the viewer (its far part behind the nucleus) is drawn dimmed
/// and dashed; the same axis tilted as much towards the viewer is bright and solid. Both project
/// to the same screen segment (mirror poses about the sky plane).
#[test]
fn gizmo_axis_pointing_away_is_dimmed_and_dashed() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  let mut stats = Vec::new();
  for (name, dir) in [("towards", [1.0, 0.0, 0.5]), ("away", [1.0, 0.0, -0.5])] {
    pose_comet(&rig, 0, crate::scene::SPHERE_GIZMO_PART_SPIN_AXIS, dir);
    set_gizmo_visible(&rig, false);
    let without = rig.grab(0, SETTLE, &format!("spin_axis_{name}_off"));
    set_gizmo_visible(&rig, true);
    let with = rig.grab(0, SETTLE, &format!("spin_axis_{name}_on"));
    let changed = changed_pixels(&with, &without);
    let max_blue = changed.iter().map(|(_, c)| c[2]).fold(0.0, f64::max);
    println!("spin axis {name}: {} pixels, max blue {max_blue}", changed.len());
    stats.push((changed.len(), max_blue));
  }
  let ((n_front, b_front), (n_away, b_away)) = (stats[0], stats[1]);
  assert!(n_front > 30 && b_front > 220.0, "the axis towards the viewer is drawn bright");
  assert!(n_away > 10, "the axis pointing away is still drawn");
  assert!(b_away < 160.0, "the axis pointing away is dimmed (max blue {b_away})");
  assert!(
    (n_away as f64) < 0.8 * n_front as f64,
    "the axis pointing away is dashed ({n_away} vs {n_front} pixels)"
  );
}

/// The nucleus in the nucleus-from-Earth shading (physical dark Lambert, exposure normalised to the
/// dark albedo): the lit side is where the Sun arrow points, the side opposite is black (no
/// ambient, no tint). Two skew poses, the arrow tilted towards the viewer.
#[test]
fn terminator_is_on_the_sun_arrow_side() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  let shading = crate::scene::CAMERA_SHADING_PHYSICAL | crate::scene::CAMERA_SHADING_LAMBERT_DARK;
  assert!(rig.ctx().set_camera_shading(rig.scene_id, rig.cams[0], shading));
  assert!(rig.ctx().set_camera_exposure(rig.scene_id, rig.cams[0], 1.0 / 0.06));
  let c0 = [W as f64 / 2.0, H as f64 / 2.0];
  // the nucleus's disk radius on screen (2 km at 40 km)
  let f = (H as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
  let disk_px = f * crate::simulation_api::comet_appearance::DEFAULT_NUCLEUS_RADIUS_KM as f64 / COMET_VIEW_KM;
  for (i, dir) in [[1.0, -0.3, 0.2], [0.7, -0.2, -0.6]].into_iter().enumerate() {
    pose_comet(&rig, 0, crate::scene::SPHERE_GIZMO_PART_SUN, dir);
    set_gizmo_visible(&rig, false);
    let without = rig.grab(0, SETTLE, &format!("terminator_{i}_off"));
    set_gizmo_visible(&rig, true);
    let with = rig.grab(0, SETTLE, &format!("terminator_{i}_on"));

    let lit: Vec<[f64; 2]> = (0..H as i32)
      .flat_map(|y| (0..W as i32).map(move |x| (x, y)))
      // on the nucleus's disk only (the sky behind may hold stars)
      .filter(|&(x, y)| ((x as f64 - c0[0]).powi(2) + (y as f64 - c0[1]).powi(2)).sqrt() <= disk_px + 2.0)
      .filter(|&(x, y)| lum(px(&without, x, y)) > 8.0)
      .map(|(x, y)| [x as f64, y as f64])
      .collect();
    assert!(lit.len() > 50, "pose {i}: the nucleus is lit ({} pixels)", lit.len());
    let centroid = |v: &[[f64; 2]]| {
      let n = v.len() as f64;
      [v.iter().map(|p| p[0]).sum::<f64>() / n, v.iter().map(|p| p[1]).sum::<f64>() / n]
    };
    let l = centroid(&lit);
    let arrow: Vec<[f64; 2]> = changed_pixels(&with, &without)
      .into_iter()
      .filter(|(_, c)| c[0] > 100.0 && c[1] > 100.0 && c[2] < 80.0)
      .map(|(p, _)| p)
      .collect();
    assert!(arrow.len() > 20, "pose {i}: the Sun arrow is drawn ({} pixels)", arrow.len());
    let a = centroid(&arrow);
    let (v1, v2) = ([l[0] - c0[0], l[1] - c0[1]], [a[0] - c0[0], a[1] - c0[1]]);
    let n1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let n2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
    assert!(n1 > 3.0 && n2 > 3.0, "pose {i}: offsets {v1:?} {v2:?}");
    let angle = ((v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)).clamp(-1.0, 1.0).acos().to_degrees();
    println!("pose {i}: lit centroid {l:?}, Sun arrow centroid {a:?}, angle {angle:.1}°");
    assert!(angle < 12.0, "pose {i}: the lit side is {angle:.1}° off the Sun arrow");
    // the anti-solar side of the disk is black: no ambient, no debug tint
    let night = [c0[0] - v1[0] / n1 * 0.6 * disk_px, c0[1] - v1[1] / n1 * 0.6 * disk_px];
    let p = mean_patch(&without, night[0].round() as i32, night[1].round() as i32, 1);
    assert!(p.iter().all(|&c| c < 3.0), "pose {i}: the night side is not black: {p:?}");
  }
}

// ── A-3: the Earth-observer view does not see the globe at all ────────────────────────────────

#[test]
fn earth_observer_frame_is_unchanged_by_the_globe() {
  let _lock = crate::simulation_api::lock_render_callback_tests();
  let Some(rig) = Rig::new() else { return };
  rig.place_earth(facing_sun(0.0, 0.0));
  rig.set_mask(0, RENDER_LAYER_DEFAULT);
  // on the night side 1 km under the globe's surface, looking out along −X... the camera looks at
  // the Earth's centre from 0.9998 R: the globe would fill the view if it were drawn
  let cam: crate::scene::EntityId = slotmap::KeyData::from_ffi(rig.cams[0]).into();
  let d = (r_km() - 1.0) / AU_KM;
  rig.with_scene(|s| {
    let _ = s.with_component_mut(cam, |t: &mut crate::scene::HighResTransformComponent| {
      t.position = Vec3f64::from_components(EARTH_AU[0] - d, 0.0, 0.0);
      // looking away from the Earth (−X): +Y backward = +X
      let q = Quat::from_rotation_matrix(&Mat3f32::from_columns(
        Vec3f32::from_components(0.0, -1.0, 0.0),
        Vec3f32::from_components(1.0, 0.0, 0.0),
        Vec3f32::from_components(0.0, 0.0, 1.0),
      ));
      t.rotation = Quat64::from_quat(q);
    });
  });
  let with_globe = rig.grab(0, SETTLE, "observer_with_globe");
  let visual = rig.earth.visual.unwrap();
  rig.with_scene(|s| {
    let _ = s.remove_component::<crate::scene::StaticMeshComponent>(visual);
    let _ = s.remove_component::<crate::simulation_api::earth_visual::PolarAxisGizmoComponent>(visual);
    let _ = s.remove_component::<crate::simulation_api::earth_visual::PoleLabelsComponent>(visual);
  });
  let without_globe = rig.grab(0, SETTLE, "observer_without_globe");
  let differing = with_globe
    .chunks_exact(4)
    .zip(without_globe.chunks_exact(4))
    .filter(|(a, b)| a != b)
    .count();
  println!("differing pixels: {differing}");
  assert_eq!(differing, 0, "the globe must not change an Earth-observer frame");
}
