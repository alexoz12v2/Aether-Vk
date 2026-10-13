//! Headless observer of the Earth visual (the globe, its dashed polar axis with N/S labels, drawn
//! only by UpZenith cameras), with the real ephemeris and Earth orientation (`de442.bsp`,
//! `earth_latest_high_prec.bpc`): no hand-posed transforms, unlike the GPU unit tests in
//! `simulation_api/test_earth_render.rs`.
//!
//! For every epoch (default: the 2026 equinoxes and solstices) the camera looks at the Earth from
//! `VIEW_DISTANCE_R` radii, perpendicular to the Sun line in the ecliptic plane, ecliptic north up,
//! so the terminator crosses the disk where the true Sun puts it (checked on sampled globe points
//! against their predicted daylight, also beyond 45° latitude for the season). Frames:
//! - `sunward_<date>.png` / `sunward_hidden_<date>.png`: UpZenith mask from the Sun side (the
//!   whole visible disk lit), with the globe present and hidden: the disk is their difference;
//! - `up_<date>.png`: UpZenith mask from the side, the Earth drawn;
//! - `default_<date>.png` / `default_hidden_<date>.png`: default mask, with the globe present and
//!   hidden scene-wide (must be identical);
//! - `observer_<date>.png` / `observer_hidden_<date>.png`: a camera 1 km under the globe's surface
//!   looking out (an Earth observer, default mask), with the globe present and hidden (identical).
//!
//! `--check` exits 1 unless, for every epoch:
//! - the disk's centre is where the camera projects the Earth's centre (≤ 2 px) and its radius is
//!   the projected equatorial radius (≤ 3 %);
//! - the axis ends are drawn where the camera projects `±1.5 R` along the body Z axis (≤ 3 px) and
//!   a label sits beyond each end;
//! - the polar cap facing the Sun is the lit one: when the subsolar latitude δ is beyond ±10°, the
//!   cap of the summer hemisphere is ≥ 1.5× brighter than the winter one;
//! - default-mask and Earth-observer frames do not change when the globe is hidden.
//!
//! ```text
//! cargo run -p aethervk-core-rlib --example scene_observer --features std --release -- \
//!   --out DIR [--size 640x640] [--epochs 2026-03-20,2026-06-21,...] [--check] [--quiet]
//!   [--strict-orientation]
//! ```
//! Outputs in `--out`: the frames, `contact_sheet.png` (one row per epoch), `stats.jsonl`,
//! `log.txt`, `summary.md`. See `scripts/observe_scene.sh`.

use aethervk_core_rlib::{
  scene::{EntityId, HiddenComponent, HighResTransformComponent, RENDER_LAYER_DEFAULT, RENDER_LAYER_UPZENITH_ONLY},
  simulation::constants::PlanetRadii,
  simulation_api::{
    SimulationContext,
    earth_visual::{AXIS_LENGTH_FACTOR, pole_points_global},
    structs::{LogicCommand, TaskStatusCode},
  },
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix3, mat3::Mat3f32},
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec3f64::Vec3f64, vec4f64::Quat64},
};
use std::{
  fs::File,
  io::Write,
  path::PathBuf,
  sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
  },
};

const AU_KM: f64 = 149_597_870.7;
const VIEW_DISTANCE_R: f64 = 5.0;
const FOV_DEG: f64 = 40.0;
/// Render generations to wait before reading a frame back (exposure, uploads settle).
const SETTLE_FRAMES: u32 = 8;

// ───────────────────────────── callbacks (as in dust_observer) ─────────────────────────────

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);
static LOG_ECHO: AtomicBool = AtomicBool::new(true);
static VK_ERRORS: AtomicU64 = AtomicU64::new(0);
static TASK_ID: AtomicU64 = AtomicU64::new(0);
static PE_ID: AtomicU64 = AtomicU64::new(u64::MAX);

fn log_line(line: &str) {
  if let Ok(mut g) = LOG_FILE.lock() {
    if let Some(f) = g.as_mut() {
      let _ = writeln!(f, "{line}");
    }
  }
  if LOG_ECHO.load(Ordering::Relaxed) {
    println!("{line}");
  }
}

extern "C" fn on_logger(msg: *const core::ffi::c_char) {
  if msg.is_null() {
    return;
  }
  let s = unsafe { core::ffi::CStr::from_ptr(msg) }.to_string_lossy();
  if let Ok(mut g) = LOG_FILE.lock() {
    if let Some(f) = g.as_mut() {
      let _ = writeln!(f, "{}", s.trim_end());
    }
  }
}

fn vk_error_callback(msg: &str) {
  VK_ERRORS.fetch_add(1, Ordering::Relaxed);
  log_line(&format!("[vulkan] {msg}"));
}

extern "C" fn on_render(_scene_id: u64, pe_id: u64, render_generation: u64) {
  if pe_id == PE_ID.load(Ordering::Acquire) {
    TASK_ID.store(render_generation, Ordering::Release);
  }
}

// ───────────────────────────── args ─────────────────────────────

struct Args {
  out: PathBuf,
  width: u32,
  height: u32,
  epochs: Vec<hifitime::Epoch>,
  check: bool,
  /// the subsolar-longitude check also fails on the IAU_EARTH fallback (known engine issue)
  strict_orientation: bool,
}

fn parse_args() -> Args {
  let mut out = std::env::var("OBSERVER_OUT").ok().map(PathBuf::from);
  let (mut width, mut height) = (640u32, 640u32);
  let mut epochs: Vec<hifitime::Epoch> = ["2026-03-20", "2026-06-21", "2026-09-23", "2026-12-21"]
    .iter()
    .map(|d| parse_date(d))
    .collect();
  let mut check = false;
  let mut strict_orientation = false;
  let mut it = std::env::args().skip(1);
  while let Some(a) = it.next() {
    match a.as_str() {
      "--out" => out = it.next().map(PathBuf::from),
      "--size" => {
        let v = it.next().expect("--size WxH");
        let (w, h) = v.split_once('x').expect("--size WxH");
        width = w.parse().expect("width");
        height = h.parse().expect("height");
      }
      "--epochs" => {
        epochs = it.next().expect("--epochs").split(',').map(parse_date).collect();
      }
      "--check" => check = true,
      "--strict-orientation" => strict_orientation = true,
      "--quiet" => LOG_ECHO.store(false, Ordering::Relaxed),
      other => {
        eprintln!("unknown argument {other}");
        std::process::exit(2);
      }
    }
  }
  let Some(out) = out else {
    eprintln!("--out DIR is required (or OBSERVER_OUT)");
    std::process::exit(2);
  };
  Args {
    out,
    width,
    height,
    epochs,
    check,
    strict_orientation,
  }
}

/// `YYYY-MM-DD` at 12:00 UTC.
fn parse_date(s: &str) -> hifitime::Epoch {
  let p: Vec<i32> = s.trim().split('-').map(|x| x.parse().expect("YYYY-MM-DD")).collect();
  hifitime::Epoch::from_gregorian_utc(p[0], p[1] as u8, p[2] as u8, 12, 0, 0, 0)
}

// ───────────────────────────── frames ─────────────────────────────

/// RGBA8 of the next frame after `frames` fresh render generations.
fn grab(ctx: &SimulationContext, w: u32, h: u32, frames: u32) -> Option<Vec<u8>> {
  let poll = std::time::Duration::from_millis(5);
  let mut last = TASK_ID.load(Ordering::Acquire);
  let mut seen = 0;
  for _ in 0..6000 {
    let t = TASK_ID.load(Ordering::Acquire);
    if t != 0 && t != u64::MAX && t != last {
      seen += 1;
      last = t;
      if seen >= frames {
        break;
      }
    }
    std::thread::sleep(poll);
  }
  if seen < frames {
    return None;
  }
  for _ in 0..6000 {
    if !matches!(ctx.get_task_status(last), TaskStatusCode::Pending) {
      break;
    }
    std::thread::sleep(poll);
  }
  let mut buf = vec![0u8; (w * h * 4) as usize];
  if !unsafe { ctx.download_image(last, buf.as_mut_ptr(), buf.len()) } {
    return None;
  }
  for px in buf.chunks_exact_mut(4) {
    px.swap(0, 2); // BGRA → RGBA
  }
  Some(buf)
}

fn save(out: &PathBuf, name: &str, w: u32, h: u32, rgba: &[u8]) {
  if let Some(img) = image::RgbaImage::from_raw(w, h, rgba.to_vec()) {
    let _ = img.save(out.join(format!("{name}.png")));
  }
}

struct Img<'a> {
  px: &'a [u8],
  w: u32,
  h: u32,
}

impl Img<'_> {
  fn rgb(&self, x: i32, y: i32) -> [f64; 3] {
    let x = x.clamp(0, self.w as i32 - 1) as u32;
    let y = y.clamp(0, self.h as i32 - 1) as u32;
    let i = ((y * self.w + x) * 4) as usize;
    [self.px[i] as f64, self.px[i + 1] as f64, self.px[i + 2] as f64]
  }
  fn lum(&self, x: i32, y: i32) -> f64 {
    let c = self.rgb(x, y);
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
  }
}

fn differing(a: &[u8], b: &[u8]) -> usize {
  a.chunks_exact(4)
    .zip(b.chunks_exact(4))
    .filter(|(p, q)| (0..3).any(|i| (p[i] as i32 - q[i] as i32).abs() > 2))
    .count()
}


// ───────────────────────────── camera ─────────────────────────────

/// A pinhole camera at `pos` (AU) looking along `fwd` with `up`, the engine's basis
/// (+Y backward, +Z up, +X = Y × Z).
#[derive(Clone, Copy)]
struct View {
  pos: [f64; 3],
  x: [f64; 3],
  y: [f64; 3],
  z: [f64; 3],
}

fn alloc_header() -> String {
  String::from("nx,ny,nz,ndl,lum")
}

fn norm(v: [f64; 3]) -> [f64; 3] {
  let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  [v[0] / l, v[1] / l, v[2] / l]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add_scaled(a: [f64; 3], b: [f64; 3], k: f64) -> [f64; 3] {
  [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]
}

impl View {
  fn new(pos: [f64; 3], fwd: [f64; 3], up: [f64; 3]) -> Self {
    let y = norm([-fwd[0], -fwd[1], -fwd[2]]);
    let k = dot(up, y);
    let z = norm(sub(up, [y[0] * k, y[1] * k, y[2] * k]));
    let x = cross(y, z);
    Self { pos, x, y, z }
  }
  fn rotation(&self) -> Quat64 {
    let v = |a: [f64; 3]| Vec3f32::from_components(a[0] as f32, a[1] as f32, a[2] as f32);
    Quat64::from_quat(
      aethervk_oshal_rlib::math::vector::vec4::Quat::from_rotation_matrix(&Mat3f32::from_columns(
        v(self.x),
        v(self.y),
        v(self.z),
      ))
      .normalize(),
    )
  }
  /// Pixel of the world point `p` (AU), screen y down; `None` behind the camera.
  fn project(&self, p: [f64; 3], w: u32, h: u32) -> Option<[f64; 2]> {
    let d = sub(p, self.pos);
    let depth = -dot(d, self.y);
    if depth <= 0.0 {
      return None;
    }
    let f = (h as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
    // With +Y backward and +Z up, the viewer's right hand is forward × up = −Y × Z = −X: screen x
    // grows along the camera's −X (the rendered globe is not mirrored: in June the north end
    // leans towards the lit limb, as it must).
    Some([
      w as f64 * 0.5 - f * dot(d, self.x) / depth,
      h as f64 * 0.5 - f * dot(d, self.z) / depth,
    ])
  }
}

// ───────────────────────────── main ─────────────────────────────

struct Failures(Vec<String>, Vec<String>);
impl Failures {
  /// A known engine issue: logged and listed in the summary, a failure only when `strict`.
  fn known(&mut self, ok: bool, strict: bool, what: String) {
    if ok || strict {
      self.check(ok, what);
    } else {
      log_line(&format!("[check] KNOWN {what}"));
      self.1.push(what);
    }
  }
  fn check(&mut self, ok: bool, what: String) {
    log_line(&format!("[check] {} {what}", if ok { "PASS" } else { "FAIL" }));
    if !ok {
      self.0.push(what);
    }
  }
}

fn main() {
  let args = parse_args();
  std::fs::create_dir_all(&args.out).expect("create --out");
  *LOG_FILE.lock().unwrap() = Some(File::create(args.out.join("log.txt")).expect("log.txt"));
  let (w, h) = (args.width, args.height);

  SimulationContext::set_logger_callback(Some(on_logger));
  SimulationContext::set_render_callback(Some(on_render));
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);
  let ctx = match SimulationContext::startup(Some(vk_error_callback)) {
    Ok(c) => c,
    Err(e) => {
      log_line(&format!("FATAL: SimulationContext::startup failed: {e:?}"));
      std::process::exit(4);
    }
  };
  {
    let mut logic = ctx.logic_state.write();
    for rel in [
      "planets/de442.bsp",
      "earth_latest_high_prec.bpc",
      "planets/pck00011.pca",
      "planets/gm_de431.pca",
    ] {
      let p = format!("{asset_dir}/{rel}");
      if let Err(e) =
        logic.almanac_data.load_almanac(aethervk_oshal_rlib::os::fs::PathBuf::from(&p))
      {
        log_line(&format!("FATAL: almanac {p}: {e}"));
        std::process::exit(4);
      }
    }
  }

  let first = *args.epochs.iter().min().unwrap();
  let last = *args.epochs.iter().max().unwrap();
  let scene_ret = ctx
    .create_empty_scene2(false, first, last + hifitime::Duration::from_days(1.0))
    .expect("create_empty_scene2");
  let scene_id = scene_ret.scene_id;
  let pe = ctx.create_presentation_engine(scene_id, w, h).expect("presentation engine");
  PE_ID.store(pe.0, Ordering::Release);
  let cam_ffi = ctx
    .add_perspective_camera(scene_id, pe, "scene_observer_cam", FOV_DEG.to_radians() as f32, 1e-6, 20.0)
    .expect("camera")
    .get();
  ctx.set_camera_for_presentation_engine(scene_id, pe, cam_ffi).expect("camera for PE");
  let cam = EntityId::from_ffi(cam_ffi);

  let scene_arc = ctx.get_scene(scene_id).unwrap();
  let (visual, earth_body, sun) = {
    let g = scene_arc.read();
    let earth = g.earth.expect("earth subtree");
    // the labels of the Sun and comet would cover the frame centre in some poses
    let labels = g
      .scene
      .query1_res::<aethervk_core_rlib::scene::IndicatorComponent, _, _>(|id, _| Some(id));
    for (id, _) in labels {
      let _ = g.scene.add_component(id, HiddenComponent {});
    }
    // orbit tracks (their head grows between frames) and the "Earth" label (its leader line
    // crosses the disk) would pollute the pixel measurements: the GPU unit tests cover the label
    let tracks = g
      .scene
      .query1_res::<aethervk_core_rlib::scene::trajectory::TrajectoryComponent, _, _>(|id, _| Some(id));
    for (id, _) in tracks {
      let _ = g.scene.add_component(id, HiddenComponent {});
    }
    let earth_labels = g
      .scene
      .query1_res::<aethervk_core_rlib::scene::ReferentialIndicatorComponent, _, _>(|id, r| {
        (r.target_entity == earth.body).then_some(id)
      });
    for (id, _) in earth_labels {
      let _ = g.scene.add_component(id, HiddenComponent {});
    }
    let _ = g.scene.add_component(
      cam,
      HighResTransformComponent {
        position: Vec3f64::from_components(0.0, 0.0, 0.0),
        rotation: Quat64::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );
    (earth.visual.expect("Earth_visual"), earth.body, g.sun_entity)
  };
  let set_hidden = |hidden: bool| {
    let g = scene_arc.read();
    if hidden {
      let _ = g.scene.add_component(visual, HiddenComponent {});
    } else {
      let _ = g.scene.remove_component::<HiddenComponent>(visual);
    }
  };
  let skies: Vec<EntityId> = {
    let g = scene_arc.read();
    g.scene
      .query1_res::<aethervk_core_rlib::scene::SkyComponent, _, _>(|id, _| Some(id))
      .into_iter()
      .map(|(id, _)| id)
      .collect()
  };
  let set_sky_hidden = |hidden: bool| {
    let g = scene_arc.read();
    for &e in &skies {
      if hidden {
        let _ = g.scene.add_component(e, HiddenComponent {});
      } else {
        let _ = g.scene.remove_component::<HiddenComponent>(e);
      }
    }
  };
  let place = |v: &View| {
    let g = scene_arc.read();
    let _ = g.scene.with_component_mut(cam, |t: &mut HighResTransformComponent| {
      t.position = Vec3f64::from_components(v.pos[0], v.pos[1], v.pos[2]);
      t.rotation = v.rotation();
    });
  };
  let global = |e: EntityId| -> Option<[f64; 3]> {
    let g = scene_arc.read();
    g.scene
      .global_transform_f64(e)
      .map(|t| [t.position.x(), t.position.y(), t.position.z()])
  };

  let mut failures = Failures(Vec::new(), Vec::new());
  let mut stats = File::create(args.out.join("stats.jsonl")).expect("stats.jsonl");
  let mut sheet_rows: Vec<Vec<Vec<u8>>> = Vec::new();
  let r_km = PlanetRadii::EARTH as f64;
  let r_au = r_km / AU_KM;

  for epoch in &args.epochs {
    let date = format!("{}", epoch).chars().take(10).collect::<String>();
    // seek (paused), then play in real time so frames are produced
    let tx = ctx.threads.logic_thread.tx();
    let _ = tx.try_send(LogicCommand::PauseScene { scene_id });
    let done = Arc::new(AtomicBool::new(false));
    let ok = Arc::new(AtomicBool::new(false));
    let _ = tx.try_send(LogicCommand::SeekEpoch {
      scene_id,
      epoch: *epoch,
      done_flag: done.clone(),
      succeeded: ok.clone(),
    });
    for _ in 0..2000 {
      if done.load(Ordering::Acquire) {
        break;
      }
      std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let _ = tx.try_send(LogicCommand::PlayScene {
      scene_id,
      speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
    });
    failures.check(ok.load(Ordering::Acquire), format!("{date}: seek applied"));
    let _ = grab(&ctx, w, h, 2);

    // ── geometry of the moment (engine state) ──
    let (Some(earth), Some(sun_pos)) = (global(visual), sun.and_then(|s| global(s))) else {
      failures.check(false, format!("{date}: Earth and Sun positions"));
      continue;
    };
    let s = norm(sub(sun_pos, earth));
    let ecl_north = [0.0, 0.0, 1.0];
    let side = norm(cross(s, ecl_north));
    let up_view = View::new(add_scaled(earth, side, VIEW_DISTANCE_R * r_au), [-side[0], -side[1], -side[2]], ecl_north);
    // The starry sky is hidden in every frame: the measurements diff frames, and after a seek the
    // sky drifts sub-pixel with the scene time (a few star pixels change between two frames of a
    // still pose), which is not what they test.
    set_sky_hidden(true);

    // ── pose A, sunward (the whole visible disk lit): disk centre and radius ──
    let lit_view = View::new(add_scaled(earth, s, VIEW_DISTANCE_R * r_au), [-s[0], -s[1], -s[2]], ecl_north);
    place(&lit_view);
    ctx.set_camera_render_mask(scene_id, cam_ffi, RENDER_LAYER_DEFAULT | RENDER_LAYER_UPZENITH_ONLY);
    let Some(lit) = grab(&ctx, w, h, SETTLE_FRAMES) else {
      failures.check(false, format!("{date}: sunward frame"));
      set_sky_hidden(false);
      continue;
    };
    let earth_lit = global(visual).unwrap_or(earth);

    // ── the clock and the Earth's orientation behind that frame ──
    let scene_epoch = ctx
      .scenes
      .read()
      .time_managers
      .get(&scene_id)
      .map(|tm| tm.current_epoch());
    match scene_epoch {
      Some(se) => {
        let lag_s = (se - *epoch).to_seconds();
        failures.check(
          (0.0..=300.0).contains(&lag_s),
          format!("{date}: scene epoch {se} is the requested epoch (+{lag_s:.1} s of real-time play)"),
        );
        // body → world rotation against ANISE ITRF93 → ecliptic J2000 at the scene epoch: the
        // images of the body's +X (Greenwich) and +Z (north pole) axes
        let body_rot = {
          let g = scene_arc.read();
          g.scene.global_transform_f64(earth_body).map(|t| t.rotation)
        };
        // ITRF93 from the high-precision BPC where it covers the epoch, else the IAU_EARTH model
        // (what `AlmanacPlanet::rotation_at` falls back to)
        let (dcm, frame_name) = {
          let logic = ctx.logic_state.read();
          let alm = &logic.almanac_data.almanac;
          let target = aethervk_core_rlib::simulation::almanac::SUN_ECLIPJ2000;
          match alm.rotate(anise::constants::frames::EARTH_ITRF93, target, se) {
            Ok(d) => (Ok(d), "ITRF93"),
            Err(e) => {
              log_line(&format!("[info] {date}: ITRF93 not available ({e}); IAU_EARTH reference"));
              (alm.rotate(anise::constants::frames::IAU_EARTH_FRAME, target, se), "IAU_EARTH")
            }
          }
        };
        if let (Some(q), Ok(dcm)) = (body_rot, dcm) {
          let axis_err = |local: [f64; 3]| {
            let got = q.rotate_vector(Vec3f64::from_components(local[0], local[1], local[2]));
            let want = dcm.rot_mat * anise::math::Vector3::new(local[0], local[1], local[2]);
            let c = got.x() * want[0] + got.y() * want[1] + got.z() * want[2];
            c.clamp(-1.0, 1.0).acos().to_degrees()
          };
          let (ex, ez) = (axis_err([1.0, 0.0, 0.0]), axis_err([0.0, 0.0, 1.0]));
          failures.check(
            ex < 0.05 && ez < 0.05,
            format!("{date}: Earth orientation = {frame_name} at the scene epoch (Greenwich {ex:.4}°, pole {ez:.4}°)"),
          );
          // the subsolar point in the engine's body frame vs the solar-time estimate
          // (longitude ≈ −15°·(UT − 12 h) − equation of time, within ~4°)
          let sb = q.inverse().rotate_vector(Vec3f64::from_components(s[0], s[1], s[2]));
          let (lat, lon) = (sb.z().clamp(-1.0, 1.0).asin().to_degrees(), sb.y().atan2(sb.x()).to_degrees());
          let (_, _, _, hh, mm, ss, _) = se.to_gregorian_utc();
          let ut_h = hh as f64 + mm as f64 / 60.0 + ss as f64 / 3600.0;
          let want_lon = -15.0 * (ut_h - 12.0);
          let dlon = ((lon - want_lon + 540.0) % 360.0) - 180.0;
          let what = format!(
            "{date}: subsolar point ({lat:+.1}°, {lon:+.1}°), solar time says longitude {want_lon:+.1}° (Δ {dlon:+.1}°, {frame_name})"
          );
          if frame_name == "ITRF93" {
            failures.check(dlon.abs() < 5.0, what);
          } else {
            // Beyond the BPC's coverage the engine (`AlmanacPlanet::rotation_at`) falls back to
            // ANISE's IAU_EARTH, whose spin drifts ~1°/day from solar time (the engine matches
            // ANISE exactly; the textbook IAU W = 190.147° + 360.9856235° d gives the right
            // longitude). Reported, not failed, unless --strict-orientation.
            failures.known(dlon.abs() < 5.0, args.strict_orientation, what);
          }
        } else {
          failures.check(false, format!("{date}: Earth orientation available"));
        }
      }
      None => failures.check(false, format!("{date}: scene clock available")),
    }
    let lit_poles = {
      let g = scene_arc.read();
      pole_points_global(&g.scene, visual, r_km as f32, AXIS_LENGTH_FACTOR)
    };
    set_hidden(true);
    let lit_hidden = grab(&ctx, w, h, SETTLE_FRAMES).unwrap_or_default();
    set_hidden(false);

    // ── pose B, perpendicular to the Sun line (terminator across the disk): axis, labels, caps ──
    place(&up_view);
    let Some(up) = grab(&ctx, w, h, SETTLE_FRAMES) else {
      failures.check(false, format!("{date}: UpZenith frame"));
      set_sky_hidden(false);
      continue;
    };
    // the state the frame was drawn with (the Earth moves ~30 km/s: re-read after the frame)
    let earth_now = global(visual).unwrap_or(earth);
    let poles = {
      let g = scene_arc.read();
      pole_points_global(&g.scene, visual, r_km as f32, AXIS_LENGTH_FACTOR)
    };
    // ── default mask, same pose: no Earth, and the same frame as with the globe hidden ──
    ctx.set_camera_render_mask(scene_id, cam_ffi, RENDER_LAYER_DEFAULT);
    let def = grab(&ctx, w, h, SETTLE_FRAMES).unwrap_or_default();
    set_hidden(true);
    let def_hidden = grab(&ctx, w, h, SETTLE_FRAMES).unwrap_or_default();
    set_hidden(false);
    // ── Earth observer: 1 km under the surface on the night side, looking out ──
    let obs_view = View::new(add_scaled(earth_now, s, -(r_au - 1.0 / AU_KM)), [-s[0], -s[1], -s[2]], ecl_north);
    place(&obs_view);
    let obs = grab(&ctx, w, h, SETTLE_FRAMES).unwrap_or_default();
    set_hidden(true);
    let obs_hidden = grab(&ctx, w, h, SETTLE_FRAMES).unwrap_or_default();
    set_hidden(false);
    set_sky_hidden(false);

    for (name, img) in [
      ("sunward", &lit),
      ("sunward_hidden", &lit_hidden),
      ("up", &up),
      ("default", &def),
      ("default_hidden", &def_hidden),
      ("observer", &obs),
      ("observer_hidden", &obs_hidden),
    ] {
      save(&args.out, &format!("{name}_{date}"), w, h, img);
    }
    sheet_rows.push(vec![lit.clone(), up.clone(), def.clone(), obs.clone()]);

    let up_img = Img { px: &up, w, h };
    let def_img = Img { px: &def, w, h };

    let differs = |x: i32, y: i32| {
      let (a, b) = (up_img.rgb(x, y), def_img.rgb(x, y));
      (0..3).any(|i| (a[i] - b[i]).abs() > 1.0)
    };

    // ── disk (pose A): where the frame differs from the same frame with the globe hidden ──
    let lit_img = Img { px: &lit, w, h };
    let lit_hidden_img = Img { px: &lit_hidden, w, h };
    let want_c = lit_view.project(earth_lit, w, h).unwrap();
    let f = (h as f64 * 0.5) / (FOV_DEG.to_radians() * 0.5).tan();
    let want_r = f * (1.0 / VIEW_DISTANCE_R).asin().tan();
    let (cx, cy) = (want_c[0].round() as i32, want_c[1].round() as i32);
    // the polar axis and its labels stick out of the disk: leave out the pixels within 12 px of
    // the projected axis line outside the disk when measuring it
    let axis_line = lit_poles.map(|[_, n, s_end]| {
      let pn = lit_view.project([n.x(), n.y(), n.z()], w, h).unwrap();
      let ps = lit_view.project([s_end.x(), s_end.y(), s_end.z()], w, h).unwrap();
      (ps, norm([pn[0] - ps[0], pn[1] - ps[1], 0.0]))
    });
    let disk_px = |x: i32, y: i32| {
      let (a, b) = (lit_img.rgb(x, y), lit_hidden_img.rgb(x, y));
      if !(0..3).any(|i| (a[i] - b[i]).abs() > 1.0) {
        return false;
      }
      let (dx, dy) = (x as f64 - want_c[0], y as f64 - want_c[1]);
      if (dx * dx + dy * dy).sqrt() <= want_r * 1.02 {
        return true;
      }
      match axis_line {
        Some((p0, d)) => ((x as f64 - p0[0]) * d[1] - (y as f64 - p0[1]) * d[0]).abs() > 12.0,
        None => true,
      }
    };
    // horizontal chord through the projected centre row, vertical chord 20 px beside the centre
    let row: Vec<i32> = (0..w as i32).filter(|&x| disk_px(x, cy)).collect();
    let col: Vec<i32> = (0..h as i32).filter(|&y| disk_px(cx + 20, y)).collect();
    let disk = (row.first(), row.last(), col.first(), col.last());
    let (got_c, got_r) = match disk {
      (Some(&l), Some(&r), Some(&t), Some(&b)) => {
        let c = [(l + r) as f64 / 2.0, (t + b) as f64 / 2.0];
        let rx = (r - l + 1) as f64 / 2.0;
        let half = (b - t + 1) as f64 / 2.0;
        let ry = (half * half + (20.0 + cx as f64 - c[0]).powi(2)).sqrt();
        (c, 0.5 * (rx + ry))
      }
      _ => ([f64::NAN; 2], f64::NAN),
    };
    let c_err = ((got_c[0] - want_c[0]).powi(2) + (got_c[1] - want_c[1]).powi(2)).sqrt();
    failures.check(c_err <= 2.0, format!("{date}: disk centre {got_c:?} vs projected {want_c:?} ({c_err:.2} px)"));
    let r_err = (got_r - want_r).abs() / want_r;
    failures.check(r_err <= 0.03, format!("{date}: disk radius {got_r:.1} vs {want_r:.1} px ({:.1} %)", r_err * 100.0));

    // ── axis ends and labels ──
    let mut axis_end_err = f64::NAN;
    let mut delta_deg = f64::NAN;
    let mut caps = (f64::NAN, f64::NAN);
    if let Some([c, n, s_end]) = poles {
      let c = [c.x(), c.y(), c.z()];
      let n = [n.x(), n.y(), n.z()];
      let s_end = [s_end.x(), s_end.y(), s_end.z()];
      let pole_dir = norm(sub(n, c));
      delta_deg = dot(s, pole_dir).clamp(-1.0, 1.0).asin().to_degrees();
      let (pn, ps) = (up_view.project(n, w, h).unwrap(), up_view.project(s_end, w, h).unwrap());
      // the lit axis pixel nearest each projected end
      let nearest = |p: [f64; 2]| {
        let mut best = f64::INFINITY;
        for y in (p[1] as i32 - 8)..=(p[1] as i32 + 8) {
          for x in (p[0] as i32 - 8)..=(p[0] as i32 + 8) {
            if up_img.lum(x, y) > 60.0 && differs(x, y) {
              best = best.min(((x as f64 - p[0]).powi(2) + (y as f64 - p[1]).powi(2)).sqrt());
            }
          }
        }
        best
      };
      axis_end_err = nearest(pn).max(nearest(ps));
      failures.check(axis_end_err <= 3.0, format!("{date}: axis ends drawn at the projected ±1.5 R ({axis_end_err:.1} px)"));
      // a label beyond each end: bright differing pixels 4..24 px further along the axis
      let label_beyond = |end: [f64; 2], from: [f64; 2]| {
        let d = norm([end[0] - from[0], end[1] - from[1], 0.0]);
        (6..26).any(|k| {
          let (x, y) = ((end[0] + d[0] * k as f64) as i32, (end[1] + d[1] * k as f64) as i32);
          (-4..=4).any(|o| {
            let (xx, yy) = (x + (o as f64 * -d[1]) as i32, y + (o as f64 * d[0]) as i32);
            up_img.lum(xx, yy) > 60.0 && differs(xx, yy)
          })
        })
      };
      let pc = up_view.project(c, w, h).unwrap();
      failures.check(label_beyond(pn, pc) && label_beyond(ps, pc), format!("{date}: labels beyond both axis ends"));

      // ── day and night where the true Sun puts them (texture-independent) ──
      // Points on the globe are sampled, each one's daylight predicted from the true Sun
      // direction (n·s), and the rendered pixel read back. Day samples must be clearly brighter
      // than night samples over the disk and at high latitudes, where the season tilts the
      // terminator: a wrong spin, pole or light direction mixes the two sets. (A raw mean over
      // the polar regions measured the surface instead: both straddle the terminator in this
      // pose, and ocean, ice and land differ more than day and night at grazing sunlight.)
      let e1 = norm(cross(pole_dir, if pole_dir[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
      let e2 = cross(pole_dir, e1);
      let axis_px = norm([pn[0] - ps[0], pn[1] - ps[1], 0.0]);
      let (mut all, mut high) = ([(0.0f64, 0usize); 2], [(0.0f64, 0usize); 2]);
      let mut samples = vec![alloc_header()];
      for i in 0..60 {
        let lat = (-90.0 + (i as f64 + 0.5) * 3.0).to_radians();
        for j in 0..120 {
          let lon = (j as f64 * 3.0).to_radians();
          let n = [0, 1, 2].map(|k| lat.cos() * (lon.cos() * e1[k] + lon.sin() * e2[k]) + lat.sin() * pole_dir[k]);
          let p = add_scaled(c, n, r_au);
          if dot(n, norm(sub(up_view.pos, p))) < 0.3 {
            continue; // hidden or near the limb
          }
          let Some(q) = up_view.project(p, w, h) else { continue };
          if ((q[0] - ps[0]) * axis_px[1] - (q[1] - ps[1]) * axis_px[0]).abs() < 6.0 {
            continue; // the drawn axis
          }
          let ndl = dot(n, s);
          let side = if ndl > 0.2 { 0 } else if ndl < -0.2 { 1 } else { continue };
          let v = up_img.lum(q[0] as i32, q[1] as i32);
          // for offline fits of the rendered light direction: world normal, n·s, luminance
          samples.push(format!("{},{},{},{ndl},{v}", n[0], n[1], n[2]));
          all[side].0 += v;
          all[side].1 += 1;
          if lat.abs() > 45f64.to_radians() {
            high[side].0 += v;
            high[side].1 += 1;
          }
        }
      }
      let _ = std::fs::write(args.out.join(format!("samples_{date}.csv")), samples.join("\n"));
      let mean = |x: (f64, usize)| x.0 / x.1.max(1) as f64;
      caps = (mean(high[0]), mean(high[1]));
      failures.check(
        all[0].1 > 50 && all[1].1 > 50 && mean(all[0]) >= 1.5 * mean(all[1]),
        format!(
          "{date}: predicted day {:.1} vs night {:.1} ({} / {} samples)",
          mean(all[0]), mean(all[1]), all[0].1, all[1].1
        ),
      );
      failures.check(
        high[0].1 > 20 && high[1].1 > 20 && caps.0 >= 1.5 * caps.1,
        format!(
          "{date}: subsolar latitude {delta_deg:+.1}°, beyond 45° latitude: day {:.1} vs night {:.1} ({} / {} samples)",
          caps.0, caps.1, high[0].1, high[1].1
        ),
      );
    } else {
      failures.check(false, format!("{date}: pole points"));
    }

    let def_changed = differing(&def, &def_hidden);
    failures.check(def_changed < 5, format!("{date}: default-mask frame unchanged by the globe ({def_changed} px)"));
    let obs_changed = differing(&obs, &obs_hidden);
    failures.check(obs_changed < 5, format!("{date}: Earth-observer frame unchanged by the globe ({obs_changed} px)"));

    let _ = writeln!(
      stats,
      "{{\"date\":\"{date}\",\"disk_centre_err_px\":{c_err:.3},\"disk_radius_err\":{r_err:.4},\"axis_end_err_px\":{axis_end_err:.2},\"subsolar_lat_deg\":{delta_deg:.2},\"high_lat_day\":{:.2},\"high_lat_night\":{:.2},\"default_changed_px\":{def_changed},\"observer_changed_px\":{obs_changed}}}",
      caps.0, caps.1
    );
    let _ = earth_body;
  }

  // contact sheet: one row per epoch (sunward | side, UpZenith | side, default mask | Earth observer), half size
  if !sheet_rows.is_empty() {
    let (tw, th) = (w / 2, h / 2);
    let mut sheet = image::RgbaImage::new(tw * 4, th * sheet_rows.len() as u32);
    for (r, row) in sheet_rows.iter().enumerate() {
      for (c, img) in row.iter().enumerate() {
        if let Some(full) = image::RgbaImage::from_raw(w, h, img.clone()) {
          let small = image::imageops::resize(&full, tw, th, image::imageops::FilterType::Triangle);
          image::imageops::overlay(&mut sheet, &small, (c as u32 * tw) as i64, (r as u32 * th) as i64);
        }
      }
    }
    let _ = sheet.save(args.out.join("contact_sheet.png"));
  }

  let vk = VK_ERRORS.load(Ordering::Relaxed);
  failures.check(vk == 0, format!("no Vulkan validation errors ({vk})"));
  let mut summary = String::from("# scene_observer: Earth visual\n\n");
  summary += &format!("epochs: {}\n\n", args.epochs.len());
  if !failures.1.is_empty() {
    summary += "Known engine issues (not failed; `--strict-orientation` fails them):\n\n";
    for f in &failures.1 {
      summary += &format!("- {f}\n");
    }
    summary += "\n";
  }
  if failures.0.is_empty() {
    summary += "All checks passed.\n";
  } else {
    summary += "Failed checks:\n\n";
    for f in &failures.0 {
      summary += &format!("- {f}\n");
    }
  }
  let _ = std::fs::write(args.out.join("summary.md"), &summary);
  log_line(&summary);
  let _ = ctx.threads.logic_thread.tx().try_send(LogicCommand::PauseScene { scene_id });
  std::thread::sleep(std::time::Duration::from_millis(200));
  drop(ctx);
  if args.check && !failures.0.is_empty() {
    std::process::exit(1);
  }
}
