//! Headless observer of the comet dust scene.
//!
//! Rebuilds the scene the .NET app shows (67P from its SPK, one dust jet, the default rotation
//! model), renders it off screen from a fixed pose at a fixed cadence over a month and writes PNG
//! frames plus per-frame numbers into an output directory, so an agent can look at the particle
//! system without the app.
//!
//! ```text
//! cargo run -p aethervk-core-rlib --example dust_observer --features std --release -- \
//!   --out DIR --days 30 --step-hours 24 --size 1280x800 --pose up-zenith [--check]
//! ```
//!
//! Outputs in `--out`: `frame_NNN.png`, `stats.jsonl`, `log.txt`, `contact_sheet.png`,
//! `summary.md`. See `scripts/README_observer.md`.

use aethervk_core_rlib::{
  scene::{
    BodyRotationalModel, CameraProjection, EntityId, StaticMeshComponent, TransformComponent,
    particles::{ParticleSystemComponent, ParticleSystemDrawParams, ParticleSystemEmitParams},
  },
  simulation_api::{
    SimulationContext, earth_observer,
    external_state::{CCometInitialized, CCometPositionSnapshot},
    structs::{KeplerianElements, LogicCommand, ReferenceOrbitMode, SceneEntityId, TaskStatusCode},
  },
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, Vector4, vec3::Vec3f32, vec3f64::DVec3, vec4::Quat, vec4f64::Quat64},
};
use hifitime::{Duration, Epoch};
use std::{
  fs::File,
  io::Write,
  path::{Path, PathBuf},
  sync::{
    Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc,
  },
  time::Instant,
};

// ───────────────────────────── constants (mirror the app) ─────────────────────────────

const AU_KM: f64 = 149_597_870.7;
const SUN_GM_M3_S2: f64 = 1.327_124_400_18e20;
const SPK_ID_67P: i32 = 1_000_012;
const HORIZONS_ID_67P: &str = "90000703";
/// `avkSimulationContext_addParticleSystem`: cluster TTL (30 scaled days).
const JET_TTL_US: i64 = 30 * 86_400 * 1_000_000;
const NUCLEUS_RADIUS_KM: f32 = 2.0;
const ROTATION_PERIOD_H: f64 = 12.4;

// ───────────────────────────── static callbacks ─────────────────────────────

static LOG_FILE: Mutex<Option<File>> = Mutex::new(None);
static LOG_ECHO: AtomicBool = AtomicBool::new(true);
static VK_ERRORS: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);
static TASK_ID: AtomicU64 = AtomicU64::new(0);
static PE_ID: AtomicU64 = AtomicU64::new(0);
static COMET_INIT_TX: Mutex<Option<mpsc::Sender<(u32, bool)>>> = Mutex::new(None);

/// Engine logger: every `oshal::log!` line lands in `log.txt` (and on stdout).
extern "C" fn on_logger(msg: *const core::ffi::c_char) {
  if msg.is_null() {
    return;
  }
  let s = unsafe { core::ffi::CStr::from_ptr(msg) }.to_string_lossy();
  let line = s.trim_end_matches(['\r', '\n']);
  if line.contains("DEVICE_LOST") || line.contains("DeviceLost") {
    DEVICE_LOST.store(true, Ordering::Relaxed);
  }
  if let Ok(mut g) = LOG_FILE.lock() {
    if let Some(f) = g.as_mut() {
      let _ = writeln!(f, "{line}");
    }
  }
  if LOG_ECHO.load(Ordering::Relaxed) {
    println!("{line}");
  }
}

fn log_line(msg: &str) {
  if let Ok(mut g) = LOG_FILE.lock() {
    if let Some(f) = g.as_mut() {
      let _ = writeln!(f, "[observer] {msg}");
    }
  }
  println!("[observer] {msg}");
}

/// Vulkan validation callback: counted (and logged) instead of panicking, so `--check` can report.
fn vk_error_callback(msg: &str) {
  VK_ERRORS.fetch_add(1, Ordering::Relaxed);
  log_line(&format!("VULKAN VALIDATION ERROR: {msg}"));
}

unsafe extern "C" fn on_render(_scene_id: u64, pe_id: u64, render_generation: u64) {
  if pe_id == PE_ID.load(Ordering::Acquire) {
    TASK_ID.store(render_generation, Ordering::Release);
  }
}

unsafe extern "C" fn on_external_state(state_id: u32, data: *const core::ffi::c_void) {
  // 4 = CometInitialized (CCometInitialized), 5 = CometPositionSnapshot (CCometPositionSnapshot)
  let ok = match state_id {
    4 => unsafe { (*(data as *const CCometInitialized)).success == 1 },
    5 => {
      let s = unsafe { *(data as *const CCometPositionSnapshot) };
      log_line(&format!(
        "CometPositionSnapshot spk {} pos AU ({:.6}, {:.6}, {:.6})",
        s.spk_id, s.pos_x, s.pos_y, s.pos_z
      ));
      true
    }
    _ => return,
  };
  if let Ok(g) = COMET_INIT_TX.lock() {
    if let Some(tx) = g.as_ref() {
      let _ = tx.send((state_id, ok));
    }
  }
}

// ───────────────────────────── small math ─────────────────────────────

type V3 = [f64; 3];
type Q = [f64; 4];

fn sub(a: V3, b: V3) -> V3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
fn norm(a: V3) -> f64 {
  dot(a, a).sqrt()
}
fn normalize(a: V3) -> V3 {
  let n = norm(a);
  if n < 1e-300 {
    return [0.0, 0.0, 0.0];
  }
  [a[0] / n, a[1] / n, a[2] / n]
}

/// Rotation with columns `x`, `y`, `z` (the camera's local axes in world space) → quaternion xyzw.
/// Camera convention: forward = local −Y, up = local +Z, right = local +X.
fn quat_from_basis(x: V3, y: V3, z: V3) -> Q {
  let (m00, m01, m02) = (x[0], y[0], z[0]);
  let (m10, m11, m12) = (x[1], y[1], z[1]);
  let (m20, m21, m22) = (x[2], y[2], z[2]);
  let trace = m00 + m11 + m22;
  if trace > 0.0 {
    let s = (trace + 1.0).sqrt() * 2.0;
    [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s]
  } else if m00 > m11 && m00 > m22 {
    let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
    [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
  } else if m11 > m22 {
    let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
    [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
  } else {
    let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
    [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]
  }
}

/// Camera basis (right, forward, up) in world space from the pose quaternion.
/// (right, forward, up) of the camera on screen. The renderer's screen +x is the camera's local
/// −X (checked 2026-10-10 against the Sun marker of a wide frame: the engine drew it at x = 397
/// where local +X put it at 563, the mirror about the centre column); only the nucleus pixel,
/// at the centre, had ever used this, so the sign went unnoticed.
fn camera_axes(q: Q) -> (V3, V3, V3) {
  (
    earth_observer::rotate(q, [-1.0, 0.0, 0.0]),
    earth_observer::rotate(q, [0.0, -1.0, 0.0]),
    earth_observer::rotate(q, [0.0, 0.0, 1.0]),
  )
}

// ───────────────────────────── arguments ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum Pose {
  UpZenith,
  AntiSun,
  Earth,
}

struct Args {
  out: PathBuf,
  days: f64,
  step_hours: f64,
  width: u32,
  height: u32,
  pose: Pose,
  play_seconds: Option<f64>,
  reemit_every: u32,
  check: bool,
  half_height_km: Option<f64>,
  fov_deg: f64,
  start: Epoch,
  spk: Option<PathBuf>,
  quiet: bool,
  frames_to_wait: u32,
  no_jet: bool,
  hide_sun: bool,
  full_teardown: bool,
  /// ignition `start + ignite_days` (default 0: the jet starts emitting at the start epoch)
  ignite_days: f64,
  /// pre-existing tail (history back 64 TTL before the start): the legacy behaviour
  prestart: bool,
  /// `--check`: the largest allowed ripple. With `--mc` it is the render's 9-px high-pass rms
  /// minus the truth's over the truth's region (a lattice is ripple the truth does not have; a
  /// fan edge or a lobed cloud has the same on both sides), default 0.05; without a truth it is
  /// the absolute high-pass rms of the τ image, default 0.25
  ripple_max: Option<f64>,
  /// nucleus rotation period (h); 0 = no rotation model (the jet site faces one direction)
  spin_hours: f64,
  /// `--mc N`: grains of the Monte-Carlo truth per frame (0 = off), see [`monte_carlo_truth`]
  mc: usize,
  /// `--mc-tier K`: the truth and the render hold tier K only (near frames: the old tiers'
  /// kernels are AU-sized and cover a 5-km frame as a background the Monte Carlo cannot sample)
  mc_tier: Option<u32>,
  /// `--mc-age-max S`: the truth and the render hold the dust younger than S seconds only (near
  /// frames: a 5-km frame shows the last hour of the jet while the older dust's kernels cover it
  /// as a background the Monte Carlo cannot sample)
  mc_age_max: Option<f64>,
  /// `--view-flags F`: dust view flags (`DUST_VIEW_TRACERS` 1, `DUST_VIEW_FLOW` 2; default 0, the
  /// plain optical depth, what the Monte-Carlo truth compares with and the app's default)
  view_flags: u32,
  /// `--check`: the largest allowed 2-D rms of the frame against its Monte-Carlo truth
  mc_max: f64,
}

fn usage() -> ! {
  eprintln!(
    "usage: dust_observer --out DIR [--days 30] [--step-hours 24] [--size 1280x800]\n\
     \x20      [--pose up-zenith|anti-sun|earth] [--play SECONDS] [--reemit-every N] [--check]\n\
     \x20      [--half-height-km KM] [--fov-deg DEG] [--start YYYY-MM-DD] [--spk FILE.bsp]\n\
     \x20      [--quiet] [--settle-frames N] [--no-jet] [--hide-sun] [--full-teardown]\n\
     \x20      [--ignite-days D] [--prestart] [--ripple-max R] [--spin-hours H] [--mc N] [--mc-max R] [--view-flags F]"
  );
  std::process::exit(2);
}

fn parse_args() -> Args {
  let mut a = Args {
    out: PathBuf::new(),
    days: 30.0,
    step_hours: 24.0,
    width: 1280,
    height: 800,
    pose: Pose::UpZenith,
    play_seconds: None,
    reemit_every: 0,
    check: false,
    half_height_km: None,
    fov_deg: 2.0,
    start: Epoch::from_gregorian_utc(2025, 10, 2, 0, 0, 0, 0),
    spk: None,
    quiet: false,
    frames_to_wait: 4,
    no_jet: false,
    hide_sun: false,
    full_teardown: false,
    ignite_days: 0.0,
    prestart: false,
    ripple_max: None,
    spin_hours: ROTATION_PERIOD_H,
    mc: 0,
    mc_max: 0.15,
    view_flags: 0,
    mc_tier: None,
    mc_age_max: None,
  };
  let argv: Vec<String> = std::env::args().skip(1).collect();
  let mut i = 0;
  let next = |i: &mut usize| -> String {
    *i += 1;
    argv.get(*i).cloned().unwrap_or_else(|| usage())
  };
  while i < argv.len() {
    match argv[i].as_str() {
      "--out" => a.out = PathBuf::from(next(&mut i)),
      "--days" => a.days = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--step-hours" => a.step_hours = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--size" => {
        let s = next(&mut i);
        let (w, h) = s.split_once('x').unwrap_or_else(|| usage());
        a.width = w.parse().unwrap_or_else(|_| usage());
        a.height = h.parse().unwrap_or_else(|_| usage());
      }
      "--pose" => {
        a.pose = match next(&mut i).as_str() {
          "up-zenith" | "zenith" => Pose::UpZenith,
          "anti-sun" | "antisun" => Pose::AntiSun,
          "earth" => Pose::Earth,
          _ => usage(),
        }
      }
      "--play" => a.play_seconds = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--reemit-every" => a.reemit_every = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--check" => a.check = true,
      "--quiet" => a.quiet = true,
      "--no-jet" => a.no_jet = true,
      "--full-teardown" => a.full_teardown = true,
      "--hide-sun" => a.hide_sun = true,
      "--prestart" => a.prestart = true,
      "--ignite-days" => a.ignite_days = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--ripple-max" => a.ripple_max = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--spin-hours" => a.spin_hours = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--mc" => a.mc = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--view-flags" => a.view_flags = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--mc-tier" => a.mc_tier = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--mc-age-max" => a.mc_age_max = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--mc-max" => a.mc_max = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--half-height-km" => {
        a.half_height_km = Some(next(&mut i).parse().unwrap_or_else(|_| usage()))
      }
      "--fov-deg" => a.fov_deg = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--settle-frames" => a.frames_to_wait = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--start" => {
        let s = next(&mut i);
        let p: Vec<i32> = s.split('-').filter_map(|x| x.parse().ok()).collect();
        if p.len() != 3 {
          usage();
        }
        a.start = Epoch::from_gregorian_utc(p[0], p[1] as u8, p[2] as u8, 0, 0, 0, 0);
      }
      "--spk" => a.spk = Some(PathBuf::from(next(&mut i))),
      "-h" | "--help" => usage(),
      other => {
        eprintln!("unknown argument {other}");
        usage();
      }
    }
    i += 1;
  }
  if a.out.as_os_str().is_empty() {
    eprintln!("--out DIR is required");
    usage();
  }
  a
}

// ───────────────────────────── SPK lookup / download ─────────────────────────────

fn ymd(e: Epoch) -> String {
  let (y, m, d, _, _, _, _) = e.to_gregorian_utc();
  format!("{y:04}-{m:02}-{d:02}")
}

fn parse_ymd(s: &str) -> Option<Epoch> {
  let p: Vec<i32> = s.split('-').filter_map(|x| x.parse().ok()).collect();
  (p.len() == 3).then(|| Epoch::from_gregorian_utc(p[0], p[1] as u8, p[2] as u8, 0, 0, 0, 0))
}

/// `~/.aethervk/spk_90000703_<from>_<to>.bsp` covering `[start, end]`, else a Horizons download.
fn find_or_fetch_spk(start: Epoch, end: Epoch) -> Result<PathBuf, String> {
  let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
  let dir = Path::new(&home).join(".aethervk");
  if let Ok(rd) = std::fs::read_dir(&dir) {
    let prefix = format!("spk_{HORIZONS_ID_67P}_");
    let mut candidates: Vec<PathBuf> = rd
      .filter_map(|e| e.ok().map(|e| e.path()))
      .filter(|p| {
        p.file_name()
          .and_then(|n| n.to_str())
          .map(|n| n.starts_with(&prefix) && n.ends_with(".bsp"))
          .unwrap_or(false)
      })
      .collect();
    candidates.sort();
    for p in candidates {
      let name = p.file_name().unwrap().to_str().unwrap();
      let body = &name[prefix.len()..name.len() - 4];
      if let Some((a, b)) = body.split_once('_') {
        if let (Some(a), Some(b)) = (parse_ymd(a), parse_ymd(b)) {
          if a <= start && b >= end {
            return Ok(p);
          }
        }
      }
    }
  }
  // download (like `fetch_67p_spk` in logic_thread_tests.rs)
  let from = ymd(start - Duration::from_days(1.0));
  let to = ymd(end + Duration::from_days(1.0));
  let url = format!(
    "https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND=%27{HORIZONS_ID_67P}%3B%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27SPK%27&OBJ_DATA=%27NO%27&START_TIME=%27{from}%27&STOP_TIME=%27{to}%27"
  );
  log_line(&format!(
    "no cached SPK covers {}..{}; downloading {url}",
    ymd(start),
    ymd(end)
  ));
  let text = reqwest::blocking::get(&url)
    .map_err(|e| format!("SPK download failed (network): {e}"))?
    .text()
    .map_err(|e| format!("SPK download failed (body): {e}"))?;
  let mut b64 = String::new();
  let mut seen = false;
  for line in text.lines() {
    if !seen {
      let t = line.trim_start();
      if t.starts_with("REFGL1NQ") {
        seen = true;
        b64.push_str(t.trim_end());
      }
      continue;
    }
    if line.trim().is_empty() {
      break;
    }
    b64.push_str(line.trim());
  }
  if !seen {
    return Err(format!(
      "SPK download failed: no SPK payload in the Horizons answer (first 300 chars): {}",
      text.chars().take(300).collect::<String>()
    ));
  }
  use base64::{Engine as _, engine::general_purpose};
  let bytes = general_purpose::STANDARD
    .decode(&b64)
    .or_else(|_| general_purpose::STANDARD_NO_PAD.decode(&b64))
    .map_err(|e| format!("SPK download failed: base64 decode: {e}"))?;
  std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
  let path = dir.join(format!("spk_{HORIZONS_ID_67P}_{from}_{to}.bsp"));
  std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
  Ok(path)
}

// ───────────────────────────── per-frame record ─────────────────────────────

#[derive(Clone, Debug)]
struct TierRec {
  live: u32,
  capacity: u32,
  youngest_s: f64,
  oldest_s: f64,
  band_min_s: f64,
  band_max_s: f64,
  caught_up: bool,
  unlit: u32,
  jet_unavailable: bool,
}

#[derive(Clone, Debug)]
struct FrameRec {
  index: u32,
  epoch: String,
  epoch_target: String,
  /// seconds from the jet's ignition to the frame's epoch (`None`: pre-existing tail)
  since_ignition_s: Option<f64>,
  seek_ok: bool,
  r_helio_au: f64,
  comet_au: V3,
  tiers: Vec<TierRec>,
  coma_km: f64,
  cam_au: V3,
  cam_quat: Q,
  half_height_au: f64,
  nucleus_px: (f64, f64),
  nucleus_rgb: (u8, u8, u8),
  mean_lum: f64,
  p99_lum: f64,
  nonblack_frac: f64,
  render_task: u64,
  passes_logged: String,
  /// the frame against its Monte-Carlo truth (`--mc`)
  mc: Option<McRec>,
}

/// The frame's τ image against the Monte-Carlo truth of the same emission ([`monte_carlo_truth`])
#[derive(Clone, Debug)]
struct McRec {
  /// 2-D rms of (render − truth)/truth over the region, both unit-sum and 5×5 box filtered
  rel_rms: f64,
  /// the same without an unresolved core: when the pixels above 1 % of the truth's peak are
  /// fewer than [`MC_CORE_RESOLVED_PX`], the coma is a sub-pixel point the truth's 1-px bins
  /// cannot place, and it would carry the whole error (1 AU: 99 % of the squared error in
  /// 3 000 px); the judged value
  rel_rms_judged: f64,
  /// the Monte-Carlo noise taken out of it (rms over the judged region)
  noise: f64,
  core_px: usize,
  /// 9-px high-pass ripple rms over the region: the render and the truth
  ripple_render: f64,
  ripple_truth: f64,
  /// region: truth ≥ 1 % of its peak after the box (px)
  region_px: usize,
  grains_px: f64,
  /// render/truth ratio percentiles over the region
  ratio_p05: f64,
  ratio_p95: f64,
  /// the truth's brightest pixel lies within 3 px of the nucleus pixel (the frame mapping)
  nucleus_ok: bool,
  grains: usize,
  batches: usize,
}

fn json_v3(v: V3) -> String {
  format!("[{:.9},{:.9},{:.9}]", v[0], v[1], v[2])
}

fn json_f(v: f64) -> String {
  if v.is_finite() {
    format!("{v}")
  } else {
    "null".to_string()
  }
}

impl FrameRec {
  fn to_json(&self) -> String {
    let tiers: Vec<String> = self
      .tiers
      .iter()
      .map(|t| {
        format!(
          "{{\"live_clusters\":{},\"capacity\":{},\"youngest_age_s\":{},\"oldest_age_s\":{},\"band_min_s\":{},\"band_max_s\":{},\"caught_up\":{},\"unlit_windows\":{},\"jet_unavailable\":{}}}",
          t.live,
          t.capacity,
          json_f(t.youngest_s),
          json_f(t.oldest_s),
          json_f(t.band_min_s),
          json_f(t.band_max_s),
          t.caught_up,
          t.unlit,
          t.jet_unavailable
        )
      })
      .collect();
    format!(
      "{{\"frame\":{},\"epoch\":\"{}\",\"epoch_target\":\"{}\",\"since_ignition_s\":{},\"seek_ok\":{},\"r_helio_au\":{:.6},\"comet_au\":{},\"tiers\":[{}],\"coma_radius_km\":{:.3},\"camera\":{{\"pos_au\":{},\"quat_xyzw\":[{:.9},{:.9},{:.9},{:.9}],\"half_height_au\":{:.9},\"half_height_km\":{:.1}}},\"nucleus_px\":[{:.2},{:.2}],\"nucleus_rgb\":[{},{},{}],\"mean_luminance\":{:.4},\"p99_luminance\":{:.2},\"nonblack_fraction\":{:.6},\"mc\":{},\"render_task\":{},\"dust_log\":\"{}\"}}",
      self.index,
      self.epoch,
      self.epoch_target,
      self.since_ignition_s.map_or(String::from("null"), |v| format!("{v:.1}")),
      self.seek_ok,
      self.r_helio_au,
      json_v3(self.comet_au),
      tiers.join(","),
      self.coma_km,
      json_v3(self.cam_au),
      self.cam_quat[0],
      self.cam_quat[1],
      self.cam_quat[2],
      self.cam_quat[3],
      self.half_height_au,
      self.half_height_au * AU_KM,
      self.nucleus_px.0,
      self.nucleus_px.1,
      self.nucleus_rgb.0,
      self.nucleus_rgb.1,
      self.nucleus_rgb.2,
      self.mean_lum,
      self.p99_lum,
      self.nonblack_frac,
      self
        .mc
        .as_ref()
        .map(|m| format!(
          "{{\"rel_rms\":{:.4},\"rel_rms_judged\":{:.4},\"noise\":{:.4},\"core_px\":{},\"ripple_render\":{:.4},\"ripple_truth\":{:.4},\"region_px\":{},\"grains_px\":{:.1},\"ratio_p05\":{:.3},\"ratio_p95\":{:.3},\"nucleus_ok\":{},\"grains\":{},\"batches\":{}}}",
          m.rel_rms, m.rel_rms_judged, m.noise, m.core_px, m.ripple_render, m.ripple_truth, m.region_px, m.grains_px, m.ratio_p05, m.ratio_p95, m.nucleus_ok, m.grains, m.batches
        ))
        .unwrap_or_else(|| "null".to_string()),
      self.render_task,
      self.passes_logged.replace('"', "'"),
    )
  }
}

// ───────────────────────────── engine helpers ─────────────────────────────

fn comet_position_au(ctx: &SimulationContext, scene_id: u64, body: EntityId) -> Option<V3> {
  let scene_arc = ctx.get_scene(scene_id)?;
  let g = scene_arc.read();
  let t = g.scene.global_transform_f64(body)?;
  Some([t.position.x(), t.position.y(), t.position.z()])
}

fn camera_pose(ctx: &SimulationContext, scene_id: u64, cam: EntityId) -> Option<(V3, Q)> {
  let scene_arc = ctx.get_scene(scene_id)?;
  let g = scene_arc.read();
  let t = g.scene.global_transform_f64(cam)?;
  Some((
    [t.position.x(), t.position.y(), t.position.z()],
    [
      t.rotation.0.x(),
      t.rotation.0.y(),
      t.rotation.0.z(),
      t.rotation.0.w(),
    ],
  ))
}

/// The synchronous camera write of `avkSimulationContext_setCameraTransform` mode 2.
fn set_camera_pose(ctx: &SimulationContext, scene_id: u64, cam: EntityId, pos_au: V3, q: Q) {
  let Some(scene_arc) = ctx.get_scene(scene_id) else {
    return;
  };
  let g = scene_arc.write();
  let _ = g.scene.set_global_transform_f64(
    cam,
    DVec3::from_components(pos_au[0], pos_au[1], pos_au[2]),
    Quat64::from_components(q[0], q[1], q[2], q[3]),
  );
  g.mark_component_changed(
    cam.as_ffi(),
    <aethervk_core_rlib::scene::HighResTransformComponent as aethervk_core_rlib::scene::ForeignSerializable>::COMPONENT_ID,
  );
}

/// `avkSimulationContext_setBodyRotationalModel`: scene component + cartesian cache + re-emit.
fn set_rotational_model(ctx: &SimulationContext, scene_id: u64, body: EntityId, period_h: f64) {
  let model = BodyRotationalModel {
    pole_ra: 0.0,
    pole_dec: 90.0,
    prime_meridian: 0.0,
    pole_ra_rate: 0.0,
    pole_dec_rate: 0.0,
    rotation_rate: 360.0 * 24.0 / period_h,
    body_fixed_orientation: true,
  };
  let scenes = ctx.scenes.read();
  let Some(scene_arc) = scenes.get_scene(scene_id) else {
    return;
  };
  let g = scene_arc.read();
  if g
    .scene
    .with_component_mut(body, |m: &mut BodyRotationalModel| *m = model)
    .is_none()
  {
    let _ = g.scene.add_component(body, model);
  }
  let key = SceneEntityId::new(scene_id, body);
  if let Some(mut state) = scenes.cartesian_state_cache.get_mut(&key) {
    if let Some(ref mut cs) = state.comet_state {
      cs.body_rotational_model = Some(model);
    }
  }
  g.scene.query1(|ps_id, ps: &ParticleSystemComponent| {
    if g.scene.get_parent(ps_id) == Some(body) {
      ps.dust.lock().request_reemit();
    }
  });
}

fn jet_params(seed: u32) -> ParticleSystemEmitParams {
  ParticleSystemEmitParams {
    latitude_rad: 20.0_f32.to_radians(),
    longitude_rad: 0.0,
    aperture_rad: 0.5,
    start_velocity_mean: 2.0,
    start_velocity_std: 0.5,
    mass_variability_perc: 0.0,
    seed,
    diametre_um: 100.0,
    density_gcm3: 0.533,
    scattering_efficiency: 1.0,
    afrho_0_cm: 100.0,
    afrho_power: 2.0,
    afrho_cutoff_au: 15.0,
    afrho_max_value_cm: 100_000.0,
  }
}

/// `avkSimulationContext_addParticleSystem` recipe.
fn add_jet(
  ctx: &SimulationContext,
  scene_id: u64,
  comet_body: EntityId,
  emit: ParticleSystemEmitParams,
  color: [f32; 4],
  emission_start_us: i64,
) -> Result<EntityId, String> {
  let scene_arc = ctx.get_scene(scene_id).ok_or("scene missing")?;
  let g = scene_arc.read();
  let parent_scale = g
    .scene
    .with_component(comet_body, |t: &TransformComponent| t.scale.x())
    .unwrap_or(1.0);
  let local_r = NUCLEUS_RADIUS_KM / parent_scale;
  let local_scale = (NUCLEUS_RADIUS_KM / 50.0) / parent_scale;
  let (lat, lon) = (emit.latitude_rad, emit.longitude_rad);
  let pos = Vec3f32::from_components(
    local_r * lat.cos() * lon.cos(),
    local_r * lat.cos() * lon.sin(),
    local_r * lat.sin(),
  );
  let jet = g.scene.spawn_entity("jet");
  g.scene.set_parent(jet, Some(comet_body));
  let _ = g.scene.add_component(
    jet,
    TransformComponent {
      position: pos,
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(local_scale, local_scale, local_scale),
    },
  );
  let sphere = std::sync::Arc::new(aethervk_core_rlib::simulation::comet::generate_uv_sphere(
    1.0, 16, 16, 1.0, false,
  ));
  let _ = g.scene.add_component(
    jet,
    StaticMeshComponent {
      asset_path: String::from("__internal_jet__"),
      mesh: sphere,
      emissive_color: color,
      is_visible: true,
    },
  );
  let rf = ctx.render_frontend().ok_or("render frontend missing")?;
  let ps = ParticleSystemComponent::new(
    rf,
    ctx.render_device_handle(),
    jet,
    emit,
    ParticleSystemDrawParams {
      stream_color: color,
    },
    JET_TTL_US,
    emission_start_us,
  )
  .map_err(|e| format!("ParticleSystemComponent::new: {e:?}"))?;
  let _ = g.scene.add_component(jet, ps);
  Ok(jet)
}

fn current_epoch(ctx: &SimulationContext, scene_id: u64) -> Option<Epoch> {
  ctx.scenes.read().time_managers.get(&scene_id).map(|t| t.current_epoch())
}

fn tiers_now(ctx: &SimulationContext, scene_id: u64) -> Vec<TierRec> {
  ctx
    .dust_stats(scene_id)
    .into_iter()
    .map(|t| TierRec {
      live: t.live_clusters,
      capacity: t.capacity,
      youngest_s: t.youngest_age_s,
      oldest_s: t.oldest_age_s,
      band_min_s: t.band_min_s,
      band_max_s: t.band_max_s,
      caught_up: t.caught_up,
      unlit: t.unlit_windows,
      jet_unavailable: t.jet_unavailable,
    })
    .collect()
}

/// Waits until every tier reports `caught_up` (the seek's history rebuild may outlive the 2 s
/// `seek_epoch_sync` budget) and no physics task is in flight; returns whether it did.
fn wait_caught_up(ctx: &SimulationContext, scene_id: u64, timeout_s: f64) -> bool {
  let t0 = Instant::now();
  loop {
    let physics_busy = ctx
      .get_scene(scene_id)
      .map(|s| s.read().active_physics_task.load(Ordering::Acquire))
      .unwrap_or(false);
    let tiers = ctx.dust_stats(scene_id);
    let caught = tiers.iter().all(|t| t.caught_up);
    if caught && !physics_busy {
      return true;
    }
    if t0.elapsed().as_secs_f64() > timeout_s {
      return false;
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
  }
}

/// `test_trajectory_occlusion_render::wait_and_download`: a few distinct render generations,
/// then the swapchain download (BGRA8).
fn wait_and_download(
  ctx: &SimulationContext,
  width: u32,
  height: u32,
  frames: u32,
  timeout_ms: u64,
) -> Option<(Vec<u8>, u64)> {
  TASK_ID.store(0, Ordering::Release);
  let poll = std::time::Duration::from_millis(5);
  let max_polls = timeout_ms / 5;
  let mut seen = 0;
  let mut last = 0;
  for _ in 0..max_polls {
    if DEVICE_LOST.load(Ordering::Relaxed) {
      return None;
    }
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
  let tid = TASK_ID.load(Ordering::Acquire);
  for _ in 0..max_polls {
    if !matches!(ctx.get_task_status(tid), TaskStatusCode::Pending) {
      break;
    }
    std::thread::sleep(poll);
  }
  let mut color = vec![0u8; (width * height * 4) as usize];
  if !unsafe { ctx.download_image(tid, color.as_mut_ptr(), color.len()) } {
    return None;
  }
  Some((color, tid))
}

fn bgra_to_rgba(buf: &mut [u8]) {
  for px in buf.chunks_exact_mut(4) {
    px.swap(0, 2);
  }
}

fn luminance_stats(rgba: &[u8]) -> (f64, f64, f64) {
  let n = rgba.len() / 4;
  let mut lums: Vec<f32> = Vec::with_capacity(n);
  let mut sum = 0.0f64;
  let mut nonblack = 0usize;
  for px in rgba.chunks_exact(4) {
    let l = 0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32;
    sum += l as f64;
    if px[0] > 8 || px[1] > 8 || px[2] > 8 {
      nonblack += 1;
    }
    lums.push(l);
  }
  lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let p99 = lums[((n as f64 * 0.99) as usize).min(n - 1)] as f64;
  (sum / n as f64, p99, nonblack as f64 / n as f64)
}

// ───────────────────────────── main ─────────────────────────────

fn main() {
  let args = parse_args();
  std::fs::create_dir_all(&args.out).expect("create --out");
  *LOG_FILE.lock().unwrap() = Some(File::create(args.out.join("log.txt")).expect("log.txt"));
  LOG_ECHO.store(!args.quiet, Ordering::Relaxed);
  let t_run = Instant::now();

  let n_frames = ((args.days * 24.0 / args.step_hours).round() as u32).max(1) + 1;
  let step = Duration::from_seconds(args.step_hours * 3600.0);
  let start = args.start;
  let range_days = (args.days + 1.0).max(31.0);
  let end = start + Duration::from_days(range_days);
  log_line(&format!(
    "run: start {start} end {end} frames {n_frames} step {}h size {}x{} pose {:?} play {:?} reemit_every {} cpu_particles {}",
    args.step_hours,
    args.width,
    args.height,
    args.pose,
    args.play_seconds,
    args.reemit_every,
    std::env::var("AETHERVK_PARTICLES_CPU").as_deref() == Ok("1")
  ));

  // SPK (before the GPU comes up, so a failed download is cheap)
  let spk = match args.spk.clone().map(Ok).unwrap_or_else(|| find_or_fetch_spk(start, end)) {
    Ok(p) => p,
    Err(e) => {
      log_line(&format!("FATAL: 67P SPK unavailable: {e}"));
      std::process::exit(3);
    }
  };
  log_line(&format!("67P SPK: {}", spk.display()));

  // callbacks (before startup, like runtime_bridge::install)
  let (init_tx, init_rx) = mpsc::channel::<(u32, bool)>();
  *COMET_INIT_TX.lock().unwrap() = Some(init_tx);
  SimulationContext::set_logger_callback(Some(on_logger));
  aethervk_core_rlib::simulation_api::set_external_state_simulation_callback(Some(
    on_external_state,
  ));
  SimulationContext::set_render_callback(Some(on_render));

  // startup (the FFI `avkSimulationContext_startup` recipe)
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
    let p = spk.to_string_lossy().to_string();
    if let Err(e) = logic.almanac_data.load_almanac(aethervk_oshal_rlib::os::fs::PathBuf::from(&p))
    {
      log_line(&format!("FATAL: SPK {p}: {e}"));
      std::process::exit(4);
    }
  }

  let scene_ret = ctx.create_empty_scene2(false, start, end).expect("create_empty_scene2");
  let scene_id = scene_ret.scene_id;
  let comet_body = EntityId::from_ffi(scene_ret.comet_body);
  log_line(&format!(
    "scene {scene_id} comet_body {} earth_body {}",
    scene_ret.comet_body, scene_ret.earth_body
  ));

  // comet init: two-phase commit, wait for CometInitialized (4) then CometPositionSnapshot (5)
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::TryInitComet {
      scene_id,
      spk_id: SPK_ID_67P,
      proposed_start: start,
      proposed_end: end,
      keplerian_elements: KeplerianElements {
        eccentricity: 0.6402,
        perihelion_distance_au: 1.2432,
        inclination_deg: 3.871,
        longitude_of_ascending_node_deg: 36.33,
        argument_of_perihelion_deg: 22.15,
        time_of_perihelion_jd_tdb: f64::NAN,
      },
      reference_mode: ReferenceOrbitMode::OsculatingAtStart,
    })
    .expect("TryInitComet send");
  let mut got_init = false;
  let t0 = Instant::now();
  while t0.elapsed().as_secs() < 60 && !got_init {
    match init_rx.recv_timeout(std::time::Duration::from_secs(1)) {
      Ok((4, ok)) => {
        got_init = true;
        if !ok {
          log_line("FATAL: TryInitComet failed (epoch range outside the SPK?)");
          std::process::exit(5);
        }
        log_line("CometInitialized");
      }
      Ok(_) => {}
      Err(mpsc::RecvTimeoutError::Timeout) => {}
      Err(_) => break,
    }
  }
  if !got_init {
    log_line("FATAL: no CometInitialized callback within 60 s");
    std::process::exit(5);
  }
  // the comet is repositioned by BuildCometTrajectory (CometPositionSnapshot is emitted before
  // CometInitialized); wait until the body has left the origin
  let t0 = Instant::now();
  while t0.elapsed().as_secs_f64() < 10.0 {
    if comet_position_au(&ctx, scene_id, comet_body)
      .map(|p| norm(p) > 1e-6)
      .unwrap_or(false)
    {
      break;
    }
    std::thread::sleep(std::time::Duration::from_millis(20));
  }

  // rotation model (app default: pole RA 0, Dec 90, 12.4 h, body-fixed); `--spin-hours 0`: none
  if args.spin_hours > 0.0 {
    set_rotational_model(&ctx, scene_id, comet_body, args.spin_hours);
  } else {
    log_line("no rotation model (--spin-hours 0)");
  }

  // one jet, ignited at `start + ignite_days` (nothing before it) unless `--prestart`
  let ignition: Option<Epoch> =
    (!args.prestart).then(|| args.start + Duration::from_days(args.ignite_days));
  let emit = jet_params(7);
  let jet_color = [1.0, 0.6, 0.2, 1.0];
  if args.no_jet {
    log_line("no jet (--no-jet): dust-free scene");
  } else {
    let emission_start_us = match ignition {
      None => aethervk_core_rlib::scene::particles::EMISSION_START_PREEXISTING,
      Some(t_on) => aethervk_core_rlib::scene::particles::emission_start_us_from_epoch(t_on),
    };
    log_line(&format!(
      "jet ignition: {}",
      if args.prestart {
        String::from("pre-existing tail")
      } else {
        format!("start + {:.2} d", args.ignite_days)
      }
    ));
    match add_jet(
      &ctx,
      scene_id,
      comet_body,
      emit,
      jet_color,
      emission_start_us,
    ) {
      Ok(j) => log_line(&format!("jet entity {}", j.as_ffi())),
      Err(e) => {
        log_line(&format!("FATAL: add_jet: {e}"));
        std::process::exit(6);
      }
    }
  }
  if args.hide_sun {
    if let Some(scene_arc) = ctx.get_scene(scene_id) {
      let g = scene_arc.read();
      let mut suns = Vec::new();
      g.scene.query1(|id, _s: &aethervk_core_rlib::scene::SunComponent| suns.push(id));
      for id in suns {
        let _ = g.scene.add_component(id, aethervk_core_rlib::scene::HiddenComponent {});
      }
      log_line("sun hidden (--hide-sun)");
    }
  }

  // view settings
  ctx.set_dust_view_flags(scene_id, args.view_flags);
  if let Some(k) = args.mc_tier {
    // the render side of `--mc-tier`: the frame's splat draws that tier only
    unsafe { std::env::set_var("AETHERVK_DUST_DEBUG_TIER", k.to_string()) };
    log_line(&format!(
      "[observer] tier {k} only (--mc-tier): render and truth"
    ));
  }
  if let Some(a) = args.mc_age_max {
    // the render side of `--mc-age-max`: every tier's age band capped there
    unsafe { std::env::set_var("AETHERVK_DUST_DEBUG_AGE_MAX", format!("{a}")) };
    log_line(&format!(
      "[observer] dust younger than {a} s only (--mc-age-max): render and truth"
    ));
  }
  ctx.set_dust_flow_speed(scene_id, 1.0);
  ctx.set_dust_softening(scene_id, 1e-2);

  // presentation engine + camera
  let pe = ctx
    .create_presentation_engine(scene_id, args.width, args.height)
    .expect("create_presentation_engine");
  PE_ID.store(pe.0 as u64, Ordering::Release);
  let aspect = args.width as f64 / args.height as f64;
  let cam_id = match args.pose {
    Pose::Earth => ctx
      .add_perspective_camera(
        scene_id,
        pe,
        "observer_cam",
        args.fov_deg.to_radians() as f32,
        1e-5,
        20.0,
      )
      .expect("add_perspective_camera"),
    _ => ctx
      .add_orthographic_camera(scene_id, pe, "observer_cam", 1.0, 0.1, 1000.0)
      .expect("add_orthographic_camera"),
  };
  let cam = EntityId::from_ffi(cam_id.get());

  // first seek: coma radius + comet distance are needed for the framing
  let mut problems: Vec<String> = Vec::new();
  let seek0 = ctx.seek_epoch_sync(scene_id, start);
  let caught0 = wait_caught_up(&ctx, scene_id, 120.0);
  log_line(&format!("initial seek ok {seek0}, caught up {caught0}"));
  if !caught0 {
    problems.push(format!(
      "initial seek: dust never caught up: {:?}",
      ctx.dust_stats(scene_id)
    ));
  }
  if ctx.dust_stats(scene_id).iter().any(|t| t.jet_unavailable) {
    problems
      .push("jet state unavailable: the comet is not in the cartesian cache (no emission)".into());
  }
  let comet0 = comet_position_au(&ctx, scene_id, comet_body).unwrap_or([0.0; 3]);
  let r0_au = norm(comet0);
  let coma0_km = ctx.dust_coma_radius_km(scene_id);
  let cfg0 = emit.dust_emit_config(r0_au as f32, JET_TTL_US);
  let g_ms2 = SUN_GM_M3_S2 / (r0_au * AU_KM * 1e3).powi(2);
  let t_tail_s = args.days * 86_400.0;
  let tail_km = cfg0.beta_ref as f64 * g_ms2 * t_tail_s * t_tail_s / 1e3; // 2 · ½ β g T²
  let half_height_km =
    args.half_height_km.unwrap_or_else(|| (3.0 * coma0_km).max(tail_km).max(2.0e4));
  let half_height_au = half_height_km / AU_KM;
  let cam_height_au = half_height_au; // camera stands one half-height above the comet
  log_line(&format!(
    "framing: r {r0_au:.4} AU, g {g_ms2:.3e} m/s², beta_ref {:.4}, coma {coma0_km:.1} km, tail(β g T²) {tail_km:.1} km => half-height {half_height_km:.1} km ({half_height_au:.6} AU), camera height {:.1} km",
    cfg0.beta_ref,
    cam_height_au * AU_KM
  ));

  // projection (AU, like the C# RequestOrthographicProjection)
  if args.pose != Pose::Earth {
    let hw = (half_height_au * aspect) as f32;
    let hh = half_height_au as f32;
    let near = (cam_height_au * 0.01) as f32;
    let far = (cam_height_au * 50.0) as f32;
    ctx
      .threads
      .logic_thread
      .tx()
      .try_send(LogicCommand::SetCameraTransform {
        scene_id,
        camera_id: cam.as_ffi(),
        transform: None,
        projection: Some(CameraProjection::Orthographic {
          left: -hw,
          right: hw,
          bottom: -hh,
          top: hh,
          near,
          far,
        }),
      })
      .expect("SetCameraTransform send");
  }

  // pose: a function of the comet position
  let pose_for = |comet: V3| -> (V3, Q) {
    match args.pose {
      Pose::UpZenith => {
        let pos = [comet[0], comet[1], comet[2] + cam_height_au];
        (
          pos,
          earth_observer::look_at_origin_from([0.0, 0.0, cam_height_au], None),
        )
      }
      Pose::AntiSun => {
        let pos = [comet[0], comet[1], comet[2] + cam_height_au];
        let forward = [0.0, 0.0, -1.0];
        let mut right = normalize([-comet[0], -comet[1], 0.0]);
        if norm(right) < 0.5 {
          right = [1.0, 0.0, 0.0];
        }
        // engine convention: right × up = forward (see look_at_origin_from)
        let up = cross(forward, right);
        (
          pos,
          quat_from_basis(right, [-forward[0], -forward[1], -forward[2]], up),
        )
      }
      Pose::Earth => (comet, [0.0, 0.0, 0.0, 1.0]),
    }
  };
  if args.pose == Pose::Earth {
    let (p, q) = (
      [comet0[0], comet0[1], comet0[2] + cam_height_au],
      earth_observer::look_at_origin_from([0.0, 0.0, cam_height_au], None),
    );
    set_camera_pose(&ctx, scene_id, cam, p, q);
    let ok = ctx.set_earth_observer(
      scene_id,
      cam.as_ffi(),
      Some(earth_observer::EarthObserverMode::CometTracking),
      scene_ret.earth_body,
      scene_ret.comet_body,
      45.0,
      0.0,
      [0.0, 0.0, 0.0, 1.0],
    );
    log_line(&format!("earth observer (comet tracking) set: {ok}"));
  } else {
    let (p, q) = pose_for(comet0);
    set_camera_pose(&ctx, scene_id, cam, p, q);
  }

  // ───── frames ─────
  let mut stats = File::create(args.out.join("stats.jsonl")).expect("stats.jsonl");
  let mut records: Vec<FrameRec> = Vec::new();
  let mut frames_rgba: Vec<image::RgbaImage> = Vec::new();
  let mut rot_toggle = false;

  for i in 0..n_frames {
    if DEVICE_LOST.load(Ordering::Relaxed) {
      problems.push(format!(
        "frame {i}: GPU device lost (VK_ERROR_DEVICE_LOST), run aborted"
      ));
      log_line(&format!(
        "frame {i}: GPU device lost, aborting the frame loop"
      ));
      break;
    }
    let target = start + step * (i as i64);
    let t_frame = Instant::now();
    let mut seek_ok;
    if let Some(play_s) = args.play_seconds {
      if i == 0 {
        seek_ok = ctx.seek_epoch_sync(scene_id, target);
      } else {
        if args.reemit_every > 0 && i % args.reemit_every == 0 {
          rot_toggle = !rot_toggle;
          let period = if rot_toggle {
            ROTATION_PERIOD_H + 0.1
          } else {
            ROTATION_PERIOD_H
          };
          set_rotational_model(&ctx, scene_id, comet_body, period);
          log_line(&format!(
            "frame {i}: rotation period toggled to {period} h (re-emit)"
          ));
        }
        let speed = args.step_hours * 3600.0 / play_s;
        let started = ctx.start_simulation(
          scene_id,
          aethervk_oshal_rlib::os::time::v2::SimSpeed::Custom(speed),
        );
        std::thread::sleep(std::time::Duration::from_secs_f64(play_s));
        let paused = ctx.pause_simulation_sync(scene_id);
        seek_ok = started && paused;
        log_line(&format!(
          "frame {i}: played {play_s}s at {speed:.1}x, started {started} paused {paused}"
        ));
      }
    } else {
      seek_ok = ctx.seek_epoch_sync(scene_id, target);
      if !seek_ok {
        // the seek may still be applying (history rebuild outlives the 2 s budget): wait and
        // check where the clock ended up
        let caught = wait_caught_up(&ctx, scene_id, 120.0);
        let now = current_epoch(&ctx, scene_id);
        if caught && now.map(|e| (e - target).abs() < Duration::from_seconds(1.0)).unwrap_or(false)
        {
          seek_ok = true;
        } else {
          log_line(&format!(
            "frame {i}: seek to {target} failed (clock {now:?})"
          ));
        }
      }
    }
    let caught = wait_caught_up(&ctx, scene_id, 120.0);
    if !caught {
      problems.push(format!(
        "frame {i}: dust tiers never caught up within 120 s"
      ));
    }
    let actual = current_epoch(&ctx, scene_id).unwrap_or(target);

    let comet = comet_position_au(&ctx, scene_id, comet_body).unwrap_or([0.0; 3]);
    if args.pose != Pose::Earth {
      let (p, q) = pose_for(comet);
      set_camera_pose(&ctx, scene_id, cam, p, q);
    }
    // settle: the pose/seek reach the renderer on the next extraction
    let t_dl = Instant::now();
    let dl = wait_and_download(&ctx, args.width, args.height, args.frames_to_wait, 60_000);
    // the render cost: `frames_to_wait` distinct render generations were waited for
    let render_ms = t_dl.elapsed().as_secs_f64() * 1e3 / args.frames_to_wait.max(1) as f64;
    log_line(&format!(
      "[observer] render ≈ {render_ms:.1} ms/frame ({} generations)",
      args.frames_to_wait
    ));
    let (mut rgba, tid) = match dl {
      Some(x) => x,
      None => {
        problems.push(format!("frame {i}: render download failed/timed out"));
        log_line(&format!("frame {i}: render download failed"));
        (vec![0u8; (args.width * args.height * 4) as usize], 0)
      }
    };
    bgra_to_rgba(&mut rgba);

    let (cam_pos, cam_q) =
      camera_pose(&ctx, scene_id, cam).unwrap_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0]));
    let (right, forward, up) = camera_axes(cam_q);
    let d = sub(comet, cam_pos);
    let (sx, sy) = (dot(d, right), dot(d, up));
    let depth = dot(d, forward);
    let (px, py) = if args.pose == Pose::Earth {
      let tan_h = (args.fov_deg.to_radians() / 2.0).tan();
      (
        args.width as f64 / 2.0 * (1.0 + sx / depth.max(1e-12) / (tan_h * aspect)),
        args.height as f64 / 2.0 * (1.0 - sy / depth.max(1e-12) / tan_h),
      )
    } else {
      (
        args.width as f64 / 2.0 * (1.0 + sx / (half_height_au * aspect)),
        args.height as f64 / 2.0 * (1.0 - sy / half_height_au),
      )
    };
    let (ix, iy) = (
      px.round().clamp(0.0, args.width as f64 - 1.0) as u32,
      py.round().clamp(0.0, args.height as f64 - 1.0) as u32,
    );
    let o = ((iy * args.width + ix) * 4) as usize;
    let rgb = (rgba[o], rgba[o + 1], rgba[o + 2]);
    let (mean_l, p99_l, nonblack) = luminance_stats(&rgba);

    let path = args.out.join(format!("frame_{i:03}.png"));
    image::save_buffer(
      &path,
      &rgba,
      args.width,
      args.height,
      image::ColorType::Rgba8,
    )
    .expect("save png");
    // the dust pyramid of this frame (raw u32 words, header first: dust::PyramidLayout) and a
    // one-line summary of it, for offline analysis without RenderDoc
    let mut mc_rec: Option<McRec> = None;
    if let Some(pyr) = ctx.download_dust_pyramid(pe) {
      use aethervk_core_rlib::scene::dust as d;
      let words: Vec<u8> = pyr.iter().flat_map(|w| w.to_le_bytes()).collect();
      std::fs::write(args.out.join(format!("frame_{i:03}.pyr")), &words).expect("save pyr");
      if args.mc > 0 && !args.no_jet {
        let t_mc = Instant::now();
        let view = McView {
          width: args.width,
          height: args.height,
          cam_pos,
          cam_q,
          half_height_au,
          aspect,
          perspective_tan_h: (args.pose == Pose::Earth)
            .then(|| (args.fov_deg.to_radians() / 2.0).tan()),
          comet_au: comet,
          nucleus_px: (px, py),
          mc_tier: args.mc_tier,
          mc_age_max: args.mc_age_max,
          nucleus_px_radius: NUCLEUS_RADIUS_KM as f64
            / (2.0 * half_height_au * AU_KM / args.height as f64),
        };
        match monte_carlo_truth(&ctx, scene_id, &pyr, &view, args.mc) {
          Some((rec, render_img, truth_img)) => {
            log_line(&format!(
              "[observer] mc frame {i}: rel rms {:.3} (judged {:.3} after noise {:.3}, core {} px) over {} px, ripple render {:.3} truth {:.3}, ratio p05/p95 {:.2}/{:.2}, {:.0} grains/px ({} grains, {} batches), nucleus {}, {:.1} s",
              rec.rel_rms,
              rec.rel_rms_judged,
              rec.noise,
              rec.core_px,
              rec.region_px,
              rec.ripple_render,
              rec.ripple_truth,
              rec.ratio_p05,
              rec.ratio_p95,
              rec.grains_px,
              rec.grains,
              rec.batches,
              if rec.nucleus_ok { "ok" } else { "OFF" },
              t_mc.elapsed().as_secs_f64()
            ));
            write_mc_triptych(
              &args.out.join(format!("mc_{i:03}.png")),
              &render_img,
              &truth_img,
              args.width,
              args.height,
            );
            // the raw images (f32 little endian: render then truth, both unit-sum and box filtered)
            let raw: Vec<u8> = render_img
              .iter()
              .chain(truth_img.iter())
              .flat_map(|v| v.to_le_bytes())
              .collect();
            let _ = std::fs::write(args.out.join(format!("mc_{i:03}.f32")), raw);
            if args.check && rec.region_px >= 2000 && rec.rel_rms_judged > args.mc_max {
              problems.push(format!(
                "frame {i}: {:.3} rms from the Monte-Carlo truth (limit {:.3}) over {} px{}",
                rec.rel_rms_judged,
                args.mc_max,
                rec.region_px,
                if rec.core_px < MC_CORE_RESOLVED_PX {
                  format!(" (the {} px core left out)", rec.core_px)
                } else {
                  String::new()
                }
              ));
            }
            if args.check && !rec.nucleus_ok {
              problems.push(format!("frame {i}: the Monte-Carlo truth's peak is not at the nucleus pixel (frame mapping)"));
            }
            mc_rec = Some(rec);
          }
          None => log_line(&format!(
            "[observer] mc frame {i}: no truth (no batches or no jet state)"
          )),
        }
      }
      let levels = pyr[d::PYR_LEVELS as usize] as usize;
      let mut per_level = Vec::new();
      for l in 0..levels {
        let off = pyr[d::PYR_TABLE as usize + 3 * l] as usize;
        let w = pyr[d::PYR_TABLE as usize + 3 * l + 1] as usize;
        let h = pyr[d::PYR_TABLE as usize + 3 * l + 2] as usize;
        let mut sum = 0u64;
        let mut nz = 0usize;
        let mut max = 0u32;
        for t in 0..w * h {
          let c = pyr[off + t * d::PYRAMID_TEXEL_WORDS as usize];
          sum += c as u64;
          nz += (c > 0) as usize;
          max = max.max(c);
        }
        per_level.push(format!("L{l}:{nz}/{sum}/{max}"));
      }
      log_line(&format!(
        "[observer] pyramid frame {i}: levels {levels} tau_max {:.3e} mask {:#x} tracer {:.1} (texels/Σcounts/max) {}",
        f32::from_bits(pyr[d::PYR_TAU_MAX as usize]),
        pyr[d::PYR_LEVEL_MASK as usize],
        f32::from_bits(pyr[d::PYR_TRACER_COUNTS as usize]),
        per_level.join(" ")
      ));
    }
    let img = image::RgbaImage::from_raw(args.width, args.height, rgba).unwrap();
    frames_rgba.push(img);

    let tiers = tiers_now(&ctx, scene_id);
    let rec = FrameRec {
      index: i,
      epoch: format!("{actual}"),
      epoch_target: format!("{target}"),
      since_ignition_s: ignition.map(|t_on| (actual - t_on).to_seconds()),
      seek_ok,
      r_helio_au: norm(comet),
      comet_au: comet,
      tiers,
      coma_km: ctx.dust_coma_radius_km(scene_id),
      cam_au: cam_pos,
      cam_quat: cam_q,
      half_height_au,
      nucleus_px: (px, py),
      nucleus_rgb: rgb,
      mean_lum: mean_l,
      p99_lum: p99_l,
      nonblack_frac: nonblack,
      render_task: tid,
      passes_logged: String::new(),
      mc: mc_rec,
    };
    let _ = writeln!(stats, "{}", rec.to_json());
    let _ = stats.flush();
    log_line(&format!(
      "frame {i:03} {actual} r {:.4} AU | tiers {} | coma {:.0} km | nucleus px ({:.0},{:.0}) rgb {:?} | mean L {:.3} p99 L {:.1} nonblack {:.4} | {:.1}s",
      rec.r_helio_au,
      rec
        .tiers
        .iter()
        .map(|t| format!(
          "{}/{}{}",
          t.live,
          t.capacity,
          if t.caught_up { "" } else { "(building)" }
        ))
        .collect::<Vec<_>>()
        .join(" "),
      rec.coma_km,
      px,
      py,
      rgb,
      mean_l,
      p99_l,
      nonblack,
      t_frame.elapsed().as_secs_f64()
    ));
    records.push(rec);
  }

  // ───── contact sheet ─────
  {
    let per_row = 6u32;
    let (tw, th) = (args.width / 4, args.height / 4);
    let rows = (n_frames + per_row - 1) / per_row;
    let mut sheet =
      image::RgbaImage::from_pixel(per_row * tw, rows * th, image::Rgba([0, 0, 0, 255]));
    for (k, img) in frames_rgba.iter().enumerate() {
      let small = image::imageops::resize(img, tw, th, image::imageops::FilterType::Triangle);
      let (cx, cy) = ((k as u32 % per_row) * tw, (k as u32 / per_row) * th);
      image::imageops::overlay(&mut sheet, &small, cx as i64, cy as i64);
      // 1-px frame separator
      for x in 0..tw {
        sheet.put_pixel(cx + x, cy, image::Rgba([40, 40, 40, 255]));
      }
      for y in 0..th {
        sheet.put_pixel(cx, cy + y, image::Rgba([40, 40, 40, 255]));
      }
    }
    sheet.save(args.out.join("contact_sheet.png")).expect("contact sheet");
  }

  // ───── checks ─────
  let vk_errors = VK_ERRORS.load(Ordering::Relaxed);
  let mut check_failures: Vec<String> = Vec::new();
  if vk_errors > 0 {
    check_failures.push(format!("{vk_errors} Vulkan validation error(s)"));
  }
  for r in records.iter().skip(1) {
    if r.tiers.iter().any(|t| !t.caught_up) {
      check_failures.push(format!("frame {}: a tier is still building", r.index));
    }
  }
  // ignition: no dust before it, and never older than the time since it (one window of slack)
  for r in &records {
    let Some(since) = r.since_ignition_s else {
      continue;
    };
    for (k, t) in r.tiers.iter().enumerate() {
      let window_s = ((t.band_max_s - t.band_min_s) / 256.0).max(1.0);
      if since <= 0.0 && t.live > 0 {
        check_failures.push(format!(
          "frame {}: tier {k} holds {} clusters {:.2} d before the ignition",
          r.index,
          t.live,
          -since / 86_400.0
        ));
      } else if t.live > 0 && t.oldest_s > since + window_s {
        check_failures.push(format!(
          "frame {}: tier {k} oldest dust {:.2} d > {:.2} d since the ignition",
          r.index,
          t.oldest_s / 86_400.0,
          since / 86_400.0
        ));
      }
    }
  }
  for r in &records {
    if r.nucleus_rgb.0 <= 8 && r.nucleus_rgb.1 <= 8 && r.nucleus_rgb.2 <= 8 {
      check_failures.push(format!("frame {}: nucleus pixel is black", r.index));
    }
  }
  for p in &problems {
    check_failures.push(p.clone());
  }
  // ───── shape evolution (segmentation of the pyramid dumps, the reference analysis) ─────
  // frames at or before the ignition hold no dust: the shape analysis starts after it
  let shape_records: Vec<FrameRec> = records
    .iter()
    .filter(|r| r.since_ignition_s.is_none_or(|s| s > 0.0))
    .cloned()
    .collect();
  match shape_analysis(&args, &shape_records, cfg0.beta_ref as f64) {
    Ok(mut f) => check_failures.append(&mut f),
    Err(e) => log_line(&format!("shape analysis skipped: {e}")),
  }

  // ───── summary ─────
  {
    let mut md = String::new();
    md.push_str("# dust_observer run\n\n");
    md.push_str(&format!(
      "- start: `{start}`  scene range end: `{end}`\n- frames: {n_frames} every {} h ({} days)\n- size: {}x{}  pose: {:?}\n- play mode: {:?}  reemit-every: {}\n- SPK: `{}`\n- CPU particles: {}\n- jet: lat 20°, lon 0°, aperture 0.5 rad, v 2.0±0.5 m/s, d 100 µm, ρ 0.533 g/cm³, Afρ₀ 100 cm, power 2, cutoff 15 AU, nucleus 2 km, seed 7, colour (1.0, 0.6, 0.2)\n- rotation: pole RA 0°, Dec 90°, period {ROTATION_PERIOD_H} h, body-fixed\n- ignition: {}\n- view: dust flags 2 (flow on, tracers off), flow speed 1.0, softening 1e-2\n- framing: half-height {half_height_km:.1} km = {half_height_au:.6} AU (3·coma {:.1} km vs β·g·T² {:.1} km, β_ref {:.4}, r₀ {r0_au:.4} AU), camera {:.1} km above the comet along +Z\n- Vulkan validation errors: {vk_errors}\n- wall time: {:.1} s\n\n",
      args.step_hours,
      args.days,
      args.width,
      args.height,
      args.pose,
      args.play_seconds,
      args.reemit_every,
      spk.display(),
      std::env::var("AETHERVK_PARTICLES_CPU").as_deref() == Ok("1"),
      ignition.map_or(String::from("pre-existing tail (--prestart)"), |t| format!("{t} (start + {:.2} d)", args.ignite_days)),
      3.0 * coma0_km,
      tail_km,
      cfg0.beta_ref,
      cam_height_au * AU_KM,
      t_run.elapsed().as_secs_f64()
    ));
    md.push_str("| # | epoch | r (AU) | tiers live/cap (age d) | coma km | nucleus px | nucleus RGB | mean L | p99 L | non-black | seek |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
    for r in &records {
      let tiers = r
        .tiers
        .iter()
        .map(|t| {
          format!(
            "{}/{} ({:.1}..{:.1}){}",
            t.live,
            t.capacity,
            t.youngest_s / 86_400.0,
            t.oldest_s / 86_400.0,
            if t.caught_up { "" } else { " BUILDING" }
          )
        })
        .collect::<Vec<_>>()
        .join("; ");
      md.push_str(&format!(
        "| {} | {} | {:.4} | {} | {:.0} | ({:.0}, {:.0}) | ({}, {}, {}) | {:.3} | {:.1} | {:.4} | {}{} |\n",
        r.index,
        r.epoch,
        r.r_helio_au,
        tiers,
        r.coma_km,
        r.nucleus_px.0,
        r.nucleus_px.1,
        r.nucleus_rgb.0,
        r.nucleus_rgb.1,
        r.nucleus_rgb.2,
        r.mean_lum,
        r.p99_lum,
        r.nonblack_frac,
        r.seek_ok,
        r.mc
          .as_ref()
          .map(|m| format!(" | mc rms {:.3} judged {:.3} (ripple {:.3} vs {:.3}, {} px)", m.rel_rms, m.rel_rms_judged, m.ripple_render, m.ripple_truth, m.region_px))
          .unwrap_or_default()
      ));
    }
    md.push_str("\n## checks\n\n");
    if check_failures.is_empty() {
      md.push_str("all checks passed\n");
    } else {
      for f in &check_failures {
        md.push_str(&format!("- {f}\n"));
      }
    }
    std::fs::write(args.out.join("summary.md"), md).expect("summary.md");
  }

  let exit_code = if args.check && !check_failures.is_empty() {
    1
  } else {
    0
  };
  log_line(&format!(
    "done: {} frames in {:.1} s, {} check failure(s), exit {exit_code}",
    n_frames,
    t_run.elapsed().as_secs_f64(),
    check_failures.len()
  ));
  for f in &check_failures {
    log_line(&format!("  - {f}"));
  }

  // teardown. Default: flush and exit without dropping the context (the GPU teardown of the
  // render thread has wedged/segfaulted at exit on this machine, which would hide the exit code of
  // `--check`; the outputs are complete at this point). `--full-teardown` follows the FFI
  // `avkSimulationContext_shutdownSync` recipe under a 20 s watchdog.
  if let Ok(mut g) = LOG_FILE.lock() {
    if let Some(f) = g.as_mut() {
      let _ = f.flush();
    }
  }
  if args.full_teardown {
    SimulationContext::set_render_callback(None);
    let _ = ctx.destroy_presentation_engine(scene_id, pe);
    let _ = ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown);
    let done = std::sync::Arc::new(AtomicBool::new(false));
    let done2 = done.clone();
    std::thread::spawn(move || {
      std::thread::sleep(std::time::Duration::from_secs(20));
      if !done2.load(Ordering::Acquire) {
        log_line("teardown watchdog: shutdown did not finish in 20 s, exiting anyway");
        unsafe { libc::_exit(exit_code) }
      }
    });
    drop(ctx);
    done.store(true, Ordering::Release);
  } else {
    std::mem::forget(ctx);
  }
  // `_exit`: no atexit handlers / static destructors while the render thread may still be
  // running (a plain `exit` segfaulted after a device loss); everything is flushed above.
  unsafe { libc::_exit(exit_code) }
}

/// Segments every frame's dust (from `frame_NNN.pyr`, τ per px² through the reference
/// `composite_sample`) with one threshold (what the last frame displays at 10 %), measures the
/// region's growth step by step (`dust::shape_step`) and writes `shape.jsonl` + `shape.png`.
/// Growth is progressive when nothing appears beyond the reach of the previous region (the
/// fastest grains over the step plus the splat footprint). Returns the check failures.
/// High-pass ripple of a τ image: `(τ − box₉(τ)) / box₉(τ)` over the pixels whose 9×9 box mean is
/// above the median of the non-empty box means; returns `(rms, p90 of |·|)`. A smooth cloud
/// gives ≈ 0; a sampling lattice (speed shells, chords between strata) gives 0.3–0.5
/// (`the_problem.rdc`: 0.35 / 0.50).
fn ripple_metric(img: &[f32], w: usize, h: usize) -> (f64, f64, usize) {
  let mut integ = vec![0.0f64; (w + 1) * (h + 1)];
  for y in 0..h {
    let mut row = 0.0f64;
    for x in 0..w {
      row += img[y * w + x] as f64;
      integ[(y + 1) * (w + 1) + (x + 1)] = integ[y * (w + 1) + (x + 1)] + row;
    }
  }
  let box_mean = |x: usize, y: usize| -> f64 {
    let r = 4usize;
    let (x0, y0) = (x.saturating_sub(r), y.saturating_sub(r));
    let (x1, y1) = ((x + r + 1).min(w), (y + r + 1).min(h));
    let s = integ[y1 * (w + 1) + x1] - integ[y0 * (w + 1) + x1] - integ[y1 * (w + 1) + x0]
      + integ[y0 * (w + 1) + x0];
    s / ((x1 - x0) * (y1 - y0)) as f64
  };
  let mut blur = vec![0.0f64; w * h];
  let mut nonzero: Vec<f64> = Vec::new();
  for y in 0..h {
    for x in 0..w {
      let b = box_mean(x, y);
      blur[y * w + x] = b;
      if b > 0.0 {
        nonzero.push(b);
      }
    }
  }
  if nonzero.is_empty() {
    return (0.0, 0.0, 0);
  }
  nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median = nonzero[nonzero.len() / 2];
  let mut hp: Vec<f64> = Vec::new();
  for i in 0..w * h {
    if blur[i] > median {
      hp.push((img[i] as f64 - blur[i]) / blur[i]);
    }
  }
  if hp.is_empty() {
    return (0.0, 0.0, 0);
  }
  let rms = (hp.iter().map(|v| v * v).sum::<f64>() / hp.len() as f64).sqrt();
  let mut abs: Vec<f64> = hp.iter().map(|v| v.abs()).collect();
  abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let p90 = abs[((abs.len() - 1) as f64 * 0.9) as usize];
  (rms, p90, hp.len())
}

fn shape_analysis(args: &Args, records: &[FrameRec], beta_ref: f64) -> Result<Vec<String>, String> {
  use aethervk_core_rlib::scene::dust as d;
  let (w, h) = (args.width, args.height);
  let layout = d::PyramidLayout::new(w, h);
  let mut frames: Vec<(usize, Vec<f32>, f32, f32)> = Vec::new();
  // `frames` keeps the record's position (the records may be a filtered subset: frames at or
  // before the ignition are left out), not the frame index
  for (pos, r) in records.iter().enumerate() {
    let path = args.out.join(format!("frame_{:03}.pyr", r.index));
    let Ok(bytes) = std::fs::read(&path) else {
      continue;
    };
    let words: Vec<u32> = bytes
      .chunks_exact(4)
      .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
      .collect();
    if words.len() < layout.total_words as usize
      || words[d::PYR_LEVELS as usize] != layout.level_count()
    {
      continue;
    }
    let tau_max = f32::from_bits(words[d::PYR_TAU_MAX as usize]);
    // the unit the frame's counts were written with (header word `PYR_UNIT`, host-written at
    // frame begin); older dumps without it: τ_max of the frame (approximate across seeks)
    let header_unit = f32::from_bits(words[d::PYR_UNIT as usize]);
    let unit = if header_unit > 0.0 && header_unit.is_finite() {
      header_unit
    } else if tau_max > 0.0 {
      tau_max * d::WHITE_TILE_UNIT_REL
    } else {
      1e-4
    };
    let (mo, mw, mh) = layout.measure;
    let (white, p50) = d::white_point_from_level(
      &words[mo as usize..mo as usize + (mw * mh * d::PYRAMID_TEXEL_WORDS) as usize],
      1 << d::DUST_WHITE_LEVEL,
      unit,
    )
    .unwrap_or((0.0, 0.0));
    let mut img = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
      for x in 0..w {
        img[(y * w + x) as usize] = d::composite_sample(&layout, &words, unit, x, y).0;
      }
    }
    if !img.iter().any(|&v| v > 0.0) {
      // the first frame's pyramid readback lags its draw: an empty dump is no frame
      log_line(&format!(
        "shape frame {}: empty pyramid dump, skipped",
        r.index
      ));
      continue;
    }
    frames.push((pos, img, white, d::auto_softening(p50, white)));
  }
  if frames.len() < 2 {
    return Err(format!("{} frames with a pyramid dump", frames.len()));
  }
  let (_, _, white, soft) = frames.last().unwrap();
  let threshold = d::display_threshold(*white, *soft, 0.1);
  if !(threshold > 0.0) {
    return Err("no white point measured".into());
  }
  // reach: the fastest grains over one step, in pixels, plus the splat footprint: the ejection
  // (v_mean + 3σ + lateral) and the radiation-pressure drift of the oldest live dust
  // (β_ref · g · age: an old trail cluster moves hundreds of m/s relative to the nucleus)
  let v_eject = 2.0 + 3.0 * 0.5 + 3.0 * 2.0 * 0.5; // m/s, the jet of this scene
  let dt_s = args.step_hours * 3600.0;
  let mut shapes = Vec::new();
  let mut failures = Vec::new();
  let mut out = std::fs::File::create(args.out.join("shape.jsonl")).map_err(|e| e.to_string())?;
  for (k, (pos, img, white_k, soft_k)) in frames.iter().enumerate() {
    let r = &records[*pos];
    let index = r.index;
    let m_per_px = 2.0 * r.half_height_au * d::AU_M / h as f64;
    let g_ms2 = 1.327_124_400_18e20 / (r.r_helio_au * d::AU_M).powi(2);
    let age_max_s = r.tiers.iter().map(|t| t.oldest_s).fold(0.0, f64::max);
    // the smallest grains of the size range (s_ref / SIZE_RANGE_FACTOR) carry
    // SIZE_RANGE_FACTOR × β_ref: they lead the tail's tip
    let v_max = v_eject + beta_ref * d::SIZE_RANGE_FACTOR * g_ms2 * age_max_s;
    let reach = (v_max * dt_s / m_per_px) as f32 + 3.0;
    let nucleus = [r.nucleus_px.0 as f32, r.nucleus_px.1 as f32];
    let sh = d::segment_shape(img, w, h, nucleus, threshold);
    // the lattice detector: high-pass residual of the τ image over the dust region; a cloud of
    // a few hundred pixels is all edge, the metric only means something once the region is
    // extended (RIPPLE_MIN_PX)
    let ripple = ripple_metric(img, w as usize, h as usize);
    const RIPPLE_MIN_PX: usize = 2000;
    // and the 9-px box must be small against the cloud: on a compact coma (radius p90 below
    // three boxes) the high-pass measures the 1/r curvature, not a lattice
    const RIPPLE_MIN_RADIUS_PX: f32 = 27.0;
    let ripple_judged = ripple.2 >= RIPPLE_MIN_PX && sh.radius_p90 >= RIPPLE_MIN_RADIUS_PX;
    // with a Monte-Carlo truth the ripple is judged as the excess over the truth's own ripple
    // (over the truth's region); without one, as the absolute high-pass rms
    let mc = r.mc.as_ref();
    let (ripple_mode, ripple_value, ripple_limit) = match mc {
      Some(m) => (
        "excess over the truth",
        m.ripple_render - m.ripple_truth,
        args.ripple_max.unwrap_or(RIPPLE_MAX_EXCESS_DEFAULT),
      ),
      None => (
        "absolute: no truth",
        ripple.0,
        args.ripple_max.unwrap_or(RIPPLE_MAX_ABSOLUTE_DEFAULT),
      ),
    };
    let ripple_judged = ripple_judged || mc.is_some_and(|m| m.region_px >= RIPPLE_MIN_PX);
    if ripple_judged && ripple_value > ripple_limit {
      failures.push(format!(
        "frame {index}: ripple {ripple_value:.3} ({ripple_mode}) above {ripple_limit:.3}: the picture shows a sampling lattice"
      ));
    }
    // the faint region of the previous frame (a quarter of the threshold): dust brightening
    // across the threshold was already there
    let faint_prev = if k > 0 {
      Some(d::segment_shape(
        &frames[k - 1].1,
        w,
        h,
        nucleus,
        0.25 * threshold,
      ))
    } else {
      None
    };
    let step = if k > 0 {
      Some(d::shape_step(&shapes[k - 1], &sh, reach))
    } else {
      None
    };
    let reach_step = faint_prev.as_ref().map(|f| d::shape_step(f, &sh, reach));
    let line = format!(
      "{{\"index\":{},\"epoch\":{:?},\"threshold\":{:.4e},\"white\":{:.4e},\"softening\":{:.4e},\"area_px\":{},\"detached_px\":{},\"total_tau\":{:.4e},\"radius_p50_px\":{:.2},\"radius_p90_px\":{:.2},\"radius_max_px\":{:.2},\"radius_p90_km\":{:.1},\"axis\":[{:.4},{:.4}],\"elongation\":{:.3},\"centroid\":[{:.1},{:.1}],\"reach_px\":{:.2},\"ripple_rms\":{:.4},\"ripple_p90\":{:.4}{}}}",
      index,
      r.epoch,
      threshold,
      white_k,
      soft_k,
      sh.area_px,
      sh.detached_px,
      sh.total_tau,
      sh.radius_p50,
      sh.radius_p90,
      sh.radius_max,
      sh.radius_p90 as f64 * m_per_px * 1e-3,
      sh.axis[0],
      sh.axis[1],
      sh.elongation,
      sh.centroid[0],
      sh.centroid[1],
      reach,
      ripple.0,
      ripple.1,
      match &step {
        Some(st) => format!(
          ",\"grown_px\":{},\"lost_px\":{},\"grown_outside_reach\":{},\"farthest_growth_px\":{:.2},\"area_ratio\":{:.4},\"radius_p90_ratio\":{:.4},\"tau_ratio\":{:.4}",
          st.grown_px,
          st.lost_px,
          reach_step.map(|r| r.grown_outside_reach).unwrap_or(0),
          reach_step.map(|r| r.farthest_growth_px).unwrap_or(0.0),
          st.area_ratio,
          st.radius_p90_ratio,
          st.tau_ratio
        ),
        None => String::new(),
      }
    );
    writeln!(out, "{line}").map_err(|e| e.to_string())?;
    if let (Some(st), Some(rs)) = (&step, &reach_step) {
      let prev = &shapes[k - 1];
      if rs.grown_outside_reach > 0 {
        failures.push(format!(
          "frame {index}: {} px of dust appeared beyond {reach:.1} px of the previous (faint) region (farthest {:.1} px): not progressive",
          rs.grown_outside_reach, rs.farthest_growth_px
        ));
      }
      if sh.radius_p90 - prev.radius_p90 > reach {
        failures.push(format!(
          "frame {index}: radius p90 jumped {:.1} → {:.1} px (reach {reach:.1})",
          prev.radius_p90, sh.radius_p90
        ));
      }
      // bounded relative growth: a young coma's radius grows linearly with the time since the
      // ignition, so its area may grow as (t_b / t_a)² between frames, never faster (the unit
      // test's bound); a steady tail stays within [0.8, 1.5]
      let prev_r = &records[frames[k - 1].0];
      let area_bound = match (prev_r.since_ignition_s, r.since_ignition_s) {
        (Some(ta), Some(tb)) if ta > 0.0 && tb > ta => (1.15 * (tb / ta).powi(2) + 0.1) as f32,
        _ => 1.5,
      }
      .max(1.5);
      // 16 px of slack: a coma of a hundred pixels is quantised by its threshold contour
      if prev.area_px >= 64
        && (st.area_ratio < 0.8 || sh.area_px as f32 > area_bound * prev.area_px as f32 + 16.0)
      {
        failures.push(format!(
          "frame {index}: area ratio {:.2} ({} → {} px, bound {area_bound:.2})",
          st.area_ratio, prev.area_px, sh.area_px
        ));
      }
      if sh.detached_px as f32 > 0.05 * sh.area_px as f32 + 4.0 {
        failures.push(format!(
          "frame {index}: {} detached px of {}",
          sh.detached_px, sh.area_px
        ));
      }
    }
    log_line(&format!(
      "shape frame {index}: area {} px, p90 {:.1} px ({:.0} km), elongation {:.2}{}",
      sh.area_px,
      sh.radius_p90,
      sh.radius_p90 as f64 * m_per_px * 1e-3,
      sh.elongation,
      step
        .zip(reach_step)
        .map(|(st, rs)| format!(
          ", grown {} px, beyond reach {} (farthest {:.1} px)",
          st.grown_px, rs.grown_outside_reach, rs.farthest_growth_px
        ))
        .unwrap_or_default()
    ));
    log_line(&format!(
      "ripple frame {index}: rms {:.3} p90 {:.3} over {} px; judged {ripple_value:.3} ({ripple_mode}, max {ripple_limit:.3}{})",
      ripple.0,
      ripple.1,
      ripple.2,
      if !ripple_judged {
        ", region too small to judge"
      } else {
        ""
      }
    ));
    shapes.push(sh);
  }
  // shape.png: radius p90, area and elongation over the frames (each normalised to its max)
  let (pw, ph) = (800u32, 300u32);
  let mut plot = image::RgbaImage::from_pixel(pw, ph, image::Rgba([16, 16, 20, 255]));
  let series: [(Vec<f32>, [u8; 3]); 3] = [
    (
      shapes.iter().map(|s| s.radius_p90).collect(),
      [255, 170, 40],
    ),
    (
      shapes.iter().map(|s| s.area_px as f32).collect(),
      [80, 200, 255],
    ),
    (
      shapes.iter().map(|s| s.elongation).collect(),
      [120, 255, 120],
    ),
  ];
  for (vals, col) in &series {
    let max = vals.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
    let n = vals.len().max(2) as f32;
    let pt = |i: usize| {
      (
        10.0 + (pw as f32 - 20.0) * i as f32 / (n - 1.0),
        ph as f32 - 10.0 - (ph as f32 - 20.0) * vals[i] / max,
      )
    };
    for i in 1..vals.len() {
      let (x0, y0) = pt(i - 1);
      let (x1, y1) = pt(i);
      let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as usize).max(1);
      for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        if x >= 0.0 && y >= 0.0 && (x as u32) < pw && (y as u32) < ph {
          plot.put_pixel(
            x as u32,
            y as u32,
            image::Rgba([col[0], col[1], col[2], 255]),
          );
        }
      }
    }
  }
  plot.save(args.out.join("shape.png")).map_err(|e| e.to_string())?;
  log_line("shape.png: orange radius p90, blue area, green elongation (each over its maximum)");
  Ok(failures)
}

// ───────────────────────────── Monte-Carlo truth (`--mc`) ─────────────────────────────

/// the truth's region: pixels above this fraction of its peak
const MC_REGION_REL: f64 = 1e-4;
/// a core (pixels above 1 % of the truth's peak) smaller than this is an unresolved point (a
/// sub-pixel coma the 1-px truth bins cannot place: at 1 AU it holds 99 % of the squared error
/// in 3 000 px) and is left out of the judged rms
const MC_CORE_RESOLVED_PX: usize = 5000;
/// `--ripple-max` defaults: the excess over the truth's ripple, and the absolute ripple
const RIPPLE_MAX_EXCESS_DEFAULT: f64 = 0.05;
const RIPPLE_MAX_ABSOLUTE_DEFAULT: f64 = 0.25;

/// The frame's geometry for [`monte_carlo_truth`]: the observer's own projection (the nucleus
/// pixel code), so the truth and the render share the camera exactly.
struct McView {
  width: u32,
  height: u32,
  cam_pos: V3,
  cam_q: Q,
  half_height_au: f64,
  aspect: f64,
  /// `Some(tan(fov/2))` for the perspective Earth pose
  perspective_tan_h: Option<f64>,
  comet_au: V3,
  nucleus_px: (f64, f64),
  /// the truth holds this tier only (`--mc-tier`)
  mc_tier: Option<u32>,
  /// the truth holds the dust younger than this only (`--mc-age-max`, s)
  mc_age_max: Option<f64>,
  /// nucleus radius on screen (px): the truth's peak is the jet site, up to this far from the
  /// nucleus pixel
  nucleus_px_radius: f64,
}

fn mc_qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
  [
    a[3] * b[0] + a[0] * b[3] + (a[1] * b[2] - a[2] * b[1]),
    a[3] * b[1] + a[1] * b[3] + (a[2] * b[0] - a[0] * b[2]),
    a[3] * b[2] + a[2] * b[3] + (a[0] * b[1] - a[1] * b[0]),
    a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]),
  ]
}
fn mc_qrot(q: [f64; 4], v: V3) -> V3 {
  let u = [q[0], q[1], q[2]];
  let t = [
    2.0 * (u[1] * v[2] - u[2] * v[1]),
    2.0 * (u[2] * v[0] - u[0] * v[2]),
    2.0 * (u[0] * v[1] - u[1] * v[0]),
  ];
  [
    v[0] + q[3] * t[0] + (u[1] * t[2] - u[2] * t[1]),
    v[1] + q[3] * t[1] + (u[2] * t[0] - u[0] * t[2]),
    v[2] + q[3] * t[2] + (u[0] * t[1] - u[1] * t[0]),
  ]
}
fn mc_qaxis_angle(axis: V3, angle: f64) -> [f64; 4] {
  let (s, c) = (0.5 * angle).sin_cos();
  [axis[0] * s, axis[1] * s, axis[2] * s, c]
}
/// uniform direction in the cone of half aperture `aperture` around unit `dir` (`dust::sample_cone`)
fn mc_sample_cone(u1: f64, u2: f64, dir: V3, aperture: f64) -> V3 {
  let z = 1.0 + (aperture.cos() - 1.0) * u2;
  let sin_t = (1.0 - z * z).max(0.0).sqrt();
  let phi = 2.0 * std::f64::consts::PI * u1;
  let local = [sin_t * phi.cos(), sin_t * phi.sin(), z];
  let up = if dir[2].abs() < 0.999 {
    [0.0, 0.0, 1.0]
  } else {
    [1.0, 0.0, 0.0]
  };
  let t = normalize(cross(up, dir));
  let b = cross(dir, t);
  [
    t[0] * local[0] + b[0] * local[1] + dir[0] * local[2],
    t[1] * local[0] + b[1] * local[1] + dir[1] * local[2],
    t[2] * local[0] + b[2] * local[1] + dir[2] * local[2],
  ]
}
fn mc_rng(state: &mut u64) -> f64 {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  (*state >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}
fn mc_gauss(state: &mut u64) -> f64 {
  let u1 = mc_rng(state).max(1e-300);
  let u2 = mc_rng(state);
  (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// The τ image (per px, the levels composited) of a pyramid dump, as `shape_analysis` builds it
fn tau_image_of_pyramid(words: &[u32], w: u32, h: u32) -> Option<Vec<f32>> {
  use aethervk_core_rlib::scene::dust as d;
  let layout = d::PyramidLayout::new(w, h);
  if words.len() < layout.total_words as usize
    || words[d::PYR_LEVELS as usize] != layout.level_count()
  {
    return None;
  }
  let tau_max = f32::from_bits(words[d::PYR_TAU_MAX as usize]);
  let header_unit = f32::from_bits(words[d::PYR_UNIT as usize]);
  let unit = if header_unit > 0.0 && header_unit.is_finite() {
    header_unit
  } else if tau_max > 0.0 {
    tau_max * d::WHITE_TILE_UNIT_REL
  } else {
    1e-4
  };
  let mut img = vec![0.0f32; (w * h) as usize];
  for y in 0..h {
    for x in 0..w {
      img[(y * w + x) as usize] = d::composite_sample(&layout, words, unit, x, y).0;
    }
  }
  Some(img)
}

fn mc_box(img: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
  // separable box of radius r, edge-clamped by renormalisation
  let mut tmp = vec![0.0f64; w * h];
  let mut out = vec![0.0f64; w * h];
  for y in 0..h {
    for x in 0..w {
      let (x0, x1) = (x.saturating_sub(r), (x + r).min(w - 1));
      let mut s = 0.0;
      for xx in x0..=x1 {
        s += img[y * w + xx];
      }
      tmp[y * w + x] = s / (x1 - x0 + 1) as f64;
    }
  }
  for y in 0..h {
    let (y0, y1) = (y.saturating_sub(r), (y + r).min(h - 1));
    for x in 0..w {
      let mut s = 0.0;
      for yy in y0..=y1 {
        s += tmp[yy * w + x];
      }
      out[y * w + x] = s / (y1 - y0 + 1) as f64;
    }
  }
  out
}

/// **The frame against the continuous emission.** Every live batch descriptor of the scene's
/// dust (what the emit shader reads: the comet state from the SPK at the window start, the lit
/// arcs, the spin, the cone, the size and speed laws, the mass) seeds `grains` grains drawn from
/// the *continuous* distribution the clusters stand for — emission time uniform over the lit
/// time, direction uniform over the cone at the attitude of that instant, size by cross-section
/// (`n(s) ∝ s^-q`, weight `s²`), speed `v_ref √(β/β_ref) (1 + σ_rel g)`, the site offset and its
/// `ω × r` — each propagated in f64 with its own `μ(1 − β)` to the frame's time, gated by its
/// tier's age band, and binned with the batch mass into the observer's own projection. The
/// renderer's τ image (the pyramid dump) and that histogram, both unit-sum and 5×5 box filtered,
/// are compared where the truth is above 1 % of its peak. Returns the record and both images.
fn monte_carlo_truth(
  ctx: &SimulationContext,
  scene_id: u64,
  pyr: &[u32],
  view: &McView,
  grains: usize,
) -> Option<(McRec, Vec<f32>, Vec<f32>)> {
  use aethervk_core_rlib::scene::dust as d;
  let (w, h) = (view.width as usize, view.height as usize);
  let render = tau_image_of_pyramid(pyr, view.width, view.height)?;
  if !render.iter().any(|&v| v > 0.0) {
    // the pyramid of this frame is empty (the first frame's readback lags the draw): nothing
    // to compare, the next frame will
    return None;
  }
  // the batches, each with its tier's age band, and the jet state now
  struct B {
    desc: d::DustBatch,
    band: (f64, f64),
    frame: Option<d::DustFrame>,
    /// the oldest tier fades its dust over the last 10 % of the band (`dust_propagate.comp`:
    /// `clamp((1 − age/ttl)·10, 0, 1)` when `ttl.z` = 0); the younger tiers hand over sharply
    fade: bool,
    tier: u32,
  }
  let mut batches: Vec<B> = Vec::new();
  // (t_now, the nucleus centre's heliocentric position now): the jet state's `r_m` is the
  // site's, the body centre sits one rotated site offset behind it, and the frame's `comet_au`
  // is the centre (the 2-km offset is 120 px at the 5-km frame)
  let mut jet_now: Option<(f64, V3)> = None;
  {
    let scene_arc = ctx.get_scene(scene_id)?;
    let g = scene_arc.read();
    g.scene.query1(|_, ps: &ParticleSystemComponent| {
      let sys = ps.dust.lock();
      for t in &sys.tiers {
        if let Some(j) = t.jet {
          // the render's anchor is the jet state's `r_m`, and the frame is centred where the
          // render puts the dust's apex: the same point (checked at 5 km: subtracting the site
          // offset moved the truth's apex 140 px off the render's)
          jet_now = Some((j.t_s, j.r_m));
        }
        let band = t.age_band_s(t.ttl_s);
        let frame = t.jet.and_then(|j| t.draw_state().map(|ds| ds.frame.at_time(j.t_s)));
        let fade = frame.as_ref().is_some_and(|f| f.ttl[2] < 0.5);
        for b in &t.ring.batches {
          batches.push(B {
            desc: b.desc,
            band,
            frame,
            fade,
            tier: t.tier,
          });
        }
      }
    });
  }
  let (t_now, r_jet_now) = jet_now?;
  if let Some(k) = view.mc_tier {
    batches.retain(|b| b.tier == k);
  }
  if let Some(a) = view.mc_age_max {
    // the render caps every tier's band there (`AETHERVK_DUST_DEBUG_AGE_MAX`): the same gate
    for b in &mut batches {
      b.band.1 = b.band.1.min(a);
    }
    batches.retain(|b| b.band.0 < b.band.1);
  }
  if batches.is_empty() {
    return None;
  }
  // OBSERVER_MC_BATCHES=<file>: the batch table (tier band min, t_start, dur, count, mass g, lit
  // fraction, break flag, window seed) for offline correlation with the residual
  if let Ok(path) = std::env::var("OBSERVER_MC_BATCHES") {
    let mut out =
      String::from("band_min_s,t_start_s,dur_s,count,mass_g,lit_fraction,break_before,seed\n");
    for b in &batches {
      let d = &b.desc;
      let dur = d.comet_v_dur_hi[3] as f64 + d.comet_v_dur_lo[3] as f64;
      let lit_frac = if d.lit[3] == d::LIT_MODE_PERIODIC && d.spin[3] > 0.0 {
        d.lit[2] as f64 / (d.spin[3] as f64 * dur)
      } else {
        1.0
      };
      out.push_str(&format!(
        "{},{},{},{},{},{:.4},{},{}\n",
        b.band.0,
        d.comet_r_t_hi[3] as f64 + d.comet_r_t_lo[3] as f64,
        dur,
        d.count,
        d.mass_params[0],
        lit_frac,
        d::batch_streams(d).1 as u8,
        d.seed
      ));
    }
    let _ = std::fs::write(path, out);
  }
  // diagnostics: how the renderer's clusters share the batch masses (`emit_cluster` is
  // deterministic from the descriptor): per cluster (t0, age, mass, stream, sample) to
  // `mc_clusters.f32`, and the concentration of the mass over the clusters in the log
  {
    let mut rows: Vec<f32> = Vec::new();
    let mut masses: Vec<f64> = Vec::new();
    let mut worst_mismatch = 0.0f64;
    for b in &batches {
      let desc = &b.desc;
      let (shift, _) = d::batch_streams(desc);
      let mut sum = 0.0f64;
      for j in 0..desc.count {
        let c = d::emit_cluster(desc, j);
        let t0 = c.r0_t0_hi[3] as f64 + c.r0_t0_lo[3] as f64;
        let m = c.v0_lo_mass[3] as f64;
        sum += m;
        masses.push(m);
        rows.extend_from_slice(&[
          t0 as f32,
          (t_now - t0) as f32,
          m as f32,
          (j & ((1 << shift) - 1)) as f32,
          (j >> shift) as f32,
          b.band.0 as f32,
        ]);
      }
      let mb = desc.mass_params[0] as f64;
      if mb > 0.0 {
        worst_mismatch = worst_mismatch.max((sum / mb - 1.0).abs());
      }
    }
    // the moments stage on a sample of the oldest tier's clusters: the chord length of the size
    // polyline and the dispersion across it, in pixels (the kernel the splat is handed)
    {
      let m_per_px = 2.0 * view.half_height_au * d::AU_M / view.height as f64;
      let (mut chords, mut s1s, mut s2s, mut expect) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
      // the time axis: how many clusters have no stream predecessor (a break), and the time
      // segment to the predecessor's same bin (bins 0 and 8) in pixels, with its part across
      // the chord (the lateral distance between the two polylines there)
      let (mut breaks, mut seen) = (0usize, 0usize);
      let (mut seg0, mut seg8, mut lat0, mut lat8) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
      for b in batches.iter().filter(|b| b.band.0 >= 20_736_000.0) {
        let Some(frame) = b.frame.as_ref() else {
          continue;
        };
        let n = 1u32 << d::batch_streams(&b.desc).0;
        for j in (0..b.desc.count).step_by(64) {
          seen += 1;
          if d::stream_breaks_before(&b.desc, j) {
            breaks += 1;
          }
          let c = d::emit_cluster(&b.desc, j);
          let (_, m) = d::packet_moments(&c, j, frame);
          if !(m.flux() > 0.0) {
            continue;
          }
          if j >= n {
            let cp = d::emit_cluster(&b.desc, j - n);
            let (_, mp) = d::packet_moments(&cp, j - n, frame);
            if mp.flux() > 0.0 {
              for (bin, segs, lats) in [
                (0usize, &mut seg0, &mut lat0),
                (8usize, &mut seg8, &mut lat8),
              ] {
                let mid = |q: &d::DustMoments| {
                  let (a, c) = (q.edge(bin), q.edge(bin + 1));
                  [
                    0.5 * (a[0] + c[0]),
                    0.5 * (a[1] + c[1]),
                    0.5 * (a[2] + c[2]),
                  ]
                };
                let (o, pm) = (mid(&m), mid(&mp));
                let st = [pm[0] - o[0], pm[1] - o[1], pm[2] - o[2]];
                let st2 = st[0] * st[0] + st[1] * st[1] + st[2] * st[2];
                segs.push(st2.sqrt() as f64 / m_per_px);
                let (a, c) = (m.edge(bin), m.edge(bin + 1));
                let ch = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let cl = (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2]).sqrt().max(1e-30);
                let along = (st[0] * ch[0] + st[1] * ch[1] + st[2] * ch[2]) / cl;
                lats.push((st2 - along * along).max(0.0).sqrt() as f64 / m_per_px);
              }
            }
          }
          let (e0, e16) = (m.edge(0), m.edge(16));
          let ch = [e16[0] - e0[0], e16[1] - e0[1], e16[2] - e0[2]];
          let len = (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2]).sqrt();
          if !(len > 0.0) {
            continue;
          }
          let u = [ch[0] / len, ch[1] / len, ch[2] / len];
          let ax = if u[0].abs() < 0.9 {
            [1.0f32, 0.0, 0.0]
          } else {
            [0.0, 1.0, 0.0]
          };
          let w1 = {
            let v = [
              ax[1] * u[2] - ax[2] * u[1],
              ax[2] * u[0] - ax[0] * u[2],
              ax[0] * u[1] - ax[1] * u[0],
            ];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
          };
          let w2 = [
            u[1] * w1[2] - u[2] * w1[1],
            u[2] * w1[0] - u[0] * w1[2],
            u[0] * w1[1] - u[1] * w1[0],
          ];
          let cv = m.cov_v();
          let quad = |w: [f32; 3]| -> f32 {
            cv[0] * w[0] * w[0]
              + cv[3] * w[1] * w[1]
              + cv[5] * w[2] * w[2]
              + 2.0 * (cv[1] * w[0] * w[1] + cv[2] * w[0] * w[2] + cv[4] * w[1] * w[2])
          };
          chords.push(len as f64 / m_per_px);
          s1s.push((quad(w1).max(0.0).sqrt()) as f64 / m_per_px);
          s2s.push((quad(w2).max(0.0).sqrt()) as f64 / m_per_px);
          expect.push((c.sigma_lat() * m.age()) as f64 / m_per_px);
        }
      }
      let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(((v.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(f64::NAN)
      };
      if !seg0.is_empty() {
        log_line(&format!(
          "[observer] mc time axis (oldest tier): breaks {breaks} of {seen}; |seg_t| px bin0 p10/p50/p90 {:.2}/{:.2}/{:.2}, bin8 {:.2}/{:.2}/{:.2}; lateral part px bin0 {:.2}/{:.2}/{:.2}, bin8 {:.2}/{:.2}/{:.2}",
          pct(&mut seg0, 0.1),
          pct(&mut seg0, 0.5),
          pct(&mut seg0, 0.9),
          pct(&mut seg8, 0.1),
          pct(&mut seg8, 0.5),
          pct(&mut seg8, 0.9),
          pct(&mut lat0, 0.1),
          pct(&mut lat0, 0.5),
          pct(&mut lat0, 0.9),
          pct(&mut lat8, 0.1),
          pct(&mut lat8, 0.5),
          pct(&mut lat8, 0.9)
        ));
      }
      if !chords.is_empty() {
        log_line(&format!(
          "[observer] mc moments (oldest tier, {} clusters): chord px p10/p50/p90 {:.1}/{:.1}/{:.1}; across σ1 px {:.2}/{:.2}/{:.2}; σ2 px {:.2}/{:.2}/{:.2}; σ_lat·age px {:.2}/{:.2}/{:.2}",
          chords.len(),
          pct(&mut chords, 0.1),
          pct(&mut chords, 0.5),
          pct(&mut chords, 0.9),
          pct(&mut s1s, 0.1),
          pct(&mut s1s, 0.5),
          pct(&mut s1s, 0.9),
          pct(&mut s2s, 0.1),
          pct(&mut s2s, 0.5),
          pct(&mut s2s, 0.9),
          pct(&mut expect, 0.1),
          pct(&mut expect, 0.5),
          pct(&mut expect, 0.9)
        ));
      }
    }
    masses.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let total: f64 = masses.iter().sum();
    let top10: f64 = masses.iter().take(masses.len() / 10).sum();
    let zero = masses.iter().filter(|&&m| !(m > 0.0)).count();
    log_line(&format!(
      "[observer] mc clusters: {} clusters, top 10 % carry {:.1} % of the mass, {} with no mass, worst batch mass mismatch {:.2} %",
      masses.len(),
      100.0 * top10 / total.max(1e-300),
      zero,
      100.0 * worst_mismatch
    ));
    let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    let _ = std::fs::write(
      std::env::var("OBSERVER_MC_CLUSTERS").unwrap_or_else(|_| "/dev/null".into()),
      bytes,
    );
  }
  let mass_total: f64 = batches.iter().map(|b| b.desc.mass_params[0] as f64).sum();
  if !(mass_total > 0.0) {
    return None;
  }
  // the grains are allotted to the batches by mass × the fraction of a pilot pass that landed in
  // the frame (importance allocation: the weight per grain stays mass / n_b, so the estimate is
  // unbiased whatever n_b; a near frame holds a few windows of the youngest tier and would
  // otherwise spend 99 % of the grains out of view). The pilot is the same draw with 64 grains
  // per batch; its fraction has a floor so no batch is starved.

  let (right, forward, up) = camera_axes(view.cam_q);
  let project = |global_au: V3| -> Option<(usize, usize)> {
    let dd = sub(global_au, view.cam_pos);
    let (sx, sy) = (dot(dd, right), dot(dd, up));
    let (px, py) = match view.perspective_tan_h {
      Some(tan_h) => {
        let depth = dot(dd, forward);
        if depth <= 0.0 {
          return None;
        }
        (
          w as f64 / 2.0 * (1.0 + sx / depth / (tan_h * view.aspect)),
          h as f64 / 2.0 * (1.0 - sy / depth / tan_h),
        )
      }
      None => (
        w as f64 / 2.0 * (1.0 + sx / (view.half_height_au * view.aspect)),
        h as f64 / 2.0 * (1.0 - sy / view.half_height_au),
      ),
    };
    if px >= 0.0 && py >= 0.0 && px < w as f64 && py < h as f64 {
      Some((px as usize, py as usize))
    } else {
      None
    }
  };
  let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32);
  let chunk = batches.len().div_ceil(threads);
  let indexed: Vec<(usize, &B)> = batches.iter().enumerate().collect();
  // one threaded pass over the batches with `alloc(batch index)` grains each: the histogram,
  // the age-weighted histogram, the grains drawn and the grains in view per batch
  let run =
    |alloc: &(dyn Fn(usize) -> usize + Sync)| -> (Vec<f64>, Vec<f64>, usize, Vec<usize>, Vec<f64>) {
      let mut truth = vec![0.0f64; w * h];
      let mut age_sum = vec![0.0f64; w * h];
      let mut drawn = 0usize;
      let mut inview = vec![0usize; batches.len()];
      let mut counts = vec![0.0f64; w * h];
      std::thread::scope(|sc| {
        let handles: Vec<_> = indexed
          .chunks(chunk)
          .map(|part| {
            let project = &project;
            sc.spawn(move || {
              let mut hist = vec![0.0f64; w * h];
              let mut age_hist = vec![0.0f64; w * h];
              let mut cnt_hist = vec![0.0f64; w * h];
              let mut n_drawn = 0usize;
              let mut seen: Vec<(usize, usize)> = Vec::with_capacity(part.len());
              for &(bi, b) in part {
                let desc = &b.desc;
                let mass = desc.mass_params[0] as f64;
                let n_b = alloc(bi);
                let drawn_before = n_drawn;
                let wgt = mass / n_b as f64;
                let mut rng = (desc.seed as u64 ^ 0x9E37_79B9_7F4A_7C15).max(1);
                let rc0 = [
                  desc.comet_r_t_hi[0] as f64 + desc.comet_r_t_lo[0] as f64,
                  desc.comet_r_t_hi[1] as f64 + desc.comet_r_t_lo[1] as f64,
                  desc.comet_r_t_hi[2] as f64 + desc.comet_r_t_lo[2] as f64,
                ];
                let vc0 = [
                  desc.comet_v_dur_hi[0] as f64 + desc.comet_v_dur_lo[0] as f64,
                  desc.comet_v_dur_hi[1] as f64 + desc.comet_v_dur_lo[1] as f64,
                  desc.comet_v_dur_hi[2] as f64 + desc.comet_v_dur_lo[2] as f64,
                ];
                let t_start = desc.comet_r_t_hi[3] as f64 + desc.comet_r_t_lo[3] as f64;
                let dur = desc.comet_v_dur_hi[3] as f64 + desc.comet_v_dur_lo[3] as f64;
                let rot_start = [
                  desc.rot_start[0] as f64,
                  desc.rot_start[1] as f64,
                  desc.rot_start[2] as f64,
                  desc.rot_start[3] as f64,
                ];
                let axis = [
                  desc.spin[0] as f64,
                  desc.spin[1] as f64,
                  desc.spin[2] as f64,
                ];
                let omega = desc.spin[3] as f64;
                let jet_dir = [
                  desc.jet_dir_aperture[0] as f64,
                  desc.jet_dir_aperture[1] as f64,
                  desc.jet_dir_aperture[2] as f64,
                ];
                let aperture = desc.jet_dir_aperture[3] as f64;
                let (s_min, s_max, e) = (
                  desc.size_params[0] as f64,
                  desc.size_params[1] as f64,
                  desc.size_params[2] as f64,
                );
                let (v_ref, v_std_rel, s_ref, beta_s) = (
                  desc.vel_params[0] as f64,
                  desc.vel_params[1] as f64,
                  desc.vel_params[2] as f64,
                  desc.vel_params[3] as f64,
                );
                let beta_ref = beta_s / s_ref;
                let off = [
                  desc.site_offset[0] as f64,
                  desc.site_offset[1] as f64,
                  desc.site_offset[2] as f64,
                ];
                let o_start = mc_qrot(rot_start, off);
                // OBSERVER_MC_MODEL=chords (diagnostic): grains on the moments stage's size polyline
                // exactly as the splat draws it — a bin picked uniformly (equal cross-section), a
                // uniform position along its straight chord, the dispersion Gaussian of the bin's
                // speed factor across — but binned by their exact f64 projection, no splat: tells the
                // chord representation from the splat's drawing of it
                if std::env::var("OBSERVER_MC_MODEL").is_ok_and(|v| v == "chords") {
                  let Some(frame) = b.frame.as_ref() else {
                    continue;
                  };
                  let anchor = frame.anchor_m();
                  let per = (n_b / desc.count.max(1) as usize).max(4);
                  for j in 0..desc.count {
                    let c = d::emit_cluster(desc, j);
                    let (_, m) = d::packet_moments(&c, j, frame);
                    if !(m.flux() > 0.0) {
                      continue;
                    }
                    let cv = m.cov_v();
                    // Cholesky of Σ_v (xx, xy, xz, yy, yz, zz)
                    let l11 = (cv[0] as f64).max(0.0).sqrt();
                    let l21 = if l11 > 0.0 { cv[1] as f64 / l11 } else { 0.0 };
                    let l31 = if l11 > 0.0 { cv[2] as f64 / l11 } else { 0.0 };
                    let l22 = (cv[3] as f64 - l21 * l21).max(0.0).sqrt();
                    let l32 = if l22 > 0.0 {
                      (cv[4] as f64 - l31 * l21) / l22
                    } else {
                      0.0
                    };
                    let l33 = (cv[5] as f64 - l31 * l31 - l32 * l32).max(0.0).sqrt();
                    let m_c = c.v0_lo_mass[3] as f64;
                    let wgt_c = m_c / per as f64;
                    for _ in 0..per {
                      let bin = (mc_rng(&mut rng) * 16.0).floor().min(15.0) as usize;
                      let u = mc_rng(&mut rng);
                      let (a0, a1) = (m.edge(bin), m.edge(bin + 1));
                      let f = (m.edge_factor(bin) as f64 + m.edge_factor(bin + 1) as f64) * 0.5;
                      let (g1, g2, g3) =
                        (mc_gauss(&mut rng), mc_gauss(&mut rng), mc_gauss(&mut rng));
                      let dx = f * (l11 * g1);
                      let dy = f * (l21 * g1 + l22 * g2);
                      let dz = f * (l31 * g1 + l32 * g2 + l33 * g3);
                      let local = [
                        a0[0] as f64 + u * (a1[0] as f64 - a0[0] as f64) + dx,
                        a0[1] as f64 + u * (a1[1] as f64 - a0[1] as f64) + dy,
                        a0[2] as f64 + u * (a1[2] as f64 - a0[2] as f64) + dz,
                      ];
                      let global = [
                        view.comet_au[0] + (anchor[0] + local[0] - r_jet_now[0]) / d::AU_M,
                        view.comet_au[1] + (anchor[1] + local[1] - r_jet_now[1]) / d::AU_M,
                        view.comet_au[2] + (anchor[2] + local[2] - r_jet_now[2]) / d::AU_M,
                      ];
                      if let Some((x, y)) = project(global) {
                        hist[y * w + x] += wgt_c;
                        n_drawn += 1;
                      }
                    }
                  }
                  seen.push((bi, n_drawn - drawn_before));
                  continue;
                }
                // OBSERVER_MC_MODEL=clusters (diagnostic): grains from the renderer's own clusters and
                // kernels (`emit_cluster`: the mean velocity with the σ_lat / σ_rad Gaussian, the size
                // polyline uniform in √β over β/F..β·F, no time spread) instead of the continuous
                // emission: tells the kernels' width from the splat's drawing of them
                if std::env::var("OBSERVER_MC_MODEL").is_ok_and(|v| v == "clusters") {
                  let per = (n_b / desc.count.max(1) as usize).max(4);
                  for j in 0..desc.count {
                    let c = d::emit_cluster(desc, j);
                    let t0 = c.r0_t0_hi[3] as f64 + c.r0_t0_lo[3] as f64;
                    let age = t_now - t0;
                    if !(age >= b.band.0 && age < b.band.1) {
                      continue;
                    }
                    let r0 = [
                      c.r0_t0_hi[0] as f64 + c.r0_t0_lo[0] as f64,
                      c.r0_t0_hi[1] as f64 + c.r0_t0_lo[1] as f64,
                      c.r0_t0_hi[2] as f64 + c.r0_t0_lo[2] as f64,
                    ];
                    let v0 = [
                      c.v0_hi_beta[0] as f64 + c.v0_lo_mass[0] as f64,
                      c.v0_hi_beta[1] as f64 + c.v0_lo_mass[1] as f64,
                      c.v0_hi_beta[2] as f64 + c.v0_lo_mass[2] as f64,
                    ];
                    let beta_c = c.v0_hi_beta[3] as f64;
                    let m_c = c.v0_lo_mass[3] as f64;
                    let (sl, sr) = (c.misc[0] as f64, c.misc[1] as f64);
                    let ej = [c.eject[0] as f64, c.eject[1] as f64, c.eject[2] as f64];
                    let e3 = normalize(ej);
                    let ax = if e3[0].abs() < 0.9 {
                      [1.0, 0.0, 0.0]
                    } else {
                      [0.0, 1.0, 0.0]
                    };
                    let e1 = normalize(cross(ax, e3));
                    let e2 = cross(e3, e1);
                    let fr = d::size_range_factor(&c) as f64;
                    let (sb_lo, sb_hi) = ((beta_c / fr).sqrt(), (beta_c * fr).sqrt());
                    let wgt_c = m_c / per as f64;
                    for _ in 0..per {
                      let sb = sb_lo + (sb_hi - sb_lo) * mc_rng(&mut rng);
                      let beta = sb * sb;
                      let f = sb / beta_c.sqrt();
                      let (g1, g2, g3) =
                        (mc_gauss(&mut rng), mc_gauss(&mut rng), mc_gauss(&mut rng));
                      let v = [
                        v0[0]
                          + ej[0] * (f - 1.0)
                          + sl * (g1 * e1[0] + g2 * e2[0])
                          + sr * g3 * e3[0],
                        v0[1]
                          + ej[1] * (f - 1.0)
                          + sl * (g1 * e1[1] + g2 * e2[1])
                          + sr * g3 * e3[1],
                        v0[2]
                          + ej[2] * (f - 1.0)
                          + sl * (g1 * e1[2] + g2 * e2[2])
                          + sr * g3 * e3[2],
                      ];
                      let (r, _) =
                        d::kepler::propagate_f64(r0, v, d::SUN_MU_M3_S2 * (1.0 - beta), age);
                      let global = [
                        view.comet_au[0] + (r[0] - r_jet_now[0]) / d::AU_M,
                        view.comet_au[1] + (r[1] - r_jet_now[1]) / d::AU_M,
                        view.comet_au[2] + (r[2] - r_jet_now[2]) / d::AU_M,
                      ];
                      if let Some((x, y)) = project(global) {
                        hist[y * w + x] += wgt_c;
                        n_drawn += 1;
                      }
                    }
                  }
                  seen.push((bi, n_drawn - drawn_before));
                  continue;
                }
                // cross-section weighted size: pdf ∝ s^(e−2), CDF exponent p = e − 1
                let p = e - 1.0;
                // OBSERVER_MC_QUANTISE_PHASE=1 (diagnostic): the ejection direction is taken at the
                // instant of the cluster the grain falls into (the renderer's model: one attitude per
                // time sample) instead of the grain's own instant
                let quantise = std::env::var("OBSERVER_MC_QUANTISE_PHASE").is_ok_and(|v| v == "1");
                let samples = (desc.count >> d::batch_streams(desc).0).max(1) as f64;
                for _ in 0..n_b {
                  let u = mc_rng(&mut rng);
                  let dt = d::lit_time_map(u as f32, desc.lit, desc.spin[3], dur as f32) as f64;
                  let t0 = t_start + dt;
                  let age = t_now - t0;
                  if !(age >= b.band.0 && age < b.band.1) {
                    continue;
                  }
                  let (rc, vc) = d::kepler::propagate_f64(rc0, vc0, d::SUN_MU_M3_S2, dt);
                  let dt_dir = if quantise {
                    let uq = ((u * samples).floor() + 1.0) / samples;
                    d::lit_time_map(uq.min(1.0) as f32, desc.lit, desc.spin[3], dur as f32) as f64
                  } else {
                    dt
                  };
                  let q = mc_qmul(mc_qaxis_angle(axis, omega * dt_dir), rot_start);
                  let dir = mc_qrot(
                    q,
                    mc_sample_cone(mc_rng(&mut rng), mc_rng(&mut rng), jet_dir, aperture),
                  );
                  let o_t0 = mc_qrot(q, off);
                  let v_site = cross([axis[0] * omega, axis[1] * omega, axis[2] * omega], o_t0);
                  let us = mc_rng(&mut rng);
                  let s = if p.abs() > 1e-9 {
                    (s_min.powf(p) + us * (s_max.powf(p) - s_min.powf(p))).powf(1.0 / p)
                  } else {
                    s_min * (s_max / s_min).powf(us)
                  };
                  let beta = beta_s / s;
                  let f = (beta / beta_ref).sqrt();
                  let v = (v_ref * f * (1.0 + v_std_rel * mc_gauss(&mut rng))).max(0.0);
                  let r0 = [
                    rc[0] + o_t0[0] - o_start[0],
                    rc[1] + o_t0[1] - o_start[1],
                    rc[2] + o_t0[2] - o_start[2],
                  ];
                  let v0 = [
                    vc[0] + dir[0] * v + v_site[0],
                    vc[1] + dir[1] * v + v_site[1],
                    vc[2] + dir[2] * v + v_site[2],
                  ];
                  let (r, _) =
                    d::kepler::propagate_f64(r0, v0, d::SUN_MU_M3_S2 * (1.0 - beta), age);
                  let global = [
                    view.comet_au[0] + (r[0] - r_jet_now[0]) / d::AU_M,
                    view.comet_au[1] + (r[1] - r_jet_now[1]) / d::AU_M,
                    view.comet_au[2] + (r[2] - r_jet_now[2]) / d::AU_M,
                  ];
                  // the renderer's age fade of the oldest tier (part of the model: the TTL end)
                  let wf = if b.fade {
                    wgt * ((1.0 - age / b.band.1) * 10.0).clamp(0.0, 1.0)
                  } else {
                    wgt
                  };
                  if let Some((x, y)) = project(global) {
                    hist[y * w + x] += wf;
                    age_hist[y * w + x] += wf * age;
                    cnt_hist[y * w + x] += 1.0;
                    n_drawn += 1;
                  }
                }
                seen.push((bi, n_drawn - drawn_before));
              }
              (hist, age_hist, cnt_hist, n_drawn, seen)
            })
          })
          .collect();
        for hd in handles {
          let (hist, ah, ch, n, seen) = hd.join().expect("mc thread");
          for (a, b) in counts.iter_mut().zip(ch.iter()) {
            *a += *b;
          }
          for (a, b) in truth.iter_mut().zip(hist.iter()) {
            *a += *b;
          }
          for (a, b) in age_sum.iter_mut().zip(ah.iter()) {
            *a += *b;
          }
          for (bi, k) in seen {
            inview[bi] = k;
          }
          drawn += n;
        }
      });
      (truth, age_sum, drawn, inview, counts)
    };
  // importance allocation: a pilot of 64 grains per batch measures the fraction landing in the
  // frame; the grains are then allotted by mass × that fraction (floor 1/32: no batch starved).
  // The weight per grain stays mass / n_b, so the estimate is unbiased whatever n_b; a near
  // frame holds a few windows of the youngest tier and would otherwise spend 99 % of the
  // grains out of view
  let (_, _, _, pilot, _) = run(&|_| 64usize);
  let weights: Vec<f64> = batches
    .iter()
    .zip(pilot.iter())
    .map(|(b, &k)| b.desc.mass_params[0] as f64 * (k as f64 / 64.0))
    .collect();
  let weight_total: f64 = weights.iter().sum::<f64>().max(1e-300);
  let (truth, age_sum, drawn, _, counts) =
    run(&|bi| ((grains as f64 * weights[bi] / weight_total).round() as usize).max(64));
  // OBSERVER_MC_AGE=<file>: the truth-weighted mean age per pixel (f32 seconds, 0 where empty)
  if let Ok(path) = std::env::var("OBSERVER_MC_AGE") {
    let bytes: Vec<u8> = truth
      .iter()
      .zip(age_sum.iter())
      .map(|(&t, &a)| if t > 0.0 { (a / t) as f32 } else { 0.0 })
      .flat_map(|v| v.to_le_bytes())
      .collect();
    let _ = std::fs::write(path, bytes);
  }
  // compare: unit sums, 5×5 box, the region where the truth is above 1 % of its peak
  let norm = |v: &[f64]| -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s.max(1e-300)).collect()
  };
  let render64: Vec<f64> = render.iter().map(|&x| x as f64).collect();
  let a = mc_box(&norm(&render64), w, h, 2);
  let b = mc_box(&norm(&truth), w, h, 2);
  let peak = b.iter().cloned().fold(0.0f64, f64::max);
  // the region: where the truth is above 1e-4 of its peak (the coma peak is thousands of times
  // the tail; 1 % would judge the coma alone)
  let region: Vec<usize> = (0..w * h).filter(|&i| b[i] >= MC_REGION_REL * peak).collect();
  let core_px = region.iter().filter(|&&i| b[i] >= 0.01 * peak).count();
  let core_unresolved = core_px < MC_CORE_RESOLVED_PX;
  // the Monte-Carlo noise of the box-filtered truth: 1/√n per pixel with n the grains in its
  // 5×5 box; its rms over the judged region (weighted like the error) is taken out of the judged
  // rms in quadrature — a faint tail end with a few grains per box is noise, not a difference
  let n5 = mc_box(&counts, w, h, 2);
  let (mut se, mut sb2) = (0.0, 0.0);
  let (mut se_j, mut sb2_j, mut sn_j) = (0.0, 0.0, 0.0);
  let mut ratios: Vec<f64> = Vec::with_capacity(region.len());
  for &i in &region {
    let d2 = (a[i] - b[i]).powi(2);
    se += d2;
    sb2 += b[i].powi(2);
    if !(core_unresolved && b[i] >= 0.01 * peak) {
      se_j += d2;
      sb2_j += b[i].powi(2);
      let n = (n5[i] * 25.0).max(1.0);
      sn_j += b[i].powi(2) / n;
    }
    ratios.push(a[i] / b[i].max(1e-300));
  }
  let noise_j = (sn_j / sb2_j.max(1e-300)).sqrt();
  let rel_j_raw = (se_j / sb2_j.max(1e-300)).sqrt();
  ratios.sort_by(|x, y| x.partial_cmp(y).unwrap());
  let pct = |q: f64| {
    ratios
      .get(((ratios.len() as f64 - 1.0) * q) as usize)
      .copied()
      .unwrap_or(f64::NAN)
  };
  let ripple = |img: &[f64]| -> f64 {
    let lo = mc_box(img, w, h, 4);
    let mut s2 = 0.0;
    for &i in &region {
      let r = (img[i] - lo[i]) / lo[i].max(1e-300);
      s2 += r * r;
    }
    (s2 / region.len().max(1) as f64).sqrt()
  };
  // the truth's peak against the nucleus pixel
  let (mut best, mut bi) = (0.0, 0usize);
  for (i, &v) in b.iter().enumerate() {
    if v > best {
      best = v;
      bi = i;
    }
  }
  let (bx, by) = ((bi % w) as f64 + 0.5, (bi / w) as f64 + 0.5);
  let tol = 3.0 + view.nucleus_px_radius;
  let nucleus_ok = (bx - view.nucleus_px.0).abs() <= tol && (by - view.nucleus_px.1).abs() <= tol;
  let rec = McRec {
    rel_rms: (se / sb2.max(1e-300)).sqrt(),
    rel_rms_judged: (rel_j_raw * rel_j_raw - noise_j * noise_j).max(0.0).sqrt(),
    noise: noise_j,
    core_px,
    ripple_render: ripple(&a),
    ripple_truth: ripple(&b),
    region_px: region.len(),
    grains_px: drawn as f64 / region.len().max(1) as f64,
    ratio_p05: pct(0.05),
    ratio_p95: pct(0.95),
    nucleus_ok,
    grains: drawn,
    batches: batches.len(),
  };
  let truth32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let render32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  Some((rec, render32, truth32))
}

/// `mc_NNN.png`: render | truth (fourth-root stretch, each to its own peak) | ratio (grey = 1,
/// black = ½, white = 3/2; outside the truth's region: dark blue)
fn write_mc_triptych(path: &Path, render: &[f32], truth: &[f32], w: u32, h: u32) {
  let (wu, hu) = (w as usize, h as usize);
  let peak = |v: &[f32]| v.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
  let (pa, pb) = (peak(render), peak(truth));
  let mut out = vec![0u8; wu * 3 * hu * 3];
  for y in 0..hu {
    for x in 0..wu {
      let i = y * wu + x;
      let ga = ((render[i] / pa).clamp(0.0, 1.0).powf(0.25) * 255.0) as u8;
      let gb = ((truth[i] / pb).clamp(0.0, 1.0).powf(0.25) * 255.0) as u8;
      let o = (y * wu * 3 + x) * 3;
      out[o..o + 3].copy_from_slice(&[ga, ga, ga]);
      let o = (y * wu * 3 + wu + x) * 3;
      out[o..o + 3].copy_from_slice(&[gb, gb, gb]);
      let o = (y * wu * 3 + 2 * wu + x) * 3;
      if truth[i] as f64 >= MC_REGION_REL * pb as f64 {
        let r = render[i] / truth[i].max(1e-30);
        let g = (((r - 0.5) / 1.0).clamp(0.0, 1.0) * 255.0) as u8;
        out[o..o + 3].copy_from_slice(&[g, g, g]);
      } else {
        out[o..o + 3].copy_from_slice(&[0, 0, 40]);
      }
    }
  }
  let _ = image::save_buffer(path, &out, w * 3, h, image::ColorType::Rgb8);
}
