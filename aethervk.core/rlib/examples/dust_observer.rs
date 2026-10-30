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
  /// `--pose inside`: a perspective camera `--inside-m D` sunward of the nucleus on the
  /// anti-solar axis, looking through the nucleus down the tail (the near field, WP3)
  Inside,
}

impl Pose {
  /// the perspective poses (the Monte-Carlo truth weights each grain by its pixel footprint at
  /// its depth and the per-pixel scale is taken at the nucleus' depth)
  fn is_perspective(self) -> bool {
    matches!(self, Pose::Earth | Pose::Inside)
  }
}

/// `--pose inside`: the camera's field of view (full, degrees) unless `--fov-deg` is given
const INSIDE_FOV_DEG: f64 = 60.0;

/// `--jet`: one Model-tab jet (see [`Args::jets`])
#[derive(Clone, Copy, Debug)]
struct JetSpec {
  lat_deg: f64,
  lon_deg: f64,
  speed_ms: f64,
  aperture_deg: f64,
  sigma_frac: f64,
}

struct Args {
  out: PathBuf,
  days: f64,
  step_hours: f64,
  width: u32,
  height: u32,
  pose: Pose,
  play_seconds: Option<f64>,
  /// `--play-live SECONDS`: after the initial seek the simulation plays at `--step-hours` sim
  /// hours per wall second and a frame is captured every `--sample-s` wall seconds WITHOUT
  /// pausing (the reviewer's "the animation doesn't work" probe); `live_NNN.png` + `stats.jsonl`
  play_live_seconds: Option<f64>,
  sample_s: f64,
  /// `--play-quality-off`: `AETHERVK_DUST_PLAY_QUALITY=0` (the A/B of the play quality, P3)
  play_quality_off: bool,
  /// `--play-live-paused`: the paused control of `--play-live`: the clock never runs; every
  /// sample seeks the epoch forward by `--sample-s` × `--step-hours` hours instead (the same
  /// sim step per frame as the play, through the paused path), so play and paused frames at
  /// the same pose and step are compared on the same `stats.jsonl` fields
  play_live_paused: bool,
  reemit_every: u32,
  check: bool,
  /// `--dust-budget-ms <ms>`: a frame whose dust GPU time (`DustGpuTimings::total_ms`) exceeds this
  /// is a problem (the app's budget: 8 ms at 1280×800), listed with the others; with `--check` the
  /// run fails on it
  dust_budget_ms: Option<f64>,
  half_height_km: Option<f64>,
  /// `--fov-deg`: the perspective poses' full field of view (default 2° for `earth`,
  /// [`INSIDE_FOV_DEG`] for `inside`)
  fov_deg: Option<f64>,
  /// `--inside-m D` / `--inside-km D`: the `inside` camera's distance from the nucleus surface
  /// (metres; default 1000; the centre distance is D + 2 km)
  inside_m: f64,
  /// `--lat`, `--lon` (degrees) and `--elev` (metres above the WGS84 ellipsoid): the `earth`
  /// pose's observing site (default 45°, 0°, 0 m)
  lat_deg: f64,
  /// `None` (`--lon auto`, the default): the longitude where the comet transits at the first
  /// frame's epoch (a fixed longitude at UTC midnight puts it below the horizon half the year)
  lon_deg: Option<f64>,
  elev_m: f64,
  /// `--north-up` (`--pose earth`): the observer's screen up is the celestial north
  /// (`ObserverUp::CelestialNorth`: north up, east left) instead of the ecliptic pole
  north_up: bool,
  /// `--earth-field-km KM` (`--pose earth`): the full vertical field at the comet (km); perspective
  /// fov `2·atan(KM/2 / D)` with D the Earth→comet distance at the start epoch, or with `--ortho`
  /// half-height `KM/2`
  earth_field_km: Option<f64>,
  /// `--ortho` (`--pose earth`): an orthographic camera at the observing site aimed at the comet
  /// (frontal), half-height `--earth-field-km`/2 (else the framing's half-height)
  earth_ortho: bool,
  /// `--nucleus-from-earth RADII` (`--pose earth`): the "nucleus from Earth" view (reviewer
  /// feedback #17, `SimulationContext::set_nucleus_view_from_earth`) instead of the observer at
  /// the site: the camera RADII nucleus radii short of the astrometric comet on the Earth → comet
  /// line, celestial north up, with a perspective fov that makes the nucleus 1/4 of the frame
  /// tall (`2·atan(4 / RADII)`), near / far `1e-3 · D` / `max(1e4 · D, 2 · coma)`
  nucleus_from_earth: Option<f64>,
  /// `--sky-image-check` (`--pose earth`): per frame, a synthetic FITS of crosses at known RA/Dec
  /// loaded as the sky overlay, rendered, the crosses found within 1 px of their WCS pixels
  sky_image_check: bool,
  start: Epoch,
  spk: Option<PathBuf>,
  quiet: bool,
  frames_to_wait: u32,
  no_jet: bool,
  hide_sun: bool,
  full_teardown: bool,
  /// `--no-sheet`: no contact sheet, so no frame kept in memory (W·H·4 bytes each: a soak run's
  /// process memory is then the engine's)
  no_sheet: bool,
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
  /// `--jet LAT,LON,SPEED_MS,APERTURE_DEG[,SIGMA_FRAC]` (repeatable): the app's jets (Model tab:
  /// latitude / longitude deg, mean speed m/s, cone half-angle deg, speed σ = SIGMA_FRAC·speed,
  /// default 0.2 like the e2e `Jet` record); empty: the one default jet ([`jet_params`])
  jets: Vec<JetSpec>,
  /// `--pole RA_DEG,DEC_DEG,W0_DEG,RATE_DEG_PER_DAY`: the app's IAU rotational model (Comet tab,
  /// pole J2000, prime meridian W0, sidereal rate; pole rates 0, body-fixed), replaces
  /// `--spin-hours`
  pole: Option<[f64; 4]>,
  /// `--no-trajectories`: hides every trajectory line (orbits, the comet's reference track, its
  /// effective path) and the reference-error annotations, so no line crosses the coma
  no_trajectories: bool,
  /// `--mc N`: grains of the Monte-Carlo truth per frame (0 = off), see [`monte_carlo_truth`]
  mc: usize,
  /// `--mc-tier K`: the truth and the render hold tier K only (near frames: the old tiers'
  /// kernels are AU-sized and cover a 5-km frame as a background the Monte Carlo cannot sample)
  mc_tier: Option<u32>,
  /// `--mc-age-max S`: the truth and the render hold the dust younger than S seconds only (near
  /// frames: a 5-km frame shows the last hour of the jet while the older dust's kernels cover it
  /// as a background the Monte Carlo cannot sample)
  mc_age_max: Option<f64>,
  /// `--mc-age-min S`: the truth (and, through `AETHERVK_DUST_DEBUG_AGE_MIN`, the render once the
  /// engine honours it) hold the dust older than S seconds only
  mc_age_min: Option<f64>,
  /// `--probe "x,y;x,y"`: replay the judged frame's GPU clusters on the CPU mirror and attribute
  /// the τ at those pixels to clusters (`dust::splat_tier_probed`; `probe_NNN.txt`)
  probe: Vec<(u32, u32)>,
  /// `--save-snapshot`: the judged frames' GPU clusters, push constants and header to
  /// `snapshot_NNN.bin` (for `--lab`; set `OBSERVER_MC_RAW=<out>/mc_NNN` too for the truth)
  save_snapshot: bool,
  /// `--lab DIR [--lab-frame N]`: no GPU; replay `DIR/snapshot_NNN.bin` on the CPU mirror and
  /// score it against `DIR/mc_NNN_truth.f32` (the offline replay lab)
  lab: Option<PathBuf>,
  lab_frame: u32,
  /// `--lab-truth N`: the lab draws its own truth (N grains) from `truth_NNN.bin` instead of
  /// reading `mc_NNN_truth.f32`; `OBSERVER_LAB_AGE=lo,hi` (s) then gates both sides by age
  lab_truth: usize,
  /// `--path-frames N` (`--pose inside`): the camera path of temporal reuse step 4 — N frames
  /// from the settled frame 0, one hold-step each (see the path block in `main`); 0 = off
  path_frames: u32,
  /// `--path-fps F`: the frame rate the path models (Δt per frame = speed / F; ω per frame)
  path_fps: f64,
  /// `--path-dt-s T`: simulation seconds per path frame (`--path-speed X` sets it to X / F)
  path_dt_s: f64,
  path_speed: Option<f64>,
  /// `--orbit-deg-per-s W`: the orbit about the nucleus (degrees per second at `path_fps`)
  orbit_deg_per_s: f64,
  /// `--eye-step-m S`: the orbit by its arc per frame instead (m)
  eye_step_m: Option<f64>,
  /// `--dolly-m-per-frame D`: the eye's approach to the nucleus per frame (m)
  dolly_m_per_frame: f64,
  /// `--path-keep 0,31`: the path frames whose pyramid / PNG / truth inputs (/ snapshot) are kept
  path_keep: Vec<u32>,
  /// `--lab-path N` (with `--lab`): the CPU mirror's cadence along a Δt = 0 orbit of N frames
  lab_path: u32,
  /// `--lab-path-eps 1,2,4,16`: the tolerances (px) the CPU path runs
  lab_path_eps: Vec<f32>,
  /// `--lab-score DIR --against FULL`: score a path capture's frame against the full frame's truth
  lab_score: Option<PathBuf>,
  lab_against: Option<PathBuf>,
  /// `--lab-label L`: the variant's name in the score lines and the table
  lab_label: Option<String>,
  /// `--lab-table T.tsv`: append the scored row; `--lab-table-md T.tsv`: print the tables
  lab_table: Option<PathBuf>,
  lab_table_md: Option<PathBuf>,
  /// `--view-flags F`: dust view flags (`DUST_VIEW_TRACERS` 1, `DUST_VIEW_FLOW` 2,
  /// `DUST_VIEW_RHO_NORM` 8; default 0, the plain optical depth, what the Monte-Carlo truth
  /// compares with and the app's default)
  view_flags: u32,
  /// `--exposure-ev X`: the display's exposure correction (`set_dust_exposure_ev`; default 0)
  exposure_ev: f32,
  /// `--check`: the largest allowed 2-D rms of the frame against its Monte-Carlo truth
  mc_max: f64,
  /// `--afrho-aperture-km R`: the aperture Afρ of the render and of the truth inside R km of the
  /// nucleus pixel, against the recipe's Afρ(r_h)·Φ(α) (meaningful for `--isotropic`)
  afrho_aperture_km: Option<f64>,
  /// `--isotropic`: the jet emits over the whole sphere (aperture π) from a non-rotating nucleus
  /// (always lit): the geometry of the Afρ closure
  isotropic: bool,
}

fn usage() -> ! {
  eprintln!(
    "usage: dust_observer --out DIR [--days 30] [--step-hours 24] [--size 1280x800]\n\
     \x20      [--pose up-zenith|anti-sun|earth|inside] [--play SECONDS] [--play-live SECONDS --sample-s S [--play-quality-off] [--play-live-paused]] [--reemit-every N] [--check] [--dust-budget-ms MS]\n\
     \x20      [--half-height-km KM] [--fov-deg DEG] [--inside-m D | --inside-km D] [--lat DEG --lon DEG --elev M] [--sky-image-check]\n\
     \x20      [--north-up] [--earth-field-km KM] [--ortho] [--nucleus-from-earth RADII] [--start YYYY-MM-DD[THH:MM:SS[Z]]] [--spk FILE.bsp]\n\
     \x20      [--quiet] [--settle-frames N] [--no-jet] [--hide-sun] [--no-trajectories] [--full-teardown] [--no-sheet]\n\
     \x20      [--jet LAT,LON,SPEED_MS,APERTURE_DEG[,SIGMA_FRAC]]... [--pole RA_DEG,DEC_DEG,W0_DEG,RATE_DEG_PER_DAY]\n\
     \x20      [--ignite-days D] [--prestart] [--ripple-max R] [--spin-hours H] [--mc N] [--mc-max R] [--mc-age-min S] [--view-flags F] [--exposure-ev X]\n\
     \x20      [--afrho-aperture-km R] [--isotropic]\n\
     \x20      [--path-frames N [--path-fps 60] [--path-speed X | --path-dt-s T] [--orbit-deg-per-s 10 | --eye-step-m S] [--dolly-m-per-frame D] [--path-keep 0,31]]\n\
     \x20      [--lab DIR [--lab-frame N] [--lab-truth N] [--lab-path N [--lab-path-eps 1,2,4,16]] [--lab-table T.tsv]]\n\
     \x20      [--lab-score DIR --against FULL --lab-frame N --lab-truth N [--lab-label L] [--lab-table T.tsv]] [--lab-table-md T.tsv]"
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
    play_live_seconds: None,
    sample_s: 1.0,
    play_quality_off: false,
    play_live_paused: false,
    reemit_every: 0,
    check: false,
    dust_budget_ms: None,
    half_height_km: None,
    fov_deg: None,
    inside_m: 1000.0,
    lat_deg: 45.0,
    lon_deg: None,
    elev_m: 0.0,
    sky_image_check: false,
    north_up: false,
    earth_field_km: None,
    earth_ortho: false,
    nucleus_from_earth: None,
    start: Epoch::from_gregorian_utc(2025, 10, 2, 0, 0, 0, 0),
    spk: None,
    quiet: false,
    frames_to_wait: 4,
    no_jet: false,
    hide_sun: false,
    full_teardown: false,
    no_sheet: false,
    ignite_days: 0.0,
    prestart: false,
    ripple_max: None,
    spin_hours: ROTATION_PERIOD_H,
    jets: Vec::new(),
    pole: None,
    no_trajectories: false,
    afrho_aperture_km: None,
    isotropic: false,
    mc: 0,
    mc_max: 0.15,
    view_flags: 0,
    exposure_ev: 0.0,
    mc_tier: None,
    mc_age_max: None,
    mc_age_min: None,
    probe: Vec::new(),
    save_snapshot: false,
    lab: None,
    lab_frame: 1,
    lab_truth: 0,
    path_frames: 0,
    path_fps: 60.0,
    path_dt_s: 0.0,
    path_speed: None,
    orbit_deg_per_s: 10.0,
    eye_step_m: None,
    dolly_m_per_frame: 0.0,
    path_keep: Vec::new(),
    lab_path: 0,
    lab_path_eps: vec![1.0],
    lab_score: None,
    lab_against: None,
    lab_label: None,
    lab_table: None,
    lab_table_md: None,
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
          "inside" => Pose::Inside,
          _ => usage(),
        }
      }
      "--inside-m" => a.inside_m = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--inside-km" => a.inside_m = next(&mut i).parse::<f64>().unwrap_or_else(|_| usage()) * 1e3,
      "--play" => a.play_seconds = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--play-live" => a.play_live_seconds = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--sample-s" => a.sample_s = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--play-quality-off" => a.play_quality_off = true,
      "--play-live-paused" => a.play_live_paused = true,
      "--reemit-every" => a.reemit_every = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--check" => a.check = true,
      "--dust-budget-ms" => a.dust_budget_ms = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--quiet" => a.quiet = true,
      "--no-jet" => a.no_jet = true,
      "--full-teardown" => a.full_teardown = true,
      "--no-sheet" => a.no_sheet = true,
      "--hide-sun" => a.hide_sun = true,
      "--no-trajectories" => a.no_trajectories = true,
      "--prestart" => a.prestart = true,
      "--ignite-days" => a.ignite_days = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--ripple-max" => a.ripple_max = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--spin-hours" => a.spin_hours = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--jet" => {
        let v: Vec<f64> =
          next(&mut i).split(',').map(|x| x.trim().parse().unwrap_or_else(|_| usage())).collect();
        if !(4..=5).contains(&v.len()) {
          usage();
        }
        a.jets.push(JetSpec {
          lat_deg: v[0],
          lon_deg: v[1],
          speed_ms: v[2],
          aperture_deg: v[3],
          sigma_frac: v.get(4).copied().unwrap_or(0.2),
        });
      }
      "--pole" => {
        let v: Vec<f64> =
          next(&mut i).split(',').map(|x| x.trim().parse().unwrap_or_else(|_| usage())).collect();
        if v.len() != 4 {
          usage();
        }
        a.pole = Some([v[0], v[1], v[2], v[3]]);
      }
      "--mc" => a.mc = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--view-flags" => a.view_flags = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--exposure-ev" => a.exposure_ev = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--mc-tier" => a.mc_tier = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--mc-age-max" => a.mc_age_max = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--mc-age-min" => a.mc_age_min = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--save-snapshot" => a.save_snapshot = true,
      "--lab" => a.lab = Some(PathBuf::from(next(&mut i))),
      "--lab-frame" => a.lab_frame = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--lab-truth" => a.lab_truth = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--path-frames" => a.path_frames = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--path-fps" => a.path_fps = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--path-dt-s" => a.path_dt_s = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--path-speed" => a.path_speed = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--orbit-deg-per-s" => a.orbit_deg_per_s = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--eye-step-m" => a.eye_step_m = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--dolly-m-per-frame" => a.dolly_m_per_frame = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--path-keep" => {
        a.path_keep = next(&mut i).split(',').map(|v| v.trim().parse().unwrap_or_else(|_| usage())).collect()
      }
      "--lab-path" => a.lab_path = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--lab-path-eps" => {
        a.lab_path_eps = next(&mut i).split(',').map(|v| v.trim().parse().unwrap_or_else(|_| usage())).collect()
      }
      "--lab-score" => a.lab_score = Some(PathBuf::from(next(&mut i))),
      "--against" => a.lab_against = Some(PathBuf::from(next(&mut i))),
      "--lab-label" => a.lab_label = Some(next(&mut i)),
      "--lab-table" => a.lab_table = Some(PathBuf::from(next(&mut i))),
      "--lab-table-md" => a.lab_table_md = Some(PathBuf::from(next(&mut i))),
      "--probe" => {
        a.probe = next(&mut i)
          .split(';')
          .filter_map(|p| {
            let mut it = p.split(',');
            Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
          })
          .collect()
      }
      "--mc-max" => a.mc_max = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--afrho-aperture-km" => {
        a.afrho_aperture_km = Some(next(&mut i).parse().unwrap_or_else(|_| usage()))
      }
      "--sky-image-check" => a.sky_image_check = true,
      "--north-up" => a.north_up = true,
      "--ortho" => a.earth_ortho = true,
      "--nucleus-from-earth" => {
        a.nucleus_from_earth = Some(next(&mut i).parse().unwrap_or_else(|_| usage()))
      }
      "--earth-field-km" => {
        a.earth_field_km = Some(next(&mut i).parse().unwrap_or_else(|_| usage()))
      }
      "--isotropic" => {
        a.isotropic = true;
        a.spin_hours = 0.0;
      }
      "--half-height-km" => {
        a.half_height_km = Some(next(&mut i).parse().unwrap_or_else(|_| usage()))
      }
      "--lat" => a.lat_deg = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--lon" => {
        a.lon_deg = match next(&mut i).as_str() {
          "auto" => None,
          v => Some(v.parse().unwrap_or_else(|_| usage())),
        }
      }
      "--elev" => a.elev_m = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--fov-deg" => a.fov_deg = Some(next(&mut i).parse().unwrap_or_else(|_| usage())),
      "--settle-frames" => a.frames_to_wait = next(&mut i).parse().unwrap_or_else(|_| usage()),
      "--start" => {
        // `YYYY-MM-DD` (00:00 UTC) or `YYYY-MM-DDTHH:MM:SS` (UTC; a trailing `Z` is allowed)
        let s = next(&mut i);
        let (date, time) = s.split_once(['T', ' ']).unwrap_or((s.as_str(), "00:00:00"));
        let p: Vec<i32> = date.split('-').filter_map(|x| x.parse().ok()).collect();
        let t: Vec<u8> =
          time.trim_end_matches('Z').split(':').filter_map(|x| x.parse().ok()).collect();
        if p.len() != 3 || t.len() != 3 || t[0] > 23 || t[1] > 59 || t[2] > 60 {
          usage();
        }
        a.start = Epoch::from_gregorian_utc(p[0], p[1] as u8, p[2] as u8, t[0], t[1], t[2], 0);
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
  if let Some(x) = a.path_speed {
    a.path_dt_s = x / a.path_fps;
  }
  if a.path_frames > 0 {
    if a.pose != Pose::Inside {
      eprintln!("--path-frames needs --pose inside (the nucleus-relative camera path)");
      usage();
    }
    if !(a.path_fps > 0.0 && a.path_dt_s >= 0.0 && a.path_dt_s.is_finite()) {
      eprintln!("--path-fps must be positive and the path's Δt finite and ≥ 0");
      usage();
    }
    if let Some(k) = a.path_keep.iter().find(|&&k| k >= a.path_frames) {
      eprintln!("--path-keep {k}: beyond the path's {} frames", a.path_frames);
      usage();
    }
  }
  if a.out.as_os_str().is_empty() {
    eprintln!("--out DIR is required");
    usage();
  }
  if a.sky_image_check && a.pose != Pose::Earth {
    eprintln!(
      "--sky-image-check needs --pose earth (the overlay is drawn for perspective cameras)"
    );
    usage();
  }
  if (a.north_up || a.earth_ortho || a.earth_field_km.is_some() || a.nucleus_from_earth.is_some())
    && a.pose != Pose::Earth
  {
    eprintln!("--north-up / --ortho / --earth-field-km / --nucleus-from-earth need --pose earth");
    usage();
  }
  if let Some(r) = a.nucleus_from_earth {
    if !(r > 0.0 && r.is_finite()) {
      eprintln!("--nucleus-from-earth: the distance in nucleus radii must be positive");
      usage();
    }
    if a.earth_ortho || a.earth_field_km.is_some() || a.sky_image_check {
      eprintln!("--nucleus-from-earth sets its own perspective field (not --ortho / --earth-field-km / --sky-image-check)");
      usage();
    }
  }
  if a.earth_field_km.is_some_and(|k| !(k > 0.0 && k.is_finite())) {
    eprintln!("--earth-field-km: the field must be positive");
    usage();
  }
  if a.earth_ortho && a.sky_image_check {
    eprintln!("--sky-image-check needs the perspective earth pose (not --ortho)");
    usage();
  }
  if !(a.inside_m > 0.0 && a.inside_m.is_finite()) {
    eprintln!("--inside-m / --inside-km: the distance from the surface must be positive");
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
  /// metres per pixel at the nucleus' depth: `2·half_height/h` for the orthographic poses,
  /// `2·depth·tan(fov/2)/h` for the perspective ones — every per-pixel scale (nucleus radius on
  /// screen, the shape analysis, the moments diagnostics) derives from it
  m_per_px_nucleus: f64,
  nucleus_px: (f64, f64),
  nucleus_rgb: (u8, u8, u8),
  mean_lum: f64,
  p99_lum: f64,
  nonblack_frac: f64,
  /// the flow-pulse exposure metric (`saturated_fraction`, `core_median`, `annulus_median`,
  /// `core_over_annulus` in `stats.jsonl`)
  flow_metric: FlowMetric,
  render_task: u64,
  /// wall time per render generation (ms) and the dust passes' GPU time
  /// (`Device::dust_gpu_timings`: the last frame measured before this one was downloaded)
  render_ms: f64,
  dust_gpu: Option<aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings>,
  /// the shader-clock profiler's counters of the last drawn frame (`AETHERVK_DUST_PROFILE=1`)
  dust_profile: Option<aethervk_core_rlib::scene::dust::DustProfileCounters>,
  /// the budget loop's telemetry as JSON (`DustBudgetTelemetry::to_json`; "null" without one)
  dust_budget: String,
  passes_logged: String,
  /// the frame against its Monte-Carlo truth (`--mc`)
  mc: Option<McRec>,
  /// the view's exposure: white point (radiance factor), V mag/arcsec², EV, softening
  exposure: Option<aethervk_core_rlib::scene::dust::DustExposure>,
  /// the displayed hue along the tail axis ([`hue_ramp_samples`])
  hue: Vec<HueSample>,
  /// the speck pass's header words of the frame's pyramid dump (WP3b), `None` without a dump
  specks: Option<SpeckRec>,
}

/// The speck header words (`dust::specks`): the shell radius `d_max` (m, f32 bits of
/// `PYR_SPECKS_DMAX`; 0 = no shell, the first ~4 frames lag the readback), the specks drawn
/// (`PYR_SPECKS`) and refused by the budget (`PYR_SPECKS_OVERFLOW`), and the number density at
/// the eye (`PYR_N_EYE` | `PYR_N_EYE_HI` << 32, 2⁻³⁶ m⁻³ per count)
#[derive(Clone, Copy, Debug)]
struct SpeckRec {
  d_max_m: f32,
  specks: u32,
  overflow: u32,
  n_eye_m3: f64,
}

/// One sample of the displayed hue ramp along the tail axis (the on-screen anti-solar direction
/// from the nucleus), a 5×5 box mean of the frame's pixels. The headless render target is
/// `R8G8B8A8_UNORM` (`device.rs` `render_target_format`): the bytes are linear light, no sRGB
/// decode (the app's swapchain is `B8G8R8A8_SRGB` and encodes on write; decoding these bytes
/// as sRGB turned the stream's hue by −17° and inflated its chroma 1.6×)
#[derive(Clone, Debug)]
struct HueSample {
  px: (f64, f64),
  /// display level: luma of the 8-bit pixel over 255 (the composite rolls off to white above 0.8)
  level: f64,
  /// OKLab hue (degrees) and chroma over lightness relative to the stream colour's (the display
  /// level scales OKLab's L, a and b together, so C/L is what the age colour sets)
  hue_deg: f64,
  chroma_rel: f64,
  dark: bool,
  highlight: bool,
  /// display level below [`HUE_FAINT_LEVEL`]: the colour words' dither (one-count deposits) is
  /// visible in a 9×9 box, reported but not judged
  faint: bool,
  /// the Monte-Carlo truth's radiance-weighted mean age and age spread at the sample, in TTL
  age_ttl: Option<f64>,
  age_spread_ttl: Option<f64>,
  /// the truth's own colour at the sample: the radiance-weighted mean of its grains' age colours
  /// (linear RGB, the mixture the render claims), as OKLab hue and relative chroma
  truth_hue_deg: Option<f64>,
  truth_chroma_rel: Option<f64>,
  /// the render's accumulated colour at the sample: the pyramid's colour words over its τ in the
  /// 9×9 box (every level, per pixel area), the quantity the check judges — the displayed pixel
  /// also holds whatever the scene shows behind the dust (the orbit line under the 1 AU tail)
  pyr_hue_deg: Option<f64>,
  pyr_chroma_rel: Option<f64>,
  /// τ counts of that box (per pixel area): below [`HUE_PYR_FAINT_COUNTS`] the sample is faint
  pyr_tau: f64,
  /// render over truth of the box's τ (both unit-sum images): outside [`HUE_TAU_RATIO`] the box
  /// is haze — dust the render has where the truth has little (the old tiers' pooled packets
  /// spread over a near frame) — and its colour is reported, not judged
  tau_ratio: Option<f64>,
}

/// the render/truth τ ratio of a box within which its colour is judged
const HUE_TAU_RATIO: f64 = 2.0;

/// τ counts (per pixel area, summed over the 9×9 box) below which the pyramid colour is faint
const HUE_PYR_FAINT_COUNTS: f64 = 50.0;

/// The pyramid's colour at a pixel box: Σ over the levels of the texel words per pixel area
/// (mirror of the composite's sum without the bilinear weights), as (τ counts, linear colour)
fn pyramid_colour_at(pyr: &[u32], px: (f64, f64), r: i64) -> (f64, Option<[f32; 3]>) {
  // `OBSERVER_HUE_LEVEL0=1`: the judged colour from level 0 only (diagnostic)
  let level0_only = std::env::var("OBSERVER_HUE_LEVEL0").as_deref() == Ok("1");
  let (mut tau, mut rgb) = (0.0f64, [0.0f64; 3]);
  for (l, (t, c)) in pyramid_levels_at(pyr, px, r).into_iter().enumerate() {
    if level0_only && l > 0 {
      break;
    }
    tau += t;
    for k in 0..3 {
      rgb[k] += c[k];
    }
  }
  (tau, colour_of_words(tau, rgb))
}

/// Per pyramid level, the τ counts (per pixel area) and the colour words (per pixel area)
/// summed over the `(2r+1)²` pixel box mapped to that level's texels: the terms
/// [`pyramid_colour_at`] adds up
fn pyramid_levels_at(pyr: &[u32], px: (f64, f64), r: i64) -> Vec<(f64, [f64; 3])> {
  use aethervk_core_rlib::scene::dust as d;
  let levels = pyr[d::PYR_LEVELS as usize] as usize;
  let (ix, iy) = (px.0.round() as i64, px.1.round() as i64);
  let mut out = Vec::with_capacity(levels);
  for l in 0..levels {
    let (mut tau, mut rgb) = (0.0f64, [0.0f64; 3]);
    let off = pyr[d::PYR_TABLE as usize + 3 * l] as i64;
    let lw = pyr[d::PYR_TABLE as usize + 3 * l + 1] as i64;
    let lh = pyr[d::PYR_TABLE as usize + 3 * l + 2] as i64;
    let s = 1i64 << l;
    let area = (s * s) as f64;
    for y in iy - r..=iy + r {
      for x in ix - r..=ix + r {
        let (tx, ty) = (x >> l, y >> l);
        if tx < 0 || ty < 0 || tx >= lw || ty >= lh {
          continue;
        }
        let i = (off + (ty * lw + tx) * d::PYRAMID_TEXEL_WORDS as i64) as usize;
        let c = pyr[i] as f64;
        if c == 0.0 {
          continue;
        }
        tau += c / area;
        for k in 0..3 {
          rgb[k] += pyr[i + 1 + k] as f64 / area;
        }
      }
    }
    out.push((tau, rgb));
  }
  out
}

/// The linear colour of summed colour words: words over `DUST_COLOR_SCALE · τ`
fn colour_of_words(tau: f64, rgb: [f64; 3]) -> Option<[f32; 3]> {
  use aethervk_core_rlib::scene::dust as d;
  if tau > 0.0 {
    let k = 1.0 / (tau * d::DUST_COLOR_SCALE as f64);
    Some([
      (rgb[0] * k) as f32,
      (rgb[1] * k) as f32,
      (rgb[2] * k) as f32,
    ])
  } else {
    None
  }
}

/// `OBSERVER_HUE_LEVELS=1`: one log line per hue sample with each level's τ and colour
fn hue_levels_line(i: u32, k: usize, pyr: &[u32], px: (f64, f64), stream: [f32; 3]) -> String {
  let (_, scl) = oklab_hue_cl(stream);
  let parts: Vec<String> = pyramid_levels_at(pyr, px, HUE_BOX_R)
    .into_iter()
    .enumerate()
    .map(|(l, (t, c))| match colour_of_words(t, c) {
      Some(rgb) => {
        let (h, cl) = oklab_hue_cl(rgb);
        format!("L{l} τ={t:.1} hue={h:.1}° C/L={:.3}", cl / scl.max(1e-6))
      }
      None => format!("L{l} τ=0"),
    })
    .collect();
  format!(
    "[observer] hue levels frame {i} sample {k} at ({:.0}, {:.0}): {}",
    px.0,
    px.1,
    parts.join(" | ")
  )
}

/// samples of the hue ramp per frame
const HUE_SAMPLES: usize = 10;
/// `--check`: the hue within this of the truth's; the chroma kept over the truth's at least
/// this as the frame's median over its judged samples (the guarantee) and this per sample (the
/// outlier guard: 8-bit frame bytes — the orange's blue channel at display level 0.2 is 10
/// counts, half a count is 5 % — and a 9×9 box across an age gradient; the month run's 135
/// judged samples: median 0.98, 5th percentile 0.87, least 0.80)
const HUE_TOL_DEG: f64 = 15.0;
const HUE_CHROMA_MIN: f64 = 0.75;
const HUE_CHROMA_MEDIAN_MIN: f64 = 0.9;
/// the colour box half-width (px) and the display level below which a sample is faint (not
/// judged: the colour words' dither and the 8-bit quantisation)
const HUE_BOX_R: i64 = 4;
const HUE_FAINT_LEVEL: f64 = 0.2;
/// the composite rolls the hue off towards white above this stretch value
/// (`DUST_HIGHLIGHT_START`): a sample whose level over its colour's luma exceeds it is a
/// highlight, not judged
const HUE_HIGHLIGHT_A: f64 = 0.8;

/// Rec. 709 luma of a linear colour
fn luma_of(c: [f32; 3]) -> f64 {
  (0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64).max(1e-6)
}

/// OKLab hue (degrees) and chroma over lightness of a linear colour
fn oklab_hue_cl(lin: [f32; 3]) -> (f64, f64) {
  let lab = aethervk_core_rlib::scene::dust::oklab_of(lin);
  let c = ((lab[1] * lab[1] + lab[2] * lab[2]) as f64).sqrt();
  (
    (lab[2] as f64).atan2(lab[1] as f64).to_degrees(),
    c / (lab[0] as f64).max(1e-6),
  )
}

/// The hue ramp's ray: from the nucleus pixel along the on-screen anti-solar direction (pixel
/// axes, unit), from `r0` px out
#[derive(Clone, Copy, Debug)]
struct HueRay {
  from: (f64, f64),
  dir: (f64, f64),
  r0: f64,
}

impl HueRay {
  /// how far the ray can run inside the frame (px)
  fn reach(&self, width: u32, height: u32) -> f64 {
    let (w, h) = (width as f64, height as f64);
    let mut reach = f64::INFINITY;
    if self.dir.0 > 1e-9 {
      reach = reach.min((w - 3.0 - self.from.0) / self.dir.0);
    }
    if self.dir.0 < -1e-9 {
      reach = reach.min((2.0 - self.from.0) / self.dir.0);
    }
    if self.dir.1 > 1e-9 {
      reach = reach.min((h - 3.0 - self.from.1) / self.dir.1);
    }
    if self.dir.1 < -1e-9 {
      reach = reach.min((2.0 - self.from.1) / self.dir.1);
    }
    reach.max(self.r0)
  }
  fn at(&self, t: f64) -> (f64, f64) {
    (self.from.0 + self.dir.0 * t, self.from.1 + self.dir.1 * t)
  }
  /// `HUE_SAMPLES` pixels from `r0` to `t_end` (at least 9 px out)
  fn samples(&self, t_end: f64) -> Vec<(f64, f64)> {
    let t_end = t_end.max(self.r0 + 9.0);
    (0..HUE_SAMPLES)
      .map(|k| self.at(self.r0 + (t_end - self.r0) * k as f64 / (HUE_SAMPLES - 1) as f64))
      .collect()
  }
}

/// The end of the lit dust along the ray on the displayed frame (the last 3×3 box above the
/// dark level): the fallback without `--mc` (the truth's region places the samples otherwise —
/// a stray lit pixel or a slightly different exposure moved this scan between two runs of one
/// frame)
fn hue_ray_extent_display(rgba: &[u8], width: u32, height: u32, ray: &HueRay) -> f64 {
  let (w, h) = (width as i64, height as i64);
  let luma_at = |cx: f64, cy: f64, r: i64| -> f64 {
    let (ix, iy) = (cx.round() as i64, cy.round() as i64);
    let (mut l, mut n) = (0.0f64, 0.0f64);
    for y in iy - r..=iy + r {
      for x in ix - r..=ix + r {
        if x < 0 || y < 0 || x >= w || y >= h {
          continue;
        }
        let o = ((y * w + x) * 4) as usize;
        l += (0.2126 * rgba[o] as f64 + 0.7152 * rgba[o + 1] as f64 + 0.0722 * rgba[o + 2] as f64)
          / 255.0;
        n += 1.0;
      }
    }
    l / n.max(1.0)
  };
  let reach = ray.reach(width, height);
  let mut t_end = ray.r0;
  let mut t = ray.r0;
  while t <= reach {
    let (cx, cy) = ray.at(t);
    if luma_at(cx, cy, 1) >= 8.0 / 255.0 {
      t_end = t;
    }
    t += 1.0;
  }
  t_end
}

/// The displayed colour at a sample pixel: the hue and relative chroma of the 9×9 box mean of
/// the frame's linear bytes, the display level, and the dark / faint / highlight flags
fn hue_sample_at(
  rgba: &[u8],
  width: u32,
  height: u32,
  px: (f64, f64),
  stream: [f32; 3],
) -> HueSample {
  let (w, h) = (width as i64, height as i64);
  let (_, stream_cl) = oklab_hue_cl(stream);
  let (ix, iy) = (px.0.round() as i64, px.1.round() as i64);
  let (mut lin, mut luma, mut n) = ([0.0f64; 3], 0.0f64, 0.0f64);
  for y in iy - HUE_BOX_R..=iy + HUE_BOX_R {
    for x in ix - HUE_BOX_R..=ix + HUE_BOX_R {
      if x < 0 || y < 0 || x >= w || y >= h {
        continue;
      }
      let o = ((y * w + x) * 4) as usize;
      let p = [
        rgba[o] as f64 / 255.0,
        rgba[o + 1] as f64 / 255.0,
        rgba[o + 2] as f64 / 255.0,
      ];
      luma += 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
      for k in 0..3 {
        lin[k] += p[k];
      }
      n += 1.0;
    }
  }
  let n = n.max(1.0);
  let level = luma / n;
  let (hue_deg, cl) = oklab_hue_cl([
    (lin[0] / n) as f32,
    (lin[1] / n) as f32,
    (lin[2] / n) as f32,
  ]);
  HueSample {
    px,
    level,
    hue_deg,
    chroma_rel: cl / stream_cl.max(1e-6),
    dark: level < 8.0 / 255.0,
    // provisional: the stretch value over the stream colour's luma (the truth colour's when known)
    highlight: level / luma_of(stream) > HUE_HIGHLIGHT_A,
    faint: level < HUE_FAINT_LEVEL,
    age_ttl: None,
    age_spread_ttl: None,
    truth_hue_deg: None,
    truth_chroma_rel: None,
    pyr_hue_deg: None,
    pyr_chroma_rel: None,
    pyr_tau: 0.0,
    tau_ratio: None,
  }
}

/// the signed turn of `hue` from `from`, in (−180, 180]
fn hue_turn_deg(hue: f64, from: f64) -> f64 {
  let d = (hue - from).rem_euclid(360.0);
  if d > 180.0 { d - 360.0 } else { d }
}

/// The hue ramp's log line and `--check` failures. A sample whose pyramid box holds at least
/// [`HUE_PYR_FAINT_COUNTS`] τ counts and has a truth colour must accumulate that colour: the
/// render's hue within [`HUE_TOL_DEG`] of the truth's and its relative chroma at least
/// [`HUE_CHROMA_MIN`] of the truth's per sample, [`HUE_CHROMA_MEDIAN_MIN`] as the frame's median
/// (the truth mixes its grains' age colours exactly as the render claims to: a pixel of one age
/// keeps the stream's chroma, a pixel of many ages greys by their hue spread); and the render's
/// turn from the stream colour must not fall by more than 10° between consecutive judged
/// samples of increasing truth age. A box whose render/truth τ is beyond [`HUE_TAU_RATIO`] is
/// haze (the render's dust where the truth has little): counted, its colour not judged. The
/// displayed colour (the pixel, with whatever the scene shows behind the dust) and the age
/// colour of the truth's mean age (the clock) are logged.
fn hue_report(
  i: u32,
  hue: &[HueSample],
  stream_hue_deg: f64,
  check: bool,
) -> (String, Vec<String>) {
  let mut problems = Vec::new();
  let (mut dark, mut highlight, mut faint, mut haze, mut judged) =
    (0usize, 0usize, 0usize, 0usize, 0usize);
  let mut chroma_min = f64::INFINITY;
  let mut chroma_ratios: Vec<f64> = Vec::new();
  let mut prev: Option<(f64, f64)> = None;
  for (k, s) in hue.iter().enumerate() {
    dark += s.dark as usize;
    highlight += s.highlight as usize;
    let (Some(age), Some(th), Some(tc), Some(ph), Some(pc)) = (
      s.age_ttl,
      s.truth_hue_deg,
      s.truth_chroma_rel,
      s.pyr_hue_deg,
      s.pyr_chroma_rel,
    ) else {
      continue;
    };
    if s.pyr_tau < HUE_PYR_FAINT_COUNTS {
      faint += 1;
      continue;
    }
    if !s.tau_ratio.is_some_and(|r| r >= 1.0 / HUE_TAU_RATIO && r <= HUE_TAU_RATIO) {
      haze += 1;
      continue;
    }
    judged += 1;
    let chroma_ratio = pc / tc.max(1e-6);
    chroma_min = chroma_min.min(chroma_ratio);
    chroma_ratios.push(chroma_ratio);
    let turn = hue_turn_deg(ph, stream_hue_deg).rem_euclid(360.0);
    let off = hue_turn_deg(ph, th);
    if check && off.abs() > HUE_TOL_DEG {
      problems.push(format!(
        "frame {i}: hue sample {k} at ({:.0}, {:.0}) : the render's colour turned {turn:.0}° from the stream, the truth's colour mix turns {:.0}° (mean age {age:.2} TTL; tolerance {HUE_TOL_DEG:.0}°)",
        s.px.0,
        s.px.1,
        hue_turn_deg(th, stream_hue_deg).rem_euclid(360.0)
      ));
    }
    if check && chroma_ratio < HUE_CHROMA_MIN {
      problems.push(format!(
        "frame {i}: hue sample {k} at ({:.0}, {:.0}) keeps {:.2} of the truth's chroma ({:.2} vs {:.2} of the stream's; mean age {age:.2} TTL, spread {:.2}; minimum {HUE_CHROMA_MIN})",
        s.px.0,
        s.px.1,
        chroma_ratio,
        pc,
        tc,
        s.age_spread_ttl.unwrap_or(0.0)
      ));
    }
    if let Some((pa, pt)) = prev
      && check
      && age > pa
      && turn < pt - 10.0
    {
      problems.push(format!(
        "frame {i}: hue sample {k} turns {turn:.0}° after {pt:.0}° at a younger sample (the ramp runs backwards)"
      ));
    }
    prev = Some((age, turn));
  }
  let chroma_median = if chroma_ratios.is_empty() {
    f64::NAN
  } else {
    chroma_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    chroma_ratios[chroma_ratios.len() / 2]
  };
  if check && chroma_median.is_finite() && chroma_median < HUE_CHROMA_MEDIAN_MIN {
    problems.push(format!(
      "frame {i}: the median of the render's chroma over the truth's is {chroma_median:.2} across {judged} judged samples (minimum {HUE_CHROMA_MEDIAN_MIN})"
    ));
  }
  let lit: Vec<&HueSample> = hue.iter().filter(|s| !s.dark && !s.highlight).collect();
  let line = if lit.is_empty() {
    format!("[observer] hue frame {i}: no lit sample ({dark} dark, {highlight} highlight)")
  } else {
    let t = |h: f64| hue_turn_deg(h, stream_hue_deg).rem_euclid(360.0);
    let (f, l) = (lit[0], lit[lit.len() - 1]);
    let truth = match (f.truth_hue_deg, l.truth_hue_deg) {
      (Some(a), Some(b)) => format!(", truth {:.0}° → {:.0}°", t(a), t(b)),
      _ => String::new(),
    };
    let render = match (f.pyr_hue_deg, l.pyr_hue_deg) {
      (Some(a), Some(b)) => format!(", render {:.0}° → {:.0}°", t(a), t(b)),
      _ => String::new(),
    };
    let clock = match (f.age_ttl, l.age_ttl) {
      (Some(a), Some(b)) => format!(
        ", clock {:.0}° → {:.0}° (ages {a:.2} → {b:.2} TTL)",
        180.0 * a.clamp(0.0, 1.0),
        180.0 * b.clamp(0.0, 1.0)
      ),
      _ => String::new(),
    };
    format!(
      "[observer] hue frame {i}: display {:.0}° → {:.0}°{render}{truth}{clock} over {} lit samples ({dark} dark, {highlight} highlight; {faint} faint in the pyramid, {haze} haze: render/truth τ beyond 2×), render chroma over the truth's {} ({judged} judged)",
      t(f.hue_deg),
      t(l.hue_deg),
      lit.len(),
      if chroma_min.is_finite() {
        format!("median {chroma_median:.2} min {chroma_min:.2}")
      } else {
        "n/a".to_string()
      }
    )
  };
  (line, problems)
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
  /// the truth's and the render's peak pixels and the nucleus radius on screen (px)
  peak_px: (f64, f64),
  render_peak_px: (f64, f64),
  nucleus_px_radius: f64,
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
  /// Σ render / Σ truth of the radiance-factor images (absolute photometry; 1 = closure)
  abs_ratio: f64,
  /// `--afrho-aperture-km`: the aperture Afρ (cm) of the render, of the truth (drawn through
  /// the renderer's direction bandwidth) and the recipe's analytic `Afρ(r_h)·Φ(α)` (isotropic
  /// geometry; the bandwidth leaks a fixed fraction of a 1/r coma past any aperture)
  afrho: Option<(f64, f64, f64)>,
  /// the hue samples along `McView::hue_ray`, placed over the truth's region, and per sample
  /// the truth's radiance-weighted mean age and age spread (s) and its mean linear colour (the
  /// grains' age colours mixed by radiance) in a 5×5 box, `None` where the truth is empty
  hue_px: Vec<(f64, f64)>,
  hue_truth: Vec<Option<(f64, f64, [f64; 3])>>,
  /// per hue sample: the render's over the truth's τ in the 5×5 box (unit-sum images)
  hue_tau_ratio: Vec<Option<f64>>,
  /// the render's τ outside the truth's region over its total: the haze the τ check does not
  /// judge (the old tiers' pooled packets spread over a near frame)
  haze_frac: f64,
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
      "{{\"frame\":{},\"epoch\":\"{}\",\"epoch_target\":\"{}\",\"since_ignition_s\":{},\"seek_ok\":{},\"r_helio_au\":{:.6},\"comet_au\":{},\"tiers\":[{}],\"coma_radius_km\":{:.3},\"camera\":{{\"pos_au\":{},\"quat_xyzw\":[{:.9},{:.9},{:.9},{:.9}],\"half_height_au\":{:.9},\"half_height_km\":{:.1},\"m_per_px_nucleus\":{:.6e}}},\"nucleus_px\":[{:.2},{:.2}],\"nucleus_rgb\":[{},{},{}],\"mean_luminance\":{:.4},\"p99_luminance\":{:.2},\"nonblack_fraction\":{:.6},\"saturated_fraction\":{},\"core_median\":{},\"annulus_median\":{},\"core_over_annulus\":{},\"mc\":{},\"specks\":{},\"exposure\":{},\"hue_ramp\":[{}],\"render_task\":{},\"render_ms\":{:.2},\"dust_gpu\":{},\"dust_profile\":{},\"dust_budget\":{},\"dust_log\":\"{}\"}}",
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
      self.m_per_px_nucleus,
      self.nucleus_px.0,
      self.nucleus_px.1,
      self.nucleus_rgb.0,
      self.nucleus_rgb.1,
      self.nucleus_rgb.2,
      self.mean_lum,
      self.p99_lum,
      self.nonblack_frac,
      json_f(self.flow_metric.saturated_fraction),
      json_f(self.flow_metric.core_median),
      json_f(self.flow_metric.annulus_median),
      json_f(self.flow_metric.core_over_annulus),
      self
        .mc
        .as_ref()
        .map(|m| format!(
          "{{\"rel_rms\":{:.4},\"rel_rms_judged\":{:.4},\"noise\":{:.4},\"core_px\":{},\"ripple_render\":{:.4},\"ripple_truth\":{:.4},\"region_px\":{},\"grains_px\":{:.1},\"ratio_p05\":{:.3},\"ratio_p95\":{:.3},\"nucleus_ok\":{},\"grains\":{},\"batches\":{},\"abs_ratio\":{:.4},\"haze_frac\":{:.4},\"afrho_cm\":{}}}",
          m.rel_rms, m.rel_rms_judged, m.noise, m.core_px, m.ripple_render, m.ripple_truth, m.region_px, m.grains_px, m.ratio_p05, m.ratio_p95, m.nucleus_ok, m.grains, m.batches, m.abs_ratio, m.haze_frac,
          m.afrho.map(|(r, t, a)| format!("{{\"render\":{r:.4},\"truth\":{t:.4},\"analytic\":{a:.4}}}")).unwrap_or_else(|| "null".to_string())
        ))
        .unwrap_or_else(|| "null".to_string()),
      self
        .specks
        .map(|k| format!(
          "{{\"d_max_m\":{},\"specks\":{},\"specks_overflow\":{},\"n_eye_m3\":{}}}",
          json_f(k.d_max_m as f64),
          k.specks,
          k.overflow,
          json_f(k.n_eye_m3)
        ))
        .unwrap_or_else(|| "null".to_string()),
      self
        .exposure
        .map(|e| {
          format!(
            "{{\"white_rf\":{:.4e},\"p50_rf\":{:.4e},\"mag_arcsec2\":{:.3},\"ev\":{:.3},\"softening\":{:.4e},\"fixed\":{}}}",
            e.white, e.p50, e.mag_arcsec2, e.ev, e.softening, e.fixed
          )
        })
        .unwrap_or_else(|| "null".to_string()),
      self
        .hue
        .iter()
        .map(|s| {
          let opt = |v: Option<f64>| v.map(|x| format!("{x:.4}")).unwrap_or_else(|| "null".to_string());
          format!(
            "{{\"px\":[{:.1},{:.1}],\"level\":{:.4},\"hue_deg\":{:.1},\"chroma_rel\":{:.3},\"dark\":{},\"faint\":{},\"highlight\":{},\"age_ttl\":{},\"age_spread_ttl\":{},\"truth_hue_deg\":{},\"truth_chroma_rel\":{},\"pyr_hue_deg\":{},\"pyr_chroma_rel\":{},\"pyr_tau\":{:.1},\"tau_ratio\":{}}}",
            s.px.0, s.px.1, s.level, s.hue_deg, s.chroma_rel, s.dark, s.faint, s.highlight, opt(s.age_ttl), opt(s.age_spread_ttl), opt(s.truth_hue_deg), opt(s.truth_chroma_rel), opt(s.pyr_hue_deg), opt(s.pyr_chroma_rel), s.pyr_tau, opt(s.tau_ratio.filter(|r| r.is_finite()))
          )
        })
        .collect::<Vec<_>>()
        .join(","),
      self.render_task,
      self.render_ms,
      self.dust_gpu.map_or(String::from("null"), |t| format!(
        "{{\"propagate_ms\":{},\"splat_ms\":{},\"splat_only_ms\":{},\"specks_ms\":{},\"call_ms\":[{}],\"measure_ms\":{},\"composite_ms\":{},\"total_ms\":{},\"total_ema_ms\":{},\"frame\":{},\"width\":{},\"height\":{},\"reused\":{}}}",
        json_f(t.propagate_ms),
        json_f(t.splat_ms),
        json_f(t.splat_only_ms),
        json_f(t.specks_ms),
        t.call_ms.iter().map(|&c| json_f(c)).collect::<Vec<_>>().join(","),
        json_f(t.measure_ms),
        json_f(t.composite_ms),
        json_f(t.total_ms),
        json_f(t.total_ema_ms),
        t.frame,
        t.width,
        t.height,
        t.reused
      )),
      self.dust_profile.as_ref().map_or(String::from("null"), dust_profile_json),
      self.dust_budget,
      self.passes_logged.replace('"', "'"),
    )
  }
}

// ───────────────────────────── the dust profiler ─────────────────────────────

/// `AETHERVK_DUST_PROFILE=1`: the `.prof` dust shaders and their counters (`dust_profile.glsl`)
fn dust_profile_on() -> bool {
  std::env::var("AETHERVK_DUST_PROFILE").is_ok_and(|v| v.trim() == "1")
}

/// the GPU's SM clock (MHz) from `nvidia-smi`, sampled once (`None` without it); converts the
/// profiler's cycles to time. `AETHERVK_GPU_SM_MHZ` overrides it.
fn sm_clock_mhz() -> Option<f64> {
  static MHZ: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
  *MHZ.get_or_init(|| {
    if let Some(v) = std::env::var("AETHERVK_GPU_SM_MHZ").ok().and_then(|v| v.trim().parse::<f64>().ok()) {
      return Some(v);
    }
    let out = std::process::Command::new("nvidia-smi")
      .args(["--query-gpu=clocks.sm", "--format=csv,noheader,nounits"])
      .output()
      .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse::<f64>().ok()
  })
}

/// the GPU's streaming multiprocessors (15 on a GTX 1070; 64 resident warps each at most on
/// Pascal) for the occupancy reading; `AETHERVK_GPU_SMS` overrides
fn gpu_sms() -> f64 {
  std::env::var("AETHERVK_GPU_SMS").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(15.0)
}

/// The exclusive stage cycles of a profiled splat frame (`dust_profile.glsl`): the timers nest,
/// `split` holds every leaf-level stage (project, scprep ⊃ qput, ewa, vplan, vdraw), `packet`
/// holds cone and split, `subpk` holds packet, `thread` holds setup, subpk and tracer.
/// Returns (name, exclusive cycles, per-unit name, units) in the printing order.
fn dust_profile_exclusive(p: &aethervk_core_rlib::scene::dust::DustProfileCounters) -> Vec<(&'static str, u64, &'static str, u64)> {
  use aethervk_core_rlib::scene::dust as d;
  let s = |name: &str| p.stage(d::DUST_PROF_STAGE_NAMES.iter().position(|n| *n == name).unwrap());
  let threads = p.word(d::DUST_PROF_THREADS) as u64;
  let packets = p.word(d::DUST_PROF_PACKETS) as u64;
  let cone = p.word(d::DUST_PROF_CONE) as u64;
  let leaves = p.word(d::DUST_PROF_LEAVES) as u64;
  let records = p.word(d::DUST_PROF_RECORDS) as u64;
  let volume = p.word(d::DUST_PROF_VOLUME) as u64;
  let nodes = p.word(d::DUST_PROF_NODES) as u64;
  let ewa_texels = p.u64_at(d::DUST_PROF_EWA_TEXELS);
  let volume_texels = p.u64_at(d::DUST_PROF_VOLUME_TEXELS);
  let leafsum = s("project") + s("scprep") + s("ewa") + s("vplan") + s("vdraw");
  // the enumerator's stages (dust_splat.comp: setup subpk ecount tracer, within `thread`) and the
  // splitter's (dust_split.comp: count packet cone split project scprep qput vemit ewa vplan
  // vdraw, within its own lane cycles)
  vec![
    ("setup", s("setup"), "cluster", threads),
    ("subpk", s("subpk"), "packet", packets),
    ("ecount", s("ecount"), "lane (the record-counting pass)", threads),
    ("packet'", s("packet").saturating_sub(s("cone") + s("split")), "packet", packets),
    ("cone", s("cone"), "cone packet", cone),
    ("split'", s("split").saturating_sub(leafsum), "node", nodes),
    ("project", s("project"), "leaf", leaves),
    ("scprep'", s("scprep").saturating_sub(s("qput")), "leaf", leaves),
    ("qput", s("qput"), "record", records),
    ("vemit", s("vemit"), "volume packet", volume),
    ("ewa", s("ewa"), "inline texel", ewa_texels),
    ("vplan", s("vplan"), "volume packet", volume),
    ("vdraw", s("vdraw"), "volume texel", volume_texels),
    ("tracer", s("tracer"), "cluster", threads),
    ("count", s("count"), "record (the splitter's seed-counting pass)", packets),
    ("(rest)", s("thread").saturating_sub(s("setup") + s("subpk") + s("tracer") + s("ecount")), "cluster", threads),
  ]
}

/// The profile report lines (`[observer] dust profile ...`): the stages' shares and per-unit
/// cycles, the thread tail from the histogram, the perfect-occupancy work against the measured
/// splat time, the queue draw's split.
fn dust_profile_report(
  p: &aethervk_core_rlib::scene::dust::DustProfileCounters,
  t: Option<&aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings>,
  mhz: Option<f64>,
) -> Vec<String> {
  use aethervk_core_rlib::scene::dust as d;
  let mut out = Vec::new();
  let thread = p.stage(0);
  let threads = p.word(d::DUST_PROF_THREADS) as u64;
  let hz = mhz.map(|m| m * 1e6);
  let ms = |cycles: u64| hz.map_or(String::from("?"), |hz| format!("{:.1}", cycles as f64 / hz * 1e3));
  out.push(format!(
    "[observer] dust profile (frame {}, {} threads, {} packets, {} leaves, {} volume packets, {} queue records; SM clock {}):",
    p.frame,
    threads,
    p.word(d::DUST_PROF_PACKETS),
    p.word(d::DUST_PROF_LEAVES),
    p.word(d::DUST_PROF_VOLUME),
    p.word(d::DUST_PROF_RECORDS),
    mhz.map_or(String::from("unknown"), |m| format!("{m:.0} MHz"))
  ));
  out.push(format!("[observer]   {:<9} {:>13} {:>7} {:>10} {:>12}  per unit", "stage", "cycles", "share", "/thread", "cycles/unit"));
  for (name, cycles, unit, units) in dust_profile_exclusive(p) {
    out.push(format!(
      "[observer]   {:<9} {:>13} {:>6.1}% {:>10.0} {:>12}  {}",
      name,
      cycles,
      if thread > 0 { 100.0 * cycles as f64 / thread as f64 } else { 0.0 },
      if threads > 0 { cycles as f64 / threads as f64 } else { 0.0 },
      if units > 0 { format!("{:.0}", cycles as f64 / units as f64) } else { String::from("-") },
      unit
    ));
  }
  // the tail: the histogram's log2 buckets
  let hist: Vec<u64> = (0..32).map(|k| p.word(d::DUST_PROF_HIST + k) as u64).collect();
  let total: u64 = hist.iter().sum();
  let quantile = |q: f64| -> u32 {
    let target = (q * total as f64).ceil() as u64;
    let mut acc = 0u64;
    for (k, &n) in hist.iter().enumerate() {
      acc += n;
      if acc >= target.max(1) {
        return k as u32;
      }
    }
    31
  };
  out.push(format!(
    "[observer]   thread    {:>13}  100.0% {:>10.0}  tail: p50 ≥2^{} p90 ≥2^{} p99 ≥2^{} p99.9 ≥2^{} cycles, slowest {} cycles ({} ms)",
    thread,
    if threads > 0 { thread as f64 / threads as f64 } else { 0.0 },
    quantile(0.5),
    quantile(0.9),
    quantile(0.99),
    quantile(0.999),
    p.word(d::DUST_PROF_MAX_THREAD),
    ms(p.word(d::DUST_PROF_MAX_THREAD) as u64)
  ));
  // the occupancy reading: a thread's cycles are wall cycles while it is resident (the
  // subgroup clock runs while other warps issue), so Σ thread cycles / (f × the splats' time) is
  // the average number of resident threads over the dispatches: per SM, in warps of 32, against
  // Pascal's 64 — the register-limited residency (65536 / (registers × 32) warps per SM)
  let sms = gpu_sms();
  match (hz, t) {
    (Some(hz), Some(t)) if t.splat_only_ms.is_finite() && t.splat_only_ms > 0.0 => {
      let resident = thread as f64 / (hz * t.splat_only_ms * 1e-3);
      let life_ms = if threads > 0 { thread as f64 / threads as f64 / hz * 1e3 } else { 0.0 };
      out.push(format!(
        "[observer]   occupancy: Σ thread cycles / ({:.0} MHz × {:.1} ms splats) = {resident:.0} resident threads on average = {:.1} warps per SM ({sms:.0} SMs; 64 at most); mean thread life {life_ms:.2} ms, {} threads",
        hz / 1e6,
        t.splat_only_ms,
        resident / 32.0 / sms,
        threads
      ));
    }
    _ => out.push(String::from("[observer]   occupancy: no SM clock or no splat timing (set AETHERVK_GPU_SM_MHZ)")),
  }
  // the splitter (dust_split.comp): its own lane cycles against the enumerator's `thread`
  let sthread = p.u64_at(d::DUST_PROF_STHREAD);
  if sthread > 0 {
    let slanes = p.word(d::DUST_PROF_SLANES) as u64;
    out.push(format!(
      "[observer]   splitter: {sthread} lane-cycles ({:.2}× the enumerator's), count {:.1}% packet {:.1}%; {slanes} lanes, slowest {} cycles ({} ms){}",
      if thread > 0 { sthread as f64 / thread as f64 } else { 0.0 },
      100.0 * p.stage(d::DUST_PROF_STAGE_NAMES.iter().position(|n| *n == "count").unwrap()) as f64 / sthread as f64,
      100.0 * p.stage(d::DUST_PROF_STAGE_NAMES.iter().position(|n| *n == "packet").unwrap()) as f64 / sthread as f64,
      p.word(d::DUST_PROF_SMAX),
      ms(p.word(d::DUST_PROF_SMAX) as u64),
      match (hz, t) {
        (Some(hz), Some(t)) if t.splat_only_ms > 0.0 => format!(
          "; {:.0} resident lanes on average over the splats' {:.1} ms",
          sthread as f64 / (hz * t.splat_only_ms * 1e-3),
          t.splat_only_ms
        ),
        _ => String::new(),
      }
    ));
  }
  // the volume consumer (dust_draw_volume.comp)
  let cthread = p.u64_at(d::DUST_PROF_CTHREAD);
  if cthread > 0 {
    let cpackets = p.word(d::DUST_PROF_CPACKETS) as u64;
    let ctexels = p.u64_at(d::DUST_PROF_CTEXELS);
    let cvplan = p.u64_at(d::DUST_PROF_CVPLAN);
    let cvdraw = p.u64_at(d::DUST_PROF_CVDRAW);
    out.push(format!(
      "[observer]   volume consumer: {cthread} lane-cycles, cvplan {:.1}% ({:.0} per packet), cvdraw {:.1}% ({:.0} per texel); {cpackets} packets, {ctexels} texels, slowest lane {} cycles ({} ms){}",
      100.0 * cvplan as f64 / cthread as f64,
      if cpackets > 0 { cvplan as f64 / cpackets as f64 } else { 0.0 },
      100.0 * cvdraw as f64 / cthread as f64,
      if ctexels > 0 { cvdraw as f64 / ctexels as f64 } else { 0.0 },
      p.word(d::DUST_PROF_CMAX),
      ms(p.word(d::DUST_PROF_CMAX) as u64),
      match (hz, t) {
        (Some(hz), Some(t)) if t.splat_only_ms > 0.0 => format!(
          "; {:.0} resident lanes on average over the splats' {:.1} ms",
          cthread as f64 / (hz * t.splat_only_ms * 1e-3),
          t.splat_only_ms
        ),
        _ => String::new(),
      }
    ));
  }
  // K1c's parts of the volumetric draw (all shaders): pass A's per-class sums, a small piece's
  // depth points, its texels' rectangle integrals, the sub-ray and line texels
  let k1c: Vec<u64> = (0..d::DUST_PROF_K1C_NAMES.len()).map(|k| p.u64_at(d::DUST_PROF_K1C0 + 2 * k)).collect();
  if k1c.iter().any(|&c| c > 0) {
    out.push(format!(
      "[observer]   K1c parts (lane-cycles): {}",
      d::DUST_PROF_K1C_NAMES.iter().zip(&k1c).map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", ")
    ));
  }
  let (sm_fast, sm_edge) = (p.word(d::DUST_PROF_SMALL_PATHS), p.word(d::DUST_PROF_SMALL_PATHS + 1));
  if sm_fast > 0 || sm_edge > 0 {
    out.push(format!("[observer]   small pieces by one projection a slice {sm_fast}, by 4 sigma points {sm_edge}"));
  }
  // the queue draw
  let q: Vec<u64> = (0..4).map(|k| p.qstage(k)).collect();
  let qrecords = p.word(d::DUST_PROF_QRECORDS) as u64;
  let qtexels = p.u64_at(d::DUST_PROF_QTEXELS);
  if q[3] > 0 {
    let share = |c: u64| 100.0 * c as f64 / q[3] as f64;
    out.push(format!(
      "[observer]   queue draw: qload {:.1}% qrows' {:.1}% qseg {:.1}% of {} lane-cycles; {} records, {} texels ({:.0} per record, {:.0} cycles per texel in qseg), slowest workgroup {} cycles ({} ms); {}queue_ms measured",
      share(q[0]),
      share(q[1].saturating_sub(q[2])),
      share(q[2]),
      q[3],
      qrecords,
      qtexels,
      if qrecords > 0 { qtexels as f64 / qrecords as f64 } else { 0.0 },
      if qtexels > 0 { q[2] as f64 / qtexels as f64 } else { 0.0 },
      p.word(d::DUST_PROF_QMAX_WG),
      ms(p.word(d::DUST_PROF_QMAX_WG) as u64),
      t.map_or(String::new(), |t| format!("{:.1} ms ", t.queue_ms))
    ));
  }
  out
}

/// the counters as JSON for `stats.jsonl`
fn dust_profile_json(p: &aethervk_core_rlib::scene::dust::DustProfileCounters) -> String {
  use aethervk_core_rlib::scene::dust as d;
  let stages: Vec<String> = d::DUST_PROF_STAGE_NAMES
    .iter()
    .enumerate()
    .map(|(k, n)| format!("\"{n}\":{}", p.stage(k)))
    .collect();
  let excl: Vec<String> = dust_profile_exclusive(p)
    .iter()
    .map(|(n, c, _, _)| format!("\"{}\":{c}", n.replace('\'', "_x").replace("(rest)", "rest")))
    .collect();
  let qstages: Vec<String> = d::DUST_PROF_QSTAGE_NAMES
    .iter()
    .enumerate()
    .map(|(k, n)| format!("\"{n}\":{}", p.qstage(k)))
    .collect();
  let hist: Vec<String> = (0..32).map(|k| p.word(d::DUST_PROF_HIST + k).to_string()).collect();
  format!(
    "{{\"frame\":{},\"stages\":{{{}}},\"exclusive\":{{{}}},\"threads\":{},\"max_thread\":{},\"packets\":{},\"cone\":{},\"leaves\":{},\"records\":{},\"volume\":{},\"nodes\":{},\"ewa_texels\":{},\"volume_texels\":{},\"hist\":[{}],\"queue\":{{{},\"records\":{},\"texels\":{},\"max_wg\":{}}}}}",
    p.frame,
    stages.join(","),
    excl.join(","),
    p.word(d::DUST_PROF_THREADS),
    p.word(d::DUST_PROF_MAX_THREAD),
    p.word(d::DUST_PROF_PACKETS),
    p.word(d::DUST_PROF_CONE),
    p.word(d::DUST_PROF_LEAVES),
    p.word(d::DUST_PROF_RECORDS),
    p.word(d::DUST_PROF_VOLUME),
    p.word(d::DUST_PROF_NODES),
    p.u64_at(d::DUST_PROF_EWA_TEXELS),
    p.u64_at(d::DUST_PROF_VOLUME_TEXELS),
    hist.join(","),
    qstages.join(","),
    p.word(d::DUST_PROF_QRECORDS),
    p.u64_at(d::DUST_PROF_QTEXELS),
    p.word(d::DUST_PROF_QMAX_WG),
  )
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

/// The app's IAU model as the Comet tab commits it (`CometTabViewModel` →
/// `BodyRotationalModelDto`: pole rates 0, body-fixed): pole RA / Dec (deg, J2000), prime
/// meridian W0 (deg), sidereal rate (deg/day).
fn rotation_model(pole_ra: f64, pole_dec: f64, w0: f64, rate_deg_day: f64) -> BodyRotationalModel {
  BodyRotationalModel {
    pole_ra,
    pole_dec,
    prime_meridian: w0,
    pole_ra_rate: 0.0,
    pole_dec_rate: 0.0,
    rotation_rate: rate_deg_day,
    body_fixed_orientation: true,
  }
}

/// `avkSimulationContext_setBodyRotationalModel` with the default pole (RA 0, Dec 90, W0 0) and
/// a `period_h` spin.
fn set_rotational_model(ctx: &SimulationContext, scene_id: u64, body: EntityId, period_h: f64) {
  set_rotation(ctx, scene_id, body, rotation_model(0.0, 90.0, 0.0, 360.0 * 24.0 / period_h));
}

/// `avkSimulationContext_setBodyRotationalModel`: scene component + cartesian cache + re-emit.
fn set_rotation(ctx: &SimulationContext, scene_id: u64, body: EntityId, model: BodyRotationalModel) {
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

fn jet_params(seed: u32, isotropic: bool) -> ParticleSystemEmitParams {
  let mut p = ParticleSystemEmitParams {
    latitude_rad: 20.0_f32.to_radians(),
    longitude_rad: 0.0,
    aperture_rad: if isotropic {
      core::f32::consts::PI
    } else {
      0.5
    },
    start_velocity_mean: 2.0,
    start_velocity_std: 0.5,
    seed,
    diametre_um: 100.0,
    density_gcm3: 0.533,
    scattering_efficiency: 1.0,
    // `OBSERVER_ALBEDO=<p>` (diagnostics): 0 turns the radiance weight off on both sides (plain τ)
    geometric_albedo: std::env::var("OBSERVER_ALBEDO")
      .ok()
      .and_then(|v| v.trim().parse().ok())
      .unwrap_or(0.04),
    size_q: 3.5,
    size_range: 10.0,
    dust_rate_1au_kgs: 0.0,
    dust_rate_power: 2.0,
    dust_rate_cutoff_au: 15.0,
    dust_rate_max_kgs: 0.0,
  };
  // the activity input is a dust rate (v9); the observer's reference recipe stays the one an Afρ
  // of 100 cm at 1 AU (max 100 000 cm) converts to, so every capture keeps its physics (with
  // `OBSERVER_ALBEDO=0` the conversion is 0: no emission, as before)
  p.dust_rate_1au_kgs = p.dust_rate_from_afrho_kgs(100.0);
  p.dust_rate_max_kgs = p.dust_rate_from_afrho_kgs(100_000.0);
  p
}

/// `--no-trajectories`: `HiddenComponent` on every entity drawing a trajectory line
/// (`TrajectoryComponent`, the effective path's `EffectiveTrajectoryComponent`; dedicated entities,
/// the bodies are skipped so a body is never hidden) and the reference-error annotations off
/// (`set_reference_error_visible`, the app's toggle). Re-run each frame: the logic thread adds
/// tracks after the setup. Returns how many entities it newly hid.
fn hide_trajectories(ctx: &SimulationContext, scene_id: u64, keep: &[EntityId]) -> usize {
  use aethervk_core_rlib::scene::{
    HiddenComponent,
    trajectory::{EffectiveTrajectoryComponent, TrajectoryComponent},
  };
  ctx.set_reference_error_visible(scene_id, false);
  let Some(scene_arc) = ctx.get_scene(scene_id) else {
    return 0;
  };
  let g = scene_arc.read();
  let mut ids = Vec::new();
  g.scene.query1(|id, _: &TrajectoryComponent| ids.push(id));
  g.scene.query1(|id, _: &EffectiveTrajectoryComponent| ids.push(id));
  // the dust trajectory lines (`--view-flags 4`) are a dust view aid, not an orbit line: kept
  let mut dust_lines = Vec::new();
  g.scene.query1(
    |id, _: &aethervk_core_rlib::scene::trajectory::DustTrajectoryLinesComponent| {
      dust_lines.push(id)
    },
  );
  ids.retain(|id| !dust_lines.contains(id));
  ids.sort_unstable_by_key(|id| id.as_ffi());
  ids.dedup();
  let mut n = 0;
  for id in ids.into_iter().filter(|id| !keep.contains(id)) {
    if g.scene.with_component(id, |_: &HiddenComponent| ()).is_none()
      && g.scene.add_component(id, HiddenComponent {}).is_ok()
    {
      n += 1;
    }
  }
  n
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
/// extra render generations a frame may wait for its exposure to settle (see the frame loop)
const EXPOSURE_SETTLE_MAX: u32 = 24;
/// extra render generations a frame may wait for the work feedback (split depth, volume and EWA
/// biases) to settle after a cold start or a jump inside the coma
const WORK_SETTLE_MAX: u32 = 12;

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

/// `--play-live`: luminance of a 32×32 grid of cells (mean luma per cell, 0..1) of an RGBA frame
fn luma_grid_32(rgba: &[u8], width: u32, height: u32) -> Vec<f64> {
  const N: usize = 32;
  let mut sum = vec![0.0f64; N * N];
  let mut cnt = vec![0u32; N * N];
  for y in 0..height as usize {
    let cy = (y * N / height as usize).min(N - 1);
    for x in 0..width as usize {
      let cx = (x * N / width as usize).min(N - 1);
      let i = (y * width as usize + x) * 4;
      let l = 0.2126 * rgba[i] as f64 + 0.7152 * rgba[i + 1] as f64 + 0.0722 * rgba[i + 2] as f64;
      sum[cy * N + cx] += l / 255.0;
      cnt[cy * N + cx] += 1;
    }
  }
  sum.iter().zip(&cnt).map(|(s, c)| if *c > 0 { s / *c as f64 } else { 0.0 }).collect()
}

/// The per-stage GPU ms of one dust frame (`DustGpuTimings`) as a JSON object (`NaN` → null)
fn dust_stages_json(t: &aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings) -> String {
  let num = |v: f64| if v.is_finite() { format!("{v:.2}") } else { String::from("null") };
  format!(
    "{{\"frame\":{},\"reused\":{},\"propagate_ms\":{},\"splat_ms\":{},\"splat_only_ms\":{},\"specks_ms\":{},\"queue_ms\":{},\"cadence_ms\":{},\"classify_ms\":{},\"builds_ms\":{},\"reproject_ms\":{},\"measure_ms\":{},\"composite_ms\":{},\"total_ms\":{},\"call_ms\":[{}]}}",
    t.frame,
    t.reused,
    num(t.propagate_ms),
    num(t.splat_ms),
    num(t.splat_only_ms),
    num(t.specks_ms),
    num(t.queue_ms),
    num(t.cadence_ms),
    num(t.classify_ms),
    num(t.builds_ms),
    num(t.reproject_ms),
    num(t.measure_ms),
    num(t.composite_ms),
    num(t.total_ms),
    t.call_ms.iter().filter(|c| c.is_finite()).map(|c| format!("{c:.2}")).collect::<Vec<_>>().join(",")
  )
}

/// [`dust_stages_json`] for the log line
fn dust_stages_brief(t: &aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings) -> String {
  let num = |v: f64| if v.is_finite() { format!("{v:.1}") } else { String::from("-") };
  format!(
    "frame {} total {} = propagate {} + splat {} (fresh splat {}, specks {}, queue {}; cadence {}, classify {}, builds {}, reproject {}) + measure {} + composite {}",
    t.frame,
    num(t.total_ms),
    num(t.propagate_ms),
    num(t.splat_ms),
    num(t.splat_only_ms),
    num(t.specks_ms),
    num(t.queue_ms),
    num(t.cadence_ms),
    num(t.classify_ms),
    num(t.builds_ms),
    num(t.reproject_ms),
    num(t.measure_ms),
    num(t.composite_ms)
  )
}

/// `--play-live SECONDS --sample-s S`: the reviewer's "the animation doesn't work" probe. After
/// the initial seek (done, caught up) frame 0 is captured paused; then the simulation is started
/// at `--step-hours` sim hours per wall second and, WITHOUT pausing, a frame is captured every
/// `S` wall seconds until `SECONDS` elapse, through the observer's usual capture path
/// (`wait_and_download`: one fresh render generation, then the swapchain download; the render
/// thread keeps rendering while the clock runs, the download reads the finished task's image, so
/// no pause is needed — the app's viewport does the same). Per frame: `live_NNN.png` and one
/// `stats.jsonl` line with the wall time, the sim epoch and its advance, the per-tier history
/// (live/cap, caught-up, ages: the logic thread's `[Dust]` line), the host tick state, the dust
/// frame ledger since the previous capture (drawn vs reused dust frames, the last frame's sim
/// time) and a motion metric against the previous frame: the fraction of the 32×32 luminance
/// grid cells that changed by more than 0.01 (`moved`: fraction > 0.02). Ends with
/// `PLAY_LIVE moved N/M frames`; exit 0 when at least one frame moved, 7 otherwise.
#[allow(clippy::too_many_arguments)]
fn play_live_probe(
  ctx: &SimulationContext,
  scene_id: u64,
  pe: aethervk_core_rlib::gpu::PresentationEngineHandle,
  cam: EntityId,
  comet_body: EntityId,
  args: &Args,
  live_s: f64,
  pose_for: &dyn Fn(V3) -> (V3, Q),
  send_inside_projection: &dyn Fn(f64),
  inside_far_now: &mut f64,
) -> i32 {
  let mut stats = File::create(args.out.join("stats.jsonl")).expect("stats.jsonl");
  let sample_s = args.sample_s.max(0.05);
  let speed = args.step_hours * 3600.0;
  let n_samples = (live_s / sample_s).floor() as usize + 1;
  log_line(&format!(
    "play-live: {live_s} s at {:.1} sim h per wall s (×{speed:.0}), a frame every {sample_s} s ({n_samples} frames)",
    args.step_hours
  ));
  let epoch0 = current_epoch(ctx, scene_id);
  let t0 = Instant::now();
  let mut prev_grid: Option<Vec<f64>> = None;
  let mut prev_epoch = epoch0;
  let mut prev_wall = 0.0f64;
  let mut moved_n = 0usize;
  let mut captured = 0usize;
  let mut last_seq = 0u64;
  let mut started = false;
  let mut exit_code = 0;
  for i in 0..n_samples {
    if i == 1 && args.play_live_paused {
      started = true;
      log_line(&format!(
        "play-live: paused control: a seek of {:.3} sim h per sample, the clock never runs",
        sample_s * args.step_hours
      ));
    } else if i == 1 {
      // frame 0 captured paused and caught up; from here the clock runs and is never paused
      started = ctx.start_simulation(
        scene_id,
        aethervk_oshal_rlib::os::time::v2::SimSpeed::Custom(speed),
      );
      log_line(&format!("play-live: start_simulation ×{speed:.0}: {started}"));
      if !started {
        exit_code = 7;
      }
    }
    if i >= 1 && args.play_live_paused {
      // the paused control: the same sim step the play would have made by this sample
      if let Some(e0) = epoch0 {
        let target = e0 + Duration::from_seconds(i as f64 * sample_s * args.step_hours * 3600.0);
        let ok = ctx.seek_epoch_sync(scene_id, target);
        let caught = wait_caught_up(ctx, scene_id, 60.0);
        if !(ok && caught) {
          log_line(&format!("play-live frame {i}: paused seek to {target}: ok {ok} caught up {caught}"));
        }
      }
    }
    // the sample's wall time (the capture itself takes a render generation)
    let due = i as f64 * sample_s;
    let wait = due - t0.elapsed().as_secs_f64();
    if wait > 0.0 {
      std::thread::sleep(std::time::Duration::from_secs_f64(wait));
    }
    if DEVICE_LOST.load(Ordering::Relaxed) {
      log_line("play-live: GPU device lost, aborting");
      exit_code = 7;
      break;
    }
    // the camera follows the comet (as the frame loop does; the pose is the same offset)
    let comet = comet_position_au(ctx, scene_id, comet_body).unwrap_or([0.0; 3]);
    if args.pose == Pose::Inside {
      let coma_km = ctx.dust_coma_radius_km(scene_id);
      let far = {
        // `inside_far_au` of the frame loop, inlined (the closure's arguments live in `main`)
        let inside_d_au = args.inside_m / aethervk_core_rlib::scene::dust::AU_M;
        (1e4 * inside_d_au).max(2.0 * coma_km / AU_KM)
      };
      if far != *inside_far_now {
        *inside_far_now = far;
        send_inside_projection(far);
      }
    }
    if args.pose != Pose::Earth {
      let (p, q) = pose_for(comet);
      set_camera_pose(ctx, scene_id, cam, p, q);
    }
    let t_cap = Instant::now();
    let wall_s = t0.elapsed().as_secs_f64();
    let dl = wait_and_download(ctx, args.width, args.height, 1, 20_000);
    let Some((mut buf, tid)) = dl else {
      log_line(&format!("play-live frame {i}: no frame downloaded within 20 s"));
      let _ = writeln!(
        stats,
        "{{\"frame\":{i},\"wall_s\":{wall_s:.3},\"error\":\"no frame downloaded\"}}"
      );
      exit_code = 7;
      continue;
    };
    let capture_ms = t_cap.elapsed().as_secs_f64() * 1e3;
    bgra_to_rgba(&mut buf);
    let epoch = current_epoch(ctx, scene_id);
    let sim_since_start_h = match (epoch, epoch0) {
      (Some(e), Some(e0)) => (e - e0).to_seconds() / 3600.0,
      _ => f64::NAN,
    };
    let sim_dt_h = match (epoch, prev_epoch) {
      (Some(e), Some(p)) => (e - p).to_seconds() / 3600.0,
      _ => f64::NAN,
    };
    let expected_dt_h = if i == 0 { 0.0 } else { (wall_s - prev_wall) * args.step_hours };
    // dust history per tier (what the logic thread's `[Dust]` line prints)
    let tiers = ctx.dust_stats(scene_id);
    let history: Vec<String> = tiers
      .iter()
      .map(|t| {
        format!(
          "{}/{} {:.3}-{:.3} d{}{}{}",
          t.live_clusters,
          t.capacity,
          t.youngest_age_s / 86400.0,
          t.oldest_age_s / 86400.0,
          if t.caught_up { "" } else { " (building)" },
          if t.unlit_windows > 0 { format!(" ({} unlit)", t.unlit_windows) } else { String::new() },
          if t.jet_unavailable { " (jet unavailable)" } else { "" }
        )
      })
      .collect();
    let tiers_json: Vec<String> = tiers
      .iter()
      .map(|t| {
        format!(
          "{{\"live_clusters\":{},\"capacity\":{},\"youngest_age_s\":{:.1},\"oldest_age_s\":{:.1},\"caught_up\":{},\"unlit_windows\":{},\"jet_unavailable\":{}}}",
          t.live_clusters, t.capacity, t.youngest_age_s, t.oldest_age_s, t.caught_up, t.unlit_windows, t.jet_unavailable
        )
      })
      .collect();
    let (physics_busy, running) = ctx
      .get_scene(scene_id)
      .map(|s| {
        let s = s.read();
        (
          s.active_physics_task.load(Ordering::Acquire),
          s.simulation_running.load(Ordering::Acquire),
        )
      })
      .unwrap_or((false, false));
    // the dust frame ledger since the previous capture: drawn vs reused dust frames, and the
    // sim time the last dust frame evaluated the clusters at (the host tick's `jet.t_s`)
    let ledger = ctx.dust_frame_ledger(pe, last_seq);
    let ledger_new: Vec<_> = ledger.iter().filter(|l| l.seq > last_seq || last_seq == 0).collect();
    let n_reused = ledger_new.iter().filter(|l| l.reused).count();
    let n_drawn = ledger_new.len() - n_reused;
    let last = ledger.last();
    let last_reused = last.map(|l| l.reused);
    let last_t_now_s = last.map(|l| l.t_now_s);
    if let Some(l) = last {
      last_seq = l.seq;
    }
    let gpu_timings = ctx.dust_gpu_timings(pe);
    let gpu_reused = gpu_timings.map(|t| t.reused);
    // the dust GPU ms of the last frame that drew (not a reused one) and the render task's frame
    let gpu_drawn = ctx.dust_gpu_timings_drawn(pe).or(gpu_timings);
    let gpu_drawn_ms = gpu_drawn.map(|t| t.total_ms);
    // the per-stage GPU ms of that drawn frame, and the budget loop / cadence telemetry (why the
    // cadence reset, its mode, the fresh vs live clusters; a readback a few frames late)
    let stages_json = gpu_drawn.map_or(String::from("null"), |t| dust_stages_json(&t));
    let budget = ctx.dust_budget_telemetry(pe);
    let budget_json = budget.as_ref().map_or(String::from("null"), |b| b.to_json());
    let cadence_brief = budget.as_ref().map_or(String::from("cadence n/a"), |b| {
      b.cadence.as_ref().map_or(String::from("cadence off"), |c| {
        format!(
          "cadence mode {:?} reset {:?} fresh {}/{} ({:.3}) images {} admitted {:?}; loop {} rung {}/{} ε {} lod {} ÷{} regime {} fb_full [d{} v{} e{}]",
          c.mode,
          c.reset,
          c.fresh,
          c.live,
          c.fresh_fraction,
          c.images_shown,
          c.admitted,
          b.state,
          b.rung,
          b.rungs.saturating_sub(1),
          b.knobs.map_or(String::from("?"), |k| format!("{}", k.eps_px)),
          b.knobs.map_or(String::from("?"), |k| format!("{}", k.lod_px)),
          b.knobs.map_or(1, |k| 1u32 << k.budget_shift),
          b.regime,
          b.fb_full.split_depth,
          b.fb_full.volume_bias,
          b.fb_full.ewa_bias
        )
      })
    });
    let wall_dt = if i == 0 { 0.0 } else { wall_s - prev_wall };
    let dust_fps = if wall_dt > 0.0 { n_drawn as f64 / wall_dt } else { 0.0 };
    let play_quality = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PLAY_QUALITY")
      .map_or(true, |v| v.trim() != "0");
    // the sim time of the dust frame vs the clock: a lagging dust frame is a host tick not run
    let dust_t_lag_s = match (last_t_now_s, epoch, epoch0) {
      (Some(t), Some(e), Some(e0)) => {
        // the ledger's `t_now_s` is scaled sim seconds since the scenario start: compare the
        // advance since frame 0 with the clock's
        (e - e0).to_seconds() - (t - ledger_t0(&ledger, i))
      }
      _ => f64::NAN,
    };
    let grid = luma_grid_32(&buf, args.width, args.height);
    let changed_fraction = prev_grid
      .as_ref()
      .map(|p| {
        let n = p.iter().zip(&grid).filter(|(a, b)| (*a - *b).abs() > 0.01).count();
        n as f64 / grid.len() as f64
      })
      .unwrap_or(0.0);
    let moved = i > 0 && changed_fraction > 0.02;
    if i > 0 {
      captured += 1;
      if moved {
        moved_n += 1;
      }
    }
    let mean_luma = grid.iter().sum::<f64>() / grid.len() as f64;
    let img = image::RgbaImage::from_raw(args.width, args.height, buf).unwrap();
    let name = format!("live_{i:03}.png");
    if let Err(e) = img.save(args.out.join(&name)) {
      log_line(&format!("play-live frame {i}: {name}: {e}"));
    }
    let epoch_str = epoch.map_or(String::from("null"), |e| format!("\"{e}\""));
    let line = format!(
      "{{\"frame\":{i},\"wall_s\":{wall_s:.3},\"capture_ms\":{capture_ms:.1},\"epoch\":{epoch_str},\"sim_since_start_h\":{sim_since_start_h:.4},\"sim_dt_h\":{sim_dt_h:.4},\"expected_sim_dt_h\":{expected_dt_h:.4},\"running\":{running},\"started\":{started},\"physics_busy\":{physics_busy},\"history\":[{}],\"tiers\":[{}],\"dust_frames_drawn\":{n_drawn},\"dust_frames_reused\":{n_reused},\"last_dust_reused\":{},\"last_dust_t_now_s\":{},\"dust_t_lag_s\":{dust_t_lag_s:.1},\"gpu_timings_reused\":{},\"dust_gpu_drawn_ms\":{},\"dust_gpu_stages\":{stages_json},\"dust_budget\":{budget_json},\"dust_fps\":{dust_fps:.2},\"play_quality\":{play_quality},\"render_task\":{tid},\"mean_luma\":{mean_luma:.4},\"changed_fraction\":{changed_fraction:.4},\"moved\":{moved},\"png\":\"{name}\"}}",
      history.iter().map(|h| format!("\"{h}\"")).collect::<Vec<_>>().join(","),
      tiers_json.join(","),
      last_reused.map_or(String::from("null"), |b| b.to_string()),
      last_t_now_s.map_or(String::from("null"), |t| format!("{t:.1}")),
      gpu_reused.map_or(String::from("null"), |b| b.to_string()),
      gpu_drawn_ms.map_or(String::from("null"), |m| format!("{m:.2}")),
    );
    let _ = writeln!(stats, "{line}");
    log_line(&format!(
      "play-live frame {i}: wall {wall_s:.2} s, sim +{sim_since_start_h:.3} h (dt {sim_dt_h:.3} h, expected {expected_dt_h:.3} h), dust frames drawn {n_drawn} reused {n_reused} ({dust_fps:.1} dust fps, {} ms), dust t lag {dust_t_lag_s:.1} s, changed {changed_fraction:.3} moved {moved}; {}",
      gpu_drawn_ms.map_or(String::from("?"), |m| format!("{m:.1}")),
      history.join(" | ")
    ));
    log_line(&format!(
      "play-live frame {i}: stages {}; {cadence_brief}",
      gpu_drawn.map_or(String::from("n/a"), |t| dust_stages_brief(&t))
    ));
    prev_grid = Some(grid);
    prev_epoch = epoch;
    prev_wall = wall_s;
  }
  let _ = ctx.pause_simulation_sync(scene_id);
  let summary = format!(
    "PLAY_LIVE moved {moved_n}/{captured} frames, play quality {}",
    if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PLAY_QUALITY").is_some_and(|v| v.trim() == "0") { "off" } else { "on" }
  );
  log_line(&summary);
  println!("{summary}");
  let _ = writeln!(stats, "{{\"summary\":\"{summary}\",\"moved\":{moved_n},\"frames\":{captured}}}");
  let _ = stats.flush();
  if captured > 0 && moved_n == 0 && exit_code == 0 {
    exit_code = 7;
  }
  exit_code
}

/// the ledger's dust sim time at the probe's frame 0 (the first entry's, kept across the run)
fn ledger_t0(ledger: &[aethervk_core_rlib::gpu::frame::DustFrameLedger], i: usize) -> f64 {
  static T0: Mutex<Option<f64>> = Mutex::new(None);
  let mut g = T0.lock().unwrap();
  if i == 0 || g.is_none() {
    *g = ledger.last().map(|l| l.t_now_s);
  }
  g.unwrap_or(f64::NAN)
}

fn bgra_to_rgba(buf: &mut [u8]) {
  for px in buf.chunks_exact_mut(4) {
    px.swap(0, 2);
  }
}

/// The flow-pulse exposure metric of a frame: how much of the display clips and how the
/// nucleus core compares with the surrounding coma (the reviewer's "flow far too bright near
/// the nucleus" is a saturated core: a core/annulus ratio far above the plain render's).
#[derive(Clone, Copy, Debug)]
struct FlowMetric {
  /// fraction of pixels whose display luminance is ≥ 250/255 (clipped or about to)
  saturated_fraction: f64,
  /// median luminance inside ρ < 2 R_nuc,px around the nucleus pixel (NaN without pixels)
  core_median: f64,
  /// median luminance in 10 R_nuc,px ≤ ρ ≤ 50 R_nuc,px (NaN without pixels)
  annulus_median: f64,
  /// `core_median / annulus_median` (NaN when either is NaN or the annulus is black)
  core_over_annulus: f64,
}

impl FlowMetric {
  const NAN: FlowMetric = FlowMetric {
    saturated_fraction: f64::NAN,
    core_median: f64::NAN,
    annulus_median: f64::NAN,
    core_over_annulus: f64::NAN,
  };
}

fn median_f32(v: &mut [f32]) -> f64 {
  if v.is_empty() {
    return f64::NAN;
  }
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let n = v.len();
  if n % 2 == 1 {
    v[n / 2] as f64
  } else {
    (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0
  }
}

/// Mean, p99 and non-black fraction of the display luminance, plus the flow metric for a
/// nucleus at `nucleus_px` with radius `nucleus_px_radius` (None: no nucleus on screen).
fn luminance_stats(
  rgba: &[u8],
  width: u32,
  nucleus: Option<((f64, f64), f64)>,
) -> (f64, f64, f64, FlowMetric) {
  let n = rgba.len() / 4;
  let mut lums: Vec<f32> = Vec::with_capacity(n);
  let mut sum = 0.0f64;
  let mut nonblack = 0usize;
  let mut saturated = 0usize;
  let mut core: Vec<f32> = Vec::new();
  let mut annulus: Vec<f32> = Vec::new();
  let w = width.max(1) as usize;
  for (idx, px) in rgba.chunks_exact(4).enumerate() {
    let l = 0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32;
    sum += l as f64;
    if px[0] > 8 || px[1] > 8 || px[2] > 8 {
      nonblack += 1;
    }
    if l >= 250.0 {
      saturated += 1;
    }
    if let Some(((cx, cy), r)) = nucleus {
      if r.is_finite() && r > 0.0 {
        let x = (idx % w) as f64 - cx;
        let y = (idx / w) as f64 - cy;
        let rho = (x * x + y * y).sqrt() / r;
        if rho < 2.0 {
          core.push(l);
        } else if (10.0..=50.0).contains(&rho) {
          annulus.push(l);
        }
      }
    }
    lums.push(l);
  }
  if n == 0 {
    return (f64::NAN, f64::NAN, f64::NAN, FlowMetric::NAN);
  }
  lums.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let p99 = lums[((n as f64 * 0.99) as usize).min(n - 1)] as f64;
  let core_median = median_f32(&mut core);
  let annulus_median = median_f32(&mut annulus);
  let core_over_annulus = if annulus_median > 0.0 {
    core_median / annulus_median
  } else {
    f64::NAN
  };
  let flow = FlowMetric {
    saturated_fraction: saturated as f64 / n as f64,
    core_median,
    annulus_median,
    core_over_annulus,
  };
  (sum / n as f64, p99, nonblack as f64 / n as f64, flow)
}

// ───────────────────────────── main ─────────────────────────────

fn main() {
  let args = parse_args();
  if args.play_quality_off {
    // before the context starts: the render thread reads it per frame (`PlayQuality::from_env`)
    unsafe { std::env::set_var("AETHERVK_DUST_PLAY_QUALITY", "0") };
  }
  if let Some(t) = args.lab_table_md.clone() {
    // the score table as Markdown (no GPU)
    std::process::exit(lab_table_markdown(&t));
  }
  if args.lab_score.is_some() {
    // a camera-path capture's frame and the full frame against one truth (no GPU)
    std::process::exit(run_lab_score(&args));
  }
  if let Some(dir) = args.lab.clone() {
    // the offline replay lab: no GPU, no simulation
    if args.lab_path > 0 {
      // the CPU mirror's cadence along a camera path (temporal reuse step 4)
      std::process::exit(run_lab_path(&dir, &args));
    }
    std::process::exit(run_lab(&dir, args.lab_frame, args.lab_truth));
  }
  std::fs::create_dir_all(&args.out).expect("create --out");
  *LOG_FILE.lock().unwrap() = Some(File::create(args.out.join("log.txt")).expect("log.txt"));
  // the dust profiler (`AETHERVK_DUST_PROFILE=1`): the pipelines' internal representations go
  // next to the frames
  if dust_profile_on() {
    unsafe { std::env::set_var("AETHERVK_DUST_PROFILE_DIR", &args.out) };
  }
  LOG_ECHO.store(!args.quiet, Ordering::Relaxed);
  let t_run = Instant::now();

  // the camera path (`--path-frames`) runs from the one settled epoch
  let n_frames = if args.path_frames > 0 {
    1
  } else {
    ((args.days * 24.0 / args.step_hours).round() as u32).max(1) + 1
  };
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
  // a jet ignited before the start (`--ignite-days < 0`) emits from the SPK before it: cover it
  let spk_from = start + Duration::from_days(args.ignite_days.min(0.0));
  let spk = match args.spk.clone().map(Ok).unwrap_or_else(|| find_or_fetch_spk(spk_from, end)) {
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
  let earth_body = EntityId::from_ffi(scene_ret.earth_body);
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

  // rotation model: `--pole` (the app's Comet tab model), else the default pole with a
  // `--spin-hours` period (12.4 h); `--spin-hours 0`: none
  let rotation_desc = if let Some([ra, dec, w0, rate]) = args.pole {
    set_rotation(&ctx, scene_id, comet_body, rotation_model(ra, dec, w0, rate));
    format!(
      "pole RA {ra}°, Dec {dec}°, W0 {w0}°, rate {rate:.4}°/d (period {:.4} h), body-fixed (--pole)",
      360.0 * 24.0 / rate
    )
  } else if args.spin_hours > 0.0 {
    set_rotational_model(&ctx, scene_id, comet_body, args.spin_hours);
    format!("pole RA 0°, Dec 90°, W0 0°, period {} h, body-fixed", args.spin_hours)
  } else {
    String::from("none (--spin-hours 0)")
  };
  log_line(&format!("rotation: {rotation_desc}"));

  // the jets (`--jet`, else one default jet), ignited at `start + ignite_days` (nothing before
  // it) unless `--prestart`. One stream colour for all (the hue ray reads one), seeds 7, 8, …
  let ignition: Option<Epoch> =
    (!args.prestart).then(|| args.start + Duration::from_days(args.ignite_days));
  let emits: Vec<ParticleSystemEmitParams> = if args.jets.is_empty() {
    vec![jet_params(7, args.isotropic)]
  } else {
    args
      .jets
      .iter()
      .enumerate()
      .map(|(k, j)| ParticleSystemEmitParams {
        // `JetNativeSync.BuildJet`: degrees → radians, σ in m/s
        latitude_rad: (j.lat_deg as f32).to_radians(),
        longitude_rad: (j.lon_deg as f32).to_radians(),
        aperture_rad: (j.aperture_deg as f32).to_radians(),
        start_velocity_mean: j.speed_ms as f32,
        start_velocity_std: (j.sigma_frac * j.speed_ms) as f32,
        ..jet_params(7 + k as u32, args.isotropic)
      })
      .collect()
  };
  let emit = emits[0];
  let jets_desc = emits
    .iter()
    .map(|e| {
      format!(
        "lat {:.2}°, lon {:.2}°, aperture {:.2}° (half-angle), v {:.2}±{:.2} m/s, seed {}",
        e.latitude_rad.to_degrees(),
        e.longitude_rad.to_degrees(),
        e.aperture_rad.to_degrees(),
        e.start_velocity_mean,
        e.start_velocity_std,
        e.seed
      )
    })
    .collect::<Vec<_>>()
    .join("; ");
  log_line(&format!("jets: {jets_desc}"));
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
    for e in &emits {
      match add_jet(&ctx, scene_id, comet_body, *e, jet_color, emission_start_us) {
        Ok(j) => log_line(&format!("jet entity {}", j.as_ffi())),
        Err(e) => {
          log_line(&format!("FATAL: add_jet: {e}"));
          std::process::exit(6);
        }
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

  if args.no_trajectories {
    let n = hide_trajectories(&ctx, scene_id, &[comet_body, earth_body]);
    log_line(&format!("trajectories hidden (--no-trajectories): {n} entities"));
  }

  // view settings
  ctx.set_dust_view_flags(scene_id, args.view_flags);
  ctx.set_dust_exposure_ev(scene_id, args.exposure_ev);
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
  if let Some(a) = args.mc_age_min {
    // the render side of `--mc-age-min`: every tier's band minimum (`DustFrame::ttl[1]`) raised
    // there — by `AETHERVK_DUST_DEBUG_AGE_MIN`, which the engine reads only once `gpu/frame.rs`
    // mirrors its `AETHERVK_DUST_DEBUG_AGE_MAX` hook (`frame.ttl[1] = frame.ttl[1].max(min)`);
    // without the hook only the truth is filtered and the comparison is truth(older than S) vs
    // render(all ages): the residual is then the young dust's share, not an error
    unsafe { std::env::set_var("AETHERVK_DUST_DEBUG_AGE_MIN", format!("{a}")) };
    log_line(&format!(
      "[observer] dust older than {a} s only (--mc-age-min): the truth; the render only where the engine honours AETHERVK_DUST_DEBUG_AGE_MIN (gpu/frame.rs)"
    ));
  }
  ctx.set_dust_flow_speed(scene_id, 1.0);
  ctx.set_dust_softening(scene_id, 1e-2);
  // lab knob: the scene's volume detail (pieces per volumetric packet, `set_dust_volume_detail`)
  if let Some(v) = std::env::var("OBSERVER_VOLUME_DETAIL").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
    let got = ctx.set_dust_volume_detail(scene_id, v);
    log_line(&format!("[observer] volume detail {v} requested, set {got:?}"));
  }
  // lab knob: the scene's trajectory detail (solved size-polyline edges 17 / 9 / 5,
  // `set_dust_propagate_detail`; round 8 step D)
  if let Some(v) = std::env::var("OBSERVER_PROPAGATE_DETAIL").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
    let got = ctx.set_dust_propagate_detail(scene_id, v);
    log_line(&format!("[observer] trajectory detail {v} requested, set {got:?}"));
  }
  // lab knob: the scene's haze detail (sub-packets per cluster at the top level,
  // `set_dust_haze_detail`; round 8 step G)
  if let Some(v) = std::env::var("OBSERVER_HAZE_DETAIL").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
    let got = ctx.set_dust_haze_detail(scene_id, v);
    log_line(&format!("[observer] haze detail {v} requested, set {got:?}"));
  }

  // presentation engine + camera
  let pe = ctx
    .create_presentation_engine(scene_id, args.width, args.height)
    .expect("create_presentation_engine");
  PE_ID.store(pe.0 as u64, Ordering::Release);
  let aspect = args.width as f64 / args.height as f64;
  // the perspective poses' field of view and `tan(fov/2)` (the observer's own projection of the
  // nucleus pixel and the Monte-Carlo truth share it with the render)
  let fov_deg = args.fov_deg.unwrap_or(match args.pose {
    Pose::Inside => INSIDE_FOV_DEG,
    _ => 2.0,
  });
  let persp_tan_h: Option<f64> =
    args.pose.is_perspective().then(|| (fov_deg.to_radians() / 2.0).tan());
  // `--pose inside`: D is the camera's distance from the nucleus SURFACE (the camera sits in
  // the dust, never inside the 2 km nucleus mesh), so the centre distance is D + 2 km (AU); the
  // camera asks for a near plane of `D·1e-3`, which the engine floors at
  // `clamp(0.5·D, 0.1 mm, 1 m)` (`scene_conversion.rs` `DEPTH_NEAR_MIN`..`DEPTH_NEAR_FLOOR`), so
  // the truth clips at `max(D·1e-3, clamp(0.5·D, 0.1 mm, 1 m))`; the far plane is
  // `max(1e4·D, 2·coma radius)`, re-applied every frame with the coma radius of the moment (0
  // until the first rendered frame)
  let au_m = aethervk_core_rlib::scene::dust::AU_M;
  let inside_d_au = args.inside_m / au_m;
  let inside_au = inside_d_au + NUCLEUS_RADIUS_KM as f64 / AU_KM;
  let inside_near_au = inside_d_au * 1e-3;
  let inside_near_floor_au = (inside_d_au * 0.5).clamp(1.0e-4 / au_m, 1.0 / au_m);
  let inside_truth_near_au = inside_near_au.max(inside_near_floor_au);
  let inside_far_au = |coma_km: f64| (1e4 * inside_d_au).max(2.0 * coma_km / AU_KM);
  let cam_id = match args.pose {
    Pose::Earth => ctx
      .add_perspective_camera(
        scene_id,
        pe,
        "observer_cam",
        fov_deg.to_radians() as f32,
        1e-5,
        20.0,
      )
      .expect("add_perspective_camera"),
    Pose::Inside => ctx
      .add_perspective_camera(
        scene_id,
        pe,
        "observer_cam",
        fov_deg.to_radians() as f32,
        inside_near_au as f32,
        inside_far_au(0.0) as f32,
      )
      .expect("add_perspective_camera"),
    _ => ctx
      .add_orthographic_camera(scene_id, pe, "observer_cam", 1.0, 0.1, 1000.0)
      .expect("add_orthographic_camera"),
  };
  let cam = EntityId::from_ffi(cam_id.get());
  // `--pose inside`: the perspective projection with the far plane of the moment (AU)
  let send_inside_projection = |far_au: f64| {
    ctx
      .threads
      .logic_thread
      .tx()
      .try_send(LogicCommand::SetCameraTransform {
        scene_id,
        camera_id: cam.as_ffi(),
        transform: None,
        projection: Some(CameraProjection::Perspective {
          fov: fov_deg.to_radians() as f32,
          aspect_ratio: aspect as f32,
          near: inside_near_au as f32,
          far: far_au as f32,
        }),
      })
      .expect("SetCameraTransform send");
  };

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

  // `--pose earth` projection (`--earth-field-km`, `--ortho`): from the Earth→comet distance D at
  // the start epoch (geocentric, geometric: the site's 6 400 km and the light time move D by
  // ~1e-4 relative). Orthographic: half-height KM/2, distance-free, near/far 0.5·D / 2·D (the
  // camera stays at the site, aimed at the comet: frontal). Perspective: fov 2·atan(KM/2 / D).
  // Shadows `fov_deg`, `persp_tan_h`, `half_height_au` for the frames below.
  let (fov_deg, persp_tan_h, half_height_au, earth_near_au, earth_far_au) = if args.pose
    == Pose::Earth
  {
    let earth_pos = ctx.get_scene(scene_id).and_then(|sc| {
      let g = sc.read();
      g.scene
        .global_transform_f64(earth_body)
        .map(|t| [t.position.x(), t.position.y(), t.position.z()])
    });
    let d_au = earth_pos.map_or(0.0, |e| norm(sub(comet0, e)));
    if d_au <= 0.0 {
      problems.push(
        "earth pose: no Earth transform at the start epoch (Earth→comet distance unknown)".into(),
      );
    }
    let (fov_deg, tan_h, hh_au, near, far) = if let Some(radii) = args.nucleus_from_earth {
      // the nucleus view: the camera `radii` R short of the comet, the nucleus 1/4 of the frame
      // tall (4 diameters = 8 R over the full height: half-height 4 R at the nucleus)
      let r_au = NUCLEUS_RADIUS_KM as f64 / AU_KM;
      let d_nuc_au = radii * r_au;
      let hh_au = 4.0 * r_au;
      let tan_h = hh_au / d_nuc_au;
      let f = (2.0 * tan_h.atan()).to_degrees();
      let (near, far) = (1e-3 * d_nuc_au, (1e4 * d_nuc_au).max(2.0 * coma0_km / AU_KM));
      ctx
        .threads
        .logic_thread
        .tx()
        .try_send(LogicCommand::SetCameraTransform {
          scene_id,
          camera_id: cam.as_ffi(),
          transform: None,
          projection: Some(CameraProjection::Perspective {
            fov: f.to_radians() as f32,
            aspect_ratio: aspect as f32,
            near: near as f32,
            far: far as f32,
          }),
        })
        .expect("SetCameraTransform send");
      log_line(&format!(
        "nucleus from Earth: camera {radii} R ({:.1} km) short of the astrometric comet, fov {f:.3}° (nucleus 1/4 of the height), near {:.3} km, far {:.1} km",
        d_nuc_au * AU_KM,
        near * AU_KM,
        far * AU_KM
      ));
      (f, Some(tan_h), hh_au, near, far)
    } else if args.earth_ortho {
      let hh_au = args.earth_field_km.map_or(half_height_au, |k| k / 2.0 / AU_KM);
      let (near, far) = (0.5 * d_au, 2.0 * d_au);
      let hw = (hh_au * aspect) as f32;
      let hh = hh_au as f32;
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
            near: near as f32,
            far: far as f32,
          }),
        })
        .expect("SetCameraTransform send");
      let fov = (2.0 * (hh_au / d_au.max(1e-300)).atan()).to_degrees();
      (fov, None, hh_au, near, far)
    } else {
      let fov_deg = match args.earth_field_km {
        Some(k) => {
          let f = (2.0 * (k / 2.0 / (d_au * AU_KM).max(1e-300)).atan()).to_degrees();
          ctx
            .threads
            .logic_thread
            .tx()
            .try_send(LogicCommand::SetCameraTransform {
              scene_id,
              camera_id: cam.as_ffi(),
              transform: None,
              projection: Some(CameraProjection::Perspective {
                fov: f.to_radians() as f32,
                aspect_ratio: aspect as f32,
                near: 1e-5,
                far: 20.0,
              }),
            })
            .expect("SetCameraTransform send");
          f
        }
        None => fov_deg,
      };
      let tan_h = (fov_deg.to_radians() / 2.0).tan();
      (fov_deg, Some(tan_h), half_height_au, 1e-5, 20.0)
    };
    let half_km = match tan_h {
      Some(t) => d_au * AU_KM * t,
      None => hh_au * AU_KM,
    };
    log_line(&format!(
      "earth projection: {} half-height {half_km:.1} km at the comet (field {:.1} km), fov {fov_deg:.6}°, D {:.1} km, {:.4}″/px, up {}",
      if tan_h.is_some() { "perspective" } else { "orthographic" },
      2.0 * half_km,
      d_au * AU_KM,
      fov_deg * 3600.0 / args.height as f64,
      if args.north_up { "celestial north" } else { "ecliptic" }
    ));
    (fov_deg, tan_h, hh_au, near, far)
  } else {
    (fov_deg, persp_tan_h, half_height_au, 1e-5, 20.0)
  };

  // `--pose inside`: the near/far of the frame being rendered (AU): the truth clips there too
  let mut inside_far_now = inside_far_au(coma0_km);
  if args.pose == Pose::Inside {
    send_inside_projection(inside_far_now);
    log_line(&format!(
      "inside pose: camera {:.6} m from the surface ({:.6} m from the centre) sunward on the anti-solar axis, fov {fov_deg:.1}°, near {:.6} m (truth clips at {:.6} m: the engine near floor), far {:.3} km, {:.6} m/px at the nucleus",
      args.inside_m,
      inside_au * au_m,
      inside_near_au * au_m,
      inside_truth_near_au * au_m,
      inside_far_now * AU_KM,
      2.0 * inside_au * au_m * persp_tan_h.unwrap_or(1.0)
        / args.height as f64
    ));
  }
  // projection (AU, like the C# RequestOrthographicProjection)
  if !args.pose.is_perspective() {
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
      Pose::Inside => {
        // the anti-solar axis is the comet's heliocentric direction (the hue ray's `anti`): the
        // camera stands D sunward of the nucleus surface (D + 2 km from the centre) on it and
        // looks along it, through the nucleus, down the tail; up = ecliptic north (+Z), the
        // nucleus at the frame centre
        let anti = normalize(comet);
        let rel = [
          -anti[0] * inside_au,
          -anti[1] * inside_au,
          -anti[2] * inside_au,
        ];
        let pos = [comet[0] + rel[0], comet[1] + rel[1], comet[2] + rel[2]];
        (
          pos,
          earth_observer::look_at_origin_from(rel, Some([0.0, 0.0, 1.0])),
        )
      }
    }
  };
  if args.pose == Pose::Earth {
    let (p, q) = (
      [comet0[0], comet0[1], comet0[2] + cam_height_au],
      earth_observer::look_at_origin_from([0.0, 0.0, cam_height_au], None),
    );
    set_camera_pose(&ctx, scene_id, cam, p, q);
    // `--lon auto`: the site's meridian through the comet at the first epoch (hour angle 0: the
    // comet at its highest for the latitude), from the Earth's body-fixed frame (+X Greenwich)
    let lon_deg = args.lon_deg.unwrap_or_else(|| {
      let earth_rot = ctx.get_scene(scene_id).and_then(|sc| {
        let g = sc.read();
        g.scene.global_transform_f64(earth_body).map(|t| {
          (
            [t.position.x(), t.position.y(), t.position.z()],
            earth_observer::qnormalize([
              t.rotation.0.x(),
              t.rotation.0.y(),
              t.rotation.0.z(),
              t.rotation.0.w(),
            ]),
          )
        })
      });
      match earth_rot {
        Some((e, q)) => {
          let bf = earth_observer::rotate(earth_observer::qinverse(q), sub(comet0, e));
          let lon = bf[1].atan2(bf[0]).to_degrees();
          let dec = (bf[2] / norm(bf)).asin().to_degrees();
          log_line(&format!(
            "earth site --lon auto: the comet transits at longitude {lon:.3}° (body-fixed declination {dec:.2}°: elevation at transit ≈ {:.1}° for latitude {:.1}°)",
            90.0 - (args.lat_deg - dec).abs(),
            args.lat_deg
          ));
          lon
        }
        None => {
          log_line("earth site --lon auto: no Earth transform, longitude 0");
          0.0
        }
      }
    });
    if let Some(radii) = args.nucleus_from_earth {
      let ok = ctx.set_nucleus_view_from_earth(
        scene_id,
        cam.as_ffi(),
        scene_ret.earth_body,
        scene_ret.comet_body,
        radii * NUCLEUS_RADIUS_KM as f64 / AU_KM,
      );
      log_line(&format!(
        "nucleus view from Earth set: {ok}, {radii} nucleus radii short of the comet, celestial north up"
      ));
    } else {
      let ok = ctx.set_earth_observer(
        scene_id,
        cam.as_ffi(),
        Some(earth_observer::EarthObserverMode::CometTracking),
        scene_ret.earth_body,
        scene_ret.comet_body,
        true,
        args.lat_deg,
        lon_deg,
        args.elev_m,
        [0.0, 0.0, 0.0, 1.0],
        if args.north_up {
          earth_observer::ObserverUp::CelestialNorth
        } else {
          earth_observer::ObserverUp::Ecliptic
        },
      );
      log_line(&format!(
        "earth observer (comet tracking) set: {ok}, site lat {:.4}° lon {:.4}° elev {:.1} m, north-up {}",
        args.lat_deg, lon_deg, args.elev_m, args.north_up
      ));
    }
  } else {
    let (p, q) = pose_for(comet0);
    set_camera_pose(&ctx, scene_id, cam, p, q);
  }

  // ───── --play-live: the simulation plays, frames captured without pausing ─────
  if let Some(live_s) = args.play_live_seconds {
    let exit_code = play_live_probe(
      &ctx,
      scene_id,
      pe,
      cam,
      comet_body,
      &args,
      live_s,
      &pose_for,
      &send_inside_projection,
      &mut inside_far_now,
    );
    if let Ok(mut g) = LOG_FILE.lock() {
      if let Some(f) = g.as_mut() {
        let _ = f.flush();
      }
    }
    std::mem::forget(ctx);
    unsafe { libc::_exit(exit_code) }
  }

  // ───── frames ─────
  let mut stats = File::create(args.out.join("stats.jsonl")).expect("stats.jsonl");
  let mut records: Vec<FrameRec> = Vec::new();
  let mut frames_rgba: Vec<image::RgbaImage> = Vec::new();
  let mut rot_toggle = false;
  let mut dust_profile_last: Option<aethervk_core_rlib::scene::dust::DustProfileCounters> = None;

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
    if args.pose == Pose::Inside {
      // the far plane follows the coma radius (`max(1e4·D, 2·coma)`): re-applied when it moves
      let far = inside_far_au(ctx.dust_coma_radius_km(scene_id));
      if far != inside_far_now {
        log_line(&format!(
          "frame {i}: inside far plane {:.3} km → {:.3} km (coma {:.1} km)",
          inside_far_now * AU_KM,
          far * AU_KM,
          ctx.dust_coma_radius_km(scene_id)
        ));
        inside_far_now = far;
        send_inside_projection(far);
      }
    }
    if args.pose != Pose::Earth {
      let (p, q) = pose_for(comet);
      set_camera_pose(&ctx, scene_id, cam, p, q);
    }
    if args.no_trajectories {
      let n = hide_trajectories(&ctx, scene_id, &[comet_body, earth_body]);
      if n > 0 {
        log_line(&format!("frame {i}: {n} more trajectory entities hidden"));
      }
    }
    if args.pose == Pose::Earth {
      log_line(
        &earth_pose_diag(
          &ctx,
          scene_id,
          cam,
          comet_body,
          earth_body,
          i,
          "before settle",
        )
        .0,
      );
    }
    // settle: the pose/seek reach the renderer on the next extraction
    let t_dl = Instant::now();
    let mut dl = wait_and_download(&ctx, args.width, args.height, args.frames_to_wait, 60_000);
    // the frame's work settles: the host moves the split depth and the volume / EWA biases
    // (header word `PYR_SPLIT_DEPTH`) after a frame over a budget — a cold start or a jump inside
    // the coma draws its first frames under the in-frame gates (`DUST_FRAME_HARD_FACTOR`); one
    // generation at a time until the word repeats across all `DUST_READBACK_SLOTS` slots, at most
    // `WORK_SETTLE_MAX`, before the exposure settles on the settled image
    let mut work_extra = 0u32;
    if dl.is_some() {
      let word_of = || {
        ctx
          .download_dust_pyramid(pe)
          .and_then(|p| p.get(aethervk_core_rlib::scene::dust::PYR_SPLIT_DEPTH as usize).copied())
      };
      let mut prev = word_of();
      let mut same_streak = 0usize;
      while work_extra < WORK_SETTLE_MAX {
        let Some(next) = wait_and_download(&ctx, args.width, args.height, 1, 60_000) else {
          break;
        };
        dl = Some(next);
        work_extra += 1;
        let now = word_of();
        if now == prev {
          same_streak += 1;
          if same_streak >= aethervk_core_rlib::gpu_backends::vulkan::device::dust::DUST_READBACK_SLOTS {
            break;
          }
        } else {
          same_streak = 0;
        }
        prev = now;
      }
    }
    // the exposure settles: the white point adapts by half its log gap per frame
    // (`WHITE_ADAPT_RATE`) from the seed (20 mag/arcsec²) or the previous epoch's value — a far
    // frame is 20 mag from the seed and its first generations are under-exposed; one more
    // generation at a time until two in a row agree within 2 %, at most `EXPOSURE_SETTLE_MAX`
    let mut settle_extra = 0u32;
    if dl.is_some() {
      let white_of = || ctx.dust_exposure(pe).map(|e| (e.white, e.fixed));
      let mut prev = white_of();
      while settle_extra < EXPOSURE_SETTLE_MAX && !matches!(prev, Some((_, true))) {
        let Some(next) = wait_and_download(&ctx, args.width, args.height, 1, 30_000) else {
          break;
        };
        dl = Some(next);
        settle_extra += 1;
        let now = white_of();
        let done = match (prev, now) {
          (Some((a, _)), Some((b, _))) if a > 0.0 && b > 0.0 => (b / a).ln().abs() < 0.02,
          _ => true,
        };
        prev = now;
        if done {
          break;
        }
      }
    }
    // the speck shell settles: the host moves `d_max` inward after a frame over the speck budget
    // and outward under a quarter of it, so a frame whose pyramid reports a speck overflow or a
    // moving shell is not judged; one generation at a time until the overflow word is 0 and the
    // radius repeats, sharing the `EXPOSURE_SETTLE_MAX` extra generations
    let mut speck_extra = 0u32;
    if dl.is_some() {
      use aethervk_core_rlib::scene::dust::specks as sp;
      let overflow_of = || {
        ctx
          .download_dust_pyramid(pe)
          .and_then(|p| p.get(sp::PYR_SPECKS_OVERFLOW as usize).copied())
          .unwrap_or(0)
      };
      // and until the shell radius stops moving (it starts at 1 m and grows toward half the
      // budget, ≤ ×2 a readback): the judged frame draws the settled shell
      let d_max_of = || {
        ctx
          .download_dust_pyramid(pe)
          .and_then(|p| p.get(sp::PYR_SPECKS_DMAX as usize).copied())
          .map(f32::from_bits)
          .unwrap_or(0.0)
      };
      let mut prev_d = d_max_of();
      while settle_extra + speck_extra < EXPOSURE_SETTLE_MAX {
        if overflow_of() == 0 && speck_extra > 0 {
          let d = d_max_of();
          if d == prev_d {
            break;
          }
          prev_d = d;
        } else if overflow_of() == 0 && prev_d == 0.0 {
          break;
        }
        let Some(next) = wait_and_download(&ctx, args.width, args.height, 1, 30_000) else {
          break;
        };
        dl = Some(next);
        speck_extra += 1;
      }
    }
    // the render cost: distinct render generations were waited for
    let generations = args.frames_to_wait + work_extra + settle_extra + speck_extra;
    let render_ms = t_dl.elapsed().as_secs_f64() * 1e3 / generations.max(1) as f64;
    log_line(&format!(
      "[observer] render ≈ {render_ms:.1} ms/frame ({generations} generations, {work_extra} to settle the work, {settle_extra} to settle the exposure, {speck_extra} to settle the speck shell)"
    ));
    // the dust passes' GPU time of the last measured frame (`DustGpuTimings`)
    let dust_gpu = ctx.dust_gpu_timings(pe);
    match dust_gpu {
      Some(t) => {
        log_line(&format!(
          "[observer] dust GPU: propagate {:.2} ms, splat+specks {:.2} ms, measure {:.2} ms, composite {:.2} ms, total {:.2} ms (average {:.2} ms) at {}x{}{}",
          t.propagate_ms, t.splat_ms, t.measure_ms, t.composite_ms, t.total_ms, t.total_ema_ms, t.width, t.height,
          if t.reused { " (image reused)" } else { "" }
        ));
        let calls: Vec<String> =
          t.call_ms.iter().filter(|c| !c.is_nan()).map(|c| format!("{c:.2}")).collect();
        log_line(&format!(
          "[observer] dust GPU: splats {:.2} ms (per dispatch, layer then tier: {} ms), specks {:.2} ms",
          t.splat_only_ms,
          calls.join(" / "),
          t.specks_ms
        ));
        // the budget loop (step 5): knobs, the cadence, the feedback regimes, the work by target
        if let Some(b) = ctx.dust_budget_telemetry(pe) {
          for l in b.lines() {
            log_line(&format!("[observer] {l}"));
          }
        }
      }
      None => log_line("[observer] dust GPU: no timings (no timestamp queries, or none measured yet)"),
    }
    // the last frame that drew (a still view reuses its image at ~0 ms; a moving one draws:
    // the budget holds for the drawn frame)
    let dust_gpu_drawn = ctx.dust_gpu_timings_drawn(pe);
    if let Some(t) = dust_gpu_drawn.filter(|_| dust_gpu.is_some_and(|l| l.reused)) {
      log_line(&format!(
        "[observer] dust GPU: the last frame reused the image; the last drawn one (frame {}): propagate {:.2} ms, splat+specks {:.2} ms, total {:.2} ms",
        t.frame, t.propagate_ms, t.splat_ms, t.total_ms
      ));
    }
    // the volume packet queue of the last drawn frame (`dust::DUST_VQUEUE_*`): what the consumer
    // drew from the queue, what overflowed it into the volume spill (drawn from the spill, or
    // lost: refused by the spill too, not drawn), the chunks
    if let Some((drawn, overflowed, chunks)) = ctx.dust_vqueue_counts(pe) {
      let (spilled, lost) = ctx
        .dust_vqueue_spill_counts(pe)
        .map_or(("?".to_string(), "?".to_string()), |(s, l)| (s.to_string(), l.to_string()));
      log_line(&format!(
        "[observer] dust volume queue: {drawn} packets drawn by the consumer, {overflowed} overflowed → {spilled} spilled / {lost} lost, {chunks} chunks; {} volume deposits split above the per-add cap",
        ctx.dust_volume_split_adds(pe).map_or("?".to_string(), |n| n.to_string())
      ));
      // track 9A (t9a2d): the small edge pieces the consumer deferred to the edge pass, and (t9a2e)
      // those the pass demoted (one projection a slice with the predicted terms, not the sigma
      // points: the demotion firing in this frame)
      let (edge_drawn, edge_lost, edge_demoted) = ctx.dust_vedge_counts(pe).map_or(
        ("?".to_string(), "?".to_string(), "?".to_string()),
        |(d, l, m)| (d.to_string(), l.to_string(), m.to_string()),
      );
      log_line(&format!(
        "[observer] dust edge pass: {edge_drawn} small edge pieces drawn, {edge_demoted} of them demoted (ε {:.0e}), {edge_lost} lost (a full region refused them)",
        aethervk_core_rlib::scene::dust::DUST_SMALL_DEMOTE_EPS
      ));
    }
    // the shader-clock profiler's counters of the last drawn frame (`AETHERVK_DUST_PROFILE=1`,
    // `dust_profile.glsl`): the stages' shares, the per-unit costs, the thread tail, the
    // occupancy verdict against the measured splat time
    if dust_profile_on() {
      match ctx.dust_profile_counters(pe) {
        Some(p) if p.any() => {
          let t = dust_gpu_drawn.or(dust_gpu);
          for line in dust_profile_report(&p, t.as_ref(), sm_clock_mhz()) {
            log_line(&line);
          }
          dust_profile_last = Some(p);
        }
        Some(_) => log_line("[observer] dust profile: all counters zero (the device lacks VK_KHR_shader_clock or subgroup arithmetic, or nothing was drawn)"),
        None => log_line("[observer] dust profile: no counters (no view)"),
      }
    }
    // the dust frame budget (`--dust-budget-ms`): the app re-splats every presentation frame
    // whose view changed; checked on the last frame that drew
    if let Some(budget) = args.dust_budget_ms {
      match dust_gpu_drawn.or(dust_gpu) {
        Some(t) if t.total_ms > budget => problems.push(format!(
          "frame {i}: dust GPU {:.2} ms over the {budget} ms budget (splat+specks {:.2} ms, propagate {:.2} ms) at {}x{}",
          t.total_ms, t.splat_ms, t.propagate_ms, t.width, t.height
        )),
        Some(_) => {}
        None => problems.push(format!("frame {i}: dust GPU time not measured (the budget cannot be checked)")),
      }
    }
    let (mut rgba, tid) = match dl {
      Some(x) => x,
      None => {
        problems.push(format!("frame {i}: render download failed/timed out"));
        log_line(&format!("frame {i}: render download failed"));
        (vec![0u8; (args.width * args.height * 4) as usize], 0)
      }
    };
    bgra_to_rgba(&mut rgba);
    if args.pose == Pose::Earth {
      let (line, elev) = earth_pose_diag(
        &ctx,
        scene_id,
        cam,
        comet_body,
        earth_body,
        i,
        "after settle",
      );
      log_line(&line);
      if let Some(e) = elev.filter(|e| *e < 0.0) {
        problems.push(format!(
          "frame {i}: the comet is {:.1}° below the site's horizon: the tracking camera faces the horizon (choose --lon auto or another site / epoch)",
          -e
        ));
      }
    }
    // the Earth observer's light-time view (`LightTimeViewComponent` on the camera): the render
    // draws the comet's micro layer at the astrometric place `x + offset_au` (the dust evaluated
    // `delay_s` earlier): the nucleus pixel and the truth's placement follow it
    let comet = match ctx.get_scene(scene_id).and_then(|sc| {
      let g = sc.read();
      g.scene.with_component(cam, |l: &earth_observer::LightTimeViewComponent| *l)
    }) {
      Some(l) if args.pose == Pose::Earth => {
        log_line(&format!(
          "frame {i}: light-time view: delay {:.1} s, offset {:.1} km",
          l.delay_s,
          norm(l.offset_au) * AU_KM
        ));
        [
          comet[0] + l.offset_au[0],
          comet[1] + l.offset_au[1],
          comet[2] + l.offset_au[2],
        ]
      }
      _ => comet,
    };

    let (cam_pos, cam_q) =
      camera_pose(&ctx, scene_id, cam).unwrap_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0]));
    let (right, forward, up) = camera_axes(cam_q);
    let d = sub(comet, cam_pos);
    let (sx, sy) = (dot(d, right), dot(d, up));
    let depth = dot(d, forward);
    let (px, py) = match persp_tan_h {
      Some(tan_h) => (
        args.width as f64 / 2.0 * (1.0 + sx / depth.max(1e-12) / (tan_h * aspect)),
        args.height as f64 / 2.0 * (1.0 - sy / depth.max(1e-12) / tan_h),
      ),
      None => (
        args.width as f64 / 2.0 * (1.0 + sx / (half_height_au * aspect)),
        args.height as f64 / 2.0 * (1.0 - sy / half_height_au),
      ),
    };
    // metres per pixel at the nucleus' depth: the per-pixel scale every analysis uses
    let m_per_px_nucleus = match persp_tan_h {
      Some(tan_h) => {
        2.0 * depth.max(1e-12) * aethervk_core_rlib::scene::dust::AU_M * tan_h / args.height as f64
      }
      None => 2.0 * half_height_au * AU_KM * 1e3 / args.height as f64,
    };
    let (ix, iy) = (
      px.round().clamp(0.0, args.width as f64 - 1.0) as u32,
      py.round().clamp(0.0, args.height as f64 - 1.0) as u32,
    );
    let o = ((iy * args.width + ix) * 4) as usize;
    let rgb = (rgba[o], rgba[o + 1], rgba[o + 2]);
    let nucleus_px_radius = NUCLEUS_RADIUS_KM as f64 * 1e3 / m_per_px_nucleus;
    let on_screen = px.is_finite()
      && py.is_finite()
      && px >= 0.0
      && py >= 0.0
      && px < args.width as f64
      && py < args.height as f64;
    let (mean_l, p99_l, nonblack, flow_metric) = luminance_stats(
      &rgba,
      args.width,
      on_screen.then_some(((px, py), nucleus_px_radius)),
    );
    log_line(&format!(
      "[observer] flow metric frame {i}: sat {:.6} core {:.2} annulus {:.2} ratio {:.3}",
      flow_metric.saturated_fraction,
      flow_metric.core_median,
      flow_metric.annulus_median,
      flow_metric.core_over_annulus
    ));
    // the hue ramp along the tail axis: the on-screen anti-solar direction from the nucleus
    // (pixel y runs against `up`), from two nucleus radii (at least 8 px: the box must clear the
    // nucleus's pixels) to the end of the lit dust
    let stream_rgb = [
      jet_color[0] as f32,
      jet_color[1] as f32,
      jet_color[2] as f32,
    ];
    let (stream_hue_deg, _) = oklab_hue_cl(stream_rgb);
    let hue_ray = {
      let anti = normalize(comet);
      let v = (dot(anti, right), -dot(anti, up));
      let n = (v.0 * v.0 + v.1 * v.1).sqrt();
      HueRay {
        from: (px, py),
        dir: if n > 1e-6 {
          (v.0 / n, v.1 / n)
        } else {
          (1.0, 0.0)
        },
        // the 9×9 box must clear the nucleus's pixels
        r0: (2.0 * nucleus_px_radius + 3.0).max(14.0),
      }
    };
    let mut hue: Vec<HueSample> = Vec::new();

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
    let mut specks_rec: Option<SpeckRec> = None;
    if let Some(pyr) = ctx.download_dust_pyramid(pe) {
      use aethervk_core_rlib::scene::dust as d;
      let words: Vec<u8> = pyr.iter().flat_map(|w| w.to_le_bytes()).collect();
      std::fs::write(args.out.join(format!("frame_{i:03}.pyr")), &words).expect("save pyr");
      if args.save_snapshot {
        save_snapshot(&ctx, pe, &pyr, &args, i as usize);
      }
      if !args.probe.is_empty() {
        probe_replay(&ctx, pe, &pyr, &args, i as usize);
      }
      if args.mc > 0 && !args.no_jet {
        let t_mc = Instant::now();
        let view = McView {
          width: args.width,
          height: args.height,
          cam_pos,
          cam_q,
          half_height_au,
          m_per_px_nucleus,
          aspect,
          perspective_tan_h: persp_tan_h,
          // the render camera's planes: the orthographic poses' from the projection above, the
          // Earth pose's from `add_perspective_camera`, the inside pose's of this frame
          near_au: match args.pose {
            Pose::Earth => earth_near_au,
            Pose::Inside => inside_truth_near_au,
            _ => cam_height_au * 0.01,
          },
          far_au: match args.pose {
            Pose::Earth => earth_far_au,
            Pose::Inside => inside_far_now,
            _ => cam_height_au * 50.0,
          },
          comet_au: comet,
          nucleus_px: (px, py),
          mc_tier: args.mc_tier,
          mc_age_max: args.mc_age_max,
          mc_age_min: args.mc_age_min,
          nucleus_px_radius,
          hue_ray,
          stream: stream_rgb,
          ttl_s: JET_TTL_US as f64 * 1e-6,
          afrho_aperture_km: args.afrho_aperture_km,
          afrho_in_cm: {
            let p = jet_params(7, args.isotropic);
            let r_h = norm(comet) as f32;
            // the recipe's Afρ(r_h): the jet's dust rate converted back (validation observable)
            p.afrho_cm_from_dust_rate(p.dust_production_rate_kgs(r_h)) as f64
          },
          isotropic: args.isotropic,
        };
        if args.save_snapshot {
          save_truth_inputs(&ctx, scene_id, &view, &args.out.join(format!("truth_{i:03}.bin")));
        }
        let saturated = pyramid_saturated_texels(&pyr, args.width, args.height);
        if saturated > 0 {
          log_line(&format!(
            "[observer] frame {i}: pyramid saturated on {saturated} texels (the fixed-point unit lags the previous frame's τ_max: the frame after the ignition), Monte-Carlo comparison skipped"
          ));
        }
        match if saturated > 0 {
          None
        } else {
          monte_carlo_truth(&ctx, scene_id, &pyr, &view, args.mc)
        } {
          Some((rec, render_img, truth_img)) => {
            log_line(&format!(
              "[observer] mc frame {i}: rel rms {:.3} (judged {:.3} after noise {:.3}, core {} px) over {} px, ripple render {:.3} truth {:.3}, ratio p05/p95 {:.2}/{:.2}, absolute render/truth {:.3}, haze {:.3}, {:.0} grains/px ({} grains, {} batches), nucleus {}, {:.1} s",
              rec.rel_rms,
              rec.rel_rms_judged,
              rec.noise,
              rec.core_px,
              rec.region_px,
              rec.ripple_render,
              rec.ripple_truth,
              rec.ratio_p05,
              rec.ratio_p95,
              rec.abs_ratio,
              rec.haze_frac,
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
            if let Some((r, t, a)) = rec.afrho {
              log_line(&format!(
                "[observer] afrho frame {i}: aperture {:.0} km: render {r:.3} cm, truth through the kernel {t:.3} cm, analytic Afρ(r_h)·Φ(α) {a:.3} cm{}",
                view.afrho_aperture_km.unwrap_or(0.0),
                if view.isotropic {
                  ""
                } else {
                  " (cone jet: the isotropic analytic value is indicative only)"
                }
              ));
              if args.check && view.isotropic && t > 0.0 && !((r / t - 1.0).abs() < 0.05) {
                problems.push(format!(
                  "frame {i}: aperture Afρ render {r:.3} cm vs truth {t:.3} cm (5 % expected)"
                ));
              }
            }
            // absolute photometry: the render's radiance factors must sum to the truth's within
            // 10 % (the sample-count dither and the pilot allocation are the tolerance)
            // (the isotropic recipe is the Afρ closure diagnostic: its whole-frame comparison is
            // reported, the aperture closure above is what `--check` enforces — a jet covering
            // the sphere with 64 Gaussian cells sits 18 % below its truth in the far field)
            if args.check
              && !view.isotropic
              && rec.region_px >= 2000
              && !(rec.abs_ratio > 0.9 && rec.abs_ratio < 1.1)
            {
              problems.push(format!(
                "frame {i}: absolute render/truth {:.3} (0.90–1.10 expected)",
                rec.abs_ratio
              ));
            }
            if args.check
              && !view.isotropic
              && rec.region_px >= 2000
              && rec.rel_rms_judged > args.mc_max
            {
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
              problems.push(format!(
                "frame {i}: the Monte-Carlo truth's core centroid ({:.0}, {:.0}) is not the render's ({:.0}, {:.0}) within {:.0} px (frame mapping)",
                rec.peak_px.0, rec.peak_px.1, rec.render_peak_px.0, rec.render_peak_px.1, 3.0 + 1.5 * rec.nucleus_px_radius + 0.02 * args.height as f64
              ));
            }
            // the hue samples where the truth put them, the display measured there
            let ttl_s = JET_TTL_US as f64 * 1e-6;
            let hue_levels = std::env::var("OBSERVER_HUE_LEVELS").as_deref() == Ok("1");
            hue = rec
              .hue_px
              .iter()
              .zip(rec.hue_truth.iter().zip(rec.hue_tau_ratio.iter()))
              .enumerate()
              .map(|(k_s, (&p, (t, ratio)))| {
                if hue_levels {
                  log_line(&hue_levels_line(i, k_s, &pyr, p, stream_rgb));
                }
                let mut s = hue_sample_at(&rgba, args.width, args.height, p, stream_rgb);
                s.tau_ratio = *ratio;
                if let Some((m, sd, rgb)) = t {
                  s.age_ttl = Some(m / ttl_s);
                  s.age_spread_ttl = Some(sd / ttl_s);
                  let rgb32 = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32];
                  let (th, tcl) = oklab_hue_cl(rgb32);
                  let (_, scl) = oklab_hue_cl(stream_rgb);
                  s.truth_hue_deg = Some(th);
                  s.truth_chroma_rel = Some(tcl / scl.max(1e-6));
                  // the stretch value from the truth colour's luma: the white roll-off
                  s.highlight = s.level / luma_of(rgb32) > HUE_HIGHLIGHT_A;
                }
                let (tau, col) = pyramid_colour_at(&pyr, p, HUE_BOX_R);
                s.pyr_tau = tau;
                if let Some(c) = col {
                  let (ph, pcl) = oklab_hue_cl(c);
                  let (_, scl) = oklab_hue_cl(stream_rgb);
                  s.pyr_hue_deg = Some(ph);
                  s.pyr_chroma_rel = Some(pcl / scl.max(1e-6));
                }
                s
              })
              .collect();
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
      // the perspective split's header words (WP3, `splat.rs`): the children made
      // (`PYR_SPLIT_CHILDREN`), the children over the budget (computed: `SplitStats::overflow`),
      // f32 bits of the largest flux dropped wholly behind the eye (`PYR_SPLIT_DROPPED_FLUX`),
      // the host-written depth limit of the frame (`PYR_SPLIT_DEPTH` bits 0–7) and the
      // volumetric packets' ray-integral evaluations and level bias (`VolumeStats`)
      let word = |k: u32| pyr.get(k as usize).copied().unwrap_or(0);
      let split = d::SplitStats::load(&pyr);
      let vol = d::VolumeStats::load(&pyr);
      let split_children = split.children;
      let split_overflow = split.overflow();
      let split_dropped = split.dropped_flux;
      let split_depth = split.depth;
      // the speck pass's words (WP3b, `dust::specks`): the shell radius, the specks drawn and
      // refused, the number density at the eye (64-bit fixed point, 2⁻³⁶ m⁻³ per count)
      let sk = {
        use aethervk_core_rlib::scene::dust::specks as sp;
        SpeckRec {
          d_max_m: f32::from_bits(word(sp::PYR_SPECKS_DMAX)),
          specks: word(sp::PYR_SPECKS),
          overflow: word(sp::PYR_SPECKS_OVERFLOW),
          n_eye_m3: (((word(sp::PYR_N_EYE_HI) as u64) << 32) | word(sp::PYR_N_EYE) as u64) as f64
            * (-36.0f64).exp2(),
        }
      };
      specks_rec = Some(sk);
      log_line(&format!(
        "[observer] pyramid frame {i}: levels {levels} tau_max {:.3e} mask {:#x} tracer {:.1} split children {split_children} overflow {split_overflow} dropped flux {split_dropped:.3e} depth {split_depth} volume evals {} bias {} ewa texels {:.4e} ewa bias {} specks {} overflow {} nodes {} truncated {} d_max {:.4e} m n_eye {:.4e} m⁻³ (texels/Σcounts/max) {}",
        f32::from_bits(pyr[d::PYR_TAU_MAX as usize]),
        pyr[d::PYR_LEVEL_MASK as usize],
        f32::from_bits(pyr[d::PYR_TRACER_COUNTS as usize]),
        vol.evals,
        vol.bias,
        d::EwaStats::load(&pyr).texels as f64,
        d::EwaStats::load(&pyr).bias,
        sk.specks,
        sk.overflow,
        pyr[aethervk_core_rlib::scene::dust::specks::PYR_SPECKS_NODES as usize],
        pyr[aethervk_core_rlib::scene::dust::specks::PYR_SPECKS_TRUNCATED as usize],
        sk.d_max_m,
        sk.n_eye_m3,
        per_level.join(" ")
      ));
      if args.check && split_overflow > 0 {
        problems.push(format!(
          "frame {i}: split budget overflow {split_overflow} children"
        ));
      }
      // the walk's spill header: did the 256-node workgroup stack spill on its own this frame?
      if sk.specks > 0 {
        if let Some(h) = ctx.download_dust_spill_header(pe) {
          log_line(&format!(
            "[observer] specks spill frame {i}: slots claimed {} of {} (stack limit word {})",
            h[0], h[1], h[8]
          ));
        }
      }
      if args.check && sk.overflow > 0 {
        problems.push(format!(
          "frame {i}: speck budget overflow {} specks",
          sk.overflow
        ));
      }
    }
    if hue.is_empty() {
      // no truth: the samples along the displayed dust
      let t_end = hue_ray_extent_display(&rgba, args.width, args.height, &hue_ray);
      hue = hue_ray
        .samples(t_end)
        .into_iter()
        .map(|p| hue_sample_at(&rgba, args.width, args.height, p, stream_rgb))
        .collect();
    }
    if args.sky_image_check {
      problems.extend(sky_image_check(
        &ctx, scene_id, cam, comet_body, &args, i, fov_deg,
      ));
    }
    let img = image::RgbaImage::from_raw(args.width, args.height, rgba).unwrap();
    if !args.no_sheet {
      frames_rgba.push(img);
    }

    let tiers = tiers_now(&ctx, scene_id);
    {
      let (line, hue_problems) =
        hue_report(i, &hue, stream_hue_deg, args.check && mc_rec.is_some());
      log_line(&line);
      problems.extend(hue_problems);
    }
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
      m_per_px_nucleus,
      nucleus_px: (px, py),
      nucleus_rgb: rgb,
      mean_lum: mean_l,
      p99_lum: p99_l,
      nonblack_frac: nonblack,
      flow_metric,
      render_task: tid,
      render_ms,
      dust_gpu,
      dust_profile: dust_profile_last,
      dust_budget: ctx.dust_budget_telemetry(pe).map_or_else(|| "null".to_string(), |b| b.to_json()),
      passes_logged: String::new(),
      mc: mc_rec,
      exposure: ctx.dust_exposure(pe),
      hue,
      specks: specks_rec,
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

  // ───── camera path (`--path-frames N`: temporal reuse step 4, the lab with cadence) ─────
  // Path frame 0 is the settled frame above. Every later frame is one hold-step: the render budget
  // holds the view (`set_render_budget(.., Some(0))`: no frame is sent), the clock seeks (paused)
  // to t₀ + k·Δt, the tiers catch up, the camera moves to nucleus(t_k) + rel_k (rel₀ turned by k·ω
  // about the orbit axis, nucleus-relative in fixed world axes), then exactly one frame is
  // released (`Some(1)`): the renderer draws Δeye and Δt together, once, as at 60 fps. The frame
  // ledger (`dust_frame_ledger`) proves it: one drawn frame per step, its eye − nucleus and its
  // simulation time the intended ones; `--check` fails otherwise.
  if args.path_frames > 1 && !DEVICE_LOST.load(Ordering::Relaxed) {
    use aethervk_core_rlib::scene::dust as d;
    let au_m = d::AU_M;
    let nf = args.path_frames;
    let dt = args.path_dt_s;
    let tan_h = persp_tan_h.unwrap_or(1.0);
    let comet_p0 = comet_position_au(&ctx, scene_id, comet_body).unwrap_or([0.0; 3]);
    let (cam_p0, cam_q0) = camera_pose(&ctx, scene_id, cam).unwrap_or(([0.0; 3], [0.0, 0.0, 0.0, 1.0]));
    let rel0 = sub(cam_p0, comet_p0);
    let r0_m = norm(rel0) * au_m;
    let axis = path_axis(rel0);
    let omega = match args.eye_step_m {
      Some(s) => s / r0_m,
      None => args.orbit_deg_per_s.to_radians() / args.path_fps,
    };
    let dolly_au = args.dolly_m_per_frame / au_m;
    let epoch0 = current_epoch(&ctx, scene_id).unwrap_or(start);
    let keep: Vec<u32> = if args.path_keep.is_empty() { vec![0, nf - 1] } else { args.path_keep.clone() };
    let env_s = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
    let variant = std::env::var("OBSERVER_PATH_VARIANT").unwrap_or_else(|_| {
      if std::env::var("AETHERVK_DUST_CADENCE").is_ok_and(|v| v.trim() == "0") {
        "full".into()
      } else {
        format!("eps{}", std::env::var("AETHERVK_DUST_CADENCE_EPS").unwrap_or_else(|_| "default".into()))
      }
    });
    let pose_label = if args.inside_m >= 1e3 { format!("{:.0}km", args.inside_m / 1e3) } else { format!("{:.0}m", args.inside_m) };
    let speed = dt * args.path_fps;
    log_line(&format!(
      "[observer] camera path: {nf} frames at {} fps, Δt {dt} s per frame (speed {speed}×), orbit {:.4}° per frame ({:.4} m per frame at {:.3} m from the nucleus centre) about {:?}, dolly {} m per frame, keep {:?}, variant {variant}",
      args.path_fps,
      omega.to_degrees(),
      omega * r0_m,
      r0_m,
      axis,
      args.dolly_m_per_frame,
      keep
    ));
    write_kv(
      &args.out.join("path.txt"),
      &[
        ("pose", "inside".into()),
        ("pose_label", pose_label.clone()),
        ("inside_m", format!("{}", args.inside_m)),
        ("width", args.width.to_string()),
        ("height", args.height.to_string()),
        ("frames", nf.to_string()),
        ("fps", format!("{}", args.path_fps)),
        ("dt_s", format!("{dt}")),
        ("speed", format!("{speed}")),
        ("omega_deg_per_frame", format!("{:.9}", omega.to_degrees())),
        ("eye_step_m", format!("{:.6}", omega * r0_m)),
        ("dolly_m_per_frame", format!("{}", args.dolly_m_per_frame)),
        ("axis", format!("{:.9},{:.9},{:.9}", axis[0], axis[1], axis[2])),
        ("rel0_m", format!("{:.6},{:.6},{:.6}", rel0[0] * au_m, rel0[1] * au_m, rel0[2] * au_m)),
        ("epoch0", format!("{epoch0}")),
        ("variant", variant.clone()),
        ("AETHERVK_DUST_CADENCE", env_s("AETHERVK_DUST_CADENCE")),
        ("AETHERVK_DUST_CADENCE_EPS", env_s("AETHERVK_DUST_CADENCE_EPS")),
        ("AETHERVK_DUST_CADENCE_PHASES", env_s("AETHERVK_DUST_CADENCE_PHASES")),
        ("AETHERVK_DUST_NO_REUSE", env_s("AETHERVK_DUST_NO_REUSE")),
      ],
    );
    // the truth's view of path frame 0 (the frame loop's McView with the far plane of frame 0,
    // kept for the whole path: a moving far plane would change the projection, a cadence reset)
    let base_view = {
      let (px0, py0, depth0) = project_px(cam_p0, cam_q0, comet_p0, args.width, args.height, tan_h);
      let mpp = 2.0 * depth0.max(1e-12) * au_m * tan_h / args.height as f64;
      McView {
        width: args.width,
        height: args.height,
        cam_pos: cam_p0,
        cam_q: cam_q0,
        half_height_au,
        m_per_px_nucleus: mpp,
        aspect,
        perspective_tan_h: persp_tan_h,
        near_au: inside_truth_near_au,
        far_au: inside_far_now,
        comet_au: comet_p0,
        nucleus_px: (px0, py0),
        mc_tier: args.mc_tier,
        mc_age_max: args.mc_age_max,
        mc_age_min: args.mc_age_min,
        nucleus_px_radius: NUCLEUS_RADIUS_KM as f64 * 1e3 / mpp,
        hue_ray: HueRay { from: (px0, py0), dir: (1.0, 0.0), r0: 14.0 },
        stream: [jet_color[0] as f32, jet_color[1] as f32, jet_color[2] as f32],
        ttl_s: JET_TTL_US as f64 * 1e-6,
        afrho_aperture_km: args.afrho_aperture_km,
        afrho_in_cm: {
          let p = jet_params(7, args.isotropic);
          let r_h = norm(comet_p0) as f32;
          // the recipe's Afρ(r_h): the jet's dust rate converted back (validation observable)
          p.afrho_cm_from_dust_rate(p.dust_production_rate_kgs(r_h)) as f64
        },
        isotropic: args.isotropic,
      }
    };
    // the hue ray and the nucleus scale as `mc_view_reposed` gives them for every kept frame
    let view_t0 = mc_view_reposed(&base_view, cam_p0, cam_q0, comet_p0);
    let layout = d::PyramidLayout::new(args.width, args.height);
    let light_of = |pyr: &[u32]| -> f64 {
      let unit = f32::from_bits(pyr.get(d::PYR_UNIT as usize).copied().unwrap_or(0)) as f64;
      frame_light(&layout, pyr) * unit
    };
    let mut path_problems: Vec<String> = Vec::new();
    let mut timings: std::collections::BTreeMap<u64, aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings> =
      std::collections::BTreeMap::new();
    let poll_timings = |m: &mut std::collections::BTreeMap<u64, aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings>| {
      if let Some(t) = ctx.dust_gpu_timings(pe) {
        m.insert(t.frame, t);
      }
    };
    // hold, and let the frames already queued drain (their state is frame 0's)
    ctx.set_render_budget(scene_id, pe, Some(0));
    let last_seq = |ctx: &SimulationContext| ctx.dust_frame_ledger(pe, 0).last().map(|e| e.seq);
    {
      let t_d = Instant::now();
      let mut last = last_seq(&ctx);
      let mut since = Instant::now();
      while since.elapsed().as_secs_f64() < 0.6 && t_d.elapsed().as_secs_f64() < 30.0 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let now = last_seq(&ctx);
        if now != last {
          last = now;
          since = Instant::now();
        }
      }
    }
    poll_timings(&mut timings);
    let ledger0 = ctx.dust_frame_ledger(pe, 0);
    let base = ledger0.iter().rev().find(|e| !e.reused).copied();
    let mut prev_seq = ledger0.last().map(|e| e.seq).unwrap_or(0);
    let t_base = base.map(|e| e.t_now_s);
    match base {
      Some(b) => log_line(&format!(
        "[observer] camera path frame 0: the last drawn frame (ledger seq {}) at t {:.3} s, eye − nucleus {:?} m ({} frames reused after it)",
        b.seq,
        b.t_now_s,
        b.nucleus_m.map(|n| [b.eye_m[0] - n[0], b.eye_m[1] - n[1], b.eye_m[2] - n[2]]),
        ledger0.iter().filter(|e| e.seq > b.seq).count()
      )),
      None => path_problems.push("camera path: no drawn frame in the ledger before the path".into()),
    }
    if keep.contains(&0) {
      save_truth_inputs(&ctx, scene_id, &view_t0, &args.out.join("truth_000.bin"));
    }
    if let Some(pyr) = ctx.download_dust_pyramid(pe) {
      let _ = writeln!(
        stats,
        "{{\"path_frame\":0,\"t_rel_s\":0,\"theta_deg\":0,\"light\":{:.6e},\"variant\":\"{variant}\"}}",
        light_of(&pyr)
      );
    }
    log_line("[observer] dust cadence: no GPU cadence telemetry (not built yet): the cadenced columns are n/a");
    let mut steps_done = 0u32;
    for k in 1..nf {
      if DEVICE_LOST.load(Ordering::Relaxed) {
        path_problems.push(format!("camera path frame {k}: GPU device lost"));
        break;
      }
      let t_step = Instant::now();
      // time (paused seek: the same incremental tier tick as play)
      if dt > 0.0 {
        let target = epoch0 + Duration::from_seconds(k as f64 * dt);
        let mut ok = ctx.seek_epoch_sync(scene_id, target);
        if !ok {
          let caught = wait_caught_up(&ctx, scene_id, 120.0);
          let now = current_epoch(&ctx, scene_id);
          ok = caught && now.map(|e| (e - target).abs() < Duration::from_seconds(1e-3)).unwrap_or(false);
        }
        if !ok {
          path_problems.push(format!("camera path frame {k}: seek to {target} failed"));
          break;
        }
      }
      if !wait_caught_up(&ctx, scene_id, 120.0) {
        path_problems.push(format!("camera path frame {k}: the tiers never caught up"));
      }
      // the camera: nucleus-relative, fixed world axes
      let comet_k = comet_position_au(&ctx, scene_id, comet_body).unwrap_or([0.0; 3]);
      let rel_k = path_rel(rel0, axis, omega, dolly_au, k);
      if norm(rel_k) * au_m <= NUCLEUS_RADIUS_KM as f64 * 1e3 + 1.0 {
        path_problems.push(format!("camera path frame {k}: the dolly reached the nucleus surface"));
        break;
      }
      let cam_pk = [comet_k[0] + rel_k[0], comet_k[1] + rel_k[1], comet_k[2] + rel_k[2]];
      let cam_qk = earth_observer::look_at_origin_from(rel_k, Some(axis));
      set_camera_pose(&ctx, scene_id, cam, cam_pk, cam_qk);
      // release exactly one frame
      ctx.set_render_budget(scene_id, pe, Some(1));
      let dl = wait_and_download(&ctx, args.width, args.height, 1, 120_000);
      let Some((mut rgba, _tid)) = dl else {
        path_problems.push(format!("camera path frame {k}: the released frame never arrived"));
        break;
      };
      // the ledger: exactly one new entry, drawn, with this step's inputs
      let mut entries = ctx.dust_frame_ledger(pe, prev_seq + 1);
      let t_l = Instant::now();
      while entries.is_empty() && t_l.elapsed().as_secs_f64() < 5.0 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        entries = ctx.dust_frame_ledger(pe, prev_seq + 1);
      }
      // the hold before step 0 drains the frames queued during the still phase until 0.6 s of
      // silence; a render stall ends it early and the rest (reused, step-0 inputs) are recorded
      // here. They are not this step's frame: drop them, the last entry is still checked in full.
      if entries.len() > 1 && entries[..entries.len() - 1].iter().all(|e| e.reused) {
        eprintln!(
          "[observer] camera path frame {k}: {} queued frame(s) from before the step (reused), skipped",
          entries.len() - 1
        );
        entries.drain(..entries.len() - 1);
      }
      let mut t_err = f64::NAN;
      let mut eye_err = f64::NAN;
      match entries.as_slice() {
        [e] => {
          prev_seq = e.seq;
          if e.reused {
            path_problems.push(format!("camera path frame {k}: the released frame reused the previous image"));
          }
          if let Some(tb) = t_base {
            t_err = (e.t_now_s - tb) - k as f64 * dt;
            if !(t_err.abs() <= 1e-3) {
              path_problems.push(format!("camera path frame {k}: drawn at t₀ + {:.6} s, not + {:.6} s", e.t_now_s - tb, k as f64 * dt));
            }
          }
          match e.nucleus_m {
            Some(n) => {
              let er = [e.eye_m[0] - n[0], e.eye_m[1] - n[1], e.eye_m[2] - n[2]];
              let want = [rel_k[0] * au_m, rel_k[1] * au_m, rel_k[2] * au_m];
              eye_err = norm(sub(er, want));
              if !(eye_err <= 1e-3 + 1e-12 * norm(want)) {
                path_problems.push(format!("camera path frame {k}: drawn with eye − nucleus {eye_err:.6} m off the path"));
              }
            }
            None => path_problems.push(format!("camera path frame {k}: the ledger has no nucleus")),
          }
        }
        es => {
          path_problems.push(format!("camera path frame {k}: {} dust frames for one step (one expected)", es.len()));
          if let Some(e) = es.last() {
            prev_seq = e.seq;
          }
        }
      }
      poll_timings(&mut timings);
      steps_done += 1;
      let pyr = ctx.download_dust_pyramid(pe);
      let light = pyr.as_ref().map(|p| light_of(p)).unwrap_or(f64::NAN);
      if keep.contains(&k) {
        if let Some(p) = pyr.as_ref() {
          let words: Vec<u8> = p.iter().flat_map(|w| w.to_le_bytes()).collect();
          let _ = std::fs::write(args.out.join(format!("frame_{k:03}.pyr")), &words);
          if args.save_snapshot {
            save_snapshot(&ctx, pe, p, &args, k as usize);
          }
        }
        bgra_to_rgba(&mut rgba);
        let _ = image::save_buffer(args.out.join(format!("frame_{k:03}.png")), &rgba, args.width, args.height, image::ColorType::Rgba8);
        let view_k = mc_view_reposed(&view_t0, cam_pk, cam_qk, comet_k);
        save_truth_inputs(&ctx, scene_id, &view_k, &args.out.join(format!("truth_{k:03}.bin")));
      }
      let tiers_k = tiers_now(&ctx, scene_id);
      let _ = writeln!(
        stats,
        "{{\"path_frame\":{k},\"epoch\":\"{}\",\"t_rel_s\":{},\"theta_deg\":{:.9},\"eye_rel_m\":[{:.6},{:.6},{:.6}],\"ledger\":{{\"seq\":{prev_seq},\"entries\":{},\"t_err_s\":{},\"eye_err_m\":{}}},\"light\":{:.6e},\"tiers_live\":[{}],\"wall_ms\":{:.1},\"variant\":\"{variant}\"}}",
        current_epoch(&ctx, scene_id).map(|e| e.to_string()).unwrap_or_default(),
        k as f64 * dt,
        (omega * k as f64).to_degrees(),
        rel_k[0] * au_m,
        rel_k[1] * au_m,
        rel_k[2] * au_m,
        entries.len(),
        json_f(t_err),
        json_f(eye_err),
        light,
        tiers_k.iter().map(|t| t.live.to_string()).collect::<Vec<_>>().join(","),
        t_step.elapsed().as_secs_f64() * 1e3
      );
      let _ = stats.flush();
      log_line(&format!(
        "[observer] camera path frame {k}: θ {:.4}°, Δt {:.0} s, ledger entries {}, t err {t_err:.2e} s, eye err {eye_err:.2e} m, light {light:.6e}, {:.0} ms",
        (omega * k as f64).to_degrees(),
        k as f64 * dt,
        entries.len(),
        t_step.elapsed().as_secs_f64() * 1e3
      ));
      // `AETHERVK_DUST_CADENCE_LOG=1`: the cadence's newest readback per path frame (diagnostics;
      // it lags the frame by the readback slots)
      if std::env::var("AETHERVK_DUST_CADENCE_LOG").is_ok_and(|v| v.trim() == "1")
        && let Some(c) = ctx.dust_budget_telemetry(pe).and_then(|b| b.cadence)
      {
        log_line(&format!(
          "[observer] cadence at path frame {k}: mode {:?}, reset {:?}, swaps {:?}, admitted {:?}, fresh {}/{}, lists {:?}, hist {:?}, aborts {:?}, images {}, shown light {:.4}, ghosts {}",
          c.mode, c.reset, c.swaps, c.admitted, c.fresh, c.live, c.lists, c.class_hist, c.aborts, c.images_shown, c.shown_light_share, c.ghost_entries
        ));
      }
    }
    // flush the timestamps (the readback lags DUST_READBACK_SLOTS frames): a few released frames
    // of the unchanged last view (reused images)
    for _ in 0..aethervk_core_rlib::gpu_backends::vulkan::device::dust::DUST_READBACK_SLOTS + 2 {
      ctx.set_render_budget(scene_id, pe, Some(1));
      if wait_and_download(&ctx, args.width, args.height, 1, 60_000).is_none() {
        break;
      }
      poll_timings(&mut timings);
    }
    ctx.set_render_budget(scene_id, pe, None);
    // the path's drawn frames are the last `steps_done` drawn ones measured (nothing else drew
    // while the budget held the view)
    let drawn: Vec<f64> = timings.values().filter(|t| !t.reused).map(|t| t.total_ms).collect();
    let path_ms: Vec<f64> = drawn.iter().rev().take(steps_done as usize).rev().copied().collect();
    // the path frames' stages (`DustGpuTimings`, one comma list per stage; diagnostics)
    let drawn_t: Vec<_> = timings.values().filter(|t| !t.reused).copied().collect();
    let path_t = &drawn_t[drawn_t.len().saturating_sub(steps_done as usize)..];
    let stage = |f: &dyn Fn(&aethervk_core_rlib::gpu_backends::vulkan::device::dust::DustGpuTimings) -> f64| {
      path_t.iter().map(|t| format!("{:.3}", f(t))).collect::<Vec<_>>().join(",")
    };
    let stages: Vec<(&str, String)> = vec![
      ("stage_propagate_ms", stage(&|t| t.propagate_ms)),
      ("stage_splat_ms", stage(&|t| t.splat_ms)),
      ("stage_splat_only_ms", stage(&|t| t.splat_only_ms)),
      ("stage_specks_ms", stage(&|t| t.specks_ms)),
      ("stage_queue_ms", stage(&|t| t.queue_ms)),
      ("stage_cadence_ms", stage(&|t| t.cadence_ms)),
      ("stage_classify_ms", stage(&|t| t.classify_ms)),
      ("stage_builds_ms", stage(&|t| t.builds_ms)),
      ("stage_reproject_ms", stage(&|t| t.reproject_ms)),
      ("stage_measure_ms", stage(&|t| t.measure_ms)),
      ("stage_composite_ms", stage(&|t| t.composite_ms)),
    ];
    let mut sorted = path_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let (ms_mean, ms_p90, ms_max) = if sorted.is_empty() {
      (f64::NAN, f64::NAN, f64::NAN)
    } else {
      (
        sorted.iter().sum::<f64>() / sorted.len() as f64,
        sorted[((sorted.len() as f64 * 0.9).ceil() as usize).clamp(1, sorted.len()) - 1],
        sorted[sorted.len() - 1],
      )
    };
    if (path_ms.len() as u32) < steps_done {
      log_line(&format!(
        "[observer] camera path: GPU timings for {} of {steps_done} path frames (a poll missed some)",
        path_ms.len()
      ));
    }
    let clean = path_problems.is_empty() && steps_done + 1 == nf;
    log_line(&format!(
      "[observer] camera path: {steps_done} of {} steps, clean {clean}, GPU ms per path frame mean {ms_mean:.2} p90 {ms_p90:.2} max {ms_max:.2} ({} frames measured)",
      nf - 1,
      path_ms.len()
    ));
    write_kv(
      &args.out.join("path_summary.txt"),
      &[
        ("clean", format!("{}", clean as u8)),
        ("steps", steps_done.to_string()),
        ("problems", path_problems.len().to_string()),
        ("gpu_frames", path_ms.len().to_string()),
        ("ms_mean", format!("{ms_mean:.3}")),
        ("ms_p90", format!("{ms_p90:.3}")),
        ("ms_max", format!("{ms_max:.3}")),
        ("gpu_ms", path_ms.iter().map(|m| format!("{m:.3}")).collect::<Vec<_>>().join(",")),
        ("cadence", "n/a".into()),
      ]
      .into_iter()
      .chain(stages.iter().map(|(k, v)| (*k, v.clone())))
      .collect::<Vec<_>>(),
    );
    if steps_done + 1 != nf && path_problems.is_empty() {
      path_problems.push(format!("camera path: {steps_done} of {} steps done", nf - 1));
    }
    problems.extend(path_problems);
  }

  // ───── contact sheet ─────
  if !args.no_sheet {
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
      // one second of slack: the provisional window's pinned sample sits at the ignition
      // instant itself (f32 rounding of the window end)
      if since < -1.0 && t.live > 0 {
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
    let flow_on = if args.view_flags & aethervk_core_rlib::scene::dust::DUST_VIEW_FLOW != 0 {
      "on"
    } else {
      "off"
    };
    let tracers_on = if args.view_flags & aethervk_core_rlib::scene::dust::DUST_VIEW_TRACERS != 0 {
      "on"
    } else {
      "off"
    };
    md.push_str(&format!(
      "- start: `{start}`  scene range end: `{end}`\n- frames: {n_frames} every {} h ({} days)\n- size: {}x{}  pose: {:?}\n- play mode: {:?}  reemit-every: {}\n- SPK: `{}`\n- CPU particles: {}\n- jets: {jets_desc}; d 100 µm, ρ 0.533 g/cm³, Afρ₀ 100 cm, power 2, cutoff 15 AU, nucleus 2 km, colour (1.0, 0.6, 0.2)\n- rotation: {rotation_desc}\n- ignition: {}\n- view: dust flags {} (flow {flow_on}, tracers {tracers_on}), flow speed 1.0, softening 1e-2\n- framing: half-height {half_height_km:.1} km = {half_height_au:.6} AU (3·coma {:.1} km vs β·g·T² {:.1} km, β_ref {:.4}, r₀ {r0_au:.4} AU), camera {:.1} km above the comet along +Z\n- Vulkan validation errors: {vk_errors}\n- wall time: {:.1} s\n\n",
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
      args.view_flags,
      3.0 * coma0_km,
      tail_km,
      cfg0.beta_ref,
      cam_height_au * AU_KM,
      t_run.elapsed().as_secs_f64()
    ));
    if persp_tan_h.is_some() {
      md.push_str(&format!(
        "- perspective: fov {fov_deg:.2}°{}; metres per pixel at the nucleus per frame in `stats.jsonl` (`camera.m_per_px_nucleus`)\n\n",
        if args.pose == Pose::Inside {
          format!(
            ", camera {:.6} m from the nucleus surface ({:.6} m from the centre) sunward on the anti-solar axis looking down the tail, near {:.6} m (truth {:.6} m: the engine near floor), far {:.3} km",
            args.inside_m,
            inside_au * au_m,
            inside_near_au * au_m,
            inside_truth_near_au * au_m,
            inside_far_now * AU_KM
          )
        } else {
          String::new()
        }
      ));
    }
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
    md.push_str("\n## hue ramp\n\nOKLab turn from the stream colour along the tail axis (10 samples outwards; dark and highlight samples skipped) of the displayed pixel, of the render's accumulated colour (the pyramid: the judged quantity, the pixel also shows the scene behind the dust) and of the truth's colour mix, the truth's mean age at the first and last lit sample (TTL, `--mc`), and the least render chroma over the truth's among the samples with dust in the pyramid.\n\n| # | display turn first → last (°) | render turn (°) | truth turn (°) | age first → last (TTL) | lit / dark / highlight | render chroma over truth min |\n|---|---|---|---|---|---|---|\n");
    for r in &records {
      let lit: Vec<&HueSample> = r.hue.iter().filter(|s| !s.dark && !s.highlight).collect();
      let (dark, highlight) = (
        r.hue.iter().filter(|s| s.dark).count(),
        r.hue.iter().filter(|s| s.highlight).count(),
      );
      let (stream_hue_deg, _) = oklab_hue_cl([
        jet_color[0] as f32,
        jet_color[1] as f32,
        jet_color[2] as f32,
      ]);
      let turn = |s: &HueSample| hue_turn_deg(s.hue_deg, stream_hue_deg).rem_euclid(360.0);
      let age =
        |s: &HueSample| s.age_ttl.map(|a| format!("{a:.2}")).unwrap_or_else(|| "–".to_string());
      let chroma_min = lit
        .iter()
        .filter(|s| s.pyr_tau >= HUE_PYR_FAINT_COUNTS)
        .filter_map(|s| match (s.pyr_chroma_rel, s.truth_chroma_rel) {
          (Some(pc), Some(tc)) => Some(pc / tc.max(1e-6)),
          _ => None,
        })
        .fold(f64::INFINITY, f64::min);
      let rturn = |s: &HueSample| {
        s.pyr_hue_deg
          .map(|h| format!("{:.0}", hue_turn_deg(h, stream_hue_deg).rem_euclid(360.0)))
          .unwrap_or_else(|| "–".to_string())
      };
      let tturn = |s: &HueSample| {
        s.truth_hue_deg
          .map(|h| format!("{:.0}", hue_turn_deg(h, stream_hue_deg).rem_euclid(360.0)))
          .unwrap_or_else(|| "–".to_string())
      };
      md.push_str(&format!(
        "| {} | {} | {} | {} | {} | {} / {dark} / {highlight} | {} |\n",
        r.index,
        if lit.is_empty() {
          "–".to_string()
        } else {
          format!("{:.0} → {:.0}", turn(lit[0]), turn(lit[lit.len() - 1]))
        },
        if lit.is_empty() {
          "–".to_string()
        } else {
          format!("{} → {}", rturn(lit[0]), rturn(lit[lit.len() - 1]))
        },
        if lit.is_empty() {
          "–".to_string()
        } else {
          format!("{} → {}", tturn(lit[0]), tturn(lit[lit.len() - 1]))
        },
        if lit.is_empty() {
          "–".to_string()
        } else {
          format!("{} → {}", age(lit[0]), age(lit[lit.len() - 1]))
        },
        lit.len(),
        if chroma_min.is_finite() {
          format!("{chroma_min:.2}")
        } else {
          "–".to_string()
        }
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
    if pyramid_saturated_texels(&words, w, h) > 0 {
      // a saturated frame (the unit lag after the ignition) has no shape to judge
      continue;
    }
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
    // the white point the composite showed this frame with (the adapted one), when the header
    // carries it; the recomputed instantaneous p99 otherwise
    let header_white = f32::from_bits(words[d::PYR_WHITE as usize]);
    let white = if header_white > 0.0 && header_white.is_finite() {
      header_white
    } else {
      white
    };
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
    // metres per pixel at the nucleus' depth (the perspective poses' scale there)
    let m_per_px = r.m_per_px_nucleus;
    let g_ms2 = 1.327_124_400_18e20 / (r.r_helio_au * d::AU_M).powi(2);
    let age_max_s = r.tiers.iter().map(|t| t.oldest_s).fold(0.0, f64::max);
    // the smallest grains of the size range (s_ref / SIZE_RANGE_FACTOR) carry
    // SIZE_RANGE_FACTOR × β_ref: they lead the tail's tip
    let v_max = v_eject + beta_ref * d::SIZE_RANGE_FACTOR * g_ms2 * age_max_s;
    let reach = (v_max * dt_s / m_per_px) as f32 + 3.0;
    // the dust region is seeded at the jet site, not the nucleus pixel: the nucleus is an
    // opaque disc (12 px at 50 km, the whole frame's centre at 5 km) and the dust leaves its
    // limb, so the seed is the brightest τ pixel within 1.5 nucleus radii of the nucleus pixel
    let nucleus = {
      let r_px = NUCLEUS_RADIUS_KM as f64 * 1e3 / m_per_px;
      let reach = 1.5 * r_px + 3.0;
      let (cx, cy) = r.nucleus_px;
      let (wu, hu) = (w as usize, h as usize);
      let (mut best, mut seed) = (0.0f32, [cx as f32, cy as f32]);
      for y in 0..hu {
        for x in 0..wu {
          let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
          if dx * dx + dy * dy <= reach * reach && img[y * wu + x] > best {
            best = img[y * wu + x];
            seed = [x as f32, y as f32];
          }
        }
      }
      seed
    };
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
      // slack for the threshold contour's quantisation: one pixel along a perimeter of ~4√area
      // (a 200-px coma at 960×600 grew 4.9× from day 1 to day 2 against the 4.7 bound, on the
      // GPU and the CPU alike), plus 16 px
      let slack = 16.0 + 4.0 * (prev.area_px as f32).sqrt();
      if prev.area_px >= 64
        && (st.area_ratio < 0.8 || sh.area_px as f32 > area_bound * prev.area_px as f32 + slack)
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
  /// metres per pixel at the nucleus' depth (`FrameRec::m_per_px_nucleus`): the orthographic
  /// poses' constant pixel scale, the perspective poses' scale at the nucleus
  m_per_px_nucleus: f64,
  aspect: f64,
  /// `Some(tan(fov/2))` for the perspective poses (`earth`, `inside`): the truth weights each
  /// grain by `1 / A_px(depth)`, `A_px = (2·depth·tan_h/h)²` m², the τ per pixel area at its
  /// depth; the phase eye is the camera point
  perspective_tan_h: Option<f64>,
  /// the render camera's near and far planes (AU, along the view direction): every pose clips
  /// there (the orthographic camera stands one half-height above the comet, and at 5 km the
  /// fast young grains pass it; the inside camera's near plane is D·1e-3; the truth must not
  /// count what the render clips)
  near_au: f64,
  far_au: f64,
  comet_au: V3,
  nucleus_px: (f64, f64),
  /// the truth holds this tier only (`--mc-tier`)
  mc_tier: Option<u32>,
  /// the truth holds the dust younger than this only (`--mc-age-max`, s)
  mc_age_max: Option<f64>,
  /// the truth holds the dust older than this only (`--mc-age-min`, s)
  mc_age_min: Option<f64>,
  /// nucleus radius on screen (px): the truth's peak is the jet site, up to this far from the
  /// nucleus pixel
  nucleus_px_radius: f64,
  /// the aperture Afρ (km) and the recipe's Afρ(r_h) (cm), `--afrho-aperture-km`
  afrho_aperture_km: Option<f64>,
  afrho_in_cm: f64,
  isotropic: bool,
  /// the hue ramp's ray: the truth places the samples along it and gives its age moments and
  /// colour there (`McRec::hue_px`, `hue_truth`); the stream colour and TTL of the age colour
  hue_ray: HueRay,
  stream: [f32; 3],
  ttl_s: f64,
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
/// `OBSERVER_MC_KICK_MAX=<k>`: the bound on a grain's whitened kicks (the speed noise and the
/// direction bandwidth's, or a packet's Gaussian across its capsule), in σ: the triple is redrawn
/// until `|g| ≤ k` — the 3-D standard normal truncated to the ball, as the kernels draw it
/// (`DUST_KICK_SIGMAS`). Default 3; `0` unbounded (the former draws, bit for bit)
/// the per-packet probe's volumetric deposits split above the per-add cap (`dust::take_volume_split_adds`
/// summed over its threads)
static PROBE_SPLIT_ADDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn mc_kick_max() -> f64 {
  static K: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
  *K.get_or_init(|| {
    std::env::var("OBSERVER_MC_KICK_MAX")
      .ok()
      .and_then(|v| v.trim().parse::<f64>().ok())
      .unwrap_or(3.0)
  })
}

/// `g` redrawn from `state` until `|g| ≤ k` ([`mc_kick_max`]; `k ≤ 0`: unchanged)
fn mc_bound_kick(mut g: [f64; 3], k: f64, state: &mut u64) -> [f64; 3] {
  if k > 0.0 {
    while g[0] * g[0] + g[1] * g[1] + g[2] * g[2] > k * k {
      g = [mc_gauss(state), mc_gauss(state), mc_gauss(state)];
    }
  }
  g
}

fn mc_gauss(state: &mut u64) -> f64 {
  let u1 = mc_rng(state).max(1e-300);
  let u2 = mc_rng(state);
  (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Level-0 texels of a pyramid dump whose count saturated the 32-bit word: the fixed-point unit
/// follows the previous frame's largest packet, and the first frame after the ignition (the
/// pinned sample alone, 1e-5 g) leaves the next frame's unit 1e9 too small — that frame's counts
/// are clipped and its τ image, shape and photometry mean nothing (the view recovers a frame
/// later from the frame's own τ_max). Returns the number of saturated texels.
fn pyramid_saturated_texels(words: &[u32], w: u32, h: u32) -> usize {
  use aethervk_core_rlib::scene::dust as d;
  let layout = d::PyramidLayout::new(w, h);
  if words.len() < layout.total_words as usize {
    return 0;
  }
  let (off, lw, lh) = layout.levels[0];
  let n = (lw * lh) as usize;
  (0..n)
    .filter(|&t| words[off as usize + t * d::PYRAMID_TEXEL_WORDS as usize] >= u32::MAX - 1)
    .count()
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
/// One emission batch the truth draws from: its descriptor, its tier's age band, the tier's
/// frame now (the diagnostic models only), the oldest tier's age fade and the tier.
#[derive(Clone, Copy)]
struct TruthBatch {
  desc: aethervk_core_rlib::scene::dust::DustBatch,
  band: (f64, f64),
  frame: Option<aethervk_core_rlib::scene::dust::DustFrame>,
  /// the oldest tier fades its dust over the last 10 % of the band (`dust_propagate.comp`:
  /// `clamp((1 − age/ttl)·10, 0, 1)` when `ttl.z` = 0); the younger tiers hand over sharply
  fade: bool,
  tier: u32,
}

/// Everything [`monte_carlo_truth_from`] reads of the live scene: the batches, the jet's time and
/// the nucleus centre now (m, heliocentric), the dust's geometric albedo (≤ 0 = unweighted τ).
/// `--save-snapshot` stores it (`truth_NNN.bin`) so the replay lab draws the same truth offline.
struct TruthInputs {
  batches: Vec<TruthBatch>,
  t_now: f64,
  r_centre_now: V3,
  albedo: f32,
}

fn gather_truth_inputs(ctx: &SimulationContext, scene_id: u64) -> Option<TruthInputs> {
  use aethervk_core_rlib::scene::dust as d;
  let mut batches: Vec<TruthBatch> = Vec::new();
  // (t_now, the nucleus centre's heliocentric position at the tick): the frame's `comet_au` is
  // the body centre, so the truth is placed relative to the centre (`JetState::centre_m`, the
  // render's anchor), not the jet site: the site sits one rotated offset away (2 km = 120 px at
  // the 5-km frame; placed at the site, the truth's apex sat on the centre pixel while the
  // render's dust left the limb)
  let mut jet_now: Option<(f64, V3)> = None;
  let mut albedo = 0.0f32;
  {
    let scene_arc = ctx.get_scene(scene_id)?;
    let g = scene_arc.read();
    g.scene.query1(|_, ps: &ParticleSystemComponent| {
      let sys = ps.dust.lock();
      for t in &sys.tiers {
        if let Some(j) = t.jet {
          jet_now = Some((j.t_s, j.centre_m()));
          albedo = sys.albedo;
        }
        let band = t.age_band_s(t.ttl_s);
        let frame: Option<d::DustFrame> =
          t.jet.and_then(|j| t.draw_state().map(|ds| ds.frame.at_time(j.t_s)));
        let fade = frame.as_ref().is_some_and(|f| f.ttl[2] < 0.5);
        for b in &t.ring.batches {
          batches.push(TruthBatch {
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
  let (t_now, r_centre_now) = jet_now?;
  Some(TruthInputs { batches, t_now, r_centre_now, albedo })
}

/// The truth's raw images besides the scored ones: τ per pixel (unboxed) and grains per pixel.
type McOut = (McRec, Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>);

fn monte_carlo_truth(
  ctx: &SimulationContext,
  scene_id: u64,
  pyr: &[u32],
  view: &McView,
  grains: usize,
) -> Option<(McRec, Vec<f32>, Vec<f32>)> {
  let inputs = gather_truth_inputs(ctx, scene_id)?;
  monte_carlo_truth_from(pyr, &inputs, view, grains).map(|(r, a, b, _, _)| (r, a, b))
}

fn monte_carlo_truth_from(
  pyr: &[u32],
  inputs: &TruthInputs,
  view: &McView,
  grains: usize,
) -> Option<McOut> {
  use aethervk_core_rlib::scene::dust as d;
  let (w, h) = (view.width as usize, view.height as usize);
  let render = tau_image_of_pyramid(pyr, view.width, view.height)?;
  if !render.iter().any(|&v| v > 0.0) {
    // the pyramid of this frame is empty (the first frame's readback lags the draw): nothing
    // to compare, the next frame will
    return None;
  }
  type B = TruthBatch;
  let mut batches: Vec<B> = inputs.batches.clone();
  let (t_now, r_centre_now, albedo) = (inputs.t_now, inputs.r_centre_now, inputs.albedo);
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
  if let Some(a) = view.mc_age_min {
    // every tier's band minimum raised there: every model gates its grains by `band.0`
    for b in &mut batches {
      b.band.0 = b.band.0.max(a);
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
      let m_per_px = view.m_per_px_nucleus;
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
  // the orthographic pixel area (m²): one constant for the whole frame
  let ortho_inv_a_px = 1.0 / (view.m_per_px_nucleus * view.m_per_px_nucleus);
  // a grain's pixel and `1 / A_px` at its depth (1/m²): the truth's histogram is τ per pixel
  // area, so a grain's cross-section is divided by the area one pixel covers where it is —
  // constant `m_per_px²` in the orthographic view, `(2·depth·tan_h/h)²` in perspective (a grain
  // near the camera covers more of its pixel). In perspective the pixel's column is oblique to
  // the z-planes `A_px` is measured on: the optical depth along the pixel's ray is
  // Σ σ / (A_px(depth) · cos θ_px), θ_px the angle between the pixel's ray and the camera
  // forward axis (cos θ_px = depth / |grain − camera|; 10–15 % at the edge of a 60° field).
  // Grains outside the camera's near/far are clipped like the render's.
  let project = |global_au: V3| -> Option<(usize, usize, f64)> {
    let dd = sub(global_au, view.cam_pos);
    let (sx, sy) = (dot(dd, right), dot(dd, up));
    let depth = dot(dd, forward);
    if !(depth >= view.near_au && depth <= view.far_au) {
      return None;
    }
    let (px, py, inv_a_px) = match view.perspective_tan_h {
      Some(tan_h) => {
        let px_m = 2.0 * depth * d::AU_M * tan_h / h as f64;
        let cos_theta = depth / norm(dd).max(1e-300);
        (
          w as f64 / 2.0 * (1.0 + sx / depth / (tan_h * view.aspect)),
          h as f64 / 2.0 * (1.0 - sy / depth / tan_h),
          1.0 / (px_m * px_m * cos_theta),
        )
      }
      None => (
        w as f64 / 2.0 * (1.0 + sx / (view.half_height_au * view.aspect)),
        h as f64 / 2.0 * (1.0 - sy / view.half_height_au),
        ortho_inv_a_px,
      ),
    };
    if px >= 0.0 && py >= 0.0 && px < w as f64 && py < h as f64 {
      Some((px as usize, py as usize, inv_a_px))
    } else {
      None
    }
  };
  let threads = observer_threads();
  // the diagnostic models draw per cluster of the whole batch: a batch is one piece there
  let whole_batches = std::env::var("OBSERVER_MC_MODEL").is_ok_and(|v| v == "chords" || v == "clusters");
  // one threaded pass over the batches with `alloc(batch index)` grains each: the histogram,
  // the age-weighted histogram, the grains drawn and the grains in view per batch
  let run = |alloc: &(dyn Fn(usize) -> usize + Sync)| -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
  ) {
    let mut truth = vec![0.0f64; w * h];
    let mut age_sum = vec![0.0f64; w * h];
    let mut age2_sum = vec![0.0f64; w * h];
    let mut col_sum = vec![0.0f64; 3 * w * h];
    let mut drawn = 0usize;
    let mut inview = vec![0usize; batches.len()];
    // per batch: Σ of its deposits' squared weights (the variance it adds: Neyman allocation)
    let mut sqs = vec![0.0f64; batches.len()];
    let mut counts = vec![0.0f64; w * h];
    // the work in fixed units of ~[`MC_UNIT_GRAINS`] grains, cut from the batches in order (small
    // batches packed together, large ones split into pieces of their grain range, each piece its
    // own stream: [`mc_piece_seed`]), whatever the thread count: the Neyman allocation gives a few
    // batches nearly all grains, and one thread per fixed share of the batches drew them alone
    // `OBSERVER_MC_UNIT_GRAINS=<n>` (diagnostic): another unit size (≥ every batch's grains: no
    // batch split, the streams of the former per-batch draw)
    let unit_grains = std::env::var("OBSERVER_MC_UNIT_GRAINS")
      .ok()
      .and_then(|v| v.trim().parse::<usize>().ok())
      .filter(|&n| n >= 1)
      .unwrap_or(MC_UNIT_GRAINS);
    let mut units: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    {
      let (mut cur, mut cur_n) = (Vec::new(), 0usize);
      for bi in 0..batches.len() {
        let n_b = alloc(bi);
        if n_b == 0 {
          cur.push((bi, 0, 0));
          continue;
        }
        let step = if whole_batches { n_b } else { unit_grains };
        let mut lo = 0;
        while lo < n_b {
          let hi = (lo + step).min(n_b);
          cur.push((bi, lo, hi));
          cur_n += hi - lo;
          lo = hi;
          if cur_n >= unit_grains {
            units.push(core::mem::take(&mut cur));
            cur_n = 0;
          }
        }
      }
      if !cur.is_empty() {
        units.push(cur);
      }
    }
    // the units' histograms are summed in unit order (bit-identical at any thread count); a
    // worker waits before starting a unit more than `window` past the next one to sum, which
    // bounds the finished-but-unsummed units held in memory
    let window = threads + 2;
    let next_unit = std::sync::atomic::AtomicUsize::new(0);
    let order = std::sync::Mutex::new((0usize, std::collections::BTreeMap::new()));
    let ready = std::sync::Condvar::new();
    let mut rhist_sum = vec![0.0f64; 0];
    let rhist_on = std::env::var("OBSERVER_LAB_RHIST").is_ok_and(|v| v == "1");
    // `OBSERVER_MC_W2_NOISE=1` (diagnostic, the replay lab): Σ w² per pixel into [`MC_W2`]
    let w2_on = mc_w2_on();
    let sink = std::sync::Mutex::new((
      &mut truth,
      &mut age_sum,
      &mut age2_sum,
      &mut col_sum,
      &mut counts,
      &mut drawn,
      &mut inview,
      &mut sqs,
      &mut rhist_sum,
    ));
    std::thread::scope(|sc| {
      for _ in 0..threads.min(units.len()).max(1) {
        let (project, units, next_unit, order, ready, sink, batches) = (&project, &units, &next_unit, &order, &ready, &sink, &batches);
        sc.spawn(move || loop {
          let u = next_unit.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
          if u >= units.len() {
            break;
          }
          {
            let mut g = order.lock().unwrap();
            while u >= g.0 + window {
              g = ready.wait(g).unwrap();
            }
          }
          let pieces = &units[u];
          let out = {
            let mut hist = vec![0.0f64; w * h];
            let mut age_hist = vec![0.0f64; w * h];
            let mut age2_hist = vec![0.0f64; w * h];
            let mut col_hist = vec![0.0f64; 3 * w * h];
            let mut cnt_hist = vec![0.0f64; w * h];
            let mut w2_hist = vec![0.0f64; if w2_on { w * h } else { 0 }];
            let mut n_drawn = 0usize;
            let mut seen: Vec<(usize, usize, f64)> = Vec::with_capacity(pieces.len());
            let sq = core::cell::Cell::new(0.0f64);
            // one grain into the histograms — the single deposit every model goes through (the
            // physical grains and the `OBSERVER_MC_MODEL` diagnostics alike, so they cannot
            // drift): `r` the grain's heliocentric position (m), `age` (s), `wgt` its
            // cross-section (m²). Physical photometry: p · Φ(α) / r_h² at the grain (the render
            // weights each texel the same way at the dust it holds); the orthographic poses' eye
            // is at infinity along the view axis (the comet → camera line), like the render's (a
            // point eye at the camera lit the dust beside the camera at α → 180°), the
            // perspective poses' the camera point. The oldest tier's age fade (the TTL end), the
            // camera's near/far clip and τ per pixel area (`project`: 1/(A_px(depth)·cos θ_px)
            // in perspective, 1/m_per_px² orthographic); the ages and the age colour carry the
            // radiance weight (the display mixes colours by it). `true` when the grain landed.
            let mut rhist = vec![0.0f64; if rhist_on { RHIST_BINS } else { 0 }];
            // `OBSERVER_LAB_RHIST_INVIEW=1`: the histogram of the light that lands in view (its τ
            // weight) instead of every grain's cross-section
            let rhist_inview = std::env::var("OBSERVER_LAB_RHIST_INVIEW").is_ok_and(|v| v == "1");
            // the grain's cell coordinates for [`lab_cell_mode`]: its speed along the jet's axis at
            // emission over v_ref, and its angle off that axis (°); set by the physical grains
            let cell_tag = core::cell::Cell::new([f64::NAN; 4]);
            let cell_mode = lab_cell_mode();
            let mut deposit = |r: V3, age: f64, wgt: f64, b: &B| -> bool {
              let mut rk = None;
              if rhist_on {
                // `OBSERVER_LAB_RHIST_MODE`: `eye` the distance from the camera, `theta` / `phi`
                // the direction from the centre (polar angle to the spin axis, azimuth from the
                // Sun about it, 4° bins), else the distance from the nucleus centre
                let dr = sub(r, r_centre_now);
                let rn = norm(r_centre_now);
                let sun = [-r_centre_now[0] / rn, -r_centre_now[1] / rn, -r_centre_now[2] / rn];
                let ax = [b.desc.spin[0] as f64, b.desc.spin[1] as f64, b.desc.spin[2] as f64];
                let k = match std::env::var("OBSERVER_LAB_RHIST_MODE").as_deref() {
                  _ if cell_mode.is_some() => {
                    let cm = cell_mode.unwrap();
                    let t = cell_tag.get();
                    let view_angle = || {
                      let fwd = camera_axes(view.cam_q).1;
                      let dc = sub(r, [view.cam_pos[0] * d::AU_M, view.cam_pos[1] * d::AU_M, view.cam_pos[2] * d::AU_M]);
                      (dot(dc, fwd) / norm(dc).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees()
                    };
                    lab_cell_bin(
                      match cm.0 {
                        "speed" => t[0],
                        "off" => t[1],
                        "tail" => t[2],
                        "fbin" => t[3].log10(),
                        "view" => view_angle(),
                        _ => age,
                      },
                      cm,
                    )
                  }
                  Ok("eye") => {
                    let dc = norm(sub(r, [view.cam_pos[0] * d::AU_M, view.cam_pos[1] * d::AU_M, view.cam_pos[2] * d::AU_M]));
                    ((dc / RHIST_BIN_M) as usize).min(RHIST_BINS - 1)
                  }
                  Ok(m @ ("theta" | "phi")) => lab_angle_bin(m == "theta", [dr[0], dr[1], dr[2]], ax, sun),
                  _ => ((norm(dr) / RHIST_BIN_M) as usize).min(RHIST_BINS - 1),
                };
                let gated = lab_view_max().is_some_and(|g| {
                  let fwd = camera_axes(view.cam_q).1;
                  let dc = sub(r, [view.cam_pos[0] * d::AU_M, view.cam_pos[1] * d::AU_M, view.cam_pos[2] * d::AU_M]);
                  (dot(dc, fwd) / norm(dc).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees() > g
                });
                if gated {
                } else if rhist_inview {
                  rk = Some(k);
                } else {
                  rhist[k] += if b.fade { wgt * ((1.0 - age / b.band.1) * 10.0).clamp(0.0, 1.0) } else { wgt };
                }
              }
              let w_rad = if albedo > 0.0 {
                let rn = norm(r);
                let sun = [
                  (-r[0] / rn) as f32,
                  (-r[1] / rn) as f32,
                  (-r[2] / rn) as f32,
                ];
                let to_eye = if view.perspective_tan_h.is_some() {
                  [
                    (view.cam_pos[0] * d::AU_M - r[0]) as f32,
                    (view.cam_pos[1] * d::AU_M - r[1]) as f32,
                    (view.cam_pos[2] * d::AU_M - r[2]) as f32,
                  ]
                } else {
                  let e = normalize(sub(view.cam_pos, view.comet_au));
                  [e[0] as f32, e[1] as f32, e[2] as f32]
                };
                d::radiance_weight(albedo, sun, to_eye, (rn / d::AU_M) as f32) as f64
              } else {
                1.0
              };
              let global = [
                view.comet_au[0] + (r[0] - r_centre_now[0]) / d::AU_M,
                view.comet_au[1] + (r[1] - r_centre_now[1]) / d::AU_M,
                view.comet_au[2] + (r[2] - r_centre_now[2]) / d::AU_M,
              ];
              let wf = if b.fade {
                wgt * ((1.0 - age / b.band.1) * 10.0).clamp(0.0, 1.0)
              } else {
                wgt
              };
              let Some((x, y, inv_a_px)) = project(global) else {
                return false;
              };
              let wv = wf * w_rad * inv_a_px;
              if let Some(k) = rk {
                rhist[k] += wv;
              }
              sq.set(sq.get() + wv * wv);
              hist[y * w + x] += wv;
              age_hist[y * w + x] += wv * age;
              age2_hist[y * w + x] += wv * age * age;
              let rgb = d::age_color(view.stream, age as f32, view.ttl_s as f32);
              for k in 0..3 {
                col_hist[3 * (y * w + x) + k] += wv * rgb[k] as f64;
              }
              cnt_hist[y * w + x] += 1.0;
              if w2_on {
                w2_hist[y * w + x] += wv * wv;
              }
              true
            };
            for &(bi, lo, hi) in pieces {
              let b = &batches[bi];
              let desc = &b.desc;
              let mass = desc.mass_params[0] as f64;
              let n_b = alloc(bi);
              let drawn_before = n_drawn;
              let sq_before = sq.get();
              if n_b == 0 {
                // not drawn in this pass (a pilot escalation of other batches)
                seen.push((bi, 0, 0.0));
                continue;
              }
              // the grain's share of the batch's cross-section (m²): mass × the distribution's
              // cross-section per gram (`emit_cluster`); the radiance weight is per grain below
              let rho_g_m3 = desc.mass_params[1] as f64 * 1.0e6;
              let xsec_per_g =
                3.0 / (4.0 * rho_g_m3 * d::size_mean_inv_s_um(desc.size_params) as f64 * 1.0e-6);
              let wgt = mass * xsec_per_g / n_b as f64;
              let mut rng = mc_piece_seed(desc.seed, lo / unit_grains);
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
                  // the cluster's age gates it into its tier's band like the render's
                  let t0 = c.r0_t0_hi[3] as f64 + c.r0_t0_lo[3] as f64;
                  let age = t_now - t0;
                  if !(age >= b.band.0 && age < b.band.1) {
                    continue;
                  }
                  // the cluster's cross-section (m²) shared by its grains
                  let m_c = c.v0_lo_mass[3] as f64;
                  let wgt_c = m_c * xsec_per_g / per as f64;
                  for _ in 0..per {
                    let bin = (mc_rng(&mut rng) * 16.0).floor().min(15.0) as usize;
                    let u = mc_rng(&mut rng);
                    let (a0, a1) = (m.edge(bin), m.edge(bin + 1));
                    let f = (m.edge_factor(bin) as f64 + m.edge_factor(bin + 1) as f64) * 0.5;
                    let [g1, g2, g3] =
                      mc_bound_kick([mc_gauss(&mut rng), mc_gauss(&mut rng), mc_gauss(&mut rng)], mc_kick_max(), &mut rng);
                    let dx = f * (l11 * g1);
                    let dy = f * (l21 * g1 + l22 * g2);
                    let dz = f * (l31 * g1 + l32 * g2 + l33 * g3);
                    let local = [
                      a0[0] as f64 + u * (a1[0] as f64 - a0[0] as f64) + dx,
                      a0[1] as f64 + u * (a1[1] as f64 - a0[1] as f64) + dy,
                      a0[2] as f64 + u * (a1[2] as f64 - a0[2] as f64) + dz,
                    ];
                    // heliocentric (m): the anchor is the frame's heliocentric origin
                    let r = [
                      anchor[0] + local[0],
                      anchor[1] + local[1],
                      anchor[2] + local[2],
                    ];
                    if deposit(r, age, wgt_c, b) {
                      n_drawn += 1;
                    }
                  }
                }
                seen.push((bi, n_drawn - drawn_before, sq.get() - sq_before));
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
                  // the cluster's cross-section (m²) shared by its grains
                  let wgt_c = m_c * xsec_per_g / per as f64;
                  for _ in 0..per {
                    let sb = sb_lo + (sb_hi - sb_lo) * mc_rng(&mut rng);
                    let beta = sb * sb;
                    let f = sb / beta_c.sqrt();
                    let [g1, g2, g3] =
                      mc_bound_kick([mc_gauss(&mut rng), mc_gauss(&mut rng), mc_gauss(&mut rng)], mc_kick_max(), &mut rng);
                    // the dispersion scales with the grain's speed factor f, as the render's
                    // (`bin_packet`: Σ_v · f²) and the physical truth's (σ · v with v ∝ f); unscaled,
                    // the largest grains (f ≈ 0.3) got 3× their spread and ~10 % went backward
                    let (slf, srf) = (sl * f, sr * f);
                    let v = [
                      v0[0] + ej[0] * (f - 1.0) + slf * (g1 * e1[0] + g2 * e2[0]) + srf * g3 * e3[0],
                      v0[1] + ej[1] * (f - 1.0) + slf * (g1 * e1[1] + g2 * e2[1]) + srf * g3 * e3[1],
                      v0[2] + ej[2] * (f - 1.0) + slf * (g1 * e1[2] + g2 * e2[2]) + srf * g3 * e3[2],
                    ];
                    let (r, _) =
                      d::kepler::propagate_f64(r0, v, d::SUN_MU_M3_S2 * (1.0 - beta), age);
                    if deposit(r, age, wgt_c, b) {
                      n_drawn += 1;
                    }
                  }
                }
                seen.push((bi, n_drawn - drawn_before, sq.get() - sq_before));
                continue;
              }
              // cross-section weighted size: pdf ∝ s^(e−2), CDF exponent p = e − 1
              let p = e - 1.0;
              // OBSERVER_MC_QUANTISE_PHASE=1 (diagnostic): the ejection direction is taken at the
              // instant of the cluster the grain falls into (the renderer's model: one attitude per
              // time sample) instead of the grain's own instant
              let quantise = std::env::var("OBSERVER_MC_QUANTISE_PHASE").is_ok_and(|v| v == "1");
              let lat_coupled = !std::env::var("OBSERVER_MC_LAT_COUPLED").is_ok_and(|v| v == "0");
              let kick_max = mc_kick_max();
              let samples = (desc.count >> d::batch_streams(desc).0).max(1) as f64;
              let streams = (1u32 << d::batch_streams(desc).0) as f64;
              // `OBSERVER_MC_SIGMA_ANG=<rad>` (diagnostic): the direction bandwidth instead of the
              // renderer's stream cell (0: the hard-edged cone, the emission model itself)
              let sigma_ang = std::env::var("OBSERVER_MC_SIGMA_ANG")
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or((aperture / streams.sqrt()).max(d::CHILD_SIGMA_V_REL as f64));
              if cell_mode.is_some() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                  let (f_lo, f_hi) = ((beta_s / s_max / beta_ref).sqrt(), (beta_s / s_min / beta_ref).sqrt());
                  println!(
                    "lab: truth jet: aperture {aperture:.4} rad, |jet_dir| {:.4}, σ_ang {sigma_ang:.4}, streams {streams}, v_ref {v_ref:.3} m/s, σ_v/v {v_std_rel:.3}, f {f_lo:.4}..{f_hi:.4}, ω {omega:.3e}",
                    norm(jet_dir)
                  );
                });
              }
              for _ in lo..hi {
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
                // the renderer's direction kernel: the cone is covered by S Gaussian stream
                // cells of lateral dispersion σ_lat = v·max(aperture/√S, CHILD_SIGMA_V_REL)
                // (the KDE bandwidth, one cone cell), so the continuous emission is drawn
                // through that bandwidth and the truth is the cone convolved with it (as in
                // `cell_kernels_reproduce_the_continuous_emission`): a sharp cone edge would
                // differ by a third at the rim of the 5-km frame
                let (g1, g2) = (mc_gauss(&mut rng), mc_gauss(&mut rng));
                let lat_ax = if dir[0].abs() < 0.9 {
                  [1.0, 0.0, 0.0]
                } else {
                  [0.0, 1.0, 0.0]
                };
                let e1 = normalize(cross(lat_ax, dir));
                let e2 = cross(dir, e1);
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
                let n_v = mc_gauss(&mut rng);
                // the speed noise and the bandwidth's kick bounded together (`mc_kick_max`)
                let [g1, g2, n_v] = mc_bound_kick([g1, g2, n_v], kick_max, &mut rng);
                let lat = [
                  sigma_ang * (g1 * e1[0] + g2 * e2[0]),
                  sigma_ang * (g1 * e1[1] + g2 * e2[1]),
                  sigma_ang * (g1 * e1[2] + g2 * e2[2]),
                ];
                let v = (v_ref * f * (1.0 + v_std_rel * n_v)).max(0.0);
                let r0 = [
                  rc[0] + o_t0[0] - o_start[0],
                  rc[1] + o_t0[1] - o_start[1],
                  rc[2] + o_t0[2] - o_start[2],
                ];
                // `OBSERVER_MC_LAT_COUPLED=0` (diagnostic): the bandwidth's kick at the bin's
                // mean speed `v_ref·f` instead of the grain's own — a cell's lateral width then
                // no longer grows with its distance from the site (the packets' assumption)
                let v0 = if lat_coupled {
                  [
                    vc[0] + (dir[0] + lat[0]) * v + v_site[0],
                    vc[1] + (dir[1] + lat[1]) * v + v_site[1],
                    vc[2] + (dir[2] + lat[2]) * v + v_site[2],
                  ]
                } else {
                  let v_lat = v_ref * f;
                  [
                    vc[0] + dir[0] * v + lat[0] * v_lat + v_site[0],
                    vc[1] + dir[1] * v + lat[1] * v_lat + v_site[1],
                    vc[2] + dir[2] * v + lat[2] * v_lat + v_site[2],
                  ]
                };
                if cell_mode.is_some() {
                  let ax_e = mc_qrot(q, jet_dir);
                  let u_rel = sub(sub(v0, vc), v_site);
                  let along = dot(u_rel, ax_e);
                  cell_tag.set([
                    along / v_ref,
                    (along / norm(u_rel).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees(),
                    (n_v * n_v + g1 * g1 + g2 * g2).sqrt(),
                    f,
                  ]);
                }
                let (r, _) = d::kepler::propagate_f64(r0, v0, d::SUN_MU_M3_S2 * (1.0 - beta), age);
                if deposit(r, age, wgt, b) {
                  n_drawn += 1;
                }
              }
              seen.push((bi, n_drawn - drawn_before, sq.get() - sq_before));
            }
            (hist, age_hist, age2_hist, col_hist, cnt_hist, n_drawn, seen, rhist, w2_hist)
          };
          let mut g = order.lock().unwrap();
          g.1.insert(u, out);
          while let Some((hist, ah, a2h, colh, ch, n, seen, rh, w2h)) = {
            let k = g.0;
            g.1.remove(&k)
          } {
            let mut acc = sink.lock().unwrap();
            let (truth, age_sum, age2_sum, col_sum, counts, drawn, inview, sqs, rhist_sum) = &mut *acc;
            for (a, b) in counts.iter_mut().zip(ch.iter()) {
              *a += *b;
            }
            for (a, b) in truth.iter_mut().zip(hist.iter()) {
              *a += *b;
            }
            for (a, b) in age_sum.iter_mut().zip(ah.iter()) {
              *a += *b;
            }
            for (a, b) in age2_sum.iter_mut().zip(a2h.iter()) {
              *a += *b;
            }
            for (a, b) in col_sum.iter_mut().zip(colh.iter()) {
              *a += *b;
            }
            // a batch split over units: its pieces' in-view grains and Σw² add up
            for (bi, k, q) in seen {
              inview[bi] += k;
              sqs[bi] += q;
            }
            **drawn += n;
            if rhist_on {
              rhist_sum.resize(RHIST_BINS, 0.0);
              for (a, b) in rhist_sum.iter_mut().zip(rh.iter()) {
                *a += *b;
              }
            }
            if w2_on {
              // in unit order, like the histograms: bit-identical at any thread count
              let mut q = MC_W2.lock().unwrap();
              q.resize(w * h, 0.0);
              for (a, b) in q.iter_mut().zip(w2h.iter()) {
                *a += *b;
              }
            }
            g.0 += 1;
          }
          drop(g);
          ready.notify_all();
        });
      }
    });
    drop(sink);
    if rhist_on {
      let mut g = RHIST.lock().unwrap();
      g.resize(RHIST_BINS, 0.0);
      for (a, b) in g.iter_mut().zip(rhist_sum.iter()) {
        *a += *b;
      }
    }
    (truth, age_sum, age2_sum, col_sum, drawn, inview, counts, sqs)
  };
  // importance allocation: a pilot of 64 grains per batch measures the fraction landing in the
  // frame; the grains are then allotted by mass × that fraction (floor 1/32: no batch starved).
  // The weight per grain stays mass / n_b, so the estimate is unbiased whatever n_b; a near
  // frame holds a few windows of the youngest tier and would otherwise spend 99 % of the
  // grains out of view
  let (_, _, _, _, _, mut pilot, _, mut pilot_sq) = run(&|_| 64usize);
  // the pilot's grains per batch: a batch whose pilot landed nothing in the frame is drawn again
  // with 64× the grains (4 096, 262 144, …) while the pilots stay within a quarter of the budget.
  // With one 64-grain pilot such a batch got the 64-grain floor: inside the coma at 100 m the
  // young cone's edge reaches the field with p ≈ 7.5·10⁻⁶ per grain, every young pilot missed it
  // and the truth held **no** young dust (a third of the real light there)
  let mut pilot_n = vec![64usize; batches.len()];
  {
    let mut spent = 64 * batches.len();
    let mut n_next = 64usize * 64;
    loop {
      let blind: Vec<usize> = (0..batches.len()).filter(|&bi| pilot[bi] == 0).collect();
      if blind.is_empty() || spent + blind.len() * n_next > grains / 4 {
        break;
      }
      let is_blind: Vec<bool> = (0..batches.len()).map(|bi| pilot[bi] == 0).collect();
      let (_, _, _, _, _, hits, _, sq) = run(&|bi| if is_blind[bi] { n_next } else { 0 });
      for &bi in &blind {
        pilot[bi] = hits[bi];
        pilot_sq[bi] = sq[bi];
        pilot_n[bi] = n_next;
      }
      spent += blind.len() * n_next;
      n_next *= 64;
    }
    let blind = (0..batches.len()).filter(|&bi| pilot[bi] == 0).count();
    if pilot_n.iter().any(|&n| n > 64) {
      log_line(&format!(
        "mc pilots: {} batches needed a larger pilot (up to {} grains), {blind} still saw nothing in the frame",
        pilot_n.iter().filter(|&&n| n > 64).count(),
        pilot_n.iter().max().copied().unwrap_or(64)
      ));
    }
  }
  RHIST.lock().unwrap().clear();
  // the pilots deposited too: only the final draw's Σ w² is kept
  MC_W2.lock().unwrap().clear();
  let weights: Vec<f64> = batches
    .iter()
    .zip(pilot.iter().zip(pilot_n.iter()))
    .map(|(b, (&k, &n))| b.desc.mass_params[0] as f64 * (k as f64 / n as f64))
    .collect();
  // half of the grains by the rule above, half by each batch's standard deviation √Σw² from the
  // pilot (Neyman: the variance of Σ (mass/n_b)·g is mass²·E[g²]/n_b) — in a perspective view
  // inside the coma a few old grains near the camera carry 1/d² weights and dominated the noise
  // (1 km: the background read 0.9 % of the truth's light instead of ~40 %, judged rms 51 →
  // 0.216 with it); still unbiased (the weight per grain is mass/n_b). `OBSERVER_MC_ALLOC=mass`:
  // the former allocation alone
  let weights: Vec<f64> = if !std::env::var("OBSERVER_MC_ALLOC").is_ok_and(|v| v == "mass") {
    let wt: f64 = weights.iter().sum::<f64>().max(1e-300);
    // √(n·Σw²) is the batch's standard deviation whatever its pilot size (Σw² ∝ 1/n); scaled by
    // n/64 so the common case (every pilot 64) is unchanged bit for bit
    let sd: Vec<f64> = pilot_sq
      .iter()
      .zip(pilot_n.iter())
      .map(|(q, &n)| (q.max(0.0) * (n as f64 / 64.0)).sqrt())
      .collect();
    let st: f64 = sd.iter().sum::<f64>().max(1e-300);
    weights.iter().zip(sd.iter()).map(|(a, b)| 0.5 * a / wt + 0.5 * b / st).collect()
  } else {
    weights
  };
  // `OBSERVER_MC_ALLOC=uniform` (diagnostic): the same grains per batch whatever the pilot saw —
  // a check that a batch whose 64-grain pilot missed the frame (weight 0: the 64-grain floor)
  // really holds nothing in view
  let weights: Vec<f64> = if std::env::var("OBSERVER_MC_ALLOC").is_ok_and(|v| v == "uniform") {
    vec![1.0; batches.len()]
  } else {
    weights
  };
  let weight_total: f64 = weights.iter().sum::<f64>().max(1e-300);
  let n_of = |bi: usize| ((grains as f64 * weights[bi] / weight_total).round() as usize).max(64);
  // OBSERVER_MC_YOUNG=1 (diagnostic): the youngest batches (ending within 2 h of now) with
  // their lit map at u = 1 (the engine's pinned sample), pilot hits and grain allotment
  if std::env::var("OBSERVER_MC_YOUNG").is_ok_and(|v| v == "1") {
    for (bi, b) in batches.iter().enumerate() {
      let d = &b.desc;
      let t_start = d.comet_r_t_hi[3] as f64 + d.comet_r_t_lo[3] as f64;
      let dur = d.comet_v_dur_hi[3] as f64 + d.comet_v_dur_lo[3] as f64;
      if t_start + dur < t_now - 7200.0 {
        continue;
      }
      let map1 = d::lit_time_map(1.0, d.lit, d.spin[3], dur as f32);
      log_line(&format!(
        "[observer] mc young batch {bi}: vel_params {:?} size_params {:?} aperture {:.4} tier {} t_end-now {:.0} s dur {:.0} s count {} mass {:.1} g lit {:?} (mode {}) provisional {} map(u=1) {:.0} s pilot {} alloc {}",
        d.vel_params,
        d.size_params,
        d.jet_dir_aperture[3],
        b.tier,
        t_start + dur - t_now,
        dur,
        d.count,
        d.mass_params[0],
        [d.lit[0], d.lit[1], d.lit[2]],
        d.lit[3],
        d::batch_provisional(d),
        map1,
        pilot[bi],
        n_of(bi)
      ));
    }
  }
  let (truth, age_sum, age2_sum, col_sum, drawn, _, counts, _) = run(&n_of);
  // `OBSERVER_MC_RAW=<prefix>` (the replay lab): the truth's raw τ image and grains per pixel,
  // f32 little endian, `<prefix>_truth.f32` / `<prefix>_counts.f32`
  if let Ok(prefix) = std::env::var("OBSERVER_MC_RAW") {
    let f = |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect() };
    let _ = std::fs::write(format!("{prefix}_truth.f32"), f(&truth));
    let _ = std::fs::write(format!("{prefix}_counts.f32"), f(&counts));
  }
  // absolute photometry: the render's radiance factors (per px², `PYR_UNIT` makes the pyramid
  // absolute) against the truth's Σ cross-section × weight / A_px(depth) — both τ per pixel
  // area, in perspective the pixel area taken at each grain's depth (no frame-wide `m_per_px`)
  let abs_ratio = {
    let truth_sum: f64 = truth.iter().sum::<f64>();
    let render_sum: f64 = render.iter().map(|&v| v as f64).sum();
    if truth_sum > 0.0 {
      render_sum / truth_sum
    } else {
      0.0
    }
  };
  // the aperture Afρ: `4 r_h² / (π ρ) · Σ_{within ρ} RF · A_px` (A'Hearn et al. 1984 with Δ
  // cancelling in the orthographic view; the image's RF already carries 1/r_h²), the truth the
  // same from its Σ cross-section × weight, and the recipe's Afρ(r_h)·Φ(α_c) at the comet.
  // Orthographic poses only: a perspective frame has no single pixel area to turn the aperture's
  // RF sum back into a cross-section
  if view.afrho_aperture_km.is_some() && view.perspective_tan_h.is_some() {
    log_line(
      "[observer] afrho: the aperture Afρ is reported for the orthographic poses only (skipped)",
    );
  }
  let afrho = view
    .afrho_aperture_km
    .filter(|_| view.perspective_tan_h.is_none())
    .map(|rho_km| {
      let m_per_px = view.m_per_px_nucleus;
      let a_px = m_per_px * m_per_px;
      let rho = rho_km * 1e3;
      let (cx, cy) = view.nucleus_px;
      let (mut s_r, mut s_t) = (0.0f64, 0.0f64);
      for y in 0..h {
        for x in 0..w {
          let (dx, dy) = (
            (x as f64 + 0.5 - cx) * m_per_px,
            (y as f64 + 0.5 - cy) * m_per_px,
          );
          if dx * dx + dy * dy <= rho * rho {
            s_r += render[y * w + x] as f64 * a_px;
            s_t += truth[y * w + x] * a_px;
          }
        }
      }
      let r_h = norm(view.comet_au);
      let to_afrho = |s: f64| 4.0 * r_h * r_h / (core::f64::consts::PI * rho) * s * 100.0;
      let sun = [
        -view.comet_au[0] / r_h,
        -view.comet_au[1] / r_h,
        -view.comet_au[2] / r_h,
      ];
      let to_eye = normalize(sub(view.cam_pos, view.comet_au));
      let alpha = dot(sun, to_eye).clamp(-1.0, 1.0).acos();
      let analytic = view.afrho_in_cm * d::phase_function(alpha as f32) as f64;
      (to_afrho(s_r), to_afrho(s_t), analytic)
    });
  // OBSERVER_MC_AGE=<file>: the truth's radiance-weighted mean age per pixel (f32 seconds, 0
  // where empty)
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
  // the hue samples: along the ray to the end of the truth's region, the truth's age moments
  // and colour mix in a 5×5 box at each
  let (hue_px, hue_truth, hue_tau_ratio) = {
    let ray = &view.hue_ray;
    let reach = ray.reach(view.width, view.height);
    let mut t_end = ray.r0;
    let mut t = ray.r0;
    while t <= reach {
      let (cx, cy) = ray.at(t);
      let (ix, iy) = (cx.round() as i64, cy.round() as i64);
      if ix >= 0
        && iy >= 0
        && ix < w as i64
        && iy < h as i64
        && b[iy as usize * w + ix as usize] >= MC_REGION_REL * peak
      {
        t_end = t;
      }
      t += 1.0;
    }
    let px = ray.samples(t_end);
    let ratio_at: Vec<Option<f64>> = px
      .iter()
      .map(|&(cx, cy)| {
        let (ix, iy) = (cx.round() as i64, cy.round() as i64);
        let (mut sa, mut sb) = (0.0f64, 0.0f64);
        for y in iy - 2..=iy + 2 {
          for x in ix - 2..=ix + 2 {
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
              continue;
            }
            let i = (y * w as i64 + x) as usize;
            sa += a[i];
            sb += b[i];
          }
        }
        (sa > 0.0 || sb > 0.0).then(|| if sb > 0.0 { sa / sb } else { f64::INFINITY })
      })
      .collect();
    let truth_at: Vec<Option<(f64, f64, [f64; 3])>> = px
      .iter()
      .map(|&(cx, cy)| {
        let (ix, iy) = (cx.round() as i64, cy.round() as i64);
        let (mut st, mut sa, mut sa2, mut sc) = (0.0f64, 0.0f64, 0.0f64, [0.0f64; 3]);
        for y in iy - 2..=iy + 2 {
          for x in ix - 2..=ix + 2 {
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
              continue;
            }
            let i = (y * w as i64 + x) as usize;
            st += truth[i];
            sa += age_sum[i];
            sa2 += age2_sum[i];
            for k in 0..3 {
              sc[k] += col_sum[3 * i + k];
            }
          }
        }
        (st > 0.0).then(|| {
          let m = sa / st;
          (
            m,
            (sa2 / st - m * m).max(0.0).sqrt(),
            [sc[0] / st, sc[1] / st, sc[2] / st],
          )
        })
      })
      .collect();
    (px, truth_at, ratio_at)
  };
  let haze_frac = {
    let total: f64 = a.iter().sum();
    let inside: f64 = region.iter().map(|&i| a[i]).sum();
    if total > 0.0 {
      (total - inside) / total
    } else {
      0.0
    }
  };
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
  // the frame mapping: the τ-weighted centroids of the truth's and the render's cores (the
  // pixels above 1 % of the truth's peak, box filtered) coincide within a nucleus radius plus
  // 2 % of the frame (the peak itself is a weak statistic on a weighted haze: with the phase
  // function the brightest box can sit at the forward-scattering edge of a near view)
  let centroid_of = |img: &[f64]| {
    let (mut sx, mut sy, mut st) = (0.0f64, 0.0f64, 0.0f64);
    for &i in &region {
      if b[i] >= 0.01 * peak {
        let v = img[i].max(0.0);
        sx += v * ((i % w) as f64 + 0.5);
        sy += v * ((i / w) as f64 + 0.5);
        st += v;
      }
    }
    if st > 0.0 {
      (sx / st, sy / st)
    } else {
      (0.0, 0.0)
    }
  };
  let (bx, by) = centroid_of(&b);
  let (ax, ay) = centroid_of(&a);
  let tol = 3.0 + 1.5 * view.nucleus_px_radius + 0.02 * h as f64;
  let nucleus_ok = (bx - ax).abs() <= tol && (by - ay).abs() <= tol;
  let rec = McRec {
    abs_ratio,
    afrho,
    peak_px: (bx, by),
    render_peak_px: (ax, ay),
    nucleus_px_radius: view.nucleus_px_radius,
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
    hue_px,
    hue_truth,
    hue_tau_ratio,
    haze_frac,
  };
  let truth32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
  let render32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
  Some((rec, render32, truth32, truth, counts))
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

// ───────────────────────────── sky-image registration check ─────────────────────────────

/// `--sky-image-check`: the synthetic image's size (px) and its crosses: a 3×3 grid at these
/// FITS pixels (1-based, the pixel centres), arms ±`SKY_CHECK_ARM` px, 3 px thick (symmetric:
/// the centroid is the centre; 9 crosses cover 1.3 % of the image, above the 99.5th-percentile
/// white point the loader takes)
const SKY_CHECK_N: u32 = 256;
const SKY_CHECK_AT: [f64; 3] = [64.0, 128.0, 192.0];
const SKY_CHECK_ARM: i64 = 8;
/// the largest allowed distance (frame px) between a detected cross and its WCS pixel
const SKY_CHECK_TOL_PX: f64 = 1.0;

/// A primary-HDU FITS file, BITPIX −32, `nx × ny`, `data` row-major from `p2 = 1`, with `cards`
/// (key, value already formatted) after the mandatory ones.
fn fits_f32(nx: u32, ny: u32, data: &[f32], cards: &[(&str, String)]) -> Vec<u8> {
  let mut h = String::new();
  let mut card = |k: &str, v: &str| {
    let c = format!("{k:<8}= {v:>20}");
    h.push_str(&format!("{c:<80}"));
  };
  card("SIMPLE", "T");
  card("BITPIX", "-32");
  card("NAXIS", "2");
  card("NAXIS1", &nx.to_string());
  card("NAXIS2", &ny.to_string());
  for (k, v) in cards {
    card(k, v);
  }
  h.push_str(&format!("{:<80}", "END"));
  while h.len() % 2880 != 0 {
    h.push(' ');
  }
  let mut out = h.into_bytes();
  for v in data {
    out.extend_from_slice(&v.to_be_bytes());
  }
  while out.len() % 2880 != 0 {
    out.push(0);
  }
  out
}

/// `--sky-image-check` for frame `i`: writes `sky_check_NNN.fits` (TAN WCS centred on the
/// camera → comet direction's RA/Dec, `CDELT` so the image spans half the frame height, north up
/// and east left), loads it as the scene's sky overlay (opacity 1, linear), renders, saves
/// `sky_check_NNN.png`, removes the overlay, and finds each cross's luminance centroid in a
/// window around the pixel the observer's own perspective maths gives its WCS direction. Returns
/// the `--check` failures (a cross missing or farther than [`SKY_CHECK_TOL_PX`]).
fn sky_image_check(
  ctx: &SimulationContext,
  scene_id: u64,
  cam: EntityId,
  comet_body: EntityId,
  args: &Args,
  i: u32,
  fov_deg: f64,
) -> Vec<String> {
  use aethervk_core_rlib::{
    scene::sky_image as sk,
    simulation_api::astrometry::{
      ecliptic_to_equatorial_j2000, equatorial_to_ecliptic_j2000, ra_dec_of,
    },
  };
  let mut problems = Vec::new();
  let (w, h) = (args.width as usize, args.height as usize);
  let tan_h = (fov_deg.to_radians() / 2.0).tan();
  let aspect = args.width as f64 / args.height as f64;
  let Some((cam_pos, _)) = camera_pose(ctx, scene_id, cam) else {
    return vec![format!("frame {i}: sky-image check: no camera pose")];
  };
  let Some(comet) = comet_position_au(ctx, scene_id, comet_body) else {
    return vec![format!("frame {i}: sky-image check: no comet position")];
  };
  // the image's centre: the camera → comet direction's astrometric RA/Dec (rad → degrees)
  let (ra, dec) = ra_dec_of(ecliptic_to_equatorial_j2000(sub(comet, cam_pos)));
  let (ra_deg, dec_deg) = (ra.to_degrees(), dec.to_degrees());
  // degrees per image pixel: the image spans half the frame's vertical field
  let cdelt = fov_deg / 2.0 / SKY_CHECK_N as f64;
  let n = SKY_CHECK_N as i64;
  let mut data = vec![0.0f32; (n * n) as usize];
  for &cy in &SKY_CHECK_AT {
    for &cx in &SKY_CHECK_AT {
      let (x0, y0) = (cx as i64 - 1, cy as i64 - 1); // 0-based storage index of the centre
      for a in -SKY_CHECK_ARM..=SKY_CHECK_ARM {
        for t in -1..=1 {
          for (x, y) in [(x0 + a, y0 + t), (x0 + t, y0 + a)] {
            if x >= 0 && y >= 0 && x < n && y < n {
              data[(y * n + x) as usize] = 1.0;
            }
          }
        }
      }
    }
  }
  let crpix = (SKY_CHECK_N as f64 + 1.0) / 2.0;
  let cards = [
    ("CTYPE1", "'RA---TAN'".to_string()),
    ("CTYPE2", "'DEC--TAN'".to_string()),
    ("RADESYS", "'ICRS'".to_string()),
    ("CRVAL1", format!("{ra_deg:.12}")),
    ("CRVAL2", format!("{dec_deg:.12}")),
    ("CRPIX1", format!("{crpix:.3}")),
    ("CRPIX2", format!("{crpix:.3}")),
    ("CDELT1", format!("{:.12E}", -cdelt)),
    ("CDELT2", format!("{:.12E}", cdelt)),
  ];
  let path = args.out.join(format!("sky_check_{i:03}.fits"));
  if let Err(e) = std::fs::write(&path, fits_f32(SKY_CHECK_N, SKY_CHECK_N, &data, &cards)) {
    return vec![format!(
      "frame {i}: sky-image check: cannot write {}: {e}",
      path.display()
    )];
  }
  let img = match sk::SkyImage::load(&path.to_string_lossy(), None) {
    Ok(img) => img,
    Err(e) => return vec![format!("frame {i}: sky-image check: load failed: {e}")],
  };
  let Some(wcs) = img.wcs else {
    return vec![format!(
      "frame {i}: sky-image check: the loader refused the WCS: {:?}",
      img.wcs_error
    )];
  };
  let Some(scene) = ctx.get_scene(scene_id) else {
    return vec![format!("frame {i}: sky-image check: no scene")];
  };
  let key = sk::scene_key(&scene);
  drop(scene);
  if !sk::store_insert(key, &img) {
    return vec![format!(
      "frame {i}: sky-image check: store_insert refused the image"
    )];
  }
  sk::store_set_params(
    key,
    sk::SkyImageParams {
      opacity: 1.0,
      stretch: 0.0,
      visible: true,
    },
  );
  let dl = wait_and_download(ctx, args.width, args.height, args.frames_to_wait, 60_000);
  sk::store_remove(key);
  let Some((mut rgba, _)) = dl else {
    return vec![format!(
      "frame {i}: sky-image check: render download failed"
    )];
  };
  bgra_to_rgba(&mut rgba);
  let _ = image::save_buffer(
    args.out.join(format!("sky_check_{i:03}.png")),
    &rgba,
    args.width,
    args.height,
    image::ColorType::Rgba8,
  );
  // the pose the overlay was drawn with (the Earth observer re-aims every tick: read it now)
  // (directions at infinity: the rotation alone places them)
  let Some((_, cam_q)) = camera_pose(ctx, scene_id, cam) else {
    return vec![format!(
      "frame {i}: sky-image check: no camera pose after the render"
    )];
  };
  let (right, forward, up) = camera_axes(cam_q);
  let lum = |x: usize, y: usize| -> f64 {
    let o = (y * w + x) * 4;
    0.2126 * rgba[o] as f64 + 0.7152 * rgba[o + 1] as f64 + 0.0722 * rgba[o + 2] as f64
  };
  // one image pixel in frame pixels, and the search window (half the cross spacing)
  let img_px = (h as f64 / 2.0) / SKY_CHECK_N as f64;
  let win = ((SKY_CHECK_AT[1] - SKY_CHECK_AT[0]) * img_px / 2.0).max(6.0);
  let mut worst = 0.0f64;
  let mut k = 0;
  for &cy in &SKY_CHECK_AT {
    for &cx in &SKY_CHECK_AT {
      k += 1;
      // the WCS direction of the cross's centre → the frame pixel (directions at infinity)
      let dir = equatorial_to_ecliptic_j2000(wcs.pixel_to_equatorial(cx, cy));
      let (sx, sy, depth) = (dot(dir, right), dot(dir, up), dot(dir, forward));
      if !(depth > 0.0) {
        problems.push(format!(
          "frame {i}: sky cross {k} ({cx}, {cy}) behind the camera"
        ));
        continue;
      }
      let px = w as f64 / 2.0 * (1.0 + sx / depth / (tan_h * aspect));
      let py = h as f64 / 2.0 * (1.0 - sy / depth / tan_h);
      // the window's background (its 20th percentile) and the luminance centroid of what stands
      // above half way to its peak
      let (x0, x1) = (
        (px - win).floor().max(0.0) as usize,
        ((px + win).ceil() as usize).min(w),
      );
      let (y0, y1) = (
        (py - win).floor().max(0.0) as usize,
        ((py + win).ceil() as usize).min(h),
      );
      if x0 >= x1 || y0 >= y1 {
        problems.push(format!(
          "frame {i}: sky cross {k} predicted at ({px:.2}, {py:.2}) is off the frame"
        ));
        continue;
      }
      let mut vals: Vec<f64> = Vec::new();
      for y in y0..y1 {
        for x in x0..x1 {
          vals.push(lum(x, y));
        }
      }
      vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
      let bg = vals[vals.len() / 5];
      let peak = *vals.last().unwrap();
      if peak - bg < 32.0 {
        problems.push(format!(
          "frame {i}: sky cross {k} not found near ({px:.2}, {py:.2}) (peak {peak:.0} over background {bg:.0})"
        ));
        continue;
      }
      let thr = bg + 0.5 * (peak - bg);
      let (mut sx_, mut sy_, mut st) = (0.0f64, 0.0f64, 0.0f64);
      for y in y0..y1 {
        for x in x0..x1 {
          let v = lum(x, y) - bg;
          if v + bg >= thr {
            sx_ += v * (x as f64 + 0.5);
            sy_ += v * (y as f64 + 0.5);
            st += v;
          }
        }
      }
      let (fx, fy) = (sx_ / st, sy_ / st);
      let err = ((fx - px).powi(2) + (fy - py).powi(2)).sqrt();
      worst = worst.max(err);
      log_line(&format!(
        "[observer] sky cross frame {i} #{k} (FITS {cx}, {cy}): predicted ({px:.2}, {py:.2}), found ({fx:.2}, {fy:.2}), error {err:.3} px"
      ));
      if err > SKY_CHECK_TOL_PX {
        problems.push(format!(
          "frame {i}: sky cross {k} (FITS {cx}, {cy}) {err:.2} px from its WCS pixel (tolerance {SKY_CHECK_TOL_PX} px)"
        ));
      }
    }
  }
  log_line(&format!(
    "[observer] sky-image check frame {i}: centre RA {ra_deg:.6}° Dec {dec_deg:.6}°, {cdelt:.3e}°/px ({img_px:.3} frame px per image px), worst error {worst:.3} px"
  ));
  problems
}

/// `--pose earth`: the camera position, its forward axis, the camera → comet direction, the angle
/// between them and the comet's elevation above the site's (geocentric) horizon
fn earth_pose_diag(
  ctx: &SimulationContext,
  scene_id: u64,
  cam: EntityId,
  comet_body: EntityId,
  earth_body: EntityId,
  i: u32,
  tag: &str,
) -> (String, Option<f64>) {
  let (Some((cp, cq)), Some(c), Some(e)) = (
    camera_pose(ctx, scene_id, cam),
    comet_position_au(ctx, scene_id, comet_body),
    comet_position_au(ctx, scene_id, earth_body),
  ) else {
    return (
      format!("earth pose frame {i} ({tag}): camera/comet/Earth transform missing"),
      None,
    );
  };
  let (_, fwd, _) = camera_axes(cq);
  let to = normalize(sub(c, cp));
  let zenith = normalize(sub(cp, e));
  let angle = dot(fwd, to).clamp(-1.0, 1.0).acos().to_degrees();
  let elev = dot(to, zenith).clamp(-1.0, 1.0).asin().to_degrees();
  let line = format!(
    "earth pose frame {i} ({tag}): camera {} AU ({:.1} km from the Earth centre), forward {}, to comet {}, angle {angle:.4}°, comet elevation {elev:.2}°",
    json_v3(cp),
    norm(sub(cp, e)) * AU_KM,
    json_v3(fwd),
    json_v3(to)
  );
  (line, Some(elev))
}

/// `--probe`: the judged frame's GPU clusters (`download_dust_snapshot`) replayed tier by tier on
/// the CPU mirror into a fresh pyramid with the GPU frame's header (its host-written words; the
/// counters zeroed), with the τ at the probe pixels attributed to clusters. First the fidelity of
/// the replay: its composite τ against the GPU's over the frame (the attribution is only as good
/// as that); then per pixel the largest contributors. Written to `probe_NNN.txt` and the log.
fn probe_replay(
  ctx: &SimulationContext,
  pe: aethervk_core_rlib::gpu::PresentationEngineHandle,
  gpu_pyr: &[u32],
  args: &Args,
  i: usize,
) {
  use aethervk_core_rlib::scene::dust as d;
  let Some(snap) = ctx.download_dust_snapshot(pe) else {
    log_line(&format!("[observer] probe frame {i}: no snapshot (no view or device)"));
    return;
  };
  if snap.is_empty() {
    log_line(&format!("[observer] probe frame {i}: empty snapshot (CPU particle mode?)"));
    return;
  }
  let layout = d::PyramidLayout::new(args.width, args.height);
  if layout.total_words as usize != gpu_pyr.len() {
    log_line(&format!(
      "[observer] probe frame {i}: layout {} words vs the GPU pyramid's {}",
      layout.total_words,
      gpu_pyr.len()
    ));
    return;
  }
  let h = d::PYRAMID_HEADER_WORDS as usize;
  let mut pyr = vec![0u32; gpu_pyr.len()];
  pyr[..h].copy_from_slice(&gpu_pyr[..h]);
  // the counters the passes accumulate start at zero (the host-written words are kept)
  {
    use aethervk_core_rlib::scene::dust::specks as sp;
    for w in [
      d::PYR_TAU_MAX,
      d::PYR_LEVEL_MASK,
      d::PYR_SPLIT_CHILDREN,
      d::PYR_VOLUME_EVALS,
      d::PYR_EWA_TEXELS,
      d::PYR_SPLIT_DROPPED_FLUX,
      sp::PYR_SPECKS,
      sp::PYR_SPECKS_OVERFLOW,
      sp::PYR_SPECKS_TRUNCATED,
      sp::PYR_N_EYE,
      sp::PYR_N_EYE_HI,
      sp::PYR_SPECKS_NODES,
      sp::PYR_SPECKS_COUNTS,
    ] {
      pyr[w as usize] = 0;
    }
  }
  let t0 = Instant::now();
  let mut reports: Vec<(usize, d::ProbeReport)> = Vec::new();
  for (k, t) in snap.iter().enumerate() {
    let r = d::splat_tier_probed(
      &t.moments,
      &t.render,
      &t.pc,
      &t.flow,
      t.tracer_counts,
      &layout,
      &mut pyr,
      &args.probe,
    );
    reports.extend(r.into_iter().map(|x| (k, x)));
  }
  // fidelity: the replay's composite τ against the GPU frame's (the same unit: the same pc)
  let unit = snap
    .first()
    .map(|t| if t.pc.inv_unit > 0.0 { 1.0 / t.pc.inv_unit } else { 1.0 })
    .unwrap_or(1.0);
  let (mut se, mut s2, mut sa, mut sb, mut peak) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
  let mut px = Vec::with_capacity((args.width * args.height) as usize);
  for y in 0..args.height {
    for x in 0..args.width {
      let a = d::composite_sample(&layout, &pyr, unit, x, y).0 as f64;
      let b = d::composite_sample(&layout, gpu_pyr, unit, x, y).0 as f64;
      peak = peak.max(b);
      px.push((a, b));
    }
  }
  for &(a, b) in &px {
    sa += a;
    sb += b;
    if b > 0.01 * peak {
      se += (a - b) * (a - b);
      s2 += b * b;
    }
  }
  let mut out = format!(
    "probe frame {i}: replay of {} tiers in {:.1} s; fidelity: Στ replay/GPU {:.4}, rel rms over the pixels above 1 % of the peak {:.4}\n",
    snap.len(),
    t0.elapsed().as_secs_f64(),
    sa / sb.max(1e-300),
    (se / s2.max(1e-300)).sqrt()
  );
  for &(x, y) in &args.probe {
    let gpu_tau = d::composite_sample(&layout, gpu_pyr, unit, x, y).0;
    let tiers: Vec<&(usize, d::ProbeReport)> =
      reports.iter().filter(|(_, r)| r.pixel == (x, y)).collect();
    let total: f32 = tiers.iter().map(|(_, r)| r.total).sum();
    out += &format!("pixel ({x},{y}): GPU τ {gpu_tau:.4e}, replay Σ cluster τ {total:.4e}\n");
    let mut all: Vec<(usize, d::ProbeHit)> = tiers
      .iter()
      .flat_map(|(k, r)| r.hits.iter().map(move |h| (*k, *h)))
      .collect();
    all.sort_by(|a, b| b.1.tau.partial_cmp(&a.1.tau).unwrap_or(std::cmp::Ordering::Equal));
    for (k, h) in all.iter().take(10) {
      out += &format!(
        "  tier {k}: {:.1}% {}\n",
        100.0 * h.tau / total.max(1e-30),
        h
      );
    }
    for (k, r) in &tiers {
      out += &format!("  tier {k} total {:.4e} from {} clusters\n", r.total, r.clusters);
    }
  }
  for l in out.lines() {
    log_line(&format!("[observer] {l}"));
  }
  let _ = std::fs::write(args.out.join(format!("probe_{i:03}.txt")), out);
}

// ───── the offline replay lab ─────

const SNAPSHOT_MAGIC: &[u8; 4] = b"DSNP";

/// `--save-snapshot`: the frame's GPU splat input (`download_dust_snapshot`) and the GPU
/// pyramid's header, to `snapshot_NNN.bin`: magic, version 1, width, height, tier count, the 256
/// header words, then per tier the push constants, the flow clock (3 f32), the tracer counts,
/// the render clusters and the moments (counts then raw bytes).
fn save_snapshot(
  ctx: &SimulationContext,
  pe: aethervk_core_rlib::gpu::PresentationEngineHandle,
  gpu_pyr: &[u32],
  args: &Args,
  i: usize,
) {
  use aethervk_core_rlib::scene::dust as d;
  let Some(snap) = ctx.download_dust_snapshot(pe) else {
    log_line(&format!("[observer] snapshot frame {i}: no view or device"));
    return;
  };
  let mut b: Vec<u8> = Vec::new();
  let u32le = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
  b.extend_from_slice(SNAPSHOT_MAGIC);
  u32le(&mut b, 1);
  u32le(&mut b, args.width);
  u32le(&mut b, args.height);
  u32le(&mut b, snap.len() as u32);
  for w in &gpu_pyr[..d::PYRAMID_HEADER_WORDS as usize] {
    u32le(&mut b, *w);
  }
  for t in &snap {
    b.extend_from_slice(bytemuck::bytes_of(&t.pc));
    for v in [t.flow.t_hi, t.flow.t_lo, t.flow.speed] {
      b.extend_from_slice(&v.to_le_bytes());
    }
    u32le(&mut b, t.tracer_counts);
    u32le(&mut b, t.render.len() as u32);
    b.extend_from_slice(bytemuck::cast_slice(&t.render));
    u32le(&mut b, t.moments.len() as u32);
    b.extend_from_slice(bytemuck::cast_slice(&t.moments));
  }
  let path = args.out.join(format!("snapshot_{i:03}.bin"));
  match std::fs::write(&path, &b) {
    Ok(()) => log_line(&format!(
      "[observer] snapshot frame {i}: {} tiers, {:.1} MB → {}",
      snap.len(),
      b.len() as f64 / 1e6,
      path.display()
    )),
    Err(e) => log_line(&format!("[observer] snapshot frame {i}: {e}")),
  }
}

/// The tiers of a loaded snapshot: push constants, flow, tracer count, render clusters, moments.
type LabTiers = Vec<(
  aethervk_core_rlib::scene::dust::DustSplatPushConstants,
  aethervk_core_rlib::scene::dust::DustFlowUniform,
  u32,
  Vec<aethervk_core_rlib::scene::dust::DustRenderCluster>,
  Vec<aethervk_core_rlib::scene::dust::DustMoments>,
)>;

/// A loaded snapshot: `(width, height, header, tiers)`.
type LabSnapshot = (
  u32,
  u32,
  Vec<u32>,
  Vec<(
    aethervk_core_rlib::scene::dust::DustSplatPushConstants,
    aethervk_core_rlib::scene::dust::DustFlowUniform,
    u32,
    Vec<aethervk_core_rlib::scene::dust::DustRenderCluster>,
    Vec<aethervk_core_rlib::scene::dust::DustMoments>,
  )>,
);

fn load_snapshot(path: &Path) -> Option<LabSnapshot> {
  use aethervk_core_rlib::scene::dust as d;
  let b = std::fs::read(path).ok()?;
  let mut o = 0usize;
  let mut take = |n: usize| -> Option<&[u8]> {
    let s = b.get(o..o + n)?;
    o += n;
    Some(s)
  };
  if take(4)? != SNAPSHOT_MAGIC {
    return None;
  }
  let rd = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
  let _version = rd(take(4)?);
  let (w, h, nt) = (rd(take(4)?), rd(take(4)?), rd(take(4)?));
  let header: Vec<u32> = (0..d::PYRAMID_HEADER_WORDS).map(|_| take(4).map(rd)).collect::<Option<_>>()?;
  let mut tiers = Vec::new();
  for _ in 0..nt {
    let pc: d::DustSplatPushConstants =
      bytemuck::pod_read_unaligned(take(core::mem::size_of::<d::DustSplatPushConstants>())?);
    let f = |s: &[u8]| f32::from_le_bytes([s[0], s[1], s[2], s[3]]);
    let flow = d::DustFlowUniform {
      t_hi: f(take(4)?),
      t_lo: f(take(4)?),
      speed: f(take(4)?),
    };
    let tracer = rd(take(4)?);
    let nr = rd(take(4)?) as usize;
    let rs = core::mem::size_of::<d::DustRenderCluster>();
    let render: Vec<d::DustRenderCluster> = take(nr * rs)?
      .chunks_exact(rs)
      .map(bytemuck::pod_read_unaligned)
      .collect();
    let nm = rd(take(4)?) as usize;
    let ms = core::mem::size_of::<d::DustMoments>();
    let moments: Vec<d::DustMoments> = take(nm * ms)?
      .chunks_exact(ms)
      .map(bytemuck::pod_read_unaligned)
      .collect();
    tiers.push((pc, flow, tracer, render, moments));
  }
  Some((w, h, header, tiers))
}

fn read_f32_file(path: &Path) -> Option<Vec<f64>> {
  let b = std::fs::read(path).ok()?;
  Some(
    b.chunks_exact(4)
      .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
      .collect(),
  )
}

const TRUTH_MAGIC: &[u8; 4] = b"DTRU";

/// `OBSERVER_LAB_RHIST=1` (the replay lab): the truth's cross-section by distance from the
/// nucleus centre (every grain drawn, in view or not; [`RHIST_BIN_M`] bins), for the packets'
/// same histogram ([`lab_packet_mc`])
static RHIST: std::sync::Mutex<Vec<f64>> = std::sync::Mutex::new(Vec::new());
/// `OBSERVER_MC_W2_NOISE=1` (diagnostic, off by default; the image and every score unchanged):
/// the truth's final draw also sums each grain's squared τ weight per pixel, so the lab prints a
/// second judged value whose noise term is the weighted variance Σw² (boxed) / (Σw)² instead of
/// the grain count's 1/n — the perspective weight 1/(A_px·cos θ) ∝ 1/depth² makes the weights
/// heavy-tailed near the eye, where 1/n understates the noise
static MC_W2: std::sync::Mutex<Vec<f64>> = std::sync::Mutex::new(Vec::new());
fn mc_w2_on() -> bool {
  std::env::var("OBSERVER_MC_W2_NOISE").is_ok_and(|v| v == "1")
}
const RHIST_BIN_M: f64 = 250.0;
const RHIST_BINS: usize = 90;

/// The cell-coordinate modes of `OBSERVER_LAB_RHIST_MODE` (the young cone's own coordinates, the
/// packets' and the truth's alike): `speed` the distance along the cell's axis in units of the
/// reference speed's travel (the speed factor f, 0.02 bins from −0.5: behind the origin is
/// negative), `off` the angle off the axis (1° bins), `age` the age (30 s bins; the packets' at
/// their piece's centre age), `tail` the draw's whitened 3-D deviation (0.1 bins: the truth's
/// √(N² + g₁² + g₂²) of its speed and direction kicks, the packets' |g| of their Gaussian across
/// the capsule), `view` the angle from the camera's forward axis (1° bins: the frame's edge),
/// `fbin` log₁₀ of the speed factor (0.05 bins from −0.5: the truth's grain's own f, the packets'
/// size bin's mid f). A value that is not finite (a grain of a diagnostic model, a packet without a cell) lands in the last bin
fn lab_cell_mode() -> Option<(&'static str, f64)> {
  match std::env::var("OBSERVER_LAB_RHIST_MODE").as_deref() {
    Ok("speed") => Some(("speed", 0.02)),
    Ok("off") => Some(("off", 1.0)),
    Ok("age") => Some(("age", 30.0)),
    Ok("tail") => Some(("tail", 0.1)),
    Ok("view") => Some(("view", 1.0)),
    Ok("fbin") => Some(("fbin", 0.05)),
    _ => None,
  }
}

/// The bin of `x` in the cell mode `(name, width)` ([`lab_cell_mode`]; `speed` from −0.5)
fn lab_cell_bin(x: f64, (name, width): (&str, f64)) -> usize {
  let x = x + lab_cell_first(name);
  if x.is_finite() { ((x.max(0.0) / width) as usize).min(RHIST_BINS - 1) } else { RHIST_BINS - 1 }
}

/// The lower edge of the first bin of a cell mode, negated
fn lab_cell_first(name: &str) -> f64 {
  if name == "speed" || name == "fbin" { 0.5 } else { 0.0 }
}

/// `OBSERVER_LAB_VIEW_MAX=<deg>`: the histograms (both sides) count only what lies within this
/// angle of the camera's forward axis — the part of a cluster at the frame's edge
fn lab_view_max() -> Option<f64> {
  static G: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
  *G.get_or_init(|| std::env::var("OBSERVER_LAB_VIEW_MAX").ok().and_then(|v| v.trim().parse::<f64>().ok()))
}

/// The direction bin of `p` (from the nucleus centre): `theta` the polar angle to the unit spin
/// `axis` in 2° bins, else the azimuth about it from the Sun's projection in 4° bins
fn lab_angle_bin(theta: bool, p: V3, axis: V3, sun: V3) -> usize {
  let pn = normalize(p);
  let c = dot(pn, axis).clamp(-1.0, 1.0);
  if theta {
    return ((c.acos().to_degrees() / 2.0) as usize).min(RHIST_BINS - 1);
  }
  let s0 = normalize(sub(sun, [axis[0] * dot(sun, axis), axis[1] * dot(sun, axis), axis[2] * dot(sun, axis)]));
  let s1 = cross(axis, s0);
  let phi = dot(pn, s1).atan2(dot(pn, s0)).to_degrees().rem_euclid(360.0);
  ((phi / 4.0) as usize).min(RHIST_BINS - 1)
}

/// `--save-snapshot`: the truth's inputs ([`TruthInputs`]) and the frame's [`McView`] to
/// `truth_NNN.bin`, so `--lab --lab-truth N` draws the same truth offline (any grain count, any
/// age gate). Every scalar as f64 LE (NaN = none), the descriptors and frames as their Pod bytes.
fn save_truth_inputs(ctx: &SimulationContext, scene_id: u64, view: &McView, path: &Path) {
  let Some(inp) = gather_truth_inputs(ctx, scene_id) else {
    log_line(&format!("truth inputs: no jet now, {} not written", path.display()));
    return;
  };
  let mut b: Vec<u8> = Vec::new();
  b.extend_from_slice(TRUTH_MAGIC);
  b.extend_from_slice(&1u32.to_le_bytes());
  let o = |v: Option<f64>| v.unwrap_or(f64::NAN);
  let mut f: Vec<f64> = vec![view.width as f64, view.height as f64];
  f.extend_from_slice(&view.cam_pos);
  f.extend_from_slice(&view.cam_q);
  f.extend_from_slice(&[
    view.half_height_au,
    view.m_per_px_nucleus,
    view.aspect,
    o(view.perspective_tan_h),
    view.near_au,
    view.far_au,
  ]);
  f.extend_from_slice(&view.comet_au);
  f.extend_from_slice(&[
    view.nucleus_px.0,
    view.nucleus_px.1,
    o(view.mc_tier.map(|t| t as f64)),
    o(view.mc_age_max),
    o(view.mc_age_min),
    view.nucleus_px_radius,
    o(view.afrho_aperture_km),
    view.afrho_in_cm,
    view.isotropic as u8 as f64,
    view.hue_ray.from.0,
    view.hue_ray.from.1,
    view.hue_ray.dir.0,
    view.hue_ray.dir.1,
    view.hue_ray.r0,
    view.stream[0] as f64,
    view.stream[1] as f64,
    view.stream[2] as f64,
    view.ttl_s,
    inp.t_now,
  ]);
  f.extend_from_slice(&inp.r_centre_now);
  f.push(inp.albedo as f64);
  b.extend_from_slice(&(f.len() as u32).to_le_bytes());
  for v in &f {
    b.extend_from_slice(&v.to_le_bytes());
  }
  b.extend_from_slice(&(inp.batches.len() as u32).to_le_bytes());
  for t in &inp.batches {
    b.extend_from_slice(bytemuck::bytes_of(&t.desc));
    b.extend_from_slice(&t.band.0.to_le_bytes());
    b.extend_from_slice(&t.band.1.to_le_bytes());
    for v in [t.fade as u32, t.tier, t.frame.is_some() as u32] {
      b.extend_from_slice(&v.to_le_bytes());
    }
    let fr = t.frame.unwrap_or_else(bytemuck::Zeroable::zeroed);
    b.extend_from_slice(bytemuck::bytes_of(&fr));
  }
  match std::fs::write(path, &b) {
    Ok(()) => log_line(&format!(
      "truth inputs: {} batches, {:.1} MB → {}",
      inp.batches.len(),
      b.len() as f64 / 1e6,
      path.display()
    )),
    Err(e) => log_line(&format!("truth inputs: {e}")),
  }
}

fn load_truth_inputs(path: &Path) -> Option<(TruthInputs, McView)> {
  use aethervk_core_rlib::scene::dust as d;
  let b = std::fs::read(path).ok()?;
  let mut o = 0usize;
  let mut take = |n: usize| -> Option<&[u8]> {
    let s = b.get(o..o + n)?;
    o += n;
    Some(s)
  };
  if take(4)? != TRUTH_MAGIC {
    return None;
  }
  let rd = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
  let rf = |s: &[u8]| f64::from_le_bytes(s.try_into().unwrap());
  let _version = rd(take(4)?);
  let nf = rd(take(4)?) as usize;
  let f: Vec<f64> = (0..nf).map(|_| take(8).map(rf)).collect::<Option<_>>()?;
  if nf != 41 {
    return None;
  }
  let o_ = |v: f64| if v.is_nan() { None } else { Some(v) };
  let view = McView {
    width: f[0] as u32,
    height: f[1] as u32,
    cam_pos: [f[2], f[3], f[4]],
    cam_q: [f[5], f[6], f[7], f[8]],
    half_height_au: f[9],
    m_per_px_nucleus: f[10],
    aspect: f[11],
    perspective_tan_h: o_(f[12]),
    near_au: f[13],
    far_au: f[14],
    comet_au: [f[15], f[16], f[17]],
    nucleus_px: (f[18], f[19]),
    mc_tier: o_(f[20]).map(|t| t as u32),
    mc_age_max: o_(f[21]),
    mc_age_min: o_(f[22]),
    nucleus_px_radius: f[23],
    afrho_aperture_km: o_(f[24]),
    afrho_in_cm: f[25],
    isotropic: f[26] != 0.0,
    hue_ray: HueRay {
      from: (f[27], f[28]),
      dir: (f[29], f[30]),
      r0: f[31],
    },
    stream: [f[32] as f32, f[33] as f32, f[34] as f32],
    ttl_s: f[35],
  };
  let (t_now, r_centre_now, albedo) = (f[36], [f[37], f[38], f[39]], f[40] as f32);
  let nb = rd(take(4)?) as usize;
  let mut batches = Vec::with_capacity(nb);
  for _ in 0..nb {
    let desc: d::DustBatch = bytemuck::pod_read_unaligned(take(core::mem::size_of::<d::DustBatch>())?);
    let band = (rf(take(8)?), rf(take(8)?));
    let (fade, tier, has_frame) = (rd(take(4)?), rd(take(4)?), rd(take(4)?));
    let fr: d::DustFrame = bytemuck::pod_read_unaligned(take(core::mem::size_of::<d::DustFrame>())?);
    batches.push(TruthBatch {
      desc,
      band,
      frame: (has_frame != 0).then_some(fr),
      fade: fade != 0,
      tier,
    });
  }
  Some((TruthInputs { batches, t_now, r_centre_now, albedo }, view))
}

/// Clusters per tier are split into this many blocks for [`lab_packet_mc`] (fixed, not per
/// thread: the block-ordered sum is then the same at any thread count).
const LAB_MC_BLOCKS: usize = 64;

/// One block's accumulators of [`lab_packet_mc`].
struct LabMcAcc {
  img: Vec<f64>,
  rh: Vec<f64>,
  cat: Vec<(f64, f64)>,
  sup: [[f64; 5]; 2],
  sup_cat: Vec<f64>,
  /// `OBSERVER_LAB_PACKET_DRAW=1`: per packet [its MC in-view Σ (τ units), the drawing's Σ, class,
  /// eye distance (m), √(tr Σ/3) (m), kept mass, cluster, age (s)]
  pk: Vec<[f64; 8]>,
}

impl LabMcAcc {
  fn add(&mut self, o: &LabMcAcc) {
    for (a, b) in self.img.iter_mut().zip(&o.img) {
      *a += b;
    }
    for (a, b) in self.rh.iter_mut().zip(&o.rh) {
      *a += b;
    }
    for (a, b) in self.cat.iter_mut().zip(&o.cat) {
      a.0 += b.0;
      a.1 += b.1;
    }
    for (a, b) in self.sup.iter_mut().flatten().zip(o.sup.iter().flatten()) {
      *a += b;
    }
    for (a, b) in self.sup_cat.iter_mut().zip(&o.sup_cat) {
      *a += b;
    }
    self.pk.extend_from_slice(&o.pk);
  }
}

/// The packet MC's stream of one (tier, cluster, packet, piece): splitmix64 of the indices,
/// nonzero (the xorshift state of [`mc_rng`]).
fn lab_mc_seed(tier: usize, cluster: usize, packet: usize, piece: usize) -> u64 {
  let mut z = 0x1234_5678_9ABC_DEF1u64
    ^ ((tier as u64) << 56)
    ^ ((cluster as u64) << 20)
    ^ ((packet as u64) << 4)
    ^ piece as u64;
  z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  (z ^ (z >> 31)).max(1)
}

/// Grains per work unit of the Monte-Carlo truth ([`monte_carlo_truth_from`]): the unit of
/// load balance and of the ordered sum (one unit's histograms are held until summed).
const MC_UNIT_GRAINS: usize = 1 << 22;

/// The stream of a batch's `piece`-th grain range of [`MC_UNIT_GRAINS`]: piece 0 is the batch's
/// own stream (a batch within one piece draws what it always drew), the others splitmix64 of it.
fn mc_piece_seed(seed: u32, piece: usize) -> u64 {
  let base = (seed as u64 ^ 0x9E37_79B9_7F4A_7C15).max(1);
  if piece == 0 {
    return base;
  }
  let mut z = base ^ (piece as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
  z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  (z ^ (z >> 31)).max(1)
}

/// `OBSERVER_THREADS=<n>`: the lab's CPU threads (the splat, the packet MC); default the
/// machine's parallelism, clamped to 1..=32. Results are the same at any count.
/// The spread of the splat's work over the clusters (`AETHERVK_DUST_WORK_LOG=1`, replay lab):
/// per counter (texel adds, split children, volume evaluations, EWA texel units) the total, the
/// clusters doing any, the quantiles per cluster and the share of the top 0.1 % / 1 % / 10 %
/// clusters; then the 8 costliest clusters by texel adds. Nothing without a log.
fn print_work_log(log: &[aethervk_core_rlib::scene::dust::ClusterWork]) {
  if log.is_empty() {
    return;
  }
  let n = log.len();
  let counters: [(&str, fn(&aethervk_core_rlib::scene::dust::ClusterWork) -> u64); 6] = [
    ("texel adds", |c| c.adds),
    ("split children", |c| c.split_children as u64),
    ("volume evals", |c| c.volume_evals as u64),
    ("ewa units", |c| c.ewa_units as u64),
    ("packets", |c| c.packets as u64),
    ("volume packets", |c| c.volume_packets as u64),
  ];
  println!("lab: work over {n} clusters (AETHERVK_DUST_WORK_LOG)");
  // per tier (the tier's live count tells them apart): the packets the level of detail lowers
  {
    let mut tiers: Vec<u32> = log.iter().map(|c| c.live).collect();
    tiers.sort_unstable();
    tiers.dedup();
    for live in tiers {
      let t: Vec<_> = log.iter().filter(|c| c.live == live).collect();
      println!(
        "lab:   tier of {live} live: {} clusters, {} packets ({:.1} per cluster), {} volume packets, {} volume evals",
        t.len(),
        t.iter().map(|c| c.packets as u64).sum::<u64>(),
        t.iter().map(|c| c.packets as f64).sum::<f64>() / t.len().max(1) as f64,
        t.iter().map(|c| c.volume_packets as u64).sum::<u64>(),
        t.iter().map(|c| c.volume_evals as u64).sum::<u64>()
      );
    }
  }
  for (name, get) in counters {
    let mut v: Vec<u64> = log.iter().map(get).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    let total: u64 = v.iter().sum();
    if total == 0 {
      println!("lab:   {name}: 0");
      continue;
    }
    let any = v.iter().filter(|&&x| x > 0).count();
    let top = |frac: f64| {
      let k = ((n as f64 * frac).ceil() as usize).clamp(1, n);
      v[..k].iter().sum::<u64>() as f64 / total as f64
    };
    let q = |p: f64| v[((1.0 - p) * (n - 1) as f64).round() as usize];
    println!(
      "lab:   {name}: total {total} ({any} clusters > 0); per cluster p50 {} p90 {} p99 {} p99.9 {} max {}; top 0.1% {:.1}% top 1% {:.1}% top 10% {:.1}%",
      q(0.5),
      q(0.9),
      q(0.99),
      q(0.999),
      v[0],
      100.0 * top(0.001),
      100.0 * top(0.01),
      100.0 * top(0.1)
    );
  }
  let mut by_adds: Vec<&aethervk_core_rlib::scene::dust::ClusterWork> = log.iter().collect();
  by_adds.sort_unstable_by(|a, b| b.adds.cmp(&a.adds));
  for c in by_adds.iter().take(8) {
    println!(
      "lab:   costliest: index {} of {} age {:.0} s adds {} children {} evals {} ewa {}",
      c.index, c.live, c.age_s, c.adds, c.split_children, c.volume_evals, c.ewa_units
    );
  }
}

/// `OBSERVER_LAB_CADENCE=1` (the replay lab, with `AETHERVK_DUST_WORK_LOG=1`): the temporal-reuse
/// study (plan "temporal reuse by per-cluster cadence classes", step 1). Per cluster the on-screen
/// rate per frame δ = |px(p + v·Δt − Δeye) − px(p)| at the polyline's nearest point (v the
/// comet-relative velocity approximated as mean / age; the real rule uses the Kepler velocity),
/// plus the growth max(extent, time segment) · Δt / age, the spiral's turn ω · Δt · extent, and for
/// a cloud enclosing the eye |v·Δt − Δeye| / σ_min of the frame; a cluster goes to the largest
/// class N ∈ {8, 16, 64} with δ · 2N ≤ ε, else it is fresh. Printed per simulation step per frame
/// (`OBSERVER_LAB_DT=<s>` or the app's speeds at 60 fps) and tolerance (`OBSERVER_LAB_CADENCE_EPS`,
/// default 1,2,4,16 px): the packets per class (the replay's own packets per cluster) and the
/// per-frame load P_fresh + Σ P_k / N_k, i.e. what a cadenced frame would draw. The eye moves
/// `OBSERVER_LAB_DEYE` m per frame (default 2.9: an orbit at 10°/s at 1 km) across the view.
fn lab_cadence_study(tiers: &LabTiers, log: &[aethervk_core_rlib::scene::dust::ClusterWork], w: u32, h: u32) {
  use aethervk_core_rlib::scene::dust as d;
  let env_list = |k: &str, default: &[f64]| -> Vec<f64> {
    std::env::var(k)
      .ok()
      .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect::<Vec<_>>())
      .filter(|v| !v.is_empty())
      .unwrap_or_else(|| default.to_vec())
  };
  let eps_list = env_list("OBSERVER_LAB_CADENCE_EPS", &[1.0, 2.0, 4.0, 16.0]);
  let dt_list = env_list("OBSERVER_LAB_DT", &[0.0, 1.0 / 60.0, 10.0 / 60.0, 1.0, 60.0, 1440.0]);
  let deye_m = std::env::var("OBSERVER_LAB_DEYE").ok().and_then(|v| v.trim().parse::<f64>().ok()).unwrap_or(2.9);
  const CLASSES: [u32; 3] = [8, 16, 64];
  // per cluster: its packets and the ingredients of δ
  struct C {
    packets: f64,
    near: [f32; 3],
    v_rel: [f32; 3],
    ext_px: f32,
    age: f32,
    spin: f32,
    enclosing: bool,
    sigma_min: f32,
    tier: usize,
  }
  let mut cs: Vec<C> = Vec::with_capacity(log.len());
  let mut behind = 0usize;
  // the log's entries come tier by tier (the tiers are splatted one after the other); two tiers
  // may share a live count, so a tier takes the contiguous block of its own entries
  let mut cursor = 0usize;
  for (ti, (pc, _, _, render, moments)) in tiers.iter().enumerate() {
    let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
    let view = d::SplitView::of(&pc.mvp, eye);
    let px = |q: [f32; 3]| d::project_point(q, &pc.mvp, w, h);
    let start = cursor;
    while cursor < log.len() && log[cursor].live == pc.live_count && cursor - start < pc.live_count as usize {
      cursor += 1;
    }
    for c in &log[start..cursor] {
      let i = c.index as usize;
      let Some(m) = moments.get(i) else { continue };
      if !(m.flux() > 0.0) {
        continue;
      }
      let pred = d::streak_pred(render, i).and_then(|q| moments.get(q));
      let age = m.age().max(1.0);
      // the nearest point of the polyline (perspective: the smallest depth along the view)
      let mut near = m.mean();
      if view.perspective {
        let mut best = f32::INFINITY;
        for j in 0..d::SIZE_EDGES as usize {
          let e = m.edge(j);
          let dep = (e[0] - eye[0]) * view.fwd[0] + (e[1] - eye[1]) * view.fwd[1] + (e[2] - eye[2]) * view.fwd[2];
          if dep < best {
            best = dep;
            near = e;
          }
        }
      }
      // a polyline with a point behind the camera (the eye inside the old coma: a cloud
      // enclosing the eye) has no screen extent: its image changes by its relative change,
      // |shift| / σ and Δt / age of the frame (the enclosing term below), not by a parallax
      let ext_px = match (d::polyline_extent_px(m, &px), d::time_segment_px(m, pred, &px)) {
        (Some(e), Some(t)) => e.max(t),
        (Some(e), None) => e,
        _ => {
          behind += 1;
          f32::INFINITY
        }
      };
      let spin = pred.and_then(|p| d::spiral_of(m, p)).map_or(0.0, |(_, omega)| omega);
      let cv = m.cov_v();
      let sig2_max = cv[0].max(cv[3]).max(cv[5]).max(0.0);
      let sig2_min = cv[0].min(cv[3]).min(cv[5]).max(1e-6);
      let dm = [m.mean()[0] - eye[0], m.mean()[1] - eye[1], m.mean()[2] - eye[2]];
      let enclosing = view.perspective && (!ext_px.is_finite() || dm[0] * dm[0] + dm[1] * dm[1] + dm[2] * dm[2] < 9.0 * sig2_max);
      cs.push(C {
        packets: c.packets as f64,
        near,
        v_rel: [m.mean()[0] / age, m.mean()[1] / age, m.mean()[2] / age],
        ext_px,
        age,
        spin,
        enclosing,
        sigma_min: sig2_min.sqrt(),
        tier: ti,
      });
    }
  }
  let total: f64 = cs.iter().map(|c| c.packets).sum();
  println!(
    "lab cadence: {} clusters, {total:.0} packets, {behind} enclosing the eye (a point behind the camera: rated by |shift| / σ_min and Δt / age of the frame); Δeye {deye_m} m per frame across the view; classes {CLASSES:?} (staleness ≤ 2N frames)",
    cs.len()
  );
  let frame_px = w.max(h) as f32;
  for &dt in &dt_list {
    // δ per cluster for this step: the eye's lateral step is along the view's x axis
    let deltas: Vec<f32> = cs
      .iter()
      .map(|c| {
        let (pc, ..) = &tiers[c.tier];
        let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
        let view = d::SplitView::of(&pc.mvp, eye);
        let f = view.fwd;
        let up = if f[0].abs() < 0.9 { [1.0f32, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]];
        let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt().max(1e-12);
        let de = [r[0] / rn * deye_m as f32, r[1] / rn * deye_m as f32, r[2] / rn * deye_m as f32];
        let dt = dt as f32;
        let shift = [c.v_rel[0] * dt - de[0], c.v_rel[1] * dt - de[1], c.v_rel[2] * dt - de[2]];
        let px = |q: [f32; 3]| d::project_point(q, &pc.mvp, w, h);
        let moved = [c.near[0] + shift[0], c.near[1] + shift[1], c.near[2] + shift[2]];
        let sn = (shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2]).sqrt();
        if !c.ext_px.is_finite() {
          // enclosing the eye: the relative change of its τ profile over the frame
          return (sn / c.sigma_min + dt / c.age) * frame_px;
        }
        let d1 = match (px(c.near), px(moved)) {
          (Some(a), Some(b)) => ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt(),
          _ => f32::INFINITY,
        };
        let d2 = c.ext_px * dt / c.age;
        let d3 = c.spin * dt * c.ext_px;
        let d4 = if c.enclosing { sn / c.sigma_min * frame_px } else { 0.0 };
        d1 + d2 + d3 + d4
      })
      .collect();
    let mut sorted: Vec<f32> = deltas.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| sorted.get(((sorted.len().saturating_sub(1)) as f64 * p) as usize).copied().unwrap_or(f32::INFINITY);
    println!(
      "lab cadence: Δt {dt:.4} s/frame: δ px per cluster p10 {:.3} p50 {:.3} p90 {:.3} p99 {:.3} ({} infinite)",
      q(0.1),
      q(0.5),
      q(0.9),
      q(0.99),
      deltas.len() - sorted.len()
    );
    for &eps in &eps_list {
      let mut per_class = [0.0f64; 4]; // fresh, 8, 16, 64
      let mut clusters = [0usize; 4];
      for (c, &delta) in cs.iter().zip(&deltas) {
        let mut k = 0usize;
        for (ci, &n) in CLASSES.iter().enumerate() {
          if (delta as f64) * 2.0 * n as f64 <= eps {
            k = ci + 1;
          }
        }
        per_class[k] += c.packets;
        clusters[k] += 1;
      }
      let load = per_class[0] + per_class[1] / 8.0 + per_class[2] / 16.0 + per_class[3] / 64.0;
      println!(
        "lab cadence:   ε {eps:>4} px: packets fresh {:.0} / N8 {:.0} / N16 {:.0} / N64 {:.0} (clusters {} / {} / {} / {}); per-frame load {:.0} = {:.1} % of the frame",
        per_class[0],
        per_class[1],
        per_class[2],
        per_class[3],
        clusters[0],
        clusters[1],
        clusters[2],
        clusters[3],
        load,
        100.0 * load / total.max(1.0)
      );
    }
  }
}

fn observer_threads() -> usize {
  std::env::var("OBSERVER_THREADS")
    .ok()
    .and_then(|v| v.trim().parse::<usize>().ok())
    .filter(|&n| n >= 1)
    .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 32))
}

/// `OBSERVER_PROBE_THREADS=<n>`: the threads of the packet MC and the per-packet probe
/// ([`lab_packet_mc`]; default `OBSERVER_THREADS`, i.e. all cores). Every result is the same at
/// any count (fixed cluster blocks summed in block order, each cluster's rounding seeds its own).
fn probe_threads() -> usize {
  std::env::var("OBSERVER_PROBE_THREADS")
    .ok()
    .and_then(|v| v.trim().parse::<usize>().ok())
    .filter(|&n| n >= 1)
    .unwrap_or_else(observer_threads)
}

/// One probe thread's reused drawing state ([`lab_packet_mc`] with `OBSERVER_LAB_PACKET_DRAW=1`):
/// the pyramid each packet is drawn alone on (all zero between packets past the header: only the
/// texel rectangles the packet's scatters logged, `d::set_touch_log`, are summed and cleared),
/// the log, the touched texels `[word index, level, tx, ty]` and the split's scratch.
#[derive(Default)]
struct ProbeScratch {
  pyr: Vec<u32>,
  log: Vec<[u32; 6]>,
  touched: Vec<[u32; 4]>,
  split: aethervk_core_rlib::scene::dust::SplitLeaves,
}

/// `OBSERVER_LAB_PACKET_MC=<k>` (the replay lab): the render's own packets ([`d::subpackets_swept`]
/// with the splat's pooling decisions) sampled as grains, `k` per packet — uniform along the
/// capsule (its linear density ramp), the covariance's Gaussian across — and deposited with the
/// truth's weighting: flux × radiance weight / (depth² · cos θ) at the sample (τ per pixel area,
/// up to one constant). The image the packets *are*, without the splat's drawing of them.
fn lab_packet_mc(tiers: &LabTiers, header: &[u32], w: u32, h: u32, k: usize, outside: Option<&[bool]>) -> Vec<f64> {
  use aethervk_core_rlib::scene::dust as d;
  let (wu, hu) = (w as usize, h as usize);
  // per (size bin, piece) of the clusters' packet lists: the image weight in total and outside
  // the truth's region (`outside`), when every bin has the same number of packets
  // `OBSERVER_LAB_SUPPORT=1`: every in-view sample classified in its cell's own coordinates — the
  // emission origin (the size polyline, linear in the speed factor f, extrapolated to f = 0), the
  // coordinate along the motion in units of f, the angle off the cell's axis in lateral bandwidths
  // — [young (cluster < 500 s), older] × [behind the origin, slower than the slowest grain,
  // faster than the fastest, > 3 bandwidths off the axis, physical]; and the young in-view light
  // by size bin × piece
  // `OBSERVER_LAB_SUPPORT_CUT=plane|cone`: each packet's samples outside the support are
  // rejected and its flux renormalised over the rest — `plane` behind the slowest grain (along the
  // motion), `cone` that and > 3 lateral bandwidths off the cell's axis: what a truncated kernel
  // would draw; `kernel` the exact cut at the kernel's own planes (renormalised by
  // `support_mass`), `trunc` the kernel's truncated packets (`support_truncated`)
  let cut = std::env::var("OBSERVER_LAB_SUPPORT_CUT").ok();
  // the cell-coordinate histograms ([`lab_cell_mode`]) need the support's cells
  let cell_mode = lab_cell_mode();
  let support = std::env::var("OBSERVER_LAB_SUPPORT").is_ok_and(|v| v == "1") || cut.is_some() || cell_mode.is_some();
  // `OBSERVER_LAB_RHIST_INVIEW=1`: the histogram of the in-view light (as the truth's)
  let rhist_inview = std::env::var("OBSERVER_LAB_RHIST_INVIEW").is_ok_and(|v| v == "1");
  // the accumulators: a fixed number of cluster blocks per tier ([`LAB_MC_BLOCKS`], whatever the
  // thread count), each into its own, summed in block order — bit-identical at any
  // `OBSERVER_THREADS`
  let fresh = || LabMcAcc {
    img: vec![0.0f64; wu * hu],
    rh: vec![0.0f64; RHIST_BINS],
    cat: vec![(0.0f64, 0.0f64); 16 * 8],
    sup: [[0.0f64; 5]; 2],
    sup_cat: vec![0.0f64; 16 * 8],
    pk: Vec::new(),
  };
  // `OBSERVER_LAB_PACKET_DRAW=1` (with `_SUPPORT_CUT=kernel`): every packet also drawn alone, as
  // the splat draws it (`project_split_scatter` under the lab's header: its volume bias, gates,
  // the `AETHERVK_DUST_LAB_*` knobs), on a fresh pyramid, against its own grains — which packets
  // the drawing over- or under-fills. The MC side is τ per pixel: (h / 2 tan(fov/2))² per unit
  let draw_cmp = std::env::var("OBSERVER_LAB_PACKET_DRAW").is_ok_and(|v| v == "1");
  // the kernel's cone of grains (`conic_slabs`, perspective views; `AETHERVK_DUST_LAB_APEX=0`:
  // the former drawing), sampled as the splat draws it
  let apex_lab = !std::env::var("AETHERVK_DUST_LAB_APEX").ok().and_then(|v| v.trim().parse::<f32>().ok()).is_some_and(|v| v <= 0.0);
  // the kernel's sub-bins per size bin (the splat's lab knob): a bin's packets are `sub_bins` runs
  // of its time pieces
  let sub_bins = std::env::var("AETHERVK_DUST_LAB_SUBBINS")
    .ok()
    .and_then(|v| v.trim().parse::<f32>().ok())
    .map_or(1, |v| (v as usize).clamp(1, 16));
  // (unset: the end bins' `DUST_END_SUBBINS` sub-bins make the bins' runs unequal; the per-bin
  // diagnostics then see no regular layout and use the cluster's own cell)
  // `OBSERVER_LAB_PIECE_AGE=lo,hi` (s): only the packets whose piece's centre age is in [lo, hi)
  // (with `_SUPPORT_CUT` or a cell mode: the piece ages come with the support's cells)
  let piece_gate = std::env::var("OBSERVER_LAB_PIECE_AGE").ok().and_then(|v| {
    let mut it = v.split(',').map(|x| x.trim().parse::<f64>().ok());
    Some((it.next()??, it.next()??))
  });
  let layout = d::PyramidLayout::new(w, h);
  // the composite's in-frame weight of each texel, per level and axis: Σ over the frame's pixels
  // of the bilinear weight it gets (texel centres at (t + 0.5)·s, clamped at the grid's edges, as
  // `composite_sample` reads it); a texel's light on screen is its counts × wx·wy / s² (the
  // bilinear weight is separable, so this is exact at every level)
  let comp_w: Vec<(Vec<f64>, Vec<f64>)> = layout
    .levels
    .iter()
    .enumerate()
    .map(|(l, &(_, lw, lh))| {
      let s = (1u64 << l) as f64;
      let axis = |n: u32, len: u32| {
        let mut v = vec![0.0f64; n as usize];
        for x in 0..len {
          let f = (x as f64 + 0.5) / s - 0.5;
          let x0 = f.floor();
          let a = f - x0;
          let t0 = (x0 as i64).clamp(0, n as i64 - 1) as usize;
          let t1 = (x0 as i64 + 1).clamp(0, n as i64 - 1) as usize;
          v[t0] += 1.0 - a;
          v[t1] += a;
        }
        v
      };
      (axis(lw, w), axis(lh, h))
    })
    .collect();
  // `AETHERVK_DUST_LAB_TRACE_CLUSTER=<id>[,<id>…]` (with `OBSERVER_LAB_PACKET_DRAW=1`): every
  // packet of these clusters drawn alone with the splat's trace on (`set_volume_trace`: its
  // route, leaves, volume plan and pieces), then its MC and drawn light
  let trace_ids: Vec<usize> = std::env::var("AETHERVK_DUST_LAB_TRACE_CLUSTER")
    .ok()
    .map(|v| v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect())
    .unwrap_or_default();
  let mut total = fresh();
  let threads = probe_threads();
  // `OBSERVER_PROBE_SPARSE=0`: each packet on a freshly allocated pyramid, its whole pyramid
  // summed (the former probe; the reference of the sparse one). `OBSERVER_PROBE_VERIFY=1`: the
  // sparse probe also sums the whole pyramid, compares the two sums' bits and checks the pyramid
  // is all zero past the header after the clear (a texel written outside the logged rectangles)
  let sparse = !std::env::var("OBSERVER_PROBE_SPARSE").is_ok_and(|v| v.trim() == "0");
  let verify = std::env::var("OBSERVER_PROBE_VERIFY").is_ok_and(|v| v.trim() == "1");
  let verify_bad = std::sync::atomic::AtomicUsize::new(0);
  // read once (it was read per sample: the environment's lock on every thread)
  let rhist_mode = std::env::var("OBSERVER_LAB_RHIST_MODE");
  let lvl0 = layout.levels[0].0 as usize;
  let cell_of = |q: &d::DustMoments| -> (V3, V3) {
    let (e0, e1) = (q.edge(0), q.edge(d::SIZE_BINS as usize));
    let (f0, f1) = (q.edge_factor(0) as f64, q.edge_factor(d::SIZE_BINS as usize) as f64);
    let c = [
      (e1[0] - e0[0]) as f64 / (f1 - f0),
      (e1[1] - e0[1]) as f64 / (f1 - f0),
      (e1[2] - e0[2]) as f64 / (f1 - f0),
    ];
    (
      [e0[0] as f64 - f0 * c[0], e0[1] as f64 - f0 * c[1], e0[2] as f64 - f0 * c[2]],
      c,
    )
  };
  for (ti, (pc, _, _, render, moments)) in tiers.iter().enumerate() {
    let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
    let view = d::SplitView::of(&pc.mvp, eye);
    let n = (pc.live_count as usize).min(moments.len()).min(render.len());
    // one cluster's packets into `a`; each packet draws from its own stream (tier, cluster,
    // packet, piece), so no draw depends on another cluster's
    let cluster = |i: usize, a: &mut LabMcAcc, ws: &mut ProbeScratch| {
      // the frame's kernel (`DUST_FRAME_LEGACY_KERNEL_BIT`), on this thread
      d::set_legacy_kernel(d::legacy_kernel(header));
      // the cluster's rounding seeds start from its ring slot, as the splat's
      // (`d::begin_cluster_packets`): its packets' drawn light is the same on any thread and in
      // any order (the thread's sequence ran on across clusters before)
      d::begin_cluster_packets(d::render_slot(render[i].age_id_dbeta_flux[1]));
      d::set_lod_px_active(d::lod_px(header));
      d::set_lod_view_active(view.perspective.then_some((view.eye, view.fwd)));
      let LabMcAcc { img, rh, cat, sup, sup_cat, pk } = a;
      let m = &moments[i];
      if !(m.flux() > 0.0) {
        return;
      }
      let pred = d::streak_pred(render, i).map(|q| &moments[q]);
      let seg = match pred {
        Some(p) => {
          let (a, b) = (p.mean(), m.mean());
          [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        }
        None => [0.0; 3],
      };
      let pred_merged = pred.is_none_or(|p| d::polyline_merged(p, &pc.mvp, w, h));
      let merged = d::polyline_merged(m, &pc.mvp, w, h)
        && pred_merged
        && pred.is_none_or(|p| d::time_shallow(m, p, &pc.mvp));
      let px = |q: [f32; 3]| d::project_point(q, &pc.mvp, w, h);
      let sun = m.sun_dir();
      // `kernel` / `trunc`: the kernel's own support planes (`subpackets_supported`)
      let kernel_cut = matches!(cut.as_deref(), Some("kernel") | Some("trunc"));
      let packets: Vec<(d::Packet, [f32; 4], f32)> = if kernel_cut {
        d::subpackets_cells(m, seg, pred, merged, pred_merged, Some(&px))
      } else {
        d::subpackets_swept(m, seg, pred, merged, pred_merged, Some(&px)).into_iter().map(|p| (p, [0.0; 4], f32::NAN)).collect()
      };
      let per_bin = if !merged && packets.len() % 16 == 0 { packets.len() / 16 } else { 0 };
      // the support classification's cell: own (origin, c per unit f), the lateral bandwidth
      // σ_ang and the relative speed spread from its Σ_v, the speed range
      let young = m.age() < 500.0;
      let (o_m, c_m) = cell_of(m);
      let cn_m = norm(c_m);
      let (sig_ang, v_std) = {
        let u = normalize(c_m);
        let cv = m.cov_v().map(|v| v as f64);
        let q = cv[0] * u[0] * u[0] + cv[3] * u[1] * u[1] + cv[5] * u[2] * u[2]
          + 2.0 * (cv[1] * u[0] * u[1] + cv[2] * u[0] * u[2] + cv[4] * u[1] * u[2]);
        let tr = cv[0] + cv[3] + cv[5];
        (((tr - q) * 0.5).max(0.0).sqrt() / cn_m.max(1e-30), q.max(0.0).sqrt() / cn_m.max(1e-30))
      };
      let (f_lo, f_hi) = (m.edge_factor(0) as f64, m.edge_factor(d::SIZE_BINS as usize) as f64);
      // `OBSERVER_LAB_CELL_PRINT=1`: the young cells' speed range, relative speed spread and
      // lateral bandwidth (from Σ_v), and their first bins' packets along the axis (f units)
      if young && std::env::var("OBSERVER_LAB_CELL_PRINT").is_ok_and(|v| v == "1") && i % 16 == 0 {
        let pk0: Vec<String> = packets
          .iter()
          .take(3)
          .map(|(p, _, _)| {
            let cn = norm(c_m).max(1e-30);
            let u = [c_m[0] / cn, c_m[1] / cn, c_m[2] / cn];
            let a0 = dot(sub([p.mean[0] as f64, p.mean[1] as f64, p.mean[2] as f64], o_m), u) / cn;
            let a1 = a0 + dot([p.seg[0] as f64, p.seg[1] as f64, p.seg[2] as f64], u) / cn;
            let cv = p.cov.map(|v| v as f64);
            let q = cv[0] * u[0] * u[0] + cv[3] * u[1] * u[1] + cv[5] * u[2] * u[2]
              + 2.0 * (cv[1] * u[0] * u[1] + cv[2] * u[0] * u[2] + cv[4] * u[1] * u[2]);
            format!("[{a0:.3}..{a1:.3} σ {:.3}]", q.max(0.0).sqrt() / cn)
          })
          .collect();
        println!(
          "lab: cell {i} age {:.0} s: f {f_lo:.4}..{f_hi:.4}, σ_v/v {v_std:.4}, σ_ang {sig_ang:.4}, edges f {:?}, packets {}",
          m.age(),
          (0..=d::SIZE_BINS as usize).step_by(4).map(|b| m.edge_factor(b)).collect::<Vec<_>>(),
          pk0.join(" ")
        );
      }
      let spiral = pred.and_then(|q| d::spiral_of(m, q));
      let ratio = d::width_ratio(m, pred);
      for (pi, (p_whole, ksup, w_o)) in packets.into_iter().enumerate() {
        let ci = if per_bin > 0 && per_bin <= 8 { Some((pi / per_bin) * 8 + pi % per_bin) } else { None };
        // the piece's cell: interpolated (spiral-turned, scaled with age) to its centre age
        let (o_u, c_u, age_u) = match (pred, per_bin) {
          (Some(q), nb) if support && nb / sub_bins > 1 && !merged => {
            // a bin's packets are its sub-bins' (`AETHERVK_DUST_LAB_SUBBINS`) runs of pieces
            let nb = nb / sub_bins;
            let t = ((pi % per_bin) % nb) as u32;
            let uc = 0.5 * (d::piece_bound(t, nb as u32, ratio) + d::piece_bound(t + 1, nb as u32, ratio));
            let (age_o, age_p) = (m.age() as f64, q.age() as f64);
            let age_u = age_o + uc as f64 * (age_p - age_o);
            let (o_p, c_p) = cell_of(q);
            match spiral {
              Some((ax, om)) => {
                let f3 = |v: V3| [v[0] as f32, v[1] as f32, v[2] as f32];
                let o = d::spiral_point(f3(o_m), f3(o_p), age_o as f32, age_p as f32, ax, om, uc);
                let rq = mc_qaxis_angle([ax[0] as f64, ax[1] as f64, ax[2] as f64], om as f64 * (age_p - age_u));
                let c = mc_qrot(rq, c_p);
                ([o[0] as f64, o[1] as f64, o[2] as f64], [c[0] * age_u / age_p, c[1] * age_u / age_p, c[2] * age_u / age_p], age_u)
              }
              None => {
                let l = uc as f64;
                (
                  [o_m[0] + l * (o_p[0] - o_m[0]), o_m[1] + l * (o_p[1] - o_m[1]), o_m[2] + l * (o_p[2] - o_m[2])],
                  [c_m[0] + l * (c_p[0] - c_m[0]), c_m[1] + l * (c_p[1] - c_m[1]), c_m[2] + l * (c_p[2] - c_m[2])],
                  age_u,
                )
              }
            }
          }
          _ => (o_m, c_m, m.age() as f64),
        };
        if let Some((lo, hi)) = piece_gate
          && !(age_u >= lo && age_u < hi)
        {
          continue;
        }
        // `trunc`: the kernel's truncation's moments (`support_truncated`), drawn uncut
        let mut subs = [p_whole; d::DUST_SUPPORT_TRUNC_MAX];
        let cut_here = d::has_support(&ksup) && d::support_mass(&p_whole, &ksup) < 1.0 - d::DUST_SUPPORT_CUT_MIN;
        // `AETHERVK_DUST_LAB_APEX=1` (the splat's lab knob): the cone of grains (`conic_slabs`), the
        // cut baked in
        let apex_n = if apex_lab && view.perspective { d::conic_slabs(&p_whole, &ksup, w_o, &mut subs) } else { 0 };
        let ns = if apex_n > 0 {
          apex_n
        } else if cut.as_deref() == Some("trunc") && cut_here {
          d::support_truncated(&p_whole, &ksup, &mut subs)
        } else {
          1
        };
        let cut_here = cut_here && apex_n == 0;
        let mut pk_mc = 0.0f64;
        // the in-view samples' nearest distance and depth, their weighted mean depth, the count
        let mut pk_near = [f64::INFINITY, f64::INFINITY, 0.0, 0.0];
        for (si, p) in subs.into_iter().take(ns).enumerate() {
        let mut rng = lab_mc_seed(ti, i, pi, si);
        let c = p.cov.map(|v| v as f64);
        let l11 = c[0].max(0.0).sqrt();
        let l21 = if l11 > 0.0 { c[1] / l11 } else { 0.0 };
        let l31 = if l11 > 0.0 { c[2] / l11 } else { 0.0 };
        let l22 = (c[3] - l21 * l21).max(0.0).sqrt();
        let l32 = if l22 > 0.0 { (c[4] - l31 * l21) / l22 } else { 0.0 };
        let l33 = (c[5] - l31 * l31 - l32 * l32).max(0.0).sqrt();
        let (r0, r1) = (p.rho[0] as f64, p.rho[1] as f64);
        // the sample's support class (see `support` above)
        let classify = |q: V3| -> usize {
          let cn = norm(c_u).max(1e-30);
          let dv = sub(q, o_u);
          let along_m = dot(dv, c_u) / cn;
          let along = along_m / cn;
          let lat = (dot(dv, dv) - along_m * along_m).max(0.0).sqrt();
          let off = lat.atan2(along_m) / sig_ang.max(1e-9);
          if along < 0.0 {
            0
          } else if along < f_lo * (1.0 - 3.0 * v_std) {
            1
          } else if along > f_hi * (1.0 + 3.0 * v_std) {
            2
          } else if off > 3.0 {
            3
          } else {
            4
          }
        };
        let rejected = |class: usize| match cut.as_deref() {
          Some("plane") | Some("apex") => class <= 1,
          Some("cone") => class <= 1 || class == 3,
          _ => false,
        };
        // one draw of the packet: a uniform position along the capsule (its density ramp) and the
        // covariance's Gaussian across (its whitened deviation kept for the `tail` histogram)
        let last_chi = core::cell::Cell::new(0.0f64);
        // bounded as the kernels draw it (`mc_kick_max`)
        let kick_max = mc_kick_max();
        let draw = |rng: &mut u64| -> V3 {
          // linear density r0 → r1 along the capsule (mean 1): inverse CDF of (r0 + (r1 − r0) t)/1
          let x = mc_rng(rng);
          let t = if (r1 - r0).abs() < 1e-9 {
            x
          } else {
            let a = 0.5 * (r1 - r0);
            ((-r0 + (r0 * r0 + 4.0 * a * x).max(0.0).sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
          };
          let [g1, g2, g3] = mc_bound_kick([mc_gauss(rng), mc_gauss(rng), mc_gauss(rng)], kick_max, rng);
          last_chi.set((g1 * g1 + g2 * g2 + g3 * g3).sqrt());
          [
            p.mean[0] as f64 + t * p.seg[0] as f64 + l11 * g1,
            p.mean[1] as f64 + t * p.seg[1] as f64 + l21 * g1 + l22 * g2,
            p.mean[2] as f64 + t * p.seg[2] as f64 + l31 * g1 + l32 * g2 + l33 * g3,
          ]
        };
        // with a cut, the kept share first (the same draws again below)
        // `kernel`: the exact cut at the kernel's plane, renormalised by its kept mass as the
        // kernel's ray integral is (`support_mass`, floor DUST_SUPPORT_MASS_MIN)
        let behind = |q: V3| -> bool {
          cut.as_deref() == Some("kernel")
            && cut_here
            && (ksup[0] as f64 * q[0] + ksup[1] as f64 * q[1] + ksup[2] as f64 * q[2]) < ksup[3] as f64
        };
        let wf = if kernel_cut {
          let mass = if cut.as_deref() == Some("kernel") && cut_here { d::support_mass(&p, &ksup).max(d::DUST_SUPPORT_MASS_MIN) } else { 1.0 };
          p.flux as f64 / (k as f64 * mass as f64)
        } else if cut.is_some() {
          let mut r2 = rng;
          let kept = (0..k).filter(|_| !rejected(classify(draw(&mut r2)))).count();
          if kept == 0 {
            continue;
          }
          p.flux as f64 / kept as f64
        } else {
          p.flux as f64 / k as f64
        };
        // `apex`: the lateral offset from the cell's axis scaled by the sample's distance along it
        // over the packet centre's (grains fan out from the origin: their lateral spread grows
        // with distance; a packet carries one width along its length)
        let a_c = {
          let pc3 = [
            p.mean[0] as f64 + 0.5 * p.seg[0] as f64,
            p.mean[1] as f64 + 0.5 * p.seg[1] as f64,
            p.mean[2] as f64 + 0.5 * p.seg[2] as f64,
          ];
          dot(sub(pc3, o_u), c_u) / norm(c_u).max(1e-30)
        };
        let apex = |q: V3| -> V3 {
          if cut.as_deref() != Some("apex") || !(a_c > 0.0) {
            return q;
          }
          let cn = normalize(c_u);
          let dv = sub(q, o_u);
          let al = dot(dv, cn);
          let lat = sub(dv, [cn[0] * al, cn[1] * al, cn[2] * al]);
          let k = (al / a_c).max(0.0);
          [o_u[0] + cn[0] * al + lat[0] * k, o_u[1] + cn[1] * al + lat[1] * k, o_u[2] + cn[2] * al + lat[2] * k]
        };
        for _ in 0..k {
          let q = apex(draw(&mut rng));
          let class = if support { classify(q) } else { 4 };
          if (!kernel_cut && rejected(class)) || behind(q) {
            continue;
          }
          let kb = match rhist_mode.as_deref() {
            _ if cell_mode.is_some() => {
              let cm = cell_mode.unwrap();
              let cn = norm(c_u).max(1e-30);
              let dv = sub(q, o_u);
              let along_m = dot(dv, c_u) / cn;
              let x = match cm.0 {
                "tail" => last_chi.get(),
                "fbin" => {
                  if per_bin > 0 {
                    let b = pi / per_bin;
                    (0.5 * (m.edge_factor(b) + m.edge_factor(b + 1)) as f64).log10()
                  } else {
                    f64::NAN
                  }
                }
                "view" => {
                  let de = [q[0] - eye[0] as f64, q[1] - eye[1] as f64, q[2] - eye[2] as f64];
                  let f = [view.fwd[0] as f64, view.fwd[1] as f64, view.fwd[2] as f64];
                  (dot(de, f) / norm(de).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees()
                }
                "speed" => along_m / cn,
                "off" => (along_m / norm(dv).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees(),
                _ => age_u,
              };
              lab_cell_bin(x, cm)
            }
            Ok("eye") => {
              let e = [q[0] - eye[0] as f64, q[1] - eye[1] as f64, q[2] - eye[2] as f64];
              (((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt() / RHIST_BIN_M) as usize).min(RHIST_BINS - 1)
            }
            Ok(md @ ("theta" | "phi")) => {
              let ax = m.spin_axis();
              lab_angle_bin(md == "theta", q, [ax[0] as f64, ax[1] as f64, ax[2] as f64], [sun[0] as f64, sun[1] as f64, sun[2] as f64])
            }
            _ => (((q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() / RHIST_BIN_M) as usize).min(RHIST_BINS - 1),
          };
          let view_gated = lab_view_max().is_some_and(|g| {
            let de = [q[0] - eye[0] as f64, q[1] - eye[1] as f64, q[2] - eye[2] as f64];
            let f = [view.fwd[0] as f64, view.fwd[1] as f64, view.fwd[2] as f64];
            (dot(de, f) / norm(de).max(1e-30)).clamp(-1.0, 1.0).acos().to_degrees() > g
          });
          if !rhist_inview && !view_gated {
            rh[kb] += wf;
          }
          let de = [q[0] - eye[0] as f64, q[1] - eye[1] as f64, q[2] - eye[2] as f64];
          let depth = de[0] * view.fwd[0] as f64 + de[1] * view.fwd[1] as f64 + de[2] * view.fwd[2] as f64;
          if !(depth > 0.0) {
            continue;
          }
          let qf = [q[0] as f32, q[1] as f32, q[2] as f32];
          let Some(s) = d::project_point(qf, &pc.mvp, w, h) else {
            continue;
          };
          if !(s[0] >= 0.0 && s[1] >= 0.0 && (s[0] as usize) < wu && (s[1] as usize) < hu) {
            continue;
          }
          let dist = (de[0] * de[0] + de[1] * de[1] + de[2] * de[2]).sqrt();
          let to_eye = [-de[0] as f32, -de[1] as f32, -de[2] as f32];
          let wr = if pc.albedo > 0.0 {
            d::radiance_weight(pc.albedo, sun, to_eye, m.helio_au()) as f64
          } else {
            1.0
          };
          let wv = wf * wr * dist / (depth * depth * depth);
          if rhist_inview && !view_gated {
            rh[kb] += wv;
          }
          let ix = s[1] as usize * wu + s[0] as usize;
          img[ix] += wv;
          pk_mc += wv;
          pk_near = [pk_near[0].min(dist), pk_near[1].min(depth), pk_near[2] + wv * depth, pk_near[3] + 1.0];
          if support {
            sup[if young { 0 } else { 1 }][class] += wv;
            if young && let Some(c) = ci {
              sup_cat[c] += wv;
            }
          }
          if let (Some(c), Some(o)) = (ci, outside) {
            cat[c].0 += wv;
            if o[ix] {
              cat[c].1 += wv;
            }
          }
        }
        }
        if draw_cmp {
          // the packet alone on a fresh pyramid carrying the lab's header (its counters zeroed):
          // the thread's reused one (zero past the header, see below), or a new one
          // (`OBSERVER_PROBE_SPARSE=0`)
          let mut fresh_pyr;
          let pyr: &mut Vec<u32> = if sparse {
            if ws.pyr.len() != layout.total_words as usize {
              ws.pyr = vec![0u32; layout.total_words as usize];
            }
            ws.pyr[..lvl0].fill(0);
            &mut ws.pyr
          } else {
            fresh_pyr = vec![0u32; layout.total_words as usize];
            &mut fresh_pyr
          };
          pyr[..header.len()].copy_from_slice(header);
          for w in [d::PYR_VOLUME_EVALS, d::PYR_EWA_TEXELS, d::PYR_SPLIT_CHILDREN, d::PYR_SPLIT_DROPPED_FLUX, d::PYR_TAU_MAX, d::PYR_LEVEL_MASK] {
            pyr[w as usize] = 0;
          }
          let mut view_d = view.clone();
          view_d.shell = d::specks::ShellCut::from_header(&pyr, eye).d_max;
          let phot = d::Photometry { albedo: pc.albedo, sun, r_h_au: m.helio_au(), support: ksup, origin: w_o };
          // (`mut`: the probe block below borrows it as `&mut scratch`)
          let mut scratch = &mut ws.split;
          let n_hard = d::hard_axis_pieces(pyr);
          let traced = trace_ids.contains(&i);
          if traced {
            println!("lab: trace cluster {i} packet {pi} (tier {ti}, class {}):", if cut_here { "cut" } else { "uncut" });
            d::set_volume_trace(true);
          }
          // as the splat draws it (its cone of grains included, `Photometry::origin`)
          if sparse {
            d::set_touch_log(true);
          }
          for j in 0..n_hard.max(1) {
            let q = d::hard_axis_leaf(&p_whole, n_hard, j);
            d::project_split_scatter(&q, 1.0, &pc.mvp, &view_d, &phot, [1.0; 3], &layout, pc.inv_unit, pyr, scratch);
          }
          d::set_volume_trace(false);
          PROBE_SPLIT_ADDS.fetch_add(d::take_volume_split_adds(), std::sync::atomic::Ordering::Relaxed);
          // the light the composite shows in the frame, exactly: each texel's counts × its
          // composite weight wx·wy / s² (`comp_w`); the raw counts summed before counted a texel
          // overhanging the frame whole (the 512-px 1 × 1 level of 480 × 300: 1.82×) and the
          // off-frame part of a texel whose centre lies outside it
          let full_sum = |pyr: &[u32]| -> f64 {
            let mut counts = 0.0f64;
            for (l, &(off, lw, lh)) in layout.levels.iter().enumerate() {
              let s2 = ((1u64 << l) as f64).powi(2);
              let (wx, wy) = &comp_w[l];
              for ty in 0..lh {
                for tx in 0..lw {
                  let c = pyr[(off + (ty * lw + tx) * d::PYRAMID_TEXEL_WORDS) as usize];
                  if c != 0 {
                    counts += c as f64 * wx[tx as usize] * wy[ty as usize] / s2;
                  }
                }
              }
            }
            counts
          };
          // sparse: the same sum over the touched texels only, in the same (word index) order —
          // the full loop's terms are its nonzero texels in increasing index, so the f64 sum is
          // the same bit for bit — then those rectangles cleared (every word of their texels)
          let counts = if sparse {
            d::set_touch_log(false);
            d::take_touch_log(&mut ws.log);
            ws.touched.clear();
            let tw = d::PYRAMID_TEXEL_WORDS;
            for &[off, lw, x0, x1, y0, y1] in ws.log.iter() {
              let l = layout.levels.iter().position(|lv| lv.0 == off).expect("touch log: a level offset");
              let lh = layout.levels[l].2;
              for ty in y0..=y1.min(lh - 1) {
                for tx in x0..=x1.min(lw - 1) {
                  let ix = off + (ty * lw + tx) * tw;
                  if pyr[ix as usize] != 0 {
                    ws.touched.push([ix, l as u32, tx, ty]);
                  }
                }
              }
            }
            ws.touched.sort_unstable_by_key(|t| t[0]);
            ws.touched.dedup_by_key(|t| t[0]);
            let mut counts = 0.0f64;
            for &[ix, l, tx, ty] in ws.touched.iter() {
              let s2 = ((1u64 << l) as f64).powi(2);
              let (wx, wy) = &comp_w[l as usize];
              let c = pyr[ix as usize];
              counts += c as f64 * wx[tx as usize] * wy[ty as usize] / s2;
            }
            if verify {
              let f = full_sum(pyr);
              if f.to_bits() != counts.to_bits() {
                verify_bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                println!("lab: probe verify: cluster {i} packet {pi}: sparse Σ {counts:e} ≠ full Σ {f:e}");
              }
            }
            for &[off, lw, x0, x1, y0, y1] in ws.log.iter() {
              let lh = layout.levels.iter().find(|lv| lv.0 == off).map_or(0, |lv| lv.2);
              for ty in y0..=y1.min(lh - 1) {
                let a = (off + (ty * lw + x0) * tw) as usize;
                let b = (off + (ty * lw + x1.min(lw - 1)) * tw + tw) as usize;
                if a < b {
                  pyr[a..b].fill(0);
                }
              }
            }
            if verify && pyr[lvl0..].iter().any(|&v| v != 0) {
              verify_bad.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
              println!("lab: probe verify: cluster {i} packet {pi}: a texel written outside its logged rectangles");
              pyr[lvl0..].fill(0);
            }
            counts
          } else {
            full_sum(pyr)
          };
          let drawn = counts / pc.inv_unit as f64;
          if traced {
            println!("lab: trace cluster {i} packet {pi}: drawn {drawn:.4e} (the composite's in-frame light)");
          }
          // the MC's τ per pixel: the focal length in px squared per unit of its weight, measured
          // on the projection 1 km ahead (1 m across)
          let px_k = {
            let f = [view.fwd[0] as f64, view.fwd[1] as f64, view.fwd[2] as f64];
            let a = if f[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
            let up = normalize(cross(f, a));
            let p0 = [eye[0] as f64 + 1e3 * f[0], eye[1] as f64 + 1e3 * f[1], eye[2] as f64 + 1e3 * f[2]];
            let p1 = [p0[0] + up[0], p0[1] + up[1], p0[2] + up[2]];
            let pr = |q: V3| d::project_point([q[0] as f32, q[1] as f32, q[2] as f32], &pc.mvp, w, h);
            match (pr(p0), pr(p1)) {
              (Some(a), Some(b)) => (((b[0] - a[0]) as f64).powi(2) + ((b[1] - a[1]) as f64).powi(2)) * 1e6,
              _ => f64::NAN,
            }
          };
          let cls = match (cut_here, d::is_volumetric(&p_whole, &view_d)) {
            (false, true) => 0.0,
            (false, false) => 1.0,
            (true, true) => 2.0,
            (true, false) => 3.0,
          };
          let de = sub([p_whole.mean[0] as f64 + 0.5 * p_whole.seg[0] as f64, p_whole.mean[1] as f64 + 0.5 * p_whole.seg[1] as f64, p_whole.mean[2] as f64 + 0.5 * p_whole.seg[2] as f64], [eye[0] as f64, eye[1] as f64, eye[2] as f64]);
          let cv = p_whole.cov;
          // `OBSERVER_LAB_PACKET_PROBE=<cluster>`: its packets the drawing under-fills, in full
          if std::env::var("OBSERVER_LAB_PACKET_PROBE").ok().and_then(|v| v.trim().parse::<usize>().ok()) == Some(i)
            && drawn < 0.5 * pk_mc * px_k
          {
            scratch.clear();
            let mut st = d::SplitStats::default();
            d::split_packet(&p_whole, &view_d, &mut scratch, &mut st);
            println!(
              "lab: probe cluster {i} packet {pi}: MC {:.3e} drawn {drawn:.3e}; mean {:?} seg {:?} cov {:?} open {:?} rho {:?} flux {:.3e}; eye {:?} fwd {:?}; volumetric {}; in-view samples {} nearest {:.0} m depth ≥ {:.0} m, weighted mean depth {:.0} m; split leaves {} volume {} EWA",
              pk_mc * px_k,
              p_whole.mean,
              p_whole.seg,
              p_whole.cov,
              p_whole.open,
              p_whole.rho,
              p_whole.flux,
              eye,
              view_d.fwd,
              d::is_volumetric(&p_whole, &view_d),
              pk_near[3],
              pk_near[0],
              pk_near[1],
              pk_near[2] / pk_mc.max(1e-300),
              scratch.volume.len(),
              scratch.ewa.len()
            );
          }
          pk.push([
            pk_mc * px_k,
            drawn,
            cls + if view_d.shell > 0.0 { 10.0 } else { 0.0 },
            norm(de),
            ((cv[0] + cv[3] + cv[5]) as f64 / 3.0).max(0.0).sqrt(),
            if cut_here { d::support_mass(&p_whole, &ksup) as f64 } else { 1.0 },
            i as f64,
            m.age() as f64,
          ]);
        }
      }
    };
    let bsz = n.div_ceil(LAB_MC_BLOCKS).max(1);
    let nb = n.div_ceil(bsz);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut parts: Vec<(usize, LabMcAcc)> = std::thread::scope(|s| {
      let handles: Vec<_> = (0..threads.min(nb).max(1))
        .map(|_| {
          let (cluster, next, fresh) = (&cluster, &next, &fresh);
          s.spawn(move || {
            let mut out = Vec::new();
            let mut ws = ProbeScratch::default();
            loop {
              let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
              if b >= nb {
                break;
              }
              let mut a = fresh();
              for i in b * bsz..((b + 1) * bsz).min(n) {
                cluster(i, &mut a, &mut ws);
              }
              out.push((b, a));
            }
            out
          })
        })
        .collect();
      handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    parts.sort_by_key(|p| p.0);
    for (_, a) in &parts {
      total.add(a);
    }
  }
  let LabMcAcc { img, rh, cat, sup, sup_cat, pk } = total;
  if draw_cmp {
    println!(
      "lab: probe drawing on {threads} threads, {}{}",
      if sparse { "sparse (the touched texels summed and cleared)" } else { "a fresh pyramid per packet" },
      if verify { format!(", verified: {} mismatches", verify_bad.load(std::sync::atomic::Ordering::Relaxed)) } else { String::new() }
    );
  }
  if draw_cmp && !pk.is_empty() {
    let names = ["uncut, eye near (volume)", "uncut, split", "cut, eye near (volume, ray_kept)", "cut, truncated"];
    let (tm, td): (f64, f64) = pk.iter().fold((0.0, 0.0), |a, r| (a.0 + r[0], a.1 + r[1]));
    println!(
      "lab: packets drawn alone: {} packets, MC Σ {tm:.4e}, drawn Σ {td:.4e} (drawn/MC {:.3}), speck shell {}, volume deposits split above the per-add cap {}",
      pk.len(),
      td / tm.max(1e-300),
      if pk.iter().any(|r| r[2] >= 10.0) { "on (the support cut is skipped)" } else { "off" },
      PROBE_SPLIT_ADDS.swap(0, std::sync::atomic::Ordering::Relaxed)
    );
    for (c, name) in names.iter().enumerate() {
      let rows: Vec<&[f64; 8]> = pk.iter().filter(|r| (r[2] as usize) % 10 == c).collect();
      let (m, dr): (f64, f64) = rows.iter().fold((0.0, 0.0), |a, r| (a.0 + r[0], a.1 + r[1]));
      println!(
        "lab:   {name:34}: {:6} packets, MC {:.4e} ({:.3} of it), drawn {:.4e}, drawn/MC {:.3}",
        rows.len(),
        m,
        m / tm.max(1e-300),
        dr,
        dr / m.max(1e-300)
      );
    }
    println!("lab:   the largest differences per class (drawn − MC, share of the MC total): eye distance, σ, kept mass, cluster, age");
    let by: Vec<&[f64; 8]> = (0..4)
      .flat_map(|c| {
        let mut v: Vec<&[f64; 8]> = pk.iter().filter(|r| (r[2] as usize) % 10 == c).collect();
        v.sort_by(|a, b| (b[1] - b[0]).abs().total_cmp(&(a[1] - a[0]).abs()));
        v.truncate(6);
        v
      })
      .collect();
    for r in by.iter() {
      println!(
        "lab:     {:+.4}  MC {:.3e} drawn {:.3e}  class {} dist {:.0} m σ {:.0} m mass {:.3} cluster {} age {:.0} s",
        (r[1] - r[0]) / tm.max(1e-300),
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        r[5],
        r[6],
        r[7]
      );
    }
  }
  if support {
    let all: f64 = sup.iter().flatten().sum();
    for (k, name) in ["young (< 500 s)", "older"].iter().enumerate() {
      let t: f64 = sup[k].iter().sum();
      println!(
        "lab: support {name}: {:.4} of the in-view light; of it behind the origin {:.4}, slower than the slowest {:.4}, faster than the fastest {:.4}, > 3 bandwidths off axis {:.4}, physical {:.4}",
        t / all.max(1e-300),
        sup[k][0] / t.max(1e-300),
        sup[k][1] / t.max(1e-300),
        sup[k][2] / t.max(1e-300),
        sup[k][3] / t.max(1e-300),
        sup[k][4] / t.max(1e-300)
      );
    }
    let yt: f64 = sup_cat.iter().sum();
    println!("lab: support: the young in-view light by size bin (rows, 0 = slow) × piece (cols, 0 = young end), share");
    for b in 0..16 {
      let mut line = format!("lab:   bin {b:2}:");
      for t in 0..8 {
        line += &format!(" {:.3}", sup_cat[b * 8 + t] / yt.max(1e-300));
      }
      println!("{line}");
    }
  }
  if outside.is_some() {
    let tot: f64 = cat.iter().map(|c| c.0).sum();
    let out: f64 = cat.iter().map(|c| c.1).sum();
    println!("lab: packets outside the truth's 5 % region: {:.4} of their light; by size bin (rows, 0 = slow/large) and piece (cols, 0 = young end): share of all outside light | outside fraction of the cell", out / tot.max(1e-300));
    for b in 0..16 {
      let mut line = format!("lab:   bin {b:2}:");
      for t in 0..8 {
        let c = cat[b * 8 + t];
        if c.0 > 0.0 {
          line += &format!("  {:.3}|{:.3}", c.1 / out.max(1e-300), c.1 / c.0);
        }
      }
      println!("{line}");
    }
  }
  // the 3-D check: the packets' and the truth's cross-section by distance from the centre
  let th = RHIST.lock().unwrap().clone();
  if th.len() == RHIST_BINS {
    let (sp, st): (f64, f64) = (rh.iter().sum(), th.iter().sum());
    let (mut cp, mut ct) = (0.0, 0.0);
    println!("lab: histogram ({}; distances in km, angles in bins of 2° θ / 4° φ): packets' share / truth's share, cumulative packets / truth; totals packets {sp:.6e} truth {st:.6e}", std::env::var("OBSERVER_LAB_RHIST_MODE").unwrap_or_else(|_| "centre".into()));
    for k in 0..RHIST_BINS {
      let (a, b) = (rh[k] / sp.max(1e-300), th[k] / st.max(1e-300));
      cp += a;
      ct += b;
      if a > 1e-4 || b > 1e-4 {
        println!(
          "lab:   {:5.2}-{:5.2}  {:.4} / {:.4}  ratio {:6.2}   cum {:.3} / {:.3}",
          k as f64 * cell_mode.map_or(RHIST_BIN_M / 1e3, |c| c.1) - cell_mode.map_or(0.0, |c| lab_cell_first(c.0)),
          (k + 1) as f64 * cell_mode.map_or(RHIST_BIN_M / 1e3, |c| c.1) - cell_mode.map_or(0.0, |c| lab_cell_first(c.0)),
          a,
          b,
          a / b.max(1e-300),
          cp,
          ct
        );
      }
    }
  }
  img
}

/// `--lab DIR`: replay `snapshot_NNN.bin` on the CPU mirror into a fresh pyramid carrying its
/// header (the counters zeroed; `OBSERVER_LAB_HARD=<n>` / `OBSERVER_LAB_TIME_CELL=1` override the
/// diagnostic header bits, the engine's own `AETHERVK_DUST_*` CPU switches apply as in the
/// engine), build the τ image as the observer does, and score it against `mc_NNN_truth.f32` with
/// the observer's formulas: Σ ratio, rel rms over the 10⁻⁴ region of the 5×5-boxed unit-sum
/// images, the judged rms (the unresolved-core rule, the 1/√n noise from `mc_NNN_counts.f32`
/// taken out in quadrature), the ratio by truth level and the light outside the truth's 5 %
/// region; plus the replay's Σ against the GPU frame (`frame_NNN.pyr`) as a fidelity check.
fn run_lab(dir: &Path, frame: u32, lab_truth: usize) -> i32 {
  use aethervk_core_rlib::scene::dust as d;
  let t0 = Instant::now();
  let Some((w, h, mut header, mut tiers)) = load_snapshot(&dir.join(format!("snapshot_{frame:03}.bin"))) else {
    eprintln!("lab: no snapshot_{frame:03}.bin in {}", dir.display());
    return 2;
  };
  // `OBSERVER_LAB_AGE=lo,hi` (s): only the dust of that age on both sides — the render's clusters
  // by their age (flux 0 outside), the drawn truth by the grains' (`--lab-truth` only)
  let age_gate: Option<(f64, f64)> = std::env::var("OBSERVER_LAB_AGE").ok().and_then(|v| {
    let (a, b) = v.split_once(',')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
  });
  let drawn_inputs = if lab_truth > 0 {
    let Some(x) = load_truth_inputs(&dir.join(format!("truth_{frame:03}.bin"))) else {
      eprintln!("lab: --lab-truth needs truth_{frame:03}.bin (an observer run with --save-snapshot)");
      return 2;
    };
    Some(x)
  } else {
    if age_gate.is_some() {
      eprintln!("lab: OBSERVER_LAB_AGE needs --lab-truth (the stored truth holds every age)");
      return 2;
    }
    None
  };
  // `OBSERVER_LAB_DUMP=<path>`: write the replayed pyramid's words there and stop after the splat
  // (no truth needed: render against render, e.g. `pyrcmp.py` of two level-of-detail settings)
  let dump = std::env::var("OBSERVER_LAB_DUMP").ok().map(PathBuf::from);
  let (stored_truth, stored_counts) = if drawn_inputs.is_none() && dump.is_none() {
    let Some(t) = read_f32_file(&dir.join(format!("mc_{frame:03}_truth.f32"))) else {
      eprintln!("lab: no mc_{frame:03}_truth.f32 (run with OBSERVER_MC_RAW=<out>/mc_{frame:03})");
      return 2;
    };
    (t, read_f32_file(&dir.join(format!("mc_{frame:03}_counts.f32"))).unwrap_or_default())
  } else {
    (Vec::new(), Vec::new())
  };
  let env_u32 = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<u32>().ok());
  let sd = d::PYR_SPLIT_DEPTH as usize;
  if let Some(n) = env_u32("OBSERVER_LAB_HARD") {
    header[sd] = (header[sd] & !(0xF << 24)) | ((n.min(15)) << 24);
  }
  if let Some(n) = env_u32("OBSERVER_LAB_VBIAS") {
    header[sd] = (header[sd] & !(0xFF << 8)) | ((n.min(255)) << 8);
  }
  if let Some(n) = env_u32("OBSERVER_LAB_GATES_OFF") {
    header[sd] = (header[sd] & !(1 << d::DUST_FRAME_GATES_OFF_BIT)) | ((n & 1) << d::DUST_FRAME_GATES_OFF_BIT);
  }
  if let Some(n) = env_u32("OBSERVER_LAB_DEPTH") {
    header[sd] = (header[sd] & !0xFF) | n.min(d::DUST_SPLIT_DEPTH_MAX);
  }
  // `OBSERVER_LAB_LOD=<px>`: the cluster level of detail's pixels per drawn segment (`d::PYR_LOD`)
  if let Some(v) = std::env::var("OBSERVER_LAB_LOD").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
    header[d::PYR_LOD as usize] = v.max(0.0).to_bits();
  }
  // `OBSERVER_LAB_DEEP_CHORD=0|1`: the deep-chord bit (`d::DUST_FRAME_DEEP_CHORD` of `PYR_FLAGS`)
  if let Some(n) = env_u32("OBSERVER_LAB_DEEP_CHORD") {
    let f = d::PYR_FLAGS as usize;
    header[f] = (header[f] & !d::DUST_FRAME_DEEP_CHORD) | if n & 1 != 0 { d::DUST_FRAME_DEEP_CHORD } else { 0 };
    println!("lab: deep chord {}", d::deep_chord(&header));
  }
  println!(
    "lab: header split depth {} volume bias {} EWA bias {} lod {} px",
    header[sd] & 0xFF,
    (header[sd] >> 8) & 0xFF,
    (header[sd] >> 18) & 0xF,
    d::lod_px(&header)
  );
  if let Some(n) = env_u32("OBSERVER_LAB_TIME_CELL") {
    header[sd] = (header[sd] & !(1 << 28)) | ((n & 1) << 28);
  }
  // `OBSERVER_LAB_SRAD=<s>` / `OBSERVER_LAB_SLAT=<s>`: every cluster's Σ_v scaled by `s` along
  // its size polyline's direction (the motion: radial) and across it (lateral), `S Σ S` with
  // `S = s_rad uuᵀ + s_lat (I − uuᵀ)` — which part of the kernel's width is off
  let lab_f = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<f32>().ok());
  if let (sr, sl) = (lab_f("OBSERVER_LAB_SRAD"), lab_f("OBSERVER_LAB_SLAT"))
    && (sr.is_some() || sl.is_some())
  {
    let (sr, sl) = (sr.unwrap_or(1.0), sl.unwrap_or(1.0));
    for t in tiers.iter_mut() {
      for m in t.4.iter_mut() {
        let (e0, e1) = (m.edge(0), m.edge(d::SIZE_BINS as usize));
        let mut u = [e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]];
        let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        if !(n > 0.0) {
          continue;
        }
        u = [u[0] / n, u[1] / n, u[2] / n];
        let c = m.cov_v();
        let cm = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
        let mut sm = [[0.0f32; 3]; 3];
        for i in 0..3 {
          for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            sm[i][j] = sr * u[i] * u[j] + sl * (id - u[i] * u[j]);
          }
        }
        let mut o = [[0.0f32; 3]; 3];
        for i in 0..3 {
          for j in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..3 {
              for l in 0..3 {
                acc += sm[i][k] * cm[k][l] * sm[l][j];
              }
            }
            o[i][j] = acc;
          }
        }
        m.cov_a = [o[0][0], o[0][1], o[0][2], o[1][1]];
        m.cov_b_age_id[0] = o[1][2];
        m.cov_b_age_id[1] = o[2][2];
      }
    }
  }
  // `OBSERVER_LAB_AGES=1`: the render's clusters by age (log bins): count, Σ flux, per tier
  if std::env::var("OBSERVER_LAB_AGES").is_ok_and(|v| v == "1") {
    let edges = [0.0, 60.0, 240.0, 468.0, 900.0, 1800.0, 3600.0, 7200.0, 21600.0, 86400.0, 4.0 * 86400.0, 1e12];
    for (ti, t) in tiers.iter().enumerate() {
      let mut rows = String::new();
      for k in 0..edges.len() - 1 {
        let (mut n, mut fl) = (0usize, 0.0f64);
        for m in t.4.iter().filter(|m| m.flux() > 0.0) {
          let a = m.age() as f64;
          if a >= edges[k] && a < edges[k + 1] {
            n += 1;
            fl += m.flux() as f64;
          }
        }
        if n > 0 {
          rows.push_str(&format!(" [{:.0},{:.0}) n {n} flux {fl:.3e};", edges[k], edges[k + 1]));
        }
      }
      println!("lab: tier {ti} clusters by age:{rows}");
      let mut young: Vec<f32> = t.4.iter().filter(|m| m.flux() > 0.0 && m.age() < 4000.0).map(|m| m.age()).collect();
      young.sort_by(|a, b| a.partial_cmp(b).unwrap());
      young.dedup();
      if !young.is_empty() {
        println!("lab: tier {ti} distinct cluster ages < 4000 s: {:?}", &young[..young.len().min(6)]);
      }
      // the eye against the youngest clusters (local frame): along each one's polyline u from
      // its slow end (edge 0), and across
      let eye = [t.0.eye_local[0], t.0.eye_local[1], t.0.eye_local[2]];
      let mut ys: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::new();
      for m in t.4.iter().filter(|m| m.flux() > 0.0 && m.age() < 500.0) {
        let (e0, e1) = (m.edge(0), m.edge(d::SIZE_BINS as usize));
        let ch = [e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]];
        let len = (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2]).sqrt();
        let u = [ch[0] / len, ch[1] / len, ch[2] / len];
        let de = [eye[0] - e0[0], eye[1] - e0[1], eye[2] - e0[2]];
        let al = de[0] * u[0] + de[1] * u[1] + de[2] * u[2];
        let lat = ((de[0] * de[0] + de[1] * de[1] + de[2] * de[2]) - al * al).max(0.0).sqrt();
        let e0n = (e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]).sqrt();
        let c = m.cov_v();
        let q = c[0] * u[0] * u[0] + c[3] * u[1] * u[1] + c[5] * u[2] * u[2]
          + 2.0 * (c[1] * u[0] * u[1] + c[2] * u[0] * u[2] + c[4] * u[1] * u[2]);
        ys.push((al, lat, len, e0n, q.max(0.0).sqrt() * m.edge_factor(0), m.flux()));
      }
      if !ys.is_empty() {
        let en = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
        ys.sort_by(|a, b| b.5.partial_cmp(&a.5).unwrap());
        println!("lab: tier {ti} eye {eye:?} |eye| {en:.0} m; youngest clusters (by flux): eye along u from edge0, across, chord, |edge0|, f0·σ_rad (m)");
        for y in ys.iter().take(8) {
          println!("lab:   along {:.0} across {:.0} chord {:.0} |e0| {:.0} f0σ {:.0} flux {:.3e}", y.0, y.1, y.2, y.3, y.4, y.5);
        }
      }
      // the moments of a few young clusters: along the polyline u (radial) and across it
      let mut shown = 0;
      for m in t.4.iter().filter(|m| m.flux() > 0.0 && (m.age() < 500.0 || (m.age() > 2290.0 && m.age() < 2400.0))) {
        if shown >= 6 {
          break;
        }
        shown += 1;
        let (e0, e1) = (m.edge(0), m.edge(d::SIZE_BINS as usize));
        let ch = [e1[0] - e0[0], e1[1] - e0[1], e1[2] - e0[2]];
        let len = (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2]).sqrt();
        let u = [ch[0] / len, ch[1] / len, ch[2] / len];
        let c = m.cov_v();
        let q = |w: [f32; 3]| {
          c[0] * w[0] * w[0] + c[3] * w[1] * w[1] + c[5] * w[2] * w[2]
            + 2.0 * (c[1] * w[0] * w[1] + c[2] * w[0] * w[2] + c[4] * w[1] * w[2])
        };
        let mn = m.mean();
        let r = (mn[0] * mn[0] + mn[1] * mn[1] + mn[2] * mn[2]).sqrt();
        let tr = c[0] + c[3] + c[5];
        let srad = q(u).max(0.0).sqrt();
        let slat = ((tr - q(u)) * 0.5).max(0.0).sqrt();
        // the reference grain's travel: the mean minus the site's position is not known here;
        // the mid edge (f = 1) distance from the centre is
        let mid = m.edge(8);
        let rmid = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
        println!(
          "lab:   age {:.0} s |mean| {r:.0} m |edge8| {rmid:.0} m chord {len:.0} m f0 {:.3} f16 {:.3} σ_rad {srad:.1} m σ_lat {slat:.1} m σ_rad/age {:.4} m/s σ_lat/age {:.4} m/s",
          m.age(),
          m.edge_factor(0),
          m.edge_factor(d::SIZE_BINS as usize),
          srad / m.age(),
          slat / m.age()
        );
      }
    }
  }
  if let Some((lo, hi)) = age_gate {
    for t in tiers.iter_mut() {
      for m in t.4.iter_mut() {
        let a = m.age() as f64;
        if !(a >= lo && a < hi) {
          m.mean_flux[3] = 0.0;
        }
      }
    }
  }
  let layout = d::PyramidLayout::new(w, h);
  let mut pyr = vec![0u32; layout.total_words as usize];
  let hn = d::PYRAMID_HEADER_WORDS as usize;
  pyr[..hn].copy_from_slice(&header);
  {
    use aethervk_core_rlib::scene::dust::specks as sp;
    for wd in [
      d::PYR_TAU_MAX,
      d::PYR_LEVEL_MASK,
      d::PYR_SPLIT_CHILDREN,
      d::PYR_VOLUME_EVALS,
      d::PYR_EWA_TEXELS,
      d::PYR_SPLIT_DROPPED_FLUX,
      sp::PYR_SPECKS,
      sp::PYR_SPECKS_OVERFLOW,
      sp::PYR_SPECKS_TRUNCATED,
      sp::PYR_N_EYE,
      sp::PYR_N_EYE_HI,
      sp::PYR_SPECKS_NODES,
      sp::PYR_SPECKS_COUNTS,
    ] {
      pyr[wd as usize] = 0;
    }
  }
  // `OBSERVER_THREADS=<n>`: the replay's splat threads (default: the machine's parallelism; 1 =
  // the serial splat). The result is bit-identical for any count: compare the hash line.
  let threads = observer_threads();
  // `OBSERVER_LAB_SPECKS=<d_max m>`: the CPU speck pass on the replayed tiers before the splat
  // (as the frame runs it since 2026-10-16: its walk mask decides the splat's hand-over), with
  // the shell radius given, the size law 50 µm / q 3.5, no fade; prints the walk's cost: the
  // bins walked (the gate), the nodes, the nodes per walk and per speck, the truncated walks
  if let Some(d_max) = std::env::var("OBSERVER_LAB_SPECKS").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
    use aethervk_core_rlib::scene::dust::specks as sp;
    sp::SpeckHeader::write(&mut pyr, d_max, 1 << 24);
    let t_sp = Instant::now();
    let (mut walked, mut clusters_walked, mut stats_all) = (0u64, 0u64, sp::SpeckPassStats::default());
    for (pc, flow, _tracer, render, moments) in tiers.iter_mut() {
      let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
      // the eye-relative mvp of the speck pass (`M (x − eye) + (M eye + t)`)
      let mut mvp_rel = pc.mvp;
      for r in 0..4 {
        mvp_rel[12 + r] = (pc.mvp[12 + r] as f64
          + pc.mvp[r] as f64 * eye[0] as f64
          + pc.mvp[4 + r] as f64 * eye[1] as f64
          + pc.mvp[8 + r] as f64 * eye[2] as f64) as f32;
      }
      let spc = sp::DustSpecksPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: pc.live_count,
        size_fade: sp::pack_size_fade(50.0, 0.0),
        exposure: pc.exposure,
        inv_unit: pc.inv_unit,
        color: sp::pack_color_q(pc.color, 3.5),
        albedo: pc.albedo,
        mvp: mvp_rel,
        eye_local: pc.eye_local,
      };
      let st = sp::specks_tier(moments, render, &spc, flow, &layout, &mut pyr);
      let n = (pc.live_count as usize).min(moments.len());
      for m in &moments[..n] {
        let bits = m.specks_mask().count_ones() as u64;
        walked += bits;
        clusters_walked += (bits > 0) as u64;
      }
      stats_all.enumerated += st.enumerated;
      stats_all.drawn += st.drawn;
      stats_all.overflow += st.overflow;
      stats_all.nodes += st.nodes;
      stats_all.truncated += st.truncated;
      for k in 0..10 {
        stats_all.start_levels[k] += st.start_levels[k];
      }
      for k in 0..4 {
        stats_all.walks_by_found[k] += st.walks_by_found[k];
        stats_all.nodes_by_found[k] += st.nodes_by_found[k];
      }
    }
    println!(
      "lab specks:   start levels (bands of 6) {:?}; walks by specks found (0 / 1–3 / 4–15 / ≥16) {:?}, their nodes {:?}",
      stats_all.start_levels,
      stats_all.walks_by_found,
      stats_all.nodes_by_found
    );
    println!(
      "lab specks: shell {d_max} m: {walked} bins walked in {clusters_walked} clusters, {} enumerated, {} drawn, {} nodes ({:.0} per walk, {:.0} per speck), {} walks truncated, {:.1} s",
      stats_all.enumerated,
      stats_all.drawn,
      stats_all.nodes,
      stats_all.nodes as f64 / walked.max(1) as f64,
      stats_all.nodes as f64 / stats_all.drawn.max(1) as f64,
      stats_all.truncated,
      t_sp.elapsed().as_secs_f64()
    );
  }
  let t_splat = Instant::now();
  for (pc, flow, tracer, render, moments) in &tiers {
    d::splat_tier_parallel(moments, render, pc, flow, *tracer, &layout, &mut pyr, threads);
  }
  let clusters: usize = tiers.iter().map(|t| (t.0.live_count as usize).min(t.3.len()).min(t.4.len())).sum();
  // FNV-1a over the replayed pyramid's words: equal hashes = the same splat, any thread count
  let hash = pyr.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &w| (h ^ w as u64).wrapping_mul(0x100_0000_01b3));
  println!(
    "lab: splat {clusters} clusters on {threads} threads, {:.1} s; replay pyramid hash {hash:016x}",
    t_splat.elapsed().as_secs_f64()
  );
  // `AETHERVK_DUST_WORK_LOG=1`: the splat's work per cluster (`d::take_work_log`): how it spreads
  // over the clusters (does a long tail of a few clusters carry the frame's GPU time?)
  let work_log = d::take_work_log();
  print_work_log(&work_log);
  if let Some(path) = dump {
    std::fs::write(&path, bytemuck::cast_slice::<u32, u8>(&pyr)).expect("lab: write OBSERVER_LAB_DUMP");
    println!("lab: replay pyramid written to {}", path.display());
    return 0;
  }
  // `OBSERVER_LAB_CADENCE=1`: the temporal-reuse study on this frame (the per-cluster on-screen
  // rate and the cadence classes' packet load per simulation speed and tolerance)
  if std::env::var("OBSERVER_LAB_CADENCE").is_ok_and(|v| v.trim() == "1") {
    lab_cadence_study(&tiers, &work_log, w, h);
  }
  let Some(render) = tau_image_of_pyramid(&pyr, w, h) else {
    eprintln!("lab: the replayed pyramid has no τ image");
    return 2;
  };
  let (wu, hu) = (w as usize, h as usize);
  let (truth, counts) = match drawn_inputs {
    Some((inputs, mut view)) => {
      // `OBSERVER_LAB_AGE_T=lo,hi`: the truth's own gate (default `OBSERVER_LAB_AGE`): a cluster
      // draws the pieces to its older predecessor, so its light holds grains older than it
      let truth_gate = std::env::var("OBSERVER_LAB_AGE_T")
        .ok()
        .and_then(|v| {
          let (a, b) = v.split_once(',')?;
          Some((a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?))
        })
        .or(age_gate);
      if let Some((lo, hi)) = truth_gate {
        view.mc_age_min = Some(view.mc_age_min.map_or(lo, |a| a.max(lo)));
        view.mc_age_max = Some(view.mc_age_max.map_or(hi, |a| a.min(hi)));
      }
      let t1 = Instant::now();
      let Some((rec, _, _, t, c)) = monte_carlo_truth_from(&pyr, &inputs, &view, lab_truth) else {
        eprintln!("lab: the truth draw returned nothing (empty render or no batch in the gate)");
        return 2;
      };
      println!(
        "lab: drew the truth: {} grains, {} batches, observer's judged {:.4} (Σ render/truth {:.4}), {:.1} s",
        rec.grains,
        rec.batches,
        rec.rel_rms_judged,
        rec.abs_ratio,
        t1.elapsed().as_secs_f64()
      );
      (t, c)
    }
    None => (stored_truth, stored_counts),
  };
  if truth.len() != wu * hu {
    eprintln!("lab: truth has {} px, the frame {}", truth.len(), wu * hu);
    return 2;
  }
  // fidelity: the replay against the GPU frame (only meaningful without overrides)
  if let Ok(gb) = std::fs::read(dir.join(format!("frame_{frame:03}.pyr"))) {
    let gw: Vec<u32> = gb.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if let Some(g) = tau_image_of_pyramid(&gw, w, h) {
      let (sa, sg): (f64, f64) = (
        render.iter().map(|&v| v as f64).sum(),
        g.iter().map(|&v| v as f64).sum(),
      );
      println!("lab: replay Στ / GPU frame Στ {:.4}", sa / sg.max(1e-300));
    }
  }
  println!(
    "lab: absolute Στ render {:.4e} truth {:.4e}",
    render.iter().map(|&v| v as f64).sum::<f64>(),
    truth.iter().sum::<f64>()
  );
  // the observer's comparison
  let mut render64: Vec<f64> = render.iter().map(|&x| x as f64).collect();
  if let Some(k) = std::env::var("OBSERVER_LAB_PACKET_MC").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
    let t1 = Instant::now();
    let outside: Vec<bool> = {
      let st: f64 = truth.iter().sum();
      let tn: Vec<f64> = truth.iter().map(|x| x / st.max(1e-300)).collect();
      let bt = mc_box(&tn, wu, hu, 2);
      let tp = bt.iter().cloned().fold(0.0f64, f64::max);
      bt.iter().map(|&v| v < 0.05 * tp).collect()
    };
    let pm = lab_packet_mc(&tiers, &header, w, h, k, Some(&outside));
    // the drawing against the packets: unit-sum, 5×5 boxed, rel rms over the packets' 10⁻⁴ region
    let norm1 = |v: &[f64]| -> Vec<f64> {
      let s: f64 = v.iter().sum();
      v.iter().map(|x| x / s.max(1e-300)).collect()
    };
    let (a, b) = (mc_box(&norm1(&render64), wu, hu, 2), mc_box(&norm1(&pm), wu, hu, 2));
    let pk = b.iter().cloned().fold(0.0f64, f64::max);
    let (mut se, mut sb) = (0.0, 0.0);
    for i in 0..wu * hu {
      if b[i] >= MC_REGION_REL * pk {
        se += (a[i] - b[i]).powi(2);
        sb += b[i] * b[i];
      }
    }
    println!(
      "lab: packet MC ({k}/packet, {:.1} s): the splat's image against its packets' rel rms {:.4}; scoring the packets' image against the truth below",
      t1.elapsed().as_secs_f64(),
      (se / sb.max(1e-300)).sqrt()
    );
    // the absolute scale: the packets' image carries the render's units up to a constant; keep
    // the render's Σ so the Σ ratio stays the render's
    let (sr, sp): (f64, f64) = (render64.iter().sum(), pm.iter().sum());
    let hash = pm.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| (h ^ x.to_bits()).wrapping_mul(0x100_0000_01b3));
    println!("lab: packet MC raw Σ {sp:.6e} (the splat's Σ {sr:.6e} in its own units), image hash {hash:016x}");
    render64 = pm.iter().map(|x| x * sr / sp.max(1e-300)).collect();
  }
  let abs_ratio = render64.iter().sum::<f64>() / truth.iter().sum::<f64>().max(1e-300);
  let norm = |v: &[f64]| -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s.max(1e-300)).collect()
  };
  let a = mc_box(&norm(&render64), wu, hu, 2);
  let b = mc_box(&norm(&truth), wu, hu, 2);
  // `OBSERVER_LAB_DUMP=<prefix>`: the two boxed unit-sum images (f32 LE): `_render.f32`, `_truth.f32`
  if let Ok(prefix) = std::env::var("OBSERVER_LAB_DUMP") {
    let f = |v: &[f64]| -> Vec<u8> { v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect() };
    let _ = std::fs::write(format!("{prefix}_render.f32"), f(&a));
    let _ = std::fs::write(format!("{prefix}_truth.f32"), f(&b));
  }
  let peak = b.iter().cloned().fold(0.0f64, f64::max);
  let region: Vec<usize> = (0..wu * hu).filter(|&i| b[i] >= MC_REGION_REL * peak).collect();
  let core_px = region.iter().filter(|&&i| b[i] >= 0.01 * peak).count();
  let core_unresolved = core_px < MC_CORE_RESOLVED_PX;
  let n5 = if counts.len() == wu * hu { mc_box(&counts, wu, hu, 2) } else { vec![1e9; wu * hu] };
  let (mut se, mut sb2, mut se_j, mut sb2_j, mut sn_j) = (0.0, 0.0, 0.0, 0.0, 0.0);
  for &i in &region {
    let d2 = (a[i] - b[i]).powi(2);
    se += d2;
    sb2 += b[i].powi(2);
    if !(core_unresolved && b[i] >= 0.01 * peak) {
      se_j += d2;
      sb2_j += b[i].powi(2);
      sn_j += b[i].powi(2) / (n5[i] * 25.0).max(1.0);
    }
  }
  let noise = (sn_j / sb2_j.max(1e-300)).sqrt();
  let rel_j = (se_j / sb2_j.max(1e-300)).sqrt();
  let judged = (rel_j * rel_j - noise * noise).max(0.0).sqrt();
  // `OBSERVER_MC_W2_NOISE=1` with `--lab-truth`: the same judged rms with the weighted noise
  // (per pixel, boxed: Σw² / (Σw)², the grains' τ weights), and over the whole region (core kept)
  if mc_w2_on() {
    let w2 = MC_W2.lock().unwrap().clone();
    if w2.len() == wu * hu && truth.len() == wu * hu {
      let (mw2, mt) = (mc_box(&w2, wu, hu, 2), mc_box(&truth, wu, hu, 2));
      let rel2 = |i: usize| if mt[i] > 0.0 { mw2[i] / (25.0 * mt[i] * mt[i]) } else { 1.0 };
      let (mut sn_w, mut sn_all, mut se_all, mut sb_all, mut sn_n_all) = (0.0, 0.0, 0.0, 0.0, 0.0);
      for &i in &region {
        let b2 = b[i].powi(2);
        se_all += (a[i] - b[i]).powi(2);
        sb_all += b2;
        sn_all += b2 * rel2(i);
        sn_n_all += b2 / (n5[i] * 25.0).max(1.0);
        if !(core_unresolved && b[i] >= 0.01 * peak) {
          sn_w += b2 * rel2(i);
        }
      }
      let noise_w = (sn_w / sb2_j.max(1e-300)).sqrt();
      let (rel_all, noise_all, noise_n_all) = (
        (se_all / sb_all.max(1e-300)).sqrt(),
        (sn_all / sb_all.max(1e-300)).sqrt(),
        (sn_n_all / sb_all.max(1e-300)).sqrt(),
      );
      // the effective grain count Kish (Σw)² / Σw² over the frame, against the grains drawn
      let (st, sw2): (f64, f64) = (truth.iter().sum(), w2.iter().sum());
      println!(
        "lab: w2 noise: judged {:.4} (raw {rel_j:.4}, weighted noise {noise_w:.4} vs count noise {noise:.4}); whole region (core kept) rel {rel_all:.4}, weighted noise {noise_all:.4} (count {noise_n_all:.4}), after weighted noise {:.4}; Kish effective grains {:.0}",
        (rel_j * rel_j - noise_w * noise_w).max(0.0).sqrt(),
        (rel_all * rel_all - noise_all * noise_all).max(0.0).sqrt(),
        st * st / sw2.max(1e-300)
      );
    } else {
      println!("lab: w2 noise: no per-pixel Σw² (needs --lab-truth N)");
    }
  }
  // the ratio by truth level and the light outside the truth's 5 % region (boxed, absolute)
  let tm = b.iter().cloned().fold(0.0f64, f64::max);
  let mut lv = [(0.0f64, 0.0f64); 4];
  let (mut in_r, mut in_t) = (0.0f64, 0.0f64);
  for i in 0..wu * hu {
    let (ra, tb) = (a[i] * abs_ratio, b[i]);
    let f = tb / tm.max(1e-300);
    let k = if f >= 0.6 { 3 } else if f >= 0.3 { 2 } else if f >= 0.1 { 1 } else if f >= 0.05 { 0 } else { 9 };
    if k < 4 {
      lv[k].0 += ra;
      lv[k].1 += tb;
      in_r += ra;
      in_t += tb;
    }
  }
  let tot_r: f64 = a.iter().sum::<f64>() * abs_ratio;
  let tot_t: f64 = b.iter().sum();
  println!(
    "lab: rel rms {:.4} (judged {:.4} after noise {:.4}, core {core_px} px) over {} px, Σ render/truth {abs_ratio:.4}",
    (se / sb2.max(1e-300)).sqrt(),
    judged,
    noise,
    region.len()
  );
  println!(
    "lab: ratio by truth level 0.05-0.1 {:.3}, 0.1-0.3 {:.3}, 0.3-0.6 {:.3}, 0.6-1 {:.3} | outside the truth's 5 % region: render {:.4}, truth {:.4} | {:.1} s",
    lv[0].0 / lv[0].1.max(1e-300),
    lv[1].0 / lv[1].1.max(1e-300),
    lv[2].0 / lv[2].1.max(1e-300),
    lv[3].0 / lv[3].1.max(1e-300),
    (tot_r - in_r) / tot_r.max(1e-300),
    (tot_t - in_t) / tot_t.max(1e-300),
    t0.elapsed().as_secs_f64()
  );
  // the same comparison by `lab_score` (the camera path's scorer): equal numbers prove it is this
  // arithmetic; and its 60-px block rms
  {
    let s = lab_score(&render64, &truth, &counts, wu, hu);
    println!(
      "lab: score check: lab_score judged {:.4} (above {judged:.4}), Σ render/truth {:.4} (above {abs_ratio:.4}); block{LAB_BLOCK_PX} rms {:.4} (shape only {:.4}, the truth's count noise {:.4}) over {} blocks",
      s.judged, s.sum_ratio, s.block, s.block_shape, s.block_noise, s.blocks
    );
  }
  0
}

// ───────────────────── temporal reuse step 4: the camera path and the image score ─────────────────────

/// The light of a pyramid in counts: Σ over the composited levels (`PYR_LEVEL_MASK`) of every
/// texel's count; × `PYR_UNIT` it is Σ τ·px² as `composite_sample` reads it (a level-ℓ texel's τ
/// per px is `count · unit / 4^ℓ` over its 4^ℓ px), whole texels (a coarse texel's overhang beyond
/// the frame included). Not `cadence::pyramid_light` (Σ count · 4^ℓ), which weights level ℓ by an
/// extra 4^ℓ: a level-11 1×1 texel holding 3 % of a 1 km frame's light is 99.9 % of it.
fn frame_light(layout: &aethervk_core_rlib::scene::dust::PyramidLayout, pyr: &[u32]) -> f64 {
  use aethervk_core_rlib::scene::dust as d;
  let mask = pyr[d::PYR_LEVEL_MASK as usize];
  let mut t = 0.0f64;
  for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
    if mask & (1 << l) == 0 {
      continue;
    }
    for i in 0..(w * h) {
      t += pyr[(off + i * d::PYRAMID_TEXEL_WORDS) as usize] as f64;
    }
  }
  t
}

/// the side of the score's blocks (px): the 60-px block rms of [`lab_score`]
const LAB_BLOCK_PX: usize = 60;
/// a block enters the block rms when its truth (or reference) light is at least this share of the
/// brightest block's
const LAB_BLOCK_REL: f64 = 0.05;

/// One frame scored against a truth ([`lab_score`]): the observer's judgement (`run_lab`'s
/// arithmetic), the Σ ratio, the ratio by truth level, the light outside the truth's region and
/// the 60-px block rms (absolute, shape only, and the truth's count noise at that scale).
#[derive(Debug, Clone, Copy, Default)]
struct LabScore {
  sum_ratio: f64,
  rel_rms: f64,
  judged: f64,
  noise: f64,
  core_px: usize,
  region_px: usize,
  levels: [f64; 4],
  out_render: f64,
  out_truth: f64,
  block: f64,
  block_shape: f64,
  block_noise: f64,
  blocks: usize,
}

/// `img` summed over `px × px` blocks (the last row and column partial): `(sums, columns, rows)`
fn block_sums(img: &[f64], w: usize, h: usize, px: usize) -> (Vec<f64>, usize, usize) {
  let (bw, bh) = (w.div_ceil(px), h.div_ceil(px));
  let mut s = vec![0.0f64; bw * bh];
  for y in 0..h {
    for x in 0..w {
      s[(y / px) * bw + x / px] += img[y * w + x];
    }
  }
  (s, bw, bh)
}

/// The block rms of `render` against `truth` (absolute images, the same units): over the blocks
/// whose truth is ≥ [`LAB_BLOCK_REL`] of the brightest, `sqrt(Σ (R_B − T_B)² / Σ T_B²)` (the
/// absolute one: the Σ error counts), the same with `R` scaled to the truth's Σ (the shape only),
/// and the truth's count noise at the block scale `sqrt(Σ_B Σ_px t² / n_px / Σ T_B²)` (0 without
/// counts; like the pixel score's noise term it ignores the grains' weights). Returns
/// `(absolute, shape, noise, blocks)`.
fn block_rms(render: &[f64], truth: &[f64], counts: &[f64], w: usize, h: usize, px: usize) -> (f64, f64, f64, usize) {
  let (rb, bw, bh) = block_sums(render, w, h, px);
  let (tb, _, _) = block_sums(truth, w, h, px);
  let mut nv = vec![0.0f64; bw * bh];
  if counts.len() == w * h {
    for y in 0..h {
      for x in 0..w {
        let i = y * w + x;
        nv[(y / px) * bw + x / px] += truth[i] * truth[i] / counts[i].max(1.0);
      }
    }
  }
  let tmax = tb.iter().cloned().fold(0.0f64, f64::max);
  let scale = truth.iter().sum::<f64>() / render.iter().sum::<f64>().max(1e-300);
  let (mut se, mut ss, mut st, mut sn, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
  for b in 0..bw * bh {
    if tmax > 0.0 && tb[b] >= LAB_BLOCK_REL * tmax {
      se += (rb[b] - tb[b]).powi(2);
      ss += (rb[b] * scale - tb[b]).powi(2);
      st += tb[b] * tb[b];
      sn += nv[b];
      n += 1;
    }
  }
  let st = st.max(1e-300);
  ((se / st).sqrt(), (ss / st).sqrt(), (sn / st).sqrt(), n)
}

/// A frame's τ image `render` against the truth `truth` (+ grains per pixel `counts`): the
/// arithmetic of `run_lab`'s comparison (5×5-boxed unit-sum images, the region at
/// `MC_REGION_REL` of the truth's peak, the unresolved-core rule, the count noise subtracted:
/// the judged value; the ratio by truth level and the light outside the region), plus the 60-px
/// block rms ([`block_rms`]). `run_lab` prints both and they agree (its `score check` line).
fn lab_score(render: &[f64], truth: &[f64], counts: &[f64], w: usize, h: usize) -> LabScore {
  let abs_ratio = render.iter().sum::<f64>() / truth.iter().sum::<f64>().max(1e-300);
  let norm = |v: &[f64]| -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|x| x / s.max(1e-300)).collect()
  };
  let a = mc_box(&norm(render), w, h, 2);
  let b = mc_box(&norm(truth), w, h, 2);
  let peak = b.iter().cloned().fold(0.0f64, f64::max);
  let region: Vec<usize> = (0..w * h).filter(|&i| b[i] >= MC_REGION_REL * peak).collect();
  let core_px = region.iter().filter(|&&i| b[i] >= 0.01 * peak).count();
  let core_unresolved = core_px < MC_CORE_RESOLVED_PX;
  let n5 = if counts.len() == w * h { mc_box(counts, w, h, 2) } else { vec![1e9; w * h] };
  let (mut se, mut sb2, mut se_j, mut sb2_j, mut sn_j) = (0.0, 0.0, 0.0, 0.0, 0.0);
  for &i in &region {
    let d2 = (a[i] - b[i]).powi(2);
    se += d2;
    sb2 += b[i].powi(2);
    if !(core_unresolved && b[i] >= 0.01 * peak) {
      se_j += d2;
      sb2_j += b[i].powi(2);
      sn_j += b[i].powi(2) / (n5[i] * 25.0).max(1.0);
    }
  }
  let noise = (sn_j / sb2_j.max(1e-300)).sqrt();
  let rel_j = (se_j / sb2_j.max(1e-300)).sqrt();
  let judged = (rel_j * rel_j - noise * noise).max(0.0).sqrt();
  let mut lv = [(0.0f64, 0.0f64); 4];
  let (mut in_r, mut in_t) = (0.0f64, 0.0f64);
  for i in 0..w * h {
    let (ra, tb) = (a[i] * abs_ratio, b[i]);
    let f = tb / peak.max(1e-300);
    let k = if f >= 0.6 { 3 } else if f >= 0.3 { 2 } else if f >= 0.1 { 1 } else if f >= 0.05 { 0 } else { 9 };
    if k < 4 {
      lv[k].0 += ra;
      lv[k].1 += tb;
      in_r += ra;
      in_t += tb;
    }
  }
  let tot_r: f64 = a.iter().sum::<f64>() * abs_ratio;
  let tot_t: f64 = b.iter().sum();
  let (block, block_shape, block_noise, blocks) = block_rms(render, truth, counts, w, h, LAB_BLOCK_PX);
  LabScore {
    sum_ratio: abs_ratio,
    rel_rms: (se / sb2.max(1e-300)).sqrt(),
    judged,
    noise,
    core_px,
    region_px: region.len(),
    levels: [0, 1, 2, 3].map(|k| lv[k].0 / lv[k].1.max(1e-300)),
    out_render: (tot_r - in_r) / tot_r.max(1e-300),
    out_truth: (tot_t - in_t) / tot_t.max(1e-300),
    block,
    block_shape,
    block_noise,
    blocks,
  }
}

fn print_lab_score(tag: &str, s: &LabScore) {
  println!(
    "{tag}: rel rms {:.4} (judged {:.4} after noise {:.4}, core {} px) over {} px, Σ render/truth {:.4}",
    s.rel_rms, s.judged, s.noise, s.core_px, s.region_px, s.sum_ratio
  );
  println!(
    "{tag}: ratio by truth level 0.05-0.1 {:.3}, 0.1-0.3 {:.3}, 0.3-0.6 {:.3}, 0.6-1 {:.3} | outside the truth's 5 % region: render {:.4}, truth {:.4}",
    s.levels[0], s.levels[1], s.levels[2], s.levels[3], s.out_render, s.out_truth
  );
  println!(
    "{tag}: block{LAB_BLOCK_PX} rms {:.4} (shape only {:.4}, the truth's count noise {:.4}) over {} blocks",
    s.block, s.block_shape, s.block_noise, s.blocks
  );
}

/// A frame against a reference frame of the same view (no truth, no noise): the Σ ratio, the
/// shape rel rms of the 5×5-boxed unit-sum images over the truth's region, the absolute 60-px
/// block rms against the reference's blocks (≥ [`LAB_BLOCK_REL`] of its brightest) and the worst
/// block `x_B / y_B − 1` with its block coordinates.
#[derive(Debug, Clone, Copy, Default)]
struct DirectCmp {
  sum_ratio: f64,
  rel_rms: f64,
  block: f64,
  worst: f64,
  worst_at: (usize, usize),
  grid: (usize, usize),
}

fn direct_compare(x: &[f64], y: &[f64], truth: &[f64], w: usize, h: usize) -> DirectCmp {
  let norm = |v: &[f64]| -> Vec<f64> {
    let s: f64 = v.iter().sum();
    v.iter().map(|q| q / s.max(1e-300)).collect()
  };
  let (sx, sy): (f64, f64) = (x.iter().sum(), y.iter().sum());
  let bt = mc_box(&norm(truth), w, h, 2);
  let peak = bt.iter().cloned().fold(0.0f64, f64::max);
  let (a, b) = (mc_box(&norm(x), w, h, 2), mc_box(&norm(y), w, h, 2));
  let (mut se, mut sb) = (0.0f64, 0.0f64);
  for i in 0..w * h {
    if bt[i] >= MC_REGION_REL * peak {
      se += (a[i] - b[i]).powi(2);
      sb += b[i] * b[i];
    }
  }
  let (xb, bw, bh) = block_sums(x, w, h, LAB_BLOCK_PX);
  let (yb, _, _) = block_sums(y, w, h, LAB_BLOCK_PX);
  let ymax = yb.iter().cloned().fold(0.0f64, f64::max);
  let (mut be, mut bs) = (0.0f64, 0.0f64);
  let (mut worst, mut worst_at) = (0.0f64, (0usize, 0usize));
  for k in 0..bw * bh {
    if ymax > 0.0 && yb[k] >= LAB_BLOCK_REL * ymax {
      be += (xb[k] - yb[k]).powi(2);
      bs += yb[k] * yb[k];
      let r = xb[k] / yb[k] - 1.0;
      if r.abs() > worst.abs() {
        worst = r;
        worst_at = (k % bw, k / bw);
      }
    }
  }
  DirectCmp {
    sum_ratio: sx / sy.max(1e-300),
    rel_rms: (se / sb.max(1e-300)).sqrt(),
    block: (be / bs.max(1e-300)).sqrt(),
    worst,
    worst_at,
    grid: (bw, bh),
  }
}

fn write_f64_file(path: &Path, v: &[f64]) -> std::io::Result<()> {
  let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
  std::fs::write(path, b)
}

fn read_f64_file(path: &Path) -> Option<Vec<f64>> {
  let b = std::fs::read(path).ok()?;
  if b.len() % 8 != 0 {
    return None;
  }
  Some(b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
}

/// `key value` lines (`path.txt`, `path_summary.txt`: the camera path's facts for the score)
fn write_kv(path: &Path, kv: &[(&str, String)]) {
  let mut s = String::new();
  for (k, v) in kv {
    s.push_str(&format!("{k} {v}\n"));
  }
  if let Err(e) = std::fs::write(path, s) {
    log_line(&format!("[observer] {}: {e}", path.display()));
  }
}

fn read_kv(path: &Path) -> Vec<(String, String)> {
  std::fs::read_to_string(path)
    .map(|s| {
      s.lines()
        .filter_map(|l| l.split_once(' ').map(|(k, v)| (k.to_string(), v.trim().to_string())))
        .collect()
    })
    .unwrap_or_default()
}

fn kv_get<'a>(kv: &'a [(String, String)], key: &str) -> Option<&'a str> {
  kv.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

/// the rotation by `angle` (rad) about the unit `axis` (Rodrigues), row-major
fn rot_axis(axis: V3, angle: f64) -> [[f64; 3]; 3] {
  let (s, c) = angle.sin_cos();
  let t = 1.0 - c;
  let [x, y, z] = axis;
  [
    [c + t * x * x, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, c + t * y * y, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, c + t * z * z],
  ]
}

fn mat3_apply(m: &[[f64; 3]; 3], v: V3) -> V3 {
  [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// The camera path's orbit axis: the ecliptic north's component across the start offset `rel0`
/// (eye − nucleus), unit; the camera keeps it as its up, so the orbit never rolls the view
fn path_axis(rel0: V3) -> V3 {
  let r = normalize(rel0);
  let z = [0.0, 0.0, 1.0];
  let d = dot(z, r);
  let a = [z[0] - d * r[0], z[1] - d * r[1], z[2] - d * r[2]];
  if norm(a) > 1e-9 { normalize(a) } else { normalize(cross(r, [1.0, 0.0, 0.0])) }
}

/// The camera path's eye offset from the nucleus at path frame `k` (the units of `rel0`): `rel0`
/// turned by `k·ω` about `axis`, its length less `k·dolly` (same units)
fn path_rel(rel0: V3, axis: V3, omega: f64, dolly: f64, k: u32) -> V3 {
  let r = mat3_apply(&rot_axis(axis, omega * k as f64), rel0);
  let n0 = norm(rel0);
  let s = (n0 - dolly * k as f64) / n0.max(1e-300);
  [r[0] * s, r[1] * s, r[2] * s]
}

/// The pixel of `target` (AU, heliocentric) and its depth along the view (AU) for a perspective
/// camera at `cam_pos` with orientation `cam_q` (the frame loop's own projection)
fn project_px(cam_pos: V3, cam_q: Q, target: V3, w: u32, h: u32, tan_h: f64) -> (f64, f64, f64) {
  let aspect = w as f64 / h as f64;
  let (right, forward, up) = camera_axes(cam_q);
  let d = sub(target, cam_pos);
  let (sx, sy, depth) = (dot(d, right), dot(d, up), dot(d, forward));
  (
    w as f64 / 2.0 * (1.0 + sx / depth.max(1e-12) / (tan_h * aspect)),
    h as f64 / 2.0 * (1.0 - sy / depth.max(1e-12) / tan_h),
    depth,
  )
}

/// `v0` seen from another camera (`cam_pos`, `cam_q`, AU) with the comet centre at `comet_au`:
/// the nucleus pixel, the metres per pixel at the nucleus, the nucleus radius on screen and the
/// hue ray follow the camera; every other field (planes, gates, photometry) is `v0`'s
/// (perspective views only)
fn mc_view_reposed(v0: &McView, cam_pos: V3, cam_q: Q, comet_au: V3) -> McView {
  let tan_h = v0.perspective_tan_h.unwrap_or(1.0);
  let (px, py, depth) = project_px(cam_pos, cam_q, comet_au, v0.width, v0.height, tan_h);
  let m_per_px = 2.0 * depth.max(1e-12) * aethervk_core_rlib::scene::dust::AU_M * tan_h / v0.height as f64;
  let (right, _, up) = camera_axes(cam_q);
  let anti = normalize(comet_au);
  let dv = (dot(anti, right), -dot(anti, up));
  let dn = (dv.0 * dv.0 + dv.1 * dv.1).sqrt();
  let nucleus_px_radius = NUCLEUS_RADIUS_KM as f64 * 1e3 / m_per_px;
  McView {
    width: v0.width,
    height: v0.height,
    cam_pos,
    cam_q,
    half_height_au: v0.half_height_au,
    m_per_px_nucleus: m_per_px,
    aspect: v0.aspect,
    perspective_tan_h: v0.perspective_tan_h,
    near_au: v0.near_au,
    far_au: v0.far_au,
    comet_au,
    nucleus_px: (px, py),
    mc_tier: v0.mc_tier,
    mc_age_max: v0.mc_age_max,
    mc_age_min: v0.mc_age_min,
    nucleus_px_radius,
    afrho_aperture_km: v0.afrho_aperture_km,
    afrho_in_cm: v0.afrho_in_cm,
    isotropic: v0.isotropic,
    hue_ray: HueRay {
      from: (px, py),
      dir: if dn > 1e-6 { (dv.0 / dn, dv.1 / dn) } else { (1.0, 0.0) },
      r0: (2.0 * nucleus_px_radius + 3.0).max(14.0),
    },
    stream: v0.stream,
    ttl_s: v0.ttl_s,
  }
}

/// the columns of the score table (`--lab-table FILE.tsv`, one row per scored frame)
const LAB_TABLE_COLUMNS: &[&str] = &[
  "source", "speed", "pose", "variant", "frame", "truth_grains", "judged", "judged_full", "sum_rt", "sum_rt_full", "sum_x_full",
  "lvl_005_01", "lvl_01_03", "lvl_03_06", "lvl_06_1", "block60", "block60_full", "direct_rel", "direct_block60", "worst_block",
  "fresh", "d3_share", "shown_light", "ghost", "ms_mean", "ms_p90", "ms_max", "clean",
];

/// one row of the score table, in [`LAB_TABLE_COLUMNS`] order (missing values `n/a`)
fn lab_table_append(path: &Path, row: &[(&str, String)]) {
  let new = !path.exists();
  let mut s = String::new();
  if new {
    s.push_str(&LAB_TABLE_COLUMNS.join("\t"));
    s.push('\n');
  }
  let cells: Vec<String> = LAB_TABLE_COLUMNS
    .iter()
    .map(|c| row.iter().find(|(k, _)| k == c).map(|(_, v)| v.clone()).unwrap_or_else(|| "n/a".into()))
    .collect();
  s.push_str(&cells.join("\t"));
  s.push('\n');
  match std::fs::OpenOptions::new().create(true).append(true).open(path) {
    Ok(mut f) => {
      let _ = f.write_all(s.as_bytes());
    }
    Err(e) => eprintln!("table {}: {e}", path.display()),
  }
}

/// `--lab-table-md FILE.tsv`: the score table as Markdown, one table per (source, speed), rows by
/// pose then variant (full, aa, eps…, fb), with the cadence's columns against the full frame
fn lab_table_markdown(path: &Path) -> i32 {
  let Ok(text) = std::fs::read_to_string(path) else {
    eprintln!("table: cannot read {}", path.display());
    return 2;
  };
  let mut lines = text.lines();
  let Some(head) = lines.next() else {
    return 2;
  };
  let cols: Vec<&str> = head.split('\t').collect();
  let rows: Vec<Vec<&str>> = lines.filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect()).collect();
  let col = |r: &Vec<&str>, name: &str| -> String {
    cols.iter().position(|c| *c == name).and_then(|i| r.get(i)).map(|s| s.to_string()).unwrap_or_else(|| "n/a".into())
  };
  let variant_rank = |v: &str| -> (u32, String) {
    match v {
      "full" => (0, String::new()),
      "aa" => (1, String::new()),
      "fb" => (3, String::new()),
      v => (2, format!("{:>12}", v)),
    }
  };
  let mut groups: Vec<(String, String)> = Vec::new();
  for r in &rows {
    let g = (col(r, "source"), col(r, "speed"));
    if !groups.contains(&g) {
      groups.push(g);
    }
  }
  for (src, speed) in groups {
    println!("\n### {src}, speed {speed} (frame scored against the same truth as the full frame)\n");
    println!(
      "| pose | variant | frame | judged (Δ full) | Σ render/truth | Σ x/full | ratio by level 0.05–0.1 / 0.1–0.3 / 0.3–0.6 / 0.6–1 | block60 (Δ full) | direct vs full: rel / block60 / worst block | fresh | D3 share | shown light | ghost | ms mean / p90 / max | clean |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut sel: Vec<&Vec<&str>> = rows.iter().filter(|r| col(r, "source") == src && col(r, "speed") == speed).collect();
    sel.sort_by(|a, b| {
      (col(a, "pose"), variant_rank(&col(a, "variant")), col(a, "frame"))
        .cmp(&(col(b, "pose"), variant_rank(&col(b, "variant")), col(b, "frame")))
    });
    let num = |s: &str| s.parse::<f64>().ok();
    for r in sel {
      let d = |a: &str, b: &str| match (num(&col(r, a)), num(&col(r, b))) {
        (Some(x), Some(y)) => format!("{x:.4} ({:+.4})", x - y),
        _ => col(r, a),
      };
      println!(
        "| {} | {} | {} | {} | {} | {} | {} / {} / {} / {} | {} | {} / {} / {} | {} | {} | {} | {} | {} / {} / {} | {} |",
        col(r, "pose"),
        col(r, "variant"),
        col(r, "frame"),
        d("judged", "judged_full"),
        col(r, "sum_rt"),
        col(r, "sum_x_full"),
        col(r, "lvl_005_01"),
        col(r, "lvl_01_03"),
        col(r, "lvl_03_06"),
        col(r, "lvl_06_1"),
        d("block60", "block60_full"),
        col(r, "direct_rel"),
        col(r, "direct_block60"),
        col(r, "worst_block"),
        col(r, "fresh"),
        col(r, "d3_share"),
        col(r, "shown_light"),
        col(r, "ghost"),
        col(r, "ms_mean"),
        col(r, "ms_p90"),
        col(r, "ms_max"),
        col(r, "clean"),
      );
    }
  }
  0
}

/// `--lab-score DIR --against FULL --lab-frame F --lab-truth N`: the frame `F` of a camera-path
/// capture (`DIR/frame_FFF.pyr`, the GPU pyramid: the composite's words) and the fully drawn
/// frame of the same path (`FULL`) scored against ONE truth: `FULL/truth_FFF.bin`'s inputs drawn
/// once at N grains and cached as `FULL/truthN_FFF.f64` (+ `_counts.f64`), both runs' truth
/// inputs byte-equal (else exit 2: not the same frame). Prints both scores ([`lab_score`]), the
/// direct comparison ([`direct_compare`]) and the path's facts (`path.txt`,
/// `path_summary.txt`); `--lab-table T.tsv` appends the row; `--check` fails when either path is
/// not clean or `|Σ x / Σ full − 1| > 1 % + the ghost share` (0 without the cadence's telemetry).
fn run_lab_score(args: &Args) -> i32 {
  let dir = args.lab_score.clone().unwrap_or_default();
  let full = args.lab_against.clone().unwrap_or_else(|| dir.clone());
  let f = args.lab_frame;
  let n = args.lab_truth;
  if n == 0 {
    eprintln!("score: --lab-truth N is required (the truth's grain count is part of its name)");
    return 2;
  }
  let tpath = full.join(format!("truth_{f:03}.bin"));
  let Ok(tb_full) = std::fs::read(&tpath) else {
    eprintln!("score: no {}", tpath.display());
    return 2;
  };
  if dir != full {
    match std::fs::read(dir.join(format!("truth_{f:03}.bin"))) {
      Ok(b) if b == tb_full => println!("score: truth inputs of frame {f} byte-equal in both runs ({} bytes): the same frame", b.len()),
      Ok(b) => {
        let first = b.iter().zip(&tb_full).position(|(x, y)| x != y).unwrap_or(b.len().min(tb_full.len()));
        eprintln!(
          "score: {} and {} are not the same frame: truth inputs {} vs {} bytes, first difference at byte {first}",
          dir.display(),
          full.display(),
          b.len(),
          tb_full.len()
        );
        return 2;
      }
      Err(_) => {
        eprintln!("score: no truth_{f:03}.bin in {}", dir.display());
        return 2;
      }
    }
  }
  let Some((inputs, view)) = load_truth_inputs(&tpath) else {
    eprintln!("score: cannot parse {}", tpath.display());
    return 2;
  };
  let (w, h) = (view.width, view.height);
  let (wu, hu) = (w as usize, h as usize);
  let load_pyr = |d: &Path| -> Option<Vec<u32>> {
    let b = std::fs::read(d.join(format!("frame_{f:03}.pyr"))).ok()?;
    Some(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
  };
  let (Some(pyr_full), Some(pyr_x)) = (load_pyr(&full), load_pyr(&dir)) else {
    eprintln!("score: frame_{f:03}.pyr missing in {} or {}", full.display(), dir.display());
    return 2;
  };
  let tc = full.join(format!("truth{n}_{f:03}.f64"));
  let cc = full.join(format!("truth{n}_{f:03}_counts.f64"));
  let (truth, counts) = match (read_f64_file(&tc), read_f64_file(&cc)) {
    (Some(t), Some(c)) if t.len() == wu * hu && c.len() == wu * hu => {
      println!("score: truth {} (cached, {n} grains)", tc.display());
      (t, c)
    }
    _ => {
      let t1 = Instant::now();
      let Some((rec, _, _, t, c)) = monte_carlo_truth_from(&pyr_full, &inputs, &view, n) else {
        eprintln!("score: the truth draw returned nothing");
        return 2;
      };
      if let Err(e) = write_f64_file(&tc, &t).and_then(|_| write_f64_file(&cc, &c)) {
        eprintln!("score: cannot cache the truth: {e}");
      }
      println!(
        "score: drew the truth: {} grains, {} batches, {:.1} s → {}",
        rec.grains,
        rec.batches,
        t1.elapsed().as_secs_f64(),
        tc.display()
      );
      (t, c)
    }
  };
  let img = |p: &[u32]| tau_image_of_pyramid(p, w, h).map(|v| v.iter().map(|&x| x as f64).collect::<Vec<f64>>());
  let (Some(rf), Some(rx)) = (img(&pyr_full), img(&pyr_x)) else {
    eprintln!("score: a pyramid has no τ image");
    return 2;
  };
  let (pk_x, ps_x) = (read_kv(&dir.join("path.txt")), read_kv(&dir.join("path_summary.txt")));
  let (pk_f, ps_f) = (read_kv(&full.join("path.txt")), read_kv(&full.join("path_summary.txt")));
  let label = args
    .lab_label
    .clone()
    .or_else(|| kv_get(&pk_x, "variant").map(String::from))
    .unwrap_or_else(|| dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default());
  let sf = lab_score(&rf, &truth, &counts, wu, hu);
  let sx = lab_score(&rx, &truth, &counts, wu, hu);
  print_lab_score(&format!("score frame {f} full"), &sf);
  print_lab_score(&format!("score frame {f} {label}"), &sx);
  let dc = direct_compare(&rx, &rf, &truth, wu, hu);
  println!(
    "score frame {f} {label} vs full: Σ {:.5}, shape rel rms {:.4}, block{LAB_BLOCK_PX} rms {:.4}, worst block {:+.4} at ({}, {}) of {}×{}",
    dc.sum_ratio, dc.rel_rms, dc.block, dc.worst, dc.worst_at.0, dc.worst_at.1, dc.grid.0, dc.grid.1
  );
  let clean = |s: &[(String, String)]| kv_get(s, "clean") == Some("1");
  let ghost: Option<f64> = kv_get(&ps_x, "ghost_share").and_then(|v| v.parse().ok());
  for (tag, s) in [("full", &ps_f), (label.as_str(), &ps_x)] {
    println!(
      "score: {tag}: path clean {} ({} steps), GPU ms per path frame mean {} p90 {} max {} over {} frames, cadence {}",
      kv_get(s, "clean").unwrap_or("?"),
      kv_get(s, "steps").unwrap_or("?"),
      kv_get(s, "ms_mean").unwrap_or("n/a"),
      kv_get(s, "ms_p90").unwrap_or("n/a"),
      kv_get(s, "ms_max").unwrap_or("n/a"),
      kv_get(s, "gpu_frames").unwrap_or("0"),
      kv_get(s, "cadence").unwrap_or("n/a")
    );
  }
  let mut fails: Vec<String> = Vec::new();
  // the same path: every parameter that moves the camera or the clock equal in both runs
  for key in ["pose", "inside_m", "width", "height", "frames", "fps", "dt_s", "omega_deg_per_frame", "dolly_m_per_frame", "axis", "rel0_m", "epoch0"] {
    if kv_get(&pk_x, key) != kv_get(&pk_f, key) {
      fails.push(format!("not the same path: {key} {:?} vs full {:?}", kv_get(&pk_x, key), kv_get(&pk_f, key)));
    }
  }
  if !clean(&ps_x) || !clean(&ps_f) {
    fails.push(format!("a camera path is not clean (full {}, {label} {})", clean(&ps_f), clean(&ps_x)));
  }
  let bound = 0.01 + ghost.unwrap_or(0.0);
  if !((dc.sum_ratio - 1.0).abs() <= bound) {
    fails.push(format!("Σ {label} / Σ full {:.5} beyond 1 % + the ghost share ({bound:.4})", dc.sum_ratio));
  }
  println!(
    "check lab_cadence_score frame {f} {label}: {} (Σ ratio {:.5} within ±{bound:.4}{})",
    if fails.is_empty() { "PASS" } else { "FAIL" },
    dc.sum_ratio,
    if ghost.is_none() { "; ghost n/a: no cadence telemetry, 0 used" } else { "" }
  );
  for e in &fails {
    println!("  - {e}");
  }
  if let Some(t) = &args.lab_table {
    let s = |k: &str| kv_get(&ps_x, k).unwrap_or("n/a").to_string();
    lab_table_append(
      t,
      &[
        ("source", "GPU".into()),
        ("speed", kv_get(&pk_x, "speed").unwrap_or("?").into()),
        ("pose", kv_get(&pk_x, "pose_label").unwrap_or("?").into()),
        ("variant", label.clone()),
        ("frame", f.to_string()),
        ("truth_grains", n.to_string()),
        ("judged", format!("{:.4}", sx.judged)),
        ("judged_full", format!("{:.4}", sf.judged)),
        ("sum_rt", format!("{:.4}", sx.sum_ratio)),
        ("sum_rt_full", format!("{:.4}", sf.sum_ratio)),
        ("sum_x_full", format!("{:.5}", dc.sum_ratio)),
        ("lvl_005_01", format!("{:.3}", sx.levels[0])),
        ("lvl_01_03", format!("{:.3}", sx.levels[1])),
        ("lvl_03_06", format!("{:.3}", sx.levels[2])),
        ("lvl_06_1", format!("{:.3}", sx.levels[3])),
        ("block60", format!("{:.4}", sx.block)),
        ("block60_full", format!("{:.4}", sf.block)),
        ("direct_rel", format!("{:.4}", dc.rel_rms)),
        ("direct_block60", format!("{:.4}", dc.block)),
        ("worst_block", format!("{:+.4}@{},{}", dc.worst, dc.worst_at.0, dc.worst_at.1)),
        ("fresh", s("fresh_fraction")),
        ("d3_share", s("d3_share")),
        ("shown_light", s("shown_light_share")),
        ("ghost", s("ghost_share")),
        ("ms_mean", s("ms_mean")),
        ("ms_p90", s("ms_p90")),
        ("ms_max", s("ms_max")),
        ("clean", format!("{}", clean(&ps_x) as u8)),
      ],
    );
  }
  if args.check && !fails.is_empty() { 1 } else { 0 }
}

/// `--lab DIR --lab-frame F --lab-path N --lab-truth M [--lab-path-eps 1,2,4,16]`: the CPU mirror
/// of the cadence (`scene::dust::cadence`) along a camera path of N frames from the captured view
/// of `snapshot_FFF.bin` — Δt = 0 (the snapshot's moments hold for every frame), the eye orbiting
/// the nucleus at `--orbit-deg-per-s` (or `--eye-step-m`) at `--path-fps`, as `--path-frames`
/// does on the GPU. The path's projection comes from the captured mvp itself: with
/// `Q = [M₀(3×3) | M₀ col3 + M₀(3×3)·eye_local]` (the projection·view of the capture, metres)
/// every frame's mvp is `Q·ΔRᵀ·[I | anchor − eye_k]` (`CadenceView` with `proj = Q`,
/// `view_rot = ΔRᵀ`, units 1), checked against every tier's captured mvp at frame 0. Every frame
/// the full parallel splat is drawn too (shared by every ε) and per ε the cadence's frame; the
/// last frame of both is scored against the truth drawn at the last view (M grains). The check
/// `lab_cadence_scores_the_path` (`--check`): frame 0 word-identical to the full draw (with the
/// in-frame gates off, `OBSERVER_LAB_GATES_OFF=1`; printed otherwise), every frame
/// `|Σ cadence / Σ full − 1| ≤ 1 %` (the CPU test's bound; no deaths at Δt = 0, so no ghost),
/// the owner bytes consistent every frame, at least one class image shown (else vacuous).
fn run_lab_path(dir: &Path, args: &Args) -> i32 {
  use aethervk_core_rlib::scene::dust as d;
  use d::cadence as cad;
  let t_all = Instant::now();
  let frame = args.lab_frame;
  let nf = args.lab_path as usize;
  if args.lab_truth == 0 || nf < 2 {
    eprintln!("lab path: needs --lab-truth N and --lab-path ≥ 2");
    return 2;
  }
  let Some((w, h, header, tiers)) = load_snapshot(&dir.join(format!("snapshot_{frame:03}.bin"))) else {
    eprintln!("lab path: no snapshot_{frame:03}.bin in {}", dir.display());
    return 2;
  };
  let Some((inputs, view0)) = load_truth_inputs(&dir.join(format!("truth_{frame:03}.bin"))) else {
    eprintln!("lab path: no truth_{frame:03}.bin in {}", dir.display());
    return 2;
  };
  let Some(tan_h) = view0.perspective_tan_h else {
    eprintln!("lab path: perspective captures only (the inside pose)");
    return 2;
  };
  let layout = d::PyramidLayout::new(w, h);
  let hn = d::PYRAMID_HEADER_WORDS as usize;
  // the frame header: the snapshot's with the frame's counters zeroed (as `run_lab`), the gates
  // and the level of detail by the lab's knobs
  let mut hdr = header.clone();
  {
    use aethervk_core_rlib::scene::dust::specks as sp;
    for wd in [
      d::PYR_TAU_MAX,
      d::PYR_LEVEL_MASK,
      d::PYR_SPLIT_CHILDREN,
      d::PYR_VOLUME_EVALS,
      d::PYR_EWA_TEXELS,
      d::PYR_SPLIT_DROPPED_FLUX,
      sp::PYR_SPECKS,
      sp::PYR_SPECKS_OVERFLOW,
      sp::PYR_SPECKS_TRUNCATED,
      sp::PYR_N_EYE,
      sp::PYR_N_EYE_HI,
      sp::PYR_SPECKS_NODES,
      sp::PYR_SPECKS_COUNTS,
    ] {
      hdr[wd as usize] = 0;
    }
  }
  let sd = d::PYR_SPLIT_DEPTH as usize;
  let env_u32 = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<u32>().ok());
  if let Some(n) = env_u32("OBSERVER_LAB_GATES_OFF") {
    hdr[sd] = (hdr[sd] & !(1 << d::DUST_FRAME_GATES_OFF_BIT)) | ((n & 1) << d::DUST_FRAME_GATES_OFF_BIT);
  }
  if let Some(v) = std::env::var("OBSERVER_LAB_LOD").ok().and_then(|v| v.trim().parse::<f32>().ok()) {
    hdr[d::PYR_LOD as usize] = v.max(0.0).to_bits();
  }
  let gates_off = (hdr[sd] >> d::DUST_FRAME_GATES_OFF_BIT) & 1 == 1;
  println!(
    "lab path: {} frames at {}×{} from {} frame {frame}; header split depth {} volume bias {} lod {} px, in-frame gates {}",
    nf,
    w,
    h,
    dir.display(),
    hdr[sd] & 0xFF,
    (hdr[sd] >> 8) & 0xFF,
    d::lod_px(&hdr),
    if gates_off { "off" } else { "on" }
  );
  let au = d::AU_M;
  let eye0 = [view0.cam_pos[0] * au, view0.cam_pos[1] * au, view0.cam_pos[2] * au];
  let nuc = [view0.comet_au[0] * au, view0.comet_au[1] * au, view0.comet_au[2] * au];
  println!(
    "lab path: nucleus (the view's comet centre) − the truth's r_centre_now {:.3} m",
    norm(sub(nuc, inputs.r_centre_now))
  );
  let r0 = sub(eye0, nuc);
  let r0n = norm(r0);
  let axis = path_axis(r0);
  let omega = match args.eye_step_m {
    Some(s) => s / r0n,
    None => args.orbit_deg_per_s.to_radians() / args.path_fps,
  };
  println!(
    "lab path: eye {:.3} m from the nucleus centre, orbit {:.4}° per frame ({:.3} m per frame) about the axis {:?}, Δt 0",
    r0n,
    omega.to_degrees(),
    omega * r0n,
    axis
  );
  // the capture's projection·view in metres (see the doc comment), from tier 0
  let pc0 = &tiers[0].0;
  let m0: Vec<f64> = pc0.mvp.iter().map(|&v| v as f64).collect();
  let el0 = [pc0.eye_local[0] as f64, pc0.eye_local[1] as f64, pc0.eye_local[2] as f64];
  let mut q = [0.0f64; 16];
  q.copy_from_slice(&m0);
  for r in 0..4 {
    q[12 + r] = m0[12 + r] + m0[r] * el0[0] + m0[4 + r] * el0[1] + m0[8 + r] * el0[2];
  }
  let anchors: Vec<[f64; 3]> = tiers
    .iter()
    .map(|t| {
      let e = t.0.eye_local;
      [eye0[0] - e[0] as f64, eye0[1] - e[1] as f64, eye0[2] - e[2] as f64]
    })
    .collect();
  let view_at = |k: usize| -> (cad::CadenceView, [f64; 3], [[f64; 3]; 3]) {
    let dr = rot_axis(axis, omega * k as f64);
    let rk = mat3_apply(&dr, r0);
    let eye = [nuc[0] + rk[0], nuc[1] + rk[1], nuc[2] + rk[2]];
    // ΔRᵀ, column-major 4×4
    let mut vr = [0.0f64; 16];
    for c in 0..3 {
      for r in 0..3 {
        vr[c * 4 + r] = dr[c][r];
      }
    }
    vr[15] = 1.0;
    (cad::CadenceView { view_rot: vr, proj: q, eye_m: eye, units_per_m: 1.0, width: w, height: h }, eye, dr)
  };
  // the reconstruction against every tier's captured mvp
  {
    let (v, _, _) = view_at(0);
    let mut worst = 0.0f64;
    for (ti, t) in tiers.iter().enumerate() {
      let m = v.mvp(anchors[ti]);
      let scale = t.0.mvp.iter().map(|x| x.abs() as f64).fold(0.0f64, f64::max).max(1e-300);
      let err = m.iter().zip(&t.0.mvp).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0f64, f64::max) / scale;
      println!("lab path: tier {ti} anchor − nucleus {:.3} m, mvp reconstructed to {err:.2e} of its largest entry", norm(sub(anchors[ti], nuc)));
      worst = worst.max(err);
    }
    if !(worst <= 1e-6) {
      eprintln!("lab path: the path's projection does not reproduce the captured mvp ({worst:.2e} > 1e-6): no score");
      return 2;
    }
  }
  // the truth's camera: the engine's look-at from the orbit's offset with the axis as up; its
  // forward must be the orbit's rotation of the captured forward
  let cam_at = |k: usize| -> (V3, Q) {
    let rk = path_rel(sub(view0.cam_pos, view0.comet_au), axis, omega, 0.0, k as u32);
    let pos = [view0.comet_au[0] + rk[0], view0.comet_au[1] + rk[1], view0.comet_au[2] + rk[2]];
    (pos, earth_observer::look_at_origin_from(rk, Some(axis)))
  };
  {
    let (_, f0, _) = camera_axes(view0.cam_q);
    let (_, fa, _) = camera_axes(cam_at(0).1);
    let (_, fl, _) = camera_axes(cam_at(nf - 1).1);
    let (_, _, drl) = view_at(nf - 1);
    let fr = mat3_apply(&drl, f0);
    let e0 = norm(sub(f0, fa));
    let el = norm(sub(fl, fr));
    println!("lab path: the truth's camera: forward at frame 0 off the capture's by {e0:.2e}, at frame {} off the rotated one by {el:.2e}", nf - 1);
    if !(e0 <= 1e-9 && el <= 1e-9) {
      eprintln!("lab path: the truth's camera does not follow the path: no score");
      return 2;
    }
  }
  let n_t: Vec<usize> = tiers.iter().map(|t| (t.0.live_count as usize).min(t.3.len()).min(t.4.len())).collect();
  let live_total: usize = n_t.iter().sum();
  let threads = observer_threads();
  let pcs_at = |v: &cad::CadenceView| -> Vec<d::DustSplatPushConstants> {
    tiers
      .iter()
      .enumerate()
      .map(|(ti, t)| {
        let mut p = t.0;
        p.mvp = v.mvp(anchors[ti]);
        let e = v.eye_local(anchors[ti]);
        p.eye_local = [e[0], e[1], e[2], t.0.eye_local[3]];
        p.live_count = n_t[ti] as u32;
        p
      })
      .collect()
  };
  let fresh_pyr = || {
    let mut p = vec![0u32; layout.total_words as usize];
    p[..hn].copy_from_slice(&hdr);
    p
  };
  let levels_end = layout.measure.0 as usize;
  let words_differ = |a: &[u32], b: &[u32]| a[hn..levels_end].iter().zip(&b[hn..levels_end]).filter(|(x, y)| x != y).count();
  // the full draws (every ε compares with them)
  let mut full_light: Vec<f64> = Vec::with_capacity(nf);
  let mut full_pl: Vec<f64> = Vec::with_capacity(nf);
  let mut full0: Vec<u32> = Vec::new();
  let mut full_last: Vec<u32> = Vec::new();
  for k in 0..nf {
    let t1 = Instant::now();
    let (v, _, _) = view_at(k);
    let pcs = pcs_at(&v);
    let mut p = fresh_pyr();
    for (ti, t) in tiers.iter().enumerate() {
      d::splat_tier_parallel(&t.4[..n_t[ti]], &t.3[..n_t[ti]], &pcs[ti], &t.1, t.2, &layout, &mut p, threads);
    }
    full_light.push(frame_light(&layout, &p));
    full_pl.push(cad::pyramid_light(&layout, &p));
    println!(
      "lab path full {k}: light Σ counts {:.6e} (cadence::pyramid_light Σ counts·4^ℓ {:.6e}), {:.1} s",
      full_light[k],
      full_pl[k],
      t1.elapsed().as_secs_f64()
    );
    if k == 0 {
      full0 = p.clone();
    }
    if k == nf - 1 {
      full_last = p;
    }
  }
  // the truth at the last view
  let (cam_l, q_l) = cam_at(nf - 1);
  let view_l = mc_view_reposed(&view0, cam_l, q_l, view0.comet_au);
  println!(
    "lab path: the last view: nucleus px ({:.2}, {:.2}), {:.4} m/px at the nucleus (tan_h {tan_h:.4})",
    view_l.nucleus_px.0, view_l.nucleus_px.1, view_l.m_per_px_nucleus
  );
  let t1 = Instant::now();
  let Some((rec, _, _, truth, counts)) = monte_carlo_truth_from(&full_last, &inputs, &view_l, args.lab_truth) else {
    eprintln!("lab path: the truth draw returned nothing");
    return 2;
  };
  println!(
    "lab path: drew the truth at frame {}: {} grains, {} batches, {:.1} s",
    nf - 1,
    rec.grains,
    rec.batches,
    t1.elapsed().as_secs_f64()
  );
  let (wu, hu) = (w as usize, h as usize);
  let img = |p: &[u32]| tau_image_of_pyramid(p, w, h).map(|v| v.iter().map(|&x| x as f64).collect::<Vec<f64>>());
  let Some(rf) = img(&full_last) else {
    eprintln!("lab path: the full pyramid has no τ image");
    return 2;
  };
  let sf = lab_score(&rf, &truth, &counts, wu, hu);
  print_lab_score(&format!("lab path frame {} full", nf - 1), &sf);
  let pose_label = {
    let dm = r0n - NUCLEUS_RADIUS_KM as f64 * 1e3;
    if dm >= 1e3 { format!("{:.0}km", dm / 1e3) } else { format!("{:.0}m", dm) }
  };
  let mut all_fail = false;
  for &eps in &args.lab_path_eps {
    let t_eps = Instant::now();
    // the production configuration (every class, two phases, the default image margin M = 0.2)
    // at this ε; `AETHERVK_DUST_CADENCE_MARGIN=<fraction>` as the device reads it (0 = the
    // footprint rule). Frame 0 is a First frame (everything fresh into the viewport pyramid), so
    // its word-identity with the full draw holds at any margin; the reconstruction check compares
    // the viewport mvp (`CadenceView::mvp`), never `mvp_image`
    let mut cfg = cad::CadenceConfig::full();
    cfg.eps_px = eps;
    if let Some(mf) = std::env::var("AETHERVK_DUST_CADENCE_MARGIN").ok().and_then(|v| v.trim().parse::<f32>().ok())
      && mf >= 0.0
      && mf.is_finite()
    {
      cfg.margin_frac = mf;
    }
    println!(
      "lab path ε {eps}: classes {:?} × {} phases, image margin {} of the viewport per side",
      cfg.classes, cfg.phases, cfg.margin_frac
    );
    let mut st = cad::DustCadenceState::new(cfg);
    st.threads = threads;
    let mut moments: Vec<Vec<d::DustMoments>> = tiers.iter().enumerate().map(|(ti, t)| t.4[..n_t[ti]].to_vec()).collect();
    let mut fails: Vec<String> = Vec::new();
    let mut last: Vec<u32> = Vec::new();
    let mut shown_frames = 0usize;
    let mut admitted_prev: Vec<bool> = Vec::new();
    let (mut last_fresh, mut last_d3, mut last_built, mut last_share) = (0usize, 0u32, 0usize, f64::NAN);
    let unit_now = 1.0 / tiers[0].0.inv_unit;
    for k in 0..nf {
      let (v, _, _) = view_at(k);
      let pcs = pcs_at(&v);
      let fm = cad::FrameMeta {
        view: v,
        t_now_s: inputs.t_now,
        speed: 0.0,
        nucleus_m: nuc,
        flags: 0,
        fingerprint: 1,
        reused: false,
        unit_now,
      };
      {
        let metas: Vec<cad::TierMeta<'_>> = tiers
          .iter()
          .enumerate()
          .map(|(ti, t)| cad::TierMeta {
            key: ti as u64,
            first_slot: 0,
            capacity: (n_t[ti].max(1)).next_power_of_two() as u32,
            live_count: n_t[ti] as u32,
            emitted: 1,
            rewritten_from: n_t[ti] as u32,
            anchor_m: anchors[ti],
            render: &t.3[..n_t[ti]],
          })
          .collect();
        // Δt = 0: every slot's moments are the snapshot's, so the plan's propagate list is moot
        let _plan = st.begin(&fm, &metas, &layout);
      }
      let mut fp = fresh_pyr();
      let rep = {
        let mut draws: Vec<cad::TierDraw<'_>> = moments
          .iter_mut()
          .enumerate()
          .map(|(ti, m)| cad::TierDraw {
            moments: m.as_mut_slice(),
            render: &tiers[ti].3[..n_t[ti]],
            pc: pcs[ti],
            flow: tiers[ti].1,
            specks_pc: None,
            // Δt = 0 on the snapshot: no propagate ran, the generation stays 0 (the CPU device
            // path's value; `cadence::TierDraw::stamp`)
            stamp: 0,
          })
          .collect();
        st.finish(&mut draws, &layout, &mut fp)
      };
      if let Err(e) = st.check_owner_bytes() {
        fails.push(format!("frame {k}: owner bytes: {e}"));
      }
      // the check's light: Σ counts of the composited levels (`frame_light`); the cadence's own
      // `pyramid_light` ratio printed beside it
      let lf = frame_light(&layout, &fp);
      let ratio = lf / full_light[k].max(1e-300);
      let ratio_pl = cad::pyramid_light(&layout, &fp) / full_pl[k].max(1e-300);
      let fresh: usize = rep.fresh.iter().map(|v| v.len()).sum();
      let shown: usize = st.tiers.values().map(|t| t.lists.iter().flatten().map(|l| l[0].len()).sum::<usize>()).sum();
      shown_frames += (rep.images_shown > 0) as usize;
      println!(
        "lab path ε {eps} {k}: mode {:?} reset {:?} fresh {fresh} ({:.1} % of {live_total}) shown {shown} images {} built {} d3 {} aborts {} swaps {:?} classified {:?} hist {:?}: Σ ratio {ratio:.5} (pyramid_light {ratio_pl:.5})",
        rep.mode,
        rep.reset,
        100.0 * fresh as f64 / live_total.max(1) as f64,
        rep.images_shown,
        rep.drawn_build.len(),
        rep.d3_rejected,
        rep.aborts.len(),
        rep.swaps,
        rep.classified,
        rep.class_hist
      );
      // the aborts in full (the rate rule, the motion budget, the predicted displacement, the
      // coverage of a shown image)
      for a in &rep.aborts {
        let why = match a.reason {
          cad::ABORT_RATE => "rate",
          cad::ABORT_BUDGET => "budget",
          cad::ABORT_PREDICTION => "prediction",
          cad::ABORT_COVERAGE => "coverage",
          _ => "?",
        };
        println!(
          "lab path ε {eps} {k}: abort class {} phase {} list {} ({}) by {why}: {} entries, excess {:.4} px vs bound {:.4} px, g_max {:.4e} px/m, shown kept {} for {} frames",
          a.k,
          a.p,
          a.list,
          ["shown", "build", "pending"].get(a.list as usize).copied().unwrap_or("?"),
          a.entries,
          a.excess_px,
          a.bound_px,
          a.g_max,
          a.shown_entries,
          a.ghost_frames_left
        );
      }
      // the class admission by the motion budget (`2N · U(N) ≤ m − G`), when it changes
      let admitted: Vec<bool> = (0..st.cfg.classes.len()).map(|c| st.class_admitted(c)).collect();
      if admitted != admitted_prev {
        let b = st.motion_budget();
        let u: Vec<[f32; 2]> = (0..st.cfg.classes.len()).map(|c| st.motion_window(c)).collect();
        println!(
          "lab path ε {eps} {k}: classes {:?} admitted {admitted:?} (motion window U per class {u:?} px, budget {b:?} px)",
          st.cfg.classes
        );
        admitted_prev = admitted;
      }
      for c in &rep.cycles {
        println!(
          "lab path ε {eps} {k}: cycle class {} phase {}: the ended build's D3 rejected {} of {} ({:.1} %); the shown images hold {:.2} % of the frame's light",
          c.k,
          c.p,
          c.d3_rejected,
          c.d3_rejected + c.drawn,
          100.0 * c.d3_rejected as f64 / (c.d3_rejected + c.drawn).max(1) as f64,
          100.0 * c.shown_light_share
        );
        last_share = c.shown_light_share;
      }
      if k == 0 {
        let diff = words_differ(&fp, &full0);
        println!("lab path ε {eps} 0: the cadence's first frame against the full draw: {diff} level words differ (gates {})", if gates_off { "off" } else { "on" });
        if gates_off && diff > 0 {
          fails.push(format!("frame 0: {diff} level words differ from the full draw with the gates off"));
        }
      }
      if !((ratio - 1.0).abs() <= 0.01) {
        fails.push(format!("frame {k}: Σ ratio {ratio:.5} beyond 1 % (no ghost at Δt = 0)"));
      }
      if k == nf - 1 {
        last_fresh = fresh;
        last_d3 = rep.d3_rejected;
        last_built = rep.drawn_build.len();
        last = fp;
      }
    }
    // the earliest path frame a class image can show: the smallest class's first cycle starts at
    // path frame 1 (frame 0 is the First frame, the cycles count the frames after it), its
    // classify fills the pending list, the next start promotes it to the build (N frames), the
    // start after shows it: 1 + 2·N_min (class 8: frame 17, its second phase 21; class 16: 33/41)
    let n_min = st.cfg.classes.iter().copied().min().unwrap_or(8) as usize;
    let earliest = 1 + 2 * n_min;
    if shown_frames == 0 {
      if nf > earliest {
        fails.push(format!(
          "vacuous: no class image was shown on any frame, though frame {earliest} is the earliest a class-{n_min} image can show"
        ));
      } else {
        println!(
          "lab path ε {eps}: {nf} frames cannot show a class image (the earliest is frame {earliest}): the vacuous rule is not applied; run ≥ {} frames",
          earliest + 2 * n_min
        );
      }
    }
    let Some(rx) = img(&last) else {
      eprintln!("lab path: the cadence's last pyramid has no τ image");
      return 2;
    };
    let sx = lab_score(&rx, &truth, &counts, wu, hu);
    print_lab_score(&format!("lab path frame {} ε {eps}", nf - 1), &sx);
    let dc = direct_compare(&rx, &rf, &truth, wu, hu);
    println!(
      "lab path frame {} ε {eps} vs full: Σ {:.5}, shape rel rms {:.4}, block{LAB_BLOCK_PX} rms {:.4}, worst block {:+.4} at ({}, {}) of {}×{}; judged {:.4} vs {:.4}; {:.1} s",
      nf - 1,
      dc.sum_ratio,
      dc.rel_rms,
      dc.block,
      dc.worst,
      dc.worst_at.0,
      dc.worst_at.1,
      dc.grid.0,
      dc.grid.1,
      sx.judged,
      sf.judged,
      t_eps.elapsed().as_secs_f64()
    );
    println!(
      "check lab_cadence_scores_the_path ε {eps}: {} ({} shown frames of {nf})",
      if fails.is_empty() { "PASS" } else { "FAIL" },
      shown_frames
    );
    for e in &fails {
      println!("  - {e}");
    }
    all_fail |= !fails.is_empty();
    if let Some(t) = &args.lab_table {
      lab_table_append(
        t,
        &[
          ("source", "CPU mirror".into()),
          ("speed", "0".into()),
          ("pose", pose_label.clone()),
          ("variant", format!("eps{eps}")),
          ("frame", (nf - 1).to_string()),
          ("truth_grains", args.lab_truth.to_string()),
          ("judged", format!("{:.4}", sx.judged)),
          ("judged_full", format!("{:.4}", sf.judged)),
          ("sum_rt", format!("{:.4}", sx.sum_ratio)),
          ("sum_rt_full", format!("{:.4}", sf.sum_ratio)),
          ("sum_x_full", format!("{:.5}", dc.sum_ratio)),
          ("lvl_005_01", format!("{:.3}", sx.levels[0])),
          ("lvl_01_03", format!("{:.3}", sx.levels[1])),
          ("lvl_03_06", format!("{:.3}", sx.levels[2])),
          ("lvl_06_1", format!("{:.3}", sx.levels[3])),
          ("block60", format!("{:.4}", sx.block)),
          ("block60_full", format!("{:.4}", sf.block)),
          ("direct_rel", format!("{:.4}", dc.rel_rms)),
          ("direct_block60", format!("{:.4}", dc.block)),
          ("worst_block", format!("{:+.4}@{},{}", dc.worst, dc.worst_at.0, dc.worst_at.1)),
          ("fresh", format!("{:.4}", last_fresh as f64 / live_total.max(1) as f64)),
          ("d3_share", format!("{:.4}", last_d3 as f64 / (last_d3 as usize + last_built).max(1) as f64)),
          ("shown_light", if last_share.is_finite() { format!("{last_share:.4}") } else { "n/a".into() }),
          ("ghost", "0".into()),
          ("clean", format!("{}", fails.is_empty() as u8)),
        ],
      );
    }
  }
  println!("lab path: done in {:.1} s", t_all.elapsed().as_secs_f64());
  if args.check && all_fail { 1 } else { 0 }
}
