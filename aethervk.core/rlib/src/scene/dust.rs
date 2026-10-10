//! Dust system v3: stateless Keplerian super-particles.
//!
//! A dust grain far from the nucleus feels only solar gravity and radiation pressure, both
//! `∝ 1/r²`, hence it follows an exact Kepler orbit with reduced gravitational parameter
//! `μ_eff = μ☉ (1 − β)` (Finson–Probstein). A *cluster* (super-particle) is therefore fully
//! described by its immutable emission record `(r₀, v₀, t₀, β)` and can be evaluated in closed form
//! at any time, without integration.
//!
//! This module is the **reference implementation**: the GLSL in `assets/sim/dust_common.glsl`
//! mirrors every function 1:1 and the CPU particle mode runs exactly this code.
//!
//! Precision: the GPU baseline has no `shaderFloat64`, so every quantity that needs more than f32
//! (heliocentric positions/velocities, times, μ, the Kepler solve) is double-float, see [`df`].
//! The Kepler solve converges in f32 first and polishes the universal anomaly in df64.
//!
//! Frames and units
//! - heliocentric root frame (Sun at origin), metres, seconds, m/s
//! - times are *scaled* simulation seconds since the start epoch
//! - grain radius in µm, density in g/cm³, mass in grams
#![allow(clippy::excessive_precision)]

pub mod df;
pub mod splat;
pub use splat::*;

use aethervk_oshal_rlib::math::FloatLike;
use df::{Df, Df3, consts};

/// ring capacity for discrete / high-end GPUs (power of two)
pub const RING_CAPACITY_HIGH: u32 = 262_144;
/// ring capacity for integrated / mobile GPUs (power of two)
pub const RING_CAPACITY_LOW: u32 = 32_768;
/// bits of the ring slot in a render cluster's `y` word ([`render_word`])
pub const RENDER_SLOT_BITS: u32 = 22;
const _: () = assert!(RING_CAPACITY_HIGH <= 1 << RENDER_SLOT_BITS);
/// Bits of the stable child-pattern id a cluster carries in the low mantissa bits of its β
/// half-spread (`DustCluster::misc.w`, copied to `DustRenderCluster::age_id_dbeta_flux.z`). The id
/// is hashed from the emission record (window seed, in-batch index), never from the ring slot, so
/// a cluster draws the same children after a seek, a rewind or in another tier's sub-ring. 16 bits
/// cost the half-spread ≤ 2⁻⁷ (truncation, irrelevant for a spread) and keep both structs at their
/// size; the id is exact on GPU and CPU (u32 ops only), the dbeta bits above it may differ by ulps.
pub const CHILD_ID_BITS: u32 = 16;
pub const CHILD_ID_MASK: u32 = (1 << CHILD_ID_BITS) - 1;
/// Emission streams per jet, at most. Stream `s` has a fixed direction in the jet cone, a fixed
/// grain-size stratum and a fixed speed draw, hashed from the jet configuration (never from the
/// window), so its clusters of consecutive time samples are points of one streakline: on a
/// spinning nucleus it sweeps a spiral / arc, at fixed β it is a syndyne. A tier uses
/// `S = 2^shift` streams ([`DustHostState::stream_shift`]: fewer on small rings, so a window still
/// holds [`STREAM_MIN_SAMPLES`] time samples). Batch layout `j = i·S + s` (time sample `i`), every
/// batch count a multiple of `S`, so in a tier's compact render buffer the stream predecessor of
/// cluster `r` is `r − S` ([`streak_pred`]).
pub const DUST_STREAMS: u32 = 64;
/// time samples per full window a tier keeps at least when choosing its stream count (2: at least
/// three samples per 2.8 h window on both default rings, a dozen per 12 h rotation for the
/// spirals, while a small ring spends its slots on directions rather than time; the strata floor
/// below keeps the size sampling fine whatever the ring)
pub const STREAM_MIN_SAMPLES: u32 = 2;
/// per-cluster jitter of a stream's direction: fraction of its cone stratum (polar) and of a turn
/// (azimuth)
pub const STREAM_DIR_JITTER: f32 = 0.03;
/// Lateral velocity dispersion of a stream in units of its cone cell's angular radius
/// (`aperture / √S`). Cells of area π r² tile the cone with centres d ≈ 1.9 r apart (hexagonal), so
/// σ = r gives σ/d ≈ 0.53: the S gaussian streams sum to a uniform cone (ripple
/// 2·exp(−2π²σ²/d²) < 1 %). At 0.5 r (σ/d ≈ 0.26, ripple ~50 %) old dust split into one island per
/// stream, thousands of km apart (`detached.rdc`).
pub const STREAM_SIGMA_CELLS: f32 = 1.0;
/// part of a time stratum drawn per cluster (the rest is shared by the streams of the sample)
pub const STREAM_TIME_JITTER: f32 = 0.1;
/// golden-ratio step of the stream azimuths (Roberts 2018, R1 sequence): any prefix of the
/// streams is evenly spread around the cone
pub const STREAK_PHI_INV: f32 = 0.618_034;
/// Sign bit of the β half-spread field (`DustCluster::misc.w` → `DustRenderCluster`): the stream
/// is interrupted before this cluster (the jet site was dark, or the previous window emitted
/// nothing), so it draws no streak towards its predecessor. The child-pattern id lives in the low
/// bits, the half-spread itself is positive.
pub const STREAM_BREAK_BIT: u32 = 1 << 31;
/// a dark gap shorter than this fraction of a rotation does not interrupt a stream
pub const STREAM_BREAK_TURNS: f32 = 0.01;
/// nor does one shorter than this fraction of the interval since the stream's previous sample:
/// an old tier's samples are days apart and span several day/night cycles, which the capsule
/// between them averages (a break there left the trail as isolated polylines: the AU-scale
/// lattice of the zoom series, ripple 0.17–0.26)
pub const STREAM_BREAK_FRACTION: f32 = 0.75;
/// [`DustBatch::mass_params`] `w` = `shift + BATCH_BREAK_FLAG · break + BATCH_PROVISIONAL_FLAG ·
/// provisional` ([`batch_word`]): the stream shift of the batch, whether the previous window of
/// the tier is missing from the ring, and whether it is the provisional preview of the open window
pub const BATCH_BREAK_FLAG: u32 = 16;
/// The provisional batch (open window, re-emitted every tick) pins its last time sample at the
/// window end, the tick time: every stream then starts at the jet (the fountain's base,
/// `comet_mode_full.rdc`); at night `lit_time_map` maps it to the last lit instant.
pub const BATCH_PROVISIONAL_FLAG: u32 = 32;
/// `DustRenderCluster::age_id_dbeta_flux.y` bits: the ring slot (low 22 bits), the tier's stream
/// shift at [`RENDER_SHIFT_BIT0`] (4 bits) and [`RENDER_LIVE_BIT`] (evaluated in the age band;
/// culled records have only the slot). The LOD never rewrites this word, so a cluster reads its
/// predecessor's validity race free.
pub const RENDER_SLOT_MASK: u32 = (1 << RENDER_SLOT_BITS) - 1;
pub const RENDER_SHIFT_BIT0: u32 = 27;
pub const RENDER_LIVE_BIT: u32 = 1 << 31;
/// tiers per system with an LOD header (see [`dust_tier_count`])
pub const LOD_MAX_TIERS: u32 = 4;
/// percentile of the non-empty tiles taken as the white point (calibrated offline on
/// blobber.rdc from 0.3× to 3000× zoom: within 3× of the pixel 99.5th percentile)
pub const WHITE_TILE_PERCENTILE: f32 = 0.99;
/// fraction of the ring targeted in steady state
pub const BUDGET_SAFETY: f64 = 0.8;

/// Default asinh softening of the dust display stretch, relative to the white point (the measured
/// brightest dust of the view, [`white_point_from_tiles`]): dust 100× fainter than the brightest
/// shows at ~30 % opacity.
pub const DUST_SOFTENING_DEFAULT: f32 = 1e-2;
/// Accepted softening range ("dust visibility" slider); 1 is close to linear.
pub const DUST_SOFTENING_MIN: f32 = 1e-5;
pub const DUST_SOFTENING_MAX: f32 = 1.0;

/// Mirror of `composite.frag`'s display stretch: opacity of a dust optical depth `tau` relative to
/// the white point (the draw exposure divides by it), black point `black`, softening `s`:
/// `min(asinh(max(τ − b, 0)/s) / asinh(1/s), 1)`. Linear below `s`, logarithmic above, 1 at the
/// white point: dust is accumulated linearly (HDR), and this maps the 1e4 dynamic range between
/// coma and tail onto the display. `s ≤ 0`: linear.
pub fn display_stretch(tau: f32, black: f32, s: f32) -> f32 {
  let tau = tau - black.max(0.0);
  if !(tau > 0.0) {
    return 0.0;
  }
  if !(s > 0.0) {
    return tau.min(1.0);
  }
  ((tau / s).asinh() / (1.0 / s).asinh()).min(1.0)
}

/// Clamps a softening to [`DUST_SOFTENING_MIN`, `DUST_SOFTENING_MAX`] (default if not finite).
pub fn clamp_dust_softening(s: f32) -> f32 {
  if s.is_finite() {
    s.clamp(DUST_SOFTENING_MIN, DUST_SOFTENING_MAX)
  } else {
    DUST_SOFTENING_DEFAULT
  }
}

/// age of the dust column that defines the exposure reference ([`DustEmitConfig::tau_ref`])
pub const TAU_REF_AGE_S: f64 = 86400.0;

// ─── View aids: tracers and flow (`first_particles.rdc` / `second_particles.rdc`) ───
// In the wide view (25 km/px) dust moves 2–150 m/s relative to the nucleus: 63 s of sim time moved
// every particle by 0.005 px, so the coma looked like one shape sliding with the comet. And ~1 M
// dots blend into a fog in which no flow shows even when fast.

/// [`DUST_VIEW_TRACERS`]: one cluster in `TRACER_EVERY` is also drawn as a bright dot at its exact
/// position (real particles to follow); [`DUST_VIEW_FLOW`]: synchrone marks ([`flow_factor`])
pub const DUST_VIEW_TRACERS: u32 = 1;
pub const DUST_VIEW_FLOW: u32 = 2;
/// diagnostics (`AETHERVK_DUST_DEBUG_STREAM=<s>`): the splat draws one stream of every tier; the
/// value sits in bits 20..27 of the flags
pub const DUST_VIEW_DEBUG_STREAM: u32 = 1 << 16;
/// diagnostics (`AETHERVK_DUST_DEBUG_SAMPLE_MOD=<m>`): one time sample in every `m`
pub const DUST_VIEW_DEBUG_SAMPLE: u32 = 1 << 17;
/// View aids by default: none. The flow pulses (`DUST_VIEW_FLOW`, brightness marks on synchrones,
/// peak 4× / trough 0.15×) were on by default until 2026-10-10: from 0.03 AU out their marks
/// are the synchrone fan itself, a set of hard rays through the nucleus that the Monte-Carlo
/// truth of the observer (`--mc`) does not have — the plain optical depth matches it to 7 % at
/// 0.03 AU and 12 % at 1 AU. They are a Model-tab toggle now ("Flow pulses"), like the tracers;
/// `AETHERVK_DUST_TRACERS=1` / `AETHERVK_DUST_FLOW=1` turn one on for a headless run.
pub fn dust_view_flags_default() -> u32 {
  let is = |k: &str, v: &str| aethervk_oshal_rlib::os::env::var(k).is_some_and(|s| s.trim() == v);
  let mut f = 0;
  if is("AETHERVK_DUST_TRACERS", "1") {
    f |= DUST_VIEW_TRACERS;
  }
  if is("AETHERVK_DUST_FLOW", "1") {
    f |= DUST_VIEW_FLOW;
  }
  f
}
/// one tracer per this many clusters (~400 in a wide view of the coma and tail)
pub const TRACER_EVERY: u32 = 256;
/// tracer dot radius (pixels)
pub const TRACER_PX: f32 = 2.0;
/// tracer peak in white-point units (the accumulation is relative to the view's white point)
pub const TRACER_LEVEL: f32 = 0.5;

// Flow: brightness marks on synchrones (the dust of one emission instant `t_e = t_sim − age`,
// Finson & Probstein 1968): Lagrangian timelines ([`flow_factor`]), marks at fixed emission epochs
// that ride the real particles (flow-visualization timelines), brightness neutral (mean 1) and
// continuous in time and age. They are evaluated on the flow clock `T = t_sim + ∫(K − 1)·dt_sim`
// ([`DustFlowClock`]): at the time-lapse factor `K` = 1 the marks move with the dust; at `K` > 1
// every crest moves at `K ×` its parcel's real speed, so the swarm keeps its true velocity field
// (direction and relative speeds) at a readable pace. `T` only advances with sim time, so paused
// frames are identical, and it is an integral, so changing `K` never jumps the marks.

/// pulse shape: modulated share `D`, crest sharpness κ (von Mises). Tuned offline on
/// first_particles.rdc: a ±60 % cosine washed out (each pixel of the tail mixes dust of many ages,
/// the asinh stretch compresses it); this keeps ±20–30 % even averaged over whole annuli.
pub const FLOW_SHARE: f32 = 0.85;
pub const FLOW_KAPPA: f32 = 4.0;
/// `I₀(κ)`, the von Mises normalization (mean of `exp(κ cos φ)` over a turn)
pub const FLOW_I0_KAPPA: f32 = 11.301_922;
/// time-lapse factor `K` of the flow marks ([`DustFlowClock::speed`]): 1 = the marks move with the
/// dust; the UI slider spans `[FLOW_SPEED_MIN, FLOW_SPEED_MAX]` (log scale)
pub const FLOW_SPEED_DEFAULT: f64 = 1.0;
pub const FLOW_SPEED_MIN: f64 = 1.0;
pub const FLOW_SPEED_MAX: f64 = 10_000.0;

/// Whether the cluster with child-pattern id `id` carries a tracer (stable: the id comes from the
/// emission record). Mirror of `dust_is_tracer`.
#[inline]
pub fn is_tracer(id: u32) -> bool {
  pcg((id & CHILD_ID_MASK) ^ 0x7AC3_12E5) % TRACER_EVERY == 0
}

/// Flow clock of the render thread: `T = t_sim + offset`, `offset = ∫(K − 1)·dt_sim` over the
/// played sim time. The marks are functions of `T − age`: at `K` = 1 they ride the particles, at
/// `K` > 1 they move `K ×` faster than the dust, in its direction. Advances only with sim time
/// (paused: identical frames, resuming continues from the same phase; backwards sim moves the
/// marks inward); a change of `K` changes the pace, never the phase (the offset is continuous).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustFlowClock {
  /// time-lapse factor `K` ≥ 1 ([`FLOW_SPEED_DEFAULT`]; `SimulationContext::set_dust_flow_speed`)
  pub speed: f64,
  /// `T − t_sim` (s)
  pub offset_s: f64,
  /// `(wall µs, sim µs)` of the last [`Self::sync`]
  last_us: Option<(i64, i64)>,
}

impl Default for DustFlowClock {
  fn default() -> Self {
    Self {
      speed: FLOW_SPEED_DEFAULT,
      offset_s: 0.0,
      last_us: None,
    }
  }
}

impl DustFlowClock {
  /// Advances to the frame stamped `wall_us` (unscaled) / `sim_us` (scaled). Several viewports
  /// rendering the same frame (same wall stamp) advance it once.
  pub fn sync(&mut self, wall_us: i64, sim_us: i64) {
    if let Some((w, s)) = self.last_us {
      if wall_us <= w {
        return;
      }
      self.advance((sim_us - s) as f64 * 1e-6);
    }
    self.last_us = Some((wall_us, sim_us));
  }

  /// One rendered frame with `sim_dt_s` of (scaled) sim time elapsed: the offset grows by
  /// `(K − 1)·sim_dt`. Paused (no sim time): holds.
  pub fn advance(&mut self, sim_dt_s: f64) {
    if !(sim_dt_s != 0.0) || !sim_dt_s.is_finite() {
      return;
    }
    self.offset_s += (self.speed - 1.0) * sim_dt_s;
  }

  /// sets `K`, clamped to `[FLOW_SPEED_MIN, FLOW_SPEED_MAX]` (NaN → default); the phase is kept
  pub fn set_speed(&mut self, k: f64) -> f64 {
    self.speed = if k.is_finite() {
      k.clamp(FLOW_SPEED_MIN, FLOW_SPEED_MAX)
    } else {
      FLOW_SPEED_DEFAULT
    };
    self.speed
  }

  /// the flow clock `T` at sim time `t_sim_s`
  pub fn time(&self, t_sim_s: f64) -> f64 {
    t_sim_s + self.offset_s
  }
}

/// What `dust_splat.comp` needs for the flow (pyramid header words `PYR_T_HI` / `PYR_T_LO`): the
/// flow clock `T` as a df64-style hi / lo pair (exact emission epochs at 10⁸ s) and `K` (diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DustFlowUniform {
  pub t_hi: f32,
  pub t_lo: f32,
  pub speed: f32,
}

impl DustFlowUniform {
  pub fn new(clock: &DustFlowClock, t_sim_s: f64) -> Self {
    let t = Df::from_f64(clock.time(t_sim_s));
    Self {
      t_hi: t.hi,
      t_lo: t.lo,
      speed: clock.speed as f32,
    }
  }
}

/// von Mises pulse of phase `x` (turns): `(1 − D) + D·exp(κ cos 2πx)/I₀(κ)`, mean 1 over a turn,
/// peak ≈ 4×, trough ≈ 0.15×
#[inline]
fn flow_pulse_of(x: f32) -> f32 {
  let x = x - <f32 as FloatLike>::floor(x);
  let c = <f32 as FloatLike>::cos(2.0 * core::f32::consts::PI * x);
  (1.0 - FLOW_SHARE) + FLOW_SHARE * <f32 as FloatLike>::exp(FLOW_KAPPA * c) / FLOW_I0_KAPPA
}

/// `fract(t_e / P)` of the emission epoch `t_e = T − age` for `P = 2^j` s, exact at any time:
/// `T / P` scales the hi / lo halves by a power of two (exact), each reduced mod 1 before the sum.
#[inline]
fn epoch_phase(t_hi: f32, t_lo: f32, age: f32, j: i32) -> f32 {
  let inv = f32::from_bits(((127 - j.clamp(-126, 127)) as u32) << 23);
  let fr = |v: f32| v - <f32 as FloatLike>::floor(v);
  fr(fr(t_hi * inv) + fr(t_lo * inv) - age * inv)
}

/// Flow brightness factor of a dot of age `age_s`: Lagrangian timelines, marks at the emission
/// epochs `t_e = T − age ≡ 0 mod 2^j` s, `j = ⌊log₂ age⌋` and `j + 1` weighted by `fract(log₂ age)`
/// (crest spacing ≈ the age, ~3 per decade; level `j + 1` marks are every second level-`j` mark,
/// so as the dust ages alternate crests fade out). A function of the emission instant (and slowly
/// of the age): the marks ride the real particles, `K ×` faster on the flow clock. Mean 1. Mirror
/// of `dust_flow_factor`.
pub fn flow_factor(age_s: f32, flow: &DustFlowUniform) -> f32 {
  let a = age_s.max(1.0);
  let l = <f32 as FloatLike>::ln(a) * core::f32::consts::LOG2_E;
  let j = <f32 as FloatLike>::floor(l);
  let w = l - j;
  let j = j as i32;
  (1.0 - w) * flow_pulse_of(epoch_phase(flow.t_hi, flow.t_lo, a, j))
    + w * flow_pulse_of(epoch_phase(flow.t_hi, flow.t_lo, a, j + 1))
}

/// fraction of the log-distance to the measured white point covered per frame (~0.3 s at 60 Hz)
pub const WHITE_ADAPT_RATE: f32 = 0.5;
/// Tile fixed-point unit relative to the view's largest cluster optical depth of the previous
/// measured frame ([`LOD_HEADER_TAU_MAX`], `DustLodHost::tile_unit`). Relative to the *white
/// point* it locked up: a view fainter than the unit measured as empty, the white point never
/// came down (`not_flow.rdc`: a 3 AU view kept a close-up coma's white, the dust sat at 3e-5 of
/// it). A tile sums at most (clusters in it)·τ_max, so 1e-4 keeps up to 4e5 such clusters per
/// tile below 2³²; the 99th-percentile tile is ≥ 1e-3·τ_max in every capture seen.
pub const WHITE_TILE_UNIT_REL: f32 = 1e-4;

/// Eye-adaptation step of the white point, in log space (exposure changes by a constant factor
/// per frame, whatever the magnitude). Non-finite or non-positive measurements keep `prev`.
pub fn adapt_white(prev: f32, measured: f32) -> f32 {
  if !(measured > 0.0) || !measured.is_finite() {
    return prev;
  }
  if !(prev > 0.0) || !prev.is_finite() {
    return measured;
  }
  (prev.ln() + (measured.ln() - prev.ln()) * WHITE_ADAPT_RATE).exp()
}

/// `AETHERVK_DUST_EXPOSURE=fixed`: no view adaptation (white point 1 = the jet's `tau_ref`, no
/// black point), the dust brightness then depends on the zoom.
pub fn dust_exposure_fixed() -> bool {
  use core::sync::atomic::{AtomicU8, Ordering};
  static FIXED: AtomicU8 = AtomicU8::new(2);
  match FIXED.load(Ordering::Relaxed) {
    0 => false,
    1 => true,
    _ => {
      let f = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_EXPOSURE")
        .is_some_and(|s| s.trim().eq_ignore_ascii_case("fixed"));
      FIXED.store(f as u8, Ordering::Relaxed);
      f
    }
  }
}

/// Black point of the display stretch, relative to the white point (`composite.frag`): the
/// faintest 1e-4 of the dynamic range shows as nothing instead of a uniform veil.
pub const DUST_BLACK_POINT: f32 = 1e-4;

/// Stable child-pattern id of a cluster from its β half-spread field (`DustCluster::misc[3]` /
/// `DustRenderCluster::age_id_dbeta_flux[2]`, see [`CHILD_ID_BITS`]).
#[inline]
pub fn child_id(dbeta_field: f32) -> u32 {
  dbeta_field.to_bits() & CHILD_ID_MASK
}

/// `dbeta` with its low [`CHILD_ID_BITS`] mantissa bits replaced by `id` (`dust_emit.comp`)
#[inline]
pub fn pack_child_id(dbeta: f32, id: u32) -> f32 {
  f32::from_bits((dbeta.to_bits() & !CHILD_ID_MASK) | (id & CHILD_ID_MASK))
}

/// Sun gravitational parameter (m³/s²), f64 reference value of [`consts::SUN_MU`]
pub const SUN_MU_M3_S2: f64 = 1.32712440018e20;
/// astronomical unit (m)
pub const AU_M: f64 = 149_597_870_700.0;
/// grain radius range is `[r/f, r·f]` around the configured radius
pub const SIZE_RANGE_FACTOR: f64 = 10.0;
/// differential size distribution exponent `n(s) ∝ s^-q`
pub const SIZE_POWER_Q: f64 = 3.5;
/// relative child velocity dispersion (fraction of the ejection speed), lower bound
pub const CHILD_SIGMA_V_REL: f32 = 0.05;

/// Ring capacity for a device tier. `AETHERVK_DUST_RING` (power of two) overrides it.
pub fn ring_capacity(high_end: bool, env_override: Option<u32>) -> u32 {
  match env_override {
    Some(c) if c.is_power_of_two() && (1024..=(1 << 22)).contains(&c) => c,
    _ => {
      if high_end {
        RING_CAPACITY_HIGH
      } else {
        RING_CAPACITY_LOW
      }
    }
  }
}

pub type V3 = [f64; 3];

#[inline]
fn add(a: V3, b: V3) -> V3 {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
#[cfg_attr(not(test), allow(dead_code))]
fn sub(a: V3, b: V3) -> V3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn scale(a: V3, s: f64) -> V3 {
  [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn norm(a: V3) -> f64 {
  <f64 as FloatLike>::sqrt(dot(a, a))
}
#[inline]
fn absf(x: f32) -> f32 {
  if x < 0.0 { -x } else { x }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kepler propagation (universal variables, Danby §6.9 / Laguerre–Conway)
// ─────────────────────────────────────────────────────────────────────────────
pub mod kepler {
  use super::*;

  /// maximum f32 Laguerre iterations (phase 1)
  pub const MAX_ITERS_F32: u32 = 32;
  /// maximum df64 Laguerre iterations (phase 2, polishing)
  pub const MAX_ITERS_DF: u32 = 4;
  /// phase 1 stops at `|Δs| ≤ TOL_F32·|s|` (2 ulp)
  pub const TOL_F32: f32 = 2.4e-7;
  /// phase 2 stops at `|Δs| ≤ TOL_DF·|s|` (2⁻⁴⁶)
  pub const TOL_DF: f32 = 1.42e-14;

  /// Stumpff functions `c0..c3(x)`, `c_k(x) = Σ (−x)^j / (2j+k)!`, f32 (phase 1 only).
  ///
  /// Only `+ − × /` are used: the argument is reduced by 4 until `|x| ≤ 0.1`, evaluated by a
  /// truncated series, then expanded back with the duplication formulas
  /// `c0(4x)=2c0²−1, c1(4x)=c0·c1, c2(4x)=c1²/2, c3(4x)=(c3+c1·c2)/4`.
  pub fn stumpff_f32(x: f32) -> [f32; 4] {
    let f = &consts::INV_FACT;
    let mut xr = x;
    let mut n = 0u32;
    while absf(xr) > 0.1 && n < 64 {
      xr *= 0.25;
      n += 1;
    }
    let c2 = f[2].hi - xr * (f[4].hi - xr * (f[6].hi - xr * (f[8].hi - xr * f[10].hi)));
    let c3 = f[3].hi - xr * (f[5].hi - xr * (f[7].hi - xr * (f[9].hi - xr * f[11].hi)));
    let mut c = [1.0 - xr * c2, 1.0 - xr * c3, c2, c3];
    for _ in 0..n {
      c = [
        2.0 * c[0] * c[0] - 1.0,
        c[0] * c[1],
        0.5 * c[1] * c[1],
        0.25 * (c[3] + c[1] * c[2]),
      ];
    }
    c
  }

  /// df64 Stumpff functions, same algorithm with the series up to `x⁶` (`|x| ≤ 0.1` → < 1e-17).
  pub fn stumpff_df(x: Df) -> [Df; 4] {
    let f = &consts::INV_FACT;
    let mut xr = x;
    let mut n = 0u32;
    while absf(xr.hi) > 0.1 && n < 64 {
      xr = xr.scale_pow2(0.25);
      n += 1;
    }
    let c2 = f[2].sub(xr.mul(f[4].sub(
      xr.mul(f[6].sub(xr.mul(f[8].sub(xr.mul(f[10].sub(xr.mul(f[12].sub(xr.mul(f[14]))))))))),
    )));
    let c3 = f[3].sub(xr.mul(f[5].sub(
      xr.mul(f[7].sub(xr.mul(f[9].sub(xr.mul(f[11].sub(xr.mul(f[13].sub(xr.mul(f[15]))))))))),
    )));
    let mut c = [Df::ONE.sub(xr.mul(c2)), Df::ONE.sub(xr.mul(c3)), c2, c3];
    for _ in 0..n {
      c = [
        c[0].mul(c[0]).scale_pow2(2.0).add_f(-1.0),
        c[0].mul(c[1]),
        c[1].mul(c[1]).scale_pow2(0.5),
        c[3].add(c[1].mul(c[2])).scale_pow2(0.25),
      ];
    }
    c
  }

  /// Laguerre–Conway step (n = 5) on Kepler's equation `F(s) = r0 s + η s² c2 + ζ s³ c3 − t`
  #[inline]
  fn laguerre_step(f: f32, fp: f32, fpp: f32) -> Option<f32> {
    let disc = 16.0 * fp * fp - 20.0 * f * fpp;
    let disc = <f32 as FloatLike>::sqrt(absf(disc));
    let denom = if fp >= 0.0 { fp + disc } else { fp - disc };
    if denom == 0.0 {
      None
    } else {
      Some(5.0 * f / denom)
    }
  }

  /// Propagates a two-body state by `dt` seconds under gravitational parameter `mu` (m³/s²).
  /// Valid for any orbit type and any sign of `mu` (`mu ≤ 0`: radiation pressure dominates,
  /// repulsive / straight-line motion). Returns `(r, v)`. Mirrors `dust_kepler` in GLSL.
  pub fn propagate(r0: &Df3, v0: &Df3, mu: Df, dt: Df) -> (Df3, Df3) {
    let r0n = r0.dot(r0).sqrt();
    if !(r0n.hi > 0.0) || dt.hi == 0.0 {
      return (*r0, *v0);
    }
    let v2 = v0.dot(v0);
    let eta = r0.dot(v0);
    let alpha = mu.scale_pow2(2.0).div(r0n).sub(v2); // = mu / a
    let zeta = mu.sub(alpha.mul(r0n));

    // bound elliptic arguments: remove whole periods (exact periodicity)
    let mut t = dt;
    if mu.hi > 0.0 && alpha.hi > 0.0 {
      let period = consts::TWO_PI.mul(mu).div(alpha.mul(alpha.sqrt()));
      let k = t.div(period);
      let k = if k.hi >= 0.0 {
        k.floor()
      } else {
        k.neg().floor().neg()
      };
      t = t.sub(k.mul(period));
    }

    // phase 1: f32 Laguerre from s = t / r0
    let (r0f, etaf, zetaf, alphaf) = (r0n.hi, eta.hi, zeta.hi, alpha.hi);
    let tf = t.hi + t.lo;
    let mut s = tf / r0f;
    for _ in 0..MAX_ITERS_F32 {
      let c = stumpff_f32(alphaf * s * s);
      let s2 = s * s;
      let f = r0f * s + etaf * s2 * c[2] + zetaf * s2 * s * c[3] - tf;
      let fp = r0f + etaf * s * c[1] + zetaf * s2 * c[2];
      let fpp = etaf * c[0] + zetaf * s * c[1];
      let Some(ds) = laguerre_step(f, fp, fpp) else {
        break;
      };
      s -= ds;
      if !(absf(ds) > TOL_F32 * absf(s)) {
        break;
      }
    }

    // phase 2: df64 residual, f32 correction (|Δs| is already ~1e-7 |s|)
    let mut sd = Df::from_f32(s);
    for _ in 0..MAX_ITERS_DF {
      let c = stumpff_df(alpha.mul(sd).mul(sd));
      let s2 = sd.mul(sd);
      let f = r0n
        .mul(sd)
        .add(eta.mul(s2).mul(c[2]))
        .add(zeta.mul(s2).mul(sd).mul(c[3]))
        .sub(t);
      let fp = r0n.add(eta.mul(sd).mul(c[1])).add(zeta.mul(s2).mul(c[2]));
      let fpp = etaf * c[0].hi + zetaf * sd.hi * c[1].hi;
      let Some(ds) = laguerre_step(f.hi, fp.hi, fpp) else {
        break;
      };
      sd = sd.add_f(-ds);
      if !(absf(ds) > TOL_DF * absf(sd.hi)) {
        break;
      }
    }

    let c = stumpff_df(alpha.mul(sd).mul(sd));
    let s2 = sd.mul(sd);
    let g1 = sd.mul(c[1]);
    let g2 = s2.mul(c[2]);
    let g3 = s2.mul(sd).mul(c[3]);
    let r = r0n.add(eta.mul(g1)).add(zeta.mul(g2));
    let f = Df::ONE.sub(mu.mul(g2).div(r0n));
    let g = t.sub(mu.mul(g3));
    let fd = mu.mul(g1).div(r.mul(r0n)).neg();
    let gd = Df::ONE.sub(mu.mul(g2).div(r));
    (
      r0.scale(f).add(&v0.scale(g)),
      r0.scale(fd).add(&v0.scale(gd)),
    )
  }

  /// f64 Stumpff (reference / host side)
  pub fn stumpff_f64(x: f64) -> [f64; 4] {
    let mut xr = x;
    let mut n = 0u32;
    while (if xr < 0.0 { -xr } else { xr }) > 0.1 && n < 64 {
      xr *= 0.25;
      n += 1;
    }
    let c2 = 1.0 / 2.0
      - xr
        * (1.0 / 24.0
          - xr
            * (1.0 / 720.0
              - xr
                * (1.0 / 40320.0
                  - xr
                    * (1.0 / 3628800.0 - xr * (1.0 / 479001600.0 - xr * (1.0 / 87178291200.0))))));
    let c3 = 1.0 / 6.0
      - xr
        * (1.0 / 120.0
          - xr
            * (1.0 / 5040.0
              - xr
                * (1.0 / 362880.0
                  - xr
                    * (1.0 / 39916800.0
                      - xr * (1.0 / 6227020800.0 - xr * (1.0 / 1307674368000.0))))));
    let mut c = [1.0 - xr * c2, 1.0 - xr * c3, c2, c3];
    for _ in 0..n {
      c = [
        2.0 * c[0] * c[0] - 1.0,
        c[0] * c[1],
        0.5 * c[1] * c[1],
        0.25 * (c[3] + c[1] * c[2]),
      ];
    }
    c
  }

  /// f64 propagation (reference for tests, and host-side use where f64 is available)
  pub fn propagate_f64(r0: V3, v0: V3, mu: f64, dt: f64) -> (V3, V3) {
    let r0n = norm(r0);
    if !(r0n > 0.0) || dt == 0.0 {
      return (r0, v0);
    }
    let v2 = dot(v0, v0);
    let eta = dot(r0, v0);
    let alpha = 2.0 * mu / r0n - v2;
    let zeta = mu - alpha * r0n;

    let mut t = dt;
    if mu > 0.0 && alpha > 0.0 {
      let period = 2.0 * core::f64::consts::PI * mu / (alpha * <f64 as FloatLike>::sqrt(alpha));
      let k = t / period;
      let k = if k >= 0.0 {
        <f64 as FloatLike>::floor(k)
      } else {
        -<f64 as FloatLike>::floor(-k)
      };
      t -= k * period;
    }

    let mut s = t / r0n;
    for _ in 0..48 {
      let c = stumpff_f64(alpha * s * s);
      let s2 = s * s;
      let f = r0n * s + eta * s2 * c[2] + zeta * s2 * s * c[3] - t;
      let fp = r0n + eta * s * c[1] + zeta * s2 * c[2];
      let fpp = eta * c[0] + zeta * s * c[1];
      let disc = 16.0 * fp * fp - 20.0 * f * fpp;
      let disc = <f64 as FloatLike>::sqrt(if disc < 0.0 { -disc } else { disc });
      let denom = if fp >= 0.0 { fp + disc } else { fp - disc };
      if denom == 0.0 {
        break;
      }
      let ds = 5.0 * f / denom;
      s -= ds;
      if (if ds < 0.0 { -ds } else { ds }) <= 1e-15 * (if s < 0.0 { -s } else { s }) {
        break;
      }
    }

    let c = stumpff_f64(alpha * s * s);
    let s2 = s * s;
    let g1 = s * c[1];
    let g2 = s2 * c[2];
    let g3 = s2 * s * c[3];
    let r = r0n + eta * g1 + zeta * g2;
    let f = 1.0 - mu * g2 / r0n;
    let g = t - mu * g3;
    let fd = -mu * g1 / (r * r0n);
    let gd = 1.0 - mu * g2 / r;
    (
      add(scale(r0, f), scale(v0, g)),
      add(scale(r0, fd), scale(v0, gd)),
    )
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU-shared layouts (std430 via buffer_reference). Keep in sync with dust_common.glsl.
// df64 values are stored as separate `hi` / `lo` vec4s.
// ─────────────────────────────────────────────────────────────────────────────

/// Immutable emission record of one cluster. 96 bytes. A cluster is a **cell** of the emission
/// distribution: one direction cell of the jet cone (its stream), the whole speed distribution
/// (`σ_rad`), the whole size distribution (drawn as a polyline along the syndyne,
/// [`crate::scene::dust::packet_moments`]) and one time sample; its mean grain is the reference
/// size `s_ref` ([`DustBatch::vel_params`] `z`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustCluster {
  /// heliocentric position (m) at emission, `w` = emission time `t0` (scaled s), df64 high part
  pub r0_t0_hi: [f32; 4],
  /// ... low part
  pub r0_t0_lo: [f32; 4],
  /// heliocentric velocity (m/s) at emission (df64 high part) of the reference grain, `w` = β of
  /// the reference grain (the sizes span `β/SIZE_RANGE_FACTOR .. β·SIZE_RANGE_FACTOR`)
  pub v0_hi_beta: [f32; 4],
  /// velocity low part, `w` = super-particle mass (g, all sizes)
  pub v0_lo_mass: [f32; 4],
  /// `x` lateral velocity dispersion of the stream σ_lat (m/s), `y` radial (speed) dispersion
  /// σ_rad = σ_rel · v_ref (m/s), `z` mean cross-section per gram of the size distribution
  /// (m²/g), `w` half log-size range `ln(SIZE_RANGE_FACTOR)`, low [`CHILD_ID_BITS`] =
  /// child-pattern id ([`child_id`]), sign bit = [`STREAM_BREAK_BIT`]
  pub misc: [f32; 4],
  /// `xyz` ejection velocity of the reference grain `dir · v_ref` (m/s, root frame; the radial
  /// axis of the speed dispersion and the size-speed correction `v(s) = v_ref·√(s_ref/s)`),
  /// `w` reference grain radius `s_ref` (µm)
  pub eject: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustCluster>() == 96);

impl DustCluster {
  pub fn r0(&self) -> Df3 {
    Df3 {
      hi: [self.r0_t0_hi[0], self.r0_t0_hi[1], self.r0_t0_hi[2]],
      lo: [self.r0_t0_lo[0], self.r0_t0_lo[1], self.r0_t0_lo[2]],
    }
  }
  pub fn v0(&self) -> Df3 {
    Df3 {
      hi: [self.v0_hi_beta[0], self.v0_hi_beta[1], self.v0_hi_beta[2]],
      lo: [self.v0_lo_mass[0], self.v0_lo_mass[1], self.v0_lo_mass[2]],
    }
  }
  pub fn t0(&self) -> Df {
    Df::new(self.r0_t0_hi[3], self.r0_t0_lo[3])
  }
  pub fn beta(&self) -> f32 {
    self.v0_hi_beta[3]
  }
  pub fn mass_g(&self) -> f32 {
    self.v0_lo_mass[3]
  }
  /// ejection velocity of the reference grain (m/s, root frame)
  pub fn eject(&self) -> [f32; 3] {
    [self.eject[0], self.eject[1], self.eject[2]]
  }
  /// lateral velocity dispersion σ_lat (m/s)
  pub fn sigma_lat(&self) -> f32 {
    self.misc[0]
  }
  /// radial (speed) dispersion σ_rad (m/s)
  pub fn sigma_rad(&self) -> f32 {
    self.misc[1]
  }
}

/// Per-frame evaluated cluster consumed by the renderer. 32 bytes. Written compactly (index =
/// live-range offset), so the renderer needs no ring arithmetic.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustRenderCluster {
  /// particle-system local position (m), `w` lateral spread σ_lat·age (m)
  pub pos_size: [f32; 4],
  /// `x` age (s), `y` u32 bits: ring slot, stream shift, live flag ([`render_word`]), `z` child β
  /// half-spread whose low [`CHILD_ID_BITS`] carry the stable child-pattern id ([`child_id`]) and
  /// whose sign bit is [`STREAM_BREAK_BIT`], `w` flux (cross-section m², 0 = culled; per child
  /// after the LOD)
  pub age_id_dbeta_flux: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustRenderCluster>() == 32);

/// Emission batch descriptor (one per emission per system). 208 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustBatch {
  /// jet (particle system entity) heliocentric position (m) at `t_start`, `w` = `t_start` (s); hi
  pub comet_r_t_hi: [f32; 4],
  /// ... low part
  pub comet_r_t_lo: [f32; 4],
  /// heliocentric velocity (m/s) at `t_start`, `w` = batch duration (s); hi
  pub comet_v_dur_hi: [f32; 4],
  /// ... low part
  pub comet_v_dur_lo: [f32; 4],
  /// particle-system → root rotation quaternion (xyzw) at `t_start`
  pub rot_start: [f32; 4],
  /// nucleus spin: unit axis (root frame) `xyz`, `w` = angular rate ω ≥ 0 (rad/s). The attitude at
  /// `t_start + dt` is `Rot(axis, ω·dt) · rot_start` (exact for any window length, no aliasing)
  pub spin: [f32; 4],
  /// jet direction in the particle-system frame (unit), `w` cone half aperture (rad)
  pub jet_dir_aperture: [f32; 4],
  /// `x` s_min (µm), `y` s_max (µm), `z` mass exponent `4 − q`, `w` mass normalization
  /// `ln(s_max/s_min) / Z` with `Z = ∫ s^(3−q) ds`
  pub size_params: [f32; 4],
  /// `x` v_ref (m/s at s_ref), `y` relative speed std, `z` s_ref (µm), `w` β·s constant (µm)
  pub vel_params: [f32; 4],
  /// `x` batch mass (g), `y` density (g/cm³), `z` stream key ([`stream_key`], u01 with 24 bits),
  /// `w` stream shift + break flag ([`batch_streams`])
  pub mass_params: [f32; 4],
  pub first_index: u32,
  pub count: u32,
  pub ring_mask: u32,
  pub seed: u32,
  /// jet site illumination over the batch window, see [`LitWindow`]: `x` spin phase `ψ_start` at
  /// `t_start` (`[−π, π)`), `y` lit half arc `ψ0` (`(0, π)`), `z` total lit phase (rad), `w` mode
  /// ([`LIT_MODE_ALWAYS`] or [`LIT_MODE_PERIODIC`]). Emission times are spread over lit time only.
  pub lit: [f32; 4],
  /// `xyz` jet site offset from the nucleus centre (m, particle-system frame), `w` unused: the
  /// site turns with the nucleus over the window, `comet_r_t` is its position at `t_start`
  pub site_offset: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustBatch>() == 208);

impl DustBatch {
  /// stores the jet state (f64 on the host) as df64
  pub fn set_comet(&mut self, r: V3, v: V3, t_start_s: f64, dur_s: f64) {
    let (r, t, v, d) = (
      Df3::from_f64(r),
      Df::from_f64(t_start_s),
      Df3::from_f64(v),
      Df::from_f64(dur_s),
    );
    self.comet_r_t_hi = [r.hi[0], r.hi[1], r.hi[2], t.hi];
    self.comet_r_t_lo = [r.lo[0], r.lo[1], r.lo[2], t.lo];
    self.comet_v_dur_hi = [v.hi[0], v.hi[1], v.hi[2], d.hi];
    self.comet_v_dur_lo = [v.lo[0], v.lo[1], v.lo[2], d.lo];
  }
}

/// Per-frame evaluation parameters (the tail of the propagate push constants). 64 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustFrame {
  /// render anchor: a fixed heliocentric point (m), `w` = `t_now` (s); hi. Particles are evaluated
  /// relative to it (`r − A`, df64) and drawn at `A − eye` (f64, host): their drawn position is
  /// `r − eye` whatever the anchor, which only buys f32 precision near it. It is NOT the comet's
  /// current position and nothing of the comet entity follows it (see `DustDrawState::anchor_m`).
  pub ps_r_t_hi: [f32; 4],
  /// ... low part
  pub ps_r_t_lo: [f32; 4],
  /// root → particle-system rotation quaternion (xyzw), i.e. the conjugate of the entity rotation
  pub rot_inv: [f32; 4],
  /// age band: `x` max age (s, the TTL), `y` min age (s, 0 for the youngest tier), `z` 1 = no fade
  /// at the max age (an older tier takes over), `w` the tier's stream shift (written to the render
  /// clusters, [`render_word`])
  pub ttl: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustFrame>() == 64);

impl DustFrame {
  pub fn new(ps_r: V3, t_now_s: f64, rot_inv: [f32; 4], ttl_s: f32) -> Self {
    let (r, t) = (Df3::from_f64(ps_r), Df::from_f64(t_now_s));
    Self {
      ps_r_t_hi: [r.hi[0], r.hi[1], r.hi[2], t.hi],
      ps_r_t_lo: [r.lo[0], r.lo[1], r.lo[2], t.lo],
      rot_inv,
      ttl: [ttl_s, 0.0, 0.0, 0.0],
    }
  }
  /// the same anchor and band evaluated at another time (`t_now`)
  pub fn at_time(mut self, t_now_s: f64) -> Self {
    let t = Df::from_f64(t_now_s);
    self.ps_r_t_hi[3] = t.hi;
    self.ps_r_t_lo[3] = t.lo;
    self
  }
  /// the anchor (df64 hi + lo), heliocentric metres
  pub fn anchor_m(&self) -> V3 {
    [
      self.ps_r_t_hi[0] as f64 + self.ps_r_t_lo[0] as f64,
      self.ps_r_t_hi[1] as f64 + self.ps_r_t_lo[1] as f64,
      self.ps_r_t_hi[2] as f64 + self.ps_r_t_lo[2] as f64,
    ]
  }
  /// evaluation time (s)
  pub fn t_now_s(&self) -> f64 {
    self.ps_r_t_hi[3] as f64 + self.ps_r_t_lo[3] as f64
  }
  /// the tier has `2^shift` streams per time sample ([`DustHostState::stream_shift`])
  pub fn with_streams(mut self, shift: u32) -> Self {
    self.ttl[3] = shift as f32;
    self
  }
  /// draws only ages in `[min_age_s, ttl]`; `fade` at the max age only when no older tier follows
  pub fn with_band(mut self, min_age_s: f32, fade: bool) -> Self {
    self.ttl[1] = min_age_s;
    self.ttl[2] = if fade { 0.0 } else { 1.0 };
    self
  }
}

/// `dust_emit.comp` push constants. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustEmitPushConstants {
  /// `DustClusterBuffer` (ring base address)
  pub clusters: u64,
  /// `DustBatchRef`
  pub batch: u64,
}
const _: () = assert!(core::mem::size_of::<DustEmitPushConstants>() == 16);

/// `dust_propagate.comp` push constants. 112 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustPropagatePushConstants {
  pub clusters: u64,
  pub render: u64,
  /// `DustMomentsBuffer` of the tier (the packets' second moments, `dust_splat.comp`)
  pub moments: u64,
  pub first_slot: u32,
  pub live_count: u32,
  pub ring_mask: u32,
  pub _pad0: u32,
  /// std430 pads the `vec4` frame to a 16-byte boundary
  pub _pad1: [u32; 2],
  pub frame: DustFrame,
}
const _: () = assert!(core::mem::size_of::<DustPropagatePushConstants>() == 112);

// ─────────────────────────────────────────────────────────────────────────────
// Streaklines. Mirror of `dust_common.glsl` `dust_streak_*`
// ─────────────────────────────────────────────────────────────────────────────

/// `DustRenderCluster::age_id_dbeta_flux.y` of a live cluster (see [`RENDER_SLOT_MASK`])
#[inline]
pub fn render_word(slot: u32, shift: u32) -> u32 {
  (slot & RENDER_SLOT_MASK) | ((shift & 0xF) << RENDER_SHIFT_BIT0) | RENDER_LIVE_BIT
}
/// ring slot of a render cluster's `y` word
#[inline]
pub fn render_slot(y: f32) -> u32 {
  y.to_bits() & RENDER_SLOT_MASK
}
/// evaluated in the tier's age band (not culled by the propagate)
#[inline]
pub fn render_live(y: f32) -> bool {
  y.to_bits() & RENDER_LIVE_BIT != 0
}
/// stream shift of the tier (`S = 2^shift` streams)
#[inline]
pub fn render_stream_shift(y: f32) -> u32 {
  (y.to_bits() >> RENDER_SHIFT_BIT0) & 0xF
}
/// the stream is interrupted before this cluster ([`STREAM_BREAK_BIT`] of the β half-spread field)
#[inline]
pub fn stream_break(dbeta_field: f32) -> bool {
  dbeta_field.to_bits() & STREAM_BREAK_BIT != 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic RNG and sampling (bit-identical u32 ops on GPU)
// ─────────────────────────────────────────────────────────────────────────────
#[inline]
pub fn pcg(v: u32) -> u32 {
  let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
  let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
  (word >> 22) ^ word
}
/// uniform in [0, 1) with 24 bits (exact in f32)
#[inline]
pub fn u01(h: u32) -> f32 {
  (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[inline]
pub(crate) fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
  // t = 2 cross(q.xyz, v); v' = v + w t + cross(q.xyz, t)
  let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
  let t = [
    2.0 * (qy * v[2] - qz * v[1]),
    2.0 * (qz * v[0] - qx * v[2]),
    2.0 * (qx * v[1] - qy * v[0]),
  ];
  [
    v[0] + qw * t[0] + (qy * t[2] - qz * t[1]),
    v[1] + qw * t[1] + (qz * t[0] - qx * t[2]),
    v[2] + qw * t[2] + (qx * t[1] - qy * t[0]),
  ]
}

/// Hamilton product `a · b` (xyzw)
#[inline]
fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
  [
    a[3] * b[0] + a[0] * b[3] + (a[1] * b[2] - a[2] * b[1]),
    a[3] * b[1] + a[1] * b[3] + (a[2] * b[0] - a[0] * b[2]),
    a[3] * b[2] + a[2] * b[3] + (a[0] * b[1] - a[1] * b[0]),
    a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]),
  ]
}

/// rotation by `angle` (any magnitude) about unit `axis`. The half angle is reduced to
/// `[−π, π)` first: GPU `sin`/`cos` have no accuracy guarantee outside that range.
#[inline]
fn qaxis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
  const TWO_PI: f32 = 2.0 * core::f32::consts::PI;
  let h = 0.5 * angle;
  let k = <f32 as FloatLike>::floor((h + core::f32::consts::PI) * (1.0 / TWO_PI));
  let h = h - k * TWO_PI;
  let s = <f32 as FloatLike>::sin(h);
  [
    axis[0] * s,
    axis[1] * s,
    axis[2] * s,
    <f32 as FloatLike>::cos(h),
  ]
}

/// Emission offset `dt ∈ [0, dur]` (s) of the stratified sample `u ∈ [0, 1)`, uniform over the
/// *lit* time of the batch window (see [`LitWindow`]). `omega` = `spin.w`.
#[inline]
pub fn lit_time_map(u: f32, lit: [f32; 4], omega: f32, dur: f32) -> f32 {
  const TWO_PI: f32 = 2.0 * core::f32::consts::PI;
  if !(lit[3] == LIT_MODE_PERIODIC) || !(omega > 0.0) || !(lit[1] > 0.0) {
    return u * dur;
  }
  let (ps, p0) = (lit[0], lit[1]);
  let arc = 2.0 * p0;
  // lit arcs are `(−ψ0, ψ0) + 2πk`; the first lit segment at or after `ψ_start`
  let (seg_start, arc_start) = if ps < -p0 {
    (-p0, -p0)
  } else if ps < p0 {
    (ps, -p0)
  } else {
    (TWO_PI - p0, TWO_PI - p0)
  };
  let seg_len = arc_start + arc - seg_start;
  let m = u * lit[2];
  let psi = if m < seg_len {
    seg_start + m
  } else {
    let m = m - seg_len;
    let k = <f32 as FloatLike>::floor(m / arc);
    let r = m - k * arc;
    arc_start + TWO_PI + k * TWO_PI + r
  };
  ((psi - ps) / omega).clamp(0.0, dur)
}

/// uniform direction inside a cone around unit `dir` with half aperture `aperture`
fn sample_cone(u1: f32, u2: f32, dir: [f32; 3], aperture: f32) -> [f32; 3] {
  let cos_a = <f32 as FloatLike>::cos(aperture);
  let z = 1.0 + (cos_a - 1.0) * u2;
  let sin_t = <f32 as FloatLike>::sqrt((1.0 - z * z).max(0.0));
  let phi = 2.0 * core::f32::consts::PI * u1;
  let local = [
    sin_t * <f32 as FloatLike>::cos(phi),
    sin_t * <f32 as FloatLike>::sin(phi),
    z,
  ];
  let up = if absf(dir[2]) < 0.999 {
    [0.0, 0.0, 1.0]
  } else {
    [1.0, 0.0, 0.0]
  };
  let t = normalize3(cross3(up, dir));
  let b = cross3(dir, t);
  [
    t[0] * local[0] + b[0] * local[1] + dir[0] * local[2],
    t[1] * local[0] + b[1] * local[1] + dir[1] * local[2],
    t[2] * local[0] + b[2] * local[1] + dir[2] * local[2],
  ]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
#[inline]
fn normalize3(a: [f32; 3]) -> [f32; 3] {
  let n = <f32 as FloatLike>::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  if n > 0.0 {
    [a[0] / n, a[1] / n, a[2] / n]
  } else {
    a
  }
}

/// standard normal from two uniforms (Box–Muller, first output)
#[inline]
fn gauss(u1: f32, u2: f32) -> f32 {
  let r = <f32 as FloatLike>::sqrt(-2.0 * <f32 as FloatLike>::ln(u1.max(1e-7)));
  r * <f32 as FloatLike>::cos(2.0 * core::f32::consts::PI * u2)
}

/// Hashed permutation of `[0, n)` keyed by `key` (Kensler, "Correlated Multi-Jittered Sampling",
/// 2013): a bijection of the next power of two built from invertible u32 steps (odd multiplies,
/// xor-shifts restricted to the mask), cycle-walked into `[0, n)` (< 2 rounds on average).
/// Bit-identical on GPU (`dust_permute`). `n ≥ 1`.
#[inline]
pub fn permute_index(i: u32, n: u32, key: u32) -> u32 {
  let mut w = n.max(1) - 1;
  w |= w >> 1;
  w |= w >> 2;
  w |= w >> 4;
  w |= w >> 8;
  w |= w >> 16;
  let p = key;
  let mut i = i;
  loop {
    i ^= p;
    i = i.wrapping_mul(0xE170_893D);
    i ^= p >> 16;
    i ^= (i & w) >> 4;
    i ^= p >> 8;
    i = i.wrapping_mul(0x0929_EB3F);
    i ^= p >> 23;
    i ^= (i & w) >> 1;
    i = i.wrapping_mul(1 | p >> 27);
    i = i.wrapping_mul(0x6935_FA69);
    i ^= (i & w) >> 11;
    i = i.wrapping_mul(0x74DC_B303);
    i ^= (i & w) >> 2;
    i = i.wrapping_mul(0x9E50_1CC3);
    i ^= (i & w) >> 2;
    i = i.wrapping_mul(0xC860_A3DF);
    i &= w;
    i ^= i >> 5;
    if i < n.max(1) {
      break;
    }
  }
  i.wrapping_add(p) % n.max(1)
}

/// `(stream shift, break before the batch)` of a batch (`mass_params.w`, [`BATCH_BREAK_FLAG`])
#[inline]
pub fn batch_streams(batch: &DustBatch) -> (u32, bool) {
  let w = batch.mass_params[3] as u32;
  (w % BATCH_BREAK_FLAG, (w / BATCH_BREAK_FLAG) & 1 != 0)
}

/// the batch is the provisional preview of the open window ([`BATCH_PROVISIONAL_FLAG`])
#[inline]
pub fn batch_provisional(batch: &DustBatch) -> bool {
  (batch.mass_params[3] as u32 / BATCH_PROVISIONAL_FLAG) & 1 != 0
}

/// `mass_params.w` of a batch with `2^shift` streams ([`batch_streams`])
#[inline]
pub fn batch_streams_word(shift: u32, break_before: bool) -> f32 {
  (shift + if break_before { BATCH_BREAK_FLAG } else { 0 }) as f32
}

/// [`batch_streams_word`] plus the provisional flag
#[inline]
pub fn batch_word(shift: u32, break_before: bool, provisional: bool) -> f32 {
  batch_streams_word(shift, break_before)
    + (if provisional {
      BATCH_PROVISIONAL_FLAG
    } else {
      0
    }) as f32
}

/// Stream key of a jet configuration (`DustBatch::mass_params.z`): u01 with 24 bits, exact in f32,
/// so the shader recovers the same u32. It fixes every stream's direction, size rank and speed,
/// independent of the window and the tier.
pub fn stream_key(cfg_seed: u32) -> f32 {
  u01(pcg(cfg_seed ^ 0xA3C5_9AC3))
}

#[inline]
fn stream_key_bits(batch: &DustBatch) -> u32 {
  (batch.mass_params[2] * 16_777_216.0) as u32
}

/// Emission-time quantile `u ∈ [0, 1)` of cluster `j = i·S + s`: time sample `i` of the batch's
/// `count / S` strata, a jitter shared by the `S` streams of the sample plus
/// [`STREAM_TIME_JITTER`] of the stratum per cluster. The last sample of a provisional batch is at
/// `u = 1`: the window end, now ([`BATCH_PROVISIONAL_FLAG`]).
pub fn stream_time_u01(batch: &DustBatch, j: u32) -> f32 {
  let shift = batch_streams(batch).0;
  let samples = (batch.count >> shift).max(1);
  let i = j >> shift;
  // the last sample of every window sits at the window end (the provisional one: now), so a
  // window's samples tile its lit time and the first sample of the next window continues from
  // it: the capsules cover the time axis end to end, with the masses of their intervals
  // ([`emit_cluster`]); randomly placed samples with equal masses were a ±50 % density ripple
  // along every stream
  if i + 1 == samples {
    return 1.0;
  }
  let hs = pcg(batch.seed ^ pcg(i ^ 0x3C6E_F372));
  let hc = pcg(batch.seed ^ pcg(j));
  (i as f32 + (1.0 - STREAM_TIME_JITTER) * u01(hs) + STREAM_TIME_JITTER * u01(hc)) / samples as f32
}

/// Emission offset (s from the window start) of the time quantile `u`: lit time only, dark
/// phases of the jet site get no clusters (full df64 when always lit)
fn emission_offset(batch: &DustBatch, u: f32) -> Df {
  let dur = Df::new(batch.comet_v_dur_hi[3], batch.comet_v_dur_lo[3]);
  if batch.lit[3] == LIT_MODE_PERIODIC {
    Df::from_f32(lit_time_map(u, batch.lit, batch.spin[3], dur.hi + dur.lo))
  } else {
    dur.mul_f(u)
  }
}

/// Whether the stream of cluster `j` is interrupted before it ([`STREAM_BREAK_BIT`]): the batch
/// follows a missing window (first time sample), or the jet site was dark in between
/// ([`stream_dark_before`]).
pub fn stream_breaks_before(batch: &DustBatch, j: u32) -> bool {
  let (shift, break_before) = batch_streams(batch);
  (j >> shift == 0 && break_before) || stream_dark_before(batch, j)
}

/// Whether the jet site spent more than [`STREAM_BREAK_TURNS`] of a rotation in the dark since the
/// previous sample of cluster `j`'s stream (the window start for the first one): the spin phase
/// elapsed minus the lit phase the samples consumed.
pub fn stream_dark_before(batch: &DustBatch, j: u32) -> bool {
  let shift = batch_streams(batch).0;
  let i = j >> shift;
  let omega = batch.spin[3];
  if !(batch.lit[3] == LIT_MODE_PERIODIC) || !(omega > 0.0) {
    return false;
  }
  let dur = batch.comet_v_dur_hi[3] + batch.comet_v_dur_lo[3];
  let u = stream_time_u01(batch, j);
  let (u_prev, dt_prev) = if i == 0 {
    (0.0, 0.0)
  } else {
    let up = stream_time_u01(batch, j - (1 << shift));
    (up, lit_time_map(up, batch.lit, omega, dur))
  };
  let dt = lit_time_map(u, batch.lit, omega, dur);
  let elapsed = omega * (dt - dt_prev);
  let dark = elapsed - (u - u_prev) * batch.lit[2];
  dark > 2.0 * core::f32::consts::PI * STREAM_BREAK_TURNS && dark > STREAM_BREAK_FRACTION * elapsed
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission (mirrors dust_emit.comp)
// ─────────────────────────────────────────────────────────────────────────────

/// Builds cluster `j` (`0 ≤ j < batch.count`) of `batch`; it goes to ring slot
/// `(batch.first_index + j) & batch.ring_mask`. Cluster `j = i·S + s` is time sample `i` of stream
/// `s` ([`DUST_STREAMS`]): the stream fixes the direction in the jet cone (a Fibonacci lattice over
/// its solid angle, jittered per cluster). The cluster is the whole **cell**: the mean ejection
/// speed `v_ref` of the reference grain with the full speed spread as its radial dispersion
/// (`misc.y`), the full size distribution (the reference grain's β in `v0_hi_beta.w`, the sizes
/// drawn as a polyline along the syndyne by the renderer) and `1/S` of the time sample's mass.
/// Nothing is drawn per stream or per size that the kernels do not cover: the sum over the
/// clusters is the continuous emission (`cell_kernels_reproduce_the_continuous_emission`).
pub fn emit_cluster(batch: &DustBatch, j: u32) -> DustCluster {
  // randomness from the in-batch index: the seed is unique per emission window, so a window
  // always produces the same clusters whatever ring slots it lands on (deterministic seek)
  let h0 = pcg(batch.seed ^ pcg(j));
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  let h5 = pcg(h4);

  let (shift, _) = batch_streams(batch);
  let n = 1u32 << shift;
  let s = j & (n - 1);
  let samples = (batch.count >> shift).max(1) as f32;
  let key = stream_key_bits(batch);
  let _ = (h3, h4, key);

  let t_start = Df::new(batch.comet_r_t_hi[3], batch.comet_r_t_lo[3]);
  let u_t = stream_time_u01(batch, j);
  let dt_in = emission_offset(batch, u_t);
  // the sample's share of the window's lit time: from the previous sample of its stream (the
  // window start for the first) to itself; the shares of a stream sum to 1 (the last sample is
  // at the window end), so the batch mass is exact and the density along the stream uniform
  let u_prev = if j >> shift == 0 {
    0.0
  } else {
    stream_time_u01(batch, j - n)
  };
  let share = (u_t - u_prev).max(0.0);
  let t0 = t_start.add(dt_in);

  // the window-start site position in free fall over the sub-interval (the site's turn is added
  // below)
  let rc0 = Df3 {
    hi: [
      batch.comet_r_t_hi[0],
      batch.comet_r_t_hi[1],
      batch.comet_r_t_hi[2],
    ],
    lo: [
      batch.comet_r_t_lo[0],
      batch.comet_r_t_lo[1],
      batch.comet_r_t_lo[2],
    ],
  };
  let vc0 = Df3 {
    hi: [
      batch.comet_v_dur_hi[0],
      batch.comet_v_dur_hi[1],
      batch.comet_v_dur_hi[2],
    ],
    lo: [
      batch.comet_v_dur_lo[0],
      batch.comet_v_dur_lo[1],
      batch.comet_v_dur_lo[2],
    ],
  };
  let (rc_start_site, vc) = kepler::propagate(&rc0, &vc0, consts::SUN_MU, dt_in);

  // direction: the stream's cone cell (polar stratum in solid angle, golden-ratio azimuth turned
  // by the key), jittered, rotated to root with the attitude at t0
  let jet = [
    batch.jet_dir_aperture[0],
    batch.jet_dir_aperture[1],
    batch.jet_dir_aperture[2],
  ];
  let aperture = batch.jet_dir_aperture[3];
  let u_pol = (s as f32 + 0.5 + STREAM_DIR_JITTER * (u01(h2) - 0.5)) / n as f32;
  let az =
    u01(pcg(key ^ 0x6A09_E667)) + s as f32 * STREAK_PHI_INV + STREAM_DIR_JITTER * (u01(h1) - 0.5);
  let dir_ps = sample_cone(az - <f32 as FloatLike>::floor(az), u_pol, jet, aperture);
  // exact nucleus spin from the window start
  let spin = qaxis_angle(
    [batch.spin[0], batch.spin[1], batch.spin[2]],
    batch.spin[3] * (dt_in.hi + dt_in.lo),
  );
  let rot = qmul(spin, batch.rot_start);
  let dir = qrot(rot, dir_ps);

  // the site turns with the nucleus: the comet's free fall carries the site's offset at the
  // window start, the emission is at its offset at t0, with the surface velocity ω × offset
  let off = [
    batch.site_offset[0],
    batch.site_offset[1],
    batch.site_offset[2],
  ];
  let (o_start, o_t0) = (qrot(batch.rot_start, off), qrot(rot, off));
  let rc = rc_start_site.add(&Df3::from_f32([
    o_t0[0] - o_start[0],
    o_t0[1] - o_start[1],
    o_t0[2] - o_start[2],
  ]));
  let w = [
    batch.spin[0] * batch.spin[3],
    batch.spin[1] * batch.spin[3],
    batch.spin[2] * batch.spin[3],
  ];
  let v_site = [
    w[1] * o_t0[2] - w[2] * o_t0[1],
    w[2] * o_t0[0] - w[0] * o_t0[2],
    w[0] * o_t0[1] - w[1] * o_t0[0],
  ];

  // the reference grain: the batch's s_ref (the configured radius); the cluster carries every
  // size of the distribution (β ∝ 1/s), the renderer spreads it along the syndyne
  let s_ref = batch.vel_params[2];
  let beta = batch.vel_params[3] / s_ref;

  // the stream's share of the batch mass: all sizes, 1/S of the lit-time interval it stands for
  let mass_g = batch.mass_params[0] * share / n as f32;
  let _ = samples;

  // ejection speed: the mean speed of the reference grain, no draw (the speed spread is the
  // packet's radial dispersion σ_rad below, so the S streams of a cell sum to one smooth cone of
  // every speed instead of S speed shells)
  let v_ref = batch.vel_params[0];
  let v_ej = v_ref.max(0.0);
  // lateral dispersion of the stream: STREAM_SIGMA_CELLS of its cone cell (angular radius
  // aperture/√S), at least CHILD_SIGMA_V_REL of the speed
  let sigma_lat = v_ej
    * (STREAM_SIGMA_CELLS * aperture / <f32 as FloatLike>::sqrt(n as f32)).max(CHILD_SIGMA_V_REL);
  // radial dispersion: the whole speed distribution of the jet
  let sigma_rad = v_ej * batch.vel_params[1].max(0.0);

  let eject = [dir[0] * v_ej, dir[1] * v_ej, dir[2] * v_ej];
  let v0 = vc.add(&Df3::from_f32([
    eject[0] + v_site[0],
    eject[1] + v_site[1],
    eject[2] + v_site[2],
  ]));

  // mean cross-section per gram of the size distribution: π s² / (4/3 π s³ ρ) = 3 / (4 ρ s)
  // averaged over the mass, 3/(4ρ) · ⟨1/s⟩ = 3/(4 ρ s_ref) for n(s) ∝ s^-3.5 over a log range
  // symmetric about s_ref (`size_mean_inv_s_um`)  [s in m, ρ in g/m³]
  let rho_g_m3 = batch.mass_params[1] * 1.0e6;
  let xsec_per_g = 3.0 / (4.0 * rho_g_m3 * size_mean_inv_s_um(batch.size_params) * 1.0e-6);
  // the half log-size range (the renderer's size polyline spans β/F .. β·F); the low mantissa
  // bits carry the child-pattern id, the sign bit the stream break
  let dbeta = 0.5 * <f32 as FloatLike>::ln(batch.size_params[1] / batch.size_params[0]);
  let dbeta = pack_child_id(dbeta, pcg(h0 ^ 0x2C1B_3C6D));
  let dbeta = if stream_breaks_before(batch, j) {
    f32::from_bits(dbeta.to_bits() | STREAM_BREAK_BIT)
  } else {
    dbeta
  };
  let _ = h5;

  DustCluster {
    r0_t0_hi: [rc.hi[0], rc.hi[1], rc.hi[2], t0.hi],
    r0_t0_lo: [rc.lo[0], rc.lo[1], rc.lo[2], t0.lo],
    v0_hi_beta: [v0.hi[0], v0.hi[1], v0.hi[2], beta],
    v0_lo_mass: [v0.lo[0], v0.lo[1], v0.lo[2], mass_g],
    misc: [sigma_lat, sigma_rad, xsec_per_g, dbeta],
    eject: [eject[0], eject[1], eject[2], s_ref],
  }
}

/// Harmonic mean size of the distribution, `1/⟨1/s⟩` over the mass (µm): the grain whose
/// cross-section per gram is the distribution's mean. `size_params` = (s_min, s_max, e = 4 − q, ·):
/// `⟨1/s⟩ = ∫ s^(e−2) ds / ∫ s^(e−1) ds`, closed form (log cases at `e = 1`, `e = 2`).
pub fn size_mean_inv_s_um(size_params: [f32; 4]) -> f32 {
  let (a, b, e) = (
    size_params[0] as f64,
    size_params[1] as f64,
    size_params[2] as f64,
  );
  let powf = <f64 as FloatLike>::pow;
  let lnf = <f64 as FloatLike>::ln;
  let int = |k: f64| -> f64 {
    // ∫_a^b s^k ds
    if (k + 1.0).abs() < 1e-9 {
      lnf(b / a)
    } else {
      (powf(b, k + 1.0) - powf(a, k + 1.0)) / (k + 1.0)
    }
  };
  let mean_inv = int(e - 2.0) / int(e - 1.0);
  if mean_inv > 0.0 && mean_inv.is_finite() {
    (1.0 / mean_inv) as f32
  } else {
    <f32 as FloatLike>::sqrt(size_params[0] * size_params[1])
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation (mirrors dust_propagate.comp)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluates the cluster in ring slot `slot` at `frame`'s time, in the particle-system frame.
pub fn evaluate_cluster(c: &DustCluster, slot: u32, frame: &DustFrame) -> DustRenderCluster {
  let t_now = Df::new(frame.ps_r_t_hi[3], frame.ps_r_t_lo[3]);
  let age = t_now.sub(c.t0());
  let age_f = age.hi + age.lo;
  let (ttl, min_age) = (frame.ttl[0], frame.ttl[1]);
  // age band of the tier (see `DustFrame::with_band`): neighbouring tiers overlap in emission time
  if !(age_f >= min_age) || !(age_f >= 0.0) || age_f > ttl {
    return DustRenderCluster::culled(slot);
  }
  let mu = consts::SUN_MU.mul(two_sum_one_minus(c.beta()));
  let (r, _v) = kepler::propagate(&c.r0(), &c.v0(), mu, age);
  let ps = Df3 {
    hi: [frame.ps_r_t_hi[0], frame.ps_r_t_hi[1], frame.ps_r_t_hi[2]],
    lo: [frame.ps_r_t_lo[0], frame.ps_r_t_lo[1], frame.ps_r_t_lo[2]],
  };
  let local = qrot(frame.rot_inv, r.sub(&ps).to_f32());
  let size = c.misc[0] * age_f;
  // fade the last 10% of the lifetime (oldest tier only)
  let fade = if frame.ttl[2] > 0.5 {
    1.0
  } else {
    ((1.0 - age_f / ttl) * 10.0).clamp(0.0, 1.0)
  };
  let flux = c.mass_g() * c.misc[2] * fade;
  DustRenderCluster {
    pos_size: [local[0], local[1], local[2], size],
    age_id_dbeta_flux: [
      age_f,
      f32::from_bits(render_word(slot, frame.ttl[3] as u32)),
      c.misc[3],
      flux,
    ],
  }
}

/// exact `1 − β` in df64
#[inline]
pub(crate) fn two_sum_one_minus(beta: f32) -> Df {
  df::two_sum(1.0, -beta)
}

impl DustRenderCluster {
  pub fn culled(slot: u32) -> Self {
    Self {
      pos_size: [0.0; 4],
      age_id_dbeta_flux: [0.0, f32::from_bits(slot), 0.0, 0.0],
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host-side batch planning
// ─────────────────────────────────────────────────────────────────────────────

/// Grain-size distribution derived from the configured grain diameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeDistribution {
  pub s_min_um: f64,
  pub s_max_um: f64,
  pub q: f64,
}

impl SizeDistribution {
  pub fn from_diameter_um(diameter_um: f32) -> Self {
    let r = (diameter_um as f64 * 0.5).max(1e-3);
    Self {
      s_min_um: r / SIZE_RANGE_FACTOR,
      s_max_um: r * SIZE_RANGE_FACTOR,
      q: SIZE_POWER_Q,
    }
  }
  /// The distribution without grains smaller than `s_min · factor` (capped at `s_max / 2`), and the
  /// fraction of the mass that remains. Old dust tiers keep only the larger grains: the small, high
  /// β ones have long left the field, so their mass is dropped, not redistributed.
  pub fn truncated(&self, factor: f64) -> (Self, f64) {
    let s_min = (self.s_min_um * factor.max(1.0)).min(self.s_max_um * 0.5).max(self.s_min_um);
    let cut = Self {
      s_min_um: s_min,
      ..*self
    };
    let z = |d: &Self| {
      let e = 4.0 - d.q;
      if (if e < 0.0 { -e } else { e }) < 1e-9 {
        <f64 as FloatLike>::ln(d.s_max_um / d.s_min_um)
      } else {
        (<f64 as FloatLike>::pow(d.s_max_um, e) - <f64 as FloatLike>::pow(d.s_min_um, e)) / e
      }
    };
    let full = z(self);
    (
      cut,
      if full > 0.0 {
        (z(&cut) / full).clamp(0.0, 1.0)
      } else {
        1.0
      },
    )
  }
  /// `(4 − q, ln(s_max/s_min) / Z)`, `Z = ∫ s^(3−q) ds` over `[s_min, s_max]`
  pub fn mass_weight_params(&self) -> (f64, f64) {
    let e = 4.0 - self.q;
    let l = <f64 as FloatLike>::ln(self.s_max_um / self.s_min_um);
    let z = if (if e < 0.0 { -e } else { e }) < 1e-9 {
      l
    } else {
      (<f64 as FloatLike>::pow(self.s_max_um, e) - <f64 as FloatLike>::pow(self.s_min_um, e)) / e
    };
    (e, l / z)
  }
}

/// Per-system host emission state (persisted in the ECS component between ticks).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EmissionAccumulator {
  /// fractional clusters owed (budget accumulates until ≥ 1)
  pub clusters: f64,
  /// mass (g) produced since the last emitted batch
  pub mass_g: f64,
  /// scaled time (s) where the pending window starts
  pub window_start_s: f64,
}

/// Result of [`plan_batch`]: how many clusters to emit and over which time window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchPlan {
  pub count: u32,
  pub mass_g: f64,
  pub window_start_s: f64,
  pub window_dur_s: f64,
}

/// Advances the accumulator by `[acc.window_start_s .. t_now_s]` worth of production and decides
/// whether to emit. Mass is conserved exactly: everything produced since the last batch goes into
/// the next one. Production only runs while the jet site is lit (`lit_dt_s ≤ dt_s`). The cluster
/// count follows the steady-state budget `capacity · Δt / TTL`, bounded by `free_slots`.
#[allow(clippy::too_many_arguments)]
pub fn plan_batch(
  acc: &mut EmissionAccumulator,
  q_dust_kgs: f64,
  dt_s: f64,
  lit_dt_s: f64,
  t_now_s: f64,
  ttl_s: f64,
  capacity: u32,
  free_slots: u32,
) -> Option<BatchPlan> {
  let dt_s = if dt_s.is_finite() { dt_s.max(0.0) } else { 0.0 };
  let q = if q_dust_kgs.is_finite() {
    q_dust_kgs.max(0.0)
  } else {
    0.0
  };
  // first call (or time jumped backwards): the pending window starts `dt` ago
  if !(acc.window_start_s > 0.0) || acc.window_start_s > t_now_s {
    acc.window_start_s = t_now_s - dt_s;
  }
  let lit_dt_s = if lit_dt_s.is_finite() {
    lit_dt_s.clamp(0.0, dt_s)
  } else {
    0.0
  };
  acc.mass_g += q * 1e3 * lit_dt_s;
  let budget = capacity as f64 * BUDGET_SAFETY;
  let ttl = if ttl_s.is_finite() && ttl_s > 0.0 {
    ttl_s
  } else {
    1.0
  };
  acc.clusters += (budget * dt_s / ttl).min(budget);
  if !(acc.mass_g > 0.0) {
    // nothing produced: do not accumulate budget forever
    acc.clusters = acc.clusters.min(1.0);
    acc.window_start_s = t_now_s;
    return None;
  }
  let want = <f64 as FloatLike>::floor(acc.clusters);
  let count = want.min(free_slots as f64);
  if count < 1.0 {
    return None;
  }
  let plan = BatchPlan {
    count: count as u32,
    mass_g: acc.mass_g,
    window_start_s: acc.window_start_s,
    window_dur_s: (t_now_s - acc.window_start_s).max(0.0),
  };
  acc.clusters -= count;
  acc.mass_g = 0.0;
  acc.window_start_s = t_now_s;
  Some(plan)
}

/// Fills the size/velocity/mass parameters of a batch from the emission parameters.
#[allow(clippy::too_many_arguments)]
pub fn batch_params(
  dist: &SizeDistribution,
  grain_diameter_um: f32,
  density_gcm3: f32,
  beta_at_configured_size: f32,
  start_velocity_mean: f32,
  start_velocity_std: f32,
  mass_g: f64,
  stream_key: f32,
) -> ([f32; 4], [f32; 4], [f32; 4]) {
  let (e, mass_norm) = dist.mass_weight_params();
  let s_ref = (grain_diameter_um * 0.5).max(1e-3);
  let v_ref = start_velocity_mean.max(0.0);
  let v_std_rel = if v_ref > 0.0 {
    (start_velocity_std / v_ref).max(0.0)
  } else {
    0.0
  };
  // β ∝ 1/s: β(s) = β(s_ref) · s_ref / s
  let beta_s = beta_at_configured_size * s_ref;
  (
    [
      dist.s_min_um as f32,
      dist.s_max_um as f32,
      e as f32,
      mass_norm as f32,
    ],
    [v_ref, v_std_rel, s_ref, beta_s],
    [mass_g as f32, density_gcm3, stream_key, 0.0],
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Jet site illumination
// ─────────────────────────────────────────────────────────────────────────────

/// [`DustBatch::lit`] mode: the site stays lit over the whole window (emission time is uniform)
pub const LIT_MODE_ALWAYS: f32 = 0.0;
/// [`DustBatch::lit`] mode: the site goes in and out of daylight with the nucleus spin
pub const LIT_MODE_PERIODIC: f32 = 1.0;

/// Illumination of the jet site over `[0, dur]` from a window start, for a nucleus spinning
/// uniformly about a fixed axis, with the Sun direction frozen over the window (it moves < 1°/day).
///
/// With `n(t) = Rot(a, ωt)·n₀`, `ŝ·n(t) = A + R cos(ωt − φ)`, `A = (ŝ·a)(a·n₀)`,
/// `R = |ŝ⊥|·|n₀⊥|`. The site is lit (binary, the production rate assumes a sunlit site) iff the
/// spin phase `ψ = ωt − φ` lies in `(−ψ0, ψ0) mod 2π`, `ψ0 = acos(−A/R)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LitWindow {
  pub mode: f32,
  /// spin phase at the window start, `[−π, π)`
  pub psi_start: f64,
  /// lit half arc, `(0, π)` in periodic mode
  pub psi0: f64,
  /// total lit phase over the window (rad, periodic mode)
  pub lit_phase: f64,
  /// lit time over the window (s)
  pub lit_time_s: f64,
}

/// wraps to `[−π, π)`
fn wrap_pi(x: f64) -> f64 {
  use core::f64::consts::PI;
  x - 2.0 * PI * <f64 as FloatLike>::floor((x + PI) / (2.0 * PI))
}

impl LitWindow {
  /// `sun`: unit direction towards the Sun; `n0`: unit site normal at the window start (root
  /// frame); `axis`: unit spin axis; `omega ≥ 0` (rad/s).
  pub fn new(sun: V3, n0: V3, axis: V3, omega: f64, dur_s: f64) -> Self {
    use core::f64::consts::PI;
    let dur_s = if dur_s.is_finite() {
      dur_s.max(0.0)
    } else {
      0.0
    };
    let constant = |lit: bool| LitWindow {
      mode: LIT_MODE_ALWAYS,
      psi_start: 0.0,
      psi0: if lit { PI } else { 0.0 },
      lit_phase: 0.0,
      lit_time_s: if lit { dur_s } else { 0.0 },
    };
    let (sa, na) = (dot(sun, axis), dot(n0, axis));
    let s_perp = sub(sun, scale(axis, sa));
    let n_perp = sub(n0, scale(axis, na));
    let a = sa * na;
    let r = norm(s_perp) * norm(n_perp);
    if !(omega > 0.0) || !omega.is_finite() || r < 1e-9 {
      return constant(dot(sun, n0) > 0.0);
    }
    let c = -a / r;
    if c <= -1.0 {
      return constant(true);
    }
    if c >= 1.0 {
      return constant(false);
    }
    let psi0 = <f64 as FloatLike>::acos(c);
    // ŝ·n(t) − A = (ŝ⊥·n⊥) cos ωt + ŝ·(a × n⊥) sin ωt = R cos(ωt − φ)
    let a_x_n = [
      axis[1] * n_perp[2] - axis[2] * n_perp[1],
      axis[2] * n_perp[0] - axis[0] * n_perp[2],
      axis[0] * n_perp[1] - axis[1] * n_perp[0],
    ];
    let phi = <f64 as FloatLike>::atan2(dot(sun, a_x_n), dot(s_perp, n_perp));
    let psi_start = wrap_pi(-phi);
    let lit_phase = lit_measure(psi_start + omega * dur_s, psi0) - lit_measure(psi_start, psi0);
    LitWindow {
      mode: LIT_MODE_PERIODIC,
      psi_start,
      psi0,
      lit_phase,
      lit_time_s: (lit_phase / omega).clamp(0.0, dur_s),
    }
  }

  /// [`DustBatch::lit`] encoding
  pub fn to_gpu(&self) -> [f32; 4] {
    [
      self.psi_start as f32,
      self.psi0 as f32,
      self.lit_phase as f32,
      self.mode,
    ]
  }
}

/// lit phase measure of `[−π, x]` (monotonic, periodic increments of `2ψ0` per turn)
fn lit_measure(x: f64, psi0: f64) -> f64 {
  use core::f64::consts::PI;
  let turns = <f64 as FloatLike>::floor((x + PI) / (2.0 * PI));
  turns * 2.0 * psi0 + (wrap_pi(x) + psi0).clamp(0.0, 2.0 * psi0)
}

/// rotates `v` by angle `angle` about unit `axis` (Rodrigues, f64)
pub fn rotate_axis_angle(v: V3, axis: V3, angle: f64) -> V3 {
  let (s, c) = (
    <f64 as FloatLike>::sin(angle),
    <f64 as FloatLike>::cos(angle),
  );
  let kxv = [
    axis[1] * v[2] - axis[2] * v[1],
    axis[2] * v[0] - axis[0] * v[2],
    axis[0] * v[1] - axis[1] * v[0],
  ];
  let kv = dot(axis, v) * (1.0 - c);
  [
    v[0] * c + kxv[0] * s + axis[0] * kv,
    v[1] * c + kxv[1] * s + axis[1] * kv,
    v[2] * c + kxv[2] * s + axis[2] * kv,
  ]
}

/// Spin `(axis, ω ≥ 0)` from two attitudes `dt_s` apart (shortest arc). Fallback for bodies without a
/// rotational model: aliases once `dt_s` exceeds half a rotation.
pub fn spin_from_attitudes(q0: [f32; 4], q1: [f32; 4], dt_s: f64) -> [f64; 4] {
  let none = [0.0, 0.0, 1.0, 0.0];
  if !(dt_s > 0.0) {
    return none;
  }
  // dq = q1 · conj(q0), in the root frame
  let d = qmul(q1, [-q0[0], -q0[1], -q0[2], q0[3]]);
  let (mut x, mut y, mut z, mut w) = (d[0] as f64, d[1] as f64, d[2] as f64, d[3] as f64);
  if w < 0.0 {
    (x, y, z, w) = (-x, -y, -z, -w);
  }
  let s = <f64 as FloatLike>::sqrt(x * x + y * y + z * z);
  if s < 1e-9 {
    return none;
  }
  let angle = 2.0 * <f64 as FloatLike>::atan2(s, w);
  [x / s, y / s, z / s, angle / dt_s]
}

// ─────────────────────────────────────────────────────────────────────────────
// Ring bookkeeping, shared by GPU and CPU modes
// ─────────────────────────────────────────────────────────────────────────────

/// [`LiveBatch::ready`]: recorded in a command buffer that has not been submitted yet
pub const READY_PENDING: u64 = u64::MAX;
/// [`LiveBatch::ready`]: ring content invalid (scene restored), must be emitted again
pub const READY_NEEDS_EMIT: u64 = 0;

/// Live batch: `[first, first + count)` monotonic indices, youngest emission time, mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LiveBatch {
  pub first: u64,
  pub count: u32,
  /// emission window index on the scaled-time grid (see [`DustHostState::tick`])
  pub window: i64,
  pub t_end_s: f64,
  pub mass_g: f64,
  /// descriptor as emitted (deterministic: re-emitting it rewrites identical clusters)
  pub desc: DustBatch,
  /// compute timeline value after which the ring slots are written, or [`READY_PENDING`] /
  /// [`READY_NEEDS_EMIT`]
  pub ready: u64,
}

/// Host-side ring bookkeeping. Single writer, FIFO expiry: no GPU readback, no atomics.
#[derive(Debug, Clone)]
pub struct RingState {
  /// power of two
  pub capacity: u32,
  /// monotonic write index
  pub head: u64,
  /// monotonic index of the oldest live cluster
  pub tail: u64,
  /// live batches, oldest first
  pub batches: alloc::collections::VecDeque<LiveBatch>,
  /// [`Self::drawable`] as of the last point with nothing pending ([`Self::publish`]): what the
  /// renderer draws. The live value drops the batches of a tick between its emission and
  /// [`Self::mark_submitted`], and the renderer reads the ring concurrently with the logic tick: the
  /// newest window would flicker.
  pub published: Option<(u32, u32, u64)>,
}

impl RingState {
  pub fn new(capacity: u32) -> Self {
    assert!(capacity.is_power_of_two());
    Self {
      capacity,
      head: 0,
      tail: 0,
      batches: alloc::collections::VecDeque::new(),
      published: None,
    }
  }
  #[inline]
  pub fn mask(&self) -> u32 {
    self.capacity - 1
  }
  pub fn live(&self) -> u32 {
    (self.head - self.tail) as u32
  }
  pub fn free_slots(&self) -> u32 {
    self.capacity - self.live()
  }
  /// Reserves `desc.count` slots at the head, fills `first_index` / `ring_mask` of `desc` and
  /// records it as [`READY_PENDING`]. Returns the completed descriptor.
  pub fn push_batch(&mut self, desc: DustBatch, t_end_s: f64, mass_g: f64) -> DustBatch {
    self.push_window_batch(desc, t_end_s, mass_g, -1)
  }
  /// [`Self::push_batch`] recording the emission window index `window`.
  pub fn push_window_batch(
    &mut self,
    mut desc: DustBatch,
    t_end_s: f64,
    mass_g: f64,
    window: i64,
  ) -> DustBatch {
    debug_assert!(desc.count <= self.free_slots());
    let first = self.head;
    self.head += desc.count as u64;
    // ring slot = monotonic index & mask (randomness comes from the seed + in-batch index)
    desc.first_index = first as u32;
    desc.ring_mask = self.mask();
    self.batches.push_back(LiveBatch {
      first,
      count: desc.count,
      window,
      t_end_s,
      mass_g,
      desc,
      ready: READY_PENDING,
    });
    desc
  }
  /// every [`READY_PENDING`] batch was submitted with compute timeline value `value`
  pub fn mark_submitted(&mut self, value: u64) {
    for b in self.batches.iter_mut().rev() {
      if b.ready == READY_PENDING {
        b.ready = value;
      }
    }
    self.publish();
  }
  /// Records [`Self::drawable`] for the renderer ([`Self::published`]) unless a batch is pending
  /// (emitted, not submitted yet): the renderer keeps the previous range until it is.
  pub fn publish(&mut self) {
    if !self.batches.iter().any(|b| b.ready == READY_PENDING) {
      self.published = Some(self.drawable());
    }
  }
  /// marks the whole ring content invalid (after a scene restore)
  pub fn invalidate_gpu(&mut self) {
    for b in self.batches.iter_mut() {
      b.ready = READY_NEEDS_EMIT;
    }
    self.published = None;
  }
  /// up to `max` oldest batches needing re-emission, marked [`READY_PENDING`]
  pub fn take_reemit(&mut self, max: usize) -> alloc::vec::Vec<DustBatch> {
    let mut out = alloc::vec::Vec::new();
    for b in self.batches.iter_mut() {
      if out.len() >= max {
        break;
      }
      if b.ready == READY_NEEDS_EMIT {
        b.ready = READY_PENDING;
        out.push(b.desc);
      }
    }
    out
  }
  /// The drawable prefix: `(first_slot, live_count, compute_wait_value)`. Stops at the first
  /// batch not yet submitted / re-emitted, so the range stays contiguous.
  pub fn drawable(&self) -> (u32, u32, u64) {
    let first_slot =
      self.batches.front().map(|b| (b.first & self.mask() as u64) as u32).unwrap_or(0);
    let mut live = 0u32;
    let mut wait = 0u64;
    for b in self.batches.iter() {
      if b.ready == READY_PENDING || b.ready == READY_NEEDS_EMIT {
        break;
      }
      live += b.count;
      wait = wait.max(b.ready);
    }
    (first_slot, live, wait)
  }
  /// retires batches whose youngest cluster is older than `ttl_s` at `t_now_s`
  pub fn retire(&mut self, t_now_s: f64, ttl_s: f64) {
    while let Some(b) = self.batches.front() {
      if t_now_s - b.t_end_s > ttl_s {
        self.tail = b.first + b.count as u64;
        self.batches.pop_front();
      } else {
        break;
      }
    }
  }
  /// drops every batch emitted after `t_s` (time scrubbed backwards)
  pub fn rewind(&mut self, t_s: f64) {
    while let Some(b) = self.batches.back() {
      if b.t_end_s > t_s {
        self.head = b.first;
        self.batches.pop_back();
      } else {
        break;
      }
    }
  }
  /// live mass (g), used for exposure normalization
  pub fn live_mass_g(&self) -> f64 {
    self.batches.iter().map(|b| b.mass_g).sum()
  }
  /// first live slot (masked)
  pub fn first_slot(&self) -> u32 {
    (self.tail & self.mask() as u64) as u32
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-system host state (lives in the ECS component, written by the logic thread, read by the
// render scene builder)
// ─────────────────────────────────────────────────────────────────────────────

/// emission windows per TTL: the scaled-time grid of [`DustHostState::tick`]
pub const WINDOWS_PER_TTL: f64 = 256.0;
/// closed emission windows emitted per tick at most (a seek fills one TTL in a few ticks)
pub const MAX_WINDOWS_PER_TICK: usize = 64;
/// ring slots kept free so a new batch never lands on slots retired moments ago (which a frame in
/// flight may still evaluate): 1/8 of the ring
pub const RING_GUARD_DIVISOR: u32 = 8;
/// batches re-emitted per tick after a restore
pub const REEMIT_PER_TICK: usize = 64;

/// Heliocentric state of the jet (particle-system entity) at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JetState {
  /// scaled seconds since the start epoch
  pub t_s: f64,
  pub r_m: V3,
  pub v_ms: V3,
  /// particle-system → root rotation (xyzw)
  pub rot: [f32; 4],
  /// unit outward surface normal at the jet site (root frame): lit iff it faces the Sun
  pub site_normal: V3,
  /// nucleus spin, unit axis (root frame) + ω ≥ 0 (rad/s). `None`: estimated by
  /// [`DustHostState::tick`] from consecutive attitudes ([`spin_from_attitudes`])
  pub spin: Option<[f64; 4]>,
  /// jet site offset from the nucleus centre (m, particle-system frame): `r_m` is the nucleus
  /// centre plus `rot · site_offset_m`. Emission turns it with the spin over a window
  /// ([`emit_cluster`]); zero for a jet at the centre.
  pub site_offset_m: [f32; 3],
}

impl JetState {
  /// unit direction towards the Sun
  pub fn sun_dir(&self) -> V3 {
    let n = norm(self.r_m);
    if n > 0.0 {
      scale(self.r_m, -1.0 / n)
    } else {
      [0.0, 0.0, 1.0]
    }
  }
  /// spin, `[0, 0, 1, 0]` (no rotation) when unknown
  pub fn spin_or_still(&self) -> [f64; 4] {
    self.spin.unwrap_or([0.0, 0.0, 1.0, 0.0])
  }
  /// illumination of the site over `[t_s, t_s + dur_s]`
  pub fn lit_window(&self, dur_s: f64) -> LitWindow {
    let w = self.spin_or_still();
    LitWindow::new(
      self.sun_dir(),
      self.site_normal,
      [w[0], w[1], w[2]],
      w[3],
      dur_s,
    )
  }
}

/// Everything the renderer needs to evaluate and draw one system this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustDrawState {
  /// heliocentric render anchor (m): the jet position of the last emission tick, a fixed point
  /// (precision only). The draw translation is `anchor − eye` from heliocentric values, never the
  /// jet / comet entity transforms: particles are not children of the comet.
  pub anchor_m: [f64; 3],
  /// index of the age tier in its system (LOD header slot, `< LOD_MAX_TIERS`)
  pub tier: u32,
  /// first slot of the tier's sub-ring in the system buffers (`first_slot` is relative to it)
  pub ring_base: u32,
  pub first_slot: u32,
  pub live_count: u32,
  /// ring capacity (render-time children budget)
  pub capacity: u32,
  /// compute timeline value the graphics submit must wait on (0 = none)
  pub compute_wait: u64,
  pub frame: DustFrame,
  /// unit anti-sun direction (root axes), w = solar gravity at the jet (m/s²)
  pub anti_sun_g: [f32; 4],
  /// exposure reference optical depth ([`DustEmitConfig::tau_ref`], 0 = unknown)
  pub tau_ref: f32,
}

impl DustDrawState {
  /// Evaluation at `t_s` (the time of the scene snapshot the camera comes from), same anchor and
  /// band. Clusters emitted after `t_s` get a negative age and are culled.
  pub fn at_time(mut self, t_s: f64) -> Self {
    self.frame = self.frame.at_time(t_s);
    self
  }
}

/// Emission inputs of one system for one tick (see `ParticleSystemEmitParams::dust_emit_config`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustEmitConfig {
  /// dust production rate at the jet's current heliocentric distance (kg/s)
  pub q_dust_kgs: f64,
  /// cluster lifetime (scaled s)
  pub ttl_s: f64,
  pub dist: SizeDistribution,
  pub diameter_um: f32,
  pub density_gcm3: f32,
  /// β at the configured grain size
  pub beta_ref: f32,
  /// ejection speed mean / std at the configured grain size (m/s)
  pub v_mean: f32,
  pub v_std: f32,
  /// unit jet axis in the particle-system frame
  pub jet_dir: [f32; 3],
  pub aperture_rad: f32,
  pub seed: u32,
}

impl DustEmitConfig {
  /// Exposure reference optical depth: one day of production (`q`, at the heliocentric distance
  /// this config was made for) spread over a disc of radius `v_mean · 1 day`. Depends only on the
  /// jet configuration: never on the history, the tiers or the camera, so the brightness of a
  /// given dust column is stable (see `dust.vert`: pixels show `exposure · τ`).
  pub fn tau_ref(&self) -> f64 {
    let t = TAU_REF_AGE_S;
    let sigma_m2 = self.q_dust_kgs.max(0.0) * 1e3 * self.xsec_per_g_ref() as f64 * t;
    let r_m = (self.v_mean as f64).max(1e-3) * t;
    sigma_m2 / (core::f64::consts::PI * r_m * r_m)
  }
  /// cross-section per gram at the configured grain radius (m²/g)
  pub fn xsec_per_g_ref(&self) -> f32 {
    let s_m = (self.diameter_um * 0.5).max(1e-3) * 1e-6;
    3.0 / (4.0 * self.density_gcm3.max(1e-3) * 1e6 * s_m)
  }
}

/// Age band of one dust tier, as multiples of the TTL: the tier draws clusters aged
/// `[min_ttl · TTL, max_ttl · TTL)` on its own emission grid of `WINDOWS_PER_TTL` windows over the
/// band. Older tiers have longer windows (fewer, heavier clusters per scaled day): long history at
/// the same memory. `s_min_factor` can drop small grains from a tier (unused by default: they carry
/// most of the optical depth).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierBand {
  pub min_ttl: f64,
  pub max_ttl: f64,
  /// minimum grain size multiplier (≥ 1), see [`SizeDistribution::truncated`]
  pub s_min_factor: f64,
  /// fade out at the max age (the oldest tier); younger tiers hand over to the next one
  pub fade: bool,
}

impl TierBand {
  /// the single-tier behaviour: ages `[0, TTL]`, all sizes, fade at the end
  pub const SINGLE: Self = Self {
    min_ttl: 0.0,
    max_ttl: 1.0,
    s_min_factor: 1.0,
    fade: true,
  };
}

/// Seed of emission window `k` (full i64) of tier `tier`: every (config seed, tier, window) draws
/// independent clusters. (Formerly `seed ^ pcg(k as u32 ^ c)`: no tier, so the three tiers drew
/// the same clusters for the same window index, and `k` truncated to 32 bits.)
pub fn window_seed(cfg_seed: u32, tier: u32, k: i64) -> u32 {
  let k = k as u64;
  pcg(cfg_seed ^ pcg(k as u32 ^ pcg((k >> 32) as u32 ^ pcg(tier ^ 0x5DEE_CE66))))
}

#[derive(Debug, Clone)]
pub struct DustHostState {
  pub ring: RingState,
  /// first slot of this tier's sub-ring in the system's cluster / render buffers
  pub ring_base: u32,
  /// index of the tier in its system (0 = youngest): part of every window seed, so tiers never
  /// draw the same clusters for the same window index
  pub tier: u32,
  /// age band (see [`TierBand`])
  pub band: TierBand,
  /// origin (scaled s) of the emission grid: the jet's **ignition**, nothing is emitted before
  /// it. `None` = a pre-existing tail: the windows run back `max_age` from the current time,
  /// before the start epoch too (`jet_at` must cover negative times); an explicit opt-in
  pub t_on_s: Option<f64>,
  /// monotonic counter of batch descriptor uploads (selects the upload slot)
  pub upload_seq: u64,
  /// latest jet state (written every tick)
  pub jet: Option<JetState>,
  /// TTL (scaled s) of the latest tick
  pub ttl_s: f64,
  /// cross-section per gram at the reference grain size (m²/g)
  pub xsec_per_g_ref: f32,
  /// next closed window to emit (None: derive it from the ring / the TTL)
  pub next_window: Option<i64>,
  /// the youngest ring batch is the provisional preview of the open window
  pub provisional: bool,
  /// window length (scaled s) the ring was filled with
  pub grid_s: f64,
  /// first window not due yet at the latest tick (`next_window ≥ due_window`: caught up)
  pub due_window: i64,
  /// upper bound of the stream count [`Self::stream_shift`] picks (default [`DUST_STREAMS`]):
  /// lets a test hold `S` fixed while the capacity changes
  pub max_streams: u32,
  /// windows that produced nothing because the jet site was unlit over the whole window (or
  /// beyond the production cutoff) since the last reset: a dark wedge in the history has a
  /// reported cause (`the_problem.rdc`: a non-spinning nucleus keeps its site in night for a
  /// third of the orbit)
  pub unlit_windows: u32,
  /// the last tick could not evaluate the jet (`jet_at` returned `None`: the comet is not in the
  /// cartesian cache, e.g. it has no rotation model and the cache fill skipped it). The tier is
  /// then *not* building: nothing can be emitted, and the emission loop must not spin on it
  /// (2026-10-10: a comet without a rotation model kept every tier "building" for ever, the logic
  /// thread submitted `MAX_SEEK_PASSES` empty compute passes per tick and the driver's host heap
  /// was gone in a minute)
  pub jet_unavailable: bool,
}

impl DustHostState {
  pub fn new(capacity: u32) -> Self {
    Self::with_band(capacity, 0, TierBand::SINGLE, Some(0.0))
  }

  /// A tier of `capacity` slots at `ring_base` drawing the ages of `band`, ignited at `t_on_s`
  /// (`None`: pre-existing tail, see [`Self::t_on_s`]).
  pub fn with_band(capacity: u32, ring_base: u32, band: TierBand, t_on_s: Option<f64>) -> Self {
    Self {
      ring: RingState::new(capacity),
      ring_base,
      tier: 0,
      band,
      t_on_s,
      upload_seq: 0,
      jet: None,
      ttl_s: 0.0,
      xsec_per_g_ref: 0.0,
      next_window: None,
      provisional: false,
      grid_s: 0.0,
      due_window: 0,
      max_streams: DUST_STREAMS,
      unlit_windows: 0,
      jet_unavailable: false,
    }
  }

  /// the same tier with system index `tier` (window seeds, see [`window_seed`])
  pub fn with_tier(mut self, tier: u32) -> Self {
    self.tier = tier;
    self
  }

  /// forgets every cluster and the emission history (simulation reset)
  pub fn reset(&mut self) {
    let max_streams = self.max_streams;
    *self = Self::with_band(self.ring.capacity, self.ring_base, self.band, self.t_on_s)
      .with_tier(self.tier);
    self.max_streams = max_streams;
  }

  /// `(min, max)` age (scaled s) of this tier for a TTL
  pub fn age_band_s(&self, ttl_s: f64) -> (f64, f64) {
    (self.band.min_ttl * ttl_s, self.band.max_ttl * ttl_s)
  }

  /// Window length (scaled s) of the emission grid for a TTL.
  pub fn window_len_s(ttl_s: f64) -> f64 {
    (ttl_s / WINDOWS_PER_TTL).max(1.0)
  }

  /// Clusters per closed window, before rounding to whole time samples
  fn raw_clusters_per_window(&self) -> u32 {
    ((self.ring.capacity as f64 * BUDGET_SAFETY / WINDOWS_PER_TTL) as u32).max(1)
  }

  /// `log₂` of the tier's streams: as many as [`DUST_STREAMS`] while a full window still holds
  /// [`STREAM_MIN_SAMPLES`] time samples (64 on a 65 536-slot tier, 16 on 16 384, 4 on 4 096, 2 on
  /// 2 048).
  pub fn stream_shift(&self) -> u32 {
    let n = (self.raw_clusters_per_window() / STREAM_MIN_SAMPLES).clamp(1, self.max_streams.max(1));
    31 - n.leading_zeros()
  }

  /// Clusters per closed window: the steady-state budget spread over one TTL of windows, whole
  /// time samples of every stream.
  pub fn clusters_per_window(&self) -> u32 {
    let n = 1 << self.stream_shift();
    (self.raw_clusters_per_window() / n).max(1) * n
  }

  /// One logic tick of host-side emission at scaled time `t_now_s`, **deterministic in time**:
  /// emission happens on a fixed scaled-time grid of windows `[t_on + kΔ, t_on + (k+1)Δ)`,
  /// `k ≥ 0`, anchored at the ignition `t_on` ([`Self::t_on_s`]; a pre-existing tail uses the
  /// start epoch and `k < 0`), `Δ = ttl / WINDOWS_PER_TTL`, and window `k` always produces the
  /// same batch (jet state at `t_on + kΔ` from `jet_at`, mass `q(r)·lit time`, budgeted count,
  /// seed from `(tier, k)`). The dust at any epoch is a pure function of the parameters, the
  /// ignition and the epoch, so playing, pausing, changing the speed or seeking all give the same
  /// result, and nothing exists before the ignition:
  ///
  /// - the provisional batch of the previous tick is dropped;
  /// - time scrubbed backwards drops the batches ending after `t_now_s`;
  /// - expired batches are retired, batches invalidated by a restore re-emitted
  ///   ([`REEMIT_PER_TICK`]);
  /// - the closed windows not emitted yet (within the last TTL) are emitted, at most
  ///   [`MAX_WINDOWS_PER_TICK`] per tick, oldest first;
  /// - once caught up, the open window `[KΔ, t_now)` is emitted as a provisional preview (so slow
  ///   playback shows emission before the window closes), replaced on the next tick.
  ///
  /// `jet_at(t)` gives the jet state at scaled time `t` (almanac), `cfg_at(jet)` the emission
  /// config there. Returns the descriptors to emit, in order (ring slots reserved,
  /// [`READY_PENDING`]); the caller records them and calls `ring.mark_submitted(value)`.
  pub fn tick(
    &mut self,
    t_now_s: f64,
    jet_at: &dyn Fn(f64) -> Option<JetState>,
    cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
  ) -> alloc::vec::Vec<DustBatch> {
    let mut budget = MAX_WINDOWS_PER_TICK;
    self.tick_budget(t_now_s, jet_at, cfg_at, &mut budget)
  }

  /// [`Self::tick`] drawing closed windows from a shared per-tick `budget` (tiers of one system).
  pub fn tick_budget(
    &mut self,
    t_now_s: f64,
    jet_at: &dyn Fn(f64) -> Option<JetState>,
    cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
    budget: &mut usize,
  ) -> alloc::vec::Vec<DustBatch> {
    let Some(jet_now) = jet_at(t_now_s).map(|j| self.with_spin(j, jet_at)) else {
      self.mark_jet_unavailable();
      return alloc::vec::Vec::new();
    };
    self.jet_unavailable = false;
    let cfg_now = cfg_at(&jet_now);
    let (min_age, max_age) = self.age_band_s(cfg_now.ttl_s);
    let dt_w = Self::window_len_s(max_age - min_age);
    if self.grid_s != dt_w {
      // another TTL changes the grid: the ring content no longer matches it
      let upload_seq = self.upload_seq;
      self.reset();
      self.upload_seq = upload_seq;
      self.grid_s = dt_w;
    }
    // after the reset (which clears them): a system filled in one pass is drawn with this tick's
    // band, not a zero one (every cluster but the newest was culled)
    self.ttl_s = cfg_now.ttl_s;
    self.xsec_per_g_ref = cfg_now.xsec_per_g_ref();
    if self.provisional {
      if let Some(b) = self.ring.batches.pop_back() {
        self.ring.head = b.first;
      }
      self.provisional = false;
    }
    if let Some(prev) = self.jet {
      if t_now_s < prev.t_s {
        self.ring.rewind(t_now_s);
        self.next_window = None;
      }
    }
    self.jet = Some(jet_now);
    self.ring.retire(t_now_s, max_age);
    let mut out = self.ring.take_reemit(REEMIT_PER_TICK);

    // Windows overlapping the band `[t_now − max_age, t_now − min_age]` of emission times. The
    // youngest tier (min 0) ends with the open window; an older one emits a window as soon as its
    // start enters the band (all of it is in the past: Δ ≤ min_age), the age gate hides the rest.
    let (origin, k_floor) = match self.t_on_s {
      Some(t_on) => (t_on, 0i64),
      None => (0.0, i64::MIN),
    };
    let t_rel = t_now_s - origin;
    let k_open = if min_age > 0.0 {
      <f64 as FloatLike>::floor((t_rel - min_age) / dt_w) as i64 + 1
    } else {
      <f64 as FloatLike>::floor(t_rel / dt_w) as i64
    };
    let k_min = (<f64 as FloatLike>::floor((t_rel - max_age) / dt_w) as i64).max(k_floor);
    let mut k = self
      .next_window
      .or_else(|| self.ring.batches.back().map(|b| b.window + 1))
      .unwrap_or(k_min)
      .max(k_min);
    while k < k_open && *budget > 0 {
      if let Some(b) = self.emit_window(k, origin + k as f64 * dt_w, dt_w, dt_w, jet_at, cfg_at) {
        out.push(b);
      }
      k += 1;
      *budget -= 1;
    }
    self.next_window = Some(k);
    self.due_window = k_open;
    if k == k_open && min_age <= 0.0 {
      let t0 = origin + k_open as f64 * dt_w;
      if t_now_s > t0 {
        if let Some(b) = self.emit_window(k_open, t0, t_now_s - t0, dt_w, jet_at, cfg_at) {
          out.push(b);
          self.provisional = true;
        }
      }
    }
    // retired / rewound batches with nothing new emitted: the renderer sees it now
    self.ring.publish();
    out
  }

  /// The jet cannot be evaluated this tick: the tier is caught up by definition (nothing can be
  /// emitted, nothing is pending), so [`DustSystemState::building`] is false and the emission
  /// loop stops. Returns whether this is new (the caller logs it once).
  pub fn mark_jet_unavailable(&mut self) -> bool {
    let new = !self.jet_unavailable;
    // `next_window` is left alone: when the jet reappears the history fills from where it was
    // (a seek's first tick can run before the cartesian cache holds the comet)
    self.jet_unavailable = true;
    new
  }

  /// `jet` with its spin, estimated from the attitude a minute later when the body has no model
  fn with_spin(&self, mut jet: JetState, jet_at: &dyn Fn(f64) -> Option<JetState>) -> JetState {
    if jet.spin.is_none() {
      const DT: f64 = 60.0;
      jet.spin = Some(
        jet_at(jet.t_s + DT)
          .map(|j1| spin_from_attitudes(jet.rot, j1.rot, DT))
          .unwrap_or([0.0, 0.0, 1.0, 0.0]),
      );
    }
    jet
  }

  /// Emits window `k` (`[t0, t0 + dur)`, `dur ≤ dt_w`): `None` when nothing is produced (night
  /// side, beyond the production cutoff) or the ring is full.
  fn emit_window(
    &mut self,
    k: i64,
    t0: f64,
    dur: f64,
    dt_w: f64,
    jet_at: &dyn Fn(f64) -> Option<JetState>,
    cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
  ) -> Option<DustBatch> {
    let jet0 = self.with_spin(jet_at(t0)?, jet_at);
    let cfg = cfg_at(&jet0);
    let lit = jet0.lit_window(dur);
    let q = if cfg.q_dust_kgs.is_finite() {
      cfg.q_dust_kgs.max(0.0)
    } else {
      0.0
    };
    // older tiers keep the larger grains only (the trail population), the rest has left the field
    let (dist, mass_fraction) = cfg.dist.truncated(self.band.s_min_factor);
    let mass_g = q * 1e3 * lit.lit_time_s * mass_fraction;
    if !(mass_g > 0.0) {
      self.unlit_windows = self.unlit_windows.saturating_add(1);
      return None;
    }
    // whole time samples: every batch count a multiple of the streams (streak predecessors)
    let shift = self.stream_shift();
    let n = 1u32 << shift;
    let full = self.clusters_per_window() as f64;
    let want = (-<f64 as FloatLike>::floor(-full * (dur / dt_w).clamp(0.0, 1.0))).max(1.0) as u32;
    let count = want.div_ceil(n) * n;
    let count = count.min(self.emit_free_slots()) / n * n;
    if count == 0 {
      return None;
    }
    // the streams continue from the youngest batch of the ring only if it is the previous window
    // (an empty ring: nothing precedes, the first samples draw no streak anyway; keeps the
    // descriptor a function of the window whatever the emission history)
    let break_before = self.ring.batches.back().is_some_and(|b| b.window != k - 1);
    let mut desc: DustBatch = bytemuck::Zeroable::zeroed();
    desc.set_comet(jet0.r_m, jet0.v_ms, t0, dur);
    desc.rot_start = jet0.rot;
    desc.site_offset = [
      jet0.site_offset_m[0],
      jet0.site_offset_m[1],
      jet0.site_offset_m[2],
      0.0,
    ];
    let spin = jet0.spin_or_still();
    desc.spin = [
      spin[0] as f32,
      spin[1] as f32,
      spin[2] as f32,
      spin[3] as f32,
    ];
    desc.lit = lit.to_gpu();
    desc.jet_dir_aperture = [
      cfg.jet_dir[0],
      cfg.jet_dir[1],
      cfg.jet_dir[2],
      cfg.aperture_rad,
    ];
    // independent per (seed, tier, window), see `window_seed`; the streams are per jet
    let seed = window_seed(cfg.seed, self.tier, k);
    let (size_params, vel_params, mass_params) = batch_params(
      &dist,
      cfg.diameter_um,
      cfg.density_gcm3,
      cfg.beta_ref,
      cfg.v_mean,
      cfg.v_std,
      mass_g,
      stream_key(cfg.seed),
    );
    desc.size_params = size_params;
    desc.vel_params = vel_params;
    desc.mass_params = mass_params;
    desc.mass_params[3] = batch_word(shift, break_before, dur < dt_w);
    desc.count = count;
    desc.seed = seed;
    Some(self.ring.push_window_batch(desc, t0 + dur, mass_g, k))
  }

  /// emission slots usable now (ring free space minus the guard band)
  pub fn emit_free_slots(&self) -> u32 {
    self.ring.free_slots().saturating_sub(self.ring.capacity / RING_GUARD_DIVISOR)
  }

  /// render-side view of the current state, `None` before the first tick or when empty
  pub fn draw_state(&self) -> Option<DustDrawState> {
    let jet = self.jet?;
    let (first_slot, live_count, compute_wait) = self.ring.published?;
    if live_count == 0 {
      return None;
    }
    let rn = norm(jet.r_m);
    let g = SUN_MU_M3_S2 / (rn * rn);
    let (min_age, max_age) = self.age_band_s(self.ttl_s);
    Some(DustDrawState {
      anchor_m: jet.r_m,
      tier: 0,
      ring_base: self.ring_base,
      first_slot,
      live_count,
      capacity: self.ring.capacity,
      compute_wait,
      frame: DustFrame::new(jet.r_m, jet.t_s, [0.0, 0.0, 0.0, 1.0], max_age as f32)
        .with_band(min_age as f32, self.band.fade)
        .with_streams(self.stream_shift()),
      anti_sun_g: [
        (jet.r_m[0] / rn) as f32,
        (jet.r_m[1] / rn) as f32,
        (jet.r_m[2] / rn) as f32,
        g as f32,
      ],
      tau_ref: 0.0,
    })
  }
}

/// Number of age tiers ([`DustSystemState`]): `AETHERVK_DUST_TIERS` (1..=3), 3 by default.
pub fn dust_tier_count() -> usize {
  aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_TIERS")
    .and_then(|s| s.trim().parse::<usize>().ok())
    .unwrap_or(DUST_TIERS_DEFAULT)
    .clamp(1, DUST_TIERS_DEFAULT)
}

/// clusters sampled by [`DustSystemState::coma_radius_m`]
pub const COMA_SAMPLES: usize = 256;

/// default age tiers: `[0, 1)`, `[1, 8)`, `[8, 64)` TTL (30 d → 240 d → ~5.3 yr)
pub const DUST_TIERS_DEFAULT: usize = 3;
const _: () = assert!(DUST_TIERS_DEFAULT <= LOD_MAX_TIERS as usize);
/// emission passes that fill every tier from scratch (seek), `MAX_WINDOWS_PER_TICK` windows each
pub const MAX_SEEK_PASSES: usize =
  DUST_TIERS_DEFAULT * (WINDOWS_PER_TTL as usize).div_ceil(MAX_WINDOWS_PER_TICK) + 1;

/// Per-tier history summary for the diagnostic overlay.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DustTierStats {
  pub live_clusters: u32,
  pub capacity: u32,
  /// `(youngest, oldest)` cluster age (scaled s) of the live batches, by window times
  pub youngest_age_s: f64,
  pub oldest_age_s: f64,
  pub band_min_s: f64,
  pub band_max_s: f64,
  /// every window due so far is emitted
  pub caught_up: bool,
  /// windows that produced nothing (unlit site, no production) since the last reset
  pub unlit_windows: u32,
  /// the last tick could not evaluate the jet ([`DustHostState::jet_unavailable`])
  pub jet_unavailable: bool,
}

/// The dust of one particle system: **age tiers** sharing one ring allocation. Tier `k` draws ages
/// in its [`TierBand`] from its own deterministic emission grid and sub-ring (`ring_base`), so old
/// dust gets coarser windows and larger grains: months to years of trail at the memory and
/// per-frame cost of one TTL. The tiers share one ignition ([`Self::set_ignition`]): a jet emits
/// from the epoch it was created at, and the tail is built in simulated time; a pre-existing tail
/// (`None`, history back to 64 TTL before the start epoch) is an explicit opt-in.
#[derive(Debug, Clone)]
pub struct DustSystemState {
  pub tiers: alloc::vec::Vec<DustHostState>,
  /// monotonic batch descriptor upload counter of the system (selects the descriptor slot)
  pub upload_seq: u64,
  /// exposure reference ([`DustEmitConfig::tau_ref`] at the jet's current distance, the last
  /// positive one beyond the production cutoff), set by the emission tick
  pub tau_ref: f64,
  /// the emission inputs changed (rotation model, jet parameters): the next [`Self::tick`] drops
  /// the history and re-emits it with the new ones ([`Self::request_reemit`])
  pub reemit: bool,
  /// nucleus rotation model the history was emitted with (`None` = not ticked yet): a change seen
  /// by the emission tick itself re-emits, whatever the order of the UI updates
  /// ([`Self::track_spin_model`])
  pub spin_model: Option<Option<crate::scene::BodyRotationalModel>>,
  /// the history was complete at some tick since the last reset (every tier caught up, nothing
  /// awaiting re-emission): only then is the system drawn ([`Self::draw_states`]). A fresh fill
  /// (start, [`Self::request_reemit`], [`Self::invalidate_gpu`]) is otherwise published window by
  /// window, oldest first, and the trail builds up in view (`error_first/second/third.rdc`:
  /// 64 windows per tick, the tier-2 trail arriving in chunks over ~1 s). Sticky, so a tier
  /// lagging a few windows at an extreme time scale never hides the dust.
  pub complete: bool,
}

impl DustSystemState {
  /// `capacity` (power of two) split among [`dust_tier_count`] tiers: 1/2, 1/4, 1/4.
  pub fn new(capacity: u32) -> Self {
    Self::with_tiers(capacity, dust_tier_count())
  }

  pub fn with_tiers(capacity: u32, tiers: usize) -> Self {
    let n = tiers.clamp(1, DUST_TIERS_DEFAULT);
    // (min, max) age in TTL, size cut. No size cut: the small, high-β grains carry most of the
    // optical depth (cross-section ∝ s^-0.5 per log size for n ∝ s^-3.5) and draw the
    // anti-sunward tail; a former ×8 / ×32 cut removed ~72 % / ~91 % of old dust's brightness.
    const BANDS: [(f64, f64, f64); DUST_TIERS_DEFAULT] =
      [(0.0, 1.0, 1.0), (1.0, 8.0, 1.0), (8.0, 64.0, 1.0)];
    let caps: alloc::vec::Vec<u32> = match n {
      1 => alloc::vec![capacity],
      2 => alloc::vec![capacity / 2, capacity / 2],
      _ => alloc::vec![capacity / 2, capacity / 4, capacity / 4],
    };
    let mut base = 0;
    let tiers = (0..n)
      .map(|k| {
        let (min_ttl, max_ttl, s_min_factor) = BANDS[k];
        let band = TierBand {
          min_ttl,
          max_ttl,
          s_min_factor,
          fade: k + 1 == n,
        };
        // `with_tiers` alone keeps the pre-existing-tail default of the unit tests; the component
        // sets the ignition ([`Self::set_ignition`])
        let t_on = if n > 1 { None } else { Some(0.0) };
        let t = DustHostState::with_band(caps[k].max(1), base, band, t_on).with_tier(k as u32);
        base += caps[k];
        t
      })
      .collect();
    Self {
      tiers,
      upload_seq: 0,
      tau_ref: 0.0,
      reemit: false,
      spin_model: None,
      complete: false,
    }
  }

  /// total ring slots of all tiers
  pub fn capacity(&self) -> u32 {
    self.tiers.iter().map(|t| t.ring.capacity).sum()
  }

  pub fn reset(&mut self) {
    for t in &mut self.tiers {
      t.reset();
    }
    self.complete = false;
  }

  /// The history is not complete yet: a tier has due windows not emitted (`next_window <
  /// due_window`, "building" in [`Self::stats`]) or a batch awaiting re-emission
  /// ([`READY_NEEDS_EMIT`]). The logic tick keeps emitting (up to [`MAX_SEEK_PASSES`] passes)
  /// while this holds, and nothing is drawn until it clears ([`Self::complete`]).
  pub fn building(&self) -> bool {
    self.tiers.iter().any(|t| {
      !t.jet_unavailable
        && (!t.next_window.is_some_and(|k| k >= t.due_window)
          || t.ring.batches.iter().any(|b| b.ready == READY_NEEDS_EMIT))
    })
  }

  /// Every tier of the system: the jet cannot be evaluated ([`DustHostState::mark_jet_unavailable`]);
  /// true the first time (log it once)
  pub fn mark_jet_unavailable(&mut self) -> bool {
    let mut new = false;
    for t in &mut self.tiers {
      new |= t.mark_jet_unavailable();
    }
    new
  }

  /// The nucleus rotation or a jet parameter changed: every window was emitted with the old ones
  /// (the dust history would show the old spin until it ages out, up to the oldest tier's band).
  /// The next tick resets and refills the tiers deterministically, youngest first, like a seek.
  pub fn request_reemit(&mut self) {
    self.reemit = true;
  }

  /// The emission grid's origin of every tier ([`DustHostState::t_on_s`]): `Some(t)` ignites
  /// the jet at scaled time `t` (nothing before it), `None` keeps a pre-existing tail. A change
  /// re-emits the history from the new origin, like a seek.
  pub fn set_ignition(&mut self, t_on_s: Option<f64>) {
    if self.tiers.iter().any(|t| t.t_on_s != t_on_s) {
      for t in &mut self.tiers {
        t.t_on_s = t_on_s;
      }
      self.reemit = true;
    }
  }

  /// the ignition shared by the tiers (`None`: pre-existing tail)
  pub fn ignition(&self) -> Option<f64> {
    self.tiers.first().and_then(|t| t.t_on_s)
  }

  /// The rotation model the next [`Self::tick`] emits with: a different one than the history's
  /// requests a re-emit ([`Self::request_reemit`]). The emission tick calls it with the model it
  /// reads, so the history always matches the model actually used.
  pub fn track_spin_model(&mut self, model: Option<crate::scene::BodyRotationalModel>) {
    if self.spin_model.is_some_and(|prev| prev != model) {
      self.reemit = true;
    }
    self.spin_model = Some(model);
  }

  pub fn invalidate_gpu(&mut self) {
    for t in &mut self.tiers {
      t.ring.invalidate_gpu();
    }
    self.complete = false;
  }

  pub fn mark_submitted(&mut self, value: u64) {
    for t in &mut self.tiers {
      t.ring.mark_submitted(value);
    }
    self.complete |= !self.building();
  }

  /// Ticks every tier (youngest first, sharing [`MAX_WINDOWS_PER_TICK`] closed windows, so the
  /// near-nucleus part fills first). Returns `(ring_base, descriptor, upload seq)` to emit.
  pub fn tick(
    &mut self,
    t_now_s: f64,
    jet_at: &dyn Fn(f64) -> Option<JetState>,
    cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
  ) -> alloc::vec::Vec<(u32, DustBatch, u64)> {
    if self.reemit {
      self.reemit = false;
      self.reset();
    }
    if self.tiers.iter().all(|t| t.ring.batches.is_empty()) {
      // nothing is drawn (ignition, a rewind below it, a reset): a fill from scratch must not
      // publish a partial history through the sticky gate
      self.complete = false;
    }
    let mut budget = MAX_WINDOWS_PER_TICK;
    let mut out = alloc::vec::Vec::new();
    for t in &mut self.tiers {
      for b in t.tick_budget(t_now_s, jet_at, cfg_at, &mut budget) {
        out.push((t.ring_base, b, self.upload_seq));
        self.upload_seq += 1;
      }
    }
    self.complete |= !self.building();
    out
  }

  /// One draw state per non-empty tier, all with the system's exposure reference; none until the
  /// history is [`Self::complete`] (a partial fill is never drawn). (A mean cluster flux over the
  /// live clusters used to set the exposure: filling the old tiers with heavy clusters dimmed
  /// the whole system 10×, `initial_burst.rdc` vs `late_*.rdc`.)
  pub fn draw_states(&self) -> alloc::vec::Vec<DustDrawState> {
    if !self.complete {
      return alloc::vec::Vec::new();
    }
    let mut states: alloc::vec::Vec<DustDrawState> = self
      .tiers
      .iter()
      .enumerate()
      .filter_map(|(i, t)| {
        t.draw_state().map(|mut s| {
          s.tier = i as u32;
          s
        })
      })
      .collect();
    for s in &mut states {
      s.tau_ref = self.tau_ref as f32;
    }
    states
  }

  /// Radius (m) of the visible coma: the flux-weighted 90th percentile distance from the jet of
  /// the youngest tier's clusters, from ≤ [`COMA_SAMPLES`] clusters evaluated on the host
  /// (`emit_cluster` + `evaluate_cluster`, the reference path). `None` without drawable dust.
  /// Used to frame the comet (Earth observer tracking preset).
  pub fn coma_radius_m(&self) -> Option<f64> {
    if !self.complete {
      return None;
    }
    let tier = self.tiers.first()?;
    let frame = tier.draw_state()?.frame;
    let total: u64 = tier.ring.batches.iter().map(|b| b.count as u64).sum();
    let stride = (total / COMA_SAMPLES as u64).max(1) as usize;
    let mut samples: alloc::vec::Vec<(f64, f64)> = alloc::vec::Vec::new();
    for b in tier.ring.batches.iter() {
      for j in (0..b.count).step_by(stride) {
        let c = emit_cluster(&b.desc, j);
        let e = evaluate_cluster(&c, 0, &frame);
        let flux = e.age_id_dbeta_flux[3] as f64;
        if flux > 0.0 {
          let p = e.pos_size;
          let d = ((p[0] as f64).powi(2) + (p[1] as f64).powi(2) + (p[2] as f64).powi(2)).sqrt();
          samples.push((d, flux));
        }
      }
    }
    if samples.is_empty() {
      return None;
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_flux: f64 = samples.iter().map(|s| s.1).sum();
    let mut acc = 0.0;
    for (d, w) in &samples {
      acc += w;
      if acc >= 0.9 * total_flux {
        return Some(*d);
      }
    }
    samples.last().map(|s| s.0)
  }

  /// per-tier history summary (diagnostic)
  pub fn stats(&self) -> alloc::vec::Vec<DustTierStats> {
    self
      .tiers
      .iter()
      .map(|t| {
        let t_now = t.jet.map(|j| j.t_s).unwrap_or(0.0);
        let t_of = |b: &LiveBatch| b.desc.comet_r_t_hi[3] as f64 + b.desc.comet_r_t_lo[3] as f64;
        let (min_s, max_s) = t.age_band_s(t.ttl_s);
        DustTierStats {
          live_clusters: t.ring.live(),
          capacity: t.ring.capacity,
          youngest_age_s: t
            .ring
            .batches
            .back()
            .map(|b| (t_now - b.t_end_s).max(0.0))
            .unwrap_or(0.0),
          oldest_age_s: t.ring.batches.front().map(|b| (t_now - t_of(b)).max(0.0)).unwrap_or(0.0),
          band_min_s: min_s,
          band_max_s: max_s,
          caught_up: t.jet_unavailable || t.next_window.is_some_and(|k| k >= t.due_window),
          unlit_windows: t.unlit_windows,
          jet_unavailable: t.jet_unavailable,
        }
      })
      .collect()
  }
}

#[cfg(test)]
mod tests;
