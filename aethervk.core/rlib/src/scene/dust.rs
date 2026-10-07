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

use aethervk_oshal_rlib::math::FloatLike;
use df::{Df, Df3, consts};

/// ring capacity for discrete / high-end GPUs (power of two)
pub const RING_CAPACITY_HIGH: u32 = 262_144;
/// ring capacity for integrated / mobile GPUs (power of two)
pub const RING_CAPACITY_LOW: u32 = 32_768;
/// render-time instance budget per cluster of ring capacity: a tier draws at most
/// `capacity · CHILDREN_PER_CLUSTER` children, spread over its **on-screen** clusters by the LOD
/// pass ([`lod_children`], `dust_lod.comp`)
pub const CHILDREN_PER_CLUSTER: u32 = 8;
/// children per cluster, maximum (the child index has [`LOD_CHILD_BITS`] bits in the instance list,
/// and `dust.vert` hashes children with this stride)
pub const MAX_CHILDREN_PER_CLUSTER: u32 = 1024;
/// instance list entry: `cluster | child << LOD_CLUSTER_BITS` (`dust_lod.comp`, `dust.vert`);
/// child [`TRACER_CHILD`] is the cluster's tracer dot
pub const LOD_CLUSTER_BITS: u32 = 22;
pub const LOD_CHILD_BITS: u32 = 32 - LOD_CLUSTER_BITS;
const _: () = assert!(MAX_CHILDREN_PER_CLUSTER <= 1 << LOD_CHILD_BITS);
const _: () = assert!(RING_CAPACITY_HIGH <= 1 << LOD_CLUSTER_BITS);
/// Bits of the stable child-pattern id a cluster carries in the low mantissa bits of its β
/// half-spread (`DustCluster::misc.w`, copied to `DustRenderCluster::age_id_dbeta_flux.z`). The id
/// is hashed from the emission record (window seed, in-batch index), never from the ring slot, so
/// a cluster draws the same children after a seek, a rewind or in another tier's sub-ring. 16 bits
/// cost the half-spread ≤ 2⁻⁷ (truncation, irrelevant for a spread) and keep both structs at their
/// size; the id is exact on GPU and CPU (u32 ops only), the dbeta bits above it may differ by ulps.
pub const CHILD_ID_BITS: u32 = 16;
pub const CHILD_ID_MASK: u32 = (1 << CHILD_ID_BITS) - 1;
const _: () = assert!(CHILD_ID_BITS + LOD_CHILD_BITS <= 32);
/// Age scale (scaled s) of the child reshaping (`child_offset`, point spreads): a cluster's
/// children turn from one random cloud to another (per-child rate) once per doubling of
/// `1 + age / τ`.
pub const CHILD_RESHAPE_TAU_S: f32 = 3600.0;
/// Emission streams per jet, at most. Stream `s` has a fixed direction in the jet cone, a fixed
/// grain-size stratum and a fixed speed draw, hashed from the jet configuration (never from the
/// window), so its clusters of consecutive time samples are points of one streakline: on a
/// spinning nucleus it sweeps a spiral / arc, at fixed β it is a syndyne. A tier uses
/// `S = 2^shift` streams ([`DustHostState::stream_shift`]: fewer on small rings, so a window still
/// holds [`STREAM_MIN_SAMPLES`] time samples). Batch layout `j = i·S + s` (time sample `i`), every
/// batch count a multiple of `S`, so in a tier's compact render buffer the stream predecessor of
/// cluster `r` is `r − S` ([`streak_pred`]).
pub const DUST_STREAMS: u32 = 64;
/// time samples per full window a tier keeps at least when choosing its stream count
pub const STREAM_MIN_SAMPLES: u32 = 4;
/// per-cluster jitter of a stream's direction: fraction of its cone stratum (polar) and of a turn
/// (azimuth)
pub const STREAM_DIR_JITTER: f32 = 0.03;
/// Lateral velocity dispersion of a stream in units of its cone cell's angular radius
/// (`aperture / √S`). Cells of area π r² tile the cone with centres d ≈ 1.9 r apart (hexagonal), so
/// σ = r gives σ/d ≈ 0.53: the S gaussian streams sum to a uniform cone (ripple
/// 2·exp(−2π²σ²/d²) < 1 %). At 0.5 r (σ/d ≈ 0.26, ripple ~50 %) old dust split into one island per
/// stream, thousands of km apart (`detached.rdc`).
pub const STREAM_SIGMA_CELLS: f32 = 1.0;
/// per-cluster jitter of a stream's size, fraction of its size stratum width
pub const STREAM_SIZE_JITTER: f32 = 0.05;
/// part of a time stratum drawn per cluster (the rest is shared by the streams of the sample)
pub const STREAM_TIME_JITTER: f32 = 0.1;
/// per-cluster speed jitter, in units of the relative speed spread
pub const STREAM_SPEED_JITTER: f32 = 0.1;
/// Sign bit of the β half-spread field (`DustCluster::misc.w` → `DustRenderCluster`): the stream
/// is interrupted before this cluster (the jet site was dark, or the previous window emitted
/// nothing), so it draws no streak towards its predecessor. The child-pattern id lives in the low
/// bits, the half-spread itself is positive.
pub const STREAM_BREAK_BIT: u32 = 1 << 31;
/// a dark gap shorter than this fraction of a rotation does not interrupt a stream
pub const STREAM_BREAK_TURNS: f32 = 0.01;
/// [`DustBatch::mass_params`] `w` = `shift + BATCH_BREAK_FLAG · break + BATCH_PROVISIONAL_FLAG ·
/// provisional + BATCH_SIZE_ROTATION_UNIT · rotation` ([`batch_word`]): the stream shift of the batch, whether the previous window of the tier is missing
/// from the ring, and whether it is the provisional preview of the open window
pub const BATCH_BREAK_FLAG: u32 = 16;
/// The provisional batch (open window, re-emitted every tick) pins its last time sample at the
/// window end, the tick time: every stream then starts at the jet (the fountain's base,
/// `comet_mode_full.rdc`); at night `lit_time_map` maps it to the last lit instant.
pub const BATCH_PROVISIONAL_FLAG: u32 = 32;
/// [`DustBatch::mass_params`] `w` also carries `BATCH_SIZE_ROTATION_UNIT · r`: the size-stratum
/// rotation of the window's first time sample ([`stream_stratum`]), `r < S ≤ 64`, so the word
/// stays below 2¹² (exact in f32)
pub const BATCH_SIZE_ROTATION_UNIT: u32 = 64;
/// ndc margin around the view where streaks are still drawn (plus 3 lateral sigmas)
pub const STREAK_MARGIN: f32 = 0.05;
/// clamp of the streak margin (ndc)
pub const STREAK_MARGIN_MAX: f32 = 8.0;
/// `DustRenderCluster::age_id_dbeta_flux.y` bits: the ring slot (low 22 bits), the tier's stream
/// shift at [`RENDER_SHIFT_BIT0`] (4 bits) and [`RENDER_LIVE_BIT`] (evaluated in the age band;
/// culled records have only the slot). The LOD never rewrites this word, so a cluster reads its
/// predecessor's validity race free.
pub const RENDER_SLOT_MASK: u32 = (1 << LOD_CLUSTER_BITS) - 1;
pub const RENDER_SHIFT_BIT0: u32 = 27;
pub const RENDER_LIVE_BIT: u32 = 1 << 31;
/// LOD header words per tier: `[0..4)` `VkDrawIndirectCommand`, `[4]` demand Σwant, `[5]`
/// attempted Σk (including the clusters dropped at the budget), then the words of
/// [`LOD_HEADER_LIST`]..[`LOD_HEADER_ON_SCREEN`]
pub const LOD_HEADER_WORDS: u32 = 64;
/// tiers per system with an LOD header (see [`dust_tier_count`])
pub const LOD_MAX_TIERS: u32 = 4;
/// white-point tile grid over the screen (`dust_lod.comp`)
pub const DUST_TILES_X: u32 = 64;
pub const DUST_TILES_Y: u32 = 36;
pub const DUST_TILE_COUNT: u32 = DUST_TILES_X * DUST_TILES_Y;
/// first word of the tile grid in the LOD buffer (after the tier headers)
pub const LOD_TILE_WORD0: u32 = LOD_MAX_TIERS * LOD_HEADER_WORDS;
/// first word of the instance lists (tier `t` starts at `ring_base · CHILDREN_PER_CLUSTER`)
pub const LOD_LIST_WORD0: u32 = LOD_TILE_WORD0 + DUST_TILE_COUNT;
/// words of the LOD header + tile grid, copied back to the host every frame
pub const LOD_READBACK_WORDS: u32 = LOD_LIST_WORD0;
/// child samples per cluster scattered into the white-point tiles
pub const DUST_TILE_SAMPLES: u32 = 4;
/// percentile of the non-empty tiles taken as the white point (calibrated offline on
/// blobber.rdc from 0.3× to 3000× zoom: within 3× of the pixel 99.5th percentile)
pub const WHITE_TILE_PERCENTILE: f32 = 0.99;
/// fraction of the instance budget the LOD controller aims for
pub const LOD_TARGET_FILL: f32 = 0.9;
/// fraction of the ring targeted in steady state
pub const BUDGET_SAFETY: f64 = 0.8;
/// bit of `DustDrawPushConstants::children`: the target is 8-bit, stochastically round the splat
/// (`dust.frag`) so optical depth below 1/255 keeps its expected value instead of vanishing
pub const DUST_DITHER_FLAG: u32 = 1 << 31;

/// Mirror of the `dust.frag` stochastic rounding to 8 bits: `floor(v·255 + u) / 255` with `u` in
/// `[0, 1)` per splat and pixel, whose expectation over `u` is exactly `v`.
pub fn stochastic_round_8bit(v: f32, u: f32) -> f32 {
  ((v * 255.0 + u).floor() / 255.0).clamp(0.0, 1.0)
}

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
/// drawn radius of every child in pixels (`dust.vert`): particles split, they never grow. A cloud
/// larger on screen is drawn with more children ([`lod_want`]), not with larger ones.
pub const DUST_CHILD_PX: f32 = 1.5;

/// Mirror of the `dust.vert` footprint: `(r_px, r_draw_m)`, the drawn child radius `r_px` in
/// pixels ([`splat_radius_px`]) and the same radius in metres at clip depth `clip_w`. `p11` is the
/// projection y scale, `px_to_ndc_y = 2 / viewport height`, `units_per_m` the layer unit.
pub fn splat_footprint(
  r_px: f32,
  units_per_m: f32,
  p11: f32,
  clip_w: f32,
  px_to_ndc_y: f32,
) -> (f32, f32) {
  let px_per_unit = p11 / clip_w / px_to_ndc_y;
  (r_px, r_px / px_per_unit / units_per_m)
}

/// Children a streak asks for: [`DUST_CHILD_PX`] dots along its visible length `len_px`, times its
/// lateral width `width_px` in dots, at least 1 (`dust_streak_want`). A point spread (`len_px` 0)
/// asks for `(width_px / DUST_CHILD_PX)²`, the former cloud cover.
pub fn streak_want(len_px: f32, width_px: f32) -> f32 {
  ((len_px.max(width_px) / DUST_CHILD_PX).max(1.0)) * (width_px / DUST_CHILD_PX).max(1.0)
}

/// Drawn radius (px) of the `k` children of a cluster asking for `want`: [`DUST_CHILD_PX`] when
/// fully sampled, `DUST_CHILD_PX·sqrt(want/k)` otherwise (the dots still cover the footprint, so a
/// tight budget or a near view gives a softer fog instead of sparse bright speckle; `near.rdc`),
/// at most [`DUST_CHILD_PX_MAX`]. The peak follows (`splat_opacity`): energy exact. Mirror of
/// `dust_splat_radius`.
pub fn splat_radius_px(want: f32, k: u32) -> f32 {
  (DUST_CHILD_PX * <f32 as FloatLike>::sqrt((want / k.max(1) as f32).max(1.0)))
    .min(DUST_CHILD_PX_MAX)
}

/// Children of a point spread of radius `spread_px` ([`streak_want`] without length).
pub fn lod_want(spread_px: f32) -> f32 {
  streak_want(0.0, spread_px)
}

/// Children drawn for `want` under the budget share `lambda`, in `[1, TRACER_CHILD)` (the last
/// child index is the tracer's). Mirror of `dust_lod.comp`.
pub fn lod_children(want: f32, lambda: f32) -> u32 {
  <f32 as FloatLike>::floor(lambda * want + 0.5).clamp(1.0, TRACER_CHILD as f32) as u32
}

// ─── View aids: tracers and flow (`first_particles.rdc` / `second_particles.rdc`) ───
// In the wide view (25 km/px) dust moves 2–150 m/s relative to the nucleus: 63 s of sim time moved
// every particle by 0.005 px, so the coma looked like one shape sliding with the comet. And ~1 M
// dots blend into a fog in which no flow shows even when fast.

/// [`DUST_VIEW_TRACERS`]: one cluster in `TRACER_EVERY` is also drawn as a bright dot at its exact
/// position (real particles to follow); [`DUST_VIEW_FLOW`]: synchrone marks ([`flow_factor`])
pub const DUST_VIEW_TRACERS: u32 = 1;
pub const DUST_VIEW_FLOW: u32 = 2;
/// view aids by default: both (`AETHERVK_DUST_TRACERS=0` / `AETHERVK_DUST_FLOW=0` turn one off)
pub fn dust_view_flags_default() -> u32 {
  let off = |k: &str| aethervk_oshal_rlib::os::env::var(k).is_some_and(|s| s.trim() == "0");
  let mut f = DUST_VIEW_TRACERS | DUST_VIEW_FLOW;
  if off("AETHERVK_DUST_TRACERS") {
    f &= !DUST_VIEW_TRACERS;
  }
  if off("AETHERVK_DUST_FLOW") {
    f &= !DUST_VIEW_FLOW;
  }
  f
}
/// one tracer per this many clusters (~400 in a wide view of the coma and tail)
pub const TRACER_EVERY: u32 = 256;
/// instance child index of a cluster's tracer dot (real children stay below it)
pub const TRACER_CHILD: u32 = MAX_CHILDREN_PER_CLUSTER - 1;
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
/// LOD header words read by `dust.vert`: the tier's instance-list address (u64, written by
/// `dust_lod.comp`), then the host's flow uniform ([`DustFlowUniform`]) and view flags
pub const LOD_HEADER_LIST: u32 = 6;
/// the time-lapse factor `K` (f32 bits; diagnostics, `dust.vert` reads the clock only)
pub const LOD_HEADER_FLOW_SPEED: u32 = 8;
pub const LOD_HEADER_FLAGS: u32 = 9;
/// the flow clock `T` as a df64-style hi / lo pair (exact emission epochs at 10⁸ s)
pub const LOD_HEADER_T_HI: u32 = 10;
pub const LOD_HEADER_T_LO: u32 = 11;
/// this frame's budget share (f32 bits, written by `dust_lod.comp` pass B for `dust.vert`)
pub const LOD_HEADER_LAMBDA: u32 = 13;
/// clusters on screen (demand pass)
pub const LOD_HEADER_ON_SCREEN: u32 = 14;
/// solar gravity at the jet (f32 bits, host): the β extent of the footprints ([`dust_extent`])
pub const LOD_HEADER_SUN_G: u32 = 15;
/// first word of the demand histogram (count, sum per bin, [`lod_lambda_from`]; demand pass)
pub const LOD_HEADER_HIST: u32 = 16;
const _: () = assert!(LOD_HEADER_HIST as usize + 2 * LOD_HIST_BINS <= LOD_HEADER_WORDS as usize);
/// largest drawn child radius (px): a cluster drawn with fewer dots than its footprint asks for
/// (`k < want`: budget, or the [`TRACER_CHILD`] cap) gets larger, softer ones ([`splat_radius_px`])
pub const DUST_CHILD_PX_MAX: f32 = 12.0;

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

/// What `dust.vert` needs for the flow (LOD header words [`LOD_HEADER_FLOW_SPEED`]..): the flow
/// clock `T` as a df64-style hi / lo pair (exact emission epochs at 10⁸ s) and `K` (diagnostics).
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

/// Bins of the demand histogram (`⌊log₂ ⌈want⌉⌋`, wants up to 2²⁰)
pub const LOD_HIST_BINS: usize = 21;

/// Demand histogram bin of a cluster's demand `d` ([`lod_demand`]). Mirror of `dust_lod_hist_bin`.
#[inline]
pub fn lod_hist_bin(d: u32) -> usize {
  ((31 - d.max(1).leading_zeros()) as usize).min(LOD_HIST_BINS - 1)
}

/// What a demand `d` adds to its bin's sum: `d / 2^bin` in 1/256 (in `[256, 512)`), so a bin of
/// 8 M clusters still fits a u32. Mirror of `dust_lod_hist_add`.
#[inline]
pub fn lod_hist_add(d: u32) -> u32 {
  (d.max(1) << 8) >> lod_hist_bin(d)
}

/// Adds the demand `d` of one on-screen cluster to `hist` (count, scaled sum per bin)
pub fn lod_hist_push(hist: &mut [u32], d: u32) {
  let b = lod_hist_bin(d);
  hist[2 * b] += 1;
  hist[2 * b + 1] = hist[2 * b + 1].saturating_add(lod_hist_add(d));
}

/// Budget share of this frame, from this frame's demand (`dust_lod.comp` pass A, the same frame:
/// no feedback lag, no ramp after a view change). `hist` holds per bin ([`lod_hist_bin`]) the
/// count and the scaled demand sum of the on-screen clusters (interleaved, [`lod_hist_push`]). A
/// cluster asking for `w` draws `clamp(round(λ·w), 1, TRACER_CHILD)`, one in [`TRACER_EVERY`] adds
/// a tracer: λ solves `Σ_b c_b·clamp(λ·w̄_b, 1, TRACER_CHILD) = LOD_TARGET_FILL·budget` by bisection
/// in log space (the floor at 1 and the cap are what a plain `budget / Σw` gets wrong: many small
/// clusters underfill, a few huge ones overfill), at most `lambda_max`.
/// Mirror of `dust_lod_lambda`.
pub fn lod_lambda_from(budget: u32, hist: &[u32], on_screen: u32, lambda_max: f32) -> f32 {
  let lmax = lambda_max.max(1e-6);
  let target = LOD_TARGET_FILL * budget as f32 - (on_screen / TRACER_EVERY) as f32;
  let cost = |l: f32| {
    let mut c = 0.0f32;
    for b in 0..LOD_HIST_BINS {
      let (n, s) = (hist[2 * b] as f32, hist[2 * b + 1] as f32);
      if n > 0.0 {
        // mean demand of the bin: s / (256 n) · 2^b
        let mean = s / n * (f32::from_bits(((b as u32) + 127) << 23) * (1.0 / 256.0));
        c += n * (l * mean).clamp(1.0, TRACER_CHILD as f32);
      }
    }
    c
  };
  if cost(lmax) <= target {
    return lmax;
  }
  // log2 λ in [log2 1e-6, log2 lmax]: 24 halvings, ~2⁻²⁰ of the range
  let (mut lo, mut hi) = (
    -19.93f32,
    <f32 as FloatLike>::ln(lmax) * core::f32::consts::LOG2_E,
  );
  for _ in 0..24 {
    let mid = 0.5 * (lo + hi);
    if cost(<f32 as FloatLike>::exp(mid * core::f32::consts::LN_2)) <= target {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  <f32 as FloatLike>::exp(lo * core::f32::consts::LN_2).clamp(1e-6, lmax)
}

/// `DustLodPushConstants::lambda` of the demand pass (pass A) of `dust_lod.comp`
pub const LOD_DEMAND_PASS: f32 = -1.0;

/// White point (exposure-scaled optical depth) from the tile grid in fixed point (`unit` = τ per
/// count): the [`WHITE_TILE_PERCENTILE`] of the non-empty tiles. `None` when all are empty.
pub fn white_point_from_tiles(tiles: &[u32], unit: f32) -> Option<f32> {
  let mut v: alloc::vec::Vec<u32> = tiles.iter().copied().filter(|&c| c > 0).collect();
  if v.is_empty() {
    return None;
  }
  v.sort_unstable();
  let i = ((v.len() - 1) as f32 * WHITE_TILE_PERCENTILE).round() as usize;
  Some(v[i.min(v.len() - 1)] as f32 * unit)
}

/// fraction of the log-distance to the measured white point covered per frame (~0.3 s at 60 Hz)
pub const WHITE_ADAPT_RATE: f32 = 0.5;
/// tile fixed-point unit relative to the current white point: τ from 1e-5 to 4e4 white
pub const WHITE_TILE_UNIT_REL: f32 = 1e-5;

/// Share of the view adaptation in the exposure (log space): the displayed white point is
/// `measured^share` in units of the system's fixed reference (white 1 = the jet's `tau_ref`). 1 =
/// full eye adaptation (`near.rdc` / `med.rdc`: the same tail patch, physically within 1.3 %,
/// showed 1.82× apart because the coma was in one view and not the other), 0 = fixed exposure.
pub const DUST_VIEW_ADAPTATION: f32 = 0.5;

/// White point the exposure divides by, from the measured one ([`DUST_VIEW_ADAPTATION`]); 0 or
/// non-finite (not measured yet) gives the fixed reference 1.
pub fn display_white(measured: f32) -> f32 {
  if !(measured > 0.0) || !measured.is_finite() {
    return 1.0;
  }
  <f32 as FloatLike>::exp(DUST_VIEW_ADAPTATION * <f32 as FloatLike>::ln(measured))
}

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

/// 3 independent N(0,1) from the hash chain started at `h0` (5 hashes); returns the last hash
#[inline]
fn gauss3(h0: u32) -> ([f32; 3], u32) {
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  (
    [
      gauss(u01(h0), u01(h1)),
      gauss(u01(h1 ^ 0x68E3_1DA4), u01(h2)),
      gauss(u01(h3), u01(h4)),
    ],
    h4,
  )
}

/// Child `child` of the cluster with child-pattern id `id` ([`child_id`]): its deterministic
/// offset from the cluster centre (`dust.vert` / `dust_lod.comp`, `dust_child_offset`):
///
/// `spread·(cos θ·N3 + sin θ·N3') + ½·Δβ·g·age²·(anti-sun)`, `θ = ρ·(π/2)·log₂(1 + age/τ)`
///
/// with two independent normals and a per-child rate `ρ ∈ [0.5, 1.5)`: the per-axis variance stays
/// `spread²` at any age, but the children move relative to each other as the cluster ages (a pure
/// `spread·N3` cloud only scales: the same blob zoomed). Child identity depends only on
/// `(id, child)`, so children `0..k` keep their place when the LOD changes `k`.
pub fn child_offset(
  id: u32,
  child: u32,
  spread: f32,
  dbeta_half: f32,
  age: f32,
  anti_sun_g: [f32; 4],
) -> [f32; 3] {
  let (n3, h4) = gauss3(pcg(
    (id & CHILD_ID_MASK)
      .wrapping_mul(MAX_CHILDREN_PER_CLUSTER)
      .wrapping_add(child)
      .wrapping_add(0x9E37_79B9),
  ));
  let h5 = pcg(h4);
  let dbeta = absf(dbeta_half) * (2.0 * u01(h5) - 1.0);
  let (m3, m4) = gauss3(pcg(h5 ^ 0x5BD1_E995));
  let rate = 0.5 + u01(pcg(m4));
  let theta = rate
    * (0.5 * core::f32::consts::PI * core::f32::consts::LOG2_E)
    * <f32 as FloatLike>::ln(1.0 + age.max(0.0) * (1.0 / CHILD_RESHAPE_TAU_S));
  let (c, s_th) = (
    <f32 as FloatLike>::cos(theta),
    <f32 as FloatLike>::sin(theta),
  );
  let s = 0.5 * dbeta * anti_sun_g[3] * age * age;
  [
    spread * (c * n3[0] + s_th * m3[0]) + s * anti_sun_g[0],
    spread * (c * n3[1] + s_th * m3[1]) + s * anti_sun_g[1],
    spread * (c * n3[2] + s_th * m3[2]) + s * anti_sun_g[2],
  ]
}

/// Mirror of the `dust.vert` splat peak: the child's cross-section spread over its **drawn** area,
/// `exposure · σ / r_draw²` (`dust.frag`'s gaussian integrates to 1 over the unit disc), so the
/// summed opacity of a pixel is `exposure ·` the dust optical depth there: independent of zoom,
/// distance and the pixel clamps (energy conserving).
pub fn splat_opacity(exposure: f32, child_flux_m2: f32, r_draw_m: f32) -> f32 {
  exposure * child_flux_m2 / (r_draw_m * r_draw_m).max(1e-30)
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

/// Immutable emission record of one cluster. 80 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustCluster {
  /// heliocentric position (m) at emission, `w` = emission time `t0` (scaled s), df64 high part
  pub r0_t0_hi: [f32; 4],
  /// ... low part
  pub r0_t0_lo: [f32; 4],
  /// heliocentric velocity (m/s) at emission (df64 high part), `w` = β
  pub v0_hi_beta: [f32; 4],
  /// velocity low part, `w` = super-particle mass (g)
  pub v0_lo_mass: [f32; 4],
  /// `x` lateral velocity dispersion of the stream σ_lat (m/s), `y` grain radius (µm),
  /// `z` cross-section per gram (m²/g), `w` child β half-spread, low [`CHILD_ID_BITS`] =
  /// child-pattern id ([`child_id`]), sign bit = [`STREAM_BREAK_BIT`]
  pub misc: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustCluster>() == 80);

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

/// `dust_propagate.comp` push constants. 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustPropagatePushConstants {
  pub clusters: u64,
  pub render: u64,
  pub first_slot: u32,
  pub live_count: u32,
  pub ring_mask: u32,
  pub _pad0: u32,
  pub frame: DustFrame,
}
const _: () = assert!(core::mem::size_of::<DustPropagatePushConstants>() == 96);

/// `dust.vert` / `dust.frag` push constants. 128 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustDrawPushConstants {
  /// `DustRenderBuffer` (compact, index = live-range offset); flux is per child after the LOD
  pub render: u64,
  /// the tier's LOD header: its words [`LOD_HEADER_LIST`] hold the instance list
  /// (`cluster | child << LOD_CLUSTER_BITS`), [`LOD_HEADER_TIME`] / [`LOD_HEADER_FLAGS`] the view aids
  pub lod_header: u64,
  /// particle-system local metres → clip
  pub mvp: [f32; 16],
  /// rgb stream color, a = exposure (gain / (reference optical depth · white point))
  pub color: [f32; 4],
  /// unit anti-sun direction (ps frame), w = solar gravity at the comet (m/s²)
  pub anti_sun_g: [f32; 4],
  /// x units per metre, y P00, z P11, w ±2 / viewport height (negative: 8-bit target, the
  /// fragment shader rounds stochastically, see [`stochastic_round_8bit`])
  pub params: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustDrawPushConstants>() == 128);

/// `dust_lod.comp` push constants. 128 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustLodPushConstants {
  /// `DustRenderBuffer` of the tier (flux rewritten per child)
  pub render: u64,
  /// the tier's LOD header ([`LOD_HEADER_WORDS`] u32)
  pub header: u64,
  /// the system's white-point tile grid ([`DUST_TILE_COUNT`] u32, fixed point)
  pub tiles: u64,
  /// the tier's instance list (`budget` u32)
  pub list: u64,
  pub live_count: u32,
  pub budget: u32,
  /// [`LOD_DEMAND_PASS`] for the demand pass; otherwise the largest budget share allowed (1 in
  /// production; the pass computes this frame's share from the demand, [`lod_lambda_from`])
  pub lambda: f32,
  /// tile counts per unit of exposure-scaled cross-section density: exposure / tile unit
  pub tile_scale: f32,
  /// particle-system local metres → clip
  pub mvp: [f32; 16],
  /// x units per metre, y P00, z P11, w 2 / viewport height
  pub params: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustLodPushConstants>() == 128);

/// What the LOD pass of one tier produced (CPU mirror of `dust_lod.comp`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DustLodResult {
  /// instances written (≤ budget)
  pub instances: u32,
  /// Σ min(⌈want⌉, TRACER_CHILD) of the on-screen clusters (demand pass, statistics)
  pub demand: u32,
  /// clusters on screen (demand pass)
  pub on_screen: u32,
  /// Σ children attempted, including the clusters dropped at the budget
  pub attempted: u32,
  /// this frame's budget share ([`lod_lambda_from`])
  pub lambda: f32,
}

/// A cluster's demand: `⌈want⌉`, at most 2²⁰ (the histogram's last bin)
#[inline]
pub fn lod_demand(want: f32) -> u32 {
  (want.ceil() as u32).clamp(1, 1 << 20)
}

/// Saturating fixed-point tile increment (`dust_lod.comp`): `v` counts, rounded, capped at 1e8 per
/// add so a handful of extreme samples cannot wrap the u32 sum.
#[inline]
pub fn tile_counts(v: f32) -> u32 {
  if !(v > 0.0) {
    return 0;
  }
  (v + 0.5).min(1e8) as u32
}

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

/// Stream predecessor of render cluster `r` of a tier: `r − S`, the same stream's previous time
/// sample (older). `None` for the first `S` clusters, after a stream break, or when the predecessor
/// is outside the tier's age band. Mirror of `dust_streak_pred`.
pub fn streak_pred(render: &[DustRenderCluster], r: usize) -> Option<usize> {
  let c = &render[r];
  let n = 1usize << render_stream_shift(c.age_id_dbeta_flux[1]);
  if r < n || stream_break(c.age_id_dbeta_flux[2]) {
    return None;
  }
  render_live(render[r - n].age_id_dbeta_flux[1]).then_some(r - n)
}

/// `mvp · (p, 1)` (column major)
#[inline]
fn mvp_mul(mvp: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
  let mut c = [0.0f32; 4];
  for (r, cr) in c.iter_mut().enumerate() {
    *cr = mvp[r] * p[0] + mvp[4 + r] * p[1] + mvp[8 + r] * p[2] + mvp[12 + r];
  }
  c
}

/// One Liang–Barsky plane `f0 + t·fd ≥ 0` on `[a, b]`; false when nothing is left (`dust_lb`).
#[inline]
fn lb_clip(f0: f32, fd: f32, a: &mut f32, b: &mut f32) -> bool {
  if fd == 0.0 {
    return f0 >= 0.0;
  }
  let t = -f0 / fd;
  if fd > 0.0 {
    if t > *a {
      *a = t;
    }
  } else if t < *b {
    *b = t;
  }
  *a <= *b
}

/// `[a, b] ⊂ [0, 1]` snapped outwards to a power-of-two grid of step in `(len/16, len/8]`: the dots
/// of a streak crossing the view edge stay put while the view moves within a grid cell, and
/// re-sample when it crosses one (`dust_snap_range`). Exact power-of-two arithmetic (bit ops for
/// `⌈log₂ len⌉`), bit-identical on the GPU.
pub fn snap_range(a: f32, b: f32) -> (f32, f32) {
  let len = (b - a).max(1e-30);
  let bits = len.to_bits();
  let ceil_log2 = ((bits >> 23) & 0xFF) as i32 - 127 + ((bits & 0x7F_FFFF) != 0) as i32;
  let e = (ceil_log2 - 3).clamp(-126, 0);
  let step = f32::from_bits(((e + 127) as u32) << 23);
  let floor = <f32 as FloatLike>::floor;
  (
    (floor(a / step) * step).max(0.0),
    (-floor(-b / step) * step).min(1.0),
  )
}

/// Visible part of a streak (see [`streak_clip`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreakClip {
  /// parameter range on `p + t·(q − p)`, snapped ([`snap_range`])
  pub t0: f32,
  pub t1: f32,
  /// on-screen length of the visible part (unsnapped), pixels
  pub len_px: f32,
  /// lateral 1σ width at its nearest point, pixels (at most the viewport height)
  pub width_px: f32,
}

/// Clips the streak `p → q` (particle-system metres, lateral σ `sp` at `p`, `sq` at `q`) to the
/// view: Liang–Barsky in homogeneous clip space against `w > 0` and `|x|, |y| ≤ (1 + m)·w`, margin
/// `m` = [`STREAK_MARGIN`] + 3σ (ndc, at the nearest visible point, ≤ [`STREAK_MARGIN_MAX`]).
/// `params` = (units per metre, P00, P11, ±2 / viewport height). `None`: nothing visible. Mirror of
/// `dust_streak_clip`.
pub fn streak_clip(
  mvp: &[f32; 16],
  p: [f32; 3],
  q: [f32; 3],
  sp: f32,
  sq: f32,
  params: [f32; 4],
) -> Option<StreakClip> {
  let [units, p00, p11, pxn] = params;
  let pxn = pxn.abs();
  if !(p00 > 0.0) || !(p11 > 0.0) || !(pxn > 0.0) {
    return None;
  }
  let c0 = mvp_mul(mvp, p);
  let c1 = mvp_mul(mvp, q);
  let d = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2], c1[3] - c0[3]];
  let (mut a, mut b) = (0.0f32, 1.0f32);
  let eps = 1e-6 * (c0[3].abs() + c1[3].abs());
  if !lb_clip(c0[3] - eps, d[3], &mut a, &mut b) {
    return None;
  }
  let w_at = |t: f32| c0[3] + t * d[3];
  let w_near = w_at(a).min(w_at(b));
  if !(w_near > 0.0) {
    return None;
  }
  let sigma = sp.max(sq) * units;
  let m = 1.0 + (STREAK_MARGIN + 3.0 * sigma * p00.max(p11) / w_near).min(STREAK_MARGIN_MAX);
  for k in 0..2 {
    if !lb_clip(m * c0[3] - c0[k], m * d[3] - d[k], &mut a, &mut b)
      || !lb_clip(m * c0[3] + c0[k], m * d[3] + d[k], &mut a, &mut b)
    {
      return None;
    }
  }
  let (wa, wb) = (w_at(a), w_at(b));
  let px_x = pxn * (p00 / p11);
  let dx = (c0[0] + b * d[0]) / wb - (c0[0] + a * d[0]) / wa;
  let dy = (c0[1] + b * d[1]) / wb - (c0[1] + a * d[1]) / wa;
  let len_px = <f32 as FloatLike>::sqrt((dx / px_x) * (dx / px_x) + (dy / pxn) * (dy / pxn));
  let width_px = (sigma * p11 / wa.min(wb) / pxn).min(2.0 / pxn);
  let (t0, t1) = snap_range(a, b);
  Some(StreakClip {
    t0,
    t1,
    len_px,
    width_px,
  })
}

/// golden-ratio conjugate: dot `c` of a streak sits at `fract(u_id + c·φ⁻¹)` of its visible range,
/// so any prefix `0..k` is evenly spread and the dots keep their place when the LOD changes `k`
pub const STREAK_PHI_INV: f32 = 0.618_034;

/// Parameter `t0 + (t1 − t0)·fract(u_id + c·φ⁻¹)` of streak dot `child` (`dust_streak_dot_t`)
#[inline]
pub fn streak_dot_t(t0: f32, t1: f32, id: u32, child: u32) -> f32 {
  let x = u01(pcg((id & CHILD_ID_MASK) ^ 0x1B87_3593)) + child as f32 * STREAK_PHI_INV;
  t0 + (t1 - t0) * (x - <f32 as FloatLike>::floor(x))
}

/// Unit vectors `(e1, e2)` completing unit `d` to an orthonormal basis (`dust_ortho`)
#[inline]
fn ortho_basis(d: [f32; 3]) -> ([f32; 3], [f32; 3]) {
  let up = if absf(d[2]) < 0.999 {
    [0.0, 0.0, 1.0]
  } else {
    [1.0, 0.0, 0.0]
  };
  let e1 = normalize3(cross3(up, d));
  (e1, cross3(d, e1))
}

/// Child `child` of the streak from `p` (this cluster, age `ap`, lateral σ `sp`) to `q` (its stream
/// predecessor) on the visible range `[t0, t1]` (`dust_streak_child`):
///
/// `lerp(p, q, t) + σ(t)·(N·e1 + N'·e2) + ½·Δβ·g·age(t)²·(anti-sun)`, `t = t0 + (t1 − t0)·u_c`
///
/// with `u_c = fract(u_id + c·φ⁻¹)` and `e1, e2 ⟂ q − p`: the dust emitted between the two time
/// samples of the stream, spread laterally by the stream's own dispersion. Stable per `(id, child)`
/// for a given range. `dbeta_field` is the raw β half-spread field (id and break bits ignored).
#[allow(clippy::too_many_arguments)]
pub fn streak_child(
  p: [f32; 3],
  q: [f32; 3],
  sp: f32,
  sq: f32,
  ap: f32,
  aq: f32,
  t0: f32,
  t1: f32,
  id: u32,
  child: u32,
  dbeta_field: f32,
  anti_sun_g: [f32; 4],
) -> [f32; 3] {
  let id = id & CHILD_ID_MASK;
  let t = streak_dot_t(t0, t1, id, child);
  let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
  let (e1, e2) = ortho_basis(normalize3(d));
  let h0 = pcg(id.wrapping_mul(MAX_CHILDREN_PER_CLUSTER).wrapping_add(child) ^ 0x7F4A_7C15);
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);
  let (n1, n2) = (gauss(u01(h0), u01(h1)), gauss(u01(h2), u01(h3)));
  let sigma = sp + (sq - sp) * t;
  let age = ap + (aq - ap) * t;
  let dbeta = absf(dbeta_field) * (2.0 * u01(h4) - 1.0);
  let s = 0.5 * dbeta * anti_sun_g[3] * age * age;
  [0, 1, 2].map(|k| p[k] + d[k] * t + sigma * (n1 * e1[k] + n2 * e2[k]) + s * anti_sun_g[k])
}

/// How the LOD draws one cluster (`dust_lod.comp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodPlan {
  /// children wanted at λ = 1 and drawn
  pub want: f32,
  pub k: u32,
  /// cross-section drawn: the flux times the visible fraction of a streak
  pub flux_in: f32,
  /// `(predecessor render index, visible part)`; `None`: a point spread
  pub streak: Option<(usize, StreakClip)>,
}

/// LOD decision for render cluster `i` (`flux > 0`): a streak towards its stream predecessor
/// ([`streak_pred`], clipped to the view, [`streak_want`] dots of the visible part) or, without
/// one, a point spread ([`lod_cluster_children`]). `None`: off screen. Mirror of `dust_lod.comp`.
pub fn lod_plan(
  render: &[DustRenderCluster],
  i: usize,
  pc: &DustLodPushConstants,
  sun_g: f32,
) -> Option<LodPlan> {
  let rc = &render[i];
  let flux = rc.age_id_dbeta_flux[3];
  let pos = [rc.pos_size[0], rc.pos_size[1], rc.pos_size[2]];
  if let Some(j) = streak_pred(render, i) {
    let pr = &render[j];
    let q = [pr.pos_size[0], pr.pos_size[1], pr.pos_size[2]];
    let (ep, eq) = (dust_extent(rc, sun_g), dust_extent(pr, sun_g));
    let c = streak_clip(&pc.mvp, pos, q, ep, eq, pc.params)?;
    let want = streak_want(c.len_px, c.width_px);
    return Some(LodPlan {
      want,
      k: lod_children(want, pc.lambda),
      flux_in: flux * (c.t1 - c.t0),
      streak: Some((j, c)),
    });
  }
  let spread = dust_extent(rc, sun_g);
  let clip = mvp_mul(&pc.mvp, pos);
  let k = lod_cluster_children(clip, spread, pc)?;
  let [units, _, p11, px_to_ndc_y] = pc.params;
  Some(LodPlan {
    want: lod_want(spread * units * p11 / clip[3] / px_to_ndc_y),
    k,
    flux_in: flux,
    streak: None,
  })
}

/// Footprint extent (m) of a render cluster's dust for the LOD: its lateral stream spread or, when
/// larger, the β spread of its size stratum `½·Δβ·g·age²` (anti-sunward) that its children are
/// scattered over. For old dust the β spread dominates: counted as the spread alone, the outer tail
/// asked for one dot per cluster and showed isolated dots (`far_1.rdc`) instead of the fan.
/// `sun_g`: solar gravity at the jet (`DustDrawState::anti_sun_g[3]`). Mirror of `dust_extent`.
pub fn dust_extent(rc: &DustRenderCluster, sun_g: f32) -> f32 {
  let [age, _, dbeta, _] = rc.age_id_dbeta_flux;
  rc.pos_size[3].max(0.5 * absf(dbeta) * sun_g * age * age)
}

/// One drawn child of a render cluster (mirror of `dust.vert`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChildSample {
  /// particle-system metres
  pub pos: [f32; 3],
  /// age of the dust there (s): interpolated along a streak ([`flow_factor`] input)
  pub age: f32,
  /// the cluster's tracer dot ([`TRACER_CHILD`]): drawn at [`TRACER_PX`], peak [`TRACER_LEVEL`]
  pub tracer: bool,
  /// drawn radius (px): [`splat_radius_px`] of the cluster's `want` and `k` at the frame's λ
  pub r_px: f32,
}

/// Child `child` of render cluster `i` (mirror of `dust.vert`): the tracer dot at the cluster's
/// exact position, a dot on its streak ([`streak_child`], the visible range rebuilt from the same
/// `mvp`) or around it ([`child_offset`]); its radius from `want` and `k` at the frame's `lambda`
/// (the LOD's [`DustLodResult::lambda`]).
pub fn child_sample(
  render: &[DustRenderCluster],
  i: usize,
  child: u32,
  mvp: &[f32; 16],
  params: [f32; 4],
  anti_sun_g: [f32; 4],
  lambda: f32,
) -> ChildSample {
  let radius = |want: f32| splat_radius_px(want, lod_children(want, lambda));
  let rc = &render[i];
  let pos = [rc.pos_size[0], rc.pos_size[1], rc.pos_size[2]];
  let [age, _, dbeta, _] = rc.age_id_dbeta_flux;
  if child == TRACER_CHILD {
    return ChildSample {
      pos,
      age,
      tracer: true,
      r_px: TRACER_PX,
    };
  }
  let id = child_id(dbeta);
  if let Some(j) = streak_pred(render, i) {
    let pr = &render[j];
    let q = [pr.pos_size[0], pr.pos_size[1], pr.pos_size[2]];
    // the LOD saw it: a disagreement can only come from rounding at the view edge
    let (ep, eq) = (
      dust_extent(rc, anti_sun_g[3]),
      dust_extent(pr, anti_sun_g[3]),
    );
    let (t0, t1, want) = streak_clip(mvp, pos, q, ep, eq, params)
      .map(|c| (c.t0, c.t1, streak_want(c.len_px, c.width_px)))
      .unwrap_or((0.0, 1.0, 1.0));
    let (sp, sq, aq) = (rc.pos_size[3], pr.pos_size[3], pr.age_id_dbeta_flux[0]);
    let t = streak_dot_t(t0, t1, id, child);
    return ChildSample {
      pos: streak_child(
        pos, q, sp, sq, age, aq, t0, t1, id, child, dbeta, anti_sun_g,
      ),
      age: age + (aq - age) * t,
      tracer: false,
      r_px: radius(want),
    };
  }
  let spread = rc.pos_size[3];
  let o = child_offset(id, child, spread, dbeta, age, anti_sun_g);
  let w = mvp_mul(mvp, pos)[3];
  let [units, _, p11, pxn] = params;
  let want = if w > 0.0 && pxn != 0.0 {
    lod_want(dust_extent(rc, anti_sun_g[3]) * units * p11 / w / pxn.abs())
  } else {
    1.0
  };
  ChildSample {
    pos: [pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]],
    age,
    tracer: false,
    r_px: radius(want),
  }
}

/// Position of child `child` of render cluster `i` ([`child_sample`])
pub fn child_position(
  render: &[DustRenderCluster],
  i: usize,
  child: u32,
  mvp: &[f32; 16],
  params: [f32; 4],
  anti_sun_g: [f32; 4],
) -> [f32; 3] {
  child_sample(render, i, child, mvp, params, anti_sun_g, 1.0).pos
}

/// CPU mirror of `dust_lod.comp` for one tier (also the CPU particle mode path): picks the
/// children of every on-screen cluster of `render` ([`lod_plan`]), rewrites its flux per child
/// (0 when dropped or off screen), appends `cluster | child << LOD_CLUSTER_BITS` to `list` up to
/// `budget` and scatters [`DUST_TILE_SAMPLES`] child samples of each cluster into `tiles`. The LOD
/// rewrites only the flux: a cluster reads its predecessor's position, spread, age and `y` word.
/// With [`DUST_VIEW_TRACERS`] in `flags`, a drawn tracer cluster ([`is_tracer`]) also gets the
/// instance [`TRACER_CHILD`] (one more entry of its reservation; not a tile sample).
pub fn lod_evaluate(
  render: &mut [DustRenderCluster],
  pc: &DustLodPushConstants,
  flags: u32,
  sun_g: f32,
  tiles: &mut [u32],
  list: &mut alloc::vec::Vec<u32>,
) -> DustLodResult {
  let mut out = DustLodResult::default();
  let [units, p00, p11, _] = pc.params;
  let live = (pc.live_count as usize).min(render.len());
  // pass A: this frame's demand, then the share that fits the budget (no feedback lag)
  let mut hist = [0u32; 2 * LOD_HIST_BINS];
  for i in 0..live {
    if !(render[i].age_id_dbeta_flux[3] > 0.0) {
      continue;
    }
    if let Some(plan) = lod_plan(render, i, pc, sun_g) {
      let d = lod_demand(plan.want);
      lod_hist_push(&mut hist, d);
      out.demand = out.demand.saturating_add(d.min(TRACER_CHILD));
      out.on_screen += 1;
    }
  }
  out.lambda = lod_lambda_from(pc.budget, &hist, out.on_screen, pc.lambda);
  let pc = &DustLodPushConstants {
    lambda: out.lambda,
    ..*pc
  };
  // pass B
  for i in 0..live {
    let flux = render[i].age_id_dbeta_flux[3];
    if !(flux > 0.0) {
      continue;
    }
    let Some(plan) = lod_plan(render, i, pc, sun_g) else {
      render[i].age_id_dbeta_flux[3] = 0.0;
      continue;
    };
    let k = plan.k;
    let tracer =
      flags & DUST_VIEW_TRACERS != 0 && is_tracer(child_id(render[i].age_id_dbeta_flux[2]));
    let n = k + tracer as u32;
    out.attempted = out.attempted.saturating_add(n);
    if out.instances as u64 + n as u64 > pc.budget as u64 {
      render[i].age_id_dbeta_flux[3] = 0.0;
    } else {
      for c in 0..k {
        list.push(i as u32 | (c << LOD_CLUSTER_BITS));
      }
      if tracer {
        list.push(i as u32 | (TRACER_CHILD << LOD_CLUSTER_BITS));
      }
      out.instances += n;
      render[i].age_id_dbeta_flux[3] = plan.flux_in / k as f32;
    }
    // white point: the cluster's cross-section over the tiles its first children land in
    // (isotropic part only: the shader has no room for the anti-sun direction)
    let rc = render[i];
    let pos = [rc.pos_size[0], rc.pos_size[1], rc.pos_size[2]];
    let [age, _, dbeta, _] = rc.age_id_dbeta_flux;
    let id = child_id(dbeta);
    for sidx in 0..DUST_TILE_SAMPLES {
      let p = match plan.streak {
        Some((j, c)) => {
          let pr = render[j];
          let q = [pr.pos_size[0], pr.pos_size[1], pr.pos_size[2]];
          let (sp, sq, aq) = (rc.pos_size[3], pr.pos_size[3], pr.age_id_dbeta_flux[0]);
          streak_child(pos, q, sp, sq, age, aq, c.t0, c.t1, id, sidx, 0.0, [0.0; 4])
        }
        None => {
          let o = child_offset(id, sidx, rc.pos_size[3], 0.0, age, [0.0; 4]);
          [pos[0] + o[0], pos[1] + o[1], pos[2] + o[2]]
        }
      };
      if let Some((t, v)) = tile_sample(
        mvp_mul(&pc.mvp, p),
        plan.flux_in,
        units,
        p00,
        p11,
        pc.tile_scale,
      ) {
        tiles[t] = tiles[t].wrapping_add(tile_counts(v));
      }
    }
  }
  out
}

/// Children of a cluster at clip position `clip` (`None` = off screen or behind the camera):
/// on screen within a 3·spread margin. Mirror of `dust_lod.comp`.
pub fn lod_cluster_children(clip: [f32; 4], spread: f32, pc: &DustLodPushConstants) -> Option<u32> {
  let [units, p00, p11, px_to_ndc_y] = pc.params;
  if !(clip[3] > 0.0) || !(p00 > 0.0) || !(p11 > 0.0) {
    return None;
  }
  let su = spread * units / clip[3];
  let (mx, my) = (3.0 * su * p00, 3.0 * su * p11);
  let (nx, ny) = (clip[0] / clip[3], clip[1] / clip[3]);
  if !(nx.abs() <= 1.0 + mx && ny.abs() <= 1.0 + my) {
    return None;
  }
  let spread_px = su * p11 / px_to_ndc_y;
  Some(lod_children(lod_want(spread_px), pc.lambda))
}

/// Tile of a child sample at clip `q` carrying `flux / DUST_TILE_SAMPLES` m², and its counts:
/// cross-section over the tile area at the sample's depth, times `tile_scale`.
pub fn tile_sample(
  q: [f32; 4],
  flux: f32,
  units: f32,
  p00: f32,
  p11: f32,
  tile_scale: f32,
) -> Option<(usize, f32)> {
  if !(q[3] > 0.0) {
    return None;
  }
  let (nx, ny) = (q[0] / q[3], q[1] / q[3]);
  let tx = ((nx * 0.5 + 0.5) * DUST_TILES_X as f32).floor();
  let ty = ((ny * 0.5 + 0.5) * DUST_TILES_Y as f32).floor();
  if !(tx >= 0.0 && tx < DUST_TILES_X as f32 && ty >= 0.0 && ty < DUST_TILES_Y as f32) {
    return None;
  }
  // tile area in m²: (2/TX)(2/TY) ndc², at w/(P·units) metres per ndc unit on each axis
  let m_per_ndc2 = (q[3] * q[3]) / (p00 * p11 * units * units);
  let area = 4.0 / DUST_TILE_COUNT as f32 * m_per_ndc2;
  let v = tile_scale * flux / DUST_TILE_SAMPLES as f32 / area.max(1e-30);
  Some((ty as usize * DUST_TILES_X as usize + tx as usize, v))
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
fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
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

/// size-stratum rotation of the batch's first time sample ([`BATCH_SIZE_ROTATION_UNIT`])
#[inline]
pub fn batch_size_rotation(batch: &DustBatch) -> u32 {
  batch.mass_params[3] as u32 / BATCH_SIZE_ROTATION_UNIT
}

/// Size stratum of cluster `j = i·S + s`: the stream's rank (a permutation keyed by
/// [`stream_key`]) advanced by one per time sample of the tier's grid, `(rank + r + i) mod S`
/// (`r`: [`batch_size_rotation`]).
///
/// Each time sample still covers every stratum once (the batch mass is exact), and over S samples
/// every cone cell emits every size, as a real jet does. A size fixed per stream tied each size to
/// one direction: old dust split into one island per stream, ~6× its width apart laterally
/// (`detached.rdc`). Consecutive samples of a stream are neighbouring strata (the streak spans
/// the sizes between them); the wrap from the largest to the smallest is a stream break
/// ([`stream_breaks_before`]).
pub fn stream_stratum(batch: &DustBatch, j: u32) -> u32 {
  let shift = batch_streams(batch).0;
  let n = 1u32 << shift;
  let s = j & (n - 1);
  let rank = permute_index(s, n, pcg(stream_key_bits(batch) ^ 0x2545_F491));
  rank.wrapping_add(batch_size_rotation(batch)).wrapping_add(j >> shift) & (n - 1)
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

/// [`batch_streams_word`] plus the provisional flag and the size rotation `rotation < 2^shift`
#[inline]
pub fn batch_word(shift: u32, break_before: bool, provisional: bool, rotation: u32) -> f32 {
  batch_streams_word(shift, break_before)
    + (if provisional { BATCH_PROVISIONAL_FLAG } else { 0 }
      + BATCH_SIZE_ROTATION_UNIT * (rotation & ((1 << shift) - 1))) as f32
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
  if i + 1 == samples && batch_provisional(batch) {
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
/// follows a missing window (first time sample), the stream's size wraps to the smallest stratum
/// ([`stream_stratum`]), or the jet site was dark in between ([`stream_dark_before`]).
pub fn stream_breaks_before(batch: &DustBatch, j: u32) -> bool {
  let (shift, break_before) = batch_streams(batch);
  (j >> shift == 0 && break_before)
    || (shift > 0 && stream_stratum(batch, j) == 0)
    || stream_dark_before(batch, j)
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
  let dark = omega * (dt - dt_prev) - (u - u_prev) * batch.lit[2];
  dark > 2.0 * core::f32::consts::PI * STREAM_BREAK_TURNS
}

/// Mass fraction of the size stratum `[p/n, (p+1)/n]` of the log-uniform size quantile under
/// `n(s) ∝ s^-q` (`size_params`: s_min, s_max, `e = 4 − q`): `(r^(e·u1) − r^(e·u0)) / (r^e − 1)`,
/// `r = s_max/s_min`. The strata of a time sample sum to 1: batch mass is conserved exactly.
pub fn size_stratum_mass(size_params: [f32; 4], p: u32, n: u32) -> f32 {
  let n = n.max(1);
  let r = size_params[1] / size_params[0];
  let e = size_params[2];
  let pow = <f32 as FloatLike>::pow;
  let full = pow(r, e) - 1.0;
  if !(absf(full) > 1e-6) {
    return 1.0 / n as f32;
  }
  let (u0, u1) = (p as f32 / n as f32, (p + 1) as f32 / n as f32);
  (pow(r, e * u1) - pow(r, e * u0)) / full
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission (mirrors dust_emit.comp)
// ─────────────────────────────────────────────────────────────────────────────

/// Builds cluster `j` (`0 ≤ j < batch.count`) of `batch`; it goes to ring slot
/// `(batch.first_index + j) & batch.ring_mask`. Cluster `j = i·S + s` is time sample `i` of stream
/// `s` ([`DUST_STREAMS`]): the stream fixes the direction in the jet cone (a Fibonacci lattice over
/// its solid angle) and the speed draw, its size stratum advances by one per time sample
/// ([`stream_stratum`]); the cluster adds small jitters. The stream's mass for the sample is its size stratum's share
/// ([`size_stratum_mass`]).
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
  let g0 = pcg(key ^ pcg(s ^ 0x5851_F42D));
  let g1 = pcg(g0);

  let t_start = Df::new(batch.comet_r_t_hi[3], batch.comet_r_t_lo[3]);
  let dt_in = emission_offset(batch, stream_time_u01(batch, j));
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

  // grain size: the stream's size stratum (log-uniform quantile), jittered within it
  let s_min = batch.size_params[0];
  let s_max = batch.size_params[1];
  let stratum = stream_stratum(batch, j);
  let u_s = (stratum as f32 + 0.5 + STREAM_SIZE_JITTER * (u01(h5) - 0.5)) / n as f32;
  let s_um = s_min * <f32 as FloatLike>::pow(s_max / s_min, u_s);
  let beta = batch.vel_params[3] / s_um;

  // the stratum's share of the time sample's mass
  let mass_g = batch.mass_params[0] / samples * size_stratum_mass(batch.size_params, stratum, n);

  // ejection speed: v_ref · sqrt(s_ref / s) · (1 + σ_rel (N_stream + jitter N)), clamped at 0
  let v_mean = batch.vel_params[0] * <f32 as FloatLike>::sqrt(batch.vel_params[2] / s_um);
  let g = gauss(u01(g0), u01(g1)) + STREAM_SPEED_JITTER * gauss(u01(h3), u01(h4));
  let v_ej = (v_mean * (1.0 + batch.vel_params[1] * g)).max(0.0);
  // lateral dispersion of the stream: STREAM_SIGMA_CELLS of its cone cell (angular radius
  // aperture/√S), at least CHILD_SIGMA_V_REL of the speed
  let sigma_lat = v_mean
    * (STREAM_SIGMA_CELLS * aperture / <f32 as FloatLike>::sqrt(n as f32)).max(CHILD_SIGMA_V_REL);

  let v0 = vc.add(&Df3::from_f32([
    dir[0] * v_ej + v_site[0],
    dir[1] * v_ej + v_site[1],
    dir[2] * v_ej + v_site[2],
  ]));

  // cross-section per gram: π s² / (4/3 π s³ ρ) = 3 / (4 ρ s)  [s in m, ρ in g/m³]
  let rho_g_m3 = batch.mass_params[1] * 1.0e6;
  let xsec_per_g = 3.0 / (4.0 * rho_g_m3 * s_um * 1.0e-6);
  // children spread β over the stream's size stratum: Δln s = ln(s_max/s_min) / S; the low
  // mantissa bits carry the child-pattern id, the sign bit the stream break
  let dbeta = beta * 0.5 * <f32 as FloatLike>::ln(s_max / s_min) / n as f32;
  let dbeta = pack_child_id(dbeta, pcg(h0 ^ 0x2C1B_3C6D));
  let dbeta = if stream_breaks_before(batch, j) {
    f32::from_bits(dbeta.to_bits() | STREAM_BREAK_BIT)
  } else {
    dbeta
  };

  DustCluster {
    r0_t0_hi: [rc.hi[0], rc.hi[1], rc.hi[2], t0.hi],
    r0_t0_lo: [rc.lo[0], rc.lo[1], rc.lo[2], t0.lo],
    v0_hi_beta: [v0.hi[0], v0.hi[1], v0.hi[2], beta],
    v0_lo_mass: [v0.lo[0], v0.lo[1], v0.lo[2], mass_g],
    misc: [sigma_lat, s_um, xsec_per_g, dbeta],
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
fn two_sum_one_minus(beta: f32) -> Df {
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
  /// emit windows before the start epoch (`jet_at` must cover negative times), so the history
  /// exists at t = 0
  pub prestart: bool,
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
}

impl DustHostState {
  pub fn new(capacity: u32) -> Self {
    Self::with_band(capacity, 0, TierBand::SINGLE, false)
  }

  /// A tier of `capacity` slots at `ring_base` drawing the ages of `band`.
  pub fn with_band(capacity: u32, ring_base: u32, band: TierBand, prestart: bool) -> Self {
    Self {
      ring: RingState::new(capacity),
      ring_base,
      tier: 0,
      band,
      prestart,
      upload_seq: 0,
      jet: None,
      ttl_s: 0.0,
      xsec_per_g_ref: 0.0,
      next_window: None,
      provisional: false,
      grid_s: 0.0,
      due_window: 0,
    }
  }

  /// the same tier with system index `tier` (window seeds, see [`window_seed`])
  pub fn with_tier(mut self, tier: u32) -> Self {
    self.tier = tier;
    self
  }

  /// forgets every cluster and the emission history (simulation reset)
  pub fn reset(&mut self) {
    *self = Self::with_band(self.ring.capacity, self.ring_base, self.band, self.prestart)
      .with_tier(self.tier);
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
  /// [`STREAM_MIN_SAMPLES`] time samples (64 on a 131 072-slot tier, 8 on 16 384, 2 on 4 096).
  pub fn stream_shift(&self) -> u32 {
    let n = (self.raw_clusters_per_window() / STREAM_MIN_SAMPLES).clamp(1, DUST_STREAMS);
    31 - n.leading_zeros()
  }

  /// Clusters per closed window: the steady-state budget spread over one TTL of windows, whole
  /// time samples of every stream.
  pub fn clusters_per_window(&self) -> u32 {
    let n = 1 << self.stream_shift();
    (self.raw_clusters_per_window() / n).max(1) * n
  }

  /// One logic tick of host-side emission at scaled time `t_now_s`, **deterministic in time**:
  /// emission happens on a fixed scaled-time grid of windows `[kΔ, (k+1)Δ)`,
  /// `Δ = ttl / WINDOWS_PER_TTL`, and window `k` always produces the same batch (jet state at
  /// `kΔ` from `jet_at`, mass `q(r)·lit time`, budgeted count, seed from `(tier, k)`). The dust at
  /// any epoch is a pure function of the parameters and the epoch, so playing, pausing, changing
  /// the speed or seeking all give the same result:
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
      return alloc::vec::Vec::new();
    };
    let cfg_now = cfg_at(&jet_now);
    self.ttl_s = cfg_now.ttl_s;
    self.xsec_per_g_ref = cfg_now.xsec_per_g_ref();
    let (min_age, max_age) = self.age_band_s(cfg_now.ttl_s);
    let dt_w = Self::window_len_s(max_age - min_age);
    if self.grid_s != dt_w {
      // another TTL changes the grid: the ring content no longer matches it
      let upload_seq = self.upload_seq;
      self.reset();
      self.upload_seq = upload_seq;
      self.grid_s = dt_w;
    }
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
    let k_open = if min_age > 0.0 {
      <f64 as FloatLike>::floor((t_now_s - min_age) / dt_w) as i64 + 1
    } else {
      <f64 as FloatLike>::floor(t_now_s / dt_w) as i64
    };
    let k_floor = if self.prestart { i64::MIN } else { 0 };
    let k_min = (<f64 as FloatLike>::floor((t_now_s - max_age) / dt_w) as i64).max(k_floor);
    let mut k = self
      .next_window
      .or_else(|| self.ring.batches.back().map(|b| b.window + 1))
      .unwrap_or(k_min)
      .max(k_min);
    while k < k_open && *budget > 0 {
      if let Some(b) = self.emit_window(k, k as f64 * dt_w, dt_w, dt_w, jet_at, cfg_at) {
        out.push(b);
      }
      k += 1;
      *budget -= 1;
    }
    self.next_window = Some(k);
    self.due_window = k_open;
    if k == k_open && min_age <= 0.0 {
      let t0 = k_open as f64 * dt_w;
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
    // size rotation of the first sample: the tier's samples are numbered on the window grid
    // (closed windows hold `full / n` samples each), so a stream steps one stratum per sample
    // across windows too
    let rotation = (k.rem_euclid(n as i64) * ((full as u32 / n) as i64 % n as i64)).rem_euclid(n as i64) as u32;
    desc.mass_params[3] = batch_word(shift, break_before, dur < dt_w, rotation);
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
}

/// The dust of one particle system: **age tiers** sharing one ring allocation. Tier `k` draws ages
/// in its [`TierBand`] from its own deterministic emission grid and sub-ring (`ring_base`), so old
/// dust gets coarser windows and larger grains: months to years of trail at the memory and
/// per-frame cost of one TTL. Every tier emits before the start epoch too (`prestart`), so the
/// history exists at t = 0.
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
        let t = DustHostState::with_band(caps[k].max(1), base, band, n > 1).with_tier(k as u32);
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
  }

  /// The nucleus rotation or a jet parameter changed: every window was emitted with the old ones
  /// (the dust history would show the old spin until it ages out, up to the oldest tier's band).
  /// The next tick resets and refills the tiers deterministically, youngest first, like a seek.
  pub fn request_reemit(&mut self) {
    self.reemit = true;
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
  }

  pub fn mark_submitted(&mut self, value: u64) {
    for t in &mut self.tiers {
      t.ring.mark_submitted(value);
    }
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
    let mut budget = MAX_WINDOWS_PER_TICK;
    let mut out = alloc::vec::Vec::new();
    for t in &mut self.tiers {
      for b in t.tick_budget(t_now_s, jet_at, cfg_at, &mut budget) {
        out.push((t.ring_base, b, self.upload_seq));
        self.upload_seq += 1;
      }
    }
    out
  }

  /// One draw state per non-empty tier, all with the system's exposure reference. (A mean cluster
  /// flux over the live clusters used to set the exposure: filling the old tiers with heavy
  /// clusters dimmed the whole system 10×, `initial_burst.rdc` vs `late_*.rdc`.)
  pub fn draw_states(&self) -> alloc::vec::Vec<DustDrawState> {
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
          caught_up: t.next_window.is_some_and(|k| k >= t.due_window),
        }
      })
      .collect()
  }
}

#[cfg(test)]
mod tests;
