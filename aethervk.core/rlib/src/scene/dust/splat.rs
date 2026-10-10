//! Dust v4 rendering: Gaussian packets, analytic capsule splats, fixed-point pyramid.
//!
//! A cluster stands for the grains of one stream (a velocity cell, a size stratum) emitted over
//! one time sample. Instead of sampling that distribution into dots per view (v3's LOD), v4
//! carries its **second moments** through the same Kepler map as the centre
//! ([`packet_moments`], mirror of `dust_propagate.comp`) and draws every packet by integrating its
//! projected density in closed form ([`capsule_tau`], mirror of `dust_splat.comp`), scattered into
//! a mip pyramid of fixed-point grids ([`splat_scatter`]) that the composite sums. Nothing is
//! sampled per view: the picture is `exposure · Σ_packets ∫ ρ_packet` exactly, so the shape of
//! the tail is a function of (parameters, epoch, camera) and the only view-dependent scalar is the
//! exposure.
//!
//! Moments. A cluster is the whole **cell** of the emission distribution it stands for, and its
//! kernel covers the cell on every axis (a kernel-density estimate whose bandwidth is the cell,
//! not the within-cell jitter: with the bandwidth below the cell spacing the picture is the
//! sampling lattice — speed shells near the nucleus, straight chords between size strata and
//! moiré at AU scale, `near_.rdc` / `the_problem.rdc`, 2026-10-09). Emission-space covariance:
//! the stream's lateral dispersion `σ_lat` (`misc.x`, 0.53 of the cone-cell spacing) across the
//! ejection axis and the jet's whole speed spread `σ_rad` (`misc.y`) along it, transported by
//! **secants over the real spreads** `a_i = K(v₀ + σ_i e_i) − r` (3 solves, `e₃` the ejection
//! direction) into `Σ_v = Σ a_i a_iᵀ`; the whole **size distribution** as a polyline along the
//! syndyne: [`SIZE_BINS`] bins of equal cross-section over `β/F .. β·F` (`F` =
//! `SIZE_RANGE_FACTOR`, equal spacing in `√β` for `n(s) ∝ s^-3.5`), the bin edges
//! `K(v₀ + e(√(β_j/β) − 1), μ(β_j))` (the smaller grains are faster, `v ∝ s^-½`;
//! [`SIZE_EDGES`] solves), each bin drawn as one packet uniform along its chord (a Gaussian of
//! σ = 0.55 of the chord, so neighbouring bins overlap into a smooth arc, ripple < 1 %) with
//! `Σ_v` scaled by the bin's speed ([`subpackets`]); and the time sample, uniform between the
//! cluster and its stream predecessor `r − S` (the capsule), with the width growing from the
//! predecessor's to the cluster's (time pieces, [`TIME_PIECES_MAX`]: the youngest samples are
//! cones, not sticks).
//!
//! Splat. The packet (mean `μ`, covariance `Σ`, segment `d` to its predecessor, flux) is projected
//! with the Jacobian of the perspective map at `μ` (EWA splatting, Zwicker et al. 2001; 3D
//! Gaussian splatting, Kerbl et al. 2023): `Σ' = J Σ Jᵀ + (0.5 px)²·I` (the pixel low-pass filter,
//! so a sub-pixel cluster deposits its whole optical depth into about one pixel: a 5-year trail is
//! visible from 3 AU without any dot budget). The optical depth at a pixel is the capsule integral
//! `τ(x) = A · ∫₀¹ G_Σ'(x − μ' − u d') du` (uniform along the segment, Gaussian across), closed
//! form with one `exp` and two `erf`; `A = exposure · flux / A_px` with `A_px` the pixel area at
//! the packet's depth, so `Σ_pixels τ = exposure · flux / A_px`: **surface brightness = optical
//! depth** at any zoom and distance.
//!
//! Pyramid. Level `ℓ` has texels of `2^ℓ` px; a splat goes to the lowest level where its smallest
//! screen σ is ≥ [`DUST_SPLAT_SIGMA_TEXELS`] texels (bilinear reconstruction error < 3 %), so the
//! texels per point packet are bounded (~600). Texels hold `(τ, τ·r, τ·g, τ·b, nearest depth)` in
//! u32: saturating fixed-point adds and an `atomicMin` of the depth bits, so the frame is
//! independent of the thread order up to the float rounding of each packet's own evaluation.
//!
//! Colour. The age rotates the stream colour's hue in OKLab (Ottosson 2020) from itself (at the
//! jet) to its complement (at [`DUST_AGE_HUE_SPAN`] TTL), log in age ([`age_color`]).
use super::{
  CHILD_ID_MASK, DUST_VIEW_FLOW, DUST_VIEW_TRACERS, DustCluster, DustFlowUniform, DustFrame,
  DustRenderCluster, TRACER_PX, child_id, consts,
  df::{Df, Df3},
  evaluate_cluster, flow_factor, is_tracer, kepler, qrot, render_live, render_stream_shift,
  stream_break, two_sum_one_minus,
};
use aethervk_oshal_rlib::math::FloatLike;

// ─── Constants (mirror of dust_common.glsl DUST_PYR_* / DUST_SPLAT_*) ───────

/// words of the pyramid header (level table, this frame's maxima and the view words)
pub const PYRAMID_HEADER_WORDS: u32 = 64;
/// words per pyramid texel: `τ`, `τ·r`, `τ·g`, `τ·b` (fixed point), nearest dust depth (AU, f32 bits)
pub const PYRAMID_TEXEL_WORDS: u32 = 5;
/// header words
pub const PYR_LEVELS: u32 = 0;
pub const PYR_WIDTH: u32 = 1;
pub const PYR_HEIGHT: u32 = 2;
/// largest packet peak optical depth per px² this frame (f32 bits, `atomicMax`): sets the next
/// frames' fixed-point unit
pub const PYR_TAU_MAX: u32 = 3;
/// view flags ([`DUST_VIEW_FLOW`], [`DUST_VIEW_TRACERS`])
pub const PYR_FLAGS: u32 = 4;
/// flow clock hi / lo (`DustFlowUniform`)
pub const PYR_T_HI: u32 = 5;
pub const PYR_T_LO: u32 = 6;
/// bit `ℓ` set when a splat touched level `ℓ` (`atomicOr`; the composite skips empty levels)
pub const PYR_LEVEL_MASK: u32 = 7;
/// tracer dot height in counts (f32 bits): `TRACER_LEVEL · white / unit`
pub const PYR_TRACER_COUNTS: u32 = 8;
/// the measurement grid `(offset, width, height)` ([`PyramidLayout::measure`])
pub const PYR_MEASURE: u32 = 12;
/// the fixed-point unit this frame's counts were written with (f32 bits, host-written; 0 when
/// unknown): what a dump reader needs to turn counts back into optical depth
pub const PYR_UNIT: u32 = 15;
/// per level `ℓ`: `(offset in words, width, height)` at `PYR_TABLE + 3ℓ`
pub const PYR_TABLE: u32 = 16;
pub const PYRAMID_MAX_LEVELS: u32 = (PYRAMID_HEADER_WORDS - PYR_TABLE) / 3;
/// a splat is scattered at the lowest level where its smallest screen σ is this many texels
pub const DUST_SPLAT_SIGMA_TEXELS: f32 = 2.0;
/// variance of the pixel low-pass filter added to every projected covariance (px²)
pub const DUST_PIXEL_FILTER_VAR: f32 = 0.25;
/// extent of a splat in σ (texels beyond it are skipped: `exp(−4.5)` = 1.1 % of the peak)
pub const DUST_SPLAT_SIGMAS: f32 = 3.0;
/// size bins of a cluster's polyline along the syndyne (equal cross-section each)
pub const SIZE_BINS: u32 = 16;
/// edges of the size polyline (`SIZE_BINS + 1` Kepler solves per cluster per frame)
pub const SIZE_EDGES: u32 = SIZE_BINS + 1;
/// variance along a bin's chord in units of the half-chord squared: σ = 1.1 half-chords = 0.55 of
/// the chord, so the sum of the bins is a smooth arc (Gaussians spaced `d` with σ = 0.55 d ripple
/// `2·exp(−2π²σ²/d²)` ≈ 0.5 %); the uniform bin's own variance would be a third (ripple 39 %)
pub const CHORD_VAR_FACTOR: f32 = 1.21;
/// pieces a capsule is cut into when its width grows along the segment (the predecessor is older
/// and wider): each piece carries the interpolated width
pub const TIME_PIECES_MAX: u32 = 4;
/// Sub-capsules a size bin is swept into between its own chord and the predecessor's same chord
/// (the time cell of the bin is that quadrilateral, not a line): one per [`TIME_SWEEP_STEP_PX`]
/// of the larger edge displacement on screen, at most this many. From 0.3 AU out the oldest
/// tier's consecutive polylines are pixels apart laterally at their far ends while the packets
/// are a few hundredths of a pixel wide, and a single Gaussian along the mid-chord time segment
/// left the quadrilateral's wide end uncovered: the sample striations of the far tail.
pub const TIME_SWEEP_MAX: u32 = 4;
/// screen displacement (px) between consecutive polylines per sweep sub-capsule
pub const TIME_SWEEP_STEP_PX: f32 = 1.0;
/// floor of the projected covariance's determinant in units of its largest eigenvalue squared:
/// `det = cov.xx·cov.yy − cov.xy²` cancels catastrophically in f32 for a packet thousands of times
/// longer than wide (an old trail bin seen from near the nucleus: a 10⁶ px chord, a 10³ px
/// width), and a garbage determinant near zero made `amp` explode to 10¹⁵ (the zoom series:
/// black frames, a white point of 10¹⁷). The floor keeps the lateral width ≥ 10⁻³ of the length,
/// below any physical packet's aspect, and the quadratic form stays consistent with it
pub const DUST_DET_ANISO_FLOOR: f32 = 1e-6;
/// a size polyline whose screen extent (farthest edge from the mean) is below this many pixels is
/// drawn as one pooled packet: below the pixel filter its shape is invisible, and a far view of a
/// 262 144-cluster system costs 1 capsule per cluster instead of SIZE_BINS
pub const DUST_MERGE_PX: f32 = 1.0;
/// largest count one texel add may contribute (the saturating sum protects the rest)
pub const DUST_COUNT_MAX_PER_ADD: u32 = 1 << 24;
/// texels one packet may visit at its level before it is moved to a coarser one (a bound on the
/// work of one thread: a long thin streak across the view stays a band, never a screen)
pub const DUST_SPLAT_MAX_TEXELS: u32 = 16384;
/// pyramid level the white point and the auto softening are measured on (texels of 16 px, the
/// former 64 × 36 tile grid at 1080p)
pub const DUST_WHITE_LEVEL: u32 = 4;
/// age at which the hue ramp reaches the complement, in TTL ([`DUST_AGE_HUE_TAU_S`]): the oldest
/// tier's end
pub const DUST_AGE_HUE_SPAN: f32 = 64.0;
/// age scale of the hue ramp (s): the jet TTL (30 d)
pub const DUST_AGE_HUE_TAU_S: f32 = 30.0 * 86400.0;
/// metres per AU (`dust::AU_M` is f64)
const AU_M_F32: f32 = 1.495_978_707e11;

// ─── Layouts ─────────────────────────────────────────────────────────────────

/// Per-frame second moments of one cluster (`dust_propagate.comp` → `dust_splat.comp`). 320 bytes.
/// The time-sample term is not stored: the splat pass reads the predecessor's mean and
/// covariance (`DustRenderCluster` word `y`, [`super::streak_pred`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustMoments {
  /// particle-system local mean of the reference grain (m), `w` flux (cross-section m², faded;
  /// 0 = culled)
  pub mean_flux: [f32; 4],
  /// covariance of the velocity dispersion `Σ_v` (m²) of the reference grain: xx, xy, xz, yy
  pub cov_a: [f32; 4],
  /// yz, zz, `z` age (s), `w` child-pattern id (u32 bits)
  pub cov_b_age_id: [f32; 4],
  /// the size polyline: `xyz` local position (m) of size-bin edge `j` (`β_j` from `β/F` to
  /// `β·F`, equal cross-section bins), `w` its speed factor `√(β_j/β)` (scales the dispersion)
  pub edges: [[f32; 4]; SIZE_EDGES as usize],
}
const _: () = assert!(core::mem::size_of::<DustMoments>() == 320);

impl Default for DustMoments {
  fn default() -> Self {
    Self {
      mean_flux: [0.0; 4],
      cov_a: [0.0; 4],
      cov_b_age_id: [0.0; 4],
      edges: [[0.0; 4]; SIZE_EDGES as usize],
    }
  }
}

impl DustMoments {
  pub fn culled() -> Self {
    Self::default()
  }
  pub fn flux(&self) -> f32 {
    self.mean_flux[3]
  }
  pub fn age(&self) -> f32 {
    self.cov_b_age_id[2]
  }
  pub fn id(&self) -> u32 {
    self.cov_b_age_id[3].to_bits()
  }
  pub fn mean(&self) -> [f32; 3] {
    [self.mean_flux[0], self.mean_flux[1], self.mean_flux[2]]
  }
  /// `Σ_v` as (xx, xy, xz, yy, yz, zz)
  pub fn cov_v(&self) -> [f32; 6] {
    [
      self.cov_a[0],
      self.cov_a[1],
      self.cov_a[2],
      self.cov_a[3],
      self.cov_b_age_id[0],
      self.cov_b_age_id[1],
    ]
  }
  /// local position of size-bin edge `j`
  pub fn edge(&self, j: usize) -> [f32; 3] {
    [self.edges[j][0], self.edges[j][1], self.edges[j][2]]
  }
  /// speed factor `√(β_j/β)` of edge `j`
  pub fn edge_factor(&self, j: usize) -> f32 {
    self.edges[j][3]
  }
  /// `tr(Σ_v)/3`: the mean variance of the velocity dispersion (m²)
  pub fn mean_var(&self) -> f32 {
    (self.cov_a[0] + self.cov_a[3] + self.cov_b_age_id[1]) / 3.0
  }
}

/// `√β_j` of size-bin edge `j` for a reference `β` and a size range `β/f .. β·f`: equal
/// cross-section bins of `n(s) ∝ s^-q` have their cumulative cross-section ∝ `s^(3−q)` ∝
/// `β^(q−3)`, i.e. `√β` for `q = 3.5` ([`super::SIZE_POWER_Q`]). Mirror of
/// `dust_size_edge_sqrt_beta`.
pub fn size_edge_sqrt_beta(beta: f32, f: f32, j: u32) -> f32 {
  let f = f.max(1.0);
  let sq = <f32 as FloatLike>::sqrt;
  let lo = sq(beta.max(0.0) / f);
  let hi = sq(beta.max(0.0) * f);
  lo + (hi - lo) * (j as f32 / SIZE_BINS as f32)
}

/// The cluster's size range factor `F = s_max/s_ref = s_ref/s_min` from `misc.w` (the half
/// log-size range with the child id in its low mantissa bits and the break in its sign bit;
/// [`super::SIZE_RANGE_FACTOR`] for the app's distributions). Mirror of `dust_size_range_factor`.
pub fn size_range_factor(c: &DustCluster) -> f32 {
  let bits = c.misc[3].to_bits() & !CHILD_ID_MASK & 0x7FFF_FFFF;
  let half_ln = f32::from_bits(bits);
  if half_ln.is_finite() && half_ln >= 0.0 {
    <f32 as FloatLike>::exp(half_ln)
  } else {
    1.0
  }
}

/// `dust_splat.comp` push constants. 128 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustSplatPushConstants {
  /// `DustMomentsBuffer` of the tier (compact, index = live-range offset)
  pub moments: u64,
  /// `DustRenderBuffer` of the tier (predecessor validity words)
  pub render: u64,
  /// the view's pyramid (header + levels)
  pub pyramid: u64,
  pub live_count: u32,
  /// view flags ([`DUST_VIEW_FLOW`], [`DUST_VIEW_TRACERS`])
  pub flags: u32,
  /// raw dust exposure: gain / reference optical depth (the white point is applied by the composite)
  pub exposure: f32,
  /// counts per unit of exposure-scaled optical depth × px² (1 / fixed-point unit)
  pub inv_unit: f32,
  /// stream colour, 8 bits per channel (`r | g << 8 | b << 16`)
  pub color: u32,
  /// layer units per metre (the mvp maps local metres × this to clip)
  pub units_per_m: f32,
  /// particle-system local metres → clip
  pub mvp: [f32; 16],
  /// eye position in particle-system local metres (depth of the packets), `w` spare
  pub eye_local: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustSplatPushConstants>() == 128);

/// Where the levels of a pyramid of a `width × height` viewport live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyramidLayout {
  pub width: u32,
  pub height: u32,
  /// per level: `(offset in words, width, height)`
  pub levels: alloc::vec::Vec<(u32, u32, u32)>,
  /// the white-point measurement grid `(offset in words, width, height)`: texels of
  /// `2^DUST_WHITE_LEVEL` px into which **every** packet scatters its optical depth whatever its
  /// own level (the levels alone hold only the packets that landed on them), read back per frame
  pub measure: (u32, u32, u32),
  /// words of the whole buffer (header + levels + measure grid)
  pub total_words: u32,
}

impl PyramidLayout {
  pub fn new(width: u32, height: u32) -> Self {
    let (width, height) = (width.max(1), height.max(1));
    let mut levels = alloc::vec::Vec::new();
    let mut off = PYRAMID_HEADER_WORDS;
    let (mut w, mut h) = (width, height);
    loop {
      levels.push((off, w, h));
      off += w * h * PYRAMID_TEXEL_WORDS;
      if (w == 1 && h == 1) || levels.len() as u32 == PYRAMID_MAX_LEVELS {
        break;
      }
      w = w.div_ceil(2);
      h = h.div_ceil(2);
    }
    let ms = 1u32 << DUST_WHITE_LEVEL;
    let (mw, mh) = (width.div_ceil(ms), height.div_ceil(ms));
    let measure = (off, mw, mh);
    off += mw * mh * PYRAMID_TEXEL_WORDS;
    Self {
      width,
      height,
      levels,
      measure,
      total_words: off,
    }
  }

  pub fn level_count(&self) -> u32 {
    self.levels.len() as u32
  }

  /// the header words (level table); the per-frame words are written by the host separately
  pub fn header(&self) -> [u32; PYRAMID_HEADER_WORDS as usize] {
    let mut h = [0u32; PYRAMID_HEADER_WORDS as usize];
    h[PYR_LEVELS as usize] = self.level_count();
    h[PYR_WIDTH as usize] = self.width;
    h[PYR_HEIGHT as usize] = self.height;
    h[PYR_MEASURE as usize] = self.measure.0;
    h[PYR_MEASURE as usize + 1] = self.measure.1;
    h[PYR_MEASURE as usize + 2] = self.measure.2;
    for (l, &(off, w, hh)) in self.levels.iter().enumerate() {
      h[PYR_TABLE as usize + 3 * l] = off;
      h[PYR_TABLE as usize + 3 * l + 1] = w;
      h[PYR_TABLE as usize + 3 * l + 2] = hh;
    }
    h
  }

  /// Level whose texels are ≤ `σ_px / DUST_SPLAT_SIGMA_TEXELS` px (the splat covers ≥ 2 texels per
  /// σ), the last level at most. Mirror of `dust_splat_level`.
  pub fn level_for_sigma(&self, sigma_px: f32) -> u32 {
    splat_level(sigma_px, self.level_count())
  }
}

/// Mirror of `dust_splat_level`: `clamp(⌊log₂(σ / DUST_SPLAT_SIGMA_TEXELS)⌋, 0, levels − 1)`.
pub fn splat_level(sigma_px: f32, levels: u32) -> u32 {
  let r = sigma_px / DUST_SPLAT_SIGMA_TEXELS;
  if !(r > 1.0) {
    return 0;
  }
  let l = <f32 as FloatLike>::floor(<f32 as FloatLike>::ln(r) * core::f32::consts::LOG2_E);
  (l.max(0.0) as u32).min(levels.saturating_sub(1))
}

// ─── Moments (mirror of dust_propagate.comp) ───────────────────────────────

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
  [a[0] * s, a[1] * s, a[2] * s]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn fl(x: f32) -> f32 {
  <f32 as FloatLike>::floor(x)
}
#[inline]
fn ce(x: f32) -> f32 {
  -<f32 as FloatLike>::floor(-x)
}
#[inline]
fn fabs(x: f32) -> f32 {
  if x < 0.0 { -x } else { x }
}

/// Secant difference `K(r0, v0 + dv, mu, dt) − r` as f32 (the df64 difference, then rounded).
fn secant_v(r0: &Df3, v0: &Df3, dv: [f32; 3], mu: Df, dt: Df, r: &Df3) -> [f32; 3] {
  let v = v0.add(&Df3::from_f32(dv));
  let (rr, _) = kepler::propagate(r0, &v, mu, dt);
  rr.sub(r).to_f32()
}

/// Evaluates the cluster in ring slot `slot` at `frame`'s time: the render cluster (as
/// [`evaluate_cluster`]) and its second moments: `Σ_v` from three secants (two lateral axes with
/// `σ_lat`, the ejection axis with `σ_rad`) and the size polyline (`SIZE_EDGES` solves, each at
/// the bin edge's β and speed). `1 + 3 + SIZE_EDGES` Kepler solves per live cluster.
pub fn packet_moments(
  c: &DustCluster,
  slot: u32,
  frame: &DustFrame,
) -> (DustRenderCluster, DustMoments) {
  let rc = evaluate_cluster(c, slot, frame);
  if !(rc.age_id_dbeta_flux[3] > 0.0) {
    return (rc, DustMoments::culled());
  }
  let t_now = Df::new(frame.ps_r_t_hi[3], frame.ps_r_t_lo[3]);
  let age = t_now.sub(c.t0());
  let age_f = age.hi + age.lo;
  let beta = c.beta();
  let mu = consts::SUN_MU.mul(two_sum_one_minus(beta));
  let (r0, v0) = (c.r0(), c.v0());
  let (r, _) = kepler::propagate(&r0, &v0, mu, age);
  let rot = frame.rot_inv;
  // the dispersion frame: e3 along the ejection, e1/e2 across it
  let (e1, e2, e3) = dispersion_frame(c.eject());
  let (sigma_lat, sigma_rad) = (c.sigma_lat().max(0.0), c.sigma_rad().max(0.0));
  let a0 = qrot(rot, secant_v(&r0, &v0, scale3(e1, sigma_lat), mu, age, &r));
  let a1 = qrot(rot, secant_v(&r0, &v0, scale3(e2, sigma_lat), mu, age, &r));
  let a2 = qrot(rot, secant_v(&r0, &v0, scale3(e3, sigma_rad), mu, age, &r));
  let cov = |i: usize, j: usize| a0[i] * a0[j] + a1[i] * a1[j] + a2[i] * a2[j];
  // the size polyline: every bin edge at its own β and speed, relative to the particle system
  let ps = Df3 {
    hi: [frame.ps_r_t_hi[0], frame.ps_r_t_hi[1], frame.ps_r_t_hi[2]],
    lo: [frame.ps_r_t_lo[0], frame.ps_r_t_lo[1], frame.ps_r_t_lo[2]],
  };
  let sqrt_beta = <f32 as FloatLike>::sqrt(beta.max(0.0));
  let eject = c.eject();
  let range = size_range_factor(c);
  let mut edges = [[0.0f32; 4]; SIZE_EDGES as usize];
  for (j, e) in edges.iter_mut().enumerate() {
    let sb = size_edge_sqrt_beta(beta, range, j as u32);
    let beta_j = sb * sb;
    let factor = if sqrt_beta > 0.0 { sb / sqrt_beta } else { 1.0 };
    let v_j = v0.add(&Df3::from_f32(scale3(eject, factor - 1.0)));
    let mu_j = consts::SUN_MU.mul(two_sum_one_minus(beta_j));
    let (r_j, _) = kepler::propagate(&r0, &v_j, mu_j, age);
    let local = qrot(rot, r_j.sub(&ps).to_f32());
    *e = [local[0], local[1], local[2], factor];
  }
  let m = DustMoments {
    mean_flux: [
      rc.pos_size[0],
      rc.pos_size[1],
      rc.pos_size[2],
      rc.age_id_dbeta_flux[3],
    ],
    cov_a: [cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 1)],
    cov_b_age_id: [
      cov(1, 2),
      cov(2, 2),
      age_f,
      f32::from_bits(child_id(c.misc[3])),
    ],
    edges,
  };
  (rc, m)
}

/// Orthonormal frame `(e1, e2, e3)` with `e3` along `eject` (mirror of `dust_dispersion_frame`):
/// the root axes when the ejection is zero.
pub fn dispersion_frame(eject: [f32; 3]) -> ([f32; 3], [f32; 3], [f32; 3]) {
  let n = <f32 as FloatLike>::sqrt(dot3(eject, eject));
  if !(n > 0.0) {
    return ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
  }
  let e3 = scale3(eject, 1.0 / n);
  // the root axis least aligned with e3
  let a = if fabs(e3[0]) < 0.9 {
    [1.0, 0.0, 0.0]
  } else {
    [0.0, 1.0, 0.0]
  };
  let e1 = cross3(a, e3);
  let n1 = <f32 as FloatLike>::sqrt(dot3(e1, e1));
  let e1 = scale3(e1, 1.0 / n1);
  let e2 = cross3(e3, e1);
  (e1, e2, e3)
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

/// A packet to splat: mean (local m), covariance (xx, xy, xz, yy, yz, zz), segment to the
/// predecessor (m, zero without one), flux.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Packet {
  /// start of the segment (local m)
  pub mean: [f32; 3],
  pub cov: [f32; 6],
  /// the segment the packet is uniform along (a size-bin chord, or the time segment of a
  /// pooled packet), from `mean`
  pub seg: [f32; 3],
  pub flux: f32,
  /// `[start, end]` open: the capsule continues into the neighbouring bin there (no Gaussian
  /// end cap, no texels beyond the endpoint), so a polyline of bins is one continuous line
  pub open: [bool; 2],
  /// relative density at the start and the end of the segment, mean 1 (`ρ₀ + ρ₁ = 2`), linear
  /// in between: a size bin's grains are uniform in `√β` while their position along the chord
  /// is linear in `β`, so the density along the chord falls as `1/√β` ([`bin_density_ends`]);
  /// a uniform capsule stair-stepped the syndyne by 30–40 % at the large-grain end
  pub rho: [f32; 2],
}

/// The relative density at the two ends of a size bin's chord whose edges have the speed
/// factors `f0 ≤ f1` (`√(β_j/β)`): `(f0 + f1) / (2 f0)` and `(f0 + f1) / (2 f1)`, mean 1 over the
/// chord (position ∝ β, cross-section uniform in √β). Mirror of `dust_bin_density_ends`.
pub fn bin_density_ends(f0: f32, f1: f32) -> [f32; 2] {
  let (f0, f1) = (f0.max(1e-6), f1.max(1e-6));
  let s = f0 + f1;
  [s / (2.0 * f0), s / (2.0 * f1)]
}

/// The packets of a cluster (mirror of `dust_subpackets`): one per size bin of the polyline,
/// uniform along its chord (a Gaussian of [`CHORD_VAR_FACTOR`] half-chords² along it) with
/// `Σ_v` scaled by the bin's speed factor, `1/SIZE_BINS` of the flux each; and when the stream
/// predecessor `pred` is wider (older), every bin is cut into `n_t ≤ TIME_PIECES_MAX` pieces along
/// the segment, each with the width interpolated between the cluster's and the predecessor's
/// (the capsule is a cone, not a stick of the newest width).
pub fn subpackets(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
) -> alloc::vec::Vec<Packet> {
  subpackets_merged(m, seg, pred, false)
}

/// The packet of size bin `b` (mirror of `dust_bin_packet`): midpoint of its edges, `Σ_v` scaled
/// by the bin's speed factor plus [`CHORD_VAR_FACTOR`] half-chords² along the chord, `1/SIZE_BINS`
/// of the flux.
pub fn bin_packet(m: &DustMoments, b: usize, seg: [f32; 3]) -> Packet {
  let cv = m.cov_v();
  let (a, c) = (m.edge(b), m.edge(b + 1));
  let mid = scale3(add3(a, c), 0.5);
  let h = scale3(sub3(c, a), 0.5);
  let f = 0.5 * (m.edge_factor(b) + m.edge_factor(b + 1));
  let f2 = f * f;
  let k = CHORD_VAR_FACTOR;
  Packet {
    mean: mid,
    cov: [
      f2 * cv[0] + k * h[0] * h[0],
      f2 * cv[1] + k * h[0] * h[1],
      f2 * cv[2] + k * h[0] * h[2],
      f2 * cv[3] + k * h[1] * h[1],
      f2 * cv[4] + k * h[1] * h[2],
      f2 * cv[5] + k * h[2] * h[2],
    ],
    seg,
    flux: m.flux() / SIZE_BINS as f32,
    open: [false, false],
    rho: [1.0, 1.0],
  }
}

/// The whole polyline as one packet (mirror of `dust_pooled_packet`): the bins' mean, their mean
/// covariance plus the scatter of their midpoints (within + between), the whole flux. Used when
/// the polyline is below [`DUST_MERGE_PX`] on screen.
pub fn pooled_packet(m: &DustMoments, seg: [f32; 3]) -> Packet {
  let n = SIZE_BINS as f32;
  let mut mean = [0.0f32; 3];
  let mut cov = [0.0f32; 6];
  let bins: alloc::vec::Vec<Packet> =
    (0..SIZE_BINS as usize).map(|b| bin_packet(m, b, seg)).collect();
  for p in &bins {
    mean = add3(mean, scale3(p.mean, 1.0 / n));
  }
  for p in &bins {
    let d = sub3(p.mean, mean);
    let idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    for (k, (i, j)) in idx.iter().enumerate() {
      cov[k] += (p.cov[k] + d[*i] * d[*j]) / n;
    }
  }
  Packet {
    mean,
    cov,
    seg,
    flux: m.flux(),
    open: [false, false],
    rho: [1.0, 1.0],
  }
}

/// The mean covariance of the bins alone (`Σ_v` scaled per bin plus each chord's Gaussian),
/// without the scatter of the bin midpoints: the width a predecessor hands to the time pieces
/// of a sample too young to have a polyline of its own. With the scatter, a 53-min predecessor
/// whose sizes span 18 km handed one fat Gaussian to a just-emitted sample, centred on the
/// nucleus (the halo behind the nucleus of the 5-km frame). Mirror of `dust_pooled_within`.
pub fn pooled_within_cov(m: &DustMoments, seg: [f32; 3]) -> [f32; 6] {
  let n = SIZE_BINS as f32;
  let mut cov = [0.0f32; 6];
  for b in 0..SIZE_BINS as usize {
    let p = bin_packet(m, b, seg);
    for k in 0..6 {
      cov[k] += p.cov[k] / n;
    }
  }
  cov
}

/// The packet drawn for size bin `b` (mirror of `dust_bin_chord`): **uniform along its chord**
/// (`mean` = the bin's start edge, `seg` = the chord), open towards its neighbours, with
/// `Σ_v` scaled by the bin's speed factor plus the time segment `seg_t` of the bin (to the same
/// bin of the predecessor) as a Gaussian of [`CHORD_VAR_FACTOR`] half-segments² along it, and
/// `1/SIZE_BINS` of the flux. The chord is exact (no Gaussian smear, no end caps between bins):
/// the polyline costs its own length in texels.
pub fn bin_chord_packet(m: &DustMoments, b: usize, seg_t: [f32; 3]) -> Packet {
  let cv = m.cov_v();
  let (a, c) = (m.edge(b), m.edge(b + 1));
  let f = 0.5 * (m.edge_factor(b) + m.edge_factor(b + 1));
  let f2 = f * f;
  let k = CHORD_VAR_FACTOR;
  let h = scale3(seg_t, 0.5);
  Packet {
    mean: a,
    cov: [
      f2 * cv[0] + k * h[0] * h[0],
      f2 * cv[1] + k * h[0] * h[1],
      f2 * cv[2] + k * h[0] * h[2],
      f2 * cv[3] + k * h[1] * h[1],
      f2 * cv[4] + k * h[1] * h[2],
      f2 * cv[5] + k * h[2] * h[2],
    ],
    seg: sub3(c, a),
    flux: m.flux() / SIZE_BINS as f32,
    open: [b > 0, b + 1 < SIZE_BINS as usize],
    rho: bin_density_ends(m.edge_factor(b), m.edge_factor(b + 1)),
  }
}

/// The time segment of a sample is treated as an **arc about the jet** when it is at least this
/// fraction of the predecessor's distance from the jet: young dust near the source, where the
/// nucleus turns between two samples (13° per 28-min sample at 12.4 h) and the dust emitted in
/// between lies on a spiral from the sample's direction to the predecessor's, not on the chord
/// (the chord from the nucleus to the predecessor sat at the predecessor's direction all along,
/// a third of the cone's width off at 5 km). Far from the source the segment is a sliver of the
/// radius and a straight segment is exact.
pub const ARC_MIN_FRACTION: f32 = 0.1;
/// ... and only within this many sample intervals of the emission: an old cluster whose mean
/// happens to pass near the jet (the trail runs through the comet) must not have its pieces
/// swung about the jet by the random angle between two of its samples' directions.
pub const ARC_MAX_AGE_SAMPLES: f32 = 8.0;

/// Rodrigues rotation of `v` by `angle` about the unit `axis`
fn rotate3(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
  let (s, c) = (
    <f32 as FloatLike>::sin(angle),
    <f32 as FloatLike>::cos(angle),
  );
  let k = cross3(axis, v);
  let d = dot3(axis, v);
  [
    v[0] * c + k[0] * s + axis[0] * d * (1.0 - c),
    v[1] * c + k[1] * s + axis[1] * d * (1.0 - c),
    v[2] * c + k[2] * s + axis[2] * d * (1.0 - c),
  ]
}

/// The directions (from the jet) of a sample and its predecessor when their time segment is an
/// arc ([`ARC_MIN_FRACTION`]): `None` for a straight segment. A sample too close to the jet to
/// have a direction of its own (the one pinned at "now") takes the rotation seen from the
/// predecessor's predecessor `pp`, continued one step. Mirror of `dust_arc_dirs`.
pub fn arc_dirs(
  own: [f32; 3],
  pred: [f32; 3],
  pp: Option<[f32; 3]>,
) -> Option<([f32; 3], [f32; 3])> {
  let sq = <f32 as FloatLike>::sqrt;
  let rp = sq(dot3(pred, pred));
  let seg = sub3(pred, own);
  if !(rp > 0.0) || sq(dot3(seg, seg)) < ARC_MIN_FRACTION * rp {
    return None;
  }
  let d_p = scale3(pred, 1.0 / rp);
  let ro = sq(dot3(own, own));
  let d_o = if ro > ARC_MIN_FRACTION * rp {
    scale3(own, 1.0 / ro)
  } else {
    match pp {
      Some(q) => {
        let rq = sq(dot3(q, q));
        if !(rq > 0.0) {
          return Some((d_p, d_p));
        }
        let d_q = scale3(q, 1.0 / rq);
        let ax = cross3(d_q, d_p);
        let sn = sq(dot3(ax, ax));
        if sn < 1e-6 {
          return Some((d_p, d_p));
        }
        let angle = <f32 as FloatLike>::atan2(sn, dot3(d_q, d_p));
        rotate3(d_p, scale3(ax, 1.0 / sn), angle)
      }
      None => d_p,
    }
  };
  Some((d_o, d_p))
}

/// The point at fraction `u` of the arc from `own` (u = 0) to `pred` (u = 1): the direction
/// interpolated between `d_o` and `d_p`, the distance from the jet linearly. Mirror of
/// `dust_arc_point`.
pub fn arc_point(own: [f32; 3], pred: [f32; 3], d_o: [f32; 3], d_p: [f32; 3], u: f32) -> [f32; 3] {
  let sq = <f32 as FloatLike>::sqrt;
  let (ro, rp) = (sq(dot3(own, own)), sq(dot3(pred, pred)));
  let r = ro + u * (rp - ro);
  let d = add3(scale3(d_o, 1.0 - u), scale3(d_p, u));
  let n = sq(dot3(d, d)).max(1e-12);
  scale3(d, r / n)
}

/// Sweep count of a bin ([`TIME_SWEEP_MAX`]): the larger displacement of its two edges to the
/// predecessor's, on screen, per [`TIME_SWEEP_STEP_PX`]; 1 without a predecessor or a projection.
pub fn time_sweep(
  m: &DustMoments,
  pred: Option<&DustMoments>,
  b: usize,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> u32 {
  let (Some(p), Some(px)) = (pred, px) else {
    return 1;
  };
  let mut gap = 0.0f32;
  for e in [b, b + 1] {
    if let (Some(a), Some(c)) = (px(m.edge(e)), px(p.edge(e))) {
      let d = [c[0] - a[0], c[1] - a[1]];
      gap = gap.max(<f32 as FloatLike>::sqrt(d[0] * d[0] + d[1] * d[1]));
    }
  }
  if !gap.is_finite() {
    return 1;
  }
  (ce(gap / TIME_SWEEP_STEP_PX) as u32).clamp(1, TIME_SWEEP_MAX)
}

/// The `k`-th of `n_s` sweep sub-capsules of bin `b` (mirror of `dust_bin_sweep`): the chord
/// interpolated at `u = (k + ½)/n_s` between the bin's own edges and the predecessor's, `Σ_v`
/// scaled by the bin's speed factor plus a Gaussian of [`CHORD_VAR_FACTOR`] × (the larger edge
/// displacement / 2 n_s)² along it (the residual between sub-capsules), `1/(SIZE_BINS · n_s)` of
/// the flux, open towards the neighbouring bins.
pub fn bin_sweep_packet(m: &DustMoments, p: &DustMoments, b: usize, k: u32, n_s: u32) -> Packet {
  let cv = m.cov_v();
  let (a, c) = (m.edge(b), m.edge(b + 1));
  let (da, dc) = (sub3(p.edge(b), a), sub3(p.edge(b + 1), c));
  let dmax = if dot3(da, da) > dot3(dc, dc) { da } else { dc };
  let u = (k as f32 + 0.5) / n_s as f32;
  let (ak, ck) = (add3(a, scale3(da, u)), add3(c, scale3(dc, u)));
  let f = 0.5 * (m.edge_factor(b) + m.edge_factor(b + 1));
  let f2 = f * f;
  let kv = CHORD_VAR_FACTOR;
  let h = scale3(dmax, 0.5 / n_s as f32);
  Packet {
    mean: ak,
    cov: [
      f2 * cv[0] + kv * h[0] * h[0],
      f2 * cv[1] + kv * h[0] * h[1],
      f2 * cv[2] + kv * h[0] * h[2],
      f2 * cv[3] + kv * h[1] * h[1],
      f2 * cv[4] + kv * h[1] * h[2],
      f2 * cv[5] + kv * h[2] * h[2],
    ],
    seg: sub3(ck, ak),
    flux: m.flux() / (SIZE_BINS * n_s) as f32,
    open: [b > 0, b + 1 < SIZE_BINS as usize],
    rho: bin_density_ends(m.edge_factor(b), m.edge_factor(b + 1)),
  }
}

/// [`subpackets`] with the polyline optionally pooled into one packet (`merged`) and, given the
/// screen projection `px`, the time sweep of the bins ([`time_sweep`]).
pub fn subpackets_merged(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
) -> alloc::vec::Vec<Packet> {
  subpackets_swept(m, seg, pred, merged, true, None, None)
}

/// [`subpackets_merged`] with the time sweep decided on screen through `px`.
pub fn subpackets_swept(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
  pp: Option<&DustMoments>,
) -> alloc::vec::Vec<Packet> {
  // the arc of a rotating jet (`arc_dirs`): the pieces then follow it, and there are enough of
  // them to draw it
  let arc = pred
    .filter(|p| m.age() <= ARC_MAX_AGE_SAMPLES * (p.age() - m.age()).max(0.0))
    .and_then(|p| arc_dirs(m.mean(), p.mean(), pp.map(|q| q.mean())));
  let n_t = if arc.is_some() {
    TIME_PIECES_MAX
  } else {
    time_pieces(m, pred)
  };
  let ratio = width_ratio(m, pred);
  let inv = 1.0 / n_t as f32;
  let r0 = 1.0 / ratio; // σ_self / σ_pred ≤ 1
  let mut out = alloc::vec::Vec::with_capacity(if merged {
    n_t as usize
  } else {
    (SIZE_BINS * n_t) as usize
  });
  if merged {
    // one pooled Gaussian, uniform along the time segment (closed capsule), the cone pieces as
    // for a bin
    let p = pooled_packet(m, seg);
    if n_t == 1 {
      out.push(p);
    } else {
      // the predecessor's width for the pieces: pooled with its midpoint scatter only when it
      // is a point on screen too, else the mean of its bins (`pooled_within_cov`)
      let base = pred
        .map(|q| {
          if pred_merged {
            pooled_packet(q, seg).cov
          } else {
            pooled_within_cov(q, seg)
          }
        })
        .unwrap_or(p.cov);
      let pm = add3(p.mean, seg);
      for t in 0..n_t {
        let u0 = t as f32 * inv;
        let sc = r0 + (1.0 - r0) * (u0 + 0.5 * inv);
        let sc2 = sc * sc;
        let (mean, seg_t) = match arc {
          Some((d_o, d_p)) => {
            let a = arc_point(p.mean, pm, d_o, d_p, u0);
            (a, sub3(arc_point(p.mean, pm, d_o, d_p, u0 + inv), a))
          }
          None => (add3(p.mean, scale3(seg, u0)), scale3(seg, inv)),
        };
        out.push(Packet {
          mean,
          cov: [
            base[0] * sc2,
            base[1] * sc2,
            base[2] * sc2,
            base[3] * sc2,
            base[4] * sc2,
            base[5] * sc2,
          ],
          seg: seg_t,
          flux: p.flux * inv,
          open: [false, false],
          rho: [1.0, 1.0],
        });
      }
    }
    return out;
  }
  for b in 0..SIZE_BINS as usize {
    // the time segment of this bin: to the *same bin* of the predecessor (its own sizes: a young
    // cluster's fast grains are ten times farther along than its slow ones)
    let seg_t = match pred {
      Some(p) => sub3(
        scale3(add3(p.edge(b), p.edge(b + 1)), 0.5),
        scale3(add3(m.edge(b), m.edge(b + 1)), 0.5),
      ),
      None => [0.0; 3],
    };
    // the longer of the bin's two axes is the exact (uniform) capsule, the other a Gaussian:
    // young dust has a time segment longer than its chords (the sample's own emission interval
    // reaches past its polyline), old dust a chord longer than its time segment
    let chord = sub3(m.edge(b + 1), m.edge(b));
    let time_major = dot3(seg_t, seg_t) > dot3(chord, chord);
    if n_t == 1 {
      if time_major {
        out.push(bin_packet(m, b, seg_t));
      } else {
        // the time cell of the bin is the quadrilateral between its chord and the
        // predecessor's: swept when its edges are pixels apart on screen
        let n_s = time_sweep(m, pred, b, px);
        if n_s == 1 {
          out.push(bin_chord_packet(m, b, seg_t));
        } else {
          let p = pred.expect("a sweep needs a predecessor");
          for k in 0..n_s {
            out.push(bin_sweep_packet(m, p, b, k, n_s));
          }
        }
      }
    } else if time_major {
      // the cone along the time capsule, as for a pooled packet
      let own = bin_packet(m, b, seg_t);
      let p = pred.expect("pieces need a predecessor");
      let base = bin_packet(p, b, seg_t).cov;
      let pm = add3(own.mean, seg_t);
      for t in 0..n_t {
        let u0 = t as f32 * inv;
        let sc = r0 + (1.0 - r0) * (u0 + 0.5 * inv);
        let sc2 = sc * sc;
        let (mean, seg_p) = match arc {
          Some((d_o, d_p)) => {
            let a = arc_point(own.mean, pm, d_o, d_p, u0);
            (a, sub3(arc_point(own.mean, pm, d_o, d_p, u0 + inv), a))
          }
          None => (add3(own.mean, scale3(seg_t, u0)), scale3(seg_t, inv)),
        };
        out.push(Packet {
          mean,
          cov: [
            base[0] * sc2,
            base[1] * sc2,
            base[2] * sc2,
            base[3] * sc2,
            base[4] * sc2,
            base[5] * sc2,
          ],
          seg: seg_p,
          flux: own.flux * inv,
          open: [false, false],
          rho: [1.0, 1.0],
        });
      }
    } else {
      // the cone: `n_t` pieces along the time segment, each with the predecessor's lateral
      // covariance scaled from the cluster's width (zero for the sample pinned at "now") to the
      // predecessor's, and the piece's own time Gaussian
      let p = pred.expect("pieces need a predecessor");
      let own = bin_chord_packet(m, b, [0.0; 3]);
      let cvp = p.cov_v();
      let f = 0.5 * (m.edge_factor(b) + m.edge_factor(b + 1));
      let f2 = f * f;
      let hseg = scale3(seg_t, 0.5 * inv);
      let k = CHORD_VAR_FACTOR;
      let mid = scale3(add3(m.edge(b), m.edge(b + 1)), 0.5);
      let mid_p = add3(mid, seg_t);
      for t in 0..n_t {
        let u0 = t as f32 * inv;
        let sc = r0 + (1.0 - r0) * (u0 + 0.5 * inv);
        let sc2 = sc * sc * f2;
        let shift = match arc {
          Some((d_o, d_p)) => sub3(arc_point(mid, mid_p, d_o, d_p, u0), mid),
          None => scale3(seg_t, u0),
        };
        out.push(Packet {
          mean: add3(own.mean, shift),
          cov: [
            cvp[0] * sc2 + k * hseg[0] * hseg[0],
            cvp[1] * sc2 + k * hseg[0] * hseg[1],
            cvp[2] * sc2 + k * hseg[0] * hseg[2],
            cvp[3] * sc2 + k * hseg[1] * hseg[1],
            cvp[4] * sc2 + k * hseg[1] * hseg[2],
            cvp[5] * sc2 + k * hseg[2] * hseg[2],
          ],
          seg: own.seg,
          flux: own.flux * inv,
          open: own.open,
          rho: own.rho,
        });
      }
    }
  }
  out
}

/// Pixel position of a local point under `mvp` (`None` behind the camera): the merge test of the
/// size polyline ([`DUST_MERGE_PX`]).
pub fn project_point(p: [f32; 3], mvp: &[f32; 16], width: u32, height: u32) -> Option<[f32; 2]> {
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let clip = |r: usize| m(r, 0) * p[0] + m(r, 1) * p[1] + m(r, 2) * p[2] + m(r, 3);
  let w = clip(3);
  if !(w > 0.0) {
    return None;
  }
  Some([
    (clip(0) / w * 0.5 + 0.5) * width as f32,
    (clip(1) / w * 0.5 + 0.5) * height as f32,
  ])
}

/// Whether the size polyline of `m` is below [`DUST_MERGE_PX`] on screen (mirror of
/// `dust_polyline_merged`): the two end edges and the middle one all within it of the mean.
pub fn polyline_merged(m: &DustMoments, mvp: &[f32; 16], width: u32, height: u32) -> bool {
  let Some(c) = project_point(m.mean(), mvp, width, height) else {
    return false;
  };
  for j in [0usize, (SIZE_BINS / 2) as usize, SIZE_BINS as usize] {
    let Some(p) = project_point(m.edge(j), mvp, width, height) else {
      return false;
    };
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    if !(dx * dx + dy * dy < DUST_MERGE_PX * DUST_MERGE_PX) {
      return false;
    }
  }
  true
}

/// `σ_pred / σ_self` of the velocity dispersion (1 without a predecessor or when it is narrower;
/// capped at [`WIDTH_RATIO_MAX`] when the cluster has no width yet — the sample pinned at "now")
pub fn width_ratio(m: &DustMoments, pred: Option<&DustMoments>) -> f32 {
  let own = m.mean_var();
  match pred {
    Some(p) if p.mean_var() > own.max(0.0) => {
      if own > 0.0 {
        <f32 as FloatLike>::sqrt(p.mean_var() / own).min(WIDTH_RATIO_MAX)
      } else {
        WIDTH_RATIO_MAX
      }
    }
    _ => 1.0,
  }
}

/// largest `σ_pred / σ_self` the cone pieces resolve (a zero-width cluster gets this)
pub const WIDTH_RATIO_MAX: f32 = 1024.0;

/// Pieces a capsule is cut into: `clamp(⌈σ_pred/σ_self⌉ − 1, 1, TIME_PIECES_MAX)` (mirror of
/// `dust_time_pieces`): one while the width changes by less than itself along the segment.
pub fn time_pieces(m: &DustMoments, pred: Option<&DustMoments>) -> u32 {
  let r = width_ratio(m, pred);
  ((ce(r) as u32).saturating_sub(1)).clamp(1, TIME_PIECES_MAX)
}

// ─── Projection and capsule (mirror of dust_project_packet / dust_capsule_tau) ──

/// A packet on screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenPacket {
  /// mean (px, origin top-left, pixel centres at +0.5)
  pub mu: [f32; 2],
  /// projected covariance + pixel filter (px²): xx, xy, yy
  pub cov: [f32; 3],
  /// projected segment (px)
  pub seg: [f32; 2],
  /// peak optical depth per px² at `mu` (exposure-scaled)
  pub amp: f32,
  /// smallest σ of `cov` (px)
  pub sigma_min: f32,
  /// determinant of `cov` with the anisotropy floor ([`DUST_DET_ANISO_FLOOR`]): the capsule's
  /// quadratic form and `amp` use this one, never a recomputed one
  pub det: f32,
  /// distance from the eye (AU)
  pub depth_au: f32,
  /// `[start, end]` open (see [`Packet::open`])
  pub open: [bool; 2],
  /// density at the segment's ends, mean 1 (see [`Packet::rho`])
  pub rho: [f32; 2],
}

/// Projects a packet with the Jacobian of `mvp` at its mean (`units_per_m` scales local metres to
/// the mvp's units; `eye_local` is the eye in local metres). `None` when the packet is behind the
/// camera or entirely off screen (beyond [`DUST_SPLAT_SIGMAS`] + its segment).
pub fn project_packet(
  p: &Packet,
  exposure: f32,
  mvp: &[f32; 16],
  width: u32,
  height: u32,
  eye_local: [f32; 3],
) -> Option<ScreenPacket> {
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let clip = |r: usize| dot3(row(r), p.mean) + m(r, 3);
  let w = clip(3);
  if !(w > 0.0) {
    return None;
  }
  let (cx, cy) = (clip(0), clip(1));
  let ndc = [cx / w, cy / w];
  let (wf, hf) = (width as f32, height as f32);
  let mu = [(ndc[0] * 0.5 + 0.5) * wf, (ndc[1] * 0.5 + 0.5) * hf];
  // Jacobian of pixel(mean): d(px)/dμ = (W/2)·(M₀ − ndc.x·M₃)/w, likewise y with M₁ and H/2
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let jx = scale3(sub3(r0, scale3(r3, ndc[0])), 0.5 * wf / w);
  let jy = scale3(sub3(r1, scale3(r3, ndc[1])), 0.5 * hf / w);
  let s = &p.cov;
  let sv = |v: [f32; 3]| -> [f32; 3] {
    [
      s[0] * v[0] + s[1] * v[1] + s[2] * v[2],
      s[1] * v[0] + s[3] * v[1] + s[4] * v[2],
      s[2] * v[0] + s[4] * v[1] + s[5] * v[2],
    ]
  };
  let (sx, sy) = (sv(jx), sv(jy));
  let cov = [
    dot3(jx, sx) + DUST_PIXEL_FILTER_VAR,
    dot3(jx, sy),
    dot3(jy, sy) + DUST_PIXEL_FILTER_VAR,
  ];
  let seg = [dot3(jx, p.seg), dot3(jy, p.seg)];
  // pixel area at the packet's depth: 1 / sqrt(det(J Jᵀ)) m² per px²
  let g = [dot3(jx, jx), dot3(jx, jy), dot3(jy, jy)];
  let det_g = (g[0] * g[2] - g[1] * g[1]).max(0.0);
  let half = 0.5 * (cov[0] + cov[2]);
  let dd = <f32 as FloatLike>::sqrt(
    (0.5 * (cov[0] - cov[2])) * (0.5 * (cov[0] - cov[2])) + cov[1] * cov[1],
  );
  let lam_max = (half + dd).max(DUST_PIXEL_FILTER_VAR);
  // the determinant with the anisotropy floor (f32 cancellation for very elongated packets)
  let det_c = (cov[0] * cov[2] - cov[1] * cov[1])
    .max(DUST_DET_ANISO_FLOOR * lam_max * lam_max)
    .max(1e-30);
  let amp = exposure * p.flux * <f32 as FloatLike>::sqrt(det_g)
    / (2.0 * core::f32::consts::PI * <f32 as FloatLike>::sqrt(det_c));
  if !(amp > 0.0) || !amp.is_finite() {
    return None;
  }
  // the smallest σ from the floored determinant, never from the cancelling `half − dd`
  let sigma_min = <f32 as FloatLike>::sqrt((det_c / lam_max).max(DUST_PIXEL_FILTER_VAR * 0.5));
  // off screen?
  let (ex, ey) = (
    DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(cov[0]),
    DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(cov[2]),
  );
  let x0 = mu[0].min(mu[0] + seg[0]) - ex;
  let x1 = mu[0].max(mu[0] + seg[0]) + ex;
  let y0 = mu[1].min(mu[1] + seg[1]) - ey;
  let y1 = mu[1].max(mu[1] + seg[1]) + ey;
  if x1 < 0.0 || y1 < 0.0 || x0 > wf || y0 > hf || !(x0 <= x1) || !(y0 <= y1) {
    return None;
  }
  let d = sub3(p.mean, eye_local);
  let depth_au = <f32 as FloatLike>::sqrt(dot3(d, d)) / AU_M_F32;
  if !(mu[0].is_finite()
    && mu[1].is_finite()
    && cov[0].is_finite()
    && cov[1].is_finite()
    && cov[2].is_finite()
    && seg[0].is_finite()
    && seg[1].is_finite()
    && sigma_min.is_finite()
    && depth_au.is_finite())
  {
    return None;
  }
  Some(ScreenPacket {
    mu,
    cov,
    seg,
    amp,
    sigma_min,
    det: det_c,
    depth_au,
    open: p.open,
    rho: p.rho,
  })
}

/// `erf` by Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). Mirror of `dust_erf`.
pub fn erf(x: f32) -> f32 {
  let s = if x < 0.0 { -1.0 } else { 1.0 };
  let x = if x < 0.0 { -x } else { x };
  let t = 1.0 / (1.0 + 0.327_591_1 * x);
  let y = 1.0
    - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
      + 0.254_829_592)
      * t
      * <f32 as FloatLike>::exp(-x * x);
  s * y
}

/// Optical depth per px² of the capsule at pixel `x`: `A · ∫₀¹ G(x − μ − u d) du` with
/// `M = Σ'⁻¹`, `e = x − μ`, `a = dᵀMd`, `b = dᵀMe`, `c = eᵀMe`:
/// `A · exp(−½(c − b²/a)) · √(2π/a) · [Φ((1 − b/a)√a) − Φ(−b/√a)]`; a point packet
/// (`a` ≈ 0) is `A · exp(−½c)`. Mirror of `dust_capsule_tau`.
pub fn capsule_tau(p: &ScreenPacket, x: [f32; 2]) -> f32 {
  let det = p.det;
  if !(det > 0.0) {
    return 0.0;
  }
  let (m00, m01, m11) = (p.cov[2] / det, -p.cov[1] / det, p.cov[0] / det);
  let e = [x[0] - p.mu[0], x[1] - p.mu[1]];
  let c = m00 * e[0] * e[0] + 2.0 * m01 * e[0] * e[1] + m11 * e[1] * e[1];
  let d = p.seg;
  let a = m00 * d[0] * d[0] + 2.0 * m01 * d[0] * d[1] + m11 * d[1] * d[1];
  if !(a > 1e-6) {
    return p.amp * <f32 as FloatLike>::exp(-0.5 * c);
  }
  let b = m00 * d[0] * e[0] + m01 * (d[0] * e[1] + d[1] * e[0]) + m11 * d[1] * e[1];
  let sa = <f32 as FloatLike>::sqrt(a);
  let u0 = b / a;
  let phi = |z: f32| 0.5 * (1.0 + erf(z * core::f32::consts::FRAC_1_SQRT_2));
  // an open end continues into the neighbouring bin: no Gaussian cap there
  let phi_end = if p.open[1] { 1.0 } else { phi((1.0 - u0) * sa) };
  let phi_start = if p.open[0] { 0.0 } else { phi(-u0 * sa) };
  let span = phi_end - phi_start;
  let q = (c - b * b / a).max(0.0);
  // the density along the segment is linear, ρ(u) = ρ₀ + (ρ₁ − ρ₀) u: the integral of
  // u · exp(−a(u − u₀)²/2) over the segment is u₀ · I₀ + (E_start − E_end)/a with the
  // Gaussian's values at the closed ends (I₀ = √(2π/a) · span)
  let i0 = <f32 as FloatLike>::sqrt(2.0 * core::f32::consts::PI / a) * span;
  let (r0, r1) = (p.rho[0], p.rho[1]);
  let weight = if r1 == r0 {
    r0 * i0
  } else {
    let e_start = if p.open[0] {
      0.0
    } else {
      <f32 as FloatLike>::exp(-0.5 * a * u0 * u0)
    };
    let e_end = if p.open[1] {
      0.0
    } else {
      <f32 as FloatLike>::exp(-0.5 * a * (1.0 - u0) * (1.0 - u0))
    };
    let i1 = u0 * i0 + (e_start - e_end) / a;
    (r0 * i0 + (r1 - r0) * i1).max(0.0)
  };
  p.amp * <f32 as FloatLike>::exp(-0.5 * q) * weight
}

// ─── Colour (mirror of dust_age_color) ──────────────────────────────────────

/// Hue ramp position of an age: `ln(1 + age/τ) / ln(1 + span)` in `[0, 1]`.
pub fn age_hue_fraction(age_s: f32) -> f32 {
  let f = <f32 as FloatLike>::ln(1.0 + age_s.max(0.0) / DUST_AGE_HUE_TAU_S)
    / <f32 as FloatLike>::ln(1.0 + DUST_AGE_HUE_SPAN);
  f.clamp(0.0, 1.0)
}

/// OKLab coordinates of a linear colour (tests)
pub fn oklab_of(c: [f32; 3]) -> [f32; 3] {
  oklab_from_linear(c)
}

/// Linear sRGB → OKLab (Ottosson 2020).
fn oklab_from_linear(c: [f32; 3]) -> [f32; 3] {
  let l = 0.412_221_47 * c[0] + 0.536_332_55 * c[1] + 0.051_445_995 * c[2];
  let m = 0.211_903_5 * c[0] + 0.680_699_5 * c[1] + 0.107_396_96 * c[2];
  let s = 0.088_302_46 * c[0] + 0.281_718_85 * c[1] + 0.629_978_7 * c[2];
  let cb = |v: f32| <f32 as FloatLike>::pow(v.max(0.0), 1.0 / 3.0);
  let (l, m, s) = (cb(l), cb(m), cb(s));
  [
    0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
    1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
    0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
  ]
}

/// OKLab → linear sRGB (Ottosson 2020).
fn linear_from_oklab(lab: [f32; 3]) -> [f32; 3] {
  let l = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
  let m = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
  let s = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
  let (l, m, s) = (l * l * l, m * m * m, s * s * s);
  [
    4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
    -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
    -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
  ]
}

/// Colour of dust of age `age_s`: the stream colour's hue rotated in OKLab by `π · f(age)`
/// ([`age_hue_fraction`]): the stream colour at the jet, its complement at the oldest ages,
/// lightness and chroma kept, gamut-clipped to `[0, 1]`. Mirror of `dust_age_color`.
pub fn age_color(stream: [f32; 3], age_s: f32) -> [f32; 3] {
  let lab = oklab_from_linear([
    stream[0].clamp(0.0, 1.0),
    stream[1].clamp(0.0, 1.0),
    stream[2].clamp(0.0, 1.0),
  ]);
  let th = core::f32::consts::PI * age_hue_fraction(age_s);
  let (s, c) = (<f32 as FloatLike>::sin(th), <f32 as FloatLike>::cos(th));
  let rot = [lab[0], c * lab[1] - s * lab[2], s * lab[1] + c * lab[2]];
  let rgb = linear_from_oklab(rot);
  [
    rgb[0].clamp(0.0, 1.0),
    rgb[1].clamp(0.0, 1.0),
    rgb[2].clamp(0.0, 1.0),
  ]
}

/// `r | g << 8 | b << 16` of a stream colour (the splat push constant)
pub fn pack_color(c: [f32; 4]) -> u32 {
  let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
  q(c[0]) | (q(c[1]) << 8) | (q(c[2]) << 16)
}
pub fn unpack_color(w: u32) -> [f32; 3] {
  [
    (w & 0xFF) as f32 / 255.0,
    ((w >> 8) & 0xFF) as f32 / 255.0,
    ((w >> 16) & 0xFF) as f32 / 255.0,
  ]
}

// ─── Scatter (mirror of dust_splat.comp) ────────────────────────────────────

/// `a + b` saturating at `u32::MAX` (the GPU does the same with a compare-and-swap loop)
#[inline]
pub fn sat_add(a: &mut u32, b: u32) {
  *a = a.saturating_add(b);
}

/// The texel band of a screen packet (mirror of `dust_band_*`): the segment's lateral Gaussian
/// band (`DUST_SPLAT_SIGMAS` across) between its ends, which carry a Gaussian cap when closed
/// and nothing when open (the neighbouring bin continues there). A point packet is its
/// bounding box.
pub struct CapsuleBand {
  mu: [f32; 2],
  d: [f32; 2],
  dd: f32,
  n: [f32; 2],
  /// half-width across (px)
  w: f32,
  /// `u` range along the segment, caps included
  u_lo: f32,
  u_hi: f32,
  /// point packet: bounding box half-extents
  point: Option<[f32; 2]>,
}

impl CapsuleBand {
  pub fn new(p: &ScreenPacket) -> Self {
    let d = p.seg;
    let dd = d[0] * d[0] + d[1] * d[1];
    let sq = <f32 as FloatLike>::sqrt;
    if !(dd > 1e-12) {
      return Self {
        mu: p.mu,
        d: [0.0; 2],
        dd: 0.0,
        n: [0.0; 2],
        w: 0.0,
        u_lo: 0.0,
        u_hi: 0.0,
        point: Some([
          DUST_SPLAT_SIGMAS * sq(p.cov[0].max(0.0)),
          DUST_SPLAT_SIGMAS * sq(p.cov[2].max(0.0)),
        ]),
      };
    }
    let len = sq(dd);
    let t = [d[0] / len, d[1] / len];
    let n = [-t[1], t[0]];
    let quad =
      |v: [f32; 2]| p.cov[0] * v[0] * v[0] + 2.0 * p.cov[1] * v[0] * v[1] + p.cov[2] * v[1] * v[1];
    let cap = DUST_SPLAT_SIGMAS * sq(quad(t).max(0.0)) / len;
    let w = DUST_SPLAT_SIGMAS * sq(quad(n).max(0.0));
    Self {
      mu: p.mu,
      d,
      dd,
      n,
      w,
      u_lo: if p.open[0] { 0.0 } else { -cap },
      u_hi: if p.open[1] { 1.0 } else { 1.0 + cap },
      point: None,
    }
  }

  /// texel rows `[y0, y1]` of the band at texel size `s`, clipped to `h` rows
  pub fn rows(&self, s: f32, h: u32) -> Option<(i64, i64)> {
    let (y_lo, y_hi) = match self.point {
      Some(e) => (self.mu[1] - e[1], self.mu[1] + e[1]),
      None => {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for u in [self.u_lo, self.u_hi] {
          for v in [-self.w, self.w] {
            let y = self.mu[1] + u * self.d[1] + v * self.n[1];
            lo = lo.min(y);
            hi = hi.max(y);
          }
        }
        (lo, hi)
      }
    };
    let y0 = fl(y_lo / s).max(0.0) as i64;
    let y1 = (ce(y_hi / s) as i64).min(h as i64 - 1);
    (y0 <= y1 && y_lo <= y_hi).then_some((y0, y1))
  }

  /// texel columns `[x0, x1]` of the band on the row at `yc` px, clipped to `w` columns. Across
  /// the segment the band is a Gaussian tail, so a superset of texels is fine; along it the open
  /// chords of a polyline tile the line, so exactly the texel centres with `u_lo ≤ u < u_hi` are
  /// visited (half-open: a centre on the joint belongs to one chord), whatever the orientation.
  pub fn columns(&self, yc: f32, s: f32, w: u32) -> Option<(i64, i64)> {
    let (mut x0, mut x1): (i64, i64);
    match self.point {
      Some(e) => {
        x0 = fl((self.mu[0] - e[0]) / s) as i64;
        x1 = ce((self.mu[0] + e[0]) / s) as i64;
      }
      None => {
        let ry = yc - self.mu[1];
        // across: |(x − μx) n.x + ry n.y| ≤ w
        if fabs(self.n[0]) > 1e-6 {
          let a = self.mu[0] + (-self.w - ry * self.n[1]) / self.n[0];
          let b = self.mu[0] + (self.w - ry * self.n[1]) / self.n[0];
          x0 = fl(a.min(b) / s) as i64;
          x1 = ce(a.max(b) / s) as i64;
        } else if fabs(ry * self.n[1]) > self.w {
          return None;
        } else {
          x0 = 0;
          x1 = w as i64 - 1;
        }
        // along: u = ((x − μx) d.x + ry d.y) / dd ∈ [u_lo, u_hi)
        if fabs(self.d[0]) > 1e-6 {
          let a = self.mu[0] + (self.u_lo * self.dd - ry * self.d[1]) / self.d[0];
          let b = self.mu[0] + (self.u_hi * self.dd - ry * self.d[1]) / self.d[0];
          let c0 = ce(a.min(b) / s - 0.5) as i64;
          let c1 = ce(a.max(b) / s - 0.5) as i64 - 1;
          x0 = x0.max(c0);
          x1 = x1.min(c1);
        } else {
          let u = ry * self.d[1] / self.dd;
          if u < self.u_lo || u >= self.u_hi {
            return None;
          }
        }
      }
    }
    let x0 = x0.max(0);
    let x1 = x1.min(w as i64 - 1);
    (x0 <= x1).then_some((x0, x1))
  }
}

/// Scatters one screen packet of colour `color` into the pyramid: texels of the level chosen by
/// its smallest σ, within [`DUST_SPLAT_SIGMAS`] of the capsule, counts
/// `round(τ · texel area · inv_unit)` (≤ [`DUST_COUNT_MAX_PER_ADD`]) into `τ` and `τ·rgb`, the
/// depth by minimum. Updates the header's level mask and `τ_max`. Returns the texels written.
pub fn splat_scatter(
  p: &ScreenPacket,
  color: [f32; 3],
  layout: &PyramidLayout,
  inv_unit: f32,
  pyramid: &mut [u32],
) -> u32 {
  // the peak is measured whatever the unit (a view fainter than the unit must still move it:
  // `not_flow.rdc`), before any texel test
  let cur = f32::from_bits(pyramid[PYR_TAU_MAX as usize]);
  if !(cur >= p.amp) {
    pyramid[PYR_TAU_MAX as usize] = p.amp.to_bits();
  }
  let mut level = layout.level_for_sigma(p.sigma_min);
  let ex = DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(p.cov[0]);
  let ey = DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(p.cov[2]);
  // safety valve against runaway threads: a packet whose capsule band would visit more than
  // DUST_SPLAT_MAX_TEXELS texels goes to a coarser level (blur, never a stall)
  loop {
    let s = (1u32 << level) as f32;
    let band = (<f32 as FloatLike>::sqrt(p.seg[0] * p.seg[0] + p.seg[1] * p.seg[1]) / s
      + 2.0 * ex / s
      + 2.0)
      * (2.0 * ey / s + 2.0);
    if band <= DUST_SPLAT_MAX_TEXELS as f32 || level + 1 >= layout.level_count() {
      break;
    }
    level += 1;
  }
  let (off, w, h) = layout.levels[level as usize];
  let written = scatter_into(
    p,
    color,
    (1u32 << level) as f32,
    off,
    w,
    h,
    inv_unit,
    pyramid,
    true,
  );
  if written > 0 {
    pyramid[PYR_LEVEL_MASK as usize] |= 1 << level;
  }
  written
}

/// `dust_measure.comp` push constants: the view's pyramid. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustMeasurePushConstants {
  pub pyramid: u64,
  pub pad: [u32; 2],
}

/// The white-point measurement grid (`PyramidLayout::measure`, texels of `2^DUST_WHITE_LEVEL`
/// px, τ only) **from the finished pyramid**: each measure texel holds the counts of the τ the
/// composite displays over its footprint, summed over the levels — the finer levels' texels
/// inside the footprint add up (their counts are τ × their area, so the sum is τ × the footprint),
/// a coarser level contributes the covering texel's count scaled to the footprint's share of it.
/// The grid is thus a box filter of the displayed τ field: a function of the dust alone, not of
/// how many packets drew it or how wide they were (the former per-packet deposit made the white
/// point and the auto softening follow the sampling, see `dust_tau_image_vs_ring_capacity`).
/// Runs once per frame after every tier is splatted. Mirror of `dust_measure.comp`.
pub fn measure_from_pyramid(layout: &PyramidLayout, pyramid: &mut [u32]) {
  let (mo, mw, mh) = layout.measure;
  let mask = pyramid[PYR_LEVEL_MASK as usize];
  for ty in 0..mh {
    for tx in 0..mw {
      let count = measure_texel(layout, pyramid, mask, tx, ty);
      let i = (mo + (ty * mw + tx) * PYRAMID_TEXEL_WORDS) as usize;
      pyramid[i] = count;
      for k in 1..PYRAMID_TEXEL_WORDS as usize {
        pyramid[i + k] = 0;
      }
    }
  }
}

/// Counts of measure texel `(tx, ty)` (see [`measure_from_pyramid`]), saturating.
pub fn measure_texel(layout: &PyramidLayout, pyramid: &[u32], mask: u32, tx: u32, ty: u32) -> u32 {
  let mut sum = 0u64;
  for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
    if mask & (1 << l) == 0 {
      continue;
    }
    let l = l as u32;
    if l <= DUST_WHITE_LEVEL {
      // n × n finer texels inside the footprint
      let n = 1u32 << (DUST_WHITE_LEVEL - l);
      for dy in 0..n {
        let ly = ty * n + dy;
        if ly >= h {
          break;
        }
        for dx in 0..n {
          let lx = tx * n + dx;
          if lx >= w {
            break;
          }
          sum += pyramid[(off + (ly * w + lx) * PYRAMID_TEXEL_WORDS) as usize] as u64;
        }
      }
    } else {
      // the covering coarser texel: its count is τ × 4^l px², the footprint is 4^DUST_WHITE_LEVEL
      let shift = l - DUST_WHITE_LEVEL;
      let lx = (tx >> shift).min(w - 1);
      let ly = (ty >> shift).min(h - 1);
      sum += (pyramid[(off + (ly * w + lx) * PYRAMID_TEXEL_WORDS) as usize] as u64) >> (2 * shift);
    }
  }
  sum.min(u32::MAX as u64) as u32
}

/// The capsule of `p` into one grid of `w × h` texels of `s` px at word `off`: for each texel row
/// only the x-range the segment's lateral band covers is visited, counts
/// `round(τ · texel area · inv_unit)` (≤ [`DUST_COUNT_MAX_PER_ADD`]) by saturating adds; with
/// `full` the colour and the nearest depth too. Mirror of `scatter_into` in `dust_splat.comp`.
#[allow(clippy::too_many_arguments)]
pub fn scatter_into(
  p: &ScreenPacket,
  color: [f32; 3],
  s: f32,
  off: u32,
  w: u32,
  h: u32,
  inv_unit: f32,
  pyramid: &mut [u32],
  full: bool,
) -> u32 {
  // a segment shorter than two texels at this level may miss every texel centre with open
  // ends: it is drawn closed (Gaussian caps), so its flux lands on the texels around it
  let mut p = *p;
  if p.seg[0] * p.seg[0] + p.seg[1] * p.seg[1] < 4.0 * s * s {
    p.open = [false, false];
  }
  let p = &p;
  let band = CapsuleBand::new(p);
  let Some((y0, y1)) = band.rows(s, h) else {
    return 0;
  };
  let area = s * s;
  let depth_bits = p.depth_au.max(0.0).to_bits();
  let mut written = 0u32;
  // error diffusion along the packet's own texel loop (deterministic: one thread, fixed order):
  // the fraction below one count is carried to the next texel instead of being dropped, so a
  // packet deposits its whole flux however faint its texels (16 bins of 1/16 flux each would
  // otherwise lose their outskirts, and the picture would depend on the zoom)
  let mut carry = 0.0f32;
  for ty in y0..=y1 {
    let yc = (ty as f32 + 0.5) * s;
    let Some((x0, x1)) = band.columns(yc, s, w) else {
      continue;
    };
    for tx in x0..=x1 {
      let x = [(tx as f32 + 0.5) * s, yc];
      let tau = capsule_tau(p, x) * area * inv_unit;
      if !(tau >= 0.0) || !tau.is_finite() {
        continue;
      }
      let tau_c = tau.min(DUST_COUNT_MAX_PER_ADD as f32) + carry;
      let counts = (tau_c + 0.5) as u32;
      carry = tau_c - counts as f32;
      if counts == 0 {
        continue;
      }
      let i = (off + (ty as u32 * w + tx as u32) * PYRAMID_TEXEL_WORDS) as usize;
      sat_add(&mut pyramid[i], counts);
      if full {
        for k in 0..3 {
          sat_add(
            &mut pyramid[i + 1 + k],
            (counts as f32 * color[k] + 0.5) as u32,
          );
        }
        if pyramid[i + 4] == 0 || depth_bits < pyramid[i + 4] {
          pyramid[i + 4] = depth_bits;
        }
      }
      written += 1;
    }
  }
  written
}

/// Stamps a tracer dot ([`TRACER_PX`] radius, `counts` high, white colour) at `mu` on level 0.
pub fn splat_tracer(mu: [f32; 2], counts: u32, layout: &PyramidLayout, pyramid: &mut [u32]) {
  if counts == 0 {
    return;
  }
  let (off, w, h) = layout.levels[0];
  let r = TRACER_PX;
  let x0 = fl(mu[0] - r).max(0.0) as i64;
  let x1 = (ce(mu[0] + r) as i64).min(w as i64 - 1);
  let y0 = fl(mu[1] - r).max(0.0) as i64;
  let y1 = (ce(mu[1] + r) as i64).min(h as i64 - 1);
  for ty in y0..=y1 {
    for tx in x0..=x1 {
      let (dx, dy) = (tx as f32 + 0.5 - mu[0], ty as f32 + 0.5 - mu[1]);
      if dx * dx + dy * dy > r * r {
        continue;
      }
      let i = (off + (ty as u32 * w + tx as u32) * PYRAMID_TEXEL_WORDS) as usize;
      for k in 0..4 {
        sat_add(&mut pyramid[i + k], counts);
      }
    }
  }
  pyramid[PYR_LEVEL_MASK as usize] |= 1;
}

/// One tier's splat pass on the host (CPU particle mode and tests; mirror of `dust_splat.comp`):
/// every live cluster → its sub-packets → projected → scattered. `render` gives the predecessor
/// validity ([`super::streak_pred`]), `flow` the flow clock, `tracer_counts` the tracer height.
#[allow(clippy::too_many_arguments)]
pub fn splat_tier(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
) {
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  splat_tier_range(
    moments,
    render,
    pc,
    flow,
    tracer_counts,
    layout,
    pyramid,
    0..n,
  );
}

/// [`splat_tier`] restricted to the compact indices in `range` (clipped to the live count). The
/// predecessor lookup still reads the whole `moments` / `render` slices, so a tier can be split
/// across threads into separate pyramids: the scatter is a saturating sum per texel word, a
/// nonzero minimum for the depth word, a maximum for [`PYR_TAU_MAX`] and an OR for
/// [`PYR_LEVEL_MASK`], all order-free, so the partial pyramids merge exactly.
#[allow(clippy::too_many_arguments)]
pub fn splat_tier_range(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  range: core::ops::Range<usize>,
) {
  let stream = unpack_color(pc.color);
  let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  for i in range.start..range.end.min(n) {
    let m = &moments[i];
    if !(m.flux() > 0.0) {
      continue;
    }
    let pred_i = streak_pred(render, i);
    let pred = pred_i.map(|q| &moments[q]);
    let pp = pred_i.and_then(|q| streak_pred(render, q)).map(|q| &moments[q]);
    let seg = match pred {
      Some(p) => sub3(p.mean(), m.mean()),
      None => [0.0; 3],
    };
    let mut exposure = pc.exposure;
    if pc.flags & DUST_VIEW_FLOW != 0 {
      exposure *= flow_factor(m.age(), flow);
    }
    let color = age_color(stream, m.age());
    let mut first: Option<[f32; 2]> = None;
    // pooled only when the predecessor's polyline is below a pixel too: a just-emitted sample is
    // a point whose time pieces are shaped by its predecessor, and a predecessor whose sizes
    // already span kilometres must be drawn bin by bin (the halo behind the nucleus of the 5-km
    // frame came from its pooled covariance)
    let pred_merged = pred.is_none_or(|p| polyline_merged(p, &pc.mvp, layout.width, layout.height));
    let merged = polyline_merged(m, &pc.mvp, layout.width, layout.height) && pred_merged;
    let px = |q: [f32; 3]| project_point(q, &pc.mvp, layout.width, layout.height);
    for p in subpackets_swept(m, seg, pred, merged, pred_merged, Some(&px), pp) {
      if let Some(sp) = project_packet(&p, exposure, &pc.mvp, layout.width, layout.height, eye) {
        splat_scatter(&sp, color, layout, pc.inv_unit, pyramid);
        first.get_or_insert(sp.mu);
      }
    }
    if pc.flags & DUST_VIEW_TRACERS != 0 && is_tracer(m.id()) {
      // the tracer sits at the exact cluster position
      let p = Packet {
        mean: m.mean(),
        cov: [0.0; 6],
        seg: [0.0; 3],
        flux: 1.0,
        open: [false, false],
        rho: [1.0, 1.0],
      };
      if let Some(sp) = project_packet(&p, 1.0, &pc.mvp, layout.width, layout.height, eye) {
        splat_tracer(sp.mu, tracer_counts, layout, pyramid);
      }
    }
    let _ = first;
  }
}

/// Stream predecessor of compact index `r` (mirror of `dust_streak_pred`, see
/// [`super::streak_pred`]): `r − S` when it is live and no break precedes `r`.
pub fn streak_pred(render: &[DustRenderCluster], r: usize) -> Option<usize> {
  let y = render[r].age_id_dbeta_flux[1];
  let z = render[r].age_id_dbeta_flux[2];
  let n = 1usize << render_stream_shift(y);
  if r < n || stream_break(z) {
    return None;
  }
  render_live(render[r - n].age_id_dbeta_flux[1]).then_some(r - n)
}

// ─── Readback: white point, auto softening, composite mirror ────────────────

/// `(p99, p50)` of the mean exposure-scaled optical depth per px² over the non-empty texels of
/// one level (`counts · unit / texel area`). `None` when all are empty.
pub fn white_point_from_level(level: &[u32], texel_px: u32, unit: f32) -> Option<(f32, f32)> {
  let mut v: alloc::vec::Vec<u32> = level
    .chunks_exact(PYRAMID_TEXEL_WORDS as usize)
    .map(|t| t[0])
    .filter(|&c| c > 0)
    .collect();
  if v.is_empty() {
    return None;
  }
  v.sort_unstable();
  let at = |q: f32| v[(((v.len() - 1) as f32 * q).round() as usize).min(v.len() - 1)] as f32;
  let area = (texel_px * texel_px) as f32;
  Some((
    at(super::WHITE_TILE_PERCENTILE) * unit / area,
    at(0.5) * unit / area,
  ))
}

/// Display level the auto softening puts the median dust texel at (`asinh` stretch, relative
/// to the white point): the same whatever the distance or the dynamic range of the view.
pub const DUST_AUTO_MEDIAN_LEVEL: f32 = 0.12;

/// Auto softening of the display stretch: the `s` for which the median dust texel of the view
/// (`p50`, relative to the white point `p99`) displays at [`DUST_AUTO_MEDIAN_LEVEL`], found by
/// bisection in log `s` (`display_stretch(x, 0, s)` decreases with `s`), clamped to the accepted
/// range. Multiplied by the user's "dust visibility" factor.
pub fn auto_softening(p50: f32, p99: f32) -> f32 {
  if !(p50 > 0.0) || !(p99 > 0.0) || !p50.is_finite() || !p99.is_finite() {
    return super::DUST_SOFTENING_DEFAULT;
  }
  let x = (p50 / p99).min(1.0);
  // beyond the range's ends the median cannot be moved to the target: the end itself
  if super::display_stretch(x, 0.0, super::DUST_SOFTENING_MIN) <= DUST_AUTO_MEDIAN_LEVEL {
    return super::DUST_SOFTENING_MIN;
  }
  if super::display_stretch(x, 0.0, super::DUST_SOFTENING_MAX) >= DUST_AUTO_MEDIAN_LEVEL {
    return super::DUST_SOFTENING_MAX;
  }
  let (ln, exp) = (<f32 as FloatLike>::ln, <f32 as FloatLike>::exp);
  let (mut lo, mut hi) = (ln(super::DUST_SOFTENING_MIN), ln(super::DUST_SOFTENING_MAX));
  for _ in 0..40 {
    let mid = 0.5 * (lo + hi);
    if super::display_stretch(x, 0.0, exp(mid)) > DUST_AUTO_MEDIAN_LEVEL {
      lo = mid; // too bright: soften less (larger s)
    } else {
      hi = mid;
    }
  }
  exp(0.5 * (lo + hi)).clamp(super::DUST_SOFTENING_MIN, super::DUST_SOFTENING_MAX)
}

/// Mirror of the composite's pyramid read at pixel centre `(x + 0.5, y + 0.5)`: bilinear per level
/// (texel centres at `(t + 0.5)·2^ℓ`), summed as mean optical depth per px²; returns
/// `(τ, rgb·τ, nearest depth AU or +∞)`.
pub fn composite_sample(
  layout: &PyramidLayout,
  pyramid: &[u32],
  unit: f32,
  x: u32,
  y: u32,
) -> (f32, [f32; 3], f32) {
  let mask = pyramid[PYR_LEVEL_MASK as usize];
  let mut tau = 0.0f32;
  let mut rgb = [0.0f32; 3];
  let mut depth = f32::INFINITY;
  for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
    if mask & (1 << l) == 0 {
      continue;
    }
    let s = (1u32 << l) as f32;
    let fx = (x as f32 + 0.5) / s - 0.5;
    let fy = (y as f32 + 0.5) / s - 0.5;
    let (x0, y0) = (fl(fx), fl(fy));
    let (ax, ay) = (fx - x0, fy - y0);
    let scale = unit / (s * s);
    for (dx, dy, wgt) in [
      (0.0, 0.0, (1.0 - ax) * (1.0 - ay)),
      (1.0, 0.0, ax * (1.0 - ay)),
      (0.0, 1.0, (1.0 - ax) * ay),
      (1.0, 1.0, ax * ay),
    ] {
      let tx = (x0 + dx).clamp(0.0, (w - 1) as f32) as u32;
      let ty = (y0 + dy).clamp(0.0, (h - 1) as f32) as u32;
      let i = (off + (ty * w + tx) * PYRAMID_TEXEL_WORDS) as usize;
      let c = pyramid[i];
      if c == 0 {
        continue;
      }
      tau += wgt * c as f32 * scale;
      for k in 0..3 {
        rgb[k] += wgt * pyramid[i + 1 + k] as f32 * scale;
      }
      let d = f32::from_bits(pyramid[i + 4]);
      if d < depth {
        depth = d;
      }
    }
  }
  (tau, rgb, depth)
}

/// White point the exposure divides by: the measured one (full adaptation; the former half
/// damping, `display_white`, is gone), 1 before any measurement.
pub fn display_white_v4(measured: f32) -> f32 {
  if !(measured > 0.0) || !measured.is_finite() {
    1.0
  } else {
    measured
  }
}

#[allow(dead_code)]
const _USED: (u32, f32) = (CHILD_ID_MASK, DUST_AGE_HUE_SPAN);

// ─── Shape analysis: segmentation of a τ image and the progressiveness of its growth ─────
// Used by the tests (`dust_grows_progressively_from_the_jet`) and the headless observer
// (`--check`): the automated version of "watch the tail grow".

/// Descriptors of the dust region of one τ image (per px², the composite's input).
#[derive(Debug, Clone, PartialEq)]
pub struct DustShape {
  pub width: u32,
  pub height: u32,
  /// threshold the segmentation used (τ per px²)
  pub threshold: f32,
  /// pixels of the connected component holding the nucleus
  pub area_px: u32,
  /// pixels above the threshold outside that component (detached blobs)
  pub detached_px: u32,
  /// Σ τ over the component
  pub total_tau: f64,
  /// τ-weighted centroid (px)
  pub centroid: [f32; 2],
  /// τ-weighted percentiles and the maximum of the distance to the nucleus pixel (px)
  pub radius_p50: f32,
  pub radius_p90: f32,
  pub radius_max: f32,
  /// unit principal axis of the τ-weighted second moments about the nucleus
  pub axis: [f32; 2],
  /// √(λ_max / λ_min) of those moments (1 = round)
  pub elongation: f32,
  /// the component's mask (row-major, `width × height`)
  pub mask: alloc::vec::Vec<bool>,
}

/// Segments `tau` (row-major `w × h`, τ per px²) at `threshold`: the 4-connected component of
/// pixels `≥ threshold` holding the nucleus pixel (the nearest above-threshold pixel within 3 px
/// of it, else an empty shape), its τ-weighted radii about the nucleus and its principal axis.
pub fn segment_shape(
  tau: &[f32],
  w: u32,
  h: u32,
  nucleus_px: [f32; 2],
  threshold: f32,
) -> DustShape {
  let (wi, hi) = (w as usize, h as usize);
  let mut shape = DustShape {
    width: w,
    height: h,
    threshold,
    area_px: 0,
    detached_px: 0,
    total_tau: 0.0,
    centroid: nucleus_px,
    radius_p50: 0.0,
    radius_p90: 0.0,
    radius_max: 0.0,
    axis: [1.0, 0.0],
    elongation: 1.0,
    mask: alloc::vec![false; wi * hi],
  };
  if wi == 0 || hi == 0 || tau.len() < wi * hi {
    return shape;
  }
  let above = |x: usize, y: usize| tau[y * wi + x] >= threshold && threshold > 0.0;
  // seed: the nucleus pixel or the nearest above-threshold pixel within 3 px
  let (nx, ny) = (nucleus_px[0], nucleus_px[1]);
  let mut seed: Option<(usize, usize)> = None;
  let mut best = f32::INFINITY;
  for dy in -3i64..=3 {
    for dx in -3i64..=3 {
      let (x, y) = ((nx as i64) + dx, (ny as i64) + dy);
      if x < 0 || y < 0 || x >= wi as i64 || y >= hi as i64 {
        continue;
      }
      let (xu, yu) = (x as usize, y as usize);
      if above(xu, yu) {
        let d = ((xu as f32 + 0.5 - nx) * (xu as f32 + 0.5 - nx)
          + (yu as f32 + 0.5 - ny) * (yu as f32 + 0.5 - ny));
        if d < best {
          best = d;
          seed = Some((xu, yu));
        }
      }
    }
  }
  let total_above = (0..hi)
    .flat_map(|y| (0..wi).map(move |x| (x, y)))
    .filter(|&(x, y)| above(x, y))
    .count();
  let Some(seed) = seed else {
    shape.detached_px = total_above as u32;
    return shape;
  };
  // flood fill (4-connected)
  let mut stack = alloc::vec![seed];
  shape.mask[seed.1 * wi + seed.0] = true;
  let mut samples: alloc::vec::Vec<(f32, f32)> = alloc::vec::Vec::new(); // (distance, τ)
  let (mut sx, mut sy, mut st) = (0.0f64, 0.0f64, 0.0f64);
  let (mut mxx, mut mxy, mut myy) = (0.0f64, 0.0f64, 0.0f64);
  while let Some((x, y)) = stack.pop() {
    let t = tau[y * wi + x];
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let (dx, dy) = (px - nx, py - ny);
    samples.push((<f32 as FloatLike>::sqrt(dx * dx + dy * dy), t));
    sx += t as f64 * px as f64;
    sy += t as f64 * py as f64;
    st += t as f64;
    mxx += t as f64 * dx as f64 * dx as f64;
    mxy += t as f64 * dx as f64 * dy as f64;
    myy += t as f64 * dy as f64 * dy as f64;
    shape.area_px += 1;
    let nb = [
      (x.wrapping_sub(1), y),
      (x + 1, y),
      (x, y.wrapping_sub(1)),
      (x, y + 1),
    ];
    for (qx, qy) in nb {
      if qx < wi && qy < hi && !shape.mask[qy * wi + qx] && above(qx, qy) {
        shape.mask[qy * wi + qx] = true;
        stack.push((qx, qy));
      }
    }
  }
  shape.detached_px = total_above.saturating_sub(shape.area_px as usize) as u32;
  shape.total_tau = st;
  if st > 0.0 {
    shape.centroid = [(sx / st) as f32, (sy / st) as f32];
    // weighted percentiles of the distance
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut acc = 0.0f64;
    let (mut p50, mut p90) = (None, None);
    for (d, t) in &samples {
      acc += *t as f64;
      if p50.is_none() && acc >= 0.5 * st {
        p50 = Some(*d);
      }
      if p90.is_none() && acc >= 0.9 * st {
        p90 = Some(*d);
      }
    }
    shape.radius_p50 = p50.unwrap_or(0.0);
    shape.radius_p90 = p90.unwrap_or(0.0);
    shape.radius_max = samples.last().map(|s| s.0).unwrap_or(0.0);
    // principal axis of the second moments about the nucleus
    let (a, b, c) = (mxx / st, mxy / st, myy / st);
    let half = 0.5 * (a + c);
    let dd = ((0.5 * (a - c)).powi(2) + b * b).sqrt();
    let (l1, l2) = (half + dd, (half - dd).max(1e-30));
    shape.elongation = (l1 / l2).sqrt() as f32;
    let (ax, ay) = if b.abs() > 1e-30 {
      (l1 - c, b)
    } else if a >= c {
      (1.0, 0.0)
    } else {
      (0.0, 1.0)
    };
    let n = (ax * ax + ay * ay).sqrt().max(1e-30);
    shape.axis = [(ax / n) as f32, (ay / n) as f32];
  }
  shape
}

/// How the dust region changed between two consecutive images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeStep {
  /// pixels in `next` not in `prev`
  pub grown_px: u32,
  /// pixels in `prev` not in `next`
  pub lost_px: u32,
  /// grown pixels farther than `reach_px` from every pixel of `prev`: dust that appeared away
  /// from the existing cloud (0 for progressive growth)
  pub grown_outside_reach: u32,
  pub area_ratio: f32,
  pub radius_p90_ratio: f32,
  pub tau_ratio: f32,
  /// the farthest grown pixel's distance to the previous region (px)
  pub farthest_growth_px: f32,
}

/// Compares two shapes of the same image size: growth is **progressive** when every new pixel is
/// within `reach_px` of the previous region (the distance the dust itself could travel over the
/// step, plus the splat footprint).
pub fn shape_step(prev: &DustShape, next: &DustShape, reach_px: f32) -> ShapeStep {
  let (w, h) = (prev.width as usize, prev.height as usize);
  let mut step = ShapeStep {
    grown_px: 0,
    lost_px: 0,
    grown_outside_reach: 0,
    area_ratio: if prev.area_px > 0 {
      next.area_px as f32 / prev.area_px as f32
    } else {
      f32::INFINITY
    },
    radius_p90_ratio: if prev.radius_p90 > 0.0 {
      next.radius_p90 / prev.radius_p90
    } else {
      f32::INFINITY
    },
    tau_ratio: if prev.total_tau > 0.0 {
      (next.total_tau / prev.total_tau) as f32
    } else {
      f32::INFINITY
    },
    farthest_growth_px: 0.0,
  };
  if next.width as usize != w || next.height as usize != h {
    return step;
  }
  // distance transform of the previous mask (brute force on the boundary: the masks are small)
  let mut boundary: alloc::vec::Vec<(f32, f32)> = alloc::vec::Vec::new();
  for y in 0..h {
    for x in 0..w {
      if !prev.mask[y * w + x] {
        continue;
      }
      let edge = x == 0
        || y == 0
        || x + 1 == w
        || y + 1 == h
        || !prev.mask[y * w + x - 1]
        || !prev.mask[y * w + x + 1]
        || !prev.mask[(y - 1) * w + x]
        || !prev.mask[(y + 1) * w + x];
      if edge {
        boundary.push((x as f32, y as f32));
      }
    }
  }
  for y in 0..h {
    for x in 0..w {
      let (p, n) = (prev.mask[y * w + x], next.mask[y * w + x]);
      if p && !n {
        step.lost_px += 1;
      }
      if n && !p {
        step.grown_px += 1;
        let (fx, fy) = (x as f32, y as f32);
        let mut best = f32::INFINITY;
        for &(bx, by) in &boundary {
          let d = (fx - bx) * (fx - bx) + (fy - by) * (fy - by);
          if d < best {
            best = d;
          }
        }
        let d = if boundary.is_empty() {
          f32::INFINITY
        } else {
          <f32 as FloatLike>::sqrt(best)
        };
        if d > step.farthest_growth_px {
          step.farthest_growth_px = d;
        }
        if d > reach_px {
          step.grown_outside_reach += 1;
        }
      }
    }
  }
  step
}

/// The segmentation threshold matching what the composite displays: the τ per px² at which the
/// display stretch reaches `level` of the white point (`white` = the measured p99, `s` the
/// softening): `τ = white · s · sinh(level · asinh(1/s))`.
pub fn display_threshold(white: f32, s: f32, level: f32) -> f32 {
  let s = s.clamp(super::DUST_SOFTENING_MIN, super::DUST_SOFTENING_MAX);
  let a = <f32 as FloatLike>::ln(1.0 / s + <f32 as FloatLike>::sqrt(1.0 / (s * s) + 1.0)); // asinh(1/s)
  let x = level * a;
  white * s * 0.5 * (<f32 as FloatLike>::exp(x) - <f32 as FloatLike>::exp(-x))
}
