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
//! jet) to its complement at one TTL of the system, linear in age, constant beyond
//! ([`age_color`]); the rotated colour is brought back into the sRGB gamut by reducing its chroma
//! ([`oklab_to_gamut`]), never by clamping channels (that turns the hue).
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
pub const PYRAMID_HEADER_WORDS: u32 = 256;
/// words per pyramid texel: `τ`, `τ·r`, `τ·g`, `τ·b` (fixed point, the colour words at
/// [`DUST_COLOR_SCALE`] times the τ counts), nearest dust depth (AU, f32 bits)
pub const PYRAMID_TEXEL_WORDS: u32 = 5;
/// the colour words' fixed point over the τ counts': a one-count deposit writes `16 · c` ± the
/// dither, not a colour corner (at one count per texel the hashed rounding of `counts · c` left
/// every faint texel a pure (1, 1, 0) / (1, 0, 0) / (0, 1, 0) speckle, unbiased in the mean but
/// visible); the composite divides the colour sums by it (`composite_sample`, `composite.frag`).
/// Mirror of `DUST_COLOR_SCALE`.
pub const DUST_COLOR_SCALE: f32 = 16.0;
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
/// f32 bits: the white point (radiance factor) the composite shows this frame with (host-written,
/// for dump readers: `.pyr` dumps are self-describing)
pub const PYR_WHITE: u32 = 9;
/// f32 bits: the auto softening of this frame (host-written)
pub const PYR_SOFTENING: u32 = 10;
/// the dust phase function Φ(α) at integer degrees 0..=180 ([`PHASE_HM_0DEG`], f32 bits, 181
/// words, host-written with the header): the splat's radiance weight reads it from here — a
/// `const float[181]` in the shader with a dynamic index changed the GPU's splats even when
/// the weight was off
pub const PYR_PHASE_LUT: u32 = 64;
/// the measurement grid `(offset, width, height)` ([`PyramidLayout::measure`])
pub const PYR_MEASURE: u32 = 12;
/// the fixed-point unit this frame's counts were written with (f32 bits, host-written; 0 when
/// unknown): what a dump reader needs to turn counts back into optical depth
pub const PYR_UNIT: u32 = 15;
/// per level `ℓ`: `(offset in words, width, height)` at `PYR_TABLE + 3ℓ`
pub const PYR_TABLE: u32 = 16;
pub const PYRAMID_MAX_LEVELS: u32 = (PYR_EWA_TEXELS - PYR_TABLE) / 3;
/// the texels the EWA packets of this frame planned to visit, in units of
/// [`DUST_EWA_TEXEL_UNIT`] (`atomicAdd` of `⌊band / unit⌋ + 1` per packet after the level choice; the
/// host mirror adds, a merge sums): the near-field frame cost (a packet near the eye crosses the
/// screen; each is capped at [`DUST_SPLAT_MAX_TEXELS`], the frame was not)
pub const PYR_EWA_TEXELS: u32 = 63;
/// texels per count of [`PYR_EWA_TEXELS`] (a frame of 3·10⁶ packets at the per-packet cap stays
/// below 2³²)
pub const DUST_EWA_TEXEL_UNIT: f32 = 64.0;
/// children made by the perspective split this frame (`atomicAdd`; [`split_packet`])
pub const PYR_SPLIT_CHILDREN: u32 = 11;
/// the ray-integral evaluations (texels × pieces) the volumetric packets of this frame made
/// (`atomicAdd`; the host mirror adds, a merge sums): the host raises the volume level bias of
/// the next frames when it passes [`DUST_VOLUME_FRAME_BUDGET`] ([`VolumeStats::suggested_bias`])
pub const PYR_VOLUME_EVALS: u32 = 245;
/// formerly the split overflow; the word is [`PYR_VOLUME_EVALS`] now
#[deprecated(
  note = "word 245 is PYR_VOLUME_EVALS; the split overflow is SplitStats::overflow(), computed from the children"
)]
pub const PYR_SPLIT_OVERFLOW: u32 = PYR_VOLUME_EVALS;
/// f32 bits (`atomicMax`): the largest flux of a packet dropped as wholly behind the eye plane
pub const PYR_SPLIT_DROPPED_FLUX: u32 = 246;
/// host-written: bits 0–7 the split depth limit of this frame (`≤ DUST_SPLIT_DEPTH_MAX`; the
/// header's default is the maximum, [`SplitStats::suggested_depth`] lowers it after a frame over
/// the budget), bits 8–15 the volume level bias ([`VolumeStats`]: levels added to every
/// volumetric packet's before its evaluation cap; 0 by default). Every packet splits and plans
/// its volume work deterministically from these: no run-time gate, so the splat stays
/// independent of the thread order
pub const PYR_SPLIT_DEPTH: u32 = 247;
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
/// a packet is split while its σ along the view direction exceeds this fraction of its depth
/// from the eye (the EWA linearisation error ∝ σ_v / depth), or its chord's depth span exceeds
/// it ([`split_packet`]; the EWA takes a chord's pixel area at its start, so the chord's error is
/// first order in span / depth: at 0.2 a chord along the view over-drew by 8 %, the shell
/// hand-over's (a)/(c) 1.081)
pub const DUST_SPLIT_EPS: f32 = 0.1;
/// the depth slice's offset of the two children along the regression direction `Σ f / σ_v`, in
/// units of it: `σ_v` shrinks by `√(1 − a²)` = 0.6 per level ([`split_step`])
pub const DUST_SPLIT_A: f32 = 0.8;
/// split levels per packet (at most `2^n` leaves)
pub const DUST_SPLIT_DEPTH_MAX: u32 = 4;
/// children per frame the host aims below by lowering the next frame's depth limit
/// ([`SplitStats::suggested_depth`], [`PYR_SPLIT_DEPTH`]); never a run-time gate
pub const DUST_SPLIT_BUDGET: u32 = 1 << 20;
/// a chord crossing the eye plane keeps `1 − ε` of its in-front part: the kept start is strictly
/// in front of the eye, where the perspective projection is finite
pub const DUST_SPLIT_CLIP_EPS: f32 = 1e-3;
/// a packet is drawn by the exact ray integral ([`volume_scatter`]) instead of the EWA when
/// `√tr Σ > DUST_VOLUME_RATIO · dist`, `dist` the distance from the eye to its capsule (the
/// eye inside it or near it: [`is_volumetric`]). The EWA's error grows with the packet's angular
/// size (the pixel scale and area vary across it): measured against the ray integral, 3 % of the
/// total at the switch (an isotropic packet 8.7 σ away), 7 % at 3 σ, so the switch is continuous
/// to 3 % (`classification_is_continuous`)
pub const DUST_VOLUME_RATIO: f32 = 0.2;
/// a volumetric packet whose capsule holds the eye within this many Mahalanobis σ covers the
/// whole frame
pub const DUST_VOLUME_SIGMAS: f32 = 3.0;
/// the volumetric texel is at most `1 /` this of the packet's angular σ (texel centres sample the
/// exact integral, the composite interpolates bilinearly: 1.2 % rms for an eye inside a packet)
pub const DUST_VOLUME_TEXELS_PER_SIGMA: f32 = 4.0;
/// the finest level a whole-frame volumetric packet is drawn at (texels of 8 px)
pub const DUST_VOLUME_MIN_LEVEL: u32 = 3;
/// a volumetric chord is integrated as at most this many Gaussian pieces along it
pub const DUST_VOLUME_PIECES_MAX: u32 = 16;
/// the pieces of a volumetric chord are this many Mahalanobis σ apart (at most): with the exact
/// `1/12` spacing² of a uniform piece added to each, σ_piece ≥ 1.15 σ, the ripple along the chord
/// below `2 exp(−2π²/3)` = 0.3 %
pub const DUST_VOLUME_PIECE_SIGMAS: f32 = 2.0;
/// a piece's variance along the chord in units of its length², when the chord needs more than
/// [`DUST_VOLUME_PIECES_MAX`] pieces (or the evaluation cap merged them): σ = 0.55 of the spacing
/// (ripple < 1 %) instead of the exact `1/12` (ripple 38 % at large spacings)
pub const DUST_VOLUME_PIECE_VAR_CAPPED: f32 = 0.3025;
/// ray-integral evaluations (Σ over the pieces of the texels each covers) one volumetric packet
/// may make: the level is coarsened, then the pieces merged in pairs, until the plan fits
/// ([`volume_plan`]). A shader thread draws at most 64 packets: ≤ 2.6·10⁵ evaluations (an
/// unbounded packet made 1.6·10⁷ in one thread and took the machine down)
pub const DUST_VOLUME_MAX_EVALS: u32 = 4096;
/// evaluations per frame above which the host coarsens every volumetric packet by one more level
/// ([`VolumeStats::suggested_bias`])
pub const DUST_VOLUME_FRAME_BUDGET: u32 = 1 << 26;
/// the largest volume level bias
pub const DUST_VOLUME_BIAS_MAX: u32 = 15;
/// texels per frame the EWA packets may plan ([`PYR_EWA_TEXELS`]) before the host raises the
/// EWA bias of the next frames ([`EwaStats::suggested_bias`])
pub const DUST_EWA_FRAME_BUDGET: u32 = 1 << 28;
/// the largest EWA bias: the per-packet texel cap is `DUST_SPLAT_MAX_TEXELS / 4^bias` (4 at 6)
pub const DUST_EWA_BIAS_MAX: u32 = 6;
/// the in-frame gates of the split, the volumetric and the EWA work sit at this many frame
/// budgets: past it a packet takes the cheapest path at once (no further split, the coarsest
/// level), so a frame the host feedback has not caught up with — a cold start or a jump inside
/// the coma — is bounded by construction. The one place the splat depends on the dispatch
/// order, only in such frames
pub const DUST_FRAME_HARD_FACTOR: u32 = 4;
/// bit of [`PYR_SPLIT_DEPTH`] that turns the in-frame gates off (host-written from
/// `AETHERVK_DUST_FRAME_GATES=0`; diagnostics: the frame's work is then unbounded)
pub const DUST_FRAME_GATES_OFF_BIT: u32 = 22;

/// `AETHERVK_DUST_JACOBIAN_AT_START=1` (diagnostics, host only: the CPU mirror reads the
/// process environment, the GPU bit 23 of [`PYR_SPLIT_DEPTH`]): the EWA linearisation at the
/// chord's start, as before the midpoint fix
pub fn jacobian_at_start() -> bool {
  #[cfg(feature = "std")]
  {
    std::env::var("AETHERVK_DUST_JACOBIAN_AT_START").is_ok_and(|v| v == "1")
  }
  #[cfg(not(feature = "std"))]
  {
    false
  }
}

/// Whether this frame's in-frame gates ([`DUST_FRAME_HARD_FACTOR`]) are on (bit
/// [`DUST_FRAME_GATES_OFF_BIT`] of [`PYR_SPLIT_DEPTH`] clear). Mirror of `dust_frame_gates_on`.
pub fn frame_gates_on(pyramid: &[u32]) -> bool {
  (pyramid[PYR_SPLIT_DEPTH as usize] >> DUST_FRAME_GATES_OFF_BIT) & 1 == 0
}
/// volumetric axis mode (bits 16–17 of [`PYR_SPLIT_DEPTH`], host-written from
/// `AETHERVK_DUST_VOLUME_AXIS`; [`volume_axis_from_str`]): pieces along the size chord only, the
/// leaf's other bounded axis kept as its `1.21·(L/2)²` Gaussian (today's behaviour, the default)
pub const DUST_VOLUME_AXIS_CHORD: u32 = 0;
/// pieces along the leaf's Gaussianised axis [`Packet::seg_t`] instead (uniform, centred as the
/// Gaussian was), the chord then as its own `1.21·(L/2)²` Gaussian about its ρ-weighted centre
pub const DUST_VOLUME_AXIS_TIME: u32 = 1;
/// pieces on both bounded axes: `n_c` along the chord × `n_t` along [`Packet::seg_t`], each the
/// leaf's covariance less the `seg_t` Gaussian plus the uniform pieces' variances; at most
/// [`DUST_VOLUME_PIECES_MAX`] in all, the evaluation cap reducing `n_t` first, then `n_c`
pub const DUST_VOLUME_AXIS_GRID: u32 = 2;

/// The volumetric axis mode of an `AETHERVK_DUST_VOLUME_AXIS` value: `time`, `grid`, else the
/// chord ([`DUST_VOLUME_AXIS_CHORD`]).
pub fn volume_axis_from_str(v: &str) -> u32 {
  match v.trim() {
    "time" => DUST_VOLUME_AXIS_TIME,
    "grid" => DUST_VOLUME_AXIS_GRID,
    _ => DUST_VOLUME_AXIS_CHORD,
  }
}
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
  /// `β·F`, equal cross-section bins). `w`: edges 0 and 16 hold their speed factors `√(β_j/β)`
  /// (the factors are linear in `j`, [`DustMoments::edge_factor`]); edge 1 holds the nucleus
  /// spin rate ω (rad/s) and edges 2..=4 its unit axis (particle-system axes), the spiral
  /// interpolation of the splat ([`spiral_point`]); edge 5 the cluster's heliocentric distance
  /// (AU) and edges 6..=8 the unit direction to the Sun (particle-system axes), the radiance
  /// weight ([`radiance_weight`])
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
  /// speed factor `√(β_j/β)` of edge `j`: linear in `j` ([`size_edge_sqrt_beta`]), from the
  /// stored factors of the first and last edge. Mirror of `dust_edge_factor`.
  pub fn edge_factor(&self, j: usize) -> f32 {
    let (lo, hi) = (self.edges[0][3], self.edges[SIZE_BINS as usize][3]);
    lo + (hi - lo) * (j as f32 / SIZE_BINS as f32)
  }
  /// nucleus spin rate ω (rad/s), 0 without rotation
  pub fn spin_rate(&self) -> f32 {
    self.edges[1][3]
  }
  /// unit spin axis (particle-system axes)
  pub fn spin_axis(&self) -> [f32; 3] {
    [self.edges[2][3], self.edges[3][3], self.edges[4][3]]
  }
  /// heliocentric distance of the cluster's mean (AU)
  pub fn helio_au(&self) -> f32 {
    self.edges[5][3]
  }
  /// unit direction from the cluster to the Sun (particle-system axes)
  pub fn sun_dir(&self) -> [f32; 3] {
    [self.edges[6][3], self.edges[7][3], self.edges[8][3]]
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
  /// exposure: the stream colour's alpha (1 = photometric: the pyramid holds radiance factors
  /// `τ · p · Φ(α) / r_h²`; the white point is applied by the composite)
  pub exposure: f32,
  /// counts per unit of exposure-scaled optical depth × px² (1 / fixed-point unit)
  pub inv_unit: f32,
  /// stream colour, 8 bits per channel (`r | g << 8 | b << 16`)
  pub color: u32,
  /// geometric albedo of the dust: every cluster's flux is weighted by `p · Φ(α) / r_h²`
  /// ([`radiance_weight`]); `≤ 0` draws the unweighted optical depth (tests, diagnostics)
  pub albedo: f32,
  /// particle-system local metres → clip
  pub mvp: [f32; 16],
  /// eye position in particle-system local metres (depth of the packets); `w` = the system's TTL
  /// (s), the age clock of the hue ramp ([`age_color`]; `≤ 0` = the stream colour)
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
    for (i, v) in super::phase::PHASE_HM_0DEG.iter().enumerate() {
      h[PYR_PHASE_LUT as usize + i] = v.to_bits();
    }
    h[PYR_SPLIT_DEPTH as usize] = DUST_SPLIT_DEPTH_MAX;
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
  // the nucleus spin in the spare `w` lanes (the factors are linear in `j`: the first and last
  // edge keep theirs, `DustMoments::edge_factor`)
  let w = frame.spin_vector();
  let omega = <f32 as FloatLike>::sqrt(dot3(w, w));
  let axis = if omega > 0.0 {
    scale3(w, 1.0 / omega)
  } else {
    [0.0, 0.0, 1.0]
  };
  edges[1][3] = omega;
  edges[2][3] = axis[0];
  edges[3][3] = axis[1];
  edges[4][3] = axis[2];
  // the heliocentric distance and the Sun direction of the mean, for the radiance weight
  let r_m = r.to_f32();
  let r_n = <f32 as FloatLike>::sqrt(dot3(r_m, r_m)).max(1.0);
  let sun = qrot(rot, scale3(r_m, -1.0 / r_n));
  edges[5][3] = r_n / super::AU_M as f32;
  edges[6][3] = sun[0];
  edges[7][3] = sun[1];
  edges[8][3] = sun[2];
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
  /// the leaf's own bounded axis that its covariance carries as a Gaussian, `1.21·(L/2)(L/2)ᵀ`
  /// centred on the capsule ([`CHORD_VAR_FACTOR`]): `L` = the time segment of a chord-major,
  /// sweep or cone-piece leaf, the size chord of a time-major leaf (scaled with its cone), zero
  /// for pooled leaves (their time axis is the exact capsule) and for depth-sliced children (their
  /// covariance no longer holds the term). The EWA ignores it; the volumetric path can integrate
  /// that axis as uniform pieces instead ([`DUST_VOLUME_AXIS_TIME`], [`DUST_VOLUME_AXIS_GRID`]).
  pub seg_t: [f32; 3],
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
    seg_t: scale3(h, 2.0),
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
    seg_t: [0.0; 3],
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
    seg_t: scale3(h, 2.0),
  }
}

/// The time segment of a sample is drawn as a **spiral about the nucleus centre** (the local
/// origin, [`spiral_point`]) when the nucleus turns by more than this angle (rad) between the
/// sample and its predecessor: young dust near the source, where the nucleus turns 13° per
/// 28-min sample at 12.4 h and the dust emitted in between lies on the spiral from the sample's
/// direction to the predecessor's, not on the chord (the chord sat at the predecessor's
/// direction all along, a third of the cone's width off at 5 km). Below it the straight
/// segment is the spiral to within `R·θ²/8`.
pub const SPIRAL_MIN_TURN: f32 = 0.01;
/// ... and only within this many sample intervals of the emission: the spiral identity holds
/// while the dust moves with the emitter's rotation; the radiation-pressure drift
/// (½·β·g·age², not turned by the spin) is ≤ 1 % of the spiral radius `v·age` within 8 samples
/// of 28 min, and grows without bound beyond.
pub const SPIRAL_MAX_AGE_SAMPLES: f32 = 8.0;

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

/// The spiral of a rigidly rotating emitter. The nucleus turns by `ω` about `axis` through the
/// local origin; its site `s` and jet direction `d` turn with it, so the dust of age `τ` (speed
/// `v`, now) sits at `R(−ωτ)·(s + vτ·d)` with `s`, `d` the site and direction *now*: a point
/// linear in `τ` once un-spun by `R(+ωτ)`. Two samples of ages `age_o` (`own`) and `age_p`
/// (`pred`) are un-spun, interpolated linearly in `τ(u) = age_o + u·(age_p − age_o)`, and
/// re-spun by `R(−ω·τ(u))`: exact for that emitter (the ejection speed and the site offset
/// enter only through the two samples; proof in `dust_v3.md`), and the straight segment when
/// `ω = 0`. Mirror of `dust_spiral_point`.
pub fn spiral_point(
  own: [f32; 3],
  pred: [f32; 3],
  age_o: f32,
  age_p: f32,
  axis: [f32; 3],
  omega: f32,
  u: f32,
) -> [f32; 3] {
  let l_o = rotate3(own, axis, omega * age_o);
  let l_p = rotate3(pred, axis, omega * age_p);
  let l = add3(l_o, scale3(sub3(l_p, l_o), u));
  rotate3(l, axis, -omega * (age_o + u * (age_p - age_o)))
}

/// The spiral gate of a sample with a predecessor ([`SPIRAL_MIN_TURN`],
/// [`SPIRAL_MAX_AGE_SAMPLES`]): `Some((axis, ω))` when its time pieces follow the spiral.
pub fn spiral_of(m: &DustMoments, p: &DustMoments) -> Option<([f32; 3], f32)> {
  let omega = m.spin_rate();
  let d_age = (p.age() - m.age()).max(0.0);
  (omega * d_age > SPIRAL_MIN_TURN && m.age() <= SPIRAL_MAX_AGE_SAMPLES * d_age)
    .then(|| (m.spin_axis(), omega))
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
    seg_t: scale3(h, 2.0),
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
  subpackets_swept(m, seg, pred, merged, true, None)
}

/// [`subpackets_merged`] with the time sweep decided on screen through `px`.
pub fn subpackets_swept(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> alloc::vec::Vec<Packet> {
  // the spiral of a rotating nucleus (`spiral_of`): the pieces then follow it, and there are
  // enough of them to draw it
  let spiral = pred.and_then(|p| spiral_of(m, p));
  let (age_o, age_p) = (m.age(), pred.map_or(m.age(), |p| p.age()));
  let along = |start: [f32; 3], end: [f32; 3], u: f32| -> [f32; 3] {
    match spiral {
      Some((axis, omega)) => spiral_point(start, end, age_o, age_p, axis, omega, u),
      None => add3(start, scale3(sub3(end, start), u)),
    }
  };
  let n_t = if spiral.is_some() {
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
        let mean = along(p.mean, pm, u0);
        let seg_t = sub3(along(p.mean, pm, u0 + inv), mean);
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
          seg_t: [0.0; 3],
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
      let base_packet = bin_packet(p, b, seg_t);
      let base = base_packet.cov;
      let pm = add3(own.mean, seg_t);
      for t in 0..n_t {
        let u0 = t as f32 * inv;
        let sc = r0 + (1.0 - r0) * (u0 + 0.5 * inv);
        let sc2 = sc * sc;
        let mean = along(own.mean, pm, u0);
        let seg_p = sub3(along(own.mean, pm, u0 + inv), mean);
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
          seg_t: scale3(base_packet.seg_t, sc),
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
        let shift = sub3(along(mid, mid_p, u0), mid);
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
          seg_t: scale3(hseg, 2.0),
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

/// The clip `w` of a local point under `mvp` (the depth along the view in a perspective
/// projection, up to the scale of the w row; constant 1 in an orthographic one). Mirror of
/// `dust_clip_w`.
pub fn clip_w(p: [f32; 3], mvp: &[f32; 16]) -> f32 {
  mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15]
}

/// Whether the size polyline of `m` is drawn as one pooled packet (mirror of
/// `dust_polyline_merged`): the two end edges and the middle one within [`DUST_MERGE_PX`] of the
/// mean on screen, **and** (perspective only) the 17 edges' depth span below
/// [`DUST_SPLIT_EPS`] of the depth at the mean. A syndyne along the line of sight is a point on
/// screen but hundreds of metres deep: pooled into one Gaussian ~175 m deep, its depth
/// distribution (pixel area, eye-plane cut, speck shell, phase angle) was wrong — the shell
/// hand-over read 7 % of its τ off (`deep_polyline_is_not_pooled`,
/// `shell_hand_over_matches_the_cut`). In an orthographic view `w` is constant and the depth
/// extent changes nothing.
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
  // the depth span of the edges (0 in an orthographic view: w is constant)
  let wm = clip_w(m.mean(), mvp);
  let (mut lo, mut hi) = (wm, wm);
  for j in 0..SIZE_EDGES as usize {
    let we = clip_w(m.edge(j), mvp);
    lo = lo.min(we);
    hi = hi.max(we);
  }
  hi - lo < DUST_SPLIT_EPS * wm
}

/// Whether the time segment from `m` to its predecessor `p` is shallow enough to be pooled with
/// the polylines (perspective: its depth span below [`DUST_SPLIT_EPS`] of the depth at `m`'s
/// mean; always in an orthographic view). Mirror of `dust_time_shallow`.
pub fn time_shallow(m: &DustMoments, p: &DustMoments, mvp: &[f32; 16]) -> bool {
  let (wm, wp) = (clip_w(m.mean(), mvp), clip_w(p.mean(), mvp));
  fabs(wp - wm) < DUST_SPLIT_EPS * wm || wp == wm
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
  /// photometry (`albedo > 0`): the packet's mean (local m), the in-plane local offset per pixel
  /// along x and y (`d = dx·kx + dy·ky`, the inverse of the projection's Jacobian within the
  /// view plane), the unit Sun direction and r_h (AU) at the mean, the eye, and the radiance
  /// weight at the mean (`PYR_TAU_MAX`): every texel then gets its own `p · Φ(α) / r_h²` at the
  /// dust it holds ([`texel_weight`]) — a packet wide against the eye distance spans a range of
  /// phase angles (the haze of a near perspective view). The eye is homogeneous: the direction
  /// to it from a point `x` is `eye − eye_w · x` — a perspective projection's eye point (local m,
  /// `eye_w` = 1) or an orthographic projection's eye at infinity along the view axis (the unit
  /// direction, `eye_w` = 0: an orthographic camera has no position, only a near plane, and a
  /// point eye at the camera's position lit the dust beside the camera at α → 180°, a forward
  /// glare no telescope sees)
  pub mean: [f32; 3],
  pub kx: [f32; 3],
  pub ky: [f32; 3],
  pub sun: [f32; 3],
  pub r_h_au: f32,
  pub eye: [f32; 3],
  pub eye_w: f32,
  pub albedo: f32,
  pub w_mean: f32,
}

/// The photometry of a cluster's packets: the dust's albedo (`≤ 0`: none), the unit Sun
/// direction and r_h (AU) at the cluster's mean ([`DustMoments::sun_dir`], [`DustMoments::helio_au`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Photometry {
  pub albedo: f32,
  pub sun: [f32; 3],
  pub r_h_au: f32,
}

impl Photometry {
  pub const NONE: Self = Self {
    albedo: 0.0,
    sun: [0.0, 0.0, 1.0],
    r_h_au: 1.0,
  };
}

/// The radiance weight at the dust a texel holds: the packet's mean displaced by the texel's
/// in-plane offset (`kx`, `ky`), the Sun direction and r_h moved with it (first order: the
/// packet is small against r_h), the phase angle to the eye. Mirror of `dust_texel_weight`.
pub fn texel_weight(p: &ScreenPacket, x: [f32; 2]) -> f32 {
  let (dx, dy) = (x[0] - p.mu[0], x[1] - p.mu[1]);
  let d = add3(scale3(p.kx, dx), scale3(p.ky, dy));
  let pos = add3(p.mean, d);
  // the heliocentric vector at the dust: −sun · r_h + d
  let r_vec = add3(scale3(p.sun, -p.r_h_au * AU_M_F32), d);
  let rn = <f32 as FloatLike>::sqrt(dot3(r_vec, r_vec)).max(1.0);
  let sun = scale3(r_vec, -1.0 / rn);
  super::phase::radiance_weight(
    p.albedo,
    sun,
    sub3(p.eye, scale3(pos, p.eye_w)),
    rn / AU_M_F32,
  )
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
  project_packet_lit(
    p,
    exposure,
    mvp,
    width,
    height,
    eye_local,
    &Photometry::NONE,
  )
}

/// The view direction of `mvp`: the depth gradient of the projection, unit; zero for a
/// degenerate matrix. Perspective: the `w` row as it is (`w > 0` in front of the camera by
/// definition: a camera inside the tail looking away from the nucleus has the scene's origin
/// behind it, and orienting the direction towards the origin reversed the eye plane there — the
/// in-front cut, the chord clip and the split kept the dust behind the camera and dropped the
/// dust in front). Orthographic: the clip-z row, whose sign says nothing about the camera, oriented
/// so that the scene's origin lies in front of the eye. The second value is `perspective`. The
/// projection ([`project_packet_lit`]) and the split ([`split_packet`]) use the same direction.
/// Mirror of `dust_view_forward`.
pub fn view_forward(mvp: &[f32; 16], eye_local: [f32; 3]) -> ([f32; 3], bool) {
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let r3 = row(3);
  let perspective = dot3(r3, r3) > 0.0;
  let fwd_raw = if perspective { r3 } else { row(2) };
  let fn2 = dot3(fwd_raw, fwd_raw);
  let fwd = if fn2 > 0.0 {
    let f = scale3(fwd_raw, 1.0 / <f32 as FloatLike>::sqrt(fn2));
    if !perspective && dot3(scale3(eye_local, -1.0), f) < 0.0 {
      scale3(f, -1.0)
    } else {
      f
    }
  } else {
    [0.0; 3]
  };
  (fwd, perspective)
}

/// `fᵀ Σ f`: the variance of the packet's Gaussian along the unit direction `f`. Mirror of
/// `dust_var_along`.
pub fn var_along(cov: &[f32; 6], f: [f32; 3]) -> f32 {
  cov[0] * f[0] * f[0]
    + cov[3] * f[1] * f[1]
    + cov[5] * f[2] * f[2]
    + 2.0 * (cov[1] * f[0] * f[1] + cov[2] * f[0] * f[2] + cov[4] * f[1] * f[2])
}

/// [`project_packet`] with the radiance weight ([`Photometry`]): the texels get
/// `p · Φ(α) / r_h²` at their own position ([`texel_weight`]). Mirror of `dust_project_packet`.
///
/// `None` when the packet's start is behind the eye (`w ≤ 0`: the projection is not finite
/// there; the split rules of [`split_packet`] drop, clip or slice a packet before it gets here,
/// so this is a guard, not the behind-the-eye rule), when nothing of it is in front of the eye
/// plane, or when it is entirely off screen.
pub fn project_packet_lit(
  p: &Packet,
  exposure: f32,
  mvp: &[f32; 16],
  width: u32,
  height: u32,
  eye_local: [f32; 3],
  phot: &Photometry,
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
  // Jacobian of pixel(x): d(px)/dx = (W/2)·(M₀ − ndc.x·M₃)/w, likewise y with M₁ and H/2, taken
  // at the chord's midpoint when its depth span is within DUST_SPLIT_EPS of its depth (|w_end −
  // w| ≤ ε·w: the split's own bound): the screen segment, the footprint and the pixel-area factor
  // √det(J Jᵀ) ∝ 1/w² are then exact to second order in the span (at the start they over-drew
  // such chords by the span's first-order term, 8 %). A deeper chord — one the split could not
  // shorten, its depth limit lowered by the children budget inside the coma — keeps the start:
  // no single linearisation holds for it, and the midpoint concentrated its light (1 km inside:
  // render/truth 1.42 → 1.94). Orthographic: M₃ = 0, w constant, the same bits at any point
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let w_end = dot3(r3, add3(p.mean, p.seg)) + m(3, 3);
  let shallow = (w_end - w).abs() <= DUST_SPLIT_EPS * w;
  let mid = add3(
    p.mean,
    scale3(p.seg, if shallow && !jacobian_at_start() { 0.5 } else { 0.0 }),
  );
  let wm = dot3(r3, mid) + m(3, 3);
  let (cj, wj) = if wm > 0.0 {
    ([dot3(r0, mid) + m(0, 3), dot3(r1, mid) + m(1, 3)], wm)
  } else {
    ([cx, cy], w)
  };
  let ndcj = [cj[0] / wj, cj[1] / wj];
  let jx = scale3(sub3(r0, scale3(r3, ndcj[0])), 0.5 * wf / wj);
  let jy = scale3(sub3(r1, scale3(r3, ndcj[1])), 0.5 * hf / wj);
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
  // local metres per pixel within the view plane: the inverse of the Jacobian's Gram matrix
  let (kx, ky) = {
    let detg = g[0] * g[2] - g[1] * g[1];
    if detg > 1e-30 {
      (
        scale3(sub3(scale3(jx, g[2]), scale3(jy, g[1])), 1.0 / detg),
        scale3(sub3(scale3(jy, g[0]), scale3(jx, g[1])), 1.0 / detg),
      )
    } else {
      ([0.0; 3], [0.0; 3])
    }
  };
  let det_g = (g[0] * g[2] - g[1] * g[1]).max(0.0);
  // the view direction and the photometric eye: the eye point when perspective, at infinity
  // along the view axis when orthographic (see `ScreenPacket::eye_w`)
  let (fwd, perspective) = view_forward(mvp, eye_local);
  let (eye_h, eye_w) = if perspective || dot3(fwd, fwd) == 0.0 {
    (eye_local, 1.0)
  } else {
    (scale3(fwd, -1.0), 0.0)
  };
  let w_mean = if phot.albedo > 0.0 {
    super::phase::radiance_weight(
      phot.albedo,
      phot.sun,
      sub3(eye_h, scale3(p.mean, eye_w)),
      phot.r_h_au,
    )
  } else {
    1.0
  };
  let half = 0.5 * (cov[0] + cov[2]);
  let dd = <f32 as FloatLike>::sqrt(
    (0.5 * (cov[0] - cov[2])) * (0.5 * (cov[0] - cov[2])) + cov[1] * cov[1],
  );
  let lam_max = (half + dd).max(DUST_PIXEL_FILTER_VAR);
  // the determinant with the anisotropy floor (f32 cancellation for very elongated packets)
  let det_c = (cov[0] * cov[2] - cov[1] * cov[1])
    .max(DUST_DET_ANISO_FLOOR * lam_max * lam_max)
    .max(1e-30);
  let mut amp = exposure * p.flux * <f32 as FloatLike>::sqrt(det_g)
    / (2.0 * core::f32::consts::PI * <f32 as FloatLike>::sqrt(det_c));
  // the part of the packet in front of the eye: its Gaussian along the view direction, centred
  // on the capsule's middle, cut at the eye's plane (the orthographic projection would draw the
  // dust behind the camera too: the fountain of a near view rises through it; the truth clips
  // at the camera's near plane). After the split rules a leaf has σ_v ≤ DUST_SPLIT_EPS · depth
  // and the cut is 1; it matters for the leaves the split budget or depth left straddling the
  // plane
  if dot3(fwd, fwd) > 0.0 {
    let depth = dot3(sub3(add3(p.mean, scale3(p.seg, 0.5)), eye_local), fwd);
    let sigma_v = <f32 as FloatLike>::sqrt(var_along(&p.cov, fwd).max(0.0));
    if sigma_v > 0.0 {
      amp *= 0.5 * (1.0 + erf(depth / (sigma_v * core::f32::consts::SQRT_2)));
    } else if depth < 0.0 {
      amp = 0.0;
    }
  }
  if !(amp > 0.0) || amp.is_infinite() {
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
    mean: p.mean,
    kx,
    ky,
    sun: phot.sun,
    r_h_au: phot.r_h_au,
    eye: eye_h,
    eye_w,
    albedo: phot.albedo,
    w_mean,
  })
}

// ─── Perspective split (mirror of dust_split_step / dust_project_split_scatter) ─

/// The split words of a frame, kept in the pyramid header: the children made
/// ([`PYR_SPLIT_CHILDREN`], `atomicAdd` on the GPU, informational), the largest dropped flux
/// ([`PYR_SPLIT_DROPPED_FLUX`], `atomicMax`) and the host-written depth limit
/// ([`PYR_SPLIT_DEPTH`]). The host mirror loads and stores them around each packet; a host
/// pyramid split across threads keeps one per partial pyramid and the merge sums the children
/// and takes the maximum of the dropped flux ([`PYR_SPLIT_OVERFLOW`] is derived, never summed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitStats {
  pub children: u32,
  pub dropped_flux: f32,
  /// this frame's split depth limit (`≤ DUST_SPLIT_DEPTH_MAX`)
  pub depth: u32,
  /// the in-frame gates are off this frame ([`frame_gates_on`]; diagnostics)
  pub gates_off: bool,
}

impl Default for SplitStats {
  fn default() -> Self {
    Self {
      children: 0,
      dropped_flux: 0.0,
      depth: DUST_SPLIT_DEPTH_MAX,
      gates_off: false,
    }
  }
}

impl SplitStats {
  pub fn load(pyramid: &[u32]) -> Self {
    Self {
      children: pyramid[PYR_SPLIT_CHILDREN as usize],
      dropped_flux: f32::from_bits(pyramid[PYR_SPLIT_DROPPED_FLUX as usize]),
      depth: (pyramid[PYR_SPLIT_DEPTH as usize] & 0xFF).min(DUST_SPLIT_DEPTH_MAX),
      gates_off: !frame_gates_on(pyramid),
    }
  }

  pub fn store(&self, pyramid: &mut [u32]) {
    pyramid[PYR_SPLIT_CHILDREN as usize] = self.children;
    pyramid[PYR_SPLIT_DROPPED_FLUX as usize] = self.dropped_flux.to_bits();
    let w = &mut pyramid[PYR_SPLIT_DEPTH as usize];
    *w = (*w & !0xFF) | self.depth.min(0xFF);
  }

  /// children beyond [`DUST_SPLIT_BUDGET`] this frame (0 within it; computed, not stored)
  pub fn overflow(&self) -> u32 {
    self.children.saturating_sub(DUST_SPLIT_BUDGET)
  }

  /// The depth limit for the next frame: one less when this frame's children exceeded the
  /// budget, one more (up to [`DUST_SPLIT_DEPTH_MAX`]) when they stayed below a quarter of it,
  /// else unchanged. The host writes it into [`PYR_SPLIT_DEPTH`] with the header (to be wired in
  /// `device/dust.rs` next to the τ_max readback); a frame's splat is then a pure function of its
  /// header, whatever the dispatch order.
  pub fn suggested_depth(&self) -> u32 {
    if self.children > DUST_SPLIT_BUDGET {
      self.depth.saturating_sub(1)
    } else if self.children < DUST_SPLIT_BUDGET / 4 {
      (self.depth + 1).min(DUST_SPLIT_DEPTH_MAX)
    } else {
      self.depth
    }
  }
}

/// The volume words of a frame: the ray-integral evaluations made ([`PYR_VOLUME_EVALS`]) and
/// the host-written level bias (bits 8–15 of [`PYR_SPLIT_DEPTH`]). The host mirror adds to the
/// evaluations as the GPU does; a merge of partial pyramids sums them.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VolumeStats {
  pub evals: u32,
  pub bias: u32,
  /// the volumetric axis mode ([`DUST_VOLUME_AXIS_CHORD`], bits 16–17 of word 247)
  pub axis: u32,
}

impl VolumeStats {
  pub fn load(pyramid: &[u32]) -> Self {
    let w = pyramid[PYR_SPLIT_DEPTH as usize];
    Self {
      evals: pyramid[PYR_VOLUME_EVALS as usize],
      bias: (w >> 8) & 0xFF,
      axis: (w >> 16) & 0x3,
    }
  }

  /// writes the evaluations, the bias (bits 8–15 of word 247) and the axis mode (bits 16–17);
  /// the depth bits are kept
  pub fn store(&self, pyramid: &mut [u32]) {
    pyramid[PYR_VOLUME_EVALS as usize] = self.evals;
    let w = &mut pyramid[PYR_SPLIT_DEPTH as usize];
    *w = (*w & !0x3_FF00) | (self.bias.min(0xFF) << 8) | ((self.axis & 0x3) << 16);
  }

  /// The level bias for the next frame: one more when this frame's evaluations passed
  /// [`DUST_VOLUME_FRAME_BUDGET`] (up to [`DUST_VOLUME_BIAS_MAX`]; a level quarters a packet's
  /// texels), one less (down to 0) below an eighth of it, else unchanged.
  pub fn suggested_bias(&self) -> u32 {
    if self.evals > DUST_VOLUME_FRAME_BUDGET {
      (self.bias + 1).min(DUST_VOLUME_BIAS_MAX)
    } else if self.evals < DUST_VOLUME_FRAME_BUDGET / 8 {
      self.bias.saturating_sub(1)
    } else {
      self.bias
    }
  }
}

/// The view the split sees: the eye (local m), the unit view direction ([`view_forward`]),
/// whether the projection is perspective (orthographic projections split only to cut at the eye
/// plane: the EWA is exact for them) and whether packets near the eye are drawn by the exact ray
/// integral ([`is_volumetric`]; the perspective projections, where the eye is a point: always
/// on in production, off in the tests of the split alone).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitView {
  pub eye: [f32; 3],
  pub fwd: [f32; 3],
  pub perspective: bool,
  pub volumetric: bool,
  /// the radius (local m) of the camera-centred shell whose grains the speck pass draws
  /// (`specks::PYR_SPECKS_DMAX`; ≤ 0: none): the continuum gives that mass up
  /// ([`split_packet_with`], `specks::ShellCut`)
  pub shell: f32,
}

impl SplitView {
  pub fn of(mvp: &[f32; 16], eye_local: [f32; 3]) -> Self {
    let (fwd, perspective) = view_forward(mvp, eye_local);
    Self {
      eye: eye_local,
      fwd,
      perspective,
      volumetric: perspective,
      shell: 0.0,
    }
  }
}

/// The leaves of a split: drawn by the EWA ([`project_packet_lit`]) or by the exact ray integral
/// ([`volume_scatter`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SplitLeaves {
  pub ewa: alloc::vec::Vec<Packet>,
  pub volume: alloc::vec::Vec<Packet>,
}

impl SplitLeaves {
  pub fn clear(&mut self) {
    self.ewa.clear();
    self.volume.clear();
  }
}

/// The parameter of the point of the segment `d` (from the origin) nearest to `q`, in `[0, 1]`.
/// Mirror of `dust_segment_param`.
pub fn segment_param(q: [f32; 3], d: [f32; 3]) -> f32 {
  let dd = dot3(d, d);
  if dd > 0.0 {
    (dot3(q, d) / dd).clamp(0.0, 1.0)
  } else {
    0.0
  }
}

/// Whether a packet is drawn by the exact ray integral: the view has an eye point
/// (`view.volumetric`) and `tr Σ > DUST_VOLUME_RATIO² · dist²`, `dist` the distance from the eye
/// to the packet's segment (the eye inside or near it, where one Jacobian at the mean cannot
/// stand for the projection). Implies the eye within 3 Mahalanobis σ (`|q|² < 9 λ_max ≤ 9 tr`)
/// and `σ_v > depth` (`σ_v² ≤ tr`, `depth ≤ dist`). Mirror of `dust_is_volumetric`.
pub fn is_volumetric(p: &Packet, view: &SplitView) -> bool {
  if !view.volumetric {
    return false;
  }
  let q = sub3(view.eye, p.mean);
  let r = sub3(q, scale3(p.seg, segment_param(q, p.seg)));
  let tr = p.cov[0] + p.cov[3] + p.cov[5];
  tr > DUST_VOLUME_RATIO * DUST_VOLUME_RATIO * dot3(r, r)
}

/// The lower Cholesky factor `L` of a covariance (`xx xy xz yy yz zz`) as `l00 l10 l20 l11 l21
/// l22`, every pivot² floored at [`DUST_DET_ANISO_FLOOR`] · tr Σ. Mirror of `dust_cholesky3`.
pub fn cholesky3(s: &[f32; 6]) -> [f32; 6] {
  let fl = DUST_DET_ANISO_FLOOR * (s[0] + s[3] + s[5]).max(1e-30);
  let l00 = <f32 as FloatLike>::sqrt(s[0].max(fl));
  let l10 = s[1] / l00;
  let l20 = s[2] / l00;
  let l11 = <f32 as FloatLike>::sqrt((s[3] - l10 * l10).max(fl));
  let l21 = (s[4] - l20 * l10) / l11;
  let l22 = <f32 as FloatLike>::sqrt((s[5] - l20 * l20 - l21 * l21).max(fl));
  [l00, l10, l20, l11, l21, l22]
}

/// `L⁻¹ x` (forward substitution): `x` whitened by the covariance whose factor is `l`. Mirror of
/// `dust_cholesky_solve3`.
pub fn cholesky_solve3(l: &[f32; 6], x: [f32; 3]) -> [f32; 3] {
  let y0 = x[0] / l[0];
  let y1 = (x[1] - l[1] * y0) / l[3];
  let y2 = (x[2] - l[2] * y0 - l[4] * y1) / l[5];
  [y0, y1, y2]
}

/// Pyramid level of a volumetric packet of angular σ `theta_px` (px): texels of `2^ℓ` px with
/// [`DUST_VOLUME_TEXELS_PER_SIGMA`] per σ, the last level at most. Mirror of `dust_volume_level`.
pub fn volume_level(theta_px: f32, levels: u32) -> u32 {
  let r = theta_px / DUST_VOLUME_TEXELS_PER_SIGMA;
  if !(r > 1.0) {
    return 0;
  }
  let l = <f32 as FloatLike>::floor(<f32 as FloatLike>::ln(r) * core::f32::consts::LOG2_E);
  (l.max(0.0) as u32).min(levels.saturating_sub(1))
}

/// The radiance weight of the dust at `x` seen along the ray `u` (unit, from the eye): the Sun
/// direction and r_h moved by `x − mean_ref` as [`texel_weight`] does. Mirror of
/// `dust_volume_weight`.
fn volume_weight(phot: &Photometry, mean_ref: [f32; 3], x: [f32; 3], u: [f32; 3]) -> f32 {
  let d = sub3(x, mean_ref);
  let r_vec = add3(scale3(phot.sun, -phot.r_h_au * AU_M_F32), d);
  let rn = <f32 as FloatLike>::sqrt(dot3(r_vec, r_vec)).max(1.0);
  let sun = scale3(r_vec, -1.0 / rn);
  super::phase::radiance_weight(phot.albedo, sun, scale3(u, -1.0), rn / AU_M_F32)
}

/// One Gaussian piece of a volumetric packet: its Cholesky factor, the eye whitened by it
/// (`L⁻¹(eye − mean)`), `exposure · flux / (2π √det Σ)`, its projected ±3σ box (px: x0 x1 y0 y1),
/// whether it covers the whole frame (the eye within [`DUST_VOLUME_SIGMAS`] of it or a box corner
/// not in front of the eye) and whether it is on screen.
#[derive(Debug, Clone, Copy, Default)]
struct VolumePiece {
  l: [f32; 6],
  qw: [f32; 3],
  k: f32,
  bbox: [f32; 4],
  full: bool,
  on: bool,
}

/// The EWA words of a frame: the texels planned ([`PYR_EWA_TEXELS`], in units of
/// [`DUST_EWA_TEXEL_UNIT`]) and the frame's EWA bias (bits 18–21 of [`PYR_SPLIT_DEPTH`],
/// host-written: the per-packet texel cap is `DUST_SPLAT_MAX_TEXELS / 4^bias`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EwaStats {
  /// texels (not units)
  pub texels: u64,
  pub bias: u32,
}

impl EwaStats {
  pub fn load(pyramid: &[u32]) -> Self {
    Self {
      texels: pyramid[PYR_EWA_TEXELS as usize] as u64 * DUST_EWA_TEXEL_UNIT as u64,
      bias: ((pyramid[PYR_SPLIT_DEPTH as usize] >> 18) & 0xF).min(DUST_EWA_BIAS_MAX),
    }
  }

  /// writes the bias (bits 18–21 of [`PYR_SPLIT_DEPTH`]); the other bits are kept
  pub fn store_bias(bias: u32, pyramid: &mut [u32]) {
    let w = &mut pyramid[PYR_SPLIT_DEPTH as usize];
    *w = (*w & !(0xF << 18)) | (bias.min(DUST_EWA_BIAS_MAX) << 18);
  }

  /// The bias for the next frame: over [`DUST_EWA_FRAME_BUDGET`], up by `⌈log₄(texels /
  /// budget)⌉` at once (a level quarters a capped packet's texels); below an eighth of it, one
  /// down; else unchanged.
  pub fn suggested_bias(&self) -> u32 {
    let budget = DUST_EWA_FRAME_BUDGET as u64;
    if self.texels > budget {
      let ratio = self.texels as f64 / budget as f64;
      let up = (ratio.ln() / 4f64.ln()).ceil().max(1.0) as u32;
      (self.bias + up).min(DUST_EWA_BIAS_MAX)
    } else if self.texels < budget / 8 {
      self.bias.saturating_sub(1)
    } else {
      self.bias
    }
  }
}

/// The work plan of a volumetric packet ([`volume_plan`]): its level, its pieces and the
/// evaluations they make there (Σ over the pieces of the texels in each one's range).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumePlan {
  pub level: u32,
  pub pieces: u32,
  pub evals: u32,
}

/// The axes a volumetric packet is integrated along (`volume_plan_pieces`): the base point and
/// covariance (pixel filter included, the gridded Gaussian axis removed), axis `a` (the chord:
/// uniform with the ρ ramp, from the base point) with `n_a` pieces of variance `cvar_a`·spacing²,
/// and when `has_b` axis `b` ([`Packet::seg_t`]: uniform, centred on the base point) likewise.
#[derive(Debug, Clone, Copy)]
struct VolumeAxes {
  mean: [f32; 3],
  s: [f32; 6],
  seg_a: [f32; 3],
  n_a: u32,
  cvar_a: f32,
  seg_b: [f32; 3],
  n_b: u32,
  cvar_b: f32,
  has_b: bool,
}

/// Builds the `n_a · n_b` pieces of a volumetric packet (piece `i · n_b + j`). Without axis `b`
/// the arithmetic is exactly the chord-only pieces'. Mirror of `dust_volume_pieces`.
#[allow(clippy::too_many_arguments)]
fn volume_pieces(
  p: &Packet,
  exposure: f32,
  ax: &VolumeAxes,
  eye: [f32; 3],
  mvp: &[f32; 16],
  wf: f32,
  hf: f32,
  out: &mut [VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
) {
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let s = &ax.s;
  let cvar = ax.cvar_a;
  let inv_n = 1.0 / ax.n_a as f32;
  let h = scale3(ax.seg_a, inv_n);
  let inv_b = 1.0 / ax.n_b as f32;
  let hb = scale3(ax.seg_b, inv_b);
  for k in 0..(ax.n_a * ax.n_b) as usize {
    let pc = &mut out[k];
    let (i, j) = (k / ax.n_b as usize, k % ax.n_b as usize);
    let uk = (i as f32 + 0.5) * inv_n;
    let mut mean_k = add3(ax.mean, scale3(ax.seg_a, uk));
    let mut sk = [
      s[0] + cvar * h[0] * h[0],
      s[1] + cvar * h[0] * h[1],
      s[2] + cvar * h[0] * h[2],
      s[3] + cvar * h[1] * h[1],
      s[4] + cvar * h[1] * h[2],
      s[5] + cvar * h[2] * h[2],
    ];
    let mut flux_k = p.flux * (p.rho[0] + (p.rho[1] - p.rho[0]) * uk) * inv_n;
    if ax.has_b {
      let vk = (j as f32 + 0.5) * inv_b - 0.5;
      mean_k = add3(mean_k, scale3(ax.seg_b, vk));
      let cb = ax.cvar_b;
      sk[0] += cb * hb[0] * hb[0];
      sk[1] += cb * hb[0] * hb[1];
      sk[2] += cb * hb[0] * hb[2];
      sk[3] += cb * hb[1] * hb[1];
      sk[4] += cb * hb[1] * hb[2];
      sk[5] += cb * hb[2] * hb[2];
      flux_k *= inv_b;
    }
    let lk = cholesky3(&sk);
    let qw = cholesky_solve3(&lk, sub3(eye, mean_k));
    let mut full = dot3(qw, qw) < DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS;
    let ext = [
      3.0 * <f32 as FloatLike>::sqrt(sk[0]),
      3.0 * <f32 as FloatLike>::sqrt(sk[3]),
      3.0 * <f32 as FloatLike>::sqrt(sk[5]),
    ];
    let (mut x0, mut x1, mut y0, mut y1) = (1e30f32, -1e30f32, 1e30f32, -1e30f32);
    if !full {
      for c in 0..8u32 {
        let sg = |bit: u32, v: f32| if c & bit != 0 { v } else { -v };
        let pt = [
          mean_k[0] + sg(1, ext[0]),
          mean_k[1] + sg(2, ext[1]),
          mean_k[2] + sg(4, ext[2]),
        ];
        let w = dot3(r3, pt) + m(3, 3);
        if !(w > 0.0) {
          full = true;
          break;
        }
        let px = ((dot3(r0, pt) + m(0, 3)) / w * 0.5 + 0.5) * wf;
        let py = ((dot3(r1, pt) + m(1, 3)) / w * 0.5 + 0.5) * hf;
        x0 = x0.min(px);
        x1 = x1.max(px);
        y0 = y0.min(py);
        y1 = y1.max(py);
      }
    }
    let on = full || !(x1 < 0.0 || y1 < 0.0 || x0 > wf || y0 > hf || !(x0 <= x1) || !(y0 <= y1));
    *pc = VolumePiece {
      l: lk,
      qw,
      k: exposure * flux_k / (2.0 * core::f32::consts::PI * (lk[0] * lk[3] * lk[5])),
      bbox: [x0, x1, y0, y1],
      full,
      on,
    };
  }
}

/// The texel range `(x0, x1, y0, y1)` of a piece at `level` (empty when off screen, the whole
/// grid when full). Mirror of `dust_volume_range`.
fn volume_range(pc: &VolumePiece, level: u32, layout: &PyramidLayout) -> (i32, i32, i32, i32) {
  let (_, lw, lh) = layout.levels[level as usize];
  if !pc.on {
    return (0, -1, 0, -1);
  }
  if pc.full {
    return (0, lw as i32 - 1, 0, lh as i32 - 1);
  }
  let sz = (1u32 << level) as f32;
  (
    fl(pc.bbox[0] / sz).max(0.0) as i32,
    ce(pc.bbox[1] / sz).min((lw - 1) as f32) as i32,
    fl(pc.bbox[2] / sz).max(0.0) as i32,
    ce(pc.bbox[3] / sz).min((lh - 1) as f32) as i32,
  )
}

/// Σ over the first `n` pieces of the texels in their ranges at `level`. Mirror of
/// `dust_volume_evals`.
fn volume_evals(
  pieces: &[VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
  n: u32,
  level: u32,
  layout: &PyramidLayout,
) -> u32 {
  let mut total = 0u32;
  for pc in pieces.iter().take(n as usize) {
    let (x0, x1, y0, y1) = volume_range(pc, level, layout);
    total = total.wrapping_add(((x1 - x0 + 1).max(0) * (y1 - y0 + 1).max(0)) as u32);
  }
  total
}

/// Plans a volumetric packet's work and builds its pieces: one piece per
/// [`DUST_VOLUME_PIECE_SIGMAS`] Mahalanobis σ of the chord (≤ [`DUST_VOLUME_PIECES_MAX`]); the
/// level from the angular σ ([`DUST_VOLUME_TEXELS_PER_SIGMA`]), at least
/// [`DUST_VOLUME_MIN_LEVEL`] when a piece covers the whole frame, plus the frame's `bias`; then
/// coarsened until the evaluations fit `max_evals`, and at the last level the pieces merged in
/// pairs (with the capped piece variance) until they do. `None` when nothing is on screen.
/// Mirror of the planning part of `dust_volume_scatter`.
#[allow(clippy::too_many_arguments)]
fn volume_plan_pieces(
  p: &Packet,
  exposure: f32,
  mvp: &[f32; 16],
  view: &SplitView,
  layout: &PyramidLayout,
  bias: u32,
  axis: u32,
  max_evals: u32,
  pieces: &mut [VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
) -> Option<VolumePlan> {
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let (wf, hf) = (layout.width as f32, layout.height as f32);
  let eye = view.eye;
  let n3 = dot3(r3, r3);
  if !(n3 > 0.0) {
    return None;
  }
  // the focal length (px per radian at the centre)
  let f_px = 0.5
    * (wf * <f32 as FloatLike>::sqrt(dot3(r0, r0)))
      .min(hf * <f32 as FloatLike>::sqrt(dot3(r1, r1)))
    / <f32 as FloatLike>::sqrt(n3);
  if !(f_px > 0.0) {
    return None;
  }
  let q0 = sub3(eye, p.mean);
  let r_seg = sub3(q0, scale3(p.seg, segment_param(q0, p.seg)));
  let dist = <f32 as FloatLike>::sqrt(dot3(r_seg, r_seg));
  let rel = dist / f_px;
  let var_f = DUST_PIXEL_FILTER_VAR * rel * rel;
  let s = [
    p.cov[0] + var_f,
    p.cov[1],
    p.cov[2],
    p.cov[3] + var_f,
    p.cov[4],
    p.cov[5] + var_f,
  ];
  let l = cholesky3(&s);
  // the axes: the chord only (default), or the leaf's Gaussianised axis `seg_t` as uniform
  // pieces instead of its 1.21 Gaussian (time: the chord then Gaussianised; grid: both gridded)
  let mode = if axis != DUST_VOLUME_AXIS_CHORD && dot3(p.seg_t, p.seg_t) > 0.0 {
    axis
  } else {
    DUST_VOLUME_AXIS_CHORD
  };
  let kf = CHORD_VAR_FACTOR;
  let ht = scale3(p.seg_t, 0.5);
  let less_t = [
    s[0] - kf * ht[0] * ht[0],
    s[1] - kf * ht[0] * ht[1],
    s[2] - kf * ht[0] * ht[2],
    s[3] - kf * ht[1] * ht[1],
    s[4] - kf * ht[1] * ht[2],
    s[5] - kf * ht[2] * ht[2],
  ];
  let mut ax = if mode == DUST_VOLUME_AXIS_TIME {
    // the chord as its 1.21 Gaussian about its ρ-weighted centre (∫ u ρ(u) du)
    let hc = scale3(p.seg, 0.5);
    let ubar = p.rho[0] * 0.5 + (p.rho[1] - p.rho[0]) / 3.0;
    VolumeAxes {
      mean: add3(p.mean, scale3(p.seg, ubar)),
      s: [
        less_t[0] + kf * hc[0] * hc[0],
        less_t[1] + kf * hc[0] * hc[1],
        less_t[2] + kf * hc[0] * hc[2],
        less_t[3] + kf * hc[1] * hc[1],
        less_t[4] + kf * hc[1] * hc[2],
        less_t[5] + kf * hc[2] * hc[2],
      ],
      seg_a: [0.0; 3],
      n_a: 1,
      cvar_a: 1.0 / 12.0,
      seg_b: p.seg_t,
      n_b: 1,
      cvar_b: 1.0 / 12.0,
      has_b: true,
    }
  } else if mode == DUST_VOLUME_AXIS_GRID {
    VolumeAxes {
      mean: p.mean,
      s: less_t,
      seg_a: p.seg,
      n_a: 1,
      cvar_a: 1.0 / 12.0,
      seg_b: p.seg_t,
      n_b: 1,
      cvar_b: 1.0 / 12.0,
      has_b: true,
    }
  } else {
    VolumeAxes {
      mean: p.mean,
      s,
      seg_a: p.seg,
      n_a: 1,
      cvar_a: 1.0 / 12.0,
      seg_b: [0.0; 3],
      n_b: 1,
      cvar_b: 1.0 / 12.0,
      has_b: false,
    }
  };
  // pieces per axis: one per DUST_VOLUME_PIECE_SIGMAS Mahalanobis σ of the base covariance
  let lb = cholesky3(&ax.s);
  let count = |seg: [f32; 3]| -> (u32, f32) {
    let dw = cholesky_solve3(&lb, seg);
    let len_w = <f32 as FloatLike>::sqrt(dot3(dw, dw)) / DUST_VOLUME_PIECE_SIGMAS;
    if len_w > DUST_VOLUME_PIECES_MAX as f32 {
      (DUST_VOLUME_PIECES_MAX, DUST_VOLUME_PIECE_VAR_CAPPED)
    } else {
      ((ce(len_w) as u32).max(1), 1.0 / 12.0)
    }
  };
  (ax.n_a, ax.cvar_a) = count(ax.seg_a);
  if ax.has_b {
    (ax.n_b, ax.cvar_b) = count(ax.seg_b);
  }
  // at most DUST_VOLUME_PIECES_MAX pieces in all: fewer along `seg_t` first
  while ax.n_a * ax.n_b > DUST_VOLUME_PIECES_MAX {
    if ax.n_b > 1 {
      ax.n_b = ax.n_b.div_ceil(2);
      ax.cvar_b = DUST_VOLUME_PIECE_VAR_CAPPED;
    } else {
      ax.n_a = ax.n_a.div_ceil(2);
      ax.cvar_a = DUST_VOLUME_PIECE_VAR_CAPPED;
    }
  }
  // the level from the angular σ
  let tr = s[0] + s[3] + s[5];
  let sig_small = 2.0 * (l[0] * l[3] * l[5]) / tr;
  let den = dist.max(sig_small);
  let theta = if den > 0.0 { sig_small / den } else { 1.0 };
  let levels = layout.level_count();
  let mut level = volume_level(theta * f_px, levels);
  volume_pieces(p, exposure, &ax, eye, mvp, wf, hf, pieces);
  let (mut any_full, mut any_on) = (false, false);
  for pc in pieces.iter().take((ax.n_a * ax.n_b) as usize) {
    any_full |= pc.full;
    any_on |= pc.on;
  }
  if !any_on {
    return None;
  }
  if any_full {
    level = level.max(DUST_VOLUME_MIN_LEVEL).min(levels - 1);
  }
  level = (level + bias).min(levels - 1);
  // the evaluation cap: a coarser level first, then fewer pieces (along `seg_t` first)
  loop {
    if volume_evals(pieces, ax.n_a * ax.n_b, level, layout) <= max_evals || level + 1 >= levels {
      break;
    }
    level += 1;
  }
  while volume_evals(pieces, ax.n_a * ax.n_b, level, layout) > max_evals {
    if ax.n_b > 1 {
      ax.n_b = ax.n_b.div_ceil(2);
      ax.cvar_b = DUST_VOLUME_PIECE_VAR_CAPPED;
    } else if ax.n_a > 1 {
      ax.n_a = ax.n_a.div_ceil(2);
      ax.cvar_a = DUST_VOLUME_PIECE_VAR_CAPPED;
    } else {
      break;
    }
    volume_pieces(p, exposure, &ax, eye, mvp, wf, hf, pieces);
  }
  let n = ax.n_a * ax.n_b;
  Some(VolumePlan {
    level,
    pieces: n,
    evals: volume_evals(pieces, n, level, layout),
  })
}

/// The work plan [`volume_scatter`] makes for `p` with the level bias `bias` under the cap
/// `max_evals` ([`DUST_VOLUME_MAX_EVALS`] in production; tests pass `u32::MAX` to see the
/// uncapped work). `None` when nothing is on screen.
pub fn volume_plan(
  p: &Packet,
  mvp: &[f32; 16],
  view: &SplitView,
  layout: &PyramidLayout,
  bias: u32,
  axis: u32,
  max_evals: u32,
) -> Option<VolumePlan> {
  let mut pieces = [VolumePiece::default(); DUST_VOLUME_PIECES_MAX as usize];
  volume_plan_pieces(
    p,
    1.0,
    mvp,
    view,
    layout,
    bias,
    axis,
    max_evals,
    &mut pieces,
  )
}

/// Draws a packet by the **exact ray integral** of its density (mirror of `dust_volume_scatter`):
/// per texel of one pyramid level, the optical depth along the ray from the eye through the
/// texel centre, `τ = flux ∫₀^∞ G(eye + s u) ds` (the column of cross-section, the same unit as
/// the EWA's `τ`: a pixel's cone holds `∫ n Ω s² ds` grains, each `σ / (Ω s²)` of its angular
/// area). In closed form with `L Lᵀ = Σ`, `u' = L⁻¹u`, `q' = L⁻¹(eye − μ)`, `a = |u'|²`,
/// `b = u'·q'`: `τ = flux / (2π √det Σ √a) · exp(−½ |q' − (b/a) u'|²) · Φ(−b/√a)` — the half-line
/// `s ≥ 0` is the cut at the eye (a ray from the eye sees nothing behind it). A chord is
/// integrated as Gaussian pieces along it ([`DUST_VOLUME_PIECE_SIGMAS`] apart; each
/// `Σ + c (d/N)(d/N)ᵀ`, `c = 1/12` (exact moments) or [`DUST_VOLUME_PIECE_VAR_CAPPED`]; flux
/// `ρ(u_k)/N`: the linear density by the midpoint rule, exact). The pixel filter of the EWA,
/// `(0.5 px)²`, is added isotropically at the packet's distance, so a far packet draws the same
/// image either way (`volumetric_and_ewa_agree_for_a_far_packet`). The work is planned first
/// ([`volume_plan`]: level, pieces, the frame's level bias from the header, the evaluation cap
/// [`DUST_VOLUME_MAX_EVALS`]); each piece is then evaluated over its own texel range only (its
/// projected ±3σ box, the whole frame when the eye is near it), so the evaluations made are
/// exactly the plan's, added to [`PYR_VOLUME_EVALS`]. Counts, colour words, depth word, error
/// diffusion and the peak word (per piece, as per EWA packet) as [`scatter_into`] /
/// [`splat_scatter`]. Returns the texel deposits made.
#[allow(clippy::too_many_arguments)]
pub fn volume_scatter(
  p: &Packet,
  exposure: f32,
  mvp: &[f32; 16],
  view: &SplitView,
  phot: &Photometry,
  color: [f32; 3],
  layout: &PyramidLayout,
  inv_unit: f32,
  pyramid: &mut [u32],
) -> u32 {
  let vs = VolumeStats::load(pyramid);
  let (bias, axis) = (vs.bias, vs.axis);
  let mut pieces = [VolumePiece::default(); DUST_VOLUME_PIECES_MAX as usize];
  let Some(plan) = volume_plan_pieces(
    p,
    exposure,
    mvp,
    view,
    layout,
    bias,
    axis,
    DUST_VOLUME_MAX_EVALS,
    &mut pieces,
  ) else {
    return 0;
  };
  let ev = &mut pyramid[PYR_VOLUME_EVALS as usize];
  let before = *ev;
  *ev = ev.wrapping_add(plan.evals);
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let (wf, hf) = (layout.width as f32, layout.height as f32);
  let eye = view.eye;
  // the in-frame gate (mirror of `dust_volume_scatter`): past DUST_FRAME_HARD_FACTOR budgets this
  // frame, at least [`gate_level`] (the evaluations only shrink)
  let level = if frame_gates_on(pyramid) && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET {
    plan.level.max(gate_level(layout.level_count()))
  } else {
    plan.level
  };
  let (off, lw, _) = layout.levels[level as usize];
  let sz = (1u32 << level) as f32;
  let area = sz * sz;
  let depth_bits = (<f32 as FloatLike>::sqrt(dot3(sub3(eye, p.mean), sub3(eye, p.mean)))
    / AU_M_F32)
    .max(0.0)
    .to_bits();
  let mut written = 0u32;
  let mut carry = 0.0f32;
  let mut peak = 0.0f32;
  for pc in pieces.iter().take(plan.pieces as usize) {
    let (tx0, tx1, ty0, ty1) = volume_range(pc, level, layout);
    for ty in ty0..=ty1 {
      let yc = (ty as f32 + 0.5) * sz;
      let ny = 2.0 * yc / hf - 1.0;
      let b_row = sub3(r1, scale3(r3, ny));
      for tx in tx0..=tx1 {
        let xc = (tx as f32 + 0.5) * sz;
        let nx = 2.0 * xc / wf - 1.0;
        let a_row = sub3(r0, scale3(r3, nx));
        // the ray through the texel centre: on both planes of constant ndc, in front of the eye
        let mut v = cross3(a_row, b_row);
        if dot3(v, r3) < 0.0 {
          v = scale3(v, -1.0);
        }
        let vn = <f32 as FloatLike>::sqrt(dot3(v, v));
        if !(vn > 0.0) {
          continue;
        }
        let u = scale3(v, 1.0 / vn);
        let uw = cholesky_solve3(&pc.l, u);
        let a = dot3(uw, uw);
        let b = dot3(uw, pc.qw);
        let rr = sub3(pc.qw, scale3(uw, b / a));
        let sa = <f32 as FloatLike>::sqrt(a);
        let mut tau_px = pc.k / sa
          * <f32 as FloatLike>::exp(-0.5 * dot3(rr, rr))
          * (0.5 * (1.0 + erf(-b / (sa * core::f32::consts::SQRT_2))));
        if phot.albedo > 0.0 {
          let x = add3(eye, scale3(u, (-b / a).max(0.0)));
          tau_px *= volume_weight(phot, p.mean, x, u);
        }
        if !(tau_px >= 0.0) || !tau_px.is_finite() {
          continue;
        }
        if tau_px > peak {
          peak = tau_px;
        }
        let tau = tau_px * area * inv_unit;
        let tau_c = tau.min(DUST_COUNT_MAX_PER_ADD as f32) + carry;
        let counts = (tau_c + 0.5) as u32;
        carry = tau_c - counts as f32;
        if counts == 0 {
          continue;
        }
        let i = (off + (ty as u32 * lw + tx as u32) * PYRAMID_TEXEL_WORDS) as usize;
        sat_add(&mut pyramid[i], counts);
        let hfr = (dust_hash32(i as u32) >> 8) as f32 * (1.0 / 16_777_216.0);
        for k in 0..3 {
          sat_add(
            &mut pyramid[i + 1 + k],
            (counts as f32 * color[k] * DUST_COLOR_SCALE + hfr) as u32,
          );
        }
        if pyramid[i + 4] == 0 || depth_bits < pyramid[i + 4] {
          pyramid[i + 4] = depth_bits;
        }
        written += 1;
      }
    }
  }
  let cur = f32::from_bits(pyramid[PYR_TAU_MAX as usize]);
  if peak > 0.0 && !(cur >= peak) {
    pyramid[PYR_TAU_MAX as usize] = peak.to_bits();
  }
  if written > 0 {
    pyramid[PYR_LEVEL_MASK as usize] |= 1 << level;
  }
  written
}

/// What [`split_step`] decided for a packet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitStep {
  /// wholly behind the eye plane
  Drop,
  /// a leaf: project and scatter it
  Keep,
  /// the chord crossed the eye plane: the in-front part replaces the packet (same level)
  Clip(Packet),
  /// two children of half the flux each
  Split(Packet, Packet),
  /// the eye is inside or near it: drawn by the exact ray integral ([`volume_scatter`])
  Volume,
}

/// The sub-chord `u0 ≤ u ≤ u1` of a chord packet: start moved along the segment, the segment and
/// the flux scaled by the kept part (the density is linear along the chord, so the flux is the
/// kept length times the mean density there), `rho` re-interpolated and renormalised to mean 1,
/// the ends as given. Mirror of `dust_sub_chord`.
pub fn sub_chord(p: &Packet, u0: f32, u1: f32, open_start: bool, open_end: bool) -> Packet {
  let r = |u: f32| p.rho[0] + (p.rho[1] - p.rho[0]) * u;
  let (ra, rb) = (r(u0), r(u1));
  let rm = (0.5 * (ra + rb)).max(1e-6);
  Packet {
    mean: add3(p.mean, scale3(p.seg, u0)),
    cov: p.cov,
    seg: scale3(p.seg, u1 - u0),
    flux: p.flux * (u1 - u0) * rm,
    open: [open_start, open_end],
    rho: [ra / rm, rb / rm],
    seg_t: p.seg_t,
  }
}

/// One step of the perspective split of a packet at split level `level` (`clipped`: it is the
/// in-front part of a chord already cut at the eye plane), seen from `view.eye` along the unit
/// view direction `view.fwd` ([`view_forward`]). With `depth = (mean − eye)·f` the depth of the
/// chord's start, `σ_v² = fᵀΣf` and `span = seg·f` the chord's signed depth span, the rules in
/// order:
///
/// 1. wholly behind the eye plane, `depth + 3σ_v + max(0, span) < 0`: [`SplitStep::Drop`];
/// 2. the chord crosses the plane (its ends on both sides): [`SplitStep::Clip`] to the part in
///    front, less [`DUST_SPLIT_CLIP_EPS`] of it so the kept start is strictly in front (the
///    perspective projection diverges on the plane); the cut end is closed (a Gaussian cap);
/// 3. `3σ_v > depth` (the ellipsoid crosses the plane: the one-sided cut of
///    [`project_packet_lit`] is then a poor estimate of the part in front) or, in perspective
///    only (the EWA is exact for an orthographic projection), `σ_v > DUST_SPLIT_EPS · depth`
///    (the EWA linearisation error) or `|span| > DUST_SPLIT_EPS · depth`, while
///    `level < depth_max` (the frame's limit, [`PYR_SPLIT_DEPTH`] ≤ [`DUST_SPLIT_DEPTH_MAX`]):
///    [`SplitStep::Split`] — the chord bisected (inner ends open,
///    `rho` linear: the two halves tile the same capsule exactly) when `|span| > 3σ_v`, else the
///    Gaussian **sliced by depth**: with `g = Σf / σ_v` (the mean lateral position regressed on
///    the depth coordinate) and `h = a·g`, `a` = [`DUST_SPLIT_A`], the children are
///    `N(μ ∓ h, Σ − hhᵀ)` with half the flux each;
/// 4. wanting a split at the depth limit in perspective: [`SplitStep::Volume`] (the exact ray
///    integral; no single Jacobian holds for it);
/// 5. else [`SplitStep::Keep`].
///
/// Proof of 3 (the mixture has the packet's moments): the mean of `½N(μ−h, Σ') + ½N(μ+h, Σ')` is
/// `μ`, its covariance `Σ' + hhᵀ = Σ`; the flux halves sum exactly. `Σ' = Σ − a²(Σf)(Σf)ᵀ/(fᵀΣf)`
/// is positive semidefinite: `Σ − (Σf)(Σf)ᵀ/(fᵀΣf)` is the covariance of the Gaussian
/// conditional on its depth coordinate (a Schur complement, ⪰ 0) and the rest is
/// `(1 − a²)(Σf)(Σf)ᵀ/(fᵀΣf) ⪰ 0`. The children's depth variance is `σ_v²(1 − a²)`: `σ_v`
/// shrinks by 0.6 per level and the EWA error (∝ depth extent / depth, the pixel scale and area
/// varying across the packet's depth) with it; the two-point rule is exact to the third moment
/// and the image does not resolve depth, so the mixture's bimodality along `f` is invisible.
/// The lateral extent is not split: the EWA integrates every texel along the central ray, which
/// leaves an obliquity error of order `(σ_lateral / depth)²` (6 % of the flux for
/// `σ_lateral / depth` = 0.25, measured by `perspective_split_converges_to_the_truth`). A
/// packet whose lateral position is strongly correlated with its depth (a tilted thin slab) has
/// children offset laterally too, and their image ripples: 3 % of the flux at a correlation of
/// 0.7, 11 % at 0.9 after four forced levels (`split_image_matches_the_unsplit_packet` prints
/// them); an uncorrelated or depth-aligned packet projects to the same picture exactly.
/// Mirror of `dust_split_step` (same arithmetic, same order).
pub fn split_step(
  p: &Packet,
  clipped: bool,
  level: u32,
  depth_max: u32,
  view: &SplitView,
  force: bool,
) -> SplitStep {
  let (eye, fwd) = (view.eye, view.fwd);
  if !(dot3(fwd, fwd) > 0.0) {
    return SplitStep::Keep;
  }
  let depth = dot3(sub3(p.mean, eye), fwd);
  let sigma_v = <f32 as FloatLike>::sqrt(var_along(&p.cov, fwd).max(0.0));
  let span_s = dot3(p.seg, fwd);
  let span = fabs(span_s);
  // (1) wholly behind the eye plane
  if depth + 3.0 * sigma_v + span_s.max(0.0) < 0.0 {
    return SplitStep::Drop;
  }
  // (2) the chord crosses the plane: keep the part in front, its start a margin inside
  let d1 = depth + span_s;
  if !clipped && span > 0.0 && (depth < 0.0) != (d1 < 0.0) {
    let u_star = -depth / span_s;
    if depth < 0.0 {
      let u_lo = u_star + DUST_SPLIT_CLIP_EPS * (1.0 - u_star);
      return SplitStep::Clip(sub_chord(p, u_lo, 1.0, false, p.open[1]));
    }
    let u_hi = u_star * (1.0 - DUST_SPLIT_CLIP_EPS);
    return SplitStep::Clip(sub_chord(p, 0.0, u_hi, p.open[0], false));
  }
  // (2b) the eye inside or near the packet: the exact ray integral, no split, no EWA
  if !force && is_volumetric(p, view) {
    return SplitStep::Volume;
  }
  // (3) too deep for one Jacobian: bisect the chord or slice the Gaussian by depth
  let want = force
    || 3.0 * sigma_v > depth
    || (view.perspective && (sigma_v > DUST_SPLIT_EPS * depth || span > DUST_SPLIT_EPS * depth));
  if want && level < depth_max.min(DUST_SPLIT_DEPTH_MAX) {
    if span > 3.0 * sigma_v {
      return SplitStep::Split(
        sub_chord(p, 0.0, 0.5, p.open[0], true),
        sub_chord(p, 0.5, 1.0, true, p.open[1]),
      );
    }
    if sigma_v > 0.0 {
      let c = &p.cov;
      let k = DUST_SPLIT_A / sigma_v;
      let h = [
        (c[0] * fwd[0] + c[1] * fwd[1] + c[2] * fwd[2]) * k,
        (c[1] * fwd[0] + c[3] * fwd[1] + c[4] * fwd[2]) * k,
        (c[2] * fwd[0] + c[4] * fwd[1] + c[5] * fwd[2]) * k,
      ];
      let cov = [
        c[0] - h[0] * h[0],
        c[1] - h[0] * h[1],
        c[2] - h[0] * h[2],
        c[3] - h[1] * h[1],
        c[4] - h[1] * h[2],
        c[5] - h[2] * h[2],
      ];
      let child = |mean: [f32; 3]| Packet {
        mean,
        cov,
        seg: p.seg,
        flux: 0.5 * p.flux,
        open: p.open,
        rho: p.rho,
        seg_t: [0.0; 3],
      };
      return SplitStep::Split(child(sub3(p.mean, h)), child(add3(p.mean, h)));
    }
  }
  // (4) still too deep for one Jacobian at the depth limit (inside the coma the children budget
  // lowers the limit to 0): the exact ray integral instead of a wrong linearisation (the EWA at
  // the start or the midpoint of such a chord drew 1.42× / 1.94× the truth at 1 km)
  if want && !force && view.perspective && view.volumetric {
    return SplitStep::Volume;
  }
  SplitStep::Keep
}

/// The leaves of the perspective split of `p` ([`split_step`] applied depth first to
/// `stats.depth` levels; with the speck shell `view.shell` > 0, the leaves wholly inside it are
/// the specks' and left out, those across it split once more while the depth allows, else kept
/// with `flux × outside_fraction`: `specks::ShellCut`, mirror of the same branch of
/// `dust_project_split_scatter`).
///
/// In full: [`split_step`] applied depth first to
/// `stats.depth` levels, the frame's limit), pushed onto `out` in a fixed order; `stats` counts
/// the children made and the largest dropped flux. The host lowers the next frame's depth when
/// a frame passes the budget ([`SplitStats::suggested_depth`]); the one run-time gate is at
/// [`DUST_FRAME_HARD_FACTOR`] budgets (the children counted before a split: past it, no further
/// split), which depends on the order only in frames the feedback has not caught up with (a
/// cold start inside the coma made 5.5·10⁷ children). `force` splits every node to the
/// depth limit (tests). An explicit stack of `DUST_SPLIT_DEPTH_MAX + 1` nodes, the same loop as
/// the GLSL mirror `dust_project_split_scatter`, which scatters each leaf as it is found.
pub fn split_packet_with(
  p: &Packet,
  view: &SplitView,
  force: bool,
  out: &mut SplitLeaves,
  stats: &mut SplitStats,
) {
  const N: usize = (DUST_SPLIT_DEPTH_MAX + 1) as usize;
  let mut stack = [(*p, 0u32, false); N];
  let mut n = 1usize;
  // the in-frame gate (mirror of `dust_project_split_scatter`): once a split finds the frame's
  // children past DUST_FRAME_HARD_FACTOR budgets, this packet's nodes split no further
  let mut depth = stats.depth;
  while n > 0 {
    n -= 1;
    let (q, level, clipped) = stack[n];
    match split_step(&q, clipped, level, depth, view, force) {
      SplitStep::Drop => {
        if !(stats.dropped_flux >= q.flux) {
          stats.dropped_flux = q.flux;
        }
      }
      step @ (SplitStep::Keep | SplitStep::Volume) => {
        // the speck shell's hand-over: a leaf wholly inside is the specks' (not drawn), one
        // across the boundary is split once more while the depth allows, else drawn with the
        // fraction of its mass outside
        let push = |out: &mut SplitLeaves, r: Packet| {
          if matches!(step, SplitStep::Volume) {
            out.volume.push(r);
          } else {
            out.ewa.push(r);
          }
        };
        let shell = super::specks::ShellCut::new(view.eye, view.shell);
        match shell.classify(&q) {
          super::specks::ShellClass::Outside => push(out, q),
          super::specks::ShellClass::Inside => {}
          super::specks::ShellClass::Straddles => {
            let forced = if level < depth.min(DUST_SPLIT_DEPTH_MAX) {
              split_step(&q, clipped, level, depth, view, true)
            } else {
              SplitStep::Keep
            };
            if let SplitStep::Split(a, b) = forced {
              if !stats.gates_off && stats.children > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET {
                depth = 0;
              }
              stats.children += 2;
              stack[n] = (b, level + 1, false);
              stack[n + 1] = (a, level + 1, false);
              n += 2;
            } else {
              let mut r = q;
              r.flux *= shell.outside_fraction(&q);
              if r.flux > 0.0 {
                push(out, r);
              }
            }
          }
        }
      }
      SplitStep::Clip(a) => {
        stack[n] = (a, level, true);
        n += 1;
      }
      SplitStep::Split(a, b) => {
        if !stats.gates_off && stats.children > DUST_FRAME_HARD_FACTOR * DUST_SPLIT_BUDGET {
          depth = 0;
        }
        stats.children += 2;
        stack[n] = (b, level + 1, false);
        stack[n + 1] = (a, level + 1, false);
        n += 2;
      }
    }
  }
}

/// [`split_packet_with`] under the production rules (no forcing).
pub fn split_packet(p: &Packet, view: &SplitView, out: &mut SplitLeaves, stats: &mut SplitStats) {
  split_packet_with(p, view, false, out, stats)
}

/// Splits `p` ([`split_packet`]), projects every EWA leaf ([`project_packet_lit`]) and scatters
/// it ([`splat_scatter`]), draws every volumetric leaf by its ray integral ([`volume_scatter`]);
/// the split counters live in the pyramid header. `scratch` is reused between packets. Returns
/// the leaves drawn. Mirror of `dust_project_split_scatter` (which draws each leaf as it is
/// found: the deposits are order-free sums).
#[allow(clippy::too_many_arguments)]
pub fn project_split_scatter(
  p: &Packet,
  exposure: f32,
  mvp: &[f32; 16],
  view: &SplitView,
  phot: &Photometry,
  color: [f32; 3],
  layout: &PyramidLayout,
  inv_unit: f32,
  pyramid: &mut [u32],
  scratch: &mut SplitLeaves,
) -> u32 {
  scratch.clear();
  let mut stats = SplitStats::load(pyramid);
  split_packet(p, view, scratch, &mut stats);
  stats.store(pyramid);
  let mut scattered = 0;
  for leaf in scratch.volume.iter() {
    if volume_scatter(
      leaf, exposure, mvp, view, phot, color, layout, inv_unit, pyramid,
    ) > 0
    {
      scattered += 1;
    }
  }
  for leaf in scratch.ewa.iter() {
    if let Some(sp) = project_packet_lit(
      leaf,
      exposure,
      mvp,
      layout.width,
      layout.height,
      view.eye,
      phot,
    ) {
      splat_scatter(&sp, color, layout, inv_unit, pyramid);
      scattered += 1;
    }
  }
  scattered
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
  // the squared distance to the line, from the residual `e − u₀ d` (not `c − b²/a`: for a chord
  // whose start is 10⁴ px off screen both terms are ~10⁷ and their f32 difference was off by ~10,
  // τ by up to e⁵ on screen; the residual's error is |e|·ε ≈ 10⁻³ px)
  let r = [e[0] - u0 * d[0], e[1] - u0 * d[1]];
  let q = (m00 * r[0] * r[0] + 2.0 * m01 * r[0] * r[1] + m11 * r[1] * r[1]).max(0.0);
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

/// Hue ramp position of an age: `clamp(age / TTL, 0, 1)` — a linear clock (a viewer reads the hue
/// back to an age; smoothstep would hide the first and last 15 % of a TTL), the complement at one
/// TTL of the system and constant beyond (the older tiers are all complement). A TTL `≤ 0` means no
/// rotation (the stream colour). Mirror of `dust_age_hue_fraction`.
pub fn age_hue_fraction(age_s: f32, ttl_s: f32) -> f32 {
  if ttl_s > 0.0 {
    (age_s / ttl_s).clamp(0.0, 1.0)
  } else {
    0.0
  }
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

/// the plain OKLab → linear conversion, unclamped (tests)
pub fn linear_from_oklab_pub(lab: [f32; 3]) -> [f32; 3] {
  linear_from_oklab(lab)
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

/// Linear sRGB of an OKLab colour brought into the `[0, 1]` gamut by reducing its chroma —
/// lightness and hue kept — with 8 bisection steps on the chroma scale (no early exit: both
/// mirrors run the same instructions), then clamped. A colour already in gamut comes back as the
/// plain conversion. Mirror of `dust_oklab_to_gamut`.
pub fn oklab_to_gamut(lab: [f32; 3]) -> [f32; 3] {
  let inside = |c: [f32; 3]| {
    c[0] >= -1e-4
      && c[0] <= 1.0 + 1e-4
      && c[1] >= -1e-4
      && c[1] <= 1.0 + 1e-4
      && c[2] >= -1e-4
      && c[2] <= 1.0 + 1e-4
  };
  let full = linear_from_oklab(lab);
  let mut rgb = full;
  if !inside(full) {
    // k ∈ [lo, hi]: lo is always in gamut (k = 0 is the grey of lightness L, inside for L in [0, 1])
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let mut best = linear_from_oklab([lab[0], 0.0, 0.0]);
    for _ in 0..8 {
      let k = 0.5 * (lo + hi);
      let c = linear_from_oklab([lab[0], k * lab[1], k * lab[2]]);
      if inside(c) {
        lo = k;
        best = c;
      } else {
        hi = k;
      }
    }
    rgb = best;
  }
  [
    rgb[0].clamp(0.0, 1.0),
    rgb[1].clamp(0.0, 1.0),
    rgb[2].clamp(0.0, 1.0),
  ]
}

/// Colour of dust of age `age_s` in a system of TTL `ttl_s`: the stream colour's hue rotated in
/// OKLab by `π · f(age)` ([`age_hue_fraction`]): the stream colour at the jet, its complement from
/// one TTL on, lightness kept, chroma kept where the gamut allows ([`oklab_to_gamut`]). Mirror of
/// `dust_age_color`.
pub fn age_color(stream: [f32; 3], age_s: f32, ttl_s: f32) -> [f32; 3] {
  let lab = oklab_from_linear([
    stream[0].clamp(0.0, 1.0),
    stream[1].clamp(0.0, 1.0),
    stream[2].clamp(0.0, 1.0),
  ]);
  let th = core::f32::consts::PI * age_hue_fraction(age_s, ttl_s);
  let (s, c) = (<f32 as FloatLike>::sin(th), <f32 as FloatLike>::cos(th));
  oklab_to_gamut([lab[0], c * lab[1] - s * lab[2], s * lab[1] + c * lab[2]])
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

/// The fraction `u1 − u0` of the screen segment `mu + u · seg`, `u ∈ [0, 1]`, inside the frame
/// `[−ex, w + ex] × [−ey, h + ey]` (Liang–Barsky; 0 when it misses it): the length of a chord
/// the band can draw, for the texel estimate of [`splat_scatter`]'s valve. Mirror of
/// `dust_visible_fraction`.
pub fn visible_fraction(mu: [f32; 2], seg: [f32; 2], ex: f32, ey: f32, w: f32, h: f32) -> f32 {
  let (mut u0, mut u1) = (0.0f32, 1.0f32);
  let clips = [
    (-seg[0], mu[0] + ex),
    (seg[0], w + ex - mu[0]),
    (-seg[1], mu[1] + ey),
    (seg[1], h + ey - mu[1]),
  ];
  for (pk, qk) in clips {
    if pk == 0.0 {
      if qk < 0.0 {
        return 0.0;
      }
    } else {
      let t = qk / pk;
      if pk < 0.0 {
        u0 = u0.max(t);
      } else {
        u1 = u1.min(t);
      }
    }
  }
  if u1 > u0 { u1 - u0 } else { 0.0 }
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
  let peak = p.amp * p.w_mean;
  if !(cur >= peak) {
    pyramid[PYR_TAU_MAX as usize] = peak.to_bits();
  }
  let mut level = layout.level_for_sigma(p.sigma_min);
  let ex = DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(p.cov[0]);
  let ey = DUST_SPLAT_SIGMAS * <f32 as FloatLike>::sqrt(p.cov[2]);
  // the part of the segment the band can draw: inside the frame expanded by the band's own
  // margin (the rows and columns clip the drawn texels to the frame; the estimate must too, or
  // a chord thousands of pixels long is pushed to 16–32-px texels for its off-screen length and
  // smears its τ and colour onto the young pixels around it)
  let seg_len = <f32 as FloatLike>::sqrt(p.seg[0] * p.seg[0] + p.seg[1] * p.seg[1])
    * visible_fraction(
      p.mu,
      p.seg,
      ex,
      ey,
      layout.width as f32,
      layout.height as f32,
    );
  // safety valve against runaway threads: a packet whose capsule band would visit more than its
  // cap goes to a coarser level (blur, never a stall). The cap is DUST_SPLAT_MAX_TEXELS
  // quartered per level of the frame's EWA bias ([`EwaStats`], host-fed from
  // [`PYR_EWA_TEXELS`]: inside the coma ~10⁶ packets cross the whole screen)
  let ewa_bias = EwaStats::load(pyramid).bias;
  let cap = DUST_SPLAT_MAX_TEXELS as f32 / (1u32 << (2 * ewa_bias)) as f32;
  let band_at = |level: u32| {
    let s = (1u32 << level) as f32;
    (seg_len / s + 2.0 * ex / s + 2.0) * (2.0 * ey / s + 2.0)
  };
  let mut band = band_at(level);
  while band > DUST_SPLAT_MAX_TEXELS as f32 && level + 1 < layout.level_count() {
    level += 1;
    band = band_at(level);
  }
  // the level the per-packet valve alone gives: beyond it (the bias, the gate) the packet is
  // widened to half its texels below, so it blurs instead of missing every texel centre
  let level_valve = level;
  // the bias stops at the gate level (≤ 16 texels a packet there; the 1 × 1 level loses a widened
  // packet's light off its single texel)
  while band > cap && level + 1 < layout.level_count() && level < gate_level(layout.level_count()) {
    level += 1;
    band = band_at(level);
  }
  // the band's texels inside the level (the rows and columns clip to it): the work counted, and
  // the in-frame gate (mirror of `scatter`): past DUST_FRAME_HARD_FACTOR budgets this frame, at
  // least [`gate_level`] (a frame the host feedback has not caught up with is bounded at once)
  {
    let (_, lw0, lh0) = layout.levels[level as usize];
    let work = band.min(lw0 as f32 * lh0 as f32);
    let t = &mut pyramid[PYR_EWA_TEXELS as usize];
    let before = *t;
    *t = t.wrapping_add((work / DUST_EWA_TEXEL_UNIT) as u32 + 1);
    if frame_gates_on(pyramid)
      && before > DUST_FRAME_HARD_FACTOR * (DUST_EWA_FRAME_BUDGET / DUST_EWA_TEXEL_UNIT as u32)
    {
      level = level.max(gate_level(layout.level_count()));
    }
  }
  let widened;
  let p = if level > level_valve {
    let s = (1u32 << level) as f32;
    widened = widen_screen_packet(p, 0.25 * s * s);
    &widened
  } else {
    p
  };
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

/// The level a packet past an in-frame gate ([`DUST_FRAME_HARD_FACTOR`]) is drawn at least at:
/// two below the coarsest, a grid of ~4 × 3 texels at any resolution (≤ 16 texels a packet; the
/// 1 × 1 level spread a widened packet mostly off the frame)
pub fn gate_level(levels: u32) -> u32 {
  levels.saturating_sub(3)
}

/// `p` with `v` px² added to both axes of its screen covariance, its integral kept (`amp ×
/// √(det / det′)`, the determinant floored as in [`project_packet_lit`]): a packet the EWA bias
/// or the in-frame gate sends to texels of `s` px coarser than its width is drawn with
/// `v = s²/4` (σ ≥ s/2: the texel-centre sum is its integral within `2·exp(−π²/2)` ≈ 1.4 %),
/// not point-sampled at their centres (a thin chord at a coarse level missed them all and its
/// light was lost).
/// Mirror of `dust_widen_screen_packet`.
pub fn widen_screen_packet(p: &ScreenPacket, v: f32) -> ScreenPacket {
  let mut q = *p;
  q.cov[0] += v;
  q.cov[2] += v;
  let half = 0.5 * (q.cov[0] + q.cov[2]);
  let dd = <f32 as FloatLike>::sqrt(
    (0.5 * (q.cov[0] - q.cov[2])) * (0.5 * (q.cov[0] - q.cov[2])) + q.cov[1] * q.cov[1],
  );
  let lam_max = (half + dd).max(DUST_PIXEL_FILTER_VAR);
  let det = (q.cov[0] * q.cov[2] - q.cov[1] * q.cov[1])
    .max(DUST_DET_ANISO_FLOOR * lam_max * lam_max)
    .max(1e-30);
  q.amp = p.amp * <f32 as FloatLike>::sqrt(p.det / det);
  q.det = det;
  q.sigma_min = <f32 as FloatLike>::sqrt((det / lam_max).max(DUST_PIXEL_FILTER_VAR * 0.5));
  q
}

/// 32-bit integer hash (lowbias32): the colour words' rounding threshold per texel, the same
/// integer arithmetic on both mirrors. Mirror of `dust_hash32`.
pub fn dust_hash32(mut x: u32) -> u32 {
  x ^= x >> 16;
  x = x.wrapping_mul(0x7feb_352d);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846c_a68b);
  x ^= x >> 16;
  x
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
/// `full` the colour and the nearest depth too. The colour words are `⌊16 · counts · c + h⌋`
/// ([`DUST_COLOR_SCALE`]) with a
/// hashed threshold `h ∈ [0, 1)` per texel (an ordered dither): rounding `counts · c + 0.5` per
/// deposit turned every faint texel (one-count deposits) into (1, 1, 0) whatever the age — r and
/// g rounded up, b down — and the tail's faint regions read 30° off their hue; with the threshold
/// a packet's faint deposits round up at some texels and down at others, so every box of texels
/// averages to the colour. The threshold is a function of the texel alone: a per-cluster seed
/// (the compact index or the cluster id) put the two mirrors' words 10 % apart at faint texels,
/// and an error-diffusion carry coupled a texel's words to dithered counts elsewhere in the
/// packet's loop; both failed the GPU parity. Mirror of `scatter_into` in `dust_splat.comp`.
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
      let mut tau = capsule_tau(p, x) * area * inv_unit;
      if p.albedo > 0.0 {
        tau *= texel_weight(p, x);
      }
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
        let hf = (dust_hash32(i as u32) >> 8) as f32 * (1.0 / 16_777_216.0);
        for k in 0..3 {
          sat_add(
            &mut pyramid[i + 1 + k],
            (counts as f32 * color[k] * DUST_COLOR_SCALE + hf) as u32,
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
      sat_add(&mut pyramid[i], counts);
      for k in 1..4 {
        sat_add(
          &mut pyramid[i + k],
          counts.saturating_mul(DUST_COLOR_SCALE as u32),
        );
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
  // the speck shell of the frame, read once per tier (0 when there is none: orthographic
  // views, the specks off)
  let mut view = SplitView::of(&pc.mvp, eye);
  view.shell = super::specks::ShellCut::from_header(pyramid, eye).d_max;
  let mut scratch = SplitLeaves::default();
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  // `AETHERVK_DUST_PROBE="x,y;x,y"` (diagnostics, CPU particle mode): per cluster, the τ it adds
  // at those pixels (composite read before and after), the largest contributors logged per tier
  #[cfg(feature = "std")]
  let probes = probe_pixels();
  #[cfg(feature = "std")]
  let unit = if pc.inv_unit > 0.0 { 1.0 / pc.inv_unit } else { 1.0 };
  #[cfg(feature = "std")]
  let mut probe_log: alloc::vec::Vec<alloc::vec::Vec<(f32, usize)>> =
    alloc::vec![alloc::vec::Vec::new(); probes.len()];
  #[cfg(feature = "std")]
  let mut probe_total = alloc::vec![0.0f32; probes.len()];
  for i in range.start..range.end.min(n) {
    let m = &moments[i];
    if !(m.flux() > 0.0) {
      continue;
    }
    #[cfg(feature = "std")]
    let before: alloc::vec::Vec<f32> = probes
      .iter()
      .map(|&(x, y)| composite_sample(layout, pyramid, unit, x, y).0)
      .collect();
    splat_cluster_inner(
      moments, render, pc, flow, tracer_counts, layout, pyramid, &view, &mut scratch, stream, eye,
      i,
    );
    #[cfg(feature = "std")]
    for (k, &(x, y)) in probes.iter().enumerate() {
      let d = composite_sample(layout, pyramid, unit, x, y).0 - before[k];
      if d > 0.0 {
        probe_total[k] += d;
        probe_log[k].push((d, i));
      }
    }
  }
  #[cfg(feature = "std")]
  for (k, &(x, y)) in probes.iter().enumerate() {
    let log = &mut probe_log[k];
    log.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
    let tot = probe_total[k].max(1e-30);
    let mut line = alloc::format!(
      "[Dust probe] tier live {} pixel ({x},{y}): τ {:.4e} from {} clusters; top:",
      n,
      probe_total[k],
      log.len()
    );
    for &(d, i) in log.iter().take(8) {
      let m = &moments[i];
      let dm = sub3(m.mean(), eye);
      let dist = <f32 as FloatLike>::sqrt(dot3(dm, dm));
      let fwd = view.fwd;
      let depth = dot3(dm, fwd);
      let cv = m.cov_v();
      let tr = cv[0] + cv[3] + cv[5];
      line += &alloc::format!(
        " [{:.1}% idx {i} age {:.0}s dist {:.0}m depth {:.0}m σ̄_v {:.0}m pred {}]",
        100.0 * d / tot,
        m.age(),
        dist,
        depth,
        <f32 as FloatLike>::sqrt((tr / 3.0).max(0.0)),
        streak_pred(render, i).is_some()
      );
    }
    aethervk_oshal_rlib::log!("{}", line);
  }
}

/// `AETHERVK_DUST_PROBE="x,y;x,y"`: the probe pixels of [`splat_tier_range`] (diagnostics)
#[cfg(feature = "std")]
fn probe_pixels() -> alloc::vec::Vec<(u32, u32)> {
  std::env::var("AETHERVK_DUST_PROBE")
    .ok()
    .map(|v| {
      v.split(';')
        .filter_map(|p| {
          let mut it = p.split(',');
          Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
        })
        .collect()
    })
    .unwrap_or_default()
}

/// One cluster of [`splat_tier_range`] (its body; split out for the probe).
#[allow(clippy::too_many_arguments)]
fn splat_cluster_inner(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  view: &SplitView,
  scratch: &mut SplitLeaves,
  stream: [f32; 3],
  eye: [f32; 3],
  i: usize,
) {
  {
    let m = &moments[i];
    let pred_i = streak_pred(render, i);
    let pred = pred_i.map(|q| &moments[q]);
    let seg = match pred {
      Some(p) => sub3(p.mean(), m.mean()),
      None => [0.0; 3],
    };
    let mut exposure = pc.exposure;
    if pc.flags & DUST_VIEW_FLOW != 0 {
      exposure *= flow_factor(m.age(), flow);
    }
    // physical photometry: every texel of the packets gets `p · Φ(α) / r_h²` at the dust it holds
    let phot = Photometry {
      albedo: pc.albedo,
      sun: m.sun_dir(),
      r_h_au: m.helio_au(),
    };
    let color = age_color(stream, m.age(), pc.eye_local[3]);
    // pooled only when the predecessor's polyline is below a pixel too: a just-emitted sample is
    // a point whose time pieces are shaped by its predecessor, and a predecessor whose sizes
    // already span kilometres must be drawn bin by bin (the halo behind the nucleus of the 5-km
    // frame came from its pooled covariance)
    let pred_merged = pred.is_none_or(|p| polyline_merged(p, &pc.mvp, layout.width, layout.height));
    // (perspective: neither the polylines nor the time segment between them may be deep)
    let merged = polyline_merged(m, &pc.mvp, layout.width, layout.height)
      && pred_merged
      && pred.is_none_or(|p| time_shallow(m, p, &pc.mvp));
    let px = |q: [f32; 3]| project_point(q, &pc.mvp, layout.width, layout.height);
    // every sub-packet is split for the perspective (clipped at the eye plane, sliced by depth:
    // `split_packet`) and its leaves are projected and scattered one by one
    for p in subpackets_swept(m, seg, pred, merged, pred_merged, Some(&px)) {
      project_split_scatter(
        &p,
        exposure,
        &pc.mvp,
        view,
        &phot,
        color,
        layout,
        pc.inv_unit,
        pyramid,
        scratch,
      );
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
        seg_t: [0.0; 3],
      };
      if let Some(sp) = project_packet(&p, 1.0, &pc.mvp, layout.width, layout.height, eye) {
        splat_tracer(sp.mu, tracer_counts, layout, pyramid);
      }
    }
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
        rgb[k] += wgt * pyramid[i + 1 + k] as f32 * (scale / DUST_COLOR_SCALE);
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
const _USED: u32 = CHILD_ID_MASK;

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
