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
//! predecessor's to the cluster's (time pieces, [`TIME_PIECES_MAX`], at equal width ratios
//! [`piece_bound`] and turned with the nucleus spiral: the youngest samples are cones, not
//! sticks; each piece is the sample that would have been taken at its centre age).
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
  flow_factor, is_tracer, kepler, qrot, render_live, render_stream_shift,
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
/// f32 bits: the cluster level of detail's pixels per drawn segment ([`lod_px`]; 0 = off, every
/// cluster at full detail), host-written per frame (the budget loop, `AETHERVK_DUST_LOD_PX`)
pub const PYR_LOD: u32 = 61;
/// the level of detail's resting value (the user's gate of 2026-10-16: the lab matrix is unchanged
/// to four digits at 1 km / 100 m / 10 m up to 4 px with 23 % fewer packets; 8 px degrades 100 m):
/// what the host writes to [`PYR_LOD`] when `AETHERVK_DUST_LOD_PX` is unset and the budget loop
/// is at rest
pub const DUST_LOD_PX_DEFAULT: f32 = 4.0;
/// the budget loop's ceiling for a moving frame (the last acceptable point of the matrix)
pub const DUST_LOD_PX_MAX: f32 = 8.0;
/// the fixed-point unit this frame's counts were written with (f32 bits, host-written; 0 when
/// unknown): what a dump reader needs to turn counts back into optical depth
pub const PYR_UNIT: u32 = 15;
/// per level `ℓ`: `(offset in words, width, height)` at `PYR_TABLE + 3ℓ`
pub const PYR_TABLE: u32 = 16;
/// the view's queue start in words from this pyramid's first word (host-written per frame and per
/// class image: the GPU shaders' `dust_queue_base`; the cadence's class images sit between the
/// frame pyramid and the queues, so an image's word is the queue start less its own offset; the
/// CPU mirror has no queues and never reads it)
pub const PYR_QUEUE_BASE: u32 = 58;
/// the most levels a pyramid holds: its table ends below [`PYR_QUEUE_BASE`] (14 levels: a
/// viewport up to 8 192 px wide; 15 before the queue word took the table's last entry)
pub const PYRAMID_MAX_LEVELS: u32 = (PYR_QUEUE_BASE - PYR_TABLE) / 3;
const _: () = assert!(PYR_TABLE + 3 * PYRAMID_MAX_LEVELS <= PYR_QUEUE_BASE);
const _: () = assert!(PYR_QUEUE_BASE < PYR_LOD);
/// diagnostics, the `.prof` volume consumer only (`DUST_PYR_PROF_PAIR`): two counters chosen by
/// the frame's [`DUST_FRAME_PROF_PAIR_SHIFT`] mode (atomicAdd; 0 in production, the host uploads
/// the header words as 0 each frame). 0: the packets found off screen at pass A (`!anyOn`), then
/// their lane cycles / 256 up to that return; 1: the packets whose pieces all land on the last
/// level (`allTop`), their lane cycles / 256 to the end; 2: those packets' pieces, then the SMALL
/// ones among them; 3: as 1 for the allTop packets whose pieces are all SMALL
pub const PYR_PROF_PAIR: u32 = 59;
const _: () = assert!(PYR_QUEUE_BASE < PYR_PROF_PAIR && PYR_PROF_PAIR + 1 < PYR_LOD);
/// [`PYR_FLAGS`] bits 18–19 (host-written, `AETHERVK_DUST_PROF_PAIR=0..3`): the mode of
/// [`PYR_PROF_PAIR`]; read by the `.prof` consumer only
pub const DUST_FRAME_PROF_PAIR_SHIFT: u32 = 18;
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
/// diagnostics of the volume consumer (Xid 31 investigation): the pieces whose packed code named a
/// level past the pyramid or an empty level row (`atomicAdd`; 0 in a healthy frame), and the first
/// such piece's `code | k << 16 | n << 24` (`atomicCompSwap` from 0)
pub const PYR_BAD_CODE: u32 = 248;
pub const PYR_BAD_CODE_FIRST: u32 = 249;
/// the same for the codes pass A packs (count, first `code`, first `k`): a bad write, not a bad read
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
/// and wider): each piece carries the interpolated width. 8: at 1 km inside the coma the young
/// cone (468 s → 2300 s, width ratio 4.9) judged 0.118 with 4 pieces, 0.102 with 8 (replay lab)
pub const TIME_PIECES_MAX: u32 = 8;
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
/// The most counts one volumetric deposit makes (in adds of at most [`DUST_COUNT_MAX_PER_ADD`]):
/// the texel word saturates at 2³² anyway
pub const DUST_COUNT_TEXEL_MAX: f32 = 4.0e9;
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
  #[cfg(any(test, feature = "std"))]
  {
    std::env::var("AETHERVK_DUST_JACOBIAN_AT_START").is_ok_and(|v| v == "1")
  }
  #[cfg(not(any(test, feature = "std")))]
  {
    false
  }
}

/// [`PYR_FLAGS`] bit (host-written from `AETHERVK_DUST_DEEP_CHORD=1` or
/// `Device::dust_set_deep_chord`; the lab's `OBSERVER_LAB_DEEP_CHORD=1`): a chord the split's
/// depth limit leaves deeper than one Jacobian holds (the far poses run at depth 0, the host's
/// budget cut) is drawn by the EWA at the point of its geometric-mean depth with the perspective's
/// light factor ([`deep_chord_factors`]) instead of by the ray integral (rule 4 of
/// [`split_step`]); off by default (the user's gate). Mirror of `dust_deep_chord`.
pub const DUST_FRAME_DEEP_CHORD: u32 = 1 << 20;

/// Whether the frame of `pyramid`'s header draws deep chords by the EWA ([`DUST_FRAME_DEEP_CHORD`])
pub fn deep_chord(pyramid: &[u32]) -> bool {
  pyramid[PYR_FLAGS as usize] & DUST_FRAME_DEEP_CHORD != 0
}

#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the deep-chord bit of the frame this thread is splatting ([`set_deep_chord`], per thread as
  /// [`set_legacy_kernel`]'s)
  static DEEP_CHORD: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
#[cfg(not(any(test, feature = "std")))]
static DEEP_CHORD_NO_STD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Sets this thread's deep-chord flag ([`project_split_scatter`] from the frame's header)
pub fn set_deep_chord(on: bool) {
  #[cfg(any(test, feature = "std"))]
  DEEP_CHORD.with(|c| c.set(on));
  #[cfg(not(any(test, feature = "std")))]
  DEEP_CHORD_NO_STD.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// This thread's deep-chord flag ([`set_deep_chord`])
pub fn deep_chord_on() -> bool {
  #[cfg(any(test, feature = "std"))]
  return DEEP_CHORD.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  return DEEP_CHORD_NO_STD.load(core::sync::atomic::Ordering::Relaxed);
}

/// The EWA of a deep chord (`w0`, `w1` > 0 the clip depths of its ends, `rho` its density ends):
/// `(u_g, k)`, the Jacobian taken at `u_g = √w0 / (√w0 + √w1)` and the flux scaled by `k`.
///
/// Proof (exact for the chord's axis). Along the chord `w(u) = w0 + (w1 − w0) u` and the screen
/// point `P(u)` moves at `dP/du = K / w(u)²` (`K` constant: the numerator of the projective map's
/// derivative is `(r·seg) w(u) − (r·x(u)) (r₃·seg)`, linear terms cancelling), so
/// `P(1) − P(0) = K / (w0 w1)`; the Jacobian at `u_g`, where `w(u_g)² = w0 w1`, maps `seg` to
/// `K / w(u_g)² = P(1) − P(0)`: the screen segment is exact. The light of a grain is `∝ 1/w²`,
/// so the chord's is `∝ ∫ ρ(u) / w(u)² du = (1/(w0 w1)) ∫ ρ(u(s)) ds` with `s` the screen
/// parameter (`ds = w0 w1 du / w²`); the pixel-area factor at `u_g` is `1/w(u_g)² = 1/(w0 w1)`,
/// and the capsule draws `ρ` linear in `s` (mean `(ρ0 + ρ1)/2`), so
/// `k = (ρ0 + (ρ1 − ρ0) ū) / ((ρ0 + ρ1)/2)` with `ū = ∫ u(s) ds = w0 w1 (ln(w1/w0) + w0/w1 − 1)
/// / (w1 − w0)²` (½ for `w0 = w1`; `k = 1` for a uniform chord). The lateral width stays the one
/// at `u_g` (no single footprint holds along the chord: the light is right, its spread across
/// the axis is not). Mirror of `dust_deep_chord_factors`.
pub fn deep_chord_factors(w0: f32, w1: f32, rho: [f32; 2]) -> (f32, f32) {
  let (s0, s1) = (<f32 as FloatLike>::sqrt(w0), <f32 as FloatLike>::sqrt(w1));
  let u_g = s0 / (s0 + s1);
  let dw = w1 - w0;
  let ubar = w0 * w1 * (<f32 as FloatLike>::ln(w1 / w0) + w0 / w1 - 1.0) / (dw * dw);
  let m = 0.5 * (rho[0] + rho[1]);
  let k = if m > 0.0 && ubar.is_finite() {
    (rho[0] + (rho[1] - rho[0]) * ubar.clamp(0.0, 1.0)) / m
  } else {
    1.0
  };
  (u_g, k)
}

/// The replay lab's CPU-only knobs (`dust_observer --lab`; std builds, read from the process
/// environment, never on the GPU): `AETHERVK_DUST_LAB_NO_TIME_CELL=1` draws every sample without
/// its time cell, `AETHERVK_DUST_LAB_PIECES=<n>` overrides [`TIME_PIECES_MAX`],
/// `AETHERVK_DUST_LAB_SIGMA=<s>` scales every packet's covariance by `s²`,
/// `AETHERVK_DUST_LAB_NO_SUPPORT=1` draws every packet without its support plane.
#[cfg(any(test, feature = "std"))]
fn lab_knob(name: &str) -> Option<f32> {
  std::env::var(name).ok().and_then(|v| v.trim().parse::<f32>().ok())
}
#[cfg(not(any(test, feature = "std")))]
fn lab_knob(_name: &str) -> Option<f32> {
  None
}

/// A lab bit mask: decimal or `0x` hexadecimal (std builds, the process environment)
#[cfg(any(test, feature = "std"))]
fn lab_mask(name: &str) -> Option<u32> {
  let v = std::env::var(name).ok()?;
  let v = v.trim();
  match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
    Some(h) => u32::from_str_radix(h, 16).ok(),
    None => v.parse::<u32>().ok(),
  }
}
#[cfg(not(any(test, feature = "std")))]
fn lab_mask(_name: &str) -> Option<u32> {
  None
}

/// A cached lab flag (0 unread, 1 off, 2 on): read once per process (`split_step` is hot).
static LAB_ALL_VOLUME: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
static LAB_NO_VOLUME: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
static LAB_VOLUME_CAP_ALL: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
static LAB_VOLUME_FULL_TOP: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// The test's switch for the K1 cap (`AETHERVK_DUST_LAB_VOLUME_FULL_TOP` without the process-wide
  /// environment): [`set_volume_full_top_test`].
  static VOLUME_FULL_TOP_TEST: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
#[cfg(not(any(test, feature = "std")))]
struct VolumeFullTopNoStd;
#[cfg(not(any(test, feature = "std")))]
impl VolumeFullTopNoStd {
  fn with<R>(&self, f: impl FnOnce(&core::cell::Cell<bool>) -> R) -> R {
    f(&core::cell::Cell::new(false))
  }
}
#[cfg(not(any(test, feature = "std")))]
static VOLUME_FULL_TOP_TEST: VolumeFullTopNoStd = VolumeFullTopNoStd;
/// Tests only: the K1 cap ([`volume_full_top`]) on this thread's volume plans.
#[cfg(any(test, feature = "std"))]
pub fn set_volume_full_top_test(on: bool) {
  VOLUME_FULL_TOP_TEST.with(|c| c.set(on));
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// The lab's trace of one packet's drawing ([`set_volume_trace`])
  static VOLUME_TRACE: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
/// Lab (CPU only, diagnostics): while on, this thread's `project_split_scatter` /
/// `split_packet_with` / `volume_scatter` print how each packet is routed and drawn (its leaves,
/// the shell's hand-over, the volume plan, every piece's mean, σ, flags, kept mass, level, class,
/// texel range and deposit). The observer turns it on for the clusters of
/// `AETHERVK_DUST_LAB_TRACE_CLUSTER=<id>[,<id>…]` in its per-packet probe.
#[cfg(any(test, feature = "std"))]
pub fn set_volume_trace(on: bool) {
  VOLUME_TRACE.with(|c| c.set(on));
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// The lab's log of the texel rectangles this thread's scatters visited ([`set_touch_log`])
  static TOUCH_LOG_ON: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
  static TOUCH_LOG: core::cell::RefCell<alloc::vec::Vec<[u32; 6]>> = const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
}
/// Lab (CPU only, diagnostics): while on, this thread's [`scatter_into`] and [`volume_scatter`]
/// log every texel rectangle they visit, `[level offset, level width, x0, x1, y0, y1]`
/// (inclusive), so a caller drawing one packet at a time on a reused pyramid can sum and clear
/// only the texels the packet touched (the observer's per-packet probe). Every texel word either
/// scatter writes lies inside a logged rectangle. Turning it on clears the log.
#[cfg(any(test, feature = "std"))]
pub fn set_touch_log(on: bool) {
  TOUCH_LOG_ON.with(|c| c.set(on));
  if on {
    TOUCH_LOG.with(|l| l.borrow_mut().clear());
  }
}
/// The rectangles logged since [`set_touch_log`] was turned on (or since the last take), moved
/// into `out` (cleared first)
#[cfg(any(test, feature = "std"))]
pub fn take_touch_log(out: &mut alloc::vec::Vec<[u32; 6]>) {
  out.clear();
  TOUCH_LOG.with(|l| core::mem::swap(&mut *l.borrow_mut(), out));
}
#[cfg(any(test, feature = "std"))]
#[inline]
fn touch_log_on() -> bool {
  TOUCH_LOG_ON.with(|c| c.get())
}
#[cfg(not(any(test, feature = "std")))]
#[inline]
fn touch_log_on() -> bool {
  false
}
#[cfg(any(test, feature = "std"))]
fn touch_rect(off: u32, w: u32, x0: i64, x1: i64, y0: i64, y1: i64) {
  if x0 <= x1 && y0 <= y1 {
    TOUCH_LOG.with(|l| l.borrow_mut().push([off, w, x0.max(0) as u32, x1.max(0) as u32, y0.max(0) as u32, y1.max(0) as u32]));
  }
}
#[cfg(not(any(test, feature = "std")))]
fn touch_rect(_: u32, _: u32, _: i64, _: i64, _: i64, _: i64) {}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the pieces this thread's `volume_scatter` drew, per class ([`take_volume_class_counts`])
  static VOLUME_CLASS_COUNTS: core::cell::Cell<[u64; 4]> = const { core::cell::Cell::new([0; 4]) };
}
/// Tests: the pieces this thread's [`volume_scatter`] drew since the last call, per class
/// (centre, small, sub-rays, lines: [`DUST_VOLUME_CLASS_CENTRE`] …); resets them.
#[cfg(any(test, feature = "std"))]
pub fn take_volume_class_counts() -> [u64; 4] {
  VOLUME_CLASS_COUNTS.with(|c| c.replace([0; 4]))
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the small pieces this thread drew by one projection a slice and by the 4 sigma points a
  /// slice ([`take_volume_small_paths`])
  static VOLUME_SMALL_PATHS: core::cell::Cell<[u64; 2]> = const { core::cell::Cell::new([0; 2]) };
}
/// Tests and the lab: the small pieces drawn since the last call by one projection a slice and by
/// the 4 sigma points a slice ([`volume_small_points`]'s two paths); resets them.
#[cfg(any(test, feature = "std"))]
pub fn take_volume_small_paths() -> [u64; 2] {
  VOLUME_SMALL_PATHS.with(|c| c.replace([0; 2]))
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the small pieces this thread's [`volume_small_points`] demoted, and those at a decision's
  /// margin ([`take_volume_small_margins`])
  static VOLUME_SMALL_MARGINS: core::cell::Cell<[u64; 3]> = const { core::cell::Cell::new([0; 3]) };
}
/// Tests: since the last call, the small pieces demoted from the sigma points to one projection
/// (track 9A, counted in [`take_volume_small_paths`]' first word), those whose D3 decision sat at
/// its margin (the centre slice's 3σ within 10⁻³ σ of the frame's edge) and those whose demotion
/// did (the predictor within 10⁻³ ε of ε, or its gate within 10⁻³ of its threshold
/// [`DUST_SMALL_DEMOTE_GATE`] ε); resets them. A float boundary can move such a piece
/// between the mirrors' paths: the GPU/CPU path counts agree within the last two
/// (`splat_parity`).
#[cfg(any(test, feature = "std"))]
pub fn take_volume_small_margins() -> [u64; 3] {
  VOLUME_SMALL_MARGINS.with(|c| c.replace([0; 3]))
}
#[cfg(all(test, feature = "std"))]
std::thread_local! {
  /// Tests: the piece predictor of the last [`volume_small_points`] call (NaN: not evaluated —
  /// an interior piece, ε = 0, or a slice without a prediction; the calibration's hook)
  static SMALL_DEMOTE_LAST_E: core::cell::Cell<f32> = const { core::cell::Cell::new(f32::NAN) };
  /// Tests: the gate ([`small_demote_gate`]) of the last [`volume_small_points`] call whose
  /// predictor ran (NaN otherwise; t9a2c)
  static SMALL_DEMOTE_LAST_G: core::cell::Cell<f32> = const { core::cell::Cell::new(f32::NAN) };
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the volumetric texel deposits this thread split into several adds ([`take_volume_split_adds`])
  static VOLUME_SPLIT_ADDS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}
/// Tests and the lab: the volumetric texel deposits above [`DUST_COUNT_MAX_PER_ADD`] since the last
/// call, each split into `⌈counts / cap⌉` adds of at most the cap (a piece whose mean τ over a
/// coarse texel exceeds `2²⁴ / s²` units: 64 at 512-px texels, 4 at 2048-px ones; until step F4
/// they were clipped at the cap and their light lost); resets it. The GPU's count is the volume
/// queue header word [`DUST_VQUEUE_SPLIT_ADDS`].
#[cfg(any(test, feature = "std"))]
pub fn take_volume_split_adds() -> u64 {
  VOLUME_SPLIT_ADDS.with(|c| c.replace(0))
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// tests: draw only this piece of each volumetric packet (`u32::MAX`: all; [`set_volume_piece_only`])
  static VOLUME_PIECE_ONLY: core::cell::Cell<u32> = const { core::cell::Cell::new(u32::MAX) };
  /// the class each piece of this thread's last volumetric packet took (255: not drawn)
  static VOLUME_PIECE_CLASSES: core::cell::Cell<[u8; DUST_VOLUME_PIECES_MAX as usize]> =
    const { core::cell::Cell::new([255; DUST_VOLUME_PIECES_MAX as usize]) };
}
/// Tests (diagnostics): this thread's `volume_scatter` draws only piece `k` of each packet
/// (`None`: every piece)
#[cfg(any(test, feature = "std"))]
pub fn set_volume_piece_only(k: Option<u32>) {
  VOLUME_PIECE_ONLY.with(|c| c.set(k.unwrap_or(u32::MAX)));
}
/// Tests (diagnostics): the class ([`DUST_VOLUME_CLASS_CENTRE`] …; 255: not drawn) each piece of
/// this thread's last volumetric packet took
#[cfg(any(test, feature = "std"))]
pub fn last_volume_piece_classes() -> [u8; DUST_VOLUME_PIECES_MAX as usize] {
  VOLUME_PIECE_CLASSES.with(|c| c.get())
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// tests: plan every volumetric packet at one level for all its pieces ([`set_volume_packet_level`])
  static VOLUME_PACKET_LEVEL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// tests: the full-piece floor at DUST_VOLUME_MIN_LEVEL (the first F8 rule; [`set_volume_level_rule_test`])
  static VOLUME_LEVEL_FLOOR: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
  /// tests: the evaluation cap's mode: 0 a uniform extra level (production), 1 the share (step F9,
  /// set aside), 2 none (the hole readings' uncapped (ii))
  static VOLUME_CAP_MODE: core::cell::Cell<u8> = const { core::cell::Cell::new(0) };
}
/// Tests (diagnostics): this thread's piece-level rule — the full-piece floor on or off, the cap
/// as a uniform extra level (0, production), the share (1) or none (2)
#[cfg(any(test, feature = "std"))]
pub fn set_volume_level_rule_test(floor: bool, cap_mode: u8) {
  VOLUME_LEVEL_FLOOR.with(|c| c.set(floor));
  VOLUME_CAP_MODE.with(|c| c.set(cap_mode));
}
/// Tests (diagnostics): this thread's volume plans put every piece at the packet's one level (the
/// rule before step F8: the angular σ's level, at least [`DUST_VOLUME_MIN_LEVEL`] with a full
/// piece, plus the bias, coarsened for the cap) instead of each piece at its own
#[cfg(any(test, feature = "std"))]
pub fn set_volume_packet_level(on: bool) {
  VOLUME_PACKET_LEVEL.with(|c| c.set(on));
}
#[cfg(any(test, feature = "std"))]
fn volume_trace() -> bool {
  VOLUME_TRACE.with(|c| c.get())
}
#[cfg(not(any(test, feature = "std")))]
fn volume_trace() -> bool {
  false
}
fn lab_flag(cell: &core::sync::atomic::AtomicU8, name: &str) -> bool {
  use core::sync::atomic::Ordering::Relaxed;
  match cell.load(Relaxed) {
    0 => {
      let on = lab_knob(name).is_some_and(|v| v > 0.0);
      cell.store(if on { 2 } else { 1 }, Relaxed);
      on
    }
    v => v == 2,
  }
}

/// The lab's step-D bisection mask (`AETHERVK_DUST_LAB_K1C_D=<mask>`, CPU only, read once per
/// process; decimal or 0x-hex; unset: [`DUST_K1C_D_PRODUCTION`], the GPU's rule): bit 0 the sub-ray order caps 3 / 2 (off:
/// `min(c, 4)`), bit 1 the one-projection slices of a small piece inside the frame (off: the 4
/// sigma points everywhere), bit 2 one radiance weight a small piece (off: one per depth point), bit
/// 3 the ±4σ cover of a small piece's texels (off: its whole range).
pub const DUST_K1C_D_SUBRAY_CAP: u32 = 1;
pub const DUST_K1C_D_FAST_SLICES: u32 = 2;
pub const DUST_K1C_D_ONE_WEIGHT: u32 = 4;
pub const DUST_K1C_D_COVER: u32 = 8;
/// bit 4: the 5-point depth rule of a small piece deep along the view (step F5; off: 3 points —
/// it did not move the bounded chord's case (b) and cost 2.3e11 lane-cycles at 10 m: off in
/// production)
pub const DUST_K1C_D_GH5: u32 = 16;
/// bit 5 (lab): a chord's pieces bounded by the packet's capsule with the bounded Monte-Carlo's
/// lateral taper `erf(√((9 − r²)/2)) / P(χ²₃ ≤ 9)` at the ray's closest approach (`r` its distance
/// from the chord's axis there), instead of each piece's own 3σ ellipsoid (whose sum along a
/// sparsely pieced chord ripples by ±15 % in the 2–3 σ shell: the ring pattern of case (b))
pub const DUST_K1C_D_TAPER: u32 = 32;
/// The production rule (the GLSL's, which has no mask): the one-projection slices and the ±4σ
/// cover; not the sub-ray order caps (+14 % on the 1–10 d volume class) nor the one radiance
/// weight (−2.5 % on its split class): the bisection of 2026-10-17
pub const DUST_K1C_D_PRODUCTION: u32 = DUST_K1C_D_FAST_SLICES | DUST_K1C_D_COVER;
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// tests: this thread's step-D mask in place of the process's ([`set_k1c_d_test`])
  static K1C_D_TEST: core::cell::Cell<u32> = const { core::cell::Cell::new(u32::MAX) };
}
/// Tests: this thread's draws use the step-D mask `mask` (`None`: the process's,
/// `AETHERVK_DUST_LAB_K1C_D` or [`DUST_K1C_D_PRODUCTION`]); CPU only, the GLSL has no mask
#[cfg(any(test, feature = "std"))]
pub fn set_k1c_d_test(mask: Option<u32>) {
  K1C_D_TEST.with(|c| c.set(mask.map_or(u32::MAX, |m| m & 0x3F)));
}
static LAB_K1C_D: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);
fn k1c_d(bit: u32) -> bool {
  use core::sync::atomic::Ordering::Relaxed;
  #[cfg(any(test, feature = "std"))]
  {
    let t = K1C_D_TEST.with(|c| c.get());
    if t != u32::MAX {
      return t & bit != 0;
    }
  }
  let mut m = LAB_K1C_D.load(Relaxed);
  if m == u32::MAX {
    // decimal or 0x-hex (`lab_knob` parses an f32 only: "0xF" read as unset made every mask the
    // default in the first bisection)
    m = lab_mask("AETHERVK_DUST_LAB_K1C_D").map_or(DUST_K1C_D_PRODUCTION, |v| v & 0x3F);
    LAB_K1C_D.store(m, Relaxed);
  }
  m & bit != 0
}
/// The lab's K1c off switch (`AETHERVK_DUST_LAB_K1C_OFF=1`, CPU only; the GLSL's `-DDUST_K1C_OFF`):
/// every volumetric piece drawn by its texel-centre rays, the plan as before K1c (a fair
/// before / after on one build)
static LAB_K1C_OFF: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
fn k1c_off() -> bool {
  lab_flag(&LAB_K1C_OFF, "AETHERVK_DUST_LAB_K1C_OFF")
}

/// The lab's per-packet evaluation cap (`AETHERVK_DUST_LAB_VOLUME_MAX_EVALS=<n>`, CPU only, read
/// once per process): replaces [`DUST_VOLUME_MAX_EVALS`] in [`volume_scatter`]'s plan, so the cap
/// can be lifted to tell its coarsening from the drawing rule's errors. Unset: the constant.
static LAB_VOLUME_MAX_EVALS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
fn volume_max_evals() -> u32 {
  use core::sync::atomic::Ordering::Relaxed;
  match LAB_VOLUME_MAX_EVALS.load(Relaxed) {
    0 => {
      let v = lab_knob("AETHERVK_DUST_LAB_VOLUME_MAX_EVALS")
        .filter(|v| *v >= 1.0)
        .map_or(DUST_VOLUME_MAX_EVALS, |v| v as u32);
      LAB_VOLUME_MAX_EVALS.store(v, Relaxed);
      v
    }
    v => v,
  }
}

/// Bit 28 of [`PYR_SPLIT_DEPTH`] (host-written from `AETHERVK_DUST_TIME_CELL=1`, diagnostics):
/// a chord-major leaf's time Gaussian is centred on its time cell — half a segment from its chord
/// towards the stream predecessor — instead of on its chord (the single chord leaf and the cone
/// pieces; the sweep pieces are centred already). Mirror of `cellOff` in `dust_splat.comp`.
pub fn time_cell_offset(pyramid: &[u32]) -> bool {
  (pyramid[PYR_SPLIT_DEPTH as usize] >> 28) & 1 != 0
}

/// Bit 29 of [`PYR_SPLIT_DEPTH`] (host-written: `AETHERVK_DUST_LEGACY_KERNEL=1` or
/// `Device::dust_set_legacy_kernel`): the frame draws the kernel before the cone of grains — no
/// [`conic_slabs`], no slow-end sub-bins ([`bin_subbins`]) — for measuring what they cost on the
/// same build. Mirror of `dust_legacy_kernel`.
pub const DUST_FRAME_LEGACY_KERNEL_BIT: u32 = 29;

/// Bits 30–31 of [`PYR_SPLIT_DEPTH`]: a GPU timing probe (host-written from
/// `AETHERVK_DUST_TIMING_PROBE`; diagnostics, never set in production, GPU only: the CPU mirror
/// ignores them), the image wrong by design. 1 = `plain_add`: `dust_atomic_add_sat` is a plain
/// `atomicAdd` in the splat (what the compare-and-swap loop costs). 2 = `no_volume_writes`: the
/// volumetric texels are evaluated but not written (the math against the writes). 3 =
/// `no_volume`: volumetric packets are dropped (what the screen part costs alone).
pub const DUST_FRAME_PROBE_SHIFT: u32 = 30;

/// The probe bits of `AETHERVK_DUST_TIMING_PROBE` (`plain_add`, `no_volume_writes`, `no_volume`;
/// [`DUST_FRAME_PROBE_SHIFT`]); 0 when unset or unknown
pub fn timing_probe_bits(v: &str) -> u32 {
  let probe = match v.trim() {
    "plain_add" => 1,
    "no_volume_writes" => 2,
    "no_volume" => 3,
    _ => 0,
  };
  probe << DUST_FRAME_PROBE_SHIFT
}

/// Whether the frame of `pyramid`'s header draws the legacy kernel ([`DUST_FRAME_LEGACY_KERNEL_BIT`])
pub fn legacy_kernel(pyramid: &[u32]) -> bool {
  (pyramid[PYR_SPLIT_DEPTH as usize] >> DUST_FRAME_LEGACY_KERNEL_BIT) & 1 != 0
}

#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// the legacy-kernel bit of the frame this thread is splatting ([`set_legacy_kernel`]): per
  /// thread, so a test's legacy frame cannot leak into another test's packets
  static LEGACY_KERNEL: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
#[cfg(not(any(test, feature = "std")))]
static LEGACY_KERNEL_NO_STD: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Sets this thread's legacy-kernel flag (the splat, per cluster, from the frame's header; replays)
pub fn set_legacy_kernel(on: bool) {
  #[cfg(any(test, feature = "std"))]
  LEGACY_KERNEL.with(|c| c.set(on));
  #[cfg(not(any(test, feature = "std")))]
  LEGACY_KERNEL_NO_STD.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// This thread's legacy-kernel flag ([`set_legacy_kernel`])
pub fn legacy_kernel_on() -> bool {
  #[cfg(any(test, feature = "std"))]
  return LEGACY_KERNEL.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  return LEGACY_KERNEL_NO_STD.load(core::sync::atomic::Ordering::Relaxed);
}

/// the frame's [`time_cell_offset`] for [`subpackets_swept`] (set per cluster by the splat from the
/// header; a diagnostic, so a process-wide flag rather than a parameter through every caller)
static TIME_CELL_OFFSET: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Bits 24–27 of [`PYR_SPLIT_DEPTH`] (host-written from `AETHERVK_DUST_HARD_AXIS=<n>`,
/// diagnostics): a leaf's Gaussianised axis ([`Packet::seg_t`], `1.21·(L/2)²`) is drawn as `n`
/// sub-leaves spread uniformly over `[−L/2, L/2]` about the leaf, each with `1.21·(L/2n)²` — a
/// hard-edged cell instead of a Gaussian's tails; 0 or 1: off. Mirror of `dust_hard_axis_pieces`.
pub fn hard_axis_pieces(pyramid: &[u32]) -> u32 {
  (pyramid[PYR_SPLIT_DEPTH as usize] >> 24) & 0xF
}

/// `p` as the `n` sub-leaves of [`hard_axis_pieces`] (the leaf itself when `n ≤ 1` or it has no
/// Gaussianised axis). The covariance's `1.21·hhᵀ` term (`h = seg_t / 2`) becomes `1.21·(h/n)(h/n)ᵀ`,
/// the means `μ + (−1 + (2j + 1)/n)·h`, the flux split evenly. Mirror of `dust_hard_axis_leaf`.
pub fn hard_axis_leaf(p: &Packet, n: u32, j: u32) -> Packet {
  let h = scale3(p.seg_t, 0.5);
  if n <= 1 || !(dot3(h, h) > 0.0) {
    return *p;
  }
  let k = CHORD_VAR_FACTOR;
  let q = 1.0 - 1.0 / (n * n) as f32;
  let c = &p.cov;
  let off = -1.0 + (2 * j + 1) as f32 / n as f32;
  Packet {
    mean: add3(p.mean, scale3(h, off)),
    cov: [
      c[0] - k * q * h[0] * h[0],
      c[1] - k * q * h[0] * h[1],
      c[2] - k * q * h[0] * h[2],
      c[3] - k * q * h[1] * h[1],
      c[4] - k * q * h[1] * h[2],
      c[5] - k * q * h[2] * h[2],
    ],
    seg: p.seg,
    flux: p.flux / n as f32,
    open: p.open,
    rho: p.rho,
    seg_t: scale3(p.seg_t, 1.0 / n as f32),
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
  /// The speck pass's walk mask (bit `b`: size bin `b` is walked for specks this frame, so the
  /// continuum gives up its shell mass; `edges[15].w` as bits, 0 from the propagate): written by
  /// [`super::specks::specks_tier`] before the splat, read by [`splat_cluster_inner`]. Mirror of
  /// `dust_specks_mask`.
  pub fn specks_mask(&self) -> u32 {
    self.edges[15][3].to_bits()
  }
  pub fn specks_walked(&self, b: usize) -> bool {
    (self.specks_mask() >> b) & 1 != 0
  }
  /// the mean's heliocentric velocity (m/s, particle-system axes; `edges[9..11].w`)
  pub fn helio_velocity(&self) -> [f32; 3] {
    [self.edges[9][3], self.edges[10][3], self.edges[11][3]]
  }
  /// the propagate generation that wrote these moments (`edges[12].w` bits; 0 on the CPU)
  pub fn propagate_stamp(&self) -> u32 {
    self.edges[12][3].to_bits()
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
    Self::new_capped(width, height, PYRAMID_MAX_LEVELS)
  }

  /// [`Self::new`] with at most `max_levels` levels (clamped to `1..=PYRAMID_MAX_LEVELS`): the
  /// coarsest level is wider than 1 × 1 when the size needs more. The cadence's class images keep
  /// their frame's level count this way (`CadenceConfig::image_geom`).
  pub fn new_capped(width: u32, height: u32, max_levels: u32) -> Self {
    let cap = max_levels.clamp(1, PYRAMID_MAX_LEVELS);
    let (width, height) = (width.max(1), height.max(1));
    let mut levels = alloc::vec::Vec::new();
    let mut off = PYRAMID_HEADER_WORDS;
    let (mut w, mut h) = (width, height);
    loop {
      levels.push((off, w, h));
      off += w * h * PYRAMID_TEXEL_WORDS;
      if (w == 1 && h == 1) || levels.len() as u32 >= cap {
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
  packet_moments_in(c, slot, super::RENDER_SLOT_BITS, frame)
}

/// [`packet_moments`] for ring slot `slot` of a ring of `2^log2cap` slots (the render word's mask
/// bits, [`super::render_word`]).
pub fn packet_moments_in(
  c: &DustCluster,
  slot: u32,
  log2cap: u32,
  frame: &DustFrame,
) -> (DustRenderCluster, DustMoments) {
  packet_moments_detail(c, slot, log2cap, frame, 1)
}

/// [`packet_moments_in`] at a size-polyline edge `stride` (1, 2 or 4: the trajectory detail,
/// [`super::propagate_edge_stride`]): every `stride`-th edge is solved, the others are placed
/// linearly between their solved neighbours (`e0 + (e1 − e0)·t`, the GPU's expression); the
/// speed factors are exact for every edge (linear in `j` anyway). `1 + 3 + SIZE_BINS / stride + 1`
/// Kepler solves. Mirror of `dust_propagate.comp`.
pub fn packet_moments_detail(
  c: &DustCluster,
  slot: u32,
  log2cap: u32,
  frame: &DustFrame,
  stride: u32,
) -> (DustRenderCluster, DustMoments) {
  let stride = if stride == 2 || stride == 4 { stride } else { 1 };
  let rc = super::evaluate_cluster_in(c, slot, log2cap, frame);
  if !(rc.age_id_dbeta_flux[3] > 0.0) {
    return (rc, DustMoments::culled());
  }
  let t_now = Df::new(frame.ps_r_t_hi[3], frame.ps_r_t_lo[3]);
  let age = t_now.sub(c.t0());
  let age_f = age.hi + age.lo;
  let beta = c.beta();
  let mu = consts::SUN_MU.mul(two_sum_one_minus(beta));
  let (r0, v0) = (c.r0(), c.v0());
  let (r, v) = kepler::propagate(&r0, &v0, mu, age);
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
    let factor = if sqrt_beta > 0.0 { sb / sqrt_beta } else { 1.0 };
    let local = if j as u32 % stride == 0 {
      let beta_j = sb * sb;
      let v_j = v0.add(&Df3::from_f32(scale3(eject, factor - 1.0)));
      let mu_j = consts::SUN_MU.mul(two_sum_one_minus(beta_j));
      let (r_j, _) = kepler::propagate(&r0, &v_j, mu_j, age);
      qrot(rot, r_j.sub(&ps).to_f32())
    } else {
      [0.0; 3]
    };
    *e = [local[0], local[1], local[2], factor];
  }
  // the skipped edges (stride > 1): linear between the solved neighbours, the GPU's expression
  for j in 0..SIZE_EDGES as usize {
    let r = j as u32 % stride;
    if r == 0 {
      continue;
    }
    let (j0, j1) = (j - r as usize, j - r as usize + stride as usize);
    let t = r as f32 / stride as f32;
    for i in 0..3 {
      edges[j][i] = edges[j0][i] + (edges[j1][i] - edges[j0][i]) * t;
    }
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
  // the mean's heliocentric velocity (m/s, particle-system axes) and the propagate's generation
  // stamp (0 on the CPU: the host mirror never skips; `dust_propagate.comp`)
  let vel = qrot(rot, v.to_f32());
  edges[9][3] = vel[0];
  edges[10][3] = vel[1];
  edges[11][3] = vel[2];
  edges[12][3] = 0.0;
  // the speck pass's walk mask lane ([`DustMoments::specks_mask`]): nothing walked until it runs
  edges[15][3] = 0.0;
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

// ─── Chord groups (8 ms plan: volumetric clusters drawn with fewer, longer packets) ──────────
// Inside the size-bin loop of [`subpackets_cells`] a "bin" can span `g` consecutive size bins
// ([`group_bins`]): its edges are `b` and `b + g`, its speed factors theirs, its share `g` bins'
// flux, its density ramp linear between its end factors ([`bin_density_ends`]). `g` is chosen per
// cluster ([`cluster_group`]) and is 1 everywhere else.
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static GROUP_ACTIVE: core::cell::Cell<usize> = const { core::cell::Cell::new(1) };
  static GROUP_REQUESTED: core::cell::Cell<usize> = const { core::cell::Cell::new(1) };
}
#[cfg(not(any(test, feature = "std")))]
static GROUP_ACTIVE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);
#[cfg(not(any(test, feature = "std")))]
static GROUP_REQUESTED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(1);

/// The size bins one bin of the loop spans now (1 outside a grouped cluster's bin loop)
pub fn group_bins() -> usize {
  #[cfg(any(test, feature = "std"))]
  return GROUP_ACTIVE.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  return GROUP_ACTIVE.load(core::sync::atomic::Ordering::Relaxed);
}
fn set_group_active(g: usize) {
  #[cfg(any(test, feature = "std"))]
  GROUP_ACTIVE.with(|c| c.set(g));
  #[cfg(not(any(test, feature = "std")))]
  GROUP_ACTIVE.store(g, core::sync::atomic::Ordering::Relaxed);
}
/// Requests chord groups of `g` bins for the next [`subpackets_cells`] (the cluster's draw;
/// [`cluster_group`]); 1 = single bins
pub fn request_group(g: usize) {
  let g = if g >= 1 && SIZE_BINS as usize % g == 0 { g } else { 1 };
  #[cfg(any(test, feature = "std"))]
  GROUP_REQUESTED.with(|c| c.set(g));
  #[cfg(not(any(test, feature = "std")))]
  GROUP_REQUESTED.store(g, core::sync::atomic::Ordering::Relaxed);
}
fn requested_group() -> usize {
  #[cfg(any(test, feature = "std"))]
  return GROUP_REQUESTED.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  return GROUP_REQUESTED.load(core::sync::atomic::Ordering::Relaxed);
}

// ─── The cluster level of detail (mirror of dust_lod_*: `PYR_LOD`) ──────────────────────────
// `lod_px` pixels per drawn segment. A polyline whose bins are shorter than that on screen is
// drawn as chord groups of 2, 4, 8 or 16 bins (the smallest power of two whose group chord is
// at least `lod_px`; 16 = one chord along the whole polyline, which keeps the shape the pooled
// packet of `polyline_merged` loses); the time pieces are capped at `⌈T / lod_px⌉` with `T` the
// time segment on screen; the sweeps step by `lod_px` instead of a pixel. Points behind the camera
// keep full detail (as `polyline_merged`). Groups form only where the group's chord is a few
// pixels long, where its Gaussianised axis's tails lie within the pixel filter; the near field,
// whose light is shape-critical, keeps every bin (the chord-group study of 2026-10-14 grouped by
// a volumetric rule, i.e. near the eye, and doubled the light).
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static LOD_PX: core::cell::Cell<f32> = const { core::cell::Cell::new(0.0) };
}
#[cfg(not(any(test, feature = "std")))]
static LOD_PX_BITS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
/// the frame's pixels per drawn segment (0 = off) of the header word [`PYR_LOD`]
pub fn lod_px(pyramid: &[u32]) -> f32 {
  let v = f32::from_bits(pyramid[PYR_LOD as usize]);
  if v.is_finite() && v > 0.0 { v } else { 0.0 }
}
/// the level of detail of the cluster being drawn (set by [`splat_cluster_inner`]; 0 = off)
pub fn lod_px_active() -> f32 {
  #[cfg(any(test, feature = "std"))]
  return LOD_PX.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  return f32::from_bits(LOD_PX_BITS.load(core::sync::atomic::Ordering::Relaxed));
}
pub fn set_lod_px_active(v: f32) {
  #[cfg(any(test, feature = "std"))]
  LOD_PX.with(|c| c.set(v));
  #[cfg(not(any(test, feature = "std")))]
  LOD_PX_BITS.store(v.to_bits(), core::sync::atomic::Ordering::Relaxed);
}
// The level of detail's depth rule: a group chord or a time piece may not span more depth than
// the split tolerates for one Jacobian ([`DUST_SPLIT_EPS`] × the depth of the polyline's nearest
// edge, `split_step`'s own criterion), so the level of detail never makes a packet the split
// would have to bisect. Without it a foreshortened polyline (pointing at the eye: a few pixels on
// screen, a depth span comparable to its depth) became one chord drawn with one Jacobian, and
// at the frames' split depth 0 the 50 000 km pose lost 40 % of its light at 2 px (2026-10-16).
// The eye and forward of the frame being drawn (a perspective view; `None`: orthographic or
// unknown, no depth rule), set by [`splat_cluster_inner`].
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static LOD_VIEW: core::cell::Cell<Option<([f32; 3], [f32; 3])>> = const { core::cell::Cell::new(None) };
}
#[cfg(not(any(test, feature = "std")))]
static LOD_VIEW_BITS: [core::sync::atomic::AtomicU32; 7] = [const { core::sync::atomic::AtomicU32::new(0) }; 7];
/// the eye and forward of the frame being drawn, for the level of detail's depth rule
pub fn lod_view_active() -> Option<([f32; 3], [f32; 3])> {
  #[cfg(any(test, feature = "std"))]
  return LOD_VIEW.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  {
    use core::sync::atomic::Ordering::Relaxed;
    if LOD_VIEW_BITS[6].load(Relaxed) == 0 {
      return None;
    }
    let f = |i: usize| f32::from_bits(LOD_VIEW_BITS[i].load(Relaxed));
    return Some(([f(0), f(1), f(2)], [f(3), f(4), f(5)]));
  }
}
pub fn set_lod_view_active(v: Option<([f32; 3], [f32; 3])>) {
  #[cfg(any(test, feature = "std"))]
  LOD_VIEW.with(|c| c.set(v));
  #[cfg(not(any(test, feature = "std")))]
  {
    use core::sync::atomic::Ordering::Relaxed;
    match v {
      Some((e, f)) => {
        for i in 0..3 {
          LOD_VIEW_BITS[i].store(e[i].to_bits(), Relaxed);
          LOD_VIEW_BITS[3 + i].store(f[i].to_bits(), Relaxed);
        }
        LOD_VIEW_BITS[6].store(1, Relaxed);
      }
      None => LOD_VIEW_BITS[6].store(0, Relaxed),
    }
  }
}
/// The polyline's depth: `(the depth span per bin, the depth of its nearest edge)` over its 17
/// edges along `fwd` from `eye` (mirror of `dust_polyline_depth`); `None` when an edge is at or
/// behind the eye (full detail, as behind the camera)
pub fn polyline_depth(m: &DustMoments, eye: [f32; 3], fwd: [f32; 3]) -> Option<(f32, f32)> {
  let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
  for j in 0..SIZE_EDGES as usize {
    let d = dot3(sub3(m.edge(j), eye), fwd);
    lo = lo.min(d);
    hi = hi.max(d);
  }
  (lo > 0.0 && hi.is_finite()).then_some(((hi - lo) / SIZE_BINS as f32, lo))
}
/// The largest chord group (a power of two ≤ 16) whose chord spans no more depth than the split
/// tolerates, `g · bin_depth ≤ DUST_SPLIT_EPS · depth_near` (mirror of `dust_lod_group_depth_cap`);
/// 16 without a depth (`None`: orthographic), 1 when even a bin is too deep
pub fn lod_group_depth_cap(depth: Option<(f32, f32)>) -> usize {
  let Some((bin_depth, near)) = depth else { return SIZE_BINS as usize };
  let lim = DUST_SPLIT_EPS * near;
  let mut g = 1usize;
  while g < SIZE_BINS as usize && (2 * g) as f32 * bin_depth <= lim {
    g *= 2;
  }
  g
}
/// The fewest time pieces whose depth span the split tolerates, `⌈span / (DUST_SPLIT_EPS ·
/// depth_near)⌉` for a time segment of depth span `span` (mirror of `dust_time_pieces_depth`);
/// 1 without a depth
pub fn time_pieces_depth(span: Option<f32>, depth: Option<(f32, f32)>) -> u32 {
  match (span, depth) {
    (Some(s), Some((_, near))) if near > 0.0 => (ce(fabs(s) / (DUST_SPLIT_EPS * near)) as u32).clamp(1, TIME_PIECES_MAX),
    _ => 1,
  }
}
/// The polyline's on-screen half-extent: the largest distance of its 17 edges from its mean, in
/// pixels (mirror of `dust_polyline_extent_px`); `None` when a point is behind the camera
pub fn polyline_extent_px(m: &DustMoments, px: &dyn Fn([f32; 3]) -> Option<[f32; 2]>) -> Option<f32> {
  let c = px(m.mean())?;
  let mut ext = 0.0f32;
  for j in 0..SIZE_EDGES as usize {
    let p = px(m.edge(j))?;
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    ext = ext.max(<f32 as FloatLike>::sqrt(dx * dx + dy * dy));
  }
  ext.is_finite().then_some(ext)
}
/// The time segment on screen: the distance between the means of `m` and its predecessor, in
/// pixels (mirror of `dust_time_segment_px`); `None` without a predecessor or behind the camera
pub fn time_segment_px(
  m: &DustMoments,
  pred: Option<&DustMoments>,
  px: &dyn Fn([f32; 3]) -> Option<[f32; 2]>,
) -> Option<f32> {
  let p = pred?;
  let (a, c) = (px(m.mean())?, px(p.mean())?);
  let (dx, dy) = (c[0] - a[0], c[1] - a[1]);
  let t = <f32 as FloatLike>::sqrt(dx * dx + dy * dy);
  t.is_finite().then_some(t)
}
/// The chord group of a polyline of half-extent `ext_px` on screen (mirror of `dust_lod_group`):
/// 1 when off, unknown (`None`: behind the camera) or when a bin is already `lod_px` long; else
/// the smallest of 2, 4, 8, 16 whose group chord `g · 2 ext / 16` reaches `lod_px` (16 when none),
/// and never more than `depth_cap` ([`lod_group_depth_cap`]: the depth the split tolerates)
pub fn lod_group(ext_px: Option<f32>, lod_px: f32, depth_cap: usize) -> usize {
  let Some(ext) = ext_px else { return 1 };
  if !(lod_px > 0.0) {
    return 1;
  }
  let bin_px = 2.0 * ext / SIZE_BINS as f32;
  if bin_px >= lod_px {
    return 1;
  }
  let mut g = 2usize;
  while g < SIZE_BINS as usize && (g as f32) * bin_px < lod_px {
    g *= 2;
  }
  let g = g.min(depth_cap.max(1));
  // lab (`AETHERVK_DUST_LAB_LOD_GMAX=<g>`, CPU only): the largest group, for the far-pose study
  match lab_knob("AETHERVK_DUST_LAB_LOD_GMAX") {
    Some(v) if v >= 1.0 => g.min((v as usize).max(1)),
    _ => g,
  }
}
/// The time pieces after the level of detail (mirror of `dust_time_pieces_lod`): `n_t` when off
/// or unknown, else `clamp(⌈t_px / lod_px⌉, 1, n_t)`, and at least `depth_min` of them
/// ([`time_pieces_depth`]: the depth span the split tolerates per piece)
pub fn time_pieces_lod(n_t: u32, t_px: Option<f32>, lod_px: f32, depth_min: u32) -> u32 {
  // lab (`AETHERVK_DUST_LAB_LOD_NO_PIECES=1`, CPU only): the pieces uncapped, for the study
  if lab_knob("AETHERVK_DUST_LAB_LOD_NO_PIECES").is_some_and(|v| v > 0.0) {
    return n_t;
  }
  match t_px {
    Some(t) if lod_px > 0.0 => (ce(t / lod_px) as u32).max(depth_min).clamp(1, n_t.max(1)),
    _ => n_t,
  }
}

/// The chord group of cluster `m` seen from `view` (mirror of `dust_cluster_group`): `g` when the
/// view is perspective and both end bins of its polyline — the narrowest, slow end and the fast
/// end — would be drawn volumetrically from their midpoints ([`is_volumetric`]): the eye is near
/// or inside every bin, the light is the smooth ray integral, and fewer, longer packets carry it.
/// 1 otherwise.
pub fn cluster_group(m: &DustMoments, view: &SplitView, g: usize) -> usize {
  if g <= 1 || !view.volumetric {
    return 1;
  }
  for b in [0usize, SIZE_BINS as usize - 1] {
    let mut p = bin_packet(m, b, [0.0; 3]);
    p.seg = [0.0; 3];
    if !is_volumetric(&p, view) {
      return 1;
    }
  }
  g
}

/// The relative density at the two ends of a size bin's chord whose edges have the speed
/// factors `f0 ≤ f1` (`√(β_j/β)`): `∝ 1/f` (position ∝ β, cross-section uniform in √β), so
/// `2 f1 / (f0 + f1)` and `2 f0 / (f0 + f1)`, mean 1 over the chord: every consumer (the
/// capsule's ramp, the volume samples' flux, the sub-chords' renormalisation) multiplies the
/// packet's flux by the ramp. (Until 2026-10-16 the ends were the reciprocals,
/// `(f0 + f1) / (2 f0)` and `(f0 + f1) / (2 f1)`: the same shape and the same first order in
/// the factor ratio, so a single bin drew 0.5 % high, but a mean of `(f0 + f1)² / (4 f0 f1)`:
/// a chord group over a 10× factor range drew 3× its flux, which is where the "chord groups
/// double the light" of 2026-10-14 and the level of detail's 1.7× came from.) Mirror of
/// `dust_bin_density_ends`.
pub fn bin_density_ends(f0: f32, f1: f32) -> [f32; 2] {
  let (f0, f1) = (f0.max(1e-6), f1.max(1e-6));
  let s = f0 + f1;
  [2.0 * f1 / s, 2.0 * f0 / s]
}

/// The packets of a cluster (mirror of `dust_subpackets`): one per size bin of the polyline,
/// uniform along its chord (a Gaussian of [`CHORD_VAR_FACTOR`] half-chords² along it) with
/// `Σ_v` scaled by the bin's speed factor, `1/SIZE_BINS` of the flux each; and when the stream
/// predecessor `pred` is wider (older), every bin is cut into `n_t ≤ TIME_PIECES_MAX` pieces along
/// the segment ([`piece_bound`]), each with the width interpolated between the cluster's and the
/// predecessor's (the capsule is a cone, not a stick of the newest width).
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
  bin_packet_sub(m, b, 0, 1, seg)
}

/// Sub-bin `j` of `nj` of size bin `b`: the bin's straight chord and its speed-factor range cut
/// into `nj` equal parts ([`sub_edges`]). A grain's spread scales with **its own** speed factor
/// (`f²Σ_v`), so one width at the bin's mid factor, plus the chord's Gaussian of
/// [`CHORD_VAR_FACTOR`] half-chords², reaches past the slowest grain at the polyline's slow end;
/// sub-bins carry their own factor and a chord `nj` times shorter.
pub fn sub_edges(m: &DustMoments, b: usize, j: usize, nj: usize) -> ([f32; 3], [f32; 3], f32, f32) {
  let g = group_bins();
  let (e0, e1) = (m.edge(b), m.edge(b + g));
  let (f0, f1) = (m.edge_factor(b), m.edge_factor(b + g));
  // the bin's own edges exactly at its ends (one sub-bin is the bin, bit for bit)
  let at = |k: usize| -> ([f32; 3], f32) {
    if k == 0 {
      (e0, f0)
    } else if k == nj {
      (e1, f1)
    } else {
      let u = k as f32 / nj as f32;
      (add3(e0, scale3(sub3(e1, e0), u)), f0 + (f1 - f0) * u)
    }
  };
  let ((a, fa), (c, fc)) = (at(j), at(j + 1));
  (a, c, fa, fc)
}

/// The share of size bin `b`'s flux in its sub-bin `j` of `nj`: the bin's density ramp
/// ([`bin_density_ends`], ∝ 1/f) integrated over the sub-bin, over the whole bin (exactly 1 for
/// one sub-bin)
pub fn sub_share(m: &DustMoments, b: usize, j: usize, nj: usize) -> f32 {
  if nj <= 1 {
    return 1.0;
  }
  let [r0, r1] = bin_density_ends(m.edge_factor(b), m.edge_factor(b + group_bins()));
  let (u0, u1) = (j as f32 / nj as f32, (j + 1) as f32 / nj as f32);
  (u1 - u0) * (r0 + (r1 - r0) * 0.5 * (u0 + u1)) / (0.5 * (r0 + r1))
}

/// [`bin_packet`] of sub-bin `j` of `nj` ([`sub_edges`]), its share of the bin's flux
/// ([`sub_share`])
pub fn bin_packet_sub(m: &DustMoments, b: usize, j: usize, nj: usize, seg: [f32; 3]) -> Packet {
  let cv = m.cov_v();
  let (a, c, fa, fc) = sub_edges(m, b, j, nj);
  let mid = scale3(add3(a, c), 0.5);
  let h = scale3(sub3(c, a), 0.5);
  let f = 0.5 * (fa + fc);
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
    flux: m.flux() * group_bins() as f32 / SIZE_BINS as f32 * sub_share(m, b, j, nj),
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
  bin_chord_packet_sub(m, b, 0, 1, seg_t)
}

/// [`bin_chord_packet`] of sub-bin `j` of `nj` ([`sub_edges`])
pub fn bin_chord_packet_sub(m: &DustMoments, b: usize, j: usize, nj: usize, seg_t: [f32; 3]) -> Packet {
  let cv = m.cov_v();
  let (a, c, fa, fc) = sub_edges(m, b, j, nj);
  let f = 0.5 * (fa + fc);
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
    flux: m.flux() * group_bins() as f32 / SIZE_BINS as f32 * sub_share(m, b, j, nj),
    open: [b > 0 || j > 0, b + group_bins() < SIZE_BINS as usize || j + 1 < nj],
    rho: bin_density_ends(fa, fc),
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

/// `R Σ Rᵀ` for the rotation by `angle` about the unit `axis` (`Σ` as xx, xy, xz, yy, yz, zz;
/// mirror of `dust_rotate_cov`)
pub fn rotate_cov(c: [f32; 6], axis: [f32; 3], angle: f32) -> [f32; 6] {
  let r = [
    rotate3([1.0, 0.0, 0.0], axis, angle),
    rotate3([0.0, 1.0, 0.0], axis, angle),
    rotate3([0.0, 0.0, 1.0], axis, angle),
  ];
  // columns of R are r[j]; (R Σ Rᵀ)_ab = Σ_ij R_ai Σ_ij R_bj
  let m = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
  let at = |a: usize, b: usize| {
    let mut acc = 0.0f32;
    for i in 0..3 {
      for j in 0..3 {
        acc += r[i][a] * m[i][j] * r[j][b];
      }
    }
    acc
  };
  [at(0, 0), at(0, 1), at(0, 2), at(1, 1), at(1, 2), at(2, 2)]
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
  for e in [b, b + group_bins()] {
    if let (Some(a), Some(c)) = (px(m.edge(e)), px(p.edge(e))) {
      let d = [c[0] - a[0], c[1] - a[1]];
      gap = gap.max(<f32 as FloatLike>::sqrt(d[0] * d[0] + d[1] * d[1]));
    }
  }
  if !gap.is_finite() {
    return 1;
  }
  // the level of detail's step ([`lod_px_active`]), at least a pixel
  let step = lod_px_active().max(TIME_SWEEP_STEP_PX);
  (ce(gap / step) as u32).clamp(1, TIME_SWEEP_MAX)
}

/// The `k`-th of `n_s` sweep sub-capsules of bin `b` (mirror of `dust_bin_sweep`): the chord
/// interpolated at `u = (k + ½)/n_s` between the bin's own edges and the predecessor's, `Σ_v`
/// scaled by the bin's speed factor plus a Gaussian of [`CHORD_VAR_FACTOR`] × (the larger edge
/// displacement / 2 n_s)² along it (the residual between sub-capsules), `1/(SIZE_BINS · n_s)` of
/// the flux, open towards the neighbouring bins.
pub fn bin_sweep_packet(m: &DustMoments, p: &DustMoments, b: usize, k: u32, n_s: u32) -> Packet {
  bin_sweep_packet_sub(m, p, b, 0, 1, k, n_s)
}

/// [`bin_sweep_packet`] of sub-bin `j` of `nj` ([`sub_edges`], the predecessor's alike)
pub fn bin_sweep_packet_sub(m: &DustMoments, p: &DustMoments, b: usize, j: usize, nj: usize, k: u32, n_s: u32) -> Packet {
  let cv = m.cov_v();
  let (a, c, fa, fc) = sub_edges(m, b, j, nj);
  let (pa, pc, _, _) = sub_edges(p, b, j, nj);
  let (da, dc) = (sub3(pa, a), sub3(pc, c));
  let dmax = if dot3(da, da) > dot3(dc, dc) { da } else { dc };
  let u = (k as f32 + 0.5) / n_s as f32;
  let (ak, ck) = (add3(a, scale3(da, u)), add3(c, scale3(dc, u)));
  let f = 0.5 * (fa + fc);
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
    flux: m.flux() * group_bins() as f32 / (SIZE_BINS * n_s) as f32 * sub_share(m, b, j, nj),
    open: [b > 0 || j > 0, b + group_bins() < SIZE_BINS as usize || j + 1 < nj],
    rho: bin_density_ends(fa, fc),
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
pub fn subpackets_supported(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> alloc::vec::Vec<(Packet, [f32; 4])> {
  subpackets_cells(m, seg, pred, merged, pred_merged, px).into_iter().map(|(p, s, _)| (p, s)).collect()
}

/// [`subpackets_supported`] with each packet's cell origin as its plane offset along the support's
/// normal, `w_o = n·o` (the apex of its cone of grains; `NaN` without a cell line)
pub fn subpackets_cells(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> alloc::vec::Vec<(Packet, [f32; 4], f32)> {
  subpackets_cells_bins(m, seg, pred, merged, pred_merged, px).into_iter().map(|(p, s, w, _)| (p, s, w)).collect()
}

/// [`subpackets_cells`] with each packet's size bin (`0xFF`: the pooled polyline, every bin): the
/// speck shell's hand-over is decided per bin ([`DustMoments::specks_walked`])
pub fn subpackets_cells_bins(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> alloc::vec::Vec<(Packet, [f32; 4], f32, u8)> {
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
  // lab (`AETHERVK_DUST_LAB_OLD_PIECES=1`, CPU only): the rule before 2026-10-12 — 4 equal
  // pieces, the predecessor's covariance unturned — for A/B replays of captured frames
  let old_rule = lab_knob("AETHERVK_DUST_LAB_OLD_PIECES").is_some_and(|v| v > 0.0);
  let n_t = if spiral.is_some() {
    lab_knob("AETHERVK_DUST_LAB_PIECES").map_or(if old_rule { 4 } else { TIME_PIECES_MAX }, |v| (v as u32).max(1))
  } else if old_rule {
    time_pieces(m, pred).min(4)
  } else {
    time_pieces(m, pred)
  };
  // the level of detail: no more pieces than the time segment has `lod_px` on screen, no fewer
  // than its depth span needs ([`time_pieces_depth`])
  let lod_depth = lod_view_active().and_then(|(eye, fwd)| polyline_depth(m, eye, fwd));
  let depth_min = time_pieces_depth(
    pred.and_then(|p| lod_view_active().map(|(_, fwd)| dot3(sub3(p.mean(), m.mean()), fwd))),
    lod_depth,
  );
  let n_t = time_pieces_lod(n_t, px.and_then(|px| time_segment_px(m, pred, px)), lod_px_active(), depth_min);
  // the predecessor's covariance (and its chord axis) turned with the spiral to the emission
  // direction of the piece's centre: by ω·(age_p − age(u_c)), the turn `spiral_point` gives the
  // piece's mean. Unturned, the radial σ (4× the lateral at 1 km) lay along the predecessor's
  // ejection direction, 0.26 rad off, and doubled the cone's lateral spread (judged 0.216 → 0.145)
  let turn_angle = |uc: f32| -> Option<([f32; 3], f32)> {
    spiral.filter(|_| !old_rule).map(|(axis, omega)| (axis, omega * (age_p - (age_o + uc * (age_p - age_o)))))
  };
  let turn = |c: [f32; 6], uc: f32| -> [f32; 6] {
    match turn_angle(uc) {
      Some((axis, a)) => rotate_cov(c, axis, a),
      None => c,
    }
  };
  let turn_vec = |v: [f32; 3], uc: f32| -> [f32; 3] {
    match turn_angle(uc) {
      Some((axis, a)) => rotate3(v, axis, a),
      None => v,
    }
  };
  let ratio = width_ratio(m, pred);
  let r0 = 1.0 / ratio; // σ_self / σ_pred ≤ 1
  let bound = |k: u32| piece_bound(k, n_t, if old_rule { 1.0 } else { ratio });
  // the support planes ([`cell_support`]): the sample's own for its bins, the cone piece's from
  // the cell line at the piece's centre age — the two samples' lines interpolated along the
  // spiral, the predecessor's direction turned to the piece's emission like its covariance
  let line_m = cell_line(m);
  let line_p = pred.and_then(cell_line);
  let v_std = line_m.map_or(0.0, |(_, c)| {
    let cn2 = dot3(c, c);
    if cn2 > 0.0 {
      <f32 as FloatLike>::sqrt(var_along(&m.cov_v(), scale3(c, 1.0 / <f32 as FloatLike>::sqrt(cn2))).max(0.0) / cn2)
    } else {
      0.0
    }
  });
  let f_lo = m.edge_factor(0);
  // the support plane and the origin's offset along its normal (`NaN` without a line)
  let with_origin = |o: [f32; 3], c: [f32; 3]| -> ([f32; 4], f32) {
    let sp = support_plane(o, c, v_std, f_lo);
    (sp, if has_support(&sp) { dot3([sp[0], sp[1], sp[2]], o) } else { f32::NAN })
  };
  let own_sup = line_m.map_or(([0.0; 4], f32::NAN), |(o, c)| with_origin(o, c));
  let piece_sup = |uc: f32| -> ([f32; 4], f32) {
    let (Some((o_m, c_m)), Some((o_p, c_p)), Some(_)) = (line_m, line_p, pred) else {
      return own_sup;
    };
    let age_u = age_o + uc * (age_p - age_o);
    let (o, c) = match spiral {
      Some((axis, omega)) => (
        spiral_point(o_m, o_p, age_o, age_p, axis, omega, uc),
        scale3(rotate3(c_p, axis, omega * (age_p - age_u)), age_u / age_p.max(1e-30)),
      ),
      None => (add3(o_m, scale3(sub3(o_p, o_m), uc)), add3(c_m, scale3(sub3(c_p, c_m), uc))),
    };
    with_origin(o, c)
  };
  let mut sup: alloc::vec::Vec<([f32; 4], f32)> = alloc::vec::Vec::new();
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
      sup.push(own_sup);
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
        let (u0, u1) = (bound(t), bound(t + 1));
        let uc = 0.5 * (u0 + u1);
        let sc = r0 + (1.0 - r0) * uc;
        let sc2 = sc * sc;
        let mean = along(p.mean, pm, u0);
        let seg_t = sub3(along(p.mean, pm, u1), mean);
        let base = turn(base, uc);
        sup.push(piece_sup(uc));
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
          flux: p.flux * (u1 - u0),
          open: [false, false],
          rho: [1.0, 1.0],
          seg_t: [0.0; 3],
        });
      }
    }
    return out.into_iter().zip(sup).map(|(p, (s, w))| (p, s, w, 0xFF)).collect();
  }
  let mut bins: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
  // the polyline's slow end bin as [`DUST_END_SUBBINS`] sub-bins ([`bin_subbins`], [`sub_edges`]):
  // there the chord's Gaussian and the mid-factor width reach past the slowest grain
  // (`AETHERVK_DUST_LAB_SUBBINS=J`, CPU only: every bin as J sub-bins; 1 the former packets)
  let nj_lab = lab_knob("AETHERVK_DUST_LAB_SUBBINS").map(|v| (v as usize).clamp(1, 16));
  // the cluster's chord groups: the level of detail's ([`lod_group`] of the polyline on screen,
  // [`lod_px_active`]) unless a group was requested ([`request_group`]: the lab's rule); `g` bins
  // per packet, no sub-bins then
  // (a cluster with bins walked for specks keeps every bin: the hand-over is per bin)
  let g = match requested_group() {
    1 if m.specks_mask() != 0 => 1,
    1 => px.map_or(1, |px| lod_group(polyline_extent_px(m, px), lod_px_active(), lod_group_depth_cap(lod_depth))),
    g => g,
  };
  set_group_active(g);
  for b in (0..SIZE_BINS as usize).step_by(g) {
  let nj = if g > 1 { 1 } else { nj_lab.unwrap_or_else(|| bin_subbins(b)) };
  let out_start = out.len();
  for j in 0..nj {
    let (ea, ec, _, _) = sub_edges(m, b, j, nj);
    // the time segment of this bin: to the *same bin* of the predecessor (its own sizes: a young
    // cluster's fast grains are ten times farther along than its slow ones)
    let seg_t = match pred {
      Some(p) => {
        let (pa, pc, _, _) = sub_edges(p, b, j, nj);
        sub3(scale3(add3(pa, pc), 0.5), scale3(add3(ea, ec), 0.5))
      }
      None => [0.0; 3],
    };
    // the longer of the bin's two axes is the exact (uniform) capsule, the other a Gaussian:
    // young dust has a time segment longer than its chords (the sample's own emission interval
    // reaches past its polyline), old dust a chord longer than its time segment
    let chord = sub3(ec, ea);
    let time_major = dot3(seg_t, seg_t) > dot3(chord, chord);
    if n_t == 1 {
      if time_major {
        out.push(bin_packet_sub(m, b, j, nj, seg_t));
        sup.push(own_sup);
      } else {
        // the time cell of the bin is the quadrilateral between its chord and the
        // predecessor's: swept when its edges are pixels apart on screen
        let n_s = time_sweep(m, pred, b, px);
        if n_s == 1 {
          let mut p = bin_chord_packet_sub(m, b, j, nj, seg_t);
          if TIME_CELL_OFFSET.load(core::sync::atomic::Ordering::Relaxed) {
            p.mean = add3(p.mean, scale3(seg_t, 0.5));
          }
          out.push(p);
          sup.push(own_sup);
        } else {
          let p = pred.expect("a sweep needs a predecessor");
          for k in 0..n_s {
            out.push(bin_sweep_packet_sub(m, p, b, j, nj, k, n_s));
            sup.push(own_sup);
          }
        }
      }
    } else if time_major {
      // the cone along the time capsule, as for a pooled packet
      let own = bin_packet_sub(m, b, j, nj, seg_t);
      let p = pred.expect("pieces need a predecessor");
      let base_packet = bin_packet_sub(p, b, j, nj, seg_t);
      let base = base_packet.cov;
      let pm = add3(own.mean, seg_t);
      for t in 0..n_t {
        let (u0, u1) = (bound(t), bound(t + 1));
        let uc = 0.5 * (u0 + u1);
        let sc = r0 + (1.0 - r0) * uc;
        let sc2 = sc * sc;
        let mean = along(own.mean, pm, u0);
        let seg_p = sub3(along(own.mean, pm, u1), mean);
        let base = turn(base, uc);
        sup.push(piece_sup(uc));
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
          flux: own.flux * (u1 - u0),
          open: [false, false],
          rho: [1.0, 1.0],
          seg_t: turn_vec(scale3(base_packet.seg_t, sc), uc),
        });
      }
    } else {
      // the cone: `n_t` pieces along the time segment, each with the predecessor's lateral
      // covariance scaled from the cluster's width (zero for the sample pinned at "now") to the
      // predecessor's, and the piece's own time Gaussian
      let p = pred.expect("pieces need a predecessor");
      let own = bin_chord_packet_sub(m, b, j, nj, [0.0; 3]);
      let cvp = p.cov_v();
      let (_, _, fa, fc) = sub_edges(m, b, j, nj);
      let f = 0.5 * (fa + fc);
      let f2 = f * f;
      let k = CHORD_VAR_FACTOR;
      let mid = scale3(add3(ea, ec), 0.5);
      let mid_p = add3(mid, seg_t);
      for t in 0..n_t {
        let (u0, u1) = (bound(t), bound(t + 1));
        let uc = 0.5 * (u0 + u1);
        let hseg = scale3(seg_t, 0.5 * (u1 - u0));
        let sc = r0 + (1.0 - r0) * uc;
        let sc2 = sc * sc * f2;
        let shift = sub3(along(mid, mid_p, u0), mid);
        let cvp = turn(cvp, uc);
        let cell = if TIME_CELL_OFFSET.load(core::sync::atomic::Ordering::Relaxed) {
          hseg
        } else {
          [0.0; 3]
        };
        sup.push(piece_sup(uc));
        out.push(Packet {
          mean: add3(add3(own.mean, shift), cell),
          cov: [
            cvp[0] * sc2 + k * hseg[0] * hseg[0],
            cvp[1] * sc2 + k * hseg[0] * hseg[1],
            cvp[2] * sc2 + k * hseg[0] * hseg[2],
            cvp[3] * sc2 + k * hseg[1] * hseg[1],
            cvp[4] * sc2 + k * hseg[1] * hseg[2],
            cvp[5] * sc2 + k * hseg[2] * hseg[2],
          ],
          seg: own.seg,
          flux: own.flux * (u1 - u0),
          open: own.open,
          rho: own.rho,
          seg_t: scale3(hseg, 2.0),
        });
      }
    }
  }
  bins.extend(core::iter::repeat(b as u8).take(out.len() - out_start));
  }
  set_group_active(1);
  out.into_iter().zip(sup).zip(bins).map(|((p, (s, w)), b)| (p, s, w, b)).collect()
}

/// [`subpackets_supported`] without the support planes.
pub fn subpackets_swept(
  m: &DustMoments,
  seg: [f32; 3],
  pred: Option<&DustMoments>,
  merged: bool,
  pred_merged: bool,
  px: Option<&dyn Fn([f32; 3]) -> Option<[f32; 2]>>,
) -> alloc::vec::Vec<Packet> {
  subpackets_supported(m, seg, pred, merged, pred_merged, px).into_iter().map(|(p, _)| p).collect()
}

/// The cell line of a sample (mirror of `dust_cell_line`): its size polyline is `o + f·c` in
/// the speed factor f (positions ∝ speed while radiation pressure has not bent the syndyne:
/// young to days-old dust), `o` the point its grains leave from (site plus drift), `c` the
/// reference grain's travel. `None` once bent: the middle edge more than 5 % of the chord off
/// that line, or a degenerate polyline.
pub fn cell_line(m: &DustMoments) -> Option<([f32; 3], [f32; 3])> {
  let nb = SIZE_BINS as usize;
  let (e0, e1) = (m.edge(0), m.edge(nb));
  let (f0, f1) = (m.edge_factor(0), m.edge_factor(nb));
  if !(f1 > f0) {
    return None;
  }
  let c = scale3(sub3(e1, e0), 1.0 / (f1 - f0));
  let o = sub3(e0, scale3(c, f0));
  let d = sub3(m.edge(nb / 2), add3(o, scale3(c, m.edge_factor(nb / 2))));
  let ch = sub3(e1, e0);
  if !(dot3(ch, ch) > 0.0) || dot3(d, d) > 0.0025 * dot3(ch, ch) {
    return None;
  }
  Some((o, c))
}

/// The support plane of a cell line ([`Photometry::support`], mirror of `dust_support_plane`):
/// normal along the motion `c`, through the slowest grain — the smallest speed factor `f_lo`
/// three relative speed spreads `v_std` slow (at the origin when that is ≤ 0): `[n, n·x]`
pub fn support_plane(o: [f32; 3], c: [f32; 3], v_std: f32, f_lo: f32) -> [f32; 4] {
  let cn = <f32 as FloatLike>::sqrt(dot3(c, c));
  if !(cn > 0.0) {
    return [0.0; 4];
  }
  let n = scale3(c, 1.0 / cn);
  let s0 = (f_lo * (1.0 - 3.0 * v_std)).max(0.0);
  let x = add3(o, scale3(n, cn * s0));
  [n[0], n[1], n[2], dot3(n, x)]
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

/// Bound `k` of `n` time pieces along a segment whose width grows `ratio`-fold (mirror of
/// `dust_piece_bound`): `(ratio^(k/n) − 1)/(ratio − 1)`, every piece the same width ratio (the
/// width ∝ `1/ratio + (1 − 1/ratio)·u` is linear in age). Equal steps left the first piece of a
/// young cone twice as wide at its old end as at its young one: its centre width smeared light
/// past the apex, where no grain is younger than the sample. Equal steps when `ratio ≈ 1`.
pub fn piece_bound(k: u32, n: u32, ratio: f32) -> f32 {
  if k >= n {
    return 1.0;
  }
  if !(ratio > 1.0001) {
    return k as f32 / n as f32;
  }
  let e = <f32 as FloatLike>::exp(<f32 as FloatLike>::ln(ratio) * (k as f32 / n as f32));
  ((e - 1.0) / (ratio - 1.0)).clamp(0.0, 1.0)
}

/// largest `σ_pred / σ_self` the cone pieces resolve (a zero-width cluster gets this)
pub const WIDTH_RATIO_MAX: f32 = 1024.0;

/// Pieces a capsule is cut into: `clamp(⌈σ_pred/σ_self⌉ − 1, 1, TIME_PIECES_MAX)` (mirror of
/// `dust_time_pieces`): one while the width changes by less than itself along the segment.
pub fn time_pieces(m: &DustMoments, pred: Option<&DustMoments>) -> u32 {
  let r = width_ratio(m, pred);
  let max = lab_knob("AETHERVK_DUST_LAB_PIECES").map_or(TIME_PIECES_MAX, |v| (v as u32).max(1));
  ((ce(r) as u32).saturating_sub(1)).clamp(1, max)
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
  /// the packet's physical support (local metres): the half-space `n·x ≥ w` (`xyz` = n, unit;
  /// zero = none) in front of which its cell's grains lie — no grain is behind the slowest grain
  /// of its cell along the motion ([`cell_support`]). A perspective packet the plane cuts is drawn
  /// by the exact ray integral over the half-space ([`volume_scatter`]) when the eye is near it,
  /// else as its truncation's moments ([`support_truncated`]), its flux renormalised by the kept
  /// mass. Mirror of `dust_phot_support`.
  pub support: [f32; 4],
  /// the packet's cell origin along the support's normal, `n·o` (the apex of its cone of grains,
  /// [`subpackets_cells`]; `NaN` = none): a perspective packet with a cell is drawn as its cone
  /// ([`conic_slabs`]). Mirror of `dust_phot_origin`.
  pub origin: f32,
}

impl Photometry {
  pub const NONE: Self = Self {
    albedo: 0.0,
    sun: [0.0, 0.0, 1.0],
    r_h_au: 1.0,
    support: [0.0; 4],
    origin: f32::NAN,
  };
}

/// Whether a support plane is set ([`Photometry::support`])
pub fn has_support(s: &[f32; 4]) -> bool {
  s[0] != 0.0 || s[1] != 0.0 || s[2] != 0.0
}

/// The mass of packet `p` in front of its support `s` (`n·x ≥ w`): the Gaussian's normal
/// marginal `Φ((n·x − w)/σ_n)` averaged over the capsule with its linear density, 16 midpoints
/// (mirror of `dust_support_mass`). 1 without a support.
pub fn support_mass(p: &Packet, s: &[f32; 4]) -> f32 {
  if !has_support(s) {
    return 1.0;
  }
  let n = [s[0], s[1], s[2]];
  let sig = <f32 as FloatLike>::sqrt(var_along(&p.cov, n).max(0.0));
  let (m0, ds) = (dot3(n, p.mean) - s[3], dot3(n, p.seg));
  let (mut acc, mut wsum) = (0.0f32, 0.0f32);
  for k in 0..16 {
    let u = (k as f32 + 0.5) / 16.0;
    let w = p.rho[0] + (p.rho[1] - p.rho[0]) * u;
    let x = m0 + ds * u;
    let f = if sig > 0.0 {
      0.5 * (1.0 + erf(x / (sig * core::f32::consts::SQRT_2)))
    } else if x >= 0.0 {
      1.0
    } else {
      0.0
    };
    acc += w * f;
    wsum += w;
  }
  if wsum > 0.0 { acc / wsum } else { 1.0 }
}

/// The least kept mass a cut packet is renormalised by: a packet almost wholly behind its
/// support (the young end of a cone piece near the apex) keeps at most 20× its density
pub const DUST_SUPPORT_MASS_MIN: f32 = 0.05;
/// A packet whose support cuts more than this share of its mass is drawn by the exact ray
/// integral over the half-space (perspective views)
pub const DUST_SUPPORT_CUT_MIN: f32 = 1.0e-3;

/// The mass of the 3σ ball of a 3-D standard normal, `P(χ²₃ ≤ 9)` = erf(3/√2) − 3√(2/π) e^{−9/2}:
/// a volumetric piece's density is its Gaussian inside its 3σ ellipsoid over this (step F: the
/// kernel's, the EWA leaves' and the Monte-Carlo's packets stop at 3σ; the unbounded integrand
/// drew the eye-near packets 1.6× their bounded grains). Mirror of `DUST_BALL3_MASS`.
pub const DUST_BALL3_MASS: f32 = 0.970_709_4;
/// The Mahalanobis radius² a volumetric piece is bounded at (3σ)
pub const DUST_BOUND_R2: f32 = 9.0;

/// A bounded small piece's variance per axis over its Gaussian's: `P(χ²₅ ≤ 9) / P(χ²₃ ≤ 9)` (the
/// 3σ ball's second moment), its screen spread narrowed by it ([`volume_small_points`])
pub const DUST_BOUND_VAR: f32 = 0.917_81;

/// The 3σ bound of a volumetric packet (step F2): its capsule — the segment `start + u·seg`
/// (`u ∈ [0, 1]`) ⊕ the 3σ ellipsoid of its base covariance — whitened by that covariance's
/// Cholesky factor `l` (`qe = L⁻¹(eye − start)`, `g = L⁻¹ seg`): the support of the bounded
/// Monte-Carlo's grains (a uniform chord point's 3σ ball). Every piece's ray integral is kept
/// inside it and inside the piece's own 3σ ellipsoid ([`ray_kept`]). Mirror of `DustVolumeBound`.
#[derive(Debug, Clone, Copy, Default)]
struct VolumeBound {
  l: [f32; 6],
  qe: [f32; 3],
  g: [f32; 3],
}

impl VolumeBound {
  fn new(base: &[f32; 6], start: [f32; 3], seg: [f32; 3], eye: [f32; 3]) -> Self {
    let l = cholesky3(base);
    Self {
      l,
      qe: cholesky_solve3(&l, sub3(eye, start)),
      g: cholesky_solve3(&l, seg),
    }
  }
}

/// The interval `[lo, hi]` of `t` where the ray `eye + t·u` is inside the capsule `b` (empty:
/// `lo > hi`): the union of its two end spheres' and its cylinder's (within the axial slab)
/// intervals — one interval, the capsule being convex. Mirror of `dust_capsule_interval`.
fn capsule_interval(b: &VolumeBound, u: [f32; 3]) -> (f32, f32) {
  let r2 = DUST_BOUND_R2;
  let w = cholesky_solve3(&b.l, u);
  let q = b.qe;
  let aa = dot3(w, w);
  let (mut lo, mut hi) = (f32::INFINITY, -f32::INFINITY);
  if !(aa > 0.0) {
    return (lo, hi);
  }
  // the end spheres
  for c in [[0.0f32; 3], b.g] {
    let qc = sub3(q, c);
    let bb = dot3(qc, w);
    let cc = dot3(qc, qc) - r2;
    let disc = bb * bb - aa * cc;
    if disc >= 0.0 {
      let sd = <f32 as FloatLike>::sqrt(disc);
      lo = lo.min((-bb - sd) / aa);
      hi = hi.max((-bb + sd) / aa);
    }
  }
  // the cylinder within its axial slab
  let gg = dot3(b.g, b.g);
  if gg > 0.0 {
    let gl = <f32 as FloatLike>::sqrt(gg);
    let gh = scale3(b.g, 1.0 / gl);
    let (wa, qa) = (dot3(w, gh), dot3(q, gh));
    let wp = sub3(w, scale3(gh, wa));
    let qp = sub3(q, scale3(gh, qa));
    let (a2, b2, c2) = (dot3(wp, wp), dot3(qp, wp), dot3(qp, qp) - r2);
    let (mut t1, mut t2) = (f32::INFINITY, -f32::INFINITY);
    if a2 > 1e-12 * aa {
      let disc = b2 * b2 - a2 * c2;
      if disc >= 0.0 {
        let sd = <f32 as FloatLike>::sqrt(disc);
        t1 = (-b2 - sd) / a2;
        t2 = (-b2 + sd) / a2;
      }
    } else if c2 <= 0.0 {
      t1 = -f32::INFINITY;
      t2 = f32::INFINITY;
    }
    if fabs(wa) > 1e-6 * <f32 as FloatLike>::sqrt(aa) {
      let (ta, tb) = ((0.0 - qa) / wa, (gl - qa) / wa);
      t1 = t1.max(ta.min(tb));
      t2 = t2.min(ta.max(tb));
    } else if !(qa >= 0.0 && qa <= gl) {
      t1 = f32::INFINITY;
      t2 = -f32::INFINITY;
    }
    if t2 > t1 {
      lo = lo.min(t1);
      hi = hi.max(t2);
    }
  }
  (lo, hi)
}

/// The share of a ray's bounded Gaussian in front of the eye and of the support (step F): the ray
/// `eye + t·u` (t ≥ 0), its Gaussian in `t` centred at `t* = −b/a` with variance `1/a` (whitened
/// `a = |L⁻¹u|²`, `b = L⁻¹u · L⁻¹(eye − μ)`), kept on the ray's chord of the piece's own 3σ
/// ellipsoid (`|t − t*| ≤ √((9 − rr2)/a)`, `rr2` the ray's Mahalanobis distance² from the piece's
/// mean; nothing when ≥ 9) and of the packet's 3σ capsule (`[cap_lo, cap_hi]`,
/// [`capsule_interval`]), on `t ≥ 0` and in the half-space `n·x ≥ w` when `s` is set: one
/// interval, the difference of two Φ, over [`DUST_BALL3_MASS`]. The piece's ellipsoid makes the
/// bounded Monte-Carlo's density exactly where the pieces are dense along the chord (a chord
/// point's 3σ ball, summed along it, is `φ₂(r)·[2Φ(√(9 − r²)) − 1]` across: what the pieces'
/// truncated Gaussians sum to); the capsule cuts the end pieces' Gaussian tails beyond the chord's
/// ends, where the Monte-Carlo's uniform chord has no grain. Mirror of `dust_ray_kept`.
#[allow(clippy::too_many_arguments)]
pub fn ray_kept(a: f32, b: f32, rr2: f32, cap_lo: f32, cap_hi: f32, eye: [f32; 3], u: [f32; 3], s: &[f32; 4]) -> f32 {
  if !(rr2 < DUST_BOUND_R2) {
    return 0.0;
  }
  let sa = <f32 as FloatLike>::sqrt(a);
  let t_star = -b / a;
  let k = sa * core::f32::consts::FRAC_1_SQRT_2;
  let h = <f32 as FloatLike>::sqrt((DUST_BOUND_R2 - rr2) / a);
  let mut lo = cap_lo.max(0.0).max(t_star - h);
  let mut hi = cap_hi.min(t_star + h);
  if has_support(s) {
    let n = [s[0], s[1], s[2]];
    let (nu, ne) = (dot3(n, u), dot3(n, eye) - s[3]);
    if fabs(nu) < 1e-12 {
      if !(ne >= 0.0) {
        return 0.0;
      }
    } else {
      let t_plane = -ne / nu;
      if nu > 0.0 {
        lo = lo.max(t_plane);
      } else {
        hi = hi.min(t_plane);
      }
    }
  }
  if !(hi > lo) {
    return 0.0;
  }
  (0.5 * (erf((hi - t_star) * k) - erf((lo - t_star) * k))).max(0.0) / DUST_BALL3_MASS
}

/// Sub-chords a cut packet the eye is not near is drawn as ([`support_truncated`])
pub const DUST_SUPPORT_TRUNC_PIECES: usize = 8;
/// Slabs along the support's normal per sub-chord ([`support_truncated`])
pub const DUST_SUPPORT_TRUNC_SLABS: usize = 4;
/// Packets [`support_truncated`] makes at most
pub const DUST_SUPPORT_TRUNC_MAX: usize = DUST_SUPPORT_TRUNC_PIECES * DUST_SUPPORT_TRUNC_SLABS;
/// The slabs' inner edges past the plane in standard deviations of the truncated marginal
/// ([`support_truncated`]); the last slab runs to infinity
const DUST_SUPPORT_SLAB_EDGES: [f32; DUST_SUPPORT_TRUNC_SLABS - 1] = [0.5, 1.2, 2.2];

/// A packet its support `s` cuts, as [`DUST_SUPPORT_TRUNC_PIECES`] sub-chords (shared out over
/// the chord's run through the plane's ±3σ zone, one each before and after it; interior ends
/// closed, Gaussian caps: the capsule integral splits exactly; open, each integrates its Gaussian
/// to infinity and relies on its neighbour's clip plane, which its own projection tilts — a
/// 4-px sub-chord of a 28-px-wide packet drew 1.7× its flux), each cut along the plane's normal
/// into [`DUST_SUPPORT_TRUNC_SLABS`] slabs replaced by their moments. The sub-chord's normal
/// marginal (the covariance's plus its uniform spread `(n·seg)²/12`), standardised, is kept for
/// `z ≥ α = −m/σ`; the slabs are `[α, α + e₁t)`, …, `[α + e₃t, ∞)` with `t` the truncated
/// marginal's deviation `√(1 + αλ − λ²)`, `λ = φ(α)/Φ(−α)` ([`DUST_SUPPORT_SLAB_EDGES`]). A slab
/// `[a, b)` holds `P = Φ(b) − Φ(a)`, its mean is `μ = (φ(a) − φ(b))/P` and its variance
/// `1 + (aφ(a) − bφ(b))/P − μ²`: the sub-chord moved by `μ·Σ_e n/σ` (`Σ_e` the covariance plus
/// the sub-chord's spread) and its covariance narrowed by `(1 − var)·(Σn)(Σn)ᵀ/σ²`. One Gaussian
/// for the whole truncation spread a quarter of its light back behind the plane (100 m: Σ 2.49
/// against ~1.8 for the exact cut). The fluxes are the kept shares renormalised over the packet
/// (its grains all lie in front; floor [`DUST_SUPPORT_MASS_MIN`]). The far side of the exact cut:
/// a small packet routed to the ray integral was point-sampled on the volume path's coarse texels
/// (1 km: Σ render/truth 1.14 → 0.42). Returns how many of `out` it filled (slabs with no mass
/// are dropped). Mirror of `dust_support_truncated`.
pub fn support_truncated(p: &Packet, s: &[f32; 4], out: &mut [Packet; DUST_SUPPORT_TRUNC_MAX]) -> usize {
  let n = [s[0], s[1], s[2]];
  let sn = [
    p.cov[0] * n[0] + p.cov[1] * n[1] + p.cov[2] * n[2],
    p.cov[1] * n[0] + p.cov[3] * n[1] + p.cov[4] * n[2],
    p.cov[2] * n[0] + p.cov[4] * n[1] + p.cov[5] * n[2],
  ];
  let v0 = dot3(n, sn).max(0.0);
  let k = DUST_SUPPORT_TRUNC_PIECES;
  // the standard normal's density and upper tail
  let pdf = |z: f32| <f32 as FloatLike>::exp(-0.5 * z * z) * 0.398_942_3;
  let upper = |z: f32| 0.5 * (1.0 - erf(z * core::f32::consts::FRAC_1_SQRT_2));
  let mut kept = [0.0f32; DUST_SUPPORT_TRUNC_MAX];
  let mut total = 0.0f32;
  let mut cnt = 0usize;
  // the sub-chords' bounds: where the chord runs through the plane's zone (its centre line
  // within 3 normal σ of the plane) the sub-chords share it equally, one each for the parts
  // before and after; equal when the zone holds the whole chord. A sub-chord's own extent along
  // the normal enters its marginal as a Gaussian spread: 4 equal sub-chords of a chord 9 σ long
  // across the plane read 10 % rms
  let mut bounds = [0.0f32; DUST_SUPPORT_TRUNC_PIECES + 1];
  for (j, b) in bounds.iter_mut().enumerate() {
    *b = j as f32 / k as f32;
  }
  let dn = dot3(n, p.seg);
  if fabs(dn) > 0.0 {
    let half = 3.0 * <f32 as FloatLike>::sqrt(v0) / fabs(dn);
    let u_x = (s[3] - dot3(n, p.mean)) / dn;
    let (za, zb) = ((u_x - half).max(0.0), (u_x + half).min(1.0));
    if zb > za && (za > 0.0 || zb < 1.0) {
      let (before, after) = ((za > 0.0) as usize, (zb < 1.0) as usize);
      let inner = k - before - after;
      bounds[0] = 0.0;
      for i in 0..=inner {
        bounds[before + i] = za + (zb - za) * i as f32 / inner as f32;
      }
      bounds[k] = 1.0;
    }
  }
  for j in 0..k {
    let (u0, u1) = (bounds[j], bounds[j + 1]);
    if !(u1 > u0) {
      continue;
    }
    let q = sub_chord(p, u0, u1, j == 0 && p.open[0], j + 1 == k && p.open[1]);
    // the ρ-weighted centre of the sub-chord and its normal marginal
    let (ra, rb) = (q.rho[0], q.rho[1]);
    let uc = if ra + rb > 0.0 { (ra + 2.0 * rb) / (3.0 * (ra + rb)) } else { 0.5 };
    let ds = dot3(n, q.seg);
    let m = dot3(n, add3(q.mean, scale3(q.seg, uc))) - s[3];
    let v = v0 + ds * ds / 12.0;
    if !(v > 0.0) {
      if m >= 0.0 {
        kept[cnt] = q.flux;
        total += q.flux;
        out[cnt] = q;
        cnt += 1;
      }
      continue;
    }
    let sig = <f32 as FloatLike>::sqrt(v);
    let alpha = -m / sig;
    let pk = upper(alpha);
    if !(pk > 1e-6) {
      continue;
    }
    let lam = pdf(alpha) / pk;
    let t = <f32 as FloatLike>::sqrt((1.0 + alpha * lam - lam * lam).max(1e-4));
    // Σ_e n over σ: the direction a standard unit along the normal moves the sub-chord
    let se = scale3(add3(sn, scale3(q.seg, ds / 12.0)), 1.0 / sig);
    for e in 0..DUST_SUPPORT_TRUNC_SLABS {
      let za = if e == 0 { alpha } else { alpha + DUST_SUPPORT_SLAB_EDGES[e - 1] * t };
      let last = e + 1 == DUST_SUPPORT_TRUNC_SLABS;
      let (pa, qa) = (pdf(za), upper(za));
      let (pb, qb, bpb) = if last {
        (0.0, 0.0, 0.0)
      } else {
        let zb = alpha + DUST_SUPPORT_SLAB_EDGES[e] * t;
        (pdf(zb), upper(zb), zb * pdf(zb))
      };
      let ps = qa - qb;
      if !(ps > 1e-9) {
        continue;
      }
      let mu = (pa - pb) / ps;
      let var = (1.0 + (za * pa - bpb) / ps - mu * mu).clamp(0.0, 1.0);
      let cv = (1.0 - var) / v;
      let cov = [
        q.cov[0] - cv * sn[0] * sn[0],
        q.cov[1] - cv * sn[0] * sn[1],
        q.cov[2] - cv * sn[0] * sn[2],
        q.cov[3] - cv * sn[1] * sn[1],
        q.cov[4] - cv * sn[1] * sn[2],
        q.cov[5] - cv * sn[2] * sn[2],
      ];
      kept[cnt] = q.flux * ps;
      total += q.flux * ps;
      out[cnt] = Packet { mean: add3(q.mean, scale3(se, mu)), cov, ..q };
      cnt += 1;
    }
  }
  let renorm = if p.flux > 0.0 { 1.0 / (total / p.flux).max(DUST_SUPPORT_MASS_MIN) } else { 0.0 };
  for j in 0..cnt {
    out[j].flux = kept[j] * renorm;
  }
  cnt
}

/// Sub-bins of the polyline's slow end bin ([`subpackets_cells`], [`sub_edges`]): the chord's
/// Gaussian ([`CHORD_VAR_FACTOR`]) and the bin's mid-factor width reached past the slowest grain
/// (100 m: the young slow end 1.4–28× the truth; 4 sub-bins: 0.81–0.99×). The fast end's gave
/// nothing measurable and cost 3× the CPU splat at 100 m, and its extra split children dropped a
/// near frame's split depth (the shell's parity case)
pub const DUST_END_SUBBINS: usize = 2;

/// The sub-bins size bin `b` is drawn as ([`DUST_END_SUBBINS`] for the slow end bin, else 1)
pub fn bin_subbins(b: usize) -> usize {
  if b == 0 && !legacy_kernel_on() { DUST_END_SUBBINS } else { 1 }
}

/// The index of size bin `b`'s first packet in a cluster's list ([`subpackets_cells`]) when every
/// sub-bin has `per` packets (its time pieces or sweeps); `b = SIZE_BINS`: the list's length
pub fn bin_first_packet(b: usize, per: usize) -> usize {
  (0..b).map(bin_subbins).sum::<usize>() * per
}

/// The slab edges of [`conic_slabs`] in standard deviations of a sub-chord's normal marginal: the
/// last slab runs to infinity, the first from the support plane (or −∞ when the plane is below
/// the first edge)
const DUST_CONIC_SLAB_EDGES: [f32; DUST_SUPPORT_TRUNC_SLABS - 1] = [-1.0, 0.0, 1.0];
/// "Infinity" of [`conic_slabs`]' slab edges, in standard deviations (the normal's density and
/// tail are exactly 0 and 1 there in f32)
const DUST_CONIC_FAR: f32 = 1.0e4;

/// A packet of a young cell as its **cone of grains** (mirror of `dust_conic_piece`): the
/// grains' lateral spread grows with their distance from the cell's origin (their own speed), a
/// packet carries one width along its length. Cut into the sub-chords and normal slabs of
/// [`support_truncated`] (the slab edges [`DUST_CONIC_SLAB_EDGES`], cut at the support plane `s`
/// where it cuts the packet), each slab's covariance mapped by the apex `A = nnᵀ + s_k(I − nnᵀ)`
/// (`Σ' = A Σ Aᵀ`) and its lateral shift (the Gaussian's tilt to the normal) scaled by `s_k`, where
/// `s_k = (n·μ_k − w_o)/(n·μ_c − w_o)` is the slab's distance from the origin plane `w_o` over
/// the packet centre's. The fluxes renormalised over the packet. Returns the count; 0 when the
/// packet has no cell (`w_o` not finite, or its centre not ahead of the origin).
pub fn conic_slabs(p: &Packet, s: &[f32; 4], w_o: f32, out: &mut [Packet; DUST_SUPPORT_TRUNC_MAX]) -> usize {
  if !has_support(s) || !w_o.is_finite() {
    return 0;
  }
  let n = [s[0], s[1], s[2]];
  let (ra, rb) = (p.rho[0], p.rho[1]);
  let uc_p = if ra + rb > 0.0 { (ra + 2.0 * rb) / (3.0 * (ra + rb)) } else { 0.5 };
  let centre = add3(p.mean, scale3(p.seg, uc_p));
  let d_c = dot3(n, centre) - w_o;
  if !(d_c > 0.0) {
    return 0;
  }
  let sn = [
    p.cov[0] * n[0] + p.cov[1] * n[1] + p.cov[2] * n[2],
    p.cov[1] * n[0] + p.cov[3] * n[1] + p.cov[4] * n[2],
    p.cov[2] * n[0] + p.cov[4] * n[1] + p.cov[5] * n[2],
  ];
  let v0 = dot3(n, sn).max(0.0);
  let cut = support_mass(p, s) < 1.0 - DUST_SUPPORT_CUT_MIN;
  let k = DUST_SUPPORT_TRUNC_PIECES;
  let pdf = |z: f32| <f32 as FloatLike>::exp(-0.5 * z * z) * 0.398_942_3;
  let upper = |z: f32| 0.5 * (1.0 - erf(z * core::f32::consts::FRAC_1_SQRT_2));
  let mut kept = [0.0f32; DUST_SUPPORT_TRUNC_MAX];
  let mut total = 0.0f32;
  let mut cnt = 0usize;
  for j in 0..k {
    let (u0, u1) = (j as f32 / k as f32, (j + 1) as f32 / k as f32);
    let q = sub_chord(p, u0, u1, j == 0 && p.open[0], j + 1 == k && p.open[1]);
    let (qa, qb) = (q.rho[0], q.rho[1]);
    let uc = if qa + qb > 0.0 { (qa + 2.0 * qb) / (3.0 * (qa + qb)) } else { 0.5 };
    let qc = add3(q.mean, scale3(q.seg, uc));
    let ds = dot3(n, q.seg);
    let v = v0 + ds * ds / 12.0;
    let sig = <f32 as FloatLike>::sqrt(v.max(1e-30));
    // the sub-chord's normal coordinate z of the plane (−DUST_CONIC_FAR when it does not cut: the
    // normal's density and tail are exactly 0 and 1 there, no special case)
    let alpha = if cut { ((s[3] - dot3(n, qc)) / sig).max(-DUST_CONIC_FAR) } else { -DUST_CONIC_FAR };
    let se = scale3(add3(sn, scale3(q.seg, ds / 12.0)), 1.0 / sig);
    // the slab edges above α
    let mut edges = [0.0f32; DUST_SUPPORT_TRUNC_SLABS + 1];
    edges[0] = alpha;
    let mut ne = 1;
    for &e in DUST_CONIC_SLAB_EDGES.iter() {
      if e > alpha {
        edges[ne] = e;
        ne += 1;
      }
    }
    edges[ne] = DUST_CONIC_FAR;
    for e in 0..ne {
      let (za, zb) = (edges[e], edges[e + 1]);
      let (pa, qa_, apa) = (pdf(za), upper(za), za * pdf(za));
      let (pb, qb_, bpb) = (pdf(zb), upper(zb), zb * pdf(zb));
      let ps = qa_ - qb_;
      if !(ps > 1e-9) {
        continue;
      }
      let mu = (pa - pb) / ps;
      let var = (1.0 + (apa - bpb) / ps - mu * mu).clamp(0.0, 1.0);
      let cv = (1.0 - var) / v.max(1e-30);
      let cov = [
        q.cov[0] - cv * sn[0] * sn[0],
        q.cov[1] - cv * sn[0] * sn[1],
        q.cov[2] - cv * sn[0] * sn[2],
        q.cov[3] - cv * sn[1] * sn[1],
        q.cov[4] - cv * sn[1] * sn[2],
        q.cov[5] - cv * sn[2] * sn[2],
      ];
      let mean = add3(q.mean, scale3(se, mu));
      // the apex map at the slab's centre
      let sk = ((dot3(n, add3(mean, scale3(q.seg, uc))) - w_o) / d_c).max(0.0);
      let a = |x: [f32; 3]| -> [f32; 3] {
        let xn = dot3(n, x);
        add3(scale3(n, xn), scale3(sub3(x, scale3(n, xn)), sk))
      };
      let cols = [a([cov[0], cov[1], cov[2]]), a([cov[1], cov[3], cov[4]]), a([cov[2], cov[4], cov[5]])];
      // A Σ Aᵀ: rows of A Σ are a(Σ's columns) (Σ symmetric), then A on the rows
      let r0 = a([cols[0][0], cols[1][0], cols[2][0]]);
      let r1 = a([cols[0][1], cols[1][1], cols[2][1]]);
      let r2 = a([cols[0][2], cols[1][2], cols[2][2]]);
      let cov = [r0[0], r0[1], r0[2], r1[1], r1[2], r2[2]];
      // the slab's lateral shift (the Gaussian's tilt to the normal), scaled; the sub-chord's
      // own place along the capsule stays
      let off = scale3(se, mu);
      let lat = sub3(off, scale3(n, dot3(n, off)));
      let mean = add3(mean, scale3(lat, sk - 1.0));
      kept[cnt] = q.flux * ps;
      total += q.flux * ps;
      out[cnt] = Packet { mean, cov, ..q };
      cnt += 1;
    }
  }
  let renorm = if p.flux > 0.0 { 1.0 / (total / p.flux).max(DUST_SUPPORT_MASS_MIN) } else { 0.0 };
  for j in 0..cnt {
    out[j].flux = kept[j] * renorm;
  }
  cnt
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
  // the deep chord's EWA ([`DUST_FRAME_DEEP_CHORD`], off by default): the Jacobian at the
  // geometric-mean depth and the flux by the perspective's light factor ([`deep_chord_factors`])
  let (u_j, deep_k) = if !shallow && deep_chord_on() && w > 0.0 && w_end > 0.0 {
    deep_chord_factors(w, w_end, p.rho)
  } else {
    (if shallow && !jacobian_at_start() { 0.5 } else { 0.0 }, 1.0)
  };
  let mid = add3(p.mean, scale3(p.seg, u_j));
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
  let mut amp = exposure * p.flux * deep_k * <f32 as FloatLike>::sqrt(det_g)
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
    self.suggested_bias_for(DUST_VOLUME_FRAME_BUDGET)
  }

  /// [`Self::suggested_bias`] against `budget` evaluations (the budget loop's knob, step 5)
  pub fn suggested_bias_for(&self, budget: u32) -> u32 {
    if self.evals > budget {
      (self.bias + 1).min(DUST_VOLUME_BIAS_MAX)
    } else if self.evals < budget / 8 {
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
  /// the volume detail (`DustSplatPushConstants::flags` bits 8–10, [`super::volume_detail_from_flags`];
  /// [`super::DUST_VOLUME_DETAIL_DEFAULT`] by default): the most pieces a volumetric packet is cut
  /// into when the frame's bias puts every piece on a 1 × 1 last level ([`volume_plan_pieces`])
  pub volume_detail: u32,
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
      volume_detail: super::DUST_VOLUME_DETAIL_DEFAULT,
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
  // lab (`AETHERVK_DUST_LAB_VOLUME_THIN=k`, CPU only): the rule on the packet's *thin* axis —
  // its smallest σ (2 det(L) / tr Σ, the volume plan's estimate) above `k` × the ratio × the
  // distance: a long needle seen from afar stays on the EWA split path
  if let Some(k) = lab_knob("AETHERVK_DUST_LAB_VOLUME_THIN").filter(|k| *k > 0.0) {
    let l = cholesky3(&p.cov);
    let sig_small = 2.0 * (l[0] * l[3] * l[5]) / tr.max(1e-30);
    let lim = k * DUST_VOLUME_RATIO;
    return sig_small * sig_small > lim * lim * dot3(r, r);
  }
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
/// not in front of the eye), whether it is on screen, and its own resolution level `lp`
/// ([`volume_level`] of its σ in px: for a non-full piece the thinnest σ of its screen covariance
/// at its mean, `√λ_min(P Σ Pᵀ)`; for a full one its angular σ, the plan's thin-σ estimate
/// `2 det L / tr Σ` over its mean's distance, × the focal length), which with the drawn level
/// picks how its texels are drawn ([`volume_class`]).
#[derive(Debug, Clone, Copy, Default)]
struct VolumePiece {
  l: [f32; 6],
  qw: [f32; 3],
  k: f32,
  bbox: [f32; 4],
  full: bool,
  on: bool,
  lp: u32,
  line: u32,
  /// its ray integral is clipped by the packet's capsule too ([`volume_ray_tau`]): the two pieces at
  /// each end of an axis, whose ellipsoids reach past the chord's ends
  cap: bool,
  /// the level of its screen σ alone (no angular estimate; `u32::MAX` when it has none): what a
  /// line piece's band must be thin by ([`volume_class`])
  ls: u32,
}

/// A volumetric piece drawn less than this many levels coarser than its own level `lp` is drawn
/// by the ray through each texel centre (the texel ≤ ~σ/2: the centre is the texel's value); from
/// `lp + DUST_VOLUME_COARSE_LEVELS` on, by its texel integral ([`volume_class`], K1c)
pub const DUST_VOLUME_COARSE_LEVELS: u32 = 2;
/// A coarse non-full piece whose projected ±3σ box is at most this many texels wide and high is
/// drawn as small ([`DUST_VOLUME_CLASS_SMALL`])
pub const DUST_VOLUME_SMALL_TEXELS: f32 = 2.0;
/// A coarse non-full piece this many levels or more coarser than its own level is drawn as small
/// whatever its box (its thin σ is under ~2 sub-ray spacings: sub-rays would miss it)
pub const DUST_VOLUME_THIN_LEVELS: u32 = 6;
/// A coarse full piece this many levels or more coarser than its own level (thin on screen: a
/// band its sub-rays would miss) is drawn by lines ([`DUST_VOLUME_CLASS_LINE`])
pub const DUST_VOLUME_LINE_LEVELS: u32 = 5;
/// [`volume_class`]: a full piece thin on screen, by `k` Gauss–Legendre lines across each texel ∩
/// frame along its band, each line's ray integral integrated across the band in closed form
/// (Laplace about the line's peak: [`volume_line_integral`]); `k` evaluations per range texel
pub const DUST_VOLUME_CLASS_LINE: u32 = 3;
/// The sub-ray order cap on the two coarsest levels and below them ([`volume_class`], step D:
/// 3 × 3 and 2 × 2 sub-rays instead of the 4 × 4 of the first K1c, the cost of the texel loop)
pub const DUST_VOLUME_SUBRAY_ORDER_TOP: u32 = 3;
pub const DUST_VOLUME_SUBRAY_ORDER_LOW: u32 = 2;
/// The largest Gauss–Legendre order per axis of a sub-ray texel, and the lines a line texel takes ([`volume_class`])
pub const DUST_VOLUME_SUBRAY_ORDER_MAX: u32 = 4;
/// [`volume_class`]: each texel of the range takes the ray through its centre (1 evaluation)
pub const DUST_VOLUME_CLASS_CENTRE: u32 = 0;
/// [`volume_class`]: the piece's screen integral by a 3-point depth quadrature, each point's
/// lateral Gaussian integrated over each texel ∩ frame (1 evaluation per range texel)
pub const DUST_VOLUME_CLASS_SMALL: u32 = 1;
/// [`volume_class`]: `k × k` Gauss–Legendre sub-rays over each texel ∩ frame (`k²` evaluations
/// per range texel)
pub const DUST_VOLUME_CLASS_SUBRAY: u32 = 2;

/// How a piece's texels at `level` are drawn and the sub-ray order it asks for (K1c, the texel's
/// integral; mirror of `dust_volume_class`): what a texel receives is the piece's mean optical
/// depth over the texel's in-frame part times the texel's area (the composite divides by the area
/// and shows it on the in-frame pixels). With `c = level − lp`: centre ray while
/// `c < DUST_VOLUME_COARSE_LEVELS`; past it a non-full piece whose box is at most
/// [`DUST_VOLUME_SMALL_TEXELS`] texels a side, or with `c ≥` [`DUST_VOLUME_THIN_LEVELS`], is small;
/// a full piece whose band is thin on screen by its screen σ alone (`level ≥ ls +`
/// [`DUST_VOLUME_LINE_LEVELS`]: the angular estimate of a far mean does not make a band) with the
/// eye outside its 3σ (it is full by a box corner behind the eye, not by holding the eye) takes
/// lines; the
/// rest takes sub-rays of order `min(c, DUST_VOLUME_SUBRAY_ORDER_MAX)` per axis (the texel is
/// `2^c / 8 … 2^c / 4` of its σ, so the sub-ray spacing stays ≲ σ up to `c = 4`).
fn volume_class(pc: &VolumePiece, level: u32, levels: u32) -> (u32, u32) {
  if level < pc.lp + DUST_VOLUME_COARSE_LEVELS || k1c_off() {
    return (DUST_VOLUME_CLASS_CENTRE, 1);
  }
  let c = level - pc.lp;
  let lim = DUST_VOLUME_SMALL_TEXELS * (1u32 << level) as f32;
  if !pc.full
    && ((pc.bbox[1] - pc.bbox[0] <= lim && pc.bbox[3] - pc.bbox[2] <= lim) || c >= DUST_VOLUME_THIN_LEVELS)
  {
    (DUST_VOLUME_CLASS_SMALL, 1)
  } else if pc.full
    && pc.line != 0
    && level >= pc.ls.saturating_add(DUST_VOLUME_LINE_LEVELS)
    && dot3(pc.qw, pc.qw) >= DUST_VOLUME_SIGMAS * DUST_VOLUME_SIGMAS
  {
    (DUST_VOLUME_CLASS_LINE, DUST_VOLUME_SUBRAY_ORDER_MAX)
  } else {
    // the order: 3 at most on the two coarsest levels (the 1 × 1 / 2 × 1 texels of a whole
    // frame), 2 below them (step D: the texel there is ≤ a quarter of the frame)
    let cap = if !k1c_d(DUST_K1C_D_SUBRAY_CAP) {
      DUST_VOLUME_SUBRAY_ORDER_MAX
    } else if level + 2 >= levels {
      DUST_VOLUME_SUBRAY_ORDER_TOP
    } else {
      DUST_VOLUME_SUBRAY_ORDER_LOW
    };
    (DUST_VOLUME_CLASS_SUBRAY, c.min(cap))
  }
}

/// The `i`-th node (in units of the half interval) and weight (summing to 1) of the `k`-point
/// Gauss–Legendre rule, `k` in 1..=4. Mirror of `dust_gl_node` (branches, no table).
pub fn gl_node(k: u32, i: u32) -> (f32, f32) {
  match (k, i) {
    (2, 0) => (-0.577_350_26, 0.5),
    (2, _) => (0.577_350_26, 0.5),
    (3, 0) => (-0.774_596_7, 5.0 / 18.0),
    (3, 1) => (0.0, 8.0 / 18.0),
    (3, _) => (0.774_596_7, 5.0 / 18.0),
    (4, 0) => (-0.861_136_3, 0.173_927_43),
    (4, 1) => (-0.339_981_04, 0.326_072_57),
    (4, 2) => (0.339_981_04, 0.326_072_57),
    (4, _) => (0.861_136_3, 0.173_927_43),
    _ => (0.0, 1.0),
  }
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
    self.suggested_bias_for(DUST_EWA_FRAME_BUDGET as u64)
  }

  /// [`Self::suggested_bias`] against `budget` texels (the budget loop's knob, step 5)
  pub fn suggested_bias_for(&self, budget: u64) -> u32 {
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
  /// the coarsest of its pieces' levels (each piece is drawn at its own: [`volume_piece_level`])
  pub level: u32,
  pub pieces: u32,
  pub evals: u32,
  /// the sub-ray order limit (K1c, [`volume_subray_limit`])
  pub subrays: u32,
  /// the texels its pieces' ranges hold at their levels (what the cap bounds; non-increasing in the
  /// bias)
  pub texels: u32,
  /// the pieces' level rule ([`volume_piece_level`])
  pub rule: VolumeLevelRule,
  /// tests ([`set_volume_packet_level`]): every piece at this one level (the pre-F8 rule)
  pub single: Option<u32>,
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

/// `L x` for the lower Cholesky factor `l` ([`cholesky3`]'s layout)
fn l_mul(l: &[f32; 6], v: [f32; 3]) -> [f32; 3] {
  [l[0] * v[0], l[1] * v[0] + l[3] * v[1], l[2] * v[0] + l[4] * v[1] + l[5] * v[2]]
}

/// `Lᵀ x` for the lower Cholesky factor `l`
fn lt_mul(l: &[f32; 6], v: [f32; 3]) -> [f32; 3] {
  [l[0] * v[0] + l[1] * v[1] + l[2] * v[2], l[3] * v[1] + l[4] * v[2], l[5] * v[2]]
}

/// The smaller σ (px) of a 2 × 2 screen covariance `xx xy yy`: `√(det / λ_max)`, no cancellation.
/// Mirror of `dust_volume_screen_sigma_min`.
fn volume_screen_sigma_min(xx: f32, xy: f32, yy: f32) -> f32 {
  let half = 0.5 * (xx + yy);
  let dd = <f32 as FloatLike>::sqrt((0.5 * (xx - yy)) * (0.5 * (xx - yy)) + xy * xy);
  let lam_max = half + dd;
  if !(lam_max > 0.0) {
    return 0.0;
  }
  <f32 as FloatLike>::sqrt(((xx * yy - xy * xy) / lam_max).max(0.0))
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
  f_px: f32,
  levels: u32,
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
    let mut on = full || !(x1 < 0.0 || y1 < 0.0 || x0 > wf || y0 > hf || !(x0 <= x1) || !(y0 <= y1));
    // a piece whose 3σ ellipsoid lies wholly behind the eye plane draws nothing (step F5: a
    // volumetric chord across the eye plane is drawn whole since F4, its pieces behind the eye
    // must cost nothing)
    let w_m0 = dot3(r3, mean_k) + m(3, 3);
    let lt3_0 = lt_mul(&lk, r3);
    if w_m0 + DUST_VOLUME_SIGMAS * <f32 as FloatLike>::sqrt(dot3(lt3_0, lt3_0)) < 0.0 {
      on = false;
      full = false;
    }
    // its own level (K1c): the thinnest σ of its screen covariance at its mean when the mean is in
    // front of the eye (always for a non-full piece: its box's corners are), and for a full one
    // the smaller of that and its angular σ (the plan's θ) — a full piece seen edge-on from
    // outside it is a thin band on screen; `line`: the screen axis its band is thinner along
    // (1 x, 2 y: [`DUST_VOLUME_CLASS_LINE`]). A full piece whose mean is behind the eye (the
    // eye in or beside a large packet whose bulk lies behind the camera: only its front flank is
    // in view) is measured at the conditional mean of its front part instead — the depth slice
    // at the truncated mean `w_f = w_m + σ_w φ(α)/Φ(−α)` (`α = −w_m/σ_w`) of its clip-w marginal,
    // its lateral covariance `Σ − (Σr₃)(Σr₃)ᵀ/σ_w²` projected there — unless its front part
    // is under Φ(−3) of it (0: no band)
    let w_m = dot3(r3, mean_k) + m(3, 3);
    let screen_at = |x: [f32; 3], w: f32, g: [f32; 3], gg: f32| -> (f32, u32) {
      let ndx = (dot3(r0, x) + m(0, 3)) / w;
      let ndy = (dot3(r1, x) + m(1, 3)) / w;
      let px = scale3(sub3(r0, scale3(r3, ndx)), 0.5 * wf / w);
      let py = scale3(sub3(r1, scale3(r3, ndy)), 0.5 * hf / w);
      let (lpx, lpy) = (lt_mul(&lk, px), lt_mul(&lk, py));
      let (pgx, pgy) = (dot3(px, g), dot3(py, g));
      let cxx = dot3(lpx, lpx) - gg * pgx * pgx;
      let cyy = dot3(lpy, lpy) - gg * pgy * pgy;
      let cxy = dot3(lpx, lpy) - gg * pgx * pgy;
      (volume_screen_sigma_min(cxx, cxy, cyy), if cxx <= cyy { 1 } else { 2 })
    };
    let (sig_scr, line) = if w_m > 0.0 {
      screen_at(mean_k, w_m, [0.0; 3], 0.0)
    } else if full {
      let lt3 = lt_mul(&lk, r3);
      let sw2 = dot3(lt3, lt3);
      let sw = <f32 as FloatLike>::sqrt(sw2);
      let alpha = if sw > 0.0 { -w_m / sw } else { f32::INFINITY };
      if alpha < 3.0 {
        let tail = 0.5 * (1.0 - erf(alpha * core::f32::consts::FRAC_1_SQRT_2));
        let lam = <f32 as FloatLike>::exp(-0.5 * alpha * alpha) / DUST_SQRT_2PI / tail;
        let w_f = w_m + sw * lam;
        let g = l_mul(&lk, lt3);
        let x_f = add3(mean_k, scale3(g, (w_f - w_m) / sw2));
        if w_f > 0.0 { screen_at(x_f, w_f, g, 1.0 / sw2) } else { (f32::INFINITY, 0) }
      } else {
        (f32::INFINITY, 0)
      }
    } else {
      (f32::INFINITY, 0)
    };
    let sig_px = if full {
      let sig_k = 2.0 * (lk[0] * lk[3] * lk[5]) / (sk[0] + sk[3] + sk[5]);
      let dq = sub3(eye, mean_k);
      let den_k = <f32 as FloatLike>::sqrt(dot3(dq, dq)).max(sig_k);
      ((if den_k > 0.0 { sig_k / den_k } else { 1.0 }) * f_px).min(sig_scr)
    } else {
      sig_scr
    };
    *pc = VolumePiece {
      l: lk,
      qw,
      k: exposure * flux_k / (2.0 * core::f32::consts::PI * (lk[0] * lk[3] * lk[5])),
      bbox: [x0, x1, y0, y1],
      full,
      on,
      // (σ capped at 1e30 px: the same level as an unbounded one, and finite for the GLSL's uint())
      lp: volume_level(sig_px.min(1e30), levels),
      cap: ax.n_a <= 4 || i < 2 || i + 2 >= ax.n_a as usize || (ax.has_b && (ax.n_b <= 4 || j < 2 || j + 2 >= ax.n_b as usize)),
      line,
      ls: if line != 0 { volume_level(sig_scr.min(1e30), levels) } else { u32::MAX },
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

/// The texels of a piece's range at `level`
fn volume_piece_texels(pc: &VolumePiece, level: u32, layout: &PyramidLayout) -> u32 {
  let (x0, x1, y0, y1) = volume_range(pc, level, layout);
  ((x1 - x0 + 1).max(0) * (y1 - y0 + 1).max(0)) as u32
}

/// The level rule of a packet's pieces (step F8): the frame's bias, the top level, the texel
/// share of the evaluation cap ([`volume_share_level`]; `u32::MAX`: none) and, for the tests'
/// former rules, a uniform `extra` and the full-piece floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeLevelRule {
  pub bias: u32,
  pub top: u32,
  pub share: u32,
  pub extra: u32,
  pub floor: bool,
}

/// The finest level from `from` (at most `top`) at which the piece's range holds at most `share`
/// texels. Mirror of `dust_volume_share_level`.
fn volume_share_level(pc: &VolumePiece, from: u32, top: u32, share: u32, layout: &PyramidLayout) -> u32 {
  let mut l = from.min(top);
  while l < top && volume_piece_texels(pc, l, layout) > share {
    l += 1;
  }
  l
}

/// A piece's level (step F8, the hole at the vanishing point of a chord seen along its length:
/// every piece was drawn at its packet's one level, set by its part nearest the eye, so a far
/// piece's few-px core was spread over the near part's 32-px texels): its own level `lp` plus the
/// frame's bias, and at least the level at which its range fits its share of the evaluation cap
/// (the share found at bias 0, [`volume_plan_pieces`]: so a bias step never refines a piece and
/// never adds texels — the budget loop's contract), at most `top`. (The tests' former rules: the
/// floor at [`DUST_VOLUME_MIN_LEVEL`] for a full piece, a uniform `extra`.) Mirror of
/// `dust_volume_piece_level`.
fn volume_piece_level(pc: &VolumePiece, rule: &VolumeLevelRule, layout: &PyramidLayout) -> u32 {
  let own = if rule.floor && pc.full { pc.lp.max(DUST_VOLUME_MIN_LEVEL) } else { pc.lp };
  let biased = own.saturating_add(rule.bias).saturating_add(rule.extra).min(rule.top);
  if rule.share == u32::MAX {
    return biased;
  }
  biased.max(volume_share_level(pc, own, rule.top, rule.share, layout))
}

/// The evaluations one texel of class `class` and order `k` costs (centre and small 1, lines `k`,
/// sub-rays `k²`)
fn volume_texel_cost(class: u32, k: u32) -> u32 {
  if class == DUST_VOLUME_CLASS_SUBRAY {
    k * k
  } else if class == DUST_VOLUME_CLASS_LINE {
    k
  } else {
    1
  }
}

/// The largest sub-ray or line order (≤ 4, at least 1) whose evaluations at `level` stay within
/// the piece's centre-class work one level above its own, `cost(k) · t(level) ≤ t(lp + 1)` (step
/// F10: a piece whose range no longer shrinks with the level cannot raise its order with the bias,
/// so a bias step never adds work — the budget loop's contract). Mirror of `dust_volume_order_bound`.
///
/// One stated exception (the closing round of F): a full sub-ray piece at its first coarse level,
/// `level = lp + 2` (c = 2, where the class asks for k = 2), keeps k = 2 when `4 · t(level) ≤ t(lp)`,
/// i.e. when its 2 × 2 sub-rays stay within its own-level centre work. The reason: under the plain
/// bound such a piece (the eye-inside packet at the gate level) fell to k = 1, one ray per 2²-px
/// texel across a range that is the piece's own core, and the texel test's gate case read Σ 1.0095,
/// worst texel 0.0330 against its 1 % / 2 % bounds. The cost: the piece's work may exceed
/// t(lp + 1) by at most t(lp) − t(lp + 1), and only at c = 2 — `volume_work_is_bounded` is the test
/// that this does not let a bias step add work.
fn volume_order_bound(pc: &VolumePiece, level: u32, class: u32, layout: &PyramidLayout) -> u32 {
  if class != DUST_VOLUME_CLASS_SUBRAY && class != DUST_VOLUME_CLASS_LINE {
    return 1;
  }
  let top = layout.level_count() - 1;
  let t1 = volume_piece_texels(pc, (pc.lp + 1).min(top), layout);
  let t = volume_piece_texels(pc, level, layout);
  let mut k = DUST_VOLUME_SUBRAY_ORDER_MAX;
  while k > 1 && volume_texel_cost(class, k).saturating_mul(t) > t1 {
    k -= 1;
  }
  if k < 2
    && class == DUST_VOLUME_CLASS_SUBRAY
    && pc.full
    && level == pc.lp + 2
    && 4u32.saturating_mul(t) <= volume_piece_texels(pc, pc.lp, layout)
  {
    k = 2;
  }
  k
}

/// A piece's class and order at `level` under the packet's sub-ray limit `k_cap` (and, outside the
/// pre-F8 hook, [`volume_order_bound`]): a non-full sub-ray piece whose order exceeds the limit is
/// drawn small; returns `(class, k, its evaluations)`. Mirror of the costs of `dust_volume_draw`.
fn volume_piece_class_k(pc: &VolumePiece, level: u32, k_cap: u32, bounded: bool, layout: &PyramidLayout) -> (u32, u32, u32) {
  let (c, kw) = volume_class(pc, level, layout.level_count());
  let t = volume_piece_texels(pc, level, layout);
  if c == DUST_VOLUME_CLASS_SUBRAY && !pc.full && kw > k_cap {
    return (DUST_VOLUME_CLASS_SMALL, 1, t);
  }
  if c != DUST_VOLUME_CLASS_SUBRAY && c != DUST_VOLUME_CLASS_LINE {
    return (c, 1, t);
  }
  let mut k = kw.min(k_cap).max(1);
  if bounded {
    k = k.min(volume_order_bound(pc, level, c, layout));
  }
  (c, k, volume_texel_cost(c, k).saturating_mul(t))
}

/// Σ over the first `n` pieces of the texels in their ranges at their own levels
/// ([`volume_piece_level`] with `bias`), or at the one level `single` (the tests' pre-F8 rule).
fn volume_evals_pp(
  pieces: &[VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
  n: u32,
  rule: &VolumeLevelRule,
  single: Option<u32>,
  layout: &PyramidLayout,
) -> u32 {
  let mut total = 0u32;
  for pc in pieces.iter().take(n as usize) {
    let lev = single.unwrap_or_else(|| volume_piece_level(pc, rule, layout));
    total = total.wrapping_add(volume_piece_texels(pc, lev, layout));
  }
  total
}

/// Σ over the first `n` pieces of the texels in their ranges at `level` (the plan's level choice
/// and the cap, as before K1c). Mirror of `dust_volume_evals`.
fn volume_evals(
  pieces: &[VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
  n: u32,
  level: u32,
  layout: &PyramidLayout,
) -> u32 {
  let mut total = 0u32;
  for pc in pieces.iter().take(n as usize) {
    total = total.wrapping_add(volume_piece_texels(pc, level, layout));
  }
  total
}

/// The sub-ray order limit of a packet at its level and the evaluations it makes (K1c): with the
/// range texels of the pieces drawn by one evaluation each (centre, small) and those of the
/// sub-ray pieces asking for order `k` (full ones `f_k`, non-full ones `n_k`), the evaluations at
/// a limit `K` are `t₁ + 4 ∧ K · t_line + Σ_k f_k min(k, K)² + Σ_{k ≤ K} n_k k² + Σ_{k > K} n_k`
/// (a line piece takes `min(4, K)` lines a texel): a non-full piece
/// the limit would under-sample is drawn as small instead (its light kept, at the texel's
/// resolution); `K` is the largest of 4, 3, 2, 1 that keeps them within `max_evals` (at 1 a full
/// piece's sub-ray texel takes one ray at its in-frame centre; the texels alone fit, the plan's
/// cap). Mirror of the sub-ray part of `dust_volume_draw`.
fn volume_subray_limit(
  pieces: &[VolumePiece; DUST_VOLUME_PIECES_MAX as usize],
  n: u32,
  rule: &VolumeLevelRule,
  single: Option<u32>,
  layout: &PyramidLayout,
  max_evals: u32,
) -> (u32, u32) {
  let evals = |kk: u32| -> u32 {
    let mut e = 0u32;
    for pc in pieces.iter().take(n as usize) {
      let level = single.unwrap_or_else(|| volume_piece_level(pc, rule, layout));
      e = e.wrapping_add(volume_piece_class_k(pc, level, kk, single.is_none(), layout).2);
    }
    e
  };
  let mut kk = DUST_VOLUME_SUBRAY_ORDER_MAX;
  while kk > 1 && evals(kk) > max_evals {
    kk -= 1;
  }
  (kk, evals(kk))
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
  bound: &mut VolumeBound,
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
  // the packet's 3σ capsule (step F2): the chord ⊕ the base covariance's ellipsoid; the time axis
  // as the segment when the chord is Gaussianised (TIME), the time axis Gaussianised into the
  // base when both are gridded (GRID: an enclosing capsule)
  *bound = if mode == DUST_VOLUME_AXIS_TIME {
    VolumeBound::new(&ax.s, sub3(ax.mean, scale3(ax.seg_b, 0.5)), ax.seg_b, eye)
  } else if mode == DUST_VOLUME_AXIS_GRID {
    let hb = scale3(ax.seg_b, 0.5);
    let sb = [
      ax.s[0] + kf * hb[0] * hb[0],
      ax.s[1] + kf * hb[0] * hb[1],
      ax.s[2] + kf * hb[0] * hb[2],
      ax.s[3] + kf * hb[1] * hb[1],
      ax.s[4] + kf * hb[1] * hb[2],
      ax.s[5] + kf * hb[2] * hb[2],
    ];
    VolumeBound::new(&sb, ax.mean, ax.seg_a, eye)
  } else {
    VolumeBound::new(&ax.s, ax.mean, ax.seg_a, eye)
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
  // the volume detail ([`SplitView::volume_detail`], PARTICLES_PERF round 2): when the bias puts
  // every piece on a 1 × 1 last level, at most that many pieces, fewer along `seg_t` first
  // (`dust_volume_draw` before its pass A; 16 changes nothing)
  {
    let lc = layout.level_count();
    let (_, tw, th) = layout.levels[(lc - 1) as usize];
    if bias + 1 >= lc && tw == 1 && th == 1 {
      let detail = view.volume_detail.max(1);
      while ax.n_a * ax.n_b > detail {
        if ax.n_b > 1 {
          ax.n_b = ax.n_b.div_ceil(2);
          ax.cvar_b = DUST_VOLUME_PIECE_VAR_CAPPED;
        } else if ax.n_a > 1 {
          ax.n_a = ax.n_a.div_ceil(2);
          ax.cvar_a = DUST_VOLUME_PIECE_VAR_CAPPED;
        } else {
          break;
        }
      }
    }
  }
  // the level from the angular σ
  let tr = s[0] + s[3] + s[5];
  let sig_small = 2.0 * (l[0] * l[3] * l[5]) / tr;
  let den = dist.max(sig_small);
  let theta = if den > 0.0 { sig_small / den } else { 1.0 };
  let levels = layout.level_count();
  let mut level = volume_level(theta * f_px, levels);
  volume_pieces(p, exposure, &ax, eye, mvp, wf, hf, f_px, levels, pieces);
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
  // K1 (2026-10-16, measured and set aside): on the 1 × 1 level a packet the eye is inside is one
  // evaluation on the centre ray spread over the frame, which loses 27 % of the old dust's light
  // and all of its structure at 10 m / 100 m. Stopping the bias at the gate level for the packets
  // with a full piece (`volume_full_top`, the EWA's rule) brought the old band only from 0.73 to
  // 0.81 of the truth and over-drew the 1–10 d band 1.03 → 1.20 (a 4 × 3 grid's centre rays
  // overhang the frame and its centre value × area is not the texel's mean), at +10 ms on the
  // GPU; the fix is the texel's integral (K1c in the record), not a level. The cap stays as the
  // lab's knobs, CPU only, off: `AETHERVK_DUST_LAB_VOLUME_FULL_TOP=1` caps the packets with a
  // full piece, `AETHERVK_DUST_LAB_VOLUME_CAP_ALL=1` every volumetric packet.
  let cap_all = lab_flag(&LAB_VOLUME_CAP_ALL, "AETHERVK_DUST_LAB_VOLUME_CAP_ALL");
  let cap_full = lab_flag(&LAB_VOLUME_FULL_TOP, "AETHERVK_DUST_LAB_VOLUME_FULL_TOP")
    || VOLUME_FULL_TOP_TEST.with(|c| c.get());
  let top = if cap_all || (any_full && cap_full) {
    volume_full_top(levels)
  } else {
    levels - 1
  };
  // tests ([`set_volume_packet_level`]): every piece at the packet's one level, the pre-F8 rule
  #[cfg(any(test, feature = "std"))]
  let packet_level = VOLUME_PACKET_LEVEL.with(|c| c.get());
  #[cfg(not(any(test, feature = "std")))]
  let packet_level = false;
  #[cfg(any(test, feature = "std"))]
  let (floor, cap_mode) = (VOLUME_LEVEL_FLOOR.with(|c| c.get()), VOLUME_CAP_MODE.with(|c| c.get()));
  #[cfg(not(any(test, feature = "std")))]
  let (floor, cap_mode) = (false, 0u8);
  let mut single = None;
  let mut rule = VolumeLevelRule { bias, top, share: u32::MAX, extra: 0, floor };
  if packet_level {
    level = (level + bias).min(top);
    loop {
      if volume_evals(pieces, ax.n_a * ax.n_b, level, layout) <= max_evals || level >= top {
        break;
      }
      level += 1;
    }
    single = Some(level);
  } else if cap_mode == 0 {
    // the evaluation cap (step F8/F10, production): every piece one level coarser until the texels
    // fit, or all are at the top (measured best within the cap on the chord seen along its length:
    // box rms 0.035 against the share's 0.055 and the first F8 rule's 0.087)
    loop {
      let n = ax.n_a * ax.n_b;
      let all_top = pieces.iter().take(n as usize).all(|pc| volume_piece_level(pc, &rule, layout) >= top);
      if volume_evals_pp(pieces, n, &rule, None, layout) <= max_evals || all_top {
        break;
      }
      rule.extra += 1;
    }
  } else if cap_mode == 1 {
    // the lab's share (step F9, set aside): when the pieces at their own levels (bias 0) hold more
    // texels than the cap, each piece takes the finest level from its own at which its range holds
    // at most `share` texels, `share` halved from the cap until the sum fits
    let n = ax.n_a * ax.n_b;
    let at_own = VolumeLevelRule { bias: 0, ..rule };
    if volume_evals_pp(pieces, n, &at_own, None, layout) > max_evals {
      let mut share = max_evals;
      while share > 1 && volume_evals_pp(pieces, n, &VolumeLevelRule { share, ..at_own }, None, layout) > max_evals {
        share /= 2;
      }
      rule.share = share;
    }
  }
  // then fewer pieces (along `seg_t` first); not under the tests' uncapped rule (cap mode 2: the
  // hole readings' uncapped (ii), whose pieces were never merged)
  while cap_mode != 2 && volume_evals_pp(pieces, ax.n_a * ax.n_b, &rule, single, layout) > max_evals {
    if ax.n_b > 1 {
      ax.n_b = ax.n_b.div_ceil(2);
      ax.cvar_b = DUST_VOLUME_PIECE_VAR_CAPPED;
    } else if ax.n_a > 1 {
      ax.n_a = ax.n_a.div_ceil(2);
      ax.cvar_a = DUST_VOLUME_PIECE_VAR_CAPPED;
    } else {
      break;
    }
    volume_pieces(p, exposure, &ax, eye, mvp, wf, hf, f_px, levels, pieces);
  }
  let n = ax.n_a * ax.n_b;
  // the sub-ray limit K: the largest of 4, 3, 2, 1 whose evaluations at the pieces' levels fit the
  // cap, every piece's order under the bound of [`volume_order_bound`]
  let (subrays, evals) = volume_subray_limit(pieces, n, &rule, single, layout, max_evals);
  let level = single.unwrap_or_else(|| {
    pieces.iter().take(n as usize).filter(|pc| pc.on).map(|pc| volume_piece_level(pc, &rule, layout)).max().unwrap_or(top)
  });
  let texels = volume_evals_pp(pieces, n, &rule, single, layout);
  #[cfg(any(test, feature = "std"))]
  sky_study_record(theta, n, any_full, evals);
  Some(VolumePlan { level, pieces: n, evals, subrays, texels, rule, single })
}

// The sky-map study (diagnostics, `AETHERVK_DUST_SKYMAP_STUDY=1` with `AETHERVK_DUST_WORK_LOG=1`
// in the replay lab): per volumetric packet its angular σ, pieces and screen evaluations, and the
// evaluations an octahedral map of base [`SKY_STUDY_BASE`]² would make at
// [`DUST_VOLUME_TEXELS_PER_SIGMA`] texels per σ (a piece's 3σ cone, the whole sphere when the eye
// is inside it), capped at [`DUST_VOLUME_MAX_EVALS`] per packet as on screen
#[cfg(any(test, feature = "std"))]
const SKY_STUDY_BASE: f32 = 128.0;
#[cfg(any(test, feature = "std"))]
#[derive(Clone, Copy, Default)]
struct SkyStudy {
  packets: u64,
  full: u64,
  screen_evals: u64,
  map_evals: u64,
  /// packets by angular σ: < 0.01, 0.01–0.03, 0.03–0.1, 0.1–0.3, 0.3–1, ≥ 1 rad
  theta_bins: [u64; 6],
  theta_min: f32,
}
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static SKY_STUDY: core::cell::Cell<SkyStudy> = const { core::cell::Cell::new(SkyStudy {
    packets: 0, full: 0, screen_evals: 0, map_evals: 0, theta_bins: [0; 6], theta_min: f32::INFINITY,
  }) };
}
#[cfg(any(test, feature = "std"))]
static SKY_STUDY_TOTAL: std::sync::Mutex<Option<SkyStudy>> = std::sync::Mutex::new(None);
#[cfg(any(test, feature = "std"))]
static LAB_SKY_STUDY: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
#[cfg(any(test, feature = "std"))]
fn sky_study_record(theta: f32, n: u32, full: bool, evals: u32) {
  if !lab_flag(&LAB_SKY_STUDY, "AETHERVK_DUST_SKYMAP_STUDY") {
    return;
  }
  let texel_min = <f32 as FloatLike>::sqrt(4.0 * core::f32::consts::PI) / SKY_STUDY_BASE;
  let texel = (theta / DUST_VOLUME_TEXELS_PER_SIGMA).max(texel_min);
  let sphere = 4.0 * core::f32::consts::PI / (texel * texel);
  let cone = core::f32::consts::PI * (3.0 * theta) * (3.0 * theta) / (texel * texel);
  let per_piece = if full { sphere } else { cone.min(sphere) };
  let map = (per_piece * n as f32).min(DUST_VOLUME_MAX_EVALS as f32) as u64;
  let bin = [0.01f32, 0.03, 0.1, 0.3, 1.0].iter().position(|&b| theta < b).unwrap_or(5);
  SKY_STUDY.with(|c| {
    let mut v = c.get();
    v.packets += 1;
    v.full += full as u64;
    v.screen_evals += evals as u64;
    v.map_evals += map;
    v.theta_bins[bin] += 1;
    v.theta_min = v.theta_min.min(theta);
    c.set(v);
  });
}
#[cfg(any(test, feature = "std"))]
fn sky_study_flush() {
  let v = SKY_STUDY.with(|c| c.replace(SkyStudy { theta_min: f32::INFINITY, ..SkyStudy::default() }));
  if v.packets == 0 {
    return;
  }
  let mut t = SKY_STUDY_TOTAL.lock().unwrap_or_else(|e| e.into_inner());
  let a = t.get_or_insert(SkyStudy { theta_min: f32::INFINITY, ..SkyStudy::default() });
  a.packets += v.packets;
  a.full += v.full;
  a.screen_evals += v.screen_evals;
  a.map_evals += v.map_evals;
  for k in 0..6 {
    a.theta_bins[k] += v.theta_bins[k];
  }
  a.theta_min = a.theta_min.min(v.theta_min);
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
    &mut VolumeBound::default(),
  )
}

/// `√3`: the 3-point Gauss–Hermite node in σ (weights 1/6, 2/3, 1/6)
pub const DUST_GH3_NODE: f32 = 1.732_050_8;
/// The 5-point Gauss–Hermite rule (probabilists'), a small piece's depth quadrature when its depth
/// spread is more than [`DUST_SMALL_DEPTH5_EPS`] of its distance: nodes 0, ±1.3556, ±2.8570
pub const DUST_GH5_N1: f32 = 1.355_626_2;
pub const DUST_GH5_N2: f32 = 2.856_970;
pub const DUST_GH5_W0: f32 = 0.533_333_3;
pub const DUST_GH5_W1: f32 = 0.222_075_9;
pub const DUST_GH5_W2: f32 = 0.011_257_4;
pub const DUST_SMALL_DEPTH5_EPS: f32 = 0.1;
/// `√2`: the lateral sigma points of a small piece's depth slice (2-D, ±√2 along two axes, ¼ each)
pub const DUST_UT2_NODE: f32 = core::f32::consts::SQRT_2;
/// `1 / 2π`
pub const DUST_INV_2PI: f32 = 0.159_154_94;
/// The σ a small point's texel cover reaches ([`volume_scatter`]: its rectangle integral beyond
/// it is under 6·10⁻⁵ of its light)
pub const DUST_SMALL_COVER: f32 = 4.0;
/// The largest correlation a small piece's lateral Gaussian is drawn with (the first-order
/// rectangle term, [`volume_small_points`])
pub const DUST_SMALL_RHO_MAX: f32 = 0.9;

/// Track 9A, the edge-piece demotion: a small piece that step D3 sends to the 4 sigma points a
/// slice (its centre slice's 3σ crosses the frame's edge) keeps its one projection a slice (with
/// the predicted second-order terms, [`volume_small_demote_fix`]) when its gate
/// ([`small_demote_gate`]) is under [`DUST_SMALL_DEMOTE_GATE`] × this and the
/// predicted change of its in-frame light, sigma points against one projection, as a fraction of
/// its mass ([`volume_small_demote_err`], mass-weighted over the slices), is **under** this. Strict:
/// at 0 the predictor is not evaluated and every decision is D3's, bit for bit. Mirror of
/// `DUST_SMALL_DEMOTE_EPS` (dust_scatter.glsl); the value comes from
/// `small_points_tests::small_demotion_calibration` (its ε_cal). CPU lab override:
/// `AETHERVK_DUST_LAB_SMALL_EPS` ([`small_demote_eps`]).
pub const DUST_SMALL_DEMOTE_EPS: f32 = 0.001;
/// t9a2c: the demotion's gate factor: a piece is predicted only when its gate
/// ([`small_demote_gate`]) is under this × ε. Part of the rule (it only removes demotions, so
/// soundness holds as before); 0.5 keeps 99.96 % of the pieces the predictor alone would demote at
/// ε 10⁻³ (≥ 99.9 % at every ε of the calibration's grid) and sends ~9 % of the others to the
/// predictor (offline f64 replay of the calibration's 15 357 edge pieces, t9a2c REVIEW); the
/// calibration applies it (`small_points_tests::small_demotion_calibration` prints what it drops).
/// Mirror of `DUST_SMALL_DEMOTE_GATE` (dust_scatter.glsl)
pub const DUST_SMALL_DEMOTE_GATE: f32 = 0.5;
/// The predictor's range: a slice is predicted only when its lateral sigma points' relative depth
/// change `√2 |κ_i|` (`κ_i = r3·lat_i / w`) is under this (every sigma point in front of the eye,
/// the series' third-order terms ≲ 10⁻³ of the second-order ones); otherwise not demoted
pub const DUST_SMALL_DEMOTE_KAPPA: f32 = 0.1;
/// The factor on the predicted second-order terms (μ, δ, the σ and ρ changes): the offline f64
/// check (2 × 10⁴ pieces) found |actual| / E ≤ 0.995 at 1 and ≤ 0.80 at 1.25
pub const DUST_SMALL_DEMOTE_SAFETY: f32 = 1.25;
/// `1 / √(2π)`
pub const DUST_SMALL_INV_SQRT_2PI: f32 = 0.398_942_3;
/// `e^{-1/2}`: the largest `τ e^{-τ²/2}`
pub const DUST_SMALL_DEMOTE_SLOPE_MAX: f32 = 0.606_530_66;

#[cfg(any(test, feature = "std"))]
std::thread_local! {
  /// tests: this thread's ε in place of the process's ([`set_small_demote_eps_test`]; NaN: none)
  static SMALL_DEMOTE_EPS_TEST: core::cell::Cell<f32> = const { core::cell::Cell::new(f32::NAN) };
}
/// Tests: this thread's draws demote with `eps` (`None`: the process's, `AETHERVK_DUST_LAB_SMALL_EPS`
/// or [`DUST_SMALL_DEMOTE_EPS`]); CPU only, the GLSL has its constant
#[cfg(any(test, feature = "std"))]
pub fn set_small_demote_eps_test(eps: Option<f32>) {
  SMALL_DEMOTE_EPS_TEST.with(|c| c.set(eps.unwrap_or(f32::NAN)));
}
/// the lab's ε (f32 bits; `u32::MAX`: not read yet)
static LAB_SMALL_EPS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);
/// The ε of [`volume_small_points`]' demotion: the test hook's, else the lab's
/// (`AETHERVK_DUST_LAB_SMALL_EPS=<ε ≥ 0>`, CPU only, read once per process), else
/// [`DUST_SMALL_DEMOTE_EPS`] (the GPU's)
fn small_demote_eps() -> f32 {
  use core::sync::atomic::Ordering::Relaxed;
  #[cfg(any(test, feature = "std"))]
  {
    let t = SMALL_DEMOTE_EPS_TEST.with(|c| c.get());
    if !t.is_nan() {
      return t;
    }
  }
  let b = LAB_SMALL_EPS.load(Relaxed);
  if b != u32::MAX {
    return f32::from_bits(b);
  }
  let v = lab_knob("AETHERVK_DUST_LAB_SMALL_EPS").filter(|v| *v >= 0.0).unwrap_or(DUST_SMALL_DEMOTE_EPS);
  LAB_SMALL_EPS.store(v.to_bits(), Relaxed);
  v
}

/// One frame edge of a demoted slice's marginal ([`volume_small_demote_err`]): the edge `tv` σ
/// from the mean (positive inside), its predicted move `dt = u + |tv| al` (σ) under the mean's
/// shift `u` (σ) and the σ's relative change `al`; returns the bound of the marginal's change
/// there `dP = φ(a) dt` (mean value theorem: `φ` decreases in `|τ|`, `a = max(0, |tv| − dt)` the
/// closest the edge comes to the mean), of its Gaussian's change `dG = dt · sup τ e^{-τ²/2}`
/// over the move, `e^{-a²/2}` (≥ the Gaussian at the edge) and the marginal's upper bound `pu`
/// (Mills: `Φ(tv) ≤ φ(tv) / |tv|` beyond 1 σ outside). Mirror of `dust_small_demote_edge`.
fn small_demote_edge(tv: f32, u: f32, al: f32) -> (f32, f32, f32, f32) {
  let dt = u + tv.abs() * al;
  let a = (tv.abs() - dt).max(0.0);
  let ea = <f32 as FloatLike>::exp(-0.5 * a * a);
  let dp = DUST_SMALL_INV_SQRT_2PI * dt * ea;
  let dg = dt * if a >= 1.0 { a * ea } else { DUST_SMALL_DEMOTE_SLOPE_MAX };
  let pu = if tv < -1.0 { DUST_SMALL_INV_SQRT_2PI * ea / -tv } else { 1.0 };
  (dp, dg, ea, pu)
}

/// Track 9A (t9a2c): the demotion's gate, from the centre slice's one projection `c` (mean
/// `mx`, `my`, `ix = 1/(√2 σx)`, `iy`) and its sigma points' relative depth spread
/// `κ = √2 max |r3·lat_i| / w` (`w = r3·mean + t3`; every predicted term is of first order in it):
/// the in-frame light an edge can carry over, `κ/√(2π) · (e^{-ax²/2} p_y + p_x e^{-ay²/2})`, with
/// `ax`, `ay` the signed distances (σ) from the mean to each axis' nearer edge and `p_x`, `p_y`
/// the marginals' upper bounds (Mills beyond 1σ outside: `φ(a)/|a|`, else 1); `+∞` when the centre
/// is behind the eye or `κ` ≥ [`DUST_SMALL_DEMOTE_KAPPA`] (the centre slice not predicted). Not a
/// bound of the predictor: a cheap filter in front of it (the GPU evaluates the predictor only for
/// the pieces under [`DUST_SMALL_DEMOTE_GATE`] × ε). Mirror of `dust_small_demote_gate`.
#[allow(clippy::too_many_arguments)]
fn small_demote_gate(
  c: &VolumeSmallPoint,
  mean: [f32; 3],
  r3: [f32; 3],
  t3: f32,
  w1: f32,
  w2: f32,
  wf: f32,
  hf: f32,
) -> f32 {
  let w0 = dot3(r3, mean) + t3;
  if !(w0 > 0.0) {
    return f32::INFINITY;
  }
  // (the centre slice's κ_i as small_demote_terms computes them)
  let iw = 1.0 / w0;
  let kap = DUST_UT2_NODE * (w1 * iw).abs().max((w2 * iw).abs());
  if !(kap < DUST_SMALL_DEMOTE_KAPPA) {
    return f32::INFINITY;
  }
  let (isx, isy) = (c.ix * core::f32::consts::SQRT_2, c.iy * core::f32::consts::SQRT_2);
  let ax = c.mx.min(wf - c.mx) * isx;
  let ay = c.my.min(hf - c.my) * isy;
  let ex = <f32 as FloatLike>::exp(-0.5 * ax * ax);
  let ey = <f32 as FloatLike>::exp(-0.5 * ay * ay);
  let pxu = if ax < -1.0 { DUST_SMALL_INV_SQRT_2PI * ex / -ax } else { 1.0 };
  let pyu = if ay < -1.0 { DUST_SMALL_INV_SQRT_2PI * ey / -ay } else { 1.0 };
  kap * DUST_SMALL_INV_SQRT_2PI * (ex * pyu + pxu * ey)
}

/// Track 9A: one predicted slice's second-order terms ([`small_demote_terms`]): the mass term `mu`
/// (the sigma points' J3 sum over the one projection's, less 1), the J3-weighted mean shift `dx`,
/// `dy` (px), the one projection's lateral covariance `cxx`, `cyy`, `cxy` (px², the variances
/// floored at 10⁻⁶ as its own) and the sigma points' change of it `dcxx`, `dcyy`, `dcxy`, the one
/// projection's mean `mx`, `my` (px). Mirror of `DustSmallTerms` (dust_scatter.glsl).
#[derive(Debug, Clone, Copy)]
struct SmallDemoteTerms {
  mu: f32,
  dx: f32,
  dy: f32,
  cxx: f32,
  cyy: f32,
  cxy: f32,
  dcxx: f32,
  dcyy: f32,
  dcxy: f32,
  mx: f32,
  my: f32,
}

/// Track 9A: the second-order terms of one small slice, its 4 lateral sigma points' J3-weighted
/// moments against its one projection's (the derivation at [`volume_small_demote_err`]); `None`
/// when not predicted (the slice behind the eye, `√2 |κ_i|` ≥ [`DUST_SMALL_DEMOTE_KAPPA`], no
/// Jacobian). The predictor bounds their effect on the in-frame light; a demoted piece's slices
/// carry them ([`volume_small_demote_fix`]). Mirror of `dust_small_demote_terms`.
#[allow(clippy::too_many_arguments)]
fn small_demote_terms(
  xq: [f32; 3],
  d: [f32; 6],
  r0: [f32; 3],
  r1: [f32; 3],
  r3: [f32; 3],
  t: [f32; 3],
  wf: f32,
  hf: f32,
) -> Option<SmallDemoteTerms> {
  let [x1, x2, y1, y2, w1, w2] = d;
  let w = dot3(r3, xq) + t[2];
  if !(w > 0.0) {
    return None;
  }
  let iw = 1.0 / w;
  let (k1, k2) = (w1 * iw, w2 * iw);
  if !(DUST_UT2_NODE * k1.abs().max(k2.abs()) < DUST_SMALL_DEMOTE_KAPPA) {
    return None;
  }
  let ndx = (dot3(r0, xq) + t[0]) * iw;
  let ndy = (dot3(r1, xq) + t[1]) * iw;
  let c = cross3(sub3(r0, scale3(r3, ndx)), sub3(r1, scale3(r3, ndy)));
  let cn2 = dot3(c, c);
  if !(cn2 > 0.0) {
    return None;
  }
  let icn = 1.0 / cn2;
  // the Jacobian's lateral columns in ndc, and c's change along them
  let (al1, al2) = ((x1 - ndx * w1) * iw, (x2 - ndx * w2) * iw);
  let (be1, be2) = ((y1 - ndy * w1) * iw, (y2 - ndy * w2) * iw);
  let (cxv, cyv) = (cross3(r1, r3), cross3(r3, r0));
  let v1 = add3(scale3(cxv, al1), scale3(cyv, be1));
  let v2 = add3(scale3(cxv, al2), scale3(cyv, be2));
  let (g1, g2) = (dot3(c, v1) * icn, dot3(c, v2) * icn);
  let n1 = (dot3(v1, v1) * icn - g1 * g1).max(0.0);
  let n2 = (dot3(v2, v2) * icn - g2 * g2).max(0.0);
  // the second-order terms: mass μ, mean shift δ (px), covariance change (px²)
  let mu = (3.0 * k1 * k1 - 3.0 * k1 * g1 + 0.5 * n1) + (3.0 * k2 * k2 - 3.0 * k2 * g2 + 0.5 * n2);
  let (h1, h2) = (g1 - 3.0 * k1, g2 - 3.0 * k2);
  let (a1, a2) = (0.5 * wf * al1, 0.5 * wf * al2);
  let (b1, b2) = (0.5 * hf * be1, 0.5 * hf * be2);
  let dx = a1 * h1 + a2 * h2;
  let dy = b1 * h1 + b2 * h2;
  let f1 = 20.0 * k1 * k1 - 10.0 * k1 * g1 + n1;
  let f2 = 20.0 * k2 * k2 - 10.0 * k2 * g2 + n2;
  let cxx = (a1 * a1 + a2 * a2).max(1e-6);
  let cyy = (b1 * b1 + b2 * b2).max(1e-6);
  let cxy = a1 * b1 + a2 * b2;
  let dcxx = a1 * a1 * f1 + a2 * a2 * f2 - mu * cxx - dx * dx;
  let dcyy = b1 * b1 * f1 + b2 * b2 * f2 - mu * cyy - dy * dy;
  let dcxy = a1 * b1 * f1 + a2 * b2 * f2 - mu * cxy - dx * dy;
  let mx = (ndx * 0.5 + 0.5) * wf;
  let my = (ndy * 0.5 + 0.5) * hf;
  Some(SmallDemoteTerms { mu, dx, dy, cxx, cyy, cxy, dcxx, dcyy, dcxy, mx, my })
}

/// Track 9A (t9a2): a demoted slice's one projection `p` with its predicted second-order terms
/// ([`small_demote_terms`]) applied: mass × (1 + μ), mean + δ, covariance + ΔC (the variances
/// floored at 10⁻⁶ px² as the one projection's, ρ clamped as [`volume_small_points`]' finish): the
/// sigma points' moments to second order. Without them a demoted piece kept the one projection's
/// systematic excess over the sigma points (`small_demotion_calibration`, f64 replay of its
/// distribution: Σ (L_fast − L_sigma) = +5.2·10⁻³ of the demoted pieces' light at ε 10⁻², +2.3·10⁻³
/// at 10⁻³; with them |L − L_sigma| ≤ 0.02 E M per piece). `p` unchanged when the slice has no
/// prediction or no mass (a demoted piece has both on every slice). Mirror of
/// `dust_small_demote_fix_of` (the GLSL applies it inside `dust_small_demote_slice`, on the terms its
/// prediction made; `p.mx`, `p.my` are bit for bit the terms' `mx`, `my`).
#[allow(clippy::too_many_arguments)]
fn volume_small_demote_fix(
  p: VolumeSmallPoint,
  xq: [f32; 3],
  d: [f32; 6],
  r0: [f32; 3],
  r1: [f32; 3],
  r3: [f32; 3],
  t: [f32; 3],
  wf: f32,
  hf: f32,
) -> VolumeSmallPoint {
  let Some(k) = small_demote_terms(xq, d, r0, r1, r3, t, wf, hf) else {
    return p;
  };
  if !(p.m > 0.0) {
    return p;
  }
  let cxx = (k.cxx + k.dcxx).max(1e-6);
  let cyy = (k.cyy + k.dcyy).max(1e-6);
  let rho_raw = (k.cxy + k.dcxy) * (1.0 / <f32 as FloatLike>::sqrt(cxx * cyy));
  let rho = if rho_raw.is_finite() { rho_raw.clamp(-DUST_SMALL_RHO_MAX, DUST_SMALL_RHO_MAX) } else { 0.0 };
  let m = p.m * (1.0 + k.mu);
  if !(m > 0.0) || !m.is_finite() {
    return VolumeSmallPoint::default();
  }
  VolumeSmallPoint {
    mx: p.mx + k.dx,
    my: p.my + k.dy,
    ix: (1.0 / <f32 as FloatLike>::sqrt(cxx)) * core::f32::consts::FRAC_1_SQRT_2,
    iy: (1.0 / <f32 as FloatLike>::sqrt(cyy)) * core::f32::consts::FRAC_1_SQRT_2,
    rho,
    m,
  }
}

/// Track 9A: the predicted change of one small slice's in-frame light, its 4 lateral sigma points
/// (D3's edge path) against its one projection (the fast path), as a fraction of the slice's mass;
/// `−1` when not predicted (the slice behind the eye, `√2 |κ_i|` ≥ [`DUST_SMALL_DEMOTE_KAPPA`], no
/// Jacobian). `xq` the slice's centre, `d` the lateral axes' dots `[x1 x2 y1 y2 w1 w2]`
/// (`r0·lat_i`, `r1·lat_i`, `r3·lat_i`), `t` the projection's offsets `(m03, m13, m33)`.
///
/// With `s` a sigma point's offset along `lat_i` (±√2), its ndc is `n + s α_i / (1 + s κ_i)`
/// (`κ_i = r3·lat_i / w`, `α_i`, `β_i` the Jacobian's columns in ndc) and its J3 weight
/// `|c(s)| / |c₀| · (1 + s κ_i)⁻²` (`c` affine in ndc: `c(n) = c₀ + Δnx (r1 × r3) + Δny (r3 × r0)`,
/// so `|c|/|c₀| = 1 + s γ_i + s²(ν_i²/2 − κ_i γ_i)`, `γ_i = c₀·v_i / |c₀|²`, `ν_i²` the rest of
/// `|v_i|² / |c₀|²`). To second order the rule's moments (E s² = 1, E s⁴ = 2 per axis) give the mass
/// term `μ = Σ_i 3κ_i² − 3κ_iγ_i + ν_i²/2` (the sigma points' J3 sum over the one projection's,
/// less 1), the J3-weighted mean shift `δ = Σ_i p_i (γ_i − 3κ_i)` (`p_i` the column in px) and the
/// covariance change `Σ_i p_i p_iᵀ (20κ_i² − 10κ_iγ_i + ν_i²) − μ C − δδᵀ`. Their effect on the
/// in-frame rectangle integral (`¼ fx fy + ρ/2π gx gy` over the frame, [`volume_scatter`]'s small
/// branch) is bounded edge by edge ([`small_demote_edge`]): the marginals' change, the ρ term's
/// shape and ρ's own change, and the mass term on the marginals' upper bounds; every predicted
/// term × [`DUST_SMALL_DEMOTE_SAFETY`]. Mirror of `dust_small_demote_err_of` (on
/// `dust_small_demote_terms`' terms).
#[allow(clippy::too_many_arguments)]
fn volume_small_demote_err(
  xq: [f32; 3],
  d: [f32; 6],
  r0: [f32; 3],
  r1: [f32; 3],
  r3: [f32; 3],
  t: [f32; 3],
  wf: f32,
  hf: f32,
) -> f32 {
  let Some(k) = small_demote_terms(xq, d, r0, r1, r3, t, wf, hf) else {
    return -1.0;
  };
  let SmallDemoteTerms { mu, dx, dy, cxx, cyy, cxy, dcxx, dcyy, dcxy, mx, my } = k;
  let lx = dcxx / (2.0 * cxx);
  let ly = dcyy / (2.0 * cyy);
  let sx = <f32 as FloatLike>::sqrt(cxx);
  let sy = <f32 as FloatLike>::sqrt(cyy);
  let isc = 1.0 / <f32 as FloatLike>::sqrt(cxx * cyy);
  let rho = cxy * isc;
  let drho = dcxy * isc - rho * (lx + ly);
  // the frame's edges in σ from the slice's mean (positive inside)
  let (tl, tr) = (mx / sx, (wf - mx) / sx);
  let (tb, tt) = (my / sy, (hf - my) / sy);
  let s = DUST_SMALL_DEMOTE_SAFETY;
  let (ux, uy) = (s * dx.abs() / sx, s * dy.abs() / sy);
  let (alx, aly) = (s * lx.abs(), s * ly.abs());
  let (adr, amu) = (s * drho.abs(), s * mu.abs());
  let el = small_demote_edge(tl, ux, alx);
  let er = small_demote_edge(tr, ux, alx);
  let eb = small_demote_edge(tb, uy, aly);
  let et = small_demote_edge(tt, uy, aly);
  let (dpx, dpy) = (el.0 + er.0, eb.0 + et.0);
  let (dgx, dgy) = (el.1 + er.1, eb.1 + et.1);
  let (gx, gy) = (el.2.max(er.2), eb.2.max(et.2));
  let pxu = 1.0f32.min(el.3).min(er.3);
  let pyu = 1.0f32.min(eb.3).min(et.3);
  let arho = rho.abs().min(DUST_SMALL_RHO_MAX);
  amu * (pxu * pyu + arho * DUST_INV_2PI * gx * gy)
    + (1.0 + amu)
      * (dpx * (pyu + dpy)
        + pxu * dpy
        + (arho + adr) * DUST_INV_2PI * (dgx * (gy + dgy) + gx * dgy)
        + adr * DUST_INV_2PI * gx * gy)
}

/// The optical depth per px² of a volumetric piece along the ray on the two planes of constant
/// ndc `a_row`, `b_row` (in front of the eye): the closed form of [`volume_scatter`], the support
/// and the eye cut ([`ray_kept`]), the photometry at the ray's nearest point. `−1` when the planes
/// give no ray. Mirror of the ray part of `dust_volume_segment`.
#[allow(clippy::too_many_arguments)]
fn volume_ray_tau(
  pc: &VolumePiece,
  bound: &VolumeBound,
  renorm: f32,
  eye: [f32; 3],
  a_row: [f32; 3],
  b_row: [f32; 3],
  r3: [f32; 3],
  phot: &Photometry,
  mean_ref: [f32; 3],
) -> f32 {
  let mut v = cross3(a_row, b_row);
  if dot3(v, r3) < 0.0 {
    v = scale3(v, -1.0);
  }
  let vn = <f32 as FloatLike>::sqrt(dot3(v, v));
  if !(vn > 0.0) {
    return -1.0;
  }
  let u = scale3(v, 1.0 / vn);
  let uw = cholesky_solve3(&pc.l, u);
  let a = dot3(uw, uw);
  let b = dot3(uw, pc.qw);
  let rr = sub3(pc.qw, scale3(uw, b / a));
  let rr2 = dot3(rr, rr);
  let taper = k1c_d(DUST_K1C_D_TAPER) && dot3(bound.g, bound.g) > 0.0;
  // the capsule only for the pieces whose ellipsoid can reach past the chord's ends (`pc.cap`:
  // the two at each end; an interior piece's ellipsoid lies inside the capsule — step F5, the
  // capsule's cost per ray), and for every piece under the lab's taper
  let (cap_lo, cap_hi) = if pc.cap || taper { capsule_interval(bound, u) } else { (-f32::INFINITY, f32::INFINITY) };
  if !(cap_hi > cap_lo.max(0.0)) {
    return 0.0;
  }
  let sa = <f32 as FloatLike>::sqrt(a);
  let kept = if taper {
    // the lab's taper (DUST_K1C_D_TAPER, CPU only): no piece ellipsoid; the Monte-Carlo's lateral
    // profile at the ray's closest approach to the piece
    let w = cholesky_solve3(&bound.l, u);
    let y = add3(bound.qe, scale3(w, -b / a));
    let gl = <f32 as FloatLike>::sqrt(dot3(bound.g, bound.g));
    let gh = scale3(bound.g, 1.0 / gl);
    let yp = sub3(y, scale3(gh, dot3(y, gh)));
    let r2 = dot3(yp, yp);
    if !(r2 < DUST_BOUND_R2) {
      return 0.0;
    }
    ray_kept(a, b, -1e30, cap_lo, cap_hi, eye, u, &phot.support)
      * erf(<f32 as FloatLike>::sqrt(0.5 * (DUST_BOUND_R2 - r2)))
  } else {
    ray_kept(a, b, rr2, cap_lo, cap_hi, eye, u, &phot.support)
  };
  let mut tau_px = pc.k * renorm / sa * <f32 as FloatLike>::exp(-0.5 * rr2) * kept;
  if phot.albedo > 0.0 {
    let x = add3(eye, scale3(u, (-b / a).max(0.0)));
    tau_px *= volume_weight(phot, mean_ref, x, u);
  }
  tau_px
}

/// `√(2π)`
pub const DUST_SQRT_2PI: f32 = 2.506_628_3;

/// `∫ τ dt` of a volumetric piece along one screen line of a texel (K1c,
/// [`DUST_VOLUME_CLASS_LINE`]): the row `y = fixed` over `x ∈ [lo, hi]` (`along_x`) or the
/// column `x = fixed` over `y ∈ [lo, hi]`. The ray direction is linear along the line, `v(t) =
/// v_m + t v₁` (`t` from the middle), so with `w = L⁻¹v` the whitened `a(t) = |w|²` is quadratic,
/// `b(t) = w·q'` linear and the ray's Mahalanobis distance² `|q'|² − b²/a` has one stationary
/// minimum in closed form, `t* = (b₀a₁ − b₁a₀)/(b₁a₁ − b₀a₂)`, of curvature
/// `2(a₂b*² − b₁²a*)/a*²`: the line's profile is the exact ray integral at `t*` times its Gaussian
/// (Laplace) about it, integrated over `[lo, hi]` by erf. A thin band across the line is then
/// integrated exactly however coarse the texel. No minimum (a line along the band): 4-point
/// Gauss–Legendre of the ray integral instead. Mirror of `dust_volume_line_integral`.
#[allow(clippy::too_many_arguments)]
fn volume_line_integral(
  pc: &VolumePiece,
  bound: &VolumeBound,
  renorm: f32,
  eye: [f32; 3],
  r0: [f32; 3],
  r1: [f32; 3],
  r3: [f32; 3],
  wf: f32,
  hf: f32,
  phot: &Photometry,
  mean_ref: [f32; 3],
  along_x: bool,
  fixed: f32,
  lo: f32,
  hi: f32,
) -> f32 {
  let mid = 0.5 * (lo + hi);
  // the planes of constant ndc at the line coordinate `t` (absolute)
  let planes = |t: f32| -> ([f32; 3], [f32; 3]) {
    if along_x {
      (sub3(r0, scale3(r3, 2.0 * t / wf - 1.0)), sub3(r1, scale3(r3, 2.0 * fixed / hf - 1.0)))
    } else {
      (sub3(r0, scale3(r3, 2.0 * fixed / wf - 1.0)), sub3(r1, scale3(r3, 2.0 * t / hf - 1.0)))
    }
  };
  let tau_at = |t: f32| -> f32 {
    let (a_r, b_r) = planes(t);
    let v = volume_ray_tau(pc, bound, renorm, eye, a_r, b_r, r3, phot, mean_ref);
    if v >= 0.0 && v.is_finite() { v } else { 0.0 }
  };
  let fallback = || -> f32 {
    let (h, mut acc) = (0.5 * (hi - lo), 0.0f32);
    for i in 0..DUST_VOLUME_SUBRAY_ORDER_MAX {
      let (ni, wi) = gl_node(DUST_VOLUME_SUBRAY_ORDER_MAX, i);
      acc += wi * tau_at(mid + h * ni);
    }
    acc * (hi - lo)
  };
  let (a_m, b_m) = planes(mid);
  let mut vm = cross3(a_m, b_m);
  let mut v1 = if along_x {
    scale3(cross3(r3, b_m), -2.0 / wf)
  } else {
    scale3(cross3(a_m, r3), -2.0 / hf)
  };
  if dot3(vm, r3) < 0.0 {
    vm = scale3(vm, -1.0);
    v1 = scale3(v1, -1.0);
  }
  let wm = cholesky_solve3(&pc.l, vm);
  let w1 = cholesky_solve3(&pc.l, v1);
  let (a0, a1, a2) = (dot3(wm, wm), dot3(wm, w1), dot3(w1, w1));
  let (b0, b1) = (dot3(wm, pc.qw), dot3(w1, pc.qw));
  let den = b1 * a1 - b0 * a2;
  if !(den != 0.0) {
    return fallback();
  }
  let ts = (b0 * a1 - b1 * a0) / den;
  let at = a0 + 2.0 * a1 * ts + a2 * ts * ts;
  let bt = b0 + b1 * ts;
  let d2 = 2.0 * (a2 * bt * bt - b1 * b1 * at) / (at * at);
  if !(d2 > 0.0) || !ts.is_finite() {
    return fallback();
  }
  let sg = <f32 as FloatLike>::sqrt(2.0 / d2);
  let t = mid + ts;
  // the band's rays reach the piece's 3σ ellipsoid where their distance² ≤ 9 (step F): about
  // the minimum `r*²` that is `|t − t*| ≤ s √(9 − r*²)`
  let r2 = dot3(pc.qw, pc.qw) - bt * bt / at;
  if !(r2 < DUST_BOUND_R2) {
    return 0.0;
  }
  let half = sg * <f32 as FloatLike>::sqrt(DUST_BOUND_R2 - r2);
  let (lo, hi) = (lo.max(t - half), hi.min(t + half));
  if !(hi > lo) {
    return 0.0;
  }
  let k = 1.0 / (sg * core::f32::consts::SQRT_2);
  tau_at(t) * sg * DUST_SQRT_2PI * 0.5 * (erf((hi - t) * k) - erf((lo - t) * k))
}

/// One point of a small piece's depth quadrature ([`volume_small_points`]): its screen position
/// (px), `1 / (σ √2)` of its lateral Gaussian along x and y, their correlation (± 0.9 at most),
/// and its share of the piece's screen integral (τ · px²); `m = 0`: none.
#[derive(Debug, Clone, Copy, Default)]
struct VolumeSmallPoint {
  mx: f32,
  my: f32,
  ix: f32,
  iy: f32,
  rho: f32,
  m: f32,
}

#[cfg(all(test, feature = "std"))]
std::thread_local! {
  /// Tests: each point's f32 conditioning in the last [`volume_small_points`] call, `(|r3| |x_q| +
  /// |t3| + √2 max |r3·lat_i|) / min |w|` over the projections it made (the w ≤ 0 ones included):
  /// the relative f32 resolution of its w is ~4 ε × this (`factored_small_points_match_the_f10_build`)
  static SMALL_POINT_COND: core::cell::Cell<[f32; 5]> = const { core::cell::Cell::new([0.0; 5]) };
}
/// The screen integral of a small volumetric piece ([`DUST_VOLUME_CLASS_SMALL`], K1c) as three
/// points along its depth axis: with `e` the unit direction from the eye to its mean and
/// `g = Σe / σ_e` (`σ_e² = eᵀΣe`), the points `μ`, `μ ∓ √3 g` (the conditional means of the
/// 3-point Gauss–Hermite rule along `e`; weights 2/3, 1/6, 1/6). Each carries `exposure · flux`
/// × the support's renormalisation × the piece's kept mass in the half-space
/// (`Φ((n·μ − w)/σ_n)`) × `J3`, the px² per steradian over the distance², exactly the Jacobian of
/// (px, py, distance) (`W H |a × b| / (4 w²)`, `a`, `b` the planes of constant ndc through a
/// point: `∫ τ dp = ∫ ρ J3 d³x`) averaged over the depth slice's 4 lateral sigma points
/// (`x_q ± √2 L b_i`, `b₁ b₂` unit axes across the depth axis in the whitened space: the slice's
/// covariance `Σ − g gᵀ` exactly), × its radiance weight; the slice's screen mean and x, y
/// variances are the J3-weighted moments of the projected sigma points (the perspective's skew of
/// an off-axis piece moves its light outwards: a frame edge cuts it right). A texel takes the
/// product of the x and y marginals over its in-frame rectangle. Mirror of
/// `dust_volume_small_points`.
#[allow(clippy::too_many_arguments)]
fn volume_small_points(
  pc: &VolumePiece,
  eye: [f32; 3],
  mvp: &[f32; 16],
  wf: f32,
  hf: f32,
  phot: &Photometry,
  mean_ref: [f32; 3],
  renorm: f32,
) -> [VolumeSmallPoint; 5] {
  let mut out = [VolumeSmallPoint::default(); 5];
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let l = &pc.l;
  // eye − mean
  let lq = l_mul(l, pc.qw);
  let mean = sub3(eye, lq);
  let dn = <f32 as FloatLike>::sqrt(dot3(lq, lq));
  if !(dn > 0.0) {
    return out;
  }
  let e = scale3(lq, -1.0 / dn);
  let lte = lt_mul(l, e);
  let se = <f32 as FloatLike>::sqrt(dot3(lte, lte));
  if !(se > 0.0) {
    return out;
  }
  // the depth axis in the whitened space and two unit axes across it (Duff et al. 2017): the
  // depth points move along `g = L ζ`, the slice spreads along `L b₁`, `L b₂`
  let z = scale3(lte, 1.0 / se);
  let g = l_mul(l, z);
  let sgn = if z[2] >= 0.0 { 1.0f32 } else { -1.0 };
  let ia = -1.0 / (sgn + z[2]);
  let ib = z[0] * z[1] * ia;
  // bounded at 3σ (step F2): the spread narrowed to the 3σ ball's second moment
  let sb = <f32 as FloatLike>::sqrt(DUST_BOUND_VAR);
  let g = scale3(g, sb);
  let lat1 = scale3(l_mul(l, [1.0 + sgn * z[0] * z[0] * ia, sgn * ib, -sgn * z[0]]), sb);
  let lat2 = scale3(l_mul(l, [ib, sgn + z[1] * z[1] * ia, -z[1]]), sb);
  let s = &phot.support;
  let kept = if has_support(s) {
    let n = [s[0], s[1], s[2]];
    let ltn = lt_mul(l, n);
    let sn = <f32 as FloatLike>::sqrt(dot3(ltn, ltn));
    let mn = dot3(n, mean) - s[3];
    if sn > 0.0 {
      0.5 * (1.0 + erf(mn / (sn * core::f32::consts::SQRT_2)))
    } else if mn >= 0.0 {
      1.0
    } else {
      0.0
    }
  } else {
    1.0
  };
  let mass = 2.0 * core::f32::consts::PI * pc.k * (l[0] * l[3] * l[5]) * renorm * kept;
  // the radiance weight once, at the mean (step D: a small piece spans a small range of phase
  // angles; three weights a piece were a third of its cost)
  let one_weight = k1c_d(DUST_K1C_D_ONE_WEIGHT);
  let w_rad = if phot.albedo > 0.0 && one_weight {
    let d = sub3(mean, eye);
    let u = scale3(d, 1.0 / <f32 as FloatLike>::sqrt(dot3(d, d)));
    volume_weight(phot, mean_ref, mean, u)
  } else {
    1.0
  };
  // one depth point: through its slice's 4 lateral sigma points (`edge`) or one projection with
  // its screen Jacobian's lateral covariance (step D: 3 projections a piece instead of 12)
  let mass_w = mass * w_rad;
  // the depth quadrature: 3 Gauss–Hermite points, or 5 for a piece deep along the view (its
  // bounded depth σ over its distance above DUST_SMALL_DEPTH5_EPS: the 1/d² of a 3-point rule
  // drew a chord seen along its length 5 % short, `bounded_chords_match_the_bounded_monte_carlo`)
  let five = k1c_d(DUST_K1C_D_GH5) && se * <f32 as FloatLike>::sqrt(DUST_BOUND_VAR) > DUST_SMALL_DEPTH5_EPS * dn;
  // the lateral axes against the projection's rows (cost track 2): a sigma point `x_q ± √2 lat_i`
  // projects through `r·x_q ± √2 r·lat_i`, and the Jacobian's lateral columns are
  // `(W / 2w) (r0·lat_i − ndx r3·lat_i)`, `(H / 2w) (r1·lat_i − ndy r3·lat_i)`; lat1, lat2 are dead
  // after these six dots (live across the slices: mean, g, the six dots, the mass — 13 floats, as
  // mean, g, lat1, lat2 and the mass were)
  let (x1, x2) = (dot3(r0, lat1), dot3(r0, lat2));
  let (y1, y2) = (dot3(r1, lat1), dot3(r1, lat2));
  let (w1, w2) = (dot3(r3, lat1), dot3(r3, lat2));
  #[cfg(all(test, feature = "std"))]
  let (cond, r3n) = (core::cell::Cell::new([0.0f32; 5]), <f32 as FloatLike>::sqrt(dot3(r3, r3)));
  // the depth node (σ) and Gauss–Hermite weight of point q
  let node = |q: usize| -> (f32, f32) {
    if five {
      match q {
        0 => (0.0, DUST_GH5_W0),
        1 => (-DUST_GH5_N1, DUST_GH5_W1),
        2 => (DUST_GH5_N1, DUST_GH5_W1),
        3 => (-DUST_GH5_N2, DUST_GH5_W2),
        _ => (DUST_GH5_N2, DUST_GH5_W2),
      }
    } else {
      match q {
        0 => (0.0, 2.0 / 3.0),
        1 => (-DUST_GH3_NODE, 1.0 / 6.0),
        _ => (DUST_GH3_NODE, 1.0 / 6.0),
      }
    }
  };
  let xq_of = |q: usize| add3(mean, scale3(g, node(q).0));
  // the radiance weight of a slice at its centre (per depth point; 1 without albedo or under the
  // lab's one weight a piece): computed before the slice's projections (the GLSL issues its LUT
  // reads ahead of them); the centre slice's serves both paths. Final round A2: the Sun's
  // direction and r_h are the centre slice's for every slice ([`volume_weight`]'s arithmetic at
  // x_0: the slices lie ≤ 2.9 σ_depth from it against r_h AU, so their own differ by ≤ ~1e-5
  // relative at the far poses), each slice with its own phase angle and table interpolation
  // ([`super::phase::radiance_weight`]'s arithmetic): the centre's weight is bit-identical
  let (sun0, r_au0) = if phot.albedo > 0.0 && !one_weight {
    let d = sub3(xq_of(0), mean_ref);
    let r_vec = add3(scale3(phot.sun, -phot.r_h_au * AU_M_F32), d);
    let rn = <f32 as FloatLike>::sqrt(dot3(r_vec, r_vec)).max(1.0);
    (scale3(r_vec, -1.0 / rn), rn / AU_M_F32)
  } else {
    ([0.0; 3], 1.0)
  };
  let weight_at = |xq: [f32; 3]| -> f32 {
    if phot.albedo > 0.0 && !one_weight {
      let d = sub3(xq, eye);
      let u = scale3(d, 1.0 / <f32 as FloatLike>::sqrt(dot3(d, d)));
      super::phase::radiance_weight(phot.albedo, sun0, scale3(u, -1.0), r_au0)
    } else {
      1.0
    }
  };
  let finish = |wq: f32, wr: f32, sj: f32, mx: f32, my: f32, cxx: f32, cyy: f32, rho_raw: f32| -> VolumeSmallPoint {
    let rho = if rho_raw.is_finite() { rho_raw.clamp(-DUST_SMALL_RHO_MAX, DUST_SMALL_RHO_MAX) } else { 0.0 };
    let mut wgt = wq * mass_w * sj;
    wgt *= wr;
    if !(wgt > 0.0) || !wgt.is_finite() {
      return VolumeSmallPoint::default();
    }
    // (round 3: by the reciprocal square root, the GLSL's inversesqrt)
    VolumeSmallPoint {
      mx,
      my,
      ix: (1.0 / <f32 as FloatLike>::sqrt(cxx)) * core::f32::consts::FRAC_1_SQRT_2,
      iy: (1.0 / <f32 as FloatLike>::sqrt(cyy)) * core::f32::consts::FRAC_1_SQRT_2,
      rho,
      m: wgt,
    }
  };
  // one projection with the screen Jacobian's lateral covariance (step D); one reciprocal of w
  // (round 3)
  let fast_point = |q: usize, wr: f32| -> VolumeSmallPoint {
    let wq = node(q).1;
    let xq = xq_of(q);
    let w = dot3(r3, xq) + m(3, 3);
    #[cfg(all(test, feature = "std"))]
    {
      let mut v = cond.get();
      v[q] = (r3n * <f32 as FloatLike>::sqrt(dot3(xq, xq)) + m(3, 3).abs()) / w.abs();
      cond.set(v);
    }
    if !(w > 0.0) {
      return VolumeSmallPoint::default();
    }
    let iw = 1.0 / w;
    let ndx = (dot3(r0, xq) + m(0, 3)) * iw;
    let ndy = (dot3(r1, xq) + m(1, 3)) * iw;
    let c = cross3(sub3(r0, scale3(r3, ndx)), sub3(r1, scale3(r3, ndy)));
    let sj = 0.25 * wf * hf * <f32 as FloatLike>::sqrt(dot3(c, c)) * (iw * iw);
    let (kx, ky) = (0.5 * wf * iw, 0.5 * hf * iw);
    let (a1, a2) = (kx * (x1 - ndx * w1), kx * (x2 - ndx * w2));
    let (b1, b2) = (ky * (y1 - ndy * w1), ky * (y2 - ndy * w2));
    let cxx = (a1 * a1 + a2 * a2).max(1e-6);
    let cyy = (b1 * b1 + b2 * b2).max(1e-6);
    let rho_raw = (a1 * b1 + a2 * b2) * (1.0 / <f32 as FloatLike>::sqrt(cxx * cyy));
    finish(wq, wr, sj, (ndx * 0.5 + 0.5) * wf, (ndy * 0.5 + 0.5) * hf, cxx, cyy, rho_raw)
  };
  // the 4 lateral sigma points of the slice, J3-weighted moments (moments about the first point's
  // position: no f32 cancellation for a thin piece far from the origin)
  let edge_point = |q: usize, wr: f32| -> VolumeSmallPoint {
    let wq = node(q).1;
    let xq = xq_of(q);
    let (bx, by, bw) = (dot3(r0, xq) + m(0, 3), dot3(r1, xq) + m(1, 3), dot3(r3, xq) + m(3, 3));
    let (mut tj, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut ox, mut oy, mut have_o) = (0.0f32, 0.0f32, false);
    #[cfg(all(test, feature = "std"))]
    let mut w_min = f32::INFINITY;
    for j in 0..4u32 {
      let s = if j & 1 == 0 { -DUST_UT2_NODE } else { DUST_UT2_NODE };
      let (dx_, dy_, dw_) = if j < 2 { (x1, y1, w1) } else { (x2, y2, w2) };
      let w = bw + s * dw_;
      #[cfg(all(test, feature = "std"))]
      {
        w_min = w_min.min(w.abs());
      }
      if !(w > 0.0) {
        continue;
      }
      let iw = 1.0 / w;
      let ndx = (bx + s * dx_) * iw;
      let ndy = (by + s * dy_) * iw;
      let c = cross3(sub3(r0, scale3(r3, ndx)), sub3(r1, scale3(r3, ndy)));
      let j3 = 0.25 * (0.25 * wf * hf * <f32 as FloatLike>::sqrt(dot3(c, c)) * (iw * iw));
      let (sxp, syp) = ((ndx * 0.5 + 0.5) * wf, (ndy * 0.5 + 0.5) * hf);
      if !have_o {
        (ox, oy, have_o) = (sxp, syp, true);
      }
      let (px, py) = (sxp - ox, syp - oy);
      tj += j3;
      sx += j3 * px;
      sy += j3 * py;
      sxx += j3 * px * px;
      syy += j3 * py * py;
      sxy += j3 * px * py;
    }
    #[cfg(all(test, feature = "std"))]
    {
      let mut v = cond.get();
      v[q] = (r3n * <f32 as FloatLike>::sqrt(dot3(xq, xq)) + m(3, 3).abs() + DUST_UT2_NODE * w1.abs().max(w2.abs())) / w_min;
      cond.set(v);
    }
    if !(tj > 0.0) {
      return VolumeSmallPoint::default();
    }
    let itj = 1.0 / tj;
    let (dx, dy) = (sx * itj, sy * itj);
    let cxx = (sxx * itj - dx * dx).max(1e-6);
    let cyy = (syy * itj - dy * dy).max(1e-6);
    let rho_raw = (sxy * itj - dx * dy) * (1.0 / <f32 as FloatLike>::sqrt(cxx * cyy));
    finish(wq, wr, tj, ox + dx, oy + dy, cxx, cyy, rho_raw)
  };
  // inside the frame one projection a slice; when the centre slice's 3σ crosses the frame's
  // edge (or it has none), the 4 lateral sigma points a slice (the perspective's skew decides
  // what the edge cuts): step D3, the test on the light, not on the box
  // round 3: every slice's weight, then (fast) every slice by one projection — the GPU's one
  // straight-line block of independent chains — then the decision on the centre's as before; an
  // edge piece redraws every slice by sigma points (its fast ones discarded), the weights shared
  let n = if five { 5 } else { 3 };
  let fast = k1c_d(DUST_K1C_D_FAST_SLICES);
  let mut wr = [1.0f32; 5];
  for q in 0..n {
    wr[q] = weight_at(xq_of(q));
  }
  let mut edge = !fast;
  if fast {
    for q in 0..n {
      out[q] = fast_point(q, wr[q]);
    }
    let c = &out[0];
    if c.m > 0.0 {
      let (ex, ey) = (3.0 / (c.ix * core::f32::consts::SQRT_2), 3.0 / (c.iy * core::f32::consts::SQRT_2));
      edge = c.mx - ex < 0.0 || c.mx + ex > wf || c.my - ey < 0.0 || c.my + ey > hf;
      #[cfg(any(test, feature = "std"))]
      {
        // D3's margin: the 3σ edge within 1e-3 σ of the frame's
        let (sx, sy) = (ex / 3.0, ey / 3.0);
        let margin = ((c.mx - ex).abs() / sx)
          .min((c.mx + ex - wf).abs() / sx)
          .min((c.my - ey).abs() / sy)
          .min((c.my + ey - hf).abs() / sy);
        if margin < 1e-3 {
          VOLUME_SMALL_MARGINS.with(|v| {
            let mut a = v.get();
            a[1] += 1;
            v.set(a);
          });
        }
      }
    } else {
      edge = true;
    }
  }
  // track 9A: an edge piece whose gate ([`small_demote_gate`], on the fast centre slice) is under
  // DUST_SMALL_DEMOTE_GATE × ε and whose predicted in-frame light change (the sigma points' against
  // the one projection's, mass-weighted over its slices, [`volume_small_demote_err`]) is under ε
  // keeps its one projection a slice. Strict, and not evaluated at ε = 0: D3's decision bit for
  // bit. The GLSL (the edge pass, t9a2d: dust_volume_small_edge_points) tests the gate first and
  // evaluates the predictor only past it, then the same slices' predictions in the same order,
  // stopping at the first slice
  // whose running Σ m_q e_q / Σ m is not under ε: that sum only grows (m > 0, e ≥ 0, f32 rounding
  // monotone), so its decision is this full sum's exactly (the CPU evaluates every slice whatever
  // the gate: the counters and the tests' E are the full sum's)
  let mut demoted = false;
  #[cfg(all(test, feature = "std"))]
  SMALL_DEMOTE_LAST_E.with(|c| c.set(f32::NAN));
  #[cfg(all(test, feature = "std"))]
  SMALL_DEMOTE_LAST_G.with(|c| c.set(f32::NAN));
  if edge && fast {
    let eps = small_demote_eps();
    if eps > 0.0 && out[..n].iter().all(|p| p.m > 0.0) {
      let d = [x1, x2, y1, y2, w1, w2];
      let t = [m(0, 3), m(1, 3), m(3, 3)];
      let gate = small_demote_gate(&out[0], mean, r3, t[2], w1, w2, wf, hf);
      #[cfg(all(test, feature = "std"))]
      SMALL_DEMOTE_LAST_G.with(|c| c.set(gate));
      let (mut acc, mut ms, mut ok) = (0.0f32, 0.0f32, true);
      for q in 0..n {
        let e = volume_small_demote_err(xq_of(q), d, r0, r1, r3, t, wf, hf);
        ok &= e >= 0.0;
        acc += out[q].m * e;
        ms += out[q].m;
      }
      if ok {
        let e = acc / ms;
        demoted = gate < DUST_SMALL_DEMOTE_GATE * eps && e < eps;
        #[cfg(all(test, feature = "std"))]
        SMALL_DEMOTE_LAST_E.with(|c| c.set(e));
        #[cfg(any(test, feature = "std"))]
        VOLUME_SMALL_MARGINS.with(|v| {
          let mut a = v.get();
          a[0] += demoted as u64;
          // (t9a2c: or its gate within 10⁻³ of its threshold)
          let ge = DUST_SMALL_DEMOTE_GATE * eps;
          a[2] += ((e - eps).abs() <= 1e-3 * eps || (gate - ge).abs() <= 1e-3 * ge) as u64;
          v.set(a);
        });
      }
    }
  }
  // t9a2: a demoted piece's slices carry their predicted second-order terms (the sigma points'
  // moments to second order, [`volume_small_demote_fix`]); the GLSL applies them as it predicts each
  // slice (dust_small_demote_slice), on the same terms
  if demoted {
    let d = [x1, x2, y1, y2, w1, w2];
    let t = [m(0, 3), m(1, 3), m(3, 3)];
    for q in 0..n {
      out[q] = volume_small_demote_fix(out[q], xq_of(q), d, r0, r1, r3, t, wf, hf);
    }
  }
  if edge && !demoted {
    for q in 0..n {
      out[q] = edge_point(q, wr[q]);
    }
  }
  #[cfg(all(test, feature = "std"))]
  SMALL_POINT_COND.with(|c| c.set(cond.get()));
  #[cfg(any(test, feature = "std"))]
  VOLUME_SMALL_PATHS.with(|c| {
    let mut v = c.get();
    v[(edge && !demoted) as usize] += 1;
    c.set(v);
  });
  out
}

/// The small points as F10 built them (before cost track 2's factoring), kept as the reference of
/// `small_points_tests::factored_small_points_match_the_f10_build`: the same light to f32
/// rounding, the same edge decision.
#[cfg(all(test, feature = "std"))]
#[allow(clippy::too_many_arguments)]
fn volume_small_points_ref(
  pc: &VolumePiece,
  eye: [f32; 3],
  mvp: &[f32; 16],
  wf: f32,
  hf: f32,
  phot: &Photometry,
  mean_ref: [f32; 3],
  renorm: f32,
) -> [VolumeSmallPoint; 5] {
  let mut out = [VolumeSmallPoint::default(); 5];
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let l = &pc.l;
  // eye − mean
  let lq = l_mul(l, pc.qw);
  let mean = sub3(eye, lq);
  let dn = <f32 as FloatLike>::sqrt(dot3(lq, lq));
  if !(dn > 0.0) {
    return out;
  }
  let e = scale3(lq, -1.0 / dn);
  let lte = lt_mul(l, e);
  let se = <f32 as FloatLike>::sqrt(dot3(lte, lte));
  if !(se > 0.0) {
    return out;
  }
  // the depth axis in the whitened space and two unit axes across it (Duff et al. 2017): the
  // depth points move along `g = L ζ`, the slice spreads along `L b₁`, `L b₂`
  let z = scale3(lte, 1.0 / se);
  let g = l_mul(l, z);
  let sgn = if z[2] >= 0.0 { 1.0f32 } else { -1.0 };
  let ia = -1.0 / (sgn + z[2]);
  let ib = z[0] * z[1] * ia;
  // bounded at 3σ (step F2): the spread narrowed to the 3σ ball's second moment
  let sb = <f32 as FloatLike>::sqrt(DUST_BOUND_VAR);
  let g = scale3(g, sb);
  let lat1 = scale3(l_mul(l, [1.0 + sgn * z[0] * z[0] * ia, sgn * ib, -sgn * z[0]]), sb);
  let lat2 = scale3(l_mul(l, [ib, sgn + z[1] * z[1] * ia, -z[1]]), sb);
  let s = &phot.support;
  let kept = if has_support(s) {
    let n = [s[0], s[1], s[2]];
    let ltn = lt_mul(l, n);
    let sn = <f32 as FloatLike>::sqrt(dot3(ltn, ltn));
    let mn = dot3(n, mean) - s[3];
    if sn > 0.0 {
      0.5 * (1.0 + erf(mn / (sn * core::f32::consts::SQRT_2)))
    } else if mn >= 0.0 {
      1.0
    } else {
      0.0
    }
  } else {
    1.0
  };
  let mass = 2.0 * core::f32::consts::PI * pc.k * (l[0] * l[3] * l[5]) * renorm * kept;
  // the radiance weight once, at the mean (step D: a small piece spans a small range of phase
  // angles; three weights a piece were a third of its cost)
  let one_weight = k1c_d(DUST_K1C_D_ONE_WEIGHT);
  let w_rad = if phot.albedo > 0.0 && one_weight {
    let d = sub3(mean, eye);
    let u = scale3(d, 1.0 / <f32 as FloatLike>::sqrt(dot3(d, d)));
    volume_weight(phot, mean_ref, mean, u)
  } else {
    1.0
  };
  // one depth point: through its slice's 4 lateral sigma points (`edge`) or one projection with
  // its screen Jacobian's lateral covariance (step D: 3 projections a piece instead of 12)
  let mass_w = mass * w_rad;
  // the depth quadrature: 3 Gauss–Hermite points, or 5 for a piece deep along the view (its
  // bounded depth σ over its distance above DUST_SMALL_DEPTH5_EPS: the 1/d² of a 3-point rule
  // drew a chord seen along its length 5 % short, `bounded_chords_match_the_bounded_monte_carlo`)
  let five = k1c_d(DUST_K1C_D_GH5) && se * <f32 as FloatLike>::sqrt(DUST_BOUND_VAR) > DUST_SMALL_DEPTH5_EPS * dn;
  let point = |q: usize, edge: bool| -> VolumeSmallPoint {
    let (off, wq) = if five {
      match q {
        0 => (0.0, DUST_GH5_W0),
        1 => (-DUST_GH5_N1, DUST_GH5_W1),
        2 => (DUST_GH5_N1, DUST_GH5_W1),
        3 => (-DUST_GH5_N2, DUST_GH5_W2),
        _ => (DUST_GH5_N2, DUST_GH5_W2),
      }
    } else {
      match q {
        0 => (0.0, 2.0 / 3.0),
        1 => (-DUST_GH3_NODE, 1.0 / 6.0),
        _ => (DUST_GH3_NODE, 1.0 / 6.0),
      }
    };
    let xq = add3(mean, scale3(g, off));
    let (sj, mx, my, cxx, cyy, rho_raw);
    if edge {
      // (moments about the first point's position: no f32 cancellation for a thin piece far from
      // the origin)
      let (mut tj, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
      let (mut ox, mut oy, mut have_o) = (0.0f32, 0.0f32, false);
      for j in 0..4u32 {
        let d = match j {
          0 => scale3(lat1, -DUST_UT2_NODE),
          1 => scale3(lat1, DUST_UT2_NODE),
          2 => scale3(lat2, -DUST_UT2_NODE),
          _ => scale3(lat2, DUST_UT2_NODE),
        };
        let x = add3(xq, d);
        let w = dot3(r3, x) + m(3, 3);
        if !(w > 0.0) {
          continue;
        }
        let ndx = (dot3(r0, x) + m(0, 3)) / w;
        let ndy = (dot3(r1, x) + m(1, 3)) / w;
        let c = cross3(sub3(r0, scale3(r3, ndx)), sub3(r1, scale3(r3, ndy)));
        let j3 = 0.25 * (0.25 * wf * hf * <f32 as FloatLike>::sqrt(dot3(c, c)) / (w * w));
        let (sxp, syp) = ((ndx * 0.5 + 0.5) * wf, (ndy * 0.5 + 0.5) * hf);
        if !have_o {
          (ox, oy, have_o) = (sxp, syp, true);
        }
        let (px, py) = (sxp - ox, syp - oy);
        tj += j3;
        sx += j3 * px;
        sy += j3 * py;
        sxx += j3 * px * px;
        syy += j3 * py * py;
        sxy += j3 * px * py;
      }
      if !(tj > 0.0) {
        return VolumeSmallPoint::default();
      }
      let (dx, dy) = (sx / tj, sy / tj);
      sj = tj;
      mx = ox + dx;
      my = oy + dy;
      cxx = (sxx / tj - dx * dx).max(1e-6);
      cyy = (syy / tj - dy * dy).max(1e-6);
      rho_raw = (sxy / tj - dx * dy) / <f32 as FloatLike>::sqrt(cxx * cyy);
    } else {
      let w = dot3(r3, xq) + m(3, 3);
      if !(w > 0.0) {
        return VolumeSmallPoint::default();
      }
      let ndx = (dot3(r0, xq) + m(0, 3)) / w;
      let ndy = (dot3(r1, xq) + m(1, 3)) / w;
      let (a, b) = (sub3(r0, scale3(r3, ndx)), sub3(r1, scale3(r3, ndy)));
      let c = cross3(a, b);
      sj = 0.25 * wf * hf * <f32 as FloatLike>::sqrt(dot3(c, c)) / (w * w);
      let (px, py) = (scale3(a, 0.5 * wf / w), scale3(b, 0.5 * hf / w));
      let (a1, a2, b1, b2) = (dot3(px, lat1), dot3(px, lat2), dot3(py, lat1), dot3(py, lat2));
      mx = (ndx * 0.5 + 0.5) * wf;
      my = (ndy * 0.5 + 0.5) * hf;
      cxx = (a1 * a1 + a2 * a2).max(1e-6);
      cyy = (b1 * b1 + b2 * b2).max(1e-6);
      rho_raw = (a1 * b1 + a2 * b2) / <f32 as FloatLike>::sqrt(cxx * cyy);
    }
    let rho = if rho_raw.is_finite() { rho_raw.clamp(-DUST_SMALL_RHO_MAX, DUST_SMALL_RHO_MAX) } else { 0.0 };
    let mut wgt = wq * mass_w * sj;
    if phot.albedo > 0.0 && !one_weight {
      let d = sub3(xq, eye);
      let u = scale3(d, 1.0 / <f32 as FloatLike>::sqrt(dot3(d, d)));
      wgt *= volume_weight(phot, mean_ref, xq, u);
    }
    if !(wgt > 0.0) || !wgt.is_finite() {
      return VolumeSmallPoint::default();
    }
    VolumeSmallPoint {
      mx,
      my,
      ix: 1.0 / (<f32 as FloatLike>::sqrt(cxx) * core::f32::consts::SQRT_2),
      iy: 1.0 / (<f32 as FloatLike>::sqrt(cyy) * core::f32::consts::SQRT_2),
      rho,
      m: wgt,
    }
  };
  // inside the frame one projection a slice; when the centre slice's 3σ crosses the frame's
  // edge (or it has none), the 4 lateral sigma points a slice (the perspective's skew decides
  // what the edge cuts): step D3, the test on the light, not on the box
  let fast = k1c_d(DUST_K1C_D_FAST_SLICES);
  let mut edge = !fast;
  if fast {
    out = [point(0, false), point(1, false), point(2, false), VolumeSmallPoint::default(), VolumeSmallPoint::default()];
    if five {
      out[3] = point(3, false);
      out[4] = point(4, false);
    }
    let c = &out[0];
    if c.m > 0.0 {
      let (ex, ey) = (3.0 / (c.ix * core::f32::consts::SQRT_2), 3.0 / (c.iy * core::f32::consts::SQRT_2));
      edge = c.mx - ex < 0.0 || c.mx + ex > wf || c.my - ey < 0.0 || c.my + ey > hf;
    } else {
      edge = true;
    }
  }
  if edge {
    out = [point(0, true), point(1, true), point(2, true), VolumeSmallPoint::default(), VolumeSmallPoint::default()];
    if five {
      out[3] = point(3, true);
      out[4] = point(4, true);
    }
  }
  #[cfg(any(test, feature = "std"))]
  VOLUME_SMALL_PATHS.with(|c| {
    let mut v = c.get();
    v[edge as usize] += 1;
    c.set(v);
  });
  out
}


#[cfg(all(test, feature = "std"))]
mod small_points_tests {
  use super::*;

  /// The factored small points (cost track 2: the sigma points and the Jacobian from the lateral
  /// axes' six dots with the projection's rows, the weights ahead of the projections, the centre
  /// weight shared by both paths) against the F10 build ([`volume_small_points_ref`]): 12 000
  /// random small pieces a mask — σ/d from 10⁻³ to 0.3, anisotropic and rotated, their means
  /// scattered over and around the frame (a fifth to either side) so most cross an edge, with and
  /// without a support plane, an eye at 200 m and one at 5·10⁶ m from the origin (the 5 000 km
  /// magnitudes) — under the production mask, the lab's one weight, the 5-point rule and the
  /// sigma points everywhere. Each piece takes the same path in both (a flip is allowed only for a
  /// centre slice whose 3σ edge is within 10⁻³ σ of the frame's). Each point's mass, position,
  /// spreads and correlation agree within 10⁻⁴ relative plus the f32 resolution of the scene's
  /// coordinates at the piece's σ (8 ε |x| / σ) plus its w's resolution (4 ε × the conditioning the
  /// build reports, `SMALL_POINT_COND`) carried through n / w into px and 1 / w² into J3: the
  /// builds round w differently, which matters only near the eye plane (the 5-point rule's node 3).
  /// A point whose ±4σ cover misses the frame is compared by its mass only (it deposits on no
  /// in-frame texel); one with a projection within ten resolutions of w = 0 (the builds may drop
  /// different sigma points) by its cover's in / out only. Prints the counts and the worst
  /// differences / tolerance per mask.
  #[test]
  fn factored_small_points_match_the_f10_build() {
    let (w, h) = (480u32, 300u32);
    let (wf, hf) = (w as f32, h as f32);
    let tan_h = (30.0f32).to_radians().tan();
    let mut state = 0x5eed_5a11_u64;
    let mut rnd = || -> f32 {
      state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      ((state >> 40) as f32) * (1.0 / 16_777_216.0)
    };
    let masks = [
      ("production", DUST_K1C_D_PRODUCTION),
      ("one weight", DUST_K1C_D_PRODUCTION | DUST_K1C_D_ONE_WEIGHT),
      ("5-point depth rule", DUST_K1C_D_PRODUCTION | DUST_K1C_D_GH5),
      ("sigma points everywhere", DUST_K1C_D_COVER),
    ];
    let mut failures = std::vec::Vec::new();
    // (track 9A's demotion off: F10 had none)
    set_small_demote_eps_test(Some(0.0));
    for (name, mask) in masks {
      set_k1c_d_test(Some(mask));
      let (mut n_pieces, mut paths_old, mut paths_new, mut flips, mut worst) = (0u32, [0u64; 2], [0u64; 2], 0u32, [0.0f64; 6]);
      let (mut n_points, mut n_off, mut n_straddle, mut n_cover_edge) = (0u32, 0u32, 0u32, 0u32);
      for case in 0..12_000u32 {
        let far = case % 2 == 1;
        let ez = if far { 5.0e6f32 } else { 200.0 };
        let mut mvp = [0.0f32; 16];
        mvp[0] = hf / (wf * tan_h);
        mvp[5] = 1.0 / tan_h;
        mvp[10] = -1e-3;
        mvp[11] = -1.0;
        mvp[15] = ez;
        let eye = [0.0f32, 0.0, ez];
        // the mean on the ray through a pixel over and around the frame, at distance d
        let (px, py) = ((-0.2 + 1.4 * rnd()) * wf, (-0.2 + 1.4 * rnd()) * hf);
        let d = if far { 2.0e3 * (1.0e3f32).powf(rnd()) } else { 5.0 * (20.0f32).powf(rnd()) };
        let v = [(2.0 * px / wf - 1.0) * tan_h * wf / hf, (2.0 * py / hf - 1.0) * tan_h, -1.0];
        let vn = (v[0] * v[0] + v[1] * v[1] + 1.0).sqrt();
        let mean = [v[0] / vn * d, v[1] / vn * d, ez + v[2] / vn * d];
        // σ/d from 1e-3 to 0.3, anisotropic up to 4:1, rotated about two axes
        let sd = 1.0e-3 * (300.0f32).powf(rnd());
        let sig = [sd * d, sd * d * (0.25 + 0.75 * rnd()), sd * d * (0.25 + 0.75 * rnd())];
        let (a, b) = (rnd() * 6.283, rnd() * 6.283);
        let (ca, sa, cb, sb) = (a.cos(), a.sin(), b.cos(), b.sin());
        let ry = [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]];
        let rx = [[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]];
        let mut r = [[0.0f32; 3]; 3];
        for i in 0..3 {
          for j in 0..3 {
            for k in 0..3 {
              r[i][j] += rx[i][k] * ry[k][j];
            }
          }
        }
        let mut cov = [[0.0f32; 3]; 3];
        for i in 0..3 {
          for j in 0..3 {
            for k in 0..3 {
              cov[i][j] += r[i][k] * sig[k] * sig[k] * r[j][k];
            }
          }
        }
        let l = cholesky3(&[cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2]]);
        let pc = VolumePiece {
          l,
          qw: cholesky_solve3(&l, sub3(eye, mean)),
          k: 1.0 / (2.0 * core::f32::consts::PI * l[0] * l[3] * l[5]),
          ..VolumePiece::default()
        };
        let support = if case % 3 == 0 {
          // a plane through the piece, its normal along the piece's first axis
          let n = [r[0][0], r[1][0], r[2][0]];
          [n[0], n[1], n[2], dot3(n, mean) - sig[0] * (rnd() - 0.5)]
        } else {
          [0.0; 4]
        };
        let sun = [0.3f32, 0.5, 0.812_404];
        let phot = Photometry { albedo: 0.1, sun, r_h_au: 1.0, support, origin: f32::NAN };
        let _ = take_volume_small_paths();
        let old = volume_small_points_ref(&pc, eye, &mvp, wf, hf, &phot, mean, 1.0);
        let po = take_volume_small_paths();
        let new = volume_small_points(&pc, eye, &mvp, wf, hf, &phot, mean, 1.0);
        let pn = take_volume_small_paths();
        let kap = SMALL_POINT_COND.with(|c| c.get());
        n_pieces += 1;
        for k in 0..2 {
          paths_old[k] += po[k];
          paths_new[k] += pn[k];
        }
        if po != pn {
          // a flip: only where the centre slice's 3σ meets the frame's edge within 1e-3 σ
          let c = &old[0];
          let (sx, sy) = (1.0 / (c.ix * core::f32::consts::SQRT_2), 1.0 / (c.iy * core::f32::consts::SQRT_2));
          let margin = [
            (c.mx - 3.0 * sx).abs() / sx,
            (c.mx + 3.0 * sx - wf).abs() / sx,
            (c.my - 3.0 * sy).abs() / sy,
            (c.my + 3.0 * sy - hf).abs() / sy,
          ]
          .into_iter()
          .fold(f32::INFINITY, f32::min);
          flips += 1;
          if !(margin < 1e-3) {
            failures.push(std::format!("{name}: case {case}: the path flipped {po:?} → {pn:?} with the edge {margin:.2e} σ away"));
          }
          continue;
        }
        let xmag = dot3(eye, eye).sqrt().max(dot3(mean, mean).sqrt());
        let tol = 1e-4 + 8.0 * f32::EPSILON as f64 * (xmag / sig[2].min(sig[1]).min(sig[0])) as f64;
        for q in 0..5 {
          let (o, n) = (&old[q], &new[q]);
          if (o.m > 0.0) != (n.m > 0.0) {
            failures.push(std::format!("{name}: case {case} point {q}: m {} vs {}", o.m, n.m));
            continue;
          }
          if !(o.m > 0.0) {
            continue;
          }
          n_points += 1;
          // the f32 resolution of the point's w (relative; ~4 roundings of its dot and offset) — the
          // two builds round w differently, and ndx = n / w, J3 ∝ 1 / w² amplify it near the eye
          // plane (the 5-point rule's node 3 at −2.86 σ_depth sits there)
          let dw = 4.0 * f32::EPSILON as f64 * kap[q] as f64;
          // a sigma point's w within ten resolutions of 0: the builds may disagree on its sign (a
          // projection dropped), so the moments may jump; only in / out is compared
          let straddle = !(dw < 0.1);
          // the frame cover (±DUST_SMALL_COVER σ): a point whose cover misses the frame deposits on
          // no in-frame texel of its own
          let covers = |p: &VolumeSmallPoint| {
            let (sx, sy) = (1.0 / (p.ix * core::f32::consts::SQRT_2), 1.0 / (p.iy * core::f32::consts::SQRT_2));
            p.mx + DUST_SMALL_COVER * sx >= 0.0
              && p.mx - DUST_SMALL_COVER * sx <= wf
              && p.my + DUST_SMALL_COVER * sy >= 0.0
              && p.my - DUST_SMALL_COVER * sy <= hf
          };
          let (on_o, on_n) = (covers(o), covers(n));
          if !on_o {
            n_off += 1;
          }
          if straddle {
            n_straddle += 1;
          }
          if straddle {
            continue;
          }
          let rel = |a: f32, b: f32| ((a - b).abs() / a.abs().max(b.abs()).max(1e-30)) as f64;
          // the tolerances: the scene's coordinates at the piece's σ (tol), plus w's resolution
          // carried through ndx = n / w into px (|p − centre| δw) in σ, and into J3 ∝ 1 / w² (m);
          // ix, iy and ρ carry the position's term
          let px_x = (o.mx - 0.5 * wf).abs() as f64 * (o.ix * core::f32::consts::SQRT_2) as f64;
          let px_y = (o.my - 0.5 * hf).abs() as f64 * (o.iy * core::f32::consts::SQRT_2) as f64;
          let t_m = tol + 4.0 * dw;
          let t_x = tol + 2.0 * dw * (1.0 + px_x);
          let t_y = tol + 2.0 * dw * (1.0 + px_y);
          let t_s = tol + 2.0 * dw * (1.0 + px_x.max(px_y));
          let tols = [t_m, t_x, t_y, t_s, t_s, t_s];
          // the cover's in / out at the frame edge (like a path flip): allowed where the old
          // cover's edge is within 1e-3 σ of the frame's plus what the tolerances let it move (4 t_s
          // for the 4σ reach, max(t_x, t_y) for the centre); the point is then compared on the
          // on-frame tolerances
          let mut on = on_o;
          if on_o != on_n {
            let (sx, sy) = (1.0 / (o.ix * core::f32::consts::SQRT_2), 1.0 / (o.iy * core::f32::consts::SQRT_2));
            let margin = [
              (o.mx + DUST_SMALL_COVER * sx).abs() / sx,
              (o.mx - DUST_SMALL_COVER * sx - wf).abs() / sx,
              (o.my + DUST_SMALL_COVER * sy).abs() / sy,
              (o.my - DUST_SMALL_COVER * sy - hf).abs() / sy,
            ]
            .into_iter()
            .fold(f32::INFINITY, f32::min) as f64;
            let allow = 1e-3 + 4.0 * t_s + t_x.max(t_y);
            if !(margin <= allow) {
              failures.push(std::format!(
                "{name}: case {case} point {q}: frame cover {on_o} vs {on_n} with its edge {margin:.2e} σ from the frame's (allowed {allow:.2e}): old {o:?} new {n:?}"
              ));
              continue;
            }
            n_cover_edge += 1;
            on = true;
          }
          // off the frame only the mass is compared (the moments of a point that touches no
          // in-frame texel decide nothing)
          let checked = if on { 6 } else { 1 };
          let d = [
            rel(o.m, n.m),
            ((o.mx - n.mx) * o.ix * core::f32::consts::SQRT_2).abs() as f64,
            ((o.my - n.my) * o.iy * core::f32::consts::SQRT_2).abs() as f64,
            rel(o.ix, n.ix),
            rel(o.iy, n.iy),
            (o.rho - n.rho).abs() as f64,
          ];
          for k in 0..checked {
            worst[k] = worst[k].max(d[k] / tols[k]);
            if d[k] > tols[k] {
              failures.push(std::format!(
                "{name}: case {case} point {q}: {} differs by {:.3e} (tolerance {:.3e}, w resolution {dw:.2e}, on frame {on_o}): old {o:?} new {n:?}",
                ["m", "mx (σ)", "my (σ)", "ix", "iy", "rho"][k],
                d[k],
                tols[k]
              ));
            }
          }
        }
      }
      std::println!(
        "[small points] {name}: {n_pieces} pieces, paths (one projection, sigma points) F10 {paths_old:?} factored {paths_new:?}, {flips} flips at the edge; {n_points} points, {n_off} with their cover off the frame (mass only), {n_straddle} with a projection within 10 f32 resolutions of the eye plane (in / out only), {n_cover_edge} cover decisions at the frame edge; worst difference / tolerance: m {:.3}, mx {:.3}, my {:.3}, ix {:.3}, iy {:.3}, rho {:.3}",
        worst[0], worst[1], worst[2], worst[3], worst[4], worst[5]
      );
      if flips * 1000 > n_pieces {
        failures.push(std::format!("{name}: {flips} of {n_pieces} pieces changed path"));
      }
    }
    set_k1c_d_test(None);
    set_small_demote_eps_test(None);
    let shown: std::vec::Vec<_> = failures.iter().take(20).cloned().collect();
    assert!(failures.is_empty(), "{} failures, the first: {shown:#?}", failures.len());
  }

  /// Acklam's inverse of the normal CDF (relative error ~10⁻⁹; the calibration's QMC truth)
  fn ndtri(p: f64) -> f64 {
    const A: [f64; 6] = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    const B: [f64; 5] = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const C: [f64; 6] = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    if p < 0.02425 {
      tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
      -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
      let q = p - 0.5;
      let r = q * q;
      (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
  }

  /// the radical inverse of `i` in base `b` (Halton)
  fn halton(mut i: u32, b: u32) -> f64 {
    let (mut f, mut r) = (1.0f64, 0.0f64);
    while i > 0 {
      f /= b as f64;
      r += f * (i % b) as f64;
      i /= b;
    }
    r
  }

  /// Track 9A's calibration and soundness test (the edge-piece demotion, [`volume_small_demote_err`]).
  /// The pieces of `factored_small_points_match_the_f10_build` (12 000: σ/d 10⁻³..0.3, anisotropic
  /// and rotated, means over and a fifth around the 480 × 300 frame, eyes at 200 m and 5·10⁶ m)
  /// without albedo or support (both paths share the weights and the kept mass: the light compared
  /// is the geometry's); each piece D3 sends to the sigma points and whose slices all have a
  /// prediction is drawn by both paths (the ε hook at +∞ keeps it on one projection — the demoted
  /// path, with its second-order terms, [`volume_small_demote_fix`] — at 0 on the
  /// sigma points), its in-frame light `L` being the frame's rectangle integral of its points
  /// (`volume_scatter`'s small branch). Its truth `T` is the bounded Gaussian's (|z| ≤ 3, the 3σ
  /// ball the paths' spread is matched to) light in the frame, `∫ J3 · 1[in frame]`, by 2 × 8 192
  /// Halton points (bases 2, 3, 5, two Cranley–Patterson shifts: their half difference is the
  /// QMC error printed), in f64. Asserts:
  /// (a) soundness at every ε of the grid: a piece with E < ε moves its in-frame light by at most
  ///     ε of its mass (+ 10⁻⁶ of it, the f32 floor);
  /// (b) ε = 0 evaluates no predictor and demotes nothing (D3's decisions bit for bit);
  /// (c) ε_cal, the largest grid ε that passes (a) and whose demoted pieces' extra error against
  ///     the truth, Σ (|L_fast − T| − |L_sigma − T|), is at most 2·10⁻³ of their Σ T (the gate's
  ///     0.5 % over 2.5), is ≥ [`DUST_SMALL_DEMOTE_EPS`].
  /// A piece is demoted at ε as [`volume_small_points`] demotes it: E < ε and its gate
  /// ([`small_demote_gate`]) under [`DUST_SMALL_DEMOTE_GATE`] × ε (t9a2c); the pieces with E < ε
  /// that the gate keeps on the sigma points are printed ("the gate kept … of the E < ε").
  /// Prints per ε the demoted share of the edge pieces (near / far eye), the worst |ΔL| / (ε M),
  /// the extra error, the signed change of the edge pieces' total in-frame light (the expected move
  /// of an image's small edge light) and the QMC error.
  #[test]
  fn small_demotion_calibration() {
    let (w, h) = (480u32, 300u32);
    let (wf, hf) = (w as f32, h as f32);
    let tan_h = (30.0f32).to_radians().tan();
    let mut state = 0x9a_ca1b_5a11_u64;
    let mut rnd = || -> f32 {
      state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      ((state >> 40) as f32) * (1.0 / 16_777_216.0)
    };
    const GRID: [f32; 5] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2];
    const NQ: u32 = 8192;
    let hal: std::vec::Vec<[f64; 3]> = (1..=NQ).map(|i| [halton(i, 2), halton(i, 3), halton(i, 5)]).collect();
    let in_frame = |pts: &[VolumeSmallPoint; 5]| -> f64 {
      let mut acc = 0.0f64;
      for q in pts.iter().filter(|q| q.m > 0.0) {
        let (uxa, uxb) = ((0.0 - q.mx) * q.ix, (wf - q.mx) * q.ix);
        let (uya, uyb) = ((0.0 - q.my) * q.iy, (hf - q.my) * q.iy);
        let fx = erf(uxb) - erf(uxa);
        let fy = erf(uyb) - erf(uya);
        let gx = <f32 as FloatLike>::exp(-uxb * uxb) - <f32 as FloatLike>::exp(-uxa * uxa);
        let gy = <f32 as FloatLike>::exp(-uyb * uyb) - <f32 as FloatLike>::exp(-uya * uya);
        acc += (q.m * (0.25 * fx * fy + q.rho * DUST_INV_2PI * gx * gy)) as f64;
      }
      acc
    };
    set_k1c_d_test(Some(DUST_K1C_D_PRODUCTION));
    let (mut n_edge, mut n_pred, mut n_none) = (0u32, 0u32, 0u32);
    let mut n_dem = [[0u32; 2]; 5];
    let mut worst = [0.0f64; 5];
    let (mut excess, mut tsum, mut dsum, mut qerr) = ([0.0f64; 5], [0.0f64; 5], [0.0f64; 5], [0.0f64; 5]);
    let mut msum = [0.0f64; 5];
    let mut n_gate = [[0u32; 2]; 5];
    let (mut edge_light, mut ratio_max) = (0.0f64, 0.0f64);
    let mut failures = std::vec::Vec::new();
    for case in 0..12_000u32 {
      let far = case % 2 == 1;
      let ez = if far { 5.0e6f32 } else { 200.0 };
      let mut mvp = [0.0f32; 16];
      mvp[0] = hf / (wf * tan_h);
      mvp[5] = 1.0 / tan_h;
      mvp[10] = -1e-3;
      mvp[11] = -1.0;
      mvp[15] = ez;
      let eye = [0.0f32, 0.0, ez];
      let (px, py) = ((-0.2 + 1.4 * rnd()) * wf, (-0.2 + 1.4 * rnd()) * hf);
      let d = if far { 2.0e3 * (1.0e3f32).powf(rnd()) } else { 5.0 * (20.0f32).powf(rnd()) };
      let v = [(2.0 * px / wf - 1.0) * tan_h * wf / hf, (2.0 * py / hf - 1.0) * tan_h, -1.0];
      let vn = (v[0] * v[0] + v[1] * v[1] + 1.0).sqrt();
      let mean = [v[0] / vn * d, v[1] / vn * d, ez + v[2] / vn * d];
      let sd = 1.0e-3 * (300.0f32).powf(rnd());
      let sig = [sd * d, sd * d * (0.25 + 0.75 * rnd()), sd * d * (0.25 + 0.75 * rnd())];
      let (a, b) = (rnd() * 6.283, rnd() * 6.283);
      let (ca, sa, cb, sb) = (a.cos(), a.sin(), b.cos(), b.sin());
      let ry = [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]];
      let rx = [[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]];
      let mut r = [[0.0f32; 3]; 3];
      for i in 0..3 {
        for j in 0..3 {
          for k in 0..3 {
            r[i][j] += rx[i][k] * ry[k][j];
          }
        }
      }
      let mut cov = [[0.0f32; 3]; 3];
      for i in 0..3 {
        for j in 0..3 {
          for k in 0..3 {
            cov[i][j] += r[i][k] * sig[k] * sig[k] * r[j][k];
          }
        }
      }
      let l = cholesky3(&[cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2]]);
      let pc = VolumePiece {
        l,
        qw: cholesky_solve3(&l, sub3(eye, mean)),
        k: 1.0 / (2.0 * core::f32::consts::PI * l[0] * l[3] * l[5]),
        ..VolumePiece::default()
      };
      let phot = Photometry { albedo: 0.0, sun: [0.3, 0.5, 0.812_404], r_h_au: 1.0, support: [0.0; 4], origin: f32::NAN };
      // (b): ε = 0 evaluates nothing, demotes nothing
      set_small_demote_eps_test(Some(0.0));
      let _ = take_volume_small_paths();
      let _ = take_volume_small_margins();
      let sigma = volume_small_points(&pc, eye, &mvp, wf, hf, &phot, mean, 1.0);
      let paths0 = take_volume_small_paths();
      let marg0 = take_volume_small_margins();
      let e_at_0 = SMALL_DEMOTE_LAST_E.with(|c| c.get());
      let g_at_0 = SMALL_DEMOTE_LAST_G.with(|c| c.get());
      if marg0[0] != 0 || !e_at_0.is_nan() || !g_at_0.is_nan() {
        failures.push(std::format!("case {case}: ε = 0 demoted {} (E {e_at_0})", marg0[0]));
      }
      if paths0[1] == 0 {
        continue;
      }
      n_edge += 1;
      // the forced one projection: every piece with a prediction
      set_small_demote_eps_test(Some(f32::INFINITY));
      let fast = volume_small_points(&pc, eye, &mvp, wf, hf, &phot, mean, 1.0);
      let e = SMALL_DEMOTE_LAST_E.with(|c| c.get()) as f64;
      let gq = SMALL_DEMOTE_LAST_G.with(|c| c.get());
      if e.is_nan() || !e.is_finite() {
        n_none += 1;
        continue;
      }
      n_pred += 1;
      let m_tot: f64 = fast.iter().map(|p| p.m as f64).sum();
      let (lf, ls) = (in_frame(&fast), in_frame(&sigma));
      let act = (lf - ls).abs() / m_tot;
      edge_light += ls;
      if e > 1e-7 {
        ratio_max = ratio_max.max(act / e);
      }
      // the truth, for the pieces some grid ε demotes
      let truth = if e < GRID[4] as f64 {
        let lq = l_mul(&pc.l, pc.qw);
        let mu = [(eye[0] - lq[0]) as f64, (eye[1] - lq[1]) as f64, (eye[2] - lq[2]) as f64];
        let lf64: [f64; 6] = core::array::from_fn(|k| pc.l[k] as f64);
        let (r0, r1, r3) = ([mvp[0] as f64, mvp[4] as f64, mvp[8] as f64], [mvp[1] as f64, mvp[5] as f64, mvp[9] as f64], [mvp[3] as f64, mvp[7] as f64, mvp[11] as f64]);
        let (t0, t1, t3) = (mvp[12] as f64, mvp[13] as f64, mvp[15] as f64);
        let d3 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let mut est = [0.0f64; 2];
        for est_k in est.iter_mut() {
          let shift = [rnd() as f64, rnd() as f64, rnd() as f64];
          let (mut acc, mut kept) = (0.0f64, 0u32);
          for u in &hal {
            let z: [f64; 3] = core::array::from_fn(|k| ndtri((u[k] + shift[k]).fract()));
            if z[0] * z[0] + z[1] * z[1] + z[2] * z[2] > 9.0 {
              continue;
            }
            kept += 1;
            let x = [
              mu[0] + lf64[0] * z[0],
              mu[1] + lf64[1] * z[0] + lf64[3] * z[1],
              mu[2] + lf64[2] * z[0] + lf64[4] * z[1] + lf64[5] * z[2],
            ];
            let ww = d3(r3, x) + t3;
            if !(ww > 0.0) {
              continue;
            }
            let (nx, ny) = ((d3(r0, x) + t0) / ww, (d3(r1, x) + t1) / ww);
            let (sxp, syp) = ((nx * 0.5 + 0.5) * wf as f64, (ny * 0.5 + 0.5) * hf as f64);
            if !(sxp >= 0.0 && sxp <= wf as f64 && syp >= 0.0 && syp <= hf as f64) {
              continue;
            }
            let p = [r0[0] - nx * r3[0], r0[1] - nx * r3[1], r0[2] - nx * r3[2]];
            let q = [r1[0] - ny * r3[0], r1[1] - ny * r3[1], r1[2] - ny * r3[2]];
            let c = [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
            acc += 0.25 * wf as f64 * hf as f64 * d3(c, c).sqrt() / (ww * ww);
          }
          *est_k = acc / kept.max(1) as f64;
        }
        Some((0.5 * (est[0] + est[1]), 0.5 * (est[0] - est[1]).abs()))
      } else {
        None
      };
      for (k, &eps) in GRID.iter().enumerate() {
        if !(e < eps as f64) {
          continue;
        }
        n_gate[k][1] += 1;
        // the rule's gate (volume_small_points: `gate < DUST_SMALL_DEMOTE_GATE * eps`, in f32)
        if !(gq < DUST_SMALL_DEMOTE_GATE * eps) {
          continue;
        }
        n_gate[k][0] += 1;
        n_dem[k][far as usize] += 1;
        worst[k] = worst[k].max(act / eps as f64);
        if act > eps as f64 + 1e-6 {
          failures.push(std::format!(
            "case {case}: ε {eps:.0e}: E {e:.3e} but the light moved {act:.3e} of the mass (fast {fast:?} sigma {sigma:?})"
          ));
        }
        dsum[k] += lf - ls;
        if let Some((t, qe)) = truth {
          excess[k] += (lf - t).abs() - (ls - t).abs();
          tsum[k] += t;
          msum[k] += m_tot;
          qerr[k] += qe;
        }
      }
    }
    set_k1c_d_test(None);
    set_small_demote_eps_test(None);
    std::println!(
      "[small demotion] {n_edge} edge pieces (D3) of 12 000, {n_pred} with a prediction, {n_none} without; |ΔL| / E at most {ratio_max:.3} (E > 1e-7)"
    );
    let mut eps_cal = 0.0f32;
    for (k, &eps) in GRID.iter().enumerate() {
      let nd = n_dem[k][0] + n_dem[k][1];
      let extra = excess[k] / tsum[k].max(1e-300);
      let sound = worst[k] <= 1.0 + 1e-6 / eps as f64;
      let ok = sound && extra <= 2e-3;
      if ok {
        eps_cal = eps;
      }
      std::println!(
        "[small demotion] ε {eps:.0e}: demoted {nd} ({:.1} % of the edge pieces; near {} far {}; the gate kept {} of the {} with E < ε on the sigma points), worst |ΔL| / (ε M) {:.3}, extra error vs the truth {:+.3e} of their light (QMC ±{:.1e}; their light {:.2e} of their mass), edge light moved {:+.3e}{}",
        100.0 * nd as f64 / n_edge.max(1) as f64,
        n_dem[k][0],
        n_dem[k][1],
        n_gate[k][1] - n_gate[k][0],
        n_gate[k][1],
        worst[k],
        extra,
        qerr[k] / tsum[k].max(1e-300),
        tsum[k] / msum[k].max(1e-300),
        dsum[k] / edge_light.max(1e-300),
        if ok { "" } else { " (fails the rule)" }
      );
    }
    std::println!("[small demotion] ε_cal = {eps_cal:.0e}; production DUST_SMALL_DEMOTE_EPS = {DUST_SMALL_DEMOTE_EPS:.0e}");
    let shown: std::vec::Vec<_> = failures.iter().take(20).cloned().collect();
    assert!(failures.is_empty(), "{} failures, the first: {shown:#?}", failures.len());
    assert!(
      DUST_SMALL_DEMOTE_EPS <= eps_cal,
      "the production ε {DUST_SMALL_DEMOTE_EPS:.0e} is above the calibrated {eps_cal:.0e}"
    );
  }
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
  // the packet's rounding seed, drawn first (mirror of `dust_volume_scatter`)
  let seed = next_packet_seed();
  #[cfg(any(test, feature = "std"))]
  VOLUME_CALLS.with(|c| c.set(c.get() + 1));
  let vs = VolumeStats::load(pyramid);
  let (bias, axis) = (vs.bias, vs.axis);
  let mut pieces = [VolumePiece::default(); DUST_VOLUME_PIECES_MAX as usize];
  let mut bound = VolumeBound::default();
  let Some(plan) = volume_plan_pieces(
    p,
    exposure,
    mvp,
    view,
    layout,
    bias,
    axis,
    volume_max_evals(),
    &mut pieces,
    &mut bound,
  ) else {
    #[cfg(any(test, feature = "std"))]
    if volume_trace() {
      std::println!("  [trace volume] not drawn (no projection, or no piece on screen):");
      for (k, pc) in pieces.iter().enumerate().filter(|(_, pc)| pc.l[0] > 0.0) {
        trace_piece(k, pc, view.eye, &phot.support, "-", 0, (0, -1, 0, -1), 0.0, 0.0);
      }
    }
    return 0;
  };
  let tracing = volume_trace();
  let ev = &mut pyramid[PYR_VOLUME_EVALS as usize];
  let before = *ev;
  *ev = ev.wrapping_add(plan.evals);
  let m = |r: usize, c: usize| mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let (wf, hf) = (layout.width as f32, layout.height as f32);
  let eye = view.eye;
  // the in-frame gate (mirror of `dust_volume_scatter`): past DUST_FRAME_HARD_FACTOR budgets this
  // frame, every piece at least at [`gate_level`] (the evaluations only shrink)
  let gated = frame_gates_on(pyramid) && before > DUST_FRAME_HARD_FACTOR * DUST_VOLUME_FRAME_BUDGET;
  let gate = gate_level(layout.level_count());
  // each piece at its own level (step F8, [`volume_piece_level`])
  let level_of = |pc: &VolumePiece| -> u32 {
    let l = plan.single.unwrap_or_else(|| volume_piece_level(pc, &plan.rule, layout));
    if gated { l.max(gate) } else { l }
  };
  let mut mask_bits = 0u32;
  let depth_bits = (<f32 as FloatLike>::sqrt(dot3(sub3(eye, p.mean), sub3(eye, p.mean)))
    / AU_M_F32)
    .max(0.0)
    .to_bits();
  let mut written = 0u32;
  let mut carry = 0.0f32;
  let mut peak = 0.0f32;
  // the support's cut: the packet's flux over its kept mass (the grains it holds all lie in front)
  let renorm = if has_support(&phot.support) {
    1.0 / support_mass(p, &phot.support).max(DUST_SUPPORT_MASS_MIN)
  } else {
    1.0
  };
  #[cfg(any(test, feature = "std"))]
  let piece_only = VOLUME_PIECE_ONLY.with(|c| c.get());
  #[cfg(any(test, feature = "std"))]
  VOLUME_PIECE_CLASSES.with(|c| c.set([255; DUST_VOLUME_PIECES_MAX as usize]));
  for (k, pc) in pieces.iter().take(plan.pieces as usize).enumerate() {
    #[cfg(any(test, feature = "std"))]
    if piece_only != u32::MAX && k as u32 != piece_only {
      continue;
    }
    let level = level_of(pc);
    let (off, lw, _) = layout.levels[level as usize];
    let sz = (1u32 << level) as f32;
    let area = sz * sz;
    let written_before = written;
    let (tx0, tx1, ty0, ty1) = volume_range(pc, level, layout);
    if tx1 < tx0 || ty1 < ty0 {
      continue;
    }
    // K1c: how its texels are drawn at this level (the gated one included)
    let (class, k_sub) = if plan.single.is_some() {
      // the pre-F8 hook: the class at the packet's level, a non-full piece the packet's sub-ray
      // limit would under-sample drawn small ([`volume_subray_limit`])
      let (class0, k_want) = volume_class(pc, level, layout.level_count());
      let class = if class0 == DUST_VOLUME_CLASS_SUBRAY && !pc.full && k_want > plan.subrays {
        DUST_VOLUME_CLASS_SMALL
      } else {
        class0
      };
      (class, k_want.min(plan.subrays).max(1))
    } else {
      // the class and order at its level (the gated one included) under the packet's limit and
      // the piece's order bound ([`volume_piece_class_k`])
      let (c, k, _) = volume_piece_class_k(pc, level, plan.subrays, true, layout);
      (c, k)
    };
    let (mut tr_raw, mut tr_in) = (0.0f64, 0.0f64);
    #[cfg(any(test, feature = "std"))]
    VOLUME_CLASS_COUNTS.with(|c| {
      let mut v = c.get();
      v[class as usize] += 1;
      c.set(v);
    });
    #[cfg(any(test, feature = "std"))]
    VOLUME_PIECE_CLASSES.with(|c| {
      let mut v = c.get();
      v[k] = class as u8;
      c.set(v);
    });
    let small = if class == DUST_VOLUME_CLASS_SMALL {
      volume_small_points(pc, eye, mvp, wf, hf, phot, p.mean, renorm)
    } else {
      [VolumeSmallPoint::default(); 5]
    };
    // a small piece deposits only on the texels its points' ±4σ cover (within its range; step D:
    // ≤ 3 × 3 instead of the range's ≤ 4 × 4), its segments from the first of them
    let (tx0, tx1, ty0, ty1) = if class == DUST_VOLUME_CLASS_SMALL && k1c_d(DUST_K1C_D_COVER) {
      let (mut lx, mut hx, mut ly, mut hy) = (f32::INFINITY, -f32::INFINITY, f32::INFINITY, -f32::INFINITY);
      for q in small.iter().filter(|q| q.m > 0.0) {
        let (ex, ey) = (DUST_SMALL_COVER / (q.ix * core::f32::consts::SQRT_2), DUST_SMALL_COVER / (q.iy * core::f32::consts::SQRT_2));
        lx = lx.min(q.mx - ex);
        hx = hx.max(q.mx + ex);
        ly = ly.min(q.my - ey);
        hy = hy.max(q.my + ey);
      }
      if !(lx <= hx && ly <= hy) {
        continue;
      }
      (
        fl(lx / sz).max(tx0 as f32) as i32,
        fl(hx / sz).min(tx1 as f32) as i32,
        fl(ly / sz).max(ty0 as f32) as i32,
        fl(hy / sz).min(ty1 as f32) as i32,
      )
    } else {
      (tx0, tx1, ty0, ty1)
    };
    if tx1 < tx0 || ty1 < ty0 {
      continue;
    }
    if touch_log_on() {
      touch_rect(off, lw, tx0 as i64, tx1 as i64, ty0 as i64, ty1 as i64);
    }
    // each piece's row segments carry their own dither ([`DUST_SEGMENT_TEXELS`])
    let piece_seed = dust_hash32(seed ^ k as u32);
    for ty in ty0..=ty1 {
      let yc = (ty as f32 + 0.5) * sz;
      let ny = 2.0 * yc / hf - 1.0;
      let b_row = sub3(r1, scale3(r3, ny));
      // the row's in-frame part
      let ya = (ty as f32 * sz).max(0.0);
      let yb = ((ty + 1) as f32 * sz).min(hf);
      for tx in tx0..=tx1 {
        if (tx - tx0) as u32 % DUST_SEGMENT_TEXELS == 0 {
          carry = segment_carry(piece_seed, (ty as u32) << 16 | tx as u32);
        }
        let tau_px = if class == DUST_VOLUME_CLASS_CENTRE {
          let xc = (tx as f32 + 0.5) * sz;
          let nx = 2.0 * xc / wf - 1.0;
          let a_row = sub3(r0, scale3(r3, nx));
          volume_ray_tau(pc, &bound, renorm, eye, a_row, b_row, r3, phot, p.mean)
        } else {
          // the texel's in-frame part: its mean optical depth there
          let xa = (tx as f32 * sz).max(0.0);
          let xb = ((tx + 1) as f32 * sz).min(wf);
          if !(xb > xa && yb > ya) {
            continue;
          }
          if class == DUST_VOLUME_CLASS_LINE {
            // k Gauss–Legendre lines across the band, each integrated along itself in closed form
            let kl = k_sub;
            let along_x = pc.line == 1;
            let (c, h) = if along_x { (0.5 * (ya + yb), 0.5 * (yb - ya)) } else { (0.5 * (xa + xb), 0.5 * (xb - xa)) };
            let (lo, hi) = if along_x { (xa, xb) } else { (ya, yb) };
            let mut acc = 0.0f32;
            for j in 0..kl {
              let (nj, wj) = gl_node(kl, j);
              acc += wj
                * volume_line_integral(pc, &bound, renorm, eye, r0, r1, r3, wf, hf, phot, p.mean, along_x, c + h * nj, lo, hi);
            }
            acc / (hi - lo)
          } else if class == DUST_VOLUME_CLASS_SUBRAY {
            // k × k Gauss–Legendre over the in-frame part
            let (cx, cy) = (0.5 * (xa + xb), 0.5 * (ya + yb));
            let (hx, hy) = (0.5 * (xb - xa), 0.5 * (yb - ya));
            let mut acc = 0.0f32;
            for j in 0..k_sub {
              let (nj, wj) = gl_node(k_sub, j);
              let ys = cy + hy * nj;
              let bs = sub3(r1, scale3(r3, 2.0 * ys / hf - 1.0));
              for i in 0..k_sub {
                let (ni, wi) = gl_node(k_sub, i);
                let xs = cx + hx * ni;
                let a_s = sub3(r0, scale3(r3, 2.0 * xs / wf - 1.0));
                let t = volume_ray_tau(pc, &bound, renorm, eye, a_s, bs, r3, phot, p.mean);
                if t >= 0.0 && t.is_finite() {
                  acc += (wi * wj) * t;
                }
              }
            }
            acc
          } else {
            let mut acc = 0.0f32;
            for q in &small {
              if q.m > 0.0 {
                let (uxa, uxb) = ((xa - q.mx) * q.ix, (xb - q.mx) * q.ix);
                let (uya, uyb) = ((ya - q.my) * q.iy, (yb - q.my) * q.iy);
                let fx = erf(uxb) - erf(uxa);
                let fy = erf(uyb) - erf(uya);
                // the correlation to first order: ∂Φ₂/∂ρ = φ₂, so the rectangle gains
                // ρ (φ(x_b) − φ(x_a)) (φ(y_b) − φ(y_a)) (it sums to 0 over the plane)
                let gx = <f32 as FloatLike>::exp(-uxb * uxb) - <f32 as FloatLike>::exp(-uxa * uxa);
                let gy = <f32 as FloatLike>::exp(-uyb * uyb) - <f32 as FloatLike>::exp(-uya * uya);
                acc += q.m * (0.25 * fx * fy + q.rho * DUST_INV_2PI * gx * gy);
              }
            }
            acc.max(0.0) / ((xb - xa) * (yb - ya))
          }
        };
        if !(tau_px >= 0.0) || !tau_px.is_finite() {
          continue;
        }
        if tau_px > peak {
          peak = tau_px;
        }
        if tracing {
          let xa = (tx as f32 * sz).max(0.0);
          let xb = ((tx + 1) as f32 * sz).min(wf);
          tr_raw += (tau_px * area) as f64;
          tr_in += (tau_px * (xb - xa).max(0.0) * (yb - ya).max(0.0)) as f64;
        }
        let tau = tau_px * area * inv_unit;
        // the counts of this texel, split into adds of at most DUST_COUNT_MAX_PER_ADD (step F4: a
        // bright piece on a coarse texel was clipped at one add; below the cap one add, as before)
        let tau_c = tau + carry;
        let counts = <f32 as FloatLike>::floor(tau_c + 0.5).min(DUST_COUNT_TEXEL_MAX);
        carry = tau_c - counts;
        if !(counts >= 1.0) {
          continue;
        }
        let i = (off + (ty as u32 * lw + tx as u32) * PYRAMID_TEXEL_WORDS) as usize;
        let hfr = (dust_hash32(i as u32) >> 8) as f32 * (1.0 / 16_777_216.0);
        #[cfg(any(test, feature = "std"))]
        if counts > DUST_COUNT_MAX_PER_ADD as f32 {
          VOLUME_SPLIT_ADDS.with(|c| c.set(c.get() + 1));
        }
        let mut rest = counts;
        while rest > 0.5 {
          let c = rest.min(DUST_COUNT_MAX_PER_ADD as f32) as u32;
          sat_add(&mut pyramid[i], c);
          for k in 0..3 {
            sat_add(&mut pyramid[i + 1 + k], (c as f32 * color[k] * DUST_COLOR_SCALE + hfr) as u32);
          }
          rest -= c as f32;
        }
        if pyramid[i + 4] == 0 || depth_bits < pyramid[i + 4] {
          pyramid[i + 4] = depth_bits;
        }
        written += 1;
      }
    }
    if written > written_before {
      mask_bits |= 1 << level;
    }
    #[cfg(any(test, feature = "std"))]
    if tracing {
      let name = ["centre", "small", "sub-rays", "lines"][class as usize];
      trace_piece(k, pc, eye, &phot.support, name, k_sub, (tx0, tx1, ty0, ty1), tr_raw, tr_in);
    }
  }
  #[cfg(any(test, feature = "std"))]
  if tracing {
    std::println!(
      "  [trace volume] plan: coarsest piece level {} ({}x{} texels of {} px), levels drawn {mask_bits:#b}, pieces {}, evals {}, sub-ray limit {}, bias {bias}, texel share {}, renorm {renorm:.4}, written {written}",
      plan.level,
      layout.levels[plan.level as usize].1,
      layout.levels[plan.level as usize].2,
      1u32 << plan.level,
      plan.rule.share,
      plan.pieces,
      plan.evals,
      plan.subrays
    );
  }
  let cur = f32::from_bits(pyramid[PYR_TAU_MAX as usize]);
  if peak > 0.0 && !(cur >= peak) {
    pyramid[PYR_TAU_MAX as usize] = peak.to_bits();
  }
  pyramid[PYR_LEVEL_MASK as usize] |= mask_bits;
  written
}

/// One piece of the lab's trace ([`set_volume_trace`]): its mean, σ along x y z, the eye's
/// Mahalanobis distance, flags, own level, the class it was drawn with (and its sub-ray order),
/// its kept mass in the support's half-space, its box, its texel range, and its deposit (Σ τ·px²
/// as stored, and the in-frame part the composite shows).
#[cfg(any(test, feature = "std"))]
#[allow(clippy::too_many_arguments)]
fn trace_piece(
  k: usize,
  pc: &VolumePiece,
  eye: [f32; 3],
  support: &[f32; 4],
  class: &str,
  k_sub: u32,
  range: (i32, i32, i32, i32),
  raw: f64,
  inframe: f64,
) {
  let l = &pc.l;
  let mean = sub3(eye, l_mul(l, pc.qw));
  let sig = [
    <f32 as FloatLike>::sqrt(l[0] * l[0]),
    <f32 as FloatLike>::sqrt(l[1] * l[1] + l[3] * l[3]),
    <f32 as FloatLike>::sqrt(l[2] * l[2] + l[4] * l[4] + l[5] * l[5]),
  ];
  let kept = if has_support(support) {
    let n = [support[0], support[1], support[2]];
    let ltn = lt_mul(l, n);
    let sn = <f32 as FloatLike>::sqrt(dot3(ltn, ltn));
    let mn = dot3(n, mean) - support[3];
    if sn > 0.0 { 0.5 * (1.0 + erf(mn / (sn * core::f32::consts::SQRT_2))) } else { (mn >= 0.0) as u32 as f32 }
  } else {
    1.0
  };
  let mass = 2.0 * core::f32::consts::PI * pc.k * (l[0] * l[3] * l[5]);
  std::println!(
    "  [trace piece] {k}: mean {:?} σ {:?} eye at {:.2} σ, on {} full {} lp {} ls {} line {} class {class} k {k_sub}, mass {mass:.4e} kept {kept:.4}, box {:?}, texels x {}..{} y {}..{}, deposit {raw:.4e} (in frame {inframe:.4e})",
    mean,
    sig,
    <f32 as FloatLike>::sqrt(dot3(pc.qw, pc.qw)),
    pc.on,
    pc.full,
    pc.lp,
    pc.ls,
    pc.line,
    pc.bbox,
    range.0,
    range.1,
    range.2,
    range.3
  );
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
/// 2. (first) the eye inside or near it in perspective ([`is_volumetric`]): [`SplitStep::Volume`]
///    whole — the ray integral cuts every ray at the eye, no clip needed (step F3);
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
///    integral; no single Jacobian holds for it) — with the frame's [`DUST_FRAME_DEEP_CHORD`]
///    bit, a chord deep only by its span is kept instead (its EWA at [`deep_chord_factors`]);
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
  let lab_no_volume = lab_flag(&LAB_NO_VOLUME, "AETHERVK_DUST_LAB_NO_VOLUME");
  // (2b) before the clip (step F3): the eye inside or near the packet — the exact ray integral
  // cuts each ray at the eye itself (`ray_kept`), so a chord across the eye plane is drawn whole;
  // the clip's closed end ramped the density in front of the eye down over the pieces' σ (a chord
  // along the view with the eye inside drew 0.79 of its bounded light,
  // `bounded_chords_match_the_bounded_monte_carlo`)
  if !force && !lab_no_volume && is_volumetric(p, view) {
    return SplitStep::Volume;
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
  // the lab's knobs (CPU only, `dust_observer --lab`): every leaf volumetric / none
  if !force && view.perspective && lab_flag(&LAB_ALL_VOLUME, "AETHERVK_DUST_LAB_ALL_VOLUME") {
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
  if want && !force && view.perspective && view.volumetric && !lab_no_volume {
    // a chord deep only by its span (the Gaussian itself within one Jacobian) with the deep-chord
    // bit: the EWA's exact screen segment and light ([`deep_chord_factors`]) instead
    let deep_only = depth > 0.0
      && span > 3.0 * sigma_v
      && !(3.0 * sigma_v > depth)
      && !(sigma_v > DUST_SPLIT_EPS * depth);
    if !(deep_only && deep_chord_on()) {
      return SplitStep::Volume;
    }
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
          super::specks::ShellClass::Inside => {
            #[cfg(any(test, feature = "std"))]
            if volume_trace() {
              std::println!("  [trace shell] a leaf wholly inside the speck shell ({} m): left to the specks, flux {:.4e}", view.shell, q.flux);
            }
          }
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
              #[cfg(any(test, feature = "std"))]
              if volume_trace() {
                std::println!("  [trace shell] a leaf across the speck shell ({} m): kept {:.4} of flux {:.4e}", view.shell, shell.outside_fraction(&q), q.flux);
              }
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
  // the frame's deep-chord bit for the split and the projection below ([`DUST_FRAME_DEEP_CHORD`])
  set_deep_chord(deep_chord(pyramid));
  #[cfg(any(test, feature = "std"))]
  if volume_trace() {
    let q = sub3(view.eye, p.mean);
    let r = sub3(q, scale3(p.seg, segment_param(q, p.seg)));
    std::println!(
      "[trace packet] mean {:?} seg {:?} cov {:?} flux {:.4e} rho {:?} seg_t {:?}; eye {:?}, distance to the chord {:.4e}, √trΣ {:.4e}, volumetric {}, support {:?} (mass {:.4}), origin {}, shell {}",
      p.mean,
      p.seg,
      p.cov,
      p.flux,
      p.rho,
      p.seg_t,
      view.eye,
      <f32 as FloatLike>::sqrt(dot3(r, r)),
      <f32 as FloatLike>::sqrt(p.cov[0] + p.cov[3] + p.cov[5]),
      is_volumetric(p, view),
      phot.support,
      support_mass(p, &phot.support),
      phot.origin,
      view.shell
    );
  }
  // a perspective packet with a cell the eye is not near: its cone of grains ([`conic_slabs`], the
  // support's cut baked in), each slab drawn uncut; not with a speck shell, whose hand-over is the
  // split's. The eye near it: the exact ray integral below (32 slabs each drawn by the ray
  // integral made 3.8·10⁹ evaluations in one frame of the shell's parity case)
  // (`AETHERVK_DUST_LAB_APEX=0`, CPU only: the former drawing, for A/B replays)
  if view.perspective
    && view.volumetric
    && !(view.shell > 0.0)
    && has_support(&phot.support)
    && phot.origin.is_finite()
    && !is_volumetric(p, view)
    && !legacy_kernel_on()
    && !lab_knob("AETHERVK_DUST_LAB_APEX").is_some_and(|v| v <= 0.0)
  {
    let mut parts = [*p; DUST_SUPPORT_TRUNC_MAX];
    let n = conic_slabs(p, &phot.support, phot.origin, &mut parts);
    #[cfg(any(test, feature = "std"))]
    if volume_trace() {
      std::println!("  [trace route] its cone of grains: {n} slabs, each drawn uncut");
    }
    if n > 0 {
      let free = Photometry { support: [0.0; 4], origin: f32::NAN, ..*phot };
      let mut drawn = 0;
      for q in parts.iter().take(n) {
        drawn += project_split_scatter(q, exposure, mvp, view, &free, color, layout, inv_unit, pyramid, scratch);
      }
      return drawn;
    }
  }
  // a perspective packet its support cuts: drawn whole by the exact ray integral over the
  // half-space (the eye plane too, per ray) when the eye is near it ([`is_volumetric`]), else
  // as its truncation's moments ([`support_truncated`]); not with a speck shell, whose hand-over
  // is the split's
  if view.perspective
    && view.volumetric
    && !(view.shell > 0.0)
    && has_support(&phot.support)
    && support_mass(p, &phot.support) < 1.0 - DUST_SUPPORT_CUT_MIN
  {
    if is_volumetric(p, view) {
      #[cfg(any(test, feature = "std"))]
      if volume_trace() {
        std::println!("  [trace route] cut by its support, the eye near: the ray integral over the half-space");
      }
      return (volume_scatter(p, exposure, mvp, view, phot, color, layout, inv_unit, pyramid) > 0) as u32;
    }
    // the eye not near: its truncation's moments, drawn as uncut packets
    let mut parts = [*p; DUST_SUPPORT_TRUNC_MAX];
    let n = support_truncated(p, &phot.support, &mut parts);
    #[cfg(any(test, feature = "std"))]
    if volume_trace() {
      std::println!("  [trace route] cut by its support: {n} truncated parts, each drawn uncut");
    }
    let free = Photometry { support: [0.0; 4], origin: f32::NAN, ..*phot };
    let mut drawn = 0;
    for q in parts.iter().take(n) {
      drawn += project_split_scatter(q, exposure, mvp, view, &free, color, layout, inv_unit, pyramid, scratch);
    }
    return drawn;
  }
  let mut stats = SplitStats::load(pyramid);
  #[cfg(any(test, feature = "std"))]
  let children_before = stats.children;
  split_packet(p, view, scratch, &mut stats);
  stats.store(pyramid);
  #[cfg(any(test, feature = "std"))]
  let tracing = volume_trace();
  #[cfg(any(test, feature = "std"))]
  if tracing {
    let fl = |v: &[Packet]| v.iter().map(|q| q.flux as f64).sum::<f64>();
    std::println!(
      "  [trace route] split (depth limit {}): {} children, leaves {} volume (flux {:.4e}) {} EWA (flux {:.4e}) of {:.4e}, largest dropped flux {:.4e}",
      stats.depth,
      stats.children - children_before,
      scratch.volume.len(),
      fl(&scratch.volume),
      scratch.ewa.len(),
      fl(&scratch.ewa),
      p.flux,
      stats.dropped_flux
    );
  }
  let mut scattered = 0;
  for leaf in scratch.volume.iter() {
    #[cfg(any(test, feature = "std"))]
    if tracing {
      std::println!("  [trace leaf] volume: mean {:?} seg {:?} cov {:?} flux {:.4e}", leaf.mean, leaf.seg, leaf.cov, leaf.flux);
    }
    if volume_scatter(
      leaf, exposure, mvp, view, phot, color, layout, inv_unit, pyramid,
    ) > 0
    {
      scattered += 1;
    }
  }
  for leaf in scratch.ewa.iter() {
    let sp = project_packet_lit(
      leaf,
      exposure,
      mvp,
      layout.width,
      layout.height,
      view.eye,
      phot,
    );
    #[cfg(any(test, feature = "std"))]
    if tracing {
      std::println!(
        "  [trace leaf] EWA: mean {:?} seg {:?} flux {:.4e} → {}",
        leaf.mean,
        leaf.seg,
        leaf.flux,
        match &sp {
          Some(q) => alloc::format!("screen μ {:?} cov {:?} amp {:.4e} depth {:.4e} AU", q.mu, q.cov, q.amp, q.depth_au),
          None => alloc::string::String::from("not projected (dropped)"),
        }
      );
    }
    if let Some(sp) = sp {
      let t = splat_scatter(&sp, color, layout, inv_unit, pyramid);
      #[cfg(any(test, feature = "std"))]
      if tracing {
        std::println!("    [trace leaf] EWA texels written {t}");
      }
      let _ = t;
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
/// `exp(x) − 1` without the cancellation at small `x` (mirror of `dust_expm1`: the same
/// polynomial and threshold on both sides)
pub fn expm1_f32(x: f32) -> f32 {
  if fabs(x) < 1e-3 {
    x * (1.0 + 0.5 * x * (1.0 + x / 3.0))
  } else {
    <f32 as FloatLike>::exp(x) - 1.0
  }
}

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
    // both ends closed: the two Gaussians' values are within a·|1 − 2u₀|/2 of each other when
    // the chord is far shorter than its σ (a ≪ 1), and their f32 difference over a was off by
    // tens of per cent (a tiny group chord with the polyline's steep ramp drew 1.7× its light,
    // 2026-10-16): e_start − e_end = −exp(z₁)·expm1(z₂ − z₁) with z₂ − z₁ = −a(1 − 2u₀)/2
    let i1 = if !p.open[0] && !p.open[1] {
      let z1 = -0.5 * a * u0 * u0;
      let z2 = -0.5 * a * (1.0 - u0) * (1.0 - u0);
      let dz = z2 - z1;
      // the expm1 form only where the two Gaussians are close (the cancellation): with a large
      // `a` (a 10 px closed chord: a ≈ 400) it is exp(−98) · expm1(+80) = 0 · ∞ = NaN past
      // u₀ ≈ 0.7, and the chord lost its far third (2026-10-16); there the plain difference
      // has no cancellation (the terms differ by a factor ≥ e)
      let diff = if fabs(dz) < 1.0 {
        -<f32 as FloatLike>::exp(z1) * expm1_f32(dz)
      } else {
        <f32 as FloatLike>::exp(z1) - <f32 as FloatLike>::exp(z2)
      };
      u0 * i0 + diff / a
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
      u0 * i0 + (e_start - e_end) / a
    };
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
  #[cfg(any(test, feature = "std"))]
  SAT_ADDS.with(|c| c.set(c.get() + 1));
}

// The splat's work per cluster (diagnostics, `AETHERVK_DUST_WORK_LOG=1`): the saturating adds
// each cluster makes ([`sat_add`]: one compare-and-swap loop each on the GPU) and its split
// children, volume evaluations and EWA texel units (the header counters' increase), logged by
// [`splat_tier_range`] and taken by [`take_work_log`]. Off by default: nothing is recorded.
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static SAT_ADDS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
  static VOLUME_CALLS: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
  static PACKET_CALLS: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
}
#[cfg(any(test, feature = "std"))]
static LAB_WORK_LOG: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
#[cfg(any(test, feature = "std"))]
static WORK_LOG: std::sync::Mutex<alloc::vec::Vec<ClusterWork>> = std::sync::Mutex::new(alloc::vec::Vec::new());

/// The texel run a dither carry stays within ([`segment_carry`]): a packet's texels are drawn in
/// row segments of this many columns (from the row's first column), each starting its carry at a
/// hashed phase, so a segment's counts do not depend on which thread draws it or in what order
/// (the packet queue draws segments in parallel). The carry keeps a faint packet's flux: a
/// segment's counts are `⌊Σ τ + u⌋` with `u` uniform, unbiased, where a carry restarting at 0
/// would drop up to half a count per segment.
pub const DUST_SEGMENT_TEXELS: u32 = 16;

// ─── The packet queue (GPU only: `dust_scatter.glsl`, `dust_draw_queue.comp`) ───────────────
// A packet whose band holds more than [`DUST_QUEUE_MIN_TEXELS`] texels is appended to the view's
// queue by its cluster's thread, as row ranges of at most [`DUST_QUEUE_RECORD_TEXELS`] texels,
// and drawn by a whole workgroup in `Device::cmd_dust_frame_end` (the row segments shared by the
// 64 invocations). The drawing is [`scatter_into`]'s, segment for segment ([`DUST_SEGMENT_TEXELS`]),
// so the CPU mirror needs no queue. The queue lives in the pyramid buffer after
// `PyramidLayout::total_words`: [`DUST_QUEUE_HEADER_WORDS`] (x, 1, 1 for the indirect dispatch; the
// raw append counter; a capacity override, 0 = [`DUST_QUEUE_CAPACITY`]; words 8.. the profiler's
// counters, [`DUST_PROF_STAGE0`]) then the records.
pub const DUST_QUEUE_HEADER_WORDS: u32 = 144;

// ─── The shader-clock profiler (GPU only: `dust_profile.glsl`, `AETHERVK_DUST_PROFILE=1`) ───
// The `.prof` variants of `dust_splat.comp` and `dust_draw_queue.comp` sum the cycles of each
// stage of a thread's work and flush them into the queue header's words 8..127 (64-bit sums as
// (lo, hi) pairs), read back with the queue counts (`Device::dust_profile_counters`). The
// constants mirror `dust_profile.glsl`'s; the words are offsets from the queue base.
/// the words after the queue's own 8 (the counter block), as read back
pub const DUST_PROFILE_WORDS: usize = (DUST_QUEUE_HEADER_WORDS - 8) as usize;
/// 16 × (lo, hi) stage cycle sums of the splat, [`DUST_PROF_STAGE_NAMES`] in order
pub const DUST_PROF_STAGE0: usize = 8;
pub const DUST_PROF_STAGE_NAMES: [&str; 16] = [
  "thread", "setup", "subpk", "packet", "cone", "split", "project", "scprep", "qput", "ewa",
  "vplan", "vdraw", "tracer", "vemit", "count", "ecount",
];
pub const DUST_PROF_MAX_THREAD: usize = 40;
pub const DUST_PROF_THREADS: usize = 41;
pub const DUST_PROF_PACKETS: usize = 42;
pub const DUST_PROF_CONE: usize = 43;
pub const DUST_PROF_LEAVES: usize = 44;
pub const DUST_PROF_RECORDS: usize = 45;
pub const DUST_PROF_VOLUME: usize = 46;
pub const DUST_PROF_NODES: usize = 47;
/// (lo, hi)
pub const DUST_PROF_EWA_TEXELS: usize = 48;
/// (lo, hi)
pub const DUST_PROF_VOLUME_TEXELS: usize = 50;
/// 32 log2 buckets of the threads' cycles
pub const DUST_PROF_HIST: usize = 52;
/// 4 × (lo, hi) stage cycle sums of the queue draw, [`DUST_PROF_QSTAGE_NAMES`] in order
pub const DUST_PROF_QSTAGE0: usize = 84;
pub const DUST_PROF_QSTAGE_NAMES: [&str; 4] = ["qload", "qrows", "qseg", "qthread"];
pub const DUST_PROF_QRECORDS: usize = 92;
/// (lo, hi)
pub const DUST_PROF_QTEXELS: usize = 93;
pub const DUST_PROF_QMAX_WG: usize = 95;
/// the volume consumer (`dust_draw_volume.comp`): (lo, hi) lane cycles, its vplan, its vdraw,
/// records drawn, (lo, hi) texels, the slowest lane
pub const DUST_PROF_CTHREAD: usize = 96;
pub const DUST_PROF_CVPLAN: usize = 98;
pub const DUST_PROF_CVDRAW: usize = 100;
pub const DUST_PROF_CPACKETS: usize = 102;
pub const DUST_PROF_CTEXELS: usize = 103;
pub const DUST_PROF_CMAX: usize = 105;
/// the splitter (`dust_split.comp`): (lo, hi) lane cycles, its lanes with a record, the slowest
pub const DUST_PROF_STHREAD: usize = 106;
pub const DUST_PROF_SLANES: usize = 108;
pub const DUST_PROF_SMAX: usize = 109;
/// K1c's parts of the volumetric draw (every draw, in-thread and consumer; nested in vplan /
/// vdraw): 8 × (lo, hi) cycle sums, [`DUST_PROF_K1C_NAMES`] in order (`dust_prof_flush_k1c`'s);
/// vsmprep / vsmfast / vsmedge are vsmallpts' parts (its prelude; the centre slice by one
/// projection and the edge decision of every piece and an interior piece's other slices; an edge
/// piece's slices by 4 sigma points), the rest of vsmallpts the divergence waits; the last four
/// (round 8 step E) are the consumer's pass B piece loop: vhead (its head: the control word, the
/// record's words, the covariance slots), vbuild (`dust_volume_piece_at`), vdecide (the level
/// table, the range, the class and order under K), vdeposit (`dust_volume_deposit`, every
/// segment function)
pub const DUST_PROF_K1C0: usize = 110;
pub const DUST_PROF_K1C_NAMES: [&str; 12] = [
  "vclass", "vsmallpts", "vsmall", "vsubline", "vcentre", "vsmprep", "vsmfast", "vsmedge", "vhead",
  "vbuild", "vdecide", "vdeposit",
];
/// 2 words: the small pieces drawn by one projection a slice and by 4 sigma points a slice (the
/// GPU's [`take_volume_small_paths`])
pub const DUST_PROF_SMALL_PATHS: usize = 134;

/// The counter block of one drawn frame ([`Device::dust_profile_counters`]): words 8..144 of the
/// queue header, indexed by the `DUST_PROF_*` offsets (from the queue base) through [`Self::word`]
/// / [`Self::u64_at`]; all zero on a frame drawn by the production shaders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DustProfileCounters {
  pub words: [u32; DUST_PROFILE_WORDS],
  /// the frame the counters belong to (the last drawn one)
  pub frame: u64,
}
impl DustProfileCounters {
  /// the word at queue-base offset `at` (≥ 8)
  pub fn word(&self, at: usize) -> u32 {
    self.words[at - 8]
  }
  /// the (lo, hi) pair at queue-base offset `at`
  pub fn u64_at(&self, at: usize) -> u64 {
    self.word(at) as u64 | ((self.word(at + 1) as u64) << 32)
  }
  /// cycles of splat stage `k` ([`DUST_PROF_STAGE_NAMES`])
  pub fn stage(&self, k: usize) -> u64 {
    self.u64_at(DUST_PROF_STAGE0 + 2 * k)
  }
  /// cycles of queue stage `k` ([`DUST_PROF_QSTAGE_NAMES`])
  pub fn qstage(&self, k: usize) -> u64 {
    self.u64_at(DUST_PROF_QSTAGE0 + 2 * k)
  }
  /// whether anything was recorded (the profiler variant ran)
  pub fn any(&self) -> bool {
    self.words.iter().any(|&w| w != 0)
  }
}
pub const DUST_QUEUE_RECORD_WORDS: u32 = 64;
/// records per frame: 2¹⁸ × 256 B = 64 MiB per view; a fuller queue draws the rest in-thread
pub const DUST_QUEUE_CAPACITY: u32 = 1 << 18;
pub const DUST_QUEUE_MIN_TEXELS: f32 = 64.0;
pub const DUST_QUEUE_RECORD_TEXELS: f32 = 4096.0;
/// the queue's words after the pyramid
pub const DUST_QUEUE_WORDS: u32 = DUST_QUEUE_HEADER_WORDS + DUST_QUEUE_CAPACITY * DUST_QUEUE_RECORD_WORDS;
/// [`PYR_FLAGS`] bits (host-written, `AETHERVK_DUST_QUEUE=0|all` or the device's test hook): no
/// queue (every packet in-thread, the A/B) / every packet queued (tests)
pub const DUST_FRAME_QUEUE_OFF: u32 = 1 << 28;
pub const DUST_FRAME_QUEUE_ALL: u32 = 1 << 29;

// ─── The volume packet queue (GPU only: `dust_scatter.glsl`, `dust_draw_volume.comp`) ─────────
// A volumetric packet is no longer drawn in its cluster's thread: the thread emits it, with its
// rounding seed, to a queue after the EWA queue, and `dust_draw_volume.comp` draws the records
// one lane per packet with the same arithmetic (`dust_volume_draw.glsl`), so the pyramid is the
// same word for word while the lanes stay coherent. The whole frame's packets (9.4 M at 1 km) do
// not fit, so `Device::cmd_dust_splat` runs a tier in chunks of clusters: per chunk it writes the
// queue header (the range [first, end), the counters), dispatches the producer over the range,
// then the consumer indirectly; the chunk is sized from the previous frames' packets per cluster
// and the records a full chunk's queue refuses go to the volume spill (`dust_vspill_records`),
// drained by a second dispatch of the same consumer ([`DUST_VIEW_VOLUME_SPILL`]); a record the
// spill refuses too is dropped and counted ([`DUST_VQUEUE_LOST`]). Nothing is drawn in-thread.
/// `dust_splat.comp`'s lanes per cluster: a warp, a lane per sub-packet (the slow end bin's
/// [`DUST_END_SUBBINS`] sub-bins, then a bin each: 17 of 32; a merged polyline's time pieces)
pub const DUST_SPLAT_LANES: u32 = 32;
pub const DUST_VQUEUE_HEADER_WORDS: u32 = 16;
pub const DUST_VQUEUE_RECORD_WORDS: u32 = 32;
/// records per chunk: 2¹⁸ × 128 B = 32 MiB per view
pub const DUST_VQUEUE_CAPACITY: u32 = 1 << 18;
pub const DUST_VQUEUE_WORDS: u32 = DUST_VQUEUE_HEADER_WORDS + DUST_VQUEUE_CAPACITY * DUST_VQUEUE_RECORD_WORDS;
/// the automatic chunk's bounds in clusters (`Device::cmd_dust_splat`: the queue's records over
/// twice the packets per cluster, clamped; 1024 measured best at 1 km). The maximum is also the
/// sub-packet queue's clusters per chunk ([`DUST_SQUEUE_CLUSTERS`]): every chunk is clamped to it.
pub const DUST_VQUEUE_CHUNK_MIN: u32 = 256;
pub const DUST_VQUEUE_CHUNK_MAX: u32 = 1024;
/// The automatic chunk (clusters) for a view of `records` volume records at `ppc` packets per
/// cluster: the records over twice the packets, clamped to [floor, `DUST_VQUEUE_CHUNK_MAX`]
/// where the floor is `DUST_VQUEUE_CHUNK_MIN` scaled by the view's share of the full capacity
/// (256 on a full queue, as measured; 16 at 16 384 records). A fixed 256 floor made a small
/// device's chunk hold more packets than its records (256 × 61 ppc > 16 384: an overflow).
pub fn dust_vqueue_auto_chunk(records: u32, ppc: f32) -> u32 {
  let auto = (records as f32 / (2.0 * ppc.max(1.0))) as u32;
  let floor = ((DUST_VQUEUE_CHUNK_MIN as u64 * records as u64 / DUST_VQUEUE_CAPACITY as u64) as u32).max(1);
  auto.clamp(floor, DUST_VQUEUE_CHUNK_MAX)
}

// ─── The sub-packet queue (GPU only: `dust_scatter.glsl`, `dust_split.comp`) ─────────────────
// After the volume queue: per chunk, the packets `dust_splat.comp`'s lanes construct (one record
// per `dust_project_split_scatter` call of the former per-cluster thread, in its sub-packet
// order; the enumerator), split and drawn by `dust_split.comp` (a workgroup per cluster; the
// splitter: the leaf path, `dust_leaf.glsl`). A header, a table of the chunk's clusters (first
// record, count, index), then structure-of-arrays records. A cluster makes at most
// [`DUST_SQUEUE_PER_CLUSTER`] records (17 sub-packets × [`DUST_TIME_PIECES_MAX`]), so a chunk of
// [`DUST_SQUEUE_CLUSTERS`] clusters cannot overflow it: 20 MiB per view, fixed.
pub const DUST_SQUEUE_HEADER_WORDS: u32 = 16;
pub const DUST_SQUEUE_TABLE_WORDS: u32 = 4;
pub const DUST_SQUEUE_CLUSTERS: u32 = DUST_VQUEUE_CHUNK_MAX;
pub const DUST_SQUEUE_PER_CLUSTER: u32 = 136;
pub const DUST_SQUEUE_CAPACITY: u32 = DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_PER_CLUSTER;
pub const DUST_SQUEUE_RECORD_WORDS: u32 = 36;
// The speck walk's spill (after the sub-packet queue, `dust_spill_base`): the walk pass keeps a
// workgroup's node stack in shared memory, whose size sets the pass's occupancy; a walk whose
// stack would overflow moves its oldest nodes to a slot here (claimed by the first spill, the
// header's word 0 counting them; word 1 the slots this view has) and takes them back when the
// shared stack runs dry. A node is 6 words; a slot holds `DUST_SPILL_NODES`; without a free slot
// the walk is truncated (counted with the caps).
pub const DUST_SPILL_HEADER_WORDS: u32 = 16;
pub const DUST_SPILL_NODE_WORDS: u32 = 6;
pub const DUST_SPILL_NODES: u32 = 1024;
/// slots on a device with the full volume queue (`DUST_VQUEUE_CAPACITY`); fewer in proportion
pub const DUST_SPILL_SLOTS: u32 = 1024;
/// After the spill's slots: the speck trace (`Device::dust_set_specks_trace`): the walk pass
/// appends `(hash, dp xyz, w xyz, coordinates xyz, κ, node level)` of every speck it draws while
/// the spill header's word 2 (the capacity) is non-zero, word 3 counting them;
/// `gpu_dust_specks_match_reference` reads them back and matches them by hash with the CPU's.
pub const DUST_SPECKS_TRACE_CAPACITY: u32 = 1 << 16;
pub const DUST_SPECKS_TRACE_WORDS: u32 = 12;
pub const fn dust_spill_words(slots: u32) -> u32 {
  DUST_SPILL_HEADER_WORDS + slots * DUST_SPILL_NODES * DUST_SPILL_NODE_WORDS + DUST_SPECKS_TRACE_CAPACITY * DUST_SPECKS_TRACE_WORDS
}
pub const DUST_SQUEUE_WORDS: u32 = DUST_SQUEUE_HEADER_WORDS
  + DUST_SQUEUE_CLUSTERS * DUST_SQUEUE_TABLE_WORDS
  + DUST_SQUEUE_CAPACITY * DUST_SQUEUE_RECORD_WORDS;
/// header word 2: records a chunk could not hold (never by construction; the test asserts 0)
pub const DUST_SQUEUE_OVERFLOWED: usize = 2;
/// header words: the chunk's cluster range, the frame's records drawn by the consumer from the
/// main queue and overflowed (refused by it: sent to the volume spill)
pub const DUST_VQUEUE_FIRST: usize = 5;
pub const DUST_VQUEUE_END: usize = 6;
pub const DUST_VQUEUE_DRAWN: usize = 7;
pub const DUST_VQUEUE_OVERFLOWED: usize = 8;
/// header word 9: the volume queue's records this view allocated (host-written each drawn
/// frame; 0 = [`DUST_VQUEUE_CAPACITY`]): the shaders place the sub-packet queue and the spill
/// after them (`dust_squeue_base`), as [`dust_queue_offsets`] does on the host
pub const DUST_VQUEUE_RECORDS: usize = 9;
/// header words 10–11 (host-written per chunk by `cmd_dust_splat`): the tier's live range start
/// (ring slot) and ring mask, so the enumerator maps a range offset `k` to the slot
/// `(first_slot + k) & mask` of the slot-indexed render / moments buffers
pub const DUST_VQUEUE_FIRST_SLOT: usize = 10;
pub const DUST_VQUEUE_RING_MASK: usize = 11;
/// header word 12: the frame's volumetric deposits above `DUST_COUNT_MAX_PER_ADD`, each split into
/// several adds (the GPU's count of [`take_volume_split_adds`]; zeroed per frame with the counts)
pub const DUST_VQUEUE_SPLIT_ADDS: usize = 12;
/// header words 13–14 (the frame's counts, zeroed per frame; written by the spill drain's lane 0
/// of each chunk, `dust_draw_volume.comp` with [`DUST_VIEW_VOLUME_SPILL`]): the records the
/// consumer drew from the volume spill, and the records the spill refused too (dropped: counted,
/// logged, never drawn). [`DUST_VQUEUE_OVERFLOWED`] (the main queue's refusals, the spill's raw
/// appends) = spill drawn + lost by construction.
pub const DUST_VQUEUE_SPILL_DRAWN: usize = 13;
pub const DUST_VQUEUE_LOST: usize = 14;
/// header word 15 (host-written each drawn frame): the volume spill's base in words relative to
/// the volume queue's ([`DustQueueOffsets::vspill`] − [`DustQueueOffsets::vqueue`];
/// `dust_vspill_base`)
pub const DUST_VQUEUE_SPILL_OFFSET: usize = 15;
/// The volume spill's records for a view of `vqueue_records` volume records: a quarter, at least
/// 64 (8 MiB on a full queue). The records the main queue refuses (a chunk whose packets the
/// packets-per-cluster estimate under-counted) go there, drained by a second dispatch of the
/// same consumer, so every volumetric record is drawn by the consumer.
pub const fn dust_vspill_records(vqueue_records: u32) -> u32 {
  let s = vqueue_records / 4;
  if s < 64 { 64 } else { s }
}
/// Track 9A (t9a2d), the deferred edge pieces: the volume consumer draws no small piece whose
/// centre slice's 3σ crosses the frame; it appends the piece (its inputs, range, level, colour,
/// depth bits and rounding seed: [`DUST_VEDGE_RECORD_WORDS`], structure-of-arrays) to the region of
/// its chunk's parity ([`DUST_VIEW_EDGE_ODD`]), and `dust_draw_small_edge.comp` draws them (the
/// demotion or the sigma points: `dust::volume_small_points`' edge branch, the same arithmetic)
/// beside the next chunk's producer. Two regions after the volume spill, each a header in the
/// volume queue's format (words 0–2 the pass's indirect dispatch, 3 the raw appends, 4 the
/// capacity; [`DUST_VEDGE_DRAWN`] / [`DUST_VEDGE_LOST`] / [`DUST_VEDGE_DEMOTED`] the frame's
/// counts) and [`dust_vedge_records`] records. A piece a full region refuses is dropped and counted.
pub const DUST_VEDGE_RECORD_WORDS: u32 = 24;
pub const DUST_VEDGE_DRAWN: usize = 5;
pub const DUST_VEDGE_LOST: usize = 6;
/// t9a2e: a deferred-edge region's frame count of the pieces the edge pass demoted (one projection
/// a slice with the predicted terms instead of the sigma points: `volume_small_points`' `demoted`);
/// mirror of `DUST_VEDGE_DEMOTED` (dust_scatter.glsl), zeroed with the region's header each frame
pub const DUST_VEDGE_DEMOTED: usize = 7;
/// A deferred-edge region's records for a view of `vqueue_records` volume records: half, at least
/// 64 (2¹⁷ × 96 B = 12 MiB a region, 24 MiB a view on a full queue; 10 m measured 2.6 M edge pieces
/// in 173 chunks, 15 k a chunk on average)
pub const fn dust_vedge_records(vqueue_records: u32) -> u32 {
  let s = vqueue_records / 2;
  if s < 64 { 64 } else { s }
}

// ─── The views' queue placement and the EWA queue's dispatch (portability) ───────────────────
/// Workgroups per dispatch dimension every Vulkan device supports (`maxComputeWorkGroupCount`'s
/// required minimum; the Mali-G610 and Intel UHD 620 report exactly this): the EWA queue draw's
/// indirect dispatch is (min(count, X), ceil(count / X), 1) and record `y · X + x`
pub const DUST_QUEUE_DISPATCH_X: u32 = 65_535;
/// The indirect dispatch `dust_queue_append`'s atomics leave for `count` records: from the
/// host's `[0, 1, 1]`, each append `r` raises x to `min(r + 1, X)` and y to `r / X + 1`
pub const fn dust_queue_dispatch(count: u32) -> [u32; 3] {
  if count == 0 {
    [0, 1, 1]
  } else {
    [
      if count < DUST_QUEUE_DISPATCH_X { count } else { DUST_QUEUE_DISPATCH_X },
      count.div_ceil(DUST_QUEUE_DISPATCH_X),
      1,
    ]
  }
}
/// The record `dust_draw_queue.comp`'s workgroup `(x, y)` draws (it returns past the count)
pub const fn dust_queue_record(x: u32, y: u32) -> u32 {
  y * DUST_QUEUE_DISPATCH_X + x
}
/// The EWA queue's records a frame held (the draw's count): the raw appends (header word 3) up
/// to the capacity (word 4, 0 = [`DUST_QUEUE_CAPACITY`]), as `dust_draw_queue.comp` reads it
pub fn dust_queue_held(header: &[u32]) -> u32 {
  let cap = if header[4] == 0 { DUST_QUEUE_CAPACITY } else { header[4].min(DUST_QUEUE_CAPACITY) };
  header[3].min(cap)
}
/// The speck walk's spill slots for a view of `vqueue_records` volume records: in proportion to
/// the full queue's [`DUST_SPILL_SLOTS`], at least 64
pub const fn dust_spill_slots(vqueue_records: u32) -> u32 {
  let s = (DUST_SPILL_SLOTS as u64 * vqueue_records as u64 / DUST_VQUEUE_CAPACITY as u64) as u32;
  if s < 64 { 64 } else { s }
}
/// Where a view's queues lie, in words after the pyramid (`PyramidLayout::total_words`): the EWA
/// queue at 0, then the volume queue of `vqueue_records` records, the sub-packet queue, the
/// spill (with the speck trace), the volume spill (a volume queue of
/// [`dust_vspill_records`] records, the same header and record format), and the words the view
/// allocates (`end`). The shaders compute the first ones from the volume header's
/// [`DUST_VQUEUE_RECORDS`] (`dust_squeue_base`, `dust_spill_base`) and the volume spill from its
/// [`DUST_VQUEUE_SPILL_OFFSET`] (`dust_vspill_base`); every host offset comes from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DustQueueOffsets {
  pub vqueue: u32,
  pub squeue: u32,
  pub spill: u32,
  pub spill_slots: u32,
  pub vspill: u32,
  pub vspill_records: u32,
  /// the deferred edge pieces' two regions (track 9A, t9a2d): region 0 at `vedge[0]`, region 1 at
  /// `vedge[1]`, each a header and `vedge_records` records
  pub vedge: [u32; 2],
  pub vedge_records: u32,
  pub end: u32,
}
pub const fn dust_queue_offsets(vqueue_records: u32) -> DustQueueOffsets {
  let vqueue = DUST_QUEUE_WORDS;
  let squeue = vqueue + DUST_VQUEUE_HEADER_WORDS + vqueue_records * DUST_VQUEUE_RECORD_WORDS;
  let spill = squeue + DUST_SQUEUE_WORDS;
  let spill_slots = dust_spill_slots(vqueue_records);
  // after everything the view had before the volume spill: no earlier offset moves
  let vspill = spill + dust_spill_words(spill_slots);
  let vspill_records = dust_vspill_records(vqueue_records);
  // after the volume spill: no earlier offset moves
  let vedge0 = vspill + DUST_VQUEUE_HEADER_WORDS + vspill_records * DUST_VQUEUE_RECORD_WORDS;
  let vedge_records = dust_vedge_records(vqueue_records);
  let vedge1 = vedge0 + DUST_VQUEUE_HEADER_WORDS + vedge_records * DUST_VEDGE_RECORD_WORDS;
  DustQueueOffsets {
    vqueue,
    squeue,
    spill,
    spill_slots,
    vspill,
    vspill_records,
    vedge: [vedge0, vedge1],
    vedge_records,
    end: vedge1 + DUST_VQUEUE_HEADER_WORDS + vedge_records * DUST_VEDGE_RECORD_WORDS,
  }
}
/// [`PYR_FLAGS`] bit (host-written, `AETHERVK_DUST_VQUEUE=0` or the device's test hook): the main
/// volume queue refuses every record, so every volumetric packet goes through the volume spill
/// (the spill path's A/B: the same pyramid, slower; the chunk sized so the spill holds it)
pub const DUST_FRAME_VQUEUE_OFF: u32 = 1 << 30;
/// [`DustSplatPushConstants::flags`] bit, host-internal (`Device::cmd_dust_splat` sets it on the
/// consumer's second dispatch of each chunk and refuses a caller's flags carrying it):
/// `dust_draw_volume.comp` drains the volume spill instead of the main volume queue. Bits 0–1,
/// 16–17 and 20–27 are the view's flags (`DUST_VIEW_*`).
pub const DUST_VIEW_VOLUME_SPILL: u32 = 1 << 28;
/// [`DustSplatPushConstants::flags`] bit, host-internal (track 9A, t9a2d; `Device::cmd_dust_splat`
/// sets it on an odd chunk's consumer dispatches and on the edge pass that drains them, and refuses
/// a caller's flags carrying it): the deferred edge pieces' region 1 instead of 0
pub const DUST_VIEW_EDGE_ODD: u32 = 1 << 29;
/// [`PYR_FLAGS`] bit (a test hook through `Device::dust_set_vqueue_mode`): `dust_split.comp`
/// walks a cluster's records in one lane with a running ordinal (the former per-cluster thread's
/// order), the word-identical reference for its counting pass and prefix
pub const DUST_FRAME_SPLIT_SERIAL: u32 = 1 << 31;
/// [`PYR_FLAGS`] bit (a test hook through `Device::dust_set_vqueue_mode`): the volume consumer
/// runs its pass A in full even when the frame's bias puts every piece on a 1 × 1 last level
/// (`bias + 1 >= levels`). By default it then only looks for a piece on screen (stopping at the
/// first) and pass B classifies each piece at that level as the gated path does, the evaluations
/// summed there: at that level a piece covers at most one texel, so the cap loops never act and
/// the sub-ray limit is 4, and the pyramid and the evaluations are the same word for word
/// (PARTICLES_PERF round 2; `gpu_dust_top_level_path_draws_the_same_pyramid`)
pub const DUST_FRAME_TOP_PASS_A: u32 = 1 << 2;
const _: () = assert!(
  DUST_FRAME_TOP_PASS_A & (crate::scene::dust::DUST_VIEW_TRACERS | crate::scene::dust::DUST_VIEW_FLOW) == 0
);

// The packet rounding seeds (mirror of `dust_cluster_seed` / `dust_packet_ord`): per cluster
// (its ring slot) and per packet in the order the cluster's thread draws them
#[cfg(any(test, feature = "std"))]
std::thread_local! {
  static PACKET_SEQ: core::cell::Cell<(u32, u32)> = const { core::cell::Cell::new((0, 0)) };
}
#[cfg(not(any(test, feature = "std")))]
static PACKET_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Starts cluster `index`'s packet seeds (mirror of the splat's `main`)
pub fn begin_cluster_packets(index: u32) {
  let seed = dust_hash32(index);
  #[cfg(any(test, feature = "std"))]
  PACKET_SEQ.with(|c| c.set((seed, 0)));
  #[cfg(not(any(test, feature = "std")))]
  PACKET_SEQ.store(seed as u64, core::sync::atomic::Ordering::Relaxed);
}

/// The next packet's rounding seed (mirror of `dust_next_packet_seed`)
pub fn next_packet_seed() -> u32 {
  #[cfg(any(test, feature = "std"))]
  let (seed, ord) = PACKET_SEQ.with(|c| {
    let v = c.get();
    c.set((v.0, v.1.wrapping_add(1)));
    v
  });
  #[cfg(not(any(test, feature = "std")))]
  let (seed, ord) = {
    let v = PACKET_SEQ.fetch_add(1 << 32, core::sync::atomic::Ordering::Relaxed);
    (v as u32, (v >> 32) as u32)
  };
  dust_hash32(seed ^ ord.wrapping_mul(0x9E37_79B9))
}

/// A segment's starting carry, uniform in [−½, ½) (mirror of `dust_segment_carry`); `key` names
/// the segment within its packet
pub fn segment_carry(seed: u32, key: u32) -> f32 {
  (dust_hash32(seed ^ dust_hash32(key)) >> 8) as f32 * (1.0 / 16_777_216.0) - 0.5
}

/// One cluster's splat work ([`take_work_log`])
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterWork {
  /// the compact index in its tier, and the tier's live count (which tier it was)
  pub index: u32,
  pub live: u32,
  pub age_s: f32,
  /// saturating texel-word adds ([`sat_add`])
  pub adds: u64,
  pub split_children: u32,
  pub volume_evals: u32,
  pub ewa_units: u32,
  /// volumetric packets drawn ([`volume_scatter`] calls)
  pub volume_packets: u32,
  /// the cluster's sub-packets ([`subpackets_cells`]: what the level of detail lowers)
  pub packets: u32,
  /// the pooling study: the size polyline's largest edge offset from the pooled mean over the
  /// bins' width along it (rule A), the polyline's largest angle from the eye about its mean (rad,
  /// rule B), and the mean's distance from the eye (m)
  pub shape_ratio: f32,
  pub spread_rad: f32,
  pub dist_m: f32,
}

/// The pooling study's shape of cluster `m` seen from `eye` ([`ClusterWork::shape_ratio`] …)
pub fn pool_study(m: &DustMoments, eye: [f32; 3]) -> ClusterWork {
  let pooled = pooled_packet(m, [0.0; 3]);
  let within = pooled_within_cov(m, [0.0; 3]);
  let mut extent = 0.0f32;
  let mut dir = [0.0f32; 3];
  let mut spread = 0.0f32;
  let to_mean = sub3(pooled.mean, eye);
  let dm = <f32 as FloatLike>::sqrt(dot3(to_mean, to_mean));
  for j in 0..=SIZE_BINS as usize {
    let d = sub3(m.edge(j), pooled.mean);
    let r = <f32 as FloatLike>::sqrt(dot3(d, d));
    if r > extent {
      extent = r;
      dir = scale3(d, 1.0 / r.max(1e-30));
    }
    let to_e = sub3(m.edge(j), eye);
    let de = <f32 as FloatLike>::sqrt(dot3(to_e, to_e));
    let c = (dot3(to_e, to_mean) / (de * dm).max(1e-30)).clamp(-1.0, 1.0);
    spread = spread.max(<f32 as FloatLike>::acos(c));
  }
  let sw = <f32 as FloatLike>::sqrt(var_along(&within, dir).max(1e-30));
  ClusterWork {
    shape_ratio: if extent > 0.0 { extent / sw } else { 0.0 },
    spread_rad: spread,
    dist_m: dm,
    ..ClusterWork::default()
  }
}

/// The work log since the last call ([`ClusterWork`]; empty unless `AETHERVK_DUST_WORK_LOG=1`).
/// With `AETHERVK_DUST_POOL_STUDY=1` it also logs which clusters' volumetric packets the two
/// pooling rules would cover: A (shape ratio ≤ 0.25 / 0.5 / 1) and B (spread ≤ 1/8, 1/4, 1/2 rad).
#[cfg(any(test, feature = "std"))]
pub fn take_work_log() -> alloc::vec::Vec<ClusterWork> {
  let log = core::mem::take(&mut *WORK_LOG.lock().unwrap_or_else(|e| e.into_inner()));
  if let Some(v) = SKY_STUDY_TOTAL.lock().unwrap_or_else(|e| e.into_inner()).take() {
    aethervk_oshal_rlib::log!(
      "[sky study] {} volumetric packets ({} with the eye inside a piece's 3σ), screen evaluations {}, map evaluations at {} texels/σ on a {}² octahedron {} ({:.1}×); angular σ min {:.4} rad, by bins <0.01 / 0.01-0.03 / 0.03-0.1 / 0.1-0.3 / 0.3-1 / ≥1 rad: {:?}",
      v.packets, v.full, v.screen_evals, DUST_VOLUME_TEXELS_PER_SIGMA, SKY_STUDY_BASE, v.map_evals,
      v.map_evals as f64 / v.screen_evals.max(1) as f64, v.theta_min, v.theta_bins
    );
  }
  if std::env::var("AETHERVK_DUST_POOL_STUDY").is_ok_and(|v| v.trim() == "1") && !log.is_empty() {
    let vp: u64 = log.iter().map(|c| c.volume_packets as u64).sum();
    let ve: u64 = log.iter().map(|c| c.volume_evals as u64).sum();
    aethervk_oshal_rlib::log!("[pool study] {} clusters, {vp} volumetric packets, {ve} evaluations", log.len());
    for k in [0.25f32, 0.5, 1.0, 2.0] {
      let a: alloc::vec::Vec<&ClusterWork> = log.iter().filter(|c| c.shape_ratio <= k).collect();
      aethervk_oshal_rlib::log!(
        "[pool study] A shape ≤ {k}: {} clusters, {} of the volumetric packets, {} of the evaluations",
        a.len(),
        a.iter().map(|c| c.volume_packets as u64).sum::<u64>(),
        a.iter().map(|c| c.volume_evals as u64).sum::<u64>()
      );
    }
    for k in [0.125f32, 0.25, 0.5, 1.0] {
      let b: alloc::vec::Vec<&ClusterWork> = log.iter().filter(|c| c.spread_rad <= k).collect();
      aethervk_oshal_rlib::log!(
        "[pool study] B spread ≤ {k} rad: {} clusters, {} of the volumetric packets, {} of the evaluations",
        b.len(),
        b.iter().map(|c| c.volume_packets as u64).sum::<u64>(),
        b.iter().map(|c| c.volume_evals as u64).sum::<u64>()
      );
    }
    let mut ratios: alloc::vec::Vec<f32> = log.iter().filter(|c| c.volume_packets > 0).map(|c| c.shape_ratio).collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    if !ratios.is_empty() {
      let q = |p: f32| ratios[((ratios.len() - 1) as f32 * p) as usize];
      aethervk_oshal_rlib::log!(
        "[pool study] shape ratio of volumetric clusters: p10 {:.3} p50 {:.3} p90 {:.3}",
        q(0.1),
        q(0.5),
        q(0.9)
      );
    }
  }
  log
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

/// The coarsest level the lab's K1 knobs let a volumetric packet reach (the gate level, a ~4 × 3
/// grid, never the 1 × 1 or 2 × 1 ones; at least [`DUST_VOLUME_MIN_LEVEL`] when the pyramid has
/// that many levels). Production uses no cap ([`volume_plan`]; the measurement of 2026-10-16).
pub fn volume_full_top(levels: u32) -> u32 {
  gate_level(levels).max(DUST_VOLUME_MIN_LEVEL).min(levels.saturating_sub(1))
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
  // error diffusion along each row segment ([`DUST_SEGMENT_TEXELS`]): the fraction below one
  // count is carried to the next texel instead of being dropped, from a hashed starting phase,
  // so a packet deposits its whole flux however faint its texels (16 bins of 1/16 flux each
  // would otherwise lose their outskirts, and the picture would depend on the zoom)
  let seed = next_packet_seed();
  let mut carry = 0.0f32;
  // the lab's touch log ([`set_touch_log`]): the rows' column range, one rectangle per call
  let log = touch_log_on();
  let (mut lx, mut hx) = (i64::MAX, i64::MIN);
  for ty in y0..=y1 {
    let yc = (ty as f32 + 0.5) * s;
    let Some((x0, x1)) = band.columns(yc, s, w) else {
      continue;
    };
    if log {
      lx = lx.min(x0);
      hx = hx.max(x1);
    }
    for tx in x0..=x1 {
      if (tx - x0) as u32 % DUST_SEGMENT_TEXELS == 0 {
        carry = segment_carry(seed, (ty as u32) << 16 | tx as u32);
      }
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
  if log {
    touch_rect(off, w, lx, hx, y0, y1);
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

/// Merges a partial pyramid into `out` (same layout): saturating sums for the count / colour
/// words, the nonzero minimum for the depth word (as `scatter_into`), the maximum of
/// [`PYR_TAU_MAX`] and the union of [`PYR_LEVEL_MASK`]; the split counters summed, the largest
/// dropped flux by maximum ([`SplitStats`]).
pub fn merge_pyramid(out: &mut [u32], part: &[u32]) {
  let h = PYRAMID_HEADER_WORDS as usize;
  let tm = PYR_TAU_MAX as usize;
  if f32::from_bits(part[tm]) > f32::from_bits(out[tm]) {
    out[tm] = part[tm];
  }
  out[PYR_LEVEL_MASK as usize] |= part[PYR_LEVEL_MASK as usize];
  let ch = PYR_SPLIT_CHILDREN as usize;
  out[ch] = out[ch].saturating_add(part[ch]);
  let ve = PYR_VOLUME_EVALS as usize;
  out[ve] = out[ve].wrapping_add(part[ve]);
  let et = PYR_EWA_TEXELS as usize;
  out[et] = out[et].wrapping_add(part[et]);
  let df = PYR_SPLIT_DROPPED_FLUX as usize;
  if f32::from_bits(part[df]) > f32::from_bits(out[df]) {
    out[df] = part[df];
  }
  let w = PYRAMID_TEXEL_WORDS as usize;
  for (o, p) in out[h..].chunks_exact_mut(w).zip(part[h..].chunks_exact(w)) {
    for k in 0..4 {
      sat_add(&mut o[k], p[k]);
    }
    if p[4] != 0 && (o[4] == 0 || p[4] < o[4]) {
      o[4] = p[4];
    }
  }
}

/// [`splat_tier`] on `threads` threads (host replays: the observer's lab). Each thread splats
/// blocks of clusters, handed out dynamically (a young near cluster splits far deeper than an old
/// one), into its own pyramid carrying `pyramid`'s header with the counters zeroed; the parts
/// merge into `pyramid` by [`merge_pyramid`] — order-free, so the result is bit-identical to
/// [`splat_tier`] (`host_rasterizer_is_thread_count_independent`). One thread, or the
/// `AETHERVK_DUST_PROBE` diagnostics (reports per call), runs [`splat_tier`] itself.
#[cfg(any(test, feature = "std"))]
#[allow(clippy::too_many_arguments)]
pub fn splat_tier_parallel(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  threads: usize,
) {
  use std::sync::atomic::{AtomicUsize, Ordering};
  // small blocks: a young near cluster costs far more than an old one, and 64-cluster blocks left
  // the last few threads drawing alone at the end of each call
  const BLOCK: usize = 4;
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  let threads = threads.min(n.div_ceil(BLOCK)).max(1);
  if threads == 1 || std::env::var_os("AETHERVK_DUST_PROBE").is_some() {
    splat_tier(moments, render, pc, flow, tracer_counts, layout, pyramid);
    return;
  }
  let hn = PYRAMID_HEADER_WORDS as usize;
  let mut header = pyramid[..hn].to_vec();
  for wd in [PYR_TAU_MAX, PYR_LEVEL_MASK, PYR_SPLIT_CHILDREN, PYR_VOLUME_EVALS, PYR_EWA_TEXELS, PYR_SPLIT_DROPPED_FLUX] {
    header[wd as usize] = 0;
  }
  let (next, len) = (AtomicUsize::new(0), pyramid.len());
  let parts: alloc::vec::Vec<alloc::vec::Vec<u32>> = std::thread::scope(|s| {
    let handles: alloc::vec::Vec<_> = (0..threads)
      .map(|_| {
        let (header, next) = (&header, &next);
        s.spawn(move || {
          let mut p = alloc::vec![0u32; len];
          p[..hn].copy_from_slice(header);
          loop {
            let lo = next.fetch_add(BLOCK, Ordering::Relaxed);
            if lo >= n {
              break;
            }
            splat_tier_range(moments, render, pc, flow, tracer_counts, layout, &mut p, lo..(lo + BLOCK).min(n));
          }
          p
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
  });
  for p in &parts {
    merge_pyramid(pyramid, p);
  }
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
  view.volume_detail = super::volume_detail_from_flags(pc.flags);
  let mut scratch = SplitLeaves::default();
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  // `AETHERVK_DUST_PROBE="x,y;x,y"` (diagnostics, CPU particle mode): the clusters' τ at those
  // pixels, the largest contributors logged per tier ([`splat_tier_probed`])
  #[cfg(any(test, feature = "std"))]
  let probes = probe_pixels();
  #[cfg(not(any(test, feature = "std")))]
  let probes: alloc::vec::Vec<(u32, u32)> = alloc::vec::Vec::new();
  let reports = splat_range_probed(
    moments, render, pc, flow, tracer_counts, layout, pyramid, &view, &mut scratch, stream, eye,
    range.start..range.end.min(n), &probes,
  );
  for r in &reports {
    let mut line = alloc::format!(
      "[Dust probe] tier live {n} pixel ({},{}): τ {:.4e} from {} clusters; top:",
      r.pixel.0,
      r.pixel.1,
      r.total,
      r.clusters
    );
    for h in r.hits.iter().take(8) {
      line += &alloc::format!(" [{h}]");
    }
    aethervk_oshal_rlib::log!("{}", line);
  }
}

/// [`splat_tier_range`] over a list of compact indices instead of a range (the temporal reuse's
/// class lists and fresh set, `cadence`): every listed cluster below the live count with flux is
/// drawn by the same body ([`splat_cluster_inner`]), in list order; the sums are order-free, so a
/// permutation of the list draws the same pyramid. The predecessor lookup reads the whole slices.
/// No probe pixels and no work log (the range form keeps those diagnostics).
#[allow(clippy::too_many_arguments)]
pub fn splat_tier_list(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  list: &[u32],
) {
  let stream = unpack_color(pc.color);
  let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
  let mut view = SplitView::of(&pc.mvp, eye);
  view.shell = super::specks::ShellCut::from_header(pyramid, eye).d_max;
  view.volume_detail = super::volume_detail_from_flags(pc.flags);
  let mut scratch = SplitLeaves::default();
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  for &i in list {
    let i = i as usize;
    if i >= n || !(moments[i].flux() > 0.0) {
      continue;
    }
    splat_cluster_inner(
      moments, render, pc, flow, tracer_counts, layout, pyramid, &view, &mut scratch, stream, eye, i,
    );
  }
}

/// [`splat_tier_list`] on `threads` threads (the cadence's CPU fresh pass and build chunks, the
/// tests): blocks of the list handed out dynamically, each into its own pyramid carrying
/// `pyramid`'s header with the counters zeroed, merged by [`merge_pyramid`] — the same scheme as
/// [`splat_tier_parallel`]. **Word-identical to [`splat_tier_list`] with the in-frame gates off**
/// (`DUST_FRAME_GATES_OFF_BIT`): the scatter is a saturating sum per word, a non-zero minimum for
/// the depth, a maximum for `PYR_TAU_MAX`, an OR for `PYR_LEVEL_MASK`, all order-free, and each
/// cluster's dither seeds from its own slot. With the gates on it may differ: the gates read the
/// running counters of the pyramid drawn into, which each partial starts at zero (2026-10-17: 976
/// words on the cadence scene, the serial draw tripping a gate the partials never reach).
#[cfg(any(test, feature = "std"))]
#[allow(clippy::too_many_arguments)]
pub fn splat_tier_list_parallel(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  list: &[u32],
  threads: usize,
) {
  use std::sync::atomic::{AtomicUsize, Ordering};
  // small blocks, as in `splat_tier_parallel` (the load balance of short lists)
  const BLOCK: usize = 4;
  let n = list.len();
  let threads = threads.min(n.div_ceil(BLOCK)).max(1);
  if threads == 1 {
    splat_tier_list(moments, render, pc, flow, tracer_counts, layout, pyramid, list);
    return;
  }
  let hn = PYRAMID_HEADER_WORDS as usize;
  let mut header = pyramid[..hn].to_vec();
  for wd in [PYR_TAU_MAX, PYR_LEVEL_MASK, PYR_SPLIT_CHILDREN, PYR_VOLUME_EVALS, PYR_EWA_TEXELS, PYR_SPLIT_DROPPED_FLUX] {
    header[wd as usize] = 0;
  }
  let (next, len) = (AtomicUsize::new(0), pyramid.len());
  let parts: alloc::vec::Vec<alloc::vec::Vec<u32>> = std::thread::scope(|s| {
    let handles: alloc::vec::Vec<_> = (0..threads)
      .map(|_| {
        let (header, next) = (&header, &next);
        s.spawn(move || {
          let mut p = alloc::vec![0u32; len];
          p[..hn].copy_from_slice(header);
          loop {
            let lo = next.fetch_add(BLOCK, Ordering::Relaxed);
            if lo >= n {
              break;
            }
            splat_tier_list(moments, render, pc, flow, tracer_counts, layout, &mut p, &list[lo..(lo + BLOCK).min(n)]);
          }
          p
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
  });
  for p in &parts {
    merge_pyramid(pyramid, p);
  }
}

/// One cluster's share of a probe pixel ([`splat_tier_probed`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeHit {
  /// the τ (per px², exposure-scaled) it added at the pixel
  pub tau: f32,
  /// the share of the pixel's τ
  pub share: f32,
  pub index: usize,
  pub age_s: f32,
  /// its mean's distance from the eye and depth along the view (m)
  pub dist_m: f32,
  pub depth_m: f32,
  /// `√(tr Σ_v / 3)` (m)
  pub sigma_m: f32,
  /// the size polyline's extent (m: the farthest edge from the mean)
  pub chord_m: f32,
  /// the time segment to the stream predecessor (m; 0 without one)
  pub tseg_m: f32,
  pub pred: bool,
  pub merged: bool,
}

impl core::fmt::Display for ProbeHit {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "{:.1}% idx {} age {:.0}s dist {:.0}m depth {:.0}m σ {:.0}m chord {:.0}m tseg {:.0}m pred {} pooled {}",
      100.0 * self.share,
      self.index,
      self.age_s,
      self.dist_m,
      self.depth_m,
      self.sigma_m,
      self.chord_m,
      self.tseg_m,
      self.pred,
      self.merged
    )
  }
}

/// A probe pixel's attribution ([`splat_tier_probed`]): the τ the tier drew there, how many
/// clusters drew some, the largest contributors (descending, at most 32).
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeReport {
  pub pixel: (u32, u32),
  pub total: f32,
  pub clusters: usize,
  pub hits: alloc::vec::Vec<ProbeHit>,
}

/// [`splat_tier`] with a per-pixel attribution (diagnostics): for every probe pixel, the τ each
/// cluster adds there (the composite read before and after the cluster, [`composite_sample`]),
/// with the cluster's age, distance, width, chord and pooling — which clusters draw a feature of
/// the frame. The pyramid is the same as [`splat_tier`]'s.
#[allow(clippy::too_many_arguments)]
pub fn splat_tier_probed(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSplatPushConstants,
  flow: &DustFlowUniform,
  tracer_counts: u32,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
  probes: &[(u32, u32)],
) -> alloc::vec::Vec<ProbeReport> {
  let stream = unpack_color(pc.color);
  let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
  let mut view = SplitView::of(&pc.mvp, eye);
  view.shell = super::specks::ShellCut::from_header(pyramid, eye).d_max;
  view.volume_detail = super::volume_detail_from_flags(pc.flags);
  let mut scratch = SplitLeaves::default();
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  splat_range_probed(
    moments, render, pc, flow, tracer_counts, layout, pyramid, &view, &mut scratch, stream, eye,
    0..n, probes,
  )
}

#[allow(clippy::too_many_arguments)]
fn splat_range_probed(
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
  range: core::ops::Range<usize>,
  probes: &[(u32, u32)],
) -> alloc::vec::Vec<ProbeReport> {
  let unit = if pc.inv_unit > 0.0 { 1.0 / pc.inv_unit } else { 1.0 };
  let mut hits: alloc::vec::Vec<alloc::vec::Vec<(f32, usize)>> =
    alloc::vec![alloc::vec::Vec::new(); probes.len()];
  let mut before = alloc::vec![0.0f32; probes.len()];
  #[cfg(any(test, feature = "std"))]
  let logging = lab_flag(&LAB_WORK_LOG, "AETHERVK_DUST_WORK_LOG");
  #[cfg(any(test, feature = "std"))]
  let mut log: alloc::vec::Vec<ClusterWork> = alloc::vec::Vec::new();
  for i in range {
    let m = &moments[i];
    if !(m.flux() > 0.0) {
      continue;
    }
    for (k, &(x, y)) in probes.iter().enumerate() {
      before[k] = composite_sample(layout, pyramid, unit, x, y).0;
    }
    #[cfg(any(test, feature = "std"))]
    let v0 = VOLUME_CALLS.with(|c| c.get());
    #[cfg(any(test, feature = "std"))]
    let p0 = PACKET_CALLS.with(|c| c.get());
    #[cfg(any(test, feature = "std"))]
    let w0 = logging.then(|| {
      (
        SAT_ADDS.with(|c| c.get()),
        pyramid[PYR_SPLIT_CHILDREN as usize],
        pyramid[PYR_VOLUME_EVALS as usize],
        pyramid[PYR_EWA_TEXELS as usize],
      )
    });
    splat_cluster_inner(
      moments, render, pc, flow, tracer_counts, layout, pyramid, view, scratch, stream, eye, i,
    );
    #[cfg(any(test, feature = "std"))]
    if let Some((a, c, v, e)) = w0 {
      log.push(ClusterWork {
        index: i as u32,
        live: pc.live_count,
        age_s: m.age(),
        adds: SAT_ADDS.with(|x| x.get()) - a,
        split_children: pyramid[PYR_SPLIT_CHILDREN as usize].wrapping_sub(c),
        volume_evals: pyramid[PYR_VOLUME_EVALS as usize].wrapping_sub(v),
        ewa_units: pyramid[PYR_EWA_TEXELS as usize].wrapping_sub(e),
        volume_packets: VOLUME_CALLS.with(|x| x.get()) - v0,
        packets: PACKET_CALLS.with(|x| x.get()) - p0,
        ..pool_study(m, eye)
      });
    }
    for (k, &(x, y)) in probes.iter().enumerate() {
      let d = composite_sample(layout, pyramid, unit, x, y).0 - before[k];
      if d > 0.0 {
        hits[k].push((d, i));
      }
    }
  }
  #[cfg(any(test, feature = "std"))]
  if !log.is_empty() {
    WORK_LOG.lock().unwrap_or_else(|e| e.into_inner()).extend(log);
  }
  #[cfg(any(test, feature = "std"))]
  sky_study_flush();
  probes
    .iter()
    .zip(hits)
    .map(|(&pixel, mut h)| {
      h.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal));
      let total: f32 = h.iter().map(|x| x.0).sum();
      let clusters = h.len();
      let top = h
        .iter()
        .take(32)
        .map(|&(tau, i)| {
          let m = &moments[i];
          let dm = sub3(m.mean(), eye);
          let cv = m.cov_v();
          let mut chord = 0.0f32;
          for j in 0..=SIZE_BINS as usize {
            let e = sub3(m.edge(j), m.mean());
            chord = chord.max(<f32 as FloatLike>::sqrt(dot3(e, e)));
          }
          let pred = streak_pred(render, i);
          ProbeHit {
            tau,
            share: tau / total.max(1e-30),
            index: i,
            age_s: m.age(),
            dist_m: <f32 as FloatLike>::sqrt(dot3(dm, dm)),
            depth_m: dot3(dm, view.fwd),
            sigma_m: <f32 as FloatLike>::sqrt(((cv[0] + cv[3] + cv[5]) / 3.0).max(0.0)),
            chord_m: chord,
            tseg_m: pred
              .map(|q| {
                let d = sub3(moments[q].mean(), m.mean());
                <f32 as FloatLike>::sqrt(dot3(d, d))
              })
              .unwrap_or(0.0),
            pred: pred.is_some(),
            merged: polyline_merged(m, &pc.mvp, layout.width, layout.height),
          }
        })
        .collect();
      ProbeReport {
        pixel,
        total,
        clusters,
        hits: top,
      }
    })
    .collect()
}

/// `AETHERVK_DUST_PROBE="x,y;x,y"`: the probe pixels of [`splat_tier_range`] (diagnostics)
#[cfg(any(test, feature = "std"))]
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
  // the cluster's seed is its ring slot (stable across ring advances; dust_split.comp)
  begin_cluster_packets(super::render_slot(render[i].age_id_dbeta_flux[1]));
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
      support: [0.0; 4],
      origin: f32::NAN,
    };
    let color = age_color(stream, m.age(), pc.eye_local[3]);
    // pooled only when the predecessor's polyline is below a pixel too: a just-emitted sample is
    // a point whose time pieces are shaped by its predecessor, and a predecessor whose sizes
    // already span kilometres must be drawn bin by bin (the halo behind the nucleus of the 5-km
    // frame came from its pooled covariance)
    let pred_merged = pred.is_none_or(|p| polyline_merged(p, &pc.mvp, layout.width, layout.height));
    // (perspective: neither the polylines nor the time segment between them may be deep)
    // (and never pooled when the speck pass walks some of its bins: the hand-over is per bin)
    let merged = polyline_merged(m, &pc.mvp, layout.width, layout.height)
      && pred_merged
      && pred.is_none_or(|p| time_shallow(m, p, &pc.mvp))
      && m.specks_mask() == 0;
    let px = |q: [f32; 3]| project_point(q, &pc.mvp, layout.width, layout.height);
    // every sub-packet is split for the perspective (clipped at the eye plane, sliced by depth:
    // `split_packet`) and its leaves are projected and scattered one by one
    let n_hard = hard_axis_pieces(pyramid);
    TIME_CELL_OFFSET.store(time_cell_offset(pyramid), core::sync::atomic::Ordering::Relaxed);
    set_legacy_kernel(legacy_kernel(pyramid));
    // the lab's knobs (CPU only)
    let no_cell = lab_knob("AETHERVK_DUST_LAB_NO_TIME_CELL").is_some_and(|v| v > 0.0);
    let sig2 = lab_knob("AETHERVK_DUST_LAB_SIGMA").map(|v| v * v);
    let no_support = lab_knob("AETHERVK_DUST_LAB_NO_SUPPORT").is_some_and(|v| v > 0.0);
    let (seg, pred) = if no_cell { ([0.0; 3], None) } else { (seg, pred) };
    // the cluster level of detail ([`lod_group`], [`PYR_LOD`]): chord groups for a polyline whose
    // bins are shorter than `lod_px` on screen; the lab knob `AETHERVK_DUST_LAB_GROUP=g` forces
    // the 2026-10-14 volumetric rule instead (`cluster_group`)
    set_lod_px_active(lod_px(pyramid));
    set_lod_view_active(view.perspective.then_some((view.eye, view.fwd)));
    if let Some(v) = lab_knob("AETHERVK_DUST_LAB_GROUP") {
      request_group(cluster_group(m, view, v as usize));
    }
    let packets = subpackets_cells_bins(m, seg, pred, merged, pred_merged, Some(&px));
    request_group(1);
    set_lod_px_active(0.0);
    set_lod_view_active(None);
    #[cfg(any(test, feature = "std"))]
    PACKET_CALLS.with(|c| c.set(c.get() + packets.len() as u32));
    for (p0, support, w_o, bin) in packets {
      let support = if no_support { [0.0; 4] } else { support };
      let phot = Photometry { support, origin: w_o, ..phot };
      // the speck shell's hand-over per bin ([`DustMoments::specks_walked`], the speck pass's
      // walk gate): a bin the specks do not walk keeps its shell mass (no shell for its packets)
      let pview = SplitView {
        shell: if bin != 0xFF && m.specks_walked(bin as usize) { view.shell } else { 0.0 },
        ..*view
      };
      let p0 = match sig2 {
        Some(k) => Packet {
          cov: [p0.cov[0] * k, p0.cov[1] * k, p0.cov[2] * k, p0.cov[3] * k, p0.cov[4] * k, p0.cov[5] * k],
          ..p0
        },
        None => p0,
      };
      for j in 0..n_hard.max(1) {
        let p = hard_axis_leaf(&p0, n_hard, j);
        project_split_scatter(
        &p,
        exposure,
        &pc.mvp,
        &pview,
        &phot,
        color,
        layout,
        pc.inv_unit,
        pyramid,
        scratch,
      );
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
        seg_t: [0.0; 3],
      };
      if let Some(sp) = project_packet(&p, 1.0, &pc.mvp, layout.width, layout.height, eye) {
        splat_tracer(sp.mu, tracer_counts, layout, pyramid);
      }
    }
  }
}

/// Stream predecessor of compact index `r` (the tier's live range in ring order, `r` ↔ slot
/// `(first_slot + r) & mask`; mirror of `dust_streak_pred`, see [`super::streak_pred`]): `r − S`
/// when `r ≥ S`, it is live, no break precedes `r`, and its slot is the slot arithmetic's
/// `(slot − S) & mask` (the mask from the render word, [`super::render_mask`]): the compact and
/// the slot forms ([`streak_pred_slot`]) agree.
pub fn streak_pred(render: &[DustRenderCluster], r: usize) -> Option<usize> {
  let y = render[r].age_id_dbeta_flux[1];
  let z = render[r].age_id_dbeta_flux[2];
  let n = 1usize << render_stream_shift(y);
  if r < n || stream_break(z) {
    return None;
  }
  let yq = render[r - n].age_id_dbeta_flux[1];
  // records captured before the mask bits (log2 capacity 0, a one-slot ring that never exists)
  // keep the compact rule alone
  let legacy = super::render_log2_capacity(y) == 0;
  let slot_q = super::render_slot(y).wrapping_sub(n as u32) & super::render_mask(y);
  (render_live(yq) && (legacy || super::render_slot(yq) == slot_q)).then_some(r - n)
}

/// Stream predecessor of the cluster in ring slot `slot` of a slot-indexed render buffer (mirror
/// of `dust_streak_pred`): `(slot − S) & mask` when the cluster's ring offset
/// `k = (slot − first_slot) & mask` is in `[S, live_count)`, no break precedes it and the record
/// there is live. The offset bound keeps a retired (reused) slot from ever being a predecessor:
/// `k − S` is then in the live range, so no retired record needs clearing.
pub fn streak_pred_slot(
  render: &[DustRenderCluster],
  slot: u32,
  first_slot: u32,
  live_count: u32,
) -> Option<u32> {
  let y = render[slot as usize].age_id_dbeta_flux[1];
  let z = render[slot as usize].age_id_dbeta_flux[2];
  let mask = super::render_mask(y);
  let n = 1u32 << render_stream_shift(y);
  let k = slot.wrapping_sub(first_slot) & mask;
  if k < n || k >= live_count || stream_break(z) {
    return None;
  }
  let p = slot.wrapping_sub(n) & mask;
  render_live(render[p as usize].age_id_dbeta_flux[1]).then_some(p)
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
