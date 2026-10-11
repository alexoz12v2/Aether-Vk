//! Physical specks (WP3b): the discrete grains of the near field.
//!
//! Where the camera is within metres of dust there is less than one grain per pixel along a ray
//! and the continuum ([`super::splat`]) is the wrong picture: a smooth haze where the eye would
//! see individual motes. This module draws those motes, **deterministically** (a speck keeps its
//! identity and moves with the flow from frame to frame, no per-speck state) and in **physical
//! number** (the count of a size bin is its mass over the grain mass).
//!
//! **Count.** A cluster of flux `F` (cross-section, m²) spreads over [`SIZE_BINS`] bins of equal
//! cross-section (`F_j = F / 16` for the size law `q = 3.5`); bin `j` holds `N_j = F_j / ⟨πs²⟩_j`
//! grains, `⟨πs²⟩_j` the mean cross-section of its size range `[s_ref / f_{j+1}², s_ref / f_j²]`
//! (`f_j` the bin edge's speed factor `√(β_j/β)`, [`DustMoments::edge_factor`]). This equals
//! `m_cluster · w_j / ⟨m_grain⟩_j` with the mass weights `w_j` of the size law and
//! `m_grain = (4/3)π s³ ρ` ([`bin_count_from_mass`] / [`bin_count_from_flux`],
//! `counts_from_mass_and_flux_agree`), without needing the density on the render side. The size
//! law `q` and the range factor `F` are [`SIZE_Q`] / [`SIZE_RANGE`] (TODO: expose as
//! `ParticleSystemEmitParams::size_q` / `size_range`, see the module constants).
//!
//! **Regime boundary.** Inside the camera-centred shell of radius `d_max = (3 / (n_eye Ω_px))^{1/3}`
//! ([`shell_radius_m`]) a pixel's ray meets less than one grain: `n_eye` is the number density at
//! the eye, `Σ_clusters Σ_bins N_j · G(eye; μ_j, Σ_j)` ([`number_density_at`]), `Ω_px` the pixel
//! solid angle ([`pixel_solid_angle_sr`]). The GPU accumulates `n_eye` into the pyramid header
//! ([`PYR_N_EYE`], 64-bit fixed point at [`N_EYE_UNIT`]); the host reads it back, sets the next
//! frame's `d_max` ([`PYR_SPECKS_DMAX`]): from [`DUST_SPECKS_START_M`] it grows (at most
//! [`DUST_SPECKS_GROW`] a readback, toward half the budget) while a frame draws under a quarter of it, shrinks by the
//! overflow's cube root when the budget overflowed ([`PYR_SPECKS_OVERFLOW`]), never past the
//! density radius ([`ShellUpdate::from_header`]). The continuum drops the shell's mass
//! ([`ShellCut`], applied by the splat pass to its packets), the specks draw it.
//!
//! **Deterministic specks.** A size bin of a cluster is the leaf the continuum draws for it
//! ([`speck_leaf`]: uniform along its longer axis — the chord with its density ramp, or the time
//! segment — plus a Gaussian holding `Σ_v` scaled by the bin's speed factor and the other axis as
//! [`super::CHORD_VAR_FACTOR`] half-lengths²), so the mass the continuum hands over inside the
//! shell is the mass the specks draw there. A grain is a fixed point `w ∈ (0,1)³` of the bin's
//! **probability cube**: coordinates `(ξ₀, ξ₁, ζ) = (Φ⁻¹(w₀), Φ⁻¹(w₁), Q(w₂))` with `Q` the exact
//! quantile of `ζ = κ t + ξ₂` (the uniform axis convolved with the Gaussian along it,
//! [`zeta_quantile`]), position `x(t) = mean_j(t) + M_j(t) (ξ₀, ξ₁, ζ)` with `M_j` the map of
//! [`leaf_map`] (the leaf's Cholesky factor in a frame along its uniform axis; without one `M_j`
//! is the Cholesky factor of `Σ_j(t)`, [`cholesky3`]; the moments carry `Σ_v = A Aᵀ`, not the secant
//! matrix `A` itself — under the near-linear Kepler map of young dust `A(t) ∝ t` so the Cholesky
//! factor evolves exactly as the secants do, `specks_follow_the_kepler_flow`). The point process
//! on the cube is realised **hierarchically** on a dyadic tree (round-robin halving of the
//! coordinates, probability `2⁻ᴸ` per node at level `L`): the bin's count `N_j` (quantised to 13
//! bits, [`quantise_count`]) is split in half deterministically while above 2²⁴, then
//! binomially `Bin(n, ½)` from the node's hash ([`child_count`]: a popcount for `n ≤ 64`, an
//! integer normal approximation above — integer arithmetic only, so both mirrors build the same
//! tree), until a node holds one grain, whose offset in the node comes from the leaf's hash. The
//! realisation is exact (a binomial-thinned uniform process on the cube) and the same whatever
//! the camera; only the traversal is camera-dependent: a node is descended only when its image
//! (centre `μ + A ξ̄`, radius `Σ_k ‖A_k‖ · half-extent_k`) meets the shell and is not wholly
//! behind the eye. Why 3D and not the five grain coordinates (dispersion, size, emission time): a
//! five-dimensional grid of a three-dimensional cloud is degenerate — its one-grain cells are
//! `N^{-1/5}` of the extent across while the grains are `N^{-1/3}` apart, so finding the cells
//! that meet a 10 m shell in a 10 km cloud visits `(E/d)²` ≈ 10⁶ mostly empty cells; the 3D
//! cube's one-grain nodes are the grain spacing and the traversal costs a few visits per speck.
//! The grain's size is drawn from the bin's number law `n(s) ∝ s^-q` by its leaf hash, so the
//! drawn cross-sections average to `⟨πs²⟩_j` exactly, [`SpeckCluster::speck_size_um`].
//!
//! **Draw.** A speck is a point source of cross-section `πs²` times the radiance weight
//! ([`radiance_weight`]) at its own position, a Gaussian of σ = [`DUST_SPECKS_SIGMA_PX`] into
//! level 0 of the pyramid ([`scatter_speck`]): `τ = exposure · πs² · w / A_px` (pixel area at the
//! speck's depth from the mvp's Jacobian), counts `⌊τ · inv_unit + u⌉` with a hashed `u` per speck
//! (unbiased however faint), spread over a 5×5 footprint by error diffusion, the colour words and
//! the depth word as the continuum's. Budget [`DUST_SPECKS_BUDGET`] per frame ([`PYR_SPECKS`]
//! counts, [`PYR_SPECKS_OVERFLOW`] the excess). CPU mirror [`specks_tier`], GPU `dust_specks.comp`.
//!
//! **Mirror agreement.** Each mirror places a speck bit-identically from frame to frame (same
//! inputs, same operations: persistence is exact). Across the mirrors the specks are the same
//! (the trees depend on `⌈log₂ N⌉` only, [`count_bound`]) but placed within ~1 px of each
//! other: the inverse normal CDF (`log`, `sqrt`, `exp`, `erf`) differs by a few f32 ulps between
//! the GPU and the CPU, and the offset `L ξ` amplifies that by the cluster's extent over the
//! speck's depth (`gpu_dust_specks_match_reference`: 0.79 px and 1.1 % for a speck 73 m from
//! the eye in a cloud of ~10⁵ m).
use super::{
  DUST_COLOR_SCALE, DUST_COUNT_MAX_PER_ADD, DUST_VIEW_FLOW, DustFlowUniform, DustMoments,
  DustRenderCluster, PYR_FLAGS, PYR_LEVEL_MASK, PYR_TAU_MAX, PYRAMID_TEXEL_WORDS, Packet,
  PyramidLayout, SIZE_BINS, age_color, bin_chord_packet, bin_packet,
  df::{Df, two_prod, two_sum},
  dust_hash32, flow_factor,
  phase::radiance_weight,
  render_slot, sat_add,
  splat::streak_pred,
  unpack_color,
};
use aethervk_oshal_rlib::math::FloatLike;

// ─── Constants (mirror of dust_specks.comp DUST_SPECKS_* / DUST_PYR_SPECKS_*) ──

/// header word: specks drawn this frame (`atomicAdd`). The speck words are 248–255 (the splat's
/// own are 11 and 245–247: `PYR_SPLIT_CHILDREN`, `PYR_SPLIT_OVERFLOW`, `PYR_SPLIT_DROPPED_FLUX`,
/// `PYR_SPLIT_DEPTH`)
pub const PYR_SPECKS: u32 = 248;
/// header word: specks not drawn for the budget (`atomicAdd`)
pub const PYR_SPECKS_OVERFLOW: u32 = 249;
/// header word: walks stopped by the per-walk or the frame's node cap (`atomicAdd`): their
/// specks within the shell are missing (formerly added to [`PYR_SPECKS_OVERFLOW`], which made the
/// host shrink the shell for walk cost — inside the coma to a few metres); word 62, free below
/// the splat's `PYR_EWA_TEXELS` (the level table ends at word 60)
pub const PYR_SPECKS_TRUNCATED: u32 = 62;
/// header word: low word of the 64-bit fixed-point sum of `N_j · G_j(eye)` over every bin of
/// every cluster of the frame, in units of [`N_EYE_UNIT`] m⁻³ (`atomicAdd` with carry into
/// [`PYR_N_EYE_HI`])
pub const PYR_N_EYE: u32 = 250;
/// header word: high word of [`PYR_N_EYE`]
pub const PYR_N_EYE_HI: u32 = 251;
/// header word: f32 bits of the shell radius `d_max` (particle-system local metres) this frame's
/// speck passes draw within, and the continuum cuts out ([`ShellCut`]); host-written, 0 = none
pub const PYR_SPECKS_DMAX: u32 = 252;
/// header word: tree nodes the speck passes visited this frame (`atomicAdd` in chunks of
/// [`DUST_SPECKS_NODE_CHUNK`]): the per-frame work bound, a walk stops once it passes
/// [`DUST_SPECKS_FRAME_NODES`] (counted as overflow)
pub const PYR_SPECKS_NODES: u32 = 253;
/// header word: fixed-point sum of the flux (exposure-scaled, counts of `inv_unit`) the speck
/// passes deposited this frame (`atomicAdd`, saturating): what the continuum gave up
pub const PYR_SPECKS_COUNTS: u32 = 254;
/// header word: the speck budget of the frame (host-written, [`DUST_SPECKS_BUDGET`] by default;
/// the observer and the tests may raise it)
pub const PYR_SPECKS_BUDGET: u32 = 255;
/// m⁻³ per count of the `n_eye` accumulator: 2⁻³⁶ (resolution 1.5e-11 m⁻³, i.e. a shell of
/// ~6e5 m at Ω_px = 1e-6 sr; range 2²⁸ m⁻³ in the 64-bit sum)
pub const N_EYE_UNIT: f64 = 1.455_191_522_836_685e-11; // 2^-36
/// specks a frame draws at most (the header counts the excess)
pub const DUST_SPECKS_BUDGET: u32 = 1 << 20;
/// tree nodes one (cluster, bin) thread visits at most per frame (a safety valve, like
/// `DUST_SPLAT_MAX_TEXELS`): the per-thread work bound by construction; a truncated walk counts
/// one overflow. A bin wholly inside the shell visits ≈ 2 N nodes (N − 1 splits, N leaves), so
/// 2¹⁸ holds 1.3·10⁵ specks per bin — the 6.4·10⁴ of a bin of `specks_and_continuum_conserve_the_flux`
/// with a factor 2 to spare (2¹⁶ truncated it to 2·10⁴); at ~100 flops per node a thread's
/// worst case is ~3·10⁷ flops (tens of ms), the frame's total stays [`DUST_SPECKS_FRAME_NODES`]
pub const DUST_SPECKS_MAX_NODES: u32 = 1 << 18;
/// the `ξ` tolerance of the node culling: the inverse normal CDF is monotone only to its own
/// approximation error (2·10⁻⁶ after the Halley step, a 4·10⁻⁶ step where the tail branch meets
/// the refined central one) plus the f32 rounding of `ξ` (≤ 5.3 · 2⁻²⁴): a leaf's `ξ` may lie
/// that far outside its node's `[Φ⁻¹(lo), Φ⁻¹(lo + width)]`, so the box is widened by it
pub const DUST_SPECKS_XI_TOL: f32 = 1.0e-5;
/// tree nodes all the speck passes of a frame visit at most ([`PYR_SPECKS_NODES`]): eight per
/// speck of the budget, the per-frame work bound by construction (a frame of the GPU path costs
/// at most this many node visits plus one per thread)
pub const DUST_SPECKS_FRAME_NODES: u32 = 8 * DUST_SPECKS_BUDGET;
/// node visits a walk accounts for at once in [`PYR_SPECKS_NODES`] (one atomic per chunk)
pub const DUST_SPECKS_NODE_CHUNK: u32 = 256;
/// σ of a speck's point-spread function (px): the pixel filter
pub const DUST_SPECKS_SIGMA_PX: f32 = 0.5;
/// footprint half-width of a speck in texels (5×5 window: ±2.5 px covers 5σ)
pub const DUST_SPECKS_FOOTPRINT: i64 = 2;
/// the shell shrinks at least by this factor the frame after the budget overflowed (by
/// `(DUST_SPECKS_TARGET · budget / demand)^⅓` when that is smaller, down to [`DUST_SPECKS_SHRINK_MIN`]: the specks
/// scale with the ball's volume)
pub const DUST_SPECKS_SHRINK: f32 = 0.8;
/// the share of the budget an overflowing shell is shrunk to (`(target · budget / demand)^⅓`):
/// headroom, so a shell settled at a shrink does not sit on the budget's edge
pub const DUST_SPECKS_TARGET: f32 = 0.75;
/// the largest single shrink of the shell (×0.01 per readback)
pub const DUST_SPECKS_SHRINK_MIN: f32 = 0.01;
/// the largest growth of the shell per readback, while the frame drew under a quarter of the
/// budget: toward half the budget (`(0.5 · budget / specks)^⅓`), by this factor while no speck is
/// drawn; never past the density radius; the shell holds otherwise. The ×0.8 shrink followed by
/// a jump back to the density radius made the shell oscillate inside the coma (it never settled
/// at 1 km: 81 862 truncated walks); a ×1.25 growth took ~27 readbacks from 1 m to a 400 m shell
pub const DUST_SPECKS_GROW: f32 = 2.0;
/// the shell a perspective view starts from before its first readback (m): small, so a cold
/// start inside the coma cannot overflow; it then grows. The hand-over to the continuum is exact
/// at any radius, so a small shell changes the detail, not the picture's light
pub const DUST_SPECKS_START_M: f32 = 1.0;
/// the shell radius is capped here (m): beyond it, the dust is the continuum's whatever `n_eye`
pub const DUST_SPECKS_DMAX_MAX: f32 = 1.0e5;
/// the walk's stack: a depth-first path of `DUST_SPECKS_MAX_LEVEL + 2` nodes plus the jump
/// start's up to 8 start nodes
pub const DUST_SPECKS_STACK: usize = (DUST_SPECKS_MAX_LEVEL + 2) as usize + 8;
/// the jump start's widening of its probability box (above the quantiles' round trip
/// `|F(Q(w)) − w| ≤ 8·10⁻⁶`, [`zeta_cdf`])
pub const DUST_SPECKS_JUMP_W_TOL: f32 = 2.0e-5;
/// the jump start's deepest split count per axis (cells of 2⁻²⁰: the box widening keeps it
/// wider)
pub const DUST_SPECKS_JUMP_MAX_SPLITS: u32 = 20;
/// deepest tree level (the root is 0): with the count quantised to 13 bits and 2²⁴ as the top of
/// the binomial regime a bin of 10¹³ grains reaches single grains by level ~45
pub const DUST_SPECKS_MAX_LEVEL: u32 = 60;
/// the count regime boundaries of the tree: deterministic halving above, binomial splits below
pub const DUST_SPECKS_DETERMINISTIC_ABOVE: u32 = 1 << 24;
/// significant bits a bin's count is quantised to ([`quantise_count`])
pub const DUST_SPECKS_COUNT_BITS: u32 = 13;
/// a size-bin count below this is realised as `Bernoulli(N)` (the fractional grain)
const _: () = assert!(DUST_SPECKS_COUNT_BITS <= 24);

/// The default of the differential size law exponent `n(s) ∝ s^-q`
/// (`ParticleSystemEmitParams::size_q`, 3.5). The specks take each system's `q` from their push
/// constants ([`DustSpecksPushConstants::size_q`]); the continuum's bin layout stays the q = 3.5
/// one whatever the parameter (see `ParticleSystemEmitParams::size_q`).
pub const SIZE_Q: f64 = super::SIZE_POWER_Q;
/// The default size range factor `F` (`s_ref / F .. s_ref · F`,
/// `ParticleSystemEmitParams::size_range`, 10). The specks read each cluster's own `F` from its
/// moments ([`moments_size_range`]).
pub const SIZE_RANGE: f64 = super::SIZE_RANGE_FACTOR;
/// `q` travels in the push constants' colour word as `round(32 q)` in bits 24..=31
/// ([`pack_color_q`]): a step of 1/32 (3.5 exact), the Model's range 2–5 fits the byte
pub const SIZE_Q_STEPS: f32 = 32.0;

const AU_M_F32: f32 = 1.495_978_707e11;
const PI32: f32 = core::f32::consts::PI;
const PI64: f64 = core::f64::consts::PI;

#[inline]
fn sqrtf(x: f32) -> f32 {
  <f32 as FloatLike>::sqrt(x)
}
#[inline]
fn expf(x: f32) -> f32 {
  <f32 as FloatLike>::exp(x)
}
#[inline]
fn lnf(x: f32) -> f32 {
  <f32 as FloatLike>::ln(x)
}
#[inline]
fn floorf(x: f32) -> f32 {
  <f32 as FloatLike>::floor(x)
}
#[inline]
fn powf(x: f32, y: f32) -> f32 {
  <f32 as FloatLike>::pow(x, y)
}
#[inline]
fn fabs(x: f32) -> f32 {
  if x < 0.0 { -x } else { x }
}
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
  [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn norm3(a: [f32; 3]) -> f32 {
  sqrtf(dot3(a, a))
}

// ─── Push constants ──────────────────────────────────────────────────────────

/// `dust_specks.comp` push constants. 128 bytes. The per-system size law rides here (`s_ref` and
/// the fade in [`Self::size_fade`], `q` in the top byte of [`Self::color`]; `F` is each
/// cluster's own, from its moments); the per-view scalars (the shell radius, the budget, the view
/// flags, the flow clock, the phase table) ride in the pyramid header ([`PYR_SPECKS_DMAX`],
/// [`PYR_SPECKS_BUDGET`], `PYR_FLAGS`, `PYR_T_HI/LO`, `PYR_PHASE_LUT`): the 128 B are full.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DustSpecksPushConstants {
  /// `DustMomentsBuffer` of the tier (compact, index = live-range offset)
  pub moments: u64,
  /// `DustRenderBuffer` of the tier (ring slots: the speck identity, predecessor words)
  pub render: u64,
  /// the view's pyramid (header + levels)
  pub pyramid: u64,
  pub live_count: u32,
  /// f16 pair ([`pack_size_fade`]): low half the reference grain radius `s_ref` (µm, the
  /// configured diameter / 2), high half the tier's fade ratio `max_age / TTL` (0 = the tier does
  /// not fade): the count uses the **unfaded** flux, so it is a constant of the cluster's life
  pub size_fade: u32,
  /// exposure: the stream colour's alpha (as the splat's)
  pub exposure: f32,
  /// counts per unit of exposure-scaled optical depth × px² (as the splat's)
  pub inv_unit: f32,
  /// stream colour, 8 bits per channel (`r | g << 8 | b << 16`), and the size law exponent as
  /// `round(32 q) << 24` ([`pack_color_q`]; 0 = the default 3.5)
  pub color: u32,
  /// geometric albedo of the dust (the radiance weight); `≤ 0` draws the unweighted cross-section
  pub albedo: f32,
  /// **eye-relative** particle-system metres → clip (`Proj · ViewRot · Scale`, no translation:
  /// the speck offsets `x − eye` are formed in df64 and projected without a cancellation against
  /// a large translation; `DustDrawCall::mvp_params_eye_relative`)
  pub mvp: [f32; 16],
  /// eye position in particle-system local metres (the shell's centre); `w` = the system's TTL
  /// (s), the age clock of the hue ramp
  pub eye_local: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustSpecksPushConstants>() == 128);

impl DustSpecksPushConstants {
  /// the reference grain radius (µm) from [`Self::size_fade`]
  pub fn s_ref_um(&self) -> f32 {
    f16_to_f32((self.size_fade & 0xFFFF) as u16)
  }
  /// the tier's fade ratio `max_age / TTL` from [`Self::size_fade`] (0 = no fade)
  pub fn fade_ratio(&self) -> f32 {
    f16_to_f32((self.size_fade >> 16) as u16)
  }
  /// the size law exponent `q` from the colour word's top byte (0 → [`SIZE_Q`])
  pub fn size_q(&self) -> f32 {
    color_q(self.color)
  }
}

/// The colour word with `q` in its top byte: `round(32 q)` clamped to 1..=255 (q ∈ (0, 8)).
pub fn pack_color_q(color: u32, q: f32) -> u32 {
  let b = if q.is_finite() && q > 0.0 {
    ((q * SIZE_Q_STEPS + 0.5) as u32).clamp(1, 255)
  } else {
    0
  };
  (color & 0x00FF_FFFF) | (b << 24)
}

/// `q` from a colour word (mirror of the shader's `dust_specks_q`): `(w >> 24) / 32`, 0 → 3.5.
pub fn color_q(w: u32) -> f32 {
  let b = w >> 24;
  if b == 0 {
    SIZE_Q as f32
  } else {
    b as f32 / SIZE_Q_STEPS
  }
}

/// The size range factor `F` of a cluster from its moments: the ratio of the stored speed
/// factors of the last and the first size edge (`√F / (1/√F)`), 1 when they are missing (a
/// culled record). Mirror of `dust_specks_range`.
pub fn moments_size_range(m: &DustMoments) -> f32 {
  let (a, b) = (m.edges[0][3], m.edges[SIZE_BINS as usize][3]);
  let f = b / a;
  if a > 0.0 && f.is_finite() && f >= 1.0 {
    f
  } else {
    1.0
  }
}

/// [`DustSpecksPushConstants::size_fade`] from the reference grain radius (µm) and the tier's
/// fade ratio (`DustFrame::ttl[0] / ttl_s` when `ttl[2] ≤ 0.5`, else 0).
pub fn pack_size_fade(s_ref_um: f32, fade_ratio: f32) -> u32 {
  pack_f16_pair(s_ref_um, fade_ratio)
}

/// Two f32 as an f16 pair, `lo | hi << 16` (`packHalf2x16`).
pub fn pack_f16_pair(lo: f32, hi: f32) -> u32 {
  f32_to_f16(lo) as u32 | (f32_to_f16(hi) as u32) << 16
}

/// The f16 pair back (`unpackHalf2x16`).
pub fn unpack_f16_pair(w: u32) -> (f32, f32) {
  (
    f16_to_f32((w & 0xFFFF) as u16),
    f16_to_f32((w >> 16) as u16),
  )
}

/// f32 → IEEE binary16 bits, round to nearest even (`packHalf2x16`).
pub fn f32_to_f16(x: f32) -> u16 {
  let b = x.to_bits();
  let sign = ((b >> 16) & 0x8000) as u16;
  let exp = ((b >> 23) & 0xFF) as i32;
  let man = b & 0x7F_FFFF;
  if exp == 0xFF {
    // inf / nan
    return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
  }
  let e = exp - 127 + 15;
  if e >= 0x1F {
    return sign | 0x7C00;
  }
  if e <= 0 {
    if e < -10 {
      return sign;
    }
    // subnormal: shift the implicit one in
    let m = man | 0x80_0000;
    let shift = (14 - e) as u32;
    let half = m >> shift;
    let rem = m & ((1 << shift) - 1);
    let mid = 1u32 << (shift - 1);
    let half = if rem > mid || (rem == mid && (half & 1) == 1) {
      half + 1
    } else {
      half
    };
    return sign | half as u16;
  }
  let half = ((e as u32) << 10) | (man >> 13);
  let rem = man & 0x1FFF;
  let half = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
    half + 1
  } else {
    half
  };
  sign | half as u16
}

/// IEEE binary16 bits → f32 (`unpackHalf2x16`), exact.
pub fn f16_to_f32(h: u16) -> f32 {
  let sign = ((h & 0x8000) as u32) << 16;
  let exp = ((h >> 10) & 0x1F) as u32;
  let man = (h & 0x3FF) as u32;
  let bits = if exp == 0 {
    if man == 0 {
      sign
    } else {
      // subnormal: normalise
      let mut m = man;
      let mut e = 1i32;
      while m & 0x400 == 0 {
        m <<= 1;
        e -= 1;
      }
      sign | (((e + 127 - 15) as u32) << 23) | ((m & 0x3FF) << 13)
    }
  } else if exp == 0x1F {
    sign | 0x7F80_0000 | (man << 13)
  } else {
    sign | ((exp + 127 - 15) << 23) | (man << 13)
  };
  f32::from_bits(bits)
}

// ─── Count law ───────────────────────────────────────────────────────────────

/// `∫ t^a dt` over `[t0, t1]` (`ln(t1/t0)` at `a = −1`).
fn power_integral(t0: f64, t1: f64, a: f64) -> f64 {
  let e = a + 1.0;
  if (if e < 0.0 { -e } else { e }) < 1e-9 {
    <f64 as FloatLike>::ln(t1 / t0)
  } else {
    (<f64 as FloatLike>::pow(t1, e) - <f64 as FloatLike>::pow(t0, e)) / e
  }
}

/// f32 mirror of [`power_integral`] (`dust_specks_power_integral`).
fn power_integral_f32(t0: f32, t1: f32, a: f32) -> f32 {
  let e = a + 1.0;
  if fabs(e) < 1e-6 {
    lnf(t1 / t0)
  } else {
    (powf(t1, e) - powf(t0, e)) / e
  }
}

/// The speed factor `√(β_j/β)` of size-bin edge `j` for a range factor `F`: equally spaced from
/// `1/√F` to `√F` (the moments' [`DustMoments::edge_factor`], here from `F` alone).
pub fn edge_factor(range: f64, j: usize) -> f64 {
  let f = range.max(1.0);
  let (lo, hi) = (
    1.0 / <f64 as FloatLike>::sqrt(f),
    <f64 as FloatLike>::sqrt(f),
  );
  lo + (hi - lo) * (j as f64 / SIZE_BINS as f64)
}

/// The size range of bin `j` as `s / s_ref`: `[1/f_{j+1}², 1/f_j²]` (`β ∝ 1/s`; edge 0 is the
/// large-grain end).
pub fn bin_size_range(range: f64, j: usize) -> (f64, f64) {
  let (f0, f1) = (edge_factor(range, j), edge_factor(range, j + 1));
  (1.0 / (f1 * f1), 1.0 / (f0 * f0))
}

/// The mass fraction `w_j` of every size bin for the law `n(s) ∝ s^-q` over `s_ref/F .. s_ref·F`
/// (`dm ∝ s^{3−q} ds`), sum 1.
pub fn bin_mass_weights(q: f64, range: f64) -> [f64; SIZE_BINS as usize] {
  let f = range.max(1.0);
  let mut w = [1.0 / SIZE_BINS as f64; SIZE_BINS as usize];
  if f <= 1.0 + 1e-9 {
    return w; // one size: the bins coincide
  }
  let total = power_integral(1.0 / f, f, 3.0 - q);
  for (j, wj) in w.iter_mut().enumerate() {
    let (t0, t1) = bin_size_range(range, j);
    *wj = power_integral(t0, t1, 3.0 - q) / total;
  }
  w
}

/// The cross-section fraction of bin `j` (what the continuum draws as `F_j / F`; `1/16` each for
/// `q = 3.5`, the spacing the bin edges were made for).
pub fn bin_xsec_weight(q: f64, range: f64, j: usize) -> f64 {
  let f = range.max(1.0);
  if f <= 1.0 + 1e-9 {
    return 1.0 / SIZE_BINS as f64;
  }
  let (t0, t1) = bin_size_range(range, j);
  power_integral(t0, t1, 2.0 - q) / power_integral(1.0 / f, f, 2.0 - q)
}

/// The mean grain cross-section `⟨πs²⟩_j` (m²) of bin `j` for a reference radius `s_ref` (µm).
pub fn bin_mean_cross_section_m2(s_ref_um: f64, q: f64, range: f64, j: usize) -> f64 {
  let (t0, t1) = bin_size_range(range, j);
  let s = s_ref_um * 1e-6;
  if t1 - t0 <= 1e-9 * t1 {
    return PI64 * s * s * t0 * t0; // one size
  }
  PI64 * s * s * power_integral(t0, t1, 2.0 - q) / power_integral(t0, t1, -q)
}

/// The mean grain mass `⟨(4/3)πs³ρ⟩_j` (g) of bin `j` for `s_ref` (µm) and `ρ` (g/cm³).
pub fn bin_mean_grain_mass_g(
  s_ref_um: f64,
  density_gcm3: f64,
  q: f64,
  range: f64,
  j: usize,
) -> f64 {
  let (t0, t1) = bin_size_range(range, j);
  let s_cm = s_ref_um * 1e-4;
  let v = (4.0 / 3.0) * PI64 * s_cm * s_cm * s_cm * density_gcm3;
  if t1 - t0 <= 1e-9 * t1 {
    return v * t0 * t0 * t0;
  }
  v * power_integral(t0, t1, 3.0 - q) / power_integral(t0, t1, -q)
}

/// The grain mass `(4/3)π s³ ρ` (g) of a radius `s` (µm) and density `ρ` (g/cm³).
pub fn grain_mass_g(s_um: f64, density_gcm3: f64) -> f64 {
  let s_cm = s_um * 1e-4;
  (4.0 / 3.0) * PI64 * s_cm * s_cm * s_cm * density_gcm3
}

/// `N_j = m_cluster · w_j / ⟨m_grain⟩_j`: the grains of bin `j` of a cluster of mass `m` (g).
pub fn bin_count_from_mass(
  mass_g: f64,
  density_gcm3: f64,
  s_ref_um: f64,
  q: f64,
  range: f64,
  j: usize,
) -> f64 {
  mass_g * bin_mass_weights(q, range)[j]
    / bin_mean_grain_mass_g(s_ref_um, density_gcm3, q, range, j)
}

/// `N_j = F_j / ⟨πs²⟩_j`: the grains of bin `j` from the bin's cross-section `F_j` (m², the
/// continuum's `flux / 16`). The same number as [`bin_count_from_mass`]
/// (`counts_from_mass_and_flux_agree`), from what the render side has.
pub fn bin_count_from_flux(flux_bin_m2: f64, s_ref_um: f64, q: f64, range: f64, j: usize) -> f64 {
  flux_bin_m2 / bin_mean_cross_section_m2(s_ref_um, q, range, j)
}

/// f32 mirror of [`bin_mean_cross_section_m2`] (`dust_specks_mean_xsec`): the render side's
/// count, `N_j = F_j / ⟨πs²⟩_j`.
pub fn bin_mean_cross_section_f32(s_ref_um: f32, q: f32, range: f32, j: u32) -> f32 {
  let f = range.max(1.0);
  let (lo, hi) = (1.0 / sqrtf(f), sqrtf(f));
  let fj = lo + (hi - lo) * (j as f32 / SIZE_BINS as f32);
  let fj1 = lo + (hi - lo) * ((j + 1) as f32 / SIZE_BINS as f32);
  let (t0, t1) = (1.0 / (fj1 * fj1), 1.0 / (fj * fj));
  let s = s_ref_um * 1e-6;
  if t1 - t0 <= 1e-6 * t1 {
    return PI32 * s * s * t0 * t0; // one size
  }
  PI32 * s * s * power_integral_f32(t0, t1, 2.0 - q) / power_integral_f32(t0, t1, -q)
}

/// The age fade of the oldest tier, mirror of `evaluate_cluster`'s: `clamp(10 (1 − age / max_age))`
/// with `max_age = fade_ratio · ttl`; 1 when the tier does not fade (`fade_ratio ≤ 0`).
pub fn age_fade(age_s: f32, ttl_s: f32, fade_ratio: f32) -> f32 {
  let max_age = fade_ratio * ttl_s;
  if !(fade_ratio > 0.0) || !(max_age > 0.0) {
    return 1.0;
  }
  ((1.0 - age_s / max_age) * 10.0).clamp(0.0, 1.0)
}

// ─── Regime boundary ─────────────────────────────────────────────────────────

/// The lower-triangular Cholesky factor `L` of a covariance (xx xy xz yy yz zz) as three
/// columns, `Σ = L Lᵀ`: `A(t)` of the specks. A floor of `1e-12 · tr Σ` (+ 1e-20) on the
/// diagonal keeps a degenerate packet (a just-emitted sample: zero spread) factorisable; a
/// negative pivot (rounding) zeroes its column. Mirror of `dust_specks_cholesky`.
pub fn cholesky3(cov: [f32; 6]) -> [[f32; 3]; 3] {
  let tr = cov[0] + cov[3] + cov[5];
  let floor = (tr * 1e-12).max(1e-20);
  let (sxx, sxy, sxz, syy, syz, szz) = (
    cov[0] + floor,
    cov[1],
    cov[2],
    cov[3] + floor,
    cov[4],
    cov[5] + floor,
  );
  let l00 = sqrtf(sxx.max(0.0));
  let (l10, l20) = if l00 > 0.0 {
    (sxy / l00, sxz / l00)
  } else {
    (0.0, 0.0)
  };
  let d1 = syy - l10 * l10;
  let l11 = if d1 > 0.0 { sqrtf(d1) } else { 0.0 };
  let l21 = if l11 > 0.0 {
    (syz - l20 * l10) / l11
  } else {
    0.0
  };
  let d2 = szz - l20 * l20 - l21 * l21;
  let l22 = if d2 > 0.0 { sqrtf(d2) } else { 0.0 };
  [[l00, l10, l20], [0.0, l11, l21], [0.0, 0.0, l22]]
}

/// `L · ξ` for the columns of [`cholesky3`].
#[inline]
pub fn chol_apply(l: &[[f32; 3]; 3], xi: [f32; 3]) -> [f32; 3] {
  [
    l[0][0] * xi[0] + l[1][0] * xi[1] + l[2][0] * xi[2],
    l[0][1] * xi[0] + l[1][1] * xi[1] + l[2][1] * xi[2],
    l[0][2] * xi[0] + l[1][2] * xi[1] + l[2][2] * xi[2],
  ]
}

/// The Gaussian density `G(x; μ, Σ)` (m⁻³) from the Cholesky factor: forward substitution
/// `L y = x − μ`, `exp(−½|y|²) / ((2π)^{3/2} Π L_kk)`. 0 when `L` is singular. Mirror of
/// `dust_specks_density`.
pub fn gaussian_density(l: &[[f32; 3]; 3], mean: [f32; 3], x: [f32; 3]) -> f32 {
  let d = sub3(x, mean);
  let (l00, l11, l22) = (l[0][0], l[1][1], l[2][2]);
  if !(l00 > 0.0 && l11 > 0.0 && l22 > 0.0) {
    return 0.0;
  }
  let y0 = d[0] / l00;
  let y1 = (d[1] - l[0][1] * y0) / l11;
  let y2 = (d[2] - l[0][2] * y0 - l[1][2] * y1) / l22;
  let q = y0 * y0 + y1 * y1 + y2 * y2;
  let det_sqrt = l00 * l11 * l22;
  let norm = 15.749_609_945_722_42; // (2π)^{3/2}
  expf(-0.5 * q) / (norm * det_sqrt)
}

/// The pixel solid angle (sr) of a perspective camera: `(2 tan(fov_y/2) / H) · (2 tan(fov_y/2) ·
/// aspect / W)`, the footprint of a pixel at unit depth. The regime boundary's `Ω_px`.
pub fn pixel_solid_angle_sr(fov_y_rad: f64, aspect: f64, width: u32, height: u32) -> f64 {
  let t = <f64 as FloatLike>::tan(0.5 * fov_y_rad);
  (2.0 * t / height.max(1) as f64) * (2.0 * t * aspect / width.max(1) as f64)
}

/// `d_max = (3 / (n_eye · Ω_px))^{1/3}` (m): the radius of the camera-centred shell inside which a
/// pixel's ray meets less than one grain (`∫₀ᵈ n Ω r² dr = n Ω d³/3 = 1`). `+∞` when either
/// factor is 0 (callers cap it at [`DUST_SPECKS_DMAX_MAX`]; an orthographic camera has no pixel
/// solid angle and gets `Ω_px = 0` → no shell, see [`ShellUpdate`]).
pub fn shell_radius_m(n_eye_m3: f64, omega_px_sr: f64) -> f64 {
  let d = n_eye_m3 * omega_px_sr;
  if !(d > 0.0) || !d.is_finite() {
    return f64::INFINITY;
  }
  <f64 as FloatLike>::pow(3.0 / d, 1.0 / 3.0)
}

/// Host-side number density at `eye` (m⁻³) over a tier's moments: `Σ_clusters Σ_bins N_j ·
/// G(eye; μ_j, Σ_j)`, with the counts from the **unfaded** flux (`s_ref_um`, `q`, `fade_ratio`
/// / `ttl_s` as the push constants carry them; `F` from each cluster's moments). The CPU mirror
/// of the GPU's `PYR_N_EYE` accumulation.
#[allow(clippy::too_many_arguments)]
pub fn number_density_at(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  live_count: u32,
  eye: [f32; 3],
  s_ref_um: f32,
  q: f32,
  ttl_s: f32,
  fade_ratio: f32,
) -> f64 {
  let n = (live_count as usize).min(moments.len()).min(render.len());
  let mut sum = 0.0f64;
  for i in 0..n {
    let m = &moments[i];
    if !(m.flux() > 0.0) {
      continue;
    }
    let pred = streak_pred(render, i).map(|p| &moments[p]);
    for b in 0..SIZE_BINS as usize {
      let seg_t = bin_time_segment(m, pred, b);
      let sc = SpeckCluster::new(m, b, seg_t, 0, s_ref_um, q, ttl_s, fade_ratio);
      sum += sc.density_count_at(eye) as f64;
    }
  }
  sum
}

/// What the host does with a frame's readback for the shell: the next frame's `d_max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellUpdate {
  /// the measured density at the eye (m⁻³)
  pub n_eye_m3: f64,
  /// specks drawn / not drawn (the budget) in the measured frame
  pub specks: u32,
  pub overflow: u32,
  /// walks stopped by a node cap ([`PYR_SPECKS_TRUNCATED`]; not demand)
  pub truncated: u32,
  /// tree nodes the speck passes visited ([`PYR_SPECKS_NODES`])
  pub nodes: u32,
  /// the shell radius for the next frame (m): the density formula capped at
  /// [`DUST_SPECKS_DMAX_MAX`]; with a previous radius, at most the previous one shrunk after an
  /// overflow (`(DUST_SPECKS_TARGET · budget / demand)^⅓`, within [[`DUST_SPECKS_SHRINK_MIN`], [`DUST_SPECKS_SHRINK`]]),
  /// grown toward half the budget (at most [`DUST_SPECKS_GROW`]) under a quarter of it, held
  /// otherwise; 0 without a
  /// pixel solid angle (orthographic)
  pub d_max_m: f32,
}

impl ShellUpdate {
  /// Reads the speck words of a finished frame's header (`PYR_N_EYE*`, `PYR_SPECKS*`) and
  /// derives the next shell radius from the view's pixel solid angle and the previous radius.
  pub fn from_header(header: &[u32], omega_px_sr: f64, prev_d_max_m: f32) -> Self {
    let lo = header[PYR_N_EYE as usize] as u64;
    let hi = header[PYR_N_EYE_HI as usize] as u64;
    let n_eye_m3 = ((hi << 32) | lo) as f64 * N_EYE_UNIT;
    let specks = header[PYR_SPECKS as usize];
    let overflow = header[PYR_SPECKS_OVERFLOW as usize];
    let mut d = if omega_px_sr > 0.0 {
      shell_radius_m(n_eye_m3, omega_px_sr).min(DUST_SPECKS_DMAX_MAX as f64) as f32
    } else {
      0.0
    };
    if prev_d_max_m > 0.0 {
      let budget = header[PYR_SPECKS_BUDGET as usize];
      let demand = specks as f64 + overflow as f64;
      let step = if overflow > 0 {
        ((DUST_SPECKS_TARGET as f64 * budget as f64 / demand.max(1.0)).cbrt() as f32)
          .clamp(DUST_SPECKS_SHRINK_MIN, DUST_SPECKS_SHRINK)
      } else if budget > 0 && specks < budget / 4 {
        // toward half the budget (the specks scale with the ball's volume), at most ×GROW; ×GROW
        // while none is drawn (no measure of the density's reach yet)
        if specks == 0 {
          DUST_SPECKS_GROW
        } else {
          ((0.5 * budget as f64 / specks as f64).cbrt() as f32).clamp(1.0, DUST_SPECKS_GROW)
        }
      } else {
        1.0
      };
      d = d.min(prev_d_max_m * step);
    }
    if !d.is_finite() {
      d = 0.0;
    }
    Self {
      n_eye_m3,
      specks,
      overflow,
      truncated: header[PYR_SPECKS_TRUNCATED as usize],
      nodes: header[PYR_SPECKS_NODES as usize],
      d_max_m: d,
    }
  }
}

// ─── Deterministic specks ────────────────────────────────────────────────────

/// `dust_hash32(h ^ salt_k)`, `salt_k = 0x6a09e667 + k · 0x9E3779B9`: the `k`-th hash word of a
/// node (no table: a dynamically indexed constant table is a GPU parity hazard). Mirror of
/// `dust_specks_word`.
#[inline]
pub fn hash_word(h: u32, k: usize) -> u32 {
  dust_hash32(h ^ 0x6a09_e667u32.wrapping_add((k as u32).wrapping_mul(0x9E37_79B9)))
}

/// A uniform in `[0, 1)` from a hash word (its high 24 bits). Mirror of `dust_u01`.
#[inline]
pub fn u01(h: u32) -> f32 {
  (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// `⌊√n⌉` for `n ≤ 2³²−1`, integer only (bit by bit): the same on both mirrors. Mirror of
/// `dust_specks_isqrt`.
pub fn isqrt_u32(n: u32) -> u32 {
  let mut x = n;
  let mut res = 0u32;
  let mut bit = 1u32 << 30;
  while bit > n {
    bit >>= 2;
  }
  while bit != 0 {
    if x >= res + bit {
      x -= res + bit;
      res = (res >> 1) + bit;
    } else {
      res >>= 1;
    }
    bit >>= 2;
  }
  res
}

/// The count of the left child of a node of `n ≤ 2²⁴` grains split in two halves of equal
/// probability, `Bin(n, ½)` from the node's hash `h`: a popcount of `n` hash bits for `n ≤ 64`,
/// else the integer normal approximation `n/2 + (8√n)·g / 4096` with `g` = the sum of 12 hash
/// bytes less 1530, an Irwin–Hall normal of variance `12 · (256² − 1)/12 = 65535` (σ = 256), so
/// the offset has σ = 8√n · 256 / 4096 = √n / 2, the binomial's; `8√n` is `isqrt(64 n)` (the
/// floor of `√n` alone lost up to 18 % of the variance just below a square). Integer arithmetic
/// only: the tree is the same on both mirrors. Mirror of `dust_specks_child_count`.
pub fn child_count(n: u32, h: u32) -> u32 {
  if n == 0 {
    return 0;
  }
  let h0 = hash_word(h, 0);
  if n <= 32 {
    let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
    return (h0 & mask).count_ones();
  }
  let h1 = hash_word(h, 1);
  if n <= 64 {
    let m = n - 32;
    let mask = if m == 32 { u32::MAX } else { (1u32 << m) - 1 };
    return h0.count_ones() + (h1 & mask).count_ones();
  }
  let h2 = hash_word(h, 2);
  let mut s = 0i32;
  for w in [h0, h1, h2] {
    s += (w & 0xFF) as i32
      + ((w >> 8) & 0xFF) as i32
      + ((w >> 16) & 0xFF) as i32
      + ((w >> 24) & 0xFF) as i32;
  }
  // the odd grain goes to a hashed side
  let base = (n + (h2 >> 31)) / 2;
  let delta = (isqrt_u32(n << 6) as i32 * (s - 1530)) / 4096;
  (base as i32 + delta).clamp(0, n as i32) as u32
}

/// A bin's grain count as `mantissa · 2^exponent` with [`DUST_SPECKS_COUNT_BITS`] significant
/// bits (the physical count is good to no more than that; the top of the tree halves it exactly,
/// and small float differences of the two mirrors' `N` rarely change the quantised value). A
/// count below 1 is one grain with probability `N` (hashed by `h`). Mirror of
/// `dust_specks_quantise`.
pub fn quantise_count(n: f32, h: u32) -> (u32, u32) {
  if !(n > 0.0) || !n.is_finite() {
    return (0, 0);
  }
  if n < 1.0 {
    return (if u01(hash_word(h, 7)) < n { 1 } else { 0 }, 0);
  }
  // floor(log2 n) from the exponent bits
  let e2 = ((n.to_bits() >> 23) & 0xFF) as i32 - 127;
  let shift = (e2 - (DUST_SPECKS_COUNT_BITS as i32 - 1)).max(0) as u32;
  let scale = f32::from_bits(((127 - shift as i32) as u32) << 23); // 2^-shift
  let mant = floorf(n * scale + 0.5) as u32;
  if mant >= 1 << DUST_SPECKS_COUNT_BITS {
    // rounding carried into the next bit
    (mant >> 1, shift + 1)
  } else {
    (mant, shift)
  }
}

/// `(mant, exp)` of [`quantise_count`] with the exponent used only above
/// [`DUST_SPECKS_DETERMINISTIC_ABOVE`]: below it the count is the plain integer `mant << exp`
/// (≤ 2²⁴, the binomial regime of the tree). Mirror of `dust_specks_normalise`.
pub fn normalise_count(c: (u32, u32)) -> (u32, u32) {
  let (mant, exp) = c;
  if exp == 0 {
    return c;
  }
  // mant < 2^13, so mant << exp ≤ 2^24 iff exp ≤ 11 (or mant small)
  let n = (mant as u64) << exp;
  if n <= DUST_SPECKS_DETERMINISTIC_ABOVE as u64 {
    (n as u32, 0)
  } else {
    c
  }
}

/// The tree's bound and the leaves' keep probability for a bin of `n` grains:
/// `(N_max, p) = (2^⌈log₂ n⌉, n / N_max)` (`n ≤ 1`: `(1, n)`), `N_max` normalised like
/// [`normalise_count`]. The tree realises `N_max` points and each leaf is kept with probability
/// `p` by its hash: the count is `Bin(N_max, p)`, mean `n`. Why not the quantised `n` itself: the
/// two mirrors evaluate `n` with different `pow`/`log`/`exp` ulps, and at 10⁹ grains a 13-bit
/// quantum is ~10⁵ grains, so ~1–10 % of the bins got a different quantised count and with it a
/// wholly different tree (the GPU parity's 49 of 948 neighbourhoods); with the bound, the tree
/// depends on `⌈log₂ n⌉` only (it changes when `n` crosses a power of two: a relative float error
/// in ~10⁻⁶ of the bins) and a float difference in `n` flips only the leaves whose keep hash lies
/// within it of the threshold. Mirror of `dust_specks_bound`.
pub fn count_bound(n: f32) -> ((u32, u32), f32) {
  if !(n > 0.0) || !n.is_finite() {
    return ((0, 0), 0.0);
  }
  if n <= 1.0 {
    return ((1, 0), n);
  }
  let bits = n.to_bits();
  let e2 = ((bits >> 23) & 0xFF) as i32 - 127;
  let k = if bits & 0x7F_FFFF == 0 { e2 } else { e2 + 1 } as u32;
  let p = n * f32::from_bits((127 - k as i32) as u32 * (1 << 23));
  (normalise_count((1, k)), p)
}

/// `Φ⁻¹(p)`, the standard normal quantile: Acklam's rational approximation (relative error
/// 1.2e-9 with f64 coefficients) and, in the central branch, one Halley step on
/// `Φ(x) − p` with the mirrors' own `erf`: the central rational function cancels (its terms reach
/// 276 for a result of 2) and with the coefficients rounded to f32 it is 1.9e-4 off at p = 0.025
/// even in f64 arithmetic; after the step the error is 2e-6 (emulated in f32 over the test's
/// grid: `Φ(x)` back within 2.4e-7). The tails (|x| > 1.97) are not refined: there the erf's
/// absolute error (1.5e-7) is a large relative error of `Φ`. `p` is clamped to
/// `[2⁻²⁴, 1 − 2⁻²⁴]` (`|ξ| ≤ 5.3`). Mirror of `dust_specks_inv_norm_cdf`.
pub fn inv_norm_cdf(p: f32) -> f32 {
  const A: [f32; 6] = [
    -3.969_683_028_665_376e1,
    2.209_460_984_245_205e2,
    -2.759_285_104_469_687e2,
    1.383_577_518_672_690e2,
    -3.066_479_806_614_716e1,
    2.506_628_277_459_239,
  ];
  const B: [f32; 5] = [
    -5.447_609_879_822_406e1,
    1.615_858_368_580_409e2,
    -1.556_989_798_598_866e2,
    6.680_131_188_771_972e1,
    -1.328_068_155_288_572e1,
  ];
  const C: [f32; 6] = [
    -7.784_894_002_430_293e-3,
    -3.223_964_580_411_365e-1,
    -2.400_758_277_161_838,
    -2.549_732_539_343_734,
    4.374_664_141_464_968,
    2.938_163_982_698_783,
  ];
  const D: [f32; 4] = [
    7.784_695_709_041_462e-3,
    3.224_671_290_700_398e-1,
    2.445_134_137_142_996,
    3.754_408_661_907_416,
  ];
  let p = p.clamp(1.0 / 16_777_216.0, 1.0 - 1.0 / 16_777_216.0);
  let p_low = 0.024_25f32;
  if p < p_low {
    let q = sqrtf(-2.0 * lnf(p));
    (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
      / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
  } else if p <= 1.0 - p_low {
    let q = p - 0.5;
    let r = q * q;
    let x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
      / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    // Halley: e = Φ(x) − p, u = e / φ(x), x ← x − u / (1 + x u / 2)
    let e = 0.5 * (1.0 + super::erf(x * core::f32::consts::FRAC_1_SQRT_2)) - p;
    let u = e * 2.506_628_3 * expf(0.5 * x * x);
    x - u / (1.0 + 0.5 * x * u)
  } else {
    let q = sqrtf(-2.0 * lnf(1.0 - p));
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
      / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
  }
}

/// The leaf of size bin `b` the continuum draws for a cluster whose bin time segment is `seg_t`
/// (`subpackets_swept`'s single-piece, unswept form): the longer of the bin's chord and its time
/// segment is the uniform axis (`Packet::seg`, from `Packet::mean`, density ramp `Packet::rho`),
/// the other a Gaussian of [`super::CHORD_VAR_FACTOR`] half-lengths² in `Packet::cov` with the bin's
/// `f² Σ_v`: [`bin_chord_packet`] when the chord is longer (old dust), [`bin_packet`] when the
/// time segment is (young dust). The specks realise exactly this distribution, so the mass the
/// continuum hands over inside the shell is the mass the specks draw there. (The cone pieces of a
/// widening sample and the on-screen time sweeps are view- or width-dependent refinements of the
/// same cell; the specks keep the single leaf, the cell's own shape.) Mirror of `dust_specks_leaf`.
pub fn speck_leaf(m: &DustMoments, b: usize, seg_t: [f32; 3]) -> Packet {
  let chord = sub3(m.edge(b + 1), m.edge(b));
  if dot3(seg_t, seg_t) > dot3(chord, chord) {
    bin_packet(m, b, seg_t)
  } else {
    bin_chord_packet(m, b, seg_t)
  }
}

/// The time segment of bin `b`: to the same bin of the stream predecessor (`subpackets_swept`'s
/// `seg_t`), zero without one.
pub fn bin_time_segment(m: &DustMoments, pred: Option<&DustMoments>, b: usize) -> [f32; 3] {
  match pred {
    Some(p) => sub3(
      scale3(add3(p.edge(b), p.edge(b + 1)), 0.5),
      scale3(add3(m.edge(b), m.edge(b + 1)), 0.5),
    ),
    None => [0.0; 3],
  }
}

/// The position `t ∈ [0, 1]` along a leaf's uniform axis of probability `u`: the inverse CDF of
/// the linear density `ρ₀ + (ρ₁ − ρ₀) t` (mean 1), `F(t) = ρ₀ t + (ρ₁ − ρ₀) t²/2`, solved in the
/// form without cancellation `t = 2u / (ρ₀ + √(ρ₀² + 2 (ρ₁ − ρ₀) u))`. Mirror of
/// `dust_specks_ramp`.
pub fn ramp_t(u: f32, rho: [f32; 2]) -> f32 {
  let (r0, r1) = (rho[0], rho[1]);
  let disc = (r0 * r0 + 2.0 * (r1 - r0) * u).max(0.0);
  let den = r0 + sqrtf(disc);
  if !(den > 0.0) {
    return u;
  }
  (2.0 * u / den).clamp(0.0, 1.0)
}

/// `(E[t], Var[t])` of the ramp density: `E[t] = ρ₀/2 + (ρ₁ − ρ₀)/3`,
/// `E[t²] = ρ₀/3 + (ρ₁ − ρ₀)/4`.
fn ramp_moments(rho: [f32; 2]) -> (f32, f32) {
  let (r0, d) = (rho[0], rho[1] - rho[0]);
  let et = 0.5 * r0 + d / 3.0;
  let et2 = r0 / 3.0 + 0.25 * d;
  (et, (et2 - et * et).max(0.0))
}

/// The moment-matched Gaussian of [`speck_leaf`] (the leaf's covariance plus the uniform axis'
/// `seg segᵀ Var[t]`, the mean moved by `seg E[t]`): the shell's number density at the eye
/// ([`SpeckCluster::density_count_rel`]) and the hand-over tests. Mirror of the density in
/// `dust_specks.comp`.
pub fn speck_gaussian(m: &DustMoments, b: usize, seg_t: [f32; 3]) -> Packet {
  let mut p = speck_leaf(m, b, seg_t);
  let (et, vt) = ramp_moments(p.rho);
  let s = p.seg;
  p.mean = add3(p.mean, scale3(s, et));
  p.cov[0] += vt * s[0] * s[0];
  p.cov[1] += vt * s[0] * s[1];
  p.cov[2] += vt * s[0] * s[2];
  p.cov[3] += vt * s[1] * s[1];
  p.cov[4] += vt * s[1] * s[2];
  p.cov[5] += vt * s[2] * s[2];
  p.seg = [0.0; 3];
  p.rho = [1.0, 1.0];
  p
}

/// The standard normal CDF and density (the mirrors' `erf`, `exp`).
fn norm_cdf(z: f32) -> f32 {
  0.5 * (1.0 + super::erf(z * core::f32::consts::FRAC_1_SQRT_2))
}
fn norm_pdf(z: f32) -> f32 {
  0.398_942_3 * expf(-0.5 * z * z)
}

/// The CDF and the density of `ζ = κ t + ξ`, `t` on `[0, 1]` with the linear density
/// `ρ₀ + (ρ₁ − ρ₀) t` (mean 1) and `ξ` standard normal. For `κ > ½` the closed form: with
/// `v = z − κ t`, `F(z) = (1/κ) ∫_{z−κ}^{z} (A + B v) Φ(v) dv`, `A = ρ₀ + (ρ₁ − ρ₀) z/κ`,
/// `B = −(ρ₁ − ρ₀)/κ`, through `∫Φ = vΦ + φ` and `∫vΦ = ((v² − 1)Φ + vφ)/2`, and
/// `f(z) = (1/κ)[A(Φ(z) − Φ(z − κ)) − B(φ(z) − φ(z − κ))]`. For `κ ≤ ½` that form cancels in f32
/// (its terms grow as `1/κ²`: 6·10⁻² error at κ = 0.002), and the Taylor series in `κ` is used:
/// `F = Φ − φ Σ_{n=1..5} c_n He_{n−1}(z)`, `f = φ (1 + Σ c_n He_n(z))`, `c_n = κⁿ E[tⁿ] / n!`,
/// `E[tⁿ] = ρ₀/(n+1) + (ρ₁ − ρ₀)/(n+2)` (the first omitted term is below 10⁻⁵ at κ = ½).
/// Emulated in f32 against a quadrature: `|F(Q(w)) − w| ≤ 8·10⁻⁶` for κ from 10⁻⁶ to 2000, ramps
/// up to 1.4/0.6, `w` from 10⁻⁶ to 1 − 10⁻⁵. Mirror of `dust_specks_zeta_cdf`.
fn zeta_cdf(z: f32, kappa: f32, rho: [f32; 2]) -> (f32, f32) {
  let (r0, d) = (rho[0], rho[1] - rho[0]);
  if kappa <= 0.5 {
    let z2 = z * z;
    let he = [
      1.0,
      z,
      z2 - 1.0,
      z * (z2 - 3.0),
      z2 * z2 - 6.0 * z2 + 3.0,
      z * (z2 * z2 - 10.0 * z2 + 15.0),
    ];
    let (mut s1, mut s2, mut kn, mut fact) = (0.0f32, 0.0f32, 1.0f32, 1.0f32);
    for n in 1..6usize {
      kn *= kappa;
      fact *= n as f32;
      let et = r0 / (n + 1) as f32 + d / (n + 2) as f32;
      let c = kn * et / fact;
      s1 += c * he[n - 1];
      s2 += c * he[n];
    }
    let p = norm_pdf(z);
    return (norm_cdf(z) - p * s1, p * (1.0 + s2));
  }
  let a = r0 + d * z / kappa;
  let bb = -d / kappa;
  let (z1, z0) = (z - kappa, z);
  let (c0, c1) = (norm_cdf(z0), norm_cdf(z1));
  let (p0, p1) = (norm_pdf(z0), norm_pdf(z1));
  let g1 = (z0 * c0 + p0) - (z1 * c1 + p1);
  let g2 = 0.5 * (((z0 * z0 - 1.0) * c0 + z0 * p0) - ((z1 * z1 - 1.0) * c1 + z1 * p1));
  let cdf = (a * g1 + bb * g2) / kappa;
  let pdf = (a * (c0 - c1) - bb * (p0 - p1)) / kappa;
  (cdf, pdf)
}

/// The quantile of `ζ = κ t + ξ` ([`zeta_cdf`]): `Φ⁻¹(w)` when `κ = 0`, else 16 safeguarded
/// Newton steps on the bracket `[−5.4, κ + 5.4]`, the bracket following the sign of `F − w` and a
/// step that leaves it (closed: a converged step equal to a bracket end stays) bisecting it,
/// started at `Φ⁻¹(w) + κ E[t]` for `κ ≤ ½` and at `κ·ramp_t(w)` above — the same fixed sequence
/// of operations on both mirrors (the former strict bracket rejected the converged step, which
/// rounds onto the bracket end, and bisected away from the root: 3·10⁻³ at κ = 0.5). `w` is
/// clamped to `[2⁻²⁴, 1 − 2⁻²⁴]`. Mirror of `dust_specks_zeta_quantile`.
pub fn zeta_quantile(w: f32, kappa: f32, rho: [f32; 2]) -> f32 {
  let w = w.clamp(1.0 / 16_777_216.0, 1.0 - 1.0 / 16_777_216.0);
  if !(kappa > 0.0) {
    return inv_norm_cdf(w);
  }
  let (mut lo, mut hi) = (-5.4f32, kappa + 5.4);
  let start = if kappa <= 0.5 {
    inv_norm_cdf(w) + kappa * (0.5 * rho[0] + (rho[1] - rho[0]) / 3.0)
  } else {
    kappa * ramp_t(w, rho)
  };
  let mut z = start.clamp(lo, hi);
  for _ in 0..16 {
    let (c, f) = zeta_cdf(z, kappa, rho);
    let e = c - w;
    if e > 0.0 {
      hi = z;
    } else {
      lo = z;
    }
    let n = if f > 0.0 { z - e / f } else { f32::NAN };
    z = if n >= lo && n <= hi {
      n
    } else {
      0.5 * (lo + hi)
    };
  }
  z
}

/// `M⁻¹` of a 3×3 map given as columns, row-major; `None` when singular.
fn inv3(m: &[[f32; 3]; 3]) -> Option<[f32; 9]> {
  // rows of M: r_i = (m[0][i], m[1][i], m[2][i])
  let a = |i: usize, j: usize| m[j][i];
  let c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
  let c01 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
  let c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
  let det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
  if !(fabs(det) > 1e-30) || !det.is_finite() {
    return None;
  }
  let id = 1.0 / det;
  Some([
    c00 * id,
    (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * id,
    (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * id,
    c01 * id,
    (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * id,
    (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * id,
    c02 * id,
    (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * id,
    (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * id,
  ])
}

fn inv3_apply(mi: &[f32; 9], d: [f32; 3]) -> [f32; 3] {
  [
    mi[0] * d[0] + mi[1] * d[1] + mi[2] * d[2],
    mi[3] * d[0] + mi[4] * d[1] + mi[5] * d[2],
    mi[6] * d[0] + mi[7] * d[1] + mi[8] * d[2],
  ]
}

/// One size bin of one cluster as the specks see it: the continuum's leaf ([`speck_leaf`]) as a
/// linear map of three coordinates. In the frame `(e₁, e₂, ŝ)` with `ŝ` along the leaf's uniform
/// axis (`dispersion_frame`), the leaf's Gaussian has the Cholesky factor `L_b`; the coordinate
/// along `ŝ` is `L_b₂₀ ξ₀ + L_b₂₁ ξ₁ + g ξ₂ + |seg| t` with `g = L_b₂₂`, so with
/// `ζ = κ t + ξ₂`, `κ = |seg|/g`, the position is `mean + M (ξ₀, ξ₁, ζ)`, `M = R L_b`, exactly
/// the leaf (uniform along the axis with its ramp, Gaussian across), and `ζ` has the closed-form
/// CDF of [`zeta_cdf`]. A 3-D tree over `(ξ₀, ξ₁, ζ)`: the map to the world is invertible, so a
/// ball's preimage is compact and the walk visits a few nodes per speck (a 4-D cube with the
/// uniform axis as a fourth coordinate maps degenerately — `t` and `ξ` along `ŝ` move a point
/// along the same line — and a ball's preimage is a tube whose cells grow with depth: 12 M nodes
/// per frame at any shell). Without a uniform axis `R = I`, `κ = 0` and `M = L`, the Gaussian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeckCluster {
  /// the start of the leaf's uniform axis (local m)
  pub mean: [f32; 3],
  /// columns of `M`: the position is `mean + M (ξ₀, ξ₁, ζ)` (`L`, a Cholesky factor, without a
  /// uniform axis)
  pub chol: [[f32; 3]; 3],
  /// `‖M_k‖`: the world extent of a unit step in each coordinate
  pub col_norm: [f32; 3],
  /// `M⁻¹`, row-major (the culling's coordinate-space bound), and `1 / ‖M⁻¹‖_F`
  pub minv: [f32; 9],
  pub sigma_lo: f32,
  /// `κ = |seg| / g` and the ramp of the uniform axis (`ζ`'s law, [`zeta_cdf`])
  pub kappa: f32,
  pub rho: [f32; 2],
  /// the moment-matched Gaussian ([`speck_gaussian`]) for the density at the eye: its mean
  /// relative to `mean` and its Cholesky factor
  pub mm_offset: [f32; 3],
  pub mm_chol: [[f32; 3]; 3],
  /// the unfaded grain count of the bin (before quantisation)
  pub count: f32,
  /// the tree's bound `mant · 2^exp` ([`count_bound`])
  pub mant: u32,
  pub exp: u32,
  /// the root hash: ring slot, child id, bin
  pub root_hash: u32,
  /// the probability a leaf of the `mant · 2^exp` tree is kept ([`count_bound`])
  pub keep: f32,
  /// the bin's edge speed factors (its sizes run from `s_ref / f1²` to `s_ref / f0²`), the
  /// reference radius (µm) and the size law exponent
  pub f0: f32,
  pub f1: f32,
  pub s_ref_um: f32,
  pub q: f32,
  pub bin: u32,
  /// the bin's faded cross-section (m²): the flux the specks draw
  pub flux: f32,
  /// the fade in force (the drawn cross-section per grain is `πs² · fade`)
  pub fade: f32,
}

/// `(M, κ)` of a leaf (see [`SpeckCluster`]): the frame with its third axis along `seg`, the
/// Cholesky factor of the leaf's covariance in that frame, mapped back. Mirror of
/// `dust_specks_map`.
pub fn leaf_map(cov: [f32; 6], seg: [f32; 3]) -> ([[f32; 3]; 3], f32) {
  let slen = norm3(seg);
  if !(slen > 0.0) {
    return (cholesky3(cov), 0.0);
  }
  let (e1, e2, e3) = super::dispersion_frame(seg);
  let sv = |v: [f32; 3]| -> [f32; 3] {
    [
      cov[0] * v[0] + cov[1] * v[1] + cov[2] * v[2],
      cov[1] * v[0] + cov[3] * v[1] + cov[4] * v[2],
      cov[2] * v[0] + cov[4] * v[1] + cov[5] * v[2],
    ]
  };
  let (s1, s2, s3) = (sv(e1), sv(e2), sv(e3));
  let lb = cholesky3([
    dot3(e1, s1),
    dot3(e1, s2),
    dot3(e1, s3),
    dot3(e2, s2),
    dot3(e2, s3),
    dot3(e3, s3),
  ]);
  let to_world = |c: [f32; 3]| add3(add3(scale3(e1, c[0]), scale3(e2, c[1])), scale3(e3, c[2]));
  let m = [to_world(lb[0]), to_world(lb[1]), to_world(lb[2])];
  let g = lb[2][2];
  let kappa = if g > 0.0 { slen / g } else { 0.0 };
  (m, kappa)
}

impl SpeckCluster {
  /// The speck view of bin `b` of a cluster: `slot` its ring slot (the identity; the compact
  /// index is **not** stable), `seg_t` the bin's time segment to its stream predecessor
  /// ([`bin_time_segment`]), the reference radius and `q` of the system, the fade (`fade_ratio`
  /// of the tier, `ttl_s` of the system); the size range `F` is the cluster's own
  /// ([`moments_size_range`]). Mirror of `dust_specks_cluster`.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    m: &DustMoments,
    b: usize,
    seg_t: [f32; 3],
    slot: u32,
    s_ref_um: f32,
    q: f32,
    ttl_s: f32,
    fade_ratio: f32,
  ) -> Self {
    let range = moments_size_range(m);
    let p = speck_leaf(m, b, seg_t);
    let fade = age_fade(m.age(), ttl_s, fade_ratio).max(1e-6);
    let xsec = bin_mean_cross_section_f32(s_ref_um, q, range, b as u32).max(1e-30);
    let count = p.flux / fade / xsec;
    let root_hash = dust_hash32(
      dust_hash32(slot.wrapping_mul(0x9E37_79B9) ^ m.id())
        .wrapping_add((b as u32).wrapping_mul(0x85EB_CA6B)),
    );
    let ((mant, exp), keep) = count_bound(count);
    let (map, kappa) = leaf_map(p.cov, p.seg);
    let mm = speck_gaussian(m, b, seg_t);
    let f = range.max(1.0);
    let (lo, hi) = (1.0 / sqrtf(f), sqrtf(f));
    Self {
      f0: lo + (hi - lo) * (b as f32 / SIZE_BINS as f32),
      f1: lo + (hi - lo) * ((b + 1) as f32 / SIZE_BINS as f32),
      ..Self::from_map(
        p.mean,
        map,
        kappa,
        p.rho,
        (sub3(mm.mean, p.mean), cholesky3(mm.cov)),
        count,
        ((mant, exp), keep),
        root_hash,
        (s_ref_um, q, b as u32, p.flux, fade),
      )
    }
  }

  /// A speck cluster from its map (`mean + M (ξ₀, ξ₁, ζ)`, `ζ`'s law `κ`, `ρ`), the
  /// moment-matched Gaussian `(offset, Cholesky)`, the count and its bound, the root hash and
  /// `(s_ref, q, bin, flux, fade)` (the size factors 1: one size; [`Self::new`] sets the bin's).
  #[allow(clippy::too_many_arguments)]
  pub fn from_map(
    mean: [f32; 3],
    map: [[f32; 3]; 3],
    kappa: f32,
    rho: [f32; 2],
    mm: ([f32; 3], [[f32; 3]; 3]),
    count: f32,
    bound: ((u32, u32), f32),
    root_hash: u32,
    sizes: (f32, f32, u32, f32, f32),
  ) -> Self {
    let (minv, sigma_lo) = match inv3(&map) {
      Some(mi) => (mi, 1.0 / sqrtf(mi.iter().map(|v| v * v).sum::<f32>())),
      None => ([0.0; 9], 0.0),
    };
    let ((mant, exp), keep) = bound;
    let (s_ref_um, q, bin, flux, fade) = sizes;
    Self {
      mean,
      chol: map,
      col_norm: [norm3(map[0]), norm3(map[1]), norm3(map[2])],
      minv,
      sigma_lo,
      kappa,
      rho,
      mm_offset: mm.0,
      mm_chol: mm.1,
      count,
      mant,
      exp,
      root_hash,
      keep,
      f0: 1.0,
      f1: 1.0,
      s_ref_um,
      q,
      bin,
      flux,
      fade,
    }
  }

  /// the tree's bound `N_max = mant · 2^exp` as a float (the expected count is `N_max · keep`)
  pub fn quantised_count(&self) -> f32 {
    self.mant as f32 * f32::from_bits(((127 + self.exp as i32) as u32) << 23)
  }

  /// `N_j · G(x)` (m⁻³), `G` the moment-matched Gaussian of the leaf: the bin's contribution to
  /// the number density at `x`.
  pub fn density_count_at(&self, x: [f32; 3]) -> f32 {
    self.density_count_rel(sub3(self.mean, x))
  }

  /// [`Self::density_count_at`] from `me = mean − x` (the eye-relative leaf start the pass forms
  /// once). Mirror of the density in `dust_specks.comp`.
  pub fn density_count_rel(&self, me: [f32; 3]) -> f32 {
    self.count * gaussian_density(&self.mm_chol, add3(me, self.mm_offset), [0.0; 3])
  }

  /// `mean − eye` in df64 per axis (an exact two-sum of the f32 inputs): the eye-relative leaf
  /// start every speck offset is added to. Mirror of `dust_specks_mean_rel`.
  pub fn mean_rel(&self, eye: [f32; 3]) -> [Df; 3] {
    [
      two_sum(self.mean[0], -eye[0]),
      two_sum(self.mean[1], -eye[1]),
      two_sum(self.mean[2], -eye[2]),
    ]
  }

  /// The speck offset from the eye `dp = (mean − eye) + M c` in df64 (exact products and
  /// two-sums of the f32 inputs, rounded once to f32 at the end): the f32 sum rounded at the
  /// scale of the cluster's extent (8 ulps of 350 km = 17 cm for a 1-day-old cluster: jitter of a
  /// near speck), this rounds at the scale of `|dp|` (≤ the shell). Mirror of
  /// `dust_specks_offset`.
  pub fn offset_df(&self, me: &[Df; 3], c: [f32; 3]) -> [f32; 3] {
    let l = &self.chol;
    let mut out = [0.0f32; 3];
    for (k, o) in out.iter_mut().enumerate() {
      let mut acc = me[k];
      for (j, x) in c.iter().enumerate() {
        acc = acc.add(two_prod(l[j][k], *x));
      }
      *o = acc.hi + acc.lo;
    }
    out
  }

  /// The grain radius (µm) of a speck from its hash: the inverse CDF of the bin's number
  /// distribution `n(s) ∝ s^-q` over `[s_ref / f1², s_ref / f0²]`, so the mean drawn
  /// cross-section is the bin's `⟨πs²⟩_j` exactly in expectation and the specks deliver the flux
  /// the count was made from. Within a bin the size does not follow the position. Mirror of
  /// `dust_specks_size_um`.
  pub fn speck_size_um(&self, hash: u32) -> f32 {
    let (a, b) = (self.f0 * self.f0, self.f1 * self.f1);
    let s_lo = self.s_ref_um / b.max(1e-12);
    let s_hi = self.s_ref_um / a.max(1e-12);
    if !(s_hi > s_lo * (1.0 + 1e-6)) {
      return s_lo;
    }
    let u = u01(hash_word(hash, 4));
    let e = 1.0 - self.q;
    if fabs(e) < 1e-6 {
      return s_lo * expf(u * lnf(s_hi / s_lo));
    }
    let (p0, p1) = (powf(s_lo, e), powf(s_hi, e));
    powf(p0 + u * (p1 - p0), 1.0 / e).clamp(s_lo, s_hi)
  }

  /// The coordinates of probability `w` (`(Φ⁻¹(w₀), Φ⁻¹(w₁), ζ(w₂))`).
  pub fn coords(&self, w: [f32; 3]) -> [f32; 3] {
    [
      inv_norm_cdf(w[0]),
      inv_norm_cdf(w[1]),
      zeta_quantile(w[2], self.kappa, self.rho),
    ]
  }

  /// The node's coordinate box: `(mid, half)` per axis.
  fn node_box(&self, lo: [f32; 3], width: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let a = self.coords(lo);
    let b = self.coords(add3(lo, width));
    let mut mid = [0.0f32; 3];
    let mut half = [0.0f32; 3];
    for k in 0..3 {
      mid[k] = 0.5 * (a[k] + b[k]);
      half[k] = 0.5 * (b[k] - a[k]);
    }
    (mid, half)
  }

  /// The circumradius of `M·[−h, h]`.
  fn box_radius(&self, h: [f32; 3]) -> f32 {
    let mut r2 = 0.0f32;
    for (s1, s2) in [(1.0f32, 1.0f32), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
      let v = chol_apply(&self.chol, [h[0], s1 * h[1], s2 * h[2]]);
      r2 = r2.max(dot3(v, v));
    }
    sqrtf(r2)
  }

  /// `M⁻¹ (eye − mean)` from `me = mean − eye`: the eye in coordinate space (the culling's
  /// second bound; [`Self::sigma_lo`] bounds `M`'s smallest singular value from below).
  pub fn eye_in_xi(&self, me: [f32; 3]) -> ([f32; 3], f32) {
    if self.sigma_lo > 0.0 {
      (inv3_apply(&self.minv, scale3(me, -1.0)), self.sigma_lo)
    } else {
      ([0.0; 3], 0.0)
    }
  }

  /// Whether a node may hold specks within `d_max` of the eye in front of the eye plane: the
  /// larger of two lower bounds of the eye–node distance (the circumsphere of the node's
  /// parallelepiped `mean + M·box`, the coordinate-space box distance × `σ_lo`) against
  /// `d_max`, and the circumsphere against the eye plane. Eye-relative (`me = mean − eye`), in
  /// f32 with a rounding slack; the box is widened by [`DUST_SPECKS_XI_TOL`] (the quantiles'
  /// non-monotonicity). Conservative: never culls a node that holds a speck the leaf test would
  /// accept. Mirror of `dust_specks_node_meets`.
  #[allow(clippy::too_many_arguments)]
  pub fn node_meets(
    &self,
    lo: [f32; 3],
    width: [f32; 3],
    me: [f32; 3],
    y_eye: [f32; 3],
    sigma_lo: f32,
    d_max: f32,
    fwd: Option<[f32; 3]>,
  ) -> bool {
    let (mid, half) = self.node_box(lo, width);
    let tol = |c: f32| DUST_SPECKS_XI_TOL * (1.0 + fabs(c));
    let half = [
      half[0] + tol(mid[0]),
      half[1] + tol(mid[1]),
      half[2] + tol(mid[2]),
    ];
    let lm = chol_apply(&self.chol, mid);
    let d = add3(me, lm);
    // the f32 rounding of `d` against the df64 leaves: every term of the sum (a sum that
    // cancels rounds at the scale of its terms, not of its result)
    let terms = self.col_norm[0] * fabs(mid[0])
      + self.col_norm[1] * fabs(mid[1])
      + self.col_norm[2] * fabs(mid[2]);
    let r = self.box_radius(half) + 8.0 * f32::EPSILON * (norm3(me) + terms + norm3(d));
    if norm3(d) - r > d_max {
      return false;
    }
    if sigma_lo > 0.0 {
      let mut q = 0.0f32;
      for k in 0..3 {
        // `y_eye` is f32 from `me`: its rounding widens the box too
        let o = (fabs(y_eye[k] - mid[k])
          - half[k]
          - 8.0 * f32::EPSILON * (fabs(y_eye[k]) + fabs(mid[k])))
        .max(0.0);
        q += o * o;
      }
      if sigma_lo * sqrtf(q) > d_max {
        return false;
      }
    }
    if let Some(f) = fwd
      && dot3(d, f) + r < 0.0
    {
      return false;
    }
    true
  }

  /// The walk's start nodes (the jump start): the ball of radius `d_max` about the eye maps to
  /// the ellipsoid `y_eye + M⁻¹·ball` in coordinate space, whose box is `y_eye ± d_max·‖row_k(M⁻¹)‖`
  /// (widened by [`DUST_SPECKS_XI_TOL`] and a relative 10⁻⁴ for the f32 inverse), and through
  /// the coordinates' CDFs (`Φ`, [`zeta_cdf`]; widened by [`DUST_SPECKS_JUMP_W_TOL`], above their
  /// quantiles' round-trip error) to a box `B` of the probability cube. At the deepest level
  /// whose cells are at least `B`'s width on every axis, at most 2 cells per axis meet `B`;
  /// each is reached by its path from the root (`split_node` along its bits: the same counts and
  /// hashes the full walk computes, no bound test, no sibling), stopping early at an empty node
  /// or a leaf (duplicates of such an ancestor merged). Every grain within the ball lies in a
  /// cell meeting `B`, so the walk from these nodes finds exactly the specks of the walk from the
  /// root — at a cost independent of the tree's depth above them (from the root, every bin paid
  /// ~2·log₂N ≈ 80–300 visits even holding ≪ 1 grain near the eye: 10⁸ visits a frame inside
  /// the coma). Without a usable `M⁻¹` (`sigma_lo = 0`), the root. Mirror of
  /// `dust_specks_start_nodes`.
  fn start_nodes(&self, y_eye: [f32; 3], sigma_lo: f32, d_max: f32) -> ([Node; 8], usize) {
    let mut out = [Node::default(); 8];
    let root = Node {
      level: 0,
      lo: [0; 3],
      mant: self.mant,
      exp: self.exp,
      hash: self.root_hash,
    };
    if !(sigma_lo > 0.0) || !(d_max > 0.0) {
      out[0] = root;
      return (out, 1);
    }
    // the box in probability coordinates
    let mut b_lo = [0.0f32; 3];
    let mut b_hi = [0.0f32; 3];
    for k in 0..3 {
      let r = [self.minv[3 * k], self.minv[3 * k + 1], self.minv[3 * k + 2]];
      let h0 = d_max * norm3(r);
      let y = y_eye[k];
      let h = h0 + 1.0e-4 * (fabs(y) + h0) + DUST_SPECKS_XI_TOL * (1.0 + fabs(y));
      let cdf = |z: f32| -> f32 {
        if k == 2 && self.kappa > 0.0 {
          zeta_cdf(z, self.kappa, self.rho).0
        } else {
          norm_cdf(z)
        }
      };
      b_lo[k] = (cdf(y - h) - DUST_SPECKS_JUMP_W_TOL).clamp(0.0, 1.0);
      b_hi[k] = (cdf(y + h) + DUST_SPECKS_JUMP_W_TOL).clamp(0.0, 1.0);
    }
    // the deepest level whose cells are as wide as the box on every axis: dim k has m_k(L) =
    // (L + 2 − k) / 3 splits at level L, so m_k ≤ M_k (cells of 2^−M_k ≥ the box) ⟺ L ≤ 3 M_k + k
    let mut level = DUST_SPECKS_MAX_LEVEL;
    for k in 0..3 {
      let w = b_hi[k] - b_lo[k];
      let mut m = 0u32;
      while m < DUST_SPECKS_JUMP_MAX_SPLITS && f32::from_bits((126 - m) << 23) >= w {
        m += 1;
      }
      level = level.min(3 * m + k as u32);
    }
    // the cells of that level meeting the box: ≤ 2 per axis
    let mut lo_i = [0u32; 3];
    let mut n_i = [1u32; 3];
    for k in 0..3 {
      let m = (level + 2 - k as u32) / 3;
      let cells = 1u32 << m;
      let scale = cells as f32;
      let a = ((b_lo[k] * scale) as u32).min(cells - 1);
      let b = ((b_hi[k] * scale) as u32).min(cells - 1);
      lo_i[k] = a;
      n_i[k] = (b - a + 1).min(2);
    }
    let mut n = 0usize;
    for c in 0..(n_i[0] * n_i[1] * n_i[2]) {
      let idx = [lo_i[0] + c % n_i[0], lo_i[1] + (c / n_i[0]) % n_i[1], lo_i[2] + c / (n_i[0] * n_i[1])];
      let mut target = [0u32; 3];
      for k in 0..3 {
        let m = (level + 2 - k as u32) / 3;
        target[k] = if m == 0 { 0 } else { idx[k] << (32 - m) };
      }
      // the path from the root
      let mut node = root;
      while node.level < level
        && node.mant != 0
        && !(node.mant == 1 && node.exp == 0)
        && node.level < DUST_SPECKS_MAX_LEVEL
      {
        let dim = (node.level % 3) as usize;
        let m = (node.level + 2 - dim as u32) / 3;
        let half = if m + 1 >= 32 { 0 } else { 1u32 << (31 - m) };
        let (left, right) = split_node(&node);
        node = if target[dim] & half != 0 { right } else { left };
      }
      if node.mant == 0 {
        continue;
      }
      if out[..n].iter().any(|o| o.level == node.level && o.lo == node.lo) {
        continue;
      }
      out[n] = node;
      n += 1;
    }
    (out, n)
  }

  /// Enumerates the specks of this bin whose tree nodes meet the ball of radius `d_max` about
  /// `eye` and are not wholly behind the plane through `eye` normal to `fwd` (`fwd` zero: no
  /// plane). Only specks inside the ball are reported (`f(speck) → false` stops). The traversal
  /// is a depth-first walk of the dyadic tree with an explicit stack (the GPU does the same,
  /// `dust_specks.comp`); at most `max_nodes` nodes are visited. Returns the statistics.
  pub fn enumerate(
    &self,
    eye: [f32; 3],
    d_max: f32,
    fwd: [f32; 3],
    max_nodes: u32,
    f: &mut dyn FnMut(&Speck) -> bool,
  ) -> EnumStats {
    self.walk(eye, d_max, fwd, max_nodes, None, f)
  }

  /// [`Self::enumerate`] with the frame's node account (`frame_nodes`, the [`PYR_SPECKS_NODES`]
  /// word): every [`DUST_SPECKS_NODE_CHUNK`] visits are added at once and the walk stops
  /// (truncated) when the account held [`DUST_SPECKS_FRAME_NODES`] or more before the add; the
  /// remainder is added at the end. Mirror of the walk in `dust_specks.comp`.
  pub fn walk(
    &self,
    eye: [f32; 3],
    d_max: f32,
    fwd: [f32; 3],
    max_nodes: u32,
    frame_nodes: Option<&mut u32>,
    f: &mut dyn FnMut(&Speck) -> bool,
  ) -> EnumStats {
    self.walk_with(eye, d_max, fwd, max_nodes, frame_nodes, true, f)
  }

  /// [`Self::enumerate`] from the root (no jump start): the reference the jump start must equal
  /// (`jump_start_finds_the_root_walks_specks`).
  pub fn enumerate_from_root(
    &self,
    eye: [f32; 3],
    d_max: f32,
    fwd: [f32; 3],
    max_nodes: u32,
    f: &mut dyn FnMut(&Speck) -> bool,
  ) -> EnumStats {
    self.walk_with(eye, d_max, fwd, max_nodes, None, false, f)
  }

  #[allow(clippy::too_many_arguments)]
  fn walk_with(
    &self,
    eye: [f32; 3],
    d_max: f32,
    fwd: [f32; 3],
    max_nodes: u32,
    mut frame_nodes: Option<&mut u32>,
    jump: bool,
    f: &mut dyn FnMut(&Speck) -> bool,
  ) -> EnumStats {
    let mut st = EnumStats::default();
    if self.mant == 0 || !(d_max > 0.0) {
      return st;
    }
    let me_df = self.mean_rel(eye);
    let me = [
      me_df[0].hi + me_df[0].lo,
      me_df[1].hi + me_df[1].lo,
      me_df[2].hi + me_df[2].lo,
    ];
    let mut stack: [Node; DUST_SPECKS_STACK] = [Node::default(); DUST_SPECKS_STACK];
    let has_plane = dot3(fwd, fwd) > 0.0;
    let plane = has_plane.then_some(fwd);
    let (y_eye, sigma_lo) = self.eye_in_xi(me);
    // the jump start: the ≤ 8 nodes whose cells meet the ball's box, reached by their path
    // (`start_nodes`), pushed in reverse so the first is visited first
    let (starts, n_starts) = self.start_nodes(y_eye, if jump { sigma_lo } else { 0.0 }, d_max);
    let mut sp = 0usize;
    for k in (0..n_starts).rev() {
      stack[sp] = starts[k];
      sp += 1;
    }
    'walk: while sp > 0 {
      sp -= 1;
      let node = stack[sp];
      st.nodes += 1;
      if st.nodes > max_nodes {
        st.truncated = true;
        break;
      }
      if st.nodes % DUST_SPECKS_NODE_CHUNK == 0
        && let Some(acc) = frame_nodes.as_deref_mut()
      {
        let old = *acc;
        *acc = old.saturating_add(DUST_SPECKS_NODE_CHUNK);
        if old >= DUST_SPECKS_FRAME_NODES {
          st.truncated = true;
          break;
        }
      }
      if node.mant == 0 {
        continue;
      }
      let (lo, width, half_bits) = node_cell(node.level, node.lo);
      if !self.node_meets(lo, width, me, y_eye, sigma_lo, d_max, plane) {
        continue;
      }
      let leaf = (node.mant == 1 && node.exp == 0) || node.level >= DUST_SPECKS_MAX_LEVEL;
      if leaf {
        let n = if node.exp == 0 { node.mant.min(64) } else { 64 };
        for k in 0..n {
          let hs = if k == 0 {
            node.hash
          } else {
            dust_hash32(node.hash.wrapping_add(k.wrapping_mul(0x9E37_79B9)))
          };
          // the thinning of the bound (`count_bound`)
          if self.keep < 1.0 && !(u01(hash_word(hs, 5)) < self.keep) {
            continue;
          }
          let mut w = [0.0f32; 3];
          for (i, wi) in w.iter_mut().enumerate() {
            *wi = lo[i] + u01(hash_word(hs, i)) * width[i];
          }
          let xi = self.coords(w);
          let dp = self.offset_df(&me_df, xi);
          if dot3(dp, dp) > d_max * d_max {
            continue;
          }
          if has_plane && dot3(dp, fwd) <= 0.0 {
            continue;
          }
          st.specks += 1;
          let size_um = self.speck_size_um(hs);
          let s = size_um * 1e-6;
          let speck = Speck {
            pos: add3(eye, dp),
            dp,
            w,
            xi,
            size_um,
            xsec_m2: PI32 * s * s * self.fade,
            hash: hs,
            bin: self.bin,
          };
          if !f(&speck) {
            st.stopped = true;
            break 'walk;
          }
        }
        continue;
      }
      // split dim level mod 3 at its probability midpoint; the right child is pushed first so
      // the left is visited first
      let _ = half_bits;
      let (left, right) = split_node(&node);
      stack[sp] = right;
      sp += 1;
      stack[sp] = left;
      sp += 1;
    }
    if let Some(acc) = frame_nodes {
      *acc = acc.saturating_add(st.nodes % DUST_SPECKS_NODE_CHUNK);
    }
    st
  }
}

/// A tree node of [`SpeckCluster::enumerate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Node {
  level: u32,
  /// the cell's lower corner, fixed point (2³² = 1)
  lo: [u32; 3],
  mant: u32,
  exp: u32,
  hash: u32,
}

/// The two children of a node split at its level's dim (`level mod 3`) at the probability
/// midpoint: above 2²⁴ exact halves (normalised back to a plain count when they drop below),
/// else `Bin(n, ½)` from the node's hash ([`child_count`]); the children's hashes from the
/// parent's. The walk and the jump start's path descent share it. Mirror of the split in
/// `dust_specks.comp`.
fn split_node(node: &Node) -> (Node, Node) {
  let dim = (node.level % 3) as usize;
  let m = (node.level + 2 - dim as u32) / 3;
  let half = if m + 1 >= 32 { 0 } else { 1u32 << (31 - m) };
  let (ml, mr, exp) = if node.exp > 0 {
    let (m, e) = normalise_count((node.mant, node.exp - 1));
    (m, m, e)
  } else {
    let l = child_count(node.mant, node.hash);
    (l, node.mant - l, 0)
  };
  let mut lo_r = node.lo;
  lo_r[dim] = lo_r[dim].wrapping_add(half);
  (
    Node {
      level: node.level + 1,
      lo: node.lo,
      mant: ml,
      exp,
      hash: dust_hash32(node.hash ^ 0x2545_F491),
    },
    Node {
      level: node.level + 1,
      lo: lo_r,
      mant: mr,
      exp,
      hash: dust_hash32(node.hash ^ 0x6C8E_9CF5),
    },
  )
}

/// The cell of a node: `(lo, width)` in probability coordinates and the half-width of its next
/// split per dim in fixed point. Dim `k` has been split `(level + 2 − k) / 3` times.
fn node_cell(level: u32, lo_bits: [u32; 3]) -> ([f32; 3], [f32; 3], [u32; 3]) {
  let mut lo = [0.0f32; 3];
  let mut width = [0.0f32; 3];
  let mut half = [0u32; 3];
  for k in 0..3 {
    let m = (level + 2 - k as u32) / 3;
    lo[k] = lo_bits[k] as f32 * (1.0 / 4_294_967_296.0);
    width[k] = f32::from_bits(((127 - m as i32) as u32) << 23); // 2^-m
    half[k] = if m + 1 >= 32 { 0 } else { 1u32 << (31 - m) };
  }
  (lo, width, half)
}

/// One realised grain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speck {
  /// particle-system local position (m): `eye + dp` in f32 (for diagnostics; the draw uses `dp`)
  pub pos: [f32; 3],
  /// the offset from the eye (m), `(μ − eye) + L ξ` summed in df64 ([`SpeckCluster::offset_df`])
  pub dp: [f32; 3],
  /// its probability coordinates (the identity in the bin's cube)
  pub w: [f32; 3],
  /// its coordinates `(ξ₀, ξ₁, ζ)` ([`SpeckCluster::coords`]): the offset from the leaf start
  /// is `M` times them
  pub xi: [f32; 3],
  pub size_um: f32,
  /// the cross-section it is drawn with (m², `πs² · fade`)
  pub xsec_m2: f32,
  /// the leaf hash (the τ rounding threshold, diagnostics)
  pub hash: u32,
  pub bin: u32,
}

/// Statistics of one [`SpeckCluster::enumerate`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnumStats {
  pub nodes: u32,
  pub specks: u32,
  /// the node cap was hit
  pub truncated: bool,
  /// the callback stopped the walk
  pub stopped: bool,
}

// ─── Draw ───────────────────────────────────────────────────────────────────

/// The photometry of a speck's cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeckPhotometry {
  pub albedo: f32,
  pub sun: [f32; 3],
  pub r_h_au: f32,
}

/// The view a speck pass draws with (from the push constants and the layout).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeckView {
  pub mvp: [f32; 16],
  pub width: u32,
  pub height: u32,
  pub eye: [f32; 3],
  /// unit view direction (the depth gradient of the projection, towards the scene)
  pub fwd: [f32; 3],
  pub perspective: bool,
  pub exposure: f32,
  pub inv_unit: f32,
}

impl SpeckView {
  /// From the push constants and the pyramid layout: the view direction as
  /// `project_packet_lit` derives it (the w row when perspective, the clip-z row when
  /// orthographic, oriented so that the local origin lies in front of the eye).
  pub fn new(pc: &DustSpecksPushConstants, layout: &PyramidLayout) -> Self {
    let m = |r: usize, c: usize| pc.mvp[c * 4 + r];
    let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
    let (r2, r3) = (row(2), row(3));
    let eye = [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]];
    let perspective = dot3(r3, r3) > 0.0;
    let raw = if perspective { r3 } else { r2 };
    let n2 = dot3(raw, raw);
    // perspective: the clip `w` grows into the view (`w > 0` is in front by the definition of
    // clip space), so the w row is the forward direction as it is; orthographic: the clip-z
    // row's sign depends on the depth convention, oriented so that the local origin lies in
    // front (the continuum's rule). The origin rule on a perspective camera flipped the plane
    // for a camera inside the cloud looking away from the anchor: every speck in front was
    // dropped (`gpu_dust_specks_match_reference`, 0 drawn)
    let fwd = if n2 > 0.0 {
      let f = scale3(raw, 1.0 / sqrtf(n2));
      if !perspective && dot3(scale3(eye, -1.0), f) < 0.0 {
        scale3(f, -1.0)
      } else {
        f
      }
    } else {
      [0.0; 3]
    };
    Self {
      mvp: pc.mvp,
      width: layout.width,
      height: layout.height,
      eye,
      fwd,
      perspective,
      exposure: pc.exposure,
      inv_unit: pc.inv_unit,
    }
  }
}

/// Scatters one speck into level 0 of the pyramid: projects its eye-relative offset `dp` through
/// the eye-relative mvp (dropped when behind the eye or more than the footprint off screen), `τ = exposure · xsec · w / A_px` with
/// the pixel area at the speck's depth from the Jacobian and the radiance weight at the speck,
/// counts `⌊τ · inv_unit + u⌉` (`u` from `hash`), spread over the 5×5 footprint by the Gaussian
/// of [`DUST_SPECKS_SIGMA_PX`] (normalised over the footprint) with error diffusion; the colour
/// words and the nearest depth as the continuum's. Updates `PYR_TAU_MAX` with the speck's peak.
/// Returns whether the speck was on screen (counts toward the budget) and the counts deposited.
/// Mirror of `scatter_speck` in `dust_specks.comp`.
#[allow(clippy::too_many_arguments)]
pub fn scatter_speck(
  dp: [f32; 3],
  xsec_m2: f32,
  hash: u32,
  color: [f32; 3],
  phot: &SpeckPhotometry,
  view: &SpeckView,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
) -> (bool, u32) {
  let m = |r: usize, c: usize| view.mvp[c * 4 + r];
  let row = |r: usize| [m(r, 0), m(r, 1), m(r, 2)];
  let clip = |r: usize| dot3(row(r), dp) + m(r, 3);
  let w = clip(3);
  if !(w > 0.0) {
    return (false, 0);
  }
  let d = dp;
  if !view.perspective && dot3(view.fwd, view.fwd) > 0.0 && dot3(d, view.fwd) <= 0.0 {
    return (false, 0);
  }
  let ndc = [clip(0) / w, clip(1) / w];
  let (wf, hf) = (view.width as f32, view.height as f32);
  let mu = [(ndc[0] * 0.5 + 0.5) * wf, (ndc[1] * 0.5 + 0.5) * hf];
  let fp = DUST_SPECKS_FOOTPRINT as f32 + 0.5;
  if !(mu[0] > -fp && mu[1] > -fp && mu[0] < wf + fp && mu[1] < hf + fp) {
    return (false, 0);
  }
  let (r0, r1, r3) = (row(0), row(1), row(3));
  let jx = scale3(sub3(r0, scale3(r3, ndc[0])), 0.5 * wf / w);
  let jy = scale3(sub3(r1, scale3(r3, ndc[1])), 0.5 * hf / w);
  let g = [dot3(jx, jx), dot3(jx, jy), dot3(jy, jy)];
  let det_g = (g[0] * g[2] - g[1] * g[1]).max(0.0);
  let weight = if phot.albedo > 0.0 {
    let to_eye = if view.perspective {
      scale3(d, -1.0)
    } else {
      scale3(view.fwd, -1.0)
    };
    radiance_weight(phot.albedo, phot.sun, to_eye, phot.r_h_au)
  } else {
    1.0
  };
  // the integrated optical depth over the PSF, exposure-scaled, per px² summed: xsec / A_px
  let tau = view.exposure * xsec_m2 * weight * sqrtf(det_g);
  if !(tau > 0.0) || !tau.is_finite() {
    return (true, 0);
  }
  // the peak per px² for the unit's adaptation (a Gaussian of σ² = 0.25 px²)
  let peak = tau / (2.0 * PI32 * DUST_SPECKS_SIGMA_PX * DUST_SPECKS_SIGMA_PX);
  let cur = f32::from_bits(pyramid[PYR_TAU_MAX as usize]);
  if !(cur >= peak) {
    pyramid[PYR_TAU_MAX as usize] = peak.to_bits();
  }
  let counts_f = (tau * view.inv_unit).min(DUST_COUNT_MAX_PER_ADD as f32);
  // word 6: words 0–3 place the speck, 4 its size, 5 its thinning
  let total = floorf(counts_f + u01(hash_word(hash, 6))) as u32;
  if total == 0 {
    return (true, 0);
  }
  let depth_bits = (norm3(d) / AU_M_F32).max(0.0).to_bits();
  let (off, lw, lh) = layout.levels[0];
  let cx = floorf(mu[0]) as i64;
  let cy = floorf(mu[1]) as i64;
  let inv2s2 = 1.0 / (2.0 * DUST_SPECKS_SIGMA_PX * DUST_SPECKS_SIGMA_PX);
  // the footprint weights, normalised over the window
  let mut wx = [0.0f32; 5];
  let mut wy = [0.0f32; 5];
  let (mut sx, mut sy) = (0.0f32, 0.0f32);
  for i in 0..5 {
    let dx = (cx + i as i64 - DUST_SPECKS_FOOTPRINT) as f32 + 0.5 - mu[0];
    let dy = (cy + i as i64 - DUST_SPECKS_FOOTPRINT) as f32 + 0.5 - mu[1];
    wx[i] = expf(-dx * dx * inv2s2);
    wy[i] = expf(-dy * dy * inv2s2);
    sx += wx[i];
    sy += wy[i];
  }
  let norm = 1.0 / (sx * sy);
  let mut carry = 0.0f32;
  let mut deposited = 0u32;
  for (iy, wyv) in wy.iter().enumerate() {
    let ty = cy + iy as i64 - DUST_SPECKS_FOOTPRINT;
    for (ix, wxv) in wx.iter().enumerate() {
      let tx = cx + ix as i64 - DUST_SPECKS_FOOTPRINT;
      let acc = total as f32 * wxv * wyv * norm + carry;
      let c = floorf(acc + 0.5) as u32;
      carry = acc - c as f32;
      if c == 0 || tx < 0 || ty < 0 || tx >= lw as i64 || ty >= lh as i64 {
        continue;
      }
      let i = (off + (ty as u32 * lw + tx as u32) * PYRAMID_TEXEL_WORDS) as usize;
      sat_add(&mut pyramid[i], c);
      let hf = (dust_hash32(i as u32) >> 8) as f32 * (1.0 / 16_777_216.0);
      for k in 0..3 {
        sat_add(
          &mut pyramid[i + 1 + k],
          (c as f32 * color[k] * DUST_COLOR_SCALE + hf) as u32,
        );
      }
      if pyramid[i + 4] == 0 || depth_bits < pyramid[i + 4] {
        pyramid[i + 4] = depth_bits;
      }
      deposited += c;
    }
  }
  if deposited > 0 {
    pyramid[PYR_LEVEL_MASK as usize] |= 1;
  }
  (true, deposited)
}

/// The per-view scalars a speck pass reads from the pyramid header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeckHeader {
  pub d_max: f32,
  pub flags: u32,
  pub budget: u32,
}

impl SpeckHeader {
  pub fn read(pyramid: &[u32]) -> Self {
    Self {
      d_max: f32::from_bits(pyramid[PYR_SPECKS_DMAX as usize]),
      flags: pyramid[PYR_FLAGS as usize],
      budget: pyramid[PYR_SPECKS_BUDGET as usize],
    }
  }
  /// Writes the host words of a frame (the header the host rebuilds every frame): the shell
  /// radius and the budget, the counters zeroed.
  pub fn write(header: &mut [u32], d_max: f32, budget: u32) {
    header[PYR_SPECKS_DMAX as usize] = d_max.to_bits();
    header[PYR_SPECKS_BUDGET as usize] = budget;
    header[PYR_SPECKS_NODES as usize] = 0;
    header[PYR_SPECKS as usize] = 0;
    header[PYR_SPECKS_OVERFLOW as usize] = 0;
    header[PYR_SPECKS_TRUNCATED as usize] = 0;
    header[PYR_N_EYE as usize] = 0;
    header[PYR_N_EYE_HI as usize] = 0;
    header[PYR_SPECKS_COUNTS as usize] = 0;
  }
}

/// Adds the 64-bit count `v` to the fixed-point `n_eye` accumulator of the header (mirror of the
/// GPU's two `atomicAdd`s: the low word with a carry, the high word).
pub fn n_eye_add(pyramid: &mut [u32], v: u64) {
  let (hi, lo) = ((v >> 32) as u32, v as u32);
  let old = pyramid[PYR_N_EYE as usize];
  let new = old.wrapping_add(lo);
  pyramid[PYR_N_EYE as usize] = new;
  let carry = (new < old) as u32;
  pyramid[PYR_N_EYE_HI as usize] =
    pyramid[PYR_N_EYE_HI as usize].wrapping_add(hi).wrapping_add(carry);
}

/// `N_j · G_j(eye)` as a 64-bit count of [`N_EYE_UNIT`] (a dense coma near the nucleus is
/// 10³–10⁶ m⁻³: far more than 2³² counts; the accumulator is 64-bit). Mirror of the split in
/// `dust_specks.comp`'s `main`.
pub fn n_eye_counts(density_m3: f32) -> u64 {
  let c = density_m3 as f64 * (1.0 / N_EYE_UNIT);
  if !(c > 0.0) {
    0
  } else if c >= 1.8e19 {
    u64::MAX / 2
  } else {
    (c + 0.5) as u64
  }
}

/// One tier's speck pass on the host (CPU particle mode and tests; mirror of
/// `dust_specks.comp`): for every live cluster and size bin, the bin's number density at the eye
/// into `PYR_N_EYE`, then — when the bin's cloud meets the shell of radius `PYR_SPECKS_DMAX`
/// about the eye — its specks inside the shell ([`SpeckCluster::enumerate`]) drawn
/// ([`scatter_speck`]) within the frame budget (`PYR_SPECKS_BUDGET`; the excess is counted in
/// `PYR_SPECKS_OVERFLOW`). `flow` is the flow clock (the flow marks modulate the exposure by the
/// cluster's age when `DUST_VIEW_FLOW` is set, as the splat's).
pub fn specks_tier(
  moments: &[DustMoments],
  render: &[DustRenderCluster],
  pc: &DustSpecksPushConstants,
  flow: &DustFlowUniform,
  layout: &PyramidLayout,
  pyramid: &mut [u32],
) -> SpeckPassStats {
  let n = (pc.live_count as usize).min(moments.len()).min(render.len());
  let hdr = SpeckHeader::read(pyramid);
  let view = SpeckView::new(pc, layout);
  let stream = unpack_color(pc.color);
  let (s_ref, fade_ratio, ttl) = (pc.s_ref_um(), pc.fade_ratio(), pc.eye_local[3]);
  let q = pc.size_q();
  let mut stats = SpeckPassStats::default();
  for i in 0..n {
    let m = &moments[i];
    if !(m.flux() > 0.0) {
      continue;
    }
    let pred = streak_pred(render, i).map(|p| &moments[p]);
    let slot = render_slot(render[i].age_id_dbeta_flux[1]);
    let mut exposure = pc.exposure;
    if hdr.flags & DUST_VIEW_FLOW != 0 {
      exposure *= flow_factor(m.age(), flow);
    }
    let phot = SpeckPhotometry {
      albedo: pc.albedo,
      sun: m.sun_dir(),
      r_h_au: m.helio_au(),
    };
    let color = age_color(stream, m.age(), ttl);
    let v = SpeckView { exposure, ..view };
    for b in 0..SIZE_BINS as usize {
      let seg_t = bin_time_segment(m, pred, b);
      let sc = SpeckCluster::new(m, b, seg_t, slot, s_ref, q, ttl, fade_ratio);
      let me_df = sc.mean_rel(view.eye);
      let me = [
        me_df[0].hi + me_df[0].lo,
        me_df[1].hi + me_df[1].lo,
        me_df[2].hi + me_df[2].lo,
      ];
      n_eye_add(pyramid, n_eye_counts(sc.density_count_rel(me)));
      if !(hdr.d_max > 0.0) || sc.mant == 0 {
        continue;
      }
      // the whole bin against the shell first (most bins are far from the eye)
      let (y_eye, sigma_lo) = sc.eye_in_xi(me);
      if !sc.node_meets([0.0; 3], [1.0; 3], me, y_eye, sigma_lo, hdr.d_max, None) {
        continue;
      }
      let mut frame_nodes = pyramid[PYR_SPECKS_NODES as usize];
      let st = sc.walk(
        view.eye,
        hdr.d_max,
        view.fwd,
        DUST_SPECKS_MAX_NODES,
        Some(&mut frame_nodes),
        &mut |s: &Speck| {
          let drawn = pyramid[PYR_SPECKS as usize];
          if drawn >= hdr.budget {
            sat_add(&mut pyramid[PYR_SPECKS_OVERFLOW as usize], 1);
            stats.overflow += 1;
            return true;
          }
          let (on_screen, counts) =
            scatter_speck(s.dp, s.xsec_m2, s.hash, color, &phot, &v, layout, pyramid);
          if on_screen {
            pyramid[PYR_SPECKS as usize] = drawn + 1;
            stats.drawn += 1;
            sat_add(&mut pyramid[PYR_SPECKS_COUNTS as usize], counts);
            stats.counts = stats.counts.saturating_add(counts);
          }
          true
        },
      );
      pyramid[PYR_SPECKS_NODES as usize] = frame_nodes;
      stats.nodes += st.nodes as u64;
      stats.enumerated += st.specks;
      if st.truncated {
        // an incomplete walk: counted apart from the refused specks ([`PYR_SPECKS_TRUNCATED`]);
        // the shell reacts to the refused specks only (a truncated walk is walk cost, not demand)
        stats.truncated += 1;
        sat_add(&mut pyramid[PYR_SPECKS_TRUNCATED as usize], 1);
      }
    }
  }
  stats
}

/// What one [`specks_tier`] did.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpeckPassStats {
  /// specks inside the shell and in front of the eye (enumerated), drawn (on screen, within the
  /// budget), beyond the budget
  pub enumerated: u32,
  pub drawn: u32,
  pub overflow: u32,
  /// τ counts deposited
  pub counts: u32,
  pub nodes: u64,
  /// (cluster, bin) walks that hit the per-walk or the frame node cap (each counts one overflow)
  pub truncated: u32,
}

// ─── The shell cut of the continuum ─────────────────────────────────────────

/// Where a continuum packet stands against the speck shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellClass {
  /// wholly outside the ball: splat it
  Outside,
  /// wholly inside: the specks draw it, drop it (count its flux)
  Inside,
  /// across the boundary: split it once more, or splat it scaled by [`ShellCut::outside_fraction`]
  Straddles,
}

/// The camera-centred shell the continuum gives up to the specks: the ball of radius `d_max`
/// about the eye (local m). Built by the splat pass from the pyramid header
/// ([`ShellCut::from_header`]); inactive (`d_max ≤ 0`) when there are no specks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellCut {
  pub eye: [f32; 3],
  pub d_max: f32,
}

impl ShellCut {
  pub fn new(eye: [f32; 3], d_max: f32) -> Self {
    Self { eye, d_max }
  }
  /// From the frame's header word [`PYR_SPECKS_DMAX`].
  pub fn from_header(pyramid: &[u32], eye: [f32; 3]) -> Self {
    Self {
      eye,
      d_max: f32::from_bits(pyramid[PYR_SPECKS_DMAX as usize]),
    }
  }
  pub fn active(&self) -> bool {
    self.d_max > 0.0 && self.d_max.is_finite()
  }

  /// The packet's centre (its segment's midpoint), its 3σ reach (√(tr Σ), conservative) plus
  /// half its segment.
  fn extent(&self, p: &Packet) -> (f32, f32) {
    let centre = add3(p.mean, scale3(p.seg, 0.5));
    let dist = norm3(sub3(centre, self.eye));
    let tr = (p.cov[0] + p.cov[3] + p.cov[5]).max(0.0);
    (dist, 3.0 * sqrtf(tr) + 0.5 * norm3(p.seg))
  }

  /// Wholly inside / outside / straddling, conservatively (the 3σ sphere of the trace).
  pub fn classify(&self, p: &Packet) -> ShellClass {
    if !self.active() {
      return ShellClass::Outside;
    }
    let (dist, reach) = self.extent(p);
    if dist + reach < self.d_max {
      ShellClass::Inside
    } else if dist - reach > self.d_max {
      ShellClass::Outside
    } else {
      ShellClass::Straddles
    }
  }

  /// The fraction of the packet's mass outside the ball, for a straddling packet that is not
  /// split further: `1 − P(|x − eye| < d_max)`, the Gaussian integrated over the ball
  /// approximately as the product of its radial part (the 1D Gaussian along the eye direction,
  /// σ_r, cut at ±d_max about the eye; a capsule's segment enters it exactly as a uniform shift
  /// of the centre along the ray, closed form) and its transverse part (a disc of radius d_max in the
  /// transverse plane, isotropised: `1 − exp(−d²/2σ_⊥²)`; the eye lies on the radial axis), times
  /// `κ = 1 − exp(−d²/2σ_⊥²)/3`, the ball's volume over the cylinder's where the packet is far
  /// larger than the shell. Exact in the two limits (a packet far smaller than the shell: 0 or 1
  /// by its centre, the boundary a plane across it; far larger: `G(eye) · (4/3)π d³`), and the
  /// straddling layer's mass is small in both (`specks_and_continuum_conserve_the_flux`).
  pub fn outside_fraction(&self, p: &Packet) -> f32 {
    if !self.active() {
      return 1.0;
    }
    let centre = add3(p.mean, scale3(p.seg, 0.5));
    let d = sub3(centre, self.eye);
    let dist = norm3(d);
    let r_hat = if dist > 0.0 {
      scale3(d, 1.0 / dist)
    } else {
      [0.0, 0.0, 1.0]
    };
    // the segment's radial and transverse parts: the radial part is uniform (exact below), the
    // transverse part joins the Gaussian with its uniform variance
    let s_r = dot3(p.seg, r_hat);
    let s_t = sub3(p.seg, scale3(r_hat, s_r));
    let c = &p.cov;
    let tr = (c[0] + c[3] + c[5]).max(1e-30);
    let var_r = (c[0] * r_hat[0] * r_hat[0]
      + c[3] * r_hat[1] * r_hat[1]
      + c[5] * r_hat[2] * r_hat[2]
      + 2.0
        * (c[1] * r_hat[0] * r_hat[1] + c[2] * r_hat[0] * r_hat[2] + c[4] * r_hat[1] * r_hat[2]))
      .max(0.0);
    let sigma_r = sqrtf(var_r).max(1e-20);
    let var_t = ((tr - var_r) * 0.5 + dot3(s_t, s_t) / 24.0).max(1e-30);
    let dm = self.d_max;
    // the radial mass inside (−d, d) of a Gaussian of σ_r whose centre is uniform over
    // [dist − |s_r|/2, dist + |s_r|/2]: with G(z) = z Φ(z) + φ(z) (∫Φ), the mean of
    // Φ((d − x)/σ) − Φ((−d − x)/σ) over the interval in closed form
    let half = 0.5 * fabs(s_r);
    let phi_cdf = |z: f32| 0.5 * (1.0 + super::erf(z * core::f32::consts::FRAC_1_SQRT_2));
    let radial = if half > 1e-3 * sigma_r {
      let g = |z: f32| z * phi_cdf(z) + expf(-0.5 * z * z) * 0.398_942_3;
      let (x1, x2) = (dist - half, dist + half);
      let sr = sigma_r;
      sr / (x2 - x1)
        * ((g((dm - x1) / sr) - g((dm - x2) / sr)) - (g((-dm - x1) / sr) - g((-dm - x2) / sr)))
    } else {
      phi_cdf((dm - dist) / sigma_r) - phi_cdf((-dm - dist) / sigma_r)
    };
    // the eye is on the radial axis: its transverse offset from the centre is zero by
    // construction, the disc term is the mass within d of the axis
    let e_t = expf(-dm * dm / (2.0 * var_t));
    let transverse = 1.0 - e_t;
    let kappa = 1.0 - e_t / 3.0;
    (1.0 - (radial * transverse * kappa).clamp(0.0, 1.0)).clamp(0.0, 1.0)
  }
}

#[cfg(test)]
mod tests;
