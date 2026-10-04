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
/// render-time children (sub-splats) per cluster, minimum (dense ring)
pub const CHILDREN_PER_CLUSTER: u32 = 8;
/// render-time children per cluster, maximum (sparse ring; `dust.vert` hashes with this stride)
pub const MAX_CHILDREN_PER_CLUSTER: u32 = 64;

/// Children per cluster for `live` clusters of a ring of `capacity`: keeps the instance count
/// near `capacity · CHILDREN_PER_CLUSTER`, so a sparse ring (early in a run: it fills over one TTL)
/// is drawn as a dense, smooth cloud instead of isolated blobs. Total light per cluster is
/// unchanged (each child carries `1/children` of the flux).
pub fn render_children(capacity: u32, live: u32) -> u32 {
  if live == 0 {
    return CHILDREN_PER_CLUSTER;
  }
  let budget = capacity as u64 * CHILDREN_PER_CLUSTER as u64;
  ((budget / live as u64) as u32).clamp(CHILDREN_PER_CLUSTER, MAX_CHILDREN_PER_CLUSTER)
}
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

/// age of the dust column that defines the exposure reference ([`DustEmitConfig::tau_ref`])
pub const TAU_REF_AGE_S: f64 = 86400.0;
/// splat radius clamp in pixels (`dust.vert`)
pub const SPLAT_MIN_PX: f32 = 1.5;
pub const SPLAT_MAX_PX: f32 = 48.0;
/// child footprint radius as a fraction of the cluster spread (`dust.vert`)
pub const SPLAT_CHILD_RADIUS_FRAC: f32 = 0.5;

/// Mirror of the `dust.vert` footprint: `(r_px, r_draw_m)`, the drawn splat radius in pixels
/// (clamped to `[SPLAT_MIN_PX, SPLAT_MAX_PX]`) and the same radius back in metres. `p11` is the
/// projection y scale, `px_to_ndc_y = 2 / viewport height`, `units_per_m` the layer unit.
pub fn splat_footprint(
  spread_m: f32,
  units_per_m: f32,
  p11: f32,
  clip_w: f32,
  px_to_ndc_y: f32,
) -> (f32, f32) {
  let r_units = (spread_m * SPLAT_CHILD_RADIUS_FRAC).max(1.0) * units_per_m;
  let px_per_unit = p11 / clip_w / px_to_ndc_y;
  let r_px = (r_units * px_per_unit).clamp(SPLAT_MIN_PX, SPLAT_MAX_PX);
  (r_px, r_px / px_per_unit / units_per_m)
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
  /// `x` child velocity dispersion σ_v (m/s), `y` grain radius (µm), `z` cross-section per gram
  /// (m²/g), `w` child β half-spread
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
  /// particle-system local position (m), `w` cluster spread radius (m)
  pub pos_size: [f32; 4],
  /// `x` age (s), `y` ring slot (u32 bits, stable child seed), `z` child β half-spread,
  /// `w` flux (cross-section m², 0 = culled)
  pub age_id_dbeta_flux: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustRenderCluster>() == 32);

/// Emission batch descriptor (one per emission per system). 192 bytes.
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
  /// `x` batch mass (g), `y` density (g/cm³), `z` per-batch low-discrepancy shift, `w` 0
  pub mass_params: [f32; 4],
  pub first_index: u32,
  pub count: u32,
  pub ring_mask: u32,
  pub seed: u32,
  /// jet site illumination over the batch window, see [`LitWindow`]: `x` spin phase `ψ_start` at
  /// `t_start` (`[−π, π)`), `y` lit half arc `ψ0` (`(0, π)`), `z` total lit phase (rad), `w` mode
  /// ([`LIT_MODE_ALWAYS`] or [`LIT_MODE_PERIODIC`]). Emission times are spread over lit time only.
  pub lit: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustBatch>() == 192);

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
  /// particle-system heliocentric position (m) at `t_now`, `w` = `t_now` (s); hi
  pub ps_r_t_hi: [f32; 4],
  /// ... low part
  pub ps_r_t_lo: [f32; 4],
  /// root → particle-system rotation quaternion (xyzw), i.e. the conjugate of the entity rotation
  pub rot_inv: [f32; 4],
  /// age band: `x` max age (s, the TTL), `y` min age (s, 0 for the youngest tier), `z` 1 = no fade
  /// at the max age (an older tier takes over), `w` 0
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
  /// `DustRenderBuffer` (compact, index = live-range offset)
  pub render: u64,
  /// render-time children per cluster, see [`render_children`]
  pub children: u32,
  pub live_count: u32,
  /// particle-system local metres → clip
  pub mvp: [f32; 16],
  /// rgb stream color, a = flux scale
  pub color: [f32; 4],
  /// unit anti-sun direction (ps frame), w = solar gravity at the comet (m/s²)
  pub anti_sun_g: [f32; 4],
  /// x units per metre, y P00, z P11, w 2 / viewport height
  pub params: [f32; 4],
}
const _: () = assert!(core::mem::size_of::<DustDrawPushConstants>() == 128);

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

/// `fract(j·φ⁻¹ + shift)` in 32-bit fixed point: full 24-bit resolution for any `j` and
/// bit-identical on GPU (an f32 `j * 0.618 + shift` loses resolution for large `j`, and drivers
/// may fuse it into an FMA). `shift ∈ [0, 1)` with 24-bit resolution.
#[inline]
pub fn lattice_u01(j: u32, shift: f32) -> f32 {
  let shift_u = ((shift * 16_777_216.0) as u32) << 8;
  u01(j.wrapping_mul(0x9E37_79B9).wrapping_add(shift_u))
}

// ─────────────────────────────────────────────────────────────────────────────
// Emission (mirrors dust_emit.comp)
// ─────────────────────────────────────────────────────────────────────────────

/// Builds cluster `j` (`0 ≤ j < batch.count`) of `batch`; it goes to ring slot
/// `(batch.first_index + j) & batch.ring_mask`.
pub fn emit_cluster(batch: &DustBatch, j: u32) -> DustCluster {
  // randomness from the in-batch index: the seed is unique per emission window, so a window
  // always produces the same clusters whatever ring slots it lands on (deterministic seek)
  let h0 = pcg(batch.seed ^ pcg(j));
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);

  // staggered emission time: one sample per time stratum
  let count = batch.count.max(1) as f32;
  let u_t = ((j as f32) + u01(h0)) / count;
  let t_start = Df::new(batch.comet_r_t_hi[3], batch.comet_r_t_lo[3]);
  let dur = Df::new(batch.comet_v_dur_hi[3], batch.comet_v_dur_lo[3]);
  // lit time only: dark phases of the jet site get no clusters (full df64 when always lit)
  let dt_in = if batch.lit[3] == LIT_MODE_PERIODIC {
    Df::from_f32(lit_time_map(u_t, batch.lit, batch.spin[3], dur.hi + dur.lo))
  } else {
    dur.mul_f(u_t)
  };
  let t0 = t_start.add(dt_in);

  // jet location at t0 (comet free fall over the sub-interval)
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
  let (rc, vc) = kepler::propagate(&rc0, &vc0, consts::SUN_MU, dt_in);

  // direction: cone in the particle-system frame, rotated to root with the interpolated attitude
  let jet = [
    batch.jet_dir_aperture[0],
    batch.jet_dir_aperture[1],
    batch.jet_dir_aperture[2],
  ];
  let dir_ps = sample_cone(u01(h1), u01(h2), jet, batch.jet_dir_aperture[3]);
  // exact nucleus spin from the window start
  let spin = qaxis_angle(
    [batch.spin[0], batch.spin[1], batch.spin[2]],
    batch.spin[3] * (dt_in.hi + dt_in.lo),
  );
  let rot = qmul(spin, batch.rot_start);
  let dir = qrot(rot, dir_ps);

  // grain size: log-uniform proposal, rank-1 lattice (decorrelated from time order)
  let s_min = batch.size_params[0];
  let s_max = batch.size_params[1];
  let u_s = lattice_u01(j, batch.mass_params[2]);
  let s_um = s_min * <f32 as FloatLike>::pow(s_max / s_min, u_s);
  let beta = batch.vel_params[3] / s_um;

  // importance weight against the mass distribution: m_j = M/N · norm · s^(4−q)
  let mass_g = batch.mass_params[0] / count
    * batch.size_params[3]
    * <f32 as FloatLike>::pow(s_um, batch.size_params[2]);

  // ejection speed: v_ref · sqrt(s_ref / s) · (1 + σ_rel N(0,1)), clamped at 0
  let v_mean = batch.vel_params[0] * <f32 as FloatLike>::sqrt(batch.vel_params[2] / s_um);
  let v_ej = (v_mean * (1.0 + batch.vel_params[1] * gauss(u01(h3), u01(h4)))).max(0.0);
  let sigma_v = (v_mean * batch.vel_params[1]).max(v_mean * CHILD_SIGMA_V_REL);

  let v0 = vc.add(&Df3::from_f32([
    dir[0] * v_ej,
    dir[1] * v_ej,
    dir[2] * v_ej,
  ]));

  // cross-section per gram: π s² / (4/3 π s³ ρ) = 3 / (4 ρ s)  [s in m, ρ in g/m³]
  let rho_g_m3 = batch.mass_params[1] * 1.0e6;
  let xsec_per_g = 3.0 / (4.0 * rho_g_m3 * s_um * 1.0e-6);
  // children spread β over this cluster's size stratum: Δln s = ln(s_max/s_min) / N
  let dbeta = beta * 0.5 * <f32 as FloatLike>::ln(s_max / s_min) / count;

  DustCluster {
    r0_t0_hi: [rc.hi[0], rc.hi[1], rc.hi[2], t0.hi],
    r0_t0_lo: [rc.lo[0], rc.lo[1], rc.lo[2], t0.lo],
    v0_hi_beta: [v0.hi[0], v0.hi[1], v0.hi[2], beta],
    v0_lo_mass: [v0.lo[0], v0.lo[1], v0.lo[2], mass_g],
    misc: [sigma_v, s_um, xsec_per_g, dbeta],
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
    age_id_dbeta_flux: [age_f, f32::from_bits(slot), c.misc[3], flux],
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
  low_discrepancy_shift: f32,
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
    [mass_g as f32, density_gcm3, low_discrepancy_shift, 0.0],
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
}

impl RingState {
  pub fn new(capacity: u32) -> Self {
    assert!(capacity.is_power_of_two());
    Self {
      capacity,
      head: 0,
      tail: 0,
      batches: alloc::collections::VecDeque::new(),
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
  }
  /// marks the whole ring content invalid (after a scene restore)
  pub fn invalidate_gpu(&mut self) {
    for b in self.batches.iter_mut() {
      b.ready = READY_NEEDS_EMIT;
    }
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
/// band. Older tiers have longer windows (fewer, heavier clusters per scaled day) and keep only the
/// larger grains (`s_min_factor`): long history at the same memory.
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

#[derive(Debug, Clone)]
pub struct DustHostState {
  pub ring: RingState,
  /// first slot of this tier's sub-ring in the system's cluster / render buffers
  pub ring_base: u32,
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

  /// forgets every cluster and the emission history (simulation reset)
  pub fn reset(&mut self) {
    *self = Self::with_band(self.ring.capacity, self.ring_base, self.band, self.prestart);
  }

  /// `(min, max)` age (scaled s) of this tier for a TTL
  pub fn age_band_s(&self, ttl_s: f64) -> (f64, f64) {
    (self.band.min_ttl * ttl_s, self.band.max_ttl * ttl_s)
  }

  /// Window length (scaled s) of the emission grid for a TTL.
  pub fn window_len_s(ttl_s: f64) -> f64 {
    (ttl_s / WINDOWS_PER_TTL).max(1.0)
  }

  /// Clusters per closed window: the steady-state budget spread over one TTL of windows.
  pub fn clusters_per_window(&self) -> u32 {
    ((self.ring.capacity as f64 * BUDGET_SAFETY / WINDOWS_PER_TTL) as u32).max(1)
  }

  /// One logic tick of host-side emission at scaled time `t_now_s`, **deterministic in time**:
  /// emission happens on a fixed scaled-time grid of windows `[kΔ, (k+1)Δ)`,
  /// `Δ = ttl / WINDOWS_PER_TTL`, and window `k` always produces the same batch (jet state at
  /// `kΔ` from `jet_at`, mass `q(r)·lit time`, budgeted count, seed from `k`). The dust at any
  /// epoch is a pure function of the parameters and the epoch, so playing, pausing, changing the
  /// speed or seeking all give the same result:
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
    let full = self.clusters_per_window() as f64;
    let want = (-<f64 as FloatLike>::floor(-full * (dur / dt_w).clamp(0.0, 1.0))).max(1.0) as u32;
    let count = want.min(self.emit_free_slots());
    if count == 0 {
      return None;
    }
    let mut desc: DustBatch = bytemuck::Zeroable::zeroed();
    desc.set_comet(jet0.r_m, jet0.v_ms, t0, dur);
    desc.rot_start = jet0.rot;
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
    let shift = (k as f64 * 0.618_033_988_749_895).rem_euclid(1.0) as f32;
    let (size_params, vel_params, mass_params) = batch_params(
      &dist,
      cfg.diameter_um,
      cfg.density_gcm3,
      cfg.beta_ref,
      cfg.v_mean,
      cfg.v_std,
      mass_g,
      shift,
    );
    desc.size_params = size_params;
    desc.vel_params = vel_params;
    desc.mass_params = mass_params;
    desc.count = count;
    desc.seed = cfg.seed ^ pcg(k as u32 ^ 0x5DEE_CE66);
    Some(self.ring.push_window_batch(desc, t0 + dur, mass_g, k))
  }

  /// emission slots usable now (ring free space minus the guard band)
  pub fn emit_free_slots(&self) -> u32 {
    self.ring.free_slots().saturating_sub(self.ring.capacity / RING_GUARD_DIVISOR)
  }

  /// render-side view of the current state, `None` before the first tick or when empty
  pub fn draw_state(&self) -> Option<DustDrawState> {
    let jet = self.jet?;
    let (first_slot, live_count, compute_wait) = self.ring.drawable();
    if live_count == 0 {
      return None;
    }
    let rn = norm(jet.r_m);
    let g = SUN_MU_M3_S2 / (rn * rn);
    let (min_age, max_age) = self.age_band_s(self.ttl_s);
    Some(DustDrawState {
      ring_base: self.ring_base,
      first_slot,
      live_count,
      capacity: self.ring.capacity,
      compute_wait,
      frame: DustFrame::new(jet.r_m, jet.t_s, [0.0, 0.0, 0.0, 1.0], max_age as f32)
        .with_band(min_age as f32, self.band.fade),
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

/// default age tiers: `[0, 1)`, `[1, 8)`, `[8, 64)` TTL (30 d → 240 d → ~5.3 yr)
pub const DUST_TIERS_DEFAULT: usize = 3;
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
}

impl DustSystemState {
  /// `capacity` (power of two) split among [`dust_tier_count`] tiers: 1/2, 1/4, 1/4.
  pub fn new(capacity: u32) -> Self {
    Self::with_tiers(capacity, dust_tier_count())
  }

  pub fn with_tiers(capacity: u32, tiers: usize) -> Self {
    let n = tiers.clamp(1, DUST_TIERS_DEFAULT);
    const BANDS: [(f64, f64, f64); DUST_TIERS_DEFAULT] =
      [(0.0, 1.0, 1.0), (1.0, 8.0, 8.0), (8.0, 64.0, 32.0)];
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
        let t = DustHostState::with_band(caps[k].max(1), base, band, n > 1);
        base += caps[k];
        t
      })
      .collect();
    Self {
      tiers,
      upload_seq: 0,
      tau_ref: 0.0,
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
    let mut states: alloc::vec::Vec<DustDrawState> =
      self.tiers.iter().filter_map(|t| t.draw_state()).collect();
    for s in &mut states {
      s.tau_ref = self.tau_ref as f32;
    }
    states
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
