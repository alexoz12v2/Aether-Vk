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
  /// particle-system → root rotation quaternion (xyzw) at `t_start + dur`
  pub rot_end: [f32; 4],
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
  pub _pad: [u32; 4],
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
  /// `x` TTL (s), `y`..`w` 0
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

#[inline]
fn nlerp(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let sgn = if d < 0.0 { -1.0 } else { 1.0 };
  let q = [
    a[0] + (sgn * b[0] - a[0]) * t,
    a[1] + (sgn * b[1] - a[1]) * t,
    a[2] + (sgn * b[2] - a[2]) * t,
    a[3] + (sgn * b[3] - a[3]) * t,
  ];
  let n = <f32 as FloatLike>::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  let n = if n > 0.0 { 1.0 / n } else { 1.0 };
  [q[0] * n, q[1] * n, q[2] * n, q[3] * n]
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
  let global = batch.first_index.wrapping_add(j);
  let h0 = pcg(batch.seed ^ pcg(global));
  let h1 = pcg(h0);
  let h2 = pcg(h1);
  let h3 = pcg(h2);
  let h4 = pcg(h3);

  // staggered emission time: one sample per time stratum
  let count = batch.count.max(1) as f32;
  let u_t = ((j as f32) + u01(h0)) / count;
  let t_start = Df::new(batch.comet_r_t_hi[3], batch.comet_r_t_lo[3]);
  let dur = Df::new(batch.comet_v_dur_hi[3], batch.comet_v_dur_lo[3]);
  let dt_in = dur.mul_f(u_t);
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
  let rot = nlerp(batch.rot_start, batch.rot_end, u_t);
  let mut dir = qrot(rot, dir_ps);
  // night side: reflect across the terminator plane (keeps the batch mass, stays deterministic)
  let sun = normalize3(rc.hi);
  let sun = [-sun[0], -sun[1], -sun[2]];
  let ds = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
  if ds < 0.0 {
    dir = [
      dir[0] - 2.0 * ds * sun[0],
      dir[1] - 2.0 * ds * sun[1],
      dir[2] - 2.0 * ds * sun[2],
    ];
  }

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
  let ttl = frame.ttl[0];
  if !(age_f >= 0.0) || age_f > ttl {
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
  // fade the last 10% of the lifetime
  let fade = ((1.0 - age_f / ttl) * 10.0).clamp(0.0, 1.0);
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
/// the next one. The cluster count follows the steady-state budget
/// `capacity · Δt / TTL`, bounded by `free_slots`.
pub fn plan_batch(
  acc: &mut EmissionAccumulator,
  q_dust_kgs: f64,
  dt_s: f64,
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
  acc.mass_g += q * 1e3 * dt_s;
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
  pub fn push_batch(&mut self, mut desc: DustBatch, t_end_s: f64, mass_g: f64) -> DustBatch {
    debug_assert!(desc.count <= self.free_slots());
    let first = self.head;
    self.head += desc.count as u64;
    // RNG index = monotonic index (wrapping u32), ring slot = index & mask
    desc.first_index = first as u32;
    desc.ring_mask = self.mask();
    self.batches.push_back(LiveBatch {
      first,
      count: desc.count,
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

/// minimum unscaled interval between two emissions of one system (µs)
pub const EMIT_INTERVAL_UNSCALED_US: i64 = 100_000;
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
}

/// Everything the renderer needs to evaluate and draw one system this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustDrawState {
  pub first_slot: u32,
  pub live_count: u32,
  /// ring capacity (render-time children budget)
  pub capacity: u32,
  /// compute timeline value the graphics submit must wait on (0 = none)
  pub compute_wait: u64,
  pub frame: DustFrame,
  /// unit anti-sun direction (root axes), w = solar gravity at the jet (m/s²)
  pub anti_sun_g: [f32; 4],
  /// mean cluster flux (m²), for exposure normalization
  pub mean_cluster_flux: f32,
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
  /// cross-section per gram at the configured grain radius (m²/g)
  pub fn xsec_per_g_ref(&self) -> f32 {
    let s_m = (self.diameter_um * 0.5).max(1e-3) * 1e-6;
    3.0 / (4.0 * self.density_gcm3.max(1e-3) * 1e6 * s_m)
  }
}

#[derive(Debug, Clone)]
pub struct DustHostState {
  pub ring: RingState,
  pub acc: EmissionAccumulator,
  /// unscaled µs of the last emission gate pass (0 = not started)
  pub last_emit_unscaled_us: i64,
  /// monotonic counter of batch descriptor uploads (selects the upload slot)
  pub upload_seq: u64,
  /// monotonic counter of created batches (seed, low-discrepancy shift)
  pub batch_seq: u64,
  /// jet state at the start of the pending emission window
  pub window_jet: Option<JetState>,
  /// jet time at the last emission gate
  pub last_gate_t_s: Option<f64>,
  /// latest jet state (written every tick)
  pub jet: Option<JetState>,
  /// TTL (scaled s) of the latest tick
  pub ttl_s: f64,
  /// cross-section per gram at the reference grain size (m²/g)
  pub xsec_per_g_ref: f32,
}

impl DustHostState {
  pub fn new(capacity: u32) -> Self {
    Self {
      ring: RingState::new(capacity),
      acc: EmissionAccumulator::default(),
      last_emit_unscaled_us: 0,
      upload_seq: 0,
      batch_seq: 0,
      window_jet: None,
      last_gate_t_s: None,
      jet: None,
      ttl_s: 0.0,
      xsec_per_g_ref: 0.0,
    }
  }

  /// forgets every cluster and the emission history (simulation reset)
  pub fn reset(&mut self) {
    *self = Self::new(self.ring.capacity);
  }

  /// One logic tick of host-side emission bookkeeping at jet state `jet`.
  ///
  /// - time scrubbed backwards: drops batches emitted after `jet.t_s` and restarts the window;
  /// - retires expired batches;
  /// - re-emits (at most [`REEMIT_PER_TICK`]) batches invalidated by a restore;
  /// - every [`EMIT_INTERVAL_UNSCALED_US`] closes the emission window `[window_jet.t_s, jet.t_s]`
  ///   into one batch (mass conserving, see [`plan_batch`]).
  ///
  /// Returns the descriptors to emit, in order (ring slots already reserved, [`READY_PENDING`]).
  /// The caller records them and calls `ring.mark_submitted(value)` after the submit.
  pub fn tick(
    &mut self,
    jet: JetState,
    now_unscaled_us: i64,
    cfg: &DustEmitConfig,
  ) -> alloc::vec::Vec<DustBatch> {
    self.ttl_s = cfg.ttl_s;
    self.xsec_per_g_ref = cfg.xsec_per_g_ref();
    if let Some(prev) = self.jet {
      if jet.t_s < prev.t_s {
        self.ring.rewind(jet.t_s);
        self.acc = EmissionAccumulator::default();
        self.window_jet = None;
        self.last_gate_t_s = None;
      }
    }
    self.jet = Some(jet);
    self.ring.retire(jet.t_s, cfg.ttl_s);
    let mut out = self.ring.take_reemit(REEMIT_PER_TICK);

    let Some(window) = self.window_jet else {
      // start the first window here
      self.window_jet = Some(jet);
      self.acc = EmissionAccumulator {
        clusters: 0.0,
        mass_g: 0.0,
        window_start_s: jet.t_s,
      };
      self.last_emit_unscaled_us = now_unscaled_us;
      return out;
    };
    if now_unscaled_us - self.last_emit_unscaled_us < EMIT_INTERVAL_UNSCALED_US {
      return out;
    }
    self.last_emit_unscaled_us = now_unscaled_us;
    // time since the last gate (the accumulator already holds everything before it)
    let last_gate_t = self.last_gate_t_s.unwrap_or(window.t_s);
    let dt = jet.t_s - last_gate_t;
    if !(dt > 0.0) {
      return out;
    }
    self.last_gate_t_s = Some(jet.t_s);
    self.acc.window_start_s = window.t_s;
    let free = self.emit_free_slots();
    let Some(plan) = plan_batch(
      &mut self.acc,
      cfg.q_dust_kgs,
      dt,
      jet.t_s,
      cfg.ttl_s,
      self.ring.capacity,
      free,
    ) else {
      return out;
    };

    let mut desc: DustBatch = bytemuck::Zeroable::zeroed();
    desc.set_comet(
      window.r_m,
      window.v_ms,
      window.t_s,
      (jet.t_s - window.t_s).max(0.0),
    );
    desc.rot_start = window.rot;
    desc.rot_end = jet.rot;
    desc.jet_dir_aperture = [
      cfg.jet_dir[0],
      cfg.jet_dir[1],
      cfg.jet_dir[2],
      cfg.aperture_rad,
    ];
    let shift = (self.batch_seq as f64 * 0.618_033_988_749_895).fract() as f32;
    let (size_params, vel_params, mass_params) = batch_params(
      &cfg.dist,
      cfg.diameter_um,
      cfg.density_gcm3,
      cfg.beta_ref,
      cfg.v_mean,
      cfg.v_std,
      plan.mass_g,
      shift,
    );
    desc.size_params = size_params;
    desc.vel_params = vel_params;
    desc.mass_params = mass_params;
    desc.count = plan.count;
    desc.seed = cfg.seed ^ pcg(self.batch_seq as u32 ^ 0x5DEE_CE66);
    self.batch_seq += 1;
    out.push(self.ring.push_batch(desc, jet.t_s, plan.mass_g));
    self.window_jet = Some(jet);
    out
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
    let live_mass: f64 = self.ring.live_mass_g();
    Some(DustDrawState {
      first_slot,
      live_count,
      capacity: self.ring.capacity,
      compute_wait,
      frame: DustFrame::new(jet.r_m, jet.t_s, [0.0, 0.0, 0.0, 1.0], self.ttl_s as f32),
      anti_sun_g: [
        (jet.r_m[0] / rn) as f32,
        (jet.r_m[1] / rn) as f32,
        (jet.r_m[2] / rn) as f32,
        g as f32,
      ],
      mean_cluster_flux: (live_mass * self.xsec_per_g_ref as f64 / live_count as f64) as f32,
    })
  }
}

#[cfg(test)]
mod tests;
