//! Dust trajectories view aid ([`DUST_VIEW_TRAJECTORIES`]): the paths of a sparse, stable subset
//! of real clusters from their emission to now, drawn as lines (one `TrajectoryComponent` per
//! jet, the jet's stream colour). Built on the logic thread after the emission tick from the
//! host ring's batch descriptors, with the dust's own physics: every cluster is rebuilt with
//! [`emit_cluster`] and every sample is the cluster's closed-form heliocentric position at that
//! age ([`kepler::propagate`] with the cluster's β-reduced μ, as [`evaluate_cluster_in`]), drawn
//! in the comet frame moved to now (the comet's free-fall motion the emission used subtracted,
//! the current comet position added back, [`TrajectorySample`]): a heliocentric arc is dominated
//! by the comet's own orbital motion (1e9 km streaks), the comet-frame path is the flow away from
//! the nucleus the dust shows. The last vertex is exactly the grain's current position.
//!
//! The subset is chosen by a hash of the cluster's own randomness (its window seed and in-batch
//! index: deterministic, so a line never flickers from tick to tick) against a threshold derived
//! from the live cluster count: ~[`DUST_TRAJECTORY_TARGET`] lines whatever the population, at
//! most [`DUST_TRAJECTORY_MAX`] × [`DUST_TRAJECTORY_SAMPLES`] propagations per update.

use super::*;
use crate::scene::trajectory::TrajectoryComponent;
use alloc::vec::Vec;

/// lines aimed at (the subset's expected size once the population exceeds it)
pub const DUST_TRAJECTORY_TARGET: u32 = 200;
/// hard cap on the lines of one update (the selection's tail is dropped past it)
pub const DUST_TRAJECTORY_MAX: usize = 250;
/// samples per line, emission (age 0) to now, geometrically spaced in age ([`sample_ages`])
pub const DUST_TRAJECTORY_SAMPLES: usize = 32;
/// first sample age after the emission (s), at least: 10 min (the near-nucleus part of a line
/// is where the curvature is)
pub const DUST_TRAJECTORY_FIRST_AGE_S: f64 = 600.0;
/// ... or this fraction of the grain's age, whichever is larger
pub const DUST_TRAJECTORY_FIRST_AGE_FRACTION: f64 = 1e-4;
/// line width (px) of the drawn paths
pub const DUST_TRAJECTORY_LINE_WIDTH: f32 = 1.5;
/// tessellation of each Hermite segment between two samples
pub const DUST_TRAJECTORY_SUBDIVISIONS: u32 = 4;
/// salt of the selection hash (distinct from the tracer dots' `is_tracer` and the emission's)
const TRAJECTORY_SALT: u32 = 0x5E7A_11B9;

/// Selection threshold for `live` live clusters: a cluster whose [`cluster_hash`] is `≤` it is
/// drawn, so about [`DUST_TRAJECTORY_TARGET`] are (all of them below the target).
pub fn trajectory_threshold(live: u64) -> u32 {
  if live <= DUST_TRAJECTORY_TARGET as u64 {
    return u32::MAX;
  }
  (((DUST_TRAJECTORY_TARGET as u64) << 32) / live).min(u32::MAX as u64) as u32
}

/// The selection hash of cluster `j` of `batch`: a function of the cluster's own randomness
/// (`emit_cluster`'s `h0`), so the same cluster whatever ring slot or tick.
#[inline]
pub fn cluster_hash(batch: &DustBatch, j: u32) -> u32 {
  pcg(pcg(batch.seed ^ pcg(j)) ^ TRAJECTORY_SALT)
}

/// Whether cluster `j` of `batch` is drawn as a trajectory at `threshold`
/// ([`trajectory_threshold`]).
#[inline]
pub fn is_trajectory_tracer(batch: &DustBatch, j: u32, threshold: u32) -> bool {
  cluster_hash(batch, j) <= threshold
}

/// One sample of a path, in the **comet frame moved to now**: `r = r_grain(t_e + a) −
/// r_comet(t_e + a) + r_comet(now)` (heliocentric m) and `v = v_grain − v_comet` (m/s) at age
/// `a` of the grain emitted at `t_e`. The comet's path is the one the emission used: the batch's
/// jet-site state propagated in free fall ([`comet_state_at`]), so at `a = 0` the sample lies on
/// the nucleus (the site's offset from the centre) at the current nucleus position, and at
/// `a = age` the shift cancels and the sample is the grain's current heliocentric position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
  pub age_s: f64,
  pub r_m: [f64; 3],
  pub v_ms: [f64; 3],
}

/// The batch's jet-site state (heliocentric m, m/s) propagated in free fall under the Sun from
/// the batch's `t_start` to `t_s`: the comet's motion as the emission sees it (`emit_cluster`'s
/// `kepler::propagate(rc0, vc0, SUN_MU, dt)`), `t_s` as df64.
pub fn comet_state_at(batch: &DustBatch, t_s: Df) -> (Df3, Df3) {
  let rc0 = Df3 {
    hi: [batch.comet_r_t_hi[0], batch.comet_r_t_hi[1], batch.comet_r_t_hi[2]],
    lo: [batch.comet_r_t_lo[0], batch.comet_r_t_lo[1], batch.comet_r_t_lo[2]],
  };
  let vc0 = Df3 {
    hi: [batch.comet_v_dur_hi[0], batch.comet_v_dur_hi[1], batch.comet_v_dur_hi[2]],
    lo: [batch.comet_v_dur_lo[0], batch.comet_v_dur_lo[1], batch.comet_v_dur_lo[2]],
  };
  let t_start = Df::new(batch.comet_r_t_hi[3], batch.comet_r_t_lo[3]);
  kepler::propagate(&rc0, &vc0, consts::SUN_MU, t_s.sub(t_start))
}

/// The `n` sample ages (s) of a grain aged `age_s`: 0, then geometric from
/// `max(DUST_TRAJECTORY_FIRST_AGE_S, age · DUST_TRAJECTORY_FIRST_AGE_FRACTION)` up to `age_s`
/// (the last one exact). Uniform spacing put the first Hermite segment hundreds of km long on a
/// month-old grain and let it loop near the nucleus; geometric ages keep the near-nucleus
/// segments short. A short age (below the first step, or `n < 3`) is spaced uniformly.
pub fn sample_ages(age_s: f64, n: usize) -> Vec<f64> {
  let n = n.max(2);
  let first = DUST_TRAJECTORY_FIRST_AGE_S.max(age_s * DUST_TRAJECTORY_FIRST_AGE_FRACTION);
  if n < 3 || !(age_s > 2.0 * first) {
    return (0..n).map(|k| age_s * k as f64 / (n - 1) as f64).collect();
  }
  let ratio = age_s / first;
  let m = (n - 2) as f64;
  let mut ages: Vec<f64> = core::iter::once(0.0)
    .chain((0..n - 1).map(|i| first * <f64 as FloatLike>::pow(ratio, i as f64 / m)))
    .collect();
  ages[n - 1] = age_s;
  ages
}

/// `n` samples of the cluster's path from emission to `t_now_s`, at [`sample_ages`] (the last
/// one at the cluster's current age), in the comet frame moved to now ([`TrajectorySample`]):
/// the grain by the propagation the renderer uses (`evaluate_cluster_in`: Kepler motion under
/// `μ·(1 − β)` from the emission state), the comet by [`comet_state_at`] of the cluster's
/// `batch`. The last sample is the grain's propagated position itself (no shift: exactly where
/// the dust is drawn). `n < 2` gives the two endpoints; a negative age (not emitted yet) gives
/// nothing.
pub fn trajectory_samples(
  c: &DustCluster,
  batch: &DustBatch,
  t_now_s: f64,
  n: usize,
) -> Vec<TrajectorySample> {
  let n = n.max(2);
  let t_now = Df::from_f64(t_now_s);
  let age = t_now.sub(c.t0());
  let age_f = age.to_f64();
  if !(age_f >= 0.0) {
    return Vec::new();
  }
  let mu = consts::SUN_MU.mul(two_sum_one_minus(c.beta()));
  let (r0, v0) = (c.r0(), c.v0());
  let (rc_now, vc_now) = comet_state_at(batch, t_now);
  let ages = sample_ages(age_f, n);
  (0..n)
    .map(|k| {
      let last = k + 1 == n;
      // the exact current age at the last sample (`t_now − t0` in df64), the grid before
      let a = if last { age } else { Df::from_f64(ages[k]) };
      let (r, v) = kepler::propagate(&r0, &v0, mu, a);
      let (rc, vc) = if last {
        (rc_now, vc_now)
      } else {
        comet_state_at(batch, c.t0().add(a))
      };
      let r_m = if last {
        r.to_f64()
      } else {
        r.sub(&rc).add(&rc_now).to_f64()
      };
      TrajectorySample {
        age_s: a.to_f64(),
        r_m,
        v_ms: v.sub(&vc).to_f64(),
      }
    })
    .collect()
}

/// One drawn path: the tier, the cluster (as emitted) and its samples.
#[derive(Debug, Clone, PartialEq)]
pub struct DustTrajectory {
  pub tier: u32,
  pub cluster: DustCluster,
  pub samples: Vec<TrajectorySample>,
}

/// Live clusters of the system's host rings (every tier, provisional window included).
pub fn live_cluster_count(sys: &DustSystemState) -> u64 {
  sys.tiers.iter().map(|t| t.ring.live() as u64).sum()
}

/// The clusters chosen for lines among `batches` (`(tier, descriptor)`, each with its `count`
/// clusters) at [`trajectory_threshold`]`(live)`: `(tier, descriptor, in-batch index)`, in ring
/// order, at most [`DUST_TRAJECTORY_MAX`].
pub fn select_trajectories<'a>(
  batches: impl IntoIterator<Item = (u32, &'a DustBatch)>,
  live: u64,
) -> Vec<(u32, DustBatch, u32)> {
  let threshold = trajectory_threshold(live);
  let mut out = Vec::with_capacity(DUST_TRAJECTORY_MAX.min(live as usize + 1));
  for (tier, b) in batches {
    for j in 0..b.count {
      if is_trajectory_tracer(b, j, threshold) {
        out.push((tier, *b, j));
        if out.len() >= DUST_TRAJECTORY_MAX {
          return out;
        }
      }
    }
  }
  out
}

/// The paths of the system at `t_now_s`: the selected clusters whose age lies in their tier's
/// band (what the tier draws, `evaluate_cluster_in`'s cull), each with
/// [`DUST_TRAJECTORY_SAMPLES`] samples. Empty before the first tick or while nothing is live.
pub fn build_trajectories(sys: &DustSystemState, t_now_s: f64) -> Vec<DustTrajectory> {
  let live = live_cluster_count(sys);
  if live == 0 {
    return Vec::new();
  }
  let batches = sys
    .tiers
    .iter()
    .filter(|t| t.jet.is_some())
    .flat_map(|t| t.ring.batches.iter().map(move |b| (t.tier, &b.desc)));
  let mut out = Vec::new();
  for (tier, desc, j) in select_trajectories(batches, live) {
    let Some(host) = sys.tiers.iter().find(|t| t.tier == tier) else {
      continue;
    };
    let (min_age, max_age) = host.age_band_s(host.ttl_s);
    let cluster = emit_cluster(&desc, j);
    let age = Df::from_f64(t_now_s).sub(cluster.t0()).to_f64();
    if !(age >= min_age) || !(age >= 0.0) || age > max_age {
      continue;
    }
    let samples = trajectory_samples(&cluster, &desc, t_now_s, DUST_TRAJECTORY_SAMPLES);
    if samples.len() < 2 {
      continue;
    }
    out.push(DustTrajectory {
      tier,
      cluster,
      samples,
    });
  }
  out
}

/// Cubic Hermite → Bezier control points (AU, heliocentric, the comet frame moved to now, 4 per
/// segment) of the paths, one segment per consecutive sample pair, lines back to back (the
/// renderer draws segments independently: nothing joins one path to the next). Handles are
/// `p ± v·Δt/3` with the comet-frame velocity.
pub fn trajectory_control_points_au(paths: &[DustTrajectory]) -> Vec<[f64; 3]> {
  let inv_au = 1.0 / AU_M;
  let mut cp = Vec::with_capacity(paths.iter().map(|p| 4 * p.samples.len().saturating_sub(1)).sum());
  for path in paths {
    for w in path.samples.windows(2) {
      let (s0, s1) = (&w[0], &w[1]);
      let dt = s1.age_s - s0.age_s;
      let h = dt / 3.0;
      let pts = [
        s0.r_m,
        [s0.r_m[0] + s0.v_ms[0] * h, s0.r_m[1] + s0.v_ms[1] * h, s0.r_m[2] + s0.v_ms[2] * h],
        [s1.r_m[0] - s1.v_ms[0] * h, s1.r_m[1] - s1.v_ms[1] * h, s1.r_m[2] - s1.v_ms[2] * h],
        s1.r_m,
      ];
      for p in pts {
        cp.push([p[0] * inv_au, p[1] * inv_au, p[2] * inv_au]);
      }
    }
  }
  cp
}

/// The jet's line component at `t_now_s` (heliocentric AU control points, for an entity under
/// the root like the comet tracks), or `None` when no path is live.
pub fn trajectory_component(
  sys: &DustSystemState,
  t_now_s: f64,
  color: [f32; 4],
) -> Option<TrajectoryComponent> {
  let paths = build_trajectories(sys, t_now_s);
  if paths.is_empty() {
    return None;
  }
  Some(TrajectoryComponent::from_f64(
    trajectory_control_points_au(&paths),
    color,
    DUST_TRAJECTORY_LINE_WIDTH,
    0,
    DUST_TRAJECTORY_SUBDIVISIONS,
  ))
}

/// Line colour of a jet's `stream_color`: the stream colour lightened half-way to white, opaque,
/// so the lines read over the dust of the same hue (the composite lays the dust over the lines
/// with its display opacity: a line keeps `1 − a` of its colour under dust of opacity `a`).
pub fn trajectory_line_color(stream: [f32; 4]) -> [f32; 4] {
  let l = |c: f32| (0.5 + 0.5 * c.clamp(0.0, 1.0)).clamp(0.0, 1.0);
  [l(stream[0]), l(stream[1]), l(stream[2]), 1.0]
}

/// `(lines, extent_km)` of a component's control points (`trajectory_control_points_au`
/// layout): the line count from the samples per line, and the largest distance (km) of any
/// control point from the first one (the drawn size). Diagnostics.
pub fn describe_control_points(cp: &[[f64; 3]]) -> (usize, f64) {
  let per_line = 4 * (DUST_TRAJECTORY_SAMPLES - 1);
  let lines = cp.len() / per_line.max(1);
  let Some(first) = cp.first() else {
    return (0, 0.0);
  };
  let extent_au = cp
    .iter()
    .map(|p| {
      let d = [p[0] - first[0], p[1] - first[1], p[2] - first[2]];
      (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    })
    .fold(0.0, f64::max);
  (lines, extent_au * AU_M * 1e-3)
}
