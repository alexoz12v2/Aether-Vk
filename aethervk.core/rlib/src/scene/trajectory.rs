//! trajectory module.

use crate::scene::Component;
use alloc::vec::Vec;

#[derive(Clone, Debug)]
/// TrajectoryComponent holds bezier control points to render a trajectory.
/// Note: This component should host either a `TransformComponent` or a `HighResTransformComponent`.
pub struct TrajectoryComponent {
  pub control_points: Vec<[f32; 4]>, // Homogeneous (x*w, y*w, z*w, w)
  /// f64 source of truth (AU, entity-local, w = 1) when the producer has it: the renderer makes
  /// it camera-relative in f64 before the f32 upload. Empty: `control_points` is used instead.
  pub control_points_f64: Vec<[f64; 3]>,
  pub color: [f32; 4],
  pub line_width: f32,
  pub texture_id: u32,
  pub subdivisions_per_segment: u32,
}

impl Component for TrajectoryComponent {}

impl TrajectoryComponent {
  /// TODO: Document this item
  pub fn new(
    control_points: Vec<[f32; 4]>,
    color: [f32; 4],
    line_width: f32,
    texture_id: u32,
    subdivisions_per_segment: u32,
  ) -> Self {
    Self {
      control_points,
      control_points_f64: Vec::new(),
      color,
      line_width,
      texture_id,
      subdivisions_per_segment,
    }
  }

  /// Builds the component from f64 control points (AU, w = 1), keeping both representations.
  pub fn from_f64(
    control_points_f64: Vec<[f64; 3]>,
    color: [f32; 4],
    line_width: f32,
    texture_id: u32,
    subdivisions_per_segment: u32,
  ) -> Self {
    let control_points = control_points_f64
      .iter()
      .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 1.0])
      .collect();
    Self {
      control_points,
      control_points_f64,
      color,
      line_width,
      texture_id,
      subdivisions_per_segment,
    }
  }

  /// 1. A generic animation method that updates the control points in-place.
  /// It avoids allocating or pushing new points by mutating the existing slice.
  pub fn animate_points<F>(&mut self, dt: f32, mut update_fn: F)
  where
    // Closure receives: (index, delta_time, mutable_point_reference)
    F: FnMut(usize, f32, &mut [f32; 4]),
  {
    for (index, point) in self.control_points.iter_mut().enumerate() {
      update_fn(index, dt, point);
    }
  }

  /// 2. Animates the trajectory into an open infinity symbol where one
  /// endpoint "chases" the other. Leverages the generic `animate_points`.
  ///
  /// * `dt` - Frame delta time (microseconds)
  /// * `elapsed_time` - Passed mutably to accumulate absolute time for the parametric curve (microseconds).
  /// * `scale_x` / `scale_y` - Controls the spatial width/height of the symbol.
  /// * `speed` - Determines how quickly the curve travels.
  pub fn animate_infinity_chase(
    &mut self,
    dt: aethervk_oshal_rlib::os::time::timeus_t,
    elapsed_time: &mut aethervk_oshal_rlib::os::time::timeus_t,
    scale_x: f32,
    scale_y: f32,
    speed: f32,
  ) {
    // Accumulate absolute time to prevent physics drift
    *elapsed_time += dt;
    let current_time_sec = (*elapsed_time as f64 / 1_000_000.0) as f32 * speed;

    let num_points = self.control_points.len();
    if num_points < 4 {
      return;
    }

    // A complete closed figure-eight loop requires a parameter distance of 2π.
    // By setting the length of our "string" to less than 2π (e.g., 1.5π),
    // we leave a visible gap, causing the tail to continuously chase the head.
    let trail_length = core::f32::consts::PI * 1.5;

    // We are building cubic bezier segments. Each segment has 4 points.
    // To keep them connected, CP0 of segment N must equal CP3 of segment N-1.
    // Instead of animating all points independently, we can evaluate the
    // lemniscate at N positions and use them as the connected segment points.
    // A cubic bezier defined by points on a curve won't be perfectly smooth
    // without proper tangent computation, but we can set the control points
    // such that it closely approximates the curve.

    // Actually, since we want a continuous curve along the trail length, we can
    // just treat the underlying array as a set of continuous cubic segments:
    let num_segments = num_points / 4;
    let spacing = trail_length / (num_segments * 3) as f32; // Each segment spans 3 "steps" of t

    #[allow(unused_imports)]
    use aethervk_oshal_rlib::math::floating::FloatOps;

    for i in 0..num_segments {
      let seg_idx = i * 4;

      for p in 0..4 {
        // The global index along the entire curve's length:
        let global_t_idx = (i * 3) + p;

        let t = current_time_sec - (global_t_idx as f32 * spacing);

        let x = scale_x * f32::sin(t);
        let y = 0.0; // Flat in Y depth (forward axis)
        let z = scale_y * f32::sin(t) * f32::cos(t); // Undulate vertically

        let point = &mut self.control_points[seg_idx + p];
        let w = point[3];

        point[0] = x * w;
        point[1] = y * w;
        point[2] = z * w;
      }
    }
  }
}

/// A heliocentric `SUN_ECLIPJ2000` state sample: km, km/s at `time_sec` (any monotonic parameter:
/// TAI seconds for ephemeris samples, an anomaly for analytical conics).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectorySample {
  pub position_km: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
  pub velocity_km: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
  pub time_sec: f64,
}

/// Cubic Hermite → Bezier control points in AU (4 per segment), f64, for a `TrajectoryComponent`
/// on a child of root: handles are `p ± v·Δt/3`.
pub fn bezier_from_samples_au_f64(samples: &[TrajectorySample]) -> Vec<[f64; 3]> {
  use aethervk_oshal_rlib::math::vector::Vector3;
  const KM_TO_AU: f64 = crate::simulation_api::reposition::KM_TO_AU;
  let mut control_points = Vec::<[f64; 3]>::with_capacity(4 * samples.len().saturating_sub(1));
  for w in samples.windows(2) {
    let (s0, s1) = (&w[0], &w[1]);
    let dt = s1.time_sec - s0.time_sec;
    for p in [
      s0.position_km,
      s0.position_km + s0.velocity_km * (dt / 3.0),
      s1.position_km - s1.velocity_km * (dt / 3.0),
      s1.position_km,
    ] {
      let cp = p * KM_TO_AU;
      control_points.push([cp.x(), cp.y(), cp.z()]);
    }
  }
  control_points
}

/// f32 `[x, y, z, 1]` version of [`bezier_from_samples_au_f64`].
pub fn bezier_from_samples_au(samples: &[TrajectorySample]) -> Vec<[f32; 4]> {
  bezier_from_samples_au_f64(samples)
    .into_iter()
    .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 1.0])
    .collect()
}

/// Path the comet actually followed during the current run (SPK states sampled by the logic
/// thread), drawn in yellow next to the analytical SBDB track. Lives on the
/// `effective_comet_trajectory` entity together with the `TrajectoryComponent` built from it; both
/// are removed on simulation reset / comet change (the entity stays).
#[derive(Debug, Clone, Default)]
pub struct EffectiveTrajectoryComponent {
  pub samples: Vec<TrajectorySample>,
  /// Latest comet state (every tick): the renderer draws a live segment from the last sample to
  /// it, so the track always ends on the nucleus.
  pub head: Option<TrajectorySample>,
}

impl Component for EffectiveTrajectoryComponent {}

impl EffectiveTrajectoryComponent {
  /// Sim time between two samples (6 h): ~120 samples for a one-month run.
  pub const SAMPLE_INTERVAL_SEC: f64 = 6.0 * 3600.0;
  pub const COLOR: [f32; 4] = [1.0, 0.85, 0.1, 1.0];

  /// Appends `s` if at least [`Self::SAMPLE_INTERVAL_SEC`] passed since the last sample. Time
  /// going backwards (scrub/rewind) drops the samples after `s.time_sec` first. Returns whether
  /// the sample list changed.
  pub fn push_sample(&mut self, s: TrajectorySample) -> bool {
    let before = self.samples.len();
    self.samples.retain(|x| x.time_sec <= s.time_sec);
    let rewound = self.samples.len() != before;
    match self.samples.last() {
      Some(last) if s.time_sec - last.time_sec < Self::SAMPLE_INTERVAL_SEC => rewound,
      _ => {
        self.samples.push(s);
        true
      }
    }
  }

  pub fn trajectory(&self) -> TrajectoryComponent {
    TrajectoryComponent::from_f64(
      bezier_from_samples_au_f64(&self.samples),
      Self::COLOR,
      2.0,
      0,
      32,
    )
  }

  /// Hermite segment (4 control points, AU f64) from the last sample to `head`, if `head` is newer.
  pub fn head_segment(&self) -> Option<Vec<[f64; 3]>> {
    let (last, head) = (self.samples.last()?, self.head?);
    (head.time_sec > last.time_sec).then(|| bezier_from_samples_au_f64(&[*last, head]))
  }
}

#[cfg(test)]
mod effective_trajectory_tests {
  use super::*;
  use aethervk_oshal_rlib::math::vector::{Vector3, vec3f64::DVec3};

  fn s(t: f64) -> TrajectorySample {
    TrajectorySample {
      position_km: DVec3::from_components(t, 0.0, 0.0),
      velocity_km: DVec3::from_components(1.0, 0.0, 0.0),
      time_sec: t,
    }
  }

  #[test]
  fn samples_are_gated_by_interval_and_monotonic() {
    let dt = EffectiveTrajectoryComponent::SAMPLE_INTERVAL_SEC;
    let mut c = EffectiveTrajectoryComponent::default();
    assert!(c.push_sample(s(0.0)));
    assert!(
      !c.push_sample(s(dt * 0.5)),
      "too close to the previous sample"
    );
    assert!(c.push_sample(s(dt)));
    assert!(c.push_sample(s(3.0 * dt)));
    assert!(c.samples.windows(2).all(|w| w[0].time_sec < w[1].time_sec));
    assert_eq!(c.samples.len(), 3);
  }

  #[test]
  fn rewind_drops_future_samples() {
    let dt = EffectiveTrajectoryComponent::SAMPLE_INTERVAL_SEC;
    let mut c = EffectiveTrajectoryComponent::default();
    for k in 0..5 {
      c.push_sample(s(k as f64 * dt));
    }
    assert!(c.push_sample(s(1.5 * dt)), "rewind must change the list");
    assert_eq!(
      c.samples.iter().map(|x| x.time_sec).collect::<Vec<_>>(),
      [0.0, dt]
    );
  }

  /// Closest point on a Hermite-approximated unit circle (km units) from an outside point.
  #[test]
  fn closest_point_on_circle_track() {
    let samples: Vec<TrajectorySample> = (0..=16)
      .map(|k| {
        let a = core::f64::consts::TAU * k as f64 / 16.0;
        TrajectorySample {
          position_km: DVec3::from_components(a.cos(), a.sin(), 0.0),
          velocity_km: DVec3::from_components(-a.sin(), a.cos(), 0.0),
          time_sec: a,
        }
      })
      .collect();
    // control points in AU: scale back to km for the check
    let km_per_au = crate::simulation_api::reposition::AU_TO_KM;
    let cps: Vec<[f64; 3]> = bezier_from_samples_au_f64(&samples)
      .into_iter()
      .map(|p| [p[0] * km_per_au, p[1] * km_per_au, p[2] * km_per_au])
      .collect();
    let (q, d) = closest_point_on_bezier_track(&cps, [2.0 * 0.6, 2.0 * 0.8, 0.0]).unwrap();
    assert!((d - 1.0).abs() < 1e-3, "distance {d}");
    assert!(
      (q[0] - 0.6).abs() < 1e-3 && (q[1] - 0.8).abs() < 1e-3,
      "{q:?}"
    );
    assert!(closest_point_on_bezier_track(&[], [0.0; 3]).is_none());
  }

  #[test]
  fn km_labels() {
    assert_eq!(format_km(12.3456), "12.35 km");
    assert_eq!(format_km(12_345.6), "12346 km");
    assert!(format_km(1.9e7).contains("e7"));
  }

  /// The head moves every tick without being gated, and the head segment ends exactly on it.
  #[test]
  fn head_segment_ends_on_the_head() {
    let dt = EffectiveTrajectoryComponent::SAMPLE_INTERVAL_SEC;
    let mut c = EffectiveTrajectoryComponent::default();
    c.push_sample(s(0.0));
    assert!(c.head_segment().is_none(), "no head yet");
    c.head = Some(s(dt * 0.25));
    assert!(
      !c.push_sample(s(dt * 0.25)),
      "head update is not a new sample"
    );
    let seg = c.head_segment().expect("segment last sample -> head");
    let au = 1.0 / crate::simulation_api::reposition::AU_TO_KM;
    assert_eq!(seg.len(), 4);
    // KM_TO_AU is 1/AU_TO_KM to ~6e-11 relative: compare relatively
    assert!(seg[0][0].abs() < 1e-18, "{seg:?}");
    assert!((seg[3][0] / (dt * 0.25 * au) - 1.0).abs() < 1e-9, "{seg:?}");
    c.head = Some(s(-1.0));
    assert!(
      c.head_segment().is_none(),
      "a head older than the last sample draws nothing"
    );
  }

  #[test]
  fn bezier_from_samples_is_hermite() {
    // straight line at constant velocity: handles at exactly 1/3 and 2/3
    let km_per_au = crate::simulation_api::reposition::AU_TO_KM;
    let a = TrajectorySample {
      position_km: DVec3::from_components(0.0, 0.0, 0.0),
      velocity_km: DVec3::from_components(km_per_au, 0.0, 0.0),
      time_sec: 0.0,
    };
    let b = TrajectorySample {
      position_km: DVec3::from_components(km_per_au, 0.0, 0.0),
      time_sec: 1.0,
      ..a
    };
    let cps = bezier_from_samples_au(&[a, b]);
    let xs: Vec<f32> = cps.iter().map(|p| p[0]).collect();
    for (x, e) in xs.iter().zip([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]) {
      assert!((x - e).abs() < 1e-6, "{xs:?}");
    }
    assert!(cps.iter().all(|p| p[1] == 0.0 && p[2] == 0.0 && p[3] == 1.0));
  }
}

/// Screen-space distance annotation `|--|` between two global points (AU, f64) with a label
/// centred above it, drawn on the overlay (UI quads + text): ticks and text keep a constant pixel
/// size at any zoom. Used for the reference-position error of the committed comet.
#[derive(Clone, Debug)]
pub struct ScreenMeasurementComponent {
  pub from_au: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
  pub to_au: aethervk_oshal_rlib::math::vector::vec3f64::DVec3,
  pub label: alloc::string::String,
  pub color: [f32; 4],
  pub font_atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  pub font_hash: u64,
}

impl Component for ScreenMeasurementComponent {}

/// Closest point (and its distance) of a cubic Bezier track (4 control points per segment, any
/// units) to `p`: coarse sampling per segment, then golden-section refinement around the best
/// sample.
pub fn closest_point_on_bezier_track(cps: &[[f64; 3]], p: [f64; 3]) -> Option<([f64; 3], f64)> {
  fn eval(s: &[[f64; 3]], t: f64) -> [f64; 3] {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    core::array::from_fn(|c| (0..4).map(|j| w[j] * s[j][c]).sum())
  }
  let dist2 = |a: [f64; 3]| (a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2) + (a[2] - p[2]).powi(2);
  const SAMPLES: usize = 32;
  let mut best: Option<(usize, usize, f64)> = None; // (segment, sample, d²)
  for (si, seg) in cps.chunks_exact(4).enumerate() {
    for k in 0..=SAMPLES {
      let d2 = dist2(eval(seg, k as f64 / SAMPLES as f64));
      if best.map_or(true, |b| d2 < b.2) {
        best = Some((si, k, d2));
      }
    }
  }
  let (si, k, _) = best?;
  let seg = &cps[si * 4..si * 4 + 4];
  let (mut a, mut b) = (
    (k as f64 - 1.0).max(0.0) / SAMPLES as f64,
    (k as f64 + 1.0).min(SAMPLES as f64) / SAMPLES as f64,
  );
  const INV_PHI: f64 = 0.618_033_988_749_894_8;
  for _ in 0..80 {
    let c = b - (b - a) * INV_PHI;
    let d = a + (b - a) * INV_PHI;
    if dist2(eval(seg, c)) < dist2(eval(seg, d)) {
      b = d;
    } else {
      a = c;
    }
  }
  let q = eval(seg, 0.5 * (a + b));
  Some((q, dist2(q).sqrt()))
}

/// Human-readable km distance for annotation labels.
pub fn format_km(km: f64) -> alloc::string::String {
  if km >= 1e7 {
    alloc::format!("{km:.3e} km")
  } else if km >= 100.0 {
    alloc::format!("{km:.0} km")
  } else {
    alloc::format!("{km:.2} km")
  }
}
