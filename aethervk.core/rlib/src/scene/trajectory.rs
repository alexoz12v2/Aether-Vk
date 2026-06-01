//! trajectory module.

use crate::scene::Component;
use alloc::vec::Vec;

#[derive(Clone, Debug)]
/// TrajectoryComponent holds bezier control points to render a trajectory.
/// Note: This component should host either a `TransformComponent` or a `HighResTransformComponent`.
pub struct TrajectoryComponent {
  pub control_points: Vec<[f32; 4]>, // Homogeneous (x*w, y*w, z*w, w)
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
