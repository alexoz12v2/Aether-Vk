use aethervk_oshal_rlib::math::{
  matrix::{MatrixVectorMul, mat4f64::Mat4x4f64},
  vector::{Vector, Vector4, vec4f64::Vec4f64},
};
use alloc::vec::Vec;

pub struct VisibleTrajectorySample {
  pub global_t: f64,
  pub ndc_pos: [f32; 2],
}

/// Sample a trajectory's Bézier curve and project each point into NDC.
/// control_points: homogeneous [x*w, y*w, z*w, w] — de-homogenized before interpolation.
pub fn sample_and_project_trajectory(
  control_points: &[[f32; 4]],
  mvp_f64: &Mat4x4f64,
  steps_per_segment: usize,
) -> Vec<VisibleTrajectorySample> {
  let mut visible_samples = Vec::new();
  let num_segments = control_points.len() / 4;

  for seg_idx in 0..num_segments {
    let base = seg_idx * 4;
    let p0_h = control_points[base];
    let p1_h = control_points[base + 1];
    let p2_h = control_points[base + 2];
    let p3_h = control_points[base + 3];

    let w0 = p0_h[3] as f64;
    let w1 = p1_h[3] as f64;
    let w2 = p2_h[3] as f64;
    let w3 = p3_h[3] as f64;

    if w0.abs() < 1e-10 || w1.abs() < 1e-10 || w2.abs() < 1e-10 || w3.abs() < 1e-10 {
      continue;
    }

    let p0 = [
      p0_h[0] as f64 / w0,
      p0_h[1] as f64 / w0,
      p0_h[2] as f64 / w0,
    ];
    let p1 = [
      p1_h[0] as f64 / w1,
      p1_h[1] as f64 / w1,
      p1_h[2] as f64 / w1,
    ];
    let p2 = [
      p2_h[0] as f64 / w2,
      p2_h[1] as f64 / w2,
      p2_h[2] as f64 / w2,
    ];
    let p3 = [
      p3_h[0] as f64 / w3,
      p3_h[1] as f64 / w3,
      p3_h[2] as f64 / w3,
    ];

    for step in 0..=steps_per_segment {
      let t = step as f64 / steps_per_segment as f64;
      let u = 1.0 - t;

      let w1_b = u * u * u;
      let w2_b = 3.0 * u * u * t;
      let w3_b = 3.0 * u * t * t;
      let w4_b = t * t * t;

      let local_x = w1_b * p0[0] + w2_b * p1[0] + w3_b * p2[0] + w4_b * p3[0];
      let local_y = w1_b * p0[1] + w2_b * p1[1] + w3_b * p2[1] + w4_b * p3[1];
      let local_z = w1_b * p0[2] + w2_b * p1[2] + w3_b * p2[2] + w4_b * p3[2];

      let local_pos = Vec4f64::from_components(local_x, local_y, local_z, 1.0);
      let clip_pos = mvp_f64.mul_vector(local_pos);

      if clip_pos.w() <= 0.0 {
        continue;
      }

      let ndc_x = (clip_pos.x() / clip_pos.w()) as f32;
      let ndc_y = (clip_pos.y() / clip_pos.w()) as f32;

      // Use a slightly tighter bound than 1.0
      if ndc_x >= -0.95 && ndc_x <= 0.95 && ndc_y >= -0.95 && ndc_y <= 0.95 {
        visible_samples.push(VisibleTrajectorySample {
          global_t: seg_idx as f64 + t,
          ndc_pos: [ndc_x, ndc_y],
        });
      }
    }
  }

  visible_samples
}

/// Return the index of the best-scoring visible sample.
/// Score = ndc_dist_to_center + 0.05 / ndc_dist_to_parent.max(0.001)
pub fn find_best_trajectory_anchor(
  samples: &[VisibleTrajectorySample],
  parent_ndc: Option<[f32; 2]>,
) -> Option<usize> {
  samples
    .iter()
    .enumerate()
    .min_by(|(_, a), (_, b)| {
      let dist_to_center_a = a.ndc_pos[0].powi(2) + a.ndc_pos[1].powi(2);
      let dist_to_center_b = b.ndc_pos[0].powi(2) + b.ndc_pos[1].powi(2);

      let penalty_a = if let Some(p) = parent_ndc {
        let d = (a.ndc_pos[0] - p[0]).powi(2) + (a.ndc_pos[1] - p[1]).powi(2);
        0.05 / d.max(0.001)
      } else {
        0.0
      };

      let penalty_b = if let Some(p) = parent_ndc {
        let d = (b.ndc_pos[0] - p[0]).powi(2) + (b.ndc_pos[1] - p[1]).powi(2);
        0.05 / d.max(0.001)
      } else {
        0.0
      };

      let score_a = dist_to_center_a + penalty_a;
      let score_b = dist_to_center_b + penalty_b;

      score_a.partial_cmp(&score_b).unwrap_or(core::cmp::Ordering::Equal)
    })
    .map(|(idx, _)| idx)
}

/// Exponential smoothing of the Bézier t parameter.
/// DECAY_TAU = 0.15 s. Snaps if |ideal - current| > snap_threshold or first frame.
pub fn smooth_trajectory_t(
  current_t: Option<f64>,
  ideal_t: f64,
  dt_s: f32,
  snap_threshold: f64,
) -> f64 {
  let Some(current) = current_t else {
    return ideal_t;
  };

  if (ideal_t - current).abs() > snap_threshold {
    return ideal_t;
  }

  const DECAY_TAU: f64 = 0.15;
  if dt_s < 1e-6 {
    return current;
  }

  let decay = (-(dt_s as f64) / DECAY_TAU).exp();
  current * decay + ideal_t * (1.0 - decay)
}

/// Evaluate the exact 3D Cartesian position on the curve at global_t.
/// De-homogenizes the four control points before Bézier evaluation.
pub fn evaluate_bezier_at(control_points: &[[f32; 4]], global_t: f64) -> Option<[f64; 3]> {
  let num_segments = control_points.len() / 4;
  if num_segments == 0 {
    return None;
  }

  let mut seg_idx = global_t.floor() as usize;
  let mut local_t = global_t.fract();

  if seg_idx >= num_segments {
    seg_idx = num_segments - 1;
    local_t = 1.0;
  }

  let base = seg_idx * 4;
  let p0_h = control_points[base];
  let p1_h = control_points[base + 1];
  let p2_h = control_points[base + 2];
  let p3_h = control_points[base + 3];

  let w0 = p0_h[3] as f64;
  let w1 = p1_h[3] as f64;
  let w2 = p2_h[3] as f64;
  let w3 = p3_h[3] as f64;

  if w0.abs() < 1e-10 || w1.abs() < 1e-10 || w2.abs() < 1e-10 || w3.abs() < 1e-10 {
    return None;
  }

  let p0 = [
    p0_h[0] as f64 / w0,
    p0_h[1] as f64 / w0,
    p0_h[2] as f64 / w0,
  ];
  let p1 = [
    p1_h[0] as f64 / w1,
    p1_h[1] as f64 / w1,
    p1_h[2] as f64 / w1,
  ];
  let p2 = [
    p2_h[0] as f64 / w2,
    p2_h[1] as f64 / w2,
    p2_h[2] as f64 / w2,
  ];
  let p3 = [
    p3_h[0] as f64 / w3,
    p3_h[1] as f64 / w3,
    p3_h[2] as f64 / w3,
  ];

  let u = 1.0 - local_t;
  let w1_b = u * u * u;
  let w2_b = 3.0 * u * u * local_t;
  let w3_b = 3.0 * u * local_t * local_t;
  let w4_b = local_t * local_t * local_t;

  let local_x = w1_b * p0[0] + w2_b * p1[0] + w3_b * p2[0] + w4_b * p3[0];
  let local_y = w1_b * p0[1] + w2_b * p1[1] + w3_b * p2[1] + w4_b * p3[1];
  let local_z = w1_b * p0[2] + w2_b * p1[2] + w3_b * p2[2] + w4_b * p3[2];

  Some([local_x, local_y, local_z])
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::vector::vec4f64::Vec4f64;

  #[test]
  fn test_dehomogenize_w_not_one() {
    let mvp = Mat4x4f64::identity();
    // Test points that will de-homogenize to [0.5, 0.5, 3.0, 1.0] so they pass NDC cull
    let cps = [
      [1.0, 1.0, 6.0, 2.0],
      [1.0, 1.0, 6.0, 2.0],
      [1.0, 1.0, 6.0, 2.0],
      [1.0, 1.0, 6.0, 2.0],
    ];
    let samples = sample_and_project_trajectory(&cps, &mvp, 1);
    assert!(!samples.is_empty());
    
    let pos = evaluate_bezier_at(&cps, 0.5).unwrap();
    // After de-homogenization (1.0/2.0), x=0.5, y=0.5, z=3.0
    assert!((pos[0] - 0.5).abs() < 1e-6);
    assert!((pos[1] - 0.5).abs() < 1e-6);
    assert!((pos[2] - 3.0).abs() < 1e-6);
  }

  #[test]
  fn test_sample_empty() {
    let mvp = Mat4x4f64::identity();
    let samples = sample_and_project_trajectory(&[], &mvp, 10);
    assert!(samples.is_empty());
  }

  #[test]
  fn test_frustum_cull_behind_camera() {
    // A matrix that puts points behind the camera (w <= 0)
    let mvp = Mat4x4f64::from_cols(
      Vec4f64::from_components(1.0, 0.0, 0.0, 0.0),
      Vec4f64::from_components(0.0, 1.0, 0.0, 0.0),
      Vec4f64::from_components(0.0, 0.0, 1.0, 0.0),
      Vec4f64::from_components(0.0, 0.0, 0.0, -1.0),
    );
    let cps = [
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
    ];
    let samples = sample_and_project_trajectory(&cps, &mvp, 10);
    assert!(samples.is_empty());
  }

  #[test]
  fn test_smooth_t_first_frame() {
    assert_eq!(smooth_trajectory_t(None, 42.0, 0.1, 5.0), 42.0);
  }

  #[test]
  fn test_smooth_t_snap_large_jump() {
    assert_eq!(smooth_trajectory_t(Some(0.0), 10.0, 0.1, 5.0), 10.0);
  }

  #[test]
  fn test_smooth_t_converges() {
    let t = smooth_trajectory_t(Some(0.0), 1.0, 10.0, 5.0); // large dt
    assert!((t - 1.0).abs() < 1e-5);
  }

  #[test]
  fn test_find_best_anchor_center_wins() {
    let samples = vec![
      VisibleTrajectorySample {
        global_t: 0.0,
        ndc_pos: [0.9, 0.9],
      }, // edge
      VisibleTrajectorySample {
        global_t: 1.0,
        ndc_pos: [0.1, 0.1],
      }, // center
    ];
    assert_eq!(find_best_trajectory_anchor(&samples, None), Some(1));
  }
}