//! Camera-relative (RTE) trajectory control points, computed in f64.
//!
//! Trajectory control points are AU-scale. Transforming them to camera-relative coordinates on
//! the GPU (or in f32 on the CPU) quantises positions to the f32 ulp of ~1 AU (≈ 18 km), which
//! hides a track passing through a km-sized nucleus. Here the subtraction is done in f64 and only
//! the (small) camera-relative result is cast to f32.
//!
//! Segments passing close to the camera are additionally split (de Casteljau, f64) until each
//! piece is about the size of the view there, so the GPU never sums huge clip-space terms (whose
//! f32 cancellation error is several pixels) or tessellates a 10⁵ km chord across a 5 km view.

use crate::scene::{CameraProjection, HighResTransformComponent, trajectory::TrajectoryComponent};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32},
};
use alloc::vec::Vec;

type P = [f64; 3];

/// How large the view is at a given distance from the camera, in the layer units.
#[derive(Debug, Clone, Copy)]
pub struct ViewScale {
  /// Orthographic half height (constant with distance), 0 for perspective.
  pub ortho_half_extent: f64,
  /// `tan(fov / 2)` for perspective, 0 for orthographic.
  pub tan_half_fov: f64,
}

impl ViewScale {
  pub fn from_projection(p: &CameraProjection) -> Self {
    match *p {
      CameraProjection::Perspective { fov, .. } => Self {
        ortho_half_extent: 0.0,
        tan_half_fov: (fov as f64 * 0.5).tan(),
      },
      CameraProjection::Orthographic { bottom, top, .. } => Self {
        ortho_half_extent: ((top - bottom) as f64).abs() * 0.5,
        tan_half_fov: 0.0,
      },
    }
  }

  /// Half extent of the view at distance `d` from the camera.
  pub fn extent_at(&self, d: f64) -> f64 {
    self.ortho_half_extent + d * self.tan_half_fov
  }
}

fn len(p: P) -> f64 {
  (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}
fn lerp(a: P, b: P, t: f64) -> P {
  [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ]
}

/// De Casteljau split of a cubic at `t`.
pub fn split(seg: [P; 4], t: f64) -> ([P; 4], [P; 4]) {
  let (a, b, c, d) = (seg[0], seg[1], seg[2], seg[3]);
  let ab = lerp(a, b, t);
  let bc = lerp(b, c, t);
  let cd = lerp(c, d, t);
  let abc = lerp(ab, bc, t);
  let bcd = lerp(bc, cd, t);
  let m = lerp(abc, bcd, t);
  ([a, ab, abc, m], [m, bcd, cd, d])
}

/// Splits `seg` (camera at the origin) until every piece the camera is close to (closer than the
/// piece size) is no larger than `PIECE_VIEWS` view extents measured at its distance. Far pieces
/// are left untouched, so the piece count grows only logarithmically.
pub fn refine_near_camera(seg: [P; 4], view: &ViewScale, out: &mut Vec<[P; 4]>) {
  const PIECE_VIEWS: f64 = 4.0;
  const MAX_DEPTH: u32 = 40;
  fn go(seg: [P; 4], view: &ViewScale, depth: u32, out: &mut Vec<[P; 4]>) {
    // the convex hull contains the curve: hull size and distance bound the curve
    let size = (0..4)
      .flat_map(|i| (i + 1..4).map(move |j| (i, j)))
      .map(|(i, j)| {
        len([
          seg[i][0] - seg[j][0],
          seg[i][1] - seg[j][1],
          seg[i][2] - seg[j][2],
        ])
      })
      .fold(0.0, f64::max);
    let dist = (seg.iter().map(|p| len(*p)).fold(f64::MAX, f64::min) - size).max(0.0);
    let extent = view.extent_at(dist).max(f64::MIN_POSITIVE);
    // only pieces the camera is close to (within one piece size) need splitting: further away the
    // f32 clip-space terms are no larger than the piece itself (an ortho extent does not grow with
    // distance, so without this a 1 AU segment was split into millions of view-sized pieces)
    if depth >= MAX_DEPTH || size <= PIECE_VIEWS * extent || dist > size {
      out.push(seg);
      return;
    }
    let (l, r) = split(seg, 0.5);
    go(l, view, depth + 1, out);
    go(r, view, depth + 1, out);
  }
  go(seg, view, 0, out)
}

/// Control points of `traj` (plus an optional extra trailing segment `extra`, 4 points in the
/// same units) relative to the camera, as `[x, y, z, 1]` f32, refined near the camera.
///
/// `rte` is the entity transform relative to the camera in the layer units (see
/// `Scene::compute_rte`): `p_rel = rte.position + rte.rotation · (rte.scale ⊙ p)`.
pub fn camera_relative_control_points(
  traj: &TrajectoryComponent,
  extra: &[P],
  rte: &HighResTransformComponent,
  view: ViewScale,
) -> Vec<[f32; 4]> {
  let src: Vec<P> = if traj.control_points_f64.is_empty() {
    traj
      .control_points
      .iter()
      .map(|p| {
        let w = if p[3] != 0.0 { p[3] as f64 } else { 1.0 };
        [p[0] as f64 / w, p[1] as f64 / w, p[2] as f64 / w]
      })
      .collect()
  } else {
    traj.control_points_f64.clone()
  };

  let (s, r, o) = (rte.scale, rte.rotation.to_quat(), rte.position);
  let identity_rot = {
    let x = r.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
    let y = r.rotate_vector(Vec3f32::from_components(0.0, 1.0, 0.0));
    (x.x() - 1.0).abs() < 1e-7 && (y.y() - 1.0).abs() < 1e-7
  };
  let to_rel = |p: &P| -> P {
    let scaled = [
      p[0] * s.x() as f64,
      p[1] * s.y() as f64,
      p[2] * s.z() as f64,
    ];
    let rotated = if identity_rot {
      scaled
    } else {
      // rotation only matters for rotated trajectory entities (none today): f32 is acceptable
      let v = r.rotate_vector(Vec3f32::from_components(
        scaled[0] as f32,
        scaled[1] as f32,
        scaled[2] as f32,
      ));
      [v.x() as f64, v.y() as f64, v.z() as f64]
    };
    [o.x() + rotated[0], o.y() + rotated[1], o.z() + rotated[2]]
  };

  let mut pieces = Vec::with_capacity(src.len() / 4 + 8);
  for seg in src.chunks_exact(4).chain(extra.chunks_exact(4)) {
    let seg = [
      to_rel(&seg[0]),
      to_rel(&seg[1]),
      to_rel(&seg[2]),
      to_rel(&seg[3]),
    ];
    refine_near_camera(seg, &view, &mut pieces);
  }
  pieces
    .iter()
    .flat_map(|seg| seg.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 1.0]))
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn eval(seg: &[P; 4], t: f64) -> P {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    core::array::from_fn(|c| (0..4).map(|j| w[j] * seg[j][c]).sum())
  }

  #[test]
  fn split_preserves_the_curve() {
    let seg = [
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 0.0],
      [3.0, -1.0, 1.0],
      [4.0, 0.0, 0.0],
    ];
    let (l, r) = split(seg, 0.3);
    for k in 0..=10 {
      let t = k as f64 / 10.0;
      let a = eval(&l, t);
      let b = eval(&seg, 0.3 * t);
      assert!(len([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) < 1e-12);
      let a = eval(&r, t);
      let b = eval(&seg, 0.3 + 0.7 * t);
      assert!(len([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) < 1e-12);
    }
  }

  /// A 1-AU-long straight segment passing 1e-8 AU (~1.5 km) from the camera, 6 km ortho view:
  /// pieces near the camera become view-sized, far pieces stay coarse, count stays small.
  #[test]
  fn refinement_makes_near_pieces_view_sized() {
    let seg = [
      [-0.5, 1e-8, 0.0],
      [-0.1666, 1e-8, 0.0],
      [0.1666, 1e-8, 0.0],
      [0.5, 1e-8, 0.0],
    ];
    let view = ViewScale {
      ortho_half_extent: 4e-8,
      tan_half_fov: 0.0,
    };
    let mut out = Vec::new();
    refine_near_camera(seg, &view, &mut out);
    assert!(out.len() < 120, "too many pieces: {}", out.len());
    let nearest = out
      .iter()
      .min_by(|a, b| {
        let da = a.iter().map(|p| len(*p)).fold(f64::MAX, f64::min);
        let db = b.iter().map(|p| len(*p)).fold(f64::MAX, f64::min);
        da.partial_cmp(&db).unwrap()
      })
      .unwrap();
    let size = len([
      nearest[3][0] - nearest[0][0],
      nearest[3][1] - nearest[0][1],
      0.0,
    ]);
    assert!(
      size <= 4.0 * 4e-8 * 1.01,
      "near piece {size} AU is larger than 4 views"
    );
  }

  /// Camera-relative f64 subtraction keeps metre-level precision near 1.5 AU where f32 absolute
  /// coordinates would be off by ~18 km.
  #[test]
  fn camera_relative_points_keep_precision_far_from_origin() {
    use aethervk_oshal_rlib::math::vector::{vec3f64::DVec3, vec4::Quat};
    let cam = [1.5, 0.3, 0.01];
    let off = 3e-9; // ~450 m
    let p: P = [cam[0] + off, cam[1], cam[2]];
    let traj = TrajectoryComponent::from_f64(alloc::vec![p, p, p, p], [1.0; 4], 2.0, 0, 32);
    let rte = HighResTransformComponent {
      position: DVec3::from_components(-cam[0], -cam[1], -cam[2]),
      rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(Quat::identity()),
      scale: Vec3f32::one(),
    };
    let view = ViewScale {
      ortho_half_extent: 4e-8,
      tan_half_fov: 0.0,
    };
    let rel = camera_relative_control_points(&traj, &[], &rte, view);
    assert!(((rel[0][0] as f64) - off).abs() < 1e-14, "{:?}", rel[0]);
    assert_eq!(rel[0][1], 0.0);
  }
}
