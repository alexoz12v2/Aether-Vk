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

/// The segments of `traj` plus `extra` (4 points each) transformed relative to the camera, in the
/// layer units and f64, unrefined (extracted from `camera_relative_control_points`, same ops).
///
/// `rte` is the entity transform relative to the camera in the layer units (see
/// `Scene::compute_rte`): `p_rel = rte.position + rte.rotation · (rte.scale ⊙ p)`.
fn relative_segments(
  traj: &TrajectoryComponent,
  extra: &[P],
  rte: &HighResTransformComponent,
) -> Vec<[P; 4]> {
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

  src
    .chunks_exact(4)
    .chain(extra.chunks_exact(4))
    .map(|seg| {
      [
        to_rel(&seg[0]),
        to_rel(&seg[1]),
        to_rel(&seg[2]),
        to_rel(&seg[3]),
      ]
    })
    .collect()
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
  let segs = relative_segments(traj, extra, rte);
  let mut pieces = Vec::with_capacity(segs.len() + 8);
  for seg in segs {
    refine_near_camera(seg, &view, &mut pieces);
  }
  pieces
    .iter()
    .flat_map(|seg| seg.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 1.0]))
    .collect()
}

/// Radius of the ball around a micro frame origin inside which layer-0 track pieces are drawn
/// relative to that origin (composite precision plan §2.3): 1e-4 AU ≈ 15 000 km.
pub const R_ANCHOR_AU: f64 = 1e-4;
const ANCHOR_MAX_DEPTH: u32 = 24;
/// pieces whose hull radius is at most `r / 64` are classified by their midpoint
const ANCHOR_MIN_PIECE: f64 = 1.0 / 64.0;

fn sub(a: P, b: P) -> P {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Centroid and radius of a ball containing the control hull (hence the curve).
fn hull_ball(seg: &[P; 4]) -> (P, f64) {
  let g: P = core::array::from_fn(|c| 0.25 * (seg[0][c] + seg[1][c] + seg[2][c] + seg[3][c]));
  (g, seg.iter().map(|p| len(sub(*p, g))).fold(0.0, f64::max))
}

/// A piece of a segment from [`classify_pieces`]: its control points, its side of the ball and
/// its parameter range `[t0, t1]` in the original segment.
#[derive(Debug, Clone, Copy)]
pub struct AnchorPiece {
  pub seg: [P; 4],
  pub inside: bool,
  pub t0: f64,
  pub t1: f64,
}

/// Splits `seg` (de Casteljau, f64, at 0.5) until each piece's control hull is inside the ball
/// `(o, r)` (all four control points within `r`: the ball is convex) or outside it (the hull ball
/// at least `r` away). A piece that has shrunk to a hull radius `ρ ≤ r/64`, or reached depth 24,
/// is classified by its midpoint instead: "inside" then means every control point within
/// `r + 2ρ ≤ 1.03 r` of `o`, "outside" a hull at least `r − 2ρ ≥ 0.97 r` from `o` (the §2.4
/// fallback argument holds with R replaced by 0.97 R). Pieces decided before that stop meet the
/// strict bounds. The stop bounds the count where a track runs along the sphere (an orbit of
/// radius exactly R: < 400 pieces instead of up to 2²⁴). Pieces are appended in curve order.
pub fn classify_pieces(seg: [P; 4], o: P, r: f64, out: &mut Vec<AnchorPiece>) {
  fn go(seg: [P; 4], o: P, r: f64, t0: f64, t1: f64, depth: u32, out: &mut Vec<AnchorPiece>) {
    if seg.iter().all(|p| len(sub(*p, o)) <= r) {
      out.push(AnchorPiece { seg, inside: true, t0, t1 });
      return;
    }
    let (g, rho) = hull_ball(&seg);
    if len(sub(g, o)) - rho >= r {
      out.push(AnchorPiece { seg, inside: false, t0, t1 });
      return;
    }
    let (l, rr) = split(seg, 0.5);
    if depth >= ANCHOR_MAX_DEPTH || rho <= ANCHOR_MIN_PIECE * r {
      out.push(AnchorPiece {
        seg,
        inside: len(sub(l[3], o)) < r,
        t0,
        t1,
      });
      return;
    }
    let tm = 0.5 * (t0 + t1);
    go(l, o, r, t0, tm, depth + 1, out);
    go(rr, o, r, tm, t1, depth + 1, out);
  }
  go(seg, o, r, 0.0, 1.0, 0, out)
}

/// The micro frame origin relative to the macro camera, AU, from the micro layer's camera
/// position in the frame (`cam_in_frame`, frame units) and the layer's AU per frame unit
/// (`ReferenceFrameComponent::scale`, the composite's `layerUnitAu`): `o_m = −cam·k`, so
/// `|o_m| = |cam|·k` (= D_m·k) by construction.
pub fn anchor_origin(cam_in_frame: P, au_per_unit: f64) -> P {
  [
    -cam_in_frame[0] * au_per_unit,
    -cam_in_frame[1] * au_per_unit,
    -cam_in_frame[2] * au_per_unit,
  ]
}

/// Anchor of a layer-0 track for the GPU (`TrajectoryGpu`'s anchor fields, filled in step 3; the
/// caller computes `anchorClip = center_clip_f64(layer-0 camera, origin_rte)`).
#[derive(Debug, Clone, Copy)]
pub struct AnchorGpu {
  /// the frame origin relative to the macro camera, AU (`o_m`)
  pub origin_rte: P,
  /// micro layer index m
  pub layer: u32,
  /// `(ô, |o| − |o| = 0)` and `1/|o|` (`rte_distance_reference(o, |o|)`)
  pub dist: [f32; 4],
  pub inv_len: f32,
}

/// Composite precision plan §2.3: control points of a layer-0 track. The pieces inside the
/// [`R_ANCHOR_AU`] ball of the nearest candidate micro-frame origin (`(o, layer)`, camera-relative
/// AU) are anchored, `[cp − o, 0]`; the others stay camera-relative, `[cp, 1]`. Both kinds are
/// refined near the camera in camera-relative f64 first (a sub-piece's hull lies in its parent's,
/// so refining keeps a piece inside). All four points of an emitted segment share their w.
/// Nearest candidate: among those whose hull-ball bound is within R, the smallest
/// `|piece midpoint − o|` over its inside pieces. Without an inside piece the anchor is `None`
/// and the output is [`camera_relative_control_points`]'s.
pub fn control_points_with_anchor(
  traj: &TrajectoryComponent,
  extra: &[P],
  rte: &HighResTransformComponent,
  view: ViewScale,
  candidates: &[(P, u32)],
) -> (Vec<[f32; 4]>, Option<AnchorGpu>) {
  let segs = relative_segments(traj, extra, rte);
  let mut best: Option<(f64, P, u32, Vec<AnchorPiece>)> = None;
  for &(o, layer) in candidates {
    let bound = segs
      .iter()
      .map(|s| {
        let (g, rho) = hull_ball(s);
        (len(sub(g, o)) - rho).max(0.0)
      })
      .fold(f64::MAX, f64::min);
    if bound >= R_ANCHOR_AU {
      continue;
    }
    let mut pieces = Vec::new();
    for s in &segs {
      classify_pieces(*s, o, R_ANCHOR_AU, &mut pieces);
    }
    let d = pieces
      .iter()
      .filter(|p| p.inside)
      .map(|p| len(sub(split(p.seg, 0.5).0[3], o)))
      .fold(f64::MAX, f64::min);
    if d < R_ANCHOR_AU && best.as_ref().map_or(true, |b| d < b.0) {
      best = Some((d, o, layer, pieces));
    }
  }
  let Some((_, o, layer, pieces)) = best else {
    return (camera_relative_control_points(traj, extra, rte, view), None);
  };
  let mut out = Vec::with_capacity(pieces.len() * 4 + 16);
  let mut refined = Vec::new();
  for p in &pieces {
    refined.clear();
    refine_near_camera(p.seg, &view, &mut refined);
    for s in &refined {
      for q in s {
        out.push(if p.inside {
          [
            (q[0] - o[0]) as f32,
            (q[1] - o[1]) as f32,
            (q[2] - o[2]) as f32,
            0.0,
          ]
        } else {
          [q[0] as f32, q[1] as f32, q[2] as f32, 1.0]
        });
      }
    }
  }
  // w = l − l = 0 exactly: the anchored offset is relative to |o|
  let (dist, inv_len) = rte_distance_reference(o, len(o));
  (
    out,
    Some(AnchorGpu {
      origin_rte: o,
      layer,
      dist,
      inv_len,
    }),
  )
}

/// Below this `|c|` (layer units) the centre reference degenerates: `1/|c|` is sent as 0 and the
/// shader uses `|u|` (the error is at most `|c|`). Far above f32 overflow of `k·|u|²`, `k²·|u|²`.
const RTE_REFERENCE_MIN_LEN: f64 = 1e-12;

/// Host half of the precise global depth (`rteDistanceOffset` in `common.glsl`): from the RTE
/// centre `c` of an object (f64, layer units) and the layer's reference distance `ref_dist`
/// (0 on layer 0), returns `((ĉ, |c| − ref_dist), 1/|c|)` cast to f32 once. Every value is
/// either a unit vector, a small difference or a reciprocal, so no large magnitude reaches f32.
pub fn rte_distance_reference(c: P, ref_dist: f64) -> ([f32; 4], f32) {
  let l = len(c);
  if !(l > RTE_REFERENCE_MIN_LEN) {
    return ([0.0, 0.0, 0.0, (l - ref_dist) as f32], 0.0);
  }
  (
    [
      (c[0] / l) as f32,
      (c[1] / l) as f32,
      (c[2] / l) as f32,
      (l - ref_dist) as f32,
    ],
    (1.0 / l) as f32,
  )
}

/// f32 mirror of `rteDistanceOffset` (`common.glsl`), same operation order and branches:
/// `|c + u| − D_ref` from the reference of [`rte_distance_reference`] and the small local offset
/// `u`. `|c+u| − |c| = n / (1 + sqrt(1 + k n))`, `n = 2ĉ·u + k|u|²`, `k = 1/|c|` (exact identity);
/// below `|c+u| = |c|/2` (`1 + k n < 1/4`, where that form cancels) the direct `|c + u| − |c|`
/// with `c = ĉ/k` rebuilt in f32.
pub fn rte_distance_offset_f32(center_dist: [f32; 4], inv_center_len: f32, u: [f32; 3]) -> f32 {
  let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if inv_center_len == 0.0 {
    return center_dist[3] + dot(u, u).sqrt();
  }
  let c_hat = [center_dist[0], center_dist[1], center_dist[2]];
  let n = 2.0 * dot(c_hat, u) + dot(u, u) * inv_center_len;
  let s2 = 1.0 + inv_center_len * n;
  if s2 < 0.25 {
    let c_len = 1.0 / inv_center_len;
    let p = [
      c_hat[0] * c_len + u[0],
      c_hat[1] * c_len + u[1],
      c_hat[2] * c_len + u[2],
    ];
    return center_dist[3] + (dot(p, p).sqrt() - c_len);
  }
  center_dist[3] + n / (1.0 + s2.sqrt())
}

#[cfg(test)]
mod tests {
  use super::*;
  extern crate std;

  /// Today's `camera_relative_control_points` body, kept verbatim: the refactor through
  /// `relative_segments` must reproduce it bit for bit.
  fn camera_relative_control_points_reference(
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

  /// Deterministic xorshift64* in [−1, 1).
  struct Rng(u64);
  impl Rng {
    fn next(&mut self) -> f64 {
      self.0 ^= self.0 >> 12;
      self.0 ^= self.0 << 25;
      self.0 ^= self.0 >> 27;
      let v = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
      (v >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
    fn vec(&mut self, r: f64) -> P {
      loop {
        let p = [self.next(), self.next(), self.next()];
        if len(p) <= 1.0 {
          return [p[0] * r, p[1] * r, p[2] * r];
        }
      }
    }
  }

  /// `|c + u| − ref` in f64 vs the shader formula in f32 (`rte_distance_offset_f32`).
  fn offset_error(c: P, ref_dist: f64, u: P) -> (f64, f64) {
    let (cd, k) = rte_distance_reference(c, ref_dist);
    let got = rte_distance_offset_f32(cd, k, [u[0] as f32, u[1] as f32, u[2] as f32]) as f64;
    // the truth uses the f32-rounded u: the shader's u is an f32 vertex offset
    let uf = [u[0] as f32 as f64, u[1] as f32 as f64, u[2] as f32 as f64];
    let truth = len([c[0] + uf[0], c[1] + uf[1], c[2] + uf[2]]) - ref_dist;
    (got, (got - truth).abs())
  }

  /// The plan's budget (§2.4): a nucleus at the frame origin seen from 0.1, 1 and 5 AU (km units,
  /// reference = the distance to the frame origin), surface points |u| ≤ 10 km, and mesh centres
  /// up to 20 km off the origin: the f32 offset is within 1 m of the f64 truth. Before, the global
  /// depth was `length(inWorldPos)` in f32, ½ulp(D) = 8.9 km at 1 AU.
  #[test]
  fn depth_offset_far_field_within_a_metre() {
    const AU_KM: f64 = 149_597_870.7;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = [0.0f64; 3];
    for (i, d_au) in [0.1, 1.0, 5.0].into_iter().enumerate() {
      for _ in 0..20_000 {
        let dir = rng.vec(1.0);
        let l = len(dir);
        if l < 1e-3 {
          continue;
        }
        let o = [dir[0] / l * d_au * AU_KM, dir[1] / l * d_au * AU_KM, dir[2] / l * d_au * AU_KM];
        let ref_dist = len(o);
        let w = rng.vec(20.0); // mesh centre offset from the frame origin, km
        let c = [o[0] + w[0], o[1] + w[1], o[2] + w[2]];
        let (_, err) = offset_error(c, ref_dist, rng.vec(10.0));
        worst[i] = worst[i].max(err);
      }
    }
    std::println!(
      "[depth_offset] far field max |err| (km): 0.1 AU {:.3e}, 1 AU {:.3e}, 5 AU {:.3e}",
      worst[0], worst[1], worst[2]
    );
    for (d, w) in [0.1, 1.0, 5.0].iter().zip(worst) {
      assert!(w < 1e-3, "{d} AU: offset error {w} km >= 1 m");
    }
  }

  /// The near field (inside poses: camera metres to km from the surface, |u| ≥ |c|, u ≈ −c):
  /// the exact identity has no truncation term and the direct branch covers |c+u| < |c|/2, so the
  /// error stays a few ulp of |c| + |u|. The plan's second-order expansion would be off by
  /// O(|u|³/|c|²) here (km at |u| ≈ |c|).
  #[test]
  fn depth_offset_near_field_is_exact_to_f32() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut worst_rel = 0.0f64;
    for &c_len in &[1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 100.0] {
      for &u_max in &[1e-3, 0.1, 2.0, 5.0] {
        for _ in 0..5_000 {
          let dir = rng.vec(1.0);
          let l = len(dir);
          if l < 1e-3 {
            continue;
          }
          let c = [dir[0] / l * c_len, dir[1] / l * c_len, dir[2] / l * c_len];
          let u = rng.vec(u_max);
          for ref_dist in [0.0, c_len] {
            let (_, err) = offset_error(c, ref_dist, u);
            let scale = c_len + len(u);
            worst_rel = worst_rel.max(err / scale);
            assert!(
              err <= 1e-6 * scale,
              "|c| {c_len} km, u {u:?}, ref {ref_dist}: error {err} km"
            );
          }
        }
        // u = −c (the camera on the surface point): distance 0
        let c = [c_len, 0.0, 0.0];
        let (got, err) = offset_error(c, c_len, [-c_len, 0.0, 0.0]);
        assert!(err <= 1e-6 * c_len, "u = −c: {got} vs {}", -c_len);
      }
    }
    std::println!("[depth_offset] near field max |err| / (|c|+|u|): {worst_rel:.3e}");
  }

  /// Layer 0 (reference 0, AU): the formula returns the absolute distance with f32 relative
  /// precision, as `length(inWorldPos)` did; the degenerate `|c| = 0` falls back to `|u|`.
  #[test]
  fn depth_offset_absolute_and_degenerate() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    for _ in 0..20_000 {
      let c = rng.vec(40.0);
      let u = rng.vec(1e-4);
      let (got, err) = offset_error(c, 0.0, u);
      assert!(err <= 4.0 * f32::EPSILON as f64 * got.abs().max(1e-30), "{c:?} {u:?}: {err}");
    }
    let (cd, k) = rte_distance_reference([0.0; 3], 2.5);
    assert_eq!((cd, k), ([0.0, 0.0, 0.0, -2.5], 0.0));
    let got = rte_distance_offset_f32(cd, k, [3.0, 4.0, 0.0]);
    assert_eq!(got, 2.5);
    // the identity ObjectData (all zero) yields |u|, the old length(model · v)
    assert_eq!(rte_distance_offset_f32([0.0; 4], 0.0, [0.0, 3.0, 4.0]), 5.0);
  }

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

  // ── step 2: anchoring (composite precision plan §2.3) ──

  fn sub3(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
  }
  fn add3(a: P, b: P) -> P {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
  }
  fn bits(v: &[[f32; 4]]) -> Vec<[u32; 4]> {
    v.iter().map(|p| p.map(f32::to_bits)).collect()
  }
  fn identity_rte() -> HighResTransformComponent {
    use aethervk_oshal_rlib::math::vector::{vec3f64::DVec3, vec4::Quat};
    HighResTransformComponent {
      position: DVec3::from_components(0.0, 0.0, 0.0),
      rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(Quat::identity()),
      scale: Vec3f32::one(),
    }
  }
  fn view_1au() -> ViewScale {
    // a telescope field: 32 km at 1 AU
    ViewScale {
      ortho_half_extent: 0.0,
      tan_half_fov: 1.07e-7,
    }
  }
  fn line(a: P, b: P) -> Vec<P> {
    (0..4).map(|k| lerp(a, b, k as f64 / 3.0)).collect()
  }
  fn traj_of(points: Vec<P>) -> TrajectoryComponent {
    TrajectoryComponent::from_f64(points, [1.0; 4], 2.0, 0, 32)
  }

  /// Checks the partition of one segment and returns its piece count.
  fn check_partition(seg: [P; 4], o: P, r: f64) -> usize {
    let mut out = Vec::new();
    classify_pieces(seg, o, r, &mut out);
    assert!(!out.is_empty());
    assert_eq!(out[0].t0, 0.0);
    assert_eq!(out.last().unwrap().t1, 1.0);
    assert_eq!(out[0].seg[0], seg[0]);
    assert_eq!(out.last().unwrap().seg[3], seg[3]);
    // rounding scale of the f64 de Casteljau: the segment size and its coordinate magnitude
    let size = (0..4).map(|i| len(sub3(seg[i], seg[0]))).fold(0.0, f64::max) + len(seg[0]);
    for w in out.windows(2) {
      assert_eq!(w[0].t1, w[1].t0, "ranges must tile [0, 1]");
      assert_eq!(w[0].seg[3], w[1].seg[0], "pieces must share their end points");
    }
    for p in &out {
      for k in 0..=10 {
        let s = k as f64 / 10.0;
        let a = eval(&p.seg, s);
        let b = eval(&seg, p.t0 + (p.t1 - p.t0) * s);
        assert!(len(sub3(a, b)) <= 1e-12 * size, "piece off the curve by {}", len(sub3(a, b)));
      }
      let (g, rho) = hull_ball(&p.seg);
      let max_d = p.seg.iter().map(|q| len(sub3(*q, o))).fold(0.0, f64::max);
      if p.inside {
        assert!(max_d <= r + 2.0 * rho, "inside piece reaches {max_d} (r {r}, rho {rho})");
      } else {
        assert!(
          len(sub3(g, o)) - rho >= r - 2.0 * rho,
          "outside piece within {} (r {r}, rho {rho})",
          len(sub3(g, o)) - rho
        );
      }
      // decided before the stop rule: strict bounds
      if rho > ANCHOR_MIN_PIECE * r {
        if p.inside {
          assert!(max_d <= r);
        } else {
          assert!(len(sub3(g, o)) - rho >= r);
        }
      }
    }
    out.len()
  }

  #[test]
  fn classify_pieces_partitions_the_curve() {
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    let r = 1e-4;
    let mut max_count = 0;
    for i in 0..2000 {
      let o = {
        let d = rng.vec(1.0);
        let l = len(d).max(1e-3);
        [1.0 + d[0] / l * 0.1, d[1] / l * 0.1, d[2] / l * 0.1]
      };
      let seg: [P; 4] = if i % 4 == 0 {
        // 1 AU long, passing within 2r of o
        let dir = rng.vec(1.0);
        let l = len(dir).max(1e-3);
        let dir = [dir[0] / l, dir[1] / l, dir[2] / l];
        let c = add3(o, rng.vec(2.0 * r));
        let a = sub3(c, [dir[0] * 0.5, dir[1] * 0.5, dir[2] * 0.5]);
        let b = add3(c, [dir[0] * 0.5, dir[1] * 0.5, dir[2] * 0.5]);
        [
          a,
          add3(lerp(a, b, 1.0 / 3.0), rng.vec(0.01)),
          add3(lerp(a, b, 2.0 / 3.0), rng.vec(0.01)),
          b,
        ]
      } else {
        [
          add3(o, rng.vec(3.0 * r)),
          add3(o, rng.vec(4.0 * r)),
          add3(o, rng.vec(4.0 * r)),
          add3(o, rng.vec(3.0 * r)),
        ]
      };
      let n = check_partition(seg, o, r);
      assert!(n < 2000, "{n} pieces");
      max_count = max_count.max(n);
    }
    std::println!("[classify_pieces] 2000 random segments, max pieces per segment {max_count}");
  }

  /// A circle of radius exactly r on the ball's surface (four cubic quarter arcs): the stop rule
  /// bounds the piece count.
  #[test]
  fn classify_pieces_bounded_on_the_sphere() {
    let r = 1e-4;
    let o = [1.0, 0.0, 0.0];
    let k = 0.552_284_749_830_793_4 * r; // quarter-arc handle length
    let pt = |x: f64, y: f64| [o[0] + x, o[1] + y, o[2]];
    let arcs = [
      [pt(r, 0.0), pt(r, k), pt(k, r), pt(0.0, r)],
      [pt(0.0, r), pt(-k, r), pt(-r, k), pt(-r, 0.0)],
      [pt(-r, 0.0), pt(-r, -k), pt(-k, -r), pt(0.0, -r)],
      [pt(0.0, -r), pt(k, -r), pt(r, -k), pt(r, 0.0)],
    ];
    let total: usize = arcs.iter().map(|a| check_partition(*a, o, r)).sum();
    std::println!("[classify_pieces] circle on the sphere: {total} pieces");
    assert!(total < 1024, "{total} pieces");
  }

  /// No anchor (no candidate, a far candidate, a track outside the ball): the output is
  /// camera_relative_control_points' bit for bit, and that refactored function reproduces
  /// today's body bit for bit.
  #[test]
  fn anchor_none_is_bit_identical() {
    let mut rng = Rng(0x5851_F42D_4C95_7F2D);
    let rte = identity_rte();
    for case in 0..50 {
      let a = add3([1.0, 0.0, 0.0], rng.vec(0.5));
      let b = add3([1.0, 0.0, 0.0], rng.vec(0.5));
      let mut pts = line(a, b);
      pts.extend(line(b, add3(b, rng.vec(1e-6))));
      let traj = traj_of(pts);
      let extra = line(b, add3(b, [1e-7, 0.0, 0.0]));
      let view = if case % 2 == 0 { view_1au() } else { ViewScale { ortho_half_extent: 4e-8, tan_half_fov: 0.0 } };
      let reference = camera_relative_control_points_reference(&traj, &extra, &rte, view);
      let today = camera_relative_control_points(&traj, &extra, &rte, view);
      assert_eq!(bits(&today), bits(&reference), "refactor changed the output (case {case})");
      for cands in [alloc::vec![], alloc::vec![([5.0, 5.0, 5.0], 1u32)]] {
        let (got, anchor) = control_points_with_anchor(&traj, &extra, &rte, view, &cands);
        assert!(anchor.is_none());
        assert_eq!(bits(&got), bits(&reference));
      }
    }
    // a track passing 2 R from the candidate: not anchored
    let o = [1.0, 0.0, 0.0];
    let traj = traj_of(line([1.0 - 0.5, 2.0 * R_ANCHOR_AU, 0.0], [1.5, 2.0 * R_ANCHOR_AU, 0.0]));
    let (got, anchor) = control_points_with_anchor(&traj, &[], &rte, view_1au(), &[(o, 1)]);
    assert!(anchor.is_none());
    assert_eq!(bits(&got), bits(&camera_relative_control_points(&traj, &[], &rte, view_1au())));
  }

  /// Reconstruction of an anchored output: the points of each 4-point segment share w; w = 0
  /// points are cp + o, w = 1 points are cp.
  fn reconstruct(out: &[[f32; 4]], o: P) -> Vec<([P; 4], bool)> {
    out
      .chunks_exact(4)
      .map(|s| {
        let w = s[0][3];
        assert!(s.iter().all(|p| p[3] == w), "mixed w in one segment: {s:?}");
        let anchored = w == 0.0;
        let seg: [P; 4] = core::array::from_fn(|i| {
          let p = [s[i][0] as f64, s[i][1] as f64, s[i][2] as f64];
          if anchored { add3(p, o) } else { p }
        });
        (seg, anchored)
      })
      .collect()
  }

  /// Distance from `q` to the straight track a→b (f64).
  fn dist_to_line(q: P, a: P, b: P) -> f64 {
    let ab = sub3(b, a);
    let t = ((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1] + (q[2] - a[2]) * ab[2])
      / (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2]);
    let p = lerp(a, b, t);
    len(sub3(q, p))
  }

  #[test]
  fn anchored_pieces_reproduce_the_track() {
    const KM: f64 = 1.0 / 149_597_870.7;
    let rte = identity_rte();
    for d_au in [1.0, 5.0] {
      let o = [0.3 * d_au, -(1.0 - 0.09f64).sqrt() * d_au, 0.0];
      let lateral = [0.0, 0.0, 3.0 * KM];
      let a = add3(add3(o, lateral), [-0.5, 0.2, 0.0]);
      let b = add3(add3(o, lateral), [0.5, -0.2, 0.0]);
      let traj = traj_of(line(a, b));
      let (out, anchor) = control_points_with_anchor(&traj, &[], &rte, view_1au(), &[(o, 3)]);
      let anchor = anchor.expect("track through the ball must be anchored");
      assert_eq!(anchor.layer, 3);
      assert_eq!(anchor.origin_rte, o);
      assert_eq!(anchor.dist[3], 0.0);
      assert_eq!(anchor.inv_len, (1.0 / len(o)) as f32);
      let (mut worst_anchored, mut worst_plain, mut worst_near, mut worst_near_f32) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64);
      let mut n_anchored = 0;
      for (seg, anchored) in reconstruct(&out, o) {
        // anchored pieces are ~R long: sample densely (~4 km apart) to reach the nucleus region
        let samples = if anchored { 4096 } else { 8 };
        for k in 0..=samples {
          let q = eval(&seg, k as f64 / samples as f64);
          let e = dist_to_line(q, a, b);
          if anchored {
            worst_anchored = worst_anchored.max(e);
            if len(sub3(q, o)) <= 10.0 * KM {
              worst_near = worst_near.max(e);
              // the same point cast camera-relative to f32, as before the anchoring
              let qf = [q[0] as f32 as f64, q[1] as f32 as f64, q[2] as f32 as f64];
              worst_near_f32 = worst_near_f32.max(len(sub3(qf, q)));
            }
          } else {
            worst_plain = worst_plain.max(e);
          }
        }
        n_anchored += anchored as usize;
      }
      std::println!(
        "[anchor] D {d_au} AU: {} segments ({n_anchored} anchored); max off-track anchored {:.3} m, \
         near (<10 km) {:.3} m vs camera-relative f32 cast {:.1} km; plain {:.1} km",
        out.len() / 4,
        worst_anchored / KM * 1e3,
        worst_near / KM * 1e3,
        worst_near_f32 / KM,
        worst_plain / KM
      );
      assert!(n_anchored > 0);
      assert!(worst_anchored < 1e-11, "anchored points {worst_anchored} AU off the track");
      assert!(worst_plain < 2.0 * 1.2e-7 * d_au, "plain points {worst_plain} AU off the track");
      assert!(worst_near_f32 > 0.0, "no anchored sample within 10 km of the origin");
      // ½ulp(1e-4 AU) = 0.54 m per axis on the anchored control points (plan §2.3)
      assert!(worst_near < 1e-11, "near points {} m off", worst_near / KM * 1e3);
      assert!(worst_near_f32 > 1.0 * KM, "the f32 cast should have been off by > 1 km");
    }
  }

  #[test]
  fn nearest_candidate_wins() {
    const KM: f64 = 1.0 / 149_597_870.7;
    let rte = identity_rte();
    let o_far = [1.0, 5000.0 * KM, 0.0];
    let o_near = [1.0, 100.0 * KM, 0.0];
    let traj = traj_of(line([0.5, 0.0, 0.0], [1.5, 0.0, 0.0]));
    for cands in [[(o_far, 1u32), (o_near, 2u32)], [(o_near, 2u32), (o_far, 1u32)]] {
      let (_, anchor) = control_points_with_anchor(&traj, &[], &rte, view_1au(), &cands);
      let anchor = anchor.unwrap();
      assert_eq!(anchor.layer, 2);
      assert_eq!(anchor.origin_rte, o_near);
    }
  }

  /// The camera inside the ball (|o| = 150 km): the output stays valid and the anchored
  /// distance offset (|u| > |o|) matches the f64 truth.
  #[test]
  fn camera_inside_the_ball() {
    const KM: f64 = 1.0 / 149_597_870.7;
    let rte = identity_rte();
    let o = [0.0, -150.0 * KM, 0.0];
    let a = [-2000.0 * KM, -100.0 * KM, 1.0 * KM];
    let b = [3000.0 * KM, -400.0 * KM, 1.0 * KM];
    let traj = traj_of(line(a, b));
    let view = ViewScale { ortho_half_extent: 0.0, tan_half_fov: 0.5 };
    let (out, anchor) = control_points_with_anchor(&traj, &[], &rte, view, &[(o, 1)]);
    let anchor = anchor.expect("anchored");
    assert!(out.iter().all(|p| p.iter().all(|v| v.is_finite())));
    let mut worst = 0.0f64;
    let mut worst_off = 0.0f64;
    for (seg, anchored) in reconstruct(&out, o) {
      assert!(anchored, "every piece is inside the ball here");
      for k in 0..=8 {
        let q = eval(&seg, k as f64 / 8.0);
        worst = worst.max(dist_to_line(q, a, b));
        // the shader's offset: |o + u| − |o| from the anchor reference and u = q − o (f32)
        let u = sub3(q, o);
        let uf = [u[0] as f32, u[1] as f32, u[2] as f32];
        let got = rte_distance_offset_f32(anchor.dist, anchor.inv_len, uf) as f64;
        let ud = [uf[0] as f64, uf[1] as f64, uf[2] as f64];
        let truth = len(add3(o, ud)) - len(o);
        worst_off = worst_off.max((got - truth).abs());
      }
    }
    std::println!(
      "[anchor] camera inside the ball: max off-track {:.3} m, max offset error {:.3} m",
      worst / KM * 1e3,
      worst_off / KM * 1e3
    );
    assert!(worst < 1e-3 * KM, "{} m off the track", worst / KM * 1e3);
    assert!(worst_off < 1e-3 * KM, "offset error {} m", worst_off / KM * 1e3);
  }

  /// o_m from the micro camera: |o_m| = D_m·scale and o_m = N − C (the frame origin relative to
  /// the camera) for cameras 0.1, 1 and 5 AU from the frame, scale = (1/AU_TO_KM) as f32.
  #[test]
  fn anchor_origin_matches_the_layer_reference() {
    let scale = (1.0 / 149_597_870.7f64) as f32 as f64;
    for d_au in [0.1, 1.0, 5.0] {
      let n = [1.0, 1.0 - d_au, 0.25 * d_au];
      let c = [1.0, 1.0, 0.0];
      // as get_relative_transform_f64: (C − N) / frame scale
      let cam = [(c[0] - n[0]) / scale, (c[1] - n[1]) / scale, (c[2] - n[2]) / scale];
      let o = anchor_origin(cam, scale);
      let d_m = len(cam);
      assert!((len(o) - d_m * scale).abs() <= 1e-15 * len(o), "{d_au}: |o| {} vs {}", len(o), d_m * scale);
      let truth = sub3(n, c);
      assert!(len(sub3(o, truth)) <= 1e-15, "{d_au}: o {o:?} vs {truth:?}");
    }
  }

  /// The vertex shader's reconstruction of an anchored piece (f64 mirror): with L the linear part
  /// of a view-projection, Σ b_i L u_i + VP·(o, 1) = VP·(C(t), 1) and Σ db_i L u_i = L·C'(t),
  /// because Σ b_i = 1 and Σ db_i = 0.
  #[test]
  fn shader_reconstruction_identity() {
    const KM: f64 = 1.0 / 149_597_870.7;
    // an arbitrary projective matrix (rows), a perspective-like 4th row
    let vp = [
      [3.0e6, 0.2, -0.1, 0.0],
      [0.1, -2.0e6, 0.3, 0.0],
      [0.0, 1e-9, 0.0, 1e-9],
      [0.01, -0.999, 0.02, 0.0],
    ];
    let mul = |p: P, w: f64| -> [f64; 4] {
      core::array::from_fn(|r| vp[r][0] * p[0] + vp[r][1] * p[1] + vp[r][2] * p[2] + vp[r][3] * w)
    };
    let o = [0.3, -0.95, 0.0];
    let rte = identity_rte();
    let traj = traj_of(line(add3(o, [-0.5, 0.2, 3.0 * KM]), add3(o, [0.5, -0.2, 3.0 * KM])));
    let (out, anchor) = control_points_with_anchor(&traj, &[], &rte, view_1au(), &[(o, 1)]);
    let anchor = anchor.unwrap();
    let a_clip = mul(anchor.origin_rte, 1.0);
    let mut worst = 0.0f64;
    let mut worst_t = 0.0f64;
    for s in out.chunks_exact(4).filter(|s| s[0][3] == 0.0) {
      let u: [P; 4] = core::array::from_fn(|i| [s[i][0] as f64, s[i][1] as f64, s[i][2] as f64]);
      for k in 0..=16 {
        let t = k as f64 / 16.0;
        let om = 1.0 - t;
        let b = [om * om * om, 3.0 * om * om * t, 3.0 * om * t * t, t * t * t];
        let db = [-3.0 * om * om, 3.0 * om * (1.0 - 3.0 * t), 3.0 * t * (2.0 - 3.0 * t), 3.0 * t * t];
        let mut clip = a_clip;
        let mut dclip = [0.0; 4];
        for i in 0..4 {
          let v = mul(u[i], 0.0);
          for r in 0..4 {
            clip[r] += b[i] * v[r];
            dclip[r] += db[i] * v[r];
          }
        }
        let c = add3(o, eval(&u, t));
        let truth = mul(c, 1.0);
        let dc: P = core::array::from_fn(|j| (0..4).map(|i| db[i] * u[i][j]).sum());
        let dtruth = mul(dc, 0.0);
        let scale = truth.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let dscale = dtruth.iter().map(|v| v.abs()).fold(1e-300, f64::max);
        for r in 0..4 {
          worst = worst.max((clip[r] - truth[r]).abs() / scale);
          worst_t = worst_t.max((dclip[r] - dtruth[r]).abs() / dscale);
        }
      }
    }
    std::println!("[anchor] clip reconstruction rel err {worst:.2e}, tangent {worst_t:.2e}");
    assert!(worst <= 1e-12 && worst_t <= 1e-12);
  }
}
