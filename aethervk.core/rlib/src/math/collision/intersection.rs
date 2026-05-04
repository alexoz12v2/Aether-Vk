//! Intersection tests
//! Contains logic for checking intersections between different shapes

use crate::{
  math::collision::bounds::{AABB, BS, OBB},
  simulation::comet::Triangle,
};
use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;
use aethervk_oshal_rlib::math::{
  FloatLike, MulAddIdentity,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix, Matrix3, Matrix4},
  vector::{Vector, Vector2, Vector3, vec3::Vec3f32},
};
use itertools::Itertools;

/// A line segment / ray defined by an origin and a direction vector and length
#[derive(Debug, Clone, Copy)]
pub struct Ray<V>
where
  V: Vector3,
  V::Scalar: FloatLike + FloatOps + FloatBits,
{
  pub origin: V,
  pub direction: V,
  pub length: V::Scalar, // also known as t_{max}
}

// ----------------------------------------------------------------------------
// Sphere vs Sphere
// ----------------------------------------------------------------------------
pub fn intersect_sphere_sphere<V>(a: &BS<V::Scalar>, b: &BS<V::Scalar>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32> + core::ops::Mul<V, Output = V>,
{
  let d: V = a.center::<V>() - b.center();
  let r_sum: V::Scalar = a.radius() + b.radius();
  d.length_squared() <= r_sum * r_sum
}

// ----------------------------------------------------------------------------
// Box vs Box
// ----------------------------------------------------------------------------
// TODO: AABB vs AABB, AABB vs OBB, OBB vs OBB (15 combinations)

pub fn intersect_aabb_aabb<V>(a: &AABB<V::Scalar>, b: &AABB<V::Scalar>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits,
{
  let a_min: V = a.min();
  let a_max: V = a.max();
  let b_min: V = b.min();
  let b_max: V = b.max();

  if a_max.x() < b_min.x() || a_min.x() > b_max.x() {
    return false;
  }
  if a_max.y() < b_min.y() || a_min.y() > b_max.y() {
    return false;
  }
  if a_max.z() < b_min.z() || a_min.z() > b_max.z() {
    return false;
  }

  true
}

/// Checks if two Oriented Bounding Boxes overlap. They should refer to a common frame of coordinates
/// (centers and axes of both)
pub fn intersect_obb_obb<S, V, M>(a: &OBB<S>, b: &OBB<S>) -> bool
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  let pos_a: V = a.translation();
  let pos_b: V = b.translation();
  let ext_a: V = a.half_extent();
  let ext_b: V = b.half_extent();
  let axes_a: [V; 3] = a.axes();
  let axes_b: [V; 3] = b.axes();

  // 1. Vector from center A to center B in world space
  let t_world = pos_b - pos_a;

  // 2. Compute translation vectyor T in A's local coordinate space
  let tx = t_world.dot(axes_a[0]);
  let ty = t_world.dot(axes_a[1]);
  let tz = t_world.dot(axes_a[2]);

  // 3. Compute rotation matrix expressing B in A's local coordinate space
  // r_{ij} = Axis of i of A dot Axis j of B
  let r00 = axes_a[0].dot(axes_b[0]);
  let r01 = axes_a[0].dot(axes_b[1]);
  let r02 = axes_a[0].dot(axes_b[2]);
  let r10 = axes_a[1].dot(axes_b[0]);
  let r11 = axes_a[1].dot(axes_b[1]);
  let r12 = axes_a[1].dot(axes_b[2]);
  let r20 = axes_a[2].dot(axes_b[0]);
  let r21 = axes_a[2].dot(axes_b[1]);
  let r22 = axes_a[2].dot(axes_b[2]);

  // 4. Compute the absolute rotation matrix with a small epsilon
  // The epsilon prevents floating point inaccuracies from returning a zero
  // vector when the two axes are perfectly parallel during the cross product test
  let eps = S::from_f32(1e-6);
  let abs_r00 = r00.abs() + eps;
  let abs_r01 = r01.abs() + eps;
  let abs_r02 = r02.abs() + eps;
  let abs_r10 = r10.abs() + eps;
  let abs_r11 = r11.abs() + eps;
  let abs_r12 = r12.abs() + eps;
  let abs_r20 = r20.abs() + eps;
  let abs_r21 = r21.abs() + eps;
  let abs_r22 = r22.abs() + eps;

  let (ax, ay, az) = (ext_a.x(), ext_a.y(), ext_a.z());
  let (bx, by, bz) = (ext_b.x(), ext_b.y(), ext_b.z());

  // --- 1. Test the 3 face normals of A ---
  if tx.abs() > ax + bx * abs_r00 + by * abs_r01 + bz * abs_r02 {
    return false;
  }
  if ty.abs() > ay + bx * abs_r10 + by * abs_r11 + bz * abs_r12 {
    return false;
  }
  if tz.abs() > az + bx * abs_r20 + by * abs_r21 + bz * abs_r22 {
    return false;
  }

  // --- 2. Test the 3 face normals of B ---
  if (tx * r00 + ty * r10 + tz * r20).abs() > bx + ax * abs_r00 + ay * abs_r10 + az * abs_r20 {
    return false;
  }
  if (tx * r01 + ty * r11 + tz * r21).abs() > by + ax * abs_r01 + ay * abs_r11 + az * abs_r21 {
    return false;
  }
  if (tx * r02 + ty * r12 + tz * r22).abs() > bz + ax * abs_r02 + ay * abs_r12 + az * abs_r22 {
    return false;
  }

  // --- 3. Test the 9 pairwise edge cross-products ---

  // A.x x B.x
  if (tz * r10 - ty * r20).abs() > ay * abs_r20 + az * abs_r10 + by * abs_r02 + bz * abs_r01 {
    return false;
  }
  // A.x x B.y
  if (tz * r11 - ty * r21).abs() > ay * abs_r21 + az * abs_r11 + bx * abs_r02 + bz * abs_r00 {
    return false;
  }
  // A.x x B.z
  if (tz * r12 - ty * r22).abs() > ay * abs_r22 + az * abs_r12 + bx * abs_r01 + by * abs_r00 {
    return false;
  }

  // A.y x B.x
  if (tx * r20 - tz * r00).abs() > ax * abs_r20 + az * abs_r00 + by * abs_r12 + bz * abs_r11 {
    return false;
  }
  // A.y x B.y
  if (tx * r21 - tz * r01).abs() > ax * abs_r21 + az * abs_r01 + bx * abs_r12 + bz * abs_r10 {
    return false;
  }
  // A.y x B.z
  if (tx * r22 - tz * r02).abs() > ax * abs_r22 + az * abs_r02 + bx * abs_r11 + by * abs_r10 {
    return false;
  }

  // A.z x B.x
  if (ty * r00 - tx * r10).abs() > ax * abs_r10 + ay * abs_r00 + by * abs_r22 + bz * abs_r21 {
    return false;
  }
  // A.z x B.y
  if (ty * r01 - tx * r11).abs() > ax * abs_r11 + ay * abs_r01 + bx * abs_r22 + bz * abs_r20 {
    return false;
  }
  // A.z x B.z
  if (ty * r02 - tx * r12).abs() > ax * abs_r12 + ay * abs_r02 + bx * abs_r21 + by * abs_r20 {
    return false;
  }

  // 4. if no separating axis is found, the OBBs are intersecting
  true
}

// ----------------------------------------------------------------------------
// Triangle vs Triangle
// ----------------------------------------------------------------------------
pub fn intersect_triangle_triangle<S, Vec3, Vec2>(t_a: &Triangle, t_b: &Triangle) -> bool
where
  Vec3: Vector3<Scalar = S> + From<Vec3f32>,
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits + core::ops::Mul<Vec3, Output = Vec3>,
{
  let eps = S::from_f32(1e-6);

  let a: [Vec3; 3] = t_a.vertices.iter().map(|v| (*v).into()).collect_array().unwrap();
  let b: [Vec3; 3] = t_b.vertices.iter().map(|v| (*v).into()).collect_array().unwrap();

  // 1. Compute plane A and signed distances of B's vertices
  let n_a = (a[1] - a[0]).cross(a[2] - a[1]); // TODO should this be normalized?
  let d_b0 = n_a.dot(b[0] - a[0]);
  let d_b1 = n_a.dot(b[1] - a[0]);
  let d_b2 = n_a.dot(b[2] - a[0]);

  // check if B is entirely on one side of Plane A (quick rejection)
  if (d_b0 > eps && d_b1 > eps && d_b2 > eps) || (d_b0 < -eps && d_b1 < -eps && d_b1 < -eps) {
    return false;
  }

  // 2. Compute Plane B and signed distances of A's vertices
  // (Symmetric test required: A could be entirely on one of B)
  let n_b = (b[1] - a[0]).cross(b[2] - b[1]);
  let d_a0 = n_b.dot(a[0] - b[0]);
  let d_a1 = n_b.dot(a[1] - b[0]);
  let d_a2 = n_b.dot(a[2] - b[0]);

  if (d_a0 > eps && d_a1 > eps && d_a2 > eps) || (d_a0 < -eps && d_a1 < -eps && d_a2 < -eps) {
    return false;
  }

  // 3. Coplanarity Checks (Book "Guide To Simulations of RigidBody and Particles, Springer" Cases 1,2,3)
  let b_coplanar = d_b0.abs() <= eps && d_b1.abs() <= eps && d_b2.abs() <= eps;
  if b_coplanar {
    // Case 3: Triangles are completely coplanar.
    // Fall back to 2D edge-edge and point-in-triangle test
    return coplanar_tri_tri_test::<S, Vec3, Vec2>(&a, &b, n_a);
  }

  // 4. General Intersection (Case 4)
  // Planes intersect in a line. We need to find the segment
  // of Triangle B that pierces Plane A, and check if it hits Triangle A

  // Find the two points of Triangle B that intersect Plane A
  let (p1, p2) = compute_intersection_segment(&b, d_b0, d_b1, d_b2);

  // Test if the segment (p1, p2) intersects Triangle A
  // TODO: Alternative: compute intersection segment for other triangle and see if two segments overlap
  segment_intersects_triangle::<S, Vec3, Vec2>(p1, p2, &a, n_a)
}

/// in triangle-triangle intersection, When all points evaluate to zero in the plane-distance check
/// the two triangles are perfectly coplanalr. We flatten both triangles in 2D, check all
/// 9 edge combinations and do point-in-triangle tests to catch where only one triangle completely
/// encompasses the other
fn coplanar_tri_tri_test<S, Vec3, Vec2>(a: &[Vec3; 3], b: &[Vec3; 3], n: Vec3) -> bool
where
  Vec3: Vector3<Scalar = S>,
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits,
{
  // Project both triangles in 2D
  let a_2d: [Vec2; 3] = [
    project_to_2d(a[0], n),
    project_to_2d(a[1], n),
    project_to_2d(a[2], n),
  ];
  let b_2d: [Vec2; 3] = [
    project_to_2d(b[0], n),
    project_to_2d(b[1], n),
    project_to_2d(b[2], n),
  ];

  // 1. Check all 9 edge-edge combinations
  for i in 0..3 {
    let a_edge_start = a_2d[i];
    let a_edge_end = a_2d[(i + 1) % 3];
    for j in 0..3 {
      let b_edge_start = b_2d[j];
      let b_edge_end = b_2d[(j + 1) % 3];
      if segments_intersect_2d(a_edge_start, a_edge_end, b_edge_start, b_edge_end) {
        return true;
      }
    }
  }

  // 2. Point-in-Triangle tets for complete containment.
  // Is triangle A completely inside triangle B
  if point_in_triangle_2d(a_2d[0], b_2d[0], b_2d[1], b_2d[2]) {
    return true;
  }
  // Is triangle B completely inside triangle A
  if point_in_triangle_2d(b_2d[0], a_2d[0], a_2d[1], a_2d[2]) {
    return true;
  }

  false
}

/// Flatten a triangle A and intersection segment p1,p2 into 2D. then we just test the segment
/// against the 3 edges of the triangle. If it misses all the edges, we check
/// if the segment is completel swallowed by the triangle
fn segment_intersects_triangle<S, Vec3, Vec2>(p1: Vec3, p2: Vec3, a: &[Vec3; 3], n_a: Vec3) -> bool
where
  Vec3: Vector3<Scalar = S>,
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits,
{
  // Project everything in 2D
  let p1_2d: Vec2 = project_to_2d(p1, n_a);
  let p2_2d: Vec2 = project_to_2d(p2, n_a);

  let a0: Vec2 = project_to_2d(a[0], n_a);
  let a1: Vec2 = project_to_2d(a[1], n_a);
  let a2: Vec2 = project_to_2d(a[2], n_a);

  // 1. Edge-Edge tets: Does the segment cross any triangle edge?
  if segments_intersect_2d(p1_2d, p2_2d, a0, a1)
    || segments_intersect_2d(p1_2d, p2_2d, a0, a2)
    || segments_intersect_2d(p1_2d, p2_2d, a1, a2)
  {
    return true;
  }

  // 2. Point-in-Triangle test: Is the segment completeli inside the triangle?
  // we only need to test one endpoint. If it's inside, then the whole segment is
  point_in_triangle_2d(p1_2d, a0, a1, a2)
}

/// Projects a 3D point onto a 3D plane by dropping the dominant axis to the normal
#[inline]
fn project_to_2d<S, Vec3, Vec2>(v: Vec3, n: Vec3) -> Vec2
where
  Vec3: Vector3<Scalar = S>,
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits,
{
  let (ax, ay, az) = (n.x().abs(), n.y().abs(), n.z().abs());
  if ax > ay && ax > az {
    // Normal points mostly towards X Axis
    Vec2::from_components(v.y(), v.z())
  } else if ay > ax && ay > az {
    // Normal points mostly towards Y Axis
    Vec2::from_components(v.x(), v.z())
  } else {
    // Normal points mostly towards Z Axis
    Vec2::from_components(v.x(), v.y())
  }
}

/// Returns > 0 if C is left of segment AB, < 0 if right, 0 if collinear
#[inline]
fn orient_2d<S, Vec2>(a: Vec2, b: Vec2, c: Vec2) -> S
where
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits,
{
  (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())
}

/// Checks if a 2D point p lies inside a triangle (a,b,c)
fn point_in_triangle_2d<Vec2>(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool
where
  Vec2: Vector2,
  Vec2::Scalar: FloatLike + FloatOps + FloatBits,
{
  let w0 = orient_2d(b, c, p);
  let w1 = orient_2d(c, a, p);
  let w2 = orient_2d(a, b, p);
  let _0 = <Vec2::Scalar as MulAddIdentity>::zero();

  // if all weights have the same sign, the point is on the triangle
  (w0 >= _0 && w1 >= _0 && w2 >= _0) || (w0 <= _0 && w1 <= _0 && w2 <= _0)
}

/// Checks if two 2D line segments (p1,p2) and (q1,q2) intersect
fn segments_intersect_2d<Vec2>(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2) -> bool
where
  Vec2: Vector2,
  Vec2::Scalar: FloatLike + FloatOps + FloatBits,
{
  let o1 = orient_2d(p1, p2, q1);
  let o2 = orient_2d(p1, p2, q2);
  let o3 = orient_2d(q1, q2, p1);
  let o4 = orient_2d(q1, q2, p2);

  // if the signs of the orientations are opposites, the segments cross
  let _0 = <Vec2::Scalar as MulAddIdentity>::zero();
  if o1 * o2 <= _0 && o3 * o4 <= _0 {
    // quick ABB overlap to check to handle the edge case of collinear disjoints segments
    if p1.x().min(p2.x()) <= q1.x().max(q2.x())
      && p1.x().max(p2.x()) >= q1.x().min(q2.x())
      && p1.y().min(p2.y()) <= q1.y().max(q2.y())
      && p1.y().max(p2.y()) >= q1.y().min(q2.y())
    {
      return true;
    }
  }

  false
}

/// Compute the exact 3D line segment where Triangle B intersects Plane A
/// let di be the signed distance from vertex bi to Plane A
#[inline]
fn compute_intersection_segment<S, Vec3>(b: &[Vec3; 3], d0: S, d1: S, d2: S) -> (Vec3, Vec3)
where
  Vec3: Vector3<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits + core::ops::Mul<Vec3, Output = Vec3>,
{
  let _0 = S::zero();
  // determine which vertex is isolated on one side of the plane
  // if d0 * d1 is positive, they have the same sign, meaning b2 is isolated
  if (d0 * d1) > _0 {
    // b0 and b1 are on the same side, b2 is isolated
    let t1 = d2 / (d2 - d0);
    let t2 = d2 / (d2 - d1);
    let p1 = b[2] + t1 * (b[0] - b[2]);
    let p2 = b[2] + t2 * (b[0] - b[2]);
    (p1, p2)
  } else if (d0 * d2) > _0 {
    // b0 and b2 are on the same side, b1 is isolated
    let t1 = d1 / (d1 - d0);
    let t2 = d1 / (d1 - d2);
    let p1 = b[1] + t1 * (b[0] - b[1]);
    let p2 = b[1] + t2 * (b[2] - b[1]);
    (p1, p2)
  } else {
    // b1 and b2 are on the same side, b0 is isolated
    let t1 = d0 / (d0 - d1);
    let t2 = d0 / (d0 - d2);
    let p1 = b[0] + t1 * (b[1] - b[0]);
    let p2 = b[0] + t2 * (b[2] - b[0]);
    (p1, p2)
  }
}

// ----------------------------------------------------------------------------
// Box vs Sphere
// ----------------------------------------------------------------------------
pub fn intersect_aabb_sphere<V>(aabb: &AABB<V::Scalar>, bs: &BS<V::Scalar>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32> + core::ops::Mul<V, Output = V>,
{
  let c: V = bs.center();
  let min: V = aabb.min();
  let max: V = aabb.max();

  let closest = c.max(min).min(max);
  let d = c - closest;

  d.length_squared() <= bs.radius() * bs.radius()
}

// ----------------------------------------------------------------------------
// Box vs Triangle
// ----------------------------------------------------------------------------
// TODO: obb_triangle by transforming the triangle itself into the OBB frame of reference,
// so that we can treat it as a AABB
/// AABB/Triangle SAT
pub fn intersect_aabb_triangle<S, Vec3, Vec2>(aabb: &AABB<Vec3::Scalar>, tri: &Triangle) -> bool
where
  Vec3: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  Vec2: Vector2<Scalar = S>,
  S: FloatLike + FloatOps + FloatBits,
{
  let vs: [Vec3; 3] = [(*tri.v0()).into(), (*tri.v1()).into(), (*tri.v2()).into()];
  let n: Vec3 = tri.normal_ccw_unnormalized().into();

  // 1. Check if a vertex is in the box. If yes for at least one of then (all coords) then true
  for v in vs {
    if between_component_wise(v, aabb.min(), aabb.max()) {
      return true;
    }
  }
  // 2. Signed distances for each vertex of box to Triangle Plane
  let aabb_vertices = aabb.vertices();
  let aabb_segment_indices = AABB::<Vec3::Scalar>::edges();
  let aabb_signed_distances: [S; 8] = unsafe {
    aabb_vertices.iter().map(|&vert| n.dot(vert - vs[0])).collect_array().unwrap_unchecked()
  };
  //  If all boxes have signed distances of same sign, no intersection
  if all_same_sign_fold(&aabb_signed_distances) {
    return false;
  }
  //  Otherwise call segment-triangle intersection function and stop on first intersection
  for segment in aabb_segment_indices {
    if segment_intersects_triangle::<S, Vec3, Vec2>(
      aabb_vertices[segment[0]],
      aabb_vertices[segment[1]],
      &vs,
      n,
    ) {
      return true;
    }
  }
  false
}

#[rustfmt::skip]
fn between_component_wise<V>(v: V, min: V, max: V) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  v.x() >= min.x() && v.x() <= max.x()
  && v.y() >= min.y() && v.y() <= max.y()
  && v.z() >= min.z() && v.z() <= max.z()
}

fn all_same_sign_fold<S>(arr: &[S]) -> bool
where
  S: FloatLike,
{
  let _0 = S::zero();
  if let Some(&first) = arr.first() {
    let first_sign = first > _0;
    arr
      .iter()
      // Fold starts with `true` and turns `false` if any sign mismatches
      .fold(true, |acc, &x| acc && ((x > _0) == first_sign))
  } else {
    true
  }
}

// ----------------------------------------------------------------------------
// Sphere vs Triangle
// ----------------------------------------------------------------------------
pub fn intersect_sphere_triangle<Vec3>(sphere: &BS<Vec3::Scalar>, tri: &Triangle) -> bool
where
  Vec3: Vector3 + From<Vec3f32> + From<[Vec3::Scalar; 3]> + Into<[Vec3::Scalar; 3]>,
  Vec3::Scalar: FloatLike + FloatBits + FloatOps + From<f32> + core::ops::Mul<Vec3, Output = Vec3>,
{
  let vs: [Vec3; 3] = [(*tri.v0()).into(), (*tri.v1()).into(), (*tri.v2()).into()];
  let n: Vec3 = tri.normal_ccw_unnormalized().normalize().into();
  let c: Vec3 = sphere.center();
  let r: Vec3::Scalar = sphere.radius();

  // Using n.dot(vs[0]) directly to get the plane constant 'd'
  let d_n: Vec3::Scalar = n.dot(vs[0]);
  let dist_to_plane = n.dot(c) - d_n;

  // 1. Triangle plane intersects sphere? If not, early exit false
  if dist_to_plane.abs() > r {
    return false;
  }

  let r_sq = r * r;

  // 2. If at least a triangle vertex is inside the sphere, early exit true
  for v in vs {
    let diff = c - v;
    // Using squared distance to avoid .sqrt()
    if diff.dot(diff) <= r_sq {
      return true;
    }
  }

  // 3. Project sphere center onto triangle plane.
  // Using the trait bound `Scalar: Mul<Vec3>` ensures we do `scalar * vector`
  let projected_center = c - (dist_to_plane * n);

  // If center inside triangle, early exit true.
  // We test this by checking if the point is on the "inside" of all 3 edge planes using cross products.
  let e0 = vs[1] - vs[0];
  let e1 = vs[2] - vs[1];
  let e2 = vs[0] - vs[2];

  let zero = <Vec3::Scalar>::from(0.0f32);

  let inside_0 = e0.cross(projected_center - vs[0]).dot(n) >= zero;
  let inside_1 = e1.cross(projected_center - vs[1]).dot(n) >= zero;
  let inside_2 = e2.cross(projected_center - vs[2]).dot(n) >= zero;

  if inside_0 && inside_1 && inside_2 {
    return true;
  }

  // 4. Line segment sphere intersection for each triangle edge with projected sphere
  // We stay in 3D and test the distance from the sphere center to the 3D line segment.
  if intersects_edge(vs[0], vs[1], c, r_sq) {
    return true;
  }
  if intersects_edge(vs[1], vs[2], c, r_sq) {
    return true;
  }
  if intersects_edge(vs[2], vs[0], c, r_sq) {
    return true;
  }

  false
}

fn intersects_edge<S, Vec3>(a: Vec3, b: Vec3, c: Vec3, r_sq: S) -> bool
where
  Vec3: Vector3<Scalar = S> + From<Vec3f32> + From<[Vec3::Scalar; 3]> + Into<[Vec3::Scalar; 3]>,
  S: FloatLike + FloatBits + FloatOps + From<f32> + core::ops::Mul<Vec3, Output = Vec3>,
{
  let zero = <Vec3::Scalar>::from(0.0f32);
  let one = <Vec3::Scalar>::from(1.0f32);

  let ab = b - a;
  let ac = c - a;

  // Project center onto the line to find the parameterized distance 't'
  let t = ac.dot(ab) / ab.dot(ab);

  // IMPORTANT OPTIMIZATION:
  // If t <= 0.0 or t >= 1.0, the closest point on the segment is a vertex.
  // We ALREADY checked the vertices in Step 2!
  // So we only need to do the math if the closest point is strictly inside the segment.
  if t > zero && t < one {
    let closest_point = a + (t * ab);
    let diff = c - closest_point;
    if diff.dot(diff) <= r_sq {
      return true;
    }
  }
  false
}

#[inline]
fn point_in_sphere<S, Vec3>(p: Vec3, c: Vec3, r: S) -> bool
where
  Vec3: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike + From<f32> + core::ops::Mul<Vec3, Output = Vec3>,
{
  (p - c).length_squared() <= r.squared()
}

#[inline]
fn plane_constant<S, Vec3>(normal: Vec3, v0: Vec3) -> S
where
  Vec3: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike + From<f32> + core::ops::Mul<Vec3, Output = Vec3>,
{
  // any vertex of a triangle can be used. Normal must be the proper unit vector
  // The constant 'd' is the dot product of the normal and the point
  normal.dot(v0)
}

// ----------------------------------------------------------------------------
// Ray vs Sphere
// ----------------------------------------------------------------------------
pub fn intersect_ray_sphere<V>(ray: &Ray<V>, bs: &BS<V::Scalar>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32> + core::ops::Mul<V, Output = V>,
{
  let m = ray.origin - bs.center();
  let b = m.dot(ray.direction);
  let c = m.dot(m) - bs.radius() * bs.radius();

  let _0 = V::Scalar::from_f32(0.0);
  if c > _0 && b > _0 {
    return false;
  }

  let discr = b * b - c;
  if discr < _0 {
    return false;
  }

  let t = -b - discr.sqrt();
  if t > ray.length {
    return false;
  }

  true
}

// ----------------------------------------------------------------------------
// Ray vs Triangle
// ----------------------------------------------------------------------------
pub fn intersect_ray_triangle<V>(ray: &Ray<V>, tri: &Triangle) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32> + core::ops::Mul<V, Output = V>,
{
  let epsilon = V::Scalar::from_f32(1e-6);
  let v0: V = (*tri.v0()).into();
  let v1: V = (*tri.v1()).into();
  let v2: V = (*tri.v2()).into();

  let edge1 = v1 - v0;
  let edge2 = v2 - v0;
  let h = ray.direction.cross(edge2);
  let a = edge1.dot(h);

  if a > -epsilon && a < epsilon {
    return false;
  }

  let f: V::Scalar = V::Scalar::from_f32(1.0) / a;
  let s = ray.origin - v0;
  let u = f * s.dot(h);

  let _0 = V::Scalar::from_f32(0.0);
  let _1 = V::Scalar::from_f32(1.0);

  if u < _0 || u > _1 {
    return false;
  }

  let q = s.cross(edge1);
  let v = f * ray.direction.dot(q);

  if v < _0 || u + v > _1 {
    return false;
  }

  let t = f * edge2.dot(q);

  if t > epsilon && t <= ray.length {
    return true;
  }

  false
}

// ----------------------------------------------------------------------------
// Ray vs OBB
// ----------------------------------------------------------------------------
pub fn intersect_ray_obb<S, V, M>(ray: &Ray<V>, obb: &OBB<S>) -> bool
where
  M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike
    + FloatOps
    + FloatBits
    + From<f32>
    + core::ops::Mul<V, Output = V>
    + core::ops::Mul<M, Output = M>,
{
  let epsilon = S::from_f32(1e-6);
  let mut tmin = S::from_f32(0.0);
  let mut tmax = ray.length;

  let p = obb.translation::<V>() - ray.origin;
  let axes: [V; 3] = obb.axes();
  let extents: V = obb.half_extent();

  // Test intersection against the 3 OBB axes
  for i in 0..3 {
    let axis = axes[i];
    let e = extents.component(i).unwrap();
    let d = axis.dot(p);
    let f = ray.direction.dot(axis);

    if f.abs() > epsilon {
      let mut t1 = (d - e) / f;
      let mut t2 = (d + e) / f;
      if t1 > t2 {
        core::mem::swap(&mut t1, &mut t2);
      }
      if t1 > tmin {
        tmin = t1;
      }
      if t2 < tmax {
        tmax = t2;
      }
      if tmin > tmax {
        return false;
      }
      if tmax < S::from_f32(0.0) {
        return false;
      }
    } else if d < -e || d > e {
      return false;
    }
  }

  true
}

// ----------------------------------------------------------------------------
// Ray vs Box
// ----------------------------------------------------------------------------
pub fn intersect_ray_aabb<V>(ray: &Ray<V>, aabb: &AABB<V::Scalar>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + FloatOps + FloatBits + From<f32>,
{
  let mut tmin = V::Scalar::from_f32(0.0);
  let mut tmax = ray.length;

  let min_val: V = aabb.min();
  let max_val: V = aabb.max();

  let dir_x = ray.direction.x();
  let dir_y = ray.direction.y();
  let dir_z = ray.direction.z();

  let origin_x = ray.origin.x();
  let origin_y = ray.origin.y();
  let origin_z = ray.origin.z();

  let epsilon = V::Scalar::from_f32(1e-6);

  if dir_x.abs() < epsilon {
    if origin_x < min_val.x() || origin_x > max_val.x() {
      return false;
    }
  } else {
    let ood: V::Scalar = V::Scalar::from_f32(1.0) / dir_x;
    let mut t1 = (min_val.x() - origin_x) * ood;
    let mut t2 = (max_val.x() - origin_x) * ood;
    if t1 > t2 {
      core::mem::swap(&mut t1, &mut t2);
    }
    if t1 > tmin {
      tmin = t1;
    }
    if t2 < tmax {
      tmax = t2;
    }
    if tmin > tmax {
      return false;
    }
  }

  if dir_y.abs() < epsilon {
    if origin_y < min_val.y() || origin_y > max_val.y() {
      return false;
    }
  } else {
    let ood: V::Scalar = V::Scalar::from_f32(1.0) / dir_y;
    let mut t1 = (min_val.y() - origin_y) * ood;
    let mut t2 = (max_val.y() - origin_y) * ood;
    if t1 > t2 {
      core::mem::swap(&mut t1, &mut t2);
    }
    if t1 > tmin {
      tmin = t1;
    }
    if t2 < tmax {
      tmax = t2;
    }
    if tmin > tmax {
      return false;
    }
  }

  if dir_z.abs() < epsilon {
    if origin_z < min_val.z() || origin_z > max_val.z() {
      return false;
    }
  } else {
    let ood: V::Scalar = V::Scalar::from_f32(1.0) / dir_z;
    let mut t1 = (min_val.z() - origin_z) * ood;
    let mut t2 = (max_val.z() - origin_z) * ood;
    if t1 > t2 {
      core::mem::swap(&mut t1, &mut t2);
    }
    if t1 > tmin {
      tmin = t1;
    }
    if t2 < tmax {
      tmax = t2;
    }
    if tmin > tmax {
      return false;
    }
  }

  true
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::math::collision::bounds::{AABB, BS, OBB};
  use crate::simulation::comet::Triangle;
  use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;
  use aethervk_oshal_rlib::math::vector::vec2::Vec2f32;
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

  fn mk_vec(x: f32, y: f32, z: f32) -> Vec3f32 {
    Vec3f32::from_components(x, y, z)
  }

  fn mk_tri(v0: Vec3f32, v1: Vec3f32, v2: Vec3f32) -> Triangle {
    Triangle {
      vertices: [v0, v1, v2],
    }
  }

  // 1. intersect_sphere_sphere
  #[test]
  fn test_intersect_sphere_sphere() {
    // Positive: overlapping
    assert!(intersect_sphere_sphere::<Vec3f32>(
      &BS::new(mk_vec(0., 0., 0.), 2.0),
      &BS::new(mk_vec(3., 0., 0.), 2.0)
    ));
    // Edge 1: touching
    assert!(intersect_sphere_sphere::<Vec3f32>(
      &BS::new(mk_vec(0., 0., 0.), 2.0),
      &BS::new(mk_vec(4., 0., 0.), 2.0)
    ));
    // Edge 2: one inside other
    assert!(intersect_sphere_sphere::<Vec3f32>(
      &BS::new(mk_vec(0., 0., 0.), 5.0),
      &BS::new(mk_vec(1., 0., 0.), 1.0)
    ));
    // Edge 3: disjoint
    assert!(!intersect_sphere_sphere::<Vec3f32>(
      &BS::new(mk_vec(0., 0., 0.), 2.0),
      &BS::new(mk_vec(5., 0., 0.), 2.0)
    ));
  }

  // 2. intersect_aabb_aabb
  #[test]
  fn test_intersect_aabb_aabb() {
    // Positive: overlapping
    assert!(intersect_aabb_aabb::<Vec3f32>(
      &AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.)),
      &AABB::new(mk_vec(1., 1., 1.), mk_vec(3., 3., 3.))
    ));
    // Edge 1: touching face
    assert!(intersect_aabb_aabb::<Vec3f32>(
      &AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.)),
      &AABB::new(mk_vec(2., 0., 0.), mk_vec(4., 2., 2.))
    ));
    // Edge 2: touching corner
    assert!(intersect_aabb_aabb::<Vec3f32>(
      &AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.)),
      &AABB::new(mk_vec(2., 2., 2.), mk_vec(4., 4., 4.))
    ));
    // Edge 3: disjoint
    assert!(!intersect_aabb_aabb::<Vec3f32>(
      &AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.)),
      &AABB::new(mk_vec(3., 0., 0.), mk_vec(4., 2., 2.))
    ));
  }

  // 3. intersect_obb_obb
  #[test]
  fn test_intersect_obb_obb() {
    let ident = Mat3f32::identity();
    // Positive: overlapping
    assert!(intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
      &OBB::new(mk_vec(0., 0., 0.), ident, mk_vec(1., 1., 1.)),
      &OBB::new(mk_vec(1., 0., 0.), ident, mk_vec(1., 1., 1.))
    ));
    // Edge 1: touching
    assert!(intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
      &OBB::new(mk_vec(0., 0., 0.), ident, mk_vec(1., 1., 1.)),
      &OBB::new(mk_vec(2., 0., 0.), ident, mk_vec(1., 1., 1.))
    ));
    // Edge 2: rotated, touching
    let rot = Mat3f32::from_array(&[0., -1., 0., 1., 0., 0., 0., 0., 1.]);
    assert!(intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
      &OBB::new(mk_vec(0., 0., 0.), ident, mk_vec(1., 1., 1.)),
      &OBB::new(mk_vec(2., 0., 0.), rot, mk_vec(1., 1., 1.))
    ));
    // Edge 3: disjoint
    assert!(!intersect_obb_obb::<f32, Vec3f32, Mat3f32>(
      &OBB::new(mk_vec(0., 0., 0.), ident, mk_vec(1., 1., 1.)),
      &OBB::new(mk_vec(3., 0., 0.), ident, mk_vec(1., 1., 1.))
    ));
  }

  // 4. intersect_triangle_triangle
  #[test]
  fn test_intersect_triangle_triangle() {
    let t1 = mk_tri(mk_vec(0., 0., 0.), mk_vec(2., 0., 0.), mk_vec(0., 2., 0.));
    // Positive: crossing
    let t2 = mk_tri(
      mk_vec(1., 1., -1.),
      mk_vec(1., 1., 1.),
      mk_vec(-1., -1., 0.),
    );
    assert!(intersect_triangle_triangle::<f32, Vec3f32, Vec2f32>(
      &t1, &t2
    ));
    // Edge 1: coplanar overlapping
    let t3 = mk_tri(mk_vec(1., 0., 0.), mk_vec(3., 0., 0.), mk_vec(1., 2., 0.));
    assert!(intersect_triangle_triangle::<f32, Vec3f32, Vec2f32>(
      &t1, &t3
    ));
    // Edge 2: coplanar disjoint
    let t4 = mk_tri(mk_vec(3., 0., 0.), mk_vec(5., 0., 0.), mk_vec(3., 2., 0.));
    assert!(!intersect_triangle_triangle::<f32, Vec3f32, Vec2f32>(
      &t1, &t4
    ));
    // Edge 3: completely disjoint
    let t5 = mk_tri(mk_vec(0., 0., 5.), mk_vec(2., 0., 5.), mk_vec(0., 2., 5.));
    assert!(!intersect_triangle_triangle::<f32, Vec3f32, Vec2f32>(
      &t1, &t5
    ));
  }

  // 5. intersect_aabb_sphere
  #[test]
  fn test_intersect_aabb_sphere() {
    let aabb = AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.));
    // Positive: overlapping
    assert!(intersect_aabb_sphere::<Vec3f32>(
      &aabb,
      &BS::new(mk_vec(1., 1., 1.), 2.0)
    ));
    // Edge 1: touching face
    assert!(intersect_aabb_sphere::<Vec3f32>(
      &aabb,
      &BS::new(mk_vec(3., 1., 1.), 1.0)
    ));
    // Edge 2: touching corner
    let corner_dist = (3.0_f32).sqrt();
    assert!(intersect_aabb_sphere::<Vec3f32>(
      &aabb,
      &BS::new(mk_vec(3., 3., 3.), corner_dist)
    ));
    // Edge 3: disjoint
    assert!(!intersect_aabb_sphere::<Vec3f32>(
      &aabb,
      &BS::new(mk_vec(4., 1., 1.), 1.0)
    ));
  }

  // 6. intersect_aabb_triangle
  #[test]
  fn test_intersect_aabb_triangle() {
    let aabb = AABB::new(mk_vec(0., 0., 0.), mk_vec(2., 2., 2.));
    // Positive: inside
    let t1 = mk_tri(
      mk_vec(0.5, 0.5, 0.5),
      mk_vec(1.5, 0.5, 0.5),
      mk_vec(0.5, 1.5, 0.5),
    );
    assert!(intersect_aabb_triangle::<f32, Vec3f32, Vec2f32>(&aabb, &t1));
    // Edge 1: crossing edge
    let t2 = mk_tri(mk_vec(-1., 1., 1.), mk_vec(3., 1., 1.), mk_vec(1., 3., 1.));
    assert!(intersect_aabb_triangle::<f32, Vec3f32, Vec2f32>(&aabb, &t2));
    // Edge 2: touching corner exactly
    let t3 = mk_tri(mk_vec(2., 2., 2.), mk_vec(3., 2., 2.), mk_vec(2., 3., 2.));
    assert!(intersect_aabb_triangle::<f32, Vec3f32, Vec2f32>(&aabb, &t3));
    // Edge 3: disjoint
    let t4 = mk_tri(mk_vec(3., 3., 3.), mk_vec(4., 3., 3.), mk_vec(3., 4., 3.));
    assert!(!intersect_aabb_triangle::<f32, Vec3f32, Vec2f32>(
      &aabb, &t4
    ));
  }

  // 7. intersect_sphere_triangle
  #[test]
  fn test_intersect_sphere_triangle() {
    let s = BS::new(mk_vec(0., 0., 0.), 2.0);
    // Positive: overlapping
    let t1 = mk_tri(mk_vec(1., 0., 0.), mk_vec(3., 0., 0.), mk_vec(1., 2., 0.));
    assert!(intersect_sphere_triangle::<Vec3f32>(&s, &t1));
    // Edge 1: touching vertex
    let t2 = mk_tri(mk_vec(2., 0., 0.), mk_vec(4., 0., 0.), mk_vec(2., 2., 0.));
    assert!(intersect_sphere_triangle::<Vec3f32>(&s, &t2));
    // Edge 2: completely inside sphere
    let t3 = mk_tri(
      mk_vec(0.1, 0., 0.),
      mk_vec(0.5, 0., 0.),
      mk_vec(0.1, 0.5, 0.),
    );
    assert!(intersect_sphere_triangle::<Vec3f32>(&s, &t3));
    // Edge 3: disjoint
    let t4 = mk_tri(mk_vec(3., 0., 0.), mk_vec(5., 0., 0.), mk_vec(3., 2., 0.));
    assert!(!intersect_sphere_triangle::<Vec3f32>(&s, &t4));
  }

  // 8. intersect_ray_sphere
  #[test]
  fn test_intersect_ray_sphere() {
    let s = BS::new(mk_vec(5., 0., 0.), 2.0);
    // Positive: crossing center
    let r1 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(1., 0., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_sphere(&r1, &s));
    // Edge 1: tangent
    let r2 = Ray {
      origin: mk_vec(0., 2., 0.),
      direction: mk_vec(1., 0., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_sphere(&r2, &s));
    // Edge 2: origin inside
    let r3 = Ray {
      origin: mk_vec(4., 0., 0.),
      direction: mk_vec(0., 1., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_sphere(&r3, &s));
    // Edge 3: disjoint / wrong direction
    let r4 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(-1., 0., 0.),
      length: 10.0,
    };
    assert!(!intersect_ray_sphere(&r4, &s));
  }

  // 9. intersect_ray_triangle
  #[test]
  fn test_intersect_ray_triangle() {
    let t = mk_tri(
      mk_vec(-1., -1., 5.),
      mk_vec(1., -1., 5.),
      mk_vec(0., 1., 5.),
    );
    // Positive: piercing
    let r1 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(0., 0., 1.),
      length: 10.0,
    };
    assert!(intersect_ray_triangle(&r1, &t));
    // Edge 1: missing triangle slightly
    let r2 = Ray {
      origin: mk_vec(0., 2., 0.),
      direction: mk_vec(0., 0., 1.),
      length: 10.0,
    };
    assert!(!intersect_ray_triangle(&r2, &t));
    // Edge 2: hitting edge
    let r3 = Ray {
      origin: mk_vec(0., -1., 0.),
      direction: mk_vec(0., 0., 1.),
      length: 10.0,
    };
    assert!(intersect_ray_triangle(&r3, &t));
    // Edge 3: too short
    let r4 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(0., 0., 1.),
      length: 4.0,
    };
    assert!(!intersect_ray_triangle(&r4, &t));
  }

  // 10. intersect_ray_aabb
  #[test]
  fn test_intersect_ray_aabb() {
    let aabb = AABB::new(mk_vec(4., -1., -1.), mk_vec(6., 1., 1.));
    // Positive: piercing
    let r1 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(1., 0., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_aabb(&r1, &aabb));
    // Edge 1: grazing face
    let r2 = Ray {
      origin: mk_vec(0., 1., 0.),
      direction: mk_vec(1., 0., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_aabb(&r2, &aabb));
    // Edge 2: origin inside
    let r3 = Ray {
      origin: mk_vec(5., 0., 0.),
      direction: mk_vec(0., 1., 0.),
      length: 10.0,
    };
    assert!(intersect_ray_aabb(&r3, &aabb));
    // Edge 3: disjoint / short ray
    let r4 = Ray {
      origin: mk_vec(0., 0., 0.),
      direction: mk_vec(1., 0., 0.),
      length: 3.0,
    };
    assert!(!intersect_ray_aabb(&r4, &aabb));
  }
}
