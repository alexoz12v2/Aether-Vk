//! Intersection tests
//! Contains logic for checking intersections between different shapes

use aethervk_oshal_rlib::math::{
  FloatLike, MulAddIdentity,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix, Matrix3, Matrix4},
  vector::{Vector, Vector2, Vector3, vec3::Vec3f32},
};
use itertools::Itertools;

use crate::{
  math::collision::bounds::{AABB, BS, OBB},
  simulation::comet::Triangle,
};

/// A line segment / ray defined by an origin and a direction vector and length
#[derive(Debug, Clone, Copy)]
pub struct Ray<V>
where
  V: Vector3,
{
  pub origin: V,
  pub direction: V,
  pub length: V::Scalar, // also known as t_{max}
}

// ----------------------------------------------------------------------------
// Sphere vs Sphere
// ----------------------------------------------------------------------------
pub fn intersect_sphere_sphere<V>(a: &BS<V>, b: &BS<V>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + From<i32> + From<f32> + core::ops::Mul<V, Output = V>,
{
  let d = a.center() - b.center();
  let r_sum = a.radius() + b.radius();
  d.length_squared() <= r_sum * r_sum
}

// ----------------------------------------------------------------------------
// Box vs Box
// ----------------------------------------------------------------------------
// TODO: AABB vs AABB, AABB vs OBB, OBB vs OBB (15 combinations)

pub fn intersect_aabb_aabb<V>(a: &AABB<V>, b: &AABB<V>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  let a_min = a.min();
  let a_max = a.max();
  let b_min = b.min();
  let b_max = b.max();

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
pub fn intersect_obb_obb<S, V, M>(a: &OBB<S, V, M>, b: &OBB<S, V, M>) -> bool
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
  let pos_a = a.translation();
  let pos_b = b.translation();
  let ext_a = a.half_extent();
  let ext_b = b.half_extent();
  let axes_a = a.axes();
  let axes_b = b.axes();

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

  let a: [Vec3; 3] = t_a
    .vertices
    .iter()
    .map(|v| (*v).into())
    .collect_array()
    .unwrap();
  let b: [Vec3; 3] = t_b
    .vertices
    .iter()
    .map(|v| (*v).into())
    .collect_array()
    .unwrap();

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
    Vec2::from_components(v.x(), v.z())
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
pub fn intersect_aabb_sphere<V>(aabb: &AABB<V>, bs: &BS<V>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + From<i32> + From<f32> + core::ops::Mul<V, Output = V>,
{
  let c = bs.center();
  let min = aabb.min();
  let max = aabb.max();

  let closest = c.max(min).min(max);
  let d = c - closest;

  d.length_squared() <= bs.radius() * bs.radius()
}

// ----------------------------------------------------------------------------
// Box vs Triangle
// ----------------------------------------------------------------------------
pub fn intersect_aabb_triangle<V>(_aabb: &AABB<V>, _tri: &Triangle) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  // TODO: AABB/Triangle SAT
  false
}

// ----------------------------------------------------------------------------
// Sphere vs Triangle
// ----------------------------------------------------------------------------
pub fn intersect_sphere_triangle<V>(_bs: &BS<V>, _tri: &Triangle) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + From<i32> + From<f32> + core::ops::Mul<V, Output = V>,
{
  // TODO: Sphere/Triangle closest point
  false
}

// ----------------------------------------------------------------------------
// Ray vs Sphere
// ----------------------------------------------------------------------------
pub fn intersect_ray_sphere<V>(ray: &Ray<V>, bs: &BS<V>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike + From<i32> + From<f32> + core::ops::Mul<V, Output = V>,
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
pub fn intersect_ray_triangle<V>(_ray: &Ray<V>, _tri: &Triangle) -> bool
where
  V: Vector3,
  V::Scalar: FloatLike,
{
  // TODO: Möller–Trumbore or watertight
  false
}

// ----------------------------------------------------------------------------
// Ray vs Box
// ----------------------------------------------------------------------------
pub fn intersect_ray_aabb<V>(_ray: &Ray<V>, _aabb: &AABB<V>) -> bool
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  // TODO: Slab method
  false
}
