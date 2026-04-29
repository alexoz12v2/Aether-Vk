//! Bounds Module
//! This module will contain all the necessary math algorithms to construct and manage bounds summary
//! descriptions of polygonal meshes, in particular
//! - Axis Aligned Bounding Boxes
//! - Object Aligned Bounding Boxes
//! - Spherical Bounds
use aethervk_oshal_rlib::{
  self as oshal,
  math::{
    FloatLike,
    floating::{FloatBits, FloatOps},
    matrix::{Matrix, Matrix3, Matrix4, MatrixVectorMul, mat3::Mat3f32, mat4::Mat4x4f32},
    vector::{Vector, Vector3, Vector4},
  },
};
use oshal::math::vector::vec3::Vec3f32;

use crate::{math::jacobi_diagonalization, simulation::comet::Triangle};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AABB<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  min: [S; 3],
  max: [S; 3],
}

impl<S> AABB<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  #[inline]
  pub fn vertices<V>(&self) -> [V; 8]
  where
    V: Vector3<Scalar = S>,
  {
    [
      V::from_components(self.min[0], self.min[1], self.min[2]), // Back Bottom Left
      V::from_components(self.max[0], self.min[1], self.min[2]), // Back Bottom Right
      V::from_components(self.min[0], self.max[1], self.min[2]), // Back Top Left
      V::from_components(self.max[0], self.max[1], self.min[2]), // Back Top Right
      V::from_components(self.min[0], self.min[1], self.max[2]), // Front Bottom Left
      V::from_components(self.max[0], self.min[1], self.max[2]), // Front Bottom Right
      V::from_components(self.min[0], self.max[1], self.max[2]), // Front Top Left
      V::from_components(self.max[0], self.max[1], self.max[2]), // Front Top Right
    ]
  }

  #[inline]
  pub fn edges() -> [[usize; 2]; 12] {
    [
      [0, 1],
      [2, 3],
      [4, 5],
      [6, 7],
      [0, 2],
      [1, 3],
      [4, 6],
      [5, 7],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ]
  }

  pub fn new<V>(min: V, max: V) -> Self
  where
    V: Vector3<Scalar = S> + Into<[S; 3]>,
  {
    Self {
      min: min.into(),
      max: max.into(),
    }
  }

  pub fn min<V>(&self) -> V
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    self.min.into()
  }

  pub fn max<V>(&self) -> V
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    self.max.into()
  }

  pub fn center<V>(&self) -> V
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    (self.min::<V>() + self.max()) * S::from_f32(0.5)
  }

  pub fn half_extents<V>(&self) -> V
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let ex: V = self.max::<V>() - self.min();
    ex * S::from_f32(0.5)
  }

  pub fn contains_aabb<V>(&self, other: &Self) -> bool
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let s_min: V = self.min();
    let s_max: V = self.max();
    let o_min: V = other.min();
    let o_max: V = other.max();
    let eps = S::from_f32(1e-4);

    o_min.x() >= s_min.x() - eps
      && o_max.x() <= s_max.x() + eps
      && o_min.y() >= s_min.y() - eps
      && o_max.y() <= s_max.y() + eps
      && o_min.z() >= s_min.z() - eps
      && o_max.z() <= s_max.z() + eps
  }

  pub fn contains_obb<V>(&self, other: &OBB<S>) -> bool
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let s_min: V = self.min();
    let s_max: V = self.max();
    // TODO configurable epsilon from collision
    let eps = S::from_f32(1e-4);

    for v in &other.vertices::<V>() {
      if v.x() < s_min.x() - eps
        || v.x() > s_max.x() + eps
        || v.y() < s_min.y() - eps
        || v.y() > s_max.y() + eps
        || v.z() < s_min.z() - eps
        || v.z() > s_max.z() + eps
      {
        return false;
      }
    }
    true
  }

  pub fn encapsulate_aabb<V>(&mut self, other: &Self)
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let mut s_min: V = self.min();
    let mut s_max: V = self.max();
    let o_min: V = other.min();
    let o_max: V = other.max();

    s_min = s_min.min(o_min);
    s_max = s_max.max(o_max);

    self.min = [s_min.x(), s_min.y(), s_min.z()];
    self.max = [s_max.x(), s_max.y(), s_max.z()];
  }

  pub fn encapsulate_obb<V>(&mut self, other: &OBB<S>)
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let mut s_min: V = self.min();
    let mut s_max: V = self.max();

    for v in other.vertices() {
      s_min = s_min.min(v);
      s_max = s_max.max(v);
    }

    self.min = [s_min.x(), s_min.y(), s_min.z()];
    self.max = [s_max.x(), s_max.y(), s_max.z()];
  }

  pub fn contains_point<V>(&self, p: V) -> bool
  where
    V: Vector3<Scalar = S> + From<[S; 3]>,
  {
    let s_min: V = self.min();
    let s_max: V = self.max();
    // TODO configurable
    let eps = V::Scalar::from_f32(1e-4);

    p.x() >= s_min.x() - eps
      && p.x() <= s_max.x() + eps
      && p.y() >= s_min.y() - eps
      && p.y() <= s_max.y() + eps
      && p.z() >= s_min.z() - eps
      && p.z() <= s_max.z() + eps
  }

  pub fn from_tris<V, I>(triangles: I) -> Self
  where
    I: IntoIterator<Item = Triangle>,
    I::IntoIter: Clone,
    V: Vector3<Scalar = S> + From<Vec3f32> + Into<[S; 3]>,
  {
    let iter = triangles.into_iter();
    let mut min = V::splat(V::Scalar::from_f32(f32::INFINITY));
    let mut max = V::splat(V::Scalar::from_f32(-f32::INFINITY));
    let mut count = 0;

    for tri in iter {
      count += 1;
      min = min
        .min((*tri.v0()).into())
        .min((*tri.v1()).into())
        .min((*tri.v2()).into());
      max = max
        .max((*tri.v0()).into())
        .max((*tri.v1()).into())
        .max((*tri.v2()).into());
    }

    if count == 0 {
      min = V::zero();
      max = V::zero();
    }

    Self {
      min: min.into(),
      max: max.into(),
    }
  }

  pub fn transform<V, M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = S> + MatrixVectorMul + From<Mat4x4f32>,
    M2::Vector: Vector4<Scalar = S>,
    V: Vector3<Scalar = S> + From<[S; 3]> + Into<[S; 3]>,
  {
    let min: V = self.min();
    let max: V = self.max();

    let corners = [
      V::from_components(min.x(), min.y(), min.z()),
      V::from_components(max.x(), min.y(), min.z()),
      V::from_components(min.x(), max.y(), min.z()),
      V::from_components(max.x(), max.y(), min.z()),
      V::from_components(min.x(), min.y(), max.z()),
      V::from_components(max.x(), min.y(), max.z()),
      V::from_components(min.x(), max.y(), max.z()),
      V::from_components(max.x(), max.y(), max.z()),
    ];

    let _1 = <V::Scalar as FloatLike>::from_f32(1.0);
    let mut new_min = V::splat(<V::Scalar as FloatLike>::from_f32(f32::INFINITY));
    let mut new_max = V::splat(<V::Scalar as FloatLike>::from_f32(-f32::INFINITY));

    for c in corners.iter() {
      let v4 = M2::Vector::from_components(c.x(), c.y(), c.z(), _1);
      let transformed = transform.mul_vector(v4);
      let tv = V::from_components(transformed.x(), transformed.y(), transformed.z());
      new_min = new_min.min(tv);
      new_max = new_max.max(tv);
    }

    Self::new(new_min, new_max)
  }

  // TODO unit test (eg 45 deg rotation and translation)
  pub fn transform_to_obb<M, V, M2>(&self, transform: &M2) -> OBB<S>
  where
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
    V: Vector3<Scalar = S> + Into<[S; 3]>,
    M2: Matrix4<Scalar = S> + MatrixVectorMul + From<Mat4x4f32>,
    M2::Vector: Vector4<Scalar = S>,
  {
    // 1. Extract the pure rotation matrix (3x3) from the 4x4 transform
    let rot: M = (*transform).into_linear();

    // 2. Reconstruct the local min and max vectors
    let min_v = V::from_components(self.min[0], self.min[1], self.min[2]);
    let max_v = V::from_components(self.max[0], self.max[1], self.max[2]);

    // 3. Calculate local center and half-extents
    // (Assuming your vector type V supports scalar multiplication and addition/subtraction)
    let _0_5 = V::Scalar::from_f32(0.5);
    let local_center = (max_v + min_v) * _0_5;
    let half_extents = (max_v - min_v) * _0_5;

    // 4. Transform the center to world space using the FULL 4x4 matrix
    // Note: Replace `transform_point` with whatever method your Mat4x4f32 uses
    // to apply rotation + translation to a 3D point (w = 1.0)
    let world_center = rot.mul_vector(local_center);

    OBB::new(world_center, rot, half_extents)
  }
}

impl AABB<f32> {
  pub fn vertices_f32(&self) -> [Vec3f32; 8] {
    self.vertices::<Vec3f32>()
  }
  pub fn min_f32(&self) -> Vec3f32 {
    self.min.into()
  }
  pub fn max_f32(&self) -> Vec3f32 {
    self.max.into()
  }
  pub fn center_f32(&self) -> Vec3f32 {
    self.center()
  }
  pub fn half_extents_f32(&self) -> Vec3f32 {
    self.half_extents()
  }
  pub fn contains_aabb_f32(&self, other: &Self) -> bool {
    self.contains_aabb::<Vec3f32>(other)
  }
  pub fn contains_obb_f32(&self, other: &OBB<f32>) -> bool {
    self.contains_obb::<Vec3f32>(other)
  }
  pub fn encapsulate_aabb_f32(&mut self, other: &Self) {
    self.encapsulate_aabb::<Vec3f32>(other)
  }
  pub fn encapsulate_obb_f32(&mut self, other: &OBB<f32>) {
    self.encapsulate_obb::<Vec3f32>(other)
  }
  pub fn contains_point_f32(&self, point: Vec3f32) -> bool {
    self.contains_point(point)
  }
  pub fn transform_f32(&self, transform: &Mat4x4f32) -> Self {
    self.transform::<Vec3f32, Mat4x4f32>(transform)
  }
  pub fn transform_to_obb_f32(&self, transform: &Mat4x4f32) -> OBB<f32> {
    self.transform_to_obb::<Mat3f32, Vec3f32, Mat4x4f32>(transform)
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BS<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  center_radius: [S; 4],
}

impl<S> BS<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  pub fn center<V>(&self) -> V
  where
    V: Vector3<Scalar = S>,
  {
    V::from_components(
      self.center_radius[0],
      self.center_radius[1],
      self.center_radius[2],
    )
  }

  pub fn radius(&self) -> S {
    self.center_radius[3]
  }

  pub fn new<V>(center: V, radius: S) -> Self
  where
    V: Vector3<Scalar = S>,
  {
    Self {
      center_radius: [center.x(), center.y(), center.z(), radius],
    }
  }

  pub fn from_tris<V, I>(triangles: I) -> Self
  where
    I: IntoIterator<Item = Triangle>,
    I::IntoIter: Clone,
    V: Vector3<Scalar = S> + From<Vec3f32>,
  {
    let iter_pass_1 = triangles.into_iter();
    let iter_pass_2 = iter_pass_1.clone();

    let mut max_x = V::splat(V::Scalar::from_f32(-f32::INFINITY));
    let mut min_x = V::splat(V::Scalar::from_f32(f32::INFINITY));
    let mut max_y = V::splat(V::Scalar::from_f32(-f32::INFINITY));
    let mut min_y = V::splat(V::Scalar::from_f32(f32::INFINITY));
    let mut max_z = V::splat(V::Scalar::from_f32(-f32::INFINITY));
    let mut min_z = V::splat(V::Scalar::from_f32(f32::INFINITY));
    let mut count = 0;

    for tri in iter_pass_1 {
      count += 1;
      for v in tri.vertices {
        if V::Scalar::from_f32(v.x()) > max_x.x() {
          max_x = v.into();
        }
        if V::Scalar::from_f32(v.x()) < min_x.x() {
          min_x = v.into();
        }
        if V::Scalar::from_f32(v.y()) > max_y.y() {
          max_y = v.into();
        }
        if V::Scalar::from_f32(v.y()) < min_y.y() {
          min_y = v.into();
        }
        if V::Scalar::from_f32(v.z()) > max_z.z() {
          max_z = v.into();
        }
        if V::Scalar::from_f32(v.z()) < min_z.z() {
          min_z = v.into();
        }
      }
    }

    if count == 0 {
      return Self::new(V::zero(), V::Scalar::from_f32(0.0));
    }

    let dist2_x = (max_x - min_x).length_squared();
    let dist2_y = (max_y - min_y).length_squared();
    let dist2_z = (max_z - min_z).length_squared();

    let mut max_dist2 = dist2_x;
    let mut p1 = min_x;
    let mut p2 = max_x;

    if dist2_y > max_dist2 {
      max_dist2 = dist2_y;
      p1 = min_y;
      p2 = max_y;
    }
    if dist2_z > max_dist2 {
      max_dist2 = dist2_z;
      p1 = min_z;
      p2 = max_z;
    }

    let _0_5 = V::Scalar::from_f32(0.5);
    let mut center = (p1 + p2) * _0_5;
    let mut radius: V::Scalar = max_dist2.sqrt() * _0_5;
    let mut rad2 = radius * radius;

    let _2_0 = V::Scalar::from_f32(2.0);
    for tri in iter_pass_2 {
      for v in tri.vertices {
        let v_vec = V::from(v);
        let offset = v_vec - center;
        let dist2 = offset.length_squared();
        if dist2 > rad2 {
          let dist = dist2.sqrt();
          let new_radius = (radius + dist) * _0_5;
          let shift_ratio = (dist - radius) / (dist * _2_0);
          center = center + (offset * shift_ratio);
          radius = new_radius;
          rad2 = radius * radius;
        }
      }
    }
    Self::new(center, radius)
  }

  pub fn transform<V, M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = S> + MatrixVectorMul,
    M2::Vector: Vector4<Scalar = S>,
    V: Vector3<Scalar = S>,
  {
    let center: V = self.center();
    let _1 = <V::Scalar as FloatLike>::from_f32(1.0);
    let center4 = M2::Vector::from_components(center.x(), center.y(), center.z(), _1);
    let new_center4 = transform.mul_vector(center4);
    let new_center = V::from_components(new_center4.x(), new_center4.y(), new_center4.z());

    let c0 = unsafe { transform.column_unchecked(0) };
    let c1 = unsafe { transform.column_unchecked(1) };
    let c2 = unsafe { transform.column_unchecked(2) };

    let s0 = c0.x() * c0.x() + c0.y() * c0.y() + c0.z() * c0.z();
    let s1 = c1.x() * c1.x() + c1.y() * c1.y() + c1.z() * c1.z();
    let s2 = c2.x() * c2.x() + c2.y() * c2.y() + c2.z() * c2.z();

    let mut max_scale2 = s0;
    if s1 > max_scale2 {
      max_scale2 = s1;
    }
    if s2 > max_scale2 {
      max_scale2 = s2;
    }
    let radius = self.radius() * max_scale2.sqrt();
    Self::new(new_center, radius)
  }
}

impl BS<f32> {
  pub fn transform_f32(&self, transform: &Mat4x4f32) -> Self {
    self.transform::<Vec3f32, Mat4x4f32>(transform)
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OBB<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  /// x_axis | y_axis | z_axis
  _axes: [[S; 3]; 3],
  _origin: [S; 3],
  _half_extents: [S; 3],
}

impl<S> OBB<S>
where
  S: FloatLike + FloatOps + FloatBits,
{
  pub fn translation<V>(&self) -> V
  where
    V: Vector3<Scalar = S>,
  {
    V::from_components(self._origin[0], self._origin[1], self._origin[2])
  }

  pub fn axes<V>(&self) -> [V; 3]
  where
    V: Vector3<Scalar = S>,
  {
    let flat: &[S; 9] = unsafe { &*(self._axes.as_ptr() as *const [S; 9]) };
    [
      V::from_components(flat[0], flat[1], flat[2]),
      V::from_components(flat[3], flat[4], flat[5]),
      V::from_components(flat[6], flat[7], flat[8]),
    ]
  }
  pub fn rotation3<M>(&self) -> M
  where
    M: Matrix3<Scalar = S>,
    M::Vector: Vector3,
  {
    let flat: &[S; 9] = unsafe { &*(self._axes.as_ptr() as *const [S; 9]) };
    M::from_array(flat)
  }

  pub fn half_extent<V>(&self) -> V
  where
    V: Vector3<Scalar = S>,
  {
    V::from_components(
      self._half_extents[0],
      self._half_extents[1],
      self._half_extents[2],
    )
  }

  pub fn half_extents<V>(&self) -> V
  where
    V: Vector3<Scalar = S>,
  {
    self.half_extent()
  }

  pub fn center<V>(&self) -> V
  where
    V: Vector3<Scalar = S>,
  {
    V::from_components(self._origin[0], self._origin[1], self._origin[2])
  }

  pub fn vertices<V>(&self) -> [V; 8]
  where
    V: Vector3<Scalar = S>,
  {
    let center: V = self.translation();
    let ax: [V; 3] = self.axes();
    let x = ax[0] * self._half_extents[0];
    let y = ax[1] * self._half_extents[1];
    let z = ax[2] * self._half_extents[2];

    [
      center - x - y - z,
      center + x - y - z,
      center - x + y - z,
      center + x + y - z,
      center - x - y + z,
      center + x - y + z,
      center - x + y + z,
      center + x + y + z,
    ]
  }

  pub fn contains_aabb<V, M>(&self, other: &AABB<S>) -> bool
  where
    V: Vector3<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3::<M>().transpose();
    let center = self.translation();

    for v in other.vertices::<V>() {
      let local_v = inv_rot.mul_vector(v - center);
      if local_v.x().abs() > self._half_extents[0] + eps
        || local_v.y().abs() > self._half_extents[1] + eps
        || local_v.z().abs() > self._half_extents[2] + eps
      {
        return false;
      }
    }
    true
  }

  pub fn contains_obb<V, M>(&self, other: &Self) -> bool
  where
    V: Vector3<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3::<M>().transpose();
    let center: V = self.translation();

    for v in other.vertices::<V>() {
      let local_v = inv_rot.mul_vector(v - center);
      if local_v.x().abs() > self._half_extents[0] + eps
        || local_v.y().abs() > self._half_extents[1] + eps
        || local_v.z().abs() > self._half_extents[2] + eps
      {
        return false;
      }
    }
    true
  }

  pub fn encapsulate_aabb<V, M>(&mut self, other: &AABB<S>)
  where
    V: Vector3<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    let inv_rot = self.rotation3::<M>().transpose();
    let center: V = self.translation();

    let mut min = V::from_components(
      -self._half_extents[0],
      -self._half_extents[1],
      -self._half_extents[2],
    );
    let mut max = V::from_components(
      self._half_extents[0],
      self._half_extents[1],
      self._half_extents[2],
    );

    for v in other.vertices::<V>() {
      let local_v = inv_rot.mul_vector(v - center);
      min = min.min(local_v);
      max = max.max(local_v);
    }

    let new_local_center = (min + max) * S::from_f32(0.5);
    let new_world_center = center + self.rotation3::<M>().mul_vector(new_local_center);
    let new_half_extents = (max - min) * S::from_f32(0.5);

    self._origin = [
      new_world_center.x(),
      new_world_center.y(),
      new_world_center.z(),
    ];
    self._half_extents = [
      new_half_extents.x(),
      new_half_extents.y(),
      new_half_extents.z(),
    ];
  }

  pub fn encapsulate_obb<V, M>(&mut self, other: &Self)
  where
    V: Vector3<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    let inv_rot = self.rotation3::<M>().transpose();
    let center: V = self.translation();

    let mut min = V::from_components(
      -self._half_extents[0],
      -self._half_extents[1],
      -self._half_extents[2],
    );
    let mut max = V::from_components(
      self._half_extents[0],
      self._half_extents[1],
      self._half_extents[2],
    );

    for v in other.vertices::<V>() {
      let local_v = inv_rot.mul_vector(v - center);
      min = min.min(local_v);
      max = max.max(local_v);
    }

    let new_local_center = (min + max) * S::from_f32(0.5);
    let new_world_center = center + self.rotation3::<M>().mul_vector(new_local_center);
    let new_half_extents = (max - min) * S::from_f32(0.5);

    self._origin = [
      new_world_center.x(),
      new_world_center.y(),
      new_world_center.z(),
    ];
    self._half_extents = [
      new_half_extents.x(),
      new_half_extents.y(),
      new_half_extents.z(),
    ];
  }

  pub fn contains_point<V, M>(&self, p: V) -> bool
  where
    V: Vector3<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3::<M>().transpose();
    let center = self.translation();

    let local_v = inv_rot.mul_vector(p - center);
    local_v.x().abs() <= self._half_extents[0] + eps
      && local_v.y().abs() <= self._half_extents[1] + eps
      && local_v.z().abs() <= self._half_extents[2] + eps
  }

  pub fn new<V, M>(center: V, rot: M, half_extent: V) -> Self
  where
    V: Vector3<Scalar = S> + Into<[S; 3]>,
    M: Matrix3<Scalar = S, Vector = V>,
  {
    debug_assert!(rot.is_pure_rotation_permissive());

    Self {
      _axes: [rot.x().into(), rot.y().into(), rot.z().into()],
      _origin: center.into(),
      _half_extents: half_extent.into(),
    }
  }

  pub fn from_tris<V, M, I>(triangles: I) -> Self
  where
    M: Matrix3<Scalar = S, Vector = V> + From<Mat3f32>,
    V: Vector3<Scalar = S> + From<Vec3f32> + Into<[S; 3]>,
    I: IntoIterator<Item = Triangle>,
    I::IntoIter: Clone,
  {
    let iter0 = triangles.into_iter();
    let iter1 = iter0.clone();
    let iter2 = iter0.clone();

    let (mean_vector, area) = {
      let (sum_vector, area) = iter0.map(|t| (t.mean_vector(), t.area())).fold(
        (Vec3f32::splat(0.0), 0.0f32),
        |acc: (Vec3f32, f32), (mu_k, a_k)| (acc.0 + a_k * mu_k, acc.1 + a_k),
      );
      if area <= 1e-8 {
        (Vec3f32::zero(), 1.0)
      } else {
        (sum_vector / area, area)
      }
    };

    let covariance_matrix = {
      let mut mat = Mat3f32::zero();
      for tri in iter1 {
        let mu_k: Vec3f32 = tri.mean_vector();
        let v0_k: Vec3f32 = (*tri.v0()).into();
        let v1_k: Vec3f32 = (*tri.v1()).into();
        let v2_k: Vec3f32 = (*tri.v2()).into();
        let a_k: f32 = tri.area();
        let factor: f32 = a_k / (12.0 * area);
        let m = 9.0 * Mat3f32::from_outer_self(mu_k)
          + Mat3f32::from_outer_self(v0_k)
          + Mat3f32::from_outer_self(v1_k)
          + Mat3f32::from_outer_self(v2_k);
        mat += factor * m;
      }
      mat -= Mat3f32::from_outer_self(mean_vector);
      mat
    };

    let (_, eigenvectors) = jacobi_diagonalization(covariance_matrix, 1e-9f32, 10);
    let e0 = unsafe { eigenvectors.column_unchecked(0) }.normalize();
    let mut e1 = unsafe { eigenvectors.column_unchecked(1) };
    e1 = (e1 - e0 * e0.dot(e1)).normalize();
    let e2 = e0.cross(e1).normalize();
    let e_s = [e0, e1, e2];

    let mut min = Vec3f32::splat(f32::INFINITY);
    let mut max = Vec3f32::splat(f32::NEG_INFINITY);

    for tri in iter2 {
      let verts: [Vec3f32; 3] = [(*tri.v0()).into(), (*tri.v1()).into(), (*tri.v2()).into()];
      for v_vec in verts {
        for i in 0..3 {
          let p = v_vec.dot(e_s[i]);
          max[i] = p.max(max[i]);
          min[i] = p.min(min[i]);
        }
      }
    }

    let h = (max - min) / 2.0;
    let c = (min[0] + max[0]) / 2.0 * e_s[0]
      + (min[1] + max[1]) / 2.0 * e_s[1]
      + (min[2] + max[2]) / 2.0 * e_s[2];

    let pure_rotation_matrix = Mat3f32::from_columns(e_s[0], e_s[1], e_s[2]);
    let pure_rotation_matrix: M = pure_rotation_matrix.into();
    let c: V = c.into();
    Self::new(c, pure_rotation_matrix, h.into())
  }

  pub fn transform<V, M, M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = S> + MatrixVectorMul,
    M2::Vector: Vector4<Scalar = S>,
    M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
    V: Vector3<Scalar = S> + Into<[S; 3]>,
  {
    let c: V = self.translation();
    let _1 = S::from_f32(1.0);
    let c4 = M2::Vector::from_components(c.x(), c.y(), c.z(), _1);
    let new_c4 = transform.mul_vector(c4);
    let new_c = V::from_components(new_c4.x(), new_c4.y(), new_c4.z());

    let m3: M = self.rotation3();
    let x_axis = m3.x();
    let y_axis = m3.y();
    let z_axis = m3.z();

    let _0 = S::from_f32(0.0);
    let x_axis4 = M2::Vector::from_components(x_axis.x(), x_axis.y(), x_axis.z(), _0);
    let y_axis4 = M2::Vector::from_components(y_axis.x(), y_axis.y(), y_axis.z(), _0);
    let z_axis4 = M2::Vector::from_components(z_axis.x(), z_axis.y(), z_axis.z(), _0);

    let new_x = transform.mul_vector(x_axis4);
    let new_y = transform.mul_vector(y_axis4);
    let new_z = transform.mul_vector(z_axis4);

    let mut nx = V::from_components(new_x.x(), new_x.y(), new_x.z());
    let mut ny = V::from_components(new_y.x(), new_y.y(), new_y.z());
    let mut nz = V::from_components(new_z.x(), new_z.y(), new_z.z());

    let len_x = nx.length();
    let len_y = ny.length();
    let len_z = nz.length();

    nx = nx.normalize();
    ny = (ny - nx * nx.dot(ny)).normalize();
    nz = (nz - nx * nx.dot(nz) - ny * ny.dot(nz)).normalize();

    let new_rot = M::from_columns(nx, ny, nz);
    let new_he = V::from_components(
      self._half_extents[0] * len_x,
      self._half_extents[1] * len_y,
      self._half_extents[2] * len_z,
    );

    Self::new(new_c, new_rot, new_he)
  }

  pub fn to_aabb<V>(&self) -> AABB<S>
  where
    V: Vector3<Scalar = S> + From<[S; 3]> + Into<[S; 3]>,
  {
    let mut min = V::splat(S::from_f32(f32::INFINITY));
    let mut max = V::splat(S::from_f32(f32::NEG_INFINITY));
    for v in self.vertices::<V>() {
      min = min.min(v);
      max = max.max(v);
    }
    AABB::new(min, max)
  }
}

impl OBB<f32> {
  pub fn translation_f32(&self) -> Vec3f32 {
    self.translation()
  }
  pub fn axes_f32(&self) -> [Vec3f32; 3] {
    self.axes()
  }
  pub fn rotation3_f32(&self) -> Mat3f32 {
    self.rotation3()
  }
  pub fn half_extent_f32(&self) -> Vec3f32 {
    self.half_extent()
  }
  pub fn half_extents_f32(&self) -> Vec3f32 {
    self.half_extents()
  }
  pub fn center_f32(&self) -> Vec3f32 {
    self.center()
  }
  pub fn vertices_f32(&self) -> [Vec3f32; 8] {
    self.vertices()
  }
  pub fn contains_aabb_f32(&self, other: &AABB<f32>) -> bool {
    self.contains_aabb::<Vec3f32, Mat3f32>(other)
  }
  pub fn contains_obb_f32(&self, other: &Self) -> bool {
    self.contains_obb::<Vec3f32, Mat3f32>(other)
  }
  pub fn encapsulate_aabb_f32(&mut self, other: &AABB<f32>) {
    self.encapsulate_aabb::<Vec3f32, Mat3f32>(other)
  }
  pub fn encapsulate_obb_f32(&mut self, other: &Self) {
    self.encapsulate_obb::<Vec3f32, Mat3f32>(other)
  }
  pub fn contains_point_f32(&self, p: Vec3f32) -> bool {
    self.contains_point::<_, Mat3f32>(p)
  }
  pub fn transform_f32(&self, transform: &Mat4x4f32) -> Self {
    self.transform::<Vec3f32, Mat3f32, _>(transform)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
  use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

  #[test]
  fn test_aabb_contains() {
    let aabb1 = AABB::new(
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(10.0, 10.0, 10.0),
    );
    let aabb2 = AABB::new(
      Vec3f32::from_components(1.0, 1.0, 1.0),
      Vec3f32::from_components(9.0, 9.0, 9.0),
    );
    let aabb3 = AABB::new(
      Vec3f32::from_components(-1.0, 1.0, 1.0),
      Vec3f32::from_components(9.0, 9.0, 9.0),
    );

    assert!(aabb1.contains_aabb::<Vec3f32>(&aabb2));
    assert!(!aabb1.contains_aabb::<Vec3f32>(&aabb3));
    assert!(aabb1.contains_point(Vec3f32::from_components(5.0, 5.0, 5.0)));
    assert!(!aabb1.contains_point(Vec3f32::from_components(11.0, 5.0, 5.0)));
  }

  #[test]
  fn test_aabb_encapsulate() {
    let mut aabb1 = AABB::new(
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(5.0, 5.0, 5.0),
    );
    let aabb2 = AABB::new(
      Vec3f32::from_components(3.0, 3.0, 3.0),
      Vec3f32::from_components(10.0, 10.0, 10.0),
    );

    aabb1.encapsulate_aabb::<Vec3f32>(&aabb2);
    assert_eq!(aabb1.min::<Vec3f32>().x(), 0.0);
    assert_eq!(aabb1.max::<Vec3f32>().x(), 10.0);
  }

  #[test]
  fn test_obb_contains_point() {
    let obb = OBB::new(
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Mat3f32::identity(),
      Vec3f32::from_components(5.0, 5.0, 5.0),
    );
    assert!(obb.contains_point::<Vec3f32, Mat3f32>(Vec3f32::from_components(4.9, 4.9, 4.9)));
    assert!(!obb.contains_point::<Vec3f32, Mat3f32>(Vec3f32::from_components(5.1, 0.0, 0.0)));
  }

  #[test]
  fn test_bs_transform() {
    use aethervk_oshal_rlib::math::matrix::{Matrix, Matrix4, SquareMatrix};
    let bs = BS::new(Vec3f32::from_components(0.0, 0.0, 0.0), 5.0);
    let mut t = Mat4x4f32::identity();
    // Set scale to 2
    t.x = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(2.0, 0.0, 0.0, 0.0);
    t.y = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 2.0, 0.0, 0.0);
    t.z = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 2.0, 0.0);
    // Translate x
    t.w = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(10.0, 0.0, 0.0, 1.0);

    let transformed = bs.transform_f32(&t);
    assert_eq!(transformed.center::<Vec3f32>().x(), 10.0);
    assert_eq!(transformed.radius(), 10.0);
  }
}
