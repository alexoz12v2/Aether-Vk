//! Bounds Module
//! This module will contain all the necessary math algorithms to construct and manage bounds summary
//! descriptions of polygonal meshes, in particular
//! - Axis Aligned Bounding Boxes
//! - Object Aligned Bounding Boxes
//! - Spherical Bounds

use core::{f32, ops};
use aethervk_oshal_rlib::{
  self as oshal,
  math::{
    FloatLike, Scalar,
    floating::{FloatBits, FloatOps},
    matrix::{
      Matrix, Matrix3, Matrix4, MatrixVectorMul, mat3::Mat3f32, mat4::Mat4x4f32 as Mat4f32,
    },
    vector::{Vector, Vector3, Vector4},
  },
};
use oshal::math::vector::vec3::Vec3f32;

use crate::{math::qr_diagonalization, simulation::comet::Triangle};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AABB<V>
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  min: [V::Scalar; 3],
  max: [V::Scalar; 3],
}

impl<V> AABB<V>
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  #[inline]
  pub fn vertices(&self) -> [V; 8] {
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
      [0, 1], [2, 3], [4, 5], [6, 7],
      [0, 2], [1, 3], [4, 6], [5, 7],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ]
  }

  pub fn new(min: V, max: V) -> Self {
    Self {
      min: min.into(),
      max: max.into(),
    }
  }

  pub fn min(&self) -> V {
    self.min.into()
  }

  pub fn max(&self) -> V {
    self.max.into()
  }

  pub fn center(&self) -> V {
    (self.min() + self.max()) * V::Scalar::from_f32(0.5)
  }

  pub fn half_extents(&self) -> V {
    (self.max() - self.min()) * V::Scalar::from_f32(0.5)
  }

  pub fn contains_aabb(&self, other: &Self) -> bool {
    let s_min = self.min();
    let s_max = self.max();
    let o_min = other.min();
    let o_max = other.max();
    let eps = V::Scalar::from_f32(1e-4);

    o_min.x() >= s_min.x() - eps && o_max.x() <= s_max.x() + eps &&
    o_min.y() >= s_min.y() - eps && o_max.y() <= s_max.y() + eps &&
    o_min.z() >= s_min.z() - eps && o_max.z() <= s_max.z() + eps
  }

  pub fn contains_obb<M>(&self, other: &OBB<V::Scalar, V, M>) -> bool 
  where
    M: Matrix3<Scalar = V::Scalar, Vector = V> + MatrixVectorMul,
    V::Scalar: FloatLike + FloatOps + FloatBits,
  {
    let s_min = self.min();
    let s_max = self.max();
    let eps = V::Scalar::from_f32(1e-4);
    
    for v in other.vertices() {
      if v.x() < s_min.x() - eps || v.x() > s_max.x() + eps ||
         v.y() < s_min.y() - eps || v.y() > s_max.y() + eps ||
         v.z() < s_min.z() - eps || v.z() > s_max.z() + eps {
        return false;
      }
    }
    true
  }

  pub fn encapsulate_aabb(&mut self, other: &Self) {
    let mut s_min = self.min();
    let mut s_max = self.max();
    let o_min = other.min();
    let o_max = other.max();

    s_min = s_min.min(o_min);
    s_max = s_max.max(o_max);

    self.min = [s_min.x(), s_min.y(), s_min.z()];
    self.max = [s_max.x(), s_max.y(), s_max.z()];
  }

  pub fn encapsulate_obb<M>(&mut self, other: &OBB<V::Scalar, V, M>) 
  where
    M: Matrix3<Scalar = V::Scalar, Vector = V> + MatrixVectorMul,
    V::Scalar: FloatLike + FloatOps + FloatBits,
  {
    let mut s_min = self.min();
    let mut s_max = self.max();

    for v in other.vertices() {
      s_min = s_min.min(v);
      s_max = s_max.max(v);
    }

    self.min = [s_min.x(), s_min.y(), s_min.z()];
    self.max = [s_max.x(), s_max.y(), s_max.z()];
  }

  pub fn contains_point(&self, p: V) -> bool {
    let s_min = self.min();
    let s_max = self.max();
    let eps = V::Scalar::from_f32(1e-4);

    p.x() >= s_min.x() - eps && p.x() <= s_max.x() + eps &&
    p.y() >= s_min.y() - eps && p.y() <= s_max.y() + eps &&
    p.z() >= s_min.z() - eps && p.z() <= s_max.z() + eps
  }

  pub fn from_tris<I>(triangles: I) -> Self
  where
    I: IntoIterator<Item = Triangle>,
    I::IntoIter: Clone,
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

  pub fn transform<M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = V::Scalar> + MatrixVectorMul,
    M2::Vector: Vector4<Scalar = V::Scalar>,
  {
    let min = self.min();
    let max = self.max();

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

    let _1 = V::Scalar::from_f32(1.0);
    let mut new_min = V::splat(V::Scalar::from_f32(f32::INFINITY));
    let mut new_max = V::splat(V::Scalar::from_f32(-f32::INFINITY));

    for c in corners.iter() {
      let v4 = M2::Vector::from_components(c.x(), c.y(), c.z(), _1);
      let transformed = transform.mul_vector(v4);
      let tv = V::from_components(transformed.x(), transformed.y(), transformed.z());
      new_min = new_min.min(tv);
      new_max = new_max.max(tv);
    }

    Self::new(new_min, new_max)
  }
}

impl<V> AABB<V>
where
  V: Vector3<Scalar = f32> + From<Vec3f32> + From<[f32; 3]> + Into<[f32; 3]>,
{
  pub fn transform_f32(&self, transform: &Mat4f32) -> Self {
    self.transform(transform)
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BS<V>
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  center_radius: [V::Scalar; 4],
}

impl<V> BS<V>
where
  V: Vector3 + From<Vec3f32> + From<[V::Scalar; 3]> + Into<[V::Scalar; 3]>,
  V::Scalar: FloatLike,
{
  pub fn center(&self) -> V {
    V::from_components(
      self.center_radius[0],
      self.center_radius[1],
      self.center_radius[2],
    )
  }

  pub fn radius(&self) -> V::Scalar {
    self.center_radius[3]
  }

  pub fn new(center: V, radius: V::Scalar) -> Self {
    Self {
      center_radius: [center.x(), center.y(), center.z(), radius],
    }
  }

  pub fn from_tris<I>(triangles: I) -> Self
  where
    I: IntoIterator<Item = Triangle>,
    I::IntoIter: Clone,
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
        if V::Scalar::from_f32(v.x()) > max_x.x() { max_x = v.into(); }
        if V::Scalar::from_f32(v.x()) < min_x.x() { min_x = v.into(); }
        if V::Scalar::from_f32(v.y()) > max_y.y() { max_y = v.into(); }
        if V::Scalar::from_f32(v.y()) < min_y.y() { min_y = v.into(); }
        if V::Scalar::from_f32(v.z()) > max_z.z() { max_z = v.into(); }
        if V::Scalar::from_f32(v.z()) < min_z.z() { min_z = v.into(); }
      }
    }

    if count == 0 { return Self::new(V::zero(), V::Scalar::from_f32(0.0)); }

    let dist2_x = (max_x - min_x).length_squared();
    let dist2_y = (max_y - min_y).length_squared();
    let dist2_z = (max_z - min_z).length_squared();

    let mut max_dist2 = dist2_x;
    let mut p1 = min_x;
    let mut p2 = max_x;

    if dist2_y > max_dist2 { max_dist2 = dist2_y; p1 = min_y; p2 = max_y; }
    if dist2_z > max_dist2 { max_dist2 = dist2_z; p1 = min_z; p2 = max_z; }

    let _0_5 = V::Scalar::from_f32(0.5);
    let mut center = (p1 + p2) * _0_5;
    let mut radius = max_dist2.sqrt() * _0_5;
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

  pub fn transform<M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = V::Scalar> + MatrixVectorMul,
    M2::Vector: Vector4<Scalar = V::Scalar>,
  {
    let center = self.center();
    let _1 = V::Scalar::from_f32(1.0);
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
    if s1 > max_scale2 { max_scale2 = s1; }
    if s2 > max_scale2 { max_scale2 = s2; }
    let radius = self.radius() * max_scale2.sqrt();
    Self::new(new_center, radius)
  }
}

impl<V> BS<V>
where
  V: Vector3<Scalar = f32> + From<Vec3f32> + From<[f32; 3]> + Into<[f32; 3]>,
{
  pub fn transform_f32(&self, transform: &Mat4f32) -> Self {
    self.transform(transform)
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OBB<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike + FloatOps,
{
  /// x_axis | y_axis | z_axis
  _axes: [[S; 3]; 3],
  _origin: [S; 3],
  _half_extents: [S; 3],

  _phantom: core::marker::PhantomData<M>,
}

impl<S, V, M> OBB<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V>,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike + FloatOps,
{
  pub fn translation(&self) -> V {
    V::from_components(self._origin[0], self._origin[1], self._origin[2])
  }

  pub fn axes(&self) -> [V; 3] {
    let flat: &[S; 9] = unsafe { &*(self._axes.as_ptr() as *const [S; 9]) };
    [
      V::from_components(flat[0], flat[1], flat[2]),
      V::from_components(flat[3], flat[4], flat[5]),
      V::from_components(flat[6], flat[7], flat[8]),
    ]
  }

  pub fn rotation3(&self) -> M {
    let flat: &[S; 9] = unsafe { &*(self._axes.as_ptr() as *const [S; 9]) };
    M::from_array(flat)
  }

  pub fn half_extent(&self) -> V {
    V::from_components(self._half_extents[0], self._half_extents[1], self._half_extents[2])
  }

  pub fn half_extents(&self) -> V {
    self.half_extent()
  }

  pub fn center(&self) -> V {
    V::from_components(self._origin[0], self._origin[1], self._origin[2])
  }

  pub fn vertices(&self) -> [V; 8] {
    let center = self.translation();
    let ax = self.axes();
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
}

impl<S, V, M> OBB<S, V, M>
where
  M: Matrix3<Scalar = S, Vector = V> + MatrixVectorMul,
  V: Vector3<Scalar = S> + From<Vec3f32> + From<[S; 3]> + Into<[S; 3]>,
  S: FloatLike + FloatOps + FloatBits,
{
  pub fn contains_aabb(&self, other: &AABB<V>) -> bool {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3().transpose();
    let center = self.translation();

    for v in other.vertices() {
      let local_v = inv_rot.mul_vector(v - center);
      if local_v.x().abs() > self._half_extents[0] + eps ||
         local_v.y().abs() > self._half_extents[1] + eps ||
         local_v.z().abs() > self._half_extents[2] + eps {
        return false;
      }
    }
    true
  }

  pub fn contains_obb(&self, other: &Self) -> bool {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3().transpose();
    let center = self.translation();

    for v in other.vertices() {
      let local_v = inv_rot.mul_vector(v - center);
      if local_v.x().abs() > self._half_extents[0] + eps ||
         local_v.y().abs() > self._half_extents[1] + eps ||
         local_v.z().abs() > self._half_extents[2] + eps {
        return false;
      }
    }
    true
  }

  pub fn encapsulate_aabb(&mut self, other: &AABB<V>) {
    let inv_rot = self.rotation3().transpose();
    let center = self.translation();

    let mut min = V::from_components(-self._half_extents[0], -self._half_extents[1], -self._half_extents[2]);
    let mut max = V::from_components(self._half_extents[0], self._half_extents[1], self._half_extents[2]);

    for v in other.vertices() {
      let local_v = inv_rot.mul_vector(v - center);
      min = min.min(local_v);
      max = max.max(local_v);
    }

    let new_local_center = (min + max) * S::from_f32(0.5);
    let new_world_center = center + self.rotation3().mul_vector(new_local_center);
    let new_half_extents = (max - min) * S::from_f32(0.5);

    self._origin = [new_world_center.x(), new_world_center.y(), new_world_center.z()];
    self._half_extents = [new_half_extents.x(), new_half_extents.y(), new_half_extents.z()];
  }

  pub fn encapsulate_obb(&mut self, other: &Self) {
    let inv_rot = self.rotation3().transpose();
    let center = self.translation();

    let mut min = V::from_components(-self._half_extents[0], -self._half_extents[1], -self._half_extents[2]);
    let mut max = V::from_components(self._half_extents[0], self._half_extents[1], self._half_extents[2]);

    for v in other.vertices() {
      let local_v = inv_rot.mul_vector(v - center);
      min = min.min(local_v);
      max = max.max(local_v);
    }

    let new_local_center = (min + max) * S::from_f32(0.5);
    let new_world_center = center + self.rotation3().mul_vector(new_local_center);
    let new_half_extents = (max - min) * S::from_f32(0.5);

    self._origin = [new_world_center.x(), new_world_center.y(), new_world_center.z()];
    self._half_extents = [new_half_extents.x(), new_half_extents.y(), new_half_extents.z()];
  }

  pub fn contains_point(&self, p: V) -> bool {
    let eps = S::from_f32(1e-4);
    let inv_rot = self.rotation3().transpose();
    let center = self.translation();

    let local_v = inv_rot.mul_vector(p - center);
    local_v.x().abs() <= self._half_extents[0] + eps &&
    local_v.y().abs() <= self._half_extents[1] + eps &&
    local_v.z().abs() <= self._half_extents[2] + eps
  }

  pub fn new(center: V, rot: M, half_extent: V) -> Self {
    debug_assert!(rot.is_pure_rotation_permissive());

    Self {
      _axes: [rot.x().into(), rot.y().into(), rot.z().into()],
      _origin: center.into(),
      _half_extents: half_extent.into(),
      _phantom: core::marker::PhantomData,
    }
  }

  pub fn from_tris<I>(triangles: I) -> Self
  where
    M: From<Mat3f32>,
    V: Into<Vec3f32>,
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
      if area <= 1e-8 { (Vec3f32::zero(), 1.0) } else { (sum_vector / area, area) }
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

    let (_, eigenvectors) = qr_diagonalization(covariance_matrix, 1e-9f32, 10);
    let e0 = unsafe { eigenvectors.column_unchecked(0) }.normalize();
    let e1 = unsafe { eigenvectors.column_unchecked(1) }.normalize();
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
    Self::new(c.into(), pure_rotation_matrix.into(), h.into())
  }

  pub fn transform<M2>(&self, transform: &M2) -> Self
  where
    M2: Matrix4<Scalar = S> + MatrixVectorMul,
    M2::Vector: Vector4<Scalar = S>,
  {
    let c = self.translation();
    let _1 = S::from_f32(1.0);
    let c4 = M2::Vector::from_components(c.x(), c.y(), c.z(), _1);
    let new_c4 = transform.mul_vector(c4);
    let new_c = V::from_components(new_c4.x(), new_c4.y(), new_c4.z());

    let m3 = self.rotation3();
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
    let new_he = V::from_components(self._half_extents[0] * len_x, self._half_extents[1] * len_y, self._half_extents[2] * len_z);

    Self::new(new_c, new_rot, new_he)
  }
}

impl<V, M> OBB<f32, V, M>
where
  M: Matrix3<Scalar = f32, Vector = V> + MatrixVectorMul,
  V: Vector3<Scalar = f32> + From<Vec3f32> + From<[f32; 3]> + Into<[f32; 3]>,
{
  pub fn transform_f32(&self, transform: &Mat4f32) -> Self {
    self.transform(transform)
  }
}
