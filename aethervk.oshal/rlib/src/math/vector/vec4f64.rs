//! vec4f64 module.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::math::{max_two, min_two};

use core::ops;

use crate::math::{
  FloatLike,
  matrix::{Matrix, mat4f64::Mat4x4f64},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3f64::Vec3f64},
};

/// A helper function to easily create our vector for testing
pub fn vec(x: f64, y: f64, z: f64, w: f64) -> Vec4f64 {
  Vec4f64::from_components(x, y, z, w)
}

#[repr(C, align(32))] // vital for proper 256-bit alignment
#[derive(Copy, Clone, Debug)]
pub struct Vec4f64 {
  #[cfg(target_arch = "x86_64")]
  pub simd: __m256d,
  #[cfg(target_arch = "aarch64")]
  pub simd: [float64x2_t; 2],
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  pub data: [f64; 4],
}

pub type DVec4 = Vec4f64;

impl Default for Vec4f64 {
  fn default() -> Self {
    Self::from_components(0.0, 0.0, 0.0, 0.0)
  }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat64(pub Vec4f64);

pub type DQuat = Quat64;

impl Default for Quat64 {
  fn default() -> Self {
    Self::identity()
  }
}

impl Quat64 {
  pub fn from_quat(q: super::vec4::Quat) -> Self {
    Self::from_components(
      q.0.x() as f64,
      q.0.y() as f64,
      q.0.z() as f64,
      q.0.w() as f64,
    )
  }

  pub fn from_components(p0: f64, p1: f64, p2: f64, p3: f64) -> Quat64 {
    Self(Vec4f64::from_components(p0, p1, p2, p3))
  }

  /// Extracts the rotation component from a 4x4 transformation matrix into a Quaternion.
  pub fn from_mat4(m: &Mat4x4f64) -> Self {
    // NOTE: assumes Mat4x4f64 column-major
    #[rustfmt::skip]
        let mut right = unsafe { Vec3f64::from_components(m.column_unchecked(0).x(), m.column_unchecked(0).y(), m.column_unchecked(0).z()) };
    #[rustfmt::skip]
        let mut backward = unsafe { Vec3f64::from_components(m.column_unchecked(1).x(), m.column_unchecked(1).y(), m.column_unchecked(1).z()) };
    #[rustfmt::skip]
        let mut up = unsafe { Vec3f64::from_components(m.column_unchecked(2).x(), m.column_unchecked(2).y(), m.column_unchecked(2).z()) };

    right = right.normalize();
    backward = backward.normalize();
    up = up.normalize();

    let m00 = right.x();
    let m01 = backward.x();
    let m02 = up.x();
    let m10 = right.y();
    let m11 = backward.y();
    let m12 = up.y();
    let m20 = right.z();
    let m21 = backward.z();
    let m22 = up.z();

    let trace = m00 + m11 + m22;

    let _0 = 0.0f64;
    let _1 = 1.0f64;
    let _2 = 2.0f64;
    let _0_25 = 0.25f64;

    if trace > _0 {
      let s = (trace + _1).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f64::from_components(
          (m21 - m12) * inv_s,
          (m02 - m20) * inv_s,
          (m10 - m01) * inv_s,
        ),
        _0_25 * s,
      )
    } else if m00 > m11 && m00 > m22 {
      let s = (_1 + m00 - m11 - m22).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f64::from_components(_0_25 * s, (m01 + m10) * inv_s, (m02 + m20) * inv_s),
        (m21 - m12) * inv_s,
      )
    } else if m11 > m22 {
      let s = (_1 + m11 - m00 - m22).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f64::from_components((m01 + m10) * inv_s, _0_25 * s, (m12 + m21) * inv_s),
        (m02 - m20) * inv_s,
      )
    } else {
      let s = (_1 + m22 - m00 - m11).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f64::from_components((m02 + m20) * inv_s, (m12 + m21) * inv_s, _0_25 * s),
        (m10 - m01) * inv_s,
      )
    }
  }
}

impl Into<[f64; 4]> for Vec4f64 {
  #[inline]
  fn into(self) -> [f64; 4] {
    let mut result: [f64; 4] = [0.0, 0.0, 0.0, 0.0];
    #[cfg(target_arch = "x86_64")]
    unsafe {
      _mm256_storeu_pd(result.as_mut_ptr(), self.simd);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      vst1q_f64(result.as_mut_ptr(), self.simd[0]);
      vst1q_f64(result.as_mut_ptr().add(2), self.simd[1]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      result = self.data;
    }

    result
  }
}

impl PartialEq for Vec4f64 {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let cmp = _mm256_cmp_pd(self.simd, other.simd, _CMP_EQ_OQ);
      _mm256_movemask_pd(cmp) == 0xF
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let cmp0 = vceqq_f64(self.simd[0], other.simd[0]);
      let cmp1 = vceqq_f64(self.simd[1], other.simd[1]);
      vminvq_u32(vreinterpretq_u32_u64(cmp0)) != 0 && vminvq_u32(vreinterpretq_u32_u64(cmp1)) != 0
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.data == other.data
    }
  }
}

impl ops::Add for Vec4f64 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_add_pd(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vaddq_f64(self.simd[0], rhs.simd[0]),
          vaddq_f64(self.simd[1], rhs.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] + rhs.data[0],
          self.data[1] + rhs.data[1],
          self.data[2] + rhs.data[2],
          self.data[3] + rhs.data[3],
        ],
      }
    }
  }
}

impl ops::Sub for Vec4f64 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_sub_pd(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vsubq_f64(self.simd[0], rhs.simd[0]),
          vsubq_f64(self.simd[1], rhs.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] - rhs.data[0],
          self.data[1] - rhs.data[1],
          self.data[2] - rhs.data[2],
          self.data[3] - rhs.data[3],
        ],
      }
    }
  }
}

impl ops::Mul<f64> for Vec4f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f64) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_mul_pd(self.simd, _mm256_set1_pd(rhs)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let vrhs = vdupq_n_f64(rhs);
      Self {
        simd: [vmulq_f64(self.simd[0], vrhs), vmulq_f64(self.simd[1], vrhs)],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] * rhs,
          self.data[1] * rhs,
          self.data[2] * rhs,
          self.data[3] * rhs,
        ],
      }
    }
  }
}

impl ops::Mul<Vec4f64> for f64 {
  type Output = Vec4f64;
  #[inline]
  fn mul(self, rhs: Vec4f64) -> Vec4f64 {
    rhs * self
  }
}

impl ops::Mul<Vec4f64> for Vec4f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_mul_pd(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vmulq_f64(self.simd[0], rhs.simd[0]),
          vmulq_f64(self.simd[1], rhs.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
      Self {
        data: [
          self.data[0] * rhs.data[0],
          self.data[1] * rhs.data[1],
          self.data[2] * rhs.data[2],
          self.data[3] * rhs.data[3],
        ],
      }
    }
  }
}

impl ops::Div<f64> for Vec4f64 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f64) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_div_pd(self.simd, _mm256_set1_pd(rhs)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let vrhs = vdupq_n_f64(rhs);
      Self {
        simd: [vdivq_f64(self.simd[0], vrhs), vdivq_f64(self.simd[1], vrhs)],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] / rhs,
          self.data[1] / rhs,
          self.data[2] / rhs,
          self.data[3] / rhs,
        ],
      }
    }
  }
}

impl ops::Div<Vec4f64> for Vec4f64 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_div_pd(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vdivq_f64(self.simd[0], rhs.simd[0]),
          vdivq_f64(self.simd[1], rhs.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] / rhs.data[0],
          self.data[1] / rhs.data[1],
          self.data[2] / rhs.data[2],
          self.data[3] / rhs.data[3],
        ],
      }
    }
  }
}

impl ops::AddAssign<Self> for Vec4f64 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl ops::SubAssign<Self> for Vec4f64 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl ops::MulAssign<Self> for Vec4f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs;
  }
}

impl ops::MulAssign<f64> for Vec4f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: f64) {
    *self = *self * rhs;
  }
}

impl ops::DivAssign<Self> for Vec4f64 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs;
  }
}

impl ops::DivAssign<f64> for Vec4f64 {
  #[inline]
  fn div_assign(&mut self, rhs: f64) {
    *self = *self / rhs;
  }
}

impl ops::Neg for Vec4f64 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_xor_pd(self.simd, _mm256_set1_pd(-0.0f64)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let neg_mask = vdupq_n_u64(0x8000_0000_0000_0000);
      Self {
        simd: [
          vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(self.simd[0]), neg_mask)),
          vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(self.simd[1]), neg_mask)),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [-self.data[0], -self.data[1], -self.data[2], -self.data[3]],
      }
    }
  }
}

impl Vector for Vec4f64 {
  type Scalar = f64;
  const DIM: usize = 4;
  #[inline]
  fn zero() -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_setzero_pd(),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [vdupq_n_f64(0f64), vdupq_n_f64(0f64)],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self { data: [0f64; 4] }
    }
  }
  #[inline]
  fn splat(v: Self::Scalar) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_set1_pd(v),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [vdupq_n_f64(v), vdupq_n_f64(v)],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self { data: [v; 4] }
    }
  }
  #[inline]
  fn component(&self, i: usize) -> Option<Self::Scalar> {
    if i < Self::DIM {
      Some(unsafe { self.component_unchecked(i) })
    } else {
      None
    }
  }
  #[inline]
  unsafe fn component_unchecked(&self, i: usize) -> Self::Scalar {
    let ptr = self as *const Self as *const f64;
    unsafe { *ptr.add(i) }
  }
  #[inline]
  fn set_component(&mut self, i: usize, value: Self::Scalar) {
    if i < Self::DIM {
      let ptr = self as *mut Self as *mut f64;
      unsafe { *ptr.add(i) = value };
    }
  }
  #[inline]
  fn dot(self, rhs: Self) -> Self::Scalar {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let mul = _mm256_mul_pd(self.simd, rhs.simd);
      let mut arr: [f64; 4] = [0.0; 4];
      _mm256_storeu_pd(arr.as_mut_ptr(), mul);
      arr[0] + arr[1] + arr[2] + arr[3]
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let m0 = vmulq_f64(self.simd[0], rhs.simd[0]);
      let m1 = vmulq_f64(self.simd[1], rhs.simd[1]);
      vaddvq_f64(m0) + vaddvq_f64(m1)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.data[0] * rhs.data[0]
        + self.data[1] * rhs.data[1]
        + self.data[2] * rhs.data[2]
        + self.data[3] * rhs.data[3]
    }
  }
  #[inline]
  fn min(self, other: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_min_pd(self.simd, other.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vminq_f64(self.simd[0], other.simd[0]),
          vminq_f64(self.simd[1], other.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          min_two(self.data[0], other.data[0]),
          min_two(self.data[1], other.data[1]),
          min_two(self.data[2], other.data[2]),
          min_two(self.data[3], other.data[3]),
        ],
      }
    }
  }
  #[inline]
  fn max(self, other: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_max_pd(self.simd, other.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [
          vmaxq_f64(self.simd[0], other.simd[0]),
          vmaxq_f64(self.simd[1], other.simd[1]),
        ],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          max_two(self.data[0], other.data[0]),
          max_two(self.data[1], other.data[1]),
          max_two(self.data[2], other.data[2]),
          max_two(self.data[3], other.data[3]),
        ],
      }
    }
  }
}

impl Vector4 for Vec4f64 {
  #[inline]
  fn from_components(x: f64, y: f64, z: f64, w: f64) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm256_set_pd(w, z, y, x),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: [vld1q_f64([x, y].as_ptr()), vld1q_f64([z, w].as_ptr())],
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self { data: [x, y, z, w] }
    }
  }
  #[inline]
  fn x(&self) -> f64 {
    unsafe { self.component_unchecked(0) }
  }
  #[inline]
  fn y(&self) -> f64 {
    unsafe { self.component_unchecked(1) }
  }
  #[inline]
  fn z(&self) -> f64 {
    unsafe { self.component_unchecked(2) }
  }
  #[inline]
  fn w(&self) -> f64 {
    unsafe { self.component_unchecked(3) }
  }
}

impl ops::Index<usize> for Vec4f64 {
  type Output = f64;
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 4);
    let ptr = self as *const Self as *const f64;
    unsafe { ptr.add(index).as_ref().unwrap_unchecked() }
  }
}

impl ops::IndexMut<usize> for Vec4f64 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 4);
    let ptr = self as *mut Self as *mut f64;
    unsafe { ptr.add(index).as_mut().unwrap_unchecked() }
  }
}

impl ops::Index<usize> for Quat64 {
  type Output = f64;
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    &self.0[index]
  }
}

impl ops::IndexMut<usize> for Quat64 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.0[index]
  }
}

impl ops::Mul for Quat64 {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self::Output {
    let (x1, y1, z1, w1) = (self.0.x(), self.0.y(), self.0.z(), self.0.w());
    let (x2, y2, z2, w2) = (rhs.0.x(), rhs.0.y(), rhs.0.z(), rhs.0.w());

    Quat64(Vec4f64::from_components(
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))
  }
}

impl ops::Add for Quat64 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self(self.0 + rhs.0)
  }
}

impl ops::Sub for Quat64 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self(self.0 - rhs.0)
  }
}

impl ops::Neg for Quat64 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    Self(-self.0)
  }
}

impl ops::Mul<f64> for Quat64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f64) -> Self::Output {
    Self(self.0 * rhs)
  }
}

impl ops::Mul<Quat64> for f64 {
  type Output = Quat64;
  #[inline]
  fn mul(self, rhs: Quat64) -> Self::Output {
    Quat64(self * rhs.0)
  }
}

impl ops::Div<f64> for Quat64 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f64) -> Self::Output {
    Self(self.0 / rhs)
  }
}

impl Quaternion for Quat64 {
  type Scalar = f64;
  type Vector = Vec3f64;

  #[inline]
  fn identity() -> Self {
    Self(Vec4f64::from_components(0.0, 0.0, 0.0, 1.0))
  }

  #[inline]
  fn from_vector_and_scalar(vector: Vec3f64, scalar: f64) -> Self {
    Self(Vec4f64::from_components(
      vector.x(),
      vector.y(),
      vector.z(),
      scalar,
    ))
  }

  #[inline]
  fn vector_part(self) -> Vec3f64 {
    Vec3f64::from_components(self.0.x(), self.0.y(), self.0.z())
  }

  #[inline]
  fn scalar_part(self) -> f64 {
    self.0.w()
  }

  #[inline]
  fn conjugate(self) -> Self {
    Self(Vec4f64::from_components(
      -self.0.x(),
      -self.0.y(),
      -self.0.z(),
      self.0.w(),
    ))
  }

  #[inline]
  fn norm_squared(self) -> f64 {
    self.0.dot(self.0)
  }

  #[inline]
  fn norm(self) -> f64 {
    crate::math::FloatLike::sqrt(self.norm_squared())
  }

  #[inline]
  fn normalize(self) -> Self {
    let n = self.norm();
    if n > 0.0 { self / n } else { Self::identity() }
  }

  #[inline]
  fn inverse(self) -> Self {
    let n_sq = self.norm_squared();
    if n_sq > 0.0 {
      self.conjugate() / n_sq
    } else {
      Self::identity()
    }
  }
}
