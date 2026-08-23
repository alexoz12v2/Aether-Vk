//! vec4 module.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::math::{max_two, min_two};

use core::ops;

use crate::math::{
  FloatLike,
  matrix::{Matrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32},
};

// A helper function to easily create our vector for testing
/// TODO: Document this item
pub fn vec(x: f32, y: f32, z: f32, w: f32) -> Vec4f32 {
  Vec4f32::from_components(x, y, z, w)
}

#[repr(C, align(16))] // vital for proper alignment
#[derive(Copy, Clone, Debug)]
/// TODO: Document this item
pub struct Vec4f32 {
  #[cfg(target_arch = "x86_64")]
  pub simd: __m128,
  #[cfg(target_arch = "aarch64")]
  pub simd: float32x4_t,
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  pub data: [f32; 4], // alignment to 16 like other branches
}

impl Default for Vec4f32 {
  fn default() -> Self {
    Self::from_components(0.0, 0.0, 0.0, 0.0)
  }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
/// TODO: Document this item
pub struct Quat(pub Vec4f32);

impl Default for Quat {
  fn default() -> Self {
    Self::identity()
  }
}

impl Quat {
  /// TODO: Document this item
  pub fn from_components(p0: f32, p1: f32, p2: f32, p3: f32) -> Quat {
    Self(Vec4f32::from_components(p0, p1, p2, p3))
  }

  /// Extracts the rotation component from a 4x4 transformation matrix into a Quaternion.
  /// This assumes the matrix is standard column-major, and normalizes the axes
  /// to strip out any scaling factors before conversion.
  /// Coordinate system context: +X=Right, +Y=Backward (-Y=Forward), +Z=Up.
  pub fn from_mat4(m: &Mat4x4f32) -> Self {
    // NOTE: assumes Mat4x4f32 column-major

    // Extract the upper 3x3 block (the basis vectors: Right, Backward, Up)
    #[rustfmt::skip]
    let mut right = unsafe { Vec3f32::from_components(m.column_unchecked(0).x(), m.column_unchecked(0).y(), m.column_unchecked(0).z()) };
    #[rustfmt::skip]
    let mut backward = unsafe { Vec3f32::from_components(m.column_unchecked(1).x(), m.column_unchecked(1).y(), m.column_unchecked(1).z()) };
    #[rustfmt::skip]
    let mut up = unsafe { Vec3f32::from_components(m.column_unchecked(2).x(), m.column_unchecked(2).y(), m.column_unchecked(2).z()) };

    // Strip scale to ensure we generate a valid unit quaternion
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

    let _0 = 0.0f32;
    let _1 = 1.0f32;
    let _2 = 2.0f32;
    let _0_25 = 0.25f32;

    // Use the standard robust trace method (identical logic to your 3x3 trait method)
    if trace > _0 {
      let s = (trace + _1).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f32::from_components(
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
        Vec3f32::from_components(_0_25 * s, (m01 + m10) * inv_s, (m02 + m20) * inv_s),
        (m21 - m12) * inv_s,
      )
    } else if m11 > m22 {
      let s = (_1 + m11 - m00 - m22).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f32::from_components((m01 + m10) * inv_s, _0_25 * s, (m12 + m21) * inv_s),
        (m02 - m20) * inv_s,
      )
    } else {
      let s = (_1 + m22 - m00 - m11).sqrt() * _2;
      let inv_s = _1 / s;
      Self::from_vector_and_scalar(
        Vec3f32::from_components((m02 + m20) * inv_s, (m12 + m21) * inv_s, _0_25 * s),
        (m10 - m01) * inv_s,
      )
    }
  }
}

impl Into<[f32; 4]> for Vec4f32 {
  #[inline]
  fn into(self) -> [f32; 4] {
    let mut result: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    #[cfg(target_arch = "x86_64")]
    {
      unsafe {
        _mm_storeu_ps(result.as_mut_ptr(), self.simd);
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      unsafe {
        vst1q_f32(result.as_mut_ptr(), self.simd);
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      result = self.data;
    }

    result
  }
}

impl PartialEq for Vec4f32 {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let cmp = _mm_cmpeq_ps(self.simd, other.simd);
      _mm_movemask_ps(cmp) == 0xF
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let cmp = vceqq_f32(self.simd, other.simd);
      vminvq_u32(cmp) != 0
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.data == other.data
    }
  }
}

impl ops::Add for Vec4f32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_add_ps(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vaddq_f32(self.simd, rhs.simd),
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

impl ops::Sub for Vec4f32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_sub_ps(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vsubq_f32(self.simd, rhs.simd),
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

impl ops::Mul<f32> for Vec4f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_mul_ps(self.simd, _mm_set1_ps(rhs)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vmulq_n_f32(self.simd, rhs),
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

impl ops::Mul<Vec4f32> for f32 {
  type Output = Vec4f32;
  #[inline]
  fn mul(self, rhs: Vec4f32) -> Vec4f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Vec4f32 {
        simd: _mm_mul_ps(_mm_set1_ps(self), rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Vec4f32 {
        simd: vmulq_n_f32(rhs.simd, self),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Vec4f32 {
        data: [
          self * rhs.data[0],
          self * rhs.data[1],
          self * rhs.data[2],
          self * rhs.data[3],
        ],
      }
    }
  }
}

impl ops::Mul<Vec4f32> for Vec4f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_mul_ps(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vmulq_f32(self.simd, rhs.simd),
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

impl ops::Div<f32> for Vec4f32 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f32) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_div_ps(self.simd, _mm_set1_ps(rhs)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vdivq_f32(self.simd, vdupq_n_f32(rhs)),
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

impl ops::Div<Vec4f32> for Vec4f32 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_div_ps(self.simd, rhs.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vdivq_f32(self.simd, rhs.simd),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          self.data[0] / rhs.simd[0],
          self.data[1] / rhs.simd[1],
          self.data[2] / rhs.simd[2],
          self.data[3] / rhs.simd[3],
        ],
      }
    }
  }
}

impl ops::AddAssign<Self> for Vec4f32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl ops::SubAssign<Self> for Vec4f32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl ops::MulAssign<Self> for Vec4f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs;
  }
}

impl ops::MulAssign<f32> for Vec4f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: f32) {
    *self = *self * rhs;
  }
}

impl ops::DivAssign<Self> for Vec4f32 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs;
  }
}

impl ops::DivAssign<f32> for Vec4f32 {
  #[inline]
  fn div_assign(&mut self, rhs: f32) {
    *self = *self / rhs;
  }
}

impl ops::Neg for Vec4f32 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self {
    #[cfg(target_arch = "x86_64")] // XOR with minus zero
    unsafe {
      Self {
        simd: _mm_xor_ps(self.simd, _mm_set1_ps(-0.0f32)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vreinterpretq_f32_u32(veorq_u32(
          vreinterpretq_u32_f32(self.simd),
          vdupq_n_u32(0x8000_0000),
        )),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [
          -1 * self.data[0],
          -1 * self.data[1],
          -1 * self.data[2],
          -1 * self.data[3],
        ],
      }
    }
  }
}

impl Vector for Vec4f32 {
  type Scalar = f32;
  const DIM: usize = 4;
  #[inline]
  fn zero() -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_setzero_ps(),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vdupq_n_f32(0f32),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        data: [0f32, 0f32, 0f32, 0f32],
      }
    }
  }
  #[inline]
  fn splat(v: Self::Scalar) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_set1_ps(v),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vdupq_n_f32(v),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self { data: [v, v, v, v] }
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
    // casting to first member is allowed because of repr(C)
    let ptr = self as *const Self as *const f32;
    unsafe { *ptr.add(i) }
  }
  #[inline]
  fn set_component(&mut self, i: usize, value: Self::Scalar) {
    if i < Self::DIM {
      let ptr = self as *mut Self as *mut f32;
      unsafe { *ptr.add(i) = value };
    }
  }
  #[inline]
  fn dot(self, rhs: Self) -> Self::Scalar {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // Note: dp instruction family is part of SSE4.1
      let res = _mm_dp_ps(self.simd, rhs.simd, 0xFF);
      _mm_cvtss_f32(res)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      // Note: vaddvq is available only on ARMv8.1+ NEON
      vaddvq_f32(vmulq_f32(self.simd, rhs.simd))
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
        simd: _mm_min_ps(self.simd, other.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vminq_f32(self.simd, other.simd),
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
        simd: _mm_max_ps(self.simd, other.simd),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vmaxq_f32(self.simd, other.simd),
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

impl Vector4 for Vec4f32 {
  #[inline]
  fn from_components(x: f32, y: f32, z: f32, w: f32) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      Self {
        simd: _mm_set_ps(w, z, y, x),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self {
        simd: vld1q_f32([x, y, z, w].as_ptr()),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self { data: [x, y, z, w] }
    }
  }
  #[inline]
  fn x(&self) -> f32 {
    unsafe { self.component_unchecked(0) }
  }
  #[inline]
  fn y(&self) -> f32 {
    unsafe { self.component_unchecked(1) }
  }
  #[inline]
  fn z(&self) -> f32 {
    unsafe { self.component_unchecked(2) }
  }
  #[inline]
  fn w(&self) -> f32 {
    unsafe { self.component_unchecked(3) }
  }
}

impl ops::Index<usize> for Vec4f32 {
  type Output = f32;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 4);
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let ptr = &self.simd as *const __m128 as *const f32;
      ptr.add(index).as_ref().unwrap_unchecked()
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let ptr = &self.simd as *const float32x4_t as *const f32;
      ptr.add(index).as_ref().unwrap_unchecked()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      &self.data[index]
    }
  }
}

impl ops::IndexMut<usize> for Vec4f32 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 4);
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let ptr = &mut self.simd as *mut __m128 as *mut f32;
      ptr.add(index).as_mut().unwrap_unchecked()
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let ptr = &mut self.simd as *mut float32x4_t as *mut f32;
      ptr.add(index).as_mut().unwrap_unchecked()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      &mut self.data[index]
    }
  }
}

impl Vec4f32 {
  #[inline(always)]
  #[cfg(target_arch = "aarch64")]
  /// TODO: Document this item
  pub(crate) fn from_neon(v: float32x4_t) -> Self {
    Self { simd: v }
  }

  #[inline(always)]
  #[cfg(target_arch = "x86_64")]
  /// TODO: Document this item
  pub(crate) fn from_sse(v: __m128) -> Self {
    Self { simd: v }
  }

  #[inline(always)]
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  /// TODO: Document this item
  pub fn from_array(v: [f32; 4]) -> Self {
    Self { data: v }
  }
}

impl ops::Index<usize> for Quat {
  type Output = f32;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 4);
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let ptr = &self.0.simd as *const __m128 as *const f32;
      ptr.add(index).as_ref().unwrap_unchecked()
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let ptr = &self.0.simd as *const float32x4_t as *const f32;
      ptr.add(index).as_ref().unwrap_unchecked()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      &self.0.data[index]
    }
  }
}

impl ops::IndexMut<usize> for Quat {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 4);
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let ptr = &mut self.0.simd as *mut __m128 as *mut f32;
      ptr.add(index).as_mut().unwrap_unchecked()
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let ptr = &mut self.0.simd as *mut float32x4_t as *mut f32;
      ptr.add(index).as_mut().unwrap_unchecked()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      &mut self.0.data[index]
    }
  }
}

/// Hamiltonian Product Then implement ops::Mul for Quat using the Hamiltonian product:
impl ops::Mul for Quat {
  type Output = Self;
  fn mul(self, rhs: Self) -> Self::Output {
    let (x1, y1, z1, w1) = (self.0.x(), self.0.y(), self.0.z(), self.0.w());
    let (x2, y2, z2, w2) = (rhs.0.x(), rhs.0.y(), rhs.0.z(), rhs.0.w());

    // TODO SIMD implementation
    Quat(Vec4f32::from_components(
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ))
  }
}

impl ops::Add for Quat {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self(self.0 + rhs.0)
  }
}

impl ops::Sub for Quat {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self(self.0 - rhs.0)
  }
}

impl ops::Neg for Quat {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    Self(-self.0)
  }
}

impl ops::Mul<f32> for Quat {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    Self(self.0 * rhs)
  }
}

impl ops::Mul<Quat> for f32 {
  type Output = Quat;
  #[inline]
  fn mul(self, rhs: Quat) -> Self::Output {
    Quat(self * rhs.0)
  }
}

impl ops::Div<f32> for Quat {
  type Output = Self;
  #[inline]
  fn div(self, rhs: f32) -> Self::Output {
    Self(self.0 / rhs)
  }
}

impl Quaternion for Quat {
  type Scalar = f32;
  type Vector = Vec3f32;

  /// Returns the identity quaternion (0, 0, 0, 1)
  #[inline]
  fn identity() -> Self {
    Self(Vec4f32::from_components(0.0, 0.0, 0.0, 1.0))
  }

  /// Creates a quaternion from a vector part (x, y, z) and a scalar part (w)
  #[inline]
  fn from_vector_and_scalar(vector: Vec3f32, scalar: f32) -> Self {
    Self(Vec4f32::from_components(
      vector.x(),
      vector.y(),
      vector.z(),
      scalar,
    ))
  }

  #[inline]
  fn vector_part(self) -> Vec3f32 {
    Vec3f32::from_components(self.0.x(), self.0.y(), self.0.z())
  }

  #[inline]
  fn scalar_part(self) -> f32 {
    self.0.w()
  }

  /// Conjugate: negates the vector part
  #[inline]
  fn conjugate(self) -> Self {
    Self(Vec4f32::from_components(
      -self.0.x(),
      -self.0.y(),
      -self.0.z(),
      self.0.w(),
    ))
  }

  /// The squared length of the quaternion.
  /// Because it's just the dot product of the underlying Vec4 with itself,
  /// we can use our fast SIMD dot product!
  #[inline]
  fn norm_squared(self) -> f32 {
    self.0.dot(self.0)
  }

  /// The length (magnitude) of the quaternion
  #[inline]
  fn norm(self) -> f32 {
    crate::math::FloatLike::sqrt(self.norm_squared())
  }

  /// Returns a normalized version of the quaternion
  #[inline]
  fn normalize(self) -> Self {
    let n = self.norm();
    if n > 0.0 { self / n } else { Self::identity() }
  }

  /// The inverse of the quaternion. For unit quaternions, this is just the conjugate.
  #[inline]
  fn inverse(self) -> Self {
    let n_sq = self.norm_squared();
    if n_sq > 0.0 {
      self.conjugate() / n_sq
    } else {
      Self::identity() // Or panic, depending on your error handling philosophy
    }
  }
}

#[cfg(test)]
mod tests;

