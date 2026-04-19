#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::math::{min_two, max_two};

use core::ops;

use crate::math::{
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32},
};

// A helper function to easily create our vector for testing
pub fn vec(x: f32, y: f32, z: f32, w: f32) -> Vec4f32 {
  Vec4f32::from_components(x, y, z, w)
}

#[repr(C, align(16))] // vital for proper alignment
#[derive(Copy, Clone, Debug)]
pub struct Vec4f32 {
  #[cfg(target_arch = "x86_64")]
  pub simd: __m128,
  #[cfg(target_arch = "aarch64")]
  pub simd: float32x4_t,
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  pub data: [f32; 4], // alignment to 16 like other branches
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Quat(pub Vec4f32);

impl Quat {
  pub fn from_components(p0: f32, p1: f32, p2: f32, p3: f32) -> Quat {
    Self {
      0: Vec4f32::from_components(p0, p1, p2, p3),
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
  pub(crate) fn from_neon(v: float32x4_t) -> Self {
    Self { simd: v }
  }

  #[inline(always)]
  #[cfg(target_arch = "x86_64")]
  pub(crate) fn from_sse(v: __m128) -> Self {
    Self { simd: v }
  }

  #[inline(always)]
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  pub(crate) fn from_array(v: [f32; 4]) -> Self {
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
    if n > 0.0 {
      self / n
    } else {
      Self::identity()
    }
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
mod tests {
  // Bring std into scope conditionally for tests, even if the crate is no_std
  extern crate std;

  use crate::math::vector::vec3::vec3;

  use super::*;
  use core::f32;

  #[test]
  fn test_initialization_and_conversion() {
    let v = vec(1.0, 2.0, 3.0, 4.0);

    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);
    assert_eq!(v.w(), 4.0);

    let arr: [f32; 4] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
  }

  #[test]
  fn test_zero_and_splat() {
    let zero = Vec4f32::zero();
    assert_eq!(zero, vec(0.0, 0.0, 0.0, 0.0));

    let splat = Vec4f32::splat(5.0);
    assert_eq!(splat, vec(5.0, 5.0, 5.0, 5.0));
  }

  #[test]
  fn test_partial_eq() {
    let v1 = vec(1.0, 2.0, 3.0, 4.0);
    let v2 = vec(1.0, 2.0, 3.0, 4.0);
    let v3 = vec(1.0, 2.0, 3.0, 5.0); // Differs in w
    let v4 = vec(0.0, 2.0, 3.0, 4.0); // Differs in x

    assert!(v1 == v2);
    assert!(v1 != v3);
    assert!(v1 != v4);
  }

  #[test]
  fn test_addition() {
    let v1 = vec(1.0, 2.0, 3.0, 4.0);
    let v2 = vec(10.0, 20.0, 30.0, 40.0);

    // Add
    assert_eq!(v1 + v2, vec(11.0, 22.0, 33.0, 44.0));

    // AddAssign
    let mut v_assign = v1;
    v_assign += v2;
    assert_eq!(v_assign, vec(11.0, 22.0, 33.0, 44.0));
  }

  #[test]
  fn test_subtraction() {
    let v1 = vec(10.0, 20.0, 30.0, 40.0);
    let v2 = vec(1.0, 2.0, 3.0, 4.0);

    // Sub
    assert_eq!(v1 - v2, vec(9.0, 18.0, 27.0, 36.0));

    // SubAssign
    let mut v_assign = v1;
    v_assign -= v2;
    assert_eq!(v_assign, vec(9.0, 18.0, 27.0, 36.0));
  }

  #[test]
  fn test_multiplication() {
    let v1 = vec(2.0, 3.0, 4.0, 5.0);
    let v2 = vec(3.0, 4.0, 5.0, 6.0);

    // Vector * Vector
    assert_eq!(v1 * v2, vec(6.0, 12.0, 20.0, 30.0));

    // Vector * Scalar
    assert_eq!(v1 * 2.0, vec(4.0, 6.0, 8.0, 10.0));

    // Scalar * Vector
    assert_eq!(2.0 * v1, vec(4.0, 6.0, 8.0, 10.0));

    // MulAssign Vector
    let mut v_assign_vec = v1;
    v_assign_vec *= v2;
    assert_eq!(v_assign_vec, vec(6.0, 12.0, 20.0, 30.0));

    // MulAssign Scalar
    let mut v_assign_scalar = v1;
    v_assign_scalar *= 2.0;
    assert_eq!(v_assign_scalar, vec(4.0, 6.0, 8.0, 10.0));
  }

  #[test]
  fn test_division() {
    let v1 = vec(10.0, 20.0, 30.0, 40.0);
    let v2 = vec(2.0, 4.0, 5.0, 8.0);

    // Vector / Vector
    assert_eq!(v1 / v2, vec(5.0, 5.0, 6.0, 5.0));

    // Vector / Scalar
    assert_eq!(v1 / 2.0, vec(5.0, 10.0, 15.0, 20.0));

    // DivAssign Vector
    let mut v_assign_vec = v1;
    v_assign_vec /= v2;
    assert_eq!(v_assign_vec, vec(5.0, 5.0, 6.0, 5.0));

    // DivAssign Scalar
    let mut v_assign_scalar = v1;
    v_assign_scalar /= 2.0;
    assert_eq!(v_assign_scalar, vec(5.0, 10.0, 15.0, 20.0));
  }

  #[test]
  fn test_negation() {
    let v = vec(1.0, -2.0, 3.0, -0.0);
    let neg_v = -v;

    assert_eq!(neg_v.x(), -1.0);
    assert_eq!(neg_v.y(), 2.0);
    assert_eq!(neg_v.z(), -3.0);
    // Using to_bits to verify exact -0.0 representation if necessary
    assert_eq!(neg_v.w().to_bits(), 0.0f32.to_bits());
  }

  #[test]
  fn test_dot_product() {
    let v1 = vec(1.0, 2.0, 3.0, 4.0);
    let v2 = vec(2.0, 3.0, 4.0, 5.0);

    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert_eq!(v1.dot(v2), 40.0);
  }

  #[test]
  fn test_min_max() {
    let v1 = vec(1.0, 5.0, 3.0, 7.0);
    let v2 = vec(2.0, 4.0, 6.0, 1.0);

    assert_eq!(v1.min(v2), vec(1.0, 4.0, 3.0, 1.0));
    assert_eq!(v1.max(v2), vec(2.0, 5.0, 6.0, 7.0));
  }

  #[test]
  fn test_indexing_and_components() {
    let mut v = vec(1.0, 2.0, 3.0, 4.0);

    // Index
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);

    // Component method
    assert_eq!(v.component(0), Some(1.0));
    assert_eq!(v.component(4), None); // Out of bounds

    // IndexMut
    v[1] = 5.0;
    assert_eq!(v[1], 5.0);

    // Set component
    v.set_component(2, 6.0);
    assert_eq!(v[2], 6.0);

    assert_eq!(v, vec(1.0, 5.0, 6.0, 4.0));
  }

  #[test]
  fn test_quaternion_indexing() {
    let mut q = Quat(vec(1.0, 2.0, 3.0, 4.0));

    // Index
    assert_eq!(q[0], 1.0);
    assert_eq!(q[1], 2.0);
    assert_eq!(q[2], 3.0);
    assert_eq!(q[3], 4.0);

    // IndexMut
    q[1] = 5.0;
    assert_eq!(q[1], 5.0);

    assert_eq!(q.0, vec(1.0, 5.0, 3.0, 4.0));
  }

  #[test]
  fn test_quaternion_traits() {
    // Constructing a dummy Vec3f32 based on your implementation snippet
    // We assume Vec3f32 is a wrapper around Vec4f32
    let vec3 = Vec3f32(vec(1.0, 2.0, 3.0, 0.0));
    let scalar = 4.0;

    let q = Quat::from_vector_and_scalar(vec3, scalar);

    assert_eq!(q.0, vec(1.0, 2.0, 3.0, 4.0));
    assert_eq!(q.scalar_part(), 4.0);

    let extracted_vec3 = q.vector_part();
    assert_eq!(extracted_vec3.0.x(), 1.0);
    assert_eq!(extracted_vec3.0.y(), 2.0);
    assert_eq!(extracted_vec3.0.z(), 3.0);
  }

  use core::f32::consts::PI;

  // Helper macro for floating point comparisons
  macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
      let eps = 1e-5;
      assert!(
        ($a - $b).abs() < eps,
        "assertion failed: `(left !== right)`\n  left: `{:?}`,\n right: `{:?}`",
        $a,
        $b
      );
    };
  }

  macro_rules! assert_vec3_approx_eq {
    ($v1:expr, $v2:expr) => {
      assert_approx_eq!(
        crate::math::vector::Vector3::x(&$v1),
        crate::math::vector::Vector3::x(&$v2)
      );
      assert_approx_eq!(
        crate::math::vector::Vector3::y(&$v1),
        crate::math::vector::Vector3::y(&$v2)
      );
      assert_approx_eq!(
        crate::math::vector::Vector3::z(&$v1),
        crate::math::vector::Vector3::z(&$v2)
      );
    };
  }

  #[test]
  fn test_quaternion_identity() {
    let id = Quat::identity();
    let v = vec3(1.0, 2.0, 3.0);

    // Rotating by identity should return the exact same vector
    let rotated = id.rotate_vector(v);
    assert_vec3_approx_eq!(rotated, v);
  }

  #[test]
  fn test_quaternion_axis_angle_rotations() {
    let x_axis = vec3(1.0, 0.0, 0.0);
    let y_axis = vec3(0.0, 1.0, 0.0);
    let z_axis = vec3(0.0, 0.0, 1.0);

    // 1. Rotate Y vector 90 degrees around X axis -> Should become Z vector
    let q_rot_x = Quat::from_axis_angle(x_axis, PI / 2.0);
    let rotated_y = q_rot_x.rotate_vector(y_axis);
    assert_vec3_approx_eq!(rotated_y, z_axis);

    // 2. Rotate X vector 90 degrees around Y axis -> Should become -Z vector
    let q_rot_y = Quat::from_axis_angle(y_axis, PI / 2.0);
    let rotated_x = q_rot_y.rotate_vector(x_axis);
    assert_vec3_approx_eq!(rotated_x, vec3(0.0, 0.0, -1.0));

    // 3. Rotate X vector 90 degrees around Z axis -> Should become Y vector
    let q_rot_z = Quat::from_axis_angle(z_axis, PI / 2.0);
    let rotated_x_around_z = q_rot_z.rotate_vector(x_axis);
    assert_vec3_approx_eq!(rotated_x_around_z, y_axis);
  }

  #[test]
  fn test_quaternion_conjugate_and_inverse() {
    let axis = vec3(1.0, 1.0, 1.0).normalize(); // Assuming Vec3 has normalize()
    let q = Quat::from_axis_angle(axis, PI / 3.0); // 60 degrees

    // For unit quaternions, conjugate == inverse
    let conjugate = q.conjugate();
    let inverse = q.inverse();

    assert_approx_eq!(conjugate.0.x(), inverse.0.x());
    assert_approx_eq!(conjugate.0.y(), inverse.0.y());
    assert_approx_eq!(conjugate.0.z(), inverse.0.z());
    assert_approx_eq!(conjugate.0.w(), inverse.0.w());

    // Rotating by q, then by its inverse, should yield the original vector
    let v = vec3(10.0, 0.0, 0.0);
    let rotated = q.rotate_vector(v);
    let unrotated = inverse.rotate_vector(rotated);

    assert_vec3_approx_eq!(v, unrotated);
  }

  #[test]
  fn test_quaternion_slerp() {
    let q1 = Quat::identity();
    let x_axis = vec3(1.0, 0.0, 0.0);
    let q2 = Quat::from_axis_angle(x_axis, PI / 2.0); // 90 degrees

    // Slerp at t = 0.0 should be q1
    let slerp_0 = Quat::slerp(q1, q2, 0.0);
    assert_approx_eq!(slerp_0.0.w(), q1.0.w());

    // Slerp at t = 1.0 should be q2
    let slerp_1 = Quat::slerp(q1, q2, 1.0);
    assert_approx_eq!(slerp_1.0.w(), q2.0.w());

    // Slerp at t = 0.5 should be a 45 degree rotation around X
    let slerp_half = Quat::slerp(q1, q2, 0.5);
    let expected_q = Quat::from_axis_angle(x_axis, PI / 4.0);

    // If dot product is ~1.0 or ~-1.0, they are the same rotation
    assert!((slerp_half.0.dot(expected_q.0)).abs() > 0.9999);
    assert_approx_eq!(slerp_half.0.x(), expected_q.0.x());
    assert_approx_eq!(slerp_half.0.w(), expected_q.0.w());
  }

  #[test]
  fn test_quaternion_norm_and_normalize() {
    // Create a non-unit quaternion manually
    let q = Quat(vec(2.0, 0.0, 0.0, 0.0));

    // Norm should be 2.0
    assert_approx_eq!(q.norm(), 2.0);

    // Normalized should be (1.0, 0.0, 0.0, 0.0)
    let q_norm = Quaternion::normalize(q);
    assert_approx_eq!(q_norm.norm(), 1.0);
    assert_approx_eq!(q_norm.0.x(), 1.0);
  }

  use crate::math::matrix::mat3::Mat3f32;

  #[test]
  fn test_quaternion_to_matrix3_identity() {
    let q = Quat::identity();
    let mat: Mat3f32 = q.to_matrix3();

    let x_axis = vec3(1.0, 0.0, 0.0);
    let y_axis = vec3(0.0, 1.0, 0.0);
    let z_axis = vec3(0.0, 0.0, 1.0);

    // The identity quaternion should produce the identity matrix.
    // Multiplying the identity matrix by basis vectors should return the basis vectors.
    assert_vec3_approx_eq!(mat * x_axis, x_axis);
    assert_vec3_approx_eq!(mat * y_axis, y_axis);
    assert_vec3_approx_eq!(mat * z_axis, z_axis);
  }

  #[test]
  fn test_quaternion_to_matrix3_specific_rotation() {
    let x_axis = vec3(1.0, 0.0, 0.0);
    let y_axis = vec3(0.0, 1.0, 0.0);
    let z_axis = vec3(0.0, 0.0, 1.0);

    // Create a quaternion representing a 90-degree (PI/2) rotation around the Y axis
    let q_rot_y = Quat::from_axis_angle(y_axis, PI / 2.0);
    let mat: Mat3f32 = q_rot_y.to_matrix3();

    // Extract the columns of the resulting matrix by multiplying by basis vectors
    let col0 = mat * x_axis;
    let col1 = mat * y_axis;
    let col2 = mat * z_axis;

    // A 90-degree rotation around Y maps:
    // X -> -Z
    // Y -> Y (unchanged)
    // Z -> X
    assert_vec3_approx_eq!(col0, vec3(0.0, 0.0, -1.0));
    assert_vec3_approx_eq!(col1, vec3(0.0, 1.0, 0.0));
    assert_vec3_approx_eq!(col2, vec3(1.0, 0.0, 0.0));
  }

  #[test]
  fn test_quaternion_to_matrix3_action_equivalency() {
    // This test ensures that rotating a vector using the quaternion's `rotate_vector`
    // produces the EXACT same result as converting the quaternion to a matrix and
    // multiplying the matrix by the vector.

    let axis = vec3(1.0, 1.0, 1.0).normalize();
    let q = Quat::from_axis_angle(axis, PI / 3.0); // 60 degree rotation
    let mat: Mat3f32 = q.to_matrix3();

    let v = vec3(10.0, -5.0, 42.0); // Arbitrary vector

    let rotated_by_quat = q.rotate_vector(v);
    let rotated_by_mat = mat * v;

    assert_vec3_approx_eq!(rotated_by_quat, rotated_by_mat);
  }
}
