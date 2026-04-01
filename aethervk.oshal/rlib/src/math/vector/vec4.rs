#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
use crate::math::{min_two, max_two};

use core::ops;

use crate::math::{
  quaternion::Quaternion,
  vector::{Vector, Vector4, vec3::Vec3f32},
};

// TODO for all other scalars too

#[repr(C, align(16))] // vital for proper alignment
#[derive(Copy, Clone)]
pub struct Vec4f32 {
  #[cfg(target_arch = "x86_64")]
  pub simd: __m128,
  #[cfg(target_arch = "aarch64")]
  pub simd: float32x4_t,
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  pub data: [f32; 4], // alignment to 16 like other branches
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
      self.data[index]
    }
  }
}

impl ops::IndexMut<usize> for Vec4f32 {
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

impl Quaternion for Vec4f32 {
  type Scalar = f32;
  type Vector = Vec3f32;
  #[inline]
  fn from_vector_and_scalar(vector: Self::Vector, scalar: Self::Scalar) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      const DEST_LANE: i32 = 3;
      const SRC_LANE: i32 = 0;
      Self::from_sse(_mm_insert_ps(
        vector.0.simd,
        _mm_set_ss(scalar),
        (DEST_LANE << 6) | (SRC_LANE << 4),
      ))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      Self::from_neon(vsetq_lane_f32::<3>(scalar, vector.0.simd))
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self::from_array([vector[0], vector[1], vector[2], scalar])
    }
  }

  #[inline]
  fn vector_part(self) -> Self::Vector {
    Vec3f32(self)
  }

  #[inline]
  fn scalar_part(self) -> Self::Scalar {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // _mm_extract_ps requires SSE4.1
      f32::from_bits(_mm_extract_ps::<3>(self.simd) as u32)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      vgetq_lane_f32::<3>(self.simd)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.data[3]
    }
  }
}
