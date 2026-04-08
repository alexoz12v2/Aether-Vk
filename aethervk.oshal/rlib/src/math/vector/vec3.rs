use core::ops;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::math::vector::{Vector, Vector3, Vector4, vec4::Vec4f32};

// Helper to keep test setup clean
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3f32 {
  Vec3f32::from_components(x, y, z)
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct Vec3f32(pub Vec4f32);

impl Vec3f32 {
  #[inline]
  pub fn from_array(data: [f32; 3]) -> Self {
    Self(Vec4f32::from_components(data[0], data[1], data[2], 0.0))
  }
}

impl From<[f32; 3]> for Vec3f32 {
  fn from(value: [f32; 3]) -> Self {
    Self::from_array(value)
  }
}

impl PartialEq for Vec3f32 {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      let cmp = _mm_cmpeq_ps(self.0.simd, other.0.simd);
      // 0x7 is binary 0111. This masks out the w-component (the 4th bit)
      // and only requires x, y, and z to be equal.
      (_mm_movemask_ps(cmp) & 0x7) == 0x7
    }
    #[cfg(target_arch = "aarch64")]
    {
      // For NEON, horizontal scalar comparison is highly efficient
      // and avoids complexities with NaN in the unused lane.
      self.x() == other.x() && self.y() == other.y() && self.z() == other.z()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.0.data[0] == other.0.data[0]
        && self.0.data[1] == other.0.data[1]
        && self.0.data[2] == other.0.data[2]
    }
  }
}

impl Into<[f32; 3]> for Vec3f32 {
  #[inline]
  fn into(self) -> [f32; 3] {
    [self.x(), self.y(), self.z()]
  }
}

impl ops::Add for Vec3f32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self(self.0 + rhs.0)
  }
}

impl ops::Sub for Vec3f32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self(self.0 - rhs.0)
  }
}

impl ops::Mul<f32> for Vec3f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    Self(self.0 * rhs)
  }
}

impl ops::Mul<Vec3f32> for f32 {
  type Output = Vec3f32;
  #[inline]
  fn mul(self, rhs: Vec3f32) -> Self::Output {
    Vec3f32(self * rhs.0)
  }
}

impl ops::Mul<Self> for Vec3f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    Self(self.0 * rhs.0)
  }
}

impl ops::Div<f32> for Vec3f32 {
  type Output = Vec3f32;
  #[inline]
  fn div(self, rhs: f32) -> Self::Output {
    Self(self.0 / rhs)
  }
}

impl ops::Div<Self> for Vec3f32 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self::Output {
    Self(self.0 / rhs.0)
  }
}

impl ops::AddAssign<Self> for Vec3f32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    self.0 = self.0 + rhs.0;
  }
}

impl ops::SubAssign<Self> for Vec3f32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    self.0 = self.0 - rhs.0;
  }
}

impl ops::MulAssign<Self> for Vec3f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    self.0 = self.0 * rhs.0;
  }
}

impl ops::MulAssign<f32> for Vec3f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: f32) {
    self.0 = self.0 * rhs
  }
}

impl ops::DivAssign<Self> for Vec3f32 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    self.0 = self.0 / rhs.0;
  }
}

impl ops::DivAssign<f32> for Vec3f32 {
  #[inline]
  fn div_assign(&mut self, rhs: f32) {
    self.0 = self.0 / rhs;
  }
}

impl ops::Neg for Vec3f32 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    Self(-self.0)
  }
}

impl ops::Index<usize> for Vec3f32 {
  type Output = f32;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 3);
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

impl ops::IndexMut<usize> for Vec3f32 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 3);
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

impl Vector for Vec3f32 {
  type Scalar = f32;
  const DIM: usize = 3;
  #[inline]
  fn zero() -> Self {
    Self(Vec4f32::zero())
  }
  #[inline]
  fn splat(v: Self::Scalar) -> Self {
    // splat to all, w *should* be ignored
    Self(Vec4f32::splat(v))
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
    unsafe { self.0.component_unchecked(i) }
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
      // Mask 0x71; Dot product of first 3 components, store in first component
      let res = _mm_dp_ps(self.0.simd, rhs.0.simd, 0x71);
      _mm_cvtss_f32(res)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let prod_u32 = vreinterpretq_u32_f32(vmulq_f32(self.0.simd, rhs.0.simd));
      let mask = vsetq_lane_u32::<3>(0u32, vdupq_n_u32(0xFFFF_FFFF));
      let prod_masked = vreinterpretq_f32_u32(vandq_u32(prod_u32, mask));
      // alternative: vgetq_lane and sum them manually
      vaddvq_f32(prod_masked)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      self.0.data[0] * rhs.0.data[0]
        + self.0.data[1] * rhs.0.data[1]
        + self.0.data[2] * rhs.0.data[2]
    }
  }
  #[inline]
  fn min(self, other: Self) -> Self {
    Self(self.0.min(other.0))
  }
  #[inline]
  fn max(self, other: Self) -> Self {
    Self(self.0.max(other.0))
  }
}

impl Vector3 for Vec3f32 {
  #[inline]
  fn from_components(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self {
    Self(Vec4f32::from_components(x, y, z, 0.0))
  }
  #[inline]
  fn x(&self) -> Self::Scalar {
    self.0.x()
  }
  #[inline]
  fn y(&self) -> Self::Scalar {
    self.0.y()
  }
  #[inline]
  fn z(&self) -> Self::Scalar {
    self.0.z()
  }
  #[inline]
  fn cross(self, rhs: Self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // Cross product: (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
      // _MM_SHUFFLE(w, z, y, x) -> indices, x,y from first arg, z,w from snd
      let a = self.0.simd;
      let b = rhs.0.simd;
      let res = _mm_sub_ps(
        _mm_mul_ps(
          _mm_shuffle_ps(a, a, super::_MM_SHUFFLE(3, 0, 2, 1)),
          _mm_shuffle_ps(b, b, super::_MM_SHUFFLE(3, 1, 0, 2)),
        ),
        _mm_mul_ps(
          _mm_shuffle_ps(a, a, super::_MM_SHUFFLE(3, 1, 0, 2)),
          _mm_shuffle_ps(b, b, super::_MM_SHUFFLE(3, 0, 2, 1)),
        ),
      );
      Self(Vec4f32 { simd: res })
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let a = self.0.simd;
      let b = rhs.0.simd;

      // We map bytes to shuffle [x, y, z, w] into [y, z, x, w]
      // Each f32 is 4 bytes.
      let mask_yzxw: uint8x16_t = core::mem::transmute([
        4u8, 5, 6, 7, // y
        8, 9, 10, 11, // z
        0, 1, 2, 3, // x
        12, 13, 14, 15, // w
      ]);

      // We map bytes to shuffle [x, y, z, w] into [z, x, y, w]
      let mask_zxyw: uint8x16_t = core::mem::transmute([
        8u8, 9, 10, 11, // z
        0, 1, 2, 3, // x
        4, 5, 6, 7, // y
        12, 13, 14, 15, // w
      ]);

      // Cast f32 vectors to u8 vectors for the table lookup
      let a_bytes = vreinterpretq_u8_f32(a);
      let b_bytes = vreinterpretq_u8_f32(b);

      // Perform the shuffles
      let a_yzxw = vreinterpretq_f32_u8(vqtbl1q_u8(a_bytes, mask_yzxw));
      let b_zxyw = vreinterpretq_f32_u8(vqtbl1q_u8(b_bytes, mask_zxyw));

      let a_zxyw = vreinterpretq_f32_u8(vqtbl1q_u8(a_bytes, mask_zxyw));
      let b_yzxw = vreinterpretq_f32_u8(vqtbl1q_u8(b_bytes, mask_yzxw));

      // Calculate cross product: (a_yzxw * b_zxyw) - (a_zxyw * b_yzxw)
      let res = vsubq_f32(vmulq_f32(a_yzxw, b_zxyw), vmulq_f32(a_zxyw, b_yzxw));

      Self(Vec4f32 { simd: res })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      let a = &self.0.data;
      let b = &rhs.0.data;
      Self(Vec4f32 {
        data: [
          a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0],
          0.0,
        ],
      })
    }
  }
}

#[cfg(test)]
mod tests {
  extern crate std;

  use super::*;

  #[test]
  fn test_initialization_and_conversion() {
    let v = vec3(1.0, 2.0, 3.0);

    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);

    let arr: [f32; 3] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0]);

    let v_from_arr = Vec3f32::from_array([4.0, 5.0, 6.0]);
    assert_eq!(v_from_arr, vec3(4.0, 5.0, 6.0));
  }

  #[test]
  fn test_zero_and_splat() {
    assert_eq!(Vec3f32::zero(), vec3(0.0, 0.0, 0.0));
    assert_eq!(Vec3f32::splat(7.0), vec3(7.0, 7.0, 7.0));
  }

  #[test]
  fn test_addition() {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(10.0, 20.0, 30.0);

    assert_eq!(v1 + v2, vec3(11.0, 22.0, 33.0));

    let mut v_assign = v1;
    v_assign += v2;
    assert_eq!(v_assign, vec3(11.0, 22.0, 33.0));
  }

  #[test]
  fn test_subtraction() {
    let v1 = vec3(10.0, 20.0, 30.0);
    let v2 = vec3(1.0, 2.0, 3.0);

    assert_eq!(v1 - v2, vec3(9.0, 18.0, 27.0));

    let mut v_assign = v1;
    v_assign -= v2;
    assert_eq!(v_assign, vec3(9.0, 18.0, 27.0));
  }

  #[test]
  fn test_multiplication() {
    let v1 = vec3(2.0, 3.0, 4.0);
    let v2 = vec3(3.0, 4.0, 5.0);

    // Vec * Vec
    assert_eq!(v1 * v2, vec3(6.0, 12.0, 20.0));

    // Vec * Scalar
    assert_eq!(v1 * 2.0, vec3(4.0, 6.0, 8.0));

    // Scalar * Vec
    assert_eq!(2.0 * v1, vec3(4.0, 6.0, 8.0));

    // Assign traits
    let mut v_assign_vec = v1;
    v_assign_vec *= v2;
    assert_eq!(v_assign_vec, vec3(6.0, 12.0, 20.0));

    let mut v_assign_scalar = v1;
    v_assign_scalar *= 2.0;
    assert_eq!(v_assign_scalar, vec3(4.0, 6.0, 8.0));
  }

  #[test]
  fn test_division() {
    let v1 = vec3(10.0, 20.0, 30.0);
    let v2 = vec3(2.0, 4.0, 5.0);

    assert_eq!(v1 / v2, vec3(5.0, 5.0, 6.0));
    assert_eq!(v1 / 2.0, vec3(5.0, 10.0, 15.0));

    let mut v_assign_vec = v1;
    v_assign_vec /= v2;
    assert_eq!(v_assign_vec, vec3(5.0, 5.0, 6.0));

    let mut v_assign_scalar = v1;
    v_assign_scalar /= 2.0;
    assert_eq!(v_assign_scalar, vec3(5.0, 10.0, 15.0));
  }

  #[test]
  fn test_negation() {
    let v = vec3(1.0, -2.0, 3.0);
    let neg_v = -v;

    assert_eq!(neg_v, vec3(-1.0, 2.0, -3.0));
  }

  #[test]
  fn test_dot_product() {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(2.0, 3.0, 4.0);

    // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    assert_eq!(v1.dot(v2), 20.0);

    // Ensure the w component doesn't bleed into the 3D dot product
    // (Even if underlying Vec4f32 had a non-zero w, though our constructors force 0.0)
    let v3 = Vec3f32(Vec4f32::from_components(1.0, 1.0, 1.0, 100.0));
    let v4 = Vec3f32(Vec4f32::from_components(1.0, 1.0, 1.0, 100.0));
    assert_eq!(v3.dot(v4), 3.0); // Should be 1*1 + 1*1 + 1*1 = 3, ignoring the 100s
  }

  #[test]
  fn test_cross_product() {
    // Standard basis vectors
    let x = vec3(1.0, 0.0, 0.0);
    let y = vec3(0.0, 1.0, 0.0);
    let z = vec3(0.0, 0.0, 1.0);

    // X x Y = Z
    assert_eq!(x.cross(y), z);
    // Y x Z = X
    assert_eq!(y.cross(z), x);
    // Z x X = Y
    assert_eq!(z.cross(x), y);
    // Y x X = -Z
    assert_eq!(y.cross(x), -z);

    // Arbitrary vectors
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(4.0, 5.0, 6.0);
    assert_eq!(v1.cross(v2), vec3(-3.0, 6.0, -3.0));
  }

  #[test]
  fn test_min_max() {
    let v1 = vec3(1.0, 5.0, 3.0);
    let v2 = vec3(2.0, 4.0, 6.0);

    assert_eq!(v1.min(v2), vec3(1.0, 4.0, 3.0));
    assert_eq!(v1.max(v2), vec3(2.0, 5.0, 6.0));
  }

  #[test]
  fn test_indexing_and_components() {
    let mut v = vec3(1.0, 2.0, 3.0);

    // Index
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);

    // Component method
    assert_eq!(v.component(0), Some(1.0));
    assert_eq!(v.component(1), Some(2.0));
    assert_eq!(v.component(2), Some(3.0));
    assert_eq!(v.component(3), None); // Bounds check

    // IndexMut
    v[1] = 5.0;
    assert_eq!(v[1], 5.0);

    // Set component
    v.set_component(2, 6.0);
    assert_eq!(v.z(), 6.0);

    assert_eq!(v, vec3(1.0, 5.0, 6.0));
  }
}
