use core::ops;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::math::vector::{Vector, Vector3, Vector4, vec4::Vec4f32};

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq)]
pub struct Vec3f32(pub Vec4f32);

impl Vec3f32 {
    #[inline]
    pub fn from_array(data: [f32; 3]) -> Self {
        Self(Vec4f32::from_components(data[0], data[1], data[2], 0.0))
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
          _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)),
          _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2)),
        ),
        _mm_mul_ps(
          _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)),
          _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)),
        ),
      );
      Self(Vec4f32 { simd: res })
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let a = self.0.simd;
      let b = rhs.0.simd;
      let a_yzx = vextq_f32(a, a, 1);
      let a_zxy = vextq_f32(a, a, 2);
      let b_yzx = vextq_f32(b, b, 1);
      let b_zxy = vextq_f32(b, b, 2);
      let res = vsubq_f32(vmulq_f32(a_yzx, b_zxy), vmulq_f32(a_zxy, b_yzx));

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
          0,
        ],
      })
    }
  }
}
