use core::ops;

use crate::math::vector::{Vector, Vector2};

// Helper to keep test setup clean
pub fn vec2(x: f32, y: f32) -> Vec2f32 {
  Vec2f32::from_components(x, y)
}

// Note: For a 2 component vector, it seemed useless vectorizing with SIMD instructions
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vec2f32 {
  pub data: [f32; 2],
}

impl Vec2f32 {
  #[inline]
  pub fn from_array(data: [f32; 2]) -> Self {
    Self { data }
  }
}

impl From<[f32; 2]> for Vec2f32 {
  #[inline]
  fn from(value: [f32; 2]) -> Self {
    Self::from_array(value)
  }
}

impl Into<[f32; 2]> for Vec2f32 {
  #[inline]
  fn into(self) -> [f32; 2] {
    self.data
  }
}

impl PartialEq for Vec2f32 {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.data[0] == other.data[0] && self.data[1] == other.data[1]
  }
}

impl ops::Add for Vec2f32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self {
      data: [self.data[0] + rhs.data[0], self.data[1] + rhs.data[1]],
    }
  }
}

impl ops::Sub for Vec2f32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      data: [self.data[0] - rhs.data[0], self.data[1] - rhs.data[1]],
    }
  }
}

impl ops::Mul<f32> for Vec2f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    Self {
      data: [self.data[0] * rhs, self.data[1] * rhs],
    }
  }
}

impl ops::Mul<Vec2f32> for f32 {
  type Output = Vec2f32;
  #[inline]
  fn mul(self, rhs: Vec2f32) -> Self::Output {
    Vec2f32 {
      data: [self * rhs.data[0], self * rhs.data[1]],
    }
  }
}

impl ops::Mul<Self> for Vec2f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    Self {
      data: [self.data[0] * rhs.data[0], self.data[1] * rhs.data[1]],
    }
  }
}

impl ops::Div<f32> for Vec2f32 {
  type Output = Vec2f32;
  #[inline]
  fn div(self, rhs: f32) -> Self::Output {
    Self {
      data: [self.data[0] / rhs, self.data[1] / rhs],
    }
  }
}

impl ops::Div<Self> for Vec2f32 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self::Output {
    Self {
      data: [self.data[0] / rhs.data[0], self.data[1] / rhs.data[1]],
    }
  }
}

impl ops::AddAssign<Self> for Vec2f32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    self.data[0] += rhs.data[0];
    self.data[1] += rhs.data[1];
  }
}

impl ops::SubAssign<Self> for Vec2f32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    self.data[0] -= rhs.data[0];
    self.data[1] -= rhs.data[1];
  }
}

impl ops::MulAssign<Self> for Vec2f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    self.data[0] *= rhs.data[0];
    self.data[1] *= rhs.data[1];
  }
}

impl ops::MulAssign<f32> for Vec2f32 {
  #[inline]
  fn mul_assign(&mut self, rhs: f32) {
    self.data[0] *= rhs;
    self.data[1] *= rhs;
  }
}

impl ops::DivAssign<Self> for Vec2f32 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    self.data[0] /= rhs.data[0];
    self.data[1] /= rhs.data[1];
  }
}

impl ops::DivAssign<f32> for Vec2f32 {
  #[inline]
  fn div_assign(&mut self, rhs: f32) {
    self.data[0] /= rhs;
    self.data[1] /= rhs;
  }
}

impl ops::Neg for Vec2f32 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    Self {
      data: [-self.data[0], -self.data[1]],
    }
  }
}

impl ops::Index<usize> for Vec2f32 {
  type Output = f32;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 2);
    &self.data[index]
  }
}

impl ops::IndexMut<usize> for Vec2f32 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 2);
    &mut self.data[index]
  }
}

impl Vector for Vec2f32 {
  type Scalar = f32;
  const DIM: usize = 2;
  #[inline]
  fn zero() -> Self {
    Self { data: [0.0, 0.0] }
  }
  #[inline]
  fn splat(v: Self::Scalar) -> Self {
    Self { data: [v, v] }
  }
  #[inline]
  fn component(&self, i: usize) -> Option<Self::Scalar> {
    if i < Self::DIM {
      Some(self.data[i])
    } else {
      None
    }
  }
  #[inline]
  unsafe fn component_unchecked(&self, i: usize) -> Self::Scalar {
    unsafe { *self.data.as_ptr().add(i) }
  }
  #[inline]
  fn set_component(&mut self, i: usize, value: Self::Scalar) {
    if i < Self::DIM {
      self.data[i] = value;
    }
  }
  #[inline]
  fn dot(self, rhs: Self) -> Self::Scalar {
    self.data[0] * rhs.data[0] + self.data[1] * rhs.data[1]
  }
  #[inline]
  fn min(self, other: Self) -> Self {
    Self {
      data: [
        f32::min(self.data[0], other.data[0]),
        f32::min(self.data[1], other.data[1]),
      ],
    }
  }
  #[inline]
  fn max(self, other: Self) -> Self {
    Self {
      data: [
        f32::max(self.data[0], other.data[0]),
        f32::max(self.data[1], other.data[1]),
      ],
    }
  }
}

impl Vector2 for Vec2f32 {
  #[inline]
  fn from_components(x: Self::Scalar, y: Self::Scalar) -> Self {
    Self { data: [x, y] }
  }
  #[inline]
  fn x(&self) -> Self::Scalar {
    self.data[0]
  }
  #[inline]
  fn y(&self) -> Self::Scalar {
    self.data[1]
  }
}
