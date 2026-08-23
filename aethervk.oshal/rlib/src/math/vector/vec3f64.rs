//! vec3f64 module — double-precision 3-component vector.

use core::ops;

use super::vec3::Vec3f32;
use crate::math::vector::{Vector, Vector3};

/// Helper to construct a `Vec3f64` from components.
pub fn vec3f64(x: f64, y: f64, z: f64) -> Vec3f64 {
  Vec3f64::from_components(x, y, z)
}

/// Double-precision 3-component vector with scalar `[f64; 4]` storage.
/// The 4th element is padding and must always be `0.0`.
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub struct Vec3f64 {
  data: [f64; 4],
}

pub type DVec3 = Vec3f64;

impl Default for Vec3f64 {
  fn default() -> Self {
    Self::zero()
  }
}

impl Vec3f64 {
  pub fn one() -> Self {
    Self::from_components(1.0, 1.0, 1.0)
  }

  #[inline]
  pub fn from_array(data: [f64; 3]) -> Self {
    Self {
      data: [data[0], data[1], data[2], 0.0],
    }
  }

  /// Lossy downcast to `Vec3f32`.
  #[inline]
  pub fn to_f32(&self) -> Vec3f32 {
    Vec3f32::from_components(
      self.data[0] as f32,
      self.data[1] as f32,
      self.data[2] as f32,
    )
  }

  /// Lossless upcast from `Vec3f32`.
  #[inline]
  pub fn from_f32(v: Vec3f32) -> Self {
    Self::from_components(v.x() as f64, v.y() as f64, v.z() as f64)
  }
}

impl From<[f64; 3]> for Vec3f64 {
  fn from(value: [f64; 3]) -> Self {
    Self::from_array(value)
  }
}

impl From<[f32; 3]> for Vec3f64 {
  fn from(value: [f32; 3]) -> Self {
    let val_f64 = [value[0] as f64, value[1] as f64, value[2] as f64];
    Self::from_array(val_f64)
  }
}

impl PartialEq for Vec3f64 {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.data[0] == other.data[0] && self.data[1] == other.data[1] && self.data[2] == other.data[2]
  }
}

impl Into<[f64; 3]> for Vec3f64 {
  #[inline]
  fn into(self) -> [f64; 3] {
    [self.data[0], self.data[1], self.data[2]]
  }
}

impl ops::Add for Vec3f64 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self {
      data: [
        self.data[0] + rhs.data[0],
        self.data[1] + rhs.data[1],
        self.data[2] + rhs.data[2],
        0.0,
      ],
    }
  }
}

impl ops::Sub for Vec3f64 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      data: [
        self.data[0] - rhs.data[0],
        self.data[1] - rhs.data[1],
        self.data[2] - rhs.data[2],
        0.0,
      ],
    }
  }
}

impl ops::Mul<f64> for Vec3f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f64) -> Self::Output {
    Self {
      data: [
        self.data[0] * rhs,
        self.data[1] * rhs,
        self.data[2] * rhs,
        0.0,
      ],
    }
  }
}

impl ops::Mul<Vec3f64> for f64 {
  type Output = Vec3f64;
  #[inline]
  fn mul(self, rhs: Vec3f64) -> Self::Output {
    Vec3f64 {
      data: [
        self * rhs.data[0],
        self * rhs.data[1],
        self * rhs.data[2],
        0.0,
      ],
    }
  }
}

impl ops::Mul<Self> for Vec3f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    Self {
      data: [
        self.data[0] * rhs.data[0],
        self.data[1] * rhs.data[1],
        self.data[2] * rhs.data[2],
        0.0,
      ],
    }
  }
}

impl ops::Div<f64> for Vec3f64 {
  type Output = Vec3f64;
  #[inline]
  fn div(self, rhs: f64) -> Self::Output {
    Self {
      data: [
        self.data[0] / rhs,
        self.data[1] / rhs,
        self.data[2] / rhs,
        0.0,
      ],
    }
  }
}

impl ops::Div<Self> for Vec3f64 {
  type Output = Self;
  #[inline]
  fn div(self, rhs: Self) -> Self::Output {
    Self {
      data: [
        self.data[0] / rhs.data[0],
        self.data[1] / rhs.data[1],
        self.data[2] / rhs.data[2],
        0.0,
      ],
    }
  }
}

impl ops::AddAssign<Self> for Vec3f64 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    self.data[0] += rhs.data[0];
    self.data[1] += rhs.data[1];
    self.data[2] += rhs.data[2];
  }
}

impl ops::SubAssign<Self> for Vec3f64 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    self.data[0] -= rhs.data[0];
    self.data[1] -= rhs.data[1];
    self.data[2] -= rhs.data[2];
  }
}

impl ops::MulAssign<Self> for Vec3f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: Self) {
    self.data[0] *= rhs.data[0];
    self.data[1] *= rhs.data[1];
    self.data[2] *= rhs.data[2];
  }
}

impl ops::MulAssign<f64> for Vec3f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: f64) {
    self.data[0] *= rhs;
    self.data[1] *= rhs;
    self.data[2] *= rhs;
  }
}

impl ops::DivAssign<Self> for Vec3f64 {
  #[inline]
  fn div_assign(&mut self, rhs: Self) {
    self.data[0] /= rhs.data[0];
    self.data[1] /= rhs.data[1];
    self.data[2] /= rhs.data[2];
  }
}

impl ops::DivAssign<f64> for Vec3f64 {
  #[inline]
  fn div_assign(&mut self, rhs: f64) {
    self.data[0] /= rhs;
    self.data[1] /= rhs;
    self.data[2] /= rhs;
  }
}

impl ops::Neg for Vec3f64 {
  type Output = Self;
  #[inline]
  fn neg(self) -> Self::Output {
    Self {
      data: [-self.data[0], -self.data[1], -self.data[2], 0.0],
    }
  }
}

impl ops::Index<usize> for Vec3f64 {
  type Output = f64;

  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < 3);
    &self.data[index]
  }
}

impl ops::IndexMut<usize> for Vec3f64 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < 3);
    &mut self.data[index]
  }
}

impl Vector for Vec3f64 {
  type Scalar = f64;
  const DIM: usize = 3;
  #[inline]
  fn zero() -> Self {
    Self { data: [0.0; 4] }
  }
  #[inline]
  fn splat(v: Self::Scalar) -> Self {
    Self {
      data: [v, v, v, 0.0],
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
    unsafe { *self.data.get_unchecked(i) }
  }
  #[inline]
  fn set_component(&mut self, i: usize, value: Self::Scalar) {
    if i < Self::DIM {
      self.data[i] = value;
    }
  }
  #[inline]
  fn dot(self, rhs: Self) -> Self::Scalar {
    self.data[0] * rhs.data[0] + self.data[1] * rhs.data[1] + self.data[2] * rhs.data[2]
  }
  #[inline]
  fn min(self, other: Self) -> Self {
    Self {
      data: [
        if self.data[0] < other.data[0] {
          self.data[0]
        } else {
          other.data[0]
        },
        if self.data[1] < other.data[1] {
          self.data[1]
        } else {
          other.data[1]
        },
        if self.data[2] < other.data[2] {
          self.data[2]
        } else {
          other.data[2]
        },
        0.0,
      ],
    }
  }
  #[inline]
  fn max(self, other: Self) -> Self {
    Self {
      data: [
        if self.data[0] > other.data[0] {
          self.data[0]
        } else {
          other.data[0]
        },
        if self.data[1] > other.data[1] {
          self.data[1]
        } else {
          other.data[1]
        },
        if self.data[2] > other.data[2] {
          self.data[2]
        } else {
          other.data[2]
        },
        0.0,
      ],
    }
  }
}

impl Vector3 for Vec3f64 {
  #[inline]
  fn from_components(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self {
    Self {
      data: [x, y, z, 0.0],
    }
  }
  #[inline]
  fn x(&self) -> Self::Scalar {
    self.data[0]
  }
  #[inline]
  fn y(&self) -> Self::Scalar {
    self.data[1]
  }
  #[inline]
  fn z(&self) -> Self::Scalar {
    self.data[2]
  }
  #[inline]
  fn cross(self, rhs: Self) -> Self {
    Self {
      data: [
        self.data[1] * rhs.data[2] - self.data[2] * rhs.data[1],
        self.data[2] * rhs.data[0] - self.data[0] * rhs.data[2],
        self.data[0] * rhs.data[1] - self.data[1] * rhs.data[0],
        0.0,
      ],
    }
  }
}

#[cfg(test)]
mod tests;
