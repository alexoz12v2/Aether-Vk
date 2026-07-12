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
mod tests {
  extern crate std;

  use super::*;
  use crate::math::FloatLike;

  #[test]
  fn test_initialization_and_conversion() {
    let v = vec3f64(1.0, 2.0, 3.0);

    assert_eq!(v.x(), 1.0);
    assert_eq!(v.y(), 2.0);
    assert_eq!(v.z(), 3.0);

    let arr: [f64; 3] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0]);

    let v_from_arr = Vec3f64::from_array([4.0, 5.0, 6.0]);
    assert_eq!(v_from_arr, vec3f64(4.0, 5.0, 6.0));

    let v_from: Vec3f64 = [7.0, 8.0, 9.0].into();
    assert_eq!(v_from, vec3f64(7.0, 8.0, 9.0));
  }

  #[test]
  fn test_zero_and_splat() {
    assert_eq!(Vec3f64::zero(), vec3f64(0.0, 0.0, 0.0));
    assert_eq!(Vec3f64::splat(7.0), vec3f64(7.0, 7.0, 7.0));
    assert_eq!(Vec3f64::default(), Vec3f64::zero());
  }

  #[test]
  fn test_addition() {
    let v1 = vec3f64(1.0, 2.0, 3.0);
    let v2 = vec3f64(10.0, 20.0, 30.0);

    assert_eq!(v1 + v2, vec3f64(11.0, 22.0, 33.0));

    let mut v_assign = v1;
    v_assign += v2;
    assert_eq!(v_assign, vec3f64(11.0, 22.0, 33.0));
  }

  #[test]
  fn test_subtraction() {
    let v1 = vec3f64(10.0, 20.0, 30.0);
    let v2 = vec3f64(1.0, 2.0, 3.0);

    assert_eq!(v1 - v2, vec3f64(9.0, 18.0, 27.0));

    let mut v_assign = v1;
    v_assign -= v2;
    assert_eq!(v_assign, vec3f64(9.0, 18.0, 27.0));
  }

  #[test]
  fn test_multiplication() {
    let v1 = vec3f64(2.0, 3.0, 4.0);
    let v2 = vec3f64(3.0, 4.0, 5.0);

    // Vec * Vec
    assert_eq!(v1 * v2, vec3f64(6.0, 12.0, 20.0));

    // Vec * Scalar
    assert_eq!(v1 * 2.0, vec3f64(4.0, 6.0, 8.0));

    // Scalar * Vec
    assert_eq!(2.0 * v1, vec3f64(4.0, 6.0, 8.0));

    // Assign traits
    let mut v_assign_vec = v1;
    v_assign_vec *= v2;
    assert_eq!(v_assign_vec, vec3f64(6.0, 12.0, 20.0));

    let mut v_assign_scalar = v1;
    v_assign_scalar *= 2.0;
    assert_eq!(v_assign_scalar, vec3f64(4.0, 6.0, 8.0));
  }

  #[test]
  fn test_division() {
    let v1 = vec3f64(10.0, 20.0, 30.0);
    let v2 = vec3f64(2.0, 4.0, 5.0);

    assert_eq!(v1 / v2, vec3f64(5.0, 5.0, 6.0));
    assert_eq!(v1 / 2.0, vec3f64(5.0, 10.0, 15.0));

    let mut v_assign_vec = v1;
    v_assign_vec /= v2;
    assert_eq!(v_assign_vec, vec3f64(5.0, 5.0, 6.0));

    let mut v_assign_scalar = v1;
    v_assign_scalar /= 2.0;
    assert_eq!(v_assign_scalar, vec3f64(5.0, 10.0, 15.0));
  }

  #[test]
  fn test_negation() {
    let v = vec3f64(1.0, -2.0, 3.0);
    let neg_v = -v;

    assert_eq!(neg_v, vec3f64(-1.0, 2.0, -3.0));
  }

  #[test]
  fn test_dot_product() {
    let v1 = vec3f64(1.0, 2.0, 3.0);
    let v2 = vec3f64(2.0, 3.0, 4.0);

    // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
    assert_eq!(v1.dot(v2), 20.0);
  }

  #[test]
  fn test_cross_product() {
    // Standard basis vectors
    let x = vec3f64(1.0, 0.0, 0.0);
    let y = vec3f64(0.0, 1.0, 0.0);
    let z = vec3f64(0.0, 0.0, 1.0);

    // X x Y = Z
    assert_eq!(x.cross(y), z);
    // Y x Z = X
    assert_eq!(y.cross(z), x);
    // Z x X = Y
    assert_eq!(z.cross(x), y);
    // Y x X = -Z
    assert_eq!(y.cross(x), -z);

    // Arbitrary vectors
    let v1 = vec3f64(1.0, 2.0, 3.0);
    let v2 = vec3f64(4.0, 5.0, 6.0);
    assert_eq!(v1.cross(v2), vec3f64(-3.0, 6.0, -3.0));
  }

  #[test]
  fn test_length_and_normalize() {
    let v = vec3f64(3.0, 4.0, 0.0);
    assert_eq!(v.length_squared(), 25.0);
    assert_eq!(v.length(), 5.0);

    let n = v.normalize();
    let expected = vec3f64(0.6, 0.8, 0.0);
    assert!((n.x() - expected.x()).abs() < 1e-15);
    assert!((n.y() - expected.y()).abs() < 1e-15);
    assert!((n.z() - expected.z()).abs() < 1e-15);

    // Normalized vector should have unit length
    assert!((n.length() - 1.0).abs() < 1e-15);
  }

  #[test]
  fn test_min_max() {
    let v1 = vec3f64(1.0, 5.0, 3.0);
    let v2 = vec3f64(2.0, 4.0, 6.0);

    assert_eq!(v1.min(v2), vec3f64(1.0, 4.0, 3.0));
    assert_eq!(v1.max(v2), vec3f64(2.0, 5.0, 6.0));
  }

  #[test]
  fn test_indexing_and_components() {
    let mut v = vec3f64(1.0, 2.0, 3.0);

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

    assert_eq!(v, vec3f64(1.0, 5.0, 6.0));
  }

  #[test]
  fn test_to_f32_conversion() {
    let v64 = vec3f64(1.5, 2.5, 3.5);
    let v32 = v64.to_f32();
    assert_eq!(v32.x(), 1.5f32);
    assert_eq!(v32.y(), 2.5f32);
    assert_eq!(v32.z(), 3.5f32);
  }

  #[test]
  fn test_from_f32_conversion() {
    let v32 = Vec3f32::from_components(1.5, 2.5, 3.5);
    let v64 = Vec3f64::from_f32(v32);
    assert_eq!(v64.x(), 1.5);
    assert_eq!(v64.y(), 2.5);
    assert_eq!(v64.z(), 3.5);
  }

  #[test]
  fn test_f64_precision_advantage() {
    // Demonstrate that f64 preserves precision that f32 cannot.
    // 0.1 + 0.2 is a classic floating-point precision example.
    let a64 = vec3f64(0.1, 0.2, 0.0);
    let b64 = vec3f64(0.2, 0.1, 0.0);
    let sum64 = a64 + b64;

    let a32 = Vec3f32::from_components(0.1, 0.2, 0.0);
    let b32 = Vec3f32::from_components(0.2, 0.1, 0.0);
    let sum32 = a32 + b32;

    // Both should be approximately 0.3, but f64 is much closer
    let err64 = (sum64.x() - 0.3f64).abs();
    let err32 = (sum32.x() as f64 - 0.3f64).abs();

    // f64 error should be smaller than f32 error
    assert!(
      err64 < err32,
      "f64 error ({err64:e}) should be < f32 error ({err32:e})"
    );

    // Verify two very close values are distinguishable in f64 but not in f32
    let close_a = 1.0000000000000002_f64;
    let close_b = 1.0000000000000004_f64;
    let va = vec3f64(close_a, 0.0, 0.0);
    let vb = vec3f64(close_b, 0.0, 0.0);
    assert_ne!(va, vb, "f64 should distinguish these values");

    // The same values collapse in f32
    let va32 = Vec3f32::from_components(close_a as f32, 0.0, 0.0);
    let vb32 = Vec3f32::from_components(close_b as f32, 0.0, 0.0);
    assert_eq!(va32, vb32, "f32 cannot distinguish these values");
  }
}