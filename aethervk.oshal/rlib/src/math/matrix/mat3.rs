//! mat3 module.

use core::ops;

use crate::math::{
  matrix::{Matrix, Matrix3, MatrixVectorMul, SquareMatrix},
  vector::{Vector, Vector3, vec3::Vec3f32},
};

/// Column-Major, f32 storage for 3x3 matrices
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Mat3f32 {
  pub x: Vec3f32,
  pub y: Vec3f32,
  pub z: Vec3f32,
}

impl ops::Add<Self> for Mat3f32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self {
      x: self.x + rhs.x,
      y: self.y + rhs.y,
      z: self.z + rhs.z,
    }
  }
}

impl ops::Sub<Self> for Mat3f32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      x: self.x - rhs.x,
      y: self.y - rhs.y,
      z: self.z - rhs.z,
    }
  }
}

impl ops::Mul<Mat3f32> for f32 {
  type Output = Mat3f32;
  #[inline]
  fn mul(self, rhs: Mat3f32) -> Self::Output {
    Self::Output {
      x: self * rhs.x,
      y: self * rhs.y,
      z: self * rhs.z,
    }
  }
}

impl ops::Mul<f32> for Mat3f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    Self {
      x: self.x * rhs,
      y: self.y * rhs,
      z: self.z * rhs,
    }
  }
}

impl ops::AddAssign<Self> for Mat3f32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl ops::SubAssign<Self> for Mat3f32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl ops::Index<usize> for Mat3f32 {
  type Output = Vec3f32;
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    let ptr = self as *const Self as *const Self::Output;
    unsafe { ptr.add(index).as_ref().unwrap_unchecked() }
  }
}

impl ops::Index<(usize, usize)> for Mat3f32 {
  type Output = f32;
  #[inline]
  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    let index = col * 4 + row;
    let ptr = self as *const Self as *const f32;
    unsafe { ptr.add(index).as_ref().unwrap_unchecked() }
  }
}

impl ops::IndexMut<usize> for Mat3f32 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    let ptr = self as *mut Self as *mut Self::Output;
    unsafe { ptr.add(index).as_mut().unwrap_unchecked() }
  }
}

impl ops::IndexMut<(usize, usize)> for Mat3f32 {
  #[inline]
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    let index = col * 4 + row;
    let ptr = self as *mut Self as *mut f32;
    unsafe { ptr.add(index).as_mut().unwrap_unchecked() }
  }
}

impl Matrix for Mat3f32 {
  type Scalar = f32;
  type Vector = Vec3f32;
  const ROWS: usize = 3;
  const COLS: usize = 3;
  #[inline]
  fn zero() -> Self {
    Self {
      x: <Self::Vector as Vector>::zero(),
      y: <Self::Vector as Vector>::zero(),
      z: <Self::Vector as Vector>::zero(),
    }
  }
  #[inline]
  fn row(&self, r: usize) -> Option<Self::Vector> {
    if r < <Self as Matrix>::ROWS {
      Some(unsafe { self.row_unchecked(r) })
    } else {
      None
    }
  }
  #[inline]
  unsafe fn row_unchecked(&self, r: usize) -> Self::Vector {
    let t = self.transpose();
    t[r]
  }
  #[inline]
  fn column(&self, r: usize) -> Option<Self::Vector> {
    if r < Self::COLS {
      Some(unsafe { self.column_unchecked(r) })
    } else {
      None
    }
  }
  #[inline]
  unsafe fn column_unchecked(&self, r: usize) -> Self::Vector {
    self[r]
  }

  fn transpose(self) -> Self {
    Self {
      x: Vec3f32::from_components(self.x.x(), self.y.x(), self.z.x()),
      y: Vec3f32::from_components(self.x.y(), self.y.y(), self.z.y()),
      z: Vec3f32::from_components(self.x.z(), self.y.z(), self.z.z()),
    }
  }
}

impl SquareMatrix for Mat3f32 {
  #[inline]
  fn identity() -> Self {
    Self {
      x: <Self::Vector as Vector3>::from_components(1.0, 0.0, 0.0),
      y: <Self::Vector as Vector3>::from_components(0.0, 1.0, 0.0),
      z: <Self::Vector as Vector3>::from_components(0.0, 0.0, 1.0),
    }
  }

  fn determinant(self) -> Self::Scalar {
    let (m00, m10, m20) = (self.x.x(), self.x.y(), self.x.z());
    let (m01, m11, m21) = (self.y.x(), self.y.y(), self.y.z());
    let (m02, m12, m22) = (self.z.x(), self.z.y(), self.z.z());

    m00 * (m11 * m22 - m21 * m12) - m01 * (m10 * m22 - m20 * m12) + m02 * (m10 * m21 - m20 * m11)
  }

  fn inverse(self) -> Option<Self>
  where
    Self::Scalar: crate::math::FloatLike,
  {
    let det = self.determinant();
    if det.abs() <= 1e-8 {
      return None;
    }

    let (m00, m10, m20) = (self.x.x(), self.x.y(), self.x.z());
    let (m01, m11, m21) = (self.y.x(), self.y.y(), self.y.z());
    let (m02, m12, m22) = (self.z.x(), self.z.y(), self.z.z());

    let inv_det = 1.0 / det;

    let adjugate = Self {
      x: Vec3f32::from_components(
        (m11 * m22 - m21 * m12) * inv_det,
        -(m10 * m22 - m20 * m12) * inv_det,
        (m10 * m21 - m20 * m11) * inv_det,
      ),
      y: Vec3f32::from_components(
        -(m01 * m22 - m21 * m02) * inv_det,
        (m00 * m22 - m20 * m02) * inv_det,
        -(m00 * m21 - m20 * m01) * inv_det,
      ),
      z: Vec3f32::from_components(
        (m01 * m12 - m11 * m02) * inv_det,
        -(m00 * m12 - m10 * m02) * inv_det,
        (m00 * m11 - m10 * m01) * inv_det,
      ),
    };

    Some(adjugate)
  }
}

impl MatrixVectorMul for Mat3f32 {
  #[inline]
  fn mul_vector(self, v: Self::Vector) -> Self::Vector {
    Vec3f32::from_components(
      self.x.x() * v.x() + self.y.x() * v.y() + self.z.x() * v.z(),
      self.x.y() * v.x() + self.y.y() * v.y() + self.z.y() * v.z(),
      self.x.z() * v.x() + self.y.z() * v.y() + self.z.z() * v.z(),
    )
  }
}

impl ops::Mul<Vec3f32> for Mat3f32 {
  type Output = Vec3f32;

  fn mul(self, rhs: Vec3f32) -> Self::Output {
    MatrixVectorMul::mul_vector(self, rhs)
  }
}

impl ops::Mul<Self> for Mat3f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    Self {
      x: self.mul_vector(rhs.x),
      y: self.mul_vector(rhs.y),
      z: self.mul_vector(rhs.z),
    }
  }
}

impl Matrix3 for Mat3f32 {
  #[inline]
  fn from_columns(c0: Self::Vector, c1: Self::Vector, c2: Self::Vector) -> Self {
    Self {
      x: c0,
      y: c1,
      z: c2,
    }
  }

  #[inline]
  fn from_array(x: &[Self::Scalar; 9]) -> Self {
    Self {
      x: Vec3f32::from_components(x[0], x[1], x[2]),
      y: Vec3f32::from_components(x[3], x[4], x[5]),
      z: Vec3f32::from_components(x[6], x[7], x[8]),
    }
  }
}

impl Mat3f32 {
  pub fn component(&self, linear_index: usize) -> Option<f32> {
    if linear_index < 9 {
      let ptr = self as *const Self as *const f32;
      unsafe { ptr.add(linear_index).as_ref().copied() }
    } else {
      None
    }
  }

  /// TODO: Document this item
  pub fn identity() -> Self {
    Mat3f32::from_array(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  }
}

#[cfg(test)]
mod tests;

