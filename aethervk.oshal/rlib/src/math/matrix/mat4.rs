#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops;

use crate::math::{
  matrix::{Matrix, Matrix4, MatrixVectorMul, SquareMatrix},
  vector::{Vector, Vector4, vec4::Vec4f32},
};

/// Column-Major, f32 storage for 4x4 matrices
#[derive(Copy, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Mat4x4f32 {
  pub x: Vec4f32,
  pub y: Vec4f32,
  pub z: Vec4f32,
  pub w: Vec4f32,
}

impl Into<[f32; 16]> for Mat4x4f32 {
  #[inline]
  fn into(self) -> [f32; 16] {
    let mut result: [f32; 16] = [0.0; 16];
    (&mut result[0..4]).copy_from_slice(&Into::<[f32; 4]>::into(self.x));
    (&mut result[4..8]).copy_from_slice(&Into::<[f32; 4]>::into(self.y));
    (&mut result[8..12]).copy_from_slice(&Into::<[f32; 4]>::into(self.z));
    (&mut result[12..16]).copy_from_slice(&Into::<[f32; 4]>::into(self.w));

    result
  }
}

impl Into<[[f32; 4]; 4]> for Mat4x4f32 {
  #[inline]
  fn into(self) -> [[f32; 4]; 4] {
    let mut result: [[f32; 4]; 4] = [[0.0; 4]; 4];
    result[0] = Into::<[f32; 4]>::into(self.x);
    result[1] = Into::<[f32; 4]>::into(self.y);
    result[2] = Into::<[f32; 4]>::into(self.z);
    result[3] = Into::<[f32; 4]>::into(self.w);

    result
  }
}

impl ops::Add<Self> for Mat4x4f32 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self {
      x: self.x + rhs.x,
      y: self.y + rhs.y,
      z: self.z + rhs.z,
      w: self.w + rhs.w,
    }
  }
}

impl ops::Sub<Self> for Mat4x4f32 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      x: self.x - rhs.x,
      y: self.y - rhs.y,
      z: self.z - rhs.z,
      w: self.w - rhs.w,
    }
  }
}

impl ops::Mul<Mat4x4f32> for f32 {
  type Output = Mat4x4f32;
  #[inline]
  fn mul(self, rhs: Mat4x4f32) -> Self::Output {
    Self::Output {
      x: self * rhs.x,
      y: self * rhs.y,
      z: self * rhs.z,
      w: self * rhs.w,
    }
  }
}

impl ops::Mul<f32> for Mat4x4f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f32) -> Self::Output {
    Self {
      x: self.x * rhs,
      y: self.y * rhs,
      z: self.z * rhs,
      w: self.w * rhs,
    }
  }
}

impl ops::AddAssign<Self> for Mat4x4f32 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl ops::SubAssign<Self> for Mat4x4f32 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl ops::Index<usize> for Mat4x4f32 {
  type Output = Vec4f32;
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    let ptr = self as *const Self as *const Self::Output;
    // Safety:
    // - Since we are using repr(C), we can take this struct as a pointer and cast it to type of first member
    // - index should be bound as specified in debug_assert
    unsafe { ptr.add(index).as_ref().unwrap_unchecked() }
  }
}

impl ops::Index<(usize, usize)> for Mat4x4f32 {
  type Output = f32;
  #[inline]
  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    let index = col * <Self as Matrix>::ROWS + row;
    let ptr = self as *const Self as *const f32;
    // Safety:
    // - Since we are using repr(C), we can take this struct as a pointer and cast it to type of first member
    // - index should be bound as specified in debug_assert
    unsafe { ptr.add(index).as_ref().unwrap_unchecked() }
  }
}

impl ops::IndexMut<usize> for Mat4x4f32 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    let ptr = self as *mut Self as *mut Self::Output;
    // Safety:
    // - Since we are using repr(C), we can take this struct as a pointer and cast it to type of first member
    // - index should be bound as specified in debug_assert
    unsafe { ptr.add(index).as_mut().unwrap_unchecked() }
  }
}

impl ops::IndexMut<(usize, usize)> for Mat4x4f32 {
  #[inline]
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    let index = col * <Self as Matrix>::ROWS + row;
    let ptr = self as *mut Self as *mut f32;
    // Safety:
    // - Since we are using repr(C), we can take this struct as a pointer and cast it to type of first member
    // - index should be bound as specified in debug_assert
    unsafe { ptr.add(index).as_mut().unwrap_unchecked() }
  }
}

impl Matrix for Mat4x4f32 {
  type Scalar = f32;
  type Vector = Vec4f32;
  const ROWS: usize = 4;
  const COLS: usize = 4;
  #[inline]
  fn zero() -> Self {
    Self {
      x: <Self::Vector as Vector>::zero(),
      y: <Self::Vector as Vector>::zero(),
      z: <Self::Vector as Vector>::zero(),
      w: <Self::Vector as Vector>::zero(),
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
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // interleave low and upper halves
      let tmp0 = _mm_unpacklo_ps(self.x.simd, self.y.simd);
      let tmp1 = _mm_unpacklo_ps(self.z.simd, self.w.simd);
      let tmp2 = _mm_unpackhi_ps(self.x.simd, self.y.simd);
      let tmp3 = _mm_unpackhi_ps(self.z.simd, self.w.simd);

      // Move combinations to final rows
      Self {
        x: Vec4f32::from_sse(_mm_movelh_ps(tmp0, tmp1)),
        y: Vec4f32::from_sse(_mm_movehl_ps(tmp1, tmp0)),
        z: Vec4f32::from_sse(_mm_movelh_ps(tmp2, tmp3)),
        w: Vec4f32::from_sse(_mm_movehl_ps(tmp3, tmp2)),
      }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      // Zip 32-bit floats (interleave adjacent elements)
      let zip0 = vzip1q_f32(self.x.simd, self.y.simd);
      let zip1 = vzip1q_f32(self.z.simd, self.w.simd);
      let zip2 = vzip2q_f32(self.x.simd, self.y.simd);
      let zip3 = vzip2q_f32(self.z.simd, self.w.simd);

      // Reinterpret as 64-bit to zip the larger blocks together
      Self {
        x: Vec4f32::from_neon(vreinterpretq_f32_f64(vzip1q_f64(
          vreinterpretq_f64_f32(zip0),
          vreinterpretq_f64_f32(zip1),
        ))),
        y: Vec4f32::from_neon(vreinterpretq_f32_f64(vzip2q_f64(
          vreinterpretq_f64_f32(zip0),
          vreinterpretq_f64_f32(zip1),
        ))),
        z: Vec4f32::from_neon(vreinterpretq_f32_f64(vzip1q_f64(
          vreinterpretq_f64_f32(zip2),
          vreinterpretq_f64_f32(zip3),
        ))),
        w: Vec4f32::from_neon(vreinterpretq_f32_f64(vzip2q_f64(
          vreinterpretq_f64_f32(zip2),
          vreinterpretq_f64_f32(zip3),
        ))),
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        x: Vec4f32::from_array([
          self.x.data[0],
          self.y.data[0],
          self.z.data[0],
          self.w.data[0],
        ]),
        y: Vec4f32::from_array([
          self.x.data[1],
          self.y.data[1],
          self.z.data[1],
          self.w.data[1],
        ]),
        z: Vec4f32::from_array([
          self.x.data[2],
          self.y.data[2],
          self.z.data[2],
          self.w.data[2],
        ]),
        w: Vec4f32::from_array([
          self.x.data[3],
          self.y.data[3],
          self.z.data[3],
          self.w.data[3],
        ]),
      }
    }
  }
}

impl SquareMatrix for Mat4x4f32 {
  #[inline]
  fn identity() -> Self {
    Self {
      x: <Self::Vector as Vector4>::from_components(1.0, 0.0, 0.0, 0.0),
      y: <Self::Vector as Vector4>::from_components(0.0, 1.0, 0.0, 0.0),
      z: <Self::Vector as Vector4>::from_components(0.0, 0.0, 1.0, 0.0),
      w: <Self::Vector as Vector4>::from_components(0.0, 0.0, 0.0, 1.0),
    }
  }

  fn determinant(self) -> Self::Scalar {
    // SIMD appproach: compute 2x2 determinants in parallel with lots of shuffles.
    // unless required, I won't do that now
    self.scalar_determinant()
  }

  fn inverse(self) -> Option<Self>
  where
    Self::Scalar: crate::math::FloatLike,
  {
    let (det, adjugate) = self.scalar_det_and_adjugate();
    if det.abs() <= 1e-8 {
      return None;
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
      let inv_det = _mm_set1_ps(1.0f32 / det);
      Some(Self {
        x: Vec4f32::from_sse(_mm_mul_ps(adjugate.x.simd, inv_det)),
        y: Vec4f32::from_sse(_mm_mul_ps(adjugate.y.simd, inv_det)),
        z: Vec4f32::from_sse(_mm_mul_ps(adjugate.z.simd, inv_det)),
        w: Vec4f32::from_sse(_mm_mul_ps(adjugate.w.simd, inv_det)),
      })
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let inv_det = vdupq_n_f32(1.0f32 / det);
      Some(Self {
        x: Vec4f32::from_neon(vmulq_f32(adjugate.x.simd, inv_det)),
        y: Vec4f32::from_neon(vmulq_f32(adjugate.y.simd, inv_det)),
        z: Vec4f32::from_neon(vmulq_f32(adjugate.z.simd, inv_det)),
        w: Vec4f32::from_neon(vmulq_f32(adjugate.w.simd, inv_det)),
      })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      let inv_det = 1.0f32 / det;
      Some(adjugate.scale_all(inv_det))
    }
  }
}

impl MatrixVectorMul for Mat4x4f32 {
  #[inline]
  fn mul_vector(self, v: Self::Vector) -> Self::Vector {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // get transpose of rhs and dot prod each term
      let vx = _mm_shuffle_ps(v.simd, v.simd, 0x00);
      let vy = _mm_shuffle_ps(v.simd, v.simd, 0x55);
      let vz = _mm_shuffle_ps(v.simd, v.simd, 0xAA);
      let vw = _mm_shuffle_ps(v.simd, v.simd, 0xFF);

      let m0 = _mm_mul_ps(self.x.simd, vx);
      let m1 = _mm_mul_ps(self.y.simd, vy);
      let m2 = _mm_mul_ps(self.z.simd, vz);
      let m3 = _mm_mul_ps(self.w.simd, vw);

      Vec4f32::from_sse(_mm_add_ps(_mm_add_ps(m0, m1), _mm_add_ps(m2, m3)))
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      let mut res = vmulq_laneq_f32::<0>(self.x.simd, v.simd);
      res = vfmaq_laneq_f32::<1>(res, self.y.simd, v.simd);
      res = vfmaq_laneq_f32::<2>(res, self.z.simd, v.simd);
      res = vfmaq_laneq_f32::<3>(res, self.w.simd, v.simd);

      Vec4f32 { simd: res }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Vec4f32::from_array([
        self.x.data[0] * v.data[0]
          + self.y.data[0] * v.data[1]
          + self.z.data[0] * v.data[2]
          + self.w.data[0] * v.data[3],
        self.x.data[1] * v.data[0]
          + self.y.data[1] * v.data[1]
          + self.z.data[1] * v.data[2]
          + self.w.data[1] * v.data[3],
        self.x.data[2] * v.data[0]
          + self.y.data[2] * v.data[1]
          + self.z.data[2] * v.data[2]
          + self.w.data[2] * v.data[3],
        self.x.data[3] * v.data[0]
          + self.y.data[3] * v.data[1]
          + self.z.data[3] * v.data[2]
          + self.w.data[3] * v.data[3],
      ])
    }
  }
}

impl ops::Mul<Self> for Mat4x4f32 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    Self {
      // A * B is just applying matrix A to every column of B
      x: self.mul_vector(rhs.x),
      y: self.mul_vector(rhs.y),
      z: self.mul_vector(rhs.z),
      w: self.mul_vector(rhs.w),
    }
  }
}

impl Matrix4 for Mat4x4f32 {
  #[inline]
  fn from_columns(c0: Self::Vector, c1: Self::Vector, c2: Self::Vector, c3: Self::Vector) -> Self {
    Self {
      x: c0,
      y: c1,
      z: c2,
      w: c3,
    }
  }

  #[inline]
  fn from_array(x: &[Self::Scalar; 16]) -> Self {
    Self {
      x: Vec4f32::from_components(x[0], x[1], x[2], x[3]),
      y: Vec4f32::from_components(x[4], x[5], x[6], x[7]),
      z: Vec4f32::from_components(x[8], x[9], x[10], x[11]),
      w: Vec4f32::from_components(x[12], x[13], x[14], x[15]),
    }
  }
}

impl Mat4x4f32 {
  /// Helper: Scalar default 4x4 determinant with Laplace expansion
  #[inline]
  fn scalar_determinant(&self) -> f32 {
    let m = [
      self.x[0], self.x[1], self.x[2], self.x[3], self.y[0], self.y[1], self.y[2], self.y[3],
      self.z[0], self.z[1], self.z[2], self.z[3], self.w[0], self.w[1], self.w[2], self.w[3],
    ];

    let coef00 = m[10] * m[15] - m[14] * m[11];
    let coef02 = m[6] * m[15] - m[14] * m[7];
    let coef03 = m[6] * m[11] - m[10] * m[7];
    let coef04 = m[2] * m[15] - m[14] * m[3];
    let coef06 = m[2] * m[11] - m[10] * m[3];
    let coef07 = m[2] * m[7] - m[6] * m[3];

    let fac0 = m[5] * coef00 - m[9] * coef02 + m[13] * coef03;
    let fac1 = -(m[1] * coef00 - m[9] * coef04 + m[13] * coef06);
    let fac2 = m[1] * coef02 - m[5] * coef04 + m[13] * coef07;
    let fac3 = -(m[1] * coef03 - m[5] * coef06 + m[9] * coef07);

    m[0] * fac0 + m[4] * fac1 + m[8] * fac2 + m[12] * fac3
  }

  /// Helper: computes the adjugate matrix (required for inverse) and the deternimant at same time
  #[inline]
  fn scalar_det_and_adjugate(&self) -> (f32, Self) {
    let m = [
      self.x[0], self.x[1], self.x[2], self.x[3], self.y[0], self.y[1], self.y[2], self.y[3],
      self.z[0], self.z[1], self.z[2], self.z[3], self.w[0], self.w[1], self.w[2], self.w[3],
    ];

    let coef00 = m[10] * m[15] - m[14] * m[11];
    let coef02 = m[6] * m[15] - m[14] * m[7];
    let coef03 = m[6] * m[11] - m[10] * m[7];
    let coef04 = m[2] * m[15] - m[14] * m[3];
    let coef06 = m[2] * m[11] - m[10] * m[3];
    let coef07 = m[2] * m[7] - m[6] * m[3];
    let coef08 = m[9] * m[15] - m[13] * m[11];
    let coef10 = m[5] * m[15] - m[13] * m[7];
    let coef11 = m[5] * m[11] - m[9] * m[7];
    let coef12 = m[1] * m[15] - m[13] * m[3];
    let coef14 = m[1] * m[11] - m[9] * m[3];
    let coef15 = m[1] * m[7] - m[5] * m[3];
    let coef16 = m[9] * m[14] - m[13] * m[10];
    let coef18 = m[5] * m[14] - m[13] * m[6];
    let coef19 = m[5] * m[10] - m[9] * m[6];
    let coef20 = m[1] * m[14] - m[13] * m[2];
    let coef22 = m[1] * m[10] - m[9] * m[2];
    let coef23 = m[1] * m[6] - m[5] * m[2];

    let fac0 = m[5] * coef00 - m[9] * coef02 + m[13] * coef03;
    let fac1 = -(m[1] * coef00 - m[9] * coef04 + m[13] * coef06);
    let fac2 = m[1] * coef02 - m[5] * coef04 + m[13] * coef07;
    let fac3 = -(m[1] * coef03 - m[5] * coef06 + m[9] * coef07);

    let fac4 = -(m[4] * coef00 - m[8] * coef02 + m[12] * coef03);
    let fac5 = m[0] * coef00 - m[8] * coef04 + m[12] * coef06;
    let fac6 = -(m[0] * coef02 - m[4] * coef04 + m[12] * coef07);
    let fac7 = m[0] * coef03 - m[4] * coef06 + m[8] * coef07;

    let fac8 = m[4] * coef08 - m[8] * coef10 + m[12] * coef11;
    let fac9 = -(m[0] * coef08 - m[8] * coef12 + m[12] * coef14);
    let fac10 = m[0] * coef10 - m[4] * coef12 + m[12] * coef15;
    let fac11 = -(m[0] * coef11 - m[4] * coef14 + m[8] * coef15);

    let fac12 = -(m[4] * coef16 - m[8] * coef18 + m[12] * coef19);
    let fac13 = m[0] * coef16 - m[8] * coef20 + m[12] * coef22;
    let fac14 = -(m[0] * coef18 - m[4] * coef20 + m[12] * coef23);
    let fac15 = m[0] * coef19 - m[4] * coef22 + m[8] * coef23;

    let adjugate = Self {
      x: <Vec4f32 as Vector4>::from_components(fac0, fac1, fac2, fac3),
      y: <Vec4f32 as Vector4>::from_components(fac4, fac5, fac6, fac7),
      z: <Vec4f32 as Vector4>::from_components(fac8, fac9, fac10, fac11),
      w: <Vec4f32 as Vector4>::from_components(fac12, fac13, fac14, fac15),
    };
    let det = m[0] * fac0 + m[4] * fac1 + m[8] * fac2 + m[12] * fac3;
    (det, adjugate)
  }

  /// Helper: scales all elements by a scalar
  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  #[inline]
  fn scale_all(self, s: f32) -> Self {
    Self {
      x: Vec4f32 {
        data: [s * self.x[0], s * self.x[1], s * self.x[2], s * self.x[3]],
      },
      y: Vec4f32 {
        data: [s * self.y[0], s * self.y[1], s * self.y[2], s * self.y[3]],
      },
      z: Vec4f32 {
        data: [s * self.z[0], s * self.z[1], s * self.z[2], s * self.z[3]],
      },
      w: Vec4f32 {
        data: [s * self.w[0], s * self.w[1], s * self.w[2], s * self.w[3]],
      },
    }
  }
}

#[cfg(test)]
mod tests {
  extern crate std;
  use super::*;

  // Helper macro to easily define a column-major matrix
  // TODO: probably this is to be exported
  macro_rules! mat {
    (
            $c0x:expr, $c1x:expr, $c2x:expr, $c3x:expr,
            $c0y:expr, $c1y:expr, $c2y:expr, $c3y:expr,
            $c0z:expr, $c1z:expr, $c2z:expr, $c3z:expr,
            $c0w:expr, $c1w:expr, $c2w:expr, $c3w:expr $(,)?
        ) => {
      Mat4x4f32 {
        x: Vec4f32::from_components($c0x, $c0y, $c0z, $c0w),
        y: Vec4f32::from_components($c1x, $c1y, $c2y, $c1w), // Fixed macro layout for columns
        z: Vec4f32::from_components($c2x, $c2y, $c2z, $c2w),
        w: Vec4f32::from_components($c3x, $c3y, $c3z, $c3w),
      }
    };
  }

  // Simpler helper to ensure exact column mapping
  fn mat4_cols(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> Mat4x4f32 {
    Mat4x4f32 {
      x: Vec4f32::from_components(c0[0], c0[1], c0[2], c0[3]),
      y: Vec4f32::from_components(c1[0], c1[1], c1[2], c1[3]),
      z: Vec4f32::from_components(c2[0], c2[1], c2[2], c2[3]),
      w: Vec4f32::from_components(c3[0], c3[1], c3[2], c3[3]),
    }
  }

  #[test]
  fn test_identity() {
    let id = Mat4x4f32::identity();

    assert_eq!(id.x.x(), 1.0);
    assert_eq!(id.y.y(), 1.0);
    assert_eq!(id.z.z(), 1.0);
    assert_eq!(id.w.w(), 1.0);

    assert_eq!(id.x.y(), 0.0); // Check a few off-diagonals
    assert_eq!(id.z.w(), 0.0);
  }

  #[test]
  fn test_into_arrays() {
    let m = mat4_cols(
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    );

    // 1D Array Conversion (Column-Major Flat)
    let arr1d: [f32; 16] = m.into();
    assert_eq!(
      arr1d,
      [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
      ]
    );

    // 2D Array Conversion
    let arr2d: [[f32; 4]; 4] = m.into();
    assert_eq!(arr2d[0], [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(arr2d[3], [13.0, 14.0, 15.0, 16.0]);
  }

  #[test]
  fn test_indexing() {
    let mut m = Mat4x4f32::identity();

    // 1D (Column) Indexing
    assert_eq!(m[0].x(), 1.0);
    assert_eq!(m[1].y(), 1.0);

    // 2D (Row, Col) Indexing
    assert_eq!(m[(0, 0)], 1.0); // col 0, row 0
    assert_eq!(m[(2, 3)], 0.0); // col 3, row 2

    // Mutability
    m[(1, 2)] = 5.0;
    assert_eq!(m[(1, 2)], 5.0);
  }

  #[test]
  fn test_addition_and_subtraction() {
    let m1 = mat4_cols(
      [1.0, 1.0, 1.0, 1.0],
      [2.0, 2.0, 2.0, 2.0],
      [3.0, 3.0, 3.0, 3.0],
      [4.0, 4.0, 4.0, 4.0],
    );
    let m2 = mat4_cols(
      [10.0, 10.0, 10.0, 10.0],
      [20.0, 20.0, 20.0, 20.0],
      [30.0, 30.0, 30.0, 30.0],
      [40.0, 40.0, 40.0, 40.0],
    );

    let add = m1 + m2;
    assert_eq!(add[0].x(), 11.0);
    assert_eq!(add[3].w(), 44.0);

    let sub = m2 - m1;
    assert_eq!(sub[1].y(), 18.0);
    assert_eq!(sub[2].z(), 27.0);
  }

  #[test]
  fn test_scalar_multiplication() {
    let m = mat4_cols(
      [1.0, 2.0, 3.0, 4.0],
      [1.0, 2.0, 3.0, 4.0],
      [1.0, 2.0, 3.0, 4.0],
      [1.0, 2.0, 3.0, 4.0],
    );

    let m_scaled1 = m * 2.0;
    let m_scaled2 = 2.0 * m;

    assert_eq!(m_scaled1[0].x(), 2.0);
    assert_eq!(m_scaled1[0].y(), 4.0);
    assert_eq!(m_scaled1, m_scaled2); // Should be commutative
  }

  #[test]
  fn test_transpose() {
    let m = mat4_cols(
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    );

    let t = m.transpose();

    // Row 0 of `m` becomes Col 0 of `t`
    assert_eq!(t[0].x(), 1.0);
    assert_eq!(t[0].y(), 5.0);
    assert_eq!(t[0].z(), 9.0);
    assert_eq!(t[0].w(), 13.0);

    // Double transpose should yield the original matrix
    assert_eq!(t.transpose(), m);
  }

  #[test]
  fn test_matrix_vector_mul() {
    let m = mat4_cols(
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [10.0, 20.0, 30.0, 1.0], // Translation vector in w column
    );
    let v = Vec4f32::from_components(5.0, 5.0, 5.0, 1.0); // A point

    let result = m.mul_vector(v);

    assert_eq!(result.x(), 15.0); // 5 + 10
    assert_eq!(result.y(), 25.0); // 5 + 20
    assert_eq!(result.z(), 35.0); // 5 + 30
    assert_eq!(result.w(), 1.0);
  }

  #[test]
  fn test_matrix_matrix_mul() {
    let m = mat4_cols(
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0],
    );
    let id = Mat4x4f32::identity();

    // M * I = M
    assert_eq!(m * id, m);
    // I * M = M
    assert_eq!(id * m, m);

    // Test with a translation matrix
    let trans = mat4_cols(
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [10.0, 10.0, 10.0, 1.0],
    );
    let scale = mat4_cols(
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 2.0, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    );

    // Trans * Scale
    let combined = trans * scale;
    assert_eq!(combined[0].x(), 2.0); // Scale is applied
    assert_eq!(combined[3].x(), 10.0); // Translation remains in col 3
  }

  #[test]
  fn test_determinant() {
    let id = Mat4x4f32::identity();
    assert_eq!(id.determinant(), 1.0);

    let m = mat4_cols(
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 3.0, 0.0, 0.0],
      [0.0, 0.0, 4.0, 0.0],
      [0.0, 0.0, 0.0, 5.0],
    );
    // Determinant of diagonal matrix is product of diagonals
    assert_eq!(m.determinant(), 120.0);
  }

  #[test]
  fn test_inverse() {
    // Simple scale + translate matrix, easy to invert
    let m = mat4_cols(
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, 0.0],
      [0.0, 0.0, 2.0, 0.0],
      [10.0, 20.0, 30.0, 1.0],
    );

    let inv_opt = m.inverse();
    assert!(inv_opt.is_some(), "Matrix should be invertible");

    let inv = inv_opt.unwrap();

    // M * M^-1 should equal Identity
    let id_test = m * inv;
    let id = Mat4x4f32::identity();

    // Floating point comparisons after a matrix inverse might have tiny inaccuracies.
    // We use a small epsilon to check if it practically resulted in the Identity matrix.
    let epsilon = 1e-6;
    for col in 0..4 {
      for row in 0..4 {
        let diff = (id_test[(row, col)] - id[(row, col)]).abs();
        assert!(
          diff < epsilon,
          "Inverse failed at ({},{}). Expected {}, got {}",
          row,
          col,
          id[(row, col)],
          id_test[(row, col)]
        );
      }
    }

    // Test singular matrix (Determinant = 0)
    let singular = mat4_cols(
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0, 1.0],
    );
    assert!(singular.inverse().is_none());
  }
}
