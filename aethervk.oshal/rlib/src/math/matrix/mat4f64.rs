//! mat4x4f64 module.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops;

use crate::math::{
  vector::vec4f64::Vec4f64,
  vector::{Vector, Vector4},
};

#[repr(C, align(32))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat4x4f64 {
  pub cols: [Vec4f64; 4],
}

pub type DMat4x4 = Mat4x4f64;
pub type Mat4f64 = Mat4x4f64;
pub type DMat4 = Mat4x4f64;

impl Default for Mat4x4f64 {
  #[inline]
  fn default() -> Self {
    Self::identity()
  }
}

impl Mat4x4f64 {
  #[inline]
  pub const fn from_cols(c0: Vec4f64, c1: Vec4f64, c2: Vec4f64, c3: Vec4f64) -> Self {
    Self {
      cols: [c0, c1, c2, c3],
    }
  }

  #[inline]
  pub fn identity() -> Self {
    Self {
      cols: [
        Vec4f64::from_components(1.0, 0.0, 0.0, 0.0),
        Vec4f64::from_components(0.0, 1.0, 0.0, 0.0),
        Vec4f64::from_components(0.0, 0.0, 1.0, 0.0),
        Vec4f64::from_components(0.0, 0.0, 0.0, 1.0),
      ],
    }
  }

  #[inline]
  pub fn zero() -> Self {
    Self {
      cols: [Vec4f64::zero(); 4],
    }
  }

  #[inline]
  pub fn column(&self, index: usize) -> &Vec4f64 {
    &self.cols[index]
  }

  /// # Safety
  /// index should be between 0 and 3
  #[inline]
  pub unsafe fn column_unchecked(&self, index: usize) -> &Vec4f64 {
    unsafe { self.cols.get_unchecked(index) }
  }

  #[inline]
  pub fn column_mut(&mut self, index: usize) -> &mut Vec4f64 {
    &mut self.cols[index]
  }

  /// # Safety
  /// index should be between 0 and 3
  #[inline]
  pub unsafe fn column_mut_unchecked(&mut self, index: usize) -> &mut Vec4f64 {
    unsafe { self.cols.get_unchecked_mut(index) }
  }

  #[inline]
  pub fn row(&self, index: usize) -> Vec4f64 {
    Vec4f64::from_components(
      self.cols[0][index],
      self.cols[1][index],
      self.cols[2][index],
      self.cols[3][index],
    )
  }

  #[inline]
  pub fn transpose(&self) -> Self {
    #[cfg(target_arch = "x86_64")]
    unsafe {
      // Highly optimized AVX2 transpose
      let tmp0 = _mm256_shuffle_pd(self.cols[0].simd, self.cols[1].simd, 0x0);
      let tmp1 = _mm256_shuffle_pd(self.cols[0].simd, self.cols[1].simd, 0xF);
      let tmp2 = _mm256_shuffle_pd(self.cols[2].simd, self.cols[3].simd, 0x0);
      let tmp3 = _mm256_shuffle_pd(self.cols[2].simd, self.cols[3].simd, 0xF);

      let row0 = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
      let row1 = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
      let row2 = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);
      let row3 = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);

      Self {
        cols: [
          Vec4f64 { simd: row0 },
          Vec4f64 { simd: row1 },
          Vec4f64 { simd: row2 },
          Vec4f64 { simd: row3 },
        ],
      }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
      // NEON interleaved transpose for float64x2_t
      let tr0_0 = vzip1q_f64(self.cols[0].simd[0], self.cols[1].simd[0]);
      let tr0_1 = vzip2q_f64(self.cols[0].simd[0], self.cols[1].simd[0]);
      let tr1_0 = vzip1q_f64(self.cols[2].simd[0], self.cols[3].simd[0]);
      let tr1_1 = vzip2q_f64(self.cols[2].simd[0], self.cols[3].simd[0]);
      let tr2_0 = vzip1q_f64(self.cols[0].simd[1], self.cols[1].simd[1]);
      let tr2_1 = vzip2q_f64(self.cols[0].simd[1], self.cols[1].simd[1]);
      let tr3_0 = vzip1q_f64(self.cols[2].simd[1], self.cols[3].simd[1]);
      let tr3_1 = vzip2q_f64(self.cols[2].simd[1], self.cols[3].simd[1]);

      Self {
        cols: [
          Vec4f64 {
            simd: [tr0_0, tr1_0],
          },
          Vec4f64 {
            simd: [tr0_1, tr1_1],
          },
          Vec4f64 {
            simd: [tr2_0, tr3_0],
          },
          Vec4f64 {
            simd: [tr2_1, tr3_1],
          },
        ],
      }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        cols: [self.row(0), self.row(1), self.row(2), self.row(3)],
      }
    }
  }

  pub fn determinant(&self) -> f64 {
    let c0 = self.cols[0];
    let c1 = self.cols[1];
    let c2 = self.cols[2];
    let c3 = self.cols[3];

    let s0 = c0.x() * c1.y() - c0.y() * c1.x();
    let s1 = c0.x() * c1.z() - c0.z() * c1.x();
    let s2 = c0.x() * c1.w() - c0.w() * c1.x();
    let s3 = c0.y() * c1.z() - c0.z() * c1.y();
    let s4 = c0.y() * c1.w() - c0.w() * c1.y();
    let s5 = c0.z() * c1.w() - c0.w() * c1.z();

    let c5 = c2.z() * c3.w() - c2.w() * c3.z();
    let c4 = c2.y() * c3.w() - c2.w() * c3.y();
    let c3_cross = c2.y() * c3.z() - c2.z() * c3.y();
    let c2_cross = c2.x() * c3.w() - c2.w() * c3.x();
    let c1_cross = c2.x() * c3.z() - c2.z() * c3.x();
    let c0_cross = c2.x() * c3.y() - c2.y() * c3.x();

    s0 * c5 - s1 * c4 + s2 * c3_cross + s3 * c2_cross - s4 * c1_cross + s5 * c0_cross
  }

  pub fn inverse(&self) -> Self {
    let c0 = self.cols[0];
    let c1 = self.cols[1];
    let c2 = self.cols[2];
    let c3 = self.cols[3];

    let s0 = c0.x() * c1.y() - c0.y() * c1.x();
    let s1 = c0.x() * c1.z() - c0.z() * c1.x();
    let s2 = c0.x() * c1.w() - c0.w() * c1.x();
    let s3 = c0.y() * c1.z() - c0.z() * c1.y();
    let s4 = c0.y() * c1.w() - c0.w() * c1.y();
    let s5 = c0.z() * c1.w() - c0.w() * c1.z();

    let c5 = c2.z() * c3.w() - c2.w() * c3.z();
    let c4 = c2.y() * c3.w() - c2.w() * c3.y();
    let c3_cross = c2.y() * c3.z() - c2.z() * c3.y();
    let c2_cross = c2.x() * c3.w() - c2.w() * c3.x();
    let c1_cross = c2.x() * c3.z() - c2.z() * c3.x();
    let c0_cross = c2.x() * c3.y() - c2.y() * c3.x();

    let inv_det =
      1.0 / (s0 * c5 - s1 * c4 + s2 * c3_cross + s3 * c2_cross - s4 * c1_cross + s5 * c0_cross);

    let inv_0 = Vec4f64::from_components(
      c1.y() * c5 - c1.z() * c4 + c1.w() * c3_cross,
      -c0.y() * c5 + c0.z() * c4 - c0.w() * c3_cross,
      c3.y() * s5 - c3.z() * s4 + c3.w() * s3,
      -c2.y() * s5 + c2.z() * s4 - c2.w() * s3,
    );

    let inv_1 = Vec4f64::from_components(
      -c1.x() * c5 + c1.z() * c2_cross - c1.w() * c1_cross,
      c0.x() * c5 - c0.z() * c2_cross + c0.w() * c1_cross,
      -c3.x() * s5 + c3.z() * s2 - c3.w() * s1,
      c2.x() * s5 - c2.z() * s2 + c2.w() * s1,
    );

    let inv_2 = Vec4f64::from_components(
      c1.x() * c4 - c1.y() * c2_cross + c1.w() * c0_cross,
      -c0.x() * c4 + c0.y() * c2_cross - c0.w() * c0_cross,
      c3.x() * s4 - c3.y() * s2 + c3.w() * s0,
      -c2.x() * s4 + c2.y() * s2 - c2.w() * s0,
    );

    let inv_3 = Vec4f64::from_components(
      -c1.x() * c3_cross + c1.y() * c1_cross - c1.z() * c0_cross,
      c0.x() * c3_cross - c0.y() * c1_cross + c0.z() * c0_cross,
      -c3.x() * s3 + c3.y() * s1 - c3.z() * s0,
      c2.x() * s3 - c2.y() * s1 + c2.z() * s0,
    );

    Self {
      cols: [
        inv_0 * inv_det,
        inv_1 * inv_det,
        inv_2 * inv_det,
        inv_3 * inv_det,
      ],
    }
  }
}

impl ops::Add for Mat4x4f64 {
  type Output = Self;
  #[inline]
  fn add(self, rhs: Self) -> Self::Output {
    Self {
      cols: [
        self.cols[0] + rhs.cols[0],
        self.cols[1] + rhs.cols[1],
        self.cols[2] + rhs.cols[2],
        self.cols[3] + rhs.cols[3],
      ],
    }
  }
}

impl ops::Sub for Mat4x4f64 {
  type Output = Self;
  #[inline]
  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      cols: [
        self.cols[0] - rhs.cols[0],
        self.cols[1] - rhs.cols[1],
        self.cols[2] - rhs.cols[2],
        self.cols[3] - rhs.cols[3],
      ],
    }
  }
}

impl ops::Mul<f64> for Mat4x4f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: f64) -> Self::Output {
    Self {
      cols: [
        self.cols[0] * rhs,
        self.cols[1] * rhs,
        self.cols[2] * rhs,
        self.cols[3] * rhs,
      ],
    }
  }
}

impl ops::Mul<Mat4x4f64> for f64 {
  type Output = Mat4x4f64;
  #[inline]
  fn mul(self, rhs: Mat4x4f64) -> Self::Output {
    rhs * self
  }
}

impl ops::Mul<Vec4f64> for Mat4x4f64 {
  type Output = Vec4f64;
  #[inline]
  fn mul(self, rhs: Vec4f64) -> Self::Output {
    // Leverages SIMD Vec4f64 operations nicely for highly optimized linear combinations
    self.cols[0] * rhs.x()
      + self.cols[1] * rhs.y()
      + self.cols[2] * rhs.z()
      + self.cols[3] * rhs.w()
  }
}

impl ops::Mul<Mat4x4f64> for Mat4x4f64 {
  type Output = Self;
  #[inline]
  fn mul(self, rhs: Self) -> Self::Output {
    // Linear combination of columns. Compiler unrolls and directly uses
    // underlying AVX/NEON instructions defined in the Vec4f64 structs.
    let mut result = Self::zero();
    for i in 0..4 {
      result.cols[i] = self.cols[0] * rhs.cols[i].x()
        + self.cols[1] * rhs.cols[i].y()
        + self.cols[2] * rhs.cols[i].z()
        + self.cols[3] * rhs.cols[i].w();
    }
    result
  }
}

impl ops::AddAssign for Mat4x4f64 {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}

impl ops::SubAssign for Mat4x4f64 {
  #[inline]
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl ops::MulAssign<f64> for Mat4x4f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: f64) {
    *self = *self * rhs;
  }
}

impl ops::MulAssign<Mat4x4f64> for Mat4x4f64 {
  #[inline]
  fn mul_assign(&mut self, rhs: Mat4x4f64) {
    *self = *self * rhs;
  }
}
