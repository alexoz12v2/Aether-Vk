//! mat4x4f64 module.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops;

use crate::math::matrix::{Matrix, Matrix4, MatrixVectorMul, SquareMatrix};
use crate::math::{
  vector::{vec3f64::Vec3f64, vec4f64::Vec4f64, Vector, Vector4},
  FloatLike,
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

  pub fn to_mat4_f32(&self) -> super::mat4::Mat4x4f32 {
    super::mat4::Mat4x4f32 {
      x: crate::math::vector::vec4::Vec4f32::from_components(
        self.cols[0].x() as f32,
        self.cols[0].y() as f32,
        self.cols[0].z() as f32,
        self.cols[0].w() as f32,
      ),
      y: crate::math::vector::vec4::Vec4f32::from_components(
        self.cols[1].x() as f32,
        self.cols[1].y() as f32,
        self.cols[1].z() as f32,
        self.cols[1].w() as f32,
      ),
      z: crate::math::vector::vec4::Vec4f32::from_components(
        self.cols[2].x() as f32,
        self.cols[2].y() as f32,
        self.cols[2].z() as f32,
        self.cols[2].w() as f32,
      ),
      w: crate::math::vector::vec4::Vec4f32::from_components(
        self.cols[3].x() as f32,
        self.cols[3].y() as f32,
        self.cols[3].z() as f32,
        self.cols[3].w() as f32,
      ),
    }
  }

  pub fn perspective_vk_reverse_z(fov: f64, aspect: f64, near: f64, far: f64) -> Self {
    let half_fov = fov / 2.0;
    let f = 1.0 / half_fov.tan();

    let c0 = Vec4f64::from_components(f / aspect, 0.0, 0.0, 0.0);
    let c1 = Vec4f64::from_components(0.0, 0.0, near / (far - near), -1.0);
    let c2 = Vec4f64::from_components(0.0, -f, 0.0, 0.0);
    let c3 = Vec4f64::from_components(0.0, 0.0, far * near / (far - near), 0.0);

    Self::from_cols(c0, c1, c2, c3)
  }

  pub fn orthographic_vk_reverse_z(
    left: f64,
    right: f64,
    bottom: f64,
    top: f64,
    near: f64,
    far: f64,
  ) -> Self {
    let _0 = 0.0;
    let _1 = 1.0;
    let _2 = 2.0;

    let c0 = Vec4f64::from_components(_2 / (right - left), _0, _0, _0);
    let c1 = Vec4f64::from_components(_0, _0, _1 / (far - near), _0);
    let c2 = Vec4f64::from_components(_0, _2 / (bottom - top), _0, _0);
    let c3 = Vec4f64::from_components(
      -(right + left) / (right - left),
      -(bottom + top) / (bottom - top),
      far / (far - near),
      _1,
    );

    Self::from_cols(c0, c1, c2, c3)
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
  pub fn transpose_impl(&self) -> Self {
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

  pub fn determinant_impl(&self) -> f64 {
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

  pub fn inverse_impl(&self) -> Self {
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


impl Into<[f64; 16]> for Mat4x4f64 {
  #[inline]
  fn into(self) -> [f64; 16] {
    let mut result: [f64; 16] = [0.0; 16];
    (&mut result[0..4]).copy_from_slice(&Into::<[f64; 4]>::into(self.cols[0]));
    (&mut result[4..8]).copy_from_slice(&Into::<[f64; 4]>::into(self.cols[1]));
    (&mut result[8..12]).copy_from_slice(&Into::<[f64; 4]>::into(self.cols[2]));
    (&mut result[12..16]).copy_from_slice(&Into::<[f64; 4]>::into(self.cols[3]));

    #[cfg(any(debug_assertions, test))]
    {
      for f in result {
        if !f.is_finite() {
          crate::os::debug::print_stacktrace();
          panic!(
            "unexpected non finite number inside Mat4x4f64 (column major): {:?}",
            result
          );
        }
      }
    }

    result
  }
}

impl Into<[[f64; 4]; 4]> for Mat4x4f64 {
  #[inline]
  fn into(self) -> [[f64; 4]; 4] {
    let result: [[f64; 4]; 4] = [
      Into::<[f64; 4]>::into(self.cols[0]),
      Into::<[f64; 4]>::into(self.cols[1]),
      Into::<[f64; 4]>::into(self.cols[2]),
      Into::<[f64; 4]>::into(self.cols[3]),
    ];

    #[cfg(any(debug_assertions, test))]
    {
      for f in result.iter().flatten() {
        if !f.is_finite() {
          crate::os::debug::print_stacktrace();
          panic!(
            "unexpected non finite number inside Mat4x4f64 (column major): {:?}",
            result
          );
        }
      }
    }

    result
  }
}

impl ops::Index<usize> for Mat4x4f64 {
  type Output = Vec4f64;
  #[inline]
  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    &self.cols[index]
  }
}

impl ops::IndexMut<usize> for Mat4x4f64 {
  #[inline]
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index < <Self as Matrix>::COLS);
    &mut self.cols[index]
  }
}

impl ops::Index<(usize, usize)> for Mat4x4f64 {
  type Output = f64;
  #[inline]
  fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    &self.cols[col][row]
  }
}

impl ops::IndexMut<(usize, usize)> for Mat4x4f64 {
  #[inline]
  fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
    debug_assert!(row < <Self as Matrix>::ROWS && col < <Self as Matrix>::COLS);
    &mut self.cols[col][row]
  }
}

impl Matrix for Mat4x4f64 {
  type Scalar = f64;
  type Vector = Vec4f64;
  const ROWS: usize = 4;
  const COLS: usize = 4;

  #[inline]
  fn zero() -> Self {
    Self::zero()
  }

  #[inline]
  fn row(&self, r: usize) -> Option<Self::Vector> {
    if r < <Self as Matrix>::ROWS {
      Some(self.row(r))
    } else {
      None
    }
  }

  #[inline]
  unsafe fn row_unchecked(&self, r: usize) -> Self::Vector {
    self.row(r)
  }

  #[inline]
  fn column(&self, r: usize) -> Option<Self::Vector> {
    if r < Self::COLS {
      Some(self.cols[r])
    } else {
      None
    }
  }

  #[inline]
  unsafe fn column_unchecked(&self, r: usize) -> Self::Vector {
    unsafe { *self.column_unchecked(r) }
  }

  #[inline]
  fn transpose(self) -> Self {
    self.transpose_impl() // Relies on the inherent method already defined
  }
}

impl SquareMatrix for Mat4x4f64 {
  #[inline]
  fn identity() -> Self {
    Self::identity()
  }

  #[inline]
  fn determinant(self) -> Self::Scalar {
    self.determinant_impl()
  }

  #[inline]
  fn inverse(self) -> Option<Self>
  where
    Self::Scalar: crate::math::FloatLike,
  {
    let det = self.determinant_impl();
    // Using a tighter epsilon for f64 vs the 1e-30 used in f32
    if det.abs() <= 1e-60 {
      None
    } else {
      Some(self.inverse_impl())
    }
  }
}

impl MatrixVectorMul for Mat4x4f64 {
  #[inline]
  fn mul_vector(self, v: Self::Vector) -> Self::Vector {
    self * v // Relies on `ops::Mul<Vec4f64>` already implemented
  }
}

impl Matrix4 for Mat4x4f64 {
  #[inline]
  fn from_columns(c0: Self::Vector, c1: Self::Vector, c2: Self::Vector, c3: Self::Vector) -> Self {
    Self::from_cols(c0, c1, c2, c3)
  }

  #[inline]
  fn from_array(x: &[Self::Scalar; 16]) -> Self {
    Self {
      cols: [
        Vec4f64::from_components(x[0], x[1], x[2], x[3]),
        Vec4f64::from_components(x[4], x[5], x[6], x[7]),
        Vec4f64::from_components(x[8], x[9], x[10], x[11]),
        Vec4f64::from_components(x[12], x[13], x[14], x[15]),
      ],
    }
  }
}

impl Mat4x4f64 {
  /// get a floating point component from a column-major linear index
  #[inline]
  pub fn component(&self, linear_index: usize) -> Option<f64> {
    if linear_index < 16 {
      let ptr = self as *const Self as *const f64;
      unsafe { ptr.add(linear_index).as_ref().copied() }
    } else {
      None
    }
  }

  /// Rotates a 3D vector using the matrix.
  /// This multiplies the vector by the upper-left 3x3 portion of the matrix,
  /// ignoring the translation component (the `w` / 4th column).
  #[inline]
  pub fn rotate_vector(&self, v: Vec3f64) -> Vec3f64 {
    let v4 = Vec4f64::from_components(v[0], v[1], v[2], 0.0);
    let res = self.mul_vector(v4);

    Vec3f64::from_array([res[0], res[1], res[2]])
  }
}