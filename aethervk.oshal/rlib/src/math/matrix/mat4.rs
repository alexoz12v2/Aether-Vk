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
#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct Mat4x4f32 {
  pub x: Vec4f32,
  pub y: Vec4f32,
  pub z: Vec4f32,
  pub w: Vec4f32,
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
        x: Vec4f32 {
          simd: _mm_movelh_ps(tmp0, tmp1),
        },
        y: Vec4f32 {
          simd: _mm_movehl_ps(tmp1, tmp0),
        },
        z: Vec4f32 {
          simd: _mm_movelh_ps(tmp2, tmp3),
        },
        w: Vec4f32 {
          simd: _mm_movehl_ps(tmp3, tmp2),
        },
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
        x: Vec4f32 {
          simd: vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(zip0),
            vreinterpretq_f64_f32(zip1),
          )),
        },
        y: Vec4f32 {
          simd: vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(zip0),
            vreinterpretq_f64_f32(zip1),
          )),
        },
        z: Vec4f32 {
          simd: vreinterpretq_f32_f64(vzip1q_f64(
            vreinterpretq_f64_f32(zip2),
            vreinterpretq_f64_f32(zip3),
          )),
        },
        w: Vec4f32 {
          simd: vreinterpretq_f32_f64(vzip2q_f64(
            vreinterpretq_f64_f32(zip2),
            vreinterpretq_f64_f32(zip3),
          )),
        },
      }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
      Self {
        x: Vec4f32 {
          data: [
            self.x.data[0],
            self.y.data[0],
            self.z.data[0],
            self.w.data[0],
          ],
        },
        y: Vec4f32 {
          data: [
            self.x.data[1],
            self.y.data[1],
            self.z.data[1],
            self.w.data[1],
          ],
        },
        z: Vec4f32 {
          data: [
            self.x.data[2],
            self.y.data[2],
            self.z.data[2],
            self.w.data[2],
          ],
        },
        w: Vec4f32 {
          data: [
            self.x.data[3],
            self.y.data[3],
            self.z.data[3],
            self.w.data[3],
          ],
        },
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
      z: <Self::Vector as Vector4>::from_components(1.0, 0.0, 1.0, 0.0),
      w: <Self::Vector as Vector4>::from_components(1.0, 0.0, 0.0, 1.0),
    }
  }

  fn determinant(self) -> Self::Scalar {
    todo!()
  }

  fn inverse(self) -> Option<Self>
  where
    Self::Scalar: crate::math::FloatLike,
  {
    todo!()
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

      Vec4f32 {
        simd: _mm_add_ps(_mm_add_ps(m0, m1), _mm_add_ps(m2, m3)),
      }
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
      Vec4f32 {
        data: [
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
        ],
      }
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
}

impl Mat4x4f32 {
  // Scalar helper to compute deternimante with laplace expansion
  #[inline]
  fn scalar_determinant(&self) -> f32 {
    // elements for readability
    let m00 = self
  }
}
