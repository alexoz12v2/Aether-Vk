use core::ops;

use crate::math::{
  FloatLike, MulAddIdentity, Scalar,
  vector::{Vector, Vector3, Vector4},
};

// Note: We are taking by copy everything (almost) cause these are supposed to be small dimensional
pub trait Matrix:
  Copy
  + Sized
  + ops::Add<Output = Self>
  + ops::AddAssign<Self>
  + ops::Sub<Output = Self>
  + ops::SubAssign<Self>
{
  type Scalar: Scalar;
  type Vector: Vector<Scalar = Self::Scalar>;

  const ROWS: usize;
  const COLS: usize;

  fn zero() -> Self;

  fn row(&self, r: usize) -> Option<Self::Vector>;
  unsafe fn row_unchecked(&self, r: usize) -> Self::Vector;
  fn column(&self, r: usize) -> Option<Self::Vector>;
  unsafe fn column_unchecked(&self, r: usize) -> Self::Vector;

  fn transpose(self) -> Self;
}

pub trait SquareMatrix: Matrix + ops::Mul<Self, Output = Self> {
  fn identity() -> Self;
  fn determinant(self) -> Self::Scalar;
  fn inverse(self) -> Option<Self>
  where
    Self::Scalar: FloatLike;
}

pub trait MatrixVectorMul: Matrix {
  fn mul_vector(self, v: Self::Vector) -> Self::Vector;
}

// 3x3 for linear/normal transforms and tangent frames
pub trait Matrix3: SquareMatrix
where
  Self::Vector: Vector3,
{
  fn from_columns(x: Self::Vector, y: Self::Vector, z: Self::Vector) -> Self;
}

// 4x4 for affine transforms, projection matrices
// Note: These implementations assume standard graphics column-major memory layout and a
// Right-Handed coordinate system (like OpenGL).
pub trait Matrix4: SquareMatrix
where
  Self::Vector: Vector4,
{
  /// Required Constructor: builds a 4x4 from 4 column vectors.
  fn from_columns(c0: Self::Vector, c1: Self::Vector, c2: Self::Vector, c3: Self::Vector) -> Self;

  /// Creates a translation matrix.
  fn translation(v: Self::Vector) -> Self {
    let id = Self::identity();
    // Safety: Matrix4 is guaranteed to be 4x4, so indices 0..3 are always valid
    unsafe {
      Self::from_columns(
        id.column_unchecked(0),
        id.column_unchecked(1),
        id.column_unchecked(2),
        v, // the 4th column is the translation vector
      )
    }
  }

  /// Creates a standard right-handed perspective projection matrix
  /// This is the general OpenGL column-major matrix formula from scratchapixel with
  /// special case left = -right, top = -bottom
  fn perspective(
    fov: Self::Scalar,
    aspect: Self::Scalar,
    near: Self::Scalar,
    far: Self::Scalar,
  ) -> Self
  where
    Self::Scalar: FloatLike,
  {
    let _0 = Self::Scalar::from_f32(0.0);
    let _1 = Self::Scalar::from_f32(1.0);
    let _2 = Self::Scalar::from_f32(2.0);

    // f = 1.0 / tan(fov / 2.0)
    let half_fov = fov / _2;
    let f = half_fov.tan().reciprocal(); // cotangent
    let neg_depth = near - far;

    let c0 = Self::Vector::from_components(f / aspect, _0, _0, _0);
    let c1 = Self::Vector::from_components(_0, f, _0, _0);
    let c2 = Self::Vector::from_components(_0, _0, (far + near) / neg_depth, -_1);
    let c3 = Self::Vector::from_components(_0, _0, _2 * far * near / neg_depth, _0);

    Self::from_columns(c0, c1, c2, c3)
  }

  /// Creates a right-handed view matrix
  fn look_at<VecType3>(eye: VecType3, center: VecType3, up: VecType3) -> Self
  where
    // scalar type must be the same
    VecType3: Vector3<Scalar = Self::Scalar>,
    // vector's `normalize` requires the scalar to be float-like
    Self::Scalar: FloatLike,
  {
    let _0 = <Self::Scalar as MulAddIdentity>::zero();
    let _1 = <Self::Scalar as MulAddIdentity>::one();

    // forward given by target (center) - start (eye), cross with up to get side, and adjust up
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);

    let c0 = Self::Vector::from_components(s.x(), u.x(), -f.x(), _0);
    let c1 = Self::Vector::from_components(s.y(), u.y(), -f.y(), _0);
    let c2 = Self::Vector::from_components(s.z(), u.z(), -f.z(), _0);
    let c3 = Self::Vector::from_components(-s.dot(eye), -u.dot(eye), f.dot(eye), _1);

    Self::from_columns(c0, c1, c2, c3)
  }

  /// Creates an orthographic projection matrix
  fn orthographic(
    left: Self::Scalar,
    right: Self::Scalar,
    bottom: Self::Scalar,
    top: Self::Scalar,
    near: Self::Scalar,
    far: Self::Scalar,
  ) -> Self
  where
    Self::Scalar: FloatLike,
  {
    let _0 = <Self::Scalar as MulAddIdentity>::zero();
    let _1 = <Self::Scalar as MulAddIdentity>::one();
    let _2 = <Self::Scalar as FloatLike>::from_f32(2.0);

    let c0 = Self::Vector::from_components(_2 / (right - left), _0, _0, _0);
    let c1 = Self::Vector::from_components(_0, _2 / (top - bottom), _0, _0);
    let c2 = Self::Vector::from_components(_0, _0, _2 / (near - far), _0);
    let c3 = Self::Vector::from_components(
      -(right + left) / (right - left),
      -(top + bottom) / (top - bottom),
      -(far + near) / (far - near),
      _1,
    );
    Self::from_columns(c0, c1, c2, c3)
  }
}

pub mod mat4;
