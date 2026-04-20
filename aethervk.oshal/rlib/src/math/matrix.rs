use core::ops;

use crate::math::{
  FloatLike, MulAddIdentity, Scalar,
  floating::{FloatBits, FloatOps},
  quaternion::Quaternion,
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
  /// column-major order
  fn from_array(x: &[Self::Scalar; 9]) -> Self;

  fn from_outer_product(v0: Self::Vector, v1: Self::Vector) -> Self {
    Self::from_columns(v0 * v1.x(), v0 * v1.y(), v0 * v1.z())
  }

  fn from_outer_self(v: Self::Vector) -> Self {
    Self::from_outer_product(v, v)
  }

  fn x(&self) -> Self::Vector {
    unsafe { self.column_unchecked(0) }
  }

  fn y(&self) -> Self::Vector {
    unsafe { self.column_unchecked(1) }
  }

  fn z(&self) -> Self::Vector {
    unsafe { self.column_unchecked(2) }
  }

  /// Columns are orthonormal (unit length + perpendicular)
  /// Determinant is +1 (not -1 → excludes reflections)
  /// No scaling or shear
  fn is_pure_rotation(&self) -> bool
  where
    Self::Scalar: FloatOps + FloatBits,
  {
    let eps = Self::Scalar::from_f32(1e-5);

    let x = unsafe { self.column_unchecked(0) };
    let y = unsafe { self.column_unchecked(1) };
    let z = unsafe { self.column_unchecked(2) };

    // 1. Unit length (||v|| ≈ 1)
    let _1 = Self::Scalar::from_f32(1.0);
    let unit =
      (x.dot(x) - _1).abs() <= eps && (y.dot(y) - _1).abs() <= eps && (z.dot(z) - _1).abs() <= eps;

    // 2. Orthogonality (dot ≈ 0)
    let orthogonal = x.dot(y).abs() <= eps && x.dot(z).abs() <= eps && y.dot(z).abs() <= eps;

    // 3. Right-handed (determinant ≈ +1)
    let det = self.determinant();
    let proper = (det - _1).abs() <= eps;

    unit && orthogonal && proper
  }

  /// No scaling or shear, but using a permissive epsilon to tolerate physics/eigen decomposition float drift.
  fn is_pure_rotation_permissive(&self) -> bool
  where
    Self::Scalar: FloatOps + FloatBits,
  {
    let eps = Self::Scalar::from_f32(1e-2);

    let x = unsafe { self.column_unchecked(0) };
    let y = unsafe { self.column_unchecked(1) };
    let z = unsafe { self.column_unchecked(2) };

    // 1. Unit length (||v|| ≈ 1)
    let _1 = Self::Scalar::from_f32(1.0);
    let unit =
      (x.dot(x) - _1).abs() <= eps && (y.dot(y) - _1).abs() <= eps && (z.dot(z) - _1).abs() <= eps;

    // 2. Orthogonality (dot ≈ 0)
    let orthogonal = x.dot(y).abs() <= eps && x.dot(z).abs() <= eps && y.dot(z).abs() <= eps;

    // 3. Right-handed (determinant ≈ +1)
    let det = self.determinant();
    let proper = (det - _1).abs() <= eps;

    unit && orthogonal && proper
  }
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

  /// column-major order
  fn from_array(x: &[Self::Scalar; 16]) -> Self;

  /// Creates a translation matrix.
  fn translation<V3>(v: V3) -> Self
  where
    V3: Vector3<Scalar = Self::Scalar>,
  {
    let id = Self::identity();
    // Safety: Matrix4 is guaranteed to be 4x4, so indices 0..3 are always valid
    unsafe {
      Self::from_columns(
        id.column_unchecked(0),
        id.column_unchecked(1),
        id.column_unchecked(2),
        <Self::Vector as Vector4>::from_components(
          v.x(),
          v.y(),
          v.z(),
          <Self::Scalar as MulAddIdentity>::one(),
        ),
      )
    }
  }

  /// Creates a standard right-handed perspective projection matrix
  /// This is the general OpenGL column-major matrix formula from scratchapixel with
  /// special case left = -right, top = -bottom
  fn perspective_gl(
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

    let c0 = Self::Vector::from_components(f / aspect, _0, _0, _0);
    let c1 = Self::Vector::from_components(_0, f, _0, _0);
    // OpenGL: Map z from [near, far] to [-1, 1]
    let c2 = Self::Vector::from_components(_0, _0, (far + near) / (near - far), -_1);
    let c3 = Self::Vector::from_components(_0, _0, _2 * far * near / (near - far), _0);

    Self::from_columns(c0, c1, c2, c3)
  }

  fn perspective_vk(
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

    let c0 = Self::Vector::from_components(f / aspect, _0, _0, _0);
    // Vulkan: Invert Y to fix upside down projection when using Z-up RH view space
    let c1 = Self::Vector::from_components(_0, _0 - f, _0, _0);
    // Vulkan: Map z from [near, far] to [0, 1]
    let c2 = Self::Vector::from_components(_0, _0, far / (near - far), -_1);
    let c3 = Self::Vector::from_components(_0, _0, far * near / (near - far), _0);

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

    let f = (center - eye).normalize();
    let s = up.cross(f).normalize();
    let u = f.cross(s);

    let c0 = Self::Vector::from_components(s.x(), s.y(), s.z(), -s.dot(eye));
    let c1 = Self::Vector::from_components(u.x(), u.y(), u.z(), -u.dot(eye));
    let c2 = Self::Vector::from_components(-f.x(), -f.y(), -f.z(), f.dot(eye));
    let c3 = Self::Vector::from_components(_0, _0, _0, _1);

    Self::from_columns(c0, c1, c2, c3).transpose()
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

  /// Creates a Vulkan-ready orthographic projection matrix
  fn orthographic_vk(
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
    let c1 = Self::Vector::from_components(_0, _0 - _2 / (top - bottom), _0, _0); // Y flipped
    // Vulkan Z is [0, 1]
    let c2 = Self::Vector::from_components(_0, _0, -_1 / (far - near), _0);
    let c3 = Self::Vector::from_components(
      -(right + left) / (right - left),
      -(top + bottom) / (top - bottom),
      -near / (far - near),
      _1,
    );
    Self::from_columns(c0, c1, c2, c3)
  }

  fn from_mat3<M, V>(m: M) -> Self
  where
    M: Matrix3<Scalar = Self::Scalar, Vector = V>,
    V: Vector3<Scalar = Self::Scalar>,
  {
    let _0 = <Self::Scalar as MulAddIdentity>::zero();
    let _1 = <Self::Scalar as MulAddIdentity>::one();
    unsafe {
      let c0 = Self::Vector::from_components(
        m.column_unchecked(0).x(),
        m.column_unchecked(1).x(),
        m.column_unchecked(2).x(),
        _0,
      );
      let c1 = Self::Vector::from_components(
        m.column_unchecked(0).y(),
        m.column_unchecked(1).y(),
        m.column_unchecked(2).y(),
        _0,
      );
      let c2 = Self::Vector::from_components(
        m.column_unchecked(0).z(),
        m.column_unchecked(1).z(),
        m.column_unchecked(2).z(),
        _0,
      );
      let c3 = Self::Vector::from_components(_0, _0, _0, _1);
      Self::from_columns(c0, c1, c2, c3)
    }
  }

  fn from_quat_and_axes<Q, V>(q: Q, right: V, forward: V, up: V) -> Self
  where
    Q: Quaternion<Scalar = Self::Scalar, Vector = V>,
    V: Vector3<Scalar = Self::Scalar>,
  {
    let s = q.scalar_part();
    let v = q.vector_part();
    let x = v.x();
    let y = v.y();
    let z = v.z();

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = s * x2;
    let wy = s * y2;
    let wz = s * z2;

    let one = <Self::Scalar as MulAddIdentity>::one();
    let zero = <Self::Scalar as MulAddIdentity>::zero();

    // 1. Calculate standard purely-mathematical rotation matrix components
    let m00 = one - (yy + zz);
    let m10 = xy + wz;
    let m20 = xz - wy;

    let m01 = xy - wz;
    let m11 = one - (xx + zz);
    let m21 = yz + wx;

    let m02 = xz + wy;
    let m12 = yz - wx;
    let m22 = one - (xx + yy);

    // 2. Helper closure to rotate an arbitrary axis by the quaternion
    let rotate = |axis: V| {
      let ax = axis.x();
      let ay = axis.y();
      let az = axis.z();
      Self::Vector::from_components(
        m00 * ax + m01 * ay + m02 * az,
        m10 * ax + m11 * ay + m12 * az,
        m20 * ax + m21 * ay + m22 * az,
        <Self::Scalar as MulAddIdentity>::zero(),
      )
    };

    // 3. Construct matrix columns from our properly rotated custom axes
    Self::from_columns(
      rotate(right),   // Col 0
      rotate(forward), // Col 1
      rotate(up),      // Col 2
      Self::Vector::from_components(zero, zero, zero, one),
    )
  }

  fn from_quat_custom_frame<Q, V>(q: Q) -> Self
  where
    Q: Quaternion<Scalar = Self::Scalar, Vector = V>,
    V: Vector3<Scalar = Self::Scalar>,
  {
    let s = q.scalar_part();
    let v = q.vector_part();
    let x = v.x();
    let y = v.y();
    let z = v.z();

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = s * x2;
    let wy = s * y2;
    let wz = s * z2;

    let one = <Self::Scalar as MulAddIdentity>::one();
    let zero = <Self::Scalar as MulAddIdentity>::zero();

    Self::from_columns(
      // Column 0: Right (+x)
      Self::Vector::from_components(one - (yy + zz), xy + wz, xz - wy, zero),
      // Column 1: Forward (-y)
      Self::Vector::from_components(xy - wz, one - (xx + zz), yz + wx, zero),
      // Column 2: Up (+z)
      Self::Vector::from_components(xz + wy, yz - wx, one - (xx + yy), zero),
      // Column 3: Translation/W
      Self::Vector::from_components(zero, zero, zero, one),
    )
  }

  fn from_scale<V>(v: V) -> Self
  where
    V: Vector3<Scalar = Self::Scalar>,
  {
    let _0 = <Self::Scalar as MulAddIdentity>::zero();
    let _1 = <Self::Scalar as MulAddIdentity>::one();
    let c0 = Self::Vector::from_components(v.x(), _0, _0, _0);
    let c1 = Self::Vector::from_components(_0, v.y(), _0, _0);
    let c2 = Self::Vector::from_components(_0, _0, v.z(), _0);
    let c3 = Self::Vector::from_components(_0, _0, _0, _1);
    Self::from_columns(c0, c1, c2, c3)
  }
}

pub mod mat3;
pub mod mat4;
