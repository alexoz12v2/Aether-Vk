use core::ops;

use crate::math::{
  FloatLike, MulAddIdentity, Scalar,
  matrix::Matrix3,
  vector::{Vector3, Vector},
};

pub trait Quaternion:
  Copy
  + Sized
  + ops::Mul<Self, Output = Self>
  + ops::Mul<Self::Scalar, Output = Self>
  + ops::Div<Self::Scalar, Output = Self>
{
  type Scalar: Scalar + FloatLike;
  type Vector: Vector3<Scalar = Self::Scalar>;

  // everything stems from these 3 functions
  fn from_vector_and_scalar(vector: Self::Vector, scalar: Self::Scalar) -> Self;
  fn vector_part(self) -> Self::Vector;
  fn scalar_part(self) -> Self::Scalar;

  fn from_vector(vector: Self::Vector) -> Self {
    Self::from_vector_and_scalar(vector, <Self::Scalar as MulAddIdentity>::ZERO)
  }

  fn identity() -> Self {
    Self::from_vector_and_scalar(Self::Vector::zero(), Self::Scalar::ONE)
  }

  fn from_axis_angle(axis: Self::Vector, angle: Self::Scalar) -> Self {
    let half_angle = angle / Self::Scalar::from_f32(2f32);
    Self::from_vector_and_scalar(axis * half_angle.sin(), half_angle.cos())
  }

  fn conjugate(self) -> Self {
    Self::from_vector_and_scalar(-self.vector_part(), self.scalar_part())
  }

  fn norm(self) -> Self::Scalar {
    (self.conjugate() * self).scalar_part().sqrt()
  }

  fn normalize(self) -> Self {
    self / self.norm()
  }

  fn inverse(self) -> Self {
    self.conjugate() / self.norm().squared()
  }

  fn rotate_vector(self, v: Self::Vector) -> Self::Vector {
    let qv = Self::from_vector(v);
    let r = self * qv * self.conjugate();
    r.vector_part()
  }

  fn to_matrix3<T>(self) -> T
  where
    T: Matrix3<Scalar = Self::Scalar, Vector = Self::Vector>,
  {
    let s = self.norm().reciprocal().squared();
    let [qr, qi, qj, qk] = {
      let vec_part = self.vector_part();
      let scalar = self.scalar_part();
      [scalar, vec_part.x(), vec_part.y(), vec_part.z()]
    };
    let _1 = Self::Scalar::from_f32(1f32);
    let _2 = Self::Scalar::from_f32(2f32);
    let col0 = Self::Vector::from_components(
      _1 - _2 * s * (qj.squared() + qk.squared()),
      _2 * s * (qi * qj + qk + qr),
      _2 * (qi * qk - qj * qr),
    );
    let col1 = Self::Vector::from_components(
      _2 * s * (qi * qj - qk * qr),
      _1 - _2 * s * (qi.squared() + qk.squared()),
      _2 * s * (qi * qk + qi * qr),
    );
    let col2 = Self::Vector::from_components(
      _2 * s * (qi * qk + qj * qr),
      _2 * s * (qj * qk - qi * qr),
      _1 - _2 * s * (qi.squared() + qj.squared()),
    );
    T::from_columns(col0, col1, col2)
  }

  fn slerp(a: Self, b: Self, t: Self::Scalar) -> Self;
}
