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
    Self::from_vector_and_scalar(vector, <Self::Scalar as MulAddIdentity>::zero())
  }

  fn identity() -> Self {
    Self::from_vector_and_scalar(Self::Vector::zero(), Self::Scalar::one())
  }

  fn from_axis_angle(axis: Self::Vector, angle: Self::Scalar) -> Self {
    let half_angle = angle / Self::Scalar::from_f32(2f32);
    Self::from_vector_and_scalar(axis * half_angle.sin(), half_angle.cos())
  }

  fn conjugate(self) -> Self {
    Self::from_vector_and_scalar(-self.vector_part(), self.scalar_part())
  }

  fn norm_squared(self) -> Self::Scalar {
    (self.conjugate() * self).scalar_part()
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
      _2 * s * (qi * qj + qk * qr),
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

  fn pow(self, t: Self::Scalar) -> Self
  where
    Self::Scalar: FloatLike + ops::Mul<Self::Vector, Output = Self::Vector>,
  {
    // A simplified valid quaternion power
    let v = self.vector_part();
    let a = self.scalar_part();
    let v_norm = v.length();
    let theta = a.acos();
    let new_theta = theta * t;
    let new_a = new_theta.cos();
    let new_v = if v_norm > Self::Scalar::from_f32(1e-6) {
      v * (new_theta.sin() / v_norm)
    } else {
      Self::Vector::zero()
    };
    Self::from_vector_and_scalar(new_v, new_a)
  }

  fn slerp(mut a: Self, b: Self, t: Self::Scalar) -> Self
  where
    Self::Scalar: FloatLike + ops::Mul<Self::Vector, Output = Self::Vector>,
  {
    let mut dot = a.vector_part().dot(b.vector_part()) + a.scalar_part() * b.scalar_part();

    // If dot < 0, SLERP will take the long way around.
    // We flip one quaternion to take the shortest path.
    let mut b_flipped = b;
    if dot < <Self::Scalar as MulAddIdentity>::zero() {
      dot = -dot;
      b_flipped = b * Self::Scalar::from_f32(-1.0);
    }

    // Standard linear interpolation if quaternions are extremely close
    // to avoid division by zero in sin(theta)
    if dot > Self::Scalar::from_f32(0.9995) {
      // Fallback to normalized LERP (Nlerp)
      let v = a.vector_part() * (Self::Scalar::one() - t) + b_flipped.vector_part() * t;
      let s = a.scalar_part() * (Self::Scalar::one() - t) + b_flipped.scalar_part() * t;
      return Self::from_vector_and_scalar(v, s).normalize();
    }

    let theta_0 = dot.acos();
    let theta = theta_0 * t;

    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();

    let s0 = (theta_0 - theta).sin() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;

    let res_v = (a.vector_part() * s0) + (b_flipped.vector_part() * s1);
    let res_s = (a.scalar_part() * s0) + (b_flipped.scalar_part() * s1);

    Self::from_vector_and_scalar(res_v, res_s)
  }
}
