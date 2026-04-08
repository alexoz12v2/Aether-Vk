use core::{ops};

use crate::math::{FloatLike, MulAddIdentity, Scalar, floating::FloatOps};

// shuffle is unstable
#[cfg(target_arch = "x86_64")]
#[allow(non_snake_case)]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
  ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

// TODO: Boolean vector, boolean comparison, masked ops, ...

// Note: Indexing is to be implemented by concrete types, not strictly required here
// Note: many functions take the vector by value as this is supposed to be a small dimensional vector
pub trait Vector:
  Copy
  + Sized
  + PartialEq
  + ops::Add<Output = Self>
  + ops::Sub<Output = Self>
  + ops::Neg<Output = Self>
  + ops::Mul<Self::Scalar, Output = Self>
  + ops::Div<Self::Scalar, Output = Self>
  + ops::Mul<Self, Output = Self>
  + ops::Div<Self, Output = Self>
  + ops::AddAssign
  + ops::SubAssign
{
  type Scalar: Scalar;

  const DIM: usize;

  fn zero() -> Self;
  fn splat(v: Self::Scalar) -> Self;

  fn is_zero(self) -> bool {
    self == Self::zero()
  }
  // TODO: epsilonEqual

  fn component(&self, i: usize) -> Option<Self::Scalar>;
  /// Safety: index should be less than `DIM`
  unsafe fn component_unchecked(&self, i: usize) -> Self::Scalar;
  // does nothing if out of bounds
  fn set_component(&mut self, i: usize, value: Self::Scalar);

  fn dot(self, rhs: Self) -> Self::Scalar;
  fn length_squared(self) -> Self::Scalar {
    self.dot(self)
  }
  fn length(self) -> Self::Scalar
  where
    Self::Scalar: FloatLike,
  {
    self.length_squared().sqrt()
  }
  fn normalize(self) -> Self
  where
    Self::Scalar: FloatLike,
  {
    self / self.length()
  }
  fn lerp(a: Self, b: Self, t: Self::Scalar) -> Self {
    a + (b - a) * t
  }
  fn min(self, other: Self) -> Self;
  fn max(self, other: Self) -> Self;
}

pub trait Vector2: Vector
where
  Self::Scalar: Scalar,
{
  fn from_components(x: Self::Scalar, y: Self::Scalar) -> Self;
  fn x(&self) -> Self::Scalar;
  fn y(&self) -> Self::Scalar;

  fn to_vec3<V>(self, z: Self::Scalar) -> V
  where
    V: Vector3<Scalar = Self::Scalar>,
  {
    V::from_components(self.x(), self.y(), z)
  }

  fn to_vec4<V>(self, z: Self::Scalar, w: Self::Scalar) -> V
  where
    V: Vector4<Scalar = Self::Scalar>,
  {
    V::from_components(self.x(), self.y(), z, w)
  }
}

pub trait Vector3: Vector
where
  Self::Scalar: Scalar,
{
  fn from_components(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar) -> Self;
  fn x(&self) -> Self::Scalar;
  fn y(&self) -> Self::Scalar;
  fn z(&self) -> Self::Scalar;

  /// Cross product: (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  fn cross(self, rhs: Self) -> Self;
  fn reflect(self, normal: Self) -> Self
  where
    Self::Scalar: FloatOps,
  {
    self - normal * (self.dot(normal) * <Self::Scalar as FloatOps>::from_i32(2))
  }
  // TODO refract
  fn face_forward(self, normal: Self) -> Self
  where
    Self::Scalar: FloatLike,
  {
    if self.dot(normal) < <Self::Scalar as MulAddIdentity>::zero() {
      -self
    } else {
      self
    }
  }

  fn to_vec4<V>(self, w: Self::Scalar) -> V
  where
    V: Vector4<Scalar = Self::Scalar>,
  {
    V::from_components(self.x(), self.y(), self.z(), w)
  }
}

pub trait Vector4: Vector
where
  Self::Scalar: Scalar,
{
  fn from_components(x: Self::Scalar, y: Self::Scalar, z: Self::Scalar, w: Self::Scalar) -> Self;
  fn x(&self) -> Self::Scalar;
  fn y(&self) -> Self::Scalar;
  fn z(&self) -> Self::Scalar;
  fn w(&self) -> Self::Scalar;
}

pub mod vec2;
pub mod vec3;
pub mod vec4;
