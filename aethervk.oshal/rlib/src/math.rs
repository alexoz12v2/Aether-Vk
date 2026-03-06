use core::ops::{self, Index, IndexMut};

pub mod floating;

// ------------------------------------------ Traits ------------------------------------------
pub trait Scalar:
  Copy
  + Sized
  + ops::Add<Output = Self>
  + ops::AddAssign
  + ops::Mul<Output = Self>
  + ops::MulAssign
  + ops::Div<Output = Self>
  + ops::DivAssign
  + ops::Sub<Output = Self>
  + ops::SubAssign
  + PartialEq
  + PartialOrd
{
  fn fma(a: Self, b: Self, c: Self) -> Self;
  fn fma_assign(&mut self, a: Self, b: Self);
}

impl<T> Scalar for T
where
  T: Copy
    + Sized
    + ops::Add<Output = T>
    + ops::AddAssign
    + ops::Mul<Output = T>
    + ops::MulAssign
    + ops::Div<Output = T>
    + ops::DivAssign
    + ops::Sub<Output = T>
    + ops::SubAssign
    + PartialEq
    + PartialOrd,
{
  fn fma(a: Self, b: Self, c: Self) -> Self {
    a * b + c
  }

  fn fma_assign(&mut self, a: Self, b: Self) {
    *self = (*self) * a + b;
  }
}

pub trait FloatLike: Scalar + Copy {}
impl FloatLike for f32 {}
impl FloatLike for f64 {}

pub trait Interval<T: Scalar + FloatLike>: Scalar + FloatLike + Index<u32> + IndexMut<u32> {
  fn from_scalar(value: T) -> Self;
  fn from_value_and_error(value: T, error: T) -> Self;
  fn upper_bound(&self) -> T;
  fn lower_bound(&self) -> T;
  fn midpoint(&self) -> T
  where
    T: From<f32>,
  {
    (self.upper_bound() + self.lower_bound()) / From::from(2f32)
  }
  fn width(&self) -> T
  where
    T: From<f32>,
  {
    self.upper_bound() - self.lower_bound()
  }
  fn value_in_range(v: T, i: Self) -> bool {
    v >= i.lower_bound() && v <= i.upper_bound()
  }
  fn interval_in_range(a: Self, b: Self) -> bool {
    a.lower_bound() <= b.lower_bound() && a.upper_bound() >= b.lower_bound()
  }
  fn inside(&self, other: Self) -> bool {
    Self::interval_in_range(*self, other)
  }
}

// -------------------------------- Scalar Types ----------------------------------------------

// ----------------------------------- Vector Types -------------------------------------------

// ---------------------------------- Matrix Types --------------------------------------------

// ---------------------------------- Quaterion -----------------------------------------------
