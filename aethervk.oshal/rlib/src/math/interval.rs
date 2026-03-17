use core::{cmp, ops};

use crate::math::{
  FloatLike, Fma, FmaAssign, MulAddIdentity, Scalar,
  floating::{BitsStorage, FloatBits, FloatOps},
};

pub(super) trait Interval<T: FloatOps + FloatBits>:
  Scalar + ops::Index<usize, Output = T> + ops::IndexMut<usize, Output = T>
{
  fn new(low: T, high: T) -> Self;
  fn from_scalar(value: T) -> Self;
  fn from_value_and_error(value: T, error: T) -> Self;

  fn upper_bound(&self) -> T;
  fn lower_bound(&self) -> T;

  fn midpoint(&self) -> T;
  fn width(&self) -> T;

  fn contains(&self, v: T) -> bool;
  fn includes(&self, other: &Self) -> bool;
  fn overlaps(&self, other: &Self) -> bool;

  fn add_scalar(&self, v: T) -> Self;
  fn add_assign_scalar(&mut self, v: T) {
    *self = self.add_scalar(v)
  }
  fn sub_scalar(&self, v: T) -> Self;
  fn rsub_scalar(v: T, i: &Self) -> Self;
  fn sub_assign_scalar(&mut self, v: T) {
    *self = self.sub_scalar(v)
  }
  fn mul_scalar(&self, v: T) -> Self;
  fn mul_assign_scalar(&mut self, v: T) {
    *self = self.mul_scalar(v)
  }
  fn div_scalar(&self, v: T) -> Self;
  fn rdiv_scalar(v: T, i: &Self) -> Self;
  fn div_assign_scalar(&mut self, v: T) {
    *self = self.add_scalar(v)
  }
  // TODO: has_nan, has_max, has_min, ...
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FloatInterval<T: FloatOps + FloatBits> {
  pub low: T,
  pub high: T,
}

impl<T> ops::Index<usize> for FloatInterval<T>
where
  T: FloatOps + FloatBits,
{
  type Output = T;

  fn index(&self, index: usize) -> &Self::Output {
    debug_assert!(index == 0 || index == 1);
    match index {
      0 => &self.low,
      _ => &self.high,
    }
  }
}
impl<T> ops::IndexMut<usize> for FloatInterval<T>
where
  T: FloatOps + FloatBits,
{
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    debug_assert!(index == 0 || index == 1);
    match index {
      0 => &mut self.low,
      _ => &mut self.high,
    }
  }
}
impl<T> PartialOrd for FloatInterval<T>
where
  T: FloatOps + FloatBits,
{
  // comparison well defined for non overlapping intervals
  fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
    // handle potential NaN values natively
    if self.high < other.low {
      // [self] ... [other]
      Some(cmp::Ordering::Less)
    } else if self.low > other.high {
      // [other] ... [low]
      Some(cmp::Ordering::Greater)
    } else if self.low == other.low && self.high == other.high {
      // [ self & other ]
      Some(cmp::Ordering::Equal)
    } else {
      // Intervals overlap, or bounds involve NaN
      None
    }
  }
}
impl<T> Fma for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
{
  fn fma(self, b: Self, c: Self) -> Self
  where
    Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self>,
  {
    // TODO: allow subnormal from some cached subnormal configuration (status register?)
    Self {
      low: self.low.fma_round_down(b.low, c.low, false),
      high: self.high.fma_round_up(b.high, c.high, false),
    }
  }
}
impl<T> FmaAssign for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  Self: Fma,
{
  fn fma_assign(&mut self, a: Self, b: Self)
  where
    Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self> + Copy,
  {
    *self = self.fma(a, b);
  }
}
impl<T> ops::Neg for FloatInterval<T>
where
  T: FloatOps + FloatBits,
{
  type Output = Self;

  fn neg(self) -> Self::Output {
    Self {
      low: -self.high,
      high: -self.low,
    }
  }
}
impl<T> ops::SubAssign for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  Self: ops::Sub<Output = Self>,
{
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}
impl<T> ops::Sub for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output {
    // TODO: allow subnormal from some cached subnormal configuration (status register?)
    Self {
      low: self.low.sub_round_down(rhs.low, false),
      high: self.high.sub_round_up(rhs.high, false),
    }
  }
}
impl<T> ops::DivAssign for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  Self: ops::Div<Output = Self>,
{
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs;
  }
}
impl<T> ops::Div for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
  Self: MulAddIdentity,
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self::Output {
    // TODO: allow subnormal from some cached subnormal configuration (status register?)
    if rhs.contains(<T as FloatOps>::from_i32(0)) {
      Self {
        low: <T as FloatOps>::NEG_INFINITY,
        high: <T as FloatOps>::INFINITY,
      }
    } else {
      let low_quot = [
        self.low.div_round_down(rhs.low, false),
        self.high.div_round_down(rhs.low, false),
        self.low.div_round_down(rhs.high, false),
        self.high.div_round_down(rhs.high, false),
      ];
      let high_quot = [
        self.low.div_round_up(rhs.low, false),
        self.high.div_round_up(rhs.low, false),
        self.low.div_round_up(rhs.high, false),
        self.high.div_round_up(rhs.high, false),
      ];
      Self {
        low: v_min!(low_quot[0], low_quot[1], low_quot[2], low_quot[3]),
        high: v_max!(high_quot[0], high_quot[1], high_quot[2], high_quot[3]),
      }
    }
  }
}
impl<T> ops::MulAssign for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  Self: ops::Mul<Output = Self>,
{
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs
  }
}
impl<T> ops::Mul for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
{
  type Output = Self;

  fn mul(self, rhs: Self) -> Self::Output {
    // TODO: allow subnormal from some cached subnormal configuration (status register?)
    let low_mult = [
      self.low.mul_round_down(rhs.low, false),
      self.high.mul_round_down(rhs.low, false),
      self.low.mul_round_down(rhs.high, false),
      self.high.mul_round_down(rhs.high, false),
    ];
    let high_mult = [
      self.low.mul_round_up(rhs.low, false),
      self.high.mul_round_up(rhs.low, false),
      self.low.mul_round_up(rhs.high, false),
      self.high.mul_round_up(rhs.high, false),
    ];
    Self {
      low: v_min!(low_mult[0], low_mult[1], low_mult[2], low_mult[3]),
      high: v_max!(high_mult[0], high_mult[1], high_mult[2], high_mult[3]),
    }
  }
}
impl<T> ops::AddAssign for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  Self: ops::Add<Output = Self>,
{
  fn add_assign(&mut self, rhs: Self) {
    *self = *self + rhs;
  }
}
impl<T> ops::Add for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output {
    // TODO: allow subnormal from some cached subnormal configuration (status register?)
    Self {
      low: self.low.add_round_down(rhs.low, false),
      high: self.high.add_round_up(rhs.high, false),
    }
  }
}

impl<T> Interval<T> for FloatInterval<T>
where
  T: FloatOps + FloatBits,
  <T as BitsStorage>::Bits: MulAddIdentity,
  Self: MulAddIdentity,
{
  #[inline]
  fn new(low: T, high: T) -> Self {
    Self { low, high }
  }
  #[inline]
  fn from_scalar(value: T) -> Self {
    Self {
      low: value,
      high: value,
    }
  }
  #[inline]
  fn from_value_and_error(value: T, error: T) -> Self {
    if error == <T as FloatOps>::from_i32(0) {
      Self {
        low: value,
        high: value,
      }
    } else {
      Self {
        // TODO: allow subnormal from some cached subnormal configuration (status register?)
        low: value.sub_round_down(error, false),
        high: value.add_round_up(error, false),
      }
    }
  }
  #[inline]
  fn upper_bound(&self) -> T {
    self.high
  }
  #[inline]
  fn lower_bound(&self) -> T {
    self.low
  }
  fn midpoint(&self) -> T {
    (self.low + self.high) / <T as FloatOps>::from_i32(2)
  }
  #[inline]
  fn width(&self) -> T {
    self.high - self.low
  }
  #[inline]
  fn includes(&self, other: &Self) -> bool {
    self.lower_bound() <= other.lower_bound() && self.upper_bound() >= other.upper_bound()
  }
  #[inline]
  fn contains(&self, v: T) -> bool {
    v >= self.lower_bound() && v <= self.upper_bound()
  }
  #[inline]
  fn overlaps(&self, other: &Self) -> bool {
    self.upper_bound() >= other.lower_bound() || self.lower_bound() <= other.upper_bound()
  }
  #[inline]
  fn add_scalar(&self, v: T) -> Self {
    *self + Self::from_scalar(v)
  }
  #[inline]
  fn sub_scalar(&self, v: T) -> Self {
    *self - Self::from_scalar(v)
  }
  #[inline]
  fn rsub_scalar(v: T, i: &Self) -> Self {
    Self::from_scalar(v) - *i
  }
  #[inline]
  fn mul_scalar(&self, v: T) -> Self {
    *self * Self::from_scalar(v)
  }
  #[inline]
  fn div_scalar(&self, v: T) -> Self {
    *self / Self::from_scalar(v)
  }
  #[inline]
  fn rdiv_scalar(v: T, i: &Self) -> Self {
    Self::from_scalar(v) / *i
  }
}

impl<T> MulAddIdentity for FloatInterval<T>
where
  T: FloatBits + FloatOps,
{
  const ONE: Self = Self {
    low: T::ZERO,
    high: T::ZERO,
  };
  const ZERO: Self = Self {
    low: T::ONE,
    high: T::ONE,
  };
}

impl<T> FloatLike for FloatInterval<T>
where
  T: FloatBits + FloatOps,
  <T as BitsStorage>::Bits: MulAddIdentity,
{
  fn is_nan(self) -> bool {
    self.high.is_nan() || self.low.is_nan()
  }

  fn is_infinite(self) -> bool {
    self.high.is_infinite() || self.low.is_infinite()
  }

  fn is_sign_negative(self) -> bool {
    self.high.is_sign_negative()
  }

  fn is_sign_positive(self) -> bool {
    self.low.is_sign_positive()
  }

  fn is_finite(self) -> bool {
    self.low.is_finite() && self.high.is_finite()
  }

  fn is_subnormal(self) -> bool {
    self.low.is_subnormal() || self.high.is_subnormal()
  }

  fn is_normal(self) -> bool {
    self.low.is_normal() && self.high.is_normal()
  }

  fn from_f32(num: f32) -> Self {
    <Self as Interval<T>>::from_scalar(<T as FloatLike>::from_f32(num))
  }

  fn squared(self) -> Self {
    self * self
  }

  fn sqrt(self) -> Self {
    todo!()
  }

  fn cos(self) -> Self {
    todo!()
  }

  fn sin(self) -> Self {
    todo!()
  }

  fn reciprocal(self) -> Self {
    Self::from_scalar(T::from_i32(1)) / self
  }

  fn tan(self) -> Self {
    todo!()
  }
}

pub(super) type Interval32 = FloatInterval<f32>;
pub(super) type Interval64 = FloatInterval<f64>;
