use core::{cmp, ops};
use crate::math::{
  floating::{FloatBits, FloatOps},
  scalar_interval::FloatInterval,
  vector::Vector,
  FloatLike, Fma, FmaAssign, MulAddIdentity,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorInterval<V: Vector> {
  pub low: V,
  pub high: V,
}

impl<V> PartialOrd for VectorInterval<V>
where
  V: Vector,
  V::Scalar: PartialOrd,
{
  fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
    let mut is_less = true;
    let mut is_greater = true;
    for i in 0..V::DIM {
      let self_high = unsafe { self.high.component_unchecked(i) };
      let other_low = unsafe { other.low.component_unchecked(i) };
      if self_high >= other_low {
        is_less = false;
      }
      let self_low = unsafe { self.low.component_unchecked(i) };
      let other_high = unsafe { other.high.component_unchecked(i) };
      if self_low <= other_high {
        is_greater = false;
      }
    }
    if is_less {
      Some(cmp::Ordering::Less)
    } else if is_greater {
      Some(cmp::Ordering::Greater)
    } else if self.low == other.low && self.high == other.high {
      Some(cmp::Ordering::Equal)
    } else {
      None
    }
  }
}

impl<V> Fma for VectorInterval<V>
where
  V: Vector,
  Self: ops::Mul<Output = Self> + ops::Add<Output = Self>,
{
  fn fma(self, b: Self, c: Self) -> Self {
    self * b + c
  }
}

impl<V> FmaAssign for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatOps + FloatBits,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
  Self: Fma + Copy,
{
  fn fma_assign(&mut self, a: Self, b: Self) {
    *self = self.fma(a, b);
  }
}

impl<V> MulAddIdentity for VectorInterval<V>
where
  V: Vector,
  V::Scalar: MulAddIdentity,
{
  fn one() -> Self {
    Self {
      low: V::splat(V::Scalar::one()),
      high: V::splat(V::Scalar::one()),
    }
  }
  fn zero() -> Self {
    Self {
      low: V::splat(V::Scalar::zero()),
      high: V::splat(V::Scalar::zero()),
    }
  }
}

impl<V> ops::Add for VectorInterval<V>
where
  V: Vector + ops::Add<Output = V>,
{
  type Output = Self;

  fn add(self, rhs: Self) -> Self::Output {
    Self {
      low: self.low + rhs.low,
      high: self.high + rhs.high,
    }
  }
}

impl<V> ops::AddAssign for VectorInterval<V>
where
  V: Vector + ops::AddAssign,
{
  fn add_assign(&mut self, rhs: Self) {
    self.low += rhs.low;
    self.high += rhs.high;
  }
}

impl<V> ops::Sub for VectorInterval<V>
where
  V: Vector + ops::Sub<Output = V>,
{
  type Output = Self;

  fn sub(self, rhs: Self) -> Self::Output {
    Self {
      low: self.low - rhs.high,
      high: self.high - rhs.low,
    }
  }
}

impl<V> ops::SubAssign for VectorInterval<V>
where
  V: Vector + ops::SubAssign + ops::AddAssign,
  Self: ops::Sub<Output = Self>,
{
  fn sub_assign(&mut self, rhs: Self) {
    *self = *self - rhs;
  }
}

impl<V> ops::Neg for VectorInterval<V>
where
  V: Vector + ops::Neg<Output = V>,
{
  type Output = Self;

  fn neg(self) -> Self::Output {
    Self {
      low: -self.high,
      high: -self.low,
    }
  }
}

// TODO: SIMD implementation
impl<V> ops::Mul for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatOps + FloatBits,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
{
  type Output = Self;

  fn mul(self, rhs: Self) -> Self::Output {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let c = unsafe { rhs.low.component_unchecked(i) };
      let d = unsafe { rhs.high.component_unchecked(i) };

      let interval_a = FloatInterval { low: a, high: b };
      let interval_b = FloatInterval { low: c, high: d };
      let result = interval_a * interval_b;
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }
}

impl<V> ops::MulAssign for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatOps + FloatBits,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
  Self: ops::Mul<Output = Self>,
{
  fn mul_assign(&mut self, rhs: Self) {
    *self = *self * rhs;
  }
}

// TODO: SIMD implementation
impl<V> ops::Div for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatOps + FloatBits,
  FloatInterval<V::Scalar>: MulAddIdentity,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
{
  type Output = Self;

  fn div(self, rhs: Self) -> Self::Output {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let c = unsafe { rhs.low.component_unchecked(i) };
      let d = unsafe { rhs.high.component_unchecked(i) };

      let interval_a = FloatInterval { low: a, high: b };
      let interval_b = FloatInterval { low: c, high: d };
      let result = interval_a / interval_b;
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }
}

impl<V> ops::DivAssign for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatOps + FloatBits,
  FloatInterval<V::Scalar>: MulAddIdentity,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
  Self: ops::Div<Output = Self>,
{
  fn div_assign(&mut self, rhs: Self) {
    *self = *self / rhs;
  }
}

impl<V> FloatLike for VectorInterval<V>
where
  V: Vector,
  V::Scalar: FloatLike + FloatOps + FloatBits,
  <V::Scalar as crate::math::floating::BitsStorage>::Bits: crate::math::MulAddIdentity,
  FloatInterval<V::Scalar>: FloatLike + MulAddIdentity,
{
  fn is_nan(self) -> bool {
    (0..V::DIM).any(|i| unsafe {
      self.low.component_unchecked(i).is_nan() || self.high.component_unchecked(i).is_nan()
    })
  }

  fn is_infinite(self) -> bool {
    (0..V::DIM).any(|i| unsafe {
      self.low.component_unchecked(i).is_infinite()
        || self.high.component_unchecked(i).is_infinite()
    })
  }

  fn is_sign_negative(self) -> bool {
    (0..V::DIM).all(|i| unsafe { self.high.component_unchecked(i).is_sign_negative() })
  }

  fn is_sign_positive(self) -> bool {
    (0..V::DIM).all(|i| unsafe { self.low.component_unchecked(i).is_sign_positive() })
  }

  fn is_finite(self) -> bool {
    (0..V::DIM).all(|i| unsafe {
      self.low.component_unchecked(i).is_finite() && self.high.component_unchecked(i).is_finite()
    })
  }

  fn is_subnormal(self) -> bool {
    (0..V::DIM).any(|i| unsafe {
      self.low.component_unchecked(i).is_subnormal()
        || self.high.component_unchecked(i).is_subnormal()
    })
  }

  fn is_normal(self) -> bool {
    (0..V::DIM).all(|i| unsafe {
      self.low.component_unchecked(i).is_normal() && self.high.component_unchecked(i).is_normal()
    })
  }

  fn from_f32(num: f32) -> Self {
    Self {
      low: V::splat(V::Scalar::from_f32(num)),
      high: V::splat(V::Scalar::from_f32(num)),
    }
  }

  fn squared(self) -> Self {
    self * self
  }

  fn sqrt(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).sqrt();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn cos(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).cos();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn sin(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).sin();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn reciprocal(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).reciprocal();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn tan(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).tan();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn pow(self, v: Self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let c = unsafe { v.low.component_unchecked(i) };
      let d = unsafe { v.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).pow(FloatInterval { low: c, high: d });
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn exp(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).exp();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn ln(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).ln();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }

  fn floor(self) -> Self {
    let mut low = V::zero();
    let mut high = V::zero();
    for i in 0..V::DIM {
      let a = unsafe { self.low.component_unchecked(i) };
      let b = unsafe { self.high.component_unchecked(i) };
      let result = (FloatInterval { low: a, high: b }).floor();
      low.set_component(i, result.low);
      high.set_component(i, result.high);
    }
    Self { low, high }
  }
}
