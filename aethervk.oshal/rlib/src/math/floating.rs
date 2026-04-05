use core::{ops, f32, f64};

use crate::math::MulAddIdentity;

pub trait BitsStorage: Sized + Copy {
  type Bits: Sized
    + ops::Add<Output = Self::Bits>
    + ops::Sub<Output = Self::Bits>
    + ops::AddAssign
    + ops::SubAssign
    + ops::Shl<Output = Self::Bits>
    + ops::Shr<Output = Self::Bits>
    + ops::ShlAssign
    + ops::ShrAssign
    + ops::Mul<Output = Self::Bits>
    + ops::MulAssign
    + ops::BitAnd<Output = Self::Bits>
    + ops::BitOr<Output = Self::Bits>
    + ops::BitXor<Output = Self::Bits>
    + ops::BitAndAssign
    + ops::BitOrAssign
    + ops::BitXorAssign;

  const SIGN_BIT_MASK: Self::Bits;
}
impl BitsStorage for f32 {
  type Bits = u32;
  const SIGN_BIT_MASK: Self::Bits = 0x8000_0000;
}
impl BitsStorage for f64 {
  type Bits = u64;
  const SIGN_BIT_MASK: Self::Bits = 0x8000_0000_0000_0000;
}

pub trait FloatBits: super::FloatLike + Sized + BitsStorage {
  fn to_bits(self) -> Self::Bits;
  fn from_bits(bits: Self::Bits) -> Self;
}

macro_rules! impl_float_bits {
  ($($t:ty),*) => {
    $(
      impl FloatBits for $t {
        fn to_bits(self) -> Self::Bits {
          <$t>::to_bits(self)
        }
        fn from_bits(bits: Self::Bits) -> Self {
          <$t>::from_bits(bits)
        }
      }
    )*
  };
}
impl_float_bits!(f32, f64);

pub trait FloatOps: Sized + super::FloatLike {
  // ----------------------------- Constants -----------------------------
  const NEGATIVE_ZERO: Self; // used for branch cuts
  const PI: Self;
  const INV_PI: Self;
  const PI_OVER_2: Self;
  const SQRT2: Self;
  /// Distance from 1.0 to next larger float number $\epsilon_M = \beta^{1-t}$ ($t=24$ for `f32`, $t=53$ for `f64`)
  const EPSILON: Self;
  /// unit roundoff $u = \frac{1}{2}\beta^{1-t}$ ($t=24$ for `f32`, $t=53$ for `f64`)
  /// *warning*: PBRTv4 refers to unit roundoff as "machine epsilon"
  const UNIT_ROUNDOFF: Self;
  const ONE_MINUS_EPSILON: Self;
  /// Small bump used in rendering as Shadow Bias to avoid self intersection in ray tracing or light mapping
  const SHADOW_EPSILON: Self;
  const NEG_INFINITY: Self;
  const INFINITY: Self;

  // ---------------------- Factory Methods ------------------------------
  fn from_i32(n: i32) -> Self;

  // ---------------------- Next Float Operations ------------------------
  // TODO: Implement this function for f32, f64 at least without the subnormal skip loop
  fn next_float_up(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity
      + ops::Add<Output = <Self as BitsStorage>::Bits>
      + ops::Sub<Output = <Self as BitsStorage>::Bits>,
  {
    debug_assert!(!self.is_nan());
    if self.is_infinite() && self.is_sign_positive() {
      self
    } else {
      // -0 -> +0, such that increments goes to next subnormal
      let val = if self == Self::zero() && self.is_sign_negative() {
        Self::zero()
      } else {
        self
      };

      if allow_subnormal {
        let mut bits = val.to_bits();
        if val >= Self::zero() {
          bits = bits + <<Self as BitsStorage>::Bits as MulAddIdentity>::one();
        } else {
          bits = bits - <<Self as BitsStorage>::Bits as MulAddIdentity>::one();
        }

        Self::from_bits(bits)
      } else {
        // Skip subnormals (without knowing exact representation, we can only cycle)
        let mut next = self.next_float_up(true);
        while next.abs() < Self::EPSILON {
          next = next.next_float_up(true);
        }

        next
      }
    }
  }
  fn next_float_down(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity
      + ops::Add<Output = <Self as BitsStorage>::Bits>
      + ops::Sub<Output = <Self as BitsStorage>::Bits>,
  {
    debug_assert!(!self.is_nan());
    if self.is_infinite() && self.is_sign_negative() {
      self
    } else {
      // +0 -> -0, as decrementing zero requires it to be -0 first in standard IEEE 754
      let val = if self == Self::zero() {
        Self::NEGATIVE_ZERO
      } else {
        self
      };
      if allow_subnormal {
        let mut bits = val.to_bits();
        if val > Self::zero() {
          bits = bits - <<Self as BitsStorage>::Bits as MulAddIdentity>::one();
        } else {
          bits = bits + <<Self as BitsStorage>::Bits as MulAddIdentity>::one();
        }

        Self::from_bits(bits)
      } else {
        let mut next = self.next_float_down(true);
        while next.abs() < Self::EPSILON {
          next = next.next_float_down(true);
        }

        next
      }
    }
  }

  // --------------------- Rounded Operations ----------------------------
  // TODO some backends have intrinsics without having to change rounding mode
  fn add_round_up(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self + that).next_float_up(allow_subnormal)
  }

  fn add_round_down(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self + that).next_float_down(allow_subnormal)
  }

  fn sub_round_up(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self - that).next_float_up(allow_subnormal)
  }

  fn sub_round_down(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self - that).next_float_down(allow_subnormal)
  }

  fn mul_round_up(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self * that).next_float_up(allow_subnormal)
  }

  fn mul_round_down(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self * that).next_float_down(allow_subnormal)
  }

  fn div_round_up(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self / that).next_float_up(allow_subnormal)
  }

  fn div_round_down(self, that: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    (self / that).next_float_down(allow_subnormal)
  }

  fn fma_round_up(self, mult: Self, added: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.fma(mult, added).next_float_up(allow_subnormal)
  }

  fn fma_round_down(self, mult: Self, added: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.fma(mult, added).next_float_down(allow_subnormal)
  }

  fn sqrt_round_up(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.sqrt().next_float_up(allow_subnormal)
  }

  fn sqrt_round_down(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.sqrt().next_float_down(allow_subnormal)
  }

  fn exp_round_up(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.exp().next_float_up(allow_subnormal)
  }

  fn exp_round_down(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.exp().next_float_down(allow_subnormal)
  }

  fn ln_round_up(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.ln().next_float_up(allow_subnormal)
  }

  fn ln_round_down(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.ln().next_float_down(allow_subnormal)
  }

  fn acos_round_down(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.acos().next_float_down(allow_subnormal)
  }

  fn acos_round_up(self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.acos().next_float_up(allow_subnormal)
  }

  fn fmod_round_down(self, modulus: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.fmod(modulus).next_float_down(allow_subnormal)
  }

  fn fmod_round_up(self, modulus: Self, allow_subnormal: bool) -> Self
  where
    Self: FloatBits,
    <Self as BitsStorage>::Bits: MulAddIdentity,
  {
    self.fmod(modulus).next_float_up(allow_subnormal)
  }

  // --------------------- Helpers ---------------------------------------
  fn gamma(n: i32) -> Self {
    let n = Self::from_i32(n);
    let eps = Self::EPSILON;
    (n * eps) / (Self::one() - n * eps)
  }

  fn abs(self) -> Self
  where
    Self: FloatBits,
  {
    Self::from_bits(self.to_bits() & <Self as BitsStorage>::SIGN_BIT_MASK)
  }
  fn lerp(t: Self, a: Self, b: Self) -> Self {
    Self::fma(t, b - a, a)
  }
  fn clamp(x: Self, low: Self, high: Self) -> Self {
    if x < low {
      low
    } else if x > high {
      high
    } else {
      x
    }
  }
  // ab - cd (TODO: Interval will rewrite this probably)
  fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
    // Kahan's algorithm for precise determinant calculation
    // TODO refactor in its own inline function
    let cd = c * d;
    let dop = Self::fma(a, b, -cd);
    let err = Self::fma(-c, d, cd);
    dop + err
  }
  // ab + cd (TODO: Interval will rewrite this probably)
  fn sum_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
    let cd = c * d;
    let sop = Self::fma(a, b, cd);
    let err = Self::fma(c, d, -cd);
    sop + err
  }

  // solve ax^2 + bx + c = 0
  // TODO See Higham for better solution
  fn quadratic(a: Self, b: Self, c: Self) -> Option<(Self, Self)> {
    let discrim = b * b - Self::from_i32(4) * a * c;
    if discrim < Self::zero() {
      return None;
    }

    let root = discrim.sqrt();
    // given b's sign, compute directly solution without cancellation
    let q = if b < Self::zero() {
      -Self::from_i32(1) * (b - root) * Self::from_i32(1) / Self::from_i32(2)
    } else {
      -Self::from_i32(1) * (b + root) * Self::from_i32(1) / Self::from_i32(2)
    };

    let t0 = q / a;
    let t1 = c / q;

    if t0 < t1 {
      Some((t0, t1))
    } else {
      Some((t1, t0))
    }
  }
}

impl FloatOps for f32 {
  const NEGATIVE_ZERO: Self = -0.0;
  const PI: Self = f32::consts::PI;
  const INV_PI: Self = f32::consts::FRAC_1_PI;
  const PI_OVER_2: Self = f32::consts::FRAC_PI_2;
  const SQRT2: Self = f32::consts::SQRT_2;
  const EPSILON: Self = f32::EPSILON;
  const UNIT_ROUNDOFF: Self = f32::EPSILON / 2f32;
  const ONE_MINUS_EPSILON: Self = 0.99999994f32; // from_bits(0x3F7F_FFFF)
  const SHADOW_EPSILON: Self = 0.0001f32;
  const NEG_INFINITY: Self = f32::NEG_INFINITY;
  const INFINITY: Self = f32::INFINITY;

  #[inline]
  fn from_i32(n: i32) -> Self {
    n as Self
  }
}

impl FloatOps for f64 {
  const NEGATIVE_ZERO: Self = -0.0;
  const PI: Self = f64::consts::PI;
  const INV_PI: Self = f64::consts::FRAC_1_PI;
  const PI_OVER_2: Self = f64::consts::FRAC_PI_2;
  const SQRT2: Self = f64::consts::SQRT_2;
  const EPSILON: Self = f64::EPSILON;
  const UNIT_ROUNDOFF: Self = f64::EPSILON / 2f64;
  const ONE_MINUS_EPSILON: Self = 0.99999999999999988897769753748434595763683319091796875; // from_bits(0x3FEF_FFFF_FFFF_FFFF)
  const SHADOW_EPSILON: Self = 0.000001;
  const NEG_INFINITY: Self = f64::NEG_INFINITY;
  const INFINITY: Self = f64::INFINITY;

  #[inline]
  fn from_i32(n: i32) -> Self {
    n as Self
  }
}
