use core::{arch::asm, cmp, ops::{self, Index, IndexMut}};

use crate::math::floating::{BitsStorage, FloatBits};

pub mod floating;

// target_pointer_width = "64" assumed, as done in lib.rs

// ------------------------------------------ Utils -------------------------------------------
/// Helper function to find the minimum between two PartialOrd elements by reference
pub(super) fn min_two<'a, T: PartialOrd + ?Sized>(a: &'a T, b: &'a T) -> &'a T {
  match a.partial_cmp(b) {
    Some(cmp::Ordering::Less) => a,
    _ => b,
  }
}

/// Helper function to find the maximum between two PartialOrd elements by reference
pub(super) fn max_two<'a, T: PartialOrd + ?Sized>(a: &'a T, b: &'a T) -> &'a T {
  match a.partial_cmp(b) {
    Some(cmp::Ordering::Greater) => a,
    _ => b,
  }
}

/// variadic macro for min
#[macro_export]
macro_rules! v_min {
  ($x:expr) => ($x);
  ($x:expr, $($y:expr),+ $(,)?) => {
    {
      let mut temp_min = &$x;
      $(
        temp_min = $crate::math::min_two(temp_min, &$y);
      )+
      *temp_min
    }
  }
}

/// variadic macro for max
#[macro_export]
macro_rules! v_max {
  ($x:expr) => ($x);
  ($x:expr, $($y:expr),+ $(,)?) => {
    {
      let mut temp_max = &$x;
      $(
        temp_max = $crate::math::max_two(temp_max, &$y);
      )+
      *temp_max
    }
  }
}

// ------------------------------------------ Traits ------------------------------------------
pub(super) trait MulAddIdentity {
  const ONE: Self;
  const ZERO: Self;
}
macro_rules! impl_mul_add_identity {
  ($($t:ty),*) => {
    $(
      impl MulAddIdentity for $t {
        const ONE: Self = 1 as $t;
        const ZERO: Self = 0 as $t; // Note: Positive zero for floating points
      }
    )*
  };
}
impl_mul_add_identity!(i8, i16, i32, i64, i128, u8, u32, u64, u128, f32, f64);

pub(super) trait Fma {
  fn fma(self, b: Self, c: Self) -> Self
  where Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self> {
    self * b + c
  }
}

pub(super) trait FmaAssign {
  fn fma_assign(&mut self, a: Self, b: Self) 
  where Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self> + Copy {
    *self = (*self) * a + b;
  }
}

// TODO: swap for inline assembly implementation
impl Fma for f32 {}
impl FmaAssign for f32 {}
impl Fma for f64 {}
impl FmaAssign for f64 {}

pub(super) trait Scalar:
  Copy
  + Sized
  + MulAddIdentity
  + ops::Add<Output = Self>
  + ops::AddAssign
  + ops::Mul<Output = Self>
  + ops::MulAssign
  + ops::Div<Output = Self>
  + ops::DivAssign
  + ops::Sub<Output = Self>
  + ops::SubAssign
  + ops::Neg<Output = Self>
  + Fma
  + FmaAssign
  + PartialEq
  + PartialOrd
{
}

impl<T> Scalar for T
where
  T: Copy
    + Sized
    + MulAddIdentity
    + ops::Add<Output = T>
    + ops::AddAssign
    + ops::Mul<Output = T>
    + ops::MulAssign
    + ops::Div<Output = T>
    + ops::DivAssign
    + ops::Sub<Output = T>
    + ops::SubAssign
    + ops::Neg<Output = Self>
    + Fma
    + FmaAssign
    + PartialEq
    + PartialOrd,
{
}

pub trait FloatLike: Scalar + Copy {
  fn is_nan(self) -> bool;
  fn is_infinite(self) -> bool;
  fn is_sign_negative(self) -> bool;
  fn is_sign_positive(self) -> bool;
  fn is_finite(self) -> bool;
  fn is_subnormal(self) -> bool;
  fn is_normal(self) -> bool;

  fn sqrt(self) -> Self; // TODO Option? See Arch's manual for sqrt behavior and tag unsafe
}
macro_rules! impl_float_like {
  // Match a type, followed by zero or more items (Functions, ecc)
  ($t:ty, $($body:item)*) => {
      impl FloatLike for $t {
        // shared ops alredy implemented in the `core` crate
        #[inline] fn is_nan(self) -> bool { <$t>::is_nan(self) }
        #[inline] fn is_infinite(self) -> bool { <$t>::is_infinite(self) }
        #[inline] fn is_sign_negative(self) -> bool { <$t>::is_sign_negative(self) }
        #[inline] fn is_sign_positive(self) -> bool { <$t>::is_sign_positive(self) }
        #[inline] fn is_finite(self) -> bool { <$t>::is_finite(self) }
        #[inline] fn is_subnormal(self) -> bool { <$t>::is_subnormal(self) }
        #[inline] fn is_normal(self) -> bool { <$t>::is_normal(self) }
        // ops which are unstable in the current used rust version (1.93.0)
        $($body)*
      }
  };
}

// TODO: all these intrinsics need to be enriched every time we add a new arch backend, eg CUDA, Vulkan's SPIR-V, ...
impl_float_like!(
  f32,
  fn sqrt(self) -> Self {
    let out: f32;
    #[cfg(target_arch = "x86_64")]
    unsafe {
      asm!(
        "sqrtss {0}, {1}", // TODO: See if VEX prefix usage is better (vsqrtss)
        out(xmm_reg) out,
        in(xmm_reg) self,
        options(pure, nomem, nostack)
      );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      asm!(
        "fsqrt {out:s}, {inp:s}", // NEON, which is assumed to be supported? (TODO check)
        out = lateout(vreg) out,
        inp = in(vreg) self,
        options(pure, nomem, nostack)
      );
    }

    out
  }
);
impl_float_like!(
  f64,
  fn sqrt(self) -> Self {
    let out: f64;
    #[cfg(target_arch = "x86_64")]
    unsafe {
      asm!(
        "sqrtsd {0}, {1}",
        out(xmm_reg) out,
        in(xmm_reg) self,
        options(pure, nomem, nostack)
      );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
      asm!(
        "fsqrt {out:d}, {inp:d}", // NEON, which is assumed to be supported? (TODO check)
        out = lateout(vreg) out,
        inp = in(vreg) self,
        options(pure, nomem, nostack)
      );
    }

    out
  }
);

// -------------------------------- Scalar Types ----------------------------------------------
pub mod interval;

// ----------------------------------- Vector Types -------------------------------------------

// ---------------------------------- Matrix Types --------------------------------------------

// ---------------------------------- Quaterion -----------------------------------------------
