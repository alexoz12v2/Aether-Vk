use core::{arch::asm, cmp, ops};

// target_pointer_width = "64" assumed, as done in lib.rs

/// Helper function to find the minimum between two PartialOrd elements by reference
#[inline]
pub fn min_two<'a, T: PartialOrd + ?Sized>(a: &'a T, b: &'a T) -> &'a T {
  match a.partial_cmp(b) {
    Some(cmp::Ordering::Less) => a,
    _ => b,
  }
}

/// Helper function to find the maximum between two PartialOrd elements by reference
#[inline]
pub fn max_two<'a, T: PartialOrd + ?Sized>(a: &'a T, b: &'a T) -> &'a T {
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

pub mod floating;

pub mod extra;
pub mod matrix;
pub mod quaternion;
pub mod scalar_interval;
pub mod vector;
pub mod vector_interval;

pub trait MulAddIdentity {
  fn one() -> Self;
  fn zero() -> Self;
}
macro_rules! impl_mul_add_identity {
  ($($t:ty),*) => {
    $(
      impl MulAddIdentity for $t {
        fn one() -> Self {
          1 as $t
        }
        fn zero() -> Self {
          0 as $t
        }
      }
    )*
  };
}
// left out unsigned values cause they are not scalars
impl_mul_add_identity!(i8, i16, i32, i64, i128, f32, f64);

pub trait Fma {
  fn fma(self, b: Self, c: Self) -> Self
  where
    Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self>,
  {
    self * b + c
  }
}

pub trait FmaAssign {
  fn fma_assign(&mut self, a: Self, b: Self)
  where
    Self: Sized + ops::Mul<Output = Self> + ops::Add<Output = Self> + Copy,
  {
    *self = (*self) * a + b;
  }
}

// TODO: swap for inline assembly implementation
macro_rules! impl_fma_fma_assign {
  ($($t:ty),*) => {
    $(
      impl Fma for $t {}
      impl FmaAssign for $t {}
    )*
  };
}
impl_fma_fma_assign!(f32, f64, i8, i16, i32, i64, i128);

pub trait Scalar:
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

impl<T> Scalar for T where
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
    + PartialOrd
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
  fn from_f32(num: f32) -> Self;

  fn sqrt(self) -> Self; // TODO Option? See Arch's manual for sqrt behavior and tag unsafe
  fn squared(self) -> Self;
  fn cos(self) -> Self;
  fn sin(self) -> Self;
  fn tan(self) -> Self;
  fn pow(self, v: Self) -> Self;
  fn exp(self) -> Self;
  fn ln(self) -> Self;
  fn reciprocal(self) -> Self;
  fn floor(self) -> Self;
}
macro_rules! impl_float_like {
  // Match a type, followed by zero or more items (Functions, ecc)
  ($t:ty, { $($body:item)* }) => {
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
impl_float_like!(f32, {
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

  fn squared(self) -> Self {
    self * self
  }

  fn from_f32(num: f32) -> Self {
    num
  }

  fn cos(self) -> Self {
    libm::cosf(self)
  }

  fn tan(self) -> Self {
    libm::tanf(self)
  }

  fn exp(self) -> Self {
    libm::expf(self)
  }

  fn ln(self) -> Self {
    libm::logf(self)
  }

  fn floor(self) -> Self {
    libm::floorf(self)
  }

  fn pow(self, v: f32) -> Self {
    libm::powf(self, v)
  }

  fn sin(self) -> Self {
    libm::sinf(self)
  }

  fn reciprocal(self) -> Self {
    let out: f32;
    #[cfg(target_arch = "x86_64")]
    {
      unsafe {
        asm!(
          "rcpss {0}, {1}", // fast reciprocal approximation
          "mulss {0}, {1}, {0}", // Newton-Raphson to refine reciprocal
          out(xmm_reg) out,
          in(xmm_reg) self,
          options(pure, nomem, nostack)
        );
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      unsafe {
        asm!(
          "frecpe {out:s}, {num:s}", // fast reciprocal approximation
          "frecps {out:s}, {out:s}, {num:s}", // newton-raphson refinement step
          out = lateout(vreg) out,
          num = in(vreg) self,
          options(pure, nomem, nostack)
        );
      }
    }

    out
  }
});
impl_float_like!(f64, {
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

  fn squared(self) -> Self {
    self * self
  }

  fn from_f32(num: f32) -> Self {
    num as _
  }

  fn cos(self) -> Self {
    libm::cos(self)
  }

  fn tan(self) -> Self {
    libm::tan(self)
  }

  fn sin(self) -> Self {
    libm::sin(self)
  }

  fn pow(self, v: Self) -> Self {
    libm::pow(self, v)
  }

  fn exp(self) -> Self {
    libm::exp(self)
  }

  fn ln(self) -> Self {
    libm::log(self)
  }

  fn floor(self) -> Self {
    libm::floor(self)
  }

  fn reciprocal(self) -> Self {
    let out: f64;
    #[cfg(target_arch = "x86_64")]
    {
      unsafe {
        asm!(
          "rcpsd {0}, {1}", // fast reciprocal approximation
          "mulsd {0}, {1}, {0}", // Newton-Raphson refinement
          out(xmm_reg) out,
          in(xmm_reg) self,
          options(pure, nomem, nostack)
        );
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      unsafe {
        asm!(
          "frecpe {out:d}, {num:d}", // fast reciprocal approximation
          "frecps {out:d}, {out:d}, {num:d}", // Newton-Raphson refinement
          out = lateout(vreg) out,
          num = in(vreg) self,
          options(pure, nomem, nostack)
        );
      }
    }

    out
  }
});
