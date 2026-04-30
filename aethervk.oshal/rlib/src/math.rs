use core::{arch::asm, cmp, ops};

// target_pointer_width = "64" assumed, as done in lib.rs

/// Safe division guard to avoid `.abs()` (which requires libm in `no_std`)
/// and bypasses Infinity/NaN scaling bounds when dividing by ~0.0
pub fn safe_div(a: f32, b: f32) -> f32 {
  if b > -1e-6_f32 && b < 1e-6_f32 {
    0.0
  } else {
    a / b
  }
}

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

  fn min(self, rhs: Self) -> Self {
    if self < rhs { self } else { rhs }
  }
  fn max(self, rhs: Self) -> Self {
    if self > rhs { self } else { rhs }
  }

  fn sqrt(self) -> Self; // TODO Option? See Arch's manual for sqrt behavior and tag unsafe
  fn squared(self) -> Self;
  fn cos(self) -> Self;
  fn acos(self) -> Self;
  fn sin(self) -> Self;
  fn tan(self) -> Self;
  fn pow(self, v: Self) -> Self;
  fn exp(self) -> Self;
  fn ln(self) -> Self;
  fn reciprocal(self) -> Self;
  fn floor(self) -> Self;
  fn fmod(self, modulus: Self) -> Self;
  fn asin(self) -> Self;
  fn atan2(first: Self, second: Self) -> Self;
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
    #[cfg(target_arch = "x86_64")]
    {
      unsafe fn sse(val: f32) -> f32 {
        let out: f32;
        unsafe {
          core::arch::asm!("sqrtss {0}, {1}", out(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      #[target_feature(enable = "avx")]
      unsafe fn avx(val: f32) -> f32 {
        let out: f32;
        // vsqrtss technically takes 3 operands (dest, upper-bits-src, lower-bits-src)
        unsafe {
          core::arch::asm!("vsqrtss {0}, {1}, {1}", out(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      let can_vex = crate::SYSTEM_INFO
        .get()
        .as_ref()
        .map(|s| s.has_avx())
        .unwrap_or(false);
      if can_vex {
        unsafe { avx(self) }
      } else {
        unsafe { sse(self) }
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      let out: f32;
      unsafe {
        core::arch::asm!("fsqrt {out:s}, {inp:s}", out = lateout(vreg) out, inp = in(vreg) self, options(pure, nomem, nostack));
      }
      out
    }
  }

  fn squared(self) -> Self {
    self * self
  }
  fn from_f32(num: f32) -> Self {
    num
  }

  fn acos(self) -> Self {
    libm::acosf(self)
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
  fn asin(self) -> Self {
    libm::asinf(self)
  }
  fn fmod(self, modulus: Self) -> Self {
    libm::fmodf(self, modulus)
  }

  fn atan2(first: Self, second: Self) -> Self {
    libm::atan2f(first, second)
  }

  fn reciprocal(self) -> Self {
    #[cfg(target_arch = "x86_64")]
    {
      unsafe fn sse(val: f32) -> f32 {
        let mut out = 1.0f32;
        // divss dest, src -> dest = dest / src
        unsafe {
          core::arch::asm!("divss {0}, {1}", inout(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      #[target_feature(enable = "avx")]
      unsafe fn avx(val: f32) -> f32 {
        let out: f32;
        // vdivss dest, src1, src2 -> dest = src1 / src2
        unsafe {
          core::arch::asm!("vdivss {0}, {1}, {2}", out(xmm_reg) out, in(xmm_reg) 1.0f32, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      let can_vex = crate::SYSTEM_INFO
        .get()
        .as_ref()
        .map(|s| s.has_avx())
        .unwrap_or(false);
      if can_vex {
        unsafe { avx(self) }
      } else {
        unsafe { sse(self) }
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      let out: f32;
      unsafe {
        // Native precise division is preferred over frecpe/frecps manual stepping
        core::arch::asm!("fdiv {out:s}, {one:s}, {num:s}", out = lateout(vreg) out, one = in(vreg) 1.0f32, num = in(vreg) self, options(pure, nomem, nostack));
      }
      out
    }
  }
});

impl_float_like!(f64, {
  fn sqrt(self) -> Self {
    #[cfg(target_arch = "x86_64")]
    {
      unsafe fn sse(val: f64) -> f64 {
        let out: f64;
        unsafe {
          core::arch::asm!("sqrtsd {0}, {1}", out(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      #[target_feature(enable = "avx")]
      unsafe fn avx(val: f64) -> f64 {
        let out: f64;
        unsafe {
          core::arch::asm!("vsqrtsd {0}, {1}, {1}", out(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      let can_vex = crate::SYSTEM_INFO
        .get()
        .as_ref()
        .map(|s| s.has_avx())
        .unwrap_or(false);
      if can_vex {
        unsafe { avx(self) }
      } else {
        unsafe { sse(self) }
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      let out: f64;
      unsafe {
        core::arch::asm!("fsqrt {out:d}, {inp:d}", out = lateout(vreg) out, inp = in(vreg) self, options(pure, nomem, nostack));
      }
      out
    }
  }

  fn squared(self) -> Self {
    self * self
  }
  fn from_f32(num: f32) -> Self {
    num as f64
  }
  fn acos(self) -> Self {
    libm::acos(self)
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
  fn fmod(self, modulus: Self) -> Self {
    libm::fmod(self, modulus)
  }
  fn asin(self) -> Self {
    libm::asin(self)
  }

  fn atan2(first: Self, second: Self) -> Self {
    libm::atan2(first, second)
  }

  fn reciprocal(self) -> Self {
    #[cfg(target_arch = "x86_64")]
    {
      unsafe fn sse(val: f64) -> f64 {
        let mut out = 1.0f64;
        unsafe {
          core::arch::asm!("divsd {0}, {1}", inout(xmm_reg) out, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      #[target_feature(enable = "avx")]
      unsafe fn avx(val: f64) -> f64 {
        let out: f64;
        unsafe {
          core::arch::asm!("vdivsd {0}, {1}, {2}", out(xmm_reg) out, in(xmm_reg) 1.0f64, in(xmm_reg) val, options(pure, nomem, nostack));
        }
        out
      }

      let can_vex = crate::SYSTEM_INFO
        .get()
        .as_ref()
        .map(|s| s.has_avx())
        .unwrap_or(false);
      if can_vex {
        unsafe { avx(self) }
      } else {
        unsafe { sse(self) }
      }
    }
    #[cfg(target_arch = "aarch64")]
    {
      let out: f64;
      unsafe {
        core::arch::asm!("fdiv {out:d}, {one:d}, {num:d}", out = lateout(vreg) out, one = in(vreg) 1.0f64, num = in(vreg) self, options(pure, nomem, nostack));
      }
      out
    }
  }
});
