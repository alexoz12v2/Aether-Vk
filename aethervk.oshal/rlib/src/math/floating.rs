pub(super) trait FloatBits: super::FloatLike + Sized {
  type Bits: Sized;

  fn to_bits(v: Self) -> Self::Bits;
  fn from_bits(bits: Self::Bits) -> Self;
}

impl FloatBits for f32 {
  type Bits = u32;
  fn to_bits(v: Self) -> Self::Bits {
    f32::to_bits(v)
  }
  fn from_bits(bits: Self::Bits) -> Self {
    f32::from_bits(bits)
  }
}
impl FloatBits for f64 {
  type Bits = u64;
  fn to_bits(v: Self) -> Self::Bits {
    f64::to_bits(v)
  }
  fn from_bits(bits: Self::Bits) -> Self {
    f64::from_bits(bits)
  }
}

pub(super) trait FloatOps<T: super::FloatLike>: Sized {
  // ----------------------------- Constants -----------------------------
  const FP: T;
  const INV_PI: T;
  const INV_2PI: T;
  const PI_OVER_2: T;
  const SQRT2: T;
  const MACHINE_EPSILON: T;
  const ONE_MINUS_EPSILON: T;
  const SHADOW_EPSILON: T;
  const INFINITY: T;

  // --------------------------- Conversions -----------------------------
  // ---------------------- Next Float Operations ------------------------
  // --------------------- Helpers ---------------------------------------
}
