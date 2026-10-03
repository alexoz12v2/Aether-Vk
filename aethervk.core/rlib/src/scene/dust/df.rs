//! Double-float ("df64") arithmetic: a value is the unevaluated sum `hi + lo` of two f32 with
//! `|lo| ≤ ulp(hi)/2`, giving ~48 bits of mantissa using only f32 `+ − ×` (plus one f32 `/` or
//! `sqrt` as an initial guess, refined in df64).
//!
//! The GPU baseline (Mali-G52) has no `shaderFloat64`, so the dust shaders use this instead of
//! `double`. `assets/sim/dust_common.glsl` mirrors every function below 1:1; there every
//! temporary is `precise` (SPIR-V `NoContraction`) because the error-free transformations break
//! if the compiler fuses `a*b+c` into an FMA. Rust never contracts f32 ops implicitly.
//!
//! Vulkan guarantees correctly rounded f32 `+ − ×`, so those match the CPU bit for bit; f32 `/`
//! and `sqrt` are only accurate to ~2.5 ulp on GPU, so df64 division and square root agree with
//! the CPU to df64 precision (~2⁻⁴⁶ relative), not bit for bit.
//!
//! Range: the f32 exponent range, minus Dekker-split headroom: operands and products must stay
//! below ~8e34 (the dust math peaks at `|r|² ≈ 1e24 m²`), and `lo` parts flush below ~1e-38.
use aethervk_oshal_rlib::math::FloatLike;

/// `hi + lo`, `|lo| ≤ ulp(hi)/2`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Df {
  pub hi: f32,
  pub lo: f32,
}

/// componentwise df64 3-vector
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Df3 {
  pub hi: [f32; 3],
  pub lo: [f32; 3],
}

/// Dekker splitter `2^12 + 1` (f32 has a 24-bit significand)
const SPLITTER: f32 = 4097.0;

#[inline]
pub fn two_sum(a: f32, b: f32) -> Df {
  let s = a + b;
  let bb = s - a;
  let e = (a - (s - bb)) + (b - bb);
  Df { hi: s, lo: e }
}

/// requires `|a| ≥ |b|` (or `a == 0`)
#[inline]
pub fn quick_two_sum(a: f32, b: f32) -> Df {
  let s = a + b;
  let e = b - (s - a);
  Df { hi: s, lo: e }
}

#[inline]
fn split(a: f32) -> (f32, f32) {
  let t = SPLITTER * a;
  let hi = t - (t - a);
  (hi, a - hi)
}

/// exact product `a·b = hi + lo` (Dekker, no FMA)
#[inline]
pub fn two_prod(a: f32, b: f32) -> Df {
  let p = a * b;
  let (ah, al) = split(a);
  let (bh, bl) = split(b);
  let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  Df { hi: p, lo: e }
}

impl Df {
  pub const ZERO: Df = Df { hi: 0.0, lo: 0.0 };
  pub const ONE: Df = Df { hi: 1.0, lo: 0.0 };

  #[inline]
  pub const fn new(hi: f32, lo: f32) -> Df {
    Df { hi, lo }
  }
  #[inline]
  pub const fn from_f32(x: f32) -> Df {
    Df { hi: x, lo: 0.0 }
  }
  /// host-side conversion (round to nearest, then the exact-ish remainder)
  #[inline]
  pub fn from_f64(x: f64) -> Df {
    let hi = x as f32;
    let lo = (x - hi as f64) as f32;
    Df { hi, lo }
  }
  #[inline]
  pub fn to_f64(self) -> f64 {
    self.hi as f64 + self.lo as f64
  }

  #[inline]
  pub fn neg(self) -> Df {
    Df { hi: -self.hi, lo: -self.lo }
  }
  #[inline]
  pub fn add(self, b: Df) -> Df {
    let s = two_sum(self.hi, b.hi);
    let t = two_sum(self.lo, b.lo);
    let s = quick_two_sum(s.hi, s.lo + t.hi);
    quick_two_sum(s.hi, s.lo + t.lo)
  }
  #[inline]
  pub fn sub(self, b: Df) -> Df {
    self.add(b.neg())
  }
  #[inline]
  pub fn add_f(self, b: f32) -> Df {
    let s = two_sum(self.hi, b);
    quick_two_sum(s.hi, s.lo + self.lo)
  }
  #[inline]
  pub fn mul(self, b: Df) -> Df {
    let p = two_prod(self.hi, b.hi);
    quick_two_sum(p.hi, p.lo + (self.hi * b.lo + self.lo * b.hi))
  }
  #[inline]
  pub fn mul_f(self, b: f32) -> Df {
    let p = two_prod(self.hi, b);
    quick_two_sum(p.hi, p.lo + self.lo * b)
  }
  /// exact scaling by a power of two
  #[inline]
  pub fn scale_pow2(self, p: f32) -> Df {
    Df { hi: self.hi * p, lo: self.lo * p }
  }
  /// long division with two correction steps
  #[inline]
  pub fn div(self, b: Df) -> Df {
    let q1 = self.hi / b.hi;
    let r = self.sub(b.mul_f(q1));
    let q2 = r.hi / b.hi;
    let r = r.sub(b.mul_f(q2));
    let q3 = r.hi / b.hi;
    quick_two_sum(q1, q2).add_f(q3)
  }
  /// one Newton step on top of the f32 root; 0 for non-positive input
  #[inline]
  pub fn sqrt(self) -> Df {
    if !(self.hi > 0.0) {
      return Df::ZERO;
    }
    let x = <f32 as FloatLike>::sqrt(self.hi);
    let r = self.sub(two_prod(x, x));
    quick_two_sum(x, r.hi / (2.0 * x))
  }
  #[inline]
  pub fn abs(self) -> Df {
    if self.hi < 0.0 { self.neg() } else { self }
  }
  /// truncation toward −∞ (for the elliptic period reduction)
  #[inline]
  pub fn floor(self) -> Df {
    let fh = <f32 as FloatLike>::floor(self.hi);
    if fh == self.hi {
      // hi is integral (|hi| ≥ 2²³ or already whole): floor the low part
      quick_two_sum(fh, <f32 as FloatLike>::floor(self.lo))
    } else {
      Df::from_f32(fh)
    }
  }
  #[inline]
  pub fn lt(self, b: Df) -> bool {
    self.hi < b.hi || (self.hi == b.hi && self.lo < b.lo)
  }
}

impl Df3 {
  pub const ZERO: Df3 = Df3 { hi: [0.0; 3], lo: [0.0; 3] };

  #[inline]
  pub fn new(x: Df, y: Df, z: Df) -> Df3 {
    Df3 { hi: [x.hi, y.hi, z.hi], lo: [x.lo, y.lo, z.lo] }
  }
  #[inline]
  pub fn get(&self, i: usize) -> Df {
    Df { hi: self.hi[i], lo: self.lo[i] }
  }
  #[inline]
  pub fn from_f64(v: [f64; 3]) -> Df3 {
    Df3::new(Df::from_f64(v[0]), Df::from_f64(v[1]), Df::from_f64(v[2]))
  }
  #[inline]
  pub fn from_f32(v: [f32; 3]) -> Df3 {
    Df3 { hi: v, lo: [0.0; 3] }
  }
  #[inline]
  pub fn to_f64(&self) -> [f64; 3] {
    [self.get(0).to_f64(), self.get(1).to_f64(), self.get(2).to_f64()]
  }
  /// nearest f32 vector (`hi + lo` rounded once)
  #[inline]
  pub fn to_f32(&self) -> [f32; 3] {
    [self.hi[0] + self.lo[0], self.hi[1] + self.lo[1], self.hi[2] + self.lo[2]]
  }
  #[inline]
  pub fn add(&self, b: &Df3) -> Df3 {
    Df3::new(self.get(0).add(b.get(0)), self.get(1).add(b.get(1)), self.get(2).add(b.get(2)))
  }
  #[inline]
  pub fn sub(&self, b: &Df3) -> Df3 {
    Df3::new(self.get(0).sub(b.get(0)), self.get(1).sub(b.get(1)), self.get(2).sub(b.get(2)))
  }
  #[inline]
  pub fn scale(&self, s: Df) -> Df3 {
    Df3::new(self.get(0).mul(s), self.get(1).mul(s), self.get(2).mul(s))
  }
  #[inline]
  pub fn dot(&self, b: &Df3) -> Df {
    self.get(0).mul(b.get(0)).add(self.get(1).mul(b.get(1))).add(self.get(2).mul(b.get(2)))
  }
}

/// df64 constants. The GLSL side spells the same `hi`/`lo` pairs as literals (9 significant
/// digits round-trip f32 exactly); `constants_match_f64` checks them against `from_f64`.
pub mod consts {
  use super::Df;

  /// Sun gravitational parameter (m³/s²)
  pub const SUN_MU: Df = Df::new(1.32712443e+20, -3.23726082e+12);
  pub const TWO_PI: Df = Df::new(6.28318548e+00, -1.74845553e-07);
  /// `1/k!` for `k = 0..=15`
  pub const INV_FACT: [Df; 16] = [
    Df::new(1.0, 0.0),
    Df::new(1.0, 0.0),
    Df::new(5.000000000e-01, 0.0),
    Df::new(1.666666716e-01, -4.967053879e-09),
    Df::new(4.166666791e-02, -1.241763470e-09),
    Df::new(8.333333768e-03, -4.346172033e-10),
    Df::new(1.388888923e-03, -3.363109444e-11),
    Df::new(1.984127011e-04, -2.725596875e-12),
    Df::new(2.480158764e-05, -3.406996094e-13),
    Df::new(2.755731884e-06, 3.793571224e-14),
    Df::new(2.755731998e-07, -7.575112209e-15),
    Df::new(2.505210794e-08, 4.417623045e-16),
    Df::new(2.087675588e-09, 1.108283981e-16),
    Df::new(1.605904437e-10, -5.352526512e-18),
    Df::new(1.147074536e-11, 2.372207689e-19),
    Df::new(7.647163610e-13, 1.220071047e-20),
  ];
}
