//! Display stretch of a FITS raster to RGBA8 (the asset library's FITS textures).
//!
//! - **Black / white points**: [`StretchMode::ZScale`] (IRAF `zscale`, astropy `ZScaleInterval`
//!   parameters: ≤ [`ZSCALE_MAX_SAMPLES`] samples on a regular grid, contrast
//!   [`ZSCALE_CONTRAST`], 2.5 σ rejection, 5 iterations), [`StretchMode::Percentile`] (percent
//!   units, e.g. 0.5 / 99.5), [`StretchMode::Manual`].
//! - **Transfer** of `x = clamp((v − black) / (white − black), 0, 1)`: linear `x`, log
//!   `log(1 + a·x) / log(1 + a)` with `a` = [`LOG_EXPONENT`] (ds9's), asinh
//!   `asinh(x / β) / asinh(1 / β)` with `β` = [`ASINH_SOFTENING`].
//! - Blank pixels (NaN: FITS `BLANK`, IEEE NaN) are excluded from the statistics and drawn black;
//!   alpha is always 255. Grey → `R = G = B`.
//! - Fill values: a run of one exact value at the minimum / maximum holding ≥
//!   [`FILL_RUN_FRACTION`] of the samples (e.g. stacking software writing −38288 into the
//!   uncovered borders without a `BLANK` card) is also left out of the zscale / percentile
//!   statistics; those pixels still render (clipped to black / white).

use alloc::vec::Vec;

/// zscale contrast (IRAF / astropy default)
pub const ZSCALE_CONTRAST: f64 = 0.25;
/// largest zscale sample (a 600 × 600 grid)
pub const ZSCALE_MAX_SAMPLES: usize = 600 * 600;
/// default percentile black / white points (percent)
pub const DEFAULT_PERCENTILE_LOW: f64 = 0.5;
pub const DEFAULT_PERCENTILE_HIGH: f64 = 99.5;
/// log transfer exponent
pub const LOG_EXPONENT: f64 = 1000.0;
/// asinh transfer softening
pub const ASINH_SOFTENING: f64 = 0.1;

/// How the black / white points are chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StretchMode {
  ZScale,
  /// percent units (`0..=100`)
  Percentile { low: f64, high: f64 },
  /// data units
  Manual { black: f64, white: f64 },
}

/// The tone curve applied between the black and white points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transfer {
  Linear,
  Log,
  Asinh,
}

/// A complete stretch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitsStretch {
  pub mode: StretchMode,
  pub transfer: Transfer,
}

impl Default for FitsStretch {
  /// zscale, linear (what a FITS import gets).
  fn default() -> Self {
    Self { mode: StretchMode::ZScale, transfer: Transfer::Linear }
  }
}

impl FitsStretch {
  /// From the FFI codes: `mode` 0 zscale, 1 percentile (`p0` low %, `p1` high %), 2 manual
  /// (`p0` black, `p1` white); `transfer` 0 linear, 1 log, 2 asinh. `None` for an unknown code,
  /// non-finite parameters or percentiles outside `0..=100`.
  pub fn from_codes(mode: u32, transfer: u32, p0: f64, p1: f64) -> Option<Self> {
    let transfer = match transfer {
      0 => Transfer::Linear,
      1 => Transfer::Log,
      2 => Transfer::Asinh,
      _ => return None,
    };
    let mode = match mode {
      0 => StretchMode::ZScale,
      1 if p0.is_finite() && p1.is_finite() && (0.0..=100.0).contains(&p0) && (0.0..=100.0).contains(&p1) => {
        StretchMode::Percentile { low: p0, high: p1 }
      }
      2 if p0.is_finite() && p1.is_finite() => StretchMode::Manual { black: p0, white: p1 },
      _ => return None,
    };
    Some(Self { mode, transfer })
  }

  /// `(mode code, transfer code)` (inverse of [`Self::from_codes`]).
  pub fn codes(&self) -> (u32, u32) {
    let m = match self.mode {
      StretchMode::ZScale => 0,
      StretchMode::Percentile { .. } => 1,
      StretchMode::Manual { .. } => 2,
    };
    let t = match self.transfer {
      Transfer::Linear => 0,
      Transfer::Log => 1,
      Transfer::Asinh => 2,
    };
    (m, t)
  }
}

/// Reverses the row order in place (FITS row 0 is the bottom of the image: after the flip the
/// first row is the top, as a viewer shows it, north up for a standard WCS).
pub fn flip_rows<T>(data: &mut [T], nx: u32, ny: u32) {
  let (nx, ny) = (nx as usize, ny as usize);
  if data.len() < nx * ny || nx == 0 {
    return;
  }
  for j in 0..ny / 2 {
    let (top, bottom) = data.split_at_mut((ny - 1 - j) * nx);
    top[j * nx..(j + 1) * nx].swap_with_slice(&mut bottom[..nx]);
  }
}

/// Finite values on a regular grid of at most `max` points, sorted.
fn grid_sample(data: &[f32], nx: u32, ny: u32, max: usize) -> Vec<f32> {
  let (nx, ny) = (nx as usize, ny as usize);
  let n = (nx * ny).min(data.len());
  let mut step = 1usize;
  while n.div_ceil(step * step) > max {
    step += 1;
  }
  let mut v = Vec::with_capacity(n.div_ceil(step * step).min(max));
  let mut y = 0;
  while y < ny {
    let mut x = 0;
    while x < nx {
      let s = data[y * nx + x];
      if s.is_finite() {
        v.push(s);
      }
      x += step;
    }
    y += step;
  }
  v.sort_unstable_by(|a, b| a.total_cmp(b));
  v
}

/// Fraction of the samples above which a run of one exact value at the minimum (or maximum) is
/// a fill value (stack borders written as a constant without `BLANK`, saturation) and left out
/// of the statistics.
pub const FILL_RUN_FRACTION: f64 = 0.01;

/// `sorted` without fill runs at either end (see [`FILL_RUN_FRACTION`]); kept as is when that
/// would leave nothing.
pub fn trim_fill_runs(sorted: &[f32]) -> &[f32] {
  let n = sorted.len();
  if n < 2 {
    return sorted;
  }
  let min_run = ((n as f64 * FILL_RUN_FRACTION).ceil() as usize).max(2);
  let lo = sorted.iter().take_while(|&&v| v == sorted[0]).count();
  let hi = sorted.iter().rev().take_while(|&&v| v == sorted[n - 1]).count();
  let start = if lo >= min_run { lo } else { 0 };
  let end = if hi >= min_run { n - hi } else { n };
  if start < end { &sorted[start..end] } else { sorted }
}

/// IRAF `zscale` black / white points on the samples without fill runs; `None` without finite
/// pixels.
pub fn zscale(data: &[f32], nx: u32, ny: u32) -> Option<(f64, f64)> {
  let s = grid_sample(data, nx, ny, ZSCALE_MAX_SAMPLES);
  zscale_sorted(trim_fill_runs(&s), ZSCALE_CONTRAST)
}

/// zscale of sorted samples (astropy `ZScaleInterval.get_limits`).
pub fn zscale_sorted(samples: &[f32], contrast: f64) -> Option<(f64, f64)> {
  const MAX_REJECT: f64 = 0.5;
  const MIN_NPIXELS: usize = 5;
  const KREJ: f64 = 2.5;
  const MAX_ITERATIONS: usize = 5;
  let npix = samples.len();
  if npix == 0 {
    return None;
  }
  let zmin = samples[0] as f64;
  let zmax = samples[npix - 1] as f64;
  let minpix = MIN_NPIXELS.max((npix as f64 * MAX_REJECT) as usize);
  let ngrow = 1usize.max((npix as f64 * 0.01) as usize);
  let mut bad = alloc::vec![false; npix];
  let mut ngood = npix;
  let mut last_ngood = npix + 1;
  let mut slope = 0.0f64;
  let mut fitted = false;
  for _ in 0..MAX_ITERATIONS {
    if ngood >= last_ngood || ngood < minpix {
      break;
    }
    // weighted least squares line through the good samples (x = index)
    let (mut s, mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for (i, &v) in samples.iter().enumerate() {
      if !bad[i] {
        let (x, y) = (i as f64, v as f64);
        s += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
      }
    }
    let d = s * sxx - sx * sx;
    if !(d.abs() > 0.0) {
      break;
    }
    slope = (s * sxy - sx * sy) / d;
    let intercept = (sy - slope * sx) / s;
    fitted = true;
    // residual σ of the good samples, reject beyond KREJ σ
    let (mut m, mut m2) = (0.0f64, 0.0f64);
    for (i, &v) in samples.iter().enumerate() {
      if !bad[i] {
        let r = v as f64 - (intercept + slope * i as f64);
        m += r;
        m2 += r * r;
      }
    }
    let mean = m / s;
    let threshold = KREJ * (m2 / s - mean * mean).max(0.0).sqrt();
    for (i, &v) in samples.iter().enumerate() {
      let r = v as f64 - (intercept + slope * i as f64);
      if r < -threshold || r > threshold {
        bad[i] = true;
      }
    }
    // grow the rejected samples (numpy convolve(bad, ones(ngrow), 'same'))
    let mut prefix = alloc::vec![0u32; npix + 1];
    for i in 0..npix {
      prefix[i + 1] = prefix[i] + bad[i] as u32;
    }
    let (left, right) = (ngrow - 1 - (ngrow - 1) / 2, (ngrow - 1) / 2);
    for (i, b) in bad.iter_mut().enumerate() {
      let lo = i.saturating_sub(left);
      let hi = (i + right).min(npix - 1);
      *b = prefix[hi + 1] - prefix[lo] > 0;
    }
    last_ngood = ngood;
    ngood = bad.iter().filter(|b| !**b).count();
  }
  let (mut lo, mut hi) = (zmin, zmax);
  if fitted && ngood >= minpix {
    if contrast > 0.0 {
      slope /= contrast;
    }
    let center = (npix - 1) / 2;
    let median = samples[center] as f64;
    lo = zmin.max(median - (center as f64 - 1.0) * slope);
    hi = zmax.min(median + (npix - center) as f64 * slope);
  }
  if !(hi > lo) {
    (lo, hi) = (zmin, zmax);
  }
  Some((lo, hi))
}

/// Black / white points of `mode` (`(0, 1)` without finite pixels; a degenerate range is widened
/// so `white > black` holds, except a manual range which is kept, inverted ranges included).
pub fn black_white(data: &[f32], nx: u32, ny: u32, mode: StretchMode) -> (f64, f64) {
  let (b, w) = match mode {
    StretchMode::Manual { black, white } => return (black, white),
    StretchMode::ZScale => zscale(data, nx, ny).unwrap_or((0.0, 1.0)),
    StretchMode::Percentile { low, high } => {
      let all = grid_sample(data, nx, ny, 4_000_000);
      let s = trim_fill_runs(&all);
      if s.is_empty() {
        (0.0, 1.0)
      } else {
        let at = |p: f64| {
          let k = ((p.clamp(0.0, 100.0) / 100.0) * (s.len() - 1) as f64).round() as usize;
          s[k.min(s.len() - 1)] as f64
        };
        (at(low), at(high))
      }
    }
  };
  if w > b { (b, w) } else { (b, b + (b.abs() * 1e-6).max(1e-6)) }
}

/// The transfer curve on `x ∈ [0, 1]` → `[0, 1]`.
pub fn transfer(x: f64, t: Transfer) -> f64 {
  let x = x.clamp(0.0, 1.0);
  match t {
    Transfer::Linear => x,
    Transfer::Log => (1.0 + LOG_EXPONENT * x).ln() / (1.0 + LOG_EXPONENT).ln(),
    Transfer::Asinh => asinh(x / ASINH_SOFTENING) / asinh(1.0 / ASINH_SOFTENING),
  }
}

fn asinh(x: f64) -> f64 {
  (x + (x * x + 1.0).sqrt()).ln()
}

/// RGBA8 of `data` (row order kept): grey `R = G = B`, alpha 255, NaN → black.
pub fn render_rgba8(data: &[f32], black: f64, white: f64, t: Transfer) -> Vec<u8> {
  // 65536-entry lookup of the transfer on the normalised value
  const LUT: usize = 65536;
  let lut: Vec<u8> = (0..LUT)
    .map(|i| (transfer(i as f64 / (LUT - 1) as f64, t) * 255.0).round().clamp(0.0, 255.0) as u8)
    .collect();
  let range = white - black;
  let inv = if range != 0.0 && range.is_finite() { 1.0 / range } else { 0.0 };
  let mut out = Vec::with_capacity(data.len() * 4);
  for &v in data {
    let g = if !v.is_finite() {
      0
    } else if inv == 0.0 {
      if v as f64 > black { 255 } else { 0 }
    } else {
      let x = ((v as f64 - black) * inv).clamp(0.0, 1.0);
      lut[(x * (LUT - 1) as f64).round() as usize]
    };
    out.extend_from_slice(&[g, g, g, 255]);
  }
  out
}
