//! Distortions: SIP (Shupe et al. 2005, `A_p_q` / `B_p_q` on the pixel offsets from `CRPIX`
//! before the linear part, `AP` / `BP` the approximate inverse), TPV (the `PVi_m` polynomial on
//! the TAN intermediate coordinates, the TPV convention of SCAMP / wcslib 7.4), the IRAF TNX /
//! ZPX correction surfaces (`lngcor` / `latcor` of the `WAT` strings, added to the intermediate
//! coordinates) and the DSS plate solution (`AMDXn` / `AMDYn` on plate millimetres). Inverses
//! are solved by Newton iteration (the `AP` / `BP` polynomial, when given, is the starting
//! guess), so pixel → sky → pixel closes to ~1e-10 px.

use super::super::FitsHeader;
use alloc::{format, string::String};

/// Largest SIP polynomial order accepted (`A_ORDER` …).
pub const SIP_MAX_ORDER: usize = 9;
const N: usize = SIP_MAX_ORDER + 1;

/// SIP coefficients `[p][q]` of `u^p v^q`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sip {
  pub a: [[f64; N]; N],
  pub b: [[f64; N]; N],
  pub ap: [[f64; N]; N],
  pub bp: [[f64; N]; N],
  pub a_order: usize,
  pub b_order: usize,
  pub ap_order: usize,
  pub bp_order: usize,
  pub has_inverse: bool,
}

fn read_poly(h: &FitsHeader, prefix: &str) -> Result<Option<([[f64; N]; N], usize)>, String> {
  let Some(order) = h.get_i64(&format!("{prefix}_ORDER")) else {
    return Ok(None);
  };
  if !(0..=SIP_MAX_ORDER as i64).contains(&order) {
    return Err(format!("{prefix}_ORDER {order} outside 0..={SIP_MAX_ORDER}"));
  }
  let order = order as usize;
  let mut c = [[0.0; N]; N];
  for (k, v) in h.cards.iter() {
    let Some(rest) = k.strip_prefix(prefix).and_then(|r| r.strip_prefix('_')) else {
      continue;
    };
    let mut it = rest.split('_');
    let (Some(p), Some(q), None) = (it.next(), it.next(), it.next()) else {
      continue;
    };
    let (Ok(p), Ok(q)) = (p.parse::<usize>(), q.parse::<usize>()) else {
      continue;
    };
    if p < N && q < N {
      c[p][q] = v.replace(['D', 'd'], "E").trim().parse::<f64>().map_err(|_| format!("{k}: not a number"))?;
    }
  }
  Ok(Some((c, order)))
}

fn eval(c: &[[f64; N]; N], order: usize, u: f64, v: f64) -> f64 {
  let mut s = 0.0;
  for p in 0..=order {
    for q in 0..=(order - p) {
      let k = c[p][q];
      if k != 0.0 {
        s += k * u.powi(p as i32) * v.powi(q as i32);
      }
    }
  }
  s
}

fn eval_du(c: &[[f64; N]; N], order: usize, u: f64, v: f64) -> f64 {
  let mut s = 0.0;
  for p in 1..=order {
    for q in 0..=(order - p) {
      let k = c[p][q];
      if k != 0.0 {
        s += k * p as f64 * u.powi(p as i32 - 1) * v.powi(q as i32);
      }
    }
  }
  s
}

fn eval_dv(c: &[[f64; N]; N], order: usize, u: f64, v: f64) -> f64 {
  let mut s = 0.0;
  for p in 0..=order {
    for q in 1..=(order - p) {
      let k = c[p][q];
      if k != 0.0 {
        s += k * q as f64 * u.powi(p as i32) * v.powi(q as i32 - 1);
      }
    }
  }
  s
}

impl Sip {
  /// Reads `A_ORDER` / `B_ORDER` and the coefficients; `Ok(None)` without any.
  pub fn from_header(h: &FitsHeader) -> Result<Option<Sip>, String> {
    let a = read_poly(h, "A")?;
    let b = read_poly(h, "B")?;
    if a.is_none() && b.is_none() {
      return Ok(None);
    }
    let ap = read_poly(h, "AP")?;
    let bp = read_poly(h, "BP")?;
    let (a, a_order) = a.unwrap_or(([[0.0; N]; N], 0));
    let (b, b_order) = b.unwrap_or(([[0.0; N]; N], 0));
    let has_inverse = ap.is_some() || bp.is_some();
    let (ap, ap_order) = ap.unwrap_or(([[0.0; N]; N], 0));
    let (bp, bp_order) = bp.unwrap_or(([[0.0; N]; N], 0));
    Ok(Some(Sip { a, b, ap, bp, a_order, b_order, ap_order, bp_order, has_inverse }))
  }

  /// Pixel offsets from `CRPIX` → corrected offsets `(u + f(u, v), v + g(u, v))`.
  pub fn forward(&self, u: f64, v: f64) -> (f64, f64) {
    (u + eval(&self.a, self.a_order, u, v), v + eval(&self.b, self.b_order, u, v))
  }

  /// Corrected offsets → pixel offsets (Newton from the `AP` / `BP` guess, 1e-11 px).
  pub fn inverse(&self, cu: f64, cv: f64) -> Option<(f64, f64)> {
    let (mut u, mut v) = if self.has_inverse {
      (cu + eval(&self.ap, self.ap_order, cu, cv), cv + eval(&self.bp, self.bp_order, cu, cv))
    } else {
      (cu, cv)
    };
    for _ in 0..50 {
      let (fu, fv) = self.forward(u, v);
      let (ru, rv) = (fu - cu, fv - cv);
      if ru.abs() < 1e-11 && rv.abs() < 1e-11 {
        break;
      }
      let j11 = 1.0 + eval_du(&self.a, self.a_order, u, v);
      let j12 = eval_dv(&self.a, self.a_order, u, v);
      let j21 = eval_du(&self.b, self.b_order, u, v);
      let j22 = 1.0 + eval_dv(&self.b, self.b_order, u, v);
      let det = j11 * j22 - j12 * j21;
      if det == 0.0 || !det.is_finite() {
        return None;
      }
      u -= (j22 * ru - j12 * rv) / det;
      v -= (-j21 * ru + j11 * rv) / det;
    }
    (u.is_finite() && v.is_finite()).then_some((u, v))
  }
}

/// TPV polynomial coefficients of both axes (`PVi_0..39`; `PVi_1` defaults to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tpv {
  /// longitude axis: `x' = Σ lng[m] · term_m(x, y)`
  pub lng: [f64; 40],
  /// latitude axis: `y' = Σ lat[m] · term_m(y, x)`
  pub lat: [f64; 40],
}

impl Default for Tpv {
  fn default() -> Self {
    let mut t = Tpv { lng: [0.0; 40], lat: [0.0; 40] };
    t.lng[1] = 1.0;
    t.lat[1] = 1.0;
    t
  }
}

fn tpv_terms(xi: f64, eta: f64) -> [f64; 40] {
  let r = xi.hypot(eta);
  let (x2, y2, x3, y3) = (xi * xi, eta * eta, xi * xi * xi, eta * eta * eta);
  let (x4, y4) = (x2 * x2, y2 * y2);
  let (x5, y5) = (x4 * xi, y4 * eta);
  let (x6, y6) = (x3 * x3, y3 * y3);
  let (x7, y7) = (x6 * xi, y6 * eta);
  [
    1.0, xi, eta, r,
    x2, xi * eta, y2,
    x3, x2 * eta, xi * y2, y3, r * r * r,
    x4, x3 * eta, x2 * y2, xi * y3, y4,
    x5, x4 * eta, x3 * y2, x2 * y3, xi * y4, y5, r * r * r * r * r,
    x6, x5 * eta, x4 * y2, x3 * y3, x2 * y4, xi * y5, y6,
    x7, x6 * eta, x5 * y2, x4 * y3, x3 * y4, x2 * y5, xi * y6, y7, r * r * r * r * r * r * r,
  ]
}

fn tpv_eval(c: &[f64; 40], xi: f64, eta: f64) -> f64 {
  let t = tpv_terms(xi, eta);
  let mut s = 0.0;
  for m in 0..40 {
    if c[m] != 0.0 {
      s += c[m] * t[m];
    }
  }
  s
}

impl Tpv {
  /// Intermediate `(x, y)` (degrees, longitude / latitude order) → distorted.
  pub fn forward(&self, x: f64, y: f64) -> (f64, f64) {
    (tpv_eval(&self.lng, x, y), tpv_eval(&self.lat, y, x))
  }

  /// Distorted → undistorted intermediate coordinates (Newton, numerical Jacobian).
  pub fn inverse(&self, xd: f64, yd: f64) -> Option<(f64, f64)> {
    newton2(|x, y| self.forward(x, y), xd, yd, (xd, yd), 1e-14)
  }
}

/// Solves `f(x, y) = (xd, yd)` from the guess by Newton iteration with a central-difference
/// Jacobian (step `1e-6`), to `tol` in the residual or 50 iterations; `None` on a singular step.
pub fn newton2(f: impl Fn(f64, f64) -> (f64, f64), xd: f64, yd: f64, guess: (f64, f64), tol: f64) -> Option<(f64, f64)> {
  let (mut x, mut y) = guess;
  for _ in 0..50 {
    let (fx, fy) = f(x, y);
    let (rx, ry) = (fx - xd, fy - yd);
    if rx.abs() < tol && ry.abs() < tol {
      break;
    }
    let h = 1e-6;
    let (ax, ay) = f(x + h, y);
    let (bx, by) = f(x - h, y);
    let (cx, cy) = f(x, y + h);
    let (dx, dy) = f(x, y - h);
    let j11 = (ax - bx) / (2.0 * h);
    let j21 = (ay - by) / (2.0 * h);
    let j12 = (cx - dx) / (2.0 * h);
    let j22 = (cy - dy) / (2.0 * h);
    let det = j11 * j22 - j12 * j21;
    if det == 0.0 || !det.is_finite() {
      return None;
    }
    x -= (j22 * rx - j12 * ry) / det;
    y -= (-j21 * rx + j11 * ry) / det;
  }
  (x.is_finite() && y.is_finite()).then_some((x, y))
}

// ─── IRAF TNX / ZPX correction surfaces ──────────────────────────────────────

/// Largest `xorder` / `yorder` (number of terms per axis) of an IRAF surface accepted.
pub const IRAF_MAX_ORDER: usize = 10;
/// Largest number of coefficients (`xorder · yorder` with full cross terms).
pub const IRAF_MAX_COEFF: usize = IRAF_MAX_ORDER * IRAF_MAX_ORDER;

/// The function family of an IRAF `gsurfit` surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrafFunction {
  Chebyshev,
  Legendre,
  Polynomial,
}

/// Which cross terms an IRAF surface carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IrafCrossTerms {
  /// `xorder + yorder − 1` coefficients: the pure `x` powers, then the pure `y` powers
  None,
  /// `xorder · yorder` coefficients
  Full,
  /// `xorder · yorder − o (o − 1) / 2` (`o = min(xorder, yorder)`): terms of total degree below
  /// `max(xorder, yorder)`
  Half,
}

/// An IRAF `gsurfit` surface as written in a TNX / ZPX `lngcor` / `latcor` attribute: `type
/// xorder yorder xterms xmin xmax ymin ymax c…`, the coefficients in `C00 C10 C20 … C01 C11 …`
/// order (the `x` index running fastest). Chebyshev and Legendre polynomials take the
/// normalised `(2 x − (xmax + xmin)) / (xmax − xmin)`; the plain polynomial takes `x` itself
/// (IRAF `gsurfit`, wcstools `tnxpos.c`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrafSurface {
  pub function: IrafFunction,
  pub xorder: usize,
  pub yorder: usize,
  pub cross: IrafCrossTerms,
  pub xmin: f64,
  pub xmax: f64,
  pub ymin: f64,
  pub ymax: f64,
  pub ncoeff: usize,
  pub coeff: [f64; IRAF_MAX_COEFF],
}

impl IrafSurface {
  /// The number of coefficients a surface of these orders and cross terms has.
  pub fn coefficient_count(xorder: usize, yorder: usize, cross: IrafCrossTerms) -> usize {
    match cross {
      IrafCrossTerms::None => xorder + yorder - 1,
      IrafCrossTerms::Full => xorder * yorder,
      IrafCrossTerms::Half => {
        let o = xorder.min(yorder);
        xorder * yorder - o * (o - 1) / 2
      }
    }
  }

  /// Parses the numbers of a `lngcor` / `latcor` attribute value.
  pub fn from_numbers(v: &[f64]) -> Result<IrafSurface, String> {
    if v.len() < 8 {
      return Err(format!("IRAF surface: {} numbers, at least 8 needed", v.len()));
    }
    let function = match v[0] as i64 {
      1 => IrafFunction::Chebyshev,
      2 => IrafFunction::Legendre,
      3 => IrafFunction::Polynomial,
      t => return Err(format!("IRAF surface: unknown function type {t}")),
    };
    let (xorder, yorder) = (v[1] as i64, v[2] as i64);
    if !(1..=IRAF_MAX_ORDER as i64).contains(&xorder) || !(1..=IRAF_MAX_ORDER as i64).contains(&yorder) {
      return Err(format!("IRAF surface: orders {xorder} × {yorder} outside 1..={IRAF_MAX_ORDER}"));
    }
    let (xorder, yorder) = (xorder as usize, yorder as usize);
    let cross = match v[3] as i64 {
      0 => IrafCrossTerms::None,
      1 => IrafCrossTerms::Full,
      2 => IrafCrossTerms::Half,
      t => return Err(format!("IRAF surface: unknown cross-term type {t}")),
    };
    let (xmin, xmax, ymin, ymax) = (v[4], v[5], v[6], v[7]);
    if function != IrafFunction::Polynomial && (xmax == xmin || ymax == ymin) {
      return Err("IRAF surface: empty normalisation range".into());
    }
    let ncoeff = Self::coefficient_count(xorder, yorder, cross);
    if v.len() - 8 < ncoeff {
      return Err(format!("IRAF surface: {} coefficients, {ncoeff} needed", v.len() - 8));
    }
    let mut coeff = [0.0; IRAF_MAX_COEFF];
    coeff[..ncoeff].copy_from_slice(&v[8..8 + ncoeff]);
    Ok(IrafSurface { function, xorder, yorder, cross, xmin, xmax, ymin, ymax, ncoeff, coeff })
  }

  /// The `order` basis functions of one axis at `x`.
  fn basis(&self, x: f64, order: usize, lo: f64, hi: f64, out: &mut [f64; IRAF_MAX_ORDER]) {
    out[0] = 1.0;
    if order == 1 {
      return;
    }
    let xn = match self.function {
      IrafFunction::Polynomial => x,
      _ => (2.0 * x - (hi + lo)) / (hi - lo),
    };
    out[1] = xn;
    for i in 2..order {
      out[i] = match self.function {
        IrafFunction::Polynomial => xn * out[i - 1],
        IrafFunction::Chebyshev => 2.0 * xn * out[i - 1] - out[i - 2],
        IrafFunction::Legendre => ((2 * i - 1) as f64 * xn * out[i - 1] - (i - 1) as f64 * out[i - 2]) / i as f64,
      };
    }
  }

  /// The surface value at `(x, y)` (IRAF `gseval`).
  pub fn eval(&self, x: f64, y: f64) -> f64 {
    let mut xb = [0.0; IRAF_MAX_ORDER];
    let mut yb = [0.0; IRAF_MAX_ORDER];
    self.basis(x, self.xorder, self.xmin, self.xmax, &mut xb);
    self.basis(y, self.yorder, self.ymin, self.ymax, &mut yb);
    let maxorder = self.xorder.max(self.yorder) + 1;
    let mut xorder = self.xorder;
    let mut k = 0;
    let mut sum = 0.0;
    for i in 0..self.yorder {
      let mut acc = 0.0;
      for xb_k in xb.iter().take(xorder) {
        acc += self.coeff[k] * xb_k;
        k += 1;
      }
      sum += acc * yb[i];
      match self.cross {
        IrafCrossTerms::None => xorder = 1,
        IrafCrossTerms::Half if i + 1 + self.xorder + 1 > maxorder => xorder -= 1,
        _ => {}
      }
    }
    sum
  }
}

/// The TNX / ZPX correction: `x' = x + lngcor(x, y)`, `y' = y + latcor(x, y)` on the
/// intermediate coordinates (degrees); either surface may be absent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrafCorrection {
  pub lng: Option<IrafSurface>,
  pub lat: Option<IrafSurface>,
}

impl IrafCorrection {
  pub fn forward(&self, x: f64, y: f64) -> (f64, f64) {
    (x + self.lng.map_or(0.0, |s| s.eval(x, y)), y + self.lat.map_or(0.0, |s| s.eval(x, y)))
  }

  pub fn inverse(&self, xd: f64, yd: f64) -> Option<(f64, f64)> {
    newton2(|x, y| self.forward(x, y), xd, yd, (xd, yd), 1e-14)
  }
}

// ─── DSS plate solution ──────────────────────────────────────────────────────

/// The Digitized Sky Survey plate solution (`AMDX1..13`, `AMDY1..13`; the GSC / DSS "Getimage"
/// convention, wcstools `dsspos.c`): plate millimetres `(x, y)` from the plate centre →
/// standard coordinates `(ξ, η)` in arcseconds,
/// `ξ = a1 x + a2 y + a3 + a4 x² + a5 x y + a6 y² + a7 (x² + y²) + a8 x³ + a9 x² y + a10 x y² +
/// a11 y³ + a12 x (x² + y²) + a13 x (x² + y²)²` and `η` the same with `x ↔ y` and the `AMDY`
/// coefficients. `AMDX14..20` (magnitude and colour terms) are not used by any DSS reader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DssPlate {
  pub amdx: [f64; 13],
  pub amdy: [f64; 13],
}

fn amd(c: &[f64; 13], x: f64, y: f64) -> f64 {
  let (x2, y2) = (x * x, y * y);
  let r2 = x2 + y2;
  c[0] * x
    + c[1] * y
    + c[2]
    + c[3] * x2
    + c[4] * x * y
    + c[5] * y2
    + c[6] * r2
    + c[7] * x2 * x
    + c[8] * x2 * y
    + c[9] * x * y2
    + c[10] * y2 * y
    + c[11] * x * r2
    + c[12] * x * r2 * r2
}

impl DssPlate {
  /// Reads `AMDX1..13` / `AMDY1..13`; `Ok(None)` without `AMDX1` and `AMDY1`.
  pub fn from_header(h: &FitsHeader) -> Result<Option<DssPlate>, String> {
    if h.get("AMDX1").is_none() && h.get("AMDY1").is_none() {
      return Ok(None);
    }
    let mut amdx = [0.0; 13];
    let mut amdy = [0.0; 13];
    for i in 0..13 {
      for (key, slot) in [("AMDX", &mut amdx[i]), ("AMDY", &mut amdy[i])] {
        let k = format!("{key}{}", i + 1);
        if let Some(v) = h.get(&k) {
          *slot = v.replace(['D', 'd'], "E").trim().parse::<f64>().map_err(|_| format!("{k}: not a number"))?;
        }
      }
    }
    Ok(Some(DssPlate { amdx, amdy }))
  }

  /// Plate millimetres from the centre → `(ξ, η)` degrees.
  pub fn forward(&self, xmm: f64, ymm: f64) -> (f64, f64) {
    (amd(&self.amdx, xmm, ymm) / 3600.0, amd(&self.amdy, ymm, xmm) / 3600.0)
  }

  /// `(ξ, η)` degrees → plate millimetres (Newton from the linear terms).
  pub fn inverse(&self, xi: f64, eta: f64) -> Option<(f64, f64)> {
    // linear guess: ξ″ ≈ a1 x + a2 y, η″ ≈ b1 y + b2 x
    let (a1, a2, b1, b2) = (self.amdx[0], self.amdx[1], self.amdy[0], self.amdy[1]);
    let det = a1 * b1 - a2 * b2;
    let guess = if det != 0.0 && det.is_finite() {
      let (xs, ys) = (xi * 3600.0, eta * 3600.0);
      ((b1 * xs - a2 * ys) / det, (a1 * ys - b2 * xs) / det)
    } else {
      (0.0, 0.0)
    };
    newton2(|x, y| self.forward(x, y), xi, eta, guess, 1e-12)
  }
}

/// The distortion of a WCS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distortion {
  None,
  Sip(Sip),
  Tpv(Tpv),
  /// IRAF TNX: the correction surfaces on the TAN intermediate coordinates
  Tnx(IrafCorrection),
  /// IRAF ZPX: the correction surfaces on the ZPN intermediate coordinates
  Zpx(IrafCorrection),
  /// DSS plate solution on plate millimetres (the linear part is the pixel size)
  Dss(DssPlate),
}

impl Distortion {
  pub fn name(&self) -> &'static str {
    match self {
      Distortion::None => "none",
      Distortion::Sip(_) => "SIP",
      Distortion::Tpv(_) => "TPV",
      Distortion::Tnx(_) => "TNX",
      Distortion::Zpx(_) => "ZPX",
      Distortion::Dss(_) => "DSS",
    }
  }
}
