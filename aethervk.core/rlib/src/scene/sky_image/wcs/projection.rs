//! Spherical projections of FITS WCS Paper II (Calabretta & Greisen 2002) and the HEALPix
//! projections of Calabretta & Lowe 2007: native spherical `(φ, θ)` ↔ projection plane `(x, y)`,
//! all in degrees, the conventions and the `PVi_m` defaults of wcslib's `prj.c`.
//!
//! Forward: [`Projection::fwd`] `(φ, θ) → (x, y)`, `None` where the projection is not defined.
//! Inverse: [`Projection::inv`] `(x, y) → (φ, θ)`, `None` outside the projection's image.
//! CSC uses wcslib's single-precision polynomial constants and arithmetic so the two agree.

use alloc::{format, string::String};

/// degrees per radian (`R0` of Paper II)
pub const R0: f64 = 180.0 / core::f64::consts::PI;
const D: f64 = core::f64::consts::PI / 180.0;
const TOL: f64 = 1e-10;

pub(crate) fn sind(a: f64) -> f64 {
  (a * D).sin()
}
pub(crate) fn cosd(a: f64) -> f64 {
  (a * D).cos()
}
pub(crate) fn tand(a: f64) -> f64 {
  (a * D).tan()
}
pub(crate) fn asind(x: f64) -> f64 {
  x.clamp(-1.0, 1.0).asin() / D
}
pub(crate) fn acosd(x: f64) -> f64 {
  x.clamp(-1.0, 1.0).acos() / D
}
pub(crate) fn atand(x: f64) -> f64 {
  x.atan() / D
}
pub(crate) fn atan2d(y: f64, x: f64) -> f64 {
  y.atan2(x) / D
}

/// The three-letter projection code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
  Azp,
  Szp,
  Tan,
  Stg,
  Sin,
  Arc,
  Zpn,
  Zea,
  Air,
  Cyp,
  Cea,
  Car,
  Mer,
  Sfl,
  Par,
  Mol,
  Ait,
  Cop,
  Coe,
  Cod,
  Coo,
  Bon,
  Pco,
  Tsc,
  Csc,
  Qsc,
  Hpx,
  Xph,
}

impl Code {
  /// `CTYPE` algorithm code → projection; `None` for a code this module has not got.
  pub fn parse(s: &str) -> Option<Code> {
    Some(match s {
      "AZP" => Code::Azp,
      "SZP" => Code::Szp,
      "TAN" => Code::Tan,
      "STG" => Code::Stg,
      "SIN" | "NCP" => Code::Sin,
      "ARC" => Code::Arc,
      "ZPN" => Code::Zpn,
      "ZEA" => Code::Zea,
      "AIR" => Code::Air,
      "CYP" => Code::Cyp,
      "CEA" => Code::Cea,
      "CAR" => Code::Car,
      "MER" => Code::Mer,
      "SFL" | "GLS" => Code::Sfl,
      "PAR" => Code::Par,
      "MOL" => Code::Mol,
      "AIT" => Code::Ait,
      "COP" => Code::Cop,
      "COE" => Code::Coe,
      "COD" => Code::Cod,
      "COO" => Code::Coo,
      "BON" => Code::Bon,
      "PCO" => Code::Pco,
      "TSC" => Code::Tsc,
      "CSC" => Code::Csc,
      "QSC" => Code::Qsc,
      "HPX" => Code::Hpx,
      "XPH" => Code::Xph,
      _ => return None,
    })
  }

  pub fn name(self) -> &'static str {
    match self {
      Code::Azp => "AZP",
      Code::Szp => "SZP",
      Code::Tan => "TAN",
      Code::Stg => "STG",
      Code::Sin => "SIN",
      Code::Arc => "ARC",
      Code::Zpn => "ZPN",
      Code::Zea => "ZEA",
      Code::Air => "AIR",
      Code::Cyp => "CYP",
      Code::Cea => "CEA",
      Code::Car => "CAR",
      Code::Mer => "MER",
      Code::Sfl => "SFL",
      Code::Par => "PAR",
      Code::Mol => "MOL",
      Code::Ait => "AIT",
      Code::Cop => "COP",
      Code::Coe => "COE",
      Code::Cod => "COD",
      Code::Coo => "COO",
      Code::Bon => "BON",
      Code::Pco => "PCO",
      Code::Tsc => "TSC",
      Code::Csc => "CSC",
      Code::Qsc => "QSC",
      Code::Hpx => "HPX",
      Code::Xph => "XPH",
    }
  }

  /// Zenithal (azimuthal) projections: the native pole is the fiducial point.
  pub fn is_zenithal(self) -> bool {
    matches!(
      self,
      Code::Azp
        | Code::Szp
        | Code::Tan
        | Code::Stg
        | Code::Sin
        | Code::Arc
        | Code::Zpn
        | Code::Zea
        | Code::Air
        | Code::Xph
    )
  }

  fn is_conic(self) -> bool {
    matches!(self, Code::Cop | Code::Coe | Code::Cod | Code::Coo)
  }
}

/// `PVi_m` of the latitude axis, `m = 0..30`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PvParams {
  pub value: [f64; 30],
  pub set: [bool; 30],
}

impl Default for PvParams {
  fn default() -> Self {
    PvParams { value: [0.0; 30], set: [false; 30] }
  }
}

impl PvParams {
  pub fn get(&self, m: usize, default: f64) -> f64 {
    if m < 30 && self.set[m] { self.value[m] } else { default }
  }
  pub fn set(&mut self, m: usize, v: f64) {
    if m < 30 {
      self.value[m] = v;
      self.set[m] = true;
    }
  }
}

/// A projection with its parameters and the derived constants (wcslib `prjset`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
  pub code: Code,
  /// native longitude / latitude of the fiducial point (degrees)
  pub phi0: f64,
  pub theta0: f64,
  pv: PvParams,
  /// ZPN: polynomial degree, `θ` of the maximum radius and that radius (`R0` units, i.e. degrees
  /// of arc / `R0`); AIR: `w1`; conics: `C`, `Y0`, `θa`, …; SZP: the projection point;
  /// BON: `θ1`, `Y0`; HPX: `H`, `K`
  w: [f64; 8],
  zpn_n: usize,
}

impl Projection {
  /// Builds the projection; `phi0` / `theta0` override the projection's defaults (`PVi_1`,
  /// `PVi_2` of the longitude axis). The conic `θa` (`PVi_1` of the latitude axis) is required.
  pub fn new(code: Code, pv: PvParams, phi0: Option<f64>, theta0: Option<f64>) -> Result<Self, String> {
    let mut p = Projection {
      code,
      phi0: phi0.unwrap_or(0.0),
      theta0: 0.0,
      pv,
      w: [0.0; 8],
      zpn_n: 0,
    };
    p.theta0 = match theta0 {
      Some(t) => t,
      None if code.is_zenithal() => 90.0,
      None if code.is_conic() => {
        if !pv.set[1] {
          return Err(format!("{}: PVi_1 (theta_a) missing", code.name()));
        }
        pv.value[1]
      }
      None => 0.0,
    };
    p.setup()?;
    Ok(p)
  }

  fn pv(&self, m: usize, default: f64) -> f64 {
    self.pv.get(m, default)
  }

  fn setup(&mut self) -> Result<(), String> {
    match self.code {
      Code::Zpn => {
        let mut n = 0;
        for m in 0..30 {
          if self.pv(m, 0.0) != 0.0 {
            n = m;
          }
        }
        self.zpn_n = n;
        if n == 0 {
          return Err("ZPN: all PVi_m zero".into());
        }
        // first zero of dR/dζ in (0, π]: the largest radius the polynomial reaches monotonically
        let d_r = |z: f64| -> f64 {
          let mut s = 0.0;
          for m in 1..=n {
            s += m as f64 * self.pv(m, 0.0) * z.powi(m as i32 - 1);
          }
          s
        };
        if d_r(0.0) <= 0.0 {
          return Err("ZPN: PVi_1 <= 0".into());
        }
        let step = 1e-3;
        let mut zmax = core::f64::consts::PI;
        let mut z = step;
        while z <= core::f64::consts::PI + 1e-12 {
          if d_r(z) <= 0.0 {
            let (mut a, mut b) = (z - step, z);
            for _ in 0..60 {
              let mid = 0.5 * (a + b);
              if d_r(mid) > 0.0 {
                a = mid;
              } else {
                b = mid;
              }
            }
            zmax = 0.5 * (a + b);
            break;
          }
          z += step;
        }
        self.w[0] = zmax;
        self.w[1] = self.zpn_poly(zmax);
      }
      Code::Air => {
        let tb = self.pv(1, 90.0);
        self.w[0] = if tb == 90.0 {
          -0.5
        } else if tb > -90.0 {
          let xb = (90.0 - tb) / 2.0;
          cosd(xb).ln() / tand(xb).powi(2)
        } else {
          return Err("AIR: PVi_1 <= -90".into());
        };
      }
      Code::Cop | Code::Coe | Code::Cod | Code::Coo => {
        let ta = self.pv(1, 0.0);
        let eta = self.pv(2, 0.0);
        let (t1, t2) = (ta - eta, ta + eta);
        self.w[2] = ta;
        if ta == 0.0 || ta.abs() > 90.0 {
          return Err(format!("{}: PVi_1 (theta_a) {ta} invalid", self.code.name()));
        }
        match self.code {
          Code::Cop => {
            self.w[0] = sind(ta);
            self.w[3] = cosd(eta);
            self.w[1] = R0 * self.w[3] / tand(ta);
          }
          Code::Coe => {
            let (s1, s2) = (sind(t1), sind(t2));
            let g = s1 + s2;
            self.w[3] = g;
            self.w[0] = g / 2.0;
            self.w[4] = s1 * s2;
            self.w[1] = (2.0 * R0 / g) * (1.0 + s1 * s2 - g * sind(ta)).sqrt();
          }
          Code::Cod => {
            if eta == 0.0 {
              self.w[0] = sind(ta);
              self.w[1] = R0 / tand(ta);
            } else {
              self.w[0] = sind(ta) * sind(eta) / (eta * D);
              self.w[1] = R0 * (eta * D) / tand(eta) / tand(ta);
            }
          }
          _ => {
            let c = if t1 == t2 {
              sind(t1)
            } else {
              (cosd(t2) / cosd(t1)).ln() / (tand((90.0 - t2) / 2.0) / tand((90.0 - t1) / 2.0)).ln()
            };
            self.w[0] = c;
            let psi = R0 * cosd(t1) / (c * tand((90.0 - t1) / 2.0).powf(c));
            self.w[3] = psi;
            self.w[1] = psi * tand((90.0 - ta) / 2.0).powf(c);
          }
        }
      }
      Code::Bon => {
        let t1 = self.pv(1, 0.0);
        if t1 == 0.0 {
          self.code = Code::Sfl;
        } else {
          self.w[0] = t1;
          self.w[1] = R0 / tand(t1) + t1;
        }
      }
      Code::Szp => {
        let (mu, pc, tc) = (self.pv(1, 0.0), self.pv(2, 0.0), self.pv(3, 90.0));
        self.w[0] = -mu * cosd(tc) * sind(pc);
        self.w[1] = mu * cosd(tc) * cosd(pc);
        self.w[2] = -mu * sind(tc);
      }
      Code::Hpx => {
        self.w[0] = self.pv(1, 4.0);
        self.w[1] = self.pv(2, 3.0);
        if self.w[0] <= 0.0 || self.w[1] <= 0.0 {
          return Err("HPX: H and K must be positive".into());
        }
      }
      _ => {}
    }
    Ok(())
  }

  fn zpn_poly(&self, z: f64) -> f64 {
    let mut s = 0.0;
    for m in (0..=self.zpn_n).rev() {
      s = s * z + self.pv(m, 0.0);
    }
    s
  }

  /// `(φ, θ) → (x, y)` degrees; `None` where the projection is undefined.
  pub fn fwd(&self, phi: f64, theta: f64) -> Option<(f64, f64)> {
    let (st, ct) = (sind(theta), cosd(theta));
    let (sp, cp) = (sind(phi), cosd(phi));
    match self.code {
      Code::Tan | Code::Stg | Code::Arc | Code::Zea | Code::Air | Code::Zpn => {
        let r = match self.code {
          Code::Tan => {
            if st <= 0.0 {
              return None;
            }
            R0 * ct / st
          }
          Code::Stg => {
            if 1.0 + st == 0.0 {
              return None;
            }
            2.0 * R0 * ct / (1.0 + st)
          }
          Code::Arc => 90.0 - theta,
          Code::Zea => 2.0 * R0 * sind((90.0 - theta) / 2.0),
          Code::Air => {
            if theta == 90.0 {
              0.0
            } else {
              let xi = (90.0 - theta) / 2.0 * D;
              -2.0 * R0 * (xi.cos().ln() / xi.tan() + self.w[0] * xi.tan())
            }
          }
          _ => {
            let z = (90.0 - theta) * D;
            if z > self.w[0] + TOL {
              return None;
            }
            R0 * self.zpn_poly(z)
          }
        };
        Some((r * sp, -r * cp))
      }
      Code::Azp => {
        let (mu, gam) = (self.pv(1, 0.0), self.pv(2, 0.0));
        let tg = tand(gam);
        let den = mu + st + ct * cp * tg;
        if den <= 0.0 {
          return None;
        }
        if mu.abs() > 1.0 && theta < asind(-1.0 / mu) {
          return None;
        }
        let r = R0 * (mu + 1.0) * ct / den;
        Some((r * sp, -r * cp / cosd(gam)))
      }
      Code::Sin => {
        let (xi, eta) = (self.pv(1, 0.0), self.pv(2, 0.0));
        if xi == 0.0 && eta == 0.0 {
          if st < 0.0 {
            return None;
          }
          return Some((R0 * ct * sp, -R0 * ct * cp));
        }
        let t = -atand(xi * sp - eta * cp);
        if theta < t {
          return None;
        }
        let z = 1.0 - st;
        Some((R0 * (ct * sp + xi * z), -R0 * (ct * cp - eta * z)))
      }
      Code::Szp => {
        let [xp, yp, zp, ..] = self.w;
        let (x, y, z) = (ct * sp, -ct * cp, st);
        if z == zp {
          return None;
        }
        let t = (1.0 - zp) / (z - zp);
        let q = [xp + t * (x - xp), yp + t * (y - yp), 1.0];
        // visible: the ray leaves the sphere at the point towards the plane
        if (q[0] - x) * x + (q[1] - y) * y + (q[2] - z) * z < -TOL || t < -TOL {
          return None;
        }
        Some((R0 * q[0], R0 * q[1]))
      }
      Code::Cyp => {
        let (mu, lam) = (self.pv(1, 1.0), self.pv(2, 1.0));
        let den = mu + ct;
        if den == 0.0 {
          return None;
        }
        Some((lam * phi, R0 * (mu + lam) * st / den))
      }
      Code::Cea => Some((phi, R0 * st / self.pv(1, 1.0))),
      Code::Car => Some((phi, theta)),
      Code::Mer => {
        if theta.abs() >= 90.0 {
          return None;
        }
        Some((phi, R0 * tand((90.0 + theta) / 2.0).ln()))
      }
      Code::Sfl => Some((phi * ct, theta)),
      Code::Par => Some((phi * (2.0 * cosd(2.0 * theta / 3.0) - 1.0), 180.0 * sind(theta / 3.0))),
      Code::Mol => {
        // solve v + sin v = π sin θ for v in [−π, π] (monotone): bisection, then Newton polish
        let u = core::f64::consts::PI * st;
        let (mut lo, mut hi) = (-core::f64::consts::PI, core::f64::consts::PI);
        let mut v = u.clamp(lo, hi);
        for _ in 0..200 {
          let f = v + v.sin() - u;
          if f.abs() < 1e-15 {
            break;
          }
          if f < 0.0 {
            lo = v;
          } else {
            hi = v;
          }
          let dv = 1.0 + v.cos();
          let vn = if dv > 1e-6 { v - f / dv } else { 0.5 * (lo + hi) };
          v = if vn > lo && vn < hi { vn } else { 0.5 * (lo + hi) };
          if hi - lo < 1e-16 {
            break;
          }
        }
        let g = v / 2.0;
        Some((
          2.0 * core::f64::consts::SQRT_2 / core::f64::consts::PI * phi * g.cos(),
          core::f64::consts::SQRT_2 * R0 * g.sin(),
        ))
      }
      Code::Ait => {
        let w = (2.0 / (1.0 + ct * cosd(phi / 2.0))).sqrt();
        Some((2.0 * R0 * w * ct * sind(phi / 2.0), R0 * w * st))
      }
      Code::Cop | Code::Coe | Code::Cod | Code::Coo => {
        let (c, y0, ta) = (self.w[0], self.w[1], self.w[2]);
        let r = match self.code {
          Code::Cop => {
            let t = theta - ta;
            if cosd(t) == 0.0 {
              return None;
            }
            let r = R0 * self.w[3] * (1.0 / tand(ta) - tand(t));
            if r * c < 0.0 {
              return None;
            }
            r
          }
          Code::Coe => {
            let w = 1.0 + self.w[4] - self.w[3] * st;
            if w < 0.0 {
              return None;
            }
            (2.0 * R0 / self.w[3]) * w.sqrt()
          }
          Code::Cod => ta - theta + y0,
          _ => {
            let tt = tand((90.0 - theta) / 2.0);
            if tt > 0.0 {
              self.w[3] * tt.powf(c)
            } else if c > 0.0 {
              0.0
            } else {
              return None;
            }
          }
        };
        let a = c * phi;
        Some((r * sind(a), -r * cosd(a) + y0))
      }
      Code::Bon => {
        let y0 = self.w[1];
        let r = y0 - theta;
        let a = if r == 0.0 { 0.0 } else { R0 * phi * ct / r };
        Some((r * sind(a), -r * cosd(a) + y0))
      }
      Code::Pco => {
        if theta == 0.0 {
          return Some((phi, 0.0));
        }
        if theta.abs() == 90.0 {
          return Some((0.0, theta));
        }
        let cot = R0 * ct / st;
        let a = phi * st;
        Some((cot * sind(a), cot * (1.0 - cosd(a)) + theta))
      }
      Code::Tsc | Code::Csc | Code::Qsc => {
        let (l, m, n) = (ct * cp, ct * sp, st);
        let (mut face, mut zeta) = (0, n);
        if l > zeta {
          face = 1;
          zeta = l;
        }
        if m > zeta {
          face = 2;
          zeta = m;
        }
        if -l > zeta {
          face = 3;
          zeta = -l;
        }
        if -m > zeta {
          face = 4;
          zeta = -m;
        }
        if -n > zeta {
          face = 5;
          zeta = -n;
        }
        let (xi, eta, x0, y0) = match face {
          0 => (m, -l, 0.0, 2.0),
          1 => (m, n, 0.0, 0.0),
          2 => (-l, n, 2.0, 0.0),
          3 => (-m, n, 4.0, 0.0),
          4 => (l, n, 6.0, 0.0),
          _ => (m, l, 0.0, -2.0),
        };
        let (xf, yf) = match self.code {
          Code::Tsc => (xi / zeta, eta / zeta),
          Code::Csc => csc_fwd(xi / zeta, eta / zeta),
          _ => {
            if xi == 0.0 && eta == 0.0 {
              (0.0, 0.0)
            } else if xi.abs() >= eta.abs() {
              let om = eta / xi;
              let tau = 1.0 + om * om;
              let mut xf = ((1.0 - zeta) / (1.0 - 1.0 / (1.0 + tau).sqrt())).sqrt();
              if xi < 0.0 {
                xf = -xf;
              }
              let yf = (xf / 15.0) * (atand(om) - asind(om / (tau + tau).sqrt()));
              (xf, yf)
            } else {
              let om = xi / eta;
              let tau = 1.0 + om * om;
              let mut yf = ((1.0 - zeta) / (1.0 - 1.0 / (1.0 + tau).sqrt())).sqrt();
              if eta < 0.0 {
                yf = -yf;
              }
              let xf = (yf / 15.0) * (atand(om) - asind(om / (tau + tau).sqrt()));
              (xf, yf)
            }
          }
        };
        Some((45.0 * (xf + x0), 45.0 * (yf + y0)))
      }
      Code::Hpx => {
        let (h, k) = (self.w[0], self.w[1]);
        if st.abs() <= (k - 1.0) / k {
          return Some((phi, 90.0 * k / h * st));
        }
        let sig = (k * (1.0 - st.abs())).sqrt();
        let mut ph = phi;
        if ph >= 180.0 {
          ph -= 360.0;
        }
        let mut kk = ((ph + 180.0) * h / 360.0).floor();
        if kk >= h {
          kk = h - 1.0;
        }
        let xc = -180.0 + (2.0 * kk + 1.0) * 180.0 / h;
        let x = xc + (ph - xc) * sig;
        let y = ((180.0 / h) * ((k + 1.0) / 2.0 - sig)).copysign(theta);
        Some((x, y))
      }
      Code::Xph => {
        let mut ph = phi.rem_euclid(360.0);
        if ph >= 180.0 {
          ph -= 360.0;
        }
        let k = ((ph + 180.0) / 90.0).floor().clamp(0.0, 3.0);
        let xc = -135.0 + 90.0 * k;
        let (hx, hy) = if st.abs() <= 2.0 / 3.0 {
          (ph, 67.5 * st)
        } else {
          let sig = (3.0 * (1.0 - st.abs())).sqrt();
          (xc + (ph - xc) * sig, (45.0 * (2.0 - sig)).copysign(theta))
        };
        let (rx, ry) = (hx - xc, hy - 90.0);
        let (sr, cr) = (sind(xc), cosd(xc));
        Some((cr * rx - sr * ry, sr * rx + cr * ry))
      }
    }
  }

  /// `(x, y) → (φ, θ)` degrees (`φ` in `(−180, 180]`); `None` outside the projection's image.
  pub fn inv(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    match self.code {
      Code::Tan | Code::Stg | Code::Arc | Code::Zea | Code::Air | Code::Zpn => {
        let r = x.hypot(y);
        let phi = if r == 0.0 { 0.0 } else { atan2d(x, -y) };
        let theta = match self.code {
          Code::Tan => atan2d(R0, r),
          Code::Stg => 90.0 - 2.0 * atand(r / (2.0 * R0)),
          Code::Arc => {
            if r > 180.0 + TOL {
              return None;
            }
            90.0 - r
          }
          Code::Zea => {
            let q = r / (2.0 * R0);
            if q > 1.0 + TOL {
              return None;
            }
            90.0 - 2.0 * asind(q)
          }
          Code::Air => {
            let rr = r / (2.0 * R0);
            let xi = if rr == 0.0 {
              0.0
            } else {
              let f = |xi: f64| -(xi.cos().ln() / xi.tan() + self.w[0] * xi.tan()) - rr;
              let (mut a, mut b) = (1e-12, core::f64::consts::FRAC_PI_2 - 1e-12);
              if f(b) < 0.0 {
                return None;
              }
              for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if f(mid) < 0.0 {
                  a = mid;
                } else {
                  b = mid;
                }
                if b - a < 1e-16 {
                  break;
                }
              }
              0.5 * (a + b)
            };
            90.0 - 2.0 * xi / D
          }
          _ => {
            let rr = r / R0;
            let n = self.zpn_n;
            if rr > self.w[1] + TOL {
              return None;
            }
            let (c0, c1, c2) = (self.pv(0, 0.0), self.pv(1, 0.0), self.pv(2, 0.0));
            let mut zd = if n == 1 {
              (rr - c0) / c1
            } else if n == 2 {
              let (a, b, cc) = (c2, c1, c0 - rr);
              let mut d = b * b - 4.0 * a * cc;
              if d < 0.0 {
                if d < -TOL {
                  return None;
                }
                d = 0.0;
              }
              let d = d.sqrt();
              let (z1, z2) = ((-b - d) / (2.0 * a), (-b + d) / (2.0 * a));
              let zd = z1.min(z2);
              if zd < -TOL { z1.max(z2) } else { zd }
            } else if rr < c0 {
              if rr < c0 - TOL {
                return None;
              }
              0.0
            } else {
              let (mut a, mut b) = (0.0, self.w[0]);
              for _ in 0..200 {
                let mid = 0.5 * (a + b);
                if self.zpn_poly(mid) < rr {
                  a = mid;
                } else {
                  b = mid;
                }
                if b - a < 1e-16 {
                  break;
                }
              }
              0.5 * (a + b)
            };
            if zd < 0.0 {
              if zd < -TOL {
                return None;
              }
              zd = 0.0;
            }
            if zd > core::f64::consts::PI {
              if zd > core::f64::consts::PI + TOL {
                return None;
              }
              zd = core::f64::consts::PI;
            }
            90.0 - zd / D
          }
        };
        Some((phi, theta))
      }
      Code::Azp => {
        let (mu, gam) = (self.pv(1, 0.0), self.pv(2, 0.0));
        let (sg, cg) = (sind(gam), cosd(gam));
        let yc = y * cg;
        let r = x.hypot(yc);
        let (phi, rho) = if r == 0.0 {
          (0.0, 0.0)
        } else {
          let den = R0 * (mu + 1.0) + y * sg;
          if den == 0.0 {
            return None;
          }
          (atan2d(x, -yc), r / den)
        };
        let mut sv = rho * mu / (rho * rho + 1.0).sqrt();
        if sv.abs() > 1.0 {
          if sv.abs() > 1.0 + TOL {
            return None;
          }
          sv = 1.0f64.copysign(sv);
        }
        let psi = atan2d(1.0, rho);
        let om = asind(sv);
        let (t1, t2) = (psi - om, psi + om - 180.0);
        let (ok1, ok2) = (t1.abs() <= 90.0 + TOL, t2.abs() <= 90.0 + TOL);
        let theta = match (ok1, ok2) {
          (true, true) => {
            if mu.abs() >= 1.0 {
              t1.max(t2)
            } else {
              t1
            }
          }
          (true, false) => t1,
          (false, true) => t2,
          (false, false) => return None,
        };
        Some((phi, theta.clamp(-90.0, 90.0)))
      }
      Code::Sin => {
        let (xi, eta) = (self.pv(1, 0.0), self.pv(2, 0.0));
        let (x0, y0) = (x / R0, y / R0);
        if xi == 0.0 && eta == 0.0 {
          let mut r2 = x0 * x0 + y0 * y0;
          if r2 > 1.0 {
            if r2 > 1.0 + TOL {
              return None;
            }
            r2 = 1.0;
          }
          let phi = if r2 == 0.0 { 0.0 } else { atan2d(x0, -y0) };
          return Some((phi, acosd(r2.sqrt())));
        }
        let a = xi * xi + eta * eta + 1.0;
        let b = xi * (x0 - xi) + eta * (y0 - eta);
        let cc = (x0 - xi).powi(2) + (y0 - eta).powi(2) - 1.0;
        let mut d = b * b - a * cc;
        if d < 0.0 {
          if d < -TOL {
            return None;
          }
          d = 0.0;
        }
        let d = d.sqrt();
        let mut st = (-b + d) / a;
        if st.abs() > 1.0 {
          if st.abs() > 1.0 + TOL {
            st = (-b - d) / a;
            if st.abs() > 1.0 + TOL {
              return None;
            }
          }
          st = st.clamp(-1.0, 1.0);
        }
        let z = 1.0 - st;
        let (xx, yy) = (x0 - xi * z, -y0 + eta * z);
        let phi = if xx == 0.0 && yy == 0.0 { 0.0 } else { atan2d(xx, yy) };
        Some((phi, atan2d(st, xx.hypot(yy))))
      }
      Code::Szp => {
        let p = [self.w[0], self.w[1], self.w[2]];
        let q = [x / R0, y / R0, 1.0];
        let dv = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
        let a = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
        let b = 2.0 * (q[0] * dv[0] + q[1] * dv[1] + q[2] * dv[2]);
        let cc = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0;
        if a == 0.0 {
          return None;
        }
        let mut d = b * b - 4.0 * a * cc;
        if d < 0.0 {
          if d < -TOL {
            return None;
          }
          d = 0.0;
        }
        let d = d.sqrt();
        let mut best: Option<(f64, f64, f64)> = None;
        for t in [(-b - d) / (2.0 * a), (-b + d) / (2.0 * a)] {
          if t < -TOL {
            continue;
          }
          let (xx, yy, zz) = (q[0] + t * dv[0], q[1] + t * dv[1], q[2] + t * dv[2]);
          let phi = if xx == 0.0 && yy == 0.0 { 0.0 } else { atan2d(xx, -yy) };
          let th = atan2d(zz, xx.hypot(yy));
          if let Some((fx, fy)) = self.fwd(phi, th) {
            let e = (fx - x).abs() + (fy - y).abs();
            if best.is_none_or(|b| e < b.0) {
              best = Some((e, phi, th));
            }
          }
        }
        match best {
          Some((e, phi, th)) if e <= 1e-6 => Some((phi, th)),
          _ => None,
        }
      }
      Code::Cyp => {
        let (mu, lam) = (self.pv(1, 1.0), self.pv(2, 1.0));
        let eta = y / (R0 * (mu + lam));
        let theta = atan2d(eta, 1.0) + asind(eta * mu / (eta * eta + 1.0).sqrt());
        Some((x / lam, theta))
      }
      Code::Cea => {
        let q = self.pv(1, 1.0) * y / R0;
        if q.abs() > 1.0 + TOL {
          return None;
        }
        Some((x, asind(q)))
      }
      Code::Car => Some((x, y)),
      Code::Mer => Some((x, 2.0 * atand((y / R0).exp()) - 90.0)),
      Code::Sfl => {
        let ct = cosd(y);
        if ct == 0.0 {
          if x != 0.0 {
            return None;
          }
          return Some((0.0, y));
        }
        Some((x / ct, y))
      }
      Code::Par => {
        let q = y / 180.0;
        if q.abs() > 1.0 + TOL {
          return None;
        }
        let theta = 3.0 * asind(q);
        let w = 1.0 - 4.0 * q * q;
        if w == 0.0 {
          if x != 0.0 {
            return None;
          }
          return Some((0.0, theta));
        }
        Some((x / w, theta))
      }
      Code::Mol => {
        let mut z = y / (core::f64::consts::SQRT_2 * R0);
        if z.abs() > 1.0 {
          if z.abs() > 1.0 + TOL {
            return None;
          }
          z = 1.0f64.copysign(z);
        }
        let cg = (1.0 - z * z).sqrt();
        let phi = if cg == 0.0 {
          if x != 0.0 {
            return None;
          }
          0.0
        } else {
          core::f64::consts::PI * x / (2.0 * core::f64::consts::SQRT_2 * cg)
        };
        let st = (2.0 * z.asin() + 2.0 * z * cg) / core::f64::consts::PI;
        if st.abs() > 1.0 + TOL {
          return None;
        }
        Some((phi, asind(st)))
      }
      Code::Ait => {
        let mut z2 = 1.0 - (x / (4.0 * R0)).powi(2) - (y / (2.0 * R0)).powi(2);
        if z2 < 0.5 {
          if z2 < 0.5 - TOL {
            return None;
          }
          z2 = 0.5;
        }
        let z = z2.sqrt();
        let (x0, y0) = (2.0 * z2 - 1.0, z * x / (2.0 * R0));
        let phi = if x0 == 0.0 && y0 == 0.0 { 0.0 } else { 2.0 * atan2d(y0, x0) };
        let t = z * y / R0;
        if t.abs() > 1.0 + TOL {
          return None;
        }
        Some((phi, asind(t)))
      }
      Code::Cop | Code::Coe | Code::Cod | Code::Coo => {
        let (c, y0, ta) = (self.w[0], self.w[1], self.w[2]);
        let dy = y0 - y;
        let mut r = x.hypot(dy);
        if ta < 0.0 {
          r = -r;
        }
        let alpha = if r == 0.0 { 0.0 } else { atan2d(x / r, dy / r) };
        let phi = alpha / c;
        let theta = match self.code {
          Code::Cop => ta + atand(1.0 / tand(ta) - r / (R0 * self.w[3])),
          Code::Coe => {
            let st = (1.0 + self.w[4] - (r * self.w[3] / (2.0 * R0)).powi(2)) / self.w[3];
            if st.abs() > 1.0 + TOL {
              return None;
            }
            asind(st)
          }
          Code::Cod => ta + y0 - r,
          _ => {
            let t = if r == 0.0 {
              0.0
            } else {
              let q = r / self.w[3];
              if q < 0.0 {
                return None;
              }
              q.powf(1.0 / c)
            };
            90.0 - 2.0 * atand(t)
          }
        };
        Some((phi, theta))
      }
      Code::Bon => {
        let (t1, y0) = (self.w[0], self.w[1]);
        let dy = y0 - y;
        let mut r = x.hypot(dy);
        if t1 < 0.0 {
          r = -r;
        }
        let alpha = if r == 0.0 { 0.0 } else { atan2d(x / r, dy / r) };
        let theta = y0 - r;
        let ct = cosd(theta);
        if ct == 0.0 {
          if alpha * r != 0.0 {
            return None;
          }
          return Some((0.0, theta));
        }
        Some((alpha * r / (R0 * ct), theta))
      }
      Code::Pco => {
        if y.abs() < 1e-12 {
          return Some((x, 0.0));
        }
        let sgn = 1.0f64.copysign(y);
        let f = |t: f64| {
          let th = sgn * t;
          let cot = R0 * cosd(th) / sind(th);
          x * x + (y - th).powi(2) - 2.0 * cot * (y - th)
        };
        let (mut a, mut b) = (1e-14, y.abs().min(90.0));
        if f(b) < 0.0 {
          return None;
        }
        for _ in 0..200 {
          let mid = 0.5 * (a + b);
          if f(mid) < 0.0 {
            a = mid;
          } else {
            b = mid;
          }
          if b - a < 1e-15 {
            break;
          }
        }
        let theta = sgn * 0.5 * (a + b);
        let st = sind(theta);
        let cot = R0 * cosd(theta) / st;
        // x = cot sin A, y − θ = cot (1 − cos A): divide by cot (negative in the south)
        let aa = atan2d(x / cot, 1.0 - (y - theta) / cot);
        Some((aa / st, theta))
      }
      Code::Tsc | Code::Csc | Code::Qsc => {
        let (mut xf, mut yf) = (x / 45.0, y / 45.0);
        if xf > 7.0 + TOL || xf < -1.0 - TOL {
          return None;
        }
        let face = if xf > 5.0 {
          xf -= 6.0;
          4
        } else if xf > 3.0 {
          xf -= 4.0;
          3
        } else if xf > 1.0 {
          xf -= 2.0;
          2
        } else if yf > 1.0 {
          yf -= 2.0;
          0
        } else if yf < -1.0 {
          yf += 2.0;
          5
        } else {
          1
        };
        if xf.abs() > 1.0 + TOL || yf.abs() > 1.0 + TOL {
          return None;
        }
        let (xi, eta, zeta) = match self.code {
          Code::Tsc => (xf, yf, 1.0),
          Code::Csc => {
            let (xi, eta) = csc_inv(xf, yf);
            (xi, eta, 1.0)
          }
          _ => {
            let (xx, yy) = (xf * xf, yf * yf);
            if xx == 0.0 && yy == 0.0 {
              (0.0, 0.0, 1.0)
            } else {
              let (xi, eta, zeta);
              if xx >= yy {
                let (sw, cw) = (sind(15.0 * yf / xf), cosd(15.0 * yf / xf));
                let w = sw / (cw - core::f64::consts::FRAC_1_SQRT_2);
                zeta = 1.0 - xx * (1.0 - 1.0 / (2.0 + w * w).sqrt());
                xi = ((1.0 - zeta * zeta) / (1.0 + w * w)).sqrt().copysign(xf);
                eta = w * xi;
              } else {
                let (sw, cw) = (sind(15.0 * xf / yf), cosd(15.0 * xf / yf));
                let w = sw / (cw - core::f64::consts::FRAC_1_SQRT_2);
                zeta = 1.0 - yy * (1.0 - 1.0 / (2.0 + w * w).sqrt());
                eta = ((1.0 - zeta * zeta) / (1.0 + w * w)).sqrt().copysign(yf);
                xi = w * eta;
              }
              if zeta < -1.0 - TOL {
                return None;
              }
              (xi, eta, zeta)
            }
          }
        };
        let (l, m, n) = match face {
          0 => (-eta, xi, zeta),
          1 => (zeta, xi, eta),
          2 => (-xi, zeta, eta),
          3 => (-zeta, -xi, eta),
          4 => (xi, -zeta, eta),
          _ => (eta, xi, -zeta),
        };
        let phi = if l == 0.0 && m == 0.0 { 0.0 } else { atan2d(m, l) };
        Some((phi, atan2d(n, l.hypot(m))))
      }
      Code::Hpx => {
        let (h, k) = (self.w[0], self.w[1]);
        if y.abs() <= 90.0 * (k - 1.0) / h + TOL {
          let st = y * h / (90.0 * k);
          if st.abs() > 1.0 + TOL {
            return None;
          }
          return Some((x, asind(st)));
        }
        let mut sig = (k + 1.0) / 2.0 - y.abs() * h / 180.0;
        if sig < -TOL {
          return None;
        }
        sig = sig.max(0.0);
        let mut xx = x;
        if xx >= 180.0 {
          xx -= 360.0;
        }
        let kk = ((xx + 180.0) * h / 360.0).floor().clamp(0.0, h - 1.0);
        let xc = -180.0 + (2.0 * kk + 1.0) * 180.0 / h;
        let phi = if sig == 0.0 {
          if (x - xc).abs() > TOL {
            return None;
          }
          xc
        } else {
          if (x - xc).abs() > sig * 180.0 / h + TOL {
            return None;
          }
          xc + (x - xc) / sig
        };
        let st = 1.0 - sig * sig / k;
        if st.abs() > 1.0 + TOL {
          return None;
        }
        Some((phi, asind(st).copysign(y)))
      }
      Code::Xph => {
        let sx = if x >= 0.0 { 1.0 } else { -1.0 };
        let xc = sx * (45.0 + if y > 0.0 { 90.0 } else { 0.0 });
        let (sr, cr) = (sind(xc), cosd(xc));
        let (rx, ry) = (cr * x + sr * y, -sr * x + cr * y);
        let (hx, hy) = (rx + xc, ry + 90.0);
        if hy.abs() > 90.0 + TOL {
          return None;
        }
        if hy.abs() <= 45.0 + TOL {
          if rx.abs() > 45.0 + TOL {
            return None;
          }
          return Some((hx, asind(hy / 67.5)));
        }
        let mut sig = 2.0 - hy.abs() / 45.0;
        if sig < -TOL {
          return None;
        }
        sig = sig.max(0.0);
        let phi = if sig == 0.0 {
          if rx.abs() > TOL {
            return None;
          }
          xc
        } else {
          if rx.abs() > 45.0 * sig + TOL {
            return None;
          }
          xc + rx / sig
        };
        let st = 1.0 - sig * sig / 3.0;
        Some((phi, asind(st).copysign(hy)))
      }
    }
  }
}

// ─── CSC: wcslib's single-precision polynomials ──────────────────────────────

fn csc_fwd(chi: f64, psi: f64) -> (f64, f64) {
  let (chi, psi) = (chi as f32, psi as f32);
  let gstar: f32 = 1.37484847732;
  let mm: f32 = 0.004869491981;
  let gamma: f32 = -0.13161671474;
  let omega1: f32 = -0.159596235474;
  let d0: f32 = 0.0759196200467;
  let d1: f32 = -0.0217762490699;
  let c00: f32 = 0.141189631152;
  let c10: f32 = 0.0809701286525;
  let c01: f32 = -0.281528535557;
  let c11: f32 = 0.15384112876;
  let c20: f32 = -0.178251207466;
  let c02: f32 = 0.106959469314;
  let chi2 = chi * chi;
  let psi2 = psi * psi;
  let chi2co = 1.0 - chi2;
  let psi2co = 1.0 - psi2;
  let chi4 = chi2 * chi2;
  let psi4 = psi2 * psi2;
  let chi2psi2 = chi2 * psi2;
  let xf = chi
    * (chi2
      + chi2co
        * (gstar
          + psi2
            * (gamma * chi2co
              + mm * chi2
              + psi2co * (c00 + c11 * chi2psi2 + c20 * chi4 + c02 * psi4 + c10 * chi2 + c01 * psi2))
          + chi2 * (omega1 - chi2co * (d0 + d1 * chi2))));
  let yf = psi
    * (psi2
      + psi2co
        * (gstar
          + chi2
            * (gamma * psi2co
              + mm * psi2
              + chi2co * (c00 + c11 * chi2psi2 + c20 * psi4 + c02 * chi4 + c10 * psi2 + c01 * chi2))
          + psi2 * (omega1 - psi2co * (d0 + d1 * psi2))));
  (xf as f64, yf as f64)
}

const CSC_P: [&[f32]; 7] = [
  &[-0.27292696, -0.07629969, -0.22797056, 0.54852384, -0.62930065, 0.25795794, 0.02584375],
  &[-0.02819452, -0.01471565, 0.48051509, -1.74114454, 1.71547508, -0.53022337],
  &[0.27058160, -0.56800938, 0.30803317, 0.98938102, -0.83180469],
  &[-0.60441560, 1.50880086, -0.93678576, 0.08693841],
  &[0.93412077, -1.41601920, 0.33887446],
  &[-0.63915306, 0.52032238],
  &[0.14381585],
];

fn csc_inv_poly(a: f32, b: f32) -> f32 {
  let (aa, bb) = (a * a, b * b);
  let mut zs = [0.0f32; 7];
  for (i, row) in CSC_P.iter().enumerate() {
    let mut z = row[row.len() - 1];
    for &cf in row[..row.len() - 1].iter().rev() {
      z = cf + aa * z;
    }
    zs[i] = z;
  }
  let mut v = zs[6];
  for z in zs[..6].iter().rev() {
    v = z + bb * v;
  }
  a + a * (1.0 - aa) * v
}

fn csc_inv(xf: f64, yf: f64) -> (f64, f64) {
  let (xf, yf) = (xf as f32, yf as f32);
  (csc_inv_poly(xf, yf) as f64, csc_inv_poly(yf, xf) as f64)
}
