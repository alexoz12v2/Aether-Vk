//! Native spherical `(φ, θ)` ↔ celestial `(α, δ)` rotation of FITS WCS Paper II section 2.3
//! (wcslib `celset` / `cel.c`): the celestial coordinates of the native pole `(αp, δp)` from
//! `CRVAL`, `(φ0, θ0)`, `LONPOLE` (`φp`) and `LATPOLE`, then the rotation matrix
//! `R = Rz(αp) · Ry(90° − δp) · Rz(180° − φp)` taking native unit vectors to celestial ones.

use super::projection::{acosd, atan2d, cosd, sind};
use alloc::string::String;

type V3 = [f64; 3];
const TOL: f64 = 1e-10;

/// The native → celestial rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
  /// celestial longitude / latitude of the native pole (degrees)
  pub lngp: f64,
  pub latp: f64,
  /// native longitude of the celestial pole (`LONPOLE`, degrees)
  pub phip: f64,
  /// columns: celestial vectors of the native x, y, z axes
  m: [[f64; 3]; 3],
}

/// Unit vector of spherical `(lon, lat)` degrees.
pub fn dir_of(lon: f64, lat: f64) -> V3 {
  let (sl, cl) = (sind(lon), cosd(lon));
  let (sb, cb) = (sind(lat), cosd(lat));
  [cb * cl, cb * sl, sb]
}

/// `(lon in [0, 360), lat)` degrees of a vector.
pub fn lon_lat_of(v: V3) -> (f64, f64) {
  let mut lon = atan2d(v[1], v[0]);
  if lon < 0.0 {
    lon += 360.0;
  }
  if lon >= 360.0 {
    lon -= 360.0;
  }
  (lon, atan2d(v[2], v[0].hypot(v[1])))
}

/// `(φ in (−180, 180], θ)` degrees of a native vector.
pub fn phi_theta_of(v: V3) -> (f64, f64) {
  (atan2d(v[1], v[0]), atan2d(v[2], v[0].hypot(v[1])))
}

impl Rotation {
  /// wcslib `celset`: `lonpole` / `latpole` `None` = keyword absent (defaults: `φp = φ0`, +180°
  /// when `δ0 < θ0`; `LATPOLE = 90°` picks the solution nearer to it).
  pub fn celset(
    lng0: f64,
    lat0: f64,
    phi0: f64,
    theta0: f64,
    lonpole: Option<f64>,
    latpole: Option<f64>,
  ) -> Result<Rotation, String> {
    let phip = lonpole.unwrap_or(phi0 + if lat0 < theta0 { 180.0 } else { 0.0 });
    let latpole = latpole.unwrap_or(90.0);
    let (lngp, latp);
    if theta0 == 90.0 {
      lngp = lng0;
      latp = lat0;
    } else {
      let (slat0, clat0) = (sind(lat0), cosd(lat0));
      let (sthe0, cthe0) = (sind(theta0), cosd(theta0));
      let (sphip, cphip) = (sind(phip - phi0), cosd(phip - phi0));
      let xx = cthe0 * cphip;
      let yy = sthe0;
      let z = xx.hypot(yy);
      if z == 0.0 {
        if slat0 != 0.0 {
          return Err("LONPOLE / CRVAL inconsistent (celset)".into());
        }
        latp = latpole;
      } else {
        if (slat0 / z).abs() > 1.0 + TOL {
          return Err("LONPOLE / CRVAL inconsistent (celset)".into());
        }
        let u = atan2d(yy, xx);
        let v = acosd(slat0 / z);
        let mut latp1 = u + v;
        if latp1 > 180.0 {
          latp1 -= 360.0;
        }
        let mut latp2 = u - v;
        if latp2 < -180.0 {
          latp2 += 360.0;
        }
        let (ok1, ok2) = (latp1.abs() < 90.0 + TOL, latp2.abs() < 90.0 + TOL);
        let lp = match (ok1, ok2) {
          (true, true) => {
            if (latp1 - latpole).abs() < (latp2 - latpole).abs() {
              latp1
            } else {
              latp2
            }
          }
          (true, false) => latp1,
          (false, true) => latp2,
          (false, false) => return Err("no valid LATPOLE (celset)".into()),
        };
        latp = lp.clamp(-90.0, 90.0);
      }
      let z = cosd(latp) * clat0;
      if z.abs() < TOL {
        if clat0.abs() < TOL {
          lngp = lng0;
        } else if latp > 0.0 {
          lngp = lng0 + phip - phi0 - 180.0;
        } else {
          lngp = lng0 - phip + phi0;
        }
      } else {
        let xx = (sthe0 - sind(latp) * slat0) / z;
        let yy = sphip * cthe0 / clat0;
        if xx == 0.0 && yy == 0.0 {
          return Err("LONPOLE / CRVAL inconsistent (celset)".into());
        }
        lngp = lng0 - atan2d(yy, xx);
      }
    }
    let lngp = lngp.rem_euclid(360.0);
    Ok(Rotation::from_pole(lngp, latp, phip))
  }

  /// The rotation of a known native pole `(αp, δp)` and `φp`.
  pub fn from_pole(lngp: f64, latp: f64, phip: f64) -> Rotation {
    let (sa, ca) = (sind(lngp), cosd(lngp));
    let (sd, cd) = (sind(latp), cosd(latp));
    let (sp, cp) = (sind(phip), cosd(phip));
    Rotation {
      lngp,
      latp,
      phip,
      m: [
        [-cp * ca * sd - sp * sa, -sp * ca * sd + cp * sa, ca * cd],
        [-cp * sa * sd + sp * ca, -sp * sa * sd - cp * ca, sa * cd],
        [cp * cd, sp * cd, sd],
      ],
    }
  }

  /// Native vector → celestial vector.
  pub fn to_celestial(&self, n: V3) -> V3 {
    let m = &self.m;
    [
      m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2],
      m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2],
      m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2],
    ]
  }

  /// Celestial vector → native vector.
  pub fn to_native(&self, c: V3) -> V3 {
    let m = &self.m;
    [
      m[0][0] * c[0] + m[1][0] * c[1] + m[2][0] * c[2],
      m[0][1] * c[0] + m[1][1] * c[1] + m[2][1] * c[2],
      m[0][2] * c[0] + m[1][2] * c[1] + m[2][2] * c[2],
    ]
  }

  /// The matrix (columns = celestial vectors of the native axes).
  pub fn matrix(&self) -> [[f64; 3]; 3] {
    self.m
  }
}
