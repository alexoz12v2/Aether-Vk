//! Celestial reference frames of a FITS WCS → ICRS, numerically the astropy 8 / ERFA
//! transformations the reference tables were made with:
//! - FK5: the frame bias (`η0 = −19.9`, `ξ0 = 9.1`, `dα0 = −22.9` mas), with the IAU 2006
//!   (Capitaine) precession for an equinox other than J2000;
//! - FK4 / FK4-NO-E: Murray 1989 B1950 → J2000 matrix, Newcomb precession for another Besselian
//!   equinox, and for FK4 the removal of the elliptic aberration E-terms at the equinox (astropy's
//!   `obstime = equinox`);
//! - galactic: the IAU 1958 / Hipparcos pole (`192.85948°, 27.12825°`, `l = 122.93192°` of the
//!   NCP) via FK5 J2000;
//! - ecliptic: astropy `BarycentricMeanEcliptic` of the equinox: IAU 2006 Fukushima–Williams
//!   bias-precession and mean obliquity (`ε0 = 84381.406″`), which differs from the engine's
//!   `ECLIPJ2000` (IAU 1976, 84381.448″) by 0.04″;
//! - helioprojective: the observer-centred rotation built in [`super::solar`].

use super::projection::{cosd, sind};
use super::rotation::dir_of;

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

/// Which frame the header's longitude / latitude are measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
  Icrs,
  Fk5,
  Fk4,
  Fk4NoE,
  Galactic,
  Ecliptic,
  /// helioprojective (`HPLN` / `HPLT`): directions seen from the observer of the header at its
  /// observation time ([`super::solar`])
  Helioprojective,
}

impl FrameKind {
  pub fn name(self) -> &'static str {
    match self {
      FrameKind::Icrs => "ICRS",
      FrameKind::Fk5 => "FK5",
      FrameKind::Fk4 => "FK4",
      FrameKind::Fk4NoE => "FK4-NO-E",
      FrameKind::Galactic => "galactic",
      FrameKind::Ecliptic => "ecliptic",
      FrameKind::Helioprojective => "helioprojective",
    }
  }
}

/// A frame and its transformation to ICRS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
  pub kind: FrameKind,
  /// the equinox the frame is referred to (Julian years, Besselian for FK4), `None` when the
  /// frame has none (ICRS, galactic)
  pub equinox: Option<f64>,
  /// rotation frame → ICRS (applied after the E-term removal)
  m: M3,
  /// FK4 elliptic aberration E-terms to remove
  eterms: Option<V3>,
}

pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mul(a: &M3, b: &M3) -> M3 {
  let mut r = [[0.0; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  r
}

pub fn transpose(a: &M3) -> M3 {
  let mut r = [[0.0; 3]; 3];
  for i in 0..3 {
    for j in 0..3 {
      r[i][j] = a[j][i];
    }
  }
  r
}

pub fn apply(m: &M3, v: V3) -> V3 {
  [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ]
}

/// Vector rotations (right-handed, degrees) about the x, y, z axes.
pub fn rx(a: f64) -> M3 {
  let (s, c) = (sind(a), cosd(a));
  [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}
pub fn ry(a: f64) -> M3 {
  let (s, c) = (sind(a), cosd(a));
  [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}
pub fn rz(a: f64) -> M3 {
  let (s, c) = (sind(a), cosd(a));
  [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

const MAS: f64 = 1.0 / 3.6e6;
const AS: f64 = 1.0 / 3600.0;

/// FK5 J2000 → ICRS (the inverse of the ICRS → FK5 frame bias).
pub fn fk5_j2000_to_icrs() -> M3 {
  let icrs_to_fk5 = mul(&mul(&rx(-19.9 * MAS), &ry(-9.1 * MAS)), &rz(22.9 * MAS));
  transpose(&icrs_to_fk5)
}

/// Galactic → FK5 J2000.
pub fn galactic_to_fk5() -> M3 {
  mul(
    &mul(&rz(192.8594812065348), &ry(90.0 - 27.12825118085622)),
    &rz(180.0 - 122.9319185680026),
  )
}

/// Galactic → ICRS.
pub fn galactic_to_icrs() -> M3 {
  mul(&fk5_j2000_to_icrs(), &galactic_to_fk5())
}

/// FK4 B1950 (no E-terms) → FK5 J2000 (Murray 1989).
const MURRAY: M3 = [
  [0.9999256794956877, -0.0111814832204662, -0.0048590038153592],
  [0.0111814832391717, 0.9999374848933135, -0.0000271625947142],
  [0.0048590037723143, -0.0000271702937440, 0.9999881946023742],
];

/// Precession J2000 → `jyear` (IAU 2006, Capitaine et al. 2003 polynomials).
pub fn precess_from_j2000(jyear: f64) -> M3 {
  let t = (jyear - 2000.0) / 100.0;
  let poly = |c: &[f64]| c.iter().fold(0.0, |acc, &k| acc * t + k);
  let zeta = poly(&[-0.0000003173, -0.000005971, 0.01801828, 0.2988499, 2306.083227, 2.650545]) * AS;
  let z = poly(&[-0.0000002904, -0.000028596, 0.01826837, 1.0927348, 2306.077181, -2.650545]) * AS;
  let th = poly(&[-0.0000001274, -0.000007089, -0.04182264, -0.4294934, 2004.191903, 0.0]) * AS;
  mul(&mul(&rz(z), &ry(-th)), &rz(zeta))
}

/// Newcomb precession between Besselian epochs (astropy `_precession_matrix_besselian`).
pub fn precess_besselian(epoch1: f64, epoch2: f64) -> M3 {
  let t1 = (epoch1 - 1850.0) / 1000.0;
  let dt = (epoch2 - 1850.0) / 1000.0 - t1;
  let zeta = (17.995 * dt * dt + (30.240 - 0.27 * t1) * dt + (23035.545 + t1 * 139.720 + 0.060 * t1 * t1)) * dt * AS;
  let z = (18.325 * dt * dt + (109.480 + 0.39 * t1) * dt + (23035.545 + t1 * 139.720 + 0.060 * t1 * t1)) * dt * AS;
  let th = (-41.80 * dt * dt + (-42.65 - 0.37 * t1) * dt + (20051.12 - 85.29 * t1 - 0.37 * t1 * t1)) * dt * AS;
  mul(&mul(&rz(z), &ry(-th)), &rz(zeta))
}

/// FK4 E-terms of aberration at a Besselian epoch (astropy `fk4_e_terms`).
pub fn eterms(bepoch: f64) -> V3 {
  let jd = 2415020.31352 + (bepoch - 1900.0) * 365.242198781;
  let t = (jd - 2451545.0) / 36525.0;
  let e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t;
  let g = 282.93735 + 1.71946 * t + 0.00046 * t * t;
  let o = 23.439291 - 0.0130042 * t - 1.64e-7 * t * t + 5.04e-7 * t * t * t;
  let k = 0.0056932 * super::DEG;
  [e * k * sind(g), -e * k * cosd(g) * cosd(o), -e * k * cosd(g) * sind(o)]
}

/// IAU 2006 Fukushima–Williams angles `(γ̄, φ̄, ψ̄, ε_A)` at `t` Julian centuries from J2000.
fn fw_angles(t: f64) -> (f64, f64, f64, f64) {
  let poly = |c: &[f64]| c.iter().rev().fold(0.0, |acc, &k| acc * t + k);
  let gamb = poly(&[-0.052928, 10.556378, 0.4932044, -0.00031238, -0.000002788, 0.0000000260]) * AS;
  let phib = poly(&[84381.412819, -46.811016, 0.0511268, 0.00053289, -0.000000440, -0.0000000176]) * AS;
  let psib = poly(&[-0.041775, 5038.481484, 1.5584175, -0.00018522, -0.000026452, -0.0000000148]) * AS;
  let epsa = poly(&[84381.406, -46.836769, -0.0001831, 0.00200340, -0.000000576, -0.0000000434]) * AS;
  (gamb, phib, psib, epsa)
}

/// Mean ecliptic of the Julian equinox `jyear` → ICRS (astropy `BarycentricMeanEcliptic`).
pub fn ecliptic_to_icrs(jyear: f64) -> M3 {
  let t = (jyear - 2000.0) / 100.0;
  let (g, p, s, e) = fw_angles(t);
  // bias-precession: ICRS → mean equator of date
  let rbp = mul(&mul(&mul(&rx(e), &rz(s)), &rx(-p)), &rz(-g));
  mul(&transpose(&rbp), &rx(e))
}

impl Frame {
  pub fn icrs() -> Frame {
    Frame { kind: FrameKind::Icrs, equinox: None, m: IDENTITY, eterms: None }
  }

  pub fn fk5(equinox: Option<f64>) -> Frame {
    let e = equinox.unwrap_or(2000.0);
    let m = if (e - 2000.0).abs() < 1e-9 {
      fk5_j2000_to_icrs()
    } else {
      mul(&fk5_j2000_to_icrs(), &transpose(&precess_from_j2000(e)))
    };
    Frame { kind: FrameKind::Fk5, equinox: Some(e), m, eterms: None }
  }

  pub fn fk4(equinox: Option<f64>, no_e: bool) -> Frame {
    let e = equinox.unwrap_or(1950.0);
    let mut m = mul(&fk5_j2000_to_icrs(), &MURRAY);
    if (e - 1950.0).abs() > 1e-9 {
      m = mul(&m, &precess_besselian(e, 1950.0));
    }
    Frame {
      kind: if no_e { FrameKind::Fk4NoE } else { FrameKind::Fk4 },
      equinox: Some(e),
      m,
      eterms: if no_e { None } else { Some(eterms(e)) },
    }
  }

  pub fn galactic() -> Frame {
    Frame { kind: FrameKind::Galactic, equinox: None, m: galactic_to_icrs(), eterms: None }
  }

  pub fn ecliptic(equinox: Option<f64>) -> Frame {
    let e = equinox.unwrap_or(2000.0);
    Frame { kind: FrameKind::Ecliptic, equinox: Some(e), m: ecliptic_to_icrs(e), eterms: None }
  }

  /// A frame given by its rotation to ICRS (no equinox, no E-terms).
  pub(super) fn from_matrix(kind: FrameKind, m: M3) -> Frame {
    Frame { kind, equinox: None, m, eterms: None }
  }

  /// `true` when the transformation is the identity (ICRS).
  pub fn is_identity(&self) -> bool {
    self.m == IDENTITY && self.eterms.is_none()
  }

  /// Unit vector in this frame → ICRS unit vector.
  pub fn to_icrs(&self, v: V3) -> V3 {
    let v = match self.eterms {
      None => v,
      Some(a) => {
        let av = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
        let u = [v[0] - a[0] + av * v[0], v[1] - a[1] + av * v[1], v[2] - a[2] + av * v[2]];
        let l = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        [u[0] / l, u[1] / l, u[2] / l]
      }
    };
    if self.m == IDENTITY { v } else { apply(&self.m, v) }
  }

  /// ICRS unit vector → unit vector in this frame.
  pub fn from_icrs(&self, v: V3) -> V3 {
    let v = if self.m == IDENTITY { v } else { apply(&transpose(&self.m), v) };
    match self.eterms {
      None => v,
      Some(a) => {
        // invert u = v − A + (A·v) v by fixed point (|A| ≈ 1e-5)
        let mut w = v;
        for _ in 0..4 {
          let aw = a[0] * w[0] + a[1] * w[1] + a[2] * w[2];
          w = [v[0] + a[0] - aw * w[0], v[1] + a[1] - aw * w[1], v[2] + a[2] - aw * w[2]];
          let l = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
          w = [w[0] / l, w[1] / l, w[2] / l];
        }
        w
      }
    }
  }

  /// `(lon, lat)` degrees in this frame → ICRS `(ra, dec)` degrees.
  pub fn lonlat_to_icrs(&self, lon: f64, lat: f64) -> (f64, f64) {
    super::rotation::lon_lat_of(self.to_icrs(dir_of(lon, lat)))
  }

  /// ICRS `(ra, dec)` degrees → `(lon, lat)` degrees in this frame.
  pub fn icrs_to_lonlat(&self, ra: f64, dec: f64) -> (f64, f64) {
    super::rotation::lon_lat_of(self.from_icrs(dir_of(ra, dec)))
  }
}
