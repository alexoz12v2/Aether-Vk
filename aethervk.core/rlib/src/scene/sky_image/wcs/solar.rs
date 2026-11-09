//! Helioprojective (`HPLN` / `HPLT`, Thompson 2006 §4) → ICRS from the header alone, the sunpy
//! `Helioprojective → HCRS` chain reduced to its rotations (a direction needs no distance):
//!
//! 1. helioprojective `(θx, θy)` as a spherical pair → the heliocentric-cartesian (HCC) axes:
//!    `x` solar west, `y` solar north, `z` towards the observer, so a direction seen by the
//!    observer is `(cos θy sin θx, sin θy, −cos θy cos θx)`;
//! 2. HCC → Stonyhurst heliographic (HGS): the observer's heliographic longitude `L0` and
//!    latitude `B0` (`HGLN_OBS`, `HGLT_OBS`; without them the Earth, `L0 = 0`, `B0` of the date);
//! 3. HGS → ICRS: `z` the solar rotation axis (IAU 2015 / sunpy: RA 286.13°, Dec 63.87°), `x` the
//!    Sun → Earth direction at the observation time (`DATE-AVG`, else `DATE-OBS`, `DATE-BEG`,
//!    `MJD-AVG`, `MJD-OBS`) made perpendicular to `z`, `y = z × x`. The Sun → Earth vector is the
//!    truncated ERFA `epv00` series ([`earth_ephemeris`](super::earth_ephemeris), 0.07″); UTC → TT
//!    uses the leap-second table below (sunpy uses the same `epv00` through astropy's built-in
//!    ephemeris, so the chain agrees with it to ~0.005″, `tests::helioprojective_frame`).
//!
//! The other solar systems of Thompson 2006 (`HGLN/HGLT`, `CRLN/CRLT`, `HRLN/HRLT`, `SOLX/SOLY`)
//! locate points on or around the Sun, not sky directions, and are refused by [`super::Wcs`].

use super::earth_ephemeris as eph;
use super::frame::{M3, Frame, FrameKind, apply, mul, transpose};
use super::super::FitsHeader;
use alloc::{format, string::String};

type V3 = [f64; 3];
const DEG: f64 = core::f64::consts::PI / 180.0;

/// `(year, month, day, TAI − UTC seconds)` from that date on (IERS Bulletin C, 1972 onwards).
const LEAP_SECONDS: &[(i32, u32, u32, f64)] = &[
  (1972, 1, 1, 10.0),
  (1972, 7, 1, 11.0),
  (1973, 1, 1, 12.0),
  (1974, 1, 1, 13.0),
  (1975, 1, 1, 14.0),
  (1976, 1, 1, 15.0),
  (1977, 1, 1, 16.0),
  (1978, 1, 1, 17.0),
  (1979, 1, 1, 18.0),
  (1980, 1, 1, 19.0),
  (1981, 7, 1, 20.0),
  (1982, 7, 1, 21.0),
  (1983, 7, 1, 22.0),
  (1985, 7, 1, 23.0),
  (1988, 1, 1, 24.0),
  (1990, 1, 1, 25.0),
  (1991, 1, 1, 26.0),
  (1992, 7, 1, 27.0),
  (1993, 7, 1, 28.0),
  (1994, 7, 1, 29.0),
  (1996, 1, 1, 30.0),
  (1997, 7, 1, 31.0),
  (1999, 1, 1, 32.0),
  (2006, 1, 1, 33.0),
  (2009, 1, 1, 34.0),
  (2012, 7, 1, 35.0),
  (2015, 7, 1, 36.0),
  (2017, 1, 1, 37.0),
];

/// Julian date at 0h of a proleptic Gregorian calendar date (Fliegel & Van Flandern 1968).
pub fn jd_of_date(y: i32, m: u32, d: u32) -> f64 {
  let (y, m, d) = (y as i64, m as i64, d as i64);
  let a = (14 - m) / 12;
  let yy = y + 4800 - a;
  let mm = m + 12 * a - 3;
  let jdn = d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045;
  jdn as f64 - 0.5
}

/// `TAI − UTC` (seconds) at a UTC Julian date: the table value in force (10 s before 1972).
pub fn tai_minus_utc(jd_utc: f64) -> f64 {
  let mut v = 10.0;
  for &(y, m, d, s) in LEAP_SECONDS {
    if jd_utc >= jd_of_date(y, m, d) {
      v = s;
    }
  }
  v
}

/// UTC → TT Julian date (`TT = TAI + 32.184 s`).
pub fn jd_tt_of_utc(jd_utc: f64) -> f64 {
  jd_utc + (tai_minus_utc(jd_utc) + 32.184) / 86400.0
}

/// A FITS date string → UTC Julian date: `YYYY-MM-DD[Thh:mm:ss[.s…]]` (Paper I), the old
/// `DD/MM/YY` (1900 + YY), or a bare year-month-day.
pub fn parse_fits_date(s: &str) -> Option<f64> {
  let s = s.trim();
  if s.is_empty() {
    return None;
  }
  let (date, time) = match s.split_once('T') {
    Some((d, t)) => (d, Some(t)),
    None => match s.split_once(' ') {
      Some((d, t)) if d.contains('-') => (d, Some(t)),
      _ => (s, None),
    },
  };
  let (y, m, d) = if let Some((dd, rest)) = date.split_once('/') {
    let (mm, yy) = rest.split_once('/')?;
    let yy = yy.trim().parse::<i32>().ok()?;
    (if yy < 100 { 1900 + yy } else { yy }, mm.trim().parse().ok()?, dd.trim().parse().ok()?)
  } else {
    let mut it = date.split('-');
    let y = it.next()?.trim().parse::<i32>().ok()?;
    let m = it.next()?.trim().parse::<u32>().ok()?;
    let d = it.next()?.trim().parse::<u32>().ok()?;
    (y, m, d)
  };
  if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
    return None;
  }
  let mut jd = jd_of_date(y, m, d);
  if let Some(t) = time {
    let mut parts = t.trim().split(':');
    let hh = parts.next()?.trim().parse::<f64>().ok()?;
    let mi = parts.next().map(|v| v.trim().parse::<f64>().ok()).unwrap_or(Some(0.0))?;
    let ss = parts.next().map(|v| v.trim().trim_end_matches('Z').parse::<f64>().ok()).unwrap_or(Some(0.0))?;
    jd += (hh * 3600.0 + mi * 60.0 + ss) / 86400.0;
  }
  Some(jd)
}

/// The observation time (UTC Julian date) of a header: `DATE-AVG`, `DATE-OBS` (`DATE_OBS`),
/// `DATE-BEG`, then `MJD-AVG`, `MJD-OBS`.
pub fn obstime_jd_utc(h: &FitsHeader) -> Option<f64> {
  for key in ["DATE-AVG", "DATE-OBS", "DATE_OBS", "DATE-BEG"] {
    if let Some(v) = h.get(key)
      && let Some(jd) = parse_fits_date(v)
    {
      return Some(jd);
    }
  }
  for key in ["MJD-AVG", "MJD-OBS"] {
    if let Some(v) = h.get_f64(key) {
      return Some(v + 2400000.5);
    }
  }
  None
}

fn series(terms: &[[f64; 3]], t: f64) -> f64 {
  let mut s = 0.0;
  for term in terms {
    s += term[0] * (term[1] + term[2] * t).cos();
  }
  s
}

/// Sun → Earth vector (au, ICRS axes) at a TT Julian date: the truncated `epv00` series.
pub fn sun_to_earth_au(jd_tt: f64) -> V3 {
  let t = (jd_tt - 2451545.0) / 365.25;
  let x = series(eph::E0X, t) + t * series(eph::E1X, t) + t * t * series(eph::E2X, t);
  let y = series(eph::E0Y, t) + t * series(eph::E1Y, t) + t * t * series(eph::E2Y, t);
  let z = series(eph::E0Z, t) + t * series(eph::E1Z, t) + t * t * series(eph::E2Z, t);
  apply(&eph::AM, [x, y, z])
}

fn unit(v: V3) -> V3 {
  let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
  [v[0] / l, v[1] / l, v[2] / l]
}

/// The solar rotation axis in ICRS (RA 286.13°, Dec 63.87°).
pub fn solar_pole() -> V3 {
  let (ra, dec) = (286.13 * DEG, 63.87 * DEG);
  [(dec).cos() * (ra).cos(), (dec).cos() * (ra).sin(), (dec).sin()]
}

/// Stonyhurst heliographic → ICRS rotation at a TT Julian date.
pub fn hgs_to_icrs(jd_tt: f64) -> M3 {
  let z = solar_pole();
  let e = sun_to_earth_au(jd_tt);
  let ez = e[0] * z[0] + e[1] * z[1] + e[2] * z[2];
  let x = unit([e[0] - ez * z[0], e[1] - ez * z[1], e[2] - ez * z[2]]);
  let y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  // columns are the HGS axes in ICRS
  [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
}

/// The Earth's heliographic latitude `B0` (degrees) at a TT Julian date.
pub fn b0_deg(jd_tt: f64) -> f64 {
  let e = unit(sun_to_earth_au(jd_tt));
  let z = solar_pole();
  (e[0] * z[0] + e[1] * z[1] + e[2] * z[2]).asin() / DEG
}

/// Heliocentric cartesian (observer at heliographic `L0`, `B0` degrees) → Stonyhurst
/// heliographic rotation (sunpy `hcc_to_hgs`).
pub fn hcc_to_hgs(l0_deg: f64, b0_deg: f64) -> M3 {
  let (sb, cb) = ((b0_deg * DEG).sin(), (b0_deg * DEG).cos());
  let (sl, cl) = ((l0_deg * DEG).sin(), (l0_deg * DEG).cos());
  let tilt = [[0.0, -sb, cb], [1.0, 0.0, 0.0], [0.0, cb, sb]];
  let rz = [[cl, -sl, 0.0], [sl, cl, 0.0], [0.0, 0.0, 1.0]];
  mul(&rz, &tilt)
}

/// Helioprojective `(θx, θy)` as a spherical direction `(cos θy cos θx, cos θy sin θx, sin θy)` →
/// the HCC direction `(cos θy sin θx, sin θy, −cos θy cos θx)`.
const HPC_TO_HCC: M3 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]];

/// The observer of a header: `(L0, B0)` degrees and whether the Earth was assumed.
pub fn observer(h: &FitsHeader, jd_tt: f64) -> (f64, f64, bool) {
  match (h.get_f64("HGLN_OBS"), h.get_f64("HGLT_OBS")) {
    (Some(l), Some(b)) => (l, b, false),
    (None, Some(b)) => (0.0, b, false),
    _ => (0.0, b0_deg(jd_tt), true),
  }
}

/// The helioprojective frame of a header: `Err` without an observation date.
pub fn frame_from_header(h: &FitsHeader) -> Result<(Frame, bool), String> {
  let Some(jd_utc) = obstime_jd_utc(h) else {
    return Err("helioprojective coordinates (HPLN / HPLT) need the observation time: no DATE-AVG, DATE-OBS, DATE-BEG, MJD-AVG or MJD-OBS card".into());
  };
  let jd_tt = jd_tt_of_utc(jd_utc);
  let (l0, b0, assumed) = observer(h, jd_tt);
  let m = mul(&mul(&hgs_to_icrs(jd_tt), &hcc_to_hgs(l0, b0)), &HPC_TO_HCC);
  Ok((Frame::from_matrix(FrameKind::Helioprojective, m), assumed))
}

/// ICRS → HCC rotation of a frame built by [`frame_from_header`] (for tests).
pub fn icrs_to_hcc(l0_deg: f64, b0_deg: f64, jd_tt: f64) -> M3 {
  transpose(&mul(&hgs_to_icrs(jd_tt), &hcc_to_hgs(l0_deg, b0_deg)))
}

/// `(JD UTC, L0, B0, Earth assumed)` as a message for diagnostics.
pub fn describe(h: &FitsHeader) -> String {
  match obstime_jd_utc(h) {
    None => "no observation time".into(),
    Some(jd) => {
      let tt = jd_tt_of_utc(jd);
      let (l0, b0, assumed) = observer(h, tt);
      format!("JD {jd:.6} UTC, observer L0 {l0:.4}° B0 {b0:.4}°{}", if assumed { " (Earth assumed)" } else { "" })
    }
  }
}
