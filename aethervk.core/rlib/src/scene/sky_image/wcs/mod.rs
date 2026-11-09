//! Celestial world coordinate systems of FITS images (WCS Papers I–III, SIP, TPV, the IRAF TNX /
//! ZPX conventions, DSS plate solutions, helioprojective solar images): any pixel of any header
//! → ICRS, matching wcslib / astropy (wcstools for TNX / ZPX, AST for DSS, sunpy for
//! helioprojective).
//!
//! ## Pipeline (`pixel → sky`)
//! 1. `(u, v) = (p1, p2) − CRPIX` of the two celestial axes (found by `CTYPE`, any axis numbers,
//!    either order);
//! 2. SIP: `(u, v) += (A(u, v), B(u, v))` when `CTYPE` ends in `-SIP` / `A_ORDER` is present;
//! 3. linear part: `CDi_j`, else `CDELTi · PCi_j`, else `CDELTi` with `CROTA2` (Paper II eq. 186,
//!    the `CDELT` ratio in the off-diagonal terms); `CUNITi` scales degrees (`deg` default,
//!    `arcmin`, `arcsec`, `mas`, `rad`) — the intermediate world coordinates `(x, y)` in degrees,
//!    longitude first;
//! 4. TPV: the `PVi_m` polynomial on `(x, y)` for `CTYPE … TPV`, or `TAN` with `PV` cards other
//!    than the identity (a lone `PV2_1 = 0`, as solar `TAN` headers carry, is ignored as wcslib
//!    does);
//!    TNX / ZPX: `(x, y) += (lngcor(x, y), latcor(x, y))`, the IRAF surfaces of the `WATi_nnn`
//!    strings ([`wat`], [`distortion::IrafSurface`]), the projection then TAN / ZPN (ZPN's
//!    `PV2_m` from the `projp` attributes); DSS: `(x, y)` are plate millimetres (`CD` = the pixel
//!    size, `CRPIX` from `PPO3` / `PPO6` / `CNPIX`) and the `AMDX` / `AMDY` polynomial gives the
//!    gnomonic `(ξ, η)` of the plate centre, the `CRVAL` (`PLTRAH…`, `PLTDEC…`);
//! 5. the fiducial offset `(x0, y0)` when `PVi_0 ≠ 0` on the longitude axis (Paper II §2.5);
//! 6. the projection's inverse `(x, y) → (φ, θ)` ([`projection`]), `None` outside its domain;
//! 7. the native → celestial rotation ([`rotation`]) with `(φ0, θ0)` (the projection's, or
//!    `PVi_1`, `PVi_2` of the longitude axis), `LONPOLE` (`PVi_3`), `LATPOLE` (`PVi_4`) and their
//!    defaults: `(lon, lat)` in the header's frame;
//! 8. the frame → ICRS ([`frame`]): `RADESYS` ICRS / FK5 (`EQUINOX`, default 2000) / FK4 and
//!    FK4-NO-E (default B1950), no `RADESYS`: ICRS without `EQUINOX`, FK4 for `EQUINOX < 1984`,
//!    else FK5; `GLON/GLAT` galactic; `ELON/ELAT` mean ecliptic of `EQUINOX`; `HPLN/HPLT`
//!    helioprojective: the observer (`HGLN_OBS` / `HGLT_OBS`, else the Earth) at the observation
//!    date ([`solar`]).
//!
//! Each step is invertible; [`Wcs::radec_to_pixel`] runs them backwards (Newton for SIP / TPV /
//! TNX / ZPX / DSS).
//!
//! ## Refused
//! Solar systems that are positions on or around the Sun rather than sky directions
//! (Stonyhurst / Carrington heliographic `HGLN/HGLT`, `CRLN/CRLT`, heliocentric radial
//! `HRLN/HRLT`, heliocentric cartesian `SOLX/SOLY/SOLZ`) and planetary body-fixed pairs
//! (`xxLN/xxLT`); a helioprojective header without an observation date; a TNX `ro` other than
//! `180/π`.
//!
//! ## Not implemented
//! `-SIP` on a non-TAN projection (accepted: SIP is applied, as astropy does), the `CROTAi` of
//! an axis other than the latitude axis, `PVi_m` with `m ≥ 30` on the latitude axis, the TNX /
//! ZPX `WAT` surfaces on a projection other than their own.

pub mod distortion;
mod earth_ephemeris;
pub mod frame;
pub mod projection;
pub mod rotation;
pub mod solar;
pub mod wat;

#[cfg(test)]
mod tests;

use alloc::{
  format,
  string::{String, ToString},
  vec::Vec,
};

use super::FitsHeader;
pub use distortion::{Distortion, DssPlate, IrafCorrection, IrafCrossTerms, IrafFunction, IrafSurface, Sip, Tpv};
pub use frame::{Frame, FrameKind};
pub use projection::{Code, Projection, PvParams};
pub use rotation::Rotation;

type V3 = [f64; 3];
const DEG: f64 = core::f64::consts::PI / 180.0;

/// A pixel ↔ sky mapping in ICRS degrees (FITS 1-based pixel coordinates: pixel `(1, 1)` is the
/// centre of the first stored pixel). The one interface the registration mesh and other image
/// kinds (HiPS tiles) share.
pub trait SkyMapping {
  /// ICRS `(ra, dec)` degrees of a pixel; `None` outside the mapping's domain.
  fn pix_to_sky(&self, x: f64, y: f64) -> Option<(f64, f64)>;
  /// Pixel of an ICRS `(ra, dec)` (degrees); `None` where the sky point has no pixel.
  fn sky_to_pix(&self, ra: f64, dec: f64) -> Option<(f64, f64)>;
}

/// Header facts of a WCS for the UI / JSON.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WcsFacts {
  /// projection code (`TAN`, `SIN`, … ; `NCP` and `GLS` read as `SIN` / `SFL`)
  pub projection: &'static str,
  /// `none`, `SIP`, `TPV`, `TNX`, `ZPX`, `DSS`
  pub distortion: &'static str,
  /// `ICRS`, `FK5`, `FK4`, `FK4-NO-E`, `galactic`, `ecliptic`, `helioprojective`
  pub frame: &'static str,
  /// the equinox (Julian years, Besselian for FK4), `None` for ICRS / galactic
  pub equinox: Option<f64>,
  /// `true` when no `RADESYS` / `EQUINOX` card named the equatorial frame (ICRS assumed), or
  /// no `HGLN_OBS` / `HGLT_OBS` named the observer of a helioprojective image (Earth assumed)
  pub frame_assumed: bool,
  pub lonpole_deg: f64,
  pub latpole_deg: f64,
}

/// A celestial WCS (`Copy`: every table is a fixed-size array).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wcs {
  /// ICRS right ascension / declination (degrees) of the reference point (`CRVAL` converted to
  /// ICRS): the "lock to image" aim
  pub crval_deg: [f64; 2],
  /// `CRVAL` of the longitude / latitude axes in the header's frame (degrees)
  pub crval_frame_deg: [f64; 2],
  /// `CRPIX` of the longitude / latitude axes' pixel axes, in pixel-axis order (`p1`, `p2`)
  pub crpix: [f64; 2],
  /// linear part, degrees per pixel: row 0 → longitude intermediate `x`, row 1 → latitude `y`;
  /// columns = the two pixel axes (`p1`, `p2`)
  pub cd_deg: [[f64; 2]; 2],
  pub projection: Projection,
  pub rotation: Rotation,
  pub distortion: Distortion,
  pub frame: Frame,
  /// no `RADESYS` / `EQUINOX` card: ICRS assumed (helioprojective: no observer cards, Earth)
  pub frame_assumed: bool,
  /// fiducial offset `(x0, y0)` degrees (`PVi_0 ≠ 0` on the longitude axis)
  pub offset: [f64; 2],
  /// the two FITS axis numbers (1-based) of the longitude and latitude axes
  pub axes: [u8; 2],
}

fn unit_scale(h: &FitsHeader, axis: u8) -> Result<f64, String> {
  let Some(u) = h.get(&format!("CUNIT{axis}")) else {
    return Ok(1.0);
  };
  let u = u.trim().to_ascii_lowercase();
  Ok(match u.as_str() {
    "" | "deg" | "degree" | "degrees" => 1.0,
    "arcmin" => 1.0 / 60.0,
    "arcsec" => 1.0 / 3600.0,
    "mas" => 1.0 / 3.6e6,
    "rad" | "radian" | "radians" => 1.0 / DEG,
    _ => return Err(format!("unsupported CUNIT{axis} '{u}'")),
  })
}

/// `CTYPEi` → `(coordinate type, algorithm code, SIP suffix)`: `RA---TAN-SIP` → `("RA", "TAN",
/// true)`; a missing algorithm (`"RA"`) is linear `("RA", "", false)`.
fn split_ctype(ct: &str) -> (String, String, bool) {
  let ct = ct.trim();
  let (ct, sip) = match ct.strip_suffix("-SIP") {
    Some(r) => (r, true),
    None => (ct, false),
  };
  let mut it = ct.splitn(2, '-');
  let kind = it.next().unwrap_or("").to_string();
  let code = it.next().unwrap_or("").trim_matches('-').to_string();
  (kind, code, sip)
}

fn pv_params(h: &FitsHeader, axis: u8) -> PvParams {
  let mut pv = PvParams::default();
  let prefix = format!("PV{axis}_");
  for (k, _) in h.cards.iter() {
    if let Some(m) = k.strip_prefix(&prefix)
      && let Ok(m) = m.parse::<usize>()
      && let Some(v) = h.get_f64(k)
    {
      pv.set(m, v);
    }
  }
  pv
}

/// `true` when the `PVi_m` cards of an axis describe a TPV polynomial: any card other than the
/// identity (`PVi_1 = 1`, the rest `0`) or a bare `PVi_1 = 0` (solar `TAN` headers carry a
/// `PV2_1 = 0` that wcslib ignores for TAN; as a TPV scale it would collapse the axis).
fn has_tpv_pv(h: &FitsHeader, axis: u8) -> bool {
  let prefix = format!("PV{axis}_");
  h.cards.iter().any(|(k, _)| {
    let Some(m) = k.strip_prefix(&prefix).and_then(|m| m.parse::<usize>().ok()) else {
      return false;
    };
    let v = h.get_f64(k).unwrap_or(0.0);
    match m {
      1 => v != 1.0 && v != 0.0,
      _ => v != 0.0,
    }
  })
}

/// Why a longitude / latitude `CTYPE` pair cannot be a sky direction (`None` for a coordinate
/// type this module does not know, which is skipped as a non-celestial axis).
fn refused_coordinate_type(kind: &str) -> Option<&'static str> {
  Some(match kind {
    "HGLN" | "HGLT" => "Stonyhurst heliographic coordinates are positions on the Sun, not sky directions",
    "CRLN" | "CRLT" => "Carrington heliographic coordinates are positions on the Sun, not sky directions",
    "HRLN" | "HRLT" => "heliocentric radial coordinates are positions around the Sun, not sky directions",
    "SOLX" | "SOLY" | "SOLZ" => "heliocentric cartesian coordinates are positions around the Sun, not sky directions",
    "HEEX" | "HEEY" | "HEEZ" | "HEQX" | "HEQY" | "HEQZ" | "GSEX" | "GSEY" | "GSEZ" | "HCIX" | "HCIY" | "HCIZ" => {
      "heliocentric / geocentric cartesian coordinates are positions, not sky directions"
    }
    "HLON" | "HLAT" | "SLON" | "SLAT" | "HCLN" | "HCLT" => "solar surface coordinates are positions on the Sun, not sky directions",
    k if k.len() == 4 && (k.ends_with("LN") || k.ends_with("LT")) && k.chars().all(|c| c.is_ascii_uppercase()) => {
      "planetary body-fixed longitude / latitude are positions on a body, not sky directions"
    }
    _ => return None,
  })
}

/// The equatorial frame of a header: `RADESYS` (`RADECSYS`), else `EQUINOX` (`< 1984` FK4, else
/// FK5), else ICRS assumed (the flag).
fn equatorial_frame(h: &FitsHeader, equinox: Option<f64>) -> Result<(Frame, bool), String> {
  let mut assumed = false;
  let radesys = h.get("RADESYS").or_else(|| h.get("RADECSYS")).map(|s| s.trim().to_ascii_uppercase());
  let radesys = match radesys.as_deref() {
    Some("") | None => match equinox {
      None => {
        assumed = true;
        "ICRS"
      }
      Some(e) if e < 1984.0 => "FK4",
      Some(_) => "FK5",
    },
    Some(r) => r,
  };
  let frame = match radesys {
    "ICRS" | "GCRS" => Frame::icrs(),
    "FK5" => Frame::fk5(equinox),
    "FK4" => Frame::fk4(equinox, false),
    "FK4-NO-E" => Frame::fk4(equinox, true),
    r => return Err(format!("unsupported RADESYS {r}")),
  };
  Ok((frame, assumed))
}

/// The TNX / ZPX `WAT` strings of the two axes: the correction surfaces (`lngcor` looked up on
/// the longitude axis first, `latcor` on the latitude axis first, as wcstools does), ZPX's
/// `projp0..9` as ZPN `PV` parameters, and a `longpole` attribute.
fn read_wat(h: &FitsHeader, ilng: u8, ilat: u8, zpx: bool) -> Result<(IrafCorrection, PvParams, Option<f64>), String> {
  let s1 = wat::wat_string(h, ilng);
  let s2 = wat::wat_string(h, ilat);
  if s1.trim().is_empty() && s2.trim().is_empty() {
    return Err(format!(
      "{} without WAT{ilng}_nnn / WAT{ilat}_nnn cards: the IRAF correction surfaces are missing",
      if zpx { "ZPX" } else { "TNX" }
    ));
  }
  let surface = |first: &str, second: &str, key: &str| -> Result<Option<IrafSurface>, String> {
    let v = wat::wat_attr(first, key).or_else(|| wat::wat_attr(second, key));
    match v {
      None => Ok(None),
      Some(text) => {
        let nums = wat::parse_numbers(text)?;
        if nums.is_empty() {
          return Ok(None);
        }
        IrafSurface::from_numbers(&nums).map(Some).map_err(|e| format!("{key}: {e}"))
      }
    }
  };
  let corr = IrafCorrection { lng: surface(&s1, &s2, "lngcor")?, lat: surface(&s2, &s1, "latcor")? };
  if let Some(ro) = wat::wat_attr_f64(&s1, "ro").or_else(|| wat::wat_attr_f64(&s2, "ro"))
    && (ro - 180.0 / core::f64::consts::PI).abs() > 1e-6
  {
    return Err(format!("{} with ro={ro}: only the standard 180/π generating sphere is supported", if zpx { "ZPX" } else { "TNX" }));
  }
  let longpole = wat::wat_attr_f64(&s1, "longpole").or_else(|| wat::wat_attr_f64(&s2, "longpole"));
  let mut projp = PvParams::default();
  if zpx {
    let mut any = false;
    for m in 0..10 {
      if let Some(v) = wat::wat_attr_f64(&s1, &format!("projp{m}")).or_else(|| wat::wat_attr_f64(&s2, &format!("projp{m}"))) {
        projp.set(m, v);
        any = true;
      }
    }
    if !any {
      // no projp attributes: the PV cards of the latitude axis, if a writer put them there
      projp = pv_params(h, ilat);
    }
  }
  Ok((corr, projp, longpole))
}

impl Wcs {
  /// Reads the celestial WCS of a header: `Ok(None)` without a celestial `CTYPE` pair, `Err`
  /// for one this module cannot honour (the message names what).
  pub fn from_header(h: &FitsHeader) -> Result<Option<Wcs>, String> {
    // the celestial axes
    let mut lng: Option<(u8, String)> = None;
    let mut lat: Option<(u8, String)> = None;
    let mut frame_of_lng: Option<FrameKind> = None;
    let mut frame_of_lat: Option<FrameKind> = None;
    let mut sip = false;
    let mut lng_code = String::new();
    let mut lat_code = String::new();
    let mut unsupported: Option<String> = None;
    for axis in 1u8..=9 {
      let Some(ct) = h.get(&format!("CTYPE{axis}")) else {
        continue;
      };
      let (kind, code, s) = split_ctype(ct);
      let (is_lng, fk) = match kind.as_str() {
        "RA" => (true, FrameKind::Icrs),
        "DEC" => (false, FrameKind::Icrs),
        "GLON" => (true, FrameKind::Galactic),
        "GLAT" => (false, FrameKind::Galactic),
        "ELON" => (true, FrameKind::Ecliptic),
        "ELAT" => (false, FrameKind::Ecliptic),
        "HPLN" => (true, FrameKind::Helioprojective),
        "HPLT" => (false, FrameKind::Helioprojective),
        k => {
          if let Some(why) = refused_coordinate_type(k) {
            unsupported = Some(format!("CTYPE{axis} {}: {why}", ct.trim()));
          }
          continue;
        }
      };
      sip |= s;
      if is_lng && lng.is_none() {
        lng = Some((axis, kind));
        frame_of_lng = Some(fk);
        lng_code = code;
      } else if !is_lng && lat.is_none() {
        lat = Some((axis, kind));
        frame_of_lat = Some(fk);
        lat_code = code;
      }
    }
    let (Some((ilng, _)), Some((ilat, _))) = (lng, lat) else {
      return match unsupported {
        Some(e) => Err(e),
        None => Self::from_dss(h),
      };
    };
    if frame_of_lng != frame_of_lat {
      return Err(format!(
        "CTYPE{ilng} / CTYPE{ilat} name different frames ({} / {})",
        h.get(&format!("CTYPE{ilng}")).unwrap_or("").trim(),
        h.get(&format!("CTYPE{ilat}")).unwrap_or("").trim()
      ));
    }
    if lng_code != lat_code {
      return Err(format!("CTYPE projection codes differ: {lng_code} / {lat_code}"));
    }
    let mut code_str = lng_code.clone();
    let mut tpv = false;
    // IRAF TNX / ZPX: the WAT strings carry the correction surfaces (and ZPN's parameters)
    let mut iraf: Option<(bool, IrafCorrection, PvParams, Option<f64>)> = None;
    if matches!(code_str.as_str(), "TNX" | "ZPX") {
      let zpx = code_str == "ZPX";
      let (corr, projp, longpole) = read_wat(h, ilng, ilat, zpx)?;
      iraf = Some((zpx, corr, projp, longpole));
      code_str = if zpx { "ZPN".into() } else { "TAN".into() };
    }
    if code_str == "TPV" {
      tpv = true;
      code_str = "TAN".into();
    } else if code_str == "TAN" && (has_tpv_pv(h, ilng) || has_tpv_pv(h, ilat)) {
      tpv = true;
    } else if code_str.is_empty() {
      // Paper I: a celestial pair without an algorithm is linear (CAR with CRVAL offsets)
      code_str = "CAR".into();
    }
    let Some(code) = Code::parse(&code_str) else {
      return Err(format!("unsupported projection {code_str} (CTYPE{ilng} / CTYPE{ilat})"));
    };
    // linear part
    let u = [unit_scale(h, ilng)?, unit_scale(h, ilat)?];
    let crval_frame = [h.get_f64(&format!("CRVAL{ilng}")).unwrap_or(0.0) * u[0], h.get_f64(&format!("CRVAL{ilat}")).unwrap_or(0.0) * u[1]];
    // pixel axes of the celestial axes: axis i of the header is pixel axis i (no PC mixing with
    // other axes is supported); the pixel axes are the same two numbers
    let (p1, p2) = (ilng.min(ilat), ilng.max(ilat));
    let crpix = [h.get_f64(&format!("CRPIX{p1}")).unwrap_or(0.0), h.get_f64(&format!("CRPIX{p2}")).unwrap_or(0.0)];
    let key_cd = |i: u8, j: u8| format!("CD{i}_{j}");
    let key_pc = |i: u8, j: u8| format!("PC{i}_{j}");
    let has_cd = [ilng, ilat].iter().any(|&i| [p1, p2].iter().any(|&j| h.get(&key_cd(i, j)).is_some()));
    let has_pc = [ilng, ilat].iter().any(|&i| [p1, p2].iter().any(|&j| h.get(&key_pc(i, j)).is_some()));
    // rows: world axes (lng, lat); columns: pixel axes (p1, p2)
    let mut cd = [[0.0; 2]; 2];
    if has_cd {
      for (r, &i) in [ilng, ilat].iter().enumerate() {
        for (c, &j) in [p1, p2].iter().enumerate() {
          cd[r][c] = h.get_f64(&key_cd(i, j)).unwrap_or(0.0) * u[r];
        }
      }
    } else {
      let cdelt = [h.get_f64(&format!("CDELT{ilng}")).unwrap_or(1.0) * u[0], h.get_f64(&format!("CDELT{ilat}")).unwrap_or(1.0) * u[1]];
      let mut pc = [[0.0; 2]; 2];
      if has_pc {
        for (r, &i) in [ilng, ilat].iter().enumerate() {
          for (c, &j) in [p1, p2].iter().enumerate() {
            pc[r][c] = h.get_f64(&key_pc(i, j)).unwrap_or(if i == j { 1.0 } else { 0.0 });
          }
        }
      } else {
        let rho = h.get_f64(&format!("CROTA{ilat}")).or_else(|| h.get_f64("CROTA2")).unwrap_or(0.0) * DEG;
        let (s, c) = (rho.sin(), rho.cos());
        if cdelt[0] == 0.0 || cdelt[1] == 0.0 {
          return Err("CDELT zero".into());
        }
        // Paper II eq. 186: PC from CROTA with the CDELT ratio; in (lng, lat) world rows and
        // (lng axis, lat axis) pixel columns
        let m = [[c, -s * cdelt[1] / cdelt[0]], [s * cdelt[0] / cdelt[1], c]];
        // columns are pixel axes p1 < p2; the lng axis is column 0 when ilng == p1
        if ilng == p1 {
          pc = m;
        } else {
          pc = [[m[0][1], m[0][0]], [m[1][1], m[1][0]]];
        }
      }
      for r in 0..2 {
        for c in 0..2 {
          cd[r][c] = cdelt[r] * pc[r][c];
        }
      }
    }
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    if !(det.abs() > 0.0) || !det.is_finite() {
      return Err("singular CD matrix".into());
    }

    // projection parameters and the rotation keywords
    let pv_lng = pv_params(h, ilng);
    let pv_lat = pv_params(h, ilat);
    let (lat_pv, phi0, theta0, lonpole, latpole, offset_on) = if tpv {
      (PvParams::default(), None, None, h.get_f64("LONPOLE"), h.get_f64("LATPOLE"), false)
    } else if let Some((_, _, projp, longpole)) = &iraf {
      // IRAF never writes PV cards for these; `longpole` may sit in the WAT string
      (*projp, None, None, h.get_f64("LONPOLE").or(*longpole), h.get_f64("LATPOLE"), false)
    } else {
      let g = |m: usize, key: &str| if pv_lng.set[m] { Some(pv_lng.value[m]) } else { h.get_f64(key) };
      (
        pv_lat,
        pv_lng.set[1].then_some(pv_lng.value[1]),
        pv_lng.set[2].then_some(pv_lng.value[2]),
        g(3, "LONPOLE"),
        g(4, "LATPOLE"),
        pv_lng.set[0] && pv_lng.value[0] != 0.0,
      )
    };
    let projection = Projection::new(code, lat_pv, phi0, theta0)?;
    let rotation = Rotation::celset(crval_frame[0], crval_frame[1], projection.phi0, projection.theta0, lonpole, latpole)?;
    let offset = if offset_on {
      let (x0, y0) = projection.fwd(projection.phi0, projection.theta0).ok_or("fiducial point outside the projection")?;
      [x0, y0]
    } else {
      [0.0, 0.0]
    };

    // distortion
    let distortion = if let Some((zpx, corr, _, _)) = iraf {
      if zpx { Distortion::Zpx(corr) } else { Distortion::Tnx(corr) }
    } else if tpv {
      let mut t = Tpv::default();
      for m in 0..30 {
        if pv_lng.set[m] {
          t.lng[m] = pv_lng.value[m];
        }
        if pv_lat.set[m] {
          t.lat[m] = pv_lat.value[m];
        }
      }
      for m in 30..40 {
        if let Some(v) = h.get_f64(&format!("PV{ilng}_{m}")) {
          t.lng[m] = v;
        }
        if let Some(v) = h.get_f64(&format!("PV{ilat}_{m}")) {
          t.lat[m] = v;
        }
      }
      Distortion::Tpv(t)
    } else {
      match Sip::from_header(h)? {
        Some(s) if sip || s.a_order > 0 || s.b_order > 0 => Distortion::Sip(s),
        _ => Distortion::None,
      }
    };

    // frame
    let equinox = h.get_f64("EQUINOX").or_else(|| h.get_f64("EPOCH"));
    let (frame, frame_assumed) = match frame_of_lng.unwrap_or(FrameKind::Icrs) {
      FrameKind::Galactic => (Frame::galactic(), false),
      FrameKind::Ecliptic => (Frame::ecliptic(equinox), false),
      FrameKind::Helioprojective => solar::frame_from_header(h)?,
      _ => equatorial_frame(h, equinox)?,
    };

    let mut w = Wcs {
      crval_deg: [0.0; 2],
      crval_frame_deg: crval_frame,
      crpix,
      cd_deg: cd,
      projection,
      rotation,
      distortion,
      frame,
      frame_assumed,
      offset,
      axes: [ilng, ilat],
    };
    w.crval_deg = if frame.is_identity() {
      crval_frame
    } else {
      let (ra, dec) = frame.lonlat_to_icrs(crval_frame[0], crval_frame[1]);
      [ra, dec]
    };
    Ok(Some(w))
  }

  /// The DSS plate solution of a header without a celestial `CTYPE` pair (`PLTRAH…`,
  /// `PLTDEC…`, `PPO3` / `PPO6`, `XPIXELSZ` / `YPIXELSZ`, `CNPIX1` / `CNPIX2`, `AMDX` / `AMDY`):
  /// `Ok(None)` without the plate cards. A DSS cutout that also carries a `TAN` is read as that
  /// `TAN` by [`Wcs::from_header`] (as wcslib / astropy do).
  pub fn from_dss(h: &FitsHeader) -> Result<Option<Wcs>, String> {
    let Some(plate) = DssPlate::from_header(h)? else {
      return Ok(None);
    };
    let need = |k: &str| h.get_f64(k).ok_or_else(|| format!("DSS plate solution without {k}"));
    if h.get("PLTRAH").is_none() {
      return Err("DSS plate solution (AMDX / AMDY) without the plate centre PLTRAH…".into());
    }
    let ra0 = 15.0 * (need("PLTRAH")? + need("PLTRAM")? / 60.0 + need("PLTRAS")? / 3600.0);
    let dec_abs = need("PLTDECD")? + need("PLTDECM")? / 60.0 + need("PLTDECS")? / 3600.0;
    let dec0 = if h.get("PLTDECSN").is_some_and(|s| s.trim_start().starts_with('-')) { -dec_abs } else { dec_abs };
    let (xsz, ysz) = (need("XPIXELSZ")?, need("YPIXELSZ")?);
    if xsz == 0.0 || ysz == 0.0 {
      return Err("DSS plate solution with a zero pixel size".into());
    }
    let (ppo3, ppo6) = (need("PPO3")?, need("PPO6")?);
    let cnpix = [h.get_f64("CNPIX1").unwrap_or(0.0), h.get_f64("CNPIX2").unwrap_or(0.0)];
    // wcstools dsspos.c: x = pixel + CNPIX1 − 0.5, xmm = (PPO3 − x · XPIXELSZ) / 1000,
    // ymm = (y · YPIXELSZ − PPO6) / 1000
    let crpix = [ppo3 / xsz - cnpix[0] + 0.5, ppo6 / ysz - cnpix[1] + 0.5];
    let cd = [[-xsz / 1000.0, 0.0], [0.0, ysz / 1000.0]];
    let (frame, frame_assumed) = equatorial_frame(h, h.get_f64("EQUINOX"))?;
    let projection = Projection::new(Code::Tan, PvParams::default(), None, None)?;
    let rotation = Rotation::celset(ra0, dec0, projection.phi0, projection.theta0, None, None)?;
    let crval_frame = [ra0, dec0];
    let (ra, dec) = frame.lonlat_to_icrs(ra0, dec0);
    Ok(Some(Wcs {
      crval_deg: [ra, dec],
      crval_frame_deg: crval_frame,
      crpix,
      cd_deg: cd,
      projection,
      rotation,
      distortion: Distortion::Dss(plate),
      frame,
      frame_assumed,
      offset: [0.0; 2],
      axes: [1, 2],
    }))
  }

  /// A TAN WCS in ICRS with the default rotation (`LONPOLE = 180°`) and no distortion.
  pub fn tan(crval_deg: [f64; 2], crpix: [f64; 2], cd_deg: [[f64; 2]; 2]) -> Wcs {
    let projection = Projection::new(Code::Tan, PvParams::default(), None, None).expect("TAN has no parameters");
    let rotation = Rotation::from_pole(crval_deg[0], crval_deg[1], 180.0);
    Wcs {
      crval_deg,
      crval_frame_deg: crval_deg,
      crpix,
      cd_deg,
      projection,
      rotation,
      distortion: Distortion::None,
      frame: Frame::icrs(),
      frame_assumed: false,
      offset: [0.0; 2],
      axes: [1, 2],
    }
  }

  /// The header facts.
  pub fn facts(&self) -> WcsFacts {
    WcsFacts {
      projection: self.projection.code.name(),
      distortion: self.distortion.name(),
      frame: self.frame.kind.name(),
      equinox: self.frame.equinox,
      frame_assumed: self.frame_assumed,
      lonpole_deg: self.rotation.phip,
      latpole_deg: self.rotation.latp,
    }
  }

  /// `true` when a distortion polynomial (SIP / TPV / TNX / ZPX / DSS) is applied.
  pub fn has_distortion(&self) -> bool {
    !matches!(self.distortion, Distortion::None)
  }

  /// `true` for an undistorted TAN with the default rotation (`θ0 = 90°`, `φp = 180°`, no
  /// fiducial offset): pixel → tangent-plane point is affine (the exact mesh registration).
  pub fn is_plain_tan(&self) -> bool {
    self.projection.code == Code::Tan
      && !self.has_distortion()
      && self.projection.theta0 == 90.0
      && self.projection.phi0 == 0.0
      && self.rotation.phip == 180.0
      && self.offset == [0.0; 2]
  }

  // ── pixel → sky ──

  /// Intermediate world coordinates `(x, y)` (degrees, longitude / latitude order) of a pixel:
  /// SIP, the linear part, TPV / TNX / ZPX / DSS and the fiducial offset.
  pub fn pixel_to_intermediate(&self, p1: f64, p2: f64) -> (f64, f64) {
    let (mut u, mut v) = (p1 - self.crpix[0], p2 - self.crpix[1]);
    if let Distortion::Sip(s) = &self.distortion {
      (u, v) = s.forward(u, v);
    }
    let mut x = self.cd_deg[0][0] * u + self.cd_deg[0][1] * v;
    let mut y = self.cd_deg[1][0] * u + self.cd_deg[1][1] * v;
    match &self.distortion {
      Distortion::Tpv(t) => (x, y) = t.forward(x, y),
      Distortion::Tnx(c) | Distortion::Zpx(c) => (x, y) = c.forward(x, y),
      Distortion::Dss(d) => (x, y) = d.forward(x, y),
      _ => {}
    }
    (x + self.offset[0], y + self.offset[1])
  }

  /// Native spherical `(φ, θ)` degrees of a pixel; `None` outside the projection.
  pub fn pixel_to_native(&self, p1: f64, p2: f64) -> Option<(f64, f64)> {
    let (x, y) = self.pixel_to_intermediate(p1, p2);
    let (phi, theta) = self.projection.inv(x, y)?;
    (theta.abs() <= 90.0 + 1e-9 && phi.abs() <= 180.0 + 1e-9 && phi.is_finite() && theta.is_finite())
      .then_some((phi, theta))
  }

  /// Unit direction of a pixel in the header's frame; `None` outside the projection.
  pub fn pixel_to_frame_dir(&self, p1: f64, p2: f64) -> Option<V3> {
    let (phi, theta) = self.pixel_to_native(p1, p2)?;
    Some(self.rotation.to_celestial(rotation::dir_of(phi, theta)))
  }

  /// `(lon, lat)` degrees in the header's frame (`RADESYS` / galactic / ecliptic) of a pixel.
  pub fn pixel_to_world(&self, p1: f64, p2: f64) -> Option<(f64, f64)> {
    Some(rotation::lon_lat_of(self.pixel_to_frame_dir(p1, p2)?))
  }

  /// ICRS unit direction of a pixel; `None` outside the projection.
  pub fn pixel_to_icrs(&self, p1: f64, p2: f64) -> Option<V3> {
    Some(self.frame.to_icrs(self.pixel_to_frame_dir(p1, p2)?))
  }

  /// ICRS unit direction of a pixel, NaN components outside the projection.
  pub fn pixel_to_equatorial(&self, p1: f64, p2: f64) -> V3 {
    self.pixel_to_icrs(p1, p2).unwrap_or([f64::NAN; 3])
  }

  /// ICRS RA / Dec (degrees) of a pixel, NaN outside the projection.
  pub fn pixel_to_radec_deg(&self, p1: f64, p2: f64) -> (f64, f64) {
    match self.pixel_to_icrs(p1, p2) {
      Some(d) => rotation::lon_lat_of(d),
      None => (f64::NAN, f64::NAN),
    }
  }

  // ── sky → pixel ──

  /// Pixel of intermediate world coordinates (undoing the offset, the intermediate distortion,
  /// the linear part, SIP).
  pub fn intermediate_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    let (mut x, mut y) = (x - self.offset[0], y - self.offset[1]);
    match &self.distortion {
      Distortion::Tpv(t) => (x, y) = t.inverse(x, y)?,
      Distortion::Tnx(c) | Distortion::Zpx(c) => (x, y) = c.inverse(x, y)?,
      Distortion::Dss(d) => (x, y) = d.inverse(x, y)?,
      _ => {}
    }
    let [[a, b], [c, d]] = self.cd_deg;
    let det = a * d - b * c;
    let (mut u, mut v) = ((d * x - b * y) / det, (-c * x + a * y) / det);
    if let Distortion::Sip(s) = &self.distortion {
      (u, v) = s.inverse(u, v)?;
    }
    Some((self.crpix[0] + u, self.crpix[1] + v))
  }

  /// Pixel of native spherical `(φ, θ)` degrees.
  pub fn native_to_pixel(&self, phi: f64, theta: f64) -> Option<(f64, f64)> {
    let (x, y) = self.projection.fwd(phi, theta)?;
    self.intermediate_to_pixel(x, y)
  }

  /// Pixel of a unit direction in the header's frame.
  pub fn frame_dir_to_pixel(&self, d: V3) -> Option<(f64, f64)> {
    let (phi, theta) = rotation::phi_theta_of(self.rotation.to_native(d));
    self.native_to_pixel(phi, theta)
  }

  /// Pixel of `(lon, lat)` degrees in the header's frame.
  pub fn world_to_pixel(&self, lon: f64, lat: f64) -> Option<(f64, f64)> {
    self.frame_dir_to_pixel(rotation::dir_of(lon, lat))
  }

  /// Pixel of an ICRS direction (any length); `None` where it has no image.
  pub fn equatorial_to_pixel(&self, d: V3) -> Option<(f64, f64)> {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if !(l > 0.0) || !l.is_finite() {
      return None;
    }
    self.frame_dir_to_pixel(self.frame.from_icrs([d[0] / l, d[1] / l, d[2] / l]))
  }

  /// Pixel of ICRS RA / Dec (degrees).
  pub fn radec_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
    self.equatorial_to_pixel(rotation::dir_of(ra_deg, dec_deg))
  }

  // ── tangent plane (TAN family) ──

  /// `(ŝ0, ê, n̂)`: the unit vector at the reference point and the local east / north unit
  /// vectors, in the header's frame.
  pub fn tangent_basis(&self) -> (V3, V3, V3) {
    let (ra, dec) = (self.crval_frame_deg[0] * DEG, self.crval_frame_deg[1] * DEG);
    let s0 = [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()];
    let e = [-ra.sin(), ra.cos(), 0.0];
    let n = [-dec.sin() * ra.cos(), -dec.sin() * ra.sin(), dec.cos()];
    (s0, e, n)
  }

  /// Standard coordinates `(ξ, η)` (radians) of a pixel: the intermediate coordinates.
  pub fn pixel_to_standard(&self, p1: f64, p2: f64) -> (f64, f64) {
    let (x, y) = self.pixel_to_intermediate(p1, p2);
    (x * DEG, y * DEG)
  }

  /// Pixel of standard coordinates `(ξ, η)` (radians).
  pub fn standard_to_pixel(&self, xi: f64, eta: f64) -> Option<(f64, f64)> {
    self.intermediate_to_pixel(xi / DEG, eta / DEG)
  }

  /// ICRS point of the tangent plane at the reference direction (`ŝ0 · p = 1`) of a pixel: the
  /// pixel's direction, not normalised. For a plain TAN this is the affine `ŝ0 + ξ ê + η n̂`
  /// (the exact mesh registration); for any other projection it is the pixel's direction
  /// rescaled onto that plane, NaN behind it or outside the projection.
  pub fn pixel_to_tangent_point(&self, p1: f64, p2: f64) -> V3 {
    if self.is_plain_tan() {
      let (xi, eta) = self.pixel_to_standard(p1, p2);
      let (s0, e, n) = self.tangent_basis();
      let p = [
        s0[0] + xi * e[0] + eta * n[0],
        s0[1] + xi * e[1] + eta * n[1],
        s0[2] + xi * e[2] + eta * n[2],
      ];
      return self.frame.to_icrs_linear(p);
    }
    let Some(d) = self.pixel_to_icrs(p1, p2) else {
      return [f64::NAN; 3];
    };
    let s0 = rotation::dir_of(self.crval_deg[0], self.crval_deg[1]);
    let w = d[0] * s0[0] + d[1] * s0[1] + d[2] * s0[2];
    if !(w > 1e-9) {
      return [f64::NAN; 3];
    }
    [d[0] / w, d[1] / w, d[2] / w]
  }

  /// Samples the mapping on an `n × n` grid over `[0.5, nx + 0.5] × [0.5, ny + 0.5]` and returns
  /// the largest angular extent (degrees) between valid samples, `None` when nothing maps.
  pub fn angular_extent_deg(&self, nx: u32, ny: u32, n: usize) -> Option<f64> {
    let n = n.max(2);
    let mut dirs: Vec<V3> = Vec::with_capacity(n * n);
    for j in 0..n {
      for i in 0..n {
        let p1 = 0.5 + nx as f64 * i as f64 / (n - 1) as f64;
        let p2 = 0.5 + ny as f64 * j as f64 / (n - 1) as f64;
        if let Some(d) = self.pixel_to_icrs(p1, p2) {
          dirs.push(d);
        }
      }
    }
    if dirs.is_empty() {
      return None;
    }
    let mut worst = 0.0f64;
    for (i, a) in dirs.iter().enumerate() {
      for b in &dirs[i + 1..] {
        let c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).clamp(-1.0, 1.0);
        worst = worst.max(c.acos() / DEG);
      }
    }
    Some(worst)
  }
}

impl Frame {
  /// A linear map of a (not necessarily unit) vector: the rotation without E-terms (the
  /// tangent-plane points of a plain TAN stay on the plane).
  pub(crate) fn to_icrs_linear(&self, v: V3) -> V3 {
    if self.is_identity() {
      return v;
    }
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let u = self.to_icrs([v[0] / l, v[1] / l, v[2] / l]);
    [u[0] * l, u[1] * l, u[2] * l]
  }
}

impl SkyMapping for Wcs {
  fn pix_to_sky(&self, x: f64, y: f64) -> Option<(f64, f64)> {
    Some(rotation::lon_lat_of(self.pixel_to_icrs(x, y)?))
  }
  fn sky_to_pix(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
    self.radec_to_pixel(ra, dec)
  }
}
