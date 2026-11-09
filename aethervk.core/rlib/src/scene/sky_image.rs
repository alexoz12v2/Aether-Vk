//! User supplied telescope image (FITS, or PNG / JPEG with a WCS sidecar) registered on the sky.
//!
//! ## Loading
//! - **FITS**: the primary HDU only (80-character cards in 2880-byte blocks), `NAXIS = 2` (extra
//!   axes of length 1 are accepted), `BITPIX` 8 / 16 / 32 / 64 / −32 / −64, big-endian, physical
//!   value `BZERO + BSCALE · raw`, integer `BLANK` → NaN (drawn transparent).
//! - **PNG / JPEG**: decoded with the engine's decoders (`png-decoder`, `zune-jpeg`) to a luminance
//!   (Rec. 709 weights on the encoded values: the HUD is grey). The WCS comes from a sidecar: a
//!   FITS header as text (astrometry.net `.wcs`: 2880-byte blocks without newlines, or one card per
//!   line). Without an explicit sidecar `<stem>.wcs` next to the image is tried.
//!
//! ## Pixel convention
//! FITS 1-based pixel coordinates `(p1, p2)`: pixel `(1, 1)` is the centre of the first stored
//! pixel, the image spans `[0.5, n + 0.5]` on each axis. Texture row `j` (upload order) is
//! `p2 = j + 1`; for PNG / JPEG the rows are in file order (top row first), which is how
//! astrometry.net numbers them, so the same mapping holds. Texture coordinates:
//! `u = (p1 − 0.5) / nx`, `v = (p2 − 0.5) / ny`.
//!
//! ## WCS
//! Any celestial WCS of Papers I–III ([`wcs`]): the linear part (`CDi_j`, `CDELTi · PCi_j`, or
//! `CDELTi` + `CROTA2`, `CUNITi`), every Paper II projection (zenithal, cylindrical,
//! pseudo-cylindrical, conic, polyconic, quad-cube) and the HEALPix `HPX` / `XPH`, their `PVi_m`
//! parameters, `LONPOLE` / `LATPOLE`, the SIP and TPV distortions (applied: flagged in
//! [`SKY_FLAG_DISTORTION_APPLIED`]), the frames ICRS / FK5 / FK4 / FK4-NO-E / galactic / ecliptic
//! (all converted to ICRS; nothing → ICRS assumed, [`SKY_FLAG_FRAME_ASSUMED`]), celestial axes
//! in either order and at any axis number. Numerically wcslib / astropy
//! (`test_assets/fits_corpus/wcs_reference`). HiPS tiles without a WCS are registered by their
//! HEALPix layout ([`hips`]).
//!
//! The registration mesh ([`grid_vertices`]): for a plain TAN (no distortion, default rotation)
//! the vertices are points of the tangent plane, `ŝ0 + ξ ê + η n̂` with `(ξ, η)` the intermediate
//! coordinates in radians — the map pixel → plane is affine, so a mesh whose texture coordinates
//! are the pixel coordinates is registered **exactly** under any pinhole projection
//! (perspective-correct interpolation is linear on the plane). Any other WCS gets unit
//! directions on a finer grid (cell ≤ 0.75° of sky, 32–256 vertices per side); cells with a
//! corner outside the projection are dropped.
//!
//! ## Registration
//! Equatorial J2000 (ICRS) → ecliptic J2000 (the engine's world axes, `SUN_ECLIPJ2000`) by the
//! rotation about +X by the ANISE `J2000_TO_ECLIPJ2000_ANGLE_RAD` (IAU 1976, 84381.448″, the
//! SPICE `ECLIPJ2000` definition the ephemerides use). Aberration: none (the WCS of a plate solved
//! against a catalogue is astrometric, like the rendered comet, see `dust_v3.md`).
//!
//! ## Store
//! Loaded images live in a process-wide store keyed by the scene's `SceneContext` address (the
//! render thread has no scene id), see [`store_insert`]. The GPU side
//! (`gpu_backends::vulkan::device::sky_overlay`) uploads a key's image when its generation
//! changes.

use alloc::{
  collections::BTreeMap,
  string::{String, ToString},
  sync::Arc,
  vec::Vec,
};
use core::sync::atomic::{AtomicU64, Ordering};

pub use wcs::{SkyMapping, Wcs, WcsFacts};

#[cfg(test)]
mod tests;

pub mod fits_metadata;
pub mod fits_stretch;
pub mod hips;
pub mod wcs;
#[cfg(test)]
pub(crate) mod fits_tests;
#[cfg(test)]
mod fits_corpus_tests;

/// J2000 equator → ecliptic J2000 rotation angle about +X (ANISE / SPICE `ECLIPJ2000`, IAU 1976
/// 84381.448″): the engine's one obliquity, `simulation_api::astrometry::OBLIQUITY_J2000_RAD`.
pub const OBLIQUITY_J2000_RAD: f64 = crate::simulation_api::astrometry::OBLIQUITY_J2000_RAD;

/// vertices per side of the registration mesh
pub const SKY_IMAGE_GRID: usize = 32;
/// largest texture side; larger images are box-averaged by an integer factor for the upload
pub const SKY_IMAGE_MAX_TEXTURE: u32 = 8192;
/// black / white points of the display stretch, percentiles of the finite pixel values
pub const SKY_IMAGE_BLACK_PERCENTILE: f64 = 1.0;
pub const SKY_IMAGE_WHITE_PERCENTILE: f64 = 99.5;

// ── CSkyImageInfo.flags bits (present / derived header facts) ────────────────
/// `date_obs_us` is valid (any keyword of the `fits_metadata` date chain: `DATE-AVG` … `DATE`)
pub const SKY_FLAG_DATE_OBS: u32 = 1 << 0;
/// `exptime_s` is valid (`EXPTIME`, or `EXPOSURE`)
pub const SKY_FLAG_EXPTIME: u32 = 1 << 1;
/// `site_lat_deg` is valid
pub const SKY_FLAG_SITE_LAT: u32 = 1 << 2;
/// `site_lon_deg` is valid
pub const SKY_FLAG_SITE_LON: u32 = 1 << 3;
/// `site_elev_m` is valid
pub const SKY_FLAG_SITE_ELEV: u32 = 1 << 4;
/// a distortion polynomial (SIP / TPV) is present and applied (before 2026-11 the bit meant
/// "SIP present and ignored"; the value is kept for the FFI / C# side)
pub const SKY_FLAG_DISTORTION_APPLIED: u32 = 1 << 5;
/// former name of [`SKY_FLAG_DISTORTION_APPLIED`] (same bit)
pub const SKY_FLAG_SIP_IGNORED: u32 = SKY_FLAG_DISTORTION_APPLIED;
/// PNG / JPEG source (already display-stretched: a linear stretch is the natural default)
pub const SKY_FLAG_RASTER: u32 = 1 << 6;
/// the WCS came from a sidecar
pub const SKY_FLAG_WCS_SIDECAR: u32 = 1 << 7;
/// no `RADESYS` / `EQUINOX`: ICRS assumed
pub const SKY_FLAG_FRAME_ASSUMED: u32 = 1 << 8;
/// `DATE-OBS` / `MJD-OBS` read in UTC because `TIMESYS` is absent
pub const SKY_FLAG_TIMESYS_ASSUMED: u32 = 1 << 9;

const DEG: f64 = core::f64::consts::PI / 180.0;

// ─── vectors (f64) ──────────────────────────────────────────────────────────

type V3 = [f64; 3];


#[allow(dead_code)]
pub(crate) fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[allow(dead_code)]
pub(crate) fn normalize(a: V3) -> V3 {
  let l = dot(a, a).sqrt();
  [a[0] / l, a[1] / l, a[2] / l]
}

/// Unit vector of right ascension / declination (degrees) in the frame they are measured in
/// (`simulation_api::astrometry::dir_of_ra_dec` in degrees).
pub fn dir_of_ra_dec(ra_deg: f64, dec_deg: f64) -> V3 {
  crate::simulation_api::astrometry::dir_of_ra_dec(ra_deg * DEG, dec_deg * DEG)
}

/// Right ascension in `[0, 360)` and declination (degrees) of a (not necessarily unit) vector.
pub fn ra_dec_of(d: V3) -> (f64, f64) {
  let mut ra = d[1].atan2(d[0]) / DEG;
  if ra < 0.0 {
    ra += 360.0;
  }
  let dec = d[2].atan2((d[0] * d[0] + d[1] * d[1]).sqrt()) / DEG;
  (ra, dec)
}

/// Equatorial J2000 (ICRS) → ecliptic J2000 (`R1(ε)`: x unchanged, the ecliptic pole is the
/// equatorial `(0, −sin ε, cos ε)`): `simulation_api::astrometry`'s, the engine's one rotation.
pub fn equatorial_to_ecliptic_j2000(v: V3) -> V3 {
  crate::simulation_api::astrometry::equatorial_to_ecliptic_j2000(v)
}

/// Inverse of [`equatorial_to_ecliptic_j2000`].
pub fn ecliptic_to_equatorial_j2000(v: V3) -> V3 {
  crate::simulation_api::astrometry::ecliptic_to_equatorial_j2000(v)
}

// ─── FITS header ─────────────────────────────────────────────────────────────

/// Keyword → value text of a FITS header (strings unquoted, comments stripped). The first
/// occurrence of a keyword wins.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitsHeader {
  pub cards: Vec<(String, String)>,
}

/// One 80-character card → `(keyword, value)`; `None` for commentary cards and cards without a
/// value indicator.
fn parse_card(card: &[u8]) -> Option<(String, String)> {
  let card = core::str::from_utf8(card).ok()?;
  let key = card.get(..8.min(card.len()))?.trim_end();
  if key.is_empty() || key == "COMMENT" || key == "HISTORY" {
    return None;
  }
  if card.get(8..10) != Some("= ") && card.get(8..9) != Some("=") {
    return None;
  }
  let raw = card.get(9..).unwrap_or("").trim_start_matches(' ');
  let value = if let Some(rest) = raw.strip_prefix('\'') {
    // string: '' is an escaped quote; trailing blanks are not significant
    let mut out = String::new();
    let mut chars = rest.chars().peekable();
    while let Some(c) = chars.next() {
      if c == '\'' {
        if chars.peek() == Some(&'\'') {
          out.push('\'');
          chars.next();
        } else {
          break;
        }
      } else {
        out.push(c);
      }
    }
    out.trim_end().to_string()
  } else {
    raw.split('/').next().unwrap_or("").trim().to_string()
  };
  Some((key.to_string(), value))
}

impl FitsHeader {
  /// Parses the cards of a header starting at `bytes[0]`. Returns the header and the byte length
  /// of the header blocks (the `END` card's block rounded up to 2880), `None` without `END`.
  pub fn parse_blocks(bytes: &[u8]) -> (Self, Option<usize>) {
    let mut h = Self::default();
    let mut off = 0usize;
    while off + 80 <= bytes.len() {
      let card = &bytes[off..off + 80];
      off += 80;
      if card.starts_with(b"END") && card[3..].iter().all(|&b| b == b' ') {
        return (h, Some(off.div_ceil(2880) * 2880));
      }
      if let Some((k, v)) = parse_card(card) {
        h.push(k, v);
      }
    }
    (h, None)
  }

  /// Parses a header written as text: one card per line, or FITS blocks without newlines (the
  /// astrometry.net `.wcs` file is a FITS header).
  pub fn parse_text(bytes: &[u8]) -> Self {
    if !bytes.contains(&b'\n') {
      return Self::parse_blocks(bytes).0;
    }
    let mut h = Self::default();
    for line in bytes.split(|&b| b == b'\n') {
      let line = line.strip_suffix(b"\r").unwrap_or(line);
      if line.starts_with(b"END") && line[3..].iter().all(|&b| b == b' ') {
        break;
      }
      // pad to the card width so the fixed columns hold
      let mut card = [b' '; 80];
      let n = line.len().min(80);
      card[..n].copy_from_slice(&line[..n]);
      if let Some((k, v)) = parse_card(&card) {
        h.push(k, v);
      }
    }
    h
  }

  /// The header of 80-column card records (the cards of one HDU before `END`), parsed exactly
  /// like [`Self::parse_blocks`] parses the same bytes.
  #[cfg(feature = "cfitsio")]
  pub fn from_records(records: &[[u8; 80]]) -> Self {
    let mut h = Self::default();
    for card in records {
      if let Some((k, v)) = parse_card(card) {
        h.push(k, v);
      }
    }
    h
  }

  fn push(&mut self, k: String, v: String) {
    if self.get(&k).is_none() {
      self.cards.push((k, v));
    }
  }

  pub fn get(&self, key: &str) -> Option<&str> {
    self.cards.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
  }

  /// A number (FITS allows `D` exponents).
  pub fn get_f64(&self, key: &str) -> Option<f64> {
    let v = self.get(key)?;
    let v = v.replace(['D', 'd'], "E");
    v.trim().parse::<f64>().ok()
  }

  pub fn get_i64(&self, key: &str) -> Option<i64> {
    let v = self.get(key)?.trim();
    v.parse::<i64>().ok().or_else(|| {
      // integers written as reals (`2.0`)
      let f = v.parse::<f64>().ok()?;
      (f.fract() == 0.0).then_some(f as i64)
    })
  }

  /// Degrees from a number or a sexagesimal string (`+45:30:00`, `-110 56 00`).
  pub fn get_degrees(&self, key: &str) -> Option<f64> {
    self.get_f64(key).or_else(|| parse_sexagesimal(self.get(key)?))
  }

  /// Merges `other` over `self` (its keywords win).
  pub fn overlay(&self, other: &FitsHeader) -> FitsHeader {
    let mut h = other.clone();
    for (k, v) in self.cards.iter() {
      h.push(k.clone(), v.clone());
    }
    h
  }
}

/// `±dd:mm:ss.s` or `±dd mm ss.s` → degrees.
pub fn parse_sexagesimal(s: &str) -> Option<f64> {
  let s = s.trim();
  let (neg, body) = match s.strip_prefix('-') {
    Some(r) => (true, r),
    None => (false, s.strip_prefix('+').unwrap_or(s)),
  };
  let mut parts = body.split([':', ' ']).filter(|p| !p.is_empty());
  let d = parts.next()?.parse::<f64>().ok()?;
  let m = parts.next().map(|p| p.parse::<f64>()).transpose().ok()?.unwrap_or(0.0);
  let sec = parts.next().map(|p| p.parse::<f64>()).transpose().ok()?.unwrap_or(0.0);
  if parts.next().is_some() {
    return None;
  }
  let v = d + m / 60.0 + sec / 3600.0;
  Some(if neg { -v } else { v })
}

// The WCS itself: `wcs` (re-exported `Wcs`, `SkyMapping`, `WcsFacts`).

// ─── observation metadata ────────────────────────────────────────────────────

/// Header facts the UI offers to apply (all optional).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SkyImageMeta {
  /// start of the exposure, TDB microseconds since J2000 TDB (`hifitime` `to_tdb_duration`, the
  /// scale `avkSimulationContext_seekEpochSync` takes)
  pub date_obs_tdb_us: Option<i64>,
  pub exptime_s: Option<f64>,
  pub site_lat_deg: Option<f64>,
  /// east positive
  pub site_lon_deg: Option<f64>,
  pub site_elev_m: Option<f64>,
  /// `TIMESYS` absent: UTC assumed
  pub timesys_assumed: bool,
}

impl SkyImageMeta {
  /// From the header facts (`fits_metadata::facts_of_header` / `FitsMetadata::facts`): the one
  /// date path of the engine. The date is the start of the exposure (`date_start_tdb_us`); the
  /// longitude is east positive after the facts' west-convention and LST rules.
  pub fn from_facts(f: &fits_metadata::FitsFacts) -> Self {
    Self {
      date_obs_tdb_us: f.date_start_tdb_us,
      exptime_s: f.exposure_s,
      site_lat_deg: f.site_lat_deg,
      site_lon_deg: f.site_lon_deg,
      site_elev_m: f.site_elev_m,
      timesys_assumed: f.timesys_assumed,
    }
  }
}

// ─── image ───────────────────────────────────────────────────────────────────

/// A loaded sky image: physical pixel values (NaN = blank), its WCS and header facts.
#[derive(Debug, Clone)]
pub struct SkyImage {
  pub nx: u32,
  pub ny: u32,
  /// row-major, row 0 = FITS `p2 = 1` (file order)
  pub data: Vec<f32>,
  pub header: FitsHeader,
  pub wcs: Option<Wcs>,
  /// why a WCS present in the header was refused
  pub wcs_error: Option<String>,
  /// a HiPS tile / Allsky recognised without a WCS (`hips::load_mapping`)
  pub hips: Option<hips::HipsImage>,
  pub hips_facts: Option<hips::HipsFacts>,
  pub meta: SkyImageMeta,
  /// `SKY_FLAG_*` facts about the source
  pub flags: u32,
  /// display black / white points ([`SKY_IMAGE_BLACK_PERCENTILE`], [`SKY_IMAGE_WHITE_PERCENTILE`])
  pub black: f32,
  pub white: f32,
}

/// Decodes the first image HDU of a FITS file into `(nx, ny, values, header)`: a thin wrapper
/// over [`fits_metadata::read_fits`] (CFITSIO with the `cfitsio` feature, else the pure-Rust
/// primary-HDU parser [`parse_fits_legacy`]).
pub fn parse_fits(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>, FitsHeader), String> {
  let (img, _) = fits_metadata::read_fits(bytes)?;
  Ok((img.nx, img.ny, img.data, img.header))
}

/// The pure-Rust parser of the primary HDU of a FITS file into `(nx, ny, values, header)`: every
/// BITPIX, BZERO / BSCALE applied, BLANK → NaN; extensions and tile compression are not read. The
/// `read_fits` path without the `cfitsio` feature, and the oracle the CFITSIO path is tested
/// against.
#[cfg(any(test, not(feature = "cfitsio")))]
pub(crate) fn parse_fits_legacy(bytes: &[u8]) -> Result<(u32, u32, Vec<f32>, FitsHeader), String> {
  if !bytes.starts_with(b"SIMPLE  =") {
    return Err("not a FITS file (SIMPLE missing)".to_string());
  }
  let (h, Some(data_off)) = FitsHeader::parse_blocks(bytes) else {
    return Err("FITS header without END".to_string());
  };
  if h.get("SIMPLE").map(str::trim) != Some("T") {
    return Err("SIMPLE is not T".to_string());
  }
  let bitpix = h.get_i64("BITPIX").ok_or("BITPIX missing")?;
  let naxis = h.get_i64("NAXIS").ok_or("NAXIS missing")?;
  if naxis < 2 {
    return Err(alloc::format!(
      "NAXIS = {naxis}: no image in the primary HDU (extensions are not read)"
    ));
  }
  let nx = h.get_i64("NAXIS1").ok_or("NAXIS1 missing")?;
  let ny = h.get_i64("NAXIS2").ok_or("NAXIS2 missing")?;
  for a in 3..=naxis {
    if h.get_i64(&alloc::format!("NAXIS{a}")).unwrap_or(1) != 1 {
      return Err(alloc::format!("NAXIS = {naxis}: only 2-D images"));
    }
  }
  if nx <= 0 || ny <= 0 || nx > 1 << 20 || ny > 1 << 20 {
    return Err(alloc::format!("bad image size {nx} × {ny}"));
  }
  let bytes_per = match bitpix {
    8 => 1,
    16 => 2,
    32 | -32 => 4,
    64 | -64 => 8,
    b => return Err(alloc::format!("unsupported BITPIX {b}")),
  };
  let n = (nx * ny) as usize;
  let data = bytes.get(data_off..data_off + n * bytes_per).ok_or("FITS data truncated")?;
  let bzero = h.get_f64("BZERO").unwrap_or(0.0);
  let bscale = h.get_f64("BSCALE").unwrap_or(1.0);
  let blank = if bitpix > 0 { h.get_i64("BLANK") } else { None };
  let phys = |raw: i64| -> f32 {
    if blank == Some(raw) {
      f32::NAN
    } else {
      (bzero + bscale * raw as f64) as f32
    }
  };
  let mut out = Vec::with_capacity(n);
  for c in data.chunks_exact(bytes_per) {
    out.push(match bitpix {
      8 => phys(c[0] as i64),
      16 => phys(i16::from_be_bytes([c[0], c[1]]) as i64),
      32 => phys(i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as i64),
      64 => phys(i64::from_be_bytes(c.try_into().unwrap())),
      -32 => (bzero + bscale * f32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f64) as f32,
      _ => (bzero + bscale * f64::from_be_bytes(c.try_into().unwrap())) as f32,
    });
  }
  Ok((nx as u32, ny as u32, out, h))
}

/// Decodes a PNG / JPEG to luminance (`[0, 1]` for 8-bit, Rec. 709 weights on encoded values),
/// rows in file order.
pub fn decode_raster(ext: &str, bytes: &[u8]) -> Result<(u32, u32, Vec<f32>), String> {
  let luma =
    |r: u8, g: u8, b: u8| (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
  match ext {
    "png" => {
      let (header, px) = png_decoder::decode(bytes).map_err(|e| alloc::format!("PNG: {e:?}"))?;
      // the decoder expands every colour type to RGBA8
      let data = px.iter().map(|p| luma(p[0], p[1], p[2])).collect();
      Ok((header.width, header.height, data))
    }
    "jpg" | "jpeg" => {
      let mut dec = zune_jpeg::JpegDecoder::new(zune_core::bytestream::ZCursor::new(bytes));
      dec.decode_headers().map_err(|e| alloc::format!("JPEG: {e:?}"))?;
      let info = dec.info().ok_or("JPEG: no info")?;
      let px = dec.decode().map_err(|e| alloc::format!("JPEG: {e:?}"))?;
      let (w, h) = (info.width as u32, info.height as u32);
      let n = (w * h) as usize;
      let comps = if n > 0 { px.len() / n } else { 0 };
      let data = match comps {
        1 => px.iter().map(|&v| v as f32 / 255.0).collect(),
        3 | 4 => px.chunks_exact(comps).map(|c| luma(c[0], c[1], c[2])).collect(),
        c => return Err(alloc::format!("JPEG: {c} components")),
      };
      Ok((w, h, data))
    }
    e => Err(alloc::format!("unsupported image type .{e}")),
  }
}

/// Percentile `p` (0..100) of the finite values, on a strided subsample of at most 1 M values.
pub fn percentile(data: &[f32], p: f64) -> Option<f32> {
  let stride = (data.len() / 1_000_000).max(1);
  let mut v: Vec<f32> = data.iter().step_by(stride).copied().filter(|x| x.is_finite()).collect();
  if v.is_empty() {
    return None;
  }
  let k = (((p / 100.0) * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
  let (_, x, _) = v.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
  Some(*x)
}

fn ext_of(path: &str) -> String {
  path.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

fn read_file(path: &str) -> Result<Vec<u8>, String> {
  let m = aethervk_oshal_rlib::os::files::Mmap::open(path)
    .map_err(|e| alloc::format!("cannot open {path}: {e:?}"))?;
  Ok(m.as_ref().to_vec())
}

impl SkyImage {
  /// Assembles an image from decoded values and its header (`sidecar` overrides the WCS and
  /// metadata keywords); the card comments are not known.
  pub fn from_parts(
    nx: u32,
    ny: u32,
    data: Vec<f32>,
    header: FitsHeader,
    sidecar: Option<FitsHeader>,
    raster: bool,
  ) -> Self {
    Self::from_parts_cards(nx, ny, data, header, None, sidecar, raster)
  }

  /// [`Self::from_parts`] with the file's cards (`fits_metadata::read_fits`), whose comments the
  /// facts read (an "averaged" `DATE-OBS`, a west-longitude wording).
  pub fn from_parts_cards(
    nx: u32,
    ny: u32,
    data: Vec<f32>,
    header: FitsHeader,
    cards: Option<&[fits_metadata::FitsCard]>,
    sidecar: Option<FitsHeader>,
    raster: bool,
  ) -> Self {
    let mut flags = if raster { SKY_FLAG_RASTER } else { 0 };
    let header = match sidecar {
      Some(s) => {
        if s.get("CTYPE1").is_some() {
          flags |= SKY_FLAG_WCS_SIDECAR;
        }
        header.overlay(&s)
      }
      None => header,
    };
    let (wcs, wcs_error) = match Wcs::from_header(&header) {
      Ok(w) => (w, None),
      Err(e) => (None, Some(e)),
    };
    if let Some(w) = &wcs {
      if w.has_distortion() {
        flags |= SKY_FLAG_DISTORTION_APPLIED;
      }
      if w.frame_assumed {
        flags |= SKY_FLAG_FRAME_ASSUMED;
      }
    }
    let (hips, hips_facts) = match wcs {
      None => match hips::load_mapping(None, &header, nx, ny) {
        Ok(Some((h, f))) => (Some(h), Some(f)),
        _ => (None, None),
      },
      Some(_) => (None, None),
    };
    let meta = SkyImageMeta::from_facts(&fits_metadata::facts_of_header(&header, cards, nx, ny));
    for (present, bit) in [
      (meta.date_obs_tdb_us.is_some(), SKY_FLAG_DATE_OBS),
      (meta.exptime_s.is_some(), SKY_FLAG_EXPTIME),
      (meta.site_lat_deg.is_some(), SKY_FLAG_SITE_LAT),
      (meta.site_lon_deg.is_some(), SKY_FLAG_SITE_LON),
      (meta.site_elev_m.is_some(), SKY_FLAG_SITE_ELEV),
      (meta.timesys_assumed, SKY_FLAG_TIMESYS_ASSUMED),
    ] {
      if present {
        flags |= bit;
      }
    }
    let black = percentile(&data, SKY_IMAGE_BLACK_PERCENTILE).unwrap_or(0.0);
    let mut white = percentile(&data, SKY_IMAGE_WHITE_PERCENTILE).unwrap_or(1.0);
    if !(white > black) {
      white = black + 1.0;
    }
    Self {
      nx,
      ny,
      data,
      header,
      wcs,
      wcs_error,
      hips,
      hips_facts,
      meta,
      flags,
      black,
      white,
    }
  }

  /// Decodes an image from its bytes; `ext` picks the decoder (`fits` / `fit` / `fts`, `png`,
  /// `jpg` / `jpeg`).
  pub fn from_bytes(ext: &str, bytes: &[u8], sidecar: Option<&[u8]>) -> Result<Self, String> {
    let sidecar = sidecar.map(FitsHeader::parse_text);
    match ext {
      "fits" | "fit" | "fts" | "fz" => {
        let (img, meta) = fits_metadata::read_fits(bytes)?;
        Ok(Self::from_parts_cards(img.nx, img.ny, img.data, img.header, Some(&meta.cards), sidecar, false))
      }
      _ => {
        let (nx, ny, data) = decode_raster(ext, bytes)?;
        Ok(Self::from_parts(
          nx,
          ny,
          data,
          FitsHeader::default(),
          sidecar,
          true,
        ))
      }
    }
  }

  /// Loads `path`; `sidecar` is a WCS header file (PNG / JPEG default: `<stem>.wcs` when it
  /// exists).
  pub fn load(path: &str, sidecar: Option<&str>) -> Result<Self, String> {
    let ext = ext_of(path);
    let bytes = read_file(path)?;
    let side = match sidecar {
      Some(s) if !s.is_empty() => Some(read_file(s)?),
      _ if !matches!(ext.as_str(), "fits" | "fit" | "fts" | "fz") => {
        let stem = path.rsplit_once('.').map(|(s, _)| s).unwrap_or(path);
        read_file(&alloc::format!("{stem}.wcs")).ok()
      }
      _ => None,
    };
    let mut img = Self::from_bytes(&ext, &bytes, side.as_deref())?;
    if img.wcs.is_none()
      && img.hips.is_none()
      && let Ok(Some((h, f))) = hips::load_mapping(Some(path), &img.header, img.nx, img.ny)
    {
      img.hips = Some(h);
      img.hips_facts = Some(f);
    }
    Ok(img)
  }
}

// ─── GPU data ────────────────────────────────────────────────────────────────

/// One registration mesh vertex: a point of the tangent plane in ecliptic J2000 (a direction,
/// not normalised: drawn at infinity, `w = 0`) and its texture coordinate.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyImageVertex {
  pub pos: [f32; 3],
  pub uv: [f32; 2],
}

/// IEEE half float bits of `v` (round to nearest even; NaN stays NaN, overflow → ±inf).
pub fn f32_to_f16_bits(v: f32) -> u16 {
  let b = v.to_bits();
  let sign = ((b >> 16) & 0x8000) as u16;
  let exp = ((b >> 23) & 0xff) as i32;
  let man = b & 0x7f_ffff;
  if exp == 0xff {
    return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
  }
  let e = exp - 127 + 15;
  if e >= 0x1f {
    return sign | 0x7c00;
  }
  if e <= 0 {
    if e < -10 {
      return sign;
    }
    // subnormal: shift the mantissa with its implicit bit
    let m = man | 0x80_0000;
    let shift = (14 - e) as u32;
    let half = 1u32 << (shift - 1);
    let rem = m & ((1u32 << shift) - 1);
    let mut r = m >> shift;
    if rem > half || (rem == half && (r & 1) == 1) {
      r += 1;
    }
    return sign | r as u16;
  }
  let mut r = ((e as u32) << 10) | (man >> 13);
  let rem = man & 0x1fff;
  if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
    r += 1; // may carry into the exponent: still correct (up to inf)
  }
  sign | r as u16
}

/// Upload form of a [`SkyImage`]: the normalised half-float texture (box-averaged to at most
/// [`SKY_IMAGE_MAX_TEXTURE`] per side) and the registration mesh.
#[derive(Debug, Clone)]
pub struct SkyImageGpuData {
  pub tex_width: u32,
  pub tex_height: u32,
  /// `R16_SFLOAT` bits, `(value − black) / (white − black)`, NaN for blank pixels
  pub texels: Vec<u16>,
  /// triangle list
  pub vertices: Vec<SkyImageVertex>,
}

/// Triangle list of the registration mesh over the image (edges at pixel coordinates 0.5 and
/// `n + 0.5`). `span` = the image size in pixels the texture's `[0, 1]` covers (the image size,
/// or the box-averaged texture's size in source pixels). A plain TAN gets the exact `n × n`
/// tangent-plane mesh; any other WCS a mesh of unit directions with at least `n` vertices per
/// side and cells of at most [`SKY_IMAGE_MAX_CELL_DEG`] of sky (at most [`SKY_IMAGE_GRID_MAX`]
/// per side), cells touching the outside of the projection dropped.
pub fn grid_vertices(wcs: &Wcs, nx: u32, ny: u32, span: [f64; 2], n: usize) -> Vec<SkyImageVertex> {
  let n = n.max(2);
  if wcs.is_plain_tan() {
    let f = |p1: f64, p2: f64| Some(wcs.pixel_to_tangent_point(p1, p2));
    return mesh_of(f, nx, ny, span, n);
  }
  let extent = wcs.angular_extent_deg(nx, ny, 9).unwrap_or(0.0);
  let n = ((extent / SKY_IMAGE_MAX_CELL_DEG).ceil() as usize + 1).clamp(n, SKY_IMAGE_GRID_MAX);
  mesh_of(|p1, p2| wcs.pixel_to_icrs(p1, p2), nx, ny, span, n)
}

/// Largest sky cell of a non-TAN registration mesh (degrees).
pub const SKY_IMAGE_MAX_CELL_DEG: f64 = 0.75;
/// Most vertices per side of a registration mesh.
pub const SKY_IMAGE_GRID_MAX: usize = 256;

/// The mesh of any pixel → ICRS direction function (`None` = outside the image's sky): vertices
/// converted to ecliptic J2000, cells with an undefined corner dropped.
pub fn mesh_of(
  dir: impl Fn(f64, f64) -> Option<V3>,
  nx: u32,
  ny: u32,
  span: [f64; 2],
  n: usize,
) -> Vec<SkyImageVertex> {
  let n = n.max(2);
  let mut grid: Vec<Option<SkyImageVertex>> = Vec::with_capacity(n * n);
  for j in 0..n {
    for i in 0..n {
      let p1 = 0.5 + nx as f64 * i as f64 / (n - 1) as f64;
      let p2 = 0.5 + ny as f64 * j as f64 / (n - 1) as f64;
      let v = dir(p1, p2).filter(|d| d.iter().all(|c| c.is_finite())).map(|d| {
        let d = equatorial_to_ecliptic_j2000(d);
        SkyImageVertex {
          pos: [d[0] as f32, d[1] as f32, d[2] as f32],
          uv: [((p1 - 0.5) / span[0]) as f32, ((p2 - 0.5) / span[1]) as f32],
        }
      });
      grid.push(v);
    }
  }
  let mut tris = Vec::with_capacity((n - 1) * (n - 1) * 6);
  for j in 0..n - 1 {
    for i in 0..n - 1 {
      if let (Some(a), Some(b), Some(c), Some(d)) =
        (grid[j * n + i], grid[j * n + i + 1], grid[(j + 1) * n + i], grid[(j + 1) * n + i + 1])
      {
        tris.extend_from_slice(&[a, b, d, a, d, c]);
      }
    }
  }
  tris
}

impl SkyImage {
  /// Integer box factor for the upload.
  pub fn texture_factor(&self) -> u32 {
    self.nx.max(self.ny).div_ceil(SKY_IMAGE_MAX_TEXTURE).max(1)
  }

  /// The upload form; `None` without a WCS (nothing to register).
  pub fn gpu_data(&self) -> Option<SkyImageGpuData> {
    if self.wcs.is_none() && self.hips.is_none() {
      return None;
    }
    let f = self.texture_factor();
    let (tw, th) = (self.nx.div_ceil(f), self.ny.div_ceil(f));
    let scale = 1.0 / (self.white - self.black);
    let mut texels = Vec::with_capacity((tw * th) as usize);
    for ty in 0..th {
      for tx in 0..tw {
        let (mut sum, mut cnt, mut blank) = (0.0f64, 0u32, 0u32);
        for y in ty * f..((ty + 1) * f).min(self.ny) {
          for x in tx * f..((tx + 1) * f).min(self.nx) {
            let v = self.data[(y * self.nx + x) as usize];
            if v.is_finite() {
              sum += v as f64;
              cnt += 1;
            } else {
              blank += 1;
            }
          }
        }
        let t = if cnt == 0 || blank > cnt {
          f32::NAN
        } else {
          (((sum / cnt as f64) as f32 - self.black) * scale).clamp(-1.0, 1000.0)
        };
        texels.push(f32_to_f16_bits(t));
      }
    }
    let span = [(tw * f) as f64, (th * f) as f64];
    Some(SkyImageGpuData {
      tex_width: tw,
      tex_height: th,
      texels,
      vertices: match (&self.wcs, &self.hips) {
        (Some(wcs), _) => grid_vertices(wcs, self.nx, self.ny, span, SKY_IMAGE_GRID),
        (None, Some(h)) => h.mesh_vertices(span, SKY_IMAGE_GRID),
        (None, None) => Vec::new(),
      },
    })
  }

  /// Ecliptic J2000 unit direction of `CRVAL` (the "lock to image" aim).
  pub fn crval_direction_ecliptic(&self) -> Option<[f64; 3]> {
    if let Some(w) = self.wcs.as_ref() {
      return Some(equatorial_to_ecliptic_j2000(dir_of_ra_dec(w.crval_deg[0], w.crval_deg[1])));
    }
    self.hips.as_ref().map(|h| h.center_direction_ecliptic())
  }

  /// The header facts of the WCS (`None` without one).
  pub fn wcs_facts(&self) -> Option<WcsFacts> {
    self.wcs.as_ref().map(|w| w.facts())
  }
}

// ─── store ───────────────────────────────────────────────────────────────────

/// Display parameters of a scene's sky image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyImageParams {
  /// `[0, 1]`
  pub opacity: f32,
  /// asinh softening: `y = asinh(s·x) / asinh(s)`, `s ≤ 1e-3` = linear
  pub stretch: f32,
  pub visible: bool,
}

impl Default for SkyImageParams {
  fn default() -> Self {
    Self {
      opacity: 0.7,
      stretch: 10.0,
      visible: true,
    }
  }
}

/// A scene's entry of the store.
#[derive(Debug, Clone)]
pub struct SkyImageEntry {
  /// changes on every load (the GPU side re-uploads)
  pub generation: u64,
  pub data: Arc<SkyImageGpuData>,
  pub params: SkyImageParams,
}

static STORE: spin::Mutex<BTreeMap<usize, SkyImageEntry>> = spin::Mutex::new(BTreeMap::new());
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// Store key of a scene: its `SceneContext` address (the `Arc` the scene map and the render
/// thread's `RenderFrame::scene` share).
pub fn scene_key<T>(scene: &Arc<T>) -> usize {
  Arc::as_ptr(scene) as *const () as usize
}

/// Installs the image of scene `key` (replacing any); keeps the previous display parameters.
/// `false` (nothing stored) without a WCS.
pub fn store_insert(key: usize, image: &SkyImage) -> bool {
  let Some(data) = image.gpu_data() else {
    return false;
  };
  let generation = GENERATION.fetch_add(1, Ordering::Relaxed);
  let mut s = STORE.lock();
  let params = s.get(&key).map(|e| e.params).unwrap_or_default();
  s.insert(
    key,
    SkyImageEntry {
      generation,
      data: Arc::new(data),
      params,
    },
  );
  true
}

/// Sets the display parameters (clamped); `false` when scene `key` has no image.
pub fn store_set_params(key: usize, params: SkyImageParams) -> bool {
  let mut s = STORE.lock();
  let Some(e) = s.get_mut(&key) else {
    return false;
  };
  e.params = SkyImageParams {
    opacity: if params.opacity.is_finite() {
      params.opacity.clamp(0.0, 1.0)
    } else {
      0.0
    },
    stretch: if params.stretch.is_finite() {
      params.stretch.clamp(0.0, 1.0e4)
    } else {
      0.0
    },
    visible: params.visible,
  };
  true
}

/// Removes the image of scene `key`; `false` when there was none.
pub fn store_remove(key: usize) -> bool {
  STORE.lock().remove(&key).is_some()
}

/// The entry of scene `key`.
pub fn store_get(key: usize) -> Option<SkyImageEntry> {
  STORE.lock().get(&key).cloned()
}
