//! Every card of a FITS primary header (in file order, with comments) and the observation facts
//! a UI shows for an imported FITS texture (`asset_library`, `avkSimulationContext_getFitsMetadataJson`).
//!
//! ## Facts (all optional)
//! - **object / telescope / instrument / filter**: `OBJECT`, `TELESCOP`, `INSTRUME`, `FILTER`
//!   (else `FILTER1`, else `FILTNAME`).
//! - **exposure**: `EXPTIME`, else `EXPOSURE` (seconds).
//! - **date**: the mid-exposure instant. `DATE-AVG` / `MJD-AVG` are mid-exposure already. Else the
//!   start from `DATE-OBS` (`T` or a space between date and time; a date-only value takes its
//!   time from `TIME-OBS` / `UTSTART` / `UT`), else `MJD-OBS`, else `MJD`, else `JD`, plus half the
//!   exposure. A `DATE-OBS` whose comment says "averaged" / "mid" is taken as the mid-exposure
//!   instant. Time scale from `TIMESYS` (UTC when absent). Reported as an ISO UTC string and as
//!   TDB `(centuries, nanoseconds)` since J2000 TDB (`hifitime::Duration::to_parts` of
//!   `to_tdb_duration`, what `CTime` / the epoch FFI take) and TDB microseconds.
//! - **pixel scale** (arcsec / px): from the `CDi_j` matrix, else `CDELTi` (with `PCi_j` /
//!   `CROTA2`), else `PIXSCALE`, else `SECPIX`. Per axis: the column norms of the matrix.
//! - **field of view**: `NAXIS1 · scale_x × NAXIS2 · scale_y` (arcmin).
//! - **WCS**: `CRVAL1/2` (when `CTYPE1` is celestial), the image centre via the TAN WCS
//!   (`Wcs::from_header`) when it is accepted, the telescope pointing from `RA`/`DEC` (or
//!   `OBJCTRA`/`OBJCTDEC`; a sexagesimal RA is in hours).
//! - **rotation**: position angle of the image +Y axis, degrees east of north,
//!   `atan2(CD1_2, CD2_2)`, in `(-180, 180]`; `mirrored` when `det CD > 0` (east to the right
//!   with north up).
//! - **site**: the first present of `SITELAT/SITELONG`, `LAT-OBS/LONG-OBS`, `OBSGEO-B/L`,
//!   `OBSGEO-X/Y/Z` (WGS84 geocentric metres → geodetic), `LATITUDE/LONGITUD`. Longitudes are
//!   returned **east positive**: a longitude card whose comment mentions "west" (and not "east")
//!   or whose string value ends in `W` is negated (`site_lon_west_convention`, and the raw value is
//!   kept in `site_lon_raw_deg`). Elevation: `SITEELEV`, `ALT-OBS`, `OBSGEO-H`, `HEIGHT`, else
//!   from `OBSGEO-X/Y/Z`.

use alloc::{
  format,
  string::{String, ToString},
  vec::Vec,
};

use super::{FitsHeader, Wcs, parse_sexagesimal};

const DEG: f64 = core::f64::consts::PI / 180.0;

/// One header card: keyword, value text (strings unquoted, `''` unescaped, trailing blanks
/// dropped; commentary cards: the text after column 8) and comment (after `/`, trimmed).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitsCard {
  pub key: String,
  pub value: String,
  pub comment: String,
}

fn ascii_lossy(b: &[u8]) -> String {
  b.iter().map(|&c| if (0x20..0x7f).contains(&c) { c as char } else { '?' }).collect()
}

/// Parses one 80-column card; `None` for an all-blank card.
pub fn parse_full_card(card: &[u8]) -> Option<FitsCard> {
  let card = ascii_lossy(card);
  let mut key = card.get(..8.min(card.len())).unwrap_or("").trim_end().to_string();
  let mut rest_start = 8usize;
  // HIERARCH long keywords: `HIERARCH ESO DET X = value / comment`
  if key == "HIERARCH" {
    if let Some(eq) = card.find('=') {
      key = card[8..eq].trim().to_string();
      rest_start = eq; // so the value indicator test below sees '='
    }
  }
  let has_value = card.get(rest_start..rest_start + 1) == Some("=")
    && (key != "COMMENT" && key != "HISTORY" && !key.is_empty());
  if !has_value {
    let text = card.get(8..).unwrap_or("").trim().to_string();
    if key.is_empty() && text.is_empty() {
      return None;
    }
    return Some(FitsCard { key, value: text, comment: String::new() });
  }
  let raw = card.get(rest_start + 1..).unwrap_or("").trim_start();
  let (value, after) = if let Some(rest) = raw.strip_prefix('\'') {
    let mut out = String::new();
    let mut consumed = rest.len();
    let mut it = rest.char_indices().peekable();
    while let Some((i, c)) = it.next() {
      if c == '\'' {
        if matches!(it.peek(), Some((_, '\''))) {
          out.push('\'');
          it.next();
        } else {
          consumed = i + 1;
          break;
        }
      } else {
        out.push(c);
      }
    }
    (out.trim_end().to_string(), &rest[consumed.min(rest.len())..])
  } else {
    match raw.find('/') {
      Some(i) => (raw[..i].trim().to_string(), &raw[i..]),
      None => (raw.trim().to_string(), ""),
    }
  };
  let comment = after
    .trim_start()
    .strip_prefix('/')
    .map(|c| c.trim().to_string())
    .unwrap_or_default();
  Some(FitsCard { key, value, comment })
}

/// Every card of the header at `bytes[0]` up to `END` (or the end of `bytes`), in order.
pub fn parse_cards(bytes: &[u8]) -> Vec<FitsCard> {
  let mut out = Vec::new();
  for card in bytes.chunks_exact(80) {
    if card.starts_with(b"END") && card[3..].iter().all(|&b| b == b' ') {
      break;
    }
    if let Some(c) = parse_full_card(card) {
      out.push(c);
    }
  }
  out
}

/// Derived observation facts (see the module doc); `None` = not in the header.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitsFacts {
  pub object: Option<String>,
  pub telescope: Option<String>,
  pub instrument: Option<String>,
  pub filter: Option<String>,
  pub exposure_s: Option<f64>,
  /// raw value of the date card used
  pub date_obs: Option<String>,
  /// keyword the date came from (`DATE-AVG`, `MJD-AVG`, `DATE-OBS`, `MJD-OBS`, `MJD`, `JD`)
  pub date_source: Option<String>,
  /// the card is the mid-exposure instant (no half exposure added)
  pub date_is_mid: bool,
  /// `YYYY-MM-DDTHH:MM:SS.sssZ`
  pub date_mid_utc: Option<String>,
  pub date_mid_tdb_centuries: Option<i16>,
  pub date_mid_tdb_nanoseconds: Option<u64>,
  /// TDB microseconds since J2000 TDB (`SkyImageMeta::date_obs_tdb_us` scale)
  pub date_mid_tdb_us: Option<i64>,
  /// no `TIMESYS`: UTC assumed
  pub timesys_assumed: bool,
  pub pixel_scale_arcsec: Option<f64>,
  pub pixel_scale_x_arcsec: Option<f64>,
  pub pixel_scale_y_arcsec: Option<f64>,
  /// `CD`, `CDELT`, `PIXSCALE`, `SECPIX`
  pub pixel_scale_source: Option<String>,
  pub fov_width_arcmin: Option<f64>,
  pub fov_height_arcmin: Option<f64>,
  pub crval_ra_deg: Option<f64>,
  pub crval_dec_deg: Option<f64>,
  pub center_ra_deg: Option<f64>,
  pub center_dec_deg: Option<f64>,
  pub pointing_ra_deg: Option<f64>,
  pub pointing_dec_deg: Option<f64>,
  pub radesys: Option<String>,
  pub equinox: Option<f64>,
  pub wcs_error: Option<String>,
  /// position angle of image +Y, degrees east of north
  pub rotation_deg: Option<f64>,
  pub mirrored: Option<bool>,
  pub site_lat_deg: Option<f64>,
  /// east positive
  pub site_lon_deg: Option<f64>,
  pub site_elev_m: Option<f64>,
  /// e.g. `LATITUDE/LONGITUD`
  pub site_source: Option<String>,
  /// the longitude was negated (comment says west / value suffixed `W`)
  pub site_lon_west_convention: bool,
  /// longitude as written in the card
  pub site_lon_raw_deg: Option<f64>,
  /// the header's `LST` at the observation start decided the longitude's sign (it matched one
  /// sign within 3° and not the other), overriding the west/east wording of the card
  pub site_lon_from_lst: bool,
}

/// Header cards and facts of a FITS image.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FitsMetadata {
  pub naxis1: u32,
  pub naxis2: u32,
  pub bitpix: i32,
  pub cards: Vec<FitsCard>,
  pub facts: FitsFacts,
}

/// A [`FitsHeader`] of `cards` (first occurrence of a keyword wins, commentary cards skipped).
pub fn header_of_cards(cards: &[FitsCard]) -> FitsHeader {
  let mut h = FitsHeader::default();
  for c in cards {
    if c.key.is_empty() || c.key == "COMMENT" || c.key == "HISTORY" {
      continue;
    }
    if h.get(&c.key).is_none() {
      h.cards.push((c.key.clone(), c.value.clone()));
    }
  }
  h
}

/// `YYYY-MM-DD[(T| )HH[:MM[:SS[.fff]]]]` (optional trailing `Z` / ` UTC`).
pub fn parse_datetime(s: &str) -> Option<(i32, u8, u8, u8, u8, u8, u32)> {
  let s = s.trim();
  let s = s.strip_suffix("UTC").unwrap_or(s).trim_end();
  let s = s.strip_suffix('Z').unwrap_or(s);
  let (date, time) = match s.find(['T', ' ']) {
    Some(i) => (&s[..i], Some(s[i + 1..].trim_start_matches(['T', ' ']).trim())),
    None => (s, None),
  };
  let mut d = date.split('-');
  let y = d.next()?.trim().parse::<i32>().ok()?;
  let m = d.next()?.trim().parse::<u8>().ok()?;
  let day = d.next()?.trim().parse::<u8>().ok()?;
  if d.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&day) {
    return None;
  }
  let (mut hh, mut mm, mut ss, mut ns) = (0u8, 0u8, 0u8, 0u32);
  if let Some(t) = time.filter(|t| !t.is_empty()) {
    let mut t = t.split(':');
    hh = t.next()?.trim().parse().ok()?;
    mm = t.next().map(|v| v.trim().parse()).transpose().ok()?.unwrap_or(0);
    if let Some(sec) = t.next() {
      let sec = sec.trim();
      let (whole, frac) = sec.split_once('.').unwrap_or((sec, ""));
      ss = whole.parse().ok()?;
      if !frac.is_empty() {
        let digits: String = frac.chars().take(9).collect();
        let scale = 10u32.pow(9 - digits.len() as u32);
        ns = digits.parse::<u32>().ok()? * scale;
      }
    }
    if t.next().is_some() {
      return None;
    }
  }
  Some((y, m, day, hh, mm, ss, ns))
}

/// WGS84 geocentric `(x, y, z)` metres → geodetic `(lat°, lon° east, height m)`.
pub fn geodetic_of_geocentric(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
  const A: f64 = 6_378_137.0;
  const F: f64 = 1.0 / 298.257_223_563;
  let e2 = F * (2.0 - F);
  let lon = y.atan2(x);
  let p = (x * x + y * y).sqrt();
  let mut lat = z.atan2(p * (1.0 - e2));
  let mut h = 0.0;
  for _ in 0..10 {
    let s = lat.sin();
    let n = A / (1.0 - e2 * s * s).sqrt();
    h = if lat.cos().abs() > 1e-12 { p / lat.cos() - n } else { z.abs() - n * (1.0 - e2) };
    lat = z.atan2(p * (1.0 - e2 * n / (n + h)));
  }
  (lat / DEG, lon / DEG, h)
}

fn wrap_180(d: f64) -> f64 {
  let mut d = d % 360.0;
  if d > 180.0 {
    d -= 360.0;
  } else if d <= -180.0 {
    d += 360.0;
  }
  d
}

fn utc_string(e: hifitime::Epoch) -> String {
  let (y, mo, d, h, mi, s, ns) = e.to_gregorian_utc();
  format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{:03}Z", ns / 1_000_000)
}

impl FitsMetadata {
  /// From the cards of a header (`naxis` / `bitpix` from them).
  pub fn from_cards(cards: Vec<FitsCard>) -> Self {
    let h = header_of_cards(&cards);
    let naxis1 = h.get_i64("NAXIS1").unwrap_or(0).max(0) as u32;
    let naxis2 = h.get_i64("NAXIS2").unwrap_or(0).max(0) as u32;
    let bitpix = h.get_i64("BITPIX").unwrap_or(0) as i32;
    let facts = facts_of(&cards, &h, naxis1, naxis2);
    Self { naxis1, naxis2, bitpix, cards, facts }
  }

  /// From the bytes of a FITS file (its primary header).
  pub fn from_fits_bytes(bytes: &[u8]) -> Self {
    Self::from_cards(parse_cards(bytes))
  }

  /// The metadata as JSON (camelCase keys, `null` for missing facts), see
  /// `docs/fits_ffi_contract.md`.
  pub fn to_json(&self) -> String {
    let mut j = String::with_capacity(256 + self.cards.len() * 96);
    j.push_str(&format!(
      "{{\"naxis1\":{},\"naxis2\":{},\"bitpix\":{},\"cards\":[",
      self.naxis1, self.naxis2, self.bitpix
    ));
    for (i, c) in self.cards.iter().enumerate() {
      if i > 0 {
        j.push(',');
      }
      j.push_str("{\"key\":");
      json_str(&mut j, &c.key);
      j.push_str(",\"value\":");
      json_str(&mut j, &c.value);
      j.push_str(",\"comment\":");
      json_str(&mut j, &c.comment);
      j.push('}');
    }
    j.push_str("],\"facts\":");
    self.facts.write_json(&mut j);
    j.push('}');
    j
  }
}

/// Appends `s` as a JSON string literal.
pub fn json_str(out: &mut String, s: &str) {
  out.push('"');
  for c in s.chars() {
    match c {
      '"' => out.push_str("\\\""),
      '\\' => out.push_str("\\\\"),
      '\n' => out.push_str("\\n"),
      '\r' => out.push_str("\\r"),
      '\t' => out.push_str("\\t"),
      c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
      c => out.push(c),
    }
  }
  out.push('"');
}

/// A finite number as JSON, else `null`.
pub fn json_f64(v: Option<f64>) -> String {
  match v {
    Some(x) if x.is_finite() => format!("{x:?}"),
    _ => "null".to_string(),
  }
}

impl FitsFacts {
  fn write_json(&self, j: &mut String) {
    let mut first = true;
    let mut key = |j: &mut String, k: &str| {
      if !first {
        j.push(',');
      }
      first = false;
      json_str(j, k);
      j.push(':');
    };
    let opt_s = |j: &mut String, v: &Option<String>| match v {
      Some(s) => json_str(j, s),
      None => j.push_str("null"),
    };
    j.push('{');
    for (k, v) in [
      ("object", &self.object),
      ("telescope", &self.telescope),
      ("instrument", &self.instrument),
      ("filter", &self.filter),
    ] {
      key(j, k);
      opt_s(j, v);
    }
    key(j, "exposureS");
    j.push_str(&json_f64(self.exposure_s));
    key(j, "dateObs");
    opt_s(j, &self.date_obs);
    key(j, "dateSource");
    opt_s(j, &self.date_source);
    key(j, "dateIsMid");
    j.push_str(if self.date_is_mid { "true" } else { "false" });
    key(j, "dateMidUtc");
    opt_s(j, &self.date_mid_utc);
    key(j, "dateMidTdbCenturies");
    j.push_str(&self.date_mid_tdb_centuries.map_or("null".to_string(), |v| format!("{v}")));
    key(j, "dateMidTdbNanoseconds");
    j.push_str(&self.date_mid_tdb_nanoseconds.map_or("null".to_string(), |v| format!("{v}")));
    key(j, "dateMidTdbUs");
    j.push_str(&self.date_mid_tdb_us.map_or("null".to_string(), |v| format!("{v}")));
    key(j, "timesysAssumed");
    j.push_str(if self.timesys_assumed { "true" } else { "false" });
    for (k, v) in [
      ("pixelScaleArcsec", self.pixel_scale_arcsec),
      ("pixelScaleXArcsec", self.pixel_scale_x_arcsec),
      ("pixelScaleYArcsec", self.pixel_scale_y_arcsec),
    ] {
      key(j, k);
      j.push_str(&json_f64(v));
    }
    key(j, "pixelScaleSource");
    opt_s(j, &self.pixel_scale_source);
    for (k, v) in [
      ("fovWidthArcmin", self.fov_width_arcmin),
      ("fovHeightArcmin", self.fov_height_arcmin),
      ("crvalRaDeg", self.crval_ra_deg),
      ("crvalDecDeg", self.crval_dec_deg),
      ("centerRaDeg", self.center_ra_deg),
      ("centerDecDeg", self.center_dec_deg),
      ("pointingRaDeg", self.pointing_ra_deg),
      ("pointingDecDeg", self.pointing_dec_deg),
    ] {
      key(j, k);
      j.push_str(&json_f64(v));
    }
    key(j, "radesys");
    opt_s(j, &self.radesys);
    key(j, "equinox");
    j.push_str(&json_f64(self.equinox));
    key(j, "wcsError");
    opt_s(j, &self.wcs_error);
    key(j, "rotationDeg");
    j.push_str(&json_f64(self.rotation_deg));
    key(j, "mirrored");
    j.push_str(match self.mirrored {
      Some(true) => "true",
      Some(false) => "false",
      None => "null",
    });
    for (k, v) in [
      ("siteLatDeg", self.site_lat_deg),
      ("siteLonDeg", self.site_lon_deg),
      ("siteElevM", self.site_elev_m),
    ] {
      key(j, k);
      j.push_str(&json_f64(v));
    }
    key(j, "siteSource");
    opt_s(j, &self.site_source);
    key(j, "siteLonWestConvention");
    j.push_str(if self.site_lon_west_convention { "true" } else { "false" });
    key(j, "siteLonRawDeg");
    j.push_str(&json_f64(self.site_lon_raw_deg));
    key(j, "siteLonFromLst");
    j.push_str(if self.site_lon_from_lst { "true" } else { "false" });
    j.push('}');
  }
}

fn card<'a>(cards: &'a [FitsCard], key: &str) -> Option<&'a FitsCard> {
  cards.iter().find(|c| c.key == key)
}

fn non_empty(h: &FitsHeader, key: &str) -> Option<String> {
  h.get(key).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// Linear part of the WCS (degrees per pixel) and the source keyword, like [`Wcs::from_header`].
fn wcs_matrix(h: &FitsHeader) -> Option<([[f64; 2]; 2], &'static str)> {
  let has_cd = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"].iter().any(|k| h.get(k).is_some());
  if has_cd {
    let g = |k: &str| h.get_f64(k).unwrap_or(0.0);
    return Some(([[g("CD1_1"), g("CD1_2")], [g("CD2_1"), g("CD2_2")]], "CD"));
  }
  let (d1, d2) = (h.get_f64("CDELT1")?, h.get_f64("CDELT2")?);
  let has_pc = ["PC1_1", "PC1_2", "PC2_1", "PC2_2"].iter().any(|k| h.get(k).is_some());
  let m = if has_pc {
    let g = |k: &str, d: f64| h.get_f64(k).unwrap_or(d);
    [
      [d1 * g("PC1_1", 1.0), d1 * g("PC1_2", 0.0)],
      [d2 * g("PC2_1", 0.0), d2 * g("PC2_2", 1.0)],
    ]
  } else {
    let rho = h.get_f64("CROTA2").unwrap_or(0.0) * DEG;
    let (s, c) = (rho.sin(), rho.cos());
    [[d1 * c, -d2 * s], [d1 * s, d2 * c]]
  };
  Some((m, "CDELT"))
}

/// Greenwich mean sidereal time at `e` (UTC taken as UT1, < 1 s), degrees in [0, 360): the
/// IAU 1982 linear term, good to well under 0.01° over decades.
fn gmst_deg(e: hifitime::Epoch) -> f64 {
  let d = e.to_jde_utc_days() - 2_451_545.0;
  (280.460_618_37 + 360.985_647_366_29 * d).rem_euclid(360.0)
}

/// A longitude card → `(east positive degrees, raw degrees, west convention)`.
fn longitude(cards: &[FitsCard], h: &FitsHeader, key: &str) -> Option<(f64, f64, bool)> {
  let c = card(cards, key)?;
  let v = c.value.trim();
  // `17.88 W` / `17:52:45W` (a plain number never ends in a letter)
  let (raw, suffix_west) = match h.get_degrees(key) {
    Some(d) => (d, None),
    None => {
      let west = match v.chars().last()? {
        'W' | 'w' => true,
        'E' | 'e' => false,
        _ => return None,
      };
      let body = v[..v.len() - 1].trim();
      let d = body.parse::<f64>().ok().or_else(|| parse_sexagesimal(body))?;
      (d, Some(west))
    }
  };
  let comment = c.comment.to_ascii_lowercase();
  let west = suffix_west.unwrap_or_else(|| comment.contains("west") && !comment.contains("east"));
  let east = if west { -raw } else { raw };
  Some((wrap_180(east), raw, west))
}

fn facts_of(cards: &[FitsCard], h: &FitsHeader, nx: u32, ny: u32) -> FitsFacts {
  use hifitime::{Duration, Epoch, TimeScale};
  let mut f = FitsFacts {
    object: non_empty(h, "OBJECT"),
    telescope: non_empty(h, "TELESCOP"),
    instrument: non_empty(h, "INSTRUME"),
    filter: non_empty(h, "FILTER")
      .or_else(|| non_empty(h, "FILTER1"))
      .or_else(|| non_empty(h, "FILTNAME")),
    exposure_s: h.get_f64("EXPTIME").or_else(|| h.get_f64("EXPOSURE")),
    ..Default::default()
  };

  // ── date ──
  let timesys = h.get("TIMESYS").map(|s| s.trim().to_ascii_uppercase());
  let scale = match timesys.as_deref() {
    Some("TT") | Some("TDT") => TimeScale::TT,
    Some("TAI") | Some("IAT") => TimeScale::TAI,
    Some("TDB") => TimeScale::TDB,
    Some("GPS") => TimeScale::GPST,
    _ => TimeScale::UTC,
  };
  let iso = |v: &str| -> Option<Epoch> {
    let mut v = v.trim().to_string();
    if !v.contains(['T', ' ']) || v.ends_with('T') {
      if let Some(t) = h.get("TIME-OBS").or_else(|| h.get("UTSTART")).or_else(|| h.get("UT")) {
        v = format!("{}T{}", v.trim_end_matches('T'), t.trim());
      }
    }
    let (y, mo, d, hh, mi, ss, ns) = parse_datetime(&v)?;
    Epoch::maybe_from_gregorian(y, mo, d, hh, mi, ss, ns, scale).ok()
  };
  let mjd = |k: &str| h.get_f64(k).map(|m| Epoch::from_mjd_in_time_scale(m, scale));
  let mut found: Option<(Epoch, &str, bool)> = None;
  if let Some(e) = h.get("DATE-AVG").and_then(iso) {
    found = Some((e, "DATE-AVG", true));
  } else if let Some(e) = mjd("MJD-AVG") {
    found = Some((e, "MJD-AVG", true));
  } else if let Some(e) = h.get("DATE-OBS").and_then(iso) {
    let c = card(cards, "DATE-OBS").map(|c| c.comment.to_ascii_lowercase()).unwrap_or_default();
    let mid = c.contains("averag") || c.contains("mid");
    found = Some((e, "DATE-OBS", mid));
  } else if let Some(e) = mjd("MJD-OBS") {
    found = Some((e, "MJD-OBS", false));
  } else if let Some(e) = mjd("MJD") {
    found = Some((e, "MJD", false));
  } else if let Some(jd) = h.get_f64("JD").filter(|jd| *jd > 2_400_000.5) {
    found = Some((Epoch::from_mjd_in_time_scale(jd - 2_400_000.5, scale), "JD", false));
  }
  let lst_epoch = found.map(|(start, _, _)| start);
  if let Some((start, src, is_mid)) = found {
    let mid = if is_mid {
      start
    } else {
      match f.exposure_s {
        Some(t) if t.is_finite() && t > 0.0 => start + Duration::from_seconds(t * 0.5),
        _ => start,
      }
    };
    f.date_obs = h.get(src).map(|s| s.trim().to_string());
    f.date_source = Some(src.to_string());
    f.date_is_mid = is_mid;
    f.date_mid_utc = Some(utc_string(mid));
    let tdb = mid.to_tdb_duration();
    let (c, ns) = tdb.to_parts();
    f.date_mid_tdb_centuries = Some(c);
    f.date_mid_tdb_nanoseconds = Some(ns);
    f.date_mid_tdb_us = Some((tdb.total_nanoseconds() / 1000) as i64);
    f.timesys_assumed = timesys.is_none();
  }

  // ── scale, rotation, field ──
  let matrix = wcs_matrix(h);
  if let Some((m, src)) = matrix {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.is_finite() && det != 0.0 {
      let sx = (m[0][0] * m[0][0] + m[1][0] * m[1][0]).sqrt() * 3600.0;
      let sy = (m[0][1] * m[0][1] + m[1][1] * m[1][1]).sqrt() * 3600.0;
      f.pixel_scale_arcsec = Some(det.abs().sqrt() * 3600.0);
      f.pixel_scale_x_arcsec = Some(sx);
      f.pixel_scale_y_arcsec = Some(sy);
      f.pixel_scale_source = Some(src.to_string());
      f.rotation_deg = Some(wrap_180(m[0][1].atan2(m[1][1]) / DEG));
      f.mirrored = Some(det > 0.0);
    }
  }
  if f.pixel_scale_arcsec.is_none() {
    for k in ["PIXSCALE", "SECPIX"] {
      if let Some(s) = h.get_f64(k).filter(|s| s.is_finite() && *s > 0.0) {
        f.pixel_scale_arcsec = Some(s);
        f.pixel_scale_x_arcsec = Some(s);
        f.pixel_scale_y_arcsec = Some(s);
        f.pixel_scale_source = Some(k.to_string());
        break;
      }
    }
  }
  if let (Some(sx), Some(sy)) = (f.pixel_scale_x_arcsec, f.pixel_scale_y_arcsec) {
    f.fov_width_arcmin = Some(nx as f64 * sx / 60.0);
    f.fov_height_arcmin = Some(ny as f64 * sy / 60.0);
  }

  // ── WCS / pointing ──
  let celestial = h.get("CTYPE1").map(|c| c.trim().starts_with("RA")).unwrap_or(false);
  if celestial {
    f.crval_ra_deg = h.get_f64("CRVAL1");
    f.crval_dec_deg = h.get_f64("CRVAL2");
  }
  match Wcs::from_header(h) {
    Ok(Some(w)) => {
      let (ra, dec) = w.pixel_to_radec_deg((nx as f64 + 1.0) * 0.5, (ny as f64 + 1.0) * 0.5);
      f.center_ra_deg = Some(ra);
      f.center_dec_deg = Some(dec);
    }
    Ok(None) => {}
    Err(e) => f.wcs_error = Some(e),
  }
  f.radesys = non_empty(h, "RADESYS").or_else(|| non_empty(h, "RADECSYS"));
  f.equinox = h.get_f64("EQUINOX").or_else(|| h.get_f64("EPOCH"));
  for (rk, dk) in [("RA", "DEC"), ("OBJCTRA", "OBJCTDEC")] {
    let (Some(r), Some(_)) = (h.get(rk), h.get(dk)) else {
      continue;
    };
    // a number is degrees, a sexagesimal string hours
    let ra = h.get_f64(rk).or_else(|| parse_sexagesimal(r).map(|hrs| hrs * 15.0));
    if let (Some(ra), Some(dec)) = (ra, h.get_degrees(dk)) {
      f.pointing_ra_deg = Some(ra);
      f.pointing_dec_deg = Some(dec);
      break;
    }
  }

  // ── site ──
  for (lat_k, lon_k) in [
    ("SITELAT", "SITELONG"),
    ("LAT-OBS", "LONG-OBS"),
    ("OBSGEO-B", "OBSGEO-L"),
    ("", ""),
    ("LATITUDE", "LONGITUD"),
  ] {
    if lat_k.is_empty() {
      let (Some(x), Some(y), Some(z)) =
        (h.get_f64("OBSGEO-X"), h.get_f64("OBSGEO-Y"), h.get_f64("OBSGEO-Z"))
      else {
        continue;
      };
      if !(x * x + y * y + z * z > 1.0e10) {
        continue;
      }
      let (lat, lon, hgt) = geodetic_of_geocentric(x, y, z);
      f.site_lat_deg = Some(lat);
      f.site_lon_deg = Some(lon);
      f.site_lon_raw_deg = Some(lon);
      f.site_elev_m = Some(hgt);
      f.site_source = Some("OBSGEO-X/Y/Z".to_string());
      break;
    }
    let lat = h.get_degrees(lat_k);
    let lon = longitude(cards, h, lon_k);
    if lat.is_none() && lon.is_none() {
      continue;
    }
    f.site_lat_deg = lat;
    if let Some((east, raw, west)) = lon {
      f.site_lon_deg = Some(east);
      f.site_lon_raw_deg = Some(raw);
      f.site_lon_west_convention = west;
    }
    f.site_source = Some(format!("{lat_k}/{lon_k}"));
    break;
  }
  // The header's local sidereal time is ground truth for the longitude's sign: a "degrees West"
  // comment on a value that is in fact east positive (the Liverpool Telescope writes
  // `LONGITUD = -17.8792 / [degrees West]` for La Palma, 17.88° W) would otherwise move the site
  // by 36°. Keep the sign whose LST at the observation start matches; with no LST, the wording.
  if let (Some(raw), Some(lst_h), Some(at)) = (
    f.site_lon_raw_deg,
    h.get("LST").and_then(parse_sexagesimal),
    card(cards, "UTSTART")
      .and_then(|c| parse_sexagesimal(c.value.trim().trim_matches('\'')))
      .zip(lst_epoch)
      .map(|(hrs, e)| {
        // UTSTART is the time of day of the date the observation started on
        let (y, mo, d, ..) = e.to_gregorian_utc();
        Epoch::from_gregorian_utc_at_midnight(y, mo, d) + Duration::from_seconds(hrs * 3600.0)
      })
      .or(lst_epoch),
  ) {
    let gmst = gmst_deg(at);
    let miss = |lon: f64| {
      let d = (gmst + lon - lst_h * 15.0).rem_euclid(360.0);
      d.min(360.0 - d)
    };
    let (keep, flip) = (miss(raw), miss(-raw));
    let east = if keep <= 3.0 && flip > 3.0 {
      Some(raw)
    } else if flip <= 3.0 && keep > 3.0 {
      Some(-raw)
    } else {
      None
    };
    if let Some(east) = east {
      f.site_lon_deg = Some(wrap_180(east));
      f.site_lon_west_convention = east != raw;
      f.site_lon_from_lst = true;
    }
  }
  if let Some(e) = ["SITEELEV", "ALT-OBS", "OBSGEO-H", "HEIGHT"].iter().find_map(|k| h.get_f64(k)) {
    f.site_elev_m = Some(e);
  }
  f
}
