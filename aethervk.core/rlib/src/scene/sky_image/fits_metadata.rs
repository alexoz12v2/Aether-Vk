//! Reading a FITS image ([`read_fits`]: the one entry point of `sky_image::parse_fits`,
//! `asset_library` and `FitsMetadata::from_fits_bytes`), every card of its header (in file order,
//! with comments) and the observation facts a UI shows for an imported FITS texture
//! (`asset_library`, `avkSimulationContext_getFitsMetadataJson`).
//!
//! ## Reading
//! With the `cfitsio` feature the file is opened in memory by CFITSIO (a whole-file compressed
//! `.fits.gz`, `.fits.bz2` or `.Z` is inflated first), which finds the
//! first HDU holding a real image: a `NAXIS = 0` primary, 1-D or zero-size images and tables are
//! skipped, an `IMAGE` extension or a tile-compressed `.fz` image (any of RICE_1, GZIP_1, GZIP_2,
//! HCOMPRESS_1, PLIO_1, NOCOMPRESS; quantised / dithered floats, ZBLANK) is read like a primary
//! image. A cube (`NAXIS > 2` with axes longer than 1) gives one 2-D plane: the first for
//! [`read_fits`], any by its flat index for [`read_fits_plane`] (the number of planes is reported:
//! `planes`, `plane`, and `plane_axes` for four or more axes). A data unit short of its last
//! 2880-byte block (some HiPS tiles) is padded. BZERO / BSCALE / BLANK are applied (a `BLANK`
//! written as a float, `-32768.0`, is read as its integer; undefined
//! pixels → NaN; unsigned integers through their BZERO), every BITPIX (8, 16, 32, 64, −32, −64)
//! comes out as `f32` (64-bit integers beyond 2²⁴ lose precision). The 80-column cards of the
//! image HDU are handed back as stored (a compressed HDU shows the header of the image it stands
//! for). A file with tables only, a random-groups file, an image over 2 GB or a side over 2²⁰
//! pixels is refused with a message saying so. Without the feature the pure-Rust parser of the
//! primary HDU runs (`sky_image::parse_fits_legacy`; a cube's plane is read by handing it the
//! header with the extra axes set to 1 and the data from that plane on). Card parsing, the typed
//! keyword reads and every fact below are the same code in both modes.
//!
//! ## Facts (all optional)
//! - **object / telescope / instrument / filter**: `OBJECT`, `TELESCOP`, `INSTRUME`, `FILTER`
//!   (else `FILTER1`, else `FILTNAME`).
//! - **exposure**: `EXPTIME`, else `EXPOSURE` (seconds).
//! - **date**: the mid-exposure instant and the start of the exposure. The keyword chain, first
//!   readable card wins: `DATE-AVG` → `MJD-AVG` (mid-exposure already) → `DATE-OBS` (mid when its
//!   comment says "averaged" / "mid") → `MJD-OBS` → `MJD` → `JD` → `DATE-BEG` → `DATE_OBS` →
//!   `MJD_OBS` → `JD-OBS` → `JUL-DATE` → `DATE` (the file's date, flagged `date_is_file_date`).
//!   Date strings go through [`parse_fits_datetime`]: quotes and blanks trimmed, `UTC` / `Z` and a
//!   zero offset dropped (a non-zero offset is refused, never shifted), `t` or a blank between
//!   date and time → `T`, `dd/mm/yyyy` → ISO, `dd/mm/yy` → `20yy` (the FITS standard says `19yy`,
//!   but the app's users image comets now; the card is flagged `date_two_digit_year` so the UI can
//!   offer `19yy`), `hh:mm` →
//!   `hh:mm:00`, a leap second `:60` → `:59.999999999`. A date-only value takes its time of day
//!   from `TIME-OBS` / `UTSTART` / `UT` / `TIME-BEG` / `UTC-OBS` (`HH:MM:SS[.f]`, `HH MM SS`,
//!   decimal hours, a trailing `Z` / `+00:00`); with none readable it stays at midnight and
//!   `date_only` is set. The start is the card itself, or for a mid-exposure card `DATE-BEG` /
//!   `MJD-BEG`, else `DATE-OBS` / `MJD-OBS` (when the mid came from `DATE-AVG` / `MJD-AVG`), else
//!   mid − half the exposure (a `DATE-OBS` flagged mid by its comment keeps its own instant). The
//!   mid is start + half the exposure. Time scale from `TIMESYS` (UTC when absent). Reported as
//!   ISO UTC strings and as TDB `(centuries, nanoseconds)` since J2000 TDB
//!   (`hifitime::Duration::to_parts` of `to_tdb_duration`, what `CTime` / the epoch FFI take) and
//!   TDB microseconds. `date_card_raw` is the card used, rebuilt as `KEY = 'value' / comment`
//!   (when nothing parses: the first card of the chain present, so a UI can show it).
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

/// The decoded image of the first HDU holding a 2-D image (extra axes of length 1 accepted).
#[derive(Debug, Clone, PartialEq)]
pub struct FitsImage {
  pub nx: u32,
  pub ny: u32,
  /// row-major, row 0 = FITS `p2 = 1` (file order); NaN = undefined (BLANK / IEEE NaN)
  pub data: Vec<f32>,
  /// keyword → value of the image HDU's header (first occurrence wins)
  pub header: FitsHeader,
  /// 1-based number of the HDU the image came from (1 = primary)
  pub hdu: u32,
  /// the image is tile-compressed (a `.fz` BINTABLE presented as the image it holds)
  pub compressed: bool,
  /// number of 2-D planes of the HDU (the product of the axes after the first two; 1 for a plain
  /// image); `data` holds plane [`Self::plane`]
  pub planes: u64,
  /// the 0-based plane `data` holds: the flat index over the axes after the second, the third
  /// axis fastest (0 for [`read_fits`])
  pub plane: u64,
}

/// The largest accepted image side (pixels), as the pure-Rust parser has always enforced; with
/// [`MAX_PIXELS`] the second guard, it bounds the geometry before any allocation.
#[cfg(feature = "cfitsio")]
const MAX_SIDE: i64 = 1 << 20;
/// The most pixels one plane may hold: 2³⁰ `f32` = 4 GB of decoded pixels. Above it (a stored
/// pixel array over 2 GB for BITPIX 16, far over for the rest) the file is refused with a
/// message; the uncompressed size check against the file length still runs below it.
#[cfg(feature = "cfitsio")]
const MAX_PIXELS: u64 = 1 << 29;

/// Decodes the first image HDU of a FITS file and the metadata of that HDU: pixels as `f32`
/// (BZERO / BSCALE applied, undefined → NaN), its header, and [`FitsMetadata`] built from the
/// cards of that HDU (for an image extension the primary HDU's cards follow them, so the image
/// HDU's keywords win while primary-only facts such as `OBJECT` are kept). A cube gives its
/// first plane ([`read_fits_plane`] for the others). Errors are the messages
/// `sky_image::parse_fits` has always produced for unreadable files.
pub fn read_fits(bytes: &[u8]) -> Result<(FitsImage, FitsMetadata), String> {
  read_fits_plane(bytes, 0)
}

/// [`read_fits`] for plane `plane` of a cube: the 0-based flat index over the axes after the
/// second (`NAXIS3` fastest: for `NAXIS3 × NAXIS4` planes, index `i3 + NAXIS3 · i4`). A plain
/// image has the single plane 0. Past the last plane: `Err("plane … out of range …")`.
pub fn read_fits_plane(bytes: &[u8], plane: u64) -> Result<(FitsImage, FitsMetadata), String> {
  let (img, mut meta) = read_fits_impl(bytes, plane)?;
  meta.naxis1 = img.nx;
  meta.naxis2 = img.ny;
  meta.hdu = img.hdu;
  meta.compressed = img.compressed;
  meta.planes = img.planes;
  meta.plane = img.plane;
  Ok((img, meta))
}

/// The number of 2-D planes of an image with `naxes` (the axes after the first two multiplied;
/// `None` when an axis is not positive or the product overflows) and those extra axes when there
/// are at least two of them (the JSON `planeAxes`).
fn plane_geometry(naxes: &[i64]) -> Option<(u64, Vec<u64>)> {
  let extra = naxes.get(2..).unwrap_or(&[]);
  let planes = extra.iter().try_fold(1u64, |acc, &n| {
    if n <= 0 { None } else { acc.checked_mul(n as u64) }
  })?;
  let axes = if extra.len() >= 2 { extra.iter().map(|&n| n as u64).collect() } else { Vec::new() };
  Some((planes, axes))
}

fn plane_out_of_range(plane: u64, planes: u64) -> String {
  format!("plane {plane} out of range: the image has {planes} plane(s)")
}

/// The FITS block size: every header and data unit is a whole number of these.
#[cfg(feature = "cfitsio")]
const FITS_BLOCK: usize = 2880;

/// `bytes` padded with zeros to a whole number of FITS blocks when they are not one (a data
/// unit written without its last padding, as some HiPS tiles are), else `bytes` as they are.
#[cfg(feature = "cfitsio")]
fn block_padded(bytes: &[u8]) -> alloc::borrow::Cow<'_, [u8]> {
  let rem = bytes.len() % FITS_BLOCK;
  if rem == 0 {
    return alloc::borrow::Cow::Borrowed(bytes);
  }
  let mut v = Vec::with_capacity(bytes.len() + FITS_BLOCK - rem);
  v.extend_from_slice(bytes);
  v.resize(bytes.len() + FITS_BLOCK - rem, 0);
  alloc::borrow::Cow::Owned(v)
}

/// The whole-file compression of `bytes` by its magic: `Some(name)` for gzip (`1f 8b`), Unix
/// compress (`1f 9d`) and bzip2 (`BZh`). The CFITSIO build inflates all three in memory; the
/// pure-Rust build refuses them with the name.
pub fn whole_file_compression(bytes: &[u8]) -> Option<&'static str> {
  if bytes.starts_with(&[0x1f, 0x8b]) {
    Some("gzip (.fits.gz)")
  } else if bytes.starts_with(&[0x1f, 0x9d]) {
    Some("Unix compress (.Z)")
  } else if bytes.starts_with(b"BZh") {
    Some("bzip2 (.bz2)")
  } else {
    None
  }
}

#[cfg(not(feature = "cfitsio"))]
fn read_fits_impl(bytes: &[u8], plane: u64) -> Result<(FitsImage, FitsMetadata), String> {
  if let Some(name) = whole_file_compression(bytes) {
    return Err(format!("{name}-compressed FITS: decompress it first (the CFITSIO build reads it)"));
  }
  // a cube: the legacy parser reads 2-D images only, so it is handed the header with the extra
  // axes set to 1 and the data unit from the requested plane on (its own checks still run on
  // that view: BITPIX, size, truncation); the header it is given back is the file's
  let (h, data_off) = super::FitsHeader::parse_blocks(bytes);
  let geometry = data_off.and_then(|off| {
    let naxis = h.get_i64("NAXIS")?;
    if naxis < 3 {
      return None;
    }
    let naxes: Vec<i64> = (1..=naxis).map(|a| h.get_i64(&format!("NAXIS{a}")).unwrap_or(1)).collect();
    let (planes, axes) = plane_geometry(&naxes)?;
    let bytes_per = match h.get_i64("BITPIX")? {
      8 => 1u64,
      16 => 2,
      32 | -32 => 4,
      64 | -64 => 8,
      _ => return None,
    };
    if naxes[0] <= 0 || naxes[1] <= 0 {
      return None;
    }
    let plane_bytes = (naxes[0] as u64).checked_mul(naxes[1] as u64)?.checked_mul(bytes_per)?;
    Some((off, naxis, planes, axes, plane_bytes))
  });
  let Some((off, naxis, planes, plane_axes, plane_bytes)) = geometry else {
    // a plain image, or a header the legacy parser will refuse with its own message
    if plane > 0 {
      let (_, _, _, _) = super::parse_fits_legacy(bytes)?;
      return Err(plane_out_of_range(plane, 1));
    }
    let (nx, ny, data, header) = super::parse_fits_legacy(bytes)?;
    let meta = FitsMetadata::from_cards(parse_cards(bytes));
    return Ok((FitsImage { nx, ny, data, header, hdu: 1, compressed: false, planes: 1, plane: 0 }, meta));
  };
  if plane >= planes {
    return Err(plane_out_of_range(plane, planes));
  }
  let skip = usize::try_from(plane.saturating_mul(plane_bytes)).map_err(|_| "FITS data truncated".to_string())?;
  let data_start = off.checked_add(skip).filter(|&s| s <= bytes.len()).ok_or("FITS data truncated")?;
  let mut view = Vec::with_capacity(off + (bytes.len() - data_start));
  view.extend_from_slice(&bytes[..off]);
  // every NAXISn card (n ≥ 3) of the header blocks becomes 1
  for a in 3..=naxis {
    let key = format!("{:<8}=", format!("NAXIS{a}"));
    let mut i = 0;
    while i + 80 <= off {
      if view[i..i + 9] == *key.as_bytes() {
        view[i + 10..i + 30].copy_from_slice(format!("{:>20}", 1).as_bytes());
      }
      i += 80;
    }
  }
  view.extend_from_slice(&bytes[data_start..]);
  let (nx, ny, data, _) = super::parse_fits_legacy(&view)?;
  let mut meta = FitsMetadata::from_cards(parse_cards(bytes));
  meta.plane_axes = plane_axes;
  Ok((FitsImage { nx, ny, data, header: h, hdu: 1, compressed: false, planes, plane }, meta))
}

#[cfg(feature = "cfitsio")]
fn read_fits_impl(bytes: &[u8], plane: u64) -> Result<(FitsImage, FitsMetadata), String> {
  use aethervk_cfitsio_rlib::{HduType, MemFits};
  // whole-file compression: inflate into its own buffer first (the codecs' lock is released
  // before the session opens)
  let inflated;
  let bytes: &[u8] = match whole_file_compression(bytes) {
    Some(name) => {
      let r = if aethervk_cfitsio_rlib::is_gzip(bytes) {
        aethervk_cfitsio_rlib::gunzip(bytes)
      } else if aethervk_cfitsio_rlib::is_bzip2(bytes) {
        aethervk_cfitsio_rlib::bunzip2(bytes)
      } else {
        aethervk_cfitsio_rlib::unlzw(bytes)
      };
      inflated = r.map_err(|e| format!("{name}-compressed FITS could not be inflated: {e}"))?;
      &inflated
    }
    None => bytes,
  };
  // a data unit short of its last block (HiPS tiles written without padding) would fail the
  // read of its last pixels with "tried to move past end of file"
  let raw_len = bytes.len() as u64;
  let padded = block_padded(bytes);
  let bytes: &[u8] = &padded;
  // one session per call, dropped before this returns: the session lock is not reentrant
  let mut f = MemFits::open(bytes).map_err(|e| format!("not a readable FITS file: {e}"))?;
  let count = f.hdu_count().map_err(|e| e.to_string())?;
  let mut found = None;
  let (mut tables, mut random_groups) = (0u32, false);
  for hdu in 1..=count {
    if f.move_abs(hdu).map_err(|e| e.to_string())? != HduType::Image {
      tables += 1;
      continue;
    }
    let p = f.image_params().map_err(|e| e.to_string())?;
    if hdu == 1 && p.naxes.first() == Some(&0) && p.naxes.len() > 1 {
      // NAXIS1 = 0 with more axes and GROUPS = T: a random-groups primary
      random_groups = f.raw_cards().map_err(|e| e.to_string())?.iter().any(|c| {
        c.starts_with(b"GROUPS  =") && c[9..].iter().find(|&&b| b != b' ') == Some(&b'T')
      });
    }
    if p.naxes.len() < 2 || p.naxes[0] <= 0 || p.naxes[1] <= 0 {
      continue; // NAXIS = 0 primary, 1-D or zero-size image: look further
    }
    found = Some((hdu, p));
    break;
  }
  let Some((hdu, p)) = found else {
    return Err(if random_groups {
      "random-groups file (GROUPS = T, NAXIS1 = 0): not supported".to_string()
    } else if tables > 0 && tables + 1 >= count {
      "no image HDU (tables only)".to_string()
    } else {
      "no image HDU (NAXIS >= 2) in the file".to_string()
    });
  };
  let (nx, ny) = (p.naxes[0], p.naxes[1]);
  if nx > MAX_SIDE || ny > MAX_SIDE {
    return Err(format!("bad image size {nx} × {ny} (a side over {MAX_SIDE} pixels)"));
  }
  let Some((planes, plane_axes)) = plane_geometry(&p.naxes) else {
    return Err(format!("bad cube geometry {:?}", p.naxes));
  };
  if plane >= planes {
    return Err(plane_out_of_range(plane, planes));
  }
  let bytes_per = match p.bitpix {
    8 => 1u64,
    16 => 2,
    32 | -32 => 4,
    64 | -64 => 8,
    b => return Err(format!("unsupported BITPIX {b}")),
  };
  let n = nx as u64 * ny as u64;
  if n > MAX_PIXELS {
    return Err(format!(
      "image too large: {nx} × {ny} = {n} pixels ({:.1} GB stored); more than {MAX_PIXELS} pixels per plane are refused",
      (n * bytes_per) as f64 / 1e9
    ));
  }
  let compressed = f.is_compressed_image();
  // size sanity before allocating and before cfitsio reads zeros out of the block padding: the
  // file must hold the requested plane of an uncompressed image (its data unit start + the
  // planes up to it, against the length before padding)
  if !compressed {
    let (_, data_start, _) = f.hdu_address().map_err(|e| e.to_string())?;
    let needed = n.saturating_mul(plane + 1).saturating_mul(bytes_per).saturating_add(data_start);
    if needed > raw_len {
      return Err("FITS data truncated".to_string());
    }
  }
  let n = n as usize;
  let mut records = f.cards().map_err(|e| e.to_string())?;
  // a BLANK written as a float (`BLANK = -32768.0`, CDS HiPS tiles) fails cfitsio's integer read
  // of the keyword: give the session the integral value ourselves (a fractional one can match
  // no integer pixel, so the smallest i64 stands for "none")
  if p.bitpix > 0 {
    if let Some(card) = records.iter().find(|c| c.starts_with(b"BLANK   =")) {
      let text = ascii_lossy(&card[9..]);
      let value = text.split('/').next().unwrap_or("").trim();
      if value.parse::<i64>().is_err() {
        if let Ok(v) = value.replace(['D', 'd'], "E").parse::<f64>() {
          let blank = if v.fract() == 0.0 && v.abs() < 9.0e18 { v as i64 } else { i64::MIN };
          f.set_image_null(blank).map_err(|e| e.to_string())?;
        }
      }
    }
  }
  let mut data = Vec::new();
  // plane `plane` is the n pixels from `plane · n` on (FITS stores the first axis fastest)
  f.read_image_f32_from(plane * n as u64, n, &mut data).map_err(|e| format!("FITS data unreadable: {e}"))?;
  if hdu > 1 {
    f.move_abs(1).map_err(|e| e.to_string())?;
    records.extend(f.cards().map_err(|e| e.to_string())?);
  }
  drop(f);
  let header = FitsHeader::from_records(&records);
  let cards: Vec<FitsCard> = records.iter().filter_map(|c| parse_full_card(c)).collect();
  let mut meta = FitsMetadata::from_cards(cards);
  // the stored BITPIX (the card; ZBITPIX for a compressed HDU), as the pure-Rust parser reports it
  meta.bitpix = header.get_i64("BITPIX").map_or(p.bitpix, |b| b as i32);
  meta.plane_axes = plane_axes;
  let img = FitsImage { nx: nx as u32, ny: ny as u32, data, header, hdu, compressed, planes, plane };
  Ok((img, meta))
}

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
  /// keyword the date came from (`DATE-AVG`, `MJD-AVG`, `DATE-OBS`, `MJD-OBS`, `MJD`, `JD`,
  /// `DATE-BEG`, `DATE_OBS`, `MJD_OBS`, `JD-OBS`, `JUL-DATE`, `DATE`)
  pub date_source: Option<String>,
  /// the date card used, rebuilt as `KEY = 'value' / comment`; when no card of the chain parses,
  /// the first one present (so a UI can show what the header says); `None` with no date card
  pub date_card_raw: Option<String>,
  /// the card had no time of day and no time card gave one: the date is at midnight
  pub date_only: bool,
  /// the date is the file's `DATE` (when it was written), not the observation's
  pub date_is_file_date: bool,
  /// start of the exposure, `YYYY-MM-DDTHH:MM:SS.sssZ`
  pub date_start_utc: Option<String>,
  /// start of the exposure, TDB microseconds since J2000 TDB (`SkyImageMeta::date_obs_tdb_us`)
  pub date_start_tdb_us: Option<i64>,
  /// the date card used wrote its year with two digits (`dd/mm/yy`), read as `20yy`; a UI can
  /// offer the FITS standard's `19yy` (the whole date shifted back 100 years)
  pub date_two_digit_year: bool,
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
  /// projection code of the celestial WCS (`TAN`, `SIN`, …; `WcsFacts::projection`), `None`
  /// without a WCS
  pub wcs_projection: Option<String>,
  /// distortion applied by the WCS: `none`, `SIP`, `TPV`, …
  pub wcs_distortion: Option<String>,
  /// celestial frame of the WCS: `ICRS`, `FK5`, `FK4`, `FK4-NO-E`, `galactic`, `ecliptic`
  pub wcs_frame: Option<String>,
  /// equinox of the WCS frame (Julian years, Besselian for FK4); `None` for ICRS / galactic
  pub wcs_equinox: Option<f64>,
  /// no `RADESYS` / `EQUINOX` card named the equatorial frame (ICRS assumed); `None` without a WCS
  pub wcs_frame_assumed: Option<bool>,
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
  /// 1-based HDU the image was read from (1 = primary; `from_cards` assumes it)
  pub hdu: u32,
  /// the image was tile-compressed (`.fz`)
  pub compressed: bool,
  /// 2-D planes in the image HDU (1 for a plain image; a cube's extra axes multiplied)
  pub planes: u64,
  /// the plane that was decoded: the flat index over the extra axes, `NAXIS3` fastest
  pub plane: u64,
  /// the extra axes (`NAXIS3`, `NAXIS4`, …) when there are at least two of them, so a UI can
  /// turn [`Self::plane`] into per-axis indices; empty for a plain image or a 3-D cube
  pub plane_axes: Vec<u64>,
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

// ─── date strings ────────────────────────────────────────────────────────────

/// The calendar fields of a FITS date/time card; `date_only` when the value carried no time of
/// day (the time fields are then 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedTime {
  pub year: i32,
  pub month: u8,
  pub day: u8,
  pub hour: u8,
  pub minute: u8,
  pub second: u8,
  pub nanosecond: u32,
  pub date_only: bool,
  /// the year was written with two digits (`dd/mm/yy`) and read as `20yy`
  pub two_digit_year: bool,
}

/// Time-of-day cards consulted for a date-only value, in order.
pub const TIME_OF_DAY_KEYS: [&str; 5] = ["TIME-OBS", "UTSTART", "UT", "TIME-BEG", "UTC-OBS"];

/// Drops a trailing zone marker: `UTC` / `UT`, `Z` / `z`, or a zero offset (`+00:00`, `-00:00`,
/// `+0000`, `+00`; looked for after the time's first `:` when `has_time`). `None` for a non-zero
/// offset: the instant is never shifted silently.
fn strip_zone(s: &str, has_time: bool) -> Option<&str> {
  let t = s.trim_end();
  for z in ["UTC", "utc", "UT", "Z", "z"] {
    if let Some(r) = t.strip_suffix(z) {
      return Some(r.trim_end());
    }
  }
  if has_time {
    if let Some(colon) = t.find(':') {
      if let Some(i) = t[colon..].rfind(['+', '-']) {
        let i = colon + i;
        let digits: String = t[i + 1..].chars().filter(|c| *c != ':').collect();
        if !digits.is_empty() && digits.len() <= 4 && digits.bytes().all(|b| b.is_ascii_digit()) {
          return if digits.bytes().all(|b| b == b'0') { Some(t[..i].trim_end()) } else { None };
        }
      }
    }
  }
  Some(t)
}

/// `[+-]?N{1,}` → the number (every character a digit).
fn digits<T: core::str::FromStr>(s: &str) -> Option<T> {
  let s = s.trim();
  if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
    return None;
  }
  s.parse().ok()
}

/// `YYYY-MM-DD`, `YYYY/MM/DD`, `dd/mm/yyyy` or `dd/mm/yy` (→ `20yy`) → `(YYYY-MM-DD, the year
/// had two digits)`.
///
/// The FITS standard reads a two-digit year as `19yy` (and so does cfitsio's `ffs2tm`), but the
/// app's users image comets now, so the year is made explicit as `20yy` here, before either
/// parser sees it, and the flag lets the UI offer `19yy` instead.
fn normalise_date(d: &str) -> Option<(String, bool)> {
  let d = d.trim();
  let sep = if d.contains('/') { '/' } else { '-' };
  let mut it = d.split(sep);
  let (a, b, c) = (it.next()?, it.next()?, it.next()?);
  if it.next().is_some() {
    return None;
  }
  let (y, m, day) = if sep == '-' || a.trim().len() == 4 { (a, b, c) } else { (c, b, a) };
  let mut year: i32 = digits(y)?;
  let two_digit = sep == '/' && y.trim().len() <= 2;
  if two_digit {
    year += 2000;
  }
  Some((format!("{year:04}-{:02}-{:02}", digits::<u8>(m)?, digits::<u8>(day)?), two_digit))
}

/// `hh[:mm[:ss[.f…]]]` → `hh:mm:ss[.f…]` (fraction capped at 9 digits; second 60 → `59.999999999`).
fn normalise_time(t: &str) -> Option<String> {
  let mut it = t.trim().split(':');
  let hh: u8 = digits(it.next()?)?;
  let mm: u8 = it.next().map(digits).unwrap_or(Some(0))?;
  let (ss, frac): (u8, &str) = match it.next() {
    Some(s) => {
      let s = s.trim();
      let (whole, frac) = s.split_once('.').unwrap_or((s, ""));
      if !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
      }
      (digits(whole)?, frac)
    }
    None => (0, ""),
  };
  if it.next().is_some() {
    return None;
  }
  if ss == 60 {
    return Some(format!("{hh:02}:{mm:02}:59.999999999"));
  }
  let frac: String = frac.chars().take(9).collect();
  Some(if frac.is_empty() { format!("{hh:02}:{mm:02}:{ss:02}") } else { format!("{hh:02}:{mm:02}:{ss:02}.{frac}") })
}

/// The canonical form of a FITS date card value: `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ss[.f]` (what
/// cfitsio's `ffs2tm` accepts), after the normalisation described in the module doc. `None` when
/// the value is not a date or carries a non-zero zone offset.
pub fn normalise_fits_datetime(raw: &str) -> Option<String> {
  normalise_fits_datetime_flagged(raw).map(|(canonical, _)| canonical)
}

/// [`normalise_fits_datetime`] plus whether the year was written with two digits (`dd/mm/yy`,
/// read as `20yy`).
pub fn normalise_fits_datetime_flagged(raw: &str) -> Option<(String, bool)> {
  let s = raw.trim().trim_matches('\'').trim();
  // the date ends at the first `T` / `t` / blank (`2022-02-28 T 23:59:59 UTC` is seen in the wild)
  let (date, time) = match s.find(['T', 't', ' ']) {
    Some(i) => (&s[..i], s[i..].trim_start_matches(['T', 't', ' '])),
    None => (s, ""),
  };
  let time = strip_zone(time, true)?;
  let date = if time.is_empty() { strip_zone(date, false)? } else { date };
  let (date, two_digit_year) = normalise_date(date)?;
  if time.is_empty() {
    return Some((date, two_digit_year));
  }
  Some((format!("{date}T{}", normalise_time(time)?), two_digit_year))
}

/// Nanoseconds of the fraction after the last `.` of a canonical string (0 without one).
fn fraction_ns(canonical: &str) -> u32 {
  let Some((_, frac)) = canonical.rsplit_once('.') else { return 0 };
  let n = frac.len().min(9) as u32;
  frac[..n as usize].parse::<u32>().map_or(0, |v| v * 10u32.pow(9 - n))
}

/// The pure-Rust parser of a canonical string (the non-cfitsio path, and the cfitsio path's
/// cross-check in tests).
pub fn parse_canonical_datetime(canonical: &str) -> Option<ParsedTime> {
  let (date, time) = match canonical.split_once('T') {
    Some((d, t)) => (d, Some(t)),
    None => (canonical, None),
  };
  let mut d = date.split('-');
  let (year, month, day): (i32, u8, u8) = (digits(d.next()?)?, digits(d.next()?)?, digits(d.next()?)?);
  if d.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
    return None;
  }
  let (mut hour, mut minute, mut second) = (0u8, 0u8, 0u8);
  if let Some(t) = time {
    let mut it = t.split(':');
    hour = digits(it.next()?)?;
    minute = digits(it.next()?)?;
    let sec = it.next()?;
    second = digits(sec.split_once('.').map_or(sec, |(w, _)| w))?;
    if it.next().is_some() || hour > 23 || minute > 59 || second > 60 {
      return None;
    }
  }
  Some(ParsedTime { year, month, day, hour, minute, second, nanosecond: fraction_ns(canonical), date_only: time.is_none(), two_digit_year: false })
}

/// cfitsio's `ffs2tm` on the canonical string (the fraction is taken from the digits, not from
/// the double it returns). Takes the wrapper's session lock: never call it with a `MemFits` open
/// on this thread.
#[cfg(feature = "cfitsio")]
fn parse_canonical_cfitsio(canonical: &str) -> Option<ParsedTime> {
  let t = aethervk_cfitsio_rlib::str2time(canonical).ok()?;
  let field = |v: i32| u8::try_from(v).ok();
  Some(ParsedTime {
    year: t.year,
    month: field(t.month)?,
    day: field(t.day)?,
    hour: field(t.hour)?,
    minute: field(t.minute)?,
    second: field(t.second.floor() as i32)?,
    nanosecond: fraction_ns(canonical),
    date_only: !canonical.contains('T'),
    two_digit_year: false,
  })
}

/// Normalises a FITS date card value ([`normalise_fits_datetime`]) and parses the canonical form
/// (cfitsio `ffs2tm` with the `cfitsio` feature, else [`parse_canonical_datetime`]). The calendar
/// validity (a 30 February, a 24th hour) is left to the epoch construction. A two-digit year is
/// already explicit (`20yy`) in the canonical string, so both parsers agree on it; the result's
/// `two_digit_year` says the card wrote it that way.
pub fn parse_fits_datetime(raw: &str) -> Option<ParsedTime> {
  let (canonical, two_digit_year) = normalise_fits_datetime_flagged(raw)?;
  #[cfg(feature = "cfitsio")]
  let parsed = parse_canonical_cfitsio(&canonical);
  #[cfg(not(feature = "cfitsio"))]
  let parsed = parse_canonical_datetime(&canonical);
  parsed.map(|p| ParsedTime { two_digit_year, ..p })
}

/// A time-of-day card value → `(hour, minute, second, nanosecond)`: `HH:MM:SS[.f]`, `HH:MM`,
/// `HH MM SS`, decimal hours (`18.5` → 18:30:00), each with an optional trailing `Z` / `UTC` /
/// zero offset. Out-of-range fields are left to the epoch construction (decimal hours must be in
/// `[0, 24)`).
pub fn parse_time_of_day(raw: &str) -> Option<(u8, u8, u8, u32)> {
  let s = strip_zone(raw.trim().trim_matches('\'').trim(), true)?;
  if s.is_empty() {
    return None;
  }
  let hms = |canonical: &str| -> Option<(u8, u8, u8, u32)> {
    let p = parse_canonical_datetime(&format!("2000-01-01T{canonical}"))?;
    Some((p.hour, p.minute, p.second, p.nanosecond))
  };
  if s.contains(':') {
    return hms(&normalise_time(s)?);
  }
  if s.contains(char::is_whitespace) {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if !(2..=3).contains(&parts.len()) {
      return None;
    }
    return hms(&normalise_time(&parts.join(":"))?);
  }
  let hours: f64 = s.parse().ok()?;
  if !(0.0..24.0).contains(&hours) {
    return None;
  }
  let total_ns = (hours * 3_600.0e9).round() as u64;
  let (h, rest) = (total_ns / 3_600_000_000_000, total_ns % 3_600_000_000_000);
  let (m, rest) = (rest / 60_000_000_000, rest % 60_000_000_000);
  Some((h as u8, m as u8, (rest / 1_000_000_000) as u8, (rest % 1_000_000_000) as u32))
}

/// The facts of a header alone (`SkyImage::from_parts`: a decoded image, possibly with a sidecar
/// overlaid); `comments` supplies the card comments where its keys match (the "averaged"
/// `DATE-OBS` and the west-longitude wording), empty otherwise.
pub fn facts_of_header(h: &FitsHeader, comments: Option<&[FitsCard]>, nx: u32, ny: u32) -> FitsFacts {
  let cards: Vec<FitsCard> = h
    .cards
    .iter()
    .map(|(k, v)| FitsCard {
      key: k.clone(),
      value: v.clone(),
      comment: comments
        .and_then(|cs| cs.iter().find(|c| &c.key == k))
        .map(|c| c.comment.clone())
        .unwrap_or_default(),
    })
    .collect();
  facts_of(&cards, h, nx, ny)
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
    // the planes of a cube header (NAXIS3 · NAXIS4 …); a missing axis card counts as 1
    let naxis = h.get_i64("NAXIS").unwrap_or(2).max(2);
    let extra: Vec<u64> = (3..=naxis).map(|a| h.get_i64(&format!("NAXIS{a}")).unwrap_or(1).max(1) as u64).collect();
    let planes = extra.iter().fold(1u64, |acc, &n| acc.saturating_mul(n));
    let plane_axes = if extra.len() >= 2 { extra } else { Vec::new() };
    Self { naxis1, naxis2, bitpix, hdu: 1, compressed: false, planes, plane: 0, plane_axes, cards, facts }
  }

  /// From the bytes of a FITS file: the metadata of its first image HDU ([`read_fits`]); when
  /// no image can be decoded, the primary header's cards alone, so a broken file still shows
  /// its header.
  pub fn from_fits_bytes(bytes: &[u8]) -> Self {
    match read_fits(bytes) {
      Ok((_, m)) => m,
      Err(_) => Self::from_cards(parse_cards(bytes)),
    }
  }

  /// The metadata as JSON (camelCase keys, `null` for missing facts), see
  /// `docs/fits_ffi_contract.md`.
  pub fn to_json(&self) -> String {
    let mut j = String::with_capacity(256 + self.cards.len() * 96);
    j.push_str(&format!(
      "{{\"naxis1\":{},\"naxis2\":{},\"bitpix\":{},\"hdu\":{},\"compressed\":{},\"planes\":{},\"plane\":{}",
      self.naxis1, self.naxis2, self.bitpix, self.hdu, self.compressed, self.planes, self.plane
    ));
    if !self.plane_axes.is_empty() {
      j.push_str(",\"planeAxes\":[");
      for (i, n) in self.plane_axes.iter().enumerate() {
        if i > 0 {
          j.push(',');
        }
        j.push_str(&n.to_string());
      }
      j.push(']');
    }
    j.push_str(",\"cards\":[");
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
    // the date provenance keys (additive: readers of the older JSON ignore them)
    key(j, "dateCardRaw");
    opt_s(j, &self.date_card_raw);
    key(j, "dateOnly");
    j.push_str(if self.date_only { "true" } else { "false" });
    key(j, "dateIsFileDate");
    j.push_str(if self.date_is_file_date { "true" } else { "false" });
    key(j, "dateStartUtc");
    opt_s(j, &self.date_start_utc);
    key(j, "dateStartTdbUs");
    j.push_str(&self.date_start_tdb_us.map_or("null".to_string(), |v| format!("{v}")));
    key(j, "dateTwoDigitYear");
    j.push_str(if self.date_two_digit_year { "true" } else { "false" });
    // the WCS keys (additive; `null` without a WCS)
    key(j, "wcsProjection");
    opt_s(j, &self.wcs_projection);
    key(j, "wcsDistortion");
    opt_s(j, &self.wcs_distortion);
    key(j, "wcsFrame");
    opt_s(j, &self.wcs_frame);
    key(j, "wcsEquinox");
    j.push_str(&json_f64(self.wcs_equinox));
    key(j, "wcsFrameAssumed");
    j.push_str(match self.wcs_frame_assumed {
      Some(true) => "true",
      Some(false) => "false",
      None => "null",
    });
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
  // the first readable time-of-day card, for a date-only value
  let time_of_day = || TIME_OF_DAY_KEYS.iter().find_map(|k| h.get(k).and_then(parse_time_of_day));
  // a date-string card → (instant, the value had no time of day and no time card gave one, the
  // year was written with two digits)
  let iso = |key: &str| -> Option<(Epoch, bool, bool)> {
    let p = parse_fits_datetime(h.get(key)?)?;
    let (mut t, mut date_only) = ((p.hour, p.minute, p.second, p.nanosecond), false);
    if p.date_only {
      match time_of_day() {
        Some(tod) => t = tod,
        None => date_only = true,
      }
    }
    let e = Epoch::maybe_from_gregorian(p.year, p.month, p.day, t.0, t.1, t.2, t.3, scale).ok()?;
    Some((e, date_only, p.two_digit_year))
  };
  let mjd = |k: &str| h.get_f64(k).filter(|m| m.is_finite()).map(|m| Epoch::from_mjd_in_time_scale(m, scale));
  let jd = |k: &str| {
    h.get_f64(k).filter(|jd| *jd > 2_400_000.5).map(|jd| Epoch::from_mjd_in_time_scale(jd - 2_400_000.5, scale))
  };
  #[derive(Clone, Copy, PartialEq)]
  enum Kind {
    Date,
    Mjd,
    Jd,
  }
  // the card as a UI shows it: `KEY     = 'value' / comment`
  let card_text = |c: &FitsCard, kind: Kind| -> String {
    let mut s = format!("{:<8}= ", c.key);
    match kind {
      Kind::Date => s.push_str(&format!("'{}'", c.value.replace('\'', "''"))),
      _ => s.push_str(c.value.trim()),
    }
    if !c.comment.is_empty() {
      s.push_str(" / ");
      s.push_str(&c.comment);
    }
    s
  };
  const CHAIN: [(&str, Kind); 12] = [
    ("DATE-AVG", Kind::Date),
    ("MJD-AVG", Kind::Mjd),
    ("DATE-OBS", Kind::Date),
    ("MJD-OBS", Kind::Mjd),
    ("MJD", Kind::Mjd),
    ("JD", Kind::Jd),
    ("DATE-BEG", Kind::Date),
    ("DATE_OBS", Kind::Date),
    ("MJD_OBS", Kind::Mjd),
    ("JD-OBS", Kind::Jd),
    ("JUL-DATE", Kind::Jd),
    ("DATE", Kind::Date),
  ];
  // (instant, keyword, mid-exposure card, date only, file date, two-digit year, card text)
  let mut found: Option<(Epoch, &str, bool, bool, bool, bool, String)> = None;
  let mut first_card: Option<String> = None;
  for (key, kind) in CHAIN {
    let Some(c) = card(cards, key).filter(|c| !c.value.trim().is_empty()) else {
      continue;
    };
    if first_card.is_none() {
      first_card = Some(card_text(c, kind));
    }
    let parsed = match kind {
      Kind::Date => iso(key),
      Kind::Mjd => mjd(key).map(|e| (e, false, false)),
      Kind::Jd => jd(key).map(|e| (e, false, false)),
    };
    if let Some((e, date_only, two_digit_year)) = parsed {
      let is_mid = match key {
        "DATE-AVG" | "MJD-AVG" => true,
        "DATE-OBS" => {
          let cm = c.comment.to_ascii_lowercase();
          cm.contains("averag") || cm.contains("mid")
        }
        _ => false,
      };
      found = Some((e, key, is_mid, date_only, key == "DATE", two_digit_year, card_text(c, kind)));
      break;
    }
  }
  let half = f.exposure_s.filter(|t| t.is_finite() && *t > 0.0).map(|t| Duration::from_seconds(t * 0.5));
  let mut lst_epoch = None;
  match found {
    Some((e, src, is_mid, date_only, file_date, two_digit_year, raw)) => {
      let (start, mid) = if !is_mid {
        (e, half.map_or(e, |d| e + d))
      } else {
        // an explicit start card; else the start card the mid was averaged from; else the mid
        // itself for a DATE-OBS flagged by its comment, or half the exposure back
        let explicit = iso("DATE-BEG").map(|(e, ..)| e).or_else(|| mjd("MJD-BEG"));
        let from_obs = || {
          if src == "DATE-OBS" {
            None
          } else {
            iso("DATE-OBS").map(|(e, ..)| e).or_else(|| mjd("MJD-OBS"))
          }
        };
        let start = explicit.or_else(from_obs).unwrap_or_else(|| {
          if src == "DATE-OBS" { e } else { half.map_or(e, |d| e - d) }
        });
        (start, e)
      };
      lst_epoch = Some(start);
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
      f.date_card_raw = Some(raw);
      f.date_only = date_only;
      f.date_is_file_date = file_date;
      f.date_two_digit_year = two_digit_year;
      f.date_start_utc = Some(utc_string(start));
      f.date_start_tdb_us = Some((start.to_tdb_duration().total_nanoseconds() / 1000) as i64);
    }
    None => f.date_card_raw = first_card,
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
      let wf = w.facts();
      f.wcs_projection = Some(wf.projection.to_string());
      f.wcs_distortion = Some(wf.distortion.to_string());
      f.wcs_frame = Some(wf.frame.to_string());
      f.wcs_equinox = wf.equinox;
      f.wcs_frame_assumed = Some(wf.frame_assumed);
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
