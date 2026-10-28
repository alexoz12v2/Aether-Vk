//! FITS texture support: decoding per BITPIX, stretch, header cards / facts.

use super::fits_metadata::*;
use super::fits_stretch::*;
use super::parse_fits;
use alloc::{format, string::String, vec::Vec};

/// The 80-column card of `text`.
fn card(text: &str) -> [u8; 80] {
  let mut c = [b' '; 80];
  c[..text.len()].copy_from_slice(text.as_bytes());
  c
}

/// A primary-HDU FITS: `nx × ny`, `bitpix`, `extra` cards, `values` raw (integers) or IEEE.
pub(crate) fn fits_bytes(bitpix: i32, nx: u32, ny: u32, extra: &[&str], values: &[f64]) -> Vec<u8> {
  let mut h = Vec::new();
  for c in [
    "SIMPLE  =                    T".to_string(),
    format!("BITPIX  = {bitpix:>20}"),
    "NAXIS   =                    2".to_string(),
    format!("NAXIS1  = {nx:>20}"),
    format!("NAXIS2  = {ny:>20}"),
  ] {
    h.extend_from_slice(&card(&c));
  }
  for c in extra {
    h.extend_from_slice(&card(c));
  }
  h.extend_from_slice(&card("END"));
  h.resize(h.len().div_ceil(2880) * 2880, b' ');
  let mut d = Vec::new();
  for &v in values {
    match bitpix {
      8 => d.push(v as u8),
      16 => d.extend_from_slice(&(v as i16).to_be_bytes()),
      32 => d.extend_from_slice(&(v as i32).to_be_bytes()),
      64 => d.extend_from_slice(&(v as i64).to_be_bytes()),
      -32 => d.extend_from_slice(&(v as f32).to_be_bytes()),
      _ => d.extend_from_slice(&v.to_be_bytes()),
    }
  }
  d.resize(d.len().div_ceil(2880) * 2880, 0);
  h.extend_from_slice(&d);
  h
}

fn assert_close(a: f32, b: f64) {
  assert!((a as f64 - b).abs() <= 1e-6 * b.abs().max(1.0), "{a} vs {b}");
}

// ─── decoding, one test per BITPIX ───────────────────────────────────────────

#[test]
fn fits_bitpix_8_bzero_bscale_blank() {
  let b = fits_bytes(8, 3, 1, &["BZERO   =                 10.0", "BSCALE  =                  0.5", "BLANK   =                  255"], &[0.0, 4.0, 255.0]);
  let (nx, ny, v, _) = parse_fits(&b).unwrap();
  assert_eq!((nx, ny), (3, 1));
  assert_close(v[0], 10.0);
  assert_close(v[1], 12.0);
  assert!(v[2].is_nan());
}

#[test]
fn fits_bitpix_16_bzero_bscale_blank() {
  // the unsigned-16 convention BZERO = 32768
  let b = fits_bytes(16, 3, 1, &["BZERO   =              32768.0", "BSCALE  =                  1.0", "BLANK   =               -32768"], &[-32767.0, 0.0, -32768.0]);
  let (_, _, v, _) = parse_fits(&b).unwrap();
  assert_close(v[0], 1.0);
  assert_close(v[1], 32768.0);
  assert!(v[2].is_nan());
}

#[test]
fn fits_bitpix_32_bzero_bscale_blank() {
  let b = fits_bytes(32, 3, 1, &["BZERO   =                 -5.0", "BSCALE  =                  2.0", "BLANK   =          -2147483648"], &[100000.0, -3.0, -2147483648.0]);
  let (_, _, v, _) = parse_fits(&b).unwrap();
  assert_close(v[0], 199995.0);
  assert_close(v[1], -11.0);
  assert!(v[2].is_nan());
}

#[test]
fn fits_bitpix_minus32_bzero_bscale_nan() {
  let b = fits_bytes(-32, 3, 1, &["BZERO   =                  1.0", "BSCALE  =                  3.0"], &[0.5, -2.0, f64::NAN]);
  let (_, _, v, _) = parse_fits(&b).unwrap();
  assert_close(v[0], 2.5);
  assert_close(v[1], -5.0);
  assert!(v[2].is_nan());
}

#[test]
fn fits_bitpix_minus64_bzero_bscale_nan() {
  let b = fits_bytes(-64, 3, 1, &["BZERO   =                  0.0", "BSCALE  =                 1D-1"], &[123.0, -7.0, f64::NAN]);
  let (_, _, v, _) = parse_fits(&b).unwrap();
  assert_close(v[0], 12.3);
  assert_close(v[1], -0.7);
  assert!(v[2].is_nan());
}

// ─── stretch ─────────────────────────────────────────────────────────────────

fn ramp(n: usize) -> Vec<f32> {
  (0..n).map(|i| i as f32).collect()
}

#[test]
fn fits_zscale_on_a_ramp_is_the_full_range() {
  let d = ramp(100 * 100);
  let (b, w) = black_white(&d, 100, 100, StretchMode::ZScale);
  assert_eq!((b, w), (0.0, 9999.0));
}

#[test]
fn fits_zscale_clips_stars_on_a_noisy_background() {
  // background 1000 ± ~10 (deterministic LCG noise), 1 % of the pixels bright stars
  let (nx, ny) = (200u32, 200u32);
  let mut seed = 12345u64;
  let mut d = Vec::new();
  for i in 0..(nx * ny) as usize {
    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5; // [-0.5, 0.5)
    let v = if i % 100 == 7 { 60000.0 } else { 1000.0 + 35.0 * u };
    d.push(v as f32);
  }
  let (b, w) = black_white(&d, nx, ny, StretchMode::ZScale);
  assert!(b > 900.0 && b < 1000.0, "black {b}");
  assert!(w > 1000.0 && w < 1200.0, "white {w}");
}

#[test]
fn fits_fill_value_borders_are_left_out_of_the_statistics() {
  // a ramp 1000..1999 whose first 10 % is a constant fill value far below
  let mut d: Vec<f32> = (0..10000).map(|i| 1000.0 + (i % 1000) as f32).collect();
  for v in d.iter_mut().take(1000) {
    *v = -38288.0;
  }
  let (b, w) = black_white(&d, 100, 100, StretchMode::ZScale);
  assert!((1000.0..1010.0).contains(&b) && (1990.0..=1999.0).contains(&w), "{b} {w}");
  let (b, _) = black_white(&d, 100, 100, StretchMode::Percentile { low: 0.5, high: 99.5 });
  assert!(b >= 1000.0, "{b}");
  // runs of >= max(2, 1 %) samples are trimmed, single extremes kept
  assert_eq!(trim_fill_runs(&[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]).len(), 4);
  assert_eq!(trim_fill_runs(&[0.0, 1.0, 2.0]).len(), 3);
  assert_eq!(trim_fill_runs(&[5.0; 4]).len(), 4);
}

#[test]
fn fits_percentile_and_manual_points() {
  let d = ramp(100 * 100);
  let (b, w) = black_white(&d, 100, 100, StretchMode::Percentile { low: 0.5, high: 99.5 });
  assert!((b - 50.0).abs() <= 1.0, "{b}");
  assert!((w - 9949.0).abs() <= 1.0, "{w}");
  let (b, w) = black_white(&d, 100, 100, StretchMode::Manual { black: 10.0, white: 20.0 });
  assert_eq!((b, w), (10.0, 20.0));
  // a constant image still gets white > black
  let c = alloc::vec![5.0f32; 16];
  let (b, w) = black_white(&c, 4, 4, StretchMode::ZScale);
  assert!(w > b && b == 5.0);
}

#[test]
fn fits_nan_and_blank_pixels_are_excluded_from_statistics() {
  let mut d = ramp(100 * 100);
  let clean_z = black_white(&d, 100, 100, StretchMode::ZScale);
  let clean_p = black_white(&d, 100, 100, StretchMode::Percentile { low: 1.0, high: 99.0 });
  // NaN at the extremes (BLANK pixels decode to NaN) would otherwise be min / max
  let mut with_nan = d.clone();
  with_nan.extend(core::iter::repeat_n(f32::NAN, 100));
  with_nan.splice(0..0, core::iter::repeat_n(f32::NAN, 100));
  let (zb, zw) = black_white(&with_nan, 100, 102, StretchMode::ZScale);
  assert_eq!((zb, zw), clean_z);
  for i in (0..10000).step_by(97) {
    d[i] = f32::NAN;
  }
  let (zb, zw) = black_white(&d, 100, 100, StretchMode::ZScale);
  assert!((zb - clean_z.0).abs() < 100.0 && (zw - clean_z.1).abs() < 100.0);
  let (pb, pw) = black_white(&d, 100, 100, StretchMode::Percentile { low: 1.0, high: 99.0 });
  assert!((pb - clean_p.0).abs() < 5.0 && (pw - clean_p.1).abs() < 5.0);
  // and they render black, opaque
  let px = render_rgba8(&d, 0.0, 9999.0, Transfer::Linear);
  assert_eq!(&px[0..4], &[0, 0, 0, 255]);
}

#[test]
fn fits_transfers_are_monotonic_and_span_0_1() {
  for t in [Transfer::Linear, Transfer::Log, Transfer::Asinh] {
    assert!(transfer(0.0, t).abs() < 1e-12);
    assert!((transfer(1.0, t) - 1.0).abs() < 1e-12);
    let mut prev = -1.0;
    for i in 0..=1000 {
      let y = transfer(i as f64 / 1000.0, t);
      assert!(y > prev, "{t:?} not increasing at {i}");
      prev = y;
    }
  }
  // log / asinh lift the faint end
  assert!(transfer(0.1, Transfer::Log) > 0.5);
  assert!(transfer(0.1, Transfer::Asinh) > 0.25);
}

#[test]
fn fits_render_is_monotonic_grey_opaque_and_clipped() {
  let d = ramp(1000);
  for t in [Transfer::Linear, Transfer::Log, Transfer::Asinh] {
    let px = render_rgba8(&d, 100.0, 900.0, t);
    assert_eq!(px.len(), 4000);
    let mut prev = 0u8;
    for (i, p) in px.chunks_exact(4).enumerate() {
      assert!(p[0] == p[1] && p[1] == p[2] && p[3] == 255);
      assert!(p[0] >= prev, "{t:?} at {i}");
      prev = p[0];
      if i <= 100 {
        assert_eq!(p[0], 0);
      }
      if i >= 900 {
        assert_eq!(p[0], 255);
      }
    }
  }
  // linear midpoint
  let px = render_rgba8(&[500.0], 100.0, 900.0, Transfer::Linear);
  assert!((px[0] as i32 - 128).abs() <= 1);
}

#[test]
fn fits_flip_rows_puts_the_last_fits_row_first() {
  let mut d: Vec<u32> = (0..12).collect(); // 4 × 3
  flip_rows(&mut d, 4, 3);
  assert_eq!(d, [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]);
  let mut d: Vec<u32> = (0..8).collect(); // 4 × 2
  flip_rows(&mut d, 4, 2);
  assert_eq!(d, [4, 5, 6, 7, 0, 1, 2, 3]);
}

#[test]
fn fits_stretch_codes_roundtrip_and_validation() {
  for (m, t) in [(0, 0), (1, 1), (2, 2)] {
    let s = FitsStretch::from_codes(m, t, 1.0, 99.0).unwrap();
    assert_eq!(s.codes(), (m, t));
  }
  assert!(FitsStretch::from_codes(3, 0, 0.0, 1.0).is_none());
  assert!(FitsStretch::from_codes(0, 3, 0.0, 1.0).is_none());
  assert!(FitsStretch::from_codes(1, 0, -1.0, 99.0).is_none());
  assert!(FitsStretch::from_codes(2, 0, f64::NAN, 1.0).is_none());
}

// ─── header cards / facts ────────────────────────────────────────────────────

#[test]
fn fits_cards_keep_order_values_and_comments() {
  let b = fits_bytes(
    16,
    1,
    1,
    &[
      "OBJECT  = 'M 31 ''x'' '         / the target / with slash",
      "COMMENT   free text here",
      "EXPTIME =                 30.5 / [s] exposure",
      "HIERARCH ESO DET DIT = 1.5 / det",
      "HISTORY   step 1",
    ],
    &[0.0],
  );
  let cards = parse_cards(&b);
  let keys: Vec<&str> = cards.iter().map(|c| c.key.as_str()).collect();
  assert_eq!(keys, ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "OBJECT", "COMMENT", "EXPTIME", "ESO DET DIT", "HISTORY"]);
  assert_eq!(cards[5].value, "M 31 'x'");
  assert_eq!(cards[5].comment, "the target / with slash");
  assert_eq!(cards[6].value, "free text here");
  assert_eq!((cards[7].value.as_str(), cards[7].comment.as_str()), ("30.5", "[s] exposure"));
  assert_eq!((cards[8].value.as_str(), cards[8].comment.as_str()), ("1.5", "det"));
  let m = FitsMetadata::from_cards(cards);
  assert_eq!(m.facts.object.as_deref(), Some("M 31 'x'"));
  assert_eq!(m.facts.exposure_s, Some(30.5));
  let j = m.to_json();
  assert!(j.contains("{\"key\":\"OBJECT\",\"value\":\"M 31 'x'\",\"comment\":\"the target / with slash\"}"));
  assert!(j.contains("\"telescope\":null"));
}

#[test]
fn fits_dates_space_separator_mid_exposure_and_mjd_fallback() {
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(
    16,
    1,
    1,
    &["DATE-OBS= '2021-12-27 00:14:47.345'", "EXPTIME =                 20.0"],
    &[0.0],
  ));
  // start + 10 s
  assert_eq!(m.facts.date_mid_utc.as_deref(), Some("2021-12-27T00:14:57.345Z"));
  assert_eq!(m.facts.date_source.as_deref(), Some("DATE-OBS"));
  assert!(!m.facts.date_is_mid && m.facts.timesys_assumed);
  // TDB parts agree with hifitime
  let e = hifitime::Epoch::maybe_from_gregorian(2021, 12, 27, 0, 14, 57, 345_000_000, hifitime::TimeScale::UTC).unwrap();
  let (c, ns) = e.to_tdb_duration().to_parts();
  assert_eq!((m.facts.date_mid_tdb_centuries, m.facts.date_mid_tdb_nanoseconds), (Some(c), Some(ns)));
  // date-only DATE-OBS + UTSTART
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["DATE-OBS= '2021-12-25'", "UTSTART = '00:07:28.858'"], &[0.0]));
  assert_eq!(m.facts.date_mid_utc.as_deref(), Some("2021-12-25T00:07:28.858Z"));
  // MJD-OBS fallback, then MJD
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["MJD-OBS =              59573.5"], &[0.0]));
  assert_eq!(m.facts.date_mid_utc.as_deref(), Some("2021-12-25T12:00:00.000Z"));
  assert_eq!(m.facts.date_source.as_deref(), Some("MJD-OBS"));
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["MJD     =              59573.25"], &[0.0]));
  assert_eq!(m.facts.date_source.as_deref(), Some("MJD"));
  assert_eq!(m.facts.date_mid_utc.as_deref(), Some("2021-12-25T06:00:00.000Z"));
  // nothing
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &[], &[0.0]));
  assert!(m.facts.date_mid_utc.is_none() && m.facts.date_mid_tdb_centuries.is_none());
  assert!(m.to_json().contains("\"dateMidUtc\":null"));
}

#[test]
fn fits_site_conventions() {
  // comment says West: negated to east positive
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(
    16,
    1,
    1,
    &["LATITUDE=           28.7624000 / [degrees] Observatory Latitude", "LONGITUD=           17.8792000 / [degrees West] Observatory Longitude"],
    &[0.0],
  ));
  assert_eq!(m.facts.site_lat_deg, Some(28.7624));
  assert_eq!(m.facts.site_lon_deg, Some(-17.8792));
  assert!(m.facts.site_lon_west_convention);
  assert_eq!(m.facts.site_source.as_deref(), Some("LATITUDE/LONGITUD"));
  // "east positive, west negative" is not a west convention
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["SITELAT = '+19:49:36'", "SITELONG=          -155.4719 / East longitude (west negative)", "SITEELEV=               4204.0"], &[0.0]));
  assert!((m.facts.site_lat_deg.unwrap() - 19.8267).abs() < 1e-3);
  assert_eq!(m.facts.site_lon_deg, Some(-155.4719));
  assert!(!m.facts.site_lon_west_convention);
  assert_eq!(m.facts.site_elev_m, Some(4204.0));
  // W suffix
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["LAT-OBS =                 32.0", "LONG-OBS= '110:56:00 W'"], &[0.0]));
  assert!((m.facts.site_lon_deg.unwrap() + 110.9333).abs() < 1e-3);
  // OBSGEO-X/Y/Z (Greenwich, on the ellipsoid at 51.4769 N, 0 E, 46 m)
  let (lat, lon, h) = (51.4769f64.to_radians(), 0.0f64, 46.0);
  let a = 6378137.0;
  let f = 1.0 / 298.257223563;
  let e2 = f * (2.0 - f);
  let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
  let (x, y, z) = ((n + h) * lat.cos() * lon.cos(), (n + h) * lat.cos() * lon.sin(), (n * (1.0 - e2) + h) * lat.sin());
  let cards: Vec<String> = [("OBSGEO-X", x), ("OBSGEO-Y", y), ("OBSGEO-Z", z)].iter().map(|(k, v)| format!("{k:<8}= {v:>20.4}")).collect();
  let refs: Vec<&str> = cards.iter().map(|s| s.as_str()).collect();
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &refs, &[0.0]));
  assert!((m.facts.site_lat_deg.unwrap() - 51.4769).abs() < 1e-6);
  assert!(m.facts.site_lon_deg.unwrap().abs() < 1e-9);
  assert!((m.facts.site_elev_m.unwrap() - 46.0).abs() < 1e-3);
}

#[test]
fn fits_scale_fallbacks_and_rotation() {
  // CDELT + CROTA2 = 30°: PA of +Y = -30° (CROTA2 rotates the axes counter-clockwise)
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 100, 50, &["CDELT1  =              -0.0001", "CDELT2  =               0.0001", "CROTA2  =                 30.0"], &[0.0; 5000]));
  assert_eq!(m.facts.pixel_scale_source.as_deref(), Some("CDELT"));
  assert!((m.facts.pixel_scale_arcsec.unwrap() - 0.36).abs() < 1e-9);
  assert!((m.facts.rotation_deg.unwrap() + 30.0).abs() < 1e-9);
  assert_eq!(m.facts.mirrored, Some(false));
  assert!((m.facts.fov_width_arcmin.unwrap() - 0.6).abs() < 1e-9);
  assert!((m.facts.fov_height_arcmin.unwrap() - 0.3).abs() < 1e-9);
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["PIXSCALE=                 0.25", "SECPIX  =                  0.3"], &[0.0]));
  assert_eq!((m.facts.pixel_scale_arcsec, m.facts.pixel_scale_source.as_deref()), (Some(0.25), Some("PIXSCALE")));
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &["SECPIX  =                  0.3"], &[0.0]));
  assert_eq!((m.facts.pixel_scale_arcsec, m.facts.pixel_scale_source.as_deref()), (Some(0.3), Some("SECPIX")));
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, &[], &[0.0]));
  assert!(m.facts.pixel_scale_arcsec.is_none() && m.facts.rotation_deg.is_none());
}

fn fixture_path() -> std::path::PathBuf {
  std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test_assets/fits_lt_67p_header_4x3.fits")
}

/// The header of the user's Liverpool Telescope image (data unit replaced by a 4 × 3 ramp).
#[test]
fn fits_liverpool_fixture_facts() {
  let bytes = std::fs::read(fixture_path()).unwrap();
  let (nx, ny, v, _) = parse_fits(&bytes).unwrap();
  assert_eq!((nx, ny), (4, 3));
  assert_eq!(v[0], 100.0);
  assert_eq!(v[11], 1200.0);
  let m = FitsMetadata::from_fits_bytes(&bytes);
  let f = &m.facts;
  assert_eq!((m.naxis1, m.naxis2, m.bitpix), (4, 3, 32));
  assert_eq!(m.cards[0].key, "SIMPLE");
  assert_eq!(m.cards.len(), 276); // every card before END, in order
  let last = m.cards.last().unwrap();
  assert_eq!((last.key.as_str(), last.value.as_str()), ("COMMENT", "JD and DATE-OBS are the average of 4 dates"));
  assert!(m.cards.iter().any(|c| c.key == "ROWORDER" && c.value == "BOTTOM-UP"));
  let lon = m.cards.iter().find(|c| c.key == "LONGITUD").unwrap();
  assert_eq!((lon.value.as_str(), lon.comment.as_str()), ("-17.8792000", "[degrees West] Observatory Longitude"));
  assert_eq!(f.object.as_deref(), Some("67P"));
  assert_eq!(f.telescope.as_deref(), Some("Liverpool Telescope"));
  assert_eq!(f.instrument.as_deref(), Some("IO:O"));
  assert_eq!(f.filter.as_deref(), Some("SDSS-G"));
  assert_eq!(f.exposure_s, Some(20.0));
  assert_eq!(f.date_obs.as_deref(), Some("2021-12-27T00:14:47.345"));
  // the comment says "UT, averaged": already mid-exposure
  assert!(f.date_is_mid);
  assert_eq!(f.date_mid_utc.as_deref(), Some("2021-12-27T00:14:47.345Z"));
  assert!(f.date_mid_tdb_centuries.is_some());
  // scale from the CD matrix
  assert_eq!(f.pixel_scale_source.as_deref(), Some("CD"));
  assert!((f.pixel_scale_arcsec.unwrap() - 0.3037).abs() < 1e-3, "{:?}", f.pixel_scale_arcsec);
  assert!((f.rotation_deg.unwrap() - 0.1361).abs() < 1e-3, "{:?}", f.rotation_deg);
  assert_eq!(f.mirrored, Some(false));
  assert!((f.crval_ra_deg.unwrap() - 135.563400742).abs() < 1e-9);
  assert!((f.crval_dec_deg.unwrap() - 28.156923795).abs() < 1e-9);
  assert!(f.center_ra_deg.is_some() && f.wcs_error.is_none());
  assert!((f.pointing_ra_deg.unwrap() - 135.5634).abs() < 1e-3);
  assert_eq!(f.radesys.as_deref(), Some("FK5"));
  // LONGITUD -17.8792 "[degrees West]": the LST overrides the comment
  assert_eq!(f.site_lat_deg, Some(28.7624));
  assert_eq!(f.site_lon_raw_deg, Some(-17.8792));
  // the "[degrees West]" comment is wrong: the LST (5:10:46 at UTSTART 00:07:29) only matches
  // the raw value as east positive, La Palma at 17.88° W
  assert!(f.site_lon_from_lst);
  assert!(!f.site_lon_west_convention);
  assert_eq!(f.site_lon_deg, Some(-17.8792));
  assert!(f.site_elev_m.is_none());
  let j = m.to_json();
  assert!(j.contains("\"siteLonDeg\":-17.8792"));
  assert!(j.contains("\"siteLonFromLst\":true"));
  assert!(j.contains("\"object\":\"67P\""));
}

/// The user's real file (not in the repo): `cargo nextest run -p aethervk-core-rlib
/// fits_real_file --run-ignored all`.
#[test]
#[ignore]
fn fits_real_file_facts() {
  let p = "Y:/Downloads/67p 20211225+26+27+28 Liverpool2m0 g+r+i+z 1x-02.fit";
  let Ok(bytes) = std::fs::read(p) else {
    std::println!("skipping: {p} not found");
    return;
  };
  let (nx, ny, v, _) = parse_fits(&bytes).unwrap();
  let m = FitsMetadata::from_fits_bytes(&bytes);
  let (b, w) = black_white(&v, nx, ny, StretchMode::ZScale);
  std::println!("{nx}x{ny} zscale {b} .. {w}");
  for p in [0.0, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0] {
    std::println!("  p{p}: {:?}", super::percentile(&v, p));
  }
  std::println!("  nan: {}", v.iter().filter(|x| !x.is_finite()).count());
  std::println!("{:#?}", m.facts);
  assert_eq!((nx, ny), (1200, 1200));
  assert!(w > b);
}
