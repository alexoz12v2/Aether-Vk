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

/// The header of the user's Liverpool Telescope image (data unit replaced by a 4 × 3 ramp).
#[test]
fn fits_liverpool_fixture_facts() {
  let bytes = std::fs::read(aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits")).unwrap();
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
  // the averaged DATE-OBS is also the start the model is run from; the card is reported
  assert_eq!(f.date_start_utc.as_deref(), Some("2021-12-27T00:14:47.345Z"));
  assert_eq!(f.date_start_tdb_us, f.date_mid_tdb_us);
  assert!(!f.date_only && !f.date_is_file_date);
  assert_eq!(f.date_card_raw.as_deref(), Some("DATE-OBS= '2021-12-27T00:14:47.345' / UT, averaged"));
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

// ─── read_fits: CFITSIO path against the pure-Rust parser ────────────────────

use aethervk_cfitsio_rlib::MemFitsWriter;

/// `100, 200, …, 1200`: the 4 × 3 ramp of the committed fixture.
const RAMP: [f32; 12] =
  [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0];

/// A FITS file written by CFITSIO in memory (the writer's session ends before this returns, so
/// the bytes can be read back through `read_fits`).
fn written(build: impl FnOnce(&mut MemFitsWriter) -> Result<(), aethervk_cfitsio_rlib::FitsError>) -> Vec<u8> {
  let mut w = MemFitsWriter::new().unwrap();
  build(&mut w).unwrap();
  w.finish().unwrap()
}

/// 4 × 3 BITPIX 16 ramp in the primary HDU.
fn written_primary() -> Vec<u8> {
  written(|w| {
    w.create_image(16, &[4, 3])?;
    w.write_pixels_f32(&RAMP)
  })
}

/// `NAXIS = 0` primary + the same ramp as an `IMAGE` extension.
fn written_extension() -> Vec<u8> {
  written(|w| {
    w.create_image(8, &[])?;
    w.create_image(16, &[4, 3])?;
    w.write_pixels_f32(&RAMP)
  })
}

/// The same ramp Rice tile-compressed (one tile per row): an empty primary + a BINTABLE.
fn written_rice() -> Vec<u8> {
  written(|w| {
    w.set_compression(aethervk_cfitsio_rlib::writer::RICE_1, &[4, 1])?;
    w.create_image(16, &[4, 3])?;
    let shorts: Vec<i16> = RAMP.iter().map(|&v| v as i16).collect();
    w.write_pixels_i16(&shorts)
  })
}

/// A 4 × 3 × 2 BITPIX 16 cube: plane 0 is the ramp, plane 1 the ramp + 5000.
fn written_cube() -> Vec<u8> {
  written(|w| {
    w.create_image(16, &[4, 3, 2])?;
    let mut v: Vec<i16> = RAMP.iter().map(|&x| x as i16).collect();
    v.extend(RAMP.iter().map(|&x| x as i16 + 5000));
    w.write_pixels(&v)
  })
}

/// A 2 × 1 image whose header carries a lowercase keyword (cfitsio uppercases keywords it
/// writes, so the card is patched into the bytes).
fn lowercase_keyword_file() -> Vec<u8> {
  let mut bytes = written(|w| {
    w.create_image(16, &[2, 1])?;
    w.write_record("LOWERKEY=                    7 / lowercase keyword")?;
    w.write_pixels(&[1i16, 2])
  });
  let i = bytes.windows(9).position(|c| c == b"LOWERKEY=").unwrap();
  bytes[i..i + 8].copy_from_slice(b"lowerkey");
  bytes
}

/// Every FITS test asset (`.fit`, `.fits`, `.fts`, `.fz`): the `fits_fixtures` group of
/// `test_assets/ASSETS.tsv` and the FITS textures of the review save, resolved (downloaded on
/// demand) through `aethervk_test_assets`.
#[cfg(feature = "cfitsio")]
fn fixture_files() -> Result<Vec<(String, Vec<u8>)>, String> {
  fn walk(dir: &std::path::Path, out: &mut Vec<(String, Vec<u8>)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
      if p.is_dir() {
        walk(&p, out);
        continue;
      }
      let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
      if matches!(ext.as_str(), "fit" | "fits" | "fts" | "fz") {
        out.push((p.display().to_string(), std::fs::read(&p).unwrap()));
      }
    }
  }
  let mut out = Vec::new();
  for rel in [
    "fits_lt_67p_header_4x3.fits",
    "fits_image_extension_4x3.fits",
    "fits_tile_compressed_4x3.fz",
    "fits_cube_4x3x3.fits",
    "fits_cube_4x3x3.fz",
    "fits_date_variants",
    "saves/review_67p_2021-12/assets/textures",
  ] {
    let p = aethervk_test_assets::path(rel).map_err(|e| std::format!("{rel}: {e}"))?;
    if p.is_dir() {
      walk(&p, &mut out);
    } else {
      out.push((p.display().to_string(), std::fs::read(&p).unwrap()));
    }
  }
  assert!(out.len() >= 15, "{} FITS test assets", out.len());
  Ok(out)
}

/// The in-memory images of this module's tests and of `sky_image::tests`, as `(name, bytes)`.
#[cfg(feature = "cfitsio")]
pub(crate) fn legacy_corpus() -> Vec<(String, Vec<u8>)> {
  let mut out: Vec<(String, Vec<u8>)> = alloc::vec![
    ("bitpix 8 bzero bscale blank".into(), fits_bytes(8, 3, 1, &["BZERO   =                 10.0", "BSCALE  =                  0.5", "BLANK   =                  255"], &[0.0, 4.0, 255.0])),
    ("bitpix 16 unsigned blank".into(), fits_bytes(16, 3, 1, &["BZERO   =              32768.0", "BSCALE  =                  1.0", "BLANK   =               -32768"], &[-32767.0, 0.0, -32768.0])),
    ("bitpix 32 bzero bscale blank".into(), fits_bytes(32, 3, 1, &["BZERO   =                 -5.0", "BSCALE  =                  2.0", "BLANK   =          -2147483648"], &[100000.0, -3.0, -2147483648.0])),
    ("bitpix -32 scaled nan".into(), fits_bytes(-32, 3, 1, &["BZERO   =                  1.0", "BSCALE  =                  3.0"], &[0.5, -2.0, f64::NAN])),
    ("bitpix -64 scaled nan".into(), fits_bytes(-64, 3, 1, &["BZERO   =                  0.0", "BSCALE  =                 1D-1"], &[123.0, -7.0, f64::NAN])),
    ("bitpix 64 plain".into(), fits_bytes(64, 2, 2, &[], &[1.0, -2.0, 3.0e9, -4.0e12])),
    ("cards with quotes hierarch commentary".into(), fits_bytes(16, 1, 1, &["OBJECT  = 'M 31 ''x'' '         / the target / with slash", "COMMENT   free text here", "EXPTIME =                 30.5 / [s] exposure", "HIERARCH ESO DET DIT = 1.5 / det", "HISTORY   step 1"], &[0.0])),
    ("date space separator".into(), fits_bytes(16, 1, 1, &["DATE-OBS= '2021-12-27 00:14:47.345'", "EXPTIME =                 20.0"], &[0.0])),
    ("date only plus utstart".into(), fits_bytes(16, 1, 1, &["DATE-OBS= '2021-12-25'", "UTSTART = '00:07:28.858'"], &[0.0])),
    ("mjd-obs".into(), fits_bytes(16, 1, 1, &["MJD-OBS =              59573.5"], &[0.0])),
    ("mjd".into(), fits_bytes(16, 1, 1, &["MJD     =              59573.25"], &[0.0])),
    ("no cards".into(), fits_bytes(16, 1, 1, &[], &[0.0])),
    ("site west comment".into(), fits_bytes(16, 1, 1, &["LATITUDE=           28.7624000 / [degrees] Observatory Latitude", "LONGITUD=           17.8792000 / [degrees West] Observatory Longitude"], &[0.0])),
    ("site east positive".into(), fits_bytes(16, 1, 1, &["SITELAT = '+19:49:36'", "SITELONG=          -155.4719 / East longitude (west negative)", "SITEELEV=               4204.0"], &[0.0])),
    ("site W suffix".into(), fits_bytes(16, 1, 1, &["LAT-OBS =                 32.0", "LONG-OBS= '110:56:00 W'"], &[0.0])),
    ("cdelt crota".into(), fits_bytes(16, 100, 50, &["CDELT1  =              -0.0001", "CDELT2  =               0.0001", "CROTA2  =                 30.0"], &[0.0; 5000])),
    ("pixscale secpix".into(), fits_bytes(16, 1, 1, &["PIXSCALE=                 0.25", "SECPIX  =                  0.3"], &[0.0])),
    ("asset ramp with blank".into(), {
      let mut v: Vec<f64> = (0..12).map(|i| (i * 100) as f64).collect();
      v[1] = -1.0;
      fits_bytes(16, 4, 3, &["BLANK   =                   -1", "OBJECT  = 'ramp    '"], &v)
    }),
    ("escaped double quote".into(), fits_bytes(16, 4, 3, &["OBJECT  = 'Test \"q\"'          / a comment"], &(0..12).map(|i| (i * 100) as f64).collect::<Vec<_>>())),
    ("roworder top-down".into(), fits_bytes(16, 2, 2, &["ROWORDER= 'TOP-DOWN'"], &[1.0, 2.0, 3.0, 4.0])),
    ("non-ascii byte in a card".into(), {
      let mut b = fits_bytes(16, 1, 1, &["OBJECT  = 'caf?'"], &[0.0]);
      let i = b.windows(4).position(|w| w == b"caf?").unwrap();
      b[i + 3] = 0xe9;
      b
    }),
    ("written by cfitsio".into(), written_primary()),
    ("unpadded data unit".into(), {
      let mut b = written_primary();
      b.truncate(2880 + 12 * 2);
      b
    }),
    ("cfitsio uint16 (BZERO 32768)".into(), written(|w| {
      w.create_image(20, &[3, 1])?;
      w.write_pixels(&[0u16, 65535, 32768])
    })),
    ("cfitsio uint32 (BZERO 2147483648)".into(), written(|w| {
      w.create_image(40, &[3, 1])?;
      w.write_pixels(&[0u32, 4_294_967_295, 2_147_483_648])
    })),
    ("cfitsio int8 (BZERO -128)".into(), written(|w| {
      w.create_image(10, &[3, 1])?;
      w.write_pixels(&[-128i8, 0, 127])
    })),
    ("cfitsio bscale bzero blank on raw shorts".into(), written(|w| {
      w.create_image(16, &[4, 1])?;
      w.write_key_f64("BSCALE", 0.5, 6, "")?;
      w.write_key_f64("BZERO", 10.0, 6, "")?;
      w.write_key_i64("BLANK", -32768, "undefined")?;
      w.set_scale(1.0, 0.0)?;
      w.write_pixels(&[0i16, 4, -2, -32768])
    })),
    ("cfitsio int64 beyond f32 precision".into(), written(|w| {
      w.create_image(64, &[3, 1])?;
      w.write_pixels(&[1i64 << 53 | 1, -(1i64 << 40) - 3, 16_777_217])
    })),
    ("cfitsio double with nan".into(), written(|w| {
      w.create_image(-64, &[3, 1])?;
      w.write_pixels(&[1.5f64, f64::NAN, -2.0e-300])
    })),
    ("cfitsio header conventions".into(), written(|w| {
      w.create_image(16, &[2, 1])?;
      w.write_comment("first comment")?;
      w.write_history("first history")?;
      w.write_record("HIERARCH ESO DET DIT = 1.5 / detector integration time")?;
      w.write_long_string("LONGKEY", &"x".repeat(150), "joined over CONTINUE cards")?;
      w.write_comment("second comment")?;
      w.write_record("LOWERKEY=                    7 / lowercase keyword")?;
      w.write_key_str("OBJECT", "it's", "an escaped quote")?;
      w.write_key_bool("FLAG", true, "a logical")?;
      w.write_history("second history")?;
      w.write_pixels(&[1i16, 2])
    })),
    ("cfitsio lowercase keyword patched in".into(), lowercase_keyword_file()),
    ("cfitsio trailing garbage after the data".into(), {
      let mut b = written_primary();
      b.extend_from_slice(b"garbage after the last block, not a header");
      b
    }),
    ("cfitsio cube 4x3x2 (legacy refuses, cfitsio reads plane 0)".into(), written_cube()),
    ("cfitsio gzip (legacy refuses)".into(), aethervk_cfitsio_rlib::gzip(&written_primary()).unwrap()),
  ];
  for (name, bytes) in super::tests::fits_corpus() {
    out.push((format!("sky_image tests: {name}"), bytes));
  }
  out
}

/// Pixel bits with every NaN mapped to one pattern.
#[cfg(feature = "cfitsio")]
fn canonical_bits(v: &[f32]) -> Vec<u32> {
  v.iter().map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() }).collect()
}

/// The permanent oracle: on every committed fixture and every in-memory test image, CFITSIO
/// gives the same size, bit-identical pixels, the same header, the same cards and the same
/// metadata JSON as the pure-Rust parser.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_cfitsio_matches_the_legacy_parser_on_every_fixture() {
  let mut compared = 0;
  let fixtures = aethervk_test_assets::skip_on_err!(fixture_files());
  for (name, bytes) in fixtures.into_iter().chain(legacy_corpus()) {
    let legacy = super::parse_fits_legacy(&bytes);
    let through = read_fits(&bytes);
    let (nx, ny, data, header) = match (legacy, &through) {
      (Ok(l), Ok(_)) => l,
      // fixtures the legacy parser cannot read (extensions, .fz) only have to decode
      (Err(e), Ok(_)) => {
        std::println!("{name}: legacy parser refuses it ({e}); cfitsio reads it");
        continue;
      }
      (l, t) => panic!("{name}: legacy {l:?} vs cfitsio {t:?}"),
    };
    let (img, meta) = through.unwrap();
    assert_eq!((img.nx, img.ny), (nx, ny), "{name}: size");
    assert_eq!(canonical_bits(&img.data), canonical_bits(&data), "{name}: pixels");
    assert_eq!(img.header, header, "{name}: header");
    assert_eq!((img.hdu, img.compressed), (1, false), "{name}: hdu");
    let legacy_meta = FitsMetadata::from_cards(parse_cards(&bytes));
    assert_eq!(meta.cards, legacy_meta.cards, "{name}: cards");
    assert_eq!(meta, legacy_meta, "{name}: metadata");
    assert_eq!(meta.to_json(), legacy_meta.to_json(), "{name}: json");
    compared += 1;
  }
  assert!(compared >= 30, "{compared} inputs compared");
}

/// The inputs the pure-Rust parser refuses are refused by the CFITSIO path too.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_cfitsio_refuses_what_the_legacy_parser_refuses() {
  let mut truncated = fits_bytes(-32, 2, 2, &[], &[0.0; 4]);
  truncated.truncate(2880);
  let mut no_image = fits_bytes(-32, 2, 2, &[], &[]);
  let i = no_image.windows(30).position(|w| w == b"NAXIS   =                    2").unwrap();
  no_image[i + 29] = b'0';
  for (name, b) in [
    ("garbage", b"NOT A FITS FILE".to_vec()),
    ("empty", Vec::new()),
    ("truncated data", truncated),
    ("NAXIS = 0 alone", no_image),
    ("BITPIX 24", fits_bytes(24, 1, 1, &[], &[0.0])),
  ] {
    assert!(super::parse_fits_legacy(&b).is_err(), "{name}: legacy");
    let r = read_fits(&b);
    assert!(r.is_err(), "{name}: cfitsio gave {:?}", r.map(|(i, _)| (i.nx, i.ny)));
  }
}

/// An image in an `IMAGE` extension behind a `NAXIS = 0` primary is found: the pixels of its
/// primary-HDU twin, the extension's cards first and the primary's after them.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_image_in_an_extension_is_found() {
  let (twin, _) = read_fits(&written_primary()).unwrap();
  let (img, meta) = read_fits(&written_extension()).unwrap();
  assert_eq!((img.nx, img.ny, img.hdu, img.compressed), (4, 3, 2, false));
  assert_eq!(img.data, twin.data);
  assert_eq!((meta.naxis1, meta.naxis2, meta.bitpix, meta.hdu, meta.compressed), (4, 3, 16, 2, false));
  assert_eq!(meta.cards[0].key, "XTENSION");
  assert_eq!(meta.cards[0].value, "IMAGE");
  // the primary's own cards (SIMPLE, EXTEND: keywords an extension never has) follow
  let simple = meta.cards.iter().position(|c| c.key == "SIMPLE").expect("primary cards appended");
  assert!(simple > 0);
  assert!(meta.cards.iter().any(|c| c.key == "EXTEND"));
  // the extension's keywords win in the header, the primary-only ones are kept
  assert_eq!(img.header.get("NAXIS1"), Some("4"));
  assert_eq!(img.header.get("NAXIS"), Some("2"));
  assert_eq!(img.header.get("SIMPLE"), Some("T"));
  assert!(meta.to_json().contains("\"bitpix\":16,\"hdu\":2,\"compressed\":false,"));
  // the thin wrappers see it too
  let (nx, ny, v, _) = parse_fits(&written_extension()).unwrap();
  assert_eq!((nx, ny), (4, 3));
  assert_eq!(v, twin.data);
  assert_eq!(FitsMetadata::from_fits_bytes(&written_extension()).hdu, 2);
}

/// A Rice tile-compressed image decodes to the pixels of its uncompressed twin, with the header
/// of the image it stands for.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_tile_compressed_image_decodes() {
  let (twin, _) = read_fits(&written_primary()).unwrap();
  let (img, meta) = read_fits(&written_rice()).unwrap();
  assert_eq!((img.nx, img.ny, img.hdu, img.compressed), (4, 3, 2, true));
  assert_eq!(img.data, twin.data);
  assert_eq!((meta.naxis1, meta.naxis2, meta.bitpix, meta.compressed), (4, 3, 16, true));
  assert!(meta.cards.iter().any(|c| c.key == "NAXIS1" && c.value == "4"));
  assert!(!meta.cards.iter().any(|c| c.key.starts_with('Z')), "{:?}", meta.cards);
  assert!(meta.to_json().contains("\"hdu\":2,\"compressed\":true,"));
}

/// A cube gives its first plane, reports the plane count in the struct and the JSON; the
/// pure-Rust parser keeps refusing cubes with its message.
#[test]
fn fits_cube_reads_its_first_plane() {
  let bytes = written_cube();
  #[cfg(feature = "cfitsio")]
  {
    let (img, meta) = read_fits(&bytes).unwrap();
    assert_eq!((img.nx, img.ny, img.hdu, img.compressed, img.planes, img.plane), (4, 3, 1, false, 2, 0));
    assert_eq!(img.data, RAMP);
    assert_eq!((meta.naxis1, meta.naxis2, meta.bitpix, meta.planes, meta.plane), (4, 3, 16, 2, 0));
    assert!(meta.to_json().contains("\"compressed\":false,\"planes\":2,\"plane\":0,\"cards\":["), "{}", meta.to_json());
    assert!(meta.cards.iter().any(|c| c.key == "NAXIS3" && c.value == "2"));
    assert_eq!(img.header.get("NAXIS"), Some("3"));
    // a 4-D cube with a singleton axis: planes multiply
    let four = written(|w| {
      w.create_image(16, &[4, 3, 2, 3])?;
      let v: Vec<i16> = (0..72).collect();
      w.write_pixels(&v)
    });
    let (img, meta) = read_fits(&four).unwrap();
    assert_eq!((img.planes, meta.planes), (6, 6));
    assert_eq!(img.data, (0..12).map(|i| i as f32).collect::<Vec<_>>());
    // a tile-compressed cube
    let fz = written(|w| {
      w.set_compression(aethervk_cfitsio_rlib::writer::RICE_1, &[4, 3, 1])?;
      w.create_image(16, &[4, 3, 2])?;
      let mut v: Vec<i16> = RAMP.iter().map(|&x| x as i16).collect();
      v.extend(RAMP.iter().map(|&x| x as i16 + 5000));
      w.write_pixels(&v)
    });
    let (img, _) = read_fits(&fz).unwrap();
    assert_eq!((img.compressed, img.planes, img.hdu), (true, 2, 2));
    assert_eq!(img.data, RAMP);
    // the primary-cards metadata of a cube header says the same planes
    assert_eq!(FitsMetadata::from_cards(parse_cards(&bytes)).planes, 2);
  }
  // the pure-Rust 2-D parser itself still refuses a cube; `read_fits` hands it one plane
  let e = super::parse_fits_legacy(&bytes).unwrap_err();
  assert!(e.contains("only 2-D images"), "{e}");
  #[cfg(not(feature = "cfitsio"))]
  {
    let (img, meta) = read_fits(&bytes).unwrap();
    assert_eq!((img.planes, img.plane, meta.planes, img.data.clone()), (2, 0, 2, RAMP.to_vec()));
    assert_eq!(img.header.get("NAXIS3"), Some("2"), "the file's header, not the 2-D view");
  }
}

/// The committed 4 × 3 × 3 cube fixture (plane p = the ramp + 1000 p) and its tile-compressed
/// twin, written by the generator below.
fn written_cube_4x3x3(compressed: bool) -> Vec<u8> {
  written(|w| {
    if compressed {
      w.set_compression(aethervk_cfitsio_rlib::writer::RICE_1, &[4, 3, 1])?;
    }
    w.create_image(16, &[4, 3, 3])?;
    w.write_key_str("OBJECT", "cube", "three planes")?;
    let v: Vec<i16> = (0..3).flat_map(|p| RAMP.iter().map(move |&x| x as i16 + 1000 * p)).collect();
    w.write_pixels(&v)
  })
}

fn cube_plane(p: u64) -> Vec<f32> {
  RAMP.iter().map(|&x| x + 1000.0 * p as f32).collect()
}

/// `read_fits_plane` gives every plane of a cube by its flat index, in both modes with the same
/// values (so the CFITSIO and pure-Rust paths agree bit for bit on every plane): the 3-D fixture
/// and its `.fz` twin, a 4-D cube (`planeAxes` in the JSON, `NAXIS3` fastest), and the errors
/// past the last plane (a plain image has plane 0 only).
#[test]
fn fits_cube_planes_by_index() {
  let fixture = std::fs::read(aethervk_test_assets::test_asset!("fits_cube_4x3x3.fits")).unwrap();
  for (name, bytes) in [("generated", written_cube_4x3x3(false)), ("fixture", fixture)] {
    for p in 0..3u64 {
      let (img, meta) = read_fits_plane(&bytes, p).unwrap_or_else(|e| panic!("{name} plane {p}: {e}"));
      assert_eq!((img.nx, img.ny, img.planes, img.plane), (4, 3, 3, p), "{name}");
      assert_eq!(img.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), cube_plane(p).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{name} plane {p}");
      assert_eq!((meta.planes, meta.plane, meta.plane_axes.clone()), (3, p, Vec::new()), "{name}");
      assert!(meta.to_json().contains(&format!("\"planes\":3,\"plane\":{p},\"cards\":[")), "{name}");
      assert_eq!(img.header.get("NAXIS3"), Some("3"), "{name}");
      assert_eq!(meta.facts.object.as_deref(), Some("cube"), "{name}");
    }
    let e = read_fits_plane(&bytes, 3).unwrap_err();
    assert!(e.contains("plane 3 out of range") && e.contains("3 plane(s)"), "{name}: {e}");
    assert!(read_fits_plane(&bytes, u64::MAX).is_err(), "{name}");
    assert_eq!(read_fits(&bytes).unwrap(), read_fits_plane(&bytes, 0).unwrap(), "{name}");
  }
  // a plain image: plane 0 only
  let plain = written_primary();
  let e = read_fits_plane(&plain, 1).unwrap_err();
  assert!(e.contains("plane 1 out of range") && e.contains("1 plane(s)"), "{e}");
  // 4 × 3 × 2 × 3: plane index = i3 + 2 · i4, pixel value = 12 · plane + i
  let four = written(|w| {
    w.create_image(16, &[4, 3, 2, 3])?;
    let v: Vec<i16> = (0..72).collect();
    w.write_pixels(&v)
  });
  for p in 0..6u64 {
    let (img, meta) = read_fits_plane(&four, p).unwrap();
    assert_eq!(img.data, (0..12).map(|i| (12 * p as i32 + i) as f32).collect::<Vec<_>>(), "plane {p}");
    assert_eq!((img.planes, meta.plane_axes.clone()), (6, vec![2, 3]));
    assert!(meta.to_json().contains(&format!("\"planes\":6,\"plane\":{p},\"planeAxes\":[2,3],\"cards\":[")), "{}", meta.to_json());
  }
  assert!(read_fits_plane(&four, 6).unwrap_err().contains("6 plane(s)"));
  assert_eq!(FitsMetadata::from_cards(parse_cards(&four)).plane_axes, vec![2, 3]);
  // a truncated cube: the first plane reads, the missing one is "FITS data truncated"
  let mut cut = written_cube_4x3x3(false);
  cut.truncate(2880 + 2 * 12 * 2);
  assert_eq!(read_fits_plane(&cut, 1).unwrap().0.data, cube_plane(1));
  assert!(read_fits_plane(&cut, 2).unwrap_err().contains("truncated"));
  // the tile-compressed twin (CFITSIO; the pure-Rust parser reads the primary HDU only)
  let fz = std::fs::read(aethervk_test_assets::test_asset!("fits_cube_4x3x3.fz")).unwrap();
  #[cfg(feature = "cfitsio")]
  for p in 0..3u64 {
    let (img, meta) = read_fits_plane(&fz, p).unwrap();
    assert_eq!((img.compressed, img.hdu, img.planes, img.plane, meta.plane), (true, 2, 3, p, p));
    assert_eq!(img.data, cube_plane(p), "fz plane {p}");
  }
  #[cfg(feature = "cfitsio")]
  assert!(read_fits_plane(&fz, 3).unwrap_err().contains("out of range"));
  #[cfg(not(feature = "cfitsio"))]
  assert!(read_fits_plane(&fz, 0).unwrap_err().contains("extensions are not read"));
}

/// A data unit written without its last block's padding (the CDS DSS2 HiPS tiles: 2880 + 512 ·
/// 512 · 2 bytes) reads whole in both modes; a data unit short of real pixels is still refused.
#[test]
fn fits_unpadded_data_unit_is_read() {
  let full = written_primary();
  let mut short = full.clone();
  short.truncate(2880 + 12 * 2);
  assert_ne!(short.len() % 2880, 0);
  let (a, ma) = read_fits(&full).unwrap();
  let (b, mb) = read_fits(&short).unwrap();
  assert_eq!((a.data, ma.cards), (b.data, mb.cards));
  let mut cut = full.clone();
  cut.truncate(2880 + 12 * 2 - 1);
  assert!(read_fits(&cut).unwrap_err().contains("truncated"), "{}", read_fits(&cut).unwrap_err());
}

/// A gzip-compressed file is inflated and read (cfitsio mode); the other whole-file
/// compressions and the pure-Rust mode refuse with a message naming the compression.
#[test]
fn fits_gzip_whole_file_compression() {
  let plain = written_primary();
  let gz = aethervk_cfitsio_rlib::gzip(&plain).unwrap();
  assert!(gz.starts_with(&[0x1f, 0x8b]));
  #[cfg(feature = "cfitsio")]
  {
    let (twin, twin_meta) = read_fits(&plain).unwrap();
    let (img, meta) = read_fits(&gz).unwrap();
    assert_eq!(img, twin);
    assert_eq!(meta, twin_meta);
    // a gzipped image extension and a gzipped .fz
    let (img, _) = read_fits(&aethervk_cfitsio_rlib::gzip(&written_extension()).unwrap()).unwrap();
    assert_eq!((img.hdu, img.data.clone()), (2, RAMP.to_vec()));
    let (img, _) = read_fits(&aethervk_cfitsio_rlib::gzip(&written_rice()).unwrap()).unwrap();
    assert_eq!((img.compressed, img.data.clone()), (true, RAMP.to_vec()));
    // a broken stream
    let e = read_fits(&gz[..gz.len() / 2]).unwrap_err();
    assert!(e.contains("gzip (.fits.gz)-compressed FITS could not be inflated"), "{e}");
    assert!(FitsMetadata::from_fits_bytes(&gz).cards.iter().any(|c| c.key == "SIMPLE"));
  }
  #[cfg(not(feature = "cfitsio"))]
  {
    let e = read_fits(&gz).unwrap_err();
    assert!(e.contains("gzip (.fits.gz)-compressed") && e.contains("decompress it first"), "{e}");
  }
  // corrupt .Z / .bz2 streams name the compression
  let e = read_fits(b"\x1f\x9d\x90SIMPLE").unwrap_err();
  assert!(e.contains("Unix compress (.Z)"), "{e}");
  let e = read_fits(b"BZh91AY&SY").unwrap_err();
  assert!(e.contains("bzip2 (.bz2)"), "{e}");
}

/// `.fits.bz2` and `.Z` (Unix compress) files inflate and read in cfitsio mode (the pure-Rust
/// mode refuses them by name); the `.Z` fixture is the committed twin of the 4 × 3 header file.
#[test]
fn fits_bzip2_and_unix_compress_whole_file() {
  let plain = written_primary();
  let bz = aethervk_cfitsio_rlib::bzip2_compress(&plain).unwrap();
  let z = std::fs::read(aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits.Z")).unwrap();
  let fixture = std::fs::read(aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits")).unwrap();
  #[cfg(feature = "cfitsio")]
  {
    let (twin, twin_meta) = read_fits(&plain).unwrap();
    let (img, meta) = read_fits(&bz).unwrap();
    assert_eq!((img, meta), (twin, twin_meta));
    let (twin, twin_meta) = read_fits(&fixture).unwrap();
    let (img, meta) = read_fits(&z).unwrap();
    assert_eq!((img.nx, img.ny, img.data.clone()), (4, 3, RAMP.to_vec()));
    assert_eq!((img, meta), (twin, twin_meta));
    assert!(FitsMetadata::from_fits_bytes(&z).facts.object.is_some());
  }
  #[cfg(not(feature = "cfitsio"))]
  {
    assert!(read_fits(&bz).unwrap_err().contains("bzip2 (.bz2)-compressed"));
    assert!(read_fits(&z).unwrap_err().contains("Unix compress (.Z)-compressed"));
  }
}

/// A `BLANK` card written as a float (`BLANK = -32768.0`, as the CDS HipsGen DSS2 tiles have
/// it) is read as its integer: the blank pixels are NaN, the others their values.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_float_blank_card_is_tolerated() {
  for (text, expect_nan) in [("-32768.0", true), ("-32768.5", false)] {
    let bytes = written(|w| {
      w.create_image(16, &[3, 1])?;
      w.write_record(&format!("BLANK   = {text:>20} / written as a float"))?;
      w.write_pixels(&[5i16, -32768, 7])
    });
    let (img, meta) = read_fits(&bytes).unwrap();
    assert_eq!((img.data[0], img.data[2]), (5.0, 7.0), "{text}");
    assert_eq!(img.data[1].is_nan(), expect_nan, "{text}: {:?}", img.data);
    assert!(meta.cards.iter().any(|c| c.key == "BLANK" && c.value.starts_with("-32768.")), "{text}");
  }
  // a tile-compressed twin
  let bytes = written(|w| {
    w.set_compression(aethervk_cfitsio_rlib::writer::RICE_1, &[3, 1])?;
    w.create_image(16, &[3, 1])?;
    w.write_record("BLANK   =             -32768.0 / written as a float")?;
    w.write_pixels(&[5i16, -32768, 7])
  });
  let (img, _) = read_fits(&bytes).unwrap();
  assert!(img.compressed);
  assert!(img.data[1].is_nan(), "{:?}", img.data);
}

/// Image selection: empty primaries, zero-size and 1-D images and tables are skipped for the
/// first real 2-D image; a tables-only file and a random-groups file are refused by name.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_image_selection_skips_what_is_not_a_2d_image() {
  let bytes = written(|w| {
    w.create_image(8, &[])?; // empty primary
    w.create_image(16, &[5])?; // 1-D
    w.write_pixels(&[1i16, 2, 3, 4, 5])?;
    w.create_table(true, 2, &[("FLUX", "1E")])?;
    w.write_column(1, 1, &[1.0f32, 2.0])?;
    w.create_image(16, &[4, 3])?; // the one
    w.write_pixels(&RAMP)?;
    w.create_image(16, &[2, 2])?; // a later image is not preferred
    w.write_pixels(&[9i16, 9, 9, 9])
  });
  let (img, meta) = read_fits(&bytes).unwrap();
  assert_eq!((img.nx, img.ny, img.hdu, meta.hdu), (4, 3, 4, 4));
  assert_eq!(img.data, RAMP);
  // tables only
  let tables = written(|w| {
    w.create_image(8, &[])?;
    w.create_table(true, 2, &[("FLUX", "1E")])?;
    w.create_table(false, 1, &[("NAME", "A8")])
  });
  let e = read_fits(&tables).unwrap_err();
  assert_eq!(e, "no image HDU (tables only)");
  // a header-only file (NAXIS = 0 alone)
  let empty = written(|w| w.create_image(8, &[]));
  assert_eq!(read_fits(&empty).unwrap_err(), "no image HDU (NAXIS >= 2) in the file");
  // a 1-D image alone
  let one_d = written(|w| {
    w.create_image(16, &[5])?;
    w.write_pixels(&[1i16, 2, 3, 4, 5])
  });
  assert_eq!(read_fits(&one_d).unwrap_err(), "no image HDU (NAXIS >= 2) in the file");
  // random groups
  let groups = written(|w| w.create_random_groups(16, &[4, 3], 1, 2));
  let e = read_fits(&groups).unwrap_err();
  assert!(e.starts_with("random-groups file"), "{e}");
  // the header of a refused file is still shown
  assert!(FitsMetadata::from_fits_bytes(&groups).cards.iter().any(|c| c.key == "GROUPS"));
  // a zero-size axis is skipped too
  let zero = fits_bytes(16, 0, 3, &[], &[]);
  let i = zero.windows(30).position(|w| w == b"NAXIS1  =                    0").unwrap();
  assert!(i > 0);
  assert_eq!(read_fits(&zero).unwrap_err(), "no image HDU (NAXIS >= 2) in the file");
}

/// Every tile compression decodes through `read_fits` to the pixels of its uncompressed twin,
/// flagged `compressed`, with the translated header (BITPIX / NAXISn, no Z keywords).
#[cfg(feature = "cfitsio")]
#[test]
fn fits_every_tile_compression_decodes() {
  use aethervk_cfitsio_rlib::writer::{GZIP_1, GZIP_2, HCOMPRESS_1, PLIO_1, RICE_1, SUBTRACTIVE_DITHER_2};
  let ints: Vec<f32> = (0..128).map(|i| i as f32).collect();
  let mut decoded = 0;
  // (cfitsio 4.7.0 writes neither 64-bit compressed integers nor NOCOMPRESS tiles: the wrapper
  // tests assert both refusals; reading them is the library's, untested here)
  for ctype in [RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1, PLIO_1] {
    for bitpix in [8i32, 16, 32] {
      let bytes = written(|w| {
        w.set_compression(ctype, &[16, 4])?;
        w.create_image(bitpix, &[16, 8])?;
        match bitpix {
          8 => w.write_pixels(&ints.iter().map(|&v| v as u8).collect::<Vec<_>>()),
          16 => w.write_pixels(&ints.iter().map(|&v| v as i16).collect::<Vec<_>>()),
          _ => w.write_pixels(&ints.iter().map(|&v| v as i32).collect::<Vec<_>>()),
        }
      });
      let (img, meta) = read_fits(&bytes).unwrap();
      assert_eq!((img.nx, img.ny, img.hdu, img.compressed), (16, 8, 2, true), "{ctype} {bitpix}");
      assert_eq!(img.data, ints, "{ctype} {bitpix}");
      assert_eq!(meta.bitpix, bitpix, "{ctype} {bitpix}");
      assert!(img.header.get("ZCMPTYPE").is_none(), "{ctype} {bitpix}");
      assert_eq!(img.header.get_i64("NAXIS1"), Some(16), "{ctype} {bitpix}");
      decoded += 1;
    }
  }
  assert_eq!(decoded, 15);
  // quantised, dithered floats with a ZBLANK-carrying integer twin
  let floats: Vec<f32> = (0..128).map(|i| (i as f32) * 0.731 + 3.0).collect();
  let bytes = written(|w| {
    w.set_compression(RICE_1, &[16, 4])?;
    w.set_quantize_method(SUBTRACTIVE_DITHER_2)?;
    w.set_quantize_level(-0.01)?;
    w.set_dither_seed(5)?;
    w.create_image(-32, &[16, 8])?;
    w.write_pixels(&floats)
  });
  let (img, meta) = read_fits(&bytes).unwrap();
  assert_eq!((img.compressed, meta.bitpix), (true, -32));
  assert!(img.data.iter().zip(&floats).all(|(a, b)| (a - b).abs() <= 0.011), "{:?}", &img.data[..8]);
  let mut with_blank: Vec<i16> = (0..128).collect();
  with_blank[3] = -1;
  let bytes = written(|w| {
    w.set_compression(GZIP_2, &[16, 4])?;
    w.create_image(16, &[16, 8])?;
    w.write_key_i64("BLANK", -1, "")?;
    w.write_pixels(&with_blank)
  });
  let (img, meta) = read_fits(&bytes).unwrap();
  assert!(img.data[3].is_nan());
  assert_eq!(img.data[4], 4.0);
  assert!(meta.cards.iter().any(|c| c.key == "BLANK" && c.value == "-1"));
  assert!(!meta.cards.iter().any(|c| c.key == "ZBLANK"));
}

/// The geometry guards: a side over 2^20, a plane over 2^29 pixels (a hand-built header claiming
/// 30000 × 30000 with no data), a cube with planes past the file's end; the messages name the
/// limit.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_size_guards_refuse_with_a_message() {
  let claim = |nx: u32, ny: u32, extra: &[&str]| fits_bytes(16, nx, ny, extra, &[]);
  let e = read_fits(&claim(1 << 20 | 1, 2, &[])).unwrap_err();
  assert!(e.starts_with("bad image size") && e.contains("1048576"), "{e}");
  let e = read_fits(&claim(30_000, 30_000, &[])).unwrap_err();
  assert!(e.starts_with("image too large") && e.contains("536870912"), "{e}");
  // under the pixel cap but more than the file holds
  let e = read_fits(&claim(1000, 1000, &[])).unwrap_err();
  assert_eq!(e, "FITS data truncated");
  // a cube whose planes do not fit the file: the planes the file holds read (the first, and
  // the ones inside the data block), the ones past its end are refused
  let mut cube = fits_bytes(16, 4, 3, &["NAXIS3  =                  500"], &[0.0; 12]);
  let i = cube.windows(30).position(|w| w == b"NAXIS   =                    2").unwrap();
  cube[i + 29] = b'3';
  assert_eq!(cube.len(), 5760);
  assert_eq!(read_fits(&cube).unwrap().0.planes, 500);
  assert_eq!(read_fits_plane(&cube, 119).unwrap().0.plane, 119, "the last plane inside the data block");
  assert_eq!(read_fits_plane(&cube, 120).unwrap_err(), "FITS data truncated");
  assert_eq!(read_fits_plane(&cube, 499).unwrap_err(), "FITS data truncated");
  assert!(read_fits_plane(&cube, 500).unwrap_err().contains("out of range"));
  // the legacy parser's twin messages
  assert!(super::parse_fits_legacy(&claim(1 << 20 | 1, 2, &[])).unwrap_err().starts_with("bad image size"));
}

/// Header conventions through `read_fits` in both modes: HIERARCH keys without the prefix,
/// CONTINUE cards kept as cards, commentary order, a lowercase keyword kept as written, an
/// escaped quote, a logical, trailing garbage after the data, a blanked EXTEND card.
#[test]
fn fits_header_conventions_are_kept() {
  let bytes = written(|w| {
    w.create_image(16, &[2, 1])?;
    w.write_comment("first comment")?;
    w.write_history("first history")?;
    w.write_record("HIERARCH ESO DET DIT = 1.5 / detector integration time")?;
    w.write_long_string("LONGKEY", &"x".repeat(150), "joined over CONTINUE cards")?;
    w.write_comment("second comment")?;
    w.write_record("LOWERKEY=                    7 / lowercase keyword")?;
    w.write_key_str("OBJECT", "it's", "an escaped quote")?;
    w.write_key_bool("FLAG", true, "a logical")?;
    w.write_history("second history")?;
    w.write_pixels(&[1i16, 2])
  });
  let (img, meta) = read_fits(&bytes).unwrap();
  assert_eq!(img.data, [1.0, 2.0]);
  let keys: Vec<&str> = meta.cards.iter().map(|c| c.key.as_str()).collect();
  let pos = |k: &str| keys.iter().position(|x| *x == k).unwrap_or_else(|| panic!("{k}: {keys:?}"));
  assert!(pos("COMMENT") < pos("HISTORY"));
  assert!(pos("HISTORY") < pos("ESO DET DIT"));
  assert!(pos("ESO DET DIT") < pos("LONGSTRN"));
  assert!(pos("LONGKEY") < pos("CONTINUE"));
  assert!(pos("CONTINUE") < pos("LOWERKEY"));
  // (cfitsio adds its own COMMENT cards: the FITS reference after NAXIS2, the LONGSTRN note)
  assert!(keys.iter().filter(|k| **k == "COMMENT").count() >= 2);
  assert_eq!(keys.iter().filter(|k| **k == "HISTORY").count(), 2);
  // (`FitsHeader` itself keeps only 8-column keywords; the HIERARCH card lives in the cards)
  let dit = meta.cards.iter().find(|c| c.key == "ESO DET DIT").unwrap();
  assert_eq!((dit.value.as_str(), dit.comment.as_str()), ("1.5", "detector integration time"));
  assert_eq!(meta.cards.iter().find(|c| c.key == "OBJECT").unwrap().value, "it's");
  assert_eq!(meta.facts.object.as_deref(), Some("it's"));
  assert_eq!(img.header.get("FLAG"), Some("T"));
  // a lowercase keyword (not a FITS keyword) is kept as written, the file still reads
  let (img2, meta2) = read_fits(&lowercase_keyword_file()).unwrap();
  assert_eq!(img2.data, [1.0, 2.0]);
  assert_eq!(img2.header.get("lowerkey"), Some("7"));
  assert_eq!(meta2.cards.iter().find(|c| c.key == "lowerkey").unwrap().comment, "lowercase keyword");
  // the first CONTINUE card's text starts the long value's continuation
  let long = meta.cards.iter().find(|c| c.key == "LONGKEY").unwrap();
  assert!(long.value.ends_with('&'), "{long:?}");
  assert_eq!(meta.cards.iter().filter(|c| c.key == "CONTINUE").count(), 2);
  // trailing garbage, blanked EXTEND
  let mut trailing = written_primary();
  trailing.extend_from_slice(b"garbage after the last block, not a header");
  assert_eq!(read_fits(&trailing).unwrap().0.data, RAMP);
  #[cfg(feature = "cfitsio")]
  {
    let mut no_extend = written_extension();
    let i = no_extend.windows(8).position(|c| c == b"EXTEND  ").unwrap();
    no_extend[i..i + 80].fill(b' ');
    let (img, _) = read_fits(&no_extend).unwrap();
    assert_eq!((img.hdu, img.data.clone()), (2, RAMP.to_vec()));
  }
}

/// Without CFITSIO the pure-Rust parser keeps refusing extensions and compressed images (and
/// says so), while the primary-HDU twin reads.
#[cfg(not(feature = "cfitsio"))]
#[test]
fn fits_legacy_refuses_extensions_and_compressed_images() {
  let (img, meta) = read_fits(&written_primary()).unwrap();
  assert_eq!((img.nx, img.ny, img.hdu, img.compressed, meta.hdu), (4, 3, 1, false, 1));
  assert_eq!(img.data, RAMP);
  let e = read_fits(&written_extension()).unwrap_err();
  assert!(e.contains("extensions are not read"), "{e}");
  assert!(read_fits(&written_rice()).unwrap_err().contains("extensions are not read"));
  // the header of a file whose image cannot be read is still shown
  assert_eq!(FitsMetadata::from_fits_bytes(&written_extension()).cards[0].key, "SIMPLE");
}

/// Both modes: the JSON carries the HDU number and the compression flag after `bitpix`, and the
/// one-call read gives the image and its metadata together.
#[test]
fn fits_metadata_json_reports_hdu_and_compression() {
  let bytes = fits_bytes(16, 2, 2, &["OBJECT  = 'x'"], &[1.0, 2.0, 3.0, 4.0]);
  let (img, meta) = read_fits(&bytes).unwrap();
  assert_eq!((img.nx, img.ny, img.hdu, img.compressed), (2, 2, 1, false));
  assert_eq!(img.data, [1.0, 2.0, 3.0, 4.0]);
  assert_eq!((meta.naxis1, meta.naxis2, meta.bitpix), (2, 2, 16));
  assert!(meta.to_json().starts_with("{\"naxis1\":2,\"naxis2\":2,\"bitpix\":16,\"hdu\":1,\"compressed\":false,\"planes\":1,\"plane\":0,\"cards\":["));
  assert_eq!((img.planes, img.plane, meta.planes, meta.plane), (1, 0, 1, 0));
  assert_eq!(meta, FitsMetadata::from_fits_bytes(&bytes));
}

// ─── the date chain ───────────────────────────────────────────────────────────

/// What the facts say about the date of a header.
#[derive(Debug, PartialEq)]
struct DateExpect {
  mid_utc: Option<&'static str>,
  source: Option<&'static str>,
  date_only: bool,
  file_date: bool,
  card_raw: Option<&'static str>,
  start_utc: Option<&'static str>,
  two_digit_year: bool,
}

impl DateExpect {
  const NONE: Self = Self { mid_utc: None, source: None, date_only: false, file_date: false, card_raw: None, start_utc: None, two_digit_year: false };

  /// A readable date: `mid` / `start` (`start` = `mid` when `None`), its source keyword and card.
  const fn at(mid: &'static str, start: Option<&'static str>, source: &'static str, card_raw: &'static str) -> Self {
    Self { mid_utc: Some(mid), source: Some(source), date_only: false, file_date: false, card_raw: Some(card_raw), start_utc: Some(match start { Some(s) => s, None => mid }), two_digit_year: false }
  }

  /// The card wrote its year with two digits (read as `20yy`).
  const fn two_digit_year(self) -> Self {
    Self { two_digit_year: true, ..self }
  }

  const fn date_only(self) -> Self {
    Self { date_only: true, ..self }
  }

  const fn file_date(self) -> Self {
    Self { file_date: true, ..self }
  }

  /// A header whose date cards are present but unreadable: only the first card is reported.
  const fn unreadable(card_raw: &'static str) -> Self {
    Self { card_raw: Some(card_raw), ..Self::NONE }
  }

  fn of(f: &FitsFacts) -> DateExpectOwned {
    DateExpectOwned {
      mid_utc: f.date_mid_utc.clone(),
      source: f.date_source.clone(),
      date_only: f.date_only,
      file_date: f.date_is_file_date,
      card_raw: f.date_card_raw.clone(),
      start_utc: f.date_start_utc.clone(),
      two_digit_year: f.date_two_digit_year,
    }
  }

  fn owned(&self) -> DateExpectOwned {
    DateExpectOwned {
      mid_utc: self.mid_utc.map(String::from),
      source: self.source.map(String::from),
      date_only: self.date_only,
      file_date: self.file_date,
      card_raw: self.card_raw.map(String::from),
      start_utc: self.start_utc.map(String::from),
      two_digit_year: self.two_digit_year,
    }
  }
}

#[derive(Debug, PartialEq)]
struct DateExpectOwned {
  mid_utc: Option<String>,
  source: Option<String>,
  date_only: bool,
  file_date: bool,
  card_raw: Option<String>,
  start_utc: Option<String>,
  two_digit_year: bool,
}

/// Every row of the date table: the header's cards and what the facts must say (both modes).
fn date_table() -> Vec<(&'static [&'static str], DateExpect)> {
  const D: fn(&'static str, Option<&'static str>, &'static str, &'static str) -> DateExpect = DateExpect::at;
  vec![
    // dd/mm/yy is 20yy (the FITS standard says 19yy; the UI offers it) and flagged, dd/mm/yyyy as
    // written; a date alone is midnight, dateOnly
    (&["DATE-OBS= '27/12/21'"], D("2021-12-27T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '27/12/21'").date_only().two_digit_year()),
    (&["DATE-OBS= '27/12/21'", "UT      = '18 30 00'", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:30:00.000Z"), "DATE-OBS", "DATE-OBS= '27/12/21'").two_digit_year()),
    // the flag follows the card used: a two-digit DATE-BEG under a four-digit DATE-AVG does not set it
    (&["DATE-AVG= '2021-12-27T18:30:10'", "DATE-BEG= '27/12/21'"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T00:00:00.000Z"), "DATE-AVG", "DATE-AVG= '2021-12-27T18:30:10'")),
    (&["DATE-OBS= '27/12/2021'"], D("2021-12-27T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '27/12/2021'").date_only()),
    (&["DATE-OBS= '2021/12/27'"], D("2021-12-27T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021/12/27'").date_only()),
    // the time of day from a time card: HH MM SS, decimal hours, a zero offset, UTSTART; TIME-OBS first
    (&["DATE-OBS= '2021-12-25'", "UT      = '18 30 00'"], D("2021-12-25T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "UT      = '18 30 00'", "TIME-OBS= '17.25'"], D("2021-12-25T17:15:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "TIME-OBS=                 17.25"], D("2021-12-25T17:15:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "TIME-OBS= '18:30:00+00:00'"], D("2021-12-25T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "TIME-BEG= '18:30:00.25Z'"], D("2021-12-25T18:30:00.250Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "UTC-OBS = '18:30'"], D("2021-12-25T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    (&["DATE-OBS= '2021-12-25'", "UTSTART = '00:07:28.858'"], D("2021-12-25T00:07:28.858Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    // an unreadable time card: midnight, dateOnly; an unreadable first card falls through to the next readable one
    (&["DATE-OBS= '2021-12-25'", "UT      = 'night'"], D("2021-12-25T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'").date_only()),
    (&["DATE-OBS= '2021-12-25'", "UT      = 'night'", "UTSTART = '06:00:00'"], D("2021-12-25T06:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-25'")),
    // zone markers, lowercase, a space or a padded `T`, a leap second, hh:mm
    (&["DATE-OBS= '2021-12-27t00:14:47.345z'"], D("2021-12-27T00:14:47.345Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27t00:14:47.345z'")),
    (&["DATE-OBS= '2021-12-27 18:30:00'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27 18:30:00'")),
    (&["DATE-OBS= '2021-12-27T18:30:00Z'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00Z'")),
    (&["DATE-OBS= '2021-12-27T18:30:00 UTC'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00 UTC'")),
    (&["DATE-OBS= '2021-12-27T18:30:00+0000'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00+0000'")),
    (&["DATE-OBS= '2021-12-27T18:30:00-00:00'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00-00:00'")),
    (&["DATE-OBS= '2021-12-27T18:30'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30'")),
    (&["DATE-OBS= '2022-02-28 T 23:59:59 UTC'"], D("2022-02-28T23:59:59.000Z", None, "DATE-OBS", "DATE-OBS= '2022-02-28 T 23:59:59 UTC'")),
    (&["DATE-OBS= '2021-12-31T23:59:60.5'"], D("2021-12-31T23:59:59.999Z", None, "DATE-OBS", "DATE-OBS= '2021-12-31T23:59:60.5'")),
    // refused: a non-zero offset (never shifted), an impossible date, a JD below the threshold
    (&["DATE-OBS= '2021-12-27T18:30:00+01:00'"], DateExpect::unreadable("DATE-OBS= '2021-12-27T18:30:00+01:00'")),
    (&["DATE-OBS= '2021-02-30'"], DateExpect::unreadable("DATE-OBS= '2021-02-30'")),
    (&["DATE-OBS= 'night of 27 Dec 2021' / when"], DateExpect::unreadable("DATE-OBS= 'night of 27 Dec 2021' / when")),
    (&["JD      =                  0.5"], DateExpect::unreadable("JD      = 0.5")),
    (&["DATE-OBS= 'night'", "MJD-OBS =              59573.5"], D("2021-12-25T12:00:00.000Z", None, "MJD-OBS", "MJD-OBS = 59573.5")),
    // the keywords added to the chain
    (&["DATE-BEG= '2021-12-27T18:30:00'", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:30:00.000Z"), "DATE-BEG", "DATE-BEG= '2021-12-27T18:30:00'")),
    (&["DATE_OBS= '2021-12-27T18:30:00'"], D("2021-12-27T18:30:00.000Z", None, "DATE_OBS", "DATE_OBS= '2021-12-27T18:30:00'")),
    (&["MJD_OBS =              59573.5"], D("2021-12-25T12:00:00.000Z", None, "MJD_OBS", "MJD_OBS = 59573.5")),
    (&["JD-OBS  =            2459575.0"], D("2021-12-26T12:00:00.000Z", None, "JD-OBS", "JD-OBS  = 2459575.0")),
    (&["JUL-DATE=            2459575.0"], D("2021-12-26T12:00:00.000Z", None, "JUL-DATE", "JUL-DATE= 2459575.0")),
    (&["DATE    = '2021-12-25'"], D("2021-12-25T00:00:00.000Z", None, "DATE", "DATE    = '2021-12-25'").date_only().file_date()),
    (&["DATE    = '2021-12-28T09:00:00' / file written"], D("2021-12-28T09:00:00.000Z", None, "DATE", "DATE    = '2021-12-28T09:00:00' / file written").file_date()),
    (&["DATE-OBS= '2021-12-27T18:30:00'", "DATE    = '2021-12-28T09:00:00'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00'")),
    // start and mid: a start card plus half the exposure; a mid card and where its start comes from
    (&["DATE-OBS= '2021-12-27T18:30:00'", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:30:00.000Z"), "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00'")),
    (&["DATE-OBS= '2021-12-27T18:30:00'", "DATE-BEG= '2021-12-27T18:29:50'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00'")),
    (&["DATE-AVG= '2021-12-27T18:30:10'", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:30:00.000Z"), "DATE-AVG", "DATE-AVG= '2021-12-27T18:30:10'")),
    (&["DATE-AVG= '2021-12-27T18:30:10'"], D("2021-12-27T18:30:10.000Z", None, "DATE-AVG", "DATE-AVG= '2021-12-27T18:30:10'")),
    (&["DATE-AVG= '2021-12-27T18:30:10'", "DATE-OBS= '2021-12-27T18:29:55'", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:29:55.000Z"), "DATE-AVG", "DATE-AVG= '2021-12-27T18:30:10'")),
    (&["DATE-AVG= '2021-12-27T18:30:10'", "DATE-BEG= '2021-12-27T18:29:50'", "DATE-OBS= '2021-12-27T18:29:55'"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:29:50.000Z"), "DATE-AVG", "DATE-AVG= '2021-12-27T18:30:10'")),
    (&["MJD-AVG =             59575.75", "MJD-BEG =              59575.5"], D("2021-12-27T18:00:00.000Z", Some("2021-12-27T12:00:00.000Z"), "MJD-AVG", "MJD-AVG = 59575.75")),
    // a DATE-OBS flagged mid by its comment keeps its own instant as the start
    (&["DATE-OBS= '2021-12-27T18:30:10' / UT, averaged", "EXPTIME =                 20.0"], D("2021-12-27T18:30:10.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:10' / UT, averaged")),
    // the time scale
    (&["DATE-OBS= '2021-12-27T18:30:00'", "TIMESYS = 'TT'"], D("2021-12-27T18:28:50.816Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00'")),
    // nothing
    (&[], DateExpect::NONE),
  ]
}

/// Every normaliser and chain variant, in whichever mode the crate is built.
#[test]
fn fits_date_variants_table() {
  for (cards, expect) in date_table() {
    let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 1, 1, cards, &[0.0]));
    assert_eq!(DateExpect::of(&m.facts), expect.owned(), "cards {cards:?}");
    let f = &m.facts;
    // the TDB fields follow the UTC strings
    match (f.date_start_utc.as_deref(), f.date_start_tdb_us) {
      (Some(s), Some(us)) => {
        // the string is truncated to the millisecond (the leap-second row sits at .999999999)
        let e = hifitime::Epoch::from_gregorian_str(s).unwrap();
        let from_string = (e.to_tdb_duration().total_nanoseconds() / 1000) as i64;
        assert!((0..=1000).contains(&(us - from_string)), "{cards:?}: start tdb {us} vs {from_string}");
      }
      (None, None) => {}
      other => panic!("{cards:?}: {other:?}"),
    }
    assert_eq!(f.date_is_mid, matches!(expect.source, Some("DATE-AVG" | "MJD-AVG")) || cards.iter().any(|c| c.contains("averaged")), "{cards:?}: is_mid");
    // the JSON carries the six keys, additively (after the site keys)
    let j = m.to_json();
    let tail = j.rsplit_once("\"siteLonFromLst\":").unwrap().1;
    assert!(tail.contains("\"dateCardRaw\":") && tail.contains("\"dateOnly\":") && tail.contains("\"dateIsFileDate\":") && tail.contains("\"dateStartUtc\":") && tail.contains("\"dateStartTdbUs\":"), "{j}");
    assert!(tail.contains(&format!("\"dateTwoDigitYear\":{},\"wcsProjection\":", expect.two_digit_year)), "{j}");
    match expect.card_raw {
      Some(raw) => {
        let mut q = String::new();
        json_str(&mut q, raw);
        assert!(j.contains(&format!("\"dateCardRaw\":{q}")), "{j}");
      }
      None => assert!(j.contains("\"dateCardRaw\":null,\"dateOnly\":false,\"dateIsFileDate\":false,\"dateStartUtc\":null,\"dateStartTdbUs\":null,\"dateTwoDigitYear\":false,\"wcsProjection\":"), "{j}"),
    }
  }
}

/// The normaliser's canonical forms (what `ffs2tm` accepts), and what it refuses.
#[test]
fn fits_date_normaliser_canonical_forms() {
  for (raw, canonical) in [
    ("'27/12/21'", Some("2021-12-27")),
    ("'01/01/99'", Some("2099-01-01")),
    (" 27/12/2021 ", Some("2021-12-27")),
    ("2021/12/27", Some("2021-12-27")),
    ("2021-12-27", Some("2021-12-27")),
    ("2021-12-27Z", Some("2021-12-27")),
    ("2021-12-27 UTC", Some("2021-12-27")),
    ("2021-12-27T", Some("2021-12-27")),
    ("2021-12-27T18", Some("2021-12-27T18:00:00")),
    ("2021-12-27T18:30", Some("2021-12-27T18:30:00")),
    ("2021-12-27t18:30:00z", Some("2021-12-27T18:30:00")),
    ("2021-12-27 18:30:00.5 UT", Some("2021-12-27T18:30:00.5")),
    ("2022-02-28 T 23:59:59 UTC", Some("2022-02-28T23:59:59")),
    ("2021-12-27T18:30:00+00:00", Some("2021-12-27T18:30:00")),
    ("2021-12-27T18:30:00.25-0000", Some("2021-12-27T18:30:00.25")),
    ("2021-12-27T18:30:00+00", Some("2021-12-27T18:30:00")),
    ("2021-12-31T23:59:60.5", Some("2021-12-31T23:59:59.999999999")),
    ("2021-12-27T18:30:00.1234567891", Some("2021-12-27T18:30:00.123456789")),
    ("2021-12-27T18:30:00+01:00", None),
    ("2021-12-27T18:30:00-05", None),
    ("2021-12-27T18:30:00.5x", None),
    ("27-12-2021", None),
    ("27/12", None),
    ("night", None),
    ("", None),
    ("''", None),
  ] {
    assert_eq!(normalise_fits_datetime(raw).as_deref(), canonical, "{raw:?}");
  }
  // only the dd/mm/yy form is flagged as a two-digit year
  for (raw, flagged) in [("'27/12/21'", Some(true)), ("27/12/21 18:30", Some(true)), ("27/12/2021", Some(false)), ("2021/12/27", Some(false)), ("2021-12-27", Some(false)), ("27/12", None)] {
    assert_eq!(normalise_fits_datetime_flagged(raw).map(|(_, f)| f), flagged, "{raw:?}");
  }
  for (raw, t) in [
    ("18:30:00", Some((18, 30, 0, 0))),
    ("'18:30:00.25'", Some((18, 30, 0, 250_000_000))),
    ("18:30", Some((18, 30, 0, 0))),
    ("18 30 00", Some((18, 30, 0, 0))),
    ("18 30", Some((18, 30, 0, 0))),
    ("18.5", Some((18, 30, 0, 0))),
    ("17.25", Some((17, 15, 0, 0))),
    ("0.000001", Some((0, 0, 0, 3_600_000))),
    ("18:30:00+00:00", Some((18, 30, 0, 0))),
    ("18:30:00Z", Some((18, 30, 0, 0))),
    ("18:30:00 UTC", Some((18, 30, 0, 0))),
    ("23:59:60", Some((23, 59, 59, 999_999_999))),
    ("18:30:00+01:00", None),
    ("24.0", None),
    ("-1", None),
    ("night", None),
    ("", None),
  ] {
    assert_eq!(parse_time_of_day(raw), t, "{raw:?}");
  }
}

/// With CFITSIO the canonical strings go through `ffs2tm`; its fields equal the pure-Rust parse
/// on every value of the table and of the normaliser test.
#[cfg(feature = "cfitsio")]
#[test]
fn fits_cfitsio_date_parse_matches_the_rust_parser() {
  let mut values: Vec<String> = Vec::new();
  for (cards, _) in date_table() {
    for c in cards.iter().filter(|c| c.starts_with("DATE")) {
      values.push(parse_full_card(&card(c)).unwrap().value);
    }
  }
  values.extend(["27/12/21", "2021-12-27 T 23:59:59 UTC", "2021-12-31T23:59:60.5", "2021-02-30", "2021-12-27T24:00:00", "2021-12-27T18:30:00.1234567891"].map(String::from));
  let mut parsed = 0;
  for v in values {
    let rust = normalise_fits_datetime_flagged(&v).and_then(|(c, two_digit_year)| parse_canonical_datetime(&c).map(|p| ParsedTime { two_digit_year, ..p }));
    let through = parse_fits_datetime(&v);
    // ffs2tm also rejects impossible calendar dates the Rust parser leaves to the epoch
    if through.is_none() && rust.is_some() {
      assert!(hifitime::Epoch::maybe_from_gregorian_utc(rust.unwrap().year, rust.unwrap().month, rust.unwrap().day, rust.unwrap().hour, rust.unwrap().minute, rust.unwrap().second, 0).is_err(), "{v:?}: cfitsio refused a valid date");
      continue;
    }
    assert_eq!(through, rust, "{v:?}");
    parsed += through.is_some() as u32;
  }
  assert!(parsed >= 30, "{parsed}");
  // ffs2tm alone would read '27/12/21' as 1921; the normaliser makes the year explicit first, so
  // both modes read 2021 and flag the two-digit year
  let through = parse_fits_datetime("'27/12/21'").unwrap();
  assert_eq!((through.year, through.month, through.day, through.two_digit_year), (2021, 12, 27, true));
  assert_eq!(Some(through), normalise_fits_datetime_flagged("'27/12/21'").and_then(|(c, f)| parse_canonical_datetime(&c).map(|p| ParsedTime { two_digit_year: f, ..p })));
}

/// The header-only path (`SkyImage::from_parts`) and the file path share the date chain: the
/// start of the exposure, east-positive longitude and the flags.
#[test]
fn fits_sky_image_meta_comes_from_the_facts() {
  use crate::scene::sky_image::{SKY_FLAG_DATE_OBS, SKY_FLAG_SITE_LON, SkyImage};
  let bytes = fits_bytes(16, 2, 2, &["DATE-OBS= '27/12/21'", "UT      = '18 30 00'", "EXPTIME =                 20.0", "LATITUDE=                 28.7", "LONGITUD=                 17.8 / [degrees West]"], &[1.0, 2.0, 3.0, 4.0]);
  let img = SkyImage::from_bytes("fits", &bytes, None).unwrap();
  let m = FitsMetadata::from_fits_bytes(&bytes);
  assert_eq!(m.facts.date_start_utc.as_deref(), Some("2021-12-27T18:30:00.000Z"));
  assert!(m.facts.date_two_digit_year);
  assert_eq!(img.meta.date_obs_tdb_us, m.facts.date_start_tdb_us);
  assert_eq!(img.meta.exptime_s, Some(20.0));
  assert_eq!(img.meta.site_lon_deg, Some(-17.8), "the comment's west wording reaches the header-only facts");
  assert!(img.flags & SKY_FLAG_DATE_OBS != 0 && img.flags & SKY_FLAG_SITE_LON != 0);
  // from_parts without the cards: the same date, the longitude without its comment
  let (nx, ny, data, h) = parse_fits(&bytes).unwrap();
  let plain = SkyImage::from_parts(nx, ny, data, h, None, false);
  assert_eq!(plain.meta.date_obs_tdb_us, m.facts.date_start_tdb_us);
  assert_eq!(plain.meta.site_lon_deg, Some(17.8));
}

// ─── committed fixtures ───────────────────────────────────────────────────────

/// `<repo>/test_assets`, where [`write_fits_fixtures`] writes for repacking
/// (`scripts/pack_test_assets.sh <out> fits_fixtures`); the tests read through `aethervk_test_assets`.
fn fixtures_source_dir() -> std::path::PathBuf {
  aethervk_test_assets::source_root().unwrap()
}

/// `test_assets/fits_date_variants/<name>`: a 4 × 3 BITPIX 16 ramp with the date cards of one
/// variant, and what the facts must say about it.
fn date_variant_fixtures() -> Vec<(&'static str, Vec<u8>, DateExpect)> {
  const D: fn(&'static str, Option<&'static str>, &'static str, &'static str) -> DateExpect = DateExpect::at;
  let ramp: Vec<f64> = RAMP.iter().map(|&v| v as f64).collect();
  let file = |name: &'static str, cards: &[&str], expect: DateExpect| {
    let mut all = vec!["OBJECT  = 'date variant'      / a 4 x 3 ramp with one date variant"];
    all.extend_from_slice(cards);
    (name, fits_bytes(16, 4, 3, &all, &ramp), expect)
  };
  vec![
    // the card's comment is part of the committed bytes; the app reads 20yy and flags the card
    file("date_obs_dd_mm_yy.fits", &["DATE-OBS= '27/12/21'            / dd/mm/yy: the FITS rule reads 1921"], D("2021-12-27T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '27/12/21' / dd/mm/yy: the FITS rule reads 1921").date_only().two_digit_year()),
    file("date_obs_dd_mm_yyyy.fits", &["DATE-OBS= '27/12/2021'"], D("2021-12-27T00:00:00.000Z", None, "DATE-OBS", "DATE-OBS= '27/12/2021'").date_only()),
    file("date_obs_date_only_ut.fits", &["DATE-OBS= '2021-12-27'", "UT      = '18 30 00'            / time of day"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27'")),
    file("date_obs_space_separator.fits", &["DATE-OBS= '2021-12-27 18:30:00'", "EXPTIME =                 20.0 / [s]"], D("2021-12-27T18:30:10.000Z", Some("2021-12-27T18:30:00.000Z"), "DATE-OBS", "DATE-OBS= '2021-12-27 18:30:00'")),
    file("date_obs_zulu.fits", &["DATE-OBS= '2021-12-27T18:30:00Z'"], D("2021-12-27T18:30:00.000Z", None, "DATE-OBS", "DATE-OBS= '2021-12-27T18:30:00Z'")),
    file("mjd_obs_only.fits", &["MJD-OBS =             59575.75 / [d]"], D("2021-12-27T18:00:00.000Z", None, "MJD-OBS", "MJD-OBS = 59575.75 / [d]")),
    file("date_file_only.fits", &["DATE    = '2021-12-28T09:00:00' / file written"], D("2021-12-28T09:00:00.000Z", None, "DATE", "DATE    = '2021-12-28T09:00:00' / file written").file_date()),
    file("date_unreadable.fits", &["DATE-OBS= 'night of 27 Dec 2021'"], DateExpect::unreadable("DATE-OBS= 'night of 27 Dec 2021'")),
  ]
}

/// Every committed FITS fixture this module generates: `(path under test_assets, bytes)`.
fn committed_fixtures() -> Vec<(String, Vec<u8>)> {
  let mut out = vec![
    ("fits_image_extension_4x3.fits".to_string(), written_extension()),
    ("fits_tile_compressed_4x3.fz".to_string(), written_rice()),
    ("fits_cube_4x3x3.fits".to_string(), written_cube_4x3x3(false)),
    ("fits_cube_4x3x3.fz".to_string(), written_cube_4x3x3(true)),
  ];
  out.extend(date_variant_fixtures().into_iter().map(|(n, b, _)| (format!("fits_date_variants/{n}"), b)));
  out
}

/// Writes the fixtures into `<repo>/test_assets/`: `cargo test -p aethervk-core-rlib --features std
/// write_fits_fixtures -- --ignored`, then repack and upload the `fits_fixtures` group (see
/// `test_assets/README.md`); `fits_committed_fixtures_match_the_generator` keeps them current.
#[test]
#[ignore]
fn write_fits_fixtures() {
  for (name, bytes) in committed_fixtures() {
    let p = fixtures_source_dir().join(&name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(&p, &bytes).unwrap();
    std::println!("wrote {} ({} bytes)", p.display(), bytes.len());
  }
}

/// The committed fixture bytes are what the generator writes (so they cannot drift from the
/// tests), and the date-variant folder holds nothing else.
#[test]
fn fits_committed_fixtures_match_the_generator() {
  for (name, bytes) in committed_fixtures() {
    let p = aethervk_test_assets::test_asset!(&name);
    let on_disk = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}; run write_fits_fixtures (--ignored)", p.display()));
    assert!(on_disk == bytes, "{}: differs from the generator; run write_fits_fixtures (--ignored)", p.display());
    assert!(bytes.len() <= 20 * 1024, "{name}: {} bytes", bytes.len());
  }
  let mut listed: Vec<String> = std::fs::read_dir(aethervk_test_assets::test_asset_dir!("fits_date_variants")).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
  listed.sort();
  let mut generated: Vec<String> = date_variant_fixtures().iter().map(|(n, ..)| n.to_string()).collect();
  generated.sort();
  assert_eq!(listed, generated, "every file of the folder is generated and tested");
}

/// Each date-variant fixture file reads back with the date facts of its variant, in both modes;
/// the pixels are the ramp.
#[test]
fn fits_date_variant_fixture_files() {
  let dir = aethervk_test_assets::test_asset_dir!("fits_date_variants");
  for (name, _, expect) in date_variant_fixtures() {
    let bytes = std::fs::read(dir.join(name)).unwrap();
    let (img, m) = read_fits(&bytes).unwrap();
    assert_eq!((img.nx, img.ny), (4, 3), "{name}");
    assert_eq!(img.data, RAMP, "{name}");
    assert_eq!(DateExpect::of(&m.facts), expect.owned(), "{name}");
    assert_eq!(m.facts.object.as_deref(), Some("date variant"), "{name}");
  }
}

/// The WCS facts in the facts JSON: a plain TAN header, a TAN-SIP header with a polynomial, a
/// galactic header, and a header without a WCS (all `null`); the keys come after every older key.
#[test]
fn fits_wcs_facts_in_the_json() {
  let tan = [
    "CTYPE1  = 'RA---TAN'",
    "CTYPE2  = 'DEC--TAN'",
    "CRVAL1  =                120.0",
    "CRVAL2  =                -30.0",
    "CRPIX1  =                  4.5",
    "CRPIX2  =                  4.5",
    "CDELT1  =             -0.00028",
    "CDELT2  =              0.00028",
  ];
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 8, 8, &tan, &[0.0; 64]));
  let f = &m.facts;
  assert_eq!(f.wcs_projection.as_deref(), Some("TAN"));
  assert_eq!(f.wcs_distortion.as_deref(), Some("none"));
  assert_eq!(f.wcs_frame.as_deref(), Some("ICRS"));
  assert_eq!(f.wcs_frame_assumed, Some(true), "no RADESYS / EQUINOX");
  assert!(f.wcs_equinox.is_none() && f.wcs_error.is_none());
  let j = m.to_json();
  assert!(
    j.contains("\"dateTwoDigitYear\":false,\"wcsProjection\":\"TAN\",\"wcsDistortion\":\"none\",\"wcsFrame\":\"ICRS\",\"wcsEquinox\":null,\"wcsFrameAssumed\":true}"),
    "{j}"
  );

  let mut fk5: Vec<&str> = tan.to_vec();
  fk5.extend(["RADESYS = 'FK5     '", "EQUINOX =               2000.0"]);
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 8, 8, &fk5, &[0.0; 64]));
  assert_eq!((m.facts.wcs_frame.as_deref(), m.facts.wcs_equinox, m.facts.wcs_frame_assumed), (Some("FK5"), Some(2000.0), Some(false)));
  assert!(m.to_json().contains("\"wcsFrame\":\"FK5\",\"wcsEquinox\":2000.0,\"wcsFrameAssumed\":false}"));

  let sip = [
    "CTYPE1  = 'RA---TAN-SIP'",
    "CTYPE2  = 'DEC--TAN-SIP'",
    "CRVAL1  =                120.0",
    "CRVAL2  =                -30.0",
    "CRPIX1  =                  4.5",
    "CRPIX2  =                  4.5",
    "CD1_1   =             -0.00028",
    "CD1_2   =                  0.0",
    "CD2_1   =                  0.0",
    "CD2_2   =              0.00028",
    "RADESYS = 'ICRS    '",
    "A_ORDER =                    2",
    "A_2_0   =               1.0E-6",
    "B_ORDER =                    2",
    "B_0_2   =               1.0E-6",
  ];
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 8, 8, &sip, &[0.0; 64]));
  assert_eq!(m.facts.wcs_projection.as_deref(), Some("TAN"));
  assert_eq!(m.facts.wcs_distortion.as_deref(), Some("SIP"));
  assert_eq!((m.facts.wcs_frame.as_deref(), m.facts.wcs_frame_assumed), (Some("ICRS"), Some(false)));
  assert!(m.to_json().contains("\"wcsDistortion\":\"SIP\""));

  let gal = [
    "CTYPE1  = 'GLON-CAR'",
    "CTYPE2  = 'GLAT-CAR'",
    "CRVAL1  =                  0.0",
    "CRVAL2  =                  0.0",
    "CRPIX1  =                  4.5",
    "CRPIX2  =                  4.5",
    "CDELT1  =                 -0.1",
    "CDELT2  =                  0.1",
  ];
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 8, 8, &gal, &[0.0; 64]));
  assert_eq!((m.facts.wcs_projection.as_deref(), m.facts.wcs_frame.as_deref()), (Some("CAR"), Some("galactic")));

  let m = FitsMetadata::from_fits_bytes(&fits_bytes(16, 8, 8, &[], &[0.0; 64]));
  assert!(m.facts.wcs_projection.is_none() && m.facts.wcs_frame_assumed.is_none());
  assert!(m.to_json().ends_with("\"wcsProjection\":null,\"wcsDistortion\":null,\"wcsFrame\":null,\"wcsEquinox\":null,\"wcsFrameAssumed\":null}}"));
}

/// A header-only HiPS tile (no WCS, no ORDER / NPIX cards) is recognised from its path alone;
/// the facts JSON names the order and the tile.
#[test]
fn fits_hips_facts_of_a_tile_path() {
  let m = FitsMetadata::from_fits_bytes(&fits_bytes(-32, 8, 8, &[], &[0.0; 64]));
  assert!(m.facts.wcs_projection.is_none());
  let h = header_of_cards(&m.cards);
  let (_, f) = super::hips::load_mapping(Some("/surveys/demo/Norder3/Dir0/Npix5.fits"), &h, 8, 8).unwrap().unwrap();
  assert!(!f.allsky);
  assert_eq!((f.order, f.npix, f.tile_width, f.ident_source), (3, Some(5), 8, "path"));
  let j = f.to_json();
  assert!(j.starts_with("{\"hips\":true,\"allsky\":false,\"order\":3,\"npix\":5,\"tile_width\":8,"), "{j}");
  assert!(super::hips::load_mapping(Some("/images/plain.fits"), &h, 8, 8).unwrap().is_none());
}
