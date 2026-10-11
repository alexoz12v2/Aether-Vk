use super::*;
use alloc::{format, string::String, vec, vec::Vec};

const ARCSEC: f64 = 1.0 / 3600.0;

/// One FITS card: `KEY     = value` padded to 80 columns.
fn card(key: &str, value: &str) -> String {
  let s = format!("{key:<8}= {value:>20}");
  format!("{s:<80}")
}

/// A FITS file: header cards, END, padding, big-endian data, padding.
fn fits_bytes(cards: &[String], data: &[u8]) -> Vec<u8> {
  let mut out = Vec::new();
  for c in cards {
    assert_eq!(c.len(), 80, "card {c:?}");
    out.extend_from_slice(c.as_bytes());
  }
  out.extend_from_slice(format!("{:<80}", "END").as_bytes());
  while out.len() % 2880 != 0 {
    out.push(b' ');
  }
  out.extend_from_slice(data);
  while out.len() % 2880 != 0 {
    out.push(0);
  }
  out
}

fn wcs_cards(ra: f64, dec: f64, crpix: [f64; 2], cdelt_arcsec: f64) -> Vec<String> {
  vec![
    card("CTYPE1", "'RA---TAN'"),
    card("CTYPE2", "'DEC--TAN'"),
    card("RADESYS", "'ICRS'"),
    card("CRVAL1", &format!("{ra:.12}")),
    card("CRVAL2", &format!("{dec:.12}")),
    card("CRPIX1", &format!("{:.6}", crpix[0])),
    card("CRPIX2", &format!("{:.6}", crpix[1])),
    card("CD1_1", &format!("{:.12E}", -cdelt_arcsec * ARCSEC)),
    card("CD1_2", "0.0"),
    card("CD2_1", "0.0"),
    card("CD2_2", &format!("{:.12E}", cdelt_arcsec * ARCSEC)),
  ]
}

#[test]
fn fits_bitpix_minus32_values_wcs_and_metadata() {
  let (nx, ny) = (7u32, 5u32);
  let values: Vec<f32> = (0..nx * ny).map(|i| i as f32 * 0.25 - 1.0).collect();
  let mut data = Vec::new();
  for v in &values {
    data.extend_from_slice(&v.to_be_bytes());
  }
  let mut cards = vec![
    card("SIMPLE", "T"),
    card("BITPIX", "-32"),
    card("NAXIS", "2"),
    card("NAXIS1", &format!("{nx}")),
    card("NAXIS2", &format!("{ny}")),
    card("DATE-OBS", "'2025-01-01T00:00:00.000'"),
    card("TIMESYS", "'UTC'"),
    card("EXPTIME", "120.0"),
    card("SITELAT", "'+45:30:00'"),
    card("SITELONG", "-110.5"),
    card("SITEELEV", "1234.5"),
    format!("{:<80}", "COMMENT a commentary card is skipped"),
  ];
  cards.extend(wcs_cards(150.0, 20.0, [4.0, 3.0], 1.5));
  let bytes = fits_bytes(&cards, &data);
  let img = SkyImage::from_bytes("fits", &bytes, None).unwrap();
  assert_eq!((img.nx, img.ny), (nx, ny));
  assert_eq!(img.data, values);
  let w = img.wcs.expect("wcs");
  assert_eq!(w.crval_deg, [150.0, 20.0]);
  assert_eq!(w.crpix, [4.0, 3.0]);
  assert!((w.cd_deg[0][0] + 1.5 * ARCSEC).abs() < 1e-15);
  assert_eq!(img.meta.exptime_s, Some(120.0));
  assert!((img.meta.site_lat_deg.unwrap() - 45.5).abs() < 1e-12);
  assert_eq!(img.meta.site_lon_deg, Some(-110.5));
  assert_eq!(img.meta.site_elev_m, Some(1234.5));
  // 2000-01-01T12:00 TT → 2025-01-01T00:00 UTC: 9131.5 days, plus TT − UTC = 37 s + 32.184 s;
  // TDB − TT is below 2 ms
  let expect_s = 9131.5 * 86400.0 + 69.184;
  let got_s = img.meta.date_obs_tdb_us.unwrap() as f64 * 1e-6;
  assert!((got_s - expect_s).abs() < 2e-3, "{got_s} vs {expect_s}");
  let all = SKY_FLAG_DATE_OBS
    | SKY_FLAG_EXPTIME
    | SKY_FLAG_SITE_LAT
    | SKY_FLAG_SITE_LON
    | SKY_FLAG_SITE_ELEV;
  assert_eq!(img.flags, all);
}

#[test]
fn fits_bitpix16_with_bzero_and_blank() {
  let (nx, ny) = (4u32, 3u32);
  // unsigned 16-bit convention: physical = raw + 32768
  let physical: Vec<u32> = (0..nx * ny).map(|i| i * 5000 + 7).collect();
  let mut data = Vec::new();
  for (i, p) in physical.iter().enumerate() {
    let raw = if i == 5 {
      -32768i16
    } else {
      (*p as i32 - 32768) as i16
    };
    data.extend_from_slice(&raw.to_be_bytes());
  }
  let cards = vec![
    card("SIMPLE", "T"),
    card("BITPIX", "16"),
    card("NAXIS", "2"),
    card("NAXIS1", &format!("{nx}")),
    card("NAXIS2", &format!("{ny}")),
    card("BZERO", "32768"),
    card("BSCALE", "1"),
    card("BLANK", "-32768"),
    card("MJD-OBS", "60676.0"),
  ];
  let (w, h, v, hdr) = parse_fits(&fits_bytes(&cards, &data)).unwrap();
  assert_eq!((w, h), (nx, ny));
  for (i, p) in physical.iter().enumerate() {
    if i == 5 {
      assert!(v[i].is_nan());
    } else {
      assert_eq!(v[i], *p as f32, "pixel {i}");
    }
  }
  let img = SkyImage::from_parts(w, h, v, hdr, None, false);
  assert!(img.wcs.is_none() && img.wcs_error.is_none());
  // MJD 60676.0 = 2025-01-01T00:00 UTC
  let got_s = img.meta.date_obs_tdb_us.unwrap() as f64 * 1e-6;
  assert!((got_s - (9131.5 * 86400.0 + 69.184)).abs() < 2e-3);
  assert!(img.flags & SKY_FLAG_TIMESYS_ASSUMED != 0);
  // no WCS: nothing to register
  assert!(img.gpu_data().is_none());
}

#[test]
fn fits_bitpix_8_32_64_and_minus64() {
  for (bitpix, enc) in [
    (8i64, (|v: i64| vec![v as u8]) as fn(i64) -> Vec<u8>),
    (32, |v| (v as i32).to_be_bytes().to_vec()),
    (64, |v| v.to_be_bytes().to_vec()),
    (-64, |v| (v as f64).to_be_bytes().to_vec()),
  ] {
    let vals = [0i64, 1, 17, 200];
    let data: Vec<u8> = vals.iter().flat_map(|&v| enc(v)).collect();
    let cards = vec![
      card("SIMPLE", "T"),
      card("BITPIX", &format!("{bitpix}")),
      card("NAXIS", "2"),
      card("NAXIS1", "2"),
      card("NAXIS2", "2"),
      card("BSCALE", "2.0D0"),
      card("BZERO", "-1.0"),
    ];
    let (_, _, v, _) = parse_fits(&fits_bytes(&cards, &data)).unwrap();
    let want: Vec<f32> = vals.iter().map(|&x| (2 * x - 1) as f32).collect();
    assert_eq!(v, want, "BITPIX {bitpix}");
  }
}

#[test]
fn fits_refuses_what_it_cannot_read() {
  let base = |naxis: &str| {
    vec![
      card("SIMPLE", "T"),
      card("BITPIX", "-32"),
      card("NAXIS", naxis),
      card("NAXIS1", "2"),
      card("NAXIS2", "2"),
      card("NAXIS3", "3"),
    ]
  };
  assert!(parse_fits(&fits_bytes(&base("0"), &[])).is_err());
  assert!(parse_fits(&fits_bytes(&base("3"), &[0; 64])).is_err());
  assert!(parse_fits(b"NOT A FITS FILE").is_err());
  // truncated data
  let mut b = fits_bytes(&base("2"), &[]);
  b.truncate(2880);
  assert!(parse_fits(&b).is_err());
}

fn test_wcs(ra: f64, dec: f64, rot_deg: f64, cdelt_arcsec: f64) -> Wcs {
  let r = rot_deg * DEG;
  let d = cdelt_arcsec * ARCSEC;
  Wcs {
    crval_deg: [ra, dec],
    crpix: [512.3, 384.7],
    cd_deg: [[-d * r.cos(), d * r.sin()], [d * r.sin(), d * r.cos()]],
    sip_ignored: false,
    frame_assumed: false,
  }
}

/// pixel → RA/Dec → pixel within 1e-6 px over the image, at several declinations, rotations and
/// scales (including a field across the pole).
#[test]
fn wcs_round_trip_pixel_radec_pixel() {
  let mut worst = 0.0f64;
  for &(ra, dec, rot, scale) in &[
    (10.0, 0.0, 0.0, 1.0),
    (200.0, 45.0, 30.0, 2.5),
    (359.9, -60.0, -120.0, 0.3),
    (80.0, 89.9, 15.0, 3.0),
    (0.0, -89.95, 200.0, 10.0),
  ] {
    let w = test_wcs(ra, dec, rot, scale);
    for j in 0..=16 {
      for i in 0..=16 {
        let p = (
          0.5 + 1024.0 * i as f64 / 16.0,
          0.5 + 768.0 * j as f64 / 16.0,
        );
        let (a, d) = w.pixel_to_radec_deg(p.0, p.1);
        let q = w.radec_to_pixel(a, d).unwrap();
        worst = worst.max((q.0 - p.0).abs()).max((q.1 - p.1).abs());
      }
    }
  }
  assert!(worst < 1e-6, "round trip error {worst} px");
}

/// Hand-checked TAN points: CRVAL at CRPIX; 100 px along +p1 with CD1_1 = −1″ (RA increasing to
/// the left, east left) at Dec 0 is RA − atan(100″) exactly; at Dec 60° the RA offset is
/// 100″ / cos 60° to second order.
#[test]
fn wcs_tan_hand_checked_points() {
  let w = test_wcs(30.0, 0.0, 0.0, 1.0);
  let (a, d) = w.pixel_to_radec_deg(w.crpix[0], w.crpix[1]);
  assert!((a - 30.0).abs() < 1e-12 && d.abs() < 1e-12);
  let (a, d) = w.pixel_to_radec_deg(w.crpix[0] + 100.0, w.crpix[1]);
  let xi = 100.0 * ARCSEC * DEG;
  assert!(((30.0 - a) * DEG - xi.atan()).abs() < 1e-13, "{a}");
  assert!(d.abs() < 1e-12);
  assert!(((30.0 - a) * 3600.0 - 100.0).abs() < 1e-5);
  // +p2 is north
  let (_, d) = w.pixel_to_radec_deg(w.crpix[0], w.crpix[1] + 100.0);
  assert!((d * 3600.0 - 100.0).abs() < 1e-5);

  let w = test_wcs(30.0, 60.0, 0.0, 1.0);
  let (a, d) = w.pixel_to_radec_deg(w.crpix[0] - 100.0, w.crpix[1]);
  let dra = (a - 30.0) * 3600.0;
  assert!((dra - 200.0).abs() < 1e-3, "ΔRA {dra}″");
  // η = 0 is the great circle through CRVAL perpendicular to its meridian: it bends towards the
  // equator, Dec falls by ξ² tan δ / 2
  let bend = (d - 60.0) * 3600.0;
  let want = -0.5 * (100.0 * ARCSEC * DEG).powi(2) * (60.0 * DEG).tan() / (ARCSEC * DEG);
  assert!((bend - want).abs() < 1e-6, "{bend} vs {want}");
}

fn header_of(cards: &[String]) -> FitsHeader {
  FitsHeader::parse_text(cards.join("\n").as_bytes())
}

#[test]
fn wcs_cd_cdelt_crota_and_pc_agree() {
  let rot: f64 = 25.0;
  let (d1, d2) = (-2.0 * ARCSEC, 2.0 * ARCSEC);
  let common = |extra: Vec<String>| {
    let mut c = vec![
      card("CTYPE1", "'RA---TAN-SIP'"),
      card("CTYPE2", "'DEC--TAN-SIP'"),
      card("CRVAL1", "120.0"),
      card("CRVAL2", "-30.0"),
      card("CRPIX1", "100.0"),
      card("CRPIX2", "50.0"),
      card("EQUINOX", "2000.0"),
    ];
    c.extend(extra);
    Wcs::from_header(&header_of(&c)).unwrap().unwrap()
  };
  let r = rot * DEG;
  let cd = common(vec![
    card("CD1_1", &format!("{:.15E}", d1 * r.cos())),
    card("CD1_2", &format!("{:.15E}", -d2 * r.sin())),
    card("CD2_1", &format!("{:.15E}", d1 * r.sin())),
    card("CD2_2", &format!("{:.15E}", d2 * r.cos())),
  ]);
  let crota = common(vec![
    card("CDELT1", &format!("{d1:.15E}")),
    card("CDELT2", &format!("{d2:.15E}")),
    card("CROTA2", &format!("{rot}")),
  ]);
  let pc = common(vec![
    card("CDELT1", &format!("{d1:.15E}")),
    card("CDELT2", &format!("{d2:.15E}")),
    card("PC1_1", &format!("{:.15E}", r.cos())),
    card("PC1_2", &format!("{:.15E}", -(d2 / d1) * r.sin())),
    card("PC2_1", &format!("{:.15E}", (d1 / d2) * r.sin())),
    card("PC2_2", &format!("{:.15E}", r.cos())),
  ]);
  assert!(cd.sip_ignored);
  for w in [crota, pc] {
    for i in 0..2 {
      for j in 0..2 {
        assert!(
          (w.cd_deg[i][j] - cd.cd_deg[i][j]).abs() < 1e-15,
          "{w:?} vs {cd:?}"
        );
      }
    }
  }
}

#[test]
fn wcs_frames_and_projections() {
  let w = |extra: Vec<String>| {
    let mut c = vec![
      card("CRVAL1", "1.0"),
      card("CRVAL2", "2.0"),
      card("CRPIX1", "1.0"),
      card("CRPIX2", "1.0"),
      card("CDELT1", "-0.001"),
      card("CDELT2", "0.001"),
    ];
    c.extend(extra);
    Wcs::from_header(&header_of(&c))
  };
  let tan = || vec![card("CTYPE1", "'RA---TAN'"), card("CTYPE2", "'DEC--TAN'")];
  let with = |mut v: Vec<String>, more: &[String]| {
    v.extend_from_slice(more);
    v
  };
  assert!(w(vec![]).unwrap().is_none());
  assert!(w(tan()).unwrap().unwrap().frame_assumed);
  assert!(!w(with(tan(), &[card("RADESYS", "'FK5'")])).unwrap().unwrap().frame_assumed);
  assert!(w(with(tan(), &[card("RADESYS", "'FK4'")])).is_err());
  assert!(w(with(tan(), &[card("EQUINOX", "1950.0")])).is_err());
  assert!(w(with(tan(), &[card("LONPOLE", "170.0")])).is_err());
  assert!(
    w(vec![
      card("CTYPE1", "'RA---SIN'"),
      card("CTYPE2", "'DEC--SIN'")
    ])
    .is_err()
  );
  assert!(
    w(vec![
      card("CTYPE1", "'GLON-TAN'"),
      card("CTYPE2", "'GLAT-TAN'")
    ])
    .is_err()
  );
}

/// The astrometry.net `.wcs` sidecar is a FITS header without newlines; one card per line works
/// too, and strings with quotes and comments parse.
#[test]
fn sidecar_header_text_forms() {
  let cards = wcs_cards(83.8, -5.4, [320.5, 240.5], 1.2);
  let mut blocks = Vec::new();
  for c in &cards {
    blocks.extend_from_slice(c.as_bytes());
  }
  blocks.extend_from_slice(format!("{:<80}", "END").as_bytes());
  let a = Wcs::from_header(&FitsHeader::parse_text(&blocks)).unwrap().unwrap();
  let b = Wcs::from_header(&FitsHeader::parse_text(cards.join("\r\n").as_bytes()))
    .unwrap()
    .unwrap();
  assert_eq!(a, b);
  assert_eq!(a.crval_deg, [83.8, -5.4]);
  let h = FitsHeader::parse_text(b"OBJECT  = 'It''s M42 '  / a comment\nEXPTIME = 30 / s");
  assert_eq!(h.get("OBJECT"), Some("It's M42"));
  assert_eq!(h.get_f64("EXPTIME"), Some(30.0));
  assert_eq!(
    parse_sexagesimal("-110 56 06"),
    Some(-(110.0 + 56.0 / 60.0 + 6.0 / 3600.0))
  );
}

/// The ecliptic rotation is SPICE's: the equinox is fixed and the north ecliptic pole is at
/// RA 18 h, Dec 90° − ε (ε = 23.4392911°).
#[test]
fn equatorial_to_ecliptic_is_the_eclipj2000_rotation() {
  assert!((OBLIQUITY_J2000_RAD / DEG - 23.4392911).abs() < 1e-7);
  let x = equatorial_to_ecliptic_j2000([1.0, 0.0, 0.0]);
  assert!((x[0] - 1.0).abs() < 1e-15);
  let pole = equatorial_to_ecliptic_j2000(dir_of_ra_dec(270.0, 90.0 - OBLIQUITY_J2000_RAD / DEG));
  assert!((pole[2] - 1.0).abs() < 1e-15, "{pole:?}");
  let v = [0.3, -0.4, 0.5];
  let r = ecliptic_to_equatorial_j2000(equatorial_to_ecliptic_j2000(v));
  for k in 0..3 {
    assert!((r[k] - v[k]).abs() < 1e-15);
  }
}

#[test]
fn half_float_bits() {
  for (v, bits) in [
    (0.0f32, 0x0000u16),
    (-0.0, 0x8000),
    (1.0, 0x3c00),
    (0.5, 0x3800),
    (-2.0, 0xc000),
    (65504.0, 0x7bff),
    (1.0e6, 0x7c00),
    (0.1, 0x2e66),
    (6.0e-8, 0x0001),
    (1.0e-9, 0x0000),
    (f32::INFINITY, 0x7c00),
  ] {
    assert_eq!(f32_to_f16_bits(v), bits, "{v}");
  }
  let nan = f32_to_f16_bits(f32::NAN);
  assert!(nan & 0x7c00 == 0x7c00 && nan & 0x3ff != 0);
}

/// Black / white points are the 1st / 99.5th percentiles; the texture holds `(v − black) / (white
/// − black)`; blank pixels stay NaN.
#[test]
fn upload_normalises_between_the_percentiles() {
  let (nx, ny) = (100u32, 100u32);
  let mut data: Vec<f32> = (0..nx * ny).map(|i| i as f32).collect();
  data[0] = f32::NAN;
  let h = header_of(&wcs_cards(10.0, 10.0, [50.0, 50.0], 1.0));
  let img = SkyImage::from_parts(nx, ny, data, h, None, false);
  assert!((img.black - 100.0).abs() <= 1.0, "{}", img.black);
  assert!((img.white - 9950.0).abs() <= 1.0, "{}", img.white);
  let g = img.gpu_data().unwrap();
  assert_eq!((g.tex_width, g.tex_height), (nx, ny));
  let half = |b: u16| -> f32 {
    // exact for the normal range used here
    let e = ((b >> 10) & 0x1f) as i32;
    let m = (b & 0x3ff) as f32;
    let s = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    if e == 0 {
      s * m * 2f32.powi(-24)
    } else {
      s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
  };
  assert!(g.texels[0] & 0x7c00 == 0x7c00 && g.texels[0] & 0x3ff != 0); // NaN
  let at = |v: f32| half(g.texels[v as usize]);
  let t = (5000.0 - img.black) / (img.white - img.black);
  assert!((at(5000.0) - t).abs() < 1e-3);
  assert_eq!(
    g.vertices.len(),
    (SKY_IMAGE_GRID - 1) * (SKY_IMAGE_GRID - 1) * 6
  );
}

/// Larger than the texture limit: box-averaged by an integer factor, texture coordinates spanning
/// the averaged blocks (a partial last block keeps the mapping exact).
#[test]
fn oversized_images_are_box_averaged() {
  let (nx, ny) = (SKY_IMAGE_MAX_TEXTURE + 5, 3u32);
  let data: Vec<f32> = (0..nx * ny).map(|i| (i % nx) as f32).collect();
  let img = SkyImage::from_parts(
    nx,
    ny,
    data,
    header_of(&wcs_cards(0.0, 0.0, [1.0, 1.0], 1.0)),
    None,
    false,
  );
  assert_eq!(img.texture_factor(), 2);
  let g = img.gpu_data().unwrap();
  assert_eq!((g.tex_width, g.tex_height), (nx.div_ceil(2), 2));
  // the last vertex column is the image edge p1 = nx + 0.5: u = nx / (2·tex_width) < 1
  let max_u = g.vertices.iter().map(|v| v.uv[0]).fold(0.0f32, f32::max);
  assert!((max_u as f64 - nx as f64 / (2 * g.tex_width) as f64).abs() < 1e-6);
}

/// Perspective camera looking along `fwd` (engine convention: view rows right, −forward, up;
/// Vulkan clip y down), `f` = 1 / tan(fov/2). Returns clip `(x, y, w)` of a direction.
fn clip_of(fwd: V3, up_hint: V3, f: f64, d: V3) -> [f64; 3] {
  let cross = |a: V3, b: V3| {
    [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]
  };
  let fwd = normalize(fwd);
  let right = normalize(cross(fwd, up_hint));
  let up = cross(right, fwd);
  [f * dot(right, d), -f * dot(up, d), dot(fwd, d)]
}

/// The registration claim: the mesh drawn at infinity under a pinhole projection, with
/// perspective-correct interpolation (what the rasteriser does), maps every screen point to the
/// texture coordinate the WCS gives its direction. The mesh vertices are the f32 data the GPU gets.
#[test]
fn registration_mesh_interpolates_the_wcs_exactly() {
  let w = test_wcs(212.0, 37.0, 18.0, 1.0);
  let (nx, ny) = (1000u32, 800u32);
  let verts = grid_vertices(&w, nx, ny, [nx as f64, ny as f64], SKY_IMAGE_GRID);
  let center = equatorial_to_ecliptic_j2000(w.pixel_to_equatorial(500.0, 400.0));
  // a 0.5° field, slightly off the image centre
  let f = 1.0 / (0.25f64 * DEG).tan();
  let fwd = [center[0] + 1e-4, center[1], center[2] - 2e-4];
  let up = [0.0, 0.0, 1.0];
  let mut worst = 0.0f64;
  let mut samples = 0;
  let mut rng = 0x2545_f491_4f6c_dd1du64;
  let mut rand = || {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    (rng >> 11) as f64 / (1u64 << 53) as f64
  };
  for _ in 0..400 {
    // a sky point inside the image, its exact pixel from the WCS
    let (p1, p2) = (
      1.0 + rand() * (nx as f64 - 1.0),
      1.0 + rand() * (ny as f64 - 1.0),
    );
    let d = equatorial_to_ecliptic_j2000(w.pixel_to_equatorial(p1, p2));
    let c = clip_of(fwd, up, f, d);
    let s = [c[0] / c[2], c[1] / c[2]];
    // find the screen triangle containing it, interpolate the uv perspective-correctly
    for t in verts.chunks_exact(3) {
      let cl: Vec<[f64; 3]> =
        t.iter().map(|v| clip_of(fwd, up, f, v.pos.map(|x| x as f64))).collect();
      let sc: Vec<[f64; 2]> = cl.iter().map(|c| [c[0] / c[2], c[1] / c[2]]).collect();
      let area = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
      };
      let tot = area(sc[0], sc[1], sc[2]);
      let l = [
        area(s, sc[1], sc[2]) / tot,
        area(sc[0], s, sc[2]) / tot,
        area(sc[0], sc[1], s) / tot,
      ];
      if l.iter().any(|&x| x < -1e-12) {
        continue;
      }
      // perspective-correct: weights l_i / w_i
      let q: Vec<f64> = (0..3).map(|i| l[i] / cl[i][2]).collect();
      let qs: f64 = q.iter().sum();
      let mut uv = [0.0f64; 2];
      for i in 0..3 {
        for k in 0..2 {
          uv[k] += q[i] * t[i].uv[k] as f64 / qs;
        }
      }
      let want = [(p1 - 0.5) / nx as f64, (p2 - 0.5) / ny as f64];
      let err_px = ((uv[0] - want[0]) * nx as f64).abs().max(((uv[1] - want[1]) * ny as f64).abs());
      worst = worst.max(err_px);
      samples += 1;
      break;
    }
  }
  assert!(samples > 300, "only {samples} samples landed on the mesh");
  // f32 vertices: ~6e-8 rad, 0.012″ at 1″/px
  assert!(worst < 0.05, "registration error {worst} px");
}

#[test]
fn raster_with_sidecar_and_the_store() {
  // a grey PNG and an RGB PNG written by the test, with a text WCS sidecar
  let dir = std::env::temp_dir().join(format!("avk_sky_image_{}", std::process::id()));
  std::fs::create_dir_all(&dir).unwrap();
  let grey = image::GrayImage::from_fn(16, 8, |x, y| image::Luma([(x * 16 + y) as u8]));
  let png = dir.join("field.png");
  grey.save(&png).unwrap();
  let side = dir.join("field.wcs");
  std::fs::write(&side, wcs_cards(10.0, 41.0, [8.5, 4.5], 2.0).join("\n")).unwrap();
  // the sidecar is found next to the image
  let img = SkyImage::load(png.to_str().unwrap(), None).unwrap();
  assert_eq!((img.nx, img.ny), (16, 8));
  assert!((img.data[16 + 3] - (3.0 * 16.0 + 1.0) / 255.0).abs() < 1e-6);
  assert_eq!(
    img.flags & (SKY_FLAG_RASTER | SKY_FLAG_WCS_SIDECAR),
    SKY_FLAG_RASTER | SKY_FLAG_WCS_SIDECAR
  );
  assert_eq!(img.wcs.unwrap().crval_deg, [10.0, 41.0]);

  let rgb = image::RgbImage::from_fn(4, 4, |_, _| image::Rgb([255, 0, 0]));
  let png2 = dir.join("red.png");
  rgb.save(&png2).unwrap();
  let img2 = SkyImage::load(png2.to_str().unwrap(), Some(side.to_str().unwrap())).unwrap();
  assert!((img2.data[0] - 0.2126).abs() < 1e-6);
  let dir_ecl = img2.crval_direction_ecliptic().unwrap();
  let back = ecliptic_to_equatorial_j2000(dir_ecl);
  let (ra, dec) = ra_dec_of(back);
  assert!((ra - 10.0).abs() < 1e-12 && (dec - 41.0).abs() < 1e-12);

  // store: insert, params (clamped), remove
  let key = 0xdead_0000usize;
  assert!(store_insert(key, &img));
  let g0 = store_get(key).unwrap().generation;
  assert_eq!(store_get(key).unwrap().params, SkyImageParams::default());
  assert!(store_set_params(
    key,
    SkyImageParams {
      opacity: 2.0,
      stretch: -1.0,
      visible: false
    }
  ));
  let e = store_get(key).unwrap();
  assert_eq!(
    (e.params.opacity, e.params.stretch, e.params.visible),
    (1.0, 0.0, false)
  );
  assert!(store_insert(key, &img2));
  let e = store_get(key).unwrap();
  assert!(
    e.generation > g0 && !e.params.visible,
    "a reload keeps the parameters"
  );
  assert!(store_remove(key) && !store_remove(key) && store_get(key).is_none());
  // without a WCS nothing is stored
  let plain = SkyImage::from_parts(2, 2, vec![0.0; 4], FitsHeader::default(), None, true);
  assert!(!store_insert(key, &plain) && store_get(key).is_none());
  let _ = std::fs::remove_dir_all(&dir);
}

// ─── engine render check ─────────────────────────────────────────────────────

static OVERLAY_TASK_ID: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
static OVERLAY_PE_ID: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

extern "C" fn overlay_render_callback(_scene: u64, pe: u64, generation: u64) {
  use core::sync::atomic::Ordering;
  if pe == OVERLAY_PE_ID.load(Ordering::Acquire) {
    OVERLAY_TASK_ID.store(generation, Ordering::Release);
  }
}

/// The overlay drawn by the engine: a synthetic FITS (64 × 64 px, 30″/px) with two 5 × 5 px stars,
/// one at CRPIX and one 20 px off along +p1, registered by its WCS and drawn in the composite
/// subpass of a 2° perspective camera aimed at CRVAL. Each star's intensity centroid in the
/// downloaded frame lands within 1 px of the engine's projection of the star's WCS direction
/// (the camera's `proj_f64 · view_f64`). Also proves the pipeline / upload record without
/// validation errors.
#[test]
fn sky_image_overlay_renders_registered_on_the_engine_camera() {
  use crate::simulation_api::{SimulationContext, components_api::CameraParams};
  use aethervk_oshal_rlib::math::{
    matrix::mat4f64::Mat4x4f64,
    quaternion::Quaternion,
    vector::{Vector3, Vector4, vec3::Vec3f32, vec4::Quat, vec4f64::Vec4f64},
  };
  use core::sync::atomic::Ordering;
  let _render_lock = crate::simulation_api::lock_render_callback_tests();
  OVERLAY_TASK_ID.store(0, Ordering::Release);

  // the image
  let (nx, ny) = (64u32, 64u32);
  let crpix = [32.0, 32.0];
  let stars = [(32u32, 32u32), (52, 32)];
  let mut values = vec![0.0f32; (nx * ny) as usize];
  for &(sx, sy) in &stars {
    for y in sy - 2..=sy + 2 {
      for x in sx - 2..=sx + 2 {
        // FITS pixel (x, y) is stored at index (y − 1)·nx + (x − 1)
        values[((y - 1) * nx + (x - 1)) as usize] = 1000.0;
      }
    }
  }
  let mut data = Vec::new();
  for v in &values {
    data.extend_from_slice(&v.to_be_bytes());
  }
  let mut cards = vec![
    card("SIMPLE", "T"),
    card("BITPIX", "-32"),
    card("NAXIS", "2"),
    card("NAXIS1", &format!("{nx}")),
    card("NAXIS2", &format!("{ny}")),
  ];
  cards.extend(wcs_cards(150.0, 20.0, crpix, 30.0));
  let img = SkyImage::from_bytes("fits", &fits_bytes(&cards, &data), None).unwrap();
  let wcs = img.wcs.unwrap();

  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);
  let Ok(ctx) = SimulationContext::startup(Some(|m: &str| panic!("Vulkan error: {m}"))) else {
    std::println!("Skipping: no Vulkan backend");
    return;
  };
  let (w, h) = (256u32, 256u32);
  let fov = 2.0f64 * DEG;
  let scene_id = ctx
    .create_empty_scene(
      true,
      hifitime::Epoch::from_gregorian_at_midnight(2025, 3, 1, hifitime::TimeScale::UTC),
      hifitime::Epoch::from_gregorian_at_midnight(2025, 3, 2, hifitime::TimeScale::UTC),
    )
    .unwrap();
  let key = scene_key(&ctx.get_scene(scene_id).unwrap());
  assert!(store_insert(key, &img));
  assert!(store_set_params(
    key,
    SkyImageParams {
      opacity: 1.0,
      stretch: 0.0,
      visible: true
    }
  ));

  // camera: forward = CRVAL (ecliptic), screen up towards the ecliptic north
  let fwd = equatorial_to_ecliptic_j2000(dir_of_ra_dec(150.0, 20.0));
  let cross = |a: V3, b: V3| {
    [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]
  };
  let right = normalize(cross(fwd, [0.0, 0.0, 1.0]));
  let up = cross(right, fwd);
  // local → world columns: R·X = −right, R·Y = −forward, R·Z = up
  let m = [
    [-right[0], -fwd[0], up[0]],
    [-right[1], -fwd[1], up[1]],
    [-right[2], -fwd[2], up[2]],
  ];
  let tr = m[0][0] + m[1][1] + m[2][2];
  let (qx, qy, qz, qw) = if tr > 0.0 {
    let s = (tr + 1.0).sqrt() * 2.0;
    (
      (m[2][1] - m[1][2]) / s,
      (m[0][2] - m[2][0]) / s,
      (m[1][0] - m[0][1]) / s,
      0.25 * s,
    )
  } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
    let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
    (
      0.25 * s,
      (m[0][1] + m[1][0]) / s,
      (m[0][2] + m[2][0]) / s,
      (m[2][1] - m[1][2]) / s,
    )
  } else if m[1][1] > m[2][2] {
    let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
    (
      (m[0][1] + m[1][0]) / s,
      0.25 * s,
      (m[1][2] + m[2][1]) / s,
      (m[0][2] - m[2][0]) / s,
    )
  } else {
    let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
    (
      (m[0][2] + m[2][0]) / s,
      (m[1][2] + m[2][1]) / s,
      0.25 * s,
      (m[1][0] - m[0][1]) / s,
    )
  };
  let rot = Quat::from_vector_and_scalar(
    Vec3f32::from_components(qx as f32, qy as f32, qz as f32),
    qw as f32,
  );
  // the quaternion reproduces the axes
  let f_back = rot.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
  assert!((f_back.x() as f64 - fwd[0]).abs() < 1e-6 && (f_back.z() as f64 - fwd[2]).abs() < 1e-6);
  let cam = ctx.spawn_entity(scene_id, "SkyImageCamera").unwrap();
  ctx
    .add_transform_component(
      scene_id,
      cam,
      Vec3f32::from_components(0.0, 0.0, 0.0),
      rot,
      Vec3f32::from_components(1.0, 1.0, 1.0),
    )
    .unwrap();
  ctx
    .add_camera_component(
      scene_id,
      cam,
      CameraParams::new_perspective(fov as f32, w as f32 / h as f32, 1e-5, 10.0),
    )
    .unwrap();
  let pe = ctx.create_presentation_engine(scene_id, w, h).unwrap();
  OVERLAY_PE_ID.store(pe.0, Ordering::Release);
  ctx.set_camera_for_presentation_engine(scene_id, pe, cam).unwrap();
  SimulationContext::set_render_callback(Some(overlay_render_callback));
  let _ = ctx.threads.logic_thread.tx().try_send(
    crate::simulation_api::structs::LogicCommand::PlayScene {
      scene_id,
      speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
    },
  );

  // download the first frame
  let mut frame = None;
  for _ in 0..1000 {
    let tid = OVERLAY_TASK_ID.load(Ordering::Acquire);
    if tid > 0 {
      for _ in 0..1000 {
        if !matches!(
          ctx.get_task_status(tid),
          crate::simulation_api::structs::TaskStatusCode::Pending
        ) {
          break;
        }
        std::thread::sleep(core::time::Duration::from_millis(10));
      }
      let mut buf = vec![0u8; (w * h * 4) as usize];
      if unsafe { ctx.download_image(tid, buf.as_mut_ptr(), buf.len()) } {
        frame = Some(buf);
      }
      break;
    }
    std::thread::sleep(core::time::Duration::from_millis(10));
  }
  SimulationContext::set_render_callback(None);
  let _ = ctx
    .threads
    .logic_thread
    .tx()
    .try_send(crate::simulation_api::structs::LogicCommand::PauseScene { scene_id });
  store_remove(key);
  let frame = frame.expect("no frame downloaded");

  // expected pixels: the engine's projection of each star's WCS direction
  let proj = Mat4x4f64::perspective_vk_reverse_z(fov, w as f64 / h as f64, 1e-5, 10.0);
  let view = crate::gpu::frame::view_from_axes_f64(right, fwd, up);
  let lum = |x: u32, y: u32| -> f64 {
    let i = ((y * w + x) * 4) as usize;
    frame[i..i + 3].iter().map(|&c| c as f64).sum::<f64>() / 3.0
  };
  let total: f64 = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| lum(x, y)).sum();
  assert!(total > 0.0, "blank frame: the overlay was not drawn");
  for &(sx, sy) in &stars {
    let d = equatorial_to_ecliptic_j2000(wcs.pixel_to_equatorial(sx as f64, sy as f64));
    let c = proj * view * Vec4f64::from_components(d[0], d[1], d[2], 0.0);
    let ex = (c.x() / c.w() + 1.0) * 0.5 * w as f64;
    let ey = (c.y() / c.w() + 1.0) * 0.5 * h as f64;
    // intensity centroid above the window's background (a uniform floor would pull it to the
    // window centre, the prediction), 15 px window, pixel centres at +0.5
    let ys = (ey as i64 - 7).max(0)..=(ey as i64 + 7).min(h as i64 - 1);
    let xs = (ex as i64 - 7).max(0)..=(ex as i64 + 7).min(w as i64 - 1);
    let mut bg = f64::MAX;
    for y in ys.clone() {
      for x in xs.clone() {
        bg = bg.min(lum(x as u32, y as u32));
      }
    }
    let (mut sum, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for y in ys.clone() {
      for x in xs.clone() {
        let l = lum(x as u32, y as u32) - bg;
        sum += l;
        cx += l * (x as f64 + 0.5);
        cy += l * (y as f64 + 0.5);
      }
    }
    assert!(
      sum > 0.0,
      "star ({sx}, {sy}) not drawn near ({ex:.1}, {ey:.1})"
    );
    let (cx, cy) = (cx / sum, cy / sum);
    std::println!(
      "[sky_image] star ({sx}, {sy}): rendered ({cx:.2}, {cy:.2}), predicted ({ex:.2}, {ey:.2})"
    );
    assert!(
      (cx - ex).abs() < 1.0 && (cy - ey).abs() < 1.0,
      "star ({sx}, {sy}) at ({cx:.2}, {cy:.2}), predicted ({ex:.2}, {ey:.2})"
    );
  }
  drop(ctx);
}
