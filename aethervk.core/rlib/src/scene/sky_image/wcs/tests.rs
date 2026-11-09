use super::projection::{Code, Projection, PvParams, R0};
use super::rotation::{Rotation, dir_of, lon_lat_of};
use super::*;
use alloc::{format, string::String, vec, vec::Vec};

fn pv(pairs: &[(usize, f64)]) -> PvParams {
  let mut p = PvParams::default();
  for &(m, v) in pairs {
    p.set(m, v);
  }
  p
}

fn prj(code: Code, params: &[(usize, f64)]) -> Projection {
  Projection::new(code, pv(params), None, None).unwrap()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
  (a - b).abs() <= tol
}

/// angular separation (degrees) of two `(lon, lat)` points, chord form
fn sep(a: (f64, f64), b: (f64, f64)) -> f64 {
  let (p, q) = (dir_of(a.0, a.1), dir_of(b.0, b.1));
  let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
  2.0 * (d / 2.0).min(1.0).asin() / DEG
}

// ─── projections: closed forms ───────────────────────────────────────────────

#[test]
fn projection_closed_forms() {
  let tan = prj(Code::Tan, &[]);
  // TAN: R = R0 cot θ, x = R sin φ, y = −R cos φ
  let (x, y) = tan.fwd(30.0, 60.0).unwrap();
  let r = R0 / (60.0f64 * DEG).tan();
  assert!(close(x, r * 0.5, 1e-12) && close(y, -r * (30.0f64 * DEG).cos(), 1e-12));
  assert!(tan.fwd(0.0, 0.0).is_none() && tan.fwd(0.0, -10.0).is_none());
  // STG: R = 2 R0 tan((90 − θ)/2)
  let (x, y) = prj(Code::Stg, &[]).fwd(0.0, 30.0).unwrap();
  assert!(close(x, 0.0, 1e-12) && close(y, -2.0 * R0 * (30.0f64 * DEG).tan(), 1e-12));
  // SIN: R = R0 cos θ, hidden hemisphere undefined
  let sin = prj(Code::Sin, &[]);
  let (_, y) = sin.fwd(0.0, 45.0).unwrap();
  assert!(close(y, -R0 * (45.0f64 * DEG).cos(), 1e-12));
  assert!(sin.fwd(0.0, -1.0).is_none());
  // ARC: R = 90 − θ (degrees)
  let (_, y) = prj(Code::Arc, &[]).fwd(0.0, 10.0).unwrap();
  assert!(close(y, -80.0, 1e-12));
  // ZEA: R = 2 R0 sin((90 − θ)/2); the antipode at 2 R0
  let (_, y) = prj(Code::Zea, &[]).fwd(0.0, -90.0).unwrap();
  assert!(close(y, -2.0 * R0, 1e-9));
  // CAR: identity
  assert_eq!(prj(Code::Car, &[]).fwd(12.5, -33.0), Some((12.5, -33.0)));
  // MER: y = R0 ln tan((90 + θ)/2)
  let (_, y) = prj(Code::Mer, &[]).fwd(0.0, 45.0).unwrap();
  assert!(close(y, R0 * (67.5f64 * DEG).tan().ln(), 1e-12));
  // CEA λ = 1: y = R0 sin θ
  let (_, y) = prj(Code::Cea, &[]).fwd(0.0, 30.0).unwrap();
  assert!(close(y, R0 * 0.5, 1e-12));
  // SFL: x = φ cos θ
  let (x, _) = prj(Code::Sfl, &[]).fwd(60.0, 60.0).unwrap();
  assert!(close(x, 30.0, 1e-12));
  // AIT: the equator at φ = 180 is at x = 4 R0 sin 90 / sqrt(1 + cos 90) … = 2 R0 sqrt 2
  let (x, y) = prj(Code::Ait, &[]).fwd(180.0, 0.0).unwrap();
  assert!(close(x, 2.0 * R0 * 2.0f64.sqrt(), 1e-9) && close(y, 0.0, 1e-9));
  // MOL: the pole at y = sqrt 2 R0, the equator x = 2 sqrt 2 R0 φ / π
  let mol = prj(Code::Mol, &[]);
  let (_, y) = mol.fwd(0.0, 90.0).unwrap();
  assert!(close(y, 2.0f64.sqrt() * R0, 1e-9));
  let (x, _) = mol.fwd(90.0, 0.0).unwrap();
  assert!(close(x, 2.0 * 2.0f64.sqrt() / core::f64::consts::PI * 90.0, 1e-9));
  // HPX H = 4, K = 3: equatorial zone y = 67.5 sin θ, the polar facet apex at y = 90
  let hpx = prj(Code::Hpx, &[]);
  let (x, y) = hpx.fwd(20.0, 30.0).unwrap();
  assert!(close(x, 20.0, 1e-12) && close(y, 67.5 * 0.5, 1e-12));
  let (x, y) = hpx.fwd(45.0, 90.0).unwrap();
  assert!(close(x, 45.0, 1e-9) && close(y, 90.0, 1e-9));
  // XPH: the pole at the origin, the φ = 90 wing along +x
  let xph = prj(Code::Xph, &[]);
  let (x, y) = xph.fwd(0.0, 90.0).unwrap();
  assert!(close(x, 0.0, 1e-9) && close(y, 0.0, 1e-9));
  let (x, y) = xph.fwd(90.0, 60.0).unwrap();
  assert!(x > 0.0 && close(y, 0.0, 1e-9), "{x} {y}");
  // COP θa = 45: the standard parallel is at distance Y0 from the apex, φ scaled by sin θa
  let cop = prj(Code::Cop, &[(1, 45.0)]);
  let (x, y) = cop.fwd(0.0, 45.0).unwrap();
  assert!(close(x, 0.0, 1e-12) && close(y, 0.0, 1e-9));
  let (x, _) = cop.fwd(10.0, 45.0).unwrap();
  let y0 = R0 / (45.0f64 * DEG).tan();
  assert!(close(x, y0 * (10.0 * (45.0f64 * DEG).sin() * DEG).sin(), 1e-9));
  // TSC: face centre of face 1 at the origin, 45° per unit
  assert_eq!(prj(Code::Tsc, &[]).fwd(0.0, 0.0), Some((0.0, 0.0)));
  let (_, y) = prj(Code::Tsc, &[]).fwd(0.0, 90.0).unwrap();
  assert!(close(y, 90.0, 1e-9));
}

/// Forward then inverse recovers `(φ, θ)` over the sphere for every projection, to 1e-9°.
#[test]
fn projection_round_trips() {
  let cases: Vec<(Code, Vec<(usize, f64)>)> = vec![
    (Code::Azp, vec![(1, 2.0), (2, 30.0)]),
    (Code::Szp, vec![(1, 2.0), (2, 180.0), (3, 60.0)]),
    (Code::Tan, vec![]),
    (Code::Stg, vec![]),
    (Code::Sin, vec![]),
    (Code::Sin, vec![(1, 0.1), (2, -0.2)]),
    (Code::Arc, vec![]),
    (Code::Zpn, vec![(1, 1.0), (3, 0.05)]),
    (Code::Zpn, vec![(0, 0.05), (1, 0.975), (2, -0.807), (3, 0.337), (4, -0.065), (5, 0.01), (6, 0.003), (7, -0.001)]),
    (Code::Zea, vec![]),
    (Code::Air, vec![(1, 45.0)]),
    (Code::Cyp, vec![(1, 1.0), (2, 0.7071067811865476)]),
    (Code::Cea, vec![(1, 0.5)]),
    (Code::Car, vec![]),
    (Code::Mer, vec![]),
    (Code::Sfl, vec![]),
    (Code::Par, vec![]),
    (Code::Mol, vec![]),
    (Code::Ait, vec![]),
    (Code::Cop, vec![(1, 45.0), (2, 25.0)]),
    (Code::Coe, vec![(1, -45.0), (2, 25.0)]),
    (Code::Cod, vec![(1, 45.0), (2, 25.0)]),
    (Code::Coo, vec![(1, 45.0), (2, 25.0)]),
    (Code::Bon, vec![(1, 45.0)]),
    (Code::Pco, vec![]),
    (Code::Tsc, vec![]),
    (Code::Csc, vec![]),
    (Code::Qsc, vec![]),
    (Code::Hpx, vec![]),
    (Code::Hpx, vec![(1, 6.0), (2, 5.0)]),
    (Code::Xph, vec![]),
  ];
  for (code, params) in &cases {
    let p = prj(*code, params);
    // CSC: wcslib's forward and inverse polynomials (COBE) only approximate each other (Calabretta
    // & Greisen 2002 §5.6.2: up to about 1e-2° near face edges)
    let tol = if *code == Code::Csc { 2e-2 } else { 1e-9 };
    // SIN at its limb (θ = 0): the inverse recovers θ from 1 − r², sqrt-limited to ~1e-6°
    let limb_tol = if *code == Code::Sin { 2e-6 } else { tol };
    let mut n = 0;
    for ti in -8..=8 {
      for pi in -11..=11 {
        let theta = ti as f64 * 11.0;
        let phi = pi as f64 * 16.0;
        let Some((x, y)) = p.fwd(phi, theta) else {
          continue;
        };
        let Some((phi2, theta2)) = p.inv(x, y) else {
          panic!("{}: ({phi}, {theta}) -> ({x}, {y}) has no inverse", code.name());
        };
        let e = sep((phi, theta), (phi2, theta2));
        let tol = if theta == 0.0 { limb_tol } else { tol };
        assert!(e <= tol, "{}: ({phi}, {theta}) -> ({x}, {y}) -> ({phi2}, {theta2}): {e:.3e} deg", code.name());
        n += 1;
      }
    }
    assert!(n > 100, "{}: only {n} points in the domain", code.name());
  }
}

/// Points off the projection's image are `None`, not garbage.
#[test]
fn projection_out_of_domain() {
  assert!(prj(Code::Sin, &[]).inv(R0 * 1.01, 0.0).is_none());
  assert!(prj(Code::Zea, &[]).inv(2.0 * R0 + 0.1, 0.0).is_none());
  assert!(prj(Code::Arc, &[]).inv(181.0, 0.0).is_none());
  assert!(prj(Code::Cea, &[]).inv(0.0, R0 * 1.01).is_none());
  assert!(prj(Code::Mol, &[]).inv(0.0, 2.0f64.sqrt() * R0 + 0.01).is_none());
  assert!(prj(Code::Ait, &[]).inv(4.0 * R0 + 1.0, 0.0).is_none());
  assert!(prj(Code::Hpx, &[]).inv(0.0, 60.0).is_none(), "the HPX gap between polar facets");
  assert!(prj(Code::Xph, &[]).inv(0.0, -83.88).is_none(), "an XPH gap");
  assert!(prj(Code::Tsc, &[]).inv(400.0, 0.0).is_none());
  assert!(prj(Code::Par, &[]).inv(0.0, 181.0).is_none());
}

// ─── rotation ────────────────────────────────────────────────────────────────

#[test]
fn rotation_defaults_and_paper_ii_examples() {
  // zenithal: the native pole is CRVAL, LONPOLE defaults to 180 (δ0 < θ0 = 90)
  let r = Rotation::celset(30.0, 40.0, 0.0, 90.0, None, None).unwrap();
  assert_eq!((r.lngp, r.latp, r.phip), (30.0, 40.0, 180.0));
  let c = r.to_celestial(dir_of(0.0, 90.0));
  let (a, d) = lon_lat_of(c);
  assert!(close(a, 30.0, 1e-9) && close(d, 40.0, 1e-9));
  // native (0, 89): one degree along φ = 0, away from the celestial pole (which is at φp = 180)
  let (a, d) = lon_lat_of(r.to_celestial(dir_of(0.0, 89.0)));
  assert!(close(a, 30.0, 1e-9) && close(d, 39.0, 1e-9), "{a} {d}");
  // cylindrical (φ0, θ0) = (0, 0): LONPOLE defaults to 0 when δ0 ≥ θ0
  let r = Rotation::celset(0.0, 0.0, 0.0, 0.0, None, None).unwrap();
  assert_eq!(r.phip, 0.0);
  assert!(close(r.latp, 90.0, 1e-9));
  let (a, d) = lon_lat_of(r.to_celestial(dir_of(10.0, 20.0)));
  assert!(close(a, 10.0, 1e-9) && close(d, 20.0, 1e-9));
  // user (φ0, θ0) = (10, 20) (PVi_1/2): the default φp is φ0 (Paper II §2.4)
  let r = Rotation::celset(30.0, 40.0, 10.0, 20.0, None, None).unwrap();
  assert!(close(r.phip, 10.0, 1e-12));
  assert!(close(r.lngp, 210.0, 1e-9) && close(r.latp, 70.0, 1e-9), "{} {}", r.lngp, r.latp);
  let (a, d) = lon_lat_of(r.to_celestial(dir_of(10.0, 20.0)));
  assert!(close(a, 30.0, 1e-9) && close(d, 40.0, 1e-9));
  // inverse
  let n = r.to_native(dir_of(30.0, 40.0));
  let (p, t) = super::rotation::phi_theta_of(n);
  assert!(close(p, 10.0, 1e-9) && close(t, 20.0, 1e-9));
}

// ─── frames ──────────────────────────────────────────────────────────────────

#[test]
fn frame_galactic_matches_astropy() {
  let g = Frame::galactic();
  // galactic centre and north galactic pole (astropy Galactic → ICRS)
  let (ra, dec) = g.lonlat_to_icrs(0.0, 0.0);
  assert!(close(ra, 266.40498829, 1e-7) && close(dec, -28.93617776, 1e-7), "{ra} {dec}");
  let (ra, dec) = g.lonlat_to_icrs(0.0, 90.0);
  assert!(close(ra, 192.85947789, 1e-7) && close(dec, 27.12825241, 1e-7), "{ra} {dec}");
  let (l, b) = g.icrs_to_lonlat(266.40498829, -28.93617776);
  assert!(sep((l, b), (0.0, 0.0)) < 1e-7);
}

#[test]
fn frame_fk5_bias_fk4_eterms_and_ecliptic() {
  // FK5 J2000 → ICRS is the 20 mas frame bias
  let f = Frame::fk5(None);
  let (ra, dec) = f.lonlat_to_icrs(0.0, 0.0);
  assert!(sep((ra, dec), (0.0, 0.0)) < 30.0 / 3.6e6 && sep((ra, dec), (0.0, 0.0)) > 10.0 / 3.6e6);
  // FK5 J2000 at (0, 0) → ICRS (astropy): RA 359.99999364, Dec −0.00000253
  assert!(close(ra, 359.99999364, 2e-8) && close(dec, -0.00000253, 2e-8), "{ra} {dec}");
  // FK4 B1950 origin → ICRS (astropy, obstime B1950): (0.64068461, 0.27840697)
  let f = Frame::fk4(None, false);
  let (ra, dec) = f.lonlat_to_icrs(0.0, 0.0);
  assert!(close(ra, 0.64068461, 3e-7) && close(dec, 0.27840697, 3e-7), "{ra} {dec}");
  // E-terms are ~0.34″: FK4 and FK4-NO-E differ by that much
  let g = Frame::fk4(None, true);
  let d = sep(f.lonlat_to_icrs(0.0, 0.0), g.lonlat_to_icrs(0.0, 0.0)) * 3600.0;
  // (0.34″ in all; at the origin only the components normal to the direction act)
  assert!(d > 0.05 && d < 0.6, "{d}″");
  // round trip through the E-terms
  let (l, b) = f.icrs_to_lonlat(ra, dec);
  assert!(sep((l, b), (0.0, 0.0)) < 1e-9);
  // ecliptic J2000: the ecliptic pole is at (270, 90 − ε), ε = 23.4392794 (IAU 2006)
  let e = Frame::ecliptic(None);
  let (ra, dec) = e.lonlat_to_icrs(0.0, 90.0);
  assert!(close(ra, 270.0, 2e-5) && close(dec, 90.0 - 23.439279444, 2e-5), "{ra} {dec}");
  // FK5 of another equinox: precession moves the origin by ~50″/yr
  let f = Frame::fk5(Some(1975.0));
  let (ra, _) = f.lonlat_to_icrs(0.0, 0.0);
  let ra = if ra > 180.0 { ra - 360.0 } else { ra };
  assert!(ra > 0.3 && ra < 0.4, "{ra}");
}

// ─── header parsing ──────────────────────────────────────────────────────────

fn header(cards: &[(&str, &str)]) -> FitsHeader {
  let text: Vec<String> = cards.iter().map(|(k, v)| format!("{k:<8}= {v}")).collect();
  FitsHeader::parse_text(text.join("\n").as_bytes())
}

#[test]
fn header_forms_linear_part_cunit_and_axis_order() {
  let base = [
    ("CTYPE1", "'RA---TAN'"),
    ("CTYPE2", "'DEC--TAN'"),
    ("CRVAL1", "10.0"),
    ("CRVAL2", "20.0"),
    ("CRPIX1", "100.0"),
    ("CRPIX2", "200.0"),
  ];
  let with = |extra: &[(&str, &str)]| {
    let mut c = base.to_vec();
    c.extend_from_slice(extra);
    Wcs::from_header(&header(&c)).unwrap().unwrap()
  };
  let cd = with(&[("CD1_1", "-0.001"), ("CD2_2", "0.001")]);
  let pc = with(&[("CDELT1", "-0.001"), ("CDELT2", "0.001"), ("PC1_1", "1"), ("PC2_2", "1")]);
  let crota = with(&[("CDELT1", "-0.001"), ("CDELT2", "0.001"), ("CROTA2", "0")]);
  let arcsec = Wcs::from_header(&header(&[
    ("CTYPE1", "'RA---TAN'"),
    ("CTYPE2", "'DEC--TAN'"),
    ("CRVAL1", "36000.0"),
    ("CRVAL2", "72000.0"),
    ("CRPIX1", "100.0"),
    ("CRPIX2", "200.0"),
    ("CDELT1", "-3.6"),
    ("CDELT2", "3.6"),
    ("CUNIT1", "'arcsec'"),
    ("CUNIT2", "'arcsec'"),
  ]))
  .unwrap()
  .unwrap();
  for w in [&pc, &crota, &arcsec] {
    for i in 0..2 {
      for j in 0..2 {
        assert!((w.cd_deg[i][j] - cd.cd_deg[i][j]).abs() < 1e-18);
      }
    }
    let e = sep(w.pixel_to_radec_deg(1.0, 1.0), cd.pixel_to_radec_deg(1.0, 1.0));
    assert!(e < 1e-10, "{e} deg: {:?} vs {:?}", w.crval_frame_deg, cd.crval_frame_deg);
  }
  // DEC first: the same sky
  let swapped = Wcs::from_header(&header(&[
    ("CTYPE1", "'DEC--TAN'"),
    ("CTYPE2", "'RA---TAN'"),
    ("CRVAL1", "20.0"),
    ("CRVAL2", "10.0"),
    ("CRPIX1", "200.0"),
    ("CRPIX2", "100.0"),
    ("CD1_1", "0.001"),
    ("CD2_2", "-0.001"),
  ]))
  .unwrap()
  .unwrap();
  assert_eq!(swapped.axes, [2, 1]);
  for (p1, p2) in [(1.0, 1.0), (300.0, 50.0), (100.0, 200.0)] {
    assert!(sep(swapped.pixel_to_radec_deg(p2, p1), cd.pixel_to_radec_deg(p1, p2)) < 1e-12);
  }
  // celestial axes 2 and 3 of a cube
  let cube = Wcs::from_header(&header(&[
    ("CTYPE1", "'FREQ'"),
    ("CTYPE2", "'RA---SIN'"),
    ("CTYPE3", "'DEC--SIN'"),
    ("CRVAL2", "10.0"),
    ("CRVAL3", "20.0"),
    ("CRPIX2", "100.0"),
    ("CRPIX3", "200.0"),
    ("CDELT2", "-0.001"),
    ("CDELT3", "0.001"),
  ]))
  .unwrap()
  .unwrap();
  assert_eq!(cube.axes, [2, 3]);
  assert_eq!(cube.facts().projection, "SIN");
  assert!(sep(cube.pixel_to_radec_deg(100.0, 200.0), (10.0, 20.0)) < 1e-12);
  // frame defaults
  assert!(cd.frame_assumed && cd.facts().frame == "ICRS");
  assert_eq!(with(&[("CD1_1", "-0.001"), ("CD2_2", "0.001"), ("EQUINOX", "2000.0")]).facts().frame, "FK5");
  assert_eq!(with(&[("CD1_1", "-0.001"), ("CD2_2", "0.001"), ("EPOCH", "1950.0")]).facts().frame, "FK4");
  assert_eq!(with(&[("CD1_1", "-0.001"), ("CD2_2", "0.001"), ("RADESYS", "'FK4-NO-E'")]).facts().frame, "FK4-NO-E");
  // singular matrix refused, unknown projection refused, no CTYPE → None
  assert!(Wcs::from_header(&header(&[("CTYPE1", "'RA---TAN'"), ("CTYPE2", "'DEC--TAN'"), ("CD1_1", "0")])).is_err());
  assert!(Wcs::from_header(&header(&[("CTYPE1", "'RA---ZZZ'"), ("CTYPE2", "'DEC--ZZZ'")])).is_err());
  assert!(Wcs::from_header(&header(&[("NAXIS", "2")])).unwrap().is_none());
}

#[test]
fn sip_round_trip_and_flag() {
  let w = Wcs::from_header(&header(&[
    ("CTYPE1", "'RA---TAN-SIP'"),
    ("CTYPE2", "'DEC--TAN-SIP'"),
    ("CRVAL1", "150.0"),
    ("CRVAL2", "2.0"),
    ("CRPIX1", "1024.5"),
    ("CRPIX2", "1024.5"),
    ("CD1_1", "-1.0E-4"),
    ("CD2_2", "1.0E-4"),
    ("A_ORDER", "2"),
    ("B_ORDER", "2"),
    ("A_2_0", "2.0E-6"),
    ("A_0_2", "-1.0E-6"),
    ("A_1_1", "5.0E-7"),
    ("B_2_0", "-1.5E-6"),
    ("B_0_2", "2.5E-6"),
    ("B_1_1", "-4.0E-7"),
  ]))
  .unwrap()
  .unwrap();
  assert!(w.has_distortion() && w.facts().distortion == "SIP" && !w.is_plain_tan());
  // the distortion moves the corner by whole pixels
  let plain = Wcs::tan(w.crval_deg, w.crpix, w.cd_deg);
  let d = sep(w.pixel_to_radec_deg(1.0, 1.0), plain.pixel_to_radec_deg(1.0, 1.0)) * 3600.0 / 0.36;
  assert!(d > 1.0, "{d} px");
  let mut worst = 0.0f64;
  for j in 0..=8 {
    for i in 0..=8 {
      let (p1, p2) = (0.5 + 2048.0 * i as f64 / 8.0, 0.5 + 2048.0 * j as f64 / 8.0);
      let (ra, dec) = w.pixel_to_radec_deg(p1, p2);
      let (q1, q2) = w.radec_to_pixel(ra, dec).unwrap();
      worst = worst.max((q1 - p1).abs()).max((q2 - p2).abs());
    }
  }
  assert!(worst < 1e-8, "SIP round trip {worst} px");
}

#[test]
fn tpv_round_trip() {
  let w = Wcs::from_header(&header(&[
    ("CTYPE1", "'RA---TPV'"),
    ("CTYPE2", "'DEC--TPV'"),
    ("CRVAL1", "150.0"),
    ("CRVAL2", "2.0"),
    ("CRPIX1", "1024.5"),
    ("CRPIX2", "1024.5"),
    ("CD1_1", "-1.0E-4"),
    ("CD2_2", "1.0E-4"),
    ("PV1_1", "1.0"),
    ("PV1_4", "0.02"),
    ("PV1_7", "0.1"),
    ("PV2_1", "1.0"),
    ("PV2_5", "-0.015"),
    ("PV2_7", "0.08"),
  ]))
  .unwrap()
  .unwrap();
  assert_eq!(w.facts().distortion, "TPV");
  assert_eq!(w.facts().projection, "TAN");
  let mut worst = 0.0f64;
  for j in 0..=8 {
    for i in 0..=8 {
      let (p1, p2) = (0.5 + 2048.0 * i as f64 / 8.0, 0.5 + 2048.0 * j as f64 / 8.0);
      let (ra, dec) = w.pixel_to_radec_deg(p1, p2);
      let (q1, q2) = w.radec_to_pixel(ra, dec).unwrap();
      worst = worst.max((q1 - p1).abs()).max((q2 - p2).abs());
    }
  }
  assert!(worst < 1e-8, "TPV round trip {worst} px");
}

/// The plain TAN keeps the former tangent-plane arithmetic: `Wcs::tan` and a header agree to
/// the bit, and the tangent point is the affine `ŝ0 + ξ ê + η n̂`.
#[test]
fn plain_tan_is_the_affine_tangent_plane() {
  let w = Wcs::from_header(&header(&[
    ("CTYPE1", "'RA---TAN'"),
    ("CTYPE2", "'DEC--TAN'"),
    ("CRVAL1", "212.0"),
    ("CRVAL2", "37.0"),
    ("CRPIX1", "512.3"),
    ("CRPIX2", "384.7"),
    ("CD1_1", "-2.5E-4"),
    ("CD1_2", "1.0E-5"),
    ("CD2_1", "1.0E-5"),
    ("CD2_2", "2.5E-4"),
    ("RADESYS", "'ICRS'"),
  ]))
  .unwrap()
  .unwrap();
  assert!(w.is_plain_tan());
  let t = Wcs::tan([212.0, 37.0], [512.3, 384.7], [[-2.5e-4, 1.0e-5], [1.0e-5, 2.5e-4]]);
  for (p1, p2) in [(1.0, 1.0), (1000.0, 800.0), (512.3, 384.7), (-50.0, 2000.0)] {
    assert_eq!(w.pixel_to_tangent_point(p1, p2), t.pixel_to_tangent_point(p1, p2));
    let (xi, eta) = w.pixel_to_standard(p1, p2);
    let (s0, e, n) = w.tangent_basis();
    let want = [s0[0] + xi * e[0] + eta * n[0], s0[1] + xi * e[1] + eta * n[1], s0[2] + xi * e[2] + eta * n[2]];
    assert_eq!(w.pixel_to_tangent_point(p1, p2), want);
    // and the generic path agrees with it to 1e-10°
    let d = w.pixel_to_icrs(p1, p2).unwrap();
    let (ra, dec) = lon_lat_of(d);
    let (ra2, dec2) = lon_lat_of(want);
    assert!(sep((ra, dec), (ra2, dec2)) < 1e-10);
  }
}

// ─── the astropy / wcslib reference tables ───────────────────────────────────

/// `fits_corpus/wcs_reference` (downloaded on demand), or skip the calling test.
macro_rules! reference_dir {
  () => {
    aethervk_test_assets::test_asset_dir!("fits_corpus/wcs_reference")
  };
}

fn parse_f(s: &str) -> f64 {
  if s == "nan" { f64::NAN } else { s.parse().unwrap_or_else(|_| panic!("bad number {s:?}")) }
}

/// Every `<name>.hdr` + `<name>.tsv` pair: pixel → world (header frame) within 1e-7° (with a
/// distortion 1e-6°), the ICRS columns within the same (or the table's `# icrs_tol_deg:`), world →
/// pixel within 1e-5 px of the oracle's own round trip of the table's world (`x_rt`, `y_rt`), and
/// the `nan` rows outside the projection here too. The oracle is astropy / wcslib, or wcstools
/// (TNX / ZPX), AST (DSS), sunpy (helioprojective): the table's `# oracle:` line.
#[test]
fn wcs_matches_the_astropy_reference_tables() {
  let dir = reference_dir!();
  let Ok(rd) = std::fs::read_dir(&dir) else {
    println!("no reference tables at {}: nothing to compare", dir.display());
    return;
  };
  let mut names: Vec<String> =
    rd.filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".hdr").map(str::to_string)).collect();
  names.sort();
  if names.is_empty() {
    println!("0 reference pairs in {}", dir.display());
    return;
  }
  let mut failures = Vec::new();
  let mut summary = Vec::new();
  for name in &names {
    let hdr = std::fs::read(dir.join(format!("{name}.hdr"))).unwrap();
    let Ok(tsv) = std::fs::read_to_string(dir.join(format!("{name}.tsv"))) else {
      continue;
    };
    let h = FitsHeader::parse_text(&hdr);
    let w = match Wcs::from_header(&h) {
      Ok(Some(w)) => w,
      Ok(None) => {
        failures.push(format!("{name}: no celestial WCS recognised"));
        continue;
      }
      Err(e) => {
        failures.push(format!("{name}: refused: {e}"));
        continue;
      }
    };
    let tol_deg = if w.has_distortion() { 1e-6 } else { 1e-7 };
    // `# icrs_tol_deg:` of the table (helioprojective: the truncated Earth ephemeris, 0.1″)
    let icrs_tol = tsv
      .lines()
      .find_map(|l| l.strip_prefix("# icrs_tol_deg:"))
      .map(|v| v.trim().parse::<f64>().unwrap())
      .unwrap_or(tol_deg);
    let (mut worst_world, mut worst_icrs, mut worst_pix) = (0.0f64, 0.0f64, 0.0f64);
    let mut rows = 0;
    for line in tsv.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
      let f: Vec<f64> = line.split('\t').map(parse_f).collect();
      assert!(f.len() >= 6, "{name}: short row {line:?}");
      let (x, y, lon, lat, ra, dec) = (f[0], f[1], f[2], f[3], f[4], f[5]);
      // the pixel to recover: astropy's own world → pixel of the table's world (columns 7–8,
      // `x_rt`, `y_rt`) when present — wcslib's CSC forward and inverse polynomials are
      // approximations of each other (0.2 px), so the true pixel is not what the forward gives
      let (xr, yr) = if f.len() >= 8 && f[6].is_finite() { (f[6], f[7]) } else { (x, y) };
      let got = w.pixel_to_world(x, y);
      if !lon.is_finite() {
        if let Some((a, b)) = got {
          failures.push(format!("{name}: ({x}, {y}) is outside the projection for wcslib, here ({a}, {b})"));
        }
        continue;
      }
      rows += 1;
      let Some((glon, glat)) = got else {
        failures.push(format!("{name}: ({x}, {y}) has no sky position here, wcslib ({lon}, {lat})"));
        continue;
      };
      let e = sep((glon, glat), (lon, lat));
      worst_world = worst_world.max(e);
      if e > tol_deg {
        failures.push(format!("{name}: ({x}, {y}) -> ({glon}, {glat}), wcslib ({lon}, {lat}): {e:.3e} deg"));
      }
      if ra.is_finite() {
        let (gra, gdec) = w.pixel_to_radec_deg(x, y);
        let e = sep((gra, gdec), (ra, dec));
        worst_icrs = worst_icrs.max(e);
        if e > icrs_tol {
          failures.push(format!("{name}: ({x}, {y}) ICRS ({gra}, {gdec}), astropy ({ra}, {dec}): {e:.3e} deg"));
        }
      }
      match w.world_to_pixel(lon, lat) {
        Some((px, py)) => {
          let e = (px - xr).abs().max((py - yr).abs());
          worst_pix = worst_pix.max(e);
          if e > 1e-5 {
            failures.push(format!("{name}: world ({lon}, {lat}) -> ({px}, {py}), expected ({xr}, {yr}): {e:.3e} px"));
          }
        }
        None => failures.push(format!("{name}: world ({lon}, {lat}) has no pixel here, expected ({x}, {y})")),
      }
    }
    summary.push(format!(
      "{name:<30} {:<4} {:<4} {:<9} rows {rows:>3}  world {worst_world:.2e}  icrs {worst_icrs:.2e}  pix {worst_pix:.2e}",
      w.facts().projection,
      w.facts().distortion,
      w.facts().frame
    ));
  }
  for s in &summary {
    println!("{s}");
  }
  println!("{} reference pairs", names.len());
  assert!(failures.is_empty(), "{} mismatches:\n  {}", failures.len(), failures.join("\n  "));
}

// ─── IRAF WAT strings, TNX / ZPX surfaces ────────────────────────────────────

#[test]
fn wat_pieces_are_padded_and_attributes_found() {
  // IRAF cuts at 68 characters: piece 1 ends in the blank between "1." and "2."; the header parser
  // strips that blank, the concatenation must put it back
  let piece1 = format!("{:<67}", "wtype=tnx axtype=ra lngcor = \"3. 2. 2. 1. -0.5 0.5 -0.5 0.5 1.");
  assert_eq!(piece1.len(), 67);
  let h = header(&[
    ("WAT1_002", "'2. 3. 4. \"'"),
    ("WAT1_001", &format!("'{piece1}'")),
    ("WAT0_001", "'system=image'"),
  ]);
  let s = wat::wat_string(&h, 1);
  assert_eq!(s.len(), 68 + "2. 3. 4. \"".len());
  assert!(s.starts_with(piece1.trim_end()) && s.ends_with(" 2. 3. 4. \""));
  assert_eq!(wat::wat_attr(&s, "wtype"), Some("tnx"));
  assert_eq!(wat::wat_attr(&s, "axtype"), Some("ra"));
  let nums = wat::parse_numbers(wat::wat_attr(&s, "lngcor").unwrap()).unwrap();
  assert_eq!(nums, vec![3.0, 2.0, 2.0, 1.0, -0.5, 0.5, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0]);
  assert_eq!(wat::wat_attr(&s, "latcor"), None);
  assert_eq!(wat::wat_attr_f64("wtype=zpx axtype=dec projp0=0. projp1=1. projp3=-5.0E-2 ro=57.29578", "projp3"), Some(-0.05));
  assert_eq!(wat::wat_attr_f64("projp10=7. projp1=1.", "projp1"), Some(1.0));
  assert_eq!(wat::parse_number("1.5D-3"), Some(1.5e-3));
  assert_eq!(wat::parse_numbers("3. 4 -0.25 1E2").unwrap(), vec![3.0, 4.0, -0.25, 100.0]);
  assert!(wat::parse_numbers("3. x").is_err());
  // the last piece is not padded
  let h = header(&[("WAT2_001", "'wtype=tnx axtype=dec'"), ("BITPIX", "16")]);
  assert_eq!(wat::wat_string(&h, 2), "wtype=tnx axtype=dec");
  assert_eq!(wat::wat_string(&h, 1), "");
}

#[test]
fn iraf_surface_counts_and_values() {
  use IrafCrossTerms::*;
  assert_eq!(IrafSurface::coefficient_count(4, 4, None), 7);
  assert_eq!(IrafSurface::coefficient_count(4, 4, Full), 16);
  assert_eq!(IrafSurface::coefficient_count(4, 4, Half), 10);
  assert_eq!(IrafSurface::coefficient_count(3, 5, Half), 12);
  assert_eq!(IrafSurface::coefficient_count(5, 3, Half), 12);
  let lims = [-0.5, 1.5, -2.0, 2.0];
  let surf = |f: f64, xo: f64, yo: f64, xt: f64, c: &[f64]| {
    let mut v = vec![f, xo, yo, xt];
    v.extend_from_slice(&lims);
    v.extend_from_slice(c);
    IrafSurface::from_numbers(&v).unwrap()
  };
  // polynomial, full cross terms 2x2: C00 + C10 x + C01 y + C11 x y, on x itself
  let s = surf(3.0, 2.0, 2.0, 1.0, &[1.0, 2.0, 3.0, 4.0]);
  assert!(close(s.eval(0.7, -1.1), 1.0 + 2.0 * 0.7 + 3.0 * -1.1 + 4.0 * 0.7 * -1.1, 1e-14));
  // polynomial, half cross terms 3x3: 6 coefficients, the top-right triangle dropped
  let s = surf(3.0, 3.0, 3.0, 2.0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
  let (x, y) = (0.3, 1.2);
  assert!(close(s.eval(x, y), 1.0 + 2.0 * x + 3.0 * x * x + y * (4.0 + 5.0 * x) + 6.0 * y * y, 1e-14));
  // no cross terms 3x3: C00 C10 C20 then C01 C02
  let s = surf(3.0, 3.0, 3.0, 0.0, &[1.0, 2.0, 3.0, 4.0, 5.0]);
  assert!(close(s.eval(x, y), 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * y + 5.0 * y * y, 1e-14));
  // Chebyshev 3x1 on the normalised x: T0 + T2 = 2 xn²
  let s = surf(1.0, 3.0, 1.0, 0.0, &[1.0, 0.0, 1.0]);
  let xn = (2.0 * x - (lims[1] + lims[0])) / (lims[1] - lims[0]);
  assert!(close(s.eval(x, y), 2.0 * xn * xn, 1e-14));
  // Legendre 1x3 on the normalised y: P0 + P2 = (3 yn² + 1) / 2
  let s = surf(2.0, 1.0, 3.0, 0.0, &[1.0, 0.0, 1.0]);
  let yn = (2.0 * y - (lims[3] + lims[2])) / (lims[3] - lims[2]);
  assert!(close(s.eval(x, y), (3.0 * yn * yn + 1.0) / 2.0, 1e-14));
  // malformed strings
  assert!(IrafSurface::from_numbers(&[3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]).is_err()); // 1 of 4 coefficients
  assert!(IrafSurface::from_numbers(&[4.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0]).is_err()); // function type 4
  assert!(IrafSurface::from_numbers(&[1.0, 11.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).is_err()); // order 11
  // the correction and its inverse
  let c = IrafCorrection { lng: Some(surf(3.0, 2.0, 2.0, 1.0, &[1e-3, 2e-3, -1e-3, 5e-4])), lat: Option::None };
  let (xd, yd) = c.forward(0.4, -0.3);
  let (x, y) = c.inverse(xd, yd).unwrap();
  assert!(close(x, 0.4, 1e-12) && close(y, -0.3, 1e-12));
}

#[test]
fn tnx_zpx_headers_from_the_registry_samples() {
  let dir = reference_dir!();
  let tnx = FitsHeader::parse_text(&std::fs::read(dir.join("tnx_registry_ctio_mosaic.hdr")).unwrap());
  let w = Wcs::from_header(&tnx).unwrap().unwrap();
  let f = w.facts();
  assert_eq!((f.projection, f.distortion, f.frame, f.equinox), ("TAN", "TNX", "FK5", Some(2000.0)));
  let Distortion::Tnx(c) = w.distortion else { panic!() };
  let s = c.lng.unwrap();
  assert_eq!((s.function, s.xorder, s.yorder, s.cross), (IrafFunction::Chebyshev, 4, 4, IrafCrossTerms::Half));
  assert_eq!(s.ncoeff, 10);
  let (ra, dec) = w.pixel_to_radec_deg(123.0, 321.0);
  let (x, y) = w.radec_to_pixel(ra, dec).unwrap();
  assert!(close(x, 123.0, 1e-9) && close(y, 321.0, 1e-9), "{x} {y}");
  let zpx = FitsHeader::parse_text(&std::fs::read(dir.join("zpx_registry_mosaic2.hdr")).unwrap());
  let w = Wcs::from_header(&zpx).unwrap().unwrap();
  let f = w.facts();
  assert_eq!((f.projection, f.distortion, f.frame), ("ZPN", "ZPX", "FK5"));
  let (ra, dec) = w.pixel_to_radec_deg(10.0, 500.0);
  let (x, y) = w.radec_to_pixel(ra, dec).unwrap();
  assert!(close(x, 10.0, 1e-9) && close(y, 500.0, 1e-9), "{x} {y}");
  // a TNX without any WAT card is refused, naming the missing cards
  let e = Wcs::from_header(&header(&[("CTYPE1", "'RA---TNX'"), ("CTYPE2", "'DEC--TNX'")])).unwrap_err();
  assert!(e.contains("WAT1_nnn"), "{e}");
  // a non-standard generating sphere is refused
  let e = Wcs::from_header(&header(&[
    ("CTYPE1", "'RA---TNX'"),
    ("CTYPE2", "'DEC--TNX'"),
    ("WAT1_001", "'wtype=tnx axtype=ra ro=60.'"),
    ("WAT2_001", "'wtype=tnx axtype=dec'"),
  ]))
  .unwrap_err();
  assert!(e.contains("ro=60"), "{e}");
}

// ─── DSS plate solutions ─────────────────────────────────────────────────────

#[test]
fn dss_plate_polynomial_and_header() {
  let amdx: [f64; 13] = core::array::from_fn(|i| (i + 1) as f64 * 0.1);
  let amdy: [f64; 13] = core::array::from_fn(|i| 2.0 - (i + 1) as f64 * 0.05);
  let p = DssPlate { amdx, amdy };
  let (x, y) = (12.5, -7.25);
  let r2 = x * x + y * y;
  let poly = |c: &[f64; 13], x: f64, y: f64| {
    c[0] * x + c[1] * y + c[2] + c[3] * x * x + c[4] * x * y + c[5] * y * y + c[6] * r2 + c[7] * x * x * x + c[8] * x * x * y + c[9] * x * y * y + c[10] * y * y * y + c[11] * x * r2 + c[12] * x * r2 * r2
  };
  let (xi, eta) = p.forward(x, y);
  assert!(close(xi * 3600.0, poly(&amdx, x, y), 1e-9) && close(eta * 3600.0, poly(&amdy, y, x), 1e-9));
  // a realistic plate: ~67″/mm, small higher-order terms; the inverse closes
  let mut amdx = [0.0; 13];
  let mut amdy = [0.0; 13];
  amdx[0] = -67.15;
  amdx[1] = 0.02;
  amdx[2] = -0.3;
  amdx[6] = 2e-5;
  amdx[11] = 1e-7;
  amdy[0] = 67.14;
  amdy[1] = -0.015;
  amdy[2] = 0.4;
  amdy[6] = 1.5e-5;
  amdy[11] = 1e-7;
  let p = DssPlate { amdx, amdy };
  let (xi, eta) = p.forward(-150.0, 120.0);
  let (x, y) = p.inverse(xi, eta).unwrap();
  assert!(close(x, -150.0, 1e-9) && close(y, 120.0, 1e-9), "{x} {y}");
  // the ESO cutout of the reference tables: the plate cards alone give a TAN at the plate centre
  let h = FitsHeader::parse_text(&std::fs::read(reference_dir!().join("dss_eso_dss1_cena.hdr")).unwrap());
  let w = Wcs::from_header(&h).unwrap().unwrap();
  let f = w.facts();
  assert_eq!((f.projection, f.distortion, f.frame, f.equinox), ("TAN", "DSS", "FK5", Some(2000.0)));
  let ra0 = 15.0 * (13.0 + 35.0 / 60.0 + 0.40093 / 3600.0);
  assert!(close(w.crval_frame_deg[0], ra0, 1e-9) && w.crval_frame_deg[1] < -45.0 && w.crval_frame_deg[1] > -46.0);
  let xsz = h.get_f64("XPIXELSZ").unwrap();
  assert!(close(w.crpix[0], h.get_f64("PPO3").unwrap() / xsz - h.get_f64("CNPIX1").unwrap() + 0.5, 1e-9));
  assert!(close(w.cd_deg[0][0], -xsz / 1000.0, 1e-15) && w.cd_deg[0][1] == 0.0);
  let (ra, dec) = w.pixel_to_radec_deg(100.0, 600.0);
  let (x, y) = w.radec_to_pixel(ra, dec).unwrap();
  assert!(close(x, 100.0, 1e-8) && close(y, 600.0, 1e-8), "{x} {y}");
  // the same header with its GetImage TAN cards is read as that TAN (as wcslib does)
  let mut cards = h.cards.clone();
  cards.push(("CTYPE1".into(), "'RA---TAN'".into()));
  cards.push(("CTYPE2".into(), "'DEC--TAN'".into()));
  cards.push(("CRVAL1".into(), "201.365".into()));
  cards.push(("CRVAL2".into(), "-43.019".into()));
  cards.push(("CRPIX1".into(), "353.0".into()));
  cards.push(("CRPIX2".into(), "353.0".into()));
  cards.push(("CDELT1".into(), "-0.00047".into()));
  cards.push(("CDELT2".into(), "0.00047".into()));
  let text: Vec<String> = cards.iter().map(|(k, v)| format!("{k:<8}= {v}")).collect();
  let w2 = Wcs::from_header(&FitsHeader::parse_text(text.join("\n").as_bytes())).unwrap().unwrap();
  assert_eq!(w2.facts().distortion, "none");
  assert!(w2.is_plain_tan());
}

// ─── helioprojective ─────────────────────────────────────────────────────────

#[test]
fn earth_ephemeris_matches_erfa() {
  // pyerfa epv00 heliocentric Earth positions (au, BCRS) at four TDB Julian dates
  let cases = [
    (2451545.0, [-0.17713507281322974, 0.8874285242954301, 0.3847428889988798]),
    (2455000.5, [-0.05715151879121985, -0.9307208182573806, -0.4034955092982167]),
    (2459000.5, [-0.3511286723084591, -0.8726911877185966, -0.3783119873302871]),
    (2462000.5, [0.821393348993706, -0.5430685406141885, -0.23541285604428303]),
  ];
  for (jd, e) in cases {
    let g = solar::sun_to_earth_au(jd);
    let (lg, le) = (g.iter().map(|v| v * v).sum::<f64>().sqrt(), e.iter().map(|v| v * v).sum::<f64>().sqrt());
    let cosang = (g[0] * e[0] + g[1] * e[1] + g[2] * e[2]) / (lg * le);
    let arcsec = cosang.clamp(-1.0, 1.0).acos() / DEG * 3600.0;
    assert!(arcsec < 0.1, "jd {jd}: {arcsec}″");
    assert!((lg - le).abs() < 1e-6, "jd {jd}: |r| {lg} vs {le}");
  }
}

#[test]
fn leap_seconds_and_fits_dates() {
  assert_eq!(solar::jd_of_date(2000, 1, 1), 2451544.5);
  assert_eq!(solar::jd_of_date(1858, 11, 17), 2400000.5);
  assert_eq!(solar::tai_minus_utc(solar::jd_of_date(2017, 1, 1)), 37.0);
  assert_eq!(solar::tai_minus_utc(solar::jd_of_date(2016, 12, 31)), 36.0);
  assert_eq!(solar::tai_minus_utc(solar::jd_of_date(1990, 6, 1)), 25.0);
  assert_eq!(solar::tai_minus_utc(solar::jd_of_date(1960, 1, 1)), 10.0);
  assert!(close(solar::jd_tt_of_utc(2456352.5055767708), 2456352.5063543636, 1e-9)); // astropy
  assert!(close(solar::parse_fits_date("2013-03-01T00:08:01.833").unwrap(), 2456352.5 + (8.0 * 60.0 + 1.833) / 86400.0, 1e-10));
  assert!(close(solar::parse_fits_date("2011-06-07T06:33:02.77").unwrap(), 2455719.772948727, 1e-9));
  assert_eq!(solar::parse_fits_date("11/03/75"), Some(solar::jd_of_date(1975, 3, 11)));
  assert_eq!(solar::parse_fits_date("1953-10-10"), Some(solar::jd_of_date(1953, 10, 10)));
  assert_eq!(solar::parse_fits_date("1996-10-14 10:14:36.123"), Some(solar::jd_of_date(1996, 10, 14) + (10.0 * 3600.0 + 14.0 * 60.0 + 36.123) / 86400.0));
  assert_eq!(solar::parse_fits_date("2013-13-01"), Option::None);
  assert_eq!(solar::parse_fits_date(""), Option::None);
  let h = header(&[("DATE-OBS", "'2011-06-07T06:33:02.77'"), ("DATE-AVG", "'2011-06-07T06:33:03.77'")]);
  assert!(close(solar::obstime_jd_utc(&h).unwrap(), 2455719.772948727 + 1.0 / 86400.0, 1e-9));
  let h = header(&[("MJD-OBS", "56352.0"), ("BITPIX", "16")]);
  assert_eq!(solar::obstime_jd_utc(&h), Some(2456352.5));
}

#[test]
fn helioprojective_frame_matches_sunpy() {
  // sunpy 8: Helioprojective (θx, θy) → HCRS direction (point − observer), two observers
  let cases: [(&[(&str, &str)], f64, f64, f64, f64, bool); 2] = [
    (
      &[("DATE-AVG", "'2013-03-01T00:08:01.833'"), ("HGLN_OBS", "130.849706562"), ("HGLT_OBS", "4.34198737590")],
      1.0,
      0.5,
      112.08310887250522,
      22.16235342970165,
      false,
    ),
    (&[("DATE-OBS", "'2011-06-07T06:33:02.77'"), ("BITPIX", "16")], 0.1, -0.2, 74.78612488993342, 22.488493829364558, true),
  ];
  for (cards, tx, ty, ra, dec, earth) in cases {
    let h = header(cards);
    let (frame, assumed) = solar::frame_from_header(&h).unwrap();
    assert_eq!(assumed, earth);
    assert_eq!(frame.kind, FrameKind::Helioprojective);
    let (gra, gdec) = frame.lonlat_to_icrs(tx, ty);
    let e = sep((gra, gdec), (ra, dec)) * 3600.0;
    assert!(e < 0.1, "({tx}, {ty}) → ({gra}, {gdec}), sunpy ({ra}, {dec}): {e}″");
    let (btx, bty) = frame.icrs_to_lonlat(gra, gdec);
    assert!(close(btx, tx, 1e-10) && close(bty, ty, 1e-10));
    // the frame is an orthogonal map: a unit vector stays one and the inverse closes
    let v = frame.to_icrs(dir_of(33.0, -12.0));
    assert!(close(v.iter().map(|c| c * c).sum::<f64>(), 1.0, 1e-12));
    let b = frame.from_icrs(v);
    let d = dir_of(33.0, -12.0);
    assert!((0..3).all(|i| close(b[i], d[i], 1e-12)));
  }
  // B0 of the AIA date (sunpy 0.03652003237°; the truncated ephemeris is good to 0.07″)
  assert!(close(solar::b0_deg(2455719.7737147454), 0.03652003237, 0.1 / 3600.0));
  // a helioprojective header through the WCS: facts and the reference point
  let h = header(&[
    ("CTYPE1", "'HPLN-TAN'"),
    ("CTYPE2", "'HPLT-TAN'"),
    ("CUNIT1", "'arcsec'"),
    ("CUNIT2", "'arcsec'"),
    ("CRVAL1", "3600.0"),
    ("CRVAL2", "1800.0"),
    ("CRPIX1", "100.0"),
    ("CRPIX2", "100.0"),
    ("CDELT1", "2.4"),
    ("CDELT2", "2.4"),
    ("DATE-OBS", "'2013-03-01T00:08:01.833'"),
    ("HGLN_OBS", "130.849706562"),
    ("HGLT_OBS", "4.34198737590"),
  ]);
  let w = Wcs::from_header(&h).unwrap().unwrap();
  let f = w.facts();
  assert_eq!((f.projection, f.distortion, f.frame, f.equinox, f.frame_assumed), ("TAN", "none", "helioprojective", Option::None, false));
  assert!(sep((w.crval_deg[0], w.crval_deg[1]), (112.08310887250522, 22.16235342970165)) * 3600.0 < 0.1);
  let (ra, dec) = w.pixel_to_radec_deg(100.0, 100.0);
  assert!(sep((ra, dec), (112.08310887250522, 22.16235342970165)) * 3600.0 < 0.1);
  // without a date it is refused, naming the cards
  let e = Wcs::from_header(&header(&[("CTYPE1", "'HPLN-TAN'"), ("CTYPE2", "'HPLT-TAN'")])).unwrap_err();
  assert!(e.contains("DATE-OBS"), "{e}");
}

#[test]
fn solar_position_systems_are_refused_with_the_reason() {
  for (a, b, word) in [
    ("HGLN-CAR", "HGLT-CAR", "Stonyhurst heliographic"),
    ("CRLN-CEA", "CRLT-CEA", "Carrington heliographic"),
    ("HRLN-TAN", "HRLT-TAN", "heliocentric radial"),
    ("SOLX", "SOLY", "heliocentric cartesian"),
    ("MALN-TAN", "MALT-TAN", "body-fixed"),
  ] {
    let e = Wcs::from_header(&header(&[("CTYPE1", &format!("'{a}'")), ("CTYPE2", &format!("'{b}'"))])).unwrap_err();
    assert!(e.contains(word) && e.contains("not sky directions"), "{a}: {e}");
  }
  // unknown non-celestial axes are simply not a WCS
  assert_eq!(Wcs::from_header(&header(&[("CTYPE1", "'WAVE'"), ("CTYPE2", "'TIME'")])).unwrap(), Option::None);
}
