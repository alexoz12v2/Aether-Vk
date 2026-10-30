//! Astrometry for the Earth observer: frames, RA/Dec, the WGS84 site, light time, aberration.
//!
//! **Frames.** The engine's world axes are SPICE `ECLIPJ2000` centred on the Sun (`SUN_ECLIPJ2000`:
//! +X vernal equinox, +Z ecliptic north, see `SKILL.md` "Axes and units"). Equatorial J2000 (the
//! ICRF axes of the DE ephemerides, which is what ANISE calls `J2000`) is a rotation about +X by the
//! obliquity. SPICE (and ANISE, `J2000_TO_ECLIPJ2000_ANGLE_RAD`) define `ECLIPJ2000` with the IAU
//! 1976 obliquity 84381.448″, so [`OBLIQUITY_J2000_RAD`] is that value: the inverse of the rotation
//! that produced the engine's positions. (The IAU 2006 value 84381.406″,
//! [`OBLIQUITY_IAU2006_RAD`], differs by 0.042″; using it would add that error to every RA/Dec.)
//!
//! **Light time.** A telescope at time `t` sees the comet where it was at the retarded epoch
//! `t_r = t − Δ(t_r)/c`, Δ the topocentric distance. [`retarded_epoch`] runs two fixed-point
//! iterations from `t_r = t`: each multiplies the error by `|dΔ/dt|/c` (≤ 1e-4 for a comet), so
//! after two the epoch is good to ~1e-8 of the delay (~1e-5 s at 6 AU, < 1 m of comet motion). Only the comet and its dust are
//! retarded in the render (`earth_observer::comet_light_time`, `Scene::dust_draw_calls`); the
//! planets and the Sun stay geometric (their light time is a separate item). The simulation clock
//! is untouched: only the evaluation epoch of the view changes.
//!
//! **Aberration decision: not applied in the render.** A WCS solved against a star catalogue
//! (astrometry.net, Gaia) maps image pixels to *catalogue* (astrometric, ICRF) directions: the
//! stars' annual aberration (≤ 20.5″) is absorbed by the plate solution. The rendered comet is
//! astrometric (light time, no aberration), so the registered overlay and the comet agree without
//! it, exactly like the astrometric RA/Dec that Horizons and the MPC publish for comparison with
//! plate solutions. [`apparent_from_astrometric`] exists for the record and is tested; nothing in
//! the render calls it.
//!
//! **Site.** [`wgs84_surface_point_bf`]: geodetic latitude / longitude / ellipsoidal height on the
//! WGS84 ellipsoid, in the Earth body-fixed frame (ITRF93, the frame of the Earth rotation the
//! almanac commits, `AlmanacPlanet::rotation_at`), in AU like `earth_observer`.

type V3 = [f64; 3];

/// one arcsecond (rad)
pub const ARCSEC_RAD: f64 = core::f64::consts::PI / (180.0 * 3600.0);
/// Obliquity of the ecliptic that defines SPICE/ANISE `ECLIPJ2000` (IAU 1976, 84381.448″): the
/// engine's world frame. Equals ANISE's `J2000_TO_ECLIPJ2000_ANGLE_RAD` (asserted by a test).
pub const OBLIQUITY_J2000_RAD: f64 = 84_381.448 * ARCSEC_RAD;
/// IAU 2006 obliquity at J2000 (84381.406″), for the record: not the engine's ecliptic.
pub const OBLIQUITY_IAU2006_RAD: f64 = 84_381.406 * ARCSEC_RAD;
/// speed of light (km/s)
pub const C_KM_S: f64 = 299_792.458;
/// WGS84 equatorial radius (km)
pub const WGS84_A_KM: f64 = 6378.137;
/// WGS84 flattening
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// km per AU (alias of `reposition::AU_TO_KM`)
pub const AU_KM: f64 = crate::simulation_api::reposition::AU_TO_KM;

fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm(a: V3) -> f64 {
  dot(a, a).sqrt()
}
fn scale(a: V3, s: f64) -> V3 {
  [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
  [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Ecliptic J2000 (the engine's world axes) → equatorial J2000 (ICRF axes): `R1(−ε)`.
pub fn ecliptic_to_equatorial_j2000(v: V3) -> V3 {
  let (s, c) = OBLIQUITY_J2000_RAD.sin_cos();
  [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]]
}

/// Equatorial J2000 (ICRF axes) → ecliptic J2000 (the engine's world axes): `R1(ε)`, ANISE's
/// `J2000 → ECLIPJ2000` rotation.
pub fn equatorial_to_ecliptic_j2000(v: V3) -> V3 {
  let (s, c) = OBLIQUITY_J2000_RAD.sin_cos();
  [v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]]
}

/// Right ascension in `[0, 2π)` and declination in `[−π/2, π/2]` (rad) of an equatorial direction
/// (any length > 0).
pub fn ra_dec_of(dir_eq: V3) -> (f64, f64) {
  let ra = dir_eq[1].atan2(dir_eq[0]);
  let ra = if ra < 0.0 {
    ra + core::f64::consts::TAU
  } else {
    ra
  };
  let dec = dir_eq[2].atan2(dir_eq[0].hypot(dir_eq[1]));
  (ra, dec)
}

/// Unit equatorial direction of `(ra, dec)` (rad).
pub fn dir_of_ra_dec(ra: f64, dec: f64) -> V3 {
  let (sd, cd) = dec.sin_cos();
  let (sa, ca) = ra.sin_cos();
  [cd * ca, cd * sa, sd]
}

/// Geodetic site on the WGS84 ellipsoid, Earth body-fixed (ITRF93: +X Greenwich meridian on the
/// equator, +Z north pole), in **AU** (the unit of `EarthObserverState::surface_bf_au`).
/// `elev_m` is the height above the ellipsoid.
pub fn wgs84_surface_point_bf(lat_deg: f64, lon_deg: f64, elev_m: f64) -> V3 {
  let (sp, cp) = lat_deg.to_radians().sin_cos();
  let (sl, cl) = lon_deg.to_radians().sin_cos();
  let e2 = WGS84_F * (2.0 - WGS84_F);
  let n = WGS84_A_KM / (1.0 - e2 * sp * sp).sqrt();
  let h = elev_m * 1e-3;
  let km = [
    (n + h) * cp * cl,
    (n + h) * cp * sl,
    (n * (1.0 - e2) + h) * sp,
  ];
  scale(km, 1.0 / AU_KM)
}

/// Unit geodetic zenith (the ellipsoid normal) at `(lat, lon)`, Earth body-fixed. Differs from the
/// geocentric direction of the site by up to 0.19° (at 45°).
pub fn wgs84_zenith_bf(lat_deg: f64, lon_deg: f64) -> V3 {
  let (sp, cp) = lat_deg.to_radians().sin_cos();
  let (sl, cl) = lon_deg.to_radians().sin_cos();
  [cp * cl, cp * sl, sp]
}

/// Retarded epoch `t_r = t − Δ(t_r)/c` (s, same scale as `t`): two fixed-point iterations from
/// `t_r = t`. `distance_km(t_r)` is the distance from the observer **at `t`** to the target at
/// `t_r` (km). See the module doc for the convergence bound.
pub fn retarded_epoch(t: f64, mut distance_km: impl FnMut(f64) -> f64) -> f64 {
  let mut t_r = t;
  for _ in 0..2 {
    t_r = t - distance_km(t_r) / C_KM_S;
  }
  t_r
}

/// Astrometric (light-time corrected, no aberration) direction from `observer_km` (at `t`) to the
/// target whose position at any epoch is `target_km_at(epoch)` (same frame and origin as the
/// observer). Returns `(unit direction, t_r, distance km)`.
pub fn topocentric_astrometric(
  observer_km: V3,
  t: f64,
  mut target_km_at: impl FnMut(f64) -> V3,
) -> (V3, f64, f64) {
  let t_r = retarded_epoch(t, |tr| norm(sub(target_km_at(tr), observer_km)));
  let d = sub(target_km_at(t_r), observer_km);
  let n = norm(d);
  (scale(d, 1.0 / n), t_r, n)
}

/// Heliocentric two-body state `(r km, v km/s)` propagated by `dt_s` (either sign): RK4 under the
/// Sun's gravity alone, 4 steps (relative local error ~(h·ω)⁵ < 1e-20 for |dt| of hours). Used
/// for the comet's light-time displacement (`|dt|` ≤ an hour): planetary perturbations over that
/// span move a comet by < 1 m.
pub fn two_body_propagate(r_km: V3, v_km_s: V3, dt_s: f64) -> (V3, V3) {
  let mu = crate::simulation::orbit_elements::MU_SUN_KM3_S2;
  let acc = |r: V3| {
    let d = norm(r);
    scale(r, -mu / (d * d * d))
  };
  let steps = 4;
  let h = dt_s / steps as f64;
  let (mut r, mut v) = (r_km, v_km_s);
  for _ in 0..steps {
    let (k1r, k1v) = (v, acc(r));
    let (k2r, k2v) = (
      add(v, scale(k1v, 0.5 * h)),
      acc(add(r, scale(k1r, 0.5 * h))),
    );
    let (k3r, k3v) = (
      add(v, scale(k2v, 0.5 * h)),
      acc(add(r, scale(k2r, 0.5 * h))),
    );
    let (k4r, k4v) = (add(v, scale(k3v, h)), acc(add(r, scale(k3r, h))));
    r = add(
      r,
      scale(
        add(add(k1r, scale(k2r, 2.0)), add(scale(k3r, 2.0), k4r)),
        h / 6.0,
      ),
    );
    v = add(
      v,
      scale(
        add(add(k1v, scale(k2v, 2.0)), add(scale(k3v, 2.0), k4v)),
        h / 6.0,
      ),
    );
  }
  (r, v)
}

/// Apparent direction (stellar/annual aberration) of an astrometric unit direction `dir` seen by
/// an observer moving at `observer_velocity_km_s` (relative to the frame of `dir`; the barycentric
/// Earth velocity for annual aberration). Special-relativistic formula
/// `u' = (u/γ + β + (u·β) β/(1 + 1/γ)) / (1 + u·β)`, normalised. Magnitude ≈ |β| sin θ (20.5″ at
/// 29.78 km/s). **Not used by the render** (see the module doc).
pub fn apparent_from_astrometric(dir: V3, observer_velocity_km_s: V3) -> V3 {
  let u = scale(dir, 1.0 / norm(dir));
  let b = scale(observer_velocity_km_s, 1.0 / C_KM_S);
  let b2 = dot(b, b);
  let inv_gamma = (1.0 - b2).sqrt();
  let ub = dot(u, b);
  let num = add(scale(u, inv_gamma), scale(b, 1.0 + ub / (1.0 + inv_gamma)));
  let w = scale(num, 1.0 / (1.0 + ub));
  scale(w, 1.0 / norm(w))
}

fn cross(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

/// angle between `a` and `b` (rad): `atan2(|a × b|, a · b)`, accurate near 0 and π unlike `acos`
fn angle_between(a: V3, b: V3) -> f64 {
  norm(cross(a, b)).atan2(dot(a, b))
}

/// The comet's ephemeris seen from the Earth at one instant ([`comet_ephemeris`]): the rows of a
/// JPL Horizons observer table, with its conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CometEphemeris {
  /// astrometric geocentric right ascension, J2000/ICRF (rad, `[0, 2π)`)
  pub ra_rad: f64,
  /// astrometric geocentric declination, J2000/ICRF (rad)
  pub dec_rad: f64,
  /// heliocentric distance `r` of the comet as seen, i.e. at the retarded epoch (AU); differs from
  /// the geometric one by `ṙ · Δ/c` (< 1e-4 AU for a comet)
  pub r_au: f64,
  /// geocentric distance `Δ` (AU): the length of the light path, Earth at `t` to comet at `t_r`
  pub delta_au: f64,
  /// phase angle: Sun–comet–Earth at the comet (rad), Horizons' `S-T-O`
  pub phase_rad: f64,
  /// solar elongation: Sun–Earth–comet at the Earth (rad), Horizons' `S-O-T`
  pub elongation_rad: f64,
  /// position angle of the extended Sun→comet radius vector on the sky at the comet (rad,
  /// `[0, 2π)`): Horizons' `PsAng`, counter-clockwise from celestial (ICRF) north through east.
  /// The anti-solar direction a tail points away from; `PsAng + 180°` points at the Sun.
  pub sun_pa_rad: f64,
  /// light time `Δ/c` (s)
  pub light_time_s: f64,
}

/// Geocentric astrometric ephemeris of a comet at `comet_au` moving at `comet_vel_km_s` (both
/// heliocentric, world = ecliptic J2000 axes) seen from the Earth's centre at `earth_au` at the same
/// instant: light time as the Earth observer renders it (`earth_observer::comet_light_time`: two
/// iterations of [`retarded_epoch`] on the comet's two-body path), no aberration (the module doc's
/// decision: Horizons' astrometric RA/Dec). Angles follow Horizons' observer-table definitions, see
/// [`CometEphemeris`]; `r` is the heliocentric distance of the comet as seen (Horizons' `r`,
/// light-time corrected).
///
/// **PsAng convention.** With `l̂` the unit line of sight (Earth → comet, equatorial) the sky-plane
/// basis at the comet is `east = ẑ × l̂` (normalised) and `north = l̂ × east`; the position angle
/// of a vector `v` is `atan2(v · east, v · north)` in `[0, 2π)`, i.e. 0 at north, 90° at east.
/// `v` is the heliocentric position of the comet as seen (Sun → comet), so the angle is that of the
/// anti-solar direction projected on the sky, which is what Horizons' PsAng is ("position angle of
/// the extended Sun-to-target radius vector ... measured counter-clockwise from reference-frame
/// north celestial pole"). On the celestial pole itself the basis is degenerate and the angle is 0.
pub fn comet_ephemeris(earth_au: V3, comet_au: V3, comet_vel_km_s: V3) -> CometEphemeris {
  let lt = super::earth_observer::comet_light_time(earth_au, comet_au, comet_vel_km_s);
  // the comet where the Earth sees it (heliocentric, ecliptic axes)
  let seen = add(comet_au, lt.offset_au);
  let to_comet = sub(seen, earth_au);
  let delta_au = norm(to_comet);
  let los = ecliptic_to_equatorial_j2000(scale(to_comet, 1.0 / delta_au));
  let (ra_rad, dec_rad) = ra_dec_of(los);
  // at the comet: comet → Sun is −seen, comet → Earth is −to_comet (the sign cancels)
  let phase_rad = angle_between(seen, to_comet);
  // at the Earth: Earth → Sun is −earth
  let elongation_rad = angle_between(scale(earth_au, -1.0), to_comet);
  let east = cross([0.0, 0.0, 1.0], los);
  let east_norm = norm(east);
  let sun_pa_rad = if east_norm < 1e-12 {
    0.0
  } else {
    let east = scale(east, 1.0 / east_norm);
    let north = cross(los, east);
    let v = ecliptic_to_equatorial_j2000(seen);
    let pa = dot(v, east).atan2(dot(v, north));
    if pa < 0.0 {
      pa + core::f64::consts::TAU
    } else {
      pa
    }
  };
  CometEphemeris {
    ra_rad,
    dec_rad,
    r_au: norm(seen),
    delta_au,
    phase_rad,
    elongation_rad,
    sun_pa_rad,
    light_time_s: lt.delay_s,
  }
}

/// [`comet_ephemeris`] of the SPK body `naif` at `epoch` from the almanac: the comet's and the
/// Earth's (399) heliocentric states in `SUN_ECLIPJ2000` (the states a commit places the bodies
/// at). `None` when the almanac does not cover either body at `epoch`.
pub fn comet_ephemeris_at(
  almanac: &crate::simulation::almanac::AlmanacPackedData,
  naif: i32,
  epoch: hifitime::Epoch,
) -> Option<CometEphemeris> {
  let frame = crate::simulation::almanac::SUN_ECLIPJ2000;
  let state = |id: i32| {
    almanac
      .get_cartesian_state(id, frame.orientation_id, frame.ephemeris_id, epoch, false)
      .ok()
      .map(|s| {
        (
          [s.radius_km[0], s.radius_km[1], s.radius_km[2]],
          [s.velocity_km_s[0], s.velocity_km_s[1], s.velocity_km_s[2]],
        )
      })
  };
  let (comet_km, comet_vel_km_s) = state(naif)?;
  let (earth_km, _) = state(anise::constants::celestial_objects::EARTH)?;
  Some(comet_ephemeris(
    scale(earth_km, 1.0 / AU_KM),
    scale(comet_km, 1.0 / AU_KM),
    comet_vel_km_s,
  ))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn angle(a: V3, b: V3) -> f64 {
    let c = [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
    norm(c).atan2(dot(a, b))
  }

  /// The world frame is ANISE's ECLIPJ2000: the obliquity is its constant to the last bit, and the
  /// IAU 2006 value is 0.042″ smaller.
  #[test]
  fn obliquity_is_the_anise_eclipj2000_angle() {
    assert_eq!(
      OBLIQUITY_J2000_RAD,
      anise::constants::orientations::J2000_TO_ECLIPJ2000_ANGLE_RAD
    );
    let d = (OBLIQUITY_J2000_RAD - OBLIQUITY_IAU2006_RAD) / ARCSEC_RAD;
    assert!((d - 0.042).abs() < 1e-9, "{d}");
  }

  /// Frame and RA/Dec round trips to 1e-12; the ecliptic north pole is at RA 18h, Dec 90° − ε
  /// (+66.56°); the equinox (+X) is RA 0 Dec 0 in both frames.
  #[test]
  fn frames_and_ra_dec_round_trip() {
    let mut seed = 0x1234_5678u64;
    let mut rnd = || {
      seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for _ in 0..1000 {
      let v = [rnd() * 3.0, rnd() * 3.0, rnd() * 3.0];
      let back = equatorial_to_ecliptic_j2000(ecliptic_to_equatorial_j2000(v));
      assert!(
        norm(sub(back, v)) < 1e-12 * norm(v).max(1.0),
        "{v:?} {back:?}"
      );
      let back = ecliptic_to_equatorial_j2000(equatorial_to_ecliptic_j2000(v));
      assert!(norm(sub(back, v)) < 1e-12 * norm(v).max(1.0));
      let (ra, dec) = ra_dec_of(v);
      assert!((0.0..core::f64::consts::TAU).contains(&ra));
      let u = dir_of_ra_dec(ra, dec);
      assert!(norm(sub(u, scale(v, 1.0 / norm(v)))) < 1e-12, "{v:?}");
    }
    let pole = ecliptic_to_equatorial_j2000([0.0, 0.0, 1.0]);
    let (ra, dec) = ra_dec_of(pole);
    assert!(
      (ra.to_degrees() - 270.0).abs() < 1e-12,
      "RA {} h",
      ra.to_degrees() / 15.0
    );
    assert!(
      (dec - (core::f64::consts::FRAC_PI_2 - OBLIQUITY_J2000_RAD)).abs() < 1e-12,
      "Dec {}°",
      dec.to_degrees()
    );
    assert!(
      (dec.to_degrees() - 66.560_708).abs() < 1e-5,
      "{}",
      dec.to_degrees()
    );
    assert_eq!(
      ra_dec_of(ecliptic_to_equatorial_j2000([1.0, 0.0, 0.0])),
      (0.0, 0.0)
    );
    // the summer solstice point (ecliptic +Y) is at RA 6h, Dec +ε
    let (ra, dec) = ra_dec_of(ecliptic_to_equatorial_j2000([0.0, 1.0, 0.0]));
    assert!((ra.to_degrees() - 90.0).abs() < 1e-12);
    assert!((dec - OBLIQUITY_J2000_RAD).abs() < 1e-12);
  }

  /// WGS84: Horizons' "Center cylindric" for lat 45°, lon 0°, h 0 is (Dxy, Dz) = (4517.59088,
  /// 4487.34841) km; the equator is at `a`, the pole at `b = a(1 − f)` = 6356.752 km; elevation adds
  /// along the geodetic normal.
  #[test]
  fn wgs84_site_matches_horizons_cylindrical() {
    let p = scale(wgs84_surface_point_bf(45.0, 0.0, 0.0), AU_KM);
    assert!((p[0].hypot(p[1]) - 4517.590_88).abs() < 1e-5, "{p:?}");
    assert!((p[2] - 4487.348_41).abs() < 1e-5, "{p:?}");
    let eq = scale(wgs84_surface_point_bf(0.0, 90.0, 0.0), AU_KM);
    assert!((eq[1] - WGS84_A_KM).abs() < 1e-9 && eq[0].abs() < 1e-9);
    let np = scale(wgs84_surface_point_bf(90.0, 0.0, 0.0), AU_KM);
    assert!((np[2] - 6356.752_314).abs() < 1e-6, "{np:?}");
    let up = sub(
      scale(wgs84_surface_point_bf(45.0, 30.0, 2500.0), AU_KM),
      scale(wgs84_surface_point_bf(45.0, 30.0, 0.0), AU_KM),
    );
    assert!((norm(up) - 2.5).abs() < 1e-9);
    assert!(angle(up, wgs84_zenith_bf(45.0, 30.0)) < 1e-9);
    // the geodetic zenith leans 0.19° from the geocentric direction at 45°
    let g = angle(
      wgs84_surface_point_bf(45.0, 0.0, 0.0),
      wgs84_zenith_bf(45.0, 0.0),
    );
    assert!((g.to_degrees() - 0.1924).abs() < 1e-3, "{}", g.to_degrees());
  }

  /// Light time on a target moving at 30 km/s, 5 AU away: two iterations solve `t_r = t − Δ(t_r)/c`
  /// to the bound of the module doc, `τ·(Δ̇/c)²` ≈ 1e-5 s here (0.3 m of the target's motion), and
  /// the delay is Δ/c.
  #[test]
  fn retarded_epoch_solves_the_light_time_equation() {
    let obs = [0.0, 0.0, 0.0];
    let r0 = [5.0 * AU_KM, 0.0, 0.0];
    let v = [20.0, 22.0, 0.5]; // km/s, receding
    let at = |t: f64| add(r0, scale(v, t));
    let t = 1000.0;
    let t_r = retarded_epoch(t, |tr| norm(sub(at(tr), obs)));
    let residual = t_r - (t - norm(at(t_r)) / C_KM_S);
    let bound = (t - t_r) * (20.0 / C_KM_S).powi(2) * 1.5;
    assert!(residual.abs() < bound, "{residual} vs {bound}");
    assert!((t - t_r - 5.0 * AU_KM / C_KM_S).abs() < 1.0, "{}", t - t_r);
    let (dir, tr2, d) = topocentric_astrometric(obs, t, at);
    assert_eq!(tr2, t_r);
    assert!((d - norm(at(t_r))).abs() < 1e-9);
    assert!(angle(dir, at(t_r)) < 1e-15);
  }

  /// The two-body propagation reverses (forward then back to 1 mm) and matches the universal-
  /// variable Kepler conic (`dust::kepler::propagate_f64`) over an hour at 1 AU to 1 m.
  #[test]
  fn two_body_propagation_matches_the_conic() {
    use crate::simulation::orbit_elements::MU_SUN_KM3_S2;
    let r = [1.2e8, -0.9e8, 0.1e8];
    let v = [18.0, 21.0, -1.0];
    let (r1, v1) = two_body_propagate(r, v, -3600.0);
    let (r2, _) = two_body_propagate(r1, v1, 3600.0);
    assert!(norm(sub(r2, r)) < 1e-6, "{}", norm(sub(r2, r)));
    let (rk, _) = crate::scene::dust::kepler::propagate_f64(r, v, MU_SUN_KM3_S2, -3600.0);
    assert!(norm(sub(rk, r1)) < 1e-3, "{} km", norm(sub(rk, r1)));
  }

  /// Annual aberration: perpendicular to a 29.7847 km/s velocity (the Earth's mean orbital speed)
  /// the shift is `atan(βγ)` = 20.49″, towards the apex, never more; along the velocity it
  /// vanishes; `≤ 20.5″` for every direction.
  #[test]
  fn aberration_magnitude_is_the_constant_of_aberration() {
    let v = [0.0, 29.7847, 0.0];
    let perp = apparent_from_astrometric([1.0, 0.0, 0.0], v);
    let a = angle(perp, [1.0, 0.0, 0.0]) / ARCSEC_RAD;
    // perpendicular: tan Δθ = β·γ exactly; 20.49″ (κ = 20.4955″ is the IAU constant for the
    // mean motion, 29.7859 km/s)
    let beta = v[1] / C_KM_S;
    let exact = (beta / (1.0 - beta * beta).sqrt()).atan() / ARCSEC_RAD;
    assert!((a - exact).abs() < 1e-6, "{a}″ vs {exact}″");
    assert!((a - 20.4925).abs() < 1e-3, "{a}″");
    assert!(perp[1] > 0.0, "shifted towards the apex");
    let along = apparent_from_astrometric([0.0, 1.0, 0.0], v);
    assert!(angle(along, [0.0, 1.0, 0.0]) < 1e-15);
    for k in 0..360 {
      let th = (k as f64).to_radians();
      let u = [th.cos(), th.sin(), 0.0];
      let s = angle(apparent_from_astrometric(u, v), u) / ARCSEC_RAD;
      assert!(s <= 20.5, "{s}″ at {k}°");
    }
    // 29.8 km/s: 20.50″ (the WP4 bound, to 0.01″)
    let s = angle(
      apparent_from_astrometric([1.0, 0.0, 0.0], [0.0, 29.8, 0.0]),
      [1.0, 0.0, 0.0],
    ) / ARCSEC_RAD;
    assert!((s - 20.5).abs() < 0.01, "{s}″");
  }

  /// Great-circle initial bearing from `(ra1, dec1)` to `(ra2, dec2)`, north through east, in
  /// `[0, 2π)`: the textbook position-angle formula, independent of the sky-plane basis
  /// `comet_ephemeris` builds.
  fn bearing(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let d = ra2 - ra1;
    let y = d.sin() * dec2.cos();
    let x = dec1.cos() * dec2.sin() - dec1.sin() * dec2.cos() * d.cos();
    let pa = y.atan2(x);
    if pa < 0.0 { pa + core::f64::consts::TAU } else { pa }
  }

  /// Every value of a [`CometEphemeris`] re-derived from the geometry it was computed from: RA/Dec
  /// from the light path, `r`/`Δ`/phase/elongation by the law of cosines on the Sun–Earth–comet
  /// triangle, PsAng by the great-circle bearing from the comet to a point a hair along the
  /// extended Sun→comet vector, the light time from `Δ`.
  fn check_consistent(earth: V3, e: &CometEphemeris, tol_rad: f64) {
    // the comet as seen, rebuilt from the RA/Dec and Δ
    let los_eq = dir_of_ra_dec(e.ra_rad, e.dec_rad);
    let seen = add(earth, scale(equatorial_to_ecliptic_j2000(los_eq), e.delta_au));
    assert!((norm(seen) - e.r_au).abs() < 1e-9, "r {} vs {}", norm(seen), e.r_au);
    let r_earth = norm(earth);
    let (r, d) = (e.r_au, e.delta_au);
    let phase = ((r * r + d * d - r_earth * r_earth) / (2.0 * r * d)).clamp(-1.0, 1.0).acos();
    assert!((phase - e.phase_rad).abs() < tol_rad, "phase {phase} vs {}", e.phase_rad);
    let elong = ((r_earth * r_earth + d * d - r * r) / (2.0 * r_earth * d)).clamp(-1.0, 1.0).acos();
    assert!((elong - e.elongation_rad).abs() < tol_rad, "elongation {elong} vs {}", e.elongation_rad);
    // the delay is Δ(t_r¹)/c, Δ the path after the second iteration: they differ by (ṙ/c)·Δ/c
    assert!((e.light_time_s - d * AU_KM / C_KM_S).abs() < 0.1, "light time {}", e.light_time_s);
    // PsAng: bearing to the point 1e-7 AU further along Sun → comet
    let ahead = add(seen, scale(seen, 1e-7 / norm(seen)));
    let (ra2, dec2) = ra_dec_of(ecliptic_to_equatorial_j2000(sub(ahead, earth)));
    let pa = bearing(e.ra_rad, e.dec_rad, ra2, dec2);
    let dpa = (pa - e.sun_pa_rad + core::f64::consts::PI).rem_euclid(core::f64::consts::TAU) - core::f64::consts::PI;
    assert!(dpa.abs() < tol_rad, "PsAng {}° vs {}°", pa.to_degrees(), e.sun_pa_rad.to_degrees());
  }

  /// Synthetic geometry: a comet at rest 1.5 AU out (no light-time displacement to speak of),
  /// RA/Dec of the light path, the triangle angles, the bearing, and the PsAng conventions at
  /// known configurations (opposition: PsAng is undefined-ish but finite; a comet due north of the
  /// Earth on the sky with the Sun behind: 180°).
  #[test]
  fn comet_ephemeris_matches_an_independent_derivation() {
    let earth = [0.98, 0.17, 0.0];
    let comet = [-0.4, 1.3, 0.5];
    let e = comet_ephemeris(earth, comet, [0.0, 0.0, 0.0]);
    let to = sub(comet, earth);
    let (ra, dec) = ra_dec_of(ecliptic_to_equatorial_j2000(to));
    // a comet at rest falls < 1 km in the light time: the astrometric place is the geometric one
    assert!((ra - e.ra_rad).abs() < 1e-7 && (dec - e.dec_rad).abs() < 1e-7);
    assert!((e.delta_au - norm(to)).abs() < 1e-7 && (e.r_au - norm(comet)).abs() < 1e-7);
    check_consistent(earth, &e, 1e-6);
    assert!((0.0..core::f64::consts::TAU).contains(&e.sun_pa_rad));
    assert!(e.phase_rad > 0.0 && e.elongation_rad > 0.0);
    // the phase angle also equals π − elongation − (Earth seen from the comet), i.e. the triangle
    // closes: Sun angle + phase + elongation = π
    let sun_angle = angle_between(earth, comet);
    assert!((sun_angle + e.phase_rad + e.elongation_rad - core::f64::consts::PI).abs() < 1e-6);
    // Sun → comet along the line of sight (comet beyond the Earth, opposition): phase 0, elongation
    // π, the sky projection vanishes and PsAng is finite
    let e = comet_ephemeris([1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0; 3]);
    assert!(e.phase_rad < 1e-9 && (e.elongation_rad - core::f64::consts::PI).abs() < 1e-9);
    assert!(e.sun_pa_rad.is_finite());
    // PsAng conventions on a non-polar line of sight: the Earth on +X, the comet at (1, 1, 0),
    // line of sight +Y ecliptic (RA 6h, Dec +ε). Sun→comet = (1, 1, 0) has the sky component +X,
    // which at RA 6h is the direction of decreasing RA, west: PsAng = 270°.
    let e = comet_ephemeris([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0; 3]);
    assert!((e.sun_pa_rad.to_degrees() - 270.0).abs() < 1e-6, "{}°", e.sun_pa_rad.to_degrees());
    assert!((e.ra_rad.to_degrees() - 90.0).abs() < 1e-6 && (e.dec_rad - OBLIQUITY_J2000_RAD).abs() < 1e-6);
    // the mirror case (comet at (1, −1, 0), line of sight −Y, RA 18h): sky component +X is east
    let e = comet_ephemeris([1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [0.0; 3]);
    assert!((e.sun_pa_rad.to_degrees() - 90.0).abs() < 1e-6, "{}°", e.sun_pa_rad.to_degrees());
  }

  /// The reviewer's 67P save (`test_assets/saves/review_67p_2021-12`) at 2021-12-27 00:00 UTC:
  /// r ≈ 1.388 AU, Δ ≈ 0.466 AU, phase ≈ 24.6° (55 days after the 2021-11-02 perihelion at 1.21 AU),
  /// every angle self-consistent (`check_consistent`), the light time ~233 s, and the
  /// light-time displacement of the astrometric place against the geometric one below 60″ (67P
  /// moves ≤ 50 km/s relative to the Earth · 233 s ≈ 11 650 km at 0.47 AU ≈ 34″ at most).
  #[test]
  fn review_67p_ephemeris_at_2021_12_27() {
    let mut almanac = crate::simulation::almanac::AlmanacPackedData::default();
    for f in [
      "../../assets/planets/pck00011.pca",
      "../../assets/planets/gm_de431.pca",
      "../../assets/planets/de442.bsp",
      "../../test_assets/saves/review_67p_2021-12/spk/spk_67P_90000703_2021-12-12_2021-12-28.bsp",
    ] {
      almanac.load_almanac(f).expect(f);
    }
    const NAIF_67P: i32 = 1000012;
    let epoch = hifitime::Epoch::from_gregorian_utc(2021, 12, 27, 0, 0, 0, 0);
    let e = comet_ephemeris_at(&almanac, NAIF_67P, epoch).expect("almanac covers 2021-12-27");
    // independent reference: a Kepler solve on the save's osculating elements (q 1.2106 AU,
    // e 0.6497, T_p JD 2459520.564) and a low-precision Sun (Meeus, precessed to J2000) for the
    // Earth gives r 1.3881 AU, Δ 0.4663 AU, phase 24.60°
    assert!((e.r_au - 1.3881).abs() < 0.003, "r {} AU", e.r_au);
    assert!((e.delta_au - 0.4663).abs() < 0.003, "Δ {} AU", e.delta_au);
    assert!((e.phase_rad.to_degrees() - 24.60).abs() < 0.2, "phase {}°", e.phase_rad.to_degrees());
    assert!((e.light_time_s - 232.7).abs() < 2.0, "light time {} s", e.light_time_s);
    // the same reference at the retarded epoch, on the equator of J2000: RA 9.0251 h, Dec +28.2698°
    assert!((e.ra_rad.to_degrees() / 15.0 - 9.0251).abs() < 0.005, "RA {} h", e.ra_rad.to_degrees() / 15.0);
    assert!((e.dec_rad.to_degrees() - 28.2698).abs() < 0.05, "Dec {}°", e.dec_rad.to_degrees());
    let frame = crate::simulation::almanac::SUN_ECLIPJ2000;
    let earth = almanac
      .get_cartesian_state(anise::constants::celestial_objects::EARTH, frame.orientation_id, frame.ephemeris_id, epoch, false)
      .unwrap();
    let earth = scale([earth.radius_km[0], earth.radius_km[1], earth.radius_km[2]], 1.0 / AU_KM);
    check_consistent(earth, &e, 1e-6);
    // the geometric place, for the record and the light-time bound
    let comet = almanac
      .get_cartesian_state(NAIF_67P, frame.orientation_id, frame.ephemeris_id, epoch, false)
      .unwrap();
    let comet = scale([comet.radius_km[0], comet.radius_km[1], comet.radius_km[2]], 1.0 / AU_KM);
    let (ra_g, dec_g) = ra_dec_of(ecliptic_to_equatorial_j2000(sub(comet, earth)));
    let shift = angle(dir_of_ra_dec(ra_g, dec_g), dir_of_ra_dec(e.ra_rad, e.dec_rad)) / ARCSEC_RAD;
    assert!(shift > 0.1 && shift < 60.0, "light-time shift {shift}″");
    println!(
      "67P 2021-12-27 00:00 UTC: RA {:.4} h Dec {:+.4}° r {:.5} AU Δ {:.5} AU phase {:.3}° elong {:.3}° PsAng {:.3}° light time {:.1} s",
      e.ra_rad.to_degrees() / 15.0, e.dec_rad.to_degrees(), e.r_au, e.delta_au,
      e.phase_rad.to_degrees(), e.elongation_rad.to_degrees(), e.sun_pa_rad.to_degrees(), e.light_time_s
    );
  }
}
