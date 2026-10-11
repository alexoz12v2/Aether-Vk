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
}
