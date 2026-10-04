//! Heliocentric osculating orbital elements ⇄ Cartesian state (SUN_ECLIPJ2000, km, km/s).
//!
//! Used to re-osculate the comet reference orbit at the simulation start epoch from its SPK state
//! (the SBDB solution epoch can be years away: 67P's elements are osculating at 2015-10-10 and are
//! 0.12 AU off its 2025 path, while elements re-osculated at 2025-10-01 stay within ~600 km over a
//! month), and to propagate the reference to any epoch (two-body, universal variables).

use crate::simulation_api::{reposition::AU_TO_KM, structs::KeplerianElements};

/// Heliocentric gravitational parameter of the Sun (DE440), km³/s².
pub const MU_SUN_KM3_S2: f64 = 1.327_124_400_419_393_8e11;
/// Seconds per day.
pub const DAY_S: f64 = 86_400.0;

type V3 = [f64; 3];
fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
fn norm(a: V3) -> f64 {
  dot(a, a).sqrt()
}
fn scale(a: V3, s: f64) -> V3 {
  [a[0] * s, a[1] * s, a[2] * s]
}

/// Perifocal unit vectors `(P̂, Q̂)` (towards perihelion, 90° ahead in the orbit plane) for the
/// 3-1-3 rotation (Ω, i, ω) from the ecliptic.
pub fn perifocal_basis(el: &KeplerianElements) -> (V3, V3) {
  let (i, om, w) = (
    el.inclination_deg.to_radians(),
    el.longitude_of_ascending_node_deg.to_radians(),
    el.argument_of_perihelion_deg.to_radians(),
  );
  let p = [
    om.cos() * w.cos() - om.sin() * i.cos() * w.sin(),
    om.sin() * w.cos() + om.cos() * i.cos() * w.sin(),
    i.sin() * w.sin(),
  ];
  let q = [
    -om.cos() * w.sin() - om.sin() * i.cos() * w.cos(),
    -om.sin() * w.sin() + om.cos() * i.cos() * w.cos(),
    i.sin() * w.cos(),
  ];
  (p, q)
}

/// Osculating elements of the state `(r_km, v_kms)` at `epoch_jd_tdb`, any eccentricity. The
/// time of perihelion is the passage closest to `epoch_jd_tdb` (the previous or next one).
/// Equatorial (i ≈ 0) orbits get Ω = 0 and ω measured from +x.
pub fn elements_from_state(r: V3, v: V3, mu: f64, epoch_jd_tdb: f64) -> KeplerianElements {
  let rn = norm(r);
  let h = cross(r, v);
  let hn = norm(h);
  let ev = {
    let vxh = cross(v, h);
    [
      vxh[0] / mu - r[0] / rn,
      vxh[1] / mu - r[1] / rn,
      vxh[2] / mu - r[2] / rn,
    ]
  };
  let e = norm(ev);
  let p = hn * hn / mu; // semi-latus rectum
  let q_km = p / (1.0 + e);

  let i = (h[2] / hn).clamp(-1.0, 1.0).acos();
  let node = [-h[1], h[0], 0.0]; // ẑ × h
  let nn = norm(node);
  let (om, w) = if nn > 1e-12 * hn {
    let om = node[1].atan2(node[0]);
    let mut w = (dot(node, ev) / (nn * e)).clamp(-1.0, 1.0).acos();
    if ev[2] < 0.0 {
      w = 2.0 * core::f64::consts::PI - w;
    }
    (om, w)
  } else {
    // equatorial: longitude of perihelion from +x, retrograde flips the sense
    let lp = ev[1].atan2(ev[0]);
    (0.0, if h[2] >= 0.0 { lp } else { -lp })
  };

  // true anomaly → mean anomaly → time since perihelion. atan2 form: acos(cos ν) is
  // ill-conditioned at perihelion (ν error ~√ε, i.e. ~0.1 s of time of perihelion)
  let nu = dot(cross(ev, r), h).atan2(dot(ev, r) * hn);
  let dt_from_peri_s = if e < 1.0 {
    let a = q_km / (1.0 - e);
    let ea = 2.0 * (((1.0 - e) / (1.0 + e)).sqrt() * (nu * 0.5).tan()).atan();
    let m = ea - e * ea.sin();
    m / (mu / (a * a * a)).sqrt()
  } else {
    let a = q_km / (1.0 - e); // negative
    let f = 2.0 * (((e - 1.0) / (e + 1.0)).sqrt() * (nu * 0.5).tan()).atanh();
    let m = e * f.sinh() - f;
    m / (mu / (-a * -a * -a)).sqrt()
  };

  KeplerianElements {
    eccentricity: e,
    perihelion_distance_au: q_km / AU_TO_KM,
    inclination_deg: i.to_degrees(),
    longitude_of_ascending_node_deg: om.to_degrees().rem_euclid(360.0),
    argument_of_perihelion_deg: w.to_degrees().rem_euclid(360.0),
    time_of_perihelion_jd_tdb: epoch_jd_tdb - dt_from_peri_s / DAY_S,
  }
}

/// Two-body state `(r_km, v_kms)` of the reference orbit at `epoch_jd_tdb`. `None` when the
/// elements carry no time of perihelion.
pub fn state_from_elements(el: &KeplerianElements, mu: f64, epoch_jd_tdb: f64) -> Option<(V3, V3)> {
  if !el.time_of_perihelion_jd_tdb.is_finite() {
    return None;
  }
  let (p_hat, q_hat) = perifocal_basis(el);
  let q_km = el.perihelion_distance_au * AU_TO_KM;
  let r0 = scale(p_hat, q_km);
  let v0 = scale(q_hat, (mu * (1.0 + el.eccentricity) / q_km).sqrt());
  let dt_s = (epoch_jd_tdb - el.time_of_perihelion_jd_tdb) * DAY_S;
  Some(crate::scene::dust::kepler::propagate_f64(r0, v0, mu, dt_s))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn close(a: V3, b: V3) -> f64 {
    norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
  }

  /// elements → state → elements round trip for elliptic and hyperbolic orbits.
  #[test]
  fn round_trip_any_eccentricity() {
    for (e, q, i, om, w) in [
      (0.6409, 1.2433, 7.04, 50.14, 12.80),
      (0.0167, 0.9833, 0.5, 120.0, 250.0),
      (0.95, 0.3, 150.0, 300.0, 80.0),
      (1.2, 1.5, 40.0, 70.0, 10.0),
      (3.5, 2.0, 95.0, 200.0, 330.0),
    ] {
      let el = KeplerianElements {
        eccentricity: e,
        perihelion_distance_au: q,
        inclination_deg: i,
        longitude_of_ascending_node_deg: om,
        argument_of_perihelion_deg: w,
        time_of_perihelion_jd_tdb: 2_460_000.5,
      };
      for days in [-40.0, -3.0, 0.0, 17.0, 90.0] {
        let t = el.time_of_perihelion_jd_tdb + days;
        let (r, v) = state_from_elements(&el, MU_SUN_KM3_S2, t).unwrap();
        let back = elements_from_state(r, v, MU_SUN_KM3_S2, t);
        assert!(
          (back.eccentricity - e).abs() < 1e-9,
          "e {e} -> {}",
          back.eccentricity
        );
        assert!((back.perihelion_distance_au - q).abs() < 1e-9 * q.max(1.0));
        assert!((back.inclination_deg - i).abs() < 1e-7);
        assert!((back.longitude_of_ascending_node_deg - om).abs() < 1e-6);
        assert!(
          (back.argument_of_perihelion_deg - w).abs() < 1e-6,
          "w {w} -> {}",
          back.argument_of_perihelion_deg
        );
        assert!(
          (back.time_of_perihelion_jd_tdb - el.time_of_perihelion_jd_tdb).abs() < 1e-7,
          "e={e} days={days} dtp={} d",
          back.time_of_perihelion_jd_tdb - el.time_of_perihelion_jd_tdb
        );
        // and the state itself is reproduced
        let (r2, v2) = state_from_elements(&back, MU_SUN_KM3_S2, t).unwrap();
        assert!(close(r, r2) < 1e-3, "position off by {} km", close(r, r2));
        assert!(close(v, v2) < 1e-9);
      }
    }
  }

  /// At tp the reference sits at perihelion q·P̂, and one period later it is back there.
  #[test]
  fn perihelion_and_period() {
    let el = KeplerianElements {
      eccentricity: 0.6409,
      perihelion_distance_au: 1.2433,
      inclination_deg: 7.04,
      longitude_of_ascending_node_deg: 50.14,
      argument_of_perihelion_deg: 12.80,
      time_of_perihelion_jd_tdb: 2_457_247.588_657_812,
    };
    let (p_hat, _) = perifocal_basis(&el);
    let q_km = el.perihelion_distance_au * AU_TO_KM;
    let (r, _) = state_from_elements(&el, MU_SUN_KM3_S2, el.time_of_perihelion_jd_tdb).unwrap();
    assert!(close(r, scale(p_hat, q_km)) < 1e-3);
    let a = q_km / (1.0 - el.eccentricity);
    let period_d = 2.0 * core::f64::consts::PI * (a * a * a / MU_SUN_KM3_S2).sqrt() / DAY_S;
    let (r2, _) =
      state_from_elements(&el, MU_SUN_KM3_S2, el.time_of_perihelion_jd_tdb + period_d).unwrap();
    assert!(
      close(r2, scale(p_hat, q_km)) < 1.0,
      "after one period: {} km",
      close(r2, scale(p_hat, q_km))
    );
  }

  #[test]
  fn no_time_of_perihelion_no_state() {
    let el = KeplerianElements {
      eccentricity: 0.5,
      perihelion_distance_au: 1.0,
      inclination_deg: 0.0,
      longitude_of_ascending_node_deg: 0.0,
      argument_of_perihelion_deg: 0.0,
      time_of_perihelion_jd_tdb: f64::NAN,
    };
    assert!(state_from_elements(&el, MU_SUN_KM3_S2, 2_460_000.0).is_none());
  }
}
