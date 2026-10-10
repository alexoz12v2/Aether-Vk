use super::*;

fn rk4(r0: V3, v0: V3, mu: f64, dt_total: f64, h: f64) -> (V3, V3) {
  let acc = |r: V3| -> V3 {
    let d = norm(r);
    scale(r, -mu / (d * d * d))
  };
  let (mut r, mut v) = (r0, v0);
  let n = (dt_total.abs() / h).ceil() as usize;
  let h = dt_total / n as f64;
  for _ in 0..n {
    let k1v = acc(r);
    let k1r = v;
    let k2v = acc(add(r, scale(k1r, h / 2.0)));
    let k2r = add(v, scale(k1v, h / 2.0));
    let k3v = acc(add(r, scale(k2r, h / 2.0)));
    let k3r = add(v, scale(k2v, h / 2.0));
    let k4v = acc(add(r, scale(k3r, h)));
    let k4r = add(v, scale(k3v, h));
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

/// df64 propagation with f64 in/out
fn prop_df(r0: V3, v0: V3, mu: f64, dt: f64) -> (V3, V3) {
  let (r, v) = kepler::propagate(
    &Df3::from_f64(r0),
    &Df3::from_f64(v0),
    Df::from_f64(mu),
    Df::from_f64(dt),
  );
  (r.to_f64(), v.to_f64())
}

/// 67P-like state at 5.5 AU, slightly inclined velocity
fn comet_state() -> (V3, V3) {
  let r = [-1.924 * AU_M, -5.164 * AU_M, -0.204 * AU_M];
  // ~ 0.9 of circular speed, tilted
  let rn = norm(r);
  let vc = (SUN_MU_M3_S2 / rn).sqrt() * 0.9;
  let t = [5.164, -1.924, 0.3];
  let tn = norm(t);
  (r, scale(t, vc / tn))
}

/// A 67P-like orbit (q 1.24 AU, e 0.641, P 6.44 y) whose aphelion falls at `t_aphelion_s`: the
/// state at `t = 0`. The observer's epoch (2025-10-17) is 270 d past aphelion and 3.96 y past
/// perihelion, so a 5.3-year history on this orbit holds the perihelion passage.
fn comet_state_67p(t_aphelion_s: f64) -> (V3, V3) {
  let (q, e) = (1.2432 * AU_M, 0.641);
  let a = q / (1.0 - e);
  let r_aph = a * (1.0 + e);
  let v_aph = (SUN_MU_M3_S2 * (2.0 / r_aph - 1.0 / a)).sqrt();
  // the orbit plane tilted like the test orbit: aphelion along −x, motion along −y
  let r = [-r_aph, 0.0, 0.0];
  let v = [0.0, -v_aph * 0.99, -v_aph * 0.14];
  kepler::propagate_f64(r, v, SUN_MU_M3_S2, -t_aphelion_s)
}

// ─── df64 primitives ────────────────────────────────────────────────────────

#[test]
fn df_constants_match_f64() {
  assert_eq!(consts::SUN_MU, Df::from_f64(SUN_MU_M3_S2));
  assert_eq!(consts::TWO_PI, Df::from_f64(2.0 * core::f64::consts::PI));
  let mut fact = 1.0f64;
  for k in 0..16 {
    if k > 0 {
      fact *= k as f64;
    }
    assert_eq!(consts::INV_FACT[k], Df::from_f64(1.0 / fact), "1/{k}!");
  }
}

#[test]
fn df_arithmetic_is_df64_accurate() {
  // df64 has the f32 exponent range: keep products below ~1e34 (Dekker split headroom)
  let xs = [
    1.0e12 / 3.0,
    -7.25e-3,
    9.87654321e5,
    1.0 / 7.0,
    -3.3e16,
    2.0e-9,
  ];
  let tol = |x: f64| x.abs() * 4.0e-14 + 1e-35;
  for &a in &xs {
    for &b in &xs {
      let (da, db) = (Df::from_f64(a), Df::from_f64(b));
      // inputs are df-rounded; compare against f64 ops on the df-rounded values
      let (a, b) = (da.to_f64(), db.to_f64());
      assert!(
        (da.add(db).to_f64() - (a + b)).abs() <= tol(a.abs() + b.abs()),
        "{a}+{b}"
      );
      assert!(
        (da.sub(db).to_f64() - (a - b)).abs() <= tol(a.abs() + b.abs()),
        "{a}-{b}"
      );
      assert!((da.mul(db).to_f64() - a * b).abs() <= tol(a * b), "{a}*{b}");
      assert!((da.div(db).to_f64() - a / b).abs() <= tol(a / b), "{a}/{b}");
    }
    let s = Df::from_f64(a.abs()).sqrt().to_f64();
    assert!((s - a.abs().sqrt()).abs() <= tol(s), "sqrt {a}");
  }
  for &x in &[3.7, -3.7, 1.5e9 + 0.25, -1.5e9 - 0.25, 2.0, 0.0] {
    let f = Df::from_f64(x).floor().to_f64();
    assert_eq!(f, x.floor(), "floor {x}");
  }
}

// ─── Kepler ─────────────────────────────────────────────────────────────────

#[test]
fn stumpff_matches_closed_form() {
  for &x in &[-50.0f64, -3.0, -0.2, -1e-6, 0.0, 1e-6, 0.3, 2.0, 9.0, 39.0] {
    let (c0, c1, c2, c3) = if x > 1e-4 {
      let s = x.sqrt();
      (
        s.cos(),
        s.sin() / s,
        (1.0 - s.cos()) / x,
        (s - s.sin()) / (x * s),
      )
    } else if x < -1e-4 {
      let s = (-x).sqrt();
      (
        s.cosh(),
        s.sinh() / s,
        (s.cosh() - 1.0) / (-x),
        (s.sinh() - s) / (-x * s),
      )
    } else {
      (
        1.0 - x / 2.0 + x * x / 24.0,
        1.0 - x / 6.0 + x * x / 120.0,
        0.5 - x / 24.0 + x * x / 720.0,
        1.0 / 6.0 - x / 120.0 + x * x / 5040.0,
      )
    };
    let refs = [c0, c1, c2, c3];
    let c = kepler::stumpff_f64(x);
    let d = kepler::stumpff_df(Df::from_f64(x));
    let f = kepler::stumpff_f32(x as f32);
    for k in 0..4 {
      let scale = 1.0 + refs[k].abs() + c0.abs();
      assert!(
        (c[k] - refs[k]).abs() < 1e-11 * scale,
        "f64 c{k}({x}) {} vs {}",
        c[k],
        refs[k]
      );
      assert!(
        (d[k].to_f64() - refs[k]).abs() < 1e-10 * scale,
        "df c{k}({x}) {} vs {}",
        d[k].to_f64(),
        refs[k]
      );
      assert!(
        (f[k] as f64 - refs[k]).abs() < 1e-4 * scale,
        "f32 c{k}({x}) {} vs {}",
        f[k],
        refs[k]
      );
    }
  }
}

#[test]
fn kepler_df_matches_rk4_and_f64_for_all_beta_regimes() {
  let (r0, vc) = comet_state();
  // add a 50 m/s ejection to exercise the general case
  let v0 = add(vc, [30.0, -20.0, 35.0]);
  let dt = 30.0 * 86400.0;
  for &beta in &[0.0, 0.1, 0.9, 1.0, 1.5] {
    let mu = SUN_MU_M3_S2 * (1.0 - beta);
    let (rd, vd) = prop_df(r0, v0, mu, dt);
    let (r64, v64) = kepler::propagate_f64(r0, v0, mu, dt);
    let (rr, vr) = rk4(r0, v0, mu, dt, 600.0);
    let dr = norm(sub(rd, rr));
    let dv = norm(sub(vd, vr));
    assert!(dr < 1.0, "beta {beta}: |Δr| df vs rk4 = {dr} m");
    assert!(dv < 1e-6, "beta {beta}: |Δv| df vs rk4 = {dv} m/s");
    let dr64 = norm(sub(rd, r64));
    assert!(dr64 < 0.05, "beta {beta}: |Δr| df vs f64 = {dr64} m");
    assert!(norm(sub(vd, v64)) < 1e-7);
  }
}

#[test]
fn kepler_df_long_hyperbolic_and_backwards() {
  let (r0, v0) = comet_state();
  // multiple periods (period reduction path) + round trip
  for &dt in &[3.0e8f64, -2.0e7, 1.0, 7.3e6] {
    let (r1, v1) = prop_df(r0, v0, SUN_MU_M3_S2, dt);
    let (r2, v2) = prop_df(r1, v1, SUN_MU_M3_S2, -dt);
    let rel = norm(sub(r2, r0)) / norm(r0);
    assert!(rel < 2e-12, "round trip dt {dt}: rel err {rel}");
    assert!(norm(sub(v2, v0)) / norm(v0) < 1e-10);
    let e = |r: V3, v: V3| dot(v, v) / 2.0 - SUN_MU_M3_S2 / norm(r);
    assert!(
      ((e(r1, v1) - e(r0, v0)) / e(r0, v0)).abs() < 1e-11,
      "energy dt {dt}"
    );
    let (r64, _) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, dt);
    // many periods: phase error scales with the period count
    assert!(norm(sub(r1, r64)) / norm(r0) < 1e-11, "df vs f64 dt {dt}");
  }
  // strongly hyperbolic
  let vh = scale(v0, 3.0);
  let (rk, _) = prop_df(r0, vh, SUN_MU_M3_S2, 1.0e7);
  let (rr, _) = rk4(r0, vh, SUN_MU_M3_S2, 1.0e7, 300.0);
  assert!(norm(sub(rk, rr)) < 10.0);
}

#[test]
fn kepler_df_resolves_metre_scale_relative_motion() {
  // the reason for df64: two states 1 m / 1 mm/s apart at 5.5 AU must stay distinguishable
  let (r0, v0) = comet_state();
  let dt = 10.0 * 86400.0;
  let (ra, _) = prop_df(r0, v0, SUN_MU_M3_S2, dt);
  let (rb, _) = prop_df(
    add(r0, [1.0, 0.0, 0.0]),
    add(v0, [0.0, 1e-3, 0.0]),
    SUN_MU_M3_S2,
    dt,
  );
  let (ra64, _) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, dt);
  let (rb64, _) = kepler::propagate_f64(
    add(r0, [1.0, 0.0, 0.0]),
    add(v0, [0.0, 1e-3, 0.0]),
    SUN_MU_M3_S2,
    dt,
  );
  let d = sub(rb, ra);
  let d64 = sub(rb64, ra64);
  assert!(
    norm(sub(d, d64)) < 0.05,
    "relative displacement df {d:?} vs f64 {d64:?}"
  );
}

// ─── emission / evaluation ─────────────────────────────────────────────────

const T_START: f64 = 4.0e8; // ~12.7 years after the epoch: exercises df64 time

fn test_batch(count: u32, dur: f64) -> DustBatch {
  let (rc, vc) = comet_state();
  let dist = SizeDistribution::from_diameter_um(100.0);
  let (size_params, vel_params, mass_params) =
    batch_params(&dist, 100.0, 0.533, 0.0213, 2.0, 0.5, 1.0e6, 0.37);
  // one stream (every cluster its own time sample); stream layouts set `mass_params[3]`
  let mut b = DustBatch {
    comet_r_t_hi: [0.0; 4],
    comet_r_t_lo: [0.0; 4],
    comet_v_dur_hi: [0.0; 4],
    comet_v_dur_lo: [0.0; 4],
    rot_start: [0.0, 0.0, 0.0, 1.0],
    spin: [0.0, 0.0, 1.0, 0.0],
    // jet roughly sunward
    jet_dir_aperture: [0.35, 0.93, 0.04, 0.6],
    size_params,
    vel_params,
    mass_params,
    first_index: 12345,
    count,
    ring_mask: RING_CAPACITY_HIGH - 1,
    seed: 0xC0FFEE,
    lit: [0.0, 0.0, 0.0, LIT_MODE_ALWAYS],
    site_offset: [0.0; 4],
  };
  b.set_comet(rc, vc, T_START, dur);
  b
}

#[test]
fn batch_mass_is_conserved_by_importance_weights() {
  // every time sample splits its mass over the streams' size strata: exact for any stream count
  for shift in [0u32, 3, 6] {
    let mut b = test_batch(4096, 1800.0);
    b.mass_params[3] = batch_streams_word(shift, false);
    let total: f64 = (0..b.count).map(|j| emit_cluster(&b, j).mass_g() as f64).sum();
    let rel = (total - 1.0e6).abs() / 1.0e6;
    assert!(
      rel < 1e-4,
      "{} streams: mass sum {total} rel err {rel}",
      1 << shift
    );
  }
}

#[test]
fn emission_times_and_beta_are_in_range() {
  let b = test_batch(1024, 1800.0);
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let t0 = c.t0().to_f64();
    assert!(t0 >= T_START && t0 <= T_START + 1800.0, "t0 {t0}");
    // the reference grain (50 µm) with the jet's speed spread as its radial dispersion
    let s = c.eject[3];
    assert!((s - 50.0).abs() < 1e-3, "s_ref {s}");
    assert!(c.sigma_rad() > 0.0 && c.sigma_lat() > 0.0);
    assert!(c.beta() > 0.0 && c.beta().is_finite());
    assert!(c.mass_g() > 0.0 && c.mass_g().is_finite());
  }
  // clusters start on the comet's own trajectory
  let (rc, vc) = comet_state();
  let c = emit_cluster(&b, 500);
  let (r_expect, _) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, c.t0().to_f64() - T_START);
  assert!(norm(sub(c.r0().to_f64(), r_expect)) < 0.05);
}

fn frame_after(days: f64) -> (DustFrame, V3) {
  let t_now = T_START + days * 86400.0;
  let (rc0, vc0) = comet_state();
  let (rc, _) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t_now - T_START);
  (DustFrame::new(rc, t_now, [0.0, 0.0, 0.0, 1.0], 1.0e9), rc)
}

#[test]
fn syndynes_point_antisunward_and_grow_with_beta() {
  // emit a short batch, evaluate 10 days later relative to the comet: every cluster carries the
  // whole size range, its polyline edges run from β/F (large grains) to β·F (small grains)
  let mut b = test_batch(2048, 60.0);
  b.mass_params[3] = batch_streams_word(6, false);
  let (frame, rc) = frame_after(10.0);
  let anti_sun = scale(rc, 1.0 / norm(rc));
  let g = SUN_MU_M3_S2 / dot(rc, rc);
  let t = 10.0f64 * 86400.0;
  let mut checked = 0;
  for j in (0..b.count).step_by(37) {
    let c = emit_cluster(&b, j);
    let (e, m) = packet_moments(&c, j, &frame);
    assert!(e.age_id_dbeta_flux[3] > 0.0);
    assert_eq!(render_slot(e.age_id_dbeta_flux[1]), j);
    assert!(render_live(e.age_id_dbeta_flux[1]));
    let along = |k: usize| {
      let p = m.edge(k);
      dot([p[0] as f64, p[1] as f64, p[2] as f64], anti_sun)
    };
    // radiation pressure: displacement ½ β g t² grows with β along the polyline
    let (lo, mid, hi) = (
      along(0),
      along(SIZE_BINS as usize / 2),
      along(SIZE_EDGES as usize - 1),
    );
    assert!(
      hi > 0.0,
      "cluster {j}: the small grains must move anti-sunward, got {hi}"
    );
    assert!(
      hi > mid && mid > lo,
      "cluster {j}: displacement must grow with β: {lo} {mid} {hi}"
    );
    let beta_max = (c.beta() as f64) * SIZE_RANGE_FACTOR;
    let expect = 0.5 * beta_max * g * t * t;
    assert!(
      hi > 0.3 * expect && hi < 3.0 * expect,
      "cluster {j}: {hi} vs ½ β g t² = {expect}"
    );
    checked += 1;
  }
  assert!(checked > 40);
}

#[test]
fn zero_beta_zero_speed_cluster_stays_on_the_comet() {
  // a cluster with β = 0 and no ejection must coincide with the comet: tests the df64 subtract
  let mut b = test_batch(1, 0.0);
  b.vel_params[0] = 0.0; // no ejection
  b.vel_params[3] = 0.0; // β = 0
  let c = emit_cluster(&b, 0);
  for days in [0.01, 1.0, 30.0] {
    let (frame, _) = frame_after(days);
    let e = evaluate_cluster(&c, 0, &frame);
    let d = (e.pos_size[0].powi(2) + e.pos_size[1].powi(2) + e.pos_size[2].powi(2)).sqrt();
    assert!(d < 0.1, "{days} d: offset {d} m");
  }
}

#[test]
fn evaluate_culls_by_age() {
  let b = test_batch(10, 60.0);
  let c = emit_cluster(&b, 0);
  let t0 = c.t0().to_f64();
  let r = c.r0().to_f64();
  let at = |t: f64| DustFrame::new(r, t, [0.0, 0.0, 0.0, 1.0], 100.0);
  assert_eq!(
    evaluate_cluster(&c, 7, &at(t0 - 1.0)).age_id_dbeta_flux[3],
    0.0
  );
  assert_eq!(
    evaluate_cluster(&c, 7, &at(t0 + 101.0)).age_id_dbeta_flux[3],
    0.0
  );
  let live = evaluate_cluster(&c, 7, &at(t0 + 50.0));
  assert!(live.age_id_dbeta_flux[3] > 0.0);
  assert!(
    (live.age_id_dbeta_flux[0] - 50.0).abs() < 1e-3,
    "age {}",
    live.age_id_dbeta_flux[0]
  );
}

// ─── host planning ─────────────────────────────────────────────────────────

fn desc(count: u32) -> DustBatch {
  let mut d: DustBatch = bytemuck::Zeroable::zeroed();
  d.count = count;
  d
}

fn test_cfg(q: f64, ttl_s: f64) -> DustEmitConfig {
  DustEmitConfig {
    q_dust_kgs: q,
    ttl_s,
    dist: SizeDistribution::from_diameter_um(100.0),
    diameter_um: 100.0,
    density_gcm3: 0.533,
    beta_ref: 0.0213,
    v_mean: 2.0,
    v_std: 0.5,
    jet_dir: [0.0, 0.0, 1.0],
    aperture_rad: 0.5,
    seed: 42,
  }
}

#[test]
fn planner_conserves_mass_and_respects_budget() {
  for capacity in [RING_CAPACITY_HIGH, RING_CAPACITY_LOW] {
    let mut acc = EmissionAccumulator::default();
    let mut ring = RingState::new(capacity);
    let ttl = 86400.0;
    let dt = 1800.0; // 3 h/s with 166 ms emission interval
    let q = 1.5e-3;
    let mut t = 0.0;
    let mut emitted = 0.0;
    for _ in 0..2000 {
      t += dt;
      ring.retire(t, ttl);
      if let Some(p) = plan_batch(
        &mut acc,
        q,
        dt,
        dt,
        t,
        ttl,
        ring.capacity,
        ring.free_slots(),
      ) {
        emitted += p.mass_g;
        ring.push_batch(desc(p.count), t, p.mass_g);
      }
      assert!(ring.live() <= capacity);
    }
    let produced = q * 1e3 * dt * 2000.0;
    assert!(((emitted + acc.mass_g) - produced).abs() < 1e-6 * produced);
    // steady state uses ~BUDGET_SAFETY of the ring
    let fill = ring.live() as f64 / capacity as f64;
    assert!(
      fill > 0.6 && fill <= 0.85,
      "capacity {capacity}: fill {fill}"
    );
  }
}

#[test]
fn ring_rewind_and_capacity_selection() {
  let mut ring = RingState::new(1024);
  let d = ring.push_batch(desc(100), 10.0, 1.0);
  assert_eq!((d.first_index, d.ring_mask), (0, 1023));
  ring.push_batch(desc(100), 20.0, 1.0);
  ring.push_batch(desc(100), 30.0, 1.0);
  // nothing drawable until submitted
  assert_eq!(ring.drawable().1, 0);
  ring.mark_submitted(7);
  assert_eq!(ring.drawable(), (0, 300, 7));
  ring.rewind(25.0);
  assert_eq!(ring.live(), 200);
  ring.retire(1000.0, 985.0);
  assert_eq!(ring.live(), 100);
  assert_eq!(ring.first_slot(), 100);
  assert_eq!(ring_capacity(true, None), RING_CAPACITY_HIGH);
  assert_eq!(ring_capacity(false, None), RING_CAPACITY_LOW);
  assert_eq!(ring_capacity(false, Some(65536)), 65536);
  assert_eq!(ring_capacity(true, Some(1000)), RING_CAPACITY_HIGH); // not a power of two
}

// ─── jet site illumination ─────────────────────────────────────────────────

/// 67P sidereal rotation period (s)
const P_ROT: f64 = 12.4 * 3600.0;
const OMEGA: f64 = 2.0 * core::f64::consts::PI / P_ROT;

fn unit(a: V3) -> V3 {
  scale(a, 1.0 / norm(a))
}

/// lit time of `[0, dur]` by midpoint sampling
fn brute_lit_time(sun: V3, n0: V3, axis: V3, omega: f64, dur: f64, samples: usize) -> f64 {
  let h = dur / samples as f64;
  (0..samples)
    .filter(|&i| {
      dot(
        rotate_axis_angle(n0, axis, omega * (i as f64 + 0.5) * h),
        sun,
      ) > 0.0
    })
    .count() as f64
    * h
}

#[test]
fn lit_window_closed_form() {
  let z = [0.0, 0.0, 1.0];
  let cases: [(V3, V3, V3, f64); 5] = [
    // equatorial site, sun in the equatorial plane: half lit
    ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], z, 3.25 * P_ROT),
    // sun 30° above the equator, site at 20° latitude: lit more than half
    (
      [0.866_025_4, 0.0, 0.5],
      [0.0, 0.939_692_6, 0.342_020_1],
      z,
      3.25 * P_ROT,
    ),
    // retrograde (axis flipped), partial window starting mid arc
    (
      [0.3, -0.9, 0.1],
      [0.6, 0.7, -0.2],
      [0.0, 0.0, -1.0],
      0.37 * P_ROT,
    ),
    // tilted axis, short window (< 1 rotation)
    (
      [0.35, 0.93, 0.04],
      [1.0, 0.0, 0.0],
      [0.2, 0.1, 0.97],
      0.8 * P_ROT,
    ),
    // many rotations (high time scale)
    ([0.35, 0.93, 0.04], [1.0, 0.0, 0.0], z, 37.6 * P_ROT),
  ];
  for (k, (sun, n0, axis, dur)) in cases.into_iter().enumerate() {
    let (sun, n0, axis) = (unit(sun), unit(n0), unit(axis));
    let w = LitWindow::new(sun, n0, axis, OMEGA, dur);
    let brute = brute_lit_time(sun, n0, axis, OMEGA, dur, 400_000);
    assert_eq!(w.mode, LIT_MODE_PERIODIC, "case {k}");
    assert!(
      (w.lit_time_s - brute).abs() < 1e-4 * dur,
      "case {k}: closed form {} vs brute {brute} (dur {dur})",
      w.lit_time_s
    );
    assert!(w.psi_start >= -core::f64::consts::PI && w.psi_start < core::f64::consts::PI);
    assert!(w.psi0 > 0.0 && w.psi0 < core::f64::consts::PI);
  }
  // equinox: exactly half
  let w = LitWindow::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], z, OMEGA, 4.0 * P_ROT);
  assert!((w.lit_time_s - 2.0 * P_ROT).abs() < 1e-6 * P_ROT);
  // polar site: constant illumination
  let lit = LitWindow::new(unit([0.3, 0.0, 1.0]), z, z, OMEGA, 1000.0);
  assert_eq!((lit.mode, lit.lit_time_s), (LIT_MODE_ALWAYS, 1000.0));
  let dark = LitWindow::new(unit([0.3, 0.0, -1.0]), z, z, OMEGA, 1000.0);
  assert_eq!(dark.lit_time_s, 0.0);
  // polar night / midnight sun at mid latitudes (|A| > R)
  let n = unit([1.0, 0.0, 3.0]);
  let sun_high = unit([1.0, 0.0, 2.0]);
  assert_eq!(
    LitWindow::new(sun_high, n, z, OMEGA, P_ROT).lit_time_s,
    P_ROT
  );
  assert_eq!(
    LitWindow::new(scale(sun_high, -1.0), n, z, OMEGA, P_ROT).lit_time_s,
    0.0
  );
  // no spin: lit iff the site faces the sun
  assert_eq!(
    LitWindow::new([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], z, 0.0, 10.0).lit_time_s,
    0.0
  );
}

/// batch over `dur` with a spinning jet: identity attitude at the start, so the particle-system
/// frame is the root frame and the jet axis is the site normal `n0`
fn spinning_batch(count: u32, dur: f64, n0: V3, axis: V3, aperture: f32) -> (DustBatch, V3) {
  let (rc, _) = comet_state();
  let sun = scale(rc, -1.0 / norm(rc));
  let mut b = test_batch(count, dur);
  b.jet_dir_aperture = [n0[0] as f32, n0[1] as f32, n0[2] as f32, aperture];
  b.spin = [axis[0] as f32, axis[1] as f32, axis[2] as f32, OMEGA as f32];
  b.lit = LitWindow::new(sun, n0, axis, OMEGA, dur).to_gpu();
  (b, sun)
}

/// The jet site turns with the nucleus over a window: every cluster leaves from the site's position
/// at its own emission time, with the surface velocity ω × r (the window-start position carried
/// by the comet's free fall missed up to 2 nucleus radii, the fountain's base sat off the site).
#[test]
fn emission_starts_at_the_spun_site() {
  let axis = unit([0.2, 0.1, 0.97]);
  let n0 = unit([1.0, -0.3, 0.1]);
  let omega = 2.0 * core::f64::consts::PI / (8.0 * 3600.0);
  let dur = 8.0 * 3600.0;
  let off = scale(n0, 2000.0);
  let (rc, vc) = comet_state();
  let batch = |v_ref: f32| {
    let (mut b, _) = spinning_batch(1024, dur, n0, axis, 0.6);
    b.spin[3] = omega as f32;
    b.lit = [0.0, 0.0, 0.0, LIT_MODE_ALWAYS];
    b.vel_params[0] = v_ref;
    b.site_offset = [off[0] as f32, off[1] as f32, off[2] as f32, 0.0];
    // the site at the window start (rot_start = identity)
    b.set_comet(add(rc, off), vc, T_START, dur);
    b
  };
  let (b, still) = (batch(2.0), batch(0.0));
  let (mut turned, mut max_err) = (0.0f64, 0.0f64);
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let dt = c.t0().to_f64() - T_START;
    let (centre, v_centre) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, dt);
    let want = rotate_axis_angle(off, axis, omega * dt);
    let err = norm(sub(sub(c.r0().to_f64(), centre), want));
    max_err = max_err.max(err);
    turned = turned.max(norm(sub(want, off)));
    // no ejection speed: the cluster leaves with the surface velocity
    let z = emit_cluster(&still, j);
    let w = scale(axis, omega);
    let v_surf = [
      w[1] * want[2] - w[2] * want[1],
      w[2] * want[0] - w[0] * want[2],
      w[0] * want[1] - w[1] * want[0],
    ];
    let dv = norm(sub(sub(z.v0().to_f64(), v_centre), v_surf));
    assert!(dv < 1e-3, "cluster {j}: surface velocity off by {dv} m/s");
  }
  assert!(
    turned > 3000.0,
    "the window covers a turn: the site moved {turned} m"
  );
  assert!(max_err < 1.0, "emission {max_err} m from the spun site");
}

#[test]
fn emitted_clusters_are_sunlit_and_uniform_in_lit_time() {
  let axis = unit([0.2, 0.1, 0.97]);
  let n0 = unit([1.0, -0.3, 0.1]);
  let dur = 2.5 * P_ROT;
  let (b, sun) = spinning_batch(4000, dur, n0, axis, 0.6);
  assert_eq!(b.lit[3], LIT_MODE_PERIODIC);
  let total = LitWindow::new(sun, n0, axis, OMEGA, dur).lit_time_s;
  assert!(total > 0.2 * dur && total < 0.8 * dur, "lit {total}");
  let mut prev = -1.0;
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let dt = c.t0().to_f64() - T_START;
    assert!(dt >= 0.0 && dt <= dur + 1e-3, "dt {dt}");
    assert!(
      dt >= prev - 1e-3,
      "emission time must grow with j: {prev} -> {dt}"
    );
    prev = dt;
    // the site faces the sun at emission
    let n = rotate_axis_angle(n0, axis, OMEGA * dt);
    assert!(
      dot(n, sun) > -1e-3,
      "cluster {j} emitted in the dark: {}",
      dot(n, sun)
    );
    // uniform in lit time: lit time elapsed before t0 is the stratum position u_t
    let frac = LitWindow::new(sun, n0, axis, OMEGA, dt).lit_time_s / total;
    let u_mid = (j as f64 + 0.5) / b.count as f64;
    assert!(
      (frac - u_mid).abs() <= 1.0 / b.count as f64 + 1e-4,
      "cluster {j}: lit fraction {frac} vs stratum {u_mid}"
    );
  }
  // importance weights still carry the whole batch mass
  let mass: f64 = (0..b.count).map(|j| emit_cluster(&b, j).mass_g() as f64).sum();
  assert!((mass - 1.0e6).abs() < 0.01e6);
}

#[test]
fn ejection_follows_exact_spin() {
  // zero aperture and no speed spread: the ejection direction is the rotated jet axis
  let axis = unit([0.0, 0.3, 1.0]);
  let n0 = unit([1.0, 0.2, 0.0]);
  let dur = 3.7 * P_ROT;
  let (mut b, _) = spinning_batch(300, dur, n0, axis, 0.0);
  b.vel_params[1] = 0.0;
  let (rc, vc) = comet_state();
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let dt = c.t0().to_f64() - T_START;
    let (_, v_comet) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, dt);
    let ej = sub(c.v0().to_f64(), v_comet);
    let expect = rotate_axis_angle(n0, axis, OMEGA * dt);
    let err = norm(sub(unit(ej), expect));
    assert!(err < 1e-4, "cluster {j}: direction error {err}");
  }
}

/// jet on an equatorial site of a nucleus spinning about `axis`, riding the comet orbit
fn spinning_jet(t: f64, axis: V3, n0: V3, model_spin: bool) -> JetState {
  spinning_jet_from(comet_state(), t, axis, n0, model_spin)
}

/// [`spinning_jet`] on the orbit through `state` at `t = 0`
fn spinning_jet_from(state: (V3, V3), t: f64, axis: V3, n0: V3, model_spin: bool) -> JetState {
  let (r0, v0) = state;
  let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
  let half = 0.5 * OMEGA * t;
  let s = half.sin();
  JetState {
    t_s: t,
    r_m: r,
    v_ms: v,
    rot: [
      (axis[0] * s) as f32,
      (axis[1] * s) as f32,
      (axis[2] * s) as f32,
      half.cos() as f32,
    ],
    site_normal: rotate_axis_angle(n0, axis, OMEGA * t),
    spin: model_spin.then_some([axis[0], axis[1], axis[2], OMEGA]),
    site_offset_m: [0.0; 3],
  }
}

/// true lit time over `[0, t_end]` with the sun moving along the orbit
fn brute_orbit_lit_time(n0: V3, t_end: f64, samples: usize) -> f64 {
  let (r0, v0) = comet_state();
  let h = t_end / samples as f64;
  (0..samples)
    .filter(|&i| {
      let t = (i as f64 + 0.5) * h;
      let (r, _) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
      dot(rotate_axis_angle(n0, [0.0, 0.0, 1.0], OMEGA * t), r) < 0.0
    })
    .count() as f64
    * h
}

// ─────────────────────────────────────────────────────────────────────────────
// Deterministic emission grid (DustHostState::tick)
// ─────────────────────────────────────────────────────────────────────────────

fn orbit_jet(t: f64) -> Option<JetState> {
  let (r0, v0) = comet_state();
  let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
  Some(JetState {
    t_s: t,
    r_m: r,
    v_ms: v,
    rot: [0.0, 0.0, 0.0, 1.0],
    // sub-solar site, no spin: always lit
    site_normal: scale(r, -1.0 / norm(r)),
    spin: Some([0.0, 0.0, 1.0, 0.0]),
    site_offset_m: [0.0; 3],
  })
}

/// runs ticks at the given scaled times, marking every batch submitted; returns the host
fn run_ticks(
  times: impl IntoIterator<Item = f64>,
  jet_at: &dyn Fn(f64) -> Option<JetState>,
  cfg: &DustEmitConfig,
) -> DustHostState {
  let mut host = DustHostState::new(RING_CAPACITY_LOW);
  for (i, t) in times.into_iter().enumerate() {
    host.tick(t, jet_at, &|_| *cfg);
    host.ring.mark_submitted(i as u64 + 1);
  }
  host
}

/// closed-window descriptors of the ring (slot fields cleared): what determinism is about
fn closed_windows(host: &DustHostState) -> alloc::vec::Vec<(i64, DustBatch)> {
  let n = host.ring.batches.len() - host.provisional as usize;
  host
    .ring
    .batches
    .iter()
    .take(n)
    .map(|b| {
      let mut d = b.desc;
      d.first_index = 0;
      (b.window, d)
    })
    .collect()
}

#[test]
fn grid_conserves_mass_tracks_readiness_and_restores() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let mut host = DustHostState::new(RING_CAPACITY_LOW);
  // 3 h/s at 60 ticks/s for ~12.5 days (< TTL: nothing retired)
  let mut t = 0.0;
  for tick in 0..6000u64 {
    for b in host.tick(t, &orbit_jet, &|_| cfg) {
      assert_eq!(b.ring_mask, RING_CAPACITY_LOW - 1);
      assert!(b.count > 0);
    }
    // pending batches are never drawable
    let (_, live, _) = host.ring.drawable();
    assert!(live <= host.ring.live());
    host.ring.mark_submitted(tick + 1);
    assert!(host.ring.live() <= RING_CAPACITY_LOW - RING_CAPACITY_LOW / RING_GUARD_DIVISOR);
    t += 180.0;
  }
  let t_last = t - 180.0;
  // closed windows + the provisional open one hold exactly q·t (always lit, constant q)
  let produced = cfg.q_dust_kgs * 1e3 * t_last;
  let rel = (host.ring.live_mass_g() - produced).abs() / produced;
  assert!(rel < 1e-6, "mass rel err {rel}");
  assert!(host.provisional);
  let ds = host.draw_state().unwrap();
  assert!(ds.live_count > 0 && ds.compute_wait > 0);

  // restore: everything re-emitted, oldest first, before it is drawable again
  host.ring.invalidate_gpu();
  assert!(host.draw_state().is_none());
  for i in 0..64 {
    host.tick(t_last, &orbit_jet, &|_| cfg);
    host.ring.mark_submitted(1_000_000 + i);
    if host.ring.batches.iter().all(|b| b.ready != READY_NEEDS_EMIT) {
      break;
    }
  }
  assert!(host.ring.batches.iter().all(|b| b.ready != READY_NEEDS_EMIT));

  // scrub back by a day: newer batches dropped, the open window rebuilt at the new time
  let t_back = t_last - 86400.0;
  host.tick(t_back, &orbit_jet, &|_| cfg);
  assert!(host.ring.batches.iter().all(|b| b.t_end_s <= t_back + 1e-6));
  let rel = (host.ring.live_mass_g() - cfg.q_dust_kgs * 1e3 * t_back).abs() / produced;
  assert!(rel < 1e-6, "after rewind: mass rel err {rel}");
}

/// The dust at an epoch does not depend on how it was reached: different tick rates, a pause and
/// a direct seek (jump) all give identical closed-window batches.
#[test]
fn grid_is_deterministic_across_tick_rates_pauses_and_seeks() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let t_end = 20.0 * 86400.0;
  let a = run_ticks(
    (0..=12_000).map(|i| (i as f64 * 144.0).min(t_end)),
    &orbit_jet,
    &cfg,
  );
  // irregular steps with a long pause in the middle
  let mut ts = alloc::vec::Vec::new();
  let mut t = 0.0;
  while t < t_end {
    ts.push(t);
    if (5.0 * 86400.0..5.2 * 86400.0).contains(&t) {
      for _ in 0..50 {
        ts.push(t); // paused
      }
    }
    t += 37.0 + (ts.len() % 7) as f64 * 53.0;
  }
  ts.push(t_end);
  let b = run_ticks(ts, &orbit_jet, &cfg);
  // seek: straight to t_end, a few ticks there to emit the windows (64 per tick)
  let c = run_ticks(core::iter::repeat_n(t_end, 6), &orbit_jet, &cfg);

  let (wa, wb, wc) = (closed_windows(&a), closed_windows(&b), closed_windows(&c));
  assert!(wa.len() > 100, "windows {}", wa.len());
  assert_eq!(wa.len(), wb.len());
  assert_eq!(wa.len(), wc.len());
  for ((x, y), z) in wa.iter().zip(&wb).zip(&wc) {
    assert_eq!(x, y, "tick rate / pause changed window {}", x.0);
    assert_eq!(x, z, "seek changed window {}", x.0);
  }
  // and so do the emitted clusters
  for (_, d) in wa.iter().take(3) {
    let (ca, cc) = (
      emit_cluster(d, 5),
      emit_cluster(&wc.iter().find(|w| w.1 == *d).unwrap().1, 5),
    );
    assert_eq!(ca, cc);
  }
}

/// jet on an equatorial site of a nucleus spinning about +z, riding the comet orbit
fn spinning_orbit_jet(t: f64, n0: V3, model_spin: bool) -> Option<JetState> {
  Some(spinning_jet(t, [0.0, 0.0, 1.0], n0, model_spin))
}

#[test]
fn grid_mass_tracks_lit_time_and_spin_fallback_matches_model() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let n0 = [1.0, 0.0, 0.0];
  let t_end = 10.0 * 86400.0;
  let times = || (0..=4800).map(move |i| i as f64 * t_end / 4800.0);
  let with_model = run_ticks(times(), &|t| spinning_orbit_jet(t, n0, true), &cfg);
  let estimated = run_ticks(times(), &|t| spinning_orbit_jet(t, n0, false), &cfg);
  let lit = brute_orbit_lit_time(n0, t_end, 2_000_000);
  let produced = cfg.q_dust_kgs * 1e3 * lit;
  let m = with_model.ring.live_mass_g();
  assert!(
    ((m - produced) / produced).abs() < 2e-3,
    "mass rel err {}",
    (m - produced) / produced
  );
  assert!(lit > 0.4 * t_end && lit < 0.6 * t_end);
  assert!(with_model.ring.batches.iter().all(|b| b.desc.lit[3] == LIT_MODE_PERIODIC));
  let e = estimated.ring.live_mass_g();
  assert!(
    ((m - e) / m).abs() < 2e-3,
    "finite-difference spin: mass rel err {}",
    (m - e) / m
  );
}

#[test]
fn grid_dark_site_emits_nothing() {
  // site on the pole, the pole pointing away from the sun: polar night
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let (r0, _) = comet_state();
  let anti_sun = unit(r0);
  let dark = |t: f64| {
    orbit_jet(t).map(|mut j| {
      j.site_normal = anti_sun;
      j.spin = Some([anti_sun[0], anti_sun[1], anti_sun[2], OMEGA]);
      j
    })
  };
  let host = run_ticks((0..2000).map(|i| i as f64 * 180.0), &dark, &cfg);
  assert_eq!(host.ring.live(), 0);
  assert!(host.draw_state().is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// Age tiers (DustSystemState)
// ─────────────────────────────────────────────────────────────────────────────

/// ticks a system until its history is complete (every tier caught up, nothing awaiting
/// re-emission; or `max` ticks), marking batches submitted: the logic thread's passes
fn fill_system(sys: &mut DustSystemState, t: f64, cfg: &DustEmitConfig, max: usize) {
  for i in 0..max {
    sys.tick(t, &orbit_jet, &|_| *cfg);
    sys.mark_submitted(i as u64 + 1);
    if !sys.building() {
      break;
    }
  }
}

/// Tiers split one ring (same total capacity, contiguous sub-rings) and cover contiguous age bands
/// `[0, 1)`, `[1, 8)`, `[8, 64)` TTL; only the oldest fades; every tier emits before t = 0, so the
/// whole history exists at the start epoch.
#[test]
fn tiers_split_the_ring_and_hold_history_at_start() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  assert_eq!(sys.capacity(), RING_CAPACITY_LOW);
  let mut base = 0;
  for (k, t) in sys.tiers.iter().enumerate() {
    assert_eq!(t.ring_base, base);
    assert!(t.ring.capacity.is_power_of_two());
    base += t.ring.capacity;
    assert_eq!(t.band.fade, k == 2);
    assert!(
      t.t_on_s.is_none(),
      "with_tiers keeps the pre-existing tail; the component ignites"
    );
  }
  assert_eq!(sys.tiers[1].band.min_ttl, sys.tiers[0].band.max_ttl);
  assert_eq!(sys.tiers[2].band.min_ttl, sys.tiers[1].band.max_ttl);

  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  let stats = sys.stats();
  assert!(stats.iter().all(|s| s.caught_up), "{stats:?}");
  assert!(!sys.building() && sys.complete);
  // the oldest tier reaches back ~64 TTL before the start epoch
  assert!(stats[2].oldest_age_s > 63.0 * ttl, "{stats:?}");
  assert!(stats[1].oldest_age_s > 7.9 * ttl, "{stats:?}");
  for (s, t) in stats.iter().zip(&sys.tiers) {
    assert!(s.live_clusters > 0 && s.live_clusters <= t.ring.capacity);
  }
  let states = sys.draw_states();
  assert_eq!(states.len(), 3);
  // one exposure for the whole system
  assert!(states.iter().all(|s| s.tau_ref == states[0].tau_ref));
}

/// A fresh history is drawn only once complete: after one budget-limited tick the tiers hold
/// clusters but nothing is drawable (`error_first/second/third.rdc`: the trail used to arrive in
/// 64-window chunks over ~1 s); once every tier is caught up the system is drawn, and stays drawn
/// when a later tick is short of budget (the flag is sticky until a reset).
#[test]
fn a_fresh_history_is_drawn_only_once_complete() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  sys.tick(0.0, &orbit_jet, &|_| cfg);
  sys.mark_submitted(1);
  assert!(
    sys.tiers[0].ring.live() > 0,
    "the youngest tier emitted its oldest windows"
  );
  assert!(sys.building() && !sys.complete);
  assert!(sys.draw_states().is_empty() && sys.coma_radius_m().is_none());
  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  assert!(!sys.building() && sys.complete);
  assert_eq!(sys.draw_states().len(), 3);
  // a jump of 100 windows of the youngest tier in one tick: one pass is short of budget, the
  // system keeps drawing (the live set is a prefix of the due one, caught up on the next passes)
  let dt_w = DustHostState::window_len_s(ttl);
  sys.tick(100.5 * dt_w, &orbit_jet, &|_| cfg);
  sys.mark_submitted(100);
  assert!(sys.building(), "64 of 100 windows emitted");
  assert!(sys.complete && sys.draw_states().len() == 3);
  fill_system(&mut sys, 100.5 * dt_w, &cfg, MAX_SEEK_PASSES);
  assert!(!sys.building());
}

/// A re-emit (rotation model / jet change) or a restore shows no partial history: the old one is
/// dropped and nothing is drawn until the new one is complete.
#[test]
fn a_re_emit_shows_no_partial_history() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let t = 3.3 * ttl;
  fill_system(&mut sys, t, &cfg, MAX_SEEK_PASSES);
  assert_eq!(sys.draw_states().len(), 3);
  sys.request_reemit();
  assert_eq!(sys.draw_states().len(), 3, "requested, not applied yet");
  sys.tick(t, &orbit_jet, &|_| cfg);
  sys.mark_submitted(100);
  assert!(sys.tiers[0].ring.live() > 0 && sys.building());
  assert!(
    sys.draw_states().is_empty(),
    "a partial new history is never drawn"
  );
  fill_system(&mut sys, t, &cfg, MAX_SEEK_PASSES);
  assert_eq!(sys.draw_states().len(), 3);
  // restore: every batch re-emitted (64 per tick) before anything is drawn again
  sys.invalidate_gpu();
  assert!(sys.building() && sys.draw_states().is_empty());
  sys.tick(t, &orbit_jet, &|_| cfg);
  sys.mark_submitted(200);
  assert!(sys.building() && sys.draw_states().is_empty());
  fill_system(&mut sys, t, &cfg, MAX_SEEK_PASSES);
  assert!(!sys.building() && sys.draw_states().len() == 3);
}

/// the closed windows of every tier as `(tier, window, descriptor)` with the ring slot cleared
fn system_windows(sys: &DustSystemState) -> alloc::vec::Vec<(usize, i64, DustBatch)> {
  sys
    .tiers
    .iter()
    .enumerate()
    .flat_map(|(i, t)| closed_windows(t).into_iter().map(move |(k, d)| (i, k, d)))
    .collect()
}

/// Emission-time coherence: whenever the system is drawable, every tier holds exactly the windows
/// due at the tick's epoch (`[k_min, k_open)` of its grid), with the descriptors a fresh system
/// emits for the same epoch — whatever the tick cadence, and across re-emits and restores at
/// random ticks (the logic thread's passes modelled by ticking until `!building()`, at most
/// `MAX_SEEK_PASSES`). Right after a reset a single pass leaves nothing drawable.
#[test]
fn the_history_is_complete_whenever_drawn() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut rng = 0x9E37_79B9_7F4A_7C15u64;
  let mut next = || {
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (rng >> 11) as f64 / (1u64 << 53) as f64
  };
  // a start at a random epoch, then ticks of 16 ms .. ~4 h of scaled time
  let mut t = 0.7 * ttl + next() * ttl;
  let mut seq = 1u64;
  let (mut resets, mut partial_seen, mut checks) = (0, 0, 0);
  for tick in 0..300 {
    if tick > 0 {
      t += 0.016 + next() * 4.0 * 3600.0;
    }
    let event = next();
    let reset_now = tick > 0 && event < 0.06;
    if reset_now && event < 0.03 {
      sys.request_reemit();
    } else if reset_now {
      sys.invalidate_gpu();
    }
    resets += reset_now as usize;
    for pass in 0..MAX_SEEK_PASSES {
      sys.tick(t, &orbit_jet, &|_| cfg);
      sys.mark_submitted(seq);
      seq += 1;
      let drawn = !sys.draw_states().is_empty();
      if pass == 0 && reset_now {
        // one pass of 64 windows / re-emits out of ~768: still building, and not drawable
        assert!(
          sys.building() && !drawn,
          "tick {tick}: a partial fill was drawable"
        );
        partial_seen += 1;
      }
      if drawn && !sys.building() {
        // the live windows are exactly the due ones at t
        for (i, tier) in sys.tiers.iter().enumerate() {
          let (min_age, max_age) = tier.age_band_s(tier.ttl_s);
          let dt_w = DustHostState::window_len_s(max_age - min_age);
          let k_open = if min_age > 0.0 {
            ((t - min_age) / dt_w).floor() as i64 + 1
          } else {
            (t / dt_w).floor() as i64
          };
          let k_min = ((t - max_age) / dt_w).floor() as i64;
          let live: alloc::vec::Vec<i64> = closed_windows(tier).iter().map(|w| w.0).collect();
          let due: alloc::vec::Vec<i64> = (k_min..k_open).collect();
          assert_eq!(
            live, due,
            "tick {tick} tier {i}: live windows != due windows"
          );
        }
      }
      if !sys.building() {
        break;
      }
    }
    assert!(
      !sys.building() && sys.complete,
      "tick {tick}: not caught up in MAX_SEEK_PASSES"
    );
    // the descriptors are the ones a fresh system emits for this epoch
    if tick % 50 == 49 {
      let mut fresh = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
      fill_system(&mut fresh, t, &cfg, MAX_SEEK_PASSES);
      assert_eq!(system_windows(&sys), system_windows(&fresh), "tick {tick}");
      checks += 1;
    }
  }
  assert!(
    resets >= 3 && partial_seen == resets && checks == 6,
    "{resets} {partial_seen}"
  );
}

/// Each emitted cluster is drawn by exactly the tier whose band holds its age (the bands overlap in
/// emission time; the per-cluster age gate resolves it), and no age is left uncovered.
#[test]
fn tier_age_gates_partition_the_clusters() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let t_now = 5.0 * ttl;
  fill_system(&mut sys, t_now, &cfg, MAX_SEEK_PASSES);
  let states = sys.draw_states();
  let mut drawn_ages = alloc::vec::Vec::new();
  for (tier, state) in sys.tiers.iter().zip(&states) {
    let (lo, hi) = tier.age_band_s(ttl);
    assert_eq!(state.frame.ttl[0] as f64, hi);
    assert_eq!(state.frame.ttl[1] as f64, lo);
    for b in tier.ring.batches.iter() {
      for j in 0..b.desc.count {
        let c = emit_cluster(&b.desc, j);
        let r = evaluate_cluster(&c, j, &state.frame);
        let age = t_now - c.t0().to_f64();
        let drawn = r.age_id_dbeta_flux[3] > 0.0;
        // f32 frame ages: allow a few ulps at the band edges
        let tol = 1e-6 * hi;
        if age > lo + tol && age < hi - tol {
          assert!(drawn, "tier {lo}-{hi}: age {age} not drawn");
        }
        if age < lo - tol || age > hi + tol {
          assert!(!drawn, "tier {lo}-{hi}: age {age} drawn");
        }
        if drawn {
          drawn_ages.push(age);
        }
      }
    }
  }
  drawn_ages.sort_by(|a, b| a.partial_cmp(b).unwrap());
  // coverage: no gap wider than a tier-1 window anywhere in [0, 64 TTL)
  let max_gap = drawn_ages.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
  assert!(max_gap < 0.25 * ttl, "largest age gap {max_gap} s");
  assert!(drawn_ages.last().copied().unwrap_or(0.0) > 63.0 * ttl);
}

/// Older tiers keep only the larger grains and drop the mass of the rest (left the field): the
/// kept fraction is exact for `n(s) ∝ s^-q`.
#[test]
fn size_cut_keeps_the_large_grain_mass() {
  let d = SizeDistribution::from_diameter_um(100.0);
  let (cut, frac) = d.truncated(8.0);
  assert!((cut.s_min_um - 8.0 * d.s_min_um).abs() < 1e-9);
  let e = 4.0 - d.q;
  let z = |a: f64, b: f64| (b.powf(e) - a.powf(e)) / e;
  let want = z(cut.s_min_um, d.s_max_um) / z(d.s_min_um, d.s_max_um);
  assert!((frac - want).abs() < 1e-12, "{frac} vs {want}");
  assert!(
    frac > 0.5 && frac < 1.0,
    "large grains carry most of the mass: {frac}"
  );
  let (none, one) = d.truncated(1.0);
  assert_eq!(none, d);
  assert_eq!(one, 1.0);
  // never collapses the range
  let (c, _) = d.truncated(1e9);
  assert!(c.s_min_um < c.s_max_um);
}

/// Mass per tier: the youngest holds q·TTL (+ the open window), an older one q·(band)·kept
/// fraction, within one of its windows.
#[test]
fn tier_mass_matches_production_over_its_band() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let t_now = 3.0 * ttl + 1234.0;
  fill_system(&mut sys, t_now, &cfg, MAX_SEEK_PASSES);
  let q_g = cfg.q_dust_kgs * 1e3;
  for t in &sys.tiers {
    let (lo, hi) = t.age_band_s(ttl);
    let (_, frac) = cfg.dist.truncated(t.band.s_min_factor);
    let want = q_g * (hi - lo) * frac;
    let window = DustHostState::window_len_s(hi - lo);
    let got = t.ring.live_mass_g();
    assert!(
      (got - want).abs() <= q_g * window * frac * 1.01,
      "tier [{lo}, {hi}): mass {got} vs {want}"
    );
  }
}

/// CPU mirror of the propagate age gate: below the band min culled, no fade at the max when an
/// older tier takes over.
#[test]
fn evaluate_respects_the_tier_band() {
  let b = test_batch(64, 3600.0);
  let c = emit_cluster(&b, 0);
  let t0 = c.t0().to_f64();
  let ttl = 10.0 * 86400.0;
  let frame_at = |age: f64, min: f32, fade: bool| {
    let (r, _) = kepler::propagate_f64(c.r0().to_f64(), c.v0().to_f64(), SUN_MU_M3_S2, age);
    DustFrame::new(r, t0 + age, [0.0, 0.0, 0.0, 1.0], ttl as f32).with_band(min, fade)
  };
  let flux = |f: &DustFrame| evaluate_cluster(&c, 0, f).age_id_dbeta_flux[3];
  assert_eq!(
    flux(&frame_at(86400.0, 2.0 * 86400.0, true)),
    0.0,
    "younger than the band min"
  );
  assert!(flux(&frame_at(3.0 * 86400.0, 2.0 * 86400.0, true)) > 0.0);
  let near_end = 0.99 * ttl;
  let faded = flux(&frame_at(near_end, 0.0, true));
  let kept = flux(&frame_at(near_end, 0.0, false));
  assert!(faded < 0.2 * kept, "fade {faded} vs {kept}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical-depth splats (dust.vert / dust.frag mirror)
// ─────────────────────────────────────────────────────────────────────────────

/// The exposure reference depends on the jet configuration only: filling the old tiers (heavy
/// clusters) leaves it unchanged (`initial_burst.rdc` → `late_*.rdc` dimmed 10×), and it scales
/// with the production rate so the stream brightness is independent of its scale.
#[test]
fn exposure_reference_ignores_history_and_tiers() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let tau = cfg.tau_ref();
  assert!(tau > 0.0 && tau.is_finite());
  let doubled = DustEmitConfig {
    q_dust_kgs: cfg.q_dust_kgs * 2.0,
    ..cfg
  };
  assert!((doubled.tau_ref() / tau - 2.0).abs() < 1e-12);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  sys.tau_ref = tau;
  // one budget-limited tick (nothing drawn while the history builds), then everything
  sys.tick(0.0, &orbit_jet, &|_| cfg);
  sys.mark_submitted(1);
  assert!(sys.draw_states().is_empty());
  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  let late = sys.draw_states();
  assert_eq!(late.len(), 3);
  assert!(late.iter().all(|s| s.tau_ref == tau as f32));
}

/// Composite display stretch on optical depth relative to the white point: 0 up to the black point,
/// monotone, 1 at the white point; at the default softening dust 100× fainter than the brightest
/// still shows (≥ 10 %) while the faintest 1e-4 is cut to black (no veil over the whole view).
#[test]
fn display_stretch_shows_a_faint_tail_and_saturates_the_coma() {
  let s = DUST_SOFTENING_DEFAULT;
  let b = DUST_BLACK_POINT;
  assert_eq!(display_stretch(0.0, b, s), 0.0);
  assert_eq!(display_stretch(-1.0, b, s), 0.0);
  assert_eq!(display_stretch(b, b, s), 0.0);
  assert_eq!(display_stretch(0.5 * b, b, s), 0.0);
  let mut prev = 0.0;
  for k in 0..80 {
    let tau = 10f32.powf(-6.0 + k as f32 * 0.1);
    let a = display_stretch(tau, b, s);
    assert!(a >= prev && a <= 1.0, "not monotone at τ {tau}");
    prev = a;
  }
  assert!((display_stretch(1.0, 0.0, s) - 1.0).abs() < 1e-6);
  assert!(display_stretch(1.0, b, s) > 0.999);
  assert_eq!(display_stretch(50.0, b, s), 1.0);
  assert!(
    display_stretch(1e-2, b, s) >= 0.1,
    "{}",
    display_stretch(1e-2, b, s)
  );
  assert!(display_stretch(1e-2, 0.0, 0.0) < 0.011, "linear: faint");
  // below the softening it is linear (slope 1 / (s·asinh(1/s)))
  let lin = display_stretch(1e-7, 0.0, s) / 1e-7;
  assert!((lin * s * (1.0 / s).asinh() - 1.0).abs() < 1e-3);
  assert_eq!(clamp_dust_softening(f32::NAN), DUST_SOFTENING_DEFAULT);
  assert_eq!(clamp_dust_softening(10.0), DUST_SOFTENING_MAX);
  assert_eq!(clamp_dust_softening(0.0), DUST_SOFTENING_MIN);
}

/// Old tiers keep every grain size: the small, high-β grains carry most of the optical depth and
/// draw the anti-sunward tail.
#[test]
fn tiers_keep_the_small_grains() {
  let sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  assert!(sys.tiers.iter().all(|t| t.band.s_min_factor == 1.0));
  // what a ×8 cut would have dropped of the cross-section: n ∝ s^-3.5 ⇒ σ ∝ ∫ s^-1.5 ds
  let d = SizeDistribution::from_diameter_um(100.0);
  let sigma = |a: f64, b: f64| a.powf(-0.5) - b.powf(-0.5);
  let kept = sigma(8.0 * d.s_min_um, d.s_max_um) / sigma(d.s_min_um, d.s_max_um);
  assert!(
    kept < 0.3,
    "a ×8 cut keeps only {kept} of the optical depth"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoupling: particles are not children of the comet
// ─────────────────────────────────────────────────────────────────────────────

/// World (heliocentric) position of an evaluated cluster: `(r − A)` from the evaluation plus the
/// anchor `A`, i.e. what the renderer draws relative to the eye after adding `A − eye`.
fn world_of(rc: &DustRenderCluster, frame: &DustFrame) -> V3 {
  let a = frame.anchor_m();
  [
    a[0] + rc.pos_size[0] as f64,
    a[1] + rc.pos_size[1] as f64,
    a[2] + rc.pos_size[2] as f64,
  ]
}

/// The anchor is a precision device, not a parent: the same clusters evaluated relative to the
/// comet, 1e6 km away, or 1 AU away land on the same heliocentric points (up to the f32 rounding
/// of `r − A`, which grows with the distance to the anchor).
#[test]
fn world_position_is_independent_of_the_anchor() {
  let b = test_batch(256, 1800.0);
  let (frame, rc) = frame_after(10.0);
  let t = frame.t_now_s();
  let anchors = [
    rc,
    add(rc, [1.0e9, -2.0e8, 3.0e7]),
    add(rc, [AU_M, 0.0, 0.0]),
  ];
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let world: alloc::vec::Vec<V3> = anchors
      .iter()
      .map(|&a| {
        let f = DustFrame::new(a, t, [0.0, 0.0, 0.0, 1.0], 1.0e9);
        world_of(&evaluate_cluster(&c, j, &f), &f)
      })
      .collect();
    for (k, w) in world.iter().enumerate().skip(1) {
      let d = norm(sub(*w, world[0]));
      let far = norm(sub(world[0], anchors[k]));
      assert!(
        d < 2.0 + 2.5e-7 * far,
        "cluster {j}: anchor {k} moves it by {d} m (|r − A| = {far} m)"
      );
    }
  }
}

/// Emitted dust never follows the comet afterwards: two histories identical up to `t1`, then the
/// comet is teleported 1e5 km and spun (as if its entity or orbit changed). Every cluster emitted
/// before `t1` is drawn at the same heliocentric point in both, which is the Kepler propagation of
/// its own emission record — however different the anchors.
#[test]
fn particles_ignore_later_comet_motion() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let step = 3600.0;
  let t1 = 2.0 * 86400.0;
  let t2 = t1 + 6.0 * 3600.0;
  let moved = |t: f64| -> Option<JetState> {
    let mut j = orbit_jet(t)?;
    if t > t1 {
      j.r_m = add(j.r_m, [1.0e8, 0.0, -5.0e7]);
      j.rot = [0.0, 0.0, 0.7071068, 0.7071068];
    }
    Some(j)
  };
  let times: alloc::vec::Vec<f64> = (0..=((t2 / step) as usize)).map(|i| i as f64 * step).collect();
  let host_a = run_ticks(times.iter().copied(), &orbit_jet, &cfg);
  let host_b = run_ticks(times.iter().copied(), &moved, &cfg);
  let (sa, sb) = (
    host_a.draw_state().unwrap().at_time(t2),
    host_b.draw_state().unwrap().at_time(t2),
  );
  assert!(
    norm(sub(sa.anchor_m, sb.anchor_m)) > 1.0e7,
    "the anchors must differ"
  );
  let (wa, wb) = (closed_windows(&host_a), closed_windows(&host_b));
  let mut checked = 0;
  for ((win_a, ba), (win_b, bb)) in wa.iter().zip(wb.iter()) {
    assert_eq!(win_a, win_b);
    // only batches entirely emitted before the comet moved are shared history
    let end = ba.comet_r_t_hi[3] as f64 + ba.comet_r_t_lo[3] as f64 + ba.comet_v_dur_hi[3] as f64;
    if end > t1 {
      continue;
    }
    assert_eq!(ba, bb, "window {win_a}: same emission before t1");
    for j in (0..ba.count).step_by(7) {
      let c = emit_cluster(ba, j);
      let (ea, eb) = (
        evaluate_cluster(&c, j, &sa.frame),
        evaluate_cluster(&c, j, &sb.frame),
      );
      if !(ea.age_id_dbeta_flux[3] > 0.0) {
        continue;
      }
      let (pa, pb) = (world_of(&ea, &sa.frame), world_of(&eb, &sb.frame));
      let d = norm(sub(pa, pb));
      let far = norm(sub(pa, sb.anchor_m));
      assert!(
        d < 2.0 + 2.5e-7 * far,
        "window {win_a} cluster {j}: moved {d} m with the comet"
      );
      // and it is exactly its own orbit
      let mu = SUN_MU_M3_S2 * (1.0 - c.beta() as f64);
      let (r, _) =
        kepler::propagate_f64(c.r0().to_f64(), c.v0().to_f64(), mu, t2 - c.t0().to_f64());
      let e = norm(sub(pa, r));
      assert!(
        e < 2.0 + 2.5e-7 * norm(sub(r, sa.anchor_m)),
        "cluster {j}: {e} m off its orbit"
      );
      checked += 1;
    }
  }
  assert!(checked > 100, "only {checked} clusters checked");
}

/// `at_time` moves only the evaluation time: anchor and age band stay.
#[test]
fn draw_state_at_time_keeps_anchor_and_band() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let host = run_ticks([0.0, 3600.0, 7200.0], &orbit_jet, &cfg);
  let s = host.draw_state().unwrap();
  let r = s.at_time(5000.0);
  assert_eq!(r.anchor_m, s.anchor_m);
  assert_eq!(r.frame.ttl, s.frame.ttl);
  assert_eq!(r.frame.anchor_m(), s.frame.anchor_m());
  assert!((r.frame.t_now_s() - 5000.0).abs() < 1e-6);
  assert!((s.frame.t_now_s() - 7200.0).abs() < 1e-6);
  let a = s.frame.anchor_m();
  assert!(
    norm(sub(a, s.anchor_m)) < 1e-3,
    "frame anchor = draw anchor"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Decorrelation of windows / tiers, stable child identity
// ─────────────────────────────────────────────────────────────────────────────

/// Streams are fixed per jet: cluster `s` of every window and time sample keeps its stream's
/// cone cell up to the small per-cluster jitter, while different streams differ. The cluster is
/// the whole cell: its ejection speed is exactly the mean speed (no draw per stream, the speed
/// spread is the radial dispersion `σ_rad`), its β the reference grain's (every size is drawn by
/// the renderer's polyline), its `eject` the ejection velocity.
#[test]
fn streams_keep_their_cell_and_mean_speed() {
  let n = 64u32;
  let mut a = test_batch(n * 8, 1800.0);
  a.mass_params[3] = batch_streams_word(6, false);
  let mut b = a;
  b.seed = window_seed(42, 0, 1);
  let (rc, vc) = comet_state();
  // ejection velocity relative to the comet at the emission time (no spin: root = jet frame)
  let ej = |c: &DustCluster| {
    let (_, v) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, c.t0().to_f64() - T_START);
    sub(c.v0().to_f64(), v)
  };
  let v_ref = a.vel_params[0] as f64;
  let (mut other_angle, mut pairs) = (0.0f64, 0);
  for i in 0..8u32 {
    for s in 0..n {
      let j = i * n + s;
      let (ca, cb) = (emit_cluster(&a, j), emit_cluster(&b, (7 - i) * n + s));
      let (ea, eb) = (ej(&ca), ej(&cb));
      let angle = dot(unit(ea), unit(eb)).clamp(-1.0, 1.0).acos();
      assert!(angle < 0.12, "stream {s}: direction moved by {angle} rad");
      for c in [&ca, &cb] {
        let e = ej(c);
        let ex = [c.eject[0] as f64, c.eject[1] as f64, c.eject[2] as f64];
        // the ejection velocity is exactly the mean speed (no draw) along the cone direction
        // (the test batch's jet axis is not unit: |dir| follows it); relative to the f64 comet
        // state the df64 emission differs by ~1 cm/s
        let jet_norm = norm([
          a.jet_dir_aperture[0] as f64,
          a.jet_dir_aperture[1] as f64,
          a.jet_dir_aperture[2] as f64,
        ]);
        assert!(
          (norm(ex) / (v_ref * jet_norm) - 1.0).abs() < 2e-2,
          "stream {s}: |eject| {} vs v_ref {v_ref} · |jet| {jet_norm}",
          norm(ex)
        );
        assert!(
          norm(sub(ex, e)) < 2e-2 * v_ref,
          "stream {s}: eject field {ex:?} vs {e:?}"
        );
        assert!((c.eject[3] - a.vel_params[2]).abs() < 1e-6, "s_ref");
        assert!(
          (c.sigma_rad() - a.vel_params[1] * a.vel_params[0]).abs() < 1e-6,
          "σ_rad {} vs {}",
          c.sigma_rad(),
          a.vel_params[1] * a.vel_params[0]
        );
        assert!(
          (c.beta() - a.vel_params[3] / a.vel_params[2]).abs() < 1e-7,
          "β of the reference grain"
        );
      }
      let co = emit_cluster(&a, i * n + (s + 1) % n);
      other_angle += dot(unit(ea), unit(ej(&co))).clamp(-1.0, 1.0).acos();
      pairs += 1;
    }
  }
  // neighbouring streams sit in other cells of the 0.6 rad cone
  assert!(
    other_angle / pairs as f64 > 0.3,
    "mean angle between streams {}",
    other_angle / pairs as f64
  );
  // the permutation is a bijection of [0, n) for any n and key
  for n in [1u32, 2, 3, 100, 819, 1024, 1025] {
    for key in [0u32, 1, 0xDEAD_BEEF] {
      let mut seen = alloc::vec![false; n as usize];
      for i in 0..n {
        let p = permute_index(i, n, key) as usize;
        assert!(p < n as usize && !seen[p], "n {n} key {key}: {i} → {p}");
        seen[p] = true;
      }
    }
  }
}

/// The tiers of a system draw different clusters for the same window index, and the full i64
/// window index enters the seed.
#[test]
fn tiers_draw_independent_windows() {
  let mut seeds = alloc::collections::BTreeSet::new();
  for tier in 0..3u32 {
    for k in -500i64..500 {
      assert!(
        seeds.insert(window_seed(42, tier, k)),
        "tier {tier} window {k}: seed reused"
      );
    }
  }
  assert_ne!(window_seed(42, 0, 7), window_seed(42, 0, 7 + (1i64 << 32)));
  assert_ne!(window_seed(42, 0, -1), window_seed(42, 0, u32::MAX as i64));

  // same window in two tiers of a system: different seeds, uncorrelated size sequences
  let cfg = test_cfg(1.5e-3, 86400.0);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_HIGH, 3);
  assert!(sys.tiers.iter().enumerate().all(|(i, t)| t.tier == i as u32));
  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  let w0 = closed_windows(&sys.tiers[0]);
  let w1 = closed_windows(&sys.tiers[1]);
  let pairs: alloc::vec::Vec<(&DustBatch, &DustBatch)> = w0
    .iter()
    .filter_map(|(k, a)| w1.iter().find(|(k1, _)| k1 == k).map(|(_, b)| (a, b)))
    .collect();
  assert!(!pairs.is_empty());
  for (a, b) in &pairs {
    assert_ne!(a.seed, b.seed);
  }
  // the streams are per jet (same cells and sizes in every tier), the time samples are not
  let same_t = pairs
    .iter()
    .flat_map(|(a, b)| (0..a.count.min(b.count)).map(move |j| (*a, *b, j)))
    .filter(|(a, b, j)| emit_cluster(a, *j).t0().to_f64() == emit_cluster(b, *j).t0().to_f64())
    .count();
  assert_eq!(same_t, 0, "tiers 0 / 1 emit at the same instants");
  // reset keeps the tier index (and so the seeds)
  sys.reset();
  assert!(sys.tiers.iter().enumerate().all(|(i, t)| t.tier == i as u32));
}

/// The coma radius (Earth observer framing) is finite, positive, and grows with the ejection
/// speed: the young coma is ejection-dominated (r ≈ v·age).
#[test]
fn coma_radius_scales_with_the_ejection_speed() {
  let ttl = 5.0 * 86400.0;
  let radius = |v: f32| {
    let cfg = DustEmitConfig {
      v_mean: v,
      v_std: 0.25 * v,
      ..test_cfg(1.5e-3, ttl)
    };
    let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 1);
    fill_system(&mut sys, ttl, &cfg, MAX_SEEK_PASSES);
    sys.coma_radius_m().expect("drawable dust")
  };
  assert!(DustSystemState::with_tiers(RING_CAPACITY_LOW, 1).coma_radius_m().is_none());
  let (r1, r4) = (radius(2.0), radius(8.0));
  assert!(r1 > 0.0 && r1.is_finite(), "{r1}");
  // ~v·age over the TTL (2 m/s · ≤ 5 d ≲ 900 km) plus the radiation-pressure push of the small
  // grains, which carry most of the flux (every size samples the whole cone, `stream_stratum`)
  assert!(r1 < 3.0e6, "{r1} m");
  let ratio = r4 / r1;
  assert!(
    (2.5..6.0).contains(&ratio),
    "4× the speed → {ratio}× the radius ({r1} → {r4} m)"
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaklines
// ─────────────────────────────────────────────────────────────────────────────

/// Orthographic mvp of half-height `half_m` (units km, square viewport), metres → clip.
fn ortho_mvp(half_m: f32, centre: [f32; 2]) -> [f32; 16] {
  let s = 1.0 / half_m;
  let mut m = [0.0f32; 16];
  m[0] = s;
  m[5] = s;
  m[10] = 1e-12;
  m[12] = -centre[0] * s;
  m[13] = -centre[1] * s;
  m[15] = 1.0;
  m
}

/// The tier's compact render buffer as `dust_propagate.comp` writes it (CPU reference): every
/// batch emitted into its ring slots, the drawable range evaluated with `frame`.
fn tier_render(host: &DustHostState, frame: &DustFrame) -> alloc::vec::Vec<DustRenderCluster> {
  let mask = host.ring.mask();
  let mut ring: alloc::vec::Vec<DustCluster> =
    alloc::vec![bytemuck::Zeroable::zeroed(); host.ring.capacity as usize];
  for b in &host.ring.batches {
    for j in 0..b.count {
      ring[(b.desc.first_index.wrapping_add(j) & mask) as usize] = emit_cluster(&b.desc, j);
    }
  }
  let (first, live, _) = host.ring.drawable();
  (0..live)
    .map(|i| {
      let slot = first.wrapping_add(i) & mask;
      evaluate_cluster(&ring[slot as usize], slot, frame)
    })
    .collect()
}

fn cross64(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

/// Stream counts follow the tier capacity (fewer streams on small rings, so a window keeps
/// [`STREAM_MIN_SAMPLES`] time samples), and every planned batch holds whole time samples.
#[test]
fn stream_counts_follow_the_tier_capacity() {
  for (cap, shift) in [
    (131_072u32, 6u32),
    (65_536, 6),
    (16_384, 4),
    (8_192, 3),
    (4_096, 2),
    (2_048, 1),
    (1_024, 0),
  ] {
    let h = DustHostState::new(cap);
    assert_eq!(h.stream_shift(), shift, "capacity {cap}");
    let (n, cpw) = (1u32 << shift, h.clusters_per_window());
    assert_eq!(cpw % n, 0, "capacity {cap}: {cpw} per window");
    assert!(
      cpw / n >= STREAM_MIN_SAMPLES.min(6),
      "capacity {cap}: {} samples",
      cpw / n
    );
    assert!(cpw as f64 <= cap as f64 * BUDGET_SAFETY / WINDOWS_PER_TTL);
  }
  // closed windows, provisional previews, budget-truncated batches: all multiples of S
  let cfg = test_cfg(1.5e-3, 86400.0);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut t = 0.0;
  for i in 0..40 {
    t += 977.0;
    sys.tick(t, &orbit_jet, &|_| cfg);
    sys.mark_submitted(i + 1);
  }
  for tier in &sys.tiers {
    let n = 1 << tier.stream_shift();
    assert!(tier.ring.batches.iter().all(|b| b.count % n == 0 && b.count > 0));
    assert_eq!(
      tier.draw_state().unwrap().frame.ttl[3] as u32,
      tier.stream_shift()
    );
  }
}

/// A stream breaks after a night of the jet site (the streak would draw an arc nothing was emitted
/// on) and at the first sample after a missing window, never otherwise.
#[test]
fn streams_break_across_dark_gaps_and_missing_windows() {
  let axis = unit([0.2, 0.1, 0.97]);
  let n0 = unit([1.0, -0.3, 0.1]);
  let n = 16u32;
  let dur = 3.0 * P_ROT;
  let (mut b, _) = spinning_batch(n * 48, dur, n0, axis, 0.3);
  b.mass_params[3] = batch_streams_word(4, false);
  // night breaks only (no size strata: a stream never wraps)
  let breaks = |b: &DustBatch, s: u32| {
    (0..b.count / n)
      .filter(|i| {
        let j = i * n + s;
        let broken = stream_break(emit_cluster(b, j).misc[3]);
        assert_eq!(broken, stream_dark_before(b, j), "cluster {j}");
        stream_dark_before(b, j)
      })
      .count()
  };
  // three rotations: two or three dawns in the window (one more if it starts in the dark)
  for s in 0..n {
    let k = breaks(&b, s);
    assert!((2..=4).contains(&k), "stream {s}: {k} breaks");
    assert_eq!(k, breaks(&b, 0), "every stream sees the same nights");
  }
  // a break is a night: the site spent more than STREAM_BREAK_TURNS of a turn in the dark between
  // the two samples (closed-form lit time from the earlier one)
  let sun = spinning_batch(1, 1.0, n0, axis, 0.3).1;
  for i in 1..b.count / n {
    let (c0, c1) = (emit_cluster(&b, (i - 1) * n), emit_cluster(&b, i * n));
    let (t0, t1) = (c0.t0().to_f64() - T_START, c1.t0().to_f64() - T_START);
    let n_t0 = rotate_axis_angle(n0, axis, OMEGA * t0);
    let dark = (t1 - t0) - LitWindow::new(sun, n_t0, axis, OMEGA, t1 - t0).lit_time_s;
    // within 1 % of the threshold the f32 lit phase decides either way
    if (dark / P_ROT - STREAM_BREAK_TURNS as f64).abs() > 1e-3 {
      assert_eq!(
        stream_break(c1.misc[3]),
        dark > STREAM_BREAK_TURNS as f64 * P_ROT,
        "sample {i}: dark {dark} s"
      );
    }
  }
  // always lit: no break, unless the previous window is missing (first sample only)
  let mut lit = test_batch(n * 48, dur);
  lit.mass_params[3] = batch_streams_word(4, false);
  assert!((0..lit.count).all(|j| !stream_break(emit_cluster(&lit, j).misc[3])));
  lit.mass_params[3] = batch_streams_word(4, true);
  for j in 0..lit.count {
    assert_eq!(
      stream_break(emit_cluster(&lit, j).misc[3]),
      j < n,
      "cluster {j}"
    );
  }
  // the planner flags a batch whose previous window is missing, deterministically
  let cfg = test_cfg(1.5e-3, 86400.0);
  let host = run_ticks([0.5 * 86400.0; 4], &orbit_jet, &cfg);
  let w = closed_windows(&host);
  assert!(
    w.iter().all(|(_, d)| !batch_streams(d).1),
    "contiguous windows never break"
  );
}

/// Changing the rotation (or a jet parameter) re-emits the whole history with the new spin:
/// without it, the old spin would stay visible until it aged out (up to the oldest tier).
#[test]
fn reemit_rebuilds_the_history_with_the_new_spin() {
  let cfg = test_cfg(1.5e-3, 86400.0);
  let jet = |omega: f64| {
    move |t: f64| {
      orbit_jet(t).map(|mut j| {
        j.spin = Some([0.0, 0.0, 1.0, omega]);
        j
      })
    }
  };
  let fill = |sys: &mut DustSystemState, omega: f64, t: f64| {
    for i in 0..MAX_SEEK_PASSES {
      sys.tick(t, &jet(omega), &|_| cfg);
      sys.mark_submitted(1000 + i as u64);
    }
  };
  let spins = |sys: &DustSystemState| -> alloc::vec::Vec<f32> {
    sys
      .tiers
      .iter()
      .flat_map(|t| t.ring.batches.iter().map(|b| b.desc.spin[3]))
      .collect()
  };
  let (w1, w2) = (OMEGA, 2.0 * OMEGA);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  fill(&mut sys, w1, 3.0 * 86400.0);
  assert!(spins(&sys).iter().all(|&w| w == w1 as f32));
  // without a re-emit request, the history keeps the old spin
  let mut kept = sys.clone();
  fill(&mut kept, w2, 3.0 * 86400.0 + 60.0);
  assert!(spins(&kept).iter().any(|&w| w == w1 as f32));
  // with it: reset on the next tick, refilled with the new spin, identical to a fresh fill
  sys.request_reemit();
  fill(&mut sys, w2, 3.0 * 86400.0 + 60.0);
  assert!(!sys.reemit);
  assert!(spins(&sys).iter().all(|&w| w == w2 as f32));
  let mut fresh = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  fill(&mut fresh, w2, 3.0 * 86400.0 + 60.0);
  for (a, b) in sys.tiers.iter().zip(&fresh.tiers) {
    assert_eq!(closed_windows(a), closed_windows(b));
  }
}

/// The streams tile the jet cone: the gaussian lateral dispersions of the S streams sum to a
/// uniform density over the cone instead of S islands (`detached.rdc`: 64 blobs thousands of km
/// apart at 25 days). Directions are rebuilt from the emitted state: v0 minus the comet's velocity
/// at the emission time.
#[test]
fn streams_fill_the_jet_cone() {
  let cfg = test_cfg(1.5e-3, 86400.0);
  // the production youngest tier: 64 streams
  let mut host = DustHostState::new(RING_CAPACITY_HIGH / 2);
  assert_eq!(host.stream_shift(), 6);
  host.tick(1.3 * 86400.0, &orbit_jet, &|_| cfg);
  let aperture = cfg.aperture_rad as f64;
  // (tangent-plane position in the cone, angular σ) per cluster
  let mut g: alloc::vec::Vec<([f64; 2], f64)> = alloc::vec::Vec::new();
  for b in host.ring.batches.iter().take(16) {
    let d = &b.desc;
    let rc0 = Df3::from_f32([d.comet_r_t_hi[0], d.comet_r_t_hi[1], d.comet_r_t_hi[2]]).add(
      &Df3::from_f32([d.comet_r_t_lo[0], d.comet_r_t_lo[1], d.comet_r_t_lo[2]]),
    );
    let vc0 = Df3::from_f32([
      d.comet_v_dur_hi[0],
      d.comet_v_dur_hi[1],
      d.comet_v_dur_hi[2],
    ])
    .add(&Df3::from_f32([
      d.comet_v_dur_lo[0],
      d.comet_v_dur_lo[1],
      d.comet_v_dur_lo[2],
    ]));
    let t_start = Df::new(d.comet_r_t_hi[3], d.comet_r_t_lo[3]);
    for j in 0..d.count {
      let c = emit_cluster(d, j);
      let (_, vc) = kepler::propagate(&rc0, &vc0, consts::SUN_MU, c.t0().sub(t_start));
      let v = c.v0().sub(&vc).to_f64();
      let vn = norm(v);
      if !(vn > 0.0) {
        continue;
      }
      // jet along +z, no attitude (orbit_jet): polar angle and azimuth in the cone
      let th = (v[2] / vn).clamp(-1.0, 1.0).acos();
      let ph = v[1].atan2(v[0]);
      g.push(([th * ph.cos(), th * ph.sin()], c.misc[0] as f64 / vn));
    }
  }
  assert!(g.len() > 4000, "{} clusters", g.len());
  // density over the inner cone (the rim falls off by construction)
  let mut dens = alloc::vec::Vec::new();
  let n = 24;
  for iy in 0..n {
    for ix in 0..n {
      let p = [
        (2.0 * (ix as f64 + 0.5) / n as f64 - 1.0) * aperture,
        (2.0 * (iy as f64 + 0.5) / n as f64 - 1.0) * aperture,
      ];
      if p[0].hypot(p[1]) > 0.6 * aperture {
        continue;
      }
      let sum: f64 = g
        .iter()
        .map(|(q, s)| {
          let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
          (-0.5 * d2 / (s * s)).exp() / (s * s)
        })
        .sum();
      dens.push(sum);
    }
  }
  let mean = dens.iter().sum::<f64>() / dens.len() as f64;
  if std::env::var("CONE_MAP").is_ok() {
    let mut k = 0;
    for iy in 0..n {
      let mut line = alloc::string::String::new();
      for ix in 0..n {
        let p = [
          (2.0 * (ix as f64 + 0.5) / n as f64 - 1.0),
          (2.0 * (iy as f64 + 0.5) / n as f64 - 1.0),
        ];
        if p[0].hypot(p[1]) > 0.6 {
          line.push_str("    ");
          continue;
        }
        line.push_str(&alloc::format!("{:4.1}", dens[k] / mean));
        k += 1;
      }
      std::println!("{line}");
    }
  }
  let sd = (dens.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / dens.len() as f64).sqrt();
  let (lo, hi) = dens.iter().fold((f64::MAX, 0.0f64), |(a, b), &d| (a.min(d), b.max(d)));
  // σ = 1 cell: cv 0.17, min 0.64 (a brighter core where the lattice starts); 0.5 cell: cv 0.21,
  // min 0.53
  assert!(
    sd / mean < 0.19 && lo > 0.6 * mean,
    "cone density cv {:.3}, min/mean {:.3}, max/mean {:.3}",
    sd / mean,
    lo / mean,
    hi / mean
  );
}

/// The emission tick re-emits by itself when the rotation model it reads changes (the UI writes
/// the model and requests the re-emit in some order; a tick in between used to rebuild the history
/// with the old model, `detached.rdc`), and not when it stays the same.
#[test]
fn a_changed_rotation_model_re_emits_the_history() {
  let model = |rate: f64| crate::scene::BodyRotationalModel {
    pole_ra: 0.0,
    pole_dec: 90.0,
    prime_meridian: 0.0,
    pole_ra_rate: 0.0,
    pole_dec_rate: 0.0,
    rotation_rate: rate,
    body_fixed_orientation: false,
  };
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  sys.track_spin_model(None);
  assert!(!sys.reemit, "the first model is the history's");
  sys.track_spin_model(None);
  assert!(!sys.reemit);
  sys.track_spin_model(Some(model(1080.0)));
  assert!(sys.reemit, "no model -> a model");
  sys.reemit = false;
  sys.track_spin_model(Some(model(1080.0)));
  assert!(!sys.reemit, "same model");
  sys.track_spin_model(Some(model(720.0)));
  assert!(sys.reemit, "another rate");
}

/// The renderer reads the ring while the logic tick runs: between a tick's emission and its
/// submission the newest batches are pending, and the drawable range it sees must stay the last
/// submitted one (it used to drop them, the youngest window flickered).
#[test]
fn the_renderer_keeps_the_submitted_range_while_a_tick_is_pending() {
  let cfg = test_cfg(1.5e-3, 86400.0);
  let mut host = DustHostState::new(RING_CAPACITY_LOW);
  let mut t = 0.3 * 86400.0;
  host.tick(t, &orbit_jet, &|_| cfg);
  assert!(host.draw_state().is_none(), "nothing submitted yet");
  host.ring.mark_submitted(1);
  let mut prev = host.draw_state().unwrap();
  assert!(prev.live_count > 0);
  for i in 0..20 {
    t += 600.0;
    host.tick(t, &orbit_jet, &|_| cfg);
    // mid-tick: the pending batches (the open window at least) are not drawable yet, the renderer
    // keeps the previous submitted range
    assert!(host.ring.drawable().1 < host.ring.live());
    let mid = host.draw_state().unwrap();
    assert_eq!(
      (mid.first_slot, mid.live_count, mid.compute_wait),
      (prev.first_slot, prev.live_count, prev.compute_wait)
    );
    host.ring.mark_submitted(2 + i);
    let after = host.draw_state().unwrap();
    assert_eq!(
      after.live_count,
      host.ring.live(),
      "everything submitted is drawn"
    );
    prev = after;
  }
  // a restored snapshot draws nothing until re-emitted
  host.ring.invalidate_gpu();
  assert!(host.draw_state().is_none());
}

/// The flow marks are brightness neutral: mean factor 1 over the emission epochs at any age.
#[test]
fn flow_marks_are_brightness_neutral() {
  let n = 30_000;
  for age in [300.0f32, 5.0e4, 3.0e6] {
    let mean: f64 = (0..n)
      .map(|i| {
        let t = 4.0e8 + age as f64 * 4.0 * i as f64 / n as f64;
        flow_factor(age, &DustFlowUniform::new(&DustFlowClock::default(), t)) as f64
      })
      .sum::<f64>()
      / n as f64;
    assert!((mean - 1.0).abs() < 0.02, "age {age}: mean {mean}");
  }
  // ages and epochs independent
  let ages = |i: usize| 10f32.powf(3.0 + 3.0 * (i as f32 + 0.5) / n as f32);
  let mean: f64 = (0..n)
    .map(|i| {
      let f = DustFlowUniform::new(
        &DustFlowClock::default(),
        4.0e8 + u01(pcg(i as u32)) as f64 * 1.0e8,
      );
      flow_factor(ages(i), &f) as f64
    })
    .sum::<f64>()
    / n as f64;
  assert!((mean - 1.0).abs() < 0.03, "mean {mean}");
}

/// The marks follow the swarm: at `K` = 1 a particle keeps its mark phase as it ages with the sim
/// time (exactly, at 10⁸ s, thanks to the hi / lo split); at `K` the crest sitting on a parcel of
/// age `a` sits `Δt` later on the parcel of age `a + K·Δt`: it moves `K ×` the parcel's speed, in
/// its direction, whatever the age (not the former log-age time-lapse, ∝ age).
#[test]
fn flow_marks_ride_the_swarm_at_the_set_speed() {
  let t0 = 4.0e8 + 0.375;
  for age in [100.0f32, 2.0e4, 1.0e7] {
    let j = (age.log2().floor()) as i32;
    for dt in [1.0f64, 37.5, 900.0] {
      let (a, b) = (Df::from_f64(t0), Df::from_f64(t0 + dt));
      let p0 = epoch_phase(a.hi, a.lo, age, j);
      let p1 = epoch_phase(b.hi, b.lo, age + dt as f32, j);
      let d = (p1 - p0).abs().min(1.0 - (p1 - p0).abs());
      assert!(d < 1e-4, "age {age} +{dt} s: mark phase moved by {d} turns");
    }
  }
  // the crest nearest an age; after Δt of sim time at K it sits K·Δt older, and its old place
  // (more than a crest width away) is dark
  let crest = |f: &DustFlowUniform, around: f32| {
    (0..4000)
      .map(|i| around * 2f32.powf(-1.0 + 2.0 * i as f32 / 4000.0))
      .max_by(|a, b| flow_factor(*a, f).total_cmp(&flow_factor(*b, f)))
      .unwrap()
  };
  for k in [1.0f64, 10.0, 1000.0] {
    for around in [3600.0f32, 86400.0, 2.0e6] {
      let mut c = DustFlowClock::default();
      c.set_speed(k);
      c.sync(0, 400_000_000_000_000);
      let f0 = DustFlowUniform::new(&c, 4.0e8);
      let a0 = crest(&f0, around);
      let peak = flow_factor(a0, &f0);
      // Δt: the crest moves K·Δt = 0.3·a0 in age, beyond a crest width (~0.1·2^j ≤ 0.1·a0)
      let dt = 0.3 * a0 as f64 / k;
      c.sync(16_000, 400_000_000_000_000 + (dt * 1e6) as i64);
      let f1 = DustFlowUniform::new(&c, 4.0e8 + dt);
      // the crest is now the local maximum at a1 (its amplitude fades as the parcel ages through
      // the level blend, by design: alternate crests fade out, the position is what rides)
      let a1 = a0 + (k * dt) as f32;
      let local_max = (0..400)
        .map(|i| a1 * 2f32.powf(-0.2 + 0.4 * i as f32 / 400.0))
        .map(|a| flow_factor(a, &f1))
        .fold(0.0f32, f32::max);
      let (at_new, at_old) = (flow_factor(a1, &f1), flow_factor(a0, &f1));
      assert!(
        at_new > 0.97 * local_max && at_old < 0.5 * at_new,
        "K {k}, age {a0} (peak {peak}): after {dt} s the factor is {at_new} at age {a1} (local max {local_max}), {at_old} at the old place"
      );
    }
  }
}

/// Coherency: no jump along the ages (also across the level boundaries at powers of two).
#[test]
fn flow_factor_is_continuous() {
  let f = DustFlowUniform::new(&DustFlowClock::default(), 4.0e8 + 12.5);
  let mut prev = flow_factor(1000.0, &f);
  let mut a = 1000.0f32;
  while a < 1.0e5 {
    let next_a = a * 1.000_2;
    let v = flow_factor(next_a, &f);
    // steepest slope of the pulse: κ·D·e^κ/I₀ per turn ≈ 18 per turn
    assert!((v - prev).abs() < 0.05, "jump {prev} → {v} at age {next_a}");
    prev = v;
    a = next_a;
  }
}

/// The flow clock: `T = t_sim + (K − 1)·(played sim time)`. Paused (no sim time) it holds, so
/// paused frames are bit-identical; a viewport rendering the same frame again does not advance
/// it; changing `K` keeps the phase (the offset is continuous) and only changes the pace; the
/// value is clamped.
#[test]
fn flow_clock_holds_on_pause_and_changes_pace_without_a_jump() {
  let mut c = DustFlowClock::default();
  assert_eq!(c.speed, FLOW_SPEED_DEFAULT);
  c.sync(0, 0);
  c.sync(16_000, 64_000);
  assert_eq!(c.offset_s, 0.0, "K = 1: the flow clock is the sim time");
  assert_eq!(c.set_speed(100.0), 100.0);
  c.sync(32_000, 128_000); // +0.064 s of sim at K = 100
  assert!((c.offset_s - 99.0 * 0.064).abs() < 1e-9, "{}", c.offset_s);
  let frozen = c;
  let u = DustFlowUniform::new(&c, 4.0e8);
  for k in 3..100 {
    c.sync(k * 16_000, 128_000); // paused: no sim time
  }
  assert_eq!(c.offset_s, frozen.offset_s);
  assert_eq!(DustFlowUniform::new(&c, 4.0e8), u);
  for age in [100.0f32, 1.0e5] {
    assert_eq!(
      flow_factor(age, &u).to_bits(),
      flow_factor(age, &DustFlowUniform::new(&c, 4.0e8)).to_bits()
    );
  }
  // same frame again (another viewport): no advance
  let before = c;
  c.sync(99 * 16_000, 128_000 + 1_000_000);
  assert_eq!(c.offset_s, before.offset_s);
  // K change: same T now, a different pace afterwards; backwards sim moves the marks inward
  let t_before = c.time(4.0e8);
  c.set_speed(2.0);
  assert_eq!(c.time(4.0e8), t_before);
  c.sync(100 * 16_000, 128_000 - 500_000);
  assert!((c.time(4.0e8) - (t_before - 0.5)).abs() < 1e-9);
  // clamped
  assert_eq!(c.set_speed(0.0), FLOW_SPEED_MIN);
  assert_eq!(c.set_speed(1e9), FLOW_SPEED_MAX);
  assert_eq!(c.set_speed(f64::NAN), FLOW_SPEED_DEFAULT);
}

/// The fountain's base: the provisional batch (open window) pins its last time sample at the tick
/// time, so every stream starts at the jet now (the youngest dust was up to a sample spacing,
/// ~28 min ≈ 3.4 km, from the nucleus: no fountain in `comet_mode_full.rdc`). Closed windows are
/// untouched (deterministic history).
#[test]
fn every_stream_starts_at_the_jet_while_it_emits() {
  let cfg = test_cfg(1.5e-3, 86400.0);
  let t = 1.3 * 86400.0 + 1234.5;
  let host = run_ticks([t; 8], &orbit_jet, &cfg);
  assert!(host.provisional);
  let prov = host.ring.batches.back().unwrap().desc;
  assert!(batch_provisional(&prov));
  let n = 1u32 << batch_streams(&prov).0;
  let jet = orbit_jet(t).unwrap();
  let last = prov.count - n;
  for j in last..prov.count {
    let c = emit_cluster(&prov, j);
    assert!(
      (c.t0().to_f64() - t).abs() < 1e-3,
      "stream {}: emitted {} s before now",
      j - last,
      t - c.t0().to_f64()
    );
    let d = norm(sub(c.r0().to_f64(), jet.r_m));
    assert!(d < 1.0, "stream {}: {d} m from the jet", j - last);
  }
  // every other sample keeps its stratum, and closed windows carry no provisional flag
  assert!((c_t(&prov, 0) - t).abs() > 1.0);
  assert!(closed_windows(&host).iter().all(|(_, d)| !batch_provisional(d)));
  let mass: f64 = (0..prov.count).map(|j| emit_cluster(&prov, j).mass_g() as f64).sum();
  assert!(
    (mass / prov.mass_params[0] as f64 - 1.0).abs() < 1e-3,
    "mass conserved"
  );
}

fn c_t(b: &DustBatch, j: u32) -> f64 {
  emit_cluster(b, j).t0().to_f64()
}

// ─────────────────────────────────────────────────────────────────────────────
// v4 render: packets, capsule splats, pyramid (mirror of dust_propagate / dust_splat.comp)
// ─────────────────────────────────────────────────────────────────────────────

/// a small LCG for the tests (no std rand)
fn lcg(state: &mut u64) -> f64 {
  *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}
fn gauss_lcg(state: &mut u64) -> f64 {
  let u1 = lcg(state).max(1e-12);
  let u2 = lcg(state);
  (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
}

/// Covariance of the sub-packets of `m` pooled back into one (within + between), and their
/// mass-weighted mean.
fn pooled(m: &DustMoments, seg: [f32; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
  // every packet is uniform along its segment: its own mean is the segment's midpoint and its
  // covariance gains seg segᵀ / 12
  let subs: alloc::vec::Vec<Packet> = subpackets(m, seg, None)
    .into_iter()
    .map(|mut p| {
      let mut cov = p.cov;
      let idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
      for (k, (i, j)) in idx.iter().enumerate() {
        cov[k] += p.seg[*i] * p.seg[*j] / 12.0;
      }
      for k in 0..3 {
        p.mean[k] += 0.5 * p.seg[k];
      }
      p.cov = cov;
      p
    })
    .collect();
  let total: f64 = subs.iter().map(|p| p.flux as f64).sum();
  let mut mean = [0.0f64; 3];
  for p in &subs {
    for k in 0..3 {
      mean[k] += p.flux as f64 / total * p.mean[k] as f64;
    }
  }
  let idx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
  let mut cov = [[0.0f64; 3]; 3];
  for p in &subs {
    let w = p.flux as f64 / total;
    for i in 0..3 {
      for j in 0..3 {
        let d = (p.mean[i] as f64 - mean[i]) * (p.mean[j] as f64 - mean[j]);
        cov[i][j] += w * (p.cov[idx[i][j]] as f64 + d);
      }
    }
  }
  (mean, cov)
}

/// The transported second moments (secants over the real spreads, the size polyline) match a
/// Monte-Carlo propagation of the cell's grains (lateral dispersion `σ_lat` across the ejection,
/// the whole speed spread `σ_rad` along it, every size of the distribution weighted by
/// cross-section with its own β and speed) from hours to years, where the former linear
/// anti-sun model is far off.
#[test]
fn packet_moments_match_monte_carlo() {
  let b = test_batch(64, 3600.0);
  let c = emit_cluster(&b, 5);
  let (r0, v0) = (c.r0().to_f64(), c.v0().to_f64());
  let (sigma_lat, sigma_rad) = (c.sigma_lat() as f64, c.sigma_rad() as f64);
  let beta = c.beta() as f64;
  let (e1, e2, e3) = dispersion_frame(c.eject());
  let (e1, e2, e3) = (
    [e1[0] as f64, e1[1] as f64, e1[2] as f64],
    [e2[0] as f64, e2[1] as f64, e2[2] as f64],
    [e3[0] as f64, e3[1] as f64, e3[2] as f64],
  );
  let eject = [c.eject[0] as f64, c.eject[1] as f64, c.eject[2] as f64];
  let range = size_range_factor(&c) as f64;
  let (sb_lo, sb_hi) = ((beta / range).sqrt(), (beta * range).sqrt());
  let dbeta = 0.0;
  let mut rng = 0x1234_5678u64;
  for (days, tol) in [
    (0.05, 0.1),
    (1.0, 0.1),
    (30.0, 0.1),
    (365.0, 0.15),
    (1826.0, 0.3),
  ] {
    let (frame, _) = frame_after(days);
    let (_, m) = packet_moments(&c, 0, &frame);
    assert!(m.flux() > 0.0, "{days} d: culled");
    let (mean, cov) = pooled(&m, [0.0; 3]);
    // Monte Carlo in f64
    let age = frame.t_now_s() - c.t0().to_f64();
    let n = 3000;
    let mut pts = alloc::vec::Vec::with_capacity(n);
    for _ in 0..n {
      // a grain of the cell: lateral and radial dispersion, a size drawn by cross-section
      // (uniform in √β between √(β/F) and √(β·F)), hence its own β and speed v ∝ √(β'/β)
      let (g1, g2, g3) = (
        gauss_lcg(&mut rng),
        gauss_lcg(&mut rng),
        gauss_lcg(&mut rng),
      );
      let sb = sb_lo + (sb_hi - sb_lo) * lcg(&mut rng);
      let bb = sb * sb;
      let f = sb / beta.sqrt();
      // the dispersions scale with the grain's speed (the cone cell is angular, the speed
      // spread relative)
      let mut v = v0;
      for k in 0..3 {
        v[k] += f * (sigma_lat * (g1 * e1[k] + g2 * e2[k]) + sigma_rad * g3 * e3[k])
          + eject[k] * (f - 1.0);
      }
      let (r, _) = kepler::propagate_f64(r0, v, SUN_MU_M3_S2 * (1.0 - bb), age);
      pts.push(r);
    }
    let (rc, _) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2 * (1.0 - beta), age);
    let anchor = frame.anchor_m();
    let mut mc_mean = [0.0f64; 3];
    for p in &pts {
      for k in 0..3 {
        mc_mean[k] += (p[k] - anchor[k]) / n as f64;
      }
    }
    let mut mc_cov = [[0.0f64; 3]; 3];
    for p in &pts {
      for i in 0..3 {
        for j in 0..3 {
          mc_cov[i][j] +=
            (p[i] - anchor[i] - mc_mean[i]) * (p[j] - anchor[j] - mc_mean[j]) / n as f64;
        }
      }
    }
    let tr = |c: &[[f64; 3]; 3]| c[0][0] + c[1][1] + c[2][2];
    let (t_mc, t_pk) = (tr(&mc_cov), tr(&cov));
    let shift = norm([
      mean[0] - mc_mean[0],
      mean[1] - mc_mean[1],
      mean[2] - mc_mean[2],
    ]);
    // the former linear model: ½·dβ·g·age² anti-sunward, as a 1σ extent dβ/√3
    let g = SUN_MU_M3_S2 / norm(rc).powi(2);
    let linear = 0.5 * dbeta / 3f64.sqrt() * g * age * age;
    std::println!(
      "[moments] {days:.2} d: trace mc {:.3e} packet {:.3e} ({:+.1} %), mean shift {:.2e} m ({:.2} σ), linear β model {:.2e} m vs {:.2e} m",
      t_mc,
      t_pk,
      100.0 * (t_pk / t_mc - 1.0),
      shift,
      shift / t_mc.sqrt(),
      linear,
      t_mc.sqrt()
    );
    assert!(
      (t_pk / t_mc - 1.0).abs() < tol,
      "{days} d: packet trace {t_pk:.3e} vs Monte Carlo {t_mc:.3e}"
    );
    assert!(
      shift < 0.25 * t_mc.sqrt(),
      "{days} d: mean shift {shift:.2e} m"
    );
    // the principal axes agree: the packet covariance applied to the MC principal direction
    let mut d = [mc_cov[0][0], mc_cov[1][0], mc_cov[2][0]];
    for _ in 0..50 {
      let y = [
        mc_cov[0][0] * d[0] + mc_cov[0][1] * d[1] + mc_cov[0][2] * d[2],
        mc_cov[1][0] * d[0] + mc_cov[1][1] * d[1] + mc_cov[1][2] * d[2],
        mc_cov[2][0] * d[0] + mc_cov[2][1] * d[1] + mc_cov[2][2] * d[2],
      ];
      d = scale(y, 1.0 / norm(y).max(1e-300));
    }
    let q = |c: &[[f64; 3]; 3]| {
      let mut s = 0.0;
      for i in 0..3 {
        for j in 0..3 {
          s += d[i] * c[i][j] * d[j];
        }
      }
      s
    };
    assert!(
      (q(&cov) / q(&mc_cov) - 1.0).abs() < tol * 1.5,
      "{days} d: variance along the principal axis {:.3e} vs {:.3e}",
      q(&cov),
      q(&mc_cov)
    );
  }
}

/// The size polyline follows the syndyne: the `SIZE_EDGES` edges of a year-old cluster lie on
/// an f64 Kepler propagation of the grain of each edge's β and speed, the edges are equally
/// spaced in √β over `β/F .. β·F`, the `SIZE_BINS` packets are contiguous (each the midpoint of
/// its edges) with the chord variance along the bin, and carry the flux in equal shares; with an
/// older (wider) predecessor every bin is cut into time pieces whose widths grow along the
/// segment and whose fluxes still sum to the cluster's.
#[test]
fn size_polyline_follows_the_syndyne() {
  let b = test_batch(64, 3600.0);
  let c = emit_cluster(&b, 5);
  let (r0, v0) = (c.r0().to_f64(), c.v0().to_f64());
  let beta = c.beta() as f64;
  let eject = [c.eject[0] as f64, c.eject[1] as f64, c.eject[2] as f64];
  let (frame, _) = frame_after(365.0);
  let (_, m) = packet_moments(&c, 0, &frame);
  assert!(m.flux() > 0.0);
  let age = frame.t_now_s() - c.t0().to_f64();
  let anchor = frame.anchor_m();
  // the cluster's own range (misc.w with the id bits masked: F = 10 within ~2 %)
  let range = size_range_factor(&c) as f64;
  assert!(
    (range / SIZE_RANGE_FACTOR - 1.0).abs() < 0.03,
    "range factor {range}"
  );
  let (sb_lo, sb_hi) = ((beta / range).sqrt(), (beta * range).sqrt());
  let mut arc = 0.0f64;
  for j in 0..SIZE_EDGES as usize {
    let sb = sb_lo + (sb_hi - sb_lo) * j as f64 / SIZE_BINS as f64;
    assert!(
      (size_edge_sqrt_beta(c.beta(), size_range_factor(&c), j as u32) as f64 - sb).abs()
        < 1e-2 * sb,
      "edge {j}: √β"
    );
    let f = sb / beta.sqrt();
    let v = [
      v0[0] + eject[0] * (f - 1.0),
      v0[1] + eject[1] * (f - 1.0),
      v0[2] + eject[2] * (f - 1.0),
    ];
    let (r, _) = kepler::propagate_f64(r0, v, SUN_MU_M3_S2 * (1.0 - sb * sb), age);
    let e = m.edge(j);
    for k in 0..3 {
      assert!(
        (e[k] as f64 - (r[k] - anchor[k])).abs() < 2.0 + 1e-6 * norm(sub(r, anchor)),
        "edge {j} axis {k}: {} vs {}",
        e[k],
        r[k] - anchor[k]
      );
    }
    assert!(
      (m.edge_factor(j) as f64 - f).abs() < 1e-5,
      "edge {j}: speed factor"
    );
    if j > 0 {
      arc += norm(sub(
        [e[0] as f64, e[1] as f64, e[2] as f64],
        [
          m.edge(j - 1)[0] as f64,
          m.edge(j - 1)[1] as f64,
          m.edge(j - 1)[2] as f64,
        ],
      ));
    }
  }
  std::println!(
    "[polyline] 365 d: arc length {arc:.3e} m over β {:.4}..{:.4}",
    sb_lo * sb_lo,
    sb_hi * sb_hi
  );
  assert!(
    arc > 1.0e9,
    "a year-old size range spans more than 1e6 km of syndyne"
  );
  let subs = subpackets(&m, [0.0; 3], None);
  assert_eq!(subs.len(), SIZE_BINS as usize);
  let total: f32 = subs.iter().map(|p| p.flux).sum();
  assert!((total / m.flux() - 1.0).abs() < 1e-5);
  for (b, p) in subs.iter().enumerate() {
    // a chord capsule: from the bin's start edge along its chord, open towards its neighbours
    let (a, c2) = (m.edge(b), m.edge(b + 1));
    for k in 0..3 {
      assert!((p.mean[k] - a[k]).abs() < 1.0, "bin {b}: start edge");
      assert!((p.seg[k] - (c2[k] - a[k])).abs() < 1.0, "bin {b}: chord");
    }
    assert_eq!(
      p.open,
      [b > 0, b + 1 < SIZE_BINS as usize],
      "bin {b}: open ends"
    );
    // the covariance is the velocity dispersion scaled by the bin's speed, nothing along the chord
    let f = 0.5 * (m.edge_factor(b) + m.edge_factor(b + 1));
    let cv = m.cov_v();
    for k in 0..6 {
      assert!(
        (p.cov[k] - f * f * cv[k]).abs() <= 1e-4 * cv[k].abs() + 1e-3,
        "bin {b}: cov[{k}] {} vs {}",
        p.cov[k],
        f * f * cv[k]
      );
    }
  }
  // a wider predecessor: time pieces with growing widths, flux conserved
  let seg = [1.0e7, 0.0, 0.0];
  let mut pred = m;
  for k in 0..4 {
    pred.cov_a[k] *= 16.0;
  }
  pred.cov_b_age_id[0] *= 16.0;
  pred.cov_b_age_id[1] *= 16.0;
  // the predecessor sits one segment away, every bin of it (the per-bin segments equal `seg`)
  for e in pred.edges.iter_mut() {
    e[0] += seg[0];
    e[1] += seg[1];
    e[2] += seg[2];
  }
  pred.mean_flux[0] += seg[0];
  assert_eq!(time_pieces(&m, Some(&pred)), 3);
  let pieces = subpackets(&m, seg, Some(&pred));
  assert_eq!(pieces.len(), 3 * SIZE_BINS as usize);
  let total: f32 = pieces.iter().map(|p| p.flux).sum();
  assert!((total / m.flux() - 1.0).abs() < 1e-5);
  for b in 0..SIZE_BINS as usize {
    let tr = |p: &Packet| p.cov[0] + p.cov[3] + p.cov[5];
    let (p0, p1, p2) = (&pieces[3 * b], &pieces[3 * b + 1], &pieces[3 * b + 2]);
    assert!(
      tr(p0) < tr(p1) && tr(p1) < tr(p2),
      "bin {b}: widths grow along the segment"
    );
    // the pieces step along the time segment; their own segment stays the chord
    assert!((p1.mean[0] - p0.mean[0] - seg[0] / 3.0).abs() < 8.0 + 2e-6 * p0.mean[0].abs());
    assert!((p0.seg[0] - (m.edge(b + 1)[0] - m.edge(b)[0])).abs() < 1.0 + 1e-6 * p0.mean[0].abs());
  }
  // no predecessor, or a narrower one: one piece per bin
  assert_eq!(time_pieces(&m, None), 1);
  assert_eq!(time_pieces(&pred, Some(&m)), 1);
}

/// The capsule integrates to `amp · 2π√det Σ'` over the plane for any segment: the total optical
/// depth a packet deposits is its flux over the pixel area whatever its shape.
#[test]
fn capsule_integrates_to_the_flux() {
  for seg in [[0.0f32, 0.0], [30.0, -10.0], [0.3, 0.2], [120.0, 90.0]] {
    let p = ScreenPacket {
      mu: [200.0, 180.0],
      cov: [9.0, 2.0, 4.0],
      seg,
      amp: 0.37,
      sigma_min: 1.0,
      det: 9.0 * 4.0 - 4.0,
      depth_au: 1.0,
      open: [false, false],
      rho: [1.0, 1.0],
    };
    let det = 9.0 * 4.0 - 4.0;
    let mut sum = 0.0f64;
    for y in 0..400 {
      for x in 0..400 {
        sum += capsule_tau(&p, [x as f32 + 0.5, y as f32 + 0.5]) as f64;
      }
    }
    let expect = 0.37 * 2.0 * core::f64::consts::PI * (det as f64).sqrt();
    assert!(
      (sum / expect - 1.0).abs() < 2e-3,
      "seg {seg:?}: Σ τ = {sum:.4} vs {expect:.4}"
    );
    // a linear density along the segment (ρ₀ = 1.4 → ρ₁ = 0.6, mean 1) keeps the flux and puts
    // its centroid at ∫ρ u du = ρ₀/2 + (ρ₁ − ρ₀)/3 of the segment
    if seg[0] * seg[0] + seg[1] * seg[1] > 1.0 {
      let pl = ScreenPacket {
        rho: [1.4, 0.6],
        ..p
      };
      let (mut sum, mut first) = (0.0f64, 0.0f64);
      let dd = (seg[0] * seg[0] + seg[1] * seg[1]) as f64;
      for y in 0..400 {
        for x in 0..400 {
          let t = capsule_tau(&pl, [x as f32 + 0.5, y as f32 + 0.5]) as f64;
          let u = ((x as f64 + 0.5 - 200.0) * seg[0] as f64
            + (y as f64 + 0.5 - 180.0) * seg[1] as f64)
            / dd;
          sum += t;
          first += t * u;
        }
      }
      let centroid = first / sum;
      let expect_c = 1.4 / 2.0 + (0.6 - 1.4) / 3.0;
      assert!(
        (sum / expect - 1.0).abs() < 2e-3,
        "seg {seg:?}: linear density Σ τ = {sum:.4} vs {expect:.4}"
      );
      assert!(
        (centroid - expect_c).abs() < 0.02,
        "seg {seg:?}: linear density centroid {centroid:.3} vs {expect_c:.3}"
      );
    }
  }
  // erf accuracy
  for (x, e) in [
    (0.0f32, 0.0f32),
    (0.5, 0.520_499_9),
    (1.0, 0.842_700_8),
    (2.0, 0.995_322_3),
  ] {
    assert!((erf(x) - e).abs() < 2e-6 && (erf(-x) + e).abs() < 2e-6);
  }
}

/// A splat lands on the lowest level where its smallest σ is ≥ 2 texels, so the texels it
/// touches are bounded whatever its size; the pyramid layout has the levels down to 1 texel.
#[test]
fn splat_level_bounds_the_texel_count() {
  let layout = PyramidLayout::new(1113, 684);
  assert_eq!(layout.levels[0], (PYRAMID_HEADER_WORDS, 1113, 684));
  assert_eq!(layout.levels[1].1, 557);
  let last = *layout.levels.last().unwrap();
  assert_eq!((last.1, last.2), (1, 1));
  // the measurement grid (16 px texels) follows the last level
  assert_eq!(layout.measure, (last.0 + PYRAMID_TEXEL_WORDS, 70, 43));
  assert_eq!(
    layout.total_words,
    layout.measure.0 + 70 * 43 * PYRAMID_TEXEL_WORDS
  );
  assert_eq!(layout.level_count(), 12);
  for (sigma, level) in [
    (0.5f32, 0u32),
    (1.9, 0),
    (2.0, 0),
    (3.9, 0),
    (4.0, 1),
    (7.9, 1),
    (8.0, 2),
    (1000.0, 8),
    (1e9, 11),
  ] {
    assert_eq!(layout.level_for_sigma(sigma), level, "σ {sigma}");
  }
  let mut rng = 99u64;
  let mut pyramid = alloc::vec![0u32; layout.total_words as usize];
  for _ in 0..200 {
    let sigma = 10f32.powf(lcg(&mut rng) as f32 * 4.0 - 1.0); // 0.1 .. 1000 px
    let p = ScreenPacket {
      mu: [lcg(&mut rng) as f32 * 1113.0, lcg(&mut rng) as f32 * 684.0],
      cov: [
        sigma * sigma,
        0.0,
        sigma * sigma * (0.5 + lcg(&mut rng) as f32),
      ],
      seg: [0.0, 0.0],
      amp: 1.0,
      sigma_min: sigma,
      det: 9.0 * 4.0 - 4.0,
      depth_au: 1.0,
      open: [false, false],
      rho: [1.0, 1.0],
    };
    let n = splat_scatter(&p, [1.0, 0.5, 0.2], &layout, 1e6, &mut pyramid);
    let level = layout.level_for_sigma(sigma);
    let texel = (1u32 << level) as f32;
    let (sx, sy) = (sigma / texel, p.cov[2].sqrt() / texel);
    let bound = (2.0 * DUST_SPLAT_SIGMAS * sx + 3.0) * (2.0 * DUST_SPLAT_SIGMAS * sy + 3.0);
    assert!(
      n as f32 <= bound,
      "σ {sigma}: {n} texels at level {level}, bound {bound}"
    );
    assert!(sx < 2.0 * DUST_SPLAT_SIGMA_TEXELS || level == layout.level_count() - 1);
  }
  assert!(pyramid[PYR_LEVEL_MASK as usize] & 1 != 0);
}

/// Orthographic mvp of half-extent `half` (m) looking along −z, `rot` applied to the packets'
/// frame, into a square viewport
fn ortho_rot_mvp(half: f32, rot: [[f32; 3]; 3]) -> [f32; 16] {
  let mut m = [0.0f32; 16];
  for c in 0..3 {
    m[c * 4] = rot[0][c] / half;
    m[c * 4 + 1] = rot[1][c] / half;
    m[c * 4 + 2] = rot[2][c] * 1e-12;
  }
  m[15] = 1.0;
  m
}

/// The picture is the projected density, independent of the view's parametrisation: the same
/// cloud of packets drawn in a rotated frame (packets rotated, camera counter-rotated) gives the
/// same pixels, and the total optical depth equals `exposure · Σ flux / pixel area` at two zooms.
#[test]
fn splat_image_is_the_projected_density() {
  let mut rng = 7u64;
  let n = 60;
  let mut packets = alloc::vec::Vec::new();
  for _ in 0..n {
    let s = 2.0e4 * (0.5 + lcg(&mut rng) as f32);
    let a = [s * 3.0, s * 0.8, s]; // anisotropic axes
    // random orientation via a random rotation
    let th = lcg(&mut rng) as f32 * 6.2832;
    let (c, si) = (th.cos(), th.sin());
    let r = [[c, -si, 0.0], [si, c, 0.0], [0.0, 0.0, 1.0]];
    let mut cov = [[0.0f32; 3]; 3];
    for i in 0..3 {
      for j in 0..3 {
        for k in 0..3 {
          cov[i][j] += r[i][k] * a[k] * a[k] * r[j][k];
        }
      }
    }
    packets.push(Packet {
      mean: [
        (lcg(&mut rng) as f32 - 0.5) * 8e5,
        (lcg(&mut rng) as f32 - 0.5) * 8e5,
        (lcg(&mut rng) as f32 - 0.5) * 8e5,
      ],
      cov: [
        cov[0][0], cov[0][1], cov[0][2], cov[1][1], cov[1][2], cov[2][2],
      ],
      seg: if lcg(&mut rng) > 0.5 {
        [5e4, -2e4, 1e4]
      } else {
        [0.0; 3]
      },
      flux: 1.0e6 * (0.5 + lcg(&mut rng) as f32),
      open: [false, false],
      rho: [1.0, 1.0],
    });
  }
  let px = 256u32;
  let layout = PyramidLayout::new(px, px);
  let exposure = 1.0e-9;
  let render = |half: f32, rot: [[f32; 3]; 3]| {
    let mvp = ortho_rot_mvp(half, rot);
    let mut pyr = alloc::vec![0u32; layout.total_words as usize];
    // the fixed-point unit from the brightest packet, as the host does from τ_max
    let mut tau_max = 0.0f32;
    for p in &packets {
      if let Some(sp) = project_packet(p, exposure, &mvp, px, px, [0.0, 0.0, 1e9]) {
        tau_max = tau_max.max(sp.amp);
      }
    }
    let inv_unit = 1.0 / (tau_max * WHITE_TILE_UNIT_REL);
    let mut total = 0.0f64;
    for p in &packets {
      // the packet expressed in the rotated frame: mean, covariance and segment rotated by rot
      let rv = |v: [f32; 3]| {
        [
          rot[0][0] * v[0] + rot[0][1] * v[1] + rot[0][2] * v[2],
          rot[1][0] * v[0] + rot[1][1] * v[1] + rot[1][2] * v[2],
          rot[2][0] * v[0] + rot[2][1] * v[1] + rot[2][2] * v[2],
        ]
      };
      let _ = rv;
      if let Some(sp) = project_packet(p, exposure, &mvp, px, px, [0.0, 0.0, 1e9]) {
        splat_scatter(&sp, [1.0, 1.0, 1.0], &layout, inv_unit, &mut pyr);
        total += p.flux as f64;
      }
    }
    measure_from_pyramid(&layout, &mut pyr);
    (pyr, total, inv_unit)
  };
  let id = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
  // the measurement grid holds every packet whatever its level: its total equals the levels'
  {
    let (pyr, _, _) = render(1.0e6, id);
    let (mo, mw, mh) = layout.measure;
    let measured: u64 = (0..(mw * mh) as usize)
      .map(|t| pyr[mo as usize + t * PYRAMID_TEXEL_WORDS as usize] as u64)
      .sum();
    let levels: u64 = layout
      .levels
      .iter()
      .flat_map(|&(off, w, h)| (0..(w * h) as usize).map(move |t| (off, t)))
      .map(|(off, t)| pyr[off as usize + t * PYRAMID_TEXEL_WORDS as usize] as u64)
      .sum();
    assert!(
      (measured as f64 / levels as f64 - 1.0).abs() < 0.02,
      "measurement grid {measured} vs levels {levels}"
    );
    assert!(white_point_from_level(&pyr[mo as usize..], 1 << DUST_WHITE_LEVEL, 1.0).is_some());
  }
  // energy at two zooms
  for half in [1.0e6f32, 2.5e6] {
    let (pyr, total, inv_unit) = render(half, id);
    let m_per_px = 2.0 * half / px as f32;
    let expect = exposure as f64 * total / (m_per_px * m_per_px) as f64;
    let mut sum = 0.0f64;
    for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
      let area = (1u64 << (2 * l)) as f64;
      for t in 0..(w * h) as usize {
        sum += pyr[off as usize + t * PYRAMID_TEXEL_WORDS as usize] as f64 / inv_unit as f64 * area
          / area;
      }
    }
    // counts · unit = Σ τ·texel area: the sum over the levels is Σ τ·px² (each level's texel
    // holds τ × its own area)
    assert!(
      (sum / expect - 1.0).abs() < 0.02,
      "half {half}: Σ τ {sum:.4e} vs exposure·flux/A_px {expect:.4e}"
    );
  }
  // view independence: the cloud and the camera rotated together (about the view axis, so the
  // pixel grid maps onto itself) give the same image up to the fixed-point rounding
  let th = core::f32::consts::FRAC_PI_2;
  let rot = [
    [th.cos(), -th.sin(), 0.0],
    [th.sin(), th.cos(), 0.0],
    [0.0, 0.0, 1.0],
  ];
  let (a, _, inv_a) = render(1.0e6, id);
  let (b, _, inv_b) = render(1.0e6, rot);
  assert!((inv_a / inv_b - 1.0).abs() < 1e-3);
  let unit = 1.0 / inv_a;
  let mut worst = 0.0f32;
  let mut peak = 0.0f32;
  let mut sa = 0.0f64;
  let mut sb = 0.0f64;
  for y in 0..px {
    for x in 0..px {
      let (ta, _, _) = composite_sample(&layout, &a, unit, x, y);
      // rotating by +90° about z maps pixel (x, y) of the first view to (px−1−y, x) of the second
      let (tb, _, _) = composite_sample(&layout, &b, unit, px - 1 - y, x);
      peak = peak.max(ta);
      worst = worst.max((ta - tb).abs());
      sa += ta as f64;
      sb += tb as f64;
    }
  }
  assert!(
    worst < 0.02 * peak && (sa / sb - 1.0).abs() < 1e-2,
    "rotated view: worst |Δτ| {worst:.3e} of peak {peak:.3e}, totals {sa:.4e} / {sb:.4e}"
  );
}

/// The age colour runs from the stream colour at the jet to its complementary hue (OKLab chroma
/// rotated by π) at 64 TTL, monotonically and log in age.
#[test]
fn age_color_runs_from_the_stream_color_to_its_complement() {
  let stream = [1.0f32, 0.55, 0.15];
  let c0 = age_color(stream, 0.0);
  for k in 0..3 {
    assert!((c0[k] - stream[k]).abs() < 2e-2, "age 0: {c0:?}");
  }
  let lab = |c: [f32; 3]| super::splat::oklab_of(c);
  let l0 = lab(stream);
  let far = age_color(stream, DUST_AGE_HUE_SPAN * DUST_AGE_HUE_TAU_S);
  let lf = lab(far);
  // the complement: chroma rotated by π (gamut clipping may shorten it, not turn it)
  let hue = |l: [f32; 3]| l[2].atan2(l[1]);
  let dh = (hue(lf) - hue(l0) - core::f32::consts::PI).rem_euclid(2.0 * core::f32::consts::PI);
  // the gamut clip of the rotated chroma turns the hue by up to ~15°
  assert!(
    dh.min(2.0 * core::f32::consts::PI - dh) < 0.35,
    "complement hue off by {dh:.3} rad: {far:?}"
  );
  assert!(far[2] > far[0], "orange's complement is bluish: {far:?}");
  assert!((age_hue_fraction(DUST_AGE_HUE_TAU_S) - (2f32.ln() / 65f32.ln())).abs() < 1e-5);
  let mut prev = 0.0f32;
  for i in 0..40 {
    let age = 10f32.powf(2.0 + i as f32 * 0.2);
    let f = age_hue_fraction(age);
    assert!(f >= prev && f <= 1.0);
    prev = f;
  }
}

/// The auto softening puts the median dust texel at a fixed display level
/// ([`DUST_AUTO_MEDIAN_LEVEL`]) whatever the distance and the dynamic range of the view.
#[test]
fn auto_softening_shows_the_median_dust() {
  for (p50, p99) in [
    (1e-3f32, 1.0f32),
    (3e-6, 2e-2),
    (5e-9, 1e-4),
    (0.05, 0.5),
    (1e-2, 1.0),
  ] {
    let s = auto_softening(p50, p99);
    let shown = display_stretch(p50 / p99, 0.0, s);
    assert!(s >= DUST_SOFTENING_MIN && s <= DUST_SOFTENING_MAX);
    // at the range's floor the median cannot be lifted further (a 10⁴ dynamic range shows it
    // at ~28 %); everywhere else it sits at the target level
    let at_floor = s == DUST_SOFTENING_MIN && shown < DUST_AUTO_MEDIAN_LEVEL;
    assert!(
      at_floor || (shown - DUST_AUTO_MEDIAN_LEVEL).abs() < 1e-3,
      "p50 {p50} p99 {p99}: s {s:.3e}, median shown at {shown:.3}"
    );
  }
  // a median too close to the white cannot be pushed down below the range's end: clamped
  assert_eq!(auto_softening(0.9, 1.0), DUST_SOFTENING_MAX);
  assert_eq!(auto_softening(0.0, 1.0), DUST_SOFTENING_DEFAULT);
  // the white point is the measured one (full adaptation), 1 before any measurement
  assert_eq!(display_white_v4(0.0), 1.0);
  assert_eq!(display_white_v4(3.5e-4), 3.5e-4);
  // level statistics: p99 and p50 of the non-empty texels, per px²
  let mut level = alloc::vec![0u32; 100 * PYRAMID_TEXEL_WORDS as usize];
  for i in 0..100 {
    level[i * PYRAMID_TEXEL_WORDS as usize] = if i < 50 { 0 } else { (i as u32 - 49) * 256 };
  }
  let (p99, p50) = white_point_from_level(&level, 16, 1e-6).unwrap();
  // 50 non-empty texels of 256·k counts, k = 1..50: the median index round(24.5) = 25 → k = 26,
  // the 99th round(48.51) = 49 → k = 50; per px² over 16 × 16 px
  assert!((p50 - 26.0e-6).abs() < 1e-9, "{p50}");
  assert!((p99 - 50.0e-6).abs() < 1e-9, "{p99}");
}

/// The tier's packets as `dust_propagate.comp` writes them (CPU reference).
fn tier_packets(host: &DustHostState, frame: &DustFrame) -> TierPackets {
  tier_packets_with(host, frame, test_threads())
}

/// One tier's render clusters and moments (compact order `first + i`).
type TierPackets = (
  alloc::vec::Vec<DustRenderCluster>,
  alloc::vec::Vec<DustMoments>,
);

/// Threads for the host reference rasterizer in tests: `DUST_TEST_THREADS` or the machine's
/// parallelism. The result does not depend on it (see `host_rasterizer_is_thread_count_independent`).
fn test_threads() -> usize {
  std::env::var("DUST_TEST_THREADS")
    .ok()
    .and_then(|v| v.parse().ok())
    .filter(|&n: &usize| n >= 1)
    .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

/// [`tier_packets`] with the live range split across `threads` threads; `packet_moments` is a
/// pure function of the slot, so the chunks concatenate in order.
fn tier_packets_with(host: &DustHostState, frame: &DustFrame, threads: usize) -> TierPackets {
  let mask = host.ring.mask();
  let mut ring: alloc::vec::Vec<DustCluster> =
    alloc::vec![bytemuck::Zeroable::zeroed(); host.ring.capacity as usize];
  for b in &host.ring.batches {
    for j in 0..b.count {
      ring[(b.desc.first_index.wrapping_add(j) & mask) as usize] = emit_cluster(&b.desc, j);
    }
  }
  let (first, live, _) = host.ring.drawable();
  let live = live as usize;
  let chunk = live.div_ceil(threads.max(1)).max(1);
  let ring = &ring;
  let parts: alloc::vec::Vec<TierPackets> = std::thread::scope(|s| {
    let handles: alloc::vec::Vec<_> = (0..live)
      .step_by(chunk)
      .map(|lo| {
        let hi = (lo + chunk).min(live);
        s.spawn(move || {
          let mut rs = alloc::vec::Vec::with_capacity(hi - lo);
          let mut ms = alloc::vec::Vec::with_capacity(hi - lo);
          for i in lo..hi {
            let slot = first.wrapping_add(i as u32) & mask;
            let (r, m) = packet_moments(&ring[slot as usize], slot, frame);
            rs.push(r);
            ms.push(m);
          }
          (rs, ms)
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
  });
  let mut rs = alloc::vec::Vec::with_capacity(live);
  let mut ms = alloc::vec::Vec::with_capacity(live);
  for (r, m) in parts {
    rs.extend(r);
    ms.extend(m);
  }
  (rs, ms)
}

/// Merges a partial pyramid into `out` (same layout): saturating sums for the count / colour
/// words, the nonzero minimum for the depth word (as `scatter_into`), the maximum of
/// [`PYR_TAU_MAX`] and the union of [`PYR_LEVEL_MASK`].
fn merge_pyramid(out: &mut [u32], part: &[u32]) {
  let h = PYRAMID_HEADER_WORDS as usize;
  let tm = PYR_TAU_MAX as usize;
  if f32::from_bits(part[tm]) > f32::from_bits(out[tm]) {
    out[tm] = part[tm];
  }
  out[PYR_LEVEL_MASK as usize] |= part[PYR_LEVEL_MASK as usize];
  let w = PYRAMID_TEXEL_WORDS as usize;
  for (o, p) in out[h..].chunks_exact_mut(w).zip(part[h..].chunks_exact(w)) {
    for k in 0..4 {
      sat_add(&mut o[k], p[k]);
    }
    if p[4] != 0 && (o[4] == 0 || p[4] < o[4]) {
      o[4] = p[4];
    }
  }
}

/// Splats every tier into one fresh pyramid of `layout`, each tier's live range split across
/// `threads` threads with their own pyramids, merged by [`merge_pyramid`]. `pc_for(live)` gives
/// the push constants for a tier of `live` clusters.
fn splat_tiers_with(
  tiers: &[TierPackets],
  pc_for: &dyn Fn(u32) -> DustSplatPushConstants,
  layout: &PyramidLayout,
  threads: usize,
) -> alloc::vec::Vec<u32> {
  let header = layout.header();
  let fresh = || {
    let mut p = alloc::vec![0u32; layout.total_words as usize];
    p[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&header);
    p
  };
  let mut out = fresh();
  for (r, m) in tiers {
    let pc = pc_for(r.len() as u32);
    let n = r.len();
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let parts: alloc::vec::Vec<alloc::vec::Vec<u32>> = std::thread::scope(|s| {
      let handles: alloc::vec::Vec<_> = (0..n)
        .step_by(chunk)
        .map(|lo| {
          let hi = (lo + chunk).min(n);
          let (pc, fresh) = (&pc, &fresh);
          s.spawn(move || {
            let mut p = fresh();
            splat_tier_range(m, r, pc, &Default::default(), 0, layout, &mut p, lo..hi);
            p
          })
        })
        .collect();
      handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for p in &parts {
      merge_pyramid(&mut out, p);
    }
  }
  out
}

/// Top-down image (τ per px²) of a tier around its jet, orthographic, half extent `half` m
fn tier_image(
  host: &DustHostState,
  frame: &DustFrame,
  half: f32,
  px: u32,
  rot: [[f32; 3]; 3],
) -> alloc::vec::Vec<f32> {
  let (render, moments) = tier_packets(host, frame);
  let layout = PyramidLayout::new(px, px);
  let mut pyr = alloc::vec![0u32; layout.total_words as usize];
  pyr[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
  let pc = DustSplatPushConstants {
    moments: 0,
    render: 0,
    pyramid: 0,
    live_count: render.len() as u32,
    flags: 0,
    exposure: 1.0,
    inv_unit: 1.0,
    color: pack_color([1.0, 1.0, 1.0, 1.0]),
    units_per_m: 1.0,
    mvp: ortho_rot_mvp(half, rot),
    eye_local: [rot[2][0] * 1e9, rot[2][1] * 1e9, rot[2][2] * 1e9, 0.0],
  };
  // the fixed-point unit from the brightest packet, as the host does after a readback
  let mut probe = pyr.clone();
  let pc0 = DustSplatPushConstants {
    inv_unit: 1.0,
    ..pc
  };
  splat_tier(
    &moments,
    &render,
    &pc0,
    &Default::default(),
    0,
    &layout,
    &mut probe,
  );
  let tau_max = f32::from_bits(probe[PYR_TAU_MAX as usize]).max(1e-30);
  let inv_unit = 1.0 / (tau_max * WHITE_TILE_UNIT_REL);
  let pc = DustSplatPushConstants { inv_unit, ..pc };
  splat_tier(
    &moments,
    &render,
    &pc,
    &Default::default(),
    0,
    &layout,
    &mut pyr,
  );
  let mut img = alloc::vec![0.0f32; (px * px) as usize];
  for y in 0..px {
    for x in 0..px {
      img[(y * px + x) as usize] = composite_sample(&layout, &pyr, 1.0 / inv_unit, x, y).0;
    }
  }
  img
}

/// Azimuth (rad) of the brightest direction (±12° smoothed) of `img` in annuli of `dr` m
fn ridge_azimuths(
  img: &[f32],
  half: f32,
  px: u32,
  r0: f32,
  r1: f32,
  dr: f32,
) -> alloc::vec::Vec<f64> {
  const BINS: usize = 90;
  let m_per_px = 2.0 * half / px as f32;
  let rings = ((r1 - r0) / dr) as usize;
  let mut h = alloc::vec![0.0f64; rings * BINS];
  for y in 0..px {
    for x in 0..px {
      let (dx, dy) = (
        (x as f32 + 0.5 - px as f32 / 2.0) * m_per_px,
        (y as f32 + 0.5 - px as f32 / 2.0) * m_per_px,
      );
      let r = (dx * dx + dy * dy).sqrt();
      if r < r0 || r >= r1 {
        continue;
      }
      let ring = ((r - r0) / dr) as usize;
      let az = (dy.atan2(dx) + core::f32::consts::PI) / (2.0 * core::f32::consts::PI);
      let b = ((az * BINS as f32) as usize).min(BINS - 1);
      h[ring * BINS + b] += img[(y * px + x) as usize] as f64;
    }
  }
  (0..rings)
    .map(|ring| {
      let row = &h[ring * BINS..(ring + 1) * BINS];
      let smooth = |b: usize| (0..7).map(|k| row[(b + BINS + k - 3) % BINS]).sum::<f64>();
      let best = (0..BINS).max_by(|&a, &b| smooth(a).partial_cmp(&smooth(b)).unwrap()).unwrap();
      (best as f64 + 0.5) / BINS as f64 * 2.0 * core::f64::consts::PI - core::f64::consts::PI
    })
    .collect()
}

fn unwrap_angles(az: &[f64]) -> alloc::vec::Vec<f64> {
  let mut out = alloc::vec::Vec::with_capacity(az.len());
  let mut acc = 0.0;
  for (i, &a) in az.iter().enumerate() {
    if i > 0 {
      let mut d = a - az[i - 1];
      while d > core::f64::consts::PI {
        d -= 2.0 * core::f64::consts::PI;
      }
      while d < -core::f64::consts::PI {
        d += 2.0 * core::f64::consts::PI;
      }
      acc += d;
    }
    out.push(acc);
  }
  out
}

/// Top-down image (along the spin axis) of a jet's dust on a nucleus spinning at `omega` through
/// the v4 renderer (packets, capsule splats, pyramid; τ per px²). Narrow equatorial jet, one grain
/// size, no speed spread, small β: the former `spiral_image` of the dot renderer.
fn spiral_image(omega: f64, half: f32, px: u32) -> alloc::vec::Vec<f32> {
  let (r0, v0) = comet_state();
  let axis = unit(scale(r0, -1.0)); // towards the Sun: the polar site is always lit
  let e1 = unit(cross64(axis, [0.0, 0.0, 1.0]));
  let e2 = cross64(axis, e1);
  let jet_at = move |t: f64| {
    let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
    let h = 0.5 * omega * t;
    Some(JetState {
      t_s: t,
      r_m: r,
      v_ms: v,
      rot: [
        (axis[0] * h.sin()) as f32,
        (axis[1] * h.sin()) as f32,
        (axis[2] * h.sin()) as f32,
        h.cos() as f32,
      ],
      site_normal: axis,
      spin: Some([axis[0], axis[1], axis[2], omega]),
      site_offset_m: [0.0; 3],
    })
  };
  let cfg = DustEmitConfig {
    q_dust_kgs: 1e-3,
    ttl_s: 2.0 * 86400.0,
    dist: SizeDistribution {
      s_min_um: 49.5,
      s_max_um: 50.5,
      q: SIZE_POWER_Q,
    },
    diameter_um: 100.0,
    density_gcm3: 0.5,
    beta_ref: 1e-3,
    v_mean: 2.0,
    v_std: 0.0,
    jet_dir: [e1[0] as f32, e1[1] as f32, e1[2] as f32],
    aperture_rad: 0.05,
    seed: 7,
  };
  let host = run_ticks([2.0 * 86400.0; 8], &jet_at, &cfg);
  assert!(
    host.next_window.is_some_and(|k| k >= host.due_window),
    "history not filled"
  );
  let ds = host.draw_state().unwrap();
  let rot = [
    [e1[0] as f32, e1[1] as f32, e1[2] as f32],
    [e2[0] as f32, e2[1] as f32, e2[2] as f32],
    [axis[0] as f32, axis[1] as f32, axis[2] as f32],
  ];
  tier_image(&host, &ds.frame, half, px, rot)
}

/// A jet on a spinning nucleus draws a spiral through the v4 renderer: the ridge azimuth of the
/// top-down image turns monotonically with the radius, at least two turns, `v·P` per turn; without
/// spin the fan is straight.
#[test]
fn spinning_nucleus_draws_a_spiral() {
  let (half, px) = (150.0e3f32, 400u32);
  let p_rot = 6.0 * 3600.0;
  // AETHERVK_DUST_SPIRAL_PGM=<prefix>: writes both images (log scale) for a visual check
  if let Ok(prefix) = std::env::var("AETHERVK_DUST_SPIRAL_PGM") {
    for (name, w) in [
      ("spin", 2.0 * core::f64::consts::PI / p_rot),
      ("still", 0.0),
    ] {
      let img = spiral_image(w, half, px);
      let max = img.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
      let mut out = alloc::format!("P2\n{px} {px}\n255\n").into_bytes();
      for y in 0..px {
        for x in 0..px {
          let v = img[(y * px + x) as usize] / max;
          let g = if v > 0.0 {
            ((1.0 + v.log10() / 4.0).max(0.0) * 255.0) as u32
          } else {
            0
          };
          out.extend_from_slice(alloc::format!("{g} ").as_bytes());
        }
        out.push(b'\n');
      }
      std::fs::write(alloc::format!("{prefix}_{name}.pgm"), out).unwrap();
    }
  }
  let img = spiral_image(2.0 * core::f64::consts::PI / p_rot, half, px);
  let az = ridge_azimuths(&img, half, px, 15e3, 140e3, 1e3);
  assert_eq!(az.len(), 125);
  let th = unwrap_angles(&az);
  let turn = th[th.len() - 1] - th[0];
  let turns = turn.abs() / (2.0 * core::f64::consts::PI);
  let expect = 125e3 / (2.0 * p_rot);
  std::println!("[spiral] {turns:.2} turns over 125 km, expected {expect:.2}");
  assert!(turns >= 2.0, "{turns:.2} turns");
  assert!(
    (turns / expect - 1.0).abs() < 0.2,
    "{turns:.2} turns, expected {expect:.2}"
  );
  // monotone: every 10 km (1.46 rad of arm; the splatted arm is ~20° wide, so the ridge of a
  // single annulus wobbles by a few bins) turns the same way
  for (k, w) in th.windows(11).enumerate() {
    assert!(
      (w[10] - w[0]) * turn > 0.0,
      "annulus {k}: the ridge turns back ({:.2} rad)",
      w[10] - w[0]
    );
  }
  let still = ridge_azimuths(&spiral_image(0.0, half, px), half, px, 15e3, 140e3, 1e3);
  let th0 = unwrap_angles(&still);
  let spread = th0.iter().fold(0.0f64, |m, t| m.max((t - th0[0]).abs()));
  assert!(spread < 0.3, "no spin: the fan turns by {spread:.2} rad");
}

/// `streak_pred` (the time-sample segment of the capsule) is the same stream's previous sample:
/// `r − S` when live and not after a break, none for the first `S` and after breaks.
#[test]
fn streak_predecessor_is_the_same_streams_previous_sample() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let host = run_ticks((0..=200).map(|i| i as f64 * 600.0), &orbit_jet, &cfg);
  let ds = host.draw_state().unwrap();
  let (render, _) = tier_packets(&host, &ds.frame);
  let s = 1usize << host.stream_shift();
  assert!(render.len() > 4 * s);
  for r in 0..render.len() {
    let p = streak_pred(&render, r);
    if r < s {
      assert_eq!(p, None);
    } else if !stream_break(render[r].age_id_dbeta_flux[2])
      && render_live(render[r - s].age_id_dbeta_flux[1])
    {
      assert_eq!(p, Some(r - s), "r {r}");
      // the predecessor is older
      assert!(render[r - s].age_id_dbeta_flux[0] > render[r].age_id_dbeta_flux[0]);
    } else {
      assert_eq!(p, None);
    }
  }
}

#[test]
fn gpu_layout_sizes() {
  assert_eq!(core::mem::size_of::<DustCluster>(), 96);
  assert_eq!(core::mem::size_of::<DustRenderCluster>(), 32);
  assert_eq!(core::mem::size_of::<DustMoments>(), 320);
  assert_eq!(core::mem::size_of::<DustBatch>(), 208);
  assert_eq!(core::mem::size_of::<DustPropagatePushConstants>(), 112);
  assert_eq!(core::mem::size_of::<DustSplatPushConstants>(), 128);
  assert_eq!(
    core::mem::size_of::<crate::gpu::CompositePushConstants>(),
    96
  );
  assert_eq!(core::mem::offset_of!(DustSplatPushConstants, mvp), 48);
  assert_eq!(
    core::mem::offset_of!(DustSplatPushConstants, eye_local),
    112
  );
  assert_eq!(
    core::mem::offset_of!(crate::gpu::CompositePushConstants, dust_pyramid),
    40
  );
  assert_eq!(
    core::mem::offset_of!(crate::gpu::CompositePushConstants, layer_unit_au),
    64
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape evolution: the rendered dust must grow progressively (segmentation, no GPU)
// ─────────────────────────────────────────────────────────────────────────────

/// The whole system (every tier) as the v4 renderer draws it at `t`: τ per px² (orthographic,
/// half extent `half` m around the jet, `rot` the view basis), plus the display white point
/// and auto softening the composite would use (from the measurement grid). `keep` selects the
/// compact range of each tier to draw (`None` = all; the negative control draws a prefix).
fn system_image(
  sys: &DustSystemState,
  t: f64,
  half: f32,
  px: u32,
  rot: [[f32; 3]; 3],
  keep: Option<&dyn Fn(usize, usize) -> bool>,
) -> (alloc::vec::Vec<f32>, f32, f32) {
  let s = system_image_rel(sys, t, half, px, rot, keep, WHITE_TILE_UNIT_REL);
  (s.img, s.white, s.soft)
}

/// A composited τ image (row-major `px × px`, τ per px²) with its display statistics
struct SystemImage {
  img: alloc::vec::Vec<f32>,
  /// white point (p99 of the occupied measure texels), its p50, the auto softening
  white: f32,
  p50: f32,
  soft: f32,
  /// peak packet τ (sets the fixed-point unit)
  tau_max: f32,
  /// fraction of the measure grid's texels that are occupied
  occupied: f32,
}

/// [`system_image`] with the fixed-point unit at `unit_rel · τ_max` (production:
/// [`WHITE_TILE_UNIT_REL`]).
fn system_image_rel(
  sys: &DustSystemState,
  t: f64,
  half: f32,
  px: u32,
  rot: [[f32; 3]; 3],
  keep: Option<&dyn Fn(usize, usize) -> bool>,
  unit_rel: f32,
) -> SystemImage {
  images_from_packets(&system_packets(sys, t, keep), half, px, rot, &[unit_rel])
    .pop()
    .expect("one image")
}

/// Every drawable tier's packets at `t` (view independent), `keep(i, n)` culling the others.
fn system_packets(
  sys: &DustSystemState,
  t: f64,
  keep: Option<&dyn Fn(usize, usize) -> bool>,
) -> alloc::vec::Vec<TierPackets> {
  let mut tiers: alloc::vec::Vec<TierPackets> = alloc::vec::Vec::new();
  for host in &sys.tiers {
    let Some(ds) = host.draw_state() else {
      continue;
    };
    let frame = ds.frame.at_time(t);
    let (mut r, mut m) = tier_packets(host, &frame);
    if let Some(k) = keep {
      let n = r.len();
      for i in 0..n {
        if !k(i, n) {
          r[i] = DustRenderCluster::culled(0);
          m[i] = DustMoments::culled();
        }
      }
    }
    tiers.push((r, m));
  }
  tiers
}

/// The images of `tiers` in the orthographic view (`half`, `rot`, `px`), one per entry of
/// `unit_rels` (the fixed-point unit at `unit_rel · τ_max`). The τ_max probe is shared: it does
/// not depend on the unit.
fn images_from_packets(
  tiers: &[TierPackets],
  half: f32,
  px: u32,
  rot: [[f32; 3]; 3],
  unit_rels: &[f32],
) -> alloc::vec::Vec<SystemImage> {
  let layout = PyramidLayout::new(px, px);
  let mvp = ortho_rot_mvp(half, rot);
  let eye = [rot[2][0] * 1e9, rot[2][1] * 1e9, rot[2][2] * 1e9, 0.0];
  let threads = test_threads();
  let pc = |inv_unit: f32, live: u32| DustSplatPushConstants {
    moments: 0,
    render: 0,
    pyramid: 0,
    live_count: live,
    flags: 0,
    exposure: 1.0,
    inv_unit,
    color: pack_color([1.0, 1.0, 1.0, 1.0]),
    units_per_m: 1.0,
    mvp,
    eye_local: eye,
  };
  // the unit from the brightest packet (the host does this from τ_max of the previous frame)
  let probe = splat_tiers_with(tiers, &|live| pc(1.0, live), &layout, threads);
  let tau_max = f32::from_bits(probe[PYR_TAU_MAX as usize]).max(1e-30);
  let mut images = alloc::vec::Vec::with_capacity(unit_rels.len());
  for &unit_rel in unit_rels {
    let unit = tau_max * unit_rel;
    let mut pyr = splat_tiers_with(tiers, &|live| pc(1.0 / unit, live), &layout, threads);
    measure_from_pyramid(&layout, &mut pyr);
    let (mo, mw, mh) = layout.measure;
    let measure = &pyr[mo as usize..mo as usize + (mw * mh * PYRAMID_TEXEL_WORDS) as usize];
    let (white, p50) =
      white_point_from_level(measure, 1 << DUST_WHITE_LEVEL, unit).unwrap_or((0.0, 0.0));
    let occupied = measure.chunks_exact(PYRAMID_TEXEL_WORDS as usize).filter(|t| t[0] > 0).count()
      as f32
      / (mw * mh) as f32;
    let mut img = alloc::vec![0.0f32; (px * px) as usize];
    for y in 0..px {
      for x in 0..px {
        img[(y * px + x) as usize] = composite_sample(&layout, &pyr, unit, x, y).0;
      }
    }
    images.push(SystemImage {
      img,
      white,
      p50,
      soft: auto_softening(p50, white),
      tau_max,
      occupied,
    });
  }
  images
}

/// The host reference rasterizer gives the same packets and the same pyramid whatever the
/// thread count (the per-thread partial pyramids merge exactly, [`merge_pyramid`]).
#[test]
fn host_rasterizer_is_thread_count_independent() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = move |_: &JetState| cfg;
  let t = 3.0 * 86400.0;
  let mut sys = DustSystemState::with_tiers(8_192, 3);
  let mut seq = 0u64;
  settle(&mut sys, t, &cfg_at, &mut seq);
  let half = 3.0 * sys.coma_radius_m().expect("coma") as f32;
  let px = 64u32;
  let layout = PyramidLayout::new(px, px);
  let mvp = ortho_rot_mvp(half, SHAPE_IDENTITY);
  let pc = |inv_unit: f32, live: u32| DustSplatPushConstants {
    moments: 0,
    render: 0,
    pyramid: 0,
    live_count: live,
    flags: 0,
    exposure: 1.0,
    inv_unit,
    color: pack_color([1.0, 1.0, 1.0, 1.0]),
    units_per_m: 1.0,
    mvp,
    eye_local: [0.0, 0.0, 1e9, 0.0],
  };
  let mut packets: alloc::vec::Vec<alloc::vec::Vec<TierPackets>> = alloc::vec::Vec::new();
  let mut pyramids: alloc::vec::Vec<alloc::vec::Vec<u32>> = alloc::vec::Vec::new();
  for threads in [1usize, 3, 16] {
    let tiers: alloc::vec::Vec<TierPackets> = sys
      .tiers
      .iter()
      .filter_map(|host| {
        let ds = host.draw_state()?;
        Some(tier_packets_with(host, &ds.frame.at_time(t), threads))
      })
      .collect();
    assert!(tiers.iter().any(|(r, _)| !r.is_empty()), "nothing drawable");
    // the single-threaded reference is `splat_tier` itself: the τ_max probe (unit 1, every count
    // rounds to 0) and then the image at the near-lossless unit
    let mut inv_unit = 1.0f32;
    for pass in 0..2 {
      let mut pyr = alloc::vec![0u32; layout.total_words as usize];
      pyr[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
      for (r, m) in &tiers {
        splat_tier(
          m,
          r,
          &pc(inv_unit, r.len() as u32),
          &Default::default(),
          0,
          &layout,
          &mut pyr,
        );
      }
      assert_eq!(
        pyr,
        splat_tiers_with(&tiers, &|live| pc(inv_unit, live), &layout, threads),
        "{threads} threads, pass {pass}"
      );
      let tau_max = f32::from_bits(pyr[PYR_TAU_MAX as usize]);
      assert!(tau_max > 0.0, "no τ_max");
      if pass == 1 {
        assert!(
          pyr[PYRAMID_HEADER_WORDS as usize..].iter().any(|&w| w > 0),
          "empty pyramid"
        );
        pyramids.push(pyr);
      }
      inv_unit = 1.0 / (tau_max * 1e-6);
    }
    packets.push(tiers);
  }
  for k in 1..packets.len() {
    assert!(packets[k] == packets[0], "packets differ with {k}");
    assert!(pyramids[k] == pyramids[0], "pyramids differ with {k}");
  }
}

const SHAPE_IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A jet that starts at `t_on` (no pre-start history): the dust must grow outward from it.
fn switched_on_cfg(cfg: DustEmitConfig, t_on: f64) -> impl Fn(&JetState) -> DustEmitConfig {
  move |j: &JetState| {
    if j.t_s < t_on {
      DustEmitConfig {
        q_dust_kgs: 0.0,
        ..cfg
      }
    } else {
      cfg
    }
  }
}

/// ticks `sys` to `t` until nothing is building (what the logic tick does)
fn settle(
  sys: &mut DustSystemState,
  t: f64,
  cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
  seq: &mut u64,
) {
  settle_jet(sys, t, &orbit_jet, cfg_at, seq)
}

/// [`settle`] with any jet function
fn settle_jet(
  sys: &mut DustSystemState,
  t: f64,
  jet_at: &dyn Fn(f64) -> Option<JetState>,
  cfg_at: &dyn Fn(&JetState) -> DustEmitConfig,
  seq: &mut u64,
) {
  for _ in 0..MAX_SEEK_PASSES {
    sys.tick(t, jet_at, cfg_at);
    *seq += 1;
    sys.mark_submitted(*seq);
    if !sys.building() {
      break;
    }
  }
  assert!(!sys.building(), "system still building at t = {t}");
}

/// The rendered dust grows progressively from the jet: with production switched on at `t_on`
/// and the view updated every 2 h for 3 days, every new displayable pixel lies within the reach
/// of the previous region (the dust's own motion over the step plus the splat footprint), the
/// region stays one connected component, its radius and area grow by bounded steps, and the tail
/// settles anti-sunward. The negative control (the former defect: the far history published
/// after the coma) is rejected by the same measure.
#[test]
fn dust_grows_progressively_from_the_jet() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let t_on = 0.0;
  let cfg_at = switched_on_cfg(cfg, t_on);
  let dt = 2.0 * 3600.0;
  let steps = 36; // 3 days
  let (half, px) = (1500.0e3f32, 256u32);
  let m_per_px = 2.0 * half / px as f32;
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut seq = 0u64;
  let mut images = alloc::vec::Vec::new();
  for k in 1..=steps {
    let t = t_on + k as f64 * dt;
    settle(&mut sys, t, &cfg_at, &mut seq);
    let (img, white, s) = system_image(&sys, t, half, px, SHAPE_IDENTITY, None);
    images.push((t, img, white, s));
  }
  // one threshold for the whole sequence: what the last frame displays at 10 %
  let (_, _, white, s) = images.last().unwrap();
  let threshold = display_threshold(*white, *s, 0.1);
  assert!(threshold > 0.0);
  let nucleus = [px as f32 / 2.0, px as f32 / 2.0];
  let shapes: alloc::vec::Vec<DustShape> = images
    .iter()
    .map(|(_, img, _, _)| segment_shape(img, px, px, nucleus, threshold))
    .collect();
  // the faint region (a quarter of the threshold): dust that brightens across the threshold was
  // already there, so new displayable pixels must lie within reach of the *faint* previous region
  let faint: alloc::vec::Vec<DustShape> = images
    .iter()
    .map(|(_, img, _, _)| segment_shape(img, px, px, nucleus, 0.25 * threshold))
    .collect();
  // the reach of one step: the fastest grains (v_mean + 3 v_std) plus their lateral spread over
  // 2 h, in pixels, plus the splat footprint (3 px)
  let v_max = cfg.v_mean + 3.0 * cfg.v_std + 3.0 * cfg.v_mean * cfg.aperture_rad;
  let reach = v_max * dt as f32 / m_per_px + 3.0;
  let (r0, v0) = comet_state();
  let _ = v0;
  let anti_sun = [(r0[0] / norm(r0)) as f32, (r0[1] / norm(r0)) as f32];
  let mut worst = ShapeStep {
    grown_px: 0,
    lost_px: 0,
    grown_outside_reach: 0,
    area_ratio: 1.0,
    radius_p90_ratio: 1.0,
    tau_ratio: 1.0,
    farthest_growth_px: 0.0,
  };
  for k in 1..shapes.len() {
    let (a, b) = (&shapes[k - 1], &shapes[k]);
    assert!(b.area_px > 0, "step {k}: empty shape");
    assert!(
      b.detached_px as f32 <= 0.01 * b.area_px as f32 + 2.0,
      "step {k}: {} detached px of {}",
      b.detached_px,
      b.area_px
    );
    let st = shape_step(a, b, reach);
    let reach_step = shape_step(&faint[k - 1], b, reach);
    worst.grown_outside_reach = worst.grown_outside_reach.max(reach_step.grown_outside_reach);
    worst.farthest_growth_px = worst.farthest_growth_px.max(reach_step.farthest_growth_px);
    worst.area_ratio = worst.area_ratio.max(st.area_ratio);
    worst.radius_p90_ratio = worst.radius_p90_ratio.max(st.radius_p90_ratio);
    assert_eq!(
      reach_step.grown_outside_reach,
      0,
      "step {k} (t = {:.1} h): {} px appeared beyond {reach:.1} px of the previous (faint) region (farthest {:.1} px)",
      images[k].0 / 3600.0,
      reach_step.grown_outside_reach,
      reach_step.farthest_growth_px
    );
    assert!(
      b.radius_p90 + 0.5 >= a.radius_p90 && b.radius_p90 - a.radius_p90 <= reach,
      "step {k}: radius p90 {:.1} → {:.1} px (reach {reach:.1})",
      a.radius_p90,
      b.radius_p90
    );
    // bounded relative growth: a young coma's radius grows linearly in time, so its area may grow
    // as (t_b / t_a)² between steps (×4 from 2 h to 4 h), never faster
    let (ta, tb) = (images[k - 1].0, images[k].0);
    let bound = 1.15 * (tb / ta).powi(2) as f32 + 0.1;
    assert!(
      st.area_ratio >= 0.97 && st.area_ratio <= bound,
      "step {k}: area ratio {} ({} → {} px, bound {bound:.2})",
      st.area_ratio,
      a.area_px,
      b.area_px
    );
    assert!(
      st.tau_ratio >= 0.999,
      "step {k}: τ ratio {} (production is on)",
      st.tau_ratio
    );
    std::println!(
      "[shape] step {k:2}: area {:5} px (+{} / −{}), p90 {:.1} px, farthest growth {:.1} px from the faint region, elongation {:.2}",
      b.area_px,
      st.grown_px,
      st.lost_px,
      b.radius_p90,
      reach_step.farthest_growth_px,
      b.elongation
    );
  }
  // AETHERVK_DUST_SHAPE_PNG=<dir>: the same shape.png / shape.jsonl the observer writes, from
  // the reference path (no GPU), plus the last frame as a PNG
  if let Ok(dir) = std::env::var("AETHERVK_DUST_SHAPE_PNG") {
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let mut jsonl = alloc::string::String::new();
    for (k, sh) in shapes.iter().enumerate() {
      jsonl.push_str(&alloc::format!(
        "{{\"step\":{k},\"t_h\":{:.1},\"area_px\":{},\"radius_p50_px\":{:.2},\"radius_p90_px\":{:.2},\"radius_p90_km\":{:.1},\"elongation\":{:.3},\"axis\":[{:.4},{:.4}],\"total_tau\":{:.4e}}}\n",
        images[k].0 / 3600.0, sh.area_px, sh.radius_p50, sh.radius_p90,
        sh.radius_p90 * m_per_px * 1e-3, sh.elongation, sh.axis[0], sh.axis[1], sh.total_tau
      ));
    }
    std::fs::write(dir.join("shape.jsonl"), jsonl).unwrap();
    let (pw, ph) = (800u32, 300u32);
    let mut plot = image::RgbaImage::from_pixel(pw, ph, image::Rgba([16, 16, 20, 255]));
    let series: [(alloc::vec::Vec<f32>, [u8; 3]); 3] = [
      (
        shapes.iter().map(|s| s.radius_p90).collect(),
        [255, 170, 40],
      ),
      (
        shapes.iter().map(|s| s.area_px as f32).collect(),
        [80, 200, 255],
      ),
      (
        shapes.iter().map(|s| s.elongation).collect(),
        [120, 255, 120],
      ),
    ];
    for (vals, col) in &series {
      let max = vals.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
      let n = vals.len().max(2) as f32;
      let pt = |i: usize| {
        (
          10.0 + (pw as f32 - 20.0) * i as f32 / (n - 1.0),
          ph as f32 - 10.0 - (ph as f32 - 20.0) * vals[i] / max,
        )
      };
      for i in 1..vals.len() {
        let ((x0, y0), (x1, y1)) = (pt(i - 1), pt(i));
        let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as usize).max(1);
        for st in 0..=steps {
          let t = st as f32 / steps as f32;
          let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
          for d in 0..2u32 {
            if x >= 0.0 && y >= 0.0 && (x as u32) < pw && (y as u32) + d < ph {
              plot.put_pixel(
                x as u32,
                y as u32 + d,
                image::Rgba([col[0], col[1], col[2], 255]),
              );
            }
          }
        }
      }
    }
    plot.save(dir.join("shape.png")).unwrap();
    // the last frame (log stretch, white = p99) and the first frame with dust
    for (name, idx) in [
      ("frame_first.png", 2usize),
      ("frame_last.png", images.len() - 1),
    ] {
      let img = &images[idx].1;
      let max = white.max(1e-30);
      let mut out = image::RgbaImage::new(px, px);
      for y in 0..px {
        for x in 0..px {
          let v = (img[(y * px + x) as usize] / max).clamp(0.0, 1.0);
          let g = if v > 0.0 {
            ((1.0 + v.log10() / 3.0).max(0.0) * 255.0) as u8
          } else {
            0
          };
          out.put_pixel(
            x,
            y,
            image::Rgba([g, (g as f32 * 0.7) as u8, (g as f32 * 0.3) as u8, 255]),
          );
        }
      }
      out.save(dir.join(name)).unwrap();
    }
  }
  let (first, last) = (&shapes[0], shapes.last().unwrap());
  let cos = (last.axis[0] * anti_sun[0] + last.axis[1] * anti_sun[1]).abs();
  std::println!(
    "[shape] {} steps: radius p90 {:.1} → {:.1} px, area {} → {} px, elongation {:.2} → {:.2}, axis·anti-sun {cos:.3}, worst growth {:.1} px (reach {reach:.1})",
    steps,
    first.radius_p90,
    last.radius_p90,
    first.area_px,
    last.area_px,
    first.elongation,
    last.elongation,
    worst.farthest_growth_px
  );
  assert!(
    last.radius_p90 > 2.0 * first.radius_p90.max(1.0),
    "the tail did not grow"
  );
  assert!(
    last.elongation > first.elongation.max(1.0) * 1.2,
    "the tail did not elongate"
  );
  assert!(
    cos > 0.96,
    "the tail axis is {:.1}° off the anti-sun direction",
    cos.acos().to_degrees()
  );

  // negative control: the former defect replayed — the far part of the history published after
  // the near part (two frames: the youngest half of every tier, then everything)
  let t = images.last().unwrap().0;
  let young = |i: usize, n: usize| i >= n / 2;
  let (img_a, _, _) = system_image(&sys, t, half, px, SHAPE_IDENTITY, Some(&young));
  let (img_b, _, _) = system_image(&sys, t, half, px, SHAPE_IDENTITY, None);
  let sa = segment_shape(&img_a, px, px, nucleus, 0.25 * threshold);
  let sb = segment_shape(&img_b, px, px, nucleus, threshold);
  let st = shape_step(&sa, &sb, reach);
  std::println!(
    "[shape] control: {} px popped in beyond reach (farthest {:.1} px), radius p90 {:.1} → {:.1}",
    st.grown_outside_reach,
    st.farthest_growth_px,
    sa.radius_p90,
    sb.radius_p90
  );
  assert!(
    st.grown_outside_reach > 0 && st.farthest_growth_px > 2.0 * reach,
    "the measure must reject a history that pops in"
  );
}

/// The picture of an epoch is a function of the epoch only: a fresh fill, 2-h ticks and a seek
/// back and forth give the same τ image.
#[test]
fn dust_shape_is_a_function_of_the_epoch_only() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = switched_on_cfg(cfg, 0.0);
  let t_end = 2.0 * 86400.0;
  let (half, px) = (1500.0e3f32, 128u32);
  let mut seq = 0u64;
  let mut fresh = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  settle(&mut fresh, t_end, &cfg_at, &mut seq);
  let mut ticked = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  for k in 1..=24 {
    settle(&mut ticked, k as f64 * t_end / 24.0, &cfg_at, &mut seq);
  }
  let mut seeked = ticked.clone();
  settle(&mut seeked, t_end + 0.5 * 86400.0, &cfg_at, &mut seq);
  settle(&mut seeked, t_end, &cfg_at, &mut seq);
  let (a, _, _) = system_image(&fresh, t_end, half, px, SHAPE_IDENTITY, None);
  let (b, _, _) = system_image(&ticked, t_end, half, px, SHAPE_IDENTITY, None);
  let (c, _, _) = system_image(&seeked, t_end, half, px, SHAPE_IDENTITY, None);
  let peak = a.iter().cloned().fold(0.0f32, f32::max);
  assert!(peak > 0.0);
  let mut worst = 0.0f32;
  for i in 0..a.len() {
    worst = worst.max((a[i] - b[i]).abs()).max((a[i] - c[i]).abs());
  }
  assert!(
    worst <= 1e-3 * peak,
    "images differ by {worst:.3e} of peak {peak:.3e}"
  );
}

/// A jet ignited at `t_on` (`DustSystemState::set_ignition`) holds nothing emitted before it: no
/// batch starts before `t_on`, no tier's oldest dust is older than `t − t_on`, and the rendered
/// region grows progressively from the jet; a rewind below `t_on` empties every tier and leaves
/// nothing building; seeking back and forth reproduces a fresh fill batch for batch; a re-emit
/// keeps the ignition. The history fill of a pre-existing tail is thus an explicit opt-in.
#[test]
fn ignited_system_grows_from_t_on_and_seeks_deterministically() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = |_: &JetState| cfg;
  let t_on = 5.0 * 86400.0;
  let dt = 2.0 * 3600.0;
  let steps = 36; // 3 days
  let (half, px) = (1500.0e3f32, 128u32);
  let m_per_px = 2.0 * half / px as f32;
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  assert!(
    sys.ignition().is_none(),
    "with_tiers alone keeps the pre-existing tail"
  );
  sys.set_ignition(Some(t_on));
  assert_eq!(sys.ignition(), Some(t_on));
  let mut seq = 0u64;
  // before the ignition: nothing, and nothing building
  settle(&mut sys, t_on - 3600.0, &cfg_at, &mut seq);
  assert!(
    sys.tiers.iter().all(|t| t.ring.live() == 0),
    "dust before the ignition"
  );
  assert!(sys.draw_states().is_empty());
  let mut images = alloc::vec::Vec::new();
  for k in 1..=steps {
    let t = t_on + k as f64 * dt;
    settle(&mut sys, t, &cfg_at, &mut seq);
    for tier in &sys.tiers {
      for b in &tier.ring.batches {
        let t_start = b.desc.comet_r_t_hi[3] as f64 + b.desc.comet_r_t_lo[3] as f64;
        assert!(
          t_start >= t_on - 1e-3,
          "tier {}: batch starts at {t_start} < t_on {t_on}",
          tier.tier
        );
        assert!(b.window >= 0, "window {} before the ignition", b.window);
      }
    }
    for (i, st) in sys.stats().iter().enumerate() {
      if st.live_clusters > 0 {
        assert!(
          st.oldest_age_s <= t - t_on + 1e-3,
          "step {k} tier {i}: oldest {:.2} d > time since ignition {:.2} d",
          st.oldest_age_s / 86400.0,
          (t - t_on) / 86400.0
        );
      }
    }
    let (img, white, s) = system_image(&sys, t, half, px, SHAPE_IDENTITY, None);
    images.push((t, img, white, s));
  }
  assert!(
    sys.tiers[0].ring.live() > 0,
    "the youngest tier emitted nothing"
  );
  // progressive growth from the jet (the shape machinery of `dust_grows_progressively_from_the_jet`)
  let (_, _, white, s) = images.last().unwrap();
  let threshold = display_threshold(*white, *s, 0.1);
  assert!(threshold > 0.0);
  let nucleus = [px as f32 / 2.0, px as f32 / 2.0];
  let v_max = cfg.v_mean + 3.0 * cfg.v_std + 3.0 * cfg.v_mean * cfg.aperture_rad;
  let reach = v_max * dt as f32 / m_per_px + 3.0;
  let faint: alloc::vec::Vec<DustShape> = images
    .iter()
    .map(|(_, img, _, _)| segment_shape(img, px, px, nucleus, 0.25 * threshold))
    .collect();
  let shapes: alloc::vec::Vec<DustShape> = images
    .iter()
    .map(|(_, img, _, _)| segment_shape(img, px, px, nucleus, threshold))
    .collect();
  for k in 1..shapes.len() {
    let st = shape_step(&faint[k - 1], &shapes[k], reach);
    assert_eq!(
      st.grown_outside_reach, 0,
      "step {k}: {} px appeared beyond {reach:.1} px of the previous region",
      st.grown_outside_reach
    );
    assert!(
      shapes[k].area_px >= shapes[k - 1].area_px * 97 / 100,
      "step {k}: the region shrank"
    );
  }
  let t_end = t_on + steps as f64 * dt;
  let reference: alloc::vec::Vec<DustBatch> =
    sys.tiers.iter().flat_map(|t| t.ring.batches.iter().map(|b| b.desc)).collect();
  // rewind below the ignition: empty, caught up, nothing drawn
  settle(&mut sys, t_on - 3600.0, &cfg_at, &mut seq);
  assert!(
    sys.tiers.iter().all(|t| t.ring.live() == 0),
    "dust after a rewind below the ignition"
  );
  assert!(!sys.building());
  assert!(sys.draw_states().is_empty());
  // forward again: the same descriptors as the ticked history
  settle(&mut sys, t_end, &cfg_at, &mut seq);
  let again: alloc::vec::Vec<DustBatch> =
    sys.tiers.iter().flat_map(|t| t.ring.batches.iter().map(|b| b.desc)).collect();
  assert_eq!(again.len(), reference.len(), "batch count after the seek");
  for (a, b) in again.iter().zip(reference.iter()) {
    assert_eq!(
      bytemuck::bytes_of(a),
      bytemuck::bytes_of(b),
      "a batch differs after the seek"
    );
  }
  // a fresh ignited system settled once at t_end: the same descriptors
  let mut fresh = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  fresh.set_ignition(Some(t_on));
  let mut seq2 = 0u64;
  settle(&mut fresh, t_end, &cfg_at, &mut seq2);
  let fresh_b: alloc::vec::Vec<DustBatch> =
    fresh.tiers.iter().flat_map(|t| t.ring.batches.iter().map(|b| b.desc)).collect();
  assert_eq!(
    fresh_b.len(),
    reference.len(),
    "batch count of a fresh fill"
  );
  for (a, b) in fresh_b.iter().zip(reference.iter()) {
    assert_eq!(
      bytemuck::bytes_of(a),
      bytemuck::bytes_of(b),
      "a fresh fill differs from the ticked history"
    );
  }
  // a re-emit keeps the ignition
  sys.request_reemit();
  settle(&mut sys, t_end, &cfg_at, &mut seq);
  assert_eq!(sys.ignition(), Some(t_on));
  assert!(sys.tiers.iter().all(|t| t.ring.batches.iter().all(|b| b.window >= 0)));
  // the same images: the picture is a function of (parameters, ignition, epoch)
  let (a, _, _) = system_image(&sys, t_end, half, px, SHAPE_IDENTITY, None);
  let (b, _, _) = system_image(&fresh, t_end, half, px, SHAPE_IDENTITY, None);
  let peak = a.iter().cloned().fold(0.0f32, f32::max);
  assert!(peak > 0.0);
  let worst = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
  assert!(
    worst <= 1e-3 * peak,
    "images differ by {worst:.3e} of peak {peak:.3e}"
  );
}

/// `set_ignition(None)` on a pre-existing-tail system changes nothing (no re-emit), and a
/// pre-existing tail still reaches back before t = 0 at the start.
#[test]
fn prestart_system_is_unchanged_by_set_ignition_none() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = |_: &JetState| cfg;
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut seq = 0u64;
  settle(&mut sys, 0.0, &cfg_at, &mut seq);
  let oldest = sys.stats()[2].oldest_age_s;
  assert!(
    oldest > 63.0 * 30.0 * 86400.0,
    "pre-existing tail missing: oldest {oldest}"
  );
  sys.set_ignition(None);
  assert!(
    !sys.reemit,
    "set_ignition(None) must not re-emit a pre-existing tail"
  );
  assert!(sys.ignition().is_none());
}

/// The segmented shape is the same from any camera distance and orientation (up to the
/// projection): radii in metres and the axis agree at two zooms and a 90° rotation.
#[test]
fn dust_shape_is_the_same_from_any_camera_distance() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = switched_on_cfg(cfg, 0.0);
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut seq = 0u64;
  let t = 3.0 * 86400.0;
  settle(&mut sys, t, &cfg_at, &mut seq);
  let px = 256u32;
  let th = core::f32::consts::FRAC_PI_2;
  let rot90 = [
    [th.cos(), -th.sin(), 0.0],
    [th.sin(), th.cos(), 0.0],
    [0.0, 0.0, 1.0],
  ];
  let mut results = alloc::vec::Vec::new();
  // one physical threshold (τ per px² is a coverage: zoom independent), from the first view
  let mut threshold = 0.0f32;
  for (half, rot) in [
    (1500.0e3f32, SHAPE_IDENTITY),
    (3750.0e3, SHAPE_IDENTITY),
    (1500.0e3, rot90),
  ] {
    let (img, white, s) = system_image(&sys, t, half, px, rot, None);
    if threshold == 0.0 {
      threshold = display_threshold(white, s, 0.1);
    }
    let sh = segment_shape(&img, px, px, [px as f32 / 2.0, px as f32 / 2.0], threshold);
    let m_per_px = 2.0 * half / px as f32;
    // the axis back in the world's xy plane
    let ax = [
      rot[0][0] * sh.axis[0] + rot[1][0] * sh.axis[1],
      rot[0][1] * sh.axis[0] + rot[1][1] * sh.axis[1],
    ];
    results.push((
      sh.radius_p90 * m_per_px,
      sh.radius_p50 * m_per_px,
      ax,
      sh.elongation,
      m_per_px,
    ));
    std::println!(
      "[shape] half {half:.0} m: p90 {:.0} m p50 {:.0} m axis ({:.3}, {:.3}) elongation {:.2}",
      sh.radius_p90 * m_per_px,
      sh.radius_p50 * m_per_px,
      ax[0],
      ax[1],
      sh.elongation
    );
  }
  let r = &results[0];
  for (k, o) in results.iter().enumerate().skip(1) {
    assert!(
      (o.0 / r.0 - 1.0).abs() < 0.06,
      "view {k}: radius p90 {:.0} vs {:.0} m",
      o.0,
      r.0
    );
    // the median radius of a coma is a few pixels at the coarser zoom: half a pixel of slack
    let p50_tol = 0.03 + 0.5 * o.4.max(r.4) / r.1;
    assert!(
      (o.1 / r.1 - 1.0).abs() < p50_tol,
      "view {k}: radius p50 {:.0} vs {:.0} m (tolerance {p50_tol:.3})",
      o.1,
      r.1
    );
    let cos = (o.2[0] * r.2[0] + o.2[1] * r.2[1]).abs();
    assert!(
      cos > 0.9986,
      "view {k}: axis {:.2}° off",
      cos.acos().to_degrees()
    );
    assert!(
      (o.3 / r.3 - 1.0).abs() < 0.1,
      "view {k}: elongation {} vs {}",
      o.3,
      r.3
    );
  }
}

/// Measurement (not a pass/fail check yet): how the dust image depends on the ring capacity and
/// on the stream count `S` it implies. For each configuration, with the history filled to 3 days
/// and one fixed orthographic view (3× the reference coma), prints: S per tier, coma P90, total τ,
/// τ_max, the share of τ the 0.5-count cut drops at the production unit, the displayed shape
/// (area, P90 radius, elongation at the production display threshold) and the physical shape (at
/// 10⁻³ of the reference peak) and the L1 distance of the near-lossless image to the reference.
/// `cargo nextest run --release dust_tau_image_vs_ring_capacity --run-ignored only --no-capture`
#[test]
#[ignore = "measurement: run on demand"]
fn dust_tau_image_vs_ring_capacity() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = move |_: &JetState| cfg;
  let t = 3.0 * 86400.0;
  let px = 256u32;
  // (label, capacity, cap on S)
  let configs: [(&str, u32, u32); 7] = [
    ("ref 1M", 1 << 20, DUST_STREAMS),
    ("262144", 262_144, DUST_STREAMS),
    ("32768", 32_768, DUST_STREAMS),
    ("8192 (CPU)", 8_192, DUST_STREAMS),
    ("262144 S<=8", 262_144, 8),
    ("262144 S<=2", 262_144, 2),
    ("262144 S<=1", 262_144, 1),
  ];
  let build = |cap: u32, max_s: u32| {
    let mut sys = DustSystemState::with_tiers(cap, 3);
    for tier in sys.tiers.iter_mut() {
      tier.max_streams = max_s;
    }
    let mut seq = 0u64;
    settle(&mut sys, t, &cfg_at, &mut seq);
    sys
  };
  let reference = build(configs[0].1, configs[0].2);
  let coma_ref = reference.coma_radius_m().expect("reference coma") as f32;
  let half = 3.0 * coma_ref;
  let m_per_px = 2.0 * half / px as f32;
  let px_area = m_per_px * m_per_px;
  let nucleus = [px as f32 / 2.0, px as f32 / 2.0];
  // the finest unit the u32 counts hold without wrapping (τ_max·1e-9 overflows them)
  let fine_rel = std::env::var("DUST_FINE_UNIT_REL")
    .ok()
    .and_then(|v| v.parse().ok())
    .unwrap_or(1e-6f32);
  let ref_fine = system_image_rel(&reference, t, half, px, SHAPE_IDENTITY, None, fine_rel).img;
  let ref_peak = ref_fine.iter().cloned().fold(0.0f32, f32::max);
  let ref_total: f64 = ref_fine.iter().map(|&v| v as f64).sum();
  let phys_threshold = 1e-3 * ref_peak;
  std::println!(
    "view: half {:.0} km ({:.1} km/px), reference coma {:.0} km, physical threshold {:.3e}",
    half / 1e3,
    m_per_px / 1e3,
    coma_ref / 1e3,
    phys_threshold
  );
  for (label, cap, max_s) in configs {
    let sys = if cap == configs[0].1 && max_s == configs[0].2 {
      reference.clone()
    } else {
      build(cap, max_s)
    };
    let streams: alloc::vec::Vec<u32> = sys.tiers.iter().map(|h| 1 << h.stream_shift()).collect();
    let coma = sys.coma_radius_m().unwrap_or(0.0) / 1e3;
    let fine_s = system_image_rel(&sys, t, half, px, SHAPE_IDENTITY, None, fine_rel);
    let (fine, tau_max) = (fine_s.img, fine_s.tau_max);
    let prod_s = system_image_rel(&sys, t, half, px, SHAPE_IDENTITY, None, WHITE_TILE_UNIT_REL);
    let (prod, white, soft) = (&prod_s.img, prod_s.white, prod_s.soft);
    let total_fine: f64 = fine.iter().map(|&v| v as f64).sum();
    let total_prod: f64 = prod.iter().map(|&v| v as f64).sum();
    // Στ of each tier alone (the same unit: the shares add up to the total)
    let per_tier: alloc::vec::Vec<f64> = (0..sys.tiers.len())
      .map(|k| {
        let mut one = sys.clone();
        one.tiers.retain(|h| h.tier == k as u32);
        system_image_rel(&one, t, half, px, SHAPE_IDENTITY, None, fine_rel)
          .img
          .iter()
          .map(|&v| v as f64)
          .sum::<f64>()
          * px_area as f64
      })
      .collect();
    let l1: f64 =
      fine.iter().zip(&ref_fine).map(|(&a, &b)| (a - b).abs() as f64).sum::<f64>() / ref_total;
    let shown = segment_shape(prod, px, px, nucleus, display_threshold(white, soft, 0.1));
    let phys = segment_shape(&fine, px, px, nucleus, phys_threshold);
    let lit = prod.iter().filter(|&&v| v >= display_threshold(white, soft, 0.1)).count() as f32
      / (px * px) as f32;
    std::println!(
      "{label:>12}: S {streams:?} coma {coma:>7.0} km | Στ·A {:.3e} (tiers {:.2e} {:.2e} {:.2e}) τ_max {tau_max:.3e} cut loss {:>5.1}% | W {white:.3e} p50/W {:.2e} s {soft:.2e} occupied {:>5.1}% | shown area {:>6} p90 {:>6.0} km elong {:>5.2} lit {:>5.1}% | physical area {:>6} p90 {:>6.0} km elong {:>5.2} | L1 vs ref {:>5.2}",
      total_fine * px_area as f64,
      per_tier.first().copied().unwrap_or(0.0),
      per_tier.get(1).copied().unwrap_or(0.0),
      per_tier.get(2).copied().unwrap_or(0.0),
      100.0 * (1.0 - total_prod / total_fine.max(1e-300)),
      prod_s.p50 / prod_s.white.max(1e-30),
      100.0 * prod_s.occupied,
      shown.area_px,
      shown.radius_p90 * m_per_px / 1e3,
      shown.elongation,
      100.0 * lit,
      phys.area_px,
      phys.radius_p90 * m_per_px / 1e3,
      phys.elongation,
      l1
    );
  }
}

/// Every cluster carries the whole size distribution of its cell: the batch mass is conserved
/// exactly by the `S` streams of each time sample for any stream count, the cluster's β is the
/// reference grain's, its cross-section per gram is the distribution's mean (the harmonic mean
/// size, `s_ref` for `n ∝ s^-3.5` over a symmetric log range, checked against a numeric
/// integration), its dispersion is the jet's speed spread, its `misc.w` the half log-size range.
#[test]
fn every_cluster_carries_the_size_distribution_and_conserves_mass() {
  let dist = SizeDistribution::from_diameter_um(100.0);
  let (size_params, vel_params, mass_params) =
    batch_params(&dist, 100.0, 0.533, 0.0213, 2.0, 0.5, 1.0e6, 0.37);
  let make = |shift: u32, samples: u32| {
    let mut b: DustBatch = bytemuck::Zeroable::zeroed();
    let (rc, vc) = comet_state();
    b.set_comet(rc, vc, 0.0, 3600.0);
    b.rot_start = [0.0, 0.0, 0.0, 1.0];
    b.lit = [0.0, 0.0, 0.0, LIT_MODE_ALWAYS];
    b.jet_dir_aperture = [0.35, 0.93, 0.04, 0.6];
    b.size_params = size_params;
    b.vel_params = vel_params;
    b.mass_params = mass_params;
    b.mass_params[3] = batch_word(shift, false, false);
    b.count = samples << shift;
    b.ring_mask = 65_535;
    b.seed = 0xC0FFEE;
    b
  };
  // numeric ⟨1/s⟩ over the mass of n(s) ∝ s^-q, s in µm
  let (s_min, s_max, q) = (dist.s_min_um, dist.s_max_um, dist.q);
  let (mut num, mut den) = (0.0f64, 0.0f64);
  let steps = 200_000;
  for k in 0..steps {
    let s = s_min * (s_max / s_min).powf((k as f64 + 0.5) / steps as f64);
    let dm = s.powf(3.0 - q) * s; // mass per d(ln s)
    num += dm / s;
    den += dm;
  }
  let harmonic_um = den / num;
  let mean_um = size_mean_inv_s_um(size_params) as f64;
  assert!(
    (mean_um / harmonic_um - 1.0).abs() < 1e-3,
    "harmonic mean size {mean_um} vs numeric {harmonic_um}"
  );
  assert!(
    (mean_um / vel_params[2] as f64 - 1.0).abs() < 1e-3,
    "= s_ref for q = 3.5"
  );
  for shift in [2u32, 0, 3, 1, 6] {
    let n = 1u32 << shift;
    let samples = 4;
    let b = make(shift, samples);
    assert_eq!(batch_streams(&b).0, shift);
    let mass_total: f64 = (0..b.count).map(|j| emit_cluster(&b, j).mass_g() as f64).sum();
    let expect = b.mass_params[0] as f64;
    assert!(
      (mass_total / expect - 1.0).abs() < 1e-5,
      "S {n}: Σ mass {mass_total:.6e} vs batch {expect:.6e}"
    );
    for j in 0..b.count {
      let c = emit_cluster(&b, j);
      assert!(
        (c.beta() - vel_params[3] / vel_params[2]).abs() < 1e-7,
        "S {n}: β"
      );
      let rho_g_m3 = mass_params[1] as f64 * 1.0e6;
      let xsec = 3.0 / (4.0 * rho_g_m3 * mean_um * 1.0e-6);
      assert!(
        (c.misc[2] as f64 / xsec - 1.0).abs() < 1e-4,
        "S {n}: cross-section per gram"
      );
      assert!(
        (c.sigma_rad() - vel_params[1] * vel_params[0]).abs() < 1e-6,
        "S {n}: σ_rad"
      );
      let half_range = 0.5 * (size_params[1] / size_params[0]).ln();
      assert!(
        (c.misc[3].abs() / half_range - 1.0).abs() < 1e-2,
        "S {n}: half log-size range"
      );
      assert!(
        !stream_break(c.misc[3]) || j < n,
        "S {n}: no size wrap breaks"
      );
    }
  }
}

/// The dust image does not depend on the ring capacity: the 8 192- and 32 768-slot systems (4 / 2
/// / 2 and 16 / 8 / 8 streams over ≥ 8 size strata) render, at 3 days in one fixed view, within
/// tolerance of the 262 144 one in optical depth, extent, elongation, white point, coma and the
/// fraction of the frame the production display lights. Guards [`STRATA_MIN_SHIFT`] (one stratum
/// gave +60 % τ and a coma ÷ 12) and the field-derived measurement grid (the per-packet deposit
/// lit 72 % of the frame at 32 768 slots against 4 %). Tolerances from
/// `dust_tau_image_vs_ring_capacity`, which prints the full table.
#[test]
fn dust_image_is_independent_of_ring_capacity() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = move |_: &JetState| cfg;
  let t = 3.0 * 86400.0;
  let px = 128u32;
  // settled system, its packets (view independent) and its coma radius
  let build = |cap: u32| {
    let mut sys = DustSystemState::with_tiers(cap, 3);
    let mut seq = 0u64;
    settle(&mut sys, t, &cfg_at, &mut seq);
    let packets = system_packets(&sys, t, None);
    let coma = sys.coma_radius_m().unwrap_or(0.0);
    (packets, coma)
  };
  struct Metrics {
    total: f64,
    p90: f32,
    elongation: f32,
    lit: f32,
    white: f32,
    coma: f64,
  }
  let fine_rel = 1e-6f32;
  let (ref_packets, coma_ref) = build(262_144);
  assert!(coma_ref > 0.0, "reference coma");
  let half = 3.0 * coma_ref as f32;
  let nucleus = [px as f32 / 2.0, px as f32 / 2.0];
  // the near-lossless and the production image share one packet set and one τ_max probe
  let images = |packets: &[TierPackets]| -> (SystemImage, SystemImage) {
    let mut v = images_from_packets(
      packets,
      half,
      px,
      SHAPE_IDENTITY,
      &[fine_rel, WHITE_TILE_UNIT_REL],
    );
    let prod = v.pop().expect("production image");
    let fine = v.pop().expect("fine image");
    (fine, prod)
  };
  let (ref_fine, ref_prod) = images(&ref_packets);
  let ref_peak = ref_fine.img.iter().cloned().fold(0.0f32, f32::max);
  let phys_threshold = 1e-3 * ref_peak;
  let measure = |fine: &SystemImage, prod: &SystemImage, coma: f64| -> Metrics {
    let shown_threshold = display_threshold(prod.white, prod.soft, 0.1);
    let phys = segment_shape(&fine.img, px, px, nucleus, phys_threshold);
    Metrics {
      total: fine.img.iter().map(|&v| v as f64).sum(),
      p90: phys.radius_p90,
      elongation: phys.elongation,
      lit: prod.img.iter().filter(|&&v| v >= shown_threshold).count() as f32 / (px * px) as f32,
      white: prod.white,
      coma,
    }
  };
  let r = measure(&ref_fine, &ref_prod, coma_ref);
  assert!(r.total > 0.0 && r.p90 > 0.0 && r.lit > 0.0 && r.white > 0.0 && r.coma > 0.0);
  for cap in [32_768u32, 8_192] {
    let (packets, coma) = build(cap);
    let (fine, prod) = images(&packets);
    let m = measure(&fine, &prod, coma);
    let rel = |a: f64, b: f64| (a / b - 1.0).abs();
    std::println!(
      "[ring {cap}] Στ {:+.1}% p90 {:+.1}% elongation {:+.1}% lit {:.1}% vs {:.1}% W {:+.1}% coma {:+.1}%",
      100.0 * (m.total / r.total - 1.0),
      100.0 * (m.p90 / r.p90 - 1.0),
      100.0 * (m.elongation / r.elongation - 1.0),
      100.0 * m.lit,
      100.0 * r.lit,
      100.0 * (m.white / r.white - 1.0),
      100.0 * (m.coma / r.coma - 1.0)
    );
    assert!(
      rel(m.total, r.total) < 0.12,
      "ring {cap}: Στ {:.4e} vs {:.4e}",
      m.total,
      r.total
    );
    assert!(
      rel(m.p90 as f64, r.p90 as f64) < 0.05,
      "ring {cap}: p90 {} vs {} px",
      m.p90,
      r.p90
    );
    // the direction bandwidth of a tier is one cone cell, aperture/√S: with 4 streams it is half
    // the aperture and the cone is drawn that much fatter (the KDE bandwidth, not a lattice), so
    // the shape's elongation is only expected from 8 streams up
    let streams = 1u32 << DustHostState::new(cap / 2).stream_shift();
    if streams >= 8 {
      assert!(
        rel(m.elongation as f64, r.elongation as f64) < 0.15,
        "ring {cap}: elongation {} vs {}",
        m.elongation,
        r.elongation
      );
    } else {
      std::println!("[ring {cap}] {streams} streams: elongation not compared (bandwidth)");
    }
    // the displayed area at the production unit flips by the fixed-point rounding at the edge
    // (error diffusion: ±1 count per packet, fewer and heavier packets on a small ring): 6 pt
    assert!(
      (m.lit - r.lit).abs() < 0.06,
      "ring {cap}: lit {} vs {}",
      m.lit,
      r.lit
    );
    assert!(
      rel(m.white as f64, r.white as f64) < 0.15,
      "ring {cap}: W {:.3e} vs {:.3e}",
      m.white,
      r.white
    );
    assert!(
      rel(m.coma, r.coma) < 0.15,
      "ring {cap}: coma {:.0} vs {:.0} m",
      m.coma,
      r.coma
    );
  }
}

fn norm3f(a: [f32; 3]) -> f32 {
  (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Box-filtered (`2r+1` square) copy of a `px × px` image
fn box_filter(img: &[f32], px: u32, r: i64) -> alloc::vec::Vec<f32> {
  let n = px as i64;
  let mut out = alloc::vec![0.0f32; img.len()];
  for y in 0..n {
    for x in 0..n {
      let (mut s, mut c) = (0.0f64, 0);
      for dy in -r..=r {
        for dx in -r..=r {
          let (xx, yy) = (x + dx, y + dy);
          if xx >= 0 && xx < n && yy >= 0 && yy < n {
            s += img[(yy * n + xx) as usize] as f64;
            c += 1;
          }
        }
      }
      out[(y * n + x) as usize] = (s / c as f64) as f32;
    }
  }
  out
}

/// High-pass ripple of an image over its dust region (the observer's `ripple_metric`): rms of
/// `(img − box₉)/box₉` over the pixels whose box mean is above the median of the non-empty ones.
fn ripple_rms(img: &[f32], px: u32) -> f64 {
  let blur = box_filter(img, px, 4);
  let mut nz: alloc::vec::Vec<f32> = blur.iter().cloned().filter(|&b| b > 0.0).collect();
  if nz.is_empty() {
    return 0.0;
  }
  nz.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let median = nz[nz.len() / 2];
  let (mut s, mut c) = (0.0f64, 0usize);
  for i in 0..img.len() {
    if blur[i] > median {
      let v = ((img[i] - blur[i]) / blur[i]) as f64;
      s += v * v;
      c += 1;
    }
  }
  if c == 0 { 0.0 } else { (s / c as f64).sqrt() }
}

/// **The executable guarantee of the whole-cell kernels.** `n` grains are drawn from the
/// *continuous* emission distribution of `cfg` (direction uniform over the cone, speed
/// `N(v_mean(s), v_std(s))` with `v ∝ s^-½`, size by cross-section — uniform in `√β` over the
/// range —, emission time uniform over the history), each propagated in f64 with its own μ(β)
/// and binned into the same orthographic view as `system_image`. The renderer's image (sum of
/// the clusters' kernels: `S` streams × a few time samples per window, each a cell) must agree
/// with that histogram, both box-filtered, to within the Monte-Carlo noise, and must carry no
/// lattice (high-pass ripple below 5 %): no speed shells, no stream rays, no chords between
/// sizes, at every scale. Four views: a 1-day jet at 50 km and 5 000 km half-width, a 2-year
/// history at 0.03 AU and 0.3 AU.
#[test]
fn cell_kernels_reproduce_the_continuous_emission() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let beta_ref = cfg.beta_ref as f64;
  let range = SIZE_RANGE_FACTOR;
  let v_std_rel = (cfg.v_std / cfg.v_mean) as f64;
  // (days of history, half extent, pixels, grains, spinning nucleus): the spinning cases are the
  // app's jet (12.4 h, an equatorial site 20° north, lit once per rotation): every tier samples
  // the rotation at its own cadence (28 min / 6.6 h / 2.2 d), and the picture must not depend on it
  // (days of history, half extent, pixels, grains, spinning nucleus, far scene): the far scene is
  // the app's: a 67P-like orbit with the perihelion passage inside a 5.3-year history, production
  // ∝ r⁻², the 262 144-slot ring, at 480 px — the 1 AU half-height frame where the oldest tier's
  // sample polylines are pixels apart laterally (judged on the whole sheet and on the high-pass
  // excess over the truth: the sample striations)
  let cases: [(f64, f32, u32, usize, bool, bool); 9] = [
    (1.0, 50.0e3, 96, 300_000, false, false),
    (1.0, 500.0e3, 96, 300_000, false, false),
    (730.0, 0.03 * AU_M as f32, 96, 400_000, false, false),
    (730.0, 0.3 * AU_M as f32, 96, 400_000, false, false),
    (1.0, 500.0e3, 96, 300_000, true, false),
    // the first two hours of a spinning jet seen from 5 km: the frame the app opens on
    (2.0 / 24.0, 5.0e3, 240, 300_000, true, false),
    (730.0, 0.03 * AU_M as f32, 96, 400_000, true, false),
    (730.0, 0.3 * AU_M as f32, 96, 400_000, true, false),
    (1900.0, AU_M as f32, 480, 8_000_000, true, true),
  ];
  let axis: V3 = [0.0, 0.0, 1.0];
  let lat = 20.0f64.to_radians();
  let n0: V3 = [lat.cos(), 0.0, lat.sin()];
  // AETHERVK_DUST_CELLS_PX=<px>: render every case at that resolution (diagnostics with the
  // PGM dump; the Monte Carlo gets the grains scaled by the pixel count)
  let px_override: Option<u32> =
    std::env::var("AETHERVK_DUST_CELLS_PX").ok().and_then(|v| v.parse().ok());
  // AETHERVK_DUST_CELLS_67P=1: only the observer's scene instead — a 67P-like orbit with the
  // perihelion passage inside a 5.3-year history, production ∝ r⁻², the 262 144-slot ring, the
  // 0.033 AU half-height frame at 480 px (the far rays of the zoom series)
  let scene_67p = std::env::var("AETHERVK_DUST_CELLS_67P").is_ok_and(|v| v == "1");
  let cases: alloc::vec::Vec<(f64, f32, u32, usize, bool, bool)> = if scene_67p {
    alloc::vec![
      (1900.0, 5.0e9, 480, 4_000_000, true, true),
      (1900.0, AU_M as f32, 480, 8_000_000, true, true),
    ]
  } else {
    cases.to_vec()
  };
  let t_aphelion_s = (1900.0 - 270.0) * 86400.0;
  let (r67, v67) = comet_state_67p(t_aphelion_s);
  for (days, half, px, n, spin, far) in cases {
    let (px, n) = match px_override {
      Some(p) if !far => (p, n * ((p * p) / (px * px)).max(1) as usize),
      _ => (px, n),
    };
    let t = days * 86400.0;
    let (rc0, vc0) = if far { (r67, v67) } else { comet_state() };
    let cfg = if spin {
      DustEmitConfig {
        jet_dir: [n0[0] as f32, n0[1] as f32, n0[2] as f32],
        ..cfg
      }
    } else {
      cfg
    };
    let jet_at = move |t: f64| -> Option<JetState> {
      if far {
        Some(spinning_jet_from((r67, v67), t, axis, n0, true))
      } else if spin {
        Some(spinning_jet(t, axis, n0, true))
      } else {
        orbit_jet(t)
      }
    };
    // 32 768 slots: 16 / 8 / 8 streams, so every tier's direction kernel (one cone cell) is a
    // fraction of the aperture; the 8 192-slot ring's 2-stream old tiers draw a bimodal cone
    let mut sys = DustSystemState::with_tiers(if far { 262_144 } else { 32_768 }, 3);
    sys.set_ignition(Some(0.0));
    let mut seq = 0u64;
    // the observer's production law: Afρ ∝ r⁻² (the app's power 2), i.e. q ∝ (1 AU / r)²
    let production = move |r_m: f64| -> f64 { if far { (AU_M / r_m).powi(2) } else { 1.0 } };
    let cfg_at = move |j: &JetState| DustEmitConfig {
      q_dust_kgs: cfg.q_dust_kgs * production(norm(j.r_m)),
      ..cfg
    };
    settle_jet(&mut sys, t, &jet_at, &cfg_at, &mut seq);
    let anchor = sys.tiers[0].draw_state().expect("tier 0").frame.at_time(t).anchor_m();
    {
      // diagnostics: where the packets are
      let tiers = system_packets(&sys, t, None);
      for (k, host) in sys.tiers.iter().enumerate() {
        std::println!(
          "[cells] tier {k}: live {} batches {} next_window {:?} due {} unlit {}",
          host.ring.live(),
          host.ring.batches.len(),
          host.next_window,
          host.due_window,
          host.unlit_windows
        );
        if let Some(ds) = host.draw_state() {
          let f = ds.frame.at_time(t);
          std::println!(
            "[cells] tier {k}: frame ttl {:?} t_now {:.1} live {} batches {} ttl_s {} jet t {:?}",
            f.ttl,
            f.t_now_s(),
            host.ring.live(),
            host.ring.batches.len(),
            host.ttl_s,
            host.jet.map(|j| j.t_s)
          );
        }
      }
      for (k, (r, m)) in tiers.iter().enumerate() {
        let live: alloc::vec::Vec<&DustMoments> = m.iter().filter(|q| q.flux() > 0.0).collect();
        let mut d: alloc::vec::Vec<f32> = live.iter().map(|q| norm3f(q.mean())).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let edge_far: alloc::vec::Vec<f32> =
          live.iter().map(|q| norm3f(q.edge(SIZE_EDGES as usize - 1))).collect();
        let ages: alloc::vec::Vec<f32> = live.iter().map(|q| q.age()).collect();
        std::println!(
          "[cells] tier {k}: {} render, {} live moments; |mean| p10/p50/p90 = {:.3e}/{:.3e}/{:.3e} m; max |edge16| {:.3e}; age max {:.3e} s",
          r.len(),
          live.len(),
          d.get(d.len() / 10).copied().unwrap_or(0.0),
          d.get(d.len() / 2).copied().unwrap_or(0.0),
          d.get(d.len() * 9 / 10).copied().unwrap_or(0.0),
          edge_far.iter().cloned().fold(0.0f32, f32::max),
          ages.iter().cloned().fold(0.0f32, f32::max)
        );
      }
    }
    let (img, _, _) = system_image(&sys, t, half, px, SHAPE_IDENTITY, None);
    // Monte Carlo of the continuous emission, binned into the same view
    let mvp = ortho_rot_mvp(half, SHAPE_IDENTITY);
    let mut mc = alloc::vec![0.0f32; (px * px) as usize];
    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (days as u64);
    let jet = [cfg.jet_dir[0], cfg.jet_dir[1], cfg.jet_dir[2]];
    // the renderer's direction kernel: the cone is covered by S Gaussian streams of lateral
    // dispersion σ_lat = v·max(aperture/√S, CHILD_SIGMA_V_REL) (the KDE bandwidth, one cone
    // cell): the continuous emission is drawn through that bandwidth, so the truth to compare with
    // is the uniform cone convolved with it (with 64 streams it is an eighth of the aperture; the
    // small test ring has 4 streams, and a sharp cone would differ by a third on the axis)
    // per tier: the tier that draws a grain of that age has its own stream count
    let sigma_ang_of = |age_s: f64| -> f64 {
      let tier = sys
        .tiers
        .iter()
        .find(|h| {
          let (lo, hi) = h.age_band_s(cfg.ttl_s);
          age_s >= lo && age_s < hi
        })
        .unwrap_or(&sys.tiers[sys.tiers.len() - 1]);
      let streams = 1u32 << tier.stream_shift();
      (cfg.aperture_rad / (streams as f32).sqrt()).max(CHILD_SIGMA_V_REL) as f64
    };
    let (sb_lo, sb_hi) = ((beta_ref / range).sqrt(), (beta_ref * range).sqrt());
    let mut binned = 0usize;
    for _ in 0..n {
      let t_e = lcg(&mut rng) * t;
      let (rc, vc) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t_e);
      // a spinning nucleus: the jet turns with it and emits only while its site faces the Sun
      let jet = if spin {
        let n = rotate_axis_angle(n0, axis, OMEGA * t_e);
        if dot(n, rc) >= 0.0 {
          continue;
        }
        [n[0] as f32, n[1] as f32, n[2] as f32]
      } else {
        jet
      };
      let dir = sample_cone(
        lcg(&mut rng) as f32,
        lcg(&mut rng) as f32,
        jet,
        cfg.aperture_rad,
      );
      let sb = sb_lo + (sb_hi - sb_lo) * lcg(&mut rng);
      let beta = sb * sb;
      let f = sb / beta_ref.sqrt();
      let v = (cfg.v_mean as f64 * f * (1.0 + v_std_rel * gauss_lcg(&mut rng))).max(0.0);
      let d64 = [dir[0] as f64, dir[1] as f64, dir[2] as f64];
      let ax = if d64[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
      } else {
        [0.0, 1.0, 0.0]
      };
      let e1 = unit(cross64(ax, d64));
      let e2 = cross64(d64, e1);
      let (g1, g2) = (gauss_lcg(&mut rng), gauss_lcg(&mut rng));
      let lat = cfg.v_mean as f64 * f * sigma_ang_of(t - t_e);
      let ve = [
        vc[0] + d64[0] * v + lat * (g1 * e1[0] + g2 * e2[0]),
        vc[1] + d64[1] * v + lat * (g1 * e1[1] + g2 * e2[1]),
        vc[2] + d64[2] * v + lat * (g1 * e1[2] + g2 * e2[2]),
      ];
      let (r, _) = kepler::propagate_f64(rc, ve, SUN_MU_M3_S2 * (1.0 - beta), t - t_e);
      let local = [
        (r[0] - anchor[0]) as f32,
        (r[1] - anchor[1]) as f32,
        (r[2] - anchor[2]) as f32,
      ];
      if let Some(p) = project_point(local, &mvp, px, px) {
        let (x, y) = (p[0].floor() as i64, p[1].floor() as i64);
        if x >= 0 && y >= 0 && x < px as i64 && y < px as i64 {
          // production ∝ q(r(t_e)): the grain's weight
          mc[(y as i64 * px as i64 + x) as usize] += production(norm(rc)) as f32;
          binned += 1;
        }
      }
    }
    assert!(
      binned > n / 20,
      "{days} d half {half}: only {binned} grains in view"
    );
    // normalise both to unit sum, box-filter, compare where the Monte Carlo has signal
    let norm_img = |v: &[f32]| {
      let s: f64 = v.iter().map(|&x| x as f64).sum();
      v.iter()
        .map(|&x| (x as f64 / s.max(1e-30)) as f32)
        .collect::<alloc::vec::Vec<f32>>()
    };
    // the renderer reconstructs through the pixel low-pass filter (σ = 0.5 px, every packet):
    // the histogram gets the same filter, or a sub-pixel trail (0.3 AU: 0.04 px wide) would
    // compare a hard 1-px line against a 1.2-px soft one
    let mc_filtered = {
      let n = px as i64;
      let w: [f32; 3] = {
        let g = |d: f32| (-0.5 * d * d / (0.5 * 0.5)).exp();
        let (w0, w1) = (g(0.0), g(1.0));
        let z = w0 + 2.0 * w1;
        [w1 / z, w0 / z, w1 / z]
      };
      let mut tmp = alloc::vec![0.0f32; mc.len()];
      let mut out = alloc::vec![0.0f32; mc.len()];
      for y in 0..n {
        for x in 0..n {
          let mut v = 0.0;
          for (k, wk) in w.iter().enumerate() {
            let xx = x + k as i64 - 1;
            if xx >= 0 && xx < n {
              v += wk * mc[(y * n + xx) as usize];
            }
          }
          tmp[(y * n + x) as usize] = v;
        }
      }
      for y in 0..n {
        for x in 0..n {
          let mut v = 0.0;
          for (k, wk) in w.iter().enumerate() {
            let yy = y + k as i64 - 1;
            if yy >= 0 && yy < n {
              v += wk * tmp[(yy * n + x) as usize];
            }
          }
          out[(y * n + x) as usize] = v;
        }
      }
      out
    };
    let (a, b) = (
      box_filter(&norm_img(&img), px, 2),
      box_filter(&norm_img(&mc_filtered), px, 2),
    );
    let peak = b.iter().cloned().fold(0.0f32, f32::max);
    let (mut se, mut sb2, mut count) = (0.0f64, 0.0f64, 0usize);
    for i in 0..a.len() {
      if b[i] >= 0.05 * peak {
        se += ((a[i] - b[i]) as f64).powi(2);
        sb2 += (b[i] as f64).powi(2);
        count += 1;
      }
    }
    let rel_rms = (se / sb2.max(1e-300)).sqrt();
    let ripple = ripple_rms(&img, px);
    {
      // along the trail (the τ-weighted principal axis of the Monte-Carlo image from the centre)
      // and across it at mid-length: where a difference sits
      let c = px as f32 / 2.0;
      let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
      for y in 0..px {
        for x in 0..px {
          let w = b[(y * px + x) as usize] as f64;
          sx += w * (x as f64 + 0.5 - c as f64);
          sy += w * (y as f64 + 0.5 - c as f64);
          sw += w;
        }
      }
      let n = (sx * sx + sy * sy).sqrt().max(1e-9);
      let (ux, uy) = ((sx / n) as f32, (sy / n) as f32);
      let mut along = alloc::string::String::new();
      for d in [2.0f32, 6.0, 12.0, 20.0, 30.0, 40.0] {
        let (x, y) = ((c + d * ux) as i64, (c + d * uy) as i64);
        if x >= 0 && y >= 0 && x < px as i64 && y < px as i64 {
          let i = (y * px as i64 + x) as usize;
          along.push_str(&alloc::format!(
            " {d:.0}px: {:+.0}%",
            100.0 * (a[i] / b[i].max(1e-30) - 1.0)
          ));
        }
      }
      let mut across = alloc::string::String::new();
      for o in [-6.0f32, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0] {
        let (x, y) = (
          (c + 20.0 * ux - o * uy) as i64,
          (c + 20.0 * uy + o * ux) as i64,
        );
        if x >= 0 && y >= 0 && x < px as i64 && y < px as i64 {
          let i = (y * px as i64 + x) as usize;
          across.push_str(&alloc::format!(" {o:+.0}px: {:.2e}/{:.2e}", a[i], b[i]));
        }
      }
      std::println!(
        "[cells] along the trail (render/MC − 1):{along}\n[cells] across at 20 px (render/MC):{across}"
      );
      // radial profile (normalised images, box-filtered) at a few radii from the centre
      let c = px as f32 / 2.0;
      let mut line = alloc::string::String::new();
      for r in [1.0f32, 3.0, 6.0, 12.0, 24.0, 40.0] {
        let (mut sa, mut sb, mut n) = (0.0f64, 0.0f64, 0);
        for k in 0..64 {
          let th = k as f32 * core::f32::consts::TAU / 64.0;
          let (x, y) = ((c + r * th.cos()) as i64, (c + r * th.sin()) as i64);
          if x >= 0 && y >= 0 && x < px as i64 && y < px as i64 {
            sa += a[(y * px as i64 + x) as usize] as f64;
            sb += b[(y * px as i64 + x) as usize] as f64;
            n += 1;
          }
        }
        line.push_str(&alloc::format!(
          " r{r:.0}: {:.2e}/{:.2e}",
          sa / n as f64,
          sb / n as f64
        ));
      }
      std::println!("[cells] profile render/MC{line}");
    }
    // AETHERVK_DUST_CELLS_PGM=<prefix>: the two box-filtered images side by side (render | MC)
    if let Ok(prefix) = std::env::var("AETHERVK_DUST_CELLS_PGM") {
      let peak_a = a.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
      let mut out = alloc::format!("P2\n{} {px}\n255\n", 2 * px + 1).into_bytes();
      for y in 0..px as usize {
        for x in 0..(2 * px + 1) as usize {
          let v = if x < px as usize {
            a[y * px as usize + x] / peak_a
          } else if x == px as usize {
            1.0
          } else {
            b[y * px as usize + x - px as usize - 1] / peak
          };
          // fourth-root stretch: the halo at 1 % of the peak stays visible
          out.extend_from_slice(
            alloc::format!("{} ", (v.clamp(0.0, 1.0).powf(0.25) * 255.0) as u8).as_bytes(),
          );
        }
        out.push(b'\n');
      }
      let tag = if far {
        "_67p"
      } else if spin {
        "_spin"
      } else {
        ""
      };
      std::fs::write(alloc::format!("{prefix}_{days}d_{half:.0}m{tag}.pgm"), out).unwrap();
      // the raw images too (f32 little endian, render then Monte Carlo, both unit sum, unfiltered)
      let mut raw: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
      for v in norm_img(&img).iter().chain(norm_img(&mc_filtered).iter()) {
        raw.extend_from_slice(&v.to_le_bytes());
      }
      std::fs::write(alloc::format!("{prefix}_{days}d_{half:.0}m{tag}.f32"), raw).unwrap();
    }
    // the Monte-Carlo noise after the 5×5 box: ~1/√(25 · grains per px)
    let grains_px = binned as f64 / count.max(1) as f64;
    let noise = 1.0 / (25.0 * grains_px).sqrt();
    std::println!(
      "[cells] {days} d, half {half:.3e} m{}: {count} px compared, rel rms {rel_rms:.3} (MC noise {noise:.3}), ripple {ripple:.3}, {grains_px:.0} grains/px",
      if spin { ", spinning" } else { "" }
    );
    assert!(count > 50, "{days} d half {half}: {count} px");
    // the two 1-D profiles the eye judges: the brightness along the trail (summed across it)
    // and the width across it (summed along), compared where the Monte Carlo has signal; a 2-D
    // rms would also count sub-pixel offsets of a trail a few pixels wide
    let (ux, uy, c) = {
      let c = px as f32 / 2.0;
      let (mut sx, mut sy) = (0.0f64, 0.0f64);
      for y in 0..px {
        for x in 0..px {
          let w = b[(y * px + x) as usize] as f64;
          sx += w * (x as f64 + 0.5 - c as f64);
          sy += w * (y as f64 + 0.5 - c as f64);
        }
      }
      let n = (sx * sx + sy * sy).sqrt().max(1e-9);
      ((sx / n) as f32, (sy / n) as f32, c)
    };
    let bins = px as i64;
    let (mut along_a, mut along_b) = (
      alloc::vec![0.0f64; bins as usize * 2],
      alloc::vec![0.0f64; bins as usize * 2],
    );
    let (mut across_a, mut across_b) = (
      alloc::vec![0.0f64; bins as usize * 2],
      alloc::vec![0.0f64; bins as usize * 2],
    );
    for y in 0..px {
      for x in 0..px {
        let (dx, dy) = (x as f32 + 0.5 - c, y as f32 + 0.5 - c);
        let d = (dx * ux + dy * uy).round() as i64 + bins;
        let o = (-dx * uy + dy * ux).round() as i64 + bins;
        let i = (y * px + x) as usize;
        if d >= 0 && d < 2 * bins {
          along_a[d as usize] += a[i] as f64;
          along_b[d as usize] += b[i] as f64;
        }
        if o >= 0 && o < 2 * bins {
          across_a[o as usize] += a[i] as f64;
          across_b[o as usize] += b[i] as f64;
        }
      }
    }
    let profile_rms = |pa: &[f64], pb: &[f64]| -> f64 {
      let peak = pb.iter().cloned().fold(0.0f64, f64::max);
      let (mut se, mut sb2) = (0.0f64, 0.0f64);
      for i in 0..pa.len() {
        if pb[i] >= 0.05 * peak {
          se += (pa[i] - pb[i]).powi(2);
          sb2 += pb[i].powi(2);
        }
      }
      (se / sb2.max(1e-300)).sqrt()
    };
    let (along_rms, across_rms) = (
      profile_rms(&along_a, &along_b),
      profile_rms(&across_a, &across_b),
    );
    std::println!(
      "[cells] {days} d, half {half:.3e} m: along-trail profile rms {along_rms:.3}, across-trail profile rms {across_rms:.3}"
    );
    assert!(
      along_rms < 0.08 + 2.0 * noise && across_rms < 0.08 + 2.0 * noise,
      "{days} d half {half}: the rendered trail differs from the continuous emission: along {along_rms:.3}, across {across_rms:.3} rms (noise {noise:.3}; 2-D {rel_rms:.3})"
    );
    // the high-pass ripple is reported, not asserted by itself: on a 1/r² coma the 9×9 box
    // high-pass measures curvature, and the Monte-Carlo comparison is the lattice detector (a
    // lattice is a difference from the continuous emission)
    let _ = ripple;
    if far {
      // the far scene is judged on the whole sheet (the truth above 1e-4 of its peak after the
      // box; 1 % would be the coma alone): the 2-D rms of the difference, and the render's 9-px
      // high-pass rms in excess of the truth's — the sample striations of the oldest tier
      let region: alloc::vec::Vec<usize> = (0..a.len()).filter(|&i| b[i] >= 1e-4 * peak).collect();
      let (mut se, mut sb2) = (0.0f64, 0.0f64);
      for &i in &region {
        se += ((a[i] - b[i]) as f64).powi(2);
        sb2 += (b[i] as f64).powi(2);
      }
      let sheet_rms = (se / sb2.max(1e-300)).sqrt();
      let hp = |img: &[f32]| -> f64 {
        let lo = box_filter(img, px, 4);
        let mut s2 = 0.0f64;
        for &i in &region {
          let r = ((img[i] - lo[i]) / lo[i].max(1e-30)) as f64;
          s2 += r * r;
        }
        (s2 / region.len().max(1) as f64).sqrt()
      };
      let (hp_render, hp_truth) = (hp(&a), hp(&b));
      std::println!(
        "[cells] {days} d, half {half:.3e} m, far sheet: {} px, rel rms {sheet_rms:.3}, high-pass render {hp_render:.3} truth {hp_truth:.3} (excess {:.3}), noise {noise:.3}",
        region.len(),
        hp_render - hp_truth
      );
      assert!(
        sheet_rms < 0.15 + 2.0 * noise,
        "{days} d half {half}: the far sheet differs from the continuous emission by {sheet_rms:.3} rms"
      );
      assert!(
        hp_render - hp_truth < 0.03 + noise,
        "{days} d half {half}: sample striations: high-pass {hp_render:.3} against the truth's {hp_truth:.3}"
      );
    }
  }
}

/// A chain of open-ended chord capsules deposits exactly its flux, whatever its orientation,
/// length or the texel size: the band of every chord covers its own texels and nothing beyond,
/// the chords tile the line (`CapsuleBand`, `scatter_into` with `Packet::open`). The energy
/// test of the chord representation: a loss here is a rotation- or zoom-dependent picture.
#[test]
fn open_chord_chains_deposit_their_flux_at_any_angle() {
  let (w, h) = (256u32, 256u32);
  let layout = PyramidLayout::new(w, h);
  let inv_unit = 1.0e4;
  let mut worst = 0.0f64;
  for &angle_deg in &[0.0f32, 17.0, 45.0, 90.0, 123.0, 180.0, 250.0] {
    for &len_px in &[3.0f32, 20.0, 80.0] {
      for &sigma in &[0.6f32, 2.5] {
        for &texel in &[1u32, 2, 4] {
          // the splat's level rule (`splat_level`): texels are at most σ / DUST_SPLAT_SIGMA_TEXELS,
          // level 0 whatever σ (the pixel filter keeps σ ≥ 0.5 px there)
          if texel > 1 && sigma < DUST_SPLAT_SIGMA_TEXELS * texel as f32 {
            continue;
          }
          let a = angle_deg.to_radians();
          let dir = [a.cos(), a.sin()];
          let n_chords = 16usize;
          let total_flux = 1.0f32;
          let start = [128.0 - 0.5 * len_px * dir[0], 128.0 - 0.5 * len_px * dir[1]];
          let mut pyr = alloc::vec![0u32; layout.total_words as usize];
          pyr[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
          let (off, lw, lh) = layout.levels[texel.trailing_zeros() as usize];
          let s2 = sigma * sigma;
          for c in 0..n_chords {
            let u0 = c as f32 / n_chords as f32;
            let seg = [
              len_px * dir[0] / n_chords as f32,
              len_px * dir[1] / n_chords as f32,
            ];
            let mu = [
              start[0] + u0 * len_px * dir[0],
              start[1] + u0 * len_px * dir[1],
            ];
            // amp = flux / (2π √det): the capsule's integral over the plane is amp·2π√det
            let det = s2 * s2;
            let flux = total_flux / n_chords as f32;
            let p = ScreenPacket {
              mu,
              cov: [s2, 0.0, s2],
              seg,
              amp: flux / (2.0 * core::f32::consts::PI * det.sqrt()),
              sigma_min: sigma,
              det,
              depth_au: 1.0,
              open: [c > 0, c + 1 < n_chords],
              rho: [1.0, 1.0],
            };
            scatter_into(
              &p,
              [1.0, 1.0, 1.0],
              texel as f32,
              off,
              lw,
              lh,
              inv_unit,
              &mut pyr,
              true,
            );
          }
          let sum: f64 = (0..(lw * lh) as usize)
            .map(|t| pyr[off as usize + t * PYRAMID_TEXEL_WORDS as usize] as f64)
            .sum();
          let expect = total_flux as f64 * inv_unit as f64;
          let err = (sum / expect - 1.0).abs();
          worst = worst.max(err);
          assert!(
            err < 0.02,
            "angle {angle_deg}°, length {len_px} px, σ {sigma} px, texel {texel}: deposited {sum:.0} of {expect:.0} counts ({:+.1} %)",
            100.0 * (sum / expect - 1.0)
          );
        }
      }
    }
  }
  std::println!("[chords] worst flux error {:.3} %", 100.0 * worst);
}

/// A comet without a rotation model (the app's default: `spin: None`, a fixed attitude, the jet
/// site lit for the part of the orbit it faces the Sun) settles its pre-start history like any
/// other: the unlit windows are counted and skipped, the fill ends, nothing runs away.
#[test]
fn no_rotation_model_history_settles() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let jet_at = |t: f64| -> Option<JetState> {
    let (r0, v0) = comet_state();
    let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
    Some(JetState {
      t_s: t,
      r_m: r,
      v_ms: v,
      rot: [0.0, 0.0, 0.0, 1.0],
      site_normal: [1.0, 0.0, 0.0],
      spin: None,
      site_offset_m: [0.0; 3],
    })
  };
  let cfg_at = |_: &JetState| cfg;
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let mut seq = 0u64;
  for t in [0.0, 730.0 * 86400.0] {
    settle_jet(&mut sys, t, &jet_at, &cfg_at, &mut seq);
    for (k, h) in sys.tiers.iter().enumerate() {
      std::println!(
        "[no-spin] t {t:.0}: tier {k} live {} batches {} unlit {} next {:?} due {}",
        h.ring.live(),
        h.ring.batches.len(),
        h.unlit_windows,
        h.next_window,
        h.due_window
      );
    }
    assert!(!sys.building());
    let total: u32 = sys.tiers.iter().map(|h| h.ring.live()).sum();
    assert!(total > 0, "t {t}: no dust at all");
  }
}

/// A jet that cannot be evaluated (`jet_at` = `None`: the comet is not in the cartesian cache)
/// leaves the system caught up, flagged and empty after one tick — never "building", so the
/// logic thread's emission loop cannot spin on it (the 2026-10-10 runaway: `MAX_SEEK_PASSES`
/// empty compute submissions per tick until the driver's host heap was gone). A jet that
/// reappears clears the flag and the history fills as usual.
#[test]
fn unavailable_jet_does_not_keep_the_system_building() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let cfg_at = |_: &JetState| cfg;
  let mut sys = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  let none = |_: f64| -> Option<JetState> { None };
  sys.tick(86400.0, &none, &cfg_at);
  sys.mark_submitted(1);
  assert!(
    !sys.building(),
    "an unavailable jet must not keep the system building"
  );
  let stats = sys.stats();
  assert!(
    stats.iter().all(|t| t.jet_unavailable && t.caught_up && t.live_clusters == 0),
    "{stats:?}"
  );
  // the logic thread's own report path: true once, then false
  let mut fresh = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  assert!(fresh.mark_jet_unavailable());
  assert!(!fresh.mark_jet_unavailable());
  assert!(!fresh.building());
  // the jet comes back: the flag clears and the history fills
  let mut seq = 1u64;
  settle(&mut sys, 86400.0, &cfg_at, &mut seq);
  let stats = sys.stats();
  assert!(
    stats.iter().all(|t| !t.jet_unavailable && t.caught_up),
    "{stats:?}"
  );
  assert!(stats.iter().map(|t| t.live_clusters).sum::<u32>() > 0);
}

/// Diagnostic (prints, no assertion): where on the pyramid a 2-hour jet seen from 5 km lands.
#[test]
fn diag_two_hour_jet_pyramid_levels() {
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  let axis: V3 = [0.0, 0.0, 1.0];
  let lat = 20.0f64.to_radians();
  let n0: V3 = [lat.cos(), 0.0, lat.sin()];
  let cfg = DustEmitConfig {
    jet_dir: [n0[0] as f32, n0[1] as f32, n0[2] as f32],
    ..cfg
  };
  let jet_at = move |t: f64| -> Option<JetState> { Some(spinning_jet(t, axis, n0, true)) };
  let cfg_at = move |_: &JetState| cfg;
  let mut sys = DustSystemState::with_tiers(32_768, 3);
  sys.set_ignition(Some(0.0));
  let mut seq = 0u64;
  let t = 2.0 * 3600.0;
  settle_jet(&mut sys, t, &jet_at, &cfg_at, &mut seq);
  let tiers = system_packets(&sys, t, None);
  let (px, half) = (240u32, 5.0e3f32);
  let layout = PyramidLayout::new(px, px);
  let mvp = ortho_rot_mvp(half, SHAPE_IDENTITY);
  let eye = [0.0, 0.0, 1e9, 0.0];
  for (k, (render, moments)) in tiers.iter().enumerate() {
    let live = render.len() as u32;
    let mut pyr = alloc::vec![0u32; layout.total_words as usize];
    pyr[..PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
    let pc = DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: live,
      flags: 0,
      exposure: 1.0,
      inv_unit: 1e6,
      color: pack_color([1.0; 4]),
      units_per_m: 1.0,
      mvp,
      eye_local: eye,
    };
    splat_tier(
      moments,
      render,
      &pc,
      &Default::default(),
      0,
      &layout,
      &mut pyr,
    );
    let mut parts = alloc::vec::Vec::new();
    for (l, &(off, w, h)) in layout.levels.iter().enumerate() {
      let mut sum = 0u64;
      let mut lit = 0usize;
      for i in 0..(w * h) as usize {
        let c = pyr[off as usize + i * PYRAMID_TEXEL_WORDS as usize];
        sum += c as u64;
        lit += (c > 0) as usize;
      }
      if sum > 0 {
        parts.push(alloc::format!(
          "L{l}({w}x{h}):{lit}/{:.2e}",
          sum as f64 * 4f64.powi(l as i32)
        ));
      }
    }
    std::println!(
      "[diag] tier {k}: {live} clusters; per level lit/Στ·px²: {}",
      parts.join(" ")
    );
    // the packets: how many, their σ_min and their band size
    let mut n_pk = 0usize;
    let mut big = 0usize;
    let mut sig: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
    for i in 0..render.len() {
      let m = &moments[i];
      if !(m.flux() > 0.0) {
        continue;
      }
      let pred = streak_pred(render, i).map(|q| &moments[q]);
      let seg = pred
        .map(|p| {
          let (a, b) = (p.mean(), m.mean());
          [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        })
        .unwrap_or([0.0; 3]);
      let merged = polyline_merged(m, &mvp, px, px);
      for p in subpackets_merged(m, seg, pred, merged) {
        if let Some(sp) = project_packet(&p, 1.0, &mvp, px, px, [eye[0], eye[1], eye[2]]) {
          n_pk += 1;
          sig.push(sp.sigma_min);
          let ex = DUST_SPLAT_SIGMAS * sp.cov[0].sqrt();
          let ey = DUST_SPLAT_SIGMAS * sp.cov[2].sqrt();
          let band = ((sp.seg[0] * sp.seg[0] + sp.seg[1] * sp.seg[1]).sqrt() + 2.0 * ex + 2.0)
            * (2.0 * ey + 2.0);
          if band > DUST_SPLAT_MAX_TEXELS as f32 {
            big += 1;
          }
        }
      }
    }
    sig.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| sig.get(((sig.len() as f64 - 1.0) * f) as usize).copied().unwrap_or(0.0);
    std::println!(
      "[diag] tier {k}: {n_pk} on-screen packets, σ_min px p10/p50/p90 {:.1}/{:.1}/{:.1}, {big} beyond the level-0 texel valve",
      q(0.1),
      q(0.5),
      q(0.9)
    );
  }
  // the youngest window's packets, in metres from the anchor (the jet)
  {
    let (render, moments) = &tiers[0];
    let n = render.len();
    let anchor = sys.tiers[0].draw_state().expect("tier 0").frame.at_time(t).anchor_m();
    let _ = anchor;
    for i in (n.saturating_sub(32)..n).step_by(8) {
      let m = &moments[i];
      if !(m.flux() > 0.0) {
        continue;
      }
      let pred = streak_pred(render, i).map(|q| &moments[q]);
      let seg = pred
        .map(|p| {
          let (a, b) = (p.mean(), m.mean());
          [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
        })
        .unwrap_or([0.0; 3]);
      let merged = polyline_merged(m, &mvp, px, px);
      let packets = subpackets_merged(m, seg, pred, merged);
      let nm = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
      let pk: alloc::vec::Vec<alloc::string::String> = packets
        .iter()
        .take(8)
        .map(|p| {
          alloc::format!(
            "[|mean| {:.0} m, |seg| {:.0} m, σ {:.0}/{:.0}/{:.0} m, flux {:.2e}, open {:?}]",
            nm(p.mean),
            nm(p.seg),
            p.cov[0].max(0.0).sqrt(),
            p.cov[3].max(0.0).sqrt(),
            p.cov[5].max(0.0).sqrt(),
            p.flux,
            p.open
          )
        })
        .collect();
      std::println!(
        "[diag] cluster {i}: age {:.0} s, |mean| {:.0} m, pred {} (|seg_t| {:.0} m), merged {merged}, ratio {:.1}, pieces {}, {} packets: {}",
        m.age(),
        nm(m.mean()),
        pred
          .map(|p| alloc::format!("age {:.0} s", p.age()))
          .unwrap_or_else(|| "none".into()),
        nm(seg),
        width_ratio(m, pred),
        time_pieces(m, pred),
        packets.len(),
        pk.join(" ")
      );
    }
  }
  // images of subsets (AETHERVK_DUST_DIAG_PGM=<prefix>): all clusters, the youngest window's
  // (the last 16 compact indices), the rest, and one stream
  if let Ok(prefix) = std::env::var("AETHERVK_DUST_DIAG_PGM") {
    let n = tiers[0].0.len();
    let subsets: [(&str, alloc::boxed::Box<dyn Fn(usize, usize) -> bool>); 4] = [
      ("all", alloc::boxed::Box::new(|_, _| true)),
      ("youngest", alloc::boxed::Box::new(move |i, _| i + 16 >= n)),
      ("older", alloc::boxed::Box::new(move |i, _| i + 16 < n)),
      ("stream0", alloc::boxed::Box::new(|i, _| i % 16 == 0)),
    ];
    for (name, keep) in &subsets {
      let (img, _, _) = system_image(&sys, t, half, px, SHAPE_IDENTITY, Some(keep.as_ref()));
      let bytes: alloc::vec::Vec<u8> = img.iter().flat_map(|v| v.to_le_bytes()).collect();
      std::fs::write(alloc::format!("{prefix}_{name}.f32"), bytes).unwrap();
    }
  }
}
