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
  };
  b.set_comet(rc, vc, T_START, dur);
  b
}

#[test]
fn batch_mass_is_conserved_by_importance_weights() {
  let b = test_batch(4096, 1800.0);
  let total: f64 = (0..b.count).map(|j| emit_cluster(&b, j).mass_g() as f64).sum();
  let rel = (total - 1.0e6).abs() / 1.0e6;
  assert!(rel < 0.01, "mass sum {total} rel err {rel}");
}

#[test]
fn emission_times_and_beta_are_in_range() {
  let b = test_batch(1000, 1800.0);
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let t0 = c.t0().to_f64();
    assert!(t0 >= T_START && t0 <= T_START + 1800.0, "t0 {t0}");
    let s = c.misc[1];
    assert!(s >= 4.99 && s <= 500.1, "s {s}");
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
  // emit a short batch, evaluate 10 days later relative to the comet
  let b = test_batch(2000, 60.0);
  let (frame, rc) = frame_after(10.0);
  let anti_sun = scale(rc, 1.0 / norm(rc));
  let mut pts: alloc::vec::Vec<(f32, f64)> = alloc::vec::Vec::new();
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let e = evaluate_cluster(&c, j, &frame);
    assert!(e.age_id_dbeta_flux[3] > 0.0);
    assert_eq!(e.age_id_dbeta_flux[1].to_bits(), j);
    let p = [
      e.pos_size[0] as f64,
      e.pos_size[1] as f64,
      e.pos_size[2] as f64,
    ];
    pts.push((c.beta(), dot(p, anti_sun)));
  }
  // radiation pressure: displacement ½ β g t² dominates ejection for the small (high β) grains
  pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  let n = pts.len();
  let mean = |s: &[(f32, f64)]| s.iter().map(|p| p.1).sum::<f64>() / s.len() as f64;
  let low = mean(&pts[..n / 5]);
  let high = mean(&pts[n - n / 5..]);
  assert!(
    high > 0.0,
    "high-β grains must move anti-sunward, got {high}"
  );
  assert!(
    high > low * 2.0,
    "displacement must grow with β: low {low} high {high}"
  );
  // expected order of magnitude for the largest β: ½ β g t²
  let g = SUN_MU_M3_S2 / dot(rc, rc);
  let beta_max = pts[n - 1].0 as f64;
  let expect = 0.5 * beta_max * g * (10.0f64 * 86400.0).powi(2);
  assert!(
    pts[n - 1].1 > 0.3 * expect && pts[n - 1].1 < 3.0 * expect,
    "{} vs {expect}",
    pts[n - 1].1
  );
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

#[test]
fn gpu_layout_sizes() {
  assert_eq!(core::mem::size_of::<DustCluster>(), 80);
  assert_eq!(core::mem::size_of::<DustRenderCluster>(), 32);
  assert_eq!(core::mem::size_of::<DustBatch>(), 192);
  assert_eq!(core::mem::size_of::<DustFrame>(), 64);
}

#[test]
fn render_children_keep_the_instance_budget() {
  let cap = RING_CAPACITY_HIGH;
  assert_eq!(render_children(cap, 0), CHILDREN_PER_CLUSTER);
  assert_eq!(render_children(cap, 1), MAX_CHILDREN_PER_CLUSTER);
  assert_eq!(render_children(cap, cap), CHILDREN_PER_CLUSTER);
  for live in [1u32, 100, 6492, 40_000, 100_000, cap] {
    let k = render_children(cap, live);
    assert!((CHILDREN_PER_CLUSTER..=MAX_CHILDREN_PER_CLUSTER).contains(&k));
    // never more than the dense budget, unless clamped at the minimum
    assert!(
      live as u64 * k as u64 <= cap as u64 * CHILDREN_PER_CLUSTER as u64
        || k == CHILDREN_PER_CLUSTER
    );
  }
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
  let (r0, v0) = comet_state();
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
  assert!(ds.live_count > 0 && ds.compute_wait > 0 && ds.mean_cluster_flux > 0.0);

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
