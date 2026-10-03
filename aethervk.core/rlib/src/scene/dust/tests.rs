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
    r = add(r, scale(add(add(k1r, scale(k2r, 2.0)), add(scale(k3r, 2.0), k4r)), h / 6.0));
    v = add(v, scale(add(add(k1v, scale(k2v, 2.0)), add(scale(k3v, 2.0), k4v)), h / 6.0));
  }
  (r, v)
}

/// df64 propagation with f64 in/out
fn prop_df(r0: V3, v0: V3, mu: f64, dt: f64) -> (V3, V3) {
  let (r, v) = kepler::propagate(&Df3::from_f64(r0), &Df3::from_f64(v0), Df::from_f64(mu), Df::from_f64(dt));
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
  let xs = [1.0e12 / 3.0, -7.25e-3, 9.87654321e5, 1.0 / 7.0, -3.3e16, 2.0e-9];
  let tol = |x: f64| x.abs() * 4.0e-14 + 1e-35;
  for &a in &xs {
    for &b in &xs {
      let (da, db) = (Df::from_f64(a), Df::from_f64(b));
      // inputs are df-rounded; compare against f64 ops on the df-rounded values
      let (a, b) = (da.to_f64(), db.to_f64());
      assert!((da.add(db).to_f64() - (a + b)).abs() <= tol(a.abs() + b.abs()), "{a}+{b}");
      assert!((da.sub(db).to_f64() - (a - b)).abs() <= tol(a.abs() + b.abs()), "{a}-{b}");
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
      (s.cos(), s.sin() / s, (1.0 - s.cos()) / x, (s - s.sin()) / (x * s))
    } else if x < -1e-4 {
      let s = (-x).sqrt();
      (s.cosh(), s.sinh() / s, (s.cosh() - 1.0) / (-x), (s.sinh() - s) / (-x * s))
    } else {
      (1.0 - x / 2.0 + x * x / 24.0, 1.0 - x / 6.0 + x * x / 120.0, 0.5 - x / 24.0 + x * x / 720.0, 1.0 / 6.0 - x / 120.0 + x * x / 5040.0)
    };
    let refs = [c0, c1, c2, c3];
    let c = kepler::stumpff_f64(x);
    let d = kepler::stumpff_df(Df::from_f64(x));
    let f = kepler::stumpff_f32(x as f32);
    for k in 0..4 {
      let scale = 1.0 + refs[k].abs() + c0.abs();
      assert!((c[k] - refs[k]).abs() < 1e-11 * scale, "f64 c{k}({x}) {} vs {}", c[k], refs[k]);
      assert!((d[k].to_f64() - refs[k]).abs() < 1e-10 * scale, "df c{k}({x}) {} vs {}", d[k].to_f64(), refs[k]);
      assert!((f[k] as f64 - refs[k]).abs() < 1e-4 * scale, "f32 c{k}({x}) {} vs {}", f[k], refs[k]);
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
    assert!(((e(r1, v1) - e(r0, v0)) / e(r0, v0)).abs() < 1e-11, "energy dt {dt}");
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
  let (rb, _) = prop_df(add(r0, [1.0, 0.0, 0.0]), add(v0, [0.0, 1e-3, 0.0]), SUN_MU_M3_S2, dt);
  let (ra64, _) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, dt);
  let (rb64, _) = kepler::propagate_f64(add(r0, [1.0, 0.0, 0.0]), add(v0, [0.0, 1e-3, 0.0]), SUN_MU_M3_S2, dt);
  let d = sub(rb, ra);
  let d64 = sub(rb64, ra64);
  assert!(norm(sub(d, d64)) < 0.05, "relative displacement df {d:?} vs f64 {d64:?}");
}

// ─── emission / evaluation ─────────────────────────────────────────────────

const T_START: f64 = 4.0e8; // ~12.7 years after the epoch: exercises df64 time

fn test_batch(count: u32, dur: f64) -> DustBatch {
  let (rc, vc) = comet_state();
  let dist = SizeDistribution::from_diameter_um(100.0);
  let (size_params, vel_params, mass_params) = batch_params(&dist, 100.0, 0.533, 0.0213, 2.0, 0.5, 1.0e6, 0.37);
  let mut b = DustBatch {
    comet_r_t_hi: [0.0; 4],
    comet_r_t_lo: [0.0; 4],
    comet_v_dur_hi: [0.0; 4],
    comet_v_dur_lo: [0.0; 4],
    rot_start: [0.0, 0.0, 0.0, 1.0],
    rot_end: [0.0, 0.0, 0.0, 1.0],
    // jet roughly sunward
    jet_dir_aperture: [0.35, 0.93, 0.04, 0.6],
    size_params,
    vel_params,
    mass_params,
    first_index: 12345,
    count,
    ring_mask: RING_CAPACITY_HIGH - 1,
    seed: 0xC0FFEE,
    _pad: [0; 4],
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
    let p = [e.pos_size[0] as f64, e.pos_size[1] as f64, e.pos_size[2] as f64];
    pts.push((c.beta(), dot(p, anti_sun)));
  }
  // radiation pressure: displacement ½ β g t² dominates ejection for the small (high β) grains
  pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
  let n = pts.len();
  let mean = |s: &[(f32, f64)]| s.iter().map(|p| p.1).sum::<f64>() / s.len() as f64;
  let low = mean(&pts[..n / 5]);
  let high = mean(&pts[n - n / 5..]);
  assert!(high > 0.0, "high-β grains must move anti-sunward, got {high}");
  assert!(high > low * 2.0, "displacement must grow with β: low {low} high {high}");
  // expected order of magnitude for the largest β: ½ β g t²
  let g = SUN_MU_M3_S2 / dot(rc, rc);
  let beta_max = pts[n - 1].0 as f64;
  let expect = 0.5 * beta_max * g * (10.0f64 * 86400.0).powi(2);
  assert!(pts[n - 1].1 > 0.3 * expect && pts[n - 1].1 < 3.0 * expect, "{} vs {expect}", pts[n - 1].1);
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
  assert_eq!(evaluate_cluster(&c, 7, &at(t0 - 1.0)).age_id_dbeta_flux[3], 0.0);
  assert_eq!(evaluate_cluster(&c, 7, &at(t0 + 101.0)).age_id_dbeta_flux[3], 0.0);
  let live = evaluate_cluster(&c, 7, &at(t0 + 50.0));
  assert!(live.age_id_dbeta_flux[3] > 0.0);
  assert!((live.age_id_dbeta_flux[0] - 50.0).abs() < 1e-3, "age {}", live.age_id_dbeta_flux[0]);
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
fn host_tick_conserves_mass_tracks_readiness_and_rewinds() {
  let (r0, v0) = comet_state();
  let mut host = DustHostState::new(RING_CAPACITY_LOW);
  let cfg = test_cfg(1.5e-3, 30.0 * 86400.0);
  // 3 h/s at 60 ticks/s: 180 scaled s per 16.6 ms tick
  let (dt_tick_s, dt_tick_us) = (180.0, 16_667_i64);
  let jet_at = |t: f64| {
    let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
    JetState { t_s: t, r_m: r, v_ms: v, rot: [0.0, 0.0, 0.0, 1.0] }
  };
  let mut emitted_mass = 0.0;
  let mut batches = 0;
  let (mut t, mut now) = (0.0, 1_000_000_i64);
  for tick in 0..6000u64 {
    let out = host.tick(jet_at(t), now, &cfg);
    for b in &out {
      emitted_mass += b.mass_params[0] as f64;
      batches += 1;
      // window coherence: duration matches the gate interval, slots are inside the ring
      assert!(b.comet_v_dur_hi[3] >= 0.0);
      assert_eq!(b.ring_mask, RING_CAPACITY_LOW - 1);
    }
    // pending batches are never drawable
    let (_, live, _) = host.ring.drawable();
    let pending: u32 = host.ring.batches.iter().filter(|b| b.ready == READY_PENDING).map(|b| b.count).sum();
    assert!(live + pending <= host.ring.live());
    host.ring.mark_submitted(tick + 1);
    assert!(host.ring.live() <= RING_CAPACITY_LOW - RING_CAPACITY_LOW / RING_GUARD_DIVISOR);
    t += dt_tick_s;
    now += dt_tick_us;
  }
  assert!(batches > 100, "batches {batches}");
  // produced up to the last gate; the remainder sits in the accumulator
  let produced = cfg.q_dust_kgs * 1e3 * host.last_gate_t_s.unwrap();
  let rel = ((emitted_mass + host.acc.mass_g) - produced).abs() / produced;
  assert!(rel < 1e-5, "mass rel err {rel}");
  let ds = host.draw_state().unwrap();
  assert!(ds.live_count > 0 && ds.compute_wait > 0 && ds.mean_cluster_flux > 0.0);

  // restore: everything must be re-emitted, oldest first, before it is drawable again
  host.ring.invalidate_gpu();
  assert!(host.draw_state().is_none());
  let n_live = host.ring.batches.len();
  let mut reemitted = 0;
  while reemitted < n_live {
    let out = host.tick(jet_at(t), now, &cfg);
    reemitted += out.len().min(REEMIT_PER_TICK);
    host.ring.mark_submitted(1_000_000);
    t += dt_tick_s;
    now += dt_tick_us;
  }
  assert!(host.ring.batches.iter().all(|b| b.ready != READY_NEEDS_EMIT));

  // scrub back by a day: newer batches dropped, accumulator restarted
  let t_back = t - 86400.0;
  host.tick(jet_at(t_back), now, &cfg);
  assert!(host.ring.batches.iter().all(|b| b.t_end_s <= t_back));
  assert_eq!(host.acc.mass_g, 0.0);
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
      if let Some(p) = plan_batch(&mut acc, q, dt, t, ttl, ring.capacity, ring.free_slots()) {
        emitted += p.mass_g;
        ring.push_batch(desc(p.count), t, p.mass_g);
      }
      assert!(ring.live() <= capacity);
    }
    let produced = q * 1e3 * dt * 2000.0;
    assert!(((emitted + acc.mass_g) - produced).abs() < 1e-6 * produced);
    // steady state uses ~BUDGET_SAFETY of the ring
    let fill = ring.live() as f64 / capacity as f64;
    assert!(fill > 0.6 && fill <= 0.85, "capacity {capacity}: fill {fill}");
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
    assert!(live as u64 * k as u64 <= cap as u64 * CHILDREN_PER_CLUSTER as u64 || k == CHILDREN_PER_CLUSTER);
  }
}
