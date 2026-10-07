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
  // 64 streams: one grain per size stratum per time sample, the whole β range
  let mut b = test_batch(2048, 60.0);
  b.mass_params[3] = batch_streams_word(6, false);
  let (frame, rc) = frame_after(10.0);
  let anti_sun = scale(rc, 1.0 / norm(rc));
  let mut pts: alloc::vec::Vec<(f32, f64)> = alloc::vec::Vec::new();
  for j in 0..b.count {
    let c = emit_cluster(&b, j);
    let e = evaluate_cluster(&c, j, &frame);
    assert!(e.age_id_dbeta_flux[3] > 0.0);
    assert_eq!(render_slot(e.age_id_dbeta_flux[1]), j);
    assert!(render_live(e.age_id_dbeta_flux[1]));
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
  assert_eq!(core::mem::size_of::<DustBatch>(), 208);
  assert_eq!(core::mem::size_of::<DustFrame>(), 64);
}

/// Particles split, they never grow (blobber.rdc: 84 % of the splats at the old 48 px clamp): the
/// drawn radius is constant, and a cloud covering more pixels asks for more children instead.
#[test]
fn lod_splits_clusters_instead_of_enlarging_children() {
  assert_eq!(lod_want(0.0), 1.0);
  assert_eq!(lod_want(DUST_CHILD_PX), 1.0);
  assert!((lod_want(10.0 * DUST_CHILD_PX) - 100.0).abs() < 1e-3);
  assert_eq!(lod_children(lod_want(0.1), 1.0), 1);
  assert_eq!(lod_children(100.0, 1.0), 100);
  assert_eq!(lod_children(100.0, 0.25), 25);
  // at most TRACER_CHILD children: indices 0..TRACER_CHILD, the last index is the tracer's
  assert_eq!(lod_children(1e9, 1.0), TRACER_CHILD);
  assert_eq!(
    lod_children(100.0, 1e-6),
    1,
    "an on-screen cluster always draws"
  );
  // the same cluster aging (spread ∝ age): footprint fixed, children grow with the projected area
  let (units, p11, w, px_to_ndc_y) = (1e-3f32, 3.0f32, 100.0f32, 2.0 / 720.0);
  let mut prev_k = 0;
  for spread_m in [1.0e3f32, 1.0e4, 1.0e5] {
    let (r_px, _) = splat_footprint(DUST_CHILD_PX, units, p11, w, px_to_ndc_y);
    assert_eq!(r_px, DUST_CHILD_PX);
    let spread_px = spread_m * units * p11 / w / px_to_ndc_y;
    let k = lod_children(lod_want(spread_px), 1.0);
    assert!(k >= prev_k, "children must not shrink as the cloud grows");
    prev_k = k;
  }
  assert!(prev_k > 100);
}

/// The budget share comes from this frame's demand (no feedback lag, no ramp after a view change:
/// the former controller started at λ = 0.05 and grew by √ratio per readback 4 frames late, the
/// "adjustment phase"): one evaluation fills 80–95 % of the budget whatever the demand, and a
/// light demand gets λ = 1.
#[test]
fn lod_share_fills_the_budget_in_the_same_frame() {
  let budget = 1_000_000u32;
  for (n, spread) in [
    (40_000u32, 1000u32),
    (200_000, 30),
    (900_000, 3),
    (5_000, 4000),
  ] {
    let wants: alloc::vec::Vec<f32> = (0..n).map(|i| 1.0 + (i % spread) as f32).collect();
    let mut hist = [0u32; 2 * LOD_HIST_BINS];
    for &w in &wants {
      lod_hist_push(&mut hist, lod_demand(w));
    }
    let lambda = lod_lambda_from(budget, &hist, n, 1.0);
    let attempted: u32 =
      wants.iter().map(|&w| lod_children(w, lambda)).sum::<u32>() + n / TRACER_EVERY;
    let fill = attempted as f32 / budget as f32;
    if lambda < 1.0 {
      assert!(
        (0.8..=0.95).contains(&fill),
        "{n} clusters: fill {fill} at λ {lambda}"
      );
    } else {
      assert!(fill <= 0.95, "{n} clusters: fill {fill}");
    }
  }
  let mut light = [0u32; 2 * LOD_HIST_BINS];
  assert_eq!(lod_lambda_from(budget, &light, 0, 1.0), 1.0);
  for _ in 0..10 {
    lod_hist_push(&mut light, 10);
  }
  assert_eq!(
    lod_lambda_from(budget, &light, 10, 1.0),
    1.0,
    "light demand: everything"
  );
  assert_eq!(
    lod_lambda_from(budget, &light, 10, 0.25),
    0.25,
    "the cap holds"
  );
}

/// Fewer dots keep covering the footprint: `k·r²` stays `want·DUST_CHILD_PX²` (until the radius
/// cap), so a tight budget or a near view is a softer fog, not sparse bright speckle (`near.rdc`);
/// fully sampled clusters keep the 1.5 px dots.
#[test]
fn splats_keep_the_footprint_covered_at_any_budget_share() {
  for want in [1.0f32, 40.0, 900.0] {
    for lambda in [1.0f32, 0.3, 0.05] {
      let k = lod_children(want, lambda);
      let r = splat_radius_px(want, k);
      let covered = k as f32 * r * r;
      let expect = want.max(k as f32) * DUST_CHILD_PX * DUST_CHILD_PX;
      if r < DUST_CHILD_PX_MAX {
        assert!(
          (covered / expect - 1.0).abs() < 1e-3,
          "want {want} λ {lambda}: k {k} r {r}"
        );
      }
      if lambda == 1.0 {
        assert_eq!(r, DUST_CHILD_PX);
      }
    }
  }
  assert_eq!(splat_radius_px(1e9, 1), DUST_CHILD_PX_MAX);
}

/// Orthographic LOD push constants over a `half_h_m` × `half_h_m` view (square, `px` pixels).
fn ortho_lod_pc(
  half_h_m: f32,
  px: u32,
  budget: u32,
  lambda: f32,
  live: u32,
) -> DustLodPushConstants {
  let units = 1e-3f32;
  let p = 1.0 / (half_h_m * units);
  let mut mvp = [0.0f32; 16];
  mvp[0] = p * units;
  mvp[5] = p * units;
  mvp[10] = 1e-9;
  mvp[15] = 1.0;
  DustLodPushConstants {
    render: 0,
    header: 0,
    tiles: 0,
    list: 0,
    live_count: live,
    budget,
    lambda,
    // production: exposure (~1e7) / tile unit (~1e-5 white)
    tile_scale: 1e12,
    mvp,
    params: [units, p, p, 2.0 / px as f32],
  }
}

fn render_cluster(slot: u32, pos: [f32; 3], spread: f32, flux: f32) -> DustRenderCluster {
  DustRenderCluster {
    pos_size: [pos[0], pos[1], pos[2], spread],
    // distinct child patterns per cluster (the id lives in the dbeta field, 0 half-spread)
    age_id_dbeta_flux: [
      86400.0,
      f32::from_bits(slot),
      pack_child_id(0.0, slot),
      flux,
    ],
  }
}

/// `lod_evaluate` (the `dust_lod.comp` mirror): off-screen clusters draw nothing, each on-screen
/// cluster gets exactly `k` list entries carrying `flux / k`, the budget is never exceeded, and
/// the clusters dropped at the budget are marked (flux 0) so the shader skips them.
#[test]
fn lod_evaluate_spends_the_budget_on_screen_only() {
  let half = 1.0e6f32;
  let mut render: alloc::vec::Vec<DustRenderCluster> = (0..1000u32)
    .map(|i| {
      // half on screen, half 5 view widths away
      let x = if i % 2 == 0 {
        (u01(pcg(i)) - 0.5) * half
      } else {
        5.0 * half
      };
      render_cluster(i, [x, (u01(pcg(i + 7)) - 0.5) * half, 0.0], 2.0e4, 10.0)
    })
    .collect();
  let fluxes: alloc::vec::Vec<f32> = render.iter().map(|r| r.age_id_dbeta_flux[3]).collect();
  let pc = ortho_lod_pc(half, 512, 1_000_000, 1.0, render.len() as u32);
  let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
  let mut list = alloc::vec::Vec::new();
  let out = lod_evaluate(&mut render, &pc, 0, 0.0, &mut tiles, &mut list);
  assert_eq!(out.instances as usize, list.len());
  assert_eq!(
    out.attempted, out.instances,
    "nothing dropped under a large budget"
  );
  let mut per_cluster = alloc::vec![0u32; render.len()];
  for &e in &list {
    let c = (e & ((1 << LOD_CLUSTER_BITS) - 1)) as usize;
    let child = e >> LOD_CLUSTER_BITS;
    assert!(child < MAX_CHILDREN_PER_CLUSTER);
    per_cluster[c] += 1;
  }
  for (i, r) in render.iter().enumerate() {
    if i % 2 == 1 {
      assert_eq!(per_cluster[i], 0, "off screen cluster {i} drawn");
      assert_eq!(r.age_id_dbeta_flux[3], 0.0);
    } else {
      let k = per_cluster[i];
      assert!(k >= 1);
      // energy: k children of flux / k
      let total = r.age_id_dbeta_flux[3] * k as f32;
      assert!((total / fluxes[i] - 1.0).abs() < 1e-5);
    }
  }
  assert!(tiles.iter().any(|&t| t > 0));

  // tight budget: never exceeded, dropped clusters marked
  let mut render2: alloc::vec::Vec<DustRenderCluster> =
    (0..1000u32).map(|i| render_cluster(i, [0.0, 0.0, 0.0], 2.0e5, 10.0)).collect();
  // (the share keeps any demand within the budget unless more clusters are on screen than it holds)
  let pc2 = ortho_lod_pc(half, 512, 500, 1.0, 1000);
  let mut list2 = alloc::vec::Vec::new();
  let out2 = lod_evaluate(&mut render2, &pc2, 0, 0.0, &mut tiles, &mut list2);
  assert!(out2.instances <= 500 && out2.attempted > 500);
  let drawn = render2.iter().filter(|r| r.age_id_dbeta_flux[3] > 0.0).count();
  let dropped = render2.len() - drawn;
  assert!(dropped > 0);
  assert_eq!(
    list2.len() as u32,
    out2.instances,
    "only drawn clusters have entries"
  );
}

/// White point from the tile grid: a high percentile of the non-empty tiles, scale-free (the tile
/// unit cancels), empty grid = no measurement; adaptation moves a fixed fraction in log space.
#[test]
fn white_point_measures_the_brightest_dust_and_adapts_smoothly() {
  let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
  assert_eq!(white_point_from_tiles(&tiles, 1.0), None);
  for (i, t) in tiles.iter_mut().enumerate().take(1000) {
    *t = (i + 1) as u32;
  }
  let w = white_point_from_tiles(&tiles, 1.0).unwrap();
  assert!((w - 990.0).abs() <= 1.0, "p99 of 1..=1000: {w}");
  let w2 = white_point_from_tiles(&tiles, 0.5).unwrap();
  assert!((w2 - 0.5 * w).abs() < 1e-3);
  // eye adaptation: log-space step, converges, ignores garbage
  assert_eq!(adapt_white(0.0, 2.0), 2.0);
  assert_eq!(adapt_white(2.0, f32::NAN), 2.0);
  assert_eq!(adapt_white(2.0, 0.0), 2.0);
  let mut x = 1.0f32;
  for _ in 0..30 {
    let next = adapt_white(x, 1e4);
    assert!(next >= x && next <= 1e4 * 1.0001);
    // a frame covers WHITE_ADAPT_RATE of the log distance, never overshoots
    let expect = (x.ln() + (1e4f32.ln() - x.ln()) * WHITE_ADAPT_RATE).exp();
    assert!((next / expect - 1.0).abs() < 1e-4);
    x = next;
  }
  for _ in 0..100 {
    x = adapt_white(x, 1e4);
  }
  assert!((x / 1e4 - 1.0).abs() < 1e-3);
  assert_eq!(tile_counts(0.0), 0);
  assert_eq!(tile_counts(-1.0), 0);
  assert_eq!(tile_counts(f32::NAN), 0);
  assert_eq!(tile_counts(1e20), 100_000_000);
}

/// The white point is a property of the view, not of the zoom: the same coma seen 30× wider has a
/// lower white point (blobber.rdc: the ±776 km view saturated everything under the fixed exposure,
/// the 30× wider one showed coma and tail). Measured through `lod_evaluate`'s tiles.
#[test]
fn white_point_follows_the_view() {
  // a 1/ρ coma: clusters uniform in radius out to 5e6 m
  let cloud = |n: u32| -> alloc::vec::Vec<DustRenderCluster> {
    (0..n)
      .map(|i| {
        let r = 5.0e6 * u01(pcg(i * 3 + 1));
        let a = 2.0 * core::f32::consts::PI * u01(pcg(i * 3 + 2));
        render_cluster(i, [r * a.cos(), r * a.sin(), 0.0], 1.0e3 + 0.05 * r, 1.0)
      })
      .collect()
  };
  let measure = |half: f32| -> f32 {
    let mut render = cloud(50_000);
    let pc = ortho_lod_pc(half, 720, 4_000_000, 1.0, 50_000);
    let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
    let mut list = alloc::vec::Vec::new();
    lod_evaluate(&mut render, &pc, 0, 0.0, &mut tiles, &mut list);
    white_point_from_tiles(&tiles, 1.0).unwrap()
  };
  let (near, far) = (measure(2.0e5), measure(6.0e6));
  assert!(near > 5.0 * far, "near {near} far {far}");
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
  assert!(turned > 3000.0, "the window covers a turn: the site moved {turned} m");
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

/// ticks a system until every tier is caught up (or `max` ticks), marking batches submitted
fn fill_system(sys: &mut DustSystemState, t: f64, cfg: &DustEmitConfig, max: usize) {
  for i in 0..max {
    sys.tick(t, &orbit_jet, &|_| *cfg);
    sys.mark_submitted(i as u64 + 1);
    if sys.stats().iter().all(|s| s.caught_up) {
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
    assert!(t.prestart);
  }
  assert_eq!(sys.tiers[1].band.min_ttl, sys.tiers[0].band.max_ttl);
  assert_eq!(sys.tiers[2].band.min_ttl, sys.tiers[1].band.max_ttl);

  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  let stats = sys.stats();
  assert!(stats.iter().all(|s| s.caught_up), "{stats:?}");
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

/// `dust.frag`: opacity at `r2 = |uv|²` of a splat with peak `v_opacity`
fn frag_opacity(v_opacity: f32, r2: f32) -> f32 {
  const GAUSS_NORM: f32 = 1.297;
  if r2 > 1.0 {
    0.0
  } else {
    v_opacity * (-4.0 * r2).exp() * GAUSS_NORM
  }
}

/// Rasterizes clusters `(x_m, y_m, spread_m, flux_m2)` through the shader mirror in an
/// orthographic view of half-height `half_h_m` (square viewport `px`): each cluster is split by the
/// LOD (`lambda`) into children at `child_offset` carrying `flux / k`, drawn at the fixed
/// footprint. Sums the fragment opacities per pixel (no saturation: the optical depth regime).
fn rasterize_ortho(
  clusters: &[(f32, f32, f32, f32)],
  half_h_m: f32,
  px: usize,
  exposure: f32,
  lambda: f32,
) -> alloc::vec::Vec<f32> {
  let mut img = alloc::vec![0.0f32; px * px];
  let (units_per_m, p11, clip_w, px_to_ndc_y) =
    (1e-3, 1.0 / (half_h_m * 1e-3), 1.0, 2.0 / px as f32);
  let px_per_m = px as f32 / (2.0 * half_h_m);
  for (id, &(x, y, spread, flux)) in clusters.iter().enumerate() {
    let k = lod_children(lod_want(spread * px_per_m), lambda);
    let (r_px, r_draw_m) = splat_footprint(DUST_CHILD_PX, units_per_m, p11, clip_w, px_to_ndc_y);
    let peak = splat_opacity(exposure, flux / k as f32, r_draw_m);
    for child in 0..k {
      let o = child_offset(id as u32, child, spread, 0.0, 0.0, [0.0; 4]);
      let (cx, cy) = (
        (x + o[0]) * px_per_m + px as f32 * 0.5,
        (y + o[1]) * px_per_m + px as f32 * 0.5,
      );
      let (x0, x1) = (
        (cx - r_px).floor().max(0.0) as usize,
        ((cx + r_px).ceil().max(0.0) as usize).min(px),
      );
      let (y0, y1) = (
        (cy - r_px).floor().max(0.0) as usize,
        ((cy + r_px).ceil().max(0.0) as usize).min(px),
      );
      for j in y0..y1 {
        for i in x0..x1 {
          let (u, v) = ((i as f32 + 0.5 - cx) / r_px, (j as f32 + 0.5 - cy) / r_px);
          img[j * px + i] += frag_opacity(peak, u * u + v * v);
        }
      }
    }
  }
  img
}

/// deterministic uniform cloud over a `side_m` square, `n` clusters of cross-section `flux`
fn uniform_cloud(
  n: usize,
  side_m: f32,
  spread_m: f32,
  flux: f32,
) -> alloc::vec::Vec<(f32, f32, f32, f32)> {
  (0..n as u32)
    .map(|i| {
      let (a, b) = (u01(pcg(i * 2 + 1)), u01(pcg(i * 2 + 2)));
      ((a - 0.5) * side_m, (b - 0.5) * side_m, spread_m, flux)
    })
    .collect()
}

/// Pixels show `exposure · τ`, τ the dust optical depth: zooming in on a cloud does not dim it
/// (`late_near.rdc` was fainter than `late_far.rdc` under the former 1/r_px stretch), and neither
/// the fixed footprint nor the children count changes it (energy conserving split).
#[test]
fn splat_brightness_is_optical_depth_at_any_zoom() {
  let exposure = 1e3;
  let side = 2.0e6; // 2000 km cloud
  let n = 20_000;
  let flux = 1.0e3; // m² per cluster
  let tau_true = n as f32 * flux / (side * side);
  let px = 256;
  for &(spread, lambda, label) in &[
    (2.0e3f32, 1.0f32, "sub-pixel clouds"),
    (2.0e4, 1.0, "clouds split in children"),
    (2.0e4, 0.05, "few children (tight budget)"),
  ] {
    let cloud = uniform_cloud(n, side, spread, flux);
    let mut means = alloc::vec::Vec::new();
    for half_h in [6.0e5f32, 1.5e5] {
      let img = rasterize_ortho(&cloud, half_h, px, exposure, lambda);
      let m: f32 = img.iter().sum::<f32>() / img.len() as f32;
      means.push(m);
      let rel = (m / (exposure * tau_true) - 1.0).abs();
      assert!(
        rel < 0.1,
        "{label}, half-height {half_h} m: mean {m} vs exposure·τ {}",
        exposure * tau_true
      );
    }
    let zoom_rel = (means[0] / means[1] - 1.0).abs();
    assert!(
      zoom_rel < 0.1,
      "{label}: brightness changes with zoom {means:?}"
    );
  }
}

/// The drawn radius is [`DUST_CHILD_PX`] at any depth, and its size in metres inverts the
/// projection: twice as far (w ×2) at a twice longer focal (p11 ×2) is the same child.
#[test]
fn splat_footprint_inverts_the_projection() {
  let (r_px, r_m) = splat_footprint(DUST_CHILD_PX, 1e-3, 3.0, 100.0, 2.0 / 720.0);
  assert_eq!(r_px, DUST_CHILD_PX);
  // px_per_unit = 3 / 100 / (2/720) = 10.8 px per km → 1.5 px = 138.9 m
  assert!((r_m - 1.5 / 10.8 * 1e3).abs() < 0.1, "{r_m}");
  let (r_px2, r_m2) = splat_footprint(DUST_CHILD_PX, 1e-3, 6.0, 200.0, 2.0 / 720.0);
  assert!((r_px - r_px2).abs() < 1e-6 && (r_m - r_m2).abs() < 1e-2);
  let (_, r_far) = splat_footprint(DUST_CHILD_PX, 1e-3, 3.0, 200.0, 2.0 / 720.0);
  assert!(
    (r_far / r_m - 2.0).abs() < 1e-4,
    "twice as far: twice the metres"
  );
}

/// Child identity depends only on `(id, child)`: raising `k` keeps children `0..k` in place.
#[test]
fn child_offsets_are_stable_under_lod_changes() {
  let g = [0.0, 1.0, 0.0, 1e-3];
  let a = child_offset(42, 3, 1.0e4, 0.01, 1.0e5, g);
  assert_eq!(a, child_offset(42, 3, 1.0e4, 0.01, 1.0e5, g));
  assert_ne!(a, child_offset(42, 4, 1.0e4, 0.01, 1.0e5, g));
  assert_ne!(a, child_offset(43, 3, 1.0e4, 0.01, 1.0e5, g));
  // isotropic scatter of spread·N(0,1) per axis, at any age (the reshaping keeps the variance)
  let n = 4000u32;
  for age in [0.0f32, 3600.0, 86400.0, 3.0e7] {
    let var: f32 = (0..n)
      .map(|c| child_offset(7, c % MAX_CHILDREN_PER_CLUSTER, 1.0, 0.0, age, [0.0; 4]))
      .chain((0..n).map(|c| child_offset(c, 0, 1.0, 0.0, age, [0.0; 4])))
      .map(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
      .sum::<f32>()
      / (2 * n) as f32;
    assert!(
      (var / 3.0 - 1.0).abs() < 0.1,
      "age {age}: per-axis variance {}",
      var / 3.0
    );
  }
}

/// The children of a cluster move relative to each other as it ages: the cloud at two ages is
/// not the same blob scaled (a fixed `spread·N3` is: "a still image sliding").
#[test]
fn children_reshape_with_age() {
  let k = 64u32;
  let cloud = |age: f32| -> alloc::vec::Vec<[f32; 3]> {
    // unit spread: only the shape, not the size
    (0..k).map(|c| child_offset(1234, c, 1.0, 0.0, age, [0.0; 4])).collect()
  };
  // normalized offsets (each cloud scaled to unit RMS): equal iff the clouds differ by a scaling
  let normalized = |pts: &[[f32; 3]]| -> alloc::vec::Vec<[f32; 3]> {
    let rms = (pts.iter().map(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sum::<f32>()
      / pts.len() as f32)
      .sqrt();
    pts.iter().map(|o| [o[0] / rms, o[1] / rms, o[2] / rms]).collect()
  };
  // mean displacement of the normalized children between two ages (√6 ≈ 2.45 when unrelated)
  let moved = |a: f32, b: f32| -> f32 {
    let (x, y) = (normalized(&cloud(a)), normalized(&cloud(b)));
    x.iter()
      .zip(&y)
      .map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt())
      .sum::<f32>()
      / k as f32
  };
  let day = 86400.0;
  assert!(
    moved(day, 4.0 * day) > 0.5,
    "1 → 4 days: {}",
    moved(day, 4.0 * day)
  );
  assert!(
    moved(day, 30.0 * day) > 0.5,
    "1 → 30 days: {}",
    moved(day, 30.0 * day)
  );
  // smooth: a minute later the children have barely moved
  assert!(
    moved(day, day + 60.0) < 0.01,
    "1 day → +1 min: {}",
    moved(day, day + 60.0)
  );
}

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
  // youngest tier only (one budget-limited tick), then everything
  sys.tick(0.0, &orbit_jet, &|_| cfg);
  sys.mark_submitted(1);
  let early: alloc::vec::Vec<f32> = sys.draw_states().iter().map(|s| s.tau_ref).collect();
  fill_system(&mut sys, 0.0, &cfg, MAX_SEEK_PASSES);
  let late = sys.draw_states();
  assert_eq!(late.len(), 3);
  assert!(early.iter().chain(late.iter().map(|s| &s.tau_ref)).all(|&t| t == tau as f32));
}

/// 8-bit fallback (`dust.frag`): stochastic rounding keeps the expected opacity of faint splats
/// exactly, where round-to-nearest drops everything below 1/510 (the old dust tail).
#[test]
fn stochastic_rounding_keeps_faint_optical_depth() {
  let n = 4096;
  for v in [1.0e-4f32, 1.5e-3, 3.0e-3, 0.37] {
    let mean: f64 = (0..n)
      .map(|i| stochastic_round_8bit(v, (i as f32 + 0.5) / n as f32) as f64)
      .sum::<f64>()
      / n as f64;
    assert!((mean - v as f64).abs() < 1e-6, "v {v}: mean {mean}");
  }
  // round to nearest (u = 0.5) loses a faint splat entirely
  assert_eq!(stochastic_round_8bit(1.5e-3, 0.5), 0.0);
  // a pixel covered by 300 faint splats with decorrelated offsets keeps its optical depth
  let v = 1.0e-3f32;
  let sum: f32 = (0..300u32).map(|k| stochastic_round_8bit(v, u01(pcg(k ^ 0xA511_E9B3)))).sum();
  assert!(
    (sum / (300.0 * v) - 1.0).abs() < 0.25,
    "sum {sum} vs {}",
    300.0 * v
  );
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
/// cone cell and speed draw up to the small per-cluster jitters, while different streams differ;
/// its size stratum steps by one per time sample, and every time sample covers each size stratum
/// once (stratified mass).
#[test]
fn streams_keep_their_cell_and_speed_and_step_their_size() {
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
  let ln_r = (a.size_params[1] / a.size_params[0]).ln();
  let stratum_of = |c: &DustCluster| {
    (((c.misc[1] / a.size_params[0]).ln() / ln_r * n as f32) as usize).min(n as usize - 1)
  };
  let (mut other_angle, mut pairs) = (0.0f64, 0);
  for i in 0..8u32 {
    let mut hit = alloc::vec![false; n as usize];
    for s in 0..n {
      let j = i * n + s;
      let (ca, cb) = (emit_cluster(&a, j), emit_cluster(&b, (7 - i) * n + s));
      // the size steps one stratum per time sample, from the stream's rank (same in every
      // window), jittered by ≤ STREAM_SIZE_JITTER of the stratum
      let k = stream_stratum(&a, j);
      assert_eq!(k, (stream_stratum(&a, s) + i) % n, "stream {s} sample {i}");
      assert_eq!(stream_stratum(&b, (7 - i) * n + s), (stream_stratum(&a, s) + 7 - i) % n);
      assert_eq!(stratum_of(&ca), k as usize, "stream {s}: size outside its stratum");
      assert!(!hit[k as usize], "sample {i}: stratum {k} twice");
      hit[k as usize] = true;
      let (ea, eb) = (ej(&ca), ej(&cb));
      let angle = dot(unit(ea), unit(eb)).clamp(-1.0, 1.0).acos();
      assert!(angle < 0.12, "stream {s}: direction moved by {angle} rad");
      // the stream's speed draw is shared (relative to the mean speed of each one's size), the
      // jitter is STREAM_SPEED_JITTER of the spread (5σ of the difference)
      let v_mean = |c: &DustCluster| {
        a.vel_params[0] as f64 * (a.vel_params[2] as f64 / c.misc[1] as f64).sqrt()
      };
      let dv = (norm(ea) / v_mean(&ca) - norm(eb) / v_mean(&cb)).abs();
      assert!(
        dv < 5.0 * 1.42 * STREAM_SPEED_JITTER as f64 * 0.25,
        "stream {s}: speed draw changed by {dv}"
      );
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

/// A cluster draws the same children whether its tier was filled by a seek or played up to the
/// same time, although it sits in another ring slot: children are keyed by the emission record.
#[test]
fn children_are_identical_after_seek_and_play() {
  let ttl = 86400.0;
  let cfg = test_cfg(1.5e-3, ttl);
  let t_end = 3.0 * ttl;
  // seek: straight to t_end
  let mut seek = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  fill_system(&mut seek, t_end, &cfg, MAX_SEEK_PASSES);
  // play: filled at 0, then played to t_end (other emission history: other slots)
  let mut play = DustSystemState::with_tiers(RING_CAPACITY_LOW, 3);
  fill_system(&mut play, 0.0, &cfg, MAX_SEEK_PASSES);
  let mut t = 0.0;
  let mut seq = 1_000;
  while t < t_end {
    t = (t + 600.0).min(t_end);
    play.tick(t, &orbit_jet, &|_| cfg);
    seq += 1;
    play.mark_submitted(seq);
  }
  fill_system(&mut play, t_end, &cfg, MAX_SEEK_PASSES);

  let g = [0.3, 0.9, 0.3, 1e-3];
  let (mut compared, mut moved_slots) = (0, 0);
  for (ts, tp) in seek.tiers.iter().zip(&play.tiers) {
    let ds = ts.draw_state().unwrap();
    let (ws, wp) = (closed_windows(ts), closed_windows(tp));
    // the live batch of window `k` (with its ring placement)
    let live =
      |h: &DustHostState, k: i64| h.ring.batches.iter().find(|x| x.window == k).unwrap().desc;
    for (k, a) in &ws {
      let Some((_, b)) = wp.iter().find(|(kp, _)| kp == k) else {
        continue;
      };
      // same window → same descriptor up to the ring placement
      assert_eq!(a, b, "window {k}");
      let (sa, sb) = (live(ts, *k), live(tp, *k));
      for j in (0..a.count).step_by(7) {
        let (slot_a, slot_b) = (
          sa.first_index.wrapping_add(j) & sa.ring_mask,
          sb.first_index.wrapping_add(j) & sb.ring_mask,
        );
        moved_slots += (slot_a != slot_b) as u32;
        let ra = evaluate_cluster(&emit_cluster(&sa, j), slot_a, &ds.frame);
        let rb = evaluate_cluster(&emit_cluster(&sb, j), slot_b, &ds.frame);
        if !(ra.age_id_dbeta_flux[3] > 0.0) {
          continue;
        }
        for child in [0u32, 1, 17, 255] {
          let off = |r: &DustRenderCluster| {
            child_offset(
              child_id(r.age_id_dbeta_flux[2]),
              child,
              r.pos_size[3],
              r.age_id_dbeta_flux[2],
              r.age_id_dbeta_flux[0],
              g,
            )
          };
          assert_eq!(off(&ra), off(&rb), "window {k} cluster {j} child {child}");
        }
        compared += 1;
      }
    }
  }
  assert!(compared > 100, "compared {compared}");
  // the histories differ: many clusters sit in other slots (a slot-keyed child hash moved them)
  assert!(
    moved_slots > 0,
    "no cluster changed slot: the test does not exercise the seek"
  );
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

/// LOD push constants of an orthographic view (metres, `units` 1) of half size `half`
fn ortho_pc(mvp: [f32; 16], half: f32, px: u32, lambda: f32, live: u32) -> DustLodPushConstants {
  DustLodPushConstants {
    render: 0,
    header: 0,
    tiles: 0,
    list: 0,
    live_count: live,
    budget: 1 << 24,
    lambda,
    tile_scale: 1.0,
    mvp,
    params: [1.0, 1.0 / half, 1.0 / half, 2.0 / px as f32],
  }
}

/// Stream counts follow the tier capacity (fewer streams on small rings, so a window keeps
/// [`STREAM_MIN_SAMPLES`] time samples), and every planned batch holds whole time samples.
#[test]
fn stream_counts_follow_the_tier_capacity() {
  for (cap, shift) in [
    (131_072u32, 6u32),
    (65_536, 5),
    (16_384, 3),
    (8_192, 2),
    (4_096, 1),
    (2_048, 0),
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

/// In a tier's render buffer the stream predecessor `r − S` of a cluster is the same stream's
/// previous time sample (older, one sample spacing), across batch boundaries too.
#[test]
fn streak_predecessor_is_the_same_streams_previous_sample() {
  let cfg = test_cfg(1.5e-3, 86400.0);
  let host = run_ticks([1.3 * 86400.0; 8], &orbit_jet, &cfg);
  let ds = host.draw_state().unwrap();
  let render = tier_render(&host, &ds.frame);
  let n = 1usize << host.stream_shift();
  assert_eq!(n, 16);
  // (window, in-batch index) of every render index, in ring order
  let ids: alloc::vec::Vec<(i64, u32, u32)> = host
    .ring
    .batches
    .iter()
    .flat_map(|b| (0..b.count).map(move |j| (b.window, j, b.count)))
    .take(render.len())
    .collect();
  let spacing =
    DustHostState::window_len_s(cfg.ttl_s) / (host.clusters_per_window() / n as u32) as f64;
  let (mut linked, mut crossed) = (0, 0);
  for r in 0..render.len() {
    // culled clusters (outside the age band) have flux 0: the LOD never asks for their streak
    if !render_live(render[r].age_id_dbeta_flux[1]) {
      continue;
    }
    let Some(p) = streak_pred(&render, r) else {
      assert!(
        r < n
          || !render_live(render[r - n].age_id_dbeta_flux[1])
          || stream_break(render[r].age_id_dbeta_flux[2])
      );
      continue;
    };
    assert_eq!(p, r - n);
    let ((wr, jr, _), (wp, jp, cp)) = (ids[r], ids[p]);
    assert_eq!(
      jr as usize % n,
      jp as usize % n,
      "render {r}: another stream"
    );
    if wr == wp {
      assert_eq!(jp + n as u32, jr);
    } else {
      // the previous window's last sample
      assert_eq!((wp + 1, jp as usize / n), (wr, cp as usize / n - 1));
      crossed += 1;
    }
    let da = (render[p].age_id_dbeta_flux[0] - render[r].age_id_dbeta_flux[0]) as f64;
    assert!(
      da > 0.0 && da < 2.0 * spacing,
      "render {r}: age step {da} s (sample {spacing} s)"
    );
    linked += 1;
  }
  // one sample in n starts a new size cycle (`stream_stratum` wraps): a break
  assert!(
    linked as f64 > (0.95 - 1.0 / n as f64) * render.len() as f64,
    "{linked} of {}",
    render.len()
  );
  assert!(crossed > 10, "streaks cross batch boundaries: {crossed}");
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
  // night breaks: the size wrap of a stream (largest stratum → smallest) also breaks it
  let wrap = |b: &DustBatch, j: u32| stream_stratum(b, j) == 0;
  let breaks = |b: &DustBatch, s: u32| {
    (0..b.count / n)
      .filter(|i| {
        let j = i * n + s;
        let broken = stream_break(emit_cluster(b, j).misc[3]);
        assert_eq!(broken, stream_dark_before(b, j) || wrap(b, j), "cluster {j}");
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
        dark > STREAM_BREAK_TURNS as f64 * P_ROT || wrap(&b, i * n),
        "sample {i}: dark {dark} s"
      );
    }
  }
  // always lit: no break, unless the previous window is missing (first sample only)
  let mut lit = test_batch(n * 48, dur);
  lit.mass_params[3] = batch_streams_word(4, false);
  assert!((0..lit.count).all(|j| stream_break(emit_cluster(&lit, j).misc[3]) == wrap(&lit, j)));
  lit.mass_params[3] = batch_streams_word(4, true);
  for j in 0..lit.count {
    assert_eq!(
      stream_break(emit_cluster(&lit, j).misc[3]),
      j < n || wrap(&lit, j),
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

/// Top-down image (along the spin axis) of a jet's dust on a nucleus spinning at `omega`: every
/// render cluster drawn through the LOD and the streak children (the `dust.vert` mirror), flux
/// binned per pixel. Narrow equatorial jet, one grain size, no speed spread, small β.
fn spiral_image(omega: f64, half: f32, px: usize) -> alloc::vec::Vec<f32> {
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
  let mut render = tier_render(&host, &ds.frame);
  let mut mvp = [0.0f32; 16];
  for k in 0..3 {
    mvp[4 * k] = e1[k] as f32 / half;
    mvp[4 * k + 1] = e2[k] as f32 / half;
    mvp[4 * k + 2] = axis[k] as f32 * 1e-9;
  }
  mvp[15] = 1.0;
  let pc = ortho_pc(mvp, half, px as u32, 1.0, render.len() as u32);
  let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
  let mut list = alloc::vec::Vec::new();
  lod_evaluate(&mut render, &pc, 0, 0.0, &mut tiles, &mut list);
  let mut img = alloc::vec![0.0f32; px * px];
  for &e in &list {
    let (i, c) = ((e & RENDER_SLOT_MASK) as usize, e >> LOD_CLUSTER_BITS);
    let p = child_position(&render, i, c, &mvp, pc.params, [0.0; 4]);
    let q = mvp_mul(&mvp, p);
    let (x, y) = (
      (q[0] * 0.5 + 0.5) * px as f32,
      (q[1] * 0.5 + 0.5) * px as f32,
    );
    if x >= 0.0 && y >= 0.0 && (x as usize) < px && (y as usize) < px {
      img[y as usize * px + x as usize] += render[i].age_id_dbeta_flux[3];
    }
  }
  img
}

/// Azimuth (rad) of the brightest direction (±12° smoothed) of `img` in annuli of 1 km from `r0`
/// to `r1`
fn ridge_azimuths(img: &[f32], half: f32, px: usize, r0: f32, r1: f32) -> alloc::vec::Vec<f64> {
  const BINS: usize = 90;
  let m_per_px = 2.0 * half / px as f32;
  let rings = ((r1 - r0) / 1000.0) as usize;
  let mut h = alloc::vec![0.0f64; rings * BINS];
  for y in 0..px {
    for x in 0..px {
      let (dx, dy) = (
        (x as f32 + 0.5) * m_per_px - half,
        (y as f32 + 0.5) * m_per_px - half,
      );
      let r = (dx * dx + dy * dy).sqrt();
      if r < r0 || r >= r1 {
        continue;
      }
      let ring = (((r - r0) / 1000.0) as usize).min(rings - 1);
      let a = (dy.atan2(dx) / (2.0 * core::f32::consts::PI) + 0.5) * BINS as f32;
      h[ring * BINS + (a as usize).min(BINS - 1)] += img[y * px + x] as f64;
    }
  }
  (0..rings)
    .filter_map(|k| {
      // circular ±3-bin box filter: a broad arm (one cone cell of dispersion) has a noisy top
      let raw = &h[k * BINS..(k + 1) * BINS];
      let row: alloc::vec::Vec<f64> = (0..BINS)
        .map(|b| (0..7).map(|d| raw[(b + BINS + d - 3) % BINS]).sum())
        .collect();
      let (b, v) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
      (*v > 0.0).then(|| ((b as f64 + 0.5) / BINS as f64 - 0.5) * 2.0 * core::f64::consts::PI)
    })
    .collect()
}

/// Unwrapped azimuths (consecutive steps taken in `(−π, π]`)
fn unwrap(az: &[f64]) -> alloc::vec::Vec<f64> {
  use core::f64::consts::PI;
  let mut out = alloc::vec![az[0]];
  for w in az.windows(2) {
    let d = w[1] - w[0];
    out.push(out.last().unwrap() + d - 2.0 * PI * ((d + PI) / (2.0 * PI)).floor());
  }
  out
}

/// The point of the redesign: a jet on a spinning nucleus draws a spiral. Seen along the spin axis,
/// the azimuth of the density ridge turns monotonically with radius, one turn per `v·P` (43 km for
/// 2 m/s and 6 h), over the ~3 turns of a 300 km field; without spin it is a straight fan.
#[test]
fn spinning_nucleus_draws_a_spiral() {
  let (half, px) = (150.0e3f32, 400usize);
  let p_rot = 6.0 * 3600.0;
  // AETHERVK_DUST_SPIRAL_PGM=<prefix>: writes both images (log scale) for a visual check, before the checks
  if let Ok(prefix) = std::env::var("AETHERVK_DUST_SPIRAL_PGM") {
    for (name, w) in [
      ("spin", 2.0 * core::f64::consts::PI / p_rot),
      ("still", 0.0),
    ] {
      let img = spiral_image(w, half, px);
      let max = img.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
      let mut out = alloc::format!("P2\n{px} {px}\n255\n").into_bytes();
      for y in (0..px).rev() {
        for x in 0..px {
          let v = img[y * px + x] / max;
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
  let az = ridge_azimuths(
    &spiral_image(2.0 * core::f64::consts::PI / p_rot, half, px),
    half,
    px,
    15e3,
    140e3,
  );
  assert!(az.len() > 110, "ridge found in {} of 125 annuli", az.len());
  let th = unwrap(&az);
  let turn = th[th.len() - 1] - th[0];
  let turns = turn.abs() / (2.0 * core::f64::consts::PI);
  let expect = 125e3 / (2.0 * p_rot);
  assert!(turns >= 2.0, "{turns:.2} turns");
  assert!(
    (turns / expect - 1.0).abs() < 0.2,
    "{turns:.2} turns, expected {expect:.2}"
  );
  // monotone: every 5 km (0.73 rad of arm, the ridge is a few 4° bins wide) turns the same way
  for (k, w) in th.windows(6).enumerate() {
    assert!(
      (w[5] - w[0]) * turn > 0.0,
      "annulus {k}: the ridge turns back ({:.2} rad)",
      w[5] - w[0]
    );
  }

  let still = ridge_azimuths(&spiral_image(0.0, half, px), half, px, 15e3, 140e3);
  assert!(still.len() > 110);
  let th0 = unwrap(&still);
  let spread = th0.iter().fold(0.0f64, |m, t| m.max((t - th0[0]).abs()));
  assert!(spread < 0.3, "no spin: the fan turns by {spread:.2} rad");
}

/// `S = 1` render clusters along the polyline `pts` (each the predecessor of the next one in
/// render order: oldest first), lateral spread `spread`, flux `flux`
fn streak_line(pts: &[[f32; 3]], spread: f32, flux: f32) -> alloc::vec::Vec<DustRenderCluster> {
  pts
    .iter()
    .enumerate()
    .rev()
    .enumerate()
    .map(|(r, (k, p))| DustRenderCluster {
      pos_size: [p[0], p[1], p[2], spread],
      age_id_dbeta_flux: [
        1000.0 * (k + 1) as f32,
        f32::from_bits(render_word(r as u32, 0)),
        pack_child_id(0.0, pcg(r as u32)),
        flux,
      ],
    })
    .collect()
}

/// The dots of the streaks carry exactly the cross-section of the visible parts: Σ over the dots
/// in view = Σ flux × the fraction of each segment inside the view, within 1 %.
#[test]
fn streak_dots_carry_the_visible_cross_section() {
  let half = 17.0e3f32;
  let pts: alloc::vec::Vec<[f32; 3]> =
    (0..25).map(|i| [-60.0e3 + 5.0e3 * i as f32, 3.0e3, 0.0]).collect();
  let mut render = streak_line(&pts, 1.0, 2.0);
  let mvp = ortho_mvp(half, [0.0, 0.0]);
  let pc = ortho_pc(mvp, half, 512, 1.0, render.len() as u32);
  let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
  let mut list = alloc::vec::Vec::new();
  lod_evaluate(&mut render, &pc, 0, 0.0, &mut tiles, &mut list);
  let mut seen = 0.0f64;
  for &e in &list {
    let (i, c) = ((e & RENDER_SLOT_MASK) as usize, e >> LOD_CLUSTER_BITS);
    let p = child_position(&render, i, c, &mvp, pc.params, [0.0; 4]);
    if p[0].abs() < half && p[1].abs() < half {
      seen += render[i].age_id_dbeta_flux[3] as f64;
    }
  }
  // 24 segments of 5 km along y = 3 km, the view spans x in ±17 km: 34 km of segments, 2 per km
  let expect = 2.0 * 34.0 / 5.0;
  assert!((seen / expect - 1.0).abs() < 0.01, "{seen} vs {expect}");
  // the first cluster has no predecessor: a point spread, off screen here
  assert_eq!(streak_pred(&render, 0), None);
  assert_eq!(render[0].age_id_dbeta_flux[3], 0.0);
}

/// Telescope zoom (broken_earth.rdc: a 34 km field inside a 5,000 km coma): a 2,000 km streak
/// crossing the view spends its dots on the visible part (the old clouds put < 1 dot on screen).
#[test]
fn telescope_zoom_streak_puts_its_dots_on_screen() {
  let half = 17.0e3f32;
  let mut render = streak_line(
    &[[-1.0e6, -0.995e6, 2.0e4], [1.0e6, 1.005e6, -2.0e4]],
    50.0,
    1.0,
  );
  let mvp = ortho_mvp(half, [0.0, 0.0]);
  let mut pc = ortho_pc(mvp, half, 512, 1.0, 2);
  pc.tile_scale = 1e12;
  let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
  let mut list = alloc::vec::Vec::new();
  let out = lod_evaluate(&mut render, &pc, 0, 0.0, &mut tiles, &mut list);
  let k = out.instances as usize;
  // the visible part crosses the view diagonally: ~512·√2 px in 1.5 px dots
  assert!(
    k > 300 && k <= MAX_CHILDREN_PER_CLUSTER as usize,
    "{k} dots"
  );
  let on = list
    .iter()
    .filter(|&&e| {
      let p = child_position(&render, 1, e >> LOD_CLUSTER_BITS, &mvp, pc.params, [0.0; 4]);
      p[0].abs() < half && p[1].abs() < half
    })
    .count();
  // the snapped range adds at most 2 grid steps (≤ 1/4) and the 5 % margin
  assert!(on as f64 > 0.7 * k as f64, "{on} of {k} dots on screen");
  // each carries flux · (t1 − t0) / k: the visible fraction (~1.7 %), not the whole streak
  let f = render[1].age_id_dbeta_flux[3] * k as f32;
  assert!(f > 0.017 && f < 0.03, "drawn fraction {f}");
  // white point tiles see the streak
  assert!(tiles.iter().any(|&t| t > 0));
}

/// Streak dots keep their place when the LOD changes `k` (golden-ratio prefix) and while the view
/// pans within a grid cell of the snapped range; a large pan re-samples.
#[test]
fn streak_dots_are_stable_under_lod_changes_and_small_pans() {
  let half = 17.0e3f32;
  let render = streak_line(&[[-1.0e6, 1.0e3, 0.0], [1.0e6, 1.0e3, 0.0]], 50.0, 1.0);
  let mvp = ortho_mvp(half, [0.0, 0.0]);
  let params = ortho_pc(mvp, half, 512, 1.0, 2).params;
  let pos = |mvp: &[f32; 16], c: u32| child_position(&render, 1, c, mvp, params, [0.0; 4]);
  // k does not enter a dot's position: any prefix is the same dots, evenly spread
  let mut ts: alloc::vec::Vec<f32> = (0..64).map(|c| pos(&mvp, c)[0]).collect();
  ts.sort_by(|a, b| a.total_cmp(b));
  let gaps: alloc::vec::Vec<f32> = ts.windows(2).map(|w| w[1] - w[0]).collect();
  let (gmin, gmax) = gaps.iter().fold((f32::MAX, 0.0f32), |(a, b), &g| (a.min(g), b.max(g)));
  assert!(gmax < 3.0 * gmin, "64-dot prefix gaps {gmin}..{gmax} m");
  let c0 = streak_clip(
    &mvp,
    [-1.0e6, 1.0e3, 0.0],
    [1.0e6, 1.0e3, 0.0],
    50.0,
    50.0,
    params,
  )
  .unwrap();
  let mvp_pan = ortho_mvp(half, [20.0, 0.0]);
  let c1 = streak_clip(
    &mvp_pan,
    [-1.0e6, 1.0e3, 0.0],
    [1.0e6, 1.0e3, 0.0],
    50.0,
    50.0,
    params,
  )
  .unwrap();
  assert_eq!(
    (c0.t0, c0.t1),
    (c1.t0, c1.t1),
    "a 20 m pan stays in the grid cell"
  );
  for c in 0..64 {
    assert_eq!(
      pos(&mvp, c),
      pos(&mvp_pan, c),
      "dot {c} moved under a small pan"
    );
  }
  let mvp_far = ortho_mvp(half, [30.0e3, 0.0]);
  assert_ne!(
    pos(&mvp, 5),
    pos(&mvp_far, 5),
    "a pan of a view width re-samples"
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
    let rc0 = Df3::from_f32([d.comet_r_t_hi[0], d.comet_r_t_hi[1], d.comet_r_t_hi[2]])
      .add(&Df3::from_f32([d.comet_r_t_lo[0], d.comet_r_t_lo[1], d.comet_r_t_lo[2]]));
    let vc0 = Df3::from_f32([d.comet_v_dur_hi[0], d.comet_v_dur_hi[1], d.comet_v_dur_hi[2]])
      .add(&Df3::from_f32([d.comet_v_dur_lo[0], d.comet_v_dur_lo[1], d.comet_v_dur_lo[2]]));
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
        let p = [(2.0 * (ix as f64 + 0.5) / n as f64 - 1.0), (2.0 * (iy as f64 + 0.5) / n as f64 - 1.0)];
        if p[0].hypot(p[1]) > 0.6 { line.push_str("    "); continue; }
        line.push_str(&alloc::format!("{:4.1}", dens[k] / mean)); k += 1;
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
    assert_eq!(after.live_count, host.ring.live(), "everything submitted is drawn");
    prev = after;
  }
  // a restored snapshot draws nothing until re-emitted
  host.ring.invalidate_gpu();
  assert!(host.draw_state().is_none());
}

/// Tracers: about one cluster in [`TRACER_EVERY`], chosen by the emission record's child id (the
/// same particles after a seek / in every tier), each drawn as one extra instance at the cluster's
/// exact position, never a tile sample (the white point does not move).
#[test]
fn tracers_are_sparse_stable_real_particles() {
  let n = (0..65_536u32).filter(|&id| is_tracer(id)).count() as f64;
  let rate = n / 65_536.0;
  assert!(
    (rate * TRACER_EVERY as f64 - 1.0).abs() < 0.15,
    "tracer rate {rate}"
  );
  // child indices 0..k stay below the tracer's
  assert!(
    lod_children(1e9, 1.0) <= TRACER_CHILD,
    "real children stay below the tracer index"
  );

  let cfg = test_cfg(1.5e-3, 86400.0);
  let host = run_ticks([1.3 * 86400.0; 8], &orbit_jet, &cfg);
  let render0 = tier_render(&host, &host.draw_state().unwrap().frame);
  let mvp = ortho_mvp(2.0e8, [0.0, 0.0]);
  let pc = ortho_pc(mvp, 2.0e8, 720, 1.0, render0.len() as u32);
  let run = |flags: u32| {
    let mut r = render0.clone();
    let mut tiles = alloc::vec![0u32; DUST_TILE_COUNT as usize];
    let mut list = alloc::vec::Vec::new();
    let out = lod_evaluate(&mut r, &pc, flags, 0.0, &mut tiles, &mut list);
    (r, tiles, list, out)
  };
  let (r0, t0, l0, o0) = run(0);
  let (r1, t1, l1, o1) = run(DUST_VIEW_TRACERS);
  assert_eq!(t0, t1, "tracers are not white-point samples");
  let tracers: alloc::vec::Vec<usize> = l1
    .iter()
    .filter(|&&e| e >> LOD_CLUSTER_BITS == TRACER_CHILD)
    .map(|&e| (e & RENDER_SLOT_MASK) as usize)
    .collect();
  let drawn = (0..r0.len()).filter(|&i| r0[i].age_id_dbeta_flux[3] > 0.0).count();
  assert_eq!(o1.instances, o0.instances + tracers.len() as u32);
  assert_eq!(l1.len() - l0.len(), tracers.len());
  assert!(
    tracers.len() > 10,
    "{} tracers of {drawn} drawn clusters",
    tracers.len()
  );
  for &i in &tracers {
    assert!(is_tracer(child_id(r1[i].age_id_dbeta_flux[2])));
    // same per-child flux as without tracers: the tracer is not a share of the cluster
    assert_eq!(r1[i].age_id_dbeta_flux[3], r0[i].age_id_dbeta_flux[3]);
    let s = child_sample(&r1, i, TRACER_CHILD, &mvp, pc.params, [0.0; 4], 1.0);
    assert!(s.tracer);
    assert_eq!(
      s.pos,
      [r1[i].pos_size[0], r1[i].pos_size[1], r1[i].pos_size[2]]
    );
  }
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
      let f = DustFlowUniform::new(&DustFlowClock::default(), 4.0e8 + u01(pcg(i as u32)) as f64 * 1.0e8);
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

/// Damped view adaptation: the displayed white point moves half of the measured one in log space,
/// so two views whose brightest dust differs 4× show 2× apart (`near.rdc` / `med.rdc`: 1.82× →
/// 1.35×); unmeasured gives the fixed reference.
#[test]
fn view_adaptation_is_damped() {
  assert_eq!(display_white(0.0), 1.0);
  assert_eq!(display_white(f32::NAN), 1.0);
  assert!((display_white(1.0) - 1.0).abs() < 1e-6);
  let (a, b) = (display_white(0.5), display_white(2.0));
  assert!(
    (b / a - 2.0).abs() < 1e-4,
    "4× measured → {}× displayed",
    b / a
  );
  assert!((display_white(1.82 * 1.82) - 1.82).abs() < 1e-3);
}

/// Old dust is spread over its size stratum's β range (`½·Δβ·g·age²`, anti-sunward), far more than
/// its stream spread: the LOD footprint counts it, so the outer tail is a fan instead of isolated
/// single dots (`far_1.rdc`). Without solar gravity the footprint is the stream spread.
#[test]
fn old_dust_footprint_includes_the_beta_spread() {
  let half = 1.0e6f32;
  let mut rc = render_cluster(0, [0.0, 0.0, 0.0], 10.0, 1.0);
  rc.age_id_dbeta_flux[0] = 1.0e6;
  rc.age_id_dbeta_flux[2] = pack_child_id(0.01, 7);
  let render = [rc];
  let pc = ortho_pc(ortho_mvp(half, [0.0, 0.0]), half, 720, 1.0, 1);
  let g = 2.0e-4;
  assert!((dust_extent(&rc, g) / (0.5 * 0.01 * g * 1.0e12) - 1.0).abs() < 1e-2);
  assert_eq!(dust_extent(&rc, 0.0), 10.0);
  let flat = lod_plan(&render, 0, &pc, 0.0).unwrap();
  let wide = lod_plan(&render, 0, &pc, g).unwrap();
  assert_eq!(flat.want, 1.0, "10 m stream spread: one dot");
  assert!(
    wide.want > 1000.0,
    "β spread over the view: {} dots",
    wide.want
  );
  // the drawn radius follows the same footprint (dust.vert mirror)
  let s = child_sample(&render, 0, 0, &pc.mvp, pc.params, [0.0, 0.0, 1.0, g], 0.01);
  assert!(
    s.r_px > DUST_CHILD_PX,
    "fewer dots than the footprint asks: larger splats ({})",
    s.r_px
  );
}
