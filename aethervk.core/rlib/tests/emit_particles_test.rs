//! Dust v3 emission through the public API: `ParticleSystemEmitParams::dust_emit_config` +
//! `DustHostState::tick` (the exact path the logic thread runs every tick).
use aethervk_core_rlib::scene::{
  dust::{
    AU_M, DustHostState, JetState, RING_CAPACITY_HIGH, RING_CAPACITY_LOW, RING_GUARD_DIVISOR,
    SUN_MU_M3_S2, kepler,
  },
  particles::v2::ParticleSystemEmitParams,
};
use bytemuck::Zeroable;

fn default_params() -> ParticleSystemEmitParams {
  let mut params = ParticleSystemEmitParams::zeroed();
  params.diametre_um = 100.0; // 67P default
  params.density_gcm3 = 0.533; // 67P default
  params.start_velocity_mean = 2.0;
  params.start_velocity_std = 0.5;
  params.aperture_rad = 0.5;
  params.afrho_0_cm = 100.0;
  params.afrho_power = 2.0;
  params.afrho_cutoff_au = 15.0;
  params.afrho_max_value_cm = 100_000.0;
  params.scattering_efficiency = 1.0;
  params
}

const TTL_US: i64 = 30 * 86_400 * 1_000_000;

/// Runs `ticks` logic ticks of `dt_scaled_s` scaled seconds each (60 Hz real time) on a comet
/// at `r_au`. Returns `(host, emitted_mass_g)`.
fn run(
  params: &ParticleSystemEmitParams,
  capacity: u32,
  r_au: f64,
  dt_scaled_s: f64,
  ticks: u64,
) -> (DustHostState, f64) {
  let r0 = [r_au * AU_M, 0.0, 0.0];
  let v0 = [0.0, (SUN_MU_M3_S2 / (r_au * AU_M)).sqrt(), 0.0];
  let mut host = DustHostState::new(capacity);
  let mut emitted = 0.0;
  let mut now_us = 1_000_000_i64;
  for tick in 0..ticks {
    let t = tick as f64 * dt_scaled_s;
    let (r, v) = kepler::propagate_f64(r0, v0, SUN_MU_M3_S2, t);
    let jet = JetState {
      t_s: t,
      r_m: r,
      v_ms: v,
      rot: [0.0, 0.0, 0.0, 1.0],
    };
    let r_au_now = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt() / AU_M;
    let cfg = params.dust_emit_config(r_au_now as f32, TTL_US);
    for b in host.tick(jet, now_us, &cfg) {
      emitted += b.mass_params[0] as f64;
      assert!(b.count > 0 && b.count <= capacity);
    }
    host.ring.mark_submitted(tick + 1);
    now_us += 16_667;
  }
  (host, emitted)
}

#[test]
fn production_rate_and_config_are_sane() {
  let p = default_params();
  let q = p.dust_production_rate_kgs(5.5);
  assert!(q.is_finite() && q > 0.0, "q {q}");
  let cfg = p.dust_emit_config(5.5, TTL_US);
  assert_eq!(cfg.q_dust_kgs, q as f64);
  assert!((cfg.ttl_s - 30.0 * 86_400.0).abs() < 1e-6);
  assert!(cfg.beta_ref > 0.0 && cfg.xsec_per_g_ref() > 0.0);
  let d = (cfg.jet_dir[0].powi(2) + cfg.jet_dir[1].powi(2) + cfg.jet_dir[2].powi(2)).sqrt();
  assert!((d - 1.0).abs() < 1e-5);
}

#[test]
fn mass_is_independent_of_sim_speed() {
  // the same 12 scaled hours at 1 h/s and at 3 h/s
  let p = default_params();
  let (h1, m1) = run(&p, RING_CAPACITY_LOW, 5.5, 60.0, 721);
  let (h3, m3) = run(&p, RING_CAPACITY_LOW, 5.5, 180.0, 241);
  let total1 = m1 + h1.acc.mass_g;
  let total3 = m3 + h3.acc.mass_g;
  let produced = p.dust_production_rate_kgs(5.5) as f64 * 1e3 * 12.0 * 3600.0;
  assert!(
    ((total1 - produced) / produced).abs() < 1e-3,
    "1 h/s {total1} vs {produced}"
  );
  assert!(
    ((total3 - produced) / produced).abs() < 1e-3,
    "3 h/s {total3} vs {produced}"
  );
}

#[test]
fn ring_budget_holds_at_extreme_speed() {
  // one scaled day per tick: the planner must stay within the guarded ring, never panic
  let p = default_params();
  for capacity in [RING_CAPACITY_LOW, RING_CAPACITY_HIGH] {
    let (host, emitted) = run(&p, capacity, 1.5, 86_400.0, 400);
    assert!(emitted > 0.0);
    assert!(host.ring.live() <= capacity - capacity / RING_GUARD_DIVISOR);
    let ds = host.draw_state().expect("drawable");
    assert!(ds.live_count > 0 && ds.live_count <= host.ring.live());
  }
}

#[test]
fn nothing_beyond_cutoff() {
  let mut p = default_params();
  p.afrho_cutoff_au = 3.0;
  let (host, emitted) = run(&p, RING_CAPACITY_LOW, 5.5, 180.0, 600);
  assert_eq!(emitted, 0.0);
  assert_eq!(host.ring.live(), 0);
  assert!(host.draw_state().is_none());
}
