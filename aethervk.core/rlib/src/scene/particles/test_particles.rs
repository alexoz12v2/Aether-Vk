use bytemuck::Zeroable;

use super::*;

#[test]
fn beta_computation() {
  use super::v2::*;
  let psep = {
    let mut x = ParticleSystemEmitParams::zeroed();
    x.diametre_um = 2_f32;
    x.density_gcm3 = 0.5_f32;
    x.scattering_efficiency = 1.03_f32;
    x.geometric_albedo = 0.04;
    x
  };
  let tol = 1e-5_f32;
  let expected_beta = 1.1829265_f32;
  let actual_beta = psep.beta();

  approx::assert_abs_diff_eq!(actual_beta, expected_beta, epsilon = tol);
}

/// Every grain and activity parameter of the FFI record reaches the jet's emission parameters
/// unchanged (no hidden unit conversion): they are the jet's own, not model-wide values.
#[test]
fn emit_params_from_dto_copies_every_grain_and_activity_field() {
  use super::v2::*;
  let mut dto = crate::scene::ParticleSystemDTO::zeroed();
  dto.diametre_um = 20.0;
  dto.density_gcm3 = 0.8;
  dto.scattering_efficiency = 1.1;
  dto.geometric_albedo = 0.05;
  dto.size_q = 3.25;
  dto.size_range = 30.0;
  dto.dust_rate_1au_kgs = 800.0;
  dto.dust_rate_power = 2.2;
  dto.dust_rate_cutoff_au = 5.5;
  dto.dust_rate_max_kgs = 150_000.0;
  dto.latitude_rad = -1.08;
  dto.longitude_rad = 0.25;
  dto.aperture_rad = 0.17;
  dto.start_velocity_mean = 60.0;
  dto.start_velocity_std = 5.0;
  dto.seed = 7;

  let ep = ParticleSystemEmitParams::from_dto(&dto);

  assert_eq!(ep.diametre_um, 20.0);
  assert_eq!(ep.density_gcm3, 0.8);
  assert_eq!(ep.scattering_efficiency, 1.1);
  assert_eq!(ep.geometric_albedo, 0.05);
  assert_eq!(ep.size_q, 3.25);
  assert_eq!(ep.size_range, 30.0);
  assert_eq!(ep.dust_rate_1au_kgs, 800.0);
  assert_eq!(ep.dust_rate_power, 2.2);
  assert_eq!(ep.dust_rate_cutoff_au, 5.5);
  assert_eq!(ep.dust_rate_max_kgs, 150_000.0);
  assert_eq!(ep.latitude_rad, -1.08);
  assert_eq!(ep.longitude_rad, 0.25);
  assert_eq!(ep.aperture_rad, 0.17);
  assert_eq!(ep.start_velocity_mean, 60.0);
  assert_eq!(ep.start_velocity_std, 5.0);
  assert_eq!(ep.seed, 7);
}

/// `EMISSION_START_NOW` resolves to the scene clock's current epoch and is refused without a
/// clock; explicit values and the pre-existing sentinel pass through untouched.
#[test]
fn emission_start_now_resolves_to_the_clock_or_refuses() {
  use anise::time::Epoch;
  let now = Epoch::from_gregorian_utc(2025, 10, 17, 12, 0, 0, 0);
  assert_eq!(
    resolve_emission_start_us(EMISSION_START_NOW, Some(now)),
    Some(emission_start_us_from_epoch(now))
  );
  assert_eq!(resolve_emission_start_us(EMISSION_START_NOW, None), None);
  let explicit = emission_start_us_from_epoch(now) - 86_400_000_000;
  assert_eq!(
    resolve_emission_start_us(explicit, Some(now)),
    Some(explicit)
  );
  assert_eq!(resolve_emission_start_us(explicit, None), Some(explicit));
  assert_eq!(
    resolve_emission_start_us(EMISSION_START_PREEXISTING, None),
    Some(EMISSION_START_PREEXISTING)
  );
  // the resolved value round-trips through the epoch conversion
  let us = resolve_emission_start_us(EMISSION_START_NOW, Some(now)).unwrap();
  assert!((epoch_from_emission_start_us(us) - now).abs().to_seconds() < 1e-5);
}

/// The size law parameters read through their validating accessors: a zeroed (or pre-v7)
/// record gets the defaults 3.5 / 10, out-of-range values too; the range reaches the size
/// distribution (radii `r/F .. r·F`), the exponent does not (the continuum keeps 3.5).
#[test]
fn size_law_params_default_and_validate() {
  use super::v2::*;
  let mut ep = ParticleSystemEmitParams::zeroed();
  ep.diametre_um = 100.0;
  assert_eq!(ep.size_q(), 3.5);
  assert_eq!(ep.size_range(), 10.0);
  ep.size_q = 3.0;
  ep.size_range = 20.0;
  assert_eq!(ep.size_q(), 3.0);
  assert_eq!(ep.size_range(), 20.0);
  let d = ep.size_distribution();
  assert!((d.s_min_um - 2.5).abs() < 1e-9 && (d.s_max_um - 1000.0).abs() < 1e-9);
  assert_eq!(d.q, 3.5, "the continuum keeps q = 3.5");
  for (q, f) in [(1.5f32, 0.5f32), (6.0, 2000.0), (f32::NAN, f32::INFINITY)] {
    ep.size_q = q;
    ep.size_range = f;
    assert_eq!(ep.size_q(), 3.5, "q {q}");
    assert_eq!(ep.size_range(), 10.0, "F {f}");
  }
}

/// The grain / speed parameters of a jet whose activity was an Afρ of `afrho_cm`, as the
/// validation conversion and the legacy law need them (seed, site and colour do not enter).
fn legacy_afrho_params(
  afrho_cm: f32,
  diametre_um: f32,
  density_gcm3: f32,
  v_mean: f32,
  v_std: f32,
  albedo: f32,
  size_range: f32,
) -> (f32, v2::ParticleSystemEmitParams) {
  use super::v2::*;
  let mut p = ParticleSystemEmitParams::zeroed();
  p.diametre_um = diametre_um;
  p.density_gcm3 = density_gcm3;
  p.scattering_efficiency = 1.0;
  p.geometric_albedo = albedo;
  p.size_q = 3.5;
  p.size_range = size_range;
  p.start_velocity_mean = v_mean;
  p.start_velocity_std = v_std;
  p.aperture_rad = 0.5;
  (afrho_cm, p)
}

/// `dust_rate_from_afrho_kgs` reproduces the former Afρ → Q_d conversion (the thesis / Agarwal
/// form with the engine's `f_v` and speed-spread factors) on an f64 reference, so a pre-v9 save's
/// activity and the closure test's input keep their value.
#[test]
fn legacy_afrho_conversion_matches_goldens() {
  // (afrho_cm, diametre_um, density_gcm3, v_mean, v_std, albedo, size_range) -> kg/s
  let goldens: [((f32, f32, f32, f32, f32, f32, f32), f64); 5] = [
    ((100.0, 100.0, 0.8, 2.0, 0.5, 0.04, 10.0), 1.5510886453350254),
    ((100.0, 100.0, 0.533, 2.0, 0.5, 0.04, 10.0), 1.0334128099544606),
    ((537.0, 250.0, 1.0, 3.1, 0.6, 0.05, 10.0), 33.05519924265463),
    ((8.128, 20.0, 0.5, 1.5, 0.3, 0.03, 3.0), 0.013690648521032838),
    ((100000.0, 100.0, 0.8, 2.5, 0.625, 0.04, 100.0), 3372.168918307603),
  ];
  for ((a, d, rho, vm, vs, p_alb, f), expected) in goldens {
    let (afrho_cm, p) = legacy_afrho_params(a, d, rho, vm, vs, p_alb, f);
    let got = p.dust_rate_from_afrho_kgs(afrho_cm) as f64;
    let rel = (got / expected - 1.0).abs();
    assert!(rel < 2e-6, "Afρ {a} cm, grain {d} µm / {rho} g/cm³, v {vm} ± {vs}, p {p_alb}, F {f}: {got} vs {expected} kg/s (rel {rel:.2e})");
    // the inverse closes
    let back = p.afrho_cm_from_dust_rate(got as f32);
    assert!((back / afrho_cm - 1.0).abs() < 2e-6, "inverse: {back} vs {afrho_cm} cm");
  }
  // undefined without an albedo
  let (_, mut p) = legacy_afrho_params(100.0, 100.0, 0.8, 2.0, 0.5, 0.0, 10.0);
  assert_eq!(p.dust_rate_from_afrho_kgs(100.0), 0.0);
  assert_eq!(p.afrho_cm_from_dust_rate(1.0), 0.0);
  p.geometric_albedo = -0.1;
  assert_eq!(p.dust_rate_from_afrho_kgs(100.0), 0.0);
}

/// The dust rate law `min(Q_d(1 AU) r^-k, Q_d max)` with the converted inputs is the removed
/// Afρ law `Q_d(min(Afρ₀ r^-k, Afρ max))` at every heliocentric distance: below the clamp kink,
/// across it and at / after the cutoff (both 0 there).
#[test]
fn dust_rate_law_matches_the_legacy_afrho_law() {
  use super::v2::*;
  use aethervk_oshal_rlib::math::FloatLike;
  /// the removed `dust_production_rate_kgs` body, verbatim, on the legacy Afρ inputs
  fn legacy_rate_kgs(
    p: &ParticleSystemEmitParams,
    afrho_0_cm: f32,
    afrho_power: f32,
    afrho_cutoff_au: f32,
    afrho_max_value_cm: f32,
    r_helio_au: f32,
  ) -> f32 {
    if r_helio_au >= afrho_cutoff_au || !(p.geometric_albedo > 0.0) {
      return 0.0_f32;
    }
    let current_afrho_cm = afrho_0_cm * <f32 as FloatLike>::pow(r_helio_au, -afrho_power);
    let clamped_afrho_cm = <f32 as FloatLike>::min(current_afrho_cm, afrho_max_value_cm);
    let afrho_m = clamped_afrho_cm / 100_f32;
    let radius_m = p.diametre_um * 0.5e-6_f32;
    let density_kgm3 = p.density_gcm3 * 1e3_f32;
    let dist = p.size_distribution();
    let f_v = dist.mean_speed_factor() as f32;
    let v_ref = p.start_velocity_mean.max(0.0);
    let v_std_rel = if v_ref > 0.0 {
      (p.start_velocity_std / v_ref).max(0.0)
    } else {
      0.0
    };
    (2.0 * density_kgm3 * radius_m * v_ref * afrho_m)
      / (3.0 * p.geometric_albedo * f_v * (1.0 + v_std_rel * v_std_rel))
  }

  // (Afρ₀ cm, power, cutoff AU, Afρ max cm) laws on two grains; the max keeps the kink inside
  // the sampled range (Afρ₀ r^-k = max at r = (Afρ₀ / max)^(1/k))
  let laws = [(100.0f32, 2.0f32, 15.0f32, 100_000.0f32), (537.0, 2.5, 5.0, 2_000.0), (8.128, 1.0, 3.0, 50.0)];
  let grains = [(100.0f32, 0.533f32, 2.0f32, 0.5f32, 0.04f32, 10.0f32), (250.0, 1.0, 3.1, 0.6, 0.05, 30.0)];
  let mut samples = 0usize;
  for (a0, k, cutoff, a_max) in laws {
    for (d, rho, vm, vs, alb, f) in grains {
      let (_, mut p) = legacy_afrho_params(a0, d, rho, vm, vs, alb, f);
      p.dust_rate_power = k;
      p.dust_rate_cutoff_au = cutoff;
      p.dust_rate_1au_kgs = p.dust_rate_from_afrho_kgs(a0);
      p.dust_rate_max_kgs = p.dust_rate_from_afrho_kgs(a_max);
      // 0.05 .. 14.9 AU in 0.05 AU steps, plus the cutoff itself and beyond it
      let mut r = 0.05f32;
      let mut rs = alloc::vec::Vec::new();
      while r < 14.95 {
        rs.push(r);
        r += 0.05;
      }
      rs.extend([cutoff, cutoff + 1e-3, cutoff * 2.0, 40.0]);
      for r in rs {
        let old = legacy_rate_kgs(&p, a0, k, cutoff, a_max, r);
        let new = p.dust_production_rate_kgs(r);
        samples += 1;
        if r >= cutoff {
          assert_eq!(old, 0.0, "legacy at {r} AU");
          assert_eq!(new, 0.0, "new at {r} AU");
          continue;
        }
        assert!(old > 0.0 && new > 0.0, "{r} AU: old {old} new {new}");
        let rel = ((new as f64) / (old as f64) - 1.0).abs();
        assert!(rel < 4e-6, "Afρ₀ {a0} cm k {k} grain {d} µm at {r} AU: new {new} vs legacy {old} kg/s (rel {rel:.2e})");
      }
    }
  }
  assert!(samples >= 200, "{samples} samples");
}

/// The activity is a mass rate: the albedo (a photometric parameter) no longer changes it.
#[test]
fn dust_rate_ignores_albedo() {
  let (_, mut p) = legacy_afrho_params(100.0, 100.0, 0.533, 2.0, 0.5, 0.04, 10.0);
  p.dust_rate_1au_kgs = 2.0;
  p.dust_rate_power = 2.0;
  p.dust_rate_cutoff_au = 15.0;
  p.dust_rate_max_kgs = 2000.0;
  let before: alloc::vec::Vec<f32> = [0.02f32, 0.5, 1.0, 1.24, 3.0, 14.9].iter().map(|&r| p.dust_production_rate_kgs(r)).collect();
  p.geometric_albedo = 0.4;
  let after: alloc::vec::Vec<f32> = [0.02f32, 0.5, 1.0, 1.24, 3.0, 14.9].iter().map(|&r| p.dust_production_rate_kgs(r)).collect();
  assert_eq!(before, after);
  assert_eq!(before[2], 2.0, "Q_d(1 AU) is the input");
  assert_eq!(before[0], 2000.0, "clamped at the max");
  assert_eq!(p.dust_production_rate_kgs(15.0), 0.0);
  // never negative
  p.dust_rate_1au_kgs = -1.0;
  assert_eq!(p.dust_production_rate_kgs(1.0), 0.0);
}
