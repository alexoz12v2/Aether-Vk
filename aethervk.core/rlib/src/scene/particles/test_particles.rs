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
  dto.afrho_0_cm = 800.0;
  dto.afrho_power = 2.2;
  dto.afrho_cutoff_au = 5.5;
  dto.afrho_max_value_cm = 150_000.0;
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
  assert_eq!(ep.afrho_0_cm, 800.0);
  assert_eq!(ep.afrho_power, 2.2);
  assert_eq!(ep.afrho_cutoff_au, 5.5);
  assert_eq!(ep.afrho_max_value_cm, 150_000.0);
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
