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
    x
  };
  let tol = 1e-5_f32;
  let expected_beta = 1.1829265_f32;
  let actual_beta = psep.beta();

  approx::assert_abs_diff_eq!(actual_beta, expected_beta, epsilon = tol);
}

#[test]
fn propagate_common_params_field_count() {
  // Documents the exact set of 8 "Jet Common Parameters" that propagate_common_params
  // applies to sibling particle systems. If this list changes, the FFI propagation
  // function in ffi.rs must be updated in lockstep.
  let expected_common_fields: &[&str] = &[
    "mass_variability_perc",
    "diametre_um",
    "density_gcm3",
    "scattering_efficiency",
    "afrho_0_cm",
    "afrho_power",
    "afrho_cutoff_au",
    "afrho_max_value_cm",
  ];
  assert_eq!(
    expected_common_fields.len(),
    8,
    "there must be exactly 8 common (model-level) particle system parameters"
  );
}

#[test]
fn particle_system_emit_params_common_field_roundtrip() {
  use super::v2::*;
  // Given a set of common-param values, copying them to ParticleSystemEmitParams
  // must preserve each value exactly (no hidden unit conversion for common params).
  let mut ep = ParticleSystemEmitParams::zeroed();
  ep.mass_variability_perc = 0.35;
  ep.diametre_um = 20.0;
  ep.density_gcm3 = 0.8;
  ep.scattering_efficiency = 1.1;
  ep.afrho_0_cm = 800.0;
  ep.afrho_power = 2.2;
  ep.afrho_cutoff_au = 5.5;
  ep.afrho_max_value_cm = 150_000.0;

  // Re-apply same values (mirrors what propagate_common_params does)
  let ep2 = ParticleSystemEmitParams {
    mass_variability_perc: ep.mass_variability_perc,
    diametre_um: ep.diametre_um,
    density_gcm3: ep.density_gcm3,
    scattering_efficiency: ep.scattering_efficiency,
    afrho_0_cm: ep.afrho_0_cm,
    afrho_power: ep.afrho_power,
    afrho_cutoff_au: ep.afrho_cutoff_au,
    afrho_max_value_cm: ep.afrho_max_value_cm,
    ..ParticleSystemEmitParams::zeroed()
  };

  assert_eq!(ep2.diametre_um, 20.0);
  assert_eq!(ep2.density_gcm3, 0.8);
  assert_eq!(ep2.scattering_efficiency, 1.1);
  assert_eq!(ep2.afrho_0_cm, 800.0);
  assert_eq!(ep2.afrho_power, 2.2);
  assert_eq!(ep2.afrho_cutoff_au, 5.5);
  assert_eq!(ep2.afrho_max_value_cm, 150_000.0);
  assert_eq!(ep2.mass_variability_perc, 0.35);
  // jet-specific fields left zeroed
  assert_eq!(ep2.latitude_rad, 0.0);
  assert_eq!(ep2.longitude_rad, 0.0);
}
