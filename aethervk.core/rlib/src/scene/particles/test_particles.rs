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
