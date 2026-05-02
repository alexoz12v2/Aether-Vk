use crate::scene::{Component, TransformComponent};
use aethervk_oshal_rlib::math::vector::{Vector, vec3::Vec3f32};
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::Vector3;
use crate::simulation::almanac::AlmanacPackedData;
use crate::types::EngineResult;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlmanacPlanet {
  pub naif_id: i32,
  pub rot_period: f64, // TODO useful?
}

impl Component for AlmanacPlanet {}

impl AlmanacPlanet {
  pub fn new(naif_id: i32, rot_period: f64) -> Self {
    Self {
      naif_id,
      rot_period,
    }
  }

  pub fn step(
    &self,
    transform: &mut TransformComponent,
    epoch: anise::time::Epoch,
    step_days: f64,
    almanac: &AlmanacPackedData,
  ) -> EngineResult<()> {
    // TODO switch to SUN_ECLIPJ2000
    let kinematic_state = almanac.get_ephem_full(
      self.naif_id,
      anise::constants::frames::SSB_J2000,
      epoch,
      true,
      false,
    )?;
    transform.position = kinematic_state.position;
    transform.scale = Vec3f32::splat(1.0);
    if let Some(rot) = kinematic_state.rotation {
      transform.rotation = rot.normalize();
    }

    Ok(())
  }
}
