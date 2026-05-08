use crate::scene::{Component, TransformComponent};
use crate::simulation::almanac::AlmanacPackedData;
use crate::types::EngineResult;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector, vec3::Vec3f32};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlmanacPlanet {
  pub naif_id: i32,
  pub rot_period: f64, // TODO useful?
  pub mu: f32,
}

impl Component for AlmanacPlanet {}

impl AlmanacPlanet {
  pub fn new(naif_id: i32, rot_period: f64, mu: f32) -> Self {
    Self {
      naif_id,
      rot_period,
      mu,
    }
  }

  pub fn step(
    &self,
    transform: &mut TransformComponent,
    epoch: anise::time::Epoch,
    step_days: f64,
    almanac: &AlmanacPackedData,
  ) -> EngineResult<()> {
    let kinematic_state = almanac.get_ephem_full(
      self.naif_id,
      anise::constants::frames::SUN_J2000, // crate::simulation::almanac::SUN_ECLIPJ2000, // TODO test if absent
      epoch,
      true,
      false,
    )?;
    transform.position = kinematic_state.position;
    if let Some(rot) = kinematic_state.rotation {
      transform.rotation = rot.normalize();
    }

    Ok(())
  }
}
