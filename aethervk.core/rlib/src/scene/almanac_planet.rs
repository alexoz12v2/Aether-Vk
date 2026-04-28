use crate::scene::{Component, TransformComponent};
use aethervk_oshal_rlib::math::vector::{Vector, vec3::Vec3f32};
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::Vector3;
use anise::prelude::Epoch;
use anise::almanac::Almanac;
use crate::simulation::almanac::get_almanac_pos;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlmanacPlanet {
  pub naif_id: i32,
  pub rot_period: f64,
}

impl Component for AlmanacPlanet {}

impl AlmanacPlanet {
  pub fn new(naif_id: i32, rot_period: f64) -> Self {
    Self { naif_id, rot_period }
  }

  pub fn step(
    &self,
    transform: &mut TransformComponent,
    epoch: Epoch,
    step_days: f64,
    almanac: &Almanac,
  ) {
    let pos = get_almanac_pos(self.naif_id, epoch, almanac);
    transform.position = pos;
    transform.scale = Vec3f32::splat(1.0);
    let rotations = if self.rot_period != 0.0 {
      step_days * 24.0 / self.rot_period
    } else {
      0.0
    };
    let radians = (rotations * core::f64::consts::TAU) as f32;
    let rot_delta = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), radians);
    transform.rotation = (transform.rotation * rot_delta).normalize();
  }
}
