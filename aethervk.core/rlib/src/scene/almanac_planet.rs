//! almanac_planet module.

use crate::{
  scene::{Component, TransformComponent},
  simulation::almanac::AlmanacPackedData,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::vec3::Vec3f32,
  vector::Vector,
  vector::Vector3,
  vector::vec4::Quat,
};

#[derive(Debug, Clone, Copy, PartialEq)]
/// TODO: Document this item
pub struct AlmanacPlanet {
  pub naif_id: i32,
  pub rot_period: f64, // TODO useful?
  pub mu: f32,
  /// Rotation from Body-Fixed (BF) frame to Principal Axis (PA) frame.
  pub bf_to_pa: Quat,
}

impl Component for AlmanacPlanet {}

impl AlmanacPlanet {
  /// TODO: Document this item
  pub fn new(naif_id: i32, rot_period: f64, mu: f32) -> Self {
    Self {
      naif_id,
      rot_period,
      mu,
      bf_to_pa: Quat::identity(),
    }
  }

  /// TODO: Document this item
  pub fn step(
    &self,
    transform: &mut TransformComponent,
    kinematic: Option<&mut crate::scene::KinematicComponent>,
    epoch: anise::time::Epoch,
    _step_days: f64,
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
    if let Some(rot_bf_world) = kinematic_state.rotation {
      // transform.rotation is PA -> World
      // PA -> BF -> World
      // bf_to_pa is BF -> PA, so PA -> BF is bf_to_pa.inverse()
      transform.rotation = (rot_bf_world * self.bf_to_pa.inverse()).normalize();
    }

    if let Some(k) = kinematic {
      k.velocity = kinematic_state.velocity;
      if let Some(ang_vel_bf) = kinematic_state.angular_velocity {
        // transform angular velocity from BF to PA (simulation space)
        k.angular_velocity = self.bf_to_pa.rotate_vector(ang_vel_bf);
      } else {
        k.angular_velocity = Vec3f32::zero();
      }
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::scene::TransformComponent;
  use aethervk_oshal_rlib::math::quaternion::Quaternion;

  #[test]
  fn test_almanac_planet_with_offset() {
    let mut transform = TransformComponent::default();
    let bf_to_pa = Quat::from_axis_angle(
      Vec3f32::from_components(0.0, 0.0, 1.0),
      45.0_f32.to_radians(),
    );
    let planet = AlmanacPlanet {
      naif_id: 399,
      rot_period: 0.0,
      mu: 0.0,
      bf_to_pa,
    };

    // Mock an Almanac rotation: identity (BF is aligned with World)
    let rot_bf_world = Quat::identity();

    // PA -> BF -> World
    // BF -> World is identity.
    // PA -> BF is bf_to_pa.inverse()
    // Result should be bf_to_pa.inverse()
    let expected = bf_to_pa.inverse();

    // We can't easily call `step` without a real Almanac and Epoch.
    // But we can test the logic directly if we extract it or just test the formula.
    let result_rot = (rot_bf_world * planet.bf_to_pa.inverse()).normalize();

    assert!((result_rot.vector_part().x() - expected.vector_part().x()).abs() < 1e-6);
    assert!((result_rot.vector_part().y() - expected.vector_part().y()).abs() < 1e-6);
    assert!((result_rot.vector_part().z() - expected.vector_part().z()).abs() < 1e-6);
    assert!((result_rot.scalar_part() - expected.scalar_part()).abs() < 1e-6);
  }
}
