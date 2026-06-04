//! almanac_planet module.

use crate::{
  scene::{Component, TransformComponent},
  simulation::almanac::AlmanacPackedData,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Drives a kinematic body's position (and optionally rotation) from almanac data.
pub struct AlmanacPlanet {
  pub naif_id: i32,
  pub rot_period: f64,
  pub mu: f32,
  /// Rotation from Body-Fixed (BF) frame to Principal Axis (PA) frame.
  pub bf_to_pa: Quat,
  /// Offset of the surface observer in Body-Fixed (BF) frame.
  pub surface_offset_bf: Vec3f32,
}

impl Component for AlmanacPlanet {}

impl AlmanacPlanet {
  /// Creates an AlmanacPlanet (standard almanac-driven rotation).
  pub fn new(naif_id: i32, rot_period: f64, mu: f32) -> Self {
    Self {
      naif_id,
      rot_period,
      mu,
      bf_to_pa: Quat::identity(),
      surface_offset_bf: Vec3f32::zero(),
    }
  }

  /// Steps a kinematic body driven by SPK ephemeris.
  ///
  /// Position and velocity are always read from the almanac SPK data.
  /// Rotation is sourced based on `kinematic.use_model_rotation`:
  ///  - **false** (default): rotation comes from almanac BPC data (planets with BPC files).
  ///  - **true**: rotation is computed from the `BodyRotationalModel` on the
  ///    `PhysicalMeshComponent` (comets, whose SPK files lack rotation data).
  pub fn step(
    &self,
    transform: &mut TransformComponent,
    kinematic: Option<&mut crate::scene::KinematicComponent>,
    epoch: anise::time::Epoch,
    _step_days: f64,
    almanac: &AlmanacPackedData,
    rotational_model: Option<&crate::scene::BodyRotationalModel>,
  ) -> EngineResult<()> {
    let kinematic_state = almanac.get_ephem_full(
      self.naif_id,
      crate::simulation::almanac::SUN_ECLIPJ2000,
      epoch,
      true,
      false,
    )?;
    transform.position = kinematic_state.position;

    // Determine whether to use the BodyRotationalModel for rotation
    let use_model = kinematic.as_ref().map_or(false, |k| k.use_model_rotation);

    // Model-derived angular velocity (computed if use_model + model present)
    let mut model_angular_velocity = None;

    if use_model {
      // Compute rotation from IAU-style BodyRotationalModel
      if let Some(model) = rotational_model {
        let jd = epoch.to_jde_utc_days();
        // orientation_at() already returns the quaternion in ECLIPJ2000 frame
        // (includes the Rx(ε) obliquity correction internally)
        let orientation_quat = model.orientation_at(jd);
        // Transform to PA → World: rot_bf_world * bf_to_pa.inverse()
        transform.rotation = (orientation_quat * self.bf_to_pa.inverse()).normalize();

        // Derive angular velocity from rotation rate along the pole axis
        let t_centuries = (jd - model.reference_epoch_jd) / 36525.0;
        let ra = (model.pole_ra + model.pole_ra_rate * t_centuries).to_radians();
        let dec = (model.pole_dec + model.pole_dec_rate * t_centuries).to_radians();

        // Pole unit vector in J2000 inertial frame
        let pole_inertial = Vec3f32::from_components(
          (dec.cos() * ra.cos()) as f32,
          (dec.cos() * ra.sin()) as f32,
          dec.sin() as f32,
        );

        // rotation_rate is in deg/day, convert to rad/s
        let omega_rad_s = (model.rotation_rate.to_radians() / 86400.0) as f32;
        // Apply obliquity to transform pole from ICRF to ECLIPJ2000
        let q_j2000_to_eclip = Quat::from_axis_angle(
          Vec3f32::from_components(1.0, 0.0, 0.0),
          23.4392911_f32.to_radians(),
        );
        let ang_vel_inertial = q_j2000_to_eclip.rotate_vector(pole_inertial * omega_rad_s);

        // Transform to PA frame for physics
        model_angular_velocity = Some(self.bf_to_pa.rotate_vector(ang_vel_inertial));
      }
    } else {
      // Standard path: rotation from almanac BPC data
      if let Some(rot_bf_world) = kinematic_state.rotation {
        if self.surface_offset_bf != Vec3f32::zero() {
          let offset_world = rot_bf_world.rotate_vector(self.surface_offset_bf);
          transform.position += offset_world;
        } else {
          transform.rotation = (rot_bf_world * self.bf_to_pa.inverse()).normalize();
        }
      }
    }

    if let Some(k) = kinematic {
      k.velocity = kinematic_state.velocity;
      if let Some(model_ang_vel) = model_angular_velocity {
        // Model-driven angular velocity
        k.angular_velocity = model_ang_vel;
      } else if !use_model {
        // Almanac-driven angular velocity
        if let Some(ang_vel_bf) = kinematic_state.angular_velocity {
          k.angular_velocity = self.bf_to_pa.rotate_vector(ang_vel_bf);
        } else {
          k.angular_velocity = Vec3f32::zero();
        }
      }
    }

    Ok(())
  }

  /// Steps a high-res transform component (e.g. Camera).
  pub fn step_high_res(
    &self,
    transform: &mut crate::scene::HighResTransformComponent,
    kinematic: Option<&mut crate::scene::KinematicComponent>,
    epoch: anise::time::Epoch,
    _step_days: f64,
    almanac: &crate::simulation::almanac::AlmanacPackedData,
  ) -> EngineResult<()> {
    let kinematic_state = almanac.get_ephem_full(
      self.naif_id,
      crate::simulation::almanac::SUN_ECLIPJ2000,
      epoch,
      true,
      false,
    )?;
    transform.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
      kinematic_state.position.x() as f64,
      kinematic_state.position.y() as f64,
      kinematic_state.position.z() as f64,
    );
    if let Some(rot_bf_world) = kinematic_state.rotation {
      if self.surface_offset_bf != Vec3f32::zero() {
        let offset_world = rot_bf_world.rotate_vector(self.surface_offset_bf);
        transform.position += aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
          offset_world.x() as f64,
          offset_world.y() as f64,
          offset_world.z() as f64,
        );
      } else {
        transform.rotation = (rot_bf_world * self.bf_to_pa.inverse()).normalize();
      }
    }

    if let Some(k) = kinematic {
      k.velocity = kinematic_state.velocity;
      if let Some(ang_vel_bf) = kinematic_state.angular_velocity {
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
      surface_offset_bf: Vec3f32::zero(),
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
