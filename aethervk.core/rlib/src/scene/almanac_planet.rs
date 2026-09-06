//! almanac_planet module.

use crate::{
  scene::Component,
  simulation::almanac::{AlmanacPackedData, VecTypeConversion},
  types::EngineResult,
};
use aethervk_oshal_rlib::math::{
  matrix::mat3::Mat3f32,
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec3f64::DVec3, vec4::Quat},
};

/// Drives a kinematic body's position (and optionally rotation) from almanac data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlmanacPlanet {
  pub naif_id: i32,
}

impl Component for AlmanacPlanet {}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct AlmanacPlanetDTO {
  pub naif_id: i32,
}

impl crate::scene::ForeignSerializable for AlmanacPlanet {
  type ForeignData = AlmanacPlanetDTO;
  const COMPONENT_ID: u64 = crate::scene::ComponentTypeId::AlmanacPlanet as u64;

  fn to_foreign(&self) -> Self::ForeignData {
    AlmanacPlanetDTO {
      naif_id: self.naif_id,
    }
  }

  fn apply_foreign(&mut self, data: &Self::ForeignData) {
    self.naif_id = data.naif_id;
  }
}

impl AlmanacPlanet {
  /// Creates an AlmanacPlanet. Rotation can be driven by almanac if bpc file loaded, or driven by
  /// an associated RotationBodyModelComponent
  pub fn new(naif_id: i32) -> Self {
    Self { naif_id }
  }

  /// Steps a kinematic body driven by SPK ephemeris.
  pub fn step(
    &self,
    epoch: anise::time::Epoch,
    almanac: &AlmanacPackedData,
    rotational_model: Option<&crate::scene::BodyRotationalModel>,
  ) -> EngineResult<(DVec3, Quat)> {
    let target_frame = crate::simulation::almanac::SUN_ECLIPJ2000;

    // fetch state. if rotational model is missing, then we demand it from IAU rotational model
    let state = almanac.get_cartesian_state(
      self.naif_id,
      target_frame.orientation_id,
      target_frame.ephemeris_id,
      epoch,
      true, // allow_barycentre_fallback
    )?;

    // - Resolve *Active* Rotation: Body-Fixed (BF) -> World (target frame)
    let q_world_from_bf = if let Some(model) = rotational_model {
      // calculate elapsed continuous TDB days since J2000 epoch
      let j2000_epoch = anise::time::J2000_REF_EPOCH;
      let d_j2000 = (epoch - j2000_epoch).to_seconds() / 86400.0;
      let t_centuries = d_j2000 / 36525.0;

      let ra_deg = model.pole_ra + model.pole_ra_rate * t_centuries;
      let dec_deg = model.pole_dec + model.pole_dec_rate * t_centuries;
      let w_deg = model.prime_meridian + model.rotation_rate * d_j2000;

      // Safely fold angles into [0,360) using a pure core implementation to avoid `libm::fmod`
      // (`rem_euclid`) which can panic no_std linkers
      let wrap_angle = |mut angle: f64| -> f32 {
        let cycles = (angle / 360.0) as i64 as f64;
        angle -= cycles * 360.0;
        if angle < 0.0 {
          angle += 360.0;
        }
        angle as f32
      };

      let deg_to_rad = core::f32::consts::PI / 180.0;
      let ra_rad = wrap_angle(ra_deg) * deg_to_rad;
      let dec_rad = wrap_angle(dec_deg) * deg_to_rad;
      let w_rad = wrap_angle(w_deg) * deg_to_rad;

      let z_axis = Vec3f32::from_components(0.0, 0.0, 1.0);
      let x_axis = Vec3f32::from_components(1.0, 0.0, 0.0);

      // Construct active rotation mapping applied right-to-left: Z(W) -> X(90 - dec) -> Z(RA + 90)
      let q_w = Quat::from_axis_angle(z_axis, w_rad);
      let q_dec = Quat::from_axis_angle(x_axis, core::f32::consts::FRAC_PI_2 - dec_rad);
      let q_ra = Quat::from_axis_angle(z_axis, ra_rad + core::f32::consts::FRAC_PI_2);

      let q_j2000_from_bf = q_ra * q_dec * q_w;

      // Evaluate Orientation from standardized J2000 equator to the specified simulation target
      // plane (which is SUN_ECLIPJ2000)
      let j2000_frame = anise::frames::Frame::new(
        target_frame.ephemeris_id, // origin matches target observer (SUN)
        anise::constants::orientations::J2000, // earth mean equator at J2000
      );

      let q_world_from_j2000 = if target_frame.orientation_id == j2000_frame.orientation_id {
        Quat::identity()
      } else if let Ok(dcm) = almanac.almanac.rotate(j2000_frame, target_frame, epoch) {
        let r_mat = Mat3f32::from_nalgebra(dcm.rot_mat);
        Quat::from_rotation_matrix(&r_mat)
      } else {
        Quat::identity()
      };

      // Cascade the orientation to offsets completely
      q_world_from_j2000 * q_j2000_from_bf
    } else if self.naif_id == anise::constants::celestial_objects::EARTH {
      // We loaded `earth_latest_high_prec.bpc` and `pck00011.tpc` (not checking the latter)
      assert!(
        almanac
          .almanac
          .bpc_data
          .keys()
          .any(|k| k.ends_with("earth_latest_high_prec.bpc")),
        "Earth rotation requires `earth_latest_high_prec.bpc` to be loaded. Instead we have [ {:?} ]",
        almanac.almanac.bpc_data.keys()
      );

      // Fallbacks for Earth: high precision Binary PCKs (BPC files like earth_latest_high_prec.bpc)
      // do not store orientation data under the ephemeris ID 399. Instead, they store Earth's
      // orientation under the International Terrestrial Reference Frame ITRF93 (ID: 13000),
      // or the standard IAU_EARTH frame (ID: 10013). Because of this quirk in NAIF convention,
      // we must explicitly try these standard Earth frames when retrieving its rotation matrix.
      use anise::constants::frames::{EARTH_ITRF93, IAU_EARTH_FRAME};
      let earth_rotation_dcm = almanac
        .almanac
        .rotate(EARTH_ITRF93, target_frame, epoch)
        .or_else(|_| almanac.almanac.rotate(IAU_EARTH_FRAME, target_frame, epoch))
        .unwrap(); // can't fail if we have BPC.

      let r_mat = Mat3f32::from_nalgebra(earth_rotation_dcm.rot_mat);
      Quat::from_rotation_matrix(&r_mat)
    } else {
      Quat::identity()
    };

    Ok((
      DVec3::from_components(state.radius_km[0], state.radius_km[1], state.radius_km[2]),
      q_world_from_bf,
    ))
  }

  // TODO: marked for deletion cause if we want the camera to follow the comet or the earth, do a
  // reparenting operation instead!
  //
  // Steps a high-res transform component (e.g. Camera).
  // pub fn step_high_res(
  //   &self,
  //   transform: &mut crate::scene::HighResTransformComponent,
  //   epoch: anise::time::Epoch,
  //   almanac: &crate::simulation::almanac::AlmanacPackedData,
  // ) -> EngineResult<()> {
  //   let kinematic_state = almanac.get_ephem_full(
  //     self.naif_id,
  //     crate::simulation::almanac::SUN_ECLIPJ2000,
  //     epoch,
  //     true,
  //     false,
  //   )?;
  //   transform.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
  //     kinematic_state.position.x() as f64,
  //     kinematic_state.position.y() as f64,
  //     kinematic_state.position.z() as f64,
  //   );
  //   if let Some(rot_bf_world) = kinematic_state.rotation {
  //     if self.surface_offset_bf != Vec3f32::zero() {
  //       let offset_world = rot_bf_world.rotate_vector(self.surface_offset_bf);
  //       transform.position += aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
  //         offset_world.x() as f64,
  //         offset_world.y() as f64,
  //         offset_world.z() as f64,
  //       );
  //     } else {
  //       transform.rotation = (rot_bf_world * self.bf_to_pa.inverse()).normalize();
  //     }
  //   }

  //   if let Some(k) = kinematic {
  //     k.velocity = kinematic_state.velocity;
  //     if let Some(ang_vel_bf) = kinematic_state.angular_velocity {
  //       k.angular_velocity = self.bf_to_pa.rotate_vector(ang_vel_bf);
  //     } else {
  //       k.angular_velocity = Vec3f32::zero();
  //     }
  //   }

  //   Ok(())
  // }
}

#[cfg(test)]
mod test_almanac_planet;
