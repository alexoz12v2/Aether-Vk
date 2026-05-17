//! almanac module.

use crate::{
  math::vee,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{
  math::{
    matrix::{Matrix, Matrix3, mat3::Mat3f32},
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  },
  os,
  os::{
    files::Mmap,
    fs::{ExtensionToStr, FileSystemObject},
  },
};
use alloc::{string::String, vec::Vec};

/// scale factor between coordinates from SPK ephemeris and simulation space
/// basically 1 km / 1 AU
pub const DISTANCE_SCALE_FACTOR: f64 = 1.0 / 6.6846e-9;

/// Defined a frame whose origin is the sun, and whose orientation (equatorial plane) is the plane
/// which contains the Sun's orbit (rotated ~23 degrees with respect to J2000, which is the plane
/// containing Earth's orbit in year 2000)
pub const SUN_ECLIPJ2000: anise::frames::Frame = anise::frames::Frame::new(
  anise::constants::celestial_objects::SUN,
  anise::constants::orientations::ECLIPJ2000,
);

/// Stores `position` and `velocity`, after being scaled by a factor of [`DISTANCE_SCALE_FACTOR`]
/// to get to *simulation units*
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KinematicState {
  pub position: Vec3f32,
  pub velocity: Vec3f32,
  pub rotation: Option<Quat>,
  pub angular_velocity: Option<Vec3f32>,
}

/// There is no need to track [`bytes::Bytes`] of SPK data cause it's an Arc based type, meaning
/// as long as ['anise::almanac::Almanac'] holds onto it, we are fine
#[derive(Default)]
pub struct AlmanacPackedData {
  pub file_names: Vec<String>,
  /// the almanac itself
  pub almanac: anise::almanac::Almanac,
}

impl AlmanacPackedData {
  // TODO `celestial_name_from_id` to create planet/sun entities
  /// TODO: Document this item
  pub fn load_almanac<P: AsRef<os::fs::Path>>(&mut self, path: P) -> EngineResult<()> {
    if let Ok(entries) = os::fs::read_dir(path.as_ref()) {
      for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().as_deref() == Some("bsp") {
          if let Some(path_cow) = path.to_str_unified() {
            let path_str = path_cow.to_str().unwrap();
            let path_ref: &os::fs::Path = path_str.into();

            // 1. Map the file natively into the Virtual Address Space. 0 RAM Overhead!
            let mmap = Mmap::open(path_ref).map_err(|e| EngineError::from(e))?;

            // 2. Safely wrap into zero-copy `bytes::Bytes` utilizing your Mmap Drop handler constraint.
            // `Bytes::from_owner` natively works since Mmap implements `AsRef<[u8]> + Send + 'static`
            let bytes = bytes::Bytes::from_owner(mmap);

            // 3. Mutates safely in place. If returns an error, `bytes` goes out of scope and is dropped,
            // while `self.almanac` remains in its previous state
            self
              .almanac
              .load_from_bytes_mut(bytes, path_str)
              .map_err(|_| EngineError::InvalidOperation("[Almanac] Error loading SPK file"))?;
            // TODO err if no counters (spk counter, bpc counter, ...) has increased
            // TODO remove

            self.file_names.push(alloc::string::String::from(path_str));
          }
        }
      }
    }

    aethervk_oshal_rlib::log!("Loaded {} SPK files", self.almanac.num_loaded_spk());

    Ok(())
  }

  /// TODO: Document this item
  pub fn get_ephem_full(
    &self,
    spk_id: i32,
    frame: anise::frames::Frame,
    epoch: anise::time::Epoch,
    allow_barycentre_fallback: bool,
    mandatory_rotation: bool,
  ) -> EngineResult<KinematicState> {
    let cartesian_state = self.get_cartesian_state(
      spk_id,
      frame.orientation_id,
      frame.ephemeris_id,
      epoch,
      allow_barycentre_fallback,
    )?;
    let pos = Vec3f32::from_nalgebra_scaled(cartesian_state.radius_km, DISTANCE_SCALE_FACTOR);
    let vel = Vec3f32::from_nalgebra_scaled(cartesian_state.velocity_km_s, DISTANCE_SCALE_FACTOR);

    // In SPICE, body-fixed IAU frame IDs conventionally match their base body ID (e.g. Earth = 399)
    let body_frame = anise::frames::Frame::new(spk_id, spk_id);
    // ask the almanac for the rotation matrix from body frame to inertial world space
    let (rotation, angular_velocity) =
      if let Ok(dcm) = self.almanac.rotate(body_frame, frame, epoch) {
        // direction cosine matrix present, extract rotational information
        let r = Mat3f32::from_nalgebra(dcm.rot_mat);

        let angular_velocity_rad_s = if let Some(rot_mat_dt_anise) = dcm.rot_mat_dt {
          // Note: This derivative is computed by Anise with respect to TDB (Barycentric Dynamical Time) seconds.
          // For our macro-scale rigid-body/particle kinematics, 1 TDB second maps 1:1 to 1 standard SI simulation second.
          // Map Anise's time derivative to your Mat3f32
          let r_dt = Mat3f32::from_nalgebra(rot_mat_dt_anise);

          // 1. Get the Hat Matrix (assuming your Mat3f32 implements Mul and Transpose)
          let omega_hat_world = r_dt * r.transpose();

          // 2. Extract the Vec3f32
          Some(vee(omega_hat_world))
        } else {
          // Fallback: Some static or low-fidelity frames don't have angular velocity
          None
        };

        (Some(Quat::from_rotation_matrix(&r)), angular_velocity_rad_s)
      } else {
        (None, None)
      };

    match (rotation, mandatory_rotation) {
      (None, true) => Err(EngineError::InvalidOperation(
        "[Almanac] get_ephem_full: Rotation was mandatory but not found",
      )),
      (maybe_rot, _) => Ok(KinematicState {
        position: pos,
        velocity: vel,
        rotation: maybe_rot,
        angular_velocity,
      }),
    }
  }

  /// TODO: Document this item
  pub fn get_ephem_frame(
    &self,
    spk_id: i32,
    frame: anise::frames::Frame,
    epoch: anise::time::Epoch,
    allow_barycentre_fallback: bool,
  ) -> EngineResult<Vec3f32> {
    self.get_ephem(
      spk_id,
      frame.orientation_id,
      frame.ephemeris_id,
      epoch,
      allow_barycentre_fallback,
    )
  }

  /// `spk_id` should be a valid celestial body identifier, eg, if planet, use [`anise::constants::celestial_objects`] constants
  /// `orientation` should be a axes orientation identifier from [`anise::constants::orientations`]
  /// `observer` should be a valid barycenter identifier from [`anise::constants::celestial_objects`]
  pub fn get_ephem(
    &self,
    spk_id: i32,
    orientation: i32,
    observer: i32,
    epoch: anise::time::Epoch,
    allow_barycentre_fallback: bool,
  ) -> EngineResult<Vec3f32> {
    let cartesian_state = self.get_cartesian_state(
      spk_id,
      orientation,
      observer,
      epoch,
      allow_barycentre_fallback,
    )?;

    let pos = Vec3f32::from_nalgebra_scaled(cartesian_state.radius_km, DISTANCE_SCALE_FACTOR);

    Ok(pos)
  }

  fn get_cartesian_state(
    &self,
    spk_id: i32,
    orientation: i32,
    observer: i32,
    epoch: anise::time::Epoch,
    allow_barycentre_fallback: bool,
  ) -> EngineResult<anise::math::cartesian::CartesianState> {
    let cartesian_state = 'cartesian_state: {
      // 1. Attempt to get the precise planet center (e.g., 399 for Earth)
      let res = self.almanac.spk_ezr(spk_id, epoch, orientation, observer, None);
      if res.is_err() && allow_barycentre_fallback {
        // 2. FALLBACK: If the precise ID fails, fall back to the planet's system
        // Barycentre (e.g., 199 / 100 = 1 for Mercury).
        let barycentre = spk_id / 100;
        if barycentre > 0 {
          break 'cartesian_state self.almanac.spk_ezr(
            barycentre,
            epoch,
            orientation,
            observer,
            None,
          );
        }
      }
      res
    }
    .map_err(|e| {
      aethervk_oshal_rlib::log!("[Almanac] error: {}", e);
      EngineError::InvalidOperation("[Almanac] couldn't get ephemeris data")
    })?;
    Ok(cartesian_state)
  }
}

// utils
trait VecTypeConversion<T> {
  fn from_nalgebra(v: T) -> Self;
}

trait VecTypeConversionScaled<T> {
  fn from_nalgebra_scaled(v: T, factor: f64) -> Self;
}

impl VecTypeConversion<anise::math::Matrix3> for Mat3f32 {
  fn from_nalgebra(value: anise::math::Matrix3) -> Self {
    Mat3f32::from_columns(
      Vec3f32::from_components(
        value[(0, 0)] as f32,
        value[(1, 0)] as f32,
        value[(2, 0)] as f32,
      ),
      Vec3f32::from_components(
        value[(0, 1)] as f32,
        value[(1, 1)] as f32,
        value[(2, 1)] as f32,
      ),
      Vec3f32::from_components(
        value[(0, 2)] as f32,
        value[(1, 2)] as f32,
        value[(2, 2)] as f32,
      ),
    )
  }
}

impl VecTypeConversionScaled<anise::math::Vector3> for Vec3f32 {
  fn from_nalgebra_scaled(value: anise::math::Vector3, factor: f64) -> Self {
    Vec3f32::from_components(
      (value[0] / DISTANCE_SCALE_FACTOR) as f32,
      (value[1] / DISTANCE_SCALE_FACTOR) as f32,
      (value[2] / DISTANCE_SCALE_FACTOR) as f32,
    )
  }
}
