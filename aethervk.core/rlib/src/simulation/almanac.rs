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

/// Component used to drive kinematic planets based on an SPK ephemeris trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct AlmanacPlanetComponent {
  pub spk_id: i32,
}

/// There is no need to track [`bytes::Bytes`] of SPK data cause it's an Arc based type, meaning
/// as long as ['anise::almanac::Almanac'] holds onto it, we are fine
#[derive(Default)]
pub struct AlmanacPackedData {
  pub file_names: Vec<String>,
  /// the almanac itself
  pub almanac: anise::almanac::Almanac,
  pub missing_rotation_logs: spin::Mutex<alloc::collections::BTreeSet<i32>>,
  /// Tracks which NAIF IDs are covered by loaded SPK data and their epoch spans.
  /// Maps spk_id -> (earliest_epoch, latest_epoch) across all loaded SPK files.
  pub spk_coverage: dashmap::DashMap<i32, (anise::time::Epoch, anise::time::Epoch)>,
}

impl AlmanacPackedData {
  // TODO `celestial_name_from_id` to create planet/sun entities
  /// TODO: Document this item
  pub fn load_almanac<P: AsRef<os::fs::Path>>(&mut self, path: P) -> EngineResult<()> {
    let path_ref = path.as_ref();
    let is_valid_ext =
      |ext: Option<&str>| ext == Some("bsp") || ext == Some("bpc") || ext == Some("tpc");

    if path_ref.is_file() && is_valid_ext(path_ref.extension().as_deref()) {
      self.load_single_spk(path_ref)?;
    } else if let Ok(entries) = os::fs::read_dir(path_ref) {
      for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_file() && is_valid_ext(entry_path.extension().as_deref()) {
          self.load_single_spk(&entry_path)?;
        }
      }
    }

    aethervk_oshal_rlib::log!(
      "Loaded {} SPK files, {} BPC files",
      self.almanac.num_loaded_spk(),
      self.almanac.num_loaded_bpc()
    );

    Ok(())
  }

  /// Loads an SPK file into a temporary almanac instance and probes whether
  /// the SPK data covers the given epoch range for the specified NAIF ID.
  /// Returns true if the SPK segment summary spans [start_epoch, end_epoch].
  ///
  /// Uses `spk_domain()` which reads SPK segment summaries directly —
  /// no ephemeris path resolution is performed, so this works with just
  /// the comet's SPK file (no de440s.bsp required).
  pub fn probe_spk_file<P: AsRef<os::fs::Path>>(
    path: P,
    spk_id: i32,
    start_epoch: anise::time::Epoch,
    end_epoch: anise::time::Epoch,
  ) -> bool {
    let (covers, _, _) = Self::probe_spk_file_with_domain(path, spk_id, start_epoch, end_epoch);
    covers
  }

  /// Like `probe_spk_file`, but also returns the actual SPK domain (start, end)
  /// and the **discovered NAIF ID** (which may differ from the requested `spk_id`).
  /// Returns `(false, None, 0)` when the file or ID lookup fails entirely.
  ///
  /// Tries the requested `spk_id` first, then its negation, then falls back
  /// to `spk_domains()` to find the target body in the file. This handles
  /// JPL Horizons SPK files where the record number (e.g. 90000702) differs
  /// from the NAIF target_id stored in the SPK segment (e.g. 1000012).
  pub fn probe_spk_file_with_domain<P: AsRef<os::fs::Path>>(
    path: P,
    spk_id: i32,
    start_epoch: anise::time::Epoch,
    end_epoch: anise::time::Epoch,
  ) -> (bool, Option<(anise::time::Epoch, anise::time::Epoch)>, i32) {
    let path_ref = path.as_ref();

    // Load the SPK directly into a bare Almanac — we intentionally bypass
    // AlmanacPackedData::load_almanac because that triggers refresh_spk_coverage,
    // which calls translate_geometric with the Sun as observer and panics when
    // the planetary DE ephemeris (de440s.bsp) is not loaded.
    let mmap = match Mmap::open(path_ref) {
      Ok(m) => m,
      Err(_) => return (false, None, 0),
    };
    let bytes = bytes::Bytes::from_owner(mmap);

    let mut almanac = anise::almanac::Almanac::default();
    let alias = path_ref
      .to_str_unified()
      .and_then(|c| c.to_str().map(alloc::string::String::from))
      .unwrap_or_else(|| alloc::string::String::from("probe"));

    if almanac.load_from_bytes_mut(bytes, &alias).is_err() {
      return (false, None, 0);
    }

    // Horizons may round SPK boundaries by ±1 day. Allow a 2-day tolerance
    // so we don't reject a valid file that's off by one day at the edges.
    let tolerance = anise::time::Duration::from_days(2.0);

    let ids_to_try = [spk_id, -spk_id];
    for &try_id in &ids_to_try {
      if let Ok((domain_start, domain_end)) = almanac.spk_domain(try_id) {
        let covers = domain_start <= start_epoch + tolerance
          && domain_end >= end_epoch - tolerance;
        return (covers, Some((domain_start, domain_end)), try_id);
      }
    }

    // Neither sign matched — fall back to spk_domains() to find the target body.
    // Filter out ID 0 (Solar System Barycenter) which is always present as the
    // center body in Horizons SPKs but is not the target we're looking for.
    if let Ok(domains) = almanac.spk_domains() {
      let target_bodies: alloc::vec::Vec<(i32, (anise::time::Epoch, anise::time::Epoch))> = domains
        .into_iter()
        .filter(|&(id, _)| id != 0) // exclude SSB center body
        .collect();

      if target_bodies.len() == 1 {
        let (found_id, (ds, de)) = target_bodies[0];
        let covers = ds <= start_epoch + tolerance
          && de >= end_epoch - tolerance;
        return (covers, Some((ds, de)), found_id);
      }

      // Multiple target bodies — return the union of all domains for diagnostic,
      // but we can't pick a single ID, so return 0.
      if !target_bodies.is_empty() {
        let mut earliest = target_bodies[0].1.0;
        let mut latest = target_bodies[0].1.1;
        for &(_, (s, e)) in &target_bodies[1..] {
          if s < earliest { earliest = s; }
          if e > latest { latest = e; }
        }
        return (false, Some((earliest, latest)), 0);
      }
    }

    (false, None, 0)
  }

  fn load_single_spk(&mut self, path: &os::fs::Path) -> EngineResult<()> {
    if let Some(path_cow) = path.to_str_unified() {
      let path_str = path_cow.to_str().unwrap();

      let mmap = Mmap::open(path).map_err(|e| EngineError::from(e))?;
      let bytes = bytes::Bytes::from_owner(mmap);

      self
        .almanac
        .load_from_bytes_mut(bytes, path_str)
        .map_err(|_| EngineError::InvalidOperation("[Almanac] Error loading SPK file"))?;

      self.file_names.push(alloc::string::String::from(path_str));

      // Attempt to populate SPK coverage from loaded data summaries.
      // We iterate SPK summaries by index and extract NAIF ID + epoch range.
      self.refresh_spk_coverage();
    }
    Ok(())
  }

  /// Unloads a previously loaded SPK file by its alias (which is the file path).
  pub fn unload_almanac_spk(&mut self, path: &str) -> EngineResult<()> {
    if let Err(e) = self.almanac.spk_unload(path) {
      aethervk_oshal_rlib::log!("Failed to unload SPK {}: {}", path, e);
      return Err(EngineError::InvalidOperation(
        "[Almanac] Error unloading SPK file",
      ));
    }

    self.file_names.retain(|f| f != path);
    // Re-scan coverage after unloading
    self.spk_coverage.clear();
    self.refresh_spk_coverage();
    aethervk_oshal_rlib::log!(
      "Unloaded SPK {}, {} remaining",
      path,
      self.almanac.num_loaded_spk()
    );

    Ok(())
  }

  /// Re-scan all loaded SPK data and rebuild the spk_coverage map.
  /// ANISE exposes `spk_summary_at_epoch` but not a full iteration API,
  /// so we scan known NAIF IDs by trying small test epochs.
  fn refresh_spk_coverage(&self) {
    // Try common solar system body IDs (planets, barycenters, comets)
    // plus any custom IDs we may encounter.
    let candidate_ids: &[i32] = &[
      // Sun, major planets, barycenters
      10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 199, 299, 399, 499, 599, 699, 799, 899, 999,
      // Barycenters
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, // Moon, common comets/asteroids (extend as needed)
      301,
    ];

    // Test epochs spanning a wide range
    let test_epochs = [
      anise::time::Epoch::from_tdb_seconds(-3.155e9), // ~1900
      anise::time::Epoch::from_tdb_seconds(0.0),      // J2000
      anise::time::Epoch::from_tdb_seconds(3.155e9),  // ~2100
      anise::time::Epoch::from_tdb_seconds(6.311e9),  // ~2200
    ];

    for &id in candidate_ids {
      let mut earliest = None;
      let mut latest = None;

      for &epoch in &test_epochs {
        // Try to evaluate ephemeris at this epoch
        let result = self.almanac.translate_geometric(
          anise::frames::Frame::new(id, anise::constants::orientations::J2000),
          anise::frames::Frame::new(
            anise::constants::celestial_objects::SUN,
            anise::constants::orientations::J2000,
          ),
          epoch,
        );
        if result.is_ok() {
          match earliest {
            None => earliest = Some(epoch),
            Some(e) if epoch < e => earliest = Some(epoch),
            _ => {}
          }
          match latest {
            None => latest = Some(epoch),
            Some(l) if epoch > l => latest = Some(epoch),
            _ => {}
          }
        }
      }

      if let (Some(start), Some(end)) = (earliest, latest) {
        self.spk_coverage.insert(id, (start, end));
      }
    }
  }

  /// Returns the cached epoch coverage for a given NAIF ID, if known.
  pub fn get_coverage(&self, spk_id: i32) -> Option<(anise::time::Epoch, anise::time::Epoch)> {
    self.spk_coverage.get(&spk_id).map(|r| *r.value())
  }

  /// Returns true if the loaded SPK data covers the entire interval [start, end] for the given NAIF ID.
  pub fn covers_interval(
    &self,
    spk_id: i32,
    start: anise::time::Epoch,
    end: anise::time::Epoch,
  ) -> bool {
    if let Some(entry) = self.spk_coverage.get(&spk_id) {
      let (cov_start, cov_end) = *entry.value();
      cov_start <= start && cov_end >= end
    } else {
      false
    }
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

    // In SPICE/NAIF, body-fixed IAU frame IDs for simple planetary bodies conventionally match
    // their base ephemeris body ID (e.g., body 499 maps to frame 499).
    let body_frame = anise::frames::Frame::new(spk_id, spk_id);

    let mut dcm_result = self.almanac.rotate(body_frame, frame, epoch);

    // Fallbacks for Earth: high precision Binary PCKs (BPC files like earth_latest_high_prec.bpc)
    // do not store orientation data under the ephemeris ID 399. Instead, they store Earth's
    // orientation under the International Terrestrial Reference Frame ITRF93 (ID: 13000),
    // or the standard IAU_EARTH frame (ID: 10013). Because of this quirk in NAIF convention,
    // we must explicitly try these standard Earth frames when retrieving its rotation matrix.
    let mut resolved_frame = spk_id;
    if dcm_result.is_err() && spk_id == 399 {
      dcm_result = self.almanac.rotate(anise::constants::frames::EARTH_ITRF93, frame, epoch);
      if dcm_result.is_err() {
        dcm_result = self.almanac.rotate(anise::constants::frames::IAU_EARTH_FRAME, frame, epoch);
        if dcm_result.is_ok() {
          resolved_frame = anise::constants::frames::IAU_EARTH_FRAME.orientation_id;
        }
      } else {
        resolved_frame = anise::constants::frames::EARTH_ITRF93.orientation_id;
      }
    }

    // ask the almanac for the rotation matrix from body frame to inertial world space
    let (rotation, angular_velocity) = if let Ok(dcm) = dcm_result {
      if spk_id == 399 {
        let mut missing_logs = self.missing_rotation_logs.lock();
        if missing_logs.insert(10399) {
          // Using 10399 just as a dummy key to ensure we only log once
          aethervk_oshal_rlib::log!(
            "[Almanac] Successfully fetched rotation for Earth using frame ID: {}",
            resolved_frame
          );
        }
      }

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
      let mut missing_logs = self.missing_rotation_logs.lock();
      if missing_logs.insert(spk_id) {
        aethervk_oshal_rlib::log!(
          "[Almanac] Could not fetch rotation for body {} to frame {:?} (Will not log again for this body)",
          spk_id,
          frame
        );
      }
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
  fn from_nalgebra_scaled(value: anise::math::Vector3, _factor: f64) -> Self {
    Vec3f32::from_components(
      (value[0] / DISTANCE_SCALE_FACTOR) as f32,
      (value[1] / DISTANCE_SCALE_FACTOR) as f32,
      (value[2] / DISTANCE_SCALE_FACTOR) as f32,
    )
  }
}