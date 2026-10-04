//! reposition — shared helpers for forced repositioning of planet/comet entity hierarchies.
//!
//! "Forced repositioning" is the operation of snapping a subtree + body entity pair to the
//! almanac-driven position at a specific epoch. It is called:
//!   - At scene creation time (when `can_move_earth` is true in `create_empty_scene2`)
//!   - When the epoch range changes (`SetEpochRange` handler in `logic_thread`)
//!
//! This is DISTINCT from the per-frame "FRAME SHIFT" in `logic_thread` (~line 2040), which only
//! triggers when the body drifts >0.1 AU during a running simulation.

use crate::{
  scene::{AlmanacPlanet, BodyRotationalModel, EntityId, Scene, TransformComponent},
  simulation::almanac::AlmanacPackedData,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec3f64::DVec3};

/// Single source of truth for km <-> AU (IAU 2012 astronomical unit). Micro frames are in AU,
/// body residuals and ANISE output in km.
pub const KM_TO_AU: f64 = 6.6845871226706e-9_f64;
pub const AU_TO_KM: f64 = 149_597_870.7_f64;
/// A body farther than this from its micro frame origin triggers a frame shift.
pub const FRAME_SHIFT_THRESHOLD_KM: f64 = 0.1 * AU_TO_KM;

/// TAI-second span of the Earth orbit trajectory for a committed `[start, end]` range.
///
/// Ranges shorter than a year get one full orbit **centred on the window**: after 365.25 days the
/// Earth is not back at the same point (sidereal year 365.256 d plus the Moon wobble), so the
/// drawn orbit has a small seam, and it must sit on the far side of the orbit, never where the
/// Earth can be. Starting the span at `start` put that ~26 000 km gap right on the Earth at the
/// start epoch (earth_trajectory_cut.rdc). Longer ranges are covered with a day of margin.
pub fn earth_orbit_span_tai(start: hifitime::Epoch, end: hifitime::Epoch) -> (f64, f64) {
  const DAY: f64 = 86_400.0;
  const YEAR: f64 = 365.25 * DAY;
  let (s, e) = (start.to_tai_seconds(), end.to_tai_seconds());
  if e - s < YEAR - 2.0 * DAY {
    let mid = 0.5 * (s + e);
    (mid - 0.5 * YEAR, mid + 0.5 * YEAR)
  } else {
    (s - DAY, e + DAY)
  }
}

/// Whether the Earth orbit trajectory covering `covered` (TAI seconds) must be rebuilt for the
/// committed `[start, end]` range. Keyed by coverage, not by calendar year: moving the start later
/// within the same year used to keep the old `[old start, old start + 1 yr]` arc.
pub fn needs_earth_orbit_rebuild(
  covered: Option<(f64, f64)>,
  start: hifitime::Epoch,
  end: hifitime::Epoch,
) -> bool {
  covered.map_or(true, |(a, b)| {
    start.to_tai_seconds() < a || end.to_tai_seconds() > b
  })
}

/// Returns TAI seconds for exactly one Julian year (365.25 days) from the given epoch.
/// Used to compute the full-year range for `UpdateTrajectoryForSpk` so the Earth orbit closes.
pub fn full_year_tai_seconds(start: hifitime::Epoch) -> (f64, f64) {
  let start_sec = start.to_tai_seconds();
  let end_sec = start_sec + 365.25 * 86400.0;
  (start_sec, end_sec)
}

pub fn compute_macro_and_residual(position_km: DVec3) -> (Vec3f32, Vec3f32) {
  let subtree_pos_f32: Vec3f32 = (position_km * KM_TO_AU).to_f32();
  let subtree_km = DVec3::from_components(
    subtree_pos_f32.x() as f64,
    subtree_pos_f32.y() as f64,
    subtree_pos_f32.z() as f64,
  ) * AU_TO_KM;
  let residual_f32: Vec3f32 = (position_km - subtree_km).to_f32();
  (subtree_pos_f32, residual_f32)
}

/// Snaps `subtree` (AU frame, child of root) and `body` (km-residual frame, child of subtree)
/// to the almanac-driven position at `epoch`.
///
/// Writes directly to the ECS via `Scene::with_component_mut`. Does NOT touch the
/// `cartesian_state_cache` (which is only populated once the logic thread starts its per-frame
/// sweep after `AlmanacPlanet` is attached to the body entity).
///
/// # Procedure
///
/// 1. `AlmanacPlanet::step(epoch)` → `(position_km: DVec3, rotation: Quat)` in SUN_ECLIPJ2000.
/// 2. Convert km → AU (f64).
/// 3. Lossy f64→f32 truncation → subtree.position (AU frame; scale=AU_TO_KM applied by renderer).
/// 4. Compute km residual from precision loss → body.position.
/// 5. Write rotation → body.rotation.
///
/// This two-level split mirrors the per-frame NORMAL DRIFT / FRAME SHIFT logic in `logic_thread`.
pub fn force_reposition(
  scene: &Scene,
  subtree: EntityId,
  body: EntityId,
  almanac: &AlmanacPackedData,
  planet: &AlmanacPlanet,
  epoch: hifitime::Epoch,
) -> EngineResult<()> {
  let rot_model = scene.with_component(body, |m: &BodyRotationalModel| *m);
  let (position_km, rotation) = planet.step(epoch, almanac, rot_model.as_ref())?;
  apply_reposition(scene, subtree, body, position_km, rotation);
  Ok(())
}

/// Second half of [`force_reposition`] for callers that stepped the almanac themselves, e.g. to
/// avoid holding a scene lock while acquiring `logic_state` (see `BuildCometTrajectory`).
pub fn apply_reposition(
  scene: &Scene,
  subtree: EntityId,
  body: EntityId,
  position_km: DVec3,
  rotation: aethervk_oshal_rlib::math::vector::vec4::Quat,
) {
  let (subtree_pos_f32, residual_f32) = compute_macro_and_residual(position_km);

  // Subtree: lossy f32 (AU frame). Scale = AU_TO_KM is applied by the renderer, not here.
  let _ = scene.with_component_mut(subtree, |t: &mut TransformComponent| {
    t.position = subtree_pos_f32;
  });

  let _ = scene.with_component_mut(body, |t: &mut TransformComponent| {
    t.position = residual_f32;
    t.rotation = rotation;
  });
}

#[cfg(test)]
mod earth_orbit_coverage_tests {
  use super::*;
  use hifitime::{Duration, Epoch};

  /// Sub-year windows get one orbit centred on them: the seam (where the orbit does not close)
  /// is half a year away from every epoch of the window.
  #[test]
  fn seam_is_opposite_the_window() {
    let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
    let end = start + Duration::from_days(30.0);
    let (a, b) = earth_orbit_span_tai(start, end);
    assert!((b - a - 365.25 * 86_400.0).abs() < 1e-3, "one full orbit");
    for t in [start, end] {
      let t = t.to_tai_seconds();
      let to_seam = (t - a).min(b - t) / 86_400.0;
      assert!(to_seam > 160.0, "seam only {to_seam} days from the window");
    }
    assert!(!needs_earth_orbit_rebuild(Some((a, b)), start, end));
  }

  #[test]
  fn later_start_rebuilds_and_inside_does_not() {
    let jan = Epoch::from_gregorian_utc(2025, 1, 10, 0, 0, 0, 0);
    let covered = earth_orbit_span_tai(jan, jan + Duration::from_days(30.0));
    let feb = Epoch::from_gregorian_utc(2025, 2, 1, 0, 0, 0, 0);
    assert!(!needs_earth_orbit_rebuild(
      Some(covered),
      feb,
      feb + Duration::from_days(10.0)
    ));
    let dec = Epoch::from_gregorian_utc(2025, 12, 20, 0, 0, 0, 0);
    assert!(needs_earth_orbit_rebuild(
      Some(covered),
      dec,
      dec + Duration::from_days(30.0)
    ));
    assert!(needs_earth_orbit_rebuild(
      None,
      feb,
      feb + Duration::from_days(1.0)
    ));
  }

  #[test]
  fn multi_year_range_is_covered_with_margin() {
    let start = Epoch::from_gregorian_utc(2025, 1, 1, 0, 0, 0, 0);
    let end = start + Duration::from_days(3.0 * 365.25);
    let (a, b) = earth_orbit_span_tai(start, end);
    assert!(a < start.to_tai_seconds() && b > end.to_tai_seconds());
    assert!(!needs_earth_orbit_rebuild(Some((a, b)), start, end));
  }
}
