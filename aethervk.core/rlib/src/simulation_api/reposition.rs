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
  scene::{AlmanacPlanet, EntityId, Scene, TransformComponent},
  simulation::almanac::AlmanacPackedData,
  types::EngineResult,
};
use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec3f64::DVec3};

const KM_TO_AU: f64 = 6.6845871226706e-9_f64;
const AU_TO_KM: f64 = 149_597_870.7_f64;

/// Returns the UTC calendar year of the given epoch as `i32`.
pub fn year_of_epoch(e: hifitime::Epoch) -> i32 {
  e.to_gregorian_utc().0
}

/// Returns TAI seconds for exactly one Julian year (365.25 days) from the given epoch.
/// Used to compute the full-year range for `UpdateTrajectoryForSpk` so the Earth orbit closes.
pub fn full_year_tai_seconds(start: hifitime::Epoch) -> (f64, f64) {
  let start_sec = start.to_tai_seconds();
  let end_sec = start_sec + 365.25 * 86400.0;
  (start_sec, end_sec)
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
  let (position_km, rotation) = planet.step(epoch, almanac, None)?;

  // AU position (f64 precision)
  let subtree_pos_f32: Vec3f32 = (position_km * KM_TO_AU).to_f32();

  // Subtree: lossy f32 (AU frame). Scale = AU_TO_KM is applied by the renderer, not here.
  let _ = scene.with_component_mut(subtree, |t: &mut TransformComponent| {
    t.position = subtree_pos_f32;
  });

  // Body: km residual after f32 truncation — recovers sub-AU precision
  let subtree_km = DVec3::from_components(
    subtree_pos_f32.x() as f64,
    subtree_pos_f32.y() as f64,
    subtree_pos_f32.z() as f64,
  ) * AU_TO_KM;
  let residual_f32: Vec3f32 = (position_km - subtree_km).to_f32();

  let _ = scene.with_component_mut(body, |t: &mut TransformComponent| {
    t.position = residual_f32;
    t.rotation = rotation;
  });

  Ok(())
}
