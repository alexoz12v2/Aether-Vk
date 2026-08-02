//! time_api module.

use crate::{
  simulation_api::structs::SceneContext,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::os::time::v2::{SimSpeed, TimeManager};

pub fn set_time_scale(scene_ctx: &SceneContext, scale: u32) -> EngineResult<()> {
  let mut time_write = scene_ctx.time_state.write();
  time_write.speed = match scale {
    1 => SimSpeed::Realtime,
    2 => SimSpeed::OneDayPerSec,
    3 => SimSpeed::OneHourPerSec,
    4 => SimSpeed::OneDayPerSec,
    _ => {
      return Err(EngineError::InvalidOperation(
        "time scale should be a number from 1 to 4",
      ));
    }
  };

  Ok(())
}

/// To be called from a sync command in the logic thread
/// Note: if this is ok, then timeline state change should be propagated through C# callback. How?
/// Dedicated callback for external states which do not belong to the scene.
pub fn set_epoch_range(
  time_mgr: &mut TimeManager,
  start: hifitime::Epoch,
  end: hifitime::Epoch,
) -> EngineResult<()> {
  if end - start < hifitime::Duration::from_days(1.0) {
    return Err(EngineError::InvalidOperation(
      "end - start should be bigger than 1 day",
    ));
  }

  time_mgr.start_epoch = start;
  time_mgr.end_epoch = end;

  Ok(())
}
