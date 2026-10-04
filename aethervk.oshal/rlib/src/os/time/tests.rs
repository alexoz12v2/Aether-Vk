use super::*;
use crate::os::time::v1::TimeInfo;
const IDEAL_DELTA_TIME: timeus_t = 16_667; // 60 FPS

#[test]
fn test_time_info_initialization() {
  let fixed_dt = timeus_milliseconds(16);
  let max_dt = timeus_milliseconds(33);
  let time_info = TimeInfo::new(fixed_dt, max_dt, 1.0);

  let readings = time_info.current();
  assert_eq!(readings.delta_time, IDEAL_DELTA_TIME);
}

#[test]
fn test_time_scale() {
  let fixed_dt = timeus_milliseconds(16);
  let max_dt = timeus_milliseconds(33);
  let mut time_info = TimeInfo::new(fixed_dt, max_dt, 1.0);

  time_info.set_time_scale(2.0);
  assert_eq!(time_info.get_time_scale(), 2.0);

  time_info.set_time_scale(-1.0); // should clamp to 0
  assert_eq!(time_info.get_time_scale(), 0.0);
}

/// Seeking moves the clock (clamped to the committed range) and drops pending fixed steps.
#[test]
fn test_time_manager_seek_clamps_and_clears_accumulator() {
  use crate::os::time::v2::{SimSpeed, TimeManager};
  use hifitime::{Duration, Epoch};
  let start = Epoch::from_gregorian_utc(2025, 10, 1, 0, 0, 0, 0);
  let end = start + Duration::from_days(30.0);
  let mut tm = TimeManager::new(start, end, SimSpeed::OneDayPerSec, 100);
  tm.state.write().scaled_accumulator = 123_456;
  let mid = start + Duration::from_days(10.0);
  assert_eq!(tm.seek(mid), mid);
  assert_eq!(tm.current_epoch(), mid);
  assert_eq!(tm.state.read().scaled_accumulator, 0);
  assert_eq!(tm.seek(end + Duration::from_days(5.0)), end);
  assert_eq!(tm.seek(start - Duration::from_days(5.0)), start);
}
