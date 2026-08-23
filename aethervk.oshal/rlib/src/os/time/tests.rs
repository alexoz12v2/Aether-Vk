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
