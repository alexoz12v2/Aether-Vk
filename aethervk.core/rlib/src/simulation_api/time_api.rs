//! time_api module.

use crate::simulation_api::SimulationContext;
use crate::simulation_api::structs::TimeScale;
use alloc::format;
use core::ffi::c_char;

impl SimulationContext {
  /// TODO: Document this item
  pub fn set_time_scale(&self, scene_id: u64, scale: u32) {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      let mut time_write = scene_read.time_state.write();
      time_write.current_scale = match scale {
        1 => TimeScale::OneDay,
        2 => TimeScale::OneWeek,
        3 => TimeScale::OneMonth,
        _ => TimeScale::Stopped,
      };
    }
  }

  /// TODO: Document this item
  pub fn get_simulation_time(&self, scene_id: u64) -> f64 {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      scene_read.time_state.read().current_epoch.to_tai_seconds()
    } else {
      0.0
    }
  }

  /// TODO: Document this item
  pub fn get_simulation_time_utc(
    &self,
    scene_id: u64,
    buffer: *mut c_char,
    buffer_len: u32,
  ) -> bool {
    if buffer.is_null() || buffer_len == 0 {
      return false;
    }

    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      let utc_str = format!("{}", scene_read.time_state.read().current_epoch);

      let bytes = utc_str.as_bytes();
      let copy_len = core::cmp::min(bytes.len(), (buffer_len - 1) as usize);

      unsafe {
        let dest = core::slice::from_raw_parts_mut(buffer as *mut u8, buffer_len as usize);
        dest[..copy_len].copy_from_slice(&bytes[..copy_len]);
        dest[copy_len] = 0;
      }

      true
    } else {
      false
    }
  }

  /// TODO: Document this item
  pub fn set_simulation_time(&self, scene_id: u64, time_tai: f64) {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      let mut time_write = scene_read.time_state.write();
      time_write.current_epoch = anise::time::Epoch::from_tai_seconds(time_tai);
    }
  }

  /// TODO: Document this item
  pub fn get_epoch_limits(&self, scene_id: u64, start_tai: *mut f64, end_tai: *mut f64) -> bool {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      let time_read = scene_read.time_state.read();
      unsafe {
        if !start_tai.is_null() {
          *start_tai = time_read.epoch_start.to_tai_seconds();
        }
        if !end_tai.is_null() {
          *end_tai = time_read.epoch_end.to_tai_seconds();
        }
      }
      true
    } else {
      false
    }
  }
}
