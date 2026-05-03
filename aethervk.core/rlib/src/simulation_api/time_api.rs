use crate::simulation_api::structs::{TimeScale};
use crate::simulation_api::SimulationContext;
use alloc::{format};
use core::ffi::{c_char};

impl SimulationContext {
  pub fn set_time_scale(&self, scene_id: u64, scale: u32) {
    if let Some(scene) = self.get_scene(scene_id) {
      let mut scene_write = scene.write();
      scene_write.time_state.current_scale = match scale {
        1 => TimeScale::OneDay,
        2 => TimeScale::OneWeek,
        3 => TimeScale::OneMonth,
        _ => TimeScale::Stopped,
      };
    }
  }

  pub fn get_simulation_time(&self, scene_id: u64) -> f64 {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      scene_read.time_state.current_epoch.to_tai_seconds()
    } else {
      0.0
    }
  }

  pub fn get_simulation_time_utc(&self, scene_id: u64, buffer: *mut c_char, buffer_len: u32) -> bool {
    if buffer.is_null() || buffer_len == 0 {
      return false;
    }
    
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      let utc_str = format!("{}", scene_read.time_state.current_epoch);

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

  pub fn set_simulation_time(&self, scene_id: u64, time_tai: f64) {
    if let Some(scene) = self.get_scene(scene_id) {
      let mut scene_write = scene.write();
      scene_write.time_state.current_epoch = anise::time::Epoch::from_tai_seconds(time_tai);
    }
  }

  pub fn get_epoch_limits(&self, scene_id: u64, start_tai: *mut f64, end_tai: *mut f64) -> bool {
    if let Some(scene) = self.get_scene(scene_id) {
      let scene_read = scene.read();
      unsafe {
        if !start_tai.is_null() {
          *start_tai = scene_read.time_state.epoch_start.to_tai_seconds();
        }
        if !end_tai.is_null() {
          *end_tai = scene_read.time_state.epoch_end.to_tai_seconds();
        }
      }
      true
    } else {
      false
    }
  }
}
