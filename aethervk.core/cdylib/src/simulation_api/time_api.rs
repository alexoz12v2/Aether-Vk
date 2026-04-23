use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{format};
use core::ffi::{c_char};

impl SimulationContext {
  pub fn set_time_scale(&mut self, scale: u32) {
    let mut logic = self.logic_state.write();
    logic.current_scale = match scale {
      1 => TimeScale::OneDay,
      2 => TimeScale::OneWeek,
      3 => TimeScale::OneMonth,
      _ => TimeScale::Stopped,
    };
  }

  pub fn get_simulation_time(&self) -> f64 {
    let logic = self.logic_state.read();
    logic.current_epoch.to_tai_seconds()
  }

  pub fn get_simulation_time_utc(&self, buffer: *mut c_char, buffer_len: u32) -> bool {
    if buffer.is_null() || buffer_len == 0 {
      return false;
    }
    let logic = self.logic_state.read();
    let utc_str = format!("{}", logic.current_epoch);

    let bytes = utc_str.as_bytes();
    let copy_len = core::cmp::min(bytes.len(), (buffer_len - 1) as usize);

    unsafe {
      let dest = core::slice::from_raw_parts_mut(buffer as *mut u8, buffer_len as usize);
      dest[..copy_len].copy_from_slice(&bytes[..copy_len]);
      dest[copy_len] = 0; 
    }

    true
  }

  pub fn set_simulation_time(&mut self, time_tai: f64) {
    let mut logic = self.logic_state.write();
    logic.current_epoch = anise::time::Epoch::from_tai_seconds(time_tai);
  }
}
