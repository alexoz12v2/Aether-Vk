use crate::gpu::{RenderDeviceHandle, RenderFrontend};
use aethervk_oshal_rlib::os::{native::sleep_for, time::{TimeInfo, TimeReadings}};

pub mod comet;

pub trait Pausable {
  fn is_paused(&self) -> bool {
    self.time_scale() == 0.0
  }
  fn set_paused(&mut self) {
    self.set_time_scale(0.0);
  }
  fn time_scale(&self) -> f32;
  fn set_time_scale(&mut self, scale: f32);
}

/// A generic update loop function.
///
/// This function structures a standard game loop with distinct update and fixed_update steps.
///
/// # Type Parameters
///
/// * `S`: The type of the state object that will be passed to the callbacks.
/// * `SR`: A closure type that determines if the loop should continue running.
/// * `U`: A closure type for the main update logic.
/// * `F`: A closure type for the fixed-step update logic (e.g., for physics).
///
/// # Parameters
///
/// * `state`: The initial state for the simulation.
/// * `should_run`: A closure that is called at the beginning of each loop iteration.
///   If it returns `false`, the loop terminates.
/// * `update`: A closure that is called once per loop iteration to perform main update logic.
/// * `fixed_update`: A closure that is called zero or more times per loop iteration to
///   perform fixed-step logic. It is called as many times as needed to catch up with the
///   current time.
/// * `fixed_delta_time`: The duration of each fixed-step in microseconds.
/// * `maximum_delta_time`: The maximum duration that a single frame's delta time can be,
///   to prevent spiraling in case of long hitches. In microseconds.
/// * `time_scale`: A factor to scale time, allowing for slow-motion or speed-up effects.
pub fn run<'a, S, SR, PE, U, F>(
  mut state: S,
  render_frontend: &'a RenderFrontend,
  render_device_handle: RenderDeviceHandle,
  mut should_run: SR,
  mut poll_events: PE,
  mut update: U,
  mut fixed_update: F,
  fixed_delta_time: i64,
  maximum_delta_time: i64,
) where
  S: Pausable,
  SR: FnMut(&S) -> bool,
  PE: FnMut(&mut S),
  U: FnMut(&mut S, &TimeReadings, &'a RenderFrontend, RenderDeviceHandle),
  F: FnMut(&mut S, &TimeReadings, &'a RenderFrontend, RenderDeviceHandle),
{
  let mut time_info = TimeInfo::new(fixed_delta_time, maximum_delta_time, state.time_scale());

  while should_run(&state) {
    poll_events(&mut state);

    time_info.set_time_scale(state.time_scale());
    if !state.is_paused() {
      time_info.ut_update();

      while time_info.needs_fixed_update() {
        time_info.ut_fixed_update();
        fixed_update(
          &mut state,
          &time_info.current(),
          render_frontend,
          render_device_handle,
        );
      }

      update(
        &mut state,
        &time_info.current(),
        render_frontend,
        render_device_handle,
      );
    } else {
      sleep_for(core::time::Duration::from_millis(16));
    }
  }
}
