use super::*;
use crate::simulation_api::SimulationContext;
use alloc::{vec::Vec, string::String, boxed::Box, sync::Arc, format, collections::BTreeMap};
use core::ffi::{c_char, CStr};
use aethervk_core_rlib::scene::*;
use aethervk_core_rlib::gpu::*;
use aethervk_core_rlib::types::*;
use aethervk_oshal_rlib::math::vector::*;
use aethervk_oshal_rlib::math::quaternion::*;
use aethervk_oshal_rlib::math::matrix::*;

impl SimulationContext {
  pub fn set_logger_callback(cb: Option<extern "C" fn(*const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    aethervk_oshal_rlib::os::debug::LOGGER_CALLBACK
      .store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  pub fn set_breadcrumb_callback(cb: Option<extern "C" fn(u32, *const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    BREADCRUMB_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  pub fn get_task_status(&mut self, task_id: u64) -> i32 {
    let mut status = 0; // 0: Pending, 1: Success, 2: Failed
    let res = self.render_frontend.take_and(|context| {
      context
        .deref_device_and(
          self.render_device_handle,
          &mut (task_id, &mut status) as *mut _ as *mut core::ffi::c_void,
          |device, data| {
            let (tid, s) = unsafe { &mut *(data as *mut (u64, &mut i32)) };
            match device.is_task_completed(*tid) {
              Ok(true) => **s = 1,
              Ok(false) => **s = 0,
              Err(_) => **s = 2,
            }
            Ok(())
          },
        )
        .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
        .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
    });

    if res.is_none() {
      return -1;
    }
    status
  }

  pub fn resize(&mut self, width: u32, height: u32) -> Result<(), EngineError> {
    self.window_width = width;
    self.window_height = height;
    
    if let Some(active) = self.active_scene() {
      let target_entity = active.active_camera_entity;
      active.scene.with_component_mut(
        target_entity,
        |c: &mut CameraComponent| {
          c.projection = Mat4x4f32::perspective_vk(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            10000.0,
          );
        },
      );
    }

    let _ = self
      .render_tx
      .try_send(RenderCommand::Resize { width, height });
    Ok(())
  }

  pub fn set_asset_path(path: *const c_char) {
    if path.is_null() {
      return;
    }

    if let Ok(c_str) = unsafe { core::ffi::CStr::from_ptr(path) }.to_str() {
      let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
      *guard = Some(alloc::string::String::from(c_str));
    }
  }

  pub fn log(msg: *const c_char) {
    let fptr =
      aethervk_oshal_rlib::os::debug::LOGGER_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
    if !fptr.is_null() {
      let cb: extern "C" fn(*const c_char) = unsafe { core::mem::transmute(fptr) };
      cb(msg);
    }
  }
}
