use super::*;
use crate::simulation_api::{SimulationContext, RenderCommand, BREADCRUMB_CALLBACK};
use aethervk_core_rlib::scene::CameraComponent;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_core_rlib::types::EngineError;
use core::ffi::c_char;

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
    let res = Some(self.render_frontend.with_device(self.render_device_handle, |device| {
        match device.is_task_completed(task_id) {
          Ok(true) => status = 1,
          Ok(false) => status = 0,
          Err(_) => status = 2,
        }
        Ok(())
    }).map_err(aethervk_core_rlib::types::EngineError::from));

    if res.is_none() {
      return -1;
    }
    status
  }

  pub fn resize(&mut self, width: u32, height: u32) -> Result<(), EngineError> {
    if let Some(scene_ctx_arc) = self.active_scene_clone() {
      let mut active = scene_ctx_arc.write();
      
      
      
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
      
      drop(active);
      let _ = self
        .render_tx
        .try_send(RenderCommand::Resize { width, height,  });
    }
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
