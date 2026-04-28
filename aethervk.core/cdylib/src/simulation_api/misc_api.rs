use super::*;
use crate::simulation_api::{SimulationContext, BREADCRUMB_CALLBACK};
use crate::structs::RenderCommand;
use aethervk_core_rlib::scene::CameraComponent;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_core_rlib::types::EngineError;
use core::ffi::c_char;
use aethervk_core_rlib::gpu::PresentationEngineHandle;
use crate::expect_scene;

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

  pub fn get_task_status(&self, task_id: u64) -> i32 {
    if (task_id & (1u64 << 63)) != 0 {
      // Logic task
      self.task_manager.read().get_status(task_id)
    } else {
      // Render task
      self
        .render_proxy
        .0
        .as_frontend()
        .and_then(|frontend| {
          frontend
            .with_device(self.render_proxy.1, |device| {
              let res = device.is_task_completed(task_id);
              match res {
                Ok(true) => Ok(1),
                Ok(false) => Ok(0),
                Err(e) => {
                  oshal::log!("is_task_completed err: {:?}", e);
                  Ok(2)
                },
              }
            })
            .ok()
        })
        .unwrap_or(-1)
    }
  }

  pub fn get_task_result_u64(&self, task_id: u64) -> u64 {
    if let Some(SimulationTaskResult::U64(val)) = self.task_manager.write().take_result(task_id) {
      val
    } else {
      0
    }
  }

  pub fn get_task_result_bool(&self, task_id: u64) -> bool {
    if let Some(SimulationTaskResult::Bool(val)) = self.task_manager.write().take_result(task_id) {
      val
    } else {
      false
    }
  }

  pub fn get_task_result_raycast(&self, task_id: u64, out_hit: *mut FfiRaycastResult) -> bool {
    if let Some(SimulationTaskResult::Raycast(res)) = self.task_manager.write().take_result(task_id)
    {
      if !out_hit.is_null() {
        unsafe {
          *out_hit = res;
        }
      }
      true
    } else {
      false
    }
  }

  pub fn resize(
    &mut self,
    scene_id: u64,
    presentation_engine_handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> EngineResult<()> {
    let scene_data = self.scenes.read();
    let scene = expect_scene!(scene_data.get_scene(scene_id), "scene_api:resize");
    {
      let mut scene_write = scene.write();
      let target_entity = scene_write
        .active_camera_entity
        .ok_or(EngineError::InvalidOperation(
          "scene_api: no active camera in scene",
        ))?;
      scene_write
        .scene
        .with_component_mut(target_entity, |c: &mut CameraComponent| {
          c.projection = Mat4x4f32::perspective_vk(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            10000.0,
          );
        });
    }

    let _ = self
      .threads
      .render_thread
      .tx()
      .try_send(RenderCommand::Resize(crate::structs::Resize {
        presentation_engine_handle,
        width,
        height,
      }));
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
