use super::*;
use crate::simulation_api::{SimulationContext, BREADCRUMB_CALLBACK};
use crate::simulation_api::structs::{RaycastResult, RenderCommand, Resize};
use crate::scene::CameraComponent;
use oshal::math::matrix::mat4::Mat4x4f32;
use crate::types::EngineError;
use core::ffi::c_char;
use crate::gpu::PresentationEngineHandle;
use crate::expect_scene;

impl SimulationContext {
  pub fn set_logger_callback(cb: Option<extern "C" fn(*const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    oshal::os::debug::LOGGER_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  pub fn set_breadcrumb_callback(cb: Option<extern "C" fn(u32, *const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    BREADCRUMB_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  // TODO enum for task status
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
                }
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

  /// Generic so that FFI type can be in cdylib crate
  pub fn get_task_result_raycast<T: From<RaycastResult>>(
    &self,
    task_id: u64,
    out_hit: *mut T,
  ) -> bool {
    if let Some(SimulationTaskResult::Raycast(res)) = self.task_manager.write().take_result(task_id)
    {
      if !out_hit.is_null() {
        unsafe {
          *out_hit = res.into();
        }
      }
      true
    } else {
      false
    }
  }

  /// Generic so that FFI type can be in cdylib crate
  pub fn get_task_result_kinematic_state<T: From<crate::simulation::almanac::KinematicState>>(
    &self,
    task_id: u64,
    out_state: *mut T,
  ) -> bool {
    if let Some(SimulationTaskResult::KinematicState(state)) = self.task_manager.write().take_result(task_id)
    {
      if !out_state.is_null() {
        unsafe {
          *out_state = state.into();
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
  ) -> EngineResult<core::num::NonZero<u64>> {
    let task_id = self
      .render_proxy
      .0
      .as_frontend()
      .ok_or(EngineError::InvalidOperation("render_frontend"))
      .and_then(|context| {
        context
          .with_device(self.render_proxy.1, |device| Ok(device.create_task()))
          .map_err(|e| EngineError::from(e))
      })?;

    let scene_data = self.scenes.read();
    let scene = expect_scene!(scene_data.get_scene(scene_id), "scene_api:resize");
    {
      let scene_write = scene.read();
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
      .try_send(RenderCommand::Resize(Resize {
        presentation_engine_handle,
        width,
        height,
        task_id,
      }));
    Ok(unsafe { core::num::NonZero::new_unchecked(task_id) })
  }

  pub fn download_image(&self, task_id: u64, buffer_ptr: *mut u8, buffer_size: usize) -> bool {
    let result = self
      .render_proxy
      .0
      .as_frontend()
      .ok_or(EngineError::InvalidOperation("render_frontend"))
      .and_then(|frontend| {
        frontend.with_device(self.render_proxy.1, |device| {
          device.read_windowless_download(task_id, unsafe {
            core::slice::from_raw_parts_mut(buffer_ptr, buffer_size)
          })
        })
        .map_err(|e| EngineError::from(e))
      });

    match result {
      Ok(_) => true,
      Err(e) => {
        oshal::log!("download_image err: {:?}", e);
        false
      }
    }
  }

  pub fn set_asset_path(path: &str) {
    let mut guard = crate::gpu::ASSET_DIR.write();
    *guard = Some(alloc::string::String::from(path));
  }
}
