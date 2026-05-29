//! misc_api module.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::{
  expect_scene,
  gpu::{PresentationEngineHandle, WeakRenderFrontendExt},
  scene::{CameraComponent, EntityId},
  simulation_api::{
    BREADCRUMB_CALLBACK, SimulationContext,
    structs::{RaycastResult, RenderCommand, Resize, SimulationTaskResult, TaskStatusCode},
  },
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{self as oshal};
use core::ffi::c_char;

impl SimulationContext {
  /// TODO: Document this item
  pub fn set_logger_callback(cb: Option<extern "C" fn(*const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    oshal::os::debug::LOGGER_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn set_breadcrumb_callback(cb: Option<extern "C" fn(u32, *const c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    BREADCRUMB_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn set_simulation_callback(
    cb: Option<extern "C" fn(u64, u64, u64, *const core::ffi::c_void)>,
  ) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    crate::simulation_api::SIMULATION_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn set_render_callback(cb: Option<extern "C" fn(u64, u64, u64)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    crate::simulation_api::RENDER_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn get_task_status(&self, task_id: u64) -> TaskStatusCode {
    if task_id == 0 || task_id == u64::MAX {
      return TaskStatusCode::Invalid;
    }

    if (task_id & (1u64 << 63)) != 0 {
      // TODO check correctness for task id construction in logic thread. If so, create RenderTaskId(u64) and LogicTaskId(u64) with new function with debug_assert!
      // Logic task
      self.task_manager.read().get_status(task_id)
    } else {
      // Render task
      let completed = self
        .render_proxy
        .0
        .as_frontend()
        .and_then(|f| {
          f.with_device(self.render_proxy.1, |device| {
            Ok(device.is_task_completed(task_id).unwrap_or(true))
          })
          .ok()
        })
        .unwrap_or(true);

      if completed {
        TaskStatusCode::Completed
      } else {
        TaskStatusCode::Pending
      }
    }
  }

  /// TODO: Document this item
  pub fn get_task_result_u64(&self, task_id: u64) -> u64 {
    if let Some(SimulationTaskResult::U64(val)) = self.task_manager.write().take_result(task_id) {
      val
    } else {
      0
    }
  }

  /// TODO: Document this item
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
    if let Some(SimulationTaskResult::KinematicState(state)) =
      self.task_manager.write().take_result(task_id)
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

  /// TODO: Document this item
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
      let scene_read = scene.read();
      let camera_entity = scene_read
        .presentation_engines
        .read()
        .get(&presentation_engine_handle)
        .and_then(|pe| pe.camera_entity);
      if let Some(cam_id) = camera_entity {
        let _ = scene_read.scene.with_component_mut(cam_id, |c: &mut CameraComponent| {
          c.update_for_extent(width, height);
        });
      }
    }

    // TODO retry if full up to a threshold
    self
      .threads
      .render_thread
      .tx()
      .try_send(RenderCommand::Resize(Resize {
        presentation_engine_handle,
        width,
        height,
      }))
      .map_err(|_| {
        EngineError::InvalidOperation(
          "[SimulationContext] resize: failed to send resize message to render_thread",
        )
      })
  }

  /// TODO: Document this item
  pub fn download_image(&self, task_id: u64, buffer_ptr: *mut u8, buffer_size: usize) -> bool {
    let result = self
      .render_proxy
      .0
      .as_frontend()
      .ok_or(EngineError::InvalidOperation("render_frontend"))
      .and_then(|frontend| {
        frontend
          .with_device(self.render_proxy.1, |device| {
            device.read_windowless_download(task_id, unsafe {
              core::slice::from_raw_parts_mut(buffer_ptr, buffer_size)
            })
          })
          .map_err(|e| EngineError::from(e))
      });

    match result {
      Ok(_) => true,
      Err(e) => {
        let err_str = alloc::format!("{:?}", e);
        if !err_str.contains("Invalid or previously consumed download ID") {
          oshal::log!("download_image err: {}", err_str);
        }
        false
      }
    }
  }

  /// TODO: Document this item
  pub fn set_asset_path(path: &str) {
    let mut guard = crate::gpu::ASSET_DIR.write();
    *guard = Some(alloc::string::String::from(path));
  }
}
