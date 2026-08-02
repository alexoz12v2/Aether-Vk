//! misc_api module.

use crate::{
  expect_scene,
  gpu::{PresentationEngineHandle, WeakRenderFrontendExt},
  scene::CameraComponent,
  simulation_api::{
    BREADCRUMB_CALLBACK, SimulationContext,
    structs::{RenderCommand, Resize, TaskStatusCode},
  },
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::{self as oshal};

impl SimulationContext {
  pub fn set_logger_callback(cb: Option<extern "C" fn(*const core::ffi::c_char)>) {
    let ptr = match cb {
      Some(f) => f as *mut (),
      None => core::ptr::null_mut(),
    };
    oshal::os::debug::LOGGER_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
  }

  pub fn set_breadcrumb_callback(cb: Option<crate::simulation_api::BreadcrumbCallback>) {
    *BREADCRUMB_CALLBACK.write() = cb;
  }

  pub fn set_simulation_callback(cb: Option<crate::simulation_api::SimulationCallback>) {
    *crate::simulation_api::SIMULATION_CALLBACK.write() = cb;
  }

  pub fn set_render_callback(cb: Option<crate::simulation_api::RenderCallback>) {
    *crate::simulation_api::RENDER_CALLBACK.write() = cb;
  }

  // TODO remove when using the composition api
  pub fn get_task_status(&self, task_id: u64) -> TaskStatusCode {
    if task_id == 0 || task_id == u64::MAX {
      return TaskStatusCode::Invalid;
    }

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

  // TODO: Rewrite so that it executes on the render thread under an autorelease pool
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

  /// # Safety
  /// - `buffer_ptr` should represent a valid piece of memory of `buffer_size` bytes
  pub unsafe fn download_image(
    &self,
    task_id: u64,
    buffer_ptr: *mut u8,
    buffer_size: usize,
  ) -> bool {
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
          .map_err(EngineError::from)
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

  pub fn set_asset_path(path: &str) {
    let mut guard = crate::gpu::ASSET_DIR.write();
    *guard = Some(alloc::string::String::from(path));
  }
}
