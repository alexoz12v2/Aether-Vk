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

/// A captured viewport frame ([`SimulationContext::capture_viewport`]): the task id its bytes are
/// read under, its size and its `gpu::CAPTURE_FORMAT_*` code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewportCapture {
  pub task_id: u64,
  pub width: u32,
  pub height: u32,
  pub format: u32,
}

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

  /// Image export: asks the windowed viewport `presentation_engine` to copy its next presented
  /// frame (drawn without the engine labels when `hide_labels`) and waits up to `timeout_ms` for
  /// that frame to be submitted. Returns the frame's task id with the image's width, height and
  /// `gpu::CAPTURE_FORMAT_*` code: poll [`Self::get_task_status`] until it is not `Pending`, then
  /// read `width * height * 4` bytes with [`Self::download_image`]. `None` when the engine is
  /// windowless, has no size, or rendered no frame within the timeout (minimised): the request
  /// is then withdrawn, so no orphan copy is left behind.
  ///
  /// Not on the render thread (it waits for it).
  pub fn capture_viewport(
    &self,
    presentation_engine: PresentationEngineHandle,
    hide_labels: bool,
    timeout_ms: u64,
  ) -> Option<ViewportCapture> {
    let request = match self.with_device(|device| {
      device.request_viewport_capture(presentation_engine, hide_labels)
    }) {
      Ok(r) => r,
      Err(e) => {
        oshal::log!("capture_viewport: {:?}", e);
        return None;
      }
    };
    let poll = core::time::Duration::from_millis(2);
    let mut waited_ms = 0u64;
    loop {
      let task_id = request.task_id.load(core::sync::atomic::Ordering::Acquire);
      if task_id == u64::MAX {
        oshal::log!("capture_viewport: the frame could not be captured");
        return None;
      }
      if task_id != 0 {
        return Some(ViewportCapture {
          task_id,
          width: request.width.load(core::sync::atomic::Ordering::Acquire),
          height: request.height.load(core::sync::atomic::Ordering::Acquire),
          format: request.format.load(core::sync::atomic::Ordering::Acquire),
        });
      }
      if waited_ms >= timeout_ms {
        // withdraw it (unless a frame took it meanwhile: then it completes and is read later)
        let _ = self.with_device(|device| {
          device.cancel_viewport_capture(presentation_engine);
          Ok(())
        });
        let task_id = request.task_id.load(core::sync::atomic::Ordering::Acquire);
        if task_id != 0 && task_id != u64::MAX {
          return Some(ViewportCapture {
            task_id,
            width: request.width.load(core::sync::atomic::Ordering::Acquire),
            height: request.height.load(core::sync::atomic::Ordering::Acquire),
            format: request.format.load(core::sync::atomic::Ordering::Acquire),
          });
        }
        oshal::log!(
          "capture_viewport: no frame of PE {:?} within {} ms (viewport hidden or minimised?)",
          presentation_engine,
          timeout_ms
        );
        return None;
      }
      oshal::os::native::this_thread::sleep_for(poll);
      waited_ms += 2;
    }
  }

  /// [TEST ONLY] Downloads the finalGlobalDepth MRT attachment for the given render task.
  /// `buffer_ptr` must point to `width * height * 8` bytes (two f32 per pixel).
  ///
  /// # Safety
  /// - `buffer_ptr` must be valid for `buffer_size` bytes.
  #[cfg(test)]
  pub unsafe fn download_global_depth_image(
    &self,
    task_id: u64,
    buffer_ptr: *mut u8,
    buffer_size: usize,
  ) -> bool {
    let gdepth_tid = task_id | crate::gpu::GLOBAL_DEPTH_TASK_BIT;
    let result = self
      .render_proxy
      .0
      .as_frontend()
      .ok_or(crate::simulation_api::EngineError::InvalidOperation(
        "render_frontend",
      ))
      .and_then(|frontend| {
        frontend
          .with_device(self.render_proxy.1, |device| {
            device.read_global_depth_download(gdepth_tid, unsafe {
              core::slice::from_raw_parts_mut(buffer_ptr, buffer_size)
            })
          })
          .map_err(crate::simulation_api::EngineError::from)
      });
    result.is_ok()
  }

  pub fn set_asset_path(path: &str) {
    let mut guard = crate::gpu::ASSET_DIR.write();
    *guard = Some(alloc::string::String::from(path));
  }
}
