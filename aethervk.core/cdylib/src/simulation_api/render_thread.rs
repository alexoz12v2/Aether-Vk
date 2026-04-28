use crate::structs::{RenderCommand, RenderFeedback, RenderTaskStatus, RenderThreadContext};
use aethervk_core_rlib::{
  gpu::scene_conversion::{RenderSceneExtraction, SceneConversionExt},
  gpu,
  gpu::{PresentationEngineHandle, RenderDevice, RenderScene, SwapchainStatus},
  types::{EngineError, EngineResult, GpuError, GpuResult},
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::matrix::mat4::Mat4x4f32,
  math::matrix::{MatrixVectorMul, SquareMatrix},
  math::quaternion::Quaternion,
  math::vector::vec3::Vec3f32,
  math::vector::{Vector4},
  os,
  os::thread::Thread,
  os::{thread, NativeError, ThreadingError},
};
use thingbuf::mpsc;

/// Retries a channel send operation. Evaluates to `Ok(())` on success,
/// or `Err($err)` if max attempts are reached.
macro_rules! try_send_with_limit {
  ($action:expr, $attempts:expr, $delay:expr$(,)?) => {{
    let success = channel_utils::retry_with_limit($action, $attempts, $delay);
    if success {
      Ok(())
    } else {
      Err(GpuError::InvalidState(
        "[Render Thread] process_command | failed to send feedback",
      ))
    }
  }};
}

pub fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  render_params: RenderThreadContext,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    assert!(render_params.is_render_single_ownership());
    let render_device_handle = render_params.render_device_handle;
    let render_frontend = {
      let r = render_params
        .render_frontend
        .try_borrow_mut()
        .map_err(|_| EngineError::InvalidOperation("Failed to borrow render_frontend"));
      if let Err(e) = r {
        oshal::log!("render_thread | render_frontend borrow: {:?}", e);
        return;
      }
      let frontend = unsafe { r.unwrap_unchecked() }
        .take()
        .ok_or(EngineError::InvalidOperation(
          "render_frontend was already None",
        ));
      if let Ok(render_frontend) = frontend {
        render_frontend
      } else {
        oshal::log!("render_thread | render_frontend acquire: {:?}", unsafe {
          frontend.unwrap_err_unchecked()
        });
        return;
      }
    };
    loop {
      match render_rx.try_recv() {
        Ok(cmd) => {
          if let RenderCommand::Shutdown = cmd {
            break;
          }
          if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
            process_command(cmd, render_device, &render_params)
          }) {
            oshal::log!("render_thread | process_command failed: {:?}", e);
          }
        }
        Err(e) => {
          if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
            break;
          }
          // Avoid pegging CPU if no commands
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
        }
      }
    }
  })
  .map_err(|err| <ThreadingError as Into<NativeError>>::into(err))
  .map_err(|err| <NativeError as Into<EngineError>>::into(err))
}

fn process_command(
  cmd: RenderCommand,
  render_device: &dyn RenderDevice,
  ctx: &RenderThreadContext,
) -> GpuResult<()> {
  let _1ms = core::time::Duration::from_millis(1);
  let max_attempts = 10;
  match cmd {
    // this is processed in render_thread function
    RenderCommand::Shutdown => Ok(()),
    RenderCommand::RenderFrame(render_frame) => {
      // The FFI Caller thread, before launching this command, should have already
      // updated the camera's projection matrix.
      let render_scene = render_frame.prepare_scene(render_device)?;
      let task_id = render_device.create_task();
      // `render_device.success_task` will be called by thread pool when timeline advances
      match do_render_scene_async(
        render_device,
        render_scene,
        render_frame.presentation_engine_handle,
        task_id,
      ) {
        Ok(()) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskCreated(Some(task_id))),
            max_attempts,
            _1ms,
          )
        }
        Err(err) => {
          render_device.fail_task(task_id, err.clone());
          channel_utils::retry_until_success(
            || {
              ctx
                .render_feedback_tx
                .try_send(RenderFeedback::TaskCreated(None))
            },
            _1ms,
          );
          Err(err)
        }
      }
    }
    RenderCommand::DownloadImage(download_image) => {
      // 1. Check completion. If true, try reading the download.
      // Both steps must succeed to return Ok(true). Any failure returns Err(e).
      let task_status = render_device
        .is_task_completed(download_image.task_id)
        .and_then(|is_completed| {
          if is_completed {
            render_device
              .read_windowless_download(download_image.task_id, unsafe {
                core::slice::from_raw_parts_mut(download_image.buffer.0, download_image.buffer_size)
              })
              .map(|_| true) // Map Ok(()) to Ok(true) to feed into the match
          } else {
            Ok(false)
          }
        });

      // 2. Handle the combined result
      match task_status {
        Ok(true) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Completed)),
            max_attempts,
            _1ms
          )
        }
        Ok(false) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Pending)),
            max_attempts,
            _1ms
          )
        }
        Err(err) => {
          // Catches errors from both `is_task_completed` and `read_windowless_download`
          channel_utils::retry_until_success(
            || {
              ctx
                .render_feedback_tx
                .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Error(
                  err.clone(),
                )))
            },
            _1ms,
          );
          Err(err)
        }
      }
    }
    RenderCommand::Resize(_) => {
      todo!();
    }
    // TODO move to logic thread which will dispatch this to an affinity thread for compute
    RenderCommand::GenerateSky => todo!(),
  }
}

fn do_render_scene_async(
  render_device: &dyn RenderDevice,
  render_scene: RenderScene,
  presentation_engine_handle: PresentationEngineHandle,
  task_id: u64,
) -> GpuResult<()> {
  render_device.start_frame()?;

  let acquire_result = render_device.acquire_next_image(presentation_engine_handle)?;
  if acquire_result.status.needs_resize() {
    // handled via resize command or next frame
    render_device.success_task(task_id);
    return Ok(());
  }

  let cmd_buffer = render_device.get_command_buffer()?;
  let cmd_scope = gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id))?;
  if let Some(sun_call) = &render_scene.sun_call {
    // TODO move to kernels
    render_device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))?;
  }

  render_device.begin_render_pass(cmd_buffer, presentation_engine_handle, &acquire_result)?;
  let render_pass_scope = gpu::ScopedRenderPass::new(render_device, cmd_buffer);

  let extent = render_device.get_presentation_engine_extent(presentation_engine_handle)?;
  render_device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  render_device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  // TODO: 2) Text not included in measurement now (inside render_frame)
  render_device.render_frame(cmd_buffer, &render_scene)?;

  // on `DownloadImage` Command, Query task status and copy data if completed with `render_device.read_windowless_download`
  render_device.record_windowless_download(cmd_buffer, presentation_engine_handle, task_id)?;

  // present and submit
  render_pass_scope.end()?;
  cmd_scope.submit()?;

  if SwapchainStatus::Optimal
    != render_device.present(
      presentation_engine_handle,
      acquire_result.image_index as usize,
      acquire_result.frame_index as usize,
    )?
  {
    oshal::log!(
      "[Render Thread] Warning: render_device.present isn't optimal. Might not be an error"
    );
  }

  Ok(())
}

// TODO possibly, group by pipeline if necessary
impl super::structs::RenderFrame {
  pub fn prepare_scene(&self, device: &dyn RenderDevice) -> GpuResult<gpu::RenderScene> {
    let render_extraction: RenderSceneExtraction = {
      let scene = self.scene.read();
      scene
        .scene
        .convert_scene(self.camera_entity, self.render_physical_meshes_outline)
    }?; // <-- THE ECS RWLOCK IS SAFELY DROPPED HERE!

    // --- PASS 2: VULKAN TRANSLATION ---
    render_extraction.build_render_scene(device, self.presentation_engine_handle)
  }
}

pub mod channel_utils {
  use super::*;

  /// Repeatedly attempts an action (like sending a message) up to `max_attempts`.
  /// Returns `true` if successful, `false` if all attempts were exhausted.
  pub fn retry_with_limit<F, E>(
    mut action: F,
    max_attempts: usize,
    delay: core::time::Duration,
  ) -> bool
  where
    F: FnMut() -> Result<(), E>,
  {
    for _ in 0..max_attempts {
      if action().is_ok() {
        return true;
      }
      os::native::this_thread::sleep_for(delay);
    }
    false
  }

  /// Repeatedly attempts an action infinitely until it succeeds.
  pub fn retry_until_success<F, E>(mut action: F, delay: core::time::Duration)
  where
    F: FnMut() -> Result<(), E>,
  {
    loop {
      if action().is_ok() {
        break;
      }
      os::native::this_thread::sleep_for(delay);
    }
  }
}


