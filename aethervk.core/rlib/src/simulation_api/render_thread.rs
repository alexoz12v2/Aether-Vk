use itertools::Itertools;
use crate::simulation_api::structs::{
  CustomRenderCallback, RenderCommand, RenderFeedback, RenderTaskStatus, RenderThreadContext,
};
use crate::{
  gpu::scene_conversion::{RenderSceneExtraction, SceneConversionExt},
  gpu,
  gpu::{PresentationEngineHandle, RenderDevice, RenderScene, SwapchainStatus},
  types::{EngineError, EngineResult, GpuError, GpuResult},
};
use aethervk_oshal_rlib as oshal;
use oshal::{
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
use crate::gpu::{FrameCancelGuard, ScopedRenderPass};

pub fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  render_params: RenderThreadContext,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    let mut first_render_map: hashbrown::HashMap<PresentationEngineHandle, bool> =
      hashbrown::HashMap::new();
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
          debug_assert_eq!(alloc::sync::Arc::strong_count(&render_frontend), 1);
          if let RenderCommand::Shutdown = cmd {
            break;
          }
          if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
            process_command(cmd, render_device, &render_params, &mut first_render_map)
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
  first_render_map: &mut hashbrown::HashMap<PresentationEngineHandle, bool>,
) -> GpuResult<()> {
  let _1ms = core::time::Duration::from_millis(1);
  let max_attempts = 10;
  match cmd {
    // this is processed in render_thread function
    RenderCommand::Shutdown => Ok(()),
    RenderCommand::RenderFrame(render_frame) => {
      // The FFI Caller thread, before launching this command, should have already
      // updated the camera's projection matrix.
      let extracted_scene = render_frame.extract_scene(Some(&ctx.thread_pool))?;
      let task_id_feedback = render_frame.task_id;
      let task_id = render_device.create_task();
      let is_first_render =
        if first_render_map.contains_key(&render_frame.presentation_engine_handle) {
          *unsafe {
            first_render_map
              .get(&render_frame.presentation_engine_handle)
              .unwrap_unchecked()
          }
        } else {
          let _ = first_render_map.insert(render_frame.presentation_engine_handle, true);
          true
        };
      // `render_device.success_task` will be called by thread pool when timeline advances
      let time_readings = render_frame
        .scene
        .read()
        .time_state
        .time_info
        .read()
        .current();

      let res = do_render_scene_async(
        render_device,
        extracted_scene,
        render_frame.presentation_engine_handle,
        task_id,
        render_frame.custom_render_callback,
        is_first_render,
        time_readings,
      );

      match res {
        Ok(did_render) => {
          if did_render && is_first_render {
            *unsafe {
              first_render_map
                .get_mut(&render_frame.presentation_engine_handle)
                .unwrap_unchecked()
            } = false;
          }
          task_id_feedback.store(task_id, core::sync::atomic::Ordering::Release);
          Ok(())
        }
        Err(err) => {
          render_device.fail_task(task_id, err.clone());
          task_id_feedback.store(task_id, core::sync::atomic::Ordering::Release);
          Err(err)
        }
      }
    }
    RenderCommand::Resize(resize_cmd) => render_device.resize_presentation_engine(
      resize_cmd.presentation_engine_handle,
      resize_cmd.width,
      resize_cmd.height,
    ),
    // TODO: maybe async? No, add it as a resource upload in scene extraction
    RenderCommand::GenerateSky => render_device.generate_sky(),
    RenderCommand::GetTaskStatus { task_id, output } => {
      let res = render_device.is_task_completed(task_id);
      let status = match res {
        Ok(true) => RenderTaskStatus::Completed,
        Ok(false) => RenderTaskStatus::Pending,
        Err(e) => {
          oshal::log!("is_task_completed err: {:?}", e);
          RenderTaskStatus::Error(e)
        }
      };
      unsafe { output.write_value(status) };

      Ok(())
    }
  }
}

fn do_render_scene_async(
  render_device: &dyn RenderDevice,
  extracted_scene: RenderSceneExtraction,
  presentation_engine_handle: gpu::PresentationEngineHandle,
  task_id: u64,
  custom_render_callback: Option<CustomRenderCallback>,
  is_first_render: bool,
  time_readings: oshal::os::time::TimeReadings,
) -> GpuResult<bool> {
  render_device.start_frame()?;

  let acquire_result = render_device.acquire_next_image(presentation_engine_handle)?;
  if acquire_result.status.needs_resize() {
    // handled via resize command or next frame
    render_device.success_task(task_id);
    return Ok(false);
  }
  let present_guard =
    FrameCancelGuard::new(render_device, presentation_engine_handle, acquire_result);

  let cmd_buffer = render_device.get_command_buffer()?;
  let cmd_scope = gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id))?;
  let render_scene = extracted_scene.build_render_scene(
    render_device,
    presentation_engine_handle,
    cmd_buffer,
    time_readings,
  )?;
  if let Some(sun_call) = &render_scene.sun_call {
    // TODO move to kernels
    render_device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))?;
  }

  if is_first_render && custom_render_callback.is_some() {
    let c = unsafe { custom_render_callback.as_ref().unwrap_unchecked() };
    (c.on_first_render_fn)(
      render_device,
      cmd_buffer,
      presentation_engine_handle,
      &render_scene,
      c.user_data.0,
    )?
  }

  render_device.begin_render_pass(cmd_buffer, presentation_engine_handle, &acquire_result)?;
  let render_pass_scope = gpu::ScopedRenderPass::new(render_device, cmd_buffer);

  let extent = render_device.get_presentation_engine_extent(presentation_engine_handle)?;
  render_device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  render_device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  // TODO: 2) Text not included in measurement now (inside render_frame)
  gpu::frame::render_frame(
    render_device,
    cmd_buffer,
    &render_scene,
    presentation_engine_handle,
  )?;

  if custom_render_callback.is_some() {
    let c = unsafe { custom_render_callback.as_ref().unwrap_unchecked() };
    (c.after_render_frame_fn)(
      render_device,
      cmd_buffer,
      presentation_engine_handle,
      &render_scene,
      c.user_data.0,
    )?;
  }

  // present and submit
  render_pass_scope.end()?;

  // on `DownloadImage` Command, Query task status and copy data if completed with `render_device.read_windowless_download`
  // if you are here, then presentation engine is valid
  let is_windowless = unsafe {
    render_device
      .is_presentation_engine_windowless(presentation_engine_handle)
      .unwrap_unchecked()
  };
  if is_windowless {
    if let Err(e) =
      render_device.record_windowless_download(cmd_buffer, presentation_engine_handle, task_id)
    {
      oshal::log!("record_windowless_download failed: {:?}", e);
      return Err(e);
    }
  }

  if let Err(e) = cmd_scope.submit() {
    oshal::log!("cmd_scope.submit failed: {:?}", e);
    return Err(e);
  }
  present_guard.defuse();

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

  Ok(true)
}

// TODO possibly, group by pipeline if necessary
impl super::structs::RenderFrame {
  pub fn extract_scene(
    &self,
    pool: Option<&oshal::os::pool::ThreadPool>,
  ) -> GpuResult<RenderSceneExtraction> {
    let scene = self.scene.read();
    scene.scene.convert_scene(
      self.camera_entity,
      self.render_physical_meshes_outline,
      pool,
    )
  }
}

pub mod channel_utils {
  use super::*;
  use thingbuf::mpsc::errors::TrySendError;

  /// Repeatedly attempts an action (like sending a message) up to `max_attempts`.
  /// Returns `true` if successful, `false` if all attempts were exhausted or channel closed.
  pub fn retry_with_limit<T: Clone + Default>(
    tx: &mpsc::Sender<T>,
    mut msg: T,
    max_attempts: usize,
    delay: core::time::Duration,
  ) -> bool {
    for _ in 0..max_attempts {
      match tx.try_send(msg) {
        Ok(()) => return true,
        Err(TrySendError::Full(m)) => {
          msg = m;
          os::native::this_thread::sleep_for(delay);
        }
        Err(TrySendError::Closed(_)) => return false,
        Err(_) => return false,
      }
    }
    false
  }

  /// Repeatedly attempts an action infinitely until it succeeds or channel is closed.
  pub fn retry_until_success<T: Default + Clone>(
    tx: &mpsc::Sender<T>,
    mut msg: T,
    delay: core::time::Duration,
  ) {
    loop {
      match tx.try_send(msg) {
        Ok(()) => break,
        Err(TrySendError::Full(m)) => {
          msg = m;
          os::native::this_thread::sleep_for(delay);
        }
        Err(TrySendError::Closed(_)) => break,
        Err(_) => break,
      }
    }
  }
}
