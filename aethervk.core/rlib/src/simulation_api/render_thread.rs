//! render_thread module.

use crate::{
  gpu::{self, PresentationEngineHandle, RenderDevice, scene_conversion::SceneConversionExt2},
  gpu_backends::vulkan::{self, utils::RwLockable},
  simulation_api::{
    invoke_main_thread_flush_cleanup, invoke_main_thread_process_cleanup,
    structs::{RenderCommand, RenderThreadContext},
  },
  types::{EngineError, EngineResult, GpuError, GpuResult},
};
use aethervk_oshal_rlib::os::{pool::tasklet::ThreadPoolExt, time::get_monotonic_time};
use aethervk_oshal_rlib::{self as oshal, os::time::timeus_t};
use ash::vk::Handle;
use oshal::{
  os,
  os::{NativeError, ThreadingError, thread, thread::Thread},
};
use thingbuf::mpsc;

/// Render Thread Entry Point
pub fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  render_params: RenderThreadContext,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    #[cfg(debug_assertions)]
    {
      oshal::os::debug::fpe::unmask_fpu_for_current_thread();
    }
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
      let frontend = unsafe { r.unwrap_unchecked() }.take().ok_or(EngineError::InvalidOperation(
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

    // periodic render discard and process main thread cleanup queue (shifted among each other)
    let mut last_discard_unscaled_us: timeus_t = 0;
    let mut last_main_thread_cleanup_unscaled_us: timeus_t = 5000;
    const CLEANUP_DELTA_UNSCALED_US: timeus_t = oshal::os::time::timeus_milliseconds(500);

    // atomic boolean used as a signaling mechanism to ensure that main thread callbacks are
    let main_thread_cb_signal_done = core::sync::atomic::AtomicBool::new(true);

    loop {
      let mut core_logic = || -> bool {
        let now = get_monotonic_time();
        let do_discard = now - last_discard_unscaled_us > CLEANUP_DELTA_UNSCALED_US;
        let do_main_queue_cleanup =
          now - last_main_thread_cleanup_unscaled_us > CLEANUP_DELTA_UNSCALED_US;
        if do_discard {
          last_discard_unscaled_us = now;
        }
        if do_main_queue_cleanup {
          last_main_thread_cleanup_unscaled_us = now;
        }
        if do_discard || do_main_queue_cleanup {
          let _ = render_frontend.with_device(render_device_handle, |dyn_device| {
            let vulkan_device: &vulkan::device::Device =
              dyn_device.as_any().downcast_ref().unwrap();
            if do_discard {
              let items = {
                let res = vulkan_device.res.read();
                let cached_timeline = res.get_timeline_semaphore_cached_value();
                if cached_timeline == 0 {
                  return Ok(()); // don't care about do_main_queue_cleanup, we just started
                }
                // not sure about this `- 1`, but it's conservative, so it's fine
                let timeline = cached_timeline - 1;
                res.discard_pool.pop_ready_items(timeline)
              };
              vulkan::device::DiscardPool::destroy_items_lock_free(&vulkan_device.device, items);
            }
            if do_main_queue_cleanup {
              use core::sync::atomic::Ordering;
              while !main_thread_cb_signal_done.load(Ordering::Acquire) {
                core::hint::spin_loop();
              }
              main_thread_cb_signal_done.store(false, Ordering::Release);
              unsafe {
                invoke_main_thread_process_cleanup(vulkan_device, &main_thread_cb_signal_done)
              };
            }
            Ok(())
          });
        }

        match render_rx.try_recv() {
          Ok(cmd) => {
            if let RenderCommand::Shutdown = cmd {
              render_frontend
                .with_device(render_device_handle, |dyn_device| {
                  use core::sync::atomic::Ordering;
                  let vulkan_device: &vulkan::device::Device =
                    dyn_device.as_any().downcast_ref().unwrap();
                  while !main_thread_cb_signal_done.load(Ordering::Acquire) {
                    core::hint::spin_loop();
                  }
                  main_thread_cb_signal_done.store(false, Ordering::Release);
                  unsafe {
                    invoke_main_thread_flush_cleanup(vulkan_device, &main_thread_cb_signal_done);
                  }

                  // since this is shutdown, wait for it to be done
                  while !main_thread_cb_signal_done.load(Ordering::Acquire) {
                    core::hint::spin_loop();
                  }
                  main_thread_cb_signal_done.store(false, Ordering::Release);

                  Ok(())
                })
                .unwrap();
              return true;
            }

            debug_assert!(alloc::sync::Arc::strong_count(&render_frontend) < 16);
            if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
              process_command(
                cmd,
                render_device,
                &render_params,
                &mut first_render_map,
                render_frontend.clone(),
                render_device_handle,
              )
            }) {
              oshal::log!("render_thread | process_command failed: {:?}", e);
            }
            false
          }
          Err(e) => {
            if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
              return true;
            }
            oshal::os::native::this_thread::yield_now();
            false
          }
        }
      };

      #[cfg(target_vendor = "apple")]
      let should_break = objc2::rc::autoreleasepool(|_| core_logic());

      #[cfg(not(target_vendor = "apple"))]
      let should_break = core_logic();

      if should_break {
        break;
      }
    }
  })
  .map_err(<ThreadingError as Into<NativeError>>::into)
  .map_err(<NativeError as Into<EngineError>>::into)
}

fn process_command(
  cmd: RenderCommand,
  render_device: &dyn RenderDevice,
  ctx: &RenderThreadContext,
  first_render_map: &mut hashbrown::HashMap<PresentationEngineHandle, bool>,
  render_frontend: gpu::RenderFrontend,
  render_device_handle: gpu::RenderDeviceHandle,
) -> GpuResult<()> {
  let _1ms = core::time::Duration::from_millis(1);
  match cmd {
    // this is processed in render_thread function
    RenderCommand::Shutdown => Ok(()),
    RenderCommand::SyncParticleRelease {
      feedback,
      feedback_ptr,
    } => {
      use crate::gpu_backends::vulkan::utils::RwLockable;
      let task_id = render_device.create_task();
      let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
        render_device.as_any().downcast_ref().unwrap();
      let store_failure = |e: &GpuError| {
        render_device.fail_task(task_id, e.clone());
        feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
      };

      let (cmd_buffer, cmd) = vulkan_device
        .get_command_buffer_and_native()
        .inspect_err(|e| store_failure(e))?;
      let cmd_scope = gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id))
        .inspect_err(|e| store_failure(e))?;

      let res = vulkan_device.res.read();
      let psm = res.particle_system_manager.as_ref().unwrap();
      psm.cmd_sync_graphics_release_front(
        &vulkan_device.device,
        cmd,
        vulkan_device.get_graphics_queue().family_index,
        vulkan_device.get_compute_queue().family_index,
      );
      drop(res);

      // if we are not beyond deadline, then we can submit and get value
      loop {
        use core::sync::atomic::Ordering;
        match feedback.compare_exchange_weak(0, 1, Ordering::AcqRel, Ordering::Acquire) {
          Ok(_) => {
            // ready
            // sumbitting signals the timeline semaphore upon completion. we can query the task_id
            cmd_scope.submit().inspect_err(|e| store_failure(e))?;
            let timeline_value =
              vulkan_device.get_task_target_value(task_id).inspect_err(|e| store_failure(e))?;
            // SAFETY: this was populated from a Boxed type. Shouldn't be null unless we are out of
            // memory, in which case we crash anyways
            let feedback_mut = unsafe { feedback_ptr.get().as_mut().unwrap() };
            feedback_mut.timeline_semaphore =
              vulkan_device.res.read().timeline_manager.semaphore.get();
            feedback_mut.timeline_release_value = timeline_value;
            drop(feedback_mut);

            feedback.store(task_id, core::sync::atomic::Ordering::Release);
            break Ok(());
          }
          Err(old) => {
            use alloc::string::ToString;
            if old == u64::MAX {
              // deadline expired, rollback command buffer and store failure, free pointer
              core::mem::forget(cmd_scope);
              store_failure(&GpuError::InvalidState("Deadline".to_string()));

              // SAFETY: if deadline, logic_thread has renounced ownership of this pointer
              let _ = unsafe { alloc::boxed::Box::from_raw(feedback_ptr.get()) };

              break Ok(());
            }
          }
        }
      }
    }
    RenderCommand::RenderFrames(render_frames) => {
      render_device.start_frame()?;

      let mut handles = alloc::vec::Vec::with_capacity(8);

      for render_frame in render_frames {
        let task_id_feedback = alloc::sync::Arc::clone(&render_frame.task_id);
        let extent_res =
          render_device.get_presentation_engine_extent(render_frame.presentation_engine_handle);
        let extent = match extent_res {
          Ok(e) => e,
          Err(_) => {
            task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
            continue;
          }
        };
        if extent[0] == 0 || extent[1] == 0 {
          task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
          continue;
        }

        let acquire_result =
          render_device.acquire_next_image(render_frame.presentation_engine_handle);
        let acquire_result = match acquire_result {
          Ok(res) => res,
          Err(_) => {
            task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
            continue;
          }
        };

        if acquire_result.status.needs_resize() {
          task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
          continue;
        }

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

        let frontend = render_frontend.clone();
        let handle = render_device_handle;
        let thread_pool = alloc::sync::Arc::clone(&ctx.thread_pool);
        let pe_handle = render_frame.presentation_engine_handle;
        let task_id_feedback_err = alloc::sync::Arc::clone(&render_frame.task_id);
        let task_id_feedback_err_clone = alloc::sync::Arc::clone(&task_id_feedback_err);

        let res: GpuResult<(gpu::CommandBufferHandle, bool, bool)> = {
          let core_logic = || -> GpuResult<(gpu::CommandBufferHandle, bool, bool)> {
            let result =
              frontend.with_device(handle, |render_device| {
                let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
                  render_device.as_any().downcast_ref().unwrap();
                let task_id = render_device.create_task();

                let present_guard = gpu::FrameCancelGuard::new(
                  render_device,
                  render_frame.presentation_engine_handle,
                  acquire_result,
                );

                let (cmd_buffer, cmd) =
                  vulkan_device.get_command_buffer_and_native().map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] get_command_buffer failed: {:?}",
                      e
                    );
                    e
                  })?;

                render_device
                  .set_command_buffer_presentation_engine(
                    cmd_buffer,
                    render_frame.presentation_engine_handle,
                  )
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!("[render tasklet] set_cmd_pe failed: {:?}", e);
                    e
                  })?;
                let mut cmd_scope =
                  gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id)).map_err(
                    |e| {
                      aethervk_oshal_rlib::log!(
                        "[render tasklet] ScopedCommandBuffer::new failed: {:?}",
                        e
                      );
                      e
                    },
                  )?;

                // Before extracting or rendering scenes, first record necessary commands to ensure
                // that Cross Sync step 4 (graphics queue new front buffers acquisition) end
                // perfectly
                if let Some(wait_timeline_val) = render_frame.particle_acquire_sync {
                  use crate::gpu_backends::vulkan::utils::RwLockable;
                  let res = vulkan_device.res.read();
                  if let Some(psm) = res.particle_system_manager.as_ref() {
                    let gfx_fam = vulkan_device.get_graphics_queue().family_index;
                    let comp_fam = vulkan_device.get_compute_queue().family_index;
                    psm.cmd_sync_graphics_acquire_new_front(
                      &vulkan_device.device,
                      cmd,
                      gfx_fam,
                      comp_fam,
                    );
                  }

                  // add wait timeline val and compute timeline semaphore to wait
                  let compute_timeline_sem = vulkan_device.kernels.timeline;
                  cmd_scope.add_sync_info(gpu::CommandBufferSyncInfo {
                    timeline_semaphore: compute_timeline_sem.as_raw(),
                    timeline_value: wait_timeline_val,
                    wait_stage_mask: gpu::CommandBufferSyncInfoStageMask::VertexAttributeInput,
                  });
                }

                // read lock for scene context
                let scene_context_read = render_frame.scene.read();

                let (
                  unscaled_time_us,
                  unscaled_time_delta_us,
                  scaled_time_us,
                  scaled_time_delta_us,
                ) = {
                  let time_state_read = scene_context_read.time_state.read();
                  (
                    time_state_read.unscaled_time,
                    time_state_read.unscaled_delta,
                    time_state_read.scaled_time,
                    time_state_read.scaled_delta,
                  )
                };
                let debug_name = scene_context_read.debug_name.clone();

                let render_scene = scene_context_read
                  .scene
                  .build_render_scene(
                    &vulkan_device,
                    pe_handle,
                    cmd_buffer,
                    render_frame.camera_entity,
                    render_frame.render_physical_meshes_outline,
                    Some(&ctx.thread_pool),
                    extent,
                    unscaled_time_us,
                    unscaled_time_delta_us,
                    scaled_time_us,
                    scaled_time_delta_us,
                    render_frame.mean_intra_grains_distance_mm,
                    render_frame.min_cumulated_mass_g,
                    &debug_name,
                  )
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] build_render_scene failed: {:?}",
                      e
                    );
                    e
                  })?;

                if let Some(layer) = render_scene.depth_layers.first()
                  && let Some(sun_call) = &layer.sun_call
                {
                  render_device
                    .update_sun(
                      cmd_buffer,
                      sun_call.entity,
                      (128, 128, 128),
                      sun_call.radius,
                    )
                    .map_err(|e| {
                      aethervk_oshal_rlib::log!("[render tasklet] update_sun failed: {:?}", e);
                      e
                    })?;
                }

                if is_first_render && render_frame.custom_render_callback.is_some() {
                  let c =
                    unsafe { render_frame.custom_render_callback.as_ref().unwrap_unchecked() };
                  (c.on_first_render_fn)(
                    render_device,
                    cmd_buffer,
                    render_frame.presentation_engine_handle,
                    &render_scene,
                    c.user_data.0,
                  )
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] on_first_render_fn failed: {:?}",
                      e
                    );
                    e
                  })?
                }

                // Always select compositing render pass
                render_device
                  .begin_compositing_render_pass(
                    cmd_buffer,
                    render_frame.presentation_engine_handle,
                    &acquire_result,
                  )
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] begin_compositing_render_pass failed: {:?}",
                      e
                    );
                    e
                  })?;
                let render_pass_scope = gpu::ScopedRenderPass::new(render_device, cmd_buffer);

                render_device
                  .set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!("[render tasklet] set_viewport failed: {:?}", e);
                    e
                  })?;
                render_device
                  .set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!("[render tasklet] set_scissor failed: {:?}", e);
                    e
                  })?;

                gpu::frame::render_frame(
                  render_device,
                  cmd_buffer,
                  render_frame.presentation_engine_handle,
                  &render_scene,
                )
                .map_err(|e| {
                  aethervk_oshal_rlib::log!("[render tasklet] render_frame failed: {:?}", e);
                  e
                })?;

                if render_frame.custom_render_callback.is_some() {
                  let c =
                    unsafe { render_frame.custom_render_callback.as_ref().unwrap_unchecked() };
                  (c.after_render_frame_fn)(
                    render_device,
                    cmd_buffer,
                    render_frame.presentation_engine_handle,
                    &render_scene,
                    c.user_data.0,
                  )
                  .map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] after_render_frame_fn failed: {:?}",
                      e
                    );
                    e
                  })?;
                }

                render_pass_scope.end().map_err(|e| {
                  aethervk_oshal_rlib::log!(
                    "[render tasklet] render_pass_scope.end failed: {:?}",
                    e
                  );
                  e
                })?;

                let is_windowless = unsafe {
                  render_device
                    .is_presentation_engine_windowless(render_frame.presentation_engine_handle)
                    .unwrap_unchecked()
                };
                if is_windowless {
                  render_device.record_windowless_download(cmd_buffer, task_id).map_err(|e| {
                    aethervk_oshal_rlib::log!(
                      "[render tasklet] record_windowless_download failed: {:?}",
                      e
                    );
                    e
                  })?;
                }

                cmd_scope.submit().map_err(|e| {
                  aethervk_oshal_rlib::log!("[render tasklet] cmd_scope.submit failed: {:?}", e);
                  e
                })?;
                present_guard.defuse();

                let task_id_feedback = alloc::sync::Arc::clone(&render_frame.task_id);
                task_id_feedback.store(task_id, core::sync::atomic::Ordering::Release);

                Ok((cmd_buffer, is_windowless, is_first_render))
              });

            if let Err(ref e) = result {
              aethervk_oshal_rlib::log!(
                "[render tasklet] tasklet failed, signalling u64::MAX: {:?}",
                e
              );
              task_id_feedback_err_clone.store(u64::MAX, core::sync::atomic::Ordering::Release);
            }
            result
          };

          #[cfg(target_os = "macos")]
          {
            objc2::rc::autoreleasepool(|_| core_logic())
          }

          #[cfg(not(target_os = "macos"))]
          {
            core_logic()
          }
        };

        if let Ok((cmd_buffer, is_windowless, is_first_render)) = res {
          handles.push((
            cmd_buffer,
            is_windowless,
            is_first_render,
            pe_handle,
            acquire_result,
          ));
        } else {
          aethervk_oshal_rlib::log!(
            "[render thread] Queue submission failed for PE '{:?}' with error {}",
            pe_handle,
            unsafe { res.unwrap_err_unchecked() }
          );
          task_id_feedback_err.store(u64::MAX, core::sync::atomic::Ordering::Release);
        }
      }

      // Step 2: Synchronize and submit sequentially
      for (_cmd_buffer, _is_windowless, is_first_render, pe_handle, acquire_result) in handles {
        if is_first_render {
          *unsafe { first_render_map.get_mut(&pe_handle).unwrap_unchecked() } = false;
        }
        match render_device.present(
          pe_handle,
          acquire_result.image_index as usize,
          acquire_result.frame_index as usize,
        ) {
          Ok(crate::gpu::SwapchainStatus::Optimal) => {}
          Ok(status) => {
            oshal::log!(
              "[Render Thread] Warning: present status={:?} for PE {:?} — may need resize",
              status,
              pe_handle
            );
          }
          Err(e) => {
            oshal::log!(
              "[Render Thread] present() error for PE {:?}: {:?}",
              pe_handle,
              e
            );
            return Err(e);
          }
        }
      }
      Ok(())
    }
    RenderCommand::Resize(resize_cmd) => render_device.resize_presentation_engine(
      resize_cmd.presentation_engine_handle,
      resize_cmd.width,
      resize_cmd.height,
    ),
    RenderCommand::GenerateSky => render_device.generate_sky(),
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