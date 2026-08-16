//! logic_thread module.

use crate::{
  gpu::{RenderDevice, WeakRenderFrontendExt},
  gpu_backends::vulkan,
  scene::{
    AlmanacPlanet, BodyRotationalModel, CameraComponent, CometMarkerComponent, CursorComponent,
    EntityId, ErasedForeignSerializable, HighResTransformComponent, PlanetMarkerComponent,
    ReferenceFrameComponent, TransformAnimationComponent, TransformComponent,
    camera::QuatToEulerAngles, particles::v2::ParticleSystemComponent,
  },
  simulation::almanac::AlmanacPackedData,
  simulation_api::{
    ComponentForeignId, emit_breadcrumb, emit_external_state_change,
    external_state::{CModelImported, CTimeRange, ExternalState},
    structs::{
      self, CartesianState, LogicCommand, LogicThreadContext, LogicWorkload, PhysicsDeviceSelfSync,
      SceneContext, SyncParticleReleaseFeedback,
    },
    time_api,
  },
  types::{EngineError, EngineResult, GpuResult},
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::{
    quaternion::Quaternion,
    vector::{
      Vector, Vector3, Vector4,
      vec3::Vec3f32,
      vec3f64::{DVec3, Vec3f64},
      vec4::{Quat, Vec4f32},
    },
  },
  os::{
    NativeError, ThreadingError,
    pool::{WorkloadStatus, tasklet::ThreadPoolExt},
    thread::{self, Thread},
    time::{get_monotonic_time, timeus_t},
  },
};
use alloc::{boxed::Box, string::ToString};
use thingbuf::mpsc;
// don't insert imports from `dashmap` or `parking_lot` or `spin` cause their lock types have the
// same names

pub fn is_logic_command_async(cmd: &LogicCommand) -> bool {
  match cmd {
    LogicCommand::ImportModel { .. }
    | LogicCommand::LoadAlmanac { .. }
    | LogicCommand::UnloadAlmanac { .. }
    | LogicCommand::UpdateTrajectoryForSpk { .. } => true,
    _ => false,
  }
}

fn logic_command_desc(cmd: &LogicCommand) -> alloc::string::String {
  match cmd {
    LogicCommand::SetEpochRange { .. } => "SetEpochRange".to_string(),
    LogicCommand::Shutdown => "Shutdown".to_string(),

    // Camera Commands
    LogicCommand::RotateCamera { .. } => "Rotate Camera".to_string(),
    LogicCommand::ZoomCamera { .. } => "Zoom Camera".to_string(),
    LogicCommand::ResetCamera { .. } => "Reset Camera".to_string(),
    LogicCommand::PanCamera { .. } => "Pan Camera".to_string(),

    // Cursor Commands
    LogicCommand::MoveCursor { .. } => "Move Cursor".to_string(),

    // Entity Commands
    LogicCommand::SnapToEntity { .. } => "Snap to Entity".to_string(),
    LogicCommand::SetEntityVisibility {
      entity, visible, ..
    } => {
      alloc::format!("Set visibility for entity {} to {}", entity, visible)
    }

    // Scene Playback Commands
    LogicCommand::StartSimlulation { .. } => "Start Simulation".to_string(),
    LogicCommand::PlaySceneToEnd { .. } => "Play Scene to End".to_string(),
    LogicCommand::PauseScene { .. } => "Pause Scene".to_string(),
    LogicCommand::PlayScene { .. } => "Play Scene".to_string(),
    LogicCommand::SnapshotScene { .. } => "Snapshot Scene".to_string(),
    LogicCommand::RestoreSnapshot { .. } => "Restore Snapshot".to_string(),

    // Data/Asset Commands
    LogicCommand::ImportModel { path, .. } => alloc::format!("Import model {}", path),
    LogicCommand::LoadAlmanac { path, .. } => alloc::format!("Load almanac {}", path),
    LogicCommand::UnloadAlmanac { path, .. } => alloc::format!("Unload almanac {}", path),

    // Trajectory
    LogicCommand::UpdateTrajectoryForSpk { spk_id, .. } => {
      alloc::format!("Update trajectory for SPK {}", spk_id)
    }

    // Comet lifecycle
    LogicCommand::InitComet { spk_id, .. } => {
      alloc::format!("Init comet SPK {}", spk_id)
    }
    LogicCommand::CleanupComet { .. } => "Cleanup comet".to_string(),

    // Animation commands
    LogicCommand::AnimateCameraTo { camera_id, .. } => {
      alloc::format!("Animate camera {} to target", camera_id)
    }
  }
}

/// Drains all immediately-available [`LogicCommand`]s from `rx` and executes
/// the fast-path synchronous ones on the calling thread.
///
/// Returns `true` if a `Shutdown` command was received (caller must exit).
/// Heavy I/O tasks (model import, almanac load, etc.) are scattered to the
/// thread pool as usual.
///
/// Called both from the outer tick loop **and** from inside
/// `execute_simulation_tick` after `dispatch_physics_step` returns, so that
/// keybinding / time-scale commands are processed within one physics-step
/// wall-time rather than being delayed until the next outer-loop iteration.
fn drain_logic_commands(
  rx: &mpsc::Receiver<LogicCommand>,
  ctx: &alloc::sync::Arc<LogicThreadContext>,
) -> bool {
  loop {
    match rx.try_recv() {
      Ok(cmd) => {
        if let LogicCommand::Shutdown = cmd {
          return true;
        }
        if is_logic_command_async(&cmd) {
          let workload = Box::new(LogicWorkload {
            cmd,
            ctx: ctx.clone(),
          });
          let _ = ctx.thread_pool.scatter(alloc::vec![workload]);
        } else {
          let cmd_desc = logic_command_desc(&cmd);
          if let Err(e) = process_command_internal(cmd, ctx) {
            crate::simulation_api::emit_breadcrumb(
              3,
              &alloc::format!("Failed: {} - {}", cmd_desc, e),
            );
          }
        }
      }
      Err(thingbuf::mpsc::errors::TryRecvError::Closed) => return true,
      Err(_) => break,
    }
  }
  false
}

struct PlayControl {
  target_frame_time: timeus_t,
  last_frame_start: timeus_t,
  last_render_ticks: alloc::vec::Vec<core::num::NonZero<u64>>,
}

impl PlayControl {
  fn new(target_frame_time: timeus_t) -> Self {
    Self {
      target_frame_time,
      last_frame_start: oshal::os::time::get_monotonic_time(),
      last_render_ticks: alloc::vec::Vec::new(),
    }
  }
}

pub fn start_logic_thread(
  logic_rx: mpsc::Receiver<LogicCommand>,
  context: alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    #[cfg(debug_assertions)]
    {
      oshal::os::debug::fpe::unmask_fpu_for_current_thread();
    }
    let target_frame_time = oshal::os::time::timeus_milliseconds(16); // ~60 FPS
    let mut play_controls: hashbrown::HashMap<u64, PlayControl> = hashbrown::HashMap::new();

    // periodic compute discard
    let mut last_discard_unscaled_us: timeus_t = 0;
    const DISCARD_DELTA_UNSCALED_US: timeus_t = oshal::os::time::timeus_milliseconds(500);

    loop {
      let mut core_logic = || -> bool {
        let mut processed_any = false;

        // perform compute discard every 500ms
        let now = oshal::os::time::get_monotonic_time();
        if now - last_discard_unscaled_us > DISCARD_DELTA_UNSCALED_US {
          last_discard_unscaled_us = now;
          let _ = context.kernels.0.with_device(context.kernels.1, |dyn_device| {
            let vulkan_device: &vulkan::device::Device =
              dyn_device.as_any().downcast_ref().unwrap();
            let items = vulkan_device.kernels.discard_pool.pop_ready_items(
              vulkan_device
                .kernels
                .next_submit_value
                .load(core::sync::atomic::Ordering::Relaxed)
                - 1,
            );
            vulkan::device::DiscardPool::destroy_items_lock_free(&vulkan_device.device, items);
            Ok(())
          });
        }

        let scene_ids: alloc::vec::Vec<u64> = {
          let scenes = context.scenes.read();
          scenes.keys().copied().collect()
        };

        for scene_id in &scene_ids {
          let pc = play_controls
            .entry(*scene_id)
            .or_insert_with(|| PlayControl::new(target_frame_time));
          let now = oshal::os::time::get_monotonic_time();
          let last = pc.last_frame_start;
          let elapsed = now.saturating_sub(last);

          // ── Physics tick (only when previous step is complete) ────────────
          let (physics_done, cross_sync_data) = {
            let scenes = context.scenes.read();
            let opt = utils::self_sync_do_if_done(
              &scenes,
              *scene_id,
              context.kernels.0.clone(),
              context.kernels.1,
              &context.render_tx,
              now,
              elapsed,
              |vulkan_device, scene_write, render_tx| {
                use core::sync::atomic::Ordering;
                use oshal::os::native::this_thread;
                // Cross Sync: send a SyncParticleRelease command to render thread, register
                // its Arc pointer for polling
                let last_render_task = scene_write.last_render_task.load(Ordering::Acquire);
                if last_render_task == 0 {
                  return None;
                }

                // 1. Polling Window: (0.2ms interval, max 2ms deadline)
                // Ensure the render thread is idle for this scene and we don't race
                // modifying scene particles and data.
                let mut render_idle = false;
                let start = get_monotonic_time();
                while (get_monotonic_time() - start) < 2000 {
                  if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                    render_idle = true;
                    break;
                  }
                  this_thread::sleep_for(core::time::Duration::from_micros(200));
                }
                // if we missed the deadline, simulation will try to update on the next
                // fixed update
                if !render_idle {
                  return None;
                }

                // 2. Request Graphics Release
                use alloc::boxed::Box;
                use bytemuck::Zeroable;
                let release_feeback = alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(0));
                let feedback_data_ptr =
                  Box::into_raw(Box::new(SyncParticleReleaseFeedback::zeroed()));
                // TODO handle error? retime submission?
                let mut done = false;
                while !done {
                  if let Ok(_) = render_tx.try_send(structs::RenderCommand::SyncParticleRelease {
                    feedback: release_feeback.clone(),
                    feedback_ptr: structs::SendPtrMut(feedback_data_ptr),
                  }) {
                    done = true;
                  }
                  core::hint::spin_loop();
                }

                Some((release_feeback, feedback_data_ptr))
              },
            );
            if opt.is_some() {
              (true, unsafe { opt.unwrap_unchecked() })
            } else {
              (false, None)
            }
          };

          // bookkeping for rendering submission before moving the data
          let do_cross_sync = cross_sync_data.is_some();

          let update_result: EngineResult<SimulationTickOutput> = {
            let scenes = context.scenes.read();
            // SAFETY: if `physics_done` then this scene exists
            let scene_arc =
              alloc::sync::Arc::clone(unsafe { scenes.get(&scene_id).as_ref().unwrap_unchecked() });

            // - Tick Time manager
            let end_epoch_reached = {
              // SAFETY: if `physics_done` then there should be time manager
              let mut time_mgr =
                unsafe { scenes.time_managers.get_mut(&scene_id).unwrap_unchecked() };
              if time_mgr.current_epoch() < time_mgr.end_epoch {
                time_mgr.tick();
                false
              } else {
                true
              }
            }; // <- dashmap write shard lock dropped

            // - Phase 1: Fixed update
            let phase1_res: Option<EngineResult<_>> = if physics_done && !end_epoch_reached {
              context
                .kernels
                .0
                .with_device(context.kernels.1, |dyn_device| {
                  let vulkan_device: &vulkan::device::Device =
                    dyn_device.as_any().downcast_ref().unwrap();
                  let mut time_mgr =
                    unsafe { scenes.time_managers.get_mut(&scene_id).unwrap_unchecked() };

                  Ok(Some(execute_simulation_tick_fixed_update_phase(
                    vulkan_device,
                    *scene_id,
                    scene_arc.upgradable_read(),
                    &mut time_mgr,
                    structs::UNSCALED_FIXED_DELTA_US,
                    cross_sync_data,
                    &scenes.cartesian_state_cache,
                    &context.logic_state.read().almanac_data,
                  )))
                })
                .unwrap_or(Some(Err(EngineError::InvalidNullArgument)))
            } else {
              None
            };

            // - Phase 2: Update
            let time_mgr = unsafe { scenes.time_managers.get(&scene_id).unwrap_unchecked() };
            execute_simulation_tick_update_phase(scene_arc.upgradable_read(), &time_mgr);

            // - Phase 3: Clear Changed
            execute_simulation_tick_clear_changed_entities_phase(
              &scene_arc,
              *scene_id,
              &context.thread_pool,
            );

            match phase1_res {
              None => Ok(SimulationTickOutput {
                pending_particle_acquire: None,
                latest_physics_sync: None,
              }),
              Some(Ok(s)) => {
                use core::sync::atomic::Ordering;
                // self sync function put this to false. Since we executed correctly a physics
                // step, put this to true
                scene_arc.read().active_physics_task.store(true, Ordering::Relaxed);
                Ok(s)
              }
              Some(Err(e)) => Err(e),
            }
          };

          // report error
          if let Err(ref e) = update_result {
            oshal::log!("[Update Error] {}", e);
            emit_breadcrumb(2, &e.to_string());
          }

          // extract the particle system compute timeline release/signal value for the render
          // command
          let pending_particle_acquire =
            update_result.map(|s| s.pending_particle_acquire).unwrap_or(None);

          // ── Render frame (always, at display rate) ────────────────────────
          // Uses active_physics_task + cached_timeline_semaphore.  If physics
          // is still running the render thread's try_wait(8ms) will fall back
          // to the cached semaphore value, keeping render independent.
          let pe_handles: alloc::vec::Vec<(
            crate::gpu::PresentationEngineHandle,
            crate::simulation_api::structs::PresentationEngineData,
          )> = {
            let scenes = context.scenes.read();
            if let Some(scene_ctx) = scenes.get(&scene_id) {
              scene_ctx
                .read()
                .presentation_engines
                .read()
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect()
            } else {
              alloc::vec::Vec::new()
            }
          };

          // Note: frame rate governing with play controls struct only for rendering submission.
          // Physics rate governing done though TimeManager
          if elapsed >= pc.target_frame_time || do_cross_sync {
            // Always reset the frame timer and submit a render frame at display
            // rate.  The physics TICK is gated separately — we only advance
            // simulation when the previous GPU compute step is done.
            pc.last_frame_start = now;

            let mut render_frames = alloc::vec::Vec::new();
            let mut last_tasks = alloc::vec::Vec::new();

            for (pe, pe_data) in pe_handles {
              let Some(camera_entity) = pe_data.camera_entity else {
                continue;
              };
              let is_windowless = pe_data.is_windowless;
              let task_id = alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(0));
              let scene = {
                let scenes = context.scenes.read();
                scenes.get(scene_id).unwrap().clone()
              };

              let (outlines, sun, sky, cursor, callback) = {
                let r = scene.read();
                (
                  r.outlines_enabled.load(core::sync::atomic::Ordering::Acquire),
                  r.sun_entity,
                  r.sky_entity,
                  r.cursor_entity,
                  r.custom_render_callback,
                )
              };

              render_frames.push(structs::RenderFrame {
                presentation_engine_handle: pe,
                task_id: alloc::sync::Arc::clone(&task_id),
                scene,
                render_physical_meshes_outline: outlines,
                camera_entity,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                sun_entity: sun,
                sky_entity: sky,
                cursor_entity: cursor,
                custom_render_callback: callback,
                particle_acquire_sync: pending_particle_acquire,
                mean_intra_grains_distance_mm:
                  structs::particle_constants::MEAN_INTRA_GRAINS_DISTANCE_MM,
                min_cumulated_mass_g: structs::particle_constants::MIN_CUMULATED_MASS_G,
              });

              last_tasks.push((task_id, is_windowless, pe.0));
            }

            let mut new_tasks = alloc::vec::Vec::new();
            if !render_frames.is_empty() {
              let send_res = context.render_tx.try_send(
                crate::simulation_api::structs::RenderCommand::RenderFrames(render_frames),
              );

              if send_res.is_ok() {
                for (idx, (task_id, is_windowless, pe_handle)) in last_tasks.iter().enumerate() {
                  // Fire-and-forget: do NOT spin-wait for the render tasklet to call
                  // create_task() and write back the task_id.  The old spin (with 1ms
                  // sleeps) blocked the logic thread for the full render frame duration
                  // (~50 ms at 20 FPS), which was the primary throughput cap.
                  //
                  // • Windowed PEs: new_tasks is no longer used to gate can_tick (see
                  //   comment below), so task_id_val is never needed here.
                  // • Windowless PEs: the WindowlessCallbackWorkload already polls
                  //   task_id asynchronously on the thread pool — no sync wait needed.
                  //
                  // Use u64::MAX as a sentinel for "task_id not yet known"; the
                  // windowless callback workload handles the 0→real transition itself.
                  let task_id_val = u64::MAX;

                  // For a successful frame, track the task so can_tick can check its status.
                  // For an error frame (u64::MAX), skip tracking — it would always show Invalid
                  // and would not block future ticks anyway.
                  if task_id_val != u64::MAX {
                    if let Some(nz) = core::num::NonZero::new(task_id_val) {
                      new_tasks.push(nz);

                      // Scene Context Bookkeping: Store last render task id
                      // Note: assuming the results we get are in order
                      if let Some(scene_arc) = context.scenes.read().get(&scene_ids[idx]) {
                        scene_arc
                          .read()
                          .last_render_task
                          .store(nz.get(), core::sync::atomic::Ordering::Release);
                      }
                    }
                  }

                  // Always fire the callback for windowless PEs, even for error frames
                  // (task_id_val == u64::MAX). C# uses the sentinel to log errors (rate-limited).
                  if *is_windowless {
                    let fptr = *crate::simulation_api::RENDER_CALLBACK.read();
                    if fptr.is_some() {
                      let _captured_task_id_val = task_id_val;

                      struct WindowlessCallbackWorkload {
                        ctx_ptr: crate::simulation_api::structs::SendPtrMut<core::ffi::c_void>,
                        task_id: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
                        scene_id: u64,
                        pe_handle: u64,
                      }

                      impl aethervk_oshal_rlib::os::pool::Workload for WindowlessCallbackWorkload {
                        fn execute(&mut self) -> aethervk_oshal_rlib::os::pool::WorkloadStatus {
                          let fptr = *crate::simulation_api::RENDER_CALLBACK.read();
                          if fptr.is_none() {
                            return aethervk_oshal_rlib::os::pool::WorkloadStatus::Complete;
                          }

                          let tid_val = self.task_id.load(core::sync::atomic::Ordering::Acquire);
                          if tid_val == 0 {
                            if alloc::sync::Arc::strong_count(&self.task_id) == 1 {
                              // Render thread dropped it without assigning
                              unsafe { fptr.unwrap()(self.scene_id, self.pe_handle, u64::MAX) };
                              return aethervk_oshal_rlib::os::pool::WorkloadStatus::Complete;
                            }
                            return aethervk_oshal_rlib::os::pool::WorkloadStatus::Yield;
                          }

                          let ctx = unsafe {
                            &*(self.ctx_ptr.get() as *mut crate::simulation_api::SimulationContext)
                          };
                          let completed = ctx
                            .render_proxy
                            .0
                            .as_frontend()
                            .and_then(|f| {
                              f.with_device(ctx.render_proxy.1, |device| {
                                Ok(device.is_task_completed(tid_val).unwrap_or(true))
                              })
                              .ok()
                            })
                            .unwrap_or(true);

                          if completed {
                            unsafe { fptr.unwrap()(self.scene_id, self.pe_handle, tid_val) };
                            aethervk_oshal_rlib::os::pool::WorkloadStatus::Complete
                          } else {
                            aethervk_oshal_rlib::os::pool::WorkloadStatus::Yield
                          }
                        }

                        fn tasklet_id(&self) -> Option<usize> {
                          None
                        }
                      }

                      let _ = context.thread_pool.scatter(alloc::vec![alloc::boxed::Box::new(
                        WindowlessCallbackWorkload {
                          ctx_ptr: context.ctx_ptr,
                          task_id: alloc::sync::Arc::clone(&task_id),
                          scene_id: *scene_id,
                          pe_handle: *pe_handle,
                        }
                      )]);
                    }
                  }
                }
              }
            }
            // last_render_ticks is retained for future use but no longer gates
            // can_tick; physics task completion is the gate now.
            pc.last_render_ticks = new_tasks;
            processed_any = true;
          }
        }

        // TODO: measure time since last device compute discard. do that every 500ms.

        // Drain all queued LogicCommands now, before sleeping.
        // drain_logic_commands also handles heavy tasks (scattered to pool).
        if drain_logic_commands(&logic_rx, &context) {
          return true; // Shutdown received
        }

        if !processed_any {
          // Sleep precisely until the next frame deadline rather than always 1 ms.
          //
          // With a fixed 1 ms sleep, the logic thread wakes ~15 times between
          // frames (16 ms target), each time finding nothing to do.  The repeated
          // scheduler wakeups add jitter to the render pipeline and waste CPU that
          // the GPU/render thread could use instead — especially harmful for
          // windowless mode where there is no display VSync as a natural gate.
          let sleep_us: i64 = if play_controls.is_empty() {
            1_000 // 1 ms fallback when no scenes exist
          } else {
            let now_us = oshal::os::time::get_monotonic_time();
            play_controls
              .values()
              .map(|pc| {
                pc.target_frame_time.saturating_sub(now_us.saturating_sub(pc.last_frame_start))
              })
              .min()
              .unwrap_or(1_000)
          };
          // Clamp: always sleep at least 100 µs (avoid tight spin) and at most
          // half a target frame (so we don't overshoot the deadline by much).
          let sleep_us = sleep_us.max(100).min(8_000);
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_micros(
            sleep_us as u64,
          ));
        }
        false
      };

      #[cfg(target_os = "macos")]
      let should_return = objc2::rc::autoreleasepool(|_| core_logic());

      #[cfg(not(target_os = "macos"))]
      let should_return = core_logic();

      if should_return {
        // ── Shutdown: drain all in-flight GPU physics tasks before exiting ──────────
        {
          let scenes_guard = context.scenes.read();
          for (_, scene_ctx_lock) in scenes_guard.iter() {
            if let Some(mut sync) = scene_ctx_lock.write().latest_physics_sync.take() {
              aethervk_oshal_rlib::log!("[shutdown] waiting for in-flight GPU physics task...");
              // Use a 5-second timeout instead of blocking indefinitely.
              // A hung GPU dispatch (e.g. from a large dt step before our safety cap)
              // would otherwise freeze the process forever on shutdown.
              let res = context
                .kernels
                .0
                .with_device(context.kernels.1, |dyn_device| {
                  let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
                    dyn_device.as_any().downcast_ref().unwrap();
                  const SHUTDOWN_GPU_TIMEOUT_US: i64 = 5_000_000;
                  use oshal::os::native::this_thread;
                  use oshal::os::time::get_monotonic_time;
                  let mut elapsed = 0_i64;
                  let mut wait_value = 16_i64;
                  let mut res = false;

                  loop {
                    if elapsed > SHUTDOWN_GPU_TIMEOUT_US {
                      break;
                    }
                    this_thread::sleep_for(core::time::Duration::from_micros(wait_value as _));
                    elapsed = get_monotonic_time();
                    if sync.try_wait(&vulkan_device.device, elapsed, wait_value) {
                      res = true;
                      break;
                    } else {
                      wait_value *= 2;
                    }
                  }

                  Ok(res)
                })
                .unwrap();
              if res {
                aethervk_oshal_rlib::log!("[shutdown] GPU physics task drained.");
              } else {
                aethervk_oshal_rlib::log!(
                  "[shutdown] GPU physics task did not finish within 5 s — forcing shutdown. \
                     The process will exit; OS will clean up GPU resources."
                );
              }
            }
          }
        }
        return;
      }
    }
  })
  .map_err(<ThreadingError as Into<NativeError>>::into)
  .map_err(<NativeError as Into<EngineError>>::into)
}

impl oshal::os::pool::Workload for LogicWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    // safety: checked by logic thread before scattering
    // Note: Task creation done at the FFI Layer `ffi.rs`
    if let Err(e) = process_command_internal(self.cmd.clone(), &self.ctx) {
      let cmd_desc = logic_command_desc(&self.cmd);
      crate::simulation_api::emit_breadcrumb(3, &alloc::format!("Failed: {} - {}", cmd_desc, e));
    }

    WorkloadStatus::Complete
  }
}

fn process_command_internal(
  command: LogicCommand,
  ctx: &alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<()> {
  match command {
    LogicCommand::StartSimlulation { scene_id, speed } => {
      let scenes = ctx.scenes.read();
      let mut time_mgr = scenes.time_managers.get_mut(&scene_id).unwrap();

      // Determine if this is a fresh start, knowing that ResetSimulation will zero out time state
      let is_fresh_start = {
        let state = time_mgr.state.read();
        state.scaled_time == 0
      };

      if is_fresh_start {
        use core::sync::atomic::Ordering;
        // If fresh start, then reset all particle systems present in the scene
        // Note: on first start this is unncecessary.
        let (now, elapsed) = {
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };

        // wait until compute queue is idle (self sync)
        utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(), // render frontend is an arc
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _render_tx| {
            use aethervk_oshal_rlib::os::time::get_monotonic_time;
            // Solve cross sync: Wait for the graphics queue to finish drawing its last frame
            // Since `StartSimulation` is a synchronous command, once we know that we are not
            // rendering we are sure that rendering won't restart until this command is processed
            let last_render_task = scene_write.last_render_task.load(Ordering::Acquire);
            if last_render_task != 0 {
              let wait_start = get_monotonic_time();
              while get_monotonic_time() - wait_start < 500_000_i64 {
                // 500ms deadline
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                core::hint::spin_loop();
              }
            }

            // now we are free to zero fill with a GPU memset all particle buffers
            // TODO error?
            let _ = unsafe { vulkan_device.reset_all_particle_systems() };

            // Reset the ECS Components so the physics compute shader starts fresh
            scene_write.scene.query1_mut(|_, comp: &mut ParticleSystemComponent| {
              comp.last_emission.store(0, Ordering::Relaxed);
              comp.last_compaction.store(0, Ordering::Relaxed);
            })
          },
        )
        .unwrap();
      }

      // 4. Finally, apply the new speed to physically resume/start the time manager
      time_mgr.set_speed(speed);

      Ok(())
    }
    LogicCommand::SetEpochRange {
      scene_id,
      start,
      end,
    } => {
      let scene_data = ctx.scenes.read();
      let mut time_mgr = scene_data
        .time_managers
        .get_mut(&scene_id)
        .ok_or(EngineError::InvalidOperation("no scene"))?;
      time_api::set_epoch_range(&mut time_mgr, start, end)?;
      emit_external_state_change(&ExternalState::TimeRange(CTimeRange::new(
        time_mgr.start_epoch,
        time_mgr.end_epoch,
      )));
      drop(time_mgr);

      // ── Forced repositioning ─────────────────────────────────────────────────────────────
      // If Earth has an AlmanacPlanet component (i.e. initEarth ran successfully), snap it
      // to the new start_epoch. This ensures the Earth sphere appears at the correct
      // heliocentric position when the user changes the timeline.
      let scene_arc = scene_data.get_scene(scene_id).ok_or(EngineError::InvalidOperation(
        "SetEpochRange: scene not found",
      ))?;
      let scene_guard = scene_arc.read();
      if let Some(earth) = scene_guard.earth {
        let planet_opt = scene_guard.scene.with_component(earth.body, |p: &AlmanacPlanet| *p);
        if let Some(planet) = planet_opt {
          let logic_state = ctx.logic_state.read();
          if let Err(e) = crate::simulation_api::reposition::force_reposition(
            &scene_guard.scene,
            earth.subtree,
            earth.body,
            &logic_state.almanac_data,
            &planet,
            start,
          ) {
            emit_breadcrumb(
              3,
              &alloc::format!("[SetEpochRange] Earth reposition failed: {}", e),
            );
          }
        }
      }

      // ── Trajectory year-change detection ─────────────────────────────────────────────────
      // Rebuild Earth orbit trajectory if start_epoch crosses into a new calendar year.
      // The trajectory covers the full year, so it only needs rebuilding at year boundaries.
      let new_year = crate::simulation_api::reposition::year_of_epoch(start);
      let stored_year = scene_guard.earth_orbit_year;
      if stored_year != Some(new_year) {
        if let Some(earth) = scene_guard.earth {
          // Only rebuild if Earth is driven by almanac (AlmanacPlanet attached)
          let has_planet =
            scene_guard.scene.with_component(earth.body, |_: &AlmanacPlanet| ()).is_some();
          if has_planet {
            let (traj_start, traj_end) =
              crate::simulation_api::reposition::full_year_tai_seconds(new_year);
            let workload = Box::new(structs::LogicWorkload {
              cmd: LogicCommand::UpdateTrajectoryForSpk {
                task_id: 0,
                scene_id,
                entity_id: EntityId::as_ffi(&earth.orbit),
                spk_id: anise::constants::celestial_objects::EARTH,
                start_epoch_tai_sec: traj_start,
                end_epoch_tai_sec: traj_end,
                sample_step_days: 1.0,
              },
              ctx: alloc::sync::Arc::clone(ctx),
            });
            let _ = ctx.thread_pool.scatter(alloc::vec![workload]);
            drop(scene_guard);
            scene_arc.write().earth_orbit_year = Some(new_year);
          }
        }
      }

      Ok(())
    }
    LogicCommand::InitComet { scene_id, spk_id } => {
      // TODO: implement comet initialization (attach AlmanacPlanet, force_reposition,
      // dispatch UpdateTrajectoryForSpk for Comet_orbit). Requires probe_spk_file_with_domain
      // to discover the NAIF ID and validate coverage.
      let _ = (scene_id, spk_id);
      Ok(())
    }
    LogicCommand::CleanupComet { scene_id } => {
      // TODO: implement comet cleanup (remove AlmanacPlanet, remove TrajectoryComponent,
      // reset Comet_subtree to 1 AU +X, reset Comet_body to origin).
      let _ = scene_id;
      Ok(())
    }
    LogicCommand::AnimateCameraTo {
      scene_id,
      camera_id,
      target_pos,
      target_rot,
      duration_s,
    } => {
      let scenes = ctx.scenes.read();
      let scene_arc = crate::expect_scene!(scenes.get_scene(scene_id), "AnimateCameraTo");
      let mut scene = scene_arc.write();

      let cam_int = scene.get_entity(camera_id).ok_or(EngineError::InvalidOperation(
        "AnimateCameraTo | camera entity not found",
      ))?;

      // If an animation is already in flight, retarget it — no snap, speed preserved.
      let retargeted = scene.scene.with_component_mut(
        cam_int,
        |anim: &mut crate::scene::animation::TransformAnimationComponent| {
          anim.retarget(target_pos, target_rot);
        },
      );

      if retargeted.is_none() {
        // No active animation — read current transform as the start point.
        let (start_pos, start_rot) = scene
          .scene
          .with_component(cam_int, |t: &HighResTransformComponent| {
            (t.position, t.rotation)
          })
          .ok_or(EngineError::InvalidOperation(
            "AnimateCameraTo | camera has no HighResTransformComponent",
          ))?;

        let _ = scene.scene.add_component(
          cam_int,
          crate::scene::animation::TransformAnimationComponent {
            start_pos,
            start_rot,
            target_pos,
            target_rot,
            duration: duration_s,
            elapsed: 0.0,
            is_finished: false,
          },
        );
      }

      // Immediately mark changed so the first interpolated frame reaches C# without
      // waiting for the next full tick.
      scene.mark_component_changed(
        camera_id,
        <HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );

      Ok(())
    }
    LogicCommand::SnapshotScene { scene_id } => {
      use oshal::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      // Retry stopping the current task for a deadline of 500ms. Otherwise die
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 500_000_i64 {
        let (now, elapsed) = {
          let time_mgr =
            scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };
        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(), // render frontend is an arc
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, render_tx| {
            let cloned_scene = (*scene_write.scene).clone();
            scene_write.scene_snapshot = Some(alloc::boxed::Box::new(cloned_scene));

            // Compute is guaranteed idle here. Snapshot the particles directly:
            if let Ok(particle_snap) = vulkan_device.snapshot_particles() {
              scene_write.particle_snapshot = Some(particle_snap);
            }
          },
        ) {
          return Ok(());
        }
      }
      emit_breadcrumb(2, "Failed to create scene snapshot");
      Err(EngineError::InvalidOperation(
        "Failed to create scene snapshot",
      ))
    }
    // TODO playtoend command!
    LogicCommand::PlaySceneToEnd { scene_id, speed } => {
      todo!()
    }

    // TODO now this command will be fused with StopScene, therefore
    //commenting out pieces as I see fit is perfectly fine
    LogicCommand::RestoreSnapshot { scene_id } => {
      use oshal::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      // Retry stopping the current task for a deadline of 500ms. Otherwise die
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 500_000_i64 {
        let (now, elapsed) = {
          let time_mgr =
            scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };
        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(), // render frontend is an arc
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, render_tx| {
            // 1 Wait for the render thread to be dile to that we can overwrite the front buffer for
            //   particle systems
            let last_render_task =
              scene_write.last_render_task.load(core::sync::atomic::Ordering::Acquire);
            if last_render_task != 0 {
              let wait_start = oshal::os::time::get_monotonic_time();
              // 500ms safety timeout
              while (oshal::os::time::get_monotonic_time() - wait_start) < 500_000_i64 {
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                oshal::os::native::this_thread::sleep_for(core::time::Duration::from_micros(200));
              }
            }

            // 2. restore GPU particle state (into both front and back buffers)
            if let Some(particle_snap) = scene_write.particle_snapshot.as_ref() {
              let _ = vulkan_device.restore_particles(particle_snap);
            }

            // - take scene overrides (BodyRotationalModel)
            // 3 Restore the snapshot
            if let Some(snapshot) = scene_write.scene_snapshot.take() {
              scene_write.scene = snapshot.into();
            }

            // 4 empty the cartesian cache
            let mut keys = alloc::vec::Vec::with_capacity(128);
            scenes.cartesian_state_cache.iter().for_each(|kv_ref| keys.push(*kv_ref.key()));
            for key in keys {
              if key.scene_id == scene_id {
                scenes.cartesian_state_cache.remove(&key);
              }
            }

            // 5 mark as changed all transform, camera, highres transform components
            utils::mark_all_serializable_as_changed(&scene_write);
          },
        ) {
          return Ok(());
        }
      }

      Err(EngineError::InvalidOperation("Failed to restore snapshot"))
    }
    LogicCommand::SetEntityVisibility {
      scene_id,
      entity,
      visible,
    } => {
      use crate::scene::EntityId;
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
        // Resolve external entity id to internal id.
        let root_id: EntityId = {
          let r = scene_ctx_guard.read();
          match r.get_entity(entity) {
            Some(id) => id,
            None => return Ok(()),
          }
        };
        // Collect full subtree under a short-lived read lock.
        let to_update: alloc::vec::Vec<EntityId> = {
          let r = scene_ctx_guard.read();
          let mut queue = alloc::vec![root_id];
          let mut all = alloc::vec![root_id];
          while let Some(current) = queue.pop() {
            if let Some(children) = r.scene.get_children(current) {
              for child in children {
                queue.push(child);
                all.push(child);
              }
            }
          }
          all
        };
        // Now acquire the write lock — safe here since the logic thread is between ticks
        // and holds no conflicting read lock at this point.
        let scene_ctx = scene_ctx_guard.write();
        for id in &to_update {
          if visible {
            let _ = scene_ctx.scene.remove_component::<crate::scene::HiddenComponent>(*id);
          } else {
            let _ = scene_ctx.scene.add_component(*id, crate::scene::HiddenComponent {});
          }
        }
      }
      Ok(())
    }
    LogicCommand::Shutdown => Ok(()),
    LogicCommand::RotateCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    } => {
      let scene_read = scene.read();

      let mut cursor_pos = None;
      if let Some((cursor_id, _)) = scene_read
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        if let Some(pos) = scene_read
          .scene
          .with_component(cursor_id, |t: &HighResTransformComponent| t.position)
        {
          cursor_pos = Some(pos);

          // Sync focus distance dynamically so zoom/pan speeds stay stable
          // based on the distance to the cursor object you are pivoting around.
          // Computed in f64 to preserve precision at extreme zoom.
          if let Some(cam_pos) = scene_read
            .scene
            .with_component(camera_entity, |t: &HighResTransformComponent| t.position)
          {
            let dist = (pos - cam_pos).length();
            let _ =
              scene_read.scene.with_component_mut(camera_entity, |c: &mut CameraComponent| {
                c.focus_distance = (dist as f32).max(0.000001);
              });
          }
        }
      }

      use crate::scene::camera::SceneCameraExt;
      let rotation_speed: f32 = 0.005;

      scene_read.scene.orbit_camera(
        camera_entity,
        -delta_x * rotation_speed,
        -delta_y * rotation_speed,
        cursor_pos,
      )?;

      scene_read.mark_component_changed(
        EntityId::as_ffi(&camera_entity),
        <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );
      Ok(())
    }

    LogicCommand::ZoomCamera {
      camera_entity,
      scene,
      amount,
    } => {
      let scene_read = scene.read();
      let mut is_ortho = false;
      let mut focus_dist = 10.0;

      let _ = scene_read.scene.with_component(camera_entity, |c: &CameraComponent| {
        is_ortho = matches!(
          c.projection,
          crate::scene::CameraProjection::Orthographic { .. }
        );
        focus_dist = c.focus_distance;
      });

      if is_ortho {
        let zoom_factor = 1.0 - (amount * 0.1);
        let _ = scene_read.scene.with_component_mut(camera_entity, |c: &mut CameraComponent| {
          if let crate::scene::CameraProjection::Orthographic {
            ref mut left,
            ref mut right,
            ref mut bottom,
            ref mut top,
            ..
          } = c.projection
          {
            *left *= zoom_factor;
            *right *= zoom_factor;
            *bottom *= zoom_factor;
            *top *= zoom_factor;
          }
        });
        scene_read.mark_component_changed(
          EntityId::as_ffi(&camera_entity),
          <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
        return Ok(());
      }

      let dist = scene_read
        .scene
        .with_component(camera_entity, |c: &crate::scene::CameraComponent| {
          c.focus_distance
        })
        .unwrap_or(10.0);

      use crate::scene::camera::SceneCameraExt;
      // Logarithmic zoom: each scroll moves 2% of current focus distance.
      // This naturally decelerates as you approach objects at any scale.
      // Computed in f64 to preserve precision at micro-scale (focus_distance ~1e-10).
      let zoom_speed = dist as f64 * 0.02;
      let move_amount = -(amount as f64) * zoom_speed;

      scene_read.scene.translate_camera_local(
        camera_entity,
        Vec3f64::from_components(0.0, move_amount, 0.0),
      )?;

      // Update focus distance so the invisible View Center stays in the exact same world position!
      // Low clamp (1e-10 AU ≈ 0.015 mm) allows zooming to micro-scale objects.
      let _ = scene_read.scene.with_component_mut(
        camera_entity,
        |c: &mut crate::scene::CameraComponent| {
          c.focus_distance = (c.focus_distance as f64 + move_amount).max(1e-10) as f32;
        },
      );

      scene_read.mark_component_changed(
          EntityId::as_ffi(&camera_entity),
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      scene_read.mark_component_changed(
        EntityId::as_ffi(&camera_entity),
        <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );
      Ok(())
    }
    LogicCommand::ResetCamera {
      camera_entity,
      scene,
    } => {
      let scene_read = scene.read();

      let mut cursor_pos = Vec3f32::zero();
      if let Some((sun_id, _)) = scene_read
        .scene
        .query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
      {
        if let Some(pos) =
          scene_read.scene.with_component(sun_id, |t: &TransformComponent| t.position)
        {
          cursor_pos = pos;
        }
      }

      if let Some((cursor_id, _)) = scene_read
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        let _ =
          scene_read
            .scene
            .with_component_mut(cursor_id, |c: &mut HighResTransformComponent| {
              c.position = cursor_pos.to_f64();
            });
      }

      const HOME_DISTANCE: f32 = 0.07;
      let pitch = (-1.0_f32 / 3.0_f32.sqrt()).asin();
      let yaw = -core::f32::consts::FRAC_PI_4;
      let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      let offset = q.rotate_vector(Vec3f32::from_components(0.0, HOME_DISTANCE, 0.0));

      scene_read
        .scene
        .with_component_mut(camera_entity, |t: &mut HighResTransformComponent| {
          t.position = (cursor_pos + offset).to_f64();
          t.rotation = q;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:ResetCamera | camera entity doesn't have HighResTransformComponent",
        ))?;

      let _ = scene_read.scene.with_component_mut(camera_entity, |c: &mut CameraComponent| {
        c.focus_distance = HOME_DISTANCE;
      });

      scene_read.mark_component_changed(
        EntityId::as_ffi(&camera_entity),
        <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );

      Ok(())
    }
    LogicCommand::PanCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    } => {
      let scene_read = scene.read();
      use crate::scene::camera::SceneCameraExt;
      scene_read.scene.pan_camera(camera_entity, delta_x, delta_y)?;

      scene_read.mark_component_changed(
        EntityId::as_ffi(&camera_entity),
        <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );

      Ok(())
    }
    LogicCommand::MoveCursor {
      scene,
      delta_x,
      delta_y,
      delta_z,
    } => {
      let speed = 0.001;
      let scene_read = scene.read();
      scene_read
        .scene
        .query2_res_first_mut(|id, t: &mut HighResTransformComponent, _c: &mut CursorComponent| {
          let translation = Vec3f32::from_components(delta_x, delta_y, delta_z) * speed;
          t.position = t.position + translation.to_f64();

          scene_read.mark_component_changed(
            EntityId::as_ffi(&id),
            <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
          );

          Some(())
        })
        .map(|_| ())
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:MoveCursor | scene doesn't have cursor",
        ))?;
      Ok(())
    }
    LogicCommand::SnapToEntity {
      snap_entity,
      target_entity,
      scene,
    } => {
      let mut scene_write = scene.write();
      // 'F' behavior: move cursor to target entity position, then position the camera dynamically
      let target_pos_dvec = {
        scene_write
          .scene
          .global_transform_f64(target_entity)
          .map(|t| t.position)
          .ok_or(EngineError::InvalidOperation(
            "logic_thread:SnapToEntity | target entity doesn't have TransformComponent",
          ))?
      };

      // Move cursor to target entity world position.
      if let Some((cursor_id, _)) = scene_write
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        if let Some(_) =
          scene_write
            .scene
            .with_component_mut(cursor_id, |c: &mut HighResTransformComponent| {
              c.position = target_pos_dvec;
            })
        {
          // Mark cursor entity as changed.
          utils::mark_component_changed::<HighResTransformComponent>(&scene_write, cursor_id);
        }
      }

      // TODO maybe: Dynamic offset calculation based on object bounds and camera FOV
      let target_radius = 0.1_f64;

      let mut fov = core::f64::consts::FRAC_PI_4;
      let mut aspect = 16.0 / 9.0;
      if let Some(cam) =
        scene_write.scene.with_component(snap_entity, |c: &CameraComponent| c.clone())
      {
        if let crate::scene::CameraProjection::Perspective {
          fov: cam_fov,
          aspect_ratio,
          ..
        } = cam.projection
        {
          fov = cam_fov as f64;
          aspect = aspect_ratio as f64;
        }
      }

      let half_min_fov = if aspect > 1.0 {
        fov / 2.0
      } else {
        ((fov / 2.0).tan() * aspect).atan()
      };

      // target distance to fill 5/6 of smallest axis
      let target_half_angle = (5.0 / 6.0) * half_min_fov;
      let snap_distance = target_radius / target_half_angle.tan();

      // Desired rotation from user request: recreating startup rotation
      let q = Quat::from_components(0.24757917, -0.098841526, -0.35735834, 0.8951145);
      let offset = q.rotate_vector(Vec3f32::from_components(0.0, snap_distance as f32, 0.0));

      let (start_pos, start_rot) = if let Some(t) = scene_write.scene.with_component(
        snap_entity,
        |t: &crate::scene::HighResTransformComponent| (t.position, t.rotation),
      ) {
        t
      } else if let Some(t) =
        scene_write.scene.with_component(snap_entity, |t: &TransformComponent| {
          (t.position, t.rotation)
        })
      {
        (t.0.to_f64(), t.1)
      } else {
        return Ok(());
      };

      let anim = crate::scene::animation::TransformAnimationComponent {
        start_pos,
        start_rot,
        target_pos: target_pos_dvec + offset.to_f64(),
        target_rot: q,
        duration: 2.0,
        elapsed: 0.0,
        is_finished: false,
      };

      let _ = scene_write
        .scene
        .remove_component::<crate::scene::animation::TransformAnimationComponent>(snap_entity);
      let _ = scene_write.scene.add_component(snap_entity, anim);

      let _ = scene_write.scene.with_component_mut(snap_entity, |c: &mut CameraComponent| {
        c.focus_distance = snap_distance as f32;
      });

      scene_write.mark_component_changed(
        EntityId::as_ffi(&snap_entity),
        <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
      );

      Ok(())
    }
    LogicCommand::PlayScene { scene_id, speed } => {
      use oshal::os::time::v2::SimSpeed;
      if speed == SimSpeed::Paused {
        return Err(EngineError::InvalidOperation(
          "can't PlayScene with SimSpeed::Paused",
        ));
      }

      // Note: For now assuming state checking for playing the scene is done C# side. In particular,
      // the following conditions should hold
      // - at least 1 particle system component fully configured
      // - associated to a fully configured comet entity either static or with a
      //   `SpiceKinematicComopnent
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        let scene_guard = scene_ctx.read();
        let mut ts = scene_guard.time_state.write();
        ts.speed = speed;
        return Ok(());
      }
      Err(EngineError::InvalidOperation("can't find scene"))
    }
    LogicCommand::PauseScene { scene_id } => {
      use oshal::os::time::v2::SimSpeed;

      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().speed = SimSpeed::Paused;
      }
      Ok(())
    }

    LogicCommand::ImportModel { task_id: _, path } => {
      let mesh_res = if path.ends_with(".obj") || path.ends_with(".OBJ") {
        crate::simulation::comet::load_comet_from_obj(&path, false, None)
      } else if path.ends_with(".ply") || path.ends_with(".PLY") {
        crate::simulation::comet::load_comet_from_ply(&path, false, None)
      } else {
        crate::simulation::comet::load_comet_from_gltf(&path, false, None)
      };
      match mesh_res {
        Ok(mesh) => {
          let mut scenes = ctx.scenes.write();
          let model_id = scenes.import_model_from_mesh(&path, mesh);
          emit_external_state_change(&ExternalState::ModelImported(CModelImported::new(
            model_id, &path,
          )));
          Ok(())
        }
        Err(e) => Err(EngineError::from(e)),
      }
    }
    LogicCommand::LoadAlmanac { task_id: _, path } => {
      ctx.load_almanac_file_internal(&path)?;
      Ok(())
    }
    LogicCommand::UnloadAlmanac { task_id: _, path } => {
      ctx.unload_almanac_file_internal(&path)?;
      Ok(())
    }

    LogicCommand::UpdateTrajectoryForSpk {
      task_id: _,
      scene_id,
      entity_id,
      spk_id,
      start_epoch_tai_sec,
      end_epoch_tai_sec,
      sample_step_days,
    } => {
      let frame = crate::simulation::almanac::SUN_ECLIPJ2000;
      if sample_step_days <= 0.0 {
        return Err(EngineError::InvalidOperation(
          "UpdateTrajectoryForSpk: sample_step_days must be > 0",
        ));
      }
      if start_epoch_tai_sec >= end_epoch_tai_sec {
        return Err(EngineError::InvalidOperation(
          "UpdateTrajectoryForSpk: start_epoch >= end_epoch",
        ));
      }

      let scenes = ctx.scenes.read();
      let scene_ctx =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let scene_guard = scene_ctx.read();
      // TODO check why we are using internal id in the command here.
      let entity = EntityId::from(slotmap::KeyData::from_ffi(entity_id));

      // Note: trajectory control points are computed in heliocentric SUN_ECLIPJ2000 AU.
      // Orbit entities (Earth_orbit, Comet_orbit) are direct children of root_entity
      // (depth_layer=0) so the renderer's RTE path applies no AU_TO_KM scale distortion.
      // This structural invariant is enforced at construction time in create_subtree;
      // no runtime parent check is needed here.

      struct SampledPoints {
        pub position_km: DVec3,
        pub velocity_km: DVec3,
      }
      let mut samples = alloc::vec::Vec::<SampledPoints>::with_capacity(256);
      let mut t = start_epoch_tai_sec;

      let logic_state = ctx.logic_state.read();

      let step_sec = sample_step_days * 86400.0;

      // Ensure we at least sample the end point precisely
      while t <= end_epoch_tai_sec {
        let epoch = anise::time::Epoch::from_tai_seconds(t);
        let state = logic_state.almanac_data.get_cartesian_state(
          spk_id,
          frame.orientation_id,
          frame.ephemeris_id,
          epoch,
          true,
        )?;
        samples.push(SampledPoints {
          position_km: DVec3::from_components(
            state.radius_km[0],
            state.radius_km[1],
            state.radius_km[2],
          ),
          velocity_km: DVec3::from_components(
            state.velocity_km_s[0],
            state.velocity_km_s[1],
            state.velocity_km_s[2],
          ),
        });

        if t == end_epoch_tai_sec {
          break;
        }
        t += step_sec;
        if t > end_epoch_tai_sec {
          t = end_epoch_tai_sec;
        }
      }

      if samples.len() < 2 {
        return Err(EngineError::InvalidOperation(
          "UpdateTrajectoryForSpk: not enough samples",
        ));
      }

      let dt_sec = step_sec;
      let mut control_points = alloc::vec::Vec::<[f32; 4]>::with_capacity(256);

      for i in 0..(samples.len() - 1) {
        const KM_TO_AU: f64 = 6.6845871226706e-9_f64;
        let s0 = &samples[i];
        let s1 = &samples[i + 1];

        let p0 = s0.position_km;
        let v0 = s0.velocity_km;
        let p1 = s1.position_km;
        let v1 = s1.velocity_km;

        let cp0 = (p0 * KM_TO_AU).to_f32();
        let cp1 = ((p0 + v0 * (dt_sec / 3.0)) * KM_TO_AU).to_f32();
        let cp2 = ((p1 - v1 * (dt_sec / 3.0)) * KM_TO_AU).to_f32();
        let cp3 = (p1 * KM_TO_AU).to_f32();

        control_points.push([cp0.x(), cp0.y(), cp0.z(), 1.0]);
        control_points.push([cp1.x(), cp1.y(), cp1.z(), 1.0]);
        control_points.push([cp2.x(), cp2.y(), cp2.z(), 1.0]);
        control_points.push([cp3.x(), cp3.y(), cp3.z(), 1.0]);
      }

      // Apply the component to the entity
      let new_comp = crate::scene::trajectory::TrajectoryComponent::new(
        control_points,
        [0.3, 0.6, 1.0, 1.0], // Default color, maybe should be parameter
        2.0,
        0,
        32,
      );

      let mut replaced = false;
      let _ = scene_guard.scene.with_component_mut(
        entity,
        |comp: &mut crate::scene::trajectory::TrajectoryComponent| {
          *comp = new_comp.clone();
          replaced = true;
        },
      );

      if !replaced {
        let res = scene_guard.scene.add_component(entity, new_comp);
        if res.is_err() {
          return Err(EngineError::InvalidOperation(
            "UpdateTrajectoryForSpk: entity does not exist or invalid component add",
          ));
        }
      }

      // Update TransformComponent to match the Sun's position
      let mut sun_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
      if let Some((sun_id, _)) = scene_guard
        .scene
        .query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
      {
        if let Some(pos) = {
          #[allow(deprecated)]
          scene_guard.scene.global_transform(sun_id)
        }
        .map(|t| t.position)
        {
          sun_pos = pos;
        }
      }

      let mut handled_highres = false;
      let _ = scene_guard.scene.with_component_mut(
        entity,
        |transform: &mut HighResTransformComponent| {
          transform.position = sun_pos.to_f64();
          handled_highres = true;
        },
      );

      if !handled_highres {
        let mut handled_transform = false;
        let _ =
          scene_guard
            .scene
            .with_component_mut(entity, |transform: &mut TransformComponent| {
              transform.position = sun_pos;
              handled_transform = true;
            });
        if !handled_transform {
          let mut new_transform = TransformComponent::default();
          new_transform.position = sun_pos;
          let _ = scene_guard.scene.add_component(entity, new_transform);
        }
      }

      Ok(())
    }
  }
}

/// Struct returned from `execute_simulation_tick`
struct SimulationTickOutput {
  /// Holds, if present, timeline value from compute queue which will be signaled when a given
  /// Cross Sync, compute queue acquisition will be completed. Should be included among the wait
  /// timeline semaphores inside the render thread when rendering the first frame of a given scene
  /// after a cross sync
  pending_particle_acquire: Option<u64>,
  latest_physics_sync: Option<PhysicsDeviceSelfSync>,
}

/// Equivalent of [`crate::gpu::ScopedCommandBuffer`] for compute submission
struct ScopedComputeCommand<'a> {
  vulkan_device: &'a crate::gpu_backends::vulkan::device::Device,
  cmd_handle: crate::gpu::CommandBufferHandle,
  cmd: ash::vk::CommandBuffer,
  gfx_release_sync_info: Option<crate::gpu::CommandBufferSyncInfo>,
  submitted: bool,
}

impl<'a> ScopedComputeCommand<'a> {
  fn new(
    vulkan_device: &'a crate::gpu_backends::vulkan::device::Device,
    cmd_handle: crate::gpu::CommandBufferHandle,
    cmd: ash::vk::CommandBuffer,
  ) -> GpuResult<Self> {
    use crate::gpu_backends::vulkan::device::QueueRole;
    vulkan_device.begin_command_buffer_all(cmd_handle, QueueRole::Compute)?;
    Ok(Self {
      vulkan_device,
      cmd_handle,
      cmd,
      gfx_release_sync_info: None,
      submitted: false,
    })
  }

  fn set_gfx_sync(&mut self, gfx_timeline_sem: ash::vk::Semaphore, gfx_release_value: u64) {
    use ash::vk::Handle;
    self.gfx_release_sync_info = Some(crate::gpu::CommandBufferSyncInfo {
      timeline_semaphore: gfx_timeline_sem.as_raw(),
      timeline_value: gfx_release_value,
      wait_stage_mask: crate::gpu::CommandBufferSyncInfoStageMask::Transfer,
    });
  }

  fn submit(mut self) -> GpuResult<(ash::vk::Semaphore, u64)> {
    use crate::gpu_backends::vulkan::device::QueueRole;
    let (compute_sem, signal_value) = self.vulkan_device.submit_command_buffer_generic(
      self.cmd_handle,
      None,
      self
        .gfx_release_sync_info
        .as_ref()
        .map(|x| core::slice::from_ref(x))
        .unwrap_or(&[]),
      &[],
      QueueRole::Compute,
    )?;
    self.submitted = true;
    Ok((compute_sem, signal_value))
  }
}

impl<'a> Drop for ScopedComputeCommand<'a> {
  fn drop(&mut self) {
    use crate::gpu_backends::vulkan::device::QueueRole;
    if !self.submitted {
      let _ = self.vulkan_device.submit_command_buffer_generic(
        self.cmd_handle,
        None,
        self
          .gfx_release_sync_info
          .as_ref()
          .map(|x| core::slice::from_ref(x))
          .unwrap_or(&[]),
        &[],
        QueueRole::Compute,
      );
    }
  }
}

/// 1/3 Step of a simulation tick: Physics Update
///
/// ticks `time_mgr always`, updates `scene` after a simulation step executed successfully
/// Should return
/// - next timeline semaphore value so that we can update our `latest_physics_sync`
/// - whether or not we reached end epoch, and therefore simulation is finished
fn execute_simulation_tick_fixed_update_phase(
  vulkan_device: &crate::gpu_backends::vulkan::device::Device,
  scene_id: u64,
  mut scene: parking_lot::lock_api::RwLockUpgradableReadGuard<parking_lot::RawRwLock, SceneContext>,
  time_mgr: &mut oshal::os::time::v2::TimeManager,
  unscaled_fixed_delta_us: oshal::os::time::timeus_t,
  cross_sync_data: Option<(
    alloc::sync::Arc<core::sync::atomic::AtomicU64>,
    *mut SyncParticleReleaseFeedback,
  )>,
  cartesian_state_cache: &dashmap::DashMap<
    crate::simulation_api::structs::SceneEntityId,
    CartesianState,
  >,
  almanac: &AlmanacPackedData,
) -> EngineResult<SimulationTickOutput> {
  use crate::gpu_backends::vulkan::device::QueueRole;
  use crate::simulation_api::structs::SceneEntityId;
  use oshal::os::time::{timeus_t, us_to_300ths_rounded, v2::SimSpeed};
  let scaled_fixed_dt_us =
    time_mgr.state.read().speed.scaled_from_unscaled(unscaled_fixed_delta_us);
  // using [`SimSpeed::Custom`] might lead to precision troubles. Assert this is not the case
  debug_assert!(scaled_fixed_dt_us > 0);

  // ------------------------------------------------------------------------------------
  // -- Fixed Update Phase (Step 1: Command buffer creation and CPU side resolution) --
  // consume accumulated time for physics: SPICE spk_ezr Kernel with current epoch and velocity
  // verlet with `scaled_fixed_dt`
  // compute whether or not we reached end_epoch. and report it. if end epoch reached, skip fixed
  // update phase and go to update
  // ------------------------------------------------------------------------------------
  let (cmd_handle, cmd) = vulkan_device.get_command_buffer_and_native_all(QueueRole::Compute)?;
  let mut cmd_scope = ScopedComputeCommand::new(vulkan_device, cmd_handle, cmd)?;

  // ------------------------------------------------------------------------------------
  // -- Cross Sync Resolution --
  // ------------------------------------------------------------------------------------

  // Spin wait for the render thread to finish the polling with a 2ms deadline, 0.2ms
  // interval. If we can't finish on time, abort the update procedure
  let mut do_cross_sync = false;
  if let Some((feedback_arc, feedback_ptr)) = cross_sync_data {
    use oshal::os::native::this_thread;
    use oshal::os::time::get_monotonic_time;
    let mut release_task_id = 0_u64;
    let start = get_monotonic_time();
    // first task will be used as "transaction in progress" state cause we are sure that first value
    // is used for rendering task, not release
    while (release_task_id == 0 && (get_monotonic_time() - start) < 2000) || release_task_id == 1 {
      release_task_id = feedback_arc.load(core::sync::atomic::Ordering::Acquire);
      if release_task_id <= 1 {
        this_thread::sleep_for(core::time::Duration::from_micros(200));
      }
    }

    do_cross_sync = release_task_id <= 1 && release_task_id != u64::MAX;
    if do_cross_sync {
      use crate::gpu::new_particles::PAGE_TABLE_BYTES;
      use crate::gpu_backends::vulkan::utils::RwLockable;
      debug_assert_eq!(alloc::sync::Arc::strong_count(&feedback_arc), 1);
      // retrieve release timeline value and semaphore handle so that we can record submit wait
      // conditions
      let (gfx_timeline_sem, gfx_release_value) = {
        // SAFETY: if render command finished and we loaded the atomic with acquire semantics, then
        // this was successfully written and shouldn't be looked at by anyone else, therefore we own
        // this.
        let the_box = unsafe { alloc::boxed::Box::from_raw(feedback_ptr) };
        (the_box.timeline_semaphore, the_box.timeline_release_value)
        // drop the box
      };

      // record in compute queue command buffer the copy and release of the particle systems
      // get graphics queue timeline value on which the release operation will be completed
      // get compute queue timeline value for next compute submission (move outside)
      // record and submit with
      // - timeline sem from graphics as wait at stage TRANSFER
      // - timeline sem from compute as signal (default in submit)
      cmd_scope.set_gfx_sync(gfx_timeline_sem, gfx_release_value);
      {
        let res = vulkan_device.res.read();
        let psm = res.particle_system_manager.as_ref().unwrap();
        psm.cmd_sync_compute_copy_and_release(
          &vulkan_device.device,
          cmd,
          PAGE_TABLE_BYTES as _,
          vulkan_device.get_graphics_queue().family_index,
          vulkan_device.get_compute_queue().family_index,
        );
      }
      {
        let mut res = vulkan_device.res.write();
        let psm = res.particle_system_manager.as_mut().unwrap();
        unsafe { psm.swap_buffers() };
      }
    } else if release_task_id != u64::MAX {
      // handle failure: free `feedback_ptr` and do nothing
      // SAFETY: allocated by caller, untouched by render thread cause we received some feedback
      let _ = unsafe { alloc::boxed::Box::from_raw(feedback_ptr) };
    } else {
      // signal to sender that we are out of deadline, therefore you should free the `feedback_ptr`
      feedback_arc.store(u64::MAX, core::sync::atomic::Ordering::Release);
    }
  }

  // -- Fixed Update Phase (Step 2: Command buffer record and GPU compute queue submit) --
  let compute_signal_value = vulkan_device
    .kernels
    .next_submit_value
    .load(core::sync::atomic::Ordering::Relaxed);
  let (sim_speed, now_unscaled_us, now_scaled_us) = {
    let time_state = time_mgr.state.read();
    (
      time_state.speed,
      time_state.unscaled_time,
      time_state.scaled_time,
    )
  };
  let now_scaled_300ths = us_to_300ths_rounded(now_scaled_us);
  let latest_physics_sync: Option<PhysicsDeviceSelfSync> = if sim_speed != SimSpeed::Paused {
    // used for SPICE EZR Data Kernel and simulation
    let current_epoch = time_mgr.current_epoch();

    // ------------------------------------------------------------------------------------
    // Particle Systems: prepare extraction for all particle systems in the scene
    // ------------------------------------------------------------------------------------
    // prepare extraction for all particle systems in the scene
    use crate::scene::particles::v2::{ParticleSystemComponent, ParticleSystemComponentExtraction};
    let ps_extraction = scene.scene.query2_res(
      |_e_id, ps: &ParticleSystemComponent, t: &TransformComponent| {
        Some(ParticleSystemComponentExtraction::from_component(ps, t))
      },
    );
    let mut force_emitters = alloc::vec::Vec::with_capacity(ps_extraction.len());
    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
    struct ParticlesExecution {
      last_emission_unscaled_us: timeus_t,
      last_compaction_unscaled_us: timeus_t,
      global_pos_f64: [f64; 3],
      global_rot: [f32; 4],
      did_compact: u32, // bool32
      dead: u32,        // bool32, means in error
      r_helio_au: f32,
      _pad: u32,
    }
    use bytemuck::Zeroable;
    let mut particle_executions = alloc::vec![ParticlesExecution::zeroed(); ps_extraction.len()];

    let from_start_epoch_scaled_us = {
      let start_epoch = time_mgr.start_epoch;
      // If you know your durations will never exceed 292,000 years, you can cast directly from 16
      // bytes to 8 bytes after division.
      ((current_epoch - start_epoch).total_nanoseconds() / 1000) as i64
    };

    for (idx, (ps, id)) in ps_extraction.iter().enumerate() {
      let (t_scene, id) = scene.scene.get_micro_frame_entity(*id).unwrap();
      let t = CartesianState::frame_data(&cartesian_state_cache, scene_id, id).unwrap_or(t_scene);
      let r_helio_au = t.position.length();
      force_emitters.push(
        vulkan_device.cmd_allocate_transient_emitter_for_particle_system(
          cmd,
          compute_signal_value,
          (t.position, t.rotation),
          (ps.framerel_pos_km, ps.framerel_rot),
        )?,
      );
      particle_executions[idx].r_helio_au = r_helio_au;
      let gt = scene.scene.global_transform_f64(id).unwrap();
      particle_executions[idx].global_pos_f64 = gt.position.into();
      particle_executions[idx].global_rot = gt.rotation.0.into();
    }

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel: Extract all cartesian states which are not in the cache and insert them
    // there. If present, then use cached value.
    // ------------------------------------------------------------------------------------
    let insert_into_cache = |e_id: EntityId,
                             t: TransformComponent,
                             iau_rot: Option<BodyRotationalModel>,
                             ap: AlmanacPlanet| {
      let key = SceneEntityId::new(scene_id, e_id);
      if !cartesian_state_cache.contains_key(&key) {
        let frame_id = scene.scene.get_parent(e_id).unwrap();
        let frame_transform =
          scene.scene.with_component(frame_id, |t: &TransformComponent| *t).unwrap();
        debug_assert_eq!(frame_transform.rotation, Quat::identity());
        debug_assert_eq!(frame_transform.scale, Vec3f32::one());
        let frame_key = SceneEntityId::new(scene_id, frame_id);
        assert!(
          scene.scene.with_component(frame_id, |_: &ReferenceFrameComponent| ()).is_some(),
          "Scene integrity violation: Comet/Planet entity should have as direct parent a reference frame component"
        );
        assert!(
          scene.scene.get_parent(frame_id) == Some(scene.root_entity),
          "Scene integrity violation: Frame entity of type micro should be child of root"
        );
        cartesian_state_cache.insert(
          key,
          CartesianState::new_comet(t, ap, iau_rot, frame_id, frame_transform),
        );
        cartesian_state_cache.insert(
          frame_key,
          CartesianState::new_frame(frame_id, frame_transform),
        );
      }
    };

    // insert into cache if absent all active comets
    scene.scene.query4(
      |e_id,
       t: &TransformComponent,
       _m: &CometMarkerComponent,
       ap: &AlmanacPlanet,
       iau_rot: &BodyRotationalModel| insert_into_cache(e_id, *t, Some(*iau_rot), *ap),
    );

    // insert into cache if absent all active planets (earth)
    scene.scene.query3(
      |e_id, t: &TransformComponent, _m: &PlanetMarkerComponent, ap: &AlmanacPlanet| {
        insert_into_cache(e_id, *t, None, *ap)
      },
    );

    // ------------------------------------------------------------------------------------
    // Fixed Update Loop
    // ------------------------------------------------------------------------------------
    let fixed_dt_scaled_s = utils::time_micro_to_seconds(scaled_fixed_dt_us);

    // Spiral of death prevention: set a max number of physics execution steps
    const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 10;
    let mut steps_executed = 0;

    if scaled_fixed_dt_us > 0 {
      while time_mgr.consume_fixed_step(scaled_fixed_dt_us) {
        use aethervk_oshal_rlib::os::time::us_to_300ths_rounded;

        // spiral of death resolution
        if steps_executed > MAX_PHYSICS_STEPS_PER_FRAME {
          oshal::log!(
            "Physics is falling behind! Dropping accumulated time to avoid spiral of death"
          );
          // Drop accumulated steps but maintain the remainder to avoid stutters
          let mut time_state = time_mgr.state.write();
          time_state.scaled_accumulator %= scaled_fixed_dt_us;

          break;
        }

        steps_executed += 1;

        // ------------------------------------------------------------------------------------
        // Particle Systems Vulkan Shader Recording (Submission inserted after the fixed accumulator
        // loop)
        // ------------------------------------------------------------------------------------
        // -- all emissions --
        let mut skip_bind = false;
        for (idx, (ps, id)) in ps_extraction.iter().enumerate() {
          if particle_executions[idx].dead == 0 {
            // some emission constants which may be moved if exposed as parameters
            let mean_intra_grains_distance_mm =
              structs::particle_constants::MEAN_INTRA_GRAINS_DISTANCE_MM;
            let min_cumulated_mass_g = structs::particle_constants::MIN_CUMULATED_MASS_G;

            let push_constants = utils::new_particles_emit(
              vulkan_device,
              id.as_ffi(),
              &ps.emission_params,
              mean_intra_grains_distance_mm,
              min_cumulated_mass_g,
              particle_executions[idx].r_helio_au,
              (-1.0 * DVec3::from_array(particle_executions[idx].global_pos_f64))
                .normalize()
                .to_f32(),
              sim_speed.scaled_from_unscaled(ps.last_emission),
              from_start_epoch_scaled_us,
            );

            if let Ok(last_emission_unscaled_us) = vulkan_device.cmd_particle_system_emission(
              cmd,
              ps.last_emission,
              now_unscaled_us,
              &push_constants,
              skip_bind,
            ) {
              particle_executions[idx].last_emission_unscaled_us = last_emission_unscaled_us;
              skip_bind = true;
            } else {
              particle_executions[idx].dead = 1;
            }
          }
        }
        skip_bind = false;
        // -- barrier --
        vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
        // -- all integrate p1_p2 --
        for (idx, (_, id)) in ps_extraction.iter().enumerate() {
          if particle_executions[idx].dead == 0 {
            let push_constants =
              utils::integrate_particles_p1_p2_new(vulkan_device, id.as_ffi(), fixed_dt_scaled_s);
            if let Ok(()) = vulkan_device.cmd_particle_system_velocity_vertlet_kick(
              cmd,
              &push_constants,
              skip_bind,
            ) {
              skip_bind = true;
            } else {
              particle_executions[idx].dead = 1;
            }
          }
        }
        skip_bind = false;
        // -- barrier --
        vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
        // -- all apply emitters --
        for (idx, (_, id)) in ps_extraction.iter().enumerate() {
          if particle_executions[idx].dead == 0 {
            let emitter_bda = force_emitters[idx].2;
            let push_constants =
              utils::apply_emitters_direct_new(vulkan_device, id.as_ffi(), emitter_bda, 1);
            if let Ok(()) =
              vulkan_device.cmd_particle_system_next_forces(cmd, &push_constants, skip_bind)
            {
              skip_bind = true;
            } else {
              particle_executions[idx].dead = 1;
            }
          }
        }
        skip_bind = false;
        // -- barrier --
        vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
        // -- all integrate p4 p5 --
        for (idx, (_, id)) in ps_extraction.iter().enumerate() {
          if particle_executions[idx].dead == 0 {
            let push_constants =
              utils::integrate_particles_p4_5_new(vulkan_device, id.as_ffi(), fixed_dt_scaled_s);
            if let Ok(()) = vulkan_device.cmd_particle_system_velocity_vertlet_correction(
              cmd,
              &push_constants,
              skip_bind,
            ) {
              skip_bind = true;
            } else {
              particle_executions[idx].dead = 1;
            }
          }
        }
        skip_bind = false;
        // -- barrier --
        vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
        // -- all compact --
        let mut has_compacted = false;
        for (idx, (ps, id)) in ps_extraction.iter().enumerate() {
          if particle_executions[idx].dead == 0 {
            let ttl_300ths = us_to_300ths_rounded(ps.ttl_us);
            let push_constants = utils::new_particles_compact(
              vulkan_device,
              id.as_ffi(),
              now_scaled_300ths,
              ttl_300ths,
            );
            if let Ok(last_compaction) = vulkan_device.cmd_particle_system_compaction(
              cmd,
              ps.last_compaction,
              now_unscaled_us,
              &push_constants,
              skip_bind,
            ) {
              has_compacted = true;
              skip_bind = true;
              particle_executions[idx].last_compaction_unscaled_us = last_compaction;
              particle_executions[idx].did_compact = if last_compaction != ps.last_compaction {
                1
              } else {
                0
              };
            } else {
              particle_executions[idx].dead = 1;
            }
          }
        }
        if has_compacted {
          skip_bind = false;
          // -- barrier --
          vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
          // -- all compact reset (only if compact) --
          for (idx, (_, id)) in ps_extraction.iter().enumerate() {
            if particle_executions[idx].dead == 0 && particle_executions[idx].did_compact == 1 {
              let push_constants = utils::new_particles_compact_reset(vulkan_device, id.as_ffi());
              if let Ok(()) =
                vulkan_device.cmd_particle_system_compaction_reset(cmd, &push_constants, skip_bind)
              {
                skip_bind = true;
              } else {
                particle_executions[idx].dead = 1;
              }
            }
          }
        }
        // -- barrier for next iteration --
        if time_mgr.has_ready_step(scaled_fixed_dt_us) {
          vulkan_device.cmd_dispatch_global_memory_barrier(cmd)?;
        }
      } // end of accumulator fixed update loop
    }

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel for Comet Cartesian state update.
    // Note: Outside of the fixed update accumulator, still inside physics update.
    // ------------------------------------------------------------------------------------
    // accumulate frame updates into a vec separately from the dashmap cause iter_mut locks shards
    // Note: rotation and scale stay fixed at identity, therefore we track only positions relative
    // to root
    let mut frame_updates = alloc::vec::Vec::<(EntityId, Vec3f32)>::with_capacity(32);
    cartesian_state_cache.iter_mut().for_each(
      |mut state: dashmap::mapref::multiple::RefMutMulti<'_, SceneEntityId, CartesianState>| {
        // skip frame updates
        const AU_TO_KM: f64 = 149_597_870.7_f64;
        let micro_frame_pos_km = state.parent_frame_transform.position.to_f64() * AU_TO_KM;
        let parent_id = state.parent_frame;
        if let Some(ref mut body_state) = state.comet_state {
          match body_state.almanac_planet.step(
            current_epoch,
            almanac,
            body_state.body_rotational_model.as_ref(),
          ) {
            Ok((global_dpos, global_rot)) => {
              // - take microframe position and comet new position. we are assuming micro is child of
              //   root here, ensured in assert in cache insertion, compute distance body to frame in
              //   world space
              const KM_TO_AU: f64 = 6.6845871226706e-9_f64;
              const THRESHOLD_KM: f64 = 0.1 * AU_TO_KM;
              let diff_km = global_dpos - micro_frame_pos_km;
              let distance_km = diff_km.length();

              if distance_km > THRESHOLD_KM {
                // --- FRAME SHIFT ---
                // The comet drifted too far. Move the Micro Frame to the Comet's exact AU location
                // works because, due to assertion, we know that micro frame is child of root
                // Note: position of the micro frame is in AU
                frame_updates.push((parent_id, (global_dpos * KM_TO_AU).to_f32()));
                // now update the comet so that its rotation relative to its parent, which in this
                // case is equal to relative to root, is updated. Position relative to frame is reset
                // to zero
                body_state.transform.position = Vec3f32::zero();
              } else {
                // --- NORMAL DRIFT ---
                // The frame stays still, just update the Comet's local offset
                body_state.transform.position = diff_km.to_f32();
              }

              body_state.transform.rotation = global_rot;
              debug_assert_eq!(body_state.transform.scale, Vec3f32::one());
            }
            Err(e) => {
              oshal::log!("Error while SPICE update: {e}");
              emit_breadcrumb(3, &e.to_string());
            }
          }
        }
      },
    );
    // drain all frame updates into the cache
    for (frame_id, position) in &frame_updates {
      if let Some(mut state) =
        cartesian_state_cache.get_mut(&SceneEntityId::new(scene_id, *frame_id))
      {
        state.parent_frame_transform.position = *position;
      }
    }

    // ------------------------------------------------------------------------------------
    // Particle Systems Vulkan Shader: Change frame of reference after comet motion
    // ------------------------------------------------------------------------------------
    let mut skip_bind = false;
    for (idx, (_, id)) in ps_extraction.iter().enumerate() {
      if particle_executions[idx].dead == 0 {
        let gt = utils::global_transform_f64_with_overrides(&scene.scene, *id, |e_id| {
          let key = SceneEntityId::new(scene_id, e_id);
          // Intercept node transforms dynamically using the async cache
          // DashMap handles lock-striping internally, so point lookups are perfectly safe here
          if let Some(cached_state) = cartesian_state_cache.get(&key) {
            // if the entity requested represents a comet/planet
            if let Some(ref comet) = cached_state.comet_state {
              return Some(HighResTransformComponent::from_transform(&comet.transform));
            }
            // if the entity represents the reference frame
            if cached_state.parent_frame == e_id {
              return Some(HighResTransformComponent::from_transform(
                &cached_state.parent_frame_transform,
              ));
            }
          }

          // not in cache -> Fallback to ECS scene transform
          None
        })
        .unwrap();
        let transform_changed = particle_executions[idx].global_pos_f64
          != Into::<[f64; 3]>::into(gt.position)
          || particle_executions[idx].global_rot == Into::<[f32; 4]>::into(gt.rotation.0);
        if transform_changed {
          if !skip_bind {
            vulkan_device.cmd_dispatch_global_memory_barrier(cmd);
          }

          // -- calculate compensation data
          let delta_pos_m = {
            // compute delta km in f64, divide by 1000 then cast to f32
            let old_pos_dvec = Into::<DVec3>::into(particle_executions[idx].global_pos_f64);
            let diff_dvec_km: DVec3 = old_pos_dvec - gt.position;
            (diff_dvec_km * 1000.0).to_f32()
          };
          let delta_rot = {
            // R_new^-1
            let q_new_inv = gt.rotation.conjugate();
            let q_old = Quat(Vec4f32::from_components(
              particle_executions[idx].global_rot[0],
              particle_executions[idx].global_rot[1],
              particle_executions[idx].global_rot[2],
              particle_executions[idx].global_rot[3],
            ));
            // ΔR = R_new^-1 * R_old
            q_new_inv * q_old
          };

          let push_constants = utils::new_particles_offset_particles_push_constants(
            vulkan_device,
            id.as_ffi(),
            delta_pos_m,
            delta_rot,
          );
          if let Ok(_) =
            vulkan_device.cmd_particle_system_offset_particles(cmd, &push_constants, skip_bind)
          {
            skip_bind = true;
          } else {
            particle_executions[idx].dead = 1;
          }
        }
      }
    }

    // ------------------------------------------------------------------------------------
    // Particle System Vulkan Command Buffer Submission
    // ------------------------------------------------------------------------------------
    let (compute_semaphore, compute_signal_value) = cmd_scope.submit()?;

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel: Commit Comet Cartesian state update from dashmap to scene
    // ------------------------------------------------------------------------------------
    if do_cross_sync {
      // 1. upgrade to a write lock to start applying updates
      let mut scene_write = parking_lot::RwLockUpgradableReadGuard::upgrade(scene);
      let mut start_time_unscaled_us = get_monotonic_time();
      for (idx, kv_ref) in cartesian_state_cache.iter().enumerate() {
        let key = kv_ref.key();
        let state = kv_ref.value();
        let entity_id = EntityId::from_ffi(key.entity_id);
        if let Some(ref body_state) = state.comet_state {
          // comet/planet, update its trasform
          scene_write
            .scene
            .with_component_mut(entity_id, |t: &mut TransformComponent| {
              *t = body_state.transform;
            })
            .unwrap();
        } else {
          // reference frame, update its transform
          scene_write
            .scene
            .with_component_mut(entity_id, |t: &mut TransformComponent| {
              *t = state.parent_frame_transform;
            })
            .unwrap();
        }

        // Now mark the entity as changed
        utils::mark_component_changed::<TransformComponent>(&scene_write, entity_id);

        // check every N iterations, eg 32
        if idx % 32 == 0 {
          let now_unscaled_us = get_monotonic_time();
          if now_unscaled_us - start_time_unscaled_us >= 2000_i64 {
            // Yield lock
            scene = parking_lot::RwLockWriteGuard::downgrade_to_upgradable(scene_write);
            core::hint::spin_loop();
            scene_write = parking_lot::RwLockUpgradableReadGuard::upgrade(scene);
            start_time_unscaled_us = get_monotonic_time();
          }
        }
      }
      // write Lock automatically dropped/downgraded here
      scene = parking_lot::RwLockWriteGuard::downgrade_to_upgradable(scene_write);

      // TODO: if more then THRESHOLD μs have elapsed, then empty the cache
      // - either keep it as &DashMap and remove entries one by one (maybe on a tasklet)
      // - or swap for &mut DashMap and use mem::replace
    }

    Some(PhysicsDeviceSelfSync::new(
      compute_semaphore,
      compute_signal_value,
    ))
  } else {
    None
  };

  Ok(SimulationTickOutput {
    pending_particle_acquire: latest_physics_sync.as_ref().map(|s| s.timeline_value),
    latest_physics_sync,
  })
}

/// 2/3 Step of a simulation tick: Non-Physics Update
///
/// update phase of the `execute_simulation_tick`, extracted into its own function so that we can
/// execute its logic in the `!physics_done` branch
/// Note: we assume entities driven in fixed update are not also driven by some update rules
/// which do not involve compute queue, therefore we won't go through cross sync window to perform
/// an update
///
/// Time manager should have been already ticked (either by simulation step or by logic thread
/// physics tasklet)
fn execute_simulation_tick_update_phase(
  scene: parking_lot::lock_api::RwLockUpgradableReadGuard<parking_lot::RawRwLock, SceneContext>,
  time_mgr: &oshal::os::time::v2::TimeManager,
) {
  let dt_unscaled_s = utils::time_micro_to_seconds(time_mgr.state.read().unscaled_delta);

  // -- Update Transform Animations --
  // cannot apply an animation to components animated by physics, hence almanac planet and particle
  // system. (checking almanac planet only at runtime, while debug_assert on particle system)
  let mut animation_step = alloc::vec::Vec::with_capacity(16);
  let mut remove_animation = alloc::vec::Vec::with_capacity(16);
  scene.scene.query1_mut(|e_id, anim: &mut TransformAnimationComponent| {
    // assert it's not a comet or planet. If it is, ignore this component
    if utils::is_entity_physics_driven(&scene.scene, e_id) {
      remove_animation.push(e_id);
    } else if !anim.is_finished {
      anim.elapsed += dt_unscaled_s;
      let smooth_t = utils::hermite_smoothstep(anim.elapsed / anim.duration);
      let new_pos_dvec = DVec3::lerp(anim.start_pos, anim.target_pos, smooth_t as f64);
      let new_rot = Quat::slerp(anim.start_rot, anim.target_rot, smooth_t);
      if anim.elapsed > anim.duration {
        anim.is_finished = true;
      }
      animation_step.push((e_id, new_pos_dvec, new_rot));
    } else {
      remove_animation.push(e_id);
    }
  });
  // remove all Animation components whose animation finished or is invalid
  // moving consumption of vec here
  for id in remove_animation {
    scene.scene.remove_component::<TransformAnimationComponent>(id).unwrap();
  }

  // moving consumption of vec here
  for (id, new_pos_dvec, new_rot) in animation_step {
    let (did_standard_res, did_high_res) = scene.scene.with_component_mut_or(
      id,
      |t: &mut TransformComponent| {
        t.position = new_pos_dvec.to_f32();
        t.rotation = new_rot;
        true
      },
      |hrt: &mut HighResTransformComponent| {
        hrt.position = new_pos_dvec;
        hrt.rotation = new_rot;
        true
      },
    );
    if let Some(true) = did_standard_res {
      utils::mark_component_changed::<TransformComponent>(&scene, id);
    }
    if let Some(true) = did_high_res {
      utils::mark_component_changed::<HighResTransformComponent>(&scene, id);
    }
  }
}

/// 3/3 Step of a simulation tick: Clear all entities marked as changed
///
/// Clear all entities marked as changed and call the [`crate::simulation_api::SIMULATION_CALLBACK`]
/// so that C# side can be notified of all entities changed for a given scene in bulk and update its
/// view models
fn execute_simulation_tick_clear_changed_entities_phase(
  scene_arc: &alloc::sync::Arc<parking_lot::RwLock<SceneContext>>,
  scene_id: u64,
  thread_pool: &oshal::os::pool::ThreadPool,
) {
  let r_lock = crate::simulation_api::SIMULATION_CALLBACK.read();
  if r_lock.is_none() {
    return;
  }
  let scene = scene_arc.read();

  // - accumulate all entities changes into a vector
  // (external entity id, component id, component data)
  let mut changes_to_stream =
    alloc::vec::Vec::<(u64, u64, utils::AlignedBoxedBytes)>::with_capacity(64);
  for (ext_id, components) in scene.changed_entities.read().iter() {
    let entity_id = EntityId::from_ffi(*ext_id);
    for comp_id in components.iter() {
      if *comp_id == ComponentForeignId::HighResTransform.as_u64() {
        // C# is not aware of the scene hierarchy, so we must emit the *world-space* global
        // transform. `global_transform_f64` accumulates parent transforms up the tree.
        if let Some(global_t) = scene.scene.global_transform_f64(entity_id) {
          let size = core::mem::size_of::<crate::scene::HighResTransformDTO>();
          let mut data = unsafe { utils::AlignedBoxedBytes::new_zeroed(size, 8) };
          // SAFETY: buffer is exactly `foreign_data_size()` bytes, 8-byte aligned.
          unsafe { global_t.write_foreign_bytes(data.ptr.as_ptr().cast()) };
          changes_to_stream.push((*ext_id, *comp_id, data));
        }
      } else {
        // All other ForeignSerializable components: serialize local component data as-is.
        let _ = scene.scene.with_component_by_id(entity_id, *comp_id, |dyn_comp| {
          let mut data =
            unsafe { utils::AlignedBoxedBytes::new_zeroed(dyn_comp.foreign_data_size(), 8) };
          // SAFETY: buffer is exactly `foreign_data_size()` bytes, 8-byte aligned.
          unsafe { dyn_comp.write_foreign_bytes(data.ptr.as_ptr().cast()) };
          changes_to_stream.push((*ext_id, *comp_id, data));
        });
      }
    }
  }
  drop(scene);

  // - pass this vector's ownership into a tasklet and let it acquire a readlock on the callback to
  //   notify C# side. Note: tasklet will wait for the current task to be finished before starting
  //   streaming changes
  let mut scene_write = scene_arc.write();
  let previous_update_tasklet = scene_write.entities_update_tasklet.take();
  
  // clear it now since we've already extracted what we need
  scene_write.changed_entities.write().clear();

  if let Ok(tasklet) = thread_pool.spawn_tasklet(None, move || {
    // wait for previous bulk update to finish
    if let Some(wait_handle) = previous_update_tasklet {
      wait_handle.wait();
    }

    // acquire function callback
    let r_lock = crate::simulation_api::SIMULATION_CALLBACK.read();
    if r_lock.is_none() {
      return;
    }

    let callback = unsafe { r_lock.unwrap_unchecked() };
    for (ext_id, comp_id, data) in changes_to_stream {
      unsafe { callback(scene_id, ext_id, comp_id, data.as_slice().as_ptr().cast()) }
    }
  }) {
    scene_write.entities_update_tasklet = Some(tasklet);
  } else {
    emit_breadcrumb(3, "Error: Couldn't spawn entities update tasklet");
  }
}

/// Utilities for logic thread module
mod utils {
  use super::*;
  use crate::gpu::{RenderDeviceHandle, RenderFrontend};
  use crate::gpu::{compute_push_constants::*, new_particles::MAX_CHUNKS};
  use crate::gpu_backends::vulkan::device::{Device, particles::PushConstantMutUnion};
  use crate::scene::ForeignSerializable;
  use crate::scene::particles::v2::{ParticleSystemEmitParams, emit_push_constants_from_params};
  use crate::simulation_api::structs::{RenderCommand, SimulationSceneData};
  use bytemuck::Zeroable;

  /// Factory function for [`NewParticlesEmitPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic
  pub fn new_particles_emit(
    vulkan_device: &Device,
    ps_id: u64,
    psep: &ParticleSystemEmitParams,
    mean_intra_grains_distance_mm: f32,
    min_cumulated_mass_g: f32,
    r_helio_au: f32,
    ps_to_sun_dir: Vec3f32,
    scaled_time_since_last_emission_us: timeus_t,
    scaled_time_since_start_epoch_us: timeus_t,
  ) -> NewParticlesEmitPushConstants {
    let mut res = emit_push_constants_from_params(
      &psep,
      mean_intra_grains_distance_mm,
      min_cumulated_mass_g,
      r_helio_au,
      ps_to_sun_dir,
      scaled_time_since_last_emission_us,
      scaled_time_since_start_epoch_us,
    );
    vulkan_device
      .complete_particle_push_constant(PushConstantMutUnion::NewParticlesEmit(&mut res), ps_id)
      .unwrap();

    res
  }

  /// Factory function for [`IntegrateParticlesP1P2NewPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic
  pub fn integrate_particles_p1_p2_new(
    vulkan_device: &Device,
    ps_id: u64,
    dt_s: f32,
  ) -> IntegrateParticlesP1P2NewPushConstants {
    let mut res = IntegrateParticlesP1P2NewPushConstants::zeroed();
    res.delta_time = dt_s;
    vulkan_device
      .complete_particle_push_constant(
        PushConstantMutUnion::IntegrateParticlesP1P2New(&mut res),
        ps_id,
      )
      .unwrap();

    res
  }

  /// Factory function for [`ApplyEmittersDirectNewPushConstants`]
  /// - Assumes that the particle system id is correct and exists, otherwise panic.
  /// - Assumes that the given emitter data is valid for the current compute timeline
  pub fn apply_emitters_direct_new(
    vulkan_device: &Device,
    ps_id: u64,
    emitter_bda: u64,
    emitter_count: u32,
  ) -> ApplyEmittersDirectNewPushConstants {
    let mut res = ApplyEmittersDirectNewPushConstants::zeroed();
    res.emitter_array = emitter_bda;
    res.emitter_count = emitter_count;
    vulkan_device
      .complete_particle_push_constant(
        PushConstantMutUnion::ApplyEmittersDirectNew(&mut res),
        ps_id,
      )
      .unwrap();

    res
  }

  /// Factory function for [`IntegrateParticlesP45NewPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic
  pub fn integrate_particles_p4_5_new(
    vulkan_device: &Device,
    ps_id: u64,
    dt_s: f32,
  ) -> IntegrateParticlesP45NewPushConstants {
    let mut res = IntegrateParticlesP45NewPushConstants::zeroed();
    res.delta_time = dt_s;
    vulkan_device
      .complete_particle_push_constant(
        PushConstantMutUnion::IntegrateParticlesP45New(&mut res),
        ps_id,
      )
      .unwrap();

    res
  }

  /// Factory function for [`NewParticlesCompactPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic
  /// uses as `max_chunks` the value [`MAX_CHUNKS`]
  pub fn new_particles_compact(
    vulkan_device: &Device,
    ps_id: u64,
    now_300ths: u32,
    ttl_300ths: u32,
  ) -> NewParticlesCompactPushConstants {
    let mut res = NewParticlesCompactPushConstants::zeroed();
    res.doomsday = ttl_300ths;
    res.now = now_300ths;
    res.max_chunks = MAX_CHUNKS as _;
    vulkan_device
      .complete_particle_push_constant(PushConstantMutUnion::NewParticlesCompact(&mut res), ps_id)
      .unwrap();

    res
  }

  /// Factory function for [`NewParticlesCompactResetPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic
  /// uses as `max_chunks` the value [`MAX_CHUNKS`]
  pub fn new_particles_compact_reset(
    vulkan_device: &Device,
    ps_id: u64,
  ) -> NewParticlesCompactResetPushConstants {
    let mut res = NewParticlesCompactResetPushConstants::zeroed();
    res.max_chunks = MAX_CHUNKS as _;
    vulkan_device
      .complete_particle_push_constant(
        PushConstantMutUnion::NewParticlesCompactReset(&mut res),
        ps_id,
      )
      .unwrap();

    res
  }

  /// Factor function for [`NewParticlesOffsetParticlesPushConstants`]
  /// Assumes that the particle system id is correct and exists, otherwise panic.
  /// - requires delta position in metres, delta rotation in radians
  pub fn new_particles_offset_particles_push_constants(
    vulkan_device: &Device,
    ps_id: u64,
    delta_pos_m: Vec3f32,
    delta_rot: Quat,
  ) -> NewParticlesOffsetParticlesPushConstants {
    let mut res = NewParticlesOffsetParticlesPushConstants::zeroed();
    res.delta_rot = delta_rot.0.into();
    res.delta_pos = [delta_pos_m.x(), delta_pos_m.y(), delta_pos_m.z(), 0.0];
    vulkan_device
      .complete_particle_push_constant(
        PushConstantMutUnion::NewParticlesOffsetParticlesPush(&mut res),
        ps_id,
      )
      .unwrap();

    res
  }

  /// Time Boundary,Step Size (Precision Loss),What it means for your data
  /// ----------------------------------------------------------------------------------------------
  /// < 16 seconds,<1μs,Perfect. You can represent every single microsecond accurately.
  /// > 16 seconds,≈1.9μs,Microsecond loss. The gap between floats becomes larger than 1μs. Consecutive microseconds round to the same f32 value.
  /// "> 8,192 sec (2.2 hours)",≈1 ms,Millisecond loss. You can no longer distinguish sub-millisecond differences.
  /// "> 131,072 sec (36.4 hours)",≈15.6 ms,"UI/Physics jitter. At 60fps (16.6ms per frame), your time steps are now larger than a video frame. Physics engines using f32 will glitch."
  /// "> 8,388,608 sec (97 days)",1 second,Total fractional loss. The f32 can no longer hold fractions. 97 days+0.5 seconds will simply round to 97 days.
  pub fn time_micro_to_seconds(time_us: timeus_t) -> f32 {
    (time_us as f64 / 1_000_000.0) as f32
  }

  /// Computes the global f64 transform of an entity by walking up the hierarchy,
  /// prioritizing temporary local transforms provided by an override closure.
  ///
  /// If `override_fn` returns `None` for a given `EntityId`, it falls back
  /// to the actual state stored inside the `Scene` ECS.
  pub fn global_transform_f64_with_overrides<F>(
    scene: &crate::scene::Scene,
    entity_id: EntityId,
    mut override_fn: F,
  ) -> Option<HighResTransformComponent>
  where
    F: FnMut(EntityId) -> Option<HighResTransformComponent>,
  {
    // Helper to get the local transform, checking the cache overrides first, then ECS
    let mut get_local_transform = |e_id: EntityId| -> Option<HighResTransformComponent> {
      if let Some(t_over) = override_fn(e_id) {
        return Some(t_over);
      }

      scene.with_component(e_id, |c: &HighResTransformComponent| *c).or_else(|| {
        scene.with_component(e_id, |c: &TransformComponent| {
          HighResTransformComponent::from_transform(c)
        })
      })
    };

    // 1. Read the initial entity's local transform
    let initial = get_local_transform(entity_id)?;

    let mut acc_pos = initial.position;
    let mut acc_rot = initial.rotation;
    let mut acc_scale = initial.scale;
    let mut current_entity = entity_id;
    let mut depth = 0;

    // 2. Traverse up the hierarchy
    loop {
      if depth > 128 {
        break;
      }
      depth += 1;

      if let Some(parent_id) = scene.get_parent(current_entity) {
        // Read parent transform, applying async overrides if available
        if let Some(parent_transform) = get_local_transform(parent_id) {
          // Frame scales are assumed structural/static and not mutated
          // by the physics async pass, so we safely read them from the ECS.
          let mut frame_scale = 1.0_f32;
          let _ = scene.with_component(parent_id, |c: &ReferenceFrameComponent| {
            frame_scale = c.scale;
          });

          let scaled_parent_scale = parent_transform.scale * frame_scale;

          // Combine logic with f64 position retention: parent_pos + parent_rot * (parent_scale * child_pos)
          let rotated = parent_transform.rotation.rotate_vector(Vec3f32::from_components(
            (scaled_parent_scale.x() as f64 * acc_pos.x()) as f32,
            (scaled_parent_scale.y() as f64 * acc_pos.y()) as f32,
            (scaled_parent_scale.z() as f64 * acc_pos.z()) as f32,
          ));

          acc_pos = parent_transform.position + rotated.to_f64();
          acc_rot = parent_transform.rotation * acc_rot;
          acc_scale = scaled_parent_scale * acc_scale;
        }
        current_entity = parent_id;
      } else {
        break;
      }
    }

    Some(HighResTransformComponent {
      position: acc_pos,
      rotation: acc_rot,
      scale: acc_scale,
    })
  }

  /// Function to assess whether an entity in the ECS scene is driven by physics or not. If the
  /// latter proposition is true, then we cannot change its cartesian state during an update
  /// function, cause it would overwrite the `cartesian_state_cache` cached position and rotation
  pub fn is_entity_physics_driven(scene: &crate::scene::Scene, entity_id: EntityId) -> bool {
    scene.has_component::<AlmanacPlanet>(entity_id).into()
      || scene.has_component::<ParticleSystemComponent>(entity_id).into()
  }

  pub use crate::scene::animation::hermite_smoothstep;

  use alloc::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
  use core::ptr::NonNull;
  use core::slice;

  /// Custom wrapper to dynamically allocate `Drop` managed memory with a given alignment
  pub struct AlignedBoxedBytes {
    pub ptr: NonNull<u8>,
    pub len: usize,
    pub align: usize,
  }

  unsafe impl Sync for AlignedBoxedBytes {}
  unsafe impl Send for AlignedBoxedBytes {}

  impl AlignedBoxedBytes {
    /// SAFETY: `align` must be a power of two and `len` must be non-zero for a real allocation.
    pub unsafe fn new_zeroed(len: usize, align: usize) -> Self {
      debug_assert!(
        align != 0 && (align & (align - 1)) == 0,
        "align must be a power of two"
      );

      if len == 0 {
        // dangling pointer with alignment 8
        return Self {
          ptr: NonNull::new(8 as *mut u8).unwrap(),
          len: 0,
          align: 0,
        };
      }

      // Force a 8-byte alignment on a layout sized exactly to `len`
      let layout = Layout::from_size_align(len, align).expect("Invalid layout");

      let ptr = unsafe { alloc_zeroed(layout) };
      if ptr.is_null() {
        // happy crash
        handle_alloc_error(layout);
      }

      Self {
        ptr: NonNull::new(ptr).unwrap(),
        len,
        align,
      }
    }

    pub fn as_slice(&self) -> &[u8] {
      unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
      unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
  }

  impl Drop for AlignedBoxedBytes {
    fn drop(&mut self) {
      if self.len != 0 {
        let layout = Layout::from_size_align(self.len, self.align).unwrap();
        unsafe {
          dealloc(self.ptr.as_ptr(), layout);
        }
      }
    }
  }

  impl core::ops::Deref for AlignedBoxedBytes {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
      self.as_slice()
    }
  }

  impl core::ops::DerefMut for AlignedBoxedBytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
      self.as_mut_slice()
    }
  }

  /// Utility function to mark a component of a given entity as changed starting from its internal id
  pub fn mark_component_changed<T: ForeignSerializable>(scene: &SceneContext, entity_id: EntityId) {
    scene.mark_component_changed(EntityId::as_ffi(&entity_id), T::COMPONENT_ID);
  }

  /// Utility to quickly mark all [`ForeignSerializable`] implementations (camera component,
  /// cursor component, transform component, hires tranform component) as changed.
  pub fn mark_all_serializable_as_changed(scene: &SceneContext) {
    scene.scene.for_each_entity(|id| {
      if Into::<bool>::into(scene.scene.has_component::<TransformComponent>(id)) {
        scene.mark_component_changed(EntityId::as_ffi(&id), TransformComponent::COMPONENT_ID);
      }
      if Into::<bool>::into(scene.scene.has_component::<HighResTransformComponent>(id)) {
        scene.mark_component_changed(
          EntityId::as_ffi(&id),
          HighResTransformComponent::COMPONENT_ID,
        );
      }
      if Into::<bool>::into(scene.scene.has_component::<CameraComponent>(id)) {
        scene.mark_component_changed(EntityId::as_ffi(&id), CameraComponent::COMPONENT_ID);
      }
    });
  }

  /// Utility function to wait and consume self synchronization and do something when task is
  /// consumed
  /// Render Frontend and device handle should point to a vulkan device
  /// Return
  /// - `None` if there were problems, silently swalloed,
  /// - `None` if task wasn't finished within established deadline
  /// - `Some(R)` if task was finished and callback returned an Ok
  pub fn self_sync_do_if_done<R>(
    scenes: &SimulationSceneData,
    scene_id: u64,
    render_frontend: RenderFrontend,
    device_handle: RenderDeviceHandle,
    render_tx: &mpsc::Sender<RenderCommand>,
    now: timeus_t,
    elapsed: timeus_t,
    f: impl FnOnce(&Device, &mut SceneContext, &mpsc::Sender<RenderCommand>) -> R,
  ) -> Option<R> {
    use core::sync::atomic::Ordering;
    // Note: Check simulattion speed after checking `physics_done`, so that we can process
    // remaining GPU tasks and then pause the simulation
    if !scenes.time_managers.contains_key(&scene_id) {
      return None;
    }

    if let Some(scene_lock) = scenes.get(&scene_id)
      && scene_lock
        .read()
        .active_physics_task
        .compare_exchange_weak(true, false, Ordering::Acquire, Ordering::Relaxed)
        .unwrap_or(false)
    {
      let scene = scene_lock.upgradable_read();
      // acquire a read lock on the timeline manager in the scene just to prevent
      // execution of a simulation step from someone else
      // SAFETY: when scene is created, time_manager is associated to it
      let _time_mgr = unsafe { scenes.time_managers.get(&scene_id).unwrap_unchecked() };

      render_frontend
        .with_device(device_handle, |dyn_device| {
          let vulkan_device: &Device = dyn_device.as_any().downcast_ref().unwrap();
          let mut scene_write = parking_lot::RwLockUpgradableReadGuard::upgrade(scene);
          // SAFETY: `latest_physics_sync` written by `execute_simulation_tick`, which was
          // executed if `active_physics_task` is `true`
          let is_done = unsafe { scene_write.latest_physics_sync.as_mut().unwrap_unchecked() }
            .try_wait(&vulkan_device.device, now, elapsed);
          if is_done {
            // Self Sync: destroy consumed synchronization primitives
            let _ = scene_write.latest_physics_sync.take();
            Ok(Some(f(vulkan_device, &mut scene_write, render_tx)))
          } else {
            scene_write.active_physics_task.store(true, Ordering::Release);
            Ok(None)
          }
        })
        .unwrap_or(None)
    } else {
      None
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::simulation_api::structs::LogicCommand;

  #[test]
  fn test_update_trajectory_command_exists() {
    let _cmd = LogicCommand::UpdateTrajectoryForSpk {
      task_id: 1,
      scene_id: 1,
      entity_id: 1,
      spk_id: 399,
      start_epoch_tai_sec: 0.0,
      end_epoch_tai_sec: 100.0,
      sample_step_days: 1.0,
    };
    // If it compiles, the variant exists and fields are correct.
    assert!(true);
  }
}
