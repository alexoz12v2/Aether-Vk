//! logic_thread module.

use super::structs::{
  LogicCommand, LogicThreadContext, LogicWorkload, SceneContext, SimulationTaskResult,
  SyncParticleReleaseFeedback,
};
use crate::{
  gpu::{RenderDevice, WeakRenderFrontendExt, vulkan},
  scene::{
    AlmanacPlanet, BodyRotationalModel, CameraComponent, CometMarkerComponent, CursorComponent,
    EntityId, FollowingComponent, HighResTransformComponent, PlanetMarkerComponent,
    ReferenceFrameComponent, StaticMeshComponent, TransformComponent,
    camera::{QuatToEulerAngles, SceneCameraExt},
  },
  simulation::almanac::AlmanacPackedData,
  simulation_api::{
    emit_breadcrumb,
    structs::{CartesianState, PhysicsDeviceSelfSync},
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
    time::timeus_t,
  },
};
use alloc::{boxed::Box, string::ToString};
use parking_lot::RwLockReadGuard;
use thingbuf::mpsc;

pub fn is_logic_command_async(cmd: &LogicCommand) -> bool {
  match cmd {
    LogicCommand::ImportModel { .. }
    | LogicCommand::LoadAlmanac { .. }
    | LogicCommand::UnloadAlmanac { .. }
    | LogicCommand::LoadCometSpk { .. }
    | LogicCommand::SpawnModelInstance { .. }
    | LogicCommand::RaycastNdc { .. }
    | LogicCommand::UpdateTrajectoryForSpk { .. }
    | LogicCommand::Raycast { .. } => true,
    _ => false,
  }
}

/// SAFETY: should be called from an async arm with a non zero task id. Debug will crash it
unsafe fn logic_command_async_get_task_id(cmd: &LogicCommand) -> u64 {
  debug_assert!(is_logic_command_async(cmd));
  match cmd {
    LogicCommand::ImportModel { task_id, .. }
    | LogicCommand::LoadAlmanac { task_id, .. }
    | LogicCommand::UnloadAlmanac { task_id, .. }
    | LogicCommand::LoadCometSpk { task_id, .. }
    | LogicCommand::SpawnModelInstance { task_id, .. }
    | LogicCommand::RaycastNdc { task_id, .. }
    | LogicCommand::UpdateTrajectoryForSpk { task_id, .. }
    | LogicCommand::Raycast { task_id, .. } => {
      debug_assert_ne!(*task_id, 0);
      *task_id
    }
    _ => panic!("unreachable"),
  }
}

fn logic_command_desc(cmd: &LogicCommand) -> alloc::string::String {
  match cmd {
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
    LogicCommand::FollowEntity { .. } => "Follow Entity".to_string(),
    LogicCommand::UnfollowEntity { .. } => "Unfollow Entity".to_string(),
    LogicCommand::SetEntityVisibility {
      entity, visible, ..
    } => {
      alloc::format!("Set visibility for entity {} to {}", entity, visible)
    }

    // Scene Playback Commands
    LogicCommand::PlaySceneToEnd { .. } => "Play Scene to End".to_string(),
    LogicCommand::PauseScene { .. } => "Pause Scene".to_string(),
    LogicCommand::PlayScene { .. } => "Play Scene".to_string(),
    LogicCommand::SnapshotScene { .. } => "Snapshot Scene".to_string(),
    LogicCommand::RestoreSnapshot { .. } => "Restore Snapshot".to_string(),

    // Data/Asset Commands
    LogicCommand::ImportModel { path, .. } => alloc::format!("Import model {}", path),
    LogicCommand::LoadAlmanac { path, .. } => alloc::format!("Load almanac {}", path),
    LogicCommand::UnloadAlmanac { path, .. } => alloc::format!("Unload almanac {}", path),
    LogicCommand::LoadCometSpk { spk_id, epoch, .. } => alloc::format!(
      "Load Ephemeris data for SPK ID: {} at epoch {}",
      spk_id,
      epoch
    ),
    LogicCommand::SpawnModelInstance { name, .. } => alloc::format!("Spawn instance {}", name),

    // Raycasting & Trajectory
    LogicCommand::RaycastNdc { .. } => "Raycast NDC".to_string(),
    LogicCommand::Raycast { .. } => "Raycast".to_string(),
    LogicCommand::UpdateTrajectoryForSpk { spk_id, .. } => {
      alloc::format!("Update trajectory for SPK {}", spk_id)
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
  scene_id: u64,
  target_frame_time: timeus_t,
  last_frame_start: timeus_t,
  last_render_ticks: alloc::vec::Vec<core::num::NonZero<u64>>,
}

impl PlayControl {
  fn new(scene_id: u64, target_frame_time: timeus_t) -> Self {
    Self {
      scene_id,
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

    loop {
      let mut core_logic = || -> bool {
        let mut processed_any = false;

        let scene_ids: alloc::vec::Vec<u64> = {
          let scenes = context.scenes.read();
          scenes.keys().copied().collect()
        };

        for scene_id in scene_ids {
          let pc = play_controls
            .entry(scene_id)
            .or_insert_with(|| PlayControl::new(scene_id, target_frame_time));
          let now = oshal::os::time::get_monotonic_time();
          let last = pc.last_frame_start;
          let elapsed = now.saturating_sub(last);

          if elapsed >= pc.target_frame_time {
            // Always reset the frame timer and submit a render frame at display
            // rate.  The physics TICK is gated separately — we only advance
            // simulation when the previous GPU compute step is done.
            pc.last_frame_start = now;

            // ── Physics tick (only when previous step is complete) ────────────
            let (physics_done, cross_sync_data) = 'physics_done_block: {
              let scenes = context.scenes.read();
              // Note: Check simulattion speed after checking `physics_done`, so that we can process
              // remaining GPU tasks and then pause the simulation
              if !scenes.time_managers.contains_key(&scene_id) {
                break 'physics_done_block (false, None);
              }
              if let Some(scene_ctx) = scenes.get(&scene_id) {
                use oshal::os::native::this_thread;
                use oshal::os::time::get_monotonic_time;
                let scene = scene_ctx.write();
                // Acquire/Release to make sure that writes to `latest_physics_sync` are done
                if scene.active_physics_task.load(core::sync::atomic::Ordering::Acquire) {
                  // acquire a read lock on the timeline manager in the scene just to prevent
                  // execution of a simulation step from someone else
                  // SAFETY: when scene is created, time_manager is associated to it
                  let _time_mgr = unsafe { scenes.time_managers.get(&scene_id).unwrap_unchecked() };

                  break 'physics_done_block context
                    .kernels
                    .0
                    .with_device(context.kernels.1, |dyn_device| {
                      let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
                        dyn_device.as_any().downcast_ref().unwrap();
                      // SAFETY: `latest_physics_sync` written by `execute_simulation_tick`, which was
                      // executed if `active_physics_task` is `true`
                      let is_done = unsafe { scene.latest_physics_sync.unwrap_unchecked() }
                        .try_wait(&vulkan_device.device, now, elapsed);
                      if is_done {
                        // Self Sync: destroy consumed synchronization primitives
                        let _ = scene.latest_physics_sync.take();

                        // Cross Sync: send a SyncParticleRelease command to render thread, register
                        // its Arc pointer for polling
                        let last_render_task =
                          scene.last_render_task.load(core::sync::atomic::Ordering::Acquire);
                        if last_render_task == 0 {
                          return Ok((is_done, None));
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
                          return Ok((is_done, None));
                        }

                        let gfx_fam = vulkan_device.get_graphics_queue().family_index;
                        let comp_queue = vulkan_device.get_compute_queue();
                        let comp_fam = comp_queue.family_index;

                        // 2. Request Graphics Release
                        use alloc::boxed::Box;
                        use bytemuck::Zeroable;
                        let release_feeback =
                          alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(0));
                        let feedback_data_ptr =
                          Box::into_raw(Box::new(SyncParticleReleaseFeedback::zeroed()));
                        context.render_tx.send(
                          crate::simulation_api::structs::RenderCommand::SyncParticleRelease {
                            feedback: release_feeback.clone(),
                            feedback_ptr: crate::simulation_api::structs::SendPtrMut(
                              feedback_data_ptr,
                            ),
                          },
                        );

                        Ok((is_done, Some((release_feeback, feedback_data_ptr))))
                      } else {
                        Ok((is_done, None))
                      }
                    })
                    .unwrap_or((false, None));
                }
              }

              (false, None)
            };

            let simulation_tick_result: EngineResult<SimulationTickOutput> = if physics_done {
              use core::ops::DerefMut;
              let dt_f32 = elapsed as f32 / 1_000_000.0;
              let scenes = context.scenes.read();
              // SAFETY: if `physics_done` then there should be time manager
              let mut time_mgr =
                unsafe { scenes.time_managers.get_mut(&scene_id).unwrap_unchecked() };
              // SAFETY: if `physics_done` then this scene exists
              let scene_arc = unsafe { scenes.get(&scene_id).as_ref().unwrap_unchecked() }.clone();
              let scene_lock = scene_arc.upgradable_read();
              // TODO: scale according to performance or not?
              const UNSCALED_FIXED_DELTA_US: timeus_t = oshal::os::time::timeus_milliseconds(16);
              // TODO return handle. Note: if err, do update too
              execute_simulation_tick(
                scene_lock,
                time_mgr.deref_mut(),
                UNSCALED_FIXED_DELTA_US,
                cross_sync_data,
              )
            } else {
              // TODO
              todo!()
            };

            let pending_particle_acquire =
              simulation_tick_result.map(|s| s.pending_particle_acquire).unwrap_or(None);

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
                scenes.get(&scene_id).unwrap().clone()
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

              render_frames.push(crate::simulation_api::structs::RenderFrame {
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
                          scene_id,
                          pe_handle,
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
            if let Some(sync) = scene_ctx_lock.read().latest_physics_sync.take() {
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
                  let start = get_monotonic_time();
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
    let task_id = unsafe { logic_command_async_get_task_id(&self.cmd) };

    let res = process_command_internal(self.cmd.clone(), &self.ctx);

    let cmd_desc = logic_command_desc(&self.cmd);

    let mut manager = self.ctx.task_manager.write();
    match res {
      Ok(result) => {
        manager.success_task(task_id, result);
      }
      Err(e) => {
        manager.fail_task(task_id, e.to_string());
        crate::simulation_api::emit_breadcrumb(3, &alloc::format!("Failed: {} - {}", cmd_desc, e));
      }
    }

    WorkloadStatus::Complete
  }
}

fn process_command_internal(
  command: LogicCommand,
  ctx: &alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<SimulationTaskResult> {
  match command {
    LogicCommand::SnapshotScene { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
        // 1. Write lock to ensure exclusive access while saving
        let mut scene_ctx = scene_ctx_guard.write();
        // 2. Device Synchronization: wait for active physics task to complete before taking the
        //    snapshot
        if let Some(task) = scene_ctx.active_physics_task.lock().take() {
          if let Err(e) = task.wait() {
            aethervk_oshal_rlib::log!("Physics tasklet failed: {:?}", e);
          }
        }

        let cloned_scene = (*scene_ctx.scene).clone();
        scene_ctx.scene_snapshot = Some(alloc::boxed::Box::new(cloned_scene));
      } else {
        emit_breadcrumb(2, "Failed to create scene snapshot");
      }
      Ok(SimulationTaskResult::None)
    }
    // TODO playtoend command!
    LogicCommand::PlaySceneToEnd { scene_id, speed } => {
      todo!()
    }

    // TODO now this command will be fused with StopScene, therefore
    //commenting out pieces as I see fit is perfectly fine
    LogicCommand::RestoreSnapshot { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
        // Upgrade to write lock to ensure thread-safety while modifying the Scene
        let mut scene_ctx = scene_ctx_guard.write();

        // Device synchronization: Wait for active physics task to complete safely
        if let Some(task) = scene_ctx.active_physics_task.lock().take() {
          if let Err(e) = task.wait() {
            aethervk_oshal_rlib::log!("Physics tasklet failed: {:?}", e);
          }
        }

        // Restore the snapshot
        if let Some(snapshot) = &scene_ctx.scene_snapshot {
          scene_ctx.scene = alloc::sync::Arc::new((**snapshot).clone());
        }

        // 1. Zero out velocities and angular velocities
        let _ = scene_ctx.scene.query1_res_mut(|_id, k: &mut crate::scene::KinematicComponent| {
          k.velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
          k.angular_velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
          Some(())
        });

        // Clear stale cached boundaries
        if let Some(ps_lock) = &scene_ctx.physics_scene {
          let mut ps = ps_lock.write();
          *ps = crate::physics::physics_scene::PhysicsScene::build_from_scene(
            scene_ctx.scene.as_ref(),
            0.016,
          );
        }

        // 2. Clear accumulated integration errors and reset time
        // let mut time_state = scene_ctx.time_state.write();
        // time_state.current_epoch = time_state.epoch_start;
        // time_state.st_seconds_elapsed = 0.0;
        // time_state.time_info.write().ut_discard_accumulator();

        // 3. Mark the Top-Level Acceleration Structure as dirty so it reconstructs bounding volumes
        scene_ctx
          .is_static_tlas_dirty
          .store(true, core::sync::atomic::Ordering::Relaxed);

        // 4. Force rendering pipeline updates for the restored entity transforms
        let ext_ids: alloc::vec::Vec<u64> = scene_ctx.entity_map.keys().copied().collect();
        use crate::scene::ForeignSerializable; // COMPONENT_ID
        for ext_id in ext_ids {
          scene_ctx.mark_component_changed(
            ext_id,
            <crate::scene::TransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
          );
          scene_ctx.mark_component_changed(
            ext_id,
            <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
          );
          // TODO: what are those number? check C# side and implement trait here
          scene_ctx.mark_component_changed(ext_id, 100);
          scene_ctx.mark_component_changed(ext_id, 101);
          scene_ctx.mark_component_changed(ext_id, 102);
        }
      }
      Ok(SimulationTaskResult::None)
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
            None => return Ok(SimulationTaskResult::None),
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
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::Shutdown => Ok(SimulationTaskResult::None),
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

      if let Some(ext_id) = scene_read
        .entity_map
        .iter()
        .find(|&(_, v)| *v == camera_entity)
        .map(|(k, _)| *k)
      {
        scene_read.mark_component_changed(
          ext_id,
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }
      Ok(SimulationTaskResult::None)
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
        if let Some(ext_id) = scene_read
          .entity_map
          .iter()
          .find(|&(_, v)| *v == camera_entity)
          .map(|(k, _)| *k)
        {
          scene_read.mark_component_changed(
            ext_id,
            <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
          );
        }
        return Ok(SimulationTaskResult::None);
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

      if let Some(ext_id) = scene_read
        .entity_map
        .iter()
        .find(|&(_, v)| *v == camera_entity)
        .map(|(k, _)| *k)
      {
        scene_read.mark_component_changed(
          ext_id,
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
        scene_read.mark_component_changed(
          ext_id,
          <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }
      Ok(SimulationTaskResult::None)
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
      if let Some(ext_id) = scene_read
        .entity_map
        .iter()
        .find(|&(_, v)| *v == camera_entity)
        .map(|(k, _)| *k)
      {
        scene_read.mark_component_changed(
          ext_id,
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }
      Ok(SimulationTaskResult::None)
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
      if let Some(ext_id) = scene_read
        .entity_map
        .iter()
        .find(|&(_, v)| *v == camera_entity)
        .map(|(k, _)| *k)
      {
        scene_read.mark_component_changed(
          ext_id,
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }
      Ok(SimulationTaskResult::None)
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

          if let Some(ext_id) =
            scene_read.entity_map.iter().find(|&(_, v)| *v == id).map(|(k, _)| *k)
          {
            scene_read.mark_component_changed(
              ext_id,
              <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
            );
          }

          Some(())
        })
        .map(|_| ())
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:MoveCursor | scene doesn't have cursor",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    // TODO analyse flow with pressing "F" and "0" and see if you can remove
    LogicCommand::SnapToEntity {
      snap_entity,
      target_entity,
      scene,
    } => {
      let mut scene_write = scene.write();
      // 'F' behavior: move cursor to target entity position, then position the camera dynamically
      let target_pos = {
        #[allow(deprecated)]
        scene_write.scene.global_transform(target_entity).map(|t| t.position).ok_or(
          EngineError::InvalidOperation(
            "logic_thread:SnapToEntity | target entity doesn't have TransformComponent",
          ),
        )?
      };

      // Move cursor to target entity world position.
      if let Some((cursor_id, _)) = scene_write
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        let _ =
          scene_write
            .scene
            .with_component_mut(cursor_id, |c: &mut HighResTransformComponent| {
              c.position = target_pos.to_f64();
            });
        // Mark cursor entity as changed.
        if let Some(ext_id) =
          scene_write.entity_map.iter().find(|&(_, v)| *v == cursor_id).map(|(k, _)| *k)
        {
          scene_write.mark_component_changed(
            ext_id,
            <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
          );
        }
      }

      // Dynamic offset calculation based on object bounds and camera FOV
      let mut r_local = 1.0_f64;
      if let Some(mesh) = scene_write
        .scene
        .with_component(target_entity, |c: &crate::scene::PhysicalMeshComponent| {
          c.clone()
        })
      {
        r_local = mesh.sphere_radius as f64;
      } else if let Some(col) = scene_write
        .scene
        .with_component(target_entity, |c: &crate::scene::ColliderComponent| {
          c.clone()
        })
      {
        r_local = match col.shape {
          crate::scene::ColliderShape::Sphere { radius } => radius as f64,
          crate::scene::ColliderShape::OBB { half_extents } => half_extents.length() as f64,
        };
      }

      let target_scale = scene_write
        .scene
        .global_transform_f64(target_entity)
        .map(|t| t.scale.x().max(t.scale.y()).max(t.scale.z()) as f64)
        .unwrap_or(1.0);
      let target_radius = r_local * target_scale;

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
      } else {
        return Ok(SimulationTaskResult::None);
      };

      let anim = crate::scene::animation::TransformAnimationComponent {
        start_pos,
        start_rot,
        target_pos: (target_pos + offset).to_f64(),
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

      if let Some(ext_id) =
        scene_write.entity_map.iter().find(|&(_, v)| *v == snap_entity).map(|(k, _)| *k)
      {
        scene_write.mark_component_changed(
          ext_id,
          <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FollowEntity {
      snap_entity,
      entity_id,
      scene,
      unfollow_other: _,
    } => {
      let scene_read = scene.read();
      use crate::scene::interaction::SceneInteractionExt;
      scene_read.scene.follow_entity(entity_id, None)?;
      try_snap_entity(snap_entity, entity_id, &scene_read)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::UnfollowEntity { entity_id, scene } => {
      let scene_read = scene.read();
      scene_read
        .scene
        .remove_component::<FollowingComponent>(entity_id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
      Ok(SimulationTaskResult::None)
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
        return Ok(SimulationTaskResult::None);
      }
      Err(EngineError::InvalidOperation("can't find scene"))
    }
    LogicCommand::PauseScene { scene_id } => {
      use oshal::os::time::v2::SimSpeed;

      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().speed = SimSpeed::Paused;
      }
      Ok(SimulationTaskResult::None)
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
          let model_id = scenes.import_model_from_mesh(path, mesh);
          Ok(SimulationTaskResult::U64(model_id))
        }
        Err(_) => Ok(SimulationTaskResult::U64(0)),
      }
    }
    // TODO: async tasklet. this should return the task_id, not take it, while a new command QueryLoadAlmanacFinished should poll the task id and return EngineResult<bool> true in case it finished
    LogicCommand::LoadAlmanac { task_id: _, path } => {
      ctx.load_almanac_file_internal(&path)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::UnloadAlmanac { task_id: _, path } => {
      ctx.unload_almanac_file_internal(&path)?;
      Ok(SimulationTaskResult::None)
    }

    // TODO: async tasklet. this should return the task_id, not take it, while a new command QueryLoadCometSpkFinished should poll the task id and return EngineResult<bool> true in case it finished
    LogicCommand::LoadCometSpk {
      task_id: _,
      spk_id,
      frame,
      epoch,
    } => {
      let logic_state = ctx.logic_state.read();
      let state = logic_state.almanac_data.get_ephem_full(spk_id, frame, epoch, false, false)?;
      Ok(SimulationTaskResult::KinematicState(state))
    }

    LogicCommand::SpawnModelInstance {
      task_id: _,
      scene_id,
      model_id,
      name,
    } => {
      // Preload if not in cache (outside lock)
      let path_opt = {
        let scenes = ctx.scenes.read();
        scenes.model_registry.get(&model_id).cloned()
      };

      if let Some(path_str) = path_opt {
        let needs_load = {
          let scenes = ctx.scenes.read();
          !scenes.mesh_cache.get(&path_str).is_some()
        };

        if needs_load {
          let mesh_res = if path_str.ends_with(".obj") || path_str.ends_with(".OBJ") {
            crate::simulation::comet::load_comet_from_obj(&path_str, false, None)
          } else if path_str.ends_with(".ply") || path_str.ends_with(".PLY") {
            crate::simulation::comet::load_comet_from_ply(&path_str, false, None)
          } else {
            crate::simulation::comet::load_comet_from_gltf(&path_str, false, None)
          };
          if let Ok(mesh) = mesh_res {
            let scenes = ctx.scenes.read();
            scenes.mesh_cache.insert(path_str.clone(), mesh);
          }
        }

        let mut scenes = ctx.scenes.write();
        let instance_id = scenes.spawn_model_instance_internal(scene_id, model_id, &name)?;
        Ok(SimulationTaskResult::U64(instance_id))
      } else {
        Err(EngineError::InvalidOperation("model not found"))
      }
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

      let frame = crate::simulation::almanac::SUN_ECLIPJ2000;

      let mut samples = alloc::vec::Vec::new();
      let mut t = start_epoch_tai_sec;

      let logic_state = ctx.logic_state.read();

      let step_sec = sample_step_days * 86400.0;

      // Ensure we at least sample the end point precisely
      while t <= end_epoch_tai_sec {
        let epoch = anise::time::Epoch::from_tai_seconds(t);
        let state = logic_state.almanac_data.get_ephem_full(spk_id, frame, epoch, true, false)?;
        samples.push(state);

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

      let dt_sec = step_sec as f32;
      let mut control_points = alloc::vec::Vec::new();

      for i in 0..(samples.len() - 1) {
        let s0 = &samples[i];
        let s1 = &samples[i + 1];

        let p0 = s0.position;
        let v0 = s0.velocity;
        let p1 = s1.position;
        let v1 = s1.velocity;

        let cp0 = p0;
        let cp1 = p0 + v0 * (dt_sec / 3.0);
        let cp2 = p1 - v1 * (dt_sec / 3.0);
        let cp3 = p1;

        control_points.push([cp0.x(), cp0.y(), cp0.z(), 1.0]);
        control_points.push([cp1.x(), cp1.y(), cp1.z(), 1.0]);
        control_points.push([cp2.x(), cp2.y(), cp2.z(), 1.0]);
        control_points.push([cp3.x(), cp3.y(), cp3.z(), 1.0]);
      }

      // Apply the component to the entity
      let scenes = ctx.scenes.read();
      let scene_ctx =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let scene_guard = scene_ctx.read();
      let entity = EntityId::from(slotmap::KeyData::from_ffi(entity_id));

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
        |transform: &mut crate::scene::HighResTransformComponent| {
          transform.position = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
            sun_pos.x() as f64,
            sun_pos.y() as f64,
            sun_pos.z() as f64,
          );
          handled_highres = true;
        },
      );

      if !handled_highres {
        let mut handled_transform = false;
        let _ = scene_guard.scene.with_component_mut(
          entity,
          |transform: &mut crate::scene::TransformComponent| {
            transform.position = sun_pos;
            handled_transform = true;
          },
        );
        if !handled_transform {
          let mut new_transform = crate::scene::TransformComponent::default();
          new_transform.position = sun_pos;
          let _ = scene_guard.scene.add_component(entity, new_transform);
        }
      }

      Ok(SimulationTaskResult::None)
    }

    // TODO one of these two should be removed
    LogicCommand::RaycastNdc {
      task_id: _,
      scene_id,
      camera_id,
      ndc_x,
      ndc_y,
    } => {
      aethervk_oshal_rlib::log!(
        "DEBUG logic_thread: RaycastNdc received scene={} camera={}",
        scene_id,
        camera_id
      );
      let res = ctx.raycast_ndc_internal(scene_id, camera_id, ndc_x, ndc_y)?;
      Ok(SimulationTaskResult::Raycast(res))
    }
    LogicCommand::Raycast {
      task_id: _,
      scene_id,
      ro,
      rd,
    } => {
      let res = ctx.raycast_internal(scene_id, ro, rd)?;
      Ok(SimulationTaskResult::Raycast(res))
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
    vulkan_device: &crate::gpu_backends::vulkan::device::Device,
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
    if !self.submitted {
      let _ = self.submit();
    }
  }
}

/// ticks `time_mgr always`, updates `scene` after a simulation step executed successfully
/// Should return
/// - next timeline semaphore value so that we can update our `latest_physics_sync`
/// - whether or not we reached end epoch, and therefore simulation is finished
fn execute_simulation_tick(
  vulkan_device: &crate::gpu_backends::vulkan::device::Device,
  scene_id: u64,
  scene: parking_lot::lock_api::RwLockUpgradableReadGuard<parking_lot::RawRwLock, SceneContext>,
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
  let scaled_fixed_dt_300ths = us_to_300ths_rounded(scaled_fixed_dt_us);

  time_mgr.tick();

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
          "Scene integrity violation: Comet entity should have as direct parent a reference frame component"
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
    while time_mgr.consume_fixed_step(scaled_fixed_dt_us) {
      let fixed_dt_scaled_s = utils::time_micro_to_seconds(scaled_fixed_dt_us);
      use aethervk_oshal_rlib::os::time::us_to_300ths_rounded;

      // ------------------------------------------------------------------------------------
      // Particle Systems Vulkan Shader Recording (Submission inserted after the fixed accumulator
      // loop)
      // ------------------------------------------------------------------------------------
      // -- all emissions --
      let mut skip_bind = false;
      for (idx, (ps, id)) in ps_extraction.iter().enumerate() {
        if particle_executions[idx].dead == 0 {
          // some emission constants which may be moved if exposed as parameters
          let mean_intra_grains_distance_mm = 1_f32;
          let min_cumulated_mass_g = 0.001_f32; // from bvh_utils.glsl

          let push_constants = utils::new_particles_emit(
            vulkan_device,
            id.as_ffi(),
            &ps.emission_params,
            mean_intra_grains_distance_mm,
            min_cumulated_mass_g,
            particle_executions[idx].r_helio_au,
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
          if let Ok(()) =
            vulkan_device.cmd_particle_system_velocity_vertlet_kick(cmd, &push_constants, skip_bind)
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
          let push_constants =
            utils::new_particles_compact(vulkan_device, id.as_ffi(), now_scaled_300ths, ttl_300ths);
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
          if let Ok((global_dpos, global_rot)) = body_state.almanac_planet.step(
            current_epoch,
            almanac,
            body_state.body_rotational_model.as_ref(),
          ) {
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
          } else {
            // TODO log error and don't update
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
        let gt = scene.scene.global_transform_f64(*id).unwrap();
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
    // SPICE EZR Kernel: Commit Comet Cartesian state update to scene
    // ------------------------------------------------------------------------------------
    if do_cross_sync {
      // TODO upgrade Scene context lock to write so that stop rendering for a bit.
      for kv_ref in cartesian_state_cache.iter() {
        let key = kv_ref.key();
        let state = kv_ref.value();
        let entity_id = EntityId::from_ffi(key.entity_id);
        if let Some(ref body_state) = state.comet_state {
          // comet/planet, update its trasform
          scene
            .scene
            .with_component_mut(entity_id, |t: &mut TransformComponent| {
              *t = body_state.transform;
            })
            .unwrap();
        } else {
          // reference frame, update its transform
          scene
            .scene
            .with_component_mut(entity_id, |t: &mut TransformComponent| {
              *t = state.parent_frame_transform;
            })
            .unwrap();
        }
      }

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

  // ------------------------------------------------------------------------------------
  // -- Update Phase --
  // ------------------------------------------------------------------------------------

  todo!()
}

/// update phase of the `execute_simulation_tick`, extracted into its own function so that we can
/// execute its logic in the `!physics_done` branch
/// Note: we assume entities driven in fixed update are not also driven by some update rules
/// which do not involve compute queue, therefore we won't go through cross sync window to perform
/// an update
fn execute_simulation_tick_update_phase(
  scene: parking_lot::lock_api::RwLockUpgradableReadGuard<parking_lot::RawRwLock, SceneContext>,
) {
  // -- Snap following entities --
  // -- Update Transform Animations --
  // -- Invoke SimulationCallback --
  // -- Clear changed entities --
  todo!()
}

// fn execute_simulation_tick_old(
//   scene_id: u64,
//   ctx: &alloc::sync::Arc<LogicThreadContext>,
//   dt: f32,
//   cmd_rx: &mpsc::Receiver<LogicCommand>,
// ) -> EngineResult<()> {
//   let (
//     time_state_arc,
//     _physics_scene_arc,
//     _scene_arc,
//     active_physics_task,
//     is_tlas_dirty,
//     static_tlas,
//   ) = {
//     let scenes = ctx.scenes.read();
//     let scene_ctx =
//       scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
//     let scene_read = scene_ctx.read();
//     (
//       scene_read.time_state.clone(),
//       scene_read.physics_scene.clone(),
//       scene_read.scene.clone(),
//       scene_read.active_physics_task.clone(),
//       scene_read.is_static_tlas_dirty.clone(),
//       scene_read.static_tlas.clone(),
//     )
//   };
//
//   // TODO: either keep this and refactor it into a method which executed every time not here but in
//   // logic loop for each scene, or, if raycasting is not needed, throw this away
//   if is_tlas_dirty.swap(false, core::sync::atomic::Ordering::Relaxed) {
//     let new_tlas = crate::physics::tlas_builder::build_selection_tlas(&_scene_arc);
//     *static_tlas.write() = new_tlas;
//   }
//
//   // Collect the previous physics task result without blocking.
//   // The outer scheduling loop (start_logic_thread) gates entry into
//   // execute_simulation_tick on `is_done()` returning true, so the task is
//   // guaranteed to be finished before we arrive here.  We use try_wait(0) to
//   // retrieve the result non-blockingly; if for any reason it is not done yet
//   // (should be impossible) we log a warning and drop the task rather than
//   // stalling the render thread.
//   if let Some(task) = active_physics_task.lock().take() {
//     match task.try_wait(0) {
//       Ok(result) => {
//         if let Err(e) = result {
//           aethervk_oshal_rlib::log!("Physics tasklet failed: {:?}", e);
//         }
//       }
//       Err(_still_running) => {
//         // Should never happen: the outer loop already confirmed is_done().
//         aethervk_oshal_rlib::log!(
//           "[execute_simulation_tick] physics task not done when expected — dropping to avoid blocking"
//         );
//         // _still_running is dropped here, which does NOT cancel the GPU work;
//         // the Arc<TaskletState> kept alive inside the thread pool workload will
//         // complete naturally.  The sync info is written directly into
//         // latest_physics_sync by the tasklet closure, so no data is lost.
//       }
//     }
//   }
//
//   // 1. Update time natively using the Arc (no scene read lock held)
//   {
//     let scenes = ctx.scenes.read();
//     if let Some(time_manager) = scenes.time_managers.get_mut(&scene_id) {
//       time_manager.tick();
//     }
//   }
//
//   let mut any_fixed_step = false;
//
//   // ── Adaptive batched dispatch ────────────────────────────────────────────
//   // Instead of looping and dispatching one GPU step per iteration (each of
//   // which previously blocked on its own fence), we now read the full
//   // accumulated lag, batch up to MAX_BATCH_STEPS worth of simulation into a
//   // single dispatch, and let the GPU shader sub-step internally.
//   //
//   // Thresholds:
//   //   MAX_BATCH_STEPS  – max sub-steps batched into one GPU dispatch.
//   //   YIELD_THRESHOLD  – if pending > this, sleep 1 ms before dispatching
//   //                      (leaky-bucket back-pressure: lets GPU drain).
//   //   DISCARD_THRESHOLD – hard cap; discard excess to prevent spiral of death.
//   const MAX_BATCH_STEPS: u32 = 4;
//   const YIELD_THRESHOLD: u32 = 2;
//   const DISCARD_THRESHOLD: u32 = 8;
//
//   use oshal::os::time::v2::SimSpeed;
//   let (is_playing, pending_steps, fixed_dt_us, base_step_days, current_epoch, max_sub_dt_us) = {
//     let ts = time_state_arc.read();
//     let pending = ti.pending_fixed_steps();
//     let fixed_dt_us = ti.fixed_delta_time.load(core::sync::atomic::Ordering::Relaxed);
//     let effective_max_sub_dt_s = ts
//       .max_sub_dt_override
//       .unwrap_or_else(|| ts.current_scale.max_physics_sub_dt_seconds());
//     let max_sub_dt_us =
//       (effective_max_sub_dt_s * 1_000_000.0) as aethervk_oshal_rlib::os::time::timeus_t;
//     let scale_days_per_sec = ts.current_scale.to_days_per_st_second();
//     let fixed_sim_seconds = fixed_dt_us as f64 / 1_000_000.0;
//     let base_step_days = scale_days_per_sec * fixed_sim_seconds;
//     (
//       ts.speed == SimSpeed::Paused,
//       pending,
//       fixed_dt_us,
//       base_step_days,
//       ts.current_epoch(),
//       max_sub_dt_us,
//     )
//   };
//
//   if is_playing && pending_steps > 0 {
//     // Compute how many steps we'll batch in this dispatch.
//     let n_steps = pending_steps.min(MAX_BATCH_STEPS);
//
//     // Back-pressure: if significantly behind, yield briefly so the GPU
//     // can drain its command queue before we enqueue more work.
//     if pending_steps > YIELD_THRESHOLD {
//       oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
//     }
//
//     // Discard runaway lag beyond the hard cap (spiral-of-death protection).
//     if pending_steps > DISCARD_THRESHOLD {
//       time_state_arc.read().time_info.write().ut_discard_accumulator();
//       aethervk_oshal_rlib::log!(
//         "Spiral-of-death protection: discarding {} pending steps",
//         pending_steps - MAX_BATCH_STEPS
//       );
//     }
//
//     // Advance epoch and accumulator by n_steps at once.
//     let batched_step_days = base_step_days * n_steps as f64;
//     let batched_dt_us = fixed_dt_us * n_steps as i64;
//
//     // ── GPU step-size safety cap ─────────────────────────────────────────────
//     // At very high timescales the batched step can be 50+ days.  The particle
//     // physics shader uses velocity-Verlet with a fixed dt; large dt causes
//     // orbital divergence, particles scatter to extreme positions, and the BVH
//     // traversal degenerates — observed as GPU hangs of 20+ seconds.
//     //
//     // We cap the dt passed to the GPU while still advancing the epoch and the
//     // fixed-time accumulator by the FULL batched amount.  At high timescale the
//     // particle physics is approximate (positions drifted relative to the real
//     // orbit) but the simulation stays responsive.  Kinematic body positions are
//     // unaffected (they are driven by the almanac every tick, not by the GPU).
//     const MAX_SAFE_GPU_STEP_DAYS: f64 = 1.0;
//     let gpu_step_days = batched_step_days.min(MAX_SAFE_GPU_STEP_DAYS);
//
//     let new_epoch = {
//       let mut ts = time_state_arc.write();
//       let mut epoch = ts.current_epoch + anise::time::Unit::Day * batched_step_days;
//       // Clamp to epoch boundaries, auto-pause if reached.
//       if epoch >= ts.epoch_end {
//         epoch = ts.epoch_end;
//         ts.is_playing = false;
//         aethervk_oshal_rlib::log!("Auto-paused: reached epoch end");
//       } else if epoch <= ts.epoch_start && batched_step_days < 0.0 {
//         epoch = ts.epoch_start;
//         ts.is_playing = false;
//         aethervk_oshal_rlib::log!("Auto-paused: reached epoch start");
//       }
//       ts.current_epoch = epoch;
//       epoch
//     };
//
//     any_fixed_step = true;
//     let step_wall_start = aethervk_oshal_rlib::os::time::get_monotonic_time();
//     if let Err(e) = dispatch_physics_step(
//       scene_id,
//       ctx,
//       gpu_step_days,
//       new_epoch,
//       batched_dt_us,
//       max_sub_dt_us,
//     ) {
//       aethervk_oshal_rlib::log!("dispatch_physics_step failed: {:?}", e);
//       // ── Cooldown: prevent tight error-retry loop ───────────────────────────
//       // Without a sleep, a device-lost error burns 600% CPU across worker threads.
//       // The batched dispatch is a single call (not a loop), so we just sleep here;
//       // the logic thread will naturally skip the watchdog/EMA update and wait for
//       // the next tick before trying again.
//       aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
//         500,
//       ));
//     }
//     let step_wall_us = aethervk_oshal_rlib::os::time::get_monotonic_time()
//       .saturating_sub(step_wall_start)
//       .max(1);
//
//     // ── Wall-time watchdog ───────────────────────────────────────────────────
//     // If the GPU dispatch blocked the logic thread for more than 2 seconds the
//     // GPU was overloaded (e.g. too many particles + large dt).  Discard the
//     // pending accumulator so we don't pile on more heavy dispatches, and reset
//     // the EMA speed estimate so the UI shows a meaningful value.
//     const GPU_OVERLOAD_THRESHOLD_US: i64 = 2_000_000; // 2 s
//     if step_wall_us > GPU_OVERLOAD_THRESHOLD_US {
//       aethervk_oshal_rlib::log!(
//         "[physics] GPU dispatch took {:.2}s — overload detected, discarding pending accumulator",
//         step_wall_us as f64 / 1_000_000.0
//       );
//       time_state_arc.read().time_info.write().ut_discard_accumulator();
//     }
//
//     // Update EMA: compare per-step sim-time vs wall-time.
//     {
//       let mut ts = time_state_arc.write();
//       let sim_us = fixed_dt_us.max(1) as f32;
//       let wall_us = (step_wall_us / n_steps as i64).max(1) as f32;
//       let sample = (sim_us / wall_us).min(2.0);
//       ts.effective_sim_speed = 0.1 * sample + 0.9 * ts.effective_sim_speed;
//     }
//
//     // Advance the fixed-time accumulator n_steps times.
//     {
//       let ts = time_state_arc.read();
//       let ti = ts.time_info.read();
//       for _ in 0..n_steps {
//         ti.ut_fixed_update();
//       }
//     }
//
//     if drain_logic_commands(cmd_rx, ctx) {
//       return Err(EngineError::InvalidOperation("shutdown"));
//     }
//     refit_kinematic_microframes(scene_id, ctx);
//   }
//
//   // Snap following entities (re-acquires brief scene graph lock since we need try_snap_entity)
//   {
//     let scenes = ctx.scenes.read();
//     if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
//       let scene_ctx = scene_ctx_guard.read();
//
//       // Ensure focus distance is fixed to the cursor plane on each tick update
//       if let Some((cursor_id, _)) =
//         scene_ctx.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id))
//       {
//         if let Some(cursor_global) = scene_ctx.scene.global_transform(cursor_id) {
//           let mut cam_updates = alloc::vec::Vec::new();
//           let _ = scene_ctx.scene.query1_res_mut(|id, _: &mut CameraComponent| {
//             cam_updates.push(id);
//             Some(())
//           });
//           for cam_id in cam_updates {
//             if let Some(cam_global) = scene_ctx.scene.global_transform(cam_id) {
//               let fwd = cam_global.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
//               let dist_global = (cursor_global.position - cam_global.position).dot(fwd);
//               let scale_z = if cam_global.scale.z().abs() > 1e-15 {
//                 cam_global.scale.z()
//               } else {
//                 1.0
//               };
//               let dist_local = dist_global / scale_z;
//               if dist_local > 0.1 {
//                 let _ = scene_ctx.scene.with_component_mut(cam_id, |cam: &mut CameraComponent| {
//                   cam.focus_distance = dist_local;
//                 });
//               }
//             }
//           }
//         }
//       }
//
//       if any_fixed_step {
//         for (ext_id, ent_id) in scene_ctx.entity_map.iter() {
//           // Check if entity has TransformComponent
//           let mut changed = false;
//           scene_ctx.scene.with_component(*ent_id, |_: &TransformComponent| {
//             changed = true;
//           });
//           if changed {
//             use crate::scene::ForeignSerializable;
//             scene_ctx
//               .mark_component_changed(*ext_id, crate::scene::TransformComponent::COMPONENT_ID);
//             // Also mark Comet and Planet and Sun (using dummy IDs until they implement ForeignSerializable)
//             scene_ctx.mark_component_changed(*ext_id, 100); // Comet
//             scene_ctx.mark_component_changed(*ext_id, 101); // Planet
//             scene_ctx.mark_component_changed(*ext_id, 102); // Sun
//           }
//         }
//       }
//
//       if let Some((target_id, _)) =
//         scene_ctx.scene.query1_first_res::<FollowingComponent, _, _>(|id, _| Some(id))
//       {
//         if let Some((cursor_id, _)) =
//           scene_ctx.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id))
//         {
//           let _ = try_snap_entity(cursor_id, target_id, &scene_ctx);
//         }
//       }
//       if let Some(st_lock) = &scene_ctx.selection_tlas {
//         let mut st = st_lock.write();
//         *st = crate::physics::tlas_builder::build_selection_tlas(scene_ctx.scene.as_ref());
//       }
//     }
//   }
//
//   // Update Transform Animations
//   {
//     let scenes = ctx.scenes.read();
//     if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
//       let scene_ctx = scene_ctx_guard.read();
//       let mut to_update = alloc::vec::Vec::new();
//       let _ = scene_ctx.scene.query1_res_mut(
//         |id, anim: &mut crate::scene::animation::TransformAnimationComponent| {
//           if !anim.is_finished {
//             anim.elapsed += dt;
//             let mut t = anim.elapsed / anim.duration;
//             if t > 1.0 {
//               t = 1.0;
//             }
//             if t < 0.0 {
//               t = 0.0;
//             }
//             // Hermite smoothstep
//             let smooth_t = t * t * (3.0 - 2.0 * t);
//
//             let new_pos = aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::lerp(
//               anim.start_pos,
//               anim.target_pos,
//               smooth_t as f64,
//             );
//             let new_rot = aethervk_oshal_rlib::math::vector::vec4::Quat::slerp(
//               anim.start_rot,
//               anim.target_rot,
//               smooth_t,
//             );
//
//             if anim.elapsed >= anim.duration {
//               anim.is_finished = true;
//             }
//             to_update.push((id, new_pos, new_rot));
//           }
//           Some(())
//         },
//       );
//
//       for (id, pos, rot) in to_update {
//         let _ = scene_ctx.scene.with_component_mut(id, |t: &mut HighResTransformComponent| {
//           t.position = pos;
//           t.rotation = rot;
//         });
//
//         if let Some(ext_id) = scene_ctx.entity_map.iter().find(|&(_, v)| *v == id).map(|(k, _)| *k)
//         {
//           scene_ctx.mark_component_changed(
//                 ext_id,
//                 <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
//             );
//         }
//       }
//     }
//   }
//
//   let sim_task_id = ctx.task_manager.write().create_task();
//   ctx.task_manager.write().success_task(
//     sim_task_id.get(),
//     crate::simulation_api::structs::SimulationTaskResult::None,
//   );
//
//   // Invoke SimulationCallback
//   let fptr = *crate::simulation_api::SIMULATION_CALLBACK.read();
//   if fptr.is_some() {
//     let tm = alloc::sync::Arc::clone(&ctx.task_manager);
//
//     // Extract changed DTOs before entering the async tasklet
//     let mut changes_to_stream = alloc::vec::Vec::new();
//     let scenes = ctx.scenes.read();
//     if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
//       let scene_ctx = scene_ctx_guard.read();
//       let changed = scene_ctx.changed_entities.read();
//
//       for (ext_id, components) in changed.iter() {
//         if let Some(internal_entity) = scene_ctx.entity_map.get(ext_id) {
//           for comp_id in components.iter() {
//             match *comp_id {
//               // ── ForeignSerializable components: send the DTO inline ───────
//               id if id == <crate::scene::TransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID => {
//                 if let Some(dto) = scene_ctx.scene.with_component(
//                   *internal_entity,
//                   |c: &crate::scene::TransformComponent| {
//                     use crate::scene::ForeignSerializable;
//                     c.to_foreign()
//                   },
//                 ) {
//                   changes_to_stream.push((
//                     *ext_id,
//                     id,
//                     Some(alloc::boxed::Box::new(dto) as alloc::boxed::Box<dyn core::any::Any + Send>),
//                   ));
//                 }
//               }
//               id if id == <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID => {
//                 if let Some(dto) = scene_ctx.scene.with_component(
//                   *internal_entity,
//                   |c: &crate::scene::CameraComponent| {
//                     use crate::scene::ForeignSerializable;
//                     c.to_foreign()
//                   },
//                 ) {
//                   changes_to_stream.push((
//                     *ext_id,
//                     id,
//                     Some(alloc::boxed::Box::new(dto) as alloc::boxed::Box<dyn core::any::Any + Send>),
//                   ));
//                 }
//               }
//               id if id == <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID => {
//                 if let Some(dto) = scene_ctx.scene.with_component(
//                   *internal_entity,
//                   |c: &crate::scene::HighResTransformComponent| {
//                     use crate::scene::ForeignSerializable;
//                     c.to_foreign()
//                   },
//                 ) {
//                   changes_to_stream.push((
//                     *ext_id,
//                     id,
//                     Some(alloc::boxed::Box::new(dto) as alloc::boxed::Box<dyn core::any::Any + Send>),
//                   ));
//                 }
//               }
//               // ── Non-serializable components: send pull-signal (null data) ─
//               // IDs 100 (Comet), 101 (Planet), 102 (Sun), and any future IDs.
//               // C# will call PullFromNative() on the matching component.
//               other_id => {
//                 changes_to_stream.push((*ext_id, other_id, None));
//               }
//             }
//           }
//         }
//       }
//     }
//
//     use aethervk_oshal_rlib::os::pool::tasklet::ThreadPoolExt;
//     let _ = ctx.thread_pool.spawn_tasklet(None, move || {
//       let fptr = *crate::simulation_api::SIMULATION_CALLBACK.read();
//       loop {
//         let status = tm.read().get_status(sim_task_id.get());
//         if status == crate::simulation_api::structs::TaskStatusCode::Completed
//           || status == crate::simulation_api::structs::TaskStatusCode::Error
//         {
//           break;
//         }
//         oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
//       }
//
//       if let Some(cb) = fptr {
//         for (ext_id, comp_id, boxed_dto) in changes_to_stream {
//           let data_ptr = match &boxed_dto {
//             Some(dto) => &**dto as *const _ as *const core::ffi::c_void,
//             None => core::ptr::null(), // Pull-signal: C# will call PullFromNative()
//           };
//           unsafe { cb(scene_id, ext_id, comp_id, data_ptr) };
//         }
//       }
//     });
//   }
//
//   // Clear changed entities
//   let scenes = ctx.scenes.read();
//   if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
//     let scene_ctx = scene_ctx_guard.read();
//     scene_ctx.changed_entities.write().clear();
//   }
//
//   Ok(())
// }
//
// /// Threshold: re-center the microframe when the comet's local position
// /// reaches this fraction of the SOI radius (in micro-frame km).
// const MICROFRAME_REFIT_THRESHOLD: f32 = 0.7;
//
// /// SUN_RADIUS_AU (same as in spawn_comet_internal).
// const SUN_RADIUS_AU_REFIT: f32 = 0.0046524726_f32;
//
// /// Re-centers kinematic comet microframes when the comet approaches the SOI
// /// boundary. Called after `dispatch_physics_step` (which updates the comet's
// /// world position via `AlmanacPlanet::step`) so the microframe tracks the comet.
// ///
// /// Algorithm:
// /// 1. Find all micro-frame entities.
// /// 2. For each, find the comet child (has `KinematicComponent` with
// ///    `use_model_rotation` indicating it's a kinematic comet, not a planet).
// /// 3. Compute the comet's displacement from the microframe center in km.
// /// 4. If displacement > threshold × SOI_km → translate the microframe center
// ///    to the comet's world position, adjust the comet's local position by the
// ///    inverse, and recompute the SOI.
// fn refit_kinematic_microframes(scene_id: u64, ctx: &alloc::sync::Arc<LogicThreadContext>) {
//   use crate::scene::{
//     ColliderComponent, ColliderShape, KinematicComponent, ReferenceFrameComponent,
//     ReferenceFrameType, TransformComponent,
//   };
//   use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
//
//   let scenes = ctx.scenes.read();
//   let Some(scene_ctx_guard) = scenes.get(&scene_id) else {
//     return;
//   };
//   let scene_ctx = scene_ctx_guard.read();
//   let scene = &scene_ctx.scene;
//
//   // Collect micro-frame entities
//   let mut micro_frames: alloc::vec::Vec<(EntityId, TransformComponent, ReferenceFrameComponent)> =
//     alloc::vec::Vec::new();
//
//   scene.query2::<TransformComponent, ReferenceFrameComponent, _>(|eid, t, f| {
//     if f.frame_type == ReferenceFrameType::Micro {
//       micro_frames.push((eid, *t, f.clone()));
//     }
//   });
//
//   for (frame_eid, frame_transform, frame_ref) in &micro_frames {
//     // Find comet child: entity with KinematicComponent + PhysicalMeshComponent
//     let children = match scene.get_children(*frame_eid) {
//       Some(c) => c,
//       None => continue,
//     };
//
//     for &child_eid in &children {
//       // Check if this child is a kinematic comet (has KinematicComponent)
//       let is_kinematic = scene
//         .with_component(child_eid, |k: &KinematicComponent| k.use_model_rotation)
//         .unwrap_or(false);
//
//       if !is_kinematic {
//         continue;
//       }
//
//       // Get comet local position (in micro-frame km space)
//       let comet_local_pos =
//         match scene.with_component(child_eid, |t: &TransformComponent| t.position) {
//           Some(pos) => pos,
//           None => continue,
//         };
//
//       // SOI radius in km: soi_radius_au / frame.scale (where scale = AU/km)
//       let soi_km = frame_ref.soi_radius / frame_ref.scale;
//       let displacement_km = comet_local_pos.length();
//
//       if displacement_km < MICROFRAME_REFIT_THRESHOLD * soi_km {
//         continue; // Comet is well within bounds
//       }
//
//       // Compute world-space delta: comet_local_km * scale → AU
//       let delta_au = comet_local_pos * frame_ref.scale;
//       let new_frame_pos = frame_transform.position + delta_au;
//
//       // Update microframe position (translate to comet's world position)
//       let _ = scene.with_component_mut(*frame_eid, |t: &mut TransformComponent| {
//         t.position = new_frame_pos;
//       });
//
//       // Inverse-translate the comet: its local position resets to origin
//       let _ = scene.with_component_mut(child_eid, |t: &mut TransformComponent| {
//         t.position = Vec3f32::zero();
//       });
//
//       // Recompute SOI radius based on new distance to sun
//       let new_dist_au = new_frame_pos.length();
//       let new_soi = (new_dist_au - SUN_RADIUS_AU_REFIT).max(SUN_RADIUS_AU_REFIT);
//
//       let _ = scene.with_component_mut(*frame_eid, |f: &mut ReferenceFrameComponent| {
//         f.soi_radius = new_soi;
//       });
//
//       // Update LCA collider half-extents to match new SOI
//       let _ = scene.with_component_mut(*frame_eid, |c: &mut ColliderComponent| {
//         c.shape = ColliderShape::OBB {
//           half_extents: Vec3f32::from_components(new_soi, new_soi, new_soi),
//         };
//       });
//
//       aethervk_oshal_rlib::log!(
//         "Microframe refit: displacement={:.1} km, threshold={:.1} km → new SOI={:.6} AU",
//         displacement_km,
//         MICROFRAME_REFIT_THRESHOLD * soi_km,
//         new_soi
//       );
//
//       break; // Only one comet per microframe
//     }
//   }
// }
//
// /// Emit new particles on the CPU side from `ParticleEmitterCirclesComponent` into
// /// each circle's child entity `ParticleSystemComponent`.
// ///
// /// This function also **ages** existing particles and **reaps** expired ones
// /// (marking them `active = 0`), freeing slots for new emission.
// ///
// /// This runs **before** the physics tasklet so that `build_particles` picks up the
// /// newly emitted particles (giving them proper `ParticleMetadata` for GPU write-back).
// ///
// /// Each circle emits `particles_per_second * dt_s` particles (fractional with carry-over)
// /// at the cached surface point, with velocity along the cached surface normal scaled
// /// by `mean_velocity` and jittered by `velocity_std_dev`.
// fn emit_particles_from_circles(
//   scene: &crate::scene::Scene,
//   tick_seed: u64,
//   dt_us: aethervk_oshal_rlib::os::time::timeus_t,
//   // Which GPU sub-step index this emission call is for (0 = first).
//   // Used to compute a velocity compensation so particles end up at the
//   // radiation-pressure position matching their emission order.
//   sub_step_idx: u32,
//   // Total GPU sub-steps per dispatch. Paired with sub_step_idx to compute
//   // the spread from 0 km (sub-step N-1) to x_max km (sub-step 0).
//   n_sub_steps: u32,
// ) {
//   use crate::scene::{
//     TransformComponent,
//     particles::{ParticleEmitterCirclesComponent, ParticleSystemComponent},
//   };
//   use aethervk_oshal_rlib::math::{quaternion::Quaternion, vector::Vector};
//
//   // Collect emission work so we don't hold the query borrow while writing child components.
//   struct Work {
//     child_entity: crate::scene::EntityId,
//     world_position: [f32; 3],
//     world_velocity_direction: [f32; 3],
//     /// Unit vector pointing AWAY from the sun at the emission point (anti-sunward).
//     /// Used as the direction for radiation-pressure velocity compensation so that
//     /// the compensation opposes the actual GPU force direction, not the surface normal.
//     anti_sunward_dir: [f32; 3],
//     parent_velocity: [f32; 3],
//     mass: f32,
//     mean_velocity: f32,
//     velocity_std_dev: f32,
//     /// Emission rate in particles/s; converted to an integer count per tick
//     /// via `floor(particles_per_second * dt_s + remainder)` with carry-over.
//     particles_per_second: f32,
//     color: [f32; 4],
//     ttl_us: u64,
//     beta: f32,
//     /// Spawn-disc radius in km (tangent plane around the surface point).
//     spawn_radius_km: f32,
//   }
//
//   let mut work: alloc::vec::Vec<(crate::scene::EntityId, alloc::vec::Vec<Work>)> =
//     alloc::vec::Vec::new();
//
//   let mut sun_pos = None;
//   scene.query2::<crate::scene::TransformComponent, crate::scene::SunComponent, _>(|_, t, _| {
//     sun_pos = Some(t.position);
//   });
//
//   let mut occluders: alloc::vec::Vec<(
//     crate::scene::TransformComponent,
//     alloc::sync::Arc<crate::simulation::comet::Comet>,
//   )> = alloc::vec::Vec::new();
//
//   scene.query2::<crate::scene::TransformComponent, crate::scene::PhysicalMeshComponent, _>(
//     |entity, t, mesh| {
//       // Exclude the Sun — it's the light source, not an occluder
//       if mesh.mesh.bvh.is_some()
//         && !Into::<bool>::into(scene.has_component::<crate::scene::SunComponent>(entity))
//       {
//         occluders.push((*t, mesh.mesh.clone()));
//       }
//     },
//   );
//
//   scene.query2::<TransformComponent, ParticleEmitterCirclesComponent, _>(
//     |entity_id, transform, emitter| {
//       // The emitter entity (comet) lives inside a micro-frame (LCA).
//       // Its TransformComponent.position is in micro-frame LOCAL KM — not AU.
//       // scale is km per mesh-unit.
//       let parent_rot = transform.rotation;
//       let parent_pos_km = transform.position; // local km relative to micro-frame
//       let parent_scale = transform.scale; // km per mesh-unit
//       let mut circles_work = alloc::vec::Vec::new();
//
//       // ── Walk up to find frame center (AU) and scale (AU/km) for occlusion ──
//       // Needed only for the occlusion ray-cast which works in macro-frame AU.
//       let mut frame_center_au = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
//       let mut frame_scale_au_per_km = 1.0_f32 / 149_597_870.7_f32; // default: 1 AU/km
//       {
//         let mut cur = scene.get_parent(entity_id);
//         while let Some(anc) = cur {
//           if let Some(frame_data) =
//             scene.with_component(anc, |rf: &crate::scene::ReferenceFrameComponent| {
//               (
//                 scene
//                   .with_component(anc, |t: &TransformComponent| t.position)
//                   .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero()),
//                 rf.scale,
//               )
//             })
//           {
//             frame_center_au = frame_data.0;
//             frame_scale_au_per_km = frame_data.1;
//             break;
//           }
//           cur = scene.get_parent(anc);
//         }
//       }
//
//       for circle in &emitter.circles {
//         let child_id = match circle.child_entity {
//           Some(id) => id,
//           None => continue,
//         };
//         let local_pos = match circle.cached_point {
//           Some(p) => p,
//           None => continue,
//         };
//         let local_normal = match circle.cached_normal {
//           Some(n) => n,
//           None => continue,
//         };
//         if circle.particles_per_second <= 0.0 {
//           continue;
//         }
//
//         // Surface point in mesh units → km relative to comet CoM
//         let surface_km = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
//           local_pos[0] * parent_scale.x(),
//           local_pos[1] * parent_scale.y(),
//           local_pos[2] * parent_scale.z(),
//         );
//         let rotated_surface_km = parent_rot.rotate_vector(surface_km);
//
//         let local_n = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
//           local_normal[0],
//           local_normal[1],
//           local_normal[2],
//         );
//         let world_n_km = {
//           let n = parent_rot.rotate_vector(local_n);
//           let l = n.length();
//           if l > 1e-6 {
//             n * (1.0 / l)
//           } else {
//             aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(0.0, 0.0, 1.0)
//           }
//         };
//
//         // Micro-frame local km: comet center (km) + surface offset (km) + 1 m nudge
//         let emit_km = parent_pos_km + rotated_surface_km + world_n_km * 1e-3;
//
//         // Occlusion check — all vectors in MICRO-FRAME LOCAL KM.
//         //
//         // Bug (pre-fix): used `emit_au - occ_t.position` which mixes AU and km,
//         // placing the BVH ray origin ~0.05 mesh-units inside the sphere for BOTH
//         // sunlit and dark faces, making every circle look occluded (or none if
//         // BVH is absent).
//         //
//         // Fix: derive sun position in local km, then compute:
//         //   local_origin = inv_rot × (emit_km - comet_center_km) / scale_km
//         //   local_dir    = inv_rot × to_sun_km.normalize()
//         // hit_t is in mesh-units; convert to km for the dist_to_sun comparison.
//         let emit_au = frame_center_au + emit_km * frame_scale_au_per_km;
//         let mut occluded = false;
//         if let Some(sun_p_au) = sun_pos {
//           // Sun in local km (same frame as emit_km and occ_t.position)
//           let sun_km = if frame_scale_au_per_km > 1e-30 {
//             (sun_p_au - frame_center_au) * (1.0 / frame_scale_au_per_km)
//           } else {
//             // Fallback: treat sun as 1 AU = 149597870.7 km in -x direction
//             aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
//               -149_597_870.7_f32,
//               0.0,
//               0.0,
//             )
//           };
//           let to_sun_km = sun_km - emit_km;
//           let dist_to_sun_km = to_sun_km.length();
//
//           // Dot product of emission normal with sun direction:
//           //   < 0 → surface faces sun (sunlit side, should NOT be occluded)
//           //   > 0 → surface faces away from sun (dark side, should be occluded)
//           let sun_dir_km = if dist_to_sun_km > 1e-4 {
//             to_sun_km * (1.0 / dist_to_sun_km)
//           } else {
//             to_sun_km
//           };
//           let nz_dot_sun = world_n_km.dot(sun_dir_km);
//
//           if dist_to_sun_km > 1e-4 {
//             for (occ_t, occ_comet) in &occluders {
//               if let Some(bvh) = &occ_comet.bvh {
//                 let inv_rot = occ_t.rotation.inverse();
//                 // Both emit_km and occ_t.position are in local km — correct frame.
//                 let local_origin =
//                   inv_rot.rotate_vector(emit_km - occ_t.position) * (1.0 / occ_t.scale.x());
//                 let local_dir = inv_rot.rotate_vector(sun_dir_km);
//                 if let Some((hit_t, _, _)) = bvh.raycast(
//                   local_origin, local_dir, &occ_comet.vertices, &occ_comet.indices,
//                 ) && hit_t > 1e-4   // ignore self-hit at emission surface
//                   && hit_t * occ_t.scale.x() < dist_to_sun_km
//                 {
//                   occluded = true;
//                   break;
//                 }
//               }
//             }
//           }
//
//           // ── Emission-side diagnostic (very low-frequency, avoids stutter) ───
//           {
//             static EMIT_SIDE_COUNTER: core::sync::atomic::AtomicU64 =
//               core::sync::atomic::AtomicU64::new(0);
//             let count = EMIT_SIDE_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
//             // 1701 calls/s × 50_000 = prints once every ~29 s — diagnostic but no stutter.
//             if count % 50_000 == 0 {
//               let bvh_present =
//                 !occluders.is_empty() && occluders.iter().any(|(_, m)| m.bvh.is_some());
//               // Use the oshal log macro (available in no_std; eprintln! is not)
//               aethervk_oshal_rlib::log!(
//                 "[EMIT-SIDE] circle child={:?} \
//                  surface_n=({:+.3},{:+.3},{:+.3}) \
//                  anti_sun=({:+.3},{:+.3},{:+.3}) \
//                  occluded={} bvh_present={} \
//                  emit_km=({:+.1},{:+.1},{:+.1})",
//                 child_id,
//                 world_n_km.x(),
//                 world_n_km.y(),
//                 world_n_km.z(),
//                 -sun_dir_km.x(),
//                 -sun_dir_km.y(),
//                 -sun_dir_km.z(),
//                 occluded,
//                 bvh_present,
//                 emit_km.x(),
//                 emit_km.y(),
//                 emit_km.z(),
//               );
//             }
//           }
//         }
//         if occluded {
//           continue;
//         }
//
//         let ttl_us = if circle.ttl > 0 {
//           circle.ttl.saturating_mul(dt_us as u64)
//         } else {
//           0
//         };
//
//         let mut parent_vel = [0.0, 0.0, 0.0];
//         if let Some(k) =
//           scene.with_component(entity_id, |k: &crate::scene::KinematicComponent| k.velocity)
//         {
//           parent_vel = [k.x(), k.y(), k.z()];
//         }
//
//         circles_work.push(Work {
//           child_entity: child_id,
//           // MICRO-FRAME LOCAL KM — what build_particles uploads and GPU shaders expect
//           world_position: [emit_km.x(), emit_km.y(), emit_km.z()],
//           world_velocity_direction: [world_n_km.x(), world_n_km.y(), world_n_km.z()],
//           // Anti-sunward direction: opposite of the emit→sun unit vector.
//           // This is the direction the radiation-pressure force acts, and the direction
//           // the velocity compensation must oppose to correctly spread particles.
//           anti_sunward_dir: {
//             if let Some(sun_p_au) = sun_pos {
//               let sun_km = if frame_scale_au_per_km > 1e-30 {
//                 (sun_p_au - frame_center_au) * (1.0 / frame_scale_au_per_km)
//               } else {
//                 aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
//                   -149_597_870.7_f32,
//                   0.0,
//                   0.0,
//                 )
//               };
//               let to_sun_km = sun_km - emit_km;
//               let d = to_sun_km.length();
//               if d > 1e-4 {
//                 // anti-sunward = negative of to_sun direction
//                 let anti = to_sun_km * (-1.0 / d);
//                 [anti.x(), anti.y(), anti.z()]
//               } else {
//                 [1.0, 0.0, 0.0] // fallback: +x
//               }
//             } else {
//               [1.0, 0.0, 0.0] // no sun found: default to +x
//             }
//           },
//           parent_velocity: parent_vel,
//           mass: circle.mass,
//           mean_velocity: circle.mean_velocity, // km/s
//           velocity_std_dev: circle.velocity_std_dev,
//           particles_per_second: circle.particles_per_second,
//           color: circle.color,
//           ttl_us,
//           beta: circle.beta,
//           spawn_radius_km: circle.spawn_radius_km,
//         });
//       }
//
//       if !circles_work.is_empty() {
//         work.push((entity_id, circles_work));
//       }
//     },
//   );
//
//   // Now age, reap, and emit into child entities' ParticleSystemComponents
//   let mut rng_state = tick_seed;
//   for (_parent_id, circles) in work {
//     for Work {
//       child_entity,
//       world_position,
//       world_velocity_direction,
//       anti_sunward_dir,
//       parent_velocity,
//       mass,
//       mean_velocity,
//       velocity_std_dev,
//       particles_per_second,
//       color,
//       ttl_us,
//       beta,
//       spawn_radius_km,
//     } in circles
//     {
//       scene.with_component_mut(child_entity, |psc: &mut ParticleSystemComponent| {
//         // Update render config, TTL, and beta from emission circle
//         psc.color = color;
//         psc.beta = beta;
//
//         // FIX 6: Write TTL unconditionally to allow UI to dynamically disable expiration to 0
//         psc.ttl_us = ttl_us as aethervk_oshal_rlib::os::time::timeus_t;
//
//         let mut particles = psc.particles.write();
//         let capacity = psc.capacity;
//         if capacity == 0 {
//           return;
//         }
//
//         // FIX 3: Prevent index overflow over long running servers/simulations.
//         // Subtracting down identically by `capacity` mathematically preserves modulo mappings!
//         if psc.head_index >= capacity {
//           let shift = (psc.head_index / capacity) * capacity;
//           psc.head_index -= shift;
//           psc.tail_index -= shift;
//         }
//
//         // ── 1. Age existing particles and reap expired ones ─────────────
//         let ttl = psc.ttl_us;
//         for idx in psc.head_index..psc.tail_index {
//           let p_idx = idx % capacity;
//           let p = &mut particles[p_idx];
//           if p.active != 0 {
//             let age = p.get_age().saturating_add(dt_us);
//             p.set_age(age);
//             if ttl > 0 && age >= ttl {
//               p.active = 0;
//             }
//           }
//         }
//
//         // Advance head if leading particles are inactive
//         while psc.head_index < psc.tail_index {
//           let p_idx = psc.head_index % capacity;
//           if particles[p_idx].active == 0 {
//             psc.head_index += 1;
//           } else {
//             break;
//           }
//         }
//
//         // ── 2. Emit new particles, writing at tail ─────────────
//         // Compute how many whole particles to emit this tick from the
//         // continuous rate, carrying the fractional remainder forward.
//         let dt_s = dt_us as f32 / 1_000_000.0;
//         let exact_count = particles_per_second * dt_s + psc.emit_remainder;
//         let emit_count = (exact_count.floor() as u32).min(capacity as u32);
//         psc.emit_remainder = exact_count - emit_count as f32;
//
//         // Throttled emit diagnostic — prints once per ~120 calls per circle (≈ every 2 s at 60 Hz).
//         // Shows rate, actual count, and buffer state so you can tell if particles ARE being generated.
//         {
//           use core::sync::atomic::{AtomicU64, Ordering};
//           static EMIT_DIAG_N: AtomicU64 = AtomicU64::new(0);
//           let n = EMIT_DIAG_N.fetch_add(1, Ordering::Relaxed);
//           if n % 120 == 0 {
//             let alive_in_ring = psc.tail_index.saturating_sub(psc.head_index);
//             aethervk_oshal_rlib::log!(
//               "[EMIT] child={:?} rate={:.1}/s dt_s={:.3} emit={} alive={}/{} gpu_alive={} render_r={:.3} km",
//               child_entity,
//               particles_per_second,
//               dt_s,
//               emit_count,
//               alive_in_ring,
//               capacity,
//               psc.gpu_alive_count,
//               psc.render_radius_km,
//             );
//           }
//         }
//
//
//         for i in 0..emit_count {
//           if psc.tail_index - psc.head_index >= capacity {
//             // Buffer is full — evict the oldest particle (FIFO) to make room.
//             // This keeps the tail continuously emitting new particles and creates
//             // a rolling trail effect rather than silently dropping new particles.
//             psc.head_index += 1;
//           }
//
//           // Re-seed per-particle by mixing the global particle id into the tick
//           // seed.  Without this, every tick starts from the same rng_state and
//           // emits the same random positions/velocities (the "disk" pattern).
//           // splitmix64-style avalanche gives good diffusion before the LCG runs.
//           let per_particle_seed = tick_seed
//             .wrapping_add(psc.next_id as u64)
//             .wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
//           rng_state = per_particle_seed;
//           rng_state ^= rng_state >> 30;
//           rng_state = rng_state.wrapping_mul(0xbf58476d1ce4e5b9);
//           rng_state ^= rng_state >> 27;
//           rng_state = rng_state.wrapping_mul(0x94d049bb133111eb);
//           rng_state ^= rng_state >> 31;
//
//           // FIX 4: Shift by 32 and divide by u32::MAX properly scales distribution evenly over [0.0, 1.0].
//           rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//           let u0 = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32);
//           rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//           let u1 = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32);
//           rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//           let u2 = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32);
//
//           // FIX: Add `u3` to decouple independent speed from emission angle direction
//           rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//           let u3 = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32);
//
//           // Velocity magnitude: Gaussian(mean_vel, vel_std) via Box-Muller
//           let r = (-2.0 * u0.max(1e-8).ln()).sqrt();
//           let theta = 2.0 * core::f32::consts::PI * u1;
//           let speed = (mean_velocity + velocity_std_dev * r * theta.cos()).max(0.0);
//
//           // Cosine-hemisphere jitter around the emission normal.
//           // The cone half-angle is derived from the velocity spread ratio
//           // (σ/μ): 0 → collimated beam, 1 → full hemisphere.
//           // This removes the hardcoded 30° cap and makes spread data-driven.
//           let phi = 2.0 * core::f32::consts::PI * u2;
//           let spread_ratio = if mean_velocity > 0.0 {
//             (velocity_std_dev / mean_velocity).min(1.0)
//           } else {
//             1.0
//           };
//           let max_spread_cos = (1.0 - spread_ratio).max(0.0);
//           let cos_theta_h = (1.0 - u3 * (1.0 - max_spread_cos)).max(0.0).sqrt();
//           let sin_theta_h = (1.0 - cos_theta_h * cos_theta_h).max(0.0).sqrt();
//           let jitter_local = [
//             sin_theta_h * phi.cos(),
//             sin_theta_h * phi.sin(),
//             cos_theta_h,
//           ];
//
//           // Build tangent frame around emission normal
//           let n =
//             aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(world_velocity_direction);
//           let mut tangent =
//             aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 0.0, 0.0);
//           if n.dot(tangent).abs() > 0.99 {
//             tangent =
//               aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(0.0, 1.0, 0.0);
//           }
//           let bitangent = n.cross(tangent);
//           let bl = bitangent.length();
//           let bitangent = if bl > 1e-6 {
//             bitangent * (1.0 / bl)
//           } else {
//             tangent
//           };
//           let tangent = bitangent.cross(n);
//           let tl = tangent.length();
//           let tangent = if tl > 1e-6 {
//             tangent * (1.0 / tl)
//           } else {
//             bitangent
//           };
//
//           let world_dir =
//             tangent * jitter_local[0] + bitangent * jitter_local[1] + n * jitter_local[2];
//           let wdl = world_dir.length();
//           let world_dir = if wdl > 1e-6 {
//             world_dir * (1.0 / wdl)
//           } else {
//             n
//           };
//
//           let vel_jittered = [
//             (world_dir * speed).x(),
//             (world_dir * speed).y(),
//             (world_dir * speed).z(),
//           ];
//
//           let p_idx = psc.tail_index % capacity;
//           let id = psc.next_id as u64;
//
//           psc.next_id = psc.next_id.wrapping_add(1);
//           psc.tail_index += 1;
//
//           // Interpolate the starting position along the comet's path during this tick
//           let fraction = if emit_count > 1 { i as f32 / ((emit_count - 1) as f32) } else { 0.0 };
//           let interp_pos = [
//             world_position[0] - parent_velocity[0] * dt_s * fraction,
//             world_position[1] - parent_velocity[1] * dt_s * fraction,
//             world_position[2] - parent_velocity[2] * dt_s * fraction,
//           ];
//
//           // ── Spawn-disc position scatter ─────────────────────────────────
//           // Offset in the tangent plane by a uniformly-sampled disc of
//           // radius `spawn_radius_km`. Using sqrt(r_frac) gives uniform
//           // area density (avoids clustering at the centre).
//           let spawn_pos = if spawn_radius_km > 0.0 {
//             rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//             let disc_r_frac = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32);
//             rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
//             let disc_phi = ((rng_state >> 32) as u32 as f32) / (core::u32::MAX as f32)
//               * 2.0 * core::f32::consts::PI;
//             // sqrt maps uniform [0,1] → uniform area in disc
//             let disc_r = disc_r_frac.sqrt() * spawn_radius_km;
//             let offset = tangent * (disc_r * disc_phi.cos())
//               + bitangent * (disc_r * disc_phi.sin());
//             [
//               interp_pos[0] + offset.x(),
//               interp_pos[1] + offset.y(),
//               interp_pos[2] + offset.z(),
//             ]
//           } else {
//             interp_pos
//           };
//
//           // NO MORE CRASH. The length of the Vec natively matches its capacity and is securely memory backed.
//           let p = &mut particles[p_idx];
//           p.set_id(id);
//           p.set_age(0);
//           p.position = spawn_pos;
//
//           // ── Per-sub-step velocity compensation ───────────────────────────────
//           // All emitted particles are initialised by the GPU in sub-step 0 and
//           // therefore receive the full N GPU sub-steps of integration.  To make
//           // particle `sub_step_idx` land at the position it would have reached
//           // with only (N - i) sub-steps, we pre-add a compensating sunward
//           // velocity:
//           //   v_comp = -0.5 × a × dt_s × i × (2N - i) / N   (< 0 = toward sun)
//           // The brief sunward trajectory occurs inside the GPU dispatch (not
//           // rendered); the user only sees the correct final positions.
//           // a_net = (beta - 1) × GM_sun / r²  (≈ 5.93e-6 km/s² for β=2, r=1 AU)
//           //
//           // DIRECTION: compensation must oppose the ANTI-SUNWARD direction (the
//           // actual radiation-pressure force axis), NOT the surface normal.
//           // `anti_sunward_dir` is the unit vector comet→anti-sun, computed when
//           // building the Work item while `sun_pos` is in scope.
//           let v_comp_scalar = if n_sub_steps > 1 && beta > 1.001 {
//             const GM_SUN: f32 = 1.327_124e11_f32;  // km³/s²
//             const R_1AU_KM: f32 = 149_597_870.7_f32; // km
//             let net_a = (beta - 1.0).max(0.0) * GM_SUN / (R_1AU_KM * R_1AU_KM);
//             let dt_s = dt_us as f32 / 1_000_000.0;
//             let i = sub_step_idx as f32;
//             let n = n_sub_steps as f32;
//             // Sunward (negative anti-sunward) compensation
//             -0.5 * net_a * dt_s * i * (2.0 * n - i) / n
//           } else {
//             0.0_f32
//           };
//           // Apply compensation along the ANTI-SUNWARD axis, not the surface normal.
//           let ad = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
//             anti_sunward_dir[0], anti_sunward_dir[1], anti_sunward_dir[2],
//           );
//           p.velocity = [
//             vel_jittered[0] + v_comp_scalar * ad[0],
//             vel_jittered[1] + v_comp_scalar * ad[1],
//             vel_jittered[2] + v_comp_scalar * ad[2],
//           ];
//           p.mass = mass;
//           p.active = 1;
//         }
//       });
//     }
//   }
// }
//
// fn dispatch_physics_step(
//   scene_id: u64,
//   ctx: &alloc::sync::Arc<LogicThreadContext>,
//   step_days: f64,
//   current_epoch: anise::time::Epoch,
//   fixed_dt_us: aethervk_oshal_rlib::os::time::timeus_t,
//   max_sub_dt_us: aethervk_oshal_rlib::os::time::timeus_t,
// ) -> EngineResult<()> {
//   let scenes = ctx.scenes.read();
//   let scene_ctx = scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
//   let scene_read = scene_ctx.read();
//   let physics_scene_arc = scene_read.physics_scene.clone();
//   let scene_arc = scene_read.scene.clone();
//
//   // Non-blocking guard: ensure no in-flight tasklet holds a write-lock on
//   // physics_scene before we rebuild it.  In practice this branch is never
//   // taken: execute_simulation_tick drains the Mutex at Site 1 (above) before
//   // calling us, and dispatch_physics_step is not called concurrently.
//   // The defensive check is kept so that callers added in the future do not
//   // accidentally introduce a race.
//   if let Some(prev_task) = scene_read.active_physics_task.lock().take() {
//     match prev_task.try_wait(0) {
//       Ok(result) => {
//         if let Err(e) = result {
//           aethervk_oshal_rlib::log!("Previous physics tasklet failed: {:?}", e);
//         }
//       }
//       Err(_still_running) => {
//         // Task not finished yet.  This should be impossible given the
//         // is_done() gate in the outer loop, but if it ever happens we must
//         // NOT block here.  Skip the physics scene rebuild for this tick;
//         // the GPU task will complete before the next call (outer gate
//         // ensures this) and the scene will be rebuilt then.
//         aethervk_oshal_rlib::log!(
//           "[dispatch_physics_step] in-flight task still running — skipping scene rebuild to avoid blocking"
//         );
//         // _still_running put back so the outer loop's is_done() check can
//         // gate the next tick correctly.
//         *scene_read.active_physics_task.lock() = Some(_still_running);
//         return Ok(());
//       }
//     }
//   }
//
//   if let Some(ps_lock) = &physics_scene_arc {
//     let mut ps = ps_lock.write();
//     let dt_s = (step_days * 86400.0) as f32;
//     *ps = crate::physics::physics_scene::PhysicsScene::build_from_scene(scene_arc.as_ref(), dt_s);
//   }
//
//   // Scope the logic_state read guard tightly — holding it across the
//   // physics tasklet spawn would block LoadAlmanac's write() on the pool.
//   {
//     let logic_state = ctx.logic_state.read();
//     if scene_arc.should_parallelize() {
//       scene_arc.query3_mut_par::<TransformComponent, crate::scene::AlmanacPlanet, crate::scene::KinematicComponent, _>(
//         &ctx.thread_pool,
//         |entity_id, transform, planet, kinematic| {
//           // Look up rotational model from PhysicalMeshComponent (if present on this entity)
//           let rot_model = scene_arc.with_component::<crate::scene::PhysicalMeshComponent, _, _>(
//             entity_id,
//             |pmc| pmc.rotational_model,
//           ).flatten();
//           let _ = planet.step(
//             transform,
//             Some(kinematic),
//             current_epoch,
//             step_days,
//             &logic_state.almanac_data,
//             rot_model.as_ref(),
//           );
//         },
//       );
//       scene_arc.query3_mut_par::<crate::scene::HighResTransformComponent, crate::scene::AlmanacPlanet, crate::scene::CameraComponent, _>(
//         &ctx.thread_pool,
//         |_, transform, planet, _| {
//           let _ = planet.step_high_res(
//             transform,
//             None,
//             current_epoch,
//             step_days,
//             &logic_state.almanac_data,
//           );
//         },
//       );
//     } else {
//       scene_arc.query3_mut::<TransformComponent, crate::scene::AlmanacPlanet, crate::scene::KinematicComponent, _>(
//         |entity_id, transform, planet, kinematic| {
//           // Look up rotational model from PhysicalMeshComponent (if present on this entity)
//           let rot_model = scene_arc.with_component::<crate::scene::PhysicalMeshComponent, _, _>(
//             entity_id,
//             |pmc| pmc.rotational_model,
//           ).flatten();
//           let _ = planet.step(
//             transform,
//             Some(kinematic),
//             current_epoch,
//             step_days,
//             &logic_state.almanac_data,
//             rot_model.as_ref(),
//           );
//         },
//       );
//       scene_arc.query3_mut::<crate::scene::HighResTransformComponent, crate::scene::AlmanacPlanet, crate::scene::CameraComponent, _>(
//         |_, transform, planet, _| {
//           let _ = planet.step_high_res(
//             transform,
//             None,
//             current_epoch,
//             step_days,
//             &logic_state.almanac_data,
//           );
//         },
//       );
//     }
//   } // logic_state read guard dropped here
//
//   // ── CPU-side particle emission + aging ──────────────────────────────────────
//   // Emission must use SIMULATION microseconds, not wall-clock microseconds.
//   //
//   // Bug (before this fix):
//   //   gpu_sub_steps = ceil(fixed_dt_us_wallclock / max_sub_dt_us_simtime)
//   //                 = ceil(16_666 / 100_000_000) = 1   ← always 1, wrong
//   //   sub_dt_us (wall-clock) → dt_s = 0.016 s
//   //   emit_count = 0.5 p/s × 0.016 s = 0.008 particles/sub-step  ← 86400× too few
//   //
//   // Fix:
//   //   total_sim_dt_us = step_days × 86400 × 1_000_000  (same formula as GPU tasklet)
//   //   gpu_sub_steps   = ceil(total_sim_dt_us / max_sub_dt_us)   ← correct ~15 for OneDay
//   //   sub_sim_dt_us   = total_sim_dt_us / gpu_sub_steps         ← ~96 s per sub-step
//   //   emit_count      = 0.5 p/sim-s × 96 s = 48  ← correct
//   {
//     // Total simulation time for this dispatch in microseconds.
//     let total_sim_dt_us =
//       (step_days * 86400.0 * 1_000_000.0) as aethervk_oshal_rlib::os::time::timeus_t;
//     // Number of GPU sub-steps the physics tasklet will run (same divisor it uses).
//     let gpu_sub_steps = ((total_sim_dt_us + max_sub_dt_us - 1) / max_sub_dt_us).max(1) as u32;
//     // Cap emission sub-steps to bound CPU aging cost.
//     // The GPU still runs `gpu_sub_steps` for physics accuracy.
//     // We spread emission over `n_emit_steps` evenly-spaced positions instead,
//     // which is enough for a visually smooth tail while keeping O(n_emit × alive) aging.
//     // With MAX_EMIT_STEPS=8 and ~200k alive particles: 8×200k×5ns ≈ 8ms per dispatch.
//     const MAX_EMIT_STEPS: u32 = 8;
//     let n_emit_steps = gpu_sub_steps.min(MAX_EMIT_STEPS);
//     // Emission sub-step covers the full dispatch time divided by the capped step count.
//     let emit_sub_dt_us = (total_sim_dt_us / n_emit_steps as i64).max(1);
//     // One-shot diagnostic: prints emission parameters on the very first dispatch.
//     {
//       use core::sync::atomic::{AtomicBool, Ordering};
//       static EMIT_DIAG_DONE: AtomicBool = AtomicBool::new(false);
//       if !EMIT_DIAG_DONE.swap(true, Ordering::Relaxed) {
//         aethervk_oshal_rlib::log!(
//           "[EMIT-DIAG] step_days={:.6}  total_sim_dt_s={:.2}  max_sub_dt_s={:.2}  \
//            gpu_sub_steps={}  n_emit_steps={}  emit_sub_dt_s={:.2}  \
//            expected_particles_per_emit_sub(rate=2.0)={:.1}  total_per_dispatch={:.0}",
//           step_days,
//           total_sim_dt_us as f64 / 1_000_000.0,
//           max_sub_dt_us as f64 / 1_000_000.0,
//           gpu_sub_steps,
//           n_emit_steps,
//           emit_sub_dt_us as f64 / 1_000_000.0,
//           2.0_f64 * (emit_sub_dt_us as f64 / 1_000_000.0),
//           2.0_f64 * (emit_sub_dt_us as f64 / 1_000_000.0) * n_emit_steps as f64,
//         );
//       }
//     }
//     for sub in 0..n_emit_steps {
//       let sub_tick_seed = current_epoch.to_tai_seconds().to_bits()
//         ^ (fixed_dt_us as u64)
//         ^ (sub as u64).wrapping_mul(0x9e3779b97f4a7c15);
//       emit_particles_from_circles(
//         scene_arc.as_ref(),
//         sub_tick_seed,
//         emit_sub_dt_us,
//         sub,
//         n_emit_steps,
//       );
//     }
//   }
//
//   if let Some(ps_lock) = &physics_scene_arc {
//     let ps_arc = ps_lock.clone();
//     let scene_clone = scene_arc.clone();
//     let pool_clone = ctx.thread_pool.clone();
//     let kernels_arc = ctx.kernels.clone();
//     let sync_info_clone = scene_read.latest_physics_sync.clone();
//     let (engine_type, collisions_enabled) = (
//       *scene_read.physics_engine_type.read(),
//       scene_read.collisions_enabled.load(core::sync::atomic::Ordering::Relaxed),
//     );
//
//     let task = ctx
//       .thread_pool
//       .spawn_tasklet(None, move || {
//         let mut ps = ps_arc.write();
//         let dt_us = (step_days * 86400.0 * 1_000_000.0) as aethervk_oshal_rlib::os::time::timeus_t;
//
//         let res = match engine_type {
//           crate::simulation_api::structs::PhysicsEngineType::CpuSimd => {
//             let kernels = crate::physics::cpu_kernels::CpuSimdKernels {
//               thread_pool: pool_clone.clone(),
//             };
//             crate::gpu_backends::simulation_step(
//               &kernels,
//               &mut ps,
//               scene_clone.as_ref(),
//               0,
//               dt_us,
//               collisions_enabled,
//               max_sub_dt_us,
//             )
//           }
//           crate::simulation_api::structs::PhysicsEngineType::CpuScalar => {
//             let kernels = crate::physics::cpu_kernels::CpuScalarKernels {};
//             crate::gpu_backends::simulation_step(
//               &kernels,
//               &mut ps,
//               scene_clone.as_ref(),
//               0,
//               dt_us,
//               collisions_enabled,
//               max_sub_dt_us,
//             )
//           }
//           crate::simulation_api::structs::PhysicsEngineType::VulkanCompute => {
//             let mut executed = Ok(None);
//             let kernels_enum = kernels_arc.read();
//             if let crate::simulation_api::structs::KernelsEnum::VulkanCompute(
//               weak_front,
//               dev_handle,
//             ) = &*kernels_enum
//             {
//               if let Some(front) = weak_front.as_frontend() {
//                 let _ = front.with_device(*dev_handle, |device_dyn| {
//                   if let Some(vulkan_dev) = device_dyn
//                     .as_any()
//                     .downcast_ref::<crate::gpu_backends::vulkan::device::Device>(
//                   ) {
//                     executed = crate::gpu_backends::simulation_step(
//                       vulkan_dev,
//                       &mut ps,
//                       scene_clone.as_ref(),
//                       0,
//                       dt_us,
//                       collisions_enabled,
//                       max_sub_dt_us,
//                     );
//                   }
//                   Ok(())
//                 });
//               }
//             }
//             executed
//           }
//           #[cfg(test)]
//           crate::simulation_api::structs::PhysicsEngineType::Mock(target) => {
//             let mut executed = Ok(None);
//             let kernels_enum = kernels_arc.read();
//             if let crate::simulation_api::structs::KernelsEnum::VulkanCompute(
//               weak_front,
//               dev_handle,
//             ) = &*kernels_enum
//             {
//               if let Some(front) = weak_front.as_frontend() {
//                 let _ = front.with_device(*dev_handle, |device_dyn| {
//                   if let Some(vulkan_dev) = device_dyn
//                     .as_any()
//                     .downcast_ref::<crate::gpu_backends::vulkan::device::Device>(
//                   ) {
//                     let mock_kernels =
//                       crate::gpu_backends::vulkan::mock_kernels::MockVulkanKernels {
//                         base: vulkan_dev,
//                         target,
//                         scene_id,
//                       };
//                     executed = crate::gpu_backends::simulation_step(
//                       &mock_kernels,
//                       &mut ps,
//                       scene_clone.as_ref(),
//                       0,
//                       dt_us,
//                       collisions_enabled,
//                       max_sub_dt_us,
//                     );
//                   }
//                   Ok(())
//                 });
//               }
//             }
//             executed
//           }
//         };
//
//         crate::physics::handoff::SpheresOfInfluenceSystem::process_handoffs_par(
//           scene_clone.as_ref(),
//           &pool_clone,
//         );
//
//         if let Err(e) = &res {
//           aethervk_oshal_rlib::log!("Physics tasklet failed internally: {:?}", e);
//         }
//
//         if let Ok(Some(sync)) = &res {
//           *sync_info_clone.write() = Some(*sync);
//         }
//
//         res
//       })
//       .map_err(|e| <aethervk_oshal_rlib::os::NativeError as Into<EngineError>>::into(e))?;
//
//     *scene_read.active_physics_task.lock() = Some(task);
//   }
//
//   let dt_seconds = fixed_dt_us as f32 / 1_000_000.0;
//   if scene_arc.should_parallelize() {
//     scene_arc.query1_mut_par::<crate::scene::script_components::UpdateComponent, _>(
//       &ctx.thread_pool,
//       |entity_id, update_comp| {
//         (update_comp.callback)(
//           entity_id,
//           scene_arc.as_ref(),
//           &mut update_comp.entities,
//           &mut update_comp.arbitrary_data,
//           update_comp.user_data.as_mut(),
//           dt_seconds,
//         );
//       },
//     );
//   } else {
//     scene_arc.query1_mut::<crate::scene::script_components::UpdateComponent, _>(
//       |entity_id, update_comp| {
//         (update_comp.callback)(
//           entity_id,
//           scene_arc.as_ref(),
//           &mut update_comp.entities,
//           &mut update_comp.arbitrary_data,
//           update_comp.user_data.as_mut(),
//           dt_seconds,
//         );
//       },
//     );
//   }
//
//   Ok(())
// }

fn try_snap_entity(
  snap_entity: EntityId,
  target_entity: EntityId,
  scene_read: &RwLockReadGuard<SceneContext>,
) -> EngineResult<()> {
  let (target_pos, target_rot) =
    {
      // Use f64 for position if available
      let t = scene_read.scene.global_transform_f64(target_entity).ok_or(
        EngineError::InvalidOperation(
          "logic_thread:SnapToEntity | snap target doesn't have TransformComponent",
        ),
      )?;
      (t.position, t.rotation)
    };

  // Update f64 component (source of truth for camera/cursor)
  let _ = scene_read.scene.with_component_mut(
    snap_entity,
    |h: &mut crate::scene::HighResTransformComponent| {
      h.position = target_pos;
      h.rotation = target_rot;
    },
  );

  // For entities with TransformComponent (non-camera), update via set_global_position_and_rotation
  let _ =
    scene_read
      .scene
      .set_global_position_and_rotation(snap_entity, target_pos.to_f32(), target_rot);
  if let Some(ext_id) =
    scene_read.entity_map.iter().find(|&(_, v)| *v == snap_entity).map(|(k, _)| *k)
  {
    scene_read.mark_component_changed(
      ext_id,
      <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
    );
    scene_read.mark_component_changed(
      ext_id,
      <crate::scene::TransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
    );
  }
  // Return Ok even if set_global_position_and_rotation fails (camera has no Transform)
  Ok(())
}

/// Utilities for logic thread module
mod utils {
  use super::*;
  use crate::gpu::{compute_push_constants::*, new_particles::MAX_CHUNKS};
  use crate::gpu_backends::vulkan::device::{Device, particles::PushConstantMutUnion};
  use crate::scene::particles::v2::{ParticleSystemEmitParams, emit_push_constants_from_params};
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
    scaled_time_since_last_emission_us: timeus_t,
    scaled_time_since_start_epoch_us: timeus_t,
  ) -> NewParticlesEmitPushConstants {
    let mut res = emit_push_constants_from_params(
      &psep,
      mean_intra_grains_distance_mm,
      min_cumulated_mass_g,
      r_helio_au,
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
    res.delta_pos = [delta_pos_m.x(), delta_pos_m.y(), delta_pos_m.z(), 0];
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
