//! logic_thread module.

#[cfg(debug_assertions)]
pub static DEBUG_LOGIC_THREAD_TIME_MS: core::sync::atomic::AtomicU64 =
  core::sync::atomic::AtomicU64::new(0);

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
    external_state::{CAlamanacImported, CModelImported, CTimeRange, ExternalState},
    structs::{
      self, CartesianState, LogicCommand, LogicThreadContext, LogicWorkload, PhysicsDeviceSelfSync,
      SceneContext,
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
    NativeError, ThreadingError, fs,
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
    | LogicCommand::UpdateTrajectoryForSpk { .. }
    | LogicCommand::BuildCometTrajectory { .. } => true,
    _ => false,
  }
}

fn logic_command_desc(cmd: &LogicCommand) -> alloc::string::String {
  match cmd {
    LogicCommand::UpdateCometNucleusRadius { radius_km, .. } => {
      alloc::format!("Update Comet Nucleus Radius to {radius_km}")
    }
    LogicCommand::CleanupParticleSystem { entity_id, .. } => {
      alloc::format!("Cleanup Particle System {}", entity_id)
    }
    LogicCommand::RemoveParticleSystem { entity_id, .. } => {
      alloc::format!("Remove Particle System {}", entity_id)
    }
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
    LogicCommand::PlaySceneToEnd { .. } => "Play Scene to End".to_string(),
    LogicCommand::PauseScene { .. } => "Pause Scene".to_string(),
    LogicCommand::PlayScene { .. } => "Play Scene".to_string(),
    LogicCommand::SnapshotScene { .. } => "Snapshot Scene".to_string(),
    LogicCommand::RestoreSnapshot { .. } => "Restore Snapshot".to_string(),
    LogicCommand::ResetSimulation { .. } => "Reset Simulation".to_string(),
    LogicCommand::SeekEpoch { epoch, .. } => alloc::format!("Seek to {epoch}"),
    LogicCommand::DumpScene { scene_id, .. } => alloc::format!("Dump Scene {}", scene_id),
    LogicCommand::RestoreSceneDump { scene_id, .. } => {
      alloc::format!("Restore Scene Dump {}", scene_id)
    }

    // Data/Asset Commands
    LogicCommand::ImportModel { path, .. } => alloc::format!("Import model {}", path),
    LogicCommand::LoadAlmanac { path, .. } => alloc::format!("Load almanac {}", path),
    LogicCommand::UnloadAlmanac { path, .. } => alloc::format!("Unload almanac {}", path),

    // Trajectory
    LogicCommand::UpdateTrajectoryForSpk { spk_id, .. } => {
      alloc::format!("Update trajectory for SPK {}", spk_id)
    }
    LogicCommand::BuildCometTrajectory { spk_id, .. } => {
      alloc::format!("Build comet trajectory for SPK {}", spk_id)
    }

    // Comet lifecycle
    LogicCommand::TryInitComet { spk_id, .. } => {
      alloc::format!("Try init comet SPK {}", spk_id)
    }
    LogicCommand::CleanupComet { .. } => "Cleanup comet".to_string(),

    // Animation commands
    LogicCommand::AnimateCameraTo { camera_id, .. } => {
      alloc::format!("Animate camera {} to target", camera_id)
    }
    LogicCommand::SetCameraTransform { camera_id, .. } => {
      alloc::format!("SetCameraTransform camera={}", camera_id)
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
    let mut last_gpu_recycled_val: u64 = 0;

    loop {
      let mut core_logic = || -> bool {
        // Single clock_gettime per outer iteration — reused for all purposes below.
        // In debug+Linux builds this also feeds the debug_perf call-rate tracer.
        #[cfg(all(debug_assertions, target_os = "linux"))]
        use crate::simulation_api::debug_perf::traced_get_monotonic_time as get_monotonic_time_outer;
        #[cfg(not(all(debug_assertions, target_os = "linux")))]
        use oshal::os::time::get_monotonic_time as get_monotonic_time_outer;

        #[cfg(debug_assertions)]
        let _tick_start;

        let mut processed_any = false;

        // perform compute discard every 500ms
        let now = get_monotonic_time_outer();

        #[cfg(debug_assertions)]
        { _tick_start = now; }

        let cpu_submit_val = context.kernels.0.with_device(context.kernels.1, |dyn_device| {
            let vulkan_device: &vulkan::device::Device = dyn_device.as_any().downcast_ref().unwrap();
            Ok(vulkan_device.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed))
        }).unwrap_or(0);

        let submit_pressure = cpu_submit_val.saturating_sub(last_gpu_recycled_val);
        // Trigger a recycle if either:
        // 1. 500ms elapsed AND there is un-recycled work in flight.
        // 2. Submit pressure is high (e.g., >= 16 submits in-flight), risking pool exhaustion (max pools is 24).
        let time_exceeded = now - last_discard_unscaled_us > DISCARD_DELTA_UNSCALED_US;
        let pressure_exceeded = submit_pressure >= 16;

        if (time_exceeded && submit_pressure > 0) || pressure_exceeded {
          last_discard_unscaled_us = now;
          let _ = context.kernels.0.with_device(context.kernels.1, |dyn_device| {
            let vulkan_device: &vulkan::device::Device = dyn_device.as_any().downcast_ref().unwrap();

            let gpu_timeline_val = unsafe {
                vulkan_device.device.timeline_semaphore.get_semaphore_counter_value(
                    vulkan_device.kernels.timeline
                )
            }.unwrap_or(last_gpu_recycled_val);

            // Update our cached recycled value
            last_gpu_recycled_val = gpu_timeline_val;

            let items = vulkan_device.kernels.discard_pool.pop_ready_items(gpu_timeline_val);
            if !items.is_empty() {
                vulkan::device::DiscardPool::destroy_items_lock_free(&vulkan_device.device, items);
            }
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
          // Reuse `now` from above — one syscall per outer loop, not one per scene.
          let last = pc.last_frame_start;
          let elapsed = now.saturating_sub(last);

          // ── Physics tick (only when previous step is complete) ────────────
          let physics_done = {
            let scenes = context.scenes.read();
            let opt = utils::self_sync_do_if_done(
              &scenes,
              *scene_id,
              context.kernels.0.clone(),
              context.kernels.1,
              &context.render_tx,
              now,
              elapsed,
              |_, _, _| (),
            );
            if opt.is_some() {
              true
            } else {
              // If there's no active physics task, we consider physics 'done' so the simulation can begin.
              if let Some(scene_arc) = scenes.get(&scene_id) {
                !scene_arc.read().active_physics_task.load(core::sync::atomic::Ordering::Relaxed)
              } else {
                false
              }
            }
          };

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
            let is_running = scene_arc.read().simulation_running.load(core::sync::atomic::Ordering::Acquire);

            let has_physics_work = {
              let mut time_mgr = unsafe { scenes.time_managers.get_mut(&scene_id).unwrap_unchecked() };
              let scaled_fixed_dt_us = time_mgr.state.read().speed.scaled_from_unscaled(crate::simulation_api::structs::UNSCALED_FIXED_DELTA_US);
              scaled_fixed_dt_us > 0 && time_mgr.has_ready_step(scaled_fixed_dt_us) && is_running
            };

            // Not gated on `physics_done`: the SPICE step and the cartesian cache commit must run
            // even while the compute queue is stalled. Only dust emission needs the GPU.
            let phase1_res: Option<EngineResult<_>> = if !end_epoch_reached && has_physics_work {
              let run = |vulkan_device: Option<&vulkan::device::Device>| {
                let mut time_mgr =
                  unsafe { scenes.time_managers.get_mut(&scene_id).unwrap_unchecked() };
                execute_simulation_tick_fixed_update_phase(
                  vulkan_device,
                  *scene_id,
                  scene_arc.upgradable_read(),
                  &mut time_mgr,
                  structs::UNSCALED_FIXED_DELTA_US,
                  &scenes.cartesian_state_cache,
                  &context.logic_state.read().almanac_data,
                )
              };
              let mut ran = None;
              if physics_done {
                let _ = context.kernels.0.with_device(context.kernels.1, |dyn_device| {
                  ran = Some(run(dyn_device.as_any().downcast_ref::<vulkan::device::Device>()));
                  Ok(())
                });
              }
              Some(ran.unwrap_or_else(|| run(None)))
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
                latest_physics_sync: None,
                did_physics_work: false,
              }),
              Some(Ok(s)) => {
                use core::sync::atomic::Ordering;

                if let Some(sync) = s.latest_physics_sync.clone() {
                  let mut scene_write = scene_arc.write();
                  scene_write.latest_physics_sync = Some(sync);
                  scene_write.active_physics_task.store(true, Ordering::Relaxed);
                }

                // self sync function put this to false. Since we executed correctly a physics
                // step, put this to true
                // scene_arc.read().active_physics_task.store(true, Ordering::Relaxed);

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
          if elapsed >= pc.target_frame_time {
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

                  // We can't know the final task_id yet because the render thread hasn't
                  // processed the frame. Instead, we just pass the feedback Arc directly
                  // to the scene context. The cross-sync guard will poll it later.
                  if let Some(scene_arc) = context.scenes.read().get(&scene_ids[idx]) {
                    scene_arc
                      .write()
                      .last_render_task = alloc::sync::Arc::clone(task_id);
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

        #[cfg(debug_assertions)]
        {
          let elapsed_ms = (oshal::os::time::get_monotonic_time() - _tick_start) as f64 / 1000.0;
          crate::simulation_api::logic_thread::DEBUG_LOGIC_THREAD_TIME_MS
            .store(elapsed_ms.to_bits(), core::sync::atomic::Ordering::Relaxed);
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

      // ── Earth trajectory coverage ────────────────────────────────────────────────────────
      // Rebuild the Earth orbit trajectory when the committed range is not inside the span it
      // covers (at least one orbit from start, extended to end for multi-year ranges).
      if crate::simulation_api::reposition::needs_earth_orbit_rebuild(
        scene_guard.earth_orbit_coverage,
        start,
        end,
      ) {
        if let Some(earth) = scene_guard.earth {
          // Only rebuild if Earth is driven by almanac (AlmanacPlanet attached)
          let has_planet =
            scene_guard.scene.with_component(earth.body, |_: &AlmanacPlanet| ()).is_some();
          if has_planet {
            let (traj_start, traj_end) =
              crate::simulation_api::reposition::earth_orbit_span_tai(start, end);
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
            scene_arc.write().earth_orbit_coverage = Some((traj_start, traj_end));
          }
        }
      }
      Ok(())
    }
    LogicCommand::TryInitComet {
      scene_id,
      spk_id,
      proposed_start,
      proposed_end,
      keplerian_elements,
      reference_mode,
    } => {
      let logic_state = ctx.logic_state.read();
      let planet = AlmanacPlanet::new(spk_id);

      // 1. Dry run validation: ensure both start and end can be evaluated
      aethervk_oshal_rlib::log!("============================================================");
      aethervk_oshal_rlib::log!(
        "=== TryInitComet: Evaluating Start Epoch for SPK {} ===",
        spk_id
      );
      aethervk_oshal_rlib::log!("============================================================");
      if let Err(e) = planet.step(proposed_start, &logic_state.almanac_data, None) {
        emit_breadcrumb(
          3,
          &alloc::format!("[TryInitComet] Validation failed for start epoch: {}", e),
        );
        crate::simulation_api::emit_external_state_change(&ExternalState::CometInitialized(
          crate::simulation_api::external_state::CCometInitialized::new(false, spk_id),
        ));
        return Ok(());
      }

      aethervk_oshal_rlib::log!("============================================================");
      aethervk_oshal_rlib::log!(
        "=== TryInitComet: Evaluating End Epoch for SPK {} ===",
        spk_id
      );
      aethervk_oshal_rlib::log!("============================================================");
      if let Err(e) = planet.step(proposed_end, &logic_state.almanac_data, None) {
        emit_breadcrumb(
          3,
          &alloc::format!("[TryInitComet] Validation failed for end epoch: {}", e),
        );
        crate::simulation_api::emit_external_state_change(&ExternalState::CometInitialized(
          crate::simulation_api::external_state::CCometInitialized::new(false, spk_id),
        ));
        return Ok(());
      }
      drop(logic_state);

      // 2. Spawn Async Trajectory Job (Phase 1 math)
      let start_sec = proposed_start.to_tai_seconds();
      let end_sec = proposed_end.to_tai_seconds();
      let workload = Box::new(structs::LogicWorkload {
        cmd: LogicCommand::BuildCometTrajectory {
          scene_id,
          spk_id,
          start_epoch_tai_sec: start_sec,
          end_epoch_tai_sec: end_sec,
          sample_step_days: 1.0,
          keplerian_elements,
          reference_mode,
        },
        ctx: alloc::sync::Arc::clone(ctx),
      });
      let _ = ctx.thread_pool.scatter(alloc::vec![workload]);

      Ok(())
    }

    LogicCommand::CleanupComet { scene_id } => {
      // Remove AlmanacPlanet and TrajectoryComponent from the comet subtree,
      // then reset the comet subtree to its 1 AU +X default position.
      let scenes = ctx.scenes.read();
      let scene_arc =
        crate::expect_scene!(scenes.get_scene(scene_id), "CleanupComet: scene not found");
      let scene_guard = scene_arc.read();

      let comet = scene_guard.comet.ok_or(EngineError::InvalidOperation(
        "CleanupComet: comet SubtreeEntities not populated",
      ))?;

      let _ = scene_guard.scene.remove_component::<AlmanacPlanet>(comet.body);
      let _ = scene_guard
        .scene
        .remove_component::<crate::scene::BodyRotationalModel>(comet.body);
      let _ = scene_guard
        .scene
        .remove_component::<crate::scene::trajectory::TrajectoryComponent>(comet.orbit);

      // Hide comet body components
      let _ = scene_guard.scene.with_component_mut(
        comet.body,
        |mesh: &mut crate::scene::StaticMeshComponent| {
          mesh.is_visible = false;
        },
      );
      let _ = scene_guard.scene.with_component_mut(
        comet.body,
        |gizmo: &mut crate::scene::SphereGizmoComponent| {
          gizmo.is_visible = false;
        },
      );

      // Remove all jet entities (children of the comet body). A camera parented to the comet
      // (CometOrbiting mode) is not a jet: move it to root keeping its world transform.
      if let Some(children) = scene_guard.scene.get_children(comet.body) {
        for child in children {
          if scene_guard
            .scene
            .with_component(child, |_: &crate::scene::CameraComponent| ())
            .is_some()
          {
            scene_guard.scene.unparent_to_root_preserving_world(child);
            scene_guard.mark_component_changed(
              child.as_ffi(),
              <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
            );
          } else {
            scene_guard.scene.remove_entity(child);
          }
        }
      }
      utils::clear_effective_trajectory(&scene_guard);
      // Stop the SPICE step from moving the comet back to its SPK position on the next tick.
      for id in [comet.body, comet.subtree] {
        scenes
          .cartesian_state_cache
          .remove(&crate::simulation_api::structs::SceneEntityId::new(
            scene_id, id,
          ));
      }

      // Reset subtree to 1 AU +X (default "no SPK" placement)
      let _ = scene_guard
        .scene
        .with_component_mut(comet.subtree, |t: &mut TransformComponent| {
          t.position =
            aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 0.0, 0.0);
        });
      // Reset body to local origin
      let _ = scene_guard.scene.with_component_mut(comet.body, |t: &mut TransformComponent| {
        t.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero();
      });

      drop(scene_guard);
      {
        let mut w = scene_arc.write();
        w.comet_orbit_year = None;
        w.comet_reference_elements = None;
      }

      Ok(())
    }

    LogicCommand::AnimateCameraTo {
      scene_id,
      camera_id,
      target_pos,
      target_rot,
      duration_s,
      orbit_pivot,
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
          // If this is a mode switch (duration >= 1.0s), override the speed-preserved duration
          // so the camera doesn't get stuck for thousands of seconds.
          if duration_s >= 1.0 {
            anim.duration = duration_s;
          }
          anim.orbit_pivot = orbit_pivot;
        },
      );

      if retargeted.is_none() {
        // Read global transform so start_pos is safely in AU, regardless of whether
        // the camera was currently parented to the comet (km) or root (AU).
        let (mut start_pos, start_rot) =
          if let Some(global_t) = scene.scene.global_transform_f64(cam_int) {
            (global_t.position, global_t.rotation)
          } else {
            return Ok(()); // Safety fallback
          };

        let start_rot = crate::scene::animation::strip_roll(start_rot);

        // Convert absolute start position into an offset from the pivot so LERP works flawlessly
        if let Some(pivot) = orbit_pivot {
          let pivot_id_bits = pivot.x().to_bits();
          if pivot_id_bits != 0 {
            let pivot_ent = EntityId::from_ffi(pivot_id_bits);
            if let Some(pivot_t) =
              utils::global_transform_f64_with_overrides(&scene.scene, pivot_ent, |_| None)
            {
              start_pos -= pivot_t.position;
            }
          }
        }

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
            orbit_pivot,
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
    LogicCommand::SetCameraTransform {
      scene_id,
      camera_id,
      transform,
      projection,
    } => {
      let scenes = ctx.scenes.read();
      let scene_arc = crate::expect_scene!(scenes.get_scene(scene_id), "SetCameraTransform");
      let scene = scene_arc.read();

      let cam_int = scene.get_entity(camera_id).ok_or(EngineError::InvalidOperation(
        "SetCameraTransform | camera entity not found",
      ))?;

      if let Some((pos, rot)) = transform {
        scene.scene.set_global_transform_f64(cam_int, pos, rot)?;
        let _ = scene
          .scene
          .remove_component::<crate::scene::animation::TransformAnimationComponent>(cam_int);

        scene.mark_component_changed(
          camera_id,
          <HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }

      if let Some(proj) = projection {
        scene
          .scene
          .with_component_mut(cam_int, |c: &mut CameraComponent| {
            c.projection = proj;
          })
          .ok_or(EngineError::InvalidOperation(
            "SetCameraTransform | camera has no CameraComponent",
          ))?;

        scene.mark_component_changed(
          camera_id,
          <crate::scene::CameraComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
        );
      }

      Ok(())
    }
    LogicCommand::SnapshotScene {
      scene_id,
      done_flag,
    } => {
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
            let mut cloned_scene = (*scene_write.scene).clone();
            // dust: the snapshot keeps the batch list (emission is deterministic); by the time it
            // is restored the GPU ring content is stale, so it will be re-emitted
            cloned_scene.query1_mut(
              |_, comp: &mut crate::scene::particles::ParticleSystemComponent| {
                comp.dust.get_mut().invalidate_gpu();
              },
            );
            scene_write.scene_snapshot = Some(alloc::boxed::Box::new(cloned_scene));
            let _ = vulkan_device;

            // Snapshot the TimeManager state
            let time_state = scenes.time_managers.get(&scene_id).unwrap().state.read().clone();
            scene_write.time_snapshot = Some(alloc::boxed::Box::new(time_state));
          },
        ) {
          done_flag.store(true, core::sync::atomic::Ordering::Release);
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
    LogicCommand::RestoreSnapshot {
      scene_id,
      done_flag,
    } => {
      use oshal::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      // Retry stopping the current task for a deadline of 10s.
      // restore_particles performs actual GPU work (staging buffer copies), which can take
      // several seconds on a cold GPU (e.g. first frame with generate_sky still in flight).
      // self_sync_do_if_done blocks inside the closure until that GPU work completes, so the
      // outer loop must accommodate the full GPU copy duration.
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 10_000_000_i64 {
        let (now, elapsed) = {
          let time_mgr =
            scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };
        if utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(), // render frontend is an arc
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _render_tx| {
            // 1 Wait for the render thread to be idle so that we can overwrite the front buffer for
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

            // - take scene overrides (BodyRotationalModel)
            // 3 Restore the snapshot
            if let Some(snapshot) = scene_write.scene_snapshot.take() {
              scene_write.scene = snapshot.into();
            }

            // Restore TimeManager state
            if let Some(ts) = scene_write.time_snapshot.take() {
              let time_mgr = scenes.time_managers.get(&scene_id).unwrap();
              *time_mgr.state.write() = *ts;
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
            utils::mark_all_serializable_as_changed(scene_write);
          },
        )
        .is_some()
        {
          done_flag.store(true, core::sync::atomic::Ordering::Release);
          return Ok(());
        }
      }

      Err(EngineError::InvalidOperation("Failed to restore snapshot"))
    }

    LogicCommand::SeekEpoch {
      scene_id,
      epoch,
      done_flag,
      succeeded,
    } => {
      use oshal::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      let Some((reached, start_epoch, now, elapsed, scaled_us)) =
        scenes.time_managers.get_mut(&scene_id).map(|mut tm| {
          let reached = tm.seek(epoch);
          let st = tm.state.read();
          (
            reached,
            tm.start_epoch,
            st.unscaled_time,
            st.unscaled_delta,
            st.scaled_time,
          )
        })
      else {
        succeeded.store(false, core::sync::atomic::Ordering::Relaxed);
        done_flag.store(true, core::sync::atomic::Ordering::Release);
        return Err(EngineError::InvalidOperation("SeekEpoch: no scene"));
      };

      let mut applied = false;
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 1_000_000_i64 {
        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _| {
            let logic_state = ctx.logic_state.read();
            utils::apply_seek(
              vulkan_device,
              scene_write,
              scene_id,
              &scenes.cartesian_state_cache,
              &logic_state.almanac_data,
              reached,
              start_epoch,
              scaled_us,
            );
            utils::mark_all_serializable_as_changed(scene_write);
          },
        ) {
          applied = true;
          break;
        }
      }

      succeeded.store(applied, core::sync::atomic::Ordering::Relaxed);
      done_flag.store(true, core::sync::atomic::Ordering::Release);
      if applied {
        Ok(())
      } else {
        Err(EngineError::InvalidOperation(
          "SeekEpoch: GPU sync timed out, seek not applied",
        ))
      }
    }

    LogicCommand::ResetSimulation {
      scene_id,
      done_flag,
      succeeded,
    } => {
      use oshal::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();

      let (now, elapsed, start_epoch) = {
        let mut time_mgr = scenes.time_managers.get_mut(&scene_id).unwrap();
        let mut state = time_mgr.state.write();

        // Zero out the clock to snap epoch back to start_epoch
        state.scaled_time = 0;
        state.scaled_accumulator = 0;

        (
          state.unscaled_time,
          state.unscaled_delta,
          time_mgr.start_epoch,
        )
      };

      let mut reset_applied = false;
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 1_000_000_i64 {
        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _| {
            // 1. Wait for render thread idle
            let last_render_task =
              scene_write.last_render_task.load(core::sync::atomic::Ordering::Acquire);
            if last_render_task != 0 {
              let w_start = get_monotonic_time();
              while get_monotonic_time() - w_start < 500_000_i64 {
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                core::hint::spin_loop();
              }
            }

            // 2-3. dust reset: forget every cluster
            scene_write.scene.query1_mut(
              |_, comp: &mut crate::scene::particles::ParticleSystemComponent| {
                comp.dust.get_mut().reset();
              },
            );

            // 3b. the recorded comet path belongs to the previous run
            utils::clear_effective_trajectory(scene_write);

            // 4. Evict cached SPICE state so interpolation restarts clean
            let mut keys = alloc::vec::Vec::with_capacity(128);
            scenes.cartesian_state_cache.iter().for_each(|kv_ref| keys.push(*kv_ref.key()));
            for key in keys {
              if key.scene_id == scene_id {
                scenes.cartesian_state_cache.remove(&key);
              }
            }

            let logic_state = ctx.logic_state.read();
            if let Some(earth) = scene_write.earth {
              if let Some(planet) = scene_write
                .scene
                .with_component(earth.body, |p: &crate::scene::AlmanacPlanet| *p)
              {
                let _ = crate::simulation_api::reposition::force_reposition(
                  &scene_write.scene,
                  earth.subtree,
                  earth.body,
                  &logic_state.almanac_data,
                  &planet,
                  start_epoch,
                );
              }
            }
            if let Some(comet) = scene_write.comet {
              if let Some(planet) = scene_write
                .scene
                .with_component(comet.body, |p: &crate::scene::AlmanacPlanet| *p)
              {
                let _ = crate::simulation_api::reposition::force_reposition(
                  &scene_write.scene,
                  comet.subtree,
                  comet.body,
                  &logic_state.almanac_data,
                  &planet,
                  start_epoch,
                );
              }
            }

            utils::mark_all_serializable_as_changed(scene_write);
          },
        ) {
          reset_applied = true;
          break;
        }
      }

      succeeded.store(reset_applied, core::sync::atomic::Ordering::Relaxed);
      done_flag.store(true, core::sync::atomic::Ordering::Release);
      if reset_applied {
        Ok(())
      } else {
        Err(EngineError::InvalidOperation(
          "ResetSimulation: GPU sync timed out, reset not applied",
        ))
      }
    }

    // ─── DumpScene ──────────────────────────────────────────────────────────
    LogicCommand::DumpScene { scene_id, base_dir } => {
      use crate::simulation_api::{
        emit_external_state_change,
        external_state::{CSceneDumped, ExternalState},
        structs::SceneDump,
      };
      use oshal::os::time::get_monotonic_time;

      let scenes = ctx.scenes.read();
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 500_000_i64 {
        let (now, elapsed) = {
          let tm = scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
          let s = tm.state.read();
          (s.unscaled_time, s.unscaled_delta)
        };
        if let Some(result) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _render_tx| -> EngineResult<()> {
            // 1. dust clusters are not serialized (they regrow from the restored parameters)
            let particle_snapshot = None;
            let _ = vulkan_device;

            // 2. Walk ECS
            let entities = crate::simulation_api::scene_dump::serialize_scene(&scene_write.scene);

            // 3. Epoch range from time_manager
            let (start_epoch_parts, end_epoch_parts) = {
              let tm =
                scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
              let start = tm.start_epoch.to_tdb_duration().to_parts();
              let end = tm.end_epoch.to_tdb_duration().to_parts();
              ((start.0, start.1), (end.0, end.1))
            };

            let (current_epoch_parts, compatibility) = {
              let tm =
                scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
              let now = tm.current_epoch().to_tdb_duration().to_parts();
              (
                (now.0, now.1),
                crate::simulation_api::scene_dump::compatibility_json(
                  scene_write,
                  tm.start_epoch,
                  tm.end_epoch,
                ),
              )
            };

            let dump = SceneDump {
              version: SceneDump::CURRENT_VERSION,
              scene_id,
              start_epoch_parts,
              end_epoch_parts,
              current_epoch_parts,
              compatibility,
              entities,
              particle_snapshot,
            };

            // 4. Encode
            let bytes = bincode::serde::encode_to_vec(&dump, bincode::config::standard())
              .map_err(|_| EngineError::InvalidOperation("DumpScene: bincode encode failed"))?;

            // 5. Write to <base_dir>/scene_<scene_id>/scene.bin
            let dir = alloc::format!("{}/scene_{}", base_dir, scene_id);
            fs::create_dir_all(&dir)
              .and_then(|_| fs::write(alloc::format!("{}/scene.bin", dir), bytes.as_ref()))
              .map_err(|_| EngineError::InvalidOperation("DumpScene: file write failed"))?;

            Ok(())
          },
        ) {
          let success = result.is_ok();
          emit_external_state_change(&ExternalState::SceneDumped(CSceneDumped {
            success: success as u32,
          }));
          return result;
        }
      }
      emit_breadcrumb(2, "DumpScene: self_sync timed out");
      emit_external_state_change(&ExternalState::SceneDumped(CSceneDumped { success: 0 }));
      Err(EngineError::InvalidOperation(
        "DumpScene: self_sync timed out",
      ))
    }

    // ─── RestoreSceneDump ───────────────────────────────────────────────────
    LogicCommand::RestoreSceneDump { scene_id, dump } => {
      use crate::simulation_api::{
        emit_external_state_change,
        external_state::{CSceneRestored, ExternalState},
      };
      use oshal::os::time::get_monotonic_time;

      let scenes = ctx.scenes.read();
      let start = get_monotonic_time();
      while get_monotonic_time() - start <= 500_000_i64 {
        let (now, elapsed) = {
          let tm = scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
          let s = tm.state.read();
          (s.unscaled_time, s.unscaled_delta)
        };
        if let Some(result) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _render_tx| -> EngineResult<()> {
            // 1. Wait for render thread to finish its last task
            let last_render_task =
              scene_write.last_render_task.load(core::sync::atomic::Ordering::Acquire);
            if last_render_task != 0 {
              let wait_start = get_monotonic_time();
              while get_monotonic_time() - wait_start < 500_000_i64 {
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                oshal::os::native::this_thread::sleep_for(core::time::Duration::from_micros(200));
              }
            }

            // 2. Compatibility: only a scene configured exactly like the dumped one (same range,
            //    comet, reference orbit, nucleus, jets) can take its state
            {
              use crate::simulation_api::structs::SceneDump;
              if dump.version != SceneDump::CURRENT_VERSION {
                return Err(EngineError::InvalidOperation(
                  "RestoreSceneDump: unsupported dump version",
                ));
              }
              let tm =
                scenes.time_managers.get(&scene_id).ok_or(EngineError::InvalidNullArgument)?;
              let live = crate::simulation_api::scene_dump::compatibility_json(
                scene_write,
                tm.start_epoch,
                tm.end_epoch,
              );
              if live != dump.compatibility {
                emit_breadcrumb(2, "Saved state does not match the current configuration");
                return Err(EngineError::InvalidOperation(
                  "RestoreSceneDump: incompatible configuration",
                ));
              }
            }

            // 3. Overwrite ECS + resolve mesh cache (overwrite-merge by asset_path)
            let scene_mut = alloc::sync::Arc::make_mut(&mut scene_write.scene);
            let deser_result = crate::simulation_api::scene_dump::deserialize_scene(
              scene_mut,
              &dump.entities,
              &scenes.mesh_cache,
            );

            // 4. Evict stale GPU mesh + sun resources
            let _ = vulkan_device.cleanup_stale_scene_resources(
              &deser_result.new_mesh_hashes,
              &deser_result.new_sun_entity_ids,
            );

            // 5. (epoch range: identical by the compatibility check above, nothing to set. A
            //    `time_managers.get_mut` here self-deadlocked: `self_sync_do_if_done` holds a
            //    read guard on the same shard for the whole closure)

            // 6. Empty the cartesian state cache for this scene
            {
              let mut keys = alloc::vec::Vec::with_capacity(32);
              scenes.cartesian_state_cache.iter().for_each(|kv| keys.push(*kv.key()));
              for key in keys {
                if key.scene_id == scene_id {
                  scenes.cartesian_state_cache.remove(&key);
                }
              }
            }

            // 7. Back to the dumped epoch: bodies and dust rebuilt deterministically (nothing
            //    GPU-side is stored in the dump)
            let seek = scenes.time_managers.get(&scene_id).map(|tm| {
              let target = hifitime::Epoch::from_tdb_duration(hifitime::Duration::from_parts(
                dump.current_epoch_parts.0,
                dump.current_epoch_parts.1,
              ));
              let reached = tm.seek(target);
              (reached, tm.start_epoch, tm.state.read().scaled_time)
            });
            if let Some((reached, start_epoch, scaled_us)) = seek {
              let logic_state = ctx.logic_state.read();
              utils::apply_seek(
                vulkan_device,
                scene_write,
                scene_id,
                &scenes.cartesian_state_cache,
                &logic_state.almanac_data,
                reached,
                start_epoch,
                scaled_us,
              );
            }

            // 8. Mark all ForeignSerializable components as changed (notifies C# side)
            utils::mark_all_serializable_as_changed(scene_write);

            Ok(())
          },
        ) {
          let success = result.is_ok();
          emit_external_state_change(&ExternalState::SceneRestored(CSceneRestored {
            success: success as u32,
          }));
          return result;
        }
      }
      emit_breadcrumb(2, "RestoreSceneDump: self_sync timed out");
      emit_external_state_change(&ExternalState::SceneRestored(CSceneRestored { success: 0 }));
      Err(EngineError::InvalidOperation(
        "RestoreSceneDump: self_sync timed out",
      ))
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
        && let Some(pos) = scene_read
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
          let _ = scene_read.scene.with_component_mut(camera_entity, |c: &mut CameraComponent| {
            c.focus_distance = (dist as f32).max(0.000001);
          });
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
        && let Some(pos) =
          scene_read.scene.with_component(sun_id, |t: &TransformComponent| t.position)
      {
        cursor_pos = pos;
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
          t.rotation = aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(q);
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
          t.position += translation.to_f64();

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
        && let Some(_) =
          scene_write
            .scene
            .with_component_mut(cursor_id, |c: &mut HighResTransformComponent| {
              c.position = target_pos_dvec;
            })
      {
        // Mark cursor entity as changed.
        utils::mark_component_changed::<HighResTransformComponent>(&scene_write, cursor_id);
      }

      // TODO maybe: Dynamic offset calculation based on object bounds and camera FOV
      let target_radius = 0.1_f64;

      let mut fov = core::f64::consts::FRAC_PI_4;
      let mut aspect = 16.0 / 9.0;
      if let Some(cam) = scene_write.scene.with_component(snap_entity, |c: &CameraComponent| *c)
        && let crate::scene::CameraProjection::Perspective {
          fov: cam_fov,
          aspect_ratio,
          ..
        } = cam.projection
      {
        fov = cam_fov as f64;
        aspect = aspect_ratio as f64;
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
        (
          t.0.to_f64(),
          aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(t.1),
        )
      } else {
        return Ok(());
      };

      let anim = crate::scene::animation::TransformAnimationComponent {
        start_pos,
        start_rot,
        target_pos: target_pos_dvec + offset.to_f64(),
        target_rot: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(q),
        duration: 2.0,
        elapsed: 0.0,
        is_finished: false,
        orbit_pivot: None,
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
      let naif_id = ctx.load_almanac_file_internal(&path)?;
      emit_external_state_change(&ExternalState::AlmanacImported(
        CAlamanacImported::new_loaded(naif_id, &path),
      ));
      Ok(())
    }
    LogicCommand::UnloadAlmanac { task_id: _, path } => {
      ctx.unload_almanac_file_internal(&path)?;
      emit_external_state_change(&ExternalState::AlmanacImported(
        CAlamanacImported::new_unloaded(&path),
      ));
      Ok(())
    }

    LogicCommand::BuildCometTrajectory {
      scene_id,
      spk_id,
      start_epoch_tai_sec,
      end_epoch_tai_sec,
      sample_step_days: _,
      keplerian_elements,
      reference_mode,
    } => {
      if start_epoch_tai_sec >= end_epoch_tai_sec {
        crate::simulation_api::emit_external_state_change(&ExternalState::CometInitialized(
          crate::simulation_api::external_state::CCometInitialized::new(false, spk_id),
        ));
        return Err(EngineError::InvalidOperation(
          "BuildCometTrajectory: start_epoch >= end_epoch",
        ));
      }

      let scenes = ctx.scenes.read();

      // Lock order: never acquire `logic_state` while holding a SceneContext lock. The logic
      // thread holds SceneContext (upgradable) while reading `logic_state`; with a queued
      // `LoadAlmanac` writer and a queued scene writer that closes a cycle. So: read what we need
      // from the scene, release it, step the almanac, then lock the scene again for the commit.
      let planet = AlmanacPlanet::new(spk_id);
      let start = anise::time::Epoch::from_tai_seconds(start_epoch_tai_sec);
      let start_state = {
        let rot_model = scenes.get(&scene_id).and_then(|sc| {
          let g = sc.read();
          g.comet
            .and_then(|c| g.scene.with_component(c.body, |m: &BodyRotationalModel| *m))
        });
        let logic_state = ctx.logic_state.read();
        planet.step_with_velocity(start, &logic_state.almanac_data, rot_model.as_ref())
      };

      // Horizons SPK segments only interpolate from a few minutes after the requested start day, so
      // a midnight start epoch fails: osculate at the first covered instant instead.
      let reosculate =
        reference_mode == crate::simulation_api::structs::ReferenceOrbitMode::OsculatingAtStart;
      let (osc_epoch, osc_state) = if reosculate {
        let logic_state = ctx.logic_state.read();
        let data = &logic_state.almanac_data;
        let end = anise::time::Epoch::from_tai_seconds(end_epoch_tai_sec);
        let osc_epoch = utils::osculation_epoch(start, end, |t| {
          planet.step_with_velocity(t, data, None).is_ok()
        })
        .unwrap_or(start);
        (
          osc_epoch,
          Some(planet.step_with_velocity(osc_epoch, data, None)),
        )
      } else {
        (start, None)
      };
      if let Some(Err(e)) = &osc_state {
        aethervk_oshal_rlib::log!(
          "[BuildCometTrajectory] re-osculation failed at {osc_epoch}: {e:?}; using SBDB elements"
        );
        emit_breadcrumb(
          2,
          "Could not re-osculate the reference orbit from the SPK; the red track uses the SBDB elements",
        );
      }

      // Reference orbit: SBDB elements are osculating at their own epoch (67P: 2015-10-10, 0.12 AU
      // off the 2025 path); re-osculating from the SPK state at the start epoch is exact there.
      let keplerian_elements = match &osc_state {
        Some(Ok((r, v, _))) => {
          use aethervk_oshal_rlib::math::vector::Vector3;
          let jd = osc_epoch.to_jde_tdb_days();
          let el = crate::simulation::orbit_elements::elements_from_state(
            [r.x(), r.y(), r.z()],
            [v.x(), v.y(), v.z()],
            crate::simulation::orbit_elements::MU_SUN_KM3_S2,
            jd,
          );
          aethervk_oshal_rlib::log!(
            "[BuildCometTrajectory] re-osculated at {}: e={:.6} q={:.6} AU (SBDB e={:.6} q={:.6})",
            osc_epoch,
            el.eccentricity,
            el.perihelion_distance_au,
            keplerian_elements.eccentricity,
            keplerian_elements.perihelion_distance_au
          );
          el
        }
        _ => keplerian_elements,
      };

      aethervk_oshal_rlib::log!(
        "[BuildCometTrajectory] Starting Keplerian track for SPK {} | e={:.4} q={:.4} AU",
        spk_id,
        keplerian_elements.eccentricity,
        keplerian_elements.perihelion_distance_au
      );
      let scene_ctx =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let scene_guard = scene_ctx.read();

      let comet = scene_guard.comet.ok_or(EngineError::InvalidOperation(
        "BuildCometTrajectory: comet SubtreeEntities not populated",
      ))?;

      // ── Analytical Keplerian orbit track (SBDB elements, ecliptic J2000 = scene frame) ──

      let e = keplerian_elements.eccentricity;
      const N_SEGMENTS: usize = utils::KEPLER_TRACK_SEGMENTS;
      let control_points = utils::keplerian_track_bezier_au_f64(&keplerian_elements);

      aethervk_oshal_rlib::log!(
        "[BuildCometTrajectory] Keplerian track: {} control points ({} segments, e={:.4})",
        control_points.len(),
        N_SEGMENTS,
        e
      );

      if control_points.len() < 8 {
        emit_breadcrumb(
          3,
          "[BuildCometTrajectory] Trajectory generation failed: not enough control points",
        );
        crate::simulation_api::emit_external_state_change(&ExternalState::CometInitialized(
          crate::simulation_api::external_state::CCometInitialized::new(false, spk_id),
        ));
        return Err(EngineError::InvalidOperation(
          "BuildCometTrajectory: not enough control points",
        ));
      }

      let new_comp = crate::scene::trajectory::TrajectoryComponent::from_f64(
        control_points,
        [1.0, 0.2, 0.2, 1.0], // Red color for comet
        2.0,
        0,
        32,
      );

      // We have successfully computed trajectory, now commit everything to ECS!
      let _ = scene_guard.scene.add_component(comet.body, planet);
      // a new comet: the path recorded for the previous one is meaningless
      utils::clear_effective_trajectory(&scene_guard);

      // Make comet body visible again
      let _ = scene_guard.scene.with_component_mut(
        comet.body,
        |mesh: &mut crate::scene::StaticMeshComponent| {
          mesh.is_visible = true;
        },
      );
      let _ = scene_guard.scene.with_component_mut(
        comet.body,
        |gizmo: &mut crate::scene::SphereGizmoComponent| {
          gizmo.is_visible = true;
        },
      );

      let mut replaced = false;
      let _ = scene_guard.scene.with_component_mut(
        comet.orbit,
        |comp: &mut crate::scene::trajectory::TrajectoryComponent| {
          *comp = new_comp.clone();
          replaced = true;
        },
      );
      if !replaced {
        let _ = scene_guard.scene.add_component(comet.orbit, new_comp);
      }

      aethervk_oshal_rlib::log!(
        "[BuildCometTrajectory] ECS commit done: AlmanacPlanet + TrajectoryComponent applied for comet.body ext_id={} (SPK {})",
        comet.body.as_ffi(),
        spk_id
      );

      // Force reposition using SPK data (body position only, not track), stepped above
      match start_state {
        Ok((position_km, _velocity, rotation)) => {
          crate::simulation_api::reposition::apply_reposition(
            &scene_guard.scene,
            comet.subtree,
            comet.body,
            position_km,
            rotation,
          )
        }
        Err(ref e) => aethervk_oshal_rlib::log!("[BuildCometTrajectory] SPK step failed: {e}"),
      }

      // Emit the post-commit comet position to C# immediately.
      if let Some(global_t) = scene_guard.scene.global_transform_f64(comet.body) {
        let pos_au = global_t.position;
        aethervk_oshal_rlib::log!(
          "[BuildCometTrajectory] force_reposition done. Emitting CometPositionSnapshot @ ({:.6}, {:.6}, {:.6}) AU for SPK {}",
          pos_au.x(),
          pos_au.y(),
          pos_au.z(),
          spk_id
        );
        let snapshot = crate::simulation_api::external_state::CCometPositionSnapshot {
          spk_id,
          _pad: 0,
          pos_x: pos_au.x(),
          pos_y: pos_au.y(),
          pos_z: pos_au.z(),
        };
        crate::simulation_api::emit_external_state_change(&ExternalState::CometPositionSnapshot(
          snapshot,
        ));
      } else {
        aethervk_oshal_rlib::log!(
          "[BuildCometTrajectory] WARNING: global_transform_f64 returned None for comet.body after force_reposition (SPK {})",
          spk_id
        );
      }

      drop(scene_guard);
      scene_ctx.write().comet_reference_elements = Some(keplerian_elements);

      aethervk_oshal_rlib::log!(
        "[BuildCometTrajectory] Emitting CometInitialized(true) for SPK {}",
        spk_id
      );
      crate::simulation_api::emit_external_state_change(&ExternalState::CometInitialized(
        crate::simulation_api::external_state::CCometInitialized::new(true, spk_id),
      ));

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
      // TODO check why we are using internal id in the command here.
      let entity = EntityId::from(slotmap::KeyData::from_ffi(entity_id));

      // Note: trajectory control points are computed in heliocentric SUN_ECLIPJ2000 AU.
      // Orbit entities (Earth_orbit, Comet_orbit) are direct children of root_entity
      // (depth_layer=0) so the renderer's RTE path applies no AU_TO_KM scale distortion.
      // This structural invariant is enforced at construction time in create_subtree;
      // no runtime parent check is needed here.

      use utils::SampledPoint as SampledPoints;
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
          time_sec: t,
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

      // almanac no longer needed: release it before locking the scene (lock order, see
      // BuildCometTrajectory)
      drop(logic_state);
      let control_points = crate::scene::trajectory::bezier_from_samples_au_f64(&samples);
      let scene_guard = scene_ctx.read();

      // Apply the component to the entity
      let new_comp = crate::scene::trajectory::TrajectoryComponent::from_f64(
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
      let mut sun_pos = aethervk_oshal_rlib::math::vector::vec3f64::DVec3::zero();
      if let Some((sun_id, _)) = scene_guard
        .scene
        .query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
        && let Some(pos) = { scene_guard.scene.global_transform_f64(sun_id) }.map(|t| t.position)
      {
        sun_pos = pos;
      }

      let mut handled_highres = false;
      let _ = scene_guard.scene.with_component_mut(
        entity,
        |transform: &mut HighResTransformComponent| {
          transform.position = sun_pos;
          handled_highres = true;
        },
      );

      if !handled_highres {
        let mut handled_transform = false;
        let _ =
          scene_guard
            .scene
            .with_component_mut(entity, |transform: &mut TransformComponent| {
              transform.position = sun_pos.to_f32();
              handled_transform = true;
            });
        if !handled_transform {
          let new_transform = crate::scene::TransformComponent {
            position: sun_pos.to_f32(),
            ..Default::default()
          };
          let _ = scene_guard.scene.add_component(entity, new_transform);
        }
      }

      Ok(())
    }

    LogicCommand::UpdateCometNucleusRadius {
      scene_id,
      radius_km,
    } => {
      let scenes = ctx.scenes.read();
      let scene_arc = crate::expect_scene!(
        scenes.get_scene(scene_id),
        "UpdateCometNucleusRadius: scene not found"
      );
      let scene_guard = scene_arc.read();

      let comet = scene_guard.comet.ok_or(EngineError::InvalidOperation(
        "UpdateCometNucleusRadius: comet SubtreeEntities not populated",
      ))?;

      // ── 1. Wireframe sphere gizmo → 2× nucleus radius ────────────────────────
      let _ = scene_guard.scene.with_component_mut(
        comet.body,
        |gizmo: &mut crate::scene::SphereGizmoComponent| {
          gizmo.radius = radius_km * 2.0;
        },
      );

      // ── 2. CPU mesh: rescale vertex positions in-place ────────────────────────
      // We deliberately do NOT bump `mesh.id` so that the `physical_mesh2_resources`
      // DashMap cache key stays stable and no duplicate Vulkan resource is created.
      // The GPU-side buffer swap is driven by `enqueue_mesh_position_update` below.
      let comet_mesh_id = scene_guard.scene.with_component_mut(
        comet.body,
        |mesh_comp: &mut crate::scene::StaticMeshComponent| {
          let original_id = mesh_comp.mesh.id;
          if let Some(m) = alloc::sync::Arc::get_mut(&mut mesh_comp.mesh) {
            // Fast path: we are the sole owner — mutate in place, zero allocation.
            crate::simulation::comet::update_uv_sphere_radius_in_place(m, radius_km);
          } else {
            // Fallback: someone else holds a reference (e.g. a render thread read).
            // Clone, mutate, and preserve the original id so the GPU cache key is unchanged.
            let mut copy = (*mesh_comp.mesh).clone();
            crate::simulation::comet::update_uv_sphere_radius_in_place(&mut copy, radius_km);
            copy.id = original_id; // critical: must not change the cache key
            mesh_comp.mesh = alloc::sync::Arc::new(copy);
          }
          original_id
        },
      );

      // ── 3. Update all Jet Previews ────────────────────────
      let parent_scale = scene_guard
        .scene
        .with_component(comet.body, |t: &crate::scene::TransformComponent| {
          t.scale.x()
        })
        .unwrap_or(1.0);
      let local_r = radius_km / parent_scale;
      let desired_global_radius = radius_km / 50.0;
      let local_scale = desired_global_radius / parent_scale;

      if let Some(children) = scene_guard.scene.get_children(comet.body) {
        for child in children {
          if scene_guard.scene.has_component::<crate::scene::ParticleSystemComponent>(child)
            == crate::scene::HasComponentResultEnum::EntityHasComponent
          {
            let _ = scene_guard.scene.with_component_mut(
              child,
              |t: &mut crate::scene::TransformComponent| {
                let mut dir = t.position.normalize();
                if dir.x().is_nan() {
                  dir = Vec3f32::from_components(1.0, 0.0, 0.0);
                }
                t.position = dir * local_r;
                t.scale = Vec3f32::from_components(local_scale, local_scale, local_scale);
              },
            );
          }
        }
      }

      // ── 4. Enqueue GPU position buffer swap for the next frame ────────────────
      // `flush_pending_mesh_updates` is called at the top of `build_render_scene`
      // and records vkCmdCopyBuffer + TRANSFER→VERTEX barrier on the main graphics
      // command buffer, then swaps the handle in `physical_mesh2_resources`.
      if let Some(mesh_id) = comet_mesh_id {
        // Collect the flat position array from the (now-updated) CPU mesh.
        let position_data: alloc::vec::Vec<f32> = scene_guard
          .scene
          .with_component(
            comet.body,
            |mesh_comp: &crate::scene::StaticMeshComponent| {
              mesh_comp
                .mesh
                .vertices
                .iter()
                .flat_map(|v| v.position.iter().copied())
                .collect()
            },
          )
          .unwrap_or_default();

        let _ = ctx.kernels.0.with_device(ctx.kernels.1, |device| {
          device.enqueue_mesh_position_update(mesh_id, position_data);
          Ok(())
        });
      }

      aethervk_oshal_rlib::log!(
        "[UpdateCometNucleusRadius] radius={:.2} km → gizmo={:.2} km, mesh queued for GPU swap",
        radius_km,
        radius_km * 2.0,
      );

      Ok(())
    }

    LogicCommand::CleanupParticleSystem {
      scene_id,
      entity_id,
      done_flag,
    } => {
      use aethervk_oshal_rlib::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      let start = get_monotonic_time();

      while get_monotonic_time() - start <= 1_000_000_i64 {
        let (now, elapsed) = {
          let time_mgr = scenes.time_managers.get(&scene_id).unwrap();
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };

        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _| {
            let last_render_task =
              scene_write.last_render_task.load(core::sync::atomic::Ordering::Acquire);
            if last_render_task != 0 {
              let wait_start = get_monotonic_time();
              while get_monotonic_time() - wait_start < 500_000_i64 {
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                core::hint::spin_loop();
              }
            }

            let _ = scene_write.scene.with_component_mut(
              slotmap::KeyData::from_ffi(entity_id).into(),
              |c: &mut crate::scene::particles::ParticleSystemComponent| {
                c.dust.get_mut().reset();
              },
            );
          },
        ) {
          break;
        }
      }
      done_flag.store(true, core::sync::atomic::Ordering::Release);
      Ok(())
    }

    LogicCommand::RemoveParticleSystem {
      scene_id,
      entity_id,
      done_flag,
    } => {
      use aethervk_oshal_rlib::os::time::get_monotonic_time;
      let scenes = ctx.scenes.read();
      let start = get_monotonic_time();

      while get_monotonic_time() - start <= 1_000_000_i64 {
        let (now, elapsed) = {
          let time_mgr = scenes.time_managers.get(&scene_id).unwrap();
          let state = time_mgr.state.read();
          (state.unscaled_time, state.unscaled_delta)
        };

        if let Some(_) = utils::self_sync_do_if_done(
          &scenes,
          scene_id,
          ctx.kernels.0.clone(),
          ctx.kernels.1,
          &ctx.render_tx,
          now,
          elapsed,
          |vulkan_device, scene_write, _| {
            let last_render_task =
              scene_write.last_render_task.load(core::sync::atomic::Ordering::Acquire);
            if last_render_task != 0 {
              let wait_start = get_monotonic_time();
              while get_monotonic_time() - wait_start < 500_000_i64 {
                if vulkan_device.is_task_completed(last_render_task).unwrap_or(true) {
                  break;
                }
                core::hint::spin_loop();
              }
            }

            // the component's resource Arc discards the GPU buffers when dropped
            scene_write
              .scene
              .remove_component::<crate::scene::particles::ParticleSystemComponent>(
                slotmap::KeyData::from_ffi(entity_id).into(),
              );
          },
        ) {
          break;
        }
      }
      done_flag.store(true, core::sync::atomic::Ordering::Release);
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
  latest_physics_sync: Option<PhysicsDeviceSelfSync>,
  did_physics_work: bool,
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
      self.gfx_release_sync_info.as_slice(),
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
        self.gfx_release_sync_info.as_slice(),
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
///
/// `vulkan_device` is `Some` only when the previous compute submission completed (`physics_done`).
/// The SPICE step and the commit of the cartesian cache into the scene never depend on the GPU:
/// with `None` (or a failing command buffer) only the dust emission of this tick is skipped, so the
/// comet keeps following its SPK kernel even while the compute queue is stalled.
fn execute_simulation_tick_fixed_update_phase(
  vulkan_device: Option<&crate::gpu_backends::vulkan::device::Device>,
  scene_id: u64,
  mut scene: parking_lot::lock_api::RwLockUpgradableReadGuard<parking_lot::RawRwLock, SceneContext>,
  time_mgr: &mut oshal::os::time::v2::TimeManager,
  unscaled_fixed_delta_us: oshal::os::time::timeus_t,
  cartesian_state_cache: &dashmap::DashMap<
    crate::simulation_api::structs::SceneEntityId,
    CartesianState,
  >,
  almanac: &AlmanacPackedData,
) -> EngineResult<SimulationTickOutput> {
  static CAPTURE_COMPUTE_ONCE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(true);
  use crate::simulation_api::structs::SceneEntityId;
  use oshal::os::time::v2::SimSpeed;
  let scaled_fixed_dt_us =
    time_mgr.state.read().speed.scaled_from_unscaled(unscaled_fixed_delta_us);

  // ------------------------------------------------------------------------------------
  // -- Fixed Update Phase: consume accumulated time, SPICE spk_ezr step at the current epoch,
  // dust emission on the compute queue (GPU permitting), commit of the cartesian cache --
  // ------------------------------------------------------------------------------------
  let (sim_speed, now_unscaled_us, now_scaled_us) = {
    let time_state = time_mgr.state.read();
    (
      time_state.speed,
      time_state.unscaled_time,
      time_state.scaled_time,
    )
  };
  let latest_physics_sync = if sim_speed != SimSpeed::Paused {
    // used for SPICE EZR Data Kernel and simulation
    let current_epoch = time_mgr.current_epoch();

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel: insert every almanac-driven body missing from the cache
    // ------------------------------------------------------------------------------------
    utils::insert_almanac_bodies_into_cache(&scene, scene_id, cartesian_state_cache);

    // ------------------------------------------------------------------------------------
    // Fixed Update Loop
    // ------------------------------------------------------------------------------------
    // Spiral of death prevention: set a max number of physics execution steps
    const MAX_PHYSICS_STEPS_PER_FRAME: u32 = 10;
    let mut steps_executed = 0;

    if scaled_fixed_dt_us > 0 {
      while time_mgr.consume_fixed_step(scaled_fixed_dt_us) {
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

        // dust v3 needs no per-substep work: clusters are evaluated in closed form per frame
      } // end of accumulator fixed update loop
    }

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel for Comet Cartesian state update.
    // Note: Outside of the fixed update accumulator, still inside physics update.
    // ------------------------------------------------------------------------------------
    utils::step_cartesian_cache(cartesian_state_cache, scene_id, current_epoch, almanac);
    utils::record_effective_trajectory(&scene, scene_id, cartesian_state_cache, current_epoch);

    // ------------------------------------------------------------------------------------
    // Dust v3: emission (GPU only). Its failure must not prevent the commit below.
    // ------------------------------------------------------------------------------------
    let gpu_sync: EngineResult<Option<PhysicsDeviceSelfSync>> = match vulkan_device {
      Some(device) => utils::emit_and_submit_dust(
        device,
        &scene.scene,
        scene_id,
        cartesian_state_cache,
        almanac,
        time_mgr.start_epoch,
        now_scaled_us,
        scene.comet_reference_elements,
        CAPTURE_COMPUTE_ONCE.swap(false, core::sync::atomic::Ordering::Relaxed),
      )
      .map(Some),
      None => Ok(None),
    };

    // ------------------------------------------------------------------------------------
    // SPICE EZR Kernel: Commit Comet Cartesian state update from dashmap to scene
    // ------------------------------------------------------------------------------------
    {
      // 1. upgrade to a write lock to start applying updates
      let mut scene_write = parking_lot::RwLockUpgradableReadGuard::upgrade(scene);

      let mut start_time_unscaled_us = get_monotonic_time();
      for (idx, kv_ref) in cartesian_state_cache
        .iter()
        .filter(|kv| kv.key().scene_id == scene_id)
        .enumerate()
      {
        let key = kv_ref.key();
        let state = kv_ref.value();
        let entity_id = EntityId::from_ffi(key.entity_id);
        if let Some(ref body_state) = state.comet_state {
          let old_rot = scene_write
            .scene
            .with_component(entity_id, |t: &TransformComponent| t.rotation)
            .unwrap();
          let new_rot = body_state.transform.rotation;

          // comet/planet, update its trasform
          scene_write
            .scene
            .with_component_mut(entity_id, |t: &mut TransformComponent| {
              *t = body_state.transform;
            })
            .unwrap();

          if !body_state
            .body_rotational_model
            .as_ref()
            .map_or(true, |m| m.body_fixed_orientation)
          {
            let mut cameras = alloc::vec::Vec::new();
            scene_write.scene.query1(|cam_id, _: &crate::scene::CameraComponent| {
              let mut is_desc = false;
              let mut curr = scene_write.scene.get_parent(cam_id);
              while let Some(p) = curr {
                if p == entity_id {
                  is_desc = true;
                  break;
                }
                curr = scene_write.scene.get_parent(p);
              }
              if is_desc {
                cameras.push(cam_id);
              }
            });

            for cam_id in cameras {
              scene_write.scene.with_component_mut(
                cam_id,
                |h: &mut crate::scene::HighResTransformComponent| {
                  utils::compensate_body_spin(h, old_rot, new_rot)
                },
              );
            }
          }
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
            // Yield the write lock briefly so readers (render thread) can proceed.
            scene = parking_lot::RwLockWriteGuard::downgrade_to_upgradable(scene_write);
            oshal::os::native::this_thread::sleep_for(core::time::Duration::from_micros(50));
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

    // the transforms are committed: now surface a GPU failure of this tick, if any
    Some((gpu_sync?, steps_executed))
  } else {
    None
  };

  Ok(SimulationTickOutput {
    latest_physics_sync: latest_physics_sync.as_ref().and_then(|(s, _)| s.clone()),
    did_physics_work: latest_physics_sync.map(|(_, steps)| steps > 0).unwrap_or(false),
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

      // Position: if orbit_pivot encodes an entity ID (bits stored in x), resolve the
      // entity's current world position and add the lerped offset to it every frame.
      // This gives zero-latency tracking — the entity position is read from the same
      // physics frame that's being rendered.  When no entity is encoded, fall back to
      // a plain world-space LERP (start_pos → target_pos).
      let new_pos_dvec = if let Some(pivot) = anim.orbit_pivot {
        let pivot_id_bits = pivot.x().to_bits();
        if pivot_id_bits != 0 {
          let pivot_ent = EntityId::from_ffi(pivot_id_bits);
          let lerped_offset = DVec3::lerp(anim.start_pos, anim.target_pos, smooth_t as f64);
          if let Some(pivot_t) =
            utils::global_transform_f64_with_overrides(&scene.scene, pivot_ent, |_| None)
          {
            pivot_t.position + lerped_offset
          } else {
            lerped_offset
          }
        } else {
          DVec3::lerp(anim.start_pos, anim.target_pos, smooth_t as f64)
        }
      } else {
        DVec3::lerp(anim.start_pos, anim.target_pos, smooth_t as f64)
      };

      // Rotation: enforce slerp_constrained unconditionally for all camera animations
      // so the camera's up-vector never crosses the global equator (no pole flip or unwanted roll).
      let new_rot =
        crate::scene::animation::slerp_constrained(anim.start_rot, anim.target_rot, smooth_t);

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
    let has_high_res = scene.scene.has_component::<HighResTransformComponent>(id);
    let has_standard = scene.scene.has_component::<TransformComponent>(id);

    // Apply the global animation target using safe hierarchy solvers
    if has_high_res.into() {
      let _ = scene.scene.set_global_transform_f64(id, new_pos_dvec, new_rot);
      utils::mark_component_changed::<HighResTransformComponent>(&scene, id);
    } else if has_standard.into() {
      let _ =
        scene
          .scene
          .set_global_position_and_rotation(id, new_pos_dvec.to_f32(), new_rot.to_quat());
      utils::mark_component_changed::<TransformComponent>(&scene, id);
    }
  }

  // -- Reference-position error annotations (also while paused) --
  utils::update_reference_errors(&scene, time_mgr.current_epoch());
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

  let mut cameras_to_mark = alloc::vec::Vec::new();

  let mut comet_changed = false;
  if let Some(comet) = &scene.comet {
    if scene.changed_entities.read().contains_key(&comet.body.as_ffi()) {
      comet_changed = true;
    }
  }

  let mut earth_changed = false;
  if let Some(earth) = &scene.earth {
    if scene.changed_entities.read().contains_key(&earth.body.as_ffi()) {
      earth_changed = true;
    }
  }

  if comet_changed || earth_changed {
    scene.scene.query1(|e_id, _: &crate::scene::CameraComponent| {
      let hierarchy = scene.scene.hierarchy.read();
      let mut current = e_id;
      while let Some(&parent) = hierarchy.parents.get(&current) {
        let is_comet = comet_changed && scene.comet.as_ref().is_some_and(|c| c.body == parent);
        let is_earth = earth_changed && scene.earth.as_ref().is_some_and(|e| e.body == parent);

        if is_comet || is_earth {
          cameras_to_mark.push(e_id);
          break;
        }
        current = parent;
      }
    });
  }

  for cam in cameras_to_mark {
    utils::mark_component_changed::<crate::scene::HighResTransformComponent>(&scene, cam);
  }

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
/// See `utils::update_reference_errors` (exposed for the FFI toggle, which refreshes immediately).
pub(crate) fn update_reference_errors(scene_ctx: &SceneContext, epoch: anise::time::Epoch) {
  utils::update_reference_errors(scene_ctx, epoch)
}

mod utils {
  use super::*;
  use crate::{
    gpu::{RenderDeviceHandle, RenderFrontend},
    gpu_backends::vulkan::device::Device,
    scene::ForeignSerializable,
    simulation_api::structs::{RenderCommand, SimulationSceneData},
  };

  /// Records the dust v3 emissions of this tick on a fresh compute command buffer and submits it.
  /// Truth source for the jet = almanac comet state of this tick (f64, already stepped in the
  /// cartesian cache) + the jet offset rotated by the body rotation, the same one used by the
  /// per-frame evaluation, so clusters and comet stay consistent to df64 precision.
  pub fn emit_and_submit_dust(
    vulkan_device: &Device,
    scene: &crate::scene::Scene,
    scene_id: u64,
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
    almanac: &AlmanacPackedData,
    start_epoch: anise::time::Epoch,
    now_scaled_us: timeus_t,
    reference: Option<crate::simulation_api::structs::KeplerianElements>,
    capture_this_tick: bool,
  ) -> EngineResult<PhysicsDeviceSelfSync> {
    use crate::gpu_backends::vulkan::device::QueueRole;
    let (cmd_handle, cmd) = vulkan_device.get_command_buffer_and_native_all(QueueRole::Compute)?;
    let mut cmd_scope = ScopedComputeCommand::new(vulkan_device, cmd_handle, cmd)?;

    let dust_systems = record_dust_emissions(
      vulkan_device,
      cmd,
      scene,
      scene_id,
      cartesian_state_cache,
      almanac,
      start_epoch,
      now_scaled_us,
      reference,
    );

    #[cfg(debug_assertions)]
    if capture_this_tick {
      unsafe {
        crate::gpu_backends::vulkan::renderdoc::start_frame_capture(
          core::ptr::null_mut(),
          core::ptr::null_mut(),
        );
        aethervk_oshal_rlib::log!("[RenderDoc] Triggered manual compute queue capture");
      }
    }
    #[cfg(not(debug_assertions))]
    let _ = capture_this_tick;

    let (compute_semaphore, compute_signal_value) = cmd_scope.submit()?;
    // batches recorded above become drawable once the render submit waits on this value
    for ps_id in &dust_systems {
      let _ = scene.with_component(
        *ps_id,
        |ps: &crate::scene::particles::ParticleSystemComponent| {
          ps.dust.lock().mark_submitted(compute_signal_value);
        },
      );
    }

    #[cfg(debug_assertions)]
    if capture_this_tick {
      unsafe {
        crate::gpu_backends::vulkan::renderdoc::end_frame_capture(
          core::ptr::null_mut(),
          core::ptr::null_mut(),
        );
      }
    }

    Ok(PhysicsDeviceSelfSync::new(
      compute_semaphore,
      compute_signal_value,
    ))
  }

  pub use crate::scene::trajectory::{
    TrajectorySample as SampledPoint, bezier_from_samples_au as samples_to_bezier_au,
  };

  /// Appends the comet SPK state of this tick (already stepped in the cartesian cache) to the
  /// `effective_comet_trajectory` entity and rebuilds its yellow `TrajectoryComponent` when a sample
  /// was taken (every `EffectiveTrajectoryComponent::SAMPLE_INTERVAL_SEC` of sim time).
  pub fn record_effective_trajectory(
    scene_ctx: &SceneContext,
    scene_id: u64,
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
    epoch: anise::time::Epoch,
  ) {
    use crate::scene::trajectory::{EffectiveTrajectoryComponent, TrajectoryComponent};
    let (Some(entity), Some(comet)) = (scene_ctx.effective_comet_trajectory, scene_ctx.comet)
    else {
      return;
    };
    let Some((position_km, velocity_km)) = cartesian_state_cache
      .get(&crate::simulation_api::structs::SceneEntityId::new(
        scene_id, comet.body,
      ))
      .and_then(|c| c.comet_state.as_ref().and_then(|b| b.helio_state_km))
    else {
      return;
    };
    let sample = SampledPoint {
      position_km,
      velocity_km,
      time_sec: epoch.to_tai_seconds(),
    };

    let scene = &scene_ctx.scene;
    // the head moves every tick (the renderer draws last sample -> head), samples are gated
    let changed = scene
      .with_component_mut(entity, |c: &mut EffectiveTrajectoryComponent| {
        c.head = Some(sample);
        c.push_sample(sample)
      })
      .unwrap_or_else(|| {
        let mut c = EffectiveTrajectoryComponent::default();
        c.head = Some(sample);
        c.push_sample(sample);
        let _ = scene.add_component(entity, c);
        true
      });
    if !changed {
      return;
    }
    // present from the first sample (zero segments of its own; the head segment is appended by
    // the renderer)
    let Some(t) = scene.with_component(entity, |c: &EffectiveTrajectoryComponent| c.trajectory())
    else {
      return;
    };
    if scene
      .with_component_mut(entity, |c: &mut TrajectoryComponent| *c = t.clone())
      .is_none()
    {
      let _ = scene.add_component(entity, t);
    }
  }

  /// Refreshes the "reference-position error" annotations of the committed comet, every loop (also
  /// while paused): the cross-track line to the closest point of the red reference track and the
  /// same-epoch line to the reference's two-body position now. Each is shown only while enabled
  /// and longer than 10 nucleus radii.
  pub fn update_reference_errors(scene_ctx: &SceneContext, epoch: anise::time::Epoch) {
    use crate::{
      scene::{
        HiddenComponent, SphereGizmoComponent,
        trajectory::{
          ScreenMeasurementComponent, TrajectoryComponent, closest_point_on_bezier_track, format_km,
        },
      },
      simulation::orbit_elements::{MU_SUN_KM3_S2, state_from_elements},
      simulation_api::reposition::{AU_TO_KM, KM_TO_AU},
    };
    use aethervk_oshal_rlib::math::vector::{Vector3, vec3f64::DVec3};
    let Some([cross_e, epoch_e]) = scene_ctx.reference_error_entities else {
      return;
    };
    let scene = &scene_ctx.scene;
    let set = |e: EntityId, data: Option<(DVec3, DVec3, alloc::string::String)>| match data {
      Some((from, to, label)) => {
        let _ = scene.with_component_mut(e, |m: &mut ScreenMeasurementComponent| {
          m.from_au = from;
          m.to_au = to;
          m.label = label;
        });
        let _ = scene.remove_component::<HiddenComponent>(e);
      }
      None => {
        if scene.with_component(e, |_: &HiddenComponent| ()).is_none() {
          let _ = scene.add_component(e, HiddenComponent {});
        }
      }
    };

    let committed = scene_ctx
      .comet
      .filter(|c| scene.with_component(c.body, |_: &AlmanacPlanet| ()).is_some());
    let comet_au = committed.and_then(|c| scene.global_transform_f64(c.body)).map(|g| g.position);
    let (Some(comet), Some(comet_au), true) = (
      committed,
      comet_au,
      scene_ctx.reference_error_enabled.load(core::sync::atomic::Ordering::Relaxed),
    ) else {
      set(cross_e, None);
      set(epoch_e, None);
      return;
    };
    let radius_km = scene
      .with_component(comet.body, |s: &SphereGizmoComponent| s.radius as f64 * 0.5)
      .unwrap_or(0.0);
    let threshold_km = 10.0 * radius_km;
    let p = [comet_au.x(), comet_au.y(), comet_au.z()];

    let cross = scene
      .with_component(comet.orbit, |t: &TrajectoryComponent| {
        closest_point_on_bezier_track(&t.control_points_f64, p)
      })
      .flatten()
      .and_then(|(q, d_au)| {
        let km = d_au * AU_TO_KM;
        (km > threshold_km).then(|| {
          (
            comet_au,
            DVec3::from_components(q[0], q[1], q[2]),
            alloc::format!("cross-track {}", format_km(km)),
          )
        })
      });
    set(cross_e, cross);

    let same_epoch = scene_ctx
      .comet_reference_elements
      .and_then(|el| state_from_elements(&el, MU_SUN_KM3_S2, epoch.to_jde_tdb_days()))
      .and_then(|(r, _)| {
        let ref_au = DVec3::from_components(r[0], r[1], r[2]) * KM_TO_AU;
        let km = (ref_au - comet_au).length() * AU_TO_KM;
        (km > threshold_km).then(|| {
          (
            comet_au,
            ref_au,
            alloc::format!("same-epoch {}", format_km(km)),
          )
        })
      });
    set(epoch_e, same_epoch);
  }

  /// Removes the effective trajectory (components only, the container entity stays): simulation
  /// reset, comet decommit or change.
  pub fn clear_effective_trajectory(scene_ctx: &SceneContext) {
    if let Some(entity) = scene_ctx.effective_comet_trajectory {
      let _ = scene_ctx
        .scene
        .remove_component::<crate::scene::trajectory::EffectiveTrajectoryComponent>(entity);
      let _ = scene_ctx
        .scene
        .remove_component::<crate::scene::trajectory::TrajectoryComponent>(entity);
    }
  }

  /// Segments of the analytical comet track built by [`keplerian_track_bezier_au`].
  /// 2048: the Hermite error of a cubic segment grows with (Δanomaly)⁴; with 128 segments the
  /// drawn 67P track missed the comet by 7.3 km at the osculation epoch (visible at km zoom),
  /// with 2048 by ~0.1 m.
  pub const KEPLER_TRACK_SEGMENTS: usize = 2048;

  /// Analytical orbit track from SBDB osculating elements, as cubic Bezier control points in AU
  /// (4 per segment, `[x, y, z, 1]`) in **SUN_ECLIPJ2000**, the scene frame (the SBDB elements are
  /// referred to the J2000 ecliptic, so no obliquity rotation: applying one tilted the track by
  /// 23.44° about +X away from the SPK path).
  ///
  /// Ellipse: uniform eccentric anomaly over a full turn. Hyperbola: true anomaly up to 5° short of
  /// the asymptote. Handles are exact Hermite tangents (`p ± dr/dθ · Δθ/3`), not straight chords,
  /// so the curve stays on the conic between samples (chord sag reached ~1e-3 AU at 3.5 AU).
  pub fn keplerian_track_bezier_au(
    el: &crate::simulation_api::structs::KeplerianElements,
  ) -> alloc::vec::Vec<[f32; 4]> {
    keplerian_track_bezier_au_f64(el)
      .into_iter()
      .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, 1.0])
      .collect()
  }

  /// Keeps a camera parented below a spinning body at its inertial pose: undoes the body's rotation
  /// change `old_rot → new_rot` on the camera's local transform. In f64: the f32 body rotations
  /// upcast exactly, so nothing is rounded (an f32 quaternion costs ~1e-7 rad, ~15 km at 1 AU).
  pub fn compensate_body_spin(
    h: &mut crate::scene::HighResTransformComponent,
    old_rot: aethervk_oshal_rlib::math::vector::vec4::Quat,
    new_rot: aethervk_oshal_rlib::math::vector::vec4::Quat,
  ) {
    use aethervk_oshal_rlib::math::quaternion::Quaternion;
    // world = parent · body · local must not change: local' = new⁻¹ · old · local. (The former
    // `(new · old⁻¹)⁻¹ · local` only matches when the two body rotations commute, which f32 ones
    // about the "same" axis, or a precessing/nutating Earth, do not: ~1e-9 rad per step.)
    let to_new_local = aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(new_rot)
      .conjugate()
      * aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_quat(old_rot);
    h.position = to_new_local.rotate_vector(h.position);
    h.rotation = (to_new_local * h.rotation).normalize();
  }

  /// Epoch at which the reference orbit is re-osculated: the first epoch in `[start, end]` where
  /// `steps` succeeds, to within a second. Horizons SPK segments only interpolate from a couple of
  /// minutes after the requested day (even inside the declared domain), so a midnight start epoch
  /// would otherwise fail and silently fall back to the SBDB elements.
  pub fn osculation_epoch(
    start: anise::time::Epoch,
    end: anise::time::Epoch,
    steps: impl Fn(anise::time::Epoch) -> bool,
  ) -> Option<anise::time::Epoch> {
    use anise::time::Duration;
    if steps(start) {
      return Some(start);
    }
    // exponential search for a covered epoch, then bisect the [fail, ok] bracket
    let (mut lo, mut hi) = (start, None);
    let mut dt = 1.0;
    while start + Duration::from_seconds(dt) <= end {
      let t = start + Duration::from_seconds(dt);
      if steps(t) {
        hi = Some(t);
        break;
      }
      lo = t;
      dt *= 2.0;
    }
    let mut hi = match hi {
      Some(h) => h,
      None if end > lo && steps(end) => end,
      None => return None,
    };
    while (hi - lo).to_seconds() > 1.0 {
      let mid = lo + (hi - lo) * 0.5;
      if steps(mid) { hi = mid } else { lo = mid }
    }
    Some(hi)
  }

  /// f64 version of [`keplerian_track_bezier_au`] (the `TrajectoryComponent` source of truth).
  pub fn keplerian_track_bezier_au_f64(
    el: &crate::simulation_api::structs::KeplerianElements,
  ) -> alloc::vec::Vec<[f64; 3]> {
    use crate::simulation_api::reposition::AU_TO_KM;
    use aethervk_oshal_rlib::math::vector::vec3f64::DVec3;
    use core::f64::consts::PI;
    let e = el.eccentricity;
    let q = el.perihelion_distance_au;
    let (i, om, w) = (
      el.inclination_deg.to_radians(),
      el.longitude_of_ascending_node_deg.to_radians(),
      el.argument_of_perihelion_deg.to_radians(),
    );
    // perifocal basis in the ecliptic frame (3-1-3: Ω, i, ω)
    let p_hat = DVec3::from_components(
      om.cos() * w.cos() - om.sin() * i.cos() * w.sin(),
      om.sin() * w.cos() + om.cos() * i.cos() * w.sin(),
      i.sin() * w.sin(),
    );
    let q_hat = DVec3::from_components(
      -om.cos() * w.sin() - om.sin() * i.cos() * w.cos(),
      -om.sin() * w.sin() + om.cos() * i.cos() * w.cos(),
      i.sin() * w.cos(),
    );
    let to_ecl = |x: f64, y: f64| p_hat * x + q_hat * y;

    // (theta, perifocal position, d position / d theta), in AU
    let sample = |k: usize| -> (f64, (f64, f64), (f64, f64)) {
      let t = k as f64 / KEPLER_TRACK_SEGMENTS as f64;
      if e < 1.0 {
        let a = q / (1.0 - e);
        let b = a * (1.0 - e * e).sqrt();
        let ea = 2.0 * PI * t;
        (
          ea,
          (a * (ea.cos() - e), b * ea.sin()),
          (-a * ea.sin(), b * ea.cos()),
        )
      } else {
        let nu_max = (1.0 / e).acos() - 5f64.to_radians();
        let nu = -nu_max + 2.0 * nu_max * t;
        let p = q * (1.0 + e);
        let den = 1.0 + e * nu.cos();
        let r = p / den;
        let dr = p * e * nu.sin() / (den * den);
        (
          nu,
          (r * nu.cos(), r * nu.sin()),
          (dr * nu.cos() - r * nu.sin(), dr * nu.sin() + r * nu.cos()),
        )
      }
    };
    // reuse the Hermite -> Bezier conversion (expects km and km/"s"; theta plays the time role)
    let samples: alloc::vec::Vec<SampledPoint> = (0..=KEPLER_TRACK_SEGMENTS)
      .map(sample)
      .map(|(theta, (x, y), (dx, dy))| SampledPoint {
        position_km: to_ecl(x, y) * AU_TO_KM,
        velocity_km: to_ecl(dx, dy) * AU_TO_KM,
        time_sec: theta,
      })
      .collect();
    crate::scene::trajectory::bezier_from_samples_au_f64(&samples)
  }

  /// Inserts every almanac-driven body of the scene (comets with a rotational model, planets)
  /// missing from the cartesian cache, together with its micro frame entry. Used by the fixed
  /// update and by `SeekEpoch` (which must step the cache while paused).
  pub fn insert_almanac_bodies_into_cache(
    scene: &SceneContext,
    scene_id: u64,
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
  ) {
    use crate::simulation_api::structs::SceneEntityId;
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

    #[cfg(debug_assertions)]
    fn debug_log_query4_match(e_id: EntityId) {
      use core::sync::atomic::{AtomicI64, Ordering};
      static LAST_LOG: AtomicI64 = AtomicI64::new(0);
      let now = aethervk_oshal_rlib::os::time::get_monotonic_time() as i64;
      let last = LAST_LOG.load(Ordering::Relaxed);
      if now - last >= 2_000_000 {
        if LAST_LOG
          .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
          .is_ok()
        {
          oshal::log!("query4 matched comet entity: {:?}", e_id);
        }
      }
    }
    #[cfg(not(debug_assertions))]
    #[inline(always)]
    fn debug_log_query4_match(_: EntityId) {}

    // insert into cache if absent all active comets
    scene.scene.query4(
      |e_id,
       t: &TransformComponent,
       _m: &CometMarkerComponent,
       ap: &AlmanacPlanet,
       iau_rot: &BodyRotationalModel| {
        debug_log_query4_match(e_id);
        insert_into_cache(e_id, *t, Some(*iau_rot), *ap)
      },
    );

    // insert into cache if absent all active planets (earth)
    scene.scene.query3(
      |e_id, t: &TransformComponent, _m: &PlanetMarkerComponent, ap: &AlmanacPlanet| {
        insert_into_cache(e_id, *t, None, *ap)
      },
    );
  }

  /// Brings the scene to `epoch` (the clock already moved there, `scaled_us` after
  /// `start_epoch`): fresh cartesian cache stepped and committed, effective trajectory cleared,
  /// and the last TTL of dust emitted right away from the deterministic grid (a few passes of
  /// `MAX_WINDOWS_PER_TICK`), so it also works while paused. Runs inside `self_sync_do_if_done`.
  #[allow(clippy::too_many_arguments)]
  pub fn apply_seek(
    vulkan_device: &Device,
    scene_write: &mut SceneContext,
    scene_id: u64,
    cache: &dashmap::DashMap<crate::simulation_api::structs::SceneEntityId, CartesianState>,
    almanac: &AlmanacPackedData,
    epoch: anise::time::Epoch,
    start_epoch: anise::time::Epoch,
    scaled_us: timeus_t,
  ) {
    let keys: alloc::vec::Vec<_> =
      cache.iter().map(|kv| *kv.key()).filter(|k| k.scene_id == scene_id).collect();
    for k in keys {
      cache.remove(&k);
    }
    insert_almanac_bodies_into_cache(scene_write, scene_id, cache);
    step_cartesian_cache(cache, scene_id, epoch, almanac);
    commit_cartesian_cache(scene_write, scene_id, cache);
    // the recorded comet path belongs to the previous timeline position
    clear_effective_trajectory(scene_write);
    // every age tier from scratch (shared per-tick window budget)
    let passes = crate::scene::dust::MAX_SEEK_PASSES;
    let reference = scene_write.comet_reference_elements;
    let mut last_sync = None;
    for _ in 0..passes {
      match emit_and_submit_dust(
        vulkan_device,
        &scene_write.scene,
        scene_id,
        cache,
        almanac,
        start_epoch,
        scaled_us,
        reference,
        false,
      ) {
        Ok(sync) => last_sync = Some(sync),
        Err(e) => {
          oshal::log!("[Seek] dust emission failed: {e}");
          break;
        }
      }
    }
    if let Some(sync) = last_sync {
      scene_write.latest_physics_sync = Some(sync);
      scene_write
        .active_physics_task
        .store(true, core::sync::atomic::Ordering::Release);
    }
    mark_all_serializable_as_changed(scene_write);
  }

  /// Writes the cartesian cache of `scene_id` into the scene (body local transforms, frame
  /// transforms) and marks them changed. The per-tick commit also compensates the spin of cameras
  /// parented to spinning bodies; a seek jumps, so it does not.
  pub fn commit_cartesian_cache(
    scene: &SceneContext,
    scene_id: u64,
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
  ) {
    for kv in cartesian_state_cache.iter().filter(|kv| kv.key().scene_id == scene_id) {
      let id = EntityId::from_ffi(kv.key().entity_id);
      let t = match kv.value().comet_state {
        Some(ref body) => body.transform,
        None => kv.value().parent_frame_transform,
      };
      let _ = scene.scene.with_component_mut(id, |c: &mut TransformComponent| *c = t);
      mark_component_changed::<TransformComponent>(scene, id);
    }
  }

  /// SPICE EZR step of every almanac-driven body of `scene_id` in the cartesian cache, to `epoch`.
  ///
  /// Writes the body local transform (km residual w.r.t. its micro frame, which is a child of root)
  /// and `helio_state_km`. When a body drifts more than [`FRAME_SHIFT_THRESHOLD_KM`] from its frame
  /// origin, the frame is moved onto the body (AU grid) and the shift is propagated both to the
  /// frame entry and to the `parent_frame_transform` copy of every entry under that frame, so the
  /// next step measures the drift against the frame's current origin.
  ///
  /// [`FRAME_SHIFT_THRESHOLD_KM`]: crate::simulation_api::reposition::FRAME_SHIFT_THRESHOLD_KM
  pub fn step_cartesian_cache(
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
    scene_id: u64,
    epoch: anise::time::Epoch,
    almanac: &AlmanacPackedData,
  ) {
    use crate::simulation_api::reposition::{
      AU_TO_KM, FRAME_SHIFT_THRESHOLD_KM, compute_macro_and_residual,
    };
    // accumulate frame updates into a vec separately from the dashmap cause iter_mut locks shards
    // Note: rotation and scale stay fixed at identity, therefore we track only positions relative
    // to root
    let mut frame_updates = alloc::vec::Vec::<(EntityId, Vec3f32)>::with_capacity(4);
    cartesian_state_cache.iter_mut().for_each(|mut kv| {
      if kv.key().scene_id != scene_id {
        return;
      }
      let state = kv.value_mut();
      let micro_frame_pos_km = state.parent_frame_transform.position.to_f64() * AU_TO_KM;
      let parent_id = state.parent_frame;
      let Some(ref mut body_state) = state.comet_state else {
        return;
      };
      match body_state.almanac_planet.step_with_velocity(
        epoch,
        almanac,
        body_state.body_rotational_model.as_ref(),
      ) {
        Ok((global_dpos, global_vel_kms, global_rot)) => {
          body_state.helio_state_km = Some((global_dpos, global_vel_kms));
          // micro frame is child of root (asserted at cache insertion), so the body local offset
          // is the world space difference body - frame
          let diff_km = global_dpos - micro_frame_pos_km;
          if diff_km.length() > FRAME_SHIFT_THRESHOLD_KM {
            // --- FRAME SHIFT ---: move the frame onto the body AU grid point, keep the exact km
            // residual after f32 truncation to prevent jumping
            let (subtree_pos_f32, residual_f32) = compute_macro_and_residual(global_dpos);
            frame_updates.push((parent_id, subtree_pos_f32));
            body_state.transform.position = residual_f32;
          } else {
            // --- NORMAL DRIFT ---: the frame stays still
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
    });

    if frame_updates.is_empty() {
      return;
    }
    // propagate to the frame entry and to every body copy under it
    cartesian_state_cache.iter_mut().for_each(|mut kv| {
      if kv.key().scene_id != scene_id {
        return;
      }
      let state = kv.value_mut();
      if let Some((_, pos)) = frame_updates.iter().find(|(f, _)| *f == state.parent_frame) {
        state.parent_frame_transform.position = *pos;
      }
    });
  }

  /// Dust v3 emission for every particle system of the scene (logic tick, compute queue).
  ///
  /// Jet truth state = comet almanac state of this tick (f64, from the cartesian cache) plus the
  /// jet offset rotated by the body rotation. Records the emit dispatches (or emits on the CPU in
  /// CPU particle mode) and returns the systems that recorded something, so the caller can mark
  /// their pending batches submitted.
  pub fn record_dust_emissions(
    vulkan_device: &Device,
    cmd: ash::vk::CommandBuffer,
    scene: &crate::scene::Scene,
    scene_id: u64,
    cartesian_state_cache: &dashmap::DashMap<
      crate::simulation_api::structs::SceneEntityId,
      CartesianState,
    >,
    almanac: &AlmanacPackedData,
    start_epoch: anise::time::Epoch,
    now_scaled_us: timeus_t,
    reference: Option<crate::simulation_api::structs::KeplerianElements>,
  ) -> alloc::vec::Vec<EntityId> {
    use crate::scene::{
      dust::{AU_M, JetState},
      particles::ParticleSystemComponent,
    };
    use aethervk_oshal_rlib::math::quaternion::Quaternion as _;
    let t_s = now_scaled_us as f64 * 1e-6;
    let mut ps_ids = alloc::vec::Vec::new();
    scene.query1(|id, _: &ParticleSystemComponent| ps_ids.push(id));
    let mut recorded = alloc::vec::Vec::new();
    for ps_id in ps_ids {
      // the jet is a direct child of the comet body (see `avkSimulationContext_addParticleSystem`)
      let Some(body_id) = scene.get_parent(ps_id) else {
        continue;
      };
      // the comet must be almanac-driven (committed and stepped at least once)
      let Some((planet, rot_model)) = cartesian_state_cache
        .get(&crate::simulation_api::structs::SceneEntityId::new(
          scene_id, body_id,
        ))
        .and_then(|c| c.comet_state.as_ref().map(|b| (b.almanac_planet, b.body_rotational_model)))
      else {
        continue;
      };
      let Some(jet_local) = scene.with_component(ps_id, |t: &TransformComponent| *t) else {
        continue;
      };
      // Jet state at any scaled time, straight from the almanac: emission windows are evaluated
      // at their own grid times, so the dust does not depend on when the ticks happened.
      //
      // Outside the SPK coverage (dust history before the start epoch, see
      // `dust::DustSystemState`) the comet follows the committed reference orbit (two-body; the
      // re-osculated one drifts ~600 km/month, nothing for dust spread over 1e5+ km) and the
      // rotational model alone.
      let jet_at = |t: f64| -> Option<JetState> {
        let epoch = start_epoch + anise::time::Duration::from_seconds(t);
        let from_reference = || {
          let (r, v) = crate::simulation::orbit_elements::state_from_elements(
            reference.as_ref()?,
            crate::simulation::orbit_elements::MU_SUN_KM3_S2,
            epoch.to_jde_tdb_days(),
          )?;
          Some((
            DVec3::from_components(r[0], r[1], r[2]),
            DVec3::from_components(v[0], v[1], v[2]),
            planet.rotation_at(epoch, almanac, rot_model.as_ref()),
          ))
        };
        let (pos_km, vel_kms, body_rot) = if t < 0.0 {
          from_reference()?
        } else {
          planet
            .step_with_velocity(epoch, almanac, rot_model.as_ref())
            .ok()
            .or_else(from_reference)?
        };
        // uniform spin about the body z (pole) axis, see `AlmanacPlanet::step_with_velocity`.
        // Without a rotational model `DustHostState` estimates it from the attitudes.
        let spin = rot_model.as_ref().map(|m| {
          let pole = body_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
          let omega = m.rotation_rate * core::f64::consts::PI / (180.0 * 86400.0);
          let sgn = if omega < 0.0 { -1.0 } else { 1.0 };
          [
            pole.x() as f64 * sgn,
            pole.y() as f64 * sgn,
            pole.z() as f64 * sgn,
            omega.abs(),
          ]
        });
        let off_km = body_rot.rotate_vector(jet_local.position);
        // the jet sits on the nucleus sphere: its outward normal is its offset direction
        let site_normal = {
          let n = [off_km.x() as f64, off_km.y() as f64, off_km.z() as f64];
          let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
          if l > 0.0 {
            [n[0] / l, n[1] / l, n[2] / l]
          } else {
            [0.0, 0.0, 1.0]
          }
        };
        let rot = (body_rot * jet_local.rotation).0;
        Some(JetState {
          t_s: t,
          r_m: [
            (pos_km.x() + off_km.x() as f64) * 1000.0,
            (pos_km.y() + off_km.y() as f64) * 1000.0,
            (pos_km.z() + off_km.z() as f64) * 1000.0,
          ],
          // nucleus rotation velocity (ω × r, < 1 m/s) is neglected
          v_ms: [
            vel_kms.x() * 1000.0,
            vel_kms.y() * 1000.0,
            vel_kms.z() * 1000.0,
          ],
          rot: [rot.x(), rot.y(), rot.z(), rot.w()],
          site_normal,
          spin,
        })
      };
      let Some(jet) = jet_at(t_s) else {
        continue;
      };
      let r_m = jet.r_m;
      let r_au = (r_m[0] * r_m[0] + r_m[1] * r_m[1] + r_m[2] * r_m[2]).sqrt() / AU_M;

      let batches = scene
        .with_component(ps_id, |ps: &ParticleSystemComponent| {
          let cfg_at = |j: &JetState| {
            let r = (j.r_m[0] * j.r_m[0] + j.r_m[1] * j.r_m[1] + j.r_m[2] * j.r_m[2]).sqrt() / AU_M;
            ps.emission_params.dust_emit_config(r as f32, ps.ttl_us)
          };
          let mut sys = ps.dust.lock();
          // exposure reference from the current activity (smooth along the orbit, independent of
          // the history and the camera); kept while the jet is beyond its production cutoff
          let tau_ref = cfg_at(&jet).tau_ref();
          if tau_ref > 0.0 && tau_ref.is_finite() {
            sys.tau_ref = tau_ref;
          }
          sys.tick(t_s, &jet_at, &cfg_at)
        })
        .unwrap_or_default();
      if batches.is_empty() {
        continue;
      }
      let mut ok = true;
      for (base, b, seq) in batches.iter() {
        if let Err(e) = vulkan_device.cmd_dust_emit(cmd, ps_id.as_ffi(), *base, b, *seq) {
          oshal::log!("[Dust] emit failed for {:?}: {}", ps_id, e);
          ok = false;
          break;
        }
      }
      if ok {
        recorded.push(ps_id);
      } else {
        // never leave batches pending forever (they would block the drawable prefix): the whole
        // ring gets re-emitted from its descriptors on the next ticks
        let _ = scene.with_component(ps_id, |ps: &ParticleSystemComponent| {
          ps.dust.lock().invalidate_gpu()
        });
      }
      static LOG_COUNTER: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
      if LOG_COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed) % 120 == 0 {
        let _ = scene.with_component(ps_id, |ps: &ParticleSystemComponent| {
          let host = ps.dust.lock();
          let tiers: alloc::vec::Vec<alloc::string::String> = host
            .stats()
            .iter()
            .map(|t| {
              alloc::format!(
                "{}/{} {:.1}-{:.1} d{}",
                t.live_clusters,
                t.capacity,
                t.youngest_age_s / 86400.0,
                t.oldest_age_s / 86400.0,
                if t.caught_up { "" } else { " (building)" }
              )
            })
            .collect();
          oshal::log!(
            "[Dust] r={:.3} AU tiers [{}] last batch={} clusters, mass {:.3e} g",
            r_au,
            tiers.join(" | "),
            batches.last().map(|b| b.1.count).unwrap_or(0),
            batches.last().map(|b| b.1.mass_params[0]).unwrap_or(0.0),
          );
        });
      }
    }
    recorded
  }

  /// Time Boundary,Step Size (Precision Loss),What it means for your data
  /// ----------------------------------------------------------------------------------------------
  /// < 16 seconds,<1μs,Perfect. You can represent every single microsecond accurately.
  /// > 16 seconds,≈1.9μs,Microsecond loss. The gap between floats becomes larger than 1μs. Consecutive microseconds round to the same f32 value.
  /// > 8,192 sec (2.2 hours)",≈1 ms,Millisecond loss. You can no longer distinguish sub-millisecond differences.
  /// > 131,072 sec (36.4 hours)",≈15.6 ms,"UI/Physics jitter. At 60fps (16.6ms per frame), your time steps are now larger than a video frame. Physics engines using f32 will glitch."
  /// > 8,388,608 sec (97 days)",1 second,Total fractional loss. The f32 can no longer hold fractions. 97 days+0.5 seconds will simply round to 97 days.
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
          let rotated = parent_transform.rotation.rotate_vector(
            aethervk_oshal_rlib::math::vector::vec3f64::Vec3f64::from_components(
              scaled_parent_scale.x() as f64 * acc_pos.x(),
              scaled_parent_scale.y() as f64 * acc_pos.y(),
              scaled_parent_scale.z() as f64 * acc_pos.z(),
            ),
          );

          acc_pos = parent_transform.position + rotated;
          acc_rot = (parent_transform.rotation * acc_rot).normalize();
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
  use core::{ptr::NonNull, slice};

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
      oshal::log!(
        "self_sync_do_if_done failed: time_managers does not contain scene_id {}",
        scene_id
      );
      return None;
    }

    if let Some(scene_lock) = scenes.get(&scene_id) {
      let had_task = scene_lock
        .read()
        .active_physics_task
        .compare_exchange_weak(true, false, Ordering::Acquire, Ordering::Relaxed)
        .unwrap_or(false);

      // Only the logic thread writes `latest_physics_sync`, so a copy taken under a short read lock
      // stays valid: the GPU wait below runs with *no* scene lock held, so FFI writers and the
      // render thread are not blocked for up to 8 ms every tick.
      let pending_sync = if had_task {
        scene_lock.read().latest_physics_sync.clone()
      } else {
        None
      };

      render_frontend
        .with_device(device_handle, |dyn_device| {
          let vulkan_device: &Device = dyn_device.as_any().downcast_ref().unwrap();

          // Block (zero CPU) until the GPU signals the compute timeline semaphore, with a hard
          // deadline of 8ms (half a frame).
          let is_done = pending_sync.as_ref().map_or(true, |sync| {
            sync.blocking_wait(&vulkan_device.device, 8_000_000)
          });

          // acquire a read lock on the timeline manager in the scene just to prevent
          // execution of a simulation step from someone else
          // SAFETY: when scene is created, time_manager is associated to it
          let _time_mgr = unsafe { scenes.time_managers.get(&scene_id).unwrap_unchecked() };
          let mut scene_write = scene_lock.write();

          if is_done {
            if had_task {
              // Self Sync: destroy consumed synchronization primitives
              let _ = scene_write.latest_physics_sync.take();
            }
            Ok(Some(f(vulkan_device, &mut scene_write, render_tx)))
          } else {
            // Normal polling timeout (e.g. physics frame takes > 8ms).
            // We just restore the flag and let the caller loop.
            scene_write.active_physics_task.store(true, Ordering::Release);
            Ok(None)
          }
        })
        .unwrap_or(None)
    } else {
      oshal::log!(
        "self_sync_do_if_done failed: scene_id {} not found in scenes",
        scene_id
      );
      None
    }
  }
}

#[cfg(test)]
#[path = "logic_thread_tests.rs"]
mod logic_thread_tests;
