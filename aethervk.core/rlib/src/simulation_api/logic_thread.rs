//! logic_thread module.

use super::{
  SimulationContext,
  structs::{LogicCommand, LogicThreadContext, LogicWorkload, SceneContext, SimulationTaskResult},
};
use crate::{
  gpu::WeakRenderFrontendExt,
  scene::{
    CameraComponent, CursorComponent, EntityId, FollowingComponent, TransformComponent,
    camera::{QuatToEulerAngles, SceneCameraExt},
  },
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::os::time::timeus_t;
use aethervk_oshal_rlib::{
  self as oshal,
  math::floating::FloatOps,
  math::quaternion::Quaternion,
  math::vector::vec3::Vec3f32,
  math::vector::vec4::Quat,
  math::vector::{Vector, Vector3, Vector4},
  os::pool::WorkloadStatus,
  os::thread,
  os::thread::Thread,
  os::{NativeError, ThreadingError},
};
use alloc::{boxed::Box, string::ToString};
use spin::{RwLockReadGuard, RwLockUpgradableGuard};
use thingbuf::mpsc;

struct PlayControl {
  scene_id: u64,
  target_frame_time: timeus_t,
  last_frame_start: timeus_t,
  last_render_tick: Option<core::num::NonZero<u64>>,
}

impl PlayControl {
  fn new(scene_id: u64, target_frame_time: timeus_t) -> Self {
    Self {
      scene_id,
      target_frame_time,
      last_frame_start: oshal::os::time::get_monotonic_time(),
      last_render_tick: None,
    }
  }
}

/// TODO: Document this item
pub fn start_logic_thread(
  logic_rx: mpsc::Receiver<LogicCommand>,
  context: alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    let target_frame_time = oshal::os::time::timeus_milliseconds(16); // ~60 FPS
    let mut play_controls: hashbrown::HashMap<u64, PlayControl> = hashbrown::HashMap::new();

    loop {
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
          let mut can_tick = true;
          if let Some(task) = pc.last_render_tick {
            let status = context.task_manager.read().get_status(task.get());
            if status == crate::simulation_api::structs::TaskStatusCode::Pending {
              can_tick = false;
            }
          }

          if can_tick {
            pc.last_frame_start = now;

            let _ = execute_simulation_tick(scene_id, &context);

            let pe_handles: alloc::vec::Vec<(crate::gpu::PresentationEngineHandle, bool)> = {
              let scenes = context.scenes.read();
              if let Some(scene_ctx) = scenes.get(&scene_id) {
                scene_ctx.read().presentation_engines.read().iter().map(|(&k, &v)| (k, v)).collect()
              } else {
                alloc::vec::Vec::new()
              }
            };

            let mut last_task = None;
            for (pe, is_windowless) in pe_handles {
              let task_id = alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(0));
              let scene = {
                let scenes = context.scenes.read();
                scenes.get(&scene_id).unwrap().clone()
              };

              let (active_cam, outlines, sun, sky, cursor, callback) = {
                let r = scene.read();
                (
                  r.active_camera_entity.unwrap_or_default(),
                  r.outlines_enabled.load(core::sync::atomic::Ordering::Acquire),
                  r.sun_entity,
                  r.sky_entity,
                  r.cursor_entity,
                  r.custom_render_callback,
                )
              };

              let _ = context.render_tx.try_send(
                crate::simulation_api::structs::RenderCommand::RenderFrame(
                  crate::simulation_api::structs::RenderFrame {
                    presentation_engine_handle: pe,
                    task_id: alloc::sync::Arc::clone(&task_id),
                    scene,
                    render_physical_meshes_outline: outlines,
                    camera_entity: active_cam,
                    clear_color: [0.0, 0.0, 0.0, 1.0],
                    sun_entity: sun,
                    sky_entity: sky,
                    cursor_entity: cursor,
                    custom_render_callback: callback,
                  },
                ),
              );

              let task_id_val = loop {
                let value = task_id.load(core::sync::atomic::Ordering::Relaxed);
                if value != 0 {
                  let _ = task_id.load(core::sync::atomic::Ordering::Acquire);
                  break value;
                }
                oshal::os::native::this_thread::yield_now();
              };
              if task_id_val == u64::MAX {
                // Render failed, continue to next
                continue;
              }
              last_task = core::num::NonZero::new(task_id_val);

              if is_windowless {
                let fptr = crate::simulation_api::RENDER_CALLBACK
                  .load(core::sync::atomic::Ordering::Relaxed);
                if !fptr.is_null() {
                  let ctx_ptr = context.ctx_ptr;
                  use aethervk_oshal_rlib::os::pool::tasklet::ThreadPoolExt;
                  let _ = context.thread_pool.spawn_tasklet(None, move || {
                    let fptr = crate::simulation_api::RENDER_CALLBACK
                      .load(core::sync::atomic::Ordering::Relaxed);
                    let ctx =
                      unsafe { &*(ctx_ptr.get() as *mut crate::simulation_api::SimulationContext) };
                    loop {
                      let completed = ctx
                        .render_proxy
                        .0
                        .as_frontend()
                        .and_then(|f| {
                          f.with_device(ctx.render_proxy.1, |device| {
                            Ok(device.is_task_completed(task_id_val).unwrap_or(true))
                          })
                          .ok()
                        })
                        .unwrap_or(true);

                      if completed {
                        break;
                      }
                      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
                        1,
                      ));
                    }
                    let cb: extern "C" fn(u64, u64, u64) = unsafe { core::mem::transmute(fptr) };
                    cb(scene_id, pe.0, task_id_val);
                  });
                }
              }
            }
            pc.last_render_tick = last_task;
            processed_any = true;
          }
        }
      }

      while let Ok(cmd) = logic_rx.try_recv() {
        if let LogicCommand::Shutdown = cmd {
          return;
        }

        match cmd {
          LogicCommand::ImportModel { .. }
          | LogicCommand::LoadAlmanac { .. }
          | LogicCommand::LoadCometSpk { .. }
          | LogicCommand::SpawnModelInstance { .. } => {
            // Heavy I/O or generation tasks are scattered to the thread pool
            let workload = Box::new(LogicWorkload {
              cmd,
              ctx: context.clone(),
            });
            let _ = context.thread_pool.scatter(alloc::vec![workload]);
          }
          _ => {
            // Synchronous orchestrator commands executed on the logic thread natively!
            let task_id = match &cmd {
              LogicCommand::RaycastNdc { task_id, .. } => Some(*task_id),
              LogicCommand::Raycast { task_id, .. } => Some(*task_id),
              LogicCommand::Custom { task_id, .. } => Some(*task_id),
              _ => None,
            };

            let cmd_desc = match &cmd {
              LogicCommand::RaycastNdc { .. } => "Raycast NDC".to_string(),
              LogicCommand::Raycast { .. } => "Raycast".to_string(),
              LogicCommand::RotateCamera { .. } => "Rotate Camera".to_string(),
              LogicCommand::ZoomCamera { .. } => "Zoom Camera".to_string(),
              LogicCommand::PlayScene { .. } => "Play Scene".to_string(),
              LogicCommand::PauseScene { .. } => "Pause Scene".to_string(),
              LogicCommand::StepScene { .. } => "Step Scene".to_string(),
              LogicCommand::SetSceneTimeScale { .. } => "Set Time Scale".to_string(),
              _ => "Logic Task".to_string(),
            };

            let res = process_command_internal(cmd, &context);

            if let Some(tid) = task_id {
              if tid != 0 {
                let mut manager = context.task_manager.write();
                match res {
                  Ok(result) => {
                    manager.success_task(tid, result);
                  }
                  Err(e) => {
                    manager.fail_task(tid, e.to_string());
                    crate::simulation_api::emit_breadcrumb(
                      3,
                      &alloc::format!("Failed: {} - {}", cmd_desc, e),
                    );
                  }
                }
              }
            }
          }
        }
        processed_any = true;
      }

      if !processed_any {
        oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
      }
    }
  })
  .map_err(|err| <ThreadingError as Into<NativeError>>::into(err))
  .map_err(|err| <NativeError as Into<EngineError>>::into(err))
}

impl oshal::os::pool::Workload for LogicWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let task_id = match &self.cmd {
      LogicCommand::ImportModel { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      LogicCommand::LoadAlmanac { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      LogicCommand::LoadCometSpk { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      LogicCommand::SpawnModelInstance { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      LogicCommand::RaycastNdc { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      LogicCommand::Raycast { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }

      LogicCommand::Custom { task_id, .. } => {
        if *task_id == 0 {
          None
        } else {
          Some(*task_id)
        }
      }
      _ => None,
    };

    let res = process_command_internal(self.cmd.clone(), &self.ctx);

    let cmd_desc = match &self.cmd {
      LogicCommand::ImportModel { path, .. } => alloc::format!("Import model {}", path),
      LogicCommand::LoadAlmanac { path, .. } => alloc::format!("Load almanac {}", path),
      LogicCommand::LoadCometSpk { spk_id, epoch, .. } => alloc::format!(
        "Load Ephemeris data for SPK ID: {} at epoch {}",
        spk_id,
        epoch
      ),
      LogicCommand::SpawnModelInstance { name, .. } => alloc::format!("Spawn instance {}", name),
      LogicCommand::RaycastNdc { .. } => "Raycast NDC".to_string(),
      LogicCommand::Raycast { .. } => "Raycast".to_string(),
      _ => "Logic Task".to_string(),
    };

    if let Some(tid) = task_id {
      let mut manager = self.ctx.task_manager.write();
      match res {
        Ok(result) => {
          manager.success_task(tid, result);
          // Too frequent
          // crate::simulation_api::emit_breadcrumb(1, &alloc::format!("Success: {}", cmd_desc));
        }
        Err(e) => {
          manager.fail_task(tid, e.to_string());
          crate::simulation_api::emit_breadcrumb(
            3,
            &alloc::format!("Failed: {} - {}", cmd_desc, e),
          );
        }
      }
    }

    WorkloadStatus::Complete
  }
}

// TODO: All logic processing should be async. therefore, each command which is not Shutdown should modify the logic thread context (new members) to keep track of ongoing commands, so that when their state
// TODO  is polled, we can answer. This means that,
// TODO - when a scene is registered in the logic thread (new command), a *timed task* is dispatched every *requested from command* milliseconds (eg 16ms)
// TODO - this task is basically an iteration of a game loop, with N fixed updates (depends on elapsed simulation time, see time module on oshal (Unity inspired)) and then update function
// TODO - all camera commands should be registered under the logic scene structure, so that the update function can update the camera, cursor and everything else
// TODO - after this task is done, its task status should be set accordingly as failure if some error was encountered in the update or ok and the task is marked as finished, such that when queried everything is fine
// TODO - after the task is done, the C# FFI caller thread (eg view model) can start its round of queries
// TODO    - query the number and ids of non hidden components in the scene (including camera, cursor, ...)
// TODO    - should query only the component properties needed by the current properties editor, therefore must have support
// TODO    - query transform, query visibility (and toggle visibility), query BVH nodes
// TODO    - these "light" edits can be synchronous, ie processed directly in the logic thread and in the end a feedback is sent to FFI caller
// TODO - I/O Heavy tasks should be dispatched to thread pool and feedback immediately to return task id. Then, task should be polled
fn process_command_internal(
  command: LogicCommand,
  ctx: &LogicThreadContext,
) -> EngineResult<SimulationTaskResult> {
  match command {
    LogicCommand::Shutdown => Ok(SimulationTaskResult::None),
    // TODO camera movement commands should use methods from [`crate::scene::camera`]
    LogicCommand::RotateCamera(crate::simulation_api::structs::RotateCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    }) => {
      let scene_read = scene.read();
      let (cursor_entity, _) =
        scene_read.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id)).ok_or(
          EngineError::InvalidOperation("logic_thread:RotateCamera | scene doesn't have cursor"),
        )?;
      let rotation_speed: f32 = 0.005;
      scene_read.scene.orbit_camera(
        camera_entity,
        cursor_entity,
        -delta_x * rotation_speed, // negate for natural mouse drag
        -delta_y * rotation_speed,
      )?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::ZoomCamera(crate::simulation_api::structs::ZoomCamera {
      camera_entity,
      scene,
      amount,
    }) => {
      let scene_read = scene.read();
      let is_ortho = scene_read
        .scene
        .with_component(camera_entity, |c: &CameraComponent| {
          c.projection.w.w().abs() > 0.5
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:ZoomCamera | scene doesn't have CameraComponent",
        ))?;
      if is_ortho {
        return Ok(SimulationTaskResult::None);
      }

      let cursor_pos = scene_read
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
        .and_then(|(id, _)| {
          scene_read.scene.with_component(id, |t: &TransformComponent| t.position)
        })
        .unwrap_or(Vec3f32::zero());

      let dist = scene_read
        .scene
        .with_component(camera_entity, |t: &TransformComponent| {
          (t.position - cursor_pos).length().max(0.1)
        })
        .unwrap_or(100.0);

      use crate::scene::camera::SceneCameraExt;
      let zoom_speed = dist * 0.3; // middle ground between 0.1 and 1.0

      scene_read.scene.translate_camera_local(
        camera_entity,
        Vec3f32::from_components(0.0, -amount * zoom_speed, 0.0),
      )?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::ResetCamera(crate::simulation_api::structs::ResetCamera {
      camera_entity,
      scene,
    }) => {
      let scene_read = scene.read();

      let mut cursor_pos = Vec3f32::zero();
      if let Some((sun_id, _)) =
        scene_read.scene.query1_first_res::<crate::scene::SunComponent, _, _>(|id, _| Some(id))
      {
        if let Some(pos) =
          scene_read.scene.with_component(sun_id, |t: &TransformComponent| t.position)
        {
          cursor_pos = pos;
        }
      }

      if let Some((cursor_id, _)) =
        scene_read.scene.query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
      {
        let _ = scene_read.scene.with_component_mut(cursor_id, |c: &mut TransformComponent| {
          c.position = cursor_pos;
        });
      }

      let pitch = (-1.0_f32 / 3.0_f32.sqrt()).asin();
      let yaw = -core::f32::consts::FRAC_PI_4;
      let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      let offset = q.rotate_vector(Vec3f32::from_components(0.0, 10.0, 0.0));

      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = cursor_pos + offset;
          c.rotation = q;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:ResetCamera | camera entity doesn't have TransformComponent",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PanCamera(crate::simulation_api::structs::PanCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    }) => {
      let scene_read = scene.read();
      let (cursor_entity, _) =
        scene_read.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id)).ok_or(
          EngineError::InvalidOperation("logic_thread:PanCamera | scene doesn't have cursor"),
        )?;
      use crate::scene::camera::SceneCameraExt;
      scene_read.scene.pan_camera_and_cursor(camera_entity, cursor_entity, delta_x, delta_y)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PanCursor(_) => Ok(SimulationTaskResult::None),
    LogicCommand::MoveCursor(crate::simulation_api::structs::MoveCursor {
      scene,
      delta_x,
      delta_y,
      delta_z,
    }) => {
      let speed = 0.001;
      let scene_read = scene.read();
      scene_read
        .scene
        .query2_res_first_mut(|_, t: &mut TransformComponent, _c: &mut CursorComponent| {
          let translation = Vec3f32::from_components(delta_x, delta_y, delta_z) * speed;
          t.position = t.position + translation;
          Some(())
        })
        .map(|_| ())
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:MoveCursor | scene doesn't have cursor",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::SnapToEntity(crate::simulation_api::structs::SnapToEntity {
      snap_entity,
      target_entity,
      scene,
    }) => {
      let scene_read = scene.read();
      try_snap_entity(snap_entity, target_entity, &scene_read)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FollowEntity(crate::simulation_api::structs::FollowEntity {
      snap_entity,
      entity_id,
      scene,
      unfollow_other: _,
    }) => {
      let scene_read = scene.read();
      use crate::scene::interaction::SceneInteractionExt;
      scene_read.scene.follow_entity(entity_id, None)?;
      try_snap_entity(snap_entity, entity_id, &scene_read)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::UnfollowEntity(crate::simulation_api::structs::UnfollowEntity {
      entity_id,
      scene,
    }) => {
      let scene_read = scene.read();
      scene_read
        .scene
        .remove_component::<FollowingComponent>(entity_id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::TogglePaintMode(crate::simulation_api::structs::TogglePaintMode {
      scene_id,
      entity_id,
    }) => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        let read_ctx = scene_ctx.read();
        let mut changed = false;
        let _ = read_ctx.scene.with_component_mut(
          entity_id,
          |mesh: &mut crate::scene::PhysicalMeshComponent| {
            mesh.paint_display_mode = (mesh.paint_display_mode + 1) % 3;
            changed = true;
          },
        );
        if changed {
          // Tell the system the transform or something changed so it re-renders/updates
          // But actually we just need to re-record the command buffer. Since we don't have a direct "re-record"
          // signal, marking Transform as changed will trigger it.
          let ext_id = read_ctx
            .entity_map
            .iter()
            .find(|&(_, &v)| v == entity_id)
            .map(|(&k, _)| k)
            .unwrap_or(0);
          read_ctx.mark_component_changed(ext_id, "Transform");
        }
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FeedbackGetSceneTimeScale { scene_id } => {
      let scenes = ctx.scenes.read();
      let scene_ctx =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let scale = scene_ctx.read().time_state.read().current_scale;
      Ok(SimulationTaskResult::U64(match scale {
        crate::simulation_api::structs::TimeScale::Stopped => 0,
        crate::simulation_api::structs::TimeScale::OneDay => 1,
        crate::simulation_api::structs::TimeScale::OneWeek => 2,
        crate::simulation_api::structs::TimeScale::OneMonth => 3,
      }))
    }
    LogicCommand::FeedbackGetSceneDateTimeUTC { scene_id } => {
      let scenes = ctx.scenes.read();
      let scene_ctx =
        scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let _utc_str = scene_ctx.read().time_state.read().current_epoch.to_string();
      // Return success with None, as this is currently handled by get_simulation_time_utc
      // TODO: decide if we want to return strings as results
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FeedbackGetSceneDateTimeLimitsUTC { scene_id: _ } => {
      Ok(SimulationTaskResult::None)
    }

    LogicCommand::SetSceneTimeScale { scene_id, scale } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().current_scale = scale;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::SetSceneEpoch {
      scene_id,
      epoch_tai_seconds,
    } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().current_epoch =
          anise::time::Epoch::from_tai_seconds(epoch_tai_seconds);
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PlayScene { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().is_playing = true;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PauseScene { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.read().time_state.write().is_playing = false;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::StepScene {
      scene_id,
      step_days,
    } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        let read_ctx = scene_ctx.read();
        let mut time_state = read_ctx.time_state.write();
        time_state.current_epoch = time_state.current_epoch + anise::time::Unit::Day * step_days;
      }
      Ok(SimulationTaskResult::None)
    }

    LogicCommand::ImportModel { task_id: _, path } => {
      let mesh_res = crate::simulation::comet::load_comet_from_gltf(&path, false);
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
    LogicCommand::LoadAlmanac { task_id, path } => {
      ctx.load_almanac_file_internal(&path)?;
      Ok(SimulationTaskResult::None)
    }
    // TODO: async tasklet. this should return the task_id, not take it, while a new command QueryLoadCometSpkFinished should poll the task id and return EngineResult<bool> true in case it finished
    LogicCommand::LoadCometSpk {
      task_id,
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
          if let Ok(mesh) = crate::simulation::comet::load_comet_from_gltf(&path_str, false) {
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
    LogicCommand::RaycastNdc {
      task_id: _,
      scene_id,
      ndc_x,
      ndc_y,
    } => {
      let res = ctx.raycast_ndc_internal(scene_id, ndc_x, ndc_y)?;
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
    LogicCommand::Custom {
      task_id: _,
      custom_fn,
      user_data,
    } => {
      let ptr = user_data.map(|p| p.0).unwrap_or(core::ptr::null_mut());
      custom_fn(ctx, ptr)
    }
  }
}

fn execute_simulation_tick(
  scene_id: u64,
  ctx: &alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<()> {
  // Get necessary ARCs without holding the scene graph read lock!
  let (time_state_arc, physics_scene_arc, scene_arc) = {
    let scenes = ctx.scenes.read();
    let scene_ctx =
      scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?.read();
    (
      scene_ctx.time_state.clone(),
      scene_ctx.physics_scene.clone(),
      scene_ctx.scene.clone(),
    )
  };

  // 1. Update time natively using the Arc (no scene read lock held)
  let fixed_dt_us = {
    let mut ts_write = time_state_arc.write();
    let mut time_info = ts_write.time_info.write();
    if ts_write.is_playing {
      time_info.ut_update();
    }
    time_info.fixed_delta_time.load(core::sync::atomic::Ordering::Relaxed)
  };

  let mut any_fixed_step = false;
  while time_state_arc.read().time_info.read().needs_fixed_update() {
    any_fixed_step = true;
    if let Some(ps_lock) = &physics_scene_arc {
      let mut ps = ps_lock.write();
      *ps = crate::physics::physics_scene::PhysicsScene::build_from_scene(scene_arc.as_ref());
    }

    let step_days = {
      let mut ts_write = time_state_arc.write();
      let scale_days_per_sec = ts_write.current_scale.to_days_per_st_second();
      let fixed_sim_seconds = fixed_dt_us as f64 / 1_000_000.0;
      let step = scale_days_per_sec * fixed_sim_seconds;
      if step > 0.0 {
        ts_write.current_epoch = ts_write.current_epoch + anise::time::Unit::Day * step;
      }
      step
    };

    let current_epoch = time_state_arc.read().current_epoch;

    if step_days > 0.0 {
      let logic_state = ctx.logic_state.read();
      // 1. Update Macro Bodies First (Planets/Comets via SPICE)
      if scene_arc.should_parallelize() {
        scene_arc.query2_mut_par::<TransformComponent, crate::scene::AlmanacPlanet, _>(
          &ctx.thread_pool,
          |_, transform, planet| {
            let _ = planet.step(
              transform,
              current_epoch,
              step_days,
              &logic_state.almanac_data,
            );
          },
        );
      } else {
        scene_arc.query2_mut::<TransformComponent, crate::scene::AlmanacPlanet, _>(
          |_, transform, planet| {
            let _ = planet.step(
              transform,
              current_epoch,
              step_days,
              &logic_state.almanac_data,
            );
          },
        );
      }

      // 2. Update Micro Bodies Second & 4. Resolve Collisions
      if let Some(ps_lock) = &physics_scene_arc {
        let mut ps = ps_lock.write();
        let dt_us = (step_days * 86400.0 * 1_000_000.0) as aethervk_oshal_rlib::os::time::timeus_t;

        if scene_arc.should_parallelize() {
          let kernels = crate::physics::cpu_kernels::CpuSimdKernels {
            thread_pool: alloc::sync::Arc::clone(&ctx.thread_pool),
          };
          let _ =
            crate::gpu_backends::simulation_step(&kernels, &mut ps, scene_arc.as_ref(), 0, dt_us);
        } else {
          let kernels = crate::physics::cpu_kernels::CpuScalarKernels {};
          let _ =
            crate::gpu_backends::simulation_step(&kernels, &mut ps, scene_arc.as_ref(), 0, dt_us);
        }
      }

      // 3. Process Handoffs
      crate::physics::handoff::SpheresOfInfluenceSystem::process_handoffs_par(
        scene_arc.as_ref(),
        &ctx.thread_pool,
      );
    }

    // Advance fixed clock step
    time_state_arc.read().time_info.read().ut_fixed_update();
  }

  // Snap following entities (re-acquires brief scene graph lock since we need try_snap_entity)
  {
    let scenes = ctx.scenes.read();
    if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
      let scene_ctx = scene_ctx_guard.read();

      if any_fixed_step {
        for (ext_id, ent_id) in scene_ctx.entity_map.iter() {
          // Check if entity has TransformComponent
          let mut changed = false;
          scene_ctx.scene.with_component(*ent_id, |_: &TransformComponent| {
            changed = true;
          });
          if changed {
            scene_ctx.mark_component_changed(*ext_id, "Transform");
            // Also mark Comet and Planet and Sun
            scene_ctx.mark_component_changed(*ext_id, "Comet");
            scene_ctx.mark_component_changed(*ext_id, "Planet");
            scene_ctx.mark_component_changed(*ext_id, "Sun");
          }
        }
      }

      if let Some((target_id, _)) =
        scene_ctx.scene.query1_first_res::<FollowingComponent, _, _>(|id, _| Some(id))
      {
        if let Some((cursor_id, _)) =
          scene_ctx.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id))
        {
          let _ = try_snap_entity(cursor_id, target_id, &scene_ctx);
        }
      }
    }
  }

  let sim_task_id = ctx.task_manager.write().create_task();
  ctx.task_manager.write().success_task(
    sim_task_id.get(),
    crate::simulation_api::structs::SimulationTaskResult::None,
  );

  // Invoke SimulationCallback
  let fptr = crate::simulation_api::SIMULATION_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    let tm = alloc::sync::Arc::clone(&ctx.task_manager);
    let ctx_ptr = ctx.ctx_ptr; // SendPtrMut is Send
    use aethervk_oshal_rlib::os::pool::tasklet::ThreadPoolExt;
    let _ = ctx.thread_pool.spawn_tasklet(None, move || {
      let fptr =
        crate::simulation_api::SIMULATION_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
      loop {
        let status = tm.read().get_status(sim_task_id.get());
        if status == crate::simulation_api::structs::TaskStatusCode::Completed
          || status == crate::simulation_api::structs::TaskStatusCode::Error
        {
          break;
        }
        oshal::os::native::this_thread::yield_now();
      }
      let cb: extern "C" fn(u64, *mut core::ffi::c_void) = unsafe { core::mem::transmute(fptr) };
      let _dummy = ctx_ptr;
      cb(scene_id, ctx_ptr.get());
    });
  }

  // Clear changed entities
  let scenes = ctx.scenes.read();
  if let Some(scene_ctx_guard) = scenes.get(&scene_id) {
    let scene_ctx = scene_ctx_guard.read();
    scene_ctx.changed_entities.write().clear();
  }

  Ok(())
}

fn try_snap_entity(
  snap_entity: EntityId,
  target_entity: EntityId,
  scene_read: &RwLockReadGuard<SceneContext>,
) -> EngineResult<()> {
  let (target_pos, target_rot) = {
    let t =
      scene_read.scene.global_transform(target_entity).ok_or(EngineError::InvalidOperation(
        "logic_thread:SnapToEntity | snap target doesn't have TransformComponent",
      ))?;
    (t.position, t.rotation)
  };
  scene_read.scene.set_global_position_and_rotation(snap_entity, target_pos, target_rot)
}
