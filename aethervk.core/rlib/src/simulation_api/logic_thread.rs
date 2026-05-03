use spin::RwLockReadGuard;
use crate::{
  scene::{
    camera::{SceneCameraExt, QuatToEulerAngles}, CameraComponent, CursorComponent, EntityId, FollowingComponent,
    TransformComponent,
  },
  types::{EngineError, EngineResult},
};
use thingbuf::mpsc;
use aethervk_oshal_rlib::{
  self as oshal,
  os::{NativeError, ThreadingError},
  math::floating::FloatOps,
  math::quaternion::Quaternion,
  math::vector::vec3::Vec3f32,
  math::vector::vec4::Quat,
  math::vector::{Vector, Vector3, Vector4},
  os::pool::WorkloadStatus,
  os::thread,
  os::thread::Thread,
};
use super::{
  SimulationContext,
  structs::{LogicCommand, LogicThreadContext, SceneContext, LogicWorkload, SimulationTaskResult},
};
use alloc::{boxed::Box, string::ToString};

pub fn start_logic_thread(
  logic_rx: mpsc::Receiver<LogicCommand>,
  context: alloc::sync::Arc<LogicThreadContext>,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    loop {
      match logic_rx.try_recv() {
        Ok(cmd) => {
          if let LogicCommand::Shutdown = cmd {
            break;
          }

          let workload = Box::new(LogicWorkload {
            cmd,
            ctx: context.clone(),
          });
          let _ = context.thread_pool.scatter(alloc::vec![workload]);
        }
        Err(e) => {
          if let mpsc::errors::TryRecvError::Closed = e {
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

impl oshal::os::pool::Workload for LogicWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let task_id = match &self.cmd {
      LogicCommand::ImportModel { task_id, .. } => Some(*task_id),
      LogicCommand::LoadAlmanac { task_id, .. } => Some(*task_id),
      LogicCommand::LoadCometSpk { task_id, .. } => Some(*task_id),
      LogicCommand::SpawnModelInstance { task_id, .. } => Some(*task_id),
      LogicCommand::RaycastNdc { task_id, .. } => Some(*task_id),
      LogicCommand::Raycast { task_id, .. } => Some(*task_id),
      LogicCommand::SimulationTick { task_id, .. } => Some(*task_id),
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
      LogicCommand::SimulationTick { .. } => "Simulation Tick".to_string(),
      _ => "Logic Task".to_string(),
    };

    if let Some(tid) = task_id {
      let mut manager = self.ctx.task_manager.write();
      match res {
        Ok(result) => {
          manager.success_task(tid, result);
          crate::simulation_api::emit_breadcrumb(1, &alloc::format!("Success: {}", cmd_desc));
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
      let (cursor_entity, _) = scene_read
        .scene
        .query1_first_res::<CursorComponent, _, _>(|id, _| Some(id))
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | scene doesn't have cursor",
        ))?;
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
        .and_then(|(id, _)| scene_read.scene.with_component(id, |t: &TransformComponent| t.position))
        .unwrap_or(Vec3f32::zero());
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          let offset = c.position - cursor_pos;
          let dist = offset.length().max(0.1);
          let zoom_speed = dist * 0.1;
          let forward = c
            .rotation
            .rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
          let new_pos = c.position + forward * (amount * zoom_speed);
          c.position = new_pos;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:ZoomCamera | scene doesn't have TransformComponent",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::ResetCamera(crate::simulation_api::structs::ResetCamera {
      camera_entity,
      scene,
    }) => {
      let scene_read = scene.read();
      let cursor_pos = scene_read
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
        .and_then(|(id, _)| scene_read.scene.with_component(id, |t: &TransformComponent| t.position))
        .unwrap_or(Vec3f32::zero());

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
      let cursor_pos = scene_read
        .scene
        .query1_first_res::<crate::scene::CursorComponent, _, _>(|id, _| Some(id))
        .and_then(|(id, _)| scene_read.scene.with_component(id, |t: &TransformComponent| t.position))
        .unwrap_or(Vec3f32::zero());
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          let offset = c.position - cursor_pos;
          let dist = offset.length().min(0.1);
          let pan_speed = dist * 0.001;
          let right = c
            .rotation
            .rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
          let up = c
            .rotation
            .rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
          let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);
          c.position = c.position + translation;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:PanCamera | camera doesn't have TranformComponent",
        ))?;
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
    LogicCommand::FeedbackGetSceneTimeScale { scene_id } => {
      let scenes = ctx.scenes.read();
      let scene_ctx = scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let scale = scene_ctx.read().time_state.current_scale;
      Ok(SimulationTaskResult::U64(match scale {
        crate::simulation_api::structs::TimeScale::Stopped => 0,
        crate::simulation_api::structs::TimeScale::OneDay => 1,
        crate::simulation_api::structs::TimeScale::OneWeek => 2,
        crate::simulation_api::structs::TimeScale::OneMonth => 3,
      }))
    }
    LogicCommand::FeedbackGetSceneDateTimeUTC { scene_id } => {
      let scenes = ctx.scenes.read();
      let scene_ctx = scenes.get(&scene_id).ok_or(EngineError::InvalidOperation("scene not found"))?;
      let _utc_str = scene_ctx.read().time_state.current_epoch.to_string();
      // Return success with None, as this is currently handled by get_simulation_time_utc
      // TODO: decide if we want to return strings as results
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FeedbackGetSceneDateTimeLimitsUTC { scene_id: _ } => Ok(SimulationTaskResult::None),

    LogicCommand::SetSceneTimeScale { scene_id, scale } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.write().time_state.current_scale = scale;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::SetSceneEpoch { scene_id, epoch_tai_seconds } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.write().time_state.current_epoch = anise::time::Epoch::from_tai_seconds(epoch_tai_seconds);
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PlayScene { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.write().time_state.is_playing = true;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PauseScene { scene_id } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        scene_ctx.write().time_state.is_playing = false;
      }
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::StepScene { scene_id, step_days } => {
      let scenes = ctx.scenes.read();
      if let Some(scene_ctx) = scenes.get(&scene_id) {
        let mut write_ctx = scene_ctx.write();
        write_ctx.time_state.current_epoch = write_ctx.time_state.current_epoch + anise::time::Unit::Day * step_days;
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
      let state = logic_state
        .almanac_data
        .get_ephem_full(spk_id, frame, epoch, false, false)?;
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
    LogicCommand::SimulationTick {
      task_id: _,
      scene_id,
      delta_time,
    } => {
      let scenes = ctx.scenes.read();
      let scene_ctx = scenes
        .get(&scene_id)
        .ok_or(EngineError::InvalidOperation("scene not found"))?
        .read();

      // 1. Update time
      let (current_epoch, step_days) = {
        let mut time_state = scene_ctx.time_state.time_info.write();
        time_state.ut_update();
        drop(time_state);
        
        // Use write to update state since we are mutating it
        drop(scene_ctx);
        let mut scene_ctx_write = scenes.get(&scene_id).unwrap().write();
        
        let step_days = if scene_ctx_write.time_state.is_playing {
          scene_ctx_write.time_state.current_scale.to_days_per_st_second() * delta_time
        } else {
          0.0
        };
        
        scene_ctx_write.time_state.current_epoch = scene_ctx_write.time_state.current_epoch + anise::time::Unit::Day * step_days;
        (scene_ctx_write.time_state.current_epoch, step_days)
      };

      // 2. Update Almanac bodies
      let scene_ctx = scenes.get(&scene_id).unwrap().read();
      let logic_state = ctx.logic_state.read();

      if scene_ctx.scene.should_parallelize() {
        scene_ctx
          .scene
          .query2_mut_par::<TransformComponent, crate::scene::AlmanacPlanet, _>(
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
        scene_ctx
          .scene
          .query2_mut::<TransformComponent, crate::scene::AlmanacPlanet, _>(
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

      // 3. Physics rebuild
      let scenes = ctx.scenes.read();
      let scene_ctx = scenes
        .get(&scene_id)
        .ok_or(EngineError::InvalidOperation("scene not found"))?
        .read();

      let time_info_lock = scene_ctx.time_state.time_info.clone();
      let time_info = time_info_lock.read();

      while time_info.needs_fixed_update() {
        if let Some(ps_lock) = &scene_ctx.physics_scene {
          let mut ps = ps_lock.write();
          *ps = crate::physics::physics_scene::PhysicsScene::build_from_scene(&scene_ctx.scene);
        }
        
        // Advance fixed clock step
        time_info.ut_fixed_update();
      }

      if let Some((target_id, _)) = scene_ctx.scene.query1_first_res::<FollowingComponent, _, _>(|id, _| Some(id)) {
        if let Some((cursor_id, _)) = scene_ctx.scene.query1_first_res::<CursorComponent, _, _>(|id, _| Some(id)) {
          let _ = try_snap_entity(cursor_id, target_id, &scene_ctx);
        }
      }

      // TODO: implement controllers update (camera, cursor, etc.)

      Ok(SimulationTaskResult::None)
    }
    LogicCommand::Custom {
      custom_fn,
      user_data,
    } => {
      let ptr = user_data.map(|p| p.0).unwrap_or(core::ptr::null_mut());
      custom_fn(ctx, ptr);
      Ok(SimulationTaskResult::None)
    }
  }
}

fn try_snap_entity(
  snap_entity: EntityId,
  target_entity: EntityId,
  scene_read: &RwLockReadGuard<SceneContext>,
) -> EngineResult<()> {
  let (target_pos, target_rot) = {
    let mut t =
      scene_read
        .scene
        .global_transform(target_entity)
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:SnapToEntity | snap target doesn't have TransformComponent",
        ))?;
    let offset = Vec3f32::from_components(0.0, -10.0, 0.0);
    t.position -= offset;
    (t.position, t.rotation)
  };
  scene_read
    .scene
    .set_global_position_and_rotation(snap_entity, target_pos, target_rot)
}
