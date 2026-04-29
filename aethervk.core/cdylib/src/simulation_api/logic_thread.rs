use spin::RwLockReadGuard;
use aethervk_core_rlib::{
  self as rlib,
  scene::{CameraComponent, CursorComponent, EntityId, FollowingComponent, TransformComponent},
  types::{EngineError, EngineResult},
};
use thingbuf::mpsc;
use aethervk_oshal_rlib::{
  os::{NativeError, ThreadingError},
  self as oshal,
  math::floating::FloatOps,
  math::quaternion::Quaternion,
  math::vector::vec3::Vec3f32,
  math::vector::vec4::Quat,
  math::vector::{Vector, Vector3, Vector4},
  os::pool::WorkloadStatus,
  os::thread,
  os::thread::Thread,
};
use crate::{
  SimulationContext,
  structs::{
    LogicCommand, LogicThreadContext, SceneContext, LogicWorkload, SimulationTaskResult,
    FfiRaycastResult,
  },
};
use alloc::{boxed::Box, string::ToString};

pub fn start_logic_thread(
  logic_rx: mpsc::Receiver<LogicCommand>,
  context: LogicThreadContext,
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
      _ => None,
    };

    let res = process_command_internal(self.cmd.clone(), &self.ctx);

    let cmd_desc = match &self.cmd {
      LogicCommand::ImportModel { path, .. } => alloc::format!("Import model {}", path),
      LogicCommand::LoadAlmanac { path, .. } => alloc::format!("Load almanac {}", path),
      LogicCommand::LoadCometSpk { path, .. } => alloc::format!("Load SPK {}", path),
      LogicCommand::SpawnModelInstance { name, .. } => alloc::format!("Spawn instance {}", name),
      LogicCommand::RaycastNdc { .. } => alloc::format!("Raycast NDC"),
      LogicCommand::Raycast { .. } => alloc::format!("Raycast"),
      _ => alloc::format!("Logic Task"),
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
          crate::simulation_api::emit_breadcrumb(3, &alloc::format!("Failed: {} - {}", cmd_desc, e));
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
    LogicCommand::RotateCamera(crate::structs::RotateCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    }) => {
      let scene_read = scene.read();
      let (cam_pos, cam_rot) = scene_read
        .scene
        .with_component::<TransformComponent, _, _>(
          camera_entity,
          |transform_component: &TransformComponent| {
            (transform_component.position, transform_component.rotation)
          },
        )
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | entity doesn't have TransformComponent",
        ))?;
      let (cursor_pos, _) = scene_read
        .scene
        .query1_first_res::<TransformComponent, _, _>(|_, t| Some(t.position))
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | scene doesn't have cursor",
        ))?;
      let offset = cam_pos - cursor_pos;
      let rotation_speed: f32 = 0.005;
      let yaw_quat = Quat::from_axis_angle(
        Vec3f32::from_components(0.0, 0.0, 1.0),
        -delta_x * rotation_speed,
      );

      let local_right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let pitch_quat = Quat::from_axis_angle(local_right, -delta_y * rotation_speed);

      let combined = pitch_quat * yaw_quat * cam_rot;
      let len_sq = combined.0.dot(combined.0);
      if len_sq < 1e-6 {
        return Ok(SimulationTaskResult::None);
      }
      let new_rot = combined.normalize();

      let rot_delta = new_rot * cam_rot.conjugate();
      let new_offset = rot_delta.rotate_vector(offset);
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = cursor_pos + new_offset;
          c.rotation = new_rot;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | failed to update transform",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::ZoomCamera(crate::structs::ZoomCamera {
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
      let (cursor_pos, _) = scene_read
        .scene
        .query1_first_res::<TransformComponent, _, _>(|_, t| Some(t.position))
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | scene doesn't have cursor",
        ))?;
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          let offset = c.position - cursor_pos;
          let dist = offset.length().min(0.1);
          let zoom_speed = dist * 0.01;
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
    LogicCommand::ResetCamera(crate::structs::ResetCamera {
      camera_entity,
      scene,
    }) => {
      let ssb = Vec3f32::from_components(0.0, 0.0, 0.0);
      let offset = SimulationContext::camera_start_pos();
      let yaw = <f32 as FloatOps>::PI;
      let pitch = 0.0;
      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();
      let scene_read = scene.read();
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = ssb + offset;
          c.rotation = new_rot;
        })
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:ResetCamera | camera entity doesn't have TransformComponent",
        ))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::PanCamera(crate::structs::PanCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    }) => {
      let scene_read = scene.read();
      let (cursor_pos, _) = scene_read
        .scene
        .query1_first_res::<TransformComponent, _, _>(|_, t| Some(t.position))
        .ok_or(EngineError::InvalidOperation(
          "logic_thread:RotateCamera | scene doesn't have cursor",
        ))?;
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
    LogicCommand::MoveCursor(crate::structs::MoveCursor {
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
    LogicCommand::SnapToEntity(crate::structs::SnapToEntity {
      snap_entity,
      target_entity,
      scene,
    }) => {
      let scene_read = scene.read();
      try_snap_entity(snap_entity, target_entity, &scene_read)?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FollowEntity(crate::structs::FollowEntity {
      snap_entity,
      entity_id,
      scene,
      unfollow_other,
    }) => {
      let scene_read = scene.read();

      if unfollow_other {
        scene_read
          .scene
          .remove_component::<FollowingComponent>(entity_id)
          .map_err(|e| EngineError::InvalidOperation(e))?;
      }

      try_snap_entity(snap_entity, entity_id, &scene_read)?;
      scene_read
        .scene
        .add_component(entity_id, FollowingComponent {})
        .map_err(|e| EngineError::from(e))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::UnfollowEntity(crate::structs::UnfollowEntity { entity_id, scene }) => {
      let scene_read = scene.read();
      scene_read
        .scene
        .remove_component::<FollowingComponent>(entity_id)
        .map_err(|e| EngineError::InvalidOperation(e))?;
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FeedbackGetTimeScale => {
      let scale = ctx.logic_state.read().current_scale;
      Ok(SimulationTaskResult::U64(match scale {
        crate::structs::TimeScale::Stopped => 0,
        crate::structs::TimeScale::OneDay => 1,
        crate::structs::TimeScale::OneWeek => 2,
        crate::structs::TimeScale::OneMonth => 3,
      }))
    }
    LogicCommand::FeedbackGetDateTimeUTC => {
      let utc_str = ctx.logic_state.read().current_epoch.to_string();
      // Return success with None, as this is currently handled by get_simulation_time_utc
      // TODO: decide if we want to return strings as results
      Ok(SimulationTaskResult::None)
    }
    LogicCommand::FeedbackGetDateTimeLimitsUTC => Ok(SimulationTaskResult::None),

    LogicCommand::ImportModel { task_id: _, path } => {
      let mut scenes = ctx.scenes.write();
      let model_id = scenes.import_model_internal(&path)?;
      Ok(SimulationTaskResult::U64(model_id))
    }
    LogicCommand::LoadAlmanac { task_id: _, path } => {
      let success = ctx.load_almanac_file_internal(&path)?;
      Ok(SimulationTaskResult::Bool(success))
    }
    LogicCommand::LoadCometSpk {
      task_id: _,
      path,
      spkid,
    } => {
      let success = ctx.load_comet_spk_internal(&path, spkid)?;
      Ok(SimulationTaskResult::Bool(success))
    }
    LogicCommand::SpawnModelInstance {
      task_id: _,
      model_id,
      name,
    } => {
      let mut scenes = ctx.scenes.write();
      let instance_id = scenes.spawn_model_instance_internal(model_id, &name)?;
      Ok(SimulationTaskResult::U64(instance_id))
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
      // 1. Update time
      let (current_epoch, step_days) = {
        let mut time_info = ctx.time_info.write();
        time_info.ut_update();
        let mut logic_state = ctx.logic_state.write();
        let step_days = logic_state.current_scale.to_days_per_st_second() * delta_time;
        logic_state.current_epoch = logic_state.current_epoch + anise::time::Unit::Day * step_days;
        (logic_state.current_epoch, step_days)
      };

      // 2. Update Almanac bodies
      {
        let scenes = ctx.scenes.read();
        let scene_ctx = scenes
          .get(&scene_id)
          .ok_or(EngineError::InvalidOperation("scene not found"))?
          .read();
        let logic_state = ctx.logic_state.read();
        
        scene_ctx.scene.query2_mut::<TransformComponent, rlib::scene::AlmanacPlanet, _>(
          |_, transform, planet| {
            planet.step(transform, current_epoch, step_days, &logic_state.almanac_data.almanac);
          }
        );
      }

      // 3. Physics rebuild
      let scenes = ctx.scenes.read();
      let scene_ctx = scenes
        .get(&scene_id)
        .ok_or(EngineError::InvalidOperation("scene not found"))?
        .read();

      if let Some(ps_lock) = &scene_ctx.physics_scene {
        let mut ps = ps_lock.write();
        *ps = rlib::physics::physics_scene::PhysicsScene::build_from_scene(&scene_ctx.scene);
      }

      // TODO: implement controllers update (camera, cursor, etc.)
      
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
