use spin::RwLockReadGuard;
use aethervk_oshal_rlib as oshal;
use aethervk_core_rlib as rlib;
use thingbuf::mpsc;
use aethervk_core_rlib::scene::{
  CameraComponent, CursorComponent, EntityId, FollowingComponent, TransformComponent,
};
use aethervk_oshal_rlib::math::floating::FloatOps;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};
use rlib::types::{EngineError, EngineResult};
use oshal::os::{NativeError, ThreadingError};
use oshal::os::thread;
use oshal::os::thread::Thread;
use crate::SimulationContext;
use crate::structs::{LogicCommand, LogicThreadContext, SceneContext};

// TODO: add update step which puts a tasklet upon request from FFI caller thread. it specified
// TODO  a scene to update (probably mesh viewer won't need it as commands are naturally processed)
// TODO each command in process_command into its own function called process_command_{snake_case_command_name}
// TODO process_command should take by reference the context and do something with it (eg feedback, pool usage for tasks, time info)
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
          if let Err(e) = process_command(cmd) {
            oshal::log!("[Logic thread] failed to process command: {:?}", e);
          }
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

fn process_command(command: LogicCommand) -> EngineResult<()> {
  match command {
    LogicCommand::Shutdown => Ok(()),
    LogicCommand::RotateCamera(super::structs::RotateCamera {
      camera_entity,
      scene,
      delta_x,
      delta_y,
    }) => {
      let scene_read = scene.read();
      // TODO figure out how to get global rotation and how to get then local one?
      // TODO process_command
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
        return Ok(());
      }
      let new_rot = combined.normalize();

      let rot_delta = new_rot * cam_rot.conjugate();
      let new_offset = rot_delta.rotate_vector(offset);
      // unwrap: since we queried initial TransformComponent, we can modify it
      scene_read
        .scene
        .with_component_mut(camera_entity, |c: &mut TransformComponent| {
          c.position = cursor_pos + new_offset;
          c.rotation = new_rot;
        })
        .unwrap();
      Ok(())
    }
    LogicCommand::ZoomCamera(super::structs::ZoomCamera {
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
        return Ok(());
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
        ))
    }
    LogicCommand::ResetCamera(super::structs::ResetCamera {
      camera_entity,
      scene,
    }) => {
      // Origin in viewport is Solar System Barycentre
      let ssb = Vec3f32::from_components(0.0, 0.0, 0.0);
      let offset = SimulationContext::CAMERA_START_POS;
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
        ))
    }
    LogicCommand::PanCamera(super::structs::PanCamera {
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
        ))
    }
    // TODO remove this, what does pan cursor even mean?
    LogicCommand::PanCursor(_) => Ok(()),
    LogicCommand::MoveCursor(super::structs::MoveCursor {
      scene,
      delta_x,
      delta_y,
      delta_z,
    }) => {
      // TODO adjust
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
        ))
    }
    LogicCommand::SnapToEntity(super::structs::SnapToEntity {
      snap_entity,
      target_entity,
      scene,
    }) => {
      let scene_read = scene.read();
      try_snap_entity(snap_entity, target_entity, &scene_read)
    }
    LogicCommand::FollowEntity(super::structs::FollowEntity {
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
        .map_err(|e| EngineError::from(e))
    }
    LogicCommand::UnfollowEntity(super::structs::UnfollowEntity { entity_id, scene }) => {
      let scene_read = scene.read();
      scene_read
        .scene
        .remove_component::<FollowingComponent>(entity_id)
        .map_err(|e| EngineError::InvalidOperation(e))
    }
    LogicCommand::FeedbackGetTimeScale => {
      // TODO async query. Integrate pool and feedback
      todo!()
    }
    LogicCommand::FeedbackGetDateTimeUTC => {
      // TODO async query. Integrate pool and feedback
      todo!()
    }
    LogicCommand::FeedbackGetDateTimeLimitsUTC => {
      // TODO async query. Integrate pool and feedback
      todo!()
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
    // TODO tweak
    let offset = Vec3f32::from_components(0.0, -10.0, 0.0);
    t.position -= offset;
    (t.position, t.rotation)
  };
  scene_read
    .scene
    .set_global_position_and_rotation(snap_entity, target_pos, target_rot)
}
