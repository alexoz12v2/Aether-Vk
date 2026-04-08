use aethervk_core_rlib::scene::{CameraComponent, EntityId, Scene, TransformComponent};
use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::FloatOps,
  matrix::{Matrix4, MatrixVectorMul, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use std::sync::{Arc, RwLock, mpsc};

pub enum LogicCommand {
  RotateCamera { delta_x: f32, delta_y: f32 },
  ZoomCamera { amount: f32 },
  PanCursor { delta_x: f32, delta_y: f32 },
  MoveCursor { axis: Vec3f32, amount: f32 },
  RaycastCursor { ndc_x: f32, ndc_y: f32 },
  ResetCursor,
  Resize { width: u32, height: u32 },
}

pub struct LogicState {
  yaw: f32,
  pitch: f32,
  camera_distance: f32,
}

pub fn start_logic_thread(
  rx: mpsc::Receiver<LogicCommand>,
  scene_shared: Arc<RwLock<Scene>>,
  camera_entity: EntityId,
  cursor_entity: EntityId,
) {
  std::thread::spawn(move || {
    let mut state = LogicState {
      yaw: 0.0,
      pitch: 0.0,
      camera_distance: 5.0,
    };

    for command in rx {
      let mut scene_guard = scene_shared.write().unwrap();

      // Update state based on command
      let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
      let mut cam_rot = Quat::identity();
      scene_guard.with_component(camera_entity, |c: &TransformComponent| {
        cam_pos = c.position;
        cam_rot = c.rotation;
      });

      let mut cursor_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
      scene_guard.with_component(cursor_entity, |c: &TransformComponent| {
        cursor_pos = c.position;
      });

      let mut offset = cam_pos - cursor_pos;
      let mut dist = offset.length();
      if dist < 0.1 { dist = 0.1; }

      match command {
        LogicCommand::RotateCamera { delta_x, delta_y } => {
          let rotation_speed = 0.005;
          state.yaw += delta_x * rotation_speed;
          state.pitch -= delta_y * rotation_speed;

          state.yaw = state.yaw.fmod(<f32 as FloatOps>::PI * 2.0);
          state.pitch = state.pitch.clamp(-1.55, 1.55);

          let rotation_y = Quat::from_axis_angle(Vec3f32::from_components(0.0, 1.0, 0.0), state.yaw);
          let rotation_x = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), state.pitch);
          let new_rot = rotation_y * rotation_x;

          let rot_delta = new_rot * cam_rot.conjugate();
          let new_offset = rot_delta.rotate_vector(offset);

          scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
            c.position = cursor_pos + new_offset;
            c.rotation = new_rot;
          });
        }
        LogicCommand::ZoomCamera { amount } => {
          let zoom_speed = dist * 0.1;
          let mut new_dist = dist - amount * zoom_speed;
          if new_dist < 0.1 {
            new_dist = 0.1;
          }
          let new_offset = offset.normalize() * new_dist;
          scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
            c.position = cursor_pos + new_offset;
          });
        }
        LogicCommand::PanCursor { delta_x, delta_y } => {
          // Pan on the camera's local X/Y plane
          let pan_speed = dist * 0.001;

          let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
          let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 1.0, 0.0));
          let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

          scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
            c.position = c.position + translation;
          });
          scene_guard.with_component_mut(camera_entity, |c: &mut TransformComponent| {
            c.position = c.position + translation;
          });
        }
        LogicCommand::MoveCursor { axis, amount } => {
          scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
            c.position = c.position + axis * amount;
          });
        }
        LogicCommand::ResetCursor => {
          scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
            c.position = Vec3f32::from_components(0.0, 0.0, 0.0);
          });
        }
        LogicCommand::Resize { width, height } => {
          scene_guard.with_component_mut(camera_entity, |c: &mut CameraComponent| {
            c.projection = Mat4x4f32::perspective_vk(
              45.0f32.to_radians(),
              width as f32 / height as f32,
              0.1,
              100.0,
            );
          });
        }
        LogicCommand::RaycastCursor { ndc_x, ndc_y } => {
          let mut view_proj_inv = Mat4x4f32::identity();
          let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);

          scene_guard.with_component(camera_entity, |c: &TransformComponent| {
            cam_pos = c.position;
            // The view matrix is the inverse of the camera transform
            let view = Mat4x4f32::from_quat(c.rotation.conjugate())
              * Mat4x4f32::translation(c.position * -1.0);

            scene_guard.with_component(camera_entity, |cam: &CameraComponent| {
              let proj = cam.projection;
              let view_proj = proj * view;
              view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
            });
          });

          // NDC ray
          let ndc_near = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
            ndc_x, ndc_y, 0.0, 1.0,
          );
          let ndc_far = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
            ndc_x, ndc_y, 1.0, 1.0,
          );

          let mut world_near = view_proj_inv.mul_vector(ndc_near);
          let mut world_far = view_proj_inv.mul_vector(ndc_far);

          if world_near.w() != 0.0 {
            world_near = world_near / world_near.w();
          }
          if world_far.w() != 0.0 {
            world_far = world_far / world_far.w();
          }

          let ray_origin = Vec3f32::from_components(world_near.x(), world_near.y(), world_near.z());
          let ray_target = Vec3f32::from_components(world_far.x(), world_far.y(), world_far.z());

          let ray_dir = (ray_target - ray_origin).normalize();

          // Intersect with XZ plane (y = 0)
          // ray_origin.y + t * ray_dir.y = 0
          // t = -ray_origin.y / ray_dir.y

          let max_distance = 2.0;
          let mut target_pos = ray_origin + ray_dir * max_distance; // Default if no intersect within range

          if ray_dir.y().abs() > 1e-6 {
            let t = -ray_origin.y() / ray_dir.y();
            if t > 0.0 && t <= max_distance {
              target_pos = ray_origin + ray_dir * t;
            }
          }

          scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
            c.position = target_pos;
          });
        }
      }

      // After applying commands, enforce constraints and update dependent transforms

      // 1. Update cursor scale based on distance to ensure it's visible
      let new_cam_pos = scene_guard.with_component(camera_entity, |c: &TransformComponent| c.position).unwrap();
      let new_cursor_pos = scene_guard.with_component(cursor_entity, |c: &TransformComponent| c.position).unwrap();
      let new_dist = (new_cam_pos - new_cursor_pos).length();
      
      let scale_factor = (new_dist * 0.01).clamp(0.02, 0.05);
      scene_guard.with_component_mut(cursor_entity, |c: &mut TransformComponent| {
        c.scale = Vec3f32::from_components(scale_factor, scale_factor, scale_factor);
      });
    }
  });
}
