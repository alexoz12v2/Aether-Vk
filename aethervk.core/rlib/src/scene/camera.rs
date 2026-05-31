//! camera module.

use crate::{
  scene::{CameraComponent, EntityId, HasComponentResultEnum, HighResTransformComponent, Scene},
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib::math::{
  FloatLike,
  floating::FloatOps,
  quaternion::Quaternion,
  vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};

// TODO add unit tests

/// TODO: Document this item
pub trait SceneCameraExt {
  /// yaw fmod to 2pi, while pitch clamped from -pi/2 to +pi/2
  fn rotate_camera(
    &self,
    camera_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
  ) -> EngineResult<()>;

  /// yaw fmod to 2pi, while pitch clamped from -pi/2 to +pi/2
  fn orbit_camera(
    &self,
    camera_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
    pivot_override: Option<Vec3f32>,
  ) -> EngineResult<()>;

  /// Translates the camera in its local space (x = right, y = backward, z = up)
  fn translate_camera_local(&self, camera_entity: EntityId, delta: Vec3f32) -> EngineResult<()>;

  /// Pans the camera along its local X (right) and Z (up) axes.
  fn pan_camera(&self, camera_entity: EntityId, delta_x: f32, delta_y: f32) -> EngineResult<()>;
}

impl SceneCameraExt for Scene {
  fn rotate_camera(
    &self,
    camera_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
  ) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    self
      .with_component_mut(camera_entity, |h: &mut HighResTransformComponent| {
        let (pitch, yaw) = updated_pitch_yaw_highres(&h, delta_pitch, delta_yaw);
        h.rotation = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] rotate_camera: camera entity not found",
      ))?;
    Ok(())
  }

  fn orbit_camera(
    &self,
    camera_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
    pivot_override: Option<Vec3f32>,
  ) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    let focus_distance = self
      .with_component(camera_entity, |c: &CameraComponent| c.focus_distance)
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] orbit_camera: camera entity doesn't have camera component",
      ))?;

    self
      .with_component_mut(camera_entity, |h: &mut HighResTransformComponent| {
        let pos_f32 = h.position.to_f32();

        // 1. Establish the explicit pivot point
        let pivot = pivot_override.unwrap_or_else(|| {
          let fwd = h.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
          pos_f32 + fwd * focus_distance
        });

        // 2. Project current offset into the camera's local space
        let q_old = h.rotation;
        let world_offset = pos_f32 - pivot;

        let old_right = q_old.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
        let old_y = q_old.rotate_vector(Vec3f32::from_components(0.0, 1.0, 0.0));
        let old_up = q_old.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));

        let local_offset = Vec3f32::from_components(
          world_offset.dot(old_right),
          world_offset.dot(old_y),
          world_offset.dot(old_up),
        );

        // 3. Compute the new orientation
        let (mut p, mut y) = q_old.to_pitch_yaw();
        p += delta_pitch;
        y += delta_yaw;
        p = p.clamp(-<f32 as FloatOps>::PI_OVER_2, <f32 as FloatOps>::PI_OVER_2);
        y = y.fmod(<f32 as FloatOps>::PI * 2.0);

        let q_new = Quat::from_pitch_and_yaw_radians(p, y);

        // 4. Transform the local offset back into the NEW world space
        let new_world_offset = q_new.rotate_vector(local_offset);

        // 5. Apply the rigid rotation
        h.position = (pivot + new_world_offset).to_f64();
        h.rotation = q_new;
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] orbit_camera: camera entity not found",
      ))?;
    Ok(())
  }

  fn translate_camera_local(&self, camera_entity: EntityId, delta: Vec3f32) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    self
      .with_component_mut(camera_entity, |h: &mut HighResTransformComponent| {
        let global_delta = h.rotation.rotate_vector(delta);
        h.position = h.position + global_delta.to_f64();
        // Clamp distance
        let dist = h.position.length();
        let max_dist = 1_000.0_f64;
        if dist > max_dist {
          h.position = h.position * (max_dist / dist);
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] translate_camera_local: camera entity not found",
      ))?;
    Ok(())
  }

  fn pan_camera(&self, camera_entity: EntityId, delta_x: f32, delta_y: f32) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    let focus_distance = self
      .with_component(camera_entity, |c: &CameraComponent| c.focus_distance)
      .unwrap_or(1.0);

    self
      .with_component_mut(camera_entity, |h: &mut HighResTransformComponent| {
        let pan_speed = focus_distance * 0.002;
        let right = h.rotation.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
        let up = h.rotation.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
        let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);
        h.position = h.position + translation.to_f64();
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] pan_camera: camera entity not found",
      ))?;
    Ok(())
  }
}

// TODO probably move somewhere
/// TODO: Document this item
pub trait QuatToEulerAngles {
  /// Converts the quaternion to pitch and yaw angles (in radians)
  /// Pitch: Rotation around the X axis (elevation). Positive is looking up (+X)
  /// Yaw: Rotation around the Z axis (heading). 0 is looking forward (-Y)
  fn to_pitch_yaw(self) -> (f32, f32);

  fn from_pitch_and_yaw_radians(pitch: f32, yaw: f32) -> Self;
}

impl QuatToEulerAngles for Quat {
  fn to_pitch_yaw(self) -> (f32, f32) {
    let (w, x, y, z) = (self.0.w(), self.0.x(), self.0.y(), self.0.z());

    // Pitch (rotation around X axis)
    let sin_pitch = 2.0 * (w * x + y * z);
    let cos_pitch = 1.0 - 2.0 * (x * x + y * y);
    let pitch = sin_pitch.atan2(cos_pitch);

    // Yaw (rotation around Z axis)
    let sin_yaw = 2.0 * (w * z + x * y);
    let cos_yaw = 1.0 - 2.0 * (y * y + z * z);
    let yaw = sin_yaw.atan2(cos_yaw);

    (pitch, yaw)
  }

  fn from_pitch_and_yaw_radians(pitch: f32, yaw: f32) -> Self {
    let half_yaw = yaw * 0.5;
    let half_pitch = pitch * 0.5;

    let (sy, cy) = (half_yaw.sin(), half_yaw.cos());
    let (sp, cp) = (half_pitch.sin(), half_pitch.cos());

    Quat::from_components(
      cy * cp,  // w
      cy * sp,  // x
      sy * sp,  // y
      sy * cp,  // z
    )
  }
}

/// Extracts pitch/yaw from a HighResTransformComponent's rotation and applies deltas
fn updated_pitch_yaw_highres(
  t: &HighResTransformComponent,
  delta_pitch: f32,
  delta_yaw: f32,
) -> (f32, f32) {
  use self::QuatToEulerAngles;
  let (mut pitch, mut yaw) = t.rotation.to_pitch_yaw();
  pitch += delta_pitch;
  yaw += delta_yaw;
  pitch = pitch.clamp(-<f32 as FloatOps>::PI_OVER_2, <f32 as FloatOps>::PI_OVER_2);
  yaw = yaw.fmod(<f32 as FloatOps>::PI * 2.0);
  (pitch, yaw)
}

/// Checks if the entity has a camera component. Used as a guard in all camera functions.
fn check_for_camera(scene: &Scene, camera_entity: EntityId) -> EngineResult<()> {
  match scene.has_component::<CameraComponent>(camera_entity) {
    HasComponentResultEnum::EntityHasComponent => Ok(()),
    _ => Err(EngineError::InvalidOperation(
      "[SceneCameraExt] entity is not a camera",
    )),
  }
}
