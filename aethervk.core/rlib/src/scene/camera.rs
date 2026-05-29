//! camera module.

use crate::{
  scene::{CameraComponent, EntityId, HasComponentResultEnum, Scene, TransformComponent},
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
      .with_component_mut(camera_entity, |t: &mut TransformComponent| {
        let (pitch, yaw) = updated_pitch_yaw(&t, delta_pitch, delta_yaw);
        t.rotation = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] rotate_camera: camera entity not found",
      ))
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
      .with_component_mut(camera_entity, |t: &mut TransformComponent| {
        // 1. Establish the explicit pivot point
        let pivot = pivot_override.unwrap_or_else(|| {
          let fwd = t.rotation.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
          t.position + fwd * focus_distance
        });

        // 2. Project current offset into the camera's local space
        let q_old = t.rotation;
        let world_offset = t.position - pivot;

        let old_right = q_old.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
        let old_y = q_old.rotate_vector(Vec3f32::from_components(0.0, 1.0, 0.0));
        let old_up = q_old.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));

        // This acts as a manual inverse rotation, mapping the vector into local bounds
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
        t.position = pivot + new_world_offset;
        t.rotation = q_new;
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] orbit_camera: camera entity not found",
      ))
  }

  fn translate_camera_local(&self, camera_entity: EntityId, delta: Vec3f32) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    self
      .with_component_mut(camera_entity, |t: &mut TransformComponent| {
        let global_delta = t.rotation.rotate_vector(delta);
        let new_pos = t.position + global_delta;

        let dist = new_pos.length();
        let max_dist = 1_000.0; // TODO parameter
        if dist > max_dist {
          t.position = (new_pos / dist) * max_dist;
        } else {
          t.position = new_pos;
        }
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] translate_camera_local: camera entity not found",
      ))
  }

  fn pan_camera(&self, camera_entity: EntityId, delta_x: f32, delta_y: f32) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    let focus_distance = self
      .with_component(camera_entity, |c: &CameraComponent| c.focus_distance)
      .unwrap_or(1.0);

    let translation = self
      .with_component(camera_entity, |t: &TransformComponent| {
        let pan_speed = focus_distance * 0.002;
        let right = t.rotation.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
        let up = t.rotation.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
        right * (-delta_x * pan_speed) + up * (delta_y * pan_speed)
      })
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] pan_camera: camera entity not found",
      ))?;

    let _ = self.with_component_mut(camera_entity, |c: &mut TransformComponent| {
      c.position = c.position + translation;
    });

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
    let (x, y, z, w) = (self.0.x(), self.0.y(), self.0.z(), self.0.w());

    // 1. Extract the forward vector (-Y axis transformed by the quaternion)
    let fwd_x = 2.0 * (w * z - x * y);
    let fwd_y = 2.0 * (x * x + z * z) - 1.0;
    let fwd_z = -2.0 * (y * z + w * x);

    // 2. Calculate Pitch
    // Clamp to [-1.0, 1.0] to prevent NaNs from floating point errors
    let pitch_clamped = fwd_z.clamp(-1.0, 1.0);
    // fully qualified path to trait necessary otherwise compiler error -> Universal Function Call Syntax (UFCS)
    let pitch = <f32 as aethervk_oshal_rlib::math::FloatLike>::asin(pitch_clamped);

    // 3. Calculate Yaw
    // we use atan2(x, -y) when facing forward (-Y), x = 0, y = -1 | atan2(0, -(-1)) = 0 rad
    // fully qualified path to trait necessary otherwise compiler error -> Universal Function Call Syntax (UFCS)
    let yaw = <f32 as aethervk_oshal_rlib::math::FloatLike>::atan2(fwd_x, -fwd_y);

    (pitch, yaw)
  }

  fn from_pitch_and_yaw_radians(pitch: f32, yaw: f32) -> Quat {
    let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
    let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), -pitch);
    (yaw_quat * pitch_quat).normalize()
  }
}

fn check_for_camera(scene: &Scene, camera_entity: EntityId) -> EngineResult<()> {
  if !<HasComponentResultEnum as Into<bool>>::into(
    scene.has_component::<CameraComponent>(camera_entity),
  ) {
    return Err(EngineError::InvalidOperation(
      "[SceneCameraExt] rotate_camera: camera entity doesn't have camera component",
    ));
  }
  Ok(())
}

fn updated_pitch_yaw(t: &TransformComponent, delta_pitch: f32, delta_yaw: f32) -> (f32, f32) {
  let (mut p, mut y) = t.rotation.to_pitch_yaw();
  p += delta_pitch;
  y += delta_yaw;
  (
    p.clamp(-<f32 as FloatOps>::PI_OVER_2, <f32 as FloatOps>::PI_OVER_2),
    y.fmod(<f32 as FloatOps>::PI * 2.0),
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::floating::FloatOps;

  #[test]
  fn test_pitch_yaw_conversion() {
    let test_cases = [
      (0.0, 0.0),
      (0.5, 0.0),
      (-0.5, 0.0),
      (0.0, 1.0),
      (0.0, -1.0),
      (1.0, 1.0),
      (-1.0, -1.0),
      (<f32 as FloatOps>::PI_OVER_2 - 0.01, <f32 as FloatOps>::PI),
      (-<f32 as FloatOps>::PI_OVER_2 + 0.01, -<f32 as FloatOps>::PI),
    ];

    for (pitch, yaw) in test_cases {
      let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      let (p_out, y_out) = q.to_pitch_yaw();

      assert!(
        (pitch - p_out).abs() < 1e-4,
        "Pitch mismatch: expected {}, got {}",
        pitch,
        p_out
      );

      // Yaw can wrap around, so we check using complex representation
      let y_diff = (yaw - y_out).abs();
      let wraps = (y_diff - <f32 as FloatOps>::PI * 2.0).abs() < 1e-4 || y_diff < 1e-4;
      assert!(wraps, "Yaw mismatch: expected {}, got {}", yaw, y_out);
    }
  }
}
