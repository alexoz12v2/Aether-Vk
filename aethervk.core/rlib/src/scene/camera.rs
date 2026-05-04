use crate::scene::{CameraComponent, EntityId, HasComponentResultEnum, Scene, TransformComponent};
use crate::types::{EngineError, EngineResult};
use aethervk_oshal_rlib::math::FloatLike;
use aethervk_oshal_rlib::math::floating::FloatOps;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};

// TODO add unit tests

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
    center_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
  ) -> EngineResult<()>;

  /// Translates the camera in its local space (x = right, y = backward, z = up)
  fn translate_camera_local(&self, camera_entity: EntityId, delta: Vec3f32) -> EngineResult<()>;
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
    center_entity: EntityId,
    delta_yaw: f32,
    delta_pitch: f32,
  ) -> EngineResult<()> {
    check_for_camera(&self, camera_entity)?;

    let center_pos = self
      .with_component(center_entity, |t: &TransformComponent| t.position)
      .ok_or(EngineError::InvalidOperation(
        "[SceneCameraExt] orbit_camera: center entity transform not found",
      ))?;

    self
      .with_component_mut(camera_entity, |t: &mut TransformComponent| {
        let distance = {
          let v = t.position - center_pos;
          if v.length_squared() > 0.001 {
            v.length()
          } else {
            0.1_f32
          }
        };
        let (pitch, yaw) = updated_pitch_yaw(&t, delta_pitch, delta_yaw);
        let q = Quat::from_pitch_and_yaw_radians(pitch, yaw);

        // Offset starts at world North (+Y)
        // Camera Forward is -Y, so placing it at +Y distance means it looks at origin
        let offset = q.rotate_vector(Vec3f32::from_components(0.0, distance, 0.0));

        t.position = center_pos + offset;
        t.rotation = q;
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
}

// TODO probably move somewhere
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
