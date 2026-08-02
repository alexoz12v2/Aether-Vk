use crate::scene::Component;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{
    Vector, Vector3,
    vec3f64::{DVec3, Vec3f64},
    vec4::Quat,
  },
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformAnimationComponent {
  pub start_pos: Vec3f64,
  pub start_rot: Quat,
  pub target_pos: Vec3f64,
  pub target_rot: Quat,
  pub duration: f32,
  pub elapsed: f32,
  pub is_finished: bool,
}

impl Default for TransformAnimationComponent {
  fn default() -> Self {
    Self {
      start_pos: Vec3f64::from_components(0.0, 0.0, 0.0),
      start_rot: Quat::identity(),
      target_pos: Vec3f64::from_components(0.0, 0.0, 0.0),
      target_rot: Quat::identity(),
      duration: 1.0,
      elapsed: 0.0,
      is_finished: false,
    }
  }
}

impl Component for TransformAnimationComponent {}

impl TransformAnimationComponent {
  /// Smoothly redirects an active animation towards a new target.
  /// It prevents "snapping" by establishing the current mid-air position s the new starting point,
  /// and preserves the original movement speed by scaling the duration accodingly
  pub fn retarget(&mut self, new_target_pos: DVec3, new_target_rot: Quat) {
    // 1. Evaluate the exact current state of the animation to prevent teleportation
    let t = if self.duration > 0.0 {
      self.elapsed / self.duration
    } else {
      1.0
    };

    // We use your exact smoothing function to find true mid-air position
    let smooth_t = hermite_smoothstep(t);
    let current_pos = DVec3::lerp(self.start_pos, self.target_pos, smooth_t as f64);
    let current_rot = Quat::slerp(self.start_rot, self.target_rot, smooth_t);

    // 2. Calculate the original average speed (units per second)
    let old_distance = (self.target_pos - self.start_pos).length();
    let speed = if self.duration > 0.0 {
      old_distance / (self.duration as f64)
    } else {
      0.0
    };

    // 3. Pin the new trajectory to start exactly at the current position
    self.start_pos = current_pos;
    self.start_rot = current_rot;
    self.target_pos = new_target_pos;
    self.target_rot = new_target_rot;

    // Reset timer so that the smoothstep operates cleanly on the new line segment
    // TODO: Check if we get a slowdown by resetting hermite's curve here. if so, remove ease-in
    self.elapsed = 0.0;
    self.is_finished = false;

    // 4. Calculate new duration based on the preserved speed
    let new_distance = (self.target_pos - self.start_pos).length();

    if speed > 1e-6 {
      self.duration = (new_distance / speed) as f32;
    } else {
      // fallback if the animation was previously stationary or purely rotation
      self.duration = self.duration.max(0.001);
    }
  }
}

/// Hermite smoothstep computation by clamping parameter from 0 to 1 before applying the cubic
/// polynomial
pub fn hermite_smoothstep(mut t: f32) -> f32 {
  if t > 1.0 {
    t = 1.0;
  } else if t < 0.0 {
    t = 0.0;
  }

  t * t * (3.0 - 2.0 * t)
}