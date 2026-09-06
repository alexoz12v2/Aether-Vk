use crate::scene::Component;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{
    Vector, Vector3,
    vec3::Vec3f32,
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
  pub duration: f32, // unscaled seconds
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

    // 3. Pin the new trajectory to start exactly at the current position.
    // Normalise start_rot: the mid-slerp quaternion is rarely exactly unit-length;
    // accumulated numerical error would otherwise compound across every retarget call.
    self.start_pos = current_pos;
    self.start_rot = current_rot.normalize();
    self.target_pos = new_target_pos;
    self.target_rot = new_target_rot;

    // Reset timer so that the smoothstep operates cleanly on the new line segment
    // TODO: Check if we get a slowdown by resetting hermite's curve here. if so, remove ease-in
    self.elapsed = 0.0;
    self.is_finished = false;

    // 4. Calculate new duration based on the preserved speed
    let new_distance = (self.target_pos - self.start_pos).length();

    if speed > 1e-6 {
      self.duration = ((new_distance / speed) as f32).max(0.001);
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

/// Strips any roll component from `q` by rebuilding the camera basis from its forward direction
/// alone, constraining world-up to +Z.
///
/// Engine convention: forward = local −Y rotated by `q`.
/// The returned quaternion has the same yaw and pitch as `q` but zero roll.
///
/// Falls back to returning `q` unchanged when the forward vector is degenerate (near-zero length).
///
/// # When to use
/// Apply after every slerp that is part of a continuously-retargeted animation (e.g. orbit
/// tracking). Slerp on SO(3) does not preserve the roll-free subspace, so each retarget cycle
/// can inject a small roll component that compounds over time.
pub fn strip_roll(q: Quat) -> Quat {
  use aethervk_oshal_rlib::math::quaternion::Quaternion as _;

  // Engine forward direction in world space: rotate local −Y by q.
  let local_neg_y = Vec3f32::from_components(0.0, -1.0, 0.0);
  let fwd = q.rotate_vector(local_neg_y);

  let fwd_len_sq = fwd.x() * fwd.x() + fwd.y() * fwd.y() + fwd.z() * fwd.z();
  if fwd_len_sq < 1e-10 {
    return q; // degenerate — return unchanged
  }

  // Normalise forward.
  let inv_len = 1.0 / fwd_len_sq.sqrt();
  let fwd = Vec3f32::from_components(fwd.x() * inv_len, fwd.y() * inv_len, fwd.z() * inv_len);

  // World-up hint: prefer +Z; fall back to −Y when looking nearly straight up/down
  // to avoid a degenerate cross product.
  let up_hint = if fwd.z().abs() < 0.99 {
    Vec3f32::from_components(0.0, 0.0, 1.0) // +Z
  } else {
    Vec3f32::from_components(0.0, -1.0, 0.0) // −Y
  };

  // right = cross(up_hint, fwd),  up = cross(fwd, right)
  // (matches the C# EngineQuatFromBasis convention used throughout CameraService)
  let right = cross(up_hint, fwd);
  let right_len_sq = right.x() * right.x() + right.y() * right.y() + right.z() * right.z();
  if right_len_sq < 1e-10 {
    return q; // degenerate — return unchanged
  }
  let inv_r = 1.0 / right_len_sq.sqrt();
  let right = Vec3f32::from_components(right.x() * inv_r, right.y() * inv_r, right.z() * inv_r);

  let up = cross(fwd, right);

  // Build a column-major 3×3 rotation matrix:
  // col0 = right (+X), col1 = backward (+Y = −fwd), col2 = up (+Z)
  // Then extract a quaternion (Shepperd / trace method, matching EngineQuatFromBasis in C#).
  let m00 = right.x();
  let m10 = right.y();
  let m20 = right.z();
  let m01 = -fwd.x(); // backward = −forward
  let m11 = -fwd.y();
  let m21 = -fwd.z();
  let m02 = up.x();
  let m12 = up.y();
  let m22 = up.z();

  let trace = m00 + m11 + m22;
  let (x, y, z, w);

  if trace > 0.0 {
    let s = (trace + 1.0_f32).sqrt() * 2.0; // s = 4w
    let inv_s = 1.0 / s;
    x = (m21 - m12) * inv_s;
    y = (m02 - m20) * inv_s;
    z = (m10 - m01) * inv_s;
    w = 0.25 * s;
  } else if m00 > m11 && m00 > m22 {
    let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0; // s = 4x
    let inv_s = 1.0 / s;
    x = 0.25 * s;
    y = (m01 + m10) * inv_s;
    z = (m02 + m20) * inv_s;
    w = (m21 - m12) * inv_s;
  } else if m11 > m22 {
    let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0; // s = 4y
    let inv_s = 1.0 / s;
    x = (m01 + m10) * inv_s;
    y = 0.25 * s;
    z = (m12 + m21) * inv_s;
    w = (m02 - m20) * inv_s;
  } else {
    let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0; // s = 4z
    let inv_s = 1.0 / s;
    x = (m02 + m20) * inv_s;
    y = (m12 + m21) * inv_s;
    z = 0.25 * s;
    w = (m10 - m01) * inv_s;
  }

  Quat::from_components(x, y, z, w)
}

/// Cross product for Vec3f32 (not in the Vector3 trait).
#[inline(always)]
fn cross(a: Vec3f32, b: Vec3f32) -> Vec3f32 {
  Vec3f32::from_components(
    a.y() * b.z() - a.z() * b.y(),
    a.z() * b.x() - a.x() * b.z(),
    a.x() * b.y() - a.y() * b.x(),
  )
}
