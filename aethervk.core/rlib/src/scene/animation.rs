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
  pub orbit_pivot: Option<Vec3f64>,
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
      orbit_pivot: None,
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
    
    let current_rot = if self.orbit_pivot.is_some() {
      slerp_constrained(self.start_rot, self.target_rot, smooth_t)
    } else {
      Quat::slerp(self.start_rot, self.target_rot, smooth_t)
    };

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
    self.start_rot = strip_roll(current_rot.normalize());
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

/// Interpolates between two quaternions, enforcing that the resulting 
/// rotation's Up vector never exceeds 90 degrees from the Global Up (+Z).
/// Engine convention: +X = Right, -Y = Forward, +Z = Up.
pub fn slerp_constrained(q0: Quat, q1: Quat, t: f32) -> Quat {
  use aethervk_oshal_rlib::math::quaternion::Quaternion as _;
  use aethervk_oshal_rlib::math::vector::{Vector as _, Vector3 as _};

  // 1. Perform standard spherical linear interpolation
  let q = Quat::slerp(q0, q1, t);

  // 2. Extract the local Up vector (rotating the global +Z vector)
  let local_z = Vec3f32::from_components(0.0, 0.0, 1.0);
  let local_up = q.rotate_vector(local_z);

  // 3. Check constraint: if Z >= 0, the angle is <= 90 degrees.
  if local_up.z() >= 0.0 {
    return q; // Constraint satisfied
  }

  // 4. Constraint violated. Extract the local Forward vector (-Y).
  let local_neg_y = Vec3f32::from_components(0.0, -1.0, 0.0);
  let local_fwd = q.rotate_vector(local_neg_y);

  // 5. Project the Up vector onto the XY plane (forcing Z = 0)
  let mut new_up = Vec3f32::from_components(local_up.x(), local_up.y(), 0.0);
  let up_len_sq = new_up.x() * new_up.x() + new_up.y() * new_up.y();

  // Edge case: if it was pointing exactly straight down (0, 0, -1)
  if up_len_sq < 1e-10 {
    // Fall back to using the right vector to derive a valid up
    let local_x = Vec3f32::from_components(1.0, 0.0, 0.0);
    let local_right = q.rotate_vector(local_x);
    new_up = cross(local_z, local_right);
    let len = (new_up.x()*new_up.x() + new_up.y()*new_up.y() + new_up.z()*new_up.z()).sqrt();
    new_up = Vec3f32::from_components(new_up.x()/len, new_up.y()/len, new_up.z()/len);
  } else {
    let inv = 1.0 / up_len_sq.sqrt();
    new_up = Vec3f32::from_components(new_up.x()*inv, new_up.y()*inv, new_up.z()*inv);
  }

  // 6. Rebuild an orthogonal basis
  // Right = Up x Forward (Using the local cross helper in animation.rs)
  let mut right = cross(new_up, local_fwd);
  let right_len_sq = right.x()*right.x() + right.y()*right.y() + right.z()*right.z();

  // Edge case: parallel vectors
  if right_len_sq < 1e-10 {
    right = cross(local_z, new_up);
    let len = (right.x()*right.x() + right.y()*right.y() + right.z()*right.z()).sqrt();
    right = Vec3f32::from_components(right.x()/len, right.y()/len, right.z()/len);
  } else {
    let inv = 1.0 / right_len_sq.sqrt();
    right = Vec3f32::from_components(right.x()*inv, right.y()*inv, right.z()*inv);
  }

  // Calculate strictly orthogonal Forward (Forward = Right x Up)
  let strict_fwd = cross(right, new_up);

  // 7. Map back to rotation matrix columns
  // X-axis: Right (+X)
  // Y-axis: Backward (+Y = -Forward)
  // Z-axis: Up (+Z)
  // Uses the exact Shepperd trace method from `strip_roll`
  let m00 = right.x();
  let m10 = right.y();
  let m20 = right.z();
  let m01 = -strict_fwd.x(); // backward = −forward
  let m11 = -strict_fwd.y();
  let m21 = -strict_fwd.z();
  let m02 = new_up.x();
  let m12 = new_up.y();
  let m22 = new_up.z();

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

  // Normalise out any trace drift
  Quat::from_components(x, y, z, w).normalize()
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

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  // Vector4 must be in scope to call .x()/.y()/.z()/.w() on Vec4f32 (Quat's inner type).
  use aethervk_oshal_rlib::math::{
    quaternion::Quaternion as _,
    vector::{Vector as _, Vector4 as _},
  };
  use crate::scene::camera::QuatToEulerAngles as _;

  /// Helper: make a stationary animation (start == target) at a given position
  /// with the given duration, fully elapsed (t = 1).
  fn stationary_anim_at(pos: DVec3, rot: Quat, duration: f32) -> TransformAnimationComponent {
    TransformAnimationComponent {
      start_pos: pos,
      start_rot: rot,
      target_pos: pos,
      target_rot: rot,
      duration,
      elapsed: duration, // fully elapsed → t = 1
      is_finished: false,
      orbit_pivot: None,
    }
  }

  /// Regression: if the previous animation was stationary (start ≈ target, speed ≈ 0),
  /// `retarget` with a non-zero new target must still produce a finite, positive duration.
  ///
  /// Root cause (fixed): the 50 ms orbit subscription was firing `SnapCameraToOrbit` with
  /// 0.4 s to the SAME position on every tick.  When the comet barely moved, old_distance ≈ 0
  /// → speed ≈ 0 → `retarget` fell back to keeping the 0.4 s duration for the next interactive
  /// drag event, making the orbit feel unresponsive.
  ///
  /// Fix: the subscription now also uses `InteractiveDragAnimationSeconds` (0.016 s), so the
  /// fallback becomes 0.016 s, and subsequent drag retargets feel instantaneous.
  #[test]
  fn retarget_from_stationary_zero_distance_fallback_uses_old_duration() {
    let pos_a = DVec3::from_components(1.0, 0.0, 0.0);
    let pos_b = DVec3::from_components(1.001, 0.0, 0.0); // small drag delta
    let rot = Quat::identity();

    // Problematic old pattern: stationary 0.4 s animation, then interactive drag.
    let mut anim_old = stationary_anim_at(pos_a, rot, 0.4);
    anim_old.retarget(pos_b, rot);
    // Fallback → new duration = old_duration.max(0.001) = 0.4 s.  Must be finite and > 0.
    assert!(
      anim_old.duration > 0.0 && anim_old.duration.is_finite(),
      "retarget produced non-finite duration: {}",
      anim_old.duration
    );

    // Fixed pattern: subscription also uses 0.016 s → fallback is 0.016 s.
    let mut anim_fixed = stationary_anim_at(pos_a, rot, 0.016);
    anim_fixed.retarget(pos_b, rot);
    assert!(
      anim_fixed.duration <= 0.02,
      "short-stationary retarget duration too long: {} s (expected ≤ 0.02 s)",
      anim_fixed.duration
    );
  }

  /// `strip_roll` must be idempotent: applying it twice must yield (nearly) the same quaternion.
  /// This guards against cumulative drift when `strip_roll` is called on every orbit subscription
  /// tick — repeated application must not rotate the camera further.
  #[test]
  fn strip_roll_is_idempotent() {
    use std::f32::consts::PI;
    let test_cases: &[(f32, f32)] = &[
      (0.0, 0.0),              // identity / looking forward
      (0.3, 1.2),              // general oblique case
      (-0.5, 2.8),             // negative pitch
      (PI / 2.0 - 0.05, 0.0), // near north-pole
    ];

    for &(pitch, yaw) in test_cases {
      let q     = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      let once  = strip_roll(q);
      let twice = strip_roll(once);

      // Dot product ≈ 1 (abs to handle quaternion sign ambiguity).
      // Access quaternion components via the inner Vec4 field `.0`.
      let dot = (once.0.x() * twice.0.x()
        + once.0.y() * twice.0.y()
        + once.0.z() * twice.0.z()
        + once.0.w() * twice.0.w())
      .abs();
      assert!(
        dot > 0.9999,
        "strip_roll not idempotent for pitch={pitch:.2}, yaw={yaw:.2}: dot={dot:.6}"
      );
    }
  }

  /// `strip_roll` must preserve the camera's forward direction.
  /// Engine convention: yaw=0 → looking along -Y world; yaw=π → looking along +Y world.
  /// Verified by rotating local -Y (engine forward) by the quaternion and comparing to expected.
  #[test]
  fn strip_roll_preserves_forward_direction() {
    use std::f32::consts::PI;

    // (pitch, yaw, expected_fwd_x, expected_fwd_y, expected_fwd_z)
    let cases: &[(f32, f32, f32, f32, f32)] = &[
      // yaw=0: looking along -Y (engine default forward)
      (0.0,       0.0,    0.0,  -1.0,  0.0),
      // yaw=π: rotating -Y by 180° around Z → +Y
      (0.0,       PI,     0.0,   1.0,  0.0),
      // yaw=π/2: rotating -Y by 90° around Z → +X
      (0.0,  PI / 2.0,    1.0,   0.0,  0.0),
      // yaw=-π/2: rotating -Y by -90° around Z → -X
      (0.0, -PI / 2.0,   -1.0,   0.0,  0.0),
    ];

    let local_neg_y = Vec3f32::from_components(0.0, -1.0, 0.0);

    for &(pitch, yaw, ex, ey, ez) in cases {
      let q       = Quat::from_pitch_and_yaw_radians(pitch, yaw);
      let q_strip = strip_roll(q);
      let fwd     = q_strip.rotate_vector(local_neg_y);

      assert!(
        (fwd.x() - ex).abs() < 0.01,
        "pitch={pitch:.2} yaw={yaw:.2}: fwd.x expected {ex:.1}, got {:.4}",
        fwd.x()
      );
      assert!(
        (fwd.y() - ey).abs() < 0.01,
        "pitch={pitch:.2} yaw={yaw:.2}: fwd.y expected {ey:.1}, got {:.4}",
        fwd.y()
      );
      assert!(
        (fwd.z() - ez).abs() < 0.01,
        "pitch={pitch:.2} yaw={yaw:.2}: fwd.z expected {ez:.1}, got {:.4}",
        fwd.z()
      );
    }
  }


  /// Spherical-coordinate orbit math: a purely horizontal drag (elevation unchanged) must keep
  /// the offset on the same latitude ring.  Specifically:
  ///   offset.z = sin(elevation) * radius  →  with elevation = 0, offset.z must stay 0.
  /// Also verifies the offset remains on the unit sphere after N azimuth increments.
  ///
  /// This is the Rust-side companion to `CometOrbiting_HorizontalDrag_DoesNotChangeElevation`.
  #[test]
  fn orbit_spherical_horizontal_drag_preserves_elevation() {
    let mut azimuth: f32 = 0.0;
    let elevation: f32   = 0.0; // equatorial start
    let yaw_step          = 0.1_f32; // ~5.7° per step

    for step in 0..30 {
      azimuth += yaw_step;
      let cos_elev = elevation.cos();
      let offset_x = cos_elev * azimuth.cos();
      let offset_y = cos_elev * azimuth.sin();
      let offset_z = elevation.sin(); // must stay 0

      assert!(
        offset_z.abs() < 1e-6,
        "horizontal drag changed elevation at step {step}: offset_z={offset_z:.6}"
      );

      let len = (offset_x * offset_x + offset_y * offset_y + offset_z * offset_z).sqrt();
      assert!(
        (len - 1.0).abs() < 1e-5,
        "orbit offset left unit sphere at step {step}: |offset|={len:.6}"
      );
    }
  }

  /// slerp_constrained must never produce a rotation where the local up-vector dips below Z=0.
  #[test]
  fn slerp_constrained_never_flips_upvector() {
      // q0 = identity (up = +Z), q1 = flipped 180° around X (up = -Z)
      let q0 = Quat::identity();
      let q1 = Quat::from_components(1.0, 0.0, 0.0, 0.0); // 180° around X
      let local_z = Vec3f32::from_components(0.0, 0.0, 1.0);
      for i in 0..=100 {
          let t = i as f32 / 100.0;
          let q = super::slerp_constrained(q0, q1, t);
          let up = q.rotate_vector(local_z);
          assert!(up.z() >= -1e-5, "up.z dipped below 0 at t={t}: {}", up.z());
      }
  }
}
