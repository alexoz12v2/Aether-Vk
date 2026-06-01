use crate::scene::Component;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3f64::Vec3f64, vec4::Quat},
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
