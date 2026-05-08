use alloc::vec::Vec;
use crate::scene::Component;

#[derive(Clone, Debug)]
pub struct TrajectoryComponent {
  pub control_points: Vec<[f32; 4]>, // Homogeneous (x*w, y*w, z*w, w)
  pub color: [f32; 4],
  pub line_width: f32,
  pub texture_id: u32,
  pub subdivisions_per_segment: u32,
}

impl Component for TrajectoryComponent {}

impl TrajectoryComponent {
  pub fn new(
    control_points: Vec<[f32; 4]>,
    color: [f32; 4],
    line_width: f32,
    texture_id: u32,
    subdivisions_per_segment: u32,
  ) -> Self {
    Self {
      control_points,
      color,
      line_width,
      texture_id,
      subdivisions_per_segment,
    }
  }
}
