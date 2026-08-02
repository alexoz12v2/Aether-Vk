//! utils module.

use crate::simulation::constants;

// TODO: Move elsewhere after probing it for earth
// Should be in kms
pub fn get_planet_radius(id: i32, assets_dir: &str) -> Option<f32> {
  let pck_path = alloc::format!("{}/planets/pck00011.tpc", assets_dir);
  if let Some(radii) = crate::simulation::pck::read_body_radii(&pck_path, id) {
    return Some(radii[0] as f32);
  }
  None
}