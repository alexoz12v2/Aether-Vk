//! pck module.

use aethervk_oshal_rlib::os::fs;
use alloc::{string::String, vec::Vec};

/// TODO: Document this item
pub fn read_body_radii(pck_path: &str, body_id: i32) -> Option<[f64; 3]> {
  let content = String::from_utf8(fs::read(&fs::PathBuf::from(pck_path)).ok()?).ok()?;
  let target = alloc::format!("BODY{}_RADII", body_id);

  for line in content.lines() {
    if line.trim().starts_with(&target) {
      // e.g. BODY10_RADII = ( 696000. 696000. 696000. )
      let parts: Vec<&str> = line.split('(').collect();
      if parts.len() > 1 {
        let nums_str = parts[1].split(')').next().unwrap_or("");
        let nums: Vec<f64> = nums_str.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if nums.len() >= 3 {
          return Some([nums[0], nums[1], nums[2]]);
        }
      }
    }
  }
  None
}
