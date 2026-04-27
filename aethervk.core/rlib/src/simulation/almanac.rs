use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::os;
use aethervk_oshal_rlib::os::fs::{ExtensionToStr, FileSystemObject};
use alloc::vec::Vec;
use alloc::string::String;

pub const DISTANCE_SCALE_FACTOR: f64 = 10000000.0;

#[derive(Default)]
pub struct AlmanacPackedData {
  /// loaded SPKs. TODO: mapped file and implement whatever to do bytes::BytesMut::from_iter
  pub data: Vec<Vec<u8>>,
  pub file_names: Vec<String>,
  /// the almanac itself
  pub almanac: anise::almanac::Almanac,
}

pub fn load_almanac(path: &os::fs::Path) -> anise::errors::AlmanacResult<AlmanacPackedData> {
  let mut result = AlmanacPackedData::default();
  if let Ok(entries) = os::fs::read_dir(path) {
    for entry in entries.flatten() {
      let path = entry.path();
      if path.is_file() {
        if let Some(ext) = path.extension() {
          if ext == "bsp" {
            if let Some(path_cow) = path.to_str_unified() {
              let path_str = path_cow.to_str().unwrap();
              let path: &os::fs::Path = path_str.into();
              result.data.push(
                os::fs::read(path).expect(&alloc::format!("Couldn't read file {}", path_str)),
              );
              let bytes = bytes::BytesMut::from(result.data.last().unwrap().as_slice());
              result.almanac = result.almanac.load_from_bytes(bytes, path_str)?;
            }
          }
        }
      }
    }
  }

  if result.almanac.num_loaded_spk() == 0 {
    panic!("No SPK files found");
  }

  aethervk_oshal_rlib::log!("Loaded {} SPK files", result.almanac.num_loaded_spk());

  Ok(result)
}

pub fn get_almanac_pos(
  id: i32,
  current_epoch: anise::time::Epoch,
  almanac: &anise::almanac::Almanac,
) -> Vec3f32 {
  // 1. Attempt to get the precise planet center (e.g., 399 for Earth)
  let state = almanac
    .spk_ezr(
      id,
      current_epoch,
      1, /* J2000 */
      0, /* SSB */
      None,
    )
    // 2. FALLBACK: If the precise ID fails, fall back to the planet's system
    // Barycenter (e.g., 199 / 100 = 1 for Mercury).
    .or_else(|e| {
      let barycenter = id / 100;
      if barycenter > 0 {
        almanac.spk_ezr(
          barycenter,
          current_epoch,
          1, /* J2000 */
          0, /* SSB */
          None,
        )
      } else {
        Err(e)
      }
    });

  match state {
    Ok(st) => {
      let pos = Vec3f32::from_components(
        (st.radius_km[0] / DISTANCE_SCALE_FACTOR) as f32,
        (st.radius_km[1] / DISTANCE_SCALE_FACTOR) as f32,
        (st.radius_km[2] / DISTANCE_SCALE_FACTOR) as f32,
      );

      // Obliquity of the Ecliptic (~23.4 degrees)
      let inclination_rad = -0.4090928f32; // -23.4392811 degrees in radians
      let rot = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), inclination_rad);
      rot.rotate_vector(pos)
    }
    Err(_) => Vec3f32::from_components(0.0, 0.0, 0.0),
  }
}
