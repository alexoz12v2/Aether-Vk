use std::fmt::format;
use anise::errors::AlmanacResult;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::os::files::MappedFile;
// Add this line
use crate::constants;

pub fn get_planet_radius(id: i32, assets_dir: &std::path::Path) -> f32 {
  let pck_path = assets_dir.join("planets").join("pck00011.tpc");
  if let Ok(content) = std::fs::read_to_string(&pck_path) {
    let mut in_data = false;
    let mut data_content = String::new();
    for line in content.lines() {
      let trimmed = line.trim();
      if trimmed == "\\begindata" {
        in_data = true;
        continue;
      } else if trimmed == "\\begintext" {
        in_data = false;
        continue;
      }
      if in_data {
        data_content.push_str(line);
        data_content.push(' ');
      }
    }

    let marker = format!("BODY{}_RADII", id);
    if let Some(idx) = data_content.find(&marker) {
      let after_marker = &data_content[idx..];
      if let Some(start) = after_marker.find('(') {
        let after_start = &after_marker[start + 1..];
        if let Some(end) = after_start.find(')') {
          let nums = &after_start[..end];
          if let Some(first_num) = nums.split_whitespace().next() {
            if let Ok(val) = first_num.parse::<f32>() {
              return val;
            }
          }
        }
      }
    }
  }
  use crate::constants::PlanetRadii;
  match id {
    constants::PlanetNaifId::SUN => PlanetRadii::SUN,
    constants::PlanetNaifId::MERCURY => PlanetRadii::MERCURY,
    constants::PlanetNaifId::VENUS => PlanetRadii::VENUS,
    constants::PlanetNaifId::EARTH => PlanetRadii::EARTH,
    constants::PlanetNaifId::MOON => PlanetRadii::MOON,
    constants::PlanetNaifId::MARS => PlanetRadii::MARS,
    constants::PlanetNaifId::JUPITER => PlanetRadii::JUPITER,
    constants::PlanetNaifId::SATURN => PlanetRadii::SATURN,
    constants::PlanetNaifId::URANUS => PlanetRadii::URANUS,
    constants::PlanetNaifId::NEPTUNE => PlanetRadii::NEPTUNE,
    constants::PlanetNaifId::PLUTO => PlanetRadii::PLUTO,
    _ => 1.0,
  }
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
      constants::FrameId::J2000,
      constants::BarycenterNaifId::SSB,
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
          constants::FrameId::J2000,
          constants::BarycenterNaifId::SSB,
          None,
        )
      } else {
        Err(e)
      }
    });

  match state {
    Ok(st) => Vec3f32::from_components(
      (st.radius_km[0] / constants::DISTANCE_SCALE_FACTOR) as f32,
      (st.radius_km[1] / constants::DISTANCE_SCALE_FACTOR) as f32,
      (st.radius_km[2] / constants::DISTANCE_SCALE_FACTOR) as f32,
    ),
    Err(e) => {
      println!(
        "Epoch: {} | Almanac spk_ezr error for NAIF ID {}: {:?}",
        current_epoch, id, e
      );
      Vec3f32::from_components(0.0, 0.0, 0.0)
    }
  }
}

#[derive(Default)]
pub struct AlmanacPackedData {
  /// loaded SPKs. TODO: mapped file and implement whatever to do bytes::BytesMut::from_iter
  pub data: Vec<Vec<u8>>,
  /// the almanac itself
  pub almanac: anise::almanac::Almanac,
}

pub fn load_almanac(path: &std::path::Path) -> anise::errors::AlmanacResult<AlmanacPackedData> {
  let mut result = AlmanacPackedData::default();
  if let Ok(entries) = std::fs::read_dir(path) {
    for entry in entries.flatten() {
      let path = entry.path();
      if path.is_file() {
        println!("Examining file {}", path.display());
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
          if ext == "bsp" {
            if let Some(path_str) = path.to_str() {
              result
                .data
                .push(std::fs::read(path_str).expect(&format!("Couldn't read file {}", path_str)));
              let bytes = bytes::BytesMut::from(result.data.last().unwrap().as_slice());
              println!("Loaded file {}", path_str);
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

  println!("Loaded {} SPK files", result.almanac.num_loaded_spk());
  for (name, _spk) in &result.almanac.spk_data {
    println!("Loading spk: {}", name);
  }

  Ok(result)
}
