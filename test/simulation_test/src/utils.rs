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

