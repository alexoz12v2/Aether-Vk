use crate::simulation::constants;

pub fn get_planet_radius(id: i32, assets_dir: &str) -> f32 {
  let pck_path = alloc::format!("{}/planets/pck00011.tpc", assets_dir);
  if let Some(radii) = crate::simulation::pck::read_body_radii(&pck_path, id) {
    return radii[0] as f32;
  }
  match id {
    constants::PlanetNaifId::SUN => constants::PlanetRadii::SUN,
    constants::PlanetNaifId::MERCURY => constants::PlanetRadii::MERCURY,
    constants::PlanetNaifId::VENUS => constants::PlanetRadii::VENUS,
    constants::PlanetNaifId::EARTH => constants::PlanetRadii::EARTH,
    constants::PlanetNaifId::MOON => constants::PlanetRadii::MOON,
    constants::PlanetNaifId::MARS => constants::PlanetRadii::MARS,
    constants::PlanetNaifId::JUPITER => constants::PlanetRadii::JUPITER,
    constants::PlanetNaifId::SATURN => constants::PlanetRadii::SATURN,
    constants::PlanetNaifId::URANUS => constants::PlanetRadii::URANUS,
    constants::PlanetNaifId::NEPTUNE => constants::PlanetRadii::NEPTUNE,
    constants::PlanetNaifId::PLUTO => constants::PlanetRadii::PLUTO,
    _ => 1.0,
  }
}

pub fn generate_gaussian_distribution(
  resolution: usize,
  mean_x: f32,
  mean_y: f32,
  std_dev_x: f32,
  std_dev_y: f32,
) -> crate::math::distribution::Distribution2D {
  let mut weights = alloc::vec::Vec::with_capacity(resolution * resolution);

  for y in 0..resolution {
    for x in 0..resolution {
      let dx = (x as f32 / resolution as f32) - mean_x;
      let dy = (y as f32 / resolution as f32) - mean_y;

      let weight = aethervk_oshal_rlib::math::FloatLike::exp(
        -(dx * dx) / (2.0 * std_dev_x * std_dev_x) - (dy * dy) / (2.0 * std_dev_y * std_dev_y),
      );
      weights.push(weight);
    }
  }

  crate::math::distribution::Distribution2D::new(&weights, resolution, resolution)
}
