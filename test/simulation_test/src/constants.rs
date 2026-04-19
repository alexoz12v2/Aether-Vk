// Simulation Scale Factors
// A single, unified scale factor for all distances (1 engine unit = 10,000,000 km)
pub const DISTANCE_SCALE_FACTOR: f64 = 10_000_000.0;

// Artificial visual multipliers to make astronomical bodies visible when viewing the solar system.
// Keeping this the same for both ensures the Sun and planets are perfectly proportionally sized relative to each other!
pub const UNIVERSAL_VISUAL_SCALE: f32 = 50.0;
pub const PLANET_VISUAL_SCALE: f32 = UNIVERSAL_VISUAL_SCALE;

pub struct BarycenterNaifId;

impl BarycenterNaifId {
  pub const SSB: i32 = 0;
  pub const MERCURY: i32 = 1;
  pub const VENUS: i32 = 2;
  pub const EARTH_MOON: i32 = 3;
  pub const MARS: i32 = 4;
  pub const JUPITER: i32 = 5;
  pub const SATURN: i32 = 6;
  pub const URANUS: i32 = 7;
  pub const NEPTUNE: i32 = 8;
  pub const PLUTO: i32 = 9;
}

// All sizes are defined in kilometers (km) based on NASA JPL Planetary Constants (PCK00011)
pub struct PlanetRadii;

impl PlanetRadii {
  pub const SUN: f32 = 695700.0;
  pub const MERCURY: f32 = 2439.7;
  pub const VENUS: f32 = 6051.8;
  pub const EARTH: f32 = 6378.1366;
  pub const MOON: f32 = 1737.4;
  pub const MARS: f32 = 3396.19;
  pub const JUPITER: f32 = 71492.0;
  pub const SATURN: f32 = 60268.0;
  pub const URANUS: f32 = 25559.0;
  pub const NEPTUNE: f32 = 24764.0;
  pub const PLUTO: f32 = 1188.3;
}

pub struct PlanetNaifId;

impl PlanetNaifId {
  pub const SUN: i32 = 10;
  pub const MERCURY: i32 = 199;
  pub const VENUS: i32 = 299;
  pub const EARTH: i32 = 399;
  pub const MOON: i32 = 301;
  pub const MARS: i32 = 499;
  pub const JUPITER: i32 = 599;
  pub const SATURN: i32 = 699;
  pub const URANUS: i32 = 799;
  pub const NEPTUNE: i32 = 899;
  pub const PLUTO: i32 = 999;
}

pub struct FrameId;

impl FrameId {
  pub const J2000: i32 = 1;
}
