//! constants module.

// Simulation Scale Factors
// A single, unified scale factor for all distances (1 engine unit = 10,000,000 km)
/// TODO: Document this item
pub const DISTANCE_SCALE_FACTOR: f64 = 10_000_000.0;

// Artificial visual multipliers to make astronomical bodies visible when viewing the solar system.
// Keeping this the same for both ensures the Sun and planets are perfectly proportionally sized relative to each other!
/// TODO: Document this item
pub const UNIVERSAL_VISUAL_SCALE: f32 = 50.0;
/// TODO: Document this item
pub const PLANET_VISUAL_SCALE: f32 = UNIVERSAL_VISUAL_SCALE;

/// TODO: Document this item
pub struct BarycenterNaifId;

impl BarycenterNaifId {
  /// TODO: Document this item
  pub const SSB: i32 = 0;
  /// TODO: Document this item
  pub const MERCURY: i32 = 1;
  /// TODO: Document this item
  pub const VENUS: i32 = 2;
  /// TODO: Document this item
  pub const EARTH_MOON: i32 = 3;
  /// TODO: Document this item
  pub const MARS: i32 = 4;
  /// TODO: Document this item
  pub const JUPITER: i32 = 5;
  /// TODO: Document this item
  pub const SATURN: i32 = 6;
  /// TODO: Document this item
  pub const URANUS: i32 = 7;
  /// TODO: Document this item
  pub const NEPTUNE: i32 = 8;
  /// TODO: Document this item
  pub const PLUTO: i32 = 9;
}

// All sizes are defined in kilometers (km) based on NASA JPL Planetary Constants (PCK00011)
/// TODO: Document this item
pub struct PlanetRadii;

impl PlanetRadii {
  /// TODO: Document this item
  pub const SUN: f32 = 695700.0;
  /// TODO: Document this item
  pub const MERCURY: f32 = 2439.7;
  /// TODO: Document this item
  pub const VENUS: f32 = 6051.8;
  /// TODO: Document this item
  pub const EARTH: f32 = 6378.1366;
  /// TODO: Document this item
  pub const MOON: f32 = 1737.4;
  /// TODO: Document this item
  pub const MARS: f32 = 3396.19;
  /// TODO: Document this item
  pub const JUPITER: f32 = 71492.0;
  /// TODO: Document this item
  pub const SATURN: f32 = 60268.0;
  /// TODO: Document this item
  pub const URANUS: f32 = 25559.0;
  /// TODO: Document this item
  pub const NEPTUNE: f32 = 24764.0;
  /// TODO: Document this item
  pub const PLUTO: f32 = 1188.3;
}

/// TODO: Document this item
pub struct PlanetNaifId;

impl PlanetNaifId {
  /// TODO: Document this item
  pub const SUN: i32 = 10;
  /// TODO: Document this item
  pub const MERCURY: i32 = 199;
  /// TODO: Document this item
  pub const VENUS: i32 = 299;
  /// TODO: Document this item
  pub const EARTH: i32 = 399;
  /// TODO: Document this item
  pub const MOON: i32 = 301;
  /// TODO: Document this item
  pub const MARS: i32 = 499;
  /// TODO: Document this item
  pub const JUPITER: i32 = 599;
  /// TODO: Document this item
  pub const SATURN: i32 = 699;
  /// TODO: Document this item
  pub const URANUS: i32 = 799;
  /// TODO: Document this item
  pub const NEPTUNE: i32 = 899;
  /// TODO: Document this item
  pub const PLUTO: i32 = 999;
}

/// TODO: Document this item
pub struct FrameId;

impl FrameId {
  /// TODO: Document this item
  pub const J2000: i32 = 1;
}
