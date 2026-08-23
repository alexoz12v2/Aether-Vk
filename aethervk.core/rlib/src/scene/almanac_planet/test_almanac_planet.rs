use super::*;

#[test]
fn test_almanac_planet_creation() {
  let planet = AlmanacPlanet::new(399);
  assert_eq!(planet.naif_id, 399);
}
