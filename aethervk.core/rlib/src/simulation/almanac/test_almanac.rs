use super::*;
use anise::time::Epoch;

#[test]
fn test_fetch_earth_position_eclipj2000() {
  let mut data = AlmanacPackedData::default();

  // Load the SPK and BPC files requested by the user
  // Note: Paths are relative to the crate root `aethervk.core/rlib` when running tests
  data.load_almanac("../../assets/planets/de442.bsp").unwrap();
  data.load_almanac("../../assets/earth_latest_high_prec.bpc").unwrap();

  // J2000 epoch
  let epoch = Epoch::from_tdb_seconds(0.0);

  // Fetch Earth (399) position with SUN_ECLIPJ2000
  let state = data
    .get_cartesian_state(
      399,
      crate::simulation::almanac::SUN_ECLIPJ2000.orientation_id,
      crate::simulation::almanac::SUN_ECLIPJ2000.ephemeris_id,
      epoch,
      true,
    )
    .unwrap();

  println!("Earth State (SUN_ECLIPJ2000) at J2000 epoch:");
  println!("Position: {:?}", state.radius_km);
  println!("Velocity: {:?}", state.velocity_km_s);

  assert_ne!(state.radius_km[0], 0.0);
}

#[test]
#[cfg(windows)]
fn spk_unload_accepts_mixed_separators() {
  let mut data = AlmanacPackedData::default();
  let mixed = r"../../assets/planets\de442.bsp";
  let other = r"../../assets\planets/de442.bsp";
  assert_eq!(spk_alias(mixed), spk_alias(other));

  data.load_almanac(mixed).unwrap();
  assert_eq!(data.file_names.len(), 1);
  assert_eq!(data.file_names[0], spk_alias(mixed));
  let loaded = data.almanac.num_loaded_spk();
  assert!(loaded >= 1);

  data.unload_almanac_spk(mixed).unwrap();
  assert!(data.file_names.is_empty());
  assert_eq!(data.almanac.num_loaded_spk(), loaded - 1);
}
