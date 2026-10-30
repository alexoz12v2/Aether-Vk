use super::*;

#[test]
fn test_almanac_planet_creation() {
  let planet = AlmanacPlanet::new(399);
  assert_eq!(planet.naif_id, 399);
}

#[test]
fn iau_days_count_from_j2000_tdb() {
  use anise::time::{Epoch, TimeScale};
  // J2000 TDB itself: d = 0, so W = prime_meridian there
  let j2000 = Epoch::from_gregorian(2000, 1, 1, 12, 0, 0, 0, TimeScale::TDB);
  assert!(tdb_days_since_j2000(j2000).abs() * 86400.0 < 1e-6);
  // the same instant labelled TAI or UTC gives the same d (only the instant matters)
  assert!((tdb_days_since_j2000(j2000.to_time_scale(TimeScale::TAI)) * 86400.0).abs() < 1e-3);
  assert!((tdb_days_since_j2000(j2000.to_time_scale(TimeScale::UTC)) * 86400.0).abs() < 1e-3);
  // hifitime's J2000_REF_EPOCH is not J2000 TDB (the bug this guards against): measured
  // 1 day + 32.184 s past it in hifitime 4.2.6
  let dt = tdb_days_since_j2000(anise::time::J2000_REF_EPOCH) * 86400.0;
  assert!((dt - (86400.0 + 32.184)).abs() < 1e-3, "J2000_REF_EPOCH at {dt} s TDB");
  // the review save's start: 2021-12-12 00:00 TDB is 8015.5 days past J2000
  let start = Epoch::from_gregorian(2021, 12, 12, 0, 0, 0, 0, TimeScale::TDB);
  assert!((tdb_days_since_j2000(start) - 8015.5).abs() * 86400.0 < 1e-3);
}
