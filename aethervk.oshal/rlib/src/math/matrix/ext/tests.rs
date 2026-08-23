
use super::*;
use crate::math::{matrix::mat3::Mat3f32, vector::vec3::vec3};

#[test]
fn test_lcp_pgs() {
  // M = [[2, -1], [-1, 2]], but expanded to 3x3 for testing
  let m = Mat3f32::from_array(&[2.0, -1.0, 0.0, -1.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
  let q = vec3(-1.0, -1.0, -1.0);

  let z = m.solve_lcp_pgs(&q, 100, 1e-6);

  // Should be approximately (1, 1, 1)
  assert!((z.x() - 1.0).abs() < 1e-4);
  assert!((z.y() - 1.0).abs() < 1e-4);
  assert!((z.z() - 1.0).abs() < 1e-4);

  // Another test where z should have 0s
  let q2 = vec3(1.0, 1.0, 1.0);
  let z2 = m.solve_lcp_pgs(&q2, 100, 1e-6);

  // Since q is positive and M is positive definite, z = 0 is the solution
  assert!(z2.x().abs() < 1e-4);
  assert!(z2.y().abs() < 1e-4);
  assert!(z2.z().abs() < 1e-4);
}

#[test]
fn test_full_piv_lu() {
  let m = Mat3f32::from_array(&[2.0, 1.0, 1.0, 4.0, -6.0, 0.0, -2.0, 7.0, 2.0]);
  // B vector: (5.0, -2.0, 9.0)
  let b = vec3(5.0, -2.0, 9.0);

  let lu = m.full_piv_lu();
  assert!(!lu.singular);

  let x_opt = lu.solve(&b);
  assert!(x_opt.is_some());

  let x = x_opt.unwrap();
  // Check M * x == b
  let check = m * x;
  assert!((check.x() - b.x()).abs() < 1e-4);
  assert!((check.y() - b.y()).abs() < 1e-4);
  assert!((check.z() - b.z()).abs() < 1e-4);
}
