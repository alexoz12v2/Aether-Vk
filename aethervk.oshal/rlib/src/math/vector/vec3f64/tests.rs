
extern crate std;

use super::*;
use crate::math::FloatLike;

#[test]
fn test_initialization_and_conversion() {
  let v = vec3f64(1.0, 2.0, 3.0);

  assert_eq!(v.x(), 1.0);
  assert_eq!(v.y(), 2.0);
  assert_eq!(v.z(), 3.0);

  let arr: [f64; 3] = v.into();
  assert_eq!(arr, [1.0, 2.0, 3.0]);

  let v_from_arr = Vec3f64::from_array([4.0, 5.0, 6.0]);
  assert_eq!(v_from_arr, vec3f64(4.0, 5.0, 6.0));

  let v_from: Vec3f64 = [7.0, 8.0, 9.0].into();
  assert_eq!(v_from, vec3f64(7.0, 8.0, 9.0));
}

#[test]
fn test_zero_and_splat() {
  assert_eq!(Vec3f64::zero(), vec3f64(0.0, 0.0, 0.0));
  assert_eq!(Vec3f64::splat(7.0), vec3f64(7.0, 7.0, 7.0));
  assert_eq!(Vec3f64::default(), Vec3f64::zero());
}

#[test]
fn test_addition() {
  let v1 = vec3f64(1.0, 2.0, 3.0);
  let v2 = vec3f64(10.0, 20.0, 30.0);

  assert_eq!(v1 + v2, vec3f64(11.0, 22.0, 33.0));

  let mut v_assign = v1;
  v_assign += v2;
  assert_eq!(v_assign, vec3f64(11.0, 22.0, 33.0));
}

#[test]
fn test_subtraction() {
  let v1 = vec3f64(10.0, 20.0, 30.0);
  let v2 = vec3f64(1.0, 2.0, 3.0);

  assert_eq!(v1 - v2, vec3f64(9.0, 18.0, 27.0));

  let mut v_assign = v1;
  v_assign -= v2;
  assert_eq!(v_assign, vec3f64(9.0, 18.0, 27.0));
}

#[test]
fn test_multiplication() {
  let v1 = vec3f64(2.0, 3.0, 4.0);
  let v2 = vec3f64(3.0, 4.0, 5.0);

  // Vec * Vec
  assert_eq!(v1 * v2, vec3f64(6.0, 12.0, 20.0));

  // Vec * Scalar
  assert_eq!(v1 * 2.0, vec3f64(4.0, 6.0, 8.0));

  // Scalar * Vec
  assert_eq!(2.0 * v1, vec3f64(4.0, 6.0, 8.0));

  // Assign traits
  let mut v_assign_vec = v1;
  v_assign_vec *= v2;
  assert_eq!(v_assign_vec, vec3f64(6.0, 12.0, 20.0));

  let mut v_assign_scalar = v1;
  v_assign_scalar *= 2.0;
  assert_eq!(v_assign_scalar, vec3f64(4.0, 6.0, 8.0));
}

#[test]
fn test_division() {
  let v1 = vec3f64(10.0, 20.0, 30.0);
  let v2 = vec3f64(2.0, 4.0, 5.0);

  assert_eq!(v1 / v2, vec3f64(5.0, 5.0, 6.0));
  assert_eq!(v1 / 2.0, vec3f64(5.0, 10.0, 15.0));

  let mut v_assign_vec = v1;
  v_assign_vec /= v2;
  assert_eq!(v_assign_vec, vec3f64(5.0, 5.0, 6.0));

  let mut v_assign_scalar = v1;
  v_assign_scalar /= 2.0;
  assert_eq!(v_assign_scalar, vec3f64(5.0, 10.0, 15.0));
}

#[test]
fn test_negation() {
  let v = vec3f64(1.0, -2.0, 3.0);
  let neg_v = -v;

  assert_eq!(neg_v, vec3f64(-1.0, 2.0, -3.0));
}

#[test]
fn test_dot_product() {
  let v1 = vec3f64(1.0, 2.0, 3.0);
  let v2 = vec3f64(2.0, 3.0, 4.0);

  // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
  assert_eq!(v1.dot(v2), 20.0);
}

#[test]
fn test_cross_product() {
  // Standard basis vectors
  let x = vec3f64(1.0, 0.0, 0.0);
  let y = vec3f64(0.0, 1.0, 0.0);
  let z = vec3f64(0.0, 0.0, 1.0);

  // X x Y = Z
  assert_eq!(x.cross(y), z);
  // Y x Z = X
  assert_eq!(y.cross(z), x);
  // Z x X = Y
  assert_eq!(z.cross(x), y);
  // Y x X = -Z
  assert_eq!(y.cross(x), -z);

  // Arbitrary vectors
  let v1 = vec3f64(1.0, 2.0, 3.0);
  let v2 = vec3f64(4.0, 5.0, 6.0);
  assert_eq!(v1.cross(v2), vec3f64(-3.0, 6.0, -3.0));
}

#[test]
fn test_length_and_normalize() {
  let v = vec3f64(3.0, 4.0, 0.0);
  assert_eq!(v.length_squared(), 25.0);
  assert_eq!(v.length(), 5.0);

  let n = v.normalize();
  let expected = vec3f64(0.6, 0.8, 0.0);
  assert!((n.x() - expected.x()).abs() < 1e-15);
  assert!((n.y() - expected.y()).abs() < 1e-15);
  assert!((n.z() - expected.z()).abs() < 1e-15);

  // Normalized vector should have unit length
  assert!((n.length() - 1.0).abs() < 1e-15);
}

#[test]
fn test_min_max() {
  let v1 = vec3f64(1.0, 5.0, 3.0);
  let v2 = vec3f64(2.0, 4.0, 6.0);

  assert_eq!(v1.min(v2), vec3f64(1.0, 4.0, 3.0));
  assert_eq!(v1.max(v2), vec3f64(2.0, 5.0, 6.0));
}

#[test]
fn test_indexing_and_components() {
  let mut v = vec3f64(1.0, 2.0, 3.0);

  // Index
  assert_eq!(v[0], 1.0);
  assert_eq!(v[1], 2.0);
  assert_eq!(v[2], 3.0);

  // Component method
  assert_eq!(v.component(0), Some(1.0));
  assert_eq!(v.component(1), Some(2.0));
  assert_eq!(v.component(2), Some(3.0));
  assert_eq!(v.component(3), None); // Bounds check

  // IndexMut
  v[1] = 5.0;
  assert_eq!(v[1], 5.0);

  // Set component
  v.set_component(2, 6.0);
  assert_eq!(v.z(), 6.0);

  assert_eq!(v, vec3f64(1.0, 5.0, 6.0));
}

#[test]
fn test_to_f32_conversion() {
  let v64 = vec3f64(1.5, 2.5, 3.5);
  let v32 = v64.to_f32();
  assert_eq!(v32.x(), 1.5f32);
  assert_eq!(v32.y(), 2.5f32);
  assert_eq!(v32.z(), 3.5f32);
}

#[test]
fn test_from_f32_conversion() {
  let v32 = Vec3f32::from_components(1.5, 2.5, 3.5);
  let v64 = Vec3f64::from_f32(v32);
  assert_eq!(v64.x(), 1.5);
  assert_eq!(v64.y(), 2.5);
  assert_eq!(v64.z(), 3.5);
}

#[test]
fn test_f64_precision_advantage() {
  // Demonstrate that f64 preserves precision that f32 cannot.
  // 0.1 + 0.2 is a classic floating-point precision example.
  let a64 = vec3f64(0.1, 0.2, 0.0);
  let b64 = vec3f64(0.2, 0.1, 0.0);
  let sum64 = a64 + b64;

  let a32 = Vec3f32::from_components(0.1, 0.2, 0.0);
  let b32 = Vec3f32::from_components(0.2, 0.1, 0.0);
  let sum32 = a32 + b32;

  // Both should be approximately 0.3, but f64 is much closer
  let err64 = (sum64.x() - 0.3f64).abs();
  let err32 = (sum32.x() as f64 - 0.3f64).abs();

  // f64 error should be smaller than f32 error
  assert!(
    err64 < err32,
    "f64 error ({err64:e}) should be < f32 error ({err32:e})"
  );

  // Verify two very close values are distinguishable in f64 but not in f32
  let close_a = 1.0000000000000002_f64;
  let close_b = 1.0000000000000004_f64;
  let va = vec3f64(close_a, 0.0, 0.0);
  let vb = vec3f64(close_b, 0.0, 0.0);
  assert_ne!(va, vb, "f64 should distinguish these values");

  // The same values collapse in f32
  let va32 = Vec3f32::from_components(close_a as f32, 0.0, 0.0);
  let vb32 = Vec3f32::from_components(close_b as f32, 0.0, 0.0);
  assert_eq!(va32, vb32, "f32 cannot distinguish these values");
}
