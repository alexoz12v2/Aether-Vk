
extern crate std;

use super::*;

#[test]
fn test_initialization_and_conversion() {
  let v = vec3(1.0, 2.0, 3.0);

  assert_eq!(v.x(), 1.0);
  assert_eq!(v.y(), 2.0);
  assert_eq!(v.z(), 3.0);

  let arr: [f32; 3] = v.into();
  assert_eq!(arr, [1.0, 2.0, 3.0]);

  let v_from_arr = Vec3f32::from_array([4.0, 5.0, 6.0]);
  assert_eq!(v_from_arr, vec3(4.0, 5.0, 6.0));
}

#[test]
fn test_zero_and_splat() {
  assert_eq!(Vec3f32::zero(), vec3(0.0, 0.0, 0.0));
  assert_eq!(Vec3f32::splat(7.0), vec3(7.0, 7.0, 7.0));
}

#[test]
fn test_addition() {
  let v1 = vec3(1.0, 2.0, 3.0);
  let v2 = vec3(10.0, 20.0, 30.0);

  assert_eq!(v1 + v2, vec3(11.0, 22.0, 33.0));

  let mut v_assign = v1;
  v_assign += v2;
  assert_eq!(v_assign, vec3(11.0, 22.0, 33.0));
}

#[test]
fn test_subtraction() {
  let v1 = vec3(10.0, 20.0, 30.0);
  let v2 = vec3(1.0, 2.0, 3.0);

  assert_eq!(v1 - v2, vec3(9.0, 18.0, 27.0));

  let mut v_assign = v1;
  v_assign -= v2;
  assert_eq!(v_assign, vec3(9.0, 18.0, 27.0));
}

#[test]
fn test_multiplication() {
  let v1 = vec3(2.0, 3.0, 4.0);
  let v2 = vec3(3.0, 4.0, 5.0);

  // Vec * Vec
  assert_eq!(v1 * v2, vec3(6.0, 12.0, 20.0));

  // Vec * Scalar
  assert_eq!(v1 * 2.0, vec3(4.0, 6.0, 8.0));

  // Scalar * Vec
  assert_eq!(2.0 * v1, vec3(4.0, 6.0, 8.0));

  // Assign traits
  let mut v_assign_vec = v1;
  v_assign_vec *= v2;
  assert_eq!(v_assign_vec, vec3(6.0, 12.0, 20.0));

  let mut v_assign_scalar = v1;
  v_assign_scalar *= 2.0;
  assert_eq!(v_assign_scalar, vec3(4.0, 6.0, 8.0));
}

#[test]
fn test_division() {
  let v1 = vec3(10.0, 20.0, 30.0);
  let v2 = vec3(2.0, 4.0, 5.0);

  assert_eq!(v1 / v2, vec3(5.0, 5.0, 6.0));
  assert_eq!(v1 / 2.0, vec3(5.0, 10.0, 15.0));

  let mut v_assign_vec = v1;
  v_assign_vec /= v2;
  assert_eq!(v_assign_vec, vec3(5.0, 5.0, 6.0));

  let mut v_assign_scalar = v1;
  v_assign_scalar /= 2.0;
  assert_eq!(v_assign_scalar, vec3(5.0, 10.0, 15.0));
}

#[test]
fn test_negation() {
  let v = vec3(1.0, -2.0, 3.0);
  let neg_v = -v;

  assert_eq!(neg_v, vec3(-1.0, 2.0, -3.0));
}

#[test]
fn test_dot_product() {
  let v1 = vec3(1.0, 2.0, 3.0);
  let v2 = vec3(2.0, 3.0, 4.0);

  // 1*2 + 2*3 + 3*4 = 2 + 6 + 12 = 20
  assert_eq!(v1.dot(v2), 20.0);

  // Ensure the w component doesn't bleed into the 3D dot product
  // (Even if underlying Vec4f32 had a non-zero w, though our constructors force 0.0)
  let v3 = Vec3f32(Vec4f32::from_components(1.0, 1.0, 1.0, 100.0));
  let v4 = Vec3f32(Vec4f32::from_components(1.0, 1.0, 1.0, 100.0));
  assert_eq!(v3.dot(v4), 3.0); // Should be 1*1 + 1*1 + 1*1 = 3, ignoring the 100s
}

#[test]
fn test_cross_product() {
  // Standard basis vectors
  let x = vec3(1.0, 0.0, 0.0);
  let y = vec3(0.0, 1.0, 0.0);
  let z = vec3(0.0, 0.0, 1.0);

  // X x Y = Z
  assert_eq!(x.cross(y), z);
  // Y x Z = X
  assert_eq!(y.cross(z), x);
  // Z x X = Y
  assert_eq!(z.cross(x), y);
  // Y x X = -Z
  assert_eq!(y.cross(x), -z);

  // Arbitrary vectors
  let v1 = vec3(1.0, 2.0, 3.0);
  let v2 = vec3(4.0, 5.0, 6.0);
  assert_eq!(v1.cross(v2), vec3(-3.0, 6.0, -3.0));
}

#[test]
fn test_min_max() {
  let v1 = vec3(1.0, 5.0, 3.0);
  let v2 = vec3(2.0, 4.0, 6.0);

  assert_eq!(v1.min(v2), vec3(1.0, 4.0, 3.0));
  assert_eq!(v1.max(v2), vec3(2.0, 5.0, 6.0));
}

#[test]
fn test_indexing_and_components() {
  let mut v = vec3(1.0, 2.0, 3.0);

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

  assert_eq!(v, vec3(1.0, 5.0, 6.0));
}
