
// Bring std into scope conditionally for tests, even if the crate is no_std
extern crate std;

use crate::math::vector::vec3::vec3;

use super::*;
use core::f32;

#[test]
fn test_initialization_and_conversion() {
  let v = vec(1.0, 2.0, 3.0, 4.0);

  assert_eq!(v.x(), 1.0);
  assert_eq!(v.y(), 2.0);
  assert_eq!(v.z(), 3.0);
  assert_eq!(v.w(), 4.0);

  let arr: [f32; 4] = v.into();
  assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_zero_and_splat() {
  let zero = Vec4f32::zero();
  assert_eq!(zero, vec(0.0, 0.0, 0.0, 0.0));

  let splat = Vec4f32::splat(5.0);
  assert_eq!(splat, vec(5.0, 5.0, 5.0, 5.0));
}

#[test]
fn test_partial_eq() {
  let v1 = vec(1.0, 2.0, 3.0, 4.0);
  let v2 = vec(1.0, 2.0, 3.0, 4.0);
  let v3 = vec(1.0, 2.0, 3.0, 5.0); // Differs in w
  let v4 = vec(0.0, 2.0, 3.0, 4.0); // Differs in x

  assert!(v1 == v2);
  assert!(v1 != v3);
  assert!(v1 != v4);
}

#[test]
fn test_addition() {
  let v1 = vec(1.0, 2.0, 3.0, 4.0);
  let v2 = vec(10.0, 20.0, 30.0, 40.0);

  // Add
  assert_eq!(v1 + v2, vec(11.0, 22.0, 33.0, 44.0));

  // AddAssign
  let mut v_assign = v1;
  v_assign += v2;
  assert_eq!(v_assign, vec(11.0, 22.0, 33.0, 44.0));
}

#[test]
fn test_subtraction() {
  let v1 = vec(10.0, 20.0, 30.0, 40.0);
  let v2 = vec(1.0, 2.0, 3.0, 4.0);

  // Sub
  assert_eq!(v1 - v2, vec(9.0, 18.0, 27.0, 36.0));

  // SubAssign
  let mut v_assign = v1;
  v_assign -= v2;
  assert_eq!(v_assign, vec(9.0, 18.0, 27.0, 36.0));
}

#[test]
fn test_multiplication() {
  let v1 = vec(2.0, 3.0, 4.0, 5.0);
  let v2 = vec(3.0, 4.0, 5.0, 6.0);

  // Vector * Vector
  assert_eq!(v1 * v2, vec(6.0, 12.0, 20.0, 30.0));

  // Vector * Scalar
  assert_eq!(v1 * 2.0, vec(4.0, 6.0, 8.0, 10.0));

  // Scalar * Vector
  assert_eq!(2.0 * v1, vec(4.0, 6.0, 8.0, 10.0));

  // MulAssign Vector
  let mut v_assign_vec = v1;
  v_assign_vec *= v2;
  assert_eq!(v_assign_vec, vec(6.0, 12.0, 20.0, 30.0));

  // MulAssign Scalar
  let mut v_assign_scalar = v1;
  v_assign_scalar *= 2.0;
  assert_eq!(v_assign_scalar, vec(4.0, 6.0, 8.0, 10.0));
}

#[test]
fn test_division() {
  let v1 = vec(10.0, 20.0, 30.0, 40.0);
  let v2 = vec(2.0, 4.0, 5.0, 8.0);

  // Vector / Vector
  assert_eq!(v1 / v2, vec(5.0, 5.0, 6.0, 5.0));

  // Vector / Scalar
  assert_eq!(v1 / 2.0, vec(5.0, 10.0, 15.0, 20.0));

  // DivAssign Vector
  let mut v_assign_vec = v1;
  v_assign_vec /= v2;
  assert_eq!(v_assign_vec, vec(5.0, 5.0, 6.0, 5.0));

  // DivAssign Scalar
  let mut v_assign_scalar = v1;
  v_assign_scalar /= 2.0;
  assert_eq!(v_assign_scalar, vec(5.0, 10.0, 15.0, 20.0));
}

#[test]
fn test_negation() {
  let v = vec(1.0, -2.0, 3.0, -0.0);
  let neg_v = -v;

  assert_eq!(neg_v.x(), -1.0);
  assert_eq!(neg_v.y(), 2.0);
  assert_eq!(neg_v.z(), -3.0);
  // Using to_bits to verify exact -0.0 representation if necessary
  assert_eq!(neg_v.w().to_bits(), 0.0f32.to_bits());
}

#[test]
fn test_dot_product() {
  let v1 = vec(1.0, 2.0, 3.0, 4.0);
  let v2 = vec(2.0, 3.0, 4.0, 5.0);

  // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
  assert_eq!(v1.dot(v2), 40.0);
}

#[test]
fn test_min_max() {
  let v1 = vec(1.0, 5.0, 3.0, 7.0);
  let v2 = vec(2.0, 4.0, 6.0, 1.0);

  assert_eq!(v1.min(v2), vec(1.0, 4.0, 3.0, 1.0));
  assert_eq!(v1.max(v2), vec(2.0, 5.0, 6.0, 7.0));
}

#[test]
fn test_indexing_and_components() {
  let mut v = vec(1.0, 2.0, 3.0, 4.0);

  // Index
  assert_eq!(v[0], 1.0);
  assert_eq!(v[1], 2.0);
  assert_eq!(v[2], 3.0);
  assert_eq!(v[3], 4.0);

  // Component method
  assert_eq!(v.component(0), Some(1.0));
  assert_eq!(v.component(4), None); // Out of bounds

  // IndexMut
  v[1] = 5.0;
  assert_eq!(v[1], 5.0);

  // Set component
  v.set_component(2, 6.0);
  assert_eq!(v[2], 6.0);

  assert_eq!(v, vec(1.0, 5.0, 6.0, 4.0));
}

#[test]
fn test_quaternion_indexing() {
  let mut q = Quat(vec(1.0, 2.0, 3.0, 4.0));

  // Index
  assert_eq!(q[0], 1.0);
  assert_eq!(q[1], 2.0);
  assert_eq!(q[2], 3.0);
  assert_eq!(q[3], 4.0);

  // IndexMut
  q[1] = 5.0;
  assert_eq!(q[1], 5.0);

  assert_eq!(q.0, vec(1.0, 5.0, 3.0, 4.0));
}

#[test]
fn test_quaternion_traits() {
  // Constructing a dummy Vec3f32 based on your implementation snippet
  // We assume Vec3f32 is a wrapper around Vec4f32
  let vec3 = Vec3f32(vec(1.0, 2.0, 3.0, 0.0));
  let scalar = 4.0;

  let q = Quat::from_vector_and_scalar(vec3, scalar);

  assert_eq!(q.0, vec(1.0, 2.0, 3.0, 4.0));
  assert_eq!(q.scalar_part(), 4.0);

  let extracted_vec3 = q.vector_part();
  assert_eq!(extracted_vec3.0.x(), 1.0);
  assert_eq!(extracted_vec3.0.y(), 2.0);
  assert_eq!(extracted_vec3.0.z(), 3.0);
}

use core::f32::consts::PI;

// Helper macro for floating point comparisons
macro_rules! assert_approx_eq {
  ($a:expr, $b:expr) => {
    let eps = 1e-5;
    assert!(
      ($a - $b).abs() < eps,
      "assertion failed: `(left !== right)`\n  left: `{:?}`,\n right: `{:?}`",
      $a,
      $b
    );
  };
}

macro_rules! assert_vec3_approx_eq {
  ($v1:expr, $v2:expr) => {
    assert_approx_eq!(
      crate::math::vector::Vector3::x(&$v1),
      crate::math::vector::Vector3::x(&$v2)
    );
    assert_approx_eq!(
      crate::math::vector::Vector3::y(&$v1),
      crate::math::vector::Vector3::y(&$v2)
    );
    assert_approx_eq!(
      crate::math::vector::Vector3::z(&$v1),
      crate::math::vector::Vector3::z(&$v2)
    );
  };
}

#[test]
fn test_quaternion_identity() {
  let id = Quat::identity();
  let v = vec3(1.0, 2.0, 3.0);

  // Rotating by identity should return the exact same vector
  let rotated = id.rotate_vector(v);
  assert_vec3_approx_eq!(rotated, v);
}

#[test]
fn test_quaternion_axis_angle_rotations() {
  let x_axis = vec3(1.0, 0.0, 0.0);
  let y_axis = vec3(0.0, 1.0, 0.0);
  let z_axis = vec3(0.0, 0.0, 1.0);

  // 1. Rotate Y vector 90 degrees around X axis -> Should become Z vector
  let q_rot_x = Quat::from_axis_angle(x_axis, PI / 2.0);
  let rotated_y = q_rot_x.rotate_vector(y_axis);
  assert_vec3_approx_eq!(rotated_y, z_axis);

  // 2. Rotate X vector 90 degrees around Y axis -> Should become -Z vector
  let q_rot_y = Quat::from_axis_angle(y_axis, PI / 2.0);
  let rotated_x = q_rot_y.rotate_vector(x_axis);
  assert_vec3_approx_eq!(rotated_x, vec3(0.0, 0.0, -1.0));

  // 3. Rotate X vector 90 degrees around Z axis -> Should become Y vector
  let q_rot_z = Quat::from_axis_angle(z_axis, PI / 2.0);
  let rotated_x_around_z = q_rot_z.rotate_vector(x_axis);
  assert_vec3_approx_eq!(rotated_x_around_z, y_axis);
}

#[test]
fn test_quaternion_conjugate_and_inverse() {
  let axis = vec3(1.0, 1.0, 1.0).normalize(); // Assuming Vec3 has normalize()
  let q = Quat::from_axis_angle(axis, PI / 3.0); // 60 degrees

  // For unit quaternions, conjugate == inverse
  let conjugate = q.conjugate();
  let inverse = q.inverse();

  assert_approx_eq!(conjugate.0.x(), inverse.0.x());
  assert_approx_eq!(conjugate.0.y(), inverse.0.y());
  assert_approx_eq!(conjugate.0.z(), inverse.0.z());
  assert_approx_eq!(conjugate.0.w(), inverse.0.w());

  // Rotating by q, then by its inverse, should yield the original vector
  let v = vec3(10.0, 0.0, 0.0);
  let rotated = q.rotate_vector(v);
  let unrotated = inverse.rotate_vector(rotated);

  assert_vec3_approx_eq!(v, unrotated);
}

#[test]
fn test_quaternion_slerp() {
  let q1 = Quat::identity();
  let x_axis = vec3(1.0, 0.0, 0.0);
  let q2 = Quat::from_axis_angle(x_axis, PI / 2.0); // 90 degrees

  // Slerp at t = 0.0 should be q1
  let slerp_0 = Quat::slerp(q1, q2, 0.0);
  assert_approx_eq!(slerp_0.0.w(), q1.0.w());

  // Slerp at t = 1.0 should be q2
  let slerp_1 = Quat::slerp(q1, q2, 1.0);
  assert_approx_eq!(slerp_1.0.w(), q2.0.w());

  // Slerp at t = 0.5 should be a 45 degree rotation around X
  let slerp_half = Quat::slerp(q1, q2, 0.5);
  let expected_q = Quat::from_axis_angle(x_axis, PI / 4.0);

  // If dot product is ~1.0 or ~-1.0, they are the same rotation
  assert!((slerp_half.0.dot(expected_q.0)).abs() > 0.9999);
  assert_approx_eq!(slerp_half.0.x(), expected_q.0.x());
  assert_approx_eq!(slerp_half.0.w(), expected_q.0.w());
}

#[test]
fn test_quaternion_norm_and_normalize() {
  // Create a non-unit quaternion manually
  let q = Quat(vec(2.0, 0.0, 0.0, 0.0));

  // Norm should be 2.0
  assert_approx_eq!(q.norm(), 2.0);

  // Normalized should be (1.0, 0.0, 0.0, 0.0)
  let q_norm = Quaternion::normalize(q);
  assert_approx_eq!(q_norm.norm(), 1.0);
  assert_approx_eq!(q_norm.0.x(), 1.0);
}

use crate::math::matrix::mat3::Mat3f32;

#[test]
fn test_quaternion_to_matrix3_identity() {
  let q = Quat::identity();
  let mat: Mat3f32 = q.to_matrix3();

  let x_axis = vec3(1.0, 0.0, 0.0);
  let y_axis = vec3(0.0, 1.0, 0.0);
  let z_axis = vec3(0.0, 0.0, 1.0);

  // The identity quaternion should produce the identity matrix.
  // Multiplying the identity matrix by basis vectors should return the basis vectors.
  assert_vec3_approx_eq!(mat * x_axis, x_axis);
  assert_vec3_approx_eq!(mat * y_axis, y_axis);
  assert_vec3_approx_eq!(mat * z_axis, z_axis);
}

#[test]
fn test_quaternion_to_matrix3_specific_rotation() {
  let x_axis = vec3(1.0, 0.0, 0.0);
  let y_axis = vec3(0.0, 1.0, 0.0);
  let z_axis = vec3(0.0, 0.0, 1.0);

  // Create a quaternion representing a 90-degree (PI/2) rotation around the Y axis
  let q_rot_y = Quat::from_axis_angle(y_axis, PI / 2.0);
  let mat: Mat3f32 = q_rot_y.to_matrix3();

  // Extract the columns of the resulting matrix by multiplying by basis vectors
  let col0 = mat * x_axis;
  let col1 = mat * y_axis;
  let col2 = mat * z_axis;

  // A 90-degree rotation around Y maps:
  // X -> -Z
  // Y -> Y (unchanged)
  // Z -> X
  assert_vec3_approx_eq!(col0, vec3(0.0, 0.0, -1.0));
  assert_vec3_approx_eq!(col1, vec3(0.0, 1.0, 0.0));
  assert_vec3_approx_eq!(col2, vec3(1.0, 0.0, 0.0));
}

#[test]
fn test_quaternion_to_matrix3_action_equivalency() {
  // This test ensures that rotating a vector using the quaternion's `rotate_vector`
  // produces the EXACT same result as converting the quaternion to a matrix and
  // multiplying the matrix by the vector.

  let axis = vec3(1.0, 1.0, 1.0).normalize();
  let q = Quat::from_axis_angle(axis, PI / 3.0); // 60 degree rotation
  let mat: Mat3f32 = q.to_matrix3();

  let v = vec3(10.0, -5.0, 42.0); // Arbitrary vector

  let rotated_by_quat = q.rotate_vector(v);
  let rotated_by_mat = mat * v;

  assert_vec3_approx_eq!(rotated_by_quat, rotated_by_mat);
}
