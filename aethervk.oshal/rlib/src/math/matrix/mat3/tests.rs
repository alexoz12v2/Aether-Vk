
extern crate std;
use super::*;
use crate::math::vector::Vector3;

// Helper macro to easily define a 3x3 column-major matrix
// TODO: expose probably
macro_rules! mat3 {
  (
          $c0x:expr, $c1x:expr, $c2x:expr,
          $c0y:expr, $c1y:expr, $c2y:expr,
          $c0z:expr, $c1z:expr, $c2z:expr $(,)?
      ) => {
    Mat3f32 {
      x: Vec3f32::from_components($c0x, $c0y, $c0z),
      y: Vec3f32::from_components($c1x, $c1y, $c2y),
      z: Vec3f32::from_components($c2x, $c2y, $c2z),
    }
  };
}

// Explicit column-based helper
fn mat3_cols(c0: [f32; 3], c1: [f32; 3], c2: [f32; 3]) -> Mat3f32 {
  Mat3f32 {
    x: Vec3f32::from_components(c0[0], c0[1], c0[2]),
    y: Vec3f32::from_components(c1[0], c1[1], c1[2]),
    z: Vec3f32::from_components(c2[0], c2[1], c2[2]),
  }
}

#[test]
fn test_identity_and_zero() {
  let id = Mat3f32::identity();
  assert_eq!(id.x.x(), 1.0);
  assert_eq!(id.y.y(), 1.0);
  assert_eq!(id.z.z(), 1.0);
  assert_eq!(id.x.y(), 0.0); // Check an off-diagonal

  let zero = Mat3f32::zero();
  assert_eq!(zero.x.x(), 0.0);
  assert_eq!(zero.z.z(), 0.0);
}

#[test]
fn test_indexing() {
  let mut m = Mat3f32::identity();

  // 1D (Column) Indexing
  assert_eq!(m[0].x(), 1.0);
  assert_eq!(m[1].y(), 1.0);

  // 2D (Row, Col) Indexing
  assert_eq!(m[(0, 0)], 1.0); // col 0, row 0
  assert_eq!(m[(2, 2)], 1.0); // col 2, row 2
  assert_eq!(m[(0, 1)], 0.0); // col 1, row 0

  // Mutability
  m[(1, 2)] = 5.0; // Row 1, Col 2
  assert_eq!(m[(1, 2)], 5.0);
  assert_eq!(m[2].y(), 5.0); // Verify it updated the correct underlying vector component
}

#[test]
fn test_addition_and_subtraction() {
  let m1 = mat3_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
  let m2 = mat3_cols([10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]);

  let add = m1 + m2;
  assert_eq!(add[0].x(), 11.0);
  assert_eq!(add[1].y(), 25.0);
  assert_eq!(add[2].z(), 39.0);

  let sub = m2 - m1;
  assert_eq!(sub[0].x(), 9.0);
  assert_eq!(sub[1].y(), 15.0);
  assert_eq!(sub[2].z(), 21.0);
}

#[test]
fn test_scalar_multiplication() {
  let m = mat3_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);

  let m_scaled1 = m * 2.0;
  let m_scaled2 = 2.0 * m;

  assert_eq!(m_scaled1[0].x(), 2.0);
  assert_eq!(m_scaled1[1].y(), 10.0);
  assert_eq!(m_scaled1[2].z(), 18.0);
  assert_eq!(m_scaled1, m_scaled2); // Should be commutative
}

#[test]
fn test_transpose() {
  let m = mat3_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);

  let t = m.transpose();

  // Row 0 of `m` [1.0, 4.0, 7.0] becomes Col 0 of `t`
  assert_eq!(t[0].x(), 1.0);
  assert_eq!(t[0].y(), 4.0);
  assert_eq!(t[0].z(), 7.0);

  // Double transpose should yield the original matrix
  assert_eq!(t.transpose(), m);
}

#[test]
fn test_matrix_vector_mul() {
  // Rotation by 90 degrees around Z axis
  let rot_z_90 = mat3_cols(
    [0.0, 1.0, 0.0],  // New X-axis
    [-1.0, 0.0, 0.0], // New Y-axis
    [0.0, 0.0, 1.0],  // New Z-axis
  );
  let v = Vec3f32::from_components(1.0, 0.0, 0.0); // Pointing down X

  let result = rot_z_90.mul_vector(v);

  // Point should now be at Y=1
  assert_eq!(result.x(), 0.0);
  assert_eq!(result.y(), 1.0);
  assert_eq!(result.z(), 0.0);
}

#[test]
fn test_matrix_matrix_mul() {
  let m1 = mat3_cols([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
  let id = Mat3f32::identity();

  // M * I = M and I * M = M
  assert_eq!(m1 * id, m1);
  assert_eq!(id * m1, m1);

  let m2 = mat3_cols([2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]);

  let combined = m1 * m2;
  // Since m2 is a uniform scale matrix, m1 * m2 should just be m1 scaled by 2
  assert_eq!(combined, m1 * 2.0);
}

#[test]
fn test_determinant() {
  let id = Mat3f32::identity();
  assert_eq!(id.determinant(), 1.0);

  let m = mat3_cols([2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]);
  // Determinant of diagonal matrix is product of diagonals
  assert_eq!(m.determinant(), 24.0);

  // Sarrus rule test on arbitrary matrix
  let m2 = mat3_cols(
    [3.0, 2.0, 1.0], // Col 0
    [1.0, 4.0, 2.0], // Col 1
    [5.0, 1.0, 3.0], // Col 2
  );
  // det = 3(4*3 - 2*1) - 1(2*3 - 1*1) + 5(2*2 - 4*1)
  // det = 3(10) - 1(5) + 5(0) = 30 - 5 = 25
  assert_eq!(m2.determinant(), 25.0);
}

#[test]
fn test_inverse() {
  // A simple invertible matrix (Scale + slight skew)
  let m = mat3_cols([2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 2.0]);

  let inv_opt = m.inverse();
  assert!(inv_opt.is_some(), "Matrix should be invertible");

  let inv = inv_opt.unwrap();

  // \mathbf{M} \times \mathbf{M}^{-1} = \mathbf{I}
  let id_test = m * inv;
  let id = Mat3f32::identity();

  // Floating point comparisons after inverses require a tiny tolerance
  let epsilon = 1e-6;
  for col in 0..3 {
    for row in 0..3 {
      let diff = (id_test[(row, col)] - id[(row, col)]).abs();
      assert!(
        diff < epsilon,
        "Inverse failed at ({},{}). Expected {}, got {}",
        row,
        col,
        id[(row, col)],
        id_test[(row, col)]
      );
    }
  }

  // Test singular matrix (Determinant = 0)
  let singular = mat3_cols([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
  assert!(singular.inverse().is_none());
}
