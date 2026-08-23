
extern crate std;
use super::*;

// Helper macro to easily define a column-major matrix
// TODO: probably this is to be exported
macro_rules! mat {
  (
          $c0x:expr, $c1x:expr, $c2x:expr, $c3x:expr,
          $c0y:expr, $c1y:expr, $c2y:expr, $c3y:expr,
          $c0z:expr, $c1z:expr, $c2z:expr, $c3z:expr,
          $c0w:expr, $c1w:expr, $c2w:expr, $c3w:expr $(,)?
      ) => {
    Mat4x4f32 {
      x: Vec4f32::from_components($c0x, $c0y, $c0z, $c0w),
      y: Vec4f32::from_components($c1x, $c1y, $c2y, $c1w), // Fixed macro layout for columns
      z: Vec4f32::from_components($c2x, $c2y, $c2z, $c2w),
      w: Vec4f32::from_components($c3x, $c3y, $c3z, $c3w),
    }
  };
}

// Simpler helper to ensure exact column mapping
fn mat4_cols(c0: [f32; 4], c1: [f32; 4], c2: [f32; 4], c3: [f32; 4]) -> Mat4x4f32 {
  Mat4x4f32 {
    x: Vec4f32::from_components(c0[0], c0[1], c0[2], c0[3]),
    y: Vec4f32::from_components(c1[0], c1[1], c1[2], c1[3]),
    z: Vec4f32::from_components(c2[0], c2[1], c2[2], c2[3]),
    w: Vec4f32::from_components(c3[0], c3[1], c3[2], c3[3]),
  }
}

#[test]
fn test_identity() {
  let id = Mat4x4f32::identity();

  assert_eq!(id.x.x(), 1.0);
  assert_eq!(id.y.y(), 1.0);
  assert_eq!(id.z.z(), 1.0);
  assert_eq!(id.w.w(), 1.0);

  assert_eq!(id.x.y(), 0.0); // Check a few off-diagonals
  assert_eq!(id.z.w(), 0.0);
}

#[test]
fn test_into_arrays() {
  let m = mat4_cols(
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
  );

  // 1D Array Conversion (Column-Major Flat)
  let arr1d: [f32; 16] = m.into();
  assert_eq!(
    arr1d,
    [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
    ]
  );

  // 2D Array Conversion
  let arr2d: [[f32; 4]; 4] = m.into();
  assert_eq!(arr2d[0], [1.0, 2.0, 3.0, 4.0]);
  assert_eq!(arr2d[3], [13.0, 14.0, 15.0, 16.0]);
}

#[test]
fn test_indexing() {
  let mut m = Mat4x4f32::identity();

  // 1D (Column) Indexing
  assert_eq!(m[0].x(), 1.0);
  assert_eq!(m[1].y(), 1.0);

  // 2D (Row, Col) Indexing
  assert_eq!(m[(0, 0)], 1.0); // col 0, row 0
  assert_eq!(m[(2, 3)], 0.0); // col 3, row 2

  // Mutability
  m[(1, 2)] = 5.0;
  assert_eq!(m[(1, 2)], 5.0);
}

#[test]
fn test_addition_and_subtraction() {
  let m1 = mat4_cols(
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0, 3.0],
    [4.0, 4.0, 4.0, 4.0],
  );
  let m2 = mat4_cols(
    [10.0, 10.0, 10.0, 10.0],
    [20.0, 20.0, 20.0, 20.0],
    [30.0, 30.0, 30.0, 30.0],
    [40.0, 40.0, 40.0, 40.0],
  );

  let add = m1 + m2;
  assert_eq!(add[0].x(), 11.0);
  assert_eq!(add[3].w(), 44.0);

  let sub = m2 - m1;
  assert_eq!(sub[1].y(), 18.0);
  assert_eq!(sub[2].z(), 27.0);
}

#[test]
fn test_scalar_multiplication() {
  let m = mat4_cols(
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
  );

  let m_scaled1 = m * 2.0;
  let m_scaled2 = 2.0 * m;

  assert_eq!(m_scaled1[0].x(), 2.0);
  assert_eq!(m_scaled1[0].y(), 4.0);
  assert_eq!(m_scaled1, m_scaled2); // Should be commutative
}

#[test]
fn test_transpose() {
  let m = mat4_cols(
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
  );

  let t = m.transpose();

  // Row 0 of `m` becomes Col 0 of `t`
  assert_eq!(t[0].x(), 1.0);
  assert_eq!(t[0].y(), 5.0);
  assert_eq!(t[0].z(), 9.0);
  assert_eq!(t[0].w(), 13.0);

  // Double transpose should yield the original matrix
  assert_eq!(t.transpose(), m);
}

#[test]
fn test_matrix_vector_mul() {
  let m = mat4_cols(
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [10.0, 20.0, 30.0, 1.0], // Translation vector in w column
  );
  let v = Vec4f32::from_components(5.0, 5.0, 5.0, 1.0); // A point

  let result = m.mul_vector(v);

  assert_eq!(result.x(), 15.0); // 5 + 10
  assert_eq!(result.y(), 25.0); // 5 + 20
  assert_eq!(result.z(), 35.0); // 5 + 30
  assert_eq!(result.w(), 1.0);
}

#[test]
fn test_matrix_matrix_mul() {
  let m = mat4_cols(
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
  );
  let id = Mat4x4f32::identity();

  // M * I = M
  assert_eq!(m * id, m);
  // I * M = M
  assert_eq!(id * m, m);

  // Test with a translation matrix
  let trans = mat4_cols(
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [10.0, 10.0, 10.0, 1.0],
  );
  let scale = mat4_cols(
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  );

  // Trans * Scale
  let combined = trans * scale;
  assert_eq!(combined[0].x(), 2.0); // Scale is applied
  assert_eq!(combined[3].x(), 10.0); // Translation remains in col 3
}

#[test]
fn test_determinant() {
  let id = Mat4x4f32::identity();
  assert_eq!(id.determinant(), 1.0);

  let m = mat4_cols(
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0],
    [0.0, 0.0, 4.0, 0.0],
    [0.0, 0.0, 0.0, 5.0],
  );
  // Determinant of diagonal matrix is product of diagonals
  assert_eq!(m.determinant(), 120.0);
}

#[test]
fn test_inverse() {
  // Simple scale + translate matrix, easy to invert
  let m = mat4_cols(
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 0.0],
    [10.0, 20.0, 30.0, 1.0],
  );

  let inv_opt = m.inverse();
  assert!(inv_opt.is_some(), "Matrix should be invertible");

  let inv = inv_opt.unwrap();

  // M * M^-1 should equal Identity
  let id_test = m * inv;
  let id = Mat4x4f32::identity();

  // Floating point comparisons after a matrix inverse might have tiny inaccuracies.
  // We use a small epsilon to check if it practically resulted in the Identity matrix.
  let epsilon = 1e-6;
  for col in 0..4 {
    for row in 0..4 {
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
  let singular = mat4_cols(
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
  );
  assert!(singular.inverse().is_none());
}
