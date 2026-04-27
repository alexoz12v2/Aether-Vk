use crate::math::{
  FloatLike, MulAddIdentity,
  floating::{FloatBits, FloatOps},
  matrix::{Matrix3, MatrixVectorMul, SquareMatrix},
  vector::{Vector, Vector3},
};

/// Extension trait for solving Linear Complementarity Problems (LCP).
///
/// The Linear Complementarity Problem (LCP) arises frequently in rigid body dynamics,
/// specifically when resolving contact constraints with resting contacts and friction.
///
/// Mathematically, the LCP is defined as finding a vector `z` such that:
///   `z >= 0`
///   `w = M * z + q >= 0`
///   `z^T * w = 0`
///
/// Where:
/// - `M` is a square matrix (e.g., the effective mass/inertia matrix in contact space).
/// - `q` is a vector (e.g., the relative velocity minus any bias terms).
/// - `z` is the vector of unknowns we want to find (e.g., the collision impulses).
/// - `w` is the resulting vector (e.g., the post-resolution relative velocity).
///
/// The Projected Gauss-Seidel (PGS) algorithm is an iterative method used to approximate
/// the solution to the LCP. It operates similarly to the standard Gauss-Seidel method
/// for solving linear systems `Ax = b`, but after computing each component `z_i`, it
/// projects the result onto the feasible set (in this case, clamping `z_i` to be >= 0).
///
/// The component-wise update rule is:
/// `z_i^{(k+1)} = max(0, - (q_i + sum_{j < i} M_{ij} z_j^{(k+1)} + sum_{j > i} M_{ij} z_j^{(k)}) / M_{ii})`
pub trait LcpSolver: SquareMatrix + MatrixVectorMul {
  /// Solves the LCP `M * z + q >= 0`, `z >= 0`, `z^T * (M * z + q) = 0` using
  /// Projected Gauss-Seidel (PGS).
  ///
  /// # Arguments
  /// * `q` - The vector `q` in the LCP formulation.
  /// * `max_iters` - Maximum number of iterations for the PGS solver.
  /// * `epsilon` - Tolerance for the termination condition.
  ///
  /// # Returns
  /// The approximate solution vector `z`.
  fn solve_lcp_pgs(
    &self,
    q: &Self::Vector,
    max_iters: usize,
    epsilon: Self::Scalar,
  ) -> Self::Vector;
}

impl<M> LcpSolver for M
where
  M: SquareMatrix + MatrixVectorMul,
  M::Scalar: FloatLike + FloatOps + FloatBits,
{
  fn solve_lcp_pgs(
    &self,
    q: &Self::Vector,
    max_iters: usize,
    epsilon: Self::Scalar,
  ) -> Self::Vector {
    let mut z = M::Vector::zero();
    let n = M::Vector::DIM;
    let zero = M::Scalar::zero();

    for _ in 0..max_iters {
      let mut max_diff = zero;

      for i in 0..n {
        // Safety: i is < DIM
        let m_ii = unsafe { self.column_unchecked(i).component_unchecked(i) };

        // If diagonal is too small, skip to avoid division by zero
        if m_ii.abs() < M::Scalar::from_f32(1e-8) {
          continue;
        }

        let mut sum = unsafe { q.component_unchecked(i) };
        for j in 0..n {
          if i != j {
            let m_ij = unsafe { self.column_unchecked(j).component_unchecked(i) };
            let z_j = unsafe { z.component_unchecked(j) };
            sum += m_ij * z_j;
          }
        }

        let mut new_z_i = -sum / m_ii;
        new_z_i = new_z_i.max(zero); // projection step

        let old_z_i = unsafe { z.component_unchecked(i) };
        max_diff = max_diff.max((new_z_i - old_z_i).abs());

        z.set_component(i, new_z_i);
      }

      if max_diff < epsilon {
        break;
      }
    }

    z
  }
}

/// Represents the Full Pivoting LU Decomposition of a 3x3 matrix.
///
/// The Full Pivoting LU Decomposition decomposes a square matrix `A` into:
/// `P * A * Q = L * U`
/// Where:
/// - `P` and `Q` are permutation matrices (represented as index arrays).
/// - `L` is a lower triangular matrix with 1s on the diagonal.
/// - `U` is an upper triangular matrix.
///
/// This decomposition is highly numerically stable, even for nearly singular matrices,
/// because it selects the largest absolute value in the remaining submatrix as the pivot.
pub struct FullPivLu3x3<S> {
  lu: [[S; 3]; 3],
  p: [usize; 3],
  q: [usize; 3],
  singular: bool,
}

impl<S: FloatLike + FloatOps + FloatBits> FullPivLu3x3<S> {
  /// Solves `A * x = b` for `x`. Returns `None` if the matrix is singular.
  pub fn solve<V>(&self, b: &V) -> Option<V>
  where
    V: Vector3<Scalar = S>,
  {
    if self.singular {
      return None;
    }

    let mut x = [S::zero(); 3];
    // Apply permutation P: y = P * b
    for i in 0..3 {
      x[i] = unsafe { b.component_unchecked(self.p[i]) };
    }

    // Forward substitution: L * z = y  (L has 1s on diagonal)
    for i in 0..3 {
      for j in 0..i {
        x[i] = x[i] - self.lu[i][j] * x[j];
      }
    }

    // Backward substitution: U * w = z
    for i in (0..3).rev() {
      for j in i + 1..3 {
        x[i] = x[i] - self.lu[i][j] * x[j];
      }
      x[i] = x[i] / self.lu[i][i];
    }

    // Apply inverse permutation Q: solution = Q * w
    let mut res = [S::zero(); 3];
    for i in 0..3 {
      res[self.q[i]] = x[i];
    }

    Some(V::from_components(res[0], res[1], res[2]))
  }
}

/// Extension trait to compute Full Pivoting LU Decomposition for a `Matrix3`.
pub trait FullPivLuExt: Matrix3
where
  Self::Vector: Vector3,
{
  fn full_piv_lu(&self) -> FullPivLu3x3<Self::Scalar>;
}

impl<M> FullPivLuExt for M
where
  M: Matrix3,
  M::Vector: Vector3<Scalar = M::Scalar>,
  M::Scalar: FloatLike + FloatOps + FloatBits,
{
  fn full_piv_lu(&self) -> FullPivLu3x3<Self::Scalar> {
    let mut lu = [[M::Scalar::zero(); 3]; 3];
    for i in 0..3 {
      let col = unsafe { self.column_unchecked(i) };
      for j in 0..3 {
        lu[j][i] = unsafe { col.component_unchecked(j) };
      }
    }

    let mut p = [0, 1, 2];
    let mut q = [0, 1, 2];
    let mut singular = false;

    for k in 0..3 {
      // Find pivot
      let mut max_val = M::Scalar::zero();
      let mut pivot_row = k;
      let mut pivot_col = k;

      for i in k..3 {
        for j in k..3 {
          let val = lu[i][j].abs();
          if val > max_val {
            max_val = val;
            pivot_row = i;
            pivot_col = j;
          }
        }
      }

      if max_val < M::Scalar::from_f32(1e-8) {
        singular = true;
        break;
      }

      // Swap rows
      if pivot_row != k {
        p.swap(k, pivot_row);
        // Also swap the elements in the matrix
        let temp = lu[k];
        lu[k] = lu[pivot_row];
        lu[pivot_row] = temp;
      }

      // Swap cols
      if pivot_col != k {
        q.swap(k, pivot_col);
        for i in 0..3 {
          let temp = lu[i][k];
          lu[i][k] = lu[i][pivot_col];
          lu[i][pivot_col] = temp;
        }
      }

      // LU factor
      let inv_pivot = M::Scalar::one() / lu[k][k];
      for i in k + 1..3 {
        lu[i][k] = lu[i][k] * inv_pivot;
        for j in k + 1..3 {
          lu[i][j] = lu[i][j] - lu[i][k] * lu[k][j];
        }
      }
    }

    FullPivLu3x3 { lu, p, q, singular }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::math::matrix::mat3::Mat3f32;
  use crate::math::vector::vec3::vec3;

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
}
