use aethervk_oshal_rlib::math::matrix::{
  Matrix, Matrix3, MatrixVectorMul, SquareMatrix, mat3::Mat3f32,
};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4, vec3::Vec3f32};
use aethervk_oshal_rlib::math::{FloatLike, MulAddIdentity};
use alloc::vec::Vec;

pub mod collision;
pub mod distribution;
pub mod particles_edu;
pub mod physics;

/// Converts a world-space point to screen space.
pub fn from_world_space_to_screen_space(
  world_pos: Vec3f32,
  view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  window_extent: (f32, f32),
) -> Option<(f32, f32)> {
  let world_vec4 = world_pos.to_point();
  let mut clip = view_proj.mul_vector(world_vec4);
  if clip.w() > 0.0 {
    clip = clip / clip.w();
    if clip.z() >= 0.0 && clip.z() <= 1.0 {
      let ndc_x = clip.x();
      let ndc_y = clip.y();

      let screen_x = (ndc_x * 0.5 + 0.5) * window_extent.0;
      let screen_y = (ndc_y * 0.5 + 0.5) * window_extent.1;

      return Some((screen_x, screen_y));
    }
  }

  None
}

pub fn compute_com_and_tensor(verts: &[Vec3f32], m: f32) -> (Vec3f32, Mat3f32) {
  let mut com = Vec3f32::zero();
  for v in verts {
    com += *v;
  }
  if !verts.is_empty() {
    com /= verts.len() as f32;
  }

  let mut i00 = 0.0;
  let mut i11 = 0.0;
  let mut i22 = 0.0;
  let mut i01 = 0.0;
  let mut i10 = 0.0;
  let mut i02 = 0.0;
  let mut i20 = 0.0;
  let mut i12 = 0.0;
  let mut i21 = 0.0;

  for v in verts {
    let v_c = *v - com;
    let x = v_c.x();
    let y = v_c.y();
    let z = v_c.z();

    i00 += m * (y * y + z * z);
    i11 += m * (x * x + z * z);
    i22 += m * (x * x + y * y);

    let xy = m * x * y;
    i01 -= xy;
    i10 -= xy;

    let xz = m * x * z;
    i02 -= xz;
    i20 -= xz;

    let yz = m * y * z;
    i12 -= yz;
    i21 -= yz;
  }

  let i = Mat3f32 {
    x: Vec3f32::from_components(i00, i10, i20),
    y: Vec3f32::from_components(i01, i11, i21),
    z: Vec3f32::from_components(i02, i12, i22),
  };

  (com, i)
}

pub trait Mat3Ext {
  fn off_diagonal_sum(&self) -> f32;
}

impl Mat3Ext for Mat3f32 {
  fn off_diagonal_sum(&self) -> f32 {
    // Sums the absolute values of all non-diagonal elements.
    // Assumes your matrix properties (x, y, z) represent columns
    // and have accessors like x() for rows.
    self.y.x().abs()
      + self.z.x().abs()
      + self.x.y().abs()
      + self.z.y().abs()
      + self.x.z().abs()
      + self.y.z().abs()
  }
}

/// Standalone QR Decomposition function using Modified Gram-Schmidt
pub fn qr_decomposition(mat: Mat3f32) -> (Mat3f32, Mat3f32) {
  let a1 = mat.x;
  let a2 = mat.y;
  let a3 = mat.z;

  // 1st Column
  let r11 = a1.length();
  let e1 = a1 / r11; // Normalize

  // 2nd Column
  let r12 = a2.dot(e1);
  let u2 = a2 - (e1 * r12);
  let r22 = u2.length();
  let e2 = u2 / r22;

  // 3rd Column
  let r13 = a3.dot(e1);
  let r23 = a3.dot(e2);
  let u3 = a3 - (e1 * r13) - (e2 * r23);
  let r33 = u3.length();
  let e3 = u3 / r33;

  // Construct Q (Orthogonal matrix)
  let mut q = Mat3f32::identity();
  q.x = e1;
  q.y = e2;
  q.z = e3;

  // Construct R (Upper triangular matrix)
  // Note: Assuming Vec3f32::new(x, y, z)
  let mut r = Mat3f32::identity();
  r.x = Vec3f32::from_components(r11, 0.0, 0.0);
  r.y = Vec3f32::from_components(r12, r22, 0.0);
  r.z = Vec3f32::from_components(r13, r23, r33);

  (q, r)
}

pub fn qr_diagonalization(mat: Mat3f32, tol: f32, max_iter: usize) -> (Mat3f32, Mat3f32) {
  let mut ak = mat;
  let mut qk = Mat3f32::identity();
  for _ in 0..max_iter {
    let (q, r) = qr_decomposition(ak);
    ak = r * q;
    qk = qk * q;

    let off_diag = ak.off_diagonal_sum();
    if off_diag < tol {
      break;
    }
  }
  (ak, qk)
}

pub fn jacobi_diagonalization(mut a: Mat3f32, tol: f32, max_iter: usize) -> (Vec3f32, Mat3f32) {
  let mut v = Mat3f32::identity();

  // Helper lambda to get a matrix element at (row, col)
  let get = |mat: &Mat3f32, r: usize, c: usize| -> f32 {
    let col = match c {
      0 => &mat.x,
      1 => &mat.y,
      _ => &mat.z,
    };
    match r {
      0 => col.x(),
      1 => col.y(),
      _ => col.z(),
    }
  };

  // Helper lambda to set a matrix element at (row, col)
  let set = |mat: &mut Mat3f32, r: usize, c: usize, val: f32| match c {
    0 => mat.x.set_component(r, val),
    1 => mat.y.set_component(r, val),
    _ => mat.z.set_component(r, val),
  };

  for _ in 0..max_iter {
    // Find the largest off-diagonal element
    let mut max_val = 0.0f32;
    let mut p = 0;
    let mut q = 1;

    let off_diag = [
      (0, 1, get(&a, 0, 1).abs()),
      (0, 2, get(&a, 0, 2).abs()),
      (1, 2, get(&a, 1, 2).abs()),
    ];

    for &(row, col, val) in &off_diag {
      if val > max_val {
        max_val = val;
        p = row;
        q = col;
      }
    }

    if max_val < tol {
      break;
    }

    // Compute Jacobi rotation
    let a_pp = get(&a, p, p);
    let a_qq = get(&a, q, q);
    let a_pq = get(&a, p, q);

    let theta = (a_qq - a_pp) / (2.0 * a_pq);
    let mut t = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
    if theta < 0.0 {
      t = -t;
    }

    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;

    // Apply rotation to A
    let mut a_new = a.clone();

    // Update diagonal elements
    let t_apq = t * a_pq;
    set(&mut a_new, p, p, a_pp - t_apq);
    set(&mut a_new, q, q, a_qq + t_apq);

    // Zero out the target off-diagonal elements (symmetric)
    set(&mut a_new, p, q, 0.0);
    set(&mut a_new, q, p, 0.0);

    // Update the rest of the matrix
    for r in 0..3 {
      if r != p && r != q {
        let a_rp = get(&a, r, p);
        let a_rq = get(&a, r, q);

        let new_arp = c * a_rp - s * a_rq;
        let new_arq = s * a_rp + c * a_rq;

        // Set A_rp and A_pr symmetrically
        set(&mut a_new, r, p, new_arp);
        set(&mut a_new, p, r, new_arp);

        // Set A_rq and A_qr symmetrically
        set(&mut a_new, r, q, new_arq);
        set(&mut a_new, q, r, new_arq);
      }
    }
    a = a_new;

    // Apply rotation to eigenvectors V
    let mut v_new = v.clone();
    for r in 0..3 {
      let v_rp = get(&v, r, p);
      let v_rq = get(&v, r, q);

      let new_vrp = c * v_rp - s * v_rq;
      let new_vrq = s * v_rp + c * v_rq;

      set(&mut v_new, r, p, new_vrp);
      set(&mut v_new, r, q, new_vrq);
    }
    v = v_new;
  }

  // Ensure right-handed coordinate system
  if v.determinant() < 0.0 {
    v.x = -v.x;
  }

  (Vec3f32::from_components(a.x.x(), a.y.y(), a.z.z()), v)
}

/// From lie algebra (pseudovector) representation of the angular velocity to its matrix hat representation (lie group, SO(3))
pub fn hat(v: Vec3f32) -> Mat3f32 {
  Mat3f32 {
    x: Vec3f32::from_components(0.0, v.z(), -v.y()),
    y: Vec3f32::from_components(-v.z(), 0.0, v.x()),
    z: Vec3f32::from_components(v.y(), -v.x(), 0.0),
  }
}

/// From matrix hat representation (lie group, SO(3)) of the angular velocity to its lie algebra (pseudovector) representation
/// - while mathematically dR/dt `matmul` R^T produces a skew symmetric matrix, we guard
/// against float imprecision by averaging opposite elements
pub fn vee(s: Mat3f32) -> Vec3f32 {
  // Vec3f32::from_components(s.y.z(), s.z.x(), s.x.y())
  Vec3f32::from_components(
    (s.y.z() - s.z.y()) * 0.5,
    (s.z.x() - s.x.z()) * 0.5,
    (s.x.y() - s.y.x()) * 0.5,
  )
}

pub fn expm_hat(w: Vec3f32) -> Mat3f32 {
  let theta = w.length();
  let id = Mat3f32::identity();
  if theta < 1e-8 {
    return id;
  }
  let w_hat = hat(w);
  let w_hat2 = w_hat * w_hat;

  let a = theta.sin() / theta;
  let b = (1.0 - theta.cos()) / (theta * theta);

  id + w_hat * a + w_hat2 * b
}

pub fn solve_12x12(a: &mut [[f32; 12]; 12], b: &mut [f32; 12]) -> bool {
  let n = 12;
  for i in 0..n {
    let mut max_el = a[i][i].abs();
    let mut max_row = i;
    for k in i + 1..n {
      if a[k][i].abs() > max_el {
        max_el = a[k][i].abs();
        max_row = k;
      }
    }
    if max_el < 1e-8 {
      return false;
    }
    if max_row != i {
      a.swap(i, max_row);
      b.swap(i, max_row);
    }

    for k in i + 1..n {
      let c = -a[k][i] / a[i][i];
      for j in i..n {
        if i == j {
          a[k][j] = 0.0;
        } else {
          a[k][j] += c * a[i][j];
        }
      }
      b[k] += c * b[i];
    }
  }

  for i in (0..n).rev() {
    let mut sum = 0.0;
    for j in i + 1..n {
      sum += a[i][j] * b[j];
    }
    b[i] = (b[i] - sum) / a[i][i];
  }
  true
}

/// Finds real roots of a cubic polynomial a*x^3 + b*x^2 + c*x + d = 0 iteratively.
pub fn solve_cubic<T: FloatLike>(a: T, b: T, c: T, d: T) -> Vec<T> {
  let mut roots = Vec::new();
  let zero = T::zero();
  let one = T::one();
  let two = one + one;
  let three = two + one;
  let four = two + two;

  let abs_t = |v: T| if v < zero { -v } else { v };

  if abs_t(a) < T::from_f32(1e-6) {
    if abs_t(b) < T::from_f32(1e-6) {
      if abs_t(c) > T::from_f32(1e-6) {
        roots.push(-d / c);
      }
      return roots;
    }
    let delta = c * c - four * b * d;
    if delta > zero {
      let sqrt_delta = delta.sqrt();
      roots.push((-c - sqrt_delta) / (two * b));
      roots.push((-c + sqrt_delta) / (two * b));
    } else if abs_t(delta) <= T::from_f32(1e-6) {
      roots.push(-c / (two * b));
    }
    return roots;
  }

  let a_inv = one / a;
  let a_a = b * a_inv;
  let a_b = c * a_inv;
  let a_c = d * a_inv;

  let f = |x: T| ((x + a_a) * x + a_b) * x + a_c;
  let df = |x: T| (three * x + two * a_a) * x + a_b;

  let max_coeff = abs_t(a_a).max(abs_t(a_b)).max(abs_t(a_c));
  let cap_m = one + max_coeff;

  let d_sub = a_a * a_a - three * a_b;

  let tol = T::from_f32(1e-6);
  let max_iter = 50;

  let find_root = |mut xl: T, mut xh: T| -> T {
    let mut fl = f(xl);
    let mut fh = f(xh);

    if abs_t(fl) < tol {
      return xl;
    }
    if abs_t(fh) < tol {
      return xh;
    }

    if fl > fh {
      let tmp_x = xl;
      xl = xh;
      xh = tmp_x;
      let tmp_f = fl;
      fl = fh;
      fh = tmp_f;
    }

    let mut x_guess = (xl + xh) / two;
    let mut step_bisection = true;

    for _ in 0..max_iter {
      let fx = f(x_guess);
      let dfx = df(x_guess);

      if !step_bisection && abs_t(dfx) > T::from_f32(1e-12) {
        let dx = fx / dfx;
        let x_next = x_guess - dx;
        if x_next > xl.min(xh) && x_next < xl.max(xh) {
          x_guess = x_next;
        } else {
          x_guess = (xl + xh) / two;
          step_bisection = true;
        }
      } else {
        x_guess = (xl + xh) / two;
        step_bisection = false;
      }

      let fx_next = f(x_guess);
      if abs_t(fx_next) < tol || abs_t(xh - xl) < tol {
        return x_guess;
      }

      if fx_next < zero {
        xl = x_guess;
        fl = fx_next;
      } else {
        xh = x_guess;
        fh = fx_next;
      }
    }
    x_guess
  };

  if d_sub <= zero {
    roots.push(find_root(-cap_m, cap_m));
  } else {
    let sqrt_d_sub = d_sub.sqrt();
    let x1 = (-a_a - sqrt_d_sub) / three;
    let x2 = (-a_a + sqrt_d_sub) / three;

    let y1 = f(x1);
    let y2 = f(x2);

    if y1 < zero {
      roots.push(find_root(x2, cap_m));
    } else if y2 > zero {
      roots.push(find_root(-cap_m, x1));
    } else {
      roots.push(find_root(-cap_m, x1));
      roots.push(find_root(x1, x2));
      roots.push(find_root(x2, cap_m));
    }
  }

  let mut unique_roots = Vec::new();
  for r in roots {
    let mut is_dup = false;
    for &u in &unique_roots {
      if abs_t(r - u) < T::from_f32(1e-4) {
        is_dup = true;
        break;
      }
    }
    if !is_dup {
      unique_roots.push(r);
    }
  }

  unique_roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
  unique_roots
}

/// Specialized f32 version for solving a cubic polynomial.
#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::matrix::{Matrix, Matrix3};

  fn assert_mat3_eq(a: Mat3f32, b: Mat3f32, tol: f32) {
    for i in 0..3 {
      let col_a = unsafe { a.column_unchecked(i) };
      let col_b = unsafe { b.column_unchecked(i) };
      assert!(
        (col_a.x() - col_b.x()).abs() < tol,
        "Mismatch at col {}, x: {} vs {}",
        i,
        col_a.x(),
        col_b.x()
      );
      assert!(
        (col_a.y() - col_b.y()).abs() < tol,
        "Mismatch at col {}, y: {} vs {}",
        i,
        col_a.y(),
        col_b.y()
      );
      assert!(
        (col_a.z() - col_b.z()).abs() < tol,
        "Mismatch at col {}, z: {} vs {}",
        i,
        col_a.z(),
        col_b.z()
      );
    }
  }

  #[test]
  fn test_jacobi_diagonalization_identity() {
    let a = Mat3f32::identity();
    let (evals, evecs) = jacobi_diagonalization(a, 1e-5, 50);

    assert!((evals.x() - 1.0).abs() < 1e-4);
    assert!((evals.y() - 1.0).abs() < 1e-4);
    assert!((evals.z() - 1.0).abs() < 1e-4);
    assert_mat3_eq(evecs, Mat3f32::identity(), 1e-4);
    assert!(evecs.is_pure_rotation_permissive());
  }

  #[test]
  fn test_jacobi_diagonalization_diagonal() {
    let a = Mat3f32 {
      x: Vec3f32::from_components(2.0, 0.0, 0.0),
      y: Vec3f32::from_components(0.0, 3.0, 0.0),
      z: Vec3f32::from_components(0.0, 0.0, 5.0),
    };
    let (evals, evecs) = jacobi_diagonalization(a, 1e-5, 50);

    assert!((evals.x() - 2.0).abs() < 1e-4);
    assert!((evals.y() - 3.0).abs() < 1e-4);
    assert!((evals.z() - 5.0).abs() < 1e-4);
    assert_mat3_eq(evecs, Mat3f32::identity(), 1e-4);
    assert!(evecs.is_pure_rotation_permissive());
  }

  #[test]
  fn test_jacobi_diagonalization_symmetric() {
    let a = Mat3f32 {
      x: Vec3f32::from_components(4.0, 1.0, 2.0),
      y: Vec3f32::from_components(1.0, 3.0, 0.0),
      z: Vec3f32::from_components(2.0, 0.0, 5.0),
    };
    let (evals, evecs) = jacobi_diagonalization(a, 1e-5, 50);

    // Check that A = V * D * V^T
    let d = Mat3f32 {
      x: Vec3f32::from_components(evals.x(), 0.0, 0.0),
      y: Vec3f32::from_components(0.0, evals.y(), 0.0),
      z: Vec3f32::from_components(0.0, 0.0, evals.z()),
    };
    let evecs_t = Mat3f32 {
      x: Vec3f32::from_components(evecs.x.x(), evecs.y.x(), evecs.z.x()),
      y: Vec3f32::from_components(evecs.x.y(), evecs.y.y(), evecs.z.y()),
      z: Vec3f32::from_components(evecs.x.z(), evecs.y.z(), evecs.z.z()),
    };

    let reconstructed = evecs * (d * evecs_t);
    assert_mat3_eq(reconstructed, a, 1e-3);
    assert!(
      evecs.is_pure_rotation_permissive(),
      "Eigenvector matrix must be a pure rotation"
    );
  }
}
