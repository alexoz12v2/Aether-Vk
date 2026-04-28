use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4, vec3::Vec3f32};
use aethervk_oshal_rlib::math::matrix::{Matrix, Matrix3, MatrixVectorMul, SquareMatrix, mat3::Mat3f32};
use aethervk_oshal_rlib::math::{FloatLike, MulAddIdentity};
use alloc::vec::Vec;

pub mod collision;

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

pub fn qr_diagonalization(mut a: Mat3f32, tol: f32, max_iter: usize) -> (Vec3f32, Mat3f32) {
  let mut v = Mat3f32::identity();

  for _ in 0..max_iter {
    // Find the largest off-diagonal element
    let mut max_val = 0.0f32;
    let mut p = 0;
    let mut q = 1;

    let off_diag = [
      (0, 1, a.y.x().abs()),
      (0, 2, a.z.x().abs()),
      (1, 2, a.z.y().abs()),
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
    let a_pp = match p {
      0 => a.x.x(),
      1 => a.y.y(),
      _ => a.z.z(),
    };
    let a_qq = match q {
      1 => a.y.y(),
      _ => a.z.z(),
    };
    let a_pq = match (p, q) {
      (0, 1) => a.y.x(),
      (0, 2) => a.z.x(),
      _ => a.z.y(),
    };

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
    let new_app = a_pp - t_apq;
    let new_aqq = a_qq + t_apq;

    match p {
      0 => a_new.x.set_component(0, new_app),
      1 => a_new.y.set_component(1, new_app),
      _ => a_new.z.set_component(2, new_app),
    }
    match q {
      1 => a_new.y.set_component(1, new_aqq),
      _ => a_new.z.set_component(2, new_aqq),
    }

    // Zero out the target off-diagonal elements
    match (p, q) {
      (0, 1) => { a_new.y.set_component(0, 0.0); a_new.x.set_component(1, 0.0); },
      (0, 2) => { a_new.z.set_component(0, 0.0); a_new.x.set_component(2, 0.0); },
      _      => { a_new.z.set_component(1, 0.0); a_new.y.set_component(2, 0.0); },
    }

    // Update the rest of the matrix
    for r in 0..3 {
      if r != p && r != q {
        let a_rp = match (r, p) {
          (0, 0) => a.x.x(), (0, 1) => a.y.x(), (0, 2) => a.z.x(),
          (1, 0) => a.x.y(), (1, 1) => a.y.y(), (1, 2) => a.z.y(),
          _ => a.x.z(),
        };
        if p == 0 && r == 1 { /* a_rp = a.x.y() */ }
        let a_rp = match r {
          0 => match p { 0 => a.x.x(), 1 => a.y.x(), _ => a.z.x() },
          1 => match p { 0 => a.x.y(), 1 => a.y.y(), _ => a.z.y() },
          _ => match p { 0 => a.x.z(), 1 => a.y.z(), _ => a.z.z() },
        };
        let a_rq = match r {
          0 => match q { 1 => a.y.x(), _ => a.z.x() },
          1 => match q { 1 => a.y.y(), _ => a.z.y() },
          _ => match q { 1 => a.y.z(), _ => a.z.z() },
        };

        let new_arp = c * a_rp - s * a_rq;
        let new_arq = s * a_rp + c * a_rq;

        // Set A_rp and A_pr
        match (r, p) {
          (0, 1) => { a_new.y.set_component(0, new_arp); a_new.x.set_component(1, new_arp); },
          (0, 2) => { a_new.z.set_component(0, new_arp); a_new.x.set_component(2, new_arp); },
          (1, 0) => { a_new.x.set_component(1, new_arp); a_new.y.set_component(0, new_arp); },
          (1, 2) => { a_new.z.set_component(1, new_arp); a_new.y.set_component(2, new_arp); },
          (2, 0) => { a_new.x.set_component(2, new_arp); a_new.z.set_component(0, new_arp); },
          (2, 1) => { a_new.y.set_component(2, new_arp); a_new.z.set_component(1, new_arp); },
          _ => {}
        }
        
        // Set A_rq and A_qr
        match (r, q) {
          (0, 1) => { a_new.y.set_component(0, new_arq); a_new.x.set_component(1, new_arq); },
          (0, 2) => { a_new.z.set_component(0, new_arq); a_new.x.set_component(2, new_arq); },
          (1, 0) => { a_new.x.set_component(1, new_arq); a_new.y.set_component(0, new_arq); },
          (1, 2) => { a_new.z.set_component(1, new_arq); a_new.y.set_component(2, new_arq); },
          (2, 0) => { a_new.x.set_component(2, new_arq); a_new.z.set_component(0, new_arq); },
          (2, 1) => { a_new.y.set_component(2, new_arq); a_new.z.set_component(1, new_arq); },
          _ => {}
        }
      }
    }
    a = a_new;

    // Apply rotation to eigenvectors V
    let mut v_new = v.clone();
    for r in 0..3 {
      let v_rp = match r {
        0 => match p { 0 => v.x.x(), 1 => v.y.x(), _ => v.z.x() },
        1 => match p { 0 => v.x.y(), 1 => v.y.y(), _ => v.z.y() },
        _ => match p { 0 => v.x.z(), 1 => v.y.z(), _ => v.z.z() },
      };
      let v_rq = match r {
        0 => match q { 1 => v.y.x(), _ => v.z.x() },
        1 => match q { 1 => v.y.y(), _ => v.z.y() },
        _ => match q { 1 => v.y.z(), _ => v.z.z() },
      };

      let new_vrp = c * v_rp - s * v_rq;
      let new_vrq = s * v_rp + c * v_rq;

      match (r, p) {
        (0, 0) => v_new.x.set_component(0, new_vrp), (0, 1) => v_new.y.set_component(0, new_vrp), (0, 2) => v_new.z.set_component(0, new_vrp),
        (1, 0) => v_new.x.set_component(1, new_vrp), (1, 1) => v_new.y.set_component(1, new_vrp), (1, 2) => v_new.z.set_component(1, new_vrp),
        (2, 0) => v_new.x.set_component(2, new_vrp), (2, 1) => v_new.y.set_component(2, new_vrp), (2, 2) => v_new.z.set_component(2, new_vrp),
        _ => {}
      }
      match (r, q) {
        (0, 1) => v_new.y.set_component(0, new_vrq), (0, 2) => v_new.z.set_component(0, new_vrq),
        (1, 1) => v_new.y.set_component(1, new_vrq), (1, 2) => v_new.z.set_component(1, new_vrq),
        (2, 1) => v_new.y.set_component(2, new_vrq), (2, 2) => v_new.z.set_component(2, new_vrq),
        _ => {}
      }
    }
    v = v_new;
  }

  // Ensure right-handed coordinate system
  if v.determinant() < 0.0 {
    v.x = -v.x;
  }

  (Vec3f32::from_components(a.x.x(), a.y.y(), a.z.z()), v)
}

pub fn hat(v: Vec3f32) -> Mat3f32 {
  Mat3f32 {
    x: Vec3f32::from_components(0.0, v.z(), -v.y()),
    y: Vec3f32::from_components(-v.z(), 0.0, v.x()),
    z: Vec3f32::from_components(v.y(), -v.x(), 0.0),
  }
}

pub fn vee(s: Mat3f32) -> Vec3f32 {
  Vec3f32::from_components(s.y.z(), s.z.x(), s.x.y())
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
      assert!((col_a.x() - col_b.x()).abs() < tol, "Mismatch at col {}, x: {} vs {}", i, col_a.x(), col_b.x());
      assert!((col_a.y() - col_b.y()).abs() < tol, "Mismatch at col {}, y: {} vs {}", i, col_a.y(), col_b.y());
      assert!((col_a.z() - col_b.z()).abs() < tol, "Mismatch at col {}, z: {} vs {}", i, col_a.z(), col_b.z());
    }
  }

  #[test]
  fn test_jacobi_diagonalization_identity() {
    let a = Mat3f32::identity();
    let (evals, evecs) = qr_diagonalization(a, 1e-5, 50);
    
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
    let (evals, evecs) = qr_diagonalization(a, 1e-5, 50);
    
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
    let (evals, evecs) = qr_diagonalization(a, 1e-5, 50);
    
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
    assert!(evecs.is_pure_rotation_permissive(), "Eigenvector matrix must be a pure rotation");
  }
}
