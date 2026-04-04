use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};
use aethervk_oshal_rlib::math::matrix::{Matrix, Matrix3, MatrixVectorMul, SquareMatrix, mat3::Mat3f32};
use aethervk_oshal_rlib::math::{FloatLike, MulAddIdentity};
use alloc::vec::Vec;

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
  let mut q_total = Mat3f32::identity();
  for _ in 0..max_iter {
    let mut q = Mat3f32::zero();
    let mut r = Mat3f32::zero();

    let v0 = a.x;
    let mut v1 = a.y;
    let mut v2 = a.z;

    let r00 = v0.length();
    r.x = Vec3f32::from_components(r00, 0.0, 0.0);
    q.x = if r00 > 0.0 {
      v0 / r00
    } else {
      Vec3f32::from_components(1.0, 0.0, 0.0)
    };

    let r01 = q.x.dot(v1);
    r.y = Vec3f32::from_components(r01, 0.0, 0.0);
    v1 = v1 - q.x * r01;
    let r11 = v1.length();
    r.y.set_component(1, r11);
    q.y = if r11 > 0.0 {
      v1 / r11
    } else {
      Vec3f32::from_components(0.0, 1.0, 0.0)
    };

    let r02 = q.x.dot(v2);
    let r12 = q.y.dot(v2);
    r.z = Vec3f32::from_components(r02, r12, 0.0);
    v2 = v2 - q.x * r02 - q.y * r12;
    let r22 = v2.length();
    r.z.set_component(2, r22);
    q.z = if r22 > 0.0 {
      v2 / r22
    } else {
      Vec3f32::from_components(0.0, 0.0, 1.0)
    };

    a = r * q;
    q_total = q_total * q;

    let off_diag_max = [
      a.x.y().abs(),
      a.x.z().abs(),
      a.y.x().abs(),
      a.y.z().abs(),
      a.z.x().abs(),
      a.z.y().abs(),
    ]
    .into_iter()
    .fold(0.0f32, f32::max);

    if off_diag_max < tol {
      break;
    }
  }

  (Vec3f32::from_components(a.x.x(), a.y.y(), a.z.z()), q_total)
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
