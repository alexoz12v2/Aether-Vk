//! physics module.

use crate::math::{expm_hat, hat, vee};
use aethervk_oshal_rlib::math::{
  FloatLike, MulAddIdentity,
  matrix::{Matrix, Matrix3, MatrixVectorMul, SquareMatrix, mat3::Mat3f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
};
use alloc::vec::Vec;

// --- Data Structures ---

/// Represents a simple point-mass particle in the simulation.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
  pub position: Vec3f32,
  pub velocity: Vec3f32,
  pub mass: f32,
  /// Force accumulated at the current temporal evaluation point.
  /// In Velocity Verlet, this single buffer seamlessly transitions from F_n to F_{n+1}.
  pub accumulated_force: Vec3f32,
}

/// Represents a massive rigid body acting under translation and rotation.
#[derive(Debug, Clone, Copy)]
pub struct RigidBody {
  pub position: Vec3f32,
  pub rotation: Mat3f32,
  pub linear_velocity: Vec3f32,
  pub angular_velocity: Vec3f32,
  pub mass: f32,
  pub inertia_tensor: Mat3f32,
}

/// Force Emitters (e.g. Gravity from a celestial body)
#[derive(Debug, Clone, Copy)]
pub enum ForceEmitter {
  Gravity {
    position: Vec3f32,
    /// Standard gravitational parameter (G * M)
    mu: f32,
  },
  Planar {
    origin: Vec3f32,
    normal: Vec3f32,
    base_force: f32,
    trunc_distance: f32,
  },
}

// --- Educational IMEX Simulation Loop ---

/// Evaluates position-dependent forces (e.g., gravity) for all particles.
/// This should be called once at initialization (t=0) and then internally during Phase 5.
pub fn compute_particle_forces(particles: &mut [Particle], emitters: &[ForceEmitter]) {
  for p in particles.iter_mut() {
    let mut f = Vec3f32::zero();
    for e in emitters {
      match e {
        ForceEmitter::Gravity { position, mu } => {
          let r = *position - p.position;
          let dist_sq = r.length_squared();
          if dist_sq > 1e-6 {
            let dist = dist_sq.sqrt();
            // F = (mu * m / dist^3) * r
            f += r * (*mu * p.mass / (dist_sq * dist));
          }
        }
        ForceEmitter::Planar {
          origin,
          normal,
          base_force,
          trunc_distance,
        } => {
          let r = p.position - *origin;
          let dist = r.dot(*normal);
          if dist >= 0.0 && dist < *trunc_distance {
            f += *normal * (*base_force / (1.0 + dist * dist));
          }
        }
      }
    }
    p.accumulated_force = f;
  }
}

/// Evaluates forces acting on the Rigid Body to provide analytical Jacobian blocks.
pub struct RigidBodyForceEval {
  pub f_world: Vec3f32,
  pub tau_body: Vec3f32,
  /// Translational stiffness matrix K = dF_world / dx_world
  pub k_translation: Mat3f32,
  /// Attachment point of the force in the body's local frame.
  pub p_body: Vec3f32,
}

/// Helper function: Computes the Right Jacobian of SO(3), necessary for Lie Group integration.
pub fn right_jacobian(theta_vec: Vec3f32) -> Mat3f32 {
  let theta = theta_vec.length();
  let id = Mat3f32::identity();
  if theta < 1e-8 {
    return id;
  }

  let hat_theta = hat(theta_vec);
  let hat_theta2 = hat_theta * hat_theta;

  let a = (1.0 - theta.cos()) / (theta * theta);
  let b = (theta - theta.sin()) / (theta * theta * theta);

  id - (hat_theta * a) + (hat_theta2 * b)
}

/// A simple 6x6 linear system solver using Gaussian Elimination with partial pivoting.
pub fn solve_6x6(a: &mut [[f32; 6]; 6], b: &mut [f32; 6]) -> bool {
  let n = 6;
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

/// Core Newton-Raphson solver for the implicit rigid body step.
pub fn rigid_body_implicit_solve(
  rb: &RigidBody,
  h: f32,
  force_eval: impl Fn(Vec3f32, Mat3f32) -> RigidBodyForceEval,
) -> (Vec3f32, Vec3f32) {
  if h <= 1e-8 {
    return (rb.linear_velocity, rb.angular_velocity);
  }

  let mut v_mid = rb.linear_velocity;
  let mut w_mid = rb.angular_velocity;

  // Helper for matrix element access
  let get_mat3 = |m: &Mat3f32, r: usize, c: usize| -> f32 {
    let col = match c {
      0 => m.x,
      1 => m.y,
      _ => m.z,
    };
    col.component(r).unwrap()
  };

  for _iter in 0..10 {
    let x_mid = rb.position + v_mid * (h / 2.0);
    let r_mid = rb.rotation * expm_hat(w_mid * (h / 2.0));

    let eval = force_eval(x_mid, r_mid);
    let f_world = eval.f_world;
    let tau_body = eval.tau_body;
    let k_mat = eval.k_translation;
    let p_body = eval.p_body;

    let m = rb.mass;
    let j = rb.inertia_tensor;

    // Residuals
    let g_v = (v_mid - rb.linear_velocity) * (2.0 * m / h) - f_world;
    let j_w_mid = j.mul_vector(w_mid);
    let g_w =
      j.mul_vector(w_mid - rb.angular_velocity) * (2.0 / h) + w_mid.cross(j_w_mid) - tau_body;

    // Convergence Check
    if g_v.length_squared() < 1e-10 && g_w.length_squared() < 1e-10 {
      break;
    }

    // Jacobian Blocks
    let jr = right_jacobian(w_mid * (h / 2.0));

    // 1. dG_v / dv_mid
    let j_vv = Mat3f32::identity() * (2.0 * m / h) - k_mat * (h / 2.0);

    // 2. dG_v / dw_mid
    let j_vw = k_mat * r_mid * hat(p_body) * jr * (h / 2.0);

    // 3. dG_w / dv_mid
    let j_wv = hat(p_body) * r_mid.transpose() * k_mat * (-h / 2.0);

    // 4. dG_w / dw_mid
    let d_coriolis = hat(w_mid) * j - hat(j_w_mid);
    let f_world_body = r_mid.transpose().mul_vector(f_world);
    let geom_stiffness = hat(p_body) * hat(f_world_body)
      - hat(p_body) * r_mid.transpose() * k_mat * r_mid * hat(p_body);
    let j_ww = j * (2.0 / h) + d_coriolis - geom_stiffness * jr * (h / 2.0);

    // Assemble 6x6 Matrix
    let mut jacobian = [[0.0; 6]; 6];
    let mut residual = [0.0; 6];

    for r in 0..3 {
      for c in 0..3 {
        jacobian[r][c] = get_mat3(&j_vv, r, c);
        jacobian[r][c + 3] = get_mat3(&j_vw, r, c);
        jacobian[r + 3][c] = get_mat3(&j_wv, r, c);
        jacobian[r + 3][c + 3] = get_mat3(&j_ww, r, c);
      }
      residual[r] = g_v.component(r).unwrap();
      residual[r + 3] = g_w.component(r).unwrap();
    }

    if !solve_6x6(&mut jacobian, &mut residual) {
      break; // Singular or uninvertible Jacobian
    }

    // Apply Newton step: y_new = y_old - J^{-1} G
    v_mid = v_mid - Vec3f32::from_components(residual[0], residual[1], residual[2]);
    w_mid = w_mid - Vec3f32::from_components(residual[3], residual[4], residual[5]);
  }

  (v_mid, w_mid)
}

/// The full IMEX loop step marrying explicit particles with an implicit rigid body.
pub fn imex_step(
  particles: &mut [Particle],
  rigid_body: &mut RigidBody,
  emitters: &[ForceEmitter],
  h: f32,
) {
  // --- PHASE 1: Particle Explicit Velocity Half-Kick ---
  // At this point, particles[i].accumulated_force contains F_n (computed at end of previous step)
  for p in particles.iter_mut() {
    p.velocity += p.accumulated_force * (h / (2.0 * p.mass));
  }

  // --- PHASE 2: Particle Drift to Midpoint ---
  for p in particles.iter_mut() {
    p.position += p.velocity * (h / 2.0);
  }
  // Now particles are at q_{mid}

  // --- PHASE 3: Lie Group Implicit Solve for Rigid Body ---
  let force_eval = |x_mid: Vec3f32, _r_mid: Mat3f32| -> RigidBodyForceEval {
    let mut f_world = Vec3f32::zero();
    let mut k_translation = Mat3f32 {
      x: Vec3f32::zero(),
      y: Vec3f32::zero(),
      z: Vec3f32::zero(),
    };

    for e in emitters {
      match e {
        ForceEmitter::Gravity { position, mu } => {
          let r = *position - x_mid;
          let dist_sq = r.length_squared();
          if dist_sq > 1e-6 {
            let dist = dist_sq.sqrt();
            let dist3 = dist_sq * dist;
            let dist5 = dist3 * dist_sq;

            let coeff = *mu * rigid_body.mass / dist3;
            f_world += r * coeff;

            // dF/dx = K
            let term1 = Mat3f32::identity() * (-coeff);

            let rr_t = Mat3f32 {
              x: r * r.x(),
              y: r * r.y(),
              z: r * r.z(),
            };

            let term2 = rr_t * (3.0 * *mu * rigid_body.mass / dist5);

            k_translation = k_translation + term1 + term2;
          }
        }
        ForceEmitter::Planar {
          origin,
          normal,
          base_force,
          trunc_distance,
        } => {
          let r = x_mid - *origin;
          let dist = r.dot(*normal);
          if dist >= 0.0 && dist < *trunc_distance {
            let denom = 1.0 + dist * dist;
            let force_mag = *base_force / denom;
            f_world += *normal * force_mag;

            // dF/dx Jacobian
            let dF_ddist = -2.0 * *base_force * dist / (denom * denom);
            let j_term = dF_ddist;
            let nn_t = Mat3f32 {
              x: *normal * normal.x(),
              y: *normal * normal.y(),
              z: *normal * normal.z(),
            };
            k_translation = k_translation + nn_t * j_term;
          }
        }
      }
    }

    RigidBodyForceEval {
      f_world,
      tau_body: Vec3f32::zero(), // Uniform gravity acts directly on COM
      k_translation,
      p_body: Vec3f32::zero(),
    }
  };

  let (v_mid, w_mid) = rigid_body_implicit_solve(rigid_body, h, force_eval);

  // --- PHASE 4: Rigid Body Full State Update ---
  rigid_body.position += v_mid * h;
  rigid_body.rotation = rigid_body.rotation * expm_hat(w_mid * h);
  rigid_body.linear_velocity = v_mid * 2.0 - rigid_body.linear_velocity;
  rigid_body.angular_velocity = w_mid * 2.0 - rigid_body.angular_velocity;

  // --- PHASE 5: Particle Second Drift and Final Kick ---
  for p in particles.iter_mut() {
    p.position += p.velocity * (h / 2.0);
  }

  // Evaluate forces at the new t_{n+1} configurations
  // This perfectly prepares F_{n+1} for Phase 1 of the NEXT step!
  compute_particle_forces(particles, emitters);

  for p in particles.iter_mut() {
    p.velocity += p.accumulated_force * (h / (2.0 * p.mass));
  }
}

// ============================================================================
// Continuous Collision Detection (CCD)
// ============================================================================

/// Computes the Time of Impact (TOI) and collision normal between a moving sphere and a static triangle.
/// TODO move to intersection module
/// TODO Actually, remove, nobody uses it
#[deprecated]
pub fn ccd_sphere_triangle(
  p0: Vec3f32,
  p1: Vec3f32,
  radius: f32,
  v0: Vec3f32,
  v1: Vec3f32,
  v2: Vec3f32,
) -> Option<(f32, Vec3f32)> {
  let dir = p1 - p0;
  let edge1 = v1 - v0;
  let edge2 = v2 - v0;
  let tri_normal = edge1.cross(edge2).normalize();

  let dist_to_plane = (p0 - v0).dot(tri_normal);
  let dir_dot_n = dir.dot(tri_normal);

  if dir_dot_n.abs() < 1e-6 {
    return None;
  }

  let t = (radius * dist_to_plane.signum() - dist_to_plane) / dir_dot_n;
  if !(0.0_f32..=1.0).contains(&t) {
    return None;
  }

  let hit_point = p0 + dir * t - tri_normal * (radius * dist_to_plane.signum());
  let w = hit_point - v0;
  let uu = edge1.dot(edge1);
  let uv = edge1.dot(edge2);
  let vv = edge2.dot(edge2);
  let wu = w.dot(edge1);
  let wv = w.dot(edge2);
  let denom = uv * uv - uu * vv;

  let s = (uv * wv - vv * wu) / denom;
  let r = (uv * wu - uu * wv) / denom;

  if s >= 0.0 && r >= 0.0 && (s + r) <= 1.0 {
    return Some((t, tri_normal * dist_to_plane.signum()));
  }

  None
}

// ============================================================================
// LCP Solver (Dantzig / Baraff / PGS)
// ============================================================================

/// A simplified Projected Gauss-Seidel solver for Linear Complementarity Problems
/// arising from multiple simultaneous contacts.
/// Solves A * x + b >= 0, x >= 0, x^T (A * x + b) = 0
pub fn solve_lcp_pgs(a: &[Vec<f32>], b: &[f32], max_iters: usize) -> Vec<f32> {
  let n = b.len();
  let mut x = alloc::vec![0.0; n];

  for _ in 0..max_iters {
    for i in 0..n {
      let mut sum = 0.0;
      for j in 0..n {
        if i != j {
          sum += a[i][j] * x[j];
        }
      }

      let a_ii = a[i][i];
      let mut new_x = 0.0;
      if a_ii.abs() > 1e-6 {
        new_x = (b[i] - sum) / a_ii;
      }

      // Project onto constraints (x >= 0)
      x[i] = new_x.max(0.0);
    }
  }
  x
}
