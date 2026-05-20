use crate::{
  gpu::{CollisionPair, ParticleGpu, RigidBodyGpu},
  math::collision::lcp::solve_lcp_pgs,
};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};

pub fn resolve_cluster_lcp(
  cluster: &[CollisionPair],
  kinematics: &[crate::gpu::KinematicBody],
  rigid_bodies: &mut [RigidBodyGpu],
  particles: &mut [ParticleGpu],
  restitution: f32,
) {
  let m = cluster.len();
  if m == 0 {
    return;
  }

  let mut a_matrix = alloc::vec![0.0; m * m];
  let mut b_vector = alloc::vec![0.0; m];
  let mut impulses = alloc::vec![0.0; m];

  let get_mass =
    |idx: usize, rigid_bodies: &[RigidBodyGpu], particles: &[ParticleGpu]| -> Option<f32> {
      if (idx & (1 << 31)) != 0 {
        None // Kinematic, infinite mass
      } else if (idx & (1 << 30)) != 0 {
        let i = idx & !(1 << 30);
        particles.get(i).map(|p| p.mass)
      } else {
        rigid_bodies.get(idx).map(|rb| rb.mass)
      }
    };

  let get_velocity =
    |idx: usize, rigid_bodies: &[RigidBodyGpu], particles: &[ParticleGpu]| -> Option<Vec3f32> {
      if (idx & (1 << 31)) != 0 {
        let i = idx & !(1 << 31);
        kinematics.get(i).map(|k| k.velocity)
      } else if (idx & (1 << 30)) != 0 {
        let i = idx & !(1 << 30);
        particles.get(i).map(|p| Vec3f32::from_array(p.velocity))
      } else {
        rigid_bodies.get(idx).map(|rb| Vec3f32::from_array(rb.linear_velocity))
      }
    };

  let add_velocity =
    |idx: usize, dv: Vec3f32, rigid_bodies: &mut [RigidBodyGpu], particles: &mut [ParticleGpu]| {
      if (idx & (1 << 31)) != 0 {
        // Kinematic
      } else if (idx & (1 << 30)) != 0 {
        let i = idx & !(1 << 30);
        if let Some(p) = particles.get_mut(i) {
          let mut v = Vec3f32::from_array(p.velocity);
          v += dv;
          p.velocity = [v.x(), v.y(), v.z()];
        }
      } else {
        if let Some(rb) = rigid_bodies.get_mut(idx) {
          let mut v = Vec3f32::from_array(rb.linear_velocity);
          v += dv;
          rb.linear_velocity = [v.x(), v.y(), v.z()];
        }
      }
    };

  // Build A matrix and b vector
  for i in 0..m {
    let pair_i = &cluster[i];
    let idx_a_i = pair_i.a.primitive_index as usize;
    let idx_b_i = pair_i.b.primitive_index as usize;

    let m_a_i = get_mass(idx_a_i, rigid_bodies, particles).unwrap_or(0.0);
    let m_b_i = get_mass(idx_b_i, rigid_bodies, particles).unwrap_or(0.0);
    let inv_m_a_i = if m_a_i > 0.0 { 1.0 / m_a_i } else { 0.0 };
    let inv_m_b_i = if m_b_i > 0.0 { 1.0 / m_b_i } else { 0.0 };

    let n_i = Vec3f32::from_array(pair_i.contact_normal);

    // Compute diagonal element A_ii
    a_matrix[i * m + i] = inv_m_a_i + inv_m_b_i;

    // Compute off-diagonal elements A_ij
    for j in (i + 1)..m {
      let pair_j = &cluster[j];
      let idx_a_j = pair_j.a.primitive_index as usize;
      let idx_b_j = pair_j.b.primitive_index as usize;

      let n_j = Vec3f32::from_array(pair_j.contact_normal);

      let mut a_ij = 0.0;
      if idx_a_i == idx_a_j {
        a_ij += inv_m_a_i * n_i.dot(n_j);
      }
      if idx_a_i == idx_b_j {
        a_ij -= inv_m_a_i * n_i.dot(n_j);
      }
      if idx_b_i == idx_a_j {
        a_ij -= inv_m_b_i * n_i.dot(n_j);
      }
      if idx_b_i == idx_b_j {
        a_ij += inv_m_b_i * n_i.dot(n_j);
      }

      a_matrix[i * m + j] = a_ij;
      a_matrix[j * m + i] = a_ij; // Symmetric
    }

    // Compute b_i
    let v_a_i = get_velocity(idx_a_i, rigid_bodies, particles).unwrap_or_default();
    let v_b_i = get_velocity(idx_b_i, rigid_bodies, particles).unwrap_or_default();
    let v_rel = v_a_i - v_b_i;
    let v_rel_n = v_rel.dot(n_i);

    let beta = 0.2;
    let slop = 0.01;
    let bias = (beta / 0.016) * (pair_i.penetration_depth - slop).max(0.0);

    // We want A x + b >= 0, so b_i is the velocity + bias
    // In PGS LCP standard form w = A x + b, where x is impulse, w is outgoing velocity
    b_vector[i] = (1.0 + restitution) * v_rel_n - bias;

    #[cfg(test)]
    aethervk_oshal_rlib::log!(
      "LCP Setup: idx_a={}, idx_b={}, a_ii={}, b_i={}, v_rel_n={}, bias={}, depth={}",
      idx_a_i,
      idx_b_i,
      a_matrix[i * m + i],
      b_vector[i],
      v_rel_n,
      bias,
      pair_i.penetration_depth
    );
  }

  // Solve LCP
  solve_lcp_pgs(&a_matrix, &b_vector, &mut impulses, 20);

  // Apply impulses
  for i in 0..m {
    #[cfg(test)]
    aethervk_oshal_rlib::log!("LCP Result: impulse[{}]={}", i, impulses[i]);

    if impulses[i] <= 0.0 {
      continue;
    }

    let pair_i = &cluster[i];
    let idx_a_i = pair_i.a.primitive_index as usize;
    let idx_b_i = pair_i.b.primitive_index as usize;

    let n_i = Vec3f32::from_array(pair_i.contact_normal);
    let impulse_vec = n_i * impulses[i];

    let m_a_i = get_mass(idx_a_i, rigid_bodies, particles).unwrap_or(0.0);
    let m_b_i = get_mass(idx_b_i, rigid_bodies, particles).unwrap_or(0.0);

    let inv_m_a_i = if m_a_i > 0.0 { 1.0 / m_a_i } else { 0.0 };
    let inv_m_b_i = if m_b_i > 0.0 { 1.0 / m_b_i } else { 0.0 };

    let impulse_a = impulse_vec * inv_m_a_i;
    let impulse_b = -impulse_vec * inv_m_b_i;

    // The impulse is currently in Global Space, but object velocities are strictly Local Space!
    // We must rotate the impulse vector backwards (Global -> Local) using the object's transform
    let mut apply_local_impulse = |idx: usize, global_dv: Vec3f32| {
      if (idx & (1 << 31)) != 0 || (idx & (1 << 30)) != 0 {
        // Particles and kinematics (currently) aren't receiving dynamic impulse responses
        // but if they were, we would need their frame inverse here as well.
        add_velocity(idx, global_dv, rigid_bodies, particles);
      } else {
        if let Some(rb) = rigid_bodies.get_mut(idx) {
          // rb.rotation is a 3x3 matrix storing the local-to-global rotation
          // Its transpose acts as the global-to-local rotation
          use aethervk_oshal_rlib::math::matrix::{
            Matrix, Matrix3, MatrixVectorMul, mat3::Mat3f32,
          };
          let rot_mat = Mat3f32::from_array(&[
            rb.rotation[0][0],
            rb.rotation[0][1],
            rb.rotation[0][2],
            rb.rotation[1][0],
            rb.rotation[1][1],
            rb.rotation[1][2],
            rb.rotation[2][0],
            rb.rotation[2][1],
            rb.rotation[2][2],
          ]);
          let local_dv = rot_mat.transpose().mul_vector(global_dv);
          let mut v = Vec3f32::from_array(rb.linear_velocity);
          v += local_dv;
          rb.linear_velocity = [v.x(), v.y(), v.z()];
        }
      }
    };

    apply_local_impulse(idx_a_i, impulse_a);
    apply_local_impulse(idx_b_i, impulse_b);
  }
}
