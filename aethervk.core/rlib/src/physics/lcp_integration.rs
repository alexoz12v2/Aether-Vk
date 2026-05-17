use crate::{
  gpu::{CollisionPair, RigidBodyGpu},
  math::collision::lcp::solve_lcp_pgs,
};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, vec3::Vec3f32};

pub fn resolve_cluster_lcp(
  cluster: &[CollisionPair],
  rigid_bodies: &mut [RigidBodyGpu],
  restitution: f32,
) {
  let m = cluster.len();
  if m == 0 {
    return;
  }

  let mut a_matrix = alloc::vec![0.0; m * m];
  let mut b_vector = alloc::vec![0.0; m];
  let mut impulses = alloc::vec![0.0; m];

  // Build A matrix and b vector
  for i in 0..m {
    let pair_i = &cluster[i];
    let idx_a_i = pair_i.a.primitive_index as usize;
    let idx_b_i = pair_i.b.primitive_index as usize;

    if idx_a_i >= rigid_bodies.len() || idx_b_i >= rigid_bodies.len() {
      continue;
    }

    let m_a_i = rigid_bodies[idx_a_i].mass;
    let m_b_i = rigid_bodies[idx_b_i].mass;
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
    let v_rel = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
      rigid_bodies[idx_a_i].linear_velocity,
    ) - aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
      rigid_bodies[idx_b_i].linear_velocity,
    );
    let v_rel_n = v_rel.dot(n_i);

    let beta = 0.2;
    let slop = 0.01;
    let bias = (beta / 0.016) * (pair_i.penetration_depth - slop).max(0.0);

    // We want A x + b >= 0, so b_i is the linear_velocity + bias
    // In PGS LCP standard form w = A x + b, where x is impulse, w is outgoing linear_velocity
    b_vector[i] = (1.0 + restitution) * v_rel_n - bias;
  }

  // Solve LCP
  solve_lcp_pgs(&a_matrix, &b_vector, &mut impulses, 20);

  // Apply impulses
  for i in 0..m {
    if impulses[i] <= 0.0 {
      continue;
    }

    let pair_i = &cluster[i];
    let idx_a_i = pair_i.a.primitive_index as usize;
    let idx_b_i = pair_i.b.primitive_index as usize;

    let n_i = Vec3f32::from_array(pair_i.contact_normal);

    let impulse_vec = n_i * impulses[i];
    let inv_m_a_i = if rigid_bodies[idx_a_i].mass > 0.0 {
      1.0 / rigid_bodies[idx_a_i].mass
    } else {
      0.0
    };
    let inv_m_b_i = if rigid_bodies[idx_b_i].mass > 0.0 {
      1.0 / rigid_bodies[idx_b_i].mass
    } else {
      0.0
    };

    let new_v_a = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
      rigid_bodies[idx_a_i].linear_velocity,
    ) + impulse_vec * inv_m_a_i;
    rigid_bodies[idx_a_i].linear_velocity = [new_v_a.x(), new_v_a.y(), new_v_a.z()];
    let new_v_b = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(
      rigid_bodies[idx_b_i].linear_velocity,
    ) - impulse_vec * inv_m_b_i;
    rigid_bodies[idx_b_i].linear_velocity = [new_v_b.x(), new_v_b.y(), new_v_b.z()];
  }
}
