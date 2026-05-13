use crate::gpu::{CollisionPair, DynamicBody};
use crate::math::collision::lcp::solve_lcp_pgs;
use aethervk_oshal_rlib::math::vector::Vector;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

pub fn resolve_cluster_lcp(
  cluster: &[CollisionPair],
  dyn_array: &mut [DynamicBody],
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

    if idx_a_i >= dyn_array.len() || idx_b_i >= dyn_array.len() {
      continue;
    }

    let m_a_i = dyn_array[idx_a_i].mass;
    let m_b_i = dyn_array[idx_b_i].mass;
    let inv_m_a_i = if m_a_i > 0.0 { 1.0 / m_a_i } else { 0.0 };
    let inv_m_b_i = if m_b_i > 0.0 { 1.0 / m_b_i } else { 0.0 };

    let pos_a_i = dyn_array[idx_a_i].transform.position;
    let pos_b_i = dyn_array[idx_b_i].transform.position;
    let mut n_i = pos_a_i - pos_b_i;
    let dist_i = n_i.length();

    if dist_i > 1e-6 {
      n_i = n_i / dist_i;
    } else {
      n_i = Vec3f32::from_array([1.0, 0.0, 0.0]);
    }

    // Compute diagonal element A_ii
    a_matrix[i * m + i] = inv_m_a_i + inv_m_b_i;

    // Compute off-diagonal elements A_ij
    for j in (i + 1)..m {
      let pair_j = &cluster[j];
      let idx_a_j = pair_j.a.primitive_index as usize;
      let idx_b_j = pair_j.b.primitive_index as usize;

      let pos_a_j = dyn_array[idx_a_j].transform.position;
      let pos_b_j = dyn_array[idx_b_j].transform.position;
      let mut n_j = pos_a_j - pos_b_j;
      if n_j.length_squared() > 1e-6 {
        n_j = n_j.normalize();
      } else {
        n_j = Vec3f32::from_array([1.0, 0.0, 0.0]);
      }

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
    let v_rel = dyn_array[idx_a_i].velocity - dyn_array[idx_b_i].velocity;
    let v_rel_n = v_rel.dot(n_i);

    let penetration = (2.0 - dist_i).max(0.0);
    let beta = 0.2;
    let slop = 0.01;
    let bias = (beta / 0.016) * (penetration - slop).max(0.0);

    // We want A x + b >= 0, so b_i is the velocity + bias
    // In PGS LCP standard form w = A x + b, where x is impulse, w is outgoing velocity
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

    let pos_a_i = dyn_array[idx_a_i].transform.position;
    let pos_b_i = dyn_array[idx_b_i].transform.position;
    let mut n_i = pos_a_i - pos_b_i;
    if n_i.length_squared() > 1e-6 {
      n_i = n_i.normalize();
    } else {
      n_i = Vec3f32::from_array([1.0, 0.0, 0.0]);
    }

    let impulse_vec = n_i * impulses[i];
    let inv_m_a_i = if dyn_array[idx_a_i].mass > 0.0 {
      1.0 / dyn_array[idx_a_i].mass
    } else {
      0.0
    };
    let inv_m_b_i = if dyn_array[idx_b_i].mass > 0.0 {
      1.0 / dyn_array[idx_b_i].mass
    } else {
      0.0
    };

    dyn_array[idx_a_i].velocity = dyn_array[idx_a_i].velocity + impulse_vec * inv_m_a_i;
    dyn_array[idx_b_i].velocity = dyn_array[idx_b_i].velocity - impulse_vec * inv_m_b_i;
  }
}
