//! cpu module.

use crate::math::{
  compute_com_and_tensor, expm_hat, hat, jacobi_diagonalization, solve_12x12, vee,
};
use aethervk_oshal_rlib::{
  math::{
    FloatLike, MulAddIdentity,
    matrix::{Matrix, Matrix3, MatrixVectorMul, mat3::Mat3f32},
    vector::{Vector, Vector3, vec3::Vec3f32},
  },
  os::pool::{ThreadPool, Workload, WorkloadStatus},
};
use alloc::{boxed::Box, vec::Vec};

/// TODO: Document this item
pub const G: f32 = 1.0;
/// TODO: Document this item
pub const ANELASTIC_LOOP_COUNT_THRESHOLD: usize = 10;
/// TODO: Document this item
pub const PARTICLE_NUCLEUS_RESTITUTION: f32 = 0.8;

/// TODO: Document this item
pub struct RigidBody {
  pub position: Vec3f32,
  pub linear_momentum: Vec3f32,
  pub orientation: Mat3f32,
  pub pi: Vec3f32,
  pub mass: f32,
  pub com_offset_body: Vec3f32,
  pub inertia_tensor_body: Mat3f32,
  pub inertia_tensor_body_inv: Mat3f32,
  pub principal_axes_r: Mat3f32,
  pub vertices_com_frame: Vec<Vec3f32>,
}

impl RigidBody {
  /// TODO: Document this item
  pub fn new(
    initial_pos_com: Vec3f32,
    initial_vel: Vec3f32,
    initial_ang_vel_raw: Vec3f32,
    raw_vertices: &[Vec3f32],
    mass_per_vertex: f32,
  ) -> Self {
    let mass = mass_per_vertex * (raw_vertices.len() as f32);
    let (com_offset_body, raw_i) = compute_com_and_tensor(raw_vertices, mass_per_vertex);
    let (principal_moments, principal_axes_r) = jacobi_diagonalization(raw_i, 1e-6, 1000);

    let inertia_tensor_body = Mat3f32 {
      x: Vec3f32::from_components(principal_moments.x(), 0.0, 0.0),
      y: Vec3f32::from_components(0.0, principal_moments.y(), 0.0),
      z: Vec3f32::from_components(0.0, 0.0, principal_moments.z()),
    };
    let inertia_tensor_body_inv = Mat3f32 {
      x: Vec3f32::from_components(1.0 / principal_moments.x(), 0.0, 0.0),
      y: Vec3f32::from_components(0.0, 1.0 / principal_moments.y(), 0.0),
      z: Vec3f32::from_components(0.0, 0.0, 1.0 / principal_moments.z()),
    };

    let mut vertices_com_frame = Vec::with_capacity(raw_vertices.len());
    let principal_axes_r_t = principal_axes_r.transpose();
    for v in raw_vertices {
      let v_com = *v - com_offset_body;
      vertices_com_frame.push(principal_axes_r_t.mul_vector(v_com));
    }

    let linear_momentum = initial_vel * mass;
    let ang_vel_principal = principal_axes_r_t.mul_vector(initial_ang_vel_raw);
    let pi = inertia_tensor_body.mul_vector(ang_vel_principal);

    Self {
      position: initial_pos_com,
      linear_momentum,
      orientation: principal_axes_r,
      pi,
      mass,
      com_offset_body,
      inertia_tensor_body,
      inertia_tensor_body_inv,
      principal_axes_r,
      vertices_com_frame,
    }
  }

  /// TODO: Document this item
  pub fn velocity(&self) -> Vec3f32 {
    self.linear_momentum / self.mass
  }

  /// TODO: Document this item
  pub fn angular_velocity_body(&self) -> Vec3f32 {
    self.inertia_tensor_body_inv.mul_vector(self.pi)
  }

  /// TODO: Document this item
  pub fn body_origin_world(&self) -> Vec3f32 {
    self.position
      - self
        .orientation
        .mul_vector(self.principal_axes_r.transpose().mul_vector(self.com_offset_body))
  }

  /// TODO: Document this item
  pub fn get_world_vertices(&self) -> Vec<Vec3f32> {
    let mut world_verts = Vec::with_capacity(self.vertices_com_frame.len());
    for v in &self.vertices_com_frame {
      world_verts.push(self.orientation.mul_vector(*v) + self.position);
    }
    world_verts
  }

  /// TODO: Document this item
  pub fn get_comet_triangles(&self) -> Vec<(Vec3f32, Vec3f32, Vec3f32)> {
    let verts = self.get_world_vertices();
    if verts.len() < 3 {
      return Vec::new();
    }
    let mut triangles = Vec::new();
    for i in 1..(verts.len() - 1) {
      triangles.push((verts[0], verts[i], verts[i + 1]));
    }
    triangles
  }
}

/// TODO: Document this item
pub struct Particle {
  pub position: Vec3f32,
  pub velocity: Vec3f32,
  pub mass: f32,
  pub radius: f32,
  pub beta: f32,
  pub lifetime: f32,
  pub collided: bool,
}

impl Particle {
  /// TODO: Document this item
  pub fn update(&mut self, dt: f32) {
    self.lifetime -= dt;
  }

  /// TODO: Document this item
  pub fn is_alive(&self) -> bool {
    self.lifetime > 0.0
  }
}

/// TODO: Document this item
pub fn gravitational_force(pos: Vec3f32, sun_pos: Vec3f32, m: f32, sun_m: f32) -> Vec3f32 {
  let r_vec = sun_pos - pos;
  let dist = r_vec.length();
  if dist < 1.0 {
    return Vec3f32::zero();
  }
  r_vec * (G * m * sun_m / (dist * dist * dist))
}

/// TODO: Document this item
pub fn get_force_and_torque(
  pos: Vec3f32,
  mass: f32,
  sun_pos: Vec3f32,
  sun_mass: f32,
) -> (Vec3f32, Vec3f32) {
  let force_world = gravitational_force(pos, sun_pos, mass, sun_mass);
  let torque_body = Vec3f32::zero();
  (force_world, torque_body)
}

/// TODO: Document this item
pub fn calculate_total_energy(body: &RigidBody, sun_pos: Vec3f32, sun_mass: f32) -> f32 {
  let r_dist = (sun_pos - body.position).length();
  let potential_energy = if r_dist > 0.0 {
    -G * body.mass * sun_mass / r_dist
  } else {
    0.0
  };
  let v = body.velocity();
  let trans_ke = 0.5 * body.mass * v.dot(v);
  let rot_ke = 0.5 * body.angular_velocity_body().dot(body.pi);
  potential_energy + trans_ke + rot_ke
}

/// TODO: Document this item
pub fn particle_implicit_midpoint_step(
  p: &mut Particle,
  dt: f32,
  sun_pos: Vec3f32,
  sun_mass: f32,
  comet: &RigidBody,
) {
  let x_n = p.position;
  let v_n = p.velocity;

  let force_n = gravitational_force(x_n, sun_pos, p.mass, sun_mass) * (1.0 - p.beta)
    + gravitational_force(x_n, comet.position, p.mass, comet.mass);

  let mut v_guess = v_n + force_n * (dt / p.mass);
  let mut x_guess = x_n + v_n * dt;

  for _ in 0..3 {
    let x_mid = (x_n + x_guess) * 0.5;
    let v_mid = (v_n + v_guess) * 0.5;

    let force_mid = gravitational_force(x_mid, sun_pos, p.mass, sun_mass) * (1.0 - p.beta)
      + gravitational_force(x_mid, comet.position, p.mass, comet.mass);

    v_guess = v_n + force_mid * (dt / p.mass);
    x_guess = x_n + v_mid * dt;
  }

  p.position = x_guess;
  p.velocity = v_guess;
}

/// TODO: Document this item
pub fn implicit_midpoint_step(body: &mut RigidBody, h: f32, sun_pos: Vec3f32, sun_mass: f32) {
  let x_n = body.position;
  let p_n = body.linear_momentum;
  let r_n = body.orientation;
  let pi_n = body.pi;
  let i_inv = body.inertia_tensor_body_inv;
  let m_inv = 1.0 / body.mass;

  let (f_n, tau_n) = get_force_and_torque(x_n, body.mass, sun_pos, sun_mass);
  let mut x_guess = x_n + p_n * (h * m_inv);
  let mut p_guess = p_n + f_n * h;
  let omega_n = i_inv.mul_vector(pi_n);
  let mut r_guess = r_n * expm_hat(omega_n * h);
  let mut pi_guess = pi_n + (pi_n.cross(omega_n) + tau_n) * h;

  for _ in 0..10 {
    let x_mid = (x_n + x_guess) * 0.5;
    let p_mid = (p_n + p_guess) * 0.5;
    let pi_mid = (pi_n + pi_guess) * 0.5;

    let omega_mid = i_inv.mul_vector(pi_mid);
    let _r_mid = r_n * expm_hat(omega_mid * (0.5 * h));
    let (f_mid, tau_mid) = get_force_and_torque(x_mid, body.mass, sun_pos, sun_mass);

    let x_res = x_guess - (x_n + p_mid * (h * m_inv));
    let p_res = p_guess - (p_n + f_mid * h);
    let pi_res = pi_guess - (pi_n + (pi_mid.cross(omega_mid) + tau_mid) * h);

    let r_err_matrix = (r_n * expm_hat(omega_mid * h)) * r_guess.transpose();
    let r_res = vee(r_err_matrix - r_err_matrix.transpose());

    let res_norm = (x_res.length_squared()
      + p_res.length_squared()
      + pi_res.length_squared()
      + r_res.length_squared())
    .sqrt();
    if res_norm < 1e-9 {
      break;
    }

    let mut j = [[0.0f32; 12]; 12];
    for i in 0..3 {
      j[i][i] = 1.0;
      j[i][i + 3] = -0.5 * h * m_inv;
      j[i + 3][i + 3] = 1.0;
      j[i + 9][i + 9] = 1.0;
    }

    let hat_o_mid = hat(omega_mid);
    let hat_i_inv_pi_mid = hat(i_inv.mul_vector(pi_mid));
    let mid_term = (hat_o_mid - hat_i_inv_pi_mid) * i_inv;

    for r in 0..3 {
      for c in 0..3 {
        j[r + 6][c + 6] = if r == c { 1.0 } else { 0.0 } - 0.5 * h * mid_term[(r, c)];
      }
    }

    let mut b = [
      -x_res.x(),
      -x_res.y(),
      -x_res.z(),
      -p_res.x(),
      -p_res.y(),
      -p_res.z(),
      -pi_res.x(),
      -pi_res.y(),
      -pi_res.z(),
      -r_res.x(),
      -r_res.y(),
      -r_res.z(),
    ];

    if !solve_12x12(&mut j, &mut b) {
      break;
    }

    x_guess += Vec3f32::from_components(b[0], b[1], b[2]);
    p_guess += Vec3f32::from_components(b[3], b[4], b[5]);
    pi_guess += Vec3f32::from_components(b[6], b[7], b[8]);
    let d_theta = Vec3f32::from_components(b[9], b[10], b[11]);
    r_guess = expm_hat(d_theta) * r_guess;
  }

  body.position = x_guess;
  body.linear_momentum = p_guess;
  body.orientation = r_guess;
  body.pi = pi_guess;
}

/// TODO: Document this item
pub fn sphere_triangle_intersection(
  _p: Vec3f32,
  p_next: Vec3f32,
  r: f32,
  tri: &(Vec3f32, Vec3f32, Vec3f32),
) -> (Option<f32>, Option<Vec3f32>) {
  let (v0, v1, v2) = tri;
  let mut normal = (*v1 - *v0).cross(*v2 - *v0);
  let len = normal.length();
  if len > 1e-8 {
    normal /= len;
  }

  let dist = (p_next - *v0).dot(normal);
  if dist.abs() < r {
    return (Some(0.99), Some(normal));
  }
  (None, None)
}

/// TODO: Document this item
pub fn continuous_collision_detection(
  particle: &Particle,
  comet: &RigidBody,
  dt: f32,
) -> (Option<f32>, Option<Vec3f32>) {
  let p_initial = particle.position;
  let p_next = p_initial + particle.velocity * dt;

  let mut earliest_toi = f32::MAX;
  let mut collision_normal = None;

  let triangles = comet.get_comet_triangles();
  for triangle in triangles {
    let (toi, normal) = sphere_triangle_intersection(p_initial, p_next, particle.radius, &triangle);
    if let Some(t) = toi {
      if t < earliest_toi {
        earliest_toi = t;
        collision_normal = normal;
      }
    }
  }

  if earliest_toi <= 1.0 {
    return (Some(earliest_toi), collision_normal);
  }
  (None, None)
}

/// TODO: Document this item
pub fn rewind_and_resolve_collision(particle: &mut Particle, toi: f32, normal: Vec3f32, dt: f32) {
  particle.position -= particle.velocity * ((1.0 - toi) * dt);

  let v_n = normal * particle.velocity.dot(normal);
  let v_t = particle.velocity - v_n;
  particle.velocity = v_t - v_n * PARTICLE_NUCLEUS_RESTITUTION;

  particle.collided = true;
}

/// TODO: Document this item
pub fn update_particles_single_threaded(
  particles: &mut [Particle],
  comet: &RigidBody,
  sun_pos: Vec3f32,
  sun_mass: f32,
  dt: f32,
) {
  for p in particles.iter_mut() {
    if p.collided {
      p.collided = false;
      continue;
    }

    let initial_position = p.position;
    let initial_velocity = p.velocity;

    particle_implicit_midpoint_step(p, dt, sun_pos, sun_mass, comet);

    let (toi, _normal) = continuous_collision_detection(p, comet, dt);

    if let Some(_) = toi {
      p.position = initial_position;
      p.velocity = initial_velocity;
      p.collided = true;
    } else {
      p.update(dt);
    }
  }
}

struct ParticleUpdateWorkload {
  particles_ptr: *mut Particle,
  particles_len: usize,
  comet: *const RigidBody,
  sun_pos: Vec3f32,
  sun_mass: f32,
  dt: f32,
}

unsafe impl Send for ParticleUpdateWorkload {}
unsafe impl Sync for ParticleUpdateWorkload {}

impl Workload for ParticleUpdateWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let particles =
      unsafe { core::slice::from_raw_parts_mut(self.particles_ptr, self.particles_len) };
    let comet = unsafe { &*self.comet };

    update_particles_single_threaded(particles, comet, self.sun_pos, self.sun_mass, self.dt);
    aethervk_oshal_rlib::os::pool::WorkloadStatus::Complete
  }
}

/// TODO: Document this item
pub fn update_particles_multi_threaded(
  particles: &mut [Particle],
  comet: &RigidBody,
  sun_pos: Vec3f32,
  sun_mass: f32,
  dt: f32,
  pool: &mut ThreadPool,
  chunk_size: usize,
) {
  let mut workloads: Vec<Box<dyn Workload>> = Vec::new();

  let comet_ptr = comet as *const RigidBody;
  let mut offset = 0;

  while offset < particles.len() {
    let size = (particles.len() - offset).min(chunk_size);
    let ptr = unsafe { particles.as_mut_ptr().add(offset) };

    workloads.push(Box::new(ParticleUpdateWorkload {
      particles_ptr: ptr,
      particles_len: size,
      comet: comet_ptr,
      sun_pos,
      sun_mass,
      dt,
    }));

    offset += size;
  }

  if workloads.is_empty() {
    return;
  }

  if let Ok(_) = pool.scatter(workloads) {
    pool.gather();
  } else {
    update_particles_single_threaded(particles, comet, sun_pos, sun_mass, dt);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use aethervk_oshal_rlib::math::{matrix::mat3::Mat3f32, vector::vec3::Vec3f32};

  #[test]
  fn test_gravitational_force() {
    let p_pos = Vec3f32::from_components(1.0e7, 0.0, 0.0);
    let s_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
    let f = gravitational_force(p_pos, s_pos, 1.0, 1.0e10);
    assert!(f.x() < 0.0);
    assert_eq!(f.y(), 0.0);
    assert_eq!(f.z(), 0.0);
  }

  #[test]
  fn test_rigidbody_initialization() {
    let verts = vec![
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(1.0, 0.0, 0.0),
      Vec3f32::from_components(0.0, 1.0, 0.0),
      Vec3f32::from_components(0.0, 0.0, 1.0),
    ];
    let rb = RigidBody::new(
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(1.0, 0.0, 0.0),
      Vec3f32::from_components(0.0, 0.0, 0.0),
      &verts,
      1.0,
    );
    assert_eq!(rb.mass, 4.0);
    assert_eq!(rb.velocity().x(), 1.0);
  }

  #[test]
  fn test_particle_update() {
    let mut p = Particle {
      position: Vec3f32::from_components(10.0, 0.0, 0.0),
      velocity: Vec3f32::from_components(-1.0, 0.0, 0.0),
      mass: 1.0,
      radius: 1.0,
      beta: 0.0,
      lifetime: 10.0,
      collided: false,
    };
    p.update(1.0);
    assert_eq!(p.lifetime, 9.0);
    assert!(p.is_alive());
  }

  #[test]
  fn test_continuous_collision_detection() {
    let p = Particle {
      position: Vec3f32::from_components(0.0, 0.0, 5.0),
      velocity: Vec3f32::from_components(0.0, 0.0, -10.0),
      mass: 1.0,
      radius: 0.5,
      beta: 0.0,
      lifetime: 10.0,
      collided: false,
    };

    let verts = vec![
      Vec3f32::from_components(-1.0, -1.0, 0.0),
      Vec3f32::from_components(1.0, -1.0, 0.0),
      Vec3f32::from_components(0.0, 1.0, 0.0),
      Vec3f32::from_components(0.0, 0.0, -1.0),
    ];
    let comet = RigidBody::new(
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(0.0, 0.0, 0.0),
      Vec3f32::from_components(0.0, 0.0, 0.0),
      &verts,
      1.0,
    );

    let (toi, normal) = continuous_collision_detection(&p, &comet, 0.5);
    assert!(toi.is_some());
    assert!(normal.is_some());
  }
}
