extern crate alloc;
use crate::scene::Component;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};
use aethervk_oshal_rlib::os::time::timeus_t;
use crate::math::collision::linear_bvh::LinearBVH;
use crate::physics::particle::Particle;
use crate::simulation::comet::Comet;
use crate::simulation::comet::uv_grid::UvGrid;

#[derive(Clone, Debug)]
pub struct GaussianParams {
  pub mean: f32,
  pub std_dev: f32,
  pub min: f32,
  pub max: f32,
}

impl GaussianParams {
  pub fn sample(&self, u: &[f32; 2]) -> f32 {
    let r = (-2.0 * u[0].max(1e-8).ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u[1];
    let z0 = r * theta.cos();
    (self.mean + self.std_dev * z0).clamp(self.min, self.max)
  }
}

#[derive(Clone, Debug)]
pub struct ParticleEmitterConfig {
  pub uv_center: [f32; 2],
  pub uv_radius: f32,
  pub delta: timeus_t,
  pub max_particles: usize,
  pub velocity_intensity: GaussianParams,
  pub emission_count: GaussianParams,
  pub particle_radius: f32,
  pub density: f32,
  pub lifetime: timeus_t,
  pub color: [f32; 4],
  pub beta: f32,
}

impl ParticleEmitterConfig {
  pub fn from_point(
    target_point_local: Vec3f32,
    comet: &Comet,
    mut config: ParticleEmitterConfig,
  ) -> Option<Self> {
    use crate::math::collision::intersection::{self, Ray};
    use crate::math::collision::linear_bvh::LinearBound;
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    let bvh = comet.bvh.as_ref()?;
    let com = Vec3f32::from_components(
      comet.mass_properties.center_of_mass[0] as f32,
      comet.mass_properties.center_of_mass[1] as f32,
      comet.mass_properties.center_of_mass[2] as f32,
    );

    let mut dir = target_point_local - com;
    if dir.length_squared() < 1e-6 {
      dir = Vec3f32::from_components(1.0, 0.0, 0.0);
    }
    dir = dir.normalize();

    let rays = [
      Ray { origin: com, direction: dir, length: f32::MAX },
      Ray { origin: com, direction: -dir, length: f32::MAX },
    ];

    let mut closest_dist_sq = f32::MAX;
    let mut best_uv = None;

    for ray in &rays {
      let mut stack = alloc::vec::Vec::new();
      if !bvh.nodes.is_empty() {
        stack.push(0);
      }

      while let Some(node_idx) = stack.pop() {
        let local_node = &bvh.nodes[node_idx];

        let hit_local_node = match &local_node.bound {
          LinearBound::AABB(aabb) => intersection::intersect_ray_aabb(ray, aabb),
          LinearBound::OBB(obb) => intersection::intersect_ray_obb::<_, _, Mat3f32>(ray, obb),
        };

        if hit_local_node {
          if local_node.primitive_count > 0 {
            let prim_start = local_node.left_child_or_primitive_offset as usize;
            let prim_end = prim_start + local_node.primitive_count as usize;

            for j in prim_start..prim_end {
              let tri_idx = bvh.primitives[j];
              let i0 = comet.indices[tri_idx * 3] as usize;
              let i1 = comet.indices[tri_idx * 3 + 1] as usize;
              let i2 = comet.indices[tri_idx * 3 + 2] as usize;

              let v0 = &comet.vertices[i0];
              let v1 = &comet.vertices[i1];
              let v2 = &comet.vertices[i2];

              let p0 = Vec3f32::from_array(v0.position);
              let p1 = Vec3f32::from_array(v1.position);
              let p2 = Vec3f32::from_array(v2.position);

              let edge1 = p1 - p0;
              let edge2 = p2 - p0;
              let h = ray.direction.cross(edge2);
              let a = edge1.dot(h);

              if a > -1e-6 && a < 1e-6 { continue; }

              let f = 1.0 / a;
              let s = ray.origin - p0;
              let u = f * s.dot(h);
              if u < 0.0 || u > 1.0 { continue; }

              let q = s.cross(edge1);
              let v = f * ray.direction.dot(q);
              if v < 0.0 || u + v > 1.0 { continue; }

              let t = f * edge2.dot(q);
              if t > 1e-5 {
                let hit_pos = ray.origin + ray.direction * t;
                let dist_sq = (hit_pos - target_point_local).length_squared();

                if dist_sq < closest_dist_sq {
                  closest_dist_sq = dist_sq;
                  let w = 1.0 - u - v;
                  best_uv = Some([
                    v0.uv[0] * w + v1.uv[0] * u + v2.uv[0] * v,
                    v0.uv[1] * w + v1.uv[1] * u + v2.uv[1] * v,
                  ]);
                }
              }
            }
          } else {
            if local_node.right_child_offset != u32::MAX {
              stack.push(local_node.right_child_offset as usize);
            }
            if local_node.left_child_or_primitive_offset != u32::MAX {
              stack.push(local_node.left_child_or_primitive_offset as usize);
            }
          }
        }
      }
    }

    if let Some(uv) = best_uv {
      config.uv_center = uv;
      Some(config)
    } else {
      None
    }
  }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct ParticleData {
  pub id_low: u32,
  pub id_high: u32,
  pub _pad0: [u32; 2],
  pub position: [f32; 3],
  pub _pad1: u32,
  pub velocity: [f32; 3],
  pub age_low: u32,
  pub age_high: u32,
  pub mass: f32,
  pub active: u32,
  pub _pad2: u32,
}

impl ParticleData {
  pub fn as_particle(&self, radius: f32) -> Particle<Vec3f32> {
    Particle {
      position: Vec3f32::from_array(self.position),
      radius,
    }
  }

  pub fn set_id(&mut self, id: u64) {
    self.id_low = (id & 0xFFFFFFFF) as u32;
    self.id_high = (id >> 32) as u32;
  }

  pub fn get_id(&self) -> u64 {
    (self.id_low as u64) | ((self.id_high as u64) << 32)
  }

  pub fn set_age(&mut self, age: timeus_t) {
    self.age_low = (age as u64 & 0xFFFFFFFF) as u32;
    self.age_high = ((age as u64) >> 32) as u32;
  }

  pub fn get_age(&self) -> timeus_t {
    ((self.age_low as u64) | ((self.age_high as u64) << 32)) as timeus_t
  }
}

pub struct ParticleSystemComponent {
  pub config: ParticleEmitterConfig,
  pub particles: alloc::vec::Vec<ParticleData>,
  pub bvh: Option<LinearBVH<f32>>,
  pub accumulator: timeus_t,
  pub next_id: usize,
}

impl core::fmt::Debug for ParticleSystemComponent {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ParticleSystemComponent")
     .field("config", &self.config)
     .field("particles_count", &self.particles.len())
     .field("bvh_is_some", &self.bvh.is_some())
     .finish()
  }
}

impl Component for ParticleSystemComponent {}

impl ParticleSystemComponent {
  pub fn new(config: ParticleEmitterConfig) -> Self {
    Self {
      particles: alloc::vec::Vec::with_capacity(config.max_particles),
      config,
      bvh: None,
      accumulator: 0,
      next_id: 0,
    }
  }

  pub fn emit_particles(
      &mut self,
      comet: &Comet,
      uv_grid: &UvGrid,
      comet_pos: Vec3f32,
      comet_rot: aethervk_oshal_rlib::math::vector::vec4::Quat,
      u_emission: &[f32; 2],
      u_particles: &[[f32; 4]],
  ) {
    let count = self.config.emission_count.sample(u_emission) as usize;
    let actual_count = count.min(u_particles.len());

    for i in 0..actual_count {
      if self.particles.len() >= self.config.max_particles { break; }
      
      let u = &u_particles[i];
      let theta = 2.0 * core::f32::consts::PI * u[0];
      let r = self.config.uv_radius * u[1].sqrt();
      let uv_x = self.config.uv_center[0] + r * theta.cos();
      let uv_y = self.config.uv_center[1] + r * theta.sin();

      let (local_pos, local_norm) = uv_grid.query([uv_x, uv_y], &comet.vertices, &comet.indices)
          .unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]));
          
      // Convert to world space
      let local_pos_vec = Vec3f32::from_components(local_pos[0], local_pos[1], local_pos[2]);
      let local_norm_vec = Vec3f32::from_components(local_norm[0], local_norm[1], local_norm[2]);
      let local_norm_vec = if local_norm_vec.length_squared() > 0.0001 { local_norm_vec.normalize() } else { Vec3f32::from_components(0.0, 0.0, 1.0) };

      use aethervk_oshal_rlib::math::quaternion::Quaternion;
      
      let world_norm = comet_rot.rotate_vector(local_norm_vec);
      let world_pos = comet_pos + comet_rot.rotate_vector(local_pos_vec) + world_norm * self.config.particle_radius;

      // Check intersection with other alive particles to avoid generating overlapping particles
      let mut intersecting = false;
      for p in self.particles.iter() {
        if p.active != 0 {
           let p_pos = Vec3f32::from_array(p.position);
           let dist_sq = (p_pos - world_pos).length_squared();           let min_dist = self.config.particle_radius * 2.0;
           if dist_sq < min_dist * min_dist {
               intersecting = true;
               break;
           }
        }
      }
      if intersecting {
          continue;
      }

      // Velocity in cosine hemisphere
      let phi = 2.0 * core::f32::consts::PI * u[2];
      let cos_theta = u[3].sqrt();
      let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
      let local_dir = Vec3f32::from_components(
        sin_theta * phi.cos(),
        sin_theta * phi.sin(),
        cos_theta
      );
      
      let mut tangent = Vec3f32::from_components(1.0, 0.0, 0.0);
      if world_norm.dot(tangent).abs() > 0.99 {
        tangent = Vec3f32::from_components(0.0, 1.0, 0.0);
      }
      let bitangent = world_norm.cross(tangent).normalize();
      tangent = bitangent.cross(world_norm).normalize();

      let world_dir = (tangent * local_dir.x() + bitangent * local_dir.y() + world_norm * local_dir.z()).normalize();

      // For the intensity, we reuse the first two random numbers to sample the Gaussian
      let intensity = self.config.velocity_intensity.sample(&[u[0], u[1]]);
      let velocity = world_dir * intensity;

      let mut p = ParticleData {
        id_low: 0,
        id_high: 0,
        _pad0: [0; 2],
        position: [world_pos.x(), world_pos.y(), world_pos.z()],
        _pad1: 0,
        velocity: [velocity.x(), velocity.y(), velocity.z()],
        age_low: 0,
        age_high: 0,
        mass: self.config.density * (4.0 / 3.0) * core::f32::consts::PI * self.config.particle_radius.powi(3),
        active: 1,
        _pad2: 0,
      };
      p.set_id(self.next_id as u64);
      p.set_age(0);
      self.particles.push(p);
      self.next_id += 1;
    }
  }

  pub fn update_bvh(&mut self) {
    use crate::physics::particle::ParticleBVHBuilder;
    use crate::math::collision::bvh_builder::BVHBuilderParams;
    use crate::math::collision::linear_bvh::LinearBVH;
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    let active_particles: alloc::vec::Vec<_> = self.particles.iter()
      .filter(|p| p.active != 0)
      .map(|p| p.as_particle(self.config.particle_radius))
      .collect();

    if active_particles.is_empty() {
      self.bvh = None;
      return;
    }

    let builder = ParticleBVHBuilder::new(BVHBuilderParams::default());
    if let Some(root) = builder.build::<_, _, Mat3f32>(&active_particles) {
      self.bvh = Some(LinearBVH::from_build_node(&root, 0));
    } else {
      self.bvh = None;
    }
  }
}
