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
  pub uv_distribution: crate::math::distribution::Distribution2D,
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
      if self.particles.len() >= self.config.max_particles {
        break;
      }

      let u = &u_particles[i];
      // TODO: why is pdf unused?
      let (uv_x, uv_y, _pdf) = self.config.uv_distribution.sample_continuous(&[u[0], u[1]]);

      let (local_pos, local_norm) = uv_grid
        .query([uv_x, uv_y], &comet.vertices, &comet.indices)
        .unwrap_or(([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]));

      // Convert to world space
      let local_pos_vec: Vec3f32 = local_pos.into();
      let local_norm_vec: Vec3f32 = local_norm.into();
      let local_norm_vec = if local_norm_vec.length_squared() > 0.0001 {
        local_norm_vec.normalize()
      } else {
        Vec3f32::zero()
      };

      use aethervk_oshal_rlib::math::quaternion::Quaternion;

      let world_norm = comet_rot.rotate_vector(local_norm_vec);
      let world_pos = comet_pos
        + comet_rot.rotate_vector(local_pos_vec)
        + world_norm * self.config.particle_radius * 1.5; // Push slightly outside

      // Check intersection with other alive particles to avoid generating overlapping particles
      let mut intersecting = false;
      for p in self.particles.iter() {
        if p.active != 0 {
          let p_pos = Vec3f32::from_array(p.position);
          let dist_sq = (p_pos - world_pos).length_squared();
          let min_dist = self.config.particle_radius * 2.0;
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
      let local_dir =
        Vec3f32::from_components(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);

      let mut tangent = Vec3f32::from_components(1.0, 0.0, 0.0);
      if world_norm.dot(tangent).abs() > 0.99 {
        tangent = Vec3f32::from_components(0.0, 1.0, 0.0);
      }
      let bitangent = world_norm.cross(tangent).normalize();
      tangent = bitangent.cross(world_norm).normalize();

      let world_dir =
        (tangent * local_dir.x() + bitangent * local_dir.y() + world_norm * local_dir.z())
          .normalize();

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
        mass: self.config.density
          * (4.0 / 3.0)
          * core::f32::consts::PI
          * self.config.particle_radius.powi(3),
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

    let active_particles: alloc::vec::Vec<_> = self
      .particles
      .iter()
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
