//! particles module.

use crate::{
  math::collision::linear_bvh::LinearBVH,
  physics::particle::Particle,
  scene::Component,
  simulation::comet::{Comet, uv_grid::UvGrid},
};
use aethervk_oshal_rlib::{
  math::vector::{Vector, Vector3, vec3::Vec3f32},
  os::time::timeus_t,
};

#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct GaussianParams {
  pub mean: f32,
  pub std_dev: f32,
  pub min: f32,
  pub max: f32,
}

impl GaussianParams {
  /// TODO: Document this item
  pub fn sample(&self, u: &[f32; 2]) -> f32 {
    let r = (-2.0 * u[0].max(1e-8).ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u[1];
    let z0 = r * theta.cos();
    (self.mean + self.std_dev * z0).clamp(self.min, self.max)
  }
}

#[deprecated]
#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct ParticleEmitterComponent {
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
  pub use_particle2: bool,
}

impl Component for ParticleEmitterComponent {}

#[repr(C)]
#[derive(Clone, Debug)]
/// TODO: Document this item
pub struct ParticleData {
  pub id_low: u32,
  pub id_high: u32,
  pub age_low: u32,
  pub age_high: u32,
  pub position: [f32; 3],
  pub mass: f32,
  pub velocity: [f32; 3],
  pub active: u32,
}

impl ParticleData {
  /// TODO: Document this item
  pub fn as_particle(&self, radius: f32) -> Particle<Vec3f32> {
    Particle {
      position: Vec3f32::from_array(self.position),
      radius,
    }
  }

  /// TODO: Document this item
  pub fn set_id(&mut self, id: u64) {
    self.id_low = (id & 0xFFFFFFFF) as u32;
    self.id_high = (id >> 32) as u32;
  }

  /// TODO: Document this item
  pub fn get_id(&self) -> u64 {
    (self.id_low as u64) | ((self.id_high as u64) << 32)
  }

  /// TODO: Document this item
  pub fn set_age(&mut self, age: timeus_t) {
    self.age_low = (age as u64 & 0xFFFFFFFF) as u32;
    self.age_high = ((age as u64) >> 32) as u32;
  }

  /// TODO: Document this item
  pub fn get_age(&self) -> timeus_t {
    ((self.age_low as u64) | ((self.age_high as u64) << 32)) as timeus_t
  }
}

/// Per-entity particle system state.
///
/// Stores the CPU-side particle buffer and render configuration.  Each tick
/// the buffer is uploaded to GPU by `build_particles`, integrated by the IMEX
/// pipeline, and written back by `write_back_to_scene`.
#[derive(Clone)]
pub struct ParticleSystemComponent {
  pub particles: alloc::sync::Arc<parking_lot::RwLock<alloc::vec::Vec<ParticleData>>>,
  pub head_index: usize,
  pub tail_index: usize,
  pub capacity: usize,
  pub bvh: Option<LinearBVH<f32>>,
  pub accumulator: timeus_t,
  pub next_id: usize,
  /// Radius of each particle for billboard rendering (km).
  pub particle_radius: f32,
  /// RGBA color used by the particle shader.
  pub color: [f32; 4],
  /// Time-to-live in microseconds. Particles older than this are reaped.
  /// A value of 0 means particles never expire.
  pub ttl_us: timeus_t,
  /// Radiation pressure coefficient (dimensionless). ~1.0 for a perfect absorber;
  /// ~2.0 for a perfect reflector. Propagated from the parent `EmissionCircle.beta`.
  pub beta: f32,
}

impl core::fmt::Debug for ParticleSystemComponent {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ParticleSystemComponent")
      .field("particles_count", &self.particles.read().len())
      .field("bvh_is_some", &self.bvh.is_some())
      .field("particle_radius", &self.particle_radius)
      .field("ttl_us", &self.ttl_us)
      .field("beta", &self.beta)
      .finish()
  }
}

impl Component for ParticleSystemComponent {}

impl ParticleSystemComponent {
  /// Create a new particle system with the given maximum capacity.
  pub fn new(max_particles: usize) -> Self {
    Self {
      particles: alloc::sync::Arc::new(parking_lot::RwLock::new(alloc::vec::Vec::with_capacity(
        max_particles,
      ))),
      head_index: 0,
      tail_index: 0,
      capacity: max_particles,
      bvh: None,
      accumulator: 0,
      next_id: 0,
      particle_radius: 0.01,       // 10 m default
      color: [1.0, 1.0, 1.0, 1.0], // white default
      ttl_us: 0,                   // 0 = never expire (set from EmissionCircle.ttl)
      beta: 0.0,                   // set from EmissionCircle.beta
    }
  }

  /// Deprecated. Use the compute shader for this.
  #[deprecated]
  pub fn emit_particles(
    &mut self,
    config: &ParticleEmitterComponent,
    comet: &Comet,
    uv_grid: &UvGrid,
    comet_pos: Vec3f32,
    comet_rot: aethervk_oshal_rlib::math::vector::vec4::Quat,
    comet_scale: Vec3f32,
    u_emission: &[f32; 2],
    u_particles: &[[f32; 4]],
  ) {
    let count = config.emission_count.sample(u_emission) as usize;

    let mut particles = self.particles.write(); // TODO this might be slow
    for i in 0..count {
      if particles.len() >= config.max_particles {
        break;
      }

      let u = &u_particles[i];
      // TODO: why is pdf unused?
      let (uv_x, uv_y, _pdf) = config.uv_distribution.sample_continuous(&[u[0], u[1]]);

      let (local_pos, local_norm) =
        match uv_grid.query([uv_x, uv_y], &comet.vertices, &comet.indices) {
          Some(res) => res,
          None => continue,
        };

      // Convert to world space
      let local_pos_vec: Vec3f32 = local_pos.into();
      let local_pos_vec = Vec3f32::from_components(
        local_pos_vec.x() * comet_scale.x(),
        local_pos_vec.y() * comet_scale.y(),
        local_pos_vec.z() * comet_scale.z(),
      );
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
        + world_norm * config.particle_radius * 1.5; // Push slightly outside

      // Intersection check removed for performance

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
      let intensity = config.velocity_intensity.sample(&[u[0], u[1]]);
      let velocity = world_dir * intensity;

      let mut p = ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [world_pos.x(), world_pos.y(), world_pos.z()],
        mass: config.density * (4.0 / 3.0) * core::f32::consts::PI * config.particle_radius.powi(3),
        velocity: [velocity.x(), velocity.y(), velocity.z()],
        active: 1,
      };
      p.set_id(self.next_id as u64);
      p.set_age(0);
      particles.push(p);
      self.next_id += 1;
    }
  }

  /// TODO: Document this item
  pub fn update_bvh(&mut self, particle_radius: f32) {
    use crate::{
      math::collision::{bvh_builder::BVHBuilderParams, linear_bvh::LinearBVH},
      physics::particle::ParticleBVHBuilder,
    };
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    let active_particles: alloc::vec::Vec<_> = self
      .particles
      .read()
      .iter()
      .filter(|p| p.active != 0)
      .map(|p| p.as_particle(particle_radius))
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

// ─────────────────────────────────────────────────────────────────────────────
// Comet circle-based particle emitter
// ─────────────────────────────────────────────────────────────────────────────

/// Defines a single circular emission zone on the surface of a comet mesh.
///
/// Angles are stored in **radians** (the UI sends degrees and converts before FFI).
/// The emission circle is centred at the point obtained by rotating `+Z` of the
/// comet's local frame by the spherical angles (latitude / longitude).
#[derive(Clone, Debug)]
pub struct EmissionCircle {
  /// Latitude in radians: −π/2 (south pole) … +π/2 (north pole).
  pub latitude_rad: f32,
  /// Longitude in radians: 0 … 2π, measured in the LCA micro-frame.
  pub longitude_rad: f32,
  /// Radius of the emission disc in km.
  pub circle_radius_km: f32,
  /// Mass of particles emitted from this circle (simulation units).
  pub mass: f32,
  /// RGBA colour of particles emitted from this circle (linear, 0–1).
  pub color: [f32; 4],
  /// Cached object-space emission point
  pub cached_point: Option<[f32; 3]>,
  /// Cached object-space normal
  pub cached_normal: Option<[f32; 3]>,
  /// Number of particles to emit per physics tick.
  pub particles_per_tick: u32,
  /// Time to live for emitted particles (in simulation ticks or microseconds depending on context).
  pub ttl: u64,
  /// Mean initial velocity of the emitted particles (simulation units).
  pub mean_velocity: f32,
  /// Standard deviation for the velocity direction (radians).
  pub velocity_std_dev: f32,
  /// Visual child entity representing the emission point.
  pub child_entity: Option<crate::scene::EntityId>,
  /// Radiation pressure coefficient (dimensionless). ~1.0 for a perfect absorber;
  /// ~2.0 for a perfect reflector. Used by the Barnes-Hut radiation pressure kernel.
  pub beta: f32,
  /// Maximum number of particles this jet can have alive simultaneously.
  /// Controls the capacity of the child entity's `ParticleSystemComponent` buffer.
  pub max_particles: u32,
}

/// Attaches a set of discrete circular emission zones to a comet mesh entity.
///
/// At simulation time, each `EmissionCircle` drives a localised particle jet
/// whose direction is the outward surface normal at the circle's centre.
#[derive(Clone, Debug, Default)]
pub struct ParticleEmitterCirclesComponent {
  pub circles: alloc::vec::Vec<EmissionCircle>,
}

impl Component for ParticleEmitterCirclesComponent {}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_particle_emitter_circles_component_default() {
    let comp = ParticleEmitterCirclesComponent::default();
    assert!(comp.circles.is_empty());
  }

  #[test]
  fn test_particle_emitter_circles_component_add() {
    let mut comp = ParticleEmitterCirclesComponent::default();
    comp.circles.push(EmissionCircle {
      latitude_rad: 0.5,
      longitude_rad: 1.0,
      circle_radius_km: 0.1,
      mass: 1.0,
      color: [1.0, 1.0, 1.0, 1.0],
      cached_normal: Some([0.0, 1.0, 0.0]),
      cached_point: Some([0.0, 0.0, 0.0]),
      particles_per_tick: 10,
      ttl: 1000,
      mean_velocity: 0.1,
      velocity_std_dev: 0.05,
      child_entity: None,
      beta: 1.0,
      max_particles: 4096,
    });
    assert_eq!(comp.circles.len(), 1);
    assert_eq!(comp.circles[0].latitude_rad, 0.5);
  }
}

#[derive(Clone, Debug, Default)]
pub struct JetComponent {
  pub radius: f32,
  pub lat: f32,
  pub lon: f32,
  pub color: [f32; 4],
  pub mass: f32,
  pub particles_per_tick: u32,
  pub ttl: f32,
  pub mean_velocity: f32,
  pub cached_emission_points: alloc::vec::Vec<aethervk_oshal_rlib::math::vector::vec3::Vec3f32>,
}

impl Component for JetComponent {}
