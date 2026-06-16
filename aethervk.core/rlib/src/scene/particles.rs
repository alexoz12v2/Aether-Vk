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
  /// End-of-frame force accumulator (km/s² units). Persisted across simulation ticks
  /// so the Velocity-Verlet predictor step in `integrate_particles_p1_p2.comp`
  /// can use F(x_n) from the previous frame rather than a zero vector.
  pub force: [f32; 3],
  pub padding: u32,
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
  /// Radius used for billboard rendering only (km). Defaults to particle_radius.
  /// Decoupled so very small physics particles can still be visible on screen.
  pub render_radius_km: f32,
  /// RGBA color used by the particle shader.
  pub color: [f32; 4],
  /// Time-to-live in microseconds. Particles older than this are reaped.
  /// A value of 0 means particles never expire.
  pub ttl_us: timeus_t,
  /// Radiation pressure coefficient (dimensionless). ~1.0 for a perfect absorber;
  /// ~2.0 for a perfect reflector. Propagated from the parent `EmissionCircle.beta`.
  pub beta: f32,
  /// GPU-side sort permutation from last frame's BVH build.
  /// `gpu_sort_order[gpu_slot] = original_dense_metadata_idx` for this system's particles.
  /// Refreshed each frame by `write_back_to_scene`. Invalidated on particle birth/death
  /// (safe: always rewritten before being consumed for write-back).
  pub gpu_sort_order: alloc::vec::Vec<u32>,
  /// Number of active particles for this system as of the last GPU upload.
  /// Refreshed each frame by `write_back_to_scene`.
  pub gpu_alive_count: u32,
  /// When `true`, the GPU physics pipeline skips BVH construction and Barnes-Hut
  /// self-gravity for this system.  Use for test-particle systems (comet dust tails,
  /// tracers) where inter-particle gravity is physically negligible and the BVH
  /// build/traversal cost would otherwise dominate frame time at high particle counts.
  pub disable_self_gravity: bool,
  /// Fractional particle carry-over from the dt-based emission rate calculation.
  /// Each tick: `exact = particles_per_second * dt_s + emit_remainder`;
  /// `emit_count = floor(exact)`; `emit_remainder = exact - emit_count`.
  /// This ensures the long-run emission rate matches `particles_per_second` exactly.
  pub emit_remainder: f32,
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
    let mut particles = alloc::vec::Vec::with_capacity(max_particles);

    // FIX 1: Prevent crash at startup by properly advancing the vector's length to match capacity.
    // We zero-initialize memory to avoid needing `Default` bounds on `ParticleData`.
    if max_particles > 0 {
      unsafe {
        core::ptr::write_bytes(particles.as_mut_ptr(), 0, max_particles);
        particles.set_len(max_particles);
      }
    }

    Self {
      particles: alloc::sync::Arc::new(parking_lot::RwLock::new(particles)),
      head_index: 0,
      tail_index: 0,
      capacity: max_particles,
      bvh: None,
      accumulator: 0,
      next_id: 0,
      particle_radius: 0.01,       // 10 m default
      render_radius_km: 1.0,        // 1 km default — visible at typical comet-approach distances
      color: [1.0, 1.0, 1.0, 1.0], // white default
      ttl_us: 0,                   // 0 = never expire (set from EmissionCircle.ttl)
      beta: 0.0,                   // set from EmissionCircle.beta
      gpu_sort_order: alloc::vec::Vec::new(),
      gpu_alive_count: 0,
      disable_self_gravity: false,
      emit_remainder: 0.0,
    }
  }

  /// TODO: Document this item
  pub fn update_bvh(&mut self, particle_radius: f32) {
    aethervk_oshal_rlib::log!("WARNING: NEVER CALL THIS. USE COMPUTE SHADER");
    use crate::{
      math::collision::{bvh_builder::BVHBuilderParams, linear_bvh::LinearBVH},
      physics::particle::ParticleBVHBuilder,
    };
    use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;

    let particles = self.particles.read();
    let capacity = self.capacity;

    let mut active_particles = alloc::vec::Vec::new();

    // FIX 2 (Optimization): Iterate only the active ring-buffer window rather than all capacity blocks
    if capacity > 0 && self.tail_index > self.head_index {
      active_particles.reserve(self.tail_index - self.head_index);
      for idx in self.head_index..self.tail_index {
        let p = &particles[idx % capacity];
        if p.active != 0 {
          active_particles.push(p.as_particle(particle_radius));
        }
      }
    }

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
  /// Emission rate in particles per second (dt-independent).
  /// The emit loop accumulates a fractional count each tick and flushes
  /// whole particles, so the actual rate is `particles_per_second * dt_s`
  /// rounded to the nearest integer over time.
  pub particles_per_second: f32,
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
  /// Radius of the spawn disc in the surface tangent plane (km).
  /// Each particle's initial position is offset by a uniformly-sampled
  /// random vector within this radius, breaking up the point-source pattern.
  /// Set to 0 for a pure point source.
  pub spawn_radius_km: f32,
  /// Visual billboard radius for rendering (km). Controls particle size on screen.
  /// Physics uses `particle_radius` separately. Default 0.01 km = 10 m.
  pub render_radius_km: f32,
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
      particles_per_second: 10.0,
      ttl: 1000,
      mean_velocity: 0.1,
      velocity_std_dev: 0.05,
      child_entity: None,
      beta: 1.0,
      max_particles: 8192,
      spawn_radius_km: 0.0,
      render_radius_km: 0.01,
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
