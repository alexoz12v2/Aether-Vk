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
      render_radius_km: 1.0,       // 1 km default — visible at typical comet-approach distances
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

/// Note: tight coupling with vulkan here
pub mod v2 {
  use aethervk_oshal_rlib::os::time::us_to_300ths_rounded;

  use super::*;
  use crate::gpu::compute_push_constants::NewParticlesEmitPushConstants;
  use crate::gpu_backends::vulkan;
  use crate::scene::EntityId;
  use crate::types::{EngineError, EngineResult};
  use core::sync::atomic::{AtomicI64, AtomicU64, Ordering};

  /// Computes some parameters for emit shader. Left to fill are
  /// - `global_particle_buffer`
  /// - `particle_page_table`
  /// - `free_list`
  pub fn emit_push_constants_from_params(
    emit_params: &ParticleSystemEmitParams,
    mean_intra_grains_distance_mm: f32,
    min_cumulated_mass_g: f32,
    r_helio_au: f32,
    scaled_time_since_last_emission_us: timeus_t,
    scaled_time_since_start_epoch_us: timeus_t,
  ) -> NewParticlesEmitPushConstants {
    use bytemuck::Zeroable;
    // TODO when writing loop, this will be moved, cause we'll have these available already
    let cluster_params =
      emit_params.cluster_params(mean_intra_grains_distance_mm, min_cumulated_mass_g);
    let beta = emit_params.beta();
    let dust_production_rate_kgs = emit_params.dust_production_rate_kgs(r_helio_au);
    let emit_count =
      emit_params.emission_count(dust_production_rate_kgs, scaled_time_since_last_emission_us);
    let current_time = us_to_300ths_rounded(scaled_time_since_start_epoch_us);
    let velocity_dir = emit_params.particle_system_relative_cone_direction();

    let mut push_constants = NewParticlesEmitPushConstants::zeroed();
    push_constants.cone_dir_aperture = [
      velocity_dir.x(),
      velocity_dir.y(),
      velocity_dir.z(),
      emit_params.aperture_rad,
    ];
    push_constants.mass_vel_mean_std = [
      cluster_params.mass_g,
      cluster_params.mass_std(emit_params.mass_variability_perc),
      emit_params.start_velocity_mean,
      emit_params.start_velocity_std,
    ];
    push_constants.emit_count = emit_count;
    push_constants.current_time = current_time;
    push_constants.seed = emit_params.seed;
    push_constants.radius = cluster_params.radius_m;
    push_constants.beta = beta;

    push_constants
  }

  // TODO C# side: copy paste from a jet to another, godot edition has params shared
  #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable)]
  pub struct ParticleSystemEmitParams {
    /// radians -π/2 to π/2, relative to particle system entity frame
    pub latitude_rad: f32,
    /// radians -π to π, relative to particle system entity frame
    pub longitude_rad: f32,
    /// aperture of the cone of directions from which starting velocity direction is uniformly
    /// sampled from
    pub aperture_rad: f32,
    /// m/s mean μ of the normally distributed initial velocity
    pub start_velocity_mean: f32,
    /// m/s std σ of the normally distributed initial velocity
    pub start_velocity_std: f32,
    /// percentage from 0 to 1 about mass variability. 100% (1) means std = mean in a normal
    /// distribution
    pub mass_variability_perc: f32,
    /// random number generation (PCG) initial seed
    pub seed: u32,
    /// radius of a *single* dust grain in μm
    pub diametre_um: f32,
    /// from 1 to 7, in g/cm3, volume density
    pub density_gcm3: f32,
    /// `Q_pr`, value between 0.5 to 2
    pub scattering_efficiency: f32,
    /// "Af-rho" photometric parameter. measures comet's dust activity, equal to product of - dust albedo - filling factor - size of viewing area .
    ///
    /// It is dependant on the
    /// sun-comet nucleus distance and is proportional to the dust production rate in kg/s.
    /// Here we report afrho parameter at distance of 1 AU, which we compose to a cutoff distance
    /// and max value with a power curve of exponent `k`
    /// Note: datasets like `locdprod` of https://pdssbn.astro.umd.edu/holdings/ear-c-phot-3-rdr-lowell-comet-db-v1.0/dataset.shtml
    /// measures its logarithm base 10 with blue frequencies albedo and UV frequencies albedo. (we want
    /// blue one. We are not registering logarithm, but base value here)
    /// Unit of measurement: cm
    /// Large range, from ~50 cm to ~250_000 cm
    /// Example from database:
    ///
    /// | activity level   | comet                 | year | value          | distance | source                                                                                        |
    /// | ---------------- | --------------------- | ---- | -------------- | -------- | --------------------------------------------------------------------------------------------- |
    /// | low activity     | 2P/Encke              | 1984 | 8.128 cm       | 0.913 AU | https://pdssbn.astro.umd.edu/holdings/ear-c-phot-3-rdr-lowell-comet-db-v1.0/data/locdprod.tab |
    /// | medium activity  | 1P/Halley             | 2007 | 23.94 cm       | 2 AU     | https://www.aanda.org/articles/aa/full_html/2013/09/aa22020-13/aa22020-13.html                |
    /// | high activity    | 47P/Ashbroook-Jackson | 1978 | 537.03 cm      | 2.288 AU | https://pdssbn.astro.umd.edu/holdings/ear-c-phot-3-rdr-lowell-comet-db-v1.0/data/locdprod.tab |
    /// | extreme activity | Hale-Bopp             | 1995 | 1,000,000 cm   | 0.914 AU | https://cara.uai.it/measuring-comets                                                          |
    // TODO C# side: function to fit in these 4 params a measurement from a given distance
    pub afrho_0_cm: f32,
    /// "Af-rho" parameter power curve exponent. from 1.0 to 4.0 (default 2.0, quadratic decay)
    pub afrho_power: f32,
    /// "Af-rho" distance parameter, over which, computed afro is dropped to zero, nullifying
    /// particle emission. eg. 5 AU or higher. Not lower than 3 AU
    pub afrho_cutoff_au: f32,
    /// "Af-rho" maxium value, so that when distance is really low, emission doesn't get infinitely
    /// large values. Unit: cm
    /// Example value: 100_000 cm
    pub afrho_max_value_cm: f32,
  }

  impl ParticleSystemEmitParams {
    /// computes the Finson-Probstein model ratio between radiation pressure force and sun
    /// gravitation
    pub fn beta(&self) -> f32 {
      // note: f32 range: 1.1754934 x 10^-38 to 3.402823 x 10^38 with 7.2 decimal digits
      // Use f64 for compile-time calculation to prevent precision loss
      const SUN_IRRADIANCE: f64 = 3.828e26; // in W
      const LIGHT_VELOCITY: f64 = 299_792_458.0; // in m/s
      const SUN_GM: f64 = 1.3271244e20; // in m3/s2
      const PI: f64 = core::f64::consts::PI;
      const COMPOSED_CONSTANT: f32 =
        ((3.0 * SUN_IRRADIANCE) / (16.0 * PI * LIGHT_VELOCITY * SUN_GM)) as f32;

      // 1 g cm-3 = 10^-4 g cm-2 μm-1
      let radius_um = self.diametre_um * 0.5f32;
      let density_x_radius_gcm2 = 1e-4_f32 * radius_um * self.density_gcm3;
      // 1 g cm^-2 = 10 kg m^-2
      let density_x_radius_kgm2 = density_x_radius_gcm2 * 10_f32;

      COMPOSED_CONSTANT * (self.scattering_efficiency / density_x_radius_kgm2)
    }

    /// `latitude_rad` and `longitude_rad` store the positioning of a dust jet relative to
    /// comet nucleus orientation. Since particle system is supposed to be a child entity of
    /// comet nucleus entity, with transform equal as the translation to a point on the bounding
    /// sphere, with identity as rotation component, we can convert lat,lon into a unit vector
    /// without any other additional data
    pub fn particle_system_relative_cone_direction(&self) -> Vec3f32 {
      use aethervk_oshal_rlib::math::FloatLike;
      let cos_lat = <f32 as FloatLike>::cos(self.latitude_rad);
      let sin_lat = <f32 as FloatLike>::sin(self.latitude_rad);
      let cos_lon = <f32 as FloatLike>::cos(self.longitude_rad);
      let sin_lon = <f32 as FloatLike>::sin(self.longitude_rad);
      Vec3f32::from_components(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    }

    /// pass dust production rate as a parameter so user can cache it if needed
    pub fn emission_count(
      &self,
      q_dust_kgs: f32,
      scaled_time_since_last_emission_us: timeus_t,
    ) -> u32 {
      use aethervk_oshal_rlib::math::FloatLike;
      // 4/3 * PI * r^3 * density
      // done in f64 to avoid loss for very small values
      let radius_cube_m3 = {
        let x = (self.diametre_um as f64 / 2.0) * 1e-6;
        x * x * x
      };
      let single_grain_mass_kg: f64 =
        (4.0 / 3.0) * core::f64::consts::PI * radius_cube_m3 * (self.density_gcm3 as f64 * 1e3);

      // i64 as time can store up to ~292271 years, so can't overflow. it is safe to cast timeus_t
      // to f64 only if it is less than 2^53, which is ~285.4 years. should be always true.
      #[cfg(debug_assertions)]
      {
        const MAX_SAFE_TIMEUS: i64 = 1_i64 << 53;
        debug_assert!(scaled_time_since_last_emission_us <= MAX_SAFE_TIMEUS);
      }
      let scaled_delta_time_since_last_emission_s =
        scaled_time_since_last_emission_us as f64 * 1e-6;

      let particles_this_tick =
        q_dust_kgs as f64 / single_grain_mass_kg * scaled_delta_time_since_last_emission_s;

      <f64 as FloatLike>::floor(particles_this_tick) as u32
    }

    /// Starting from dataset observation "Lowell Observatory Cometary Database" from NASA PDS
    /// (Planetary Data System) Small Bodies Node. Link: https://pdssbn.astro.umd.edu/holdings/ear-c-phot-3-rdr-lowell-comet-db-v1.0/dataset.shtml
    /// It contains various parameters for ~100 comets, including $A f \rho$, which has to do with
    /// comet dust production. Explaination of the `afrho` parameter: https://cara.uai.it/measuring-comets
    /// inside the file `locdprod` we have LOG_AFRHO_B_CONT (afrho for blue continuum), UV
    /// continuum is contaminate by gas emissions and therefore less reliable for dust production
    /// rate estimation
    /// Source Papers (Note: $Q_d$ is the unknown dust production rate)
    /// 1. Agarwal, J., Müller, M., & Grün, E. (2010). "Dust Environment Modelling of Comet 67P/Churyumov-Gerasimenko."
    ///    PDF Link: https://arxiv.org/pdf/1001.3010
    ///    Where to look: See Equation 2 on page 8. The paper defines $Afρ$ as a
    ///    function of the dust production rate $Q_{d,j}$,grain size $s_j$, and velocity $v_{d,j}$.
    ///    If you substitute the mass of a spherical grain ($m=3/4 π s^3 ρ$) into their Equation 2
    ///    and solve for $Q_d$, it yields 2/3
    /// 2. Cremonese, G. et al. (2020) / Aravind, K. (2022). "Observational analysis of Cometary bodies in the Solar System" (PhD Thesis, Physical Research Laboratory).
    ///    PDF Link: https://www.prl.res.in/~library/gpdf/prl-theses/aravind_k_2022.pdf
    ///    Where to look: See equation 7.4 on page 167. It explicitly states the rearranged
    ///    relation used to mod3el intersetellar comet 2I/Borisov: $ Afρ = 3 A Q_d / ( 2 ρ_d v_d s_0 )$
    ///    using $s_0$ as the radius
    pub fn dust_production_rate_kgs(&self, r_helio_au: f32) -> f32 {
      use aethervk_oshal_rlib::math::FloatLike;
      // clamp to cutoff distance
      if r_helio_au >= self.afrho_cutoff_au {
        return 0.0_f32;
      }

      let current_afrho_cm =
        self.afrho_0_cm * <f32 as FloatLike>::pow(r_helio_au, -self.afrho_power);
      let clamped_afrho_cm = <f32 as FloatLike>::min(current_afrho_cm, self.afrho_max_value_cm);
      let afrho_m = clamped_afrho_cm / 100_f32;
      let radius_m = self.diametre_um * 0.5e-6_f32;
      let density_kgm3 = self.density_gcm3 * 1e3_f32;

      // does dust velocity depend on heliocentric distance? Probably, but we don't care as godot
      // version doesn't care too
      (2.0 * density_kgm3 * radius_m * self.start_velocity_mean * afrho_m)
        / (3.0 * self.scattering_efficiency)
    }

    pub fn cluster_params(
      &self,
      mean_intra_grains_distance_mm: f32,
      min_cumulated_mass_g: f32,
    ) -> ParticleSystemEmitClusterParams {
      const PI: f32 = core::f32::consts::PI;

      // 1. Calculate the mass of a single grain
      // radius in cm: 1 μm = 10^-4 cm
      let radius_cm = (self.diametre_um * 0.5) * 1e-4;
      let volume_cm3 = (4.0 / 3.0) * PI * radius_cm.powi(3);
      let single_grain_mass_g = volume_cm3 * self.density_gcm3;

      // 2. Calculate how many grains we need to reach the minimum mass
      let num_spots = (min_cumulated_mass_g / single_grain_mass_g).ceil();
      let total_mass_g = num_spots * single_grain_mass_g;

      // 3. Calculate the radius of the macro-particle (cluster)
      // Assume grains are spaced out in a spherical volume.
      // The volume dedicated to one spot is a sphere with radius (mean_intra_grains_distance_mm / 2)
      let spot_radius_mm = mean_intra_grains_distance_mm * 0.5;
      // Total cluster volume in mm3 = num_spots * single_spot_volume
      // Since V = 4/3 * PI * R^3, R_cluster = R_spot * cbrt(num_spots)
      let cluster_radius_mm = spot_radius_mm * num_spots.powf(1.0 / 3.0);
      let cluster_radius_m = cluster_radius_mm * 1e-3;

      ParticleSystemEmitClusterParams {
        num_spots: num_spots as u32,
        radius_m: cluster_radius_m,
        mass_g: total_mass_g,
      }
    }
  }

  #[derive(Debug, Clone, Copy, PartialEq)]
  pub struct ParticleSystemEmitClusterParams {
    /// number of dust grains to render for our simulated cluster-particle
    pub num_spots: u32,
    /// radius of the sphere which represents the cluster-particle
    pub radius_m: f32,
    ///cumulated mass of the cluster particle, assuming uniform distribution
    pub mass_g: f32,
  }

  impl ParticleSystemEmitClusterParams {
    pub fn mass_std(&self, mass_variability_perc: f32) -> f32 {
      debug_assert!(mass_variability_perc >= 0.0 && mass_variability_perc <= 1.0);
      self.mass_g * mass_variability_perc
    }

    /// `grain_diametre_um` from [`ParticleSystemEmitParams`]
    pub fn compute_scales(
      &self,
      fov_y_rad: f32,
      viewport_height_px: f32,
      grain_diametre_um: f32,
    ) -> ParticleSystemRenderScales {
      use aethervk_oshal_rlib::math::FloatLike;
      // 1. Compute macroScale (World to Screen-Space Pixels)
      let tan_half_fov = <f32 as FloatLike>::tan(fov_y_rad * 0.5);
      let macro_scale = (self.radius_m * viewport_height_px) / tan_half_fov;

      // 2. Compute microRadius (World to UV-Space Ratio)
      let grain_radius_m = (grain_diametre_um * 0.5) * 1e-6; // μm to m
      let cluster_radius_m = self.radius_m;

      // in the fragment shader, the cluster spans UV [0.0, 1.0], meaning its UV radius is 0.5
      // we need to find the UV radius of a single grain through a simple proportion
      let micro_radius_uv = 0.5 * (grain_radius_m / cluster_radius_m);

      ParticleSystemRenderScales {
        macro_scale,
        micro_radius: micro_radius_uv,
      }
    }
  }

  #[derive(Debug, Clone, Copy, PartialEq)]
  pub struct ParticleSystemRenderScales {
    /// Screen Space size, in NDC coordinates, of a particle cluster/macro-particle `pc.macroScale`
    pub macro_scale: f32,
    /// Screen Space size, in NDC coordinates, of a dust grain. `pc.microRadius`
    pub micro_radius: f32,
  }

  pub struct ParticleSystemComponent {
    /// Strong reference to vulkan device resources
    pub device_data: (crate::gpu::RenderFrontend, crate::gpu::RenderDeviceHandle),
    /// value of the compute timeline which represents when the last particle system update
    /// will be completed
    pub timeline_value: AtomicU64,
    /// used to measure whether particle system should emit or not in next simulation step.
    /// initialized at zero so that first simulation step always emits (timeus_t)
    pub last_emission: AtomicI64,
    /// used to measure whether we should perform compaction or not in the next step
    /// gets initialized to zero in construcor, but if last_emission is zero, then in the first
    /// emission this is assigned to the last_emission value, such that we skip a useless
    /// compaction at start
    pub last_compaction: AtomicI64,
    /// necessary evil for Drop
    pub id: u64,
    // emission parameters
    pub emission_params: ParticleSystemEmitParams,
  }

  impl Component for ParticleSystemComponent {}

  impl core::fmt::Debug for ParticleSystemComponent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.debug_struct("ParticleSystemComponent")
        .field("device_data", &self.device_data.1)
        .field("timeline_value", &self.timeline_value)
        .finish()
    }
  }

  impl Clone for ParticleSystemComponent {
    fn clone(&self) -> Self {
      todo!()
    }
  }

  impl ParticleSystemComponent {
    pub fn new(
      render_frontend: crate::gpu::RenderFrontend,
      render_device_handle: crate::gpu::RenderDeviceHandle,
      entity_id: EntityId,
      emission_params: ParticleSystemEmitParams,
    ) -> EngineResult<Self> {
      let entity_u64 = entity_id.as_ffi();
      render_frontend
        .with_device(render_device_handle, |dyn_device: &_| {
          // unwrap cause the only particle system we support here is with vulkan
          let vulkan_device = dyn_device.as_any().downcast_ref::<vulkan::device::Device>().unwrap();
          vulkan_device.create_particle_system(entity_u64)
        })
        .map_err(EngineError::from)
        .map(|(_bda, timeline)| Self {
          device_data: (render_frontend, render_device_handle),
          timeline_value: AtomicU64::new(timeline),
          last_emission: AtomicI64::new(0),
          last_compaction: AtomicI64::new(0),
          id: entity_u64,
          emission_params,
        })
    }

    /// To be called after we submit a gpu workload to our particle system
    pub fn tick(&self) {
      self.timeline_value.fetch_add(1, Ordering::Relaxed);
    }
  }

  impl Drop for ParticleSystemComponent {
    fn drop(&mut self) {
      let _ = self.device_data.0.with_device(self.device_data.1, |dyn_device: &_| {
        // unwrap cause the only particle system we support here is with vulkan
        let vulkan_device = dyn_device.as_any().downcast_ref::<vulkan::device::Device>().unwrap();
        vulkan_device.discard_particle_system(self.id, self.timeline_value.load(Ordering::Relaxed))
      });
    }
  }
}

#[cfg(test)]
mod tests {
  use bytemuck::Zeroable;

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

  #[test]
  fn beta_computation() {
    use super::v2::*;
    let psep = {
      let mut x = ParticleSystemEmitParams::zeroed();
      x.diametre_um = 2_f32;
      x.density_gcm3 = 0.5_f32;
      x.scattering_efficiency = 1.03_f32;
      x
    };
    let tol = 1e-5_f32;
    let expected_beta = 1.1829265_f32;
    let actual_beta = psep.beta();

    approx::assert_abs_diff_eq!(actual_beta, expected_beta, epsilon = tol);
  }
}