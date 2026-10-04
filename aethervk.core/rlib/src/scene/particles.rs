//! particles module.

use crate::scene::Component;
use aethervk_oshal_rlib::{
  math::vector::{Vector3, vec3::Vec3f32},
  os::time::timeus_t,
};

pub use v2::*;

/// Note: tight coupling with vulkan here
pub mod v2 {
  use super::*;
  use crate::{
    gpu_backends::vulkan,
    scene::EntityId,
    types::{EngineError, EngineResult},
  };

  // TODO C# side: copy paste from a jet to another, godot edition has params shared
  #[derive(
    Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, serde::Serialize, serde::Deserialize,
  )]
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
  }

  /// Owns the GPU-side resources for one particle system.
  /// Wrapped in `Arc` so that a shallow `Scene` clone (for snapshots) can share the handle
  /// without duplicating GPU memory.  GPU cleanup only fires when the *last* `Arc` is dropped.
  pub struct ParticleSystemResource {
    /// Strong reference to vulkan device resources
    pub device_data: (crate::gpu::RenderFrontend, crate::gpu::RenderDeviceHandle),
    /// key into the device's `DustManager` (= ECS EntityId as u64)
    pub id: u64,
  }

  impl Drop for ParticleSystemResource {
    fn drop(&mut self) {
      let _ = self.device_data.0.with_device(self.device_data.1, |dyn_device: &_| {
        use crate::gpu_backends::vulkan::utils::RwLockable;
        // unwrap cause the only particle system we support here is with vulkan
        let vulkan_device = dyn_device.as_any().downcast_ref::<vulkan::device::Device>().unwrap();
        // heuristic: use the next release timeline value as discard value
        let gfx_release = vulkan_device.res.read().get_timeline_semaphore_cached_value() + 1;
        let comp_release = vulkan_device
          .kernels
          .next_submit_value
          .load(core::sync::atomic::Ordering::Relaxed);
        vulkan_device.discard_particle_system(self.id, gfx_release, comp_release)
      });
    }
  }

  pub struct ParticleSystemComponent {
    /// Shared GPU resource handle.  `Arc` allows a safe shallow clone when snapshotting a
    /// `Scene`: the clone holds a second reference to the same GPU allocations, preventing
    /// premature `discard_particle_system` until *all* clones are dropped.
    pub resource: alloc::sync::Arc<ParticleSystemResource>,
    /// Dust v3 host state: ring bookkeeping, emission window, latest jet state. Written by the
    /// logic thread, read by the render scene builder. A cloned (snapshotted) component carries
    /// a copy, whose GPU ring content must be re-emitted on restore
    /// (`DustHostState::ring.invalidate_gpu`).
    pub dust: spin::Mutex<crate::scene::dust::DustHostState>,
    /// time to live for each cluster. Scaled time.
    pub ttl_us: timeus_t,
    /// emission parameters
    pub emission_params: ParticleSystemEmitParams,
    /// Draw parameters
    pub draw_params: ParticleSystemDrawParams,
  }

  /// Non physically based draw parameters
  #[derive(Clone)]
  pub struct ParticleSystemDrawParams {
    pub stream_color: [f32; 4],
  }

  impl ParticleSystemEmitParams {
    /// Emission inputs for [`crate::scene::dust::DustHostState::tick`] at heliocentric distance
    /// `r_helio_au`.
    pub fn dust_emit_config(
      &self,
      r_helio_au: f32,
      ttl_us: timeus_t,
    ) -> crate::scene::dust::DustEmitConfig {
      let dir = self.particle_system_relative_cone_direction();
      crate::scene::dust::DustEmitConfig {
        q_dust_kgs: self.dust_production_rate_kgs(r_helio_au) as f64,
        ttl_s: ttl_us.max(0) as f64 * 1e-6,
        dist: crate::scene::dust::SizeDistribution::from_diameter_um(self.diametre_um),
        diameter_um: self.diametre_um,
        density_gcm3: self.density_gcm3,
        beta_ref: self.beta(),
        v_mean: self.start_velocity_mean,
        v_std: self.start_velocity_std,
        jet_dir: [dir.x(), dir.y(), dir.z()],
        aperture_rad: self.aperture_rad,
        seed: self.seed,
      }
    }
  }

  impl Component for ParticleSystemComponent {}

  impl core::fmt::Debug for ParticleSystemComponent {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      f.debug_struct("ParticleSystemComponent")
        .field("device_data", &self.resource.device_data.1)
        .finish()
    }
  }

  impl Clone for ParticleSystemComponent {
    fn clone(&self) -> Self {
      Self {
        resource: alloc::sync::Arc::clone(&self.resource),
        dust: spin::Mutex::new(self.dust.lock().clone()),
        ttl_us: self.ttl_us,
        emission_params: self.emission_params,
        draw_params: self.draw_params.clone(),
      }
    }
  }

  impl ParticleSystemComponent {
    pub fn new(
      render_frontend: crate::gpu::RenderFrontend,
      render_device_handle: crate::gpu::RenderDeviceHandle,
      entity_id: EntityId,
      emission_params: ParticleSystemEmitParams,
      draw_params: ParticleSystemDrawParams,
      ttl_us: timeus_t,
    ) -> EngineResult<Self> {
      let entity_u64 = entity_id.as_ffi();
      render_frontend
        .with_device(render_device_handle, |dyn_device: &_| {
          // unwrap cause the only particle system we support here is with vulkan
          let vulkan_device = dyn_device.as_any().downcast_ref::<vulkan::device::Device>().unwrap();
          vulkan_device.create_particle_system(entity_u64)
        })
        .map_err(EngineError::from)
        .map(|capacity| Self {
          resource: alloc::sync::Arc::new(ParticleSystemResource {
            device_data: (render_frontend, render_device_handle),
            id: entity_u64,
          }),
          dust: spin::Mutex::new(crate::scene::dust::DustHostState::new(capacity)),
          ttl_us,
          emission_params,
          draw_params,
        })
    }
  }
  // Note: Drop is implemented on ParticleSystemResource (inside the Arc).
  // When the last Arc clone is dropped, GPU resources are freed exactly once.
}

#[cfg(test)]
mod test_particles;

#[cfg(test)]
mod test_dust_config;
