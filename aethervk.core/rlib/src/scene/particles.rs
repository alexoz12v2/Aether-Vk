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
    /// random number generation (PCG) initial seed
    pub seed: u32,
    /// diameter of a *single* dust grain in μm
    pub diametre_um: f32,
    /// from 1 to 7, in g/cm3, volume density
    pub density_gcm3: f32,
    /// radiation-pressure efficiency `Q_pr` (0.5 to 2): the grain's momentum coupling to sunlight,
    /// enters β only ([`ParticleSystemEmitParams::beta`]); not an albedo
    pub scattering_efficiency: f32,
    /// geometric albedo `p` of the dust (0.01 to 0.5; cometary dust 0.03–0.06, default 0.04): the
    /// reflected radiance factor at zero phase is `p`, the Bond-like `A(0) = 4p` of the Afρ
    /// literature (A'Hearn et al. 1984, AJ 89, 579; Hanner & Newburn 1989, AJ 97, 254: 0.02–0.06 at
    /// small phase; the 67P coma grains 0.04–0.05, Fornasier et al. 2015). Enters the brightness
    /// (`τ · p · Φ(α) / r_h²`) and the Afρ → production conversion
    /// ([`ParticleSystemEmitParams::dust_production_rate_kgs`])
    pub geometric_albedo: f32,
    /// exponent `q` of the differential size law `n(s) ∝ s^-q` (2 to 5, default 3.5: Fulle 2004
    /// for 67P-like dust, Agarwal et al. 2010). **Enters the speck counts only**
    /// (`dust::specks`: grains per size bin and the drawn sizes); the continuum's size polyline
    /// (equal-cross-section bins equally spaced in `√β`, `size_edge_sqrt_beta`, with the speed
    /// factors linear in the bin index and `bin_density_ends`) and the mass → cross-section
    /// conversion (`size_mean_inv_s_um`) are laid out for q = 3.5, the law for which the
    /// cumulative cross-section `∝ s^(3−q)` is `√β`; another q would redistribute the flux along
    /// the syndyne (the propagate / splat mirrors), which is not done. 0 or out of range reads
    /// as 3.5 ([`Self::size_q`]).
    pub size_q: f32,
    /// size range factor `F` of the grains: radii `r/F .. r·F` about the configured radius (1 to
    /// 1000, default 10). Enters everything: the clusters' size polyline (`β/F .. β·F`), the
    /// mass → cross-section conversion, the Afρ closure and the speck counts. 0 or out of range
    /// reads as 10 ([`Self::size_range`]).
    pub size_range: f32,
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
    /// the size law exponent in force: [`Self::size_q`] when within 2..=5, else 3.5 (a zeroed or
    /// pre-v7 record)
    pub fn size_q(&self) -> f32 {
      if self.size_q.is_finite() && (2.0..=5.0).contains(&self.size_q) {
        self.size_q
      } else {
        crate::scene::dust::SIZE_POWER_Q as f32
      }
    }

    /// the size range factor in force: [`Self::size_range`] when within 1..=1000, else 10
    pub fn size_range(&self) -> f32 {
      if self.size_range.is_finite() && (1.0..=1000.0).contains(&self.size_range) {
        self.size_range
      } else {
        crate::scene::dust::SIZE_RANGE_FACTOR as f32
      }
    }

    /// the grain size distribution of the parameters (radius range from [`Self::size_range`])
    pub fn size_distribution(&self) -> crate::scene::dust::SizeDistribution {
      crate::scene::dust::SizeDistribution::from_diameter_um_range(
        self.diametre_um,
        self.size_range() as f64,
      )
    }

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
    ///
    /// Closure used here (proof in `dust_v3.md`, "Photometry"): for a steady outflow the aperture
    /// of radius ρ holds `π Q_n ρ / (2 v)` grains, so with the A'Hearn definition
    /// `Afρ = (2 Δ r_h)² / ρ · F_c / F☉` and the reflected flux `F = F☉ p Φ(α) σ / (π r_h² Δ²)` per
    /// grain, `Afρ = 2 p Φ(0) · Q_d · xsec_per_g · ⟨1/v⟩_σ` (cross-section-weighted mean of 1/v).
    /// The engine's grains have `v(s) = v_ref √(s_ref/s)` over the size distribution and a
    /// relative speed spread `v_std/v_ref`, hence the two factors below: `f_v = ⟨√(s/s_ref)⟩_σ`
    /// ([`crate::scene::dust::SizeDistribution::mean_speed_factor`], 0.809 for the default range)
    /// and `1 + (v_std/v_ref)²` (second order). With `xsec_per_g = 3/(4 ρ_d s_ref)` (the harmonic
    /// mean size of the q = 3.5 distribution is `s_ref`) this is the thesis / Agarwal form
    /// `Q_d = 2 ρ_d s_ref v_ref Afρ / (3 p f_v (1 + v_std_rel²))`. Verified by
    /// `afrho_closure_from_a_steady_isotropic_coma`.
    pub fn dust_production_rate_kgs(&self, r_helio_au: f32) -> f32 {
      use aethervk_oshal_rlib::math::FloatLike;
      // clamp to cutoff distance
      if r_helio_au >= self.afrho_cutoff_au || !(self.geometric_albedo > 0.0) {
        return 0.0_f32;
      }

      let current_afrho_cm =
        self.afrho_0_cm * <f32 as FloatLike>::pow(r_helio_au, -self.afrho_power);
      let clamped_afrho_cm = <f32 as FloatLike>::min(current_afrho_cm, self.afrho_max_value_cm);
      let afrho_m = clamped_afrho_cm / 100_f32;
      let radius_m = self.diametre_um * 0.5e-6_f32;
      let density_kgm3 = self.density_gcm3 * 1e3_f32;
      let dist = self.size_distribution();
      let f_v = dist.mean_speed_factor() as f32;
      let v_ref = self.start_velocity_mean.max(0.0);
      let v_std_rel = if v_ref > 0.0 {
        (self.start_velocity_std / v_ref).max(0.0)
      } else {
        0.0
      };

      // the ejection speed does not depend on the heliocentric distance (as the Godot version)
      (2.0 * density_kgm3 * radius_m * v_ref * afrho_m)
        / (3.0 * self.geometric_albedo * f_v * (1.0 + v_std_rel * v_std_rel))
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
    pub dust: spin::Mutex<crate::scene::dust::DustSystemState>,
    /// time to live for each cluster. Scaled time.
    pub ttl_us: timeus_t,
    /// emission parameters
    pub emission_params: ParticleSystemEmitParams,
    /// Draw parameters
    pub draw_params: ParticleSystemDrawParams,
    /// Ignition: the epoch the jet starts emitting at (TDB µs since J2000,
    /// [`emission_start_us_from_epoch`]). The dust at any epoch is what was emitted between the
    /// ignition and that epoch; nothing exists before it. [`EMISSION_START_PREEXISTING`] = a
    /// pre-existing tail (history back to 64 TTL before the current epoch), an explicit opt-in.
    pub emission_start_us: i64,
  }

  /// `emission_start_us` of a pre-existing tail: the history is filled back 64 TTL before the
  /// current epoch, before the scene start too. An explicit opt-in, never a default.
  pub const EMISSION_START_PREEXISTING: i64 = i64::MIN;
  /// DTO sentinel: ignite at the scene's current epoch when the jet is added.
  pub const EMISSION_START_NOW: i64 = i64::MAX;

  /// TDB microseconds since J2000 of `epoch`, the unit of
  /// [`ParticleSystemComponent::emission_start_us`] (absolute: the scene's epoch range may move,
  /// the jet's birth must not).
  pub fn emission_start_us_from_epoch(epoch: anise::time::Epoch) -> i64 {
    (epoch.to_tdb_duration().total_nanoseconds() / 1000) as i64
  }

  /// The DTO's `emission_start_us` resolved against the scene's clock: [`EMISSION_START_NOW`]
  /// becomes the clock's current epoch, and is `None` without a clock (the jet is refused: the
  /// clock is created with the scene, so a scene without one is broken and must not get a silent
  /// pre-existing tail); every other value, [`EMISSION_START_PREEXISTING`] included, passes
  /// through.
  pub fn resolve_emission_start_us(
    requested: i64,
    clock: Option<anise::time::Epoch>,
  ) -> Option<i64> {
    match requested {
      EMISSION_START_NOW => clock.map(emission_start_us_from_epoch),
      us => Some(us),
    }
  }

  /// Inverse of [`emission_start_us_from_epoch`].
  pub fn epoch_from_emission_start_us(us: i64) -> anise::time::Epoch {
    anise::time::Epoch::from_tdb_duration(anise::time::Duration::from_total_nanoseconds(
      us as i128 * 1000,
    ))
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
        dist: self.size_distribution(),
        diameter_um: self.diametre_um,
        density_gcm3: self.density_gcm3,
        beta_ref: self.beta(),
        albedo: self.geometric_albedo,
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
        emission_start_us: self.emission_start_us,
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
      emission_start_us: i64,
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
          dust: spin::Mutex::new(crate::scene::dust::DustSystemState::new(capacity)),
          ttl_us,
          emission_params,
          draw_params,
          emission_start_us,
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
