extern crate alloc;
use aethervk_core_rlib::scene::ParticleSystemComponent;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3};

pub trait ParticleSystemExt {
  fn update_particles(
    &mut self,
    dt: f32,
    sun_pos: Vec3f32,
    comet_pos: Vec3f32,
    comet_mass: f32,
    u_roulette: &[f32],
  );
}

impl ParticleSystemExt for ParticleSystemComponent {
  fn update_particles(
    &mut self,
    dt: f32,
    sun_pos: Vec3f32,
    comet_pos: Vec3f32,
    comet_mass: f32,
    u_roulette: &[f32],
  ) {
    let mut roulette_idx = 0;
    for p in self.particles.write().iter_mut().filter(|p| p.active != 0) {
      let mut age = p.get_age();
      age += (dt * 1_000_000.0) as i64;
      p.set_age(age);

      // Russian roulette
      if age > self.config.lifetime as i64 {
        let age_excess = (age - self.config.lifetime as i64) as f32 / 1_000_000.0;
        let death_prob = 1.0 - (-age_excess).exp(); // Exponential decay

        let u = if roulette_idx < u_roulette.len() {
          u_roulette[roulette_idx]
        } else {
          0.5
        };
        roulette_idx += 1;

        if u < death_prob {
          p.active = 0;
          continue;
        }
      }

      let p_pos = Vec3f32::from_array(p.position);
      let mut p_vel = Vec3f32::from_array(p.velocity);

      // Gravity
      let to_sun = sun_pos - p_pos;
      let dist_sq_sun = to_sun.length_squared().max(1e-4);
      let force_sun = to_sun.normalize()
        * (aethervk_core_rlib::physics::cpu::G * 100000000.0 * p.mass / dist_sq_sun)
        * (1.0 - self.config.beta);

      let to_comet = comet_pos - p_pos;
      let dist_sq_comet = to_comet.length_squared().max(1e-4);
      let force_comet = to_comet.normalize()
        * (aethervk_core_rlib::physics::cpu::G * comet_mass * p.mass / dist_sq_comet);

      let acceleration = (force_sun + force_comet) / p.mass;
      p_vel += acceleration * dt;
      let new_pos = p_pos + p_vel * dt;

      p.position = [new_pos.x(), new_pos.y(), new_pos.z()];
      p.velocity = [p_vel.x(), p_vel.y(), p_vel.z()];
    }

    // Clean up dead particles
    if self.particles.read().len() > self.config.max_particles {
      self.particles.write().retain(|p| p.active == 0);
    }
  }
}
