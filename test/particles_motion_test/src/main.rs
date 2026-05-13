use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::{CameraComponent, TransformComponent};
use aethervk_core_rlib::simulation_api::components_api::{CameraParams, PerspectiveCameraParams};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::{EngineResult, GpuResult};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector3};
use rand::Rng;
use rayon::prelude::*;
use test_utils::cycle_get_asset_path_from_exe;
use test_utils::sim_app::{run_simulation_app, SimulationDelegate};
use winit::window::Window;

struct ParticlesDelegate {
  camera_ext_entity: u64,
  particle_sys_entity: u64,
  startup_time: std::time::Instant,
}

impl SimulationDelegate for ParticlesDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene()
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    window: &Window,
  ) -> EngineResult<()> {
    let root_entity = ctx.spawn_entity(scene_id, "root").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        root_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let camera_entity =
      ctx.add_perspective_camera(scene_id, pe_handle, "camera", 45.0, 0.1, 1000.0).unwrap().get();
    ctx.set_parent(scene_id, camera_entity, root_entity).unwrap();
    ctx
      .set_transform_component(
        scene_id,
        camera_entity,
        Vec3f32::from_components(0.0, 0.0, -100.0),
        Quat::from_axis_angle(
          Vec3f32::from_components(0.0, 1.0, 0.0),
          std::f32::consts::PI,
        ),
        Vec3f32::one(),
      )
      .unwrap();
    self.camera_ext_entity = camera_entity;

    self.particle_sys_entity = ctx.spawn_entity(scene_id, "particles").unwrap();
    ctx.set_parent(scene_id, self.particle_sys_entity, root_entity).unwrap();
    ctx
      .add_transform_component(
        scene_id,
        self.particle_sys_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::one(),
      )
      .unwrap();

    let config = aethervk_core_rlib::scene::particles::ParticleEmitterComponent {
        uv_distribution: aethervk_core_rlib::math::distribution::Distribution2D::new(
          &[1.0, 1.0, 1.0, 1.0],
          2,
          2,
        ),
        delta: 1000,
        max_particles: 50_000,
        velocity_intensity: aethervk_core_rlib::scene::particles::GaussianParams {
          mean: 1.0,
          std_dev: 0.0,
          min: 0.0,
          max: 1.0,
        },
        emission_count: aethervk_core_rlib::scene::particles::GaussianParams {
          mean: 1.0,
          std_dev: 0.0,
          min: 0.0,
          max: 1.0,
        },
        particle_radius: 0.25,
        density: 1.0,
        lifetime: 1000000,
        color: [1.0, 0.5, 0.25, 1.0],
        beta: 0.0,
        use_particle2: false,
      };

    let mut sys = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(config.max_particles);

    // Initialize 1M particles
    let mut rng = rand::thread_rng();
    for i in 0..1_000_000 {
      let mut p = aethervk_core_rlib::scene::particles::ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [
          rng.gen_range(-40.0..40.0),
          rng.gen_range(-40.0..40.0),
          rng.gen_range(-40.0..40.0),
        ],
        mass: 1.0,
        velocity: [0.0, 0.0, 0.0],
        active: 1,
      };
      p.set_id(i as u64);
      p.set_age(0);
      sys.particles.write().push(p);
    }

    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let mut active_scene = scene_ctx.write();
      let sys_entity_id = active_scene.get_entity(self.particle_sys_entity).unwrap();
      active_scene.scene.add_component(sys_entity_id, sys).unwrap();
      active_scene.scene.add_component(sys_entity_id, config).unwrap();
    }

    let _ = ctx.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
        scene_id,
        scale: aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay,
      },
    );
    let _ =
      ctx.threads.logic_thread.tx().try_send(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
      );

    Ok(())
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, delta_time: f32) {
    if delta_time <= 0.0 {
      return;
    }

    // Brownian Motion Update on 1M particles using Rayon!
    if let Some(scene_ctx) = ctx.get_scene(scene_id) {
      let read_scene = scene_ctx.read();
      if let Some(particle_sys_entity) = read_scene.get_entity(self.particle_sys_entity) {
        read_scene.scene.with_component(
          particle_sys_entity,
          |sys: &aethervk_core_rlib::scene::particles::ParticleSystemComponent| {
            let mut particles = sys.particles.write();
            let dt_sqrt = delta_time.sqrt();
            let drift_strength = 5.0; // Moderate drift speed

            let time_sec = self.startup_time.elapsed().as_secs_f32();

            particles.par_iter_mut().for_each(|p| {
              if p.active != 0 {
                let id = p.id_low ^ p.id_high;
                let seed = id as f32 + time_sec;

                // Simple cheap hash on CPU
                let hx = (seed * 12.9898).sin() * 43758.5453;
                let hy = (seed * 78.233).sin() * 43758.5453;
                let hz = (seed * 37.719).sin() * 43758.5453;

                let dx = (hx - hx.floor() - 0.5) * 2.0;
                let dy = (hy - hy.floor() - 0.5) * 2.0;
                let dz = (hz - hz.floor() - 0.5) * 2.0;

                p.position[0] += dx * drift_strength * dt_sqrt;
                p.position[1] += dy * drift_strength * dt_sqrt;
                p.position[2] += dz * drift_strength * dt_sqrt;
              }
            });
          },
        );
      }
    }
  }

  fn on_mouse_motion(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: (f64, f64),
    middle_mouse_down: bool,
    shift_down: bool,
    ctrl_down: bool,
  ) {
    let scene = ctx.get_scene(scene_id).unwrap();
    let camera_entity = scene.read().get_entity(self.camera_ext_entity).unwrap();
    if let Some(logic_command) = test_utils::command::process_mouse_motion_camera_commands(
      delta,
      middle_mouse_down,
      shift_down,
      ctrl_down,
      camera_entity,
      scene.clone(),
    ) {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }
}

fn main() {
  let _assets_dir = cycle_get_asset_path_from_exe(true);
  let delegate = ParticlesDelegate {
    camera_ext_entity: 0,
    particle_sys_entity: 0,
    startup_time: std::time::Instant::now(),
  };
  run_simulation_app("AetherVk Particles Brownian Motion", delegate);
}
