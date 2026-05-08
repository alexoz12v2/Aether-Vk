use aethervk_core_rlib::gpu;
use aethervk_core_rlib::gpu::scene_conversion::SceneConversionExt;
use aethervk_core_rlib::scene::{CameraComponent, TransformComponent};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::types::GpuResult;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::matrix::Matrix4;
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec4::Quat, Vector3};
use rand::Rng;
use rayon::prelude::*;
use test_utils::{
  create_winit_window_and_event_loop, cycle_get_asset_path_from_exe, get_handle_and_window_info,
};

struct AppState {
  ctx: Box<SimulationContext>,
  scene_id: u64,
  presentation_engine: gpu::PresentationEngineHandle,
  window: Option<winit::window::Window>,
  particle_sys_entity: u64,
  is_resizing: bool,
  is_exiting: bool,
}

impl Drop for AppState {
  fn drop(&mut self) {
    println!("Dropping AppState");
  }
}

fn panic_error_callback(msg: &str) {
  panic!("Vulkan Error: {}", msg);
}

fn main() {
  let start_time = std::time::Instant::now();
  println!("[{:.2?}] Application starting.", start_time.elapsed());

  let assets_dir = cycle_get_asset_path_from_exe(true);

  let (window, event_loop) =
    create_winit_window_and_event_loop("AetherVk Particles Brownian Motion");

  let mut simulation_context =
    SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback)).unwrap();

  let (native_handles, window_info) = {
    let render_frontend = simulation_context.render_frontend().unwrap();
    let render_device_handle = simulation_context.render_device_handle();
    get_handle_and_window_info(&render_frontend, render_device_handle, &window)
  };

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let scene_id = simulation_context.create_empty_scene().unwrap();

  let presentation_engine = simulation_context
    .create_presentation_engine_windowed(scene_id, width, height, native_handles)
    .unwrap();

  let particle_sys_entity;

  // Populate scene
  let particle_ext_sys_entity = {
    let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
    let mut active_scene = scene_ctx.write();
    let root_entity = active_scene.root_entity;

    active_scene.scene.register_component::<TransformComponent>(&[]);
    active_scene
      .scene
      .register_component::<CameraComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
    active_scene
      .scene
      .register_component::<aethervk_core_rlib::scene::particles::ParticleSystemComponent>(&[
        std::any::TypeId::of::<TransformComponent>(),
      ]);

    let camera_entity = active_scene.scene.spawn_entity("camera");
    active_scene.scene.set_parent(camera_entity, Some(root_entity));
    active_scene
      .scene
      .add_component(
        camera_entity,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, -100.0),
          rotation: Quat::from_axis_angle(
            Vec3f32::from_components(0.0, 1.0, 0.0),
            std::f32::consts::PI,
          ),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    active_scene
      .scene
      .add_component(
        camera_entity,
        CameraComponent {
          projection: Mat4x4f32::perspective_vk(
            45.0f32.to_radians(),
            width as f32 / height as f32,
            0.1,
            1000.0,
          ),
          near_plane: 0.1,
          far_plane: 1000.0,
        },
      )
      .unwrap();
    active_scene.active_camera_entity = Some(camera_entity);

    particle_sys_entity = active_scene.scene.spawn_entity("particles");
    active_scene.scene.set_parent(particle_sys_entity, Some(root_entity));
    active_scene
      .scene
      .add_component(
        particle_sys_entity,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();

    let mut sys = aethervk_core_rlib::scene::particles::ParticleSystemComponent::new(
      aethervk_core_rlib::scene::particles::ParticleEmitterConfig {
        uv_distribution: aethervk_core_rlib::math::distribution::Distribution2D::new(
          &[1.0, 1.0, 1.0, 1.0],
          2,
          2,
        ),
        delta: 1000,
        max_particles: 1_000_000,
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
        particle_radius: 0.15,
        density: 1.0,
        lifetime: 100000000,
        color: [1.0, 0.7, 0.4, 0.08], // Orange/brown dust
        beta: 0.0,
        use_particle2: true, // USE PARTICLE 2!
      },
    );

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

    active_scene.scene.add_component(particle_sys_entity, sys).unwrap();
    active_scene.register_entity(particle_sys_entity)
  };

  let app_state = AppState {
    ctx: simulation_context,
    scene_id,
    presentation_engine,
    window: Some(window),
    particle_sys_entity: particle_ext_sys_entity,
    is_resizing: false,
    is_exiting: false,
  };

  let _ = app_state.ctx.threads.logic_thread.tx().try_send(
    aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
      scene_id,
      scale: aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay,
    },
  );
  let _ = app_state
    .ctx
    .threads
    .logic_thread
    .tx()
    .try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id });

  let sim_app = SimApp {
    app_state,
    last_sim_time: std::time::Instant::now(),
    window_info,
  };

  test_utils::app::run_app(sim_app, event_loop);
}

struct SimApp {
  app_state: AppState,
  last_sim_time: std::time::Instant,
  window_info: test_utils::WindowPlatformData,
}

impl test_utils::app::App for SimApp {
  fn window(&self) -> Option<&winit::window::Window> {
    self.app_state.window.as_ref()
  }
  fn is_resizing(&self) -> bool {
    self.app_state.is_resizing
  }
  fn set_resizing(&mut self, resizing: bool) {
    self.app_state.is_resizing = resizing;
  }
  fn is_exiting(&self) -> bool {
    self.app_state.is_exiting
  }
  fn set_exiting(&mut self, exiting: bool) {
    self.app_state.is_exiting = exiting;
  }

  fn on_resize(&mut self, width: u32, height: u32) {
    let _ = self.app_state.ctx.resize(
      self.app_state.scene_id,
      self.app_state.presentation_engine,
      width,
      height,
    );
    #[cfg(target_os = "macos")]
    {
      self.window_info.metal_layer.setDrawableSize(objc2_core_foundation::CGSize {
        width: width as f64,
        height: height as f64,
      });
    }
  }

  fn on_close_requested(&mut self) {
    self.app_state.window = None;
  }

  fn on_about_to_wait(&mut self) {
    let current_time = std::time::Instant::now();
    let delta_time = current_time.duration_since(self.last_sim_time).as_secs_f32();
    self.last_sim_time = current_time;

    if self.app_state.window.is_none() {
      return;
    }

    let size = self.app_state.window.as_ref().unwrap().inner_size();
    if size.width == 0 || size.height == 0 {
      return;
    }

    // Brownian Motion Update on 1M particles using Rayon!
    {
      let scene_ctx = self.app_state.ctx.get_scene(self.app_state.scene_id).unwrap();
      let read_scene = scene_ctx.read();
      let particle_sys_entity = read_scene.get_entity(self.app_state.particle_sys_entity).unwrap();
      read_scene.scene.with_component(
        particle_sys_entity,
        |sys: &aethervk_core_rlib::scene::particles::ParticleSystemComponent| {
          let mut particles = sys.particles.write();
          let dt_sqrt = delta_time.sqrt();
          let drift_strength = 5.0; // Moderate drift speed

          // We use a small pseudo-random noise directly on the GPU for visual swirling,
          // but the CPU can do the large-scale true random walk (Brownian).
          let time_sec = current_time.elapsed().as_secs_f32();

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
