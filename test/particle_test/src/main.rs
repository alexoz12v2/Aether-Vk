use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
  scene::{
    CameraComponent, EntityId, PhysicalMeshComponent, Scene, TransformComponent, SunComponent,
  },
  types::RuntimeParams,
  gpu::{FrameCancelGuard, ScopedCommandBuffer, ScopedRenderPass},
  scene::{GaussianParams, ParticleEmitterConfig, ParticleSystemComponent},
  types::GpuResult,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector3, Vector},
};
use heapless::index_map::FnvIndexMap;
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe, get_handle_and_window_info, scene_to_render_scene,
  setup_resize_hook, AppEvent,
};
use winit::{event_loop::EventLoopBuilder, window::WindowBuilder};
use test_utils::simulation::kernels::CpuKernels;

// Particle emitter component
// - Note: the methods which require randomness receive a list of random numbers in input like in pbrt-v4
// - particle component should have an emitter configuration with
//    - radius: this will be the extent of a circle in the uv space of the mesh,
//      because on the next emission we sample a random position in UV space and map it back
//      to object space
//    - timeus_t delta: since we receive the delta time from last fixed update, we can compare that
//      with this parameter to decide whether to spawn new particles
//    - max particles: maximum number of live particles
//    - initial velocity distribution: the initial velocity will be
//      - direction: sampled randomly in the cosine hemisphere
//      - intensity: "Gaussian-Like" distribution, but bounded,
//        whose mean should be some_scaling_factor * beta(from JPL data)^0.9
//    - number of particles per emission parameters: "Gaussian-Like" distribution, but bounded,
//      whose mean is inversely proportional to density as 1/x^2 * some scaling factor
//  - diameter (size of a single particle, we can also store the radius)
//  - density (from this input we can compute mass, and store the mass)
//  - bvh: bounded volume hierarchies of clusters of particles. Must be fast to recompute cause it's dynamic
//      This is still stored in object space (particle emitter object space = mesh object space)
//  - lifetime parameter in timeus. After crossing this lifetime, we use the time since the lifetime threshold to scale
//    killing probability (basically roussian roulette)
//  - color (each particle system gets tagged with a color)

struct AppState {
  is_resizing: bool,
  is_exiting: bool,
  scene: Arc<Scene>,
  window: Option<winit::window::Window>,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  camera_entity: EntityId,
  mesh_entity: EntityId,
  width: u32,
  height: u32,
}

struct ParticleApp {
  app_state: AppState,
  render_frontend: gpu::RenderFrontend,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: Option<gpu::PresentationEngineHandle>,
  camera_entity: EntityId,
  mesh_entity: EntityId,
  sun_entity: EntityId,
  right_mouse_button_down: bool,
  cam_dist: f32,
  cam_yaw: f32,
  cam_pitch: f32,
  width: u32,
  height: u32,
  window_info: test_utils::WindowPlatformData,
  time_info: aethervk_oshal_rlib::os::time::TimeInfo,
  kernels: CpuKernels,
}

impl ParticleApp {
  fn simulate_particles(&mut self, dt: f32) {
    let mut sun_pos = Vec3f32::from_array([0.0, 0.0, 0.0]);
    self
      .app_state
      .scene
      .with_component(self.sun_entity, |t: &TransformComponent| {
        sun_pos = t.position;
      });

    let mut comet_pos = Vec3f32::from_array([0.0, 0.0, 0.0]);
    let mut comet_rot = Quat::identity();
    self
      .app_state
      .scene
      .with_component(self.mesh_entity, |t: &TransformComponent| {
        comet_pos = t.position;
        comet_rot = t.rotation;
      });

    let mesh_arc = self
      .app_state
      .scene
      .with_component(self.mesh_entity, |m: &PhysicalMeshComponent| m.mesh.clone())
      .unwrap();

    // Cache UvGrid - in a real app this would be computed once per mesh
    let uv_grid = aethervk_core_rlib::simulation::comet::uv_grid::UvGrid::new(
      &mesh_arc.vertices,
      &mesh_arc.indices,
      64,
    );

    self
      .app_state
      .scene
      .query1_mut::<ParticleSystemComponent, _>(|_, sys| {
        sys.accumulator += (dt * 1_000_000.0) as i64;

        // Emission
        while sys.accumulator >= sys.config.delta {
          sys.accumulator -= sys.config.delta;

          let u_emission = [rand::random::<f32>(), rand::random::<f32>()];

          let count = sys.config.emission_count.sample(&u_emission) as usize;
          let mut u_particles = std::vec::Vec::with_capacity(count);
          for _ in 0..count {
            u_particles.push([
              rand::random::<f32>(),
              rand::random::<f32>(),
              rand::random::<f32>(),
              rand::random::<f32>(),
            ]);
          }

          sys.emit_particles(
            &mesh_arc,
            &uv_grid,
            comet_pos,
            comet_rot,
            Vec3f32::from_components(1.0, 1.0, 1.0),
            &u_emission,
            &u_particles,
          );
        }

        // Update
        let mut u_roulette = std::vec::Vec::with_capacity(sys.particles.len());
        for _ in 0..sys.particles.len() {
          u_roulette.push(rand::random::<f32>());
        }

        let mut roulette_idx = 0;
        for p in sys.particles.iter_mut().filter(|p| p.active != 0) {
          let mut age = p.get_age();
          age += (dt * 1_000_000.0) as i64;
          p.set_age(age);

          // Russian roulette
          if age > sys.config.lifetime as i64 {
            let age_excess = (age - sys.config.lifetime as i64) as f32 / 1_000_000.0;
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
        }

        sys.particles.retain(|p| p.active != 0);

        sys.update_bvh();
      });

    let mut physics_scene =
      aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(
        &self.app_state.scene,
      );
    let t0 = self.time_info.current().unscaled_time;
    let t1 = t0 + (dt * 1_000_000.0) as i64;

    aethervk_core_rlib::gpu_backends::simulation_step(
      &self.kernels,
      &mut physics_scene,
      &self.app_state.scene,
      t0,
      t1,
    )
    .unwrap();
  }
}

impl test_utils::app::App for ParticleApp {
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
    self.width = width;
    self.height = height;
    #[cfg(target_os = "macos")]
    {
      self
        .window_info
        .metal_layer
        .setDrawableSize(objc2_core_foundation::CGSize {
          width: width as f64,
          height: height as f64,
        });
    }
    self
      .app_state
      .scene
      .with_component_mut(self.camera_entity, |c: &mut CameraComponent| {
        c.projection = Mat4x4f32::perspective_vk(
          std::f32::consts::FRAC_PI_4,
          width as f32 / height as f32,
          0.1,
          100.0,
        );
      });
  }

  fn on_close_requested(&mut self) {
    self.app_state.window = None;
  }

  fn on_mouse_input(
    &mut self,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
  ) {
    if button == winit::event::MouseButton::Right {
      self.right_mouse_button_down = state == winit::event::ElementState::Pressed;
    }
  }

  fn on_keyboard_input(
    &mut self,
    event: &winit::event::KeyEvent,
    modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state == winit::event::ElementState::Pressed {
      if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
        if keycode == winit::keyboard::KeyCode::Escape {
          // Do not exit on escape
        }
        #[cfg(target_os = "macos")]
        if keycode == winit::keyboard::KeyCode::KeyQ && modifiers.super_key() {
          self.app_state.is_exiting = true;
        }
      }
    }
  }

  fn on_mouse_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
    let scroll_amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32,
    };
    let zoom_speed = self.cam_dist * 0.1;
    self.cam_dist = (self.cam_dist - scroll_amount * zoom_speed).max(0.1);

    let yaw_quat = Quat::from_axis_angle(Vec3f32::from_array([0.0, 0.0, 1.0]), self.cam_yaw);
    let pitch_quat = Quat::from_axis_angle(Vec3f32::from_array([1.0, 0.0, 0.0]), self.cam_pitch);
    let new_rot = (yaw_quat * pitch_quat).normalize();
    let offset = Vec3f32::from_array([0.0, self.cam_dist, 0.0]);
    let new_offset = new_rot.rotate_vector(offset);
    self
      .app_state
      .scene
      .with_component_mut(self.camera_entity, |c: &mut TransformComponent| {
        c.position = new_offset;
        c.rotation = new_rot;
      });
    if let Some(w) = self.app_state.window.as_ref() {
      w.request_redraw();
    }
  }

  fn on_mouse_motion(&mut self, delta: (f64, f64)) {
    if self.right_mouse_button_down {
      let rotation_speed = 0.005;
      self.cam_yaw += delta.0 as f32 * rotation_speed;
      self.cam_pitch += delta.1 as f32 * rotation_speed;
      self.cam_yaw = self.cam_yaw % (std::f32::consts::PI * 2.0);
      self.cam_pitch = self.cam_pitch.clamp(-1.55, 1.55);

      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_array([0.0, 0.0, 1.0]), self.cam_yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_array([1.0, 0.0, 0.0]), self.cam_pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();
      let offset = Vec3f32::from_array([0.0, self.cam_dist, 0.0]);
      let new_offset = new_rot.rotate_vector(offset);
      self
        .app_state
        .scene
        .with_component_mut(self.camera_entity, |c: &mut TransformComponent| {
          c.position = new_offset;
          c.rotation = new_rot;
        });
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }
  }

  fn on_about_to_wait(&mut self) {
    self.time_info.ut_update();
    while self.time_info.needs_fixed_update() {
      self.time_info.ut_fixed_update();
      let dt = self
        .time_info
        .fixed_delta_time
        .load(core::sync::atomic::Ordering::Relaxed) as f32
        / 1_000_000.0;
      self.simulate_particles(dt);
    }
    if let Some(w) = self.app_state.window.as_ref() {
      w.request_redraw();
    }
  }

  fn on_redraw(&mut self) {
    if self.width == 0 || self.height == 0 {
      return;
    }

    let mut payload = RenderPayloadData {
      presentation_engine: self.presentation_engine.unwrap(),
      scene: &self.app_state.scene,
      camera_entity: self.camera_entity,
      mesh_entity: self.mesh_entity,
      width: self.width,
      height: self.height,
    };

    self
      .render_frontend
      .with_device(self.render_device_handle, |device| {
        if let Err(e) = render_function(device, &mut payload) {
          println!("Render function error: {:?}", e);
        }
        Ok(())
      })
      .unwrap();
  }
}

fn render_function(device: &dyn RenderDevice, payload: &mut RenderPayloadData) -> GpuResult<()> {
  let mut render_scene = scene_to_render_scene(
    &payload.scene,
    device,
    payload.presentation_engine,
    payload.camera_entity,
    false,
  )?;

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.width,
      payload.height,
    )?;
    return Ok(());
  }
  let present_guard = FrameCancelGuard::new(device, payload.presentation_engine, acquire_result);

  let cmd_buffer = device.get_command_buffer()?;
  let cmd_guard = ScopedCommandBuffer::new(device, cmd_buffer, None)?;

  device.upload_particle_systems(cmd_buffer, &mut render_scene.particle_calls)?;

  if let Some(sun_call) = &render_scene.sun_call {
    // TODO gather resolution from extraction
    device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))?;
  }

  device.begin_render_pass(cmd_buffer, payload.presentation_engine, &acquire_result)?;
  let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer);

  let extent = device.get_presentation_engine_extent(payload.presentation_engine)?;
  device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  device.render_frame(cmd_buffer, &render_scene)?;

  // defuse of drop guards (Order *must* be like this)
  render_pass_guard.end()?;
  cmd_guard.submit()?;
  present_guard.defuse();

  let present_status = device.present(
    payload.presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;
  if present_status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.width,
      payload.height,
    )?;
  }

  Ok(())
}

fn vulkan_rendering_validation_callback(err_msg: &str) {
  panic!("Vulkan Validation Error: {}", err_msg);
}

fn build_uv_distribution(
  comet: &aethervk_core_rlib::simulation::comet::Comet,
  resolution: usize,
) -> aethervk_core_rlib::math::distribution::Distribution2D {
  let mut min_u = std::f32::MAX;
  let mut max_u = std::f32::MIN;
  let mut min_v = std::f32::MAX;
  let mut max_v = std::f32::MIN;

  for v in &comet.vertices {
    min_u = min_u.min(v.uv[0]);
    max_u = max_u.max(v.uv[0]);
    min_v = min_v.min(v.uv[1]);
    max_v = max_v.max(v.uv[1]);
  }
  if max_u <= min_u {
    max_u = min_u + 1.0;
  }
  if max_v <= min_v {
    max_v = min_v + 1.0;
  }

  let mut weights = vec![0.0; resolution * resolution];
  let inv_w = resolution as f32 / (max_u - min_u);
  let inv_h = resolution as f32 / (max_v - min_v);

  for chunk in comet.indices.chunks_exact(3) {
    let i0 = chunk[0] as usize;
    let i1 = chunk[1] as usize;
    let i2 = chunk[2] as usize;
    let v0 = &comet.vertices[i0];
    let v1 = &comet.vertices[i1];
    let v2 = &comet.vertices[i2];

    let p0 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(v0.position);
    let p1 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(v1.position);
    let p2 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(v2.position);
    let area = 0.5 * (p1 - p0).cross(p2 - p1).length();

    let cu = (v0.uv[0] + v1.uv[0] + v2.uv[0]) / 3.0;
    let cv = (v0.uv[1] + v1.uv[1] + v2.uv[1]) / 3.0;

    let mut x = if cu < min_u {
      0
    } else {
      ((cu - min_u) * inv_w) as usize
    };
    let mut y = if cv < min_v {
      0
    } else {
      ((cv - min_v) * inv_h) as usize
    };
    if x >= resolution {
      x = resolution - 1;
    }
    if y >= resolution {
      y = resolution - 1;
    }

    weights[y * resolution + x] += area;
  }

  aethervk_core_rlib::math::distribution::Distribution2D::new(&weights, resolution, resolution)
}

fn main() {
  let mut event_loop_builder = EventLoopBuilder::<AppEvent>::with_user_event();
  #[cfg(target_os = "macos")]
  {
    use winit::platform::macos::EventLoopBuilderExtMacOS;
    event_loop_builder.with_default_menu(false);
  }

  let event_loop = event_loop_builder.build().unwrap();
  let asset_path = cycle_get_asset_path_from_exe(false);
  // 1. Spawn a scene with camera, sun, and a particle emitter component
  // 2. render_thread as usual receives render commands for everything every redraw request
  // 3. logic thread listens for input commands on the update, and on a fixed update function
  // dispatches particle update

  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_path.to_str().unwrap().to_string());
  drop(guard);

  let render_frontend = {
    let runtime_params = Box::new(RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
      validation_error_callback: Some(vulkan_rendering_validation_callback),
    });
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, &runtime_params).unwrap()
  };

  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = {
    let mut write_render_frontend = render_frontend.write();
    write_render_frontend
      .init_device(0, &additional_params)
      .unwrap()
  };

  let proxy = event_loop.create_proxy();
  let proxy_ptr = unsafe { std::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(proxy))) };

  let window = WindowBuilder::new()
    .with_title("Particle Test")
    .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
    .build(&event_loop)
    .unwrap();
  setup_resize_hook(&window, proxy_ptr);

  let (native_handles, _window_info) =
    get_handle_and_window_info(&render_frontend, render_device_handle, &window);

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let presentation_engine = {
    let params = gpu::PresentationEngineParams {
      width,
      height,
      vsync: true,
      ty: gpu::PresentationEngineType::Window,
      window_info: native_handles,
    };
    render_frontend
      .with_device(render_device_handle, |device| {
        let pe = device.create_presentation_engine(&params)?;
        Ok(pe)
      })
      .unwrap()
  };

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene
    .register_component::<PhysicalMeshComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene
    .register_component::<ParticleSystemComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[std::any::TypeId::of::<TransformComponent>()]);

  let camera_entity = scene.spawn_entity("camera");
  let cam_dist = 5.0;
  let cam_yaw: f32 = std::f32::consts::PI;
  let cam_pitch: f32 = 0.0;

  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_array([0.0, -cam_dist, 0.0]),
        rotation: Quat::from_axis_angle(Vec3f32::from_array([0.0, 0.0, 1.0]), std::f32::consts::PI),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(
          std::f32::consts::FRAC_PI_4,
          width as f32 / height as f32,
          0.1,
          100.0,
        ),
        near_plane: 0.1,
        far_plane: 100.0,
      },
    )
    .unwrap();

  let mesh_entity = scene.spawn_entity("mesh");
  scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();

  let loaded_mesh = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
    asset_path.join("Comet.glb").to_str().unwrap(),
    false,
  )
  .expect("Failed to load mesh");

  let uv_dist = build_uv_distribution(&loaded_mesh, 64);

  scene
    .add_component(
      mesh_entity,
      PhysicalMeshComponent {
        asset_path: asset_path.join("Comet.glb").to_str().unwrap().to_string(),
        mesh: Arc::from(loaded_mesh),
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
      },
    )
    .unwrap();

  let particle_sys_1 = scene.spawn_entity("particle_sys_1");
  scene
    .add_component(
      particle_sys_1,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      particle_sys_1,
      ParticleSystemComponent::new(ParticleEmitterConfig {
        uv_distribution: uv_dist.clone(),
        delta: 100_000,
        max_particles: 1000,
        velocity_intensity: GaussianParams {
          mean: 5.0,
          std_dev: 1.0,
          min: 0.0,
          max: 10.0,
        },
        emission_count: GaussianParams {
          mean: 10.0,
          std_dev: 2.0,
          min: 1.0,
          max: 20.0,
        },
        particle_radius: 0.1,
        density: 1000.0,
        lifetime: 5_000_000,
        color: [1.0, 0.5, 0.0, 1.0],
        beta: 0.1,
      }),
    )
    .unwrap();

  let particle_sys_2 = scene.spawn_entity("particle_sys_2");
  scene
    .add_component(
      particle_sys_2,
      TransformComponent {
        position: Vec3f32::from_array([0.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      particle_sys_2,
      ParticleSystemComponent::new(ParticleEmitterConfig {
        uv_distribution: uv_dist.clone(),
        delta: 100_000,
        max_particles: 500,
        velocity_intensity: GaussianParams {
          mean: 3.0,
          std_dev: 0.5,
          min: 0.0,
          max: 5.0,
        },
        emission_count: GaussianParams {
          mean: 5.0,
          std_dev: 1.0,
          min: 1.0,
          max: 10.0,
        },
        particle_radius: 0.05,
        density: 500.0,
        lifetime: 3_000_000,
        color: [0.0, 0.5, 1.0, 1.0],
        beta: 0.5,
      }),
    )
    .unwrap();

  let sun_entity = scene.spawn_entity("sun");
  scene
    .add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_array([100.0, 0.0, 0.0]),
        rotation: Quat::identity(),
        scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_entity,
      SunComponent {
        resolution: (64, 64, 64),
      },
    )
    .unwrap();

  let kernels = CpuKernels::new();

  render_frontend
    .with_device(render_device_handle, |device| {
      device.init_archetypes(presentation_engine)?;
      device.generate_sky()?;
      Ok(())
    })
    .unwrap();

  let app_state = AppState {
    is_resizing: false,
    is_exiting: false,
    scene: Arc::new(scene),
    window: Some(window),
  };

  let particle_app = ParticleApp {
    app_state,
    render_frontend,
    render_device_handle,
    presentation_engine: Some(presentation_engine),
    camera_entity,
    mesh_entity,
    sun_entity,
    right_mouse_button_down: false,
    cam_dist,
    cam_yaw,
    cam_pitch,
    width,
    height,
    window_info: _window_info,
    time_info: aethervk_oshal_rlib::os::time::TimeInfo::new(16_666, 100_000, 1.0),
    kernels,
  };

  test_utils::app::run_app(particle_app, event_loop);
}
