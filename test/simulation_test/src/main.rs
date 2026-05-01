use aethervk_core_rlib::{
  gpu::ASSET_DIR,
  scene::text::FontAtlas,
  gpu::{self},
  scene::{
    AlmanacPlanet, CameraComponent, CursorComponent, PhysicalMeshComponent, Scene,
    TransformComponent,
  },
  simulation as core_simulation, types,
  types::RuntimeParams,
};
use aethervk_oshal_rlib::{
  math::vector::Vector,
  math::{
    matrix::{mat4::Mat4x4f32, Matrix4},
    quaternion::Quaternion,
    vector::{vec3::Vec3f32, vec4::Quat, Vector3},
  },
};
use heapless::index_map::FnvIndexMap;
use logic_thread::{start_logic_thread, LogicCommand};
use render_thread::{start_render_thread, RenderPacket};
use std::{
  io::Read,
  sync::{atomic::AtomicBool, mpsc, Arc},
  time::Instant,
};
use std::any::TypeId;
use aethervk_core_rlib::scene::GizmoComponent;
use test_utils::{
  create_winit_window_and_event_loop, cycle_get_asset_path_from_exe, get_handle_and_window_info,
  get_monospace_font_path_from_asset_path, SceneTestUtilsExt,
};
use test_utils::simulation::kernels::CpuKernels;

mod constants;
mod logic_thread;
mod render_thread;
mod utils;

struct AppState {
  scene: Arc<Scene>,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: aethervk_core_rlib::scene::EntityId,
  is_paused: bool,
  time_scale: f32,
  root_entity: aethervk_core_rlib::scene::EntityId,
  window: Option<winit::window::Window>,
  is_resizing: bool,
  is_exiting: bool,
  outlines_enabled: Arc<AtomicBool>,
  is_command_prompt_open: bool,
  console_open_progress: f32,
  console_scroll_offset: usize,
  command_history: std::collections::VecDeque<String>,
  current_command: String,
}

impl Drop for AppState {
  fn drop(&mut self) {
    println!("Dropping AppState");
  }
}

impl core_simulation::Pausable for AppState {
  fn is_paused(&self) -> bool {
    self.is_paused
  }
  fn set_paused(&mut self) {
    self.is_paused = true;
  }
  fn time_scale(&self) -> f32 {
    self.time_scale
  }
  fn set_time_scale(&mut self, scale: f32) {
    self.time_scale = scale;
  }
}

fn add_gizmos_to_transforms(scene: &Scene) {
  let mut entities = Vec::new();
  scene.query1::<TransformComponent, _>(|id, _| {
    entities.push(id);
  });

  for id in entities {
    if scene.with_component(id, |_g: &GizmoComponent| {}).is_none() {
      let mut scale = 1.0;
      if let Some(t) = scene.global_transform(id) {
        scale = t.scale.x().max(t.scale.y()).max(t.scale.z()) * 2.0;
      }
      let _ = scene.add_component(
        id,
        GizmoComponent {
          gizmo_visible: false,
          gizmo_scale: scale,
        },
      );
    }
  }
}

fn panic_error_callback(msg: &str) {
  panic!("Vulkan Error: {}", msg);
}

fn main() {
  std::panic::set_hook(Box::new(|panic_info| {
    println!("CRASH DETECTED: {}", panic_info);
    println!("Press Enter to close the application...");
    let _ = std::io::stdin().read(&mut [0u8]);
  }));

  let assets_dir = cycle_get_asset_path_from_exe(true);
  let start_time = Instant::now();
  println!("[{:.2?}] Application starting.", start_time.elapsed());

  let (window, event_loop) = create_winit_window_and_event_loop("AetherVk Simulation");

  let render_frontend = {
    let runtime_params = Box::new(RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
      validation_error_callback: Some(panic_error_callback),
    });
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, &runtime_params).unwrap()
  };

  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .init_device(0, &additional_params)
    .unwrap();

  let (native_handles, window_info) =
    get_handle_and_window_info(&render_frontend, render_device_handle, &window);

  let (presentation_engine, font_id) = {
    let params = gpu::PresentationEngineParams {
      width: window.inner_size().width,
      height: window.inner_size().height,
      vsync: true,
      ty: gpu::PresentationEngineType::Window,
      window_info: native_handles,
    };
    render_frontend
      .with_device(render_device_handle, |device| {
        create_presentation_engine_and_init_archetypes(device, &params)
      })
      .unwrap()
  };
  println!(
    "[{:.2?}] GPU initialization complete.",
    start_time.elapsed()
  );

  let scene = Scene::new().with_all_dbg_components();
  scene.register_component::<AlmanacPlanet>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<GizmoComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<aethervk_core_rlib::scene::ParticleSystemComponent>(&[TypeId::of::<
    TransformComponent,
  >()]);
  let model_path = assets_dir.join("Comet.glb");
  let comet = core_simulation::comet::load_comet_from_gltf(model_path.to_str().unwrap(), false)
    .expect("Failed to load comet");

  let root_entity = scene.add_root_entity().unwrap();

  #[cfg(not(feature = "spotless_rendering"))]
  {
    let mesh_entity = scene.spawn_entity("comet");
    // The comet is very small physically. To make it visible, we need to significantly
    // boost its visual scale. Using a large constant for visibility.
    let comet_radius = 50.0;

    let initial_rotation = if let Some(ref axes) = comet.principal_axes {
      Quat::from_rotation_matrix(axes)
    } else {
      Quat::identity()
    };

    scene
      .add_component(
        mesh_entity,
        TransformComponent {
          position: Vec3f32::from_components(10.0, 0.0, 0.0),
          rotation: initial_rotation,
          scale: Vec3f32::from_components(comet_radius, comet_radius, comet_radius),
        },
      )
      .unwrap();
    scene
      .add_component(
        mesh_entity,
        PhysicalMeshComponent {
          asset_path: "".to_string(),
          mesh: Arc::from(comet),
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
        },
      )
      .unwrap();
    scene.set_parent(mesh_entity, Some(root_entity));

    let uv_dist = utils::generate_gaussian_distribution(64, 0.5, 0.5, 0.5, 0.5);
    scene
      .add_component(
        mesh_entity,
        aethervk_core_rlib::scene::ParticleSystemComponent::new(
          aethervk_core_rlib::scene::ParticleEmitterConfig {
            uv_distribution: uv_dist,
            delta: 100_000,
            max_particles: 100000,
            velocity_intensity: aethervk_core_rlib::scene::GaussianParams {
              mean: 0.5,
              std_dev: 0.1,
              min: 0.0,
              max: 1.0,
            },
            emission_count: aethervk_core_rlib::scene::GaussianParams {
              mean: 100.0,
              std_dev: 20.0,
              min: 10.0,
              max: 200.0,
            },
            particle_radius: 1.0,
            density: 1000.0,
            lifetime: 5_000_000,
            color: [1.0, 0.5, 0.0, 1.0],
            beta: 0.1,
          },
        ),
      )
      .unwrap();
  }

  #[cfg(not(feature = "spotless_rendering"))]
  let planets = [
    (
      "Mercury",
      "planets/textures/Mercury.jpg",
      crate::constants::PlanetNaifId::MERCURY,
      1407.6,
    ),
    (
      "Venus",
      "planets/textures/Venus.jpg",
      crate::constants::PlanetNaifId::VENUS,
      -5832.6,
    ),
    (
      "Earth",
      "planets/textures/Earth.jpg",
      crate::constants::PlanetNaifId::EARTH,
      23.93,
    ),
    (
      "Mars",
      "planets/textures/Mars.jpg",
      crate::constants::PlanetNaifId::MARS,
      24.62,
    ),
    (
      "Jupiter",
      "planets/textures/Jupiter.jpg",
      crate::constants::PlanetNaifId::JUPITER,
      9.92,
    ),
    (
      "Saturn",
      "planets/textures/Saturn.jpg",
      crate::constants::PlanetNaifId::SATURN,
      10.65,
    ),
    (
      "Uranus",
      "planets/textures/Uranus.jpg",
      crate::constants::PlanetNaifId::URANUS,
      -17.24,
    ),
    (
      "Neptune",
      "planets/textures/Neptune.jpg",
      crate::constants::PlanetNaifId::NEPTUNE,
      16.11,
    ),
  ];

  let mut planets_ids = Vec::new();

  #[cfg(not(feature = "spotless_rendering"))]
  for (name, tex_path, naif_id, rot_period) in planets.iter() {
    let planet_radius = (utils::get_planet_radius(*naif_id, &assets_dir)
      / constants::DISTANCE_SCALE_FACTOR as f32)
      * constants::PLANET_VISUAL_SCALE;
    let initial_pos = Vec3f32::zero();

    let sphere = {
      let mut sphere = core_simulation::comet::generate_uv_sphere(planet_radius, 64, 64);
      let tex =
        core_simulation::comet::load_texture_from_file(assets_dir.join(tex_path).to_str().unwrap())
          .expect(&format!("Failed to load texture for {}", name));
      sphere.albedo_map = Some(tex);
      Arc::from(sphere)
    };
    let planet_entity = scene
      .add_mesh(*name, root_entity)
      .with_position(initial_pos)
      .with_mesh("", sphere)
      .build()
      .unwrap();
    scene
      .add_component(planet_entity, AlmanacPlanet::new(*naif_id, *rot_period))
      .unwrap();
    scene
      .add_component(
        planet_entity,
        GizmoComponent {
          gizmo_visible: false,
          gizmo_scale: planet_radius * 2.0,
        },
      )
      .unwrap();
    planets_ids.push((*naif_id, planet_entity, *rot_period, planet_radius));
  }

  let cursor_entity = scene.spawn_entity("entity");
  scene
    .add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(cursor_entity, CursorComponent {})
    .unwrap();
  scene.set_parent(cursor_entity, Some(root_entity));

  let camera_entity = scene.spawn_entity("entity");
  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, -400.0, 0.0),
        rotation: Quat::from_axis_angle(
          Vec3f32::from_components(0.0, 0.0, 1.0),
          std::f32::consts::PI,
        ),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(
          45.0f32.to_radians(),
          800.0 / 600.0, // Default aspect ratio, will be updated by resize
          0.1,
          1000000.0,
        ),
        near_plane: 0.1,
        far_plane: 1000000.0,
      },
    )
    .unwrap();
  scene.set_parent(camera_entity, Some(root_entity));

  let sun_entity = scene.spawn_entity("sun");
  let sun_radius = (utils::get_planet_radius(constants::PlanetNaifId::SUN, &assets_dir)
    / constants::DISTANCE_SCALE_FACTOR as f32)
    * constants::UNIVERSAL_VISUAL_SCALE;
  let sun_scale = sun_radius / 0.45;
  let sun_pos = Vec3f32::zero();

  scene
    .add_component(
      sun_entity,
      TransformComponent {
        position: sun_pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(sun_scale, sun_scale, sun_scale),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_entity,
      aethervk_core_rlib::scene::SunComponent {
        resolution: (128, 128, 128),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_entity,
      AlmanacPlanet::new(constants::PlanetNaifId::SUN, 25.05),
    )
    .unwrap();

  // Add emissive core for the sun
  let sun_core_entity = scene.spawn_entity("sun_core");
  let mut sun_sphere = core_simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64);
  sun_sphere.albedo_map = None;
  scene
    .add_component(
      sun_core_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_core_entity,
      PhysicalMeshComponent {
        asset_path: "".to_string(),
        mesh: Arc::from(sun_sphere),
        emissive_intensity: 0.9, // Reduced to prevent SDR whiteout clamp
        emissive_color: [1.0, 0.35, 0.02], // Pure rich orange/red
      },
    )
    .unwrap();
  scene.set_parent(sun_core_entity, Some(sun_entity));

  let sky_entity = scene.spawn_entity("entity");
  scene
    .add_component(sky_entity, aethervk_core_rlib::scene::SkyComponent {})
    .unwrap();

  let grid_entity = scene.spawn_entity("grid");
  scene
    .add_component(grid_entity, aethervk_core_rlib::scene::GridComponent {})
    .unwrap();
  scene.set_parent(sun_entity, Some(root_entity));

  add_gizmos_to_transforms(&scene);

  let scene_shared = Arc::new(scene);
  let outlines_enabled = Arc::new(AtomicBool::new(true));

  let mut app_state = AppState {
    scene: Arc::clone(&scene_shared),
    presentation_engine,
    camera_entity,
    is_paused: false,
    time_scale: 1.0,
    root_entity,
    is_resizing: false,
    is_exiting: false,
    window: Some(window),
    outlines_enabled: Arc::clone(&outlines_enabled),
    is_command_prompt_open: false,
    console_open_progress: 0.0,
    console_scroll_offset: 0,
    command_history: std::collections::VecDeque::with_capacity(1000),
    current_command: String::new(),
  };

  // --- Start Render Thread ---
  let (render_tx, render_rx) = mpsc::sync_channel::<Option<RenderPacket>>(1);
  let render_thread_handle = start_render_thread(
    render_rx,
    Arc::clone(&scene_shared),
    render_frontend,
    render_device_handle,
    presentation_engine,
    font_id,
  );
  let cpu_kernels = CpuKernels::new();

  // --- Start Logic Thread ---
  let (logic_tx, logic_rx) = mpsc::channel::<LogicCommand>();
  let (response_tx, response_rx) = mpsc::channel::<String>();
  let logic_thread_handle = start_logic_thread(
    logic_rx,
    response_tx,
    Arc::clone(&scene_shared),
    root_entity,
    camera_entity,
    cursor_entity,
    grid_entity,
    planets_ids,
    assets_dir,
    Arc::clone(&outlines_enabled),
    cpu_kernels,
  );

  let _app_threads = test_utils::threading::AppThreads {
    logic_thread: Some(logic_thread_handle),
    render_thread: Some(render_thread_handle),
  };

  let mut initial_width = app_state.window.as_ref().unwrap().inner_size().width;
  let mut initial_height = app_state.window.as_ref().unwrap().inner_size().height;
  if initial_width == 0 {
    initial_width = 800;
  }
  if initial_height == 0 {
    initial_height = 600;
  }

  let _ = logic_tx.send(LogicCommand::Resize {
    width: initial_width,
    height: initial_height,
  });

  // --- Main Event Loop ---

  let sim_app = SimApp {
    app_state,
    logic_tx,
    render_tx,
    response_rx,
    right_mouse_button_down: false,
    middle_mouse_button_down: false,
    ctrl_down: false,
    mouse_x: 0.0,
    mouse_y: 0.0,
    last_log_time: std::time::Instant::now(),
    _app_threads,
    window_info,
  };

  test_utils::app::run_app(sim_app, event_loop);
  println!("Event loop returned. AppThreads will join threads on drop. Exiting main().");
}

fn create_presentation_engine_and_init_archetypes(
  device: &dyn gpu::RenderDevice,
  params: &gpu::PresentationEngineParams,
) -> types::GpuResult<(gpu::PresentationEngineHandle, (u64, u32))> {
  let asset_dir = ASSET_DIR.read();
  let mono_font = get_monospace_font_path_from_asset_path(asset_dir.as_ref().unwrap());
  assert!(mono_font.is_file());
  let mono_font = FontAtlas::from_path(mono_font.to_str().unwrap(), 12.0).unwrap();
  let pe = device.create_presentation_engine(params)?;
  device.init_archetypes(pe)?;
  device.generate_sky()?;
  let mono_font_hash = mono_font.hash_metadata();
  let mono_font_id = device.allocate_rasterized_font_atlas(mono_font_hash, mono_font)?;
  Ok((pe, (mono_font_hash, mono_font_id)))
}

struct SimApp {
  app_state: AppState,
  logic_tx: std::sync::mpsc::Sender<LogicCommand>,
  render_tx: std::sync::mpsc::SyncSender<Option<RenderPacket>>,
  response_rx: std::sync::mpsc::Receiver<String>,
  right_mouse_button_down: bool,
  middle_mouse_button_down: bool,
  ctrl_down: bool,
  mouse_x: f64,
  mouse_y: f64,
  last_log_time: std::time::Instant,
  _app_threads: test_utils::threading::AppThreads,
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
    let _ = self.logic_tx.send(LogicCommand::Resize { width, height });
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
  }

  fn on_close_requested(&mut self) {
    self.app_state.window = None;
    let _ = self.logic_tx.send(LogicCommand::Exit);
    let _ = self.render_tx.try_send(None);
  }

  fn on_mouse_input(
    &mut self,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
  ) {
    match button {
      winit::event::MouseButton::Right => {
        self.right_mouse_button_down = state == winit::event::ElementState::Pressed
      }
      winit::event::MouseButton::Middle => {
        self.middle_mouse_button_down = state == winit::event::ElementState::Pressed
      }
      winit::event::MouseButton::Left => {
        if state == winit::event::ElementState::Pressed {
          let size = self.app_state.window.as_ref().unwrap().inner_size();
          if size.width > 0 && size.height > 0 {
            let ndc_x = (self.mouse_x as f32 / size.width as f32) * 2.0 - 1.0;
            let ndc_y = (self.mouse_y as f32 / size.height as f32) * 2.0 - 1.0;
            let _ = self
              .logic_tx
              .send(LogicCommand::RaycastCursor { ndc_x, ndc_y });
            if let Some(w) = self.app_state.window.as_ref() {
              w.request_redraw();
            }
          }
        }
      }
      _ => {}
    }
  }

  fn on_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
    self.mouse_x = position.x;
    self.mouse_y = position.y;
  }

  fn on_keyboard_input(
    &mut self,
    event: &winit::event::KeyEvent,
    modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state == winit::event::ElementState::Pressed {
      if self.app_state.is_command_prompt_open {
        if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) =
          event.physical_key
        {
          self.app_state.is_command_prompt_open = false;
          if let Some(w) = self.app_state.window.as_ref() {
            w.request_redraw();
          }
        } else {
          match &event.logical_key {
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace) => {
              self.app_state.current_command.pop();
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageUp) => {
              self.app_state.console_scroll_offset =
                self.app_state.console_scroll_offset.saturating_add(1);
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageDown) => {
              self.app_state.console_scroll_offset =
                self.app_state.console_scroll_offset.saturating_sub(1);
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) => {
              if !self.app_state.current_command.is_empty() {
                let cmd = self.app_state.current_command.clone();
                self
                  .app_state
                  .command_history
                  .push_back(format!("> {}", cmd));
                let _ = self.logic_tx.send(LogicCommand::ExecuteCommand(cmd));
                if self.app_state.command_history.len() > 1000 {
                  self.app_state.command_history.pop_front();
                }
                self.app_state.current_command.clear();
                self.app_state.console_scroll_offset = 0;
              }
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
              self.app_state.current_command.push(' ');
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            winit::keyboard::Key::Character(c) => {
              self.app_state.current_command.push_str(c.as_str());
              if let Some(w) = self.app_state.window.as_ref() {
                w.request_redraw();
              }
            }
            _ => {}
          }
        }
      } else {
        if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
          let speed = 0.5;
          #[cfg(target_os = "macos")]
          if keycode == winit::keyboard::KeyCode::KeyQ && modifiers.super_key() {
            self.app_state.is_exiting = true;
            self.on_close_requested();
            println!("You Clicked exit");
            return;
          }

          if let Some(axis) = test_utils::command::get_camera_movement_axis(keycode) {
            let _ = self.logic_tx.send(LogicCommand::MoveCursor {
              axis,
              amount: speed,
            });
            if let Some(w) = self.app_state.window.as_ref() {
              w.request_redraw();
            }
          } else {
            match keycode {
              winit::keyboard::KeyCode::KeyM => {
                self.app_state.is_command_prompt_open = true;
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyX => {
                let _ = self.logic_tx.send(LogicCommand::CycleTimeScale);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyA => {
                let _ = self
                  .logic_tx
                  .send(LogicCommand::CyclePlanet { forward: false });
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyD => {
                let _ = self
                  .logic_tx
                  .send(LogicCommand::CyclePlanet { forward: true });
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyG => {
                let _ = self.logic_tx.send(LogicCommand::ToggleGrid);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyH => {
                let _ = self.logic_tx.send(LogicCommand::TogglePlanetOutlines);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyT => {
                let _ = self.logic_tx.send(LogicCommand::ResetCamera);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::Digit0 | winit::keyboard::KeyCode::Numpad0 => {
                let _ = self.logic_tx.send(LogicCommand::ResetCursor);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::Digit1 | winit::keyboard::KeyCode::Numpad1 => {
                let _ = self.logic_tx.send(LogicCommand::SnapCursorToSun);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::Digit2 | winit::keyboard::KeyCode::Numpad2 => {
                let _ = self.logic_tx.send(LogicCommand::SnapCameraToCursor);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              winit::keyboard::KeyCode::KeyV => {
                let _ = self.logic_tx.send(LogicCommand::ToggleMeasureTool);
                if let Some(w) = self.app_state.window.as_ref() {
                  w.request_redraw();
                }
              }
              _ => {}
            }
          }
        }
      }
    }
  }

  fn on_mouse_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
    let scroll_amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32,
    };
    if self.app_state.is_command_prompt_open {
      if scroll_amount > 0.0 {
        self.app_state.console_scroll_offset =
          self.app_state.console_scroll_offset.saturating_add(1);
      } else if scroll_amount < 0.0 {
        self.app_state.console_scroll_offset =
          self.app_state.console_scroll_offset.saturating_sub(1);
      }
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }
  }

  fn on_modifiers_changed(&mut self, modifiers: winit::keyboard::ModifiersState) {
    self.ctrl_down = modifiers.control_key() || modifiers.super_key();
  }

  fn on_mouse_motion(&mut self, delta: (f64, f64)) {
    if self.right_mouse_button_down {
      if self.ctrl_down {
        let _ = self.logic_tx.send(LogicCommand::ZoomCamera {
          amount: delta.1 as f32,
        });
      } else {
        let _ = self.logic_tx.send(LogicCommand::RotateCamera {
          delta_x: delta.0 as f32,
          delta_y: delta.1 as f32,
        });
      }
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    } else if self.middle_mouse_button_down {
      let _ = self.logic_tx.send(LogicCommand::PanCursor {
        delta_x: delta.0 as f32,
        delta_y: delta.1 as f32,
      });
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }
  }

  fn on_redraw(&mut self) {
    let mut show_particle_count = false;
    let mut particle_count = 0;

    let _ = self.app_state.scene.with_component(
      self.app_state.camera_entity,
      |_c: &logic_thread::ParticleCountDisplayComponent| {
        show_particle_count = true;
      },
    );

    if show_particle_count {
      self
        .app_state
        .scene
        .query1::<aethervk_core_rlib::scene::ParticleSystemComponent, _>(|_, sys| {
          particle_count += sys.particles.len();
        });
    }

    let packet = RenderPacket {
      camera_entity: self.app_state.camera_entity,
      window_size: self.app_state.window.as_ref().unwrap().inner_size(),
      outlines_enabled: self
        .app_state
        .outlines_enabled
        .load(std::sync::atomic::Ordering::Relaxed),
      is_command_prompt_open: self.app_state.is_command_prompt_open,
      console_open_progress: self.app_state.console_open_progress,
      console_scroll_offset: self.app_state.console_scroll_offset,
      command_history: self.app_state.command_history.clone(),
      current_command: self.app_state.current_command.clone(),
      show_particle_count,
      particle_count,
    };

    match self.render_tx.try_send(Some(packet)) {
      Ok(_) => {}
      Err(mpsc::TrySendError::Full(_)) => {}
      Err(mpsc::TrySendError::Disconnected(_)) => {
        self.app_state.is_exiting = true;
      }
    }
  }

  fn on_about_to_wait(&mut self) {
    let dt = 0.016;
    if self.app_state.is_command_prompt_open {
      self.app_state.console_open_progress += dt * 5.0;
      if self.app_state.console_open_progress > 1.0 {
        self.app_state.console_open_progress = 1.0;
      } else {
        if let Some(w) = self.app_state.window.as_ref() {
          w.request_redraw();
        }
      }
    } else {
      self.app_state.console_open_progress -= dt * 5.0;
      if self.app_state.console_open_progress < 0.0 {
        self.app_state.console_open_progress = 0.0;
      } else {
        if let Some(w) = self.app_state.window.as_ref() {
          w.request_redraw();
        }
      }
    }

    let mut got_responses = false;
    while let Ok(response) = self.response_rx.try_recv() {
      if response == "___CLEAR___" {
        self.app_state.command_history.clear();
      } else {
        self.app_state.command_history.push_back(response);
        if self.app_state.command_history.len() > 1000 {
          self.app_state.command_history.pop_front();
        }
      }
      got_responses = true;
    }
    if got_responses && self.app_state.is_command_prompt_open {
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }

    if self.last_log_time.elapsed().as_secs() >= 5 {
      self.last_log_time = std::time::Instant::now();
    }
    if !self.app_state.is_resizing {
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }
  }
}
