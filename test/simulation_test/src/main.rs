use aethervk_core_rlib::{
  scene::GizmoComponent,
  simulation::constants,
  simulation_api::{structs, SimulationContext},
  gpu::{self},
  scene::{AlmanacPlanet, PhysicalMeshComponent, TransformComponent},
};
use aethervk_oshal_rlib::{
  math::vector::Vector,
  math::{
    quaternion::Quaternion,
    vector::{vec3::Vec3f32, vec4::Quat, Vector3},
  },
};
use std::{sync::Arc, time::Instant};
use std::sync::atomic::AtomicBool;
use winit::keyboard::KeyCode;
use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::text::FontAtlas;
use aethervk_core_rlib::simulation_api::structs::{CustomRenderCallback, SendPtrMut};
use aethervk_core_rlib::types::GpuResult;
use aethervk_core_rlib::simulation_api::core_api::SimulationContextExt;
use test_utils::{
  create_winit_window_and_event_loop, cycle_get_asset_path_from_exe, get_handle_and_window_info,
};

struct AppState {
  ctx: Box<SimulationContext>,
  custom_data_ring: [CustomRenderData; 3],
  scene_id: u64,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: u64,
  window: Option<winit::window::Window>,
  is_resizing: bool,
  is_exiting: bool,
  is_command_prompt_open: bool,
  font_atlas: Arc<std::sync::Mutex<Option<FontAtlas>>>, // uploaded then dropped
  // TODO reuse it as before in rendering function, which therefore should be customizable in simulation context
  console_open_progress: f32,
  console_scroll_offset: usize,
  command_history: std::collections::VecDeque<String>,
  current_command: String,
  current_epoch: anise::time::Epoch,
  step_days: f64,
}

impl AppState {
  fn cycle_first_free_render_custom_data(&mut self) -> &'_ mut CustomRenderData {
    let mut index: usize = 0;
    loop {
      if self.custom_data_ring[index].is_free_relaxed() {
        let _ = self.custom_data_ring[index].is_free_acquire();
        return unsafe { self.custom_data_ring.get_unchecked_mut(index) };
      }
      index = (index + 1) % self.custom_data_ring.len();
      std::thread::sleep(std::time::Duration::from_millis(1));
    }
  }
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
  // std::panic::set_hook(Box::new(|panic_info| {
  //   println!("CRASH DETECTED: {}", panic_info);
  //   println!("Press Enter to close the application...");
  //   let mut buf = [0u8; 1];
  //   let _ = std::io::Read::read(&mut std::io::stdin(), &mut buf);
  // }));

  let assets_dir = cycle_get_asset_path_from_exe(true);
  let start_time = Instant::now();
  println!("[{:.2?}] Application starting.", start_time.elapsed());

  let (window, event_loop) = create_winit_window_and_event_loop("AetherVk Simulation");

  let simulation_context =
    SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_error_callback)).unwrap();

  let render_frontend = simulation_context.render_frontend().unwrap();
  let render_device_handle = simulation_context.render_device_handle();
  let (native_handles, window_info) =
    get_handle_and_window_info(&render_frontend, render_device_handle, &window);

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let presentation_engine = simulation_context
    .create_presentation_engine_windowed(width, height, native_handles)
    .unwrap();

  let scene_id = simulation_context.create_default_scene().unwrap();

  // Populate scene with custom planets and comet
  {
    let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
    let mut active_scene = scene_ctx.write();
    let root_entity = active_scene.root_entity;

    let model_path = assets_dir.join("Comet.glb");
    let comet = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      model_path.to_str().unwrap(),
      false,
    )
    .expect("Failed to load comet");

    let mesh_entity = active_scene.scene.spawn_entity("comet");
    // Scale comet so its size represents 1.7km (radius ~0.85km).
    let comet_radius = (0.85
      / aethervk_core_rlib::simulation::constants::DISTANCE_SCALE_FACTOR as f32)
      * aethervk_core_rlib::simulation::constants::PLANET_VISUAL_SCALE;

    let initial_rotation = if let Some(ref axes) = comet.principal_axes {
      Quat::from_rotation_matrix(axes)
    } else {
      Quat::identity()
    };

    active_scene
      .scene
      .add_component(
        mesh_entity,
        TransformComponent {
          position: Vec3f32::from_components(10.0, 0.0, 0.0),
          rotation: initial_rotation,
          scale: Vec3f32::from_components(comet_radius, comet_radius, comet_radius),
        },
      )
      .unwrap();
    active_scene
      .scene
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
    active_scene
      .scene
      .set_parent(mesh_entity, Some(root_entity));
    active_scene.register_entity(mesh_entity);

    let uv_dist =
      aethervk_core_rlib::simulation::utils::generate_gaussian_distribution(64, 0.5, 0.5, 0.5, 0.5);
    active_scene
      .scene
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

    let planets = [
      (
        "Mercury",
        "planets/textures/Mercury.jpg",
        anise::constants::celestial_objects::MERCURY,
        1407.6,
      ),
      (
        "Venus",
        "planets/textures/Venus.jpg",
        anise::constants::celestial_objects::VENUS,
        -5832.6,
      ),
      (
        "Earth",
        "planets/textures/Earth.jpg",
        anise::constants::celestial_objects::EARTH,
        23.93,
      ),
      (
        "Mars",
        "planets/textures/Mars.jpg",
        anise::constants::celestial_objects::MARS,
        24.62,
      ),
      (
        "Jupiter",
        "planets/textures/Jupiter.jpg",
        anise::constants::celestial_objects::JUPITER,
        9.92,
      ),
      (
        "Saturn",
        "planets/textures/Saturn.jpg",
        anise::constants::celestial_objects::SATURN,
        10.65,
      ),
      (
        "Uranus",
        "planets/textures/Uranus.jpg",
        anise::constants::celestial_objects::URANUS,
        -17.24,
      ),
      (
        "Neptune",
        "planets/textures/Neptune.jpg",
        anise::constants::celestial_objects::NEPTUNE,
        16.11,
      ),
    ];

    for (name, tex_path, naif_id, rot_period) in planets.iter() {
      let planet_radius = (aethervk_core_rlib::simulation::utils::get_planet_radius(
        *naif_id,
        assets_dir.to_str().unwrap(),
      ) / constants::DISTANCE_SCALE_FACTOR as f32)
        * constants::PLANET_VISUAL_SCALE;
      let initial_pos = Vec3f32::zero();

      let sphere = {
        let mut sphere =
          aethervk_core_rlib::simulation::comet::generate_uv_sphere(planet_radius, 64, 64);
        let tex = aethervk_core_rlib::simulation::comet::load_texture_from_file(
          assets_dir.join(tex_path).to_str().unwrap(),
        )
        .expect(&format!("Failed to load texture for {}", name));
        sphere.albedo_map = Some(tex);
        Arc::from(sphere)
      };

      let planet_entity = active_scene.scene.spawn_entity(*name);
      active_scene
        .scene
        .set_parent(planet_entity, Some(root_entity));
      active_scene
        .scene
        .add_component(
          planet_entity,
          TransformComponent {
            position: initial_pos,
            rotation: Quat::identity(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
          },
        )
        .unwrap();
      active_scene
        .scene
        .add_component(
          planet_entity,
          PhysicalMeshComponent {
            asset_path: "".to_string(),
            mesh: sphere,
            emissive_intensity: 0.0,
            emissive_color: [0.0, 0.0, 0.0],
          },
        )
        .unwrap();
      active_scene
        .scene
        .add_component(planet_entity, AlmanacPlanet::new(*naif_id, *rot_period))
        .unwrap();
      active_scene
        .scene
        .add_component(
          planet_entity,
          GizmoComponent {
            gizmo_visible: false,
            gizmo_scale: planet_radius * 2.0,
          },
        )
        .unwrap();
      active_scene.register_entity(planet_entity);
    }
  }

  let ext_camera_entity = {
    let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
    let read_ctx = scene_ctx.read();
    let int_cam = read_ctx.active_camera_entity.unwrap();
    read_ctx
      .entity_map
      .iter()
      .find(|&(_, &v)| v == int_cam)
      .map(|(&k, _)| k)
      .unwrap()
  };

  let font_path = assets_dir.join("fonts/JetBrainsMono-Regular.ttf");
  let atlas =
    aethervk_core_rlib::scene::text::FontAtlas::from_path(font_path.to_str().unwrap(), 32.0)
      .expect("Failed to load asset font");

  let almanac_path = assets_dir.join("planets");
  simulation_context
    .threads
    .logic_thread
    .tx()
    .try_send(structs::LogicCommand::LoadAlmanac {
      task_id: 0, // TODO when switching to async API
      path: almanac_path.to_str().unwrap().to_string(),
    })
    .unwrap();

  use core::str::FromStr;
  let current_epoch = anise::time::Epoch::from_str("2024-03-24 12:00:00 TDB").unwrap();

  let app_state = AppState {
    ctx: simulation_context,
    custom_data_ring: [
      CustomRenderData::default(),
      CustomRenderData::default(),
      CustomRenderData::default(),
    ],
    scene_id,
    presentation_engine,
    camera_entity: ext_camera_entity,
    window: Some(window),
    is_resizing: false,
    is_exiting: false,
    is_command_prompt_open: false,
    font_atlas: Arc::new(std::sync::Mutex::new(Some(atlas))),
    console_open_progress: 0.0,
    console_scroll_offset: 0,
    command_history: std::collections::VecDeque::with_capacity(1000),
    current_command: String::new(),
    current_epoch,
    step_days: 0.016,
  };

  let sim_app = SimApp {
    app_state,
    right_mouse_button_down: false,
    middle_mouse_button_down: false,
    ctrl_down: false,
    mouse_x: 0.0,
    mouse_y: 0.0,
    last_log_time: std::time::Instant::now(),
    last_sim_time: std::time::Instant::now(),
    window_info,
    last_frame_start_time: std::time::Instant::now(),
    last_sim_tick_task: None,
    last_render_tick_task: None,
  };

  test_utils::app::run_app(sim_app, event_loop);
  println!("Event loop returned. Exiting main().");
}

struct SimApp {
  app_state: AppState,
  right_mouse_button_down: bool,
  middle_mouse_button_down: bool,
  ctrl_down: bool,
  mouse_x: f64,
  mouse_y: f64,
  last_log_time: std::time::Instant,
  last_sim_time: std::time::Instant,
  window_info: test_utils::WindowPlatformData,
  last_frame_start_time: std::time::Instant,
  last_sim_tick_task: Option<core::num::NonZero<u64>>,
  last_render_tick_task: Option<core::num::NonZero<u64>>,
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
            // TODO select intersected object (insert selected component, remove the other selected component if present. Add an extension method to scene in rlib to handle this)
            let _ = self
              .app_state
              .ctx
              .raycast_ndc(self.app_state.scene_id, ndc_x, ndc_y);
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
        } else {
          match &event.logical_key {
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace) => {
              self.app_state.current_command.pop();
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageUp) => {
              self.app_state.console_scroll_offset =
                self.app_state.console_scroll_offset.saturating_add(1);
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageDown) => {
              self.app_state.console_scroll_offset =
                self.app_state.console_scroll_offset.saturating_sub(1);
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) => {
              if !self.app_state.current_command.is_empty() {
                let cmd = self.app_state.current_command.clone();
                self
                  .app_state
                  .command_history
                  .push_back(format!("> {}", cmd));

                // TODO: Execute Command functionality
                // Currently omitted as it requires C FFI or direct parsing implementation in main.rs

                if self.app_state.command_history.len() > 1000 {
                  self.app_state.command_history.pop_front();
                }
                self.app_state.current_command.clear();
                self.app_state.console_scroll_offset = 0;
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
              self.app_state.current_command.push(' ');
            }
            winit::keyboard::Key::Character(c) => {
              self.app_state.current_command.push_str(c.as_str());
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
            let scene = self
              .app_state
              .ctx
              .get_scene(self.app_state.scene_id)
              .unwrap();
            // TODO check success
            let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
              structs::LogicCommand::MoveCursor(structs::MoveCursor {
                scene,
                delta_x: axis.x() * speed,
                delta_y: axis.y() * speed,
                delta_z: axis.z() * speed,
              }),
            );
          } else {
            match keycode {
              winit::keyboard::KeyCode::KeyM => {
                self.app_state.is_command_prompt_open = true;
              }
              winit::keyboard::KeyCode::KeyH => {
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  let current = read_ctx
                    .outlines_enabled
                    .load(std::sync::atomic::Ordering::Relaxed);
                  read_ctx
                    .outlines_enabled
                    .store(!current, std::sync::atomic::Ordering::Relaxed);
                }
              }
              winit::keyboard::KeyCode::KeyA | winit::keyboard::KeyCode::KeyD => {
                let forward = keycode == winit::keyboard::KeyCode::KeyD;
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  let mut entities: Vec<aethervk_core_rlib::scene::EntityId> = Vec::new();

                  read_ctx
                    .scene
                    .query1_first_res::<aethervk_core_rlib::scene::PhysicalMeshComponent, _, _>(
                      |id, _| {
                        entities.push(id);
                        None::<()>
                      },
                    );

                  entities.sort_by_key(|e| *e);

                  if !entities.is_empty() {
                    let mut current_idx = 0;
                    read_ctx
                      .scene
                      .query1_first_res::<aethervk_core_rlib::scene::FollowingComponent, _, _>(
                        |id, _| {
                          if let Ok(idx) = entities.binary_search_by_key(&id, |e| *e) {
                            current_idx = idx;
                          }
                          Some(())
                        },
                      );

                    let next_idx = if forward {
                      (current_idx + 1) % entities.len()
                    } else {
                      if current_idx == 0 {
                        entities.len() - 1
                      } else {
                        current_idx - 1
                      }
                    };

                    let target_entity = entities[next_idx];
                    if let Some((snap_entity, _)) = read_ctx
                      .scene
                      .query1_first_res::<aethervk_core_rlib::scene::CursorComponent, _, _>(
                      |id, _| Some(id),
                    ) {
                      let scene = scene_ctx.clone();
                      let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                        aethervk_core_rlib::simulation_api::structs::LogicCommand::FollowEntity(
                          aethervk_core_rlib::simulation_api::structs::FollowEntity {
                            snap_entity,
                            entity_id: target_entity,
                            scene,
                            unfollow_other: true,
                          },
                        ),
                      );
                    }
                  }
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
    }
  }

  fn on_mouse_motion(&mut self, delta: (f64, f64)) {
    let ctx = &self.app_state.ctx;
    let scene = ctx.get_scene(self.app_state.scene_id).unwrap();
    let camera_entity = scene
      .read()
      .get_entity(self.app_state.camera_entity)
      .unwrap();

    let logic_command = if self.right_mouse_button_down {
      Some(structs::LogicCommand::RotateCamera(structs::RotateCamera {
        camera_entity,
        scene: scene.clone(),
        delta_x: delta.0 as f32,
        delta_y: delta.1 as f32,
      }))
    } else if self.middle_mouse_button_down {
      if self.ctrl_down {
        Some(structs::LogicCommand::ZoomCamera(structs::ZoomCamera {
          camera_entity,
          scene: scene.clone(),
          amount: delta.1 as f32,
        }))
      } else {
        Some(structs::LogicCommand::PanCursor(structs::PanCursor {
          scene: scene.clone(),
          delta_x: delta.0 as f32,
          delta_y: delta.1 as f32,
        }))
      }
    } else {
      None
    };

    // TODO proper feedback
    if let Some(logic_command) = logic_command {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }

  fn on_modifiers_changed(&mut self, modifiers: winit::keyboard::ModifiersState) {
    self.ctrl_down = modifiers.control_key() || modifiers.super_key();
  }

  fn on_redraw(&mut self) {
    let size = self.app_state.window.as_ref().unwrap().inner_size();
    let console_open_progress = self.app_state.console_open_progress;
    let console_scroll_offset = self.app_state.console_scroll_offset;
    let command_history = self.app_state.command_history.clone();
    let current_command = self.app_state.current_command.clone();

    let callback = {
      let atlas = { self.app_state.font_atlas.lock().unwrap().take() };
      let current_custom_data = self.app_state.cycle_first_free_render_custom_data();
      assert_eq!(
        current_custom_data
          .mt_sent_not_received
          .load(std::sync::atomic::Ordering::Relaxed),
        false
      );

      current_custom_data.console_open_progress = console_open_progress;
      current_custom_data.console_scroll_offset = console_scroll_offset;
      current_custom_data.command_history = command_history;
      current_custom_data.current_command = current_command;
      current_custom_data.font_atlas = atlas;
      current_custom_data.size = size;
      current_custom_data
        .mt_sent_not_received
        .store(true, std::sync::atomic::Ordering::Relaxed);

      current_custom_data.make_callback()
    };

    let ctx = &self.app_state.ctx;
    if let Ok(task) = ctx.render_tick_custom(
      self.app_state.presentation_engine,
      self.app_state.scene_id,
      [size.width, size.height],
      Some(callback),
    ) {
      self.last_render_tick_task = Some(task);
    }
  }

  fn on_about_to_wait(&mut self) {
    let time_readings = self.app_state.ctx.time_info.read().current();
    self.app_state.ctx.govern_frame_rate_and_tasks(
      &mut self.last_sim_tick_task,
      &mut self.last_render_tick_task,
      time_readings,
      16_667,
    );

    let current_time = std::time::Instant::now();
    let delta_time = current_time
      .duration_since(self.last_sim_time)
      .as_secs_f64();
    self.last_sim_time = current_time;

    let dt = delta_time as f32;
    if self.app_state.is_command_prompt_open {
      self.app_state.console_open_progress += dt * 5.0;
      if self.app_state.console_open_progress > 1.0 {
        self.app_state.console_open_progress = 1.0;
      }
    } else {
      self.app_state.console_open_progress -= dt * 5.0;
      if self.app_state.console_open_progress < 0.0 {
        self.app_state.console_open_progress = 0.0;
      }
    }

    struct StepData {
      scene_id: u64,
      epoch: anise::time::Epoch,
      step_days: f64,
    }

    let step_data = Box::new(StepData {
      scene_id: self.app_state.scene_id,
      epoch: self.app_state.current_epoch,
      step_days: self.app_state.step_days,
    });

    let _ = self.app_state.ctx.dispatch_logic_command_custom(
      |ctx, user_data| {
        let step_data = unsafe { Box::from_raw(user_data as *mut StepData) };
      },
      Some(aethervk_core_rlib::simulation_api::structs::SendPtrMut(
        Box::into_raw(step_data) as *mut core::ffi::c_void,
      )),
    );

    self.app_state.current_epoch += anise::time::Duration::from_days(self.app_state.step_days);

    let ctx = &self.app_state.ctx;
    if let Ok(task) = ctx.simulation_tick(self.app_state.scene_id, delta_time) {
      self.last_sim_tick_task = Some(task);
    }

    if let Some(w) = self.app_state.window.as_ref() {
      w.request_redraw();
    }
  }
}

#[derive(Default)]
struct CustomRenderData {
  console_open_progress: f32,
  console_scroll_offset: usize,
  command_history: std::collections::VecDeque<String>,
  current_command: String,
  font_atlas: Option<FontAtlas>,
  font_id: (u64, u32),
  size: winit::dpi::PhysicalSize<u32>,
  rt_in_use: AtomicBool,
  rt_first_in_use: AtomicBool,
  mt_sent_not_received: AtomicBool,
}

impl CustomRenderData {
  fn make_callback(&mut self) -> CustomRenderCallback {
    CustomRenderCallback {
      after_render_frame_fn: ui_custom_render_fn,
      on_first_render_fn: first_render_update_atlas,
      user_data: SendPtrMut(self as *mut Self as *mut core::ffi::c_void),
    }
  }

  fn is_free_relaxed(&self) -> bool {
    !self.rt_in_use.load(std::sync::atomic::Ordering::Relaxed)
      && !self
        .rt_first_in_use
        .load(std::sync::atomic::Ordering::Relaxed)
      && !self
        .mt_sent_not_received
        .load(std::sync::atomic::Ordering::Relaxed)
  }

  fn is_free_acquire(&self) -> bool {
    !self.rt_in_use.load(std::sync::atomic::Ordering::Acquire)
      && !self
        .rt_first_in_use
        .load(std::sync::atomic::Ordering::Acquire)
      && !self
        .mt_sent_not_received
        .load(std::sync::atomic::Ordering::Acquire)
  }
}

struct AtomicCounterGuard<'a> {
  value: &'a AtomicBool,
}

impl<'a> AtomicCounterGuard<'a> {
  fn new(value: &'a AtomicBool) -> Self {
    value.store(true, std::sync::atomic::Ordering::Release);
    Self { value }
  }
}

impl<'a> Drop for AtomicCounterGuard<'a> {
  fn drop(&mut self) {
    self
      .value
      .store(false, std::sync::atomic::Ordering::Release);
  }
}

static FONT_HASH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FONT_INTERNAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn first_render_update_atlas(
  device: &dyn aethervk_core_rlib::gpu::RenderDevice,
  cmd_buffer: aethervk_core_rlib::gpu::CommandBufferHandle,
  _presentation_engine_handle: PresentationEngineHandle,
  _render_scene: &aethervk_core_rlib::gpu::RenderScene,
  user_data: *mut core::ffi::c_void,
) -> GpuResult<()> {
  let data = unsafe { &mut *(user_data as *mut CustomRenderData) };
  let _use_guard = AtomicCounterGuard::new(&data.rt_first_in_use);
  data
    .mt_sent_not_received
    .store(false, std::sync::atomic::Ordering::Relaxed);

  if let Some(atlas) = data.font_atlas.take() {
    let font_hash = atlas.hash_metadata();
    let font_internal = device.allocate_rasterized_font_atlas(cmd_buffer, font_hash, atlas)?;
    FONT_HASH.store(font_hash, std::sync::atomic::Ordering::Relaxed);
    FONT_INTERNAL.store(font_internal, std::sync::atomic::Ordering::Relaxed);
  }
  Ok(())
}

fn ui_custom_render_fn(
  device: &dyn aethervk_core_rlib::gpu::RenderDevice,
  cmd_buffer: aethervk_core_rlib::gpu::CommandBufferHandle,
  presentation_engine_handle: PresentationEngineHandle,
  render_scene: &aethervk_core_rlib::gpu::RenderScene,
  user_data: *mut core::ffi::c_void,
) -> GpuResult<()> {
  let data = unsafe { &mut *(user_data as *mut CustomRenderData) };
  let _use_guard = AtomicCounterGuard::new(&data.rt_in_use);
  data
    .mt_sent_not_received
    .store(false, std::sync::atomic::Ordering::Relaxed);
  let size = data.size;
  let screen_extent = [size.width as f32, size.height as f32];
  let font_id = (
    FONT_HASH.load(std::sync::atomic::Ordering::Relaxed),
    FONT_INTERNAL.load(std::sync::atomic::Ordering::Relaxed),
  );

  if !render_scene.measurement_calls.is_empty() {
    let _ = device
      .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);

    let view_proj = render_scene.camera_data.view_proj;
    for m in &render_scene.measurement_calls {
      let p1 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
        m.p1[0], m.p1[1], m.p1[2],
      );
      let p2 = Vec3f32::from_components(m.p2[0], m.p2[1], m.p2[2]);

      use aethervk_oshal_rlib::math::vector::Vector;
      let mid = p1 + (p2 - p1) * 0.5;

      let offset_mid = mid + Vec3f32::from_components(0.0, 0.0, 5.0);

      if let Some((screen_x, screen_y)) = aethervk_core_rlib::math::from_world_space_to_screen_space(
        offset_mid,
        view_proj,
        (screen_extent[0], screen_extent[1]),
      ) {
        let distance = (p2 - p1).length() as f64;
        let text = format!("{:.1}", distance);

        let ndc_x = (screen_x / screen_extent[0]) * 2.0 - 1.0;
        let ndc_y = (screen_y / screen_extent[1]) * 2.0 - 1.0;

        let _ = device.render_text(
          cmd_buffer,
          &text,
          [ndc_x, ndc_y],
          screen_extent,
          font_id,
          m.points,
          [1.0, 1.0, 1.0, 1.0],
        );
      }
    }
  }

  let console_open_progress = data.console_open_progress;
  let slide_y = -1.0 + (console_open_progress * 1.0);
  let base_y = 0.18 + slide_y;

  if console_open_progress > 0.0 {
    let width = 2.0;
    let height = 1.0;
    let box_y = -1.0 - height + (console_open_progress * height);

    let _ = device.render_ui_rect(
      cmd_buffer,
      presentation_engine_handle,
      [0.05, 0.1, 0.05, 0.7],
      [-1.0, box_y],
      [width, height],
    );

    let mut console_text = String::new();
    let max_lines = 12;
    let history_len = data.command_history.len();
    let scroll = data
      .console_scroll_offset
      .min(history_len.saturating_sub(max_lines));
    let start_idx = history_len.saturating_sub(max_lines + scroll);
    let end_idx = history_len.saturating_sub(scroll);

    for cmd in data
      .command_history
      .iter()
      .skip(start_idx)
      .take(end_idx - start_idx)
    {
      console_text.push_str(cmd);
      console_text.push('\n');
    }

    let prompt_y = box_y + height - 0.08;
    let text_start_y = box_y + 0.05;

    let _ = device
      .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);

    let _ = device.render_text(
      cmd_buffer,
      &console_text,
      [-0.98, text_start_y],
      screen_extent,
      font_id,
      14.0,
      [0.8, 0.8, 0.8, 1.0],
    );

    let mut prompt_text = String::from("> ");
    prompt_text.push_str(&data.current_command);
    prompt_text.push('_');

    let _ = device.render_text(
      cmd_buffer,
      &prompt_text,
      [-0.98, prompt_y],
      screen_extent,
      font_id,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
    );
  }

  Ok(())
}
