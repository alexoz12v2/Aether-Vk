use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::text::FontAtlas;
use aethervk_core_rlib::scene::{GridComponent, HiddenComponent};
use aethervk_core_rlib::simulation_api::structs::{CustomRenderCallback, SendPtrMut};
use aethervk_core_rlib::types::GpuResult;
use aethervk_core_rlib::{
  gpu::{self},
  scene::GizmoComponent,
  scene::{AlmanacPlanet, PhysicalMeshComponent, TransformComponent},
  simulation::constants,
  simulation_api::{SimulationContext, structs},
};
use aethervk_oshal_rlib::{
  math::vector::Vector,
  math::{
    quaternion::Quaternion,
    vector::{Vector3, vec3::Vec3f32, vec4::Quat},
  },
};
use std::sync::atomic::AtomicBool;
use std::{sync::Arc, time::Instant};
use test_utils::{
  create_winit_window_and_event_loop, cycle_get_asset_path_from_exe, get_handle_and_window_info,
};

struct AppState {
  ctx: Box<SimulationContext>,
  custom_data: Arc<std::sync::Mutex<CustomRenderData>>,
  scene_id: u64,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: u64,
  window: Option<winit::window::Window>,
  is_resizing: bool,
  is_exiting: bool,
  is_command_prompt_open: bool,
  // TODO reuse it as before in rendering function, which therefore should be customizable in simulation context
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

  let (native_handles, window_info) = {
    let render_frontend = simulation_context.render_frontend().unwrap();
    let render_device_handle = simulation_context.render_device_handle();
    get_handle_and_window_info(&render_frontend, render_device_handle, &window)
  };

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let scene_id = simulation_context.create_default_scene().unwrap();

  let presentation_engine = simulation_context
    .create_presentation_engine_windowed(scene_id, width, height, native_handles)
    .unwrap();

  // Populate scene with custom planets and comet
  {
    let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
    let mut active_scene = scene_ctx.write(); // TODO DEADLOCK with SimulationTick
    let root_entity = active_scene.root_entity;

    let model_path = assets_dir.join("Comet.glb");
    let comet = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      model_path.to_str().unwrap(),
      false,
    )
    .expect("Failed to load comet");

    let mesh_entity = active_scene.scene.spawn_entity("comet");
    // Scale comet so its size represents 1.7km (radius ~0.85km).
    // Using an artificially larger scale so it's visible.
    let comet_radius = 1.0;

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
    active_scene.scene.set_parent(mesh_entity, Some(root_entity));
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
        // TODO insert true mass of planet (hard code or from constants kernel)
        let mut sphere =
          aethervk_core_rlib::simulation::comet::generate_uv_sphere(planet_radius, 64, 64, 1.0);
        let tex = aethervk_core_rlib::simulation::comet::load_texture_from_file(
          assets_dir.join(tex_path).to_str().unwrap(),
        )
        .expect(&format!("Failed to load texture for {}", name));
        sphere.albedo_map = Some(tex);
        Arc::from(sphere)
      };

      let planet_entity = active_scene.scene.spawn_entity(*name);
      active_scene.scene.set_parent(planet_entity, Some(root_entity));
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
    read_ctx.entity_map.iter().find(|&(_, &v)| v == int_cam).map(|(&k, _)| k).unwrap()
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

  let _ = simulation_context.threads.logic_thread.tx().try_send(
    aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
      scene_id,
      scale: aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay,
    },
  );
  let _ = simulation_context
    .threads
    .logic_thread
    .tx()
    .try_send(aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id });

  let mut custom_render_data = CustomRenderData::default();
  custom_render_data.font_atlas = Some(atlas);
  let custom_data = Arc::new(std::sync::Mutex::new(custom_render_data));

  let arc_scenes = simulation_context.get_scene(scene_id).unwrap();
  let mut scene_write = arc_scenes.write();
  let cb = custom_data.lock().unwrap().make_callback(Arc::clone(&custom_data));
  scene_write.register_custom_render_callback(Some(cb));
  drop(scene_write);
  drop(arc_scenes);

  let app_state = AppState {
    ctx: simulation_context,
    custom_data,
    scene_id,
    presentation_engine,
    camera_entity: ext_camera_entity,
    window: Some(window),
    is_resizing: false,
    is_exiting: false,
    is_command_prompt_open: false,
    console_open_progress: 0.0,
    console_scroll_offset: 0,
    command_history: std::collections::VecDeque::with_capacity(1000),
    current_command: String::new(),
  };

  let sim_app = SimApp {
    app_state,
    right_mouse_button_down: false,
    middle_mouse_button_down: false,
    ctrl_down: false,
    mouse_x: 0.0,
    mouse_y: 0.0,
    last_sim_time: std::time::Instant::now(),
    window_info,
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
            let _ = self.app_state.ctx.raycast_ndc(self.app_state.scene_id, ndc_x, ndc_y);
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
                self.app_state.command_history.push_back(format!("> {}", cmd));

                let mut responses = Vec::new();
                let parts: Vec<&str> = cmd.split_whitespace().collect();
                if let Some(&command) = parts.first() {
                  match command {
                    "help" => {
                      responses.push("Commands:".to_string());
                      responses.push("  help               - Shows this help message".to_string());
                      responses
                        .push("  clear              - Clears the console output".to_string());
                      responses.push("  play               - Plays the simulation".to_string());
                      responses.push("  pause              - Pauses the simulation".to_string());
                      responses.push("  scale <0|1|2|3>    - Sets the time scale (0=Stopped, 1=Day, 2=Week, 3=Month)".to_string());
                      responses.push(
                        "  step <days>        - Steps the simulation by the given number of days"
                          .to_string(),
                      );
                    }
                    "clear" => {
                      self.app_state.command_history.clear();
                    }
                    "play" => {
                      let scene_id = self.app_state.scene_id;
                      let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene {
                          scene_id,
                        },
                      );
                      responses.push("Simulation playing.".to_string());
                    }
                    "pause" => {
                      let scene_id = self.app_state.scene_id;
                      let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                        aethervk_core_rlib::simulation_api::structs::LogicCommand::PauseScene {
                          scene_id,
                        },
                      );
                      responses.push("Simulation paused.".to_string());
                    }
                    "scale" => {
                      if parts.len() > 1 {
                        let scale_val = parts[1].parse::<u32>().unwrap_or(0);
                        let next_scale = match scale_val {
                          1 => aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay,
                          2 => aethervk_core_rlib::simulation_api::structs::TimeScale::OneWeek,
                          3 => aethervk_core_rlib::simulation_api::structs::TimeScale::OneMonth,
                          _ => aethervk_core_rlib::simulation_api::structs::TimeScale::Stopped,
                        };
                        let scene_id = self.app_state.scene_id;
                        let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                          aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
                            scene_id,
                            scale: next_scale,
                          }
                        );
                        responses.push(format!("Time scale set to {}.", scale_val));
                      } else {
                        responses.push("Usage: scale <0|1|2|3>".to_string());
                      }
                    }
                    "step" => {
                      if parts.len() > 1 {
                        if let Ok(days) = parts[1].parse::<f64>() {
                          let scene_id = self.app_state.scene_id;
                          let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                            aethervk_core_rlib::simulation_api::structs::LogicCommand::StepScene {
                              scene_id,
                              step_days: days,
                            },
                          );
                          responses.push(format!("Stepped simulation by {} days.", days));
                        } else {
                          responses.push("Invalid number of days.".to_string());
                        }
                      } else {
                        responses.push("Usage: step <days>".to_string());
                      }
                    }

                    "scene" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        responses.push(format!(
                          "Scene has {} entities",
                          read_ctx.scene.entity_count()
                        ));
                      }
                    }
                    "select" => {
                      if parts.len() > 1 {
                        let name = parts[1..].join(" ");
                        let scene_id = self.app_state.scene_id;
                        if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                          let read_ctx = scene_ctx.read();
                          if let Some(id) = read_ctx.scene.get_entity_by_name(&name) {
                            use aethervk_core_rlib::scene::interaction::SceneInteractionExt;
                            let _ = read_ctx.scene.select_entity(id, None);
                            responses.push(format!("Selected entity '{}'.", name));
                          } else {
                            responses.push(format!("Entity '{}' not found.", name));
                          }
                        }
                      } else {
                        responses.push("Usage: select <entity>".to_string());
                      }
                    }
                    "printsel" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          if let Some(name) = read_ctx.scene.get_name(id) {
                            responses.push(format!("Currently selected: {}", name));
                          }
                        } else {
                          responses.push("No entity selected.".to_string());
                        }
                      }
                    }
                    "deselect" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        use aethervk_core_rlib::scene::interaction::SceneInteractionExt;
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          let _ = read_ctx.scene.unselect_entity(id);
                          responses.push("Deselected entity.".to_string());
                        }
                      }
                    }
                    "goto" => {
                      if parts.len() > 1 {
                        let name = parts[1..].join(" ");
                        let scene_id = self.app_state.scene_id;
                        if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                          let read_ctx = scene_ctx.read();
                          if let Some(id) = read_ctx.scene.get_entity_by_name(&name) {
                            use aethervk_core_rlib::scene::interaction::SceneInteractionExt;
                            let _ = read_ctx.scene.select_entity(id, None);
                            let _ = read_ctx.scene.follow_entity(id, None);
                            responses.push(format!("Selected and following '{}'.", name));
                          } else {
                            responses.push(format!("Entity '{}' not found.", name));
                          }
                        }
                      } else {
                        responses.push("Usage: goto <entity>".to_string());
                      }
                    }
                    "unfollow" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        use aethervk_core_rlib::scene::interaction::SceneInteractionExt;
                        let f = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::FollowingComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = f {
                          let _ = read_ctx.scene.unfollow_entity(id);
                          responses.push("Unfollowed entity.".to_string());
                        }
                      }
                    }
                    "following" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let f = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::FollowingComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = f {
                          if let Some(name) = read_ctx.scene.get_name(id) {
                            responses.push(format!("Currently following: {}", name));
                          }
                        } else {
                          responses.push("Not following any entity.".to_string());
                        }
                      }
                    }
                    "showgizmo" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          let mut is_visible = false;
                          let _ = read_ctx.scene.with_component_mut(
                            id,
                            |g: &mut aethervk_core_rlib::scene::GizmoComponent| {
                              g.gizmo_visible = !g.gizmo_visible;
                              is_visible = g.gizmo_visible;
                            },
                          );
                          responses.push(format!("Gizmo visibility set to {}.", is_visible));
                        } else {
                          responses.push("No entity selected.".to_string());
                        }
                      }
                    }
                    "printbvh" => {
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          let min_depth: i32 = parts.get(1).unwrap_or(&"-1").parse().unwrap_or(-1);
                          let max_depth: i32 = parts.get(2).unwrap_or(&"-1").parse().unwrap_or(-1);
                          if min_depth != -1 && max_depth != -1 && min_depth > max_depth {
                            responses.push("illegal arguments: min_depth > max_depth".to_string());
                          } else {
                            read_ctx.scene.with_component(
                              id,
                              |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                                if let Some(bvh) = &mesh.mesh.bvh {
                                  responses.push("BVH Nodes:".to_string());
                                  let mut node_stack = vec![(0, 0)]; // (node_idx, depth)
                                  while let Some((idx, depth)) = node_stack.pop() {
                                    let node = &bvh.nodes[idx];
                                    if (min_depth == -1 || depth as i32 >= min_depth)
                                      && (max_depth == -1 || depth as i32 <= max_depth)
                                    {
                                      responses.push(format!(
                                        "{}Node {} (Depth: {}) - Bound: {:?}",
                                        "  ".repeat(depth),
                                        idx,
                                        depth,
                                        node.bound
                                      ));
                                    }
                                    if node.primitive_count == 0 {
                                      node_stack
                                        .push((node.right_child_offset as usize, depth + 1));
                                      node_stack.push((
                                        node.left_child_or_primitive_offset as usize,
                                        depth + 1,
                                      ));
                                    }
                                  }
                                } else {
                                  responses.push("Entity has no BVH.".to_string());
                                }
                              },
                            );
                          }
                        } else {
                          responses.push("No entity selected.".to_string());
                        }
                      }
                    }
                    "bvh-show"
                    | "show-bvh"
                    | "bvh-hide"
                    | "hide-bvh"
                    | "bvh-node-dbgrender-set"
                    | "set-bvh-dbgrender" => {
                      let is_show = command.contains("show") || command.contains("set");
                      let scene_id = self.app_state.scene_id;
                      if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          let mut parts_iter = parts.iter().skip(1);
                          let depth_str = parts_iter.next().unwrap_or(&"all");
                          let idx_str = parts_iter.next();
                          let target_idx: Option<u32> =
                            idx_str.and_then(|s| if *s == "all" { None } else { s.parse().ok() });

                          let mut min_d = 0;
                          let mut max_d = u32::MAX;

                          if *depth_str != "all" {
                            if let Some(dash_pos) = depth_str.find('-') {
                              let (start, end) = depth_str.split_at(dash_pos);
                              let end = &end[1..];
                              if !start.is_empty() {
                                min_d = start.parse().unwrap_or(0);
                              }
                              if !end.is_empty() {
                                max_d = end.parse().unwrap_or(u32::MAX);
                              }
                            } else {
                              let d: u32 = depth_str.parse().unwrap_or(0);
                              min_d = d;
                              max_d = d;
                            }
                          }

                          let mut flat_indices = Vec::new();
                          let mut max_depth_found = 0;
                          read_ctx.scene.with_component(
                            id,
                            |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                              if let Some(bvh) = &mesh.mesh.bvh {
                                let mut node_stack = vec![(0, 0)];
                                let mut current_child_index_at_depth =
                                  std::collections::HashMap::new();

                                while let Some((idx, d)) = node_stack.pop() {
                                  if d > max_depth_found {
                                    max_depth_found = d;
                                  }
                                  if d >= min_d && d <= max_d {
                                    let child_index =
                                      current_child_index_at_depth.entry(d).or_insert(0);
                                    if target_idx.is_none() || target_idx == Some(*child_index) {
                                      flat_indices.push((d, idx));
                                    }
                                    *child_index += 1;
                                  }
                                  let node = &bvh.nodes[idx];
                                  if node.primitive_count == 0 {
                                    node_stack.push((node.right_child_offset as usize, d + 1));
                                    node_stack
                                      .push((node.left_child_or_primitive_offset as usize, d + 1));
                                  }
                                }
                              }
                            },
                          );

                          if flat_indices.is_empty() {
                            if min_d > max_depth_found && min_d != u32::MAX {
                              responses.push(format!(
                                "Error: Max depth is {}, requested {}.",
                                max_depth_found, min_d
                              ));
                            } else {
                              responses.push("No nodes found for the given criteria.".to_string());
                            }
                          } else {
                            let mut bvh_len = 0;
                            read_ctx.scene.with_component(
                              id,
                              |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
                                if let Some(bvh) = &mesh.mesh.bvh {
                                  bvh_len = bvh.nodes.len();
                                }
                              },
                            );

                            if bvh_len > 0 {
                              let mut added = false;
                              let _ = read_ctx.scene.with_component_mut(
                                id,
                                |dbg: &mut aethervk_core_rlib::scene::BvhDebugComponent| {
                                  for &(_, idx) in &flat_indices {
                                    if idx < dbg.node_render_states.len() {
                                      dbg.node_render_states[idx] = is_show;
                                    }
                                  }
                                  added = true;
                                },
                              );
                              if !added {
                                let mut states = vec![false; bvh_len];
                                for &(_, idx) in &flat_indices {
                                  if idx < states.len() {
                                    states[idx] = is_show;
                                  }
                                }
                                let _ = read_ctx.scene.add_component(
                                  id,
                                  aethervk_core_rlib::scene::BvhDebugComponent {
                                    node_render_states: states,
                                  },
                                );
                              }
                              responses.push(format!(
                                "Updated visibility for {} nodes.",
                                flat_indices.len()
                              ));
                            }
                          }
                        } else {
                          responses.push("No entity selected.".to_string());
                        }
                      }
                    }
                    "measure" => {
                      if parts.len() != 3 {
                        responses.push("Usage: measure <entity1> <entity2>".to_string());
                      } else {
                        let name1 = parts[1];
                        let name2 = parts[2];
                        let scene_id = self.app_state.scene_id;
                        if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                          let read_ctx = scene_ctx.read();
                          let id1 = read_ctx.scene.get_entity_by_name(name1);
                          let id2 = read_ctx.scene.get_entity_by_name(name2);

                          if id1.is_none() || id2.is_none() {
                            responses.push("One or both entities not found.".to_string());
                          } else {
                            let id1 = id1.unwrap();
                            let id2 = id2.unwrap();
                            let mut pos1 = None;
                            let mut pos2 = None;
                            read_ctx.scene.with_component(
                              id1,
                              |t: &aethervk_core_rlib::scene::TransformComponent| {
                                pos1 = Some(t.position)
                              },
                            );
                            read_ctx.scene.with_component(
                              id2,
                              |t: &aethervk_core_rlib::scene::TransformComponent| {
                                pos2 = Some(t.position)
                              },
                            );

                            if let (Some(p1), Some(p2)) = (pos1, pos2) {
                              let measure_name = format!("measure_{}_{}", name1, name2);
                              let measure_id = read_ctx.scene.spawn_entity(&measure_name);
                              let _ = read_ctx.scene.add_component(
                                measure_id,
                                aethervk_core_rlib::scene::MeasurementComponent {
                                  pos1: p1,
                                  pos2: p2,
                                  points: 12.0,
                                  significant_digits: 4,
                                },
                              );
                              let _ = read_ctx.scene.add_component(
                                            measure_id,
                                            aethervk_core_rlib::scene::TransformComponent {
                                                position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(0.0, 0.0, 0.0),
                                                rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
                                                scale: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(1.0, 1.0, 1.0),
                                            },
                                        );
                              let root_entity = read_ctx.root_entity;
                              read_ctx.scene.set_parent(measure_id, Some(root_entity));

                              use aethervk_oshal_rlib::math::vector::Vector;
                              let distance = (p1 - p2).length();
                              responses.push(format!(
                                "Created measurement {} between {} and {}: {:.4}",
                                measure_name, name1, name2, distance
                              ));
                            } else {
                              responses
                                .push("Both entities must have a TransformComponent.".to_string());
                            }
                          }
                        }
                      }
                    }
                    _ => {
                      responses.push(format!("Unrecognized command: {}", command));
                    }
                  }
                }

                for resp in responses {
                  self.app_state.command_history.push_back(resp);
                }

                if self.app_state.command_history.len() > 1000 {
                  // Keep the last 1000 elements
                  while self.app_state.command_history.len() > 1000 {
                    self.app_state.command_history.pop_front();
                  }
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
            let scene = self.app_state.ctx.get_scene(self.app_state.scene_id).unwrap();
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
              winit::keyboard::KeyCode::Digit0 => {
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let scene = scene_ctx.clone();
                  let camera_entity = scene_ctx.read().active_camera_entity.unwrap();
                  let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                    aethervk_core_rlib::simulation_api::structs::LogicCommand::ResetCamera(
                      aethervk_core_rlib::simulation_api::structs::ResetCamera {
                        camera_entity,
                        scene,
                      },
                    ),
                  );
                }
              }
              winit::keyboard::KeyCode::KeyM => {
                self.app_state.is_command_prompt_open = true;
              }
              winit::keyboard::KeyCode::KeyP => {
                let scene_id = self.app_state.scene_id;
                let is_playing = {
                  let scene = self.app_state.ctx.get_scene(scene_id).unwrap();
                  scene.read().time_state.read().is_playing
                };
                let cmd = if is_playing {
                  aethervk_core_rlib::simulation_api::structs::LogicCommand::PauseScene { scene_id }
                } else {
                  aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id }
                };
                let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(cmd);
              }
              winit::keyboard::KeyCode::KeyX => {
                let scene_id = self.app_state.scene_id;
                let current_scale = {
                  let scene = self.app_state.ctx.get_scene(scene_id).unwrap();
                  scene.read().time_state.read().current_scale
                };
                let next_scale = match current_scale {
                  aethervk_core_rlib::simulation_api::structs::TimeScale::Stopped => {
                    aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay
                  }
                  aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay => {
                    aethervk_core_rlib::simulation_api::structs::TimeScale::OneWeek
                  }
                  aethervk_core_rlib::simulation_api::structs::TimeScale::OneWeek => {
                    aethervk_core_rlib::simulation_api::structs::TimeScale::OneMonth
                  }
                  aethervk_core_rlib::simulation_api::structs::TimeScale::OneMonth => {
                    aethervk_core_rlib::simulation_api::structs::TimeScale::Stopped
                  }
                };
                let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
                  aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
                    scene_id,
                    scale: next_scale,
                  },
                );
              }
              winit::keyboard::KeyCode::KeyG => {
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  let grid_entity = read_ctx
                    .scene
                    .query1_first_res(|_, _g: &GridComponent| Some(()))
                    .map(|(_, e)| e)
                    .unwrap();
                  if read_ctx.scene.has_component::<HiddenComponent>(grid_entity).into() {
                    let _ = read_ctx.scene.remove_component::<HiddenComponent>(grid_entity);
                  } else {
                    let _ = read_ctx
                      .scene
                      .add_component::<HiddenComponent>(grid_entity, HiddenComponent {});
                  }
                }
              }
              winit::keyboard::KeyCode::KeyH => {
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  let current =
                    read_ctx.outlines_enabled.load(std::sync::atomic::Ordering::Relaxed);
                  read_ctx.outlines_enabled.store(!current, std::sync::atomic::Ordering::Relaxed);
                }
              }
              winit::keyboard::KeyCode::KeyA | winit::keyboard::KeyCode::KeyD => {
                let forward = keycode == winit::keyboard::KeyCode::KeyD;
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  let mut entities: Vec<aethervk_core_rlib::scene::EntityId> = Vec::new();

                  let res = read_ctx
                    .scene
                    .query1_res::<aethervk_core_rlib::scene::PhysicalMeshComponent, _, ()>(
                      |_id, _| Some(()),
                    );
                  for (_, id) in res {
                    entities.push(id);
                  }

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
    let camera_entity = scene.read().get_entity(self.app_state.camera_entity).unwrap();

    let logic_command = if self.middle_mouse_button_down {
      if self.ctrl_down {
        Some(
          aethervk_core_rlib::simulation_api::structs::LogicCommand::ZoomCamera(
            aethervk_core_rlib::simulation_api::structs::ZoomCamera {
              camera_entity,
              scene: scene.clone(),
              amount: (delta.1 * 0.1) as f32,
            },
          ),
        )
      } else {
        Some(
          aethervk_core_rlib::simulation_api::structs::LogicCommand::RotateCamera(
            aethervk_core_rlib::simulation_api::structs::RotateCamera {
              camera_entity,
              scene: scene.clone(),
              delta_x: delta.0 as f32,
              delta_y: delta.1 as f32,
            },
          ),
        )
      }
    } else if self.right_mouse_button_down {
      Some(structs::LogicCommand::PanCursor(structs::PanCursor {
        scene: scene.clone(),
        delta_x: delta.0 as f32,
        delta_y: delta.1 as f32,
      }))
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
    // Rendering is now handled in on_about_to_wait via play_control
  }

  fn on_about_to_wait(&mut self) {
    let current_time = std::time::Instant::now();
    let delta_time = current_time.duration_since(self.last_sim_time).as_secs_f64();
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

    if self.app_state.window.is_none() {
      return;
    }

    let size = self.app_state.window.as_ref().unwrap().inner_size();
    let console_open_progress = self.app_state.console_open_progress;
    let console_scroll_offset = self.app_state.console_scroll_offset;
    let command_history = self.app_state.command_history.clone();
    let current_command = self.app_state.current_command.clone();

    let should_render = size.width > 0 && size.height > 0;

    if should_render {
      let mut lock = self.app_state.custom_data.lock().unwrap();
      lock.console_open_progress = console_open_progress;
      lock.console_scroll_offset = console_scroll_offset;
      lock.command_history = command_history;
      lock.current_command = current_command;
      lock.size = size;
      // We don't tick here, logic thread ticks autonomously
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
  size: winit::dpi::PhysicalSize<u32>,
}

impl CustomRenderData {
  fn make_callback(&mut self, arc_self: Arc<std::sync::Mutex<Self>>) -> CustomRenderCallback {
    CustomRenderCallback {
      after_render_frame_fn: ui_custom_render_fn,
      on_first_render_fn: first_render_update_atlas,
      user_data: SendPtrMut(Arc::into_raw(arc_self) as *mut core::ffi::c_void),
    }
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
  let data_mutex = unsafe { &*(user_data as *const std::sync::Mutex<CustomRenderData>) };
  let mut data = data_mutex.lock().unwrap();

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
  let data_mutex = unsafe { &*(user_data as *const std::sync::Mutex<CustomRenderData>) };
  let data = data_mutex.lock().unwrap();

  let size = data.size;
  let screen_extent = [size.width as f32, size.height as f32];
  let font_id = (
    FONT_HASH.load(std::sync::atomic::Ordering::Relaxed),
    FONT_INTERNAL.load(std::sync::atomic::Ordering::Relaxed),
  );
  if font_id.0 == 0 {
    panic!("font_id is 0! first_render_update_atlas wasn't called?");
  }

  let mut mem_text = String::new();
  if let Some(vk_dev) =
    device.as_any().downcast_ref::<aethervk_core_rlib::gpu_backends::vulkan::device::Device>()
  {
    let (budget, usage) = vk_dev.get_vma_budget_usage();
    mem_text.push_str(&format!(
      "VMA Usage: {:.2} MB / {:.2} MB\n",
      usage as f32 / 1048576.0,
      budget as f32 / 1048576.0
    ));
  }
  let os_mem = aethervk_oshal_rlib::os::memory::query_process_memory();
  mem_text.push_str(&format!(
    "OS Virt: {:.2} MB / Phys: {:.2} MB\nFile-backed: {:.2} MB",
    os_mem.virtual_bytes as f32 / 1048576.0,
    os_mem.physical_bytes as f32 / 1048576.0,
    os_mem.file_backed_bytes as f32 / 1048576.0
  ));

  let _ = device
    .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);
  if let Err(e) = device.render_text(
    cmd_buffer,
    &mem_text,
    [-0.98, -0.95],
    screen_extent,
    font_id,
    16.0,
    [0.0, 1.0, 0.0, 1.0],
  ) {
    println!("Render text error mem: {:?}", e);
  }

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

        if let Err(e) = device.render_text(
          cmd_buffer,
          &text,
          [ndc_x, ndc_y],
          screen_extent,
          font_id,
          m.points,
          [1.0, 1.0, 1.0, 1.0],
        ) {
          println!("Render text error: {:?}", e);
        }
      }
    }
  }

  let console_open_progress = data.console_open_progress;

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
    let scroll = data.console_scroll_offset.min(history_len.saturating_sub(max_lines));
    let start_idx = history_len.saturating_sub(max_lines + scroll);
    let end_idx = history_len.saturating_sub(scroll);

    for cmd in data.command_history.iter().skip(start_idx).take(end_idx - start_idx) {
      console_text.push_str(cmd);
      console_text.push('\n');
    }

    let prompt_y = box_y + height - 0.08;
    let text_start_y = box_y + 0.05;

    let _ = device
      .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);

    if let Err(e) = device.render_text(
      cmd_buffer,
      &console_text,
      [-0.98, text_start_y],
      screen_extent,
      font_id,
      14.0,
      [0.8, 0.8, 0.8, 1.0],
    ) {
      println!("Render text error console: {:?}", e);
    }

    let mut prompt_text = String::from("> ");
    prompt_text.push_str(&data.current_command);
    prompt_text.push('_');

    if let Err(e) = device.render_text(
      cmd_buffer,
      &prompt_text,
      [-0.98, prompt_y],
      screen_extent,
      font_id,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
    ) {
      println!("Render text error prompt: {:?}", e);
    }
  }

  Ok(())
}

#[cfg(test)]
mod depth_tests {
  use super::*;
  use aethervk_core_rlib::simulation::comet::{TexelFormat, Texture};
  use aethervk_core_rlib::simulation_api::components_api::CameraParams;
  use std::sync::atomic::{AtomicU64, Ordering};

  static LAST_RENDER_TASK_ID: AtomicU64 = AtomicU64::new(0);

  extern "C" fn render_callback_impl(_scene_id: u64, _pe_id: u64, render_generation: u64) {
    LAST_RENDER_TASK_ID.store(render_generation, Ordering::Release);
  }

  fn debug_color(color: [f32; 3]) -> Option<Texture> {
    let u8_color = vec![
      (color[0] * 255.0).clamp(0.0, 255.0) as u8,
      (color[1] * 255.0).clamp(0.0, 255.0) as u8,
      (color[2] * 255.0).clamp(0.0, 255.0) as u8,
      255,
    ];
    Some(Texture {
      data: u8_color,
      format: TexelFormat::R8G8B8A8_UNORM,
      width: 1,
      height: 1,
      has_mipmaps: false,
    })
  }

  fn setup_test_scene(name: &str, front_emissive: bool, back_emissive: bool, test_outlines: bool) {
    LAST_RENDER_TASK_ID.store(0, Ordering::Release);
    let mut home_dir = std::env::current_exe().unwrap();
    let mut iter = 0;
    while !home_dir.join("assets").is_dir() && iter < 32 {
      home_dir.pop();
      iter += 1;
    }
    let assets_dir = home_dir.join("assets");
    aethervk_core_rlib::simulation_api::SimulationContext::set_asset_path(
      assets_dir.to_str().unwrap(),
    );

    fn panic_cb(msg: &str) {
      panic!("{}", msg);
    }
    let mut ctx_box =
      SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_cb)).unwrap();
    let ctx = ctx_box.as_mut();

    let scene_id = ctx.create_empty_scene().unwrap();
    let width = 64;
    let height = 64;
    let _pe = ctx.create_presentation_engine(scene_id, width, height).unwrap();

    // Add a sun so non-emissive meshes are visible
    let sun_entity = ctx.spawn_entity(scene_id, "sun").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        sun_entity,
        Vec3f32::from_components(0.0, 0.0, 100.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx.add_sun_component(scene_id, sun_entity, (128, 128, 128), 1.2).unwrap();
    {
      let scene_data_opt = ctx.get_scene(scene_id).unwrap();
      let mut write_scene_context = scene_data_opt.write();
      let sun_entity = write_scene_context.get_entity(sun_entity).unwrap();
      write_scene_context.sun_entity = Some(sun_entity);
    }

    let ext_red = ctx.spawn_procedural_sphere(scene_id, std::ptr::null(), 5.0, 1.0).unwrap();
    let int_red = ctx.get_scene(scene_id).unwrap().read().get_entity(ext_red).unwrap();
    let red_pos = Vec3f32::from_components(-2.0, -10.0, -2.0);
    ctx
      .set_transform_component(
        scene_id,
        ext_red,
        red_pos,
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx.get_scene(scene_id).unwrap().write().scene.with_component_mut(
      int_red,
      |c: &mut aethervk_core_rlib::scene::PhysicalMeshComponent| {
        c.emissive_color = [1.0, 0.0, 0.0];
        c.emissive_intensity = if front_emissive { 1.0 } else { 0.0 };
        // set albedo so sun light works
        if let Some(mesh) = Arc::get_mut(&mut c.mesh) {
          mesh.albedo_map = debug_color([1.0, 0.0, 0.0]);
        }
      },
    );

    if test_outlines {
      ctx.set_entity_selected(scene_id, ext_red, true).unwrap();
      ctx.get_scene(scene_id).unwrap().read().outlines_enabled.store(true, Ordering::Relaxed);
    } else {
      ctx.get_scene(scene_id).unwrap().read().outlines_enabled.store(false, Ordering::Relaxed);
    }

    let ext_blue = ctx.spawn_procedural_sphere(scene_id, std::ptr::null(), 5.0, 1.0).unwrap();
    let int_blue = ctx.get_scene(scene_id).unwrap().read().get_entity(ext_blue).unwrap();
    let blue_pos = Vec3f32::from_components(2.0, -20.0, 2.0);
    ctx
      .set_transform_component(
        scene_id,
        ext_blue,
        blue_pos,
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx.get_scene(scene_id).unwrap().write().scene.with_component_mut(
      int_blue,
      |c: &mut aethervk_core_rlib::scene::PhysicalMeshComponent| {
        c.emissive_color = [0.0, 0.0, 1.0];
        c.emissive_intensity = if back_emissive { 1.0 } else { 0.0 };
        if let Some(mesh) = Arc::get_mut(&mut c.mesh) {
          mesh.albedo_map = debug_color([0.0, 0.0, 1.0]);
        }
      },
    );

    let cam_entity = ctx.get_scene(scene_id).unwrap().read().active_camera_entity.unwrap();
    let ext_cam = ctx
      .get_scene(scene_id)
      .unwrap()
      .read()
      .entity_map
      .iter()
      .find(|(_, v)| **v == cam_entity)
      .unwrap()
      .0
      .clone();
    ctx
      .set_transform_component(
        scene_id,
        ext_cam,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx
      .set_camera_component(
        scene_id,
        ext_cam,
        CameraParams::new_orthographic(-10.0, 10.0, -10.0, 10.0, 0.1, 100.0),
      )
      .unwrap();

    SimulationContext::set_render_callback(Some(render_callback_impl));

    let _ =
      ctx.threads.logic_thread.tx().try_send(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
      );

    let mut attempts = 0;
    let mut task_id = 0;
    while attempts < 100 {
      task_id = LAST_RENDER_TASK_ID.load(Ordering::Acquire);
      if task_id != 0 {
        break;
      }
      aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
        10,
      ));
      attempts += 1;
    }

    assert!(task_id > 0, "Render task was never completed");

    let mut status = ctx.get_task_status(task_id);
    attempts = 0;
    while matches!(status, structs::TaskStatusCode::Pending) && attempts < 50 {
      aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
        10,
      ));
      status = ctx.get_task_status(task_id);
      attempts += 1;
    }

    assert!(
      matches!(status, structs::TaskStatusCode::Completed),
      "Render task did not complete successfully"
    );

    let mut buffer = vec![0u8; (width * height * 4) as usize];
    let success = ctx.download_image(task_id, buffer.as_mut_ptr(), buffer.len());
    assert!(success, "Download image failed");

    let mut img = image::ImageBuffer::new(width, height);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
      let idx = ((y * width + x) * 4) as usize;
      let b = buffer[idx];
      let g = buffer[idx + 1];
      let r = buffer[idx + 2];
      let a = buffer[idx + 3];
      *pixel = image::Rgba([r, g, b, a]);
    }
    let out_path = assets_dir.join(format!("../test_output_{}.png", name));
    img.save(&out_path).expect("Failed to save debug image");

    // X=-2, Y=-2 -> NDC ~ (-0.2, -0.2) -> pixel ~ (25, 25)
    // Red sphere overlaps here natively
    let r_idx = ((25 * width + 25) * 4) as usize;
    let b_r = buffer[r_idx];
    let g_r = buffer[r_idx + 1];
    let r_r = buffer[r_idx + 2];
    assert!(
      r_r > 50 && b_r < 50 && g_r < 50,
      "({name}) Red sphere area should be red, found r:{}, g:{}, b:{}",
      r_r,
      g_r,
      b_r
    );

    // X=2, Y=2 -> NDC ~ (0.2, 0.2) -> pixel ~ (38, 38)
    // Blue sphere overlaps here natively (unoccluded)
    let b_idx = ((38 * width + 38) * 4) as usize;
    let b_b = buffer[b_idx];
    let g_b = buffer[b_idx + 1];
    let r_b = buffer[b_idx + 2];
    assert!(
      b_b > 50 && r_b < 50 && g_b < 50,
      "({name}) Blue sphere area should be blue, found r:{}, g:{}, b:{}",
      r_b,
      g_b,
      b_b
    );

    // Center (0,0) -> NDC (0,0) -> pixel (32, 32)
    // Both overlap, red is in front. So it should be red.
    let center_idx = ((32 * width + 32) * 4) as usize;
    let b_c = buffer[center_idx];
    let g_c = buffer[center_idx + 1];
    let r_c = buffer[center_idx + 2];
    assert!(
      r_c > 50 && b_c < 50 && g_c < 50,
      "({name}) Center pixel (overlap) should be red, found r:{}, g:{}, b:{}",
      r_c,
      g_c,
      b_c
    );

    if test_outlines {
      // Outline is drawn around the red sphere.
      // Sphere at (-2, -2). Radius 5. Edge around X = -7. -> NDC X = -0.7. pixel X ~ 9.
      let left_edge_idx = ((25 * width + 8) * 4) as usize;
      let b_ol = buffer[left_edge_idx];
      let g_ol = buffer[left_edge_idx + 1];
      let r_ol = buffer[left_edge_idx + 2];

      // Outline default is Yellow (r > 0, g > 0, b < 50)
      assert!(
        r_ol > 100 && g_ol > 100 && b_ol < 50,
        "({name}) Outline pixel should be yellow, found r:{}, g:{}, b:{}",
        r_ol,
        g_ol,
        b_ol
      );

      // Verify outline doesn't render ON the red sphere (e.g. at its center X=25, Y=25)
      assert!(
        g_r < 50,
        "({name}) Outline should not render inside the red sphere, but found G={}",
        g_r
      );
    }

    ctx.destroy_presentation_engine(scene_id, _pe).unwrap();
  }

  #[test]
  fn test_depth_emissive_emissive() {
    setup_test_scene("depth_ee", true, true, false);
  }

  #[test]
  fn test_depth_emissive_nonemissive() {
    setup_test_scene("depth_en", true, false, false);
  }

  #[test]
  fn test_depth_nonemissive_emissive() {
    setup_test_scene("depth_ne", false, true, false);
  }

  #[test]
  fn test_depth_nonemissive_nonemissive() {
    setup_test_scene("depth_nn", false, false, false);
  }

  #[test]
  fn test_outline_emissive() {
    setup_test_scene("outline_e", true, false, true);
  }

  #[test]
  fn test_outline_nonemissive() {
    setup_test_scene("outline_n", false, false, true);
  }

  #[test]
  fn test_comet_rendering() {
    LAST_RENDER_TASK_ID.store(0, Ordering::Release);
    let mut home_dir = std::env::current_exe().unwrap();
    let mut iter = 0;
    while !home_dir.join("assets").is_dir() && iter < 32 {
      home_dir.pop();
      iter += 1;
    }
    let assets_dir = home_dir.join("assets");
    aethervk_core_rlib::simulation_api::SimulationContext::set_asset_path(
      assets_dir.to_str().unwrap(),
    );

    fn panic_cb(msg: &str) {
      panic!("{}", msg);
    }
    let mut ctx_box =
      SimulationContext::startup(gpu::VULKAN_RENDER_BACKEND, Some(panic_cb)).unwrap();
    let ctx = ctx_box.as_mut();

    let scene_id = ctx.create_empty_scene().unwrap();
    let width = 256;
    let height = 256;
    let _pe = ctx.create_presentation_engine(scene_id, width, height).unwrap();

    // Add a sun
    let sun_entity = ctx.spawn_entity(scene_id, "sun").unwrap();
    ctx
      .add_transform_component(
        scene_id,
        sun_entity,
        Vec3f32::from_components(100.0, 100.0, 100.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx.add_sun_component(scene_id, sun_entity, (128, 128, 128), 1.2).unwrap();
    {
      let scene_data_opt = ctx.get_scene(scene_id).unwrap();
      let mut write_scene_context = scene_data_opt.write();
      let sun_entity = write_scene_context.get_entity(sun_entity).unwrap();
      write_scene_context.sun_entity = Some(sun_entity);
    }

    // Load comet
    let model_path = assets_dir.join("Comet.glb");
    let comet = aethervk_core_rlib::simulation::comet::load_comet_from_gltf(
      model_path.to_str().unwrap(),
      false,
    )
    .expect("Failed to load comet");

    let mesh_entity = ctx.spawn_entity(scene_id, "comet").unwrap();
    ctx
      .set_transform_component(
        scene_id,
        mesh_entity,
        Vec3f32::zero(),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();

    let scene_ctx = ctx.get_scene(scene_id).unwrap();
    {
      let mut write_ctx = scene_ctx.write();
      let internal_id = write_ctx.get_entity(mesh_entity).unwrap();
      write_ctx
        .scene
        .add_component(
          internal_id,
          PhysicalMeshComponent {
            asset_path: "".to_string(),
            mesh: Arc::from(comet),
            emissive_intensity: 0.0,
            emissive_color: [0.0, 0.0, 0.0],
          },
        )
        .unwrap();
    }

    // Position camera
    let cam_entity = scene_ctx.read().active_camera_entity.unwrap();
    let ext_cam =
      scene_ctx.read().entity_map.iter().find(|&(_, &v)| v == cam_entity).map(|(&k, _)| k).unwrap();
    ctx
      .set_transform_component(
        scene_id,
        ext_cam,
        Vec3f32::from_components(0.0, -3.0, 0.0),
        Quat::identity(), // Looks in -Y natively? Let's use look_at if possible, or just ortho camera.
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();
    ctx
      .set_camera_component(
        scene_id,
        ext_cam,
        CameraParams::new_orthographic(-2.0, 2.0, -2.0, 2.0, 0.1, 100.0),
      )
      .unwrap();

    SimulationContext::set_render_callback(Some(render_callback_impl));

    let _ =
      ctx.threads.logic_thread.tx().try_send(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
      );

    let mut attempts = 0;
    let mut task_id = 0;
    while attempts < 100 {
      task_id = LAST_RENDER_TASK_ID.load(Ordering::Acquire);
      if task_id != 0 {
        break;
      }
      aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
        10,
      ));
      attempts += 1;
    }

    assert!(task_id > 0, "Render task was never completed");

    let mut status = ctx.get_task_status(task_id);
    attempts = 0;
    while matches!(status, structs::TaskStatusCode::Pending) && attempts < 50 {
      aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
        10,
      ));
      status = ctx.get_task_status(task_id);
      attempts += 1;
    }

    let mut buffer = vec![0u8; (width * height * 4) as usize];
    let success = ctx.download_image(task_id, buffer.as_mut_ptr(), buffer.len());
    assert!(success, "Download image failed");

    let mut img = image::ImageBuffer::new(width, height);
    let mut non_empty_pixels = 0;
    for (x, y, pixel) in img.enumerate_pixels_mut() {
      let idx = ((y * width + x) * 4) as usize;
      let b = buffer[idx];
      let g = buffer[idx + 1];
      let r = buffer[idx + 2];
      let a = buffer[idx + 3];
      *pixel = image::Rgba([r, g, b, a]);
      if r > 10 || g > 10 || b > 10 {
        non_empty_pixels += 1;
      }
    }
    let out_path = assets_dir.join(format!("../test_output_comet.png"));
    img.save(&out_path).expect("Failed to save debug image");

    assert!(non_empty_pixels > 100, "Comet did not render anything!");
    ctx.destroy_presentation_engine(scene_id, _pe).unwrap();
  }
}
