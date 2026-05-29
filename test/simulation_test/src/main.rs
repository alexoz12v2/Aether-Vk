#[cfg(all(debug_assertions, feature = "debug_gpu"))]
#[global_allocator]
static ALLOC: aethervk_oshal_rlib::os::memory::tracking::TrackingAllocator<std::alloc::System> =
  aethervk_oshal_rlib::os::memory::tracking::TrackingAllocator(std::alloc::System);

use aethervk_core_rlib::{
  gpu::PresentationEngineHandle,
  scene::{
    AlmanacPlanet, GizmoComponent, GridComponent, HiddenComponent, PhysicalMeshComponent,
    TransformComponent, text::FontAtlas,
  },
  simulation::constants,
  simulation_api::{
    SimulationContext, structs,
    structs::{CustomRenderCallback, SendPtrMut},
  },
  types::{EngineResult, GpuResult},
};
use aethervk_oshal_rlib::math::{
  matrix::SquareMatrix,
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};
use std::{
  sync::{Arc, atomic::AtomicBool},
  time::Instant,
};
use test_utils::{
  cycle_get_asset_path_from_exe,
  sim_app::{SimulationDelegate, run_simulation_app},
};
use winit::window::Window;

#[derive(Default)]
pub struct CustomRenderData {
  pub console_open_progress: f32,
  pub console_scroll_offset: usize,
  pub command_history: std::collections::VecDeque<String>,
  pub current_command: String,
  pub font_atlas: Option<FontAtlas>,
  pub size: winit::dpi::PhysicalSize<u32>,
}

impl CustomRenderData {
  pub fn make_callback(&mut self, arc_self: Arc<std::sync::Mutex<Self>>) -> CustomRenderCallback {
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
    let font_internal =
      device.allocate_rasterized_font_atlas(cmd_buffer, font_hash, std::sync::Arc::new(atlas))?;
    FONT_HASH.store(font_hash, std::sync::atomic::Ordering::Relaxed);
    FONT_INTERNAL.store(font_internal, std::sync::atomic::Ordering::Relaxed);
  }
  Ok(())
}

fn ui_custom_render_fn(
  device: &dyn aethervk_core_rlib::gpu::RenderDevice,
  cmd_buffer: aethervk_core_rlib::gpu::CommandBufferHandle,
  presentation_engine_handle: PresentationEngineHandle,
  _render_scene: &aethervk_core_rlib::gpu::RenderScene,
  user_data: *mut core::ffi::c_void,
) -> GpuResult<()> {
  let data_mutex = unsafe { &*(user_data as *const std::sync::Mutex<CustomRenderData>) };
  let data = data_mutex.lock().unwrap();

  let size = data.size;
  let font_id = (
    FONT_HASH.load(std::sync::atomic::Ordering::Relaxed),
    FONT_INTERNAL.load(std::sync::atomic::Ordering::Relaxed),
  );
  if font_id.0 == 0 {
    return Ok(()); // Font not ready yet
  }

  #[rustfmt::skip]
  let view_proj_arr: [f32; 16] = [
    2.0 / size.width as f32, 0.0, 0.0, 0.0,
    0.0, 2.0 / size.height as f32, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    -1.0, -1.0, 0.0, 1.0,
  ];

  let mut mem_text = String::new();
  if let Some(vk_dev) = device
    .as_any()
    .downcast_ref::<aethervk_core_rlib::gpu_backends::vulkan::device::Device>()
  {
    let (budget, usage) = vk_dev.get_vma_budget_usage();
    mem_text.push_str(&format!(
      "VMA Usage: {:.2} MB / {:.2} MB\n",
      usage as f32 / 1048576.0,
      budget as f32 / 1048576.0
    ));
  }

  let _ = device
    .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);
  let _ = device.render_text(
    cmd_buffer,
    &mem_text,
    [10.0, 10.0],
    view_proj_arr,
    font_id,
    16.0,
    [0.0, 1.0, 0.0, 1.0],
  );

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
      [10.0, (size.height as f32) / 2.0], // Arbitrary position for UI coordinates
      view_proj_arr,
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
      [10.0, (size.height as f32) - 30.0],
      view_proj_arr,
      font_id,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
    ) {
      println!("Render text error prompt: {:?}", e);
    }
  }

  Ok(())
}

struct SimulationPlaygroundDelegate {
  custom_data: Arc<std::sync::Mutex<CustomRenderData>>,
  is_command_prompt_open: bool,
  console_open_progress: f32,
  console_scroll_offset: usize,
  command_history: std::collections::VecDeque<String>,
  current_command: String,
  ext_camera_entity: u64,
}

impl SimulationDelegate for SimulationPlaygroundDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_default_scene(true)
  }

  fn on_setup(
    &mut self,
    simulation_context: &mut SimulationContext,
    scene_id: u64,
    presentation_engine_handle: PresentationEngineHandle,
    window: &Window,
  ) -> EngineResult<()> {
    let assets_dir = cycle_get_asset_path_from_exe(true);
    let jpl_service = test_utils::horizon_jpl::HorizonJplService::new().unwrap();

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

      // Fetch Comet 67P SPK and physical constants using Horizon JPL Service
      let spk_dir = assets_dir.join("planets");
      let spk_filename = "67P.bsp";
      let _ = jpl_service.download_spk(
        "90000702",
        "2024-01-01",
        "2025-01-01",
        &spk_dir,
        spk_filename,
      );

      let mut comet_radius_km = 1.7; // default 67P radius
      let mut comet_rot_period = 12.4043;
      if let Ok(c_str) = jpl_service.download_object_constants("90000702") {
        let (r, rot, _) = jpl_service.parse_object_constants(&c_str);
        if let Some(rad) = r {
          comet_radius_km = rad;
        }
        if let Some(rot) = rot {
          comet_rot_period = rot as f64;
        }
      }

      let comet_visual_radius = (comet_radius_km / constants::DISTANCE_SCALE_FACTOR as f32)
        * constants::PLANET_VISUAL_SCALE;

      let mesh_entity = active_scene.scene.spawn_entity("comet");

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
            scale: Vec3f32::from_components(
              comet_visual_radius,
              comet_visual_radius,
              comet_visual_radius,
            ),
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
            use_new_path: false,
            paint_display_mode: 0,
            sphere_center: [0.0, 0.0, 0.0],
            sphere_radius: 1.0,
            grid_color: [0.0, 0.0, 0.0],
            grid_density: 1.0,
          },
        )
        .unwrap();
      active_scene
        .scene
        .add_component(
          mesh_entity,
          AlmanacPlanet::new(90000702, comet_rot_period, 1.0),
        )
        .unwrap();
      active_scene.scene.set_parent(mesh_entity, Some(root_entity));
      active_scene.register_entity(mesh_entity);

      let uv_dist = aethervk_core_rlib::simulation::utils::generate_gaussian_distribution(
        64, 0.5, 0.5, 0.5, 0.5,
      );
      active_scene
        .scene
        .add_component(
          mesh_entity,
          aethervk_core_rlib::scene::ParticleSystemComponent::new(100_000),
        )
        .unwrap();

      active_scene
        .scene
        .add_component(
          mesh_entity,
          aethervk_core_rlib::scene::ParticleEmitterComponent {
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
            use_particle2: false,
          },
        )
        .unwrap();

      // -------------------------------------------------------------
      // Add physical_mesh2 variants
      // -------------------------------------------------------------
      let sphere_variant =
        aethervk_core_rlib::simulation::comet::generate_uv_sphere(0.5, 32, 32, 1.0);
      let sphere_variant_arc = Arc::from(sphere_variant);

      let mut add_variant = |name: &str,
                             pos: Vec3f32,
                             intensity: f32,
                             color: [f32; 3],
                             paint_mode: u32,
                             is_outline: bool| {
        let entity = active_scene.scene.spawn_entity(name);
        active_scene.scene.set_parent(entity, Some(root_entity));
        active_scene
          .scene
          .add_component(
            entity,
            TransformComponent {
              position: pos,
              rotation: Quat::identity(),
              scale: Vec3f32::from_components(1.0, 1.0, 1.0),
            },
          )
          .unwrap();
        active_scene
          .scene
          .add_component(
            entity,
            PhysicalMeshComponent {
              asset_path: "".to_string(),
              mesh: sphere_variant_arc.clone(),
              emissive_intensity: intensity,
              emissive_color: color,
              use_new_path: true,
              paint_display_mode: paint_mode,
              sphere_center: [0.0, 0.0, 0.0],
              sphere_radius: 1.0,
              grid_color: [0.0, 0.0, 0.0],
              grid_density: 1.0,
            },
          )
          .unwrap();
        if is_outline {
          active_scene
            .scene
            .add_component(entity, aethervk_core_rlib::scene::SelectedComponent {})
            .unwrap();
        }
        active_scene.register_entity(entity);
      };

      add_variant(
        "pm2_normal",
        Vec3f32::from_components(10.0, -5.0, 0.0),
        0.0,
        [0.0, 0.0, 0.0],
        0,
        false,
      );
      add_variant(
        "pm2_emissive",
        Vec3f32::from_components(10.0, -2.5, 0.0),
        5.0,
        [1.0, 0.0, 0.0],
        0,
        false,
      );
      add_variant(
        "pm2_painted",
        Vec3f32::from_components(10.0, 0.0, 0.0),
        0.0,
        [0.0, 1.0, 0.0],
        1,
        false,
      );
      add_variant(
        "pm2_outlined",
        Vec3f32::from_components(10.0, 2.5, 0.0),
        0.0,
        [0.0, 0.0, 1.0],
        0,
        true,
      );

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
        let mut planet_radius = (aethervk_core_rlib::simulation::utils::get_planet_radius(
          *naif_id,
          assets_dir.to_str().unwrap(),
        ) / constants::DISTANCE_SCALE_FACTOR as f32)
          * constants::PLANET_VISUAL_SCALE;

        let mut planet_rot_period = *rot_period;

        if let Ok(c_str) = jpl_service.download_object_constants(&naif_id.to_string()) {
          let (r, rot, _) = jpl_service.parse_object_constants(&c_str);
          if let Some(rad) = r {
            planet_radius =
              (rad / constants::DISTANCE_SCALE_FACTOR as f32) * constants::PLANET_VISUAL_SCALE;
          }
          if let Some(rot) = rot {
            planet_rot_period = rot as f64;
          }
        }

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
              use_new_path: false,
              paint_display_mode: 0,
              sphere_center: [0.0, 0.0, 0.0],
              sphere_radius: 1.0,
              grid_color: [0.0, 0.0, 0.0],
              grid_density: 1.0,
            },
          )
          .unwrap();
        active_scene
          .scene
          .add_component(
            planet_entity,
            AlmanacPlanet::new(*naif_id, planet_rot_period, 1.0),
          )
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

    self.ext_camera_entity = {
      let scene_ctx = simulation_context.get_scene(scene_id).unwrap();
      let read_ctx = scene_ctx.read();
      let int_cam = read_ctx
        .presentation_engines
        .read()
        .get(&presentation_engine_handle)
        .unwrap()
        .camera_entity
        .unwrap();
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

    let _ = simulation_context.threads.logic_thread.tx().try_send(
      aethervk_core_rlib::simulation_api::structs::LogicCommand::SetSceneTimeScale {
        scene_id,
        scale: aethervk_core_rlib::simulation_api::structs::TimeScale::OneDay,
      },
    );
    let _ =
      simulation_context.threads.logic_thread.tx().try_send(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id },
      );

    let mut custom_render_data = CustomRenderData::default();
    custom_render_data.font_atlas = Some(atlas);
    self.custom_data = Arc::new(std::sync::Mutex::new(custom_render_data));

    let arc_scenes = simulation_context.get_scene(scene_id).unwrap();
    let mut scene_write = arc_scenes.write();
    let cb = self.custom_data.lock().unwrap().make_callback(Arc::clone(&self.custom_data));
    scene_write.register_custom_render_callback(Some(cb));
    drop(scene_write);
    drop(arc_scenes);
    Ok(())
  }

  fn on_about_to_wait(&mut self, ctx: &mut SimulationContext, scene_id: u64, delta_time: f32) {
    #[cfg(all(debug_assertions, feature = "debug_gpu"))]
    {
      static FRAMES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
      if FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 120 == 0 {
        aethervk_oshal_rlib::os::memory::tracking::print_memory_state();
        let _ = ctx.with_device(|device| {
          device.dump_memory_stats();
          Ok(())
        });
      }
    }
    let dt = delta_time;
    if self.is_command_prompt_open {
      self.console_open_progress += dt * 5.0;
      if self.console_open_progress > 1.0 {
        self.console_open_progress = 1.0;
      }
    } else {
      self.console_open_progress -= dt * 5.0;
      if self.console_open_progress < 0.0 {
        self.console_open_progress = 0.0;
      }
    }

    let mut lock = self.custom_data.lock().unwrap();
    lock.console_open_progress = self.console_open_progress;
    lock.console_scroll_offset = self.console_scroll_offset;
    lock.command_history = self.command_history.clone();
    lock.current_command = self.current_command.clone();
  }

  fn on_keyboard_input(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    event: &winit::event::KeyEvent,
    modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state == winit::event::ElementState::Pressed {
      if self.is_command_prompt_open {
        if let winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Escape) =
          event.physical_key
        {
          self.is_command_prompt_open = false;
        } else {
          match &event.logical_key {
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace) => {
              self.current_command.pop();
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageUp) => {
              self.console_scroll_offset = self.console_scroll_offset.saturating_add(1);
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageDown) => {
              self.console_scroll_offset = self.console_scroll_offset.saturating_sub(1);
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) => {
              if !self.current_command.is_empty() {
                let cmd = self.current_command.clone();
                self.command_history.push_back(format!("> {}", cmd));

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
                      self.command_history.clear();
                    }
                    "play" => {
                      let _ = ctx.threads.logic_thread.tx().try_send(
                        aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene {
                          scene_id,
                        },
                      );
                      responses.push("Simulation playing.".to_string());
                    }
                    "pause" => {
                      let _ = ctx.threads.logic_thread.tx().try_send(
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
                        let _ = ctx.threads.logic_thread.tx().try_send(
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
                          let _ = ctx.threads.logic_thread.tx().try_send(
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
                      if let Some(scene_ctx) = ctx.get_scene(scene_id) {
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
                        if let Some(scene_ctx) = ctx.get_scene(scene_id) {
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
                      if let Some(scene_ctx) = ctx.get_scene(scene_id) {
                        let read_ctx = scene_ctx.read();
                        let sel = read_ctx
                          .scene
                          .query1_first_res::<aethervk_core_rlib::scene::SelectedComponent, _, _>(
                            |id, _| Some(id),
                          );
                        if let Some((id, _)) = sel {
                          if let Some(name) = read_ctx.scene.get_name(id) {
                            responses.push(format!("Selected: '{}'", name));
                          } else {
                            responses.push(format!("Selected: ID {:?}", id));
                          }
                        } else {
                          responses.push("Nothing selected.".to_string());
                        }
                      }
                    }
                    _ => {
                      responses.push(format!("Unrecognized command: {}", command));
                    }
                  }
                }

                for resp in responses {
                  self.command_history.push_back(resp);
                }

                if self.command_history.len() > 1000 {
                  while self.command_history.len() > 1000 {
                    self.command_history.pop_front();
                  }
                }
                self.current_command.clear();
                self.console_scroll_offset = 0;
              }
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
              self.current_command.push(' ');
            }
            winit::keyboard::Key::Character(c) => {
              self.current_command.push_str(c.as_str());
            }
            _ => {}
          }
        }
      } else {
        if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
          let speed = 0.5;

          if let Some(axis) = test_utils::command::get_camera_movement_axis(keycode) {
            let scene = ctx.get_scene(scene_id).unwrap();
            let _ = ctx.threads.logic_thread.tx().try_send(
              aethervk_core_rlib::simulation_api::structs::LogicCommand::MoveCursor(
                aethervk_core_rlib::simulation_api::structs::MoveCursor {
                  scene,
                  delta_x: axis.x() * speed,
                  delta_y: axis.y() * speed,
                  delta_z: axis.z() * speed,
                },
              ),
            );
          } else {
            match keycode {
              winit::keyboard::KeyCode::Digit0 => {
                if let Some(scene_ctx) = ctx.get_scene(scene_id) {
                  let scene = scene_ctx.clone();
                  let camera_entity = scene_ctx.read().get_entity(self.ext_camera_entity).unwrap();
                  let _ = ctx.threads.logic_thread.tx().try_send(
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
                self.is_command_prompt_open = true;
              }
              winit::keyboard::KeyCode::KeyP => {
                let is_playing = {
                  let scene = ctx.get_scene(scene_id).unwrap();
                  scene.read().time_state.read().is_playing
                };
                let cmd = if is_playing {
                  aethervk_core_rlib::simulation_api::structs::LogicCommand::PauseScene { scene_id }
                } else {
                  aethervk_core_rlib::simulation_api::structs::LogicCommand::PlayScene { scene_id }
                };
                let _ = ctx.threads.logic_thread.tx().try_send(cmd);
              }
              winit::keyboard::KeyCode::KeyG => {
                if let Some(scene_ctx) = ctx.get_scene(scene_id) {
                  let read_ctx = scene_ctx.read();
                  if let Some((_, grid_entity)) =
                    read_ctx.scene.query1_first_res(|_, _g: &GridComponent| Some(()))
                  {
                    if read_ctx.scene.has_component::<HiddenComponent>(grid_entity).into() {
                      let _ = read_ctx.scene.remove_component::<HiddenComponent>(grid_entity);
                    } else {
                      let _ = read_ctx
                        .scene
                        .add_component::<HiddenComponent>(grid_entity, HiddenComponent {});
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

  fn on_mouse_input(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
    mouse_pos: (f64, f64),
  ) {
    if button == winit::event::MouseButton::Left {
      if state == winit::event::ElementState::Pressed {
        let size = self.custom_data.lock().unwrap().size;
        if size.width > 0 && size.height > 0 {
          let ndc_x = (mouse_pos.0 as f32 / size.width as f32) * 2.0 - 1.0;
          let ndc_y = (mouse_pos.1 as f32 / size.height as f32) * 2.0 - 1.0;
          let _ = ctx.raycast_ndc(scene_id, self.ext_camera_entity, ndc_x, ndc_y);
        }
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
    let camera_entity = scene.read().get_entity(self.ext_camera_entity).expect(&format!(
      "There is not camera entity with id {} in scene {}",
      self.ext_camera_entity, scene_id
    ));

    let logic_command = test_utils::command::process_mouse_motion_camera_commands(
      delta,
      middle_mouse_down,
      shift_down,
      ctrl_down,
      camera_entity,
      scene.clone(),
    );

    if let Some(logic_command) = logic_command {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }

  fn on_mouse_wheel(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    delta: winit::event::MouseScrollDelta,
  ) {
    let scroll_amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32,
    };
    if self.is_command_prompt_open {
      if scroll_amount > 0.0 {
        self.console_scroll_offset = self.console_scroll_offset.saturating_add(1);
      } else if scroll_amount < 0.0 {
        self.console_scroll_offset = self.console_scroll_offset.saturating_sub(1);
      }
    }
  }

  fn on_resize(&mut self, ctx: &mut SimulationContext, scene_id: u64, width: u32, height: u32) {
    let mut lock = self.custom_data.lock().unwrap();
    lock.size = winit::dpi::PhysicalSize::new(width, height);
  }
}

fn main() {
  let custom_data = Arc::new(std::sync::Mutex::new(CustomRenderData::default()));
  let delegate = SimulationPlaygroundDelegate {
    custom_data,
    is_command_prompt_open: false,
    console_open_progress: 0.0,
    console_scroll_offset: 0,
    command_history: std::collections::VecDeque::new(),
    current_command: String::new(),
    ext_camera_entity: 0,
  };
  run_simulation_app("AetherVk Simulation", delegate);
}

#[cfg(test)]
mod depth_tests {
  use super::*;
  use aethervk_core_rlib::{
    gpu,
    simulation::comet::{TexelFormat, Texture},
    simulation_api::components_api::CameraParams,
  };
  use bytes::Bytes;
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
      data: Bytes::from(u8_color),
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

    let scene_id = ctx.create_empty_scene(true).unwrap();
    let width = 64;
    let height = 64;
    let _pe = ctx.create_presentation_engine(scene_id, width, height).unwrap();
    let ext_cam = ctx
      .add_orthographic_camera(scene_id, _pe, "camera", -10.0, -10.0, 0.1, 100.0)
      .unwrap()
      .get();
    ctx
      .set_transform_component(
        scene_id,
        ext_cam,
        Vec3f32::from_components(0.0, 0.0, 0.0),
        Quat::identity(),
        Vec3f32::from_components(1.0, 1.0, 1.0),
      )
      .unwrap();

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
      ctx
        .get_scene(scene_id)
        .unwrap()
        .read()
        .outlines_enabled
        .store(true, Ordering::Relaxed);
    } else {
      ctx
        .get_scene(scene_id)
        .unwrap()
        .read()
        .outlines_enabled
        .store(false, Ordering::Relaxed);
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

    // Camera setup moved up

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

    let scene_id = ctx.create_empty_scene(true).unwrap();
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
            use_new_path: false,
            paint_display_mode: 0,
            sphere_center: [0.0, 0.0, 0.0],
            sphere_radius: 1.0,
            grid_color: [0.0, 0.0, 0.0],
            grid_density: 1.0,
          },
        )
        .unwrap();
    }

    // Camera setup moved up

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
