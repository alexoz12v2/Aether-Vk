use aethervk_core_rlib::gpu::PresentationEngineHandle;
use aethervk_core_rlib::scene::text::FontAtlas;
use aethervk_core_rlib::scene::{GridComponent, HiddenComponent};
use aethervk_core_rlib::simulation_api::SimulationContext;
use aethervk_core_rlib::simulation_api::structs::{CustomRenderCallback, SendPtrMut};
use aethervk_core_rlib::types::GpuResult;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use std::sync::Arc;

pub struct AppState {
  pub ctx: Box<SimulationContext>,
  pub custom_data: Arc<std::sync::Mutex<CustomRenderData>>,
  pub scene_id: u64,
  pub presentation_engine: PresentationEngineHandle,
  pub camera_entity: u64,
  pub window: Option<winit::window::Window>,
  pub is_resizing: bool,
  pub is_exiting: bool,
  pub is_command_prompt_open: bool,
  pub console_open_progress: f32,
  pub console_scroll_offset: usize,
  pub command_history: std::collections::VecDeque<String>,
  pub current_command: String,
}

impl Drop for AppState {
  fn drop(&mut self) {
    println!("Dropping AppState");
  }
}

pub struct SimApp {
  pub app_state: AppState,
  pub right_mouse_button_down: bool,
  pub middle_mouse_button_down: bool,
  pub ctrl_down: bool,
  pub mouse_x: f64,
  pub mouse_y: f64,
  pub last_sim_time: std::time::Instant,
  pub window_info: crate::WindowPlatformData,
}

impl SimApp {
  pub fn new(
    ctx: Box<SimulationContext>,
    custom_data: Arc<std::sync::Mutex<CustomRenderData>>,
    scene_id: u64,
    presentation_engine: PresentationEngineHandle,
    camera_entity: u64,
    window: winit::window::Window,
    window_info: crate::WindowPlatformData,
  ) -> Self {
    let app_state = AppState {
      ctx,
      custom_data,
      scene_id,
      presentation_engine,
      camera_entity,
      window: Some(window),
      is_resizing: false,
      is_exiting: false,
      is_command_prompt_open: false,
      console_open_progress: 0.0,
      console_scroll_offset: 0,
      command_history: std::collections::VecDeque::with_capacity(1000),
      current_command: String::new(),
    };

    Self {
      app_state,
      right_mouse_button_down: false,
      middle_mouse_button_down: false,
      ctrl_down: false,
      mouse_x: 0.0,
      mouse_y: 0.0,
      last_sim_time: std::time::Instant::now(),
      window_info,
    }
  }
}

impl crate::app::App for SimApp {
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
                      responses.push("  scale <0|1|2|3>    - Sets the time scale".to_string());
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
                    _ => {
                      responses.push(format!("Unrecognized command: {}", command));
                    }
                  }
                }

                for resp in responses {
                  self.app_state.command_history.push_back(resp);
                }

                if self.app_state.command_history.len() > 1000 {
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

          if let Some(axis) = crate::command::get_camera_movement_axis(keycode) {
            let scene = self.app_state.ctx.get_scene(self.app_state.scene_id).unwrap();
            let _ = self.app_state.ctx.threads.logic_thread.tx().try_send(
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
              winit::keyboard::KeyCode::KeyG => {
                let scene_id = self.app_state.scene_id;
                if let Some(scene_ctx) = self.app_state.ctx.get_scene(scene_id) {
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
      Some(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PanCursor(
          aethervk_core_rlib::simulation_api::structs::PanCursor {
            scene: scene.clone(),
            delta_x: delta.0 as f32,
            delta_y: delta.1 as f32,
          },
        ),
      )
    } else {
      None
    };

    if let Some(logic_command) = logic_command {
      let _ = ctx.threads.logic_thread.tx().try_send(logic_command);
    }
  }

  fn on_modifiers_changed(&mut self, modifiers: winit::keyboard::ModifiersState) {
    self.ctrl_down = modifiers.control_key() || modifiers.super_key();
  }

  fn on_redraw(&mut self) {}

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
    }
  }
}

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
    return Ok(()); // Font not ready yet
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

  let _ = device
    .prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);
  let _ = device.render_text(
    cmd_buffer,
    &mem_text,
    [-0.98, -0.95],
    screen_extent,
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
