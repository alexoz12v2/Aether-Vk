//! Reusable console for testing binaries.

use aethervk_core_rlib::gpu::{PresentationEngineHandle, RenderDevice, CommandBufferHandle};
use aethervk_core_rlib::types::GpuResult;
use std::collections::VecDeque;

#[derive(Default)]
pub struct Console {
  pub is_open: bool,
  pub open_progress: f32,
  pub scroll_offset: usize,
  pub command_history: VecDeque<String>,
  pub current_command: String,
}

impl Console {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn update(&mut self, dt: f32) {
    if self.is_open {
      self.open_progress += dt * 5.0;
      if self.open_progress > 1.0 {
        self.open_progress = 1.0;
      }
    } else {
      self.open_progress -= dt * 5.0;
      if self.open_progress < 0.0 {
        self.open_progress = 0.0;
      }
    }
  }

  pub fn render(
    &self,
    device: &dyn RenderDevice,
    cmd_buffer: CommandBufferHandle,
    presentation_engine_handle: PresentationEngineHandle,
    font_id: (u64, u32),
    size: winit::dpi::PhysicalSize<u32>,
  ) -> GpuResult<()> {
    if self.open_progress <= 0.0 {
      return Ok(());
    }

    #[rustfmt::skip]
    let view_proj_arr: [f32; 16] = [
      2.0 / size.width as f32, 0.0, 0.0, 0.0,
      0.0, 2.0 / size.height as f32, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      -1.0, -1.0, 0.0, 1.0,
    ];

    let width = 2.0;
    let height = 1.0;
    let box_y = -1.0 - height + (self.open_progress * height);

    let _ = device.render_ui_rect(
      cmd_buffer,
      presentation_engine_handle,
      [0.05, 0.1, 0.05, 0.7],
      [-1.0, box_y],
      [width, height],
    );

    let mut console_text = String::new();
    let max_lines = 12;
    let history_len = self.command_history.len();
    let scroll = self.scroll_offset.min(history_len.saturating_sub(max_lines));
    let start_idx = history_len.saturating_sub(max_lines + scroll);
    let end_idx = history_len.saturating_sub(scroll);

    for cmd in self.command_history.iter().skip(start_idx).take(end_idx - start_idx) {
      console_text.push_str(cmd);
      console_text.push('\n');
    }

    let _ = device.prepare_text_archetype_for_render_and_bind_pipeline(cmd_buffer, presentation_engine_handle);

    let _ = device.render_text(
      cmd_buffer,
      &console_text,
      [10.0, (size.height as f32) / 2.0], // Arbitrary position for UI coordinates
      view_proj_arr,
      font_id,
      14.0,
      [0.8, 0.8, 0.8, 1.0],
    );

    let mut prompt_text = String::from("> ");
    prompt_text.push_str(&self.current_command);
    prompt_text.push('_');

    let _ = device.render_text(
      cmd_buffer,
      &prompt_text,
      [10.0, (size.height as f32) - 30.0],
      view_proj_arr,
      font_id,
      16.0,
      [1.0, 1.0, 0.2, 1.0],
    );

    Ok(())
  }

  pub fn handle_event(
    &mut self,
    event: &winit::event::KeyEvent,
    on_command: &mut impl FnMut(&mut Self, &str),
  ) -> bool {
    if !self.is_open {
      return false;
    }
    use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};

    if event.state == winit::event::ElementState::Pressed {
      if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
        self.is_open = false;
        return true;
      }

      match &event.logical_key {
        Key::Named(NamedKey::Backspace) => {
          self.current_command.pop();
        }
        Key::Named(NamedKey::PageUp) => {
          self.scroll_offset = self.scroll_offset.saturating_add(1);
        }
        Key::Named(NamedKey::PageDown) => {
          self.scroll_offset = self.scroll_offset.saturating_sub(1);
        }
        Key::Named(NamedKey::Enter) => {
          if !self.current_command.is_empty() {
            let cmd = self.current_command.clone();
            self.command_history.push_back(format!("> {}", cmd));
            
            on_command(self, &cmd);

            if self.command_history.len() > 1000 {
              while self.command_history.len() > 1000 {
                self.command_history.pop_front();
              }
            }
            self.current_command.clear();
            self.scroll_offset = 0;
          }
        }
        Key::Named(NamedKey::Space) => {
          self.current_command.push(' ');
        }
        Key::Character(c) => {
          self.current_command.push_str(c.as_str());
        }
        _ => {
          // Additional handling for NumpadEnter and space physical keys
          match event.physical_key {
            PhysicalKey::Code(KeyCode::NumpadEnter) => {
              if !self.current_command.is_empty() {
                let cmd = self.current_command.clone();
                self.command_history.push_back(format!("> {}", cmd));
                
                on_command(self, &cmd);

                if self.command_history.len() > 1000 {
                  while self.command_history.len() > 1000 {
                    self.command_history.pop_front();
                  }
                }
                self.current_command.clear();
                self.scroll_offset = 0;
              }
            }
            PhysicalKey::Code(KeyCode::Space) => {
              self.current_command.push(' ');
            }
            _ => {}
          }
        }
      }
      return true;
    }
    false
  }
}
