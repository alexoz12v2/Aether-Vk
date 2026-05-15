//! Reusable input form rendering and logic.

#[derive(Default)]
pub struct InputForm {
  pub is_active: bool,
  pub buffer: String,
  pub prompt: String,
}

impl InputForm {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn activate(&mut self, prompt: String) {
    self.is_active = true;
    self.prompt = prompt;
    self.buffer.clear();
  }

  pub fn update_text_component(&self, text: &mut String) {
    if self.is_active {
      text.push_str("\n--- INPUT FORM ---\n");
      text.push_str(&self.prompt);
      text.push_str(&self.buffer);
    }
  }

  pub fn handle_event(&mut self, event: &winit::event::KeyEvent, on_submit: &mut impl FnMut(&str)) -> bool {
    if !self.is_active {
      return false;
    }
    use winit::keyboard::{KeyCode, PhysicalKey};

    if event.state == winit::event::ElementState::Pressed {
      match event.physical_key {
        PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
          self.is_active = false;
          on_submit(&self.buffer);
        }
        PhysicalKey::Code(KeyCode::Backspace) => {
          self.buffer.pop();
        }
        PhysicalKey::Code(KeyCode::Space) => {
          self.buffer.push(' ');
        }
        PhysicalKey::Code(KeyCode::Period) => {
          self.buffer.push('.');
        }
        PhysicalKey::Code(c) => {
            let char_opt = match c {
              KeyCode::Digit0 | KeyCode::Numpad0 => Some('0'),
              KeyCode::Digit1 | KeyCode::Numpad1 => Some('1'),
              KeyCode::Digit2 | KeyCode::Numpad2 => Some('2'),
              KeyCode::Digit3 | KeyCode::Numpad3 => Some('3'),
              KeyCode::Digit4 | KeyCode::Numpad4 => Some('4'),
              KeyCode::Digit5 | KeyCode::Numpad5 => Some('5'),
              KeyCode::Digit6 | KeyCode::Numpad6 => Some('6'),
              KeyCode::Digit7 | KeyCode::Numpad7 => Some('7'),
              KeyCode::Digit8 | KeyCode::Numpad8 => Some('8'),
              KeyCode::Digit9 | KeyCode::Numpad9 => Some('9'),
              KeyCode::KeyA => Some('A'),
              KeyCode::KeyB => Some('B'),
              KeyCode::KeyC => Some('C'),
              KeyCode::KeyD => Some('D'),
              KeyCode::KeyE => Some('E'),
              KeyCode::KeyF => Some('F'),
              KeyCode::KeyG => Some('G'),
              KeyCode::KeyH => Some('H'),
              KeyCode::KeyI => Some('I'),
              KeyCode::KeyJ => Some('J'),
              KeyCode::KeyK => Some('K'),
              KeyCode::KeyL => Some('L'),
              KeyCode::KeyM => Some('M'),
              KeyCode::KeyN => Some('N'),
              KeyCode::KeyO => Some('O'),
              KeyCode::KeyP => Some('P'),
              KeyCode::KeyQ => Some('Q'),
              KeyCode::KeyR => Some('R'),
              KeyCode::KeyS => Some('S'),
              KeyCode::KeyT => Some('T'),
              KeyCode::KeyU => Some('U'),
              KeyCode::KeyV => Some('V'),
              KeyCode::KeyW => Some('W'),
              KeyCode::KeyX => Some('X'),
              KeyCode::KeyY => Some('Y'),
              KeyCode::KeyZ => Some('Z'),
              _ => None,
            };
            if let Some(ch) = char_opt {
              self.buffer.push(ch);
            }
        }
        _ => {}
      }
      return true;
    }
    false
  }
}
