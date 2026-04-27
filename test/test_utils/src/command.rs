use std::collections::HashMap;
use std::sync::mpsc;

pub type CommandFn<T> = Box<dyn Fn(&mut T, &[&str], &mpsc::Sender<String>) + Send + Sync + 'static>;

pub struct CommandRegistry<T> {
  commands: HashMap<String, CommandFn<T>>,
}

impl<T> CommandRegistry<T> {
  pub fn new() -> Self {
    Self {
      commands: HashMap::new(),
    }
  }

  pub fn register<F>(&mut self, name: &str, func: F)
  where
    F: Fn(&mut T, &[&str], &mpsc::Sender<String>) + Send + Sync + 'static,
  {
    self.commands.insert(name.to_string(), Box::new(func));
  }

  pub fn execute(&self, cmd_line: &str, state: &mut T, tx: &mpsc::Sender<String>) {
    let mut parts = cmd_line.split_whitespace();
    if let Some(cmd_name) = parts.next() {
      let args: Vec<&str> = parts.collect();
      if let Some(cmd_fn) = self.commands.get(cmd_name) {
        cmd_fn(state, &args, tx);
      } else {
        let _ = tx.send(format!("Unknown command: {}", cmd_name));
      }
    }
  }
}

use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::Vector3;
use winit::keyboard::KeyCode;

pub fn get_camera_movement_axis(keycode: KeyCode) -> Option<Vec3f32> {
  match keycode {
    KeyCode::ArrowUp => Some(Vec3f32::from_components(0.0, 0.0, -1.0)),
    KeyCode::ArrowDown => Some(Vec3f32::from_components(0.0, 0.0, 1.0)),
    KeyCode::ArrowLeft => Some(Vec3f32::from_components(-1.0, 0.0, 0.0)),
    KeyCode::ArrowRight => Some(Vec3f32::from_components(1.0, 0.0, 0.0)),
    KeyCode::KeyQ => Some(Vec3f32::from_components(0.0, -1.0, 0.0)),
    KeyCode::KeyE => Some(Vec3f32::from_components(0.0, 1.0, 0.0)),
    _ => None,
  }
}

