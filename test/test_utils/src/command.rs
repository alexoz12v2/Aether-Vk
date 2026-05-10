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

use aethervk_core_rlib::gpu;
use aethervk_core_rlib::scene::EntityId;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
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

pub fn process_mouse_motion_camera_commands(
  delta: (f64, f64),
  middle_down: bool,
  shift_down: bool,
  ctrl_down: bool,
  camera_entity: EntityId,
  scene: std::sync::Arc<gpu::RwLock<aethervk_core_rlib::simulation_api::structs::SceneContext>>,
) -> Option<aethervk_core_rlib::simulation_api::structs::LogicCommand> {
  if middle_down {
    if shift_down {
      Some(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::PanCamera(
          aethervk_core_rlib::simulation_api::structs::PanCamera {
            camera_entity,
            scene,
            delta_x: delta.0 as f32,
            delta_y: delta.1 as f32,
          },
        ),
      )
    } else if ctrl_down {
      Some(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::ZoomCamera(
          aethervk_core_rlib::simulation_api::structs::ZoomCamera {
            camera_entity,
            scene,
            amount: (delta.1 * 0.1) as f32,
          },
        ),
      )
    } else {
      Some(
        aethervk_core_rlib::simulation_api::structs::LogicCommand::RotateCamera(
          aethervk_core_rlib::simulation_api::structs::RotateCamera {
            camera_entity,
            scene,
            delta_x: delta.0 as f32,
            delta_y: delta.1 as f32,
          },
        ),
      )
    }
  } else {
    None
  }
}
