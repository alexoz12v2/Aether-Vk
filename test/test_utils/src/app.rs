use winit::event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{ModifiersState, PhysicalKey, KeyCode};
use crate::{AppEvent, command::get_camera_movement_axis};

pub trait App {
  fn window(&self) -> Option<&winit::window::Window>;
  fn is_resizing(&self) -> bool;
  fn set_resizing(&mut self, resizing: bool);
  fn is_exiting(&self) -> bool;
  fn set_exiting(&mut self, exiting: bool);

  fn on_resize(&mut self, width: u32, height: u32) {}
  fn on_close_requested(&mut self) {}
  fn on_mouse_input(&mut self, _button: MouseButton, _state: ElementState) {}
  fn on_cursor_moved(&mut self, _position: winit::dpi::PhysicalPosition<f64>) {}
  fn on_keyboard_input(&mut self, _event: &winit::event::KeyEvent, _modifiers: ModifiersState) {}
  fn on_mouse_wheel(&mut self, _delta: winit::event::MouseScrollDelta) {}
  fn on_mouse_motion(&mut self, _delta: (f64, f64)) {}
  fn on_redraw(&mut self) {}
  fn on_about_to_wait(&mut self) {}
}

pub fn run_app<A: App + 'static>(mut app: A, event_loop: EventLoop<AppEvent>) {
  let mut modifiers_state = winit::keyboard::ModifiersState::empty();

  event_loop.set_control_flow(ControlFlow::Poll);
  event_loop
    .run(move |event, elwt| match event {
      Event::UserEvent(app_event) => match app_event {
        AppEvent::ResizeStarted => {
          app.set_resizing(true);
        }
        AppEvent::ResizeEnded => {
          app.set_resizing(false);
          if let Some(w) = app.window() {
            w.request_redraw();
          }
        }
      },
      Event::WindowEvent { event, window_id }
        if app.window().map_or(false, |w| w.id() == window_id) =>
      {
        match event {
          WindowEvent::CloseRequested => {
            app.on_close_requested();
            app.set_exiting(true);
            elwt.exit();
          }
          WindowEvent::Resized(physical_size) => {
            app.on_resize(physical_size.width, physical_size.height);
          }
          WindowEvent::ModifiersChanged(modifiers) => {
            modifiers_state = modifiers.state();
          }
          WindowEvent::CursorMoved { position, .. } => {
            app.on_cursor_moved(position);
          }
          WindowEvent::MouseInput { state, button, .. } => {
            app.on_mouse_input(button, state);
          }
          WindowEvent::KeyboardInput { event, .. } => {
            app.on_keyboard_input(&event, modifiers_state);
            if app.is_exiting() {
              elwt.exit();
            }
          }
          WindowEvent::MouseWheel { delta, .. } => {
            app.on_mouse_wheel(delta);
          }
          WindowEvent::RedrawRequested => {
            if !app.is_resizing() && !app.is_exiting() {
              app.on_redraw();
            }
            if app.is_exiting() {
              elwt.exit();
            }
          }
          _ => {}
        }
      }
      Event::DeviceEvent {
        event: DeviceEvent::MouseMotion { delta },
        ..
      } => {
        app.on_mouse_motion(delta);
      }
      Event::LoopExiting => {
        // By the time we hit LoopExiting, winit is already shutting down
        if !app.is_exiting() {
          app.on_close_requested();
          app.set_exiting(true);
        }
      }
      Event::AboutToWait => {
        app.on_about_to_wait();
        if app.is_exiting() {
          elwt.exit();
        }
      }
      _ => {}
    })
    .unwrap();
}
