use crate::cycle_get_asset_path_from_exe;
use aethervk_core_rlib::{
  gpu,
  gpu::{ASSET_DIR, PresentationEngineHandle},
  simulation_api::SimulationContext,
  types::EngineResult,
};
use winit::window::Window;

pub trait SimulationDelegate {
  fn create_scene(&mut self, ctx: &mut SimulationContext) -> EngineResult<u64> {
    ctx.create_empty_scene(true)
  }

  fn on_setup(
    &mut self,
    ctx: &mut SimulationContext,
    scene_id: u64,
    pe_handle: PresentationEngineHandle,
    window: &Window,
  ) -> EngineResult<()>;

  fn on_about_to_wait(&mut self, _ctx: &mut SimulationContext, _scene_id: u64, _delta_time: f32) {}

  fn on_keyboard_input(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _event: &winit::event::KeyEvent,
    _modifiers: winit::keyboard::ModifiersState,
  ) {
  }

  fn on_mouse_input(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _button: winit::event::MouseButton,
    _state: winit::event::ElementState,
    _mouse_pos: (f64, f64),
  ) {
  }

  fn on_cursor_moved(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _position: winit::dpi::PhysicalPosition<f64>,
  ) {
  }

  fn on_mouse_motion(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _delta: (f64, f64),
    _middle_mouse_down: bool,
    _shift_down: bool,
    _ctrl_down: bool,
  ) {
  }

  fn on_mouse_wheel(
    &mut self,
    _ctx: &mut SimulationContext,
    _scene_id: u64,
    _delta: winit::event::MouseScrollDelta,
  ) {
  }

  fn on_resize(&mut self, _ctx: &mut SimulationContext, _scene_id: u64, _width: u32, _height: u32) {
  }
}

pub fn run_simulation_app<D: SimulationDelegate + 'static>(title: &str, mut delegate: D) {
  let start_time = std::time::Instant::now();
  println!(
    "[{:.2?}] Application starting: {}",
    start_time.elapsed(),
    title
  );

  let (window, event_loop) = crate::create_winit_window_and_event_loop(title);

  *ASSET_DIR.write() = Some(cycle_get_asset_path_from_exe(true).to_string_lossy().to_string());

  let mut simulation_context = SimulationContext::startup(
    gpu::VULKAN_RENDER_BACKEND,
    Some(|msg: &str| panic!("Vulkan Error: {}", msg)),
  )
  .unwrap();

  let (native_handles, window_info) = {
    let render_frontend = simulation_context.render_frontend().unwrap();
    let render_device_handle = simulation_context.render_device_handle();
    crate::get_handle_and_window_info_create_layer(&render_frontend, render_device_handle, &window)
  };

  let width = window.inner_size().width;
  let height = window.inner_size().height;

  let scene_id = delegate.create_scene(&mut simulation_context).unwrap();

  let presentation_engine = simulation_context
    .create_presentation_engine_windowed(scene_id, width, height, native_handles)
    .unwrap();

  delegate
    .on_setup(
      &mut simulation_context,
      scene_id,
      presentation_engine,
      &window,
    )
    .unwrap();

  let sim_app = GenericSimApp {
    ctx: simulation_context,
    scene_id,
    presentation_engine,
    window: Some(window),
    window_info,
    delegate,
    is_resizing: false,
    is_exiting: false,
    last_sim_time: std::time::Instant::now(),
    mouse_x: 0.0,
    mouse_y: 0.0,
    middle_mouse_button_down: false,
    shift_down: false,
    ctrl_down: false,
    needs_presentation_resize: false,
    resize_width: width,
    resize_height: height,
  };

  crate::app::run_app(sim_app, event_loop);
}

struct GenericSimApp<D: SimulationDelegate> {
  ctx: Box<SimulationContext>,
  scene_id: u64,
  presentation_engine: PresentationEngineHandle,
  window: Option<Window>,
  window_info: crate::WindowPlatformData,
  delegate: D,
  is_resizing: bool,
  is_exiting: bool,
  last_sim_time: std::time::Instant,
  mouse_x: f64,
  mouse_y: f64,
  middle_mouse_button_down: bool,
  shift_down: bool,
  ctrl_down: bool,
  needs_presentation_resize: bool,
  resize_width: u32,
  resize_height: u32,
}

impl<D: SimulationDelegate> crate::app::App for GenericSimApp<D> {
  fn window(&self) -> Option<&Window> {
    self.window.as_ref()
  }

  fn is_resizing(&self) -> bool {
    self.is_resizing
  }

  fn set_resizing(&mut self, resizing: bool) {
    self.is_resizing = resizing;
  }

  fn is_exiting(&self) -> bool {
    self.is_exiting
  }

  fn set_exiting(&mut self, exiting: bool) {
    self.is_exiting = exiting;
  }

  fn on_resize(&mut self, width: u32, height: u32) {
    self.resize_width = width;
    self.resize_height = height;

    #[cfg(target_os = "macos")]
    {
      use objc2_core_foundation::{CGPoint, CGRect, CGSize};

      // 1. Get the current monitor's scale factor (Retina displays are usually 2.0).
      // Fallback to 1.0 if the window has somehow been destroyed.
      let scale_factor = self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0);

      // 2. Set Physical Pixels (What Vulkan and Metal care about)
      self.window_info.metal_layer.setDrawableSize(CGSize {
        width: width as f64,
        height: height as f64,
      });

      // 3. Set Logical Points (What Core Animation and the GPU Debugger care about)
      let logical_width = (width as f64) / scale_factor;
      let logical_height = (height as f64) / scale_factor;

      self.window_info.metal_layer.setFrame(CGRect {
        origin: CGPoint { x: 0.0, y: 0.0 },
        size: CGSize {
          width: logical_width,
          height: logical_height,
        },
      });

      // 4. Ensure the layer knows how to map the pixels to the points
      // equal to msg_send![&self.window_info.metal_layer, setContentsScale: scale_factor]
      self.window_info.metal_layer.setContentsScale(scale_factor);
    }

    if !self.is_resizing {
      self.needs_presentation_resize = true;
    }

    self.delegate.on_resize(&mut self.ctx, self.scene_id, width, height);
  }

  fn on_resize_ended(&mut self) {
    self.needs_presentation_resize = true;
  }

  fn on_close_requested(&mut self) {
    // Flush all remaining window-tied resources before the device is dropped.
    // This MUST happen on the main thread before Device::Drop runs.
    let _ = self.ctx.flush_main_thread_cleanup_queue();
    self.window = None;
  }

  fn on_mouse_input(
    &mut self,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
  ) {
    if button == winit::event::MouseButton::Middle {
      self.middle_mouse_button_down = state == winit::event::ElementState::Pressed;
    }
    self.delegate.on_mouse_input(
      &mut self.ctx,
      self.scene_id,
      button,
      state,
      (self.mouse_x, self.mouse_y),
    );
  }

  fn on_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
    self.mouse_x = position.x;
    self.mouse_y = position.y;
    self.delegate.on_cursor_moved(&mut self.ctx, self.scene_id, position);
  }

  fn on_keyboard_input(
    &mut self,
    event: &winit::event::KeyEvent,
    modifiers: winit::keyboard::ModifiersState,
  ) {
    #[cfg(target_os = "macos")]
    {
      if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
        if keycode == winit::keyboard::KeyCode::KeyQ && modifiers.super_key() {
          self.is_exiting = true;
          self.on_close_requested();
          return;
        }
      }
    }
    self.delegate.on_keyboard_input(&mut self.ctx, self.scene_id, event, modifiers);
  }

  fn on_mouse_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
    self.delegate.on_mouse_wheel(&mut self.ctx, self.scene_id, delta);
  }

  fn on_mouse_motion(&mut self, delta: (f64, f64)) {
    self.delegate.on_mouse_motion(
      &mut self.ctx,
      self.scene_id,
      delta,
      self.middle_mouse_button_down,
      self.shift_down,
      self.ctrl_down,
    );
  }

  fn on_modifiers_changed(&mut self, modifiers: winit::keyboard::ModifiersState) {
    self.ctrl_down = modifiers.control_key() || modifiers.super_key();
    self.shift_down = modifiers.shift_key();
  }

  fn on_about_to_wait(&mut self) {
    let current_time = std::time::Instant::now();
    let delta_time = current_time.duration_since(self.last_sim_time).as_secs_f32();
    self.last_sim_time = current_time;

    if self.needs_presentation_resize && !self.is_resizing {
      self.needs_presentation_resize = false;
      if self.resize_width > 0 && self.resize_height > 0 {
        let _ = self.ctx.resize(
          self.scene_id,
          self.presentation_engine,
          self.resize_width,
          self.resize_height,
        );
      }
    }

    // Drain main-thread cleanup queue (swapchain/surface destruction from resize)
    let _ = self.ctx.process_main_thread_cleanup_queue();

    self.delegate.on_about_to_wait(&mut self.ctx, self.scene_id, delta_time);
  }
}
