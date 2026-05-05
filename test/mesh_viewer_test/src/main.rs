use aethervk_core_rlib::{
  types::GpuResult,
  gpu::{self, frame::RenderScene, RenderDevice},
  scene::{
    CameraComponent, EntityId, PhysicalMeshComponent, RenderableDataRef, Scene, SunComponent,
    TransformComponent,
  },
  types::RuntimeParams,
  gpu::{PresentationEngineHandle, ScopedCommandBuffer, ScopedRenderPass},
};
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix4, SquareMatrix},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector3},
};
use heapless::index_map::FnvIndexMap;
use rfd::FileDialog;
use std::sync::Arc;
use test_utils::{
  cycle_get_asset_path_from_exe, get_handle_and_window_info, scene_to_render_scene,
  setup_resize_hook, AppEvent,
};
use winit::{event_loop::EventLoopBuilder, window::WindowBuilder};
use aethervk_core_rlib::gpu::FrameCancelGuard;

struct AppState {
  is_resizing: bool,
  is_exiting: bool,
  scene: Arc<Scene>,
  window: Option<winit::window::Window>,
}

fn main() {
  let mut event_loop_builder = EventLoopBuilder::<AppEvent>::with_user_event();
  #[cfg(target_os = "macos")]
  {
    use winit::platform::macos::EventLoopBuilderExtMacOS;
    event_loop_builder.with_default_menu(false);
  }

  let event_loop = event_loop_builder.build().unwrap();

  let file_path = FileDialog::new()
    .add_filter("GLTF/GLB", &["glb", "gltf"])
    .pick_file();

  let file_path = match file_path {
    Some(p) => p,
    None => {
      println!("No file selected, exiting.");
      return;
    }
  };
  let asset_path = cycle_get_asset_path_from_exe(false);

  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_path.to_str().unwrap().to_string());
  drop(guard);

  let render_frontend = {
    let runtime_params = Box::new(RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
      validation_error_callback: None,
    });
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, &runtime_params).unwrap()
  };

  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = {
    let mut write_render_frontend = render_frontend.write();
    write_render_frontend
      .init_device(0, &additional_params)
      .unwrap()
  };

  let proxy = event_loop.create_proxy();
  let proxy_ptr = unsafe { std::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(proxy))) };

  let window = WindowBuilder::new()
    .with_title("Mesh Viewer")
    .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
    .build(&event_loop)
    .unwrap();
  setup_resize_hook(&window, proxy_ptr);

  let (native_handles, _window_info) =
    get_handle_and_window_info(&render_frontend, render_device_handle, &window);

  let mut width = window.inner_size().width;
  let mut height = window.inner_size().height;

  let presentation_engine = {
    let params = gpu::PresentationEngineParams {
      width,
      height,
      vsync: true,
      ty: gpu::PresentationEngineType::Window,
      window_info: native_handles,
    };
    render_frontend
      .with_device(render_device_handle, |device| {
        device.create_presentation_engine(&params)
      })
      .unwrap()
  };

  let asset_path = cycle_get_asset_path_from_exe(false);
  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_path.to_str().unwrap().to_string());
  drop(guard);

  // archetypes need asset path to locate shaders TODO pass asset path as argument
  render_frontend
    .with_device(render_device_handle, |device| {
      device.init_archetypes(presentation_engine)
    })
    .unwrap();

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene
    .register_component::<PhysicalMeshComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[std::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<aethervk_core_rlib::scene::SelectedComponent>(&[]);

  let camera_entity = scene.spawn_entity("camera");
  let mut cam_dist = 5.0;
  let mut cam_yaw: f32 = std::f32::consts::PI;
  let mut cam_pitch: f32 = 0.0;

  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, -cam_dist, 0.0),
        rotation: Quat::from_axis_angle(
          Vec3f32::from_components(0.0, 0.0, 1.0),
          std::f32::consts::PI,
        ),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(
          std::f32::consts::FRAC_PI_4,
          width as f32 / height as f32,
          0.1,
          100.0,
        ),
        near_plane: 0.1,
        far_plane: 100.0,
      },
    )
    .unwrap();

  let mesh_entity = scene.spawn_entity("mesh");
  scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();

  let loaded_mesh =
    aethervk_core_rlib::simulation::comet::load_comet_from_gltf(file_path.to_str().unwrap(), true)
      .expect("Failed to load mesh");
  scene
    .add_component(
      mesh_entity,
      PhysicalMeshComponent {
        asset_path: file_path.to_str().unwrap().to_string(),
        mesh: Arc::from(loaded_mesh),
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
      },
    )
    .unwrap();

  let sun_entity = scene.spawn_entity("sun");
  scene
    .add_component(
      sun_entity,
      TransformComponent {
        position: Vec3f32::from_components(10.0, 10.0, 10.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_entity,
      SunComponent {
        resolution: (128, 128, 128),
        radius: 1.0,
      },
    )
    .unwrap();

  let app_state = AppState {
    is_resizing: false,
    is_exiting: false,
    scene: Arc::new(scene),
    window: Some(window),
  };

  let mesh_app = MeshApp {
    app_state,
    render_frontend,
    render_device_handle,
    presentation_engine,
    camera_entity,
    mesh_entity,
    sun_entity,
    right_mouse_button_down: false,
    cam_dist,
    cam_yaw,
    cam_pitch,
    width,
    height,
    window_info: _window_info,
  };

  test_utils::app::run_app(mesh_app, event_loop);
}

struct MeshApp {
  app_state: AppState,
  render_frontend: gpu::RenderFrontend,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: EntityId,
  mesh_entity: EntityId,
  sun_entity: EntityId,
  right_mouse_button_down: bool,
  cam_dist: f32,
  cam_yaw: f32,
  cam_pitch: f32,
  width: u32,
  height: u32,
  window_info: test_utils::WindowPlatformData,
}

impl test_utils::app::App for MeshApp {
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
    self.width = width;
    self.height = height;
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
    self
      .app_state
      .scene
      .with_component_mut(self.camera_entity, |c: &mut CameraComponent| {
        c.projection = Mat4x4f32::perspective_vk(
          std::f32::consts::FRAC_PI_4,
          width as f32 / height as f32,
          0.1,
          100.0,
        );
      });
  }

  fn on_mouse_input(
    &mut self,
    button: winit::event::MouseButton,
    state: winit::event::ElementState,
  ) {
    if button == winit::event::MouseButton::Right {
      self.right_mouse_button_down = state == winit::event::ElementState::Pressed;
    }
  }

  fn on_keyboard_input(
    &mut self,
    event: &winit::event::KeyEvent,
    modifiers: winit::keyboard::ModifiersState,
  ) {
    if event.state == winit::event::ElementState::Pressed {
      if let winit::keyboard::PhysicalKey::Code(keycode) = event.physical_key {
        if keycode == winit::keyboard::KeyCode::Escape {
          // Do not exit on escape
        }
        #[cfg(target_os = "macos")]
        if keycode == winit::keyboard::KeyCode::KeyQ && modifiers.super_key() {
          self.app_state.is_exiting = true;
        }
      }
    }
  }

  fn on_mouse_wheel(&mut self, delta: winit::event::MouseScrollDelta) {
    let scroll_amount = match delta {
      winit::event::MouseScrollDelta::LineDelta(_, y) => y,
      winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32,
    };
    self.cam_dist = (self.cam_dist - scroll_amount).max(0.1);

    let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), self.cam_yaw);
    let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), self.cam_pitch);
    let new_rot = yaw_quat * pitch_quat;
    let offset = Vec3f32::from_components(0.0, -1.0, 0.0) * self.cam_dist;
    let new_offset = new_rot.rotate_vector(offset);
    self
      .app_state
      .scene
      .with_component_mut(self.camera_entity, |c: &mut TransformComponent| {
        c.position = new_offset;
        c.rotation = new_rot;
      });
    if let Some(w) = self.app_state.window.as_ref() {
      w.request_redraw();
    }
  }

  fn on_mouse_motion(&mut self, delta: (f64, f64)) {
    if self.right_mouse_button_down {
      let rotation_speed = 0.005;
      self.cam_yaw += delta.0 as f32 * rotation_speed;
      self.cam_pitch -= delta.1 as f32 * rotation_speed;
      self.cam_yaw = self.cam_yaw % (std::f32::consts::PI * 2.0);
      self.cam_pitch = self.cam_pitch.clamp(-1.55, 1.55);

      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), self.cam_yaw);
      let pitch_quat =
        Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), self.cam_pitch);
      let new_rot = yaw_quat * pitch_quat;
      let offset = Vec3f32::from_components(0.0, -1.0, 0.0) * self.cam_dist;
      let new_offset = new_rot.rotate_vector(offset);
      self
        .app_state
        .scene
        .with_component_mut(self.camera_entity, |c: &mut TransformComponent| {
          c.position = new_offset;
          c.rotation = new_rot;
        });
      if let Some(w) = self.app_state.window.as_ref() {
        w.request_redraw();
      }
    }
  }

  fn on_redraw(&mut self) {
    if self.width == 0 || self.height == 0 {
      return;
    }

    let res = self
      .render_frontend
      .with_device(self.render_device_handle, |device| {
        render_function(
          device,
          &self.app_state.scene,
          self.camera_entity,
          self.presentation_engine,
          [self.width, self.height],
        )
      });
    if let Err(err) = res {
      aethervk_oshal_rlib::log!("Rendering Error: {}", err);
    }
  }
}

fn render_function(
  device: &dyn RenderDevice,
  scene: &Scene,
  camera_entity: EntityId,
  presentation_engine_handle: PresentationEngineHandle,
  screen_extent: [u32; 2],
) -> GpuResult<()> {
  device.start_frame()?;
  let acquire_result = device.acquire_next_image(presentation_engine_handle)?;
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      presentation_engine_handle,
      screen_extent[0],
      screen_extent[1],
    )?;
    return Ok(());
  }
  let present_guard = FrameCancelGuard::new(device, presentation_engine_handle, acquire_result);

  let cmd_buffer = device.get_command_buffer()?;
  let cmd_guard = ScopedCommandBuffer::new(device, cmd_buffer, None)?;
  let render_scene = scene_to_render_scene(
    &scene,
    device,
    presentation_engine_handle,
    camera_entity,
    false,
    cmd_buffer,
  )?;

  if let Some(sun_call) = &render_scene.sun_call {
    device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128), 1.0)?;
  }
  device.begin_render_pass(cmd_buffer, presentation_engine_handle, &acquire_result)?;
  let render_pass_guard = ScopedRenderPass::new(device, cmd_buffer);

  let extent = device.get_presentation_engine_extent(presentation_engine_handle)?;
  device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  gpu::frame::render_frame(
    device,
    cmd_buffer,
    &render_scene,
    presentation_engine_handle,
  )?;

  // defuse of drop guards (Order *must* be like this)
  render_pass_guard.end()?;
  cmd_guard.submit()?;
  present_guard.defuse();

  let present_status = device.present(
    presentation_engine_handle,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;
  if present_status.needs_resize() {
    device.resize_presentation_engine(
      presentation_engine_handle,
      screen_extent[0],
      screen_extent[1],
    )?;
  }

  Ok(())
}
