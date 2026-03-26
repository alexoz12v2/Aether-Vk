use aethervk_core_rlib::{
  gpu::{self, frame, RenderDevice, RenderDeviceHandle, RenderFrontend, SwapchainStatus},
  scene::{self, PhysicalMeshComponent, Scene, TransformComponent},
  simulation,
  types::GpuResult,
};
use aethervk_oshal::os::time::TimeReadings;
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::vec3::{Vec3f32, self},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{env, ptr};
use std::path::PathBuf;
use winit::{
  event::{Event, WindowEvent, DeviceEvent, ElementState, RawKeyEvent},
  event_loop::{ControlFlow, EventLoop},
  window::{WindowBuilder, Window},
  keyboard::{Key, PhysicalKey, KeyCode},
};

struct AppState {
  scene: Scene,
  presentation_engine: gpu::PresentationEngineHandle,
  paused: bool,
  time_scale: f32,
  should_run: bool,
  input_state: InputState,
}

struct InputState {
  forward: bool,
  backward: bool,
  left: bool,
  right: bool,
  up: bool,
  down: bool,
}

impl Default for InputState {
  fn default() -> Self {
    Self {
      forward: false,
      backward: false,
      left: false,
      right: false,
      up: false,
      down: false,
    }
  }
}

impl simulation::Pausable for AppState {
  fn is_paused(&self) -> bool {
    self.paused
  }

  fn set_paused(&mut self) {
    self.paused = true;
  }

  fn time_scale(&self) -> f32 {
    self.time_scale
  }

  fn set_time_scale(&mut self, scale: f32) {
    self.time_scale = scale;
  }
}

fn handle_input(state: &mut AppState, event: &WindowEvent) {
  if let WindowEvent::KeyboardInput { event, .. } = event {
    if let PhysicalKey::Code(code) = event.physical_key {
      let is_pressed = event.state == ElementState::Pressed;
      match code {
        KeyCode::KeyW => state.input_state.forward = is_pressed,
        KeyCode::KeyS => state.input_state.backward = is_pressed,
        KeyCode::KeyA => state.input_state.left = is_pressed,
        KeyCode::KeyD => state.input_state.right = is_pressed,
        KeyCode::Space => state.input_state.up = is_pressed,
        KeyCode::ShiftLeft => state.input_state.down = is_pressed,
        _ => {}
      }
    }
  }
}

#[test]
fn test_simulation() {
  let event_loop = EventLoop::new().unwrap();
  let window = WindowBuilder::new()
    .with_title("AetherVk Simulation")
    .build(&event_loop)
    .unwrap();

  let mut render_frontend = gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND).unwrap();

  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .take_mut_and(|context| context.init_device(0, &additional_params))
    .unwrap()
    .unwrap();

  let presentation_engine = {
    let display_handle = window.display_handle().unwrap().as_raw();
    let window_handle = window.window_handle().unwrap().as_raw();
    let params = gpu::PresentationEngineParams {
      width: window.inner_size().width,
      height: window.inner_size().height,
      vsync: true,
      window_info: gpu::OpaqueNativeHandleInfo {
        ptr0: display_handle as *mut _,
        ptr1: window_handle as *mut _,
      },
    };

    render_frontend
      .take_and(|context| {
        context
          .deref_device_and(
            render_device_handle,
            &params as *const _ as *mut core::ffi::c_void,
            |dev, params| {
              let params = unsafe { &*(params as *const gpu::PresentationEngineParams) };
              dev.create_presentation_engine(params)
            },
          )
          .unwrap()
      })
      .unwrap()
      .unwrap()
  };

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[]);

  let home_dir = {
    let mut home_dir = std::env::current_exe().unwrap();
    for _ in 0..6 {
      home_dir.pop();
      assert!(home_dir);
    }
    home_dir
  };
  let comet_path = home_dir.join();
  let model_path = PathBuf::from(home_dir)
    .join("model.gltf")
    .to_str()
    .unwrap()
    .to_string();

  let comet = simulation::comet::load_comet_from_gltf(&model_path).expect("Failed to load comet");

  let entity = scene.spawn_entity();
  scene
    .add_component(
      entity,
      TransformComponent {
        position: Vec3f32::new(0.0, 0.0, -5.0),
        rotation: Quaternion::identity().into(),
        scale: Vec3f32::new(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(entity, PhysicalMeshComponent { mesh: comet })
    .unwrap();

  let mut app_state = AppState {
    scene,
    presentation_engine,
    paused: false,
    time_scale: 1.0,
    should_run: true,
    input_state: InputState::default(),
  };

  event_loop.set_control_flow(ControlFlow::Poll);

  let mut events_buffer = Vec::new();

  event_loop
    .run(move |event, elwt| match event {
      Event::WindowEvent { event, .. } => {
        if event == WindowEvent::CloseRequested {
          app_state.should_run = false;
          elwt.exit();
        } else {
          events_buffer.push(event);
        }
      }
      Event::AboutToWait => {
        simulation::run(
          app_state,
          &render_frontend,
          render_device_handle,
          |s| s.should_run,
          |s| {
            for event in events_buffer.drain(..) {
              handle_input(s, &event);
            }
          },
          |s, time, render_frontend, render_device_handle| {
            render(s, time, render_frontend, render_device_handle, &window);
          },
          |s, time, _render_frontend, _render_device_handle| {
            update_transform(s, time);
          },
          16_667,
          100_000,
        );
        app_state = app_state;
      }
      _ => (),
    })
    .unwrap();
}

fn update_transform(state: &mut AppState, time: &TimeReadings) {
  let rotation_speed = 0.5;
  let translation_speed = 1.0;

  state
    .scene
    .query1_mut(|_, transform: &mut TransformComponent| {
      let mut rotation_delta = Quaternion::identity();
      if state.input_state.left {
        rotation_delta = rotation_delta
          * Quaternion::from_axis_angle(
            Vec3f32::new(0.0, 1.0, 0.0),
            -rotation_speed * time.dt as f32,
          );
      }
      if state.input_state.right {
        rotation_delta = rotation_delta
          * Quaternion::from_axis_angle(
            Vec3f32::new(0.0, 1.0, 0.0),
            rotation_speed * time.dt as f32,
          );
      }

      let current_rotation = Quaternion::from(transform.rotation);
      transform.rotation = (rotation_delta * current_rotation).into();

      if state.input_state.forward {
        transform.position.z += translation_speed * time.dt as f32;
      }
      if state.input_state.backward {
        transform.position.z -= translation_speed * time.dt as f32;
      }
      if state.input_state.up {
        transform.position.y += translation_speed * time.dt as f32;
      }
      if state.input_state.down {
        transform.position.y -= translation_speed * time.dt as f32;
      }
    });
}

fn render(
  state: &mut AppState,
  time: &TimeReadings,
  render_frontend: &RenderFrontend,
  render_device_handle: RenderDeviceHandle,
  window: &Window,
) {
  let res: GpuResult<()> = render_frontend
    .take_and(|context| {
      context
        .deref_device_and(render_device_handle, ptr::from_mut(state).cast(), do_render)
        .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
    })
    .unwrap()
    .unwrap();

  if let Err(e) = res {
    println!("Render error: {:?}", e);
  }
}

fn do_render(device: &dyn gpu::RenderDevice, user_data: *mut ffi::c_void) -> GpuResult<()> {
  let state = unsafe { user_data.cast::<AppState>().as_mut().unwrap_unchecked() };
  dev.start_frame()?;
  let acquire_result = dev.acquire_next_image(state.presentation_engine)?;
  if acquire_result.status == SwapchainStatus::NeedsRecreation {
    let size = window.inner_size();
    dev.resize_presentation_engine(state.presentation_engine, size.width, size.height)?;
    return Ok(());
  }

  let mut frame = frame::Frame::new();
  state.scene.query2(
    |entity_id, transform: &TransformComponent, mesh: &PhysicalMeshComponent| {
      frame
        .add_renderable(
          dev,
          entity_id,
          transform,
          scene::RenderableDataRef::PhysicalMesh(mesh),
          state.presentation_engine,
        )
        .unwrap();
    },
  );

  let render_path = frame::ForwardRenderPath;
  // We need a camera
  // For now, let's just create a dummy one
  let camera_transform = TransformComponent {
    position: Vec3f32::new(0.0, 0.0, 0.0),
    rotation: Quaternion::identity().into(),
    scale: Vec3f32::new(1.0, 1.0, 1.0),
  };
  let camera_component = scene::CameraComponent {
    projection: aethervk_oshal_rlib::math::matrix::mat4::perspective_vk(
      45.0f32.to_radians(),
      window.inner_size().width as f32 / window.inner_size().height as f32,
      0.1,
      100.0,
    ),
  };

  render_path.record_commands(
    dev,
    (&camera_transform, &camera_component),
    &frame,
    // This is a placeholder for the render pass handle
    // We will need to create a render pass in the device
    gpu::GpuResourceHandle(acquire_result.image_index as u64),
  )?;

  dev.present(
    state.presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;

  Ok(())
}
