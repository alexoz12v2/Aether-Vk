pub mod constants;
mod logic_thread;
mod render_thread;
pub mod utils;
mod windowing;

use aethervk_core_rlib::{
  gpu::{self},
  scene::{
    CameraComponent, CursorComponent, GridComponent, PhysicalMeshComponent, Scene, SkyComponent,
    SunComponent, TransformComponent,
  },
  simulation,
  types::RuntimeParams,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};
use heapless::index_map::FnvIndexMap;
#[cfg(target_os = "macos")]
use winit::platform::macos::EventLoopBuilderExtMacOS;
use std::{
  any::TypeId,
  io::Read,
  sync::{Arc, RwLock, mpsc, atomic::AtomicBool},
  time::Instant,
};
use winit::{
  event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
  event_loop::{ControlFlow, EventLoopBuilder},
  keyboard::{KeyCode, PhysicalKey},
  window::WindowBuilder,
};
use aethervk_oshal_rlib::math::vector::Vector;
use logic_thread::{start_logic_thread, LogicCommand};
use render_thread::{RenderItem, RenderPacket, start_render_thread};
use windowing::{AppEvent, WindowExtractHandlesParams, extract_native_handles, setup_resize_hook};

struct AppState {
  scene: Arc<Scene>,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: aethervk_core_rlib::scene::EntityId,
  is_paused: bool,
  time_scale: f32,
  root_entity: aethervk_core_rlib::scene::EntityId,
  window: winit::window::Window,
  is_resizing: bool,
  is_exiting: bool,
  outlines_enabled: Arc<AtomicBool>,
  is_command_prompt_open: bool,
  console_open_progress: f32,
  console_scroll_offset: usize,
  command_history: std::collections::VecDeque<String>,
  current_command: String,
}

impl Drop for AppState {
  fn drop(&mut self) {
    println!("Dropping AppState");
  }
}

struct RenderFrontendDropTracker(Arc<RwLock<aethervk_core_rlib::gpu::RenderFrontend<'static>>>);
impl Drop for RenderFrontendDropTracker {
  fn drop(&mut self) {
    println!(
      "Dropping RenderFrontend wrapper in main (strong count: {})",
      Arc::strong_count(&self.0)
    );
  }
}

impl simulation::Pausable for AppState {
  fn is_paused(&self) -> bool {
    self.is_paused
  }
  fn set_paused(&mut self) {
    self.is_paused = true;
  }
  fn time_scale(&self) -> f32 {
    self.time_scale
  }
  fn set_time_scale(&mut self, scale: f32) {
    self.time_scale = scale;
  }
}

fn main() {
  std::panic::set_hook(Box::new(|panic_info| {
    println!("CRASH DETECTED: {}", panic_info);
    println!("Press Enter to close the application...");
    let _ = std::io::stdin().read(&mut [0u8]);
  }));

  let mut event_loop_builder = EventLoopBuilder::<AppEvent>::with_user_event();
  // Disable default macOS menu. This disables default macOS bindings (so that we can customize interception of Super + Q)
  #[cfg(target_os = "macos")]
  event_loop_builder.with_default_menu(false);

  let event_loop = event_loop_builder.build().unwrap();
  let proxy = event_loop.create_proxy();
  let proxy_ptr = unsafe { std::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(proxy))) };

  let start_time = Instant::now();
  println!("[{:.2?}] Application starting.", start_time.elapsed());

  let window = WindowBuilder::new()
    .with_title("AetherVk Simulation")
    .build(&event_loop)
    .unwrap();
  setup_resize_hook(&window, proxy_ptr);

  let runtime_params = Box::leak(Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
  }));
  let render_frontend = Arc::new(RwLock::new(
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, runtime_params).unwrap(),
  ));

  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = render_frontend
    .write()
    .unwrap()
    .take_mut_and(|context| Ok(context.init_device(0, &additional_params)?))
    .unwrap()
    .unwrap();

  let params: WindowExtractHandlesParams;
  #[cfg(not(target_os = "macos"))]
  {
    params = WindowExtractHandlesParams {};
  }
  #[cfg(target_os = "macos")]
  {
    let mtl_device_id = render_frontend
      .read()
      .unwrap()
      .take_and(|context| {
        let mut mtl_device_id = core::ptr::null::<core::ffi::c_void>();
        context.deref_device_and(
          render_device_handle,
          core::ptr::from_mut(&mut mtl_device_id) as *mut _,
          |device, ptr_dev_id| {
            let ptr = ptr_dev_id as *mut *const core::ffi::c_void;
            unsafe {
              *ptr = device
                .get_native_prop(gpu::NativeGpuProperty::VulkanMetalDeviceId)
                .unwrap();
            };
            Ok(())
          },
        );
        let dev_ptr =
          mtl_device_id as *mut objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>;
        let metal_device = unsafe { objc2::rc::Retained::retain(dev_ptr).unwrap() };
        Ok(metal_device)
      })
      .unwrap()
      .unwrap();
    params = WindowExtractHandlesParams::new_macos(mtl_device_id);
  }
  let (native_handles, window_info) = extract_native_handles(&window, &params);

  let presentation_engine = {
    let params = gpu::PresentationEngineParams {
      width: window.inner_size().width,
      height: window.inner_size().height,
      vsync: true,
      ty: gpu::PresentationEngineType::Window,
      window_info: native_handles,
    };
    render_frontend
      .write()
      .unwrap()
      .take_and(|context| {
        use aethervk_core_rlib::types::GpuError;

        let mut handle_result: aethervk_core_rlib::types::GpuResult<gpu::PresentationEngineHandle> =
          Err(GpuError::InvalidState);
        let mut closure_data = (&params, &mut handle_result);

        let closure = |device: &dyn gpu::RenderDevice, data: *mut core::ffi::c_void| {
          type ClosureData<'a> = (
            &'a gpu::PresentationEngineParams,
            &'a mut aethervk_core_rlib::types::GpuResult<gpu::PresentationEngineHandle>,
          );

          let data_ptr = data as *mut ClosureData;
          let (params_ref, handle_result) = unsafe { &mut *data_ptr };
          let pe_result = device.create_presentation_engine(*params_ref);
          if let Ok(pe) = pe_result {
            device
              .init_archetypes(pe)
              .expect("Failed to initialize archetypes");
          }
          **handle_result = pe_result;

          device
            .generate_sky()
            .expect("Failed to generate background sky map!");

          Ok(())
        };

        context
          .deref_device_and(
            render_device_handle,
            &mut closure_data as *mut _ as *mut core::ffi::c_void,
            closure,
          )
          .unwrap()?;
        Ok(handle_result?)
      })
      .unwrap()
      .unwrap()
  };
  println!(
    "[{:.2?}] GPU initialization complete.",
    start_time.elapsed()
  );

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SkyComponent>(&[]);
  scene.register_component::<GridComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::BvhDebugComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::SelectedComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::FollowingComponent>(&[]);

  let assets_dir = {
    let mut args = std::env::args();
    if args.len() > 1 {
      let _ = args.next().unwrap();
      std::path::PathBuf::from(args.next().unwrap())
    } else {
      let mut home_dir = std::env::current_exe().unwrap();
      let mut iter: i32 = 0;
      const MAX_ITER: i32 = 32;
      while {
        let d = home_dir.join("assets");
        !d.is_dir() && iter < MAX_ITER
      } {
        home_dir.pop();
        iter += 1;
        assert!(home_dir.is_dir());
      }
      home_dir.join("assets")
    }
  };

  let model_path = {
    let mut args = std::env::args();
    if args.len() > 1 {
      let _ = args.next().unwrap();
      std::path::PathBuf::from(args.next().unwrap()).join("Comet.glb")
    } else {
      assets_dir.join("Comet.glb")
    }
  };
  let comet = simulation::comet::load_comet_from_gltf(model_path.to_str().unwrap(), false)
    .expect("Failed to load comet");

  let root_entity = scene.spawn_entity("entity");
  scene
    .add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();

  #[cfg(not(feature = "spotless_rendering"))]
  {
    let mesh_entity = scene.spawn_entity("entity");
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
    scene
      .add_component(
        mesh_entity,
        PhysicalMeshComponent {
          mesh: comet,
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
        },
      )
      .unwrap();
    scene.set_parent(mesh_entity, Some(root_entity));
  }

  #[cfg(not(feature = "spotless_rendering"))]
  let planets = [
    (
      "Mercury",
      "planets/textures/Mercury.jpg",
      crate::constants::PlanetNaifId::MERCURY,
      1407.6,
    ),
    (
      "Venus",
      "planets/textures/Venus.jpg",
      crate::constants::PlanetNaifId::VENUS,
      -5832.6,
    ),
    (
      "Earth",
      "planets/textures/Earth.jpg",
      crate::constants::PlanetNaifId::EARTH,
      23.93,
    ),
    (
      "Mars",
      "planets/textures/Mars.jpg",
      crate::constants::PlanetNaifId::MARS,
      24.62,
    ),
    (
      "Jupiter",
      "planets/textures/Jupiter.jpg",
      crate::constants::PlanetNaifId::JUPITER,
      9.92,
    ),
    (
      "Saturn",
      "planets/textures/Saturn.jpg",
      crate::constants::PlanetNaifId::SATURN,
      10.65,
    ),
    (
      "Uranus",
      "planets/textures/Uranus.jpg",
      crate::constants::PlanetNaifId::URANUS,
      -17.24,
    ),
    (
      "Neptune",
      "planets/textures/Neptune.jpg",
      crate::constants::PlanetNaifId::NEPTUNE,
      16.11,
    ),
  ];

  let mut planets_ids = Vec::new();

  #[cfg(not(feature = "spotless_rendering"))]
  for (name, tex_path, naif_id, rot_period) in planets.iter() {
    let planet_radius = (utils::get_planet_radius(*naif_id, &assets_dir)
      / constants::DISTANCE_SCALE_FACTOR as f32)
      * constants::PLANET_VISUAL_SCALE;
    let initial_pos = Vec3f32::zero();

    let mut sphere = simulation::comet::generate_uv_sphere(planet_radius, 64, 64);
    let tex =
      simulation::comet::load_texture_from_file(assets_dir.join(tex_path).to_str().unwrap())
        .expect(&format!("Failed to load texture for {}", name));
    sphere.albedo_map = Some(tex);

    let planet_entity = scene.spawn_entity(*name);
    planets_ids.push((*naif_id, planet_entity, *rot_period, planet_radius));
    scene
      .add_component(
        planet_entity,
        TransformComponent {
          position: initial_pos,
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene
      .add_component(
        planet_entity,
        PhysicalMeshComponent {
          mesh: sphere,
          emissive_intensity: 0.0,
          emissive_color: [0.0, 0.0, 0.0],
        },
      )
      .unwrap();
    scene.set_parent(planet_entity, Some(root_entity));
  }

  let cursor_entity = scene.spawn_entity("entity");
  scene
    .add_component(
      cursor_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(cursor_entity, CursorComponent {})
    .unwrap();
  scene.set_parent(cursor_entity, Some(root_entity));

  let camera_entity = scene.spawn_entity("entity");
  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, -40.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(
          45.0f32.to_radians(),
          800.0 / 600.0, // Default aspect ratio, will be updated by resize
          0.1,
          1000000.0,
        ),
        near_plane: 0.1,
        far_plane: 1000000.0,
      },
    )
    .unwrap();
  scene.set_parent(camera_entity, Some(root_entity));

  let sun_entity = scene.spawn_entity("sun");
  let sun_radius = (utils::get_planet_radius(constants::PlanetNaifId::SUN, &assets_dir)
    / constants::DISTANCE_SCALE_FACTOR as f32)
    * constants::UNIVERSAL_VISUAL_SCALE;
  let sun_scale = sun_radius / 0.45;
  let sun_pos = Vec3f32::zero();

  scene
    .add_component(
      sun_entity,
      TransformComponent {
        position: sun_pos,
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(sun_scale, sun_scale, sun_scale),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_entity,
      aethervk_core_rlib::scene::SunComponent {
        resolution: (128, 128, 128),
      },
    )
    .unwrap();

  // Add emissive core for the sun
  let sun_core_entity = scene.spawn_entity("sun_core");
  let mut sun_sphere = simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64);
  sun_sphere.albedo_map = None;
  scene
    .add_component(
      sun_core_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      sun_core_entity,
      PhysicalMeshComponent {
        mesh: sun_sphere,
        emissive_intensity: 0.9, // Reduced to prevent SDR whiteout clamp
        emissive_color: [1.0, 0.35, 0.02], // Pure rich orange/red
      },
    )
    .unwrap();
  scene.set_parent(sun_core_entity, Some(sun_entity));

  let sky_entity = scene.spawn_entity("entity");
  scene
    .add_component(sky_entity, aethervk_core_rlib::scene::SkyComponent {})
    .unwrap();

  let grid_entity = scene.spawn_entity("grid");
  scene
    .add_component(grid_entity, aethervk_core_rlib::scene::GridComponent {})
    .unwrap();
  scene.set_parent(sun_entity, Some(root_entity));

  let scene_shared = Arc::new(scene);
  let outlines_enabled = Arc::new(AtomicBool::new(true));

  let mut app_state = AppState {
    scene: Arc::clone(&scene_shared),
    presentation_engine,
    camera_entity,
    is_paused: false,
    time_scale: 1.0,
    root_entity,
    is_resizing: false,
    is_exiting: false,
    window,
    outlines_enabled: Arc::clone(&outlines_enabled),
    is_command_prompt_open: false,
    console_open_progress: 0.0,
    console_scroll_offset: 0,
    command_history: std::collections::VecDeque::with_capacity(1000),
    current_command: String::new(),
  };

  let _render_frontend_tracker = RenderFrontendDropTracker(Arc::clone(&render_frontend));

  // --- Start Render Thread ---
  let (render_tx, render_rx) = mpsc::sync_channel::<Option<RenderPacket>>(1);
  let render_thread_handle = start_render_thread(
    render_rx,
    Arc::clone(&scene_shared),
    Arc::clone(&render_frontend),
    render_device_handle,
    presentation_engine,
    cursor_entity,
    sun_entity,
    assets_dir.clone(),
  );

  // --- Start Logic Thread ---
  let (logic_tx, logic_rx) = mpsc::channel::<LogicCommand>();
  let (response_tx, response_rx) = mpsc::channel::<String>();
  let logic_thread_handle = start_logic_thread(
    logic_rx,
    response_tx,
    Arc::clone(&scene_shared),
    root_entity,
    camera_entity,
    cursor_entity,
    grid_entity,
    planets_ids,
    sun_entity,
    sun_radius,
    assets_dir,
    Arc::clone(&outlines_enabled),
  );

  let mut initial_width = app_state.window.inner_size().width;
  let mut initial_height = app_state.window.inner_size().height;
  if initial_width == 0 {
    initial_width = 800;
  }
  if initial_height == 0 {
    initial_height = 600;
  }

  let _ = logic_tx.send(LogicCommand::Resize {
    width: initial_width,
    height: initial_height,
  });

  // --- Main Event Loop ---
  let mut right_mouse_button_down = false;
  let mut middle_mouse_button_down = false;
  let mut mouse_x = 0.0;
  let mut mouse_y = 0.0;
  let mut last_log_time = Instant::now();
  let mut modifiers_state = winit::keyboard::ModifiersState::empty();

  event_loop.set_control_flow(ControlFlow::Poll);
  event_loop
    .run(move |event, elwt| match event {
      Event::UserEvent(app_event) => match app_event {
        AppEvent::ResizeStarted => {
          app_state.is_resizing = true;
        }
        AppEvent::ResizeEnded => {
          app_state.is_resizing = false;
          app_state.window.request_redraw();
        }
      },

      Event::WindowEvent { event, window_id } if window_id == app_state.window.id() => {
        match event {
          WindowEvent::CloseRequested => {
            app_state.is_exiting = true;
            let _ = logic_tx.send(LogicCommand::Exit);
            let _ = render_tx.try_send(None);
            elwt.exit();
          }
          WindowEvent::Resized(physical_size) => {
            let _ = logic_tx.send(LogicCommand::Resize {
              width: physical_size.width,
              height: physical_size.height,
            });

            #[cfg(target_os = "macos")]
            {
              window_info
                .metal_layer
                .setDrawableSize(objc2_core_foundation::CGSize {
                  width: physical_size.width as f64,
                  height: physical_size.height as f64,
                });
            }
          }
          WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
            #[cfg(target_os = "macos")]
            {
              window_info.metal_layer.setContentsScale(scale_factor);
            }
          }
          WindowEvent::ModifiersChanged(modifiers) => {
            modifiers_state = modifiers.state();
          }
          WindowEvent::CursorMoved { position, .. } => {
            mouse_x = position.x;
            mouse_y = position.y;
          }
          WindowEvent::MouseInput {
            state: element_state,
            button,
            ..
          } => match button {
            MouseButton::Right => right_mouse_button_down = element_state == ElementState::Pressed,
            MouseButton::Middle => {
              middle_mouse_button_down = element_state == ElementState::Pressed
            }
            MouseButton::Left => {
              if element_state == ElementState::Pressed {
                let size = app_state.window.inner_size();
                if size.width > 0 && size.height > 0 {
                  let ndc_x = (mouse_x as f32 / size.width as f32) * 2.0 - 1.0;
                  let ndc_y = (mouse_y as f32 / size.height as f32) * 2.0 - 1.0;
                  let _ = logic_tx.send(LogicCommand::RaycastCursor { ndc_x, ndc_y });
                  app_state.window.request_redraw();
                }
              }
            }
            _ => {}
          },
          WindowEvent::KeyboardInput { event, .. } => {
            if event.state == ElementState::Pressed {
              if app_state.is_command_prompt_open {
                if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                  app_state.is_command_prompt_open = false;
                  app_state.window.request_redraw();
                } else {
                  match &event.logical_key {
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Backspace) => {
                      app_state.current_command.pop();
                      app_state.window.request_redraw();
                    }
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageUp) => {
                      app_state.console_scroll_offset = app_state.console_scroll_offset.saturating_add(1);
                      app_state.window.request_redraw();
                    }
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::PageDown) => {
                      app_state.console_scroll_offset = app_state.console_scroll_offset.saturating_sub(1);
                      app_state.window.request_redraw();
                    }
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Enter) => {
                      if !app_state.current_command.is_empty() {
                        let cmd = app_state.current_command.clone();
                        app_state.command_history.push_back(format!("> {}", cmd));
                        let _ = logic_tx.send(LogicCommand::ExecuteCommand(cmd));
                        if app_state.command_history.len() > 1000 {
                          app_state.command_history.pop_front();
                        }
                        app_state.current_command.clear();
                        app_state.console_scroll_offset = 0;
                      }
                      app_state.window.request_redraw();
                    }
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                      app_state.current_command.push(' ');
                      app_state.window.request_redraw();
                    }
                    winit::keyboard::Key::Character(c) => {
                      app_state.current_command.push_str(c.as_str());
                      app_state.window.request_redraw();
                    }
                    _ => {}
                  }
                }
              } else {
                if let PhysicalKey::Code(keycode) = event.physical_key {
                  let speed = 0.5;
                  match keycode {
                    KeyCode::KeyM => {
                      app_state.is_command_prompt_open = true;
                      app_state.window.request_redraw();
                    }
                    KeyCode::ArrowUp => {
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(0.0, 0.0, -1.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::ArrowDown => {
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(0.0, 0.0, 1.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::ArrowLeft => {
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(-1.0, 0.0, 0.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::ArrowRight => {
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(1.0, 0.0, 0.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyQ => {
                      #[cfg(target_os = "macos")]
                      if modifiers_state.super_key() {
                        // So it doesn't even get here
                        app_state.is_exiting = true;
                        let _ = logic_tx.send(LogicCommand::Exit);
                        let _ = render_tx.try_send(None);
                        println!("You Clicked exit");
                        elwt.exit();
                        return;
                      }
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(0.0, -1.0, 0.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyE => {
                      let _ = logic_tx.send(LogicCommand::MoveCursor {
                        axis: Vec3f32::from_components(0.0, 1.0, 0.0),
                        amount: speed,
                      });
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyX => {
                      let _ = logic_tx.send(LogicCommand::CycleTimeScale);
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyA => {
                      let _ = logic_tx.send(LogicCommand::CyclePlanet { forward: false });
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyD => {
                      let _ = logic_tx.send(LogicCommand::CyclePlanet { forward: true });
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyG => {
                      let _ = logic_tx.send(LogicCommand::ToggleGrid);
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyH => {
                      let _ = logic_tx.send(LogicCommand::TogglePlanetOutlines);
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyT => {
                      let _ = logic_tx.send(LogicCommand::ResetCamera);
                      app_state.window.request_redraw();
                    }
                    KeyCode::Digit0 | KeyCode::Numpad0 => {
                      let _ = logic_tx.send(LogicCommand::ResetCursor);
                      app_state.window.request_redraw();
                    }
                    KeyCode::Digit1 | KeyCode::Numpad1 => {
                      let _ = logic_tx.send(LogicCommand::SnapCursorToSun);
                      app_state.window.request_redraw();
                    }
                    KeyCode::Digit2 | KeyCode::Numpad2 => {
                      let _ = logic_tx.send(LogicCommand::SnapCameraToCursor);
                      app_state.window.request_redraw();
                    }
                    KeyCode::KeyV => {
                      let _ = logic_tx.send(LogicCommand::ToggleMeasureTool);
                      app_state.window.request_redraw();
                    }
                    _ => {}
                  }
                }
              }
            }
          }
          WindowEvent::MouseWheel { delta, .. } => {
            let scroll_amount = match delta {
              winit::event::MouseScrollDelta::LineDelta(_, y) => y,
              winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32, // tone down pixel delta
            };
            if app_state.is_command_prompt_open {
               if scroll_amount > 0.0 {
                 app_state.console_scroll_offset = app_state.console_scroll_offset.saturating_add(1);
               } else if scroll_amount < 0.0 {
                 app_state.console_scroll_offset = app_state.console_scroll_offset.saturating_sub(1);
               }
               app_state.window.request_redraw();
            } else {
              let _ = logic_tx.send(LogicCommand::ZoomCamera {
                amount: scroll_amount,
              });
              app_state.window.request_redraw();
            }
          }
          WindowEvent::RedrawRequested => {
            if app_state.is_resizing || app_state.is_exiting {
              return;
            }

            let mut render_items = Vec::new();
            let mut matrix_stack = vec![Mat4x4f32::identity()];
            let scene_guard = app_state.scene.as_ref();

            scene_guard.traverse_with_hooks(
              app_state.root_entity,
              &mut matrix_stack,
              &mut |stack: &mut Vec<Mat4x4f32>,
                    entity,
                    transform_opt: Option<TransformComponent>,
                    mesh_opt: Option<&PhysicalMeshComponent>| {
                let local_transform = transform_opt
                  .map(|c| {
                    Mat4x4f32::translation(c.position)
                      * Mat4x4f32::from_quat_custom_frame(c.rotation)
                      * Mat4x4f32::from_scale(c.scale)
                  })
                  .unwrap_or(Mat4x4f32::identity());

                let parent_transform = stack.last().unwrap();
                let global_transform = *parent_transform * local_transform;

                if mesh_opt.is_some() {
                  render_items.push(RenderItem {
                    entity_id: entity,
                    model_matrix: global_transform,
                  });
                }

                stack.push(global_transform);
                true
              },
              &mut |stack, _| {
                stack.pop();
              },
            );

            let mut camera_transform = TransformComponent {
              position: Vec3f32::from_components(0.0, 0.0, 0.0),
              rotation: Quat::identity(),
              scale: Vec3f32::from_components(1.0, 1.0, 1.0),
            };
            let mut camera_component = CameraComponent {
              projection: Mat4x4f32::identity(),
              near_plane: 0.1,
              far_plane: 10000000.0,
            };

            if let Some(global) = scene_guard.global_transform(app_state.camera_entity) {
              camera_transform = global;
            }
            scene_guard.with_component(app_state.camera_entity, |c| camera_component = *c);

            // Free read lock before potentially blocking on full channel to avoid deadlocking with Logic Thread

            let packet = RenderPacket {
              render_items,
              camera_transform,
              camera_component,
              window_size: app_state.window.inner_size(),
              outlines_enabled: app_state
                .outlines_enabled
                .load(std::sync::atomic::Ordering::Relaxed),
              is_command_prompt_open: app_state.is_command_prompt_open,
              console_open_progress: app_state.console_open_progress,
              console_scroll_offset: app_state.console_scroll_offset,
              command_history: app_state.command_history.clone(),

              current_command: app_state.current_command.clone(),
            };

            match render_tx.try_send(Some(packet)) {
              Ok(_) => {}
              Err(std::sync::mpsc::TrySendError::Full(_)) => {}
              Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                elwt.exit();
              }
            }
          }
          _ => {}
        }
      }
      Event::DeviceEvent {
        event: DeviceEvent::MouseMotion { delta },
        ..
      } => {
        if right_mouse_button_down {
          let _ = logic_tx.send(LogicCommand::RotateCamera {
            delta_x: delta.0 as f32,
            delta_y: delta.1 as f32,
          });
          app_state.window.request_redraw();
        } else if middle_mouse_button_down {
          let _ = logic_tx.send(LogicCommand::PanCursor {
            delta_x: delta.0 as f32,
            delta_y: delta.1 as f32,
          });
          app_state.window.request_redraw();
        }
      }
      Event::LoopExiting => {
        app_state.is_exiting = true;
        let _ = logic_tx.send(LogicCommand::Exit);
        let _ = render_tx.try_send(None);
      }
      Event::AboutToWait => {
        if app_state.is_exiting {
          return;
        }

        let dt = 0.016; // approx 60fps
        if app_state.is_command_prompt_open {
          app_state.console_open_progress += dt * 5.0;
          if app_state.console_open_progress > 1.0 {
            app_state.console_open_progress = 1.0;
          } else {
            app_state.window.request_redraw();
          }
        } else {
          app_state.console_open_progress -= dt * 5.0;
          if app_state.console_open_progress < 0.0 {
            app_state.console_open_progress = 0.0;
          } else {
            app_state.window.request_redraw();
          }
        }

        let mut got_responses = false;
        while let Ok(response) = response_rx.try_recv() {
          if response == "___CLEAR___" {
            app_state.command_history.clear();
          } else {
            app_state.command_history.push_back(response);
            if app_state.command_history.len() > 1000 {
              app_state.command_history.pop_front();
            }
          }
          got_responses = true;
        }
        if got_responses && app_state.is_command_prompt_open {
          app_state.window.request_redraw();
        }

        if last_log_time.elapsed().as_secs() >= 5 {
          last_log_time = Instant::now();
        }
        if !app_state.is_resizing {
          app_state.window.request_redraw();
        }
      }
      _ => (),
    })
    .unwrap();

  println!("Event loop returned. Joining threads...");
  let _ = render_thread_handle.join();
  let _ = logic_thread_handle.join();
  println!("Threads joined. Exiting main().");
}
