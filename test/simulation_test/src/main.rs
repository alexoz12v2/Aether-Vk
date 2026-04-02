use aethervk_core_rlib::{
  gpu::{
    self, OpaqueNativeHandleInfo, RenderDevice,
    frame::{self, RenderPath},
  },
  scene::{self, CameraComponent, PhysicalMeshComponent, Scene, TransformComponent},
  simulation,
  types::{GpuResult, RuntimeParams},
};
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix4, SquareMatrix},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Vec4f32, Vector3},
};
use heapless::index_map::FnvIndexMap;
#[cfg(target_os = "macos")]
use raw_window_handle::RawWindowHandle;
#[cfg(all(target_os = "linux", feature = "linux_xcb"))]
use raw_window_handle::RawWindowHandle;
#[cfg(target_os = "linux")]
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
#[cfg(target_os = "linux")]
use core::ffi;
#[cfg(windows)]
use raw_window_handle::RawWindowHandle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
#[cfg(all(target_os = "linux", feature = "linux_wayland"))]
use spirv::Op;
#[cfg(windows)]
use core::ffi;
use std::{
  io::Read,
  collections::HashMap,
  sync::{
    Arc, RwLock, mpsc,
    atomic::{self, AtomicU64},
  },
  time::Instant,
};
use winit::{
  event::{DeviceEvent, ElementState, Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::{Window, WindowBuilder},
};
use aethervk_core_rlib::scene::EntityId;

// ----------------------------------------------
// Structures to cross FFI boundaries safely
// ----------------------------------------------
struct RenderItem {
  entity_id: EntityId,
  model_matrix: Mat4x4f32,
}

struct RenderPacket {
  render_items: Vec<RenderItem>,
  camera_transform: TransformComponent,
  camera_component: CameraComponent,
  window_size: winit::dpi::PhysicalSize<u32>,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  packet: &'a mut RenderPacket,
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
}

/// Utility to extract native handles from [`winit::Window`]
fn extract_native_handles(window: &Window) -> OpaqueNativeHandleInfo {
  // extract raw handles from winit window
  let window_handle = window.window_handle().unwrap().as_raw();
  let display_handle = window.display_handle().unwrap().as_raw();

  match (window_handle, display_handle) {
    #[cfg(windows)]
    (RawWindowHandle::Win32(w), _) => OpaqueNativeHandleInfo {
      ptr0: w.hinstance.map(|h| h.get()).unwrap_or(0) as *mut ffi::c_void,
      ptr1: w.hwnd.get() as *mut ffi::c_void,
    },

    #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
    (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => OpaqueNativeHandleInfo {
      ptr0: d.display.as_ptr() as *mut ffi::c_void,
      ptr1: w.surface.as_ptr() as *mut ffi::c_void,
    },

    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => OpaqueNativeHandleInfo {
      ptr0: d
        .display
        .map(|d| d.as_ptr())
        .unwrap_or(std::ptr::null_mut()) as *mut ffi::c_void,
      ptr1: w.window as usize as *mut ffi::c_void,
    },

    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => OpaqueNativeHandleInfo {
      ptr0: d
        .connection
        .map(|c| c.as_ptr())
        .unwrap_or(std::ptr::null_mut()) as *mut ffi::c_void,
      ptr1: w.window.get() as usize as *mut ffi::c_void,
    },

    #[cfg(target_os = "macos")]
    (RawWindowHandle::AppKit(w), _) => {
      use core::ffi;

      use objc::{class, msg_send, sel, sel_impl};
      // raw-window-handle gives us a NSView
      let ns_view = w.ns_view.as_ptr() as *mut objc::runtime::Object;
      // we must use Objcetive-C to create a CAMetalLayer and attach it
      let layer: *mut ffi::c_void = unsafe {
        // 1. Check if winit already gave the view a CAMetalLayer
        let metal_layer_class = class!(CAMetalLayer);
        let current_layer: *mut objc::runtime::Object = msg_send![ns_view, layer];
        let is_metal_layer = if current_layer.is_null() {
          false
        } else {
          msg_send![current_layer, isKindOfClass: metal_layer_class]
        };

        if is_metal_layer {
          current_layer as *mut ffi::c_void
        } else {
          // 2. `id layer = [CAMetalLayer layer];` with this, layer doesn't get a +1 to its retail count
          // that means we release the object as soon as Object's drop is called. Hence use new to prevent
          // premature destruction of the Objective-C object
          let new_layer: *mut objc::runtime::Object = msg_send![metal_layer_class, new];

          // 3. Set the layer BEFORE wantsLayer to YES t ocreate a layer-hosting view
          // [view setLayer: layer];
          let () = msg_send![ns_view, setLayer: new_layer];
          // [view setWantsLayer: YES];
          let () = msg_send![ns_view, setWantsLayer: true];

          // 4. Now the view retains the layer (+2 on retain count). We can release our manual `new`
          // so that we don't leak memory
          let () = msg_send![new_layer, release];

          new_layer as *mut ffi::c_void
        }
      };

      OpaqueNativeHandleInfo {
        ptr0: layer,
        ptr1: std::ptr::null_mut(),
      }
    }

    _ => panic!("unsupported platform or handle mismatch"),
  }
}

struct AppState {
  scene: Arc<RwLock<Scene>>,
  presentation_engine: gpu::PresentationEngineHandle,
  camera_entity: scene::EntityId,
  is_paused: bool,
  time_scale: f32,
  root_entity: scene::EntityId,
  window: Window,
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
  // 1. Override the default panic behavior
  std::panic::set_hook(Box::new(|panic_info| {
    // Print out the panic details so you know what went wrong
    println!("CRASH DETECTED: {}", panic_info);

    // Wait for user input before allowing the program to exit.
    // This keeps the RenderDoc window / terminal open!
    println!("Press Enter to close the application...");
    let _ = std::io::stdin().read(&mut [0u8]);
  }));

  let start_time = Instant::now();
  println!("[{:.2?}] Application starting.", start_time.elapsed());

  // 1. Setup
  let event_loop = EventLoop::new().unwrap();
  let window = WindowBuilder::new()
    .with_title("AetherVk Simulation")
    .build(&event_loop)
    .unwrap();
  let native_handles = extract_native_handles(&window);

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

  let presentation_engine = {
    let params = gpu::PresentationEngineParams {
      width: window.inner_size().width,
      height: window.inner_size().height,
      vsync: true,
      window_info: native_handles,
    };
    render_frontend
      .write()
      .unwrap()
      .take_and(|context| {
        use aethervk_core_rlib::types::GpuError;

        let mut handle_result: GpuResult<gpu::PresentationEngineHandle> =
          Err(GpuError::InvalidState);
        let mut closure_data = (&params, &mut handle_result);

        let closure = |device: &dyn RenderDevice, data: *mut core::ffi::c_void| {
          // 1. Explicitly define the exact tuple type (two references)
          type ClosureData<'a> = (
            &'a gpu::PresentationEngineParams,
            &'a mut GpuResult<gpu::PresentationEngineHandle>,
          );

          let data_ptr = data as *mut ClosureData;

          // 2. Destructure. Thanks to match ergonomics, `params_ref` is `&mut &PresentationEngineParams`
          let (params_ref, handle_result) = unsafe { &mut *data_ptr };

          // 3. Deref `params_ref` once so we pass `&PresentationEngineParams`
          **handle_result = device.create_presentation_engine(*params_ref);
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
  scene.register_component::<PhysicalMeshComponent>(&[]);
  scene.register_component::<CameraComponent>(&[]);

  let model_path = {
    let mut args = std::env::args();
    println!("You passed {} arguments", args.len());
    if args.len() > 1 {
      let _ = args.next().unwrap(); // discard useless exe path
                                    // interpret first argument as a custom asset path
      std::path::PathBuf::from(args.next().unwrap()).join("Comet.glb")
    } else {
      let mut home_dir = std::env::current_exe().unwrap();
      let mut iter: i32 = 0;
      const MAX_ITER: i32 = 32;
      while {
        let d = home_dir.join("assets/Comet.glb");
        println!("Checking path {:?}", d);
        !d.is_file() && iter < MAX_ITER
      } {
        home_dir.pop();
        iter += 1;
        assert!(home_dir.is_dir());
      }

      home_dir.join("assets/Comet.glb")
    }
  };
  println!("Searching for comet in `{:?}`", &model_path);
  let comet = simulation::comet::load_comet_from_gltf(model_path.to_str().unwrap())
    .expect("Failed to load comet");

  let root_entity = scene.spawn_entity();
  scene
    .add_component(
      root_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: <Vec4f32 as Quaternion>::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();

  let mesh_entity = scene.spawn_entity();
  scene
    .add_component(
      mesh_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: <Vec4f32 as Quaternion>::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(mesh_entity, PhysicalMeshComponent { mesh: comet })
    .unwrap();
  scene.set_parent(mesh_entity, Some(root_entity));

  let camera_entity = scene.spawn_entity();
  scene
    .add_component(
      camera_entity,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 5.0),
        rotation: <Vec4f32 as Quaternion>::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    )
    .unwrap();
  scene
    .add_component(
      camera_entity,
      CameraComponent {
        projection: Mat4x4f32::identity(), // Will be updated on resize
      },
    )
    .unwrap();
  scene.set_parent(camera_entity, Some(root_entity));

  let scene_shared = Arc::new(RwLock::new(scene));

  let mut app_state = AppState {
    scene: Arc::clone(&scene_shared),
    presentation_engine,
    camera_entity,
    is_paused: false,
    time_scale: 1.0,
    root_entity,
    window,
  };

  // --- 2: Spawn render thread and Semaphore channel
  let (render_tx, render_rx) = mpsc::sync_channel::<RenderPacket>(1);
  let render_frontend_clone = Arc::clone(&render_frontend);
  let scene_render_clone = Arc::clone(&scene_shared);
  std::thread::spawn(move || {
    for mut packet in render_rx {
      let scene_guard = scene_render_clone.read().unwrap();
      let mut c_payload = RenderPayloadData {
        packet: &mut packet,
        presentation_engine,
        scene: &scene_guard,
      };

      let res = render_frontend_clone.write().unwrap().take_and(|context| {
        context
          .deref_device_and(
            render_device_handle,
            &mut c_payload as *mut _ as *mut core::ffi::c_void,
            render_payload_ffi,
          )
          .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
      });
      if let Some(Err(e)) = res {
        println!("Render error: {:?}", e);
      }
    }
  });

  // --- 3. Main Event Loop
  let mut right_mouse_button_down = false;
  let camera_focus_point = Vec3f32::from_components(0.0, 0.0, 0.0);
  let mut camera_distance = 5.0;
  let generation = AtomicU64::new(0);
  let mut last_log_time = Instant::now();

  let render_frontend_events = Arc::clone(&render_frontend);

  event_loop.set_control_flow(ControlFlow::Poll);
  event_loop
    .run(move |event, elwt| {
      match event {
        Event::WindowEvent { event, window_id } if window_id == app_state.window.id() => {
          match event {
            WindowEvent::CloseRequested => {
              elwt.exit();
            }
            WindowEvent::Resized(physical_size) => {
              println!("[{:.2?}] Handling Resized event.", start_time.elapsed());
              let mut state = &mut app_state;
              let scene_guard = state.scene.read().unwrap();

              scene_guard.with_component_mut(
                state.camera_entity,
                |camera: &mut CameraComponent| {
                  camera.projection = Mat4x4f32::perspective(
                    45.0f32.to_radians(),
                    physical_size.width as f32 / physical_size.height as f32,
                    0.1,
                    100.0,
                  );
                },
              );

              generation.fetch_add(1, atomic::Ordering::Relaxed);
              state.window.request_redraw();
              println!("[{:.2?}] Finished Resized event.", start_time.elapsed());
            }
            WindowEvent::RedrawRequested => {
              println!(
                "[{:.2?}] Handling RedrawRequested event.",
                start_time.elapsed()
              );

              // --- 1. Data Collection Phase ---
              // Traverse the scene to get all renderable items without holding any GPU locks.
              let mut render_items = Vec::new();
              let mut matrix_stack = vec![Mat4x4f32::identity()];
              let scene_guard = app_state.scene.read().unwrap();

              scene_guard.traverse_with_hooks(
                app_state.root_entity,
                &mut matrix_stack,
                &mut |stack, entity, transform_opt, mesh_opt| {
                  let local_transform = transform_opt
                    .map(|c| {
                      Mat4x4f32::translation(c.position.to_vec4::<Vec4f32>(1.0))
                        * Mat4x4f32::from_quat(c.rotation)
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
                  true // Continue traversal
                },
                &mut |stack, _| {
                  stack.pop();
                },
              );

              // Fetch camera data
              let mut camera_transform = TransformComponent {
                position: Vec3f32::from_components(0.0, 0.0, 0.0),
                rotation: <Vec4f32 as Quaternion>::identity(),
                scale: Vec3f32::from_components(1.0, 1.0, 1.0),
              };
              let mut camera_component = CameraComponent {
                projection: Mat4x4f32::perspective(45.0f32.to_radians(), 800.0 / 600.0, 0.1, 100.0),
              };

              scene_guard.with_component(app_state.camera_entity, |c| camera_transform = *c);
              scene_guard.with_component(app_state.camera_entity, |c| camera_component = *c);

              // Send packet. If the Render Thread is busy, This blocks, locking hte framerate to 16.67 ms
              let packet = RenderPacket {
                render_items,
                camera_transform,
                camera_component,
                window_size: app_state.window.inner_size(),
              };

              if render_tx.send(packet).is_err() {
                elwt.exit(); // Render thread panicked/died
              }
            }
            WindowEvent::MouseInput {
              state: element_state,
              button: winit::event::MouseButton::Right,
              ..
            } => {
              right_mouse_button_down = element_state == ElementState::Pressed;
            }
            _ => {}
          }
        }
        Event::DeviceEvent {
          event: DeviceEvent::MouseMotion { delta },
          ..
        } if right_mouse_button_down => {
          let state = &mut app_state;
          let scene_guard = state.scene.read().unwrap();
          scene_guard.with_component_mut(
            state.camera_entity,
            |camera_transform: &mut TransformComponent| {
              let rotation_speed = 0.005;
              let yaw_delta = delta.0 as f32 * rotation_speed;
              let pitch_delta = delta.1 as f32 * rotation_speed;

              let rotation_y = <Vec4f32 as Quaternion>::from_axis_angle(
                Vec3f32::from_components(0.0, 1.0, 0.0),
                -yaw_delta,
              );
              let right: Vec4f32 = camera_transform.rotation
                * Vec3f32::from_components(1.0, 0.0, 0.0).to_vec4::<Vec4f32>(0.0);
              let rotation_x =
                <Vec4f32 as Quaternion>::from_axis_angle(right.vector_part(), -pitch_delta);

              let new_rotation = rotation_y * rotation_x * camera_transform.rotation;
              camera_transform.rotation = new_rotation;

              let new_forward: Vec4f32 =
                new_rotation * Vec3f32::from_components(0.0, 0.0, -1.0).to_vec4::<Vec4f32>(0.0);
              camera_transform.position =
                camera_focus_point - new_forward.vector_part() * camera_distance;
            },
          );

          generation.fetch_add(1, atomic::Ordering::Relaxed);
          state.window.request_redraw();
        }
        Event::DeviceEvent {
          event: DeviceEvent::MouseWheel { delta, .. },
          ..
        } => {
          let scroll_amount = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
          };
          camera_distance -= scroll_amount * 0.1;
          if camera_distance < 1.0 {
            camera_distance = 1.0;
          }

          let state = &mut app_state;
          let scene_guard = state.scene.read().unwrap();
          scene_guard.with_component_mut(
            state.camera_entity,
            |camera_transform: &mut TransformComponent| {
              let forward: Vec4f32 = camera_transform.rotation
                * Vec3f32::from_components(0.0, 0.0, -1.0).to_vec4::<Vec4f32>(0.0);
              camera_transform.position =
                camera_focus_point - forward.vector_part() * camera_distance;
            },
          );

          generation.fetch_add(1, atomic::Ordering::Relaxed);
          state.window.request_redraw();
        }
        Event::AboutToWait => {
          if last_log_time.elapsed().as_secs() >= 5 {
            println!("[{:.2?}] Liveliness: In AboutToWait.", start_time.elapsed());
            last_log_time = Instant::now();
          }
          let state = &mut app_state;
          state.window.request_redraw();
        }
        _ => (),
      }
    })
    .unwrap();
}

// 4. Render payload executes on the render thtread wiht full type safety
fn render_payload_ffi(device: &dyn RenderDevice, data: *mut core::ffi::c_void) -> GpuResult<()> {
  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  // handle resize
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.packet.window_size.width,
      payload.packet.window_size.height,
    )?;
    return Ok(());
  }

  let mut frame = frame::Frame::new();
  for item in &payload.packet.render_items {
    payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        frame
          .add_renderable(
            device,
            item.entity_id,
            item.model_matrix,
            scene::RenderableDataRef::PhysicalMesh(mesh),
            payload.presentation_engine,
          )
          .unwrap();
        Ok(())
      },
    );
  }

  // Record commands
  let render_path = frame::ForwardRenderPath;
  render_path.record_commands(
    device,
    (
      &payload.packet.camera_transform,
      &payload.packet.camera_component,
    ),
    &frame,
    payload.presentation_engine,
    &acquire_result,
  )?;
  // present
  let present_status = device.present(
    payload.presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  )?;
  if present_status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.packet.window_size.width,
      payload.packet.window_size.height,
    )?;
  }

  Ok(())
}
