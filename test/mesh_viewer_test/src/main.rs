pub mod windowing;

use aethervk_core_rlib::{
  gpu::{self, RenderDevice, frame::RenderScene},
  scene::{
    CameraComponent, EntityId, PhysicalMeshComponent, RenderableDataRef, Scene, SunComponent,
    TransformComponent,
  },
  types::RuntimeParams,
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, SquareMatrix, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector3},
};
use heapless::index_map::FnvIndexMap;
use std::{
  sync::{Arc, RwLock},
  time::Instant,
};
use winit::{
  event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
  event_loop::{ControlFlow, EventLoopBuilder},
  keyboard::{KeyCode, PhysicalKey},
  window::WindowBuilder,
};
use rfd::FileDialog;

use windowing::{AppEvent, WindowExtractHandlesParams, extract_native_handles, setup_resize_hook};

struct AppState {
  is_resizing: bool,
  is_exiting: bool,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  camera_entity: EntityId,
  mesh_entity: EntityId,
  sun_entity: EntityId,
  width: u32,
  height: u32,
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
  let asset_path = {
    let mut path = std::env::current_exe()
      .unwrap()
      .parent()
      .unwrap()
      .to_owned();
    while !path.join("assets").exists() {
      path = path.parent().unwrap().to_owned();
    }
    path.join("assets")
  };

  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_path.to_str().unwrap().to_string());
  drop(guard);

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

  let proxy = event_loop.create_proxy();
  let proxy_ptr = unsafe { std::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(proxy))) };

  let window = WindowBuilder::new()
    .with_title("Mesh Viewer")
    .with_inner_size(winit::dpi::PhysicalSize::new(800, 600))
    .build(&event_loop)
    .unwrap();
  setup_resize_hook(&window, proxy_ptr);

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
  let (native_handles, _window_info) = extract_native_handles(&window, &params);

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
      .write()
      .unwrap()
      .take_and(|context| {
        let mut handle_result: aethervk_core_rlib::types::GpuResult<gpu::PresentationEngineHandle> =
          Err(aethervk_core_rlib::types::GpuError::InvalidState);
        let mut closure_data = (&params, &mut handle_result);

        let closure = |device: &dyn gpu::RenderDevice, data: *mut core::ffi::c_void| {
          type ClosureData<'a> = (
            &'a gpu::PresentationEngineParams,
            &'a mut aethervk_core_rlib::types::GpuResult<gpu::PresentationEngineHandle>,
          );

          let data_ptr = data as *mut ClosureData;
          let (params_ref, handle_result) = unsafe { &mut *data_ptr };
          let res = device.create_presentation_engine(*params_ref);
          if let Ok(pe) = res {
            device
              .init_archetypes(pe)
              .expect("Failed to initialize archetypes");
          }
          **handle_result = res;
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

  let asset_path = {
    let mut path = std::env::current_exe()
      .unwrap()
      .parent()
      .unwrap()
      .to_owned();
    while !path.join("assets").exists() && path.parent().is_some() {
      path = path.parent().unwrap().to_owned();
    }
    path.join("assets")
  };
  let mut guard = aethervk_core_rlib::gpu::ASSET_DIR.write();
  *guard = Some(asset_path.to_str().unwrap().to_string());
  drop(guard);

  let mut scene = Scene::new();
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
      },
    )
    .unwrap();

  let mut app_state = AppState {
    is_resizing: false,
    is_exiting: false,
  };

  let mut right_mouse_button_down = false;

  event_loop.set_control_flow(ControlFlow::Poll);
  event_loop
    .run(move |event, elwt| match event {
      Event::UserEvent(app_event) => match app_event {
        AppEvent::ResizeStarted => {
          app_state.is_resizing = true;
        }
        AppEvent::ResizeEnded => {
          app_state.is_resizing = false;
          window.request_redraw();
        }
      },
      Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
        WindowEvent::CloseRequested => {
          app_state.is_exiting = true;
          elwt.exit();
        }
        WindowEvent::Resized(physical_size) => {
          width = physical_size.width;
          height = physical_size.height;
          #[cfg(target_os = "macos")]
          {
            _window_info
              .metal_layer
              .setDrawableSize(objc2_core_foundation::CGSize {
                width: physical_size.width as f64,
                height: physical_size.height as f64,
              });
          }
          scene.with_component_mut(camera_entity, |c: &mut CameraComponent| {
            c.projection = Mat4x4f32::perspective_vk(
              std::f32::consts::FRAC_PI_4,
              width as f32 / height as f32,
              0.1,
              100.0,
            );
          });
        }
        WindowEvent::MouseInput {
          state: element_state,
          button,
          ..
        } => {
          if button == MouseButton::Right {
            right_mouse_button_down = element_state == ElementState::Pressed;
          }
        }
        WindowEvent::MouseWheel { delta, .. } => {
          let scroll_amount = match delta {
            winit::event::MouseScrollDelta::LineDelta(_, y) => y,
            winit::event::MouseScrollDelta::PixelDelta(pos) => (pos.y / 10.0) as f32,
          };
          cam_dist = (cam_dist - scroll_amount).max(0.1);

          let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), cam_yaw);
          let pitch_quat =
            Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), cam_pitch);
          let new_rot = yaw_quat * pitch_quat;
          let offset = Vec3f32::from_components(0.0, -1.0, 0.0) * cam_dist;
          let new_offset = new_rot.rotate_vector(offset);
          scene.with_component_mut(camera_entity, |c: &mut TransformComponent| {
            c.position = new_offset;
            c.rotation = new_rot;
          });
          window.request_redraw();
        }
        WindowEvent::KeyboardInput { event, .. } => {
          if event.state == ElementState::Pressed {
            if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
              app_state.is_exiting = true;
              elwt.exit();
            }
          }
        }
        WindowEvent::RedrawRequested => {
          if app_state.is_resizing || app_state.is_exiting || width == 0 || height == 0 {
            return;
          }

          let mut payload = RenderPayloadData {
            presentation_engine,
            scene: &scene,
            camera_entity,
            mesh_entity,
            sun_entity,
            width,
            height,
          };

          let res = render_frontend.write().unwrap().take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                &mut payload as *mut _ as *mut core::ffi::c_void,
                |device: &dyn RenderDevice,
                 data: *mut core::ffi::c_void|
                 -> aethervk_core_rlib::types::GpuResult<()> {
                  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

                  device.start_frame()?;
                  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
                  if acquire_result.status.needs_resize() {
                    device.resize_presentation_engine(
                      payload.presentation_engine,
                      payload.width,
                      payload.height,
                    )?;
                    return Ok(());
                  }

                  let mut camera_transform = TransformComponent {
                    position: Vec3f32::from_components(0.0, 0.0, 0.0),
                    rotation: Quat::identity(),
                    scale: Vec3f32::from_components(1.0, 1.0, 1.0),
                  };
                  let mut camera_component = CameraComponent {
                    projection: Mat4x4f32::identity(),
                    near_plane: 0.1,
                    far_plane: 100.0,
                  };
                  payload
                    .scene
                    .with_component(payload.camera_entity, |c: &TransformComponent| {
                      camera_transform = *c
                    });
                  payload
                    .scene
                    .with_component(payload.camera_entity, |c: &CameraComponent| {
                      camera_component = *c
                    });

                  let mut render_scene = RenderScene::new((camera_transform, camera_component));

                  payload.scene.with_component(
                    payload.mesh_entity,
                    |mesh: &PhysicalMeshComponent| {
                      render_scene
                        .add_renderable(
                          device,
                          payload.mesh_entity,
                          Mat4x4f32::identity(),
                          RenderableDataRef::PhysicalMesh(mesh),
                          payload.presentation_engine,
                          "mesh",
                          false,
                          [1.0, 1.0, 1.0, 1.0],
                        )
                        .unwrap();
                    },
                  );

                  let mut sun_transform = TransformComponent {
                    position: Vec3f32::from_components(0.0, 0.0, 0.0),
                    rotation: Quat::identity(),
                    scale: Vec3f32::from_components(1.0, 1.0, 1.0),
                  };
                  let mut sun_component = SunComponent {
                    resolution: (128, 128, 128),
                  };
                  payload
                    .scene
                    .with_component(payload.sun_entity, |c: &TransformComponent| {
                      sun_transform = *c
                    });
                  payload
                    .scene
                    .with_component(payload.sun_entity, |c: &SunComponent| sun_component = *c);
                  render_scene.sun =
                    Some((payload.sun_entity, sun_component, sun_transform.into()));

                  let cmd_buffer = device.get_command_buffer()?;
                  device.begin_command_buffer(cmd_buffer)?;
                  device.update_sun(cmd_buffer, payload.sun_entity, &sun_component)?;
                  device.begin_render_pass(
                    cmd_buffer,
                    payload.presentation_engine,
                    &acquire_result,
                  )?;

                  let extent =
                    device.get_presentation_engine_extent(payload.presentation_engine)?;
                  let root_viewport = gpu::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent[0] as f32,
                    height: extent[1] as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                  };
                  device.set_viewport(cmd_buffer, &root_viewport)?;
                  device.set_scissor(
                    cmd_buffer,
                    &gpu::Rect2D {
                      offset: [0, 0],
                      extent,
                    },
                  );

                  let quad_tree = gpu::ViewportQuadTree {
                    root: gpu::viewport::ViewportNode {
                      viewport: root_viewport,
                      scissor: gpu::Rect2D {
                        offset: [0, 0],
                        extent,
                      },
                      program: gpu::viewport::DrawingProgram::Viewport3D {
                        camera_entity: Some(payload.camera_entity),
                      },
                      children: None,
                    },
                  };

                  device.render_frame(cmd_buffer, &quad_tree, &render_scene)?;
                  device.end_render_pass(cmd_buffer)?;
                  device.submit_command_buffer(cmd_buffer, None)?;

                  let present_status = device.present(
                    payload.presentation_engine,
                    acquire_result.image_index as usize,
                    acquire_result.frame_index as usize,
                  )?;
                  if present_status.needs_resize() {
                    device.resize_presentation_engine(
                      payload.presentation_engine,
                      payload.width,
                      payload.height,
                    )?;
                  }

                  Ok(())
                },
              )
              .unwrap()?;
            Ok(())
          });
        }
        _ => {}
      },
      Event::DeviceEvent {
        event: DeviceEvent::MouseMotion { delta },
        ..
      } => {
        if right_mouse_button_down {
          let rotation_speed = 0.005;
          cam_yaw += delta.0 as f32 * rotation_speed;
          cam_pitch -= delta.1 as f32 * rotation_speed;
          cam_yaw = cam_yaw % (std::f32::consts::PI * 2.0);
          cam_pitch = cam_pitch.clamp(-1.55, 1.55);

          let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), cam_yaw);
          let pitch_quat =
            Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), cam_pitch);
          let new_rot = yaw_quat * pitch_quat;
          let offset = Vec3f32::from_components(0.0, -1.0, 0.0) * cam_dist;
          let new_offset = new_rot.rotate_vector(offset);
          scene.with_component_mut(camera_entity, |c: &mut TransformComponent| {
            c.position = new_offset;
            c.rotation = new_rot;
          });
          window.request_redraw();
        }
      }
      Event::AboutToWait => {
        if !app_state.is_resizing && !app_state.is_exiting {
          window.request_redraw();
        }
      }
      _ => (),
    })
    .unwrap();
}
