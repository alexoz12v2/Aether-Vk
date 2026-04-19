use aethervk_core_rlib::{
  gpu::{self, OpaqueNativeHandleInfo, RenderDevice},
  scene::{
    CameraComponent, CursorComponent, EntityId, GridComponent, PhysicalMeshComponent,
    RenderableDataRef, Scene, SkyComponent, SunComponent, TransformComponent,
  },
  simulation,
  types::RuntimeParams,
  types::GpuResult,
};
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::math::{
  matrix::mat4::Mat4x4f32,
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat},
};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec, collections::BTreeMap};
use heapless::index_map::FnvIndexMap;
use core::{
  any::TypeId,
  ffi::{c_char, CStr},
  sync::{atomic::AtomicBool},
};
use thingbuf::mpsc;
use oshal::os::thread::{self, Thread};
use spin::rwlock::RwLock;
use aethervk_oshal_rlib::math::matrix::{Matrix4, SquareMatrix};
use aethervk_oshal_rlib::math::vector::Vector3;

#[derive(Debug, Clone)]
pub struct RenderItem {
  pub entity_id: EntityId,
  pub model_matrix: Mat4x4f32,
}

#[derive(Debug, Clone)]
pub struct RenderPacket {
  pub render_items: Vec<RenderItem>,
  pub camera_transform: TransformComponent,
  pub camera_component: CameraComponent,
  pub window_width: u32,
  pub window_height: u32,
  pub outlines_enabled: bool,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  packet: &'a mut RenderPacket,
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  cursor_entity: EntityId,
  sun_entity: EntityId,
}

fn start_render_thread(
  render_rx: mpsc::Receiver<Option<RenderPacket>>,
  scene_shared: Arc<Scene>,
  render_frontend: Arc<RwLock<aethervk_core_rlib::gpu::RenderFrontend<'static>>>,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  cursor_entity: EntityId,
  sun_entity: EntityId,
) -> Thread {
  thread::spawn(move || {
    loop {
      match render_rx.try_recv() {
        Ok(Some(mut packet)) => {
          let scene_guard = scene_shared.as_ref();
          let mut c_payload = RenderPayloadData {
            packet: &mut packet,
            presentation_engine,
            scene: &scene_guard,
            cursor_entity,
            sun_entity,
          };

          let res = render_frontend.write().take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                &mut c_payload as *mut _ as *mut core::ffi::c_void,
                render_payload_ffi,
              )
              .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          });
          if let Some(Err(_e)) = res {
            // Handle render error
          }
        }
        Ok(None) => break, // Exit loop
        Err(_) => {
          core::hint::spin_loop();
        }
      }
    }
  })
  .unwrap()
}

fn render_payload_ffi(device: &dyn RenderDevice, data: *mut core::ffi::c_void) -> GpuResult<()> {
  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  if acquire_result.status.needs_resize() {
    device.resize_presentation_engine(
      payload.presentation_engine,
      payload.packet.window_width,
      payload.packet.window_height,
    )?;
    return Ok(());
  }

  let mut render_scene = gpu::frame::RenderScene::new((
    payload.packet.camera_transform,
    payload.packet.camera_component,
  ));

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SunComponent, _>(|entity, comp| {
      if let Some(transform) = payload.scene.global_transform(entity) {
        render_scene.sun = Some((entity, *comp, transform));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
      render_scene.sky = Some((entity, *comp));
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
      render_scene.grid = Some((entity, *comp));
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::CursorComponent, _>(|entity, comp| {
      if let Some(transform) = payload.scene.global_transform(entity) {
        render_scene
          .add_renderable(
            device,
            entity,
            transform.to_mat4(),
            RenderableDataRef::Cursor(comp),
            payload.presentation_engine,
            "Cursor",
            false,
            [1.0, 1.0, 1.0, 1.0],
          )
          .unwrap();
      }
    });

  for item in &payload.packet.render_items {
    payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        render_scene
          .add_renderable(
            device,
            item.entity_id,
            item.model_matrix,
            RenderableDataRef::PhysicalMesh(mesh),
            payload.presentation_engine,
            "Comet",
            false,
            [1.0, 1.0, 1.0, 1.0],
          )
          .unwrap();
        Ok(())
      },
    );
  }

  if payload.packet.outlines_enabled {
    for item in &payload.packet.render_items {
      payload.scene.with_component(
        item.entity_id,
        |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
          render_scene
            .add_renderable(
              device,
              item.entity_id,
              item.model_matrix,
              RenderableDataRef::PhysicalMesh(mesh),
              payload.presentation_engine,
              "Outline",
              true,
              [1.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
          Ok(())
        },
      );
    }
  }

  let mut sun_opt = None;
  payload.scene.with_component(
    payload.sun_entity,
    |sun_comp: &aethervk_core_rlib::scene::SunComponent| {
      sun_opt = Some(*sun_comp);
    },
  );
  if let Some(sun_comp) = sun_opt {
    if let Some(sun_transform) = payload.scene.global_transform(payload.sun_entity) {
      let mut sky_opt = None;
      payload
        .scene
        .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|id, comp| {
          sky_opt = Some((id, *comp));
        });

      let mut grid_opt = None;
      payload
        .scene
        .query1::<aethervk_core_rlib::scene::GridComponent, _>(|id, comp| {
          grid_opt = Some((id, *comp));
        });

      render_scene.sun = Some((payload.sun_entity, sun_comp, sun_transform.into()));
      if let Some((id, comp)) = sky_opt {
        render_scene.sky = Some((id, comp));
      }
      if let Some((id, comp)) = grid_opt {
        render_scene.grid = Some((id, comp));
      }

      let cmd_buffer = device.get_command_buffer()?;
      device.begin_command_buffer(cmd_buffer)?;
      device.update_sun(cmd_buffer, payload.sun_entity, &sun_comp)?;
      device.begin_render_pass(cmd_buffer, payload.presentation_engine, &acquire_result)?;

      let extent = device.get_presentation_engine_extent(payload.presentation_engine)?;
      let root_viewport = gpu::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent[0] as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
      };
      device.set_viewport(cmd_buffer, &root_viewport)?;
      let _ = device.set_scissor(
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
            camera_entity: None,
          },
          children: None,
        },
      };
      device.render_frame(cmd_buffer, &quad_tree, &render_scene)?;

      device.end_render_pass(cmd_buffer)?;
      device.submit_command_buffer(cmd_buffer)?;

      let present_status = device.present(
        payload.presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;
      if present_status.needs_resize() {
        let _ = device.resize_presentation_engine(
          payload.presentation_engine,
          payload.packet.window_width,
          payload.packet.window_height,
        );
      }
    }
  }

  Ok(())
}

pub struct SimulationContext {
  pub scene: Arc<Scene>,
  pub presentation_engine: gpu::PresentationEngineHandle,
  pub render_frontend: Arc<RwLock<aethervk_core_rlib::gpu::RenderFrontend<'static>>>,
  pub render_device_handle: gpu::RenderDeviceHandle,

  pub entity_map: BTreeMap<u64, EntityId>,
  pub next_entity_id: u64,

  pub root_entity: EntityId,
  pub camera_entity: EntityId,
  pub cursor_entity: EntityId,
  pub sun_entity: EntityId,
  pub grid_entity: EntityId,

  pub outlines_enabled: Arc<AtomicBool>,
  pub render_tx: Option<mpsc::Sender<Option<RenderPacket>>>,
  pub render_thread_handle: Option<Thread>,

  pub window_width: u32,
  pub window_height: u32,
}

impl SimulationContext {
  fn register_entity(&mut self, id: EntityId) -> u64 {
    let external_id = self.next_entity_id;
    self.next_entity_id += 1;
    self.entity_map.insert(external_id, id);
    external_id
  }

  fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    self.entity_map.get(&external_id).copied()
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  ptr0: *mut core::ffi::c_void,
  ptr1: *mut core::ffi::c_void,
  width: u32,
  height: u32,
) -> *mut SimulationContext {
  let runtime_params = Box::leak(Box::new(RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
  }));

  let render_frontend = Arc::new(RwLock::new(
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, runtime_params).unwrap(),
  ));

  let additional_params = gpu::DeviceAdditionalParams::new();
  let mut write_render_frontend = render_frontend.write();
  let render_device_handle = write_render_frontend
    .take_mut_and(|context| Ok(context.init_device(0, &additional_params).unwrap()))
    .unwrap()
    .unwrap();

  let native_handles = OpaqueNativeHandleInfo { ptr0, ptr1 };

  let presentation_engine = {
    let params = gpu::PresentationEngineParams {
      width,
      height,
      vsync: true,
      ty: gpu::PresentationEngineType::Window,
      window_info: native_handles,
    };
    write_render_frontend
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
          let pe_result = device.create_presentation_engine(*params_ref);
          if let Ok(pe) = pe_result {
            device.init_archetypes(pe).unwrap();
          }
          **handle_result = pe_result;

          device.generate_sky().unwrap();
          Ok(())
        };

        context
          .deref_device_and(
            render_device_handle,
            &mut closure_data as *mut _ as *mut core::ffi::c_void,
            closure,
          )
          .unwrap()
          .unwrap();
        Ok(handle_result.unwrap())
      })
      .unwrap()
      .unwrap()
  };
  drop(write_render_frontend);

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SkyComponent>(&[]);
  scene.register_component::<GridComponent>(&[]);

  let root_entity = scene.spawn_entity("root");
  let _ = scene.add_component(
    root_entity,
    TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    },
  );

  let camera_entity = scene.spawn_entity("camera");
  let _ = scene.add_component(
    camera_entity,
    TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 400.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    },
  );
  let _ = scene.add_component(
    camera_entity,
    CameraComponent {
      projection: Mat4x4f32::perspective_vk(
        45.0f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        10000.0,
      ),
    },
  );
  scene.set_parent(camera_entity, Some(root_entity));

  let cursor_entity = scene.spawn_entity("cursor");
  let _ = scene.add_component(
    cursor_entity,
    TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    },
  );
  let _ = scene.add_component(cursor_entity, CursorComponent {});
  scene.set_parent(cursor_entity, Some(root_entity));

  let sun_entity = scene.spawn_entity("sun");
  let _ = scene.add_component(
    sun_entity,
    TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    },
  );
  let _ = scene.add_component(
    sun_entity,
    SunComponent {
      resolution: (128, 128, 128),
    },
  );
  scene.set_parent(sun_entity, Some(root_entity));

  let grid_entity = scene.spawn_entity("grid");
  let _ = scene.add_component(grid_entity, GridComponent {});
  scene.set_parent(grid_entity, Some(root_entity));

  let mut ctx = Box::new(SimulationContext {
    scene: Arc::new(scene),
    presentation_engine,
    render_frontend,
    render_device_handle,
    entity_map: BTreeMap::new(),
    next_entity_id: 1,
    root_entity,
    camera_entity,
    cursor_entity,
    sun_entity,
    grid_entity,
    outlines_enabled: Arc::new(AtomicBool::new(false)),
    render_tx: None,
    render_thread_handle: None,
    window_width: width,
    window_height: height,
  });

  ctx.register_entity(root_entity);
  ctx.register_entity(camera_entity);
  ctx.register_entity(cursor_entity);
  ctx.register_entity(sun_entity);
  ctx.register_entity(grid_entity);

  Box::into_raw(ctx)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if !ctx.is_null() {
    let mut ctx = unsafe { Box::from_raw(ctx) };
    unsafe { avkSimulationContext_stopThreads(&mut *ctx) };
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startThreads(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  let (render_tx, render_rx) = mpsc::channel::<Option<RenderPacket>>(32);
  let render_thread_handle = start_render_thread(
    render_rx,
    Arc::clone(&ctx.scene),
    Arc::clone(&ctx.render_frontend),
    ctx.render_device_handle,
    ctx.presentation_engine,
    ctx.cursor_entity,
    ctx.sun_entity,
  );
  ctx.render_tx = Some(render_tx);
  ctx.render_thread_handle = Some(render_thread_handle);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_stopThreads(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  if let Some(tx) = &ctx.render_tx {
    let _ = tx.try_send(None);
  }
  if let Some(handle) = ctx.render_thread_handle.take() {
    handle.join();
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnEntity(
  ctx: *mut SimulationContext,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx = unsafe { &mut *ctx };
  let name_str = if name.is_null() {
    "Entity"
  } else {
    unsafe { CStr::from_ptr(name).to_str().unwrap_or("Entity") }
  };
  let id = ctx.scene.spawn_entity(name_str);
  ctx.register_entity(id)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setParent(
  ctx: *mut SimulationContext,
  entity: u64,
  parent: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  let entity_id = match ctx.get_entity(entity) {
    Some(id) => id,
    None => return,
  };
  let parent_opt = if parent == 0 {
    None
  } else {
    ctx.get_entity(parent)
  };
  ctx.scene.set_parent(entity_id, parent_opt);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addTransformComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  pos_x: f32,
  pos_y: f32,
  pos_z: f32,
  rot_w: f32,
  rot_x: f32,
  rot_y: f32,
  rot_z: f32,
  scale_x: f32,
  scale_y: f32,
  scale_z: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(
      entity_id,
      TransformComponent {
        position: Vec3f32::from_components(pos_x, pos_y, pos_z),
        rotation: Quat::from_components(rot_w, rot_x, rot_y, rot_z),
        scale: Vec3f32::from_components(scale_x, scale_y, scale_z),
      },
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addPhysicalMeshComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  gltf_path: *const c_char,
  emissive_intensity: f32,
  emissive_r: f32,
  emissive_g: f32,
  emissive_b: f32,
) -> bool {
  if ctx.is_null() || gltf_path.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let path_str = unsafe { CStr::from_ptr(gltf_path).to_str().unwrap_or("") };
    if let Ok(mesh) = simulation::comet::load_comet_from_gltf(path_str, true) {
      let _ = ctx.scene.add_component(
        entity_id,
        PhysicalMeshComponent {
          mesh,
          emissive_intensity,
          emissive_color: [emissive_r, emissive_g, emissive_b],
        },
      );
      return true;
    }
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSkyComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(entity_id, SkyComponent {});
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  fov: f32,
  aspect: f32,
  near: f32,
  far: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(
      entity_id,
      CameraComponent {
        projection: Mat4x4f32::perspective_vk(fov, aspect, near, far),
      },
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCursorComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(entity_id, CursorComponent {});
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSunComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  res_x: u32,
  res_y: u32,
  res_z: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(
      entity_id,
      SunComponent {
        resolution: (res_x, res_y, res_z),
      },
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addGridComponent(
  ctx: *mut SimulationContext,
  entity: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let _ = ctx.scene.add_component(entity_id, GridComponent {});
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_renderTick(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  let mut render_items = Vec::new();
  let mut matrix_stack = vec![Mat4x4f32::identity()];

  ctx.scene.traverse_with_hooks(
    ctx.root_entity,
    &mut matrix_stack,
    &mut |stack, entity, transform_opt, mesh_opt: Option<&PhysicalMeshComponent>| {
      let local_transform = transform_opt
        .map(|c| {
          Mat4x4f32::translation(c.position)
            * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(c.rotation)
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
  };

  if let Some(global) = ctx.scene.global_transform(ctx.camera_entity) {
    camera_transform = global;
  }
  ctx
    .scene
    .with_component(ctx.camera_entity, |c| camera_component = *c);

  let packet = RenderPacket {
    render_items,
    camera_transform,
    camera_component,
    window_width: ctx.window_width,
    window_height: ctx.window_height,
    outlines_enabled: ctx
      .outlines_enabled
      .load(core::sync::atomic::Ordering::Relaxed),
  };

  if let Some(tx) = &ctx.render_tx {
    let _ = tx.try_send(Some(packet));
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_resize(
  ctx: *mut SimulationContext,
  width: u32,
  height: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  ctx.window_width = width;
  ctx.window_height = height;
  ctx
    .scene
    .with_component_mut(ctx.camera_entity, |c: &mut CameraComponent| {
      c.projection = Mat4x4f32::perspective_vk(
        45.0f32.to_radians(),
        width as f32 / height as f32,
        0.1,
        10000.0,
      );
    });
}
