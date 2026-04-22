use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
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
  matrix::{mat4::Mat4x4f32, Matrix, MatrixVectorMul},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector4},
};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec, collections::BTreeMap, string::String, format};
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
use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;
use aethervk_oshal_rlib::math::vector::Vector3;

#[derive(Default)]
pub struct AlmanacPackedData {
  pub data: Vec<Vec<u8>>,
  pub file_names: Vec<String>,
  pub almanac: anise::almanac::Almanac,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TimeScale {
  Stopped,
  OneDay,
  OneWeek,
  OneMonth,
}

impl TimeScale {
  fn to_days_per_st_second(self) -> f64 {
    match self {
      TimeScale::Stopped => 0.0,
      TimeScale::OneDay => 1.0,
      TimeScale::OneWeek => 7.0,
      TimeScale::OneMonth => 30.436875,
    }
  }
}

pub struct LogicState {
  pub almanac_data: AlmanacPackedData,
  pub current_scale: TimeScale,
  pub current_epoch: anise::time::Epoch,
  pub epoch_start: anise::time::Epoch,
  pub epoch_end: anise::time::Epoch,
  pub st_seconds_elapsed: f64,
}

impl Default for LogicState {
  fn default() -> Self {
    Self {
      almanac_data: AlmanacPackedData::default(),
      current_scale: TimeScale::Stopped,
      current_epoch: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_start: anise::time::Epoch::from_gregorian_utc_at_midnight(2000, 1, 1),
      epoch_end: anise::time::Epoch::from_gregorian_utc_at_midnight(2100, 1, 1),
      st_seconds_elapsed: 0.0,
    }
  }
}

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
  pub clear_color: [f32; 4],
}

pub enum RenderCommand {
  RenderFrame {
    packet: RenderPacket,
    task_id: u64,
  },
  DownloadImage {
    buffer: *mut u8,
    buffer_size: usize,
    success: *mut bool,
    done_signal: Arc<AtomicBool>,
  },
  SetClearColor([f32; 4]),
  Resize {
    width: u32,
    height: u32,
  },
  GenerateSky,
  Shutdown,
}

unsafe impl Send for RenderCommand {}

#[repr(C)]
struct RenderPayloadData<'a> {
  packet: &'a mut RenderPacket,
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  cursor_entity: EntityId,
  sun_entity: EntityId,
  task_id: u64,
}

fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  scene_shared: Arc<Scene>,
  frontend: Arc<aethervk_core_rlib::gpu::RenderFrontend<'static>>,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  cursor_entity: EntityId,
  sun_entity: EntityId,
) -> Thread {
  thread::spawn(move || {
    let mut clear_color = [0.0, 0.0, 0.0, 1.0];
    loop {
      match render_rx.recv() {
        Ok(RenderCommand::RenderFrame { mut packet, task_id }) => {
          packet.clear_color = clear_color;
          let scene_guard = scene_shared.as_ref();
          let mut c_payload = RenderPayloadData {
            packet: &mut packet,
            presentation_engine,
            scene: &scene_guard,
            cursor_entity,
            sun_entity,
            task_id,
          };

          let res = frontend.take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                &mut c_payload as *mut _ as *mut core::ffi::c_void,
                render_payload_ffi,
              )
              .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          });
          
          if let Some(Err(e)) = res {
            // Report failure to task registry
            let _ = frontend.take_and(|context| {
              context.deref_device_and(
                render_device_handle,
                &mut (task_id, e) as *mut _ as *mut core::ffi::c_void,
                |device, data| {
                  let (tid, err) = unsafe { &*(data as *mut (u64, aethervk_core_rlib::types::EngineError)) };
                  if let aethervk_core_rlib::types::EngineError::Gpu(gpu_err) = err {
                    device.fail_task(*tid, gpu_err.clone());
                  } else {
                    device.fail_task(*tid, aethervk_core_rlib::types::GpuError::InvalidState);
                  }
                  Ok(())
                }
              ).unwrap_or(Ok(()))
            });
          }
        }
        Ok(RenderCommand::DownloadImage {
          buffer,
          buffer_size,
          success,
          done_signal,
        }) => {
          let slice = unsafe { core::slice::from_raw_parts_mut(buffer, buffer_size) };
          let mut payload = (presentation_engine, slice);
          let res = frontend.take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                &mut payload as *mut _ as *mut core::ffi::c_void,
                |device, data| {
                  let (engine, buf) =
                    unsafe { &mut *(data as *mut (gpu::PresentationEngineHandle, &mut [u8])) };
                  device.download_windowless_image(*engine, *buf)
                },
              )
              .unwrap()
          });
          unsafe { *success = res.is_ok() };
          done_signal.store(true, core::sync::atomic::Ordering::Release);
        }
        Ok(RenderCommand::SetClearColor(color)) => {
          clear_color = color;
        }
        Ok(RenderCommand::Resize { width, height }) => {
          let mut data = (presentation_engine, width, height);
          let _ = frontend.take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                &mut data as *mut _ as *mut core::ffi::c_void,
                |device, data_ptr| {
                  let (pe, w, h) =
                    unsafe { &mut *(data_ptr as *mut (gpu::PresentationEngineHandle, u32, u32)) };
                  device.resize_presentation_engine(*pe, *w, *h)
                },
              )
              .unwrap();
            Ok(())
          });
        }
        Ok(RenderCommand::GenerateSky) => {
          let _ = frontend.take_and(|context| {
            context
              .deref_device_and(
                render_device_handle,
                core::ptr::null_mut(),
                |device, _| device.generate_sky(),
              )
              .unwrap();
            Ok(())
          });
        }
        Ok(RenderCommand::Shutdown) => break,
        Err(e) => {
          if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
            break;
          }
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
    // handled via resize command or next frame
    return Ok(());
  }

  let mut render_scene = gpu::frame::RenderScene::new((
    payload.packet.camera_transform,
    payload.packet.camera_component,
  ));

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SunComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        if let Some(transform) = payload.scene.global_transform(entity) {
          render_scene.sun = Some((entity, *comp, transform));
        }
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        render_scene.sky = Some((entity, *comp));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        render_scene.grid = Some((entity, *comp));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::CursorComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
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
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::MeasurementComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        render_scene
          .add_renderable(
            device,
            entity,
            aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity(),
            RenderableDataRef::Measurement(comp),
            payload.presentation_engine,
            "Measurement",
            false,
            [1.0, 1.0, 1.0, 1.0],
          )
          .unwrap();
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::ImageBillboardComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        let mut model_matrix = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity();
        if let Some(transform) = payload.scene.global_transform(entity) {
          model_matrix = transform.to_mat4();
        }
        render_scene
          .add_renderable(
            device,
            entity,
            model_matrix,
            RenderableDataRef::ImageBillboard(comp),
            payload.presentation_engine,
            "ImageBillboard",
            false,
            [1.0, 1.0, 1.0, 1.0],
          )
          .unwrap();
      }
    });
  for item in &payload.packet.render_items {
    let is_hidden = payload
      .scene
      .with_component(
        item.entity_id,
        |_c: &aethervk_core_rlib::scene::HiddenComponent| {},
      )
      .is_some();
    if is_hidden {
      continue;
    }
    payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        let mut draw_outline = payload.packet.outlines_enabled;
        let mut outline_color = [0.0, 0.0, 0.0, 0.0]; // Hidden by default, unless overriden

        let is_selected = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::SelectedComponent| {},
          )
          .is_some();
        let is_following = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::FollowingComponent| {},
          )
          .is_some();

        if is_selected {
          draw_outline = true;
          outline_color = [1.0, 1.0, 1.0, 1.0]; // White precedence
        } else if is_following {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 1.0]; // Blueish outline
        } else if payload.packet.outlines_enabled {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 0.5]; // faint blue for global outlines if enabled
        }

        render_scene
          .add_renderable(
            device,
            item.entity_id,
            item.model_matrix,
            RenderableDataRef::PhysicalMesh(mesh),
            payload.presentation_engine,
            "Comet",
            draw_outline,
            outline_color,
          )
          .unwrap();
        Ok(())
      },
    );
  }

  // BVH debug rendering
  let mut all_bvh_nodes = Vec::new();
  for item in &payload.packet.render_items {
    let is_hidden = payload
      .scene
      .with_component(
        item.entity_id,
        |_c: &aethervk_core_rlib::scene::HiddenComponent| {},
      )
      .is_some();
    if is_hidden {
      continue;
    }
    let mut dbg_states = None;
    payload.scene.with_component(
      item.entity_id,
      |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
        dbg_states = Some(dbg.node_render_states.clone());
      },
    );

    if let Some(states) = dbg_states {
      payload
        .scene
        .with_component(item.entity_id, |mesh: &PhysicalMeshComponent| {
          if let Some(bvh) = &mesh.mesh.bvh {
            for (i, &render) in states.iter().enumerate() {
              if render && i < bvh.nodes.len() {
                all_bvh_nodes.push((bvh.nodes[i].bound.clone(), item.model_matrix));
              }
            }
          }
        });
    }
  }

  let mut sun_opt = None;
  if payload.scene.with_component(payload.sun_entity, |_: &aethervk_core_rlib::scene::HiddenComponent| {}).is_none() {
    payload.scene.with_component(
      payload.sun_entity,
      |sun_comp: &aethervk_core_rlib::scene::SunComponent| {
        sun_opt = Some(*sun_comp);
      },
    );
  }
  if let Some(sun_comp) = sun_opt {
    if let Some(sun_transform) = payload.scene.global_transform(payload.sun_entity) {
      let mut sky_opt = None;
      payload
        .scene
        .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
          if payload.scene.with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {}).is_none() {
            sky_opt = Some((entity, *comp));
          }
        });

      let mut grid_opt = None;
      payload
        .scene
        .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
          if payload.scene.with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {}).is_none() {
            grid_opt = Some((entity, *comp));
          }
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

      let _ = device.render_ui_rect(
        cmd_buffer,
        payload.packet.clear_color,
        [-1.0, -1.0],
        [2.0, 2.0],
        payload.presentation_engine,
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

      // Compute view matrix to print
      let view =
      <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        payload.packet.camera_transform.rotation.conjugate(),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::translation(payload.packet.camera_transform.position * -1.0);

      let view_proj = payload.packet.camera_component.projection * view;

      if !all_bvh_nodes.is_empty() {
        let _ = device.render_bvh(
          cmd_buffer,
          &all_bvh_nodes,
          view_proj.into(),
          payload.presentation_engine,
        );
      }

      let font_path = if cfg!(target_os = "windows") {
        "C:/Windows/Fonts/segoeui.ttf"
      } else if cfg!(target_os = "macos") {
        "/System/Library/Fonts/SFNS.ttf"
      } else {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
      };

      payload
        .scene
        .query1::<aethervk_core_rlib::scene::MeasurementComponent, _>(|entity, comp| {
          if payload
            .scene
            .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
            .is_none()
          {
            let mid = comp.pos1 + (comp.pos2 - comp.pos1) * 0.5;
            let mid_vec4 = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
              mid.x(),
              mid.y(),
              mid.z(),
              1.0,
            );
            let mut clip = view_proj.mul_vector(mid_vec4);
            if clip.w() > 0.0 {
              clip = clip / clip.w();
              if clip.z() >= 0.0 && clip.z() <= 1.0 {
                let ndc_x = clip.x();
                let ndc_y = clip.y();

                let screen_x = (ndc_x * 0.5 + 0.5) * payload.packet.window_width as f32;
                let screen_y = (ndc_y * 0.5 + 0.5) * payload.packet.window_height as f32;

                let distance = (comp.pos2 - comp.pos1).length();
                let text = alloc::format!("{:.3} m", distance);

                let _ = device.render_text(
                  cmd_buffer,
                  &text,
                  font_path,
                  24.0, // Or some reasonable font size
                  [1.0, 1.0, 1.0, 1.0],
                  [screen_x, screen_y],
                  payload.presentation_engine,
                );
              }
            }
          }
        });

      device.end_render_pass(cmd_buffer)?;
      device.submit_command_buffer(cmd_buffer, Some(payload.task_id))?;

      let present_status = device.present(
        payload.presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      )?;
      if present_status.needs_resize() {
        // handled via next frame or resize command
      }
    }
  }

  Ok(())
}

pub struct SimulationContext {
  pub scene: Arc<Scene>,
  pub presentation_engine: gpu::PresentationEngineHandle,
  pub render_frontend: Arc<aethervk_core_rlib::gpu::RenderFrontend<'static>>,
  pub render_device_handle: gpu::RenderDeviceHandle,
  pub render_tx: mpsc::Sender<RenderCommand>,
  pub render_thread_handle: Option<Thread>,

  pub entity_map: BTreeMap<u64, EntityId>,
  pub next_entity_id: u64,

  pub root_entity: EntityId,
  pub camera_entity: EntityId,
  pub cursor_entity: EntityId,
  pub sun_entity: EntityId,
  pub grid_entity: EntityId,

  pub outlines_enabled: Arc<AtomicBool>,
  pub asset_path: Option<alloc::string::String>,

  pub window_width: u32,
  pub window_height: u32,
  pub logic_state: RwLock<LogicState>,

  pub model_registry: BTreeMap<u64, String>,
  pub next_model_id: u64,
  pub mesh_cache: Arc<aethervk_core_rlib::scene::AssetCache<simulation::comet::Comet>>,
  pub physics_scene: Arc<RwLock<aethervk_core_rlib::physics::physics_scene::PhysicsScene>>,
  pub thread_pool: Arc<RwLock<aethervk_oshal_rlib::os::pool::ThreadPool>>,
  pub time_info: Arc<RwLock<aethervk_oshal_rlib::os::time::TimeInfo>>,
  pub clear_color: [f32; 4],
}

struct PhysicsRebuildWorkload {
  scene: Arc<aethervk_core_rlib::scene::Scene>,
  physics_scene: Arc<RwLock<aethervk_core_rlib::physics::physics_scene::PhysicsScene>>,
}

impl aethervk_oshal_rlib::os::pool::Workload for PhysicsRebuildWorkload {
  fn execute(&self) {
    let new_physics =
      aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(&self.scene);
    let mut guard = self.physics_scene.write();
    *guard = new_physics;
  }
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

static LOGGER_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setLoggerCallback(
  cb: Option<extern "C" fn(*const c_char)>,
) {
  let ptr = match cb {
    Some(f) => f as *mut (),
    None => core::ptr::null_mut(),
  };
  LOGGER_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
}

static BREADCRUMB_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBreadcrumbCallback(
  cb: Option<extern "C" fn(u32, *const c_char)>,
) {
  let ptr = match cb {
    Some(f) => f as *mut (),
    None => core::ptr::null_mut(),
  };
  BREADCRUMB_CALLBACK.store(ptr, core::sync::atomic::Ordering::Relaxed);
}

fn emit_breadcrumb(status: u32, msg: &str) {
  let fptr = BREADCRUMB_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    if let Ok(c_msg) = alloc::ffi::CString::new(msg) {
      let cb: extern "C" fn(u32, *const c_char) = unsafe { core::mem::transmute(fptr) };
      cb(status, c_msg.as_ptr());
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() || path.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

  {
    let logic = ctx.logic_state.read();
    if logic.almanac_data.file_names.iter().any(|f| f == path_str) {
      return true; // Already loaded
    }
  }

  emit_breadcrumb(0, &format!("Loading almanac file: {}", path_str));

  let mut path_buf = oshal::os::fs::PathBuf::new();
  path_buf.push(path_str);

  if let Ok(data) = oshal::os::fs::read(path_buf.as_ref()) {
    let mut logic = ctx.logic_state.write();
    logic.almanac_data.data.push(data);
    let bytes = bytes::BytesMut::from(logic.almanac_data.data.last().unwrap().as_slice());
    logic.almanac_data.file_names.push(String::from(path_str));
    if let Ok(new_almanac) = logic
      .almanac_data
      .almanac
      .clone()
      .load_from_bytes(bytes, path_str)
    {
      logic.almanac_data.almanac = new_almanac;
      emit_breadcrumb(1, &format!("Successfully loaded: {}", path_str));
      return true;
    } else {
      logic.almanac_data.data.pop();
      logic.almanac_data.file_names.pop();
      emit_breadcrumb(3, &format!("Failed to parse: {}", path_str));
    }
  } else {
    emit_breadcrumb(3, &format!("Failed to read file: {}", path_str));
  }
  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() || path.is_null() {
    return 0;
  }
  let ctx = unsafe { &mut *ctx };
  let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

  emit_breadcrumb(0, &format!("Trying to load GLTF from path: {}", path_str));

  if let Ok(mesh) = simulation::comet::load_comet_from_gltf(path_str, false) {
    emit_breadcrumb(1, &format!("Generating BVH for path: {}", path_str));
    let model_id = ctx.next_model_id;
    ctx.next_model_id += 1;

    // Add to cache
    ctx.mesh_cache.insert(String::from(path_str), mesh);

    // Add to registry
    ctx.model_registry.insert(model_id, String::from(path_str));
    return model_id;
  }

  emit_breadcrumb(3, &format!("Failed to load GLTF from path: {}", path_str));
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_unloadModel(
  ctx: *mut SimulationContext,
  model_id: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  if ctx.model_registry.remove(&model_id).is_some() {
    // NOTE: Cascade removal from the scene of instances is complex without an 'InstanceComponent'.
    // For now, we only remove it from the registry. Full ECS cleanup should happen on user request.
    emit_breadcrumb(1, &format!("Unloaded model {}", model_id));
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnModelInstance(
  ctx: *mut SimulationContext,
  model_id: u64,
  name: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx = unsafe { &mut *ctx };

  if let Some(path_str) = ctx.model_registry.get(&model_id) {
    let name_str = if name.is_null() {
      "ModelInstance"
    } else {
      unsafe { CStr::from_ptr(name).to_str().unwrap_or("ModelInstance") }
    };

    let entity_id = ctx.scene.spawn_entity(name_str);

    let _ = ctx.scene.add_component(
      entity_id,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
    );

    let mesh_arc = if let Some(cached_mesh) = ctx.mesh_cache.get(path_str) {
      cached_mesh
    } else {
      if let Ok(loaded_mesh) = simulation::comet::load_comet_from_gltf(path_str, false) {
        ctx.mesh_cache.insert(path_str.clone(), loaded_mesh)
      } else {
        return 0; // Failed to load mesh
      }
    };

    let _ = ctx.scene.add_component(
      entity_id,
      PhysicalMeshComponent {
        asset_path: path_str.clone(),
        mesh: mesh_arc,
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
      },
    );

    ctx.scene.set_parent(entity_id, Some(ctx.root_entity));
    return ctx.register_entity(entity_id);
  }
  0
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAlmanacLoadedFiles(
  ctx: *mut SimulationContext,
  count: *mut u32,
) -> *mut *mut c_char {
  if ctx.is_null() {
    if !count.is_null() {
      unsafe {
        *count = 0;
      }
    }
    return core::ptr::null_mut();
  }
  let ctx = unsafe { &mut *ctx };
  let logic = ctx.logic_state.read();
  if !count.is_null() {
    unsafe {
      *count = logic.almanac_data.file_names.len() as u32;
    }
  }

  let mut ptrs: Vec<*mut c_char> = logic
    .almanac_data
    .file_names
    .iter()
    .map(|s| alloc::ffi::CString::new(s.as_str()).unwrap().into_raw())
    .collect();
  let ptr = ptrs.as_mut_ptr();
  core::mem::forget(ptrs);
  ptr
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTimeScale(
  ctx: *mut SimulationContext,
  scale: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  let mut logic = ctx.logic_state.write();
  logic.current_scale = match scale {
    1 => TimeScale::OneDay,
    2 => TimeScale::OneWeek,
    3 => TimeScale::OneMonth,
    _ => TimeScale::Stopped,
  };
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTime(
  ctx: *mut SimulationContext,
) -> f64 {
  if ctx.is_null() {
    return 0.0;
  }
  let ctx = unsafe { &mut *ctx };
  let logic = ctx.logic_state.read();
  logic.current_epoch.to_tai_seconds()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTimeUtc(
  ctx: *mut SimulationContext,
  buffer: *mut c_char,
  buffer_len: u32,
) -> bool {
  if ctx.is_null() || buffer.is_null() || buffer_len == 0 {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  let logic = ctx.logic_state.read();
  let utc_str = format!("{}", logic.current_epoch);

  let bytes = utc_str.as_bytes();
  let copy_len = core::cmp::min(bytes.len(), (buffer_len - 1) as usize);

  let dest = unsafe { core::slice::from_raw_parts_mut(buffer as *mut u8, buffer_len as usize) };
  dest[..copy_len].copy_from_slice(&bytes[..copy_len]);
  dest[copy_len] = 0; // null terminator

  true
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setSimulationTime(
  ctx: *mut SimulationContext,
  time_tai: f64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  let mut logic = ctx.logic_state.write();
  logic.current_epoch = anise::time::Epoch::from_tai_seconds(time_tai);
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycastNdc(
  ctx: *mut SimulationContext,
  ndc_x: f32,
  ndc_y: f32,
  out_hit_entity: *mut u64,
  out_px: *mut f32,
  out_py: *mut f32,
  out_pz: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };

  let mut view_proj_inv = Mat4x4f32::identity();
  let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);

  let mut view = Mat4x4f32::identity();
  ctx
    .scene
    .with_component(ctx.camera_entity, |c: &TransformComponent| {
      cam_pos = c.position;
      view = <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, -1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        c.rotation.conjugate(),
      ) * <Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::translation(
        c.position * -1.0,
      );
    });

  ctx
    .scene
    .with_component(ctx.camera_entity, |cam: &CameraComponent| {
      let proj = cam.projection;
      let view_proj = proj * view;
      view_proj_inv = view_proj.inverse().unwrap_or(Mat4x4f32::identity());
    });

  let ndc_near =
    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 0.0, 1.0);
  let ndc_far =
    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(ndc_x, ndc_y, 1.0, 1.0);

  let mut world_near = view_proj_inv.mul_vector(ndc_near);
  let mut world_far = view_proj_inv.mul_vector(ndc_far);

  if world_near.w() != 0.0 {
    world_near = world_near / world_near.w();
  }
  if world_far.w() != 0.0 {
    world_far = world_far / world_far.w();
  }

  let ro = Vec3f32::from_components(world_near.x(), world_near.y(), world_near.z());
  let target = Vec3f32::from_components(world_far.x(), world_far.y(), world_far.z());
  let rd = (target - ro).normalize();

  // Forward to standard raycast
  unsafe {
    avkSimulationContext_raycast(
      ctx,
      ro.x(),
      ro.y(),
      ro.z(),
      rd.x(),
      rd.y(),
      rd.z(),
      out_hit_entity,
      out_px,
      out_py,
      out_pz,
    )
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_raycast(
  ctx: *mut SimulationContext,
  ro_x: f32,
  ro_y: f32,
  ro_z: f32,
  rd_x: f32,
  rd_y: f32,
  rd_z: f32,
  out_hit_entity: *mut u64,
  out_px: *mut f32,
  out_py: *mut f32,
  out_pz: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };

  let ro = Vec3f32::from_components(ro_x, ro_y, ro_z);
  let rd = Vec3f32::from_components(rd_x, rd_y, rd_z).normalize();

  let mut closest_t = core::f32::MAX;
  let mut hit_point = Vec3f32::from_components(0.0, 0.0, 0.0);
  let mut hit_entity = None;

  let ray = aethervk_core_rlib::math::collision::intersection::Ray {
    origin: ro,
    direction: rd,
    length: core::f32::MAX,
  };

  let mut hit_instances = alloc::vec::Vec::new();
  {
    let ps = ctx.physics_scene.read();
    for node in ps.world_bvh.nodes.iter() {
      let hits_instance = match &node.bound {
        aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
          aethervk_core_rlib::math::collision::intersection::intersect_ray_aabb(&ray, aabb)
        }
        aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
          aethervk_core_rlib::math::collision::intersection::intersect_ray_obb::<
            f32,
            Vec3f32,
            aethervk_oshal_rlib::math::matrix::mat3::Mat3f32,
          >(&ray, obb)
        }
      };

      if hits_instance {
        hit_instances.push(ps.entity_mappings[node.left_child_or_primitive_offset as usize]);
      }
    }
  }

  ctx
    .scene
    .query2::<PhysicalMeshComponent, TransformComponent, _>(|entity, mesh, transform| {
      if !hit_instances.contains(&entity) {
        return;
      }

      if let Some(bvh) = &mesh.mesh.bvh {
        let model_matrix = Mat4x4f32::translation(transform.position)
          * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
          * Mat4x4f32::from_scale(transform.scale);

        let inv_model = model_matrix.inverse().unwrap_or(Mat4x4f32::identity());

        let local_ro = inv_model.mul_vector(
          aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
            ro.x(),
            ro.y(),
            ro.z(),
            1.0,
          ),
        );
        let local_rd = inv_model.mul_vector(
          aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
            rd.x(),
            rd.y(),
            rd.z(),
            0.0,
          ),
        );

        let local_ro = Vec3f32::from_components(local_ro.x(), local_ro.y(), local_ro.z());
        let local_rd =
          Vec3f32::from_components(local_rd.x(), local_rd.y(), local_rd.z()).normalize();

        let local_ray = aethervk_core_rlib::math::collision::intersection::Ray {
          origin: local_ro,
          direction: local_rd,
          length: core::f32::MAX,
        };

        let mut stack = alloc::vec::Vec::new();
        if !bvh.nodes.is_empty() {
          stack.push(0);
        }

        while let Some(node_idx) = stack.pop() {
          let local_node = &bvh.nodes[node_idx];

          let hit_local_node = match &local_node.bound {
            aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
              aethervk_core_rlib::math::collision::intersection::intersect_ray_aabb(
                &local_ray, aabb,
              )
            }
            aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
              aethervk_core_rlib::math::collision::intersection::intersect_ray_obb::<
                f32,
                Vec3f32,
                Mat3f32,
              >(&local_ray, obb)
            }
          };

          if hit_local_node {
            if local_node.primitive_count > 0 {
              let prim_start = local_node.left_child_or_primitive_offset as usize;
              let prim_end = prim_start + local_node.primitive_count as usize;
              for j in prim_start..prim_end {
                let tri_idx = bvh.primitives[j];
                let v0 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3] as usize].position;
                let v1 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3 + 1] as usize].position;
                let v2 = mesh.mesh.vertices[mesh.mesh.indices[tri_idx * 3 + 2] as usize].position;

                let v0 = Vec3f32::from_components(v0[0], v0[1], v0[2]);
                let v1 = Vec3f32::from_components(v1[0], v1[1], v1[2]);
                let v2 = Vec3f32::from_components(v2[0], v2[1], v2[2]);

                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let h = local_rd.cross(edge2);
                let a = edge1.dot(h);

                if a > -1e-6 && a < 1e-6 {
                  continue;
                }

                let f = 1.0 / a;
                let s = local_ro - v0;
                let u = f * s.dot(h);
                if u < 0.0 || u > 1.0 {
                  continue;
                }

                let q = s.cross(edge1);
                let v = f * local_rd.dot(q);
                if v < 0.0 || u + v > 1.0 {
                  continue;
                }

                let t = f * edge2.dot(q);
                if t > 1e-5 && t < closest_t {
                  closest_t = t;
                  let local_hit = local_ro + local_rd * t;

                  let global_hit = model_matrix.mul_vector(
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
                      local_hit.x(),
                      local_hit.y(),
                      local_hit.z(),
                      1.0,
                    ),
                  );
                  hit_point =
                    Vec3f32::from_components(global_hit.x(), global_hit.y(), global_hit.z());
                  hit_entity = Some(entity);
                }
              }
            } else {
              if local_node.right_child_offset != u32::MAX {
                stack.push(local_node.right_child_offset as usize);
              }
              if local_node.left_child_or_primitive_offset != u32::MAX {
                stack.push(local_node.left_child_or_primitive_offset as usize);
              }
            }
          }
        }
      }
    });

  if let Some(entity) = hit_entity {
    // Find the external u64 ID
    let mut external_id = 0;
    for (ext_id, internal_id) in &ctx.entity_map {
      if *internal_id == entity {
        external_id = *ext_id;
        break;
      }
    }

    if !out_hit_entity.is_null() {
      unsafe {
        *out_hit_entity = external_id;
      }
    }
    if !out_px.is_null() {
      unsafe {
        *out_px = hit_point.x();
      }
    }
    if !out_py.is_null() {
      unsafe {
        *out_py = hit_point.y();
      }
    }
    if !out_pz.is_null() {
      unsafe {
        *out_pz = hit_point.z();
      }
    }
    return true;
  }

  false
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  backend: *const c_char,
  width: u32,
  height: u32,
) -> *mut SimulationContext {
  let backend_str = if backend.is_null() {
    ""
  } else {
    unsafe { CStr::from_ptr(backend).to_str().unwrap_or("") }
  };

  if backend_str != "Vulkan" {
    return core::ptr::null_mut(); // Unsupported backend
  }

  let width = if width == 0 { 800 } else { width };
  let height = if height == 0 { 600 } else { height };

  let runtime_params = RuntimeParams {
    render_backend_params: FnvIndexMap::new(),
  };

  let frontend = Arc::new(
    gpu::new_render_frontend(gpu::VULKAN_RENDER_BACKEND, &runtime_params).unwrap()
  );
  let additional_params = gpu::DeviceAdditionalParams::new();
  let render_device_handle = frontend.take_mut_and(|context| {
     Ok(context.init_device(0, &additional_params).unwrap())
  }).unwrap().unwrap();

  let mut presentation_engine_result: GpuResult<gpu::PresentationEngineHandle> =
    Err(aethervk_core_rlib::types::GpuError::InvalidState);

  frontend.take_and(|context| {
    context
      .deref_device_and(
        render_device_handle,
        &mut (&gpu::PresentationEngineParams::windowless(width, height), &mut presentation_engine_result) as *mut _ as *mut core::ffi::c_void,
        |device, data| {
          let (params, out) = unsafe {
            &mut *(data as *mut (&gpu::PresentationEngineParams, *mut GpuResult<gpu::PresentationEngineHandle>))
          };
          let pe_res = device.create_presentation_engine(**params);
          if let Ok(pe) = pe_res {
            device.init_archetypes(pe).unwrap();
          }
          unsafe { **out = pe_res };
          Ok(())
        },
      )
      .unwrap();
    Ok(())
  }).unwrap();

  let presentation_engine = match presentation_engine_result {
    Ok(pe) => pe,
    Err(_) => return core::ptr::null_mut(),
  };

  let (render_tx, render_rx) = mpsc::channel(128);

  let scene = Scene::new();
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<PhysicalMeshComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CameraComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<CursorComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SunComponent>(&[TypeId::of::<TransformComponent>()]);
  scene.register_component::<SkyComponent>(&[]);
  scene.register_component::<GridComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::MarkersComponent>(&[TypeId::of::<
    TransformComponent,
  >()]);
  scene.register_component::<aethervk_core_rlib::scene::SelectedComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::FollowingComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::HiddenComponent>(&[]);
  scene.register_component::<aethervk_core_rlib::scene::BvhDebugComponent>(&[]);

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
      position: Vec3f32::from_components(0.0, -400.0, 0.0),
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
      near_plane: 0.1,
      far_plane: 10000.0,
    },
  );
  scene.set_parent(camera_entity, Some(root_entity));

  let sky_entity = scene.spawn_entity("sky");
  let _ = scene.add_component(sky_entity, SkyComponent {});
  scene.set_parent(sky_entity, Some(root_entity));

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

  // Add emissive core for the sun during startup to match C# logic
  let sun_core_id = scene.spawn_entity("sun_core");
  let sun_sphere = aethervk_core_rlib::simulation::comet::generate_uv_sphere(0.45 * 0.95, 64, 64);
  let _ = scene.add_component(
    sun_core_id,
    TransformComponent {
      position: Vec3f32::from_components(0.0, 0.0, 0.0),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(1.0, 1.0, 1.0),
    },
  );
  let _ = scene.add_component(
    sun_core_id,
    PhysicalMeshComponent {
      asset_path: alloc::string::String::new(),
      mesh: alloc::sync::Arc::from(sun_sphere),
      emissive_intensity: 0.9,
      emissive_color: [1.0, 0.35, 0.02],
    },
  );
  scene.set_parent(sun_core_id, Some(sun_entity));

  let grid_entity = scene.spawn_entity("grid");
  let _ = scene.add_component(grid_entity, GridComponent {});
  scene.set_parent(grid_entity, Some(root_entity));

  let physics_scene = Arc::new(RwLock::new(
    aethervk_core_rlib::physics::physics_scene::PhysicsScene::build_from_scene(&scene),
  ));

  let thread_pool = Arc::new(RwLock::new(
    aethervk_oshal_rlib::os::pool::ThreadPool::new(4).unwrap(),
  ));

  let time_info = Arc::new(RwLock::new(
    aethervk_oshal_rlib::os::time::TimeInfo::new(16667, 100000, 1.0),
  ));

  let scene_arc = Arc::new(scene);

  let render_thread_handle = start_render_thread(
    render_rx,
    Arc::clone(&scene_arc),
    Arc::clone(&frontend),
    render_device_handle,
    presentation_engine,
    cursor_entity,
    sun_entity,
  );

  // Wire callbacks for completion tracking
  frontend.take_and(|context| {
    context.deref_device_and(
      render_device_handle,
      &mut Arc::clone(&thread_pool) as *mut _ as *mut core::ffi::c_void,
      |device, data| {
        let pool = unsafe { &*(data as *mut Arc<RwLock<aethervk_oshal_rlib::os::pool::ThreadPool>>) };
        device.wire_callbacks(Arc::clone(pool))
      }
    ).unwrap()
  }).unwrap();

  render_tx.send(RenderCommand::GenerateSky).unwrap();

  let mut ctx = Box::new(SimulationContext {
    scene: scene_arc,
    presentation_engine,
    render_frontend: frontend,
    render_device_handle,
    render_tx,
    render_thread_handle: Some(render_thread_handle),
    entity_map: BTreeMap::new(),
    next_entity_id: 1,
    root_entity,
    camera_entity,
    cursor_entity,
    sun_entity,
    grid_entity,
    outlines_enabled: Arc::new(AtomicBool::new(false)),
    asset_path: None,
    window_width: width,
    window_height: height,
    logic_state: RwLock::new(LogicState::default()),
    model_registry: BTreeMap::new(),
    next_model_id: 1,
    mesh_cache: Arc::new(aethervk_core_rlib::scene::AssetCache::new()),
    physics_scene,
    thread_pool,
    time_info,
    clear_color: [0.0, 0.0, 0.0, 1.0],
  });

  ctx.register_entity(root_entity); // 1
  ctx.register_entity(camera_entity); // 2
  ctx.register_entity(cursor_entity); // 3
  ctx.register_entity(sun_entity); // 4
  ctx.register_entity(sun_core_id); // 5
  ctx.register_entity(grid_entity); // 6

  Box::into_raw(ctx)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if !ctx.is_null() {
    let mut ctx = unsafe { Box::from_raw(ctx) };
    unsafe { avkSimulationContext_stopThreads(&mut *ctx) };
    
    // Explicitly clear caches to drop Arcs and trigger GPU resource cleanup
    ctx.mesh_cache.clear();
    
    // The rest of SimulationContext will be dropped here
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startThreads(ctx: *mut SimulationContext) {
  // Now handled in startup
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_stopThreads(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  let _ = ctx.render_tx.send(RenderCommand::Shutdown);
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
pub unsafe extern "C" fn avkSimulationContext_removeEntity(
  ctx: *mut SimulationContext,
  entity: u64,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    ctx.scene.remove_entity(entity_id);
    ctx.entity_map.remove(&entity);
    true
  } else {
    false
  }
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
        rotation: Quat::from_components(rot_x, rot_y, rot_z, rot_w),
        scale: Vec3f32::from_components(scale_x, scale_y, scale_z),
      },
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setTransformComponent(
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
    ctx
      .scene
      .with_component_mut(entity_id, |c: &mut TransformComponent| {
        c.position = Vec3f32::from_components(pos_x, pos_y, pos_z);
        c.rotation = Quat::from_components(rot_x, rot_y, rot_z, rot_w);
        c.scale = Vec3f32::from_components(scale_x, scale_y, scale_z);
      });
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTransformComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  pos_x: *mut f32,
  pos_y: *mut f32,
  pos_z: *mut f32,
  rot_w: *mut f32,
  rot_x: *mut f32,
  rot_y: *mut f32,
  rot_z: *mut f32,
  scale_x: *mut f32,
  scale_y: *mut f32,
  scale_z: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    if let Some(transform) = ctx.scene.global_transform(entity_id) {
      if !pos_x.is_null() {
        unsafe {
          *pos_x = transform.position.x();
        }
      }
      if !pos_y.is_null() {
        unsafe {
          *pos_y = transform.position.y();
        }
      }
      if !pos_z.is_null() {
        unsafe {
          *pos_z = transform.position.z();
        }
      }
      if !rot_w.is_null() {
        unsafe {
          *rot_w = transform.rotation.scalar_part();
        }
      }
      let v = transform.rotation.vector_part();
      if !rot_x.is_null() {
        unsafe {
          *rot_x = v.x();
        }
      }
      if !rot_y.is_null() {
        unsafe {
          *rot_y = v.y();
        }
      }
      if !rot_z.is_null() {
        unsafe {
          *rot_z = v.z();
        }
      }
      if !scale_x.is_null() {
        unsafe {
          *scale_x = transform.scale.x();
        }
      }
      if !scale_y.is_null() {
        unsafe {
          *scale_y = transform.scale.y();
        }
      }
      if !scale_z.is_null() {
        unsafe {
          *scale_z = transform.scale.z();
        }
      }
      return true;
    }
  }
  false
}
#[repr(C)]
pub struct FfiBvhNode {
  pub node_type: u32, // 0 = AABB, 1 = OBB
  pub min_x: f32,
  pub min_y: f32,
  pub min_z: f32,
  pub max_x: f32,
  pub max_y: f32,
  pub max_z: f32,
  pub center_x: f32,
  pub center_y: f32,
  pub center_z: f32,
  pub extents_x: f32,
  pub extents_y: f32,
  pub extents_z: f32,
  pub left_child: u32,
  pub right_child: u32,
  pub primitive_count: u32,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBvhNodes(
  ctx: *mut SimulationContext,
  entity: u64,
  count: *mut u32,
) -> *mut FfiBvhNode {
  if ctx.is_null() {
    if !count.is_null() {
      unsafe {
        *count = 0;
      }
    }
    return core::ptr::null_mut();
  }
  let ctx = unsafe { &mut *ctx };

  if let Some(entity_id) = ctx.get_entity(entity) {
    let mut ffi_nodes = Vec::new();

    ctx
      .scene
      .with_component(entity_id, |mesh: &PhysicalMeshComponent| {
        if let Some(bvh) = &mesh.mesh.bvh {
          for node in &bvh.nodes {
            let mut ffi_node = FfiBvhNode {
              node_type: 0,
              min_x: 0.0,
              min_y: 0.0,
              min_z: 0.0,
              max_x: 0.0,
              max_y: 0.0,
              max_z: 0.0,
              center_x: 0.0,
              center_y: 0.0,
              center_z: 0.0,
              extents_x: 0.0,
              extents_y: 0.0,
              extents_z: 0.0,
              left_child: node.left_child_or_primitive_offset,
              right_child: node.right_child_offset,
              primitive_count: node.primitive_count,
            };

            match &node.bound {
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::AABB(aabb) => {
                ffi_node.node_type = 0;
                ffi_node.min_x = aabb.min::<Vec3f32>().x();
                ffi_node.min_y = aabb.min::<Vec3f32>().y();
                ffi_node.min_z = aabb.min::<Vec3f32>().z();
                ffi_node.max_x = aabb.max::<Vec3f32>().x();
                ffi_node.max_y = aabb.max::<Vec3f32>().y();
                ffi_node.max_z = aabb.max::<Vec3f32>().z();
              }
              aethervk_core_rlib::math::collision::linear_bvh::LinearBound::OBB(obb) => {
                ffi_node.node_type = 1;
                let t: Vec3f32 = obb.translation();
                let ext: Vec3f32 = obb.half_extent();
                ffi_node.center_x = t.x();
                ffi_node.center_y = t.y();
                ffi_node.center_z = t.z();
                ffi_node.extents_x = ext.x();
                ffi_node.extents_y = ext.y();
                ffi_node.extents_z = ext.z();
              }
            }
            ffi_nodes.push(ffi_node);
          }
        }
      });

    if !count.is_null() {
      unsafe {
        *count = ffi_nodes.len() as u32;
      }
    }

    if ffi_nodes.is_empty() {
      return core::ptr::null_mut();
    }

    let ptr = ffi_nodes.as_mut_ptr();
    core::mem::forget(ffi_nodes);
    return ptr;
  }

  if !count.is_null() {
    unsafe {
      *count = 0;
    }
  }
  core::ptr::null_mut()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(ptr: *mut FfiBvhNode, count: u32) {
  if !ptr.is_null() {
    let _ = unsafe { Vec::from_raw_parts(ptr, count as usize, count as usize) };
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBvhNodeVisibility(
  ctx: *mut SimulationContext,
  entity: u64,
  node_index: u32,
  is_visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(entity) {
    let mut bvh_len = 0;
    ctx.scene.with_component(
      entity_id,
      |mesh: &aethervk_core_rlib::scene::PhysicalMeshComponent| {
        if let Some(bvh) = &mesh.mesh.bvh {
          bvh_len = bvh.nodes.len();
        }
      },
    );

    if (node_index as usize) < bvh_len {
      let mut dbg_opt = None;
      ctx
        .scene
        .with_component(entity_id, |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
          dbg_opt = Some(dbg.node_render_states.clone());
        });

      let mut states = match dbg_opt {
        Some(s) => s,
        None => {
          let mut s = Vec::with_capacity(bvh_len);
          s.resize(bvh_len, false);
          s
        }
      };

      states[node_index as usize] = is_visible;

      let _ = ctx.scene.add_component(
        entity_id,
        aethervk_core_rlib::scene::BvhDebugComponent {
          node_render_states: states,
        },
      );
    }
  }
}

fn collect_render_packet(ctx: &SimulationContext) -> RenderPacket {
  let mut render_items = Vec::new();
  let mut matrix_stack = vec![Mat4x4f32::identity()];

  ctx.scene.traverse_with_hooks(
    ctx.root_entity,
    &mut matrix_stack,
    &mut |stack: &mut Vec<Mat4x4f32>,
          entity: EntityId,
          transform_opt: Option<TransformComponent>,
          mesh_opt: Option<&PhysicalMeshComponent>| {
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
    &mut |stack: &mut Vec<Mat4x4f32>, _| {
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
    far_plane: 10000.0,
  };

  if let Some(global) = ctx.scene.global_transform(ctx.camera_entity) {
    camera_transform = global;
  }
  ctx
    .scene
    .with_component(ctx.camera_entity, |c| camera_component = *c);

  RenderPacket {
    render_items,
    camera_transform,
    camera_component,
    window_width: ctx.window_width,
    window_height: ctx.window_height,
    outlines_enabled: ctx
      .outlines_enabled
      .load(core::sync::atomic::Ordering::Relaxed),
    clear_color: ctx.clear_color,
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_simulationTick(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  // Update time
  {
    let mut time = ctx.time_info.write();
    time.ut_update();
  }

  // Concurrent physics rebuild
  {
    let workload = Box::new(PhysicsRebuildWorkload {
      scene: Arc::clone(&ctx.scene),
      physics_scene: Arc::clone(&ctx.physics_scene),
    });
    let mut pool = ctx.thread_pool.write();
    let _ = pool.scatter(vec![workload]);
    pool.gather(); // Synchronous for now to ensure raycast works immediately after tick, but running in pool
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_renderTick(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx = unsafe { &mut *ctx };

  let mut task_id = 0;
  let _ = ctx.render_frontend.take_and(|context| {
    context.deref_device_and(
      ctx.render_device_handle,
      &mut task_id as *mut _ as *mut core::ffi::c_void,
      |device, data| {
        let tid = unsafe { &mut *(data as *mut u64) };
        *tid = device.create_task();
        Ok(())
      }
    ).unwrap_or(Ok(()))
  });

  let packet = collect_render_packet(ctx);
  let _ = ctx.render_tx.send(RenderCommand::RenderFrame { packet, task_id });
  
  task_id
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getTaskStatus(
  ctx: *mut SimulationContext,
  task_id: u64,
) -> i32 {
  if ctx.is_null() {
    return -1;
  }
  let ctx = unsafe { &mut *ctx };

  let mut status = 0; // 0: Pending, 1: Success, 2: Failed
  let res = ctx.render_frontend.take_and(|context| {
    context.deref_device_and(
      ctx.render_device_handle,
      &mut (task_id, &mut status) as *mut _ as *mut core::ffi::c_void,
      |device, data| {
        let (tid, s) = unsafe { &mut *(data as *mut (u64, &mut i32)) };
        match device.is_task_completed(*tid) {
          Ok(true) => **s = 1,
          Ok(false) => **s = 0,
          Err(_) => **s = 2,
        }
        Ok(())
      }
    ).unwrap_or(Ok(()))
  });

  if res.is_none() {
    return -1;
  }
  status
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

  let _ = ctx.render_tx.send(RenderCommand::Resize { width, height });
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setActiveCamera(
  ctx: *mut SimulationContext,
  camera: u64,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  if let Some(entity_id) = ctx.get_entity(camera) {
    ctx.camera_entity = entity_id;
  }
}

#[repr(C)]
pub struct FfiLogicCommand {
  pub cmd_type: u32,
  pub float_val_1: f32,
  pub float_val_2: f32,
  pub ulong_val: u64,
  pub bool_val: bool,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_processCommand(
  ctx: *mut SimulationContext,
  command: FfiLogicCommand,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };

  let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  let mut cam_rot = Quat::identity();
  ctx
    .scene
    .with_component(ctx.camera_entity, |c: &TransformComponent| {
      cam_pos = c.position;
      cam_rot = c.rotation;
    });

  let mut cursor_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  ctx
    .scene
    .with_component(ctx.cursor_entity, |c: &TransformComponent| {
      cursor_pos = c.position;
    });

  let offset = cam_pos - cursor_pos;
  let mut dist = offset.length();
  if dist < 0.1 {
    dist = 0.1;
  }

  match command.cmd_type {
    0 => {
      // RotateCamera
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let rotation_speed = 0.005;

      let yaw_quat = Quat::from_axis_angle(
        Vec3f32::from_components(0.0, 0.0, 1.0),
        -delta_x * rotation_speed,
      );

      let local_right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let pitch_quat = Quat::from_axis_angle(local_right, -delta_y * rotation_speed);

      let new_rot = (pitch_quat * yaw_quat * cam_rot).normalize();

      let rot_delta = new_rot * cam_rot.conjugate();
      let new_offset = rot_delta.rotate_vector(offset);

      ctx
        .scene
        .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
          c.position = cursor_pos + new_offset;
          c.rotation = new_rot;
        });
    }
    1 => {
      // ZoomCamera
      let amount = command.float_val_1;

      let is_ortho = ctx
        .scene
        .with_component(ctx.camera_entity, |c: &CameraComponent| {
          c.projection.column(3).unwrap().w().abs() > 0.5
        })
        .unwrap_or(false);

      if !is_ortho {
        let zoom_speed = dist * 0.01;
        let forward = cam_rot.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
        let new_pos = cam_pos + forward * (amount * zoom_speed);

        ctx
          .scene
          .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
            c.position = new_pos;
          });
      }
    }
    2 => {
      // ResetCamera
      let ssb = Vec3f32::from_components(0.0, 0.0, 0.0);
      let offset = Vec3f32::from_components(0.0, -400.0, 0.0);
      let yaw = core::f32::consts::PI;
      let pitch = 0.0;
      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();

      ctx
        .scene
        .with_component_mut(ctx.cursor_entity, |c: &mut TransformComponent| {
          c.position = ssb;
        });
      ctx
        .scene
        .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
          c.position = ssb + offset;
          c.rotation = new_rot;
        });
    }
    3 => {
      // PanCursor
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let pan_speed = dist * 0.001;

      let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
      let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

      ctx
        .scene
        .with_component_mut(ctx.cursor_entity, |c: &mut TransformComponent| {
          c.position = c.position + translation;
        });
      ctx
        .scene
        .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
          c.position = c.position + translation;
        });
    }
    4 => {
      // SnapToEntity
      let target_entity_id = ctx.get_entity(command.ulong_val);
      if let Some(target) = target_entity_id {
        let mut t_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
        let mut t_scale = Vec3f32::from_components(1.0, 1.0, 1.0);
        if let Some(t) = ctx.scene.global_transform(target) {
          t_pos = t.position;
          t_scale = t.scale;
        }

        // Distance relative to scale, or default
        let dist = t_scale.x().max(t_scale.y()).max(t_scale.z()) * 3.0;
        let offset = Vec3f32::from_components(0.0, -dist.max(400.0), 0.0);

        let yaw = core::f32::consts::PI;
        let pitch = 0.0;
        let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
        let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), pitch);
        let new_rot = (yaw_quat * pitch_quat).normalize();

        ctx
          .scene
          .with_component_mut(ctx.cursor_entity, |c: &mut TransformComponent| {
            c.position = t_pos;
          });
        ctx
          .scene
          .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
            c.position = t_pos + offset;
            c.rotation = new_rot;
          });
      }
    }
    5 => {
      // FollowEntity
      let target_entity_id = ctx.get_entity(command.ulong_val);
      if let Some(target) = target_entity_id {
        let _ = ctx
          .scene
          .add_component(target, aethervk_core_rlib::scene::FollowingComponent {});
      }
    }
    6 => {
      // UnfollowEntity
      let mut following_entities = Vec::new();
      ctx
        .scene
        .query1::<aethervk_core_rlib::scene::FollowingComponent, _>(|entity, _| {
          following_entities.push(entity);
        });
      for entity in following_entities {
        let _ = ctx
          .scene
          .remove_component::<aethervk_core_rlib::scene::FollowingComponent>(entity);
      }
    }
    7 => {
      // PanCamera (does not move cursor)
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let pan_speed = dist * 0.001;

      let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
      let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

      ctx
        .scene
        .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
          c.position = c.position + translation;
        });
    }
    8 => {
      // MoveCursor
      let axis_x = command.float_val_1;
      let axis_y = command.float_val_2;
      let axis_z = 0.0; // not passed via 2 floats.
      let pan_speed = dist * 0.001;

      let translation = Vec3f32::from_components(axis_x, axis_y, axis_z) * pan_speed;

      ctx
        .scene
        .with_component_mut(ctx.cursor_entity, |c: &mut TransformComponent| {
          c.position = c.position + translation;
        });
      ctx
        .scene
        .with_component_mut(ctx.camera_entity, |c: &mut TransformComponent| {
          c.position = c.position + translation;
        });
    }
    _ => {}
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setClearColor(
  ctx: *mut SimulationContext,
  r: f32,
  g: f32,
  b: f32,
  a: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx = unsafe { &mut *ctx };
  let _ = ctx.render_tx.send(RenderCommand::SetClearColor([r, g, b, a]));
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_downloadImage(
  ctx: *mut SimulationContext,
  buffer_ptr: *mut u8,
  buffer_size: usize,
) -> bool {
  if ctx.is_null() || buffer_ptr.is_null() {
    return false;
  }
  let ctx = unsafe { &mut *ctx };
  
  let mut success = false;
  let done_signal = Arc::new(AtomicBool::new(false));
  
  let _ = ctx.render_tx.send(RenderCommand::DownloadImage {
    buffer: buffer_ptr,
    buffer_size,
    success: &mut success,
    done_signal: Arc::clone(&done_signal),
  });

  while !done_signal.load(core::sync::atomic::Ordering::Acquire) {
    core::hint::spin_loop();
  }

  success
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn aethervk_core_cdylib_log(msg: *const c_char) {
  let fptr =
    aethervk_oshal_rlib::os::debug::LOGGER_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    let cb: extern "C" fn(*const c_char) = unsafe { core::mem::transmute(fptr) };
    cb(msg);
  }
}
