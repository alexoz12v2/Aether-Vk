use aethervk_core_rlib::{
  gpu::{self, RenderDevice},
  scene::{
    CameraComponent, CursorComponent, EntityId, GridComponent, PhysicalMeshComponent,
    RenderableDataRef, Scene, SkyComponent, SunComponent, TransformComponent,
  },
  simulation,
  types::RuntimeParams,
  types::{GpuResult, EngineError, GpuError},
};
use aethervk_oshal_rlib as oshal;
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix, MatrixVectorMul},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, vec4::Quat, Vector, Vector4},
};
use alloc::{boxed::Box, sync::Arc, vec, vec::Vec, collections::BTreeMap, string::String, format};
use alloc::string::ToString;
use heapless::index_map::FnvIndexMap;
use core::{
  any::TypeId,
  ffi::{c_char, CStr},
  sync::{atomic::AtomicBool},
};
use thingbuf::mpsc;
use oshal::os::thread::{self, Thread};
use spin::rwlock::RwLock;
use spin::{RwLockReadGuard, RwLockWriteGuard};
use aethervk_oshal_rlib::math::matrix::{Matrix4, SquareMatrix};
use aethervk_oshal_rlib::math::matrix::mat3::Mat3f32;
use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::os::pool::tasklet::ThreadPoolExt;

pub mod components_api;
pub mod core_api;
pub mod misc_api;
pub mod models_api;
pub mod scene_api;
pub mod time_api;

pub struct AlmanacPackedData {
  pub data: Vec<Vec<u8>>,
  pub file_names: Vec<String>,
  pub almanac: anise::almanac::Almanac,
}

impl Default for AlmanacPackedData {
  fn default() -> Self {
    Self {
      data: Vec::new(),
      file_names: Vec::new(),
      almanac: anise::almanac::Almanac::default(),
    }
  }
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

#[derive(Debug, Clone, Default)]
pub struct RenderItem {
  pub entity_id: EntityId,
  pub model_matrix: Mat4x4f32,
}

#[derive(Debug, Clone, Default)]
pub struct RenderPacket {
  pub render_items: Vec<RenderItem>,
  pub camera_transform: TransformComponent,
  pub camera_component: CameraComponent,
  pub window_width: u32,
  pub window_height: u32,
  pub outlines_enabled: bool,
  pub clear_color: [f32; 4],
}

#[derive(Clone, Copy)]
struct SendPtr<T>(pub *const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

#[derive(Clone, Copy)]
pub struct SendPtrMut<T>(pub *mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

#[derive(Clone)]
pub enum RenderCommand {
  None,
  RenderFrame {
    packet: RenderPacket,
    task_id: u64,
  },
  DownloadImage {
    buffer: SendPtrMut<u8>,
    buffer_size: usize,
    success: SendPtrMut<bool>,
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

impl Default for RenderCommand {
  fn default() -> Self {
    Self::None
  }
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

#[derive(Clone, Default)]
pub struct LogicCommand {
  pub ffi_logic_command: FfiLogicCommand,
  pub active_scene: Option<Arc<RwLock<SceneContext>>>,
}

impl core::fmt::Debug for LogicCommand {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    self.ffi_logic_command.fmt(f)
  }
}

fn start_logic_thread(logic_rx: mpsc::Receiver<LogicCommand>) -> Option<Thread> {
  thread::spawn(move || {
    loop {
      match logic_rx.try_recv() {
        Ok(cmd) => {
          oshal::log!("[Logic thread] received command: {:?}", cmd);
          if cmd.ffi_logic_command.cmd_type == FfiLogicCommandType::Shutdown {
            break;
          }
          if let Some(active_scene) = cmd.active_scene {
            let active_scene = active_scene.read();
            process_command(cmd.ffi_logic_command, &active_scene);
          }
        }
        Err(e) => {
          if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
            break;
          }
          // Avoid pegging CPU if no commands
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
        }
      }
    }
  })
  .ok()
}

fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  scene_shared: Arc<Scene>,
  frontend: Arc<aethervk_core_rlib::gpu::RenderFrontend>,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  cursor_entity: EntityId,
  sun_entity: EntityId,
) -> Option<Thread> {
  thread::spawn(move || {
    let mut clear_color = [0.0, 0.0, 0.0, 1.0];
    loop {
      match render_rx.try_recv() {
        Ok(cmd) => match cmd {
          RenderCommand::RenderFrame {
            mut packet,
            task_id,
          } => {
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
                .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
            });

            if let Some(Err(e)) = res {
              oshal::log!(
                "[RenderThread] task_id={} | Error on render_payload_ffi={}",
                task_id,
                e.to_string()
              );
              // Report failure to task registry
              let _ = frontend.take_and(|context| {
                let _ = context
                  .deref_device_and(
                    render_device_handle,
                    &mut (task_id, e) as *mut _ as *mut core::ffi::c_void,
                    |device, data| {
                      let (tid, err) =
                        unsafe { &*(data as *mut (u64, aethervk_core_rlib::types::EngineError)) };
                      if let aethervk_core_rlib::types::EngineError::Gpu(gpu_err) = err {
                        device.fail_task(*tid, gpu_err.clone());
                      } else {
                        device.fail_task(*tid, aethervk_core_rlib::types::GpuError::InvalidState);
                      }
                      Ok(())
                    },
                  )
                  .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
                  .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from));
                Ok(())
              });
            }
          }
          RenderCommand::DownloadImage {
            buffer,
            buffer_size,
            success,
            done_signal,
          } => {
            let slice = unsafe { core::slice::from_raw_parts_mut(buffer.0, buffer_size) };
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
                .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
                .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
            });
            unsafe { *(success.0) = matches!(res, Some(Ok(()))) };
            done_signal.store(true, core::sync::atomic::Ordering::Release);
          }
          RenderCommand::SetClearColor(color) => {
            clear_color = color;
          }
          RenderCommand::Resize { width, height } => {
            let mut data = (presentation_engine, width, height);
            let _ = frontend.take_and(|context| {
              let _ = context.deref_device_and(
                render_device_handle,
                &mut data as *mut _ as *mut core::ffi::c_void,
                |device, data_ptr| {
                  let (pe, w, h) =
                    unsafe { &mut *(data_ptr as *mut (gpu::PresentationEngineHandle, u32, u32)) };
                  device.resize_presentation_engine(*pe, *w, *h)
                },
              );
              Ok(())
            });
          }
          RenderCommand::GenerateSky => {
            let _ = frontend.take_and(|context| {
              let _ = context.deref_device_and(
                render_device_handle,
                core::ptr::null_mut(),
                |device, _| device.generate_sky(),
              );
              Ok(())
            });
          }
          RenderCommand::Shutdown => break,
          RenderCommand::None => {}
        },
        Err(e) => {
          if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
            break;
          }
          // Avoid pegging CPU if no commands
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
        }
      }
    }
  })
  .ok()
}

fn render_payload_ffi(device: &dyn RenderDevice, data: *mut core::ffi::c_void) -> GpuResult<()> {
  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  if acquire_result.status.needs_resize() {
    // handled via resize command or next frame
    device.success_task(payload.task_id);
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
          let _ = render_scene.add_renderable(
            device,
            entity,
            transform.to_mat4(),
            RenderableDataRef::Cursor(comp),
            payload.presentation_engine,
            "Cursor",
            false,
            [1.0, 1.0, 1.0, 1.0],
          );
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
        let _ = render_scene.add_renderable(
          device,
          entity,
          aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity(),
          RenderableDataRef::Measurement(comp),
          payload.presentation_engine,
          "Measurement",
          false,
          [1.0, 1.0, 1.0, 1.0],
        );
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
        let _ = render_scene.add_renderable(
          device,
          entity,
          model_matrix,
          RenderableDataRef::ImageBillboard(comp),
          payload.presentation_engine,
          "ImageBillboard",
          false,
          [1.0, 1.0, 1.0, 1.0],
        );
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
    let _ = payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        let mut draw_outline = payload.packet.outlines_enabled;
        let mut outline_color = [0.0, 0.0, 0.0, 0.0];

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
          outline_color = [1.0, 1.0, 1.0, 1.0];
        } else if is_following {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 1.0];
        } else if payload.packet.outlines_enabled {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 0.5];
        }

        let _ = render_scene.add_renderable(
          device,
          item.entity_id,
          item.model_matrix,
          RenderableDataRef::PhysicalMesh(mesh),
          payload.presentation_engine,
          &alloc::format!("Comet_{:?}", item.entity_id),
          draw_outline,
          outline_color,
        );
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
  if payload
    .scene
    .with_component(
      payload.sun_entity,
      |_: &aethervk_core_rlib::scene::HiddenComponent| {},
    )
    .is_none()
  {
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
          if payload
            .scene
            .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
            .is_none()
          {
            sky_opt = Some((entity, *comp));
          }
        });

      let mut grid_opt = None;
      payload
        .scene
        .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
          if payload
            .scene
            .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
            .is_none()
          {
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

      device.render_frame(cmd_buffer, &render_scene)?;

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
                  24.0,
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

      let _ = device.present(
        payload.presentation_engine,
        acquire_result.image_index as usize,
        acquire_result.frame_index as usize,
      );
    }
  }

  Ok(())
}

pub struct SceneContext {
  pub scene: Arc<Scene>,
  pub entity_map: BTreeMap<u64, EntityId>,
  pub next_entity_id: u64,
  pub root_entity: EntityId,
  pub active_camera_entity: EntityId,
  pub cursor_entity: EntityId,
  pub sun_entity: EntityId,
  pub grid_entity: EntityId,
  pub outlines_enabled: Arc<AtomicBool>,
  pub physics_scene: Arc<RwLock<aethervk_core_rlib::physics::physics_scene::PhysicsScene>>,
}

impl SceneContext {
  pub fn register_entity(&mut self, id: EntityId) -> u64 {
    let external_id = self.next_entity_id;
    self.next_entity_id += 1;
    self.entity_map.insert(external_id, id);
    external_id
  }

  pub fn get_entity(&self, external_id: u64) -> Option<EntityId> {
    self.entity_map.get(&external_id).copied()
  }
}

pub struct SimulationContext {
  pub scenes: BTreeMap<u64, Arc<RwLock<SceneContext>>>,
  pub active_scene_id: u64,
  pub next_scene_id: u64,

  pub presentation_engine: gpu::PresentationEngineHandle,
  pub window_width: u32,
  pub window_height: u32,

  pub render_frontend: Arc<aethervk_core_rlib::gpu::RenderFrontend>,
  pub render_device_handle: gpu::RenderDeviceHandle,

  pub render_tx: mpsc::Sender<RenderCommand>,
  pub render_thread_handle: Option<Thread>,

  pub logic_tx: mpsc::Sender<LogicCommand>,
  pub logic_thread_handle: Option<Thread>,

  pub asset_path: Option<alloc::string::String>,

  pub logic_state: RwLock<Box<LogicState>>,

  pub model_registry: BTreeMap<u64, String>,
  pub next_model_id: u64,
  pub mesh_cache: Arc<aethervk_core_rlib::scene::AssetCache<simulation::comet::Comet>>,
  pub thread_pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  pub time_info: Arc<RwLock<aethervk_oshal_rlib::os::time::TimeInfo>>,
  pub clear_color: [f32; 4],
}

impl SimulationContext {
  pub fn active_scene_clone(&self) -> Option<Arc<RwLock<SceneContext>>> {
    self.scenes.get(&self.active_scene_id).cloned()
  }

  pub fn active_scene(&self) -> Option<RwLockReadGuard<'_, SceneContext>> {
    self.scenes.get(&self.active_scene_id).map(|l| l.read())
  }

  pub fn active_scene_mut(&mut self) -> Option<RwLockWriteGuard<'_, SceneContext>> {
    self
      .scenes
      .get_mut(&self.active_scene_id)
      .map(|l| l.write())
  }
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

static BREADCRUMB_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

fn emit_breadcrumb(status: u32, msg: &str) {
  let fptr = BREADCRUMB_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
  if !fptr.is_null() {
    if let Ok(c_msg) = alloc::ffi::CString::new(msg) {
      let cb: extern "C" fn(u32, *const c_char) = unsafe { core::mem::transmute(fptr) };
      cb(status, c_msg.as_ptr());
    }
  }
}

#[repr(u32)]
pub enum NodeType {
  AABB = 0,
  OBB = 1,
}

#[repr(C)]
pub struct FfiBvhNode {
  pub node_type: NodeType,
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

fn collect_render_packet(ctx: &SimulationContext) -> Option<RenderPacket> {
  let mut render_items = Vec::new();
  let mut matrix_stack = vec![Mat4x4f32::identity()];

  let active = ctx.active_scene()?;
  active.scene.traverse_with_hooks(
    active.root_entity,
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

      if let Some(parent_transform) = stack.last() {
        let global_transform = *parent_transform * local_transform;

        if mesh_opt.is_some() {
          render_items.push(RenderItem {
            entity_id: entity,
            model_matrix: global_transform,
          });
        }
        stack.push(global_transform);
      }
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

  if let Some(global) = active.scene.global_transform(active.active_camera_entity) {
    camera_transform = global;
  }
  let _ = active
    .scene
    .with_component(active.active_camera_entity, |c| camera_component = *c);

  Some(RenderPacket {
    render_items,
    camera_transform,
    camera_component,
    window_width: ctx.window_width,
    window_height: ctx.window_height,
    outlines_enabled: active
      .outlines_enabled
      .load(core::sync::atomic::Ordering::Relaxed),
    clear_color: ctx.clear_color,
  })
}

#[repr(u32)]
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum FfiLogicCommandType {
  RotateCamera = 0,
  ZoomCamera = 1,
  ResetCamera = 2,
  PanCursor = 3,
  SnapToEntity = 4,
  FollowEntity = 5,
  UnfollowEntity = 6,
  PanCamera = 7,
  MoveCursor = 8,
  #[default]
  Shutdown = 9,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct FfiLogicCommand {
  pub cmd_type: FfiLogicCommandType,
  pub float_val_1: f32,
  pub float_val_2: f32,
  pub ulong_val: u64,
  pub bool_val: bool,
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setLoggerCallback(
  cb: Option<extern "C" fn(*const c_char)>,
) {
  SimulationContext::set_logger_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setBreadcrumbCallback(
  cb: Option<extern "C" fn(u32, *const c_char)>,
) {
  SimulationContext::set_breadcrumb_callback(cb)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_loadAlmanacFile(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.load_almanac_file(path) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_loadAlmanacFile failed: {}",
        e.to_string()
      );
      emit_breadcrumb(1, &alloc::format!("Almanac load failed: {}", e.to_string()));
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_importModel(
  ctx: *mut SimulationContext,
  path: *const c_char,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.import_model(path) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_importModel failed: {}", e.to_string());
      emit_breadcrumb(1, &alloc::format!("Model import failed: {}", e.to_string()));
      0
    }
  }
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.unload_model(model_id)
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.spawn_model_instance(model_id, name) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_spawnModelInstance failed: {}",
        e.to_string()
      );
      emit_breadcrumb(1, &alloc::format!("Model spawn failed: {}", e.to_string()));
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getAlmanacLoadedFiles(
  ctx: *mut SimulationContext,
  count: *mut u32,
) -> *mut *mut c_char {
  if ctx.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_almanac_loaded_files(count)
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_time_scale(scale)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTime(
  ctx: *mut SimulationContext,
) -> f64 {
  if ctx.is_null() {
    return 0.0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_simulation_time()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getSimulationTimeUtc(
  ctx: *mut SimulationContext,
  buffer: *mut c_char,
  buffer_len: u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_simulation_time_utc(buffer, buffer_len)
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_simulation_time(time_tai)
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.raycast_ndc(ndc_x, ndc_y, out_hit_entity, out_px, out_py, out_pz) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_raycastNdc failed: {}", e.to_string());
      false
    }
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.raycast(
    ro_x,
    ro_y,
    ro_z,
    rd_x,
    rd_y,
    rd_z,
    out_hit_entity,
    out_px,
    out_py,
    out_pz,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_raycast failed: {}", e.to_string());
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startup(
  backend: *const c_char,
  width: u32,
  height: u32,
) -> *mut SimulationContext {
  match SimulationContext::startup(backend, width, height) {
    Ok(ctx) => ctx,
    Err(e) => {
      oshal::log!("avkSimulationContext_startup failed: {}", e.to_string());
      emit_breadcrumb(1, &alloc::format!("Startup failed: {}", e.to_string()));
      core::ptr::null_mut()
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_shutdown(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.shutdown();
  let _ = unsafe { Box::from_raw(ctx) };
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_startThreads(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.start_threads()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_stopThreads(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.stop_threads()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_spawnProceduralSphere(
  ctx: *mut SimulationContext,
  name: *const c_char,
  radius: f32,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.spawn_procedural_sphere(name, radius) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_spawnProceduralSphere failed: {}",
        e.to_string()
      );
      0
    }
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.spawn_entity(name) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_spawnEntity failed: {}", e.to_string());
      0
    }
  }
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.remove_entity(entity) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_removeEntity failed: {}",
        e.to_string()
      );
      false
    }
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_parent(entity, parent) {
    oshal::log!("avkSimulationContext_setParent failed: {}", e.to_string());
  }
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    oshal::log!(
      "avkSimulationContext_addTransformComponent failed: {}",
      e.to_string()
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    oshal::log!(
      "avkSimulationContext_setTransformComponent failed: {}",
      e.to_string()
    );
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
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_transform_component(
    entity, pos_x, pos_y, pos_z, rot_w, rot_x, rot_y, rot_z, scale_x, scale_y, scale_z,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_getTransformComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getBvhNodes(
  ctx: *mut SimulationContext,
  entity: u64,
  count: *mut u32,
) -> *mut FfiBvhNode {
  if ctx.is_null() {
    return core::ptr::null_mut();
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_bvh_nodes(entity, count) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!("avkSimulationContext_getBvhNodes failed: {}", e.to_string());
      core::ptr::null_mut()
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_freeBvhNodes(ptr: *mut FfiBvhNode, count: u32) {
  SimulationContext::free_bvh_nodes(ptr, count)
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_bvh_node_visibility(entity, node_index, is_visible) {
    oshal::log!(
      "avkSimulationContext_setBvhNodeVisibility failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_simulationTick(ctx: *mut SimulationContext) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.simulation_tick()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_renderTick(ctx: *mut SimulationContext) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.render_tick()
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_task_status(task_id)
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.resize(width, height) {
    oshal::log!("avkSimulationContext_resize failed: {}", e.to_string());
  }
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_active_camera(camera) {
    oshal::log!(
      "avkSimulationContext_setActiveCamera failed: {}",
      e.to_string()
    );
  }
}

// TODO error handling
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_processCommand(
  ctx: *mut SimulationContext,
  command: FfiLogicCommand,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  let active_scene = ctx_ref.active_scene_clone().unwrap();
  let command = LogicCommand {
    ffi_logic_command: command,
    active_scene: Some(active_scene),
  };
  ctx_ref.logic_tx.try_send(command).unwrap();
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
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.set_clear_color(r, g, b, a)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_downloadImage(
  ctx: *mut SimulationContext,
  buffer_ptr: *mut u8,
  buffer_size: usize,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.download_image(buffer_ptr, buffer_size)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setAssetPath(path: *const c_char) {
  SimulationContext::set_asset_path(path)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn aethervk_core_cdylib_log(msg: *const c_char) {
  SimulationContext::log(msg)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityCount(ctx: *mut SimulationContext) -> u32 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_count()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityIds(
  ctx: *mut SimulationContext,
  out_ids: *mut u64,
  max_count: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_ids(out_ids, max_count)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityName(
  ctx: *mut SimulationContext,
  entity: u64,
  out_name: *mut c_char,
  max_len: u32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_name(entity, out_name, max_len)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getEntityParent(
  ctx: *mut SimulationContext,
  entity: u64,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  ctx_ref.get_entity_parent(entity)
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_createDefaultScene(
  ctx: *mut SimulationContext,
) -> u64 {
  if ctx.is_null() {
    return 0;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.create_default_scene() {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_createDefaultScene failed: {}",
        e.to_string()
      );
      0
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityVisibility(
  ctx: *mut SimulationContext,
  entity: u64,
  visible: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_visibility(entity, visible) {
    oshal::log!(
      "avkSimulationContext_setEntityVisibility failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntitySelected(
  ctx: *mut SimulationContext,
  entity: u64,
  selected: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_selected(entity, selected) {
    oshal::log!(
      "avkSimulationContext_setEntitySelected failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setEntityFollowing(
  ctx: *mut SimulationContext,
  entity: u64,
  following: bool,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_entity_following(entity, following) {
    oshal::log!(
      "avkSimulationContext_setEntityFollowing failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  fov: f32,
  aspect_ratio: f32,
  near_plane: f32,
  far_plane: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_camera_component(entity, fov, aspect_ratio, near_plane, far_plane) {
    oshal::log!(
      "avkSimulationContext_addCameraComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  is_orthographic: bool,
  fov: f32,
  aspect_ratio: f32,
  near_plane: f32,
  far_plane: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_camera_component(
    entity,
    is_orthographic,
    fov,
    aspect_ratio,
    near_plane,
    far_plane,
  ) {
    oshal::log!(
      "avkSimulationContext_setCameraComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_getCameraComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  proj_out: *mut f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.get_camera_component(entity, proj_out) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_getCameraComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addPhysicalMeshComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  gltf_path: *const c_char,
  emissive_intensity: f32,
  emissive_color_r: f32,
  emissive_color_g: f32,
  emissive_color_b: f32,
) -> bool {
  if ctx.is_null() {
    return false;
  }
  let ctx_ref = unsafe { &mut *ctx };
  match ctx_ref.add_physical_mesh_component(
    entity,
    gltf_path,
    emissive_intensity,
    emissive_color_r,
    emissive_color_g,
    emissive_color_b,
  ) {
    Ok(res) => res,
    Err(e) => {
      oshal::log!(
        "avkSimulationContext_addPhysicalMeshComponent failed: {}",
        e.to_string()
      );
      false
    }
  }
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_sky_component(entity) {
    oshal::log!(
      "avkSimulationContext_addSkyComponent failed: {}",
      e.to_string()
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_cursor_component(entity) {
    oshal::log!(
      "avkSimulationContext_addCursorComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addSunComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  resolution_x: u32,
  resolution_y: u32,
  resolution_z: u32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_sun_component(entity, resolution_x, resolution_y, resolution_z) {
    oshal::log!(
      "avkSimulationContext_addSunComponent failed: {}",
      e.to_string()
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
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_grid_component(entity) {
    oshal::log!(
      "avkSimulationContext_addGridComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addMeasurementComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  p1_x: f32,
  p1_y: f32,
  p1_z: f32,
  p2_x: f32,
  p2_y: f32,
  p2_z: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_measurement_component(entity, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z) {
    oshal::log!(
      "avkSimulationContext_addMeasurementComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_addImageBillboardComponent(
  ctx: *mut SimulationContext,
  entity: u64,
  is_screen_space: bool,
  width: f32,
  height: f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.add_image_billboard_component(entity, is_screen_space, width, height) {
    oshal::log!(
      "avkSimulationContext_addImageBillboardComponent failed: {}",
      e.to_string()
    );
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSimulationContext_setMarkers(
  ctx: *mut SimulationContext,
  entity: u64,
  count: u32,
  px: *const f32,
  py: *const f32,
  pz: *const f32,
  cr: *const f32,
  cg: *const f32,
  cb: *const f32,
  sizes: *const f32,
) {
  if ctx.is_null() {
    return;
  }
  let ctx_ref = unsafe { &mut *ctx };
  if let Err(e) = ctx_ref.set_markers(entity, count, px, py, pz, cr, cg, cb, sizes) {
    oshal::log!("avkSimulationContext_setMarkers failed: {}", e.to_string());
  }
}

pub fn process_command(command: FfiLogicCommand, active_scene: &SceneContext) {
  let mut cam_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  let mut cam_rot = Quat::identity();
  let _ = active_scene.scene.with_component(
    active_scene.active_camera_entity,
    |c: &TransformComponent| {
      cam_pos = c.position;
      cam_rot = c.rotation;
    },
  );

  let mut cursor_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
  let _ =
    active_scene
      .scene
      .with_component(active_scene.cursor_entity, |c: &TransformComponent| {
        cursor_pos = c.position;
      });

  let offset = cam_pos - cursor_pos;
  let mut dist = offset.length();
  if dist < 0.1 {
    dist = 0.1;
  }

  match command.cmd_type {
    FfiLogicCommandType::RotateCamera => {
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let rotation_speed = 0.005;

      let yaw_quat = Quat::from_axis_angle(
        Vec3f32::from_components(0.0, 0.0, 1.0),
        -delta_x * rotation_speed,
      );

      let local_right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let pitch_quat = Quat::from_axis_angle(local_right, -delta_y * rotation_speed);

      let combined = pitch_quat * yaw_quat * cam_rot;
      let len_sq = combined.0.dot(combined.0);
      if len_sq < 1e-6 {
        return;
      }
      let new_rot = combined.normalize();

      let rot_delta = new_rot * cam_rot.conjugate();
      let new_offset = rot_delta.rotate_vector(offset);

      {
        let target_entity = active_scene.active_camera_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = cursor_pos + new_offset;
              c.rotation = new_rot;
            });
      }
    }
    FfiLogicCommandType::ZoomCamera => {
      let amount = command.float_val_1;

      let is_ortho = active_scene
        .scene
        .with_component(active_scene.active_camera_entity, |c: &CameraComponent| {
          if let Some(col3) = c.projection.column(3) {
            col3.w().abs() > 0.5
          } else {
            false
          }
        })
        .unwrap_or(false);

      if !is_ortho {
        let zoom_speed = dist * 0.01;
        let forward = cam_rot.rotate_vector(Vec3f32::from_components(0.0, -1.0, 0.0));
        let new_pos = cam_pos + forward * (amount * zoom_speed);

        {
          let target_entity = active_scene.active_camera_entity;
          let _ =
            active_scene
              .scene
              .with_component_mut(target_entity, |c: &mut TransformComponent| {
                c.position = new_pos;
              });
        }
      }
    }
    FfiLogicCommandType::ResetCamera => {
      let ssb = Vec3f32::from_components(0.0, 0.0, 0.0);
      let offset = Vec3f32::from_components(0.0, -400.0, 0.0);
      let yaw = core::f32::consts::PI;
      let pitch = 0.0;
      let yaw_quat = Quat::from_axis_angle(Vec3f32::from_components(0.0, 0.0, 1.0), yaw);
      let pitch_quat = Quat::from_axis_angle(Vec3f32::from_components(1.0, 0.0, 0.0), pitch);
      let new_rot = (yaw_quat * pitch_quat).normalize();

      {
        let target_entity = active_scene.cursor_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = ssb;
            });
      }
      {
        let target_entity = active_scene.active_camera_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = ssb + offset;
              c.rotation = new_rot;
            });
      }
    }
    FfiLogicCommandType::PanCursor => {
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let pan_speed = dist * 0.001;

      let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
      let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

      {
        let target_entity = active_scene.cursor_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = c.position + translation;
            });
      }
      {
        let target_entity = active_scene.active_camera_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = c.position + translation;
            });
      }
    }
    FfiLogicCommandType::SnapToEntity => {
      let target_entity_id = active_scene.get_entity(command.ulong_val);
      if let Some(target) = target_entity_id {
        let mut t_pos = Vec3f32::from_components(0.0, 0.0, 0.0);
        let mut t_scale = Vec3f32::from_components(1.0, 1.0, 1.0);
        if let Some(t) = active_scene.scene.global_transform(target) {
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

        {
          let target_entity = active_scene.cursor_entity;
          let _ =
            active_scene
              .scene
              .with_component_mut(target_entity, |c: &mut TransformComponent| {
                c.position = t_pos;
              });
        }
        {
          let target_entity = active_scene.active_camera_entity;
          let _ =
            active_scene
              .scene
              .with_component_mut(target_entity, |c: &mut TransformComponent| {
                c.position = t_pos + offset;
                c.rotation = new_rot;
              });
        }
      }
    }
    FfiLogicCommandType::FollowEntity => {
      let target_entity_id = active_scene.get_entity(command.ulong_val);
      if let Some(target) = target_entity_id {
        let _ = active_scene
          .scene
          .add_component(target, aethervk_core_rlib::scene::FollowingComponent {});
      }
    }
    FfiLogicCommandType::UnfollowEntity => {
      let mut following_entities = Vec::new();
      active_scene
        .scene
        .query1::<aethervk_core_rlib::scene::FollowingComponent, _>(|entity, _| {
          following_entities.push(entity);
        });
      for entity in following_entities {
        let _ = active_scene
          .scene
          .remove_component::<aethervk_core_rlib::scene::FollowingComponent>(entity);
      }
    }
    FfiLogicCommandType::PanCamera => {
      // PanCamera (does not move cursor)
      let delta_x = command.float_val_1;
      let delta_y = command.float_val_2;
      let pan_speed = dist * 0.001;

      let right = cam_rot.rotate_vector(Vec3f32::from_components(1.0, 0.0, 0.0));
      let up = cam_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
      let translation = right * (-delta_x * pan_speed) + up * (delta_y * pan_speed);

      {
        let target_entity = active_scene.active_camera_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = c.position + translation;
            });
      }
    }
    FfiLogicCommandType::MoveCursor => {
      let axis_x = command.float_val_1;
      let axis_y = command.float_val_2;
      let axis_z = 0.0; // not passed via 2 floats.
      let pan_speed = dist * 0.001;

      let translation = Vec3f32::from_components(axis_x, axis_y, axis_z) * pan_speed;

      {
        let target_entity = active_scene.cursor_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = c.position + translation;
            });
      }
      {
        let target_entity = active_scene.active_camera_entity;
        let _ =
          active_scene
            .scene
            .with_component_mut(target_entity, |c: &mut TransformComponent| {
              c.position = c.position + translation;
            });
      }
    }
    _ => {}
  }
}
