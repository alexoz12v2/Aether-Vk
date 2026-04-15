use aethervk_core_rlib::{
  gpu::{
    self, RenderDevice,
    frame::{self, RenderPath},
  },
  scene::{
    CameraComponent, CursorComponent, EntityId, PhysicalMeshComponent, RenderableDataRef, Scene,
    TransformComponent,
  },
  types::GpuResult,
};
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use std::sync::{Arc, RwLock, mpsc};

pub struct RenderItem {
  pub entity_id: EntityId,
  pub model_matrix: Mat4x4f32,
}

pub struct RenderPacket {
  pub render_items: Vec<RenderItem>,
  pub camera_transform: TransformComponent,
  pub camera_component: CameraComponent,
  pub window_size: winit::dpi::PhysicalSize<u32>,
}

#[repr(C)]
struct RenderPayloadData<'a> {
  packet: &'a mut RenderPacket,
  presentation_engine: gpu::PresentationEngineHandle,
  scene: &'a Scene,
  cursor_entity: EntityId,
  sun_entity: EntityId,
  // TODO sky_entity
}

pub fn start_render_thread(
  render_rx: mpsc::Receiver<RenderPacket>,
  scene_shared: Arc<Scene>,
  render_frontend: Arc<RwLock<aethervk_core_rlib::gpu::RenderFrontend<'static>>>,
  render_device_handle: gpu::RenderDeviceHandle,
  presentation_engine: gpu::PresentationEngineHandle,
  cursor_entity: EntityId,
  sun_entity: EntityId,
) -> std::thread::JoinHandle<()> {
  std::thread::spawn(move || {
    for mut packet in render_rx {
      let scene_guard = scene_shared.as_ref();
      let mut c_payload = RenderPayloadData {
        packet: &mut packet,
        presentation_engine,
        scene: &scene_guard,
        cursor_entity,
        sun_entity,
      };

      let res = render_frontend.write().unwrap().take_and(|context| {
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
  })
}

fn render_payload_ffi(device: &dyn RenderDevice, data: *mut core::ffi::c_void) -> GpuResult<()> {
  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
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
            RenderableDataRef::PhysicalMesh(mesh),
            payload.presentation_engine,
          )
          .unwrap();
        Ok(())
      },
    );
  }

  payload.scene.with_component(
    payload.cursor_entity,
    |cursor: &CursorComponent| -> GpuResult<()> {
      let t = payload
        .scene
        .global_transform(payload.cursor_entity)
        .unwrap();
      frame
        .add_renderable(
          device,
          payload.cursor_entity,
          t.to_mat4(),
          RenderableDataRef::Cursor(cursor),
          payload.presentation_engine,
        )
        .unwrap();
      Ok(())
    },
  );

  let mut sun_opt = None;
  payload.scene.with_component(
    payload.sun_entity,
    |sun_comp: &aethervk_core_rlib::scene::SunComponent| {
      sun_opt = Some(*sun_comp);
    },
  );
  let sun_comp = sun_opt.unwrap();
  let sun_transform = payload
    .scene
    .global_transform(payload.sun_entity)
    .unwrap();

  let mut sky_opt = None;
  payload.scene.query1::<aethervk_core_rlib::scene::SkyComponent, _>(|id, comp| {
    sky_opt = Some((id, *comp));
  });

  let mut grid_opt = None;
  payload.scene.query1::<aethervk_core_rlib::scene::GridComponent, _>(|id, comp| {
    grid_opt = Some((id, *comp));
  });

  let render_path = frame::ForwardRenderPath;
  render_path.record_commands(
    device,
    (
      &payload.packet.camera_transform,
      &payload.packet.camera_component,
    ),
    Some((payload.sun_entity, &sun_comp, &sun_transform.into())),
    sky_opt.as_ref().map(|(id, comp)| (*id, comp)),
    grid_opt.as_ref().map(|(id, comp)| (*id, comp)),
    &frame,
    payload.presentation_engine,
    &acquire_result,
  )?;

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
