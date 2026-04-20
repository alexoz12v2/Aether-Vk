use crate::gpu::{
  AcquireResult, GpuResourceHandle, PipelineKey, PresentationEngineHandle, RenderDevice,
};
use crate::scene::{
  CameraComponent, EntityId, RenderableDataRef, TransformComponent, SunComponent, SkyComponent,
  GridComponent,
};
use crate::simulation::comet::{PushConstants, TextureFlags};
use crate::types::{GpuError, GpuResult};
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix4, SquareMatrix, MatrixVectorMul, Matrix},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, Vector3, Vector4, Vector},
};
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq)]
pub struct ResourceUploadResult {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  pub outline_pipeline: Option<PipelineKey>,
  /// The vertex buffer to bind.
  pub buffers: GpuResourceHandle,
  pub texture_flags: TextureFlags,
  pub emissive_intensity: f32,
  pub emissive_color: [f32; 3],
}

/// Represents a single draw call with all necessary information.
#[derive(Clone)]
pub struct DrawCall {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  pub outline_pipeline: Option<PipelineKey>,
  /// The vertex buffer to bind.
  pub buffers: GpuResourceHandle,
  /// index count
  pub index_count: u32,
  /// The model matrix of the object to draw.
  pub model_matrix: Mat4x4f32,
  pub texture_flags: TextureFlags,
  pub emissive_intensity: f32,
  pub emissive_color: [f32; 3],
  pub draw_outline: bool,
  pub outline_color: [f32; 4],
}

impl DrawCall {
  pub(crate) fn from_handles_and_matrix(
    result: ResourceUploadResult,
    index_count: u32,
    model_matrix: Mat4x4f32,
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      outline_pipeline: result.outline_pipeline,
      buffers: result.buffers,
      index_count,
      model_matrix,
      texture_flags: result.texture_flags,
      emissive_intensity: result.emissive_intensity,
      emissive_color: result.emissive_color,
      draw_outline: false,
      outline_color: [1.0, 1.0, 1.0, 1.0],
    }
  }
}

/// Represents a draw call for a cursor.
#[derive(Clone)]
pub struct CursorDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
}

impl CursorDrawCall {
  pub(crate) fn from_result_and_matrix(
    result: ResourceUploadResult,
    vertex_count: u32,
    model_matrix: Mat4x4f32,
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      vertex_count,
      model_matrix,
    }
  }
}

/// Represents a draw call for a marker.
#[derive(Clone)]
pub struct MarkerDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub local_pos: [f32; 3],
  pub size: f32,
  pub color: [f32; 3],
}

pub struct MeasurementDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub p1: [f32; 3],
  pub p2: [f32; 3],
  pub distance: f32,
}

pub struct BillboardDrawCall {
  pub pipeline: PipelineKey,
  pub vertex_count: u32,
  pub model_matrix: Mat4x4f32,
  pub texture_id: u64,
  pub billboard_type: crate::scene::BillboardType,
}

/// A proper clear-cut struct representing the rendering scene.
pub struct RenderScene {
  /// A list of draw calls to be executed for this frame.
  pub draw_calls: Vec<DrawCall>,
  /// A list of cursor draw calls.
  pub cursor_calls: Vec<CursorDrawCall>,
  /// A list of marker draw calls.
  pub marker_calls: Vec<MarkerDrawCall>,
  /// A list of measurement draw calls.
  pub measurement_calls: Vec<MeasurementDrawCall>,
  /// A list of billboard draw calls.
  pub billboard_calls: Vec<BillboardDrawCall>,
  pub camera: (TransformComponent, CameraComponent),
  pub sun: Option<(EntityId, SunComponent, TransformComponent)>,
  pub sky: Option<(EntityId, SkyComponent)>,
  pub grid: Option<(EntityId, GridComponent)>,
}

impl RenderScene {
  pub fn new(camera: (TransformComponent, CameraComponent)) -> Self {
    Self {
      draw_calls: Vec::new(),
      cursor_calls: Vec::new(),
      marker_calls: Vec::new(),
      measurement_calls: Vec::new(),
      billboard_calls: Vec::new(),
      camera,
      sun: None,
      sky: None,
      grid: None,
    }
  }

  /// Registers a renderable entity to be drawn in this frame.
  pub fn add_renderable(
    &mut self,
    device: &dyn RenderDevice,
    entity_id: EntityId,
    model_matrix: Mat4x4f32,
    renderable: RenderableDataRef,
    presentation_engine_handle: PresentationEngineHandle,
    debug_name: &str,
    draw_outline: bool,
    outline_color: [f32; 4],
  ) -> GpuResult<()> {
    match renderable {
      RenderableDataRef::ImageBillboard(component) => {
        let res: ResourceUploadResult =
          device.get_or_create_billboard_resources(presentation_engine_handle)?;
        self.billboard_calls.push(BillboardDrawCall {
          pipeline: res.pipeline,
          vertex_count: 4,
          model_matrix,
          texture_id: component.texture_id,
          billboard_type: component.billboard_type,
        });
      }
      RenderableDataRef::PhysicalMesh(component) => {
        let res: ResourceUploadResult = device.get_or_create_physical_mesh_resources(
          entity_id,
          &component,
          presentation_engine_handle,
          debug_name,
        )?;
        let index_count = component.mesh.indices.len() as u32;
        let mut dc = DrawCall::from_handles_and_matrix(res, index_count, model_matrix);
        dc.draw_outline = draw_outline;
        dc.outline_color = outline_color;
        self.draw_calls.push(dc);
      }
      RenderableDataRef::Cursor(_) => {
        let res: ResourceUploadResult =
          device.get_or_create_cursor_resources(presentation_engine_handle)?;
        self
          .cursor_calls
          .push(CursorDrawCall::from_result_and_matrix(res, 4, model_matrix));
      }
      RenderableDataRef::Markers(component) => {
        let res: ResourceUploadResult =
          device.get_or_create_marker_resources(presentation_engine_handle)?;
        for marker in &component.markers {
          self.marker_calls.push(MarkerDrawCall {
            pipeline: res.pipeline,
            vertex_count: 4,
            model_matrix,
            local_pos: marker.local_pos,
            size: marker.size,
            color: marker.color,
          });
        }
      }
      RenderableDataRef::Measurement(component) => {
        let res: ResourceUploadResult =
          device.get_or_create_measurement_resources(presentation_engine_handle)?;
        self.measurement_calls.push(MeasurementDrawCall {
          pipeline: res.pipeline,
          vertex_count: 6,
          p1: [component.pos1.x(), component.pos1.y(), component.pos1.z()],
          p2: [component.pos2.x(), component.pos2.y(), component.pos2.z()],
          distance: (component.pos2 - component.pos1).length(),
        });
      }
    }
    Ok(())
  }
}
pub fn do_draw_cursor(
  device: &dyn RenderDevice,
  view: Mat4x4f32,
  view_proj: Mat4x4f32,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &CursorDrawCall,
) -> Result<(), crate::types::GpuError> {
  // 2. Bind pipeline
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let push_constants = crate::gpu::CursorPushConstants {
    view: view.into(),
    view_proj: view_proj.into(),
    model: draw_call.model_matrix.into(),
    cursor_size: 0.05, // TODO extract from draw call
  };

  device.push_cursor_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

pub fn do_draw_marker(
  device: &dyn RenderDevice,
  view: Mat4x4f32,
  view_proj: Mat4x4f32,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &MarkerDrawCall,
) -> Result<(), crate::types::GpuError> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let mut camera_up = [0.0; 3];
  let mut camera_right = [0.0; 3];

  if let Some(inv_view) = view.inverse() {
    let up: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(1).unwrap();
    let right: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(0).unwrap();
    camera_up = [up.x(), up.y(), up.z()];
    camera_right = [right.x(), right.y(), right.z()];
  }

  let global_center = draw_call.model_matrix.mul_vector(
    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
      draw_call.local_pos[0], draw_call.local_pos[1], draw_call.local_pos[2], 1.0
    )
  );

  let push_constants = crate::gpu::MarkerPushConstants {
    view_proj: view_proj.into(),
    center_pos: [global_center.x(), global_center.y(), global_center.z()],
    size: draw_call.size,
    color: draw_call.color,
    _pad0: 0.0,
    camera_up,
    _pad1: 0.0,
    camera_right,
    _pad2: 0.0,
  };
  device.push_marker_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

pub fn do_draw_measurement(
  device: &dyn RenderDevice,
  view: Mat4x4f32,
  view_proj: Mat4x4f32,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &MeasurementDrawCall,
) -> Result<(), crate::types::GpuError> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let mut camera_up = [0.0; 3];
  let mut camera_right = [0.0; 3];

  if let Some(inv_view) = view.inverse() {
    let up: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(1).unwrap();
    let right: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(0).unwrap();
    camera_up = [up.x(), up.y(), up.z()];
    camera_right = [right.x(), right.y(), right.z()];
  }

  let push_constants = crate::gpu::MeasurementPushConstants {
    view_proj: view_proj.into(),
    p1: draw_call.p1,
    _pad0: 0.0,
    p2: draw_call.p2,
    _pad1: 0.0,
    camera_up,
    _pad2: 0.0,
    camera_right,
    _pad3: 0.0,
    color: [1.0, 1.0, 1.0], // White
    _pad4: 0.0,
  };
  device.push_measurement_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

pub fn do_draw_billboard(
  device: &dyn RenderDevice,
  view: Mat4x4f32,
  view_proj: Mat4x4f32,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &BillboardDrawCall,
) -> Result<(), crate::types::GpuError> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;

  let mut camera_up = [0.0; 3];
  let mut camera_right = [0.0; 3];

  if let Some(inv_view) = view.inverse() {
    let up: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(1).unwrap();
    let right: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = inv_view.column(0).unwrap();
    camera_up = [up.x(), up.y(), up.z()];
    camera_right = [right.x(), right.y(), right.z()];
  }

  let center_pos: aethervk_oshal_rlib::math::vector::vec4::Vec4f32 = draw_call.model_matrix.column(3).unwrap();

  let (size, is_screen_space) = match draw_call.billboard_type {
    crate::scene::BillboardType::WorldSpace { width, height } => ([width, height], 0),
    crate::scene::BillboardType::ScreenSpace { pct_width, pct_height } => ([pct_width, pct_height], 1),
  };

  let push_constants = crate::gpu::BillboardPushConstants {
    view_proj: view_proj.into(),
    center_pos: [center_pos.x(), center_pos.y(), center_pos.z()],
    _pad0: 0.0,
    camera_up,
    _pad1: 0.0,
    camera_right,
    _pad2: 0.0,
    size,
    is_screen_space,
    texture_id: draw_call.texture_id as u32,
  };
  device.push_billboard_constants(cmd_buffer, &push_constants)?;
  device.draw(cmd_buffer, draw_call.vertex_count)?;

  Ok(())
}

pub fn do_draw_call(
  device: &dyn RenderDevice,
  view_proj: Mat4x4f32,
  camera_pos: Vec3f32,
  sun_pos: Vec3f32,
  sun_color: [f32; 4],
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &DrawCall,
) -> Result<(), crate::types::GpuError> {
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.bind_buffers(cmd_buffer, draw_call.pipeline, draw_call.buffers)?;

  let model = draw_call.model_matrix;
  let mvp = view_proj * model;
  let push_constants = PushConstants {
    model_view_proj: mvp.into(),
    model: model.into(),
    sun_pos: sun_pos.into(),
    texture_flags: draw_call.texture_flags,
    sun_color,
    camera_pos: camera_pos.into(),
    emissive_intensity: draw_call.emissive_intensity,
    emissive_color: draw_call.emissive_color,
    _unused_pad: 0,
  };
  device.push_constants(cmd_buffer, &push_constants)?;
  device.draw_indexed(cmd_buffer, draw_call.index_count)?;

  if draw_call.draw_outline {
    if let Some(outline_pipeline) = draw_call.outline_pipeline {
      device.bind_pipeline(cmd_buffer, outline_pipeline)?;
      // Note: same buffers because geometry is identical, only pipeline changes
      // but wait, bind_buffers also requires pipeline_key to identify layout in some engines
      // Let's assume it works or we use the regular pipeline key for bind_buffers
      device.bind_buffers(cmd_buffer, outline_pipeline, draw_call.buffers)?;

      let outline_push = PushConstants {
        model_view_proj: mvp.into(),
        model: model.into(),
        sun_pos: sun_pos.into(),
        texture_flags: draw_call.texture_flags,
        sun_color,
        camera_pos: camera_pos.into(),
        emissive_intensity: draw_call.outline_color[3], // using intensity for alpha? Or just packing color
        emissive_color: [
          draw_call.outline_color[0],
          draw_call.outline_color[1],
          draw_call.outline_color[2],
        ], // Emissive color abused for outline color
        _unused_pad: 0,
      };
      device.push_constants(cmd_buffer, &outline_push)?;
      device.set_line_width(cmd_buffer, 1.0)?;
      device.draw_indexed(cmd_buffer, draw_call.index_count)?;
    }
  }

  Ok(())
}
