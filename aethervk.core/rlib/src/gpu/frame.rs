use crate::gpu::{
  AcquireResult, GpuResourceHandle, PipelineKey, PresentationEngineHandle, RenderDevice,
};
use crate::scene::{CameraComponent, EntityId, RenderableDataRef, TransformComponent, SunComponent, SkyComponent, GridComponent};
use crate::simulation::comet::{PushConstants, TextureFlags};
use crate::types::{GpuError, GpuResult};
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix4},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, Vector3},
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

/// A proper clear-cut struct representing the rendering scene.
pub struct RenderScene {
  /// A list of draw calls to be executed for this frame.
  pub draw_calls: Vec<DrawCall>,
  /// A list of cursor draw calls.
  pub cursor_calls: Vec<CursorDrawCall>,
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
      RenderableDataRef::ImageBillboard(_component) => {
        todo!();
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
  // Use non-indexed drawing for the cursor
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
        emissive_color: [draw_call.outline_color[0], draw_call.outline_color[1], draw_call.outline_color[2]], // Emissive color abused for outline color
        _unused_pad: 0,
      };
      device.push_constants(cmd_buffer, &outline_push)?;
      device.draw_indexed(cmd_buffer, draw_call.index_count)?;
    }
  }

  Ok(())
}