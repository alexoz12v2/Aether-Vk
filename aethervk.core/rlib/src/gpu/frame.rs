use crate::gpu::{
  AcquireResult, GpuResourceHandle, PipelineKey, PresentationEngineHandle, RenderDevice,
};
use crate::scene::{CameraComponent, EntityId, RenderableDataRef, TransformComponent};
use crate::simulation::comet::{PushConstants, TextureFlags};
use crate::types::GpuResult;
use aethervk_oshal_rlib::math::{
  matrix::{mat4::Mat4x4f32, Matrix4},
  quaternion::Quaternion,
  vector::{vec3::Vec3f32, Vector, Vector3},
};
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceUploadResult {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  /// The vertex buffer to bind.
  pub buffers: GpuResourceHandle,
  pub texture_flags: TextureFlags,
}

/// Represents a single draw call with all necessary information.
#[derive(Clone)]
pub struct DrawCall {
  /// The pipeline to use for this draw call.
  pub pipeline: PipelineKey,
  /// The vertex buffer to bind.
  pub buffers: GpuResourceHandle,
  /// index count
  pub index_count: u32,
  /// The model matrix of the object to draw.
  pub model_matrix: Mat4x4f32,
  pub texture_flags: TextureFlags,
}

impl DrawCall {
  pub(crate) fn from_handles_and_matrix(
    result: ResourceUploadResult,
    index_count: u32,
    model_matrix: Mat4x4f32,
  ) -> Self {
    Self {
      pipeline: result.pipeline,
      buffers: result.buffers,
      index_count,
      model_matrix,
      texture_flags: result.texture_flags,
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

/// A collection of all draw calls and resources for a single frame.
pub struct Frame {
  /// A list of draw calls to be executed for this frame.
  pub draw_calls: Vec<DrawCall>,
  /// A list of cursor draw calls.
  pub cursor_calls: Vec<CursorDrawCall>,
}

impl Frame {
  pub fn new() -> Self {
    Self {
      draw_calls: Vec::new(),
      cursor_calls: Vec::new(),
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
        )?;
        let index_count = component.mesh.indices.len() as u32;
        self.draw_calls.push(DrawCall::from_handles_and_matrix(
          res,
          index_count,
          model_matrix,
        ));
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

/// A trait for a render path, which defines a strategy for rendering a frame.
/// For now, we assume a render pass abstraction is available.
pub trait RenderPath {
  /// Records the rendering commands for a given frame into a command buffer.
  /// The `render_pass` parameter is a placeholder for a render pass abstraction.
  fn record_commands(
    &self,
    device: &dyn RenderDevice,
    camera: (&TransformComponent, &CameraComponent),
    frame: &Frame,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()>;
}

/// A simple forward rendering path.
pub struct ForwardRenderPath;

impl RenderPath for ForwardRenderPath {
  fn record_commands(
    &self,
    device: &dyn RenderDevice,
    camera: (&TransformComponent, &CameraComponent),
    frame: &Frame,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    let cmd_buffer = device.get_command_buffer()?;

    device.begin_command_buffer(cmd_buffer)?;
    device.begin_render_pass(cmd_buffer, presentation_engine, acquire_result)?;

    let extent = device.get_presentation_engine_extent(presentation_engine)?;
    device.set_viewport(
      cmd_buffer,
      &super::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent[0] as f32,
        height: extent[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
      },
    )?;
    device.set_scissor(
      cmd_buffer,
      &super::Rect2D {
        offset: [0, 0],
        extent,
      },
    )?;
    let view = Mat4x4f32::look_at(
      camera.0.position,
      camera.0.position
        + camera
          .0
          .rotation
          .rotate_vector(<Vec3f32 as Vector3>::from_components(0.0, 0.0, -1.0)),
      <Vec3f32 as Vector3>::from_components(0.0, 1.0, 0.0),
    );
    let proj = camera.1.projection;
    let view_proj = proj * view;

    for draw_call in &frame.draw_calls {
      let _ = do_draw_call(device, view_proj, camera.0.position, cmd_buffer, draw_call);
    }
    for cursor_call in &frame.cursor_calls {
      let _ = do_draw_cursor(device, view, view_proj, cmd_buffer, cursor_call);
    }

    device.end_render_pass(cmd_buffer)?;
    device.submit_command_buffer(cmd_buffer)?;

    Ok(())
  }
}

fn do_draw_cursor(
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
    cursor_pos: {
      let pos_arr: [f32; 4] = draw_call.model_matrix.w.into();
      [pos_arr[0], pos_arr[1], pos_arr[2]]
    }, // position is the 4th column of model matrix
    cursor_size: 0.05, // TODO extract from draw call
  };
  device.push_cursor_constants(cmd_buffer, &push_constants)?;
  // Use non-indexed drawing for the cursor
  device.draw(cmd_buffer, draw_call.vertex_count)?;
  Ok(())
}

fn do_draw_call(
  device: &dyn RenderDevice,
  view_proj: Mat4x4f32,
  camera_pos: Vec3f32,
  cmd_buffer: super::CommandBufferHandle,
  draw_call: &DrawCall,
) -> Result<(), crate::types::GpuError> {
  // 2. Bind pipeline and buffers
  device.bind_pipeline(cmd_buffer, draw_call.pipeline)?;
  device.bind_buffers(cmd_buffer, draw_call.pipeline, draw_call.buffers)?;

  // 3. Math & Push constants
  let model = draw_call.model_matrix;
  let mvp = view_proj * model;
  let push_constants = PushConstants {
    model_view_proj: mvp.into(),
    model: model.into(),
    sun_dir: [0.0, -1.0, 0.0],
    texture_flags: draw_call.texture_flags,
    sun_color: [1.0, 1.0, 1.0, 1.0],
    camera_pos: camera_pos.into(),
    _unused: 0,
  };
  device.push_constants(cmd_buffer, &push_constants)?;
  device.draw_indexed(cmd_buffer, draw_call.index_count)?;
  Ok(())
}
