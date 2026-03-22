use crate::gpu::{GpuResourceHandle, RenderDevice};
use crate::scene::{CameraComponent, EntityId, RenderableDataRef, TransformComponent};
use crate::types::GpuResult;
use alloc::vec::Vec;

// ----------------------------------------------------------------------------
//                     PER-FRAME RENDERING ABSTRACTION
// ----------------------------------------------------------------------------

/// Represents a single draw call with all necessary information.
pub struct DrawCall {
  /// The pipeline to use for this draw call.
  pub pipeline: GpuResourceHandle,
  /// The vertex buffer to bind.
  pub vertex_buffer: GpuResourceHandle,
  /// The index buffer to bind.
  pub index_buffer: GpuResourceHandle,
  /// Number of indices to draw.
  pub index_count: u32,
  // In a real scenario, this would also contain descriptor sets for uniforms, textures, etc.
}

/// A collection of all draw calls and resources for a single frame.
pub struct Frame {
  /// A list of draw calls to be executed for this frame.
  pub draw_calls: Vec<DrawCall>,
}

impl Frame {
  pub fn new() -> Self {
    Self {
      draw_calls: Vec::new(),
    }
  }

  /// Registers a renderable entity to be drawn in this frame.
  pub fn add_renderable(
    &mut self,
    device: &dyn RenderDevice,
    entity_id: EntityId,
    transform: &TransformComponent,
    renderable: RenderableDataRef,
  ) -> GpuResult<()> {
    self.draw_calls.push(match renderable {
      RenderableDataRef::ImageBillboard(component) => {
        todo!();
      }
      RenderableDataRef::PhysicalMesh(component) => {
        let (pipeline, vertex_buffer, index_buffer) = todo!(); // device.get_or_create_physical_mesh_resources(&component, &transform)
        let index_count = component.mesh.indices.len() as u32;
        DrawCall {
          pipeline,
          vertex_buffer,
          index_buffer,
          index_count,
        }
      }
    });
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
    render_pass: GpuResourceHandle, // Placeholder for render pass abstraction
                                    // In a real implementation, this would take a command buffer handle.
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
    render_pass: GpuResourceHandle,
  ) -> GpuResult<()> {
    // 0. Query system for state for previous render operation, if finished. If not, then you'll wait here
    todo!();
    // 1. Begin the render pass.
    todo!(); // device.renderpass_for_frame(() -> function executed for each frame internal's representation, gives a proxy
    // after end of renderpass, if result ok, submit and present immediately. Problem: these are async operations. Solution: Next call the query system call will tell you if everything was fine or not
    // 2. Iterate through the frame's draw calls.
    // 3. For each draw call:
    //    a. Bind the pipeline.
    //    b. Bind vertex and index buffers.
    //    c. Bind descriptor sets (for transforms, materials, etc.).
    //    d. Issue a draw indexed command.
    // 4. End the render pass.
    todo!();
  }
}
