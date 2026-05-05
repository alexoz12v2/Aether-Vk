use crate::gpu::{
  AcquireResult, CommandBufferHandle, CursorPushConstants, DeviceAdditionalParams,
  GpuResourceHandle, NativeGpuProperty, PipelineKey, PresentationEngineHandle,
  PresentationEngineParams, Rect2D, RenderBackendId, RenderContext, RenderDevice,
  RenderDeviceHandle, SkyPushConstants, SunPushConstants, SwapchainStatus, Viewport, MeasurementPushConstants,
  METAL_RENDER_BACKEND,
};
use crate::gpu::frame::ResourceUploadResult;
use crate::scene::{
  EntityId, PhysicalMeshComponent, SkyComponent, SunComponent, TransformComponent, GridComponent,
};
use crate::simulation::comet::{PushConstants, Texture};
use crate::traits::InitWithRuntime;
use crate::types::{EngineResult, GpuResult, RuntimeParams, GpuError};
use alloc::boxed::Box;
#[cfg(debug_assertions)]
use alloc::string::String;
use core::ffi;

// Bring in the objc2 dependencies just to ensure they are used if we wanted to
// (removed due to import errors)

use alloc::sync::Arc;
use spin::rwlock::RwLock;

pub struct MetalRenderContext {}

impl InitWithRuntime<Self> for MetalRenderContext {
  fn init_with_runtime(_params: &RuntimeParams) -> EngineResult<Self> {
    Ok(Self {})
  }
}

impl RenderContext for MetalRenderContext {
  fn backend_id(&self) -> RenderBackendId {
    METAL_RENDER_BACKEND
  }

  fn init_device(
    &mut self,
    _index: usize,
    _additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle> {
    Ok(RenderDeviceHandle(1))
  }

  fn deref_device_and(
    &self,
    _dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    let dev = MetalRenderDevice {};
    Some(f(&dev, p_user_data))
  }
}

pub struct MetalRenderDevice {}

impl RenderDevice for MetalRenderDevice {
  fn get_native_prop(&self, _prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void> {
    None
  }

  fn print_info(&self) -> String {
    String::from("Metal Render Device")
  }

  fn context_id(&self) -> u64 {
    0
  }

  fn start_frame(&self) -> GpuResult<()> {
    Ok(())
  }

  fn init_archetypes(&self, _handle: PresentationEngineHandle) -> GpuResult<()> {
    Ok(())
  }

  fn set_line_width(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _width: f32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_frame(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _render_scene: &crate::gpu::frame::RenderScene,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn create_presentation_engine(
    &self,
    _params: &PresentationEngineParams,
  ) -> GpuResult<PresentationEngineHandle> {
    Ok(PresentationEngineHandle(1))
  }

  fn resize_presentation_engine(
    &self,
    _handle: PresentationEngineHandle,
    _width: u32,
    _height: u32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn get_presentation_engine_extent(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<[u32; 2]> {
    Ok([0, 0])
  }

  fn acquire_next_image(&self, _handle: PresentationEngineHandle) -> GpuResult<AcquireResult> {
    Ok(AcquireResult {
      image_index: 0,
      status: SwapchainStatus::Optimal,
      frame_index: 0,
    })
  }

  fn get_or_create_physical_mesh_resources(
    &self,
    _entity_id: EntityId,
    _component: &PhysicalMeshComponent,
    _handle: PresentationEngineHandle,
    _debug_name: &str,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn generate_sky(&self) -> GpuResult<()> {
    Ok(())
  }

  fn get_or_create_billboard_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn get_cursor_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn create_cursor_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn present(
    &self,
    _handle: PresentationEngineHandle,
    _image_index: usize,
    _frame_index: usize,
  ) -> GpuResult<SwapchainStatus> {
    Ok(SwapchainStatus::Optimal)
  }

  fn download_windowless_image(
    &self,
    _handle: PresentationEngineHandle,
    _buffer: &mut [u8],
    _task_id: Option<u64>,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn get_command_buffer(&self) -> GpuResult<CommandBufferHandle> {
    Ok(CommandBufferHandle(1))
  }

  fn begin_command_buffer(&self, _cmd_buffer: CommandBufferHandle) -> GpuResult<()> {
    Ok(())
  }

  fn begin_render_pass(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _presentation_engine: PresentationEngineHandle,
    _acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn set_viewport(&self, _cmd_buffer: CommandBufferHandle, _viewport: &Viewport) -> GpuResult<()> {
    Ok(())
  }

  fn set_scissor(&self, _cmd_buffer: CommandBufferHandle, _scissor: &Rect2D) -> GpuResult<()> {
    Ok(())
  }

  fn bind_pipeline(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _pipeline: PipelineKey,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn bind_buffers(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _pipeline: PipelineKey,
    _buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn push_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &PushConstants,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn push_sun_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &SunPushConstants,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn push_cursor_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &CursorPushConstants,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn draw_indexed(&self, _cmd_buffer: CommandBufferHandle, _index_count: u32) -> GpuResult<()> {
    Ok(())
  }

  fn draw(&self, _cmd_buffer: CommandBufferHandle, _vertex_count: u32) -> GpuResult<()> {
    Ok(())
  }

  fn prepare_particle_archetype_for_render_and_bind_pipeline(
    &self,
    _cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    todo!()
  }

  fn draw_indirect(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _indirect_buffer: GpuResourceHandle,
    _offset: u64,
    _draw_count: u32,
    _stride: u32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn update_sun(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _resolution: (u32, u32, u32),
    _radius: f32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_sun(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &SunComponent,
    _transform: &TransformComponent,
    _view: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    _view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_sky(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &SkyComponent,
    _view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_grid(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &GridComponent,
    _view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    _camera_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    _near_plane: f32,
    _far_plane: f32,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_minimap(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _player_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    _max_distance: f32,
    _planets: &[(
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
      f32,
      [f32; 4],
    )],
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_bvh(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _nodes: &[(crate::math::collision::linear_bvh::LinearBound<f32>, aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32)],
    _view_proj: [f32; 16],
    _presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_ui_rect(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _color: [f32; 4],
    _position: [f32; 2],
    _size: [f32; 2],
    _presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn render_text(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _text: &str,
    _font_path: &str,
    _points: f32,
    _color: [f32; 4],
    _position: [f32; 2],
    _presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn get_or_create_gizmo_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn update_gizmo_instance(
    &self,
    _entity: EntityId,
    _model: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<u32> {
    Err(GpuError::UnsupportedFeature)
  }

  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    _cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    Err(GpuError::UnsupportedFeature)
  }

  fn get_or_create_measurement_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    todo!()
  }

  fn push_measurement_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &MeasurementPushConstants,
  ) -> GpuResult<()> {
    todo!()
  }

  fn push_gizmo_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &crate::gpu::GizmoPushConstants,
  ) -> GpuResult<()> {
    todo!()
  }

  fn push_billboard_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &crate::gpu::BillboardPushConstants,
  ) -> GpuResult<()> {
    todo!()
  }

  fn get_marker_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }

  fn create_marker_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    Err(GpuError::UnsupportedFeature)
  }
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &crate::gpu::MarkerPushConstants,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn end_render_pass(&self, _cmd_buffer: CommandBufferHandle) -> GpuResult<()> {
    Ok(())
  }

  fn submit_command_buffer(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _task_id: Option<u64>,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn wire_callbacks(
      &self,
      pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> GpuResult<()> {
    Ok(())
  }

  fn is_task_completed(&self, _task_id: u64) -> GpuResult<bool> {
    Ok(true)
  }

  fn create_task(&self) -> u64 {
    0
  }

  fn fail_task(&self, _task_id: u64, _error: GpuError) {}

  fn success_task(&self, _task_id: u64) {}

  fn check_billboard_texture_id(&self, texture_id: u64) -> GpuResult<()> {
    todo!()
  }

  fn add_billboard_texture(&self, texture: &Texture) -> GpuResult<()> {
    todo!()
  }

  fn destroy_presentation_engine(&self, handle: PresentationEngineHandle) -> GpuResult<()> {
    todo!()
  }
}

}
esult<()> {
    todo!()
  }
}
self, texture_id: u64) -> GpuResult<()> {
    todo!()
  }

  fn add_billboard_texture(&self, texture: &Texture) -> GpuResult<()> {
    todo!()
  }

  fn destroy_presentation_engine(&self, handle: PresentationEngineHandle) -> GpuResult<()> {
    todo!()
  }
}

}
esult<()> {
    todo!()
  }
}
}
esult<()> {
    todo!()
  }
}
