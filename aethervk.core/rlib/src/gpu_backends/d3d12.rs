use core::ffi;
#[cfg(debug_assertions)]
use alloc::string::String;

use crate::{
  gpu::{
    AcquireResult, CommandBufferHandle, CursorPushConstants, D3D12_RENDER_BACKEND,
    DeviceAdditionalParams, GpuResourceHandle, NativeGpuProperty, PipelineKey, PushConstants,
    Rect2D, RenderBackendId, RenderContext, RenderDevice, RenderDeviceHandle, SunPushConstants,
    SwapchainStatus, Viewport, PresentationEngineHandle, PresentationEngineParams,
  },
  gpu::viewport::ViewportQuadTree,
  gpu::frame::{RenderScene, ResourceUploadResult},
  traits::InitWithRuntime,
  types::{GpuResult, RuntimeParams, EngineResult},
  scene::{
    EntityId, GridComponent, PhysicalMeshComponent, SkyComponent, SunComponent, TransformComponent,
  },
};

use aethervk_oshal_rlib::math::{matrix::mat4::Mat4x4f32, vector::vec3::Vec3f32};

pub struct D3d12RenderDevice;

impl RenderDevice for D3d12RenderDevice {
  fn get_native_prop(&self, _prop: NativeGpuProperty) -> Option<*mut ffi::c_void> {
    unimplemented!()
  }

  fn print_info(&self) -> String {
    unimplemented!()
  }

  fn context_id(&self) -> u64 {
    D3D12_RENDER_BACKEND.0
  }

  fn start_frame(&self) -> GpuResult<()> {
    unimplemented!()
  }

  fn init_archetypes(&self, _handle: PresentationEngineHandle) -> GpuResult<()> {
    unimplemented!()
  }

  fn set_line_width(&self, _cmd_buffer: CommandBufferHandle, _width: f32) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_frame(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _render_scene: &RenderScene,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn create_presentation_engine(
    &self,
    _params: &PresentationEngineParams,
  ) -> GpuResult<PresentationEngineHandle> {
    unimplemented!()
  }

  fn destroy_present_engine(&self, _present_engine: PresentationEngineHandle) {
    todo!()
  }

  fn resize_presentation_engine(
    &self,
    _handle: PresentationEngineHandle,
    _width: u32,
    _height: u32,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn get_presentation_engine_extent(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<[u32; 2]> {
    unimplemented!()
  }

  fn acquire_next_image(&self, _handle: PresentationEngineHandle) -> GpuResult<AcquireResult> {
    unimplemented!()
  }

  fn get_or_create_physical_mesh_resources(
    &self,
    _entity_id: EntityId,
    _component: &PhysicalMeshComponent,
    _handle: PresentationEngineHandle,
    _debug_name: &str,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn generate_sky(&self) -> GpuResult<()> {
    unimplemented!()
  }

  fn get_marker_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn create_marker_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn create_billboard_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn get_marker_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn create_marker_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    unimplemented!()
  }

  fn present(
    &self,
    _handle: PresentationEngineHandle,
    _image_index: usize,
    _frame_index: usize,
  ) -> GpuResult<SwapchainStatus> {
    unimplemented!()
  }

  fn download_windowless_image(
    &self,
    _handle: PresentationEngineHandle,
    _buffer: &mut [u8],
    _task_id: Option<u64>,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn get_command_buffer(&self) -> GpuResult<CommandBufferHandle> {
    unimplemented!()
  }

  fn begin_command_buffer(&self, _cmd_buffer: CommandBufferHandle) -> GpuResult<()> {
    unimplemented!()
  }

  fn begin_render_pass(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _presentation_engine: PresentationEngineHandle,
    _acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn set_viewport(&self, _cmd_buffer: CommandBufferHandle, _viewport: &Viewport) -> GpuResult<()> {
    unimplemented!()
  }

  fn set_scissor(&self, _cmd_buffer: CommandBufferHandle, _scissor: &Rect2D) -> GpuResult<()> {
    unimplemented!()
  }

  fn bind_pipeline(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _pipeline: PipelineKey,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn bind_buffers(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _pipeline: PipelineKey,
    _buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn push_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &PushConstants,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn push_sun_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &SunPushConstants,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn push_cursor_constants(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _push_constants: &CursorPushConstants,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn draw_indexed(&self, _cmd_buffer: CommandBufferHandle, _index_count: u32) -> GpuResult<()> {
    unimplemented!()
  }

  fn draw(&self, _cmd_buffer: CommandBufferHandle, _vertex_count: u32) -> GpuResult<()> {
    unimplemented!()
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
    unimplemented!()
  }

  fn update_sun(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &SunComponent,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_sun(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &SunComponent,
    _transform: &TransformComponent,
    _view: Mat4x4f32,
    _view_proj: Mat4x4f32,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_sky(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &SkyComponent,
    _view_proj: Mat4x4f32,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_grid(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _entity_id: EntityId,
    _component: &GridComponent,
    _view_proj: Mat4x4f32,
    _camera_pos: Vec3f32,
    _near_plane: f32,
    _far_plane: f32,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_minimap(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _player_pos: Vec3f32,
    _max_distance: f32,
    _planets: &[(Vec3f32, f32, [f32; 4])],
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_bvh(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _nodes: &[(
      crate::math::collision::linear_bvh::LinearBound<
        f32,
        aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
        aethervk_oshal_rlib::math::matrix::mat3::Mat3f32,
      >,
      aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    )],
    _view_proj: [f32; 16],
    _presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    unimplemented!()
  }

  fn render_ui_rect(
    &self,
    _cmd_buffer: CommandBufferHandle,
    _color: [f32; 4],
    _position: [f32; 2],
    _size: [f32; 2],
    _presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    unimplemented!()
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
    unimplemented!()
  }

  fn end_render_pass(&self, _cmd_buffer: CommandBufferHandle) -> GpuResult<()> {
    unimplemented!()
  }

  fn submit_command_buffer(&self, _cmd_buffer: CommandBufferHandle) -> GpuResult<()> {
    unimplemented!()
  }
}

pub struct D3d12RenderContext {
  device: D3d12RenderDevice,
}

impl InitWithRuntime for D3d12RenderContext {
  fn init_with_runtime(_params: &RuntimeParams) -> EngineResult<Self> {
    Ok(Self {
      device: D3d12RenderDevice,
    })
  }
}

impl RenderContext for D3d12RenderContext {
  fn backend_id(&self) -> RenderBackendId {
    D3D12_RENDER_BACKEND
  }

  fn init_device(
    &mut self,
    _index: usize,
    _additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle> {
    unimplemented!()
  }

  fn deref_device_and(
    &self,
    _dev_handle: RenderDeviceHandle,
    _p_user_data: *mut ffi::c_void,
    _f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    unimplemented!()
  }
}
 {
    unimplemented!()
  }
}
