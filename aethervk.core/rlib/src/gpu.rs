use core::{
  ffi,
  hash::{Hash, Hasher},
};
use ahash::AHasher;

// Note: this is a no_std environment
// Do not add anything unrelated to gpu interface here. If you want to define components, scene,
// and other modeling, create new files

use crate::{
  gpu::frame::ResourceUploadResult,
  scene::{EntityId, PhysicalMeshComponent, TransformComponent},
  simulation::comet::PushConstants,
};
use crate::types::{EngineResult, GpuResult};

// Re-export what is necessary from backends
pub use super::gpu_backends::new_render_frontend;
pub use super::gpu_backends::get_available_render_backends;
pub use super::gpu_backends::get_available_kernels;
pub use super::gpu_backends::{vulkan::constants};

pub use self::viewport::*;
pub use self::frame::RenderScene;

use heapless::index_map::FnvIndexMap;
use alloc::boxed::Box;
#[cfg(debug_assertions)]
use alloc::string::String;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderBackendId(pub u64);
pub const NULL_RENDER_BACEKND: RenderBackendId = RenderBackendId(0);
pub const VULKAN_RENDER_BACKEND: RenderBackendId = RenderBackendId(1);
pub const METAL_RENDER_BACKEND: RenderBackendId = RenderBackendId(2);
pub const D3D12_RENDER_BACKEND: RenderBackendId = RenderBackendId(3);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GpuResourceHandle(pub u64);
pub const NULL_GPU_RESOURCE: GpuResourceHandle = GpuResourceHandle(0);

impl GpuResourceHandle {
  pub fn from_raw(raw: u64) -> Self {
    Self(raw)
  }
}

pub struct KinematicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
}

pub struct DynamicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
  pub velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  pub mass: f32,
}

/// Ephemeral structure rebuilt every frame holding the snapshot of the simulated physical scene.
pub struct PhysicalScene {
  pub kinematic_bodies: alloc::vec::Vec<KinematicBody>,
  pub dynamic_bodies: alloc::vec::Vec<DynamicBody>,
}

impl PhysicalScene {
  pub fn new() -> Self {
    Self {
      kinematic_bodies: alloc::vec::Vec::new(),
      dynamic_bodies: alloc::vec::Vec::new(),
    }
  }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CommandBufferHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderableInstanceId(pub u64);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SkyPushConstants {
  pub inv_view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SunPushConstants {
  pub model_view_proj: [f32; 16],
  pub local_camera_pos: [f32; 3],
  pub _unused: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CursorPushConstants {
  pub view: [f32; 16],
  pub view_proj: [f32; 16],
  pub model: [f32; 16],
  pub cursor_size: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MarkerPushConstants {
  pub view_proj: [f32; 16],
  pub center_pos: [f32; 3],
  pub size: f32,
  pub color: [f32; 3],
  pub _pad0: f32,
  pub camera_up: [f32; 3],
  pub _pad1: f32,
  pub camera_right: [f32; 3],
  pub _pad2: f32,
}

impl RenderableInstanceId {
  pub fn from_physical_mesh(
    entity_id: EntityId,
    _physical_mesh_component: &PhysicalMeshComponent,
  ) -> Self {
    let mut hasher = AHasher::default();
    entity_id.hash(&mut hasher);
    Self(hasher.finish())
  }
}

impl From<RenderableInstanceId> for GpuResourceHandle {
  fn from(value: RenderableInstanceId) -> Self {
    Self(value.0)
  }
}

// Allow the host application to configure the core assets path uniformly
pub static ASSET_DIR: spin::RwLock<Option<alloc::string::String>> = spin::RwLock::new(None);

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub struct PipelineKey(pub u64);

pub trait PipelineKeyable {
  fn pipeline_key(&self) -> PipelineKey;
}

#[derive(Default, Clone, Copy)]
pub struct Rect2D {
  pub offset: [i32; 2],
  pub extent: [u32; 2],
}

#[derive(Clone, Copy)]
pub struct Viewport {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub min_depth: f32,
  pub max_depth: f32,
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum NativeGpuProperty {
  VulkanMetalDeviceId = 0,
}

pub trait RenderDevice: Send + Sync {
  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void>;

  #[cfg(debug_assertions)]
  fn print_info(&self) -> String;

  fn context_id(&self) -> u64;

  /// Prepare all the necessary state for a rendering operation. In particular
  /// - Update frame index within device and (vulkan) refresh timeline semaphore value
  /// - Refresh VMA memory budgets
  fn start_frame(&self) -> GpuResult<()>;

  /// Eagerly compiles pipelines and creates archetypes during initialization
  /// so we don't lazily compile on the first frame a specific mesh is requested.
  fn init_archetypes(&self, handle: PresentationEngineHandle) -> GpuResult<()>;

  /// Traverses a QuadTree of viewports and issues the respective drawing programs
  /// (3D Viewport or GUI elements)
  fn set_line_width(
    &self,
    cmd_buffer: CommandBufferHandle,
    width: f32,
  ) -> GpuResult<()>;

  fn render_frame(
    &self,
    cmd_buffer: CommandBufferHandle,
    viewports: &crate::gpu::viewport::ViewportQuadTree,
    render_scene: &crate::gpu::frame::RenderScene,
  ) -> GpuResult<()>;

  /// Creates the surface and initial swapchain
  fn create_presentation_engine(
    &self,
    params: &PresentationEngineParams,
  ) -> GpuResult<PresentationEngineHandle>;

  /// Allows caller to explicitly trigger a resize/recreation
  fn resize_presentation_engine(
    &self,
    handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()>;

  fn get_presentation_engine_extent(&self, handle: PresentationEngineHandle)
    -> GpuResult<[u32; 2]>;

  /// Acquires the next image. If it returns NeedsRecreation, the caller should
  /// discard the frame, call resize, and try again next frame
  fn acquire_next_image(&self, handle: PresentationEngineHandle) -> GpuResult<AcquireResult>;

  /// Returns (pipeline, vertex_buffer, index_buffer)
  fn get_or_create_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult>;

  /// Generates the background sky image using compute shader
  fn generate_sky(&self) -> GpuResult<()>;

  /// Returns resources for the cursor rendering
  fn get_or_create_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Returns resources for marker rendering
  fn get_or_create_marker_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Presents the image. Takes semaphore signaled by a rendering command buffer
  fn present(
    &self,
    handle: PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<SwapchainStatus>;

  /// Start for an interface to draw something on the screen. Gets a handle to store rendering
  /// state setting commands

  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
  ) -> GpuResult<()>;

  fn get_command_buffer(&self) -> GpuResult<CommandBufferHandle>;

  fn begin_command_buffer(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// responsible to acquire an image and store it in the associated command buffer structure
  fn begin_render_pass(
    &self,
    cmd_buffer: CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()>;

  fn set_viewport(&self, cmd_buffer: CommandBufferHandle, viewport: &Viewport) -> GpuResult<()>;

  fn set_scissor(&self, cmd_buffer: CommandBufferHandle, scissor: &Rect2D) -> GpuResult<()>;

  /// alter internal state for current command buffer binding a new graphics pipeline
  fn bind_pipeline(&self, cmd_buffer: CommandBufferHandle, pipeline: PipelineKey) -> GpuResult<()>;

  /// alter internal state for current command buffer to use a specific set of buffers, coherent with pipeline
  fn bind_buffers(
    &self,
    cmd_buffer: CommandBufferHandle,
    pipeline: PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()>;

  fn push_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &PushConstants,
  ) -> GpuResult<()>;

  fn push_sun_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &SunPushConstants,
  ) -> GpuResult<()>;

  fn push_cursor_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &CursorPushConstants,
  ) -> GpuResult<()>;

  fn push_marker_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &MarkerPushConstants,
  ) -> GpuResult<()>;

  fn draw_indexed(&self, cmd_buffer: CommandBufferHandle, index_count: u32) -> GpuResult<()>;

  fn draw(&self, cmd_buffer: CommandBufferHandle, vertex_count: u32) -> GpuResult<()>;

  fn update_sun(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SunComponent,
  ) -> GpuResult<()>;

  fn render_sun(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SunComponent,
    transform: &crate::scene::TransformComponent,
    view: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()>;

  fn render_sky(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SkyComponent,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()>;

  fn render_grid(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::GridComponent,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    camera_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    near_plane: f32,
    far_plane: f32,
  ) -> GpuResult<()>;

  fn render_minimap(
    &self,
    cmd_buffer: CommandBufferHandle,
    player_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    max_distance: f32,
    planets: &[(
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
      f32,
      [f32; 4],
    )],
  ) -> GpuResult<()>;

  fn render_bvh(
    &self,
    cmd_buffer: CommandBufferHandle,
    nodes: &[(crate::math::collision::linear_bvh::LinearBound<f32, aethervk_oshal_rlib::math::vector::vec3::Vec3f32, aethervk_oshal_rlib::math::matrix::mat3::Mat3f32>, aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32)],
    view_proj: [f32; 16],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn render_ui_rect(
    &self,
    cmd_buffer: CommandBufferHandle,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn render_text(
    &self,
    cmd_buffer: CommandBufferHandle,
    text: &str,
    font_path: &str,
    points: f32,
    color: [f32; 4],
    position: [f32; 2],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn end_render_pass(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  fn submit_command_buffer(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderDeviceHandle(pub u64);

/// backend specific additional device init parameters
pub type DeviceAdditionalParams = FnvIndexMap<u64, usize, 8>;

pub trait RenderContext: Send + Sync {
  fn backend_id(&self) -> RenderBackendId;

  fn init_device(
    &mut self,
    index: usize,
    additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle>;

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>>;
}

// NOTE: This is a box like type, so we don't need to box it when returning it to cdylib,
// we can instead use the ManualDrop mechanism
pub struct RenderFrontend<'a> {
  backend: spin::RwLock<Box<dyn RenderContext + 'a>>,
}

impl<'a> RenderFrontend<'a> {
  pub fn take_and<T>(
    &self,
    f: impl FnOnce(&dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_read() {
      Some(guard) => Some(f(guard.as_ref())),
      None => None,
    }
  }

  pub fn take_mut_and<T>(
    &mut self,
    f: impl FnOnce(&mut dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_write() {
      Some(mut guard) => Some(f(guard.as_mut())),
      None => None,
    }
  }
}

// Boxing mechanism used by factory method in `gpu_backends` `new_render_frontend`
impl<'a, T> From<T> for RenderFrontend<'a>
where
  T: RenderContext + 'a,
{
  fn from(value: T) -> Self {
    RenderFrontend {
      backend: spin::RwLock::new(Box::new(value)),
    }
  }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct PresentationEngineHandle(pub u64);

/// Status returned by acquire and present operations
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum SwapchainStatus {
  Optimal = 0,
  /// Swapchain is usable but dimensions/properties no longer match the underlying surface
  Suboptimal = 1,
  /// The surface changed drastically (resize) and the current frame must be discarded
  NeedsRecreation = 2,
}

impl SwapchainStatus {
  pub fn needs_resize(self) -> bool {
    self != Self::Optimal
  }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AcquireResult {
  pub image_index: u32,
  pub status: SwapchainStatus,
  /// handle to frame synchronization resources recognized by the presentation engine
  pub frame_index: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OpaqueNativeHandleInfo {
  pub ptr0: *mut ffi::c_void,
  pub ptr1: *mut ffi::c_void,
}

/// Parameters passed from Avalonia to create the surface/swapchain
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PresentationEngineType {
  Window,
  WindowLess,
}

pub struct PresentationEngineParams {
  pub width: u32,
  pub height: u32,
  pub vsync: bool,
  pub window_info: OpaqueNativeHandleInfo,
  pub ty: PresentationEngineType,
}

impl PresentationEngineParams {
  pub fn windowless(width: u32, height: u32) -> Self {
    Self {
      width,
      height,
      vsync: false,
      ty: PresentationEngineType::WindowLess,
      window_info: OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
    }
  }
}

/// Computes execution for physics, particle systems, and interval arithmetic.
pub trait Kernels: Send + Sync {
  /// Dispatches compute shaders to step the physical simulation dynamically.
  fn dispatch_physics_step(
    &self,
    cmd_buffer: CommandBufferHandle,
    physical_scene: &PhysicalScene,
    dt: f32,
  ) -> GpuResult<()>;

  /// Dispatches compute shaders for other effects (e.g., particles).
  fn dispatch_particles(&self, cmd_buffer: CommandBufferHandle, dt: f32) -> GpuResult<()>;
}

/// Bridges synchronization between Compute (Kernels) and Graphics (RenderDevice).
pub trait KernelRenderBridge: Send + Sync {
  /// Inserts pipeline barriers or queue ownership transfers from Compute to Graphics.
  fn sync_compute_to_graphics(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Inserts pipeline barriers or queue ownership transfers from Graphics to Compute.
  fn sync_graphics_to_compute(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;
}

pub mod frame;
pub mod viewport;
