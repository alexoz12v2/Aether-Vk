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
use crate::types::{EngineResult, GpuError, GpuResult};

// Re-export what is necessary from backends
pub use super::gpu_backends::new_render_frontend;
pub use super::gpu_backends::get_available_render_backends;
pub use super::gpu_backends::get_available_kernels;
pub use super::gpu_backends::{vulkan::constants};

pub use self::viewport::*;
pub use self::frame::RenderScene;

pub type RwLock<T> = spin::rwlock::RwLock<T>;

use heapless::index_map::FnvIndexMap;
use alloc::boxed::Box;
#[cfg(debug_assertions)]
use alloc::string::String;
use alloc::sync::Arc;
use aethervk_oshal_rlib::os::time::timeus_t;
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::simulation::comet::Texture;

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

#[derive(Clone, Copy)]
pub struct KinematicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
}

#[derive(Clone, Copy)]
pub struct DynamicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
  pub velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  pub mass: f32,
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
pub struct MeasurementPushConstants {
  pub view_proj: [f32; 16],
  pub p1: [f32; 3],
  pub _pad0: f32,
  pub p2: [f32; 3],
  pub _pad1: f32,
  pub camera_up: [f32; 3],
  pub _pad2: f32,
  pub camera_right: [f32; 3],
  pub _pad3: f32,
  pub color: [f32; 3],
  pub _pad4: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BillboardPushConstants {
  pub view_proj: [f32; 16],
  pub center_pos: [f32; 3],
  pub _pad0: f32,
  pub camera_up: [f32; 3],
  pub _pad1: f32,
  pub camera_right: [f32; 3],
  pub _pad2: f32,
  pub size: [f32; 2],
  pub is_screen_space: u32,
  pub texture_id: u32,
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
  fn set_line_width(&self, cmd_buffer: CommandBufferHandle, width: f32) -> GpuResult<()>;

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

  /// Returns resources for the billboard rendering
  fn get_or_create_billboard_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Returns resources for the cursor rendering
  fn get_or_create_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Returns resources for marker rendering
  fn get_or_create_measurement_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

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

  /// check for billboard texture existance
  fn check_billboard_texture_id(&self, texture_id: u64) -> GpuResult<()>;

  // `init_archetypes` should have already been called
  fn add_billboard_texture(&self, texture: &Texture) -> GpuResult<()>;

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

  fn push_measurement_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &MeasurementPushConstants,
  ) -> GpuResult<()>;

  fn push_billboard_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    push_constants: &BillboardPushConstants,
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
    nodes: &[(
      crate::math::collision::linear_bvh::LinearBound<f32>,
      aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    )],
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

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
  ) -> GpuResult<()>;

  fn wire_callbacks(
    &self,
    pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  ) -> GpuResult<()>;

  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool>;

  fn create_task(&self) -> u64;

  fn fail_task(&self, task_id: u64, error: GpuError);
  
  fn success_task(&self, task_id: u64);
}

/// An RAII guard ensuring the command buffer is always submitted.
pub struct ScopedCommandBuffer<'a> {
  device: &'a dyn RenderDevice,
  cmd_buffer: CommandBufferHandle,
  submitted: bool,
}

impl<'a> ScopedCommandBuffer<'a> {
  pub fn new(device: &'a dyn RenderDevice, cmd_buffer: CommandBufferHandle) -> GpuResult<Self> {
    device.begin_command_buffer(cmd_buffer)?;
    Ok(Self {
      device,
      cmd_buffer,
      submitted: false,
    })
  }

  pub fn cmd(&self) -> CommandBufferHandle {
    self.cmd_buffer
  }

  /// Explicitly submits the command buffer.
  pub fn submit(mut self) -> GpuResult<()> {
    self.submitted = true;
    self.device.submit_command_buffer(self.cmd_buffer, None)
  }
}

impl<'a> Drop for ScopedCommandBuffer<'a> {
  fn drop(&mut self) {
    if !self.submitted {
      // Force submission on early exit/panic. Result is ignored to prevent double panics.
      let _ = self.device.submit_command_buffer(self.cmd_buffer, None);
    }
  }
}

/// An RAII guard ensuring the render pass is always ended.
pub struct ScopedRenderPass<'a> {
  device: &'a dyn RenderDevice,
  cmd_buffer: CommandBufferHandle,
  ended: bool,
}

impl<'a> ScopedRenderPass<'a> {
  pub fn new(device: &'a dyn RenderDevice, cmd_buffer: CommandBufferHandle) -> Self {
    Self {
      device,
      cmd_buffer,
      ended: false,
    }
  }

  /// Explicitly ends the render pass.
  pub fn end(mut self) -> GpuResult<()> {
    self.ended = true;
    self.device.end_render_pass(self.cmd_buffer)
  }
}

impl<'a> Drop for ScopedRenderPass<'a> {
  fn drop(&mut self) {
    if !self.ended {
      // Force end on early exit/panic.
      let _ = self.device.end_render_pass(self.cmd_buffer);
    }
  }
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
    &self,
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

// -- Compute Engine traits --
/// Abstract handle for a compute queue (Vulkan/Metal Command Buffer, CUDA Stream, ...)
/// Represents a thread of execution
pub trait CommandBuffer: Send + Sync {
  /// Dispatches the recorded command graph to the backend hardware queue
  fn submit(&mut self) -> EngineResult<()>;
}

/// Continuous array residing entirely in backend memory
pub trait DeviceBuffer<T>: Send + Sync {
  type Cmd: CommandBuffer;
  /// Handle type representing pending GPU-to-CPU DMA transfer.
  /// Lifetime constraint: This handle should die before device buffer
  type ReadHandle<'a>: WaitHandle<alloc::vec::Vec<timeus_t>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize;

  /// Enqueues a DMA copy-back command to the CPU. the returned Future does NOT
  /// borrow `cmd`, allowing you to submit the command buffer while the tasklet
  /// awaits the GPU synchronization primitive (fence)
  fn enqueue_read_to_cpu<'a>(&self, cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>>;
}

/// A handle representing a pending GPU-to-CPU DMA transfer.
/// WARN: Any WaitHandle implementation should implement Drop, so that if we early exit from a function we know wait has been done.
pub trait WaitHandle<T>: Send + Sync {
  /// Blocks the current thread (or yields the tasklet back to your custom
  /// engine scheduler) until the hardware signals completion. Consumes the handle.
  fn wait(self) -> EngineResult<T>;
}

/// Specialized `DeviceBuffer` with a dynamic length managed by an atomic counter on the GPU
/// This is heavily used for Stream compaction
pub trait DeviceList<T>: DeviceBuffer<T> {
  fn clear(&mut self, cmd: &mut Self::Cmd) -> EngineResult<()>;
}

/// Opaque trait for backend-specific Bounding Volume Hierarchy
pub trait DeviceBvh: Send + Sync {
  type Cmd: CommandBuffer;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColliderId {
  pub entity_id: u32,
  /// Set to `u32::MAX` if it's a monolithic body. Otherwise, it is the particle instance index.
  pub primitive_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CollisionPair {
  pub a: ColliderId,
  pub b: ColliderId,
  pub time_of_impact: f32,
}

/// Computes execution for physics, particle systems, and interval arithmetic.
pub trait Kernels: Send + Sync {
  type Cmd: CommandBuffer;

  // --- Associated Types mapping to the underlying Backend ---
  type Buffer<T: Copy + Send + Sync>: DeviceBuffer<T, Cmd = Self::Cmd>;
  type List<T: Copy + Send + Sync>: DeviceList<T, Cmd = Self::Cmd>;
  type MotionBvh: DeviceBvh<Cmd = Self::Cmd>;

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd>;

  // 1. & 2. Build Collections
  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>>;
  fn build_dynamic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<DynamicBody>>;

  // 3. Apply gravitational / position-dependent forces
  fn compute_forces(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    dynamics: &mut Self::Buffer<DynamicBody>,
  ) -> EngineResult<()>;

  // 4. ODE Solver Substep
  fn step_ode(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  // 5. Collision Pipeline
  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<Self::MotionBvh>;
  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>>;
  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::List<CollisionPair>>;

  /// Stream compaction shrink logic evaluated entirely on the GPU.
  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>>;

  /// Parallel reduction to find the lowest `time_of_impact`.
  /// Returns a tiny buffer of length 1 containing $t_c$.
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>>;

  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    collisions: &Self::List<CollisionPair>,
    force_inelastic: bool,
  ) -> EngineResult<()>;

  // --- CCD Rewind Subsystem ---
  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<Self::Buffer<DynamicBody>>;
  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &mut Self::Buffer<DynamicBody>,
    snapshot: &Self::Buffer<DynamicBody>,
  ) -> EngineResult<()>;

  // --- Write back dynamic state ---
  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    dynamics: &Self::Buffer<DynamicBody>,
    physical_scene: &mut PhysicsScene,
    scene: &mut Scene,
  ) -> EngineResult<()>;
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
