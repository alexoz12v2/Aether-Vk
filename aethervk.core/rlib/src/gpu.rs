//! gpu module.

use crate::{
  gpu::{self, frame::ResourceUploadResult},
  scene::{EntityId, StaticMeshComponent, text::FontAtlas},
  simulation::comet::Texture,
  types::{EngineResult, GpuError, GpuResult},
};
use alloc::sync::{Arc, Weak};
use bitflags::bitflags;
use core::{ffi, hash::Hash};
use heapless::index_map::FnvIndexMap;

pub mod compute_push_constants;
pub mod frame;
pub mod scene_conversion;

pub use self::frame::RenderScene;

/// An opaque task that MUST be executed on the main UI thread.
/// Used to defer window-system-tied destruction (swapchain, surface) from
/// background threads on macOS where MoltenVK requires CAMetalLayer teardown
/// on the main thread.
pub type MainThreadCleanupTask = alloc::boxed::Box<dyn FnOnce() + Send>;

/// Shared queue of tasks pending main-thread execution.
/// Uses `spin::Mutex` for `no_std` compatibility.
pub type MainThreadCleanupQueue = Arc<spin::Mutex<alloc::vec::Vec<MainThreadCleanupTask>>>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderBackendId(pub u64);
pub const NULL_RENDER_BACKEND: RenderBackendId = RenderBackendId(0);
pub const VULKAN_RENDER_BACKEND: RenderBackendId = RenderBackendId(1);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GpuResourceHandle(pub u64);
pub const NULL_GPU_RESOURCE: GpuResourceHandle = GpuResourceHandle(0);

impl GpuResourceHandle {
  pub fn from_raw(raw: u64) -> Self {
    Self(raw)
  }
}

pub mod new_particles {
  // maximum supported subgroup size is 128, and this is a multiple of it
  pub const PCHUNK_SIZE: usize = 256;
  pub const MAX_PARTICLES: usize = 1_000_000;
  pub const MAX_PARTICLES_PER_SYSTEM: usize = 100_000;
  pub const MAX_CHUNKS: usize = MAX_PARTICLES.div_ceil(PCHUNK_SIZE);
  pub const PARTICLE_PAGE_TABLE_HEADER_SIZE: usize = 32;
  pub const PAGE_TABLE_BYTES: u64 =
    (PARTICLE_PAGE_TABLE_HEADER_SIZE + 4 * MAX_PARTICLES_PER_SYSTEM.div_ceil(PCHUNK_SIZE)) as _;

  #[repr(C)]
  #[derive(Debug, Copy, Clone, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
  pub struct ParticleChunk {
    pub position_x: [f32; PCHUNK_SIZE],
    pub position_y: [f32; PCHUNK_SIZE],
    pub position_z: [f32; PCHUNK_SIZE],
    pub velocity_x: [f32; PCHUNK_SIZE],
    pub velocity_y: [f32; PCHUNK_SIZE],
    pub velocity_z: [f32; PCHUNK_SIZE],
    pub inv_mass: [f32; PCHUNK_SIZE],
    pub force_x: [f32; PCHUNK_SIZE],
    pub force_y: [f32; PCHUNK_SIZE],
    pub force_z: [f32; PCHUNK_SIZE],
    pub beta: [f32; PCHUNK_SIZE],
    pub spawn_time: [u32; PCHUNK_SIZE],
  }

  /// Push Constant layout for `dust.vert/frag` shaders
  #[repr(C)]
  #[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
  pub struct DustPushConstants {
    pub global_particle_buffer: u64,
    pub particle_page_table: u64,
    pub view_proj: [f32; 16],
    pub stream_color: [f32; 4],
    pub chunk_offset: u32,
    pub current_time: u32,
    pub max_ttl: f32,
    pub macro_scale: f32,
    pub micro_radius: f32,
    pub num_spots: u32,
    pub dispersion_rate: f32,
    pub _pad: u32,
  }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct CommandBufferHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderableInstanceId(pub u64);

pub trait GpuCometExt {
  fn texture_flags(&self) -> TextureFlags;
}

impl GpuCometExt for crate::simulation::comet::Comet {
  fn texture_flags(&self) -> TextureFlags {
    let mut flags = TextureFlags::empty();
    if self.albedo_map.is_some() {
      flags |= TextureFlags::ALBEDO;
    }
    if self.normal_map.is_some() {
      flags |= TextureFlags::NORMAL;
    }
    if self.roughness_map.is_some() {
      flags |= TextureFlags::ROUGHNESS;
    }
    if self.ao_map.is_some() {
      flags |= TextureFlags::AO;
    }
    flags
  }
}

bitflags! {
  #[repr(C)]
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
  pub struct TextureFlags: u32 {
    const ALBEDO    = 1 << 0;
    const NORMAL    = 1 << 1;
    const ROUGHNESS = 1 << 2;
    const AO        = 1 << 3;
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPushConstants {
  pub map_ptr: u64,
  pub traj_ptr: u64,
  pub view_proj: [f32; 16],
  pub viewport_size: [f32; 2],
  pub _pad: [f32; 2],
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct RationalBezierGpu {
  pub cp0: [f32; 4],
  pub cp1: [f32; 4],
  pub cp2: [f32; 4],
  pub cp3: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
pub struct TrajectoryGpu {
  pub segments_ptr: u64,
  pub _pad0: u64,
  pub color: [f32; 4],
  pub line_width: f32,
  pub texture_id: u32,
  pub _pad1: u64,
}

#[repr(C, align(4))]
#[derive(Copy, Clone)]
pub struct SegmentMapGpu {
  pub trajectory_id: u32,
  pub local_segment_id: u32,
  pub subdivisions: u32,
}

/// `common.glsl` buffer_reference struct definition
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SceneData {
  pub view_proj: [f32; 16],
  pub camera_pos: [f32; 4], // w is padding
  pub sun_pos: [f32; 4],    // w is padding
  pub sun_color: [f32; 4],
  pub window_extent: [f32; 2],
  pub _pad: [f32; 2],
}

/// `common.glsl` buffer_reference struct definition
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct MaterialData {
  pub base_albedo: [f32; 4],    // w is base_roughness
  pub emissive_color: [f32; 4], // w is emissive_intensity
  pub base_ao: f32,
  pub paint_display_mode: u32,
  pub texture_flags: u32,
  pub _pad0: f32,
  /// useful only for spherical grid mode (not used now)
  pub sphere_center_radius: [f32; 4],
  /// useful only for spherical grid mode (not used now)
  pub grid_color_density: [f32; 4],
}

/// `common.glsl` model matrix pointer. This allows us to store all model matrices in a single
/// buffer if needed
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ObjectData {
  pub model: [f32; 16],
}

/// push constants layout for `physical_mesh2.vert/frag`
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct PhysicalMesh2PushConstants {
  // BDA to [`SceneData`]
  pub scene_addr: u64,
  // BDA to [`MaterialData`]
  pub material_addr: u64,
  // BDA to [`ObjectData`]
  pub object_addr: u64,
  pub _pad: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CometSpecializationConstants {
  pub base_albedo_r: f32,
  pub base_albedo_g: f32,
  pub base_albedo_b: f32,
  pub base_roughness: f32,
  pub base_ao: f32,
}

impl Default for CometSpecializationConstants {
  fn default() -> Self {
    Self {
      base_albedo_r: 0.04,
      base_albedo_g: 0.04,
      base_albedo_b: 0.04,
      base_roughness: 0.9,
      base_ao: 1.0,
    }
  }
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
  pub view_proj: [f32; 16],
  pub right_proj11: [f32; 4],
  pub screen_y_win_x: [f32; 4],
  pub relative_cam_pos_win_y: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SkyPushConstants {
  pub inv_view_proj: [f32; 16],
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
  pub color: [f32; 3],
  pub _pad3: f32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GridPushConstants {
  pub view_proj: [f32; 16],
  pub camera_pos: [f32; 3],
  pub near_plane: f32,
  pub far_plane: f32,
  pub density: f32,
  pub _pad1: [f32; 2],
  pub grid_color: [f32; 3],
  pub _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SphereGizmoDataGpu {
  pub model: [f32; 16],
  pub radius: f32,
  pub subdivisions: f32,
  pub _pad: [f32; 2],
}

/// push constant layout for `sphere_gizmo.frag/vert`
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Zeroable, bytemuck::Pod)]
pub struct SphereGizmoPushConstants {
  // Must match GLSL: layout(push_constant, std430) uniform PushConstants {
  //     mat4 viewProj;             // offset 0,  64 bytes
  //     SphereGizmoArray gizmoPtr; // offset 64,  8 bytes
  //     uint64_t _pad;             // offset 72,  8 bytes
  // };
  pub view_proj: [f32; 16], // 64 bytes at offset 0
  pub gizmo_ptr: u64,       //  8 bytes at offset 64
  pub _pad: u64,            //  8 bytes padding to align block to 16 bytes (80 total)
}

/// Push constants for the depth-compositing fullscreen pass.
/// Must match `composite.frag` layout: macroNear, macroFar, microNear, microFar.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CompositePushConstants {
  pub macro_near: f32,
  pub macro_far: f32,
  pub micro_near: f32,
  pub micro_far: f32,
  pub macro_scale: f32,
  pub micro_scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GizmoPushConstants {
  pub view_proj: [f32; 16],
  pub scale: f32,
  pub instance_id: u32,
  pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct TextGlyphGpu {
  pub pos: [f32; 2],
  pub size: [f32; 2],
  pub uv_bounds: [f32; 4],
  pub color: [f32; 4],
  pub texture_id: u32,
  pub style: u32,
  pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Text2PushConstants {
  pub glyphs_ptr: u64,
  pub _pad0: u64,
  pub view_proj: [[f32; 4]; 4],
}

impl RenderableInstanceId {
  pub fn from_physical_mesh(asset_hash: u64) -> Self {
    Self(asset_hash)
  }
}

impl From<RenderableInstanceId> for GpuResourceHandle {
  fn from(value: RenderableInstanceId) -> Self {
    Self(value.0)
  }
}

// Allow the host application to configure the core assets path uniformly
pub static ASSET_DIR: parking_lot::RwLock<Option<alloc::string::String>> =
  parking_lot::RwLock::new(None);

#[cfg(test)]
pub fn set_asset_dir_for_tests() {
  #[cfg(not(windows))]
  unsafe {
    let mut rlim: libc::rlimit = core::mem::zeroed();
    if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlim) == 0 {
      rlim.rlim_cur = rlim.rlim_max;
      libc::setrlimit(libc::RLIMIT_NOFILE, &rlim);
    }
  }

  if ASSET_DIR.read().is_some() {
    return;
  }
  let mut home_dir = std::env::current_exe().unwrap_or_default();
  let mut iter = 0;
  while !home_dir.join("assets").is_dir() && iter < 32 {
    home_dir.pop();
    iter += 1;
  }
  if !home_dir.join("assets").is_dir() {
    if std::path::Path::new("assets").is_dir() {
      home_dir = std::env::current_dir().unwrap_or_default();
    } else if let Ok(env_path) = std::env::var("ASSET_DIR") {
      *ASSET_DIR.write() = Some(env_path);
      return;
    }
  }
  if home_dir.join("assets").is_dir() {
    *ASSET_DIR.write() = Some(home_dir.join("assets").to_str().unwrap().to_string());
  } else {
    // Fallback using compile-time CARGO_MANIFEST_DIR just in case
    let mut manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.pop();
    manifest_dir.pop();
    if manifest_dir.join("assets").is_dir() {
      *ASSET_DIR.write() = Some(manifest_dir.join("assets").to_str().unwrap().to_string());
    }
  }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferSyncInfoStageMask {
  #[default]
  TopBottom, // signal at bottom, wait on top
  Transfer,
  VertexAttributeInput,
}

/// Information about the synchronization payload after submitting a command buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommandBufferSyncInfo {
  pub timeline_semaphore: u64, // Opaque handle for the backend
  pub timeline_value: u64,
  pub wait_stage_mask: CommandBufferSyncInfoStageMask,
}

pub trait CommandBuffer: Send + Sync {
  fn submit(&mut self) -> EngineResult<Option<CommandBufferSyncInfo>>;
}

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

impl Rect2D {
  pub fn from_extent(extent: [u32; 2]) -> Self {
    Self {
      offset: [0, 0],
      extent,
    }
  }
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

impl Viewport {
  pub fn from_extent(extent: [u32; 2]) -> Self {
    Self {
      x: 0.0,
      y: 0.0,
      width: extent[0] as f32,
      height: extent[1] as f32,
      min_depth: 0.0,
      max_depth: 1.0,
    }
  }
}

pub const UI_FLAG_HAS_CLIP: u32 = 1 << 0;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct UiElementGpu {
  pub bounds: [f32; 4],
  pub clip_rect: [f32; 4],
  pub color_start: [f32; 4],
  pub color_end: [f32; 4],
  pub color_border: [f32; 4],
  pub color_shadow: [f32; 4],
  pub border_radius: [f32; 4],
  pub shadow_params: [f32; 4],
  pub gradient_dir: [f32; 2],
  pub border_width: f32,
  pub texture_id: u32,
  pub flags: u32,
  pub opacity: f32,
  pub rotation: f32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct UiPushConstants {
  pub elements_ptr: u64,
  pub _pad0: u64,
  pub view_proj: [[f32; 4]; 4],
}

pub struct UiBatchCall {
  pub elements_ptr: u64,
  pub total_elements: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BackgroundPushConstants {
  pub color_top: [f32; 4],
  pub color_bottom: [f32; 4],
}

pub struct Text2BatchCall {
  pub glyphs_ptr: u64,
  pub total_glyphs: u32,
}

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum NativeGpuProperty {
  VulkanMetalDeviceId = 0,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArchetypeId {
  Sun,
  Mesh,
  Billboard,
  Cursor,
  Marker,
  Measurement,
  Sky,
  Grid,
  Text,
  Gizmo,
  SphereGizmo,
  Trajectory,
  Ui,
  Background,
  Particles,
}

pub trait RenderDevice: Send + Sync + core::any::Any {
  fn as_any(&self) -> &dyn core::any::Any;

  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void>;

  fn print_info(&self) -> alloc::string::String;

  fn dump_memory_stats(&self);

  fn context_id(&self) -> u64;

  fn subgroup_size(&self) -> u32;

  /// Returns `true` when the underlying Vulkan device is a CPU software
  /// renderer (e.g. Lavapipe / llvmpipe).  Used by callers that need to adapt
  /// workgroup sizes or scheduling behaviour to the CPU execution model.
  fn is_cpu_device(&self) -> bool;

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

  /// Creates the surface and initial swapchain
  fn create_presentation_engine(
    &self,
    params: &PresentationEngineParams,
  ) -> GpuResult<PresentationEngineHandle>;

  /// Destroys a swapchain
  fn destroy_presentation_engine(&self, handle: PresentationEngineHandle) -> GpuResult<()>;

  /// Drains and executes all pending cleanup tasks that require main-thread affinity.
  ///
  /// On macOS, MoltenVK translates Vulkan swapchain/surface destruction into
  /// Core Animation `CAMetalLayer` modifications. Apple **strictly** requires these
  /// to happen on the main UI thread. Performing them on a background thread causes:
  ///
  /// > `warning, deleted thread with uncommitted CATransaction; set
  /// > CA_DEBUG_TRANSACTIONS=1 in environment to log backtraces, or set
  /// > CA_ASSERT_MAIN_THREAD_TRANSACTIONS=1 to abort when an implicit transaction
  /// > isn't created on a main thread.`
  ///
  /// **Call this periodically from the main UI thread** (e.g., in `winit`'s
  /// `AboutToWait` event, or from Avalonia's UI-thread tick).
  ///
  /// For windowless presentation engines this is always a no-op.
  fn process_main_thread_cleanup_queue(&self) -> GpuResult<()>;

  /// Flushes **all** remaining window-tied resources from the device, executing
  /// their destruction tasks immediately on the calling thread.
  ///
  /// **Must be called from the main thread after the render thread has exited
  /// but before `Device` is dropped.** This ensures `destroy_swapchain` and
  /// `destroy_surface` run before `destroy_device`.
  ///
  /// For windowless-only usage (tests, Avalonia headless), this is a no-op drain.
  fn flush_main_thread_cleanup_queue(&self) -> GpuResult<()>;

  /// Allows caller to explicitly trigger a resize/recreation
  fn resize_presentation_engine(
    &self,
    handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()>;

  fn get_presentation_engine_extent(&self, handle: PresentationEngineHandle)
  -> GpuResult<[u32; 2]>;

  fn is_presentation_engine_windowless(&self, handle: PresentationEngineHandle) -> GpuResult<bool>;

  /// Acquires the next image. If it returns NeedsRecreation, the caller should
  /// discard the frame, call resize, and try again next frame
  fn acquire_next_image(&self, handle: PresentationEngineHandle) -> GpuResult<AcquireResult>;

  fn cancel_acquired_image(
    &self,
    handle: PresentationEngineHandle,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()>;

  fn get_physical_mesh2_resources(
    &self,
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_physical_mesh2_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    asset_hash: u64,
    component: &StaticMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult>;
  fn draw_physical_mesh2(
    &self,
    cmd_buffer: CommandBufferHandle,
    pipeline: PipelineKey,
    buffers: GpuResourceHandle,
    camera: &crate::gpu::frame::CameraRenderData,
    sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    sun_color: [f32; 4],
    window_extent: [f32; 2],
    handle: PresentationEngineHandle,
    draw_call: &crate::gpu::frame::DrawCall,
  ) -> GpuResult<()>;

  /// Generates the background sky image using compute shader
  fn generate_sky(&self) -> GpuResult<()>;

  /// Returns resources for the billboard rendering
  fn get_billboard_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  fn create_billboard_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Returns resources for the cursor rendering
  fn get_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  fn create_cursor_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  /// Returns resources for marker rendering
  fn get_measurement_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_measurement_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  fn get_marker_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_marker_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;

  fn get_gizmo_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_gizmo_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn update_gizmo_instance(
    &self,
    entity: EntityId,
    model: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    handle: PresentationEngineHandle,
  ) -> GpuResult<u32>;

  fn allocate_sphere_gizmo_instance(&self, entity: EntityId) -> GpuResult<u32>;

  fn free_sphere_gizmo_instance(&self, entity: EntityId) -> GpuResult<()>;

  fn upload_sphere_gizmos_batch(
    &self,
    cmd_buffer: CommandBufferHandle,
    gizmos: &[(u32, crate::gpu::SphereGizmoDataGpu)],
  ) -> GpuResult<Option<crate::gpu::frame::SphereGizmoBatchCall>>;

  // --- Removed get_or_create_particle_resources ---
  fn upload_trajectories(
    &self,
    cmd_buffer: CommandBufferHandle,
    trajectories: &[(
      crate::scene::EntityId,
      crate::scene::trajectory::TrajectoryComponent,
      aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    )],
  ) -> GpuResult<Option<crate::gpu::frame::TrajectoryBatchCall>>;

  fn upload_ui(
    &self,
    cmd_buffer: CommandBufferHandle,
    ui_elements: &[crate::gpu::UiElementGpu],
  ) -> GpuResult<Option<crate::gpu::UiBatchCall>>;

  fn clear_depth(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn upload_text2(
    &self,
    cmd_buffer: CommandBufferHandle,
    glyphs: &[crate::gpu::TextGlyphGpu],
  ) -> GpuResult<Option<crate::gpu::Text2BatchCall>>;

  fn get_trajectory_pipeline_key(&self, handle: PresentationEngineHandle)
  -> GpuResult<PipelineKey>;
  fn get_sun_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_sky_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_background_pipeline_key(&self, handle: PresentationEngineHandle)
  -> GpuResult<PipelineKey>;
  fn get_grid_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;

  /// Given FontAtlas (moved), try to allocate a rasterized representation of it
  /// for the render device. Returns internal id used by RenderDevice (as descriptor index)
  /// Given `hash` should also be kept by caller, in case removal is desired
  fn allocate_rasterized_font_atlas(
    &self,
    cmd: CommandBufferHandle,
    hash: u64,
    font_atlas: alloc::sync::Arc<FontAtlas>,
  ) -> GpuResult<u32>;

  fn free_rasterized_font_atlas(&self, hash: u64, font_atlas_id: u32) -> GpuResult<()>;

  /// Presents the image. Takes semaphore signaled by a rendering command buffer
  fn present(
    &self,
    handle: PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<SwapchainStatus>;

  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
    task_id: Option<u64>,
  ) -> GpuResult<()>;

  fn get_command_buffer(&self) -> GpuResult<CommandBufferHandle>;

  fn set_command_buffer_presentation_engine(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn begin_command_buffer(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// responsible to acquire an image and store it in the associated command buffer structure
  fn begin_render_pass(
    &self,
    cmd_buffer: CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()>;

  /// Begin a 3-subpass compositing render pass for multi-scale rendering.
  /// Subpass 0: macro layer (color=[2], depth=[3])
  /// Subpass 1: micro layer (color=[4], depth=[5])
  /// Subpass 2: composite (color=[0], depth=[1], input=[2,3,4,5])
  /// The caller must call `next_subpass()` between each subpass.
  fn begin_compositing_render_pass(
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
  fn add_billboard_texture(
    &self,
    cmd_buffer: CommandBufferHandle,
    texture_id: u64,
    texture: &Texture,
    current_frame: u64,
  ) -> GpuResult<u32>;

  fn push_constants_raw(
    &self,
    cmd_buffer: CommandBufferHandle,
    archetype: ArchetypeId,
    push_constants_bytes: &[u8],
  ) -> GpuResult<()>;

  fn draw_indexed(&self, cmd_buffer: CommandBufferHandle, index_count: u32) -> GpuResult<()>;

  fn draw(&self, cmd_buffer: CommandBufferHandle, vertex_count: u32) -> GpuResult<()>;

  fn draw_instanced(
    &self,
    cmd_buffer: CommandBufferHandle,
    vertex_count: u32,
    instance_count: u32,
  ) -> GpuResult<()>;

  fn draw_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_buffer: GpuResourceHandle,
    offset: u64,
    draw_count: u32,
    stride: u32,
  ) -> GpuResult<()>;

  fn update_sun(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    resolution: (u32, u32, u32),
    radius: f32,
  ) -> GpuResult<()>;

  fn prepare_billboard_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn push_sphere_gizmo_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    constants: &SphereGizmoPushConstants,
  ) -> GpuResult<()>;

  fn prepare_sphere_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn get_sphere_gizmo_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey>;

  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  /// Allocates Descriptor (not image, that is done in `generate_sky`) and updates if not done yet
  fn prepare_sun_for_render(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity: EntityId,
  ) -> GpuResult<()>;

  fn prepare_trajectory_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn prepare_ui_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn prepare_sky_for_render(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  fn prepare_text2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  /// Advance to the next subpass within the current render pass.
  /// Uses VK_KHR_create_renderpass2's cmd_next_subpass2.
  fn next_subpass(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Draw the fullscreen compositing triangle (3 vertices, no vertex buffer).
  /// Binds the composite pipeline, pushes near/far constants, and draws.
  fn draw_composite(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
    constants: &CompositePushConstants,
  ) -> GpuResult<()>;

  fn end_render_pass(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Call this right after `end_render_pass`. It allocates a staging buffer
  /// and natively injects the GPU Image-to-Buffer copy directly into your main frame.
  fn record_windowless_download(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: u64,
  ) -> GpuResult<()>;

  /// Call this once `is_task_completed(task_id)` for a `record_windowless_download` task is true.
  /// It completes the CPU memory copy instantly without touching any Vulkan Queues.
  fn read_windowless_download(&self, task_id: u64, buffer: &mut [u8]) -> GpuResult<()>;

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
    sync_infos: &[CommandBufferSyncInfo],
  ) -> GpuResult<()>;

  fn wire_callbacks(&self, pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) -> GpuResult<()>;

  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool>;

  fn create_task(&self) -> u64;

  fn fail_task(&self, task_id: u64, error: GpuError);

  fn success_task(&self, task_id: u64);
  fn prepare_background_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;
}

macro_rules! implement_render_device_ext {
  (
    $(
        // Match the desired method name, the target ArchetypeId variant, and the specific push constant type
        fn $method_name:ident($archetype:ident, $struct_ty:ty);
    )*
  ) => {
    /// Extension trait for `RenderDevice` such that we can give more ergonomic methods without
    /// losing dyn compatibility
    pub trait RenderDeviceExt {
      /// Generically push any struct as push constants.
      /// Struct should be `#[repr(C)]`
      fn push_constants<T>(
        &self,
        cmd_buffer: crate::gpu::CommandBufferHandle,
        archetype: ArchetypeId,
        push_constants: &T,
      ) -> GpuResult<()>;

      // Generate a trait method for each rule
      $(
        fn $method_name(
          &self,
          cmd_buffer: crate::gpu::CommandBufferHandle,
          push_constants: &$struct_ty,
        ) -> GpuResult<()>;
      )*
    }

    impl<R: RenderDevice + ?Sized> RenderDeviceExt for R {
      fn push_constants<T>(
        &self,
        cmd_buffer: crate::gpu::CommandBufferHandle,
        archetype: ArchetypeId,
        push_constants: &T,
      ) -> GpuResult<()> {
        // Convert generic struct to bytes
        let bytes = unsafe {
            core::slice::from_raw_parts(
                push_constants as *const T as *const u8,
                core::mem::size_of::<T>(),
            )
        };
        self.push_constants_raw(cmd_buffer, archetype, bytes)
      }

      // Generate the boilerplate implementation for each rule
      $(
        fn $method_name(
          &self,
          cmd_buffer: crate::gpu::CommandBufferHandle,
          push_constants: &$struct_ty,
        ) -> GpuResult<()> {
          self.push_constants(cmd_buffer, ArchetypeId::$archetype, push_constants)
        }
      )*
    }
  };
}

// Call the macro with your specific methods, archetype enum variants, and structs
implement_render_device_ext! {
  fn push_sun_constants(Sun, SunPushConstants);
  fn push_constants_mesh2(Mesh, PhysicalMesh2PushConstants);
  fn push_billboard_constants(Billboard, BillboardPushConstants);
  fn push_cursor_constants(Cursor, CursorPushConstants);
  fn push_marker_constants(Marker, MarkerPushConstants);
  fn push_measurement_constants(Measurement, MeasurementPushConstants);
  fn push_sky_constants(Sky, SkyPushConstants);
  fn push_grid_constants(Grid, GridPushConstants);
  fn push_gizmo_constants(Gizmo, GizmoPushConstants);
  fn push_text2_constants(Text, Text2PushConstants);
  fn push_trajectory_constants(Trajectory, TrajectoryPushConstants);
  fn push_ui_constants(Ui, UiPushConstants);
  fn push_background_constants(Background, BackgroundPushConstants);

  // to add new archetypes, add one line here:
}

/// An RAII guard ensuring the command buffer is always submitted.
pub struct ScopedCommandBuffer<'a> {
  device: &'a dyn RenderDevice,
  cmd_buffer: CommandBufferHandle,
  task_id: Option<u64>,
  sync_infos: heapless::Vec<CommandBufferSyncInfo, 4>,
  submitted: bool,
}

impl<'a> ScopedCommandBuffer<'a> {
  pub fn new(
    device: &'a dyn RenderDevice,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
  ) -> GpuResult<Self> {
    device.begin_command_buffer(cmd_buffer)?;
    Ok(Self {
      device,
      cmd_buffer,
      task_id,
      sync_infos: heapless::Vec::new(),
      submitted: false,
    })
  }

  /// Attaches synchronization info to this command buffer scope.
  pub fn add_sync_info(&mut self, sync_info: CommandBufferSyncInfo) {
    let _ = self.sync_infos.push(sync_info);
  }

  pub fn cmd(&self) -> CommandBufferHandle {
    self.cmd_buffer
  }

  /// Explicitly submits the command buffer.
  pub fn submit(mut self) -> GpuResult<()> {
    self.submitted = true;
    let res = self
      .device
      .submit_command_buffer(self.cmd_buffer, self.task_id, &self.sync_infos);
    if let Err(e) = &res {
      if let Some(task_id) = self.task_id {
        self.device.fail_task(task_id, e.clone());
      }
    }
    res
  }
}

impl<'a> Drop for ScopedCommandBuffer<'a> {
  fn drop(&mut self) {
    if !self.submitted {
      // Force submission on early exit/panic. Result is ignored to prevent double panics.
      let res = self
        .device
        .submit_command_buffer(self.cmd_buffer, self.task_id, &self.sync_infos);
      if let Err(e) = res {
        if let Some(task_id) = self.task_id {
          self.device.fail_task(task_id, e);
        }
      }
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

pub struct FrameCancelGuard<'a> {
  device: &'a dyn RenderDevice,
  engine: PresentationEngineHandle,
  acquire_result: Option<AcquireResult>,
}

impl<'a> FrameCancelGuard<'a> {
  pub fn new(
    device: &'a dyn RenderDevice,
    engine: PresentationEngineHandle,
    acquire_result: AcquireResult,
  ) -> Self {
    Self {
      device,
      engine,
      acquire_result: Some(acquire_result),
    }
  }

  pub fn defuse(mut self) {
    self.acquire_result = None;
  }
}

impl<'a> Drop for FrameCancelGuard<'a> {
  fn drop(&mut self) {
    if let Some(ar) = self.acquire_result.take() {
      // Guard fell out of scope without being defused. An error happened!
      // Cancel the frame cleanly to avoid swapchain leaks and deadlocks.
      let _ = self
        .device
        .cancel_acquired_image(self.engine, ar.image_index, ar.frame_index as u32);
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

  /// Returns which Vulkan surface extensions were actually enabled on the
  /// instance.  On Linux the RenderDoc capture layer may strip
  /// `VK_KHR_wayland_surface`, making a Wayland window unusable.  Call this
  /// *before* creating a winit window so the right backend can be forced.
  ///
  /// The default implementation returns all-supported so that backends that
  /// don't track per-extension surface support (D3D12, Metal) don't need to
  /// override this.
  #[cfg(target_os = "linux")]
  fn linux_surface_support(&self) -> crate::gpu_backends::vulkan::instance::LinuxSurfaceSupport {
    crate::gpu_backends::vulkan::instance::LinuxSurfaceSupport {
      wayland: true,
      xcb: true,
      xlib: true,
    }
  }
}

// NOTE: This is a box like type, so we don't need to box it when returning it to cdylib,
// we can instead use the ManualDrop mechanism
#[derive(Clone)]
pub struct RenderFrontend {
  backend: Arc<parking_lot::RwLock<dyn RenderContext + 'static>>,
}

pub type WeakRenderFrontend = Weak<parking_lot::RwLock<dyn RenderContext + 'static>>;
pub trait WeakRenderFrontendExt {
  fn as_frontend(&self) -> Option<RenderFrontend>;
}
impl WeakRenderFrontendExt for WeakRenderFrontend {
  fn as_frontend(&self) -> Option<RenderFrontend> {
    // aethervk_oshal_rlib::log!("--- as_frontend ---");
    self.upgrade().map(|s| RenderFrontend { backend: s })
  }
}

impl core::ops::Deref for RenderFrontend {
  type Target = Arc<parking_lot::RwLock<dyn RenderContext + 'static>>;

  fn deref(&self) -> &Self::Target {
    &self.backend
  }
}

unsafe impl Sync for RenderFrontend {}
unsafe impl Send for RenderFrontend {}

impl RenderFrontend {
  pub fn weak_self(&self) -> WeakRenderFrontend {
    Arc::downgrade(&self.backend)
  }

  /// Executes a closure with the specified render device safely.
  /// We use a trampoline to pass a safe Rust closure through the C-style
  /// `deref_device_and` `*mut c_void` parameter.
  pub fn with_device<F, R>(&self, device_id: RenderDeviceHandle, f: F) -> GpuResult<R>
  where
    F: FnOnce(&dyn RenderDevice) -> GpuResult<R>,
  {
    // 1. Prepare storage for our generic result since `deref_device_and`
    // strictly expects the callback to return `GpuResult<()>`.
    let mut result: Option<GpuResult<R>> = None;

    // 2. Bundle the closure and the result destination together.
    let mut payload = (Some(f), &mut result);
    let p_user_data = &mut payload as *mut _ as *mut core::ffi::c_void;

    // 3. Define the C-compatible trampoline function.
    fn trampoline<F, R>(
      dev: &dyn RenderDevice,
      p_user_data: *mut core::ffi::c_void,
    ) -> GpuResult<()>
    where
      F: FnOnce(&dyn RenderDevice) -> GpuResult<R>,
    {
      // Cast the void pointer back to our known payload type
      let payload_ptr = p_user_data as *mut (Option<F>, &mut Option<GpuResult<R>>);

      // Unsafe block is required to dereference the raw pointer, but it's
      // sound here because the payload lives in the parent stack frame.
      let payload = unsafe { &mut *payload_ptr };

      // Take the closure out of the Option so we can consume it (FnOnce)
      let closure = payload.0.take().expect("Closure called multiple times");

      // Execute the closure and store the actual result
      *payload.1 = Some(closure(dev));

      // Return a dummy success to satisfy the `fn(...) -> GpuResult<()>` signature
      Ok(())
    }

    // 4. Lock the backend and execute the trampoline
    let backend_guard = self.backend.read();
    let call_result = backend_guard.deref_device_and(device_id, p_user_data, trampoline::<F, R>);

    // 5. If the device was found and the callback executed, return our captured result.
    // Otherwise, return None (device not found) or propagate a backend internal error.
    match call_result {
      Some(Ok(())) => unsafe { result.unwrap_unchecked() },
      Some(Err(e)) => Err(e),
      None => Err(GpuError::DeviceLost),
    }
  }
  pub fn take_and<T>(
    &self,
    f: impl FnOnce(&dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_read() {
      Some(guard) => Some(f(&*guard)),
      None => None,
    }
  }

  pub fn take_mut_and<T>(
    &self,
    f: impl FnOnce(&mut dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_write() {
      Some(mut guard) => Some(f(&mut *guard)),
      None => None,
    }
  }

  /// Returns which Linux Vulkan surface extensions were actually enabled on
  /// the instance.  Must be called after `SimulationContext::startup` but
  /// **before** creating a winit window, so the correct windowing backend
  /// (Wayland vs XCB) can be selected.
  #[cfg(target_os = "linux")]
  pub fn linux_surface_support(
    &self,
  ) -> crate::gpu_backends::vulkan::instance::LinuxSurfaceSupport {
    self.backend.read().linux_surface_support()
  }
}

// Boxing mechanism used by factory method in `gpu_backends` `new_render_frontend`
impl<T> From<T> for RenderFrontend
where
  T: RenderContext + 'static,
{
  fn from(value: T) -> Self {
    RenderFrontend {
      backend: Arc::new(parking_lot::RwLock::new(value)),
    }
  }
}

#[derive(Default, Debug, Copy, Clone, Eq, Ord, PartialOrd, PartialEq, Hash)]
pub struct PresentationEngineHandle(pub u64);

impl PresentationEngineHandle {
  pub fn is_valid(self) -> bool {
    self.0 != 0
  }
}

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
  pub swapchain_generation: u64,
}

/// Discriminates the windowing system whose handles are stored in [`OpaqueNativeHandleInfo`].
/// This is used at runtime so the Vulkan backend can choose the correct surface creation path
/// without relying on compile-time Cargo features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum NativeHandleType {
  /// No windowing system / headless / windowless.
  Unknown = 0,
  /// Win32 HWND + HINSTANCE.
  Win32 = 1,
  /// Wayland `wl_display` + `wl_surface`.
  Wayland = 2,
  /// Xlib `Display` + `Window`.
  Xlib = 3,
  /// XCB `xcb_connection_t` + `xcb_window_t`.
  Xcb = 4,
  /// macOS `CAMetalLayer`.
  Metal = 5,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
/// Opaque, platform-agnostic window handle passed across the FFI boundary.
/// - `ptr0` / `ptr1` carry the platform-specific pointers (see [`NativeHandleType`]).
/// - `handle_type` tells the Vulkan backend which Vulkan surface extension to use.
pub struct OpaqueNativeHandleInfo {
  pub ptr0: *mut ffi::c_void,
  pub ptr1: *mut ffi::c_void,
  pub handle_type: NativeHandleType,
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
  pub buffer_count: u32,
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
        handle_type: NativeHandleType::Unknown,
      },
      buffer_count: 3,
    }
  }
}

/// GPU hardware subgroup (warp/SIMD) size, clamped to the valid range for dispatch.
/// Powers of two from 4 to 128. Lavapipe reports 8; Apple Silicon 32; AMD 64; Nvidia 32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupSize {
  Size4 = 4,
  Size8 = 8,
  Size16 = 16,
  Size32 = 32,
  Size64 = 64,
  Size128 = 128,
}
