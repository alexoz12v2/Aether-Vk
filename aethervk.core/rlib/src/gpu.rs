use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::scene::text::{FontAtlas, GlyphInfo};
use crate::simulation::comet::Texture;
use crate::types::{EngineResult, GpuError, GpuResult};
use crate::{
  gpu::frame::ResourceUploadResult,
  scene::{EntityId, PhysicalMeshComponent, TransformComponent},
};
use ab_glyph::PxScale;
use aethervk_oshal_rlib::os::time::timeus_t;
use ahash::AHasher;
#[cfg(debug_assertions)]
use alloc::string::String;
use alloc::sync::Arc;
use alloc::sync::Weak;
use bitflags::bitflags;
use core::{
  ffi,
  hash::{Hash, Hasher},
};
use heapless::index_map::FnvIndexMap;

pub use super::gpu_backends::*;

pub mod frame;
pub mod scene_conversion;

pub use self::frame::RenderScene;

pub type RwLock<T> = spin::rwlock::RwLock<T>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderBackendId(pub u64);
pub const NULL_RENDER_BACKEND: RenderBackendId = RenderBackendId(0);
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
pub struct PushConstants {
  pub model_view_proj: [[f32; 4]; 4],
  pub model: [[f32; 4]; 4],
  pub sun_pos: [f32; 3],
  pub texture_flags: TextureFlags,
  pub sun_color: [f32; 4],
  pub camera_pos: [f32; 3],
  pub emissive_intensity: f32,
  pub emissive_color: [f32; 3],
  pub _unused_pad: u32,
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
  pub view: [f32; 16],
  pub view_proj: [f32; 16],
  pub model: [f32; 16],
  pub cursor_size: f32,
  pub _padding: f32,
  pub window_extent: [f32; 2],
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
#[derive(Debug, Clone, Copy)]
pub struct BvhPushConstants {
  pub mvp_arr: [f32; 16],
  pub center_type: [f32; 4],
  pub extents_arr: [f32; 4],
  pub axes_x: [f32; 4],
  pub axes_y: [f32; 4],
  pub axes_z: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ParticlePushConstants {
  pub view_proj: [f32; 16],
  pub camera_up: [f32; 3],
  pub _pad0: f32,
  pub camera_right: [f32; 3],
  pub _pad1: f32,
  pub color: [f32; 4],
  pub radius: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GizmoPushConstants {
  pub view_proj: [f32; 16],
  pub scale: f32,
  pub instance_id: u32,
}

// TODO remove on text rendering v2
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextPushConstants {
  pub pos: [f32; 2],
  pub scale: [f32; 2],
  pub color: [f32; 4],
  pub uv_bounds: [f32; 4],
  pub texture_id: u32,
}

impl TextPushConstants {
  pub(crate) fn from_glyph(
    glyph: &GlyphInfo,
    cursor_position: [f32; 2],
    screen_extent: [f32; 2],
    desired_points: f32,
    atlas_scale: PxScale,
    texture_id: u32,
    color: [f32; 4],
  ) -> Self {
    Self {
      pos: glyph.screen_position(cursor_position, screen_extent, desired_points, atlas_scale),
      scale: glyph.screen_size(screen_extent, desired_points, atlas_scale),
      color,
      uv_bounds: glyph.uv_bounds(),
      texture_id,
    }
  }
}

impl RenderableInstanceId {
  pub fn from_physical_mesh(entity_id: EntityId) -> Self {
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

#[repr(u32)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum NativeGpuProperty {
  VulkanMetalDeviceId = 0,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArchetypeId {
  Sun,
  PhysicalMesh,
  Billboard,
  Cursor,
  Marker,
  Measurement,
  Sky,
  Grid,
  Minimap,
  Text,
  Bvh,
  Particle,
  Gizmo,
}

/// `RenderCompute` bridges the gap between purely physical workloads (`Kernels`) and
/// presentation workloads (`RenderDevice`). It handles compute shaders that are
/// strictly associated with visual aspects (e.g. `skygen.comp`, `sungen.comp`, screen-space reflections,
/// bloom, volumetric fog calculations) rather than physics or logic.
///
/// **Implementation Guidelines & Architecture:**
/// 1.  **Queue Ownership:** Since `Kernels` often runs on a dedicated async-compute queue to prevent
///     stalling the graphics pipeline, any shared resources (like a 3D texture generated by `RenderCompute`)
///     would require explicit `VkImageMemoryBarrier` queue family ownership transfers if they were
///     generated on the compute queue but consumed on the graphics queue.
/// 2.  **Execution Location:** To avoid complex queue transfers, `RenderCompute` should be implemented
///     by `RenderDevice` directly, executing on the Graphics Queue. This ensures that assets like the
///     sun volume or sky map are natively available for fragment shaders in the same queue family.
/// 3.  **Synchronization:** Methods in this trait should accept a `CommandBufferHandle` and internally
///     issue pipeline barriers (`vkCmdPipelineBarrier`) to transition image layouts from
///     `VK_IMAGE_LAYOUT_GENERAL` (for compute writing) to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
///     (for fragment reading) before the graphics passes begin.
/// 4.  **Resource Allocation:** Any allocations (like `sunVolume`) should be handled internally or via
///     the `RenderDevice` resource allocator to ensure they are bound to the correct descriptor sets
///     used by the graphics pipeline.
pub trait RenderCompute: Send + Sync {
  /// Generates the volumetric data for the sun and transitions the output image for fragment shader consumption.
  fn dispatch_sun_volume_generation(
    &self,
    cmd_buffer: CommandBufferHandle,
    resolution: (u32, u32, u32),
  ) -> GpuResult<()>;

  /// Generates the procedural skybox (e.g. using octahedral mapping) and transitions the output image.
  fn dispatch_sky_generation(
    &self,
    cmd_buffer: CommandBufferHandle,
    resolution: (u32, u32),
  ) -> GpuResult<()>;

  // Future visual compute passes can be added here (e.g. post-processing, light culling)
}

pub trait RenderDevice: Send + Sync + core::any::Any {
  fn as_any(&self) -> &dyn core::any::Any;

  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void>;

  fn print_info(&self) -> String;

  fn context_id(&self) -> u64;

  fn subgroup_size(&self) -> u32;

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

  // TODO: see how to refactor get_or_create functions
  /// Returns (pipeline, vertex_buffer, index_buffer)
  fn get_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_physical_mesh_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult>;

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
  ) -> GpuResult<u32>;

  // --- Removed get_or_create_particle_resources ---

  /// Uploads particle systems into the mega-buffers. Should be called before rendering.
  fn upload_particle_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::ParticleDrawCall],
  ) -> GpuResult<()>;

  /// Draws a particle system using the mega-buffer
  fn draw_particle_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()>;

  fn get_particle_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_sun_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_sky_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_grid_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_bvh_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;

  /// Given FontAtlas (moved), try to allocate a rasterized representation of it
  /// for the render device. Returns internal id used by RenderDevice (as descriptor index)
  /// Given `hash` should also be kept by caller, in case removal is desired
  fn allocate_rasterized_font_atlas(
    &self,
    cmd: CommandBufferHandle,
    hash: u64,
    font_atlas: FontAtlas,
  ) -> GpuResult<u32>;

  fn free_rasterized_font_atlas(&self, hash: u64, font_atlas_id: u32) -> GpuResult<()>;

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
    task_id: Option<u64>,
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
  /// TODO rework to 1) not take pipeline key 2) support multiple archetypes which use buffers
  fn bind_buffers(
    &self,
    cmd_buffer: CommandBufferHandle,
    pipeline: PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()>;

  fn push_constants_raw(
    &self,
    cmd_buffer: CommandBufferHandle,
    archetype: ArchetypeId,
    push_constants_bytes: &[u8],
  ) -> GpuResult<()>;

  fn draw_indexed(&self, cmd_buffer: CommandBufferHandle, index_count: u32) -> GpuResult<()>;

  fn draw(&self, cmd_buffer: CommandBufferHandle, vertex_count: u32) -> GpuResult<()>;

  fn draw_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_buffer: GpuResourceHandle,
    offset: u64,
    draw_count: u32,
    stride: u32,
  ) -> GpuResult<()>;

  // TODO move to kernels trait
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
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn prepare_bvh_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  /// Allocates Descriptor (not image, that is done in `generate_sky`) and updates if not done yet
  /// TODO probably move into bridge between Kernels and RenderDevice when Kernels has generates_sky
  /// TODO remove entity. Support for only one sun
  fn prepare_sun_for_render(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity: EntityId,
  ) -> GpuResult<()>;

  /// Allocates Descriptor (not image, that is done in `generate_sky`) and updates if not done yet
  /// TODO probably move into bridge between Kernels and RenderDevice when Kernels has generates_sky
  fn prepare_particle_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  fn prepare_sky_for_render(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Screen extent should be the chosen presentation engine extent to correctly display screen size and position
  /// `atlas_id` is composed of the `hash` and internal id for the font atlas
  fn prepare_text_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()>;

  // TODO move in frame as a ui rendering
  #[deprecated]
  fn render_minimap(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
    player_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    max_distance: f32,
    planets: &[(
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
      f32,
      [f32; 4],
    )],
    screen_extent: [f32; 2],
  ) -> GpuResult<()>;

  fn render_ui_rect(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
  ) -> GpuResult<()>;

  // TODO instead of rendering a character at a time, we should pass letters as vertices and use 4 instances to create a quad. vertex data should have the necessary glyph position/size and texture id. This means that we need a "streaming buffer" (See VMA guidelines) instead of push constants
  /// `prepare_text_archetype_for_render_and_bind_pipeline` should have already been called
  /// therefore assumes text pipeline, descriptor sets, are already in place.
  fn render_text(
    &self,
    cmd_buffer: CommandBufferHandle,
    text: &str,
    start_cursor_position: [f32; 2],
    screen_extent: [f32; 2],
    atlas_id: (u64, u32),
    desired_points: f32,
    color: [f32; 4],
  ) -> GpuResult<()>;

  fn end_render_pass(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Call this right after `end_render_pass`. It allocates a staging buffer
  /// and natively injects the GPU Image-to-Buffer copy directly into your main frame.
  fn record_windowless_download(
    &self,
    cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
    task_id: u64,
  ) -> GpuResult<()>;

  /// Call this once `is_task_completed(task_id)` for a `record_windowless_download` task is true.
  /// It completes the CPU memory copy instantly without touching any Vulkan Queues.
  fn read_windowless_download(&self, task_id: u64, buffer: &mut [u8]) -> GpuResult<()>;

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
  ) -> GpuResult<()>;

  fn wire_callbacks(&self, pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) -> GpuResult<()>;

  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool>;

  fn create_task(&self) -> u64;

  fn fail_task(&self, task_id: u64, error: GpuError);

  fn success_task(&self, task_id: u64);
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
  fn push_constants_mesh(PhysicalMesh, PushConstants);
  fn push_billboard_constants(Billboard, BillboardPushConstants);
  fn push_cursor_constants(Cursor, CursorPushConstants);
  fn push_marker_constants(Marker, MarkerPushConstants);
  fn push_measurement_constants(Measurement, MeasurementPushConstants);
  fn push_sky_constants(Sky, SkyPushConstants);
  fn push_grid_constants(Grid, GridPushConstants);
  fn push_gizmo_constants(Gizmo, GizmoPushConstants);
  // Minimap, TODO
  // fn push_minimap_constants(Minimap, MinimapPushConstants);
  // Text,
  fn push_text_constants(Text, TextPushConstants);
  // Bvh,
  fn push_bvh_constants(Bvh, BvhPushConstants);

  // to add new archetypes, add one line here:
}

/// An RAII guard ensuring the command buffer is always submitted.
pub struct ScopedCommandBuffer<'a> {
  device: &'a dyn RenderDevice,
  cmd_buffer: CommandBufferHandle,
  task_id: Option<u64>,
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
      submitted: false,
    })
  }

  pub fn cmd(&self) -> CommandBufferHandle {
    self.cmd_buffer
  }

  /// Explicitly submits the command buffer.
  pub fn submit(mut self) -> GpuResult<()> {
    self.submitted = true;
    self.device.submit_command_buffer(self.cmd_buffer, self.task_id)
  }
}

impl<'a> Drop for ScopedCommandBuffer<'a> {
  fn drop(&mut self) {
    if !self.submitted {
      // Force submission on early exit/panic. Result is ignored to prevent double panics.
      let _ = self.device.submit_command_buffer(self.cmd_buffer, self.task_id);
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
      let _ = self.device.cancel_acquired_image(self.engine, ar.image_index, ar.frame_index as u32);
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
#[derive(Clone)]
pub struct RenderFrontend {
  backend: Arc<spin::RwLock<dyn RenderContext + 'static>>,
}

pub type WeakRenderFrontend = Weak<spin::RwLock<dyn RenderContext + 'static>>;
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
  type Target = Arc<spin::RwLock<dyn RenderContext + 'static>>;

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
}

// Boxing mechanism used by factory method in `gpu_backends` `new_render_frontend`
impl<T> From<T> for RenderFrontend
where
  T: RenderContext + 'static,
{
  fn from(value: T) -> Self {
    RenderFrontend {
      backend: Arc::new(spin::RwLock::new(value)),
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
  type ReadHandle<'a>: WaitHandle<alloc::vec::Vec<T>>
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
    scene: &Scene,
  ) -> EngineResult<()>;
}

/// Bridges synchronization between Compute (Kernels) and Graphics (RenderDevice).
pub trait KernelRenderBridge: Send + Sync {
  /// Inserts pipeline barriers or queue ownership transfers from Compute to Graphics.
  fn sync_compute_to_graphics(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Inserts pipeline barriers or queue ownership transfers from Graphics to Compute.
  fn sync_graphics_to_compute(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;
}
