//! gpu module.

pub use super::gpu_backends::*;
use crate::{
  gpu,
  gpu::frame::ResourceUploadResult,
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
  scene::{
    EntityId, PhysicalMeshComponent, Scene, TransformComponent,
    text::{FontAtlas, GlyphInfo},
  },
  simulation::comet::Texture,
  types::{EngineResult, GpuError, GpuResult},
};
use ab_glyph::PxScale;
use aethervk_oshal_rlib::os::time::timeus_t;
use alloc::sync::{Arc, Weak};
use bitflags::bitflags;
pub use compute_push_constants::{RigidBodyImex, Wrench};
use core::{
  ffi,
  hash::{Hash, Hasher},
};
use heapless::index_map::FnvIndexMap;

pub mod compute_push_constants;
pub mod frame;
pub mod scene_conversion;

pub use self::frame::RenderScene;

/// TODO: Document this item
pub type RwLock<T> = spin::rwlock::RwLock<T>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
/// TODO: Document this item
pub struct RenderBackendId(pub u64);
/// TODO: Document this item
pub const NULL_RENDER_BACKEND: RenderBackendId = RenderBackendId(0);
/// TODO: Document this item
pub const VULKAN_RENDER_BACKEND: RenderBackendId = RenderBackendId(1);
/// TODO: Document this item
pub const METAL_RENDER_BACKEND: RenderBackendId = RenderBackendId(2);
/// TODO: Document this item
pub const D3D12_RENDER_BACKEND: RenderBackendId = RenderBackendId(3);

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
/// TODO: Document this item
pub struct GpuResourceHandle(pub u64);
/// TODO: Document this item
pub const NULL_GPU_RESOURCE: GpuResourceHandle = GpuResourceHandle(0);

impl GpuResourceHandle {
  /// TODO: Document this item
  pub fn from_raw(raw: u64) -> Self {
    Self(raw)
  }
}

/// Abstract representation of Vulkan sharing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharingMode {
  Exclusive = 0,
  Concurrent = 1,
}

/// Represents queue sharing configuration for GPU resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueSharingInfo {
  pub mode: SharingMode,
  pub queue_family_indices: alloc::vec::Vec<u32>,
}

/// TODO: Document this item
#[derive(Clone, Copy)]
pub struct KinematicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
  pub velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  pub parent_frame_id: u32,
  pub mu: f32,
  pub own_frame_id: u32,
  pub frame_type: u32,
  pub scale: f32,
  pub shape_type: u32,
  pub shape_data: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[deprecated(
  since = "0.0.0",
  note = "Use `RigidBodyImex` for the new IMEX pipeline"
)]
/// Legacy rigid-body GPU layout (rotation-matrix based).
/// New code should use [`RigidBodyImex`] (quaternion-based).
pub struct RigidBodyGpu {
  pub position: [f32; 3],
  pub mass: f32,
  pub rotation: [[f32; 3]; 3], // Column-major
  pub linear_velocity: [f32; 3],
  pub _pad0: f32,
  pub angular_velocity: [f32; 3],
  pub _pad1: f32,
  pub inertia_tensor: [[f32; 3]; 3],
  pub force: [f32; 3],
  pub torque: [f32; 3],
  pub entity_id: EntityId,
  pub parent_frame_id: u32,
  pub shape_type: u32,
  pub shape_data: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
/// Represents a particle for explicit integration (Velocity Verlet).
/// Matches the AOSOA layout expected by compute shaders.
pub struct ParticleGpu {
  pub position: [f32; 3],
  pub velocity: [f32; 3],
  pub mass: f32,
  pub force: [f32; 3],
  // Metadata for CPU/Logic
  pub entity_id: EntityId,
  pub parent_frame_id: u32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ParticleMetadata {
  pub entity_id: EntityId,
  pub parent_frame_id: u32,
  pub original_index: u32,
}

pub fn pack_particles_aosoa(particles: &[[f32; 10]], subgroup_size: usize) -> alloc::vec::Vec<f32> {
  let num_particles = particles.len();
  let num_blocks = (num_particles + subgroup_size - 1) / subgroup_size;
  let mut buffer = alloc::vec::Vec::with_capacity(num_blocks * 10 * subgroup_size);
  buffer.resize(num_blocks * 10 * subgroup_size, 0.0);

  for (i, p) in particles.iter().enumerate() {
    let block = i / subgroup_size;
    let lane = i % subgroup_size;
    let base = block * (10 * subgroup_size) + lane;
    buffer[base] = p[0];
    buffer[base + 1 * subgroup_size] = p[1];
    buffer[base + 2 * subgroup_size] = p[2];
    buffer[base + 3 * subgroup_size] = p[3];
    buffer[base + 4 * subgroup_size] = p[4];
    buffer[base + 5 * subgroup_size] = p[5];
    buffer[base + 6 * subgroup_size] = p[6];
    buffer[base + 7 * subgroup_size] = p[7];
    buffer[base + 8 * subgroup_size] = p[8];
    buffer[base + 9 * subgroup_size] = p[9];
  }
  buffer
}

pub fn unpack_particles_aosoa(
  buffer: &[f32],
  subgroup_size: usize,
  count: usize,
) -> alloc::vec::Vec<[f32; 10]> {
  let mut particles = alloc::vec::Vec::with_capacity(count);
  for i in 0..count {
    let block = i / subgroup_size;
    let lane = i % subgroup_size;
    let base = block * (10 * subgroup_size) + lane;
    particles.push([
      buffer[base],
      buffer[base + 1 * subgroup_size],
      buffer[base + 2 * subgroup_size],
      buffer[base + 3 * subgroup_size],
      buffer[base + 4 * subgroup_size],
      buffer[base + 5 * subgroup_size],
      buffer[base + 6 * subgroup_size],
      buffer[base + 7 * subgroup_size],
      buffer[base + 8 * subgroup_size],
      buffer[base + 9 * subgroup_size],
    ]);
  }
  particles
}

#[derive(Clone, Copy)]
#[deprecated(note = "Use RigidBodyGpu or ParticleGpu instead")]
/// TODO: Document this item
pub struct DynamicBody {
  pub entity_id: EntityId,
  pub transform: TransformComponent,
  pub velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  pub mass: f32,
  pub parent_frame_id: u32,
  pub force: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
  pub shape_type: u32,
  pub shape_data: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
/// GPU-side representation of a force emitter.
///
/// For `type_id == 0` (Gravity): `mu` is the standard gravitational parameter G*M
/// in **km³/s²** (JPL Horizons default). `position` is the emitter's world-space
/// position in **AU** (macro frame). The shader transforms it into the target
/// body's local frame using a `GpuReferenceFrameArray` BDA.
///
/// For `type_id == 1` (Planar): `mu` holds the base force magnitude, `beta` is unused.
pub struct ForceEmitter {
  pub position: [f32; 3],
  pub mu: f32, // G*M in km³/s² for Gravity; base_force for Planar
  pub normal: [f32; 3],
  pub type_id: u32, // 0 = Gravity, 1 = Planar
  pub trunc_distance: f32,
  pub beta: f32, // radiation-pressure β; mu_eff = (1−β)·mu. 0 = pure gravity.
  pub _pad: [u32; 2],
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
/// TODO: Document this item
pub struct CommandBufferHandle(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// TODO: Document this item
pub struct RenderableInstanceId(pub u64);

/// TODO: Document this item
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
  /// TODO: Document this item
  pub struct TextureFlags: u32 {
    const ALBEDO    = 1 << 0;
    const NORMAL    = 1 << 1;
    const ROUGHNESS = 1 << 2;
    const AO        = 1 << 3;
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct TrajectoryPushConstants {
  pub map_ptr: u64,
  pub traj_ptr: u64,
  pub view_proj: [f32; 16],
  pub viewport_size: [f32; 2],
  pub _pad: [f32; 2],
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
/// TODO: Document this item
pub struct RationalBezierGpu {
  pub cp0: [f32; 4],
  pub cp1: [f32; 4],
  pub cp2: [f32; 4],
  pub cp3: [f32; 4],
}

#[repr(C, align(8))]
#[derive(Copy, Clone)]
/// TODO: Document this item
pub struct TrajectoryGpu {
  pub segments_ptr: u64,
  pub color: [f32; 4],
  pub line_width: f32,
  pub texture_id: u32,
}

#[repr(C, align(4))]
#[derive(Copy, Clone)]
/// TODO: Document this item
pub struct SegmentMapGpu {
  pub trajectory_id: u32,
  pub local_segment_id: u32,
  pub subdivisions: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
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
/// TODO: Document this item
pub struct SceneData {
  pub view_proj: [f32; 16],
  pub camera_pos: [f32; 4], // w is padding
  pub sun_pos: [f32; 4],    // w is padding
  pub sun_color: [f32; 4],
  pub window_extent: [f32; 2],
  pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct MaterialData {
  pub base_albedo: [f32; 4],    // w is base_roughness
  pub emissive_color: [f32; 4], // w is emissive_intensity
  pub base_ao: f32,
  pub paint_display_mode: u32,
  pub texture_flags: u32,
  pub _pad0: f32,
  pub sphere_center_radius: [f32; 4],
  pub grid_color_density: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct ObjectData {
  pub model: [f32; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct PhysicalMesh2PushConstants {
  pub scene_addr: u64,
  pub material_addr: u64,
  pub object_addr: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
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
/// TODO: Document this item
pub struct SunPushConstants {
  pub model_view_proj: [f32; 16],
  pub local_camera_pos: [f32; 3],
  pub _unused: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
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
/// TODO: Document this item
pub struct SkyPushConstants {
  pub inv_view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
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
/// TODO: Document this item
pub struct BvhPushConstants {
  pub mvp_arr: [f32; 16],
  pub center_type: [f32; 4],
  pub extents_arr: [f32; 4],
  pub axes_x: [f32; 4],
  pub axes_y: [f32; 4],
  pub axes_z: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Bvhwire2DataGpu {
  pub center_type: [f32; 4],
  pub extents: [f32; 4],
  pub axes_x: [f32; 4],
  pub axes_y: [f32; 4],
  pub axes_z: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Bvhwire2PushConstants {
  pub bvh_ptr: u64,
  pub _pad: u64,
  pub view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct SphereGizmoDataGpu {
  pub model: [f32; 16],
  pub radius: f32,
  pub subdivisions: f32,
  pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SphereGizmoPushConstants {
  pub gizmo_ptr: u64,
  pub _pad: u64,
  pub view_proj: [f32; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct ParticlePushConstants {
  pub view_proj: [f32; 16],
  pub camera_up: [f32; 3],
  pub time: f32,
  pub camera_right: [f32; 3],
  pub seed: f32,
  pub color: [f32; 4],
  pub radius: f32,
  pub camera_pos: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct Particle2PushConstants {
  pub view_proj: [f32; 16],
  pub camera_up: [f32; 3],
  pub time: f32,
  pub camera_right: [f32; 3],
  pub seed: f32,
  pub color: [f32; 4],
  pub radius: f32,
  pub camera_pos: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct GizmoPushConstants {
  pub view_proj: [f32; 16],
  pub scale: f32,
  pub instance_id: u32,
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

// TODO remove on text rendering v2
#[repr(C)]
#[derive(Debug, Clone, Copy)]
/// TODO: Document this item
pub struct TextPushConstants {
  pub pos: [f32; 2],
  pub scale: [f32; 2],
  pub color: [f32; 4],
  pub uv_bounds: [f32; 4],
  pub texture_id: u32,
  pub _padding: [u32; 3],
  pub view_proj: [f32; 16],
}

impl TextPushConstants {
  /// TODO: Document this item
  pub(crate) fn from_glyph(
    glyph: &GlyphInfo,
    cursor_position: [f32; 2],
    view_proj: [f32; 16],
    desired_points: f32,
    atlas_scale: PxScale,
    texture_id: u32,
    color: [f32; 4],
  ) -> Self {
    Self {
      pos: glyph.screen_position(cursor_position, desired_points, atlas_scale),
      scale: glyph.screen_size(desired_points, atlas_scale),
      color,
      uv_bounds: glyph.uv_bounds(),
      texture_id,
      _padding: [0; 3],
      view_proj,
    }
  }
}

impl RenderableInstanceId {
  /// TODO: Document this item
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
/// TODO: Document this item
pub static ASSET_DIR: spin::RwLock<Option<alloc::string::String>> = spin::RwLock::new(None);

/// Information about the synchronization payload after submitting a command buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CommandBufferSyncInfo {
  pub timeline_semaphore: u64, // Opaque handle for the backend
  pub timeline_value: u64,
}

/// TODO: Document this item
pub trait CommandBuffer: Send + Sync {
  /// TODO: Document this item
  fn submit(&mut self) -> EngineResult<Option<CommandBufferSyncInfo>>;
}

#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
/// TODO: Document this item
pub struct PipelineKey(pub u64);

/// TODO: Document this item
pub trait PipelineKeyable {
  fn pipeline_key(&self) -> PipelineKey;
}

#[derive(Default, Clone, Copy)]
/// TODO: Document this item
pub struct Rect2D {
  pub offset: [i32; 2],
  pub extent: [u32; 2],
}

impl Rect2D {
  /// TODO: Document this item
  pub fn from_extent(extent: [u32; 2]) -> Self {
    Self {
      offset: [0, 0],
      extent,
    }
  }
}

#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct Viewport {
  pub x: f32,
  pub y: f32,
  pub width: f32,
  pub height: f32,
  pub min_depth: f32,
  pub max_depth: f32,
}

impl Viewport {
  /// TODO: Document this item
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
  pub viewport_size: [f32; 2],
  pub _pad: [f32; 2],
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
/// TODO: Document this item
pub enum NativeGpuProperty {
  VulkanMetalDeviceId = 0,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
/// TODO: Document this item
pub enum ArchetypeId {
  Sun,
  PhysicalMesh,
  PhysicalMesh2,
  Billboard,
  Cursor,
  Marker,
  Measurement,
  Sky,
  Grid,
  Minimap,
  Text,
  Text2,
  Bvh,
  Bvhwire2,
  Particle,
  Gizmo,
  Particle2,
  Trajectory,
  Ui,
  Background,
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

/// TODO: Document this item
pub trait RenderDevice: Send + Sync + core::any::Any {
  fn as_any(&self) -> &dyn core::any::Any;

  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void>;

  fn print_info(&self) -> alloc::string::String;

  fn dump_memory_stats(&self);

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
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_physical_mesh_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    asset_hash: u64,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult>;

  fn get_physical_mesh2_resources(
    &self,
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult>;
  fn create_physical_mesh2_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    asset_hash: u64,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult>;

  #[allow(clippy::too_many_arguments)]
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

  /// Uploads particle systems into the mega-buffers. Should be called before rendering.
  fn upload_particle_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::ParticleDrawCall],
  ) -> GpuResult<()>;

  fn upload_particle2_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::Particle2DrawCall],
  ) -> GpuResult<()>;

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

  fn upload_text2(
    &self,
    cmd_buffer: CommandBufferHandle,
    glyphs: &[crate::gpu::TextGlyphGpu],
  ) -> GpuResult<Option<crate::gpu::Text2BatchCall>>;

  /// Draws a particle system using the mega-buffer
  fn draw_particle_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()>;

  fn draw_particle2_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()>;

  fn get_particle_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_particle2_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_trajectory_pipeline_key(&self, handle: PresentationEngineHandle)
  -> GpuResult<PipelineKey>;
  fn get_sun_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_sky_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_background_pipeline_key(&self, handle: PresentationEngineHandle)
  -> GpuResult<PipelineKey>;
  fn get_grid_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_bvh_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;
  fn get_bvhwire2_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey>;

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

  /// Start for an interface to draw something on the screen. Gets a handle to store rendering
  /// state setting commands

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

  /// Returns the mapped memory pointer of the emissive paint image for a given physical mesh instance
  fn get_emissive_paint_image_mapped_ptr(
    &self,
    mesh_id: crate::gpu::RenderableInstanceId,
  ) -> Option<*mut u8>;

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
  fn add_billboard_texture(
    &self,
    cmd_buffer: CommandBufferHandle,
    texture_id: u64,
    texture: &Texture,
    current_frame: u64,
  ) -> GpuResult<u32>;

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
  ) -> GpuResult<()>;

  fn prepare_bvh_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn prepare_bvhwire2_archetype_for_render_and_bind_pipeline(
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

  fn upload_bvhwire2_batch(
    &self,
    cmd_buffer: CommandBufferHandle,
    bvh_data: &[crate::gpu::Bvhwire2DataGpu],
  ) -> GpuResult<Option<crate::gpu::frame::Bvhwire2BatchCall>>;

  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
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
  ) -> GpuResult<()>;

  fn prepare_particle2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
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

  /// Screen extent should be the chosen presentation engine extent to correctly display screen size and position
  /// `atlas_id` is composed of the `hash` and internal id for the font atlas
  fn prepare_text_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  fn prepare_text2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()>;

  // TODO move in frame as a ui rendering
  #[deprecated]
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
    screen_extent: [f32; 2],
  ) -> GpuResult<()>;

  fn render_ui_rect(
    &self,
    cmd_buffer: CommandBufferHandle,
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
    view_proj: [f32; 16],
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
    task_id: u64,
  ) -> GpuResult<()>;

  /// Call this once `is_task_completed(task_id)` for a `record_windowless_download` task is true.
  /// It completes the CPU memory copy instantly without touching any Vulkan Queues.
  fn read_windowless_download(&self, task_id: u64, buffer: &mut [u8]) -> GpuResult<()>;

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
    sync_info: Option<CommandBufferSyncInfo>,
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
  fn push_constants_mesh(PhysicalMesh, PushConstants);
  fn push_constants_mesh2(PhysicalMesh2, PhysicalMesh2PushConstants);
  fn push_billboard_constants(Billboard, BillboardPushConstants);
  fn push_cursor_constants(Cursor, CursorPushConstants);
  fn push_marker_constants(Marker, MarkerPushConstants);
  fn push_measurement_constants(Measurement, MeasurementPushConstants);
  fn push_sky_constants(Sky, SkyPushConstants);
  fn push_grid_constants(Grid, GridPushConstants);
  fn push_gizmo_constants(Gizmo, GizmoPushConstants);
  // Text,
  fn push_text_constants(Text, TextPushConstants);
  fn push_text2_constants(Text2, Text2PushConstants);
  // Bvh,
  fn push_bvh_constants(Bvh, BvhPushConstants);
  fn push_bvhwire2_constants(Bvhwire2, Bvhwire2PushConstants);
  fn push_particle_constants(Particle, ParticlePushConstants);
  fn push_particle2_constants(Particle2, Particle2PushConstants);
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
  sync_info: Option<CommandBufferSyncInfo>,
  submitted: bool,
}

impl<'a> ScopedCommandBuffer<'a> {
  /// TODO: Document this item
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
      sync_info: None,
      submitted: false,
    })
  }

  /// Attaches synchronization info to this command buffer scope.
  pub fn set_sync_info(&mut self, sync_info: CommandBufferSyncInfo) {
    self.sync_info = Some(sync_info);
  }

  /// TODO: Document this item
  pub fn cmd(&self) -> CommandBufferHandle {
    self.cmd_buffer
  }

  /// Explicitly submits the command buffer.
  pub fn submit(mut self) -> GpuResult<()> {
    self.submitted = true;
    let res = self.device.submit_command_buffer(self.cmd_buffer, self.task_id, self.sync_info);
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
      let res = self.device.submit_command_buffer(self.cmd_buffer, self.task_id, self.sync_info);
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
  /// TODO: Document this item
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

/// TODO: Document this item
pub struct FrameCancelGuard<'a> {
  device: &'a dyn RenderDevice,
  engine: PresentationEngineHandle,
  acquire_result: Option<AcquireResult>,
}

impl<'a> FrameCancelGuard<'a> {
  /// TODO: Document this item
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

  /// TODO: Document this item
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
/// TODO: Document this item
pub struct RenderDeviceHandle(pub u64);

/// backend specific additional device init parameters
pub type DeviceAdditionalParams = FnvIndexMap<u64, usize, 8>;

/// TODO: Document this item
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
/// TODO: Document this item
pub struct RenderFrontend {
  backend: Arc<spin::RwLock<dyn RenderContext + 'static>>,
}

/// TODO: Document this item
pub type WeakRenderFrontend = Weak<spin::RwLock<dyn RenderContext + 'static>>;
/// TODO: Document this item
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
  /// TODO: Document this item
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
  /// TODO: Document this item
  pub fn take_and<T>(
    &self,
    f: impl FnOnce(&dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_read() {
      Some(guard) => Some(f(&*guard)),
      None => None,
    }
  }

  /// TODO: Document this item
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
/// TODO: Document this item
pub struct PresentationEngineHandle(pub u64);

impl PresentationEngineHandle {
  /// TODO: Document this item
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
  /// TODO: Document this item
  pub fn needs_resize(self) -> bool {
    self != Self::Optimal
  }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
/// TODO: Document this item
pub struct AcquireResult {
  pub image_index: u32,
  pub status: SwapchainStatus,
  /// handle to frame synchronization resources recognized by the presentation engine
  pub frame_index: u64,
  pub swapchain_generation: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
/// TODO: Document this item
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

/// TODO: Document this item
pub struct PresentationEngineParams {
  pub width: u32,
  pub height: u32,
  pub vsync: bool,
  pub window_info: OpaqueNativeHandleInfo,
  pub ty: PresentationEngineType,
  pub buffer_count: u32,
}

impl PresentationEngineParams {
  /// TODO: Document this item
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
      buffer_count: 3,
    }
  }
}

// -- Compute Engine traits --
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

  /// Returns the buffer device address (BDA) as a `u64`, suitable for use as
  /// a `layout(buffer_reference)` pointer in compute push constants.
  fn address(&self) -> u64;

  /// Enqueues a DMA copy-back command to the CPU. the returned Future does NOT
  /// borrow `cmd`, allowing you to submit the command buffer while the tasklet
  /// awaits the GPU synchronization primitive (fence)
  fn enqueue_read_to_cpu(&self, cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>>;
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
  /// Returns the BDA of the root BVH buffer (used as TLAS addr in push constants).
  fn address(&self) -> u64;
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CrossPair {
  pub macro_id: u32,
  pub micro_id: u32,
  pub lca_id: u32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct ColliderId {
  pub entity_id: u32,
  /// Set to `u32::MAX` if it's a monolithic body. Otherwise, it is the particle instance index.
  pub primitive_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct CollisionPair {
  pub a: ColliderId,
  pub b: ColliderId,
  pub time_of_impact: f32,
  pub is_lca: u32,
  pub lca_id: u32,
  pub frame_bda_low: u32,
  pub contact_normal: [f32; 3],
  pub frame_bda_high: u32,
  pub contact_point: [f32; 3],
  pub penetration_depth: f32,
}

/// GPU hardware subgroup (warp) size, clamped to the valid range for dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubgroupSize {
  Size16 = 16,
  Size32 = 32,
  Size64 = 64,
  Size128 = 128,
}

/// Computes execution for physics, particle systems, and interval arithmetic.
pub trait Kernels: Send + Sync {
  type Cmd: CommandBuffer;

  // --- Associated Types mapping to the underlying Backend ---
  type Buffer<T: Copy + Send + Sync>: DeviceBuffer<T, Cmd = Self::Cmd>;
  type List<T: Copy + Send + Sync>: DeviceList<T, Cmd = Self::Cmd>;
  type MotionBvh: DeviceBvh<Cmd = Self::Cmd>;
  /// Opaque GPU buffer holding the per-tick flat `TlasMultiNode<N>[]` array.
  type MotionTlas: DeviceBvh<Cmd = Self::Cmd>;

  fn discard_buffer<T: Copy + Send + Sync>(&self, buffer: Self::Buffer<T>);
  fn discard_list<T: Copy + Send + Sync>(&self, list: Self::List<T>);
  fn discard_bvh(&self, bvh: Self::MotionBvh);
  fn discard_tlas(&self, tlas: Self::MotionTlas);

  /// Returns the hardware subgroup size.
  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize>;

  fn wait_sync(&self, sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()>;

  fn refit_motion_blas(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    depth_indices: &Self::Buffer<u32>,
    total_nodes: u32,
  ) -> EngineResult<()>;

  /// Upload a CPU-built flat `TlasMultiNode<N>` node array (as raw bytes)
  /// to a device-visible STORAGE_BUFFER | SHADER_DEVICE_ADDRESS buffer.
  /// `node_bytes` = `bytemuck::cast_slice(&nodes_vec)`.
  /// Particle-BLAS leaf slots that were written with the sentinel `u32::MAX`
  /// in `child_indices[i]` are patched by the implementation to point to the
  /// GPU-built particle LBVH address before returning (Vulkan path only).
  fn upload_motion_tlas(
    &self,
    cmd: &mut Self::Cmd,
    node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas>;

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd>;

  /// Allocates a fresh, zero-initialised device list of `capacity` elements.
  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>>;

  fn build_leaves(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>>;

  // 1. & 2. Build Collections
  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<KinematicBody>>;

  /// Build IMEX rigid-body buffer + zero-initialised wrench buffer from ECS.
  /// Returns `(Buffer<RigidBodyImex>, Buffer<Wrench>)` — both per-frame,
  /// discarded at end of `simulation_step` via the timeline-safe discard pool.
  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)>;

  /// Upload `physical_scene.gpu_frames` as a GPU buffer for LCA broad-phase.
  fn build_frames(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>>;
  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<ParticleMetadata>)>;

  /// Uploads the `parent_frame_id` field from each `ParticleMetadata` entry as a
  /// tightly-packed `u32[]` GPU buffer in AOSOA invocation order (same order as the
  /// particle float buffer produced by [`build_particles`]).
  ///
  /// This is the BDA the `apply_emitters_to_particles.comp` shader reads via
  /// `particle_frame_ids.frame_ids[gid]`.
  ///
  /// Cheap: just one host→device upload per frame, zero extra scene queries.
  fn build_particle_frame_ids(
    &self,
    cmd: &mut Self::Cmd,
    particle_metadata: &[ParticleMetadata],
  ) -> EngineResult<Self::Buffer<u32>>;

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<ForceEmitter>>;

  fn emit_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()>;

  // ── Legacy ODE phases (deprecated — new code uses imex_integrate_* below) ──

  /// VV predictor for particles. **Deprecated**: use [`imex_integrate_particles_p1_p2`].
  #[deprecated(since = "0.0.0", note = "use imex_integrate_particles_p1_p2")]
  fn step_ode_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  /// RB IMR solve. **Deprecated**: use [`imex_integrate_bodies_p3`].
  #[deprecated(since = "0.0.0", note = "use imex_integrate_bodies_p3")]
  fn step_ode_p3_p4(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  // 4.5 Compute self gravity (Barnes-Hut or fallback)
  fn compute_self_gravity(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<f32>,
  ) -> EngineResult<()>;

  /// VV corrector for particles. **Deprecated**: use [`imex_integrate_particles_p4_5`].
  #[deprecated(since = "0.0.0", note = "use imex_integrate_particles_p4_5")]
  fn step_ode_p5(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  // ── New Symmetric Strang-Split IMEX Integrators ────────────────────────────

  /// VV predictor — half-kick + full position drift to x_{n+1}.
  /// Clears particle force accumulator slots so force generators start from zero.
  fn imex_integrate_particles_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  /// RB Implicit Midpoint Rule with Picard gyroscopic stabilisation.
  /// Integrates RBs from (x_n, v_n) to (x_{n+1}, v_{n+1}).
  /// Clears the wrench buffer on entry.
  fn imex_integrate_bodies_p3(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &mut Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
    emitters: &Self::Buffer<ForceEmitter>,
    frames: &Self::Buffer<crate::physics::physics_scene::GpuReferenceFrame>,
    dt: timeus_t,
  ) -> EngineResult<()>;

  /// Reduces per-leaf wrenches into each rigid body's CoM wrench slot.
  /// Must be called after all force generators have written to the wrench buffer
  /// and before `imex_integrate_bodies_p3`.
  fn imex_rb_force_assign(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
  ) -> EngineResult<()>;

  /// VV corrector — advances v_{n+½} → v_{n+1} using F(x_{n+1}).
  /// Thread 0 simultaneously advances the 64-bit engine clock.
  fn imex_integrate_particles_p4_5(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) -> EngineResult<()>;

  /// Applies macro-frame gravity emitters to microframe particles (GPU-inline frame transform).
  ///
  /// Must run between Barnes-Hut self-gravity and the P4_5 VV corrector.
  /// `particle_frame_ids` — one `u32` frame index per particle in AOSOA invocation order.
  fn apply_emitters_to_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<ForceEmitter>,
    frames: &Self::Buffer<crate::physics::physics_scene::GpuReferenceFrame>,
    particle_frame_ids: &Self::Buffer<u32>,
    num_emitters: u32,
  ) -> EngineResult<()>;

  // ── New Broad-Phase Suite ──────────────────────────────────────────────────

  /// Zeroes all four pair-list count fields (rb_rb, rb_ps, lca, raw).
  /// Must be the first broad-phase shader each frame.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_clear(
    &self,
    cmd: &mut Self::Cmd,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    internal_pairs_addr: u64,
    out_sparse_addr: u64,
  ) -> EngineResult<()>;

  /// Generates one swept AABB (TLASLeaf) per entity.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_bounds_gen(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    leaves_addr: u64,
    lca_entities_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) -> EngineResult<()>;

  /// Subgroup-cooperative TLAS traversal — writes raw overlapping entity pairs.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_scene(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) -> EngineResult<()>;

  /// Classifies raw pairs into RB-RB, RB-PS, and cross-LCA typed queues.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()>;

  /// Transforms macro-frame RB AABBs into micro-frame space and traverses
  /// the micro-frame BVH to produce refined narrow-phase candidate pairs.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    lca_entities_addr: u64,
    macro_leaves_addr: u64,
    entity_headers_addr: u64,
    lca_query_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_cross_pairs_addr: u64,
    total_queries: u32,
    max_pairs: u32,
    num_rigid_bodies: u32,
  ) -> EngineResult<()>;

  /// Subgroup-cooperative LBVH traversal for particle–particle self-collision.
  /// Computes Hookean repulsion and atomicAdds forces directly into AOSOA slots.
  #[cfg(any(test, feature = "collisions"))]
  fn bp_particle_self(
    &self,
    cmd: &mut Self::Cmd,
    bvh_addr: u64,
    particles: &mut Self::Buffer<f32>,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) -> EngineResult<()>;

  // ── Collision Pipeline ────────────────────────────────────────────────────
  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh>;

  /// **Deprecated**: broad-phase now handled by `bp_clear` → `bp_bounds_gen` → `bp_scene`.
  #[deprecated(since = "0.0.0", note = "use the bp_* broad-phase suite")]
  #[cfg(any(test, feature = "collisions"))]
  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<CollisionPair>>;

  /// **Deprecated**: use `bp_cross_lca` for LCA pairs and narrow CCD for rb-rb/rb-ps.
  #[deprecated(since = "0.0.0", note = "use bp_cross_lca + narrow CCD")]
  #[cfg(any(test, feature = "collisions"))]
  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<CollisionPair>,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<CollisionPair>>;

  /// Evaluates narrow phase CCD for a list of broad-phase entity pairs.
  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd(
    &self,
    cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<CollisionPair>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    lca_entities_addr: u64,
    space_type: u32,
    dt: f32,
    output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()>;

  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<CrossPair>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    lca_entities_addr: u64,
    space_type: u32,
    dt: f32,
    output_list: &Self::List<CollisionPair>,
  ) -> EngineResult<()>;

  /// Stream compaction shrink logic evaluated entirely on the GPU.
  #[cfg(any(test, feature = "collisions"))]
  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<CollisionPair>>;

  /// Parallel reduction to find the lowest `time_of_impact`.
  /// Returns a tiny buffer of length 1 containing $t_c$.
  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<CollisionPair>,
    dt: f32,
  ) -> EngineResult<Self::Buffer<u32>>;

  #[cfg(any(test, feature = "collisions"))]
  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    collisions: &Self::List<CollisionPair>,
    lca_entities_addr: u64,
    force_inelastic: bool,
  ) -> EngineResult<()>;

  // ── CCD Rewind Subsystem ───────────────────────────────────────────────────
  #[cfg(any(test, feature = "collisions"))]
  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)>;

  #[cfg(any(test, feature = "collisions"))]
  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    snapshot: &(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>),
  ) -> EngineResult<()>;

  // ── Write back dynamic state ───────────────────────────────────────────────
  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    particle_metadata: &[ParticleMetadata],
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<Option<CommandBufferSyncInfo>>;
}

/// Bridges synchronization between Compute (Kernels) and Graphics (RenderDevice).
pub trait KernelRenderBridge: Send + Sync {
  /// Inserts pipeline barriers or queue ownership transfers from Compute to Graphics.
  fn sync_compute_to_graphics(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;

  /// Inserts pipeline barriers or queue ownership transfers from Graphics to Compute.
  fn sync_graphics_to_compute(&self, cmd_buffer: CommandBufferHandle) -> GpuResult<()>;
}
