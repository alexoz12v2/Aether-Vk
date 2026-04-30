use crate::{
  simulation::comet::Texture,
  gpu::{
    ArchetypeId, PipelineKey, RenderDeviceExt, vulkan::device::archetypes_struct::Archetypes, self,
    frame::ResourceUploadResult, AcquireResult, CommandBufferHandle, GpuResourceHandle,
    NativeGpuProperty, PipelineKeyable, PresentationEngineHandle, RenderDevice,
    RenderableInstanceId, TextureFlags,
  },
  gpu_backends::vulkan::{
    self,
    device::{
      commands::CommandBufferId,
      memory::GlobalDeviceAllocator,
      renderpasses::{RenderPassSpecification, RenderPassType},
      resources::{DiscardableResource, ForwardMeshRenderResource, Image},
      shader_manager::ShaderKey,
    },
    instance,
    utils::{self, NonZeroHandle},
  },
  scene::{EntityId, PhysicalMeshComponent},
  simulation::comet::Comet,
  types::{GpuError, GpuResult},
  scene::text::FontAtlas,
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::vector::vec3::Vec3f32,
  math::vector::{Vector, Vector3},
  os::fs::FileSystemObject,
  os::pool::WorkloadStatus,
};
use alloc::{
  boxed::Box,
  collections::BTreeMap,
  format,
  string::{String, ToString},
  sync::Arc,
  vec,
  vec::Vec,
};
use ash::vk::{self, Handle, PhysicalDeviceProperties};
use core::{
  fmt::Formatter,
  sync::atomic::AtomicU32,
  fmt,
  hash::{Hash, Hasher},
  ptr::{self, NonNull},
  sync::atomic::{AtomicU64, Ordering},
};
use heapless::index_map::FnvIndexMap;
use oshal::{
  hash::FnvHasher,
  os::{
    fs::PathBuf,
    memory::{MaxAlignedStorage, StackAllocator},
    native::this_thread,
  },
};
use spin::Mutex;
use vk_mem::Alloc;
use aethervk_oshal_rlib::math::safe_div;
use crate::gpu::vulkan::device::resources::ParticleRenderResourceArchetype;

// TODO refactor queue to use a Mutex for VkQueue handle or a struct to ensure submits are sync

mod archetypes_struct;
mod commands;
mod descriptors;
mod memory;
mod pipelines;
mod renderpasses;
mod resources;
mod shader_manager;
mod swapchain;
mod timeline_manager;

// TODO standardize error strings with prefix used in all modules. Use concat! for &'static str messages and possibly write some macros to diminish repeated code
// TODO No vulkan calls while holding locks
// TODO remove all unwrap and unwrap_unchecked (unless absolutely necessary or sure)

#[derive(Debug)]
pub(super) struct TaskEntry {
  pub(super) target_value: AtomicU64,
  pub(super) status: AtomicU32, // 0: Pending, 1: Success, 2: Failed
  pub(super) error: spin::RwLock<Option<GpuError>>,
}

const TASK_STATUS_PENDING: u32 = 0;
const TASK_STATUS_SUCCESS: u32 = 1;
const TASK_STATUS_FAILED: u32 = 2;

struct TimelinePollingWorkload {
  timeline_sem_device: ash::khr::timeline_semaphore::Device,
  timeline_semaphore: vk::Semaphore,
  timeline_semaphore_cached_value: Arc<AtomicU64>,
  task_registry: Arc<spin::RwLock<BTreeMap<u64, Arc<TaskEntry>>>>,
  stop_signal: Arc<core::sync::atomic::AtomicBool>,
}

impl oshal::os::pool::Workload for TimelinePollingWorkload {
  fn execute(&mut self) -> WorkloadStatus {
    let mut last_check = oshal::os::time::TimeInfo::new(16667, 100000, 1.0);
    // oshal::log!("TimelinePollingWorkload started execution");
    while !self.stop_signal.load(Ordering::Acquire) {
      last_check.ut_update();

      // Poll semaphore
      if let Ok(gpu_value) = unsafe {
        self
          .timeline_sem_device
          .get_semaphore_counter_value(self.timeline_semaphore)
      } {
        self
          .timeline_semaphore_cached_value
          .fetch_max(gpu_value, Ordering::Relaxed);

        // Resolve tasks
        let registry = self.task_registry.read();
        let completed_ids: Vec<u64> = registry
          .iter()
          .filter(|(_, entry)| {
            entry.status.load(Ordering::Acquire) == TASK_STATUS_PENDING
              && gpu_value >= entry.target_value.load(Ordering::Acquire)
          })
          .map(|(id, _)| *id)
          .collect();

        for id in completed_ids {
          if let Some(entry) = registry.get(&id) {
            entry.status.store(TASK_STATUS_SUCCESS, Ordering::Release);
          }
        }
      }

      // TODO testing
      return WorkloadStatus::Yield;
      // Yield/Sleep ~16.67ms
      // oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(16));
    }
    oshal::os::pool::WorkloadStatus::Complete
  }
}

trait DeviceResource {
  /// Cleanup function to facilitate hierarchical manual Drop of resources
  /// without having to propagate through `Arc` or other means a reference
  /// to device handle and its function pointers
  /// Note: This function is not responsible to setup the proper state for cleanup (eg synchronization)
  fn cleanup(&mut self, device: &ash::Device);
}

struct FunctionalDeviceResource<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> {
  handle: H,
  cleanup: Option<F>,
}

impl<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> FunctionalDeviceResource<H, F> {
  fn new(handle: H, cleanup: F) -> Self {
    Self {
      handle,
      cleanup: Some(cleanup),
    }
  }
}

impl<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> DeviceResource
  for FunctionalDeviceResource<H, F>
{
  fn cleanup(&mut self, device: &ash::Device) {
    let h = self.handle;
    if let Some(cleanup) = self.cleanup.take() {
      cleanup(h, device);
    }
  }
}

struct DeviceResourceJanitor<'a, const N: usize> {
  device: &'a ash::Device,
  resources: heapless::Vec<NonNull<dyn DeviceResource + 'a>, N>,
  heap_resources: Vec<Box<dyn DeviceResource + 'a>>,
  allocator: StackAllocator,
  storage: MaxAlignedStorage<N>,
}

impl<'a, const N: usize> DeviceResourceJanitor<'a, N> {
  fn new(device: &'a ash::Device) -> Self {
    Self {
      device,
      allocator: StackAllocator::new(),
      resources: heapless::Vec::new(),
      heap_resources: Vec::new(),
      storage: MaxAlignedStorage([0; N]),
    }
  }

  pub fn clear(&mut self) {
    // The `drop` implementation will handle cleanup of existing resources.
    // Here we just need to reset the state.
    self.allocator = StackAllocator::new();
    self.resources.clear();
    self.heap_resources.clear();
  }

  pub fn push<T: DeviceResource + 'a>(&mut self, resource: T) -> Result<(), &'static str> {
    // Check if there's space in the inline allocator
    let layout = core::alloc::Layout::new::<T>();
    let start = self.allocator.offset.get();
    let align_offset = unsafe {
      self
        .storage
        .0
        .as_ptr()
        .add(start)
        .align_offset(layout.align())
    };

    let aligned_start = match start.checked_add(align_offset) {
      Some(s) => s,
      None => {
        // overflow, definitely no space
        self.heap_resources.push(Box::new(resource));
        return Ok(());
      }
    };
    let end = match aligned_start.checked_add(layout.size()) {
      Some(e) => e,
      None => {
        // overflow
        self.heap_resources.push(Box::new(resource));
        return Ok(());
      }
    };

    if end > N || self.resources.is_full() {
      // Inline storage is full, or the pointer vec is full. Fallback to heap.
      self.heap_resources.push(Box::new(resource));
      return Ok(());
    }

    // There is space, so we can safely allocate.
    unsafe {
      let base_ptr = self.storage.0.as_mut_ptr();
      let ptr: *mut T = self.allocator.allocate(base_ptr, N, resource)?;

      let dyn_ptr: *mut (dyn DeviceResource + 'a) = ptr;
      let non_null = NonNull::new_unchecked(dyn_ptr);

      // This push should not fail because we checked is_full()
      self.resources.push(non_null).ok().unwrap();
    }

    Ok(())
  }
}

impl<'a, const N: usize> Drop for DeviceResourceJanitor<'a, N> {
  fn drop(&mut self) {
    // Destroy heap-allocated resources first, in reverse order of allocation
    for mut resource_box in self.heap_resources.drain(..).rev() {
      resource_box.cleanup(self.device);
    }

    // Destroy most recently allocated inline resources first
    for resource in self.resources.iter_mut().rev() {
      unsafe {
        let resource = resource.as_mut();
        resource.cleanup(self.device);

        core::ptr::drop_in_place(ptr::from_mut(resource));
      }
    }
  }
}

pub(super) struct PendingDownload {
  pub(super) staging_buffer: vk::Buffer,
  pub(super) allocation: vk_mem::Allocation,
  pub(super) size: usize,
}

/// Device Resources. Each member here implements `DeviceResources` trait and is either
/// - implementing `Sync`
/// - Wrapped into a RwLock/Mutex
/// - Native Vulkan Handle, externally synchronized
struct DeviceResources {
  // TODO remove the GlobalDeviceAllocator struct as it adds nothing
  allocator: memory::GlobalDeviceAllocator,
  discard_pool: resources::DiscardPool,
  live_presentation_engines: spin::RwLock<
    hashbrown::HashMap<PresentationEngineHandle, spin::RwLock<swapchain::PresentationState>>,
  >,
  command_pools: spin::RwLock<
    heapless::Vec<Option<Arc<commands::CommandPools>>, { utils::MAX_QUEUE_FAMILY_COUNT }>,
  >,
  descriptor_pool: spin::RwLock<Option<Arc<descriptors::DescriptorPools>>>,
  pipeline_pool: spin::RwLock<pipelines::PipelinePool>,
  renderpasses: renderpasses::RenderPasses,
  shader_manager: spin::RwLock<shader_manager::ShaderManager>,

  timeline_manager: timeline_manager::TimelineManager,
  next_cmd_id: Arc<AtomicU64>,

  linear_sampler: NonZeroHandle<vk::Sampler>,

  physical_mesh_resources:
    spin::RwLock<Option<hashbrown::HashMap<RenderableInstanceId, ForwardMeshRenderResource>>>,
  sun_resources: spin::RwLock<Option<hashbrown::HashMap<EntityId, resources::SunRenderResource>>>,
  sky_image: spin::RwLock<Option<Image>>,
  billboard_resources: spin::RwLock<Vec<Image>>,

  pending_downloads: spin::RwLock<hashbrown::HashMap<u64, PendingDownload>>,

  frame_staging_arena: spin::RwLock<Option<memory::FrameStagingArena>>,

  archetypes: Archetypes,
}

// TODO: each member should derive it so that this can derive it too
impl fmt::Debug for DeviceResources {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    f.write_str("DeviceResources")
  }
}

impl DeviceResource for DeviceResources {
  /// cleanup in reverse order of declaration in the struct
  fn cleanup(&mut self, device: &ash::Device) {
    for (_, mut download) in self.pending_downloads.write().drain() {
      unsafe {
        self
          .allocator
          .allocator
          .destroy_buffer(download.staging_buffer, &mut download.allocation);
      }
    }
    // all discardable resources should have been already discarded
    if self.has_discardables() {
      self.clear_discardables(&device);
    }
    self.discard_pool.cleanup(device);

    self.timeline_manager.cleanup(device);

    self.renderpasses.cleanup(device);

    self.shader_manager.write().destroy(device);

    // Safety: If this is a properly constructed `DeviceResources`, then `descriptor_pool = Some(_)`
    let dp_opt = self.descriptor_pool.write().take();
    if let Some(pool) = dp_opt {
      assert_eq!(Arc::strong_count(&pool), 1);
      let mut descriptor_pool: descriptors::DescriptorPools = Arc::try_unwrap(pool).unwrap();
      descriptor_pool.cleanup(device);
    }

    self.pipeline_pool.write().cleanup(device);

    let mut cp_lock = self.command_pools.write();
    for command_pool in cp_lock.iter_mut() {
      if let Some(pool) = command_pool.take() {
        assert_eq!(Arc::strong_count(&pool), 1);
        let mut command_pool = Arc::try_unwrap(pool).unwrap();
        command_pool.cleanup(device);
      }
    }

    for (_, presentation_state) in self.live_presentation_engines.write().drain() {
      presentation_state.write().cleanup(device);
    }

    // - Linear Sampler
    unsafe { device.destroy_sampler(self.linear_sampler.get(), None) };

    if let Some(sky_image) = self.sky_image.write().take() {
      unsafe {
        vk_mem::ffi::vmaDestroyImage(
          self.allocator.allocator.get_raw(),
          sky_image.image.get(),
          sky_image.allocation.get_raw(),
        );
        device.destroy_image_view(sky_image.image_view.get(), None);
      }
    }

    if let Some(mut arena) = self.frame_staging_arena.write().take() {
      arena.destroy(&self.allocator.allocator);
    }
    self.allocator.cleanup(device);
  }
}

impl DeviceResources {
  /// update [`pipelines::FragmentOut`] and [`vk::RenderPass`] inside [`pipelines::GraphicsInfo`]
  /// disard old and create updated graphics [`vk::Pipeline`]
  /// Note: Update is performed only if archetype initialized once
  fn update_physical_mesh_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] update_physical_mesh_archetype_for_presentation_engine",
      ))?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self
      .archetypes
      .update_physical_mesh_archetype_for_presentation_engine(
        device,
        &presentation_engine_state,
        &mut write_pipeline,
        &self.renderpasses,
        &self.allocator.allocator,
        &self.discard_pool,
        timeline,
      )
  }

  fn update_cursor_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] update_cursor_archetype_for_presentation_engine",
      ))?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self
      .archetypes
      .update_cursor_archetype_for_presentation_engine(
        device,
        &presentation_engine_state,
        &mut write_pipeline,
        &self.renderpasses,
        &self.allocator.allocator,
        &self.discard_pool,
        timeline,
      )
  }

  fn update_sun_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] update_sun_archetype_for_presentation_engine",
      ))?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self
      .archetypes
      .update_sun_archetype_for_presentation_engine(
        device,
        &presentation_engine_state,
        &mut write_pipeline,
        &self.renderpasses,
        &self.allocator.allocator,
        &self.discard_pool,
        timeline,
      )
  }

  fn create_physical_mesh_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    outline_vertex_shader_key: ShaderKey,
    outline_fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_physical_mesh_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_physical_mesh_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      outline_vertex_shader_key,
      outline_fragment_shader_key,
      depth_stencil_format,
      queue,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_sun_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_sun_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_sun_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_sky_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_sky_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_grid_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_grid_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_minimap_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engine_lock = self.live_presentation_engines.read();
    let pe = presentation_engine_lock.get(&handle).unwrap().read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_minimap_archetype(
      device,
      shader_manager,
      vkey,
      fkey,
      depth_stencil_format,
      &pe,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_measurement_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_measurement_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_measurement_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_marker_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_marker_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_marker_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_billboard_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_billboard_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_billboard_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_particle_archetype(
    // TODO
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_particle_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_particle_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_cursor_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock =
      live_presentation_engines_lock
        .get(&handle)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] create_cursor_archetype",
        ))?;
    let presentation_engine_state = presentation_engine_lock.read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_cursor_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_text_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_text_archetype(
      device,
      shader_manager,
      vertex_shader_key,
      fragment_shader_key,
      depth_stencil_format,
      queue,
      &presentation_engine_state,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn create_bvh_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let live_engines_lock = self.live_presentation_engines.read();
    let pe = live_engines_lock.get(&handle).unwrap().read();
    let mut write_pipeline = self.pipeline_pool.write();
    self.archetypes.create_bvh_archetype(
      device,
      shader_manager,
      vkey,
      fkey,
      depth_stencil_format,
      &pe,
      &self.allocator.allocator,
      &self.discard_pool,
      &self.renderpasses,
      &mut write_pipeline,
      timeline,
    )
  }

  fn has_discardables(&self) -> bool {
    self.archetypes.has_discardables()
      || self.physical_mesh_resources.read().is_some()
      || self.sun_resources.read().is_some()
      || !self.billboard_resources.read().is_empty()
  }

  fn clear_discardables(&mut self, device: &ash::Device) {
    debug_assert!(self.has_discardables());
    if let Some(mut resources) = self.physical_mesh_resources.write().take() {
      for (_, mut resource) in resources.drain() {
        resource.discard(device, &self.discard_pool, u64::MAX);
      }
    }
    if let Some(mut resources) = self.sun_resources.write().take() {
      for (_, resource) in resources.drain() {
        if let Some(mut img) = resource.image {
          unsafe {
            self
              .allocator
              .allocator
              .destroy_image(img.image.get(), &mut img.allocation);
            device.destroy_image_view(img.image_view.get(), None);
          }
        }
        if let Some(buffer) = resource.params_buffer {
          unsafe {
            vk_mem::ffi::vmaDestroyBuffer(
              self.allocator.allocator.get_raw(),
              buffer,
              resource.params_alloc.unwrap().get_raw(),
            );
          }
        }
        if let Some(layout) = resource.compute_pipeline_layout {
          unsafe { device.destroy_pipeline_layout(layout, None) };
        }
        if let Some(pool) = resource.compute_descriptor_pool {
          unsafe { device.destroy_descriptor_pool(pool, None) };
        }
        if let Some(layout) = resource.compute_descriptor_set_layout {
          unsafe { device.destroy_descriptor_set_layout(layout, None) };
        }
      }
    }
    for image in self.billboard_resources.write().drain(..) {
      self.discard_pool.discard_image(
        self.allocator.allocator.get_raw(),
        image.image.get(),
        image.allocation,
        u64::MAX,
      );
      self
        .discard_pool
        .discard_image_view(image.image_view.get(), u64::MAX);
    }

    if let Some(sky_image) = self.sky_image.write().take() {
      self.discard_pool.discard_image(
        self.allocator.allocator.get_raw(),
        sky_image.image.get(),
        sky_image.allocation,
        u64::MAX,
      );
      self
        .discard_pool
        .discard_image_view(sky_image.image_view.get(), u64::MAX);
    }

    self.archetypes.discard(device, &self.discard_pool);

    debug_assert!(!self.has_discardables());
  }

  fn new<'a>(
    instance: &instance::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    unique_family_indices_iter: impl Iterator<Item = &'a u32>,
  ) -> GpuResult<Self> {
    // - linear sampler
    let sampler_info = vk::SamplerCreateInfo::default()
      .mag_filter(vk::Filter::LINEAR)
      .min_filter(vk::Filter::LINEAR)
      .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
      .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let linear_sampler = unsafe { device.create_sampler(&sampler_info, None) }?;
    // - VMA Device Allocator
    // TODO: this function should cleanup everything on the first error, not leak everything
    let mut allocator = match unsafe {
      GlobalDeviceAllocator::new(
        &instance.instance,
        &device,
        physical_device,
        instance.api_version(),
      )
    } {
      Ok(allocator) => allocator,
      Err(e) => {
        unsafe { device.destroy_sampler(linear_sampler, None) };
        return Err(e);
      }
    };
    // - Timeline Semaphore
    let mut timeline_manager =
      match timeline_manager::TimelineManager::new(&instance.instance, device) {
        Ok(t) => t,
        Err(e) => {
          allocator.cleanup(device);
          unsafe { device.destroy_sampler(linear_sampler, None) };
          return Err(e);
        }
      };

    // - Descriptor Pool
    let mut descriptor_pool = match descriptors::DescriptorPools::new(device, 256) {
      Ok(pool) => pool,
      Err(e) => {
        allocator.cleanup(device);
        timeline_manager.cleanup(device);
        unsafe { device.destroy_sampler(linear_sampler, None) };
        return Err(e);
      }
    };

    let renderpasses =
      renderpasses::RenderPasses::new(&instance.instance, &device, &allocator.allocator);

    let pipeline_pool = match pipelines::PipelinePool::new(device, None) {
      Ok(pool) => spin::RwLock::new(pool),
      Err(e) => {
        let descriptor_pool = unsafe { Arc::get_mut(&mut descriptor_pool).unwrap_unchecked() };
        descriptor_pool.cleanup(device);
        allocator.cleanup(device);
        timeline_manager.cleanup(device);
        unsafe { device.destroy_sampler(linear_sampler, None) };
        return Err(e);
      }
    };

    // - Discard Pool
    let discard_pool = unsafe { resources::DiscardPool::new(64) };
    // - Command Pools
    let mut command_pools = heapless::Vec::new();
    for &queue_family_index in unique_family_indices_iter {
      unsafe {
        command_pools.push_unchecked(Some(Arc::new(commands::CommandPools::new(
          queue_family_index,
        ))))
      };
    }
    // - Swapchain hashmap
    let live_presentation_engines = spin::RwLock::new(hashbrown::HashMap::new());

    // timeline semaphore promoted to core after 1.2 (included)
    debug_assert!(instance.api_version() < vk::API_VERSION_1_2);

    let frame_staging_arena = memory::FrameStagingArena::new(&allocator.allocator, 128 * 1024 * 1024)?;

    Ok(Self {
      allocator,
      command_pools: spin::RwLock::new(command_pools),
      discard_pool,
      live_presentation_engines,
      descriptor_pool: spin::RwLock::new(Some(descriptor_pool)),
      pipeline_pool,
      renderpasses,
      shader_manager: spin::RwLock::new(shader_manager::ShaderManager::new()),
      linear_sampler: unsafe { NonZeroHandle::new_unchecked(linear_sampler) },
      timeline_manager,
      physical_mesh_resources: spin::RwLock::new(None),
      sun_resources: spin::RwLock::new(None),
      billboard_resources: spin::RwLock::new(Vec::with_capacity(16)),
      archetypes: Archetypes::default(),
      sky_image: spin::RwLock::new(None),
      next_cmd_id: Arc::new(AtomicU64::new(1)),
      pending_downloads: spin::RwLock::new(hashbrown::HashMap::new()),
      frame_staging_arena: spin::RwLock::new(Some(frame_staging_arena)),
    })
  }

  fn get_timeline_semaphore_cached_value(&self) -> u64 {
    self.timeline_manager.get_cached_value()
  }
}

#[derive(Clone, Copy)]
struct RecordingCmdBufferDataPresentation {
  acquire_result: AcquireResult,
  presentation_engine: PresentationEngineHandle,
}

struct RecordingCmdBufferData {
  command_buffer: NonZeroHandle<vk::CommandBuffer>,
  bound_pipeline: Option<NonZeroHandle<vk::Pipeline>>,
  presentation: Option<RecordingCmdBufferDataPresentation>,
  has_begun: bool,
}

impl RecordingCmdBufferData {
  fn new(command_buffer: NonZeroHandle<vk::CommandBuffer>) -> Self {
    Self {
      command_buffer,
      bound_pipeline: None,
      presentation: None,
      has_begun: false,
    }
  }

  /// command buffer is automatically recycled by [`commands::CommandPools`]
  fn discard(
    &mut self,
    cmd_buf_id: CommandBufferId,
    discard_pool: &resources::DiscardPool,
    cmd_pools: Arc<commands::CommandPools>,
    timeline: u64,
  ) {
    let tid = this_thread::id();
    if self.has_begun {
      discard_pool.discard_command_buffer(
        tid,
        cmd_buf_id,
        self.command_buffer.get(),
        cmd_pools,
        timeline,
      );
    } else {
      // Not recorded, so just recycle it immediately.
      let _ = cmd_pools.recycle(tid, cmd_buf_id, self.command_buffer.get());
    }
  }
}

pub(super) struct LogicalDevice {
  pub handle: ash::Device,
  pub submission_lock: Mutex<()>,
  /// Note: Remove if API_VERSION_1_2
  pub create_renderpass2: ash::khr::create_renderpass2::Device,
  pub buffer_device_address: ash::khr::buffer_device_address::Device,
  pub timeline_semaphore: ash::khr::timeline_semaphore::Device,
  /// Note: Remove if API_VERSION_1_3
  pub synchronization2: ash::khr::synchronization2::Device,

  #[cfg(debug_assertions)]
  pub debug_utils: ash::ext::debug_utils::Device,

  #[cfg(target_vendor = "apple")]
  pub metal_objects: ash::ext::metal_objects::Device,
}

impl core::ops::Deref for LogicalDevice {
  type Target = ash::Device;

  fn deref(&self) -> &Self::Target {
    &self.handle
  }
}

impl LogicalDevice {
  #[cfg(debug_assertions)]
  pub fn set_debug_name<T: vk::Handle>(&self, object: T, name: &str) {
    use core::str::FromStr;

    if name.is_empty() {
      return;
    }
    let name_cstr = alloc::ffi::CString::from_str(name).unwrap();
    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
      .object_handle(object)
      .object_name(&name_cstr);

    unsafe {
      self
        .debug_utils
        .set_debug_utils_object_name(&name_info)
        .expect(&alloc::format!("failed to set name {}", name));
    }
  }

  #[cfg(not(debug_assertions))]
  #[inline]
  pub fn set_debug_name<T: vk::Handle>(&self, _object: T, _name: &str) {
    // This is a no-op in release builds, and should be optimized away.
  }

  pub fn locked_queue_submit(
    &self,
    queue: vk::Queue,
    submits: &[vk::SubmitInfo],
    fence: vk::Fence,
  ) -> ash::prelude::VkResult<()> {
    let _guard = self.submission_lock.lock();
    unsafe { self.handle.queue_submit(queue, submits, fence) }
  }

  pub fn wait_for_semaphore_value(
    &self,
    semaphore: vk::Semaphore,
    value: u64,
    timeout_ns: u64,
  ) -> ash::prelude::VkResult<()> {
    let semaphores = [semaphore];
    let values = [value];
    let wait_info = vk::SemaphoreWaitInfo::default()
      .semaphores(&semaphores)
      .values(&values);

    unsafe {
      self
        .timeline_semaphore
        .wait_semaphores(&wait_info, timeout_ns)
    }
  }
}

pub trait VulkanDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}
pub trait VmaDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}

// 2. Apply to Results containing Vulkan Handles
impl<T: vk::Handle + Copy> VulkanDebugNameExt for ash::prelude::VkResult<T> {
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok(handle) = &self {
      device.set_debug_name(*handle, name);
    }
    self
  }
}

// Implements the trait for ANY `Result` returning a VMA Tuple (Buffer/Image + Allocation)
impl<T, A> VmaDebugNameExt for ash::prelude::VkResult<(T, A)>
where
  T: vk::Handle + Copy,
{
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok((handle, _alloc)) = &self {
      // Apply the debug name to the Vulkan handle (VkBuffer / VkImage)
      // This guarantees it shows up properly in RenderDoc!
      device.set_debug_name(*handle, name);
    }

    // Pass the Result unmodified down the chain
    self
  }
}

impl<T, A1, A2> VmaDebugNameExt for ash::prelude::VkResult<(T, A1, A2)>
where
  T: vk::Handle + Copy,
{
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok((handle, _alloc, _alloc_info)) = &self {
      // Apply the debug name to the Vulkan handle (VkBuffer / VkImage)
      // This guarantees it shows up properly in RenderDoc!
      device.set_debug_name(*handle, name);
    }

    // Pass the Result unmodified down the chain
    self
  }
}

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  queues: Queues,
  instance: &'a instance::Instance,

  device: LogicalDevice,

  res: Arc<spin::rwlock::RwLock<DeviceResources>>,
  callback_stop_signal: Arc<core::sync::atomic::AtomicBool>,

  // Some bookkeeping I don't know where to put
  depth_stencil_format: vk::Format,
  /// Recording command buffers
  recording_command_buffers:
    spin::RwLock<hashbrown::HashMap<CommandBufferHandle, RecordingCmdBufferData>>,
}

const MAX_QUEUE_COUNT: usize = 4;

/// internal queue indicator for `Queues` struct to reference a given queue. Metadata is still held by QueryResult
/// These values are used as shift amounts for bitmasks
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum QueueId {
  GRAPHICS = 1,
  COMPUTE = 2,
  TRANSFER = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Queue {
  handle: vk::Queue,
  index: u32,
  family_index: u32,
}

// ~28 bytes per queue. total for `MAX_QUEUE_COUNT` = 4 at 96 bytes
#[ouroboros::self_referencing]
struct Queues {
  queue_buffer: heapless::Vec<Queue, MAX_QUEUE_COUNT>,
  #[borrows(queue_buffer)]
  #[covariant]
  queue_ref_map: FnvIndexMap<QueueId, &'this Queue, MAX_QUEUE_COUNT>,
}

impl Queues {
  fn from_device(device: &ash::Device, query_result: &utils::PhysicalDeviceQueryResult) -> Self {
    let unique_queue_families = query_result.unique_family_indices_set();
    let mut queue_buffer: heapless::Vec<Queue, MAX_QUEUE_COUNT> = heapless::Vec::new();
    for &family_index in unique_queue_families.iter() {
      let queue_info = vk::DeviceQueueInfo2::default()
        .queue_family_index(family_index)
        .queue_index(0);
      let handle = unsafe { device.get_device_queue2(&queue_info) };
      unsafe {
        queue_buffer.push_unchecked(Queue {
          handle,
          index: 0,
          family_index,
        })
      };
    }

    QueuesBuilder {
      queue_buffer,
      queue_ref_map_builder: |queue_buffer: &heapless::Vec<_, _>| {
        let mut queue_ref_map: FnvIndexMap<QueueId, &Queue, MAX_QUEUE_COUNT> = FnvIndexMap::new();
        let mut queue_type_inserted: u32 = 0;
        for i in 0..queue_buffer.len() {
          if (queue_type_inserted & (1u32 << QueueId::GRAPHICS as u32)) == 0
            && query_result.graphics_queue_family_index == queue_buffer[i].family_index
          {
            queue_ref_map
              .insert(QueueId::GRAPHICS, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::GRAPHICS as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::COMPUTE as u32)) == 0
            && query_result.compute_queue_family_index == queue_buffer[i].family_index
          {
            queue_ref_map
              .insert(QueueId::COMPUTE, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::COMPUTE as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::TRANSFER as u32)) == 0
            && query_result.transfer_queue_family_index == queue_buffer[i].family_index
          {
            queue_ref_map
              .insert(QueueId::TRANSFER, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::TRANSFER as u32;
          }
        }

        queue_ref_map
      },
    }
    .build()
  }

  fn get_graphics_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::GRAPHICS).unwrap())
  }

  fn get_compute_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::COMPUTE).unwrap())
  }

  fn get_transfer_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::TRANSFER).unwrap())
  }
}

impl<'a> Device<'a> {
  /// Initializes a Device directly into the provided memory location
  /// This avoids returning a Device by value (which would probably cause stack overflow)
  pub(super) unsafe fn init_at_ptr(
    dst: *mut Self,
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<()> {
    unsafe { ptr::write(dst, Self::new(instance, index, query_input)?) };
    Ok(())
  }

  pub(super) fn new(
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<Self> {
    let eligible_physical_devices = instance.get_eligible_devices(query_input)?;

    let chosen_physical_device_query_result = match eligible_physical_devices.get(index) {
      Some(chosen_physical_device_query_result) => Ok(chosen_physical_device_query_result),
      None => Err(GpuError::BackendSpecific(format!(
        "There isn't a Vulkan capable device at index {}",
        index
      ))),
    }?;
    let physical_device = chosen_physical_device_query_result.physical_device;

    // 1. enable required
    let mut required_features = utils::RequiredFeatures::new();
    required_features.populate();
    let mut features2 = required_features.as_features2();

    // 2. Setup queue create infos for necessary queues from query result
    let queue_priorities = [1f32];
    let queue_infos_len = chosen_physical_device_query_result.family_count();
    let queue_infos: Vec<_> = (0..queue_infos_len)
      .map(|i| {
        vk::DeviceQueueCreateInfo::default()
          .queue_family_index(i as _)
          .queue_priorities(&queue_priorities)
      })
      .collect();

    // 3. Device creation
    let enabled_extension_names: Vec<_> =
      chosen_physical_device_query_result.enabled_extension_names();
    let device_create_info = vk::DeviceCreateInfo::default()
      .enabled_extension_names(&enabled_extension_names)
      .push_next(&mut features2)
      .queue_create_infos(&queue_infos);

    let device = unsafe {
      instance
        .instance
        .create_device(physical_device, &device_create_info, None)
    }?;

    let queues = Queues::from_device(&device, chosen_physical_device_query_result);
    let res = DeviceResources::new(
      instance,
      physical_device,
      &device,
      chosen_physical_device_query_result
        .unique_family_indices_set()
        .iter(),
    )?;

    // bookkeeping data instantiation
    let depth_stencil_format: vk::Format = 'block: {
      // specification says that at least one of D24/S8 or D32/S8 must be supported
      let mut props = vk::FormatProperties2::default();
      for f in [
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
      ] {
        unsafe {
          instance
            .instance
            .get_physical_device_format_properties2(physical_device, f, &mut props)
        };
        if props
          .format_properties
          .optimal_tiling_features
          .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
          break 'block Ok(f);
        }
      }
      // never reached
      Err(GpuError::UnsupportedFeature)
    }?;

    let create_renderpass2 = ash::khr::create_renderpass2::Device::new(&instance.instance, &device);
    let synchronization2 = ash::khr::synchronization2::Device::new(&instance.instance, &device);
    let buffer_device_address =
      ash::khr::buffer_device_address::Device::new(&instance.instance, &device);
    let timeline_semaphore = ash::khr::timeline_semaphore::Device::new(&instance.instance, &device);

    #[cfg(debug_assertions)]
    let debug_utils = ash::ext::debug_utils::Device::new(&instance.instance, &device);
    #[cfg(target_vendor = "apple")]
    let metal_objects = ash::ext::metal_objects::Device::new(&instance.instance, &device);

    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device: LogicalDevice {
        timeline_semaphore,
        handle: device,
        submission_lock: Mutex::new(()),
        create_renderpass2,
        synchronization2,
        buffer_device_address,
        #[cfg(target_vendor = "apple")]
        metal_objects,
        #[cfg(debug_assertions)]
        debug_utils,
      },
      queues,
      res: Arc::new(spin::rwlock::RwLock::new(res)),
      callback_stop_signal: Arc::new(core::sync::atomic::AtomicBool::new(false)),

      instance,
      depth_stencil_format,
      recording_command_buffers: spin::RwLock::new(hashbrown::HashMap::new()),
    })
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }
}

impl<'a> Drop for Device<'a> {
  fn drop(&mut self) {
    aethervk_oshal_rlib::log!("Device::drop started. Waiting for device idle...");
    // Signal stop to timeline polling task
    self.callback_stop_signal.store(true, Ordering::Release);

    // Wait for the timeline polling task to exit before destroying the device
    while Arc::strong_count(&self.callback_stop_signal) > 1 {
      unsafe { oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1)) };
    }

    unsafe { self.device.device_wait_idle().unwrap_unchecked() };
    aethervk_oshal_rlib::log!("Device::drop device_wait_idle complete. Starting cleanup...");

    self.res.write().cleanup(&self.device);

    aethervk_oshal_rlib::log!("Device::drop cleanup complete. Destroying device...");
    // in the end, destroy the device
    unsafe { self.device.destroy_device(None) };
    aethervk_oshal_rlib::log!("Device::drop finished.");
  }
}

impl<'a> RenderDevice for Device<'a> {
  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void> {
    #[cfg(target_vendor = "apple")]
    {
      if prop == NativeGpuProperty::VulkanMetalDeviceId {
        let mut metal_device_info = vk::ExportMetalDeviceInfoEXT::default();
        let mut metal_objects_info =
          vk::ExportMetalObjectsInfoEXT::default().push_next(&mut metal_device_info);
        unsafe {
          (self.device.metal_objects.fp().export_metal_objects_ext)(
            self.device.handle(),
            core::ptr::from_mut(&mut metal_objects_info),
          );
        };

        return Some(metal_device_info.mtl_device);
      }
    }

    None
  }

  fn print_info(&self) -> String {
    let props = &self.query_result.physical_device_properties;
    let device_name = props
      .device_name_as_c_str()
      .unwrap()
      .to_string_lossy()
      .into_owned();
    let device_type = match props.device_type {
      vk::PhysicalDeviceType::CPU => "CPU",
      vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
      vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
      vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
      _ => "Other",
    };

    let api_major = vk::api_version_major(props.api_version);
    let api_minor = vk::api_version_minor(props.api_version);
    let api_patch = vk::api_version_patch(props.api_version);

    pretty_print_vulkan_device(
      props,
      &device_name,
      device_type,
      self.query_result.family_count() as _,
      api_major,
      api_minor,
      api_patch,
    )
  }

  fn context_id(&self) -> u64 {
    vulkan::VULKAN_RENDER_BACKEND.0
  }

  fn start_frame(&self) -> GpuResult<()> {
    self
      .res
      .read()
      .timeline_manager
      .refresh_cached_value()
      .map(|_| ())
  }

  /// Initializes all archetypes in the order they are declared inside `DeviceResources`
  fn init_archetypes(&self, handle: crate::gpu::PresentationEngineHandle) -> GpuResult<()> {
    // TODO: remove all logs
    let res_guard = self.res.read();
    let timeline = res_guard.timeline_manager.get_cached_value() + 1;
    let mut shader_manager = res_guard.shader_manager.write();

    if res_guard
      .archetypes
      .physical_mesh_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("create_physical_mesh_archetype before shaders");
      let (vkey, fkey, ovkey, ofkey) =
        ensure_physical_mesh_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("create_physical_mesh_archetype after shaders shaders");
      res_guard.create_physical_mesh_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        ovkey,
        ofkey,
        self.depth_stencil_format,
        &self.queues.get_graphics_queue(),
        handle,
        timeline,
      )?;
      oshal::log!("create_physical_mesh_archetype archetype created");
    }

    if res_guard.archetypes.sun_render_archetype.read().is_none() {
      oshal::log!("sun_render_archetype before shaders");
      let (vkey, fkey) = ensure_sun_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("sun_render_archetype after shaders");
      res_guard.create_sun_archetype(
        &self.device,
        &*shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("sun_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .billboard_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("billboard_render_archetype before shaders");
      let (vkey, fkey) = ensure_billboard_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("billboard_render_archetype after shaders");
      res_guard.create_billboard_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("billboard_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .particle_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("particle_render_archetype before shaders");
      let (vkey, fkey) = ensure_particle_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("particle_render_archetype after shaders");
      res_guard.create_particle_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
    }

    if res_guard
      .archetypes
      .particle_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("particle_render_archetype before shaders");
      let (vkey, fkey) = ensure_particle_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("particle_render_archetype after shaders");
      let pe_lock = res_guard.live_presentation_engines.read();
      let pe = pe_lock.get(&handle).unwrap().read();
      let mut write_pipeline = res_guard.pipeline_pool.write();
      res_guard.archetypes.create_particle_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        &pe,
        &res_guard.allocator.allocator,
        &res_guard.discard_pool,
        &res_guard.renderpasses,
        &mut write_pipeline,
        timeline,
      )?;
      oshal::log!("particle_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .cursor_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("cursor_render_archetype before shaders");
      let (vkey, fkey) = ensure_cursor_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("cursor_render_archetype after shaders");
      res_guard.create_cursor_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("cursor_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .marker_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("marker_render_archetype before shaders");
      let (vkey, fkey) = ensure_marker_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("marker_render_archetype after shaders");
      res_guard.create_marker_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("marker_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .measurement_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("measurement_render_archetype before shaders");
      let (vkey, fkey) = ensure_measurement_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("measurement_render_archetype after shaders");
      res_guard.create_measurement_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("measurement_render_archetype archetype created");
    }

    if res_guard.archetypes.sky_render_archetype.read().is_none() {
      oshal::log!("sky_render_archetype before shaders");
      let (vkey, fkey) = ensure_sky_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("sky_render_archetype after shaders");
      res_guard.create_sky_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("sky_render_archetype archetype created");
    }

    if res_guard.archetypes.grid_render_archetype.read().is_none() {
      oshal::log!("grid_render_archetype before shaders");
      let (vkey, fkey) = ensure_grid_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("grid_render_archetype after shaders");
      res_guard.create_grid_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("grid_render_archetype archetype created");
    }

    if res_guard
      .archetypes
      .minimap_render_archetype
      .read()
      .is_none()
    {
      oshal::log!("minimap_render_archetype before shaders");
      let (vkey, fkey) = ensure_minimap_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("minimap_render_archetype after shaders");
      res_guard.create_minimap_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("minimap_render_archetype archetype created");
    }

    if res_guard.archetypes.text_render_archetype.read().is_none() {
      oshal::log!("text_render_archetype before shaders");
      let (vkey, fkey) = ensure_text_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("text_render_archetype after shaders");
      res_guard.create_text_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        &self.queues.get_graphics_queue(),
        handle,
        timeline,
      )?;
      oshal::log!("text_render_archetype archetype created");
    }

    if res_guard.archetypes.bvh_render_archetype.read().is_none() {
      oshal::log!("bvh_render_archetype before shaders");
      let (vkey, fkey) = ensure_bvh_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("bvh_render_archetype after shaders");
      res_guard.create_bvh_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        timeline,
      )?;
      oshal::log!("bvh_render_archetype archetype created");
    }

    if res_guard.archetypes.gizmo_render_archetype.read().is_none() {
      oshal::log!("gizmo_render_archetype before shaders");
      let (vkey, fkey) = ensure_gizmo_shader_modules(&self.device, &mut shader_manager)?;
      oshal::log!("gizmo_render_archetype after shaders");
      let pe_lock = res_guard.live_presentation_engines.read();
      let pe = pe_lock.get(&handle).unwrap().read();
      let mut write_pipeline = res_guard.pipeline_pool.write();
      res_guard.archetypes.create_gizmo_archetype(
        &self.device,
        &shader_manager,
        vkey,
        fkey,
        self.depth_stencil_format,
        &pe,
        &res_guard.allocator.allocator,
        &res_guard.discard_pool,
        &res_guard.renderpasses,
        &mut write_pipeline,
        timeline,
      )?;
      oshal::log!("gizmo_render_archetype archetype created");
    }

    Ok(())
  }

  fn set_line_width(&self, cmd_buffer: gpu::CommandBufferHandle, width: f32) -> GpuResult<()> {
    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] set_line_width: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };
    unsafe {
      self.device.cmd_set_line_width(cmd, width);
    }
    Ok(())
  }

  // TODO move to frame.rs
  fn render_frame(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    render_scene: &gpu::RenderScene,
  ) -> GpuResult<()> {
    if let Some(draw_call) = &render_scene.sky_call {
      gpu::frame::do_draw_sky(self, cmd_buffer, draw_call)?;
    }

    let sun_pos = if let Some(draw_call) = &render_scene.sun_call {
      draw_call.sun_pos()
    } else {
      Vec3f32::zero()
    };
    // TODO setup method (binds pipeline, descriptor set, ...)
    for draw_call in &render_scene.draw_calls {
      gpu::frame::do_draw_call(
        self,
        &render_scene.camera_data,
        sun_pos,
        [1.0, 1.0, 1.0, 1.0],
        cmd_buffer,
        draw_call,
      )?;
    }

    // Draw Sun Volume after opaque meshes so it properly blends over them instead of being overwritten
    if let Some(draw_call) = &render_scene.sun_call {
      gpu::frame::do_draw_sun(self, &render_scene.camera_data, cmd_buffer, draw_call)?;
    }

    if let Some(draw_call) = &render_scene.grid_call {
      let grid_camera = render_scene.camera_data.with_far_plane(10000.0);
      gpu::frame::do_draw_grid(self, cmd_buffer, &grid_camera, draw_call)?;
    }

    for particle_call in &render_scene.particle_calls {
      gpu::frame::do_draw_particle(self, &render_scene.camera_data, cmd_buffer, particle_call)?;
    }

    if let Some(cursor_call) = &render_scene.cursor_call {
      gpu::frame::do_draw_cursor(self, &render_scene.camera_data, cmd_buffer, cursor_call)?;
    }

    for marker_call in &render_scene.marker_calls {
      gpu::frame::do_draw_marker(self, &render_scene.camera_data, cmd_buffer, marker_call)?;
    }

    for measurement_call in &render_scene.measurement_calls {
      gpu::frame::do_draw_measurement(
        self,
        &render_scene.camera_data,
        cmd_buffer,
        measurement_call,
      )?;
    }

    if !render_scene.gizmo_calls.is_empty() {
      // bind the descriptor set
      self.prepare_gizmo_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
      for gizmo_call in &render_scene.gizmo_calls {
        gpu::frame::do_draw_gizmo(self, &render_scene.camera_data, cmd_buffer, gizmo_call)?;
      }
    }

    if render_scene.billboard_calls.len() > 0 {
      self.prepare_billboard_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
    }
    for billboard_call in &render_scene.billboard_calls {
      gpu::frame::do_draw_billboard(self, &render_scene.camera_data, cmd_buffer, billboard_call)?;
      // TODO draw associated text
    }

    if !render_scene.bvh_draw_calls.is_empty() {
      self.prepare_bvh_archetype_for_render_and_bind_pipeline(cmd_buffer)?;
      for bvh_call in &render_scene.bvh_draw_calls {
        gpu::frame::do_bvh_draw_call(
          self,
          cmd_buffer,
          &render_scene.camera_data,
          bvh_call,
          &render_scene.draw_calls,
        )?
      }
    }

    Ok(())
  }

  fn create_presentation_engine(
    &self,
    params: &crate::gpu::PresentationEngineParams,
  ) -> GpuResult<crate::gpu::PresentationEngineHandle> {
    let entry =
      self
        .instance
        .entry_wrapper
        .weak_entry()
        .upgrade()
        .ok_or(GpuError::BackendSpecific(
          "Vulkan Entry wasn't loaded".to_string(),
        ))?;
    let physical_device_handle = unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };
    let presentation_state = swapchain::PresentationState::new(
      &entry,
      &self.instance.instance,
      &self.device,
      physical_device_handle,
      params,
    )?;

    // TODO This should be inside the Device class. not static?
    static NEXT_HANDLE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
    let handle =
      PresentationEngineHandle(NEXT_HANDLE.fetch_add(1, core::sync::atomic::Ordering::Relaxed));

    let res_guard = self.res.read();
    res_guard
      .live_presentation_engines
      .write()
      .insert(handle, spin::RwLock::new(presentation_state));

    Ok(handle)
  }

  // TODO write unit tests about creating a presentation engine and destroying it. Check that, for both
  // TODO windowless and windowed, there shouldn't be any validation errors
  fn destroy_presentation_engine(&self, handle: PresentationEngineHandle) -> GpuResult<()> {
    // check existance
    let presentation_engine_lock = {
      let res = self.res.write();
      let mut engines = res.live_presentation_engines.write();
      if !engines.contains_key(&handle) {
        return Err(GpuError::BackendSpecific(alloc::format!(
          "[Vulkan RenderDevice] destroy_presentation_engine doesn't contain presentation engine {}",
          handle.0
        )));
      }

      unsafe { engines.remove(&handle).unwrap_unchecked() }
    };

    let mut presentation_engine = presentation_engine_lock.write();
    presentation_engine.cleanup(&self.device);
    // presentation engine doesn't implement drop, so we are fine like this
    Ok(())
  }

  fn resize_presentation_engine(
    &self,
    handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    let physical_device_handle = unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };

    // Acquire a single write lock to perform the entire resize operation atomically.
    // This prevents deadlocks and satisfies the borrow checker.
    let res_guard = self.res.read();
    let timeline = res_guard.get_timeline_semaphore_cached_value();

    // Get a mutable reference to the presentation engine to resize it.
    // borrow checker enforces us to reacquire the lock after we call a mutating method
    {
      let engine_lock = res_guard.live_presentation_engines.read();
      let engine = engine_lock.get(&handle).ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] resize_presentation_engine",
      ))?;
      engine.write().resize(
        &self.instance.instance,
        &self.device,
        physical_device_handle,
        width,
        height,
      )?;
    }
    drop(res_guard);

    // After resizing, update dependent resources like pipelines/renderpasses.
    // `update_physical_mesh_archetype_for_presentation_engine` takes `&mut self` (for `wres`)
    // and an immutable `&PresentationState` (for `engine`). This is a valid borrow pattern.
    let res_guard = self.res.read();
    res_guard.update_physical_mesh_archetype_for_presentation_engine(
      &self.device,
      handle,
      timeline,
    )?;
    res_guard.update_cursor_archetype_for_presentation_engine(&self.device, handle, timeline)?;
    res_guard.update_sun_archetype_for_presentation_engine(&self.device, handle, timeline)?;

    Ok(())
  }

  fn get_presentation_engine_extent(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<[u32; 2]> {
    let res_guard = self.res.read();
    let live_engines_lock = res_guard.live_presentation_engines.read();
    if let Some(engine) = live_engines_lock.get(&handle) {
      let e = engine.read().extent();
      Ok([e.0, e.1])
    } else {
      Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] get_presentation_engine_extent",
      ))
    }
  }

  fn acquire_next_image(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<crate::gpu::AcquireResult> {
    let res_guard = self.res.read();
    let live_engines_lock = res_guard.live_presentation_engines.read();
    if let Some(engine) = live_engines_lock.get(&handle) {
      engine
        .write()
        .acquire_next_image(&self.device, self.queues.get_graphics_queue().handle)
    } else {
      Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] acquire_next_image",
      ))
    }
  }

  fn cancel_acquired_image(
    &self,
    handle: PresentationEngineHandle,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let live_engines_lock = res_guard.live_presentation_engines.read();
    let engine = live_engines_lock
      .get(&handle)
      .ok_or(GpuError::InvalidArgument(
        "Vulkan RenderDevice] cancel_aquired_image",
      ))?;
    engine.write().cancel_image(
      &self.device,
      self.queues.get_graphics_queue().handle,
      image_index,
      frame_index,
    )
  }

  /// Note: This function may have the following side effects
  /// - Creation of VkBuffer/VkMemory through VMA for vertex and index buffer associated with given mesh
  /// - Creation of VkImage/VkMemory + VkImageView through VMA for each texture associated with given mesh
  /// for each instance of physical mesh requested to render.
  /// The following resources are instead for every physical mesh, and hence lazily initialized when the first
  /// physical mesh is requested to be rendered
  /// - VkPipeline, VkPipelineLayout, VkPushConstantRange, VkDescriptorSets
  /// - VkRenderPass (and associated VkFramebuffer), which are linked to swapchain,
  ///   hence possibly refreshed each time the swapchain is resized
  /// What is not created by the following function
  /// - VkCommandBuffer, which is instead created through the `record_commands` function from render_path
  /// Note: it assumes that you are preparing for the next frame
  fn get_or_create_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let next_frame_timeline = res_guard.get_timeline_semaphore_cached_value() + 1;
    let current_frame_timeline = next_frame_timeline - 1;

    let (pipeline_key, outline_pipeline_key) = {
      let archetype_guard = res_guard.archetypes.physical_mesh_render_archetype.read();
      if archetype_guard.is_none() {
        return Err(GpuError::InvalidState(
          "[Vulkan RenderDevice] get_or_create_physical_mesh_resources",
        ));
      }
      let archetype_ref = archetype_guard.as_ref().unwrap();
      (
        unsafe { archetype_ref.pipeline_key.unwrap_unchecked() },
        archetype_ref.outline_pipeline_key,
      )
    };

    // Get rendering system Internal Mesh Identifier
    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(entity_id, component);

    // Does the mesh already exist? If so, return cached resource
    {
      let read_resources = res_guard.physical_mesh_resources.read();
      if let Some(resources) = read_resources.as_ref() {
        if let Some(resource) = resources.get(&physical_mesh_id) {
          return Ok(ResourceUploadResult {
            pipeline: pipeline_key,
            outline_pipeline: outline_pipeline_key,
            buffers: physical_mesh_id.into(),
            texture_flags: resource.frontend_texture_flags(),
            emissive_intensity: component.emissive_intensity,
            emissive_color: component.emissive_color,
            indirect_buffer: None,
            descriptor_index: None,
          });
        }
      }
    }

    drop(res_guard);

    // Otherwise, create it inside the resources registry
    // Upload is a blocking operation, so we need to release the read lock on res
    // and acquire it later.
    let graphics_queue = self.queues.get_graphics_queue();
    // Inefficient creation of one-shot command pool. Hopefully, mesh update is infrequent
    let command_pool = {
      let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue.family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
      unsafe { self.device.create_command_pool(&create_info, None) }
    }?;

    let command_buffer = {
      let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0]
    };

    let (resource, texture_flags) = 'resource_creation: {
      let res_guard = self.res.read();
      let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
      unsafe {
        self
          .device
          .begin_command_buffer(command_buffer, &begin_info)?
      };

      let position_data = extract_position_data(&component.mesh);
      let attribute_data = extract_attribute_data(&component.mesh);
      let mut texture_flags: TextureFlags = TextureFlags::empty();
      let albedo_image = component.mesh.albedo_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::ALBEDO;
        Image::new_2d(
          &self.device,
          &res_guard.allocator.allocator,
          command_buffer,
          &res_guard.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureAlbedo_{}", debug_name),
        )
        .ok()
      });
      let normal_image = component.mesh.normal_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::NORMAL;
        Image::new_2d(
          &self.device,
          &res_guard.allocator.allocator,
          command_buffer,
          &res_guard.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureNormal_{}", debug_name),
        )
        .ok()
      });
      let roughness_image = component.mesh.roughness_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::ROUGHNESS;
        Image::new_2d(
          &self.device,
          &res_guard.allocator.allocator,
          command_buffer,
          &res_guard.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureRoughness_{}", debug_name),
        )
        .ok()
      });
      let ao_image = component.mesh.ao_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::AO;
        Image::new_2d(
          &self.device,
          &res_guard.allocator.allocator,
          command_buffer,
          &res_guard.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureAO_{}", debug_name),
        )
        .ok()
      });

      let archetype = res_guard.archetypes.physical_mesh_render_archetype.read();
      let archetype_ref = archetype.as_ref().unwrap();

      let resource = unsafe {
        let dp_lock = res_guard.descriptor_pool.read();
        let descriptor_set = archetype_ref.create_descriptor_set_from_layout_at_index(
          &self.device,
          dp_lock.as_ref().unwrap_unchecked(),
          &res_guard.discard_pool,
          0,
          debug_name,
        )?;
        ForwardMeshRenderResource::new(
          &self.device,
          &res_guard.allocator.allocator,
          command_buffer,
          &res_guard.discard_pool,
          current_frame_timeline,
          &position_data,
          &attribute_data,
          &component.mesh.indices,
          albedo_image,
          normal_image,
          roughness_image,
          ao_image,
          res_guard
            .sky_image
            .read()
            .as_ref()
            .map(|sky| resources::Image {
              image: sky.image,
              image_view: sky.image_view,
              allocation: sky.allocation,
            })
            .or_else(|| {
              Some(resources::Image {
                image: archetype_ref.dummy_texture_handle.image,
                image_view: archetype_ref.dummy_texture_handle.image_view,
                allocation: archetype_ref.dummy_texture_handle.allocation,
              })
            }),
          res_guard.linear_sampler,
          descriptor_set,
          &archetype_ref.dummy_texture_handle,
          debug_name,
        )?
      };

      unsafe {
        self.device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let fence = unsafe {
          self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
        }?;
        self
          .device
          .locked_queue_submit(graphics_queue.handle, &submits, fence)
          .map_err(GpuError::from)?;
        unsafe {
          self.device.wait_for_fences(&[fence], true, u64::MAX)?;
          self.device.destroy_fence(fence, None);
        }
      };

      break 'resource_creation (resource, texture_flags);
    };
    unsafe {
      self.device.destroy_command_pool(command_pool, None);
    }

    let res_guard = self.res.read();
    let mut wresources = res_guard.physical_mesh_resources.write();
    if wresources.is_none() {
      *wresources = Some(hashbrown::HashMap::new());
    }
    // Safety: already checked for existance above
    unsafe {
      wresources
        .as_mut()
        .unwrap_unchecked()
        .insert_unique_unchecked(physical_mesh_id, resource)
    };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: outline_pipeline_key,
      buffers: physical_mesh_id.into(),
      texture_flags,
      emissive_intensity: component.emissive_intensity,
      emissive_color: component.emissive_color,
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn generate_sky(&self) -> GpuResult<()> {
    let graphics_queue = self.queues.get_graphics_queue();
    let compute_queue = self.queues.get_compute_queue();
    let (sky_image, compute_pipeline, descriptor_set, pipeline_layout, descriptor_pool, set_layout) = {
      // Acquire all locks here
      let res_guard = self.res.read();
      let sky_image = res_guard.sky_image.read();
      if sky_image.is_some() {
        return Ok(()); // Sky is already generated, do not destroy and recreate it
      }
      drop(sky_image);

      let comp_key = {
        let mut shader_manager = res_guard.shader_manager.write();
        ensure_skygen_shader_module(&self.device, &mut shader_manager)?
      };

      let shader_module = {
        let shader_manager = res_guard.shader_manager.read();
        let shader = shader_manager
          .get(comp_key)
          .ok_or(GpuError::InvalidShader)?;
        shader.module.get()
      };

      // Create sky image 2048x2048
      let sky_image = resources::Image::new_storage_2d(
        &self.device,
        &res_guard.allocator.allocator,
        2048,
        2048,
        vk::Format::R16G16B16A16_SFLOAT,
        graphics_queue.family_index,
        compute_queue.family_index,
        "Sky",
      )?;

      // Create Descriptor Set Layout
      let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

      let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
      let set_layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

      // Create Pipeline Layout
      let set_layouts = [set_layout];
      let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
      let pipeline_layout = unsafe {
        self
          .device
          .create_pipeline_layout(&pipeline_layout_info, None)
      }?;

      // Create Descriptor Pool and Set
      let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)];
      let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
      let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

      let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
      let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

      // Write descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(sky_image.image_view.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      // Get or Create Compute Pipeline
      let mut compute_info = pipelines::ComputeInfo::default();
      compute_info.shader_module = shader_module;
      compute_info.pipeline_layout = pipeline_layout;

      compute_info.add_specialization_constant_u32(
        vk::SpecializationMapEntry {
          constant_id: 0,
          offset: 0,
          size: 4,
        },
        16,
      );
      compute_info.add_specialization_constant_u32(
        vk::SpecializationMapEntry {
          constant_id: 1,
          offset: 4,
          size: 4,
        },
        16,
      );

      let compute_pipeline = res_guard
        .pipeline_pool
        .write()
        .get_or_create_compute_pipeline(&self.device, &compute_info)?;
      (
        sky_image,
        compute_pipeline.get(),
        descriptor_set,
        pipeline_layout,
        descriptor_pool,
        set_layout,
      )
    }; // <--- All Locks dropped here

    // Create Command Pool and Buffer for compute
    let command_pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(compute_queue.family_index)
      .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }?;

    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
      .command_pool(command_pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);
    let command_buffer = unsafe { self.device.allocate_command_buffers(&command_buffer_info) }?[0];

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self
        .device
        .begin_command_buffer(command_buffer, &begin_info)?;

      // Transition to GENERAL
      let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sky_image.image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );
      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dep_info);

      // Dispatch
      self.device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipeline,
      );
      self.device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
      );
      self
        .device
        .cmd_dispatch(command_buffer, 2048 / 16, 2048 / 16, 1);

      // Transition to SHADER_READ_ONLY_OPTIMAL
      let mut barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(sky_image.image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );

      barrier2 = barrier2
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dep_info2);

      self.device.end_command_buffer(command_buffer)?;

      // Submit
      let submit_info =
        vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
      let fence = {
        self
          .device
          .create_fence(&vk::FenceCreateInfo::default(), None)
      }?;
      oshal::log!("generate_sky: submitting to compute queue...");
      self
        .device
        .locked_queue_submit(compute_queue.handle, &[submit_info], fence)
        .map_err(GpuError::from)?;
      // Never do wait for fences while holding a lock.
      oshal::log!("generate_sky: waiting for fences...");
      self.device.wait_for_fences(&[fence], true, u64::MAX)?;
      oshal::log!("generate_sky: done waiting");
      self.device.destroy_fence(fence, None);

      self.device.destroy_command_pool(command_pool, None);
      self.device.destroy_descriptor_pool(descriptor_pool, None);
      self.device.destroy_pipeline_layout(pipeline_layout, None);
      self.device.destroy_descriptor_set_layout(set_layout, None);
    }

    // In case it already has an image, destroy it
    let res_guard = self.res.read();
    let mut wsky_image = res_guard.sky_image.write();
    if wsky_image.is_some() {
      unsafe {
        vk_mem::ffi::vmaDestroyImage(
          res_guard.allocator.allocator.get_raw(),
          wsky_image.as_ref().unwrap().image.get(),
          wsky_image.as_ref().unwrap().allocation.get_raw(),
        );
        self
          .device
          .destroy_image_view(wsky_image.as_ref().unwrap().image_view.get(), None);
      }
    }
    *wsky_image = Some(sky_image);

    Ok(())
  }

  fn get_or_create_billboard_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    // ensure that the archetype for billboards exists
    let res_guard = self.res.read();
    let archetype = res_guard.archetypes.billboard_render_archetype.read();
    let archetype_not_exists = archetype.is_none();
    if archetype_not_exists {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_or_create_billboard_resources",
      ));
    }

    let archetype_ref = archetype.as_ref().unwrap();

    // Safety: Archetype, once properly constructed, has everything populated
    let pipeline_key = unsafe { archetype_ref.pipeline_key.unwrap_unchecked() };

    // the billboard doesn't have descriptor sets or vertex/index buffers
    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn get_or_create_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let archetype = res_guard.archetypes.cursor_render_archetype.read();
    let archetype_not_exists = archetype.is_none();
    // ensure that the archetype for cursors exists
    if archetype_not_exists {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_or_create_cursor_resources | archetype doesn't exist",
      ));
    }

    let archetype_ref = archetype.as_ref().unwrap();

    // Safety: Archetype, once properly constructed, has everything populated
    let pipeline_key = archetype_ref.pipeline_key.ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] get_or_create_cursor_resources | no pipeline key",
    ))?;

    // the cursor doesn't have descriptor sets or vertex/index buffers
    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let archetype_guard = res_guard.archetypes.gizmo_render_archetype.read();
    if archetype_guard.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] prepare_gizmo_archetype_for_render_and_bind_pipeline",
      ));
    }
    let archetype = archetype_guard.as_ref().unwrap();
    let pipeline = archetype.pipeline_key.unwrap();

    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument("[Vulkan RenderDevice] prepare_gizmo_archetype_for_render_and_bind_pipeline | invalid command buffer handle"))?;
      data.command_buffer.get()
    };

    let p = res_guard
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline)
      .unwrap();

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.get());
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        archetype.pipeline_layout.get(),
        0,
        &[archetype.descriptor_set.get()],
        &[],
      );
    }
    Ok(())
  }

  fn update_gizmo_instance(
    &self,
    entity: EntityId,
    model: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<u32> {
    let res_guard = self.res.read();
    let mut archetype_guard = res_guard.archetypes.gizmo_render_archetype.write();
    let archetype = archetype_guard.as_mut().ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] update_gizmo_instance | no archetype",
    ))?;

    // Check if we already have a buffer for this entity (we hash the entity ID)
    let mut hasher = aethervk_oshal_rlib::hash::FnvHasher::new();
    core::hash::Hash::hash(&entity, &mut hasher);
    let entity_hash = core::hash::Hasher::finish(&hasher);
    let buffer_index =
      (entity_hash % resources::GizmoRenderResourceArchetype::MAX_BUFFER_COUNT as u64) as u32;

    let mut buffers = archetype.host_buffers.write();

    // We just recreate the buffer every frame for simplicity, or we can update it if it is mapped
    // Let's just create a new one and discard the old one
    let data: [[f32; 16]; 1] = [model.into()];

    // We need a command buffer to create it using staging, BUT `GizmoRenderResourceArchetype::MAX_BUFFER_COUNT` is big.
    // Actually we can just create a buffer with HOST_VISIBLE and HOST_COHERENT and write to it directly without staging.
    // Wait, the `create_buffer_with_staging` requires a command buffer. I should write a simple `create_host_visible_buffer`.
    let buffer_size = core::mem::size_of::<[f32; 16]>() as u64;
    let buffer_info = vk::BufferCreateInfo::default()
      .size(buffer_size)
      .usage(vk::BufferUsageFlags::STORAGE_BUFFER);

    let alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::Auto,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT,
      ..Default::default()
    };

    let (vk_buf, alloc, alloc_info_res) = unsafe {
      res_guard
        .allocator
        .allocator
        .create_buffer_get_info(&buffer_info, &alloc_info)
        .map_err(|e| GpuError::BackendSpecific(e.to_string()))?
    };

    unsafe {
      core::ptr::copy_nonoverlapping(
        data.as_ptr() as *const u8,
        alloc_info_res.mapped_data as *mut u8,
        buffer_size as usize,
      );
    }

    let new_buffer = resources::Buffer {
      buffer: unsafe { NonZeroHandle::new_unchecked(vk_buf) },
      allocation: alloc,
    };

    if let Some(old_buffer) = buffers.insert(buffer_index, new_buffer) {
      let timeline = res_guard.get_timeline_semaphore_cached_value() + 1;
      res_guard.discard_pool.discard_buffer(
        res_guard.allocator.allocator.get_raw(),
        old_buffer.buffer.get(),
        old_buffer.allocation,
        timeline,
      );
    }

    // Update descriptor set
    let buffer_info_vk = vk::DescriptorBufferInfo::default()
      .buffer(vk_buf)
      .offset(0)
      .range(vk::WHOLE_SIZE);

    let write = vk::WriteDescriptorSet::default()
      .dst_set(archetype.descriptor_set.get())
      .dst_binding(0)
      .dst_array_element(buffer_index)
      .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
      .buffer_info(core::slice::from_ref(&buffer_info_vk));

    unsafe {
      self
        .device
        .update_descriptor_sets(core::slice::from_ref(&write), &[]);
    }

    Ok(buffer_index)
  }

  fn get_or_create_measurement_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let arch = res_guard.archetypes.measurement_render_archetype.read();
    let archetype_not_exists = arch.is_none();
    if archetype_not_exists {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_or_create_measurement_resources",
      ));
    }

    let arch_ref = arch.as_ref().unwrap();
    let pipeline_key = unsafe { arch_ref.pipeline_key.unwrap_unchecked() };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE,
      texture_flags: TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn get_or_create_marker_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let arch = res_guard.archetypes.marker_render_archetype.read();
    let archetype_not_exists = arch.is_none();
    if archetype_not_exists {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_or_create_marker_resources",
      ));
    }

    let arch_ref = arch.as_ref().unwrap();
    let pipeline_key = unsafe { arch_ref.pipeline_key.unwrap_unchecked() };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE,
      texture_flags: TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn get_or_create_gizmo_resources(
    &self,
    _handle: PresentationEngineHandle,
  ) -> GpuResult<crate::gpu::frame::ResourceUploadResult> {
    let res_guard = self.res.read();
    let archetype_guard = res_guard.archetypes.gizmo_render_archetype.read();
    let archetype = archetype_guard.as_ref().ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] get_or_create_gizmo_resources | archetype missing",
    ))?;

    let pipeline_key = archetype.pipeline_key.ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] get_or_create_gizmo_resources | pipeline key absent",
    ))?;

    Ok(crate::gpu::frame::ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::GpuResourceHandle(0),
      texture_flags: crate::gpu::TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0, 0.0, 0.0],
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  fn upload_particle_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::ParticleDrawCall],
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let archetype_guard = res_guard.archetypes.particle_render_archetype.read();
    if archetype_guard.is_none() {
      return Ok(());
    }
    let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

    let mut staging_arena_guard = res_guard.frame_staging_arena.write();
    let staging_arena = staging_arena_guard.as_mut().unwrap();

    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] upload_particle_systems: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };

    let particle_size = core::mem::size_of::<crate::scene::particles::ParticleData>();
    let indirect_size = core::mem::size_of::<vk::DrawIndirectCommand>();

    let mut current_particle_offset = 0;
    let mut current_indirect_offset = 0;

    for call in particle_calls.iter_mut() {
      let particles = &call.particles;
      if particles.is_empty() {
        continue;
      }

      call.system_particle_offset = current_particle_offset as u32;
      call.system_indirect_offset = current_indirect_offset as u32;

      let particle_data_size = particles.len() * particle_size;
      let p_dst_offset = (current_particle_offset as usize * particle_size) as vk::DeviceSize;
      
      let i_dst_offset = (current_indirect_offset as usize * indirect_size) as vk::DeviceSize;

      let total_size = particle_data_size + indirect_size;
      let (staging_offset, ptr) = staging_arena.allocate(total_size, 16)
          .ok_or(GpuError::InvalidState("Staging arena exhausted!"))?;

      unsafe {
          core::ptr::copy_nonoverlapping(particles.as_ptr() as *const u8, ptr, particle_data_size);
          
          let indirect_cmd = vk::DrawIndirectCommand {
              vertex_count: 4,
              instance_count: particles.len() as u32,
              first_vertex: 0,
              first_instance: current_particle_offset as u32,
          };
          core::ptr::copy_nonoverlapping(&indirect_cmd as *const _ as *const u8, ptr.add(particle_data_size), indirect_size);
      }

      let p_copy = vk::BufferCopy::default().src_offset(staging_offset as u64).dst_offset(p_dst_offset).size(particle_data_size as u64);
      let i_copy = vk::BufferCopy::default().src_offset((staging_offset + particle_data_size) as u64).dst_offset(i_dst_offset).size(indirect_size as u64);

      unsafe {
          self.device.cmd_copy_buffer(cmd, staging_arena.buffer, archetype.mega_particle_buffer, core::slice::from_ref(&p_copy));
          self.device.cmd_copy_buffer(cmd, staging_arena.buffer, archetype.mega_indirect_buffer, core::slice::from_ref(&i_copy));
      }

      let mut p_barrier = vk::BufferMemoryBarrier::default().buffer(archetype.mega_particle_buffer).offset(p_dst_offset).size(particle_data_size as u64);
      let mut i_barrier = vk::BufferMemoryBarrier::default().buffer(archetype.mega_indirect_buffer).offset(i_dst_offset).size(indirect_size as u64);

      p_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE; p_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
      i_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE; i_barrier.dst_access_mask = vk::AccessFlags::INDIRECT_COMMAND_READ;
      let src_stage = vk::PipelineStageFlags::TRANSFER;
      let dst_stage = vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT;

      unsafe {
          self.device.cmd_pipeline_barrier(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), &[], &[p_barrier, i_barrier], &[]);
      }

      current_particle_offset += particles.len();
      current_indirect_offset += 1;
    }

    Ok(())
  }

  fn draw_particle_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()> {
    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] draw_particle_indirect: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };

    let res_guard = self.res.read();
    let archetype_guard = res_guard.archetypes.particle_render_archetype.read();
    let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

    let i_offset = (indirect_offset as usize * core::mem::size_of::<vk::DrawIndirectCommand>()) as vk::DeviceSize;
    unsafe {
        self.device.cmd_draw_indirect(
            cmd, 
            archetype.mega_indirect_buffer, 
            i_offset, 
            1, 
            core::mem::size_of::<vk::DrawIndirectCommand>() as u32
        );
    }
    Ok(())
  }

  fn get_particle_pipeline_key(&self) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let archetype = res_guard.archetypes.particle_render_archetype.read();
    archetype
      .as_ref()
      .and_then(|a| a.pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_particle_pipeline_key failed",
      ))
  }

  fn get_sun_pipeline_key(&self) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let sun_archetype = res_guard.archetypes.sun_render_archetype.read();
    sun_archetype
      .as_ref()
      .and_then(|a| a.pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_sun_pipeline_key failed",
      ))
  }

  fn get_sky_pipeline_key(&self) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let sky_archetype = res_guard.archetypes.sky_render_archetype.read();
    sky_archetype
      .as_ref()
      .and_then(|a| a.pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_sky_pipeline_key failed",
      ))
  }

  fn get_grid_pipeline_kay(&self) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let grid_archetype = res_guard.archetypes.grid_render_archetype.read();
    grid_archetype
      .as_ref()
      .and_then(|a| a.pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_grid_pipeline_key failed",
      ))
  }

  fn get_bvh_pipeline_kay(&self) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let bvh_archetype = res_guard.archetypes.bvh_render_archetype.read();
    bvh_archetype
      .as_ref()
      .and_then(|a| a.pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] get_bvh_pipeline_key failed]",
      ))
  }

  fn allocate_rasterized_font_atlas(&self, hash: u64, font_atlas: FontAtlas) -> GpuResult<u32> {
    let res_guard = self.res.read();
    let timeline = res_guard.get_timeline_semaphore_cached_value();
    let mut archetype = res_guard.archetypes.text_render_archetype.write();
    archetype
      .as_mut()
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] allocate_rasterized_font_atlas: text archetype not found",
      ))
      .and_then(|a| {
        a.upload_font_atlas(
          &self.device,
          &self.queues.get_graphics_queue(),
          &res_guard.allocator.allocator,
          &res_guard.discard_pool,
          timeline,
          hash,
          font_atlas,
        )
      })
  }

  fn free_rasterized_font_atlas(&self, hash: u64, _font_atlas_id: u32) -> GpuResult<()> {
    let res_guard = self.res.read();
    // TODO verify that we don't need + 1 is correct to discard on next timeline?
    let timeline = res_guard.get_timeline_semaphore_cached_value();
    let mut archetype = res_guard.archetypes.text_render_archetype.write();
    archetype
      .as_mut()
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] free_rasterized_font_atlas: text archetype not found",
      ))
      .and_then(|a| a.remove_font_atlas(hash, &res_guard.discard_pool, timeline))
  }

  fn present(
    &self,
    handle: PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<crate::gpu::SwapchainStatus> {
    let res_guard = self.res.read();
    let live_engines_lock = res_guard.live_presentation_engines.read();
    if let Some(engine) = live_engines_lock.get(&handle) {
      let graphics_queue = self.queues.get_graphics_queue().handle;
      unsafe {
        engine.write().submit_image(
          &self.device,
          graphics_queue,
          image_index as u32,
          frame_index as u32,
        )
      }
    } else {
      Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] present: invalid presentation engine",
      ))
    }
  }

  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
    task_id: Option<u64>,
  ) -> GpuResult<()> {
    // SCOPE 1: Lock briefly to extract required state, then drop the lock!
    let (image, width, height, mut wait_value, timeline_sem, task_entry) = {
      let res_guard = self.res.read();
      let engine_lock = res_guard.live_presentation_engines.read();
      let state_lock = engine_lock.get(&handle).ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] download_windowless_image: invalid presentation engine",
      ))?;
      let mut state = state_lock.write();

      if let swapchain::PresentationState::Windowless(windowless) = &mut *state {
        let image = windowless.get_last_submitted_image()?;
        let (width, height) = windowless.extent();

        let (wait_val, entry) = match task_id {
          Some(id) => {
            let registry = res_guard.timeline_manager.task_registry.read();
            if let Some(entry) = registry.get(&id) {
              (
                entry.target_value.load(Ordering::Acquire),
                Some(entry.clone()),
              )
            } else {
              return Err(GpuError::InvalidArgument(
                "[Vulkan RenderDevice] download_windowless_image: no task id",
              ));
            }
          }
          None => (windowless.get_last_submitted_timeline_value(), None),
        };

        (
          image,
          width,
          height,
          wait_val,
          res_guard.timeline_manager.semaphore.get(),
          entry,
        )
      } else {
        return Err(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] download_windowless_image: windowed presentation engine cannot download (TODO)",
        ));
      }
    }; // <-- Locks safely dropped here!

    let buffer_size = (width * height * 4) as vk::DeviceSize;
    if buffer.len() != buffer_size as usize {
      return Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] download_windowless_image: buffer size does not match",
      ));
    }

    // FIX HANG: If wait_value is u64::MAX, the RenderThread hasn't called
    // `submit_command_buffer` yet. We must briefly spin loop until it has been assigned.
    if let Some(entry) = task_entry {
      while wait_value == u64::MAX {
        core::hint::spin_loop();
        wait_value = entry.target_value.load(Ordering::Acquire);
      }
    }

    // BLOCKING WAIT 1 (Safe, because `self.res` is no longer locked!)
    oshal::log!("DEBUG RUST: waiting for timeline {}", wait_value);
    self
      .device
      .wait_for_semaphore_value(timeline_sem, wait_value, u64::MAX)?;
    oshal::log!("DEBUG RUST: timeline {} reached", wait_value);

    // Staging buffer creation
    let buffer_info = vk::BufferCreateInfo::default()
      .size(buffer_size)
      .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED;

    // SCOPE 2: Lock briefly to allocate resource memory
    let graphics_queue = self.queues.get_graphics_queue();
    let (staging_buffer, alloc, command_pool, command_buffer, alloc_info_res) = {
      let res_guard = self.res.read();
      let (staging_buffer, alloc) = unsafe {
        res_guard
          .allocator
          .allocator
          .create_buffer(&buffer_info, &alloc_info)
      }?;
      let alloc_info_res = res_guard.allocator.allocator.get_allocation_info(&alloc);

      let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue.family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
      let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }?;

      let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      let command_buffer =
        unsafe { self.device.allocate_command_buffers(&command_buffer_info) }?[0];

      (
        staging_buffer,
        alloc,
        command_pool,
        command_buffer,
        alloc_info_res,
      )
    }; // <-- Locks dropped again

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self
        .device
        .begin_command_buffer(command_buffer, &begin_info)?;

      let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );
      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&image_barrier));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dep_info);

      let region = vk::BufferImageCopy::default()
        .image_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .image_extent(vk::Extent3D {
          width,
          height,
          depth: 1,
        });

      self.device.cmd_copy_image_to_buffer(
        command_buffer,
        image.get(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        staging_buffer,
        &[region],
      );

      self.device.end_command_buffer(command_buffer)?;
    }

    let submit_info =
      vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
    let fence = unsafe {
      self
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)?
    };

    self
      .device
      .locked_queue_submit(graphics_queue.handle, &[submit_info], fence)
      .map_err(GpuError::from)?;

    oshal::log!("DEBUG RUST: waiting for fences");
    // BLOCKING WAIT 2 (Safe, locks are dropped)
    unsafe {
      self.device.wait_for_fences(&[fence], true, u64::MAX)?;
      oshal::log!("DEBUG RUST: fences done");
      self.device.destroy_fence(fence, None);
      self.device.destroy_command_pool(command_pool, None);
    };

    // SCOPE 3: Lock briefly to invalidate mapped memory and run cleanup
    let mapped_ptr = alloc_info_res.mapped_data as *const u8;
    if !mapped_ptr.is_null() {
      let res_guard = self.res.read();
      res_guard
        .allocator
        .allocator
        .invalidate_allocation(&alloc, 0, vk::WHOLE_SIZE)?;
      unsafe {
        core::ptr::copy_nonoverlapping(mapped_ptr, buffer.as_mut_ptr(), buffer_size as usize);
      }
    }

    unsafe {
      let mut mut_alloc = alloc;
      let res_guard = self.res.read();
      res_guard
        .allocator
        .allocator
        .destroy_buffer(staging_buffer, &mut mut_alloc);
    }

    Ok(())
  }

  fn get_command_buffer(&self) -> GpuResult<gpu::CommandBufferHandle> {
    let res_guard = self.res.read();
    // Command buffers just get a dummy increasing ID for the hash map. Not the timeline.
    let cmd_id = res_guard
      .next_cmd_id
      .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    // even increasing, so it shouldn't be there
    debug_assert!(
      !self
        .recording_command_buffers
        .read()
        .contains_key(&CommandBufferHandle(cmd_id))
    );
    let cmd = unsafe {
      res_guard
        .command_pools
        .read()
        .get_unchecked(self.queues.get_graphics_queue().index as usize)
        .as_ref()
        .unwrap_unchecked()
        .allocate_primary(&self.device, this_thread::id(), CommandBufferId(cmd_id))
    }?;
    self.recording_command_buffers.write().insert(
      CommandBufferHandle(cmd_id),
      RecordingCmdBufferData::new(unsafe { NonZeroHandle::new_unchecked(cmd) }),
    );

    Ok(CommandBufferHandle(cmd_id))
  }

  // TODO group all &'static str error message
  fn begin_command_buffer(&self, cmd_buffer: gpu::CommandBufferHandle) -> GpuResult<()> {
    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = cmd_buffers
      .get_mut(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] begin_command_buffer: invalid command buffer handle",
      ))?;

    if data.has_begun {
      return Ok(());
    }

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self
        .device
        .begin_command_buffer(data.command_buffer.get(), &begin_info)?;
    }
    data.has_begun = true;

    Ok(())
  }

  fn begin_render_pass(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let timeline = res_guard.get_timeline_semaphore_cached_value();
    let presentation_engines_guard = res_guard.live_presentation_engines.read();
    let cmd_buffers = self.recording_command_buffers.read();
    if !cmd_buffers.contains_key(&cmd_buffer) {
      return Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] begin_render_pass invalid command buffer handle",
      ));
    }
    if !presentation_engines_guard.contains_key(&presentation_engine) {
      return Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] begin_render_pass invalid presentation_engine handle",
      ));
    }

    let data = unsafe { cmd_buffers.get(&cmd_buffer).unwrap_unchecked() };

    if !data.has_begun {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] begin_render_pass command buffer not begun",
      ));
    }

    let wpresentation_engine = unsafe {
      presentation_engines_guard
        .get(&presentation_engine)
        .unwrap_unchecked()
    }
    .write();
    if acquire_result.status.needs_resize() {
      // The caller should handle the resize.
      return Err(GpuError::ResizeRequired);
    }
    drop(cmd_buffers);

    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = unsafe { cmd_buffers.get_mut(&cmd_buffer).unwrap_unchecked() };
    data.presentation = Some(RecordingCmdBufferDataPresentation {
      acquire_result: *acquire_result,
      presentation_engine,
    });
    let (render_pass, framebuffer) = self.res.read().renderpasses.get_or_create_render_pass(
      RenderPassSpecification::single_pass(&wpresentation_engine, self.depth_stencil_format),
      acquire_result.frame_index as u32,
      &self.device,
      &self.res.read().allocator.allocator,
      &self.res.read().discard_pool,
      timeline,
    )?;

    let cmd = data.command_buffer.get();
    let mut black = [vk::ClearValue::default(), vk::ClearValue::default()]; // 2 attachments
    self
      .res
      .read()
      .renderpasses
      .get_clear_values_render_pass(RenderPassType::ColorDepthSingleSubpass, &mut black)?;

    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
      .render_pass(render_pass.get())
      .framebuffer(framebuffer.get())
      .render_area(vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
          width: wpresentation_engine.extent().0,
          height: wpresentation_engine.extent().1,
        },
      })
      .clear_values(&black);
    let subpass_begin_info = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);

    unsafe {
      self.device.create_renderpass2.cmd_begin_render_pass2(
        cmd,
        &render_pass_begin_info,
        &subpass_begin_info,
      )
    };

    Ok(())
  }

  fn set_viewport(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    viewport: &crate::gpu::Viewport,
  ) -> GpuResult<()> {
    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] set_viewport: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };
    let vk_viewport = vk::Viewport {
      x: viewport.x,
      y: viewport.y,
      width: viewport.width,
      height: viewport.height,
      min_depth: viewport.min_depth,
      max_depth: viewport.max_depth,
    };
    unsafe {
      self.device.cmd_set_viewport(cmd, 0, &[vk_viewport]);
    }

    Ok(())
  }

  fn set_scissor(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    scissor: &gpu::Rect2D,
  ) -> GpuResult<()> {
    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] set_scissor: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };
    let vk_scissor = vk::Rect2D {
      offset: vk::Offset2D {
        x: scissor.offset[0],
        y: scissor.offset[1],
      },
      extent: vk::Extent2D {
        width: scissor.extent[0],
        height: scissor.extent[1],
      },
    };
    unsafe {
      self.device.cmd_set_scissor(cmd, 0, &[vk_scissor]);
    }

    Ok(())
  }

  fn bind_pipeline(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    pipeline_key: crate::gpu::PipelineKey,
  ) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    if !cmd_buffers.contains_key(&cmd_buffer) {
      return Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] bind_pipeline: invalid command buffer handle]",
      ));
    }

    drop(cmd_buffers);
    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = unsafe { cmd_buffers.get_mut(&cmd_buffer).unwrap_unchecked() };
    let cmd = data.command_buffer.get();

    let pipeline = self
      .res
      .read()
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline_key)
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] bind_pipeline: get_graphics_pipeline",
      ))?;

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
    }
    // ready to discard it if necessary (on resize)
    data.bound_pipeline = Some(pipeline);

    Ok(())
  }

  fn check_billboard_texture_id(&self, texture_id: u64) -> GpuResult<()> {
    let res = self.res.read();
    let billboard_resources = res.billboard_resources.read();
    if billboard_resources.len() > texture_id as usize {
      Ok(())
    } else {
      Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] check_billboard_texture_id: billboard resources non existent",
      ))
    }
  }

  fn add_billboard_texture(&self, texture: &Texture) -> GpuResult<()> {
    let graphics_queue = self.queues.get_graphics_queue();

    let command_buffer = {
      let res = self.res.read();
      let timeline = res.get_timeline_semaphore_cached_value();
      let mut billboard_resources = res.billboard_resources.write();
      let billboard_render_archetype = res.archetypes.billboard_render_archetype.read();

      // Create throwaway command buffer
      let command_pool = {
        let create_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(graphics_queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        unsafe { self.device.create_command_pool(&create_info, None) }
      }?;

      let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0]
      };

      // start command buffer
      let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
      unsafe {
        self
          .device
          .begin_command_buffer(command_buffer, &begin_info)?
      };

      let image = {
        let billboard_render_archetype_ref = billboard_render_archetype.as_ref().unwrap();
        billboard_render_archetype_ref.add_texture(
          &self.device,
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          timeline,
          texture,
          res.linear_sampler,
          billboard_resources.len() as _,
          &alloc::format!("BillBoard_{}", billboard_resources.len()),
        )?
      };

      billboard_resources.push(image);

      // TODO: refactor this together with command buffer creation into a utility function called
      // "one_time_gpu_upload"
      unsafe {
        self.device.end_command_buffer(command_buffer)?;
      }

      command_buffer
    }; // <-- All locks released here (ready for wait_for_fences)

    unsafe {
      let command_buffers = [command_buffer];
      let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
      let fence = self
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)?;
      self
        .device
        .locked_queue_submit(graphics_queue.handle, &submits, fence)
        .map_err(GpuError::from)?;
      oshal::log!("DEBUG RUST: waiting for fences");
      self.device.wait_for_fences(&[fence], true, u64::MAX)?;
      oshal::log!("DEBUG RUST: fences done");
      self.device.destroy_fence(fence, None);
    };

    Ok(())
  }

  fn bind_buffers(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    _pipeline: crate::gpu::PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] bind_buffers: invalid command buffer handle",
        ))?;
      data.command_buffer.get()
    };

    let (
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      pipeline_layout,
      descriptor_set,
    ) = {
      let physical_mesh_id = RenderableInstanceId(buffers.0);
      let res_guard = self.res.read();
      let physical_mesh_resources_guard = res_guard.physical_mesh_resources.read();
      let resource = physical_mesh_resources_guard
        .as_ref()
        .and_then(|map| map.get(&physical_mesh_id))
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] bind_buffers: couldn't get render mesh resource",
        ))?;

      let physical_mesh_render_archetype_guard =
        res_guard.archetypes.physical_mesh_render_archetype.read();
      let archetype =
        physical_mesh_render_archetype_guard
          .as_ref()
          .ok_or(GpuError::InvalidState(
            "[Vulkan RenderDevice] bind_buffers: archetype absent",
          ))?;

      (
        resource.position_vertex_buffer.buffer.get(),
        resource.attributes_vertex_buffer.buffer.get(),
        resource.index_buffer.buffer.get(),
        archetype.pipeline_layout.get(),
        resource.descriptor_set.get(),
      )
    };

    // Bind vertex buffers
    unsafe {
      self.device.cmd_bind_vertex_buffers(
        cmd,
        0,
        &[position_vertex_buffer, attributes_vertex_buffer],
        &[0, 0],
      );
    }

    // Bind index buffer
    unsafe {
      self
        .device
        .cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
    }

    // Update and bind descriptor sets
    // Errata: modifying a descriptor set while the GPU is executing (or about to execute) commands that use that set is a severe data race. Since multiple meshes share that one archetype set, Mesh B will overwrite Mesh A's descriptors before the GPU even gets a chance to render Mesh A
    unsafe {
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
      );
    }

    Ok(())
  }

  fn push_constants_raw(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    archetype: ArchetypeId,
    push_constants_bytes: &[u8],
  ) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] push_constants_raw: invalid command buffer handle",
      ))?;

    let res_guard = self.res.read();

    let layout = match archetype {
      ArchetypeId::Sun => res_guard
        .archetypes
        .sun_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::PhysicalMesh => res_guard
        .archetypes
        .physical_mesh_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Billboard => res_guard
        .archetypes
        .billboard_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Cursor => res_guard
        .archetypes
        .cursor_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Marker => res_guard
        .archetypes
        .marker_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Measurement => res_guard
        .archetypes
        .measurement_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Sky => res_guard
        .archetypes
        .sky_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Grid => res_guard
        .archetypes
        .grid_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Minimap => res_guard
        .archetypes
        .minimap_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Text => res_guard
        .archetypes
        .text_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Bvh => res_guard
        .archetypes
        .bvh_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Particle => res_guard
        .archetypes
        .particle_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
      ArchetypeId::Gizmo => res_guard
        .archetypes
        .gizmo_render_archetype
        .read()
        .as_ref()
        .map(|a| a.pipeline_layout.get()),
    }
    .ok_or(GpuError::InvalidArgument(
      "[Vulkan RenderDevice] push_constant_raw | archetype not initialized",
    ))?;

    // The slice is already bytes, just pass it directly to Vulkan
    unsafe {
      self.device.cmd_push_constants(
        data.command_buffer.get(),
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants_bytes,
      );
    }
    Ok(())
  }

  fn draw_indexed(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    index_count: u32,
  ) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] draw_indexed: invalid command buffer handle",
      ))?;

    let cmd = data.command_buffer.get();

    unsafe {
      self.device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
    }

    Ok(())
  }

  fn draw(&self, cmd_buffer: crate::gpu::CommandBufferHandle, vertex_count: u32) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] draw: invalid command buffer handle]",
      ))?;

    let cmd = data.command_buffer.get();

    unsafe {
      self.device.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }

    Ok(())
  }

  fn draw_indirect(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    indirect_buffer: GpuResourceHandle,
    offset: u64,
    draw_count: u32,
    stride: u32,
  ) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] draw_indirect: invalid command buffer handle",
      ))?;

    let cmd = data.command_buffer.get();
    let buffer = ash::vk::Buffer::from_raw(indirect_buffer.0);

    unsafe {
      self
        .device
        .cmd_draw_indirect(cmd, buffer, offset, draw_count, stride);
    }

    Ok(())
  }

  fn update_sun(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    resolution: (u32, u32, u32),
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let timeline = res_guard.get_timeline_semaphore_cached_value();

    let sun_archetype_not_exists = res_guard.archetypes.sun_render_archetype.read().is_none();
    if sun_archetype_not_exists {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] update_sun: archetype doesn't exist",
      ));
    }

    let mut sun_res_lock = res_guard.sun_resources.write();
    if sun_res_lock.is_none() {
      *sun_res_lock = Some(hashbrown::HashMap::new());
    }
    let map = sun_res_lock.as_mut().unwrap();
    if !map.contains_key(&entity_id) {
      let graphics_queue = self.queues.get_graphics_queue();
      let compute_queue = self.queues.get_compute_queue();
      let image = resources::Image::new_storage_3d(
        &self.device,
        &res_guard.allocator.allocator,
        resolution.0,
        resolution.1,
        resolution.2,
        vk::Format::R16G16B16A16_SFLOAT,
        graphics_queue.family_index,
        compute_queue.family_index,
        "Sun",
      )?;

      // Now run sungen.comp
      let comp_key = {
        let mut shader_manager = res_guard.shader_manager.write();
        ensure_sungen_shader_module(&self.device, &mut shader_manager)?
      };
      let shader_module = {
        let shader_manager = res_guard.shader_manager.read();
        let shader = shader_manager
          .get(comp_key)
          .ok_or(GpuError::InvalidShader)?;
        shader.module.get()
      };

      // Create Descriptor Set Layout
      let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

      let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
      let set_layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

      // Create Pipeline Layout
      let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(8)]; // 64-bit pointer
      let set_layouts = [set_layout];
      let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
      let pipeline_layout = unsafe {
        self
          .device
          .create_pipeline_layout(&pipeline_layout_info, None)
      }?;

      // Create Descriptor Pool and Set
      let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)];
      let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
      let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

      let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
      let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

      // Write descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(image.image_view.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      // Get or Create Compute Pipeline
      let mut compute_info = pipelines::ComputeInfo::default();
      compute_info.shader_module = shader_module;
      compute_info.pipeline_layout = pipeline_layout;

      let compute_pipeline = res_guard
        .pipeline_pool
        .write()
        .get_or_create_compute_pipeline(&self.device, &compute_info)?;

      // Buffer for SunParams
      let params_size = core::mem::size_of::<[f32; 6]>() as u64; // 6 floats
      let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
      allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
      allocation_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED
        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;

      let buffer_info = vk::BufferCreateInfo::default()
        .size(params_size)
        .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
      let (params_buffer, params_alloc) = unsafe {
        res_guard
          .allocator
          .allocator
          .create_buffer(&buffer_info, &allocation_create_info)?
      };

      let alloc_info = res_guard
        .allocator
        .allocator
        .get_allocation_info(&params_alloc);

      unsafe {
        let ptr = alloc_info.mapped_data as *mut [f32; 6];
        *ptr = [
          timeline as f32 * 0.016, // time
          5778.0,                  // photosphereTemp
          1000000.0,               // coronaTemp
          0.6,                     // radius
          0.05,                    // scaleHeight
          15.0,                    // noiseScale
        ];
      }

      let sun_render_archetype = res_guard.archetypes.sun_render_archetype.read();
      let archetype = sun_render_archetype.as_ref().ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] update_sun: archetype absent",
      ))?;

      let graphics_descriptor_set = res_guard
        .descriptor_pool
        .read()
        .as_ref()
        .unwrap()
        .allocate(
          &self.device,
          archetype.descriptor_set_layout.get(),
          &res_guard.discard_pool,
          timeline,
          "Sun",
        )?
        .get();

      // Write graphics descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image.image_view.get())
        .sampler(res_guard.linear_sampler.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(graphics_descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      map.insert(
        entity_id,
        resources::SunRenderResource {
          resolution,
          image: Some(image),
          descriptor_set: Some(unsafe { NonZeroHandle::new_unchecked(graphics_descriptor_set) }),
          is_generated: false,
          compute_descriptor_pool: Some(descriptor_pool),
          compute_descriptor_set_layout: Some(set_layout),
          compute_descriptor_set: Some(descriptor_set),
          compute_pipeline: Some(compute_pipeline),
          compute_pipeline_layout: Some(pipeline_layout),
          params_buffer: Some(params_buffer),
          params_alloc: Some(params_alloc),
        },
      );
    }

    let sun_resource = map.get_mut(&entity_id).unwrap();
    if sun_resource.compute_pipeline.is_none() {
      return Ok(());
    }

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&cmd_buffer).ok_or(GpuError::InvalidState(
      "update_sun: invalid command buffer handle",
    ))?;
    let cmd = data.command_buffer.get();

    unsafe {
      let alloc_info = self
        .res
        .read()
        .allocator
        .allocator
        .get_allocation_info(sun_resource.params_alloc.as_ref().unwrap());
      let ptr = alloc_info.mapped_data as *mut f32;
      *ptr = timeline as f32 * 0.016;

      let _ = self.res.read().allocator.allocator.flush_allocation(
        sun_resource.params_alloc.as_ref().unwrap(),
        0,
        vk::WHOLE_SIZE as u64,
      );

      let bda_info =
        vk::BufferDeviceAddressInfo::default().buffer(sun_resource.params_buffer.unwrap());
      let buffer_address = self
        .device
        .buffer_device_address
        .get_buffer_device_address(&bda_info);

      let old_layout = if sun_resource.is_generated {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
      } else {
        vk::ImageLayout::UNDEFINED
      };
      let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(if sun_resource.is_generated {
          vk::PipelineStageFlags2::FRAGMENT_SHADER
        } else {
          vk::PipelineStageFlags2::NONE
        })
        .src_access_mask(if sun_resource.is_generated {
          vk::AccessFlags2::SHADER_READ
        } else {
          vk::AccessFlags2::NONE
        })
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sun_resource.image.as_ref().unwrap().image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );

      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &dep_info);

      self.device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sun_resource.compute_pipeline.unwrap().get(),
      );
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sun_resource.compute_pipeline_layout.unwrap(),
        0,
        &[sun_resource.compute_descriptor_set.unwrap()],
        &[],
      );
      let push_constants_bytes =
        core::slice::from_raw_parts(&buffer_address as *const _ as *const u8, 8);
      self.device.cmd_push_constants(
        cmd,
        sun_resource.compute_pipeline_layout.unwrap(),
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );

      let group_count_x = (sun_resource.resolution.0 + 7) / 8;
      let group_count_y = (sun_resource.resolution.1 + 7) / 8;
      let group_count_z = (sun_resource.resolution.2 + 7) / 8;
      self
        .device
        .cmd_dispatch(cmd, group_count_x, group_count_y, group_count_z);

      let barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sun_resource.image.as_ref().unwrap().image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );

      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &dep_info2);
    }

    sun_resource.is_generated = true;

    Ok(())
  }

  fn prepare_billboard_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline, layout, d) = {
      let res = self.res.read();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument("[Vulkan RenderDevice] prepare_billboard_archetype_for_render_and_bind_pipeline: invalid command buffer handle"))?;
      let cmd = data.command_buffer.get();
      let billboard_render_archetype = res.archetypes.billboard_render_archetype.read();
      let billboard_render_archetype_ref = billboard_render_archetype
        .as_ref()
        .ok_or(GpuError::InvalidState("billboard archetype absent"))?;
      let d: vk::DescriptorSet = billboard_render_archetype_ref.descriptor_set.get();
      let layout = billboard_render_archetype_ref.pipeline_layout.get();
      let pipeline_key =
        billboard_render_archetype_ref
          .pipeline_key
          .ok_or(GpuError::InvalidState(
            "billboard archetype | pipeline key absent",
          ))?;
      let pipeline = res
        .pipeline_pool
        .read()
        .get_graphics_pipeline(pipeline_key)
        .ok_or(GpuError::InvalidState(
          "billboard archetype | pipeline absent in pool",
        ))?
        .get();
      (cmd, pipeline, layout, d)
    }; // <-- locks released here
    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        1,
        core::slice::from_ref(&d),
        &[],
      )
    };
    Ok(())
  }

  fn prepare_bvh_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline) = {
      let res = self.res.read();

      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers.get(&cmd_buffer).ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] prepare_bvh_archetype_for_render_and_bind_pipeline no cmd buffer",
      ))?;
      let cmd = data.command_buffer.get();

      let a_lock = res.archetypes.bvh_render_archetype.read();
      let bvh_render_archetype = a_lock.as_ref().ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] prepare_bvh_archetype no BVH archetype",
      ))?;
      let pipeline_key = bvh_render_archetype
        .pipeline_key
        .ok_or(GpuError::InvalidState(
          "[Vulkan RenderDevice] prepare_bvh_archetype no BVH pipeline key",
        ))?;
      let pipeline = res
        .pipeline_pool
        .read()
        .get_graphics_pipeline(pipeline_key)
        .ok_or(GpuError::InvalidState(
          "[Vulkan RenderDevice] prepare_bvh_archetype no BVH pipeline",
        ))?
        .get();
      (cmd, pipeline)
    };
    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
      self.device.cmd_set_line_width(cmd, 1.0);
      // now each BVH needs to do push constant and draw.
    }
    Ok(())
  }

  fn prepare_sun_for_render(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity: EntityId,
  ) -> GpuResult<()> {
    let (layout, ds, cmd) = {
      let res = self.res.read();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] prepare_sun_for_render: invalid command buffer handle",
        ))?;
      let cmd = data.command_buffer.get();
      let sun_resource = res.sun_resources.read();
      let sun_archetype = res.archetypes.sun_render_archetype.read();
      let layout = sun_archetype
        .as_ref()
        .map(|a| a.pipeline_layout.get())
        .ok_or(GpuError::InvalidState(
          "[Vulkan RenderDevice] render_frame | couldn't get sun pipeline layout",
        ))?;
      let ds = sun_resource
        .as_ref()
        .and_then(|s_map| s_map.get(&entity))
        .and_then(|s| s.descriptor_set)
        .map(|d| d.get())
        .ok_or(GpuError::InvalidState(
          "[Vulkan RenderDevice] render_frame:couldn't find sun descriptor set",
        ))?;
      (layout, ds, cmd)
    };
    unsafe {
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[ds],
        &[],
      );
    }
    Ok(())
  }

  fn prepare_particle_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let archetype_guard = res_guard.archetypes.particle_render_archetype.read();
    if archetype_guard.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] prepare_particle_archetype_for_render_and_bind_pipeline | archetype absent",
      ));
    }
    let archetype = archetype_guard.as_ref().unwrap();
    let pipeline = archetype.pipeline_key.unwrap();

    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument("[Vulkan RenderDevice] prepare_particle_archetype_for_render_and_bind_pipeline | invalid command buffer handle"))?;
      data.command_buffer.get()
    };

    let p = res_guard
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline)
      .unwrap();

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, p.get());
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        archetype.pipeline_layout.get(),
        0,
        &[archetype.descriptor_set.get()],
        &[],
      );
    }
    Ok(())
  }

  fn prepare_sky_for_render(&self, cmd_buffer: gpu::CommandBufferHandle) -> GpuResult<()> {
    let res_guard = self.res.read();
    let sky_image_guard = res_guard.sky_image.read();
    let timeline = res_guard.get_timeline_semaphore_cached_value();
    if sky_image_guard.is_none() {
      oshal::log!("[RenderThread] render_sky ERROR: sky_image_guard.is_none");
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] prepare_sky_for_render: sky image absent",
      ));
    }

    let do_alloc = {
      let sky_render_archetype_guard = res_guard.archetypes.sky_render_archetype.read();
      if sky_render_archetype_guard.is_none() {
        oshal::log!("[RenderThread] render_sky ERROR: sky_render_archetype_guard.is_none");
        return Err(GpuError::InvalidState(
          "[Vulkan RenderDevice] prepare_sky_for_render: absent archetype",
        ));
      }

      let needs_desc = sky_render_archetype_guard
        .as_ref()
        .unwrap()
        .descriptor_set
        .is_none_or(|d| d.is_null());

      let mut do_alloc = false;
      if needs_desc {
        {
          let arch_ref = sky_render_archetype_guard.as_ref();
          if let Some(arch) = arch_ref {
            if arch.descriptor_set.is_none() {
              do_alloc = true;
            }
          }
        }
      }

      do_alloc
    };

    if do_alloc {
      let mut sky_render_archetype_guard = res_guard.archetypes.sky_render_archetype.write();
      let layout = sky_render_archetype_guard
        .as_ref()
        .unwrap()
        .descriptor_set_layout
        .get();

      let pool_guard = res_guard.descriptor_pool.read();
      let new_set = pool_guard
        .as_ref()
        .unwrap()
        .allocate(
          &self.device,
          layout,
          &res_guard.discard_pool,
          timeline,
          "Sky",
        )?
        .get();

      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(
          sky_image_guard
            .as_ref()
            .map(|img| img.image_view.get())
            .unwrap_or_else(|| {
              // Actually, let's find the dummy texture from physical mesh archetype if possible,
              // but sky render has its own set layout.
              // Let's just use the image view from sky_image if it exists,
              // or if it doesn't, we probably shouldn't be here yet or should have a generic dummy.
              sky_image_guard.as_ref().unwrap().image_view.get()
            }),
        )
        .sampler(res_guard.linear_sampler.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(new_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      sky_render_archetype_guard.as_mut().unwrap().descriptor_set =
        Some(unsafe { NonZeroHandle::new_unchecked(new_set) });
    }

    let (layout, descriptor_set) = {
      let sky_archetype_lock = res_guard.archetypes.sky_render_archetype.read();
      let sky_archetype = sky_archetype_lock
        .as_ref()
        .ok_or(GpuError::InvalidState("device"))?;
      let descriptor_set = sky_archetype
        .descriptor_set
        .ok_or(GpuError::InvalidState("no descriptor set"))?
        .get();
      let layout = sky_archetype.pipeline_layout.get();
      (layout, descriptor_set)
    };

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&cmd_buffer).unwrap();
    let cmd = data.command_buffer.get();
    unsafe {
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[descriptor_set],
        &[],
      );
    }

    Ok(())
  }

  fn prepare_text_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline, layout, set) = {
      let res = self.res.read();
      let archetype_lock = res.archetypes.text_render_archetype.read();
      let archetype = archetype_lock.as_ref().ok_or(GpuError::InvalidState("[Vulkan RenderDevice] prepare_text_archetype_for_render_and_bind_pipeline: text archetype absent"))?;

      let cmd = {
        let cmd_buffers = self.recording_command_buffers.read();
        let data = cmd_buffers.get(&cmd_buffer).ok_or(GpuError::InvalidArgument("[Vulkan RenderDevice] prepare_text_archetype_for_render_and_bind_pipeline: invalid command buffer handle"))?;
        data.command_buffer.get()
      };

      let pipeline = archetype.pipeline_key.and_then(|k| res.pipeline_pool.read().get_graphics_pipeline(k)).map(|p| p.get()).ok_or(GpuError::InvalidState("[Vulkan RenderDevice] prepare_text_archetype_for_render_and_bind_pipeline: couldn't fetch text pipeline"))?;
      let layout = archetype.pipeline_layout.get();
      let set = archetype.descriptor_set.ok_or(GpuError::InvalidState("[Vulkan RenderDevice] prepare_text_archetype_for_render_and_bind_pipeline: text descriptor set absent"))?;

      (cmd, pipeline, layout, set)
    };
    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[set],
        &[],
      );
    }
    Ok(())
  }

  fn render_minimap(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    player_pos: Vec3f32,
    max_distance: f32,
    planets: &[(Vec3f32, f32, [f32; 4])],
    screen_extent: [f32; 2],
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let minimap_render_archetype_guard = res_guard.archetypes.minimap_render_archetype.read();
    if minimap_render_archetype_guard.is_none() {
      return Err(GpuError::InvalidState(
        "[Vulkan RenderDevice] render_minimap: archetype absent",
      ));
    }

    let archetype = minimap_render_archetype_guard.as_ref().unwrap();
    let pipeline = res_guard
      .pipeline_pool
      .read()
      .get_graphics_pipeline(archetype.pipeline_key.unwrap())
      .unwrap();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] render_minimap: invalid command buffer handle",
      ))?;
    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    // Fetch aspect ratio from the active presentation engine (we assume there's at least one, or we default to 1.0)
    let aspect_ratio = {
      let b = screen_extent[1];
      let res = if b > -1e-6_f32 && b < 1e-6_f32 {
        0.0
      } else {
        screen_extent[0] / b
      };
      if res == 0.0 { 1.0 } else { res }
    };

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());

      let mut push_bytes = [0u8; 544];

      let offset = [0.7f32, 0.7f32];
      let size = [0.25f32, 0.25f32 * aspect_ratio];
      let player_pos_arr = [player_pos.x(), player_pos.y()];
      let max_distance_f = max_distance;
      let num_planets = planets.len() as u32;

      core::ptr::copy_nonoverlapping(
        &offset as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(0),
        8,
      );
      core::ptr::copy_nonoverlapping(
        &size as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(8),
        8,
      );
      core::ptr::copy_nonoverlapping(
        &player_pos_arr as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(16),
        8,
      );
      core::ptr::copy_nonoverlapping(
        &max_distance_f as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(24),
        4,
      );
      core::ptr::copy_nonoverlapping(
        &num_planets as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(28),
        4,
      );

      for (i, p) in planets.iter().enumerate().take(16) {
        let base = 32 + i * 32;
        let p_pos = [p.0.x(), p.0.y()];
        let p_size = p.1;
        let p_pad = 0.0f32;
        let p_color = p.2;

        core::ptr::copy_nonoverlapping(
          &p_pos as *const _ as *const u8,
          push_bytes.as_mut_ptr().add(base + 0),
          8,
        );
        core::ptr::copy_nonoverlapping(
          &p_size as *const _ as *const u8,
          push_bytes.as_mut_ptr().add(base + 8),
          4,
        );
        core::ptr::copy_nonoverlapping(
          &p_pad as *const _ as *const u8,
          push_bytes.as_mut_ptr().add(base + 12),
          4,
        );
        core::ptr::copy_nonoverlapping(
          &p_color as *const _ as *const u8,
          push_bytes.as_mut_ptr().add(base + 16),
          16,
        );
      }

      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        &push_bytes,
      );

      self.device.cmd_draw(cmd, 4, 1, 0, 0);
    }
    Ok(())
  }

  fn render_ui_rect(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
  ) -> GpuResult<()> {
    let (pipeline, layout, set) = {
      let res_guard = self.res.read();
      let text_render_archetype_guard = res_guard.archetypes.text_render_archetype.read();
      let archetype = text_render_archetype_guard
        .as_ref()
        .ok_or(GpuError::InvalidState(
          "[Vulkan RenderDevice] render_ui_rect: archetype absent",
        ))?;
      let pipeline_key = archetype.pipeline_key.ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] render_minimap: pipeline key absent",
      ))?;
      let pipeline = res_guard
        .pipeline_pool
        .read()
        .get_graphics_pipeline(pipeline_key)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] render_minimap: pipeline key invalid (pipeline not found)",
        ))?
        .get();
      let layout = archetype.pipeline_layout.get();
      let set = archetype.descriptor_set.ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] render_minimap: descriptor set not found",
      ))?;
      (pipeline, layout, set)
    };

    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] render_minimap: cmd_buffer invalid",
        ))?;
      data.command_buffer.get()
    };

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);

      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[set],
        &[],
      );

      let mut push_bytes = [0u8; 52];
      let pos_arr = [position[0], position[1]];
      let scale_arr = [size[0], size[1]];
      let uv_bounds = [0.0f32, 0.0f32, -1.0f32, -1.0f32];
      let texture_id = 0u32;

      core::ptr::copy_nonoverlapping(
        &pos_arr as *const _ as *const u8,
        push_bytes.as_mut_ptr(),
        8,
      );
      core::ptr::copy_nonoverlapping(
        &scale_arr as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(8),
        8,
      );
      core::ptr::copy_nonoverlapping(
        &color as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(16),
        16,
      );
      core::ptr::copy_nonoverlapping(
        &uv_bounds as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(32),
        16,
      );
      core::ptr::copy_nonoverlapping(
        &texture_id as *const _ as *const u8,
        push_bytes.as_mut_ptr().add(48),
        4,
      );

      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        &push_bytes,
      );
      self.device.cmd_draw(cmd, 4, 1, 0, 0);
    }

    Ok(())
  }

  fn render_text(
    &self,
    cmd_buffer: CommandBufferHandle,
    text: &str,
    start_cursor_position: [f32; 2],
    screen_extent: [f32; 2],
    atlas_id: (u64, u32),
    desired_points: f32,
    color: [f32; 4],
  ) -> GpuResult<()> {
    let res_guard = self.res.read();
    let archetype_lock = res_guard.archetypes.text_render_archetype.read();
    let archetype = archetype_lock.as_ref().ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] render_text: archetype missing",
    ))?;
    let font = archetype
      .uploaded_fonts
      .get(&atlas_id.0)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] render_text: inexistent font",
      ))?;
    if atlas_id.1 != font.descriptor_index {
      return Err(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] render_text: inconsistent descriptor index",
      ));
    }

    let cmd = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] render_text: cmd_buffer invalid",
        ))?;
      let cmd = data.command_buffer.get();
      cmd
    };

    let mut cursor_x = start_cursor_position[0];
    let mut cursor_y = start_cursor_position[1];

    let default_glyph = font.atlas.glyphs.get(&'?').ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] render_text: text atlas doesn't have default \"?\" glyph",
    ))?;

    unsafe {
      for c in text.chars() {
        if c == '\n' {
          cursor_x = start_cursor_position[0];
          cursor_y += (font.atlas.scaled_height(desired_points) * 1.5) * (2.0 / screen_extent[1]);
          continue;
        }

        let glyph = font.atlas.glyphs.get(&c).unwrap_or(default_glyph);
        let push_constants = gpu::TextPushConstants::from_glyph(
          glyph,
          [cursor_x, cursor_y],
          screen_extent,
          desired_points,
          font.atlas.scale,
          atlas_id.1,
          color,
        );
        self.push_text_constants(cmd_buffer, &push_constants)?;
        self.device.cmd_draw(cmd, 4, 1, 0, 0);

        cursor_x +=
          glyph.scaled_advance(desired_points, font.atlas.scale) * (2.0 / screen_extent[0]);
      }
    }

    Ok(())
  }

  fn end_render_pass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] end_render_pass: command buffer handle invalid",
      ))?;

    let cmd = data.command_buffer.get();
    let subpass_end_info = vk::SubpassEndInfo::default();

    unsafe {
      self
        .device
        .create_renderpass2
        .cmd_end_render_pass2(cmd, &subpass_end_info);
    }

    Ok(())
  }

  fn record_windowless_download(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
    task_id: u64,
  ) -> GpuResult<()> {
    let (cmd, acquire_result) = {
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers
        .get(&cmd_buffer)
        .ok_or(GpuError::InvalidArgument(
          "[Vulkan RenderDevice] record_windowless_download: command buffer handle invalid",
        ))?;
      let presentation = data.presentation.ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] record_windowless_download: no active render pass",
      ))?;
      (data.command_buffer.get(), presentation.acquire_result)
    };

    let res_guard = self.res.read();

    let engine_lock = res_guard.live_presentation_engines.read();
    let state_lock = engine_lock.get(&handle).ok_or(GpuError::InvalidArgument(
      "[Vulkan RenderDevice] record_windowless_download: invalid presentation engine",
    ))?;
    let state = state_lock.read();

    let (image, width, height) =
      if let swapchain::PresentationState::Windowless(windowless) = &*state {
        let (img, _, _) =
          unsafe { windowless.get_image_resources(acquire_result.image_index as usize) };
        (img, windowless.extent().0, windowless.extent().1)
      } else {
        return Err(GpuError::InvalidState(
          "[Vulkan RenderDevice] record_windowless_download: cEngine is not windowless",
        ));
      };

    let buffer_size = (width * height * 4) as vk::DeviceSize;

    let buffer_info = vk::BufferCreateInfo::default()
      .size(buffer_size)
      .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED;

    let (staging_buffer, alloc) = unsafe {
      res_guard
        .allocator
        .allocator
        .create_buffer(&buffer_info, &alloc_info)
    }?;

    unsafe {
      let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );

      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&image_barrier));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &dep_info);

      let region = vk::BufferImageCopy::default()
        .image_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .image_extent(vk::Extent3D {
          width,
          height,
          depth: 1,
        });

      self.device.cmd_copy_image_to_buffer(
        cmd,
        image.get(),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        staging_buffer,
        &[region],
      );

      let buffer_barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)
        .buffer(staging_buffer)
        .size(buffer_size)
        .offset(0);

      let buf_dep_info = vk::DependencyInfo::default()
        .buffer_memory_barriers(core::slice::from_ref(&buffer_barrier));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &buf_dep_info);
    }

    res_guard.pending_downloads.write().insert(
      task_id,
      PendingDownload {
        staging_buffer,
        allocation: alloc,
        size: buffer_size as usize,
      },
    );

    Ok(())
  }

  fn read_windowless_download(&self, task_id: u64, buffer: &mut [u8]) -> GpuResult<()> {
    if !self.is_task_completed(task_id)? {
      return Err(GpuError::InvalidState("Task not completed"));
    }

    let res_guard = self.res.read();
    let mut pending_lock = res_guard.pending_downloads.write();

    let mut download = pending_lock
      .remove(&task_id)
      .ok_or(GpuError::InvalidArgument(
        "Invalid or previously consumed download ID",
      ))?;

    if buffer.len() != download.size {
      unsafe {
        res_guard
          .allocator
          .allocator
          .destroy_buffer(download.staging_buffer, &mut download.allocation);
      }
      return Err(GpuError::InvalidArgument("Output buffer size mismatch"));
    }

    let alloc_info = res_guard
      .allocator
      .allocator
      .get_allocation_info(&download.allocation);
    let mapped_ptr = alloc_info.mapped_data as *const u8;

    if !mapped_ptr.is_null() {
      res_guard.allocator.allocator.invalidate_allocation(
        &download.allocation,
        0,
        vk::WHOLE_SIZE,
      )?;
      unsafe {
        core::ptr::copy_nonoverlapping(mapped_ptr, buffer.as_mut_ptr(), download.size);
      }
    } else {
      unsafe {
        res_guard
          .allocator
          .allocator
          .destroy_buffer(download.staging_buffer, &mut download.allocation);
      }
      return Err(GpuError::InvalidState("Memory mapping failed"));
    }

    unsafe {
      res_guard
        .allocator
        .allocator
        .destroy_buffer(download.staging_buffer, &mut download.allocation);
    }

    Ok(())
  }

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
  ) -> GpuResult<()> {
    let mut cmd_buffers = self.recording_command_buffers.write();
    let mut data = cmd_buffers
      .remove(&cmd_buffer)
      .ok_or(GpuError::InvalidArgument(
        "[Vulkan RenderDevice] submit_command_buffer: invalid command buffer handle",
      ))?;

    unsafe {
      self.device.end_command_buffer(data.command_buffer.get())?;
    }

    let presentation = data.presentation.ok_or(GpuError::InvalidState(
      "[Vulkan RenderDevice] submit_command_buffer: inconsistent presentation engine state",
    ))?;
    let res_guard = self.res.read();
    let presentation_engines_guard = res_guard.live_presentation_engines.read();
    let presentation_engine = presentation_engines_guard
      .get(&presentation.presentation_engine)
      .ok_or(GpuError::InvalidArgument("[Vulkan RenderDevice] submit_command_buffer: presentation engine state inside submit invalid"))?;

    let rpresentation_engine = presentation_engine.read();
    let (wait_semaphore, submission_fence) = unsafe {
      rpresentation_engine.get_frame_resources(presentation.acquire_result.frame_index as usize)
    };
    let (_, _, signal_semaphore) = unsafe {
      rpresentation_engine.get_image_resources(presentation.acquire_result.image_index as usize)
    };
    let res_guard = self.res.read();
    let graphics_queue = self.queues.get_graphics_queue();

    // CRITICAL FIX: Lock submission to ensure ordering
    let next_timeline_value = {
      let _queue_lock = self.device.submission_lock.lock();
      // ALLOCATE STRICT TIMELINE VALUE RIGHT BEFORE SUBMISSION
      let next_timeline_value = res_guard.timeline_manager.allocate_submit_value();

      if let swapchain::PresentationState::Windowless(windowless) = &*rpresentation_engine {
        windowless
          .last_timeline_value
          .store(next_timeline_value, Ordering::Release);
      }

      let mut signal_semaphores = Vec::new();
      let mut timeline_values = Vec::new();

      if let Some(sem) = signal_semaphore {
        signal_semaphores.push(sem.get());
        timeline_values.push(0);
      }

      signal_semaphores.push(res_guard.timeline_manager.semaphore.get());
      timeline_values.push(next_timeline_value);

      let mut wait_semaphores = Vec::new();
      let mut wait_semaphore_values = Vec::new();
      let mut wait_dst_stage_mask = Vec::new();

      if let Some(wait_semaphore) = wait_semaphore {
        wait_semaphores.push(wait_semaphore.get());
        wait_semaphore_values.push(0);
        wait_dst_stage_mask.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
      }

      let command_buffers = [data.command_buffer.get()];
      let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_semaphore_values)
        .signal_semaphore_values(&timeline_values);

      let submit_info = vk::SubmitInfo::default()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&wait_dst_stage_mask)
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline_info);

      // CRITICAL: Tell the task what timeline value to wait for! ---
      if let Some(tid) = task_id {
        let registry = res_guard.timeline_manager.task_registry.read();
        if let Some(entry) = registry.get(&tid) {
          entry
            .target_value
            .store(next_timeline_value, Ordering::Release);
        }
      }

      unsafe {
        self.device.queue_submit(
          graphics_queue.handle,
          &[submit_info],
          submission_fence.get(),
        )
      }
      .map_err(GpuError::from)?;
      // safely drop lock here
      next_timeline_value
    };

    let cmd_pools = res_guard
      .command_pools
      .read()
      .get(graphics_queue.index as usize)
      .and_then(|opt| opt.as_ref())
      .cloned()
      .ok_or(GpuError::InvalidState(
        "[Vulkan RenderDevice] submit_command_buffer: couldn't get command pools",
      ))?;

    data.discard(
      cmd_buffer.into(),
      &res_guard.discard_pool,
      cmd_pools,
      next_timeline_value,
    );

    Ok(())
  }

  fn wire_callbacks(&self, pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) -> GpuResult<()> {
    let workload = self
      .res
      .read()
      .timeline_manager
      .create_polling_workload(Arc::clone(&self.callback_stop_signal));
    pool
      .scatter(vec![Box::new(workload)])
      .map_err(|_| GpuError::InvalidState("[Vulkan RenderDevice] wire_callbacks"))?;
    Ok(())
  }

  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool> {
    self.res.read().timeline_manager.is_task_completed(task_id)
  }

  fn create_task(&self) -> u64 {
    self.res.read().timeline_manager.create_task()
  }

  fn fail_task(&self, task_id: u64, error: GpuError) {
    self.res.read().timeline_manager.fail_task(task_id, error)
  }

  fn success_task(&self, task_id: u64) {
    self.res.read().timeline_manager.success_task(task_id)
  }
}

fn extract_position_data(comet: &Comet) -> Vec<f32> {
  let mut position_data = Vec::with_capacity(comet.vertices.len() * 3);
  for vertex in &comet.vertices {
    position_data.extend_from_slice(&vertex.position);
  }
  position_data
}

fn extract_attribute_data(comet: &Comet) -> Vec<f32> {
  let mut attribute_data = Vec::with_capacity(comet.vertices.len() * 9);
  for vertex in &comet.vertices {
    attribute_data.extend_from_slice(&vertex.normal);
    attribute_data.extend_from_slice(&vertex.uv);
    attribute_data.extend_from_slice(&vertex.tangent);
  }
  attribute_data
}

fn ensure_text_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("text.vert.spv");
  frag_path = assets_dir.join("text.frag.spv");

  let vkey = shader_manager.get_or_load(
    device,
    vert_path.as_ref(),
    "main",
    spirv::ExecutionModel::Vertex,
  )?;
  let fkey = shader_manager.get_or_load(
    device,
    frag_path.as_ref(),
    "main",
    spirv::ExecutionModel::Fragment,
  )?;

  Ok((vkey, fkey))
}

fn ensure_physical_mesh_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey, ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;
  let outline_vert_path: PathBuf;
  let outline_frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("physical_mesh.vert.spv");
  #[cfg(not(feature = "physical_mesh_debug_normals"))]
  {
    frag_path = assets_dir.join("physical_mesh.frag.spv");
  }
  #[cfg(feature = "physical_mesh_debug_normals")]
  {
    frag_path = assets_dir.join("physical_mesh_debug_normals.frag.spv");
  }

  outline_vert_path = assets_dir.join("physical_mesh_outline.vert.spv");
  outline_frag_path = assets_dir.join("physical_mesh_outline.frag.spv");

  let vert_key =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  let outline_vert_key = shader_manager.get_or_load(
    &device,
    &outline_vert_path,
    "main",
    spirv::ExecutionModel::Vertex,
  )?;
  let outline_frag_key = shader_manager.get_or_load(
    &device,
    &outline_frag_path,
    "main",
    spirv::ExecutionModel::Fragment,
  )?;

  Ok((vert_key, frag_key, outline_vert_key, outline_frag_key))
}

fn ensure_skygen_shader_module(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<ShaderKey> {
  let comp_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;
  comp_path = assets_dir.join("skygen.comp.spv");

  let comp_key = shader_manager.get_or_load(
    &device,
    &comp_path,
    "main",
    spirv::ExecutionModel::GLCompute,
  )?;

  Ok(comp_key)
}

fn ensure_sungen_shader_module(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<ShaderKey> {
  let comp_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;
  comp_path = assets_dir.join("sungen.comp.spv");

  let comp_key = shader_manager.get_or_load(
    &device,
    &comp_path,
    "main",
    spirv::ExecutionModel::GLCompute,
  )?;

  Ok(comp_key)
}

fn ensure_grid_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("grid.vert.spv");
  frag_path = assets_dir.join("grid.frag.spv");

  let vkey =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let fkey =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vkey, fkey))
}

fn ensure_sky_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("sky.vert.spv");
  frag_path = assets_dir.join("sky.frag.spv");

  let vkey =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let fkey =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vkey, fkey))
}

fn ensure_sun_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("sun_volume.vert.spv");
  frag_path = assets_dir.join("sun_volume.frag.spv");

  let vert_key =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vert_key, frag_key))
}

fn ensure_particle_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("particle.vert.spv");
  frag_path = assets_dir.join("particle.frag.spv");

  let vert_key =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vert_key, frag_key))
}

fn ensure_marker_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("marker.vert.spv");
  frag_path = assets_dir.join("marker.frag.spv");

  let vkey =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let fkey =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;
  Ok((vkey, fkey))
}

fn ensure_measurement_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("measurement.vert.spv");
  frag_path = assets_dir.join("measurement.frag.spv");

  let vkey =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let fkey =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;
  Ok((vkey, fkey))
}

fn ensure_billboard_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("billboard.vert.spv");
  frag_path = assets_dir.join("billboard.frag.spv");

  let vert_key =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vert_key, frag_key))
}

fn ensure_cursor_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("cursor.vert.spv");
  #[cfg(feature = "cursor_debug")]
  {
    frag_path = assets_dir.join("cursor_debug.frag.spv");
  }
  #[cfg(not(feature = "cursor_debug"))]
  {
    frag_path = assets_dir.join("cursor.frag.spv");
  }

  let vert_key =
    shader_manager.get_or_load(&device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(&device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;

  Ok((vert_key, frag_key))
}

fn ensure_gizmo_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("gizmo.vert.spv");
  frag_path = assets_dir.join("gizmo.frag.spv");

  let vkey = shader_manager.get_or_load(
    device,
    vert_path.as_ref(),
    "main",
    spirv::ExecutionModel::Vertex,
  )?;
  let fkey = shader_manager.get_or_load(
    device,
    frag_path.as_ref(),
    "main",
    spirv::ExecutionModel::Fragment,
  )?;

  Ok((vkey, fkey))
}

fn ensure_bvh_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("bvh_debug.vert.spv");
  frag_path = assets_dir.join("bvh_debug.frag.spv");

  let vkey = shader_manager.get_or_load(
    device,
    vert_path.as_ref(),
    "main",
    spirv::ExecutionModel::Vertex,
  )?;
  let fkey = shader_manager.get_or_load(
    device,
    frag_path.as_ref(),
    "main",
    spirv::ExecutionModel::Fragment,
  )?;

  Ok((vkey, fkey))
}

fn ensure_minimap_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("minimap.vert.spv");
  frag_path = assets_dir.join("minimap.frag.spv");

  let vert_key =
    shader_manager.get_or_load(device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key =
    shader_manager.get_or_load(device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;
  Ok((vert_key, frag_key))
}

fn shaders_asset_dir() -> GpuResult<PathBuf> {
  if let Some(path_str) = &*crate::gpu::ASSET_DIR.read() {
    Ok(PathBuf::from(path_str))
  } else {
    use aethervk_oshal_rlib::os::fs;
    let exe_path = fs::current_exe().map_err(|_| {
      GpuError::BackendSpecific("Failed to get executable path for asset loading".into())
    })?;
    exe_path
      .parent()
      .ok_or(GpuError::BackendSpecific(
        "Exe has no parent directory".into(),
      ))
      .and_then(|p| {
        let pt = p.join("assets");
        if pt.is_dir() {
          Ok(pt)
        } else {
          Err(GpuError::BackendSpecific(alloc::format!(
            "'{:?}' is not a valid directory",
            pt
          )))
        }
      })
  }
}

fn pretty_print_vulkan_device(
  props: &PhysicalDeviceProperties,
  device_name: &str,
  device_type: &str,
  family_count: u32,
  api_major: u32,
  api_minor: u32,
  api_patch: u32,
) -> String {
  alloc::format!(
    "Vulkan Device Info
       ------------------
       Name: {}
       Vendor ID: {:#X} ({})
       Device ID: {:#X}
       Type: {}
       API Version: {}.{}.{}
       Driver Version: {}
       Queue Families: {}
      ",
    device_name,
    props.vendor_id,
    match props.vendor_id {
      0x10DE => "NVIDIA",
      0x1002 | 0x1022 => "AMD",
      0x106B => "Apple",
      0x8086 => "Intel",
      0x13B5 => "ARM",
      0x5143 => "Qualcomm",
      0x1010 => "ImgTec",
      _ => "Unknown",
    },
    props.device_id,
    device_type,
    api_major,
    api_minor,
    api_patch,
    props.driver_version,
    family_count,
  )
}

#[cfg(test)]
mod test_render;
