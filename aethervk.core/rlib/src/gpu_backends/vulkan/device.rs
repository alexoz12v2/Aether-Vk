//! device module.
//! Only `RenderDevice` methods are allowed to start a Vulkan Transaction

use crate::{
  gpu::{
    self, AcquireResult, ArchetypeId, CommandBufferHandle, GpuResourceHandle, NativeGpuProperty,
    PipelineKey, PipelineKeyable, PresentationEngineHandle, RenderDevice, RenderableInstanceId,
    TextureFlags,
    frame::ResourceUploadResult,
    vulkan::{
      device::{locks::DebugTrackedRwLock, swapchain::PresentationState},
      physics::VulkanComputeKernels,
    },
  },
  gpu_backends::vulkan::{
    self,
    device::{
      commands::CommandBufferId,
      memory::GlobalDeviceAllocator,
      renderpasses::RenderPassSpecification,
      resources::{DerefArchetype, DiscardableResource, ForwardMeshRenderResource, Image},
      shader_manager::ShaderKey,
    },
    instance,
    utils::{self, NonZeroHandle, RwLockable},
  },
  scene::{EntityId, PhysicalMeshComponent, text::FontAtlas},
  simulation::comet::{Comet, Texture},
  types::{GpuError, GpuResult},
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::vector::{Vector3, vec3::Vec3f32},
  os::{fs::FileSystemObject, pool::WorkloadStatus},
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
use ash::vk;
use core::{
  any::Any,
  fmt,
  fmt::Formatter,
  hash::Hash,
  ptr::{self, NonNull},
  sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use function_name::named;
use heapless::index_map::FnvIndexMap;
use oshal::os::{
  fs::PathBuf,
  memory::{MaxAlignedStorage, StackAllocator},
  native::this_thread,
};
use vk_mem::{Alloc, AsAllocatorView};

struct AllPreps {
  physical_mesh:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  physical_mesh2:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  cursor: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  particle: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  particle2:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  sun: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  sky: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  grid: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  minimap: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  measurement:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  marker: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  text: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  text2: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  bvh: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  bvhwire2: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  gizmo: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  trajectory:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  ui: Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  background:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
  billboard:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::PreparedArchetypeUpdate>,
}

struct AllCompiled {
  physical_mesh:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  physical_mesh2:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  cursor: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  particle: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  particle2: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  sun: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  sky: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  grid: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  minimap: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  measurement:
    Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  marker: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  text: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  text2: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  bvh: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  bvhwire2: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  gizmo: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  trajectory: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  ui: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  background: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
  billboard: Option<crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData>,
}

/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err {
  // 1. No arguments
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - device error",
      "function",
      core::file!(),
      core::line!()
    ))
  };

  // 2. Variadic arguments
  ($fmt:expr, $($arg:tt)+) => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - {}",
      "function",
      core::file!(),
      core::line!(),
      core::format_args!($fmt, $($arg)+)
    ))
  };

  // 3. Single expression
  ($msg:expr) => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - {}",
      "function",
      core::file!(),
      core::line!(),
      $msg
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_invalid_arg {
  // 1. No arguments
  () => {
    $crate::types::GpuError::InvalidArgument(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - invalid argument",
      "function",
      core::file!(),
      core::line!()
    ))
  };

  // 2. Variadic arguments (e.g., gpu_invalid_arg!("expected {}, got {}", a, b))
  ($fmt:expr, $($arg:tt)+) => {
    $crate::types::GpuError::InvalidArgument(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - {}",
      "function",
      core::file!(),
      core::line!(),
      core::format_args!($fmt, $($arg)+)
    ))
  };

  // 3. Single expression (e.g., gpu_invalid_arg!("missing buffer"))
  ($msg:expr) => {
    $crate::types::GpuError::InvalidArgument(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - {}",
      "function",
      core::file!(),
      core::line!(),
      $msg
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_invalid_pe {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - invalid presentation engine handle",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}

#[macro_export]
macro_rules! extract_pe {
  ($state:expr, $h:expr) => {{
    let mut retry_count = 0;
    loop {
      if let Some(kv) = $state.live_presentation_engines.remove(&$h) {
        break Ok(kv.1);
      }
      if retry_count > 1_000_000 {
        break Err(gpu_err_invalid_pe!());
      }

      retry_count += 1;
      if retry_count < 1000 {
        core::hint::spin_loop();
      } else if retry_count < 2000 {
        aethervk_oshal_rlib::os::native::this_thread::yield_now();
      } else {
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          1,
        ));
      }
    }
  }};
}

#[macro_export]
#[macro_export]
macro_rules! wait_for_pe_direct {
  ($map:expr, $h:expr) => {{
    let mut retry_count = 0;
    loop {
      if let Some(entry) = $map.get(&$h) {
        break Ok(entry);
      }
      if retry_count > 7000 {
        break Err(gpu_err_invalid_pe!());
      }
      retry_count += 1;
      if retry_count < 1000 {
        core::hint::spin_loop();
      } else if retry_count < 2000 {
        aethervk_oshal_rlib::os::native::this_thread::yield_now();
      } else {
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          1,
        ));
      }
    }
  }};
}

#[macro_export]
macro_rules! wait_for_pe_mut_direct {
  ($map:expr, $h:expr) => {{
    let mut retry_count = 0;
    loop {
      if let Some(entry) = $map.get_mut(&$h) {
        break Ok(entry);
      }
      if retry_count > 7000 {
        break Err(gpu_err_invalid_pe!());
      }
      retry_count += 1;
      if retry_count < 1000 {
        core::hint::spin_loop();
      } else if retry_count < 2000 {
        aethervk_oshal_rlib::os::native::this_thread::yield_now();
      } else {
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          1,
        ));
      }
    }
  }};
}

macro_rules! wait_for_pe {
  ($state:expr, $h:expr) => {{
    let mut retry_count = 0;
    loop {
      if let Some(entry) = $state.live_presentation_engines.get(&$h) {
        break Ok(entry);
      }
      if retry_count > 7000 {
        break Err(gpu_err_invalid_pe!());
      }

      retry_count += 1;
      if retry_count < 1000 {
        core::hint::spin_loop();
      } else if retry_count < 2000 {
        aethervk_oshal_rlib::os::native::this_thread::yield_now();
      } else {
        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_millis(
          1,
        ));
      }
    }
  }};
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_cmd_no_pe {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - Cmd doesn't have presentation engine handle",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_invalid_cmd {
  () => {
    $crate::types::GpuError::InvalidArgument(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - invalid command buffer handle",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_device {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - device error",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_pipeline_key_absent {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - pipeline key absent",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_pipeline_absent {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - vulkan pipeline absent in pipeline pool",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}
/// can be used only on a #[named] function
#[macro_export]
macro_rules! gpu_err_archetype_absent {
  () => {
    $crate::types::GpuError::InvalidState(alloc::format!(
      "[Vulkan RenderDevice] {} {}:{} - archetype absent",
      "function",
      core::file!(),
      core::line!()
    ))
  };
}

pub(super) mod archetypes_struct;
pub(super) mod commands;
pub(super) mod descriptors;
#[cfg(any(debug_assertions, test))]
pub mod hooks;
pub(super) mod locks;
pub(super) mod memory;
pub(super) mod pipelines;
pub(super) mod renderpasses;
pub(super) mod resources;
pub(super) mod shader_manager;
pub(super) mod swapchain;
pub(super) mod timeline_manager;

// TODO No vulkan calls while holding locks
// TODO remove all unwrap and unwrap_unchecked (unless absolutely necessary or sure)

#[derive(Debug)]
/// TODO: Document this item
pub(super) struct TaskEntry {
  pub(super) target_value: AtomicU64,
  pub(super) status: AtomicU32, // 0: Pending, 1: Success, 2: Failed
  pub(super) error: DebugTrackedRwLock<Option<GpuError>>,
}

const TASK_STATUS_PENDING: u32 = 0;
const TASK_STATUS_SUCCESS: u32 = 1;
const TASK_STATUS_FAILED: u32 = 2;

struct TimelinePollingWorkload {
  timeline_sem_device: ash::khr::timeline_semaphore::Device,
  timeline_semaphore: vk::Semaphore,
  timeline_semaphore_cached_value: Arc<AtomicU64>,
  task_registry: Arc<DebugTrackedRwLock<BTreeMap<u64, Arc<TaskEntry>>>>,
  stop_signal: Arc<core::sync::atomic::AtomicBool>,
}

impl oshal::os::pool::Workload for TimelinePollingWorkload {
  #[named]
  fn execute(&mut self) -> WorkloadStatus {
    if self.stop_signal.load(Ordering::Acquire) {
      return WorkloadStatus::Complete;
    }

    // Poll semaphore
    if let Ok(gpu_value) =
      unsafe { self.timeline_sem_device.get_semaphore_counter_value(self.timeline_semaphore) }
    {
      self.timeline_semaphore_cached_value.fetch_max(gpu_value, Ordering::Relaxed);

      // Resolve tasks
      let completed_ids: Vec<u64> = {
        let registry = DebugTrackedRwLock::write(&self.task_registry);
        registry
          .iter()
          .filter(|(_, entry)| {
            entry.status.load(Ordering::Acquire) == TASK_STATUS_PENDING
              && gpu_value >= entry.target_value.load(Ordering::Acquire)
          })
          .map(|(id, _)| *id)
          .collect()
      };

      if !completed_ids.is_empty() {
        let mut registry = DebugTrackedRwLock::write(&self.task_registry);
        for id in completed_ids {
          if let Some(entry) = registry.get(&id) {
            entry.status.store(TASK_STATUS_SUCCESS, Ordering::Release);
          }
          registry.remove(&id);
        }
      }
    }

    // Sleep briefly so we don't peg the CPU at 100% while idle,
    // but returning Yield drops this back into the queue to allow other workloads a turn.
    // Note: this will block the current thread pool worker for 16ms.
    oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(16));
    WorkloadStatus::Yield
  }
}

trait DeviceResource {
  /// Cleanup function to facilitate hierarchical manual Drop of resources
  /// without having to propagate through `Arc` or other means a reference
  /// to device handle and its function pointers
  /// Note: This function is not responsible to setup the proper state for cleanup (eg synchronization)
  fn cleanup(&mut self, device: &ash::Device);
}

struct FunctionalDeviceResource<H: vk::Handle + Copy, F: FnOnce(H, &ash::Device)> {
  handle: H,
  cleanup: Option<F>,
}

impl<H: vk::Handle + Copy, F: FnOnce(H, &ash::Device)> FunctionalDeviceResource<H, F> {
  #[named]
  fn new(handle: H, cleanup: F) -> Self {
    Self {
      handle,
      cleanup: Some(cleanup),
    }
  }
}

impl<H: vk::Handle + Copy, F: FnOnce(H, &ash::Device)> DeviceResource
  for FunctionalDeviceResource<H, F>
{
  #[named]
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
  #[named]
  fn new(device: &'a ash::Device) -> Self {
    Self {
      device,
      allocator: StackAllocator::new(),
      resources: heapless::Vec::new(),
      heap_resources: Vec::new(),
      storage: MaxAlignedStorage([0; N]),
    }
  }

  /// TODO: Document this item
  #[named]
  pub fn clear(&mut self) {
    // The `drop` implementation will handle cleanup of existing resources.
    // Here we just need to reset the state.
    self.allocator = StackAllocator::new();
    self.resources.clear();
    self.heap_resources.clear();
  }

  /// TODO: Document this item
  #[named]
  pub fn push<T: DeviceResource + 'a>(&mut self, resource: T) -> Result<(), &'static str> {
    // Check if there's space in the inline allocator
    let layout = core::alloc::Layout::new::<T>();
    let start = self.allocator.offset.get();
    let align_offset = unsafe { self.storage.0.as_ptr().add(start).align_offset(layout.align()) };

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
  #[named]
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

        ptr::drop_in_place(ptr::from_mut(resource));
      }
    }
  }
}

/// TODO: Document this item
pub(super) struct PendingDownload {
  pub(super) staging_buffer: vk::Buffer,
  pub(super) allocation: vk_mem::Allocation,
  pub(super) size: usize,
  pub(super) presentation_engine: Option<PresentationEngineHandle>,
}

/// Device Resources. Each member here implements `DeviceResources` trait and is either
/// - implementing `Sync`
/// - Wrapped into a RwLock/Mutex
/// - Native Vulkan Handle, externally synchronized
pub struct DeviceResources {
  pub allocator: GlobalDeviceAllocator,
  /// Discardpool driven by render timeline (which is inside timeline manager)
  /// Not pub on purpose cause Kernels has its own discard pool driven by compute timeline
  discard_pool: resources::DiscardPool,
  live_presentation_engines: dashmap::DashMap<PresentationEngineHandle, PresentationState>,
  pub command_pools: DebugTrackedRwLock<
    heapless::Vec<Option<Arc<commands::CommandPools>>, { utils::MAX_QUEUE_FAMILY_COUNT }>,
  >,
  pub descriptor_pool: DebugTrackedRwLock<Option<Arc<descriptors::DescriptorPools>>>,
  pub pipeline_pool: pipelines::PipelinePool,
  renderpasses: renderpasses::RenderPasses,
  pub shader_manager: DebugTrackedRwLock<shader_manager::ShaderManager>,

  timeline_manager: timeline_manager::TimelineManager,
  next_cmd_id: Arc<AtomicU64>,

  linear_sampler: NonZeroHandle<vk::Sampler>,

  physical_mesh_resources: dashmap::DashMap<
    RenderableInstanceId,
    resources::ResourceState<resources::ForwardMeshRenderResource>,
  >,
  physical_mesh2_resources: dashmap::DashMap<
    RenderableInstanceId,
    resources::ResourceState<resources::ForwardMesh2RenderResource>,
  >,
  sun_resources: dashmap::DashMap<EntityId, resources::ResourceState<resources::SunRenderResource>>,

  sky_image: DebugTrackedRwLock<Option<Image>>,
  billboard_resources: DebugTrackedRwLock<Vec<Image>>, // TODO switch to dashmap::DashSet

  // TODO switch to dashmap::DashMap
  pending_downloads: DebugTrackedRwLock<hashbrown::HashMap<u64, PendingDownload>>,

  frame_staging_arena: DebugTrackedRwLock<Option<memory::FrameStagingArena>>,

  sun_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::SunRenderResourceArchetypeArena>>>,
  physical_mesh_render_archetype_arena: Option<
    alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMeshRenderResourceArchetypeArena>>,
  >,
  physical_mesh2_render_archetype_arena: Option<
    alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMesh2RenderResourceArchetypeArena>>,
  >,
  billboard_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::BillboardRenderResourceArchetypeArena>>>,
  particle_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::ParticleRenderResourceArchetypeArena>>>,
  particle2_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Particle2RenderResourceArchetypeArena>>>,
  cursor_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::CursorRenderResourceArchetypeArena>>>,
  marker_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::MarkerRenderResourceArchetypeArena>>>,
  measurement_render_archetype_arena: Option<
    alloc::sync::Arc<DebugTrackedRwLock<resources::MeasurementRenderResourceArchetypeArena>>,
  >,
  sky_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::SkyRenderResourceArchetypeArena>>>,
  grid_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::GridRenderResourceArchetypeArena>>>,
  minimap_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::MinimapRenderResourceArchetypeArena>>>,
  text_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::TextRenderResourceArchetypeArena>>>,
  text2_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Text2RenderResourceArchetypeArena>>>,
  bvh_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::BvhRenderResourceArchetypeArena>>>,
  bvhwire2_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Bvhwire2RenderResourceArchetypeArena>>>,
  sphere_gizmo_render_archetype_arena: Option<
    alloc::sync::Arc<DebugTrackedRwLock<resources::SphereGizmoRenderResourceArchetypeArena>>,
  >,
  gizmo_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::GizmoRenderResourceArchetypeArena>>>,
  trajectory_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::TrajectoryRenderResourceArchetypeArena>>>,
  ui_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::UiRenderResourceArchetypeArena>>>,
  background_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::BackgroundRenderResourceArchetypeArena>>>,

  /// Queue of window-system cleanup tasks that must execute on the main thread.
  /// On macOS, MoltenVK requires `CAMetalLayer` teardown on the main UI thread.
  /// Populated by `WindowedPresentationState::cleanup()` and drained by
  /// `process_main_thread_cleanup_queue()`.
  pub(crate) main_thread_cleanup_queue: crate::gpu::MainThreadCleanupQueue,
}

// TODO: each member should derive it so that this can derive it too
impl fmt::Debug for DeviceResources {
  #[named]
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    f.write_str("DeviceResources")
  }
}

impl DeviceResources {
  /// cleanup in reverse order of declaration in the struct
  #[named]
  fn cleanup(&mut self, device: &LogicalDevice) {
    for (_, mut download) in DebugTrackedRwLock::write(&self.pending_downloads).drain() {
      unsafe {
        self
          .allocator
          .allocator
          .destroy_buffer(download.staging_buffer, &mut download.allocation);
      }
    }

    // BUG FIX (2025-05): The original code called pe_state.value_mut().cleanup(device) HERE
    // in addition to the removal loop at the bottom of this function.  That caused every
    // swapchain semaphore, fence, and image to be destroyed twice, producing
    // VK_ERROR_UNKNOWN / VUID-vkDestroySemaphore validation errors.
    // Fix: only call archetypes_mut().discard() here (to enqueue pipeline-layout / buffer
    // handles into the discard_pool before the arenas are unwrapped below).  The actual
    // PE Vulkan objects are destroyed exactly once in the removal loop further below.
    for mut pe_state in self.live_presentation_engines.iter_mut() {
      pe_state.value_mut().archetypes_mut().discard(device, &self.discard_pool);
    }

    macro_rules! discard_arena {
      ($field:ident) => {
        if let Some(arena_arc) = self.$field.take() {
          match alloc::sync::Arc::try_unwrap(arena_arc) {
            Ok(arena_lock) => {
              let mut arena =
                crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::into_inner(
                  arena_lock,
                );
              aethervk_oshal_rlib::log!("Successfully unwrapped arena {}", stringify!($field));
              arena.discard(device, &self.discard_pool, u64::MAX);
            }
            Err(_) => {
              panic!("if device.wait_idle was called, then nobody should hold a strong arena");
            }
          }
        }
      };
    }

    discard_arena!(sun_render_archetype_arena);
    discard_arena!(physical_mesh_render_archetype_arena);
    discard_arena!(physical_mesh2_render_archetype_arena);
    discard_arena!(billboard_render_archetype_arena);
    discard_arena!(particle_render_archetype_arena);
    discard_arena!(particle2_render_archetype_arena);
    discard_arena!(cursor_render_archetype_arena);
    discard_arena!(marker_render_archetype_arena);
    discard_arena!(measurement_render_archetype_arena);
    discard_arena!(sky_render_archetype_arena);
    discard_arena!(grid_render_archetype_arena);
    discard_arena!(minimap_render_archetype_arena);
    discard_arena!(text_render_archetype_arena);
    discard_arena!(text2_render_archetype_arena);
    discard_arena!(bvh_render_archetype_arena);
    discard_arena!(bvhwire2_render_archetype_arena);
    discard_arena!(sphere_gizmo_render_archetype_arena);
    discard_arena!(gizmo_render_archetype_arena);
    discard_arena!(trajectory_render_archetype_arena);
    discard_arena!(ui_render_archetype_arena);
    discard_arena!(background_render_archetype_arena);

    // all discardable resources should have been already discarded
    if self.has_discardables() {
      self.clear_discardables(device);
    }
    self.discard_pool.cleanup(device);

    self.timeline_manager.cleanup(device);

    self.renderpasses.cleanup(device);

    DebugTrackedRwLock::write(&self.shader_manager).destroy(device);

    // Safety: If this is a properly constructed `DeviceResources`, then `descriptor_pool = Some(_)`
    let dp_opt = DebugTrackedRwLock::write(&self.descriptor_pool).take();
    if let Some(pool) = dp_opt {
      assert_eq!(Arc::strong_count(&pool), 1);
      let mut descriptor_pool: descriptors::DescriptorPools = Arc::try_unwrap(pool).unwrap();
      descriptor_pool.cleanup(device);
    }

    self.pipeline_pool.cleanup(device);

    let mut cp_lock = DebugTrackedRwLock::write(&self.command_pools);
    for command_pool in cp_lock.iter_mut() {
      if let Some(pool) = command_pool.take() {
        assert_eq!(Arc::strong_count(&pool), 1);
        let mut command_pool = Arc::try_unwrap(pool).unwrap();
        command_pool.cleanup(device);
      }
    }

    let keys: alloc::vec::Vec<_> =
      self.live_presentation_engines.iter().map(|kv| *kv.key()).collect();
    for k in keys {
      if let Some((_, mut presentation_state)) = self.live_presentation_engines.remove(&k) {
        presentation_state.cleanup(device);
      }
    }

    // - Linear Sampler
    unsafe { device.destroy_sampler(self.linear_sampler.get(), None) };

    if let Some(sky_image) = DebugTrackedRwLock::write(&self.sky_image).take() {
      unsafe {
        vk_mem::ffi::vmaDestroyImage(
          self.allocator.allocator.get_raw(),
          sky_image.image.get(),
          sky_image.allocation.get_raw(),
        );
        device.destroy_image_view(sky_image.image_view.get(), None);
      }
    }

    let taken_frame = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
      &self.frame_staging_arena,
    )
    .take();
    aethervk_oshal_rlib::log!("taken_frame is some: {}", taken_frame.is_some());
    if let Some(mut arena) = taken_frame {
      arena.destroy(self.allocator.allocator.as_allocator_view());
    }
    self.allocator.cleanup(device);
  }
}

fn get_shader(
  shader_manager: &shader_manager::ShaderManager,
  key: ShaderKey,
  expected_stage: ash::vk::ShaderStageFlags,
) -> GpuResult<alloc::sync::Arc<shader_manager::Shader>> {
  let shader = shader_manager.get(key).ok_or(GpuError::InvalidShader)?;
  if shader.shader_stage != expected_stage {
    return Err(GpuError::InvalidShader);
  }
  Ok(shader)
}

impl DeviceResources {
  #[named]
  fn has_discardables(&self) -> bool {
    let mut archetypes_have_discardables = false;
    for pe in &self.live_presentation_engines {
      if pe.archetypes().has_discardables() {
        archetypes_have_discardables = true;
        break;
      }
    }
    archetypes_have_discardables
      || !self.physical_mesh_resources.is_empty()
      || !self.physical_mesh2_resources.is_empty()
      || !self.sun_resources.is_empty()
      || !DebugTrackedRwLock::read(&self.billboard_resources).is_empty()
  }

  #[named]
  fn clear_discardables(&mut self, device: &LogicalDevice) {
    aethervk_oshal_rlib::log!("clear_discardables started!");
    debug_assert!(self.has_discardables());

    for mut pe_state in self.live_presentation_engines.iter_mut() {
      pe_state.value_mut().archetypes_mut().discard(device, &self.discard_pool);
    }

    let pm_keys: alloc::vec::Vec<_> =
      self.physical_mesh_resources.iter().map(|kv| *kv.key()).collect();
    for key in pm_keys {
      if let Some((_, state)) = self.physical_mesh_resources.remove(&key) {
        if let resources::ResourceState::Ready(mut resource) = state {
          resource.discard(device, &self.discard_pool, u64::MAX);
        }
      }
    }

    let pm2_keys: alloc::vec::Vec<_> =
      self.physical_mesh2_resources.iter().map(|kv| *kv.key()).collect();
    for key in pm2_keys {
      if let Some((_, state)) = self.physical_mesh2_resources.remove(&key) {
        if let resources::ResourceState::Ready(mut resource) = state {
          resource.discard(device, &self.discard_pool, u64::MAX);
        }
      }
    }

    let sun_keys: alloc::vec::Vec<_> = self.sun_resources.iter().map(|kv| *kv.key()).collect();
    for key in sun_keys {
      if let Some((_, state)) = self.sun_resources.remove(&key) {
        if let resources::ResourceState::Ready(mut resource) = state {
          resource.discard(
            device,
            self.allocator.allocator.as_allocator_view(),
            &self.discard_pool,
            u64::MAX,
          );
        }
      }
    }

    for image in DebugTrackedRwLock::write(&self.billboard_resources).drain(..) {
      self.discard_pool.discard_image(
        self.allocator.allocator.get_raw(),
        image.image.get(),
        image.allocation,
        u64::MAX,
      );
      self.discard_pool.discard_image_view(image.image_view.get(), u64::MAX);
    }

    if let Some(sky_image) = DebugTrackedRwLock::write(&self.sky_image).take() {
      self.discard_pool.discard_image(
        self.allocator.allocator.get_raw(),
        sky_image.image.get(),
        sky_image.allocation,
        u64::MAX,
      );
      self.discard_pool.discard_image_view(sky_image.image_view.get(), u64::MAX);
    }

    debug_assert!(!self.has_discardables());
  }

  #[named]
  fn new<'a>(
    instance: &instance::Instance,
    physical_device: vk::PhysicalDevice,
    device: &LogicalDevice,
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
    let linear_sampler = unsafe { device.create_sampler(&sampler_info, None) }
      .with_name(device, "Global Linear Sampler")?;
    // - VMA Device Allocator
    // TODO: this function should cleanup everything on the first error, not leak everything
    let mut allocator = match unsafe {
      GlobalDeviceAllocator::new(
        &instance.instance,
        device,
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

    let renderpasses = renderpasses::RenderPasses::new(
      &instance.instance,
      &device,
      allocator.allocator.as_allocator_view(),
    );

    let pipeline_pool = match pipelines::PipelinePool::new(device, None) {
      Ok(pool) => pool,
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
    // TODO add a test only log to check that this is full and sorted (eg 0 1 2)
    for &queue_family_index in unique_family_indices_iter {
      unsafe {
        command_pools.push_unchecked(Some(Arc::new(commands::CommandPools::new(
          queue_family_index,
        ))))
      };
    }
    // - Swapchain hashmap
    let live_presentation_engines = dashmap::DashMap::new();

    // timeline semaphore promoted to core after 1.2 (included)
    debug_assert!(instance.api_version() < vk::API_VERSION_1_2);

    let frame_staging_arena =
      memory::FrameStagingArena::new(&allocator.allocator, 128 * 1024 * 1024)?;

    Ok(Self {
      allocator,
      command_pools: DebugTrackedRwLock::new(command_pools),
      discard_pool,
      live_presentation_engines,
      descriptor_pool: DebugTrackedRwLock::new(Some(descriptor_pool)),
      pipeline_pool,
      renderpasses,
      shader_manager: DebugTrackedRwLock::new(shader_manager::ShaderManager::new()),
      linear_sampler: unsafe { NonZeroHandle::new_unchecked(linear_sampler) },
      timeline_manager,
      physical_mesh_resources: dashmap::DashMap::new(),
      physical_mesh2_resources: dashmap::DashMap::new(),
      sun_resources: dashmap::DashMap::new(),
      billboard_resources: DebugTrackedRwLock::new(Vec::with_capacity(16)),
      sky_image: DebugTrackedRwLock::new(None),
      next_cmd_id: Arc::new(AtomicU64::new(1)),
      pending_downloads: DebugTrackedRwLock::new(hashbrown::HashMap::new()),
      frame_staging_arena: DebugTrackedRwLock::new(Some(frame_staging_arena)),
      sun_render_archetype_arena: None,
      physical_mesh_render_archetype_arena: None,
      physical_mesh2_render_archetype_arena: None,
      billboard_render_archetype_arena: None,
      particle_render_archetype_arena: None,
      particle2_render_archetype_arena: None,
      cursor_render_archetype_arena: None,
      marker_render_archetype_arena: None,
      measurement_render_archetype_arena: None,
      sky_render_archetype_arena: None,
      grid_render_archetype_arena: None,
      minimap_render_archetype_arena: None,
      text_render_archetype_arena: None,
      text2_render_archetype_arena: None,
      bvh_render_archetype_arena: None,
      bvhwire2_render_archetype_arena: None,
      sphere_gizmo_render_archetype_arena: None,
      gizmo_render_archetype_arena: None,
      trajectory_render_archetype_arena: None,
      ui_render_archetype_arena: None,
      background_render_archetype_arena: None,
      main_thread_cleanup_queue: alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new())),
    })
  }

  #[named]
  pub fn get_timeline_semaphore_cached_value(&self) -> u64 {
    self.timeline_manager.get_cached_value()
  }
}

#[derive(Clone, Copy)]
struct RecordingCmdBufferDataPresentation {
  acquire_result: AcquireResult,
  presentation_engine: PresentationEngineHandle,
  swapchain_generation: u64,
  wait_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
  signal_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
  submission_fence: NonZeroHandle<vk::Fence>,
}

/// Compositing context stored per-command-buffer when inside a compositing
/// render pass. Used by `bind_pipeline` to transparently create compositing-
/// compatible pipeline variants.
struct CompositingContext {
  /// The compositing render pass handle for pipeline variant creation
  render_pass: vk::RenderPass,
  /// The current subpass index (0 = macro, 1 = micro, 2 = composite)
  subpass: u32,
  /// The PE handle for this render pass
  pe_handle: PresentationEngineHandle,
}

struct RecordingCmdBufferData {
  command_buffer: NonZeroHandle<vk::CommandBuffer>,
  bound_pipeline: Option<NonZeroHandle<vk::Pipeline>>,
  presentation: Option<RecordingCmdBufferDataPresentation>,
  presentation_engine: Option<PresentationEngineHandle>,
  has_begun: bool,
  /// Set when inside a compositing render pass; enables transparent
  /// pipeline adaptation in bind_pipeline.
  compositing_ctx: Option<CompositingContext>,
}

impl RecordingCmdBufferData {
  #[named]
  fn new(command_buffer: NonZeroHandle<vk::CommandBuffer>) -> Self {
    Self {
      command_buffer,
      bound_pipeline: None,
      presentation: None,
      presentation_engine: None,
      has_begun: false,
      compositing_ctx: None,
    }
  }

  /// command buffer is automatically recycled by [`commands::CommandPools`]
  #[named]
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

/// TODO: Document this item
pub struct LogicalDevice {
  pub handle: ash::Device,
  pub submission_lock: spin::Mutex<()>,
  /// Note: Remove if API_VERSION_1_2
  pub create_renderpass2: ash::khr::create_renderpass2::Device,
  pub buffer_device_address: ash::khr::buffer_device_address::Device,
  pub timeline_semaphore: ash::khr::timeline_semaphore::Device,
  /// Note: Remove if API_VERSION_1_3
  pub synchronization2: ash::khr::synchronization2::Device,

  pub swapchain_maintenance1: Option<ash::ext::swapchain_maintenance1::Device>,

  #[cfg(debug_assertions)]
  pub debug_utils: ash::ext::debug_utils::Device,

  #[cfg(target_vendor = "apple")]
  pub metal_objects: ash::ext::metal_objects::Device,
}

impl core::ops::Deref for LogicalDevice {
  type Target = ash::Device;

  #[named]
  fn deref(&self) -> &Self::Target {
    &self.handle
  }
}

impl LogicalDevice {
  #[cfg(debug_assertions)]
  /// TODO: Document this item
  #[named]
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
  /// TODO: Document this item
  #[named]
  pub fn set_debug_name<T: vk::Handle>(&self, _object: T, _name: &str) {
    // This is a no-op in release builds, and should be optimized away.
  }

  /// TODO: Document this item
  #[named]
  pub fn locked_queue_submit(
    &self,
    queue: vk::Queue,
    submits: &[vk::SubmitInfo],
    fence: vk::Fence,
  ) -> ash::prelude::VkResult<()> {
    let _guard = self.submission_lock.lock();
    unsafe { self.handle.queue_submit(queue, submits, fence) }
  }

  /// TODO: Document this item
  #[named]
  pub fn wait_for_semaphore_value(
    &self,
    semaphore: vk::Semaphore,
    value: u64,
    timeout_ns: u64,
  ) -> ash::prelude::VkResult<()> {
    let semaphores = [semaphore];
    let values = [value];
    let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);

    unsafe { self.timeline_semaphore.wait_semaphores(&wait_info, timeout_ns) }
  }
}

/// TODO: Document this item
pub trait VulkanDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}
/// TODO: Document this item
pub trait VmaDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}

// 2. Apply to Results containing Vulkan Handles
impl<T: vk::Handle + Copy> VulkanDebugNameExt for ash::prelude::VkResult<T> {
  #[inline]
  #[named]
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
  #[named]
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
  #[named]
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

/// TODO: Document this item
pub struct Device {
  pub query_result: utils::PhysicalDeviceQueryResult,
  queues: Queues,
  pub instance: Arc<instance::Instance>,

  pub device: LogicalDevice,
  pub kernels: VulkanComputeKernels,

  pub res: Arc<DebugTrackedRwLock<DeviceResources>>,
  callback_stop_signal: Arc<core::sync::atomic::AtomicBool>,

  // Some bookkeeping I don't know where to put
  depth_stencil_format: vk::Format,
  /// Recording command buffers
  recording_command_buffers:
    DebugTrackedRwLock<hashbrown::HashMap<CommandBufferHandle, RecordingCmdBufferData>>,
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
pub struct Queue {
  pub handle: vk::Queue,
  pub index: u32,
  pub family_index: u32,
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
  #[named]
  fn from_device(device: &ash::Device, query_result: &utils::PhysicalDeviceQueryResult) -> Self {
    let unique_queue_families = query_result.unique_family_indices_set();
    let mut queue_buffer: heapless::Vec<Queue, MAX_QUEUE_COUNT> = heapless::Vec::new();
    for &family_index in unique_queue_families.iter() {
      let queue_info =
        vk::DeviceQueueInfo2::default().queue_family_index(family_index).queue_index(0);
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

  #[named]
  fn get_graphics_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::GRAPHICS).unwrap())
  }

  #[named]
  fn get_compute_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::COMPUTE).unwrap())
  }

  #[named]
  fn get_transfer_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::TRANSFER).unwrap())
  }
}

impl Device {
  pub fn get_compute_queue(&self) -> Queue {
    self.queues.get_compute_queue()
  }

  pub fn submit_paint_image_transition(
    &self,
    cmd_handle: gpu::CommandBufferHandle,
    mesh_id: crate::gpu::RenderableInstanceId,
    old_layout: ash::vk::ImageLayout,
    new_layout: ash::vk::ImageLayout,
  ) -> GpuResult<()> {
    let cmd_buffers = DebugTrackedRwLock::read(&self.recording_command_buffers);
    let data = cmd_buffers.get(&cmd_handle).ok_or(gpu_err_invalid_cmd!())?;
    let cmd = data.command_buffer.get();

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let mesh2_res = &res_guard.physical_mesh2_resources;
    let resource_ref = mesh2_res.get(&mesh_id).unwrap();
    let paint_image_resource = match resource_ref.value() {
      resources::ResourceState::Ready(r) => r,
      _ => panic!("resource not ready"),
    };
    let paint_image = paint_image_resource.emissive_paint_image.as_ref().unwrap();

    let image_barrier = ash::vk::ImageMemoryBarrier2::default()
      .src_stage_mask(ash::vk::PipelineStageFlags2::HOST)
      .src_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
      .dst_stage_mask(ash::vk::PipelineStageFlags2::FRAGMENT_SHADER)
      .dst_access_mask(ash::vk::AccessFlags2::SHADER_READ)
      .old_layout(old_layout)
      .new_layout(new_layout)
      .image(paint_image.image.get())
      .subresource_range(
        ash::vk::ImageSubresourceRange::default()
          .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
          .base_mip_level(0)
          .level_count(1)
          .base_array_layer(0)
          .layer_count(1),
      );

    let dep_info = ash::vk::DependencyInfo::default()
      .image_memory_barriers(core::slice::from_ref(&image_barrier));

    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);
    }
    Ok(())
  }

  /// Initializes a Device directly into the provided memory location
  /// This avoids returning a Device by value (which would probably cause stack overflow)
  #[named]
  pub(super) unsafe fn init_at_ptr(
    dst: *mut Self,
    instance: Arc<instance::Instance>,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<()> {
    unsafe { ptr::write(dst, Self::new(instance, index, query_input)?) };
    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn new(
    instance: Arc<instance::Instance>,
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

    let mut swapchain_maintenance1_features =
      vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default().swapchain_maintenance1(true);

    let mut device_create_info = vk::DeviceCreateInfo::default()
      .enabled_extension_names(&enabled_extension_names)
      .push_next(&mut features2)
      .queue_create_infos(&queue_infos);

    if chosen_physical_device_query_result
      .optional_extensions
      .contains(utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1)
    {
      device_create_info = device_create_info.push_next(&mut swapchain_maintenance1_features);
    }

    #[cfg(any(debug_assertions, test))]
    let device = unsafe {
      crate::gpu_backends::vulkan::device::hooks::load_device_with_hooks(
        &instance.instance,
        physical_device,
        &device_create_info,
      )
    }?;
    #[cfg(not(any(debug_assertions, test)))]
    let device =
      unsafe { instance.instance.create_device(physical_device, &device_create_info, None) }?;

    let queues = Queues::from_device(&device, chosen_physical_device_query_result);

    // bookkeeping data instantiation
    let depth_stencil_format = 'block: {
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
    };
    let depth_stencil_format = match depth_stencil_format {
      Ok(f) => f,
      Err(e) => {
        aethervk_oshal_rlib::log!("Device::new error in depth_stencil_format, destroying device!");
        unsafe { device.destroy_device(None) };
        return Err(e);
      }
    };

    let create_renderpass2 = ash::khr::create_renderpass2::Device::new(&instance.instance, &device);
    let synchronization2 = ash::khr::synchronization2::Device::new(&instance.instance, &device);
    let buffer_device_address =
      ash::khr::buffer_device_address::Device::new(&instance.instance, &device);
    let timeline_semaphore = ash::khr::timeline_semaphore::Device::new(&instance.instance, &device);

    let swapchain_maintenance1 = if chosen_physical_device_query_result
      .optional_extensions
      .contains(utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1)
    {
      Some(ash::ext::swapchain_maintenance1::Device::new(
        &instance.instance,
        &device,
      ))
    } else {
      None
    };

    #[cfg(debug_assertions)]
    let debug_utils = ash::ext::debug_utils::Device::new(&instance.instance, &device);
    #[cfg(target_vendor = "apple")]
    let metal_objects = ash::ext::metal_objects::Device::new(&instance.instance, &device);
    let device = LogicalDevice {
      timeline_semaphore,
      handle: device,
      submission_lock: spin::Mutex::new(()),
      create_renderpass2,
      synchronization2,
      buffer_device_address,
      swapchain_maintenance1,
      #[cfg(target_vendor = "apple")]
      metal_objects,
      #[cfg(debug_assertions)]
      debug_utils,
    };
    let mut res = match DeviceResources::new(
      instance.as_ref(),
      physical_device,
      &device,
      chosen_physical_device_query_result.unique_family_indices_set().iter(),
    ) {
      Ok(r) => r,
      Err(e) => {
        aethervk_oshal_rlib::log!("Device::new error in DeviceResources::new, destroying device!");
        unsafe { device.handle.destroy_device(None) };
        return Err(e);
      }
    };
    let unique_indices: alloc::vec::Vec<u32> = chosen_physical_device_query_result
      .unique_family_indices_set()
      .into_iter()
      .copied()
      .collect();
    let queue_sharing_info = crate::gpu::QueueSharingInfo {
      mode: if unique_indices.len() > 1 {
        crate::gpu::SharingMode::Concurrent
      } else {
        crate::gpu::SharingMode::Exclusive
      },
      queue_family_indices: unique_indices,
    };

    let kernels = match VulkanComputeKernels::new(
      &device,
      res.allocator.allocator.as_allocator_view(),
      queue_sharing_info,
      chosen_physical_device_query_result.debug_shaders,
      chosen_physical_device_query_result.subgroup_size,
    ) {
      Ok(k) => k,
      Err(e) => {
        aethervk_oshal_rlib::log!(
          "Device::new error in VulkanComputeKernels::new, destroying device!"
        );
        // We must clean up res manually before destroying the device!
        res.cleanup(&device);
        drop(res);
        unsafe { device.handle.destroy_device(None) };
        return Err(e);
      }
    };

    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device,
      queues,
      res: Arc::new(DebugTrackedRwLock::new(res)),
      callback_stop_signal: Arc::new(core::sync::atomic::AtomicBool::new(false)),
      kernels,
      instance,
      depth_stencil_format,
      recording_command_buffers: DebugTrackedRwLock::new(hashbrown::HashMap::new()),
    })
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }
}

impl Drop for Device {
  #[named]
  fn drop(&mut self) {
    aethervk_oshal_rlib::log!("Device::drop started. Waiting for device idle...");
    locks::set_disable_lock_assertions(true);

    // Signal stop to timeline polling task
    self.callback_stop_signal.store(true, Ordering::Release);

    // Wait for the timeline polling task to exit before destroying the device
    while Arc::strong_count(&self.callback_stop_signal) > 1 {
      oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
    }

    let mut cleanup = || {
      if let Err(e) = unsafe { self.device.device_wait_idle() } {
        aethervk_oshal_rlib::log!("Device::drop device_wait_idle failed: {:?}", e);
      }
      aethervk_oshal_rlib::log!("Device::drop device_wait_idle complete. Starting cleanup...");

      let allocator = DebugTrackedRwLock::read(&self.res).allocator.allocator.as_allocator_view();
      self.kernels.cleanup(&self.device, allocator);

      let arena_opt = self.res.read().frame_staging_arena.write().take();
      if let Some(mut arena) = arena_opt {
        arena.destroy(self.res.read().allocator.allocator.as_allocator_view());
      }

      DebugTrackedRwLock::write(&self.res).cleanup(&self.device);

      // Drain the main-thread cleanup queue. DeviceResources::cleanup() above
      // pushes swapchain/surface destruction tasks to the queue. We MUST execute
      // them before destroy_device(), or VkImage/VkSwapchain objects will leak.
      //
      // In production, flush_main_thread_cleanup_queue() is called from the main
      // thread (via GenericSimApp::on_close_requested or Avalonia's UI thread)
      // BEFORE Device::Drop runs. But in unit tests there is no event loop, so
      // we drain here as a safety net.
      {
        let res = DebugTrackedRwLock::read(&self.res);
        let mut queue = res.main_thread_cleanup_queue.lock();
        let tasks: alloc::vec::Vec<_> = queue.drain(..).collect();
        drop(queue);
        drop(res);
        for task in tasks {
          task();
        }
      }

      aethervk_oshal_rlib::log!("Device::drop cleanup complete. Destroying device...");
      // in the end, destroy the device
      unsafe { self.device.destroy_device(None) };
      aethervk_oshal_rlib::log!("Device::drop finished.");
    };

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    objc2::rc::autoreleasepool(|_| cleanup());

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    cleanup();
  }
}

impl RenderDevice for Device {
  #[named]
  fn get_measurement_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let arch = DebugTrackedRwLock::read(&pe.archetypes().measurement_render_archetype);
    if arch.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }

    let arch_ref = unsafe { arch.as_ref().unwrap_unchecked() };
    let pipeline_key = { arch_ref.pipeline_key };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE,
      texture_flags: TextureFlags::empty(),
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  #[named]
  fn get_gizmo_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().gizmo_render_archetype);
    let archetype = archetype_guard.as_ref().ok_or(gpu_err_archetype_absent!())?;

    let pipeline_key = Some(archetype.pipeline_key);

    Ok(ResourceUploadResult {
      pipeline: pipeline_key.expect("missing pipeline key"),
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE,
      texture_flags: TextureFlags::empty(),
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  #[named]
  fn get_marker_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let arch = DebugTrackedRwLock::read(&pe.archetypes().marker_render_archetype);
    if arch.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }

    let arch_ref = unsafe { arch.as_ref().unwrap_unchecked() };

    let pipeline_key = { arch_ref.pipeline_key };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE,
      texture_flags: TextureFlags::empty(),
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  #[named]
  fn create_billboard_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let prep = pe_lock.archetypes().prepare_update_billboard_archetype(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);
          crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
            &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &outline_info, rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
            pipeline_key: update.main_graphics_info.pipeline_key(),
            graphics_info: update.main_graphics_info,
            outline_data,
          })
        } else {
          None
        };
        Ok((h, compiled))
      })
      .commit_read(|state, execute_result| {
        let (h, compiled) = execute_result?;
        if let Some(c) = compiled {
          let pe_lock = wait_for_pe!(state, h)?;
          pe_lock.archetypes().commit_update_billboard_archetype(c);
        }
        Ok(())
      })
      .and_then(|_| self.get_billboard_resources(handle))
  }

  #[named]
  fn create_cursor_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let prep = pe_lock.archetypes().prepare_update_cursor_archetype(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);
          crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
            &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &outline_info, rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
            pipeline_key: update.main_graphics_info.pipeline_key(),
            graphics_info: update.main_graphics_info,
            outline_data,
          })
        } else {
          None
        };
        Ok((h, compiled))
      })
      .commit_read(|state, execute_result| {
        let (h, compiled) = execute_result?;
        if let Some(c) = compiled {
          let pe_lock = wait_for_pe!(state, h)?;
          pe_lock.archetypes().commit_update_cursor_archetype(c);
        }
        Ok(())
      })
      .and_then(|_| self.get_cursor_resources(handle))
  }

  #[named]
  fn create_measurement_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let prep = pe_lock.archetypes().prepare_update_measurement_archetype(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);
          crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
            &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &outline_info, rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
            pipeline_key: update.main_graphics_info.pipeline_key(),
            graphics_info: update.main_graphics_info,
            outline_data,
          })
        } else {
          None
        };
        Ok((h, compiled))
      })
      .commit_read(|state, execute_result| {
        let (h, compiled) = execute_result?;
        if let Some(c) = compiled {
          let pe_lock = wait_for_pe!(state, h)?;
          pe_lock.archetypes().commit_update_measurement_archetype(c);
        }
        Ok(())
      })
      .and_then(|_| self.get_measurement_resources(handle))
  }

  #[named]
  fn create_marker_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let prep = pe_lock.archetypes().prepare_update_marker_archetype(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);
          crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
            &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &outline_info, rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
            pipeline_key: update.main_graphics_info.pipeline_key(),
            graphics_info: update.main_graphics_info,
            outline_data,
          })
        } else {
          None
        };
        Ok((h, compiled))
      })
      .commit_read(|state, execute_result| {
        let (h, compiled) = execute_result?;
        if let Some(c) = compiled {
          let pe_lock = wait_for_pe!(state, h)?;
          pe_lock.archetypes().commit_update_marker_archetype(c);
        }
        Ok(())
      })
      .and_then(|_| self.get_marker_resources(handle))
  }

  #[named]
  fn create_gizmo_resources(
    &self,
    _cmd_buffer: CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let prep = pe_lock.archetypes().prepare_update_gizmo_archetype(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);
          crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
            &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &outline_info, rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
            pipeline_key: update.main_graphics_info.pipeline_key(),
            graphics_info: update.main_graphics_info,
            outline_data,
          })
        } else {
          None
        };
        Ok((h, compiled))
      })
      .commit_read(|state, execute_result| {
        let (h, compiled) = execute_result?;
        if let Some(c) = compiled {
          let pe_lock = wait_for_pe!(state, h)?;
          pe_lock.archetypes().commit_update_gizmo_archetype(c);
        }
        Ok(())
      })
      .and_then(|_| self.get_gizmo_resources(handle))
  }

  #[named]
  fn get_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let archetype = DebugTrackedRwLock::read(&pe.archetypes().cursor_render_archetype);
    if archetype.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }
    let archetype_ref = unsafe { archetype.as_ref().unwrap_unchecked() };

    let pipeline_key = archetype_ref.pipeline_key;

    // the cursor doesn't have descriptor sets or vertex/index buffers
    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  #[named]
  fn as_any(&self) -> &dyn Any {
    self
  }

  #[named]
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

  #[named]
  fn print_info(&self) -> String {
    let props = &self.query_result.physical_device_properties;
    let device_name = props.device_name_as_c_str().unwrap().to_string_lossy().into_owned();
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

  #[named]
  fn dump_memory_stats(&self) {
    #[cfg(all(debug_assertions, feature = "debug_gpu"))]
    {
      if let Ok(stats) =
        DebugTrackedRwLock::read(&self.res).allocator.allocator.build_stats_string(true)
      {
        aethervk_oshal_rlib::log!("[VMA STATS DUMP]\n{}", stats);
      }
    }
  }

  #[named]
  fn context_id(&self) -> u64 {
    vulkan::VULKAN_RENDER_BACKEND.0
  }

  #[named]
  fn subgroup_size(&self) -> u32 {
    self.query_result.subgroup_size
  }

  #[named]
  fn start_frame(&self) -> GpuResult<()> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        // 1. Refresh timeline and extract items ready for destruction
        let timeline_val = state.timeline_manager.refresh_cached_value()?;
        let items_to_destroy = state.discard_pool.pop_ready_items(timeline_val);
        state.allocator.set_current_frame_index(timeline_val);

        Ok(items_to_destroy)
      })?
      .execute(|items_to_destroy, _rollback| {
        // 2. Lock-free execution of Vulkan drop calls
        // (Note: No rollback defer is needed here because we are permanently destroying data, not creating it)
        resources::DiscardPool::destroy_items_lock_free(&self.device, items_to_destroy);

        Ok(())
      })
      .commit_read(|state, execute_result| {
        execute_result?;

        // 3. Reset the staging arena safely while we have the state context
        if let Some(arena) = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &state.frame_staging_arena,
        )
        .as_mut()
        {
          arena.reset();
        }

        Ok(())
      })
  }

  /// Initializes all archetypes in the order they are declared inside `DeviceResources`
  /// Initializes all archetypes in the order they are declared inside `DeviceResources`
  #[named]
  fn init_archetypes(&self, handle: crate::gpu::PresentationEngineHandle) -> GpuResult<()> {
    struct ExtractedArenas {
      mesh: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMeshRenderResourceArchetypeArena>>,
      >,
      mesh2: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMesh2RenderResourceArchetypeArena>>,
      >,
      sun: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::SunRenderResourceArchetypeArena>>>,
      sky: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::SkyRenderResourceArchetypeArena>>>,
      background: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::BackgroundRenderResourceArchetypeArena>>,
      >,
      grid:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::GridRenderResourceArchetypeArena>>>,
      minimap: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::MinimapRenderResourceArchetypeArena>>,
      >,
      measurement: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::MeasurementRenderResourceArchetypeArena>>,
      >,
      marker:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::MarkerRenderResourceArchetypeArena>>>,
      billboard: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::BillboardRenderResourceArchetypeArena>>,
      >,
      particle: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::ParticleRenderResourceArchetypeArena>>,
      >,
      particle2: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::Particle2RenderResourceArchetypeArena>>,
      >,
      trajectory: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::TrajectoryRenderResourceArchetypeArena>>,
      >,
      ui: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::UiRenderResourceArchetypeArena>>>,
      cursor:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::CursorRenderResourceArchetypeArena>>>,
      text:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::TextRenderResourceArchetypeArena>>>,
      text2:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Text2RenderResourceArchetypeArena>>>,
      bvh: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::BvhRenderResourceArchetypeArena>>>,
      bvhwire2: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::Bvhwire2RenderResourceArchetypeArena>>,
      >,
      sphere_gizmo: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::SphereGizmoRenderResourceArchetypeArena>>,
      >,
      gizmo:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::GizmoRenderResourceArchetypeArena>>>,
    }

    let queue = self.queues.get_graphics_queue();
    let depth_stencil_format = self.depth_stencil_format;

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_write(handle, |state, h| {
        let pe: swapchain::PresentationState = extract_pe!(state, h)?;
        let timeline = state.timeline_manager.get_cached_value() + 1;

        let vma = state.allocator.allocator.as_allocator_view();
        let discard_pool_ptr = &state.discard_pool as *const _;
        let renderpasses_ptr = &state.renderpasses as *const _;
        let shader_manager_ptr = &state.shader_manager as *const _;
        let pipeline_pool_ptr = &state.pipeline_pool as *const _;
        let frame_staging_arena_ptr = state.frame_staging_arena.read().as_ref().map(|a| a as *const memory::FrameStagingArena).ok_or(gpu_err!("no frame staging arena"))?;

        let arenas = ExtractedArenas {
          mesh: state.physical_mesh_render_archetype_arena.clone(),
          mesh2: state.physical_mesh2_render_archetype_arena.clone(),
          sun: state.sun_render_archetype_arena.clone(),
          sky: state.sky_render_archetype_arena.clone(),
          background: state.background_render_archetype_arena.clone(),
          grid: state.grid_render_archetype_arena.clone(),
          minimap: state.minimap_render_archetype_arena.clone(),
          measurement: state.measurement_render_archetype_arena.clone(),
          marker: state.marker_render_archetype_arena.clone(),
          billboard: state.billboard_render_archetype_arena.clone(),
          particle: state.particle_render_archetype_arena.clone(),
          particle2: state.particle2_render_archetype_arena.clone(),
          trajectory: state.trajectory_render_archetype_arena.clone(),
          ui: state.ui_render_archetype_arena.clone(),
          cursor: state.cursor_render_archetype_arena.clone(),
          text: state.text_render_archetype_arena.clone(),
          text2: state.text2_render_archetype_arena.clone(),
          bvh: state.bvh_render_archetype_arena.clone(),
          bvhwire2: state.bvhwire2_render_archetype_arena.clone(),
          sphere_gizmo: state.sphere_gizmo_render_archetype_arena.clone(),
          gizmo: state.gizmo_render_archetype_arena.clone(),
        };

        Ok::<_, GpuError>((pe, timeline, vma, discard_pool_ptr, renderpasses_ptr, shader_manager_ptr, pipeline_pool_ptr, frame_staging_arena_ptr, arenas))
      })?
      .execute(|(mut pe, timeline, vma, discard_pool_ptr, renderpasses_ptr, shader_manager_ptr, pipeline_pool_ptr, frame_staging_arena_ptr, mut arenas), rollback| {
        let allocator = vma;
        let discard_pool = unsafe { &*discard_pool_ptr };
        let renderpasses = unsafe { &*renderpasses_ptr };
        let shader_manager = unsafe { &*shader_manager_ptr };
        let pipeline_pool = unsafe { &*pipeline_pool_ptr };
        let staging_arena = if frame_staging_arena_ptr.is_null() { None } else { Some(unsafe { &*frame_staging_arena_ptr }) };
        let device = &self.device;

        // Extract the format ONCE before the closure to prevent borrow checking overlap
        let pe_format = pe.format();

        let mut run = || -> GpuResult<()> {
          macro_rules! init_arch {
            // standard
            ($arena_field:ident, $ensure_fn:ident, $archetype_field:ident, $arena_type:ident, $create_fn:ident) => {
              let needs_init = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&pe.archetypes().$archetype_field).is_none();
              if needs_init {
                let (vkey, fkey) = {
                  let mut sm = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(shader_manager);
                  $ensure_fn(device, &mut sm)?
                };
                let (vertex_shader, fragment_shader) = {
                  let sm_read = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(shader_manager);
                  let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                  let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                  (vs, fs)
                };

                if arenas.$arena_field.is_none() {
                  let ctx = resources::ArenaCreationContext {
                    device, allocator, discard_pool, queue: Some(&queue), staging_arena,
                    vertex_shader: Some(&*vertex_shader), fragment_shader: Some(&*fragment_shader),
                    outline_vertex_shader: None, outline_fragment_shader: None,
                  };
                  let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&ctx)?;
                  arenas.$arena_field = Some(alloc::sync::Arc::new(crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(new_arena)));
                }
                pe.archetypes_mut().$create_fn(
                  device, &*vertex_shader, &*fragment_shader, depth_stencil_format, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), rollback
                )?;
              }
            };
            // text
            ($arena_field:ident, $ensure_fn:ident, $archetype_field:ident, $arena_type:ident, $create_fn:ident, text) => {
              let needs_init = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&pe.archetypes().$archetype_field).is_none();
              if needs_init {
                let (vkey, fkey) = {
                  let mut sm = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(shader_manager);
                  $ensure_fn(device, &mut sm)?
                };
                let (vertex_shader, fragment_shader) = {
                  let sm_read = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(shader_manager);
                  let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                  let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                  (vs, fs)
                };

                if arenas.$arena_field.is_none() {
                  let ctx = resources::ArenaCreationContext {
                    device, allocator, discard_pool, queue: Some(&queue), staging_arena,
                    vertex_shader: Some(&*vertex_shader), fragment_shader: Some(&*fragment_shader),
                    outline_vertex_shader: None, outline_fragment_shader: None,
                  };
                  let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&ctx)?;
                  arenas.$arena_field = Some(alloc::sync::Arc::new(crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(new_arena)));
                }
                pe.archetypes_mut().$create_fn(
                  device, &*vertex_shader, &*fragment_shader, depth_stencil_format, &queue, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), rollback
                )?;
              }
            };
            // ref_alloc
            ($arena_field:ident, $ensure_fn:ident, $archetype_field:ident, $arena_type:ident, $create_fn:ident, ref_alloc) => {
              init_arch!($arena_field, $ensure_fn, $archetype_field, $arena_type, $create_fn);
            };
            // mesh
            ($arena_field:ident, $ensure_fn:ident, $archetype_field:ident, $arena_type:ident, $create_fn:ident, mesh) => {
              let needs_init = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&pe.archetypes().$archetype_field).is_none();
              if needs_init {
                let (vkey, fkey, ovkey, ofkey) = {
                  let mut sm = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(shader_manager);
                  $ensure_fn(device, &mut sm)?
                };
                let (vs, fs, ovs, ofs) = {
                  let sm_read = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(shader_manager);
                  let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                  let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                  let ovs = get_shader(&sm_read, ovkey, ash::vk::ShaderStageFlags::VERTEX)?;
                  let ofs = get_shader(&sm_read, ofkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                  (vs, fs, ovs, ofs)
                };

                if arenas.$arena_field.is_none() {
                  let ctx = resources::ArenaCreationContext {
                    device, allocator, discard_pool, queue: Some(&queue), staging_arena,
                    vertex_shader: Some(&*vs), fragment_shader: Some(&*fs),
                    outline_vertex_shader: Some(&*ovs), outline_fragment_shader: Some(&*ofs),
                  };
                  let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&ctx)?;
                  arenas.$arena_field = Some(alloc::sync::Arc::new(crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(new_arena)));
                }
                pe.archetypes_mut().$create_fn(
                  device, &*vs, &*fs, &*ovs, &*ofs, depth_stencil_format, &queue, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), rollback
                )?;
              }
            };
          }

          init_arch!(mesh, ensure_physical_mesh_shader_modules, physical_mesh_render_archetype, ForwardMeshRenderResourceArchetypeArena, create_physical_mesh_archetype, mesh);
          init_arch!(mesh2, ensure_physical_mesh2_shader_modules, physical_mesh2_render_archetype, ForwardMesh2RenderResourceArchetypeArena, create_physical_mesh2_archetype, mesh);
          init_arch!(sun, ensure_sun_shader_modules, sun_render_archetype, SunRenderResourceArchetypeArena, create_sun_archetype);
          init_arch!(sky, ensure_sky_shader_modules, sky_render_archetype, SkyRenderResourceArchetypeArena, create_sky_archetype);
          init_arch!(background, ensure_background_shader_modules, background_render_archetype, BackgroundRenderResourceArchetypeArena, create_background_archetype);
          init_arch!(grid, ensure_grid_shader_modules, grid_render_archetype, GridRenderResourceArchetypeArena, create_grid_archetype);
          init_arch!(minimap, ensure_minimap_shader_modules, minimap_render_archetype, MinimapRenderResourceArchetypeArena, create_minimap_archetype);
          init_arch!(measurement, ensure_measurement_shader_modules, measurement_render_archetype, MeasurementRenderResourceArchetypeArena, create_measurement_archetype);
          init_arch!(marker, ensure_marker_shader_modules, marker_render_archetype, MarkerRenderResourceArchetypeArena, create_marker_archetype);
          init_arch!(billboard, ensure_billboard_shader_modules, billboard_render_archetype, BillboardRenderResourceArchetypeArena, create_billboard_archetype);
          init_arch!(particle, ensure_particle_shader_modules, particle_render_archetype, ParticleRenderResourceArchetypeArena, create_particle_archetype, ref_alloc);
          init_arch!(particle2, ensure_particle2_shader_modules, particle2_render_archetype, Particle2RenderResourceArchetypeArena, create_particle2_archetype, ref_alloc);
          init_arch!(trajectory, ensure_trajectory_shader_modules, trajectory_render_archetype, TrajectoryRenderResourceArchetypeArena, create_trajectory_archetype, ref_alloc);
          init_arch!(ui, ensure_ui_shader_modules, ui_render_archetype, UiRenderResourceArchetypeArena, create_ui_archetype, ref_alloc);
          init_arch!(cursor, ensure_cursor_shader_modules, cursor_render_archetype, CursorRenderResourceArchetypeArena, create_cursor_archetype);
          init_arch!(text, ensure_text_shader_modules, text_render_archetype, TextRenderResourceArchetypeArena, create_text_archetype, text);
          init_arch!(text2, ensure_text2_shader_modules, text2_render_archetype, Text2RenderResourceArchetypeArena, create_text2_archetype, text);
          init_arch!(bvh, ensure_bvh_shader_modules, bvh_render_archetype, BvhRenderResourceArchetypeArena, create_bvh_archetype);
          init_arch!(bvhwire2, ensure_bvhwire2_shader_modules, bvhwire2_render_archetype, Bvhwire2RenderResourceArchetypeArena, create_bvhwire2_archetype);
          init_arch!(sphere_gizmo, ensure_sphere_gizmo_shader_modules, sphere_gizmo_render_archetype, SphereGizmoRenderResourceArchetypeArena, create_sphere_gizmo_archetype);
          init_arch!(gizmo, ensure_gizmo_shader_modules, gizmo_render_archetype, GizmoRenderResourceArchetypeArena, create_gizmo_archetype);

          Ok(())
        };

        let result = run();

        Ok((pe, result, arenas))
      })
      .commit(|state, execute_result| {
        let (pe, result, arenas) = execute_result?;

        state.live_presentation_engines.insert(handle, pe);

        // Always save arenas first, even if result is an Error, to prevent resource leaks
        // and safely reuse successfully established arenas upon subsequent retries.
        state.physical_mesh_render_archetype_arena = arenas.mesh;
        state.physical_mesh2_render_archetype_arena = arenas.mesh2;
        state.sun_render_archetype_arena = arenas.sun;
        state.sky_render_archetype_arena = arenas.sky;
        state.background_render_archetype_arena = arenas.background;
        state.grid_render_archetype_arena = arenas.grid;
        state.minimap_render_archetype_arena = arenas.minimap;
        state.measurement_render_archetype_arena = arenas.measurement;
        state.marker_render_archetype_arena = arenas.marker;
        state.billboard_render_archetype_arena = arenas.billboard;
        state.particle_render_archetype_arena = arenas.particle;
        state.particle2_render_archetype_arena = arenas.particle2;
        state.trajectory_render_archetype_arena = arenas.trajectory;
        state.ui_render_archetype_arena = arenas.ui;
        state.cursor_render_archetype_arena = arenas.cursor;
        state.text_render_archetype_arena = arenas.text;
        state.text2_render_archetype_arena = arenas.text2;
        state.bvh_render_archetype_arena = arenas.bvh;
        state.bvhwire2_render_archetype_arena = arenas.bvhwire2;
        state.sphere_gizmo_render_archetype_arena = arenas.sphere_gizmo;
        state.gizmo_render_archetype_arena = arenas.gizmo;

        result?;

        Ok(())
      })?;

    Ok(())
  }

  #[named]
  fn set_line_width(&self, cmd_buffer: gpu::CommandBufferHandle, width: f32) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;
    unsafe {
      self.device.cmd_set_line_width(cmd, width);
    }
    Ok(())
  }

  #[named]
  fn create_presentation_engine(
    &self,
    params: &crate::gpu::PresentationEngineParams,
  ) -> GpuResult<crate::gpu::PresentationEngineHandle> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_state, _| {
        static NEXT_HANDLE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
        let handle =
          PresentationEngineHandle(NEXT_HANDLE.fetch_add(1, core::sync::atomic::Ordering::Relaxed));
        Ok::<_, GpuError>(handle)
      })?
      .execute(|handle, rollback| {
        let entry =
          self
            .instance
            .entry_wrapper
            .weak_entry()
            .upgrade()
            .ok_or(GpuError::BackendSpecific(
              "Vulkan Entry wasn't loaded".to_string(),
            ))?;
        let physical_device_handle =
          unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };

        let main_thread_queue = {
          let res = DebugTrackedRwLock::read(&self.res);
          res.main_thread_cleanup_queue.clone()
        };

        let pe = swapchain::PresentationState::new(
          &entry,
          &self.instance.instance,
          &self.device,
          physical_device_handle,
          self.device.swapchain_maintenance1.clone(),
          params,
          rollback,
          main_thread_queue,
        )?;

        Ok((handle, pe))
      })
      .commit_read(|state, execute_result| {
        let (handle, pe) = execute_result?;
        state.live_presentation_engines.insert(handle, pe);
        Ok(handle)
      })
      .and_then(|handle| {
        self.init_archetypes(handle)?;
        Ok(handle)
      })
  }

  // TODO write unit tests about creating a presentation engine and destroying it. Check that, for both
  // TODO windowless and windowed, there shouldn't be any validation errors
  #[named]
  fn destroy_presentation_engine(&self, handle: PresentationEngineHandle) -> GpuResult<()> {
    // check existance
    let presentation_engine = {
      let res = DebugTrackedRwLock::read(&self.res);
      let engines = &res.live_presentation_engines;
      match engines.remove(&handle) {
        Some((_, engine)) => engine,
        None => {
          return Err(GpuError::BackendSpecific(alloc::format!(
            "[Vulkan RenderDevice] destroy_presentation_engine doesn't contain presentation engine {}",
            handle.0
          )));
        }
      }
    };

    let res_write = DebugTrackedRwLock::read(&self.res);
    let timeline = res_write.timeline_manager.get_next_submit_value();

    // Also, clear all pending windowless downloads for this presentation engine to free staging memory
    let mut pending_downloads = DebugTrackedRwLock::write(&res_write.pending_downloads);
    let mut to_remove = Vec::new();
    for (&tid, download) in pending_downloads.iter() {
      if let Some(p) = download.presentation_engine
        && p == handle
      {
        to_remove.push(tid);
      }
    }
    for tid in to_remove {
      if let Some(mut download) = pending_downloads.remove(&tid) {
        unsafe {
          res_write
            .allocator
            .allocator
            .destroy_buffer(download.staging_buffer, &mut download.allocation);
        }
      }
    }

    // Discard using the next submit value to ensure all currently queued frames are completed
    res_write.discard_pool.discard_type_erased(presentation_engine, timeline);

    Ok(())
  }

  fn process_main_thread_cleanup_queue(&self) -> GpuResult<()> {
    let res = DebugTrackedRwLock::read(&self.res);
    let mut queue = res.main_thread_cleanup_queue.lock();
    let tasks: alloc::vec::Vec<_> = queue.drain(..).collect();
    drop(queue); // release lock before executing tasks
    drop(res);
    for task in tasks {
      task();
    }
    Ok(())
  }

  #[named]
  fn flush_main_thread_cleanup_queue(&self) -> GpuResult<()> {
    // 1. Drain any pending tasks from runtime (resize discards)
    self.process_main_thread_cleanup_queue()?;

    // 2. Extract and destroy all WINDOWED presentation engines directly.
    //    We're on the main thread, so CAMetalLayer teardown is safe.
    let res = DebugTrackedRwLock::read(&self.res);
    let keys: alloc::vec::Vec<_> = res
      .live_presentation_engines
      .iter()
      .filter(|kv| matches!(kv.value(), swapchain::PresentationState::Windowed(_)))
      .map(|kv| *kv.key())
      .collect();

    for k in keys {
      if let Some((_, mut pe)) = res.live_presentation_engines.remove(&k) {
        aethervk_oshal_rlib::log!(
          "flush_main_thread_cleanup_queue: cleaning up windowed PE {:?} on main thread",
          k
        );
        // This calls WindowedPresentationState::cleanup() which pushes to the queue
        pe.cleanup(&self.device);
      }
    }
    drop(res);

    // 3. Drain any tasks that were just pushed by step 2
    self.process_main_thread_cleanup_queue()?;

    Ok(())
  }

  #[named]
  fn resize_presentation_engine(
    &self,
    handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    let physical_device_handle = unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe = extract_pe!(state, h)?;

        let backup = match &pe {
          swapchain::PresentationState::Windowed(w) => Some(w.backup_resize_state()),
          _ => None,
        };
        Ok((pe, backup))
      })?
      .execute(|(mut pe, backup), rollback| {
        let extent = pe.extent();
        if extent.0 == width && extent.1 == height {
          return Ok((pe, backup, Ok(())));
        }
        let resize_res = pe.resize(
          &self.instance.instance,
          &self.device,
          physical_device_handle,
          width,
          height,
          rollback,
        );
        Ok((pe, backup, resize_res))
      })
      .and_then_prepare_read((), |state, (pe, backup, resize_result), _| {
        if resize_result.is_err() {
          return Ok((pe, backup, resize_result, None));
        }

        let format = pe.format();
        // Encapsulate preparations to catch `?` aborts without dropping `pe`
        let preps_res = (|| -> GpuResult<AllPreps> {
          Ok(AllPreps {
            physical_mesh: pe.archetypes().prepare_update_physical_mesh_archetype(format, &state.renderpasses)?,
            physical_mesh2: pe.archetypes().prepare_update_physical_mesh2_archetype(format, &state.renderpasses)?,
            cursor: pe.archetypes().prepare_update_cursor_archetype(format, &state.renderpasses)?,
            particle: pe.archetypes().prepare_update_particle_archetype(format, &state.renderpasses)?,
            particle2: pe.archetypes().prepare_update_particle2_archetype(format, &state.renderpasses)?,
            sun: pe.archetypes().prepare_update_sun_archetype(format, &state.renderpasses)?,
            sky: pe.archetypes().prepare_update_sky_archetype(format, &state.renderpasses)?,
            grid: pe.archetypes().prepare_update_grid_archetype(format, &state.renderpasses)?,
            minimap: pe.archetypes().prepare_update_minimap_archetype(format, &state.renderpasses)?,
            measurement: pe.archetypes().prepare_update_measurement_archetype(format, &state.renderpasses)?,
            marker: pe.archetypes().prepare_update_marker_archetype(format, &state.renderpasses)?,
            text: pe.archetypes().prepare_update_text_archetype(format, &state.renderpasses)?,
            text2: pe.archetypes().prepare_update_text2_archetype(format, &state.renderpasses)?,
            bvh: pe.archetypes().prepare_update_bvh_archetype(format, &state.renderpasses)?,
            bvhwire2: pe.archetypes().prepare_update_bvhwire2_archetype(format, &state.renderpasses)?,
            gizmo: pe.archetypes().prepare_update_gizmo_archetype(format, &state.renderpasses)?,
            trajectory: pe.archetypes().prepare_update_trajectory_archetype(format, &state.renderpasses)?,
            ui: pe.archetypes().prepare_update_ui_archetype(format, &state.renderpasses)?,
            background: pe.archetypes().prepare_update_background_archetype(format, &state.renderpasses)?,
            billboard: pe.archetypes().prepare_update_billboard_archetype(format, &state.renderpasses)?,
          })
        })();

        let all_preps_opt = match preps_res {
          Ok(preps) => Some(Ok(preps)),
          Err(e) => Some(Err(e)),
        };
        Ok((pe, backup, resize_result, all_preps_opt))
      })
      .execute(|(pe, backup, resize_result, all_preps_opt), rollback| {
        // Evaluate compilation safely returning out the Result encapsulated
        let compiled_res = (|| -> GpuResult<Option<AllCompiled>> {
          let prep = match all_preps_opt {
            Some(Ok(p)) => p,
            Some(Err(e)) => return Err(e),
            None => return Ok(None),
          };

          let state = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.res);

          let physical_mesh = if let Some(update) = prep.physical_mesh {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let physical_mesh2 = if let Some(update) = prep.physical_mesh2 {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let cursor = if let Some(update) = prep.cursor {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let particle = if let Some(update) = prep.particle {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let particle2 = if let Some(update) = prep.particle2 {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let sun = if let Some(update) = prep.sun {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let sky = if let Some(update) = prep.sky {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let grid = if let Some(update) = prep.grid {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let minimap = if let Some(update) = prep.minimap {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let measurement = if let Some(update) = prep.measurement {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let marker = if let Some(update) = prep.marker {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let text = if let Some(update) = prep.text {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let text2 = if let Some(update) = prep.text2 {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let bvh = if let Some(update) = prep.bvh {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let bvhwire2 = if let Some(update) = prep.bvhwire2 {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let gizmo = if let Some(update) = prep.gizmo {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let trajectory = if let Some(update) = prep.trajectory {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let ui = if let Some(update) = prep.ui {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let background = if let Some(update) = prep.background {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          let billboard = if let Some(update) = prep.billboard {
            crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
              &state.pipeline_pool, &self.device, &update.main_graphics_info, rollback,
            )?;
            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              crate::gpu_backends::vulkan::device::pipelines::PipelinePool::get_or_create_graphics_pipeline(
                &state.pipeline_pool, &self.device, &outline_info, rollback,
              )?;
              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            Some(crate::gpu_backends::vulkan::device::archetypes_struct::CompiledArchetypeData {
              pipeline_key: update.main_graphics_info.pipeline_key(),
              graphics_info: update.main_graphics_info,
              outline_data,
            })
          } else { None };

          Ok(Some(AllCompiled {
            physical_mesh,
            physical_mesh2,
            cursor,
            particle,
            particle2,
            sun,
            sky,
            grid,
            minimap,
            measurement,
            marker,
            text,
            text2,
            bvh,
            bvhwire2,
            gizmo,
            trajectory,
            ui,
            background,
            billboard,
          }))
        })();

        Ok((pe, backup, resize_result, compiled_res))
      })
      .commit_read(|state, execute_result| {
        let (mut pe, backup, resize_result, compiled_res) = execute_result.unwrap();

        // Safely catch errors here where 'pe' can be cleanly re-inserted to DashMap
        if resize_result.is_err() || compiled_res.is_err() {
          if let (swapchain::PresentationState::Windowed(w), Some(bkp)) = (&mut pe, backup) {
            w.restore_resize_state(bkp);
          }
          state.live_presentation_engines.insert(handle, pe);

          resize_result?;
          return Err(unsafe { compiled_res.unwrap_err_unchecked() });
        } else if let Some(compiled) = unsafe { compiled_res.unwrap_unchecked() } {
          if let Some(c) = compiled.physical_mesh { pe.archetypes().commit_update_physical_mesh_archetype(c); }
          if let Some(c) = compiled.physical_mesh2 { pe.archetypes().commit_update_physical_mesh2_archetype(c); }
          if let Some(c) = compiled.cursor { pe.archetypes().commit_update_cursor_archetype(c); }
          if let Some(c) = compiled.particle { pe.archetypes().commit_update_particle_archetype(c); }
          if let Some(c) = compiled.particle2 { pe.archetypes().commit_update_particle2_archetype(c); }
          if let Some(c) = compiled.sun { pe.archetypes().commit_update_sun_archetype(c); }
          if let Some(c) = compiled.sky { pe.archetypes().commit_update_sky_archetype(c); }
          if let Some(c) = compiled.grid { pe.archetypes().commit_update_grid_archetype(c); }
          if let Some(c) = compiled.minimap { pe.archetypes().commit_update_minimap_archetype(c); }
          if let Some(c) = compiled.measurement { pe.archetypes().commit_update_measurement_archetype(c); }
          if let Some(c) = compiled.marker { pe.archetypes().commit_update_marker_archetype(c); }
          if let Some(c) = compiled.text { pe.archetypes().commit_update_text_archetype(c); }
          if let Some(c) = compiled.text2 { pe.archetypes().commit_update_text2_archetype(c); }
          if let Some(c) = compiled.bvh { pe.archetypes().commit_update_bvh_archetype(c); }
          if let Some(c) = compiled.bvhwire2 { pe.archetypes().commit_update_bvhwire2_archetype(c); }
          if let Some(c) = compiled.gizmo { pe.archetypes().commit_update_gizmo_archetype(c); }
          if let Some(c) = compiled.trajectory { pe.archetypes().commit_update_trajectory_archetype(c); }
          if let Some(c) = compiled.ui { pe.archetypes().commit_update_ui_archetype(c); }
          if let Some(c) = compiled.background { pe.archetypes().commit_update_background_archetype(c); }
          if let Some(c) = compiled.billboard { pe.archetypes().commit_update_billboard_archetype(c); }
        }

        state.live_presentation_engines.insert(handle, pe);
        Ok(())
      })
  }

  #[named]
  fn get_presentation_engine_extent(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<[u32; 2]> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let engine = wait_for_pe!(res_guard, handle)?;
    let e = engine.extent();
    Ok([e.0, e.1])
  }

  #[named]
  fn is_presentation_engine_windowless(&self, handle: PresentationEngineHandle) -> GpuResult<bool> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    match &*pe {
      PresentationState::Windowed(_) => Ok(false),
      PresentationState::Windowless(_) => Ok(true),
    }
  }

  #[named]
  fn acquire_next_image(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<crate::gpu::AcquireResult> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        // EXTRACT the engine from the DashMap.
        let pe = extract_pe!(state, h)?;

        let backup = pe.backup_resize_state();
        Ok((pe, backup))
      })?
      .execute(|(mut pe, backup), rollback| {
        // EXECUTE lock-free!
        // `vkAcquireNextImageKHR` natively blocks the CPU waiting for VSync.
        // Because `pe` is extracted, streaming/audio threads can still lock `self.res`!
        let result = pe.acquire_next_image(&self.device, rollback);

        // We cannot fail `execute` directly via `?` because we would lose `pe`.
        Ok((pe, backup, result))
      })
      .commit_read(|state, ok_result| {
        // REPLACE: Put the engine back unconditionally, even if `acquire` threw an OUT_OF_DATE error.
        let (mut pe, backup, acquire_result) = ok_result.unwrap();

        if acquire_result.is_err() {
          pe.restore_resize_state(backup);
        }

        state.live_presentation_engines.insert(handle, pe);
        acquire_result
      })
  }

  #[named]
  fn cancel_acquired_image(
    &self,
    handle: PresentationEngineHandle,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| extract_pe!(state, h))?
      .execute(|mut pe, _rollback| {
        let result = pe.cancel_image(
          &self.device,
          self.queues.get_graphics_queue().handle,
          image_index,
          frame_index,
        );
        Ok((pe, result))
      })
      .commit_read(|state, ok_result| {
        let (pe, result) = ok_result.unwrap();
        state.live_presentation_engines.insert(handle, pe);
        result
      })
  }

  #[named]
  fn get_physical_mesh_resources(
    &self,
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = wait_for_pe_direct!(pe_lock, handle)?;
    let mesh_archetypes = DebugTrackedRwLock::read(&pe.archetypes().physical_mesh_render_archetype);
    let (pipeline_key, outline_pipeline_key) = {
      let archetype_ref = { mesh_archetypes.as_ref().ok_or(gpu_err_archetype_absent!()) }?;
      let pipeline_key = archetype_ref.pipeline_key;
      let outline_pipeline_key = archetype_ref.outline_pipeline_key;
      (pipeline_key, outline_pipeline_key)
    };

    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(asset_hash);

    if let Some(entry) = res_guard.physical_mesh_resources.get(&physical_mesh_id) {
      if let resources::ResourceState::Ready(resource) = entry.value() {
        return Ok(ResourceUploadResult {
          pipeline: pipeline_key,
          outline_pipeline: Some(outline_pipeline_key),
          buffers: physical_mesh_id.into(),
          texture_flags: resource.frontend_texture_flags(),
          indirect_buffer: None,
          descriptor_index: None,
        });
      }
    }
    Err(GpuError::NotFound)
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
  #[named]
  fn create_physical_mesh_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    asset_hash: u64,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult> {
    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(asset_hash);
    let cmd = self.get_cmd(cmd_buffer)?;
    let timeline = DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;

    let mut is_winner = false;
    loop {
      // 1. Define the actions you want to take outside the lock
      enum Action {
        Return,
        Yield,
        BreakWinner,
      }

      // 2. Create an inner scope to strictly bound the lifetime of `res_guard`
      let action = {
        let res_guard = DebugTrackedRwLock::read(&self.res);

        match res_guard.physical_mesh_resources.entry(physical_mesh_id) {
          dashmap::mapref::entry::Entry::Occupied(e) => match e.get() {
            resources::ResourceState::Ready(_) => Action::Return,
            resources::ResourceState::Pending => Action::Yield,
          },
          dashmap::mapref::entry::Entry::Vacant(e) => {
            e.insert(resources::ResourceState::Pending);
            Action::BreakWinner
          }
        }
        // Rust automatically drops `e` and then `res_guard` right here!
      };

      // 3. Execute the heavy/blocking actions without holding the lock
      match action {
        Action::Return => {
          return self.get_physical_mesh_resources(asset_hash, handle);
        }
        Action::Yield => {
          aethervk_oshal_rlib::os::native::this_thread::yield_now();
          continue;
        }
        Action::BreakWinner => {
          is_winner = true;
          break;
        }
      }
    }

    let execution_result = (|| -> GpuResult<Option<ResourceUploadResult>> {
      crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
        .prepare_read(handle, |state, h| {
          let live_pes = &state.live_presentation_engines;
          let presentation_engine = wait_for_pe_direct!(live_pes, h)?;
          let archetype = DebugTrackedRwLock::read(
            &presentation_engine.archetypes().physical_mesh_render_archetype,
          );
          if archetype.as_ref().is_none() {
            return Err(gpu_err_invalid_pe!());
          }
          let archetype_ref = unsafe { archetype.as_ref().unwrap_unchecked() };

          let pipeline_key = archetype_ref.pipeline_key;
          let outline_pipeline_key = archetype_ref.outline_pipeline_key;
          let arena_arc = archetype_ref.deref_arena().ok_or(gpu_err!("arena absent"))?.clone();

          let position_data = extract_position_data(component.mesh.as_ref());
          let attribute_data = extract_attribute_data(component.mesh.as_ref());

          let vma = state.allocator.allocator.as_allocator_view();
          let staging_arena_ptr = state.frame_staging_arena.read().as_ref().unwrap() as *const _;
          let discard_pool_ptr = &state.discard_pool as *const _;
          let descriptor_pool_arc = state.descriptor_pool.read().as_ref().unwrap().clone();
          let sky_image_clone = state.sky_image.read().as_ref().map(|sky| resources::Image {
            image: sky.image,
            image_view: sky.image_view,
            allocation: sky.allocation,
          });
          let linear_sampler = state.linear_sampler;

          // Return cloned/extracted data needed for execution
          Ok((
            pipeline_key,
            outline_pipeline_key,
            arena_arc,
            position_data,
            attribute_data,
            vma,
            staging_arena_ptr,
            discard_pool_ptr,
            descriptor_pool_arc,
            sky_image_clone,
            linear_sampler,
          ))
        })?
        .execute(
          |(
            pipeline_key,
            outline_pipeline_key,
            arena_arc,
            position_data,
            attribute_data,
            vma,
            staging_arena_ptr,
            discard_pool_ptr,
            descriptor_pool_arc,
            sky_image,
            linear_sampler,
          ),
           rollback| {
            let allocator = vma;
            let staging_arena = unsafe { &*staging_arena_ptr };
            let discard_pool = unsafe { &*discard_pool_ptr };

            let mut resource_opt = None;
            let mut texture_flags_out = TextureFlags::empty();

            let transient_res = self.run_transient_commands(|transient_cmd| {
              let mut texture_flags: TextureFlags = TextureFlags::empty();
              let albedo_image = component.mesh.albedo_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::ALBEDO;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureAlbedo_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let mut alloc_h = img.allocation;
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vma.destroy_image(img_h, &mut alloc_h);
                });
                Some(img)
              });

              let normal_image = component.mesh.normal_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::NORMAL;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureNormal_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let mut alloc_h = img.allocation;
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vma.destroy_image(img_h, &mut alloc_h);
                });
                Some(img)
              });

              let roughness_image = component.mesh.roughness_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::ROUGHNESS;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureRoughness_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let mut alloc_h = img.allocation;
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vma.destroy_image(img_h, &mut alloc_h);
                });
                Some(img)
              });

              let ao_image = component.mesh.ao_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::AO;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureAO_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let mut alloc_h = img.allocation;
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vma.destroy_image(img_h, &mut alloc_h);
                });
                Some(img)
              });

              let (_, descriptor_set) = {
                let layout =
                  crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc)
                    .descriptor_set_layouts
                    .get(0)
                    .ok_or(crate::gpu_invalid_arg!("invalid argument"))?
                    .get();
                descriptor_pool_arc.allocate_and_get_active_pool(
                  &self.device,
                  layout,
                  discard_pool,
                  u64::MAX,
                  debug_name,
                  rollback,
                )?
              };

              let dummy_texture = {
                let arena_r =
                  crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc);
                resources::Image {
                  image: arena_r.dummy_texture_handle.image,
                  image_view: arena_r.dummy_texture_handle.image_view,
                  allocation: arena_r.dummy_texture_handle.allocation,
                }
              };

              let resource = unsafe {
                ForwardMeshRenderResource::new(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &position_data,
                  &attribute_data,
                  &component.mesh.indices,
                  albedo_image,
                  normal_image,
                  roughness_image,
                  ao_image,
                  sky_image.or_else(|| {
                    Some(resources::Image {
                      image: dummy_texture.image,
                      image_view: dummy_texture.image_view,
                      allocation: dummy_texture.allocation,
                    })
                  }),
                  linear_sampler,
                  NonZeroHandle::new_unchecked(descriptor_set),
                  &dummy_texture,
                  debug_name,
                )?
              };

              resource_opt = Some(resource);
              texture_flags_out = texture_flags;
              Ok(())
            })?;
            let timeline = DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;
            discard_pool.discard_type_erased(transient_res, timeline + 2);

            Ok((
              pipeline_key,
              outline_pipeline_key,
              resource_opt.unwrap(),
              texture_flags_out,
            ))
          },
        )
        .commit_read(|state, execute_result| {
          let (pipeline_key, outline_pipeline_key, resource, texture_flags) = execute_result?;

          let old_resource = state
            .physical_mesh_resources
            .insert(physical_mesh_id, resources::ResourceState::Ready(resource));

          if let Some(resources::ResourceState::Ready(mut old)) = old_resource {
            let timeline = state.timeline_manager.get_next_submit_value();
            old.discard(&self.device, &state.discard_pool, timeline);
          }

          Ok(Some(ResourceUploadResult {
            pipeline: pipeline_key,
            outline_pipeline: Some(outline_pipeline_key),
            buffers: physical_mesh_id.into(),
            texture_flags,
            indirect_buffer: None,
            descriptor_index: None,
          }))
        })
    })();

    if execution_result.is_err() && is_winner {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      if let Some(entry) = res_guard.physical_mesh_resources.get(&physical_mesh_id) {
        if matches!(entry.value(), resources::ResourceState::Pending) {
          drop(entry);
          res_guard.physical_mesh_resources.remove(&physical_mesh_id);
        }
      }
    }

    match execution_result {
      Ok(Some(r)) => Ok(r),
      Ok(None) => self.get_physical_mesh_resources(asset_hash, handle),
      Err(e) => Err(e),
    }
  }

  #[named]
  fn get_physical_mesh2_resources(
    &self,
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let (pipeline_key, outline_pipeline_key) = {
      let live_pes = &res_guard.live_presentation_engines;
      let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
      let pe = pe_lock;
      let archetypes = DebugTrackedRwLock::read(&pe.archetypes().physical_mesh2_render_archetype);
      if archetypes.as_ref().is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype_ref = unsafe { archetypes.as_ref().unwrap_unchecked() };
      let pipeline_key = archetype_ref.pipeline_key;
      let outline_pipeline_key = archetype_ref.outline_pipeline_key;
      (pipeline_key, outline_pipeline_key)
    };

    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(asset_hash);

    if let Some(entry) = res_guard.physical_mesh2_resources.get(&physical_mesh_id) {
      if let resources::ResourceState::Ready(resource) = entry.value() {
        return Ok(ResourceUploadResult {
          pipeline: pipeline_key,
          outline_pipeline: Some(outline_pipeline_key),
          buffers: physical_mesh_id.into(),
          texture_flags: resource.frontend_texture_flags(),
          indirect_buffer: None,
          descriptor_index: None,
        });
      }
    }
    Err(GpuError::NotFound)
  }

  #[named]
  fn create_physical_mesh2_resources(
    &self,
    cmd_buffer: CommandBufferHandle,
    asset_hash: u64,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
    debug_name: &str,
  ) -> GpuResult<ResourceUploadResult> {
    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(asset_hash);
    let cmd = self.get_cmd(cmd_buffer)?;

    let mut is_winner = false;
    loop {
      enum Action {
        Return,
        Yield,
        BreakWinner,
      }

      let action = {
        let res_guard = DebugTrackedRwLock::read(&self.res);
        match res_guard.physical_mesh2_resources.entry(physical_mesh_id) {
          dashmap::mapref::entry::Entry::Occupied(e) => match e.get() {
            resources::ResourceState::Ready(_) => Action::Return,
            resources::ResourceState::Pending => Action::Yield,
          },
          dashmap::mapref::entry::Entry::Vacant(e) => {
            e.insert(resources::ResourceState::Pending);
            Action::BreakWinner
          }
        }
        // Rust automatically drops `e` and then `res_guard`
      };

      match action {
        Action::Return => return self.get_physical_mesh2_resources(asset_hash, handle),
        Action::Yield => aethervk_oshal_rlib::os::native::this_thread::sleep_for(
          core::time::Duration::from_millis(1),
        ),
        Action::BreakWinner => {
          is_winner = true;
          break;
        }
      }
    }

    let execution_result = (|| -> GpuResult<Option<ResourceUploadResult>> {
      crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
        .prepare_read(handle, |state, h| {
          let live_pes = &state.live_presentation_engines;
          let pe_lock = wait_for_pe_direct!(live_pes, h)?;
          let pe = pe_lock;
          let archetypes =
            DebugTrackedRwLock::read(&pe.archetypes().physical_mesh2_render_archetype);
          if archetypes.as_ref().is_none() {
            return Err(gpu_err_archetype_absent!());
          }
          let archetype_ref = unsafe { archetypes.as_ref().unwrap_unchecked() };

          let pipeline_key = archetype_ref.pipeline_key;
          let outline_pipeline_key = archetype_ref.outline_pipeline_key;
          let arena_arc = archetype_ref.deref_arena().ok_or(gpu_err!("arena absent"))?.clone();

          let position_data = extract_position_data(component.mesh.as_ref());
          let attribute_data = extract_attribute_data(component.mesh.as_ref());

          let vma = state.allocator.allocator.get_raw();
          let staging_arena_ptr = state.frame_staging_arena.read().as_ref().unwrap() as *const _;
          let discard_pool_ptr = &state.discard_pool as *const _;
          let descriptor_pool_arc = state.descriptor_pool.read().as_ref().unwrap().clone();
          let sky_image_clone = state.sky_image.read().as_ref().map(|sky| resources::Image {
            image: sky.image,
            image_view: sky.image_view,
            allocation: sky.allocation,
          });
          let linear_sampler = state.linear_sampler;

          // Return cloned/extracted data needed for execution
          Ok((
            pipeline_key,
            outline_pipeline_key,
            arena_arc,
            position_data,
            attribute_data,
            vma,
            staging_arena_ptr,
            discard_pool_ptr,
            descriptor_pool_arc,
            sky_image_clone,
            linear_sampler,
          ))
        })?
        .execute(
          |(
            pipeline_key,
            outline_pipeline_key,
            arena_arc,
            position_data,
            attribute_data,
            vma,
            staging_arena_ptr,
            discard_pool_ptr,
            descriptor_pool_arc,
            sky_image,
            linear_sampler,
          ),
           rollback| {
            let allocator = unsafe { vk_mem::AllocatorView::from_raw(vma) };
            let staging_arena = unsafe { &*staging_arena_ptr };
            let discard_pool = unsafe { &*discard_pool_ptr };

            let mut resource_opt = None;
            let mut texture_flags_out = TextureFlags::empty();

            let transient_res = self.run_transient_commands(|transient_cmd| {
              let mut texture_flags: TextureFlags = TextureFlags::empty();
              let albedo_image = component.mesh.albedo_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::ALBEDO;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureAlbedo_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let alloc_h = img.allocation.get_raw();
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
                });
                Some(img)
              });

              let normal_image = component.mesh.normal_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::NORMAL;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureNormal_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let alloc_h = img.allocation.get_raw();
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
                });
                Some(img)
              });

              let roughness_image = component.mesh.roughness_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::ROUGHNESS;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureRoughness_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let alloc_h = img.allocation.get_raw();
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
                });
                Some(img)
              });

              let ao_image = component.mesh.ao_map.as_ref().and_then(|t| {
                texture_flags |= TextureFlags::AO;
                let img = Image::new_2d(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &t,
                  vk::ImageUsageFlags::SAMPLED,
                  &alloc::format!("TextureAO_{}", debug_name),
                )
                .ok()?;
                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let alloc_h = img.allocation.get_raw();
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
                });
                Some(img)
              });

              let material_data = crate::gpu::MaterialData {
                base_albedo: [1.0, 1.0, 1.0, 1.0],
                emissive_color: [
                  component.emissive_color[0],
                  component.emissive_color[1],
                  component.emissive_color[2],
                  component.emissive_intensity,
                ],
                base_ao: 1.0,
                paint_display_mode: component.paint_display_mode,
                texture_flags: texture_flags.bits(),
                _pad0: 0.0,
                sphere_center_radius: [
                  component.sphere_center[0],
                  component.sphere_center[1],
                  component.sphere_center[2],
                  component.sphere_radius,
                ],
                grid_color_density: [
                  component.grid_color[0],
                  component.grid_color[1],
                  component.grid_color[2],
                  component.grid_density,
                ],
              };

              let object_data = crate::gpu::ObjectData {
                #[rustfmt::skip]
                model: [
                  1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0,
                ],
              };

              let (_, descriptor_set) = {
                let layout =
                  crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc)
                    .descriptor_set_layouts
                    .get(0)
                    .ok_or(crate::gpu_invalid_arg!("invalid argument"))?
                    .get();
                descriptor_pool_arc.allocate_and_get_active_pool(
                  &self.device,
                  layout,
                  discard_pool,
                  u64::MAX,
                  debug_name,
                  rollback,
                )?
              };

              let emissive_paint_image = {
                let img = crate::gpu_backends::vulkan::device::resources::Image::new_paint_image(
                  &self.device,
                  allocator,
                  1024,
                  1024,
                  &alloc::format!("EmissivePaint_{}", debug_name),
                )?;

                let image_barrier = ash::vk::ImageMemoryBarrier2::default()
                  .src_stage_mask(ash::vk::PipelineStageFlags2::NONE)
                  .src_access_mask(ash::vk::AccessFlags2::NONE)
                  .dst_stage_mask(ash::vk::PipelineStageFlags2::FRAGMENT_SHADER)
                  .dst_access_mask(ash::vk::AccessFlags2::SHADER_READ)
                  .old_layout(ash::vk::ImageLayout::UNDEFINED)
                  .new_layout(ash::vk::ImageLayout::GENERAL)
                  .image(img.image.get())
                  .subresource_range(
                    ash::vk::ImageSubresourceRange::default()
                      .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                      .base_mip_level(0)
                      .level_count(1)
                      .base_array_layer(0)
                      .layer_count(1),
                  );

                let dep_info = ash::vk::DependencyInfo::default()
                  .image_memory_barriers(core::slice::from_ref(&image_barrier));
                unsafe {
                  self.device.synchronization2.cmd_pipeline_barrier2(transient_cmd, &dep_info);
                }

                let img_h = img.image.get();
                let view_h = img.image_view.get();
                let alloc_h = img.allocation.get_raw();
                rollback.defer(move |dev| unsafe {
                  dev.destroy_image_view(view_h, None);
                  vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
                });
                img
              };

              let dummy_texture = {
                let arena_r =
                  crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc);
                resources::Image {
                  image: arena_r.dummy_texture_handle.image,
                  image_view: arena_r.dummy_texture_handle.image_view,
                  allocation: arena_r.dummy_texture_handle.allocation,
                }
              };

              let resource = unsafe {
                crate::gpu_backends::vulkan::device::resources::ForwardMesh2RenderResource::new(
                  &self.device,
                  allocator,
                  transient_cmd,
                  staging_arena,
                  &position_data,
                  &attribute_data,
                  &component.mesh.indices,
                  &material_data,
                  &object_data,
                  albedo_image,
                  normal_image,
                  roughness_image,
                  ao_image,
                  sky_image.or_else(|| {
                    Some(resources::Image {
                      image: dummy_texture.image,
                      image_view: dummy_texture.image_view,
                      allocation: dummy_texture.allocation,
                    })
                  }),
                  Some(emissive_paint_image),
                  linear_sampler,
                  NonZeroHandle::new_unchecked(descriptor_set),
                  &dummy_texture,
                  debug_name,
                )?
              }; // last op, so no rollback defer

              resource_opt = Some(resource);
              texture_flags_out = texture_flags;
              Ok(())
            })?;
            let timeline = DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;
            discard_pool.discard_type_erased(transient_res, timeline + 2);

            Ok((
              pipeline_key,
              outline_pipeline_key,
              resource_opt.unwrap(),
              texture_flags_out,
            ))
          },
        )
        .commit_read(|state, execute_result| {
          let (pipeline_key, outline_pipeline_key, resource, texture_flags) = execute_result?;

          let old_resource = state
            .physical_mesh2_resources
            .insert(physical_mesh_id, resources::ResourceState::Ready(resource));

          if let Some(resources::ResourceState::Ready(mut old)) = old_resource {
            let timeline = state.timeline_manager.get_next_submit_value();
            old.discard(&self.device, &state.discard_pool, timeline);
          }

          Ok(Some(ResourceUploadResult {
            pipeline: pipeline_key,
            outline_pipeline: Some(outline_pipeline_key),
            buffers: physical_mesh_id.into(),
            texture_flags,
            indirect_buffer: None,
            descriptor_index: None,
          }))
        })
    })();

    if execution_result.is_err() && is_winner {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      if let Some(entry) = res_guard.physical_mesh2_resources.get(&physical_mesh_id) {
        if matches!(entry.value(), resources::ResourceState::Pending) {
          drop(entry);
          res_guard.physical_mesh2_resources.remove(&physical_mesh_id);
        }
      }
    }

    match execution_result {
      Ok(Some(r)) => Ok(r),
      Ok(None) => self.get_physical_mesh2_resources(asset_hash, handle),
      Err(e) => Err(e),
    }
  }

  #[named]
  fn draw_physical_mesh2(
    &self,
    cmd_buffer: CommandBufferHandle,
    _pipeline: PipelineKey,
    buffers: GpuResourceHandle,
    camera: &crate::gpu::frame::CameraRenderData,
    sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    sun_color: [f32; 4],
    window_extent: [f32; 2],
    handle: PresentationEngineHandle,
    draw_call: &crate::gpu::frame::DrawCall,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;
    let physical_mesh_id = RenderableInstanceId(buffers.0);

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        // 1. Fetch Mesh Resource
        let read_resources = &state.physical_mesh2_resources;
        let resource_ref = read_resources
          .get(&physical_mesh_id)
          .ok_or(gpu_err!("Physical mesh 2 resource missing"))?;
        let resource = match resource_ref.value() {
          resources::ResourceState::Ready(r) => r,
          _ => return Err(gpu_err!("Physical mesh 2 resource not ready")),
        };

        let pos_buf = resource.position_vertex_buffer.buffer.get();
        let attr_buf = resource.attributes_vertex_buffer.buffer.get();
        let idx_buf = resource.index_buffer.buffer.get();
        let desc_set = resource.descriptor_set.get();

        // 2. Fetch Pipeline Layout
        let pe = wait_for_pe!(state, handle)?;
        let archetypes = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &pe.archetypes().physical_mesh2_render_archetype,
        );
        let archetype_ref = archetypes.as_ref().ok_or(gpu_err_archetype_absent!())?;
        let mesh_arena = archetype_ref.deref_arena().ok_or(gpu_err!("arena absent"))?;
        let pipeline_layout =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*mesh_arena)
            .pipeline_layout
            .get();

        // 3. Allocate from Staging Arena
        let mut staging_arena_guard =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
            &state.frame_staging_arena,
          );
        let arena = staging_arena_guard.as_mut().ok_or(gpu_err!("staging arena missing"))?;

        let (scene_offset, scene_ptr) = arena
          .allocate(core::mem::size_of::<crate::gpu::SceneData>(), 8)
          .ok_or(GpuError::OutOfMemory)?;

        let (material_offset, material_ptr) = arena
          .allocate(core::mem::size_of::<crate::gpu::MaterialData>(), 8)
          .ok_or(GpuError::OutOfMemory)?;

        let (object_offset, object_ptr) = arena
          .allocate(core::mem::size_of::<crate::gpu::ObjectData>(), 8)
          .ok_or(GpuError::OutOfMemory)?;

        let base_addr = unsafe {
          self
            .device
            .buffer_device_address
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
        };

        Ok::<_, GpuError>((
          pos_buf,
          attr_buf,
          idx_buf,
          desc_set,
          pipeline_layout,
          scene_offset,
          scene_ptr,
          material_offset,
          material_ptr,
          object_offset,
          object_ptr,
          base_addr,
        ))
      })?
      .execute(
        |(
          pos_buf,
          attr_buf,
          idx_buf,
          desc_set,
          pipeline_layout,
          scene_offset,
          scene_ptr,
          material_offset,
          material_ptr,
          object_offset,
          object_ptr,
          base_addr,
        ),
         _rollback| {
          // 4. Construct Data lock-free
          let scene_data = crate::gpu::SceneData {
            view_proj: (camera.view_proj).into(),
            camera_pos: [camera.pos.x(), camera.pos.y(), camera.pos.z(), 0.0],
            sun_pos: [sun_pos.x(), sun_pos.y(), sun_pos.z(), 0.0],
            sun_color,
            window_extent,
            _pad: [0.0, 0.0],
          };

          let material_data = crate::gpu::MaterialData {
            base_albedo: [1.0, 1.0, 1.0, 1.0],
            emissive_color: [
              draw_call.emissive_color[0],
              draw_call.emissive_color[1],
              draw_call.emissive_color[2],
              draw_call.emissive_intensity,
            ],
            base_ao: 1.0,
            paint_display_mode: draw_call.paint_display_mode,
            texture_flags: draw_call.texture_flags.bits(),
            _pad0: 0.0,
            sphere_center_radius: [
              draw_call.sphere_center[0],
              draw_call.sphere_center[1],
              draw_call.sphere_center[2],
              draw_call.sphere_radius,
            ],
            grid_color_density: [
              draw_call.grid_color[0],
              draw_call.grid_color[1],
              draw_call.grid_color[2],
              draw_call.grid_density,
            ],
          };

          let object_data = crate::gpu::ObjectData {
            model: draw_call.model_matrix.into(),
          };

          unsafe {
            core::ptr::copy_nonoverlapping(
              &scene_data as *const _ as *const u8,
              scene_ptr,
              core::mem::size_of::<crate::gpu::SceneData>(),
            );
            core::ptr::copy_nonoverlapping(
              &material_data as *const _ as *const u8,
              material_ptr,
              core::mem::size_of::<crate::gpu::MaterialData>(),
            );
            core::ptr::copy_nonoverlapping(
              &object_data as *const _ as *const u8,
              object_ptr,
              core::mem::size_of::<crate::gpu::ObjectData>(),
            );
          }

          let push = crate::gpu::PhysicalMesh2PushConstants {
            scene_addr: base_addr + scene_offset as u64,
            material_addr: base_addr + material_offset as u64,
            object_addr: base_addr + object_offset as u64,
          };

          // 5. Directly record Vulkan Commands (bypassing abstract generic bounds/locks)
          unsafe {
            self.device.cmd_bind_descriptor_sets(
              cmd,
              vk::PipelineBindPoint::GRAPHICS,
              pipeline_layout,
              0,
              &[desc_set],
              &[],
            );

            let vertex_buffers = [pos_buf, attr_buf];
            let offsets = [0, 0];
            self.device.cmd_bind_vertex_buffers(cmd, 0, &vertex_buffers, &offsets);
            self.device.cmd_bind_index_buffer(cmd, idx_buf, 0, vk::IndexType::UINT32);

            // Bypassing `self.push_constants_mesh2(...)`
            let push_bytes = core::slice::from_raw_parts(
              &push as *const _ as *const u8,
              core::mem::size_of::<crate::gpu::PhysicalMesh2PushConstants>(),
            );
            self.device.cmd_push_constants(
              cmd,
              pipeline_layout,
              vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
              0,
              push_bytes,
            );

            // Bypassing `self.draw_indexed(...)`
            #[cfg(feature = "std")]
            {
              println!("DEBUG DRAW MESH 2: index_count={}", draw_call.index_count);
            }
            self.device.cmd_draw_indexed(cmd, draw_call.index_count, 1, 0, 0, 0);
          }

          Ok(())
        },
      )
      .commit_read(|_state, execute_result| execute_result)?;

    Ok(())
  }

  #[named]
  fn generate_sky(&self) -> GpuResult<()> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        // 1. Quick check: Is sky already generated?
        if crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&state.sky_image)
          .is_some()
        {
          return Ok::<_, GpuError>(None); // Signal to skip execution
        }

        // 2. Safely acquire shader module
        let comp_key = {
          let mut sm = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
            &state.shader_manager,
          );
          ensure_skygen_shader_module(&self.device, &mut sm)?
        };

        let shader_module = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &state.shader_manager,
        )
        .get(comp_key)
        .ok_or(GpuError::InvalidShader)?
        .module
        .get();

        // Extract raw pointers to bypass borrow checker in the execution block
        let vma = state.allocator.allocator.get_raw();
        let pipeline_pool_ptr = &state.pipeline_pool as *const _;

        Ok(Some((shader_module, vma, pipeline_pool_ptr)))
      })?
      .execute(|prep, rollback| {
        let Some((shader_module, vma, pipeline_pool_ptr)) = prep else {
          return Ok(None);
        };

        let allocator = unsafe { vk_mem::AllocatorView::from_raw(vma) };
        let pipeline_pool = unsafe { &*pipeline_pool_ptr };

        let graphics_queue = self.queues.get_graphics_queue();
        let compute_queue = self.queues.get_compute_queue();

        // --- 1. Persistent Resource (Sky Image) ---
        let sky_image = resources::Image::new_storage_2d(
          &self.device,
          allocator,
          2048,
          2048,
          vk::Format::R16G16B16A16_SFLOAT,
          graphics_queue.family_index,
          compute_queue.family_index,
          "Sky",
        )?;

        let img_h = sky_image.image.get();
        let view_h = sky_image.image_view.get();
        let alloc_h = sky_image.allocation.get_raw();

        // Defer destruction in case the transaction aborts down the line
        rollback.defer(move |dev| unsafe {
          dev.destroy_image_view(view_h, None);
          vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
        });

        // --- 2. Transient Setup ---
        let bindings = [vk::DescriptorSetLayoutBinding::default()
          .binding(0)
          .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
          .descriptor_count(1)
          .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let set_layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

        let set_layouts = [set_layout];
        let pipeline_layout_info =
          vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout =
          unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None) }?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
          .ty(vk::DescriptorType::STORAGE_IMAGE)
          .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);
        let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

        let command_pool_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(graphics_queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }?;
        let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }?;

        // RAII Cleanup Guard for transient resources.
        // Ensures they are always destroyed when the block ends (success or error).
        struct TransientCleanup {
          device: ash::Device,
          resources: Option<TransientCleanupResources>,
        }
        struct TransientCleanupResources {
          set_layout: vk::DescriptorSetLayout,
          pipeline_layout: vk::PipelineLayout,
          descriptor_pool: vk::DescriptorPool,
          command_pool: vk::CommandPool,
          fence: vk::Fence,
        }
        impl DeviceResource for TransientCleanupResources {
          fn cleanup(&mut self, device: &ash::Device) {
            unsafe {
              device.destroy_command_pool(self.command_pool, None);
              device.destroy_descriptor_pool(self.descriptor_pool, None);
              device.destroy_pipeline_layout(self.pipeline_layout, None);
              device.destroy_descriptor_set_layout(self.set_layout, None);
              // We do not destroy the fence here anymore as it's not a fence.
            }
          }
        }
        impl Drop for TransientCleanup {
          fn drop(&mut self) {
            if let Some(mut res) = self.resources.take() {
              res.cleanup(&self.device);
            }
          }
        }

        let mut _cleanup = TransientCleanup {
          device: self.device.clone(),
          resources: Some(TransientCleanupResources {
            set_layout,
            pipeline_layout,
            descriptor_pool,
            command_pool,
            fence,
          }),
        };

        // --- 3. Write Descriptors ---
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
          .descriptor_pool(descriptor_pool)
          .set_layouts(&set_layouts);
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

        let image_info = vk::DescriptorImageInfo::default()
          .image_layout(vk::ImageLayout::GENERAL)
          .image_view(view_h);
        let write_descriptor_set = vk::WriteDescriptorSet::default()
          .dst_set(descriptor_set)
          .dst_binding(0)
          .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
          .image_info(core::slice::from_ref(&image_info));
        unsafe { self.device.update_descriptor_sets(&[write_descriptor_set], &[]) };

        // --- 4. Pipeline & Specialization ---
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
        compute_info.add_specialization_constant_u32(
          vk::SpecializationMapEntry {
            constant_id: 10,
            offset: 8,
            size: 4,
          },
          if self.query_result.debug_shaders && cfg!(debug_assertions) {
            1
          } else {
            0
          },
        );

        let compute_pipeline = pipelines::PipelinePool::get_or_create_compute_pipeline(
          pipeline_pool,
          &self.device,
          &compute_info,
          rollback, // RollbackContext injected!
        )?;

        // --- 5. Command Recording ---
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        let command_buffer =
          unsafe { self.device.allocate_command_buffers(&command_buffer_info) }?[0];

        let begin_info =
          vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
          self.device.begin_command_buffer(command_buffer, &begin_info)?;

          let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img_h)
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
          self.device.synchronization2.cmd_pipeline_barrier2(command_buffer, &dep_info);

          self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute_pipeline.get(),
          );
          self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
          );
          self.device.cmd_dispatch(command_buffer, 2048 / 16, 2048 / 16, 1);

          let barrier2 = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img_h)
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
          self.device.synchronization2.cmd_pipeline_barrier2(command_buffer, &dep_info2);

          self.device.end_command_buffer(command_buffer)?;
        }        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
          .semaphore_type(vk::SemaphoreType::TIMELINE)
          .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_info, None) }?;
        
        let signal_semaphores = [timeline_semaphore];
        let signal_values = [1];
        let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
          .signal_semaphore_values(&signal_values);

        let submit_info = vk::SubmitInfo::default()
          .command_buffers(core::slice::from_ref(&command_buffer))
          .signal_semaphores(&signal_semaphores)
          .push_next(&mut timeline_info);
          
        oshal::log!("generate_sky: submitting to graphics queue...");
        self
          .device
          .locked_queue_submit(graphics_queue.handle, &[submit_info], vk::Fence::null())
          .map_err(GpuError::from)?;

        oshal::log!("generate_sky: waiting for timeline semaphore...");
        self.device.wait_for_semaphore_value(timeline_semaphore, 1, u64::MAX)?;
        
        unsafe {
          self.device.destroy_semaphore(timeline_semaphore, None);
        }
        oshal::log!("generate_sky: done waiting");

        unsafe {
          self.device.destroy_fence(fence, None);
        }

        // VVL Bug Fix: The validation layer associates image layout transitions with the command pool they were allocated from.
        // If the pool is destroyed *at any point* during the app's lifetime, VVL loses the layout state for the image and reverts to UNDEFINED.
        // We defer the pool's destruction to device shutdown (u64::MAX) to prevent VVL false positives on future frames.
        Ok(Some((sky_image, _cleanup.resources.take())))
      })
      .commit_read(|state, execute_result| {
        let Some((sky_image, cleanup_res)) = execute_result? else {
          return Ok(()); // Skipped execution (already generated)
        };

        if let Some(res) = cleanup_res {
          let timeline = state.get_timeline_semaphore_cached_value() + 2;
          aethervk_oshal_rlib::log!("generate_sky queuing cleanup at timeline {}", timeline);
          state.discard_pool.discard_type_erased(res, timeline);
        }

        // Apply new sky image and safely clean up the previous one if it exists
        let mut wsky_image =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&state.sky_image);

        if let Some(old) = wsky_image.take() {
          unsafe {
            vk_mem::ffi::vmaDestroyImage(
              state.allocator.allocator.get_raw(),
              old.image.get(),
              old.allocation.get_raw(),
            );
            self.device.destroy_image_view(old.image_view.get(), None);
          }
        }

        *wsky_image = Some(sky_image);

        Ok(())
      })?;

    Ok(())
  }

  #[named]
  fn get_billboard_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    // ensure that the archetype for billboards exists
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let archetype = DebugTrackedRwLock::read(&pe.archetypes().billboard_render_archetype);
    if archetype.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }
    let archetype_ref = unsafe { archetype.as_ref().unwrap_unchecked() };
    let pipeline_key = archetype_ref.pipeline_key;

    // the billboard doesn't have descriptor sets or vertex/index buffers
    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
      indirect_buffer: None,
      descriptor_index: None,
    })
  }

  #[named]
  fn update_gizmo_instance(
    &self,
    entity: EntityId,
    model: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    handle: PresentationEngineHandle,
  ) -> GpuResult<u32> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, _h| {
        let pe = wait_for_pe!(state, handle)?;

        let archetype_guard = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &pe.archetypes().gizmo_render_archetype,
        );
        let archetype = archetype_guard.as_ref().ok_or(gpu_err_archetype_absent!())?;

        let arena_arc = archetype.deref_arena().ok_or(crate::gpu_err!("arena absent"))?.clone();

        let descriptor_set =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc)
            .descriptor_set
            .get();

        let vma_view = state.allocator.allocator.as_allocator_view();
        let timeline = state.timeline_manager.get_cached_value() + 1;

        Ok::<_, GpuError>((arena_arc, descriptor_set, vma_view, timeline))
      })?
      .execute(
        |(arena_arc, descriptor_set, vma_view, timeline), rollback| {
          // Calculate hash index lock-free
          let mut hasher = aethervk_oshal_rlib::hash::FnvHasher::new();
          core::hash::Hash::hash(&entity, &mut hasher);
          let entity_hash = core::hash::Hasher::finish(&hasher);
          let buffer_index = (entity_hash
            % resources::GizmoRenderResourceArchetypeArena::MAX_BUFFER_COUNT as u64)
            as u32;

          let data: [[f32; 16]; 1] = [model.into()];
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
          crate::apply_test_dedicated_alloc!(alloc_info);

          // Lock-free allocation
          let (vk_buf, alloc) = unsafe { vma_view.create_buffer(&buffer_info, &alloc_info)? };

          // Defer destruction in case the transaction aborts
          let mut alloc_mut = alloc;
          rollback.defer(move |_| unsafe {
            vma_view.destroy_buffer(vk_buf, &mut alloc_mut);
          });

          // Map and copy matrix data
          let alloc_info_res = vma_view.get_allocation_info(&alloc);
          unsafe {
            core::ptr::copy_nonoverlapping(
              data.as_ptr() as *const u8,
              alloc_info_res.mapped_data as *mut u8,
              buffer_size as usize,
            );
          }

          // Lock-free descriptor set update
          let buffer_info_vk = vk::DescriptorBufferInfo::default()
            .buffer(vk_buf)
            .offset(0)
            .range(vk::WHOLE_SIZE);

          let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(buffer_index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(core::slice::from_ref(&buffer_info_vk));

          unsafe {
            self.device.update_descriptor_sets(core::slice::from_ref(&write), &[]);
          }

          Ok((arena_arc, buffer_index, vk_buf, alloc, timeline))
        },
      )
      .commit_read(|state, execute_result| {
        let (arena_arc, buffer_index, vk_buf, alloc, timeline) = execute_result?;

        let new_buffer = resources::Buffer {
          buffer: unsafe { NonZeroHandle::new_unchecked(vk_buf) },
          allocation: alloc,
        };

        // Briefly lock to insert into the circular array
        let arena =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc);
        let mut buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &arena.host_buffers,
        );

        if let Some(old_buffer) = buffers.insert(buffer_index, new_buffer) {
          state.discard_pool.discard_buffer(
            state.allocator.allocator.get_raw(),
            old_buffer.buffer.get(),
            old_buffer.allocation,
            timeline,
          );
        }

        Ok(buffer_index)
      })
  }

  #[named]
  fn upload_particle_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::ParticleDrawCall],
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().particle_render_archetype);
    if archetype_guard.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }
    let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

    let mut staging_arena_guard = DebugTrackedRwLock::write(&res_guard.frame_staging_arena);
    let staging_arena = staging_arena_guard.as_mut().unwrap();

    let particle_size = core::mem::size_of::<crate::scene::particles::ParticleData>();
    let indirect_size = core::mem::size_of::<vk::DrawIndirectCommand>();

    let mut current_particle_offset = 0;
    let mut current_indirect_offset = 0;

    for call in particle_calls.iter_mut() {
      let particles_arc = call.particles.upgrade();
      if particles_arc.is_none() {
        continue;
      }
      let particles_arc = unsafe { particles_arc.unwrap_unchecked() };
      let particles = particles_arc.read();
      if particles.is_empty() {
        continue;
      }

      call.system_particle_offset = current_particle_offset as u32;
      call.system_indirect_offset = current_indirect_offset as u32;

      let particle_data_size = particles.len() * particle_size;
      let p_dst_offset = (current_particle_offset as usize * particle_size) as vk::DeviceSize;

      let i_dst_offset = (current_indirect_offset as usize * indirect_size) as vk::DeviceSize;

      let total_size = particle_data_size + indirect_size;
      let (staging_offset, ptr) =
        staging_arena.allocate(total_size, 16).ok_or(crate::gpu_err_device!())?;

      unsafe {
        core::ptr::copy_nonoverlapping(particles.as_ptr() as *const u8, ptr, particle_data_size);

        let indirect_cmd = vk::DrawIndirectCommand {
          vertex_count: 4,
          instance_count: particles.len() as u32,
          first_vertex: 0,
          first_instance: current_particle_offset as u32,
        };
        core::ptr::copy_nonoverlapping(
          &indirect_cmd as *const _ as *const u8,
          ptr.add(particle_data_size),
          indirect_size,
        );
      }

      let p_copy = vk::BufferCopy::default()
        .src_offset(staging_offset as u64)
        .dst_offset(p_dst_offset)
        .size(particle_data_size as u64);
      let i_copy = vk::BufferCopy::default()
        .src_offset((staging_offset + particle_data_size) as u64)
        .dst_offset(i_dst_offset)
        .size(indirect_size as u64);

      let (mega_particle_buffer, mega_indirect_buffer) = {
        let arena_arc = archetype.deref_arena().ok_or(crate::gpu_err_device!())?;
        let arena = DebugTrackedRwLock::read(&*arena_arc);
        (arena.mega_particle_buffer, arena.mega_indirect_buffer)
      };

      unsafe {
        self.device.cmd_copy_buffer(
          cmd,
          staging_arena.buffer,
          mega_particle_buffer,
          core::slice::from_ref(&p_copy),
        );
        self.device.cmd_copy_buffer(
          cmd,
          staging_arena.buffer,
          mega_indirect_buffer,
          core::slice::from_ref(&i_copy),
        );
      }

      let mut p_barrier = vk::BufferMemoryBarrier::default()
        .buffer(mega_particle_buffer)
        .offset(p_dst_offset)
        .size(particle_data_size as u64);
      let mut i_barrier = vk::BufferMemoryBarrier::default()
        .buffer(mega_indirect_buffer)
        .offset(i_dst_offset)
        .size(indirect_size as u64);

      p_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
      p_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
      i_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
      i_barrier.dst_access_mask = vk::AccessFlags::INDIRECT_COMMAND_READ;
      let src_stage = vk::PipelineStageFlags::TRANSFER;
      let dst_stage = vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT;

      unsafe {
        self.device.cmd_pipeline_barrier(
          cmd,
          src_stage,
          dst_stage,
          vk::DependencyFlags::empty(),
          &[],
          &[p_barrier, i_barrier],
          &[],
        );
      }

      current_particle_offset += particles.len();
      current_indirect_offset += 1;
    }

    Ok(())
  }

  #[named]
  fn upload_particle2_systems(
    &self,
    cmd_buffer: CommandBufferHandle,
    particle_calls: &mut [crate::gpu::frame::Particle2DrawCall],
  ) -> GpuResult<()> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().particle2_render_archetype);
    if archetype_guard.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }
    let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

    let mut staging_arena_guard = DebugTrackedRwLock::write(&res_guard.frame_staging_arena);
    let staging_arena = staging_arena_guard.as_mut().unwrap();

    let particle_size = core::mem::size_of::<crate::scene::particles::ParticleData>();
    let indirect_size = core::mem::size_of::<vk::DrawIndirectCommand>();

    let mut current_particle_offset = 0;
    let mut current_indirect_offset = 0;

    for call in particle_calls.iter_mut() {
      let particles_arc = call.particles.upgrade();
      if particles_arc.is_none() {
        continue;
      }
      let particles_arc = unsafe { particles_arc.unwrap_unchecked() };
      let particles = particles_arc.read();
      if particles.is_empty() {
        continue;
      }

      call.system_particle_offset = current_particle_offset as u32;
      call.system_indirect_offset = current_indirect_offset as u32;

      let particle_data_size = particles.len() * particle_size;
      let p_dst_offset = (current_particle_offset as usize * particle_size) as vk::DeviceSize;

      let i_dst_offset = (current_indirect_offset as usize * indirect_size) as vk::DeviceSize;

      let total_size = particle_data_size + indirect_size;
      let (staging_offset, ptr) =
        staging_arena.allocate(total_size, 16).ok_or(crate::gpu_err_device!())?;

      unsafe {
        core::ptr::copy_nonoverlapping(particles.as_ptr() as *const u8, ptr, particle_data_size);

        let indirect_cmd = vk::DrawIndirectCommand {
          vertex_count: 4,
          instance_count: particles.len() as u32,
          first_vertex: 0,
          first_instance: current_particle_offset as u32,
        };
        core::ptr::copy_nonoverlapping(
          &indirect_cmd as *const _ as *const u8,
          ptr.add(particle_data_size),
          indirect_size,
        );
      }

      let p_copy = vk::BufferCopy::default()
        .src_offset(staging_offset as u64)
        .dst_offset(p_dst_offset)
        .size(particle_data_size as u64);
      let i_copy = vk::BufferCopy::default()
        .src_offset((staging_offset + particle_data_size) as u64)
        .dst_offset(i_dst_offset)
        .size(indirect_size as u64);

      let (mega_particle_buffer, mega_indirect_buffer) = {
        let arena_arc = archetype.deref_arena().ok_or(crate::gpu_err_device!())?;
        let arena = DebugTrackedRwLock::read(&*arena_arc);
        (arena.mega_particle_buffer, arena.mega_indirect_buffer)
      };

      unsafe {
        self.device.cmd_copy_buffer(
          cmd,
          staging_arena.buffer,
          mega_particle_buffer,
          core::slice::from_ref(&p_copy),
        );
        self.device.cmd_copy_buffer(
          cmd,
          staging_arena.buffer,
          mega_indirect_buffer,
          core::slice::from_ref(&i_copy),
        );
      }

      let mut p_barrier = vk::BufferMemoryBarrier::default()
        .buffer(mega_particle_buffer)
        .offset(p_dst_offset)
        .size(particle_data_size as u64);
      let mut i_barrier = vk::BufferMemoryBarrier::default()
        .buffer(mega_indirect_buffer)
        .offset(i_dst_offset)
        .size(indirect_size as u64);

      p_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
      p_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
      i_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
      i_barrier.dst_access_mask = vk::AccessFlags::INDIRECT_COMMAND_READ;
      let src_stage = vk::PipelineStageFlags::TRANSFER;
      let dst_stage = vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT;

      unsafe {
        self.device.cmd_pipeline_barrier(
          cmd,
          src_stage,
          dst_stage,
          vk::DependencyFlags::empty(),
          &[],
          &[p_barrier, i_barrier],
          &[],
        );
      }

      current_particle_offset += particles.len();
      current_indirect_offset += 1;
    }

    Ok(())
  }

  #[named]
  fn upload_trajectories(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    trajectories: &[(
      crate::scene::EntityId,
      crate::scene::trajectory::TrajectoryComponent,
      aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    )],
  ) -> GpuResult<Option<crate::gpu::frame::TrajectoryBatchCall>> {
    if trajectories.is_empty() {
      return Ok(None);
    }

    #[repr(C, align(16))]
    #[derive(Copy, Clone)]
    struct RationalBezierGpu {
      cp0: [f32; 4],
      cp1: [f32; 4],
      cp2: [f32; 4],
      cp3: [f32; 4],
    }
    #[repr(C, align(16))]
    #[derive(Copy, Clone)]
    struct TrajectoryGpu {
      segments_ptr: u64,
      _pad0: u64,
      color: [f32; 4],
      line_width: f32,
      texture_id: u32,
      _pad1: u64,
    }
    #[repr(C, align(4))]
    #[derive(Copy, Clone)]
    struct SegmentMapGpu {
      trajectory_id: u32,
      local_segment_id: u32,
      subdivisions: u32,
    }

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let mut archetype_guard =
      DebugTrackedRwLock::write(&pe.archetypes_mut().trajectory_render_archetype);
    if archetype_guard.as_ref().is_none() {
      return Err(gpu_err_archetype_absent!());
    }
    let archetype = unsafe { archetype_guard.as_mut().unwrap_unchecked() };

    let arena_arc = archetype.deref_arena().ok_or(gpu_err!("arena absent"))?;
    let mut arena_mut = DebugTrackedRwLock::write(&*arena_arc);

    arena_mut.tick = arena_mut.tick.wrapping_add(1);
    let current_tick = arena_mut.tick;

    // 1. GARBAGE COLLECTION: Purge curves missing for > 10 frames
    let mut to_remove = alloc::vec::Vec::new();
    for (id, alloc) in arena_mut.curves.iter() {
      if current_tick.saturating_sub(alloc.last_seen_tick) > 10 {
        to_remove.push(*id);
      }
    }

    for id in to_remove {
      if let Some(alloc) = arena_mut.curves.remove(&id) {
        arena_mut
          .segment_allocator
          .free(alloc.segments_offset, alloc.segment_capacity as u64);
      }
    }

    let mut segment_maps = alloc::vec::Vec::new();
    let mut traj_gpus = alloc::vec::Vec::new();

    let mut all_segments_to_upload = alloc::vec::Vec::new();
    let mut segment_copies = alloc::vec::Vec::new(); // Tracks sparse uploads

    let mut total_segments = 0;
    let mut max_subdivs = 0;

    for (i, (entity_id, traj_comp, model_mat)) in trajectories.iter().enumerate() {
      let local_segments_count = traj_comp.control_points.len() / 4;
      if local_segments_count == 0 {
        continue;
      }

      let current_hash = crate::gpu_backends::vulkan::device::resources::hash_trajectory(
        &traj_comp.control_points,
        model_mat,
      );
      let mut needs_upload = false;
      let mut offset = 0;
      let mut do_free: Option<(u64, u64)> = None;
      let mut do_update = false;

      // 2. RECOGNIZE & ALLOCATE
      if let Some(alloc) = arena_mut.curves.get_mut(&entity_id) {
        alloc.last_seen_tick = current_tick;

        if alloc.segment_capacity < local_segments_count {
          do_free = Some((alloc.segments_offset, alloc.segment_capacity as u64));
          do_update = true;
          // Curve grew -> Free old and allocate bigger block (+ 50% padding to prevent endless reallocation stutter)
          //
          // moved reallocation in do_free, do_update block, so we don't mutable borrow more than once

          needs_upload = true;
        } else {
          offset = alloc.segments_offset;
          if alloc.last_hash != current_hash {
            alloc.last_hash = current_hash;
            needs_upload = true; // Control points physically moved via Animation System
          }
        }
      } else {
        let new_cap = local_segments_count + local_segments_count / 2;

        offset = arena_mut.segment_allocator.allocate(new_cap as u64).unwrap_or(0);
        arena_mut.curves.insert(
          *entity_id,
          crate::gpu_backends::vulkan::device::resources::CurveAllocation {
            segments_offset: offset,
            segment_capacity: new_cap,
            last_seen_tick: current_tick,
            last_hash: current_hash,
          },
        );
        needs_upload = true;
      }

      if let Some((segments_offset, segment_capacity)) = do_free {
        arena_mut.segment_allocator.free(segments_offset, segment_capacity);
        let new_cap = local_segments_count + local_segments_count / 2;
        offset = arena_mut.segment_allocator.allocate(new_cap as u64).unwrap_or(0);
      }

      if do_update {
        if let Some(alloc) = arena_mut.curves.get_mut(&entity_id) {
          let new_cap = local_segments_count + local_segments_count / 2;

          alloc.segments_offset = offset;
          alloc.segment_capacity = new_cap;
          alloc.last_hash = current_hash;
        }
      }

      // 3. STAGE GEOMETRY (Only runs when genuinely modified!)
      if needs_upload {
        use aethervk_oshal_rlib::math::{
          matrix::MatrixVectorMul,
          vector::{Vector4, vec4::Vec4f32},
        };

        let start_idx = all_segments_to_upload.len();
        for j in 0..local_segments_count {
          let pt0 = model_mat
            .mul_vector(Vec4f32::from_components(
              traj_comp.control_points[j * 4][0],
              traj_comp.control_points[j * 4][1],
              traj_comp.control_points[j * 4][2],
              traj_comp.control_points[j * 4][3],
            ))
            .into();
          let pt1 = model_mat
            .mul_vector(Vec4f32::from_components(
              traj_comp.control_points[j * 4 + 1][0],
              traj_comp.control_points[j * 4 + 1][1],
              traj_comp.control_points[j * 4 + 1][2],
              traj_comp.control_points[j * 4 + 1][3],
            ))
            .into();
          let pt2 = model_mat
            .mul_vector(Vec4f32::from_components(
              traj_comp.control_points[j * 4 + 2][0],
              traj_comp.control_points[j * 4 + 2][1],
              traj_comp.control_points[j * 4 + 2][2],
              traj_comp.control_points[j * 4 + 2][3],
            ))
            .into();
          let pt3 = model_mat
            .mul_vector(Vec4f32::from_components(
              traj_comp.control_points[j * 4 + 3][0],
              traj_comp.control_points[j * 4 + 3][1],
              traj_comp.control_points[j * 4 + 3][2],
              traj_comp.control_points[j * 4 + 3][3],
            ))
            .into();
          all_segments_to_upload.push(RationalBezierGpu {
            cp0: pt0,
            cp1: pt1,
            cp2: pt2,
            cp3: pt3,
          });
        }

        segment_copies.push((
          start_idx as u64 * core::mem::size_of::<RationalBezierGpu>() as u64, // src offset in staging
          offset * core::mem::size_of::<RationalBezierGpu>() as u64, // dest offset in Device Local Memory
          (local_segments_count * core::mem::size_of::<RationalBezierGpu>()) as u64,
        ));
      }

      // 4. METADATA (Small arrays densely rebuilt per frame for flawless sequential instanced rendering)
      traj_gpus.push(TrajectoryGpu {
        segments_ptr: arena_mut.segments_ptr
          + (offset * core::mem::size_of::<RationalBezierGpu>() as u64),
        _pad0: 0,
        color: traj_comp.color,
        line_width: traj_comp.line_width,
        texture_id: traj_comp.texture_id,
        _pad1: 0,
      });

      for j in 0..local_segments_count {
        segment_maps.push(SegmentMapGpu {
          trajectory_id: i as u32,
          local_segment_id: j as u32,
          subdivisions: traj_comp.subdivisions_per_segment,
        });
      }

      if traj_comp.subdivisions_per_segment > max_subdivs {
        max_subdivs = traj_comp.subdivisions_per_segment;
      }
      total_segments += local_segments_count as u32;
    }

    drop(arena_mut);

    if total_segments == 0 {
      return Ok(None);
    }

    let segments_upload_size =
      (all_segments_to_upload.len() * core::mem::size_of::<RationalBezierGpu>()) as vk::DeviceSize;
    let traj_size = (traj_gpus.len() * core::mem::size_of::<TrajectoryGpu>()) as vk::DeviceSize;
    let map_size = (segment_maps.len() * core::mem::size_of::<SegmentMapGpu>()) as vk::DeviceSize;
    let total_staging_size = segments_upload_size + traj_size + map_size;

    // 5. COPY OPERATIONS (Only copying dirtied buffers)
    if total_staging_size > 0 {
      let (staging_offset, staging_ptr) = DebugTrackedRwLock::write(&res_guard.frame_staging_arena)
        .as_mut()
        .unwrap()
        .allocate(total_staging_size as usize, 16)
        .ok_or(crate::gpu_err_device!())?;

      unsafe {
        if segments_upload_size > 0 {
          core::ptr::copy_nonoverlapping(
            all_segments_to_upload.as_ptr() as *const u8,
            staging_ptr,
            segments_upload_size as usize,
          );
        }
        if traj_size > 0 {
          core::ptr::copy_nonoverlapping(
            traj_gpus.as_ptr() as *const u8,
            staging_ptr.add(segments_upload_size as usize),
            traj_size as usize,
          );
        }
        if map_size > 0 {
          core::ptr::copy_nonoverlapping(
            segment_maps.as_ptr() as *const u8,
            staging_ptr.add((segments_upload_size + traj_size) as usize),
            map_size as usize,
          );
        }
      }

      let staging_buffer = DebugTrackedRwLock::write(&res_guard.frame_staging_arena)
        .as_mut()
        .unwrap()
        .buffer;

      let mut vk_buffer_copies = alloc::vec::Vec::new();
      for (src_offset, dst_offset, size) in segment_copies {
        vk_buffer_copies.push(
          vk::BufferCopy::default()
            .src_offset(staging_offset as vk::DeviceSize + src_offset as vk::DeviceSize)
            .dst_offset(dst_offset)
            .size(size),
        );
      }

      let traj_copy = vk::BufferCopy::default()
        .src_offset(staging_offset as vk::DeviceSize + segments_upload_size)
        .dst_offset(0)
        .size(traj_size);
      let map_copy = vk::BufferCopy::default()
        .src_offset(staging_offset as vk::DeviceSize + segments_upload_size + traj_size)
        .dst_offset(0)
        .size(map_size);

      unsafe {
        if !vk_buffer_copies.is_empty() {
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
              &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
            )
            .segments_buffer
            .get(),
            &vk_buffer_copies,
          );
        }
        if traj_size > 0 {
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
              &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
            )
            .trajectories_buffer
            .get(),
            &[traj_copy],
          );
        }
        if map_size > 0 {
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
              &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
            )
            .map_buffer
            .get(),
            &[map_copy],
          );
        }

        let memory_barrier = vk::MemoryBarrier2::default()
          .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
          .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
          .dst_stage_mask(
            vk::PipelineStageFlags2::VERTEX_SHADER
              | vk::PipelineStageFlags2::FRAGMENT_SHADER
              | vk::PipelineStageFlags2::COMPUTE_SHADER,
          )
          .dst_access_mask(vk::AccessFlags2::SHADER_READ);

        let dependency_info =
          vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&memory_barrier));
        self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dependency_info);
      }
    }

    let pipeline = archetype.pipeline_key;

    Ok(Some(crate::gpu::frame::TrajectoryBatchCall {
      pipeline,
      total_vertices: (max_subdivs + 1) * 2,
      total_segments,
      map_ptr: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .map_ptr,
      traj_ptr: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .trajectories_ptr,
    }))
  }

  #[named]
  fn upload_ui(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    ui_elements: &[crate::gpu::UiElementGpu],
  ) -> GpuResult<Option<crate::gpu::UiBatchCall>> {
    if ui_elements.is_empty() {
      return Ok(None);
    }

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let mut archetype_guard = DebugTrackedRwLock::write(&pe.archetypes_mut().ui_render_archetype);
    let archetype = archetype_guard.as_mut().ok_or(gpu_err_archetype_absent!())?;

    let elements_ptr = unsafe {
      let data_ptr = res_guard.allocator.allocator.map_memory(
        &mut crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .elements_alloc,
      )?;

      core::ptr::copy_nonoverlapping(
        ui_elements.as_ptr() as *const u8,
        data_ptr as *mut u8,
        ui_elements.len() * core::mem::size_of::<crate::gpu::UiElementGpu>(),
      );

      let _ = res_guard.allocator.allocator.flush_allocation(
        &crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .elements_alloc,
        0,
        vk::WHOLE_SIZE as u64,
      );

      res_guard.allocator.allocator.unmap_memory(
        &mut crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .elements_alloc,
      );

      let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::HOST)
        .src_access_mask(vk::AccessFlags2::HOST_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT | vk::PipelineStageFlags2::VERTEX_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
            &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
          )
          .elements_buffer
          .get(),
        )
        .offset(0)
        .size(vk::WHOLE_SIZE);

      let dep_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .elements_ptr
    };

    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
      &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
    )
    .tick += 1;

    Ok(Some(crate::gpu::UiBatchCall {
      elements_ptr,
      total_elements: ui_elements.len() as u32,
    }))
  }

  #[named]
  fn upload_text2(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    glyphs: &[crate::gpu::TextGlyphGpu],
  ) -> GpuResult<Option<crate::gpu::Text2BatchCall>> {
    if glyphs.is_empty() {
      return Ok(None);
    }

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let mut archetype_guard =
      DebugTrackedRwLock::write(&pe.archetypes_mut().text2_render_archetype);
    let archetype = archetype_guard.as_mut().ok_or(gpu_err_archetype_absent!())?;

    let glyphs_ptr = unsafe {
      let data_ptr = res_guard.allocator.allocator.map_memory(
        &mut crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .glyphs_alloc,
      )?;

      core::ptr::copy_nonoverlapping(
        glyphs.as_ptr() as *const u8,
        data_ptr as *mut u8,
        glyphs.len() * core::mem::size_of::<crate::gpu::TextGlyphGpu>(),
      );

      let _ = res_guard.allocator.allocator.flush_allocation(
        &crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .glyphs_alloc,
        0,
        vk::WHOLE_SIZE as u64,
      );

      res_guard.allocator.allocator.unmap_memory(
        &mut crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .glyphs_alloc,
      );

      let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::HOST)
        .src_access_mask(vk::AccessFlags2::HOST_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
            | vk::PipelineStageFlags2::VERTEX_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
            &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
          )
          .glyphs_buffer
          .get(),
        )
        .offset(0)
        .size(vk::WHOLE_SIZE);

      let dep_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .glyphs_ptr
    };

    Ok(Some(crate::gpu::Text2BatchCall {
      glyphs_ptr,
      total_glyphs: glyphs.len() as u32,
    }))
  }

  #[named]
  fn draw_particle_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let mut archetype_guard = DebugTrackedRwLock::write(&pe.archetypes().particle_render_archetype);
    let archetype = archetype_guard.as_mut().ok_or(gpu_err_archetype_absent!())?;

    let i_offset = (indirect_offset as usize * core::mem::size_of::<vk::DrawIndirectCommand>())
      as vk::DeviceSize;
    unsafe {
      self.device.cmd_draw_indirect(
        cmd,
        DebugTrackedRwLock::read(&*archetype.deref_arena().ok_or(crate::gpu_err_device!())?)
          .mega_indirect_buffer,
        i_offset,
        1,
        core::mem::size_of::<vk::DrawIndirectCommand>() as u32,
      );
    }
    Ok(())
  }

  #[named]
  fn draw_particle2_indirect(
    &self,
    cmd_buffer: CommandBufferHandle,
    indirect_offset: u32,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let mut archetype_guard =
      DebugTrackedRwLock::write(&pe.archetypes().particle2_render_archetype);
    let archetype = archetype_guard.as_mut().ok_or(gpu_err_archetype_absent!())?;

    let i_offset = (indirect_offset as usize * core::mem::size_of::<vk::DrawIndirectCommand>())
      as vk::DeviceSize;
    unsafe {
      self.device.cmd_draw_indirect(
        cmd,
        DebugTrackedRwLock::read(&*archetype.deref_arena().ok_or(crate::gpu_err_device!())?)
          .mega_indirect_buffer,
        i_offset,
        1,
        core::mem::size_of::<vk::DrawIndirectCommand>() as u32,
      );
    }
    Ok(())
  }

  #[named]
  fn get_particle_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
    let archetype = DebugTrackedRwLock::read(&pe.archetypes().particle_render_archetype);
    archetype.as_ref().map(|a| a.pipeline_key).ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_particle2_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
    let archetype = DebugTrackedRwLock::read(&pe.archetypes().particle2_render_archetype);
    archetype.as_ref().map(|a| a.pipeline_key).ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_trajectory_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = live_pes.get(&handle).ok_or(GpuError::NotFound)?;
    let pe = pe_lock;
    let archetype = DebugTrackedRwLock::read(&pe.archetypes().trajectory_render_archetype);
    archetype.as_ref().map(|a| a.pipeline_key).ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_sun_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let sun_archetype = DebugTrackedRwLock::read(&pe.archetypes().sun_render_archetype);
    sun_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_sky_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let sky_archetype = DebugTrackedRwLock::read(&pe.archetypes().sky_render_archetype);
    sky_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_background_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe = wait_for_pe!(res_guard, handle)?;
    let background_archetype =
      DebugTrackedRwLock::read(&pe.archetypes().background_render_archetype);
    background_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_grid_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
    let grid_archetype = DebugTrackedRwLock::read(&pe.archetypes().grid_render_archetype);
    grid_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_bvh_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
    let bvh_archetype = DebugTrackedRwLock::read(&pe.archetypes().bvh_render_archetype);
    bvh_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  #[named]
  fn get_bvhwire2_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let pe_lock = &res_guard.live_presentation_engines;
    let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
    let bvh_archetype = DebugTrackedRwLock::read(&pe.archetypes().bvhwire2_render_archetype);
    bvh_archetype
      .as_ref()
      .map(|a| a.pipeline_key)
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  // In device.rs

  #[named]
  fn allocate_rasterized_font_atlas(
    &self,
    cmd: CommandBufferHandle,
    hash: u64,
    font_atlas: alloc::sync::Arc<FontAtlas>,
  ) -> GpuResult<u32> {
    let (command_buffer, handle) = self.get_cmd_and_pe(cmd)?;

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, _| {
        let a1_arc = state
          .text_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text1"))?
          .clone();
        let a2_arc = state
          .text2_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text2"))?
          .clone();

        // 1. Prepare: Check for existing allocations and reserve new indices
        let (prep1, prep2) = {
          let mut a1 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a1_arc);
          let mut a2 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a2_arc);
          (
            a1.prepare_upload_font_atlas(hash)?,
            a2.prepare_upload_font_atlas(hash)?,
          )
        };

        if prep1.descriptor_index != prep2.descriptor_index {
          return Err(gpu_err!(
            "Text and Text2 font atlas descriptor indices drifted out of sync!"
          ));
        }

        if prep1.is_already_uploaded && prep2.is_already_uploaded {
          return Ok((a1_arc, a2_arc, prep1, prep2, None));
        }

        let vma_view = state.allocator.allocator.as_allocator_view();
        let staging_arena_ptr = state
          .frame_staging_arena
          .read()
          .as_ref()
          .map(|a| a as *const _)
          .ok_or(gpu_err!("staging arena missing"))?;

        Ok((
          a1_arc,
          a2_arc,
          prep1,
          prep2,
          Some((vma_view, staging_arena_ptr)),
        ))
      })?
      .execute(|(a1_arc, a2_arc, prep1, prep2, exec_data), rollback| {
        let Some((vma_view, staging_arena_ptr)) = exec_data else {
          return Ok((a1_arc, a2_arc, prep1, prep2, None)); // Pass through if already uploaded
        };

        let staging_arena = unsafe { &*staging_arena_ptr };

        let texture = crate::simulation::comet::Texture {
          data: font_atlas.image_data.clone().into(),
          format: crate::simulation::comet::TexelFormat::R8_UNORM,
          width: font_atlas.width,
          height: font_atlas.height,
          has_mipmaps: false,
        };

        // 2. Execute: Lock-Free Vulkan Resource Upload and Descriptor Binding
        let image1 = resources::TextRenderResourceArchetypeArena::execute_upload_font_atlas(
          &self.device,
          vma_view,
          command_buffer,
          staging_arena,
          &texture,
          &prep1,
          "FontAtlas Dynamic 1",
          rollback,
        )?;

        let image2 = resources::Text2RenderResourceArchetypeArena::execute_upload_font_atlas(
          &self.device,
          vma_view,
          command_buffer,
          staging_arena,
          &texture,
          &prep2,
          "FontAtlas Dynamic 2",
          rollback,
        )?;

        Ok((a1_arc, a2_arc, prep1, prep2, Some((image1, image2))))
      })
      .commit_read(|_state, execute_result| {
        let (a1_arc, a2_arc, prep1, prep2, images_opt) = execute_result?;

        // 3. Commit: Finalize map insertion with new data
        if let Some((image1, image2)) = images_opt {
          let mut a1 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a1_arc);
          let mut a2 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a2_arc);

          a1.commit_upload_font_atlas(hash, font_atlas.clone(), image1, prep1.descriptor_index);
          a2.commit_upload_font_atlas(hash, font_atlas.clone(), image2, prep2.descriptor_index);
        }

        Ok(prep1.descriptor_index)
      })
  }

  #[named]
  fn free_rasterized_font_atlas(&self, hash: u64, _font_atlas_id: u32) -> GpuResult<()> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        let a1_arc = state
          .text_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text1"))?
          .clone();
        let a2_arc = state
          .text2_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text2"))?
          .clone();

        // 1. Prepare: Mutate maps instantly, returning raw handle data
        let (prep1, prep2) = {
          let mut a1 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a1_arc);
          let mut a2 =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*a2_arc);
          (
            a1.prepare_remove_font_atlas(hash)?,
            a2.prepare_remove_font_atlas(hash)?,
          )
        };

        let allocator_raw = state.allocator.allocator.get_raw();
        let timeline = state.timeline_manager.get_cached_value();
        let discard_pool_ptr = &state.discard_pool as *const _;

        Ok((prep1, prep2, allocator_raw, timeline, discard_pool_ptr))
      })?
      .execute(
        |(prep1, prep2, allocator_raw, timeline, discard_pool_ptr), _rollback| {
          // 2. Execute: Push extracted handles straight into lock-free DiscardPool limits
          let discard_pool = unsafe { &*discard_pool_ptr };

          resources::TextRenderResourceArchetypeArena::execute_remove_font_atlas(
            &prep1,
            discard_pool,
            allocator_raw,
            timeline,
          );
          resources::Text2RenderResourceArchetypeArena::execute_remove_font_atlas(
            &prep2,
            discard_pool,
            allocator_raw,
            timeline,
          );

          Ok(())
        },
      )
      .commit_read(|_state, execute_result| {
        // 3. Commit: Map cleanups are natively immediate so nothing extra is required
        execute_result
      })
  }

  #[named]
  fn present(
    &self,
    handle: PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<crate::gpu::SwapchainStatus> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| extract_pe!(state, h))?
      .execute(|mut pe, _rollback| {
        let graphics_queue = self.queues.get_graphics_queue().handle;
        let result = unsafe {
          pe.submit_image(
            &self.device,
            graphics_queue,
            image_index as u32,
            frame_index as u32,
          )
        };
        Ok((pe, result))
      })
      .commit_read(|state, ok_result| {
        let (pe, result) = ok_result.unwrap();
        state.live_presentation_engines.insert(handle, pe);
        result
      })
  }

  // TODO rewrite with transaction behaviour
  #[named]
  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
    task_id: Option<u64>,
  ) -> GpuResult<()> {
    // SCOPE 1: Lock briefly to extract required state, then drop the lock!
    let (image, width, height, mut wait_value, timeline_sem, task_entry) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe = wait_for_pe!(res_guard, handle)?;

      if let swapchain::PresentationState::Windowless(windowless) = &*pe {
        let image = windowless.get_last_submitted_image()?;
        let (width, height) = windowless.extent();

        let (wait_val, entry) = match task_id {
          Some(id) => {
            let registry = DebugTrackedRwLock::read(&res_guard.timeline_manager.task_registry);
            if let Some(entry) = registry.get(&id) {
              (
                entry.target_value.load(Ordering::Acquire),
                Some(entry.clone()),
              )
            } else {
              return Err(gpu_invalid_arg!("no task id"));
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
        return Err(gpu_invalid_arg!(
          "windowed presentation engine cannot download (TODO)"
        ));
      }
    }; // <-- Locks safely dropped here!

    let buffer_size = (width * height * 4) as vk::DeviceSize;
    if buffer.len() != buffer_size as usize {
      return Err(gpu_invalid_arg!(
        "buffer size mismatch: {} vs {}",
        buffer.len(),
        buffer_size
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
    self.device.wait_for_semaphore_value(timeline_sem, wait_value, u64::MAX)?;
    oshal::log!("DEBUG RUST: timeline {} reached", wait_value);

    // Staging buffer creation
    let buffer_info = vk::BufferCreateInfo::default()
      .size(buffer_size)
      .usage(vk::BufferUsageFlags::TRANSFER_DST);

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED;
    crate::apply_test_dedicated_alloc!(alloc_info);

    // SCOPE 2: Lock briefly to allocate resource memory
    let graphics_queue = self.queues.get_graphics_queue();
    let (staging_buffer, alloc, command_pool, command_buffer, alloc_info_res) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let (staging_buffer, alloc) =
        unsafe { res_guard.allocator.allocator.create_buffer(&buffer_info, &alloc_info) }?;
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
      self.device.begin_command_buffer(command_buffer, &begin_info)?;

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
      self.device.synchronization2.cmd_pipeline_barrier2(command_buffer, &dep_info);

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
    let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None)? };

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
      let res_guard = DebugTrackedRwLock::read(&self.res);
      res_guard.allocator.allocator.invalidate_allocation(&alloc, 0, vk::WHOLE_SIZE)?;
      unsafe {
        core::ptr::copy_nonoverlapping(mapped_ptr, buffer.as_mut_ptr(), buffer_size as usize);
      }
    }

    unsafe {
      let mut mut_alloc = alloc;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      res_guard.allocator.allocator.destroy_buffer(staging_buffer, &mut mut_alloc);
    }

    Ok(())
  }

  #[named]
  fn get_command_buffer(&self) -> GpuResult<gpu::CommandBufferHandle> {
    let (cmd_id, cmd_pool_arc) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let cmd_id = res_guard.next_cmd_id.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
      let cmd_pool_arc = unsafe {
        DebugTrackedRwLock::read(&res_guard.command_pools)
          .get_unchecked(self.queues.get_graphics_queue().index as usize)
          .as_ref()
          .unwrap_unchecked()
          .clone()
      };
      (cmd_id, cmd_pool_arc)
    };

    // even increasing, so it shouldn't be there
    debug_assert!(
      !DebugTrackedRwLock::read(&self.recording_command_buffers)
        .contains_key(&CommandBufferHandle(cmd_id))
    );

    let cmd =
      cmd_pool_arc.allocate_primary(&self.device, this_thread::id(), CommandBufferId(cmd_id))?;

    DebugTrackedRwLock::write(&self.recording_command_buffers).insert(
      CommandBufferHandle(cmd_id),
      RecordingCmdBufferData::new(unsafe { NonZeroHandle::new_unchecked(cmd) }),
    );

    Ok(CommandBufferHandle(cmd_id))
  }

  #[named]
  fn set_command_buffer_presentation_engine(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
    let data = cmd_buffers.get_mut(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
    data.presentation_engine = Some(handle);
    Ok(())
  }

  // TODO group all &'static str error message
  #[named]
  fn begin_command_buffer(&self, cmd_buffer: gpu::CommandBufferHandle) -> GpuResult<()> {
    let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
    let data = cmd_buffers.get_mut(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;

    if data.has_begun {
      return Ok(());
    }

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self.device.begin_command_buffer(data.command_buffer.get(), &begin_info)?;
    }
    data.has_begun = true;

    Ok(())
  }

  #[named]
  fn get_emissive_paint_image_mapped_ptr(
    &self,
    mesh_id: crate::gpu::RenderableInstanceId,
  ) -> Option<*mut u8> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let mesh2_res = &res_guard.physical_mesh2_resources;
    let resource_ref = mesh2_res.get(&mesh_id)?;
    let paint_image_resource = match resource_ref.value() {
      resources::ResourceState::Ready(r) => r,
      _ => return None,
    };
    let paint_image = paint_image_resource.emissive_paint_image.as_ref()?;

    let alloc_info = res_guard.allocator.allocator.get_allocation_info(&paint_image.allocation);
    let mapped_ptr = alloc_info.mapped_data as *mut u8;
    if mapped_ptr.is_null() {
      None
    } else {
      Some(mapped_ptr)
    }
  }

  fn flush_emissive_paint_image(&self, mesh_id: crate::gpu::RenderableInstanceId) -> GpuResult<()> {
    let res_guard = DebugTrackedRwLock::read(&self.res);
    if let Some(resource_ref) = res_guard.physical_mesh2_resources.get(&mesh_id) {
      if let resources::ResourceState::Ready(r) = resource_ref.value() {
        if let Some(paint_image) = &r.emissive_paint_image {
          res_guard.allocator.allocator.flush_allocation(
            &paint_image.allocation,
            0,
            ash::vk::WHOLE_SIZE as u64,
          )?;
        }
      }
    }
    Ok(())
  }

  #[named]
  fn begin_render_pass(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    self.begin_render_pass_impl(
      cmd_buffer,
      presentation_engine,
      acquire_result,
      false, // single-subpass
    )
  }

  #[named]
  fn begin_compositing_render_pass(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    self.begin_render_pass_impl(
      cmd_buffer,
      presentation_engine,
      acquire_result,
      true, // compositing (3 subpasses)
    )
  }

  #[named]
  fn set_viewport(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    viewport: &crate::gpu::Viewport,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;
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

  #[named]
  fn set_scissor(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    scissor: &gpu::Rect2D,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;
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

  #[named]
  fn bind_pipeline(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    pipeline_key: crate::gpu::PipelineKey,
  ) -> GpuResult<()> {
    let res_guard = DebugTrackedRwLock::read(&self.res);

    // Check if we're inside a compositing render pass and need to adapt
    let actual_pipeline_key = {
      let cmd_buffers = DebugTrackedRwLock::read(&self.recording_command_buffers);
      let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
      if let Some(ref ctx) = data.compositing_ctx {
        // Look up the original GraphicsInfo and create a compositing variant
        if let Some(info) = res_guard.pipeline_pool.get_graphics_info(pipeline_key) {
          let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&self.device);
          let (key, _) = res_guard.pipeline_pool.get_or_create_compositing_variant(
            &self.device,
            &info,
            ctx.render_pass,
            ctx.subpass,
            &mut rollback,
          )?;
          key
        } else {
          // Pipeline was created without stored GraphicsInfo (e.g. composite pipeline)
          // — use original key as-is
          pipeline_key
        }
      } else {
        pipeline_key
      }
    };

    let pipeline = res_guard
      .pipeline_pool
      .get_graphics_pipeline(actual_pipeline_key)
      .ok_or(gpu_err_pipeline_absent!())?;

    let cmd = self.get_cmd(cmd_buffer)?;

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
    }

    {
      let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
      let data = unsafe { cmd_buffers.get_mut(&cmd_buffer).unwrap_unchecked() };
      // ready to discard it if necessary (on resize)
      data.bound_pipeline = Some(pipeline);
    }

    Ok(())
  }

  #[named]
  fn check_billboard_texture_id(&self, texture_id: u64) -> GpuResult<()> {
    let res = DebugTrackedRwLock::read(&self.res);
    let billboard_resources = DebugTrackedRwLock::read(&res.billboard_resources);
    if billboard_resources.len() > texture_id as usize {
      Ok(())
    } else {
      Err(gpu_invalid_arg!(
        "billboard resources {} non existent",
        texture_id
      ))
    }
  }

  // TODO: Don't use a fence, try fully GPU sync approach with [`gpu::ParticleSyncMode`]
  #[named]
  fn add_billboard_texture(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    _texture_id: u64,
    texture: &Texture,
    _current_frame: u64,
  ) -> GpuResult<u32> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_write(handle, |state, h| {
        let live_pes = &state.live_presentation_engines;
        let pe = live_pes.get(&h).ok_or(gpu_err_cmd_no_pe!())?;

        let archetype_guard = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &pe.archetypes().billboard_render_archetype,
        );
        let archetype = archetype_guard.as_ref().ok_or(gpu_err_archetype_absent!())?;

        let arena_arc = archetype.deref_arena().ok_or(crate::gpu_err_device!())?.clone();
        let descriptor_set =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc)
            .descriptor_set
            .get();

        // Reserve index in the global billboard array
        let mut billboard_resources =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
            &state.billboard_resources,
          );
        let array_index = billboard_resources.len() as u32;

        // Insert a dummy image to securely reserve our bindless slot while we allocate lock-free
        billboard_resources.push(resources::Image {
          image: unsafe { NonZeroHandle::new_unchecked(vk::Image::null()) },
          image_view: unsafe { NonZeroHandle::new_unchecked(vk::ImageView::null()) },
          allocation: unsafe { core::mem::zeroed() },
        });

        let vma_view = state.allocator.allocator.as_allocator_view();
        let staging_arena_ptr =
          state.frame_staging_arena.read().as_ref().map(|a| a as *const _).unwrap();
        let sampler = state.linear_sampler;

        Ok((
          descriptor_set,
          vma_view,
          staging_arena_ptr,
          sampler,
          array_index,
        ))
      })?
      .execute(
        |(descriptor_set, vma_view, staging_arena_ptr, sampler, array_index), rollback| {
          let staging_arena = unsafe { &*staging_arena_ptr };
          let debug_name = alloc::format!("BillBoard_{}", array_index);

          // 1. Heavy Allocation + Upload (Lock-Free)
          let image = resources::Image::new_2d(
            &self.device,
            vma_view,
            cmd,
            staging_arena,
            texture,
            vk::ImageUsageFlags::SAMPLED,
            &debug_name,
          )?;

          // 2. Protect with RollbackContext in case of aborts
          let img_h = image.image.get();
          let view_h = image.image_view.get();
          let mut alloc_h = image.allocation;
          rollback.defer(move |dev| unsafe {
            dev.destroy_image_view(view_h, None);
            vma_view.destroy_image(img_h, &mut alloc_h);
          });

          // 3. Update the array slot in the bindless descriptor set
          let image_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view_h)
            .sampler(sampler.get());

          let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(array_index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(core::slice::from_ref(&image_info));

          unsafe {
            self.device.update_descriptor_sets(core::slice::from_ref(&write), &[]);
          }

          Ok((array_index, image))
        },
      )
      .commit_read(|state, execute_result| {
        let (array_index, image) = execute_result?;

        // Safely replace the reserved dummy slot with the real uploaded image
        let mut billboard_resources =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
            &state.billboard_resources,
          );
        billboard_resources[array_index as usize] = image;

        Ok(array_index)
      })
  }

  #[named]
  fn bind_buffers(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    _pipeline: crate::gpu::PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    let (
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      pipeline_layout,
      descriptor_set,
    ) = {
      let physical_mesh_id = RenderableInstanceId(buffers.0);
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let physical_mesh_resources_guard = &res_guard.physical_mesh_resources;
      let resource_ref = physical_mesh_resources_guard.get(&physical_mesh_id).ok_or(
        gpu_invalid_arg!("couldn't get render mesh resource {:?}", physical_mesh_id),
      )?;
      let resource = match resource_ref.value() {
        resources::ResourceState::Ready(r) => r,
        _ => return Err(gpu_err!("resource not ready")),
      };
      let live_pes = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;

      let physical_mesh_render_archetype_guard =
        DebugTrackedRwLock::read(&pe.archetypes().physical_mesh_render_archetype);
      let archetype = physical_mesh_render_archetype_guard
        .as_ref()
        .ok_or(gpu_err_archetype_absent!())?;

      (
        resource.position_vertex_buffer.buffer.get(),
        resource.attributes_vertex_buffer.buffer.get(),
        resource.index_buffer.buffer.get(),
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
        )
        .pipeline_layout
        .get(),
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
      self.device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
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

  #[named]
  fn push_constants_raw(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    archetype: ArchetypeId,
    push_constants_bytes: &[u8],
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    let res_guard = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res_guard.live_presentation_engines;
    let pe = wait_for_pe_direct!(live_pes, handle)?;

    let layout = match archetype {
      ArchetypeId::Sun => DebugTrackedRwLock::read(&pe.archetypes().sun_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::PhysicalMesh => {
        DebugTrackedRwLock::read(&pe.archetypes().physical_mesh_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Billboard => {
        DebugTrackedRwLock::read(&pe.archetypes().billboard_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Cursor => DebugTrackedRwLock::read(&pe.archetypes().cursor_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Marker => DebugTrackedRwLock::read(&pe.archetypes().marker_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Measurement => {
        DebugTrackedRwLock::read(&pe.archetypes().measurement_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Sky => DebugTrackedRwLock::read(&pe.archetypes().sky_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Grid => DebugTrackedRwLock::read(&pe.archetypes().grid_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Minimap => DebugTrackedRwLock::read(&pe.archetypes().minimap_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Text => DebugTrackedRwLock::read(&pe.archetypes().text_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Bvh => DebugTrackedRwLock::read(&pe.archetypes().bvh_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Particle => DebugTrackedRwLock::read(&pe.archetypes().particle_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Gizmo => DebugTrackedRwLock::read(&pe.archetypes().gizmo_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::PhysicalMesh2 => {
        DebugTrackedRwLock::read(&pe.archetypes().physical_mesh2_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Particle2 => {
        DebugTrackedRwLock::read(&pe.archetypes().particle2_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Trajectory => {
        DebugTrackedRwLock::read(&pe.archetypes().trajectory_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Ui => DebugTrackedRwLock::read(&pe.archetypes().ui_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Background => {
        DebugTrackedRwLock::read(&pe.archetypes().background_render_archetype)
          .as_ref()
          .and_then(|a| a.deref_arena())
          .map(|a| {
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
              .pipeline_layout
              .get()
          })
      }
      ArchetypeId::Text2 => DebugTrackedRwLock::read(&pe.archetypes().text_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
      ArchetypeId::Bvhwire2 => DebugTrackedRwLock::read(&pe.archetypes().bvhwire2_render_archetype)
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        }),
    }
    .ok_or(gpu_err_archetype_absent!())?;

    // The slice is already bytes, just pass it directly to Vulkan
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants_bytes,
      );
    }
    Ok(())
  }

  #[named]
  fn draw_indexed(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    index_count: u32,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    unsafe {
      self.device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
    }

    Ok(())
  }

  #[named]
  fn draw(&self, cmd_buffer: crate::gpu::CommandBufferHandle, vertex_count: u32) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    unsafe {
      self.device.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }

    Ok(())
  }

  #[named]
  fn draw_instanced(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    vertex_count: u32,
    instance_count: u32,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    unsafe {
      self.device.cmd_draw(cmd, vertex_count, instance_count, 0, 0);
    }

    Ok(())
  }

  #[named]
  fn draw_indirect(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    indirect_buffer: GpuResourceHandle,
    offset: u64,
    draw_count: u32,
    stride: u32,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    use ash::vk::Handle;
    let buffer = vk::Buffer::from_raw(indirect_buffer.0);

    unsafe {
      self.device.cmd_draw_indirect(cmd, buffer, offset, draw_count, stride);
    }

    Ok(())
  }

  #[named]
  fn update_sun(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    resolution: (u32, u32, u32),
    radius: f32,
  ) -> GpuResult<()> {
    let mut is_winner = false;
    loop {
      enum Action {
        Yield,
        Break,
        BreakWinner,
      }

      let action = {
        let res_guard = DebugTrackedRwLock::read(&self.res);
        match res_guard.sun_resources.entry(entity_id) {
          dashmap::mapref::entry::Entry::Occupied(e) => match e.get() {
            resources::ResourceState::Ready(_) => Action::Break,
            resources::ResourceState::Pending => Action::Yield,
          },
          dashmap::mapref::entry::Entry::Vacant(e) => {
            e.insert(resources::ResourceState::Pending);
            Action::BreakWinner
          }
        }
      };

      match action {
        Action::Break => break,
        Action::BreakWinner => {
          is_winner = true;
          break;
        }
        Action::Yield => aethervk_oshal_rlib::os::native::this_thread::sleep_for(
          core::time::Duration::from_millis(1),
        ),
      }
    }

    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let graphics_queue = self.queues.get_graphics_queue();
    let compute_queue = self.queues.get_compute_queue();

    // Data packages for the lock-free execution phase
    enum SunOperation {
      InitAndDispatch {
        shader_module: vk::ShaderModule,
        graphics_ds_layout: vk::DescriptorSetLayout,
        vma: vk_mem::ffi::VmaAllocator,
        pipeline_pool_ptr: *const pipelines::PipelinePool,
        descriptor_pool_arc: alloc::sync::Arc<descriptors::DescriptorPools>,
        discard_pool_ptr: *const resources::DiscardPool,
        linear_sampler: vk::Sampler,
      },
      DispatchOnly {
        vma: vk_mem::ffi::VmaAllocator,
        image: vk::Image,
        compute_pipeline: vk::Pipeline,
        compute_pipeline_layout: vk::PipelineLayout,
        compute_descriptor_set: vk::DescriptorSet,
        params_buffer: vk::Buffer,
        params_alloc: vk_mem::Allocation,
        is_generated: bool,
      },
      None,
    }

    enum ExecuteResult {
      Created(resources::SunRenderResource),
      Updated(u64),
      None,
    }

    let execution_result = (|| -> GpuResult<()> {
      crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
        .prepare_read(handle, |state, _h| {
          let pe = wait_for_pe!(state, handle)?;
          let timeline = state.timeline_manager.get_cached_value();
          let vma = state.allocator.allocator.get_raw();

          // 1. Check if the sun resource already exists
          if let Some(entry) = state.sun_resources.get(&entity_id) {
            if let resources::ResourceState::Ready(sun_res) = entry.value() {
              if sun_res.last_timeline == timeline {
                return Ok((timeline, SunOperation::None));
              }
              let op = SunOperation::DispatchOnly {
                vma,
                image: sun_res.image.as_ref().unwrap().image.get(),
                compute_pipeline: sun_res.compute_pipeline.unwrap().get(),
                compute_pipeline_layout: sun_res.compute_pipeline_layout.unwrap(),
                compute_descriptor_set: sun_res.compute_descriptor_set.unwrap(),
                params_buffer: sun_res.params_buffer.unwrap(),
                params_alloc: sun_res.params_alloc.unwrap(),
                is_generated: sun_res.is_generated,
              };
              return Ok((timeline, op));
            }
          }

          // 2. Resource doesn't exist. Prepare everything needed for Initialization.
          let comp_key = {
            let mut sm = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
              &state.shader_manager,
            );
            ensure_sungen_shader_module(&self.device, &mut sm)?
          };
          let shader_module = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
            &state.shader_manager,
          )
          .get(comp_key)
          .ok_or(GpuError::InvalidShader)?
          .module
          .get();

          let archetype_guard =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
              &pe.archetypes().sun_render_archetype,
            );
          let archetype = archetype_guard.as_ref().ok_or(gpu_err_archetype_absent!())?;
          let arena_arc = archetype.deref_arena().ok_or(crate::gpu_err_device!())?;
          let graphics_ds_layout =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc)
              .descriptor_set_layout
              .get();

          let op = SunOperation::InitAndDispatch {
            shader_module,
            graphics_ds_layout,
            vma,
            pipeline_pool_ptr: &state.pipeline_pool as *const _, // < this is a RwLock, so cast
            // invalid
            descriptor_pool_arc: state.descriptor_pool.read().as_ref().unwrap().clone(),
            discard_pool_ptr: &state.discard_pool as *const _,
            linear_sampler: state.linear_sampler.get(),
          };

          Ok((timeline, op))
        })?
        .execute(|(timeline, op), rollback| {
          // Shared Closure: Records the actual dispatch to avoid code duplication
          let record_dispatch = |device: &LogicalDevice,
                                 target_cmd: vk::CommandBuffer,
                                 buffer_address: u64,
                                 image: vk::Image,
                                 pipeline: vk::Pipeline,
                                 layout: vk::PipelineLayout,
                                 descriptor_set: vk::DescriptorSet,
                                 is_generated: bool| {
            let old_layout = if is_generated {
              vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            } else {
              vk::ImageLayout::UNDEFINED
            };

            unsafe {
              let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(if is_generated {
                  vk::PipelineStageFlags2::FRAGMENT_SHADER
                } else {
                  vk::PipelineStageFlags2::NONE
                })
                .src_access_mask(if is_generated {
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
                .image(image)
                .subresource_range(
                  vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .base_array_layer(0)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

              let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(core::slice::from_ref(&barrier));
              device.synchronization2.cmd_pipeline_barrier2(target_cmd, &dep_info);

              device.cmd_bind_pipeline(target_cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
              device.cmd_bind_descriptor_sets(
                target_cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor_set],
                &[],
              );

              let push_constants_bytes =
                core::slice::from_raw_parts(&buffer_address as *const _ as *const u8, 8);
              device.cmd_push_constants(
                target_cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants_bytes,
              );

              let group_count_x = (resolution.0 + 7) / 8;
              let group_count_y = (resolution.1 + 7) / 8;
              let group_count_z = (resolution.2 + 7) / 8;
              device.cmd_dispatch(target_cmd, group_count_x, group_count_y, group_count_z);

              let barrier2 = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(
                  vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(vk::REMAINING_MIP_LEVELS)
                    .base_array_layer(0)
                    .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

              let dep_info2 = vk::DependencyInfo::default()
                .image_memory_barriers(core::slice::from_ref(&barrier2));
              device.synchronization2.cmd_pipeline_barrier2(target_cmd, &dep_info2);
            }
          };

          match op {
            SunOperation::InitAndDispatch {
              shader_module,
              graphics_ds_layout,
              vma,
              pipeline_pool_ptr,
              descriptor_pool_arc,
              discard_pool_ptr,
              linear_sampler,
            } => {
              let allocator = unsafe { vk_mem::AllocatorView::from_raw(vma) };
              let pipeline_pool = unsafe { &*pipeline_pool_ptr };
              let discard_pool = unsafe { &*discard_pool_ptr };

              // 1. Create Image
              let image = resources::Image::new_storage_3d(
                &self.device,
                allocator,
                resolution.0,
                resolution.1,
                resolution.2,
                vk::Format::R16G16B16A16_SFLOAT,
                graphics_queue.family_index,
                compute_queue.family_index,
                "Sun",
              )?;

              let img_h = image.image.get();
              let view_h = image.image_view.get();
              let alloc_h = image.allocation.get_raw();
              rollback.defer(move |dev| unsafe {
                dev.destroy_image_view(view_h, None);
                vk_mem::ffi::vmaDestroyImage(vma, img_h, alloc_h);
              });

              // 2. Create Layouts & Pools
              let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
              let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
              let set_layout =
                unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;
              rollback
                .defer(move |dev| unsafe { dev.destroy_descriptor_set_layout(set_layout, None) });

              let push_constant_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(8)];
              let set_layouts = [set_layout];
              let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_constant_ranges);
              let pipeline_layout =
                unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None) }?;
              rollback
                .defer(move |dev| unsafe { dev.destroy_pipeline_layout(pipeline_layout, None) });

              let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)];
              let pool_info =
                vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);
              let descriptor_pool =
                unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;
              rollback
                .defer(move |dev| unsafe { dev.destroy_descriptor_pool(descriptor_pool, None) });

              let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(&set_layouts);
              let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

              let image_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(view_h);
              let write_descriptor_set = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(core::slice::from_ref(&image_info));
              unsafe { self.device.update_descriptor_sets(&[write_descriptor_set], &[]) };

              // 3. Compute Pipeline
              let mut compute_info = pipelines::ComputeInfo::default();
              compute_info.shader_module = shader_module;
              compute_info.pipeline_layout = pipeline_layout;
              compute_info.add_specialization_constant_u32(
                vk::SpecializationMapEntry {
                  constant_id: 10,
                  offset: 0,
                  size: 4,
                },
                if self.query_result.debug_shaders && cfg!(debug_assertions) {
                  1
                } else {
                  0
                },
              );

              let compute_pipeline = pipelines::PipelinePool::get_or_create_compute_pipeline(
                pipeline_pool,
                &self.device,
                &compute_info,
                rollback,
              )?;

              // 4. Params Buffer
              let params_size = core::mem::size_of::<[f32; 6]>() as u64;
              let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
              allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
              allocation_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
              crate::apply_test_dedicated_alloc!(allocation_create_info);

              let buffer_info = vk::BufferCreateInfo::default()
                .size(params_size)
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
              let (params_buffer, params_alloc) =
                unsafe { allocator.create_buffer(&buffer_info, &allocation_create_info)? };
              let mut params_alloc_mut = params_alloc;
              let alloc_clone = allocator.get_raw();
              rollback.defer(move |_| unsafe {
                let alloc = vk_mem::AllocatorView::from_raw(alloc_clone);
                alloc.destroy_buffer(params_buffer, &mut params_alloc_mut)
              });

              let alloc_info = allocator.get_allocation_info(&params_alloc);
              unsafe {
                let ptr = alloc_info.mapped_data as *mut [f32; 6];
                *ptr = [
                  timeline as f32 * 0.016,
                  5778.0,
                  1000000.0,
                  radius,
                  0.05,
                  15.0,
                ];
                allocator.flush_allocation(&params_alloc, 0, vk::WHOLE_SIZE as u64)?;
              }

              let bda_info = vk::BufferDeviceAddressInfo::default().buffer(params_buffer);
              let buffer_address =
                unsafe { self.device.buffer_device_address.get_buffer_device_address(&bda_info) };

              // 5. Graphics Descriptor Set
              let (_, graphics_descriptor_set) = descriptor_pool_arc.allocate_and_get_active_pool(
                &self.device,
                graphics_ds_layout,
                discard_pool,
                timeline,
                "Sun",
                rollback,
              )?;

              let image_info_gfx = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(view_h)
                .sampler(linear_sampler);
              let write_descriptor_set_gfx = vk::WriteDescriptorSet::default()
                .dst_set(graphics_descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(core::slice::from_ref(&image_info_gfx));

              unsafe {
                self
                  .device
                  .update_descriptor_sets(core::slice::from_ref(&write_descriptor_set_gfx), &[]);
              }

              // 6. Record Dispatch synchronously to ensure layout transitions happen before any other camera tries to draw
              let transient_res = self.run_transient_commands(|transient_cmd| {
                record_dispatch(
                  &self.device,
                  transient_cmd,
                  buffer_address,
                  img_h,
                  compute_pipeline.get(),
                  pipeline_layout,
                  descriptor_set,
                  false,
                );
                Ok(())
              })?;
              let timeline = DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;
              discard_pool.discard_type_erased(transient_res, timeline + 2);

              let new_resource = resources::SunRenderResource {
                resolution,
                image: Some(image),
                descriptor_set: Some(unsafe {
                  NonZeroHandle::new_unchecked(graphics_descriptor_set)
                }),
                is_generated: false, // Will be set to true in the commit phase
                compute_descriptor_pool: Some(descriptor_pool),
                compute_descriptor_set_layout: Some(set_layout),
                compute_descriptor_set: Some(descriptor_set),
                compute_pipeline: Some(compute_pipeline),
                compute_pipeline_layout: Some(pipeline_layout),
                params_buffer: Some(params_buffer),
                params_alloc: Some(params_alloc),
                last_timeline: timeline,
              };

              Ok(ExecuteResult::Created(new_resource))
            }
            SunOperation::DispatchOnly {
              vma,
              image,
              compute_pipeline,
              compute_pipeline_layout,
              compute_descriptor_set,
              params_buffer,
              params_alloc,
              is_generated,
            } => {
              let allocator = unsafe { vk_mem::AllocatorView::from_raw(vma) };

              // 1. Update params buffer
              let alloc_info = allocator.get_allocation_info(&params_alloc);
              unsafe {
                let ptr = alloc_info.mapped_data as *mut f32;
                *ptr = timeline as f32 * 0.016;
                allocator.flush_allocation(&params_alloc, 0, vk::WHOLE_SIZE as u64)?;
              }

              let bda_info = vk::BufferDeviceAddressInfo::default().buffer(params_buffer);
              let buffer_address =
                unsafe { self.device.buffer_device_address.get_buffer_device_address(&bda_info) };

              // 2. Record Dispatch
              record_dispatch(
                &self.device,
                cmd,
                buffer_address,
                image,
                compute_pipeline,
                compute_pipeline_layout,
                compute_descriptor_set,
                is_generated,
              );

              Ok::<_, GpuError>(ExecuteResult::Updated(timeline))
            }
            SunOperation::None => {
              Ok(ExecuteResult::None)
            }
          }
        })
        .commit_read(|state, execute_result| {
          let res = execute_result?;

          match res {
            ExecuteResult::Created(mut new_resource) => {
              new_resource.is_generated = true;
              state
                .sun_resources
                .insert(entity_id, resources::ResourceState::Ready(new_resource));
            }
            ExecuteResult::Updated(timeline) => {
              if let Some(mut entry) = state.sun_resources.get_mut(&entity_id) {
                if let resources::ResourceState::Ready(sun_res) = entry.value_mut() {
                  sun_res.last_timeline = timeline;
                }
              }
            }
            ExecuteResult::None => {}
          }

          Ok(())
        })
    })();

    if execution_result.is_err() && is_winner {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      if let Some(entry) = res_guard.sun_resources.get(&entity_id) {
        if matches!(entry.value(), resources::ResourceState::Pending) {
          drop(entry);
          res_guard.sun_resources.remove(&entity_id);
        }
      }
    }

    execution_result
  }

  #[named]
  fn prepare_billboard_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let (pipeline_key, layout, d) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let billboard_render_archetype =
        DebugTrackedRwLock::read(&pe.archetypes().billboard_render_archetype);
      let billboard_render_archetype_ref =
        billboard_render_archetype.as_ref().ok_or(gpu_err_archetype_absent!())?;
      let (d, layout) = {
        let arena_arc =
          billboard_render_archetype_ref.deref_arena().ok_or(gpu_err!("arena absent"))?;
        let arena = DebugTrackedRwLock::read(&*arena_arc);
        (arena.descriptor_set.get(), arena.pipeline_layout.get())
      };

      (billboard_render_archetype_ref.pipeline_key, layout, d)
    }; // <-- locks released here
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
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

  #[named]
  fn prepare_bvh_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;

      let mut a_lock = DebugTrackedRwLock::write(&pe.archetypes().bvh_render_archetype);
      let bvh_render_archetype = a_lock.as_mut().ok_or(gpu_err_archetype_absent!())?;
      (cmd, bvh_render_archetype.pipeline_key)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
      self.device.cmd_set_line_width(cmd, 1.0);
      // now each BVH needs to do push constant and draw.
    }
    Ok(())
  }

  #[named]
  fn prepare_bvhwire2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res.live_presentation_engines;
      let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
      let a_lock = DebugTrackedRwLock::read(&pe.archetypes().bvhwire2_render_archetype);
      let bvhwire2_render_archetype = a_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;
      (cmd, bvhwire2_render_archetype.pipeline_key)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
      self.device.cmd_set_line_width(cmd, 1.0);
    }
    Ok(())
  }

  #[named]
  fn allocate_sphere_gizmo_instance(&self, entity: crate::scene::EntityId) -> GpuResult<u32> {
    let res = DebugTrackedRwLock::read(&self.res);
    let arena_arc = res
      .sphere_gizmo_render_archetype_arena
      .as_ref()
      .ok_or(gpu_err!("arena absent"))?;
    let mut arena = DebugTrackedRwLock::write(&*arena_arc);
    arena.allocate_sphere_gizmo_instance(entity)
  }

  #[named]
  fn free_sphere_gizmo_instance(&self, entity: crate::scene::EntityId) -> GpuResult<()> {
    let res = DebugTrackedRwLock::read(&self.res);
    let arena_arc = res
      .sphere_gizmo_render_archetype_arena
      .as_ref()
      .ok_or(gpu_err!("arena absent"))?;
    let mut arena = DebugTrackedRwLock::write(&*arena_arc);
    arena.free_sphere_gizmo_instance(entity);
    Ok(())
  }

  #[named]
  fn prepare_sphere_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().sphere_gizmo_render_archetype);
      let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;
      (cmd, archetype.pipeline_key)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
      self.device.cmd_set_line_width(cmd, 1.0);
    }
    Ok(())
  }

  #[named]
  fn push_sphere_gizmo_constants(
    &self,
    cmd_buffer: CommandBufferHandle,
    constants: &crate::gpu::SphereGizmoPushConstants,
  ) -> GpuResult<()> {
    let (cmd, layout) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let (cmd, _) = self.get_cmd_and_pe(cmd_buffer)?;
      let arena_arc = res
        .sphere_gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);
      (cmd, arena.pipeline_layout.get())
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        constants as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::SphereGizmoPushConstants>(),
      )
    };
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        bytes,
      );
    }
    Ok(())
  }

  #[named]
  fn get_sphere_gizmo_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey> {
    let res = DebugTrackedRwLock::read(&self.res);
    let live_pes = &res.live_presentation_engines;
    let pe = wait_for_pe_direct!(live_pes, handle)?;
    let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().sphere_gizmo_render_archetype);
    let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;
    Ok(archetype.pipeline_key)
  }

  #[named]
  fn upload_sphere_gizmos_batch(
    &self,
    cmd_buffer: CommandBufferHandle,
    gizmos: &[(u32, crate::gpu::SphereGizmoDataGpu)],
  ) -> GpuResult<Option<crate::gpu::frame::SphereGizmoBatchCall>> {
    if gizmos.is_empty() {
      return Ok(None);
    }

    let _total_gizmos = gizmos.len() as u32;

    let (cmd, staging_offset, staging_ptr, data_buffer, staging_buffer, data_ptr, pipeline) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().sphere_gizmo_render_archetype);
      let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;

      let mut staging_arena = DebugTrackedRwLock::write(&res.frame_staging_arena);
      let staging = staging_arena.as_mut().ok_or(GpuError::InvalidState(
        "SphereGizmo missing staging arena".to_string(),
      ))?;

      let arena_arc = res
        .sphere_gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);
      let data_buffer = arena.data_buffer;
      let data_ptr = arena.data_ptr;

      let max_idx = gizmos.iter().map(|(idx, _)| *idx).max().unwrap_or(0);
      let total_capacity = (max_idx + 1) as usize;
      let data_size =
        (total_capacity * core::mem::size_of::<crate::gpu::SphereGizmoDataGpu>()) as u64;

      let (staging_offset, staging_ptr) =
        staging.allocate(data_size as usize, 16).ok_or(GpuError::OutOfMemory)?;

      (
        cmd,
        staging_offset,
        staging_ptr,
        data_buffer.get(),
        staging.buffer,
        data_ptr,
        archetype.pipeline_key,
      )
    };

    let total_capacity = (gizmos.iter().map(|(idx, _)| *idx).max().unwrap_or(0) + 1) as usize;
    let data_size =
      (total_capacity * core::mem::size_of::<crate::gpu::SphereGizmoDataGpu>()) as u64;

    unsafe {
      core::ptr::write_bytes(staging_ptr, 0, data_size as usize); // zero init
      let typed_ptr = staging_ptr as *mut crate::gpu::SphereGizmoDataGpu;
      for (idx, data) in gizmos {
        core::ptr::write(typed_ptr.add(*idx as usize), *data);
      }
    }

    let copy_region = vk::BufferCopy::default()
      .size(data_size)
      .src_offset(staging_offset as u64)
      .dst_offset(0);

    unsafe {
      self.device.cmd_copy_buffer(cmd, staging_buffer, data_buffer, &[copy_region]);

      let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .buffer(data_buffer)
        .offset(0)
        .size(data_size);

      let dependency_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dependency_info);
    }

    let mut total_vertices = 0;
    for (_, data) in gizmos {
      let sub_divs = data.subdivisions.max(4.0) as u32;
      let lat_segments = sub_divs;
      let lon_segments = sub_divs;
      let total_sphere_vertices = lon_segments * (2 * lat_segments - 1) * 2;
      let total_axes_vertices = 6;
      let total_arrowhead_vertices = 4 * 2 * 3;
      total_vertices =
        total_vertices.max(total_sphere_vertices + total_axes_vertices + total_arrowhead_vertices);
    }

    Ok(Some(crate::gpu::frame::SphereGizmoBatchCall {
      pipeline,
      total_gizmos: (gizmos.iter().map(|(idx, _)| *idx).max().unwrap_or(0) + 1) as u32,
      total_vertices,
      data_ptr,
    }))
  }
  #[named]
  fn upload_bvh_box_data(
    &self,
    cmd_buffer: CommandBufferHandle,
    data: &crate::gpu::BvhBoxData,
  ) -> GpuResult<u64> {
    let data_size = core::mem::size_of::<crate::gpu::BvhBoxData>();
    let res = DebugTrackedRwLock::read(&self.res);
    let _cmd = self.get_cmd(cmd_buffer)?;

    let mut staging_arena_guard = DebugTrackedRwLock::write(&res.frame_staging_arena);
    let arena = staging_arena_guard
      .as_mut()
      .ok_or(gpu_err!("BVH box data: staging arena missing"))?;

    let (offset, ptr) = arena.allocate(data_size, 4).ok_or(GpuError::OutOfMemory)?;

    unsafe {
      core::ptr::copy_nonoverlapping(data as *const _ as *const u8, ptr, data_size);
    }

    let base_addr = unsafe {
      self
        .device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
    };

    Ok(base_addr + offset as u64)
  }

  #[named]
  fn upload_mesh_push_extra(
    &self,
    cmd_buffer: CommandBufferHandle,
    data: &crate::gpu::MeshPushExtra,
  ) -> GpuResult<u64> {
    let data_size = core::mem::size_of::<crate::gpu::MeshPushExtra>();
    let res = DebugTrackedRwLock::read(&self.res);
    let _cmd = self.get_cmd(cmd_buffer)?;

    let mut staging_arena_guard = DebugTrackedRwLock::write(&res.frame_staging_arena);
    let arena = staging_arena_guard
      .as_mut()
      .ok_or(gpu_err!("Mesh push extra: staging arena missing"))?;

    let (offset, ptr) = arena.allocate(data_size, 8).ok_or(GpuError::OutOfMemory)?;

    unsafe {
      core::ptr::copy_nonoverlapping(data as *const _ as *const u8, ptr, data_size);
    }

    let base_addr = unsafe {
      self
        .device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
    };

    Ok(base_addr + offset as u64)
  }

  #[named]
  fn upload_minimap_planets(
    &self,
    cmd_buffer: CommandBufferHandle,
    planets: &[crate::gpu::MinimapPlanetGpu],
  ) -> GpuResult<u64> {
    if planets.is_empty() {
      return Ok(0);
    }
    let data_size = planets.len() * core::mem::size_of::<crate::gpu::MinimapPlanetGpu>();
    let res = DebugTrackedRwLock::read(&self.res);
    let _cmd = self.get_cmd(cmd_buffer)?;

    let mut staging_arena_guard = DebugTrackedRwLock::write(&res.frame_staging_arena);
    let arena = staging_arena_guard
      .as_mut()
      .ok_or(gpu_err!("Minimap planets: staging arena missing"))?;

    let (offset, ptr) = arena.allocate(data_size, 4).ok_or(GpuError::OutOfMemory)?;

    unsafe {
      core::ptr::copy_nonoverlapping(planets.as_ptr() as *const u8, ptr, data_size);
    }

    let base_addr = unsafe {
      self
        .device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(arena.buffer))
    };

    Ok(base_addr + offset as u64)
  }
  #[named]
  fn upload_bvhwire2_batch(
    &self,
    cmd_buffer: CommandBufferHandle,
    bvh_data: &[crate::gpu::Bvhwire2DataGpu],
  ) -> GpuResult<Option<crate::gpu::frame::Bvhwire2BatchCall>> {
    aethervk_oshal_rlib::log!("upload_bvhwire2_batch called with {} boxes", bvh_data.len());
    if bvh_data.is_empty() {
      return Ok(None);
    }
    let total_boxes = bvh_data.len() as u32;
    let data_size = (bvh_data.len() * core::mem::size_of::<crate::gpu::Bvhwire2DataGpu>()) as u64;

    let (cmd, staging_offset, staging_ptr, data_buffer, staging_buffer, data_ptr, pipeline) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().bvhwire2_render_archetype);
      let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;

      let mut staging_arena = DebugTrackedRwLock::write(&res.frame_staging_arena);
      let staging = staging_arena.as_mut().ok_or(GpuError::InvalidState(
        "BVHWire2 missing staging arena".to_string(),
      ))?;

      // TODO transaction
      let (staging_offset, staging_ptr) =
        staging.allocate(data_size as usize, 16).ok_or(GpuError::OutOfMemory)?;

      let arena_arc =
        res.bvhwire2_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);
      let data_buffer = arena.data_buffer;
      let data_ptr = arena.data_ptr;

      (
        cmd,
        staging_offset,
        staging_ptr,
        data_buffer.get(),
        staging.buffer,
        data_ptr,
        archetype.pipeline_key,
      )
    }; // <- locks released here

    unsafe {
      core::ptr::copy_nonoverlapping(
        bvh_data.as_ptr(),
        staging_ptr as *mut crate::gpu::Bvhwire2DataGpu,
        bvh_data.len(),
      );
    }

    let copy_region = vk::BufferCopy::default()
      .size(data_size)
      .src_offset(staging_offset as u64)
      .dst_offset(0);

    unsafe {
      self.device.cmd_copy_buffer(cmd, staging_buffer, data_buffer, &[copy_region]);

      let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .buffer(data_buffer)
        .offset(0)
        .size(data_size);

      let dependency_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dependency_info);
    }

    Ok(Some(crate::gpu::frame::Bvhwire2BatchCall {
      pipeline,
      total_boxes,
      data_ptr,
    }))
  }

  #[named]
  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().gizmo_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_pipeline_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

      let arena_arc = res_guard
        .gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);
      (
        cmd,
        archetype.pipeline_key,
        arena.pipeline_layout.get(),
        arena.descriptor_set.get(),
      )
    }; // <- locks released here

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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

  #[named]
  fn prepare_sun_for_render(
    &self,
    cmd_buffer: CommandBufferHandle,
    entity: EntityId,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let (layout, ds) = {
      let res = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let sun_resource = &res.sun_resources;
      let sun_archetype = DebugTrackedRwLock::read(&pe.archetypes().sun_render_archetype);
      let layout = sun_archetype
        .as_ref()
        .and_then(|a| a.deref_arena())
        .map(|a| {
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*a)
            .pipeline_layout
            .get()
        })
        .ok_or(gpu_err_archetype_absent!())?;

      let resource_ref =
        sun_resource.get(&entity).ok_or(gpu_err!("couldn't find sun descriptor set"))?;
      let ds = match resource_ref.value() {
        resources::ResourceState::Ready(s) => s
          .descriptor_set
          .map(|d| d.get())
          .ok_or(gpu_err!("couldn't find sun descriptor set"))?,
        _ => return Err(gpu_err!("sun resource not ready")),
      };
      (layout, ds)
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

  #[named]
  fn prepare_particle_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().particle_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

      let arena_arc = res_guard
        .particle_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);

      (
        cmd,
        archetype.pipeline_key,
        arena.pipeline_layout.get(),
        arena.descriptor_set.get(),
      )
    }; // <- locks released here

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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

  #[named]
  fn prepare_particle2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let live_pes = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().particle2_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

      let arena_arc = res_guard
        .particle2_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);

      (
        cmd,
        archetype.pipeline_key,
        arena.pipeline_layout.get(),
        arena.descriptor_set.get(),
      )
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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

  #[named]
  fn prepare_trajectory_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(pe_lock, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().trajectory_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

      let arena_arc = res_guard
        .trajectory_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);

      (
        cmd,
        archetype.pipeline_key,
        arena.pipeline_layout.get(),
        arena.descriptor_set.get(),
      )
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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

  #[named]
  fn prepare_ui_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(pe_lock, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().ui_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };

      let arena_arc =
        res_guard.ui_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena = DebugTrackedRwLock::read(&*arena_arc);

      (
        cmd,
        archetype.pipeline_key,
        arena.pipeline_layout.get(),
        arena.descriptor_set.get(),
      )
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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

  #[named]
  fn prepare_sky_for_render(&self, cmd_buffer: gpu::CommandBufferHandle) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let timeline = DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        // 1. Check sky image
        let sky_image_guard =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&state.sky_image);
        let sky_image = sky_image_guard.as_ref().ok_or(gpu_err!("sky image absent"))?;
        let sky_image_view = sky_image.image_view.get();

        // 2. Fetch Presentation Engine
        let live_pes = &state.live_presentation_engines;
        let pe = wait_for_pe_direct!(live_pes, h)?;

        // 3. Fetch Archetype Arena
        let sky_render_archetype_guard =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
            &pe.archetypes().sky_render_archetype,
          );
        if sky_render_archetype_guard.is_none() {
          aethervk_oshal_rlib::log!(
            "[RenderThread] render_sky ERROR: sky_render_archetype_guard.is_none"
          );
          return Err(gpu_err_archetype_absent!());
        }

        let arena_arc = state
          .sky_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent"))?
          .clone();

        // 4. Extract needed layouts and check if allocation is necessary
        let (do_alloc, layout, pipeline_layout, existing_set) = {
          let arena =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&*arena_arc);
          (
            arena.descriptor_set.is_none(),
            arena.descriptor_set_layout.get(),
            arena.pipeline_layout.get(),
            arena.descriptor_set,
          )
        };

        // 5. Extract shared dependencies
        let pool_guard = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &state.descriptor_pool,
        );
        let dp_arc = pool_guard.as_ref().ok_or(gpu_err!("descriptor pool absent"))?.clone();
        let linear_sampler = state.linear_sampler.get();
        let discard_pool_ptr = &state.discard_pool as *const resources::DiscardPool;

        Ok((
          arena_arc,
          dp_arc,
          layout,
          pipeline_layout,
          sky_image_view,
          linear_sampler,
          do_alloc,
          existing_set,
          discard_pool_ptr,
        ))
      })?
      .execute(
        |(
          arena_arc,
          dp_arc,
          layout,
          pipeline_layout,
          sky_image_view,
          linear_sampler,
          do_alloc,
          existing_set,
          discard_pool_ptr,
        ),
         rollback| {
          let descriptor_set = if do_alloc {
            let discard_pool = unsafe { discard_pool_ptr.as_ref_unchecked() };
            // Allocate using the new persistent pool structure + rollback context
            let set = dp_arc.allocate(
              &self.device,
              layout,
              discard_pool,
              timeline,
              "Sky",
              rollback,
            )?;

            let image_info = vk::DescriptorImageInfo::default()
              .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
              .image_view(sky_image_view)
              .sampler(linear_sampler);

            let write_descriptor_set = vk::WriteDescriptorSet::default()
              .dst_set(set.get())
              .dst_binding(0)
              .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
              .image_info(core::slice::from_ref(&image_info));

            unsafe { self.device.update_descriptor_sets(&[write_descriptor_set], &[]) };

            set
          } else {
            existing_set.unwrap()
          };

          // Bind the descriptor set
          unsafe {
            self.device.cmd_bind_descriptor_sets(
              cmd,
              vk::PipelineBindPoint::GRAPHICS,
              pipeline_layout,
              0,
              &[descriptor_set.get()],
              &[],
            );
          }

          Ok((arena_arc, do_alloc, descriptor_set))
        },
      )
      .commit_read(|_state, execute_result| {
        let (arena_arc, do_alloc, descriptor_set) = execute_result?;

        if do_alloc {
          let mut arena =
            crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&*arena_arc);
          arena.descriptor_set = Some(descriptor_set);
        }

        Ok(())
      })?;

    Ok(())
  }

  #[named]
  fn prepare_text_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, layout, set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res.live_presentation_engines;
      let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
      let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().text_render_archetype);
      let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;

      let layout = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .pipeline_layout
      .get();
      let set = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .descriptor_set
      .ok_or(crate::gpu_err_device!())?;

      (cmd, archetype.pipeline_key, layout, set)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
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

  #[named]
  fn prepare_text2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, layout, set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res.live_presentation_engines;
      let pe = pe_lock.get(&handle).ok_or(GpuError::NotFound)?;
      let archetype_lock = DebugTrackedRwLock::read(&pe.archetypes().text2_render_archetype);
      let archetype = archetype_lock.as_ref().ok_or(gpu_err_archetype_absent!())?;

      let layout = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .pipeline_layout
      .get();
      let set = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .descriptor_set
      .ok_or(crate::gpu_err_device!())?;

      (cmd, archetype.pipeline_key, layout, set)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
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

  #[named]
  fn render_minimap(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    player_pos: Vec3f32,
    max_distance: f32,
    planets: &[(Vec3f32, f32, [f32; 4])],
    screen_extent: [f32; 2],
  ) -> GpuResult<()> {
    let (pipeline_key, layout) = {
      let (_, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(pe_lock, handle)?;
      let minimap_render_archetype_guard =
        DebugTrackedRwLock::read(&pe.archetypes().minimap_render_archetype);
      if minimap_render_archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { minimap_render_archetype_guard.as_ref().unwrap_unchecked() };
      let layout = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .pipeline_layout
      .get();
      (archetype.pipeline_key, layout)
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;

    let cmd = self.get_cmd(cmd_buffer)?;

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

    // Upload planet data to staging buffer via BDA
    let planet_count = planets.len().min(16);
    let mut planet_gpu = [crate::gpu::MinimapPlanetGpu::default(); 16];
    for (i, p) in planets.iter().enumerate().take(16) {
      planet_gpu[i] = crate::gpu::MinimapPlanetGpu {
        pos: [p.0.x(), p.0.y()],
        size: p.1,
        _pad: 0.0,
        color: p.2,
      };
    }
    let planets_ptr = self.upload_minimap_planets(cmd_buffer, &planet_gpu[..planet_count])?;

    let push = crate::gpu::MinimapPushConstants {
      offset: [0.7f32, 0.7f32],
      size: [0.25f32, 0.25f32 * aspect_ratio],
      player_pos: [player_pos.x(), player_pos.y()],
      max_distance,
      num_planets: planet_count as u32,
      planets_ptr,
      _pad: 0,
    };

    unsafe {
      let push_bytes = core::slice::from_raw_parts(
        &push as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::MinimapPushConstants>(),
      );
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_bytes,
      );

      self.device.cmd_draw(cmd, 4, 1, 0, 0);
    }
    Ok(())
  }

  #[named]
  fn render_ui_rect(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let (pipeline_key, layout, set) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(pe_lock, handle)?;
      let text_render_archetype_guard =
        DebugTrackedRwLock::read(&pe.archetypes().text_render_archetype);
      let archetype = text_render_archetype_guard.as_ref().ok_or(gpu_err_archetype_absent!())?;
      let layout = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .pipeline_layout
      .get();
      let set = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &*archetype.deref_arena().ok_or(crate::gpu_err_device!())?,
      )
      .descriptor_set
      .ok_or(gpu_err!("descriptor set not found"))?;
      (archetype.pipeline_key, layout, set)
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    unsafe {
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

  #[named]
  fn render_text(
    &self,
    cmd_buffer: CommandBufferHandle,
    text: &str,
    start_cursor_position: [f32; 2],
    view_proj: [f32; 16],
    atlas_id: (u64, u32),
    desired_points: f32,
    color: [f32; 4],
  ) -> GpuResult<()> {
    let (cmd, _) = self.get_cmd_and_pe(cmd_buffer)?;

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        let arena_arc =
          state.text_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
        let arena =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&**arena_arc);

        let font = arena
          .uploaded_fonts
          .get(&atlas_id.0)
          .ok_or(gpu_invalid_arg!("inexistent font, {:?}", atlas_id))?;

        if atlas_id.1 != font.descriptor_index {
          return Err(gpu_invalid_arg!(
            "inconsistent descriptor index: {} vs {}",
            atlas_id.1,
            font.descriptor_index
          ));
        }

        let pipeline_layout = arena.pipeline_layout.get();
        let default_glyph = font
          .atlas
          .glyphs
          .get(&'?')
          .ok_or(gpu_err!("text atlas doesn't have default \"?\" glyph"))?;

        let mut cursor_x = start_cursor_position[0];
        let mut cursor_y = start_cursor_position[1];

        // Pre-calculate all push constants to avoid locking during Vulkan calls
        let mut push_constants_list = alloc::vec::Vec::with_capacity(text.len());

        for c in text.chars() {
          if c == '\n' {
            cursor_x = start_cursor_position[0];
            cursor_y += font.atlas.scaled_height(desired_points) * 1.5;
            continue;
          }

          let glyph = font.atlas.glyphs.get(&c).unwrap_or(default_glyph);
          let push_constants = crate::gpu::TextPushConstants::from_glyph(
            glyph,
            [cursor_x, cursor_y],
            view_proj,
            desired_points,
            font.atlas.scale,
            atlas_id.1,
            color,
          );

          push_constants_list.push(push_constants);
          cursor_x += glyph.scaled_advance(desired_points, font.atlas.scale);
        }

        Ok((pipeline_layout, push_constants_list))
      })?
      .execute(|(pipeline_layout, push_constants_list), _rollback| {
        // Execute all Vulkan commands completely lock-free
        unsafe {
          for push_constants in push_constants_list {
            let push_bytes = core::slice::from_raw_parts(
              &push_constants as *const _ as *const u8,
              core::mem::size_of::<crate::gpu::TextPushConstants>(),
            );

            self.device.cmd_push_constants(
              cmd,
              pipeline_layout,
              ash::vk::ShaderStageFlags::VERTEX | ash::vk::ShaderStageFlags::FRAGMENT,
              0,
              push_bytes,
            );

            self.device.cmd_draw(cmd, 4, 1, 0, 0);
          }
        }
        Ok(())
      })
      .commit_read(|_state, execute_result| execute_result)
  }

  fn next_subpass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;
    let subpass_begin_info = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);
    let subpass_end_info = vk::SubpassEndInfo::default();
    unsafe {
      self
        .device
        .create_renderpass2
        .cmd_next_subpass2(cmd, &subpass_begin_info, &subpass_end_info);
    }

    // Advance compositing context subpass index
    {
      let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
      if let Some(data) = cmd_buffers.get_mut(&cmd_buffer) {
        if let Some(ref mut ctx) = data.compositing_ctx {
          ctx.subpass += 1;
        }
      }
    }

    Ok(())
  }

  fn draw_composite(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: crate::gpu::PresentationEngineHandle,
    constants: &crate::gpu::CompositePushConstants,
  ) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    // Get composite pipeline resources from the render pass bundle
    let res_guard = DebugTrackedRwLock::read(&self.res);
    let (descriptor_set, pipeline_layout, pipeline_key) = res_guard
      .renderpasses
      .get_composite_resources(handle)
      .ok_or(crate::gpu_err!("composite resources not initialized"))?;

    // Get the cached composite pipeline
    let pipeline = res_guard
      .pipeline_pool
      .get_graphics_pipeline(pipeline_key)
      .ok_or(crate::gpu_err!("composite pipeline not found"))?;

    unsafe {
      // Bind composite pipeline
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());

      // Bind descriptor set with input attachments
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline_layout,
        0, // first set
        &[descriptor_set],
        &[], // no dynamic offsets
      );

      // Push the near/far constants
      let push_data = core::slice::from_raw_parts(
        constants as *const crate::gpu::CompositePushConstants as *const u8,
        core::mem::size_of::<crate::gpu::CompositePushConstants>(),
      );
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::FRAGMENT,
        0,
        push_data,
      );

      // Draw fullscreen triangle (3 vertices, no vertex buffer)
      self.device.cmd_draw(cmd, 3, 1, 0, 0);
    }

    Ok(())
  }

  #[named]
  fn end_render_pass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let cmd = self.get_cmd(cmd_buffer)?;

    let subpass_end_info = vk::SubpassEndInfo::default();

    unsafe {
      self.device.create_renderpass2.cmd_end_render_pass2(cmd, &subpass_end_info);
    }

    // Clear compositing context
    {
      let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
      if let Some(data) = cmd_buffers.get_mut(&cmd_buffer) {
        data.compositing_ctx = None;
      }
    }

    Ok(())
  }

  #[named]
  fn record_windowless_download(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    task_id: u64,
  ) -> GpuResult<()> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let acquire_result = {
      let cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &self.recording_command_buffers,
      );
      let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
      data.presentation.ok_or(gpu_err_cmd_no_pe!())?.acquire_result
    };

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        let engine_lock = &state.live_presentation_engines;
        let pe = wait_for_pe_direct!(engine_lock, handle)?;

        let (image, width, height) =
          if let swapchain::PresentationState::Windowless(windowless) = &*pe {
            let (img, _, _) =
              unsafe { windowless.get_image_resources(acquire_result.image_index as usize) };
            (img.get(), windowless.extent().0, windowless.extent().1)
          } else {
            return Err(gpu_err!("presentation engine is not windowless"));
          };

        let vma = state.allocator.allocator.as_allocator_view();

        Ok::<_, GpuError>((image, width, height, vma))
      })?
      .execute(|(image, width, height, vma), rollback| {
        let allocator = vma;
        let buffer_size = (width * height * 4) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
          .size(buffer_size)
          .usage(vk::BufferUsageFlags::TRANSFER_DST);

        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags =
          vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
        crate::apply_test_dedicated_alloc!(alloc_info);

        // Lock-free allocation
        let (staging_buffer, alloc) =
          unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }?;

        // Protect against aborts
        let mut alloc_mut = alloc;
        rollback.defer(move |_dev| unsafe {
          allocator.destroy_buffer(staging_buffer, &mut alloc_mut);
        });

        // Record Vulkan Commands lock-free
        unsafe {
          let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .image(image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
            );

          let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(core::slice::from_ref(&image_barrier));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

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
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[region],
          );

          let image_barrier_back = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
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
            .buffer_memory_barriers(core::slice::from_ref(&buffer_barrier))
            .image_memory_barriers(core::slice::from_ref(&image_barrier_back));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &buf_dep_info);
        }

        Ok((staging_buffer, alloc, buffer_size as usize))
      })
      .commit_read(|state, execute_result| {
        let (staging_buffer, allocation, size) = execute_result?;

        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &state.pending_downloads,
        )
        .insert(
          task_id,
          PendingDownload {
            staging_buffer,
            allocation,
            size,
            presentation_engine: Some(handle),
          },
        );

        Ok(())
      })?;

    Ok(())
  }

  /// Reads the downloaded image for the given `task_id` into `buffer`.
  ///
  /// **Note:** To prevent memory leaks when frames are skipped and not downloaded,
  /// this function automatically cleans up and deallocates any pending downloads for the same
  /// presentation engine that have a `task_id` strictly less than the provided `task_id`.
  /// Therefore, `task_id`s must be strictly increasing across sequential downloads for a given engine.
  #[named]
  fn read_windowless_download(&self, task_id: u64, buffer: &mut [u8]) -> GpuResult<()> {
    // 1. Verify task completion completely lock-free
    if !self.is_task_completed(task_id)? {
      return Err(crate::gpu_err_device!());
    }

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        // 2. Briefly lock to extract the pending download from the Hashmap
        let mut pending_lock =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
            &state.pending_downloads,
          );

        let download = pending_lock.remove(&task_id).ok_or(gpu_invalid_arg!(
          "Invalid or previously consumed download ID: {}",
          task_id
        ))?;

        // Automatically cleanup older downloads for the same presentation engine
        if let Some(engine_handle) = download.presentation_engine {
          let to_remove: Vec<u64> = pending_lock
            .iter()
            .filter_map(|(&tid, dl)| {
              if tid < task_id && dl.presentation_engine == Some(engine_handle) {
                Some(tid)
              } else {
                None
              }
            })
            .collect();

          for tid in to_remove {
            if let Some(mut old_dl) = pending_lock.remove(&tid) {
              #[cfg(test)]
              {
                aethervk_oshal_rlib::log!(
                  "Automatically cleaning up un-consumed download: {} (engine {:?})",
                  tid,
                  engine_handle
                );
              }
              unsafe {
                state
                  .allocator
                  .allocator
                  .as_allocator_view()
                  .destroy_buffer(old_dl.staging_buffer, &mut old_dl.allocation);
              }
            }
          }
        }

        // Create a Copy-able view of the VMA allocator to bypass Drop limitations
        let vma_view = state.allocator.allocator.as_allocator_view();

        Ok((download, vma_view))
      })?
      .execute(|(download, vma_view), _rollback| {
        // 3. RAII guard guarantees the staging buffer is destroyed no matter how the closure exits
        struct StagingCleanup {
          allocator: vk_mem::AllocatorView,
          buffer: vk::Buffer,
          allocation: vk_mem::Allocation,
        }
        impl Drop for StagingCleanup {
          fn drop(&mut self) {
            unsafe {
              self.allocator.destroy_buffer(self.buffer, &mut self.allocation);
            }
          }
        }

        let _cleanup = StagingCleanup {
          allocator: vma_view,
          buffer: download.staging_buffer,
          allocation: download.allocation,
        };

        let alloc_info = vma_view.get_allocation_info(&download.allocation);
        let mapped_ptr = alloc_info.mapped_data as *const u8;

        if !mapped_ptr.is_null() {
          // 4. Lock-free VMA cache invalidation
          vma_view.invalidate_allocation(&download.allocation, 0, vk::WHOLE_SIZE)?;

          let copy_size = core::cmp::min(buffer.len(), download.size);

          // 5. Heavy memory copy operation executed lock-free!
          unsafe {
            core::ptr::copy_nonoverlapping(mapped_ptr, buffer.as_mut_ptr(), copy_size);
          }
        } else {
          return Err(crate::gpu_err_device!());
        }

        Ok(())
      })
      .commit_read(|_state, execute_result| execute_result)
  }

  #[named]
  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
    sync_info: Option<crate::gpu::CommandBufferSyncInfo>,
  ) -> GpuResult<()> {
    // 1. Extract command buffer data and drop the lock immediately
    let data = {
      let mut cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
        &self.recording_command_buffers,
      );
      cmd_buffers.remove(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?
    };

    unsafe {
      self.device.end_command_buffer(data.command_buffer.get())?;
    }

    let presentation = data.presentation.ok_or(gpu_err_cmd_no_pe!())?;
    let graphics_queue = self.queues.get_graphics_queue();

    // 2. Start Vulkan Transaction for the submission process
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(presentation.presentation_engine, |state, pe_handle| {
        let pe = wait_for_pe!(state, pe_handle)?;

        let is_resize_required = pe.swapchain_generation() != presentation.swapchain_generation;

        let timeline_sem = state.timeline_manager.semaphore.get();

        let cmd_pools = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
          &state.command_pools,
        )
        .get(graphics_queue.index as usize)
        .and_then(|opt| opt.as_ref())
        .cloned()
        .ok_or(gpu_err!("couldn't get command pools"))?;

        let task_registry = state.timeline_manager.task_registry.clone();
        let timeline_manager_ptr =
          &state.timeline_manager as *const timeline_manager::TimelineManager;

        Ok((
          pe_handle,
          is_resize_required,
          timeline_sem,
          cmd_pools,
          task_registry,
          timeline_manager_ptr,
        ))
      })?
      .execute(
        |(
          pe_handle,
          is_resize_required,
          timeline_sem,
          cmd_pools,
          task_registry,
          timeline_manager_ptr,
        ),
         _rollback| {
          let timeline_manager = unsafe { &*timeline_manager_ptr };

          let mut signal_semaphores = heapless::Vec::<_, 4>::new();

          if let Some(sem) = presentation.signal_semaphore {
            unsafe {
              signal_semaphores.push_unchecked(sem.get());
            }
          }

          let mut wait_semaphores = heapless::Vec::<_, 4>::new();
          let mut wait_semaphore_values = heapless::Vec::<_, 4>::new();
          let mut wait_dst_stage_mask = heapless::Vec::<_, 4>::new();

          if let Some(wait_semaphore) = presentation.wait_semaphore {
            unsafe {
              wait_semaphores.push_unchecked(wait_semaphore.get());
              wait_semaphore_values.push_unchecked(0);
              wait_dst_stage_mask.push_unchecked(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
            }
          }

          if let Some(sync) = sync_info {
            use ash::vk::Handle;
            let vk_semaphore = vk::Semaphore::from_raw(sync.timeline_semaphore);
            unsafe {
              wait_semaphores.push_unchecked(vk_semaphore);
              wait_semaphore_values.push_unchecked(sync.timeline_value);
              // Graphics reads compute buffers at Vertex Input / Compute shader stages
              wait_dst_stage_mask.push_unchecked(
                vk::PipelineStageFlags::VERTEX_INPUT | vk::PipelineStageFlags::COMPUTE_SHADER,
              );
            }
          }

          let command_buffers = [data.command_buffer.get()];

          // TAKE SUBMISSION LOCK BEFORE ALLOCATING TIMELINE!
          // This ensures that the order we get timeline values exactly matches the order we submit to the queue.
          let next_timeline_value = {
            let _guard = self.device.submission_lock.lock();

            let next_timeline_value = timeline_manager.allocate_submit_value();

            let mut timeline_values = heapless::Vec::<_, 4>::new();
            if presentation.signal_semaphore.is_some() {
              unsafe {
                timeline_values.push_unchecked(0);
              }
            }
            unsafe {
              signal_semaphores.push_unchecked(timeline_sem);
              timeline_values.push_unchecked(next_timeline_value);
            }

            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
              .wait_semaphore_values(&wait_semaphore_values)
              .signal_semaphore_values(&timeline_values);

            let submit_info = vk::SubmitInfo::default()
              .wait_semaphores(&wait_semaphores)
              .wait_dst_stage_mask(&wait_dst_stage_mask)
              .command_buffers(&command_buffers)
              .signal_semaphores(&signal_semaphores)
              .push_next(&mut timeline_info);

            unsafe {
              self
                .device
                .handle
                .queue_submit(
                  graphics_queue.handle,
                  &[submit_info],
                  presentation.submission_fence.get(),
                )
                .map_err(|e| {
                  aethervk_oshal_rlib::log!("Queue submit failed: {:?}", e);
                  GpuError::from(e)
                })?;
            }

            next_timeline_value
          };

          // Inform the task registry of the timeline value to wait for
          if let Some(tid) = task_id {
            let registry =
              crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&task_registry);
            if let Some(entry) = registry.get(&tid) {
              entry
                .target_value
                .store(next_timeline_value, core::sync::atomic::Ordering::Release);
            }
          }

          // Pass 'data' through to be discarded in the commit phase
          Ok((
            data,
            pe_handle,
            is_resize_required,
            next_timeline_value,
            cmd_pools,
          ))
        },
      )
      .commit_read(|state, execute_result| {
        let (mut data, pe_handle, is_resize_required, next_timeline_value, cmd_pools) =
          execute_result?;

        if let Some(mut pe) = state.live_presentation_engines.get_mut(&pe_handle) {
          pe.mark_fence_submitted(data.presentation.unwrap().acquire_result.frame_index as u32);
          if let swapchain::PresentationState::Windowless(windowless) = pe.value() {
            windowless
              .last_timeline_value
              .store(next_timeline_value, core::sync::atomic::Ordering::Release);
          }
        }

        // Discard resources now that submission is safely recorded
        data.discard(
          cmd_buffer.into(),
          &state.discard_pool,
          cmd_pools,
          next_timeline_value,
        );

        if is_resize_required {
          Err(GpuError::ResizeRequired)
        } else {
          Ok(())
        }
      })
  }

  #[named]
  fn wire_callbacks(&self, pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) -> GpuResult<()> {
    let workload = DebugTrackedRwLock::read(&self.res)
      .timeline_manager
      .create_polling_workload(Arc::clone(&self.callback_stop_signal));
    pool.scatter(vec![Box::new(workload)]).map_err(|_| crate::gpu_err_device!())?;
    Ok(())
  }

  #[named]
  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool> {
    DebugTrackedRwLock::read(&self.res).timeline_manager.is_task_completed(task_id)
  }

  #[named]
  fn create_task(&self) -> u64 {
    DebugTrackedRwLock::read(&self.res).timeline_manager.create_task()
  }

  #[named]
  fn fail_task(&self, task_id: u64, error: GpuError) {
    DebugTrackedRwLock::read(&self.res).timeline_manager.fail_task(task_id, error)
  }

  #[named]
  fn success_task(&self, task_id: u64) {
    DebugTrackedRwLock::read(&self.res).timeline_manager.success_task(task_id)
  }

  #[named]
  fn prepare_background_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let pipeline_key = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe_lock = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(pe_lock, handle)?;
      let archetype_guard = DebugTrackedRwLock::read(&pe.archetypes().background_render_archetype);
      if archetype_guard.is_none() {
        return Err(gpu_err_archetype_absent!());
      }
      let archetype = unsafe { archetype_guard.as_ref().unwrap_unchecked() };
      archetype.pipeline_key
    };

    self.bind_pipeline(cmd_buffer, pipeline_key)?;
    // No descriptor sets for background archetype
    Ok(())
  }

  fn clear_depth(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> Result<(), crate::types::GpuError> {
    let extent = self.get_presentation_engine_extent(handle)?;
    let clear_value = ash::vk::ClearValue {
      depth_stencil: ash::vk::ClearDepthStencilValue {
        depth: 0.0,
        stencil: 0,
      },
    };
    let clear_attachment = ash::vk::ClearAttachment {
      aspect_mask: ash::vk::ImageAspectFlags::DEPTH,
      color_attachment: 0,
      clear_value,
    };
    let clear_rect = ash::vk::ClearRect {
      rect: ash::vk::Rect2D {
        offset: ash::vk::Offset2D { x: 0, y: 0 },
        extent: ash::vk::Extent2D {
          width: extent[0],
          height: extent[1],
        },
      },
      base_array_layer: 0,
      layer_count: 1,
    };
    let cmd = self.get_cmd(cmd_buffer)?;
    unsafe {
      self.device.cmd_clear_attachments(cmd, &[clear_attachment], &[clear_rect]);
    }
    Ok(())
  }
}

pub(super) struct TransientCmdPoolResource {
  pub(super) pool: ash::vk::CommandPool,
  pub(super) cmd: ash::vk::CommandBuffer,
}
impl DeviceResource for TransientCmdPoolResource {
  fn cleanup(&mut self, device: &ash::Device) {
    aethervk_oshal_rlib::log!("Destroying TransientCmdPoolResource");
    unsafe {
      device.free_command_buffers(self.pool, &[self.cmd]);
      device.destroy_command_pool(self.pool, None);
    }
  }
}

impl Device {
  #[named]
  pub(super) fn run_transient_commands<F>(&self, f: F) -> GpuResult<TransientCmdPoolResource>
  where
    F: FnOnce(vk::CommandBuffer) -> GpuResult<()>,
  {
    aethervk_oshal_rlib::log!("run_transient_commands called!");
    let queue = self.queues.get_graphics_queue();
    let pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(queue.family_index)
      .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { self.device.create_command_pool(&pool_info, None) }?;

    let alloc_info = vk::CommandBufferAllocateInfo::default()
      .command_pool(pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);
    let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0];

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { self.device.begin_command_buffer(cmd, &begin_info) }?;

    if let Err(e) = f(cmd) {
      aethervk_oshal_rlib::log!("run_transient_commands error: {:?}", e);
      unsafe {
        self.device.free_command_buffers(pool, &[cmd]);
        self.device.destroy_command_pool(pool, None);
      }
      return Err(e);
    }

    unsafe { self.device.end_command_buffer(cmd) }?;

    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
      .semaphore_type(vk::SemaphoreType::TIMELINE)
      .initial_value(0);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
    let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_info, None) }?;
    
    let signal_semaphores = [timeline_semaphore];
    let signal_values = [1];
    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
      .signal_semaphore_values(&signal_values);

    let submit_info = vk::SubmitInfo::default()
      .command_buffers(core::slice::from_ref(&cmd))
      .signal_semaphores(&signal_semaphores)
      .push_next(&mut timeline_info);

    self
      .device
      .locked_queue_submit(queue.handle, &[submit_info], vk::Fence::null())
      .map_err(GpuError::from)?;
      
    self.device.wait_for_semaphore_value(timeline_semaphore, 1, u64::MAX)?;
    unsafe {
      self.device.destroy_semaphore(timeline_semaphore, None);
    }

    Ok(TransientCmdPoolResource { pool, cmd })
  }

  #[named]
  fn begin_render_pass_impl(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
    compositing: bool,
  ) -> GpuResult<()> {
    let (
      timeline,
      cmd,
      vma,
      discard_pool_ptr,
      renderpasses_ptr,
      pipeline_pool_ptr,
      render_pass_spec,
    ) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let _presentation_engines_guard = &res_guard.live_presentation_engines;
      let cmd_buffers = DebugTrackedRwLock::read(&self.recording_command_buffers);
      if !cmd_buffers.contains_key(&cmd_buffer) {
        return Err(gpu_err_invalid_cmd!());
      }
      let wpresentation_engine = wait_for_pe!(res_guard, presentation_engine)?;

      let data = unsafe { cmd_buffers.get(&cmd_buffer).unwrap_unchecked() };
      if !data.has_begun {
        return Err(gpu_err!("command buffer not begun"));
      }

      if acquire_result.status.needs_resize() {
        return Err(GpuError::ResizeRequired);
      }
      if wpresentation_engine.swapchain_generation() != acquire_result.swapchain_generation {
        return Err(GpuError::ResizeRequired);
      }
      drop(cmd_buffers);

      let (wait_semaphore, submission_fence) =
        unsafe { wpresentation_engine.get_frame_resources(acquire_result.frame_index as usize) };
      let (_, _, signal_semaphore) =
        unsafe { wpresentation_engine.get_image_resources(acquire_result.image_index as usize) };

      let timeline = res_guard.timeline_manager.get_next_submit_value() - 1;

      let cmd = {
        let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
        let data = unsafe { cmd_buffers.get_mut(&cmd_buffer).unwrap_unchecked() };
        data.presentation = Some(RecordingCmdBufferDataPresentation {
          acquire_result: *acquire_result,
          presentation_engine,
          swapchain_generation: acquire_result.swapchain_generation,
          wait_semaphore,
          signal_semaphore,
          submission_fence,
        });
        data.command_buffer.get()
      };

      let vma = res_guard.allocator.allocator.get_raw();
      let discard_pool_ptr = &res_guard.discard_pool as *const _;
      let renderpasses_ptr = &res_guard.renderpasses as *const renderpasses::RenderPasses;
      let pipeline_pool_ptr = &res_guard.pipeline_pool as *const pipelines::PipelinePool;
      let render_pass_spec = if compositing {
        RenderPassSpecification::compositing_pass(&wpresentation_engine, self.depth_stencil_format)
      } else {
        RenderPassSpecification::single_pass(&wpresentation_engine, self.depth_stencil_format)
      };

      (
        timeline,
        cmd,
        vma,
        discard_pool_ptr,
        renderpasses_ptr,
        pipeline_pool_ptr,
        render_pass_spec,
      )
    };

    let allocator = unsafe { vk_mem::AllocatorView::from_raw(vma) };
    let discard_pool = unsafe { &*discard_pool_ptr };
    let renderpasses = unsafe { &*renderpasses_ptr };
    let pipeline_pool = unsafe { &*pipeline_pool_ptr };

    let (render_pass, framebuffer) = renderpasses.get_or_create_render_pass(
      presentation_engine,
      render_pass_spec.clone(),
      acquire_result.image_index,
      &self.device,
      allocator,
      discard_pool,
      timeline,
    )?;

    // Initialize composite pipeline resources on first use
    if compositing {
      renderpasses.init_composite_pipeline(presentation_engine, &self.device, pipeline_pool)?;
    }

    let extent = render_pass_spec.extent();

    // Get clear values — 6 for compositing, 2 for single-subpass
    let num_clear = render_pass_spec.num_attachments();
    let mut clear_values = [vk::ClearValue::default(); renderpasses::MAX_ATTACHMENTS];
    renderpasses
      .get_clear_values_render_pass(presentation_engine, &mut clear_values[..num_clear])?;

    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
      .render_pass(render_pass.get())
      .framebuffer(framebuffer.get())
      .render_area(vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
          width: extent.0,
          height: extent.1,
        },
      })
      .clear_values(&clear_values[..num_clear]);
    let subpass_begin_info = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);

    unsafe {
      self.device.create_renderpass2.cmd_begin_render_pass2(
        cmd,
        &render_pass_begin_info,
        &subpass_begin_info,
      )
    };

    // Set compositing context on the command buffer so bind_pipeline
    // can transparently create compositing-compatible pipeline variants
    if compositing {
      let mut cmd_buffers = DebugTrackedRwLock::write(&self.recording_command_buffers);
      if let Some(data) = cmd_buffers.get_mut(&cmd_buffer) {
        data.compositing_ctx = Some(CompositingContext {
          render_pass: render_pass.get(),
          subpass: 0, // Start at subpass 0 (macro)
          pe_handle: presentation_engine,
        });
      }
    }

    Ok(())
  }

  #[named]
  fn get_cmd(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
  ) -> GpuResult<ash::vk::CommandBuffer> {
    let cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
      &self.recording_command_buffers,
    );
    let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
    Ok(data.command_buffer.get())
  }

  #[named]
  fn get_cmd_and_pe(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
  ) -> GpuResult<(ash::vk::CommandBuffer, crate::gpu::PresentationEngineHandle)> {
    let cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
      &self.recording_command_buffers,
    );
    let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
    let handle = data.presentation_engine.ok_or(gpu_err_cmd_no_pe!())?;
    Ok((data.command_buffer.get(), handle))
  }

  /// TODO: Document this item
  pub fn get_vma_budget_usage(&self) -> (u64, u64) {
    let mut res = DebugTrackedRwLock::write(&self.res);
    res.allocator.refresh_vma_budgets();
    // VmaBudget struct has 'budget' and 'usage' properties, both vk::DeviceSize
    let mut total_budget = 0;
    let mut total_usage = 0;
    // Assuming we can access the underlying slice of memory_budgets
    for budget in res.allocator.memory_budgets.iter() {
      total_budget += budget.budget;
      total_usage += budget.usage;
    }
    (total_budget, total_usage)
  }

  #[cfg(test)]
  #[named]
  fn record_test_depth_stencil_download(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
    task_id: u64,
  ) -> GpuResult<()> {
    // 1. Extract command buffer and acquire result
    let (cmd, _) = {
      let cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &self.recording_command_buffers,
      );
      let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
      let presentation = data.presentation.ok_or(gpu_err_cmd_no_pe!())?;
      (data.command_buffer.get(), presentation.acquire_result)
    };

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        // 2. Fetch Presentation Engine Extent
        let engine_lock = &state.live_presentation_engines;
        let pe = wait_for_pe_direct!(engine_lock, h)?;
        let (width, height) = pe.extent();

        // 3. Fetch Image
        let depth_image = state.renderpasses.get_test_depth_stencil_image(h).unwrap();

        // 4. Extract safe VMA view
        let vma_view = state.allocator.allocator.as_allocator_view();

        Ok::<_, GpuError>((depth_image.get(), width, height, vma_view))
      })?
      .execute(|(depth_image, width, height, vma_view), rollback| {
        // even with format D24, when copy to buffer, depth aspect's format in D24_UNORM_S8_UINT
        // is equivalent to X8_D24_UNORM_PACK32 (meaning 4 bytes) (see docs, 1.4, chapter 56)
        let depth_size = width * height * 4;
        let stencil_size = width * height * 1;
        let buffer_size = (depth_size + stencil_size) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
          .size(buffer_size)
          .usage(vk::BufferUsageFlags::TRANSFER_DST);

        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        crate::apply_test_dedicated_alloc!(alloc_info);
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags =
          vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;

        // 5. Lock-free Memory Allocation
        let (staging_buffer, alloc) = unsafe { vma_view.create_buffer(&buffer_info, &alloc_info) }?;

        // Register with Rollback in case of aborts
        let mut alloc_mut = alloc;
        rollback.defer(move |_dev| unsafe {
          vma_view.destroy_buffer(staging_buffer, &mut alloc_mut);
        });

        // 6. Record Vulkan Commands Lock-free
        unsafe {
          let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
              vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(
              vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(depth_image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
            );
          let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(core::slice::from_ref(&image_barrier));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

          let depth_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .image_subresource(
              vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
            )
            .image_extent(vk::Extent3D {
              width,
              height,
              depth: 1,
            });
          let stencil_region = vk::BufferImageCopy::default()
            .buffer_offset(depth_size as u64)
            .image_subresource(
              vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::STENCIL)
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
            depth_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[depth_region, stencil_region],
          );

          // image barrier for the next frame
          let image_barrier_back = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(
              vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            )
            .dst_access_mask(
              vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            )
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(depth_image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
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
            .buffer_memory_barriers(core::slice::from_ref(&buffer_barrier))
            .image_memory_barriers(core::slice::from_ref(&image_barrier_back));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &buf_dep_info);
        }

        Ok((staging_buffer, alloc, buffer_size as usize))
      })
      .commit_read(|state, execute_result| {
        let (staging_buffer, allocation, size) = execute_result?;

        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &state.pending_downloads,
        )
        .insert(
          task_id,
          PendingDownload {
            staging_buffer,
            allocation,
            size,
            presentation_engine: Some(handle),
          },
        );

        Ok(())
      })?;

    Ok(())
  }

  #[cfg(test)]
  #[named]
  fn record_test_sun_download(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    task_id: u64,
  ) -> GpuResult<()> {
    // 1. Extract command buffer
    let cmd = {
      let cmd_buffers = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &self.recording_command_buffers,
      );
      let data = cmd_buffers.get(&cmd_buffer).ok_or(gpu_err_invalid_cmd!())?;
      data.command_buffer.get()
    };

    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        // 2. Fetch Sun Image and Resolution
        let sun_map = &state.sun_resources;
        let sun_res_ref = sun_map
          .get(&entity_id)
          .ok_or(gpu_invalid_arg!("invalid sun entity: {:?}", entity_id))?;
        let sun_res = match sun_res_ref.value() {
          resources::ResourceState::Ready(r) => r,
          _ => return Err(gpu_err!("sun resource not ready")),
        };
        let sun_image = sun_res
          .image
          .as_ref()
          .ok_or(gpu_err!("sun resource doesn't have image"))?
          .image
          .get();

        let (width, height, depth) = sun_res.resolution;

        // 3. Extract safe VMA view
        let vma_view = state.allocator.allocator.as_allocator_view();

        Ok((sun_image, width, height, depth, vma_view))
      })?
      .execute(|(sun_image, width, height, depth, vma_view), rollback| {
        // R16G16B16A16_SFLOAT is 8 bytes per texel
        let buffer_size = (width * height * depth * 8) as vk::DeviceSize;

        let buffer_info = vk::BufferCreateInfo::default()
          .size(buffer_size)
          .usage(vk::BufferUsageFlags::TRANSFER_DST);

        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags =
          vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
        crate::apply_test_dedicated_alloc!(alloc_info);

        // 4. Lock-free Memory Allocation
        let (staging_buffer, alloc) = unsafe { vma_view.create_buffer(&buffer_info, &alloc_info) }?;

        // Register with Rollback in case of aborts
        let mut alloc_mut = alloc;
        rollback.defer(move |_dev| unsafe {
          vma_view.destroy_buffer(staging_buffer, &mut alloc_mut);
        });

        // 5. Record Vulkan Commands Lock-free
        unsafe {
          let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(
              vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            )
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(sun_image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
            );
          let dep_info = vk::DependencyInfo::default()
            .image_memory_barriers(core::slice::from_ref(&image_barrier));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

          let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
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
              depth,
            });

          self.device.cmd_copy_image_to_buffer(
            cmd,
            sun_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            staging_buffer,
            &[copy_region],
          );

          let image_barrier_back = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(sun_image)
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
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
            .buffer_memory_barriers(core::slice::from_ref(&buffer_barrier))
            .image_memory_barriers(core::slice::from_ref(&image_barrier_back));
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &buf_dep_info);
        }

        Ok::<_, GpuError>((staging_buffer, alloc, buffer_size as usize))
      })
      .commit_read(|state, execute_result| {
        let (staging_buffer, allocation, size) = execute_result?;

        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &state.pending_downloads,
        )
        .insert(
          task_id,
          PendingDownload {
            staging_buffer,
            allocation,
            size,
            presentation_engine: None,
          },
        );

        Ok(())
      })?;

    Ok(())
  }

  #[cfg(test)]
  fn separate_depth_stencil(&self, buffer: &[u8], width: u32, height: u32) -> (Vec<f32>, Vec<u8>) {
    let depth_size = (width * height * 4) as usize; // we'll adapt it to float
    let stencil_size = (width * height * 1) as usize;
    assert_eq!(buffer.len(), depth_size + stencil_size);

    let mut depth_buffer = Vec::with_capacity((width * height) as usize);
    let mut stencil_buffer = Vec::with_capacity((width * height) as usize);

    if self.depth_stencil_format == vk::Format::D24_UNORM_S8_UINT {
      for i in 0..(width * height) as usize {
        let val_bytes = [
          buffer[i * 4],
          buffer[i * 4 + 1],
          buffer[i * 4 + 2],
          buffer[i * 4 + 3],
        ];
        let val = u32::from_le_bytes(val_bytes);
        // Convert [0, 2^24-1] to [0.0, 1.0]
        depth_buffer.push((val & 0xFFFFFF) as f32 / 16777215.0);
      }
    } else {
      // D32_SFLOAT_S8_UINT
      for i in 0..(width * height) as usize {
        let val_bytes = [
          buffer[i * 4],
          buffer[i * 4 + 1],
          buffer[i * 4 + 2],
          buffer[i * 4 + 3],
        ];
        depth_buffer.push(f32::from_le_bytes(val_bytes));
      }
    }

    for i in 0..(width * height) as usize {
      stencil_buffer.push(buffer[depth_size + i]);
    }

    (depth_buffer, stencil_buffer)
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

fn ensure_text2_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("text2.vert.spv");
  frag_path = assets_dir.join("text2.frag.spv");

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

fn ensure_physical_mesh2_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey, ShaderKey, ShaderKey)> {
  let vkey = shader_manager
    .get_or_load(
      device,
      &oshal::os::fs::PathBuf::from(format!(
        "{}/physical_mesh2.vert.spv",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      )),
      "main",
      spirv::ExecutionModel::Vertex,
    )
    .unwrap();
  let fkey = shader_manager
    .get_or_load(
      device,
      &oshal::os::fs::PathBuf::from(format!(
        "{}/physical_mesh2.frag.spv",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      )),
      "main",
      spirv::ExecutionModel::Fragment,
    )
    .unwrap();
  let ovkey = shader_manager
    .get_or_load(
      device,
      &oshal::os::fs::PathBuf::from(format!(
        "{}/physical_mesh2_outline.vert.spv",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      )),
      "main",
      spirv::ExecutionModel::Vertex,
    )
    .unwrap();
  let ofkey = shader_manager
    .get_or_load(
      device,
      &oshal::os::fs::PathBuf::from(format!(
        "{}/physical_mesh2_outline.frag.spv",
        crate::gpu::ASSET_DIR.read().as_ref().unwrap()
      )),
      "main",
      spirv::ExecutionModel::Fragment,
    )
    .unwrap();
  Ok((vkey, fkey, ovkey, ofkey))
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

fn ensure_background_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("background.vert.spv");
  frag_path = assets_dir.join("background.frag.spv");

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

fn ensure_particle2_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(ShaderKey, ShaderKey)> {
  let vert_path: PathBuf;
  let frag_path: PathBuf;

  let assets_dir = shaders_asset_dir()?;

  vert_path = assets_dir.join("particle2.vert.spv");
  frag_path = assets_dir.join("particle2.frag.spv");

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

fn ensure_trajectory_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("trajectory.vert.spv");
  frag_path = assets_dir.join("trajectory.frag.spv");

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

fn ensure_ui_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("ui.vert.spv");
  frag_path = assets_dir.join("ui.frag.spv");

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

fn ensure_bvhwire2_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("bvhwire2.vert.spv");
  frag_path = assets_dir.join("bvhwire2.frag.spv");

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

fn ensure_sphere_gizmo_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("sphere_gizmo.vert.spv");
  frag_path = assets_dir.join("sphere_gizmo.frag.spv");

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
  props: &vk::PhysicalDeviceProperties,
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

// TODO RE-ENABLE
// #[cfg(test)]
// mod test_render;

#[cfg(test)]
mod test_swapchain;

#[cfg(test)]
mod test_ui_text;
