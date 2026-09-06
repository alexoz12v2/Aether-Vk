//! device module.
//! Only `RenderDevice` methods are allowed to start a Vulkan Transaction
//!
//! # Troubleshooting File Descriptor Leaks
//! Unlike memory leaks that cause rapid Out-of-Memory (OOM) crashes, an FD leak is stealthy.
//! Your application works normally at first but crashes after a sustained period of heavy queue
//! submissions or device recreation (like during `cargo nextest` runs).
//! The crash log outputs standard OS errors like `Too many open files` (error 24) or Vulkan
//! failures like `VK_ERROR_INITIALIZATION_FAILED`.
//!
//! ## How to Fix or Workaround
//! 1. **Pinpoint the Leaking Objects**: Verify the layer is causing the leak via `lsof -p <PID> | wc -l`.
//!    If the count stops growing when GPU-AV is disabled, it's a validation layer issue.
//! 2. **Avoid Device Re-creation Loops**: Cache device contexts globally instead of recreating them mid-execution.
//! 3. **Loader Workaround**: On Linux, an underlying bug with dynamic library unloads can cause driver FDs to persist.
//!    Setting the environment variable `VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1` before execution stops the
//!    loader from forcefully closing drivers before the layer clears its handles.
//! 4. **Update Validation Layers**: Update to the latest Vulkan SDK release or build from Khronos main.
//! 5. **Bump the System Limit**: Artificially expand the OS's descriptor ceiling (`ulimit -n`).

use crate::{
  gpu::{
    self, AcquireResult, ArchetypeId, CommandBufferHandle, GpuResourceHandle, NativeGpuProperty,
    PipelineKey, PipelineKeyable, PresentationEngineHandle, RenderDevice, RenderableInstanceId,
    TextureFlags, compute_push_constants::ResetParticlesPushConstants, frame::ResourceUploadResult,
    new_particles::DustPushConstants,
  },
  gpu_backends::vulkan::{
    self,
    device::{
      commands::CommandBufferId,
      locks::DebugTrackedRwLock,
      memory::GlobalDeviceAllocator,
      particles::{ParticleSystemManager, PushConstantMutUnion},
      renderpasses::RenderPassSpecification,
      resources::Image,
      shader_manager::ShaderKey,
      swapchain::PresentationState,
    },
    instance,
    physics::VulkanComputeKernels,
    utils::{self, NonZeroHandle, RwLockable},
  },
  scene::{EntityId, StaticMeshComponent, text::FontAtlas},
  simulation::comet::{Comet, Texture},
  types::{GpuError, GpuResult},
};
use aethervk_oshal_rlib::{
  self as oshal,
  math::{
    quaternion::Quaternion,
    vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
  },
  os::{fs::FileSystemObject, fs::PathBuf, native::this_thread, pool::WorkloadStatus},
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
use ash::vk::{self, Handle};
use bytemuck::Zeroable;
use core::{
  any::Any,
  fmt,
  fmt::Formatter,
  hash::Hash,
  ptr,
  sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use function_name::named;
use heapless::index_map::FnvIndexMap;
use vk_mem::{Alloc, AsAllocatorView};

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
pub(super) mod debug_labels;

pub use resources::DiscardPool;

#[derive(Debug)]
pub struct TaskEntry {
  pub target_value: AtomicU64,
  pub status: AtomicU32, // 0: Pending, 1: Success, 2: Failed
  pub error: DebugTrackedRwLock<Option<GpuError>>,
}

const TASK_STATUS_PENDING: u32 = 0;
const TASK_STATUS_SUCCESS: u32 = 1;
const TASK_STATUS_FAILED: u32 = 2;

pub struct TimelinePollingWorkload {
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

// TOOD delete
trait DeviceResource {
  /// Cleanup function to facilitate hierarchical manual Drop of resources
  /// without having to propagate through `Arc` or other means a reference
  /// to device handle and its function pointers
  /// Note: This function is not responsible to setup the proper state for cleanup (eg synchronization)
  fn cleanup(&mut self, device: &LogicalDevice);
}

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
  pub discard_pool: resources::DiscardPool,
  live_presentation_engines: dashmap::DashMap<PresentationEngineHandle, PresentationState>,
  pub command_pools: Option<Arc<commands::CommandPools>>,
  pub descriptor_pool: DebugTrackedRwLock<Option<Arc<descriptors::DescriptorPools>>>,
  pub pipeline_pool: pipelines::PipelinePool,
  renderpasses: renderpasses::RenderPasses,
  pub shader_manager: DebugTrackedRwLock<shader_manager::ShaderManager>,

  pub timeline_manager: timeline_manager::TimelineManager,
  next_cmd_id: Arc<AtomicU64>,

  linear_sampler: NonZeroHandle<vk::Sampler>,

  physical_mesh2_resources: dashmap::DashMap<
    RenderableInstanceId,
    resources::ResourceState<resources::ForwardMesh2RenderResource>,
  >,
  sun_resources: dashmap::DashMap<EntityId, resources::ResourceState<resources::SunRenderResource>>,

  sky_image: DebugTrackedRwLock<Option<Image>>,
  billboard_resources: DebugTrackedRwLock<Vec<Option<Image>>>,

  pending_downloads: DebugTrackedRwLock<hashbrown::HashMap<u64, PendingDownload>>,

  frame_staging_arena: DebugTrackedRwLock<Option<memory::FrameStagingArena>>,

  sun_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::SunRenderResourceArchetypeArena>>>,
  physical_mesh2_render_archetype_arena: Option<
    alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMesh2RenderResourceArchetypeArena>>,
  >,
  billboard_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::BillboardRenderResourceArchetypeArena>>>,
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
  text2_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Text2RenderResourceArchetypeArena>>>,
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
  dust_render_archetype_arena:
    Option<alloc::sync::Arc<DebugTrackedRwLock<resources::DustRenderArchetypeArena>>>,

  /// Particle system management 2.0
  pub particle_system_manager: Option<ParticleSystemManager>,

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

    let _ = self.particle_system_manager.take();

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
              let mut arena = locks::DebugTrackedRwLock::into_inner(arena_lock);
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
    discard_arena!(physical_mesh2_render_archetype_arena);
    discard_arena!(billboard_render_archetype_arena);
    discard_arena!(cursor_render_archetype_arena);
    discard_arena!(marker_render_archetype_arena);
    discard_arena!(measurement_render_archetype_arena);
    discard_arena!(sky_render_archetype_arena);
    discard_arena!(grid_render_archetype_arena);
    discard_arena!(text2_render_archetype_arena);
    discard_arena!(sphere_gizmo_render_archetype_arena);
    discard_arena!(gizmo_render_archetype_arena);
    discard_arena!(trajectory_render_archetype_arena);
    discard_arena!(ui_render_archetype_arena);
    discard_arena!(background_render_archetype_arena);
    discard_arena!(dust_render_archetype_arena);

    // all discardable resources should have been already discarded
    if self.has_discardables() {
      self.clear_discardables(device);
    }
    self.discard_pool.cleanup(device);

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

    if let Some(mut command_pools) = self.command_pools.take() {
      alloc::sync::Arc::get_mut(&mut command_pools).unwrap().cleanup(device);
    }

    let keys: alloc::vec::Vec<_> =
      self.live_presentation_engines.iter().map(|kv| *kv.key()).collect();
    for k in keys {
      if let Some((_, mut presentation_state)) = self.live_presentation_engines.remove(&k) {
        presentation_state.cleanup(device);
      }
    }

    self.timeline_manager.cleanup(device);

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

    let taken_frame = locks::DebugTrackedRwLock::write(&self.frame_staging_arena).take();
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
      || !self.physical_mesh2_resources.is_empty()
      || !self.sun_resources.is_empty()
      || !self.billboard_resources.read().is_empty()
  }

  #[named]
  fn clear_discardables(&mut self, device: &LogicalDevice) {
    aethervk_oshal_rlib::log!("clear_discardables started!");
    debug_assert!(self.has_discardables());

    for mut pe_state in self.live_presentation_engines.iter_mut() {
      pe_state.value_mut().archetypes_mut().discard(device, &self.discard_pool);
    }

    let pm2_keys: alloc::vec::Vec<_> =
      self.physical_mesh2_resources.iter().map(|kv| *kv.key()).collect();
    for key in pm2_keys {
      if let Some((_, state)) = self.physical_mesh2_resources.remove(&key) {
        if let resources::ResourceState::Ready(mut resource) = state {
          resource.discard(&self.discard_pool, u64::MAX);
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

    for image in self.billboard_resources.write().drain(..).flatten() {
      self.discard_pool.discard_image(
        self.allocator.allocator.as_allocator_view(),
        image.image.get(),
        image.allocation,
        u64::MAX,
      );
      self.discard_pool.discard_image_view(image.image_view.get(), u64::MAX);
    }

    if let Some(sky_image) = self.sky_image.write().take() {
      self.discard_pool.discard_image(
        self.allocator.allocator.as_allocator_view(),
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
    _unique_family_indices_iter: impl Iterator<Item = &'a u32>,
    compute_queue: Queue,
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
    let command_pools = Some(Arc::new(commands::CommandPools::new()));
    // - Swapchain hashmap
    let live_presentation_engines = dashmap::DashMap::new();

    // timeline semaphore promoted to core after 1.2 (included)
    debug_assert!(instance.api_version() < vk::API_VERSION_1_2);

    let frame_staging_arena =
      memory::FrameStagingArena::new(&allocator.allocator, 32 * 1024 * 1024)?;

    let particle_system_manager = ParticleSystemManager::new(
      device,
      allocator.allocator.as_allocator_view(),
      compute_queue,
      crate::gpu::new_particles::MAX_PARTICLES,
    )?;

    Ok(Self {
      allocator,
      command_pools,
      discard_pool,
      live_presentation_engines,
      descriptor_pool: DebugTrackedRwLock::new(Some(descriptor_pool)),
      pipeline_pool,
      renderpasses,
      shader_manager: DebugTrackedRwLock::new(shader_manager::ShaderManager::new()),
      linear_sampler: unsafe { NonZeroHandle::new_unchecked(linear_sampler) },
      timeline_manager,
      physical_mesh2_resources: dashmap::DashMap::new(),
      sun_resources: dashmap::DashMap::new(),
      billboard_resources: DebugTrackedRwLock::new(Vec::with_capacity(16)),
      sky_image: DebugTrackedRwLock::new(None),
      next_cmd_id: Arc::new(AtomicU64::new(1)),
      pending_downloads: DebugTrackedRwLock::new(hashbrown::HashMap::new()),
      frame_staging_arena: DebugTrackedRwLock::new(Some(frame_staging_arena)),
      sun_render_archetype_arena: None,
      physical_mesh2_render_archetype_arena: None,
      billboard_render_archetype_arena: None,
      cursor_render_archetype_arena: None,
      marker_render_archetype_arena: None,
      measurement_render_archetype_arena: None,
      sky_render_archetype_arena: None,
      grid_render_archetype_arena: None,
      text2_render_archetype_arena: None,
      sphere_gizmo_render_archetype_arena: None,
      gizmo_render_archetype_arena: None,
      trajectory_render_archetype_arena: None,
      ui_render_archetype_arena: None,
      background_render_archetype_arena: None,
      dust_render_archetype_arena: None,
      particle_system_manager: Some(particle_system_manager), // turn off with None
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
  submission_fence: Option<NonZeroHandle<vk::Fence>>,
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
  #[cfg(debug_assertions)]
  debug_query_index: Option<u32>,
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
      #[cfg(debug_assertions)]
      debug_query_index: None,
      compositing_ctx: None,
    }
  }

  /// command buffer is automatically recycled by [`commands::CommandPools`]
  /// Since this is expoded
  #[named]
  fn discard(
    &mut self,
    device: &LogicalDevice,
    cmd_buf_id: CommandBufferId,
    discard_pool: &resources::DiscardPool,
    cmd_pools: Arc<commands::CommandPools>,
    family_index: u32,
    timeline: u64,
  ) {
    let tid = this_thread::id();
    if self.has_begun {
      discard_pool.discard_command_buffer(
        tid,
        cmd_buf_id,
        self.command_buffer.get(),
        family_index,
        cmd_pools,
        timeline,
      );
    } else {
      // Not recorded, so just recycle it immediately.
      let _ = cmd_pools.recycle(device, tid, family_index, self.command_buffer.get());
    }
  }
}

/// TODO: Document this item
pub struct LogicalDevice {
  pub handle: ash::Device,
  pub submission_lock: spin::Mutex<()>,
  pub submission_lock_compute: spin::Mutex<()>,
  /// Note: Remove if API_VERSION_1_2
  pub create_renderpass2: ash::khr::create_renderpass2::Device,
  pub buffer_device_address: ash::khr::buffer_device_address::Device,
  pub timeline_semaphore: ash::khr::timeline_semaphore::Device,
  /// Note: Remove if API_VERSION_1_3
  pub synchronization2: ash::khr::synchronization2::Device,

  pub swapchain_maintenance1: Option<ash::ext::swapchain_maintenance1::Device>,

  #[cfg(debug_assertions)]
  pub debug_utils: ash::ext::debug_utils::Device,

  #[cfg(debug_assertions)]
  pub telemetry_query_pool: Option<vk::QueryPool>,

  #[cfg(target_vendor = "apple")]
  pub metal_objects: ash::ext::metal_objects::Device,

  pub max_per_stage_descriptor_update_after_bind_samplers: u32,
  pub max_per_stage_descriptor_samplers: u32,
  pub max_descriptor_set_update_after_bind_samplers: u32,
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
  #[named]
  pub fn set_debug_name<T: vk::Handle>(&self, _object: T, _name: &str) {
    // This is a no-op in release builds, and should be optimized away.
  }

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

  #[named]
  pub fn locked_queue_submit_compute(
    &self,
    queue: vk::Queue,
    submits: &[vk::SubmitInfo],
    fence: vk::Fence,
  ) -> ash::prelude::VkResult<()> {
    let _guard = self.submission_lock_compute.lock();
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
    dashmap::DashMap<(CommandBufferHandle, QueueRole), RecordingCmdBufferData>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueRole {
  Graphics,
  Compute,
}

#[cfg(debug_assertions)]
static GRAPHICS_COUNT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

impl Device {
  const PAGE_TABLE_BYTES: u64 = (crate::gpu::new_particles::PARTICLE_PAGE_TABLE_HEADER_SIZE
    + 4
      * crate::gpu::new_particles::MAX_PARTICLES_PER_SYSTEM
        .div_ceil(crate::gpu::new_particles::PCHUNK_SIZE)) as _;

  /// Executes synchronously the `reset_particles` shader for
  /// # Safety
  /// There should be no compute or graphics command in flight. Externally synchronized
  pub unsafe fn reset_all_particle_systems(&self) -> GpuResult<()> {
    // create a quick one off compute queue compatible command buffer (ARM guidelines)
    // already in record state here
    let (cmd_handle, cmd) = self.get_compute_command_buffer_and_native()?;

    self.begin_command_buffer_all(cmd_handle, QueueRole::Compute)?;
    unsafe {
      self.device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.kernels.pipelines.reset_particles,
      );
    }
    // record, for each particle system, a reset
    // Note we are only resetting the back state. will this cause problems for rendering until next
    // cross sync?
    {
      let res = self.res.read();
      let psm = res.particle_system_manager.as_ref().unwrap();
      let back_state = psm.back();
      for kvref in back_state.page_tables.iter() {
        let ps = kvref.value();
        let push_constants = ResetParticlesPushConstants {
          particle_page_table: ps.address,
          free_list: back_state.free_list.address,
        };

        let local_size_x = unsafe {
          self
            .kernels
            .pipelines
            .wg_sizes
            .get(&self.kernels.pipelines.reset_particles.as_raw())
            .unwrap_unchecked()
        }[0];
        let particle_count = gpu::new_particles::MAX_PARTICLES_PER_SYSTEM as u32;
        let num_workgroups_x =
          Self::particle_system_shaders_num_workgroups(local_size_x, particle_count);

        unsafe {
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants),
          );

          self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
        }
      }
    }

    // submit and cleanup
    let (timeline_sem, value) =
      self.submit_command_buffer_generic(cmd_handle, None, &[], &[], QueueRole::Compute)?;

    self
      .device
      .wait_for_semaphore_value(timeline_sem, value, u64::MAX)
      .map_err(|e| gpu_err!("wait_for_semaphore_value failed: {:?}", e))?;

    Ok(())
  }

  /// `push_constants` should be fully populated except for some zeroed out fields populated here:
  /// - `global_particle_buffer`,
  /// - `particle_page_table`
  pub fn complete_graphics_particle_push_constant(
    &self,
    particle_system_id: u64,
    push_constants: &mut DustPushConstants,
  ) -> GpuResult<vk::Buffer> {
    let (global_particle_buffer_address, page_table_buffer_address, page_table_buffer) =
      self.grab_particle_system_data_and_buffer(particle_system_id, QueueRole::Graphics)?;

    push_constants.global_particle_buffer = global_particle_buffer_address;
    push_constants.particle_page_table = page_table_buffer_address;

    Ok(page_table_buffer)
  }

  /// Low level function to record a particle system draw call
  /// Notes:
  /// - assumes the function [`ParticleSystemManager::swap_buffers`] has already been called
  ///   after a "Cross Sync". We are guaranteed that this is the case if the incoming Render Command
  ///   contains the particle_acquire_sync compute timeline value.
  /// - does not bind the pipeline.
  pub fn cmd_draw_particle_system(
    &self,
    cmd: vk::CommandBuffer,
    indirect_command_buffer: vk::Buffer,
    push_constants: &DustPushConstants,
  ) -> GpuResult<()> {
    let layout = {
      let res = self.res.read();
      let arena_lock = res.dust_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena_arc = arena_lock.read();
      arena_arc.pipeline_layout.get()
    };
    let pc_bytes = bytemuck::bytes_of(push_constants);

    unsafe {
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        pc_bytes,
      );
      self.device.cmd_draw_indirect(cmd, indirect_command_buffer, 0, 1, 0);
    }

    Ok(())
  }

  // TODO delete other pipeline key getters
  pub fn get_archetype_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
    archetype_id: ArchetypeId,
  ) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, archetype_id)
  }

  #[named]
  fn get_pipeline_key_internal(
    &self,
    handle: PresentationEngineHandle,
    archetype_id: ArchetypeId,
  ) -> GpuResult<PipelineKey> {
    let res_guard = self.res.read();
    let pe = res_guard.live_presentation_engines.get(&handle).ok_or(GpuError::NotFound)?;
    pe.archetypes()
      .registry
      .read()
      .get(&archetype_id)
      .map(|a| a.pipeline_key())
      .ok_or(gpu_err_pipeline_key_absent!())
  }

  /// Submit a given command buffer either for compute or for graphics
  /// Returns timeline semaphore (graphics queue or compute) and the timeline value which will be
  /// reached once execution reaches BOTTOM_OF_PIPE
  ///
  /// Note: To maintain old behaviour, we register tasks only for graphics queue submissions
  ///   this means that `task_id` is used only by graphics role.
  #[named]
  pub fn submit_command_buffer_generic(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
    wait_infos: &[crate::gpu::CommandBufferSyncInfo],
    signal_infos: &[crate::gpu::CommandBufferSyncInfo],
    role: QueueRole,
  ) -> GpuResult<(vk::Semaphore, u64)> {
    use super::utils::RwLockable;
    const MAX_WAIT_SYNC_INFOS: usize = 8;
    const MAX_SIGNAL_SYNC_INFOS: usize = 8;
    if wait_infos.len() > MAX_WAIT_SYNC_INFOS || signal_infos.len() > MAX_SIGNAL_SYNC_INFOS {
      return Err(gpu_err!(
        "wait_info or signal_infos crossed the maximum allowed threshold"
      ));
    }

    let cmd_key = (cmd_buffer, role);
    // 1. Extract command buffer data and drop the lock immediately
    let mut data = {
      let cmd_buffers = &self.recording_command_buffers;
      cmd_buffers.remove(&cmd_key).ok_or(gpu_err_invalid_cmd!())?
    }
    .1;

    unsafe {
      #[cfg(debug_assertions)]
      if let Some(pool) = self.device.telemetry_query_pool {
        if let Some(query_index) = data.debug_query_index {
          self.device.cmd_write_timestamp(
            data.command_buffer.get(),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            pool,
            query_index + 1,
          );
        }
      }

      self.device.end_command_buffer(data.command_buffer.get())?;
    }

    // Read back the results from 4 command buffers ago (guaranteed to be finished)
    #[cfg(debug_assertions)]
    if let Some(pool) = self.device.telemetry_query_pool {
      if role == QueueRole::Graphics {
        let g_count = GRAPHICS_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if g_count >= 4 {
          let old_query_index = ((g_count - 4) % 512) as u32 * 2;
          let mut results = [[0u64; 2]; 2];
          if unsafe {
            self.device.get_query_pool_results(
              pool,
              old_query_index,
              results.as_mut_slice(),
              vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
          }
          .is_ok()
          {
            if results[0][1] != 0 && results[1][1] != 0 {
              let start_ts = results[0][0];
              let end_ts = results[1][0];
              if end_ts > start_ts {
                let diff_ticks = end_ts - start_ts;
                let period = self.query_result.physical_device_properties.limits.timestamp_period;
                let diff_ms = (diff_ticks as f64) * (period as f64) / 1_000_000.0;
                crate::gpu_backends::vulkan::DEBUG_RENDER_THREAD_GPU_TIME_MS
                  .store(diff_ms.to_bits(), core::sync::atomic::Ordering::Relaxed);
              }
            }
          }
        }
      }
    }

    let is_graphics = role == QueueRole::Graphics;
    let queue = if is_graphics {
      self.get_graphics_queue()
    } else {
      self.get_compute_queue()
    };
    let presentation: Option<_> = if is_graphics {
      // TODO: test whether there's a problem if we remove this check
      // Some(data.presentation.take().ok_or(gpu_err_cmd_no_pe!())?)
      data.presentation.take()
    } else {
      None
    };

    let timeline_sem = if is_graphics {
      self.res.read().timeline_manager.semaphore.get()
    } else {
      self.kernels.timeline
    };

    // 2. Start Vulkan Transaction for the submission process
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(
        presentation.map(|p| p.presentation_engine),
        |state, pe_handle_opt| {
          let pe_opt: Option<
            dashmap::mapref::one::Ref<
              '_,
              gpu::PresentationEngineHandle,
              swapchain::PresentationState,
            >,
          > = if let Some(pe_handle) = &pe_handle_opt {
            Some(wait_for_pe!(state, pe_handle)?)
          } else {
            None
          };

          let is_resize_required = pe_opt
            .map(|pe| {
              pe.swapchain_generation()
                != unsafe { presentation.unwrap_unchecked() }.swapchain_generation
            })
            .unwrap_or(false);
          let cmd_pools = state
            .command_pools
            .as_ref()
            .cloned()
            .ok_or(gpu_err!("couldn't get command pools"))?;

          let task_registry = state.timeline_manager.task_registry.clone();
          let timeline_manager_ptr = if is_graphics {
            // this is ensured to be non null. can't use NonNull cause it's *const
            Some(&state.timeline_manager as *const timeline_manager::TimelineManager)
          } else {
            None
          };

          Ok((
            pe_handle_opt,
            is_resize_required,
            timeline_sem,
            cmd_pools,
            task_registry,
            timeline_manager_ptr,
          ))
        },
      )?
      .execute(
        |(
          pe_handle_opt,
          is_resize_required,
          timeline_sem,
          cmd_pools,
          task_registry,
          timeline_manager_ptr,
        ),
         _rollback| {
          // +1 to account for possible presentation engine signal semaphore
          let mut signal_semaphore_infos =
            heapless::Vec::<vk::SemaphoreSubmitInfo, { MAX_SIGNAL_SYNC_INFOS + 1 }>::new();
          let mut wait_semaphore_infos =
            heapless::Vec::<vk::SemaphoreSubmitInfo, { MAX_WAIT_SYNC_INFOS + 1 }>::new();

          if pe_handle_opt.is_some() {
            let pres = unsafe { presentation.unwrap_unchecked() };
            if let Some(sem) = pres.signal_semaphore {
              unsafe {
                // This is not a timeline semaphore, so value is don't care
                signal_semaphore_infos.push_unchecked(
                  vk::SemaphoreSubmitInfo::default()
                    .semaphore(sem.get())
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                );
              }
            }

            if let Some(wait_semaphore) = pres.wait_semaphore {
              unsafe {
                wait_semaphore_infos.push_unchecked(
                  vk::SemaphoreSubmitInfo::default()
                    .semaphore(wait_semaphore.get())
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                );
              }
            }
          }

          for sync in wait_infos {
            use ash::vk::Handle;
            let vk_semaphore = vk::Semaphore::from_raw(sync.timeline_semaphore);
            unsafe {
              wait_semaphore_infos.push_unchecked(
                vk::SemaphoreSubmitInfo::default()
                  .semaphore(vk_semaphore)
                  .value(sync.timeline_value)
                  .stage_mask(gpu_sync_info_to_flags(sync.wait_stage_mask, false)),
              );
            }
          }

          for sync in signal_infos {
            use ash::vk::Handle;
            let vk_semaphore = vk::Semaphore::from_raw(sync.timeline_semaphore);
            unsafe {
              signal_semaphore_infos.push_unchecked(
                vk::SemaphoreSubmitInfo::default()
                  .semaphore(vk_semaphore)
                  .value(sync.timeline_value)
                  .stage_mask(gpu_sync_info_to_flags(sync.wait_stage_mask, false)),
              );
            }
          }

          let next_timeline_value = if let Some(timeline_manager_ptr_) = timeline_manager_ptr {
            debug_assert!(is_graphics);
            let timeline_manager = unsafe { &*timeline_manager_ptr_ };
            timeline_manager.allocate_submit_value()
          } else {
            self
              .kernels
              .next_submit_value
              .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
          };

          unsafe {
            signal_semaphore_infos.push_unchecked(
              vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline_sem)
                .value(next_timeline_value)
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            );
          }

          // TAKE SUBMISSION LOCK BEFORE ALLOCATING TIMELINE!
          // This ensures that the order we get timeline values exactly matches the order we submit to the queue.
          let _guard = if is_graphics {
            self.device.submission_lock.lock()
          } else {
            self.device.submission_lock_compute.lock()
          };

          let command_buffer_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(data.command_buffer.get());

          let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .command_buffer_infos(core::slice::from_ref(&command_buffer_info));

          #[cfg(debug_assertions)]
          let _render_start = aethervk_oshal_rlib::os::time::get_monotonic_time();

          unsafe {
            self
              .device
              .synchronization2
              .queue_submit2(
                queue.handle,
                core::slice::from_ref(&submit_info),
                presentation
                  .and_then(|p| p.submission_fence)
                  .map(|f| f.get())
                  .unwrap_or(vk::Fence::null()),
              )
              .map_err(|e| {
                aethervk_oshal_rlib::log!("Queue submit failed: {:?}", e);
                GpuError::from(e)
              })?;
          }
          drop(_guard); // unlock here

          #[cfg(debug_assertions)]
          {
            let elapsed_ms =
              (aethervk_oshal_rlib::os::time::get_monotonic_time() - _render_start) as f64 / 1000.0;
            crate::gpu_backends::vulkan::DEBUG_RENDER_THREAD_CPU_TIME_MS
              .store(elapsed_ms.to_bits(), core::sync::atomic::Ordering::Relaxed);
          }

          // Inform the task registry of the timeline value to wait for
          if is_graphics {
            if let Some(tid) = task_id {
              let registry = locks::DebugTrackedRwLock::write(&task_registry);
              if let Some(entry) = registry.get(&tid) {
                entry
                  .target_value
                  .store(next_timeline_value, core::sync::atomic::Ordering::Release);
              }
            }
          }

          // Pass 'data' through to be discarded in the commit phase
          Ok((
            data,
            pe_handle_opt,
            is_resize_required,
            next_timeline_value,
            cmd_pools,
          ))
        },
      )
      .commit_read(|state, execute_result| {
        let (mut data, pe_handle_opt, is_resize_required, next_timeline_value, cmd_pools) =
          execute_result?;

        if is_graphics && pe_handle_opt.is_some() {
          // SAFETY: graphics queue submissions through this function are always rendering commands,
          // which therefore have a presentation engine
          let pe_handle = unsafe { pe_handle_opt.unwrap_unchecked() };
          if let Some(mut pe) = state.live_presentation_engines.get_mut(&pe_handle) {
            if let Some(pres) = &data.presentation {
              pe.mark_fence_submitted(pres.acquire_result.frame_index as u32);
            }
            if let swapchain::PresentationState::Windowless(windowless) = pe.value() {
              windowless
                .last_timeline_value
                .store(next_timeline_value, core::sync::atomic::Ordering::Release);
            }
          }
        }

        // Discard resources now that submission is safely recorded
        data.discard(
          &self.device,
          cmd_buffer.into(),
          &state.discard_pool,
          cmd_pools,
          queue.family_index,
          next_timeline_value,
        );

        if is_resize_required {
          Err(GpuError::ResizeRequired)
        } else {
          Ok((timeline_sem, next_timeline_value))
        }
      })
  }

  #[named]
  pub fn begin_command_buffer_all(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    role: QueueRole,
  ) -> GpuResult<()> {
    let cmd_buffers = &self.recording_command_buffers;
    let mut data = cmd_buffers.get_mut(&(cmd_buffer, role)).ok_or(gpu_err_invalid_cmd!())?;

    if data.has_begun {
      return Ok(());
    }

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self.device.begin_command_buffer(data.command_buffer.get(), &begin_info)?;

      #[cfg(debug_assertions)]
      if let Some(pool) = self.device.telemetry_query_pool {
        if role == QueueRole::Graphics {
          // cmd_buffer.0 is monotonically increasing. We have a 1024-query pool.
          // 2 queries per cmd buffer means we can hold 512 cmd buffers.
          let query_index =
            (GRAPHICS_COUNT.load(core::sync::atomic::Ordering::Relaxed) % 512) as u32 * 2;
          data.debug_query_index = Some(query_index);
          self
            .device
            .cmd_reset_query_pool(data.command_buffer.get(), pool, query_index, 2);
          self.device.cmd_write_timestamp(
            data.command_buffer.get(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
            pool,
            query_index,
          );
        }
      }
    }
    data.has_begun = true;

    Ok(())
  }

  /// Get the target graphics queue timeline value for which a given task id will be completed
  pub fn get_task_target_value(&self, task_id: u64) -> GpuResult<u64> {
    let res = self.res.read();
    res.timeline_manager.get_task_target_value(task_id)
  }

  /// version of the [`crate::gpu::RenderDevice`] method `get_command_buffer` which also returns
  /// the native `VkCommandBuffer`
  pub fn get_command_buffer_and_native(
    &self,
  ) -> GpuResult<(gpu::CommandBufferHandle, vk::CommandBuffer)> {
    self.get_command_buffer_and_native_all(QueueRole::Graphics)
  }

  #[named]
  pub fn get_compute_command_buffer_and_native(
    &self,
  ) -> GpuResult<(gpu::CommandBufferHandle, vk::CommandBuffer)> {
    self.get_command_buffer_and_native_all(QueueRole::Compute)
  }

  /// version of the [`crate::gpu::RenderDevice`] method `get_command_buffer` which also returns
  /// the native `VkCommandBuffer`. generalized to support compute too
  #[named]
  pub fn get_command_buffer_and_native_all(
    &self,
    role: QueueRole,
  ) -> GpuResult<(gpu::CommandBufferHandle, vk::CommandBuffer)> {
    use super::utils::RwLockable;
    let is_compute = match role {
      QueueRole::Graphics => false,
      _ => true,
    };
    let (cmd_id, cmd_pool_arc, current_timeline, discard_pool_ptr) = {
      let res_guard = self.res.read();
      let cmd_id = if is_compute {
        self.kernels.next_cmd_id.fetch_add(1, core::sync::atomic::Ordering::SeqCst)
      } else {
        res_guard.next_cmd_id.fetch_add(1, core::sync::atomic::Ordering::SeqCst)
      };
      let cmd_pool_arc = unsafe { res_guard.command_pools.as_ref().unwrap_unchecked().clone() };
      let discard_pool_ptr = core::ptr::from_ref(if is_compute {
        &self.kernels.discard_pool
      } else {
        &res_guard.discard_pool
      });
      let current_timeline = if is_compute {
        self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed)
      } else {
        res_guard.get_timeline_semaphore_cached_value()
      };
      (cmd_id, cmd_pool_arc, current_timeline, discard_pool_ptr)
    };

    let family_index = if is_compute {
      self.get_compute_queue().family_index
    } else {
      self.get_graphics_queue().family_index
    };

    let cmd = {
      super::allocate_primary_vk_command_buffer(
        &self.device,
        &cmd_pool_arc,
        // SAFETY: cannot destroy without dropping `self`
        unsafe { discard_pool_ptr.as_ref_unchecked() },
        family_index,
        CommandBufferId(cmd_id),
        current_timeline,
      )
    }?;

    // should be `None`, but we don't care
    let _ = self.recording_command_buffers.insert(
      (CommandBufferHandle(cmd_id), role),
      RecordingCmdBufferData::new(unsafe { NonZeroHandle::new_unchecked(cmd) }),
    );

    Ok((CommandBufferHandle(cmd_id), cmd))
  }

  /// returns, when ok currently cached compute timeline value
  #[named]
  pub fn create_particle_system(&self, id: u64) -> GpuResult<u64> {
    if self.res.read().particle_system_manager.is_none() {
      return Err(gpu_err!("particle_system_manager absent"));
    }
    let compute_timeline_value =
      self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        let allocator = state.allocator.allocator.as_allocator_view();
        let discard_pool_ptr = &self.kernels.discard_pool as *const resources::DiscardPool;
        Ok((allocator, discard_pool_ptr))
      })?
      .execute(|(allocator, discard_pool_ptr), rollback| {
        let device = &self.device;
        let discard_pool = unsafe { &*discard_pool_ptr };
        let compute_queue = self.get_compute_queue();

        let command_pool_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(compute_queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
          .with_name(device, "CommandPool_ParticleSystemTransient")?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
          .with_name(device, "Fence_ParticleSystemAllocationTransient")?;
        let mut _cleanup = TransientCleanup::command_only(device, command_pool, fence);

        // allocate GPU-only page table resource
        let buffer_info = vk::BufferCreateInfo::default().size(Self::PAGE_TABLE_BYTES).usage(
          vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC // for copy synchronization
            | vk::BufferUsageFlags::TRANSFER_DST // for copy synchronization
            | vk::BufferUsageFlags::INDIRECT_BUFFER, // to be fed to `vkCmdDrawIndirect`
        );
        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        crate::apply_test_dedicated_alloc!(alloc_info);
        alloc_info.priority = 1.0f32;

        let (buffer_0, alloc_0) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
          .with_name(
            device,
            &alloc::format!("ParticleSystem_0_t{}", compute_timeline_value),
          )?;
        let (buffer_1, alloc_1) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
          .with_name(
            device,
            &alloc::format!("ParticleSystem_1_t{}", compute_timeline_value),
          )?;

        let roll_alloc_0 = alloc_0.clone();
        let roll_alloc_1 = alloc_1.clone();
        let raw_allocator = allocator.as_allocator_view();
        rollback.defer(move |_| {
          discard_pool.discard_buffer(
            raw_allocator,
            buffer_0,
            roll_alloc_0,
            compute_timeline_value,
          );
          discard_pool.discard_buffer(
            raw_allocator,
            buffer_1,
            roll_alloc_1,
            compute_timeline_value,
          );
        });

        // fill both buffers with 0xFFFF'FFFF
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        let command_buffer = unsafe {
          let mut c = vk::CommandBuffer::null();
          (device.fp_v1_0().allocate_command_buffers)(
            device.handle(),
            core::ptr::from_ref(&command_buffer_info),
            core::ptr::from_mut(&mut c),
          )
          .result_with_success(c)
        }
        .with_name(device, "CommandBuffer_Transient_particleSystem")?;

        unsafe {
          // record phase
          let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
          self.device.begin_command_buffer(command_buffer, &begin_info)?;

          // 1. Define the 32-byte header data inline (8 u32s)
          // Layout: [particleCount, instanceCount, firstVertex, firstInstance, activeChunkCount, pad0, pad1, pad2]
          let header_data: [u32; 8] = [0, 1, 0, 0, 0, 0, 0, 0];

          // Cast to a byte slice in a `no_std` compatible way
          let header_bytes = core::slice::from_raw_parts(
            header_data.as_ptr() as *const u8,
            core::mem::size_of_val(&header_data),
          );

          // --- Configure buffer_0 ---

          // Write the 32-byte header to the start of the buffer (offset 0)
          self.device.cmd_update_buffer(command_buffer, buffer_0, 0, header_bytes);
          // Fill the rest of the buffer with u32::MAX (offset 32 to the end)
          self
            .device
            .cmd_fill_buffer(command_buffer, buffer_0, 32, vk::WHOLE_SIZE, u32::MAX);

          // --- Configure buffer_1 ---

          // Write the 32-byte header to the start of the buffer (offset 0)
          self.device.cmd_update_buffer(command_buffer, buffer_1, 0, header_bytes);
          // Fill the rest of the buffer with u32::MAX (offset 32 to the end)
          self
            .device
            .cmd_fill_buffer(command_buffer, buffer_1, 32, vk::WHOLE_SIZE, u32::MAX);

          self.device.end_command_buffer(command_buffer)?;

          // submit and wait phase
          let submit_info =
            vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
          device
            .locked_queue_submit_compute(
              compute_queue.handle,
              core::slice::from_ref(&submit_info),
              fence,
            )
            .map_err(GpuError::from)?;
          let _ = device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX);
        }

        let bda_info_0 = vk::BufferDeviceAddressInfo::default().buffer(buffer_0);
        let bda_0 = unsafe { device.buffer_device_address.get_buffer_device_address(&bda_info_0) };
        let bda_info_1 = vk::BufferDeviceAddressInfo::default().buffer(buffer_1);
        let bda_1 = unsafe { device.buffer_device_address.get_buffer_device_address(&bda_info_1) };

        Ok(((bda_0, buffer_0, alloc_1), (bda_1, buffer_1, alloc_1)))
      })
      .commit_read(|state, res| {
        if let Ok(((bda_0, buffer_0, alloc_0), (bda_1, buffer_1, alloc_1))) = res {
          // safety checked at the beginning of the function
          let psm = unsafe { state.particle_system_manager.as_ref().unwrap_unchecked() };

          psm.add_page_tables(
            id,
            particles::BufferAlloc::new(buffer_0, alloc_0, bda_0),
            particles::BufferAlloc::new(buffer_1, alloc_1, bda_1),
          );

          Ok(compute_timeline_value)
        } else {
          Err(unsafe { res.unwrap_err_unchecked() })
        }
      })
  }

  /// discard in kernel's discard pool a specified particle system, discarding it's underlying
  /// buffer and deleting it's association inside the `[crate::gpu_backends::vulkan::device::particles::ParticleSystemManager]`, therefore should
  /// never be used after this
  #[named]
  pub fn discard_particle_system(
    &self,
    id: u64,
    gfx_timeline: u64,
    comp_timeline: u64,
  ) -> GpuResult<()> {
    let (allocator, buf_gfx, alloc_gfx, buf_comp, alloc_comp) = {
      let res = self.res.read();
      let allocator = res.allocator.allocator.as_allocator_view();
      let psm = res
        .particle_system_manager
        .as_ref()
        .ok_or(gpu_err!("particle_system_manager absent"))?;
      let (pt_arr, gfx_index) = psm
        .remove_pages_tables(id)
        .ok_or(gpu_err!("particle system with id {} not found", id))?;
      (
        allocator,
        pt_arr[gfx_index].buffer,
        pt_arr[gfx_index].alloc,
        pt_arr[1 - gfx_index].buffer,
        pt_arr[1 - gfx_index].alloc,
      )
    };

    self.kernels.discard_pool.discard_buffer(
      allocator.as_allocator_view(),
      buf_comp,
      alloc_comp,
      comp_timeline,
    );
    self.res.read().discard_pool.discard_buffer(
      allocator.as_allocator_view(),
      buf_gfx,
      alloc_gfx,
      gfx_timeline,
    );
    Ok(())
  }

  pub fn cmd_dispatch_global_memory_barrier(&self, cmd: vk::CommandBuffer) -> GpuResult<()> {
    // Define a global memory barrier for compute-to-compute synchronization
    let memory_barrier = vk::MemoryBarrier2::default()
      // wait for previous compute shaders to finish executing ...
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      // .. and ensure all their writes to memory are fully flushed
      .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
      // block the next compute shaders from executing ...
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      //... until they can safely read and write their own data
      .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE);
    let dependency_info =
      vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&memory_barrier));
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dependency_info);
    }
    Ok(())
  }

  /// Sync: needs a read lock on `[Device::res]`
  /// returns respectively
  /// - global_buffer_address
  /// - page_table_buffer_address
  /// - free_list_address
  #[named]
  fn grab_particle_system_data(&self, id: u64, role: QueueRole) -> GpuResult<(u64, u64, u64)> {
    let res = self.res.read();
    if res.particle_system_manager.is_none() {
      return Err(gpu_err!("particle_system_manager absent"));
    }
    let psm = unsafe { res.particle_system_manager.as_ref().unwrap_unchecked() };

    psm
      .get_addresses(id, role)
      .ok_or(gpu_err!("particle system with id {} not found", id))
  }

  /// Sync: needs a read lock on `[Device::res]`
  /// returns respectively
  /// - global_buffer_address
  /// - page_table_buffer_address
  /// - page_table_buffer vkBuffer handle
  #[named]
  fn grab_particle_system_data_and_buffer(
    &self,
    id: u64,
    role: QueueRole,
  ) -> GpuResult<(u64, u64, vk::Buffer)> {
    let res = self.res.read();
    if res.particle_system_manager.is_none() {
      return Err(gpu_err!("particle_system_manager absent"));
    }
    let psm = unsafe { res.particle_system_manager.as_ref().unwrap_unchecked() };

    psm
      .get_addresses_and_buffer(id, role)
      .ok_or(gpu_err!("particle system with id {} not found", id))
  }

  /// Note: for ApplyEmittersDirectNew emitters are not populated
  /// Caller is supposed to be physics tasklet thread
  pub fn complete_particle_push_constant<'a>(
    &self,
    push_constants: PushConstantMutUnion<'a>,
    particle_system_id: u64,
  ) -> GpuResult<()> {
    let (global_particle_buffer_address, page_table_buffer_address, free_list_buffer_address) =
      self.grab_particle_system_data(particle_system_id, QueueRole::Compute)?;
    match push_constants {
      PushConstantMutUnion::ApplyEmittersDirectNew(value) => {
        value.global_particle_buffer_address = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
        // emitters elsewhere
      }
      PushConstantMutUnion::IntegrateParticlesP1P2New(value) => {
        value.global_particle_buffer_address = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
      }
      PushConstantMutUnion::IntegrateParticlesP45New(value) => {
        value.global_particle_buffer_address = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
      }
      PushConstantMutUnion::NewParticlesCompactReset(value) => {
        value.particle_page_table = page_table_buffer_address;
      }
      PushConstantMutUnion::NewParticlesEmit(value) => {
        value.global_particle_buffer = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
        value.free_list = free_list_buffer_address;
      }
      PushConstantMutUnion::NewParticlesCompact(value) => {
        value.global_particle_buffer_address = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
        value.free_list = free_list_buffer_address;
      }
      PushConstantMutUnion::NewParticlesOffsetParticlesPush(value) => {
        value.global_particle_buffer = global_particle_buffer_address;
        value.particle_page_table = page_table_buffer_address;
      }
    }

    Ok(())
  }

  /// Notes:
  /// - externallly synchronized with global memory barriers with [`Device::cmd_dispatch_global_memory_barrier`]
  #[named]
  pub fn cmd_particle_system_emission(
    &self,
    cmd: vk::CommandBuffer,
    last_emission_unscaled_us: i64,
    now_unscaled_us: i64,
    push_constants: &gpu::compute_push_constants::NewParticlesEmitPushConstants,
    skip_bind: bool,
  ) -> GpuResult<i64> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );

    {
      let res = self.res.read();
      if self.res.read().particle_system_manager.is_none() {
        return Err(gpu_err!("particle_system_manager absent"));
      }
      let psm = unsafe { res.particle_system_manager.as_ref().unwrap_unchecked() };

      // check emission interval
      if now_unscaled_us - last_emission_unscaled_us <= psm.emission_us() {
        return Ok(last_emission_unscaled_us);
      }
    }

    if !skip_bind {
      let pipeline: vk::Pipeline = self.kernels.pipelines.new_particles_emit;
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout: vk::PipelineLayout = self.kernels.pipelines.pipeline_layout;
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      // we are independent from the workgroup size.
      let needed_chunks = push_constants.emit_count.div_ceil(gpu::new_particles::PCHUNK_SIZE as _);
      self.device.cmd_dispatch(cmd, needed_chunks, 1, 1);
    }

    Ok(now_unscaled_us)
  }

  /// compute the number of workgroups for particle system shaders
  /// - velocity kick
  /// - apply emitters
  /// - velocity correction
  fn particle_system_shaders_num_workgroups(local_size_x: u32, particles_count: u32) -> u32 {
    const PARTICLES_PER_THREAD: u32 = 4;
    let particles_per_workgroup = local_size_x * PARTICLES_PER_THREAD;

    (particles_count + particles_per_workgroup - 1) / particles_per_workgroup
  }

  /// 1
  #[named]
  pub fn cmd_particle_system_velocity_vertlet_kick(
    &self,
    cmd: vk::CommandBuffer,
    push_constants: &gpu::compute_push_constants::IntegrateParticlesP1P2NewPushConstants,
    skip_bind: bool,
  ) -> GpuResult<()> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );
    let pipeline: vk::Pipeline = self.kernels.pipelines.integrate_particles_p1_p2_new;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout: vk::PipelineLayout = self.kernels.pipelines.pipeline_layout;
    // SAFETY: property constructed Pipelines has local sizes for all shaders
    let local_size_x =
      unsafe { self.kernels.pipelines.wg_sizes.get(&pipeline.as_raw()).unwrap_unchecked() }[0];
    let particle_count = gpu::new_particles::MAX_PARTICLES_PER_SYSTEM as u32;
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      let num_workgroups_x: u32 =
        Self::particle_system_shaders_num_workgroups(local_size_x, particle_count);
      self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
    }

    Ok(())
  }

  /// When using Point Gravity (Type 0), you calculate distance with rx = s_em_posMu.x - px_n;. Because 32-bit floats lose fractional precision as numbers get larger, space eventually quantizes into a "grid."
  ///
  /// We can calculate the exact size of this grid using the Unit in the Last Place (ULP) of IEEE 754 floats:
  /// - At 1,000 km (106 m): f32 space resolves to a 6.25 cm grid.
  /// - At 8,388 km (8.38×106 m): The exponent shifts, and f32 precision drops to exactly 1.0 metre.
  /// - At 1 AU / The Sun (1.5×1011 m): f32 precision drops to 16 kilometres.
  ///
  /// Furthermore, our shader computes for point gravity distanc^5, meaning
  /// (5.08×10^7)^5=3.4×10^38 (max float32) -> Anything further than ~50,000 km will cause dist5 to overflow to +Infinity
  pub fn cmd_allocate_transient_emitter_for_particle_system(
    &self,
    cmd: vk::CommandBuffer,
    discard_timeline: u64,
    frame_position_au: (Vec3f32, Quat),
    particle_system_pos_framerel_km: (Vec3f32, Quat),
  ) -> GpuResult<(vk::Buffer, vk_mem::Allocation, u64)> {
    use aethervk_oshal_rlib::math::{vector::vec3f64::DVec3, vector::vec4f64::DQuat};
    // constants
    const AU_TO_KM: f64 = 149_597_870.7;
    const KM_TO_M: f64 = 1000.0;
    const SUN_MU_KM3_S2: f64 = 1.32712440018e11;
    const SUN_MU_M3_S2: f64 = SUN_MU_KM3_S2 * (KM_TO_M * KM_TO_M * KM_TO_M);

    // - sun is at the origin of the root frame
    let sun_root_au = DVec3::zero();

    // - convert to frame's local space (units: AU)
    let frame_pos_au = DVec3::from(Into::<[f32; 3]>::into(frame_position_au.0));
    let frame_rot = DQuat::from_quat(frame_position_au.1);
    // go from parent -> child, we subtract child's position, then apply the inverse rotation
    // this is the sun position in the frame's coordinate system
    let sun_frame_au: DVec3 = frame_rot.conjugate().rotate_vector(sun_root_au - frame_pos_au);

    // - scale units to kilometres
    let sun_frame_km: DVec3 = sun_frame_au * AU_TO_KM;

    // - convert to particle system's local space (units: Km)
    let ps_pos_km = DVec3::from(Into::<[f32; 3]>::into(particle_system_pos_framerel_km.0));
    let ps_rot = DQuat::from_quat(particle_system_pos_framerel_km.1);
    // this is the sun position in the particle system coordinate system
    let sun_ps_km = ps_rot.conjugate().rotate_vector(sun_frame_km - ps_pos_km);

    // final scale to metres
    let sun_ps_m = sun_ps_km * KM_TO_M;
    let distance_m: f64 = sun_ps_m.length();

    const POINT_GRAVITY_THRESHOLD_M: f64 = 10_000_000.0; // 10,000 km
    let (emitter_pos, emitter_mu, type_id) = if distance_m > POINT_GRAVITY_THRESHOLD_M {
      // FALLBACK: Directional (Planar) Gravity
      // Force magnitude: `GM / r^2` for all particles
      let force_magnitude: f64 = SUN_MU_M3_S2 / (distance_m * distance_m);
      // direction vector to the sun (normalized)
      let direction: Vec3f32 = (sun_ps_m / distance_m).to_f32();
      // `position` for planar force is direction
      (
        Into::<[f32; 3]>::into(direction),
        force_magnitude as f32,
        1u32,
      )
    } else {
      // STANDARD: Point Gravity
      // only used if particle system is close enough to sun
      (
        Into::<[f32; 3]>::into(sun_ps_m.to_f32()),
        SUN_MU_M3_S2 as f32,
        0u32,
      )
    };

    #[repr(C, align(16))]
    #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct ParticleForceEmitter {
      // x, y, z from pos, w from mu
      pub position_mu: [f32; 4],
      pub type_id: u32,
      // Note: Rust will automatically insert 12 bytes of padding here
      // to satisfy the 16-byte alignment requirement of the struct.
      pub _pad: [u32; 3],
    }

    let emitter = {
      let mut x = ParticleForceEmitter::zeroed();
      x.position_mu = [emitter_pos[0], emitter_pos[1], emitter_pos[2], emitter_mu];
      x.type_id = type_id;
      x
    };

    // Strategy: we are going to call VMA allocation every single frame, hoping that it's
    // suballocation strategy is strong enough. Furthermore, we are going to use
    // `VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT` so that VMA can choose a
    // memory `DEVICE_LOCAL` and `HOST_VISIBLE` if possible, fallback to `DEVICE_LOCAL`, which is
    // the suggested strategy for frequent uploads, by VMA guide
    let buf_create_info = vk::BufferCreateInfo::default()
      .size(core::mem::size_of_val(&emitter) as _)
      .usage(
        vk::BufferUsageFlags::TRANSFER_DST
          | vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      );
    let mut alloc_create_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(alloc_create_info);
    alloc_create_info.usage = vk_mem::MemoryUsage::Auto;
    alloc_create_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
      | vk_mem::AllocationCreateFlags::MAPPED;

    // SAFETY: we assume you are not going to delete the resource while simulating
    let allocator = self.res.read().allocator.allocator.as_allocator_view();
    let (gpu_buffer, gpu_alloc, gpu_alloc_info) =
      unsafe { allocator.create_buffer_get_info(&buf_create_info, &alloc_create_info) }?;
    let gpu_janitor = AllocJanitor {
      buffer: gpu_buffer,
      alloc: gpu_alloc,
      allocator,
    };

    let gpu_buffer_address = unsafe {
      let buffer_bda_info = vk::BufferDeviceAddressInfo::default().buffer(gpu_buffer);
      self.device.buffer_device_address.get_buffer_device_address(&buffer_bda_info)
    };

    let mem_props = unsafe { allocator.get_allocation_memory_properties(&gpu_alloc) };
    if (mem_props & vk::MemoryPropertyFlags::HOST_VISIBLE) != vk::MemoryPropertyFlags::empty() {
      // allocation ended up in a mappable memory, so we can memcpy directly and issue a
      // host->compute barrier, so that we flush to system memory before gpu dispatch
      unsafe { allocator.copy_memory_to_allocation(&gpu_alloc, bytemuck::bytes_of(&emitter), 0) }?;
      let mem_barrier = vk::BufferMemoryBarrier2::default()
        // the host side ..
        .src_stage_mask(vk::PipelineStageFlags2::HOST)
        // ... should have finished its memory transactions from cache -> system RAM
        .src_access_mask(vk::AccessFlags2::HOST_WRITE)
        // before gpu compute shader ...
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        // ... performs any read ...
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
        // ... towards this buffer
        .buffer(gpu_buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        // ... without any queue ownership transfer
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
      let dep_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&mem_barrier));
      unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info) };
    } else {
      // allocation is not host visible. Staging buffer and transfer needed
      let staging_buf_create_info = vk::BufferCreateInfo::default()
        .size(core::mem::size_of_val(&emitter) as _)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC);
      let mut staging_alloc_create_info = vk_mem::AllocationCreateInfo::default();
      crate::apply_test_dedicated_alloc!(staging_alloc_create_info);
      staging_alloc_create_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
      staging_alloc_create_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED;
      let (staging_buf, staging_alloc, staging_alloc_info) = unsafe {
        allocator.create_buffer_get_info(&staging_buf_create_info, &staging_alloc_create_info)
      }?;
      let staging_janitor = AllocJanitor {
        buffer: staging_buf,
        alloc: staging_alloc,
        allocator,
      };

      // copy everything to staging and issue a host->transfer barrier
      unsafe {
        allocator.copy_memory_to_allocation(&staging_alloc, bytemuck::bytes_of(&emitter), 0)
      }?;
      let transfer_barrier = vk::BufferMemoryBarrier2::default()
        // the host side ...
        .src_stage_mask(vk::PipelineStageFlags2::HOST)
        // ... should have finished its memory write transactions from cache -> system RAM
        .src_access_mask(vk::AccessFlags2::HOST_WRITE)
        // before gpu transfer operations ...
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        // ... reads ...
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        // ... from this very buffer
        .buffer(staging_buf)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        // without queue ownership transfer
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
      let dep_info = vk::DependencyInfo::default()
        .buffer_memory_barriers(core::slice::from_ref(&transfer_barrier));
      unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info) };

      // now issue the copy operation
      let copy_region = vk::BufferCopy::default().size(core::mem::size_of_val(&emitter) as _);
      unsafe {
        self.device.cmd_copy_buffer(
          cmd,
          staging_buf,
          gpu_buffer,
          core::slice::from_ref(&copy_region),
        );
      };

      // issue transfer->compute barrier
      let compute_barrier = vk::BufferMemoryBarrier2::default()
        // transfer operation ...
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        // ... should have finished writing
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        // before gpu compute shader ...
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        // ... accesses by read ...
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
        // ... this buffer
        .buffer(gpu_buffer)
        .offset(0)
        .size(vk::WHOLE_SIZE)
        // .. without queue ownership transfer
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
      let dep_info = vk::DependencyInfo::default()
        .buffer_memory_barriers(core::slice::from_ref(&compute_barrier));
      unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info) };

      // defuse cleaner for staging alloc and discard it for next timeline
      core::mem::forget(staging_janitor);
      self.kernels.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        staging_buf,
        staging_alloc,
        discard_timeline,
      );
    }

    // defuse the cleaner for our allocation
    core::mem::forget(gpu_janitor);

    // discard gpu emitter allocator for next timeline
    self.kernels.discard_pool.discard_buffer(
      allocator.as_allocator_view(),
      gpu_buffer,
      gpu_alloc,
      discard_timeline,
    );

    Ok((gpu_buffer, gpu_alloc, gpu_buffer_address))
  }

  /// 2
  /// Note: force should have been already converted in SI, relative to particle system frame
  #[named]
  pub fn cmd_particle_system_next_forces(
    &self,
    cmd: vk::CommandBuffer,
    push_constants: &gpu::compute_push_constants::ApplyEmittersDirectNewPushConstants,
    skip_bind: bool,
  ) -> GpuResult<()> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );
    let pipeline: vk::Pipeline = self.kernels.pipelines.apply_emitters_direct_new;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout: vk::PipelineLayout = self.kernels.pipelines.pipeline_layout;

    // SAFETY: property constructed Pipelines has local sizes for all shaders
    let local_size_x =
      unsafe { self.kernels.pipelines.wg_sizes.get(&pipeline.as_raw()).unwrap_unchecked() }[0];
    let particle_count = gpu::new_particles::MAX_PARTICLES_PER_SYSTEM as u32;

    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      let num_workgroups_x: u32 =
        Self::particle_system_shaders_num_workgroups(local_size_x, particle_count);
      self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
    }

    Ok(())
  }

  /// 3
  #[named]
  pub fn cmd_particle_system_velocity_vertlet_correction(
    &self,
    cmd: vk::CommandBuffer,
    push_constants: &gpu::compute_push_constants::IntegrateParticlesP45NewPushConstants,
    skip_bind: bool,
  ) -> GpuResult<()> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );
    let pipeline: vk::Pipeline = self.kernels.pipelines.integrate_particles_p4_5_new;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout: vk::PipelineLayout = self.kernels.pipelines.pipeline_layout;
    // SAFETY: properly constructed [`super::physics::PhysicsPipelines`] have sizes for all
    // pipelines
    let local_size_x =
      unsafe { self.kernels.pipelines.wg_sizes.get(&pipeline.as_raw()).unwrap_unchecked() }[0];
    let particle_count = gpu::new_particles::MAX_PARTICLES_PER_SYSTEM as u32;
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      let num_workgroups_x =
        Self::particle_system_shaders_num_workgroups(local_size_x, particle_count);
      self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
    }

    Ok(())
  }

  // Because the activeChunkCount is modified dynamically on the GPU
  // (by the emitter and the compactor), the CPU doesn't strictly know how many chunks are
  // mapped at any given frame. To avoid stalling the pipeline to read that value back to
  // the CPU, use vkCmdDispatchIndirect.
  //
  // TODO maintain a tiny Vulkan buffer containing a VkDispatchIndirectCommand struct.
  // Before this pass, a tiny compute shader writes the activeChunkCount into the
  // x dimension of that buffer (setting y and z to 1).
  // <pre>
  //   // Instructs the GPU to read the X workgroup count directly from the buffer
  //   device.cmd_dispatch_indirect(command_buffer, indirect_buffer, 0);
  // </pre>
  #[named]
  pub fn cmd_particle_system_compaction(
    &self,
    cmd: vk::CommandBuffer,
    last_compaction_unscaled_us: i64,
    now_unscaled_us: i64,
    push_constants: &gpu::compute_push_constants::NewParticlesCompactPushConstants,
    skip_bind: bool,
  ) -> GpuResult<i64> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );

    {
      let res = self.res.read();
      if res.particle_system_manager.is_none() {
        return Err(gpu_err!("particle_system_manager absent"));
      }
      let psm = unsafe { res.particle_system_manager.as_ref().unwrap_unchecked() };

      // check emission interval
      if last_compaction_unscaled_us - now_unscaled_us <= psm.compaction_us() {
        return Ok(last_compaction_unscaled_us);
      }
    }

    let pipeline = self.kernels.pipelines.new_particles_compact;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout = self.kernels.pipelines.pipeline_layout;
    // for now we are not using indirect dispatch but over-dispatch
    let num_workgroups_x = gpu::new_particles::MAX_CHUNKS as u32;

    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
    }

    Ok(now_unscaled_us)
  }

  /// Caller is responsible to check whether it is appropriate to call this by first calling
  /// compaction (if bulk calling for skip bind, build a map of particle system id -> compact yes or
  /// no)
  #[named]
  pub fn cmd_particle_system_compaction_reset(
    &self,
    cmd: vk::CommandBuffer,
    push_constants: &gpu::compute_push_constants::NewParticlesCompactResetPushConstants,
    skip_bind: bool,
  ) -> GpuResult<()> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );

    let pipeline = self.kernels.pipelines.new_particles_compact_reset;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout = self.kernels.pipelines.pipeline_layout;
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      self.device.cmd_dispatch(cmd, 1, 1, 1);
    }

    Ok(())
  }

  #[named]
  pub fn cmd_particle_system_offset_particles(
    &self,
    cmd: vk::CommandBuffer,
    push_constants: &gpu::compute_push_constants::NewParticlesOffsetParticlesPushConstants,
    skip_bind: bool,
  ) -> GpuResult<()> {
    debug_assert!(
      super::physics::USE_PARTICLE_SYSTEM_V2.load(core::sync::atomic::Ordering::Relaxed)
    );
    let pipeline = self.kernels.pipelines.new_particles_offset_particles;
    if !skip_bind {
      unsafe { self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
    }

    let push_constants_bytes: &[u8] = bytemuck::bytes_of(push_constants);
    let pipeline_layout = self.kernels.pipelines.pipeline_layout;
    // SAFETY: properly constructed [`super::physics::PhysicsPipelines`] have sizes for all
    // pipelines
    let local_size_x =
      unsafe { self.kernels.pipelines.wg_sizes.get(&pipeline.as_raw()).unwrap_unchecked() }[0];
    let particles_count = gpu::new_particles::MAX_PARTICLES_PER_SYSTEM as u32;
    let num_workgroups_x =
      Self::particle_system_shaders_num_workgroups(local_size_x, particles_count);
    unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );
      self.device.cmd_dispatch(cmd, num_workgroups_x, 1, 1);
    }

    Ok(())
  }

  pub fn get_compute_queue(&self) -> Queue {
    self.queues.get_compute_queue()
  }

  pub fn get_graphics_queue(&self) -> Queue {
    self.queues.get_graphics_queue()
  }

  pub fn submit_paint_image_transition(
    &self,
    cmd_handle: gpu::CommandBufferHandle,
    mesh_id: crate::gpu::RenderableInstanceId,
    old_layout: ash::vk::ImageLayout,
    new_layout: ash::vk::ImageLayout,
  ) -> GpuResult<()> {
    let cmd_buffers = &self.recording_command_buffers;
    let data = cmd_buffers
      .get(&(cmd_handle, QueueRole::Graphics))
      .ok_or(gpu_err_invalid_cmd!())?;
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

  /// Constructor
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

    if chosen_physical_device_query_result
      .optional_extensions
      .contains(utils::OptionalExtensionSupportFlags::NATIVE_FLOAT16)
    {
      required_features.shader_float16_int8.shader_float16 = vk::TRUE;
    }

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
      hooks::load_device_with_hooks(&instance.instance, physical_device, &device_create_info)
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

    #[cfg(debug_assertions)]
    let telemetry_query_pool = {
      let create_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::TIMESTAMP)
        .query_count(1024);
      unsafe { device.create_query_pool(&create_info, None).ok() }
    };

    #[cfg(target_vendor = "apple")]
    let metal_objects = ash::ext::metal_objects::Device::new(&instance.instance, &device);
    let device = LogicalDevice {
      timeline_semaphore,
      handle: device,
      submission_lock: spin::Mutex::new(()),
      submission_lock_compute: spin::Mutex::new(()),
      create_renderpass2,
      synchronization2,
      buffer_device_address,
      swapchain_maintenance1,
      #[cfg(target_vendor = "apple")]
      metal_objects,
      #[cfg(debug_assertions)]
      debug_utils,
      #[cfg(debug_assertions)]
      telemetry_query_pool,
      max_per_stage_descriptor_update_after_bind_samplers: chosen_physical_device_query_result
        .max_per_stage_descriptor_update_after_bind_samplers,
      max_per_stage_descriptor_samplers: chosen_physical_device_query_result
        .physical_device_properties
        .limits
        .max_per_stage_descriptor_samplers,
      max_descriptor_set_update_after_bind_samplers: chosen_physical_device_query_result
        .max_descriptor_set_update_after_bind_samplers,
    };
    let mut res = match DeviceResources::new(
      instance.as_ref(),
      physical_device,
      &device,
      chosen_physical_device_query_result.unique_family_indices_set().iter(),
      queues.get_compute_queue(),
    ) {
      Ok(r) => r,
      Err(e) => {
        aethervk_oshal_rlib::log!("Device::new error in DeviceResources::new, destroying device!");
        unsafe { device.handle.destroy_device(None) };
        return Err(e);
      }
    };

    let kernels = match VulkanComputeKernels::new(
      &device,
      chosen_physical_device_query_result.debug_shaders,
      chosen_physical_device_query_result.subgroup_size,
      chosen_physical_device_query_result.is_cpu,
      chosen_physical_device_query_result
        .optional_extensions
        .contains(utils::OptionalExtensionSupportFlags::NATIVE_FLOAT16),
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
      recording_command_buffers: dashmap::DashMap::with_capacity(32),
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

      #[cfg(debug_assertions)]
      if let Some(pool) = self.device.telemetry_query_pool {
        unsafe { self.device.destroy_query_pool(pool, None) };
      }

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
    let arch = pe.archetypes().registry.read();
    let arch_ref = arch.get(&ArchetypeId::Measurement).ok_or(gpu_err_archetype_absent!())?;
    let pipeline_key = arch_ref.pipeline_key();

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
    let res_guard = self.res.read();
    let pe = wait_for_pe!(res_guard, handle)?;
    let archetype_guard = pe.archetypes().registry.read();
    let archetype = archetype_guard.get(&ArchetypeId::Gizmo).ok_or(gpu_err_archetype_absent!())?;
    let pipeline_key = archetype.pipeline_key();

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
  fn get_marker_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let pe = wait_for_pe!(res_guard, handle)?;
    let arch = pe.archetypes().registry.read();
    let arch_ref = arch.get(&ArchetypeId::Marker).ok_or(gpu_err_archetype_absent!())?;
    let pipeline_key = arch_ref.pipeline_key();

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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let arch_lock = pe_lock.archetypes().registry.read();
        let arch = arch_lock.get(&ArchetypeId::Billboard).ok_or(gpu_err_archetype_absent!())?;
        let prep = arch.prepare_update(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = self.res.read();
          // vkCreateGraphicsPipelines is allowed execution with debug tracked locks
          state.pipeline_pool.get_or_create_graphics_pipeline(
            &self.device,
            &update.main_graphics_info,
            rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &outline_info,
              rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(archetypes_struct::CompiledArchetypeData {
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
          let mut arch_lock = pe_lock.archetypes().registry.write();
          let arch =
            arch_lock.get_mut(&ArchetypeId::Billboard).ok_or(gpu_err_archetype_absent!())?;
          arch.commit_update(c);
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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let arch_lock = pe_lock.archetypes().registry.read();
        let arch = arch_lock.get(&ArchetypeId::Cursor).ok_or(gpu_err_archetype_absent!())?;
        let prep = arch.prepare_update(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = self.res.read();
          // vkCreateGraphicsPipelines can be executed while holding debug tracked locks
          state.pipeline_pool.get_or_create_graphics_pipeline(
            &self.device,
            &update.main_graphics_info,
            rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &outline_info,
              rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(archetypes_struct::CompiledArchetypeData {
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
          let mut arch_lock = pe_lock.archetypes().registry.write();
          let arch = arch_lock.get_mut(&ArchetypeId::Cursor).ok_or(gpu_err_archetype_absent!())?;
          arch.commit_update(c);
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
        let arch_lock = pe_lock.archetypes().registry.read();
        let arch = arch_lock.get(&ArchetypeId::Measurement).ok_or(gpu_err_archetype_absent!())?;
        let prep = arch.prepare_update(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = self.res.read();
          state.pipeline_pool.get_or_create_graphics_pipeline(
            &self.device,
            &update.main_graphics_info,
            rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &outline_info,
              rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(archetypes_struct::CompiledArchetypeData {
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
          let mut arch_lock = pe_lock.archetypes().registry.write();
          let arch = arch_lock
            .get_mut(&ArchetypeId::Measurement)
            .ok_or(gpu_err_archetype_absent!())?;
          arch.commit_update(c);
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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let arch_lock = pe_lock.archetypes().registry.read();
        let arch = arch_lock.get(&ArchetypeId::Marker).ok_or(gpu_err_archetype_absent!())?;
        let prep = arch.prepare_update(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = self.res.read();
          state.pipeline_pool.get_or_create_graphics_pipeline(
            &self.device,
            &update.main_graphics_info,
            rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &outline_info,
              rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(archetypes_struct::CompiledArchetypeData {
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
          let mut arch_lock = pe_lock.archetypes().registry.write();
          let arch = arch_lock.get_mut(&ArchetypeId::Marker).ok_or(gpu_err_archetype_absent!())?;
          arch.commit_update(c);
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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        let pe_lock = wait_for_pe!(state, h)?;
        let format = pe_lock.format();
        let arch_lock = pe_lock.archetypes().registry.read();
        let arch = arch_lock.get(&ArchetypeId::Gizmo).ok_or(gpu_err_archetype_absent!())?;
        let prep = arch.prepare_update(format, &state.renderpasses)?;
        Ok((h, prep))
      })?
      .execute(|(h, prep), rollback| {
        let compiled = if let Some(update) = prep {
          let state = self.res.read();
          state.pipeline_pool.get_or_create_graphics_pipeline(
            &self.device,
            &update.main_graphics_info,
            rollback,
          )?;
          let mut outline_data = None;
          if let Some(outline_info) = update.outline_graphics_info {
            state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &outline_info,
              rollback,
            )?;
            outline_data = Some((outline_info.pipeline_key(), outline_info));
          }
          Some(archetypes_struct::CompiledArchetypeData {
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
          let mut arch_lock = pe_lock.archetypes().registry.write();
          let arch = arch_lock.get_mut(&ArchetypeId::Gizmo).ok_or(gpu_err_archetype_absent!())?;
          arch.commit_update(c);
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
    let res_guard = self.res.read();
    let pe = wait_for_pe!(res_guard, handle)?;
    let archetype = pe.archetypes().registry.read();
    let archetype_ref = archetype.get(&ArchetypeId::Cursor).ok_or(gpu_err_archetype_absent!())?;
    let pipeline_key = archetype_ref.pipeline_key();

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
  fn is_cpu_device(&self) -> bool {
    self.kernels.pipelines.is_lavapipe
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
        if let Some(arena) = locks::DebugTrackedRwLock::write(&state.frame_staging_arena).as_mut() {
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
      measurement: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::MeasurementRenderResourceArchetypeArena>>,
      >,
      marker:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::MarkerRenderResourceArchetypeArena>>>,
      billboard: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::BillboardRenderResourceArchetypeArena>>,
      >,
      trajectory: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::TrajectoryRenderResourceArchetypeArena>>,
      >,
      ui: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::UiRenderResourceArchetypeArena>>>,
      cursor:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::CursorRenderResourceArchetypeArena>>>,
      text2:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::Text2RenderResourceArchetypeArena>>>,
      sphere_gizmo: Option<
        alloc::sync::Arc<DebugTrackedRwLock<resources::SphereGizmoRenderResourceArchetypeArena>>,
      >,
      gizmo:
        Option<alloc::sync::Arc<DebugTrackedRwLock<resources::GizmoRenderResourceArchetypeArena>>>,
      dust: Option<alloc::sync::Arc<DebugTrackedRwLock<resources::DustRenderArchetypeArena>>>,
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
          mesh2: state.physical_mesh2_render_archetype_arena.clone(),
          sun: state.sun_render_archetype_arena.clone(),
          sky: state.sky_render_archetype_arena.clone(),
          background: state.background_render_archetype_arena.clone(),
          grid: state.grid_render_archetype_arena.clone(),
          measurement: state.measurement_render_archetype_arena.clone(),
          marker: state.marker_render_archetype_arena.clone(),
          billboard: state.billboard_render_archetype_arena.clone(),
          trajectory: state.trajectory_render_archetype_arena.clone(),
          ui: state.ui_render_archetype_arena.clone(),
          cursor: state.cursor_render_archetype_arena.clone(),
          text2: state.text2_render_archetype_arena.clone(),
          sphere_gizmo: state.sphere_gizmo_render_archetype_arena.clone(),
          gizmo: state.gizmo_render_archetype_arena.clone(),
          dust: state.dust_render_archetype_arena.clone(),
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

        let mut arena_ctx = resources::ArenaCreationContext::new_empty(device, allocator, discard_pool, rollback);
        arena_ctx.queue = Some(&queue);
        arena_ctx.staging_arena = staging_arena;

        // Extract the format ONCE before the closure to prevent borrow checking overlap
        let pe_format = pe.format();

        macro_rules! init_arch {
          // standard
          ($arena_field:ident, $ensure_fn:ident, $archetype_id:expr, $arena_type:ident, $create_fn:ident) => {
            let needs_init = !pe.archetypes().registry.read().contains_key(&$archetype_id);
            if needs_init {
              let (vkey, fkey) = {
                let mut sm = locks::DebugTrackedRwLock::write(shader_manager);
                $ensure_fn(device, &mut sm)?
              };
              let (vertex_shader, fragment_shader) = {
                let sm_read = locks::DebugTrackedRwLock::read(shader_manager);
                let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                (vs, fs)
              };

              if arenas.$arena_field.is_none() {
                arena_ctx.vertex_shader = Some(vertex_shader.module.get());
                arena_ctx.fragment_shader = Some(fragment_shader.module.get());
                let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&mut arena_ctx)?;
                arenas.$arena_field = Some(alloc::sync::Arc::new(locks::DebugTrackedRwLock::new(new_arena)));
              }
              pe.archetypes_mut().$create_fn(
                device, &*vertex_shader, &*fragment_shader, depth_stencil_format, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), arena_ctx.rollback
              )?;
            }
          };
          // text
          ($arena_field:ident, $ensure_fn:ident, $archetype_id:expr, $arena_type:ident, $create_fn:ident, text) => {
            let needs_init = !pe.archetypes().registry.read().contains_key(&$archetype_id);
            if needs_init {
              let (vkey, fkey) = {
                let mut sm = locks::DebugTrackedRwLock::write(shader_manager);
                $ensure_fn(device, &mut sm)?
              };
              let (vertex_shader, fragment_shader) = {
                let sm_read = locks::DebugTrackedRwLock::read(shader_manager);
                let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                (vs, fs)
              };

              if arenas.$arena_field.is_none() {
                arena_ctx.vertex_shader = Some(vertex_shader.module.get());
                arena_ctx.fragment_shader = Some(fragment_shader.module.get());
                let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&mut arena_ctx)?;
                arenas.$arena_field = Some(alloc::sync::Arc::new(locks::DebugTrackedRwLock::new(new_arena)));
              }
              pe.archetypes_mut().$create_fn(
                device, &*vertex_shader, &*fragment_shader, depth_stencil_format, &queue, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), arena_ctx.rollback
              )?;
            }
          };
          // ref_alloc
          ($arena_field:ident, $ensure_fn:ident, $archetype_id:expr, $arena_type:ident, $create_fn:ident, ref_alloc) => {
            init_arch!($arena_field, $ensure_fn, $archetype_id, $arena_type, $create_fn);
          };
          // mesh
          ($arena_field:ident, $ensure_fn:ident, $archetype_id:expr, $arena_type:ident, $create_fn:ident, mesh) => {
            let needs_init = !pe.archetypes().registry.read().contains_key(&$archetype_id);
            if needs_init {
              let (vkey, fkey, ovkey, ofkey) = {
                let mut sm = locks::DebugTrackedRwLock::write(shader_manager);
                $ensure_fn(device, &mut sm)?
              };
              let (vs, fs, ovs, ofs) = {
                let sm_read = locks::DebugTrackedRwLock::read(shader_manager);
                let vs = get_shader(&sm_read, vkey, ash::vk::ShaderStageFlags::VERTEX)?;
                let fs = get_shader(&sm_read, fkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                let ovs = get_shader(&sm_read, ovkey, ash::vk::ShaderStageFlags::VERTEX)?;
                let ofs = get_shader(&sm_read, ofkey, ash::vk::ShaderStageFlags::FRAGMENT)?;
                (vs, fs, ovs, ofs)
              };

              if arenas.$arena_field.is_none() {
                arena_ctx.vertex_shader = Some(vs.module.get());
                arena_ctx.fragment_shader = Some(fs.module.get());
                arena_ctx.outline_vertex_shader = Some(ovs.module.get());
                arena_ctx.outline_fragment_shader = Some(ofs.module.get());
                let new_arena = <resources::$arena_type as resources::ArchetypeArenaCreate>::new_arena(&mut arena_ctx)?;
                arenas.$arena_field = Some(alloc::sync::Arc::new(locks::DebugTrackedRwLock::new(new_arena)));
              }
              pe.archetypes_mut().$create_fn(
                device, &*vs, &*fs, &*ovs, &*ofs, depth_stencil_format, &queue, pe_format, allocator, discard_pool, renderpasses, pipeline_pool, timeline, arenas.$arena_field.as_ref().unwrap().clone(), arena_ctx.rollback
              )?;
            }
          };
        }

        init_arch!(mesh2, ensure_physical_mesh2_shader_modules, ArchetypeId::Mesh, ForwardMesh2RenderResourceArchetypeArena, create_physical_mesh2_archetype, mesh);
        init_arch!(sun, ensure_sun_shader_modules, ArchetypeId::Sun, SunRenderResourceArchetypeArena, create_sun_archetype);
        init_arch!(sky, ensure_sky_shader_modules, ArchetypeId::Sky, SkyRenderResourceArchetypeArena, create_sky_archetype);
        init_arch!(background, ensure_background_shader_modules, ArchetypeId::Background, BackgroundRenderResourceArchetypeArena, create_background_archetype);
        init_arch!(grid, ensure_grid_shader_modules, ArchetypeId::Grid, GridRenderResourceArchetypeArena, create_grid_archetype);
        init_arch!(measurement, ensure_measurement_shader_modules, ArchetypeId::Measurement, MeasurementRenderResourceArchetypeArena, create_measurement_archetype);
        init_arch!(marker, ensure_marker_shader_modules, ArchetypeId::Marker, MarkerRenderResourceArchetypeArena, create_marker_archetype);
        init_arch!(billboard, ensure_billboard_shader_modules, ArchetypeId::Billboard, BillboardRenderResourceArchetypeArena, create_billboard_archetype);
        init_arch!(trajectory, ensure_trajectory_shader_modules, ArchetypeId::Trajectory, TrajectoryRenderResourceArchetypeArena, create_trajectory_archetype, ref_alloc);
        init_arch!(ui, ensure_ui_shader_modules, ArchetypeId::Ui, UiRenderResourceArchetypeArena, create_ui_archetype, ref_alloc);
        init_arch!(cursor, ensure_cursor_shader_modules, ArchetypeId::Cursor, CursorRenderResourceArchetypeArena, create_cursor_archetype);
        init_arch!(text2, ensure_text2_shader_modules, ArchetypeId::Text, Text2RenderResourceArchetypeArena, create_text2_archetype, text);
        init_arch!(sphere_gizmo, ensure_sphere_gizmo_shader_modules, ArchetypeId::SphereGizmo, SphereGizmoRenderResourceArchetypeArena, create_sphere_gizmo_archetype);
        init_arch!(gizmo, ensure_gizmo_shader_modules, ArchetypeId::Gizmo, GizmoRenderResourceArchetypeArena, create_gizmo_archetype);
        init_arch!(dust, ensure_dust_shader_modules, ArchetypeId::Particles, DustRenderArchetypeArena, create_dust_archetype);


        Ok((pe, arenas))
      })
      .commit(|state, execute_result| {
        let (pe, arenas) = execute_result?;

        state.live_presentation_engines.insert(handle, pe);

        // Always save arenas first, even if result is an Error, to prevent resource leaks
        // and safely reuse successfully established arenas upon subsequent retries.
        state.physical_mesh2_render_archetype_arena = arenas.mesh2;
        state.sun_render_archetype_arena = arenas.sun;
        state.sky_render_archetype_arena = arenas.sky;
        state.background_render_archetype_arena = arenas.background;
        state.grid_render_archetype_arena = arenas.grid;
        state.measurement_render_archetype_arena = arenas.measurement;
        state.marker_render_archetype_arena = arenas.marker;
        state.billboard_render_archetype_arena = arenas.billboard;
        state.trajectory_render_archetype_arena = arenas.trajectory;
        state.ui_render_archetype_arena = arenas.ui;
        state.cursor_render_archetype_arena = arenas.cursor;
        state.text2_render_archetype_arena = arenas.text2;
        state.sphere_gizmo_render_archetype_arena = arenas.sphere_gizmo;
        state.gizmo_render_archetype_arena = arenas.gizmo;
        state.dust_render_archetype_arena = arenas.dust;

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

    utils::VulkanTransaction::new(&*self.res, &self.device)
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
        let preps_res = {
          let registry = pe.archetypes().registry.read();
          let mut preps = alloc::vec::Vec::with_capacity(16);
          for (id, arch) in registry.iter() {
            if let Some(prep) = arch.prepare_update(format, &state.renderpasses)? {
              preps.push((*id, prep));
            }
          }
          Ok(preps)
        };

        let all_preps_opt = match preps_res {
          Ok(preps) => {
            if preps.is_empty() {
              None
            } else {
              Some(Ok(preps))
            }
          }
          Err(e) => Some(Err(e)),
        };
        Ok((pe, backup, resize_result, all_preps_opt))
      })
      .execute(|(pe, backup, resize_result, all_preps_opt), rollback| {
        let preps = match all_preps_opt {
          Some(Ok(p)) => p,
          Some(Err(e)) => return Ok((pe, backup, resize_result, e)),
          None => return Ok((pe, backup, resize_result, Ok(None))),
        };

        let mut compiled = alloc::vec::Vec::with_capacity(16);
        let is_err = 'compiled_resize: {
          // create pipeline is one of these methods which is not lock-guarded
          let state = self.res.read();

          for (id, update) in preps {
            if let Err(e) = state.pipeline_pool.get_or_create_graphics_pipeline(
              &self.device,
              &update.main_graphics_info,
              rollback,
            ) {
              break 'compiled_resize Some(e);
            }

            let mut outline_data = None;
            if let Some(outline_info) = update.outline_graphics_info {
              if let Err(e) = state.pipeline_pool.get_or_create_graphics_pipeline(
                &self.device,
                &outline_info,
                rollback,
              ) {
                break 'compiled_resize Some(e);
              }

              outline_data = Some((outline_info.pipeline_key(), outline_info));
            }
            compiled.push((
              id,
              archetypes_struct::CompiledArchetypeData {
                pipeline_key: update.main_graphics_info.pipeline_key(),
                graphics_info: update.main_graphics_info,
                outline_data,
              },
            ));
          }

          None
        };

        Ok((
          pe,
          backup,
          resize_result,
          match is_err {
            Some(e) => Err(e),
            None => Ok(Some(compiled)),
          },
        ))
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
          let mut registry = pe.archetypes_mut().registry.write();
          for (id, data) in compiled {
            if let Some(arch) = registry.get_mut(&id) {
              arch.commit_update(data);
            }
          }
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
    let res_guard = self.res.read();
    let engine = wait_for_pe!(res_guard, handle)?;
    let e = engine.extent();
    Ok([e.0, e.1])
  }

  #[named]
  fn is_presentation_engine_windowless(&self, handle: PresentationEngineHandle) -> GpuResult<bool> {
    let res_guard = self.res.read();
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
        let timeline_sem = state.timeline_manager.semaphore.get();
        Ok((pe, backup, timeline_sem))
      })?
      .execute(|(mut pe, backup, timeline_sem), rollback| {
        // EXECUTE lock-free!
        // `vkAcquireNextImageKHR` natively blocks the CPU waiting for VSync.
        // Because `pe` is extracted, streaming/audio threads can still lock `self.res`!
        let result = pe.acquire_next_image(&self.device, timeline_sem, rollback);

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
    utils::VulkanTransaction::new(&*self.res, &self.device)
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
  fn get_physical_mesh2_resources(
    &self,
    asset_hash: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let res_guard = self.res.read();
    let (pipeline_key, outline_pipeline_key) = {
      let live_pes = &res_guard.live_presentation_engines;
      let pe = wait_for_pe_direct!(live_pes, handle)?;
      let arch_guard = pe.archetypes().registry.read();
      let arch = arch_guard.get(&ArchetypeId::Mesh).ok_or(gpu_err_archetype_absent!())?;
      let pipeline_key = arch.pipeline_key();
      let outline_pipeline_key = arch.outline_pipeline_key();
      (pipeline_key, outline_pipeline_key)
    };

    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(asset_hash);

    if let Some(entry) = res_guard.physical_mesh2_resources.get(&physical_mesh_id) {
      if let resources::ResourceState::Ready(resource) = entry.value() {
        return Ok(ResourceUploadResult {
          pipeline: pipeline_key,
          outline_pipeline: outline_pipeline_key,
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
    component: &StaticMeshComponent,
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
        let res_guard = self.res.read();
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
          // Return cloned/extracted data needed for execution
          Ok(meshutils::CreateResourcesState::new(state, component, h)?)
        })?
        .execute(|create_resource_state, rollback| {
          let allocator = create_resource_state.vma;

          let mut resource_opt = None;
          let mut texture_flags_out = TextureFlags::empty();

          let ptr = create_resource_state.staging_arena_ptr;
          let discard_pool_ptr = create_resource_state.discard_pool_ptr;
          let transient_res = self.run_transient_commands(|transient_cmd| {
            let staging_arena = unsafe { ptr.as_ref().unwrap() };
            let discard_pool = unsafe { discard_pool_ptr.as_ref().unwrap() };
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
                allocator.destroy_image(img_h, &mut alloc_h);
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
                allocator.destroy_image(img_h, &mut alloc_h);
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
                allocator.destroy_image(img_h, &mut alloc_h);
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
              let mut alloc_h = img.allocation;
              rollback.defer(move |dev| unsafe {
                dev.destroy_image_view(view_h, None);
                allocator.destroy_image(img_h, &mut alloc_h);
              });
              Some(img)
            });

            let material_data = meshutils::pbr_material_data(component, texture_flags);
            let object_data = meshutils::object_data_identity_matrix();

            let (_, descriptor_set) = {
              let layout = create_resource_state.arena_arc.read().descriptor_set_layout;
              create_resource_state.descriptor_pool_arc.allocate_and_get_active_pool(
                &self.device,
                layout,
                discard_pool,
                u64::MAX,
                debug_name,
                rollback,
              )?
            };

            let emissive_paint_image = {
              let img = resources::Image::new_paint_image(
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
              let mut alloc_h = img.allocation;
              rollback.defer(move |dev| unsafe {
                dev.destroy_image_view(view_h, None);
                allocator.destroy_image(img_h, &mut alloc_h);
              });
              img
            };

            let dummy_texture = {
              let arena_r = create_resource_state.arena_arc.read();
              resources::Image {
                image: arena_r.dummy_texture_handle.image,
                image_view: arena_r.dummy_texture_handle.image_view,
                allocation: arena_r.dummy_texture_handle.allocation,
              }
            };

            let resource = unsafe {
              resources::ForwardMesh2RenderResource::new(
                &self.device,
                allocator,
                transient_cmd,
                staging_arena,
                resources::ForwardMesh2RenderResourceParams {
                  position_data: &create_resource_state.position_data,
                  attribute_data: &create_resource_state.attribute_data,
                  index_data: &component.mesh.indices,
                  material_data: &material_data,
                  object_data: &object_data,
                  albedo_image,
                  normal_image,
                  roughness_image,
                  ao_image,
                  sky_image: create_resource_state.sky_image_clone.or_else(|| {
                    Some(resources::Image {
                      image: dummy_texture.image,
                      image_view: dummy_texture.image_view,
                      allocation: dummy_texture.allocation,
                    })
                  }),
                  emissive_paint_image: Some(emissive_paint_image),
                  sampler: utils::NonZeroHandle::new_unchecked(
                    create_resource_state.linear_sampler,
                  ),
                  descriptor_set: NonZeroHandle::new_unchecked(descriptor_set),
                  dummy_texture: &dummy_texture,
                  debug_name,
                },
                rollback,
              )?
            }; // last op, so no rollback defer

            resource_opt = Some(resource);
            texture_flags_out = texture_flags;
            Ok(())
          })?;
          let timeline =
            DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;
          let discard_pool = unsafe { discard_pool_ptr.as_ref().unwrap() };
          discard_pool.discard_type_erased(transient_res, timeline + 2);

          Ok((
            create_resource_state.pipeline_key,
            create_resource_state.outline_pipeline_key,
            resource_opt.unwrap(),
            texture_flags_out,
          ))
        })
        .commit_read(|state, execute_result| {
          let (pipeline_key, outline_pipeline_key, resource, texture_flags) = execute_result?;

          let old_resource = state
            .physical_mesh2_resources
            .insert(physical_mesh_id, resources::ResourceState::Ready(resource));

          if let Some(resources::ResourceState::Ready(mut old)) = old_resource {
            let timeline = state.timeline_manager.get_next_submit_value();
            old.discard(&state.discard_pool, timeline);
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

    utils::VulkanTransaction::new(&*self.res, &self.device)
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
        let arch_guard = pe.archetypes().registry.read();
        let archetype_ref =
          arch_guard.get(&ArchetypeId::Mesh).ok_or(gpu_err_archetype_absent!())?;
        let mesh_arena = state
          .physical_mesh2_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent"))?;
        let pipeline_layout = mesh_arena.read().pipeline_layout.get();

        // 3. Allocate from Staging Arena
        let mut staging_arena_guard = state.frame_staging_arena.write();
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
            sphere_center_radius: [0.0; 4],
            grid_color_density: [0.0; 4],
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
            _pad: 0,
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
        if locks::DebugTrackedRwLock::read(&state.sky_image).is_some() {
          return Ok::<_, GpuError>(None); // Signal to skip execution
        }

        // 2. Safely acquire shader module
        let comp_key = {
          let mut sm = locks::DebugTrackedRwLock::write(&state.shader_manager);
          ensure_skygen_shader_module(&self.device, &mut sm)?
        };

        let shader_module = locks::DebugTrackedRwLock::read(&state.shader_manager)
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
          unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None) }
            .with_name(&self.device, "VkPipelineLayout_Sky")?;

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

        let mut _cleanup = TransientCleanup {
          device: &self.device,
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
        }
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
          .semaphore_type(vk::SemaphoreType::TIMELINE)
          .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_info, None) }?;

        let signal_semaphores = [timeline_semaphore];
        let signal_values = [1];
        let mut timeline_info =
          vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

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
        let mut wsky_image = locks::DebugTrackedRwLock::write(&state.sky_image);

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
    let res_guard = self.res.read();
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_direct!(live_pes, handle)?;
    let pe = pe_lock;
    let arch_guard = pe.archetypes().registry.read();
    let archetype_ref =
      arch_guard.get(&ArchetypeId::Billboard).ok_or(gpu_err_archetype_absent!())?;
    let pipeline_key = archetype_ref.pipeline_key();

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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, _h| {
        let pe = wait_for_pe!(state, handle)?;

        let archetype_guard = pe.archetypes().registry.read();
        archetype_guard.get(&ArchetypeId::Gizmo).ok_or(gpu_err_archetype_absent!())?;
        let arena_arc = state
          .gizmo_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent"))?
          .clone();
        let descriptor_set = arena_arc.read().descriptor_set.get();
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
        let arena = arena_arc.read();
        let mut buffers = arena.host_buffers.write();

        if let Some(old_buffer) = buffers.insert(buffer_index, new_buffer) {
          state.discard_pool.discard_buffer(
            state.allocator.allocator.as_allocator_view(),
            old_buffer.buffer.get(),
            old_buffer.allocation,
            timeline,
          );
        }

        Ok(buffer_index)
      })
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

    let res_guard = self.res.read();
    let live_pes = &res_guard.live_presentation_engines;
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let mut archetype_guard = pe.archetypes_mut().registry.write();
    let archetype = archetype_guard
      .get_mut(&ArchetypeId::Trajectory)
      .ok_or(gpu_err_archetype_absent!())?;

    let arena_arc = res_guard
      .trajectory_render_archetype_arena
      .as_ref()
      .ok_or(gpu_err!("arena absent"))?;
    let mut arena_mut = arena_arc.write();

    arena_mut.tick = arena_mut.tick.wrapping_add(1);
    let current_tick = arena_mut.tick;

    // 1. GARBAGE COLLECTION: Purge curves missing for > 10 frames
    let mut to_remove = alloc::vec::Vec::with_capacity(16);
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

      let current_hash = resources::hash_trajectory(&traj_comp.control_points, model_mat);
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
          resources::CurveAllocation {
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
    let arena_read = arena_arc.read();

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
      let mut frame_staging_arena_wlock = res_guard.frame_staging_arena.write();
      let frame_staging_arena = frame_staging_arena_wlock.as_mut().unwrap();
      let (staging_offset, staging_ptr) = frame_staging_arena
        .allocate(total_staging_size as usize, 16)
        .ok_or(crate::gpu_err_device!())?;
      let staging_buffer = frame_staging_arena.buffer;
      drop(frame_staging_arena_wlock);

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
            arena_read.segments_buffer.get(),
            &vk_buffer_copies,
          );
        }
        if traj_size > 0 {
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            arena_read.trajectories_buffer.get(),
            &[traj_copy],
          );
        }
        if map_size > 0 {
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            arena_read.map_buffer.get(),
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

    let pipeline = archetype.pipeline_key();

    Ok(Some(crate::gpu::frame::TrajectoryBatchCall {
      pipeline,
      total_vertices: (max_subdivs + 1) * 2,
      total_segments,
      map_ptr: arena_read.map_ptr,
      traj_ptr: arena_read.trajectories_ptr,
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

    let res_guard = self.res.read();
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let archetype_guard = pe.archetypes_mut().registry.read();
    archetype_guard.get(&ArchetypeId::Ui).ok_or(gpu_err_archetype_absent!())?;
    let arena_lock = res_guard.ui_render_archetype_arena.as_ref().unwrap();

    let elements_ptr = unsafe {
      let mut arena_write = arena_lock.write();
      let data_ptr = res_guard.allocator.allocator.map_memory(&mut arena_write.elements_alloc)?;

      core::ptr::copy_nonoverlapping(
        ui_elements.as_ptr() as *const u8,
        data_ptr as *mut u8,
        ui_elements.len() * core::mem::size_of::<crate::gpu::UiElementGpu>(),
      );

      let valid_size =
        (ui_elements.len() * core::mem::size_of::<crate::gpu::UiElementGpu>()) as u64;
      let _ = res_guard.allocator.allocator.flush_allocation(
        &arena_write.elements_alloc,
        0,
        valid_size,
      );

      res_guard.allocator.allocator.unmap_memory(&mut arena_write.elements_alloc);

      let barrier = vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::HOST)
        .src_access_mask(vk::AccessFlags2::HOST_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT | vk::PipelineStageFlags2::VERTEX_SHADER,
        )
        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(arena_write.elements_buffer.get())
        .offset(0)
        .size(valid_size);

      let dep_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      let elements_ptr = arena_write.elements_ptr;
      arena_write.tick += 1;
      drop(arena_write); // drop lock before vulkan call

      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

      elements_ptr
    };

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

    let res_guard = self.res.read();
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
    let live_pes = &res_guard.live_presentation_engines;
    let pe_lock = wait_for_pe_mut_direct!(live_pes, handle)?;
    let mut pe = pe_lock;
    let archetype_guard = pe.archetypes_mut().registry.read();
    archetype_guard.get(&ArchetypeId::Text).ok_or(gpu_err_archetype_absent!())?;
    let arena_lock = res_guard.text2_render_archetype_arena.as_ref().unwrap();

    let glyphs_ptr = unsafe {
      let mut arena_write = arena_lock.write();
      let data_ptr = res_guard.allocator.allocator.map_memory(&mut arena_write.glyphs_alloc)?;

      core::ptr::copy_nonoverlapping(
        glyphs.as_ptr() as *const u8,
        data_ptr as *mut u8,
        glyphs.len() * core::mem::size_of::<crate::gpu::TextGlyphGpu>(),
      );

      let valid_size =
        (glyphs.len() * core::mem::size_of::<crate::gpu::TextGlyphGpu>()) as u64;
      let _ = res_guard.allocator.allocator.flush_allocation(
        &arena_write.glyphs_alloc,
        0,
        valid_size,
      );

      res_guard.allocator.allocator.unmap_memory(&mut arena_write.glyphs_alloc);

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
        .buffer(arena_write.glyphs_buffer.get())
        .offset(0)
        .size(valid_size);

      let dep_info =
        vk::DependencyInfo::default().buffer_memory_barriers(core::slice::from_ref(&barrier));
      let glyphs_ptr = arena_write.glyphs_ptr;
      drop(arena_write); // drop lock before vulkan call

      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);

      glyphs_ptr
    };

    Ok(Some(crate::gpu::Text2BatchCall {
      glyphs_ptr,
      total_glyphs: glyphs.len() as u32,
    }))
  }

  fn get_trajectory_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, ArchetypeId::Trajectory)
  }

  fn get_sun_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, ArchetypeId::Sun)
  }

  fn get_sky_pipeline_key(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, ArchetypeId::Sky)
  }

  #[named]
  fn get_background_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, ArchetypeId::Background)
  }

  #[named]
  fn get_grid_pipeline_kay(&self, handle: PresentationEngineHandle) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, ArchetypeId::Grid)
  }

  #[named]
  fn allocate_rasterized_font_atlas(
    &self,
    cmd: CommandBufferHandle,
    hash: u64,
    font_atlas: alloc::sync::Arc<FontAtlas>,
  ) -> GpuResult<u32> {
    let (command_buffer, handle) = self.get_cmd_and_pe(cmd)?;

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, _| {
        let a2_arc = state
          .text2_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text2"))?
          .clone();

        // 1. Prepare: Check for existing allocations and reserve new indices
        let prep2 = {
          let mut a2 = a2_arc.write();
          a2.prepare_upload_font_atlas(hash)?
        };

        if prep2.is_already_uploaded {
          return Ok((a2_arc, prep2, None));
        }

        let vma_view = state.allocator.allocator.as_allocator_view();
        let staging_arena_ptr = state
          .frame_staging_arena
          .read()
          .as_ref()
          .map(|a| a as *const _)
          .ok_or(gpu_err!("staging arena missing"))?;

        Ok((a2_arc, prep2, Some((vma_view, staging_arena_ptr))))
      })?
      .execute(|(a2_arc, prep2, exec_data), rollback| {
        let Some((vma_view, staging_arena_ptr)) = exec_data else {
          return Ok((a2_arc, prep2, None)); // Pass through if already uploaded
        };

        let staging_arena = unsafe { &*staging_arena_ptr };

        let texture = crate::simulation::comet::Texture {
          data: font_atlas.image_data.clone().into(),
          format: crate::simulation::comet::TexelFormat::R8_UNORM,
          width: font_atlas.width,
          height: font_atlas.height,
          has_mipmaps: false,
        };

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

        Ok((a2_arc, prep2, Some(image2)))
      })
      .commit_read(|_state, execute_result| {
        let (a2_arc, prep2, images_opt) = execute_result?;

        // 3. Commit: Finalize map insertion with new data
        if let Some(image2) = images_opt {
          let mut a2 = a2_arc.write();

          a2.commit_upload_font_atlas(hash, font_atlas.clone(), image2, prep2.descriptor_index);
        }

        Ok(prep2.descriptor_index)
      })
  }

  #[named]
  fn free_rasterized_font_atlas(&self, hash: u64, _font_atlas_id: u32) -> GpuResult<()> {
    crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |state, _| {
        let a2_arc = state
          .text2_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent text2"))?
          .clone();

        // 1. Prepare: Mutate maps instantly, returning raw handle data
        let prep2 = {
          let mut a2 = a2_arc.write();
          a2.prepare_remove_font_atlas(hash)?
        };

        let allocator_raw = state.allocator.allocator.as_allocator_view();
        let timeline = state.timeline_manager.get_cached_value();
        let discard_pool_ptr = &state.discard_pool as *const _;

        Ok((prep2, allocator_raw, timeline, discard_pool_ptr))
      })?
      .execute(
        |(prep2, allocator_raw, timeline, discard_pool_ptr), _rollback| {
          // 2. Execute: Push extracted handles straight into lock-free DiscardPool limits
          let discard_pool = unsafe { &*discard_pool_ptr };

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

  #[named]
  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
    task_id: Option<u64>,
  ) -> GpuResult<()> {
    // We fetch the wait value upfront without holding the `self.res` lock!
    let wait_value = match task_id {
      Some(id) => self.get_task_target_value(id)?,
      None => {
        let res_guard = self.res.read();
        let pe = wait_for_pe!(res_guard, handle)?;
        if let swapchain::PresentationState::Windowless(windowless) = &*pe {
          windowless.get_last_submitted_timeline_value()
        } else {
          return Err(gpu_invalid_arg!("presentation engine is not windowless"));
        }
      }
    };

    // SCOPE 1: Lock briefly to extract required state, then drop the lock!
    let (image, width, height, timeline_sem) = {
      let res_guard = DebugTrackedRwLock::read(&self.res);
      let pe = wait_for_pe!(res_guard, handle)?;

      if let swapchain::PresentationState::Windowless(windowless) = &*pe {
        let image = windowless.get_last_submitted_image()?;
        let (width, height) = windowless.extent();
        (
          image,
          width,
          height,
          res_guard.timeline_manager.semaphore.get(),
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
    self.get_command_buffer_and_native().map(|(cmd_id, _)| cmd_id)
  }

  #[named]
  fn set_command_buffer_presentation_engine(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let cmd_buffers = &self.recording_command_buffers;
    let mut data = cmd_buffers
      .get_mut(&(cmd_buffer, QueueRole::Graphics))
      .ok_or(gpu_err_invalid_cmd!())?;
    data.presentation_engine = Some(handle);
    Ok(())
  }

  /// Since it comes from the RenderDevice trait, doesn't work for now on Compute role.
  fn begin_command_buffer(&self, cmd_buffer: gpu::CommandBufferHandle) -> GpuResult<()> {
    self.begin_command_buffer_all(cmd_buffer, QueueRole::Graphics)
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
    use super::utils::RwLockable;
    let res_guard = self.res.read();

    // Check if we're inside a compositing render pass and need to adapt
    let actual_pipeline_key = {
      let cmd_buffers = &self.recording_command_buffers;
      let data = cmd_buffers
        .get(&(cmd_buffer, QueueRole::Graphics))
        .ok_or(gpu_err_invalid_cmd!())?;
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
      let cmd_buffers = &self.recording_command_buffers;
      let mut data =
        unsafe { cmd_buffers.get_mut(&(cmd_buffer, QueueRole::Graphics)).unwrap_unchecked() };
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

  #[named]
  fn add_billboard_texture(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    _texture_id: u64,
    texture: &Texture,
    _current_frame: u64,
  ) -> GpuResult<u32> {
    let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_write(handle, |state, h| {
        let live_pes = &state.live_presentation_engines;
        let pe = live_pes.get(&h).ok_or(gpu_err_cmd_no_pe!())?;

        let archetype_guard = pe.archetypes().registry.read();
        let archetype = archetype_guard
          .get(&ArchetypeId::Billboard)
          .ok_or(gpu_err_archetype_absent!())?;

        let arena_arc = state
          .billboard_render_archetype_arena
          .as_ref()
          .ok_or(crate::gpu_err_device!())?
          .clone();
        let descriptor_set = arena_arc.read().descriptor_set.get();

        // Reserve index in the global billboard array
        let mut billboard_resources = state.billboard_resources.write();
        let array_index = billboard_resources.len() as u32;

        // Insert a dummy image to securely reserve our bindless slot while we allocate lock-free
        billboard_resources.push(None);

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
        let mut billboard_resources = locks::DebugTrackedRwLock::write(&state.billboard_resources);
        billboard_resources[array_index as usize] = Some(image);

        Ok(array_index)
      })
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

    let layout = pe
      .archetypes()
      .registry
      .read()
      .get(&archetype)
      .map(|a| a.pipeline_layout())
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

          // 1. Check if the sun resource already exists.
          // NOTE: The previous `last_timeline == timeline` early-exit guard has been
          // intentionally removed.  That guard caused DispatchOnly to be skipped
          // whenever the cached GPU timeline hadn't advanced (common when update_sun
          // is called early in the frame before the previous submit is signalled),
          // making the sun volume permanently black after the first frame.
          // The user requires the compute dispatch to run every frame (once per frame,
          // not once per viewport).  DispatchOnly is safe to call repeatedly: the
          // layout transitions are idempotent and it records into the per-viewport
          // graphics command buffer before any render pass begins.
          if let Some(entry) = state.sun_resources.get(&entity_id) {
            if let resources::ResourceState::Ready(sun_res) = entry.value() {
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
            let mut sm = state.shader_manager.write();
            ensure_sungen_shader_module(&self.device, &mut sm)?
          };
          let shader_module = state
            .shader_manager
            .read()
            .get(comp_key)
            .ok_or(GpuError::InvalidShader)?
            .module
            .get();

          let archetype_guard = &pe.archetypes().registry.read();
          let archetype =
            archetype_guard.get(&ArchetypeId::Sun).ok_or(gpu_err_archetype_absent!())?;
          let arena_arc =
            state.sun_render_archetype_arena.as_ref().ok_or(crate::gpu_err_device!())?;
          let graphics_ds_layout =
            locks::DebugTrackedRwLock::read(&*arena_arc).descriptor_set_layout.get();

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
                unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None) }
                  .with_name(&self.device, "VkPipelineLayout_Sun")?;
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

              // 4. Params Data (Inline)
              let params_data = [
                timeline as f32 * 0.016,
                5778.0,
                1000000.0,
                radius,
                0.25, // scaleHeight: extended corona (was 0.05 — density hit zero at r≈0.75)
                2.0,  // noiseScale:  well-scaled granulation (was 15.0 — too fine / aliased)
              ];

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

              let alloc_info_vma = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                flags: vk_mem::AllocationCreateFlags::MAPPED
                  | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                ..Default::default()
              };
              let buffer_info = vk::BufferCreateInfo::default()
                .size(256) // Pad to 256 to avoid Lavapipe out-of-bounds false positives
                .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

              let (params_buffer, params_alloc) = unsafe {
                allocator.create_buffer(&buffer_info, &alloc_info_vma).map_err(|e| {
                  gpu_err!(format!(
                    "Failed to create params buffer for SunGen: {:?}",
                    e
                  ))
                })?
              };

              // Update params buffer
              let alloc_info_vma = allocator.get_allocation_info(&params_alloc);
              unsafe {
                let ptr = alloc_info_vma.mapped_data as *mut f32;
                *ptr = timeline as f32 * 0.016;
                *(ptr.add(1)) = 5778.0;
                *(ptr.add(2)) = 1000000.0;
                *(ptr.add(3)) = radius;
                *(ptr.add(4)) = 0.25; // scaleHeight: extended corona (was 0.05)
                *(ptr.add(5)) = 2.0; // noiseScale:  balanced granulation (was 15.0)
                allocator.flush_allocation(&params_alloc, 0, vk::WHOLE_SIZE as u64)?;
              }

              let bda_info = vk::BufferDeviceAddressInfo::default().buffer(params_buffer);
              let buffer_address =
                unsafe { self.device.buffer_device_address.get_buffer_device_address(&bda_info) };

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
              let timeline =
                DebugTrackedRwLock::read(&*self.res).get_timeline_semaphore_cached_value() + 1;
              discard_pool.discard_type_erased(transient_res, timeline + 2);

              let new_resource = resources::SunRenderResource {
                resolution,
                image: Some(image),
                descriptor_set: Some(unsafe {
                  NonZeroHandle::new_unchecked(graphics_descriptor_set)
                }),
                is_generated: false,
                params_buffer: Some(params_buffer),
                params_alloc: Some(params_alloc),
                compute_descriptor_pool: Some(descriptor_pool),
                compute_descriptor_set_layout: Some(set_layout),
                compute_descriptor_set: Some(descriptor_set),
                compute_pipeline: Some(compute_pipeline),
                compute_pipeline_layout: Some(pipeline_layout),
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
              let alloc_info_vma = allocator.get_allocation_info(&params_alloc);
              unsafe {
                let ptr = alloc_info_vma.mapped_data as *mut f32;
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
            SunOperation::None => Ok(ExecuteResult::None),
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
      let res = self.res.read();
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, handle)?;
      let billboard_render_archetype = pe.archetypes().registry.read();
      let billboard_render_archetype_ref = billboard_render_archetype
        .get(&ArchetypeId::Billboard)
        .ok_or(gpu_err_archetype_absent!())?;
      let (d, layout) = {
        let arena_arc =
          res.billboard_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
        let arena = arena_arc.read();
        (arena.descriptor_set.get(), arena.pipeline_layout.get())
      };

      (billboard_render_archetype_ref.pipeline_key(), layout, d)
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
  fn allocate_sphere_gizmo_instance(&self, entity: crate::scene::EntityId) -> GpuResult<u32> {
    let res = self.res.read();
    let arena_arc = res
      .sphere_gizmo_render_archetype_arena
      .as_ref()
      .ok_or(gpu_err!("arena absent"))?;
    arena_arc.write().allocate_sphere_gizmo_instance(entity)
  }

  #[named]
  fn free_sphere_gizmo_instance(&self, entity: crate::scene::EntityId) -> GpuResult<()> {
    let res = self.res.read();
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
      let res = self.res.read();
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, handle)?;
      let archetype_lock = pe.archetypes().registry.read();
      let archetype = archetype_lock
        .get(&ArchetypeId::SphereGizmo)
        .ok_or(gpu_err_archetype_absent!())?;
      (cmd, archetype.pipeline_key())
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
      let res = self.res.read();
      let (cmd, _) = self.get_cmd_and_pe(cmd_buffer)?;
      let arena_arc = res
        .sphere_gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = arena_arc.read();
      (cmd, arena.pipeline_layout.get())
    };
    let bytes = bytemuck::bytes_of(constants);
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
    self.get_pipeline_key_internal(handle, ArchetypeId::SphereGizmo)
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
      let res = self.res.read();
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, handle)?;
      let archetype_lock = pe.archetypes().registry.read();
      let archetype = archetype_lock
        .get(&ArchetypeId::SphereGizmo)
        .ok_or(gpu_err_archetype_absent!())?;

      let mut staging_arena = res.frame_staging_arena.write();
      let staging = staging_arena.as_mut().ok_or(gpu_err!("SphereGizmo missing staging arena"))?;

      let arena_arc = res
        .sphere_gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = arena_arc.read();

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
        arena.data_buffer.get(),
        staging.buffer,
        arena.data_ptr,
        archetype.pipeline_key(),
      )
    }; // <-- all locks dropped

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
  fn prepare_gizmo_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = self.res.read();
      let pe = wait_for_pe_direct!(&res_guard.live_presentation_engines, handle)?;
      let archetype_guard = pe.archetypes().registry.read();
      let archetype =
        archetype_guard.get(&ArchetypeId::Gizmo).ok_or(gpu_err_archetype_absent!())?;

      let arena_arc = res_guard
        .gizmo_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = arena_arc.read();
      (
        cmd,
        archetype.pipeline_key(),
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
    let (layout, pipeline_key, ds) = {
      let res = self.res.read();
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, handle)?;
      let pipeline_key = pe
        .archetypes()
        .registry
        .read()
        .get(&ArchetypeId::Sun)
        .ok_or(gpu_err_archetype_absent!())?
        .pipeline_key();
      let layout = res
        .sun_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?
        .read()
        .pipeline_layout
        .get();

      let resource_ref = res
        .sun_resources
        .get(&entity)
        .ok_or(gpu_err!("couldn't find sun descriptor set"))?;
      let ds = match resource_ref.value() {
        resources::ResourceState::Ready(s) => s
          .descriptor_set
          .map(|d| d.get())
          .ok_or(gpu_err!("couldn't find sun descriptor set"))?,
        _ => return Err(gpu_err!("sun resource not ready")),
      };
      (layout, pipeline_key, ds)
    };
    self.bind_pipeline(cmd_buffer, pipeline_key)?;
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
  fn prepare_trajectory_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, pipeline_layout, descriptor_set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res_guard = self.res.read();
      let pe = wait_for_pe_direct!(&res_guard.live_presentation_engines, handle)?;
      let archetype_guard = pe.archetypes().registry.read();
      let archetype = archetype_guard
        .get(&ArchetypeId::Trajectory)
        .ok_or(gpu_err_archetype_absent!())?;

      let arena_arc = res_guard
        .trajectory_render_archetype_arena
        .as_ref()
        .ok_or(gpu_err!("arena absent"))?;
      let arena = arena_arc.read();

      (
        cmd,
        archetype.pipeline_key(),
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
      let res_guard = self.res.read();
      let pe = wait_for_pe_direct!(&res_guard.live_presentation_engines, handle)?;
      let archetype_guard = pe.archetypes().registry.read();
      let archetype = archetype_guard.get(&ArchetypeId::Ui).ok_or(gpu_err_archetype_absent!())?;

      let arena_arc =
        res_guard.ui_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena = arena_arc.read();

      (
        cmd,
        archetype.pipeline_key(),
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
    let timeline = self.res.read().get_timeline_semaphore_cached_value() + 1;

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read(handle, |state, h| {
        // 1. Check sky image
        let sky_image_guard = state.sky_image.read();
        let sky_image = sky_image_guard.as_ref().ok_or(gpu_err!("sky image absent"))?;
        let sky_image_view = sky_image.image_view.get();

        // 2. Fetch Presentation Engine
        let live_pes = &state.live_presentation_engines;
        let pe = wait_for_pe_direct!(live_pes, h)?;

        // 3. Fetch Archetype Arena
        let arena_arc = state
          .sky_render_archetype_arena
          .as_ref()
          .ok_or(gpu_err!("arena absent"))?
          .clone();

        // 4. Extract needed layouts and check if allocation is necessary
        let (do_alloc, layout, pipeline_layout, existing_set) = {
          let arena = arena_arc.read();
          (
            arena.descriptor_set.is_none(),
            arena.descriptor_set_layout.get(),
            arena.pipeline_layout.get(),
            arena.descriptor_set,
          )
        };

        // 5. Extract shared dependencies
        let pool_guard = state.descriptor_pool.read();
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
            let discard_pool = unsafe { &*discard_pool_ptr };
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
          let mut arena = arena_arc.write();
          arena.descriptor_set = Some(descriptor_set);
        }

        Ok::<_, GpuError>(())
      })?;

    Ok(())
  }

  #[named]
  fn prepare_text2_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: CommandBufferHandle,
  ) -> GpuResult<()> {
    let (cmd, pipeline_key, layout, set) = {
      let (cmd, handle) = self.get_cmd_and_pe(cmd_buffer)?;
      let res = self.res.read();
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, handle)?;
      let archetype_lock = pe.archetypes().registry.read();
      let archetype = archetype_lock.get(&ArchetypeId::Text).ok_or(gpu_err_archetype_absent!())?;
      let arena_arc = res.text2_render_archetype_arena.as_ref().ok_or(gpu_err!("arena absent"))?;
      let arena_read = arena_arc.read();

      let layout = arena_read.pipeline_layout.get();
      let set = arena_read.descriptor_set.ok_or(crate::gpu_err_device!())?;

      (cmd, archetype.pipeline_key(), layout, set)
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

  fn debug_label_begin(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    name: &'static core::ffi::CStr,
    color: [f32; 4],
  ) {
    if let Ok(cmd) = self.get_cmd(cmd_buffer) {
      #[cfg(debug_assertions)]
      {
        let label = ash::vk::DebugUtilsLabelEXT::default()
          .label_name(name)
          .color(color);
        unsafe {
          self
            .device
            .debug_utils
            .cmd_begin_debug_utils_label(cmd, &label)
        };
      }
    }
  }

  fn debug_label_end(&self, cmd_buffer: crate::gpu::CommandBufferHandle) {
    #[cfg(debug_assertions)]
    {
      if let Ok(cmd) = self.get_cmd(cmd_buffer) {
        unsafe {
          self
            .device
            .debug_utils
            .cmd_end_debug_utils_label(cmd)
        };
      }
    }
  }

  fn debug_label_insert(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    name: &'static core::ffi::CStr,
    color: [f32; 4],
  ) {
    #[cfg(debug_assertions)]
    {
      if let Ok(cmd) = self.get_cmd(cmd_buffer) {
        let label = ash::vk::DebugUtilsLabelEXT::default()
          .label_name(name)
          .color(color);
        unsafe {
          self
            .device
            .debug_utils
            .cmd_insert_debug_utils_label(cmd, &label)
        };
      }
    }
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
      let cmd_buffers = &self.recording_command_buffers;
      if let Some(mut data) = cmd_buffers.get_mut(&(cmd_buffer, QueueRole::Graphics)) {
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
      let cmd_buffers = &self.recording_command_buffers;
      if let Some(mut data) = cmd_buffers.get_mut(&(cmd_buffer, QueueRole::Graphics)) {
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
      let cmd_buffers = &self.recording_command_buffers;
      let data = cmd_buffers
        .get(&(cmd_buffer, QueueRole::Graphics))
        .ok_or(gpu_err_invalid_cmd!())?;
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

        let mut pending_lock = locks::DebugTrackedRwLock::write(&state.pending_downloads);

        // Preemptive cleanup was removed because it bypassed standard DiscardPool and caused VMA memory corruption when Lavapipe used the buffers asynchronously.

        pending_lock.insert(
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
        let mut pending_lock = locks::DebugTrackedRwLock::write(&state.pending_downloads);

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

  fn submit_command_buffer(
    &self,
    cmd_buffer: CommandBufferHandle,
    task_id: Option<u64>,
    sync_infos: &[crate::gpu::CommandBufferSyncInfo],
  ) -> GpuResult<()> {
    self.submit_command_buffer_generic(
      cmd_buffer,
      task_id,
      sync_infos,
      &[],
      QueueRole::Graphics,
    )?;
    Ok(())
  }

  #[named]
  fn wire_callbacks(&self, pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>) -> GpuResult<()> {
    let workload = self
      .res
      .read()
      .timeline_manager
      .create_polling_workload(Arc::clone(&self.callback_stop_signal));
    pool.scatter(vec![Box::new(workload)]).map_err(|_| crate::gpu_err_device!())?;
    Ok(())
  }

  #[named]
  fn is_task_completed(&self, task_id: u64) -> GpuResult<bool> {
    self.res.read().timeline_manager.is_task_completed(task_id)
  }

  #[named]
  fn create_task(&self) -> u64 {
    self.res.read().timeline_manager.create_task()
  }

  #[named]
  fn fail_task(&self, task_id: u64, error: GpuError) {
    self.res.read().timeline_manager.fail_task(task_id, error)
  }

  #[named]
  fn success_task(&self, task_id: u64) {
    self.res.read().timeline_manager.success_task(task_id)
  }

  #[named]
  fn prepare_background_archetype_for_render_and_bind_pipeline(
    &self,
    cmd_buffer: gpu::CommandBufferHandle,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let pipeline_key = {
      let res_guard = self.res.read();
      let pe = wait_for_pe_direct!(&res_guard.live_presentation_engines, handle)?;
      let archetype_guard = pe.archetypes().registry.read();
      let archetype = archetype_guard
        .get(&ArchetypeId::Background)
        .ok_or(gpu_err_archetype_absent!())?;
      archetype.pipeline_key()
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
  fn cleanup(&mut self, device: &LogicalDevice) {
    aethervk_oshal_rlib::log!("Destroying TransientCmdPoolResource");
    unsafe {
      device.free_command_buffers(self.pool, &[self.cmd]);
      device.destroy_command_pool(self.pool, None);
    }
  }
}

/// Struct for `run_transient_commands` and `run_transient_compute_commands`
struct PoolGuard<'a> {
  device: &'a LogicalDevice,
  pool: vk::CommandPool,
  cmd: vk::CommandBuffer,
  disarmed: bool,
}
impl<'a> Drop for PoolGuard<'a> {
  fn drop(&mut self) {
    if !self.disarmed {
      unsafe {
        // self.device.free_command_buffers(self.pool, &[self.cmd]);
        self.device.destroy_command_pool(self.pool, None);
      }
    }
  }
}

impl Device {
  /// Returns `(vk_device_ptr, window_ptr)` suitable for RenderDoc's
  /// `StartFrameCapture` / `EndFrameCapture`, or `None` when the PE is
  /// windowless, unknown, or this is a release build.
  ///
  /// - `vk_device_ptr` — raw `VkDevice` handle cast to `*mut c_void`.
  /// - `window_ptr` — the native surface handle stored at swapchain creation:
  ///   XCB → `xcb_window_t` reinterpreted as pointer; Wayland → `wl_surface *`;
  ///   Win32 → `HWND`.
  #[cfg(debug_assertions)]
  pub fn get_windowed_pe_renderdoc_handles(
    &self,
    pe: crate::gpu::PresentationEngineHandle,
  ) -> Option<(*mut core::ffi::c_void, *mut core::ffi::c_void)> {
    use crate::gpu_backends::vulkan::utils::RwLockable;
    let state = self.res.read();
    let pe_ref = state.live_presentation_engines.get(&pe)?;
    if let swapchain::PresentationState::Windowed(w) = pe_ref.value() {
      // VkDevice is a dispatchable (pointer-sized) handle on all 64-bit targets.
      let dev_ptr = ash::vk::Handle::as_raw(self.device.handle()) as *mut core::ffi::c_void;
      // ptr1 always holds the window-side handle regardless of windowing system.
      Some((dev_ptr, w.native_handle.ptr1))
    } else {
      None
    }
  }

  pub fn get_pipeline_key(
    &self,
    handle: PresentationEngineHandle,
    archetype: ArchetypeId,
  ) -> GpuResult<PipelineKey> {
    self.get_pipeline_key_internal(handle, archetype)
  }

  /// Copies the back state of the particle system onto CPU backed memory buffers. To be called
  /// during a successful "Self Sync" operation, in which we are ensured that the compute queue is
  /// not running a step of the simulation withouth having to extract a compute timeline value to
  /// wait on
  pub fn snapshot_particles(
    &self,
  ) -> GpuResult<crate::simulation_api::structs::ParticleSystemSnapshot> {
    let res = self.res.read();
    let psm = res.particle_system_manager.as_ref().ok_or(gpu_err!("NO PSM"))?;

    let back = psm.back();
    let pt_size = Self::PAGE_TABLE_BYTES as vk::DeviceSize;
    let total_size =
      psm.buffer_size + psm.free_list_size + (pt_size * back.page_tables.len() as u64);

    if total_size == 0 {
      return Ok(crate::simulation_api::structs::ParticleSystemSnapshot::default());
    }

    // - Allocate mapping RAM memory using VMA
    let allocator = res.allocator.allocator.as_allocator_view();
    let buffer_info = vk::BufferCreateInfo::default()
      .size(total_size)
      .usage(vk::BufferUsageFlags::TRANSFER_DST);
    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    alloc_info.flags =
      vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;

    let (staging_buffer, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }?;
    let alloc_info_res = allocator.get_allocation_info(&alloc);
    let mapped_ptr = alloc_info_res.mapped_data;

    // - Issue GPU to RAM deep copy explicitly on the compute queue
    // (which exclusively owns the BACK buffers)
    let res_cmd = self.run_transient_compute_commands(|cmd| {
      unsafe {
        let mut offset = 0;
        let copy_global = vk::BufferCopy::default().size(psm.buffer_size).dst_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          back.buffer.buffer,
          staging_buffer,
          core::slice::from_ref(&copy_global),
        );
        offset += psm.buffer_size;

        let copy_free = vk::BufferCopy::default().size(psm.free_list_size).dst_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          back.free_list.buffer,
          staging_buffer,
          core::slice::from_ref(&copy_free),
        );
        offset += psm.free_list_size;

        for pt in back.page_tables.iter() {
          let copy_pt = vk::BufferCopy::default().size(pt_size).dst_offset(offset);
          self.device.cmd_copy_buffer(
            cmd,
            pt.value().buffer,
            staging_buffer,
            core::slice::from_ref(&copy_pt),
          );
          offset += pt_size;
        }
      }
      Ok(())
    });

    if let Err(e) = res_cmd {
      let mut a = alloc;
      unsafe { allocator.destroy_buffer(staging_buffer, &mut a) };
      return Err(e);
    }

    let transient_res = res_cmd.unwrap();
    unsafe { self.device.destroy_command_pool(transient_res.pool, None) };
    drop(transient_res);

    unsafe {
      // vkInvalidateMappedMemoryRanges is a core API function in the Vulkan Graphics API.
      // It forces the CPU cache to refresh so that the host application can explicitly see the most up-to-date data written by the GPU
      allocator.invalidate_allocation(&alloc, 0, vk::WHOLE_SIZE)?;
      let mut snap = crate::simulation_api::structs::ParticleSystemSnapshot::default();

      let mut offset = 0;
      snap.global_buffer.extend_from_slice(core::slice::from_raw_parts(
        mapped_ptr.add(offset as usize).cast(),
        psm.buffer_size as usize,
      ));
      offset += psm.buffer_size;

      snap.free_list.extend_from_slice(core::slice::from_raw_parts(
        mapped_ptr.add(offset as usize).cast(),
        psm.free_list_size as usize,
      ));
      offset += psm.free_list_size;

      for pt in back.page_tables.iter() {
        let mut pt_vec = alloc::vec![0u8; pt_size as usize];
        pt_vec.copy_from_slice(core::slice::from_raw_parts(
          mapped_ptr.add(offset as usize).cast(),
          pt_size as usize,
        ));
        snap.page_tables.insert(*pt.key(), pt_vec);
        offset += pt_size;
      }

      let mut a = alloc;
      allocator.destroy_buffer(staging_buffer, &mut a);
      Ok(snap)
    }
  }

  pub fn restore_particles(
    &self,
    snap: &crate::simulation_api::structs::ParticleSystemSnapshot,
  ) -> GpuResult<()> {
    if snap.global_buffer.is_empty() {
      return Ok(());
    }

    // Safety Pass: Reconcile Page Tables
    // Ensure te system didn't dynamically allocate/drop particle systems since the snapshot
    let active_keys: alloc::vec::Vec<u64> = self
      .res
      .read()
      .particle_system_manager
      .as_ref()
      .unwrap()
      .back()
      .page_tables
      .iter()
      .map(|kv| *kv.key())
      .collect();
    for key in active_keys {
      if !snap.page_tables.contains_key(&key) {
        let tl = self.res.read().timeline_manager.get_cached_value() + 1;
        let ctl = self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
        let _ = self.discard_particle_system(key, tl, ctl);
      }
    }
    for key in snap.page_tables.keys() {
      if !self
        .res
        .read()
        .particle_system_manager
        .as_ref()
        .unwrap()
        .back()
        .page_tables
        .contains_key(key)
      {
        let _ = self.create_particle_system(*key);
      }
    }

    let res = self.res.read();
    let psm = res.particle_system_manager.as_ref().unwrap();
    let pt_size = Self::PAGE_TABLE_BYTES;
    let total_size =
      psm.buffer_size + psm.free_list_size + (pt_size * snap.page_tables.len() as u64);

    let allocator = res.allocator.allocator.as_allocator_view();
    let buffer_info = vk::BufferCreateInfo::default()
      .size(total_size)
      .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED;

    let (staging_buffer, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }?;
    let alloc_info_res = allocator.get_allocation_info(&alloc);
    let mapped_ptr = alloc_info_res.mapped_data as *mut u8;

    unsafe {
      core::ptr::copy_nonoverlapping(
        snap.global_buffer.as_ptr(),
        mapped_ptr,
        psm.buffer_size as usize,
      );
      core::ptr::copy_nonoverlapping(
        snap.free_list.as_ptr(),
        mapped_ptr.add(psm.buffer_size as usize),
        psm.free_list_size as usize,
      );
      let mut offset = (psm.buffer_size + psm.free_list_size) as usize;
      for kv in psm.back().page_tables.iter() {
        if let Some(pt_vec) = snap.page_tables.get(kv.key()) {
          core::ptr::copy_nonoverlapping(pt_vec.as_ptr(), mapped_ptr.add(offset), pt_size as usize);
        }
        offset += pt_size as usize;
      }
      allocator.flush_allocation(&alloc, 0, vk::WHOLE_SIZE)?;
    }

    // 1. Copy RAM to Back Buffer (enforces compute queue idleness and ownership requirement)
    let res_cmd_back = self.run_transient_compute_commands(|cmd| {
      unsafe {
        let mut offset = 0;
        let copy_global = vk::BufferCopy::default().size(psm.buffer_size).src_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          staging_buffer,
          psm.back().buffer.buffer,
          core::slice::from_ref(&copy_global),
        );
        offset += psm.buffer_size;

        let copy_free = vk::BufferCopy::default().size(psm.free_list_size).src_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          staging_buffer,
          psm.back().free_list.buffer,
          core::slice::from_ref(&copy_free),
        );
        offset += psm.free_list_size;

        for pt in psm.back().page_tables.iter() {
          let copy_pt = vk::BufferCopy::default().size(pt_size).src_offset(offset);
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            pt.value().buffer,
            core::slice::from_ref(&copy_pt),
          );
          offset += pt_size;
        }
      }
      Ok(())
    });

    if let Err(e) = res_cmd_back {
      let mut a = alloc;
      unsafe { allocator.destroy_buffer(staging_buffer, &mut a) };
      return Err(e);
    }
    unsafe { self.device.destroy_command_pool(res_cmd_back.unwrap().pool, None) };

    // 2. Copy RAM to Front buffer (assumes Graphics queue has ownership, therefore there is no
    //    cross sync in bound)
    let res_cmd_front = self.run_transient_commands(|cmd| {
      unsafe {
        let mut offset = 0;
        let copy_global = vk::BufferCopy::default().size(psm.buffer_size).src_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          staging_buffer,
          psm.front().buffer.buffer,
          core::slice::from_ref(&copy_global),
        );
        offset += psm.buffer_size;

        let copy_free = vk::BufferCopy::default().size(psm.free_list_size).src_offset(offset);
        self.device.cmd_copy_buffer(
          cmd,
          staging_buffer,
          psm.front().free_list.buffer,
          core::slice::from_ref(&copy_free),
        );
        offset += psm.free_list_size;

        for pt in psm.front().page_tables.iter() {
          let copy_pt = vk::BufferCopy::default().size(pt_size).src_offset(offset);
          self.device.cmd_copy_buffer(
            cmd,
            staging_buffer,
            pt.value().buffer,
            core::slice::from_ref(&copy_pt),
          );
          offset += pt_size;
        }
      }
      Ok(())
    });

    let mut a = alloc;
    unsafe { allocator.destroy_buffer(staging_buffer, &mut a) };
    // return if error on graphics "?"
    unsafe { self.device.destroy_command_pool(res_cmd_front?.pool, None) };

    Ok(())
  }

  #[named]
  pub(super) fn run_transient_compute_commands<F>(
    &self,
    f: F,
  ) -> GpuResult<TransientCmdPoolResource>
  where
    F: FnOnce(vk::CommandBuffer) -> GpuResult<()>,
  {
    let queue = self.queues.get_compute_queue();
    let pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(queue.family_index)
      .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let pool = unsafe { self.device.create_command_pool(&pool_info, None) }?;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
      .command_pool(pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);
    let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0];

    let mut guard = PoolGuard {
      device: &self.device,
      pool,
      cmd,
      disarmed: false,
    };
    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { self.device.begin_command_buffer(cmd, &begin_info) }?;
    f(cmd)?;
    unsafe { self.device.end_command_buffer(cmd) }?;

    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
      .semaphore_type(vk::SemaphoreType::TIMELINE)
      .initial_value(0);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
    let timeline_semaphore = unsafe { self.device.create_semaphore(&semaphore_info, None) }?;

    let sem_submit_info = vk::SemaphoreSubmitInfo::default()
      .semaphore(timeline_semaphore)
      .value(1)
      .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE);
    let cmd_submit_info = vk::CommandBufferSubmitInfo::default().command_buffer(cmd);
    let submit_info = vk::SubmitInfo2::default()
      .signal_semaphore_infos(core::slice::from_ref(&sem_submit_info))
      .command_buffer_infos(core::slice::from_ref(&cmd_submit_info));

    // locked submit with synchronization2
    unsafe {
      let _lock = self.device.submission_lock_compute.lock();
      self.device.synchronization2.queue_submit2(
        queue.handle,
        core::slice::from_ref(&submit_info),
        vk::Fence::null(),
      )?;

      let wait_info = vk::SemaphoreWaitInfo::default()
        .semaphores(core::slice::from_ref(&timeline_semaphore))
        .values(&[1]);
      self.device.timeline_semaphore.wait_semaphores(&wait_info, u64::MAX)?;

      self.device.destroy_semaphore(timeline_semaphore, None);
    }

    guard.disarmed = false;
    Ok(TransientCmdPoolResource { pool, cmd })
  }

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
    let mut guard = PoolGuard {
      device: &self.device,
      pool,
      cmd,
      disarmed: false,
    };

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { self.device.begin_command_buffer(cmd, &begin_info) }?;

    if let Err(e) = f(cmd) {
      aethervk_oshal_rlib::log!("run_transient_commands error: {:?}", e);
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
    let mut timeline_info =
      vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

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

    guard.disarmed = true;
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
      use super::utils::RwLockable;
      let res_guard = self.res.read();
      let _presentation_engines_guard = &res_guard.live_presentation_engines;
      let cmd_buffers = &self.recording_command_buffers;
      if !cmd_buffers.contains_key(&(cmd_buffer, QueueRole::Graphics)) {
        return Err(gpu_err_invalid_cmd!());
      }
      let wpresentation_engine = wait_for_pe!(res_guard, presentation_engine)?;

      let data = unsafe { cmd_buffers.get(&(cmd_buffer, QueueRole::Graphics)).unwrap_unchecked() };
      if !data.has_begun {
        return Err(gpu_err!("command buffer not begun"));
      }

      if acquire_result.status.needs_resize() {
        return Err(GpuError::ResizeRequired);
      }
      if wpresentation_engine.swapchain_generation() != acquire_result.swapchain_generation {
        return Err(GpuError::ResizeRequired);
      }
      drop(data);

      let (wait_semaphore, submission_fence) =
        unsafe { wpresentation_engine.get_frame_resources(acquire_result.frame_index as usize) };
      let (_, _, signal_semaphore) =
        unsafe { wpresentation_engine.get_image_resources(acquire_result.image_index as usize) };

      let timeline = res_guard.timeline_manager.get_next_submit_value() - 1;

      let cmd = {
        let cmd_buffers = &self.recording_command_buffers;
        let mut data =
          unsafe { cmd_buffers.get_mut(&(cmd_buffer, QueueRole::Graphics)).unwrap_unchecked() };
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
      let cmd_buffers = &self.recording_command_buffers;
      if let Some(mut data) = cmd_buffers.get_mut(&(cmd_buffer, QueueRole::Graphics)) {
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
  pub fn get_cmd(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
  ) -> GpuResult<ash::vk::CommandBuffer> {
    let cmd_buffers = &self.recording_command_buffers;
    let data = cmd_buffers
      .get(&(cmd_buffer, QueueRole::Graphics))
      .ok_or(gpu_err_invalid_cmd!())?;
    Ok(data.command_buffer.get())
  }

  #[named]
  fn get_cmd_and_pe(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
  ) -> GpuResult<(ash::vk::CommandBuffer, crate::gpu::PresentationEngineHandle)> {
    let cmd_buffers = &self.recording_command_buffers;
    let data = cmd_buffers
      .get(&(cmd_buffer, QueueRole::Graphics))
      .ok_or(gpu_err_invalid_cmd!())?;
    let handle = data.presentation_engine.ok_or(gpu_err_cmd_no_pe!())?;
    Ok((data.command_buffer.get(), handle))
  }

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
      let cmd_buffers = &self.recording_command_buffers;
      let data = cmd_buffers
        .get(&(cmd_buffer, QueueRole::Graphics))
        .ok_or(gpu_err_invalid_cmd!())?;
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

        locks::DebugTrackedRwLock::write(&state.pending_downloads).insert(
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
      let cmd_buffers = &self.recording_command_buffers;
      let data = cmd_buffers
        .get(&(cmd_buffer, QueueRole::Graphics))
        .ok_or(gpu_err_invalid_cmd!())?;
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

        locks::DebugTrackedRwLock::write(&state.pending_downloads).insert(
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

// TODO refactor in utils

fn ensure_dust_shader_modules(
  device: &LogicalDevice,
  shader_manager: &mut shader_manager::ShaderManager,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  let assets_dir = shaders_asset_dir()?;
  vert_path = assets_dir.join("dust.vert.spv");
  frag_path = assets_dir.join("dust.frag.spv");

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

/// module for the new, GPU-only, particle system management
pub mod particles {
  use aethervk_oshal_rlib::os::time::timeus_t;

  use super::*;

  #[derive(Clone, Copy, Debug)]
  pub struct BufferAlloc {
    pub buffer: vk::Buffer,
    pub alloc: vk_mem::Allocation,
    pub address: u64,
  }

  impl BufferAlloc {
    pub fn new(buffer: vk::Buffer, alloc: vk_mem::Allocation, address: u64) -> Self {
      Self {
        buffer,
        alloc,
        address,
      }
    }
  }

  pub struct Settings {
    pub emission_us: core::sync::atomic::AtomicI64,
    pub compaction_us: core::sync::atomic::AtomicI64,
  }

  // TODO move rustdoc about cross sync elsewhere
  /// When Buffer is created with `VK_SHARING_MODE_EXCLUSIVE`, moving it across 2 queue families
  /// requires a "Queue Family Ownership Transfer", which is composed of a "Two-Way Handshake"
  /// procedure.
  /// - Record a "Release Barrier" on a command buffer submitted to the source queue family
  /// - Record a "Acquire Barrier" on a command buffer submitted to the destination queue family
  /// We therefore need 2 command buffers, from compute and graphics, if different
  /// We assume flow is:
  /// <pre>
  ///  compute -> |               | Acquire Front -> Copy back to front -> Release Back
  ///             | timeline sync |
  ///  render  -> |               | Release Front ->                    -> Acquire Back
  /// </pre>
  pub(super) struct ParticleSystemState {
    /// global buffer holding data for all particles
    pub(super) buffer: BufferAlloc,
    pub(super) free_list: BufferAlloc,
    /// Store ECS entity_id - particle system page table association
    pub(super) page_tables: dashmap::DashMap<u64, BufferAlloc>,
  }

  pub struct ParticleSystemManager {
    /// Double buffered state: one for render (front), one for compute (back)
    states: [ParticleSystemState; 2],
    front_index: usize,

    /// Size cached to easily issue buffer synchronization copies
    pub(super) buffer_size: vk::DeviceSize,
    pub(super) free_list_size: vk::DeviceSize,

    /// necessary hnadle duplicated for drop trait
    allocator_view: vk_mem::AllocatorView,
    settings: Settings,
  }

  impl ParticleSystemManager {
    pub fn emission_us(&self) -> timeus_t {
      self.settings.emission_us.load(core::sync::atomic::Ordering::Relaxed)
    }

    pub fn compaction_us(&self) -> timeus_t {
      self.settings.compaction_us.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Get the "front" buffers meant for reading in the render thread/graphics queue
    #[inline]
    pub(super) fn front(&self) -> &ParticleSystemState {
      &self.states[self.front_index]
    }

    /// Get the "back" buffers meant for reading/writing in the compute thread/queue
    #[inline]
    pub(super) fn back(&self) -> &ParticleSystemState {
      &self.states[1 - self.front_index]
    }

    /// Get mutable access to the "front" buffers
    #[inline]
    fn front_mut(&mut self) -> &mut ParticleSystemState {
      &mut self.states[self.front_index]
    }

    /// Get mutable access to the "back" buffers
    #[inline]
    fn back_mut(&mut self) -> &mut ParticleSystemState {
      &mut self.states[1 - self.front_index]
    }

    /// `Cross Sync` Step 1: Recorded on GRAPHICS queue during the `sync window`
    /// Releases the current `front` buffer so Compute queue family can take ownership
    pub fn cmd_sync_graphics_release_front(
      &self,
      device: &LogicalDevice,
      cmd: vk::CommandBuffer,
      graphics_family: u32,
      compute_family: u32,
    ) {
      // If same queue family, ownership transfers are illegal.
      // Sync is handled entirely by standard execution barriers in Step 2
      if graphics_family == compute_family {
        return;
      }

      let front = self.front();
      let barriers = self.create_state_barriers(
        front,
        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
          | vk::PipelineStageFlags2::VERTEX_SHADER
          | vk::PipelineStageFlags2::FRAGMENT_SHADER
          | vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
          | vk::AccessFlags2::SHADER_READ
          | vk::AccessFlags2::INDIRECT_COMMAND_READ,
        vk::PipelineStageFlags2::NONE, // Vulkan Spec: Release Dst must be NONE
        vk::AccessFlags2::NONE,
        graphics_family,
        compute_family,
      );

      let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
      unsafe {
        device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info);
      }
    }

    /// `Cross Sync` Step 2: Recorded on the COMPUTE queue during the Sync Window
    /// Acquires `Front`, performs `Back -> Front` copy, prepares buffers for their post swap role
    pub fn cmd_sync_compute_copy_and_release(
      &self,
      device: &LogicalDevice,
      cmd: vk::CommandBuffer,
      page_table_size: vk::DeviceSize,
      graphics_family: u32,
      compute_family: u32,
    ) {
      let is_cross_family = graphics_family != compute_family;
      let front = self.front();
      let back = self.back();
      let mut pre_barriers = alloc::vec::Vec::with_capacity(32);

      // A. Pre-Copy Barrier for `front` (Destination)
      if is_cross_family {
        self.create_state_barriers_inline(
          &mut pre_barriers,
          front,
          vk::PipelineStageFlags2::NONE,
          vk::AccessFlags2::NONE, // Vulkan Spec: Acquire Src must be NONE
          vk::PipelineStageFlags2::TRANSFER, // transfer ownership before copy op
          vk::AccessFlags2::TRANSFER_WRITE,
          graphics_family,
          compute_family,
        );
      } else {
        // Wait for graphics to finish reading before transfer can write
        self.create_state_barriers_inline(
          &mut pre_barriers,
          front,
          vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT // src
            | vk::PipelineStageFlags2::VERTEX_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
          vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::SHADER_READ,
          vk::PipelineStageFlags2::TRANSFER, // dst
          vk::AccessFlags2::TRANSFER_WRITE,
          vk::QUEUE_FAMILY_IGNORED,
          vk::QUEUE_FAMILY_IGNORED,
        );
      }

      // B. Pre-Copy Barrier for `back` (Source) - Wait for compute shader to finish writing
      // Note: This means that logic layer shouldn't request manual emission
      self.create_state_barriers_inline(
        &mut pre_barriers,
        back,
        vk::PipelineStageFlags2::COMPUTE_SHADER, // src
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::TRANSFER, // dst
        vk::AccessFlags2::TRANSFER_READ,
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_IGNORED,
      );

      let pre_dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&pre_barriers);
      unsafe { device.synchronization2.cmd_pipeline_barrier2(cmd, &pre_dep_info) };

      // C. Execute Copies Back -> Front
      unsafe {
        let b_copy = vk::BufferCopy::default().src_offset(0).dst_offset(0).size(self.buffer_size);
        device.cmd_copy_buffer(
          cmd,
          back.buffer.buffer,
          front.buffer.buffer,
          core::slice::from_ref(&b_copy),
        );

        let fl_copy =
          vk::BufferCopy::default().src_offset(0).dst_offset(0).size(self.free_list_size);
        device.cmd_copy_buffer(
          cmd,
          back.free_list.buffer,
          front.free_list.buffer,
          core::slice::from_ref(&fl_copy),
        );

        for back_entry in back.page_tables.iter() {
          if let Some(front_pt) = front.page_tables.get(back_entry.key()) {
            let pt_copy =
              vk::BufferCopy::default().src_offset(0).dst_offset(0).size(page_table_size);
            device.cmd_copy_buffer(
              cmd,
              back_entry.value().buffer,
              front_pt.buffer,
              core::slice::from_ref(&pt_copy),
            );
          }
        }
      }

      // D. Post-Copy Barriers (Prep for CPY Swap)
      pre_barriers.clear();
      let mut post_barriers = pre_barriers;

      if is_cross_family {
        // `back` will become the new `front`. Release it to graphics
        self.create_state_barriers_inline(
          &mut post_barriers,
          back,
          vk::PipelineStageFlags2::TRANSFER,
          vk::AccessFlags2::TRANSFER_READ,
          vk::PipelineStageFlags2::NONE, // Vulkan Spec: Release Dst must be NONE
          vk::AccessFlags2::NONE,
          compute_family,
          graphics_family,
        );
      }

      // `front` will become the new `back`. Sync for next compute
      // Note: this means that we don't need to externally synchronize
      self.create_state_barriers_inline(
        &mut post_barriers,
        front,
        vk::PipelineStageFlags2::TRANSFER, // src transfer must be finished...
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER, // before dst shader read write
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        vk::QUEUE_FAMILY_IGNORED,
        vk::QUEUE_FAMILY_IGNORED,
      );

      let post_dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&post_barriers);
      unsafe { device.synchronization2.cmd_pipeline_barrier2(cmd, &post_dep_info) };
    }

    /// `Cross Sync` Step 3: Recorded on GRAPHICS Queue AFTER `swap_buffers()` has been called.
    /// Acquires the new `Front` buffer for rendering
    pub fn cmd_sync_graphics_acquire_new_front(
      &self,
      device: &LogicalDevice,
      cmd: vk::CommandBuffer,
      graphics_family: u32,
      compute_family: u32,
    ) {
      // Because swap_buffers() was called, self.front() is now the newly copied buffer
      let new_front = self.front();
      let is_cross_family = graphics_family != compute_family;

      let barriers = if is_cross_family {
        self.create_state_barriers(
          new_front,
          vk::PipelineStageFlags2::NONE, // Vulkan Spec: Acquire src is NONE
          vk::AccessFlags2::NONE,
          vk::PipelineStageFlags2::DRAW_INDIRECT,
          vk::AccessFlags2::INDIRECT_COMMAND_READ,
          compute_family,
          graphics_family,
        )
      } else {
        // Wait for Transfer Read to finish before we start reading in Vertex
        self.create_state_barriers(
          new_front,
          vk::PipelineStageFlags2::TRANSFER,
          vk::AccessFlags2::TRANSFER_READ, // we have copied to new_back
          vk::PipelineStageFlags2::DRAW_INDIRECT,
          vk::AccessFlags2::INDIRECT_COMMAND_READ,
          vk::QUEUE_FAMILY_IGNORED,
          vk::QUEUE_FAMILY_IGNORED,
        )
      };

      let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&barriers);
      unsafe { device.synchronization2.cmd_pipeline_barrier2(cmd, &dep_info) };
    }

    /// Swaps the roles of the front and back buffers.
    /// # Safety
    /// GPU side needs to be externally synchronized
    #[inline]
    pub unsafe fn swap_buffers(&mut self) {
      self.front_index = 1 - self.front_index;
    }

    pub fn new(
      device: &LogicalDevice,
      allocator: vk_mem::AllocatorView,
      transfer_queue: Queue,
      max_particles: usize,
    ) -> GpuResult<Self> {
      use crate::gpu::new_particles::*;

      const PAGE_TABLES_MAP_START_CAP: usize = 64;
      // Note: measured in unscaled time
      const DEFAULT_EMISSION_US: i64 = oshal::os::time::timeus_milliseconds(166);
      // Note: measured in unscaled time
      const DEFAULT_COMPACTION_US: i64 = oshal::os::time::timeus_milliseconds(5000);

      let num_chunks = max_particles.div_ceil(PCHUNK_SIZE);
      let buffer_size = (core::mem::size_of::<ParticleChunk>() * num_chunks) as vk::DeviceSize;
      // TODO: check consistency with allocate_free_list
      let free_list_size = ((num_chunks * 4) + 4) as vk::DeviceSize;

      // isolate creation closure since we need to create an identical pair
      let mut create_state = |name_suffix: &str| -> GpuResult<ParticleSystemState> {
        let buffer_info = vk::BufferCreateInfo::default().size(buffer_size as _).usage(
          vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC  // back->front syncing
            | vk::BufferUsageFlags::TRANSFER_DST, // back->front syncing
        );
        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        // it's supposed to be large, so dedicated always
        // `crate::apply_test_dedicated_alloc!(alloc_info);` not needed
        alloc_info.flags = vk_mem::AllocationCreateFlags::DEDICATED_MEMORY;
        alloc_info.priority = 1.0f32;

        let (buffer, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
          .with_name(
            device,
            &alloc::format!("GlobalParticleBuffer_{name_suffix}"),
          )?;
        let address = unsafe {
          let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
          device.buffer_device_address.get_buffer_device_address(&info)
        };

        let free_list = Self::allocate_free_list(device, allocator, transfer_queue, num_chunks)?;

        Ok(ParticleSystemState {
          free_list,
          buffer: BufferAlloc {
            buffer,
            alloc,
            address,
          },
          page_tables: dashmap::DashMap::with_capacity(PAGE_TABLES_MAP_START_CAP),
        })
      };

      let state_front = create_state("Front")?;
      let state_back = create_state("Back")?;

      Ok(Self {
        states: [state_front, state_back],
        front_index: 0,
        buffer_size,
        free_list_size,
        allocator_view: allocator,
        settings: Settings {
          emission_us: core::sync::atomic::AtomicI64::new(DEFAULT_EMISSION_US),
          compaction_us: core::sync::atomic::AtomicI64::new(DEFAULT_COMPACTION_US),
        },
      })
    }

    fn allocate_free_list(
      device: &LogicalDevice,
      allocator: vk_mem::AllocatorView,
      transfer_queue: Queue,
      num_chunks: usize,
    ) -> GpuResult<BufferAlloc> {
      // - allocate a gpu-only buffer
      let buffer_size = (num_chunks * 4) + 4;
      let gpu_b = {
        let buffer_info = vk::BufferCreateInfo::default().size(buffer_size as _).usage(
          vk::BufferUsageFlags::TRANSFER_DST
          | vk::BufferUsageFlags::TRANSFER_SRC // support copying free list
          | vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        crate::apply_test_dedicated_alloc!(alloc_info);
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        alloc_info.priority = 1.0f32;

        let (b, a) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }?;
        AllocJanitor {
          buffer: b,
          alloc: a,
          allocator,
        }
      };

      // - allocate a staging buffer, memory mapped
      let (staging_b, staging_alloc_info) = {
        let buffer_info = vk::BufferCreateInfo::default()
          .size(buffer_size as _)
          .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        crate::apply_test_dedicated_alloc!(alloc_info);
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
          | vk_mem::AllocationCreateFlags::MAPPED;
        let (buffer, alloc, info) =
          unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }?;
        (
          AllocJanitor {
            buffer,
            alloc,
            allocator,
          },
          info,
        )
      };

      // - write free list content: count = num_chunks as u32,
      // while indices write from `num_chunks - 1` to 0. Place barrier host to copy
      let p_mem = staging_alloc_info.mapped_data.cast::<u32>();
      unsafe {
        p_mem.write(num_chunks as u32);
        for i in 0..num_chunks as u32 {
          // advance + 1 for the size block
          p_mem.add(i as usize + 1).write(num_chunks as u32 - 1 - i);
        }
      }
      allocator.flush_allocation(&staging_b.alloc, 0, vk::WHOLE_SIZE);

      // - copy buffer operation
      let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(transfer_queue.family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
      let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
        .with_name(device, "CommandPool_ParticleSystemTransient")?;
      let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
        .with_name(device, "Fence_ParticleSystemAllocationTransient")?;
      let mut _cleanup = TransientCleanup::command_only(device, command_pool, fence);
      let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
      let command_buffer = unsafe {
        let mut c = vk::CommandBuffer::null();
        (device.fp_v1_0().allocate_command_buffers)(
          device.handle(),
          core::ptr::from_ref(&command_buffer_info),
          core::ptr::from_mut(&mut c),
        )
        .result_with_success(c)
      }
      .with_name(device, "CommandBuffer_Transient_PSM_FreeList")?;

      unsafe {
        // record phase
        let begin_info =
          vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;
        let copy_region =
          vk::BufferCopy::default().src_offset(0).dst_offset(0).size(buffer_size as _);
        device.cmd_copy_buffer(
          command_buffer,
          staging_b.buffer,
          gpu_b.buffer,
          core::slice::from_ref(&copy_region),
        );
        device.end_command_buffer(command_buffer)?;
        // submit and nuke fence
        let submit_info =
          vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
        device.locked_queue_submit(
          transfer_queue.handle,
          core::slice::from_ref(&submit_info),
          fence,
        )?;
        device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX)?;
      }
      core::mem::drop(staging_b);

      let free_list_bda = unsafe {
        let info = vk::BufferDeviceAddressInfo::default().buffer(gpu_b.buffer);
        device.buffer_device_address.get_buffer_device_address(&info)
      };
      let (gpu_buffer, gpu_alloc) = (gpu_b.buffer, gpu_b.alloc);
      core::mem::forget(gpu_b);

      Ok(BufferAlloc::new(gpu_buffer, gpu_alloc, free_list_bda))
    }

    /// Helper to generate memory barriers for all buffers inside a [`ParticleSystemManager`]
    fn create_state_barriers(
      &self,
      state: &ParticleSystemState,
      src_stage: vk::PipelineStageFlags2,
      src_access: vk::AccessFlags2,
      dst_stage: vk::PipelineStageFlags2,
      dst_access: vk::AccessFlags2,
      src_family: u32,
      dst_family: u32,
    ) -> alloc::vec::Vec<vk::BufferMemoryBarrier2> {
      let mut barriers = alloc::vec::Vec::with_capacity(2 + state.page_tables.len());
      let mut push = |buffer: vk::Buffer, size: vk::DeviceSize| {
        barriers.push(
          vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .buffer(buffer)
            .offset(0)
            .size(size),
        );
      };

      push(state.buffer.buffer, self.buffer_size);
      push(state.free_list.buffer, self.free_list_size);
      for pt_entry in state.page_tables.iter() {
        push(pt_entry.value().buffer, vk::WHOLE_SIZE)
      }

      barriers
    }

    /// Helper to generate memory barriers for all buffers inside a [`ParticleSystemManager`]
    fn create_state_barriers_inline(
      &self,
      the_vec: &mut alloc::vec::Vec<vk::BufferMemoryBarrier2>,
      state: &ParticleSystemState,
      src_stage: vk::PipelineStageFlags2,
      src_access: vk::AccessFlags2,
      dst_stage: vk::PipelineStageFlags2,
      dst_access: vk::AccessFlags2,
      src_family: u32,
      dst_family: u32,
    ) {
      let mut push = |buffer: vk::Buffer, size: vk::DeviceSize| {
        the_vec.push(
          vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .buffer(buffer)
            .offset(0)
            .size(size),
        );
      };

      push(state.buffer.buffer, self.buffer_size);
      push(state.free_list.buffer, self.free_list_size);
      for pt_entry in state.page_tables.iter() {
        push(pt_entry.value().buffer, vk::WHOLE_SIZE)
      }
    }

    /// Inserts the allocated page tables into both the front and back states
    pub fn add_page_tables(&self, id: u64, pt_0: BufferAlloc, pt_1: BufferAlloc) {
      self.states[0].page_tables.insert(id, pt_0);
      self.states[1].page_tables.insert(id, pt_1);
    }

    /// Removes the page tables from both states
    /// returns the index of the front (owned by GRAPHICS queue)
    pub fn remove_pages_tables(&self, id: u64) -> Option<([BufferAlloc; 2], usize)> {
      let (_, pt_0) = self.states[0].page_tables.remove(&id)?;
      let (_, pt_1) = self.states[1].page_tables.remove(&id).unwrap();
      Some(([pt_0, pt_1], self.front_index))
    }

    /// Retrieves the BDA pointers (Global, PageTable, FreeList) for a specific queue role
    pub fn get_addresses(&self, id: u64, role: QueueRole) -> Option<(u64, u64, u64)> {
      let state_index = match role {
        QueueRole::Graphics => self.front_index,
        QueueRole::Compute => 1 - self.front_index,
      };
      let state = &self.states[state_index];
      let pt = state.page_tables.get(&id)?;
      Some((state.buffer.address, pt.address, state.free_list.address))
    }

    /// Retrieves the BDA pointers (Global, PageTable) and page table buffer for a specific queue role
    pub(super) fn get_addresses_and_buffer(
      &self,
      id: u64,
      role: QueueRole,
    ) -> Option<(u64, u64, vk::Buffer)> {
      let state_index = match role {
        QueueRole::Graphics => self.front_index,
        QueueRole::Compute => 1 - self.front_index,
      };
      let state = &self.states[state_index];
      let pt = state.page_tables.get(&id)?;
      Some((state.buffer.address, pt.address, pt.buffer))
    }
  }

  impl Drop for ParticleSystemManager {
    fn drop(&mut self) {
      unsafe {
        // free all double-buffered state maps safely
        for state in self.states.iter_mut() {
          self.allocator_view.destroy_buffer(state.buffer.buffer, &mut state.buffer.alloc);
          self
            .allocator_view
            .destroy_buffer(state.free_list.buffer, &mut state.free_list.alloc);

          for mut pt_entry in state.page_tables.iter_mut() {
            let b_alloc = pt_entry.value_mut();
            self.allocator_view.destroy_buffer(b_alloc.buffer, &mut b_alloc.alloc);
          }
          state.page_tables.clear();
        }
      }
    }
  }

  pub enum PushConstantMutUnion<'a> {
    ApplyEmittersDirectNew(
      &'a mut gpu::compute_push_constants::ApplyEmittersDirectNewPushConstants,
    ),
    IntegrateParticlesP1P2New(
      &'a mut gpu::compute_push_constants::IntegrateParticlesP1P2NewPushConstants,
    ),
    IntegrateParticlesP45New(
      &'a mut gpu::compute_push_constants::IntegrateParticlesP45NewPushConstants,
    ),
    NewParticlesCompactReset(
      &'a mut gpu::compute_push_constants::NewParticlesCompactResetPushConstants,
    ),
    NewParticlesEmit(&'a mut gpu::compute_push_constants::NewParticlesEmitPushConstants),
    NewParticlesCompact(&'a mut gpu::compute_push_constants::NewParticlesCompactPushConstants),
    NewParticlesOffsetParticlesPush(
      &'a mut gpu::compute_push_constants::NewParticlesOffsetParticlesPushConstants,
    ),
  }
}

// RAII Cleanup Guard for transient resources.
// Ensures they are always destroyed when the block ends (success or error).
struct TransientCleanup<'a> {
  device: &'a LogicalDevice,
  resources: Option<TransientCleanupResources>,
}
impl<'a> TransientCleanup<'a> {
  fn command_only(
    device: &'a LogicalDevice,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
  ) -> Self {
    Self {
      device,
      resources: Some(TransientCleanupResources {
        set_layout: vk::DescriptorSetLayout::null(),
        pipeline_layout: vk::PipelineLayout::null(),
        descriptor_pool: vk::DescriptorPool::null(),
        command_pool,
        fence,
      }),
    }
  }
}
struct TransientCleanupResources {
  set_layout: vk::DescriptorSetLayout,
  pipeline_layout: vk::PipelineLayout,
  descriptor_pool: vk::DescriptorPool,
  command_pool: vk::CommandPool,
  fence: vk::Fence,
}
impl DeviceResource for TransientCleanupResources {
  fn cleanup(&mut self, device: &LogicalDevice) {
    unsafe {
      if !self.command_pool.is_null() {
        device.destroy_command_pool(self.command_pool, None);
      }
      if !self.descriptor_pool.is_null() {
        device.destroy_descriptor_pool(self.descriptor_pool, None);
      }
      if !self.pipeline_layout.is_null() {
        device.destroy_pipeline_layout(self.pipeline_layout, None);
      }
      if !self.set_layout.is_null() {
        device.destroy_descriptor_set_layout(self.set_layout, None);
      }
      if !self.fence.is_null() {
        device.destroy_fence(self.fence, None);
      }
    }
  }
}
impl<'a> Drop for TransientCleanup<'a> {
  fn drop(&mut self) {
    if let Some(mut res) = self.resources.take() {
      res.cleanup(&self.device);
    }
  }
}

struct AllocJanitor {
  buffer: vk::Buffer,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::AllocatorView,
}

impl Drop for AllocJanitor {
  fn drop(&mut self) {
    unsafe { self.allocator.destroy_buffer(self.buffer, &mut self.alloc) };
  }
}

fn gpu_sync_info_to_flags(
  info: gpu::CommandBufferSyncInfoStageMask,
  is_signal: bool,
) -> vk::PipelineStageFlags2 {
  use gpu::CommandBufferSyncInfoStageMask;
  match info {
    CommandBufferSyncInfoStageMask::TopBottom => {
      if is_signal {
        vk::PipelineStageFlags2::BOTTOM_OF_PIPE
      } else {
        vk::PipelineStageFlags2::TOP_OF_PIPE
      }
    }
    CommandBufferSyncInfoStageMask::Transfer => vk::PipelineStageFlags2::TRANSFER,
    CommandBufferSyncInfoStageMask::VertexAttributeInput => {
      vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
    }
  }
}

/// value used to constrain the maximum amount of dispatched workgroups for compute shaders using
/// the grid stride loop strategy
pub const GRID_STRIDE_WORKGROUP_SIZE_SATURATION_VALUE: u32 = 4096;

mod meshutils {
  use super::*;

  pub struct CreateResourcesState {
    pub pipeline_key: gpu::PipelineKey,
    pub outline_pipeline_key: gpu::PipelineKey,
    pub arena_arc:
      alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMesh2RenderResourceArchetypeArena>>,
    pub position_data: alloc::vec::Vec<f32>,
    pub attribute_data: alloc::vec::Vec<f32>,
    pub vma: vk_mem::AllocatorView,
    pub staging_arena_ptr: *const memory::FrameStagingArena,
    pub discard_pool_ptr: *const resources::DiscardPool,
    pub descriptor_pool_arc: alloc::sync::Arc<descriptors::DescriptorPools>,
    pub sky_image_clone: Option<resources::Image>,
    pub linear_sampler: vk::Sampler,
  }

  impl CreateResourcesState {
    #[named]
    pub fn new(
      res: &DeviceResources,
      comp: &StaticMeshComponent,
      pe_handle: gpu::PresentationEngineHandle,
    ) -> GpuResult<Self> {
      let pe = wait_for_pe_direct!(&res.live_presentation_engines, pe_handle)?;
      let (pipeline_key, outline_pipeline_key) = {
        let lock = pe.archetypes().registry.read();
        let a = lock.get(&ArchetypeId::Mesh).ok_or(gpu_err_archetype_absent!())?;
        (a.pipeline_key(), a.outline_pipeline_key().unwrap())
      };
      let arena_arc =
        alloc::sync::Arc::clone(res.physical_mesh2_render_archetype_arena.as_ref().unwrap());
      let position_data = extract_position_data(comp.mesh.as_ref());
      let attribute_data = extract_attribute_data(comp.mesh.as_ref());
      let vma = res.allocator.allocator.as_allocator_view();
      let sky_image_clone = res.sky_image.read().as_ref().map(|sky| resources::Image {
        image: sky.image,
        image_view: sky.image_view,
        allocation: sky.allocation,
      });
      Ok(Self {
        pipeline_key,
        outline_pipeline_key,
        arena_arc,
        position_data,
        attribute_data,
        vma,
        staging_arena_ptr: res.frame_staging_arena.read().as_ref().unwrap() as *const _,
        discard_pool_ptr: (&res.discard_pool) as *const _,
        descriptor_pool_arc: alloc::sync::Arc::clone(res.descriptor_pool.read().as_ref().unwrap()),
        sky_image_clone,
        linear_sampler: res.linear_sampler.get(),
      })
    }

    pub fn discard_pool(&self) -> &resources::DiscardPool {
      unsafe { self.discard_pool_ptr.as_ref_unchecked() }
    }

    pub fn frame_staging_arena(&self) -> &memory::FrameStagingArena {
      unsafe { self.staging_arena_ptr.as_ref_unchecked() }
    }
  }

  pub fn pbr_material_data(
    comp: &StaticMeshComponent,
    tex_flags: TextureFlags,
  ) -> gpu::MaterialData {
    use bytemuck::Zeroable;
    let mut res = gpu::MaterialData::zeroed();
    res.base_albedo = [0.8, 0.8, 0.8, 1.0];
    res.emissive_color = comp.emissive_color;
    res.base_ao = 1.0;
    res.texture_flags = tex_flags.bits();

    res
  }

  pub fn object_data_identity_matrix() -> gpu::ObjectData {
    gpu::ObjectData {
      #[rustfmt::skip]
      model: [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
      ],
    }
  }
}

// TODO RE-ENABLE
// #[cfg(test)]
// mod test_render;

#[cfg(test)]
mod test_swapchain;

// #[cfg(test)]
// mod test_ui_text;

#[cfg(test)]
pub mod test_utils;
#[cfg(test)]
mod ui_tests;

#[cfg(test)]
mod test_pipelines;
