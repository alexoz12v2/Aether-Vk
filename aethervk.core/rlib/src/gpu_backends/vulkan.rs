//! vulkan module.

#[cfg(debug_assertions)]
pub static DEBUG_RENDER_THREAD_CPU_TIME_MS: core::sync::atomic::AtomicU64 =
  core::sync::atomic::AtomicU64::new(0);

#[cfg(debug_assertions)]
pub static DEBUG_RENDER_THREAD_GPU_TIME_MS: core::sync::atomic::AtomicU64 =
  core::sync::atomic::AtomicU64::new(0);

use core::{
  ffi::{self, CStr},
  str::FromStr,
};

use crate::{
  gpu::{
    DeviceAdditionalParams, RenderBackendId, RenderContext, RenderDevice, RenderDeviceHandle,
    VULKAN_RENDER_BACKEND,
  },
  gpu_backends::{MAX_DEVICES, vulkan::utils::PhysicalDeviceQueryInput},
  traits::InitWithRuntime,
  types::{EngineResult, GpuError, GpuResult, RuntimeParams, RuntimeParamsIndex},
};

use alloc::{ffi::CString, string::ToString, sync};
use heapless::index_map::FnvIndexMap;

pub mod device;
pub mod instance;
pub mod physics;
pub mod utils;

pub mod shader_tests;

#[cfg(debug_assertions)]
pub mod renderdoc;

#[cfg(test)]
pub mod mock_kernels;

#[cfg(test)]
pub mod mock_scene_data;

#[cfg(test)]
pub mod trajectory_tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncMode {
  /// CPU uploads data. Needs Transfer Write -> Vertex Read barrier.
  CpuUpload,
  /// Compute writes data on the same queue. Needs Compute Write -> Vertex Read.
  SameQueueCompute,
  /// Compute writes data on a different queue family.
  /// `is_release_pass`: True when recording on the Compute Queue, False for Graphics Queue.
  CrossQueueCompute {
    src_family: u32,
    dst_family: u32,
    is_release_pass: bool,
  },
}

// ---------------------------- Runtime Params ----------------------------
pub mod constants {
  /// TODO: Document this item
  pub const RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR: super::RuntimeParamsIndex = 1000;
}

/// Structure containing main vulkan handles. Shared by both Runtime Interface and compute interface
/// - Massive, supposed to be heap allocated and constructed on the heap in-place
pub(super) struct VulkanCore {
  instance: alloc::sync::Arc<instance::Instance>,
  live_devices:
    FnvIndexMap<RenderDeviceHandle, core::mem::MaybeUninit<device::Device>, MAX_DEVICES>,
}

unsafe impl Sync for VulkanCore {}
unsafe impl Send for VulkanCore {}

static S_VULKAN_CORE: spin::Mutex<sync::Weak<parking_lot::RwLock<VulkanCore>>> =
  spin::Mutex::new(sync::Weak::new());

/// TODO: Document this item
pub(super) struct VulkanRenderContext {
  core: sync::Arc<parking_lot::RwLock<VulkanCore>>,
  // graphics specific members
}

impl VulkanCore {
  fn from_path(
    base_override_path: Option<&CStr>,
    validation_error_callback: Option<fn(&str)>,
  ) -> GpuResult<Self> {
    let instance = alloc::sync::Arc::new(unsafe {
      instance::Instance::new(base_override_path, validation_error_callback)
    }?);
    let live_devices = FnvIndexMap::new();

    Ok(Self {
      instance,
      live_devices,
    })
  }
}

impl Drop for VulkanCore {
  fn drop(&mut self) {
    while let Some(k) = self.live_devices.keys().next().copied() {
      if let Some(mut dev_uninit) = self.live_devices.remove(&k) {
        // SAFETY: items remaining in `live_devices` were initialized in `init_device` therefore
        // they are well defined
        unsafe {
          core::ptr::drop_in_place(dev_uninit.as_mut_ptr());
        }
      }
    }
  }
}

impl VulkanRenderContext {
  fn device_id_from_index(&self, dev_idx: usize) -> RenderDeviceHandle {
    RenderDeviceHandle((dev_idx as u64) + 1)
  }

  /// Test-only: call `f` with the concrete Vulkan `Device`, which implements
  /// the `Kernels` trait, allowing shader unit tests to call compute kernels
  /// directly without going through the `dyn RenderDevice` abstraction.
  #[cfg(test)]
  pub(super) fn with_device_as_kernels<F, R>(
    &self,
    dev_handle: RenderDeviceHandle,
    f: F,
  ) -> Option<R>
  where
    F: FnOnce(&device::Device) -> R,
  {
    let core = self.core.read();
    core
      .live_devices
      .get(&dev_handle)
      .map(|device| unsafe { f(device.assume_init_ref()) })
  }
}

// TODO inject runtime callbacks (eg logging)
impl InitWithRuntime<VulkanRenderContext> for VulkanRenderContext {
  fn init_with_runtime(params: &RuntimeParams) -> EngineResult<Self> {
    let base_override_path = params
      .render_backend_params
      .get(&constants::RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR)
      .map(|str| CString::from_str(str))
      .transpose()
      .map_err(|_| {
        GpuError::BackendSpecific("Invalid RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR".to_string())
      })?;

    // --- TEST FIX: Bypass the global cache so every test gets an isolated GPU Device ---
    #[cfg(test)]
    {
      let core = sync::Arc::new(parking_lot::RwLock::new(VulkanCore::from_path(
        base_override_path.as_deref(),
        params.validation_error_callback,
      )?));
      return Ok(Self { core });
    }

    // --- PRODUCTION BEHAVIOR ---
    #[cfg(not(test))]
    {
      let mut s_core = S_VULKAN_CORE.lock();
      let core = if let Some(core) = s_core.upgrade() {
        core
      } else {
        let new_core = sync::Arc::new(parking_lot::RwLock::new(VulkanCore::from_path(
          base_override_path.as_deref(),
          params.validation_error_callback,
        )?));
        *s_core = sync::Arc::downgrade(&new_core);
        new_core
      };

      Ok(Self { core })
    }
  }
}

// reference utils/PhysicalDeviceQueryInput
#[allow(unused)]
/// TODO: Document this item
pub const DEVICE_ADDIDITIONAL_PARAM_WL_DISPLAY: u64 = 0;
#[allow(unused)]
/// TODO: Document this item
pub const DEVICE_ADDIDITIONAL_PARAM_XCB_CONNECTION: u64 = 1;
#[allow(unused)]
/// TODO: Document this item
pub const DEVICE_ADDIDITIONAL_PARAM_XCB_VISUALID: u64 = 2;
#[allow(unused)]
/// TODO: Document this item
pub const DEVICE_ADDIDITIONAL_PARAM_DPY: u64 = 3;
#[allow(unused)]
/// TODO: Document this item
pub const DEVICE_ADDIDITIONAL_PARAM_VISUAL_ID: u64 = 4;
pub const DEVICE_ADDIDITIONAL_PARAM_DEBUG_SHADERS: u64 = 5;

impl RenderContext for VulkanRenderContext {
  fn backend_id(&self) -> RenderBackendId {
    VULKAN_RENDER_BACKEND
  }

  fn init_device(
    &mut self,
    index: usize,
    additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle> {
    // A device created right after another process destroyed its own (back-to-back test processes,
    // an app relaunch) can fail transiently while the driver still tears the old one down
    // (`vkCreateDevice` → VK_ERROR_INITIALIZATION_FAILED / DEVICE_LOST in the NVIDIA ICD). That is
    // back-pressure, not a configuration error: back off exponentially (with jitter) and retry.
    let attempts = device_init_attempts();
    let mut attempt = 0u32;
    loop {
      CREATING_DEVICE.store(true, core::sync::atomic::Ordering::Release);
      let res = self.init_device_once(index, additional_params);
      CREATING_DEVICE.store(false, core::sync::atomic::Ordering::Release);
      match res {
        Err(e) if attempt + 1 < attempts && is_transient_device_init_error(&e) => {
          let delay_ms = device_init_backoff_ms(attempt);
          aethervk_oshal_rlib::log!(
            "[Vulkan] device creation failed transiently ({e}), retry {}/{} in {delay_ms} ms",
            attempt + 1,
            attempts - 1
          );
          aethervk_oshal_rlib::os::native::this_thread::sleep_for(
            core::time::Duration::from_millis(delay_ms),
          );
          attempt += 1;
        }
        other => return other,
      }
    }
  }

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    let core = self.core.read();
    // SAFETY: if it exists in the map, it was inserted with `init_device`
    core
      .live_devices
      .get(&dev_handle)
      .map(|device| unsafe { f(device.assume_init_ref(), p_user_data) })
  }

  #[cfg(target_os = "linux")]
  fn linux_surface_support(&self) -> instance::LinuxSurfaceSupport {
    self.core.read().instance.linux_surface_support
  }
}

/// `vkCreateDevice` (and the setup around it) is running: the debug messenger reports the loader's
/// `terminator_CreateDevice` failure of a transient attempt as a log line instead of forwarding it
/// to the (test: panicking) validation callback, so [`VulkanRenderContext::init_device`] can retry.
pub(crate) static CREATING_DEVICE: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

/// Device creation attempts (`AETHERVK_DEVICE_INIT_ATTEMPTS`, default 6: ~3 s of backoff in all).
pub fn device_init_attempts() -> u32 {
  aethervk_oshal_rlib::os::env::var("AETHERVK_DEVICE_INIT_ATTEMPTS")
    .and_then(|s| s.trim().parse::<u32>().ok())
    .unwrap_or(6)
    .clamp(1, 20)
}

/// Failures of device creation worth retrying: the driver is busy tearing down another device.
pub fn is_transient_device_init_error(e: &GpuError) -> bool {
  match e {
    GpuError::DeviceLost | GpuError::OutOfMemory => true,
    GpuError::BackendSpecific(s) => {
      s.contains("ERROR_INITIALIZATION_FAILED")
        || s.contains("Initialization of an object has failed")
    }
    _ => false,
  }
}

/// Exponential backoff with jitter: 100 ms · 2^attempt, ±25 %, capped at 2 s.
pub fn device_init_backoff_ms(attempt: u32) -> u64 {
  let base = (100u64 << attempt.min(5)).min(2000);
  let t = aethervk_oshal_rlib::os::time::get_monotonic_time() as u64;
  let jitter = (t ^ (t >> 17)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 54; // 0..1023
  base * (768 + jitter / 2) / 1024
}

impl VulkanRenderContext {
  /// One device creation attempt (see the retrying [`RenderContext::init_device`]).
  fn init_device_once(
    &mut self,
    index: usize,
    additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle> {
    let handle = self.device_id_from_index(index);
    let mut query_input = PhysicalDeviceQueryInput::from_params(additional_params)
      .ok_or(GpuError::InvalidArgument("vulkan.rs:128".to_string()))?;

    let mut core = self.core.write();

    // Propagate which Linux surface extensions are actually enabled into query_input
    #[cfg(target_os = "linux")]
    {
      query_input.linux_surface_support = core.instance.linux_surface_support;
    }

    if !core.live_devices.contains_key(&handle) {
      let instance = alloc::sync::Arc::clone(&core.instance);

      // 1. We need to reserve space in the heapless map.
      // Since heapless doesn't have an 'entry' API for uninitialized memory,
      // we insert a "dummy" (zeroed) value first.
      // To avoid 1.5KB of zeros on the stack, we use unsafe to bit-copy an uninit value.
      unsafe {
        let uninit_val = core::mem::MaybeUninit::<device::Device>::uninit();
        core.live_devices.insert(handle, uninit_val).unwrap_unchecked();
      }

      struct UninitGuard<'a> {
        map: &'a mut FnvIndexMap<
          RenderDeviceHandle,
          core::mem::MaybeUninit<device::Device>,
          MAX_DEVICES,
        >,
        handle: RenderDeviceHandle,
        defused: bool,
      }
      impl<'a> Drop for UninitGuard<'a> {
        fn drop(&mut self) {
          if !self.defused {
            self.map.remove(&self.handle);
          }
        }
      }

      let mut guard = UninitGuard {
        map: &mut core.live_devices,
        handle,
        defused: false,
      };

      // 2. Get a mutable pointer to the slot we just created in the heap-resident map.
      let dst_ptr = unsafe { guard.map.get_mut(&handle).unwrap_unchecked().as_mut_ptr() };

      // 3. Construct the device directly into that heap location.
      unsafe {
        device::Device::init_at_ptr(dst_ptr, instance, index, &query_input)?;
      }

      guard.defused = true;
    }

    Ok(handle)
  }
}

#[derive(Clone, Copy)]
pub struct SimpleSimulationStepParams<'a> {
  pub device: &'a device::Device,
  pub scene: &'a crate::scene::Scene,
  // assumes that on a given loop multiple particle systems emit on the same compute timeline value
  pub particles_emit_sync: Option<crate::gpu::CommandBufferSyncInfo>,
  pub t_start: aethervk_oshal_rlib::os::time::timeus_t,
  pub t_end: aethervk_oshal_rlib::os::time::timeus_t,
}

#[derive(Clone, Copy)]
pub struct SimpleParticleEmissionStepParams<'a> {
  pub device: &'a device::Device,
  pub scene: &'a crate::scene::Scene,
  pub t_prev: aethervk_oshal_rlib::os::time::timeus_t,
  pub t_current: aethervk_oshal_rlib::os::time::timeus_t,
}

/// Vulkan Utility to allocate a vulkan command buffer from a given
/// [`crate::gpu_backends::vulkan::device::commands::CommandPools`], with the possibility to retry
/// when Pool fails because out of host memory (fixed capacities in use there). Therefore, on
/// failure, a discard pool recycling phase is called and another attempt is made. This is done
/// every 0.1ms interval with a deadline of 1ms
/// Note: `current_timeline` is the value which will be signaled from the previous graphics queue
/// command, not the next signal value.
fn allocate_primary_vk_command_buffer(
  device: &device::LogicalDevice,
  command_pools: &device::commands::CommandPools,
  discard_pool: &device::resources::DiscardPool,
  queue_family_index: u32,
  cmd_id: device::commands::CommandBufferId,
  current_timeline: u64,
) -> GpuResult<ash::vk::CommandBuffer> {
  use aethervk_oshal_rlib::os::{native::this_thread, time::get_monotonic_time};

  let tid = this_thread::id();
  let start = get_monotonic_time();
  while (get_monotonic_time() - start) < 1_000_i64 {
    if let Ok(cmd) = command_pools.allocate_primary(device, tid, queue_family_index, cmd_id) {
      return Ok(cmd);
    }

    let items = discard_pool.pop_ready_items(current_timeline);
    device::resources::DiscardPool::destroy_items_lock_free(device, items);

    this_thread::sleep_for(core::time::Duration::from_micros(100));
  }

  // Deadline reached. Die.
  Err(GpuError::BackendSpecific(
    "Couldn't allocate command buffer within deadline. OOM".to_string(),
  ))
}

#[cfg(test)]
mod device_init_retry_tests {
  use super::*;

  /// Only "driver busy" failures are retried; configuration errors fail at once.
  #[test]
  fn transient_device_init_errors_are_classified() {
    assert!(is_transient_device_init_error(&GpuError::DeviceLost));
    assert!(is_transient_device_init_error(&GpuError::BackendSpecific(
      "Initialization of an object has failed".to_string()
    )));
    assert!(is_transient_device_init_error(&GpuError::BackendSpecific(
      "ERROR_INITIALIZATION_FAILED".to_string()
    )));
    assert!(!is_transient_device_init_error(
      &GpuError::UnsupportedFeature
    ));
    assert!(!is_transient_device_init_error(&GpuError::InvalidArgument(
      "x".to_string()
    )));
  }

  /// Backoff grows exponentially (±25 % jitter), capped at 2 s; the default budget is ~3 s.
  #[test]
  fn device_init_backoff_is_exponential_with_jitter() {
    for a in 0..10u32 {
      let base = (100u64 << a.min(5)).min(2000);
      let d = device_init_backoff_ms(a);
      assert!(
        d >= base * 3 / 4 - 1 && d <= base * 5 / 4 + 1,
        "attempt {a}: {d} ms (base {base})"
      );
    }
    let total: u64 = (0..device_init_attempts() - 1).map(|a| (100u64 << a).min(2000)).sum();
    assert!(total >= 1500 && total <= 6000, "{total} ms");
  }
}
