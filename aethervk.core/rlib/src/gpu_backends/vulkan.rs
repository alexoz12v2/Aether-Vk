//! vulkan module.

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

#[cfg(test)]
pub mod mock_kernels;

#[cfg(test)]
pub mod mock_scene_data;

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
  live_devices: FnvIndexMap<RenderDeviceHandle, device::Device, MAX_DEVICES>,
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
      if let Some(dev) = self.live_devices.remove(&k) {
        drop(dev);
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
    core.live_devices.get(&dev_handle).map(|device| f(device))
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
        #[allow(invalid_value)]
        let uninit_val = core::mem::MaybeUninit::<device::Device>::uninit().assume_init();
        core.live_devices.insert(handle, uninit_val).unwrap_unchecked();
      }

      struct UninitGuard<'a> {
        map: &'a mut FnvIndexMap<RenderDeviceHandle, device::Device, MAX_DEVICES>,
        handle: RenderDeviceHandle,
        defused: bool,
      }
      impl<'a> Drop for UninitGuard<'a> {
        fn drop(&mut self) {
          if !self.defused {
            core::mem::forget(self.map.remove(&self.handle));
          }
        }
      }

      let mut guard = UninitGuard {
        map: &mut core.live_devices,
        handle,
        defused: false,
      };

      // 2. Get a mutable pointer to the slot we just created in the heap-resident map.
      let dst_ptr = guard.map.get_mut(&handle).unwrap() as *mut device::Device;

      // 3. Construct the device directly into that heap location.
      unsafe {
        let init_result = device::Device::init_at_ptr(dst_ptr, instance, index, &query_input);
        if let Err(e) = init_result {
          return Err(e);
        }
      }

      guard.defused = true;
    }

    Ok(handle)
  }

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    let core = self.core.read();
    core.live_devices.get(&dev_handle).map(|device| f(device, p_user_data))
  }
}
