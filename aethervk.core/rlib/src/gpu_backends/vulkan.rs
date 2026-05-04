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

mod device;
mod instance;
mod utils;

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

static S_VULKAN_CORE: spin::Mutex<sync::Weak<spin::RwLock<VulkanCore>>> =
  spin::Mutex::new(sync::Weak::new());

pub(super) struct VulkanRenderContext {
  core: sync::Arc<spin::RwLock<VulkanCore>>,
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

impl VulkanRenderContext {
  fn device_id_from_index(&self, dev_idx: usize) -> RenderDeviceHandle {
    RenderDeviceHandle((dev_idx as u64) + 1)
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

    let mut s_core = S_VULKAN_CORE.lock();
    let core = if let Some(core) = s_core.upgrade() {
      core
    } else {
      let new_core = sync::Arc::new(spin::RwLock::new(VulkanCore::from_path(
        base_override_path.as_deref(),
        params.validation_error_callback,
      )?));
      *s_core = sync::Arc::downgrade(&new_core);
      new_core
    };

    Ok(Self { core })
  }
}

// reference utils/PhysicalDeviceQueryInput
#[allow(unused)]
pub const DEVICE_ADDIDITIONAL_PARAM_WL_DISPLAY: u64 = 0;
#[allow(unused)]
pub const DEVICE_ADDIDITIONAL_PARAM_XCB_CONNECTION: u64 = 1;
#[allow(unused)]
pub const DEVICE_ADDIDITIONAL_PARAM_XCB_VISUALID: u64 = 2;
#[allow(unused)]
pub const DEVICE_ADDIDITIONAL_PARAM_DPY: u64 = 3;
#[allow(unused)]
pub const DEVICE_ADDIDITIONAL_PARAM_VISUAL_ID: u64 = 4;

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
    let query_input = PhysicalDeviceQueryInput::from_params(additional_params)
      .ok_or(GpuError::InvalidArgument("vulkan.rs:128"))?;

    let mut core = self.core.write();

    if !core.live_devices.contains_key(&handle) {
      // 1. We need to reserve space in the heapless map.
      // Since heapless doesn't have an 'entry' API for uninitialized memory,
      // we insert a "dummy" (zeroed) value first.
      // To avoid 1.5KB of zeros on the stack, we use unsafe to bit-copy an uninit value.
      unsafe {
        #[allow(invalid_value)]
        let uninit_val = core::mem::MaybeUninit::<device::Device>::uninit().assume_init();
        core.live_devices.insert(handle, uninit_val).unwrap_unchecked();
      }

      // 2. Get a mutable pointer to the slot we just created in the heap-resident map.
      let dst_ptr = core.live_devices.get_mut(&handle).unwrap() as *mut device::Device;

      // 3. Construct the device directly into that heap location.
      unsafe {
        device::Device::init_at_ptr(
          dst_ptr,
          alloc::sync::Arc::clone(&core.instance),
          index,
          &query_input,
        )?;
      }
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
