use core::{
  str::FromStr,
  ffi::{self, CStr},
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

use alloc::{ffi::CString, string::ToString, sync };
use ash::vk;
use heapless::index_map::FnvIndexMap;

pub(super) mod device;
pub(super) mod instance;
pub(super) mod utils;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
  index: u32,
  generation: u32,
}

// ---------------------------- Runtime Params ----------------------------
pub mod constants {
  pub const RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR: super::RuntimeParamsIndex = 1000;
}

/// Structure containing main vulkan handles. Shared by both Runtime Interface and compute interface
/// - Massive in size, supposed to be heap allocated and constructed on the heap in-place
#[ouroboros::self_referencing]
pub(super) struct VulkanCore {
  instance: instance::Instance,
  #[borrows(instance)]
  #[covariant]
  live_devices: FnvIndexMap<RenderDeviceHandle, device::Device<'this>, MAX_DEVICES>,
}

static S_VULKAN_CORE: spin::Mutex<sync::Weak<spin::RwLock<VulkanCore>>> =
  spin::Mutex::new(sync::Weak::new());

pub(super) struct VulkanRenderContext {
  core: sync::Arc<spin::RwLock<VulkanCore>>,
  // graphics specific members
}

impl VulkanCore {
  fn from_path(base_override_path: Option<&CStr>) -> GpuResult<Self> {
    let instance = unsafe { instance::Instance::new(base_override_path) }?;
    let live_devices = FnvIndexMap::new();

    Ok(
      VulkanCoreBuilder {
        instance,
        live_devices_builder: |_| live_devices,
      }
      .build(),
    )
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
    let query_input =
      PhysicalDeviceQueryInput::from_params(additional_params).ok_or(GpuError::InvalidArgument)?;

    self.core.write().with_mut(|fields| {
      if !fields.live_devices.contains_key(&handle) {
        // 1. We need to reserve space in the heapless map.
        // Since heapless doesn't have an 'entry' API for uninitialized memory,
        // we insert a "dummy" (zeroed) value first.
        // To avoid 1.5KB of zeros on the stack, we use unsafe to bit-copy an uninit value.
        unsafe {
          #[allow(invalid_value)]
          let uninit_val = core::mem::MaybeUninit::<device::Device>::uninit().assume_init();
          fields
            .live_devices
            .insert(handle, uninit_val)
            .unwrap_unchecked();
        }

        // 2. Get a mutable pointer to the slot we just created in the heap-resident map.
        let dst_ptr = fields.live_devices.get_mut(&handle).unwrap() as *mut device::Device;

        // 3. Construct the device directly into that heap location.
        unsafe {
          device::Device::init_at_ptr(dst_ptr, fields.instance, index, &query_input)?;
        }
      }

      Ok(handle)
    })
  }

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    self.core.read().with_live_devices(|live_devices| {
      live_devices
        .get(&dev_handle)
        .map(|device| f(device, p_user_data))
    })
  }
}
