use core::{ffi::CStr, str::FromStr};

use crate::{
  gpu::{RenderBackendId, RenderContext, RenderDevice, RenderDeviceHandle, VULKAN_RENDER_BACKEND},
  gpu_backends::MAX_DEVICES,
  traits::InitWithRuntime,
  types::{EngineResult, GpuError, GpuResult, RuntimeParams, RuntimeParamsIndex},
};

use alloc::{ffi::CString, string::ToString};
use heapless::index_map::FnvIndexMap;

pub(super) mod device;
pub(super) mod instance;
pub(super) mod utils;

// ---------------------------- Runtime Params ----------------------------
pub mod constants {
  pub const RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR: super::RuntimeParamsIndex = 1000;
}

// ---------------------------- Context Interface -------------------------
pub(super) struct VulkanRenderContext {
  instance: instance::Instance,
  live_devices: FnvIndexMap<RenderDeviceHandle, device::Device, MAX_DEVICES>,
}

impl VulkanRenderContext {
  fn device_id_from_index(&self, dev_idx: usize) -> RenderDeviceHandle {
    todo!()
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

    let instance = unsafe { instance::Instance::new(base_override_path.as_deref()) }?;
    let live_devices = FnvIndexMap::new();

    Ok(Self {
      instance,
      live_devices,
    })
  }
}

impl RenderContext for VulkanRenderContext {
  fn backend_id(&self) -> RenderBackendId {
    VULKAN_RENDER_BACKEND
  }

  fn init_device(&self, index: usize) -> GpuResult<RenderDeviceHandle> {
    let handle = self.device_id_from_index(index);
    if !self.live_devices.contains_key(&handle) {
      todo!()
    } else {
      Ok(handle)
    }
  }

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    f: &mut dyn FnMut(&dyn RenderDevice) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    self.live_devices.get(&dev_handle).map(|device| f(device))
  }
}
