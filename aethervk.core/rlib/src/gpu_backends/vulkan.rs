use core::{str::FromStr};

use crate::{
  gpu::{
    DeviceAdditionalParams, RenderBackendId, RenderContext, RenderDevice, RenderDeviceHandle,
    VULKAN_RENDER_BACKEND,
  },
  gpu_backends::{MAX_DEVICES, vulkan::utils::PhysicalDeviceQueryInput},
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
#[ouroboros::self_referencing]
pub(super) struct VulkanRenderContext {
  instance: instance::Instance,
  #[borrows(instance)]
  #[covariant]
  live_devices: FnvIndexMap<RenderDeviceHandle, device::Device<'this>, MAX_DEVICES>,
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

    Ok(
      VulkanRenderContextBuilder {
        instance,
        live_devices_builder: |_| live_devices,
      }
      .build(),
    )
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

    self.with_mut(|fields| {
      if !fields.live_devices.contains_key(&handle) {
        let device = device::Device::new(fields.instance, index, &query_input)?;
        unsafe {
          fields
            .live_devices
            .insert(handle, device)
            .unwrap_unchecked();
        }
      }

      Ok(handle)
    })
  }

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    f: &mut dyn FnMut(&dyn RenderDevice) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    self.with_live_devices(|live_devices| live_devices.get(&dev_handle).map(|device| f(device)))
  }
}
