use fchashmap::FcHashMap;

use crate::{
  gpu_backends::MAX_DEVICES,
  gpu::{RenderContext, RenderDeviceHandle, RenderDevice},
  traits::InitWithRuntime,
  types::{GpuResult, RuntimeParams, EngineResult},
};

pub(super) mod device;
pub(super) mod instance;
pub(super) mod utils;

pub(super) struct VulkanRenderContext {
  instance: instance::Instance,
  live_devices: FcHashMap<RenderDeviceHandle, device::Device, MAX_DEVICES>,
}

// TODO inject runtime callbacks (eg logging)
impl InitWithRuntime<VulkanRenderContext> for VulkanRenderContext {
  fn init_with_runtime(_params: &RuntimeParams) -> EngineResult<Self> {
    let instance = unsafe { instance::Instance::new() }?;
    let live_devices = FcHashMap::new();

    Ok(Self { instance, live_devices })
  }
}

impl RenderContext for VulkanRenderContext {
  fn init_device(&self, index: usize) -> GpuResult<RenderDeviceHandle> {
    todo!()
  }

  fn deref_device_and(
    &mut self,
    dev_handle: RenderDeviceHandle,
    f: &mut dyn FnMut(&dyn RenderDevice) -> GpuResult<()>,
  ) -> Option<GpuResult<()>> {
    todo!()
  }
}
