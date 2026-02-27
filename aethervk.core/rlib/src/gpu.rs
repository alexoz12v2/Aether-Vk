// Note: This doesn't import `gpu_backends`
#[cfg(debug_assertions)]
use alloc::string::String;

use crate::types::GpuResult;

pub trait RenderDevice {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> String;

  fn context_id(&self) -> u64;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderDeviceHandle {
  context_id: u64,
}

pub trait RenderContext: Send + Sync {
  fn init_device(&self, index: usize) -> GpuResult<RenderDeviceHandle>;

  fn deref_device_and(
    &mut self,
    dev_handle: RenderDeviceHandle,
    f: &mut dyn FnMut(&dyn RenderDevice) -> GpuResult<()>,
  ) -> Option<GpuResult<()>>;
}
