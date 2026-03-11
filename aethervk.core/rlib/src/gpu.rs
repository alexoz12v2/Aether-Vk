use core::ffi;

use crate::types::{EngineResult, GpuResult};

// Re-export what is necessary from backends
pub use super::gpu_backends::new_render_frontend;
pub use super::gpu_backends::{vulkan::constants};

use heapless::index_map::FnvIndexMap;
use alloc::boxed::Box;
#[cfg(debug_assertions)]
use alloc::string::String;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderBackendId(pub u64);
pub const NULL_RENDER_BACEKND: RenderBackendId = RenderBackendId(0);
pub const VULKAN_RENDER_BACKEND: RenderBackendId = RenderBackendId(1);

pub trait RenderDevice: Send + Sync {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> String;

  fn context_id(&self) -> u64;
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderDeviceHandle(pub u64);

/// backend specific additional device init parameters
pub type DeviceAdditionalParams = FnvIndexMap<u64, usize, 8>;

pub trait RenderContext: Send + Sync {
  fn backend_id(&self) -> RenderBackendId;

  fn init_device(
    &mut self,
    index: usize,
    additional_params: &DeviceAdditionalParams,
  ) -> GpuResult<RenderDeviceHandle>;

  fn deref_device_and(
    &self,
    dev_handle: RenderDeviceHandle,
    p_user_data: *mut ffi::c_void,
    f: fn(dev: &dyn RenderDevice, p_user_data: *mut ffi::c_void) -> GpuResult<()>,
  ) -> Option<GpuResult<()>>;
}

// NOTE: This is a box like type, so we don't need to box it when returning it to cdylib,
// we can instead use the ManualDrop mechanism
pub struct RenderFrontend<'a> {
  backend: spin::RwLock<Box<dyn RenderContext + 'a>>,
}

impl<'a> RenderFrontend<'a> {
  pub fn take_and<T>(
    &self,
    f: impl FnOnce(&dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_read() {
      Some(guard) => Some(f(guard.as_ref())),
      None => None,
    }
  }

  pub fn take_mut_and<T>(
    &mut self,
    f: impl FnOnce(&mut dyn RenderContext) -> EngineResult<T>,
  ) -> Option<EngineResult<T>> {
    match self.backend.try_write() {
      Some(mut guard) => Some(f(guard.as_mut())),
      None => None,
    }
  }
}

// Boxing mechanism used by factory method in `gpu_backends` `new_render_frontend`
impl<'a, T> From<T> for RenderFrontend<'a>
where
  T: RenderContext + 'a,
{
  fn from(value: T) -> Self {
    RenderFrontend {
      backend: spin::RwLock::new(Box::new(value)),
    }
  }
}
