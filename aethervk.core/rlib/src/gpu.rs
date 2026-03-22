use core::ffi;

// Note: this is a no_std environment
// Do not add anything unrelated to gpu interface here. If you want to define components, scene,
// and other modeling, create new files

use crate::scene::{EntityId, PhysicalMeshComponent, TransformComponent};
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

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GpuResourceHandle(pub u64);
pub const NULL_GPU_RESOURCE: GpuResourceHandle = GpuResourceHandle(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderableInstanceId(pub u64);

impl RenderableInstanceId {
  pub fn from_physical_mesh(
    entity_id: EntityId,
    physical_mesh_component: &PhysicalMeshComponent,
  ) -> Self {
    todo!()
  }
}

pub trait RenderDevice: Send + Sync {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> String;

  fn context_id(&self) -> u64;

  /// Creates the surface and initial swapchain
  fn create_presentation_engine(
    &self,
    params: &PresentationEngineParams,
  ) -> GpuResult<PresentationEngineHandle>;

  /// Allows caller to explicitly trigger a resize/recreation
  fn resize_presentation_engine(
    &self,
    handle: PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()>;

  /// Acquires the next image. If it returns NeedsRecreation, the caller should
  /// discard the frame, call resize, and try again next frame
  fn acquire_next_image(&self, handle: PresentationEngineHandle) -> GpuResult<AcquireResult>;

  /// Returns (pipeline, vertex_buffer, index_buffer)
  fn get_or_create_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    transform: &TransformComponent,
    handle: PresentationEngineHandle,
  ) -> GpuResult<(GpuResourceHandle, GpuResourceHandle, GpuResourceHandle)>;

  /// Presents the image. Takes semaphore signaled by a rendering command buffer
  fn present(
    &self,
    handle: PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<SwapchainStatus>;
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct PresentationEngineHandle(pub u64);

/// Status returned by acquire and present operations
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum SwapchainStatus {
  Optimal = 0,
  /// Swapchain is usable but dimensions/properties no longer match the underlying surface
  Suboptimal = 1,
  /// The surface changed drastically (resize) and the current frame must be discarded
  NeedsRecreation = 2,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct AcquireResult {
  pub image_index: u32,
  pub status: SwapchainStatus,
  /// handle to frame synchronization resources recognized by the presentation engine
  pub frame_index: u64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OpaqueNativeHandleInfo {
  pub ptr0: *mut ffi::c_void,
  pub ptr1: *mut ffi::c_void,
}

/// Parameters passed from Avalonia to create the surface/swapchain
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PresentationEngineParams {
  pub width: u32,
  pub height: u32,
  pub vsync: bool,
  pub window_info: OpaqueNativeHandleInfo,
}

pub mod frame;
