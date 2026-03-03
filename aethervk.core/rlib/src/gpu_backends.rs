use crate::{
  gpu::{RenderBackendId, RenderFrontend, VULKAN_RENDER_BACKEND},
  traits::InitWithRuntime,
  types::{EngineError, EngineResult, GpuError, RuntimeParams},
};

#[cfg(all(
  not(target_arch = "wasm32"),
  any(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "android",
    target_os = "ios"
  )
))]
pub(super) mod vulkan;

pub(self) const MAX_DEVICES: usize = 8;

pub fn new_render_frontend(
  ty: RenderBackendId,
  params: &'_ RuntimeParams,
) -> EngineResult<RenderFrontend<'_>> {
  match ty {
    VULKAN_RENDER_BACKEND => {
      vulkan::VulkanRenderContext::init_with_runtime(params).map(|back| back.into())
    }
    _ => Err(EngineError::Gpu(GpuError::UnsupportedFeature)),
  }
}
