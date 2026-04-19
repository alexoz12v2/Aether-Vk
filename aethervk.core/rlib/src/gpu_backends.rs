use crate::{
  gpu::{RenderBackendId, RenderFrontend, VULKAN_RENDER_BACKEND, METAL_RENDER_BACKEND, D3D12_RENDER_BACKEND},
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

#[cfg(target_os = "macos")]
pub(super) mod metal;

#[cfg(target_os = "windows")]
pub(super) mod d3d12;

pub(self) const MAX_DEVICES: usize = 4;

pub fn new_render_frontend(
  ty: RenderBackendId,
  params: &'_ RuntimeParams,
) -> EngineResult<RenderFrontend<'_>> {
  match ty {
    VULKAN_RENDER_BACKEND => {
      vulkan::VulkanRenderContext::init_with_runtime(params).map(|back| back.into())
    }
    #[cfg(target_os = "macos")]
    METAL_RENDER_BACKEND => {
      metal::MetalRenderContext::init_with_runtime(params).map(|back| back.into())
    }
    #[cfg(target_os = "windows")]
    D3D12_RENDER_BACKEND => {
      d3d12::D3d12RenderContext::init_with_runtime(params).map(|back| back.into())
    }
    _ => Err(EngineError::Gpu(GpuError::UnsupportedFeature)),
  }
}
