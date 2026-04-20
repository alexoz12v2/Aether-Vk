use crate::{
  gpu::{RenderBackendId, RenderFrontend, VULKAN_RENDER_BACKEND, METAL_RENDER_BACKEND, D3D12_RENDER_BACKEND},
  traits::InitWithRuntime,
  types::{EngineError, EngineResult, GpuError, RuntimeParams},
};
use alloc::vec::Vec;

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

pub fn get_available_render_backends() -> Vec<&'static str> {
  let mut backends = Vec::new();

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
  {
    let params = RuntimeParams {
      render_backend_params: heapless::index_map::FnvIndexMap::new(),
    };
    if let Ok(mut context) = vulkan::VulkanRenderContext::init_with_runtime(&params) {
      use crate::gpu::RenderContext;
      if context.init_device(0, &crate::gpu::DeviceAdditionalParams::new()).is_ok() {
        backends.push("Vulkan");
      }
    }
  }

  #[cfg(target_os = "macos")]
  backends.push("Metal");

  #[cfg(target_os = "windows")]
  backends.push("Direct3D12");

  backends
}

pub fn get_available_kernels() -> Vec<&'static str> {
  let mut kernels = alloc::vec!["CPU Scalar", "CPU SSE/AVX", "CPU NEON"];

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
  {
    let params = RuntimeParams {
      render_backend_params: heapless::index_map::FnvIndexMap::new(),
    };
    if let Ok(mut context) = vulkan::VulkanRenderContext::init_with_runtime(&params) {
      use crate::gpu::RenderContext;
      if context.init_device(0, &crate::gpu::DeviceAdditionalParams::new()).is_ok() {
        kernels.push("Vulkan Compute");
      }
    }
  }

  kernels.push("CUDA");
  kernels
}
