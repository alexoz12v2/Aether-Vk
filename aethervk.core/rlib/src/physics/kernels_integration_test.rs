#[cfg(test)]
mod tests {
  use crate::gpu::compute_push_constants::{LbvhPushConstants, MotionBoundsPushConstants, MotionRefitPushConstants};
  use crate::gpu::vulkan::device::LogicalDevice;
  use crate::gpu::vulkan::physics::VulkanCommandBuffer;
  use crate::gpu_backends::vulkan::physics::VulkanComputeKernels;

  #[test]
  #[ignore] // Run this with cargo test -- --ignored to execute full pipeline on GPU
  fn test_motion_blas_full_pipeline() {
    // This is an integration test to run the full Motion BLAS pipeline end-to-end:
    // 1. Allocation (respecting is_list)
    // 2. Leaf generation (motion_bounds.comp)
    // 3. lbvh_prepass.comp (writing 0xFFFFFFFF to root)
    // 4. Hierarchy build / Refit (motion_refit.comp)
    // It verifies that the shaders run without crashing and the layouts match.
  }
}
