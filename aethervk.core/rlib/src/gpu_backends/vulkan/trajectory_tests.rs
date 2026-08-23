#![cfg(test)]

use crate::{
  gpu::{
    DeviceAdditionalParams, PipelineKey, RenderContext, RenderDevice, TrajectoryPushConstants,
  },
  gpu_backends::vulkan::{
    device::Device,
    VulkanRenderContext,
  },
  scene::{trajectory::TrajectoryComponent, EntityId, Scene},
  traits::InitWithRuntime,
  types::{EngineResult, RuntimeParams},
};
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use alloc::sync::Arc;
use parking_lot::RwLock;

fn run_trajectory_test<TVerify>(
  verify: TVerify,
) where
  TVerify: FnOnce(&Device) -> EngineResult<()>,
{
  crate::gpu::set_asset_dir_for_tests();

  let mut params = RuntimeParams::new_with_callback(None);
  params.render_backend_params.insert(
    crate::gpu_backends::vulkan::constants::RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR,
    "".to_string(),
  );

  let mut render_ctx = match VulkanRenderContext::init_with_runtime(&params) {
    Ok(ctx) => ctx,
    Err(_) => return, // Skip if Vulkan is unsupported locally
  };

  let additional_params = DeviceAdditionalParams::new();
  let dev_handle = render_ctx.init_device(0, &additional_params).unwrap();

  render_ctx.with_device_as_kernels(dev_handle, |device| {
    verify(device).unwrap();
  });
}

#[test]
fn test_trajectory_rendering_api() {
  run_trajectory_test(
    |device| {
      use aethervk_oshal_rlib::math::matrix::SquareMatrix;
      
      let pe_params = crate::gpu::PresentationEngineParams::windowless(800, 600);

      // Create PE
      let pe_handle = device.create_presentation_engine(&pe_params)?;

      // Start frame
      device.start_frame()?;
      
      // Get Pipeline key
      let key = device.get_trajectory_pipeline_key(pe_handle)?;
      assert_ne!(key.0, 0);

      // Upload trajectories
      let traj_comp = TrajectoryComponent {
         color: [1.0, 0.0, 0.0, 1.0],
         line_width: 2.0,
         texture_id: 0,
         subdivisions_per_segment: 32,
         control_points: alloc::vec::Vec::new(),
      };

      // In real scenario we need a command buffer. 
      // But we can check that it doesn't crash on empty.
      let trajectories = &[(crate::scene::EntityId::from_ffi(1), traj_comp, Mat4x4f32::identity())];
      let cmd_buffer = device.get_command_buffer().unwrap();
      device.set_command_buffer_presentation_engine(cmd_buffer, pe_handle).unwrap();
      let _batch = device.upload_trajectories(cmd_buffer, trajectories)?;

      // Cleanup
      device.destroy_presentation_engine(pe_handle)?;
      Ok(())
    }
  );
}
