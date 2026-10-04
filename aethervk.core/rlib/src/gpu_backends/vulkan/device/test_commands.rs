#[cfg(test)]
use super::*;
#[cfg(test)]
use crate::gpu::{RenderDevice, ScopedCommandBuffer};
#[cfg(test)]
use crate::gpu_backends::vulkan::device::test_utils::{
  setup_assets_dir, setup_render_frontend_for_tests,
};
#[cfg(test)]
use aethervk_oshal_rlib::os::native::this_thread;

#[test]
fn stress_test_command_pool_rotation() {
  setup_assets_dir();
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);

  let res = render_frontend.with_device(render_device_handle, |device| {
    let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
      device.as_any().downcast_ref().unwrap();

    // Submit 100 commands (far exceeding MAX_BUFFERS_PER_POOL of 32)
    // This will force the CommandPool allocator to rotate `active` to `pending`
    // multiple times, and allocate brand new pools.
    for _ in 0..100 {
      let task_id = device.create_task();

      // Get compute queue command buffer
      let (cmd_buffer_handle, _cmd_native) = vulkan_device.get_command_buffer_and_native()?;
      let cmd_scope = ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      // We could record a `cmd_copy_buffer` here, but simply opening, closing, and
      // submitting the command buffer is enough to put it in flight and stress the pool allocation ring.
      cmd_scope.submit()?;

      // Simulate the LogicThread doing a GPU timeline poll to recycle discarded command buffers.
      // This prevents exhaustion of the 24 max pending pools limit.
      let gpu_timeline_val = unsafe {
        vulkan_device
          .device
          .timeline_semaphore
          .get_semaphore_counter_value(vulkan_device.kernels.timeline)
      }
      .unwrap_or(0);

      let items = vulkan_device.kernels.discard_pool.pop_ready_items(gpu_timeline_val);
      if !items.is_empty() {
        crate::gpu_backends::vulkan::device::DiscardPool::destroy_items_lock_free(
          &vulkan_device.device,
          items,
        );
      }
    }

    // Final synchronization to wait for remaining GPU tasks
    let final_task_id = device.create_task();
    let (cmd, _) = vulkan_device.get_command_buffer_and_native()?;
    let scope = ScopedCommandBuffer::new(device, cmd, Some(final_task_id))?;
    scope.submit()?;

    // Wait for device to idle
    unsafe { vulkan_device.device.device_wait_idle() }.unwrap();

    crate::types::GpuResult::Ok(())
  });

  assert!(res.is_ok(), "Command pool stress test failed");
}
