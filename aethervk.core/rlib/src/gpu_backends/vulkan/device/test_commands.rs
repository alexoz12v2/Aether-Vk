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

/// `AETHERVK_SINGLE_QUEUE=1` maps every role to the graphics queue and routes compute submissions
/// through the graphics submission lock.
#[test]
fn single_queue_mode_maps_every_role_to_the_graphics_queue() {
  setup_assets_dir();
  unsafe { std::env::set_var("AETHERVK_SINGLE_QUEUE", "1") };
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);
  render_frontend
    .with_device(render_device_handle, |device| {
      let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
        device.as_any().downcast_ref().unwrap();
      let g = vulkan_device.get_graphics_queue();
      let c = vulkan_device.get_compute_queue();
      assert_eq!(g.handle, c.handle, "compute must be the graphics queue");
      assert_eq!(g.family_index, c.family_index);
      assert!(vulkan_device.device.single_queue);
      assert!(core::ptr::eq(
        vulkan_device.device.compute_submission_lock(),
        &vulkan_device.device.submission_lock
      ));
      // a compute submission still works on the shared queue
      let (handle, _) = vulkan_device.get_compute_command_buffer_and_native()?;
      vulkan_device.begin_command_buffer_all(handle, QueueRole::Compute)?;
      let (sem, value) =
        vulkan_device.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Compute)?;
      vulkan_device.device.wait_for_semaphore_value(sem, value, 2_000_000_000)?;
      unsafe { vulkan_device.device.device_wait_idle() }.unwrap();
      crate::types::GpuResult::Ok(())
    })
    .unwrap();
  unsafe { std::env::remove_var("AETHERVK_SINGLE_QUEUE") };
}

/// The fault report names the buffer a faulting address falls in.
#[test]
fn fault_address_map_names_registered_buffers() {
  use crate::gpu_backends::vulkan::device::fault::{
    describe_address, register_address_range, unregister_address_range,
  };
  register_address_range("TestBufA", 0x1000_0000, 0x1000);
  register_address_range("TestBufB", 0x2000_0000, 0x100);
  assert_eq!(
    describe_address(0x1000_0230),
    "TestBufA + 0x230 (size 0x1000)"
  );
  assert_eq!(
    describe_address(0x2000_0100 + 0x40),
    "0x40 past the end of TestBufB"
  );
  assert_eq!(describe_address(0x9000_0000_0000), "no registered buffer");
  unregister_address_range("TestBufA");
  unregister_address_range("TestBufB");
}
