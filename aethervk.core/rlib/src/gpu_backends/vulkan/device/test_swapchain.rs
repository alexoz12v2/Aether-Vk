//! test_swapchain module.

#[cfg(test)]
mod tests {
  extern crate std;
  use crate::gpu_backends::vulkan::device::{swapchain::*, *};
  use alloc::{boxed::Box, vec::Vec};
  use ash::vk;

  /// Creates a shared main-thread cleanup queue for test PEs.
  fn new_test_cleanup_queue() -> crate::gpu::MainThreadCleanupQueue {
    alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new()))
  }

  /// Drains the test cleanup queue, executing all deferred swapchain/surface
  /// destruction tasks. MUST be called after PE cleanup and before destroy_device.
  fn drain_test_cleanup_queue(queue: &crate::gpu::MainThreadCleanupQueue) {
    let mut q = queue.lock();
    let tasks: Vec<_> = q.drain(..).collect();
    drop(q);
    for task in tasks {
      task();
    }
  }

  // Helper to get real Vulkan device objects for testing swapchain directly
  fn setup_test_render(
    enable_maintenance1: bool,
  ) -> (
    alloc::sync::Arc<ash::Entry>,
    crate::gpu_backends::vulkan::instance::Instance,
    ash::Device,
    crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::PhysicalDevice>,
    LogicalDevice,
    vk::Queue,
    vk::CommandPool,
    crate::gpu::PresentationEngineParams,
  ) {
    use crate::gpu_backends::vulkan::instance::Instance;
    crate::gpu::set_asset_dir_for_tests();

    fn panic_on_validation_error(msg: &str) {
      panic!("Vulkan validation error occurred during testing: {}", msg);
    }

    let instance = unsafe {
      crate::gpu_backends::vulkan::instance::Instance::new(
        None,
        Some(panic_on_validation_error as fn(&str)),
      )
      .unwrap()
    };
    let entry = instance.entry_wrapper.weak_entry().upgrade().unwrap();
    let query_input = crate::gpu_backends::vulkan::utils::PhysicalDeviceQueryInput::from_params(
      &crate::gpu::DeviceAdditionalParams::new(),
    )
    .unwrap();
    let eligible = instance.get_eligible_devices(&query_input).unwrap();
    let phys_device = eligible.into_iter().next().unwrap();

    let mut required_features = crate::gpu_backends::vulkan::utils::RequiredFeatures::new();
    required_features.populate();
    let mut features2 = required_features.as_features2();
    let priorities = [1.0];
    let queue_info = vk::DeviceQueueCreateInfo::default()
      .queue_family_index(phys_device.graphics_queue_family_index as u32)
      .queue_priorities(&priorities);
    let extensions = phys_device.enabled_extension_names();

    let mut swapchain_maintenance1_features =
      vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default().swapchain_maintenance1(true);

    let mut device_create_info = vk::DeviceCreateInfo::default()
      .queue_create_infos(core::slice::from_ref(&queue_info))
      .enabled_extension_names(&extensions)
      .push_next(&mut features2);

    if enable_maintenance1
      && phys_device.optional_extensions.contains(
        crate::gpu_backends::vulkan::utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1,
      )
    {
      device_create_info = device_create_info.push_next(&mut swapchain_maintenance1_features);
    }

    #[cfg(any(debug_assertions, test))]
    let device = unsafe {
      crate::gpu_backends::vulkan::device::hooks::load_device_with_hooks(
        &instance.instance,
        phys_device.physical_device,
        &device_create_info,
      )
      .unwrap()
    };
    #[cfg(not(any(debug_assertions, test)))]
    let device = unsafe {
      instance
        .instance
        .create_device(phys_device.physical_device, &device_create_info, None)
        .unwrap()
    };
    let queue =
      unsafe { device.get_device_queue(phys_device.graphics_queue_family_index as u32, 0) };
    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(phys_device.graphics_queue_family_index as u32)
      .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None).unwrap() };

    let log_device = LogicalDevice {
      timeline_semaphore: ash::khr::timeline_semaphore::Device::new(&instance.instance, &device),
      create_renderpass2: ash::khr::create_renderpass2::Device::new(&instance.instance, &device),
      swapchain_maintenance1: if enable_maintenance1
        && phys_device.optional_extensions.contains(
          crate::gpu_backends::vulkan::utils::OptionalExtensionSupportFlags::SWAPCHAIN_MAINTENANCE1,
        ) {
        Some(ash::ext::swapchain_maintenance1::Device::new(
          &instance.instance,
          &device,
        ))
      } else {
        None
      },
      buffer_device_address: ash::khr::buffer_device_address::Device::new(
        &instance.instance,
        &device,
      ),
      synchronization2: ash::khr::synchronization2::Device::new(&instance.instance, &device),
      handle: device.clone(),
      submission_lock: spin::Mutex::new(()),
      #[cfg(target_vendor = "apple")]
      metal_objects: ash::ext::metal_objects::Device::new(&instance.instance, &device),
      #[cfg(debug_assertions)]
      debug_utils: ash::ext::debug_utils::Device::new(&instance.instance, &device),
      max_per_stage_descriptor_update_after_bind_samplers: phys_device.max_per_stage_descriptor_update_after_bind_samplers,
      max_per_stage_descriptor_samplers: phys_device.physical_device_properties.limits.max_per_stage_descriptor_samplers,
      max_descriptor_set_update_after_bind_samplers: phys_device.max_descriptor_set_update_after_bind_samplers,
    };

    let params = crate::gpu::PresentationEngineParams::windowless(256, 256);

    (
      entry,
      instance,
      device,
      unsafe {
        crate::gpu_backends::vulkan::utils::NonZeroHandle::new_unchecked(
          phys_device.physical_device,
        )
      },
      log_device,
      queue,
      cmd_pool,
      params,
    )
  }

  unsafe fn simulate_gpu_frame(
    device: &ash::Device,
    logical_device: &LogicalDevice,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    image: vk::Image,
    acquire_sem: Option<crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Semaphore>>,
    present_sem: Option<crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Semaphore>>,
    submit_fence: Option<crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Fence>>,
    clear_color: [f32; 4],
  ) {
    unsafe {
      let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      let cmd = device.allocate_command_buffers(&alloc_info).unwrap()[0];

      device
        .begin_command_buffer(
          cmd,
          &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .unwrap();

      let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

      let barrier1 = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(image)
        .subresource_range(subresource_range);

      device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        core::slice::from_ref(&barrier1),
      );

      let clear_value = vk::ClearColorValue {
        float32: clear_color,
      };
      device.cmd_clear_color_image(
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &clear_value,
        core::slice::from_ref(&subresource_range),
      );

      let final_layout = if present_sem.is_some() {
        vk::ImageLayout::PRESENT_SRC_KHR
      } else {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
      };
      let barrier2 = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(final_layout)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(if present_sem.is_some() {
          vk::AccessFlags::empty()
        } else {
          vk::AccessFlags::TRANSFER_READ
        })
        .image(image)
        .subresource_range(subresource_range);

      let dst_stage = if present_sem.is_some() {
        vk::PipelineStageFlags::BOTTOM_OF_PIPE
      } else {
        vk::PipelineStageFlags::TRANSFER
      };
      device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        core::slice::from_ref(&barrier2),
      );

      device.end_command_buffer(cmd).unwrap();

      let wait_sems: Vec<vk::Semaphore> = acquire_sem.map(|s| s.get()).into_iter().collect();
      let sig_sems: Vec<vk::Semaphore> = present_sem.map(|s| s.get()).into_iter().collect();
      let wait_dst_stage = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; wait_sems.len()];

      let mut submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd));
      if !wait_sems.is_empty() {
        submit_info = submit_info.wait_semaphores(&wait_sems).wait_dst_stage_mask(&wait_dst_stage);
      }
      if !sig_sems.is_empty() {
        submit_info = submit_info.signal_semaphores(&sig_sems);
      }

      logical_device
        .locked_queue_submit(
          queue,
          core::slice::from_ref(&submit_info),
          submit_fence.map(|f| f.get()).unwrap_or(vk::Fence::null()),
        )
        .unwrap();
    }
  }

  unsafe fn download_windowless_to_png(
    instance: &ash::Instance,
    device: &ash::Device,
    logical_device: &LogicalDevice,
    phys_device: vk::PhysicalDevice,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    image: vk::Image,
    width: u32,
    height: u32,
    filename: &str,
  ) {
    unsafe {
      let size = (width * height * 4) as u64;

      let buf_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);
      let buffer = device.create_buffer(&buf_info, None).unwrap();
      let reqs = device.get_buffer_memory_requirements(buffer);

      let mem_props = instance.get_physical_device_memory_properties(phys_device);
      let mem_type_index = mem_props
        .memory_types
        .iter()
        .enumerate()
        .find(|(i, ty)| {
          (reqs.memory_type_bits & (1 << i)) != 0
            && ty.property_flags.contains(
              vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        })
        .unwrap()
        .0 as u32;

      let memory = device
        .allocate_memory(
          &vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type_index),
          None,
        )
        .unwrap();
      device.bind_buffer_memory(buffer, memory, 0).unwrap();

      let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      let cmd = device.allocate_command_buffers(&alloc_info).unwrap()[0];
      device
        .begin_command_buffer(
          cmd,
          &vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .unwrap();

      let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
          aspect_mask: vk::ImageAspectFlags::COLOR,
          mip_level: 0,
          base_array_layer: 0,
          layer_count: 1,
        })
        .image_extent(vk::Extent3D {
          width,
          height,
          depth: 1,
        });

      device.cmd_copy_image_to_buffer(
        cmd,
        image,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        buffer,
        core::slice::from_ref(&region),
      );
      device.end_command_buffer(cmd).unwrap();

      let submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd));
      let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap();

      logical_device
        .locked_queue_submit(queue, core::slice::from_ref(&submit_info), fence)
        .unwrap();
      device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX).unwrap();

      let data = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()).unwrap();
      let pixel_slice = std::slice::from_raw_parts(data as *const u8, size as usize);

      let mut rgba = vec![0u8; size as usize];
      for i in (0..rgba.len()).step_by(4) {
        rgba[i] = pixel_slice[i + 2]; // R
        rgba[i + 1] = pixel_slice[i + 1]; // G
        rgba[i + 2] = pixel_slice[i]; // B
        rgba[i + 3] = pixel_slice[i + 3]; // A
      }

      image::save_buffer(filename, &rgba, width, height, image::ColorType::Rgba8).unwrap();

      device.unmap_memory(memory);
      device.destroy_fence(fence, None);
      device.free_command_buffers(cmd_pool, core::slice::from_ref(&cmd));
      device.destroy_buffer(buffer, None);
      device.free_memory(memory, None);
    }
  }

  fn test_lifecycle_wraparound_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, params) =
      setup_test_render(enable_maintenance1);
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);
    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new())),
    )
    .unwrap();

    for _ in 0..20 {
      let acq = engine
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = engine.get_image_resources(acq.image_index as usize);
        let (acquire_sem, submit_fence) = engine.get_frame_resources(acq.frame_index as usize);

        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [0.0, 1.0, 0.0, 1.0],
        );
        let _ = engine
          .submit_image(&log_device, queue, acq.image_index, acq.frame_index as u32)
          .unwrap();
      }
    }
    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    engine.cleanup(&device);
    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_lifecycle_wraparound_legacy() {
    test_lifecycle_wraparound_internal(false);
  }

  #[test]
  fn test_lifecycle_wraparound_modern() {
    test_lifecycle_wraparound_internal(true);
  }

  fn test_cancel_image_recovery_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, params) =
      setup_test_render(enable_maintenance1);
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);
    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new())),
    )
    .unwrap();

    println!("Acquiring acq1...");
    let acq1 = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();

    println!("Canceling image...");
    engine
      .cancel_image(
        &log_device,
        queue,
        acq1.image_index,
        acq1.frame_index as u32,
      )
      .unwrap();

    println!("Acquiring acq2...");
    let acq2 = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    assert_ne!(acq1.frame_index, acq2.frame_index);

    println!("Simulating frame...");
    unsafe {
      let (image, _, present_sem) = engine.get_image_resources(acq2.image_index as usize);
      let (acquire_sem, submit_fence) = engine.get_frame_resources(acq2.frame_index as usize);
      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        present_sem,
        submit_fence,
        [1.0, 0.0, 0.0, 1.0],
      );
      println!("Submitting image...");
      let _ = engine
        .submit_image(
          &log_device,
          queue,
          acq2.image_index,
          acq2.frame_index as u32,
        )
        .unwrap();
    }

    println!("Queue wait idle...");
    unsafe { device.queue_wait_idle(queue).unwrap() };
    println!("Cleaning up...");
    rollback.defuse();
    engine.cleanup(&device);
    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_cancel_image_recovery_legacy() {
    test_cancel_image_recovery_internal(false);
  }

  #[test]
  fn test_cancel_image_recovery_modern() {
    test_cancel_image_recovery_internal(true);
  }

  fn test_resize_in_flight_discard_bins_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, mut params) =
      setup_test_render(enable_maintenance1);
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);
    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new())),
    )
    .unwrap();

    let acq = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    unsafe {
      let (image, _, present_sem) = engine.get_image_resources(acq.image_index as usize);
      let (acquire_sem, submit_fence) = engine.get_frame_resources(acq.frame_index as usize);

      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        present_sem,
        submit_fence,
        [1.0, 0.0, 0.0, 1.0],
      );
      let _ = engine
        .submit_image(&log_device, queue, acq.image_index, acq.frame_index as u32)
        .unwrap();
    }

    engine
      .resize(
        &instance.instance,
        &log_device,
        phys_device,
        params.width + 100,
        params.height + 100,
        &mut rollback,
      )
      .unwrap();

    let acq2 = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    unsafe {
      let (image, _, present_sem) = engine.get_image_resources(acq2.image_index as usize);
      let (acquire_sem, submit_fence) = engine.get_frame_resources(acq2.frame_index as usize);
      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        present_sem,
        submit_fence,
        [0.0, 0.0, 1.0, 1.0],
      );
      let _ = engine
        .submit_image(
          &log_device,
          queue,
          acq2.image_index,
          acq2.frame_index as u32,
        )
        .unwrap();
    }

    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    engine.cleanup(&device);
    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_resize_in_flight_discard_bins_legacy() {
    test_resize_in_flight_discard_bins_internal(false);
  }

  #[test]
  fn test_resize_in_flight_discard_bins_modern() {
    test_resize_in_flight_discard_bins_internal(true);
  }

  fn test_windowless_export_png_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, mut params) =
      setup_test_render(enable_maintenance1);
    params.ty = crate::gpu::PresentationEngineType::WindowLess;
    params.width = 128;
    params.height = 128;

    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);

    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      alloc::sync::Arc::new(spin::Mutex::new(alloc::vec::Vec::new())),
    )
    .unwrap();

    let acq = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    unsafe {
      let (image, _, _) = engine.get_image_resources(acq.image_index as usize);
      let (acquire_sem, submit_fence) = engine.get_frame_resources(acq.frame_index as usize);

      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        None,
        submit_fence,
        [0.0, 0.0, 1.0, 1.0],
      );
      let _ = engine
        .submit_image(&log_device, queue, acq.image_index, acq.frame_index as u32)
        .unwrap();
    }

    if let PresentationState::Windowless(ref state) = engine {
      let last_image = state.get_last_submitted_image().unwrap();
      unsafe {
        download_windowless_to_png(
          &instance.instance,
          &device,
          &log_device,
          phys_device.get(),
          queue,
          cmd_pool,
          last_image.get(),
          params.width,
          params.height,
          "test_output_windowless.png",
        );
      }
    } else {
      panic!("Expected Windowless state");
    }

    let path = std::path::Path::new("test_output_windowless.png");
    assert!(path.exists());

    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    engine.cleanup(&device);
    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_windowless_export_png_legacy() {
    test_windowless_export_png_internal(false);
  }

  #[test]
  fn test_windowless_export_png_modern() {
    test_windowless_export_png_internal(true);
  }

  fn test_windowed_presentation_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, _) =
      setup_test_render(enable_maintenance1);
    let cleanup_queue = new_test_cleanup_queue();

    let params = crate::gpu::PresentationEngineParams {
      ty: crate::gpu::PresentationEngineType::Window,
      window_info: crate::gpu::OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
      width: 800,
      height: 600,
      vsync: false,
      buffer_count: 3,
    };

    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);

    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      cleanup_queue.clone(),
    )
    .unwrap();

    for _ in 0..20 {
      let acq = engine
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = engine.get_image_resources(acq.image_index as usize);
        let (acquire_sem, submit_fence) = engine.get_frame_resources(acq.frame_index as usize);

        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [0.0, 1.0, 0.0, 1.0],
        );
        let _ = engine
          .submit_image(&log_device, queue, acq.image_index, acq.frame_index as u32)
          .unwrap();
      }
    }

    engine
      .resize(
        &instance.instance,
        &log_device,
        phys_device,
        1024,
        768,
        &mut rollback,
      )
      .unwrap();

    let acq2 = engine
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    unsafe {
      let (image, _, present_sem) = engine.get_image_resources(acq2.image_index as usize);
      let (acquire_sem, submit_fence) = engine.get_frame_resources(acq2.frame_index as usize);

      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        present_sem,
        submit_fence,
        [0.0, 0.0, 1.0, 1.0],
      );
      let _ = engine
        .submit_image(
          &log_device,
          queue,
          acq2.image_index,
          acq2.frame_index as u32,
        )
        .unwrap();
    }

    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    engine.cleanup(&device);
    drain_test_cleanup_queue(&cleanup_queue);
    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_windowed_presentation_legacy() {
    test_windowed_presentation_internal(false);
  }

  #[test]
  fn test_windowed_presentation_modern() {
    test_windowed_presentation_internal(true);
  }

  fn test_multiple_viewports_and_mesh_viewer_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, _) =
      setup_test_render(enable_maintenance1);
    let cleanup_queue = new_test_cleanup_queue();

    let params1 = crate::gpu::PresentationEngineParams {
      ty: crate::gpu::PresentationEngineType::Window,
      window_info: crate::gpu::OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
      width: 800,
      height: 600,
      vsync: false,
      buffer_count: 3,
    };
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);
    let mut vp1 = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params1,
      &mut rollback,
      cleanup_queue.clone(),
    )
    .unwrap();

    let params2 = crate::gpu::PresentationEngineParams {
      ty: crate::gpu::PresentationEngineType::Window,
      window_info: crate::gpu::OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
      width: 800,
      height: 600,
      vsync: false,
      buffer_count: 3,
    };
    let mut vp2 = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params2,
      &mut rollback,
      cleanup_queue.clone(),
    )
    .unwrap();

    let params3 = crate::gpu::PresentationEngineParams {
      ty: crate::gpu::PresentationEngineType::Window,
      window_info: crate::gpu::OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
      width: 400,
      height: 400,
      vsync: false,
      buffer_count: 3,
    };
    let mut mv = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params3,
      &mut rollback,
      cleanup_queue.clone(),
    )
    .unwrap();

    for _ in 0..5 {
      // VP1
      let acq1 = vp1
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = vp1.get_image_resources(acq1.image_index as usize);
        let (acquire_sem, submit_fence) = vp1.get_frame_resources(acq1.frame_index as usize);
        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [1.0, 0.0, 0.0, 1.0],
        );
        vp1
          .submit_image(
            &log_device,
            queue,
            acq1.image_index,
            acq1.frame_index as u32,
          )
          .unwrap();
      }

      // VP2
      let acq2 = vp2
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = vp2.get_image_resources(acq2.image_index as usize);
        let (acquire_sem, submit_fence) = vp2.get_frame_resources(acq2.frame_index as usize);
        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [0.0, 1.0, 0.0, 1.0],
        );
        vp2
          .submit_image(
            &log_device,
            queue,
            acq2.image_index,
            acq2.frame_index as u32,
          )
          .unwrap();
      }

      // MV
      let acq3 = mv
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = mv.get_image_resources(acq3.image_index as usize);
        let (acquire_sem, submit_fence) = mv.get_frame_resources(acq3.frame_index as usize);
        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [0.0, 0.0, 1.0, 1.0],
        );
        mv.submit_image(
          &log_device,
          queue,
          acq3.image_index,
          acq3.frame_index as u32,
        )
        .unwrap();
      }
    }

    // Concurrent resize
    vp1
      .resize(
        &instance.instance,
        &log_device,
        phys_device,
        1024,
        768,
        &mut rollback,
      )
      .unwrap();
    vp2
      .resize(
        &instance.instance,
        &log_device,
        phys_device,
        640,
        480,
        &mut rollback,
      )
      .unwrap();
    mv.resize(
      &instance.instance,
      &log_device,
      phys_device,
      800,
      600,
      &mut rollback,
    )
    .unwrap();

    // Render after resize
    let acq1 = vp1
      .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
      .unwrap();
    unsafe {
      let (image, _, present_sem) = vp1.get_image_resources(acq1.image_index as usize);
      let (acquire_sem, submit_fence) = vp1.get_frame_resources(acq1.frame_index as usize);
      simulate_gpu_frame(
        &device,
        &log_device,
        queue,
        cmd_pool,
        image.get(),
        acquire_sem,
        present_sem,
        submit_fence,
        [1.0, 1.0, 1.0, 1.0],
      );
      vp1
        .submit_image(
          &log_device,
          queue,
          acq1.image_index,
          acq1.frame_index as u32,
        )
        .unwrap();
    }

    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    vp1.cleanup(&device);
    vp2.cleanup(&device);
    mv.cleanup(&device);
    drain_test_cleanup_queue(&cleanup_queue);

    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_multiple_viewports_and_mesh_viewer_legacy() {
    test_multiple_viewports_and_mesh_viewer_internal(false);
  }

  #[test]
  fn test_multiple_viewports_and_mesh_viewer_modern() {
    test_multiple_viewports_and_mesh_viewer_internal(true);
  }

  fn test_rapid_resize_stress_internal(enable_maintenance1: bool) {
    let (entry, instance, device, phys_device, log_device, queue, cmd_pool, _) =
      setup_test_render(enable_maintenance1);
    let cleanup_queue = new_test_cleanup_queue();

    let mut params = crate::gpu::PresentationEngineParams {
      ty: crate::gpu::PresentationEngineType::Window,
      window_info: crate::gpu::OpaqueNativeHandleInfo {
        ptr0: core::ptr::null_mut(),
        ptr1: core::ptr::null_mut(),
      },
      width: 800,
      height: 600,
      vsync: false,
      buffer_count: 3,
    };

    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);

    let mut engine = PresentationState::new(
      &entry,
      &instance.instance,
      &log_device,
      phys_device,
      log_device.swapchain_maintenance1.clone(),
      &params,
      &mut rollback,
      cleanup_queue.clone(),
    )
    .unwrap();

    let mut width = 800;
    let mut height = 600;

    for i in 0..50 {
      if i % 3 == 0 {
        width += 10;
        height += 10;
        engine
          .resize(
            &instance.instance,
            &log_device,
            phys_device,
            width,
            height,
            &mut rollback,
          )
          .unwrap();
      }

      let acq = engine
        .acquire_next_image(&log_device, vk::Semaphore::null(), &mut rollback)
        .unwrap();
      unsafe {
        let (image, _, present_sem) = engine.get_image_resources(acq.image_index as usize);
        let (acquire_sem, submit_fence) = engine.get_frame_resources(acq.frame_index as usize);

        simulate_gpu_frame(
          &device,
          &log_device,
          queue,
          cmd_pool,
          image.get(),
          acquire_sem,
          present_sem,
          submit_fence,
          [0.0, 1.0, 0.0, 1.0],
        );
        let _ = engine
          .submit_image(&log_device, queue, acq.image_index, acq.frame_index as u32)
          .unwrap();
      }
    }

    unsafe { device.queue_wait_idle(queue).unwrap() };
    rollback.defuse();
    engine.cleanup(&device);
    drain_test_cleanup_queue(&cleanup_queue);

    unsafe {
      device.destroy_command_pool(cmd_pool, None);
      device.destroy_device(None);
    }
  }

  #[test]
  fn test_rapid_resize_stress_legacy() {
    test_rapid_resize_stress_internal(false);
  }

  #[test]
  fn test_rapid_resize_stress_modern() {
    test_rapid_resize_stress_internal(true);
  }
}
