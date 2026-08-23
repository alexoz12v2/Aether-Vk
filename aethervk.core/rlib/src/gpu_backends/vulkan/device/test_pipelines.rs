//! test_pipelines module.

#[cfg(test)]
mod tests {
  use crate::gpu::PipelineKeyable;
  use crate::gpu_backends::vulkan::device::pipelines::{
    ComputeInfo, FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PipelinePool,
    PreRasterization, StencilCompareOp, StencilLogicOp, VertexIn,
  };
  use crate::gpu_backends::vulkan::device::{LogicalDevice, VulkanDebugNameExt};
  use alloc::vec::Vec;
  use ash::vk;
  use ash::vk::Handle;

  // Borrowed setup function for Vulkan resources
  fn setup_test_render() -> Option<(
    crate::gpu_backends::vulkan::instance::Instance,
    ash::Device,
    crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::PhysicalDevice>,
    LogicalDevice,
  )> {
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
    if !instance.has_headless_surface {
      return None;
    }
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

    let device_create_info = vk::DeviceCreateInfo::default()
      .queue_create_infos(core::slice::from_ref(&queue_info))
      .enabled_extension_names(&extensions)
      .push_next(&mut features2);

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

    let log_device = LogicalDevice {
      submission_lock_compute: spin::Mutex::new(()),
      timeline_semaphore: ash::khr::timeline_semaphore::Device::new(&instance.instance, &device),
      create_renderpass2: ash::khr::create_renderpass2::Device::new(&instance.instance, &device),
      swapchain_maintenance1: None,
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
      max_per_stage_descriptor_update_after_bind_samplers: phys_device
        .max_per_stage_descriptor_update_after_bind_samplers,
      max_per_stage_descriptor_samplers: phys_device
        .physical_device_properties
        .limits
        .max_per_stage_descriptor_samplers,
      max_descriptor_set_update_after_bind_samplers: phys_device
        .max_descriptor_set_update_after_bind_samplers,
    };

    Some((
      instance,
      device,
      unsafe {
        crate::gpu_backends::vulkan::utils::NonZeroHandle::new_unchecked(
          phys_device.physical_device,
        )
      },
      log_device,
    ))
  }

  #[test]
  fn test_compute_info_eq_and_hash() {
    let mut info1 = ComputeInfo::default();
    let mut info2 = ComputeInfo::default();

    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    info1.with_shader_module(vk::ShaderModule::from_raw(1));
    assert!(info1 != info2);
    assert_ne!(info1.pipeline_key(), info2.pipeline_key());

    info2.with_shader_module(vk::ShaderModule::from_raw(1));
    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    let entry = vk::SpecializationMapEntry::default().constant_id(0).size(4);
    info1.add_specialization_constant_u32(entry, 42);
    assert!(info1 != info2);
    assert_ne!(info1.pipeline_key(), info2.pipeline_key());

    info2.add_specialization_constant_u32(entry, 42);
    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    // Different value
    let mut info3 = ComputeInfo::default();
    info3.with_shader_module(vk::ShaderModule::from_raw(1));
    info3.add_specialization_constant_u32(entry, 43);

    assert!(info1 != info3);
    assert_ne!(info1.pipeline_key(), info3.pipeline_key());
  }

  #[test]
  fn test_graphics_info_eq_and_hash() {
    let mut info1 = GraphicsInfo::default();
    let mut info2 = GraphicsInfo::default();

    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    info1 = info1.with_pipeline_flags(PipelineFlags::CULL_ALL);
    assert!(info1 != info2);
    assert_ne!(info1.pipeline_key(), info2.pipeline_key());

    info2 = info2.with_pipeline_flags(PipelineFlags::CULL_ALL);
    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    let vertex_in = VertexIn::default()
      .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
      .add_binding(0, 16, vk::VertexInputRate::VERTEX)
      .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0);

    info1 = info1.with_vertex_in(vertex_in.clone());
    assert!(info1 != info2);
    assert_ne!(info1.pipeline_key(), info2.pipeline_key());

    info2 = info2.with_vertex_in(vertex_in.clone());
    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());

    info1 = info1.with_stencil_compare_op(StencilCompareOp::Equal);
    assert!(info1 != info2);
    assert_ne!(info1.pipeline_key(), info2.pipeline_key());

    info2 = info2.with_stencil_compare_op(StencilCompareOp::Equal);
    assert!(info1 == info2);
    assert_eq!(info1.pipeline_key(), info2.pipeline_key());
  }

  #[test]
  fn test_pipeline_pool_compilation() {
    let setup = setup_test_render();
    if setup.is_none() {
      return; // Skip if no physical device / headless presentation
    }
    let (_instance, device, _phys_device, log_device) = setup.unwrap();

    let pool = PipelinePool::new(&device, None).unwrap();
    let mut rollback = crate::gpu_backends::vulkan::utils::RollbackContext::new(&log_device);

    let spv_data = std::fs::read("comp.spv").unwrap();

    // Ensure properly aligned for u32
    let mut words = vec![0u32; spv_data.len() / 4];
    unsafe {
      std::ptr::copy_nonoverlapping(
        spv_data.as_ptr(),
        words.as_mut_ptr() as *mut u8,
        spv_data.len(),
      );
    }

    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let shader_module = unsafe { device.create_shader_module(&module_info, None).unwrap() };

    let push_constant_range = vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::COMPUTE)
      .offset(0)
      .size(128);
    let layout_infos = [push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&layout_infos);
    let pipeline_layout = unsafe {
      device
        .create_pipeline_layout(&layout_info, None)
        .with_name(&log_device, "VkPipelineLayout_Test")
        .unwrap()
    };

    let mut compute_info = ComputeInfo::default();
    compute_info
      .with_shader_module(shader_module)
      .with_pipeline_layout(pipeline_layout);

    let pipeline1 = pool
      .get_or_create_compute_pipeline(&log_device, &compute_info, &mut rollback)
      .unwrap();
    let pipeline2 = pool
      .get_or_create_compute_pipeline(&log_device, &compute_info, &mut rollback)
      .unwrap();

    // Test that cache returned exactly the same handle
    assert_eq!(pipeline1.get(), pipeline2.get());
    assert_eq!(
      pool.get_compute_pipeline(compute_info.pipeline_key()).unwrap().get(),
      pipeline1.get()
    );

    unsafe {
      device.destroy_pipeline_layout(pipeline_layout, None);
      device.destroy_shader_module(shader_module, None);
    }

    rollback.defuse();

    // Test cleanup
    use crate::gpu_backends::vulkan::device::DeviceResource;
    let mut pool = pool;
    pool.cleanup(&log_device);

    unsafe {
      device.destroy_device(None);
    }
  }
}