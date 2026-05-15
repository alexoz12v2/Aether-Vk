pub mod cpu_clustering;

use aethervk_core_rlib::gpu::{
  DeviceAdditionalParams, RenderFrontend, VULKAN_RENDER_BACKEND, new_render_frontend,
};
use aethervk_core_rlib::gpu_backends::vulkan::device::Device;
use aethervk_core_rlib::types::RuntimeParams;
use ash::vk;
use heapless::index_map::FnvIndexMap;
use std::fs;
use vk_mem::Alloc;

fn panic_on_validation_error(msg: &str) {
  panic!("Vulkan validation error: {}", msg);
}

pub struct VulkanContext {
  pub frontend: RenderFrontend,
  pub device_handle: aethervk_core_rlib::gpu::RenderDeviceHandle,
  // Keep pool alive
  _pool: std::sync::Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
}

impl Default for VulkanContext {
  fn default() -> Self {
    Self::new()
  }
}

impl VulkanContext {
  pub fn new() -> Self {
    let runtime_params = Box::new(RuntimeParams {
      render_backend_params: FnvIndexMap::new(),
      validation_error_callback: Some(panic_on_validation_error as fn(&str)),
    });

    let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
    let pool_arc = std::sync::Arc::new(pool);

    let frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();
    let mut additional_params = DeviceAdditionalParams::new();
    additional_params
      .insert(
        aethervk_core_rlib::gpu_backends::vulkan::DEVICE_ADDIDITIONAL_PARAM_DEBUG_SHADERS,
        1,
      )
      .unwrap();
    let device_handle = frontend.write().init_device(0, &additional_params).unwrap();

    frontend
      .with_device(device_handle, |device| {
        device.wire_callbacks(pool_arc.clone())
      })
      .unwrap();

    Self {
      frontend,
      device_handle,
      _pool: pool_arc,
    }
  }

  pub fn create_buffer<T>(
    &self,
    data: &[T],
    usage: vk::BufferUsageFlags,
  ) -> (vk::Buffer, vk_mem::Allocation, u64) {
    self
      .frontend
      .with_device(self.device_handle, |dev| {
        let actual_device = dev.as_any().downcast_ref::<Device>().unwrap();
        let size = (std::mem::size_of::<T>() * data.len()) as u64;
        let buffer_info = vk::BufferCreateInfo::default()
          .size(size)
          .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        let alloc_info = vk_mem::AllocationCreateInfo {
          usage: vk_mem::MemoryUsage::AutoPreferDevice,
          flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            | vk_mem::AllocationCreateFlags::MAPPED,
          required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
          ..Default::default()
        };

        let allocator = &*actual_device.res.read().allocator.allocator;

        let (buffer, alloc, info) =
          unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }.unwrap();

        if size > 0 {
          unsafe {
            std::ptr::copy_nonoverlapping(
              data.as_ptr() as *const u8,
              info.mapped_data as *mut u8,
              size as usize,
            );
          }
        }

        let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe {
          actual_device.device.buffer_device_address.get_buffer_device_address(&device_address_info)
        };

        aethervk_core_rlib::types::GpuResult::Ok((buffer, alloc, address))
      })
      .unwrap()
  }

  pub fn destroy_buffer(&self, buffer: vk::Buffer, mut alloc: vk_mem::Allocation) {
    self
      .frontend
      .with_device(self.device_handle, |dev| {
        let actual_device = dev.as_any().downcast_ref::<Device>().unwrap();
        let allocator = &*actual_device.res.read().allocator.allocator;
        unsafe {
          allocator.destroy_buffer(buffer, &mut alloc);
        }
        aethervk_core_rlib::types::GpuResult::Ok(())
      })
      .unwrap();
  }

  pub fn read_buffer<T: Copy>(
    &self,
    _buffer: vk::Buffer,
    alloc: &mut vk_mem::Allocation,
    count: usize,
  ) -> Vec<T> {
    self
      .frontend
      .with_device(self.device_handle, |dev| {
        let actual_device = dev.as_any().downcast_ref::<Device>().unwrap();
        let allocator = &*actual_device.res.read().allocator.allocator;
        let info = allocator.get_allocation_info(alloc);

        let mut result = Vec::with_capacity(count);
        if count > 0 {
          unsafe {
            let src_ptr = info.mapped_data as *const T;
            for i in 0..count {
              result.push(*src_ptr.add(i));
            }
          }
        }
        aethervk_core_rlib::types::GpuResult::Ok(result)
      })
      .unwrap()
  }
}

pub fn run_compute_shader(
  ctx: &VulkanContext,
  spv_path: &str,
  push_constants: &[u8],
  dispatch_x: u32,
  dispatch_y: u32,
  dispatch_z: u32,
) {
  ctx
    .frontend
    .with_device(ctx.device_handle, |dev| {
      let actual_device = dev.as_any().downcast_ref::<Device>().unwrap();
      let ash_dev = &actual_device.device;

      let spv_code = fs::read(spv_path).unwrap();
      let (prefix, code, suffix) = unsafe { spv_code.align_to::<u32>() };
      assert!(prefix.is_empty() && suffix.is_empty());

      let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
      let shader_module = unsafe { ash_dev.create_shader_module(&shader_info, None) }.unwrap();

      let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(push_constants.len() as u32);

      let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

      let pipeline_layout = unsafe { ash_dev.create_pipeline_layout(&layout_info, None) }.unwrap();

      let main_name = std::ffi::CString::new("main").unwrap();

      let mut spec_map_entries = vec![];
      let mut spec_data = vec![];
      let sg_size = 32u32;
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: 4,
      });
      spec_data.extend_from_slice(&sg_size.to_le_bytes());

      let debug_shaders = 1u32;
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 10,
        offset: 4,
        size: 4,
      });
      spec_data.extend_from_slice(&debug_shaders.to_le_bytes());

      let spec_info =
        vk::SpecializationInfo::default().map_entries(&spec_map_entries).data(&spec_data);

      let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&main_name)
        .specialization_info(&spec_info);

      let compute_info =
        vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);

      let pipelines = unsafe {
        ash_dev.create_compute_pipelines(
          vk::PipelineCache::null(),
          std::slice::from_ref(&compute_info),
          None,
        )
      }
      .unwrap();
      let pipeline = pipelines[0];

      let queue_family_index = actual_device.query_result.compute_queue_family_index;
      let queue = unsafe { ash_dev.get_device_queue(queue_family_index, 0) };

      let pool_create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
      let command_pool = unsafe { ash_dev.create_command_pool(&pool_create_info, None) }.unwrap();

      let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

      let cmd_buffer = unsafe { ash_dev.allocate_command_buffers(&alloc_info) }.unwrap()[0];

      unsafe {
        ash_dev.begin_command_buffer(cmd_buffer, &vk::CommandBufferBeginInfo::default()).unwrap();
        ash_dev.cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        ash_dev.cmd_push_constants(
          cmd_buffer,
          pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          push_constants,
        );
        ash_dev.cmd_dispatch(cmd_buffer, dispatch_x, dispatch_y, dispatch_z);

        let memory_barrier = vk::MemoryBarrier2::default()
          .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
          .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
          .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
          .dst_access_mask(vk::AccessFlags2::SHADER_READ);

        let dependency_info =
          vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        actual_device.device.synchronization2.cmd_pipeline_barrier2(cmd_buffer, &dependency_info);

        ash_dev.end_command_buffer(cmd_buffer).unwrap();

        let submit_info =
          vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buffer));
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = ash_dev.create_fence(&fence_create_info, None).unwrap();

        actual_device
          .device
          .locked_queue_submit(queue, std::slice::from_ref(&submit_info), fence)
          .unwrap();
        ash_dev.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX).unwrap();

        ash_dev.destroy_fence(fence, None);
        ash_dev.free_command_buffers(command_pool, std::slice::from_ref(&cmd_buffer));
        ash_dev.destroy_command_pool(command_pool, None);
        ash_dev.destroy_pipeline(pipeline, None);
        ash_dev.destroy_pipeline_layout(pipeline_layout, None);
        ash_dev.destroy_shader_module(shader_module, None);
      }

      aethervk_core_rlib::types::GpuResult::Ok(())
    })
    .unwrap()
}

pub fn ensure_test_data(json_path: &str, python_script: &str) {
  if !std::path::Path::new(json_path).exists() {
    println!("Generating test data using {}", python_script);
    let status = std::process::Command::new("python")
      .arg(python_script)
      .arg("--out")
      .arg(json_path)
      .status()
      .expect("Failed to execute python script to generate test data");
    assert!(
      status.success(),
      "Python script failed with status: {}",
      status
    );
  }
}
