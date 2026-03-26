use core::{mem, ptr};
use alloc::{boxed::Box};
use ash::vk;

use crate::{
  types::{GpuResult},
  gpu_backends::{ vulkan::device::DeviceResource },
};

pub(super) struct GlobalDeviceAllocator {
  pub allocator: mem::ManuallyDrop<vk_mem::Allocator>,
  memory_budgets: Box<[vk_mem::ffi::VmaBudget]>,
}

#[cfg(debug_assertions)]
#[allow(unused)]
unsafe extern "C" fn on_device_alloc(
  allocator: vk_mem::ffi::VmaAllocator,
  memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  p_user_data: *mut core::ffi::c_void,
) {
  // TODO logging
  todo!()
}
#[cfg(debug_assertions)]
#[allow(unused)]
unsafe extern "C" fn on_device_free(
  allocator: vk_mem::ffi::VmaAllocator,
  memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  p_user_data: *mut core::ffi::c_void,
) {
  // TODO logging
  todo!()
}

impl GlobalDeviceAllocator {
  // safety: expects instance and device to have their function pointers already loaded
  pub unsafe fn new(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    api_version: u32,
  ) -> GpuResult<Self> {
    let mut allocator_create_info =
      vk_mem::AllocatorCreateInfo::new(instance, device, physical_device);
    allocator_create_info.vulkan_api_version = api_version;
    allocator_create_info.flags = vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET
      | vk_mem::AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION;
    #[cfg(debug_assertions)]
    let callbacks = vk_mem::ffi::VmaDeviceMemoryCallbacks {
      pfnAllocate: Some(on_device_alloc),
      pfnFree: Some(on_device_free),
      pUserData: ptr::null_mut(),
    };
    #[cfg(debug_assertions)]
    {
      allocator_create_info.device_memory_callbacks = Some(&callbacks);
    }

    let allocator = unsafe { vk_mem::Allocator::new(allocator_create_info) }?;

    let mut memory_properties = vk::PhysicalDeviceMemoryProperties2::default();
    unsafe {
      instance.get_physical_device_memory_properties2(physical_device, &mut memory_properties)
    };
    let heap_count = memory_properties.memory_properties.memory_heap_count as _;

    Ok(Self {
      allocator: mem::ManuallyDrop::new(allocator),
      memory_budgets: unsafe { Box::new_zeroed_slice(heap_count).assume_init() },
    })
  }

  pub fn refresh_vma_budgets(&mut self) {
    unsafe {
      self
        .allocator
        .get_heap_budgets_cached(&mut self.memory_budgets)
    };
  }

  pub fn set_current_frame_index(&self, frame_index: u32) {
    unsafe {
      self.allocator.set_current_frame_index(frame_index);
    };
  }

  // TODO: allocate buffer, image, ...
}

impl DeviceResource for GlobalDeviceAllocator {
  fn cleanup(&mut self, _device: &ash::Device) {
    unsafe { mem::ManuallyDrop::drop(&mut self.allocator) };
  }
}
