use core::{ffi::CStr, marker::PhantomData, ptr};

use crate::{
  gpu::RenderDevice,
  gpu_backends::vulkan,
  types::{GpuError, GpuResult},
};
use super::utils;

use alloc::{boxed::Box, format, vec::Vec};
use ash::{vk};
use vk_mem::AllocatorCreateFlags;

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  device: ash::Device,

  // manually dropped stuff
  global_device_allocator: Option<GlobalDeviceAllocator>,

  _instance: PhantomData<&'a vulkan::instance::Instance>,
}

impl<'a> Device<'a> {
  pub(super) fn new(
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<Self> {
    let eligible_physical_devices = instance.get_eligible_devices(query_input)?;

    let chosen_physical_device_query_result = match eligible_physical_devices.get(index) {
      Some(chosen_physical_device_query_result) => Ok(chosen_physical_device_query_result),
      None => Err(GpuError::BackendSpecific(format!(
        "There isn't a Vulkan capable device at index {}",
        index
      ))),
    }?;
    let physical_device = chosen_physical_device_query_result.physical_device;

    // 1. enable required and TODO optional features
    let mut required_features = utils::RequiredFeatures::new();
    required_features.populate();
    let mut features2 = required_features.as_features2();

    // 2. Setup queue create infos for necessary queues from query result
    // TODO: right now we are assuming 1 queue per queue family.
    let queue_priorities = [1f32];
    let queue_infos_len = chosen_physical_device_query_result.family_count();
    let mut queue_infos: Vec<_> = (0..queue_infos_len)
      .map(|i| {
        vk::DeviceQueueCreateInfo::default()
          .queue_family_index(i as _)
          .queue_priorities(&queue_priorities)
      })
      .collect();

    // 3. Device creation
    let enabled_extension_names: Vec<_> =
      chosen_physical_device_query_result.enabled_extension_names();
    let device_create_info = vk::DeviceCreateInfo::default()
      .enabled_extension_names(&enabled_extension_names)
      .push_next(&mut features2)
      .queue_create_infos(&queue_infos);

    let device = unsafe {
      instance
        .instance
        .create_device(physical_device, &device_create_info, None)
    }?;

    // 4. Global VMA Allocator creation, Queue handles, Global Discard Pool, Command Buffer Pool
    let global_device_allocator = Some(unsafe {
      GlobalDeviceAllocator::new(&instance.instance, &device, physical_device, instance.api_version())
    });

    todo!();
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }
}

impl<'a> Drop for Device<'a> {
  fn drop(&mut self) {
    // TODO log error
    unsafe { self.device.device_wait_idle().unwrap_unchecked() };
    // TODO Destroy allocator, queue handles, global discard pool, ... (ManuallyDrop or take from Option)
    todo!()
  }
}

impl<'a> RenderDevice for Device<'a> {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> alloc::string::String {
    todo!()
  }

  fn context_id(&self) -> u64 {
    vulkan::VULKAN_RENDER_BACKEND.0
  }
}

struct GlobalDeviceAllocator {
  allocator: vk_mem::Allocator,
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
  unsafe fn new(
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
      allocator,
      memory_budgets: unsafe { Box::new_zeroed_slice(heap_count).assume_init() },
    })
  }

  fn refresh_vma_budgets(&mut self) {
    unsafe {
      self
        .allocator
        .get_heap_budgets_cached(&mut self.memory_budgets)
    };
  }

  fn set_current_frame_index(&self, frame_index: u32) {
    unsafe {
      self.allocator.set_current_frame_index(frame_index);
    };
  }
}
