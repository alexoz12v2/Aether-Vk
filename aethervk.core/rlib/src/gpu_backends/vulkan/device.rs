use core::{marker::PhantomData, ptr};

use crate::{
  gpu::RenderDevice,
  gpu_backends::vulkan,
  types::{GpuError, GpuResult},
};
use super::utils;

use alloc::{boxed::Box, format, vec::Vec};
use ash::{vk};

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  pub device: ash::Device,

  // manually dropped stuff
  global_device_allocator: Option<GlobalDeviceAllocator>,

  _instance: PhantomData<&'a vulkan::instance::Instance>,
}

impl<'a> Device<'a> {
  /// Initializes a Device directly into the provided memory location
  /// This avoids returning a Device by value (which would probably cause stack overflow)
  pub(super) unsafe fn init_at_ptr(
    dst: *mut Self,
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<()> {
    unsafe { ptr::write(dst, Self::new(instance, index, query_input)?) };
    Ok(())
  }

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
    let queue_infos: Vec<_> = (0..queue_infos_len)
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
      GlobalDeviceAllocator::new(
        &instance.instance,
        &device,
        physical_device,
        instance.api_version(),
      )
    }?);

    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device,
      global_device_allocator,
      _instance: PhantomData,
    })
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }

  pub(super) fn with_device_allocator<T>(&self, f: impl FnOnce(&GlobalDeviceAllocator) -> T) -> T {
    let dalloc = self.global_device_allocator.as_ref().unwrap();
    f(dalloc)
  }

  pub(super) fn with_device_allocator_mut<T>(
    &mut self,
    f: impl FnOnce(&mut GlobalDeviceAllocator) -> T,
  ) -> T {
    let mut dalloc = self.global_device_allocator.as_mut().unwrap();
    f(dalloc)
  }
}

impl<'a> Drop for Device<'a> {
  fn drop(&mut self) {
    // TODO log error
    unsafe { self.device.device_wait_idle().unwrap_unchecked() };

    // TODO Destroy allocator, queue handles, global discard pool, ... (ManuallyDrop or take from Option)
    drop(self.global_device_allocator.take());

    // in the end, destroy the device
    unsafe { self.device.destroy_device(None) };
  }
}

impl<'a> RenderDevice for Device<'a> {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> alloc::string::String {
    use alloc::format;

    let props = &self.query_result.physical_device_properties;
    let device_name = props
      .device_name_as_c_str()
      .unwrap()
      .to_string_lossy()
      .into_owned();
    let device_type = match props.device_type {
      vk::PhysicalDeviceType::CPU => "CPU",
      vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
      vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
      vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
      _ => "Other",
    };

    let api_major = vk::api_version_major(props.api_version);
    let api_minor = vk::api_version_minor(props.api_version);
    let api_patch = vk::api_version_patch(props.api_version);

    format!(
      "Vulkan Device Info\n\
       ------------------\n\
       Name: {}\n\
       Vendor ID: {:#X} ({})\n\
       Device ID: {:#X}\n\
       Type: {}\n\
       API Version: {}.{}.{}\n\
       Driver Version: {}\n\
       Queue Families: {}\n",
      device_name,
      props.vendor_id,
      match props.vendor_id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x106B => "Apple",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1010 => "ImgTec",
        _ => "Unknown",
      },
      props.device_id,
      device_type,
      api_major,
      api_minor,
      api_patch,
      props.driver_version,
      self.query_result.family_count()
    )
  }

  fn context_id(&self) -> u64 {
    vulkan::VULKAN_RENDER_BACKEND.0
  }
}

pub(super) struct GlobalDeviceAllocator {
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
      allocator,
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
