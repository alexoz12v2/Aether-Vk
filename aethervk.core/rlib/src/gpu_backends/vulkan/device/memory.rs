use alloc::boxed::Box;
use ash::vk;
use core::{mem, ptr};

use crate::{gpu_backends::vulkan::device::DeviceResource, types::GpuResult};
use aethervk_oshal_rlib as oshal;

pub struct GlobalDeviceAllocator {
  pub allocator: mem::ManuallyDrop<vk_mem::Allocator>,
  pub memory_budgets: Box<[vk_mem::ffi::VmaBudget]>,
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
  use ash::vk::Handle;

  oshal::log!(
    "[VMA] Alloc: size: {} bytes, type: {}, mem: {:#X}",
    size,
    memory_type,
    memory.as_raw()
  );
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
  use ash::vk::Handle;

  oshal::log!(
    "[VMA] Free:  size: {} bytes, type: {}, mem: {:#X}",
    size,
    memory_type,
    memory.as_raw()
  );
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
      | vk_mem::AllocatorCreateFlags::KHR_DEDICATED_ALLOCATION
      | vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
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
    unsafe { self.allocator.get_heap_budgets_cached(&mut self.memory_budgets) };
  }

  pub fn set_current_frame_index(&self, frame_index: u32) {
    unsafe {
      self.allocator.set_current_frame_index(frame_index);
    };
  }

  // TODO: allocate buffer, image, ...
}

use core::sync::atomic::{AtomicUsize, Ordering};
use vk_mem::Alloc;

pub struct FrameStagingArena {
  pub buffer: vk::Buffer,
  pub mapped_ptr: *mut u8,
  pub capacity: usize,
  pub offset: AtomicUsize,
  pub allocation: vk_mem::Allocation,
}

unsafe impl Send for FrameStagingArena {}
unsafe impl Sync for FrameStagingArena {}

impl FrameStagingArena {
  pub fn new(allocator: &vk_mem::Allocator, capacity: usize) -> GpuResult<Self> {
    let buffer_info = vk::BufferCreateInfo::default()
      .size(capacity as u64)
      .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
    let alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::Auto,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT,
      ..Default::default()
    };

    let (buffer, allocation, alloc_info_res) =
      unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }
        .map_err(|_| crate::gpu_err!("device error"))?;

    Ok(Self {
      buffer,
      mapped_ptr: alloc_info_res.mapped_data as *mut u8,
      capacity,
      offset: AtomicUsize::new(0),
      allocation,
    })
  }

  pub fn reset(&self) {
    self.offset.store(0, Ordering::Relaxed);
  }

  pub fn allocate(&self, size: usize, alignment: usize) -> Option<(usize, *mut u8)> {
    let mut current = self.offset.load(Ordering::Relaxed);
    loop {
      let padding = (alignment - (current % alignment)) % alignment;
      let aligned = current + padding;
      let next = aligned + size;

      if next > self.capacity {
        return None;
      }

      match self.offset.compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::Relaxed) {
        Ok(_) => return Some((aligned, unsafe { self.mapped_ptr.add(aligned) })),
        Err(val) => current = val,
      }
    }
  }

  pub fn destroy(&mut self, allocator: &vk_mem::Allocator) {
    unsafe {
      allocator.destroy_buffer(self.buffer, &mut self.allocation);
    }
  }
}

impl DeviceResource for GlobalDeviceAllocator {
  fn cleanup(&mut self, _device: &ash::Device) {
    unsafe { mem::ManuallyDrop::drop(&mut self.allocator) };
  }
}
