//! memory module.

use crate::{gpu_backends::vulkan::device::DeviceResource, types::GpuResult};
use alloc::boxed::Box;
use ash::vk;
use core::mem;
use function_name::named;

/// TODO: Document this item
pub struct GlobalDeviceAllocator {
  pub allocator: mem::ManuallyDrop<vk_mem::Allocator>,
  pub memory_budgets: Box<[vk_mem::ffi::VmaBudget]>,
  #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
  frame_index: alloc::boxed::Box<core::sync::atomic::AtomicU64>,
}

#[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
macro_rules! track_gpu_alloc {
  ($addr:expr, $size:expr) => {{
    aethervk_oshal_rlib::os::memory::tracking::GPU_ALLOCATED
      .fetch_add($size as usize, core::sync::atomic::Ordering::Relaxed);
    aethervk_oshal_rlib::os::memory::tracking::track_hotspot($size as usize);
    aethervk_oshal_rlib::os::memory::tracking::track_gpu_allocation($addr as u64, $size as usize);
    aethervk_oshal_rlib::os::memory::tracking::check_memory_threshold();
  }};
}

#[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
macro_rules! track_gpu_free {
  ($addr:expr, $size:expr) => {{
    aethervk_oshal_rlib::os::memory::tracking::GPU_ALLOCATED
      .fetch_sub($size as usize, core::sync::atomic::Ordering::Relaxed);
    aethervk_oshal_rlib::os::memory::tracking::untrack_gpu_allocation($addr as u64);
  }};
}

#[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
#[allow(unused)]
unsafe extern "C" fn on_device_alloc(
  allocator: vk_mem::ffi::VmaAllocator,
  memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  p_user_data: *mut core::ffi::c_void,
) {
  use ash::vk::Handle;

  let frame_index: u64 = if !p_user_data.is_null() {
    unsafe { &*p_user_data.cast::<core::sync::atomic::AtomicU64>() }
      .load(core::sync::atomic::Ordering::Relaxed)
  } else {
    0
  };

  track_gpu_alloc!(memory.as_raw(), size);

  aethervk_oshal_rlib::log!(
    "{} - [VMA] Alloc: size: {} bytes, type: {}, mem: {:#X}",
    frame_index,
    size,
    memory_type,
    memory.as_raw()
  );
  aethervk_oshal_rlib::os::debug::print_aethervk_stacktrace(7, 4);
}
#[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
#[allow(unused)]
unsafe extern "C" fn on_device_free(
  allocator: vk_mem::ffi::VmaAllocator,
  memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  p_user_data: *mut core::ffi::c_void,
) {
  use ash::vk::Handle;

  track_gpu_free!(memory.as_raw(), size);

  let frame_index: u64 = if !p_user_data.is_null() {
    unsafe { &*p_user_data.cast::<core::sync::atomic::AtomicU64>() }
      .load(core::sync::atomic::Ordering::Relaxed)
  } else {
    0
  };

  aethervk_oshal_rlib::log!(
    "{} - [VMA] Free:  size: {} bytes, type: {}, mem: {:#X}",
    frame_index,
    size,
    memory_type,
    memory.as_raw()
  );
  aethervk_oshal_rlib::os::debug::print_aethervk_stacktrace(7, 4);
}

impl GlobalDeviceAllocator {
  // safety: expects instance and device to have their function pointers already loaded
  /// TODO: Document this item
  #[named]
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
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    let frame_index = alloc::boxed::Box::new(core::sync::atomic::AtomicU64::new(0));
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    let callbacks = vk_mem::ffi::VmaDeviceMemoryCallbacks {
      pfnAllocate: Some(on_device_alloc),
      pfnFree: Some(on_device_free),
      pUserData: frame_index.as_ptr() as *mut _,
    };
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
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
      #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
      frame_index,
    })
  }

  // TODO: Pretty print them in avalonia debug view
  pub fn refresh_vma_budgets(&mut self) {
    unsafe { self.allocator.get_heap_budgets_cached(&mut self.memory_budgets) };
  }

  /// TODO: Document this item
  pub fn set_current_frame_index(&self, frame_index: u64) {
    unsafe {
      // wrapping works fine in terms of vma budget
      self.allocator.set_current_frame_index(frame_index as _);
    };
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    let _ = self.frame_index.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
  }

  // TODO: allocate buffer, image, ...
}

use core::sync::atomic::{AtomicUsize, Ordering};
use vk_mem::Alloc;

/// TODO: Document this item
pub struct FrameStagingArena {
  pub buffer: vk::Buffer,
  pub mapped_ptr: *mut u8,
  pub capacity: usize,
  pub offset: AtomicUsize,
  pub allocation: vk_mem::Allocation,
}

unsafe impl Send for FrameStagingArena {}
unsafe impl Sync for FrameStagingArena {}

#[macro_export]
macro_rules! apply_test_dedicated_alloc {
  ($alloc_info:expr) => {
    #[cfg(all(test, feature = "test_dedicated_alloc"))]
    {
      $alloc_info.flags |= vk_mem::AllocationCreateFlags::DEDICATED_MEMORY;
    }
  };
}

impl FrameStagingArena {
  /// TODO: Document this item
  #[named]
  pub fn new(allocator: &vk_mem::Allocator, capacity: usize) -> GpuResult<Self> {
    aethervk_oshal_rlib::log!("FrameStagingArena::new called! capacity={}", capacity);
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
    apply_test_dedicated_alloc!(alloc_info);

    let (buffer, allocation, alloc_info_res) =
      unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }
        .map_err(|_| crate::gpu_err_device!())?;

    aethervk_oshal_rlib::log!(
      "FrameStagingArena created alloc: {:?}",
      allocation.get_raw()
    );

    Ok(Self {
      buffer,
      mapped_ptr: alloc_info_res.mapped_data as *mut u8,
      capacity,
      offset: AtomicUsize::new(0),
      allocation,
    })
  }

  /// TODO: Document this item
  pub fn reset(&self) {
    self.offset.store(0, Ordering::Relaxed);
  }

  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn destroy(&mut self, allocator: vk_mem::AllocatorView) {
    aethervk_oshal_rlib::log!(
      "FrameStagingArena::destroy called! buf: {:?} alloc: {:?}",
      self.buffer,
      self.allocation.get_raw()
    );
    unsafe { allocator.destroy_buffer(self.buffer, &mut self.allocation) };
  }
}

impl DeviceResource for GlobalDeviceAllocator {
  fn cleanup(&mut self, _device: &ash::Device) {
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    aethervk_oshal_rlib::os::memory::tracking::report_leaked_gpu_allocations();
    unsafe { mem::ManuallyDrop::drop(&mut self.allocator) };
  }
}
