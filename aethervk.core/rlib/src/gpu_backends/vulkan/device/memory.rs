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
  _allocator: vk_mem::ffi::VmaAllocator,
  _memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  _p_user_data: *mut core::ffi::c_void,
) {
  use ash::vk::Handle;
  // SAFETY: this callback runs while VMA holds its internal C++ std::mutex.
  // ONLY lock-free atomic operations are safe here. Anything that touches the
  // heap allocator (log!, format!, alloc::*), a blocking lock (spin::Mutex,
  // backtrace, dladdr) or /proc/self/maps will deadlock (EDEADLK) against
  // VMA's mutex or glibc's internal allocator mutex.
  //
  // The commented-out code below uses the full-fidelity tracking path; it is
  // preserved for reference and can be restored once the EDEADLK hazard is
  // removed (e.g. by rebuilding with a separate allocator domain or by moving
  // the Vulkan device to a thread where none of those locks are re-entered).
  //
  // HOW TRACKING WORKS NOW:
  //   push_vma_event_alloc writes {addr, size} into a 1024-slot lock-free ring
  //   buffer (pure atomics, no alloc). Safe code calls drain_vma_events() — e.g.
  //   after cmd.submit() or at frame start — to process the events with
  //   full capabilities: backtrace capture, BTreeMap updates, logging.
  //
  // --- original code (DO NOT DELETE) ---
  // let frame_index: u64 = if !p_user_data.is_null() {
  //   unsafe { &*p_user_data.cast::<core::sync::atomic::AtomicU64>() }
  //     .load(core::sync::atomic::Ordering::Relaxed)
  // } else { 0 };
  // track_gpu_alloc!(memory.as_raw(), size);   // <- full BTreeMap + backtrace
  // aethervk_oshal_rlib::log!(
  //   "{} - [VMA] Alloc: size: {} bytes, type: {}, mem: {:#X}",
  //   frame_index, size, memory_type, memory.as_raw()
  // );
  // aethervk_oshal_rlib::os::debug::print_aethervk_stacktrace(7, 4);
  // --- end original code ---

  // Enqueue into the lock-free ring; drain_vma_events() does the real tracking.
  aethervk_oshal_rlib::os::memory::tracking::push_vma_event_alloc(memory.as_raw(), size);
}

#[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
#[allow(unused)]
unsafe extern "C" fn on_device_free(
  _allocator: vk_mem::ffi::VmaAllocator,
  _memory_type: u32,
  memory: vk::DeviceMemory,
  size: vk::DeviceSize,
  _p_user_data: *mut core::ffi::c_void,
) {
  use ash::vk::Handle;
  // Same SAFETY constraint as on_device_alloc — atomics only.
  //
  // --- original code (DO NOT DELETE) ---
  // track_gpu_free!(memory.as_raw(), size);   // <- BTreeMap remove
  // let frame_index: u64 = if !p_user_data.is_null() {
  //   unsafe { &*p_user_data.cast::<core::sync::atomic::AtomicU64>() }
  //     .load(core::sync::atomic::Ordering::Relaxed)
  // } else { 0 };
  // aethervk_oshal_rlib::log!(
  //   "{} - [VMA] Free:  size: {} bytes, type: {}, mem: {:#X}",
  //   frame_index, size, memory_type, memory.as_raw()
  // );
  // aethervk_oshal_rlib::os::debug::print_aethervk_stacktrace(7, 4);
  // --- end original code ---

  aethervk_oshal_rlib::os::memory::tracking::push_vma_event_free(memory.as_raw(), size);
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

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use vk_mem::Alloc;

/// The frame staging arena's reuse policy: two halves of one buffer, one per frame in flight.
///
/// A frame bump-allocates from the current half. `advance(last_submit)` records the graphics
/// timeline value of the last submission that could have used the half and switches to the other
/// one; `wait_value_for_next_half(completed)` says which timeline value the GPU must have reached
/// before that other half may be overwritten (`None` when it already has). Without this gate a
/// frame whose GPU work is delayed (a long dust fill on the compute queue, say) still had its
/// staging bytes — trajectory tables, scene/material/object data, gizmo uploads — overwritten by
/// the next frame's CPU work before its copies executed: the copies then moved zeros/garbage into
/// device buffers, the trajectory shader followed a null segment pointer and the GPU faulted
/// (Xid 31, `GPCCLIENT_T1 faulted @ 0x0`, 3/3 observer runs; 0/3 with the trajectory draw skipped).
pub struct StagingRing {
  pub half_capacity: usize,
  /// 0 or 1
  pub half: AtomicUsize,
  /// absolute bump pointer inside the current half
  pub offset: AtomicUsize,
  /// the graphics timeline value of the last submission that may have read each half
  pub used_until: [AtomicU64; 2],
}

impl StagingRing {
  pub fn new(half_capacity: usize) -> Self {
    Self {
      half_capacity,
      half: AtomicUsize::new(0),
      offset: AtomicUsize::new(0),
      used_until: [AtomicU64::new(0), AtomicU64::new(0)],
    }
  }

  pub fn base(&self) -> usize {
    self.half.load(Ordering::Relaxed) * self.half_capacity
  }

  /// Bump-allocates `size` bytes at `alignment` inside the current half; absolute offset.
  pub fn allocate(&self, size: usize, alignment: usize) -> Option<usize> {
    let end = self.base() + self.half_capacity;
    let mut current = self.offset.load(Ordering::Relaxed);
    loop {
      let padding = (alignment - (current % alignment)) % alignment;
      let aligned = current + padding;
      let next = aligned + size;
      if next > end {
        return None;
      }
      match self
        .offset
        .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::Relaxed)
      {
        Ok(_) => return Some(aligned),
        Err(val) => current = val,
      }
    }
  }

  /// The timeline value the GPU must reach before the next half can be reused, if it has not.
  pub fn wait_value_for_next_half(&self, completed: u64) -> Option<u64> {
    let next = 1 - self.half.load(Ordering::Relaxed);
    let needed = self.used_until[next].load(Ordering::Relaxed);
    (needed > completed).then_some(needed)
  }

  /// Ends the current frame's use of the arena: the current half is busy until the GPU passes
  /// `last_submit_value`; the next half becomes current and empty. Caller: once the wait of
  /// `wait_value_for_next_half` is satisfied.
  pub fn advance(&self, last_submit_value: u64) {
    let cur = self.half.load(Ordering::Relaxed);
    self.used_until[cur].fetch_max(last_submit_value, Ordering::Relaxed);
    let next = 1 - cur;
    self.half.store(next, Ordering::Relaxed);
    self.offset.store(next * self.half_capacity, Ordering::Relaxed);
  }
}

/// Host-visible scratch for the frame's uploads (copies and device-address reads recorded into
/// the frame's command buffers): two halves rotated per frame, see [`StagingRing`].
pub struct FrameStagingArena {
  pub buffer: vk::Buffer,
  pub mapped_ptr: *mut u8,
  /// the whole buffer: two halves of `ring.half_capacity`
  pub capacity: usize,
  pub ring: StagingRing,
  pub allocation: vk_mem::Allocation,
}

unsafe impl Send for FrameStagingArena {}
unsafe impl Sync for FrameStagingArena {}

#[macro_export]
macro_rules! apply_test_dedicated_alloc {
  ($alloc_info:ident) => {
    #[cfg(all(test, feature = "test_dedicated_alloc"))]
    let mut $alloc_info = $alloc_info;
    #[cfg(all(test, feature = "test_dedicated_alloc"))]
    {
      $alloc_info.flags |= vk_mem::AllocationCreateFlags::DEDICATED_MEMORY;
    }
  };
}

impl FrameStagingArena {
  /// TODO: Document this item
  #[named]
  pub fn new(allocator: &vk_mem::Allocator, per_frame_capacity: usize) -> GpuResult<Self> {
    // two halves: one per frame in flight (see `StagingRing`)
    let capacity = per_frame_capacity * 2;
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
      ring: StagingRing::new(per_frame_capacity),
      allocation,
    })
  }

  /// The graphics timeline value the GPU must reach before the next frame may reuse its half of
  /// the arena (`None`: already reached). `completed` = the current counter value.
  pub fn wait_value_for_next_frame(&self, completed: u64) -> Option<u64> {
    self.ring.wait_value_for_next_half(completed)
  }

  /// Switches to the other half for the next frame; `last_submit_value` = the value of the last
  /// graphics submission issued so far (`next_submit_value - 1`), which bounds every submission
  /// that may read the half being left. The caller has waited for `wait_value_for_next_frame`.
  pub fn advance(&self, last_submit_value: u64) {
    self.ring.advance(last_submit_value);
  }

  /// Bump-allocates `size` bytes at `alignment` in this frame's half: (absolute offset, host ptr).
  pub fn allocate(&self, size: usize, alignment: usize) -> Option<(usize, *mut u8)> {
    self
      .ring
      .allocate(size, alignment)
      .map(|aligned| (aligned, unsafe { self.mapped_ptr.add(aligned) }))
  }

  /// TODO: Document this item
  pub fn destroy(&mut self, allocator: vk_mem::AllocatorView) {
    aethervk_oshal_rlib::log!(
      "FrameStagingArena::destroy called! buf: {:?} alloc: {:?}",
      self.buffer,
      self.allocation.get_raw()
    );
    super::fault::unregister_address_range("FrameStagingArena");
    unsafe {
      vk_mem::ffi::vmaDestroyBuffer(allocator.get_raw(), self.buffer, self.allocation.get_raw());
      core::ptr::drop_in_place(&mut self.allocation);
    }
  }
}

impl DeviceResource for GlobalDeviceAllocator {
  fn cleanup(&mut self, _device: &super::LogicalDevice) {
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    {
      if let Ok(stats) = self.allocator.build_stats_string(true) {
        aethervk_oshal_rlib::log!("VMA STATS BEFORE DROP:\n{}", stats);
      }
    }
    unsafe { mem::ManuallyDrop::drop(&mut self.allocator) };
    // vmaDestroyAllocator (above) fires pfnFree for every remaining VkDeviceMemory
    // block. Those FREE events are now sitting in the lock-free ring buffer.
    // Drain them NOW, before report_leaked_gpu_allocations() reads GPU_ALLOCATIONS,
    // so the BTreeMap reflects all frees and the leak report is accurate.
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    {
      aethervk_oshal_rlib::os::memory::tracking::drain_vma_events();
      aethervk_oshal_rlib::os::memory::tracking::report_leaked_gpu_allocations();
    }
  }
}

#[cfg(test)]
mod staging_ring_tests {
  use super::StagingRing;

  /// Two frames alternate halves; a half is reusable only once the GPU passed the submission
  /// that used it; offsets of consecutive frames never overlap.
  #[test]
  fn staging_ring_alternates_halves_and_waits_for_the_gpu() {
    let ring = StagingRing::new(1024);
    // frame A (half 0)
    let a = ring.allocate(100, 16).unwrap();
    assert!(a < 1024);
    assert!(
      ring.allocate(2000, 16).is_none(),
      "a frame cannot exceed its half"
    );
    // frame A submitted as value 1; nothing completed yet
    assert_eq!(
      ring.wait_value_for_next_half(0),
      None,
      "half 1 was never used"
    );
    ring.advance(1);
    // frame B (half 1)
    let b = ring.allocate(100, 16).unwrap();
    assert!(b >= 1024 && b < 2048);
    // frame B submitted as value 2; the GPU has completed nothing: half 0 (frame A) is busy
    assert_eq!(ring.wait_value_for_next_half(0), Some(1));
    // the GPU finished frame A
    assert_eq!(ring.wait_value_for_next_half(1), None);
    ring.advance(2);
    // frame C (half 0 again), starts at the half's base
    let c = ring.allocate(100, 16).unwrap();
    assert_eq!(c, 0);
    // frame C submitted as value 3; the GPU is two frames behind: half 1 (frame B) is busy
    assert_eq!(ring.wait_value_for_next_half(1), Some(2));
    assert_eq!(ring.wait_value_for_next_half(2), None);
  }

  #[test]
  fn staging_ring_alignment_and_exhaustion() {
    let ring = StagingRing::new(256);
    assert_eq!(ring.allocate(1, 16).unwrap(), 0);
    assert_eq!(ring.allocate(1, 16).unwrap(), 16);
    assert_eq!(ring.allocate(200, 8).unwrap(), 24);
    assert!(ring.allocate(100, 8).is_none());
    ring.advance(7);
    assert_eq!(ring.allocate(1, 16).unwrap(), 256);
  }
}
