//! Memory query utilities.

use core::{alloc::Layout, cell::Cell, mem::MaybeUninit, ptr::NonNull};

#[cfg(target_os = "linux")]
pub use linux_memory::*;
#[cfg(target_os = "macos")]
pub use macos_memory::*;
#[cfg(target_os = "windows")]
pub use windows_memory::*;

#[derive(Debug, Clone, Copy, Default)]
/// TODO: Document this item
pub struct ProcessMemory {
  pub virtual_bytes: u64,
  pub physical_bytes: u64,
  pub file_backed_bytes: u64,
}

#[cfg(target_os = "windows")]
mod windows_memory {
  use super::ProcessMemory;
  use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQueryEx,
  };
  use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
  use windows::Win32::System::Threading::GetCurrentProcess;

  /// TODO: Document this item
  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

  /// TODO: Document this item
  pub fn query_memory_status() -> MemoryStatus {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut mem_status = MEMORYSTATUSEX {
      dwLength: core::mem::size_of::<MEMORYSTATUSEX>() as u32,
      ..Default::default()
    };
    unsafe {
      let _ = GlobalMemoryStatusEx(&mut mem_status);
    }
    MemoryStatus {
      total_bytes: mem_status.ullTotalPhys,
      available_bytes: mem_status.ullAvailPhys,
    }
  }

  /// TODO: Document this item
  pub fn query_process_memory() -> ProcessMemory {
    let mut mem = ProcessMemory::default();
    let process = unsafe { GetCurrentProcess() };

    let mut counters = PROCESS_MEMORY_COUNTERS::default();
    counters.cb = core::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    unsafe {
      if GetProcessMemoryInfo(process, &mut counters, counters.cb).is_ok() {
        mem.virtual_bytes = counters.PagefileUsage as u64;
        mem.physical_bytes = counters.WorkingSetSize as u64;
      }
    }

    let mut address: usize = 0;
    let mut info = MEMORY_BASIC_INFORMATION::default();
    unsafe {
      while VirtualQueryEx(
        process,
        Some(address as *const core::ffi::c_void),
        &mut info,
        core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
      ) != 0
      {
        if info.State == MEM_COMMIT {
          if info.Type == MEM_MAPPED || info.Type == MEM_IMAGE {
            mem.file_backed_bytes += info.RegionSize as u64;
          }
        }
        address += info.RegionSize;
      }
    }

    mem
  }
}

#[cfg(target_os = "macos")]
mod macos_memory {
  use super::ProcessMemory;
  use libc::{c_void, sysctl};
  use mach2::task::task_info;
  use mach2::task_info::{MACH_TASK_BASIC_INFO, MACH_TASK_BASIC_INFO_COUNT, mach_task_basic_info};
  use mach2::traps::mach_task_self;
  use mach2::vm_prot::VM_PROT_READ;
  use mach2::vm_region::{VM_REGION_BASIC_INFO_64, vm_region_basic_info_64};

  /// TODO: Document this item
  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

  /// TODO: Document this item
  pub fn query_memory_status() -> MemoryStatus {
    let mut total_bytes: u64 = 0;
    let mut size = core::mem::size_of::<u64>();
    let mut mib = [libc::CTL_HW, libc::HW_MEMSIZE];
    unsafe {
      sysctl(
        mib.as_mut_ptr(),
        2,
        &mut total_bytes as *mut _ as *mut c_void,
        &mut size,
        core::ptr::null_mut(),
        0,
      );
    }

    let available_bytes = 0; // Placeholder
    MemoryStatus {
      total_bytes,
      available_bytes,
    }
  }

  /// TODO: Document this item
  pub fn query_process_memory() -> ProcessMemory {
    let mut mem = ProcessMemory::default();
    unsafe {
      let mut info: mach_task_basic_info = core::mem::zeroed();
      let mut count = MACH_TASK_BASIC_INFO_COUNT;
      let ret = task_info(
        mach_task_self(),
        MACH_TASK_BASIC_INFO,
        &mut info as *mut _ as *mut i32,
        &mut count,
      );
      if ret == 0 {
        // KERN_SUCCESS
        mem.virtual_bytes = info.virtual_size;
        mem.physical_bytes = info.resident_size;
      }

      let mut address: mach2::vm_types::mach_vm_address_t = 0;
      let mut size: mach2::vm_types::mach_vm_size_t = 0;
      let mut obj_name: mach2::port::mach_port_t = 0;
      loop {
        let mut info: vm_region_basic_info_64 = core::mem::zeroed();
        let mut count = mach2::vm_region::VM_REGION_BASIC_INFO_COUNT_64;

        let ret = mach2::vm::mach_vm_region(
          mach_task_self(),
          &mut address,
          &mut size,
          VM_REGION_BASIC_INFO_64,
          &mut info as *mut _ as *mut i32,
          &mut count,
          &mut obj_name,
        );

        if ret != 0 {
          break;
        }

        // shared/external memory usually corresponds to file-backed mappings (mmap)
        if info.shared != 0 || info.reserved != 0 {
          mem.file_backed_bytes += size;
        }

        address += size;
      }
    }
    mem
  }
}

#[cfg(target_os = "linux")]
mod linux_memory {
  use super::ProcessMemory;

  /// TODO: Document this item
  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

  /// TODO: Document this item
  pub fn query_memory_status() -> MemoryStatus {
    MemoryStatus {
      total_bytes: 0,
      available_bytes: 0,
    }
  }

  /// TODO: Document this item
  pub fn query_process_memory() -> ProcessMemory {
    let mut mem = ProcessMemory::default();
    let page_size = 4096; // typical page size

    use crate::os::fs::{PathBuf, read};
    if let Ok(bytes) = read(PathBuf::from("/proc/self/statm").as_ref()) {
      if let Ok(s) = core::str::from_utf8(&bytes) {
        let mut iter = s.split_whitespace();
        if let (Some(size), Some(resident)) = (iter.next(), iter.next()) {
          mem.virtual_bytes = size.parse::<u64>().unwrap_or(0) * page_size;
          mem.physical_bytes = resident.parse::<u64>().unwrap_or(0) * page_size;
        }
      }
    }

    if let Ok(bytes) = read(PathBuf::from("/proc/self/maps").as_ref()) {
      if let Ok(s) = core::str::from_utf8(&bytes) {
        for line in s.lines() {
          let parts: alloc::vec::Vec<&str> = line.split_whitespace().collect();
          if parts.len() >= 6 {
            let path = parts[5];
            if path.starts_with('/') {
              // indicates a file mapping
              let addr_part = parts[0];
              let mut addrs = addr_part.split('-');
              if let (Some(start_str), Some(end_str)) = (addrs.next(), addrs.next()) {
                if let (Ok(start), Ok(end)) = (
                  u64::from_str_radix(start_str, 16),
                  u64::from_str_radix(end_str, 16),
                ) {
                  mem.file_backed_bytes += end.saturating_sub(start);
                }
              }
            }
          }
        }
      }
    }

    mem
  }
}

/// Wraps the byte array to guarantee strict 4K page alignment for the memory block.
#[repr(C, align(4096))]
pub struct PageAlignedStorage<const N: usize>(pub [u8; N]);

#[repr(C, align(8))]
/// TODO: Document this item
pub struct MaxAlignedStorage<const N: usize>(pub [u8; N]);

/// The bump allocator only tracks the offset. It takes the base pointer
/// dynamically to prevent self-referencing struct issues.
pub struct StackAllocator {
  pub offset: Cell<usize>,
}

impl StackAllocator {
  /// TODO: Document this item
  pub const fn new() -> Self {
    Self {
      offset: Cell::new(0),
    }
  }

  /// Allocates memory, writes the value, and returns a raw pointer.
  pub unsafe fn allocate<T>(
    &self,
    base: *mut u8,
    len: usize,
    value: T,
  ) -> Result<*mut T, &'static str> {
    let layout = Layout::new::<T>();
    let start = self.offset.get();

    let current_ptr = unsafe { base.add(start) };
    let align_offset = current_ptr.align_offset(layout.align());

    let aligned_start = start.checked_add(align_offset).ok_or("Math overflow")?;
    let end = aligned_start.checked_add(layout.size()).ok_or("Math overflow")?;

    if end > len {
      return Err("Janitor out of memory");
    }

    self.offset.set(end);

    unsafe {
      let ptr = base.add(aligned_start) as *mut T;
      ptr.write(value);

      Ok(ptr)
    }
  }
}

#[cfg(all(debug_assertions, feature = "debug_gpu"))]
pub mod tracking {
  use core::alloc::{GlobalAlloc, Layout};
  use core::sync::atomic::{AtomicUsize, Ordering};
  extern crate alloc;
  use alloc::collections::BTreeMap;
  use spin::Mutex;

  pub static CPU_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
  pub static GPU_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

  #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
  pub struct AllocTrace(pub [usize; 3]);

  pub static HOTSPOTS: Mutex<Option<BTreeMap<AllocTrace, usize>>> = Mutex::new(None);

  pub struct TrackingAllocator<A: GlobalAlloc>(pub A);

  pub fn track_hotspot(size: usize) {
    if let Some(trace) = crate::os::debug::capture_aethervk_trace() {
      let mut lock = HOTSPOTS.lock();
      if lock.is_none() {
        *lock = Some(BTreeMap::new());
      }
      if let Some(map) = lock.as_mut() {
        let trace_key = AllocTrace(trace);
        let current = map.get(&trace_key).copied().unwrap_or(0);
        map.insert(trace_key, current + size);
      }
    }
  }

  pub fn print_memory_state() {
    let process_mem = super::query_process_memory();
    let real_mem = process_mem.physical_bytes;
    let gpu_mem = GPU_ALLOCATED.load(Ordering::Relaxed);
    let cpu_mem = CPU_ALLOCATED.load(Ordering::Relaxed);
    crate::log!(
      "[MEMORY STATE] Real Mem: {:.2} MB, GPU Mem: {:.2} MB, CPU Tracked: {:.2} MB",
      real_mem as f64 / 1_048_576.0,
      gpu_mem as f64 / 1_048_576.0,
      cpu_mem as f64 / 1_048_576.0
    );
  }

  pub fn check_memory_threshold() {
    let process_mem = super::query_process_memory();
    let real_mem = process_mem.physical_bytes as usize;
    let gpu_mem = GPU_ALLOCATED.load(Ordering::Relaxed);
    let threshold = 3_006_477_107; // 2.8 GB

    if real_mem > threshold || gpu_mem > threshold {
      crate::log!(
        "[MEMORY WARNING] Threshold Exceeded! Real Mem: {} bytes, GPU Mem: {} bytes, CPU Tracked: {} bytes",
        real_mem,
        gpu_mem,
        CPU_ALLOCATED.load(Ordering::Relaxed)
      );
    }
  }

  unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
      CPU_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
      check_memory_threshold();
      self.0.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
      CPU_ALLOCATED.fetch_sub(layout.size(), Ordering::Relaxed);
      self.0.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
      CPU_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
      track_hotspot(layout.size());
      check_memory_threshold();
      self.0.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
      if new_size > layout.size() {
        CPU_ALLOCATED.fetch_add(new_size - layout.size(), Ordering::Relaxed);
      } else {
        CPU_ALLOCATED.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
      }
      check_memory_threshold();
      self.0.realloc(ptr, layout, new_size)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_query_memory_status() {
    let status = query_memory_status();
    // Since different platforms return different things, just check it doesn't crash
    // and if it returns total > 0 on supported platforms
    #[cfg(not(target_os = "linux"))]
    assert!(status.total_bytes > 0);
  }

  #[test]
  fn test_stack_allocator() {
    let allocator = StackAllocator::new();
    let mut buffer = [0u8; 1024];
    let base_ptr = buffer.as_mut_ptr();

    let val_ptr = unsafe { allocator.allocate(base_ptr, 1024, 42u32).unwrap() };
    unsafe { assert_eq!(*val_ptr, 42) };

    let val_ptr2 = unsafe { allocator.allocate(base_ptr, 1024, 100u64).unwrap() };
    unsafe { assert_eq!(*val_ptr2, 100) };
  }
}
