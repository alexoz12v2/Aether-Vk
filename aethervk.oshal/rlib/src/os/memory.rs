//! Memory query utilities.

use core::{alloc::Layout, cell::Cell, mem::MaybeUninit, ptr::NonNull};

#[cfg(target_os = "linux")]
pub use linux_memory::*;
#[cfg(target_os = "macos")]
pub use macos_memory::*;
#[cfg(target_os = "windows")]
pub use windows_memory::*;

#[cfg(target_os = "windows")]
mod windows_memory {
  use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

  pub fn query_memory_status() -> MemoryStatus {
    let mut mem_status = MEMORYSTATUSEX {
      dwLength: core::mem::size_of::<MEMORYSTATUSEX>() as u32,
      ..Default::default()
    };
    unsafe {
      GlobalMemoryStatusEx(&mut mem_status).unwrap_or_default();
    }
    MemoryStatus {
      total_bytes: mem_status.ullTotalPhys,
      available_bytes: mem_status.ullAvailPhys,
    }
  }
}

#[cfg(target_os = "macos")]
mod macos_memory {
  use libc::{c_void, sysctl};

  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

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

    // Getting available memory is more complex and requires parsing vm_stat
    // This is a simplified version and not fully accurate.
    let available_bytes = 0; // Placeholder

    MemoryStatus {
      total_bytes,
      available_bytes,
    }
  }
}

#[cfg(target_os = "linux")]
mod linux_memory {
  // Reading /proc/meminfo requires file operations, which are not ideal in a no_std environment.
  // A proper implementation would require a no_std compatible way to read files.
  // For now, we return 0.
  pub struct MemoryStatus {
    pub total_bytes: u64,
    pub available_bytes: u64,
  }

  pub fn query_memory_status() -> MemoryStatus {
    MemoryStatus {
      total_bytes: 0,
      available_bytes: 0,
    }
  }
}

/// Wraps the byte array to guarantee strict 4K page alignment for the memory block.
#[repr(C, align(4096))]
pub struct PageAlignedStorage<const N: usize>(pub [u8; N]);

#[repr(C, align(8))]
pub struct MaxAlignedStorage<const N: usize>(pub [u8; N]);

/// The bump allocator only tracks the offset. It takes the base pointer
/// dynamically to prevent self-referencing struct issues.
pub struct StackAllocator {
  pub offset: Cell<usize>,
}

impl StackAllocator {
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
