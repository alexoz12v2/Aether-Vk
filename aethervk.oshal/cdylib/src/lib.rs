#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

use aethervk_oshal_rlib as r#impl;
pub use r#impl::{AvkSystemInfo};

extern crate core;
extern crate alloc;

use core::{ptr, mem};

#[cfg(not(windows))]
use ctor_bare::{register_ctor};
use spin::once;

// ----------- Allocator Setup --------------------------------------
#[cfg(not(target_env = "msvc"))]
#[cfg(not(feature = "std"))]
use tikv_jemallocator::Jemalloc;
#[cfg(target_env = "msvc")]
#[cfg(not(feature = "std"))]
use mimalloc::MiMalloc;

#[cfg(not(target_env = "msvc"))]
#[cfg(not(feature = "std"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(target_env = "msvc")]
#[cfg(not(feature = "std"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// ----------------- Panic Handler ----------------------------------
#[panic_handler]
#[cfg(all(not(test), feature = "std"))]
fn the_panic(_info: &core::panic::PanicInfo) -> ! {
  r#impl::panic_handler_impl();
}

// -------------------- Static Storage ------------------------------
static SYSTEM_INFO: once::Once<AvkSystemInfo> = once::Once::new();

// -------------------- C Exposed API -------------------------------
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn avkSystemInfo_get(system_info: *mut AvkSystemInfo) {
  if !system_info.is_null() {
    unsafe {
      ptr::copy_nonoverlapping(
        SYSTEM_INFO.as_mut_ptr(),
        system_info,
        mem::size_of::<AvkSystemInfo>(),
      );
    }
  }
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse2()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE3(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse3()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSSE3(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_ssse3()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE4_1(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse4_1()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasSSE4_2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_sse4_2()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasFMA(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_fma()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasAVX(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_avx()
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
#[cfg(all(target_arch = "x86_64", not(target_os = "none")))]
pub unsafe extern "C" fn avkSystemInfo_hasAVX2(system_info: *const AvkSystemInfo) -> bool {
  if system_info.is_null() {
    return false;
  }

  unsafe { *system_info }.has_avx2()
}

// -------------------- Linker Functions ----------------------------
#[used]
#[cfg(not(windows))]
#[register_ctor]
unsafe fn on_load() {
  unsafe { load() };
}

// this is the /ENTRY for link.exe, so it shouldn't be elided even with LTO
#[cfg(windows)] // this runs after link_section `.CRT$XCU.`, but it's fine
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
unsafe extern "system" fn DllMain(
  _hinst: *mut core::ffi::c_void,
  fdw_reason: u32,
  _reserved: *mut core::ffi::c_void,
) -> i32 {
  use windows::Win32::System::SystemServices::DLL_PROCESS_ATTACH;

  if fdw_reason == DLL_PROCESS_ATTACH {
    unsafe { load() };
  }

  1
}

unsafe fn load() {
  SYSTEM_INFO.call_once(|| AvkSystemInfo::new());
}
