#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

use aethervk_core_rlib as r#impl;

use core::{panic::PanicInfo};

extern crate core;
extern crate alloc;

pub mod oshal;

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
#[cfg(not(any(test, feature = "std")))]
fn the_panic(_info: &PanicInfo) -> ! {
  aethervk_oshal_rlib::panic_handler_impl();
}

// -------------------- Linker Functions ----------------------------
pub use oshal::*;

// -------------------- Necessary Evilness --------------------------
// `liballoc` expects some symbols for unwinding panic even though we specified abort.
// 2 fixes: 1) enable thin LTO in debug (no.) 2) dummy symbol (this one)
#[cfg(debug_assertions)]
#[cfg(target_arch = "aarch64")] // I observed this only on my Apple Silicon
#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}

// Link against `libSystem`, which brings in some macOS native functions needed for jemalloc
#[cfg(target_os = "macos")]
#[link(name = "System")]
unsafe extern "C" {}

// -------------------- Linker Functions ----------------------------
// https://stackoverflow.com/questions/30700596/with-mach-o-is-there-a-way-to-register-a-function-that-will-run-before-main
#[used]
#[cfg(all(not(windows), target_family = "unix", target_vendor = "apple"))]
#[unsafe(link_section = "__DATA,__mod_init_func")]
static MACH_O_CONSTRUCTOR: fn() = on_load;

#[used]
#[cfg(all(not(windows), target_family = "unix", not(target_vendor = "apple")))]
#[unsafe(link_section = ".init_array")]
static ELF64_CONSTRUCTOR: fn() = on_load;

#[cfg(not(windows))]
fn on_load() {
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
