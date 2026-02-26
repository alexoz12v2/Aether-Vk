#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

use aethervk_core_rlib as r#impl;
pub use r#impl::{};

use core::{panic::PanicInfo};

extern crate core;
extern crate alloc;

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
#[cfg(all(not(test), not(feature = "std")))]
fn the_panic(_info: &PanicInfo) -> ! {
  r#impl::panic_handler_impl();
}

// -------------------- Linker Functions ----------------------------
// -------------------- Necessary Evilness --------------------------
// `liballoc` expects some symbols for unwinding panic even though we specified abort.
// 2 fixes: 1) enable thin LTO in debug (no.) 2) dummy symbol (this one)
#[cfg(debug_assertions)]
#[cfg(target_arch = "aarch64")] // I observed this only on my Apple Silicon
#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}
