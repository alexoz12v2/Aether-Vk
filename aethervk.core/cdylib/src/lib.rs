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
