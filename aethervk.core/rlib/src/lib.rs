// disable std only for non tests
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

#[cfg(any(test, feature = "std"))]
extern crate std;

#[macro_use]
extern crate static_assertions as sa;

// panic_handler and global allocator declared for non tests at cdylib/FFI level
extern crate core;
extern crate alloc;

// publish panic_handler for our own cdylib
pub use aethervk_oshal_rlib::panic_handler_impl;

// TODO organize pub
pub mod gpu_backends;

pub mod gpu;
pub mod math;
pub mod physics;
pub mod scene;
pub mod simulation;
pub mod traits;
pub mod types;
