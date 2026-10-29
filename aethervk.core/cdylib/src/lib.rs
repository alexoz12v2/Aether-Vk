//! lib module.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;
// std is linked on every desktop target anyway (see `the_panic`); only its panic hook is used
#[cfg(all(not(feature = "std"), not(target_arch = "wasm32")))]
extern crate std;

use aethervk_core_rlib as r#impl;

use core::panic::PanicInfo;

extern crate alloc;
extern crate core;

pub mod ffi;
pub mod ffi_assets;
pub mod ffi_fits;
pub mod oshal;

// ----------- Allocator Setup --------------------------------------
#[cfg(target_env = "msvc")]
#[cfg(not(feature = "std"))]
use mimalloc::MiMalloc;
#[cfg(not(target_env = "msvc"))]
#[cfg(not(feature = "std"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[cfg(not(feature = "std"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(target_env = "msvc")]
#[cfg(not(feature = "std"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[panic_handler]
#[cfg(not(any(test, feature = "std")))]
#[cfg(target_arch = "wasm32")] // Hack to disable it on non-wasm, because we know macOS is linking std due to ab_glyph
fn the_panic(_info: &PanicInfo) -> ! {
  unsafe {
    if let Some(cb) = crate::PANIC_CALLBACK {
      // In no_std, we avoid allocation if we can.
      // We pass a simple static message to avoid complex formatting which might allocate or fail further.
      let msg = b"Fatal Rust Engine Panic: Check console logs or use a debugger.\0";
      cb(msg.as_ptr(), msg.len());
    }
  }
  aethervk_oshal_rlib::panic_handler_impl();
}

// -------------------- Linker Functions ----------------------------
pub use ffi::*;
pub use oshal::*;

// -------------------- Necessary Evilness --------------------------
// `liballoc` expects some symbols for unwinding panic even though we specified abort.
// 2 fixes: 1) enable thin LTO in debug (no.) 2) dummy symbol (this one)
#[cfg(debug_assertions)]
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))] // I observed this only on my Apple Silicon
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

  let _ = log::set_logger(&LOGGER);
  #[cfg(not(target_arch = "wasm32"))]
  install_panic_hook();
  #[cfg(debug_assertions)]
  log::set_max_level(log::LevelFilter::Trace);
  #[cfg(not(debug_assertions))]
  log::set_max_level(log::LevelFilter::Debug);
}

/// Prefix of the log line a panic sends through the logger callback. The cdylib aborts on panic, so
/// that line is the host's only chance to write a crash report (the .NET app writes it
/// synchronously when it sees this prefix; a Release build on Windows has no console for stderr).
pub const PANIC_LOG_PREFIX: &str = "[PANIC]";

/// The panic message, the thread and a backtrace go through the logger callback, then the default
/// hook (stderr) runs.
#[cfg(not(target_arch = "wasm32"))]
fn install_panic_hook() {
  use alloc::boxed::Box;
  let previous = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    let thread = std::thread::current();
    let backtrace = std::backtrace::Backtrace::force_capture();
    aethervk_oshal_rlib::log!(
      "{} thread '{}' {}\nbacktrace:\n{}",
      PANIC_LOG_PREFIX,
      thread.name().unwrap_or("<unnamed>"),
      info,
      backtrace
    );
    previous(info);
  }));
}

struct AvkLogger;

impl log::Log for AvkLogger {
  fn enabled(&self, _metadata: &log::Metadata) -> bool {
    true
  }

  fn log(&self, record: &log::Record) {
    // TODO put false if you need anise logs
    // if self.enabled(record.metadata()) {
    //   aethervk_oshal_rlib::log!("[{}] {}", record.level(), record.args());
    // }
  }

  fn flush(&self) {}
}

static LOGGER: AvkLogger = AvkLogger;

#[cfg(test)]
mod panic_hook_tests {
  use std::{ffi::CStr, io::Write, string::String};

  const CHILD_ENV: &str = "AETHERVK_PANIC_HOOK_CHILD_LOG";

  extern "C" fn to_file(msg: *const core::ffi::c_char) {
    let path = std::env::var(CHILD_ENV).unwrap();
    let text = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
    let _ = f.write_all(text.as_bytes());
  }

  /// Run by [`a_panic_reaches_the_logger_callback`] in a child process.
  #[test]
  #[ignore = "child of a_panic_reaches_the_logger_callback"]
  fn panic_hook_child() {
    if std::env::var(CHILD_ENV).is_err() {
      return;
    }
    crate::r#impl::simulation_api::SimulationContext::set_logger_callback(Some(to_file));
    std::thread::Builder::new()
      .name(String::from("logic"))
      .spawn(|| panic!("boom at frame 42"))
      .unwrap()
      .join()
      .ok();
  }

  /// The cdylib aborts on panic: the panic text (thread, message, location, backtrace) must reach
  /// the host through the logger callback with [`crate::PANIC_LOG_PREFIX`], so the .NET app can
  /// write its crash report before the process dies (reviewer report 2026-10-29: a crash after a few
  /// minutes left nothing to send).
  #[test]
  fn a_panic_reaches_the_logger_callback() {
    let dir = std::env::temp_dir().join(std::format!("avk_panic_hook_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("child.log");
    let _ = std::fs::remove_file(&log);
    let out = std::process::Command::new(std::env::current_exe().unwrap())
      .args(["--exact", "panic_hook_tests::panic_hook_child", "--ignored", "--nocapture"])
      .env(CHILD_ENV, &log)
      .output()
      .unwrap();
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
      text.starts_with("[PANIC] thread 'logic' "),
      "child stdout {}\nstderr {}\nlog {text}",
      String::from_utf8_lossy(&out.stdout),
      String::from_utf8_lossy(&out.stderr)
    );
    assert!(text.contains("boom at frame 42"), "{text}");
    assert!(text.contains("lib.rs"), "the location: {text}");
    assert!(text.contains("backtrace:"), "{text}");
  }
}
