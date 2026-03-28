//! Debugging utilities.

#[macro_export]
macro_rules! log {
  ($($arg:tt)*) => {
    $crate::os::debug::log_message(core::format_args!($($arg)*));
  };
}

#[cfg(target_os = "windows")]
pub use windows_debug::*;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use unix_debug::*;

#[cfg(target_os = "windows")]
mod windows_debug {
  use core::fmt;
  use windows::core::HSTRING;
  use windows::Win32::System::Diagnostics::Debug::OutputDebugStringW;

  pub fn log_message(args: fmt::Arguments) {
    let msg = fmt::format(args);
    let h_string = HSTRING::from(msg.as_str());
    unsafe {
      OutputDebugStringW(&h_string);
    }
  }

  pub fn print_stacktrace() {
    // Implementation for Windows stacktrace here
    // This is a complex topic and requires careful implementation.
    // For now, we'll just log a message.
    log!("Stacktrace (Windows): Not yet implemented.");
  }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_debug {
  use core::fmt;

  pub fn log_message(args: fmt::Arguments) {
    // Simple write to stderr.
    // In a real no_std environment, this would need a different approach.
    // For now, this requires the `std` feature.
    #[cfg(feature = "std")]
    eprintln!("{}", args);
    #[cfg(not(feature = "std"))]
    {
      // In a true no_std environment, you would need a different way to output logs.
      // For example, a serial port or a specific memory buffer.
      // This is a placeholder.
      let _ = args;
    }
  }

  pub fn print_stacktrace() {
    // Placeholder for non-Windows stacktrace
    log!("Stacktrace (Unix): Not yet implemented.");
  }
}
