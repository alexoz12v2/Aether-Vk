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
      // sends a string to the debugger for display. no debugger = no op
      // TODO: Better implementation
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

/// Drop-in replacement for Option to track when it becomes None
#[cfg(debug_assertions)]
pub struct TrackedOption<T, const V: i32>(Option<T>);

#[cfg(debug_assertions)]
impl<T, const V: i32> TrackedOption<T, { V }> {
  pub const fn some(val: T) -> Self {
    Self(Some(val))
  }

  pub const fn none() -> Self {
    Self(None)
  }

  /// Intercepts `take` and logs/panics with the caller's location
  #[track_caller]
  pub fn take(&mut self) -> Option<T> {
    let caller = core::panic::Location::caller();

    log!(
      "Option tag {} Taken at {}:{}",
      V,
      caller.file(),
      caller.line()
    );
    self.0.take()
  }

  #[track_caller]
  pub fn replace(&mut self, value: T) -> Option<T> {
    let caller = core::panic::Location::caller();

    log!(
      "Option tag {} Replaced at {}:{}",
      V,
      caller.file(),
      caller.line()
    );
    self.0.replace(value)
  }

  // Pass through other common Option methods you might need
  pub fn is_none(&self) -> bool {
    self.0.is_none()
  }
  pub fn is_some(&self) -> bool {
    self.0.is_some()
  }
  pub fn as_ref(&self) -> Option<&T> {
    self.0.as_ref()
  }
  pub fn as_mut(&mut self) -> Option<&mut T> {
    self.0.as_mut()
  }
  #[track_caller]
  pub fn unwrap(self) -> T {
    // Option::unwrap already tracks caller natively, but adding the
    // attribute here ensures the panic points to *our* code, not this wrapper.
    self.0.unwrap()
  }

  // --- Unwrapping Methods ---
  #[track_caller]
  pub fn expect(self, msg: &str) -> T {
    self.0.expect(msg)
  }

  pub fn unwrap_or(self, default: T) -> T {
    self.0.unwrap_or(default)
  }

  pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
    self.0.unwrap_or_else(f)
  }

  pub fn unwrap_or_default(self) -> T
  where
    T: Default,
  {
    self.0.unwrap_or_default()
  }

  // --- Functional Combinators ---
  pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> {
    self.0.map(f)
  }

  pub fn and_then<U, F: FnOnce(T) -> Option<U>>(self, f: F) -> Option<U> {
    self.0.and_then(f)
  }
}

#[cfg(debug_assertions)]
pub struct DropTracker<T, const V: i32> {
  pub inner: T,
}

#[cfg(debug_assertions)]
impl<T, const V: i32> DropTracker<T, { V }> {
  pub fn new(inner: T) -> Self {
    Self { inner }
  }
}

#[cfg(debug_assertions)]
impl<T, const V: i32> Drop for DropTracker<T, { V }> {
  #[track_caller]
  fn drop(&mut self) {
    let caller = core::panic::Location::caller();
    // Put a debugger breakpoint on the line below!
    // When the debugger halts here, look at your call stack to see
    // exactly what triggered the drop (and thus, the None state).

    // OR, if you have logging enabled:
    log!(
      "The inner value tag {} was just dropped! at {}:{}",
      V,
      caller.file(),
      caller.line()
    );
  }
}

// Allows `&*tracker` to transparently act as `&T`
// You can now call methods of `T` directly on `DropTracker<T>`
#[cfg(debug_assertions)]
impl<T, const V: i32> core::ops::Deref for DropTracker<T, { V }> {
  type Target = T;

  fn deref(&self) -> &Self::Target {
    &self.inner
  }
}

// Allows `&mut *tracker` to transparently act as `&mut T`
#[cfg(debug_assertions)]
impl<T, const V: i32> core::ops::DerefMut for DropTracker<T, { V }> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.inner
  }
}

// Allows passing `&DropTracker<T>` to functions expecting `impl AsRef<T>`
#[cfg(debug_assertions)]
impl<T, const V: i32> AsRef<T> for DropTracker<T, { V }> {
  fn as_ref(&self) -> &T {
    &self.inner
  }
}

// Allows passing `&mut DropTracker<T>` to functions expecting `impl AsMut<T>`
impl<T, const V: i32> AsMut<T> for DropTracker<T, { V }> {
  fn as_mut(&mut self) -> &mut T {
    &mut self.inner
  }
}

// Highly recommended for debugging: pass through Debug formatting if T supports it
impl<T: core::fmt::Debug, const V: i32> core::fmt::Debug for DropTracker<T, { V }> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    self.inner.fmt(f)
  }
}
