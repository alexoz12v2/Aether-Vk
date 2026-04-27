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

pub static LOGGER_CALLBACK: core::sync::atomic::AtomicPtr<()> = core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

#[cfg(target_os = "windows")]
mod windows_debug {
  use core::fmt;
  use windows::core::HSTRING;
  use windows::Win32::System::Diagnostics::Debug::OutputDebugStringW;

  #[cfg(feature = "console_log")]
  use spin::Once;
  #[cfg(feature = "console_log")]
  use windows::core::w;
  #[cfg(feature = "console_log")]
  use windows::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
  #[cfg(feature = "console_log")]
  use windows::Win32::Storage::FileSystem::{
    CreateFileW, WriteFile, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
  };
  #[cfg(feature = "console_log")]
  use windows::Win32::System::Console::{
    GetConsoleMode, SetConsoleMode, SetConsoleOutputCP, CONSOLE_MODE,
    ENABLE_VIRTUAL_TERMINAL_PROCESSING,
  };

  #[cfg(feature = "console_log")]
  struct SyncHandle(HANDLE);

  #[cfg(feature = "console_log")]
  unsafe impl Sync for SyncHandle {}
  #[cfg(feature = "console_log")]
  unsafe impl Send for SyncHandle {}
  #[cfg(feature = "console_log")]
  impl From<HANDLE> for SyncHandle {
    fn from(value: HANDLE) -> Self {
      Self(value)
    }
  }

  #[cfg(feature = "console_log")]
  static CONSOLE_HANDLE: Once<Option<SyncHandle>> = Once::new();

  #[cfg(feature = "console_log")]
  fn init_console() -> Option<SyncHandle> {
    unsafe {
      // Target the active console buffer explicitly, even if stdout is redirected
      // Note: 0x80000000 | 0x40000000 equates to GENERIC_READ | GENERIC_WRITE.
      // Using raw values prevents import hell across different windows crate versions.
      let handle = CreateFileW(
        w!("CONOUT$"),
        0x80000000 | 0x40000000,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
      )
          .unwrap_or(INVALID_HANDLE_VALUE);

      if handle.is_invalid() {
        return None;
      }

      // Set the output code page to UTF-8 (CP_UTF8 = 65001)
      let _ = SetConsoleOutputCP(65001);

      // Enable virtual terminal processing (ANSI escape sequences)
      let mut mode = CONSOLE_MODE(0);
      if GetConsoleMode(handle, &mut mode).is_ok() {
        let _ = SetConsoleMode(
          handle,
          CONSOLE_MODE(mode.0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING.0),
        );
      }

      Some(handle.into())
    }
  }

  pub fn log_message(args: fmt::Arguments) {
    let msg = alloc::fmt::format(args) + "\r\n";

    if let Ok(c_msg) = alloc::ffi::CString::new(msg.clone()) {
      let fptr = super::LOGGER_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
      if !fptr.is_null() {
        let cb: extern "C" fn(*const core::ffi::c_char) = unsafe { core::mem::transmute(fptr) };
        cb(c_msg.as_ptr());
        return;
      }
    }

    #[cfg(feature = "console_log")]
    {
      // Initializes the console on the first call, subsequent calls just fetch the Option<HANDLE>
      let handle_opt = CONSOLE_HANDLE.call_once(init_console);

      if let Some(handle) = handle_opt {
        unsafe {
          let mut bytes_written = 0;
          // WriteFile expects a byte slice, so msg.as_bytes() maps perfectly to UTF-8 output
          let _ = WriteFile(
            handle.0,
            Some(msg.as_bytes()),
            Some(&mut bytes_written),
            None,
          );
        }
      }
    }

    let h_string = HSTRING::from(msg.as_str());
    unsafe {
      // sends a string to the debugger for display. no debugger = no op
      // TODO: Better implementation with WinDbg
      OutputDebugStringW(&h_string);
    }
  }

  pub fn print_stacktrace() {
    use core::mem::{size_of, MaybeUninit};

    // We define standard Win32 structures to circumvent differing
    // pointer/handle types mapped across versions of the windows-rs crate.
    #[link(name = "kernel32")]
    unsafe extern "system" {
      fn RtlCaptureStackBackTrace(
        FramesToSkip: u32,
        FramesToCapture: u32,
        BackTrace: *mut *mut core::ffi::c_void,
        BackTraceHash: *mut u32,
      ) -> u16;
      fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    #[link(name = "dbghelp")]
    unsafe extern "system" {
      fn SymInitialize(
        hProcess: *mut core::ffi::c_void,
        UserSearchPath: *const core::ffi::c_char,
        fInvadeProcess: i32,
      ) -> i32;
      fn SymSetOptions(SymOptions: u32) -> u32;
      fn SymFromAddr(
        hProcess: *mut core::ffi::c_void,
        Address: u64,
        Displacement: *mut u64,
        Symbol: *mut SYMBOL_INFO,
      ) -> i32;
    }

    #[repr(C)]
    struct SYMBOL_INFO {
      SizeOfStruct: u32,
      TypeIndex: u32,
      Reserved: [u64; 2],
      Index: u32,
      Size: u32,
      ModBase: u64,
      Flags: u32,
      Value: u64,
      Address: u64,
      Register: u32,
      Scope: u32,
      Tag: u32,
      NameLen: u32,
      MaxNameLen: u32,
      Name: [core::ffi::c_char; 1],
    }

    unsafe {
      let process = GetCurrentProcess();

      // Ensures DbgHelp's initialization happens safely exactly once.
      static SYM_INIT: ::spin::Once<()> = ::spin::Once::new();
      SYM_INIT.call_once(|| {
        SymSetOptions(0x00000002); // SYMOPT_UNDNAME (Demangles C++ names)
        SymInitialize(process, core::ptr::null(), 1); // 1 = TRUE (fInvadeProcess)
      });

      let mut buffer: [*mut core::ffi::c_void; 64] = [core::ptr::null_mut(); 64];
      let frames = RtlCaptureStackBackTrace(0, 64, buffer.as_mut_ptr(), core::ptr::null_mut());

      if frames == 0 {
        crate::log!("Stacktrace: (empty or failed to capture)");
        return;
      }

      crate::log!("Stacktrace:");

      // Align memory to 8-byte bounds. SYMBOL_INFO requires contiguous trailing space
      // for the dynamically sized name string since DbgHelp natively maps into it.
      const SYMBOL_BUFFER_SIZE: usize = size_of::<SYMBOL_INFO>() + 256;
      #[repr(C, align(8))]
      struct SymbolBuffer([u8; SYMBOL_BUFFER_SIZE]);

      for i in 0..frames {
        let addr = buffer[i as usize];

        let mut sym_buf = MaybeUninit::<SymbolBuffer>::zeroed();
        let symbol_info = sym_buf.as_mut_ptr() as *mut SYMBOL_INFO;

        (*symbol_info).SizeOfStruct = size_of::<SYMBOL_INFO>() as u32;
        (*symbol_info).MaxNameLen = 255;

        let mut displacement: u64 = 0;
        let success = SymFromAddr(process, addr as u64, &mut displacement, symbol_info);

        if success != 0 {
          let name_len = core::cmp::min((*symbol_info).NameLen as usize, 254);
          let name_ptr = (*symbol_info).Name.as_ptr() as *const u8;
          let name_slice = core::slice::from_raw_parts(name_ptr, name_len);

          if let Ok(name_str) = core::str::from_utf8(name_slice) {
            crate::log!("  [{:2}] {} +0x{:x} ({:p})", i, name_str, displacement, addr);
            continue;
          }
        }

        // Fallback if the symbol wasn't natively resolvable / valid utf-8
        crate::log!("  [{:2}] <unknown> ({:p})", i, addr);
      }
    }
  }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod unix_debug {
  use core::fmt;

  #[cfg(not(feature = "std"))]
  struct StdoutWriter;

  #[cfg(not(feature = "std"))]
  impl fmt::Write for StdoutWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
      unsafe { libc::write(libc::STDOUT_FILENO, s.as_ptr().cast(), s.len()) };
      Ok(())
    }
  }

  #[cfg(not(feature = "std"))]
  pub fn log_message(args: fmt::Arguments) {
    let msg = alloc::fmt::format(args) + "\n";
    if let Ok(c_msg) = alloc::ffi::CString::new(msg.clone()) {
      let fptr = super::LOGGER_CALLBACK.load(core::sync::atomic::Ordering::Relaxed);
      if !fptr.is_null() {
        let cb: extern "C" fn(*const core::ffi::c_char) = unsafe { core::mem::transmute(fptr) };
        cb(c_msg.as_ptr());
        return;
      }
    }

    // Simple write to stderr.
    // In a real no_std environment, this would need a different approach.
    // For now, this requires the `std` feature.
    #[cfg(feature = "std")]
    eprintln!("{}", args);
    #[cfg(all(not(feature = "std"), feature = "console_log"))]
    {
      use core::fmt::Write;

      let mut writer = StdoutWriter;
      // format the passed arguments
      let _ = writer.write_fmt(args);
      // then add the newline
      let _ = writer.write_str("\n");
    }
  }

  #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
  pub fn print_stacktrace() {
    unsafe extern "C" {
      // Exposed by default in libc without any std dependencies
      fn backtrace(buffer: *mut *mut core::ffi::c_void, size: core::ffi::c_int) -> core::ffi::c_int;
      fn backtrace_symbols(
        buffer: *const *mut core::ffi::c_void,
        size: core::ffi::c_int,
      ) -> *mut *mut core::ffi::c_char;
    }

    unsafe {
      let mut buffer: [*mut core::ffi::c_void; 64] = [core::ptr::null_mut(); 64];
      let size = backtrace(buffer.as_mut_ptr(), 64);

      if size > 0 {
        let symbols = backtrace_symbols(buffer.as_ptr(), size);
        if !symbols.is_null() {
          crate::log!("Stacktrace:");
          for i in 0..size {
            let ptr = *symbols.add(i as usize);
            if !ptr.is_null() {
              let c_str = core::ffi::CStr::from_ptr(ptr);
              if let Ok(s) = c_str.to_str() {
                crate::log!("  [{:2}] {}", i, s);
              } else {
                crate::log!("  [{:2}] <invalid utf8> {:p}", i, buffer[i as usize]);
              }
            } else {
              crate::log!("  [{:2}] {:p}", i, buffer[i as usize]);
            }
          }
          // libc's backtrace_symbols explicitly allocates its return array pointer
          libc::free(symbols as *mut core::ffi::c_void);
        } else {
          crate::log!("Stacktrace: (symbols temporarily unavailable)");
          for i in 0..size {
            crate::log!("  [{:2}] {:p}", i, buffer[i as usize]);
          }
        }
      } else {
        crate::log!("Stacktrace: (empty)");
      }
    }
  }

  #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
  pub fn print_stacktrace() {
    crate::log!("Stacktrace: Not natively supported by libc in this target environment.");
  }
}
