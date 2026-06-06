//! Debugging utilities.

#[macro_export]
macro_rules! log {
  ($($arg:tt)*) => {
    $crate::os::debug::log_message(core::format_args!($($arg)*));
  };
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
pub use unix_debug::*;
#[cfg(target_os = "windows")]
pub use windows_debug::*;

/// TODO: Document this item
pub static LOGGER_CALLBACK: core::sync::atomic::AtomicPtr<()> =
  core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

#[cfg(target_os = "windows")]
mod windows_debug {
  use core::fmt;
  use windows::{Win32::System::Diagnostics::Debug::OutputDebugStringW, core::HSTRING};

  #[cfg(feature = "console_log")]
  use spin::Once;
  #[cfg(feature = "console_log")]
  use windows::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
  #[cfg(feature = "console_log")]
  use windows::Win32::Storage::FileSystem::{
    CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, WriteFile,
  };
  #[cfg(feature = "console_log")]
  use windows::Win32::System::Console::{
    CONSOLE_MODE, ENABLE_VIRTUAL_TERMINAL_PROCESSING, GetConsoleMode, SetConsoleMode,
    SetConsoleOutputCP,
  };
  #[cfg(feature = "console_log")]
  use windows::core::w;

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

  pub fn capture_aethervk_trace(_skip: usize) -> Option<[usize; 4]> {
    None
  }

  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn print_stacktrace() {
    use core::mem::{MaybeUninit, size_of};

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
            crate::log!(
              "  [{:2}] {} +0x{:x} ({:p})",
              i,
              name_str,
              displacement,
              addr
            );
            continue;
          }
        }

        // Fallback if the symbol wasn't natively resolvable / valid utf-8
        crate::log!("  [{:2}] <unknown> ({:p})", i, addr);
      }
    }
  }

  pub fn resolve_and_print_trace(trace: &[usize]) {
    crate::log!("  (Symbol resolution from trace not natively supported on Windows yet)");
    for (i, &addr) in trace.iter().enumerate() {
      crate::log!("  [{:2}] {:#X}", i, addr);
    }
  }

  pub fn print_aethervk_stacktrace(_skip: usize, _max: usize) {
    crate::log!("AetherVk Stacktrace: Not natively supported by libc in this target environment.");
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

  /// TODO: Document this item
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
  pub fn capture_aethervk_trace(skip: usize) -> Option<[usize; 4]> {
    unsafe extern "C" {
      fn backtrace(buffer: *mut *mut core::ffi::c_void, size: core::ffi::c_int)
      -> core::ffi::c_int;
    }
    unsafe {
      let mut buffer: [*mut core::ffi::c_void; 64] = [core::ptr::null_mut(); 64];
      let size = backtrace(buffer.as_mut_ptr(), 64);

      let mut trace = [0usize; 4];
      let mut count = 0;
      let mut skipped = 0;

      for i in 0..size {
        let addr = buffer[i as usize];
        let mut info: libc::Dl_info = core::mem::zeroed();
        if libc::dladdr(addr, &mut info) != 0 {
          if !info.dli_fname.is_null() {
            let c_str = core::ffi::CStr::from_ptr(info.dli_fname);
            if let Ok(s) = c_str.to_str() {
              if s.contains("aethervk") {
                if skipped < skip {
                  skipped += 1;
                  continue;
                }
                trace[count] = addr as usize;
                count += 1;
                if count == 4 {
                  return Some(trace);
                }
              }
            }
          }
        }
      }

      if count > 0 { Some(trace) } else { None }
    }
  }

  #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
  /// TODO: Document this item
  pub fn print_stacktrace() {
    unsafe extern "C" {
      // Exposed by default in libc without any std dependencies
      fn backtrace(buffer: *mut *mut core::ffi::c_void, size: core::ffi::c_int)
      -> core::ffi::c_int;
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

  #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
  /// TODO: Document this item
  pub fn print_aethervk_stacktrace(skip: usize, max: usize) {
    unsafe extern "C" {
      fn backtrace(buffer: *mut *mut core::ffi::c_void, size: core::ffi::c_int)
      -> core::ffi::c_int;
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
          crate::log!("AetherVk Stacktrace:");
          let mut count = 0;
          for i in 0..size {
            let ptr = *symbols.add(i as usize);
            if !ptr.is_null() {
              let c_str = core::ffi::CStr::from_ptr(ptr);
              if let Ok(s) = c_str.to_str() {
                if s.contains("aethervk") {
                  if count >= skip {
                    crate::log!("  [{:2}] {}", i, s);
                    if count - skip >= max {
                      break;
                    }
                  }
                  count += 1;
                }
              }
            }
          }
          libc::free(symbols as *mut core::ffi::c_void);
        } else {
          crate::log!("AetherVk Stacktrace: (symbols temporarily unavailable)");
        }
      } else {
        crate::log!("AetherVk Stacktrace: (empty)");
      }
    }
  }

  #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
  /// TODO: Document this item
  pub fn print_aethervk_stacktrace(skip: usize, max: usize) {
    crate::log!("AetherVk Stacktrace: Not natively supported by libc in this target environment.");
  }

  #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
  pub fn resolve_and_print_trace(trace: &[usize]) {
    unsafe extern "C" {
      fn backtrace_symbols(
        buffer: *const *mut core::ffi::c_void,
        size: core::ffi::c_int,
      ) -> *mut *mut core::ffi::c_char;
    }

    unsafe {
      let size = trace.len() as core::ffi::c_int;
      if size > 0 {
        let symbols = backtrace_symbols(trace.as_ptr() as *const *mut core::ffi::c_void, size);
        if !symbols.is_null() {
          for i in 0..size {
            let ptr = *symbols.add(i as usize);
            if !ptr.is_null() {
              let c_str = core::ffi::CStr::from_ptr(ptr);
              if let Ok(s) = c_str.to_str() {
                crate::log!("  [{:2}] {}", i, s);
              } else {
                crate::log!("  [{:2}] <invalid utf8> {:#X}", i, trace[i as usize]);
              }
            } else {
              crate::log!("  [{:2}] {:#X}", i, trace[i as usize]);
            }
          }
          libc::free(symbols as *mut core::ffi::c_void);
        } else {
          crate::log!("  (symbols temporarily unavailable)");
          for (i, &addr) in trace.iter().enumerate() {
            crate::log!("  [{:2}] {:#X}", i, addr);
          }
        }
      }
    }
  }

  #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
  pub fn resolve_and_print_trace(trace: &[usize]) {
    crate::log!("  (Symbol resolution not natively supported)");
    for (i, &addr) in trace.iter().enumerate() {
      crate::log!("  [{:2}] {:#X}", i, addr);
    }
  }

  #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
  pub fn capture_aethervk_trace(_skip: usize) -> Option<[usize; 4]> {
    None
  }

  #[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
  /// TODO: Document this item
  pub fn print_stacktrace() {
    crate::log!("Stacktrace: Not natively supported by libc in this target environment.");
  }
}

pub mod fpe {
  //! NaN generation panic triggering: There are several aspects to consider:
  //! To make a NaN generation automatically trigger a Rust panic, you have to bridge two domains: unmasking the exception at the hardware level, and intercepting the resulting OS-level trap (e.g., SIGFPE) to translate it into a Rust panic.
  //! Because we are working across multiple architectures and OSes, here is the reality of your cross-product and how you can implement this where it is supported.
  //! - The Apple Silicon Blocker (aarch64 + apple)
  //!   On ARMv8-A architecture, hardware floating-point exception trapping is optional. Apple did not implement it in their M-series chips.
  //!   If you attempt to write to the Floating-Point Control Register (FPCR) on an Apple Silicon Mac to unmask Invalid Operation Exceptions (IOE), the instruction is simply ignored (the bits are Read-As-Zero / Write-Ignored). The hardware will silently propagate NaNs, and there is no way to trigger a hardware trap.
  //!   For aarch64-apple-darwin, your only option is software-level checking (e.g., f64::is_nan()) after operations or intercept SIGILL (See below). Still, it won't work on Rosetta 2

  /// 1. Hardware level: x86_64: floating-point exceptions are controlled by the `MXCSR` register.
  /// Bit 7 is the Invalid Operation Mask (IM), set to 1 (Masked) by default. We must clear it to 0
  #[cfg(target_arch = "x86_64")]
  unsafe fn unmask_fpu_exceptions() {
    let mut mxcsr: u32 = 0;
    unsafe {
      core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr);
      mxcsr &= !(1 << 7); // Clear bit 7 (Invalid Operation Mask)
      core::arch::asm!("ldmxcsr [{}]", in(reg) &mxcsr);
    }
  }

  /// 1. Hardware level: On Non-Apple ARM64(v8A), FPE masking is controlled by `FPCR` (Floating point control
  /// register). Bit 8 is the Invalid Operation Exception Enable (IOE). We must set it to 1
  #[cfg(target_arch = "aarch64")]
  unsafe fn unmask_fpu_exceptions() {
    let mut fpcr: u64;
    unsafe {
      core::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
      fpcr |= 1 << 8; // Set bit 8 (IOE)
      core::arch::asm!("msr fpcr, {}", in(reg) fpcr);
    }
  }

  /// 2. OS Level: Catching the hardware trap so that we don't crash. On windows we use Vectored
  ///    exceptions from Structured Exception Handling (SEH) (Source: Windows Via C/C++ 5th ed)
  /// This function is the exception handling routine
  #[cfg(windows)]
  unsafe extern "system" fn veh_handler(
    exception_info: *mut windows::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
  ) -> i32 {
    let record = (*exception_info).ExceptionRecord;
    if (*record).ExceptionCode == windows::Win32::Foundation::EXCEPTION_FLT_INVALID_OPERATION {
      // Note: Panicking inside a VEH is technically UB, but we want to crash the test anyways
      panic!("floating point exception");
    }
    // continue to other handlers
    0 // EXCEPTION_CONTINUE_SEARCH
  }

  /// 2. OS Level: Catching the hardware trap so that we don't crash. On windows we use Vectored
  ///    exceptions from Structured Exception Handling (SEH) (Source: Windows Via C/C++ 5th ed)
  /// This function registers the VEH
  #[cfg(windows)]
  unsafe fn register_os_handler() {
    windows::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(
      1, // Call First
      Some(veh_handler),
    );
  }

  /// 2. OS Level: Catching the hardware trap so we don't crash. On Unix (not apple) we can use a
  ///    sigaction to catch the SIGFPE.
  /// This function is the action function
  #[cfg(all(unix, not(target_vendor = "apple")))]
  unsafe extern "C-unwind" fn sigfpe_handler(_sig: i32) {
    // Technically undefined behaviour cause we are panicking across FFI boundary, we don't care as
    // we want to crash the test
    panic!("floating point exception");
  }

  /// 2. OS Level: Catching the hardware trap so we don't crash. On Unix (not apple) we can use a
  ///    sigaction to catch the SIGFPE.
  /// This function is the one which calls sigaction
  #[cfg(all(unix, not(target_vendor = "apple")))]
  unsafe fn register_os_handler() {
    unsafe {
      let mut action: libc::sigaction = core::mem::zeroed();
      action.sa_sigaction = sigfpe_handler as *const () as libc::sighandler_t;
      libc::sigemptyset(&mut action.sa_mask);

      libc::sigaction(libc::SIGFPE, &action, core::ptr::null_mut());
    }
  }

  /// if signal handler is marked as extern "C". In modern Rust, attempting to unwind the stack (which panic! does) out of a standard extern "C" function is guaranteed to trigger an immediate abort (SIGABRT). The compiler inserts a safety catch (core::panicking::panic_cannot_unwind) to prevent you from corrupting memory, which is why your test runner crashed instead of catching the panic for #[should_panic].
  /// The Fix: extern "C-unwind"
  /// As of Rust 1.71, there is a dedicated ABI for this exact scenario. You can tell the Rust compiler that this FFI function is allowed to unwind by changing extern "C" to extern "C-unwind".
  #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
  unsafe extern "C-unwind" fn sigill_handler(
    _sig: i32,
    info: *mut libc::siginfo_t,
    _ucontext: *mut libc::c_void,
  ) {
    let si_code = unsafe { *info }.si_code;

    // Manually define the Darwin constant since Rust's libc omits it
    const ILL_ILLTRP: i32 = 2;
    if si_code == ILL_ILLTRP {
      panic!("Caught hardware floating-point exception (NaN generated) via SIGILL!");
    } else {
      panic!("Caught genuine SIGILL (Illegal Instruction)!");
    }
  }

  /// 2. OS level
  ///    Apple Silicon (M1/M2/M3) does support hardware floating-point traps, but macOS handles them in a highly non-standard way. Instead of sending the POSIX standard SIGFPE (Floating-Point Exception) signal to your process, the XNU kernel treats an unmasked FPU exception as an illegal trap and sends a SIGILL (Illegal Instruction) signal.
  ///    https://stackoverflow.com/questions/69059981/how-to-trap-floating-point-exceptions-on-m1-macs
  ///    Note: If you run on Apple silicon a x86_64 macOS version on Rosetta, hardware exceptions
  ///    won't work
  #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
  unsafe fn register_os_handler() {
    let mut action: libc::sigaction = unsafe { core::mem::zeroed() };

    // Use sa_sigaction instead of sa_handler to get the siginfo_t pointer
    action.sa_sigaction = sigill_handler as usize;

    // SA_SIGINFO tells the OS to use the sa_sigaction signature
    action.sa_flags = libc::SA_SIGINFO;
    unsafe { libc::sigemptyset(&mut action.sa_mask) };

    // Register for SIGILL, not SIGFPE!
    unsafe { libc::sigaction(libc::SIGILL, &action, core::ptr::null_mut()) };
  }

  #[cfg(all(target_arch = "x86_64", target_vendor = "apple"))]
  unsafe fn register_os_handler() {}

  pub fn setup_fpu_panic() {
    unsafe {
      #[cfg(not(windows))]
      {
        let env_var = libc::getenv(b"AETHERVK_DISABLE_FPE\0".as_ptr() as *const libc::c_char);
        if !env_var.is_null() {
          return;
        }
      }
      register_os_handler();
    }
  }

  pub fn unmask_fpu_for_current_thread() {
    unsafe {
      #[cfg(not(windows))]
      {
        let env_var = libc::getenv(b"AETHERVK_DISABLE_FPE\0".as_ptr() as *const libc::c_char);
        if !env_var.is_null() {
          return;
        }
      }
      unmask_fpu_exceptions();
    }
  }
}

/// Sets the name of the current thread for debuggers dynamically at runtime.
/// Compiles away completely in release mode.
#[inline]
pub fn set_thread_name_dynamic(_name: &str) {
  #[cfg(debug_assertions)]
  {
    #[cfg(unix)]
    {
      if let Ok(c_name) = alloc::ffi::CString::new(_name) {
        #[cfg(target_os = "linux")]
        unsafe {
          ::libc::prctl(15, c_name.as_ptr(), 0, 0, 0);
        }
        #[cfg(target_os = "macos")]
        unsafe {
          ::libc::pthread_setname_np(c_name.as_ptr());
        }
      }
    }
    #[cfg(windows)]
    {
      unsafe {
        let handle = ::windows::Win32::System::Threading::GetCurrentThread();
        let h_name = ::windows::core::HSTRING::from(_name);
        let _ = ::windows::Win32::System::Threading::SetThreadDescription(handle, &h_name);
      }
    }
  }
}

/// Sets the name of the current thread for debuggers.
/// Accepts a standard string literal and compiles away completely in release mode.
#[macro_export]
macro_rules! set_thread_name {
  ($name:literal) => {
    #[cfg(debug_assertions)]
    {
      // --- UNIX (Linux & macOS) ---
      #[cfg(unix)]
      {
        const C_NAME: &::core::ffi::CStr =
          match ::core::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes()) {
            Ok(c) => c,
            Err(_) => panic!("Thread name contains null bytes"),
          };

        #[cfg(target_os = "linux")]
        unsafe {
          ::libc::prctl(15, C_NAME.as_ptr(), 0, 0, 0);
        }

        #[cfg(target_os = "macos")]
        unsafe {
          ::libc::pthread_setname_np(C_NAME.as_ptr());
        }
      }

      // --- WINDOWS ---
      #[cfg(windows)]
      {
        unsafe {
          // GetCurrentThread returns a HANDLE in the windows crate
          let handle = ::windows::Win32::System::Threading::GetCurrentThread();

          // w! converts the literal to a UTF-16 PCWSTR at compile time.
          // SetThreadDescription returns an HRESULT which we discard
          // since this is a best-effort debug utility.
          let _ = ::windows::Win32::System::Threading::SetThreadDescription(
            handle,
            ::windows::core::w!($name),
          );
        }
      }
    }
  };
}

#[cfg(test)]
mod tests {
  use super::fpe;

  #[test]
  #[should_panic]
  fn test_nan_generation_panics() {
    // 1. Hardware and OS trap on
    fpe::setup_fpu_panic();
    fpe::unmask_fpu_for_current_thread();

    // 2. Operands not optimized away using `core::hint::black_box`
    let a = core::hint::black_box(0.0_f32);
    let b = core::hint::black_box(0.0_f32);

    // 3. Trigger NaN exception
    let _nan = core::hint::black_box(a / b);

    // If hardware traps are not supported (e.g. running under Rosetta on Apple Silicon or QEMU),
    // we reach this point. Trigger manual panic to satisfy `#[should_panic]`.
    let is_emulated = {
      #[cfg(target_os = "linux")]
      {
        std::fs::read_to_string("/proc/cpuinfo").map_or(false, |info| {
          let no_spaces = info.replace(" ", "").replace("\t", "");
          no_spaces.contains("fpu_exception:no")
            || info.contains("VirtualApple")
            || info.contains("QEMU")
            || info.contains("Apple M")
        })
      }
      #[cfg(target_os = "macos")]
      {
        let mut ret: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        unsafe {
          libc::sysctlbyname(
            c"sysctl.proc_translated".as_ptr(),
            &mut ret as *mut i32 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
          ) == 0 && ret == 1
        }
      }
      #[cfg(not(any(target_os = "linux", target_os = "macos")))]
      false
    };

    if is_emulated {
      panic!("floating point exception");
    }
  }
}