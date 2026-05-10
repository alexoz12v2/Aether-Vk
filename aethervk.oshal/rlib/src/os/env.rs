//! env module.

use alloc::{string::String, vec::Vec};

use crate::os::NativeResult;

/// TODO: Document this item
pub fn args() -> NativeResult<Vec<String>> {
  #[cfg(windows)]
  {
    use crate::os::NativeError;
    use windows::Win32::Foundation::{HLOCAL, LocalFree};
    use windows::Win32::System::Environment::GetCommandLineW;
    use windows::Win32::UI::Shell::CommandLineToArgvW;

    unsafe {
      let cmd_line = GetCommandLineW();
      if cmd_line.as_ptr().is_null() {
        return Err(NativeError::UnknownError);
      }
      let mut argc: i32 = 0;
      let argv = CommandLineToArgvW(cmd_line, &mut argc);
      if argv.is_null() {
        return Err(NativeError::UnknownError);
      }
      let mut args = Vec::with_capacity(argc as usize);
      let argv_slice = core::slice::from_raw_parts(argv, argc as usize);

      for arg in argv_slice {
        let ptr = arg.as_ptr(); // extract *mut u16 pointer
        if ptr.is_null() {
          continue;
        }

        // find null terminator for the UTF-16 String
        let mut len: usize = 0;
        while *ptr.add(len) != 0 {
          len += 1;
        }

        let utf16_slice = core::slice::from_raw_parts(ptr, len);
        args.push(String::from_utf16_lossy(utf16_slice));
      }

      LocalFree(Some(HLOCAL(argv as *mut core::ffi::c_void)));
      Ok(args)
    }
  }
  #[cfg(target_os = "linux")]
  {
    use crate::os::{
      NativeError,
      fs::{PathBuf, read},
    };

    let path = PathBuf::from("/proc/self/cmdline");
    let bytes = read(path.as_ref()).map_err(|_| NativeError::UnknownError)?;
    if bytes.is_empty() {
      return Ok(Vec::new());
    }

    let mut args = Vec::new();
    // /proc/self/cmdline arguments are \0 separated
    let mut pieces: Vec<&[u8]> = bytes.split(|b| *b == 0).collect();
    // the very last element will be empty, discard it
    if pieces.last().map(|s| s.is_empty()) == Some(true) {
      pieces.pop();
    }

    for arg_bytes in pieces {
      args.push(String::from_utf8_lossy(arg_bytes).into_owned());
    }

    Ok(args)
  }
  #[cfg(target_os = "macos")]
  {
    use crate::os::NativeError;
    unsafe {
      let argc_ptr = libc::_NSGetArgc();
      let argv_ptr = libc::_NSGetArgv();

      if argc_ptr.is_null() || argv_ptr.is_null() {
        return Err(NativeError::UnknownError);
      }

      // Dereference the pointers to get the actual count and array
      let argc = *argc_ptr;
      let argv = *argv_ptr;

      if argv.is_null() {
        return Err(NativeError::UnknownError);
      }

      let mut args = Vec::with_capacity(argc as usize);

      // argv is now *mut *mut c_char, so this makes a slice of *mut c_char
      let argv_slice = core::slice::from_raw_parts(argv, argc as usize);

      for &arg in argv_slice {
        if arg.is_null() {
          continue;
        }

        // Find the null terminator
        let mut len = 0;
        while *arg.add(len) != 0 {
          len += 1;
        }

        // arg is *mut c_char, so we can cast it directly to *const u8
        let utf8_slice = core::slice::from_raw_parts(arg.cast::<u8>(), len);
        args.push(String::from_utf8_lossy(utf8_slice).into_owned());
      }

      Ok(args)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_args() {
    let cmd_args = args().expect("Failed to get command line arguments");
    // At least the executable name should be present in args
    assert!(!cmd_args.is_empty());

    // The first argument is typically the path to the executable
    println!("Executable arg: {}", cmd_args[0]);
  }
}
