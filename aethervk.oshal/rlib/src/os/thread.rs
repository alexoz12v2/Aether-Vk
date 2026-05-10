//! OS-level thread creation and management.

use alloc::boxed::Box;
use alloc::string::String;
use core::ffi::c_void;

#[cfg(any(unix, target_os = "macos"))]
use core::ptr;

#[cfg(any(unix, target_os = "macos"))]
use libc;

#[cfg(windows)]
use windows::Win32::{
  Foundation::WAIT_OBJECT_0,
  Security::SECURITY_ATTRIBUTES,
  System::Threading::{CreateThread, WaitForSingleObject},
};

use crate::os::ThreadingResult;

/// TODO: Document this item
pub struct Thread {
  #[cfg(any(unix, target_os = "macos"))]
  native: libc::pthread_t,
  #[cfg(windows)]
  native: isize,
}

impl Thread {
  /// TODO: Document this item
  pub fn join(self) {
    #[cfg(any(unix, target_os = "macos"))]
    unsafe {
      let ret = libc::pthread_join(self.native, ptr::null_mut());
      assert_eq!(ret, 0);
    }

    #[cfg(windows)]
    unsafe {
      let res = WaitForSingleObject(
        windows::Win32::Foundation::HANDLE(self.native as *mut _),
        u32::MAX,
      );
      assert_eq!(res, WAIT_OBJECT_0);
    }
  }
}

/// TODO: Document this item
pub struct Builder {
  name: Option<String>,
  stack_size: Option<usize>,
}

impl Builder {
  /// TODO: Document this item
  pub fn new() -> Builder {
    Builder {
      name: None,
      stack_size: None,
    }
  }

  /// TODO: Document this item
  pub fn name(mut self, name: String) -> Builder {
    self.name = Some(name);
    self
  }

  /// TODO: Document this item
  pub fn stack_size(mut self, size: usize) -> Builder {
    self.stack_size = Some(size);
    self
  }

  /// TODO: Document this item
  pub fn spawn<F>(self, f: F) -> ThreadingResult<Thread>
  where
    F: FnOnce(),
    F: Send + 'static,
  {
    unsafe { self.spawn_unsafe(f) }
  }

  unsafe fn spawn_unsafe<F>(self, f: F) -> ThreadingResult<Thread>
  where
    F: FnOnce(),
    F: Send + 'static,
  {
    let main: Box<dyn FnOnce()> = Box::new(f);
    let main: Box<Box<dyn FnOnce()>> = Box::new(main);

    #[cfg(any(unix, target_os = "macos"))]
    {
      extern "C" fn thread_start(main: *mut c_void) -> *mut c_void {
        unsafe {
          Box::from_raw(main as *mut Box<dyn FnOnce()>)();
        }
        ptr::null_mut()
      }

      let mut native: libc::pthread_t = 0;
      let mut attr: libc::pthread_attr_t = unsafe { core::mem::zeroed() };
      unsafe { libc::pthread_attr_init(&mut attr) };

      if let Some(stack_size) = self.stack_size {
        unsafe { libc::pthread_attr_setstacksize(&mut attr, stack_size) };
      }

      let ret = unsafe {
        libc::pthread_create(
          &mut native,
          &attr,
          thread_start,
          Box::into_raw(main) as *mut c_void,
        )
      };
      unsafe { libc::pthread_attr_destroy(&mut attr) };

      if ret != 0 {
        use crate::os::NativeError;

        return Err(NativeError::threading_from_raw_os_error(ret));
      }

      if let Some(name) = self.name {
        // TODO: figure out how to set thread name in apple, which must be done from calling
        // thread as `pthread_setname_np` takes only one argument (name) for that
        let name_c = alloc::ffi::CString::new(name).unwrap();
        #[cfg(not(target_vendor = "apple"))]
        {
          unsafe { libc::pthread_setname_np(native, name_c.as_ptr() as *const c_char) };
        }
      }

      Ok(Thread { native })
    }

    #[cfg(windows)]
    {
      extern "system" fn thread_start(main: *mut c_void) -> u32 {
        unsafe {
          Box::from_raw(main as *mut Box<dyn FnOnce()>)();
        }
        0
      }

      // TODO: _beginthreadex
      let native = unsafe {
        let sa = SECURITY_ATTRIBUTES::default();
        CreateThread(
          Some(core::ptr::from_ref(&sa)),
          self.stack_size.unwrap_or(0),
          Some(thread_start),
          Some(Box::into_raw(main) as *mut c_void),
          windows::Win32::System::Threading::THREAD_CREATION_FLAGS::default(),
          None,
        )
      };

      if let Ok(handle) = native {
        Ok(Thread {
          native: handle.0 as isize,
        })
      } else {
        Err(crate::os::ThreadingError::Unknown)
      }
    }
  }
}

/// TODO: Document this item
pub fn spawn<F>(f: F) -> ThreadingResult<Thread>
where
  F: FnOnce(),
  F: Send + 'static,
{
  Builder::new().spawn(f)
}

#[cfg(test)]
mod tests {
  use super::*;
  use alloc::sync::Arc;
  use core::sync::atomic::{AtomicBool, Ordering};

  #[test]
  fn test_thread_spawn_and_join() {
    let flag = Arc::new(AtomicBool::new(false));
    let flag_clone = Arc::clone(&flag);

    let thread = spawn(move || {
      flag_clone.store(true, Ordering::SeqCst);
    })
    .expect("Failed to spawn thread");

    thread.join();
    assert!(flag.load(Ordering::SeqCst));
  }

  #[test]
  fn test_thread_builder() {
    let thread = Builder::new()
      .name(String::from("test_thread"))
      .stack_size(1024 * 1024)
      .spawn(|| {
        // do nothing
      })
      .expect("Failed to spawn thread with builder");

    thread.join();
  }
}
