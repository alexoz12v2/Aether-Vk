//! OS-level thread creation and management.

use alloc::boxed::Box;
use alloc::string::String;
use core::ffi::{c_void};
use core::ptr;

#[cfg(any(unix, target_os = "macos"))]
use libc;

#[cfg(windows)]
use windows::Win32::System::Threading::{CreateThread, WaitForSingleObject, WAIT_OBJECT_0};
#[cfg(windows)]
use windows::Win32::System::SystemServices::LPSECURITY_ATTRIBUTES;

use crate::os::ThreadingResult;

pub struct Thread {
  #[cfg(any(unix, target_os = "macos"))]
  native: libc::pthread_t,
  #[cfg(windows)]
  native: isize,
}

impl Thread {
  pub fn join(self) {
    #[cfg(any(unix, target_os = "macos"))]
    unsafe {
      let ret = libc::pthread_join(self.native, ptr::null_mut());
      assert_eq!(ret, 0);
    }

    #[cfg(windows)]
    unsafe {
      let res = WaitForSingleObject(self.native, u32::MAX);
      assert_eq!(res, WAIT_OBJECT_0);
    }
  }
}

pub struct Builder {
  name: Option<String>,
  stack_size: Option<usize>,
}

impl Builder {
  pub fn new() -> Builder {
    Builder {
      name: None,
      stack_size: None,
    }
  }

  pub fn name(mut self, name: String) -> Builder {
    self.name = Some(name);
    self
  }

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
      let native = CreateThread(
        LPSECURITY_ATTRIBUTES::default(),
        self.stack_size.unwrap_or(0),
        Some(thread_start),
        Some(Box::into_raw(main) as *mut c_void),
        0,
        None,
      );

      if let Ok(handle) = native {
        Ok(Thread {
          native: handle.0 as isize,
        })
      } else {
        Err(io::Error::last_os_error())
      }
    }
  }
}

pub fn spawn<F>(f: F) -> ThreadingResult<Thread>
where
  F: FnOnce(),
  F: Send + 'static,
{
  Builder::new().spawn(f)
}
