//! Native OS functions.

use core::time::Duration;

#[cfg(any(unix, target_os = "macos"))]
use libc;
#[cfg(windows)]
use windows::Win32::System::Threading::{GetCurrentThreadId, Sleep, SwitchToThread};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// TODO: Document this item
pub struct ThreadId(u64);

pub mod this_thread {
  use super::*;

  /// TODO: Document this item
  pub fn id() -> ThreadId {
    let id: ThreadId;
    #[cfg(any(unix, target_os = "macos"))]
    {
      id = ThreadId(unsafe { libc::pthread_self() } as _);
    }
    #[cfg(windows)]
    {
      id = ThreadId(unsafe { GetCurrentThreadId() } as _);
    }

    id
  }

  /// different than [`core::hint::spin_loop`], which instead inserts instructions such as pause.
  pub fn yield_now() {
    #[cfg(target_family = "unix")]
    {
      unsafe { libc::sched_yield() };
    }
    #[cfg(windows)]
    {
      let _ = unsafe { SwitchToThread() };
    }
  }

  /// TODO: Document this item
  pub fn sleep_for(duration: Duration) {
    #[cfg(windows)]
    {
      let ms = duration.as_millis();
      if ms == 0 {
        return;
      }
      // Sleep takes a u32. Cap at u32::MAX.
      let sleep_ms = if ms > u32::MAX as u128 {
        u32::MAX
      } else {
        ms as u32
      };
      unsafe { Sleep(sleep_ms) };
    }

    #[cfg(any(unix, target_os = "macos"))]
    {
      let mut req = libc::timespec {
        tv_sec: duration.as_secs() as libc::time_t,
        tv_nsec: duration.subsec_nanos() as libc::c_long,
      };

      // Loop until sleep is complete, as nanosleep can be interrupted.
      loop {
        let mut rem = libc::timespec {
          tv_sec: 0,
          tv_nsec: 0,
        };
        let ret = unsafe { libc::nanosleep(&req, &mut rem) };
        if ret == 0 {
          break; // Sleep completed.
        } else {
          // Interrupted, sleep for the remaining time.
          req = rem;
        }
      }
    }
  }
}
