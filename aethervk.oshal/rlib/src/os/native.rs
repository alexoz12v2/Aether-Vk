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

  /// Blocks the calling thread for at least `duration` (zero returns at once).
  ///
  /// Windows: a high-resolution waitable timer (~0.5 ms granularity, Windows 10 1803+), else
  /// `Sleep` rounded **up** to whole milliseconds. `Sleep` alone cannot express sub-millisecond
  /// waits, and truncating them to `Sleep(0)`/no-op turned every idle loop that sleeps 50–500 µs
  /// (render thread, logic thread) into a busy spin: a full core at idle (feedback #21).
  pub fn sleep_for(duration: Duration) {
    #[cfg(windows)]
    {
      if duration.is_zero() {
        return;
      }
      if !high_resolution_wait(duration) {
        // Sleep takes a u32. Round up, cap at u32::MAX - 1 (u32::MAX is INFINITE).
        let ms = duration.as_nanos().div_ceil(1_000_000);
        let sleep_ms = if ms >= u32::MAX as u128 {
          u32::MAX - 1
        } else {
          ms as u32
        };
        unsafe { Sleep(sleep_ms) };
      }
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

  /// Waits `duration` on a one-shot high-resolution waitable timer. False when the timer is
  /// unavailable (Windows before 10 1803) or fails, so the caller falls back to `Sleep`.
  #[cfg(windows)]
  fn high_resolution_wait(duration: Duration) -> bool {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::System::Threading::{
      CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateWaitableTimerExW, INFINITE, SetWaitableTimer,
      TIMER_ALL_ACCESS, WaitForSingleObject,
    };
    // negative = relative, in 100 ns units; at least one unit
    let ticks = duration.as_nanos().div_ceil(100).clamp(1, i64::MAX as u128) as i64;
    let due: i64 = -ticks;
    unsafe {
      let Ok(timer) = CreateWaitableTimerExW(
        None,
        windows::core::PCWSTR::null(),
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
        TIMER_ALL_ACCESS.0,
      ) else {
        return false;
      };
      let ok = SetWaitableTimer(timer, &due, 0, None, None, false).is_ok()
        && WaitForSingleObject(timer, INFINITE) == WAIT_OBJECT_0;
      let _ = CloseHandle(timer);
      ok
    }
  }
}

#[cfg(test)]
mod tests {
  extern crate std;
  use super::this_thread::sleep_for;
  use core::time::Duration;

  /// Sub-millisecond sleeps must block: on Windows they used to return at once, turning every
  /// idle loop that sleeps 50–500 µs into a busy spin.
  #[test]
  fn sub_millisecond_sleeps_actually_wait() {
    for us in [50_u64, 200, 500] {
      let n = 20;
      let t0 = std::time::Instant::now();
      for _ in 0..n {
        sleep_for(Duration::from_micros(us));
      }
      let each = t0.elapsed() / n;
      assert!(each >= Duration::from_micros(us), "sleep_for({us} µs) took only {each:?}");
      // generous: the Sleep fallback can round up to a timer tick
      assert!(each < Duration::from_millis(20), "sleep_for({us} µs) took {each:?}");
    }
  }

  #[test]
  fn zero_returns_at_once() {
    let t0 = std::time::Instant::now();
    sleep_for(Duration::ZERO);
    assert!(t0.elapsed() < Duration::from_millis(1));
  }
}
