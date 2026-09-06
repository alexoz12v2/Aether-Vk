//! debug_perf: Linux-only call-site frequency tracer.
//!
//! Tracks how often `get_monotonic_time()` (→ clock_gettime) and
//! `vkGetSemaphoreCounterValue` are called, and periodically dumps
//! a backtrace + call-rate report to stderr.
//!
//! Compiled **only** when `debug_assertions` is set AND `target_os = "linux"`.
//! Every public symbol is a no-op stub on other configurations.

use aethervk_oshal_rlib as oshal;

#[cfg(all(debug_assertions, target_os = "linux"))]
mod inner {
  use super::*;
  use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

  pub static CLOCK_CALL_COUNT: AtomicU64 = AtomicU64::new(0);
  pub static SEM_POLL_COUNT: AtomicU64 = AtomicU64::new(0);
  static LAST_REPORT_US: AtomicU64 = AtomicU64::new(0);
  static REPORTING: AtomicBool = AtomicBool::new(false);

  /// Dump stats + a backtrace every 2 seconds.
  const REPORT_INTERVAL_US: u64 = 2_000_000;
  /// Sample one backtrace every N clock calls to limit tracing overhead.
  const BACKTRACE_SAMPLE_EVERY: u64 = 50_000;

  pub fn on_clock_call(now_us: oshal::os::time::timeus_t) {
    let count = CLOCK_CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    if count % BACKTRACE_SAMPLE_EVERY == 0 {
      maybe_dump_report(now_us as u64);
    }
  }

  pub fn on_semaphore_poll() {
    SEM_POLL_COUNT.fetch_add(1, Ordering::Relaxed);
  }

  fn maybe_dump_report(now_us: u64) {
    let last = LAST_REPORT_US.load(Ordering::Relaxed);
    if now_us.wrapping_sub(last) < REPORT_INTERVAL_US {
      return;
    }
    // CAS prevents concurrent reporters.
    if REPORTING
      .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
      .is_err()
    {
      return;
    }
    if LAST_REPORT_US
      .compare_exchange(last, now_us, Ordering::AcqRel, Ordering::Relaxed)
      .is_err()
    {
      REPORTING.store(false, Ordering::Release);
      return;
    }

    let clk = CLOCK_CALL_COUNT.swap(0, Ordering::Relaxed);
    let sem = SEM_POLL_COUNT.swap(0, Ordering::Relaxed);

    aethervk_oshal_rlib::log!(
      "[debug_perf] clock_gettime calls/2s={clk}  semaphore_polls/2s={sem}"
    );

    // Capture backtrace via libc — works without `std`, no allocation.
    const MAX_FRAMES: usize = 32;
    let mut frames = [core::ptr::null_mut::<libc::c_void>(); MAX_FRAMES];
    let n = unsafe { libc::backtrace(frames.as_mut_ptr(), MAX_FRAMES as libc::c_int) } as usize;

    aethervk_oshal_rlib::log!("[debug_perf] backtrace ({n} frames) -> see stderr");
    if n > 0 {
      // Write symbolised frames directly to fd 2 (stderr), no heap needed.
      unsafe { libc::backtrace_symbols_fd(frames.as_ptr(), n as libc::c_int, 2) };
    }

    REPORTING.store(false, Ordering::Release);
  }
}

// -- Public API ---------------------------------------------------------------

/// Drop-in replacement for `get_monotonic_time()` that also records a call.
///
/// Shadow the import at the top of a function:
/// ```ignore
/// #[cfg(all(debug_assertions, target_os = "linux"))]
/// use crate::simulation_api::debug_perf::traced_get_monotonic_time as get_monotonic_time;
/// #[cfg(not(all(debug_assertions, target_os = "linux")))]
/// use oshal::os::time::get_monotonic_time;
/// ```
#[inline(always)]
pub fn traced_get_monotonic_time() -> oshal::os::time::timeus_t {
  let t = oshal::os::time::get_monotonic_time();
  #[cfg(all(debug_assertions, target_os = "linux"))]
  inner::on_clock_call(t);
  t
}

/// Call once per `vkGetSemaphoreCounterValue` invocation to count polls.
#[inline(always)]
pub fn traced_semaphore_poll() {
  #[cfg(all(debug_assertions, target_os = "linux"))]
  inner::on_semaphore_poll();
}