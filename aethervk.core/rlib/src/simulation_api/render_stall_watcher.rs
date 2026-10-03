//! render_stall_watcher.rs — Debug-only render-stall watcher thread.
//!
//! Enabled only on Linux debug builds (`#[cfg(all(target_os = "linux", debug_assertions))]`).
//!
//! The watcher runs as a dedicated OS thread named `"aethervk-stall-watch"`. It polls:
//! - [`LAST_SUBMIT_NS`] from `hooks.rs` — updated atomically on every `vkQueueSubmit`
//! - A shared `simulation_active` [`AtomicBool`] — `true` while any scene simulation is running
//!
//! If `simulation_active` is `true` and no `vkQueueSubmit` has occurred for
//! [`STALL_THRESHOLD_NS`] (3 s), it calls
//! [`dump_all_thread_backtraces_gdb`](aethervk_oshal_rlib::os::debug::dump_all_thread_backtraces_gdb)
//! once (rate-limited to at most once per 10 s), which forks GDB and logs `thread apply all bt`.

#[cfg(all(target_os = "linux", debug_assertions))]
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(all(target_os = "linux", debug_assertions))]
use alloc::sync::Arc;

/// No `vkQueueSubmit` for this long while simulation is active → trigger GDB dump.
#[cfg(all(target_os = "linux", debug_assertions))]
const STALL_THRESHOLD_NS: u64 = 3_000_000_000; // 3 s

/// How often the watcher thread wakes and checks the timestamp.
#[cfg(all(target_os = "linux", debug_assertions))]
const POLL_INTERVAL_US: u64 = 250_000; // 250 ms

/// Minimum time between successive GDB dumps (rate-limit).
#[cfg(all(target_os = "linux", debug_assertions))]
const DUMP_COOLDOWN_NS: u64 = 10_000_000_000; // 10 s

/// Returns the current CLOCK_MONOTONIC time in nanoseconds.
#[cfg(all(target_os = "linux", debug_assertions))]
fn monotonic_ns() -> u64 {
  let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
  unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
  (ts.tv_sec as u64)
    .wrapping_mul(1_000_000_000)
    .wrapping_add(ts.tv_nsec as u64)
}

/// Spawns the render-stall watcher thread.
///
/// # Parameters
/// - `simulation_active`: shared flag, `true` whenever any scene simulation is running.
///   The watcher only arms the stall detector while this is `true`.
/// - `shutdown`: set to `true` to stop the watcher thread gracefully.
///
/// No-op on non-Linux platforms or release builds.
#[cfg(all(target_os = "linux", debug_assertions))]
pub fn spawn_render_stall_watcher(
  simulation_active: Arc<AtomicBool>,
  shutdown: Arc<AtomicBool>,
) {
  use crate::gpu_backends::vulkan::device::hooks::LAST_SUBMIT_NS;

  let handle = aethervk_oshal_rlib::os::thread::Builder::new()
    .name(alloc::string::String::from("aethervk-stall-watch"))
    .spawn(move || {
      aethervk_oshal_rlib::log!("[StallWatcher] started (threshold=3 s, poll=250 ms)");

      // Seed LAST_SUBMIT_NS with now so we don't fire immediately before the first GPU submit.
      let seed = monotonic_ns();
      LAST_SUBMIT_NS.fetch_max(seed, Ordering::Relaxed);

      let mut last_dump_ns: u64 = 0;

      loop {
        if shutdown.load(Ordering::Acquire) {
          aethervk_oshal_rlib::log!("[StallWatcher] shutting down");
          break;
        }

        aethervk_oshal_rlib::os::native::this_thread::sleep_for(
          core::time::Duration::from_micros(POLL_INTERVAL_US),
        );

        if !simulation_active.load(Ordering::Acquire) {
          continue;
        }

        let now = monotonic_ns();
        let last_submit = LAST_SUBMIT_NS.load(Ordering::Relaxed);

        let stalled = last_submit > 0
          && now.saturating_sub(last_submit) >= STALL_THRESHOLD_NS;
        let cooled_down = now.saturating_sub(last_dump_ns) >= DUMP_COOLDOWN_NS;

        if stalled && cooled_down {
          aethervk_oshal_rlib::log!(
            "[StallWatcher] No vkQueueSubmit for ≥3 s while simulation active \
             (last submit {:.1} s ago) — triggering GDB backtrace dump",
            now.saturating_sub(last_submit) as f64 / 1_000_000_000.0,
          );
          last_dump_ns = now;
          aethervk_oshal_rlib::os::debug::dump_all_thread_backtraces_gdb();
        }
      }
    });

  if let Err(e) = handle {
    aethervk_oshal_rlib::log!("[StallWatcher] failed to spawn watcher thread: {:?}", e);
  }
}

/// No-op on non-Linux platforms or release builds.
#[cfg(not(all(target_os = "linux", debug_assertions)))]
pub fn spawn_render_stall_watcher(
  _simulation_active: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
  _shutdown: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
) {
}
