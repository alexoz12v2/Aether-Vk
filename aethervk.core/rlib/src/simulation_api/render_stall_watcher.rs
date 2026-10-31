//! render_stall_watcher.rs — Debug-only render-stall watcher thread.
//!
//! Enabled only on Linux and Windows debug builds (`#[cfg(all(any(target_os = "linux", target_os =
//! "windows"), debug_assertions))]`).
//!
//! The watcher runs as a dedicated OS thread named `"aethervk-stall-watch"`. It polls:
//! - [`LAST_SUBMIT_NS`] from `hooks.rs` — updated atomically on every `vkQueueSubmit`
//! - A shared `simulation_active` [`AtomicBool`] — `true` while any scene simulation is running
//!
//! If `simulation_active` is `true` and no `vkQueueSubmit` has occurred for
//! [`STALL_THRESHOLD_NS`] (3 s), it dumps every thread's stack once (rate-limited to at most once
//! per 10 s):
//! - Linux: [`dump_all_thread_backtraces_gdb`](aethervk_oshal_rlib::os::debug::dump_all_thread_backtraces_gdb),
//!   which forks GDB and logs `thread apply all bt`;
//! - Windows: [`capture_stall`](aethervk_oshal_rlib::os::debug::crash_dump::capture_stall), a
//!   minidump in `%LOCALAPPDATA%\Aether-Vk\logs` and its thread stacks read back by cdb, logged.

#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
use alloc::sync::Arc;
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
use core::sync::atomic::{AtomicBool, Ordering};

/// No `vkQueueSubmit` for this long while simulation is active → trigger the stack dump.
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
const STALL_THRESHOLD_NS: u64 = 3_000_000_000; // 3 s

/// How often the watcher thread wakes and checks the timestamp.
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
const POLL_INTERVAL_US: u64 = 250_000; // 250 ms

/// Minimum time between successive stack dumps (rate-limit).
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
const DUMP_COOLDOWN_NS: u64 = 10_000_000_000; // 10 s

/// The monotonic clock in nanoseconds (CLOCK_MONOTONIC on Linux, QueryPerformanceCounter on
/// Windows), as [`LAST_SUBMIT_NS`] stores it.
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
pub(crate) fn monotonic_ns() -> u64 {
  (aethervk_oshal_rlib::os::time::get_monotonic_time() as u64).wrapping_mul(1_000)
}

/// Dumps every thread's stack to the log (and a file): gdb on Linux, a minidump read by cdb on
/// Windows.
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
fn dump_all_thread_stacks() {
  #[cfg(target_os = "linux")]
  aethervk_oshal_rlib::os::debug::dump_all_thread_backtraces_gdb();
  #[cfg(target_os = "windows")]
  for line in aethervk_oshal_rlib::os::debug::crash_dump::capture_stall() {
    aethervk_oshal_rlib::log!("{}", line);
  }
}

/// Spawns the render-stall watcher thread.
///
/// # Parameters
/// - `simulation_active`: shared flag, `true` whenever any scene simulation is running.
///   The watcher only arms the stall detector while this is `true`.
/// - `shutdown`: set to `true` to stop the watcher thread gracefully.
///
/// No-op on other platforms or release builds.
#[cfg(all(any(target_os = "linux", target_os = "windows"), debug_assertions))]
pub fn spawn_render_stall_watcher(simulation_active: Arc<AtomicBool>, shutdown: Arc<AtomicBool>) {
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

        aethervk_oshal_rlib::os::native::this_thread::sleep_for(core::time::Duration::from_micros(
          POLL_INTERVAL_US,
        ));

        if !simulation_active.load(Ordering::Acquire) {
          continue;
        }

        let now = monotonic_ns();
        let last_submit = LAST_SUBMIT_NS.load(Ordering::Relaxed);

        let stalled = last_submit > 0 && now.saturating_sub(last_submit) >= STALL_THRESHOLD_NS;
        let cooled_down = now.saturating_sub(last_dump_ns) >= DUMP_COOLDOWN_NS;

        if stalled && cooled_down {
          aethervk_oshal_rlib::log!(
            "[StallWatcher] No vkQueueSubmit for ≥3 s while simulation active \
             (last submit {:.1} s ago) — dumping every thread's stack",
            now.saturating_sub(last_submit) as f64 / 1_000_000_000.0,
          );
          last_dump_ns = now;
          dump_all_thread_stacks();
        }
      }
    });

  if let Err(e) = handle {
    aethervk_oshal_rlib::log!("[StallWatcher] failed to spawn watcher thread: {:?}", e);
  }
}

/// No-op on other platforms or release builds.
#[cfg(not(all(any(target_os = "linux", target_os = "windows"), debug_assertions)))]
pub fn spawn_render_stall_watcher(
  _simulation_active: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
  _shutdown: alloc::sync::Arc<core::sync::atomic::AtomicBool>,
) {
}

#[cfg(all(test, target_os = "windows", debug_assertions))]
mod tests {
  use super::*;

  /// A forced stall: the watcher armed with no `vkQueueSubmit` writes the stall dump and its stacks
  /// file after the 3 s threshold. Needs cdb and writes to the Aether-Vk log folder, so it is
  /// opt-in (`-- --ignored`); run it in its own process (nextest), as another test's submits would
  /// keep the watcher quiet.
  #[test]
  #[ignore = "needs cdb.exe; writes a minidump to the Aether-Vk log folder"]
  fn the_watcher_dumps_a_forced_stall() {
    use aethervk_oshal_rlib::os::debug::crash_dump;
    let dir = crash_dump::log_dir().expect("LOCALAPPDATA");
    std::fs::create_dir_all(&dir).unwrap();
    let stacks = std::format!("{dir}\\aethervk-stall-{}.txt", crash_dump::process_id());
    let _ = std::fs::remove_file(&stacks);
    let (active, shutdown) = (Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(false)));
    spawn_render_stall_watcher(Arc::clone(&active), Arc::clone(&shutdown));
    let t0 = std::time::Instant::now();
    // (the file appears empty, then cdb's output lands in one write)
    let (mut text, mut first) = (std::string::String::new(), None);
    while !text.contains("aethervk-stall-watch") && t0.elapsed().as_secs() < 60 {
      std::thread::sleep(core::time::Duration::from_millis(250));
      text = std::fs::read_to_string(&stacks).unwrap_or_default();
      if first.is_none() && std::path::Path::new(&stacks).exists() {
        first = Some(t0.elapsed().as_secs_f64());
      }
    }
    shutdown.store(true, Ordering::Release);
    assert!(first.is_some_and(|t| t >= 2.5), "dumped before the 3 s threshold ({first:?} s)");
    assert!(text.contains("aethervk-stall-watch"), "no stacks file, or the watcher thread is missing from it");
  }
}
