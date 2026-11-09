//! Tests only: keeps the dust profiler's pipelines out of every other test's device.
//!
//! The profiler request ([`physics::DUST_PROFILE_OVERRIDE`]) is a process-wide flag read at device
//! creation. A device created while a profiler test has it set builds every `.prof` dust
//! pipeline with the capture-statistics / internal-representation flags. One such compile kept
//! the NVIDIA compiler spinning, with the other test threads waiting on its lock.
//!
//! The gate is a shared/exclusive lock on one file in the temp directory:
//! - a profiler session takes it exclusively;
//! - every other device creation ([`super::VulkanRenderContext`]'s `init_device`) takes it
//!   shared.
//!
//! A file lock is a kernel object (`LockFileEx` on Windows, `flock` on Linux and macOS), so it
//! works between the threads of one libtest process and between nextest's one-process-per-test
//! runs. The kernel releases it when the holder's process dies.
//!
//! The session's own device creations skip the shared lock. The session marks its thread, and
//! `setup_render_frontend_for_tests` carries the mark into the thread that creates the device.

use crate::gpu_backends::vulkan::physics::DUST_PROFILE_OVERRIDE;
use core::cell::Cell;
use core::sync::atomic::Ordering;
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::time::{Duration, Instant};

std::thread_local! {
  static HOLDS_PROFILE_GATE: Cell<bool> = const { Cell::new(false) };
}

/// The lock file, shared by every test process of this user.
pub fn gate_path() -> PathBuf {
  std::env::temp_dir().join("aethervk-dust-profile.lock")
}

fn open_gate() -> File {
  let path = gate_path();
  OpenOptions::new()
    .create(true)
    .truncate(false)
    .read(true)
    .write(true)
    .open(&path)
    .unwrap_or_else(|e| panic!("dust profile gate {}: {e}", path.display()))
}

/// Waits for the lock by polling, so a deadlock fails the test with a message instead of hanging
/// it. 30 min is longer than any profiler test.
fn acquire(file: &File, exclusive: bool) {
  const LIMIT: Duration = Duration::from_secs(30 * 60);
  let start = Instant::now();
  loop {
    let r = if exclusive { file.try_lock() } else { file.try_lock_shared() };
    match r {
      Ok(()) => return,
      Err(std::fs::TryLockError::WouldBlock) => {
        assert!(
          start.elapsed() < LIMIT,
          "dust profile gate ({}) still held after {LIMIT:?} ({} lock wanted)",
          gate_path().display(),
          if exclusive { "exclusive" } else { "shared" }
        );
        std::thread::sleep(Duration::from_millis(20));
      }
      Err(std::fs::TryLockError::Error(e)) => panic!("dust profile gate {}: {e}", gate_path().display()),
    }
  }
}

/// The profiler request for one test, held exclusively across threads and processes.
///
/// On drop it clears the request and the thread mark, then releases the lock. Drop also runs
/// when the test panics.
pub struct DustProfileSession {
  // dropped in this order: the gate, then the test's device slot
  _file: File,
  _slot: std::sync::Arc<super::test_device_slots::DeviceSlot>,
}

impl DustProfileSession {
  pub fn begin() -> Self {
    // the test's device slot first (`test_device_slots` lock order): plain device creations hold a
    // slot while they wait for the gate, so a session that waited for a slot under the gate could
    // wait for ever
    super::test_gpu_watch::name_gpu_test_thread();
    let slot = super::test_device_slots::acquire();
    let file = open_gate();
    acquire(&file, true);
    HOLDS_PROFILE_GATE.with(|h| h.set(true));
    DUST_PROFILE_OVERRIDE.store(true, Ordering::SeqCst);
    Self { _file: file, _slot: slot }
  }
}

impl Drop for DustProfileSession {
  fn drop(&mut self) {
    DUST_PROFILE_OVERRIDE.store(false, Ordering::SeqCst);
    HOLDS_PROFILE_GATE.with(|h| h.set(false));
    // the File drops after this body, which releases the lock
  }
}

/// Whether this thread runs inside a profiler session (its device creations skip the gate).
pub fn holds_session() -> bool {
  HOLDS_PROFILE_GATE.with(|h| h.get())
}

/// Marks a helper thread as part of its spawner's session (`setup_render_frontend_for_tests`
/// creates the device on its own thread).
pub fn adopt_session(holds: bool) {
  HOLDS_PROFILE_GATE.with(|h| h.set(holds));
}

/// The shared side, taken around every device creation outside a session. While it is held, no
/// session can start in any process. Returns `None` inside a session.
pub fn plain_device_guard() -> Option<File> {
  if holds_session() {
    return None;
  }
  let file = open_gate();
  acquire(&file, false);
  Some(file)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The exclusion the gate exists for, between threads and between processes:
  /// - a plain device created while a session is open waits for the session to end;
  /// - that device has none of the profiler's pipelines;
  /// - a second process that wants the shared side waits too.
  #[test]
  fn profile_gate_excludes_threads_and_processes() {
    use crate::gpu_backends::vulkan::device::Device;
    use crate::gpu_backends::vulkan::device::test_utils::{setup_assets_dir, setup_render_frontend_for_tests};
    const HOLD: Duration = Duration::from_secs(3);
    setup_assets_dir();

    let prof_of = |frontend: &crate::gpu::RenderFrontend, handle| {
      frontend
        .with_device(handle, |d| {
          let d: &Device = d.as_any().downcast_ref().unwrap();
          let p = &d.kernels.pipelines;
          Ok((p.dust_splat_prof.is_some(), p.dust_draw_volume_mono_prof.is_some(), p.dust_draw_volume_prof.is_some()))
        })
        .unwrap()
    };

    // control: the session's own device (created on setup's thread, which inherits the mark) does
    // build the profiler pipelines, so the plain device's `None`s below are not a device limit
    let session = DustProfileSession::begin();
    let (_pool, frontend, handle, _) = setup_render_frontend_for_tests(false);
    let own = prof_of(&frontend, handle);
    drop(frontend);
    std::println!("[profile gate] the session's own device: prof pipelines {own:?}");
    let has_clock = own.0;
    if !has_clock {
      std::println!("[profile gate] no VK_KHR_shader_clock: the pipeline check below is vacuous");
    }

    // threads: B creates a device while A (this thread) holds the session
    let opened = Instant::now();
    let b = std::thread::spawn(move || {
      let (_pool, frontend, handle, _) = setup_render_frontend_for_tests(false);
      let created = Instant::now();
      (created, prof_of(&frontend, handle))
    });
    std::thread::sleep(HOLD);
    let dropped = Instant::now();
    drop(session);
    let (created, prof) = b.join().unwrap();
    std::println!(
      "[profile gate] session held {:?}; the plain device was created {:?} after the drop; prof pipelines {prof:?}",
      dropped - opened,
      created.saturating_duration_since(dropped)
    );
    assert!(created >= dropped, "the plain device was created while the session was open");
    assert_eq!(prof, (false, false, false), "a plain device built profiler pipelines");

    // processes: a child of this binary takes the shared side while the session is open
    let session = DustProfileSession::begin();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
      .args([
        "gpu_backends::vulkan::test_profile_gate::tests::profile_gate_child_takes_the_shared_side",
        "--exact",
        "--ignored",
        "--nocapture",
      ])
      .env_remove("NEXTEST")
      .stdout(std::process::Stdio::piped())
      .spawn()
      .unwrap();
    std::thread::sleep(HOLD);
    let early = child.try_wait().unwrap();
    drop(session);
    let released = Instant::now();
    let status = loop {
      if let Some(s) = child.try_wait().unwrap() {
        break s;
      }
      if released.elapsed() > Duration::from_secs(60) {
        let _ = child.kill();
        panic!("the child process never got the shared side after the session ended");
      }
      std::thread::sleep(Duration::from_millis(20));
    };
    let mut out = std::string::String::new();
    std::io::Read::read_to_string(child.stdout.as_mut().unwrap(), &mut out).unwrap();
    std::println!("[profile gate] child: waited {HOLD:?} while held, exited {:?} after the drop", released.elapsed());
    assert!(early.is_none(), "the child got the shared side while the session was open: {early:?}\n{out}");
    assert!(status.success(), "child failed: {status:?}\n{out}");
    assert!(out.contains("1 passed"), "the child did not run its test:\n{out}");
  }

  /// The child side of `profile_gate_excludes_threads_and_processes`. It runs only when that test
  /// launches it (`--ignored`).
  #[test]
  #[ignore = "launched as a child process by profile_gate_excludes_threads_and_processes"]
  fn profile_gate_child_takes_the_shared_side() {
    let guard = plain_device_guard();
    assert!(guard.is_some());
  }
}
