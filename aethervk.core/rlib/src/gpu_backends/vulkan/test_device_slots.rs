//! Tests only: caps the live Vulkan devices of every test process, and creates them one at a time.
//!
//! **Why.** A libtest run at the default thread count kept dozens of devices alive at once. Each
//! device pins driver memory outside the process's memory cgroup:
//! - the machine ran out of memory;
//! - device creation failed (VK_ERROR_INITIALIZATION_FAILED);
//! - devices were lost with no Xid.
//! Creating a device while another is created or torn down also made the NVIDIA driver raise an
//! Xid 32 inside `vkCreateDevice` itself.
//!
//! **Slots.** `N` lock files in the temp directory (`aethervk-device-slot-<i>.lock`; `N` from
//! `AETHERVK_TEST_DEVICE_SLOTS`, default [`DEFAULT_SLOTS`]):
//! - a device takes a free slot before it is created and holds it until it is destroyed (the slot
//!   lives in the device's `VulkanCore`, which drops after its device);
//! - the locks are kernel objects (`flock`, `LockFileEx`), so the cap holds across threads and
//!   across processes (nextest), and the kernel releases a slot when its holder's process dies.
//!
//! **One slot per test.** All the devices of one test share its slot, so a test that opens a second
//! device never waits for itself. A test is recognised by its short OS thread name
//! (`test_gpu_watch`), which the threads it spawns inherit on Linux and macOS. A thread without a
//! test name (Windows helper threads) takes a slot per device.
//!
//! **Creation lock.** One more lock file (`aethervk-device-create.lock`) is held exclusively around
//! device creation, so creations never overlap.
//!
//! **Lock order:** slot, then the dust profile gate (`test_profile_gate`), then the creation lock.
//! A profiler session takes its slot before its exclusive gate, so no wait forms a cycle.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::string::String;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};
use std::vec::Vec;

/// Live devices allowed at once over all test processes, unless `AETHERVK_TEST_DEVICE_SLOTS` says
/// otherwise.
pub const DEFAULT_SLOTS: usize = 8;

/// A test's longest legitimate wait for a slot: tests can hold their device for 15 min, so the
/// limit only catches a deadlock.
const SLOT_WAIT_LIMIT: Duration = Duration::from_secs(90 * 60);
const CREATE_WAIT_LIMIT: Duration = Duration::from_secs(30 * 60);

/// One held slot. The lock is released when the last device (and session) holding it drops.
pub struct DeviceSlot {
  _file: File,
  pub index: usize,
  key: Option<String>,
}

impl Drop for DeviceSlot {
  fn drop(&mut self) {
    aethervk_oshal_rlib::log!(
      "[device slots] slot {} released ({})",
      self.index,
      self.key.as_deref().unwrap_or("no test name")
    );
  }
}

/// The slots of live tests: (lock directory, test key) → slot. Entries whose slot dropped are
/// pruned on the next lookup.
static SHARED: parking_lot::Mutex<Vec<(PathBuf, String, Weak<DeviceSlot>)>> = parking_lot::Mutex::new(Vec::new());

/// The number of slots (`AETHERVK_TEST_DEVICE_SLOTS`, at least 1).
pub fn slot_count() -> usize {
  std::env::var("AETHERVK_TEST_DEVICE_SLOTS")
    .ok()
    .and_then(|v| v.trim().parse::<usize>().ok())
    .filter(|&n| n >= 1)
    .unwrap_or(DEFAULT_SLOTS)
}

fn open_lock(path: &Path) -> File {
  OpenOptions::new()
    .create(true)
    .truncate(false)
    .read(true)
    .write(true)
    .open(path)
    .unwrap_or_else(|e| panic!("device slot lock {}: {e}", path.display()))
}

/// The slot of the calling thread's test, taking a free one (and waiting for it) if the test holds
/// none. Called before every device creation.
pub fn acquire() -> Arc<DeviceSlot> {
  acquire_in(&std::env::temp_dir(), slot_count(), crate::gpu_backends::vulkan::test_gpu_watch::current_test_key())
}

/// [`acquire`] on the lock files `aethervk-device-slot-<0..n>.lock` in `dir`, for the test `key`.
pub fn acquire_in(dir: &Path, n: usize, key: Option<String>) -> Arc<DeviceSlot> {
  if let Some(k) = &key {
    let mut shared = SHARED.lock();
    shared.retain(|(_, _, w)| w.strong_count() > 0);
    if let Some(slot) = shared.iter().find(|(d, kk, _)| d == dir && kk == k).and_then(|(_, _, w)| w.upgrade()) {
      return slot;
    }
  }
  let start = Instant::now();
  let mut announced = false;
  loop {
    for index in 0..n {
      let file = open_lock(&dir.join(alloc::format!("aethervk-device-slot-{index}.lock")));
      match file.try_lock() {
        Ok(()) => {
          aethervk_oshal_rlib::log!(
            "[device slots] slot {index}/{n} taken ({}) after {:?}",
            key.as_deref().unwrap_or("no test name"),
            start.elapsed()
          );
          let slot = Arc::new(DeviceSlot { _file: file, index, key: key.clone() });
          if let Some(k) = key {
            SHARED.lock().push((dir.to_path_buf(), k, Arc::downgrade(&slot)));
          }
          return slot;
        }
        Err(std::fs::TryLockError::WouldBlock) => {}
        Err(std::fs::TryLockError::Error(e)) => panic!("device slot {index} in {}: {e}", dir.display()),
      }
    }
    if !announced {
      announced = true;
      aethervk_oshal_rlib::log!(
        "[device slots] all {n} slots in use: {} waits",
        key.as_deref().unwrap_or("a thread without a test name")
      );
    }
    assert!(
      start.elapsed() < SLOT_WAIT_LIMIT,
      "no free device slot in {} after {SLOT_WAIT_LIMIT:?} ({n} slots; a deadlock?)",
      dir.display()
    );
    std::thread::sleep(Duration::from_millis(50));
  }
}

/// The creation lock: held exclusively while one device is created, in any test process.
pub fn creation_guard() -> File {
  creation_guard_in(&std::env::temp_dir())
}

fn creation_guard_in(dir: &Path) -> File {
  let path = dir.join("aethervk-device-create.lock");
  let file = open_lock(&path);
  let start = Instant::now();
  loop {
    match file.try_lock() {
      Ok(()) => return file,
      Err(std::fs::TryLockError::WouldBlock) => {
        assert!(start.elapsed() < CREATE_WAIT_LIMIT, "device creation lock {} held for {CREATE_WAIT_LIMIT:?}", path.display());
        std::thread::sleep(Duration::from_millis(10));
      }
      Err(std::fs::TryLockError::Error(e)) => panic!("device creation lock {}: {e}", path.display()),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use core::sync::atomic::{AtomicUsize, Ordering};

  /// A lock directory of this test's own, so these checks never compete with the real slots.
  fn lab_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(alloc::format!("aethervk-device-slots-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
  }

  /// With 2 slots, 6 tests (threads with their own keys) never hold more than 2 at once, all of
  /// them finish, and two devices of one test share its slot without waiting while the others are
  /// all taken.
  #[test]
  fn device_slots_cap_threads_and_share_within_a_test() {
    let dir = lab_dir("threads");
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..6)
      .map(|t| {
        let (dir, live, peak) = (dir.clone(), live.clone(), peak.clone());
        std::thread::spawn(move || {
          let slot = acquire_in(&dir, 2, Some(alloc::format!("test{t}")));
          let now = live.fetch_add(1, Ordering::SeqCst) + 1;
          peak.fetch_max(now, Ordering::SeqCst);
          // the same test's second device: the same slot, at once
          let t0 = Instant::now();
          let again = acquire_in(&dir, 2, Some(alloc::format!("test{t}")));
          assert!(Arc::ptr_eq(&slot, &again), "a second device of one test took another slot");
          assert!(t0.elapsed() < Duration::from_millis(50), "a test waited for its own slot");
          std::thread::sleep(Duration::from_millis(150));
          live.fetch_sub(1, Ordering::SeqCst);
          slot.index
        })
      })
      .collect();
    let used: Vec<usize> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    std::println!("[device slots] 6 tests on 2 slots: peak {} live, slots used {used:?}", peak.load(Ordering::SeqCst));
    assert!(peak.load(Ordering::SeqCst) <= 2, "more live devices than slots");
    assert!(used.iter().all(|&i| i < 2));
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// Across processes: while this process holds both slots, a child process of this binary waits
  /// for one, and it gets it once a slot is released.
  #[test]
  fn device_slots_cap_processes() {
    let dir = lab_dir("processes");
    let a = acquire_in(&dir, 2, Some("parent-a".into()));
    let b = acquire_in(&dir, 2, Some("parent-b".into()));
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
      .args([
        "gpu_backends::vulkan::test_device_slots::tests::device_slots_child_takes_a_slot",
        "--exact",
        "--ignored",
        "--nocapture",
      ])
      .env("AETHERVK_DEVICE_SLOTS_LAB", &dir)
      .env_remove("NEXTEST")
      .stdout(std::process::Stdio::piped())
      .spawn()
      .unwrap();
    std::thread::sleep(Duration::from_secs(2));
    let early = child.try_wait().unwrap();
    drop(a);
    let released = Instant::now();
    let status = loop {
      if let Some(s) = child.try_wait().unwrap() {
        break s;
      }
      if released.elapsed() > Duration::from_secs(60) {
        let _ = child.kill();
        panic!("the child never got the released slot");
      }
      std::thread::sleep(Duration::from_millis(20));
    };
    let mut out = String::new();
    std::io::Read::read_to_string(child.stdout.as_mut().unwrap(), &mut out).unwrap();
    drop(b);
    std::println!("[device slots] child waited while both slots were held; finished {:?} after a release", released.elapsed());
    assert!(early.is_none(), "the child got a slot while both were held:\n{out}");
    assert!(status.success() && out.contains("1 passed"), "child: {status:?}\n{out}");
    let _ = std::fs::remove_dir_all(&dir);
  }

  /// The child side of `device_slots_cap_processes`.
  #[test]
  #[ignore = "launched as a child process by device_slots_cap_processes"]
  fn device_slots_child_takes_a_slot() {
    let dir = PathBuf::from(std::env::var("AETHERVK_DEVICE_SLOTS_LAB").unwrap());
    let slot = acquire_in(&dir, 2, Some("child".into()));
    assert!(slot.index < 2);
  }

  /// A real device holds its test's slot while it lives, and releases it when it is destroyed.
  #[test]
  fn gpu_device_holds_its_test_slot_until_dropped() {
    use crate::gpu_backends::vulkan::device::test_utils::{setup_assets_dir, setup_render_frontend_for_tests};
    setup_assets_dir();
    crate::gpu_backends::vulkan::test_gpu_watch::name_gpu_test_thread();
    let Some(key) = crate::gpu_backends::vulkan::test_gpu_watch::current_test_key() else {
      std::println!("[device slots] no test thread name on this platform: nothing to check");
      return;
    };
    let (_pool, frontend, _handle, _) = setup_render_frontend_for_tests(false);
    let tmp = std::env::temp_dir();
    let held = SHARED
      .lock()
      .iter()
      .find(|(d, k, _)| *d == tmp && *k == key)
      .and_then(|(_, _, w)| w.upgrade());
    let held = held.expect("the device's test holds no slot");
    // the device's core and this handle
    assert!(Arc::strong_count(&held) >= 2, "the device does not hold its slot");
    let weak = Arc::downgrade(&held);
    drop(held);
    drop(frontend);
    assert!(weak.upgrade().is_none(), "the slot outlived its device");
  }
}
