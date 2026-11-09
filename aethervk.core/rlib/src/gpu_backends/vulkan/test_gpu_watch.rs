//! Tests only: names GPU test threads, and freezes the process at the first device loss.
//!
//! **Thread names.** The NVIDIA kernel driver names a faulting process by the OS name of its
//! channel's thread (`NVRM: Xid …, pid=…, name=…`). That name is 15 bytes on Linux, so libtest's
//! thread names (the full test path) all read `gpu_backends::v`.
//! [`name_gpu_test_thread`] gives the thread a short name built from the test instead:
//! - up to 10 bytes of the test function's name, `~`, and 4 hex digits of a hash of the full path;
//! - the pair is logged and appended to `<temp>/aethervk_gpu_test_threads_<pid>.txt`, so a
//!   kernel line maps back to the test.
//!
//! The hooks run before a GPU test spawns its device or runtime threads, which inherit the name:
//! - `init_device`;
//! - `setup_render_frontend_for_tests`;
//! - the simulation runtime's thread pool.
//!
//! **Freeze.** With `AETHERVK_GPU_LOST_FREEZE=1`, the first `VK_ERROR_DEVICE_LOST` result
//! ([`on_device_lost`], called first thing in `capture_host_diagnostics`) stops the whole process
//! with SIGSTOP, before any teardown:
//! - the faulted channel's rings and command buffers stay mapped;
//! - a watcher (`nvpb.py watch`) sees the stopped process, snapshots it and resumes it;
//! - a marker file `<temp>/aethervk_gpu_lost_<pid>.freeze` names the thread and the test.
//! (Unix only; Windows logs the names and carries on.)

use core::sync::atomic::{AtomicBool, Ordering};
use std::io::Write;
use std::string::String;

/// The OS thread name for a test path (≤ 15 bytes, the Linux limit).
pub fn short_name(full: &str) -> String {
  let leaf = full.rsplit("::").next().unwrap_or(full);
  let leaf = leaf.strip_prefix("gpu_").unwrap_or(leaf);
  let leaf = leaf.strip_prefix("test_").unwrap_or(leaf);
  let mut h: u32 = 0x811c_9dc5; // FNV-1a over the full path: different tests, different suffixes
  for b in full.bytes() {
    h = (h ^ b as u32).wrapping_mul(0x0100_0193);
  }
  let head: String = leaf.chars().filter(|c| c.is_ascii_graphic()).take(10).collect();
  alloc::format!("{head}~{:04x}", (h ^ (h >> 16)) & 0xffff)
}

/// The full test path of this thread:
/// - libtest names each test's thread after the test;
/// - single-threaded runs (nextest: one test per process) run on `main`, so the test then comes
///   from the `--exact` argument when exactly one test is named.
pub fn current_test_path() -> Option<String> {
  if let Some(n) = std::thread::current().name() {
    if n.contains("::") {
      return Some(n.into());
    }
  }
  let args: std::vec::Vec<String> = std::env::args().collect();
  // exactly one test named with `--exact`; a run listing several names says nothing per thread
  let names: std::vec::Vec<&String> = args.iter().skip(1).filter(|a| a.contains("::") && !a.starts_with('-')).collect();
  if args.iter().any(|a| a == "--exact") && names.len() == 1 {
    return Some(names[0].clone());
  }
  None
}

std::thread_local! {
  static NAMED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// The short names given to test threads in this process (`test_device_slots` keys a test's
/// devices by them).
static TEST_NAMES: parking_lot::Mutex<std::vec::Vec<String>> = parking_lot::Mutex::new(std::vec::Vec::new());

/// The calling thread's test, as its short OS thread name, when that name was given by
/// [`name_gpu_test_thread`] (here or in the thread that spawned this one: the name is inherited on
/// Linux and macOS). `None` for any other thread.
pub fn current_test_key() -> Option<String> {
  let name = os_thread_name();
  TEST_NAMES.lock().iter().any(|n| *n == name).then_some(name)
}

/// Names this thread after its test, once per thread. Threads it spawns afterwards inherit the
/// name (Linux, macOS). A thread that is not a test's own (a spawned device thread) keeps the name
/// it inherited.
pub fn name_gpu_test_thread() {
  if NAMED.with(|n| n.replace(true)) {
    return;
  }
  let Some(full) = current_test_path() else {
    return;
  };
  let short = short_name(&full);
  set_os_thread_name(&short);
  {
    let mut names = TEST_NAMES.lock();
    if !names.contains(&short) {
      names.push(short.clone());
    }
  }
  aethervk_oshal_rlib::log!("[gpu test] thread '{short}' = {full}");
  let path = std::env::temp_dir().join(alloc::format!("aethervk_gpu_test_threads_{}.txt", std::process::id()));
  if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
    let _ = writeln!(f, "{short}\t{full}");
  }
}

fn set_os_thread_name(name: &str) {
  let Ok(c) = std::ffi::CString::new(name) else {
    return;
  };
  #[cfg(any(target_os = "linux", target_os = "android"))]
  unsafe {
    libc::prctl(libc::PR_SET_NAME, c.as_ptr() as libc::c_ulong, 0, 0, 0);
  }
  #[cfg(target_vendor = "apple")]
  unsafe {
    libc::pthread_setname_np(c.as_ptr());
  }
  #[cfg(windows)]
  unsafe {
    let wide: std::vec::Vec<u16> = name.encode_utf16().chain(core::iter::once(0)).collect();
    let _ = windows::Win32::System::Threading::SetThreadDescription(
      windows::Win32::System::Threading::GetCurrentThread(),
      windows::core::PCWSTR(wide.as_ptr()),
    );
  }
  let _ = c;
}

/// This thread's OS name (what the kernel logs).
fn os_thread_name() -> String {
  #[cfg(any(target_os = "linux", target_os = "android"))]
  {
    let mut buf = [0u8; 16];
    unsafe { libc::prctl(libc::PR_GET_NAME, buf.as_mut_ptr() as libc::c_ulong, 0, 0, 0) };
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    return String::from_utf8_lossy(&buf[..n]).into_owned();
  }
  #[allow(unreachable_code)]
  std::thread::current().name().unwrap_or("?").into()
}

static FROZEN: AtomicBool = AtomicBool::new(false);

/// The first device loss of the process: logs the thread and its test and, under
/// `AETHERVK_GPU_LOST_FREEZE=1`, stops the process until a watcher resumes it.
pub fn on_device_lost(origin: &str) {
  if FROZEN.swap(true, Ordering::AcqRel) {
    return;
  }
  let os_name = os_thread_name();
  let test = current_test_path().unwrap_or_else(|| "(not a test thread; see the thread-name file)".into());
  aethervk_oshal_rlib::log!("[GPU lost] {origin} on thread '{os_name}' ({test})");
  let freeze = std::env::var("AETHERVK_GPU_LOST_FREEZE").is_ok_and(|v| v.trim() == "1");
  if !freeze {
    return;
  }
  let pid = std::process::id();
  let marker = std::env::temp_dir().join(alloc::format!("aethervk_gpu_lost_{pid}.freeze"));
  let _ = std::fs::write(&marker, alloc::format!("pid {pid}\nthread {os_name}\ntest {test}\norigin {origin}\n"));
  aethervk_oshal_rlib::log!("[GPU lost] freezing pid {pid} for the watcher ({})", marker.display());
  #[cfg(unix)]
  unsafe {
    libc::raise(libc::SIGSTOP);
  }
  aethervk_oshal_rlib::log!("[GPU lost] pid {pid} resumed");
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The names fit the kernel's 15 bytes, keep the test's words, differ between tests that share
  /// their first ten characters, and are stable.
  #[test]
  fn gpu_thread_names_fit_and_differ() {
    let a = "gpu_backends::vulkan::device::test_dust::gpu_dust_splat_matches_reference_haze";
    let b = "gpu_backends::vulkan::device::test_dust::gpu_dust_splat_matches_reference_volume_texel_integral";
    for n in [short_name(a), short_name(b)] {
      assert!(n.len() <= 15, "{n} is {} bytes", n.len());
      assert!(n.starts_with("dust_splat"), "{n}");
    }
    assert_ne!(short_name(a), short_name(b));
    assert_eq!(short_name(a), short_name(a));
    // this test's own thread gets its name, and the OS reports it
    name_gpu_test_thread();
    #[cfg(target_os = "linux")]
    if let Some(full) = current_test_path() {
      assert_eq!(os_thread_name(), short_name(&full));
    }
  }

  /// The freeze hook, without any GPU fault: a child process of this binary calls
  /// [`on_device_lost`] with `AETHERVK_GPU_LOST_FREEZE=1`. The child must stop itself, its marker
  /// must name its thread and test, and after SIGCONT it must finish its test and pass.
  #[cfg(target_os = "linux")]
  #[test]
  fn gpu_lost_freeze_stops_the_process_until_resumed() {
    use std::time::{Duration, Instant};
    const CHILD: &str = "gpu_backends::vulkan::test_gpu_watch::tests::gpu_lost_freeze_child";
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
      .args([CHILD, "--exact", "--ignored", "--nocapture", "--test-threads=1"])
      .env("AETHERVK_GPU_LOST_FREEZE", "1")
      // a watcher (nvpb.py watch) leaves this deliberate freeze to the test
      .env("AETHERVK_WATCH_IGNORE", "1")
      .env_remove("NEXTEST")
      .stdout(std::process::Stdio::piped())
      .spawn()
      .unwrap();
    let pid = child.id();
    let state = || {
      std::fs::read_to_string(alloc::format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| s.rsplit_once(')').and_then(|(_, r)| r.split_whitespace().next().map(String::from)))
    };
    let start = Instant::now();
    while state().as_deref() != Some("T") {
      assert!(child.try_wait().unwrap().is_none(), "the child exited without stopping");
      assert!(start.elapsed() < Duration::from_secs(60), "the child never stopped");
      std::thread::sleep(Duration::from_millis(10));
    }
    let marker = std::env::temp_dir().join(alloc::format!("aethervk_gpu_lost_{pid}.freeze"));
    let text = std::fs::read_to_string(&marker).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(state().as_deref(), Some("T"), "the child resumed on its own");
    unsafe { libc::kill(pid as i32, libc::SIGCONT) };
    let status = child.wait().unwrap();
    let mut out = String::new();
    std::io::Read::read_to_string(child.stdout.as_mut().unwrap(), &mut out).unwrap();
    let _ = std::fs::remove_file(&marker);
    std::println!("[gpu lost freeze] marker:\n{text}");
    assert!(text.contains(&alloc::format!("test {CHILD}")), "{text}");
    assert!(text.contains(&alloc::format!("thread {}", short_name(CHILD))), "{text}");
    assert!(status.success(), "{status:?}\n{out}");
    assert!(out.contains("resumed") && out.contains("1 passed"), "{out}");
  }

  /// The child side of `gpu_lost_freeze_stops_the_process_until_resumed`.
  #[test]
  #[ignore = "launched as a child process by gpu_lost_freeze_stops_the_process_until_resumed"]
  fn gpu_lost_freeze_child() {
    name_gpu_test_thread();
    on_device_lost("unit test");
  }
}
