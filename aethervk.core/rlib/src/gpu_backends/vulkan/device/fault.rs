//! GPU fault diagnostics: what to say when the device is lost.
//!
//! - `VK_EXT_device_fault` (`vkGetDeviceFaultInfoEXT`, loaded by hand: ash 0.38 has the types
//!   but no loader): the driver's description and the faulting addresses, matched against the
//!   buffers registered here ([`register_address_range`]: the dust buffers, the frame staging
//!   arena), so an NVIDIA Xid 13 reads "DustMoments_42 + 0x1230" instead of a bare address.
//! - `VK_NV_device_diagnostic_checkpoints`: [`Device::cmd_checkpoint`] markers (the debug labels
//!   and the dust phases) and, on loss, the last checkpoints each queue reached.
//! - [`Device::report_device_lost`] runs once per device, logs everything and emits
//!   `ExternalState::GpuDeviceLost` (id 12) to the host with the matched address and the last
//!   graphics checkpoint; the device then refuses further submissions ([`Device::is_lost`]).
//! - [`capture_host_diagnostics`] (Linux, once per process; on in debug builds unless
//!   `AETHERVK_GPU_LOST_DUMP=0`, on in release builds with `AETHERVK_GPU_LOST_DUMP=1`): the kernel's `NVRM` lines (the Xid naming this pid), `nvidia-smi
//!   -q`, every thread's backtrace and the engine's timeline state, into
//!   `/tmp/aethervk_gpu_lost_<pid>.txt` and the log. Reached from [`Device::report_device_lost`]
//!   and from every `VK_ERROR_DEVICE_LOST` mapped to [`GpuError::DeviceLost`].
//! - [`EARLY_RECYCLES`]: command buffers recycled before their own queue's timeline reached the
//!   value they signal (`DiscardItem::CommandPool`), i.e. a pool reset while the GPU may still read it.
use super::*;

/// Command buffers recycled while their own timeline was still behind the value they signal
/// (logged per occurrence, summed in the device-loss dump)
pub static EARLY_RECYCLES: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Set by the first [`capture_host_diagnostics`] of the process
static HOST_CAPTURED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Dumps what the host knows about a device loss, once per process (Linux; debug builds unless
/// `AETHERVK_GPU_LOST_DUMP=0`, release builds only with `AETHERVK_GPU_LOST_DUMP=1`): `engine_state` (timeline values, may be empty
/// when the caller has no device), the kernel's `NVRM` lines since the process started (polled
/// for ~2 s until one names this pid, each tagged `[this pid]`/`[other]`), `nvidia-smi -q` on an
/// NVIDIA driver and every thread's gdb backtrace. Everything goes to
/// `/tmp/aethervk_gpu_lost_<pid>.txt`; the Xid lines, a few `nvidia-smi` fields and the
/// backtraces are also logged.
pub fn capture_host_diagnostics(origin: &str, engine_state: &str) {
  #[cfg(target_os = "linux")]
  {
    use aethervk_oshal_rlib::os::debug::{
      append_to_file, capture_all_thread_backtraces_gdb, run_capture,
    };
    use alloc::format;
    let setting = aethervk_oshal_rlib::os::env::var("AETHERVK_GPU_LOST_DUMP");
    let enabled = match setting.as_deref().map(str::trim) {
      Some("0") => false,
      Some("1") => true,
      _ => cfg!(debug_assertions),
    };
    if !enabled {
      return;
    }
    let pid = unsafe { libc::getpid() };
    let out = format!("/tmp/aethervk_gpu_lost_{pid}.txt");
    if HOST_CAPTURED.swap(true, core::sync::atomic::Ordering::AcqRel) {
      // a later report (typically `report_device_lost` after the result mapping) adds its state
      if !engine_state.is_empty() {
        append_to_file(
          &out,
          &format!("== engine state ({origin})\n{engine_state}\n"),
        );
        for l in engine_state.lines() {
          aethervk_oshal_rlib::log!("[GPU lost] {l}");
        }
      }
      return;
    }
    let scratch = format!("/tmp/aethervk_gpu_lost_{pid}.scratch");
    aethervk_oshal_rlib::log!("[GPU lost] host capture ({origin}) → {out}");
    unsafe { libc::unlink(format!("{out}\0").as_ptr().cast()) }; // a stale file of a recycled pid
    let early = EARLY_RECYCLES.load(core::sync::atomic::Ordering::Relaxed);
    append_to_file(
      &out,
      &format!(
        "== device lost: pid {pid}, origin {origin}\n== engine state\n{engine_state}\nearly command-buffer recycles: {early}\n"
      ),
    );
    aethervk_oshal_rlib::log!("[GPU lost] early command-buffer recycles so far: {early}");
    for l in engine_state.lines() {
      aethervk_oshal_rlib::log!("[GPU lost] {l}");
    }

    // the kernel log: /proc/<pid> is created with the process, so its mtime is the start time
    let since = {
      let mut st: libc::stat = unsafe { core::mem::zeroed() };
      let path = format!("/proc/{pid}\0");
      if unsafe { libc::stat(path.as_ptr().cast(), &mut st) } == 0 {
        format!("--since=@{}", st.st_mtime.saturating_sub(1))
      } else {
        alloc::string::String::from("--since=-10min")
      }
    };
    let own = format!("pid={pid},");
    let mut nvrm = alloc::vec::Vec::new();
    for _ in 0..10 {
      let text = run_capture(
        &[
          "journalctl",
          "-k",
          "--no-pager",
          "-o",
          "short-monotonic",
          &since,
        ],
        &scratch,
        false,
        5,
      )
      .unwrap_or_default();
      nvrm = text
        .lines()
        .filter(|l| l.contains("NVRM"))
        .map(alloc::string::String::from)
        .collect();
      if nvrm.iter().any(|l| l.contains(&own)) {
        break;
      }
      unsafe { libc::usleep(200_000) };
    }
    append_to_file(&out, "== kernel NVRM lines since process start\n");
    if nvrm.is_empty() {
      append_to_file(
        &out,
        "(none: no NVRM line, or the kernel log is not readable)\n",
      );
      aethervk_oshal_rlib::log!("[GPU lost] kernel: no NVRM line since process start");
    }
    for l in &nvrm {
      let tag = if l.contains(&own) {
        "[this pid]"
      } else {
        "[other]"
      };
      append_to_file(&out, &format!("{tag} {l}\n"));
      aethervk_oshal_rlib::log!("[GPU lost] kernel {tag} {l}");
    }

    // the GPU's view: PCIe replays, clocks, throttling, temperature, power
    if unsafe { libc::access(c"/proc/driver/nvidia/version".as_ptr(), libc::F_OK) } == 0 {
      append_to_file(&out, "== nvidia-smi -q\n");
      let smi = run_capture(&["nvidia-smi", "-q"], &out, true, 10).unwrap_or_default();
      for l in smi.lines() {
        let t = l.trim_start();
        if [
          "Replays Since Reset",
          "Replay Number Rollovers",
          "Performance State",
          "GPU Current Temp",
          "Power Draw",
        ]
        .iter()
        .any(|k| t.starts_with(k))
        {
          aethervk_oshal_rlib::log!("[GPU lost] nvidia-smi: {t}");
        }
      }
    }

    // what every thread was doing
    append_to_file(&out, "== thread backtraces (gdb)\n");
    if let Some(bt) = capture_all_thread_backtraces_gdb(&out, true) {
      for l in bt.lines() {
        aethervk_oshal_rlib::log!("[GDB] {l}");
      }
    }
    unsafe { libc::unlink(format!("{scratch}\0").as_ptr().cast()) };
    aethervk_oshal_rlib::log!("[GPU lost] host capture complete → {out}");
  }
  #[cfg(not(target_os = "linux"))]
  let _ = (origin, engine_state);
}

/// `(name, device address, size)` of the buffers worth naming in a fault report
static ADDRESS_MAP: spin::Mutex<alloc::vec::Vec<(alloc::string::String, u64, u64)>> =
  spin::Mutex::new(alloc::vec::Vec::new());

/// Registers a buffer's device address range for fault reports (replaces an entry of the same
/// name).
pub fn register_address_range(name: &str, address: u64, size: u64) {
  if address == 0 {
    return;
  }
  let mut m = ADDRESS_MAP.lock();
  m.retain(|(n, _, _)| n != name);
  if m.len() < 4096 {
    m.push((alloc::string::String::from(name), address, size));
  }
}

/// The last ranges forgotten, so a stale pointer into freed memory still gets a name.
static FREED_MAP: spin::Mutex<alloc::vec::Vec<(alloc::string::String, u64, u64)>> =
  spin::Mutex::new(alloc::vec::Vec::new());

/// Forgets a registered range (it stays matchable as "freed").
pub fn unregister_address_range(name: &str) {
  let mut m = ADDRESS_MAP.lock();
  let mut f = FREED_MAP.lock();
  for e in m.iter().filter(|(n, _, _)| n == name) {
    if f.len() >= 1024 {
      f.remove(0);
    }
    f.push(e.clone());
  }
  m.retain(|(n, _, _)| n != name);
}

/// The registered buffer containing `address` (or the nearest one within 64 MB after its end),
/// as "name + offset" / "name end + distance".
pub fn describe_address(address: u64) -> alloc::string::String {
  let m = ADDRESS_MAP.lock();
  for (n, a, s) in m.iter() {
    if address >= *a && address < a + s {
      return alloc::format!("{n} + {:#x} (size {:#x})", address - a, s);
    }
  }
  for (n, a, s) in FREED_MAP.lock().iter().rev() {
    if address >= *a && address < a + s {
      return alloc::format!("FREED {n} + {:#x} (size {:#x})", address - a, s);
    }
  }
  let mut best: Option<(&str, u64)> = None;
  for (n, a, s) in m.iter() {
    if address >= a + s {
      let d = address - (a + s);
      if d < 64 << 20 && best.is_none_or(|b| d < b.1) {
        best = Some((n, d));
      }
    }
  }
  match best {
    Some((n, d)) => alloc::format!("{:#x} past the end of {n}", d),
    None => alloc::string::String::from("no registered buffer"),
  }
}

/// What a device loss looked like (also the host payload, `CGpuDeviceLost`).
#[derive(Debug, Clone, Default)]
pub struct DeviceLossReport {
  pub description: alloc::string::String,
  pub fault_address: u64,
  pub fault_address_type: u32,
  pub matched: alloc::string::String,
  pub last_graphics_checkpoint: alloc::string::String,
  pub last_compute_checkpoint: alloc::string::String,
}

impl Device {
  /// Graphics and compute timelines: next value to submit, the GPU's counter (an error after a
  /// loss is reported as such) and the cached completed value, as text for a fault report.
  fn timeline_state(&self) -> alloc::string::String {
    let counter = |sem: vk::Semaphore| match unsafe {
      self.device.timeline_semaphore.get_semaphore_counter_value(sem)
    } {
      Ok(v) => alloc::format!("{v}"),
      Err(e) => alloc::format!("{e:?}"),
    };
    let graphics = match locks::DebugTrackedRwLock::try_read(&self.res) {
      Some(res) => alloc::format!(
        "graphics: next submit {}, counter {}, cached completed {}",
        res.timeline_manager.get_next_submit_value(),
        counter(res.timeline_manager.semaphore.get()),
        res.timeline_manager.get_cached_value()
      ),
      None => alloc::string::String::from("graphics: resources locked, not read"),
    };
    alloc::format!(
      "{graphics}\ncompute: next submit {}, counter {}",
      self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed),
      counter(self.kernels.timeline)
    )
  }

  /// The device was lost (`VK_ERROR_DEVICE_LOST` seen): every later submission is refused.
  pub fn is_lost(&self) -> bool {
    self.device.lost.load(core::sync::atomic::Ordering::Acquire)
  }

  /// Inserts a diagnostic checkpoint named `name` (a `'static` string, so the marker pointer can
  /// be read back after a loss). No-op without `VK_NV_device_diagnostic_checkpoints`.
  pub fn cmd_checkpoint(&self, cmd: vk::CommandBuffer, name: &'static core::ffi::CStr) {
    if let Some(cp) = self.device.checkpoints.as_ref() {
      unsafe { cp.cmd_set_checkpoint(cmd, name.as_ptr().cast()) };
    }
  }

  /// Marks the device lost and reports the fault once: the driver's fault info (addresses matched
  /// to registered buffers), the last checkpoint of each queue, then `ExternalState::GpuDeviceLost`.
  pub fn report_device_lost(&self, origin: &str) -> Option<DeviceLossReport> {
    if self.device.lost.swap(true, core::sync::atomic::Ordering::AcqRel) {
      return None;
    }
    let mut report = DeviceLossReport::default();
    aethervk_oshal_rlib::log!("[GPU lost] VK_ERROR_DEVICE_LOST at {origin}");
    // VK_EXT_device_fault
    if let Some(get_fault) = self.device.device_fault {
      let mut counts = vk::DeviceFaultCountsEXT::default();
      let r = unsafe {
        get_fault(
          self.device.handle.handle(),
          &mut counts,
          core::ptr::null_mut(),
        )
      };
      if r == vk::Result::SUCCESS || r == vk::Result::INCOMPLETE {
        let na = counts.address_info_count as usize;
        let nv = counts.vendor_info_count as usize;
        let mut addresses = alloc::vec![vk::DeviceFaultAddressInfoEXT::default(); na.max(1)];
        let mut vendors = alloc::vec![vk::DeviceFaultVendorInfoEXT::default(); nv.max(1)];
        let mut binary = alloc::vec![0u8; counts.vendor_binary_size as usize];
        let mut info = vk::DeviceFaultInfoEXT::default();
        info.p_address_infos = addresses.as_mut_ptr();
        info.p_vendor_infos = vendors.as_mut_ptr();
        info.p_vendor_binary_data = if binary.is_empty() {
          core::ptr::null_mut()
        } else {
          binary.as_mut_ptr().cast()
        };
        counts.vendor_binary_size = binary.len() as u64;
        let r = unsafe { get_fault(self.device.handle.handle(), &mut counts, &mut info) };
        let desc = unsafe { core::ffi::CStr::from_ptr(info.description.as_ptr()) }
          .to_string_lossy()
          .into_owned();
        aethervk_oshal_rlib::log!(
          "[GPU lost] fault info ({r:?}): \"{desc}\", {na} address(es), {nv} vendor info(s), {} B vendor data",
          binary.len()
        );
        report.description = desc;
        for (i, a) in addresses.iter().take(na).enumerate() {
          let matched = describe_address(a.reported_address);
          aethervk_oshal_rlib::log!(
            "[GPU lost]   address {i}: type {:?} {:#x} (precision {:#x}) → {matched}",
            a.address_type,
            a.reported_address,
            a.address_precision
          );
          if i == 0 {
            report.fault_address = a.reported_address;
            report.fault_address_type = a.address_type.as_raw() as u32;
            report.matched = matched;
          }
        }
        for v in vendors.iter().take(nv) {
          let d = unsafe { core::ffi::CStr::from_ptr(v.description.as_ptr()) }.to_string_lossy();
          aethervk_oshal_rlib::log!(
            "[GPU lost]   vendor: \"{d}\" code {:#x} data {:#x}",
            v.vendor_fault_code,
            v.vendor_fault_data
          );
        }
      } else {
        aethervk_oshal_rlib::log!("[GPU lost] vkGetDeviceFaultInfoEXT: {r:?}");
      }
    } else {
      aethervk_oshal_rlib::log!("[GPU lost] VK_EXT_device_fault not available: no fault info");
    }
    // VK_NV_device_diagnostic_checkpoints: the last markers each queue reached
    if let Some(cp) = self.device.checkpoints.as_ref() {
      for (role, queue) in [
        ("graphics", self.get_graphics_queue()),
        ("compute", self.get_compute_queue()),
      ] {
        let n = unsafe { cp.get_queue_checkpoint_data_len(queue.handle) };
        let mut data = alloc::vec![vk::CheckpointDataNV::default(); n];
        if n > 0 {
          unsafe { cp.get_queue_checkpoint_data(queue.handle, &mut data) };
        }
        let names: alloc::vec::Vec<alloc::string::String> = data
          .iter()
          .map(|d| {
            let name = if d.p_checkpoint_marker.is_null() {
              alloc::string::String::from("?")
            } else {
              // markers are 'static C strings (cmd_checkpoint)
              unsafe { core::ffi::CStr::from_ptr(d.p_checkpoint_marker.cast()) }
                .to_string_lossy()
                .into_owned()
            };
            alloc::format!("{name} @ {:?}", d.stage)
          })
          .collect();
        aethervk_oshal_rlib::log!("[GPU lost] {role} queue checkpoints: {}", names.join(" | "));
        let last = names.last().cloned().unwrap_or_default();
        if role == "graphics" {
          report.last_graphics_checkpoint = last;
        } else {
          report.last_compute_checkpoint = last;
        }
      }
    } else {
      aethervk_oshal_rlib::log!("[GPU lost] VK_NV_device_diagnostic_checkpoints not available");
    }
    // the host side: kernel log, nvidia-smi, backtraces, timeline state
    capture_host_diagnostics(origin, &self.timeline_state());
    // the host
    let mut payload = crate::simulation_api::external_state::CGpuDeviceLost {
      reason: 1,
      has_fault_info: (self.device.device_fault.is_some() && !report.description.is_empty()) as u32,
      fault_address: report.fault_address,
      matched: [0; 64],
      last_checkpoint: [0; 64],
    };
    let copy = |dst: &mut [u8; 64], s: &str| {
      let b = s.as_bytes();
      let n = b.len().min(63);
      dst[..n].copy_from_slice(&b[..n]);
    };
    copy(&mut payload.matched, &report.matched);
    copy(
      &mut payload.last_checkpoint,
      &report.last_graphics_checkpoint,
    );
    crate::simulation_api::emit_external_state_change(
      &crate::simulation_api::external_state::ExternalState::GpuDeviceLost(payload),
    );
    Some(report)
  }
}

#[cfg(all(test, target_os = "linux", debug_assertions))]
mod tests {
  /// The host capture writes every section, runs once, and a later report only appends its state.
  /// No device is needed: this is the path a `VK_ERROR_DEVICE_LOST` result takes.
  #[test]
  fn host_capture_writes_every_section_once() {
    let pid = unsafe { libc::getpid() };
    let path = alloc::format!("/tmp/aethervk_gpu_lost_{pid}.txt");
    super::capture_host_diagnostics("unit test", "");
    let first = std::fs::read_to_string(&path).expect("dump file");
    assert!(first.contains("== device lost: pid"), "{first}");
    assert!(
      first.contains("== kernel NVRM lines since process start"),
      "{first}"
    );
    assert!(first.contains("== thread backtraces (gdb)"), "{first}");
    if std::path::Path::new("/proc/driver/nvidia/version").exists() {
      assert!(
        first.contains("== nvidia-smi -q") && first.contains("Driver Version"),
        "{first}"
      );
    }
    if std::path::Path::new("/usr/bin/gdb").exists() {
      // gdb attached to this process (PR_SET_PTRACER under Yama ptrace_scope 1)
      assert!(
        first.contains("host_capture_writes_every_section_once"),
        "{first}"
      );
    }

    super::capture_host_diagnostics("second report", "graphics: test state");
    let second = std::fs::read_to_string(&path).unwrap();
    assert!(second.starts_with(&first));
    assert_eq!(
      second[first.len()..].trim_end(),
      "== engine state (second report)\ngraphics: test state"
    );
    let _ = std::fs::remove_file(&path);
  }
}
