//! Windows crash and device-loss capture, the counterpart of the Linux path in `fault.rs`.
//!
//! Linux gets gdb thread backtraces, `nvidia-smi` and the kernel log. Windows gets a minidump of
//! the process (every thread's stack, the loaded modules, the faulting exception record) and the
//! same external reports: `nvidia-smi -q` and the display driver's TDR events (System log, provider
//! `Display`, event 4101) through `wevtutil`. Everything is written next to the .NET log sink, in
//! `%LOCALAPPDATA%\Aether-Vk\logs`.
//!
//! Only the `windows` crate is used (no `std::process`), so the capture is available in every
//! build flavour, including the cdylib that the .NET host loads.
use alloc::{format, string::String, vec::Vec};
use core::sync::atomic::{AtomicBool, Ordering};
use windows::{
  Win32::{
    Foundation::{CloseHandle, HANDLE, HANDLE_FLAG_INHERIT, HANDLE_FLAGS, SetHandleInformation},
    Security::SECURITY_ATTRIBUTES,
    Storage::FileSystem::{
      CREATE_ALWAYS, CreateFileW, FILE_APPEND_DATA, FILE_ATTRIBUTE_NORMAL, FILE_GENERIC_WRITE,
      FILE_SHARE_READ, OPEN_ALWAYS, ReadFile, WriteFile,
    },
    System::{
      Diagnostics::Debug::{
        EXCEPTION_POINTERS, MINIDUMP_EXCEPTION_INFORMATION, MiniDumpWithThreadInfo,
        MiniDumpWriteDump, SetUnhandledExceptionFilter,
      },
      Pipes::CreatePipe,
      Threading::{
        CREATE_NO_WINDOW, CreateProcessW, GetCurrentProcess, GetCurrentProcessId, GetCurrentThreadId,
        PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOW, WaitForSingleObject,
      },
    },
  },
  core::{BOOL, PCWSTR},
};

/// This process's id, as the file names of the dumps and reports carry it.
pub fn process_id() -> u32 {
  unsafe { GetCurrentProcessId() }
}

/// The per-user log folder shared with the .NET side (`LogFileSink.DefaultDirectory`).
/// `None` when `LOCALAPPDATA` is unset.
pub fn log_dir() -> Option<String> {
  let base = crate::os::env::var("LOCALAPPDATA")?;
  Some(format!("{base}\\Aether-Vk\\logs"))
}

fn wide(text: &str) -> Vec<u16> {
  text.encode_utf16().chain(core::iter::once(0)).collect()
}

/// Writes a minidump of this process to `path`, replacing any file there. With `exception`, the
/// faulting thread's context is included. Returns whether the dump was written.
pub fn write_minidump(path: &str, exception: Option<*const EXCEPTION_POINTERS>) -> bool {
  let path = wide(path);
  unsafe {
    let file = match CreateFileW(
      PCWSTR(path.as_ptr()),
      FILE_GENERIC_WRITE.0,
      FILE_SHARE_READ,
      None,
      CREATE_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      None,
    ) {
      Ok(file) => file,
      Err(_) => return false,
    };

    let info = exception.map(|pointers| MINIDUMP_EXCEPTION_INFORMATION {
      ThreadId: GetCurrentThreadId(),
      ExceptionPointers: pointers as *mut EXCEPTION_POINTERS,
      ClientPointers: BOOL(0),
    });
    let info_ptr = info.as_ref().map(|info| info as *const MINIDUMP_EXCEPTION_INFORMATION);

    let written = MiniDumpWriteDump(
      GetCurrentProcess(),
      GetCurrentProcessId(),
      file,
      MiniDumpWithThreadInfo,
      info_ptr,
      None,
      None,
    )
    .is_ok();
    let _ = CloseHandle(file);
    written
  }
}

/// Runs `command_line` (a full command line, as a shell would take it) with its output captured,
/// and returns what it wrote to stdout and stderr. Empty when the program cannot start.
pub fn run_capture(command_line: &str) -> String {
  let mut command = wide(command_line);
  unsafe {
    let security = SECURITY_ATTRIBUTES {
      nLength: core::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
      lpSecurityDescriptor: core::ptr::null_mut(),
      bInheritHandle: BOOL(1),
    };

    let mut read = HANDLE::default();
    let mut write = HANDLE::default();
    if CreatePipe(&mut read, &mut write, Some(&security), 0).is_err() {
      return String::new();
    }
    // The child gets the write end only; our read end must not leak into it
    let _ = SetHandleInformation(read, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));

    let startup = STARTUPINFOW {
      cb: core::mem::size_of::<STARTUPINFOW>() as u32,
      dwFlags: STARTF_USESTDHANDLES,
      hStdOutput: write,
      hStdError: write,
      ..Default::default()
    };
    let mut process = PROCESS_INFORMATION::default();
    let started = CreateProcessW(
      PCWSTR::null(),
      Some(windows::core::PWSTR(command.as_mut_ptr())),
      None,
      None,
      true,
      CREATE_NO_WINDOW,
      None,
      PCWSTR::null(),
      &startup,
      &mut process,
    );
    // Closing our copy of the write end makes ReadFile report the end once the child exits
    let _ = CloseHandle(write);
    if started.is_err() {
      let _ = CloseHandle(read);
      return String::new();
    }

    let mut output = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
      let mut got = 0u32;
      if ReadFile(read, Some(&mut chunk), Some(&mut got), None).is_err() || got == 0 {
        break;
      }
      output.extend_from_slice(&chunk[..got as usize]);
    }

    let _ = WaitForSingleObject(process.hProcess, 10_000);
    let _ = CloseHandle(process.hProcess);
    let _ = CloseHandle(process.hThread);
    let _ = CloseHandle(read);
    String::from_utf8_lossy(&output).into_owned()
  }
}

/// Device-loss report for the Windows host: the minidump next to `out_path`, then the driver's
/// view (`nvidia-smi -q`, when the program is on PATH) and the TDR events, appended to `out_path`.
/// Returns the lines worth echoing to the log.
pub fn capture_gpu_lost(out_path: &str, dump_path: &str) -> Vec<String> {
  let mut echo = Vec::new();
  let dumped = write_minidump(dump_path, None);
  echo.push(format!(
    "[GPU lost] minidump {}: {dump_path}",
    if dumped { "written" } else { "FAILED" }
  ));

  let smi = run_capture("nvidia-smi -q");
  let tdr = run_capture(
    "wevtutil qe System /c:20 /rd:true /f:text /q:\"*[System[Provider[@Name='Display'] and (EventID=4101)]]\"",
  );

  let mut report = String::new();
  report.push_str("== nvidia-smi -q\n");
  report.push_str(if smi.is_empty() { "(not available)\n" } else { &smi });
  report.push_str("\n== display driver TDR events (System log, Display 4101)\n");
  report.push_str(if tdr.is_empty() { "(none readable)\n" } else { &tdr });
  append_text(out_path, &report);

  for l in smi.lines() {
    let t = l.trim_start();
    if [
      "Replays Since Reset",
      "Performance State",
      "GPU Current Temp",
      "Power Draw",
    ]
    .iter()
    .any(|k| t.starts_with(k))
    {
      echo.push(format!("[GPU lost] nvidia-smi: {t}"));
    }
  }
  if tdr.contains("4101") {
    echo.push("[GPU lost] display driver reported a TDR (event 4101) in the System log".into());
  }
  echo
}

/// Path of cdb.exe, as `scripts\windows\AvkDebug.psm1` (`Get-AvkCdb`) looks for it: `AVK_CDB`, the
/// copy `Setup-DebugTools.ps1` makes in `%LOCALAPPDATA%\dbg\amd64`, then the Windows Kits debuggers.
/// `None` when none exists.
fn find_cdb() -> Option<String> {
  use windows::Win32::Storage::FileSystem::{GetFileAttributesW, INVALID_FILE_ATTRIBUTES};
  let mut candidates = Vec::new();
  candidates.extend(crate::os::env::var("AVK_CDB"));
  candidates.extend(crate::os::env::var("LOCALAPPDATA").map(|d| format!("{d}\\dbg\\amd64\\cdb.exe")));
  for var in ["ProgramFiles(x86)", "ProgramFiles"] {
    candidates.extend(crate::os::env::var(var).map(|d| format!("{d}\\Windows Kits\\10\\Debuggers\\x64\\cdb.exe")));
  }
  candidates
    .into_iter()
    .find(|c| unsafe { GetFileAttributesW(PCWSTR(wide(c).as_ptr())) } != INVALID_FILE_ATTRIBUTES)
}

/// Render-stall report (the debug builds' stall watcher, the counterpart of the Linux gdb dump):
/// a minidump of this process to `aethervk-stall-<pid>.dmp` in [`log_dir`], then every thread's
/// stack read back out of that dump by cdb (`~*kn`, when cdb is found) into
/// `aethervk-stall-<pid>.txt`. The stacks come from the dump, out of process: no thread of ours is
/// suspended while a lock it holds might be needed. Returns the lines worth echoing to the log
/// (the paths, then cdb's stacks).
pub fn capture_stall() -> Vec<String> {
  let mut echo = Vec::new();
  let Some(dir) = log_dir() else {
    echo.push("[StallWatcher] LOCALAPPDATA unset: no dump written".into());
    return echo;
  };
  let pid = process_id();
  let dump_path = format!("{dir}\\aethervk-stall-{pid}.dmp");
  let dumped = write_minidump(&dump_path, None);
  echo.push(format!(
    "[StallWatcher] minidump {}: {dump_path}",
    if dumped { "written" } else { "FAILED" }
  ));
  if !dumped {
    return echo;
  }
  let Some(cdb) = find_cdb() else {
    echo.push(
      "[StallWatcher] cdb.exe not found (scripts\\windows\\Setup-DebugTools.ps1, or AVK_CDB): open the dump with windbgx -z".into(),
    );
    return echo;
  };
  // our PDBs are found through the absolute paths in the images; `_NT_SYMBOL_PATH`, when set,
  // adds the system ones
  let stacks = run_capture(&format!("\"{cdb}\" -z \"{dump_path}\" -c \"~*kn 64; q\""));
  let stacks_path = format!("{dir}\\aethervk-stall-{pid}.txt");
  let _ = crate::os::fs::remove_file(&stacks_path);
  append_text(&stacks_path, &stacks);
  echo.push(format!("[StallWatcher] thread stacks (cdb ~*kn): {stacks_path}"));
  // the stacks only: cdb's banner and symbol chatter end where it echoes the command, the
  // unload chatter starts at `quit:`
  let body = stacks.split_once("~*kn 64; q'").map_or(stacks.as_str(), |(_, b)| b);
  let body = body.split_once("quit:").map_or(body, |(b, _)| b);
  echo.extend(body.lines().filter(|l| !l.trim().is_empty()).map(|l| format!("[cdb] {l}")));
  echo
}

static INSTALLED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn unhandled_exception(pointers: *const EXCEPTION_POINTERS) -> i32 {
  if let Some(dir) = log_dir() {
    let pid = unsafe { GetCurrentProcessId() };
    let _ = write_minidump(&format!("{dir}\\aethervk-crash-{pid}.dmp"), Some(pointers));
  }
  // EXCEPTION_CONTINUE_SEARCH: the OS still reports and ends the process
  0
}

/// Installs the unhandled-exception filter that writes a minidump before the process dies. Safe
/// to call more than once; the first call wins.
pub fn install_crash_dump_handler() {
  if INSTALLED.swap(true, Ordering::AcqRel) {
    return;
  }
  unsafe {
    SetUnhandledExceptionFilter(Some(unhandled_exception));
  }
}

/// Minidump for a Rust panic, written before the process is terminated.
pub fn write_panic_dump() {
  if let Some(dir) = log_dir() {
    let pid = unsafe { GetCurrentProcessId() };
    let _ = write_minidump(&format!("{dir}\\aethervk-panic-{pid}.dmp"), None);
  }
}

/// Appends `text` to the file at `path`, creating it if needed. The handle has append access, so
/// every write lands at the end whatever the file position is.
pub fn append_text(path: &str, text: &str) {
  let path = wide(path);
  unsafe {
    let file = match CreateFileW(
      PCWSTR(path.as_ptr()),
      FILE_APPEND_DATA.0,
      FILE_SHARE_READ,
      None,
      OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      None,
    ) {
      Ok(file) => file,
      Err(_) => return,
    };
    let mut written = 0u32;
    let _ = WriteFile(file, Some(text.as_bytes()), Some(&mut written), None);
    let _ = CloseHandle(file);
  }
}

#[cfg(test)]
mod tests {
  /// The stall capture end to end: the minidump is written and cdb reads the other threads' stacks
  /// out of it with our symbols (the harness's main thread waits in `test::test_main`). The calling
  /// thread's own stack ends in `NtGetContextThread` (the dump is taken mid-call), which is why the
  /// watcher thread is the one that calls it. Needs cdb (`scripts\windows\Setup-DebugTools.ps1`) and writes to
  /// `%LOCALAPPDATA%\Aether-Vk\logs`, so it is opt-in: `cargo test -p aethervk-oshal-rlib
  /// capture_stall -- --ignored`.
  #[test]
  #[ignore = "needs cdb.exe; writes a minidump to the Aether-Vk log folder"]
  fn capture_stall_writes_the_dump_and_the_stacks() {
    std::fs::create_dir_all(super::log_dir().expect("LOCALAPPDATA")).unwrap();
    let lines = super::capture_stall();
    for l in &lines {
      std::println!("{l}");
    }
    assert!(lines.iter().any(|l| l.contains("minidump written")), "no dump: {lines:?}");
    assert!(
      lines.iter().any(|l| l.starts_with("[cdb]") && l.contains("!test::test_main")),
      "cdb did not symbolise the main thread's stack"
    );
  }
}
