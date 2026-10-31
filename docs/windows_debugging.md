# Windows debugging and diagnostics

The scripts are in `scripts/windows/` and use `cdb.exe` from WinDbg. They need PowerShell 7 (`pwsh`). Reports and dumps go to `target\diagnostics\` (gitignored).

## Linux tool → Windows equivalent

| Linux | Windows |
|---|---|
| `gdb --pid <pid> --batch -ex "thread apply all bt"` | `Get-HungStacks.ps1 -Id <pid>` (non-invasive cdb attach) |
| `gcore` / core dump of a live process | `Get-HungStacks.ps1 -Id <pid> -Dump Full` |
| gdb automated tracing: run, wait for the hang, SIGINT, `thread apply all bt` | `Watch-TestHang.ps1 -- cargo nextest run ...` |
| `docker-gdb-run.sh` (`gdb -batch`, `bt full` on SIGSEGV) | `Invoke-UnderDebugger.ps1 <exe> [args]` |
| `cargo nextest run --debugger 'rust-gdb --args'` | `cargo nextest run --debugger 'pwsh -NoProfile -File scripts/windows/Invoke-UnderDebugger.ps1'` |
| core dumps (`ulimit -c`) | WER LocalDumps (`Setup-DebugTools.ps1 -EnableCrashDumps`, elevated) |
| `perf top -p <pid>` | `Get-StackSamples.ps1 -Id <pid>` (quick look), or ETW: `wpr` + WPA (full profile) |
| `ls /proc/<pid>/task` + `top -H` (one thread at 100 %?) | `Get-StackSamples.ps1`: the CPU-per-thread table |
| In-process gdb dumps (render-stall watcher, `AETHERVK_GPU_LOST_DUMP`) | Already on Windows: minidumps on an unhandled exception, on a panic and on device loss, in `%LOCALAPPDATA%\Aether-Vk\logs` (`crash_dump.rs`). The render-stall watcher is still Linux-only. |

## One-time setup

```powershell
winget install Microsoft.WinDbg              # if WinDbg is not installed yet
scripts\windows\Setup-DebugTools.ps1         # copies cdb out of the Store package, writes symsrv.ini, prints the status
scripts\windows\Setup-DebugTools.ps1 -Status # status only
```

- **cdb location.** The Store package's `cdb.exe` can't be run from WindowsApps. The script copies the `amd64` folder to `%LOCALAPPDATA%\dbg\amd64`. Set `AVK_CDB` to use a different cdb.
- **Symbols.** The scripts build the symbol path in this order:
  1. local folders: `target\...`, the exe's own folder, and `%VULKAN_SDK%\Bin` (that's where the validation-layer PDBs are);
  2. `srv*C:\symbols*https://msdl.microsoft.com/download/symbols`. Set `AVK_SYMBOL_CACHE` to change the cache folder.

  Symbol loading stays deferred: there is no `.reload /f`, so only modules that show up on a stack get looked up. A warm attach takes about 0.5 s for a small exe and about 3 s for a 200 MB test PDB.
- **Why `symsrv.ini` matters.** It goes next to cdb and holds the symbol-server exclusion list: the NVIDIA, AMD and Intel ICDs, `VkLayer_*`, our own PDBs, and Skia. The Microsoft server has no PDB for any of these, and asking for one (`nvoglv64.pdb` especially) used to cost a timeout. Local folders are still searched.
- **Where the test exes live.** The pinned nightly cargo uses the new build-dir layout: test exes are at `target\debug\build\<pkg>\<hash>\out\<name>-<hash>.exe`. `target\debug\deps` holds only stale binaries. cdb finds the PDB through the path embedded in the exe, so this doesn't matter for symbols, but look there when you need an exe path.

## A test or the app hangs

**Stacks of a live process.** `-Dump Mini|Full` also writes a dump:

```powershell
scripts\windows\Get-HungStacks.ps1 -Name aethervk_core_rlib*   # or -Id <pid>
scripts\windows\Get-HungStacks.ps1 -Id <pid> -Dump Full
scripts\windows\Get-HungStacks.ps1 -Name AetherVk -Managed     # the .NET app: adds SOS !clrstack -all
```

How the attach works and what the report holds:
- The attach is non-invasive (`-pv`). Threads are suspended while cdb reads them and resumed when it detaches, so the process carries on afterwards.
- The report starts with a one-line-per-thread summary:
  - the leaf frame;
  - the first application frame;
  - tags: `loader` (`vulkan_1`), `layer` (`VkLayer_*`), `driver` (`nvoglv64` …), `waiting`.
- Then it has the full `~*kn`, `!runaway` (CPU per thread) and `!locks`.
- Rust names each test thread after its test, so the summary also tells you *which test* is hung.

**Watchdog for a whole run.** It catches a test process that stalls or runs too long, before nextest kills it:

```powershell
scripts\windows\Watch-TestHang.ps1 -- cargo nextest run -p aethervk-core-rlib
scripts\windows\Watch-TestHang.ps1 -StallSec 20 -Dump Full -- cargo test -p aethervk-core-rlib --lib -- --test-threads=1
scripts\windows\Watch-TestHang.ps1          # no command: watch every test exe on the machine (second terminal)
```

Triggers:
- **stall:** less than `-MinCpuSec` (0.25 s) of CPU over `-StallSec` (30 s);
- **age:** alive for `-MaxAgeSec` (270 s). This catches spinning hangs, and fires before the 5 × 60 s `terminate-after` in `.config/nextest.toml`.

Each process is reported once per trigger, and the summary is echoed into the run's output. The script exits with the command's exit code. nextest has no hook that runs before a timeout kill, so the watchdog watches from outside. That way it works the same for plain `cargo test`, where one process runs the whole suite (the loader-stall case in `docs/vulkan_test_loader_stall.md`).

## A test or the app crashes

**Run under the debugger.** On a second-chance exception the script prints the stack of the faulting thread, then every thread's stack, and writes a dump. It catches:
- access violation;
- `__fastfail` / Rust `abort` (c0000409);
- stack overflow;
- `ud2`;
- divide by zero;
- uncaught C++ / CLR exceptions;
- `int 3`;
- any other unhandled exception.

The exit code is the exception code, so nextest still reports ABORT. Options come from environment variables (`AVK_CRASH_DIR`, `AVK_CRASH_DUMP=Full|Mini|None`), so every argument goes to the program untouched.

```powershell
# one test (nextest's --debugger runs exactly one)
cargo nextest run -p aethervk-core-rlib -E 'test(=some::test)' --debugger 'pwsh -NoProfile -File scripts/windows/Invoke-UnderDebugger.ps1'
# every test of a run (nextest or cargo test) under the debugger, as the target runner (~0.6 s per test process)
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER = 'pwsh -NoProfile -File Y:/Aether-Vk/scripts/windows/Invoke-UnderDebugger.ps1'
# an exe directly
scripts\windows\Invoke-UnderDebugger.ps1 target\debug\build\aethervk-core-rlib\<hash>\out\aethervk_core_rlib-<hash>.exe --exact some::test --nocapture
```

A `--list` call (nextest's discovery step) runs the exe directly, without cdb, so the script works as a runner.

**Crash dumps without a debugger.** In an elevated shell, run `scripts\windows\Setup-DebugTools.ps1 -EnableCrashDumps` (add `-DumpType Mini` for small dumps). Windows Error Reporting then writes `%LOCALAPPDATA%\CrashDumps\<exe>.<pid>.dmp` for any process that crashes, and `-DisableCrashDumps` turns it off again. The key is machine-wide, because the test exe names change with every build. This hasn't been verified for processes started by nextest: if nextest sets `SEM_NOGPFAULTERRORBOX`, WER skips them. Use `Invoke-UnderDebugger.ps1` when you have to be certain.

**The app's own dumps.** `crash_dump.rs` writes these to `%LOCALAPPDATA%\Aether-Vk\logs`:
- `aethervk-crash-<pid>.dmp` from the unhandled-exception filter;
- `aethervk-panic-<pid>.dmp` from a panic in the cdylib;
- `aethervk-gpu-lost-<pid>.dmp` and `gpu_lost_<pid>.txt` on device loss.

`StartupWatchdog.cs` adds a dump when startup goes silent.

## Opening a dump

```powershell
$sym = "srv*C:\symbols*https://msdl.microsoft.com/download/symbols"
& "$env:LOCALAPPDATA\dbg\amd64\cdb.exe" -z <file.dmp> -y $sym -lines -c "!analyze -v; .ecxr; kn; ~*kn 60; q"
windbgx -z <file.dmp>       # GUI
```

## CPU profiling

**Quick look, no setup:**

```powershell
scripts\windows\Get-StackSamples.ps1 -Name dust_observer -Samples 30
```

The report has CPU per thread over the window (from the OS thread times), the hottest leaf functions (self), the functions most often on the stack (inclusive), and the hottest leaf per thread. Threads sitting in a kernel wait are left out unless you pass `-IncludeWaiting`. Each sample is a short attach, so expect a sample every 0.5–3 s. It's good for "where is it spinning", not for fine profiles.

**Real profile (ETW).** The Windows Performance Toolkit is installed (`wpr`, `xperf`, `wpa`). Run this in an elevated shell:

```powershell
wpr -start CPU -filemode
# ... reproduce ...
wpr -stop target\diagnostics\cpu.etl
wpa target\diagnostics\cpu.etl     # Trace > Configure Symbol Paths: add the exe folder, then Load Symbols
```

This hasn't been run by the tooling author: it needs elevation. To get symbols for a stripped release binary, build a profiling copy as the Linux recipe in `.agents/skills/AetherVk/SKILL.md` describes (`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`, own target dir).

## Caveats

- **Killing cdb mid-attach.** While cdb is attached non-invasively, the target's threads are suspended. If cdb is killed during the attach (the scripts' 120 s timeout), the target can stay suspended. With `symsrv.ini` in place an attach takes seconds.
- **No Windows render-stall watcher.** `render_stall_watcher.rs` and `dump_all_thread_backtraces_gdb` are Linux-only. On Windows, use `Watch-TestHang.ps1` / `Get-HungStacks.ps1` from outside.
