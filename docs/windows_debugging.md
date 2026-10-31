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
| `perf top -p <pid>` | `Get-StackSamples.ps1 -Id <pid>` (quick look) |
| `perf record -g` / `samply record` | `Record-Cpu.ps1` (samply over ETW, the same Firefox Profiler UI; elevated), or `Record-Cpu.ps1 -Wpr` for WPA |
| `RENDERDOC_CAPTURE` / `renderdoccmd` on Linux | `Capture-RenderDoc.ps1 -Test <path::to::test>` |
| choosing an SDK with `source setup-env.sh` | `. Use-VulkanSdk.ps1 -Version <x.y.z.w>` |
| `ls /proc/<pid>/task` + `top -H` (one thread at 100 %?) | `Get-StackSamples.ps1`: the CPU-per-thread table |
| In-process gdb dumps (render-stall watcher, `AETHERVK_GPU_LOST_DUMP`) | Already on Windows: minidumps on an unhandled exception, on a panic and on device loss, in `%LOCALAPPDATA%\Aether-Vk\logs` (`crash_dump.rs`). The render-stall watcher is still Linux-only. |

## One-time setup

```powershell
winget install Microsoft.WinDbg              # if WinDbg is not installed yet
scripts\windows\Setup-DebugTools.ps1         # copies cdb out of the Store package, writes symsrv.ini, prints the status
scripts\windows\Setup-DebugTools.ps1 -Status # status only (tools, SDKs, Vulkan implicit layers)
scripts\windows\Setup-DebugTools.ps1 -RemoveStaleLayers   # elevated: drop layer registry entries whose manifest is gone
cargo install --locked samply                # CPU profiler
winget install Microsoft.Sysinternals.Suite  # procdump, Process Explorer, handle
dotnet tool restore                          # dotnet-trace / -counters / -dump / -stack
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

## A clean Vulkan environment for tests

Every Vulkan process on this machine gets the registered *implicit* layers injected: OBS game capture, the Steam overlay and fossilize, Nsight Graphics, RenderDoc. `Setup-DebugTools.ps1 -Status` lists them. They change timing and can hide or cause bugs, so `.config/nextest.toml` turns them off for every test:

- `VK_LOADER_LAYERS_DISABLE=~implicit~`, with `VK_LOADER_LAYERS_ALLOW` keeping the NVIDIA Optimus / present and AMD switchable-graphics layers that pick the GPU.
- The layers the engine enables by name (`VK_LAYER_KHRONOS_validation`) are explicit and still load.
- To capture a test with RenderDoc or Nsight, clear the disable: `$env:VK_LOADER_LAYERS_DISABLE = ""` (nextest's own env wins over the shell's, so run the test exe directly, as `Capture-RenderDoc.ps1` does).

**Validation-layer versions.** Several SDKs install side by side under `C:\VulkanSDK`. `. scripts\windows\Use-VulkanSdk.ps1` lists them; `-Version latest` or `-Version 1.4.321.0` points this shell's `VULKAN_SDK`, `PATH` and `VK_ADD_LAYER_PATH` at one, so the tests load its validation layer. Use it to check whether a GPU-AV problem is a layer bug fixed upstream.

**Triage runs.** `cargo nextest run --profile triage --retries 0 ...` gives up on a test after 30 s instead of 15 min and runs everything (`fail-fast = false`). `--retries 0` is needed on the command line because the `vulkan-tests` override's retries beat the profile's.

## GPU and shader debugging

**Shader source lines.** The `.d.spv` shader twins are compiled with `-g` (`compile_shaders.sh`). With `AETHERVK_SHADER_DEBUG_INFO=1` the engine loads them for every compute pipeline, in any build, tests included. GPU-AV out-of-bounds reports then name the GLSL file and line instead of "Unable to source", and RenderDoc shows GLSL in its shader viewer and debugger. Without the variable, tests always use the plain `.spv`.

**Is it the driver's compiler?** A GPU result that differs between two blobs of the same source points at the driver, not the shader:
- `AETHERVK_SHADER_NO_VMM=1` loads the plain blobs instead of the `.vmm` (Vulkan memory model) ones.
- `AETHERVK_SHADER_DEBUG_INFO=1` loads the unoptimised `-g` twins.
- `spirv-opt -O --target-env=vulkan1.1spv1.4 -o <blob> <blob>` on one blob (back it up first) feeds the driver very different SPIR-V with the same meaning. Shaders load at runtime, so run the existing test exe directly (`target\debug\build\aethervk-core-rlib\<hash>\out\aethervk_core_rlib-<hash>.exe --exact <test> --nocapture`); going through cargo relinks.

This is how the 2026-10-31 dust mismatch was found: the GTX 1070's Windows driver (576.80) builds the unoptimised `dust_splat_mono.comp` blob wrong (twice the texels of the CPU reference) and the `spirv-opt -O` one right. The `dust_splat.comp` stage chain has the same problem: unoptimised, it spreads the spinning-site spiral and gives a peak about 8x low. `compile_shaders.sh` now always optimises both splat enumerators.

**RenderDoc capture of one test:**

```powershell
scripts\windows\Capture-RenderDoc.ps1 -Test gpu_backends::vulkan::device::dust::cadence::tests::gpu_dust_cadence_state_matches_reference -ShaderDebugInfo
```

- It runs the newest `aethervk_core_rlib-*.exe` test binary under `renderdoccmd capture` with `AETHERVK_RENDERDOC_CAPTURE=1`.
- In debug builds that variable makes each `Device` capture its whole life (`renderdoc::DeviceLifeCapture`): tests have no swapchain and no frames, so this is how their dispatches end up in a capture.
- The `.rdc` goes to `target\diagnostics\renderdoc\` and opens in qrenderdoc (`-NoOpen` to skip). `-Exe <program> -- <args>` captures something else.

**Nsight Graphics** (2025.4, installed) supports frame debugging on Pascal; GPU Trace needs Turing or newer. Launch the app or a test exe from its *Start Activity* dialog, with the implicit-layer disable cleared as above.

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

**Real profile: samply.** samply samples over ETW and opens the Firefox Profiler, the same workflow as on Linux. ETW needs an elevated shell.

```powershell
cargo build --profile profiling --example dust_observer   # release + line tables, not stripped
scripts\windows\Record-Cpu.ps1 -- target\profiling\examples\dust_observer.exe --frames 600
scripts\windows\Record-Cpu.ps1 -Id <pid> -Seconds 15       # attach to the running app
```

Profiles go to `target\diagnostics\profiles\cpu-<stamp>.json.gz` (reopen with `samply load <file>`).

**Who waits on whom: WPR + WPA.** `Record-Cpu.ps1 -Wpr -Seconds 20` (or `-Wpr -- <command>`) records CPU and context-switch ETW events to an `.etl`. Open it with `wpa`, then Trace > Configure Symbol Paths: add the exe folder, then Load Symbols.

The `[profile.profiling]` cargo profile is the one to profile on both OSes: a debug build symbolises but doesn't show release timing.

**.NET side.** `dotnet tool restore` installs `dotnet-trace` (`dotnet-trace collect -p <pid>`, then open the `.nettrace` in PerfView or VS), `dotnet-counters monitor -p <pid>`, `dotnet-dump collect -p <pid>` and `dotnet-stack report -p <pid>` (managed stacks of every thread, no debugger).

## Caveats

- **Killing cdb mid-attach.** While cdb is attached non-invasively, the target's threads are suspended. If cdb is killed during the attach (the scripts' 120 s timeout), the target can stay suspended. With `symsrv.ini` in place an attach takes seconds.
- **No Windows render-stall watcher.** `render_stall_watcher.rs` and `dump_all_thread_backtraces_gdb` are Linux-only. On Windows, use `Watch-TestHang.ps1` / `Get-HungStacks.ps1` from outside.
