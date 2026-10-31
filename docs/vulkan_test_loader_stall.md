# Rust GPU test suite stalls on Windows / NVIDIA

## Symptom
Seen on Windows 10 19045 with a GTX 1070, Vulkan SDK 1.4.321.0, debug build (validation layers and GPU-AV on).

`cargo test -p aethervk-core-rlib --lib -- --test-threads=1` hung twice at test #87 of 656:
- The process sat at 0 % CPU with 13 threads.
- The last output each time was the `vkCreateDevice` log (`Using "NVIDIA GeForce GTX 1070"`, `[Disabling gpuav_validate_ray_query]`).
- The hung test differed between the runs (`test_commands::command_pools_take_back_every_pool_after_a_burst`, then `test_commands::stress_test_command_pool_rotation` once the first was skipped).

With `--test-threads=4`, the suite crashed instead with an access violation (0xc0000005) after about 67 tests.

## Diagnosis
**1. The loader's reference count is unbalanced; this is the main suspect.**
- `gpu_backends/vulkan/utils.rs` (`EntryWrapper::new_internal`) takes an already-loaded `vulkan-1.dll` with `GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT`, so it adds no reference.
- `Drop for EntryWrapper` calls `FreeLibrary` once its own `Arc<Entry>` is the last one.
- Tests skip the shared `S_VULKAN_CORE` cache (`gpu_backends/vulkan.rs`), so every GPU test builds its own entry.

Consequences:
- **Run sequentially:** every GPU test unloads `vulkan-1.dll`, probably taking the validation layers and the NVIDIA ICD (`nvoglv64.dll`) with it, and the next test loads them all again. Driver state leaked over about 86 unload/reload cycles is the likely cause of the positional hang.
- **Run in parallel:** test B frees a DLL that test A loaded and is still using, which explains the access violation.

**2. Instance churn.** With GPU-AV wanted, `instance.rs` creates and destroys a throwaway instance (the Lavapipe check) before the real one, so there are two instance cycles per test.

**3. The hung tests are both submit bursts.** `command_pools_take_back…` does 1 920 submits under GPU-AV. Accumulated state could end in a blocked `vkDeviceWaitIdle` / `vkQueueSubmit`, or in the polling worker blocked in the driver while `Device::drop` (`device.rs`, the stop-signal loop) waits on it with no timeout.

**4. Teardown is otherwise correct.**
- Device → instance order is right, and the debug messenger is destroyed.
- The per-device `ThreadPool` is joined.
- 13 threads at the hang means pools don't pile up.

**5. Online reports.** Nothing matches exactly. The closest is NVIDIA reporting `vkCreateInstance` failing after about 30 create/destroy cycles in one process (forums.developer.nvidia.com, thread 176978).

**6. An unrelated stale registry entry.** Every instance logs `loader_get_json: Failed to open JSON file ...Nsight Systems 2019.5.2...VkLayer_nsight-sys_windows.json`. It's a leftover implicit-layer value under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers`; the layer is skipped.

## Scope: `cargo test` only, not nextest
The supported runner is `cargo nextest` (`testing_guide.md`, the `cargo nt` alias, CI).
- nextest runs every test in its own process, so each process loads and frees the loader once and nothing accumulates. A process-wide pin is a harmless no-op there.
- The stall appears only under plain `cargo test`, which runs all ~650 tests in one process (`RUST_TEST_THREADS = 1` from `.cargo/config.toml`).
- `.config/nextest.toml` already sets `VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1`, for a GPU-AV descriptor leak seen during rapid setup and teardown. That's the same family of problem.

## Experiments, cheapest first
0. **Loader env var, no code change.** Run `cargo test` with `VK_LOADER_DISABLE_DYNAMIC_LIBRARY_UNLOADING=1`: the loader then never frees the ICD and the layers, even when `vulkan-1.dll` itself goes.
1. **Pin the loader.** In tests, never `FreeLibrary` `vulkan-1.dll`. If the stall goes away, cause 1 is confirmed.
2. **Turn validation off in steps:** `AETHERVK_DISABLE_GPU_AV=1`, then add `AETHERVK_DISABLE_SYNC_VAL=1`, then `--release` (no layers).
3. **Watch per-test growth during the suite:**
   - `Get-Process aethervk*`: `HandleCount`, `Threads.Count`, `PrivateMemorySize64`, every 2 s;
   - `nvidia-smi --query-gpu=memory.used --format=csv -l 2`.
4. **Get stacks of a hung run.** This needs WinDbg (`winget install Microsoft.WinDbg`).
   - The store package's `cdb.exe` can't be run in place (WindowsApps denies execution). Copy the `amd64` folder elsewhere (e.g. `%LOCALAPPDATA%\dbg\amd64`) and use it from there.
   - Attach to the live process without stopping it: `cdb -pv -p <pid> -y "<sympath>" -c ".reload /f; lm; ~*kn 80; !locks; q"`.
   - The NVIDIA ICD has no public symbols, so its frames show as `nvoglv64!<export>+offset`. Which thread is inside it, and the caller frames from our PDB, are still enough to see where the wait is.
   - Take the dump: Task Manager → Details → *Create dump file*, or `procdump -ma <pid>`.
   - Open it: `windbgx -z <dmp>`.
   - Set symbols: `.sympath srv*C:\symbols*https://msdl.microsoft.com/download/symbols;<the test exe folder: target\debug\build\<pkg>\<hash>\out with the pinned nightly; the PDB path embedded in the exe also works>`, then `.reload /f`, then `~*kn 50`.

## Fix direction
- Keep `vulkan-1.dll` loaded for the life of the process (a never-freed static), or always take a real reference so each `FreeLibrary` is balanced.
- Cache the GPU-AV / Lavapipe check instead of creating a throwaway instance per test.
- Give the `Device::drop` stop wait and the `u64::MAX` fence waits a timeout that logs.

## Results
**2026-10-31: the `test_commands` hang is a deadlock inside the validation layer's GPU-AV.** It is not positional.

**It hangs on its own.** `cargo test ... gpu_backends::vulkan::device::test_commands` in a fresh process hung on its first test (`command_pools_take_back_every_pool_after_a_burst`) after 3.4 s of CPU.

**Stacks** (`cdb -pv`, WinDbg 10.0.29661), with three threads blocked inside `VkLayer_khronos_validation.dll` (SDK 1.4.321.0):
- The test thread, in a Vulkan call from our code: `RtlAcquireSRWLockShared`, waiting.
- Our `worker_0` (the device polling workload, i.e. the timeline-semaphore poll): `RtlAcquireSRWLockExclusive`, waiting.
- A thread the layer started itself (`ucrtbase!thread_start` → the layer): `MSVCP140!mtx_do_lock` → `RtlAcquireSRWLockExclusive`, waiting.
- The rest is idle: NVIDIA ICD worker threads, and `graphics_hook64`, the OBS game-capture hook injected into every Vulkan process.

This is a three-way lock-order deadlock between the test thread, our polling thread and the layer's internal thread.

**Our frames came out as raw offsets.** The test exe's PDB wasn't picked up, so they show as `aethervk_core_rlib_<hash>+0x…`. Resolve them with `ln` and the PDB path for the exact Vulkan entry points.

**The same test with `AETHERVK_DISABLE_GPU_AV=1`** passes in 16 s.

**The separate `cargo test` abort** (all tests in one process): `Exceeded MAX_TRACKED_THREADS for lock tracking` (`gpu_backends/vulkan/device/locks.rs:52`). The debug lock tracker's table of thread hashes only ever grows. nextest, with one process per test, doesn't hit it.

Next steps:
- Resolve our frames.
- Try a newer SDK's validation layer.
- If the deadlock remains, report it upstream (KhronosGroup/Vulkan-ValidationLayers) with the repro test.

The loader pin experiment for the positional stall is still pending.

## Tooling
Getting the stacks of a hung run on Windows is now scripted (see [windows_debugging.md](windows_debugging.md)):
- `scripts\windows\Watch-TestHang.ps1 -- cargo test -p aethervk-core-rlib --lib -- --test-threads=1` writes every thread's stack, plus a minidump, of a test process that sits at about 0 % CPU for 30 s. The output goes to `target\diagnostics\hangs\`.
- `scripts\windows\Get-HungStacks.ps1 -Id <pid> [-Dump Full]` does the same for one process by hand.
- `scripts\windows\Setup-DebugTools.ps1` writes a `symsrv.ini` exclusion list, so `nvoglv64.dll` no longer stalls symbol loading, and no `.reload /f` is needed.
- The symbol path in item 4 points at `target\debug\deps`, which is stale. The pinned nightly cargo puts test exes in `target\debug\build\<pkg>\<hash>\out\`; the PDB is found through the path embedded in the exe.
