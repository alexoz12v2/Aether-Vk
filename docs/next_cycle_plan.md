# Next cycle plan (after v1.0.1-12-sxs)

Written 2026-10-31 from: the open items of the previous plan and both session transcripts, a
tooling audit on the GTX 1070 (driver 576.80, SDK 1.4.363), and the reviewer's mail on
v1.0.1-9-sxs (`review.eml`, 18 points, attachment "vk test.pptx").

Theme of the cycle: reviewer-driven features and rework, together with a debug/profiling session
and Windows graphics-driver tooling.

## 0. First jobs (blockers and missing verification)

| # | Item | Notes |
|---|---|---|
| 0.1 | Linux CI compile (exit 143) | cause: CI rebuilt all shaders (fresh-checkout mtimes) and the Linux glslang of SDK 1.4.363 never finishes `dust_draw_volume.comp` with `DUST_SUBGROUP=1 DUST_PROFILE=1` (>7.5 GB; Windows: 0.3 s). Fixed with `AETHERVK_PREBUILT_SHADERS=1` in CI. Next: reduce the shader to a minimal repro and report it to KhronosGroup/glslang; a Linux developer running `compile_shaders.sh` still hits it |
| 0.2 | Run the nextest `slow` tier to completion | `CARGO_PROFILE_DEV_OPT_LEVEL=2 CARGO_TARGET_DIR=target/opt2 cargo nextest run -p aethervk-core-rlib --profile slow`, ~3.5 h, hands-off; no result exists yet for any of the 22 |
| 0.3 | Fixtures for 4 tests (`BoxTextured.glb`, `review_67p` SPK) | fetch step in the test setup, or skip-if-missing with a clear message |
| 0.4 | `compile_shaders.ps1` stale (no `spirv-opt -O`, no `.vmm` variants) | make it call `compile_shaders.sh` through Git Bash, or delete it |

## 1. Reviewer feedback (review.eml), by priority

1. **Dates**: one clear control for the observation date (the image) and the generation period
   (N rotations before it), instead of two independent ranges.
2. **All controls on one page** (no hunting across tabs for a basic run).
3. **Nucleus orientation view side by side with the model view** (also docx item #13).
4. **Grain-size law as diameter range + exponent**, dropping F: s0 = sqrt(min*max),
   F = sqrt(max/min) map exactly. Explain which grains are simulated: the continuum is fixed at
   q = 3.5, per-jet q only changes the specks (say so in the UI and the help).
5. **Declutter**: remove the "Comet" label (`CameraService.cs:2534-2549`); readable jet-angle
   text (now tiny and yellow); legend for the red/blue gizmo lines; star field toggle; explain
   "Jet Preview Size" (nucleus radius / 50); image export with black or transparent background.
6. **Trajectories**: show only the central one by default, smaller tracer dots, explain or hide
   "Flow Pulses".
7. **Physical lighting as the default in Nucleus view**, with an arrow legend.
8. Remaining points of the mail (lower priority) and the drafted answers to the reviewer's
   questions: reply once 1-4 are in a build.

Also open from the earlier feedback (`docs/feedback_terza_versione.md`): #12 Gaussian jet spread
and per-jet density (L), #2 orbital-elements table wiring (S), visible busy state while "Go to
epoch" jumps (`TimelineTabViewModel.cs:376`). Update rows 4/15/20 of that file: those tests passed
on 2026-10-31.

## 2. Profiling session (perf)

Measure first, then fix:

- H5: one samply and one Tracy capture of 10 s/s playback **of the .NET app** (never captured),
  plus Nsight Systems `--trace=vulkan` for GPU vs submit bound.
- Playback 24 fps vs 55 on the 1070. Candidates: stride the propagate (recommended), EWA bias,
  Balanced cap 64, fps limit following the preset budget.
- Budget ladder: keeps raising the volume bias after saturation and never engages the haze
  detail knob (`device/dust.rs` ~1345-1380, `dust_volume_draw.glsl:74`, flags bits 11-13); bottom
  rung 2.0 s/frame vs 172-179 ms on 2026-10-19.
- oshal `math.rs` sqrt/reciprocal: AVX check and a non-inlinable asm call on every call (~1.3x on
  the CPU volume path); dispatch once or use `#[cfg(target_feature)]`/intrinsics.
- H4: whole suite in one `cargo test` process, and the loader pin experiment
  (`docs/vulkan_test_loader_stall.md:95`).

## 3. Windows GPU and driver tooling

Already there: device_fault, NV checkpoints, TDR event capture, minidumps on device loss,
pipeline executable statistics, VMA budget, calibrated timestamps into Tracy, RenderDoc, render
stall watchdog, 11 scripts in `scripts\windows\`.

| Prio | Tool | Effort | Why |
|---|---|---|---|
| P1 | Crash Diagnostic Layer (`VK_LAYER_LUNARG_crash_diagnostic`) switch in `Invoke-UnderDebugger`/`Watch-TestHang` | S | the 0 % CPU waits inside the driver: which command buffer never finished |
| P1 | GFXReconstruct capture/replay script (`Capture-GfxRecon.ps1 -Test`) | S | replayable stream of a failing GPU test, for NVIDIA (splat miscompile) and VVL #13615 |
| P1 | Debug-utils object names: wire `set_object_name` (`debug_labels.rs:40`, 0 call sites) | S-M | names in RenderDoc, Nsight, GPU-AV and CDL output |
| P2 | Nsight Aftermath (opt-in feature, `VK_NV_device_diagnostics_config`) | M | GPU crash dump with faulting shader line; Pascal supported, driver >= 531 |
| P2 | Nsight Graphics frame debugging on the 1070: verify 2025.4 works (Pascal dropped only in 2026.1), fix `docs/windows_debugging.md:155` | S | SASS of the `-O` vs unoptimised splat blobs |
| P2 | Nsight Systems wrapper (`Record-Gpu.ps1 -Nsys`) | S | playback and budget-loop timelines |
| P2 | GPUView / WPR GPU ETW profile | S | kernel scheduler view of stalls, paging, TDR |
| P2 | `Set-Tdr.ps1` (TdrDelay/TdrDdiDelay/TdrLevel, `-Status`/`-Reset`) | S | budget-loop frames of 2-7 s exceed the 2 s default |
| P2 | vkconfig presets (tests-clean, gpuav-printf, crash-diag, api-dump, gfxrecon) and the GPU-AV shared-memory-race setting until #13615 is fixed | S | layer stacks for the app, GPU-AV usable on the dust shaders |
| P3 | API dump switch, pipeline statistics CSV, periodic VMA stats, driver known-bad list (576.80 needs `-O` splat blobs) | S each | diffing runs, spills, paging, silent regressions |

Still unverified from the previous plan: RenderDoc capture of a dust test (D2), `dotnet-counters`
on the app (F), `Enable-AppVerifier.ps1 -Enable` (G1, elevated).

## 4. Suggested order

1. CI fix and release v1.0.1-12; start the `slow` tier in the background.
2. Zero-code tooling (CDL, GFXReconstruct, vkconfig presets, TDR script) and object names.
3. Profiling session (H5, nsys, GPUView), then the perf fixes; H4.
4. Reviewer points 1-4, then 5-7; reply to the reviewer.
5. Aftermath and the Nsight verification; leftover features (#12, #2) and verification checklist.
6. Watch VVL #13615 and retest on the next SDK.
