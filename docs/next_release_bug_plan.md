# Next release: bug plan

Planning notes for eight open issues (2026-11-01). Nothing here is implemented; root causes marked
*hypothesis* were read from the code, not reproduced. Line numbers are as of commit `1d78bbc1`.

Effort: S = under half a day, M = one to two days, L = more.

---

## 1. e2e `SaveReopened_WithTheWindowOnTheSecondMonitor_KeepsTheBillboardOverTheViewport`

**Symptom.** After `Win32.MaximizeOn(second)` (secondary monitor at x = −1920) the billboard's
centre is at (−820.8, 462.2), on the second monitor, while the viewport read through WinAppDriver
is still at (534, 130) with a window shift of (0, 0).

**Root cause (hypothesis).** Two coordinate sources disagree and the test's correction cannot
reconcile them:

- The app's billboards use `VulkanViewportControl.PushOrigin` →
  `PointToScreen(0,0)` (`aethervk.ui-app/Controls/VulkanViewportControl.cs:108-119`), i.e. true
  desktop px, negative on the left monitor. That is the −820 figure, and it is right.
- The test reads the viewport through WinAppDriver (`s.Find("ViewportHost").Location`) and shifts
  it by `WindowCorner(real) − App.Manage().Window.Position`
  (`aethervk.ui.e2e/FitsComparisonSteps.cs:266-271`, `BillboardMonitorTests.cs:261-265`). The log
  shows shift (0, 0): WinAppDriver's *window* position agreed with DWM's, but the *element*
  location stayed at the primary-monitor value (534). WinAppDriver's element `BoundingRectangle`
  is known to be unreliable for windows at negative desktop coordinates (reported relative to the
  window or clamped), so the shift derived from the window rectangle does not apply to it.
  The comment at `FitsComparisonSteps.cs:259-264` documents a different earlier observation (the
  shift was the window's), so the behaviour depends on the WinAppDriver/UIA path taken.

**Proposed fix (test side, preferred).** Stop deriving the viewport rectangle from a WinAppDriver
element:

1. Expose the viewport's desktop rectangle from the app as text, next to the billboard geometry:
   a `TextBlock` with `AutomationId="ViewportDesktopRect"` bound to the origin and size
   `VulkanViewportControlViewModel.NotifyScreenOrigin` already receives (the Billboards section of
   `SettingsTabView.axaml`, where `BillboardGeometry` lives at line 507, or the camera HUD).
2. In the e2e, `DesktopLocation(s, vp)` becomes "read `ViewportDesktopRect`"; keep the WinAppDriver
   path as a fallback when the element is absent. `CornerOffset` and `AssertBillboardOverTheViewport`
   use it; `CometScreenPx` too (it adds the HUD's viewport-relative comet px to the same origin).

Alternative without an app change: find the Vulkan child HWND under `s.MainWindow`
(`EnumChildWindows`) and `GetWindowRect` it. Works only while the viewport is a native child window.

**Risk.** Low. The new text is read-only and diagnostics-only. If the HUD/settings control is not
visible when read, the test must expand the section first (it already does for `BillboardGeometry`).

**Tests.** Rerun the three `BillboardMonitorTests` on the two-monitor machine. Add a logic test
that `NotifyScreenOrigin(-1920+534, 130, 1.0)` formats the new text with negative x.

**Effort.** S (app text + test helper).

---

## 2. e2e `Billboards_FollowTheCometAndTheViewportCorner_ThroughOrbitZoomResizeAndReopen`

**Symptom.** `AssertNucleusOnComet`: nucleus at (1245.5, 395.0), comet at (1095.5, 465), 165.53 px.
The offset is exactly (+150, −70).

**Root cause (hypothesis, strong).** The test's orbit drag is `DragFromTo(cx−60, cy+40, cx+90, cy−30)`
(`BillboardMonitorTests.cs:112`): a delta of (+150, −70). The FITS billboard is a separate
transparent top-level window over the viewport (`DesktopBillboardWindow`), sitting on the comet at
the viewport centre. The press lands on its image, so `OnPointerPressed`
(`aethervk.ui-app/UI/DesktopBillboardWindow.cs:140-153`) hands it to the controller, a left press
enters `State.Moving` (`BillboardInteractionController.cs:50-53`) and `PointerMoved` translates the
billboard by the delta (`:77-78`). The billboard moved by (150, −70); the camera never orbited. The
"nucleus bound to the comet" invariant is not enforced during a manual move, so the billboard stays
where it was dropped until the next recentre.

**Proposed fix.** Two parts:

- *App:* a billboard whose nucleus pixel is bound to the comet should not be translatable by a plain
  left drag (the bound point defines its position). In `BillboardInteractionController.PointerPressed`
  treat `Left` on a comet-bound billboard as a no-op (or let the press fall through to the viewport:
  `DesktopBillboardWindow.OnPointerPressed` returns without `Handled` so the viewport orbits). Scale
  and rotate handles stay. Alternatively re-anchor on release (`PointerReleased`) by snapping the
  nucleus back onto the comet, keeping the user's intent for size/rotation only.
- *Test:* start the orbit drag off the billboard: read `ReadGeometry(s, FitsRow)` and pick a start
  point outside its rectangle (e.g. the viewport's lower-left quarter), and assert the comet moved
  (`cometBefore != cometAfterOrbit`) so a swallowed drag fails loudly.

**Risk.** Medium for the app part: users may rely on nudging a bound billboard. Document that moving
a bound billboard changes its nucleus binding, or gate the behaviour behind the gizmo's move handle
only (right-button `MoveFree`), which already exists.

**Tests.** `aethervk.logic.tests`: a `BillboardInteractionController` test that a left drag on a
comet-bound billboard returns null / the same geometry; e2e assertion above.

**Effort.** S (test) + S/M (app).

---

## 3. e2e `ThesisScenarioTests` "67p-2021-12-27-south": ortho half-width 61650 → 103300 after reopen

**Symptom.** `AssertOrthographic` (`ThesisScenarioTests.cs:491-497`) expected 61650 km, got 103300.
103300 / 61650 = 1.6756, a viewport aspect (W/H ≈ 1120×668).

**Root cause (verified: the app is right, the test's expectation was stale).** The save already
held the half-width 103 300 km: `TestResults/e2e/67p-2021-12-27-south/67p-2021-12-27-south/
aethervk_index.json`, `viewports[0].camera.projection`: `right = 0.00069054175 AU` (103 304 km),
`top = 0.00041198838 AU` (61 633 km), `aspect = 1.6761194`; the reopen restored it faithfully
(`RestorePersistedState` → deferred `ApplyProjectionSnapshot`, `CameraService.cs:1125-1126`,
`:2650-2663`, applied after the tracking preset of `:545`/`:1735-1736`).

The value changed in the *build* session, after `AssertFan` (which uses the 61 650 km half-width for
its km/px) and before Save As. Only "south" has a `Fan` (`Scenarios.cs:175-181`), so only it runs
`FitsComparisonSteps.Run` (`ThesisScenarioTests.cs:82-89`) instead of the billboard step: "Set up
comparison" (`FitsImageViewModel.SetUpComparisonAsync:460-489`) → `DesktopBillboardsViewModel.
MatchCameraToImage:484-498` ("the viewport's height becomes the image's field") →
`CameraService.SetEarthObserverField:2085-2110`, by design (doc `:2078-2079`, logic test
`CameraServiceTests.SetEarthObserverField_Orthographic_HalfHeightIsHalfTheImageHeight`):

```
2107  double halfH = heightKm * 0.5 / AuToKm;   // the 1200 px × 0.3037″ stack at Δ: 123 265 km → 61 633 km
2108  float halfW = (float)(halfH * aspect);    // × 1.6761 = 103 304 km
```

The FITS is square, so fitting its height to a landscape viewport shows the whole frame. The
scenario constant 61 650 km is the thesis figure's *half-width* (`Scenarios.cs:154`); after the
comparison the framing is "image height = viewport height", and the two disagree by the aspect.
Note the field is 123 265 km (half 61 633), not 123 300: even a width-based match would miss the
test's exact (precision 0) check by 18 km. None of the earlier hypotheses (preset after snapshot,
VM lock/unit recompute, `OnViewportResized`) applies: the reopened value equals the saved one.

**Fix (done, test-only; no app change).**

1. `ThesisScenarioTests.cs`: `AssertOrthographic(s, sc.OrthoHalfWidthKm)` runs in the build
   session before the fan block (where "tracking and the run must not have replaced the projection
   the user set" holds for every scenario); the half-width is read just before Save As and the
   reopen checks the round trip against it (`AssertRestored(s, sc, savedHalfWidthKm)`); a Fan
   scenario also asserts the comparison framing (`AssertComparisonFraming`: half-height = the
   FITS row's conversion km / 2, half-width = half-height × the viewport's aspect, 0.5 %).
2. `FitsComparisonSteps.FieldKmAtComet` exposes the conversion's km; `Scenarios.cs` documents
   that `OrthoHalfWidthKm` is the framing before the comparison.
3. `CameraServiceTests.RestorePersistedState_Orthographic_KeepsTheSavedFrustumAfterTheTrackingPreset`
   guards the restore ordering the hypotheses worried about: a save made after
   `SetEarthObserverField` reopens on the same L/R/B/T after the tracking preset, a resize at the
   same aspect keeps it, a new aspect re-derives the width from the height.

**Risk.** Low (tests only).

**Effort.** S.

---

## 4. Slow tier `scene::dust::tests::cadence_keeps_the_light` hit the 2 h kill limit

**Does today's `light_tol = 0.0` change slow it?** No, not this test. `cad_run_b`
(`aethervk.core/rlib/src/scene/dust/tests.rs:12409-12426`) sets `light_tol = 0.0` only when
`scenario` is true; `cadence_keeps_the_light` calls `cad_run_b(3.0, 270, false, "B")` (`:12401`) and
keeps `DUST_CADENCE_LIGHT_TOL`. The only caller with `scenario = true` is
`cadence_keeps_the_light_whole_clusters` (`:12770`). The last commit touching `tests.rs` added only
those lines (6 insertions), and `cadence.rs` was last changed on 2026-10-30. The test is unchanged in
work; the time-out is environmental or a pre-existing underestimate:

- The 50 min figure (`.config/nextest.toml`, `[profile.slow]` comment) was measured at opt-level 2
  on 16 threads. Each frame of run B also `settle`s the 8192-slot, 3-tier system and runs the lazy
  propagate; with the GPU e2e tests (PlaybackPerf) or the app on the same machine the per-frame
  time doubles easily. `[t8 time]` lines are printed per frame, so the real per-frame cost is in the
  log of the killed run.
- `cadence_keeps_the_light_whole_clusters` *does* get slower with `light_tol = 0`: without the light
  bound more members are admitted to the slow classes, so every build pass carries more clusters.
  Watch it on the next slow run.

**Proposed fix.**

1. Split the test: `cadence_keeps_the_light_static` (run A, 230 frames) and
   `cadence_keeps_the_light_moving` (run B, 270 frames). Each gets its own kill budget; a failure
   names the half. Update the three `default-filter` lists in `.config/nextest.toml`.
2. Make the frame counts overridable: `DUST_TEST_FRAMES` (like `DUST_TEST_THREADS`, `:4331-4334`)
   scaling 230/270 down for a smoke run; the full counts stay the default for the slow profile.
3. Add a `[[profile.slow.overrides]]` for the two halves with `terminate-after = 180` (3 h) and a
   note of the measured per-frame time, or lower `px` for run B from 128 to 96 if the asserts are
   insensitive to it (the Σ-ratio and ghost-share bounds are relative).
4. Print a cumulative elapsed line every 10 frames so a killed run reports how far it got.

**Risk.** Low. Splitting changes test names (CI filters, docs). A smaller `px` changes the printed
D3 lines and may move the 1 % class-share margins; keep 128 unless measured.

**Tests.** The test itself; a one-off timing run of each half with
`CARGO_PROFILE_DEV_OPT_LEVEL=2 ... --profile slow -E 'test(=...)'` while nothing else runs.

**Effort.** S.

---

## 5. Playback perf: 24 fps vs 55 on a GTX 1070 (e2e `PlaybackPerfTests`)

**Investigated 2026-11-01 (GTX 1070, driver 576.80, Windows 10, Release engine, Low quality,
Balanced preview, the reviewer's 67P save).** Not a bug in the engine's frame pacing: the frame is
GPU work-bound inside the Balanced controller's own band. The e2e's floor was recalibrated to that
band; the engine is unchanged.

**Measurements** (`TestResults/e2e/playback-perf/perf.csv`, five runs, the baseline configuration identical within 1 ms across them):

| phase | dust fps | total ms | propagate | splat | packet queue draw | stride |
|---|---|---|---|---|---|---|
| 10 s/s | 24–24.5 | 35.2 | 11.0 | 4.4 | 19.0 | 32 (Balanced's cap) |
| 1 d/s | 23–23.5 | 35–39 (grows with the live clusters) | 11–12.5 | 4.5 | 18.6–21 | 32 |
| paused full render (calibration) | – | 781 | – | 755 (splat + queue) | – | 1 |
| 10 s/s, queue off (`AETHERVK_DUST_QUEUE=0`, same image) | 26.5 | 32 | 11.0 | 20.5 | 0 | 32 |
| 1 d/s, queue off | 26 | 32–34 | 11.5 | 20.7 | 0 | 32 |
| paused full render, queue off | – | 799 | – | 798 | – | 1 |

- **GPU-bound, not sync-bound.** `nvidia-smi` every 2 s through both phases: utilisation 100 %,
  P0, SM 1923 MHz (max 1999), no power or thermal cap; ~44 % paused. The render thread presents
  60 frames/s throughout (`presented frame N` lines: 600 per 10 s), so present mode / vsync is not
  the limiter: the non-dust frames reuse the image (`cmd_dust_frame_begin_reusable`). No
  `Physics step backlog` lines during the measured seconds; the two per run sit at the Play
  clicks, where the first preview frame draws every cluster (stride 1, 418–670 ms) before the
  controller's first readback.
- **Why 24 and not 55.** `PlayStride::absorb` (play_quality.rs) raises the stride only when a frame
  exceeds 1.25 × budget = 41 ms, lowers it under 0.5 × (effectively 0.4 × after the power-of-two
  rounding), and Balanced caps the stride at 32. A 35 ms frame is inside the band: the controller
  is content, by design, with anything between 24 and 60 fps, and spends the budget on cluster
  density. The e2e's `FpsPerScore = 60` was Smooth's budget (16 ms), not Balanced's (33 ms).
- **Where the 35 ms go.** Propagate: 11 ms, already at the minimum trajectory detail (5), over the
  live range of the four systems every frame (never skipped while the epoch moves). Drawing: 23 ms
  for 1/32 of the clusters, split between the in-thread splat (4.4 ms) and the packet queue draw
  (`dust_draw_queue.comp`, 19 ms). `AETHERVK_DUST_PROFILE=1` at stride 32 in play: 147 k cluster
  threads, 496 k packets, **209 k queue records holding only 1.24·10⁷ texels (~60 per record)**,
  slowest workgroup 75 k cycles, lane-cycles qrows 4.0·10¹¹ of which qseg 0.55·10¹¹ — at the play
  pin's EWA bias 3 every packet is capped to 1/64 of its texels, so the queue (meant for the big
  bias-0 packets, up to 4096 texels a record) is full of packets barely past its 64-texel
  threshold and spends ~85 % of its time in per-record set-up. Drawing them in-thread instead
  (queue off, rows above) costs 16 ms for the same packets: the queue is overhead-heavy but the
  packets are expensive either way, a ±10 % lever, and the paused render slightly prefers the
  queue. Not changed.
- **A/B, EWA bias 5** (`AETHERVK_DUST_PLAY_FEEDBACK=0,5`, env only): the queue draw vanishes
  (0.003 ms: no packet reaches the 64-texel threshold), splat 12.5 ms at the stride the controller
  then settles on (16), total 24 ms, 33 fps; at stride 32 it would be ~17 ms (~55 fps). **Rejected
  by the reviewer**: the preview quality dropped visibly (the large coma packets blur). Not applied.

**Fix applied: the e2e's limits** (`aethervk.ui.e2e/PlaybackPerfTests.cs`): the fps floor is the
band's slowest frame less a 15 % margin (`0.85 × 1000 / (1.25 × 33)` = 20.6 fps at score 1, floor
12, cap 55); the median / p95 frame-time limits are `1.25 × 33 / score` and twice that; the
backlog count covers the measured seconds, not the warm-up with the Play click. With these the
reference machine passes with the engine as it is (23–27 fps; medians 32–38 ms). The reviewer's
steer (2026-11-01): no fps bought with preview quality; a balanced default between 24 and 55 fps.

**Recommended engine fix, no quality cost (the next release):** a *strided propagate* for the
preview. While the splat draws only the slots that are multiples of `S`, the propagate still
evaluates every live cluster (11–12.5 of the 35 ms; the undrawn clusters' moments are unused by
a preview frame). Evaluating only the strided slots (`dust_propagate.comp`: slot = `first_slot +
i·S`, the chunks over `live_count / S`; the `propagated` key must not mark the tier done, so the
first paused frame re-propagates every cluster) would take the frame from 35 to ~24 ms (~40 fps)
with the same image. Second, smaller: the controller's raise threshold at 1.0 × budget instead of
1.25 × keeps the frame under 33 ms with no visual change when a stride step is available (none on
the 1070 at Balanced, which is at the cap). Both need the GPU reference tests
(`gpu_dust_*_matches_reference`) and the propagate skip tests.

**Resolved (2026-11-02, Windows e2e pending):**
- The e2e now reads the committed range, runs its UIA searches before the Play click, and refuses
  to measure a stopped clock (`PlaybackPerfTests.CheckRoom`).
- A play session now starts at the previous session's stride (`PlayStride::play`).
  Superseded: every play session now starts at the preset's cap
  (`play_quality::session_start_stride`); the stride memory was removed because a remembered low
  stride made the next, heavier session's first frames slow.
- The dust view now keeps its host across the Play/Pause rebuild. Play turns the temporal reuse
  off and Pause back on, which changes the class-image span and recreates the view. The new view
  now takes the old host (`DustViewHost::carried_over`: white point, median, unit, feedback, play
  pin, stride, frame counter) instead of a fresh one; only the readback slots and the drawn image
  are forgotten. Before, the first second after Play was dark (the white re-seeded) and the pin
  was entered on a cold host. Tests: `carried_over_keeps_the_exposure_feedback_pin_and_stride`
  and `gpu_dust_view_keeps_its_exposure_when_the_cadence_toggles`.

The original note follows. **Open:** the e2e's 1 d/s phase hit the committed range's end 8 s in (epoch 693921600) in one of
five runs (and in the 17:26 run) after the seek to start + 1 h: the comment's "16 days" margin does not hold for this
save; `dust_frame` and the clock then freeze and the fast-movement check fails for a reason that
is not perf. Also the Play-click transient (one dropped physics step per Play) could be avoided by
starting the preview at the stride of the last play session instead of 1.

Original plan, kept for reference:

1. **Reproduce with the perf line.** Run the app with `AETHERVK_PERF_LOG=1` on the reviewer's 67P
   save at 1 d/s, Low quality, Balanced preview (what `PlaybackPerfTests.cs:8-17` does). The
   `[perf]` JSON (`aethervk.ui-logic/Services/PerfLogService.cs:76`) carries the dust pass timings
   and the drawn stride: compare the splat/propagate/classify/builds ms against the 55 fps baseline
   to see whether the budget loop is striding (CPU-bound logic) or the GPU passes grew.
2. **Which thread.** `scripts\windows\Get-StackSamples.ps1 -Name AetherVk -Samples 30` for a first
   look at the hottest leaf per thread (`docs/windows_debugging.md`, "CPU profiling"). If the logic
   or render thread is saturated, go to `Record-Cpu.ps1 -Id <pid> -Seconds 15` (samply + Firefox
   Profiler) on a `--profile profiling` cdylib.
3. **Who waits on whom.** `Record-Cpu.ps1 -Wpr -Seconds 20` and WPA's CPU Usage (Precise) for the
   render ↔ logic ↔ UI hand-offs: a 24 fps plateau that is not CPU-bound points at a wait (fence,
   timeline semaphore, `ps.dust` lock spin at `logic_thread.rs:3583-3584`, present).
4. **Frame anatomy.** Build the cdylib with `--features profile-tracy`, run `tracy-capture`, and read
   the GPU row (`dust frame` > propagate / classify / builds / reproject / specks / splat) against
   the CPU zones (`logic: tick`, `render: submit`, `vk: present`). This tells whether the 33 ms
   budget is spent in GPU passes (then: reduce builds per frame, class ladder) or in CPU recording.
5. **GPU side.** `Capture-RenderDoc.ps1` for one frame at each speed to count dispatches and
   barriers; check the swapchain present mode (FIFO caps at 60; mailbox/immediate for the
   measurement) and whether the compute and graphics queues serialise.
6. **.NET side.** `dotnet-counters monitor -p <pid>` for GC and thread-pool pressure (the 500 ms HUD
   timers, `Observable.Timer` polls); `dotnet-trace` if the UI thread blocks the FFI calls listed in
   the debugging doc.
7. **Bisect by setting.** Toggle preview Off/Smooth/Balanced/Quality and particle quality in the
   same run; the e2e's `perf_reference.json` calibration seeks give a fixed workload for A/B.

Deliverable: a table of per-pass ms at 24 vs 55 fps and the owning thread, then a targeted fix.
**Effort.** M for the investigation.

---

## 6. Shutdown crash `0xC0000005` in `avkSimulationContext_earthObserverStatus`

**Root cause (verified from code).** `CameraMatrixDebugViewModel` polls the runtime every 500 ms on
the background scheduler (`aethervk.ui-logic/ViewModels/Debug/CameraMatrixDebugViewModel.cs:122-144`):
`DustStats()`, `EarthObserverStatus()`, `CometEphemeris()`, `DebugCameraState()`. The runtime's
`EarthObserverStatus` guards only `_ctx == 0` (`INativeRuntimeService.cs:2741-2749`), but `_ctx` is
assigned once at startup (`:2237`) and never cleared: `ShutdownSync` (`:2243-2247`) calls
`avkSimulationContext_shutdownSync(_ctx)` and leaves `_ctx` as is. The app's `Exit` handler does not
dispose the runtime (`App.axaml.cs:275-287`, commented out), so disposal happens in the DI host's
teardown, in no defined order relative to the view models; the timer keeps firing into a destroyed
context after `Device::drop`.

**Proposed fix (layered).**

1. **Runtime gate.** In `NativeRuntimeService`: a `volatile bool _shutdown` set at the top of
   `Dispose` before `ShutdownSync`, `_ctx = 0` right after, and an in-flight counter
   (`Interlocked`) around every P/Invoke that takes `_ctx`; `ShutdownSync` waits for the counter to
   drain (bounded, e.g. 2 s) before calling the native shutdown. Query methods return null/false
   when `_shutdown`.
2. **Timer ownership.** Stop the pollers first: subscribe the VM's timer to
   `runtimeService.ShutdownToken` (`:1849`) via `TakeUntil(Observable.Create(token))`, or have the
   VM subscribe to a `ShuttingDown` event. Dispose diagnostic VMs in `desktop.ShutdownRequested`
   together with `DetachViewport` (`App.axaml.cs:273`), before the host disposes the runtime.
3. **Order.** Make the runtime's `Dispose` explicit in `Exit` (after the VMs), so the host teardown
   finds it already shut down.

**Risk.** Low to medium: the drain wait must not deadlock with a poll that blocks on the logic thread
(the FFI calls listed as "block the UI thread" in `docs/windows_debugging.md`). Use a timeout and log.

**Tests.** `aethervk.logic.tests/DiagnosticPanelsTests.cs` already builds the VM with a
`TestSchedulerProvider` (`:92`, `:114`): add a test that after `ShutdownToken` is cancelled, advancing
the scheduler produces no further `EarthObserverStatus()` calls on the mock. A runtime test: calling
`EarthObserverStatus()` after `Dispose()` returns null without a P/Invoke (mock the P/Invoke layer or
assert on `_shutdown`). e2e: every test's `AppSession.Dispose` already exits the app; assert the
process exit code is 0 and no WER dump in `TestResults/e2e/*/`.

**Effort.** M.

---

## 7. SPK unload path mismatch (`...\spk/spk_...bsp` reported "not loaded")

**Follow-up (2026-11-02).** Once the unload worked, it exposed a race: `LoadAlmanac` and
`UnloadAlmanac` both ran on the thread pool. On a recommit of the same file, the load (a duplicate, so
a no-op) could run before the unload, which then dropped the comet's SPK; e2e `RunAtThisDate` timed
out. Fixed: `UnloadAlmanac` runs inline on the logic thread. Test:
`test_unload_then_load_of_the_same_spk_leaves_it_loaded`. See `docs/overnight_report.md`.


**Root cause (verified from code).**

- C#: `SaveFileService.SaveSpk` builds `rel = "spk/" + Path.GetFileName(...)` and
  `dest = Path.Combine(dir, rel)` (`SaveFileService.cs:310-311`): a mixed-separator absolute path on
  Windows. That same string is handed to `LoadAlmanacFileAsync` and remembered as
  `_lastCommittedFilePath` for `UnloadAlmanacFile` (`CometConfigService.cs:184`, `:221-222`).
- Rust load: `load_almanac_file_internal` (`simulation_api/structs.rs:569-576`) dedupes on the raw
  string (`:571`) but loads through `PathBuf::from(path)` → `load_single_spk` registers the alias
  from `path.to_str_unified()` (`simulation/almanac.rs:200-212`), a normalised form.
- Rust unload: `unload_almanac_file_internal` (`structs.rs:603-605`) passes the raw string to
  `almanac.spk_unload(path)` and `file_names.retain(|f| f != path)` (`almanac.rs:222-230`), so the
  alias never matches.

**Proposed fix.** Normalise once, in Rust, at the boundary: a private
`fn spk_alias(path: &str) -> String` that does `PathBuf::from(path).to_str_unified()` and is used by
the dedupe at `structs.rs:571`, the registration in `load_single_spk`, and `unload_almanac_spk`.
Belt and braces in C#: `Path.GetFullPath(Path.Combine(dir, "spk", name))` so the engine receives a
native-separator path, and keep `rel` with `/` only inside the JSON index (as `docs/savefile.md`
specifies).

**Risk.** Low. Watch non-UTF-8 paths (`to_str_unified` returns `Option`): fall back to the raw string.

**Tests.** Rust unit test in `simulation/almanac.rs` tests: load with a `\`/`/` mixed path, unload with
the same raw string, assert `num_loaded_spk()` drops and `file_names` is empty; a second load of the
same file with the other separator is deduped. C# `aethervk.logic.tests`: `SaveSpk` produces a path
with native separators (compare against `Path.Combine(dir, "spk", name)`).

**Effort.** S.

---

## 8. A jet added while paused gets no dust history until Play or a seek

**Root cause (verified from code).** `has_physics_work` requires `is_running`
(`simulation_api/logic_thread.rs:363-369`), so the fixed-update phase, and with it
`fill_dust_history` (`:3057-3070`), never runs while paused. The paused branch (`:402-408`) only
calls `update_dust_trajectories`. The seek path does fill while paused (`:4477-4487`), which is why a
seek "fixes" it. A newly added `ParticleSystemComponent` reports `building()`
(`scene/dust.rs:3162-3168`: `next_window < due_window`) until its windows are emitted.

**Proposed fix.** In the paused branch (`:402-408`), when the device is available and any
`ParticleSystemComponent` is `building()`, run one `fill_dust_history` at the current scaled time
(same arguments as `:3058-3068`, `capture = false`) and store the returned sync as the seek path does
(`:4492-4494`). `fill_dust_history_passes` already stops when nothing is building (`:3525-3529`) and
is bounded by `MAX_SEEK_PASSES`, so the paused tick stays idle once the history is complete. Gate it
with `physics_done` like the running branch (`:388`) so it never overlaps the previous GPU work.

Sketch:

```rust
if phase1_res.is_none() && !is_running {
  // ... existing trajectory refresh ...
  if physics_done && any_building(&scene_arc.read()) {
    context.kernels.0.with_device(context.kernels.1, |d| { /* fill_dust_history(...) once */ Ok(()) });
  }
}
```

`any_building` is the closure at `:3525-3529`, lifted to a `utils` fn.

**Risk.** Medium-low: a paused tick now submits compute work; make sure the render thread's dust lock
(`:3583-3584` spin) and the swapchain sync tolerate it (the seek path already does this while paused,
so the pattern exists). Avoid running it when `end_epoch_reached`.

**Tests.** `logic_thread_tests` (Vulkan group): pause, add a jet, tick N times, assert
`ps.dust.lock().building()` becomes false and `stats()` reports emitted windows, without Play or seek.
e2e: in `ThesisScenarioTests` the jets are added paused; add an assertion that the dust history stat
in the diagnostics panel (`DustHistory` text of `CameraMatrixDebugViewModel`) is non-empty before
Play.

**Effort.** S/M.

---

## Suggested order

1. **7 (SPK path)** — S, verified, isolated; unblocks correct decommit in saves.
2. **6 (shutdown crash)** — M, verified; a crash at exit is visible to every user and breaks e2e
   exit checks.
3. **8 (paused jet history)** — S/M, verified; user-visible modelling bug with a contained fix.
4. **4 (slow test split)** — S; do it before the next slow-tier run so the result is interpretable.
5. **2 (billboard drag in e2e)** — S for the test, then decide the app behaviour.
6. **1 (viewport rect on the second monitor)** — S; needs the two-monitor machine to confirm.
7. **3 (ortho half-width after reopen)** — M; instrument first, then fix the restore ordering.
8. **5 (playback perf)** — investigation, after the above, as planned.
