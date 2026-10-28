# Third feedback round — `Aether Vk.docx` triage (2026-10-27)

Source: `terza_versione/Aether Vk.docx` (16 comments, Italian, each next to a screenshot).
Status: **done** = in this round, **covered** = addressed by another item of the round,
**deferred** = noted for later, with an effort estimate (S < ½ day, M ≈ 1–2 days, L > 2 days).

| # | Comment (translated) | Where | Status |
|---|---|---|---|
| 1 | "What is this General settings tab for? Nothing happens when entering data." | Settings → General (`SettingsTabView.axaml`) | **done**: the section was an unbound placeholder (TODO bindings); now hidden. |
| 2 | "The orbital elements table does not fill in, even after loading a comet." | Settings → Orbital Elements | **done** (bug check): its TextBoxes had no bindings at all, so it could never fill. Hidden until wired to the committed SBDB/osculating elements (deferred, S). |
| 3 | "Hard to understand which parameters to enter in this tab." | Comet tab | deferred: per-field help text / a guided "load comet" flow (M). |
| 4 | "Replace all these commands with simple icons in the top bar." | Main toolbar | deferred: icon toolbar (M). |
| 5 | "Redo this tab as telescope-image management: image size, arcsec/px, automatic FOV in the viewport; km and/or arcsec instead of AU." | Viewport settings / billboards | **covered**: billboard physical size + "Match camera to image" + arcsec/px and arcmin FOV read-outs, km half-sizes (S6.2, S6.4). |
| 6 | "Open and read FITS images; their header has the size and resolution for the FOV." | Imports / billboards | **covered**: FITS decode, auto stretch, metadata grid, plate scale from the WCS (S7). |
| 7 | "This tab is pointless since dates can only be entered in the timeline; either drop it or enter dates here and make the timeline only show progress." | General / Timeline | partly **done** (General hidden, #1) and the timeline gained a current-epoch box that seeks (S4). Moving range entry out of the timeline is deferred (M). |
| 8 | "Dot instead of comma for decimals." | App-wide | **done**: number formatting uses the invariant number format on every OS locale (`Program.cs`); dates keep the locale. |
| 8b | "RA rate and Dec rate are unclear (precession?), remove them." | Comet → rotational model | **done**: moved into a collapsed "Advanced: pole precession" expander (0 by default). |
| 8c | "Rotation period in hours rather than degrees per day." | Comet → rotational model | **done**: "Rotation Period (h)" next to the rate, both editable, sign kept. |
| 9 | "These two tabs are disabled, better remove them." | Model → Sun, Nucleus and Spin Axis | **done**: hidden (the gizmo toggle moved to an enabled section in S2). |
| 10 | "Reference position error, dust visibility coefficient, flow time lapse are unclear; nucleus radius belongs in the Comet tab." | Dust / particle settings | deferred: tooltips with units and effect, move nucleus radius (S). |
| 11 | "Needed: density, diameter, albedo. Q_pr ≈ 1, Afρ and the size power law / range factor are confusing with a single diameter." | Dust grain parameters | deferred: regroup into density / diameter / albedo / β with an "advanced" size distribution (M). |
| 12 | "Jets fine; add Gaussian spread around the main trajectory and the point density (brightness). β and Q_dust belong with the dust." | Jets | deferred: Gaussian spread + per-jet density (L, touches the dust shaders). |
| 13 | "I reached the nucleus view by chance; make it easier, and keep a small orientation mini-view beside the comet view." | Viewport / camera modes | partly **covered** (quick keys, gizmo toggle); mini-view deferred (L, second render target). |
| 14 | "Orbit debug table is not very useful; remove it or allow closing it." | Overlay (CometOrbiting) | **done**: a ✕ closes it for the session. |
| 15 | "Replace the cartesian position table with an ephemeris table at the shown epoch (comet coordinates, r, Sun direction and phase angle, Δ)." | Overlay | deferred (M; the values exist in the measurement block). |
| 16 | "Allow hiding the grid and showing only the nucleus; show only spin axis, Sun and velocity directions; state whether the view is body-fixed / orbit plane / from Earth, and at which date. Why can't we rotate the nucleus freely?" | Comet-orbiting view | partly **covered** (gizmo on/off S2, jet lines S5); axis selection, frame label and free orbit rotation deferred (M). |

## Additions in the 2026-10-28 copy (`Downloads/Aether Vk.docx`, 7 new paragraphs + 2 images)

Identical to the 2026-10-27 text up to item 16; the comments below are new. The reviewer's
screenshot (image 14) shows the comparison they attempted:
- the **comet-orbiting camera** ("Orbit Debug": dist 5 100 km, azimuth 0°, elevation −54°, "ortho: FAIL"),
  not the Earth observer, so the dust is projected along an arbitrary direction, not Earth's line of sight;
- date **2021-12-19 13:17**, not the image date 2021-12-27 00:14:47;
- only **4.6 days** of dust (tier T0 0.0–4.6 d);
- ortho half-height 61 000 km set by hand;
- the image overlaid as a billboard.

Any of these changes the fan's direction and curvature; together they explain the "jet too straight /
perpendicular to the trajectory / swept back" impression. The Linux check (handoff §13) shows that the
correct Earth view at the right date and scale reproduces the observed curvature.

| # | Comment (translated) | Where | Status |
|---|---|---|---|
| 17 | "Best would be to show the nucleus orientation as seen from Earth (like the box below). Besides the yellow Sun arrow, show the nucleus' illumination, which always appears black." (image 13: a lit, textured sphere with spin axis, Sun and velocity arrows) | Comet-orbiting / nucleus view | deferred: an "as seen from Earth" nucleus view (camera on the Earth→comet line, celestial-north up, reuses `ObserverUp`) + Sun-lit nucleus shading (M). Check why the nucleus renders black (lighting of the nucleus mesh in the comet view) (S). |
| 18 | "I could see the model only after overlaying the image and changing the FOV by hand. I couldn't find how to see the model from the image's perspective, i.e. from Earth. The numeric keypad does nothing!" | Camera modes / input | **done (keypad, 2026-10-28 Windows)**: with NumLock off the keypad's non-extended navigation VKs are mapped back to VK_NUMPAD* (`NativeInputConverter.NormalizeWin32NumPad`, lParam bit 24), and on Linux the X11 keypad keycodes, which passed through raw and hit letters (KP_1 = 87 = 'W'), are now mapped. Tests `NativeInputConverterTests`. Visible mode buttons: see 20. Original note — **Bug to check on Windows:** NumPad1/2/3/5/. are bound (`ViewportInputRegistrar.cs`), and the native converter maps only VK 0x60–0x69/0x6E (`NativeInputConverter.cs:32-34`). With **NumLock off**, Windows sends VK_END/DOWN/NEXT/CLEAR/DELETE (0x23/0x28/0x22/0x0C/0x2E) for the same keys, so nothing fires. Laptops also often have no keypad. Fix: also map the non-extended navigation codes to NumPadN, and make the mode switch reachable by visible buttons (see 20) (S). Also: keys only reach the viewport when it has focus. |
| 19 | "I couldn't compare with the image because the timeline won't let me enter the useful dates: it ignores my input and picks an end date four years away instead of the image date." | Timeline | **done (2026-10-28 Windows)**: root cause found. The Start/End boxes proposed on every keystroke and *snapped both boxes back* to the last proposal on any unparsable or < 28-day text, so a date could not be typed: after changing the start to 2021, every edit of the end reverted to the default 2025-11-11, the "four years away". Now invalid text stays, the reason shows in the timeline's error line, and the last valid range stays proposed; `TimeUtils.TryParseIso8601` is culture-invariant and reads a time without an offset as UTC (it used the OS culture and local zone, one hour off on UTC+1). Tests in `TimelineTabViewModelEpochBoxTests`. The 28-day minimum remains (see 20). Original note — **to reproduce on Windows** (M): the SPK request window defaults to ±5 years around today (`CometTabViewModel.cs:478-479`), a likely source of the "four years away" end date. The timeline epoch box/seek path is `TimelineService.cs:164`. This round's FITS "Set up comparison" seeks to DATE-OBS automatically. Wanted: end = image date, start = end − N rotations (user-set). |
| 20 | "Switching view mode must be simpler and immediate, e.g. three icons: **orbit viewer** (the orbit in the Solar System, animated over the chosen interval), **nucleus** (orientation, Sun direction, illumination, active areas), **model** (compare with the telescope image, with a coherent time scale: end date = image date, start date backwards by the user's number of rotations)." | Main toolbar / camera modes | deferred (M–L): three mode icons in the toolbar (maps to item 4); "model" mode = Earth observer + image match + timeline preset end = DATE-OBS, start = end − N·P_rot. |
| 21 | "The model presentation is very CPU-hungry: on a 14th-gen i5 the fan runs flat out. Offload at least part of the workload to the GPU." | Performance | **done (2026-10-28 Windows)**: measured one native thread at 100 % of a core with the app idle, sampling `RtlQueryPerformanceCounter`; Avalonia was not the cause. `oshal::this_thread::sleep_for` on Windows truncated to whole ms and returned at once below 1 ms, so every 50–500 µs idle sleep (render thread 500 µs, logic thread ≥ 100 µs) was a busy spin (Linux `nanosleep` honoured them). Now a high-resolution waitable timer, `Sleep` rounded up as fallback; test `os::native::tests`. Idle: 100.6 % → 16.4 % of one core (Release, 16-thread desktop). Playing with a comet not yet measured. Original note — **to measure on Windows** (M). Leading suspect: Avalonia renders in **software** on Windows and X11 (`Program.cs:144,158`, `Win32RenderingMode.Software`, commit 69c0ac6 "disabled GPU for avalonia on windows" for hybrid-GPU black screens), so every UI repaint is CPU-composited. Also check for a busy render/logic loop (frame pacing, `Probe` timers) and that the dust path stays on the GPU. Measure per-thread CPU (Process Explorer) with the viewport idle and while the timeline plays. |

## Plan for the deferred items 16, 17, 20 (2026-10-28; implemented, see the status below)

Ordered by value to the reviewer's goal (comparing the model with a telescope image). Effort as above.

**A. "Model" view preset, item 20 (third icon) + the end of item 19 (M).** The reviewer's main path.
- A preset on the Timeline tab: end = the image's epoch (DATE-OBS mid-exposure of a FITS billboard, or a typed
  date for a JPEG), start = end − N rotations, with N user-set and P_rot taken from `CometTabViewModel.RotationPeriodHours`.
- Then commit, seek to the end, and run the FITS "Set up comparison" (or, for a JPEG, Earth observer + Match).
- **Decision needed:** the timeline refuses ranges under 28 days (`TimelineTabViewModel.MinRangeDays`). Ten
  rotations of 67P are 5 d, and the dust also needs its lead time (the Linux renders used −28 d ignition). Either
  lower the minimum, or set start = end − max(N·P_rot, dust lead time).
- Files: `TimelineTabViewModel.cs` (preset command + N box), `TimelineTabView.axaml`,
  `FitsImageViewModel.cs` (reuse `SetUpComparisonAsync`). Tests: the range from (DATE-OBS, N, P_rot), and the clamp
  to the SPK coverage.

**B. Three mode icons in the toolbar, item 20 (+ item 4) (M).**
- **Orbit:** the solar-system overview animated over the range. There is no such mode today: it is
  `CometOrbiting` zoomed out, or a new preset distance.
- **Nucleus:** `CometOrbiting` at nucleus scale, with the gizmo, spin axis, Sun and velocity arrows, and the jet lines.
- **Model:** A above (`EarthPosition`, tracking, matched to the image).
- The icons call the same `CameraService` switches as the 1/2/3 keys, so there is no keyboard dependency (laptops
  have no keypad). Files: `MainWindow.axaml` toolbar, `CameraService` mode switch, `styling.md` tokens.
  Tests: logic test that each icon sets the mode, plus an e2e click.

**C. Nucleus that is not black, item 17a (S).**
- (Corrected diagnosis: `BASE_ALBEDO` 0.04 belongs to the debug-normals shader, not the nucleus.) The display
  shading of `physical_mesh2.frag` is Oren–Nayar with the BRDF's 1/π under a unit sun, so a white surface facing
  the Sun is drawn at 1/π ≈ 32 % grey, and the albedo darkens it further; physical lighting is off by default.
- Add an exposure for the nucleus mesh in the comet view (a slider or auto-exposure to the lit
  hemisphere's mean), applied in the shader after the physical term.
- Check: a render test that the lit limb's mean is above a threshold, and that the terminator follows the Sun arrow.

**D. Nucleus "as seen from Earth", item 17b (M).**
- In `CometOrbiting`, a "View from Earth" button that puts the camera on the Earth→comet line (light-time
  retarded, as `earth_observer.rs` does), at nucleus scale, with `ObserverUp.CelestialNorth`.
- The result: the nucleus orientation, lit side and jets as a telescope would see them.
- Reuses the observer-up and light-time code. Test: the camera direction equals the Earth→comet unit vector.

**E. Comet-view clean-up, item 16 (M).**
- Per-element toggles: grid, spin axis, Sun arrow, velocity arrow (some exist as the gizmo on/off from S2).
- A frame label in the viewport: "body-fixed / orbit plane / from Earth · yyyy-MM-dd HH:mm".
- Free orbit rotation around the nucleus. Today orbiting is constrained; it needs a quaternion orbit instead of
  azimuth/elevation.

### Status (2026-10-28, Windows)

Decisions: A lower the minimum; B yes; C exposure control; D consistent with the rest of the work (the Earth
observer); E yes.

- **A.** `TimelineTabViewModel.MinRangeDays` 28 → 1. Timeline tab "Model range: N rotations ending at <epoch>"
  (`ApplyModelRangeCommand`; start = end − max(N·P_rot, 1 d), P_rot from the comet session's rotation rate).
  A FITS row's "Model range" button fills the end with the mid-exposure (else DATE-OBS) via
  `TimelineService.RequestModelRange`. The range is proposed; commit as usual. The dust lead time is the user's
  choice of N (no automatic lead).
- **B.** The overlay's mode indicator is three buttons: Orbit = UpZenith, Nucleus = CometOrbiting, Model =
  `CameraService.EnterModelView` (Earth observer, comet tracking, celestial north up).
- **C.** Per-camera `CameraExposureComponent` (linear factor on the sunlight, `frame.rs`), set by
  `CameraService.ExposureFor`: UpZenith 1; otherwise 2^EV × π for the display shading (undoes its 1/π), 2^EV
  for the physical one. Settings → "Nucleus exposure" slider (−4 … +6 EV, default 0).
- **D.** Settings → "Nucleus from Earth" (`CameraService.ViewNucleusFromEarth`): the model view with a field 4
  nucleus diameters tall. Not checked yet in the app: the float precision of a ~10 km field at ~1 AU.
- **E.** Model tab gizmo parts (grid, body X/Y, spin axis, Sun, velocity; `SphereGizmoPartsComponent`, saved in
  the model session, default all but the velocity); the velocity arrow (magenta) is the comet's heliocentric
  velocity, refreshed each logic commit. The overlay shows the view's frame under the epoch. The comet orbit's
  elevation is free (wraps over the poles, up = the elevation tangent) instead of the ±80° clamp.
