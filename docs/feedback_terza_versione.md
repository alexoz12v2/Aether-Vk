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
| 18 | "I could see the model only after overlaying the image and changing the FOV by hand. I couldn't find how to see the model from the image's perspective, i.e. from Earth. The numeric keypad does nothing!" | Camera modes / input | partly **covered**: Earth observer + "Match camera to image" (km / FITS scale, north-up) in this round. **Bug to check on Windows:** NumPad1/2/3/5/. are bound (`ViewportInputRegistrar.cs`), and the native converter maps only VK 0x60–0x69/0x6E (`NativeInputConverter.cs:32-34`). With **NumLock off**, Windows sends VK_END/DOWN/NEXT/CLEAR/DELETE (0x23/0x28/0x22/0x0C/0x2E) for the same keys, so nothing fires. Laptops also often have no keypad. Fix: also map the non-extended navigation codes to NumPadN, and make the mode switch reachable by visible buttons (see 20) (S). Also: keys only reach the viewport when it has focus. |
| 19 | "I couldn't compare with the image because the timeline won't let me enter the useful dates: it ignores my input and picks an end date four years away instead of the image date." | Timeline | **to reproduce on Windows** (M): the SPK request window defaults to ±5 years around today (`CometTabViewModel.cs:478-479`), a likely source of the "four years away" end date. The timeline epoch box/seek path is `TimelineService.cs:164`. This round's FITS "Set up comparison" seeks to DATE-OBS automatically. Wanted: end = image date, start = end − N rotations (user-set). |
| 20 | "Switching view mode must be simpler and immediate, e.g. three icons: **orbit viewer** (the orbit in the Solar System, animated over the chosen interval), **nucleus** (orientation, Sun direction, illumination, active areas), **model** (compare with the telescope image, with a coherent time scale: end date = image date, start date backwards by the user's number of rotations)." | Main toolbar / camera modes | deferred (M–L): three mode icons in the toolbar (maps to item 4); "model" mode = Earth observer + image match + timeline preset end = DATE-OBS, start = end − N·P_rot. |
| 21 | "The model presentation is very CPU-hungry: on a 14th-gen i5 the fan runs flat out. Offload at least part of the workload to the GPU." | Performance | **to measure on Windows** (M). Leading suspect: Avalonia renders in **software** on Windows and X11 (`Program.cs:144,158`, `Win32RenderingMode.Software`, commit 69c0ac6 "disabled GPU for avalonia on windows" for hybrid-GPU black screens), so every UI repaint is CPU-composited. Also check for a busy render/logic loop (frame pacing, `Probe` timers) and that the dust path stays on the GPU. Measure per-thread CPU (Process Explorer) with the viewport idle and while the timeline plays. |
