# Reviewer feedback plan (review.eml of 2026-10-31, tested v1.0.1-9-sxs)

The mail points to its attachment "vk test.pptx" (5 slides, Italian). Summary of the slides:

| Slide | Content |
|---|---|
| 1 | Two tests on two dates, models "sufficiently faithful". Setting the observation period vs the generation interval is cumbersome: separate the observation date from the generation period (N rotations), so several dates over weeks/months do not mean redoing every model. Switching Billboard / Model (jets) / Comet (spin axis) tabs is painful: everything on one page. Missing: a nucleus orientation view beside the model. |
| 2 | Remove the "Comet" label and its indicator; the small yellow text on the nucleus is unreadable; the light-blue and red lines: orbit? why two?; "the three icons at the top right do nothing" (they are the overlay's Orbit/Nucleus/Model buttons); what is "Jet Preview Size 34.0 m"; remove the stars (white dots on the black sky outside the image: they are the engine's procedural sky, a decorative field that does not match the real sky, see D6); save the model with black and with transparent background (for slides). |
| 3 | Trajectories confusing: keep only the central one (highest point density); grains in the dust view too large; the flow view is more realistic, but which grains generate it, and what are "Flow Pulses" (they worsen the picture). |
| 4 | Explain the size-law parameters: 30 µm + F 10 plots 1-150 µm where 30 µm is 1 %; "which grains are my models generated with?". Wants a diameter range + exponent (e.g. 30-100 µm, 2.1), no F. |
| 5 | Spin-axis / Sun / velocity arrows unclear; the lit extent vs the Sun direction looks wrong (should be like a lit sphere); what are the two white and two grey lines from the centre, and the light-blue trace. |

Conventions: view-models in `aethervk.ui-logic/ViewModels`, sessions in `Services/Sessions`, saves through
`SaveFile/SaveIndexMapping.cs` DTOs with nullable new fields (precedent `LegacyAfrho.cs`, `FormatVersion`
stays 1); strings in `Resources/AppStrings.resx` + `AppStrings.it.resx` (parity test
`AppStringsParityTests`); help through `Controls/PropertyHelpButton`; every control has an
`AutomationProperties.AutomationId` (e2e `aethervk.ui.e2e`, headless `aethervk.app.tests`); numbers via
`NumericTextBoxBehavior` (invariant culture).

## Steps (each shippable on its own)

| # | Step | Size | Engine/FFI |
|---|---|---|---|
| A | P4 grain law as D_min/D_max + q (no F in the UI) | M | none |
| B | P1 one "Observation" control (date + N rotations) that re-runs | M | none |
| C | P7 physical lighting default in Nucleus view | S | none |
| D | P5 declutter: Comet label, jet-angle text, legend, Jet Preview Size, overlay icons, stars | S each | S |
| E | P6 trajectories: central line default, smaller tracers, Flow pulses in Advanced | M | M + shader recompile |
| F | P2 one page: F1 layout preset; F2 "Compare" page | S / M-L | none |
| G | P3 nucleus view side by side (second viewport) | M | S-M (dust layer per camera) |
| H | P5 image export, black / transparent background | M | M |
| I | P8 remaining points + reply to the reviewer | S | none |

A-D first (the reply needs them), then E-G, H, I.

### A. Grain law as a diameter range (P4)
- UI: "smallest / largest grain diameter (µm)" + "size law exponent q"; plot in diameter over exactly
  D_min..D_max; note "Simulated grains: D_min-D_max. The continuous dust assumes q = 3.5 over this range;
  your q changes only the bright specks." Today "30 µm, F 10" simulates diameters 3-300 µm (radii s0/F..s0·F,
  `dust.rs:1934-1946`) and the plot shows radii: the reviewer's "1-150 µm" confusion.
- Engine stays on (D, F): D = sqrt(D_min·D_max), F = sqrt(D_max/D_min); back D_min = D/F, D_max = D·F. Same
  physics. Validate D_min <= D_max (error text, no swap), ratio <= 1e6 (F clamp 1-1000, `particles.rs:98-105`).
- Files: `Models/JetGrain.cs:16-27`, `ViewModels/JetViewModel.cs:145-173`, `Services/Sessions/ModelSession.cs:74-124`,
  `Views/ModelTabView.axaml:646-724, 964-991`, `Utils/DustCurves.cs:216-249`, resx keys (SizeRange, HelpSizeRange,
  HelpJetSizeRange, HelpSizeDistribution, HelpJetSizeQ, new SimulatedGrainsNote, + it), `SaveIndexV1.cs:140-144, 194-198`
  and `SaveIndexMapping.cs:152-259` (nullable min/max, old saves derived from D/F).
- Tests: `JetGrainDiameterRangeTests` (round trip, old save, min = max), `DustParameterViewTests`,
  `ModelTabViewModelTests.ExposedStrings_*`, e2e `ReviewerMailTests` ids (hands-off).

### B. Observation date vs generation period (P1)
- One block at the top of Comet › Simulated range: Observation date (UTC) [from image]; N rotations before it
  (= x d); dust lead; [Run at this date]; Start/End/Restore/Recommit move to an "Advanced" expander.
- "Run at this date": propose end = date, start = end - max(N·P_rot, 1 d), recommit, seek to the date on commit,
  then set up the FITS comparison when there is an image (the mode bar's Model uses the same chain).
- Files: `RangeEntryViewModel.cs:223-273` (keep ids `ModelEndEpochBox`/`ModelRotationsBox`), `TimelineService.cs:199-214`,
  `CometTabViewModel.cs` (`DownloadAndCommitAsync` :702: skip the SPK download when the current SPK covers the range),
  `CometTabView.axaml:297-392`, `ViewModeBarViewModel.cs:157-184`, `TimelineSession`/`TimelineDto` (nullable N, epoch).
- Tests: `CometTabRangeEntryTests` (chain order, no download within coverage, save round trip),
  `ViewModeBarTests.Model_UsesTheObservationChain`, e2e `FitsModelRangeTests` extension (hands-off).

### C. Physical lighting by default (P7)
- `CameraService.cs:2453` and `ViewportSettingsViewModel.cs:272` -> true; persist as a preference
  (`PreferencesService`, default true). Physical = terminator at the Sun direction (answers slide 5's lit extent).
- Tests: `CameraServiceTests.PhysicalLighting_IsOnByDefault...`, preferences round trip.

### D. Declutter (P5)
1. Comet label only in the Orbit view (`CameraService.ShowsCometLabel` :2534).
2. Jet-angle text 16 pt, white, "lat +23°" / "lon 118°" (`gpu/scene_conversion.rs:1593-1594, 2108, 2160+`).
3. Legend panel in Nucleus view (`UI/OverlayWindow.axaml`, `CollapsibleHudPanel`): red body X, green body Y,
   blue spin axis, yellow Sun, magenta velocity, jet lines in jet colours. In the Model and Orbit views the legend
   lists the orbit lines instead: red = comet reference orbit, light blue = Earth orbit (not the comet's
   reference orbit), yellow = path travelled in this run. The default gizmo parts stay as they are.
4. Remove "Jet Preview Size" from the overlay (`ViewportOverlayViewModel.cs:295-316`, `OverlayWindow.axaml:382`).
5. Remove the overlay's three mode buttons (`OverlayWindow.axaml:53-97`): the top mode bar replaces them.
6. Stars: done by a setting. The dots are the procedural sky (`scene_api.rs` sky entity, `skygen.comp`), now on the
   opt-out render layer `RENDER_LAYER_SKY`. Settings > Viewport > "Background stars" (Auto / On / Off): Auto = off in the
   Model view (black backdrop), on elsewhere. "Orbit lines" (`RENDER_LAYER_ORBITS`) hides the red, light-blue and
   yellow lines. A transparent background stays with the image export (H3).

### E. Trajectories (P6)
- Per-jet central line (the syndyne of the reference size along the jet axis at the mean speed, via
  `emit_cluster` + `kepler::propagate`) as the default; new flag `DUST_VIEW_TRAJECTORIES_CENTRAL`
  (`dust.rs:308-325`); C# `DustTrajectoryMode {Off, Central, Sample}`; old `dust_trajectories: true` -> Central.
- Tracer dots 2 -> 1 px (`dust_common.glsl:70`, `scene/dust.rs:411`): `compile_shaders.sh` + dust parity tests.
- Flow pulses into the Dust Advanced expander, renamed "(motion cue, visual only)".
- Tests: Rust `central_line_ends_at_the_mean_grain_now`, `central_line_lies_inside_the_sample_fan`; C# mode/flag
  mapping and save round trip.

### F. One page (P2)
- F1: default layout with Comet and Model stacked on the left, Settings (Billboards) on the right, viewport
  centre, timeline bottom; View › Layout › Comparison / Classic (`LayoutViewModels.cs:347-410`).
- F2: a "Compare" page built from extracted panels (Observation, Image, Nucleus, Jets, View) bound to the same
  view-model instances (a second `ModelTabViewModel` would push jets twice).

### G. Nucleus view side by side (P3)
- Mode bar toggle opens a second viewport right of the main one, its camera on "nucleus from Earth"
  (`CameraService.ViewNucleusFromEarth` :2142), gizmo + physical lighting + legend; relax
  `Viewport3DViewModel.IsSingleInstance` (or a `NucleusViewportViewModel`).
- Engine: a dust render-layer bit honoured per camera, so the second view does not splat the dust again.

### H. Image export (P5)
- H1 background gradient | black (`BackgroundDrawCall` push constants, new FFI).
- H2 File › Export image… (`avkSimulationContext_captureViewport`, PNG).
- H3 transparent: capture re-render without the background, coverage alpha in `composite.frag`.

### I. Reply
- With the build containing A-D: which grains are simulated, the observation chain, the legend, the lighting,
  the icons, the dots (the procedural sky, now a setting; in the Model view it is off by default and the
  backdrop is black), the lines (red = comet reference orbit, light blue = Earth orbit, yellow = travelled path; a
  setting hides them and the legend names them), tracers and pulses. Also #12 Gaussian jet spread, #2 orbital elements table, the busy state
  on seek (`TimelineTabViewModel.cs:376`).

## Decisions (user, 2026-11-01)
1. Default grain range 10-1000 µm (same physics as today's 100 µm / F 10); 30-100 µm only in the 67P review save.
2. Gizmo parts: keep the current default; only add the legend (D3).
3. Old saves with trajectories on -> Central.
4. F1 layout preset + G first; F2 Compare page only if the reviewer still asks.
