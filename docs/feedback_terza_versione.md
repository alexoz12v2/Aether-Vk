# Third feedback round — `Aether Vk.docx` triage (2026-10-27)

Source: `terza_versione/Aether Vk.docx` (16 comments, Italian, each next to a screenshot).
Status as of **2026-10-30** (Linux, current tree): **done** = implemented, with the test(s) that prove it,
**covered** = addressed by another item, **partial** = part done, the rest named, **in progress** = being built,
**superseded** = replaced by a later reviewer item, **deferred** = noted for later, with an effort estimate
(S < ½ day, M ≈ 1–2 days, L > 2 days). Tests are in `aethervk.logic.tests/` unless a path is given; Rust tests are
`file::fn`; "e2e" tests are in `aethervk.ui.e2e/` and run on Windows only.

| # | Comment (translated) | Where | Status |
|---|---|---|---|
| 1 | "What is this General settings tab for? Nothing happens when entering data." | Settings → General (`SettingsTabView.axaml`) | **done**: the unbound placeholder section is hidden. Test `ReviewFeedbackVerificationTests.Review_docx_1_2_GeneralAndOrbitalElementsSectionsAreHidden` (a text check of the axaml). |
| 2 | "The orbital elements table does not fill in, even after loading a comet." | Settings → Orbital Elements | **done** (bug check): its TextBoxes had no bindings, so it could never fill; hidden until wired to the committed SBDB/osculating elements (deferred, S). Test: the same `Review_docx_1_2_…`. |
| 3 | "Hard to understand which parameters to enter in this tab." | Comet tab | **done**: a numbered step strip (1 Search → 2 SPK record → 3 Range → 4 Commit, then done) driven by the tab's state (`CometTabViewModel.GuidedStepFor`), with per-step help and field tooltips. Tests `CometTabRangeEntryTests.GuidedStep_ForEachState`, `GuidedStep_FollowsTheTab_DoneStepsTicked_CurrentHighlighted`. |
| 4 | "Replace all these commands with simple icons in the top bar." | Main toolbar | **done** with 20: a top bar of three mode icons (`ViewModeBarViewModel`). Tests `ViewModeBarTests` (`Buttons_SetTheCameraMode_TheOverlayReads`, `Highlight_FollowsModeChangesMadeElsewhere`, …); e2e `TopModeBarTests.TopModeBar_SwitchesTheViewMode` (Windows run pending). |
| 5 | "Redo this tab as telescope-image management: image size, arcsec/px, automatic FOV in the viewport; km and/or arcsec instead of AU." | Viewport settings / billboards | **covered**: billboard physical size, "Match camera to image", arcsec/px and FOV read-outs, km half-sizes (S6.2, S6.4), and (R5) the "1 image px = N viewport px" read-out. Tests `DesktopBillboardTests.ConversionText_ShowsImageAngleViewportScaleAndPxRatio`, `PhysicalSize_IsImagePixelsTimesThePlateScaleOverTheViewportScale`, `MatchCameraToImage_KmAndPlateScaleGiveTheSameField_AndTheBillboardFillsTheHeight`; `FitsBillboardTests.PixelRatio_*`; `CameraServiceTests.MatchArcsecPerPx_SetsTheFov_AndCometPositionIsObserverOnly`. |
| 6 | "Open and read FITS images; their header has the size and resolution for the FOV." | Imports / billboards | **covered**: FITS decode, auto stretch, metadata grid, plate scale from the WCS (S7). Tests `FitsBillboardTests.FitsRow_DetectsFits_SummarisesFacts_ListsCards_AndAutoFillsThePlateScale`, `Metadata_ParsesCardsFactsAndTheExactTdbParts`, `SetUpComparison_KeepsTheCometCentred_RotatesFromTheWcs_AndLocksTheViewport`; Rust `scene/sky_image/tests.rs::fits_refuses_what_it_cannot_read`. |
| 7 | "This tab is pointless since dates can only be entered in the timeline; either drop it or enter dates here and make the timeline only show progress." | General / Timeline | **done**: General hidden (#1); the range (Start/End, Model range, dust lead time, lock/Recommit/Restore) is entered in the Comet tab (`RangeEntryViewModel`), the Timeline shows the committed range read-only and the progress. Tests `CometTabRangeEntryTests.EditingTheRangeInTheCometTab_ProposesIt_AndTheCommitUsesIt`, `TheTimeline_ExposesTheRangeReadOnly_AndItsBoxesEqualTheCommittedRange`, `AfterACommit_AnEditShowsTheLockText_AndRecommitAppliesIt`; `TimelineRangeLockTests`. |
| 8 | "Dot instead of comma for decimals." | App-wide | **done**: number formatting uses the invariant number format on every OS locale (`Program.cs`); dates keep the locale. Test `ReviewFeedbackVerificationTests.Review_docx_8_NumbersUseADot_UnderAnItalianLocale` (re-applies the `Program.cs` recipe; the four `Program.cs` lines themselves run only in the app). |
| 8b | "RA rate and Dec rate are unclear (precession?), remove them." | Comet → rotational model | **done**: in a collapsed "Advanced" rates expander, 0 by default. Tests `aethervk.app.tests/DustParameterViewTests.cs::Comet_RotationRateIsAdvanced_PeriodAndPoleHaveHelp`, `ReviewFeedbackVerificationTests.Review_docx_8b_PrecessionRatesDefaultToZero`. |
| 8c | "Rotation period in hours rather than degrees per day." | Comet → rotational model | **done**: "Rotation Period (h)" next to the rate, both editable, sign kept. Test `CometReferenceOrbitTests.RotationPeriodHours_MirrorsTheRate_AndKeepsItsSign`. |
| 9 | "These two tabs are disabled, better remove them." | Model → Sun, Nucleus and Spin Axis | **done**: hidden behind the legacy flag (the gizmo toggles moved to an enabled section). Test `ReviewFeedbackVerificationTests.Review_docx_9_SunAndNucleusSectionsAreHiddenBehindTheLegacyFlag`. |
| 10 | "Reference position error, dust visibility coefficient, flow time lapse are unclear; nucleus radius belongs in the Comet tab." | Dust / particle settings | **done**: tooltips on the three settings (`ReviewFeedbackVerificationTests.Review_docx_10_DustSettingsHaveTooltips`, which checks a `ToolTip.Tip` on each, not its text); the nucleus radius is edited in the Comet tab, written to the model session and pushed to the engine (`CometTabNucleusRadiusTests.RadiusEdit_WritesTheModelSession_AndPushesNativeOnce`, `Vm_ReadsTheModelSessionRadius_WithoutAPush`, `ClearedRadius_FallsBackToTheCatalogueRadius`). |
| 11 | "Needed: density, diameter, albedo. Q_pr ≈ 1, Afρ and the size power law / range factor are confusing with a single diameter." | Dust grain parameters | **done**: per-jet grain (density, diameter, albedo), activity and size law in collapsed advanced expanders with plots and help, Afρ replaced by Q_d (mail R6/R7). Tests `aethervk.app.tests/DustParameterViewTests.cs::Model_ActivityAndSizeLawAreAdvanced_WithPlotsAndHelp`, `::Model_JetPanel_HasItsOwnGrainAndActivity`, `::ModelTabView_MentionsNoAfrho`; `ModelTabViewModelTests.ExposedStrings_MentionNoAfrho`. |
| 12 | "Jets fine; add Gaussian spread around the main trajectory and the point density (brightness). β and Q_dust belong with the dust." | Jets | **deferred** (L): the Gaussian spread and per-jet density change the dust emission and splat shaders, which need the byte, parity and occupancy gates on all three GPU targets. Q_d now sits with each jet's grain (R6). |
| 13 | "I reached the nucleus view by chance; make it easier, and keep a small orientation mini-view beside the comet view." | Viewport / camera modes | partly **covered**: the Nucleus icon of the top bar (4/20, `ViewModeBarTests`). The mini-view is **deferred** (L): it needs a second render target and camera in the engine. |
| 14 | "Orbit debug table is not very useful; remove it or allow closing it." | Overlay (CometOrbiting) | **superseded** by mail R4: the panel is hidden by default; it shows only with Settings › "Show diagnostic panels" on *and* the camera in nucleus mode, and it collapses to its icon (remembered). There is no ✕. Tests `DiagnosticPanelsTests.Hidden_ByDefault_AndTheChoiceSurvivesARestart`, `TheOrbitPanel_NeedsBothTheNucleusModeAndThePreference`; `CometReferenceOrbitTests.OrbitDebugPanel_CollapsedIsStillShown`, `Toggle_FlipsIsExpanded_AndWritesPreference`. |
| 15 | "Replace the cartesian position table with an ephemeris table at the shown epoch (comet coordinates, r, Sun direction and phase angle, Δ)." | Overlay | **in progress** (2026-10-30, D2). |
| 16 | "Allow hiding the grid and showing only the nucleus; show only spin axis, Sun and velocity directions; state whether the view is body-fixed / orbit plane / from Earth, and at which date. Why can't we rotate the nucleus freely?" | Comet-orbiting view | **done**: per-part gizmo toggles (grid, body X/Y, spin axis, Sun, velocity; saved in the model session), a frame label "frame · date" in the overlay, and free orbit rotation (the elevation wraps over the poles instead of the ±80° clamp). Tests `ModelTabViewModelTests.SphereGizmoPart_EachToggle_ShowsItsPartAlone`, `ViewModesTests.FrameLabel_NamesTheFrameOfEachMode`, `FrameLabel_EndsWithTheShownEpoch`, `CameraServiceTests.OrbitRotation_IsContinuousOverThePole_AndWraps`. |

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
| 17 | "Best would be to show the nucleus orientation as seen from Earth (like the box below). Besides the yellow Sun arrow, show the nucleus' illumination, which always appears black." (image 13: a lit, textured sphere with spin axis, Sun and velocity arrows) | Comet-orbiting / nucleus view | **partial**. Done: the Earth-observer camera always looks at the target with celestial north up (Rust `simulation_api/earth_observer.rs::view_and_up_do_not_turn_with_the_earth_over_a_day`, `::nucleus_view_from_earth_matches_the_geocentric_observer`). The black nucleus was inward winding on imported meshes, now repaired in `finalize_comet` (Rust `simulation/comet/test_comet.rs::finalize_comet_repairs_inward_winding`, `gpu/frame.rs::mesh_sun_light_tests::sun_facing_side_is_lit`); the nucleus exposure slider stays. Open: the C# "from Earth" nucleus framing is not wired to the new engine view (Settings › "Nucleus from Earth", `CameraService.ViewNucleusFromEarth`, is tested only for its refusal without a comet, `CameraServiceTests.ModelView_IsTheEarthObserverNorthUp_AndNeedsAComet`). |
| 18 | "I could see the model only after overlaying the image and changing the FOV by hand. I couldn't find how to see the model from the image's perspective, i.e. from Earth. The numeric keypad does nothing!" | Camera modes / input | **done**: with NumLock off, Windows' non-extended navigation VKs map back to VK_NUMPAD* (`NativeInputConverter.NormalizeWin32NumPad`, lParam bit 24); on Linux the X11 keypad keycodes, which passed through raw and hit letters (KP_1 = 87 = 'W'), map to the keypad (`NativeInputConverter.NormalizeX11Keycode`). Tests `NativeInputConverterTests.NumLockOff_KeypadKeys_MapToTheNumPadKeys`, `DedicatedNavigationKeys_AreLeftAlone`, `OtherKeys_PassThrough`, `X11_KeypadKeycodes_MapToTheNumPadKeys`, `X11_NonKeypadKeycodes_AreNotTheKeypad`, `X11_UnmappedKeycode_PassesThroughRaw`. The "from Earth" view is the Model icon (20, `ViewModeBarTests`). Keys reach the viewport only when it has focus (Windows e2e, not written). |
| 19 | "I couldn't compare with the image because the timeline won't let me enter the useful dates: it ignores my input and picks an end date four years away instead of the image date." | Timeline | **done**. Root cause: the range boxes snapped back to the last proposal on any unparsable or short text, so the end reverted to the default 2025-11-11; now invalid text stays with its reason and the last valid range stays proposed, and parsing is culture-invariant and UTC (`TimelineTabViewModelEpochBoxTests.RangeBoxes_TypedCharByChar_KeepTheText_AndProposeTheFinalRange`, `RangeBoxes_InvalidRange_ShowsWhy_AndKeepsTheLastProposal`, `Parse_IsInvariantAndUtc_AndRoundTripsTheDisplay`). The range is entered in the Comet tab (7) with the Model range (end = image epoch, 20). The Comet tab's ±5 years around today only lists the SPK records and never sets the range; the SPK download is the committed range's days plus 30 days on each side (`CometTabViewModel.SpkDownloadDays`, `HorizonJplService.SpkRequestDays`). Tests `CometTabRangeEntryTests.SpkDownloadWindow_HoldsTheRange`, `SpkDownloadWindow_HoldsTheProposedAndTheCommittedRange`, `FitsModelRange_FillsTheCometTabModelEndBox_WithTheImageMidExposure`. |
| 20 | "Switching view mode must be simpler and immediate, e.g. three icons: **orbit viewer** (the orbit in the Solar System, animated over the chosen interval), **nucleus** (orientation, Sun direction, illumination, active areas), **model** (compare with the telescope image, with a coherent time scale: end date = image date, start date backwards by the user's number of rotations)." | Main toolbar / camera modes | **done**: the top bar's Orbit / Nucleus / Model icons (`ViewModeBarViewModel`). Model runs the Model range (end = the FITS DATE-OBS mid-exposure, start = end − N·P_rot, at least 1 day), then "Set up comparison". Tests `ViewModeBarTests.Buttons_SetTheCameraMode_TheOverlayReads`, `Model_WithAFitsImage_RequestsTheModelRange_ThenSetsUpTheComparison`, `Model_WithoutAFitsImage_EntersTheEarthObserver_AndHints`; `CometTabRangeEntryTests.ModelRange_SetsStartToEndMinusNRotationPeriods`; e2e `TopModeBarTests`, `FitsModelRangeTests` (Windows run pending). |
| 21 | "The model presentation is very CPU-hungry: on a 14th-gen i5 the fan runs flat out. Offload at least part of the workload to the GPU." | Performance | **done** (2026-10-28, Windows): sub-millisecond sleeps were busy spins on Windows (`oshal::this_thread::sleep_for` truncated to whole ms); now a high-resolution waitable timer. Rust `aethervk.oshal/rlib/src/os/native.rs::sub_millisecond_sleeps_actually_wait`, `::zero_returns_at_once`. Idle: 100.6 % → 16.4 % of one core (measured by hand, Release). Playing with a comet is not measured yet. |

### Verification 2026-10-30

Every **done** row above names a test in the tree; the new ones today are the X11 keypad tests (18),
the SPK window and the FITS model-end-box tests (19, P4) in `NativeInputConverterTests` and
`CometTabRangeEntryTests`. Checked only on Windows (e2e, or by hand) so far: the top mode bar (4/20,
`TopModeBarTests`) and the FITS Model range through the UI (`FitsModelRangeTests`), both with the run
pending; keys reaching the viewport only with focus (18, no e2e yet); the idle CPU figure and the playing
case (21); the `Program.cs` culture lines on an Italian locale (8, no e2e yet). Rows 1, 2, 9 and 10 are
checked as text in the axaml, not on a loaded view. The engine-side 17 tests are Rust; the lit nucleus as
the reviewer sees it has not been looked at on Windows.

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
