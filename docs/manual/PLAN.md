# AetherVk user manual: inventory and screenshot plan

Goal: a LuaLaTeX chapter documenting every panel, section and control of the desktop app, in
the fixed format of `STYLE.md`, with screenshots that a script captures again for every
release.

Files in this folder:

| File | What it is |
|---|---|
| `PLAN.md` | this plan: inventory, capture fixture spec, states, upkeep, effort |
| `STYLE.md` | the writing standard (panel template, property entry, terms, typography, refs) |
| `aethervk-manual.sty` | the LuaLaTeX macros (`\uilabel`, `\menu`, `\uikey`, `uiproperty`, callouts, ...) |
| `sample-settings-viewport.tex` | Settings › Viewport written to the standard: the model for every other panel |
| `inventory.md` | panel map plus a generated table of every section and AutomationId |
| `tools/axaml_inventory.py` | the generator for `inventory.md` and `inventory.json` |
| `img/` | screenshots and `*.callouts.tex`, written by the fixture (empty for now) |

The `.sty` and the sample were compiled once here with LuaLaTeX (MiKTeX, siunitx v2 as in
the thesis' `commands.tex`, hyperref, booktabs, amssymb) to catch errors. The thesis itself
was not built.

> **Update (current state).** The output is now **AsciiDoc plus tagged PNGs**, not LaTeX
> callouts (`aethervk-manual.sty`, `*.callouts.tex` and the `.tex` sample are superseded).
> Where §2 and §3 say `Crop.cs`, `img/` or `callouts.tex`, read this:
>
> - Code: `aethervk.ui.e2e/Manual/ManualCaptureTests.cs` (the walk), `Shots.cs` (the shot list as
>   data), `ShotTools.cs` (PNG with a 150 dpi `pHYs`, round numbered markers, manifest),
>   `Win32.Manual.cs` (a few extra user32 calls). No stitching: tall sections are cropped at the
>   window edge (`clipped` in the manifest) or split into several shots.
> - Per shot, in `TestResults/e2e/manual/<yyyyMMdd-HHmm>/`: `<key>--<state>--light.png` (clean crop),
>   `<key>--<state>--light--tagged.png` (numbered markers just left of each control),
>   `<key>--<state>--light.callouts.json` (`[{n, id, label, x, y, w, h, type, help_id}]`, relative to
>   the crop; every AutomationId inside the crop, `*Help` buttons attached to their property, at
>   most 15, top to bottom), and `manifest.json` (version, commit, date, culture, monitor, per-shot
>   rect and status). A failing shot is logged, saved as `<key>--<state>--FAILED.png` and skipped.
> - The app is launched with `AETHERVK_UI_CULTURE=en-US`, diagnostic panels off, the `theme`
>   preference patched (the file is backed up and restored), on the primary monitor, maximised
>   once and never moved.
> - Run (the machine must be idle: the fixture owns the mouse and keyboard):
>
>   ```
>   $env:AETHERVK_MANUAL_CAPTURE=1
>   dotnet test aethervk.ui.e2e/aethervk.ui.e2e.csproj -p:BuildE2E=true --filter FullyQualifiedName~ManualCaptureTests
>   # a subset: $env:AETHERVK_MANUAL_ONLY='^settings-viewport'   (regex on "key--state")
>   py -3 -I docs/manual/tools/axaml_inventory.py --repo . --out docs/manual/inventory.md --splice --json docs/manual/tools/inventory.json
>   py -3 -I docs/manual/tools/make_adoc.py --capture TestResults/e2e/manual/<stamp> --sync
>   ```
>
>   `make_adoc.py` writes `docs/manual/adoc/{main,comet,timeline,model,settings,imports,viewport,dialogs}.adoc`
>   and `index.adoc` (help texts from `AppStrings.resx` through `inventory.json`; `// TODO` where
>   a control has none), and `--sync` copies the PNGs into `docs/manual/img/`.
> - Not captured: `dlg-manage-imports`, `dlg-spawn-mesh`, `dlg-spawn-image` (no UI path opens them).
>   `dlg-fatal` is best effort (the crash switch may write a report instead of showing the window).

---

## 1. Inventory method

### 1.1 Static pass: what the views declare (done)

`tools/axaml_inventory.py` parses every `.axaml` under `aethervk.ui-app` as XML. For each
`Expander` it records the section, and for each element with an
`AutomationProperties.AutomationId` it records the control. For each entry it writes:

- control type, AutomationId, and the path of enclosing expanders (`Simulated range › Advanced: ...`);
- the visible label: a literal `Content`/`Text`/`Header`, else `{Binding StrX}` resolved to
  the English string. The `Str*` property maps to a resx key through the source-generated
  `*.LocalizedStrings.g.cs` (`StrHelpViewFov = GetString("Tabs_Settings_HelpViewFov")`), with
  a fallback on the view's key prefix (`Tabs_Settings_`, `Tabs_Comet_`, ...);
- for a `PropertyHelpButton`, its `HelpHeader`/`HelpContent` resx keys. These are the
  help-text source the manual cites.

```
python -I docs/manual/tools/axaml_inventory.py --repo . --out docs/manual/inventory.md --splice \
       --json docs/manual/tools/inventory.json
```

It is read-only and needs no build. The generated `*.LocalizedStrings.g.cs` files exist
because the project emits compiler-generated files. Without them the prefix fallback still
resolves most labels. First run: 361 rows, of which 25 are expanders and 115 are `?` help
buttons. `inventory.md` §"Gaps" lists the expanders and windows that have no id.

### 1.2 Runtime pass: what the user actually sees (at capture time)

The static pass cannot see `ItemsControl` instances (one viewport section per viewport, one
row per billboard, jet or FITS field), labels computed in view models, or controls that
exist only in some states. `TreeDumpTests` already dumps the UIA tree per tab
(`AETHERVK_E2E_DUMP=1` → `TestResults/e2e/tree/<tab>.xml`). The capture fixture (§2) does
the same at every step it photographs, and writes `img/<shot>.tree.xml` next to the PNG.
A second script (`tools/coverage.py`, to write) then compares:

- AutomationIds in `inventory.json` and the tree dumps, against
- AutomationIds cited in the `.tex` sources (`id=` keys, `\uiid{}`, `\co{}`).

Any id in the app that the manual does not cite is listed as "undocumented", unless
`tools/coverage-allow.txt` lists it as intentionally omitted (debug panels, progress bars).
This makes "every panel and button" a checked claim rather than a hope.

---

## 2. Capture: the `ManualCapture` fixture

### 2.1 Where it lives and how it runs

- `aethervk.ui.e2e/Manual/ManualCaptureTests.cs` (the walk), `Manual/Shots.cs` (the
  declarative list of shots), `Manual/Crop.cs` (crop, stitch, PNG dpi, callout writer).
- Skipped unless `AETHERVK_MANUAL_CAPTURE=1`, like `TreeDumpTests`. A normal e2e run never
  rewrites the manual's images.
- Output: `docs/manual/img/`. Each run first writes into `TestResults/e2e/manual/<date>/`,
  then copies to `img/` only the files whose pixels changed (§5).
- It reuses everything that exists: `AppSession.Launch`, `ReviewSave.Prepare/Open`,
  `PreferencesBackup`, `Ui.SelectTab/Expand/ScrollIntoView/Find/Choose`,
  `Win32.MaximizeOn/Monitors/SavePng`.

### 2.2 A fixed capture environment

| Setting | Value | How |
|---|---|---|
| Monitor | the right one of the two (1920×1080) | `Win32.MaximizeOn(s.MainWindow, Win32.Monitors().Last())`, as the app sits there today |
| Window | maximised, 1920×1040 client | checked after maximise. Abort if the size differs (taskbar moved, other monitor) |
| DPI | 100 % (96 dpi) on that monitor | checked: abort if the element sizes of a reference control (`Tab.SettingsTab`) differ from the recorded ones. 125 % or 150 % give sharper print but reflow the panels: choose once, never mix |
| Theme | **light** for the printed thesis, dark as a second set for slides | write `"theme": "light"` into `~/.aethervk/preferences.json` before `Launch` (read once at start by `App.axaml.cs`). `PreferencesBackup` restores the file. Run the fixture once per theme |
| Language | English (decided) | the chapter is in English. The app follows the Windows UI culture, so the capture fixture launches it with `AETHERVK_UI_CULTURE=en-US` (read at startup in `Program.cs`, `NumberCulture.ApplyOverride`): English labels and dates; numbers always print with a dot (`NumberCulture.Apply`). |
| Diagnostic panels | off | `Launch(diagnosticPanels: false)` |
| Pointer | parked | after every click, `Win32` moves the cursor to the far corner of the left monitor, then waits 300 ms so hover styles and tooltips fade |
| Animation | paused, settled | the save opens at its end epoch. Pause playback (`ReviewSave.Pause`), and before viewport shots wait for the dust to settle (`WaitForDustSettled` pattern from `ThesisScenarioTests`) |
| Particle quality | fixed preset (e.g. Medium, 3) | `ReviewSave.SetQuality(s, 3, "Medium")`, so the viewport shots look alike from run to run |
| Scale per figure | 150 dpi written into each PNG | the `pHYs` chunk: LuaLaTeX then places every crop at its natural size (1 UI px = 0.169 mm). All UI figures share one scale, and a 480 px panel prints 8.1 cm wide |

### 2.3 What a shot is

```csharp
// Manual/Shots.cs -- the whole capture as data; the walker below interprets it
public sealed record Shot(
    string Key,                      // file stem without state/theme: "settings-viewport"
    string State,                    // "orbit", "nucleus", "model", "empty", "committed", ...
    Action<AppSession> Setup,        // bring the UI into the state (select tab, expand, mode)
    Crop Crop,                       // what rectangle to keep
    string[] Callouts,               // AutomationIds, in marker order 1..n ("name:Sun" = by Name)
    bool Stitch = false);            // the crop is taller than the visible pane: scroll + stitch

public abstract record Crop
{
    public sealed record Element(string Id, int Pad = 6) : Crop;          // one element's bounds
    public sealed record Union(string[] Ids, int Pad = 6) : Crop;         // several (no expander)
    public sealed record Window(string? Title = null) : Crop;             // whole app or dialog
    public sealed record Pane(string TabId) : Crop;                       // a docking pane
}

public static class Shots
{
    public static IEnumerable<Shot> All()
    {
        yield return new("settings-viewport", "orbit",
            s => { Mode(s, "ModeBarOrbitButton"); OpenSection(s, "ViewportSettingsSection"); Collapse(s, "ProjectionAdvancedExpander"); },
            new Crop.Element("ViewportSettingsSection"),
            ["CameraModeText", "DiagnosticPanelsCheckBox", "ComparisonLockCheckBox", "ProjectionPerspective",
             "ProjectionAdvancedExpander", "StarBackgroundBox", "OrbitLinesCheckBox",
             "ViewNucleusFromEarthButton", "name:Sun"]);
        yield return new("settings-viewport-persp", "orbit",
            s => { Mode(s, "ModeBarOrbitButton"); s.Expand("ViewportSettingsSection", "ProjectionPerspective");
                   s.Click("ProjectionPerspective"); s.Expand("ProjectionAdvancedExpander", "FovSlider"); },
            new Crop.Element("ProjectionAdvancedExpander"),
            ["FovSlider", "FovUnitPicker", "PerspNearHelp", "PerspFarHelp", "DistanceUnitsHelp"]);
        // ... one entry per row of the table in §3
    }
}
```

The `Callouts` list sets the marker numbers. The `.tex` cites the same AutomationIds
(`\co{FovSlider}`, `id=FovSlider`), so reordering markers never means renumbering text by
hand.

### 2.4 The walker

```csharp
[Trait("Category", "Manual")]
public sealed class ManualCaptureTests(ITestOutputHelper output)
{
    [SkippableFact]
    public void CaptureEveryPanel()
    {
        Skip.If(Environment.GetEnvironmentVariable("AETHERVK_MANUAL_CAPTURE") != "1", "set AETHERVK_MANUAL_CAPTURE=1");
        Skip.If(AppSession.UnavailableReason() is { } why, why);
        var theme = Environment.GetEnvironmentVariable("AETHERVK_MANUAL_THEME") ?? "light";
        using var prefs = new PreferencesBackup();
        Preferences.Patch(p => { p["theme"] = theme; p["showDiagnosticPanels"] = false; });

        var (artifacts, saveDir) = ReviewSave.Prepare("manual");
        var manifest = new Manifest(theme, AppVersion(), DateTime.UtcNow);

        // 1. the empty app: no save, no comet
        using (var s = Launch(artifacts, "empty"))
            foreach (var shot in Shots.All().Where(x => x.State == "empty"))
                Take(s, shot, theme, artifacts, manifest);

        // 2. the committed review save, every other state
        using (var s = Launch(artifacts, "committed"))
        {
            ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
            ReviewSave.Pause(s);
            ReviewSave.SetQuality(s, 3, "Medium");
            foreach (var shot in Shots.All().Where(x => x.State != "empty" && !x.State.StartsWith("dialog")))
                Take(s, shot, theme, artifacts, manifest);
        }
        // 3. dialogs that need a session of their own (fatal error: AETHERVK_TEST_CRASH)
        // ...
        manifest.Write(Path.Combine(artifacts, "manifest.json"));
        Sync(artifacts, ImgDir());   // copy changed files into docs/manual/img (section 5)
    }

    private static AppSession Launch(string artifacts, string name)
    {
        var s = AppSession.Launch(Path.Combine(artifacts, $"app-{name}.log"), diagnosticPanels: false);
        Win32.MaximizeOn(s.MainWindow, Win32.Monitors().Last());   // the right monitor
        AssertWindowSize(s, 1920, 1040);
        return s;
    }

    private void Take(AppSession s, Shot shot, string theme, string dir, Manifest m)
    {
        s.EnsureAlive($"before {shot.Key}");
        shot.Setup(s);
        ParkPointer();
        Thread.Sleep(600);                                   // layout and hover fade
        var rect = Bounds(s, shot.Crop);                      // physical desktop px
        var file = $"{shot.Key}--{shot.State}--{theme}";
        var png = Path.Combine(dir, file + ".png");
        if (shot.Stitch && rect.Height > VisibleHeight(s))
            Crop.Stitched(s, shot.Crop, png);                 // section 2.6
        else
            Win32.SavePng(png, rect.X, rect.Y, rect.Width, rect.Height, dpi: 150);  // + pHYs
        File.WriteAllText(Path.Combine(dir, file + ".tree.xml"), s.App.PageSource);
        File.WriteAllText(Path.Combine(dir, file + ".callouts.tex"), Callouts(s, shot, rect));
        m.Add(file, shot, rect);
        output.WriteLine($"{file}: {rect}");
    }
}
```

### 2.5 Cropping to an element

- `AppiumWebElement.Location`/`Size` are physical desktop pixels, the same space as
  `Win32.WindowCorner` and `Win32.SavePng`, which BitBlts the screen and includes the overlay
  windows. Crop with `SavePng(path, x - pad, y - pad, w + 2*pad, h + 2*pad)`. Do not crop a
  WinAppDriver screenshot: that takes seconds per call and is relative to the window.
- An expander's bounds include its header. That is wanted: the header shows the section
  title and chevron.
- `Crop.Union` takes the min/max over several elements. Use it for the Comet tab's commit
  block (`ReferenceOrbitPicker`, `DownloadCommitButton`, `DownloadStatusText`), which has
  no container id.
- Popups (combo drop-downs, help flyouts, the overlay window) are top-level windows. Find
  them through `s.Desktop` (as `Ui.Choose` does). Their coordinates are in the same space.
- Clamp the rectangle to the monitor, and warn if clamping cut more than the padding.

### 2.6 Panels taller than the pane

The Settings and Model tabs scroll. Expanded *Billboards*, *Dust* (with a jet selected) and
*Viewport* (with *Advanced: projection* open) are taller than the 1040 px window. In order
of preference:

1. **Split the figure.** Document the section in two shots, each one expander or sub-block
   (e.g. `settings-viewport` + `settings-viewport-persp`). This is what the sample does, and
   it reads better in print anyway.
2. **Stitch.** Scroll the section's top under the pane header (`Ui.ScrollIntoView` logic),
   capture the visible part, wheel down by about 80 % of the visible height, and capture
   again. Find the overlap by matching the last 120 px rows of strip *n* in strip *n+1*
   (sum of absolute differences on luminance; `Win32.CaptureLuminance` exists). Repeat until
   the element's bottom is visible. Stop on a 0 px scroll (end of the pane). Shot entries opt
   in with `Stitch: true`.
3. **Taller window.** Not possible: a maximised window is clamped to the monitor, and
   BitBlt cannot see off-screen pixels.

### 2.7 Callouts: drawn in LaTeX, positioned by the fixture

The PNGs stay clean. The markers are drawn by `aethervk-manual.sty`, so they are vector,
re-numberable, and look the same in light and dark sets. For each id in `Shot.Callouts` the
fixture writes one line into `<file>.callouts.tex`:

```csharp
static string Callouts(AppSession s, Shot shot, Rectangle crop)
{
    var sb = new StringBuilder($"% generated by ManualCapture {DateTime.UtcNow:u}; do not edit\n");
    for (int i = 0; i < shot.Callouts.Length; i++)
    {
        var e = FindForCallout(s, shot.Callouts[i]);      // "name:Sun" -> by Name inside the crop
        // the marker sits 10 px left of the control's left edge, centred vertically, inside the crop
        double x = Math.Max(8, e.Location.X - crop.X - 10) / (double)crop.Width;
        double y = (e.Location.Y - crop.Y + e.Size.Height / 2.0) / crop.Height;
        sb.Append(FormattableString.Invariant(
            $"\\uidefinecallout{{{shot.Key}}}{{{Synthetic(shot.Callouts[i])}}}{{{i + 1}}}{{{x:0.0000}}}{{{y:0.0000}}}%\n"));
    }
    return sb.ToString();
}
```

The `.tex` loads it with `\InputIfFileExists{img/<file>.callouts.tex}{}{<hand-placed
fallback>}` (see the top of the sample). One figure key per screenshot (`settings-viewport`,
`settings-viewport-persp`, ...). The property entries use `\uiusefigure{key}` to pick which
markers their headers show.

### 2.8 File naming

```
img/<panel-key>[-<part>]--<state>--<theme>.png         settings-viewport-persp--orbit--light.png
img/<same>.callouts.tex                                 \uidefinecallout lines
img/<same>.tree.xml                                     UIA tree at the moment of capture
img/manifest.json                                       app version, git commit, date, theme,
                                                        culture, monitor, dpi, per-file rects
```

- `panel-key` is the panel's LaTeX key (`STYLE.md` §6): lowercase, hyphens, tab first
  (`comet-simulated-range`, `model-dust-jet`, `settings-billboards-fits`).
- `state` is one word from §3: `empty`, `committed`, `orbit`, `nucleus`, `model`, `locked`,
  `perspective`, `orthographic`, `selected`, `dropdown`, `flyout`.
- `theme` is `light` or `dark`. Use only ASCII and hyphens, never spaces. Double hyphens
  separate the fields. graphicx handles the extra dots of `.callouts.tex`, but the PNG
  names deliberately avoid them.

---

## 3. What to capture

Two sessions per theme: **empty** (fresh launch, no save) and **committed**
(`review_67p_2021-12`: 67P, four jets, one FITS billboard, committed 2021-12-12 → 2021-12-28,
opened at the end epoch, paused). Ordered as the chapter is.

| # | Shot key | State(s) | Setup | Crop |
|---|---|---|---|---|
| 1 | `main` | empty, committed | none | window |
| 2 | `main-modebar` | orbit, nucleus, model, locked | mode buttons. `locked`: Comet › range edit after commit, so `ModeBarRangeLockText` + `ModeBarRecommitButton` show | `ModeBar*` union |
| 3 | `menu-file`, `menu-edit` | committed | `Ui.Menu("File")` and leave it open | desktop popup |
| 4 | `docking` | committed | none | the tab strip of one pane + its +, ⇆, × |
| 5 | `comet-guided` | empty, committed | Tab.CometTab | `GuidedSteps` + `GuidedStepHelpText` |
| 6 | `comet-selection` | empty (search typed, results), committed | type "67P", `CometSearchButton` | *Comet Selection* expander (by header name: no id) |
| 7 | `comet-simulated-range` | committed, locked | `SimulatedRangeSection`. `locked` after editing `ModelRotationsBox` | element |
| 8 | `comet-range-advanced` | committed | `RangeAdvancedExpander` | element |
| 9 | `comet-commit` | empty, committed | none | union `ReferenceOrbitPicker`, `DownloadCommitButton`, `DownloadStatusText` |
| 10 | `comet-nucleus`, `comet-rotation` | committed | `NucleusSection`, `RotationalModelSection` (+ `PolePrecessionExpander`) | element |
| 11 | `timeline` | empty, committed (paused), playing | Tab.TimelineTab | pane |
| 12 | `model-sun`, `model-nucleus-spin` | committed | `ModelSunSection`, `NucleusSpinSection` | element |
| 13 | `model-dust` | empty (no-radius warning), committed | `DustSection` collapsed sub-expanders | element, stitched |
| 14 | `model-dust-jet` | selected | click the first jet row. `JetGrainExpander`, `JetActivityExpander` open | jet detail panel |
| 15 | `model-dust-defaults` | committed | `JetDefaultsHelp` … `DustSizeQBox`, `SizeRangeNote`, `ActivityAdvancedExpander` | element |
| 16 | `settings-time-scale`, `settings-general`, `settings-orbital-elements` | committed | expanders | element |
| 17 | `settings-comet-appearance` | committed, custom | `CometDisplayModeCustom` to show *Custom mesh* | element |
| 18 | `settings-billboards`, `settings-billboards-fits` | committed | `BillboardsSection`, the row expander, `BillboardFitsSection` | element, stitched |
| 19 | `settings-particle-rendering`, `settings-input-bindings` | committed | `ParticleRenderingSection`, `InputBindingsSection` | element |
| 20 | `settings-viewport` (+ `-persp`, `-ortho`, `-nucleus`, `-observer`) | orbit, nucleus, model | as in the sample | element |
| 21 | `imports` | empty, committed | Tab.ImportsTab | pane |
| 22 | `viewport` | orbit, nucleus, model | mode button, dust settled | `ViewportHost` |
| 23 | `overlay` | orbit, model | legend + ephemeris expanded | overlay window parts |
| 24 | `billboard-window` | committed | billboard visible (the fixture's save hides it: show it) | desktop billboard window |
| 25 | `help-flyout` | flyout | click `FovSliderHelp` | the flyout popup |
| 26 | `combo-dropdown` | dropdown | open `SpeedPicker` | popup |
| 27 | `dlg-settings` | dialog | Edit › Settings... | window "Settings" |
| 28 | `dlg-manage-imports`, `dlg-spawn-mesh`, `dlg-spawn-image` | dialog | from the Imports tab | window |
| 29 | `dlg-fatal` | dialog | own session with `AETHERVK_TEST_CRASH` (e2e-only switch) | window |
| 30 | `splash` | launch | poll the top-level windows for the "Starting" title right after `Process.Start`, capture once | window |

Not captured: native Open/Save dialogs (Windows' own), debug-only panels (appendix at most).

About **95 PNGs per theme** (30 rows, many with 2 to 5 states or parts).

---

## 4. Keeping the images current

- **Per release.** Before tagging (`docs/Release_Guide.md`), run
  `AETHERVK_MANUAL_CAPTURE=1 dotnet test aethervk.ui.e2e --filter Category=Manual` once per
  theme on the capture machine. It takes about 10 to 15 minutes per theme, most of it
  waiting for the save to open and the dust to settle.
- **Only real changes land.** `Sync` compares each new PNG with the one in `img/` (per-pixel
  luminance difference over 2 % of pixels, ignoring the viewport shots' dust noise inside
  `ViewportHost`). It copies changed files only and writes `TestResults/e2e/manual/changes.md`
  listing them, so the review shows which figures moved.
- **Text drift.** Rerun `tools/axaml_inventory.py --splice --json`. `git diff` of
  `inventory.md` shows new, renamed or removed controls and help keys. `tools/coverage.py`
  (§1.2) fails if a new AutomationId has no entry and is not allowlisted.
- **Help texts.** The chapter cites resx keys (`help=` in each entry). A small check can
  compare every quoted `uihelptext` block with the current resx value and flag differences.
- **Version stamp.** `manifest.json` records the app version and commit. The chapter's
  figure list can print it ("Screenshots: AetherVk 1.0.x, captured 2026-..").

---

## 5. Effort and order

| Step | Work | Estimate |
|---|---|---|
| 1 | Review `STYLE.md` and the sample, and fix the open choices (language, theme, DPI) | 1 h |
| 2 | Small app changes: AutomationIds for the expanders and buttons listed in `inventory.md` "Gaps"; optional `AETHERVK_UI_CULTURE` | 1–2 h |
| 3 | `ManualCapture` fixture: walker, crops, callouts writer, pHYs, manifest (no stitching) | 4–6 h |
| 4 | Shot list for all 30 rows, first full run, fix flaky setups | 3–4 h |
| 5 | Stitching for the 3 tall sections (or split them instead) | 2 h |
| 6 | Write the panels, 0.5–1.5 h each: Settings (8 sections), Comet (5), Model (4), Timeline, Imports, main window, overlay, dialogs | 15–25 h |
| 7 | `coverage.py`, `Sync` diff, help-text check | 2 h |
| 8 | Dark set, Italian set if wanted (rerun only) | 0.5 h each |

Order: 1 → 2 → 3 with only 3 or 4 shots (the sample's) → compile the sample with the real
images on the thesis machine → 4 → 6 tab by tab (Comet, Timeline and Model first: they are
the scientific workflow; Settings next; dialogs last) → 5 and 7 → 8.

## 6. Open questions

- Chapter language: English labels (resx) or Italian (`AppStrings.it.resx`)? Labels and
  screenshots must use the same language.
- Should the number format follow the UI culture? Typed examples (`\uival{34,1}`) should
  match the screenshots.
- Defaults marked `% VERIFY` in the sample: the view model's initial values (FOV 30°, near
  0.0001 AU, far 200 AU, half-extent 0.0155 AU) may be replaced by the engine's first
  projection. Read them from a fresh launch's `Advanced: projection` before writing
  "Default".
- `booktabs` is used by the sample table. Check that the thesis preamble loads it.
