using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// One session on the reviewer's 67P save, Low quality, the diagnostic panels on (the Earth
/// observer HUD and the per-overlay viewport rectangle), with as many 3D viewports side by side
/// as the layout lets the tests open (<see cref="ViewportProbe.TryAddViewportBeside"/>), each on
/// its own observer: viewport i on <see cref="Observer.All"/>[i % 3].
/// </summary>
public sealed class MultiViewportFixture : ReviewSaveFixture
{
  /// <summary>Viewports the tests try to open: Earth, Sun and the comet.</summary>
  public const int WantedViewports = 3;

  protected override string ArtifactsName => "multi-viewport";

  protected override bool LowQuality => true;

  private IReadOnlyList<Viewport>? _viewports;
  private string? _unavailable;

  /// <summary>The observer assigned to viewport <paramref name="index"/>.</summary>
  public Observer ObserverOf(int index) => Observer.All[index % Observer.All.Length];

  /// <summary>
  /// The session with the viewports opened and each on its observer; a <c>SkipException</c>
  /// (the test skips) when the layout holds a single 3D viewport: the multi-viewport checks need
  /// at least two. Done once; a later call returns the same viewports (re-discovered: the host
  /// elements go stale across a tab switch).
  /// </summary>
  public (AppSession Session, IReadOnlyList<Viewport> Viewports) Arrange(ITestOutputHelper output)
  {
    var s = Session(output);
    Skip.If(_unavailable is not null, _unavailable);
    if (_viewports is not null)
      return (s, _viewports = ViewportProbe.Discover(s));

    ShowDiagnosticPanels(s);
    var viewports = ViewportProbe.Discover(s);
    output.WriteLine($"{DateTime.Now:HH:mm:ss} {viewports.Count} viewport(s) in the opened layout");
    while (viewports.Count < WantedViewports)
    {
      if (!ViewportProbe.TryAddViewportBeside(s, viewports[^1], output.WriteLine))
        break;
      viewports = ViewportProbe.Discover(s);
      output.WriteLine($"{DateTime.Now:HH:mm:ss} now {viewports.Count} viewport(s): {string.Join("; ", viewports)}");
    }
    Screenshot("viewports-opened.png");
    if (viewports.Count < 2)
      _unavailable = $"the layout holds {viewports.Count} 3D viewport (a second one is refused: Viewport3DViewModel.IsSingleInstance); " +
                     "the multi-viewport checks need at least two";
    Skip.If(_unavailable is not null, _unavailable);
    for (int i = 0; i < viewports.Count; i++)
    {
      output.WriteLine($"{DateTime.Now:HH:mm:ss} {viewports[i]} -> {ObserverOf(i)}");
      ViewportProbe.SetObserver(s, viewports[i], ObserverOf(i));
    }
    Screenshot("viewports-observers.png");
    return (s, _viewports = viewports);
  }

  /// <summary>Settings › viewport 1 › "Show diagnostic panels (all viewports)" on (the preference is put back by the quality backup).</summary>
  public static void ShowDiagnosticPanels(AppSession s)
  {
    var section = ViewportProbe.SettingsSection(s, 0);
    var box = ViewportProbe.SectionControl(s, section, "DiagnosticPanelsCheckBox");
    if (!box.Selected)
    {
      box.RealClick();
      Ui.WaitUntil(() => ViewportProbe.SectionControl(s, ViewportProbe.SettingsSection(s, 0), "DiagnosticPanelsCheckBox").Selected,
        TimeSpan.FromSeconds(10), "the diagnostic panels on");
    }
    Thread.Sleep(1500);
  }
}

/// <summary>
/// Several 3D viewports open at once, each observing from a different point (Earth, the Sun's
/// orbit-plane view, the comet nucleus): the overlay indicators and the rendering are consistent
/// per viewport and across them. One shared clock; per-viewport observer, frame label, scale
/// bar and ephemeris; a change in one viewport leaves the others alone; a resize and a
/// Save As › New › Open keep every viewport's observer. The tests run in name order on the
/// fixture's session; the single-viewport sweep (V0) runs on any build, the others skip while
/// the layout refuses a second viewport. Artifacts go to TestResults/e2e/multi-viewport/.
/// </summary>
[TestCaseOrderer(NaturalNameOrderer.TypeName, NaturalNameOrderer.AssemblyName)]
public sealed class MultiViewportConsistencyTests(MultiViewportFixture fixture, ITestOutputHelper output)
  : IClassFixture<MultiViewportFixture>
{
  /// <summary>A frame counts as changed past this share of moved cells (PlaybackTests' seek threshold).</summary>
  private const double ChangedFrame = 0.02;

  /// <summary>A paused, untouched viewport moves at most this share of cells (its dust refine and HUD text).</summary>
  private const double StillFrame = 0.10;

  /// <summary>A rendered viewport's mean luminance (0 on a black one; the sky gradient alone is above this).</summary>
  private const double NotBlack = 0.005;

  /// <summary>
  /// One viewport, the three observers in turn: the frame label names the observer, the clock
  /// label is the timeline's epoch, the scale bar and the ephemeris table are shown, the comet is
  /// at the centre of the nucleus view, the picture is not black and changes with the observer.
  /// Runs on a single-viewport layout too: the per-viewport checks the multi-viewport tests use.
  /// </summary>
  [SkippableFact]
  public void V0_SingleViewport_IndicatorsFollowTheObserver()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var s = fixture.Session(output);
    MultiViewportFixture.ShowDiagnosticPanels(s);
    var vp = ViewportProbe.Discover(s)[0];
    s.SelectTab("TimelineTab");
    var clock = ReviewSave.Epoch(s);
    double[]? previous = null;
    var rulers = new Dictionary<Observer, string>();
    // Comet first: the save opens on the Earth observer, so Earth last proves a change back
    foreach (var observer in new[] { Observer.Comet, Observer.Sun, Observer.Earth })
    {
      Step($"{vp} -> {observer}");
      ViewportProbe.SetObserver(s, vp, observer);
      var r = ViewportProbe.Read(s, vp);
      output.WriteLine($"  {r}");
      s.Desktop.SaveScreenshot(Path.Combine(fixture.Artifacts, $"v0-{observer.CameraMode}.png"));

      Assert.True(observer.Labels(r.FrameLabel), $"the frame label \"{r.FrameLabel}\" does not name the {observer} observer");
      Assert.True(ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(r.EpochLabel), clock),
        $"the clock label \"{r.EpochLabel}\" is not the timeline's epoch {clock:u}");
      Assert.True(r.HasEphemeris, $"the ephemeris table is empty on the {observer} observer: {r}");
      Assert.True(ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(r.EphemerisEpoch), clock),
        $"the ephemeris epoch \"{r.EphemerisEpoch}\" is not the timeline's epoch {clock:u}");
      Assert.False(string.IsNullOrWhiteSpace(r.Ruler), $"no scale bar on the {observer} observer");
      rulers[observer] = r.Ruler!;
      if (observer == Observer.Earth)
        Assert.False(string.IsNullOrWhiteSpace(r.EarthObserver), "the Earth observer HUD is empty in the Earth view (diagnostic panels on)");

      var grid = ViewportProbe.Sample(vp);
      Assert.True(ViewportProbe.Mean(grid) > NotBlack, $"the {observer} view is black (mean luminance {ViewportProbe.Mean(grid):F4})");
      if (previous is not null)
      {
        double changed = FrameMetrics.Changed(previous, grid);
        output.WriteLine($"  changed against the previous observer: {ViewportProbe.Pct(changed)}");
        Assert.True(changed > ChangedFrame, $"the picture barely changed with the observer ({ViewportProbe.Pct(changed)})");
      }
      previous = grid;
      if (observer == Observer.Comet)
      {
        var (ring, sky) = ViewportProbe.CentreRingAndSky(vp);
        output.WriteLine($"  nucleus view: ring {ring:F4}, sky {sky:F4}");
        Assert.True(ring > sky, $"nothing brighter than the sky at the centre of the nucleus view (ring {ring:F4}, sky {sky:F4})");
      }
    }
    // the scale differs between the views: km at the nucleus, AU over the orbit plane
    Assert.NotEqual(rulers[Observer.Comet], rulers[Observer.Sun]);
    // r and Δ are two distances: the Sun's and the Earth's to the comet
    var last = ViewportProbe.Read(s, vp);
    Assert.NotEqual(last.R, last.Delta);
  }

  /// <summary>Every viewport names its own observer, with a scale bar, the ephemeris and a rendered picture.</summary>
  [SkippableFact]
  public void V1_EachViewport_ShowsItsOwnObserver()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);
    var readouts = ViewportProbe.ReadAll(s, viewports);
    output.WriteLine(ViewportProbe.Describe(viewports, readouts));
    for (int i = 0; i < viewports.Count; i++)
    {
      var (vp, r, observer) = (viewports[i], readouts[i], fixture.ObserverOf(i));
      Assert.True(r.Observer == observer, $"{vp} shows \"{r.FrameLabel}\", not the {observer} observer");
      Assert.False(string.IsNullOrWhiteSpace(r.Ruler), $"{vp} has no scale bar");
      Assert.True(r.HasEphemeris, $"{vp} has no ephemeris table: {r}");
      double mean = ViewportProbe.Mean(ViewportProbe.Sample(vp));
      Assert.True(mean > NotBlack, $"{vp} is black (mean luminance {mean:F4})");
      if (observer == Observer.Comet)
      {
        var (ring, sky) = ViewportProbe.CentreRingAndSky(vp);
        Assert.True(ring > sky, $"{vp}: nothing at the centre of the nucleus view (ring {ring:F4}, sky {sky:F4})");
      }
    }
    Assert.True(readouts.Select(r => r.Observer).Distinct().Count() == Math.Min(viewports.Count, Observer.All.Length),
      "two viewports ended on the same observer");
  }

  /// <summary>
  /// One clock: every viewport's clock label and ephemeris epoch are the timeline's epoch, and
  /// identical across viewports, while paused, after a play and a pause, and after a seek.
  /// </summary>
  [SkippableFact]
  public void V2_SharedClock_AcrossViewports_PlayPauseSeek()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);

    Step("paused");
    AssertOneClock(s, viewports);

    Step("play 1 h/s for 5 s, then pause");
    s.SelectTab("TimelineTab");
    var before = ReviewSave.Epoch(s);
    s.ChooseAt("SpeedPicker", ReviewSave.OneHourPerSecondIndex);
    // the save opens at its end: back to the start so the clock can advance
    ReviewSave.ResetToStart(s);
    ReviewSave.Play(s);
    Thread.Sleep(5000);
    var playing = ViewportProbe.ReadAll(s, viewports);
    Assert.True(playing.Select(r => r.EpochLabel).Distinct().Count() <= 2,
      $"the clock labels disagree while playing (read within one poll): {string.Join(" / ", playing.Select(r => r.EpochLabel))}");
    ReviewSave.Pause(s);
    Thread.Sleep(2000);
    var paused = ReviewSave.Epoch(s);
    output.WriteLine($"epoch {before:u} -> reset -> played to {paused:u}");
    AssertOneClock(s, viewports);

    Step("seek 3 days on");
    var target = paused.Date.AddDays(3);
    s.TypeInto(s.Find("CurrentEpochBox"), target.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture), enter: false);
    s.Click("SeekToEpochButton");
    Ui.WaitUntil(() => ReviewSave.Near(ReviewSave.Epoch(s), target), TimeSpan.FromSeconds(30), $"the clock at {target:u}");
    Ui.WaitUntil(() => viewports.All(vp => ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(ViewportProbe.OverlayText(s, vp, "ViewportEpochLabel")), target)),
      TimeSpan.FromSeconds(60), $"every viewport's clock label at {target:u}");
    Thread.Sleep(ViewportProbe.ModeSettle);
    AssertOneClock(s, viewports);
    fixture.Screenshot("v2-after-seek.png");

    Step("the observers survived the playback and the seek");
    var readouts = ViewportProbe.ReadAll(s, viewports);
    for (int i = 0; i < viewports.Count; i++)
      Assert.True(readouts[i].Observer == fixture.ObserverOf(i), $"{viewports[i]} is on \"{readouts[i].FrameLabel}\" after the seek");
  }

  /// <summary>
  /// Per viewport the indicators belong to its observer and differ where the geometry does: the
  /// scale bar of the nucleus view is not the orbit view's; r (Sun–comet) and Δ (Earth–comet)
  /// differ; the Earth viewport carries the Earth observer HUD, the others do not; the geocentric
  /// ephemeris table (the same epoch, the same Earth) is identical in every viewport, so the
  /// Earth viewport's indicators equal the panel's.
  /// </summary>
  [SkippableFact]
  public void V3_Indicators_DifferByGeometry_AndMatchTheEphemeris()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);
    var readouts = ViewportProbe.ReadAll(s, viewports);
    output.WriteLine(ViewportProbe.Describe(viewports, readouts));

    for (int i = 0; i < viewports.Count; i++)
    {
      var (vp, r, observer) = (viewports[i], readouts[i], fixture.ObserverOf(i));
      Assert.True(r.HasEphemeris, $"{vp} has no ephemeris table");
      Assert.NotEqual(r.R, r.Delta);
      bool earthHud = !string.IsNullOrWhiteSpace(r.EarthObserver);
      Assert.True(earthHud == (observer == Observer.Earth),
        $"{vp} ({observer}) {(earthHud ? "shows" : "lacks")} the Earth observer HUD: \"{r.EarthObserver}\"");
      // the two Sun position angles are 180° apart in every viewport
      double pa = Ui.ParseNum(r.PaOfSun!), ps = Ui.ParseNum(r.PsAng!);
      double diff = ((pa - ps) % 360 + 360) % 360;
      Assert.InRange(diff, 179.8, 180.2);
    }

    var keys = readouts.Select(r => r.EphemerisKey).Distinct().ToList();
    Assert.True(keys.Count == 1, $"the geocentric ephemeris differs between viewports:\n  {string.Join("\n  ", keys)}");

    var byObserver = viewports.Select((vp, i) => (Observer: fixture.ObserverOf(i), Readout: readouts[i])).ToList();
    var comet = byObserver.FirstOrDefault(x => x.Observer == Observer.Comet).Readout;
    var sun = byObserver.FirstOrDefault(x => x.Observer == Observer.Sun).Readout;
    if (comet is not null && sun is not null)
      Assert.NotEqual(comet.Ruler, sun.Ruler);
    var earth = byObserver.FirstOrDefault(x => x.Observer == Observer.Earth).Readout;
    if (earth is not null)
    {
      // the HUD names the look direction and, with the panels on, the comet's place on screen
      Assert.Matches("Free|CometTracking|SunTracking", earth.EarthObserver!);
      output.WriteLine($"Earth viewport: Δ {earth.Delta}, phase {earth.Phase}, HUD \"{earth.EarthObserver}\"");
    }
    fixture.Screenshot("v3-indicators.png");
  }

  /// <summary>
  /// A change in one viewport stays in it: its observer rotated to the next, then its projection
  /// switched, the other viewports keep their frame label, scale bar, ephemeris and picture
  /// (within the paused noise floor), while the changed viewport's picture moves.
  /// </summary>
  [SkippableFact]
  public void V4_ChangingOneViewport_LeavesTheOthersAlone()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);
    var a = viewports[0];
    var others = viewports.Skip(1).ToList();

    Step("the noise floor of the untouched viewports, paused");
    var first = others.Select(ViewportProbe.Sample).ToList();
    Thread.Sleep(2000);
    var second = others.Select(ViewportProbe.Sample).ToList();
    var noise = first.Zip(second, FrameMetrics.Changed).ToList();
    output.WriteLine($"noise: {string.Join(", ", noise.Select(ViewportProbe.Pct))}");
    var beforeReadouts = ViewportProbe.ReadAll(s, others);
    var beforeA = ViewportProbe.Sample(a);

    Step($"{a}: observer {fixture.ObserverOf(0)} -> {NextObserver(fixture.ObserverOf(0))}");
    var rotated = NextObserver(fixture.ObserverOf(0));
    ViewportProbe.SetObserver(s, a, rotated);
    try
    {
      var afterReadouts = ViewportProbe.ReadAll(s, others);
      var afterA = ViewportProbe.Sample(a);
      double changedA = FrameMetrics.Changed(beforeA, afterA);
      output.WriteLine($"{a} changed {ViewportProbe.Pct(changedA)}; {string.Join("; ", others.Select((vp, i) => $"{vp} changed {ViewportProbe.Pct(FrameMetrics.Changed(second[i], ViewportProbe.Sample(vp)))}"))}");
      Assert.True(changedA > ChangedFrame, $"{a}'s picture barely changed with its observer ({ViewportProbe.Pct(changedA)})");
      Assert.True(ViewportProbe.Read(s, a).Observer == rotated, $"{a} is not on {rotated}");
      AssertUntouched(others, beforeReadouts, afterReadouts, second, noise, "the observer change in viewport 1");

      Step($"{a}: projection -> orthographic (its settings section)");
      var section = ViewportProbe.SettingsSection(s, a.Index);
      var ortho = ViewportProbe.SectionControl(s, section, "ProjectionOrthographic");
      bool wasOrtho = ortho.Selected;
      ViewportProbe.SectionControl(s, section, wasOrtho ? "ProjectionPerspective" : "ProjectionOrthographic").RealClick();
      Thread.Sleep(ViewportProbe.ModeSettle);
      // the host elements go stale across the tab switch
      viewports = ViewportProbe.Discover(s);
      others = viewports.Skip(1).ToList();
      var afterProjection = ViewportProbe.ReadAll(s, others);
      AssertUntouched(others, beforeReadouts, afterProjection, second, noise, "the projection change in viewport 1");
      fixture.Screenshot("v4-after-changes.png");

      Step("put viewport 1's projection back");
      section = ViewportProbe.SettingsSection(s, a.Index);
      ViewportProbe.SectionControl(s, section, wasOrtho ? "ProjectionOrthographic" : "ProjectionPerspective").RealClick();
      Thread.Sleep(1000);
    }
    finally
    {
      viewports = ViewportProbe.Discover(s);
      ViewportProbe.SetObserver(s, viewports[0], fixture.ObserverOf(0));
    }
  }

  /// <summary>A window resize (restored, smaller, then maximised again) keeps every viewport's observer and clock label, with a scale bar still shown.</summary>
  [SkippableFact]
  public void V5_Resize_KeepsEachViewportsObserver()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);
    var before = ViewportProbe.ReadAll(s, viewports);
    // the monitor the window is on (a move to another would shift WinAppDriver's coordinates)
    var corner = Win32.WindowCorner(s.MainWindow);
    var monitors = Win32.Monitors();
    var monitor = monitors.FirstOrDefault(m => corner.X >= m.Left - 16 && corner.X < m.Left + m.Width
                                               && corner.Y >= m.Top - 16 && corner.Y < m.Top + m.Height) ?? monitors[0];

    Step("restore the window at 70 % of the monitor");
    Win32.PlaceAt(s.MainWindow, monitor.Left + 40, monitor.Top + 40, monitor.Width * 7 / 10, monitor.Height * 7 / 10);
    Thread.Sleep(3000);
    var small = ViewportProbe.Discover(s);
    output.WriteLine($"resized: {string.Join("; ", small)}");
    Assert.Equal(viewports.Count, small.Count);
    AssertSameObservers(s, small, before, "after the resize");
    fixture.Screenshot("v5-resized.png");

    Step("maximise again");
    s.App.Manage().Window.Maximize();
    Thread.Sleep(3000);
    var back = ViewportProbe.Discover(s);
    Assert.Equal(viewports.Count, back.Count);
    AssertSameObservers(s, back, before, "after maximising again");
    Assert.True(back.Zip(small, (b, m) => b.Rect.Width != m.Rect.Width || b.Rect.Height != m.Rect.Height).Any(),
      "the viewports did not change size with the window");
  }

  /// <summary>
  /// Save As Folder, New, Open: the index holds one <c>viewports[i].mode</c> per viewport with
  /// its observer, and the reopened save shows each viewport that comes back on its saved
  /// observer. Closes the shared session (the last test).
  /// </summary>
  [SkippableFact]
  public void V6_SaveAsNewReopen_RestoresEveryViewportsObserver()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (s, viewports) = fixture.Arrange(output);
    var expected = viewports.Select((_, i) => fixture.ObserverOf(i)).ToList();
    var saveDir = Path.Combine(fixture.Artifacts, "multi_viewport_save");
    if (Directory.Exists(saveDir))
      Directory.Delete(saveDir, recursive: true);
    Directory.CreateDirectory(saveDir);

    try
    {
      Step("Save As Folder");
      s.Menu("File", "Save As Folder...");
      s.AnswerFileDialog(saveDir);
      var index = Path.Combine(saveDir, "aethervk_index.json");
      Ui.WaitUntil(() => File.Exists(index) && !s.WindowTitle.EndsWith('*'), TimeSpan.FromSeconds(90), "the folder save");
      var modes = ViewportProbe.SavedViewportModes(index);
      output.WriteLine($"saved viewport modes: {string.Join(", ", modes)}");
      Assert.True(modes.Count >= viewports.Count, $"the index saves {modes.Count} viewport(s) for {viewports.Count} open");
      for (int i = 0; i < viewports.Count; i++)
        Assert.True(modes[i] == expected[i].SaveMode, $"viewports[{i}].mode is \"{modes[i]}\", not \"{expected[i].SaveMode}\" ({expected[i]})");

      Step("File › New");
      s.Menu("File", "New");
      Thread.Sleep(1000);
      // a prompt only when something changed since the save
      try
      {
        Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("Discard").FirstOrDefault(e => e.Displayed), TimeSpan.FromSeconds(3), "the save prompt").Click();
      }
      catch (TimeoutException)
      {
      }
      Ui.WaitUntil(() => !s.WindowTitle.Contains("multi_viewport_save", StringComparison.OrdinalIgnoreCase)
                         && !s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(60), "a new session");
      Thread.Sleep(3000);
      var fresh = ViewportProbe.Discover(s);
      output.WriteLine($"after New: {fresh.Count} viewport(s); frame labels {string.Join(" / ", fresh.Select(vp => $"\"{ViewportProbe.OverlayText(s, vp, "ViewportFrameLabel")}\""))}");

      Step("reopen the folder save");
      Thread.Sleep(1000);
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(index);
      Ui.WaitUntil(() => s.WindowTitle.Contains("multi_viewport_save", StringComparison.OrdinalIgnoreCase),
        TimeSpan.FromSeconds(90), "the save to reopen");
      Thread.Sleep(8000);
      var reopened = ViewportProbe.Discover(s);
      var readouts = ViewportProbe.ReadAll(s, reopened);
      output.WriteLine($"reopened: {reopened.Count} viewport(s)\n{ViewportProbe.Describe(reopened, readouts)}");
      fixture.Screenshot("v6-reopened.png");
      // the save restores the cameras of the viewports the layout opens with (ApplyViewports pairs
      // them by index): every viewport that came back is on its saved observer
      Assert.True(reopened.Count >= 1, "no viewport after the reopen");
      for (int i = 0; i < Math.Min(reopened.Count, expected.Count); i++)
      {
        Assert.True(readouts[i].Observer == expected[i], $"{reopened[i]} reopened on \"{readouts[i].FrameLabel}\", not {expected[i]}");
        Assert.True(readouts[i].HasEphemeris, $"{reopened[i]} has no ephemeris after the reopen");
      }
      Assert.True(reopened.Count == viewports.Count,
        $"the layout reopened with {reopened.Count} viewport(s) of {viewports.Count} saved: the viewport layout is not part of the save");
    }
    finally
    {
      fixture.Close();
    }
  }

  // ── helpers ──

  private static Observer NextObserver(Observer o) => Observer.All[(Array.IndexOf(Observer.All, o) + 1) % Observer.All.Length];

  /// <summary>Every viewport's clock label and ephemeris epoch are the timeline's epoch, and one string across viewports.</summary>
  private void AssertOneClock(AppSession s, IReadOnlyList<Viewport> viewports)
  {
    s.SelectTab("TimelineTab");
    var clock = ReviewSave.Epoch(s);
    var readouts = ViewportProbe.ReadAll(s, viewports);
    output.WriteLine($"clock {clock:u}; labels {string.Join(" / ", readouts.Select(r => $"\"{r.EpochLabel}\""))}; ephemeris {string.Join(" / ", readouts.Select(r => $"\"{r.EphemerisEpoch}\""))}");
    Assert.True(readouts.Select(r => r.EpochLabel ?? string.Empty).Distinct().Count() == 1, "the viewports' clock labels differ");
    Assert.True(readouts.Select(r => r.EphemerisEpoch ?? string.Empty).Distinct().Count() == 1, "the viewports' ephemeris epochs differ");
    var start = ReviewSave.StartEpoch(s);
    foreach (var r in readouts)
    {
      // the label is empty at the committed start by design
      if (!(string.IsNullOrEmpty(r.EpochLabel) && ReviewSave.Near(clock, start)))
        Assert.True(ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(r.EpochLabel), clock),
          $"the clock label \"{r.EpochLabel}\" is not the timeline's epoch {clock:u}");
      Assert.True(ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(r.EphemerisEpoch), clock),
        $"the ephemeris epoch \"{r.EphemerisEpoch}\" is not the timeline's epoch {clock:u}");
    }
  }

  /// <summary>The untouched viewports keep their frame label, scale bar and ephemeris, and their picture within the noise floor.</summary>
  private void AssertUntouched(IReadOnlyList<Viewport> others, IReadOnlyList<ViewportReadout> before, IReadOnlyList<ViewportReadout> after,
    IReadOnlyList<double[]> baseline, IReadOnlyList<double> noise, string what)
  {
    for (int i = 0; i < others.Count; i++)
    {
      var vp = others[i];
      Assert.True(before[i].FrameLabel == after[i].FrameLabel, $"{what} changed {vp}'s frame label: \"{before[i].FrameLabel}\" -> \"{after[i].FrameLabel}\"");
      Assert.True(before[i].Ruler == after[i].Ruler, $"{what} changed {vp}'s scale bar: \"{before[i].Ruler}\" -> \"{after[i].Ruler}\"");
      Assert.True(before[i].EphemerisKey == after[i].EphemerisKey, $"{what} changed {vp}'s ephemeris: {before[i]} -> {after[i]}");
      Assert.True(before[i].EarthObserver == after[i].EarthObserver, $"{what} changed {vp}'s Earth observer HUD: \"{before[i].EarthObserver}\" -> \"{after[i].EarthObserver}\"");
      double changed = FrameMetrics.Changed(baseline[i], ViewportProbe.Sample(vp));
      double allowed = Math.Max(StillFrame, 3 * noise[i]);
      output.WriteLine($"{vp} after {what}: changed {ViewportProbe.Pct(changed)} (allowed {ViewportProbe.Pct(allowed)})");
      Assert.True(changed <= allowed, $"{what} moved {vp}'s picture by {ViewportProbe.Pct(changed)} (noise {ViewportProbe.Pct(noise[i])})");
    }
  }

  private void AssertSameObservers(AppSession s, IReadOnlyList<Viewport> viewports, IReadOnlyList<ViewportReadout> before, string when)
  {
    var now = ViewportProbe.ReadAll(s, viewports);
    output.WriteLine($"{when}:\n{ViewportProbe.Describe(viewports, now)}");
    for (int i = 0; i < viewports.Count; i++)
    {
      Assert.True(now[i].Observer == fixture.ObserverOf(i), $"{viewports[i]} is on \"{now[i].FrameLabel}\" {when}, not {fixture.ObserverOf(i)}");
      Assert.True(now[i].EpochLabel == before[i].EpochLabel, $"{viewports[i]}'s clock label changed {when}: \"{before[i].EpochLabel}\" -> \"{now[i].EpochLabel}\"");
      Assert.False(string.IsNullOrWhiteSpace(now[i].Ruler), $"{viewports[i]} lost its scale bar {when}");
      Assert.True(now[i].EphemerisKey == before[i].EphemerisKey, $"{viewports[i]}'s ephemeris changed {when}");
    }
  }

  private void Step(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");
}
