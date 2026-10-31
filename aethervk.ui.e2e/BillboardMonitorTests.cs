using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// Billboards stay with the viewport across monitors: the reviewer's 67P save (its comparison
/// billboard saved in desktop px of another machine) opens with its billboard over the viewport
/// and its nucleus on the comet; the main window moved to the second monitor takes the billboard
/// along; a save made there reopens over the viewport on that monitor. The comparison billboard
/// keeps its nucleus on the comet through an orbit, a zoom and a resize, a free billboard keeps
/// its offset from the viewport's corner, and a reopen with the same view puts the billboard on
/// the comet without a wait. Steps needing two monitors are skipped with one. Screenshots go to
/// TestResults/e2e/billboard-monitor/.
/// </summary>
public sealed class BillboardMonitorTests(ITestOutputHelper output)
{
  /// <summary>The fixture's FITS billboard row.</summary>
  private const int FitsRow = 0;

  /// <summary>Past the restored cameras' flight and the re-centring that waits for it.</summary>
  private static readonly TimeSpan RecenterSettle = TimeSpan.FromSeconds(10);

  /// <summary>A reopen with the same view confirms at once: the billboard is on the comet within this.</summary>
  private static readonly TimeSpan SameViewRecenter = TimeSpan.FromSeconds(1);

  [SkippableFact]
  public void SaveReopened_WithTheWindowOnTheSecondMonitor_KeepsTheBillboardOverTheViewport()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var monitors = Win32.Monitors();
    Skip.If(monitors.Count < 2, $"needs two monitors ({monitors.Count} found)");
    var second = monitors[1];
    output.WriteLine($"monitors: {string.Join("; ", monitors)}");

    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "billboard-monitor");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var avk = Path.Combine(artifacts, "review_67p_second_monitor.avk");

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      // the save's billboard is in desktop px of another machine: it is brought over the viewport
      // and, being bound with a nucleus pixel, back on the comet
      Thread.Sleep(2000);
      OpenSave(s, Path.Combine(saveDir, "aethervk_index.json"), "review_67p");
      Thread.Sleep(RecenterSettle);
      AssertBillboardOverTheViewport(s, "after the open on the first monitor");
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "01-first-monitor.png"));

      // the window dragged to the second monitor: the billboard follows the viewport
      Win32.MaximizeOn(s.MainWindow, second);
      Thread.Sleep(3000);
      AssertBillboardOverTheViewport(s, "after the move to the second monitor");
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "02-second-monitor.png"));

      // saved there, reopened there: over the viewport, nucleus on the comet
      s.Menu("File", "Save As...");
      s.AnswerFileDialog(avk);
      Ui.WaitUntil(() => File.Exists(avk) && !s.WindowTitle.EndsWith('*'), TimeSpan.FromSeconds(90), "the .avk save");
      OpenSave(s, avk, "review_67p_second_monitor");
      Thread.Sleep(RecenterSettle);
      AssertBillboardOverTheViewport(s, "after the reopen on the second monitor");
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "03-reopened-second-monitor.png"));
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  /// <summary>
  /// On one monitor: the comparison billboard's nucleus stays on the comet through a mouse orbit
  /// and a wheel zoom; a free billboard keeps its offset from the viewport's corner when a docked
  /// panel opens (a resize); a reopen with the same view puts the comparison billboard on the comet
  /// within a second of the title change (the restored view is confirmed at once, not waited for).
  /// </summary>
  [SkippableFact]
  public void Billboards_FollowTheCometAndTheViewportCorner_ThroughOrbitZoomResizeAndReopen()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);

    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "billboard-anchor");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var png = Path.Combine(artifacts, "free_square.png");
    TestPng.Write(png);
    var avk = Path.Combine(artifacts, "review_67p_same_view.avk");

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      Thread.Sleep(2000);
      OpenSave(s, Path.Combine(saveDir, "aethervk_index.json"), "review_67p");
      Thread.Sleep(RecenterSettle);
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);

      // a mouse orbit on the viewport, then a wheel zoom: the nucleus follows the comet frame by frame
      var vp = s.Find("ViewportHost");
      var cometBefore = FitsComparisonSteps.CometScreenPx(s);
      int cx = vp.Location.X + vp.Size.Width / 2, cy = vp.Location.Y + vp.Size.Height / 2;
      Win32.SetForegroundWindow(s.MainWindow);
      Win32.DragFromTo(cx - 60, cy + 40, cx + 90, cy - 30);
      Thread.Sleep(1500);
      var cometAfterOrbit = FitsComparisonSteps.CometScreenPx(s);
      output.WriteLine($"comet before the orbit {cometBefore}, after {cometAfterOrbit}");
      FitsComparisonSteps.WaitForSettledGeometry(s, FitsRow);
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      Win32.WheelAt(cx, cy, -3);
      Thread.Sleep(1500);
      output.WriteLine($"comet after the zoom {FitsComparisonSteps.CometScreenPx(s)}");
      FitsComparisonSteps.WaitForSettledGeometry(s, FitsRow);
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "01-after-orbit-and-zoom.png"));

      // a free billboard: its offset from the viewport's corner survives a docked panel opening
      FitsComparisonSteps.ImportImage(s, png);
      int freeRow = FitsComparisonSteps.CreateBillboard(s, png);
      Thread.Sleep(1000);
      var (offsetBefore, vpBefore) = CornerOffset(s, freeRow);
      s.SelectTab("SettingsTab");
      s.Expand("ViewportSettingsSection", "DiagnosticPanelsCheckBox");
      s.Click("DiagnosticPanelsCheckBox");
      Thread.Sleep(2000);
      var (offsetAfter, vpAfter) = CornerOffset(s, freeRow);
      output.WriteLine($"viewport {vpBefore} → {vpAfter}; free billboard offset {offsetBefore} → {offsetAfter}");
      Assert.True(vpBefore != vpAfter, $"the diagnostic panels did not resize the viewport ({vpBefore})");
      Assert.True(Math.Abs(offsetAfter.X - offsetBefore.X) <= 2 && Math.Abs(offsetAfter.Y - offsetBefore.Y) <= 2,
        $"the free billboard's corner offset went from {offsetBefore} to {offsetAfter}");
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow); // the anchored one is still on the comet
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "02-after-panel-resize.png"));

      // saved and reopened with the same view: on the comet at once, not after a settle wait
      s.Menu("File", "Save As...");
      s.AnswerFileDialog(avk);
      Ui.WaitUntil(() => File.Exists(avk) && !s.WindowTitle.EndsWith('*'), TimeSpan.FromSeconds(90), "the .avk save");
      OpenSave(s, avk, "review_67p_same_view");
      var opened = DateTime.UtcNow;
      Ui.WaitUntil(() => NucleusOnComet(s, FitsRow), SameViewRecenter, "the billboard on the comet after the reopen");
      output.WriteLine($"on the comet {(DateTime.UtcNow - opened).TotalMilliseconds:F0} ms after the title change");
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "03-after-same-view-reopen.png"));
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  /// <summary>
  /// The window moved to the second monitor, of another scaling: the free billboard keeps its
  /// corner offset in DIPs (its px offset scaled by the ratio of the scalings) and the comparison
  /// billboard stays on the comet. Two monitors needed; the scalings are reported either way.
  /// </summary>
  [SkippableFact]
  public void WindowMovedToAnotherScaling_KeepsTheCornerOffsetInDips_AndTheNucleusOnTheComet()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var monitors = Win32.Monitors();
    Skip.If(monitors.Count < 2, $"needs two monitors ({monitors.Count} found)");
    var (first, second) = (monitors[0], monitors[1]);
    double k = Win32.ScalingOf(second) / Win32.ScalingOf(first);
    output.WriteLine($"monitors: {first} at {Win32.ScalingOf(first):P0}; {second} at {Win32.ScalingOf(second):P0}");
    Skip.If(Math.Abs(k - 1) < 0.01, $"both monitors at {Win32.ScalingOf(first):P0}: no change of scaling to exercise");

    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "billboard-scaling");
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, "review_67p_2021-12");
    ReviewerMailTests.CopyDir(Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", "review_67p_2021-12"), saveDir);
    var png = Path.Combine(artifacts, "free_square.png");
    TestPng.Write(png);

    using var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    try
    {
      Thread.Sleep(2000);
      Win32.MaximizeOn(s.MainWindow, first);
      Thread.Sleep(1000);
      OpenSave(s, Path.Combine(saveDir, "aethervk_index.json"), "review_67p");
      Thread.Sleep(RecenterSettle);
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      FitsComparisonSteps.ImportImage(s, png);
      int freeRow = FitsComparisonSteps.CreateBillboard(s, png);
      Thread.Sleep(1000);
      var (offsetBefore, vpBefore) = CornerOffset(s, freeRow);
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "01-first-monitor.png"));

      Win32.MaximizeOn(s.MainWindow, second);
      Thread.Sleep(3000);
      var (offsetAfter, vpAfter) = CornerOffset(s, freeRow);
      output.WriteLine($"viewport {vpBefore} → {vpAfter}; free billboard offset {offsetBefore} → {offsetAfter} (expected ×{k:F2})");
      Assert.True(Math.Abs(offsetAfter.X - offsetBefore.X * k) <= 2 && Math.Abs(offsetAfter.Y - offsetBefore.Y * k) <= 2,
        $"the free billboard's corner offset went from {offsetBefore} to {offsetAfter}, expected ×{k:F2}");
      FitsComparisonSteps.WaitForSettledGeometry(s, FitsRow);
      FitsComparisonSteps.AssertNucleusOnComet(s, FitsRow);
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "02-second-monitor.png"));
    }
    catch
    {
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png"));
      throw;
    }
  }

  /// <summary>File › Open of <paramref name="path"/>, waited until the title names <paramref name="titlePart"/>.</summary>
  private static void OpenSave(AppSession s, string path, string titlePart)
  {
    s.Menu("File", "Open...");
    Thread.Sleep(1000);
    s.AnswerFileDialog(path);
    Ui.WaitUntil(() => s.WindowTitle.Contains(titlePart, StringComparison.OrdinalIgnoreCase),
      TimeSpan.FromSeconds(90), "the save to open");
  }

  /// <summary>The row's nucleus pixel lies within 2 px of the comet (false while the rows are being rebuilt).</summary>
  private static bool NucleusOnComet(AppSession s, int row)
  {
    try
    {
      s.SelectTab("SettingsTab");
      s.Expand("BillboardsSection", "CreateBillboardButton");
      FitsComparisonSteps.AssertNucleusOnComet(s, row);
      return true;
    }
    catch (Exception e) when (e is Xunit.Sdk.XunitException or FormatException or InvalidOperationException or OpenQA.Selenium.WebDriverException)
    {
      return false;
    }
  }

  /// <summary>The row's centre as an offset from the viewport's top-left corner (physical px), with
  /// the viewport's rectangle as WinAppDriver reports it.</summary>
  private static ((double X, double Y) Offset, (int X, int Y, int W, int H) Viewport) CornerOffset(AppSession s, int row)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    FitsComparisonSteps.WaitForSettledGeometry(s, row);
    var g = FitsComparisonSteps.ReadGeometry(s, row);
    var vp = s.Find("ViewportHost");
    return ((g.CenterX - vp.Location.X, g.CenterY - vp.Location.Y), (vp.Location.X, vp.Location.Y, vp.Size.Width, vp.Size.Height));
  }

  /// <summary>The FITS row's centre lies inside the viewport as WinAppDriver places it on screen.</summary>
  private void AssertBillboardOverTheViewport(AppSession s, string when)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    FitsComparisonSteps.WaitForSettledGeometry(s, FitsRow);
    var g = FitsComparisonSteps.ReadGeometry(s, FitsRow);
    var vp = s.Find("ViewportHost");
    // the billboard's centre is in desktop px: the viewport in the same frame
    var at = FitsComparisonSteps.DesktopLocation(s, vp);
    int vx = at.X, vy = at.Y, ox = vx - vp.Location.X, oy = vy - vp.Location.Y;
    output.WriteLine($"{when}: billboard centre ({g.CenterX:F1}, {g.CenterY:F1}), viewport at ({vx}, {vy}) {vp.Size.Width}×{vp.Size.Height} (WinAppDriver ({vp.Location.X}, {vp.Location.Y}), window shift ({ox}, {oy}))");
    Assert.True(
      g.CenterX >= vx && g.CenterX < vx + vp.Size.Width
      && g.CenterY >= vy && g.CenterY < vy + vp.Size.Height,
      $"{when}: the billboard's centre ({g.CenterX:F1}, {g.CenterY:F1}) is off the viewport at ({vx}, {vy}) {vp.Size.Width}×{vp.Size.Height} (window shift ({ox}, {oy}))");
  }
}
