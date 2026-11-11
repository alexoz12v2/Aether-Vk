using System.Diagnostics;
using AetherVk.UI.E2E.Shared;
using Xunit.Abstractions;
using static AetherVk.UI.E2E.Shared.E2ESteps;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// The Mac harness's self-tests, run before any shared scenario on a new machine (each one
/// alone: <c>dotnet test aethervk.ui.e2e.mac -p:BuildE2E=true --filter "FullyQualifiedName~MacHarnessTests.TreeDump"</c>).
/// They prove the seam the scenarios stand on: launch/attach/quit, AutomationIds as accessibility
/// identifiers, the menu bar, both file panels, the CoreGraphics pixel probe and keyboard input
/// reaching the native (CAMetalLayer) viewport. Artifacts go to TestResults/e2e-mac/harness-*/.
/// </summary>
public sealed class MacHarnessTests(ITestOutputHelper output)
{
  private static readonly MacE2EHost Host = new();

  private void Log(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  private static void SkipUnlessAvailable()
  {
    var unavailable = Host.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
  }

  private static MacE2EApp Launch(string artifacts, string name) =>
    (MacE2EApp)Host.Launch(Path.Combine(artifacts, name + ".log"));

  private void With(string artifacts, string name, Action<MacE2EApp> body)
  {
    var app = Launch(artifacts, name);
    try
    {
      body(app);
    }
    catch
    {
      try { app.SaveScreenshot(Path.Combine(artifacts, $"failure-{name}.png")); } catch (Exception) { }
      try { File.WriteAllText(Path.Combine(artifacts, $"failure-{name}.xml"), app.Session.Driver.PageSource); } catch (Exception) { }
      throw;
    }
    finally
    {
      app.Dispose();
    }
  }

  [SkippableFact]
  public void Launch_Attach_Quit_3x()
  {
    SkipUnlessAvailable();
    var artifacts = Artifacts(Host, "harness-launch");
    for (int run = 1; run <= 3; run++)
    {
      var app = Launch(artifacts, $"app-{run}");
      int pid = app.ProcessId;
      try
      {
        Log($"run {run}: pid {pid}, title \"{app.WindowTitle}\", driver launch {app.Session.LaunchedByDriver}");
        Assert.StartsWith("AetherVk", app.WindowTitle);
        E2EWait.Until(() => app.Lines.Count > 0, TimeSpan.FromSeconds(30), "the app's first log line");
        var exit = app.Quit(TimeSpan.FromSeconds(30));
        Log($"  {exit}, {app.Lines.Count} log lines");
        Assert.True(exit.Exited && !exit.Killed, $"run {run}: {exit}");
      }
      finally
      {
        app.Dispose();
      }
      var orphans = Process.GetProcessesByName(MacAppSession.ExecutableName).Where(p => p.Id == pid).ToList();
      Assert.True(orphans.Count == 0, $"run {run}: pid {pid} is still running");
    }
  }

  /// <summary>Classic tabs, then the compact panel, each as page source.</summary>
  private static readonly string[] Tabs = ["SettingsTab", "CometTab", "ModelTab", "ImportsTab", "TimelineTab"];

  [SkippableFact]
  public void TreeDump()
  {
    SkipUnlessAvailable();
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(Host, "harness-tree");
    var dir = Path.Combine(Host.ArtifactsRoot, "tree");
    Directory.CreateDirectory(dir);
    With(artifacts, "app", app =>
    {
      var source = app.Session.Driver.PageSource;
      File.WriteAllText(Path.Combine(dir, "startup.xml"), source);
      app.SaveScreenshot(Path.Combine(dir, "startup.png"));
      Assert.True(source.Contains(MainViewportId, StringComparison.Ordinal),
        "STOP: the page source has no \"ViewportHost\": Avalonia does not expose AutomationId as the accessibility identifier here");
      if (IsCompact(app))
        SetLayout(app, compact: false);
      foreach (var tab in Tabs)
      {
        if (app.TryFind("Tab." + tab) is null)
          continue;
        app.Click("Tab." + tab);
        Thread.Sleep(1000);
        File.WriteAllText(Path.Combine(dir, tab + ".xml"), app.Session.Driver.PageSource);
        app.SaveScreenshot(Path.Combine(dir, tab + ".png"));
      }
      SetLayout(app, compact: true);
      // no comet is loaded here, so the mini-view shows its note instead of the native host
      // (NucleusMiniViewModel.IsHostVisible): open the expander only, SetMiniExpanded would wait
      // for the host; the host's id is checked by the NucleusMiniView scenarios, which open a save
      if (app.TryFind(Ids.NucleusMiniViewCaption) is not { Displayed: true })
        app.Toggle(Ids.NucleusMiniViewExpander);
      E2EWait.Until(() => app.TryFind(Ids.NucleusMiniViewCaption) is { Displayed: true },
        TimeSpan.FromSeconds(10), "the mini-view expander's content");
      var compact = app.Session.Driver.PageSource;
      File.WriteAllText(Path.Combine(dir, "compact.xml"), compact);
      app.SaveScreenshot(Path.Combine(dir, "compact.png"));
      var missing = Ids.All.Where(id => !compact.Contains($"\"{id}\"", StringComparison.Ordinal)).ToList();
      Log($"compact ids missing from the page source: {(missing.Count == 0 ? "none" : string.Join(", ", missing))}");
      Assert.True(compact.Contains($"\"{Ids.CompactPanel}\"", StringComparison.Ordinal), "the compact panel's id is not in the page source");
    });
  }

  [SkippableFact]
  public void MenuBar_LayoutRadio_ReflectsState()
  {
    SkipUnlessAvailable();
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(Host, "harness-menu");
    With(artifacts, "app", app =>
    {
      foreach (var compact in new[] { true, false })
      {
        SetLayout(app, compact);
        var c = app.MenuItemState(LayoutMenu, CompactItem);
        var k = app.MenuItemState(LayoutMenu, ClassicItem);
        Log($"layout {(compact ? "compact" : "classic")}: Compact {c.Raw} → {c.Checked}; Classic {k.Raw} → {k.Checked}");
        Assert.True(c.Checked is not null && k.Checked is not null,
          $"Mac2 exposes no checked state for the Layout items (Compact {c.Raw}, Classic {k.Raw})");
        Assert.Equal((bool?)compact, c.Checked);
        Assert.Equal((bool?)!compact, k.Checked);
      }
    });
  }

  [SkippableFact]
  public void OpenDialog_RoundTrip()
  {
    SkipUnlessAvailable();
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(Host, "harness-open");
    var index = CopyFixture(artifacts);
    var fits = CopyFits(artifacts);
    With(artifacts, "app", app =>
    {
      Log("File › Open... on the reviewer's save");
      OpenSave(app, index);
      SetLayout(app, compact: true);
      int before = app.ListCount(Ids.FitsList);
      Log($"import a FITS through the panel ({before} rows)");
      app.Click(Ids.FitsImportAndOverlay);
      Thread.Sleep(1000);
      app.AnswerOpenDialog(fits);
      E2EWait.Until(() => app.ListCount(Ids.FitsList) > before, TimeSpan.FromSeconds(90), "a new FITS row");
    });
  }

  [SkippableFact]
  public void SaveDialog_RoundTrip()
  {
    SkipUnlessAvailable();
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(Host, "harness-save");
    var index = CopyFixture(artifacts);
    var avk = Path.Combine(artifacts, "saved", "harness_save.avk");
    Directory.CreateDirectory(Path.GetDirectoryName(avk)!);
    With(artifacts, "app", app =>
    {
      OpenSave(app, index);
      SaveAs(app, avk);
      Log($"{avk}: {new FileInfo(avk).Length} bytes");
      Assert.True(new FileInfo(avk).Length > 0);
    });
  }

  /// <summary>
  /// The CoreGraphics probe against Mac2's own element screenshot of the paused main viewport:
  /// the same 32² grid within a mean |Δ| of 0.02. All black fails too (no Screen Recording
  /// permission, or Metal content not captured). Also checks the probe's point→pixel scale.
  /// </summary>
  [SkippableFact]
  public void PixelProbe_MatchesElementScreenshot()
  {
    SkipUnlessAvailable();
    var artifacts = Artifacts(Host, "harness-pixels");
    var index = CopyFixture(artifacts);
    With(artifacts, "app", app =>
    {
      OpenSave(app, index);
      Thread.Sleep(5000);
      var rect = MainViewport(app);
      var probe = MacScreen.Capture(rect);
      Log($"viewport {rect} px, scale {MacScreen.Scale}, capture by {MacScreen.LastMethod}");
      PngCodec.WriteRgb(Path.Combine(artifacts, "probe.png"), probe.Rgba, probe.Width, probe.Height);
      var shot = app.ElementScreenshot(MainViewportId);
      File.WriteAllBytes(Path.Combine(artifacts, "element.png"), PngCodec.EncodeRgb(shot.Rgba, shot.Width, shot.Height));
      Log($"element screenshot {shot.Width}×{shot.Height}");
      // the screenshot's size is the element's pixel size when the scale is right (±2 px of rounding)
      Assert.InRange(shot.Width, rect.Width - 2, rect.Width + 2);
      Assert.InRange(shot.Height, rect.Height - 2, rect.Height + 2);

      var a = FrameMetricsCore.Grid(probe.Luminance(), probe.Width, probe.Height, FrameMetricsCore.GridSize);
      var shotLum = ScreenScale.Resample(shot.Luminance(), shot.Width, shot.Height, rect.Width, rect.Height);
      var b = FrameMetricsCore.Grid(shotLum, rect.Width, rect.Height, FrameMetricsCore.GridSize);
      double meanAbs = a.Zip(b, (x, y) => Math.Abs(x - y)).Average();
      Log($"probe mean {FrameMetricsCore.Mean(a):0.0000}, screenshot mean {FrameMetricsCore.Mean(b):0.0000}, mean |Δ| {meanAbs:0.0000}");
      Assert.True(FrameMetricsCore.Mean(a) > 0.005,
        "the probe reads black: grant Screen Recording to the terminal running dotnet test (or Metal content is not captured)");
      Assert.True(FrameMetricsCore.Mean(b) > 0.005, "Mac2's element screenshot is black");
      Assert.True(meanAbs < 0.02, $"the probe and the element screenshot differ (mean |Δ| {meanAbs:0.0000})");
    });
  }

  [SkippableFact]
  public void Keyboard_ReachesNativeViewport()
  {
    SkipUnlessAvailable();
    var artifacts = Artifacts(Host, "harness-keyboard");
    var index = CopyFixture(artifacts);
    With(artifacts, "app", app =>
    {
      OpenSave(app, index);
      var main = MainViewport(app);
      // the reviewer's save shows a desktop billboard over the viewport, and on macOS its window
      // takes the click even in its transparent margin: click the viewport beside it
      var (clickX, clickY) = ViewportClickPoint(app, main);
      Log($"viewport {main} px, billboards [{string.Join("; ", app.BillboardFrames())}], click at {clickX},{clickY}");
      double[] After(char key)
      {
        app.Foreground();
        app.ClickAt(clickX, clickY);
        Thread.Sleep(400);
        app.PressKey(key);
        Thread.Sleep(TimeSpan.FromSeconds(6));
        return Grid(app, MainViewportId);
      }
      var three = After('3');
      var one = After('1');
      double changed = FrameMetricsCore.Changed(three, one);
      Log($"key 3 → key 1: changed {Pct(changed)}");
      Assert.True(changed > 0.05, $"the camera keys did not reach the viewport ({Pct(changed)} changed)");
    });
  }
}
