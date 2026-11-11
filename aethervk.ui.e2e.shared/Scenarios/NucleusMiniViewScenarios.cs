// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using Xunit;
using static AetherVk.UI.E2E.Shared.E2ESteps;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Shared.Scenarios;

/// <summary>
/// The nucleus mini-view at the top of the compact panel ("Nucleus seen from Earth"), on the
/// reviewer's save: it shows the lit nucleus centred; collapsing it leaves the main view
/// rendering; it survives layout toggles in place; the export and the camera keys stay the main
/// view's; five sessions with it open quit cleanly.
/// </summary>
public static class NucleusMiniViewScenarios
{
  /// <summary>The bright centroid lies within this share of the side from the host's centre.</summary>
  public const double CentroidTolerance = 0.10;

  /// <summary>Two camera modes differ in at least this share of the grid.</summary>
  public const double ModeChanged = 0.05;

  /// <summary>A camera mode switch flies the camera for 2.5 s; the dust hold and full render follow.</summary>
  public static readonly TimeSpan ModeSettle = TimeSpan.FromSeconds(6);

  private static void OpenCompactWithMini(IE2EApp app, string index, Action<string> log)
  {
    Step(log, "open the reviewer's save, compact layout, mini-view expanded");
    OpenSave(app, index);
    SetLayout(app, compact: true);
    SetMiniExpanded(app, expanded: true);
  }

  public static void ShowsTheLitNucleusCentred(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-centred");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      OpenCompactWithMini(app, index, log);
      Thread.Sleep(3000);
      var rect = app.ViewportRect(Ids.NucleusMiniViewHost);
      app.SavePng(rect, Path.Combine(artifacts, "mini.png"));
      var lum = app.CaptureLuminance(rect);
      double lit = FrameMetricsCore.LitShare(lum, LitLuminance);
      var (cx, cy, count) = FrameMetricsCore.BrightCentroid(lum, rect.Width, rect.Height, LitLuminance);
      log($"  mini host {rect}: lit {Pct(lit)}, centroid ({cx:0.000}, {cy:0.000}) of {count} px");
      Assert.True(lit > MinLitShare, $"the mini-view is black ({Pct(lit)} lit)");
      Assert.True(FrameMetricsCore.CentreOffset(cx, cy) <= CentroidTolerance,
        $"the lit nucleus is off centre: centroid ({cx:0.000}, {cy:0.000}), tolerance {CentroidTolerance:0.00}");
    });
  }

  public static void CollapseStopsItsFrames_MainKeepsRendering(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-collapse");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      OpenCompactWithMini(app, index, log);
      Play(app);
      var open = ChangedPairs(app, MainViewportId, 3, TimeSpan.FromSeconds(1));
      log($"  main with the mini open: {string.Join(", ", open.Select(Pct))}");
      Assert.All(open, c => Assert.True(c > PlayingChanged, $"the main view barely changed with the mini open ({Pct(c)})"));

      Step(log, "collapse the mini-view");
      SetMiniExpanded(app, expanded: false);
      Assert.False(MiniShown(app), "the mini host is still shown");
      var closed = ChangedPairs(app, MainViewportId, 3, TimeSpan.FromSeconds(1));
      log($"  main with the mini collapsed: {string.Join(", ", closed.Select(Pct))}");
      Assert.All(closed, c => Assert.True(c > PlayingChanged, $"the main view barely changed with the mini collapsed ({Pct(c)})"));
      Pause(app);
    });
  }

  public static void LayoutToggle10x_SameHost(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-toggle");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      OpenCompactWithMini(app, index, log);
      var first = app.ViewportRect(Ids.NucleusMiniViewHost);
      log($"  mini host {first}");
      for (int i = 1; i <= LayoutToggleScenario.Toggles; i++)
      {
        SetLayout(app, compact: false);
        Assert.False(MiniShown(app), $"toggle {i}: the mini host is shown in Classic");
        SetLayout(app, compact: true);
        E2EWait.Until(() => MiniShown(app), TimeSpan.FromSeconds(10), $"toggle {i}: the mini host back in Compact");
        Thread.Sleep(2000);
        var now = app.ViewportRect(Ids.NucleusMiniViewHost);
        double lit = FrameMetricsCore.LitShare(app.CaptureLuminance(now), LitLuminance);
        log($"  toggle {i}: mini host {now}, lit {Pct(lit)}");
        Assert.True(now.Near(first, 2), $"toggle {i}: the mini host moved to {now} (first {first})");
        Assert.True(lit > MinLitShare, $"toggle {i}: the mini-view is black after the toggle ({Pct(lit)} lit)");
        app.EnsureAlive($"toggle {i}");
      }
    });
  }

  public static void ExportUsesMainOnly(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-export");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      OpenCompactWithMini(app, index, log);
      var main = MainViewport(app);
      var mini = app.ViewportRect(Ids.NucleusMiniViewHost);
      log($"  main {main}, mini {mini}");
      var (model, _) = Export(app, Path.Combine(artifacts, "shot.png"));
      var png = AssertModelPng(model, main, log);
      if (Math.Abs(main.Width - mini.Width) > 4 || Math.Abs(main.Height - mini.Height) > 4)
        Assert.False(Math.Abs(png.Width - mini.Width) <= 2 && Math.Abs(png.Height - mini.Height) <= 2,
          "the export is the mini-view's size");
    });
  }

  public static void KeyboardStillDrivesMainCamera(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-keyboard");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      OpenCompactWithMini(app, index, log);
      var main = MainViewport(app);
      double[] ModeGrid(char key)
      {
        // a click in the main view's upper middle focuses its native window, then the row key
        app.Foreground();
        app.ClickAt((int)main.CentreX, main.Y + main.Height / 4);
        Thread.Sleep(400);
        app.PressKey(key);
        Thread.Sleep(ModeSettle);
        return Grid(app, MainViewportId);
      }
      var nucleus = ModeGrid('3');
      app.SavePng(main, Path.Combine(artifacts, "key3.png"));
      var earth = ModeGrid('1');
      app.SavePng(main, Path.Combine(artifacts, "key1.png"));
      double changed = FrameMetricsCore.Changed(nucleus, earth);
      log($"  key 3 → key 1 changed {Pct(changed)}");
      Assert.True(changed > ModeChanged, $"the camera keys did not change the main view ({Pct(changed)})");
      Assert.True(MiniShown(app), "the mini-view closed under the key presses");
    });
  }

  public static void ShutdownClean_5x(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "nucleus-mini-shutdown");
    var index = CopyFixture(artifacts);
    for (int run = 1; run <= 5; run++)
    {
      Step(log, $"run {run}");
      var since = DateTime.UtcNow;
      var app = host.Launch(Path.Combine(artifacts, $"app-{run}.log"));
      try
      {
        OpenCompactWithMini(app, index, log);
        Play(app);
        Thread.Sleep(3000);
        QuitClean(host, app, since, log);
      }
      catch
      {
        try { app.SaveScreenshot(Path.Combine(artifacts, $"failure-{run}.png")); } catch (Exception) { }
        throw;
      }
      finally
      {
        app.Dispose();
      }
    }
  }
}
