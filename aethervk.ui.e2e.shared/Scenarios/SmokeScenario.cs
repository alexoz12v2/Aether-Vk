// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.IO;
using System.Threading;
using Xunit;
using static AetherVk.UI.E2E.Shared.E2ESteps;

namespace AetherVk.UI.E2E.Shared.Scenarios;

/// <summary>The shortest end-to-end proof: the 67P save, play 3 s, export, a clean quit.</summary>
public static class SmokeScenario
{
  public static void Run(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "smoke");
    var index = CopyFixture(artifacts);
    var since = DateTime.UtcNow;
    var app = host.Launch(Path.Combine(artifacts, "app.log"));
    try
    {
      Step(log, "open the reviewer's save, compact layout");
      OpenSave(app, index);
      SetLayout(app, compact: true);

      Step(log, "play 3 s");
      Play(app);
      var start = Epoch(app);
      Thread.Sleep(3000);
      var end = Epoch(app);
      log($"  clock {start:u} → {end:u}");
      Assert.True(start is { } a && end is { } b && b > a, $"the clock did not advance ({start:u} → {end:u})");
      Pause(app);

      Step(log, "export");
      var main = MainViewport(app);
      var (model, _) = Export(app, Path.Combine(artifacts, "smoke.png"));
      AssertModelPng(model, main, log);

      Step(log, "quit");
      QuitClean(host, app, since, log);
    }
    catch
    {
      try { app.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch (Exception) { }
      throw;
    }
    finally
    {
      app.Dispose();
    }
  }
}
