using AetherVk.UI.E2E.Shared;
using Xunit.Abstractions;
using static AetherVk.UI.E2E.Shared.E2ESteps;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// The Compact panel's "Nucleus seen from Earth" mini-view on the reviewer's save: a picture of it
/// (TestResults/e2e-mac/mini-view/mini.png) to judge the comet's exposure and that its track runs
/// through the nucleus. The test itself asserts only that the mini-view shows a picture: the
/// nucleus at the centre is lit (display exposure, not physical lighting: never black).
/// </summary>
public sealed class MacMiniViewTests(ITestOutputHelper output)
{
  private static readonly MacE2EHost Host = new();

  private void Log(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  [SkippableFact]
  public void MiniView_ShowsTheNucleus()
  {
    var unavailable = Host.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(Host, "mini-view");
    var index = CopyFixture(artifacts);
    var app = (MacE2EApp)Host.Launch(Path.Combine(artifacts, "app.log"));
    try
    {
      OpenSave(app, index);
      SetLayout(app, compact: true);
      SetMiniExpanded(app, true);
      Thread.Sleep(3000); // the mini attaches after the main viewport and renders a few frames
      var shot = app.ElementScreenshot(Ids.NucleusMiniViewHost);
      File.WriteAllBytes(Path.Combine(artifacts, "mini.png"), PngCodec.EncodeRgb(shot.Rgba, shot.Width, shot.Height));
      app.SaveScreenshot(Path.Combine(artifacts, "app.png"));
      var lum = shot.Luminance();
      // the central fifth of the picture: the nucleus
      int w = shot.Width, h = shot.Height;
      double centre = 0;
      int n = 0;
      for (int y = h * 2 / 5; y < h * 3 / 5; y++)
        for (int x = w * 2 / 5; x < w * 3 / 5; x++, n++)
          centre = Math.Max(centre, lum[y * w + x]);
      Log($"mini {w}×{h} px: mean luminance {lum.Average():0.0000}, brightest at the centre {centre:0.000}");
      Assert.True(centre > 0.2, $"the nucleus at the mini-view's centre is dark (max luminance {centre:0.000})");
      app.EnsureAlive("showing the mini-view");
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
