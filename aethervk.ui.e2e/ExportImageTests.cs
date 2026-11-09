using System.Drawing;
using System.Globalization;
using AetherVk.UI.E2E.Manual;
using OpenQA.Selenium;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// File › Export image... on the committed 67P review save: the two PNGs are written next to the
/// chosen name, they are the 3D viewport's size, the model image is a real render (not black) and
/// it is the engine's image only: inside the ephemeris panel's rectangle it differs from what the
/// screen shows there (the Avalonia overlay is not part of the export). Artifacts go to
/// TestResults/e2e/export-image/.
/// </summary>
public sealed class ExportImageTests(ITestOutputHelper output)
{
  private void Step(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  private static double Luma(byte[] bgra, int k) => (0.0722 * bgra[k] + 0.7152 * bgra[k + 1] + 0.2126 * bgra[k + 2]) / 255.0;

  [SkippableFact]
  public void ExportImage_WritesModelAndCompositePngsOfTheViewport()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (artifacts, saveDir) = ReviewSave.Prepare("export-image");
    var log = Path.Combine(artifacts, "app.log");
    var chosen = Path.Combine(artifacts, "shot.png");
    var modelPath = Path.Combine(artifacts, "shot_model.png");
    var fitsPath = Path.Combine(artifacts, "shot_with_fits.png");

    var s = AppSession.Launch(log, diagnosticPanels: false);
    int? exitCode = null;
    try
    {
      Step("open the review save");
      ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
      var viewports = Ui.WaitFor(() => ViewportProbe.Discover(s) is { Count: > 0 } v ? v : null,
        TimeSpan.FromSeconds(30), "a 3D viewport");
      var vp = viewports[0];
      output.WriteLine(vp.ToString());
      Ui.WaitUntil(() => ReviewSave.OverlayText(s, "EphemerisR") is { Length: > 0 }, TimeSpan.FromSeconds(30), "the ephemeris panel");

      Step("File › Export image...");
      s.Menu("File", "Export image...");
      Thread.Sleep(1000);
      s.AnswerFileDialog(chosen);
      Ui.WaitUntil(() => File.Exists(modelPath) && File.Exists(fitsPath), TimeSpan.FromSeconds(60), "both PNGs");
      Thread.Sleep(500);

      Step("the PNGs: viewport size, model not black");
      var model = ShotTools.ReadPng(modelPath, out int w, out int h);
      ShotTools.ReadPng(fitsPath, out int fw, out int fh);
      output.WriteLine($"model {w}×{h}, composite {fw}×{fh}, viewport {vp.Rect.Width}×{vp.Rect.Height}");
      Assert.Equal((w, h), (fw, fh));
      Assert.InRange(w, vp.Rect.Width - 2, vp.Rect.Width + 2);
      Assert.InRange(h, vp.Rect.Height - 2, vp.Rect.Height + 2);
      int lit = 0;
      for (int k = 0; k < w * h * 4; k += 4)
        if (Luma(model, k) > 0.05) lit++;
      output.WriteLine($"lit pixels {lit} of {w * h}");
      Assert.True(lit > w * h / 200, $"the model image is black ({lit} lit pixels)");

      Step("the model image is the engine's: the ephemeris panel is not in it");
      var panel = Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("EphemerisR").FirstOrDefault(e => e.Displayed),
        TimeSpan.FromSeconds(10), "the ephemeris R row");
      var rect = new Rectangle(panel.Location.X, panel.Location.Y, panel.Size.Width, panel.Size.Height);
      rect.Intersect(vp.Rect);
      output.WriteLine($"ephemeris row at {rect}");
      Assert.True(rect.Width > 10 && rect.Height > 5, $"the ephemeris row {rect} is not over the viewport {vp.Rect}");
      Win32.SavePng(Path.Combine(artifacts, "screen-ephemeris.png"), rect.Left, rect.Top, rect.Width, rect.Height);
      var screen = Win32.CaptureLuminance(rect.Left, rect.Top, rect.Width, rect.Height);
      int differing = 0;
      double sumAbs = 0;
      for (int y = 0; y < rect.Height; y++)
        for (int x = 0; x < rect.Width; x++)
        {
          int mx = rect.Left - vp.Rect.Left + x, my = rect.Top - vp.Rect.Top + y;
          if (mx < 0 || my < 0 || mx >= w || my >= h) continue;
          double d = Math.Abs(screen[y * rect.Width + x] - Luma(model, 4 * (my * w + mx)));
          sumAbs += d;
          if (d > 0.2) differing++;
        }
      output.WriteLine($"screen vs model in the ephemeris row: {differing} px differ by > 0.2, mean |Δ| {sumAbs / (rect.Width * rect.Height):0.000}");
      // the row's text is light on the dark sky: dozens of pixels at least
      Assert.True(differing > 20, "the model PNG shows the ephemeris overlay");

      Step("close");
      s.Process.CloseMainWindow();
      if (!s.Process.WaitForExit(5_000))
        try
        {
          Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("Discard").FirstOrDefault(e => e.Displayed),
            TimeSpan.FromSeconds(10), "the save prompt").Click();
        }
        catch (Exception e) when (e is TimeoutException or WebDriverException) { }
      if (s.Process.WaitForExit(30_000))
        exitCode = s.Process.ExitCode;
    }
    catch
    {
      try { s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch { }
      throw;
    }
    finally
    {
      s.Dispose();
    }
    output.WriteLine($"exit code {exitCode?.ToString(CultureInfo.InvariantCulture) ?? "none (killed after 30 s)"}");
    Assert.Equal(0, exitCode);
  }
}
