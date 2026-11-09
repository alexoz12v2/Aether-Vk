using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.ImageExport;
using AetherVk.Logic.Utils;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// File › Export image... on the real application with the committed 67P review save (one visible
/// FITS billboard over the viewport): both PNGs are written next to the chosen name, they are the
/// viewport's size, the model image is a real render (not black), the composite differs from the
/// model only where the billboard is, and the model image carries no engine label (a capture with
/// the labels shown differs from it in a small region).
/// </summary>
[Collection("X11")]
public sealed class ExportImageX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public ExportImageX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  private void Step(string s)
  {
    _output.WriteLine($"── {s}");
    Console.Error.WriteLine($"[ExportImageX11Tests] {s}");
  }

  private static bool Readable(string path)
  {
    try
    {
      using var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
      return true;
    }
    catch (IOException)
    {
      return false;
    }
  }

  private static double Luma(byte[] rgba, int k) => (0.2126 * rgba[k] + 0.7152 * rgba[k + 1] + 0.0722 * rgba[k + 2]) / 255.0;

  /// <summary>Pixels where the two images differ by more than <paramref name="tolerance"/> in a channel,
  /// and the bounding box of those pixels.</summary>
  private static (int Count, int X0, int Y0, int X1, int Y1) Diff(byte[] a, byte[] b, int w, int h, int tolerance = 8)
  {
    int count = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        int k = 4 * (y * w + x);
        if (Math.Abs(a[k] - b[k]) <= tolerance && Math.Abs(a[k + 1] - b[k + 1]) <= tolerance && Math.Abs(a[k + 2] - b[k + 2]) <= tolerance)
          continue;
        count++;
        x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
      }
    return (count, x0, y0, x1, y1);
  }

  [SkippableFact]
  public void ExportImage_WritesModelAndCompositePngs()
  {
    var fixture = AetherVk.TestAssets.TestAssets.Require("saves/review_67p_2021-12");
    using var rig = new AppRig();

    Step("open the review save");
    rig.Dialogs.Answer(rig.CopyOfSave(fixture));
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_67p_2021-12", StringComparison.Ordinal), 90_000),
      $"title names the save: {rig.XTitle}");
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();
    var billboards = rig.Host.Services.GetRequiredService<DesktopBillboardService>();
    Assert.True(WaitFor(() => Ui(() => billboards.ViewportFrame) is { WidthPx: > 0, HeightPx: > 0 }, 30_000),
      "the viewport reported its desktop frame");
    Assert.True(WaitFor(() => Ui(() => billboards.Current.Any(b => b.Visible)), 30_000), "the FITS billboard restored");
    Settle(3000); // the first frames after a restore

    Step("File › Export image...");
    var chosen = Path.Combine(rig.Root, "export", "shot.png");
    var (modelPath, fitsPath) = ImageExportService.FileNames(chosen);
    rig.Dialogs.Answer(chosen);
    rig.Menu("File", "Export image...");
    Assert.True(WaitFor(() => File.Exists(modelPath) && File.Exists(fitsPath)
      && Ui(() => rig.Vm.ExportStatus) is { Length: > 0 }, 60_000), $"both PNGs written; status {Ui(() => rig.Vm.ExportStatus)}");
    var status = Ui(() => rig.Vm.ExportStatus)!;
    _output.WriteLine(status);
    Assert.Contains(Path.GetFileName(fitsPath), status, StringComparison.Ordinal);
    Assert.StartsWith("save: ", rig.Dialogs.Requests[^1], StringComparison.Ordinal);

    Step("the model PNG: viewport size, not black");
    var model = PngCodec.ReadRgba(modelPath, out int w, out int h);
    var frame = Ui(() => billboards.ViewportFrame)!.Value;
    _output.WriteLine($"model {w}×{h}; viewport frame {frame.WidthPx}×{frame.HeightPx} at {frame.Origin} scaling {frame.Scaling}");
    Assert.InRange(w, (int)Math.Round(frame.WidthPx) - 2, (int)Math.Round(frame.WidthPx) + 2);
    Assert.InRange(h, (int)Math.Round(frame.HeightPx) - 2, (int)Math.Round(frame.HeightPx) + 2);
    int lit = 0;
    for (int k = 0; k < w * h * 4; k += 4)
      if (Luma(model, k) > 0.05) lit++;
    _output.WriteLine($"lit pixels {lit} of {w * h}");
    Assert.True(lit > w * h / 200, $"the model image is black ({lit} lit pixels)");

    Step("the composite differs from the model inside the billboard, nowhere else");
    var fits = PngCodec.ReadRgba(fitsPath, out int fw, out int fh);
    Assert.Equal((w, h), (fw, fh));
    var billboard = Ui(() => billboards.Current.First(b => b.Visible));
    // the billboard's bounding box in frame pixels
    double hw = billboard.WidthPx / 2, hh = billboard.HeightPx / 2;
    var corners = new[] { new PointD(-hw, -hh), new PointD(hw, -hh), new PointD(hw, hh), new PointD(-hw, hh) }
      .Select(c => DesktopBillboardGeometry.ToDesktop(billboard, c)).ToArray();
    double sx = frame.WidthPx / w, sy = frame.HeightPx / h;
    int bx0 = Math.Max(0, (int)Math.Floor((corners.Min(c => c.X) - frame.Origin.X) / sx)) - 1;
    int by0 = Math.Max(0, (int)Math.Floor((corners.Min(c => c.Y) - frame.Origin.Y) / sy)) - 1;
    int bx1 = Math.Min(w - 1, (int)Math.Ceiling((corners.Max(c => c.X) - frame.Origin.X) / sx)) + 1;
    int by1 = Math.Min(h - 1, (int)Math.Ceiling((corners.Max(c => c.Y) - frame.Origin.Y) / sy)) + 1;
    var d = Diff(model, fits, w, h, tolerance: 0);
    _output.WriteLine($"composite differs at {d.Count} px in [{d.X0},{d.Y0}]..[{d.X1},{d.Y1}]; billboard box [{bx0},{by0}]..[{bx1},{by1}]");
    Assert.True(d.Count > 0, "the composite equals the model although a billboard is visible");
    Assert.True(d.X0 >= bx0 && d.Y0 >= by0 && d.X1 <= bx1 && d.Y1 <= by1, "the composite differs outside the billboard");

    Step("the model PNG has no engine labels");
    // the same viewport with its labels drawn: the difference to a labels-hidden capture is the labels
    ulong pe = Ui(() => rig.Vm.ActiveViewport?.PresentationEngineId) ?? runtime.PresentationEngineId ?? 0;
    Assert.NotEqual(0UL, pe);
    // the engine draws the comet label in the Orbit view only (CameraService.ShowsCometLabel), so
    // the Model view the save opens in has no label to hide: check the flag in the Orbit view
    Ui(() => rig.Vm.ModeBar!.SelectOrbitCommand.Execute(null));
    Settle(3000);
    // the dust's adaptive level of detail settles over the first seconds after a restore: capture
    // until two consecutive labels-hidden frames agree, then the labels-shown one right after
    var hidden1 = runtime.CaptureViewportAsync(pe, hideLabels: true).GetAwaiter().GetResult();
    var hidden2 = runtime.CaptureViewportAsync(pe, hideLabels: true).GetAwaiter().GetResult();
    for (int attempt = 0; attempt < 30 && hidden1 is not null && hidden2 is not null
      && Diff(hidden1.ToRgba(), hidden2.ToRgba(), hidden1.Width, hidden1.Height).Count > hidden1.Width * hidden1.Height / 100; attempt++)
    {
      Settle(1000);
      hidden1 = hidden2;
      hidden2 = runtime.CaptureViewportAsync(pe, hideLabels: true).GetAwaiter().GetResult();
    }
    var shown = runtime.CaptureViewportAsync(pe, hideLabels: false).GetAwaiter().GetResult();
    Assert.NotNull(hidden1);
    Assert.NotNull(hidden2);
    Assert.NotNull(shown);
    Assert.Equal((w, h), (shown!.Width, shown.Height));
    var h1 = hidden1!.ToRgba();
    var h2 = hidden2!.ToRgba();
    var sh = shown.ToRgba();
    var drift = Diff(h1, h2, w, h);
    _output.WriteLine($"two labels-hidden captures differ at {drift.Count} px");
    var labels = Diff(h1, sh, w, h);
    _output.WriteLine($"labels shown vs hidden differ at {labels.Count} px in [{labels.X0},{labels.Y0}]..[{labels.X1},{labels.Y1}]");
    Skip.If(drift.Count > w * h / 100, "the scene is animating too much between frames to isolate the labels");
    Assert.True(labels.Count > drift.Count, "showing the labels changed nothing: the capture ignores the flag");
    // the Orbit view labels several bodies across the frame: bound the changed pixels, not their box
    Assert.True(labels.Count < w * h / 20, $"the labels change {labels.Count} px: more than text");
    // an export in the Orbit view: its model PNG has the hidden capture's pixels where the labels are
    var orbitChosen = Path.Combine(rig.Root, "export", "orbit.png");
    var (orbitModelPath, _) = ImageExportService.FileNames(orbitChosen);
    rig.Dialogs.Answer(orbitChosen);
    rig.Menu("File", "Export image...");
    // written in place: the PNG exists (held exclusively) before its last byte, so wait until it can be read
    Assert.True(WaitFor(() => File.Exists(orbitModelPath) && Readable(orbitModelPath), 60_000), "the Orbit view export written");
    var orbitModel = PngCodec.ReadRgba(orbitModelPath, out int ow, out int oh);
    Assert.Equal((w, h), (ow, oh));
    int asShown = 0, asHidden = 0;
    for (int y = labels.Y0; y <= labels.Y1; y++)
      for (int x = labels.X0; x <= labels.X1; x++)
      {
        int k = 4 * (y * w + x);
        bool differs(byte[] r) => Math.Abs(orbitModel[k] - r[k]) > 8 || Math.Abs(orbitModel[k + 1] - r[k + 1]) > 8
          || Math.Abs(orbitModel[k + 2] - r[k + 2]) > 8;
        if (differs(sh)) asShown++;
        if (differs(h2)) asHidden++;
      }
    _output.WriteLine($"Orbit export in the label box: {asShown} px differ from labels-shown, {asHidden} px from labels-hidden");
    Assert.True(asShown > asHidden, "the exported model PNG looks like the labels-shown capture: the labels were drawn");
  }
}
