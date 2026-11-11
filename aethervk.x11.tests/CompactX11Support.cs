using System.Diagnostics;
using AetherVk.Controls;
using AetherVk.Logic.Automation;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using AetherVk.UI;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Primitives;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// What the compact-layout and nucleus mini-view tests share: the layout switched through the real
/// Layout menu, the main viewport's native host, the review save opened, captures with a time
/// limit, image measures, and screen crops of the X root window.
/// </summary>
internal static class CompactX11
{
  public const string LayoutMenu = "Layout";
  public const string ClassicEntry = "Classic";
  public const string CompactEntry = "Compact";

  /// <summary>The fixture save every reviewer-flow test opens.</summary>
  public const string ReviewSave = "saves/review_67p_2021-12";

  // ── The layout ────────────────────────────────────────────────────────────────────────

  /// <summary>Layout › Compact or Layout › Classic, both real clicks; waits until the window shows it.</summary>
  public static void SetLayoutByMenu(AppRig rig, bool compact)
  {
    rig.Menu(LayoutMenu, compact ? CompactEntry : ClassicEntry);
    Assert.True(WaitFor(() => Ui(() => rig.Vm.IsCompactLayout) == compact, 10_000),
      $"Layout › {(compact ? CompactEntry : ClassicEntry)}: the window's layout is "
      + $"{(Ui(() => rig.Vm.IsCompactLayout) ? "Compact" : "Classic")}; status '{Ui(() => rig.Vm.LayoutStatus)}'");
    Settle(500);
  }

  /// <summary>The checked state of the Layout menu's two radio entries (the real MenuItems).</summary>
  public static (bool Classic, bool Compact) LayoutChecks(AppRig rig)
  {
    var layout = rig.TopMenu(LayoutMenu);
    var classic = AppRig.Child(layout, ClassicEntry);
    var compact = AppRig.Child(layout, CompactEntry);
    return Ui(() => (classic.IsChecked, compact.IsChecked));
  }

  // ── The main viewport ─────────────────────────────────────────────────────────────────

  /// <summary>Every native viewport control in the window with its model.</summary>
  public static List<(VulkanViewportControl Control, VulkanViewportControlViewModel? Vm)> ViewportControls(AppRig rig) =>
    Ui(() => rig.Main.GetVisualDescendants().OfType<VulkanViewportControl>()
      .Select(c => (c, c.DataContext as VulkanViewportControlViewModel)).ToList());

  /// <summary>The main viewport's native control (x:Name ViewportHost in Viewport3DView, role Main).</summary>
  public static VulkanViewportControl MainHost(AppRig rig)
  {
    VulkanViewportControl? host = null;
    Assert.True(WaitFor(() => (host = ViewportControls(rig)
      .Where(v => v.Vm is { Role: ViewportRole.Main })
      .Select(v => v.Control).FirstOrDefault()) is not null, 30_000), "the main viewport's native control is in the window");
    return host!;
  }

  /// <summary>The controls named ViewportHost (the main viewport's host; exactly one is expected).</summary>
  public static int ViewportHostCount(AppRig rig) =>
    Ui(() => rig.Main.GetVisualDescendants().OfType<VulkanViewportControl>().Count(c => c.Name == "ViewportHost"));

  public static VulkanViewportControlViewModel HostVm(VulkanViewportControl host) =>
    Ui(() => (VulkanViewportControlViewModel)host.DataContext!);

  public static Viewport3DViewModel MainViewport(AppRig rig) =>
    Ui(() => rig.Main.GetVisualDescendants().Select(v => v.DataContext).OfType<Viewport3DViewModel>().First());

  public static INativeRuntimeService Runtime(AppRig rig) => rig.Host.Services.GetRequiredService<INativeRuntimeService>();

  /// <summary>The main engine id once the main viewport attached it.</summary>
  public static ulong RequireMainPe(AppRig rig)
  {
    var runtime = Runtime(rig);
    Assert.True(WaitFor(() => runtime.PresentationEngineId is not null, 30_000), "the main viewport attached its engine");
    ulong pe = runtime.PresentationEngineId!.Value;
    Assert.True(WaitFor(() => runtime.ViewportFrameCount(pe) > 0, 30_000),
      $"the main viewport presented no frame (frame count {runtime.ViewportFrameCount(pe)})");
    return pe;
  }

  /// <summary>The X server still has the window.</summary>
  public static bool XidAlive(nint xid) =>
    xid != 0 && Run("xwininfo", $"-id {Hex(xid)}").Contains("Map State:", StringComparison.Ordinal);

  public static bool XidViewable(nint xid) =>
    xid != 0 && Run("xwininfo", $"-id {Hex(xid)}").Contains("Map State: IsViewable", StringComparison.Ordinal);

  /// <summary>The viewport overlay windows (one per Viewport3DView) and their X ids.</summary>
  /// <summary>The overlay windows of <paramref name="rig"/>'s main window (an earlier rig of the
  /// same test process may have left its own in the application's window list).</summary>
  public static List<nint> OverlayXids(AppRig rig) =>
    Ui(() => (Application.Current!.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.Windows
      .OfType<OverlayWindow>()
      .Where(w => ReferenceEquals(w.Owner, rig.Main))
      .Select(w => w.TryGetPlatformHandle()?.Handle ?? 0)
      .Where(h => h != 0)
      .ToList() ?? []);

  // ── The review save ───────────────────────────────────────────────────────────────────

  /// <summary>Opens a copy of the review save through File › Open... and waits for its comet.</summary>
  public static void OpenReviewSave(AppRig rig, Action<string> log)
  {
    var fixture = AetherVk.TestAssets.TestAssets.Require(ReviewSave);
    log("open the review save");
    rig.Dialogs.Answer(rig.CopyOfSave(fixture));
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_67p_2021-12", StringComparison.Ordinal), 90_000),
      $"the review save opened, title: {rig.XTitle}");
    var runtime = Runtime(rig);
    Assert.True(WaitFor(() => runtime.CometEntityId is not null, 60_000), "the review save's comet is in the engine");
    Assert.Equal(0, rig.Prompt.Asked);
  }

  // ── Clicking things in the compact panel ──────────────────────────────────────────────

  public static void ClickInView(AppRig rig, Control c)
  {
    Ui(() => c.BringIntoView());
    Settle(400);
    rig.Click(c);
  }

  /// <summary>Real keystrokes into a box: select all, type, Enter.</summary>
  public static void TypeInto(AppRig rig, Control box, string text)
  {
    ClickInView(rig, box);
    XDo("key --clearmodifiers ctrl+a");
    // "--": a value such as -49 is text, not an option
    XDo($"type --delay 40 -- {text}");
    XDo("key Return");
    Settle(400);
  }

  /// <summary>The text box inside a NumericUpDown (the id is on the NumericUpDown).</summary>
  public static TextBox NumericText(AppRig rig, string automationId)
  {
    var nud = rig.Require<NumericUpDown>(automationId);
    TextBox? box = null;
    Assert.True(WaitFor(() => (box = Ui(() => nud.GetVisualDescendants().OfType<TextBox>().FirstOrDefault())) is not null, 5_000),
      $"{automationId} has a text box");
    return box!;
  }

  // ── The nucleus mini-view ─────────────────────────────────────────────────────────────

  public static NucleusMiniViewModel Mini(AppRig rig) => Ui(() => rig.Vm.CompactPanel.NucleusMini);

  /// <summary>Expands or collapses the mini-view with a real click on its Expander's header.</summary>
  public static void SetMiniExpanded(AppRig rig, bool expanded)
  {
    var mini = Mini(rig);
    if (Ui(() => mini.IsExpanded) == expanded)
      return;
    var exp = rig.Require<Expander>(CompactAutomationIds.NucleusMiniViewExpander);
    var toggle = Ui(() => exp.GetVisualDescendants().OfType<ToggleButton>().First());
    rig.Click(toggle);
    Assert.True(WaitFor(() => Ui(() => mini.IsExpanded) == expanded, 5_000),
      $"the mini-view's expander is {(Ui(() => mini.IsExpanded) ? "expanded" : "collapsed")} after the header click");
    Settle(300);
  }

  /// <summary>The mini-view in the Compact layout, expanded, with its engine attached, the Earth view
  /// installed, resumed, and presenting frames. Returns its engine id.</summary>
  public static ulong ShowMini(AppRig rig)
  {
    if (!Ui(() => rig.Vm.IsCompactLayout))
      SetLayoutByMenu(rig, compact: true);
    SetMiniExpanded(rig, true);
    var mini = Mini(rig);
    var runtime = Runtime(rig);
    Assert.True(WaitFor(() => Ui(() => mini.Viewport?.PresentationEngineId ?? 0) != 0, 30_000),
      "the mini-view attached its engine once shown");
    Assert.True(WaitFor(() => Ui(() => mini.HasNucleusView && mini.IsActive), 60_000),
      $"the mini-view installed the view from Earth and resumed: has view {Ui(() => mini.HasNucleusView)}, "
      + $"active {Ui(() => mini.IsActive)}, comet {runtime.CometEntityId}, earth {runtime.EarthEntityId}");
    ulong pe = Ui(() => mini.Viewport!.PresentationEngineId);
    Assert.Equal(pe, runtime.NucleusMiniPresentationEngineId);
    ulong f0 = runtime.ViewportFrameCount(pe);
    Assert.True(WaitFor(() => runtime.ViewportFrameCount(pe) > f0 + 1, 15_000),
      $"the shown mini-view presents frames (count stayed {runtime.ViewportFrameCount(pe)})");
    return pe;
  }

  // ── Captures and image measures ───────────────────────────────────────────────────────

  /// <summary>A frame of <paramref name="pe"/> (null: none within the limit).</summary>
  public static CapturedImage? Capture(INativeRuntimeService runtime, ulong pe, bool hideLabels = false, int timeoutMs = 20_000)
  {
    var task = runtime.CaptureViewportAsync(pe, hideLabels);
    return task.Wait(timeoutMs) ? task.Result : null;
  }

  public static CapturedImage RequireCapture(INativeRuntimeService runtime, ulong pe, string what, bool hideLabels = false)
  {
    var sw = Stopwatch.StartNew();
    var image = Capture(runtime, pe, hideLabels);
    Assert.True(image is { Width: > 0, Height: > 0 },
      $"{what}: engine {pe} returned {(image is null ? "no frame" : $"an empty {image.Width}×{image.Height} frame")} in {sw.ElapsedMilliseconds} ms");
    return image!;
  }

  public static double Luma(byte[] rgba, int k) => (0.2126 * rgba[k] + 0.7152 * rgba[k + 1] + 0.0722 * rgba[k + 2]) / 255.0;

  /// <summary>Pixels brighter than <paramref name="threshold"/> (luma 0..1).</summary>
  public static int LitCount(byte[] rgba, int w, int h, double threshold = 0.05)
  {
    int lit = 0;
    for (int k = 0; k < w * h * 4; k += 4)
      if (Luma(rgba, k) > threshold) lit++;
    return lit;
  }

  /// <summary>
  /// The luma-weighted centroid of the bright pixels (luma above max(0.05, 30 % of the image's
  /// brightest)), their count, and that threshold.
  /// </summary>
  public static (double X, double Y, int Count, double Threshold) BrightCentroid(byte[] rgba, int w, int h)
  {
    double max = 0;
    for (int k = 0; k < w * h * 4; k += 4)
      max = Math.Max(max, Luma(rgba, k));
    double thr = Math.Max(0.05, 0.3 * max);
    double sx = 0, sy = 0, sw = 0;
    int count = 0;
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
      {
        double l = Luma(rgba, 4 * (y * w + x));
        if (l <= thr) continue;
        sx += x * l; sy += y * l; sw += l; count++;
      }
    return sw > 0 ? (sx / sw, sy / sw, count, thr) : (double.NaN, double.NaN, 0, thr);
  }

  /// <summary>The fraction of pixels whose colour differs by more than <paramref name="tolerance"/> in a channel.</summary>
  public static double DiffFraction(byte[] a, byte[] b, int w, int h, int tolerance = 24, int stride = 4)
  {
    int count = 0;
    for (int i = 0; i < w * h; i++)
    {
      int k = i * stride;
      if (Math.Abs(a[k] - b[k]) > tolerance || Math.Abs(a[k + 1] - b[k + 1]) > tolerance || Math.Abs(a[k + 2] - b[k + 2]) > tolerance)
        count++;
    }
    return (double)count / Math.Max(1, w * h);
  }

  // ── The X root window ─────────────────────────────────────────────────────────────────

  /// <summary>
  /// What the screen shows in the rectangle, as 8-bit RGB (3 bytes per pixel, rows top to bottom),
  /// read with ImageMagick's <c>import -window root</c>; null when <c>import</c> is missing.
  /// </summary>
  public static byte[]? RootCrop(int x, int y, int w, int h, string scratchDir)
  {
    if (!File.Exists("/usr/bin/import"))
      return null;
    Directory.CreateDirectory(scratchDir);
    var path = Path.Combine(scratchDir, $"crop-{Guid.NewGuid():N}.rgb");
    Run("import", $"-window root -crop {w}x{h}+{x}+{y} +repage -depth 8 rgb:{path}");
    if (!File.Exists(path))
      return null;
    var bytes = File.ReadAllBytes(path);
    File.Delete(path);
    return bytes.Length == w * h * 3 ? bytes : null;
  }

  /// <summary>The most frequent colour (channels quantized to 8 levels per 32) of an RGB buffer.</summary>
  public static (int R, int G, int B) ModeColour(byte[] rgb)
  {
    var counts = new Dictionary<int, (int N, long R, long G, long B)>();
    for (int k = 0; k + 2 < rgb.Length; k += 3)
    {
      int key = (rgb[k] >> 3) << 10 | (rgb[k + 1] >> 3) << 5 | (rgb[k + 2] >> 3);
      counts.TryGetValue(key, out var c);
      counts[key] = (c.N + 1, c.R + rgb[k], c.G + rgb[k + 1], c.B + rgb[k + 2]);
    }
    var best = counts.Values.MaxBy(c => c.N);
    return ((int)(best.R / best.N), (int)(best.G / best.N), (int)(best.B / best.N));
  }

  /// <summary>The fraction of an RGB buffer's pixels within <paramref name="tolerance"/> of <paramref name="colour"/>.</summary>
  public static double FractionNear(byte[] rgb, (int R, int G, int B) colour, int tolerance = 16)
  {
    int n = rgb.Length / 3, near = 0;
    for (int k = 0; k + 2 < rgb.Length; k += 3)
      if (Math.Abs(rgb[k] - colour.R) <= tolerance && Math.Abs(rgb[k + 1] - colour.G) <= tolerance
          && Math.Abs(rgb[k + 2] - colour.B) <= tolerance)
        near++;
    return (double)near / Math.Max(1, n);
  }

  /// <summary>A sub-rectangle of an RGB buffer <paramref name="w"/> pixels wide.</summary>
  public static byte[] SubRect(byte[] rgb, int w, int x0, int y0, int sw, int sh)
  {
    var r = new byte[sw * sh * 3];
    for (int y = 0; y < sh; y++)
      Array.Copy(rgb, ((y0 + y) * w + x0) * 3, r, y * sw * 3, sw * 3);
    return r;
  }

  // ── Engine poses ──────────────────────────────────────────────────────────────────────

  /// <summary>An entity's world position (AU) and rotation, from the engine's global transform.</summary>
  public static (Vector3d Pos, Quaterniond Rot)? Pose(INativeRuntimeService runtime, ulong entity)
  {
    if (entity == 0 || !runtime.DebugCameraState(entity, out var px, out var py, out var pz, out var rx, out var ry, out var rz, out var rw))
      return null;
    return (new Vector3d(px, py, pz), new Quaterniond(rx, ry, rz, rw));
  }

  /// <summary>The camera's view direction: its local −Y (CameraService.LookAtOriginFrom maps
  /// forward to engine −Y).</summary>
  public static Vector3d Forward(Quaterniond rot) => Vector3d.Normalize(Vector3d.Transform(new Vector3d(0, -1, 0), rot));

  public static double AngleDeg(Vector3d a, Vector3d b)
  {
    double c = Vector3d.Dot(Vector3d.Normalize(a), Vector3d.Normalize(b));
    return Math.Acos(Math.Clamp(c, -1.0, 1.0)) * 180.0 / Math.PI;
  }

  public static bool Readable(string path)
  {
    try
    {
      using var f = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
      return f.Length > 8;
    }
    catch (IOException)
    {
      return false;
    }
  }

  /// <summary>The file starts with the PNG signature.</summary>
  public static bool HasPngSignature(string path)
  {
    var sig = new byte[8];
    using var f = File.OpenRead(path);
    return f.Read(sig, 0, 8) == 8 && sig.SequenceEqual(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
  }
}
