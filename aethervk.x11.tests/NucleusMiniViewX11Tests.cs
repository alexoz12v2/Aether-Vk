using System.Diagnostics;
using System.Globalization;
using AetherVk.Logic.Automation;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.ImageExport;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The compact panel's "Nucleus seen from Earth" view on the real application: a second engine
/// viewport (<see cref="ViewportRole.NucleusMini"/>) whose camera the engine poses on the Earth →
/// comet line. Every test opens the review save, switches to the Compact layout through the
/// Layout menu and shows the mini-view expanded, unless it says otherwise.
/// </summary>
[Collection("X11")]
public sealed class NucleusMiniViewX11Tests
{
  private readonly ITestOutputHelper _output;

  public NucleusMiniViewX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  private void Report(string s)
  {
    _output.WriteLine(s);
    Console.Error.WriteLine($"[NucleusMiniViewX11Tests] {s}");
  }

  private void Step(string s) => Report($"── {s}");

  private sealed record Scene(AppRig Rig, INativeRuntimeService Runtime, NucleusMiniViewModel Mini, ulong MainPe, ulong MiniPe)
  {
    public nint MiniXid => Ui(() => Mini.Viewport?.NativeHandle ?? 0);
    public ulong MiniCamera => Ui(() => Mini.Viewport?.CameraEntityId ?? 0);
    public ulong MiniFrames => Runtime.ViewportFrameCount(MiniPe);
    public ulong MainFrames => Runtime.ViewportFrameCount(MainPe);
  }

  /// <summary>The review save open, the Compact layout, the mini-view shown and presenting.</summary>
  private Scene Build(AppRig rig)
  {
    var runtime = CompactX11.Runtime(rig);
    ulong mainPe = CompactX11.RequireMainPe(rig);
    CompactX11.OpenReviewSave(rig, Step);
    Step("Compact, the mini-view expanded");
    ulong miniPe = CompactX11.ShowMini(rig);
    Assert.NotEqual(mainPe, miniPe);
    Assert.Equal(mainPe, runtime.PresentationEngineId);
    var mini = CompactX11.Mini(rig);
    Assert.Equal(Ui(() => mini.Viewport!.CameraEntityId), runtime.NucleusMiniCameraEntityId ?? 0);
    var s = new Scene(rig, runtime, mini, mainPe, miniPe);
    Report($"main PE {mainPe}, mini PE {miniPe}, mini XID {Hex(s.MiniXid)}, mini camera {s.MiniCamera}, main camera {runtime.CameraEntityId}");
    Settle(2000); // the first frames of the view from Earth
    return s;
  }

  // ── The picture ─────────────────────────────────────────────────────────────────────────

  /// <summary>
  /// Proves the mini-view shows the nucleus, lit, in the middle of the view: a capture of the
  /// mini engine is not black (more than 0.2 % of its pixels brighter than luma 0.05) and the
  /// luma-weighted centroid of its bright pixels (above 30 % of the brightest) lies within 10 % of
  /// the image size from the image centre on both axes.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_ShowsTheLitNucleusCentred()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var image = CompactX11.RequireCapture(s.Runtime, s.MiniPe, "the mini-view");
    var rgba = image.ToRgba();
    int w = image.Width, h = image.Height;
    int lit = CompactX11.LitCount(rgba, w, h);
    var c = CompactX11.BrightCentroid(rgba, w, h);
    Report($"mini capture {w}×{h}: lit {lit} px ({(double)lit / (w * h):P2}); bright centroid ({c.X:0.#}, {c.Y:0.#}) "
      + $"over {c.Count} px above luma {c.Threshold:0.###}; centre ({w / 2.0:0.#}, {h / 2.0:0.#})");
    Assert.True(lit > w * h / 500, $"the mini-view is black: {lit} of {w * h} pixels lit");
    Assert.True(Math.Abs(c.X - w / 2.0) <= 0.1 * w && Math.Abs(c.Y - h / 2.0) <= 0.1 * h,
      $"the nucleus is off centre: centroid ({c.X:0.#}, {c.Y:0.#}) in a {w}×{h} view, allowed ±{0.1 * w:0.#}, ±{0.1 * h:0.#} px");
  }

  // ── Following the Earth ─────────────────────────────────────────────────────────────────

  private (double FwdDeg, double PosDeg, Vector3d Fwd, string Text) EarthAlignment(Scene s)
  {
    var cam = CompactX11.Pose(s.Runtime, s.MiniCamera);
    var comet = CompactX11.Pose(s.Runtime, s.Runtime.CometEntityId ?? 0);
    var earth = CompactX11.Pose(s.Runtime, s.Runtime.EarthEntityId ?? 0);
    if (cam is null || comet is null || earth is null)
      return (double.NaN, double.NaN, Vector3d.Zero,
        $"no pose: camera {cam is not null}, comet {comet is not null}, earth {earth is not null}");
    var earthToComet = comet.Value.Pos - earth.Value.Pos;
    var fwd = CompactX11.Forward(cam.Value.Rot);
    double fwdDeg = CompactX11.AngleDeg(fwd, earthToComet);
    // The camera stands on the Earth → comet line, on the Earth's side of the nucleus. It is
    // measured from the Earth, not from the comet: the engine places the camera behind the comet
    // as the Earth sees it (its light-time position, thousands of km from the geometric comet
    // transform, while the camera is only ~7500 km away), so the angle seen from the comet's
    // geometric position is meaningless; seen from the Earth that displacement is < 0.01°.
    var earthToCam = cam.Value.Pos - earth.Value.Pos;
    bool earthSide = earthToCam.Length() < earthToComet.Length();
    double posDeg = earthSide ? CompactX11.AngleDeg(earthToCam, earthToComet) : 180.0;
    string text = string.Create(CultureInfo.InvariantCulture,
      $"forward vs Earth→comet {fwdDeg:0.####}°, Earth→camera vs Earth→comet {posDeg:0.####}° (Earth's side {earthSide}), "
      + $"camera {(cam.Value.Pos - comet.Value.Pos).Length():0.###e0} AU from the comet, Earth–comet {earthToComet.Length():0.####} AU");
    return (fwdDeg, posDeg, fwd, text);
  }

  private static long ClockUs(INativeRuntimeService runtime)
  {
    Assert.True(runtime.GetSimulationClock(out short centuries, out ulong ns), "the engine clock");
    const long UsPerDay = 86_400L * 1_000_000L;
    return centuries * 36525L * UsPerDay + (long)(ns / 1000UL);
  }

  /// <summary>
  /// Proves the mini camera follows the Earth → comet direction as the clock moves: its view
  /// direction (the camera's local −Y in the engine's global transform) is within 0.5° of the
  /// Earth → comet direction and its position seen from the Earth within 0.5° of that direction,
  /// nearer to the Earth than the comet is (Earth and comet positions from the engine's own global transforms), before
  /// and after a seek; the seek turns the direction by more than 0.1°, so the check is not met by
  /// a pose frozen at the start. The seek goes 30 days back; the review save's committed range is
  /// shorter (16 days), and a seek outside it is refused, so the test then seeks to 15 days back
  /// and reports it. The engine exposes no "mini camera forward" query: the forward is read from
  /// <see cref="INativeRuntimeService.DebugCameraState"/> (the global transform of any entity).
  /// The comet ephemeris (RA/Dec from the Earth's centre) is reported as a cross-check.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_FollowsEarthDirection()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var timeline = Ui(() => rig.Vm.CompactPanel.Timeline!);

    var a0 = EarthAlignment(s);
    Assert.True(WaitFor(() => (a0 = EarthAlignment(s)) is { FwdDeg: <= 0.5, PosDeg: <= 0.5 }, 30_000),
      $"before the seek: {a0.Text}");
    Report($"before the seek: {a0.Text}");
    ReportEphemeris(s, a0.Fwd, "before the seek");

    long clock0 = ClockUs(s.Runtime);
    string box = Ui(() => timeline.CurrentEpochText);
    Assert.True(DateTime.TryParseExact(box, TimelineTabViewModel.EpochBoxFormat, CultureInfo.InvariantCulture, DateTimeStyles.None, out var now),
      $"the epoch box reads '{box}'");
    int days = 30;
    string Seek(int d)
    {
      string text = now.AddDays(-d).ToString(TimelineTabViewModel.EpochBoxFormat, CultureInfo.InvariantCulture);
      Ui(() =>
      {
        timeline.CurrentEpochText = text;
        timeline.SeekToEpochCommand.Execute(null);
      });
      Settle(1000);
      return text;
    }
    string target = Seek(days);
    string error = Ui(() => timeline.CurrentEpochError);
    if (error.Length > 0)
    {
      Report($"a {days}-day seek to {target} was refused: {error}");
      days = 15;
      target = Seek(days);
      error = Ui(() => timeline.CurrentEpochError);
    }
    Assert.True(error.Length == 0, $"the seek to {target} was refused: {error}");
    Assert.True(WaitFor(() => ClockUs(s.Runtime) < clock0 - (days - 1) * 86_400_000_000L, 60_000),
      $"the {days}-day seek to {target} did not move the clock ({clock0} → {ClockUs(s.Runtime)} µs)");
    Report($"seeked {days} days back, to {target}");

    var a1 = a0;
    bool aligned = WaitFor(() => (a1 = EarthAlignment(s)) is { FwdDeg: <= 0.5, PosDeg: <= 0.5 }, 30_000);
    double turned = CompactX11.AngleDeg(a0.Fwd, a1.Fwd);
    Report($"after the seek: {a1.Text}; the view direction turned {turned:0.###}°");
    ReportEphemeris(s, a1.Fwd, "after the seek");
    Assert.True(aligned, $"after the {days}-day seek: {a1.Text}");
    Assert.True(turned > 0.1, $"the view direction turned only {turned:0.####}° over a {days}-day seek: the pose did not follow the clock");
  }

  private void ReportEphemeris(Scene s, Vector3d fwd, string when)
  {
    if (s.Runtime.CometEphemeris() is not { } eph)
    {
      Report($"{when}: no comet ephemeris");
      return;
    }
    const double RadToDeg = 180.0 / Math.PI;
    var dir = SkyMath.EclipticDirection(eph.RaRad * RadToDeg, eph.DecRad * RadToDeg);
    Report(string.Create(CultureInfo.InvariantCulture,
      $"{when}: ephemeris {eph.Epoch:O} RA {eph.RaRad * RadToDeg:0.####}° Dec {eph.DecRad * RadToDeg:0.####}°, Δ {eph.DeltaAu:0.####} AU; "
      + $"forward vs ephemeris direction {CompactX11.AngleDeg(fwd, dir):0.####}°"));
  }

  // ── Pausing ─────────────────────────────────────────────────────────────────────────────

  /// <summary>The mini's frame count does not move over 2.5 s while the main one does.</summary>
  private void AssertMiniFrozen(Scene s, string phase)
  {
    Settle(1000); // a frame in flight when it was paused
    ulong mini0 = s.MiniFrames, main0 = s.MainFrames;
    Thread.Sleep(2500);
    ulong mini1 = s.MiniFrames, main1 = s.MainFrames;
    Report($"{phase}: over 2.5 s mini frames {mini0} → {mini1}, main frames {main0} → {main1}, mini active {Ui(() => s.Mini.IsActive)}");
    Assert.True(mini1 == mini0, $"{phase}: the mini-view kept presenting ({mini0} → {mini1} frames)");
    Assert.True(main1 > main0, $"{phase}: the main viewport stopped presenting ({main0} → {main1} frames)");
    Assert.False(Ui(() => s.Mini.IsActive), $"{phase}: the mini-view is still marked active");
  }

  private void AssertMiniResumes(Scene s, string phase)
  {
    ulong mini0 = s.MiniFrames;
    var sw = Stopwatch.StartNew();
    bool resumed = WaitFor(() => s.MiniFrames > mini0, 2_000);
    Report($"{phase}: mini frames {mini0} → {s.MiniFrames} after {sw.ElapsedMilliseconds} ms");
    Assert.True(resumed, $"{phase}: the mini-view presented no frame within 2 s (count {s.MiniFrames})");
  }

  /// <summary>
  /// Proves the mini-view draws only while it is shown: collapsed (a real click on its Expander's
  /// header), its presented-frame count stays fixed for 2.5 s while the main viewport's count
  /// grows; expanded again, it presents a new frame within 2 s. The same holds for the Classic
  /// layout (the panel hidden: paused) and back to Compact (resumed). Its engine id never changes.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_CollapseStopsItsFrames_MainKeepsRendering()
  {
    using var rig = new AppRig();
    var s = Build(rig);

    Step("collapse");
    CompactX11.SetMiniExpanded(rig, false);
    AssertMiniFrozen(s, "collapsed");
    Step("expand");
    CompactX11.SetMiniExpanded(rig, true);
    AssertMiniResumes(s, "expanded");

    Step("Classic");
    CompactX11.SetLayoutByMenu(rig, compact: false);
    AssertMiniFrozen(s, "Classic layout");
    Step("Compact");
    CompactX11.SetLayoutByMenu(rig, compact: true);
    AssertMiniResumes(s, "Compact again");

    Assert.Equal(s.MiniPe, s.Runtime.NucleusMiniPresentationEngineId);
    Assert.Equal(s.MainPe, s.Runtime.PresentationEngineId);
  }

  /// <summary>
  /// Proves the layout switch never creates or loses an engine: ten Compact → Classic → Compact
  /// round trips through the Layout menu keep the main engine id and the mini engine id
  /// (<see cref="INativeRuntimeService.NucleusMiniPresentationEngineId"/>) unchanged, the mini's
  /// native window (same XID, still known to the X server, handed over once) and its view model
  /// instance; the window never holds more than the two native viewports, and no other engine id
  /// appears on them; the mini resumes presenting in every Compact phase and is paused in Classic.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_LayoutToggle10x_SamePeNoLeak()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var viewport = Ui(() => s.Mini.Viewport!);
    nint xid = s.MiniXid;
    int inits = Ui(() => viewport.InitializeCount);
    Assert.NotEqual((nint)0, xid);

    void Same(string when)
    {
      Assert.True(s.Runtime.PresentationEngineId == s.MainPe, $"{when}: the main engine is {s.Runtime.PresentationEngineId}, was {s.MainPe}");
      Assert.True(s.Runtime.NucleusMiniPresentationEngineId == s.MiniPe,
        $"{when}: the mini engine is {s.Runtime.NucleusMiniPresentationEngineId}, was {s.MiniPe}");
      Assert.True(ReferenceEquals(Ui(() => s.Mini.Viewport), viewport), $"{when}: the mini-view's native view model was replaced");
      Assert.True(s.MiniXid == xid, $"{when}: the mini's native window changed, {Hex(xid)} → {Hex(s.MiniXid)}");
      Assert.True(CompactX11.XidAlive(xid), $"{when}: the X server lost the mini's window {Hex(xid)}");
      Assert.True(Ui(() => viewport.InitializeCount) == inits, $"{when}: the mini's window was handed over {Ui(() => viewport.InitializeCount)} times");
      var controls = CompactX11.ViewportControls(rig);
      var pes = controls.Select(c => Ui(() => c.Vm?.PresentationEngineId ?? 0)).Where(p => p != 0).Distinct().ToList();
      Assert.True(controls.Count <= 2, $"{when}: {controls.Count} native viewports in the window");
      Assert.True(pes.All(p => p == s.MainPe || p == s.MiniPe), $"{when}: engines [{string.Join(", ", pes)}] besides main {s.MainPe} and mini {s.MiniPe}");
    }

    for (int i = 1; i <= 10; i++)
    {
      CompactX11.SetLayoutByMenu(rig, compact: false);
      Same($"switch {i} to Classic");
      Assert.True(WaitFor(() => !Ui(() => s.Mini.IsActive), 5_000), $"switch {i} to Classic: the mini-view was not paused");
      CompactX11.SetLayoutByMenu(rig, compact: true);
      Same($"switch {i} to Compact");
      ulong f0 = s.MiniFrames;
      Assert.True(WaitFor(() => s.MiniFrames > f0, 5_000), $"switch {i} to Compact: the mini-view did not resume (frames {s.MiniFrames})");
      Report($"round trip {i}: main PE {s.Runtime.PresentationEngineId}, mini PE {s.Runtime.NucleusMiniPresentationEngineId}, "
        + $"mini frames {s.MiniFrames}, main frames {s.MainFrames}");
    }
  }

  // ── Cost ────────────────────────────────────────────────────────────────────────────────

  /// <summary>
  /// Proves the shown mini-view costs the main viewport at most 10 % of its frame rate: the main
  /// engine's presented frames per second over 20 s windows, three with the mini-view shown and
  /// three with it collapsed, interleaved (shown, collapsed, shown, ...) so a drift of the Debug
  /// build's frame rate over the run affects both alike; the median of each three is compared
  /// (a single Debug-build window is noisy). All six rates are reported.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_MainFrameRateWithin10Percent()
  {
    using var rig = new AppRig();
    var s = Build(rig);

    double Window(string phase)
    {
      Settle(1000);
      ulong f0 = s.MainFrames;
      var sw = Stopwatch.StartNew();
      Thread.Sleep(20_000);
      ulong f1 = s.MainFrames;
      double fps = (f1 - f0) / sw.Elapsed.TotalSeconds;
      Report(string.Create(CultureInfo.InvariantCulture, $"{phase}: main {f1 - f0} frames in {sw.Elapsed.TotalSeconds:0.##} s = {fps:0.###} fps"));
      return fps;
    }

    var shown = new List<double>();
    var collapsed = new List<double>();
    for (int i = 1; i <= 3; i++)
    {
      CompactX11.SetMiniExpanded(rig, true);
      Assert.True(WaitFor(() => Ui(() => s.Mini.IsActive), 5_000), "the mini-view resumed");
      shown.Add(Window($"window {i}, mini shown"));
      CompactX11.SetMiniExpanded(rig, false);
      collapsed.Add(Window($"window {i}, mini collapsed"));
    }
    static double Median(List<double> v) => v.OrderBy(x => x).ElementAt(v.Count / 2);
    double ms = Median(shown), mc = Median(collapsed);
    double drop = mc > 0 ? (mc - ms) / mc : double.NaN;
    Report(string.Create(CultureInfo.InvariantCulture,
      $"median main fps: mini shown {ms:0.###} [{string.Join(", ", shown.Select(x => x.ToString("0.###", CultureInfo.InvariantCulture)))}], "
      + $"collapsed {mc:0.###} [{string.Join(", ", collapsed.Select(x => x.ToString("0.###", CultureInfo.InvariantCulture)))}]; drop {drop:P1}"));
    Assert.True(mc > 0, "the main viewport presented no frame with the mini-view collapsed");
    Assert.True(Math.Abs(ms - mc) <= 0.1 * mc,
      string.Create(CultureInfo.InvariantCulture, $"main fps with the mini-view shown {ms:0.###} vs collapsed {mc:0.###}: {drop:P1} apart, more than 10 %"));
  }

  // ── Stacking ────────────────────────────────────────────────────────────────────────────

  /// <summary>The union of a top menu's visible entries on the screen (physical pixels).</summary>
  private static (int X, int Y, int W, int H)? DropDownRect(MenuItem top) => Ui(() =>
  {
    int x0 = int.MaxValue, y0 = int.MaxValue, x1 = int.MinValue, y1 = int.MinValue;
    foreach (var item in top.Items.OfType<MenuItem>().Where(m => m.IsVisible && m.GetVisualRoot() is not null))
    {
      var tl = item.PointToScreen(new Point(0, 0));
      var br = item.PointToScreen(new Point(item.Bounds.Width, item.Bounds.Height));
      x0 = Math.Min(x0, tl.X); y0 = Math.Min(y0, tl.Y); x1 = Math.Max(x1, br.X); y1 = Math.Max(y1, br.Y);
    }
    return x1 > x0 && y1 > y0 ? ((int X, int Y, int W, int H)?)(x0, y0, x1 - x0, y1 - y0) : null;
  });

  /// <summary>
  /// Proves Avalonia drop-downs draw above the mini-view's native window: for each top menu
  /// (File, Edit, Layout) whose drop-down overlaps the mini's rectangle (its X geometry) by at
  /// least 16 × 16 px, the screen (<c>import -window root</c>) is read in the overlap before and
  /// with the drop-down open. Open, the overlap must look like the drop-down: at least 50 % of its
  /// pixels within 16 levels of the drop-down's background, taken as the most frequent colour of a
  /// strip of the same drop-down beside the mini (or of the overlap itself when the drop-down has
  /// no part beside the mini), and at least 50 % of the pixels must differ from what the 3D view
  /// showed there before. At least one menu must overlap. ComboBox drop-downs are not part of it:
  /// the panel's ComboBoxes sit in the cards below the mini-view and open downwards, away from it.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_MenuAndComboDropDownsDrawAboveIt()
  {
    Skip.If(!File.Exists("/usr/bin/import"), "ImageMagick's import is not installed");
    using var rig = new AppRig();
    var s = Build(rig);
    var scratch = Path.Combine(rig.Root, "crops");
    var mr = XWinInfo(s.MiniXid);
    Report($"mini window {mr.Width}×{mr.Height}+{mr.X}+{mr.Y}, mapped {mr.Viewable}");
    Assert.True(mr.Viewable && mr.Width > 32 && mr.Height > 32, $"the mini-view's window is not shown: {mr}");
    var before = CompactX11.RootCrop(mr.X, mr.Y, mr.Width, mr.Height, scratch);
    Assert.True(before is not null, "the screen could not be read with import");

    int tested = 0;
    foreach (var header in new[] { "File", "Edit", CompactX11.LayoutMenu })
    {
      var top = rig.TopMenu(header);
      rig.Click(top);
      Assert.True(WaitFor(() => Ui(() => top.IsSubMenuOpen), 5_000), $"{header} menu open");
      Settle(1000);
      var dd = DropDownRect(top);
      if (dd is not { } d)
      {
        rig.CloseMenus();
        Report($"{header}: no visible entries");
        continue;
      }
      int ix0 = Math.Max(d.X, mr.X), iy0 = Math.Max(d.Y, mr.Y);
      int ix1 = Math.Min(d.X + d.W, mr.X + mr.Width), iy1 = Math.Min(d.Y + d.H, mr.Y + mr.Height);
      Report($"{header}: drop-down {d.W}×{d.H}+{d.X}+{d.Y}, overlap with the mini {Math.Max(0, ix1 - ix0)}×{Math.Max(0, iy1 - iy0)}");
      if (ix1 - ix0 < 16 || iy1 - iy0 < 16)
      {
        rig.CloseMenus();
        continue;
      }
      int ow = ix1 - ix0, oh = iy1 - iy0;
      var open = CompactX11.RootCrop(ix0, iy0, ow, oh, scratch);
      // a strip of the drop-down beside the mini, on the same rows: its background
      byte[]? strip = null;
      if (mr.X - d.X >= 12)
        strip = CompactX11.RootCrop(d.X + 2, iy0, Math.Min(mr.X - d.X - 4, 40), oh, scratch);
      else if (d.X + d.W - (mr.X + mr.Width) >= 12)
        strip = CompactX11.RootCrop(mr.X + mr.Width + 2, iy0, Math.Min(d.X + d.W - (mr.X + mr.Width) - 4, 40), oh, scratch);
      rig.CloseMenus();
      Assert.True(open is not null, $"{header}: the screen could not be read");

      var bg = CompactX11.ModeColour(strip ?? open!);
      double near = CompactX11.FractionNear(open!, bg);
      var was = CompactX11.SubRect(before!, mr.Width, ix0 - mr.X, iy0 - mr.Y, ow, oh);
      double changed = CompactX11.DiffFraction(was, open!, ow, oh, stride: 3);
      double wasNear = CompactX11.FractionNear(was, bg);
      Report($"{header}: drop-down background rgb{bg} (from {(strip is null ? "the overlap" : "a strip beside the mini")}); "
        + $"open: {near:P1} of the overlap is that colour, {changed:P1} differs from the 3D view; the 3D view had {wasNear:P1} of it");
      Assert.True(near >= 0.5, $"{header}: only {near:P1} of the overlap shows the drop-down's background rgb{bg}: the 3D view draws over it");
      Assert.True(changed >= 0.5, $"{header}: only {changed:P1} of the overlap changed when the drop-down opened");
      tested++;
    }
    Assert.True(tested > 0, $"no top menu's drop-down overlaps the mini-view at {mr.Width}×{mr.Height}+{mr.X}+{mr.Y}");
  }

  // ── Input ───────────────────────────────────────────────────────────────────────────────

  /// <summary>
  /// Proves the keyboard keeps driving the main 3D view while the mini-view has the focus: with
  /// the main view in Orbit, a click on the mini-view (no drag) focuses it, then the camera hotkey
  /// V (cycle camera mode) is sent with xdotool; the main camera's mode changes, the main capture
  /// changes, and the mini camera's pose (engine global transform) and its view from Earth stay as
  /// they were.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_KeyboardStillDrivesMainCamera()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var camera = CompactX11.MainViewport(rig).CameraService;
    Ui(() => rig.Vm.ModeBar!.SelectOrbitCommand.Execute(null));
    Assert.True(WaitFor(() => Ui(() => rig.Vm.ModeBar!.IsOrbitActive), 10_000), "the main view in Orbit");
    Settle(3000);

    var miniPose = CompactX11.Pose(s.Runtime, s.MiniCamera);
    Assert.NotNull(miniPose);
    var mainBefore = CompactX11.RequireCapture(s.Runtime, s.MainPe, "the main view before the key", hideLabels: true);
    var modeBefore = Ui(() => camera.CurrentMode);

    var mr = XWinInfo(s.MiniXid);
    XDo($"mousemove {mr.X + mr.Width / 2} {mr.Y + mr.Height / 2} sleep 0.2 click 1");
    Settle(500);
    Assert.True(Ui(() => s.Mini.IsEarthView), "the click on the mini-view turned it away from the view from Earth");
    XDo("key v");
    bool switched = WaitFor(() => Ui(() => camera.CurrentMode) != modeBefore, 10_000);
    var modeAfter = Ui(() => camera.CurrentMode);
    Settle(3000);
    var mainAfter = CompactX11.RequireCapture(s.Runtime, s.MainPe, "the main view after the key", hideLabels: true);
    double changed = mainBefore.Width == mainAfter.Width && mainBefore.Height == mainAfter.Height
      ? CompactX11.DiffFraction(mainBefore.ToRgba(), mainAfter.ToRgba(), mainBefore.Width, mainBefore.Height)
      : 1.0;
    var miniAfter = CompactX11.Pose(s.Runtime, s.MiniCamera);
    double miniMovedAu = miniAfter is { } p ? (p.Pos - miniPose!.Value.Pos).Length() : double.NaN;
    double miniTurnedDeg = miniAfter is { } q ? CompactX11.AngleDeg(CompactX11.Forward(q.Rot), CompactX11.Forward(miniPose!.Value.Rot)) : double.NaN;
    Report($"key V with the mini focused: main mode {modeBefore} → {modeAfter}, main capture changed at {changed:P1}; "
      + $"mini camera moved {miniMovedAu:0.###e0} AU, turned {miniTurnedDeg:0.####}°, Earth view {Ui(() => s.Mini.IsEarthView)}");
    Assert.True(switched, $"the key did not reach the main camera: its mode stayed {modeBefore}");
    Assert.True(changed > 0.01, $"the main view did not change after the mode switch ({changed:P2} of the pixels)");
    Assert.True(miniAfter is not null && miniMovedAu < 1e-9 && miniTurnedDeg < 0.01,
      $"the key moved the mini camera: {miniMovedAu:0.###e0} AU, {miniTurnedDeg:0.####}°");
    Assert.True(Ui(() => s.Mini.IsEarthView), "the mini-view left the view from Earth");
  }

  /// <summary>
  /// Proves the camera keys reach the main 3D view after a click on a desktop billboard (the review
  /// save's FITS image, shown over the view in Compact): a real click on the billboard's image,
  /// then V (cycle camera mode), changes the main camera's mode exactly once; then a real click on
  /// the main view itself, away from the billboard's window, and V again changes it exactly once
  /// more. The mode changes are counted on the camera's mode stream (a key delivered twice would
  /// count two).
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_KeyAfterAClickOnTheBillboardDrivesMainCamera()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var camera = CompactX11.MainViewport(rig).CameraService;
    var billboards = rig.Host.Services.GetService(typeof(AetherVk.UI.DesktopBillboardWindowManager)) as AetherVk.UI.DesktopBillboardWindowManager;
    Assert.NotNull(billboards);
    AetherVk.UI.DesktopBillboardWindow? window = null;
    Assert.True(WaitFor(() => (window = Ui(() => billboards!.Windows.FirstOrDefault(w => w.IsVisible
      && w.GetVisualDescendants().OfType<Avalonia.Controls.Image>().Any(i => i.IsVisible && i.Source is not null)))) is not null, 30_000),
      "the review save's billboard shows its image in Compact");
    var image = Ui(() => window!.GetVisualDescendants().OfType<Avalonia.Controls.Image>().First(i => i.IsVisible && i.Source is not null));
    var imageCentre = Ui(() => image.PointToScreen(new Point(image.Bounds.Width / 2, image.Bounds.Height / 2)));
    var bw = XWinInfo(Ui(() => window!.TryGetPlatformHandle()!.Handle));
    var vw = XWinInfo(CompactX11.HostVm(CompactX11.MainHost(rig)).NativeHandle);
    Report($"billboard window {bw.Width}×{bw.Height}+{bw.X}+{bw.Y}, image centre {imageCentre.X},{imageCentre.Y}; "
      + $"main view {vw.Width}×{vw.Height}+{vw.X}+{vw.Y}");

    int changes = 0;
    using var sub = camera.CameraModeChanged.Subscribe(_ => Interlocked.Increment(ref changes));
    Settle(500);
    int baseline = Volatile.Read(ref changes); // a replayed current value, if the stream has one

    // 1. the billboard's image, then V
    Step("click on the billboard image, V");
    var mode0 = Ui(() => camera.CurrentMode);
    XDo($"mousemove {imageCentre.X} {imageCentre.Y} sleep 0.2 click 1");
    Settle(500);
    XDo("key v");
    Assert.True(WaitFor(() => Ui(() => camera.CurrentMode) != mode0, 10_000),
      $"a click on the billboard, then V: the main camera stayed {mode0}");
    Settle(1500);
    var mode1 = Ui(() => camera.CurrentMode);
    int afterBillboard = Volatile.Read(ref changes) - baseline;
    Report($"after the billboard click: {mode0} → {mode1}, {afterBillboard} mode change(s)");
    Assert.Equal(1, afterBillboard);

    // 2. the main view, at a point outside the billboard's window, then V
    var point = new[] { (vw.X + 12, vw.Y + 12), (vw.X + vw.Width - 12, vw.Y + 12), (vw.X + 12, vw.Y + vw.Height - 12),
        (vw.X + vw.Width - 12, vw.Y + vw.Height - 12) }
      .FirstOrDefault(p => p.Item1 < bw.X || p.Item1 >= bw.X + bw.Width || p.Item2 < bw.Y || p.Item2 >= bw.Y + bw.Height);
    Skip.If(point == default, "the billboard window covers every corner of the main view");
    Step($"click on the main view at {point.Item1},{point.Item2}, V");
    XDo($"mousemove {point.Item1} {point.Item2} sleep 0.2 click 1");
    Settle(500);
    XDo("key v");
    Assert.True(WaitFor(() => Ui(() => camera.CurrentMode) != mode1, 10_000),
      $"a click on the main view, then V: the main camera stayed {mode1}");
    Settle(1500);
    int afterView = Volatile.Read(ref changes) - baseline - afterBillboard;
    Report($"after the main view click: {mode1} → {Ui(() => camera.CurrentMode)}, {afterView} mode change(s)");
    Assert.Equal(1, afterView);
  }

  /// <summary>
  /// Proves dragging over the mini-view orbits its camera and the reset button brings back the
  /// view from Earth: an xdotool drag of 80 × 30 px over the mini's window turns the view
  /// (IsEarthView false) and changes the capture (more than 2 % of the pixels differ by more than
  /// 24 levels from the Earth view's capture); after a real click on the reset button
  /// (<see cref="CompactAutomationIds.NucleusMiniViewResetToEarth"/>) the view is the Earth view
  /// again and its capture is close to the original one, "close" meaning at most 2 % of the
  /// pixels differ by more than 24 levels and the bright-pixel centroid is within 2 % of the image
  /// size of the original's.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_DragOrbitsAndResetReturnsToEarthView()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var c0 = CompactX11.RequireCapture(s.Runtime, s.MiniPe, "the view from Earth");
    var rgba0 = c0.ToRgba();
    var centre0 = CompactX11.BrightCentroid(rgba0, c0.Width, c0.Height);

    Step("drag over the mini-view");
    var mr = XWinInfo(s.MiniXid);
    XDo($"mousemove {mr.X + mr.Width / 2 - 40} {mr.Y + mr.Height / 2 - 15} sleep 0.2 mousedown 1 sleep 0.1");
    for (int i = 0; i < 10; i++)
      XDo("mousemove_relative 8 3 sleep 0.05");
    XDo("sleep 0.1 mouseup 1");
    Assert.True(WaitFor(() => !Ui(() => s.Mini.IsEarthView), 5_000), "the drag did not take the mini camera off the view from Earth");
    Settle(1500);
    var c1 = CompactX11.RequireCapture(s.Runtime, s.MiniPe, "after the drag");
    Assert.Equal((c0.Width, c0.Height), (c1.Width, c1.Height));
    double dragged = CompactX11.DiffFraction(rgba0, c1.ToRgba(), c0.Width, c0.Height);
    Report($"after the drag: {dragged:P2} of the pixels differ from the view from Earth");
    Assert.True(dragged > 0.02, $"the drag changed only {dragged:P2} of the mini-view's pixels");

    Step("reset to the view from Earth");
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.NucleusMiniViewResetToEarth));
    Assert.True(WaitFor(() => Ui(() => s.Mini.IsEarthView), 5_000), "the reset did not return to the view from Earth");
    double diff = 1;
    double shift = double.NaN;
    bool close = WaitFor(() =>
    {
      var c2 = CompactX11.Capture(s.Runtime, s.MiniPe);
      if (c2 is null || c2.Width != c0.Width || c2.Height != c0.Height)
        return false;
      var rgba2 = c2.ToRgba();
      diff = CompactX11.DiffFraction(rgba0, rgba2, c0.Width, c0.Height);
      var centre2 = CompactX11.BrightCentroid(rgba2, c2.Width, c2.Height);
      shift = Math.Max(Math.Abs(centre2.X - centre0.X) / c0.Width, Math.Abs(centre2.Y - centre0.Y) / c0.Height);
      return diff <= 0.02 && shift <= 0.02;
    }, 15_000);
    Report($"after the reset: {diff:P2} of the pixels differ from the original, centroid shifted {shift:P2} of the size");
    Assert.True(close, $"after the reset the mini-view is not the original view from Earth: {diff:P2} of the pixels differ, centroid shifted {shift:P2}");
  }

  // ── Shutdown and export ─────────────────────────────────────────────────────────────────

  /// <summary>
  /// Proves the application shuts down cleanly with the mini-view's engine attached, five times
  /// in a row: each rig opens the review save with the mini-view shown (collapsed again before
  /// the shutdown in the even rounds); the shutdown completes within its limit, throws nothing,
  /// and by the time the engine stops both the main and the mini engine are detached.
  /// x11_e2e.sh runs each round in its own test host (X11_E2E_SHUTDOWN_ROUND = 1…5), as the
  /// application runs one engine per process: on the NVIDIA driver a second engine in a process
  /// that already shut one down with two windowed views stalls ~22 s in every swapchain call.
  /// Without the variable all five rounds run in this process.
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_ShutdownClean_5x()
  {
    var only = Environment.GetEnvironmentVariable("X11_E2E_SHUTDOWN_ROUND");
    int first = 1, last = 5;
    if (int.TryParse(only, out int r) && r is >= 1 and <= 5)
      first = last = r;
    for (int round = first; round <= last; round++)
    {
      var rig = new AppRig();
      bool disposed = false;
      try
      {
        var s = Build(rig);
        if (round % 2 == 0)
          CompactX11.SetMiniExpanded(rig, false);
        Report($"round {round}: main PE {s.MainPe}, mini PE {s.MiniPe}, mini {(round % 2 == 0 ? "collapsed" : "shown")}");

        bool hookRan = false;
        ulong? mainAtStop = ulong.MaxValue, miniAtStop = ulong.MaxValue;
        rig.BeforeEngineShutdown = () =>
        {
          mainAtStop = s.Runtime.PresentationEngineId;
          miniAtStop = s.Runtime.NucleusMiniPresentationEngineId;
          hookRan = true;
        };
        var sw = Stopwatch.StartNew();
        disposed = true;
        rig.Dispose();
        Report($"round {round}: shutdown took {sw.ElapsedMilliseconds} ms, completed {rig.ShutdownCompleted}, "
          + $"error {rig.ShutdownError?.GetType().Name ?? "none"}");
        Assert.True(rig.ShutdownError is null, $"round {round}: the shutdown threw: {rig.ShutdownError}");
        Assert.True(rig.ShutdownCompleted, $"round {round}: the shutdown did not finish within its time limit");
        Assert.True(hookRan, $"round {round}: the shutdown never reached the engine stop");
        Assert.True(mainAtStop is null, $"round {round}: the main engine {mainAtStop} was still attached at the engine stop");
        Assert.True(miniAtStop is null, $"round {round}: the mini engine {miniAtStop} was still attached at the engine stop");
      }
      finally
      {
        if (!disposed)
          rig.Dispose();
      }
    }
  }

  /// <summary>
  /// Proves the export takes the main viewport only: with the mini-view shown, the compact
  /// panel's export button (scripted save dialog) writes a model PNG whose size is the main
  /// viewport's capture size (±2 px) and not the mini-view's (a square of at most 320 DIPs).
  /// </summary>
  [SkippableFact]
  public void NucleusMiniView_ExportUsesMainOnly()
  {
    using var rig = new AppRig();
    var s = Build(rig);
    var mini = CompactX11.RequireCapture(s.Runtime, s.MiniPe, "the mini-view");
    var main = CompactX11.RequireCapture(s.Runtime, s.MainPe, "the main viewport");
    Report($"main capture {main.Width}×{main.Height}, mini capture {mini.Width}×{mini.Height}");
    Assert.NotEqual((main.Width, main.Height), (mini.Width, mini.Height));

    var chosen = Path.Combine(rig.Root, "export", "review_67p_mini_shown.png");
    var (modelPath, _) = ImageExportService.FileNames(chosen);
    rig.Dialogs.Answer(chosen);
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.ExportButton));
    Assert.True(WaitFor(() => File.Exists(modelPath) && CompactX11.Readable(modelPath), 90_000),
      $"{modelPath} not written; status '{Ui(() => rig.Vm.CompactPanel.ExportStatus)}'");
    Assert.True(CompactX11.HasPngSignature(modelPath), $"{modelPath} is not a PNG");
    var rgba = PngCodec.ReadRgba(modelPath, out int w, out int h);
    Report($"export {w}×{h}, lit {CompactX11.LitCount(rgba, w, h)} px; mini-view still shown {Ui(() => s.Mini.IsShown)}");
    Assert.True(Math.Abs(w - main.Width) <= 2 && Math.Abs(h - main.Height) <= 2,
      $"the export is {w}×{h}, the main viewport {main.Width}×{main.Height}");
    Assert.False(w == mini.Width && h == mini.Height, $"the export has the mini-view's size {mini.Width}×{mini.Height}");
    Assert.True(Ui(() => s.Mini.IsShown), "the export hid the mini-view");
  }
}
