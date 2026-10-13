using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Concurrency;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Microsoft.Reactive.Testing;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Desktop billboards, logic layer (plan B-3 … B-7): geometry, pointer rules, the service, the
/// activation aggregator and the Settings view models.
/// </summary>
public class DesktopBillboardTests
{
  private const double Eps = 1e-9;

  private static DesktopBillboard Bb(double cx = 500, double cy = 400, double w = 200, double h = 100, double rot = 0) =>
    new(Guid.NewGuid(), "/data/tail.png", cx, cy, w, h, 0.5, rot);

  private static void Near(double want, double got, double tol = 1e-6) =>
    Assert.True(Math.Abs(want - got) <= tol, $"expected {want}, got {got}");

  private static void Near(PointD want, PointD got, double tol = 1e-6)
  {
    Near(want.X, got.X, tol);
    Near(want.Y, got.Y, tol);
  }

  // ── B-3 geometry ────────────────────────────────────────────────────────────────────────────

  [Theory]
  [InlineData(0)]
  [InlineData(30)]
  [InlineData(45)]
  [InlineData(90)]
  [InlineData(180)]
  public void WindowRect_IsTheRotatedImageBoxPlusTheMargin(double rot)
  {
    var b = Bb(rot: rot);
    var r = DesktopBillboardGeometry.WindowRect(b);
    double rad = rot * Math.PI / 180;
    double halfW = (Math.Abs(Math.Cos(rad)) * b.WidthPx + Math.Abs(Math.Sin(rad)) * b.HeightPx) / 2;
    double halfH = (Math.Abs(Math.Sin(rad)) * b.WidthPx + Math.Abs(Math.Cos(rad)) * b.HeightPx) / 2;
    double m = DesktopBillboardGeometry.GizmoMargin;
    // rounded outward by at most one pixel on each side
    Assert.InRange(r.X, b.CenterX - halfW - m - 1, b.CenterX - halfW - m);
    Assert.InRange(r.Y, b.CenterY - halfH - m - 1, b.CenterY - halfH - m);
    Assert.InRange(r.X + r.Width, b.CenterX + halfW + m, b.CenterX + halfW + m + 1);
    Assert.InRange(r.Y + r.Height, b.CenterY + halfH + m, b.CenterY + halfH + m + 1);
    // every handle of every mode is inside the window
    foreach (var mode in Enum.GetValues<BillboardGizmoMode>())
      foreach (var hs in DesktopBillboardGeometry.Handles(b, mode))
        Assert.True(r.Contains(hs.Center.X, hs.Center.Y), $"{mode} {hs.Kind} at {rot}°");
  }

  [Fact]
  public void Rotation_IsClockwiseOnScreen()
  {
    // 90° clockwise turns screen right (+x) into screen down (+y, y grows downwards)
    Near(new PointD(0, 1), DesktopBillboardGeometry.Rotate(new PointD(1, 0), 90));
    var b = Bb(rot: 90);
    var tl = DesktopBillboardGeometry.Corners(b)[0];
    // the top-left corner (−100, −50) lands at (+50, −100) from the centre
    Near(new PointD(550, 300), tl);
    Near(new PointD(-100, -50), DesktopBillboardGeometry.ToLocal(b, tl));
  }

  private static readonly ScreenInfo[] ThreeScreens =
  [
    new(new RectI(-1280, 0, 1280, 1024), 1.0),
    new(new RectI(0, 0, 2560, 1440), 1.5, IsPrimary: true),
    new(new RectI(2560, -200, 3840, 2160), 2.0),
  ];

  [Theory]
  [InlineData(-640, 512, 0)]
  [InlineData(10, 10, 1)]
  [InlineData(2559, 1439, 1)]
  [InlineData(2560, -200, 2)]
  [InlineData(6399, 1959, 2)]
  public void DesktopPoints_RoundTripThroughTheirDisplay(double x, double y, int display)
  {
    var (d, local) = DesktopBillboardGeometry.ToDisplayLocal(ThreeScreens, new PointD(x, y));
    Assert.Equal(display, d);
    Near(new PointD(x, y), DesktopBillboardGeometry.FromDisplayLocal(ThreeScreens, d, local));
    Assert.True(local.X >= 0 && local.Y >= 0);
  }

  [Fact]
  public void PointsOffEveryDisplay_AreReportedAndBillboardsComeBackToThePrimary()
  {
    Assert.Equal(-1, DesktopBillboardGeometry.DisplayOf(ThreeScreens, new PointD(-2000, 0)));
    Assert.Equal(-1, DesktopBillboardGeometry.DisplayOf(ThreeScreens, new PointD(100, 1500)));
    var lost = Bb(cx: 9000, cy: 9000);
    var back = DesktopBillboardGeometry.KeepOnScreen(lost, ThreeScreens);
    Near(new PointD(1280, 720), DesktopBillboardGeometry.Center(back));
    var fine = Bb(cx: 3000, cy: 100);
    Assert.Same(fine, DesktopBillboardGeometry.KeepOnScreen(fine, ThreeScreens));
  }

  [Fact]
  public void HandleHitTest_FindsEveryHandleOfEveryMode_AndNothingElsewhere()
  {
    foreach (var rot in new[] { 0.0, 37.0, -120.0 })
    {
      var b = Bb(rot: rot);
      foreach (var mode in Enum.GetValues<BillboardGizmoMode>())
      {
        foreach (var hs in DesktopBillboardGeometry.Handles(b, mode))
        {
          Assert.Equal(hs.Kind, DesktopBillboardGeometry.HitTest(b, mode, hs.Center));
          Assert.Equal(hs.Kind, DesktopBillboardGeometry.HitTest(b, mode, hs.Center + new PointD(hs.Radius, 0)));
        }
        // a point on the image away from every handle
        var p = DesktopBillboardGeometry.ToDesktop(b, new PointD(-50, 25));
        Assert.Null(DesktopBillboardGeometry.HitTest(b, mode, p));
        Assert.True(DesktopBillboardGeometry.IsOnImage(b, p));
      }
      Assert.Empty(DesktopBillboardGeometry.Handles(b, BillboardGizmoMode.None));
    }
  }

  [Fact]
  public void RotateDrag_FollowsThePointerAngle_AndSnapsTo15Degrees()
  {
    var b = Bb();
    var c = DesktopBillboardGeometry.Center(b);
    var start = c + new PointD(100, 0);
    var at37 = c + DesktopBillboardGeometry.Rotate(new PointD(100, 0), 37);
    Near(37, DesktopBillboardGeometry.RotateDrag(b, start, at37, snap: false).RotationDeg);
    Near(30, DesktopBillboardGeometry.RotateDrag(b, start, at37, snap: true).RotationDeg);
    var at53 = c + DesktopBillboardGeometry.Rotate(new PointD(100, 0), 53);
    Near(60, DesktopBillboardGeometry.RotateDrag(b, start, at53, snap: true).RotationDeg);
    // wraps into (−180, 180]
    var b170 = b with { RotationDeg = 170 };
    var at30 = c + DesktopBillboardGeometry.Rotate(new PointD(100, 0), 30);
    Near(-160, DesktopBillboardGeometry.RotateDrag(b170, start, at30, snap: false).RotationDeg);
  }

  [Theory]
  [InlineData(BillboardHandle.ScaleBottomRight, 0)]
  [InlineData(BillboardHandle.ScaleTopLeft, 0)]
  [InlineData(BillboardHandle.ScaleTopRight, 33)]
  [InlineData(BillboardHandle.ScaleBottomLeft, -70)]
  public void ScaleDrag_MovesTheCornerAndKeepsTheOppositeOneFixed(BillboardHandle corner, double rot)
  {
    var b = Bb(rot: rot);
    int ci = corner switch
    {
      BillboardHandle.ScaleTopLeft => 0,
      BillboardHandle.ScaleTopRight => 1,
      BillboardHandle.ScaleBottomRight => 2,
      _ => 3,
    };
    var cs0 = DesktopBillboardGeometry.Corners(b);
    var outward = DesktopBillboardGeometry.Rotate(
      new PointD(ci is 1 or 2 ? 40 : -40, ci >= 2 ? 30 : -30), rot);
    var pointer = cs0[ci] + outward;
    var s = DesktopBillboardGeometry.ScaleDrag(b, corner, pointer, keepAspect: false);
    var cs1 = DesktopBillboardGeometry.Corners(s);
    Near(cs0[(ci + 2) % 4], cs1[(ci + 2) % 4], 1e-6);
    Near(pointer, cs1[ci], 1e-6);
    Near(240, s.WidthPx);
    Near(130, s.HeightPx);

    var a = DesktopBillboardGeometry.ScaleDrag(b, corner, pointer, keepAspect: true);
    Near(b.WidthPx / b.HeightPx, a.WidthPx / a.HeightPx);
    Near(1.3, a.HeightPx / b.HeightPx); // the larger relative change (130/100 > 240/200) wins
    Near(cs0[(ci + 2) % 4], DesktopBillboardGeometry.Corners(a)[(ci + 2) % 4], 1e-6);
  }

  [Fact]
  public void ScaleDrag_NeverGoesBelowTheMinimumSide()
  {
    var b = Bb();
    var s = DesktopBillboardGeometry.ScaleDrag(b, BillboardHandle.ScaleBottomRight, new PointD(0, 0), keepAspect: false);
    Near(DesktopBillboardGeometry.MinSidePx, s.WidthPx);
    Near(DesktopBillboardGeometry.MinSidePx, s.HeightPx);
    var a = DesktopBillboardGeometry.ScaleDrag(b, BillboardHandle.ScaleBottomRight, new PointD(0, 0), keepAspect: true);
    Assert.True(a.HeightPx >= DesktopBillboardGeometry.MinSidePx - Eps);
    Near(2, a.WidthPx / a.HeightPx);
  }

  [Fact]
  public void TranslateAlong_ProjectsTheDragOnTheBillboardAxis()
  {
    var b = Bb(rot: 90);
    // the billboard's x axis points screen-down at 90°: a diagonal drag keeps only its y part
    var t = DesktopBillboardGeometry.TranslateAlong(b, new PointD(30, 20), alongX: true);
    Near(new PointD(500, 420), DesktopBillboardGeometry.Center(t));
    var free = DesktopBillboardGeometry.Translate(b, new PointD(30, 20));
    Near(new PointD(530, 420), DesktopBillboardGeometry.Center(free));
  }

  [Fact]
  public void InitialSize_IsNativeShrunkToHalfTheDisplay()
  {
    var screen = new ScreenInfo(new RectI(0, 0, 1920, 1080), 1);
    Assert.Equal((300.0, 200.0), DesktopBillboardGeometry.InitialSize(300, 200, screen));
    var (w, h) = DesktopBillboardGeometry.InitialSize(4000, 2000, screen);
    Near(960, w);
    Near(480, h);
  }

  [Theory]
  [InlineData(0, BillboardGizmoMode.None)]
  [InlineData(25, BillboardGizmoMode.Rotate)]
  [InlineData(-60, BillboardGizmoMode.Scale)]
  public void InputRegion_CoversTheImageAndHandles_AndLeavesTheMarginOpen(double rot, BillboardGizmoMode mode)
  {
    var b = Bb(rot: rot);
    var win = DesktopBillboardGeometry.WindowRect(b);
    var rects = DesktopBillboardGeometry.InputRegion(b, mode);
    bool Covered(PointD desktop) =>
      rects.Any(r => r.Contains(desktop.X - win.X, desktop.Y - win.Y));
    // every image pixel centre and every handle is covered
    for (double ly = -b.HeightPx / 2 + 0.5; ly < b.HeightPx / 2; ly += 3)
      for (double lx = -b.WidthPx / 2 + 0.5; lx < b.WidthPx / 2; lx += 3)
        Assert.True(Covered(DesktopBillboardGeometry.ToDesktop(b, new PointD(lx, ly))), $"image ({lx}, {ly})");
    foreach (var hs in DesktopBillboardGeometry.Handles(b, mode))
      Assert.True(Covered(hs.Center), hs.Kind.ToString());
    // the window corners (margin) are open
    Assert.False(Covered(new PointD(win.X + 1, win.Y + 1)));
    Assert.False(Covered(new PointD(win.X + win.Width - 2, win.Y + win.Height - 2)));
    // far less than the whole window
    double area = rects.Sum(r => (double)r.Width * r.Height);
    Assert.True(area < 0.8 * win.Width * win.Height);
  }

  // ── B-4 pointer rules ───────────────────────────────────────────────────────────────────────

  [Fact]
  public void LeftDrag_MovesTheBillboardInEveryGizmoMode()
  {
    foreach (var mode in Enum.GetValues<BillboardGizmoMode>())
    {
      var b = Bb() with { GizmoMode = mode };
      var c = new BillboardInteractionController();
      var p0 = new PointD(470, 410); // on the image, off the handles
      Assert.Null(c.PointerPressed(b, BillboardPointerButton.Left, p0));
      var moved = c.PointerMoved(p0 + new PointD(25, -15), false, false);
      Near(new PointD(525, 385), DesktopBillboardGeometry.Center(moved!));
      var done = c.PointerReleased(BillboardPointerButton.Left, p0 + new PointD(30, -20), false, false);
      Near(new PointD(530, 380), DesktopBillboardGeometry.Center(done!));
      Assert.Equal(mode, done!.GizmoMode);
      Assert.False(c.IsDragging);
    }
  }

  [Fact]
  public void RightClick_CyclesTheGizmo()
  {
    var b = Bb();
    var c = new BillboardInteractionController();
    var seen = new List<BillboardGizmoMode>();
    for (int i = 0; i < 4; i++)
    {
      // a point on the image away from every handle in every mode
      var p = new PointD(440, 430);
      c.PointerPressed(b, BillboardPointerButton.Right, p);
      b = c.PointerReleased(BillboardPointerButton.Right, p + new PointD(2, 1), false, false)!;
      seen.Add(b.GizmoMode);
    }
    Assert.Equal(
      [BillboardGizmoMode.Translate, BillboardGizmoMode.Rotate, BillboardGizmoMode.Scale, BillboardGizmoMode.None],
      seen);
  }

  [Fact]
  public void RightDragOffHandles_DoesNothing()
  {
    var b = Bb() with { GizmoMode = BillboardGizmoMode.Rotate };
    var c = new BillboardInteractionController();
    var p = new PointD(440, 430);
    c.PointerPressed(b, BillboardPointerButton.Right, p);
    Assert.Null(c.PointerMoved(p + new PointD(20, 0), false, false));
    Assert.Null(c.PointerReleased(BillboardPointerButton.Right, p + new PointD(20, 0), false, false));
  }

  [Fact]
  public void RightDragOnHandles_EditsThroughThem_WithSnapAndAspect()
  {
    // rotate knob, Ctrl snaps
    var b = Bb() with { GizmoMode = BillboardGizmoMode.Rotate };
    var knob = DesktopBillboardGeometry.Handles(b, BillboardGizmoMode.Rotate)[0].Center;
    var c = new BillboardInteractionController();
    c.PointerPressed(b, BillboardPointerButton.Right, knob);
    var to = DesktopBillboardGeometry.Center(b) + DesktopBillboardGeometry.Rotate(knob - DesktopBillboardGeometry.Center(b), 20);
    var r = c.PointerReleased(BillboardPointerButton.Right, to, ctrl: true, shift: false);
    Near(15, r!.RotationDeg);
    Assert.Equal(BillboardGizmoMode.Rotate, r.GizmoMode); // a handle drag never cycles

    // scale corner, Shift keeps the aspect
    var s = Bb() with { GizmoMode = BillboardGizmoMode.Scale };
    var br = DesktopBillboardGeometry.Corners(s)[2];
    c.PointerPressed(s, BillboardPointerButton.Right, br);
    var scaled = c.PointerReleased(BillboardPointerButton.Right, br + new PointD(100, 10), ctrl: false, shift: true);
    Near(2.0, scaled!.WidthPx / scaled.HeightPx);
    Near(300, scaled.WidthPx);

    // translate centre handle moves freely, a short right press on it does not cycle
    var t = Bb() with { GizmoMode = BillboardGizmoMode.Translate };
    var centre = DesktopBillboardGeometry.Center(t);
    c.PointerPressed(t, BillboardPointerButton.Right, centre);
    var tiny = c.PointerReleased(BillboardPointerButton.Right, centre + new PointD(1, 0), false, false);
    Assert.Equal(BillboardGizmoMode.Translate, tiny!.GizmoMode);
    Near(centre + new PointD(1, 0), DesktopBillboardGeometry.Center(tiny));
  }

  [Fact]
  public void PressesDuringADrag_AreIgnored_AndCancelEndsTheDrag()
  {
    var b = Bb();
    var c = new BillboardInteractionController();
    c.PointerPressed(b, BillboardPointerButton.Left, new PointD(470, 410));
    Assert.Null(c.PointerPressed(b, BillboardPointerButton.Right, new PointD(470, 410)));
    c.Cancel();
    Assert.False(c.IsDragging);
    Assert.Null(c.PointerMoved(new PointD(0, 0), false, false));
  }

  // ── B-5 service ─────────────────────────────────────────────────────────────────────────────

  private static ImportedAsset Tex(string key = "/data/tail.png", ulong w = 4000, ulong h = 2000) =>
    new(7, AssetKind.Texture, "tail", key, w, h, [], null, null) { IsDisplayable = true };

  [Fact]
  public void Create_PlacesTheBillboardAtTheDisplayCentre_FittingHalfOfIt()
  {
    using var svc = new DesktopBillboardService();
    var screen = new ScreenInfo(new RectI(1920, 0, 2560, 1440), 1);
    var b = svc.Create(Tex(), screen);
    Near(new PointD(3200, 720), DesktopBillboardGeometry.Center(b));
    Near(1280, b.WidthPx);
    Near(640, b.HeightPx);
    Assert.Equal("/data/tail.png", b.TextureKey);
    Assert.Single(svc.Current);
    Assert.Throws<ArgumentException>(() => svc.Create(Tex() with { Kind = AssetKind.Mesh }, screen));
  }

  [Fact]
  public void UpdateAndRemove_PublishTheList_AndSanitize()
  {
    using var svc = new DesktopBillboardService();
    var emissions = new List<int>();
    using var sub = svc.Billboards.Subscribe(l => emissions.Add(l.Length));
    var b = svc.Create(Tex(), new ScreenInfo(new RectI(0, 0, 1920, 1080), 1));
    Assert.True(svc.Update(b.Id, x => x with { Alpha = 3, WidthPx = 1, RotationDeg = 540 }));
    var u = svc.Find(b.Id)!;
    Near(1, u.Alpha);
    Near(DesktopBillboardGeometry.MinSidePx, u.WidthPx);
    Near(180, u.RotationDeg);
    int before = emissions.Count;
    Assert.True(svc.Update(u)); // unchanged: no emission
    Assert.Equal(before, emissions.Count);
    Assert.False(svc.Update(Bb())); // unknown id
    Assert.True(svc.Remove(b.Id));
    Assert.False(svc.Remove(b.Id));
    Assert.Equal([0, 1, 1, 0], emissions);
  }

  [Fact]
  public void Restore_ReplacesTheList_WithGizmosHidden_AndKeepOnScreenRecoversLostOnes()
  {
    using var svc = new DesktopBillboardService();
    var a = Bb(cx: 9000, cy: 9000) with { GizmoMode = BillboardGizmoMode.Scale };
    var c = Bb(cx: 100, cy: 100);
    svc.Restore([a, c]);
    Assert.All(svc.Current, x => Assert.Equal(BillboardGizmoMode.None, x.GizmoMode));
    svc.KeepOnScreen(ThreeScreens);
    Near(new PointD(1280, 720), DesktopBillboardGeometry.Center(svc.Find(a.Id)!));
    Near(new PointD(100, 100), DesktopBillboardGeometry.Center(svc.Find(c.Id)!));
  }

  // ── B-6 activation ──────────────────────────────────────────────────────────────────────────

  [Fact]
  public void FocusHopsBetweenAppWindows_DoNotDeactivateTheApp()
  {
    var sched = new TestScheduler();
    using var agg = new AppActivationAggregator(sched, initiallyActive: false);
    var seen = new List<bool>();
    using var sub = agg.IsAppActive.Subscribe(seen.Add);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(200).Ticks);
    object main = new(), billboard = new();
    agg.Activated(main);
    sched.AdvanceBy(1);
    // main → billboard → main inside the debounce window
    agg.Deactivated(main);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(50).Ticks);
    agg.Activated(billboard);
    agg.Deactivated(billboard);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(50).Ticks);
    agg.Activated(main);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(500).Ticks);
    Assert.Equal([false, true], seen);
    // the app really goes to the background: one false after the debounce
    agg.Deactivated(main);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(149).Ticks);
    Assert.Equal([false, true], seen);
    sched.AdvanceBy(TimeSpan.FromMilliseconds(2).Ticks);
    Assert.Equal([false, true, false], seen);
    // an active window that closes stops keeping the app active
    agg.Activated(billboard);
    sched.AdvanceBy(1);
    agg.Closed(billboard);
    sched.AdvanceBy(TimeSpan.FromSeconds(1).Ticks);
    Assert.Equal([false, true, false, true, false], seen);
  }

  // ── B-7 view models ─────────────────────────────────────────────────────────────────────────

  private sealed class FakeScreens(params ScreenInfo[] s) : IScreenInfoProvider
  {
    public IReadOnlyList<ScreenInfo> Screens => s;
    public ScreenInfo MainWindowScreen => s[0];
  }

  private static (DesktopBillboardsViewModel vm, DesktopBillboardService svc, FakeAssetRuntime fake, AssetLibraryService lib) BuildVm()
  {
    var fake = new FakeAssetRuntime();
    fake.NativeAssets.Add(Tex("/data/tail.png", 800, 400));
    fake.NativeAssets.Add(new ImportedAsset(8, AssetKind.Texture, "bc7", "/data/bc7.ktx2", 64, 64, [], null, null)
    { IsDisplayable = false });
    var lib = fake.CreateService();
    lib.Refresh();
    var svc = new DesktopBillboardService();
    var vm = new DesktopBillboardsViewModel(svc, lib, new FakeScreens(ThreeScreens[1], ThreeScreens[0]), fake.Schedulers);
    fake.Drain();
    return (vm, svc, fake, lib);
  }

  [Fact]
  public void TexturePicker_ListsDisplayableTexturesOnly_AndCreateAddsARow()
  {
    var (vm, svc, fake, _) = BuildVm();
    Assert.Equal(["tail"], vm.TextureOptions.Select(t => t.Label));
    Assert.True(vm.HasTextures);
    Assert.True(vm.CreateCommand.CanExecute(null));
    vm.CreateCommand.Execute(null);
    fake.Drain();
    var row = Assert.Single(vm.Items);
    Near(1280, row.X);
    Near(720, row.Y);
    Near(800, row.Width);
    Assert.Equal("Display 1 · 1280, 720", row.DisplayLabel);
    Assert.Equal("tail", row.TextureLabel);
    Assert.False(row.IsMissing);
    Assert.Single(svc.Current);
  }

  [Fact]
  public void RowEdits_ReachTheService_AndWindowDrags_ReachTheRow()
  {
    var (vm, svc, fake, _) = BuildVm();
    vm.CreateCommand.Execute(null);
    fake.Drain();
    var row = vm.Items[0];
    row.X = 100;
    row.Alpha = 0.25;
    row.Rotation = 45;
    var b = svc.Current[0];
    Near(100, b.CenterX);
    Near(0.25, b.Alpha);
    Near(45, b.RotationDeg);
    // aspect lock: width drives height
    row.Width = 400;
    Near(200, svc.Current[0].HeightPx);
    fake.Drain();
    Near(200, row.Height);
    row.IsAspectLocked = false;
    row.Height = 50;
    Near(400, svc.Current[0].WidthPx);
    // a drag of the window (service edit) shows up in the row, on the display it landed on
    svc.Update(b.Id, x => x with { CenterX = -640, CenterY = 10 });
    fake.Drain();
    Near(-640, row.X);
    Assert.Equal("Display 2 · 640, 10", row.DisplayLabel);
    row.IsVisible = false;
    Assert.False(svc.Current[0].Visible);
    row.DeleteCommand.Execute(null);
    fake.Drain();
    Assert.Empty(vm.Items);
    Assert.Empty(svc.Current);
  }

  [Fact]
  public void UnloadedTexture_MarksTheRowMissing_AndReimportRestoresIt()
  {
    var (vm, svc, fake, lib) = BuildVm();
    vm.CreateCommand.Execute(null);
    fake.Drain();
    var row = vm.Items[0];
    fake.NativeAssets.RemoveAll(a => a.Key == "/data/tail.png");
    lib.Refresh();
    fake.Drain();
    Assert.True(row.IsMissing);
    Assert.Single(svc.Current); // the billboard itself stays
    Assert.False(vm.CreateCommand.CanExecute(null));
    fake.NativeAssets.Add(Tex("/data/tail.png", 800, 400) with { Id = 9 });
    lib.Refresh();
    fake.Drain();
    Assert.False(row.IsMissing);
  }
}
