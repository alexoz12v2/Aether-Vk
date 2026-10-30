using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Billboards keep their place against the viewport, not the desktop: a save made with the app on
/// one monitor reopens over the viewport on another, older saves are brought over it as a group,
/// free billboards keep their offset from the viewport's top-left corner through moves and
/// resizes, comparison billboards sit on the comet whenever it is in view (and after an open, once
/// the restored view is confirmed), a drop off the comet frees one, and offsets convert between
/// screens of different scaling.
/// </summary>
public class BillboardViewportPlacementTests
{
  private static readonly ViewportFrame ScreenA = new(new PointD(0, 0), 1920, 1080, 1.0);
  private static readonly ViewportFrame ScreenB = new(new PointD(2560, 0), 1920, 1080, 1.0);

  private static DesktopBillboard Bb(double cx, double cy) =>
    new(Guid.NewGuid(), "/data/tail.png", cx, cy, 300, 200) { ImageWidthPx = 300, ImageHeightPx = 200 };

  /// <summary>A billboard bound to its physical size (1″ per image px on a 1″/px viewport: its native size).</summary>
  private static DesktopBillboard Bound(double cx, double cy) =>
    Bb(cx, cy) with { PhysicalSizeBound = true, ImageArcsecPerPx = 1.0 };

  /// <summary>A billboard with a nucleus pixel 49.5 px left of and above the image's centre (149.5, 99.5).</summary>
  private static DesktopBillboard WithNucleus(double cx, double cy) =>
    Bb(cx, cy) with { ImageNucleusPx = new PointD(100, 50) };

  private static void Near(PointD want, PointD got, double tol = 1e-9) =>
    Assert.True(Math.Abs(want.X - got.X) <= tol && Math.Abs(want.Y - got.Y) <= tol, $"expected {want}, got {got}");

  private static PointD Center(DesktopBillboardService svc, Guid id) => DesktopBillboardGeometry.Center(svc.Find(id)!);

  private static PointD Nucleus(DesktopBillboardService svc, Guid id) => DesktopBillboardGeometry.NucleusDesktop(svc.Find(id)!);

  /// <summary>Saves <paramref name="billboards"/> against <paramref name="viewport"/> and reads them
  /// back as a savefile open does (<see cref="SaveIndexMapping"/> both ways).</summary>
  private static List<(DesktopBillboard, ViewportOffset?)> SaveAndOpen(ViewportFrame? viewport, params DesktopBillboard[] billboards)
  {
    using var saving = new DesktopBillboardService();
    if (viewport is { } v)
      saving.FollowViewport(v);
    saving.Restore(billboards);
    var dtos = saving.Current.Select(b => SaveIndexMapping.ToDto(b, saving.ViewportOffsetOf(b))).ToList();
    return dtos.Select(d => (SaveIndexMapping.FromDto(d), SaveIndexMapping.ViewportOffsetOf(d))).ToList();
  }

  // ── save and open across monitors ──

  [Fact]
  public void SavedWithTheViewportOnScreenA_OpensOverItOnScreenB()
  {
    var b = Bb(800, 500);
    var opened = SaveAndOpen(ScreenA, b);
    Assert.Equal(new ViewportOffset(800, 500, 1.0), opened[0].Item2);

    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenB);
    svc.RestoreOverViewport(opened);
    Near(new PointD(2560 + 800, 500), Center(svc, b.Id));
  }

  [Fact]
  public void OpenedBeforeTheViewportIsLaidOut_IsPlacedAtItsFirstReport()
  {
    var b = Bb(800, 500);
    var opened = SaveAndOpen(ScreenA, b);
    using var svc = new DesktopBillboardService();
    svc.RestoreOverViewport(opened);
    Near(new PointD(800, 500), Center(svc, b.Id)); // nothing to place it on yet
    // a save made meanwhile keeps the offset it is waiting with
    Assert.Equal(new ViewportOffset(800, 500, 1.0), svc.ViewportOffsetOf(svc.Find(b.Id)!));
    svc.FollowViewport(ScreenB);
    Near(new PointD(3360, 500), Center(svc, b.Id));
    // the first report is a placement, not a move: a later one is
    svc.FollowViewport(ScreenB with { Origin = new PointD(2600, 50) });
    Near(new PointD(3400, 550), Center(svc, b.Id));
  }

  [Fact]
  public void OlderSave_OffTheViewport_IsBroughtOverIt_WithItsLayoutKept()
  {
    var a = Bb(100, 100);
    var c = Bb(400, 300);
    var opened = SaveAndOpen(null, a, c); // no viewport known at save: desktop positions only
    Assert.All(opened, x => Assert.Null(x.Item2));

    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenB);
    svc.RestoreOverViewport(opened);
    // the pair's centroid (250, 200) on the viewport's centre (3520, 540): the layout translates
    Near(new PointD(3370, 440), Center(svc, a.Id));
    Near(new PointD(3670, 640), Center(svc, c.Id));
    Near(Center(svc, c.Id) - Center(svc, a.Id), new PointD(300, 200));
  }

  [Fact]
  public void OlderSave_PartlyOverTheViewport_MovesAsAGroup()
  {
    var a = Bb(3000, 500); // on the viewport
    var c = Bb(2400, 500); // just left of it: never left behind alone
    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenB);
    svc.RestoreOverViewport([(a, null), (c, null)]);
    // the pair's centroid (2700, 500) on the viewport's centre (3520, 540): both on it now
    Near(new PointD(3820, 540), Center(svc, a.Id));
    Near(new PointD(3220, 540), Center(svc, c.Id));
    Assert.True(ScreenB.Contains(Center(svc, a.Id)) && ScreenB.Contains(Center(svc, c.Id)));
  }

  [Fact]
  public void OlderSave_AllOverTheViewport_IsUnchanged()
  {
    var a = Bb(3000, 500);
    var c = Bb(4000, 900);
    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenB);
    svc.RestoreOverViewport([(a, null), (c, null)]);
    Near(new PointD(3000, 500), Center(svc, a.Id));
    Near(new PointD(4000, 900), Center(svc, c.Id));
  }

  // ── comparison billboards after an open ──

  [Fact]
  public async Task Reopen_WithTheSameView_RecentresWithoutWaiting()
  {
    var rig = new VmRig();
    rig.Probe.RestoredView.SetResult(true); // the saved view is the one shown: confirmed at once
    rig.Probe.Comet = new PointD(1000, 600);
    var (bound, nucleus, plain) = (Bound(800, 500), WithNucleus(300, 300), Bb(1200, 700));
    rig.Svc.RestoreOverViewport([(bound, null), (nucleus, null), (plain, null)]);
    rig.Fake.Drain();
    Assert.NotNull(rig.Vm.RestoreRecenter);
    await rig.Vm.RestoreRecenter!;
    Assert.False(rig.Vm.RestoreRecenterTimedOut);
    Near(new PointD(1000, 600), Nucleus(rig.Svc, bound.Id));
    Near(new PointD(1000, 600), Nucleus(rig.Svc, nucleus.Id));
    Near(new PointD(1049.5, 649.5), Center(rig.Svc, nucleus.Id)); // its nucleus pixel, not its centre
    Near(new PointD(1200, 700), Center(rig.Svc, plain.Id)); // not a comparison billboard
  }

  [Fact]
  public async Task Reopen_WithAnotherView_RecentresAfterTheRestoredViewIsConfirmed()
  {
    var rig = new VmRig();
    rig.Probe.Comet = new PointD(900, 540); // the view before the open, still drawn
    var (bound, plain) = (Bound(800, 500), Bb(1200, 700));
    rig.Svc.RestoreOverViewport([(bound, null), (plain, null)]);
    rig.Fake.Drain();
    Assert.False(rig.Vm.RestoreRecenter!.IsCompleted);
    // the flight to the saved pose: the comet is followed frame by frame meanwhile
    rig.Probe.Comet = new PointD(950, 560);
    rig.Probe.ViewChangedNow();
    Near(new PointD(950, 560), Nucleus(rig.Svc, bound.Id));
    rig.Probe.Comet = new PointD(1000, 600);
    rig.Probe.RestoredView.SetResult(true);
    await rig.Vm.RestoreRecenter;
    Assert.False(rig.Vm.RestoreRecenterTimedOut);
    Near(new PointD(1000, 600), Nucleus(rig.Svc, bound.Id));
    Near(new PointD(1200, 700), Center(rig.Svc, plain.Id));
  }

  [Fact]
  public async Task Reopen_WithoutConfirmation_RecentresAfterTheBoundedWait()
  {
    var rig = new VmRig();
    rig.Probe.Comet = new PointD(1000, 600);
    var bound = Bound(800, 500);
    rig.Svc.Restore([bound]);
    await rig.Vm.RecenterRestoredOnCometAsync(TimeSpan.FromMilliseconds(1)); // never confirmed
    Assert.True(rig.Vm.RestoreRecenterTimedOut);
    Near(new PointD(1000, 600), Nucleus(rig.Svc, bound.Id)); // on whatever the viewport shows
  }

  // ── the comet anchor ──

  [Fact]
  public void AnchoredBillboard_NucleusStaysOnTheComet_AsTheCometMoves()
  {
    var rig = new VmRig();
    var (bound, nucleus, plain) = (Bound(800, 500), WithNucleus(300, 300), Bb(1200, 700));
    rig.Svc.Restore([bound, nucleus, plain]);
    PointD[] path = [new(900, 540), new(910, 530), new(950, 500), new(1200, 400), new(400, 800)];
    for (int i = 0; i < path.Length; i++)
    {
      rig.Probe.Comet = path[i];
      if (i % 2 == 0)
        rig.Vm.Probe(); // the probe's tick
      else
        rig.Probe.ViewChangedNow(); // an orbit's confirmed frame between ticks
      Near(path[i], Nucleus(rig.Svc, bound.Id));
      Near(path[i], Nucleus(rig.Svc, nucleus.Id));
      Near(path[i] + new PointD(49.5, 49.5), Center(rig.Svc, nucleus.Id));
      Near(new PointD(1200, 700), Center(rig.Svc, plain.Id));
    }
    Assert.True(rig.Svc.IsAnchored(bound.Id) && rig.Svc.IsAnchored(nucleus.Id) && !rig.Svc.IsAnchored(plain.Id));
  }

  [Fact]
  public void AnchoredBillboard_OutOfViewComet_FollowsTheCorner()
  {
    var rig = new VmRig();
    var (bound, plain) = (Bound(800, 500), Bb(1200, 700));
    rig.Svc.Restore([bound, plain]);
    rig.Probe.Comet = null; // the comet left the viewport
    rig.Vm.Probe();
    Near(new PointD(800, 500), Center(rig.Svc, bound.Id));
    rig.Probe.ViewportFrame = ScreenA with { Origin = new PointD(300, 200) };
    rig.Probe.Move();
    Near(new PointD(1100, 700), Center(rig.Svc, bound.Id)); // like the free one
    Near(new PointD(1500, 900), Center(rig.Svc, plain.Id));
    // back in view: on the comet again
    rig.Probe.Comet = new PointD(1000, 600);
    rig.Probe.ViewChangedNow();
    Near(new PointD(1000, 600), Nucleus(rig.Svc, bound.Id));
    Near(new PointD(1500, 900), Center(rig.Svc, plain.Id));
  }

  [Fact]
  public void DroppedOffTheComet_ReleasesTheAnchor_CenterOnCometRestoresIt()
  {
    var rig = new VmRig();
    var bound = Bound(800, 500);
    rig.Svc.Restore([bound]);
    rig.Probe.Comet = new PointD(900, 540);
    rig.Vm.Probe();
    Near(new PointD(900, 540), Center(rig.Svc, bound.Id));

    // the user drags it away: pinned while dragging, free once dropped
    rig.Svc.SetDragging(bound.Id, true);
    rig.Svc.Update(bound.Id, b => b with { CenterX = 1300, CenterY = 800 });
    rig.Probe.ViewChangedNow();
    Near(new PointD(1300, 800), Center(rig.Svc, bound.Id));
    rig.Svc.SetDragging(bound.Id, false);
    Assert.False(rig.Svc.IsAnchored(bound.Id));
    rig.Probe.Comet = new PointD(950, 560);
    rig.Vm.Probe();
    Near(new PointD(1300, 800), Center(rig.Svc, bound.Id)); // never overridden
    Assert.True(rig.Svc.Find(bound.Id)!.PhysicalSizeBound); // the size binding stays

    // "Center on comet" anchors it again
    Assert.True(rig.Vm.CenterOnComet(bound.Id));
    Assert.True(rig.Svc.IsAnchored(bound.Id));
    rig.Probe.Comet = new PointD(1000, 600);
    rig.Probe.ViewChangedNow();
    Near(new PointD(1000, 600), Center(rig.Svc, bound.Id));

    // dropped back within reach of the comet: it snaps on and stays anchored
    rig.Svc.SetDragging(bound.Id, true);
    rig.Svc.Update(bound.Id, b => b with { CenterX = 1002, CenterY = 601 });
    rig.Svc.SetDragging(bound.Id, false);
    Assert.True(rig.Svc.IsAnchored(bound.Id));
    Near(new PointD(1000, 600), Center(rig.Svc, bound.Id));

    // a turn about its centre is not a move: still anchored
    rig.Svc.SetDragging(bound.Id, true);
    rig.Svc.Update(bound.Id, b => b with { RotationDeg = 30 });
    rig.Svc.SetDragging(bound.Id, false);
    Assert.True(rig.Svc.IsAnchored(bound.Id));

    // a position typed into the row is the user's too
    rig.Svc.Update(bound.Id, b => b with { CenterX = 500 });
    Assert.False(rig.Svc.IsAnchored(bound.Id));
    rig.Vm.Probe();
    Near(new PointD(500, 600), Center(rig.Svc, bound.Id));
  }

  // ── the window moving mid-session ──

  [Fact]
  public void WindowMove_TranslatesTheBillboards_ButNotOneBeingDragged()
  {
    var rig = new VmRig();
    var a = Bb(800, 500);
    var dragged = Bb(1200, 700);
    rig.Svc.Restore([a, dragged]);
    rig.Svc.SetDragging(dragged.Id, true);
    rig.Probe.ViewportFrame = ScreenA with { Origin = new PointD(300, 200) };
    rig.Probe.Move(); // the control's position report, not the probe timer
    Near(new PointD(1100, 700), Center(rig.Svc, a.Id));
    Near(new PointD(1200, 700), Center(rig.Svc, dragged.Id));
    rig.Svc.SetDragging(dragged.Id, false);
    rig.Probe.ViewportFrame = ScreenB;
    rig.Probe.Move();
    Near(new PointD(3360, 500), Center(rig.Svc, a.Id));
    Near(new PointD(3460, 500), Center(rig.Svc, dragged.Id));
  }

  [Fact]
  public void FreeBillboard_KeepsItsCornerOffset_WhenTheViewportMovesOrResizes()
  {
    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenA);
    var b = Bb(400, 300);
    svc.Restore([b]);
    svc.FollowViewport(ScreenA with { WidthPx = 1200, HeightPx = 700 }); // a docked panel opened
    Near(new PointD(400, 300), Center(svc, b.Id)); // the same distance from the corner
    svc.FollowViewport(new ViewportFrame(new PointD(100, 50), 1200, 700, 1.0)); // then dragged
    Near(new PointD(500, 350), Center(svc, b.Id));
    svc.FollowViewport(new ViewportFrame(new PointD(2600, 80), 1920, 1080, 1.0)); // maximised elsewhere
    Near(new PointD(3000, 380), Center(svc, b.Id));
  }

  // ── mixed DPI ──

  [Fact]
  public void SavedAt100Percent_OpenedAt150Percent_ScalesTheOffset()
  {
    var b = Bb(400, 300);
    var opened = SaveAndOpen(ScreenA, b);
    using var svc = new DesktopBillboardService();
    var hiDpi = new ViewportFrame(new PointD(2560, 0), 2880, 1620, 1.5);
    svc.FollowViewport(hiDpi);
    svc.RestoreOverViewport(opened);
    Near(new PointD(2560 + 600, 450), Center(svc, b.Id)); // the same distance in DIPs

    // and back: a save at 150 % opened at 100 %
    var reopened = SaveAndOpen(hiDpi, svc.Find(b.Id)!);
    Assert.Equal(new ViewportOffset(600, 450, 1.5), reopened[0].Item2);
    using var low = new DesktopBillboardService();
    low.FollowViewport(ScreenA);
    low.RestoreOverViewport(reopened);
    Near(new PointD(400, 300), Center(low, b.Id));
  }

  [Fact]
  public void FreeBillboard_On150PercentMonitor_ScalesItsCornerOffset()
  {
    using var svc = new DesktopBillboardService();
    svc.FollowViewport(ScreenA);
    var b = Bb(1060, 640); // 1060 px right of and 640 px below the corner
    svc.Restore([b]);
    svc.FollowViewport(new ViewportFrame(new PointD(2560, 0), 2880, 1620, 1.5)); // the window moved there
    Near(new PointD(2560 + 1590, 960), Center(svc, b.Id)); // 1.5× as far in px: the same DIPs
  }

  // ── the save format ──

  [Fact]
  public void Dto_RoundTripsTheViewportOffset_AndOlderSavesHaveNone()
  {
    var b = Bb(1, 2);
    var dto = SaveIndexMapping.ToDto(b, new ViewportOffset(640, 360, 1.25));
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [dto] });
    Assert.Contains("\"viewport_offset_x\": 640", json);
    Assert.Contains("\"viewport_offset_y\": 360", json);
    Assert.Contains("\"viewport_scaling\": 1.25", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));
    Assert.Equal(new ViewportOffset(640, 360, 1.25), SaveIndexMapping.ViewportOffsetOf(back.Billboards[0]));

    Assert.Null(SaveIndexMapping.ViewportOffsetOf(SaveIndexMapping.ToDto(b)));
    var old = new BillboardDto { Id = b.Id, TextureKey = "/a.png", CenterX = 1, CenterY = 2 };
    Assert.Null(SaveIndexMapping.ViewportOffsetOf(old));
    // one coordinate alone is no offset; a missing scaling is 100 %
    Assert.Null(SaveIndexMapping.ViewportOffsetOf(new BillboardDto { ViewportOffsetX = 3 }));
    Assert.Equal(new ViewportOffset(3, 4, 1.0), SaveIndexMapping.ViewportOffsetOf(new BillboardDto { ViewportOffsetX = 3, ViewportOffsetY = 4 }));
  }

  // ── rig ──

  /// <summary>A viewport whose comet position, place on the desktop, confirmed frames and
  /// restored-view signal the test drives.</summary>
  private sealed class FakeProbe : IBillboardViewportProbe
  {
    /// <summary>The comet on the desktop (null: out of view).</summary>
    public PointD? Comet = new(900, 540);
    public double? ArcsecPerPx => 1.0;
    public PointD? CometDesktopPx => Comet;
    public double? CometDistanceKm => null;
    public double ViewportHeightPx => 1080;
    public double? VerticalFovRad => null;
    public bool IsOrthographic => false;
    public bool MatchField(double heightKm, double angleRad) => false;
    public bool IsEarthObserver => true;
    public bool EnterEarthObserver() => true;
    public void SetObserverSite(double latDeg, double lonDeg, double? elevM) { }
    public bool LockToDirection(Vector3d worldDirection) => true;
    public bool ViewUpIsEclipticSouth => false;
    public bool ViewUpIsCelestialNorth => true;
    public PointD? ViewportCenterDesktopPx => ViewportFrame?.Center;
    public ViewportFrame? ViewportFrame { get; set; } = ScreenA;
    public event Action? ViewportMoved;
    public event Action? ViewChanged;
    public readonly TaskCompletionSource<bool> RestoredView = new(TaskCreationOptions.RunContinuationsAsynchronously);
    public Task RestoredViewConfirmed => RestoredView.Task;
    public bool IsComparisonLocked { get; set; }

    /// <summary>What the control reports when the window moves.</summary>
    public void Move() => ViewportMoved?.Invoke();

    /// <summary>The runtime confirmed a camera transform (one frame of an orbit or a flight).</summary>
    public void ViewChangedNow() => ViewChanged?.Invoke();
  }

  private sealed class OneScreen : IScreenInfoProvider
  {
    private static readonly ScreenInfo S = new(new RectI(0, 0, 1920, 1080), 1, true);
    public IReadOnlyList<ScreenInfo> Screens => [S];
    public ScreenInfo MainWindowScreen => S;
  }

  private sealed class VmRig
  {
    public readonly FakeAssetRuntime Fake = new();
    public readonly DesktopBillboardService Svc = new();
    public readonly DesktopBillboardsViewModel Vm;
    public readonly FakeProbe Probe = new();

    public VmRig()
    {
      Fake.NativeAssets.Add(new ImportedAsset(3, AssetKind.Texture, "tail", "/data/tail.png", 300, 200, [], null, null) { IsDisplayable = true });
      var lib = Fake.CreateService();
      lib.Refresh();
      Vm = new DesktopBillboardsViewModel(Svc, lib, new OneScreen(), Fake.Schedulers);
      Fake.Drain();
      Vm.AttachViewport(Probe, poll: false);
    }
  }
}
