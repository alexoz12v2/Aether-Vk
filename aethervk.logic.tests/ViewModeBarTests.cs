using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Reactive.Testing;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The main window's Orbit / Nucleus / Model strip (reviewer items 4 and 20): each button sets the
/// camera mode the viewport overlay reads, the highlight follows mode changes made elsewhere, and
/// Model runs the FITS comparison (model range at the image's date, then set-up comparison) or,
/// without a FITS image, enters the Earth observer with the import hint. The Timeline's range lock
/// is shown with its recommit button, never bypassed.
/// </summary>
[Collection("Sequential")]
public class ViewModeBarTests
{
  private sealed class FakeRegistry : ICameraServiceRegistry
  {
    private readonly List<CameraService> _all = [];
    public readonly Subject<ulong> Created = new();
    public readonly Subject<ulong> Destroyed = new();
    public void RegisterSelf(ulong cameraId, CameraService cameraService) => _all.Add(cameraService);
    public void UnregisterSelf(ulong cameraId) { }
    public CameraService? Get(ulong cameraId) => _all.FirstOrDefault();
    public IReadOnlyList<CameraService> All => _all;
    public IObservable<ulong> ViewportCreated => Created;
    public IObservable<ulong> ViewportDestroyed => Destroyed;
  }

  private static ISchedulerProvider Schedulers(TestScheduler scheduler)
  {
    var sp = new Mock<ISchedulerProvider>();
    sp.Setup(s => s.MainThread).Returns(scheduler);
    sp.Setup(s => s.Background).Returns(scheduler);
    return sp.Object;
  }

  private static ITranslationService Translation()
  {
    var t = new Mock<ITranslationService>();
    t.Setup(x => x.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
    t.Setup(x => x.GetString(It.IsAny<string>())).Returns<string>(k => k);
    return t.Object;
  }

  /// <summary>A camera like <c>CameraServiceTests.BuildService</c>, registered in a fake registry.</summary>
  private sealed class CameraRig
  {
    public readonly TestScheduler Scheduler = new();
    public readonly FakeRegistry Registry = new();
    public readonly CameraService Camera;
    public readonly CometConfigService CometConfig;
    public readonly TimelineService Timeline;
    public readonly ViewModeBarViewModel Bar;
    public readonly Mock<ITabStateService<TimelineSession>> Sessions = new();
    public readonly TimelineSession Session = new();

    public CameraRig()
    {
      var runtime = new Mock<INativeRuntimeService>();
      runtime.Setup(r => r.EarthEntityId).Returns(42UL);
      runtime.Setup(r => r.SetCameraParent(It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<bool>())).Returns(true);
      runtime.Setup(r => r.SetCameraParentToComet(It.IsAny<ulong>(), It.IsAny<bool>())).Returns(true);
      var schedulers = Schedulers(Scheduler);
      var breadcrumb = new BreadcrumbService();
      CometConfig = new CometConfigService(runtime.Object, schedulers);
      Timeline = new TimelineService(runtime.Object, schedulers, CometConfig, breadcrumb);
      var tracker = new CometPositionTrackerService(runtime.Object, schedulers, Timeline);
      Camera = new CameraService(runtime.Object, schedulers, tracker, CometConfig, breadcrumb,
        Mock.Of<ICometMessenger>(), Registry);
      Camera.OnViewportReady(100UL, 800, 600);

      var id = new SessionId(typeof(TimelineSession), 1);
      Sessions.Setup(s => s.ActiveSessionIds).Returns(new List<SessionId> { id });
      Sessions.Setup(s => s.GetSession(id)).Returns(Session);

      Bar = new ViewModeBarViewModel(Translation(), schedulers, Registry, Timeline, Sessions.Object, pollRangeLock: false);
    }

    /// <summary>Delivers the camera's mode notifications (they observe on the main-thread scheduler).</summary>
    public void Pump() => Scheduler.AdvanceBy(TimeSpan.FromSeconds(1).Ticks);

    public void CommitComet()
    {
      const System.Reflection.BindingFlags Private =
        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
      ((BehaviorSubject<bool>)typeof(CometConfigService)
        .GetField("_isCommittedSubject", Private)!.GetValue(CometConfig)!).OnNext(true);
    }
  }

  [Fact]
  public void Buttons_SetTheCameraMode_TheOverlayReads()
  {
    var rig = new CameraRig();
    rig.Pump();
    Assert.Same(rig.Camera, rig.Bar.Camera); // no focused viewport yet: the first registered camera
    Assert.True(rig.Bar.IsOrbitActive);

    rig.CommitComet();
    rig.Bar.SelectNucleusCommand.Execute(null);
    Assert.Equal(CameraMode.CometOrbiting, rig.Camera.CurrentMode);
    rig.Pump();
    Assert.True(rig.Bar.IsNucleusActive);
    Assert.False(rig.Bar.IsOrbitActive);

    rig.Bar.SelectOrbitCommand.Execute(null);
    Assert.Equal(CameraMode.UpZenith, rig.Camera.CurrentMode);
    rig.Pump();
    Assert.True(rig.Bar.IsOrbitActive);
  }

  [Fact]
  public async Task Model_WithoutAFitsImage_EntersTheEarthObserver_AndHints()
  {
    var rig = new CameraRig();
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    Assert.Equal(CameraMode.EarthPosition, rig.Camera.CurrentMode);
    rig.Pump();
    Assert.True(rig.Bar.IsModelActive);
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint); // the localized key, from the mock translation
    Assert.True(rig.Bar.HasHint);
  }

  [Fact]
  public async Task Model_WithAFitsImage_RequestsTheModelRange_ThenSetsUpTheComparison()
  {
    var rig = new CameraRig();
    var fits = new FitsBillboardTests.Rig(withModelRange: true);
    var row = fits.Create("/data/lt_67p.fits");
    fits.Probe.IsEarthObserver = false;
    fits.Probe.ArcsecPerPx = null;
    fits.Vm.Probe();
    using var attached = rig.Bar.AttachBillboards(fits.Vm);

    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    fits.Fake.Drain();

    // the model range first (end = the image's mid-exposure), then the comparison steps in order
    Assert.Equal(["range", "enter", "seek", "site", "match 364.416″", "lock True"], fits.Probe.Log);
    Assert.Equal(DateTimeOffset.Parse("2021-12-27T00:14:47.345Z", System.Globalization.CultureInfo.InvariantCulture),
      Assert.Single(fits.ModelRanges));
    Assert.Equal(((short)0, 693749756529000001UL), Assert.Single(fits.Seeks)); // the image's TDB parts
    Assert.True(fits.Probe.IsComparisonLocked);
    Assert.Equal(row.Fits!.StatusMessage, rig.Bar.Hint);
    Assert.False(rig.Bar.IsBusy);

    attached.Dispose();
    await rig.Bar.SelectModelCommand.ExecuteAsync(null); // detached: the no-FITS path again
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);
  }

  [Fact]
  public void Highlight_FollowsModeChangesMadeElsewhere()
  {
    var rig = new CameraRig();
    rig.Pump();
    Assert.True(rig.Bar.IsOrbitActive);

    // the overlay's Model button / Numpad key: the same CameraService call
    rig.Camera.EnterModelView();
    rig.Pump();
    Assert.True(rig.Bar.IsModelActive);
    Assert.False(rig.Bar.IsOrbitActive);
    Assert.Equal(CameraMode.EarthPosition, rig.Bar.CurrentMode);

    rig.Camera.SetCameraMode(CameraMode.UpZenith);
    rig.Pump();
    Assert.True(rig.Bar.IsOrbitActive);
    Assert.False(rig.Bar.IsModelActive);
  }

  [Fact]
  public void RangeLock_IsShown_AndRecommitGoesThroughTheTimeline()
  {
    var rig = new CameraRig();
    var recommits = 0;
    rig.Timeline.RecommitRequests.Subscribe(_ => recommits++);

    rig.Bar.RefreshRangeLock();
    Assert.False(rig.Bar.IsRangeLocked); // nothing committed

    rig.Session.CommittedStartEpoch = "2021-12-20T00:00:00";
    rig.Session.CommittedEndEpoch = "2021-12-27T00:00:00";
    rig.Session.ProposedStartEpoch = rig.Session.CommittedStartEpoch;
    rig.Session.ProposedEndEpoch = rig.Session.CommittedEndEpoch;
    rig.Bar.RefreshRangeLock();
    Assert.False(rig.Bar.IsRangeLocked); // committed and unchanged

    rig.Bar.RecommitWithRangeCommand.Execute(null);
    rig.Pump();
    Assert.Equal(0, recommits); // not locked: no recommit

    rig.Session.ProposedEndEpoch = "2021-12-27T00:14:47"; // the model range moved the end
    rig.Bar.RefreshRangeLock();
    Assert.True(rig.Bar.IsRangeLocked);
    Assert.Equal(RangeEntryViewModel.RangeLockedText, rig.Bar.RangeLockText);

    rig.Bar.RecommitWithRangeCommand.Execute(null);
    rig.Pump();
    Assert.Equal(1, recommits);
  }
}
