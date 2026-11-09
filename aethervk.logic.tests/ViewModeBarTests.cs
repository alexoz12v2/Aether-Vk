using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.Utils;
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

  private static ISchedulerProvider Immediate()
  {
    var sp = new Mock<ISchedulerProvider>();
    sp.Setup(s => s.MainThread).Returns(System.Reactive.Concurrency.Scheduler.Immediate);
    sp.Setup(s => s.Background).Returns(System.Reactive.Concurrency.Scheduler.Immediate);
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

    public readonly Mock<INativeRuntimeService> Runtime = new();

    /// <param name="immediate">Main-thread work delivered at once (the Model chain awaits the Comet tab's answer).</param>
    /// <param name="translation">The bar's strings (default: each key returns itself).</param>
    public CameraRig(bool immediate = false, ITranslationService? translation = null)
    {
      var runtime = Runtime;
      runtime.Setup(r => r.EarthEntityId).Returns(42UL);
      runtime.Setup(r => r.SetCameraParent(It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<bool>())).Returns(true);
      runtime.Setup(r => r.SetCameraParentToComet(It.IsAny<ulong>(), It.IsAny<bool>())).Returns(true);
      var schedulers = immediate ? Immediate() : Schedulers(Scheduler);
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

      Bar = new ViewModeBarViewModel(translation ?? Translation(), schedulers, Registry, Timeline, Sessions.Object, pollRangeLock: false);
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

  /// <summary>
  /// Model with a FITS image: the Comet tab's "Run at this date" chain (reviewer 2026-10-31, slide 1),
  /// one implementation, through the timeline service: N rotations ending at the image's date are
  /// proposed, the comet recommitted, the clock sought to the date, then the comparison set up.
  /// </summary>
  [Fact]
  public async Task Model_UsesTheObservationChain()
  {
    var rig = new CameraRig(immediate: true);
    var fits = new FitsBillboardTests.Rig();
    var row = fits.Create("/data/lt_67p.fits");
    fits.Probe.IsEarthObserver = false;
    fits.Probe.ArcsecPerPx = null;
    fits.Vm.Probe();
    using var attached = rig.Bar.AttachBillboards(fits.Vm);
    rig.Timeline.ObservationImageSource = fits.Vm.ObservationImage; // as the Settings tab wires it

    // the Comet tab's range entry (the chain's owner) and its Download & Commit, faked
    var log = new List<string>();
    rig.Sessions.Setup(s => s.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
      .Callback<SessionId, Action<TimelineSession>>((_, a) => a(rig.Session));
    using var range = new RangeEntryViewModel(rig.Sessions.Object, rig.Timeline, null, () => 696.0,
      new TimeScaleService(new FakeTimeScaleConverter()) { Scale = TimeScale.Utc });
    rig.Timeline.RecommitRequests.Subscribe(r =>
    {
      log.Add("recommit");
      Assert.True(TimeUtils.TryParseIso8601(rig.Session.ProposedEndEpoch, out var end));
      Assert.Equal(new DateTimeOffset(2021, 12, 27, 0, 14, 47, TimeSpan.Zero), end); // the image's mid-exposure
      range.OnCommitted(null);
      r.Complete(true);
    });
    rig.Runtime.Setup(r => r.SeekEpochSync(It.IsAny<short>(), It.IsAny<ulong>()))
      .Returns<short, ulong>((_, _) => { log.Add("seek"); return true; });
    range.ModelRotations = 2;

    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    fits.Fake.Drain();

    Assert.Equal(["recommit", "seek"], log);
    Assert.Equal(["enter", "seek", "site", "match 364.416″", "lock True"], fits.Probe.Log); // the comparison, after the chain's seek
    Assert.Equal("2021-12-27T00:14:47", range.ObservationEpoch);
    Assert.Equal(((short)0, 693749756529000001UL), Assert.Single(fits.Seeks)); // the image's TDB parts
    Assert.True(fits.Probe.IsComparisonLocked);
    Assert.Equal(row.Fits!.StatusMessage, rig.Bar.Hint);
    Assert.False(rig.Bar.IsBusy);

    attached.Dispose();
    await rig.Bar.SelectModelCommand.ExecuteAsync(null); // detached: the no-FITS path again
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);
  }

  /// <summary>The English AppStrings.resx, as the app shows it.</summary>
  private static ITranslationService English() => new TranslationService(
    new System.Resources.ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly),
    new System.Globalization.CultureInfo("en"));

  /// <summary>A FITS image whose date the app cannot read is not "no FITS": the hint says so and
  /// shows the raw date card.</summary>
  [Fact]
  public async Task Model_WithAFitsImageWithoutADate_ShowsTheRawDateCard()
  {
    var rig = new CameraRig(translation: English());
    var fits = new FitsBillboardTests.Rig();
    fits.Fake.Runtime.Setup(r => r.GetFitsMetadataJson(FitsBillboardTests.FitsId)).Returns(FitsBillboardTests.UndatedJson);
    var row = fits.Create("/data/lt_67p.fits");
    Assert.NotNull(row.Fits);
    Assert.Null(row.Fits!.ObservationUtc);
    using var attached = rig.Bar.AttachBillboards(fits.Vm);

    await rig.Bar.SelectModelCommand.ExecuteAsync(null);

    Assert.Equal(CameraMode.EarthPosition, rig.Camera.CurrentMode); // still the Earth observer
    Assert.Equal("The FITS image has no observation date the app can read (DATE-OBS= '27/12/21')", rig.Bar.Hint);
    Assert.DoesNotContain("Import a FITS image", rig.Bar.Hint);
  }

  /// <summary>Without the engine's raw card and without a date value, the keywords tried are named.</summary>
  [Fact]
  public async Task Model_WithAFitsImageWithoutAnyDateCard_NamesTheKeywordsTried()
  {
    var rig = new CameraRig(translation: English());
    var fits = new FitsBillboardTests.Rig();
    var json = FitsBillboardTests.UndatedJson.Replace(", \"dateCardRaw\": \"DATE-OBS= '27/12/21'\"", "");
    fits.Fake.Runtime.Setup(r => r.GetFitsMetadataJson(FitsBillboardTests.FitsId)).Returns(json);
    fits.Create("/data/lt_67p.fits");
    using var attached = rig.Bar.AttachBillboards(fits.Vm);

    await rig.Bar.SelectModelCommand.ExecuteAsync(null);

    Assert.Equal("The FITS image has no observation date the app can read (DATE-OBS / MJD-OBS / JD not found)", rig.Bar.Hint);
  }

  /// <summary>The no-FITS hint goes once a FITS row is added (it no longer applies).</summary>
  [Fact]
  public async Task Hint_IsClearedWhenTheFitsRowsChange()
  {
    var rig = new CameraRig();
    var fits = new FitsBillboardTests.Rig();
    using var attached = rig.Bar.AttachBillboards(fits.Vm);
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);

    var row = fits.Create("/data/lt_67p.fits");

    Assert.NotNull(row.Fits);
    Assert.Equal(string.Empty, rig.Bar.Hint);
    Assert.False(rig.Bar.HasHint);

    // a row removed clears it too
    rig.Bar.Hint = "stale";
    row.DeleteCommand.Execute(null);
    fits.Fake.Drain();
    Assert.Equal(string.Empty, rig.Bar.Hint);
  }

  /// <summary>A FITS row whose metadata read found the library busy: once the retry finds it, the
  /// stale no-FITS hint goes and Model takes the image.</summary>
  [Fact]
  public async Task Hint_IsClearedWhenABusyMetadataReadSucceeds()
  {
    var rig = new CameraRig();
    var fits = new FitsBillboardTests.Rig();
    bool busy = true;
    fits.Fake.Runtime.Setup(r => r.GetFitsMetadataJson(FitsBillboardTests.FitsId))
      .Returns(() => busy ? null : FitsBillboardTests.LtJson);
    using var attached = rig.Bar.AttachBillboards(fits.Vm);
    // the row is created while the library is busy (the main thread is not drained yet)
    fits.Vm.SelectedTexture = fits.Vm.TextureOptions.Single(t => t.Key == "/data/lt_67p.fits");
    fits.Vm.CreateCommand.Execute(null);
    fits.Fake.Schedulers.Background.Start();
    fits.Fake.Schedulers.MainThread.AdvanceBy(1); // the row arrives, its first read is busy
    var row = fits.Vm.Items.Single();
    Assert.Null(row.Fits);
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);

    busy = false;
    fits.Fake.Drain(); // the delayed attempts

    Assert.NotNull(row.Fits);
    Assert.Equal(string.Empty, rig.Bar.Hint);
    Assert.Same(row.Fits, fits.Vm.ObservationImage());
  }

  /// <summary>Leaving the Model view drops its hint.</summary>
  [Fact]
  public async Task Hint_IsClearedWhenTheCameraLeavesTheModelView()
  {
    var rig = new CameraRig();
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    rig.Pump(); // Model's own mode change arrives after the hint: it stays
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);

    rig.Bar.SelectOrbitCommand.Execute(null);
    rig.Pump();

    Assert.True(rig.Bar.IsOrbitActive);
    Assert.Equal(string.Empty, rig.Bar.Hint);
  }

  /// <summary>A language change drops the hint (it was written in the previous language).</summary>
  [Fact]
  public async Task Hint_IsClearedWhenTheLanguageChanges()
  {
    var cultures = new BehaviorSubject<System.Globalization.CultureInfo>(new System.Globalization.CultureInfo("en"));
    var t = new Mock<ITranslationService>();
    t.Setup(x => x.CultureChanged).Returns(cultures);
    t.Setup(x => x.GetString(It.IsAny<string>())).Returns<string>(k => k);
    var rig = new CameraRig(translation: t.Object);
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    rig.Pump();
    Assert.Equal("ModeBar_NoFitsHint", rig.Bar.Hint);

    cultures.OnNext(new System.Globalization.CultureInfo("it"));
    rig.Pump();

    Assert.Equal(string.Empty, rig.Bar.Hint);
  }

  [Fact]
  public async Task Model_WithoutACometTab_ShowsTheChainsAnswer()
  {
    var rig = new CameraRig(immediate: true);
    var fits = new FitsBillboardTests.Rig();
    fits.Create("/data/lt_67p.fits");
    using var attached = rig.Bar.AttachBillboards(fits.Vm);
    await rig.Bar.SelectModelCommand.ExecuteAsync(null);
    Assert.Equal(ObservationRunResult.NoHandler.Status, rig.Bar.Hint);
  }

  /// <summary>The main window's automation-only <c>ViewportDesktopRect</c> (the end-to-end test
  /// reads it there, whichever tab is shown): the attached billboards' rectangle, followed as the
  /// viewport moves, empty once they are detached.</summary>
  [Fact]
  public void ViewportDesktopRect_FollowsTheAttachedBillboards_AndClearsOnDetach()
  {
    var rig = new CameraRig();
    var fits = new FitsBillboardTests.Rig();
    Assert.Equal(string.Empty, rig.Bar.ViewportDesktopRect);
    fits.Probe.ViewportFrame = new ViewportFrame(new PointD(0, 0), 1920, 1080, 1.0);
    fits.Vm.Probe();
    var attached = rig.Bar.AttachBillboards(fits.Vm);
    Assert.Equal("0.000,0.000,1920.000,1080.000", rig.Bar.ViewportDesktopRect); // read at once

    // the window moved to a monitor left of the primary
    fits.Probe.ViewportFrame = new ViewportFrame(new PointD(-1920 + 534, 130), 1386, 950.5, 1.0);
    fits.Vm.Probe();
    Assert.Equal("-1386.000,130.000,1386.000,950.500", rig.Bar.ViewportDesktopRect);

    attached.Dispose();
    Assert.Equal(string.Empty, rig.Bar.ViewportDesktopRect);
    fits.Probe.ViewportFrame = new ViewportFrame(new PointD(10, 20), 800, 600, 1.0);
    fits.Vm.Probe();
    Assert.Equal(string.Empty, rig.Bar.ViewportDesktopRect); // no longer followed
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
