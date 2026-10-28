using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Linq;
using System.Numerics;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// <see cref="AppSaveFileHost"/> over the real sessions and services with a mocked native runtime
/// (the commit callbacks are simulated): commit and capture agree, the model / timeline /
/// billboards are applied into the sessions, New resets everything, and the reuse trigger fires
/// once per burst of changes.
/// </summary>
public sealed class AppSaveFileHostTests
{
  private delegate bool GetAppearance(out CometAppearanceDto appearance);

  private sealed class Rig
  {
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly CometSession Comet = new();
    public readonly TimelineSession Timeline = new();
    public readonly ModelSession Model = new();
    public readonly CometConfigService CometConfig;
    public readonly TimelineService TimelineService;
    public readonly AssetLibraryService Assets;
    public readonly DesktopBillboardService Billboards = new();
    public readonly AppSaveFileHost Host;
    public ImportedAsset[] NativeAssets = [];
    public CometAppearanceDto NativeAppearance;
    private readonly Dictionary<ExternalStateType, Action<nint>> _listeners = [];

    public Rig()
    {
      Runtime
        .Setup(r => r.RegisterExternalStateListener(It.IsAny<ExternalStateType>(), It.IsAny<Action<nint>>()))
        .Callback<ExternalStateType, Action<nint>>((t, a) => _listeners[t] = a)
        .Returns(Mock.Of<IDisposable>());
      Runtime.Setup(r => r.ShutdownToken).Returns(System.Threading.CancellationToken.None);
      Runtime.Setup(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>())).Returns(true);
      Runtime.Setup(r => r.LoadAlmanacFileAsync(It.IsAny<string>())).Returns(() =>
      {
        Fire(ExternalStateType.AlmanacImported, 1, 1000012);
        return Task.FromResult(1ul);
      });
      ulong body;
      Runtime
        .Setup(r => r.TryInitComet(It.IsAny<int>(), It.IsAny<TimeRange>(), It.IsAny<SmallBodyDataComponent>(),
          It.IsAny<ReferenceOrbitMode>(), out body))
        .Returns(() =>
        {
          Fire(ExternalStateType.CometInitialized, 1, 1000012);
          return true;
        });
      Runtime.Setup(r => r.GetAssets()).Returns(() => NativeAssets);
      // a comet with a rotational model (Capture saves the rotation only then)
      Runtime.Setup(r => r.GetCompatibilityJson()).Returns("{\"rotation\":[0.0,90.0,0.0,0.0,0.0,0.0,true]}");
      Runtime.Setup(r => r.GetAssetContentHash(It.IsAny<ulong>())).Returns<ulong>(id => 0xabc0ul + id);
      CometAppearanceDto ignored;
      Runtime.Setup(r => r.GetCometAppearance(out ignored))
        .Returns(new GetAppearance((out CometAppearanceDto a) => { a = NativeAppearance; return true; }));
      Runtime.Setup(r => r.CometEntityId).Returns(7ul);
      ParticleSystemComputedProperties? computed = null;
      ulong firstPs = 101, nextPs = 102;
      Runtime.Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out firstPs))
        .Returns(computed);
      Runtime.Setup(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out nextPs))
        .Returns(true);

      CometConfig = new CometConfigService(Runtime.Object, Schedulers);
      TimelineService = new TimelineService(Runtime.Object, Schedulers, CometConfig, new BreadcrumbService());
      var storage = new Mock<ILocalStorageService>();
      storage.Setup(s => s.GetSessionPath(It.IsAny<string>())).Returns("/tmp/avk_host_tests");
      Assets = new AssetLibraryService(Runtime.Object, Schedulers, storage.Object);
      Host = new AppSaveFileHost(Runtime.Object, Sessions(Comet).Object, Sessions(Timeline).Object, Sessions(Model).Object,
        CometConfig, TimelineService, Assets, Billboards, new CameraServiceRegistry(), new Mock<ICometMessenger>().Object,
        new BreadcrumbService(), Schedulers);
    }

    private void Fire(ExternalStateType type, uint first, int second)
    {
      var p = Marshal.AllocHGlobal(64);
      try
      {
        for (int i = 0; i < 64; i++) Marshal.WriteByte(p, i, 0);
        Marshal.WriteInt32(p, 0, (int)first);
        Marshal.WriteInt32(p, 4, second);
        _listeners[type](p);
      }
      finally
      {
        Marshal.FreeHGlobal(p);
      }
    }

    public static Mock<ITabStateService<T>> Sessions<T>(T session) where T : class, ITabSession, new()
    {
      var m = new Mock<ITabStateService<T>>();
      m.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new(typeof(T), 1) });
      m.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(session);
      m.Setup(s => s.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(session));
      m.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>([]));
      return m;
    }
  }

  private static SpkDto Spk() => new()
  {
    NaifId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko", RecordId = "90000703",
    Sbdb = new SbdbDto { SpkId = 1000012, Designation = "67P", E = 0.64, Q = 1.24, I = 7.04, Om = 50.1, W = 12.8, Tp = 2460000.5 },
    ReferenceOrbitMode = "osculating_at_start", NucleusRadiusKm = 2.2f,
    Rotation = new RotationDto { PoleRaDeg = 69.3, PoleDecDeg = 64.1, RateDegDay = 649.1, BodyFixedOrientation = false },
  };

  [Fact]
  public async Task Commit_SetsTheSessionsAndJets_AndCaptureSavesTheSameEntry()
  {
    var rig = new Rig();
    rig.Host.ApplyModel(new ModelDto { DiametreUm = 150, ManualNucleusRadiusKm = 3.5f },
      [new JetDto { LatitudeDeg = 10, Seed = 11 }, new JetDto { LatitudeDeg = 20, Seed = 22 }]);
    var range = new TimeRange(0, 1_000, 0, 9_000);
    Assert.True(await rig.Host.CommitCometAsync("/save/spk/spk_67P_90000703.bsp", Spk(), range));

    Assert.True(rig.Comet.IsAlmanacLoaded);
    Assert.Equal(("67P", "90000703", "/save/spk/spk_67P_90000703.bsp", 1, 2.2f),
      (rig.Comet.CommittedDesignation, rig.Comet.CommittedSpkRecordId, rig.Comet.CommittedSpkFilePath,
        rig.Comet.ReferenceOrbitIndex, rig.Comet.NucleusRadiusKm));
    Assert.Equal((69.3, 64.1, 649.1, 1), (rig.Comet.RotPoleRaDeg, rig.Comet.RotPoleDecDeg, rig.Comet.RotRateDegDay,
      rig.Comet.RotBodyFixedOrientationIndex));
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(2.2f), Times.Once);
    rig.Runtime.Verify(r => r.SetBodyRotationalModel(7ul, It.Is<BodyRotationalModelDto>(d => d.PoleRaDeg == 69.3)), Times.Once);
    // jets are created natively with their seeds and the manual radius, the first one first
    ulong ps;
    rig.Runtime.Verify(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(),
      It.Is<ParticleSystemJet>(j => j.Seed == 11 && j.NucleusRadiusKm == 3.5f), out ps), Times.Once);
    rig.Runtime.Verify(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(),
      It.Is<ParticleSystemJet>(j => j.Seed == 22), out ps), Times.Once);

    var snap = rig.Host.Capture();
    var spk = snap.Comet!.Spk;
    Assert.Equal("/save/spk/spk_67P_90000703.bsp", snap.Comet.SpkSourcePath);
    Assert.Equal(SaveIndexIo.ToJson(new SaveIndex { Spk = [Spk()] }).Replace("\"coverage\": null", ""),
      SaveIndexIo.ToJson(new SaveIndex { Spk = [Without(spk)] }).Replace("\"coverage\": null", ""));
    Assert.Equal((0, 1_000ul, 0, 9_000ul), (snap.Timeline.CommittedStart!.Centuries, snap.Timeline.CommittedStart.Nanoseconds,
      snap.Timeline.CommittedEnd!.Centuries, snap.Timeline.CommittedEnd.Nanoseconds));
    Assert.Equal([11u, 22u], snap.Jets.Select(j => j.Seed));
    Assert.Equal(150, snap.Model.DiametreUm);
  }

  [Fact]
  public async Task Commit_IsRefused_WhenTheEngineDoesNotTakeTheSavedRange()
  {
    // the engine would keep its previous range and Run to End would land on that stale end
    var rig = new Rig();
    rig.Runtime.Setup(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>())).Returns(false);

    Assert.False(await rig.Host.CommitCometAsync("/save/spk/spk_67P_90000703.bsp", Spk(), new TimeRange(0, 1_000, 0, 9_000)));

    Assert.False(rig.Comet.IsAlmanacLoaded);
    rig.Runtime.Verify(r => r.LoadAlmanacFileAsync(It.IsAny<string>()), Times.Never);
  }

  private static SpkDto Without(SpkDto s) => new()
  {
    NaifId = s.NaifId, Designation = s.Designation, FullName = s.FullName, RecordId = s.RecordId, Sbdb = s.Sbdb,
    ReferenceOrbitMode = s.ReferenceOrbitMode, NucleusRadiusKm = s.NucleusRadiusKm, Rotation = s.Rotation,
  };

  [Fact]
  public void ApplyModelTimelineBillboards_WriteTheSessions_AndRaiseRestored()
  {
    var rig = new Rig();
    int modelRestored = 0, timelineRestored = 0;
    rig.Model.Restored += () => modelRestored++;
    rig.Timeline.Restored += () => timelineRestored++;

    rig.Host.ApplyModel(new ModelDto { SizeQ = 4.25f, DustTracers = true }, [new JetDto { ApertureDeg = 12, Seed = 5 }]);
    Assert.Equal((4.25f, true), (rig.Model.SizeQ, rig.Model.DustTracers));
    var jet = Assert.Single(rig.Model.Jets);
    Assert.Equal((12f, 5u, 1), (jet.ApertureDeg, jet.Seed, jet.DisplayIndex));
    Assert.Equal(1, modelRestored);

    var start = Utils.TimeUtils.ToTaiParts(new DateTimeOffset(2025, 10, 1, 0, 0, 0, TimeSpan.Zero));
    var end = Utils.TimeUtils.ToTaiParts(new DateTimeOffset(2025, 12, 1, 0, 0, 0, TimeSpan.Zero));
    rig.Host.ApplyTimeline(new TimelineDto
    {
      ProposedStart = "2025-10-01T00:00:00Z", ProposedEnd = "2025-11-11T00:00:00Z",
      CommittedStart = new EpochParts(start.centuries, start.nanoseconds), CommittedEnd = new EpochParts(end.centuries, end.nanoseconds),
    }, committed: true);
    Assert.Equal("2025-10-01T00:00:00Z", rig.Timeline.ProposedStartEpoch);
    // the same instant keeps the proposed text (the tab compares strings); another is ISO-8601
    Assert.Equal("2025-10-01T00:00:00Z", rig.Timeline.CommittedStartEpoch);
    Assert.Equal("2025-12-01T00:00:00Z", rig.Timeline.CommittedEndEpoch);
    Assert.Equal(1, timelineRestored);

    var id = Guid.NewGuid();
    rig.Host.ApplyBillboards([new BillboardDto { Id = id, TextureKey = "/t.png", CenterX = 5, WidthPx = 10, HeightPx = 20, Alpha = 0.3 }]);
    var b = Assert.Single(rig.Billboards.Current);
    Assert.Equal((id, "/t.png", 5.0, 0.3), (b.Id, b.TextureKey, b.CenterX, b.Alpha));
  }

  [Fact]
  public async Task Reset_DecommitsAndClearsEverything()
  {
    var rig = new Rig();
    rig.Host.ApplyModel(new ModelDto { DiametreUm = 150 }, [new JetDto { Seed = 1 }]);
    await rig.Host.CommitCometAsync("/s.bsp", Spk(), new TimeRange(0, 1, 0, 2));
    rig.Billboards.Restore([new DesktopBillboard(Guid.NewGuid(), "/t.png", 0, 0, 10, 10)]);
    rig.NativeAssets =
    [
      new ImportedAsset(3, AssetKind.Mesh, "rock", "/rock.glb", 1, 3, [], null, null),
      new ImportedAsset(4, AssetKind.Texture, "sky", "/sky.png", 8, 8, [], null, null),
    ];
    rig.Assets.Refresh();
    int cometRestored = 0;
    rig.Comet.Restored += () => cometRestored++;

    await rig.Host.ResetAsync();
    rig.Runtime.Verify(r => r.ReconfigureComet(0x2, It.IsAny<int>(), out It.Ref<ulong>.IsAny), Times.Once);
    Assert.False(rig.CometConfig.IsAlmanacCommittedValue);
    Assert.False(rig.Comet.IsAlmanacLoaded);
    Assert.Equal((0.0, 90.0, 0f), (rig.Comet.RotPoleRaDeg, rig.Comet.RotPoleDecDeg, rig.Comet.NucleusRadiusKm));
    Assert.Equal(1, cometRestored);
    Assert.Empty(rig.Model.Jets);
    Assert.Equal(new ModelSession().DiametreUm, rig.Model.DiametreUm);
    Assert.Empty(rig.Billboards.Current);
    rig.Runtime.Verify(r => r.RemoveAsset(It.IsAny<ulong>(), 3ul), Times.Once);
    rig.Runtime.Verify(r => r.RemoveAsset(It.IsAny<ulong>(), 4ul), Times.Once);
    rig.Runtime.Verify(r => r.SetCometAppearance(It.Is<CometAppearanceDto>(a => a.Mode == CometDisplayMode.Default && a.Mesh == 0)), Times.Once);
  }

  /// <summary>New with nothing committed still detaches natively: jets added outside the session go.</summary>
  [Fact]
  public async Task Reset_WithoutACommittedComet_StillCleansTheEngineComet()
  {
    var rig = new Rig();
    await rig.Host.ResetAsync();
    rig.Runtime.Verify(r => r.ReconfigureComet(0x2, It.IsAny<int>(), out It.Ref<ulong>.IsAny), Times.Once);
  }

  /// <summary>
  /// A closed Comet tab stops listening: it used to push its own (stale, default) rotation on
  /// every later commit, overwriting the rotation a savefile open had just applied.
  /// </summary>
  [Fact]
  public async Task ClosedCometTab_DoesNotPushItsRotation_OnALaterCommit()
  {
    var rig = new Rig();
    var translation = new Mock<ITranslationService>();
    translation.Setup(t => t.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
    var breadcrumbs = new BreadcrumbService();
    var storage = new Mock<ILocalStorageService>();
    var tab = new CometTabViewModel(translation.Object, rig.Schedulers, Rig.Sessions(rig.Comet).Object,
      new HorizonJplService(null, breadcrumbs, storage.Object), rig.CometConfig, rig.TimelineService, breadcrumbs,
      storage.Object, rig.Runtime.Object, Rig.Sessions(rig.Model).Object, new Mock<ICometMessenger>().Object,
      new Mock<ITabFactory>().Object);
    tab.Dispose();

    var spk = Spk();
    Assert.True(await rig.Host.CommitCometAsync("/s.bsp", spk, new TimeRange(0, 1, 0, 2)));
    rig.Schedulers.MainThread.AdvanceBy(TimeSpan.FromSeconds(2).Ticks);
    rig.Runtime.Verify(r => r.SetBodyRotationalModel(It.IsAny<ulong>(), It.Is<BodyRotationalModelDto>(d => d.PoleRaDeg != 69.3)), Times.Never);
    rig.Runtime.Verify(r => r.SetBodyRotationalModel(7ul, It.Is<BodyRotationalModelDto>(d => d.PoleRaDeg == 69.3)), Times.Once);
  }

  [Fact]
  public void Capture_NamesAssetsByKey_WithTheirContentHashes()
  {
    var rig = new Rig();
    rig.NativeAssets =
    [
      new ImportedAsset(3, AssetKind.Mesh, "rock", "/rock.glb", 1, 3, [5, 0, 0, 0], null, null),
      new ImportedAsset(5, AssetKind.Texture, "rock_albedo", "/rock.glb#image0", 8, 8, [], TextureChannel.Albedo, null),
    ];
    rig.Assets.Refresh();
    rig.NativeAppearance = new CometAppearanceDto { Mesh = 3, Albedo = 5, Roll = 9, Mode = CometDisplayMode.Custom };
    var snap = rig.Host.Capture();
    Assert.Null(snap.Comet);
    Assert.Equal([(3ul, 0xabc3ul), (5ul, 0xabc5ul)], snap.Assets.Select(a => (a.Id, a.ContentHash)));
    Assert.Equal("/rock.glb#image0", snap.Assets[0].BundledTextureKeys!["albedo"]);
    Assert.Equal(("custom", "/rock.glb", "/rock.glb#image0", 9f), (snap.Appearance!.Mode, snap.Appearance.MeshKey,
      snap.Appearance.AlbedoKey, snap.Appearance.YawPitchRollDeg[2]));
  }

  [Fact]
  public async Task ConfigurationChanged_FiresOncePerBurst()
  {
    var rig = new Rig();
    int fired = 0;
    using var sub = rig.Host.ConfigurationChanged.Subscribe(_ => fired++);
    await rig.Host.CommitCometAsync("/s.bsp", Spk(), new TimeRange(0, 1, 0, 2)); // commit + radius
    rig.Model.Jets.Add(new JetViewModel(0, 0, 10, 1, 0.1f, Vector4.One, true, 3));
    rig.Schedulers.Background.AdvanceBy(TimeSpan.FromMilliseconds(500).Ticks);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(0, fired);
    rig.Schedulers.Background.AdvanceBy(TimeSpan.FromSeconds(1).Ticks);
    rig.Schedulers.MainThread.AdvanceBy(1);
    Assert.Equal(1, fired);
  }
}
