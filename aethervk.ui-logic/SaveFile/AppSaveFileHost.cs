using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using System.Reactive;
using System.Reactive.Linq;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.SaveFile;

/// <summary>
/// <see cref="ISaveFileHost"/> over the running application: the exclusive Comet, Timeline and
/// Model sessions, the comet and timeline services, the asset library, the desktop billboards,
/// the viewport cameras and the native runtime. Live tabs follow through the sessions'
/// <c>Restored</c> events. Runs on the UI thread.
/// </summary>
public sealed class AppSaveFileHost : ISaveFileHost
{
  /// <summary>How long a native dump / restore may take before it counts as failed.</summary>
  public static readonly TimeSpan NativeTimeout = TimeSpan.FromSeconds(60);

  private readonly INativeRuntimeService _runtime;
  private readonly ITabStateService<CometSession> _cometSessions;
  private readonly ITabStateService<TimelineSession> _timelineSessions;
  private readonly ITabStateService<ModelSession> _modelSessions;
  private readonly CometConfigService _cometConfig;
  private readonly TimelineService _timeline;
  private readonly AssetLibraryService _assets;
  private readonly DesktopBillboardService _billboards;
  private readonly ICameraServiceRegistry _cameras;
  private readonly ICometMessenger _messenger;
  private readonly BreadcrumbService _breadcrumbs;
  // asset ids are never reused: their content hashes are fetched once
  private readonly ConcurrentDictionary<ulong, ulong> _hashes = new();

  public AppSaveFileHost(
    INativeRuntimeService runtime,
    ITabStateService<CometSession> cometSessions,
    ITabStateService<TimelineSession> timelineSessions,
    ITabStateService<ModelSession> modelSessions,
    CometConfigService cometConfig,
    TimelineService timeline,
    AssetLibraryService assets,
    DesktopBillboardService billboards,
    ICameraServiceRegistry cameras,
    ICometMessenger messenger,
    BreadcrumbService breadcrumbs,
    ISchedulerProvider schedulers)
  {
    _runtime = runtime;
    _cometSessions = cometSessions;
    _timelineSessions = timelineSessions;
    _modelSessions = modelSessions;
    _cometConfig = cometConfig;
    _timeline = timeline;
    _assets = assets;
    _billboards = billboards;
    _cameras = cameras;
    _messenger = messenger;
    _breadcrumbs = breadcrumbs;
    var radius = cometConfig.NucleusRadiusKm.DistinctUntilChanged().Skip(1).Select(_ => Unit.Default);
    var committed = cometConfig.IsAlmanacCommitted.Where(c => c).Select(_ => Unit.Default);
    var jets = Observable
      .FromEventPattern<System.Collections.Specialized.NotifyCollectionChangedEventHandler,
        System.Collections.Specialized.NotifyCollectionChangedEventArgs>(
        h => Model.Jets.CollectionChanged += h, h => Model.Jets.CollectionChanged -= h)
      .Select(_ => Unit.Default);
    ConfigurationChanged = radius.Merge(committed).Merge(jets)
      .Throttle(TimeSpan.FromSeconds(1), schedulers.Background)
      .ObserveOn(schedulers.MainThread)
      .Publish()
      .RefCount();
  }

  public IObservable<Unit> ConfigurationChanged { get; }

  private static T Session<T>(ITabStateService<T> service) where T : class, ITabSession, new() =>
    service.ActiveSessionIds.Count > 0 ? service.GetSession(service.ActiveSessionIds[0]) : new T();

  private CometSession Comet => Session(_cometSessions);
  private TimelineSession Timeline => Session(_timelineSessions);
  private ModelSession Model => Session(_modelSessions);

  // ── save ───────────────────────────────────────────────────────────────────

  public SaveSnapshot Capture()
  {
    var comet = Comet;
    var timeline = Timeline;
    var model = Model;
    var inputs = _cometConfig.IsAlmanacCommittedValue ? _cometConfig.CommittedInputs : null;

    var tl = new TimelineDto
    {
      ProposedStart = NullIfEmpty(timeline.ProposedStartEpoch),
      ProposedEnd = NullIfEmpty(timeline.ProposedEndEpoch),
      Speed = timeline.Speed,
    };
    LiveComet? live = null;
    if (inputs is not null)
    {
      var (start, end) = SaveIndexMapping.ToParts(inputs.Range);
      tl.CommittedStart = start;
      tl.CommittedEnd = end;
      if (_runtime.GetSimulationClock(out var c, out var ns))
        tl.CurrentEpoch = new EpochParts(c, ns);
      tl.StartPaused = tl.CurrentEpoch is { } now && (now.Centuries != start.Centuries || now.Nanoseconds != start.Nanoseconds);
      live = new LiveComet(inputs.SpkFilePath, new SpkDto
      {
        NaifId = inputs.NaifId,
        Designation = comet.CommittedDesignation,
        FullName = comet.CommittedFullName,
        RecordId = comet.CommittedSpkRecordId,
        Coverage = new CoverageDto { Start = start, End = end },
        Sbdb = SaveIndexMapping.ToDto(inputs.SmallBody),
        ReferenceOrbitMode = SaveIndexMapping.Snake(inputs.ReferenceMode),
        NucleusRadiusKm = _cometConfig.NucleusRadiusKmValue > 0 ? _cometConfig.NucleusRadiusKmValue : comet.NucleusRadiusKm,
        // only a rotation the engine has: a comet never given one has none ("rotation":null), and
        // pushing the session's defaults on open would change the configuration
        Rotation = HasRotationalModel(_runtime.GetCompatibilityJson()) ? SaveIndexMapping.RotationOf(comet) : null,
      });
    }

    var assets = _assets.CurrentMeshes.Concat(_assets.CurrentTextures).ToList();
    string? KeyOf(ulong id) => id == 0 ? null : assets.FirstOrDefault(a => a.Id == id)?.Key;
    var liveAssets = assets.Select(a => ToLive(a, KeyOf)).ToList();

    CometAppearanceIndexDto? appearance = _runtime.GetCometAppearance(out var dto)
      ? SaveIndexMapping.ToDto(dto, KeyOf)
      : null;

    return new SaveSnapshot(
      tl,
      live,
      SaveIndexMapping.ToDto(model),
      [.. model.Jets.Select(SaveIndexMapping.ToDto)],
      [.. _cameras.All.Select(c => SaveIndexMapping.ToDto(c.GetPersistedState()))],
      liveAssets,
      appearance,
      [.. _billboards.Current.Select(SaveIndexMapping.ToDto)]);
  }

  private LiveAsset ToLive(ImportedAsset a, Func<ulong, string?> keyOf)
  {
    Dictionary<string, string?>? bundled = null;
    if (a.Kind == AssetKind.Mesh && !a.BundledTextures.IsDefaultOrEmpty)
    {
      bundled = [];
      foreach (TextureChannel ch in Enum.GetValues(typeof(TextureChannel)))
        if (a.BundledTexture(ch) is var id and not 0)
          bundled[SaveIndexMapping.Snake(ch)] = keyOf(id);
    }
    return new LiveAsset(a.Id, a.Kind, a.Key, a.Label, HashOf(a.Id), a.ChannelHint, bundled);
  }

  private ulong HashOf(ulong id)
  {
    if (_hashes.TryGetValue(id, out var h)) return h;
    if (_runtime.GetAssetContentHash(id) is { } fresh)
    {
      _hashes[id] = fresh;
      return fresh;
    }
    return 0;
  }

  /// <summary>Whether the comet has a rotational model, per the engine's compatibility JSON.</summary>
  internal static bool HasRotationalModel(string? compatibilityJson) =>
    compatibilityJson is not null
    && compatibilityJson.Contains("\"rotation\":")
    && !compatibilityJson.Contains("\"rotation\":null");

  private static string? NullIfEmpty(string s) => string.IsNullOrEmpty(s) ? null : s;

  public string? GetCompatibilityJson() => _runtime.GetCompatibilityJson();

  public uint SceneDumpVersion => _runtime.SceneDumpVersion;

  public Task<bool> DumpSceneAsync(string file) =>
    AwaitNative(ExternalStateType.SceneDumped, () => _runtime.DumpSceneToFile(file));

  public ExportedAssetFile? ExportAsset(ulong assetId, string destDir) => _runtime.ExportAsset(assetId, destDir);

  /// <summary>Runs <paramref name="start"/> and completes with the success flag of the next <paramref name="state"/> callback.</summary>
  private async Task<bool> AwaitNative(ExternalStateType state, Func<bool> start)
  {
    var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    using var listener = _runtime.RegisterExternalStateListener(state, ptr =>
    {
      unsafe { tcs.TrySetResult(ptr != 0 && *(uint*)ptr != 0); }
    });
    if (!start()) return false;
    using var timeout = new CancellationTokenSource(NativeTimeout);
    using (timeout.Token.Register(() => tcs.TrySetResult(false)))
      return await tcs.Task;
  }

  // ── open ───────────────────────────────────────────────────────────────────

  public async Task ResetAsync()
  {
    // pause invariant, as the Comet tab's decommit: back to the start before the comet goes
    if (_timeline.IsSimulationRunningValue || _cometConfig.IsAlmanacCommittedValue)
      _timeline.Reset();

    // the comet, as the Comet tab decommits it
    var comet = Comet;
    if (_cometConfig.IsAlmanacCommittedValue)
    {
      _cometConfig.DecommitComet();
      _messenger.Send(new CometDecommittedMessage());
    }
    else
    {
      // nothing committed, but the engine's comet may still carry jets added outside the Model
      // session: the native detach (CleanupComet) removes them too, so New leaves a clean scene
      _runtime.ReconfigureComet(0x2, 0, out _);
    }
    var fresh = new CometSession();
    comet.SpkId = null;
    comet.CommittedSpkRecordId = fresh.CommittedSpkRecordId;
    comet.CommittedDesignation = fresh.CommittedDesignation;
    comet.CommittedFullName = fresh.CommittedFullName;
    comet.CommittedSpkFilePath = null;
    comet.IsAlmanacLoaded = false;
    comet.ReferenceOrbitIndex = fresh.ReferenceOrbitIndex;
    comet.ProposedDesignation = fresh.ProposedDesignation;
    comet.ProposedSpkFilePath = null;
    comet.NucleusRadiusKm = fresh.NucleusRadiusKm;
    SaveIndexMapping.Apply(SaveIndexMapping.RotationOf(fresh), comet);
    comet.NotifyRestored();

    var model = Model;
    SaveIndexMapping.Apply(SaveIndexMapping.ToDto(new ModelSession()), model);
    model.Jets.Clear();
    model.NotifyRestored();

    var timeline = Timeline;
    var t0 = new TimelineSession();
    timeline.CommittedStartEpoch = t0.CommittedStartEpoch;
    timeline.CommittedEndEpoch = t0.CommittedEndEpoch;
    timeline.CurrentEpochString = t0.CurrentEpochString;
    timeline.SnapshotStartEpoch = t0.SnapshotStartEpoch;
    timeline.SnapshotEndEpoch = t0.SnapshotEndEpoch;
    timeline.NotifyRestored();

    _billboards.Restore([]);
    var defaultAppearance = new CometAppearanceDto { Mode = CometDisplayMode.Default };
    _runtime.SetCometAppearance(defaultAppearance);
    // meshes first: their bundled textures may only be unloadable after them
    foreach (var a in _assets.CurrentMeshes.Concat(_assets.CurrentTextures).ToList())
      await _assets.RemoveAsync(a.Id);
    _assets.NotifyAppearanceChanged(defaultAppearance);

    foreach (var camera in _cameras.All)
      camera.ResetToModeDefault();
  }

  public async Task<bool> ImportAssetAsync(string path) => (await _assets.ImportAsync(path)).Success;

  public IReadOnlyList<LiveAsset> ListAssets()
  {
    _assets.Refresh();
    var all = _assets.CurrentMeshes.Concat(_assets.CurrentTextures).ToList();
    string? KeyOf(ulong id) => all.FirstOrDefault(a => a.Id == id)?.Key;
    return [.. all.Select(a => ToLive(a, KeyOf))];
  }

  public void ApplyModel(ModelDto model, IReadOnlyList<JetDto> jets)
  {
    var session = Model;
    SaveIndexMapping.Apply(model, session);
    session.Jets.Clear();
    // a jet of an older save has no grain of its own: it inherits the model's
    var modelGrain = session.DefaultGrain;
    for (int i = 0; i < jets.Count; i++)
    {
      var jet = SaveIndexMapping.FromDto(jets[i], modelGrain);
      jet.DisplayIndex = i + 1;
      session.Jets.Add(jet);
    }
    session.NotifyRestored();
  }

  public void ApplyTimeline(TimelineDto timeline, bool committed)
  {
    var session = Timeline;
    if (timeline.ProposedStart is { } ps) session.ProposedStartEpoch = ps;
    if (timeline.ProposedEnd is { } pe) session.ProposedEndEpoch = pe;
    if (timeline.Speed is { } speed) session.Speed = speed;
    if (committed && timeline.CommittedStart is { } cs && timeline.CommittedEnd is { } ce)
    {
      session.CommittedStartEpoch = EpochText(cs, session.ProposedStartEpoch);
      session.CommittedEndEpoch = EpochText(ce, session.ProposedEndEpoch);
    }
    if (TimeUtils.TryParseIso8601(session.ProposedStartEpoch, out var s)
        && TimeUtils.TryParseIso8601(session.ProposedEndEpoch, out var e))
    {
      var a = TimeUtils.ToTaiParts(s);
      var b = TimeUtils.ToTaiParts(e);
      _timeline.ProposeEpochRange(new TimeRange(a.centuries, a.nanoseconds, b.centuries, b.nanoseconds));
    }
    session.NotifyRestored();
  }

  /// <summary>
  /// The session's text for a committed epoch: the proposed text when it names the same instant
  /// (the usual case: the tab compares the two strings), else ISO-8601 UTC.
  /// </summary>
  internal static string EpochText(EpochParts epoch, string proposed) =>
    TimeUtils.EpochText(epoch.Centuries, epoch.Nanoseconds, proposed);

  public async Task<bool> CommitCometAsync(string spkPath, SpkDto spk, TimeRange range)
  {
    var mode = SaveIndexMapping.ParseSnake(spk.ReferenceOrbitMode, ReferenceOrbitMode.Sbdb);
    var sb = SaveIndexMapping.FromDto(spk.Sbdb ?? new SbdbDto { SpkId = spk.NaifId });
    // the saved range must reach the engine first (Run to End seeks its end); ResetAsync decommitted
    if (!_timeline.RequestEpochRange(range))
      return false;
    if (!await _cometConfig.CommitCometAsync(spkPath, spk.NaifId, range, sb, mode))
      return false;

    // as the Comet tab after a successful commit
    var comet = Comet;
    comet.SpkId = spk.NaifId;
    comet.CommittedSpkRecordId = spk.RecordId;
    comet.CommittedDesignation = spk.Designation;
    comet.CommittedFullName = spk.FullName;
    comet.CommittedSpkFilePath = spkPath;
    comet.IsAlmanacLoaded = true;
    comet.ReferenceOrbitIndex = (int)mode;
    comet.NucleusRadiusKm = spk.NucleusRadiusKm;
    if (spk.Rotation is { } rot)
    {
      SaveIndexMapping.Apply(rot, comet);
      _cometConfig.SetRotationalModel(SaveIndexMapping.ToNative(rot));
    }
    if (spk.NucleusRadiusKm > 0)
      _cometConfig.SetNucleusRadiusKm(spk.NucleusRadiusKm);
    var model = Model;
    JetNativeSync.EnsureNative(_runtime, model, JetNativeSync.EffectiveNucleusRadiusKm(model, comet));
    _messenger.Send(new CometCommittedMessage(range));
    comet.NotifyRestored();
    return true;
  }

  public bool ApplyAppearance(CometAppearanceDto appearance)
  {
    var status = _runtime.SetCometAppearance(appearance);
    if (status is not (CometAppearanceStatus.Queued or CometAppearanceStatus.NotAvailable))
      return false;
    _assets.NotifyAppearanceChanged(appearance);
    return true;
  }

  /// <summary>
  /// Polls the compatibility JSON until two reads 50 ms apart agree (at most 2 s): the radius
  /// update goes through the logic thread.
  /// </summary>
  public async Task SettleAsync()
  {
    var previous = _runtime.GetCompatibilityJson();
    for (int i = 0; i < 40; i++)
    {
      await Task.Delay(50);
      var now = _runtime.GetCompatibilityJson();
      if (string.Equals(now, previous, StringComparison.Ordinal)) return;
      previous = now;
    }
  }

  public Task<bool> RestoreSceneAsync(string file) =>
    AwaitNative(ExternalStateType.SceneRestored, () => _runtime.RestoreSceneFromFile(file));

  public Task<bool> SeekAsync(EpochParts epoch) =>
    Task.FromResult(_runtime.SeekEpochSync(epoch.Centuries, epoch.Nanoseconds));

  public void ApplyViewports(IReadOnlyList<ViewportDto> viewports)
  {
    var cameras = _cameras.All;
    for (int i = 0; i < Math.Min(cameras.Count, viewports.Count); i++)
      cameras[i].RestorePersistedState(SaveIndexMapping.FromDto(viewports[i]));
  }

  public void ApplyBillboards(IReadOnlyList<BillboardDto> billboards) =>
    _billboards.Restore(billboards.Select(SaveIndexMapping.FromDto));

  public void SyncClock()
  {
    var session = Timeline;
    if (_runtime.GetSimulationClock(out var c, out var ns))
      session.CurrentEpochString = ViewModels.TimelineTabViewModel.ClockLabel(c, ns, session.CommittedStartEpoch);
    // a live tab re-reads the session and polls the clock (its progress)
    session.NotifyRestored();
  }

  /// <summary>Past a camera mode switch (2.5 s) with margin: the restored pose has been reached.</summary>
  internal static readonly TimeSpan DustFeedbackSettle = TimeSpan.FromSeconds(3);

  public void RestartDustFeedbackWhenSettled()
  {
    // the engine already restarted it at the restore's seek; this second restart drops what the
    // frames drawn during the cameras' flight to their saved poses taught it
    _ = Task.Delay(DustFeedbackSettle).ContinueWith(_ => _runtime.ResetDustViewFeedback(), TaskScheduler.Default);
  }

  public bool IsSimulationRunning => _timeline.IsSimulationRunningValue;

  public void Notify(string title, string message, bool error) =>
    _ = _breadcrumbs.ShowMessageAsync(title, message, TimeSpan.FromSeconds(error ? 6 : 3), status: error ? 3 : 1);
}
