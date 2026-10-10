using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(
  keyPrefix:    "Tabs_Model_",
  designTitle:  "Model",
  designIcon:   "⬡")]
public partial class ModelTabViewModel
  : StatefulTabViewModelBase<ModelSession>,
    IModelTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly INativeRuntimeService _runtimeService;
  private readonly ITabStateService<CometSession> _cometSessionService;
  private readonly CometConfigService _cometConfigService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly IUiThreadDispatcher _dispatcher;
  private readonly IPlatformWindowService _platformWindowService;
  private readonly CompositeDisposable _disposables = [];

  // Tracks the active session model-change subscription so it can be replaced
  // when the user switches sessions (SerialDisposable disposes the old one first).
  private readonly SerialDisposable _modelChangeSub = new();

  public string? StrPM { get; set; } = "PM";

  // Subject used to debounce manual nucleus radius edits. Throttle(250 ms) prevents
  // flooding the native FFI layer when the user drags the slider continuously.
  private readonly System.Reactive.Subjects.Subject<float> _radiusChanges = new();

  // ── Observable properties ───────────────────────────────────────────────────────

  /// <summary>The currently selected jet, or <c>null</c> when none is selected.</summary>
  [ObservableProperty]
  private JetViewModel? _selectedJet;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(AddJetCommand))]
  private float _manualNucleusRadiusKm = 2.0f;

  /// <summary>
  /// <c>true</c> when a comet has been committed to the native runtime
  /// (i.e. <see cref="CometConfigService.IsAlmanacCommitted"/> has emitted <c>true</c>).
  /// <see cref="AddJetCommand"/> is disabled until this is <c>true</c> because
  /// <c>avkSimulationContext_addParticleSystem</c> requires a comet entity in the scene.
  /// </summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(AddJetCommand))]
  private bool _isCometCommitted;

  [ObservableProperty]
  private bool _enableLegacyExpanders;

  [ObservableProperty]
  private int _isGizmoVisibleIndex = 0; // 0 = Yes, 1 = No

  partial void OnIsGizmoVisibleIndexChanged(int value)
  {
    _runtimeService?.SetSphereGizmoVisibility(value == 0);
  }

  [ObservableProperty]
  private bool _isSimulationRunning;

  private bool _snapshotExists;
  private readonly TimelineService _timelineService;

  // ── Construction ─────────────────────────────────────────────────────────────────

  public ModelTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<ModelSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    CometConfigService cometConfigService,
    INativeRuntimeService runtimeService,
    IUiThreadDispatcher dispatcher,
    ICometMessenger cometMessenger,
    IPlatformWindowService platformWindowService,
    TimelineService timelineService)
    : base("Model", sessionService, cometMessenger)
  {
    _translationService = translationService;
    _cometSessionService = cometSessionService;
    _cometConfigService = cometConfigService;
    _schedulerProvider = schedulerProvider;
    _runtimeService = runtimeService;
    _dispatcher = dispatcher;
    _platformWindowService = platformWindowService;
    _timelineService = timelineService;
    Icon = "⬡"; // hexagon / 3D object — U+2B21
    SubscribeToStrings(schedulerProvider);

    _timelineService.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(running =>
      {
          IsSimulationRunning = running;
          if (running) _snapshotExists = true;
      })
      .AddDisposableTo(_disposables);

    // Track _modelChangeSub and _radiusChanges lifetime alongside all other subs.
    _disposables.Add(_modelChangeSub);
    _disposables.Add(_radiusChanges);

    // Wire the debounced radius subject: fire SetNucleusRadiusKm at most once per 250 ms
    // so the FFI layer is not flooded while the user drags the slider.
    _radiusChanges
      .Throttle(TimeSpan.FromMilliseconds(250), schedulerProvider.MainThread)
      .Subscribe(r => _cometConfigService.SetNucleusRadiusKm(r))
      .AddDisposableTo(_disposables);

    // Push the UI default value at startup so the engine doesn't fall back to 50 km
    _radiusChanges.OnNext(EffectiveNucleusRadiusKm);

    // Re-wire model-session changes now that _schedulerProvider is set.
    // The base constructor already fired OnPropertyChanged(CurrentSession) but
    // _schedulerProvider was null at that point (DefaultScheduler fallback). Replace
    // that subscription with one that uses the correct injected scheduler.
    if (CurrentSession is not null)
      WireModelSessionChanges(CurrentSession);

    // Seed with current committed state and subscribe to future changes.
    IsCometCommitted = cometConfigService.IsAlmanacCommittedValue;
    cometConfigService.IsAlmanacCommitted
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(committed =>
      {
        IsCometCommitted = committed;
        var session = CurrentSession;
        if (session == null) return;

        if (!committed)
        {
          foreach (var jet in session.Jets)
          {
            jet.NativePsId = 0;
          }
        }
        else
        {
          bool isFirst = true;
          foreach (var jet in session.Jets)
          {
            AddJetNatively(jet, session, isFirst);
            isFirst = false;
          }
        }
      })
      .AddDisposableTo(_disposables);

    _cometConfigService.NucleusRadiusKm
      .Subscribe(_ =>
      {
        _dispatcher.Dispatch(() =>
        {
          AddJetCommand.NotifyCanExecuteChanged();
          OnPropertyChanged(nameof(IsNucleusRadiusUnknown));
        });
      })
      .AddDisposableTo(_disposables);
  }

  // ── Session passthrough ──────────────────────────────────────────────────────────

  /// <summary>
  /// The live jet list from the current session, suitable for direct
  /// <c>ItemsSource</c> binding in the view.
  /// </summary>
  public ObservableCollection<JetViewModel>? Jets => CurrentSession?.Jets;

  private CometSession? GetCometSession()
  {
    if (_cometSessionService.ActiveSessionIds.Count == 0) return null;
    return _cometSessionService.GetSession(_cometSessionService.ActiveSessionIds[0]);
  }

  private float EffectiveNucleusRadiusKm =>
    ManualNucleusRadiusKm > 0f
      ? ManualNucleusRadiusKm
      : (GetCometSession()?.NucleusRadiusKm ?? 0f);

  private bool CanAddJet() => IsCometCommitted && EffectiveNucleusRadiusKm > 0f;

  /// <summary>
  /// Nullable proxy for <see cref="ManualNucleusRadiusKm"/> so that the
  /// <c>NumericUpDown</c> shows a watermark when no manual radius is entered.
  /// <c>null</c> ↔ internal value 0 (not yet set).
  /// </summary>
  public float? ManualNucleusRadiusKmNullable
  {
    get => ManualNucleusRadiusKm > 0f ? ManualNucleusRadiusKm : null;
    set
    {
      ManualNucleusRadiusKm = value ?? 0f;
      OnPropertyChanged();
    }
  }

  /// <summary>
  /// <c>true</c> when no nucleus radius is available from either Horizon or manual entry.
  /// Drives the "Enter a radius to enable jet creation" hint in the view.
  /// </summary>
  public bool IsNucleusRadiusUnknown => EffectiveNucleusRadiusKm == 0f;



  /// <summary>
  /// Raised automatically by the MVVM toolkit when <see cref="ManualNucleusRadiusKm"/> changes.
  /// Keeps <see cref="ManualNucleusRadiusKmNullable"/> and <see cref="IsNucleusRadiusUnknown"/> in sync.
  /// </summary>
  partial void OnManualNucleusRadiusKmChanged(float value)
  {
    OnPropertyChanged(nameof(ManualNucleusRadiusKmNullable));
    OnPropertyChanged(nameof(IsNucleusRadiusUnknown));
    // Publish to the debounced subject — the subscription in the constructor
    // fires SetNucleusRadiusKm (→ UpdateCometNucleusRadius FFI) at most once
    // per 250 ms, preventing FFI flood while the user drags the slider.
    _radiusChanges.OnNext(EffectiveNucleusRadiusKm);
  }

  /// <summary>
  /// Called whenever an observable property changes. Intercepts <see cref="CurrentSession"/>
  /// changes (raised by the base class when the user switches sessions) to re-wire the
  /// model-session property-change subscription so that edits to the shared grain/dust
  /// parameters are pushed to all live jets in the new session.
  /// </summary>
  protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

    if (e.PropertyName == nameof(ManualNucleusRadiusKm) && !_timelineService.IsSimulationRunningValue && _snapshotExists)
    {
        _timelineService.SnapshotRestore();
        _snapshotExists = false;
    }

    if (e.PropertyName == nameof(CurrentSession))
    {
      if (CurrentSession is null)
        _modelChangeSub.Disposable = null;
      else
        WireModelSessionChanges(CurrentSession);
    }
  }

  // ── Commands ───────────────────────────────────────────────────────────────────

  private void AddJetNatively(JetViewModel jet, ModelSession session, bool isFirst)
  {
    var model = BuildModel(session);
    var psJet = BuildJet(jet);

    if (isFirst)
    {
      var computed = _runtimeService.AddFirstParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
      if (computed is not null)
      {
        jet.Beta = computed.Beta;
        jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
      }
    }
    else
    {
      _runtimeService.AddParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
    }
  }

  public void SetCursorPosition(int x, int y)
  {
      _platformWindowService.SetCursorPosition(x, y);
  }

  /// <summary>
  /// Adds a new jet with physically-reasonable random defaults and registers it
  /// with the native particle system runtime.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanAddJet))]
  private void AddJet()
  {
    var session = CurrentSession;
    if (session is null) return;

    var jet = new JetViewModel();
    jet.DisplayIndex = session.Jets.Count + 1;
    bool isFirst = session.Jets.Count == 0;

    session.Jets.Add(jet);
    SelectedJet = jet;

    if (IsCometCommitted)
    {
      AddJetNatively(jet, session, isFirst);
    }

    // Subscribe to this jet's property changes to push updates to native
    SubscribeJetChanges(jet, session);
  }

  /// <summary>
  /// Removes the given jet from the list.
  /// Native removal is a TODO pending <c>avkSimulationContext_removeParticleSystem</c> FFI.
  /// </summary>
  [RelayCommand]
  private void RemoveJet(JetViewModel? jet)
  {
    if (jet is null || CurrentSession is null) return;

    // Remove from native ECS — Drop impl handles GPU timeline teardown
    if (jet.NativePsId != 0)
      _runtimeService.RemoveParticleSystem(jet.NativePsId);

    CurrentSession.Jets.Remove(jet);

    // Re-index remaining jets for display
    for (int i = 0; i < CurrentSession.Jets.Count; i++)
      CurrentSession.Jets[i].DisplayIndex = i + 1;

    if (SelectedJet == jet)
      SelectedJet = null;
  }

  // ── Localization helper ──────────────────────────────────────────────────────────

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────────

  private static ParticleSystemModel BuildModel(ModelSession s) => new(
    MassVariabilityPerc: s.MassVariabilityPerc,
    DiametreUm:          s.DiametreUm,
    DensityGCm3:         s.DensityGCm3,
    ScatteringEfficiency: s.ScatteringEfficiency,
    Afrho0Cm:            s.Afrho0Cm,
    AfrhoPower:          s.AfrhoPower,
    AfrhoCutoffAu:       s.AfrhoCutoffAu,
    AfrhoMaxValueCm:     s.AfrhoMaxValueCm);

  private ParticleSystemJet BuildJet(JetViewModel j) => new(
    LatitudeRad:         j.LatitudeRad,
    LongitudeRad:        j.LongitudeRad,
    ApertureRad:         j.ApertureRad,
    StartVelocityMean:   j.StartVelocityMeanMs,
    StartVelocityStd:    j.StartVelocityStdMs,
    StreamColor:         j.StreamColor,
    NucleusRadiusKm:     EffectiveNucleusRadiusKm,
    Seed:                j.Seed);

  /// <summary>
  /// Subscribes to <paramref name="session"/>'s <see cref="System.ComponentModel.INotifyPropertyChanged"/>
  /// so that any edit to the shared grain/dust model properties is forwarded to all live jets
  /// via <see cref="INativeRuntimeService.ModifyParticleSystem"/> (debounced 250 ms).
  /// The subscription is stored in <see cref="_modelChangeSub"/> which disposes the previous
  /// subscription automatically when this is called again on a session switch.
  /// </summary>
  private void WireModelSessionChanges(ModelSession session)
  {
    // tabs are scoped: a reopened tab gets a new VM over the same session (native still has it on)
    ShowReferencePositionError = session.ShowReferencePositionError;
    DustSoftening = session.DustSoftening;
    DustTracers = session.DustTracers;
    DustFlow = session.DustFlow;
    PushDustViewFlags();
    DustFlowSpeed = session.DustFlowSpeed;
    _runtimeService?.SetDustFlowSpeed((float)DustFlowSpeed);

    _modelChangeSub.Disposable = Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => session.PropertyChanged += h,
        h => session.PropertyChanged -= h)
      .Throttle(TimeSpan.FromMilliseconds(250),
        _schedulerProvider?.Background ?? System.Reactive.Concurrency.DefaultScheduler.Instance)
      .Subscribe(_ => PushModelToAllJets(session));
  }

  /// <summary>
  /// Pushes the current <see cref="ModelSession"/> common properties to every live jet
  /// by calling <see cref="INativeRuntimeService.ModifyParticleSystem"/> for each one.
  /// Called whenever a shared model property (grain size, density, Afρ, …) changes.
  /// </summary>
  private void PushModelToAllJets(ModelSession session)
  {
    var model = BuildModel(session);
    foreach (var jet in session.Jets)
    {
      if (jet.NativePsId == 0) continue;
      var psJet = BuildJet(jet);
      bool ok = _runtimeService.ModifyParticleSystem(
        jet.NativePsId, model, psJet,
        out ParticleSystemComputedProperties computed);
      if (ok)
      {
        jet.Beta = computed.Beta;
        jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
      }
    }
  }

  /// <summary>
  /// Subscribes to a jet's <see cref="INotifyPropertyChanged"/> so that any edit
  /// is forwarded to the native runtime (debounced 250 ms to avoid flooding).
  /// </summary>
  /// <summary>Draws the comet's reference-position error lines (cross-track, same-epoch).</summary>
  [ObservableProperty]
  private bool _showReferencePositionError;

  /// <summary>
  /// Dust display stretch softening ("dust visibility"). Dust accumulates linear optical depth;
  /// the composite maps it through asinh(τ/s)/asinh(1/s): a smaller <c>s</c> lifts the faint, old
  /// tail (1e4× fainter than the coma) while the coma saturates. Visual only (no restore).
  /// </summary>
  [ObservableProperty]
  private double _dustSoftening = ModelTabDefaults.DustSoftening;

  partial void OnDustSofteningChanged(double value)
  {
    double s = ModelTabDefaults.ClampDustSoftening(value);
    if (s != value)
    {
      DustSoftening = s;
      return;
    }
    if (CurrentSession is { } session)
      session.DustSoftening = s;
    _runtimeService?.SetDustSoftening((float)s);
  }

  /// <summary>
  /// Dust tracers: one particle in 256 is also drawn as a bright dot at its exact position. In a
  /// wide view the dust moves far less than a pixel per second at low sim speed (it drifts at
  /// 2–150 m/s while a pixel spans tens of km): the tracers are real particles to follow, visible
  /// through the fog of the coma. Visual only.
  /// </summary>
  [ObservableProperty]
  private bool _dustTracers = false;

  partial void OnDustTracersChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.DustTracers = value;
    PushDustViewFlags();
  }

  /// <summary>
  /// Flow pulses: brightness marks riding the dust along synchrones (peak 4×, trough 0.15×), a
  /// motion cue near the nucleus. Off by default: from 0.03 AU out the marks are the synchrone fan
  /// itself, hard rays through the nucleus that the physical optical depth does not have. Stored
  /// in the model session; visual only.
  /// </summary>
  [ObservableProperty]
  private bool _dustFlow = false;

  partial void OnDustFlowChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.DustFlow = value;
    PushDustViewFlags();
  }

  /// <summary>
  /// Time-lapse factor K of the dust flow marks (brightness marks riding the dust): 1 = the marks
  /// move with the dust, K = that many times faster in its direction, so the swarm keeps its true
  /// velocity field at a readable pace from afar. Stored in the model session; visual only.
  /// </summary>
  [ObservableProperty]
  private double _dustFlowSpeed = DustFlowSpeedDefault;

  public const double DustFlowSpeedDefault = 1.0;
  public const double DustFlowSpeedMin = 1.0;
  public const double DustFlowSpeedMax = 10_000.0;

  partial void OnDustFlowSpeedChanged(double value)
  {
    var clamped = ClampDustFlowSpeed(value);
    if (clamped != value)
    {
      DustFlowSpeed = clamped;
      return;
    }
    if (CurrentSession is { } session)
      session.DustFlowSpeed = value;
    _runtimeService?.SetDustFlowSpeed((float)value);
  }

  /// <summary>Native range of the flow time-lapse factor (<c>dust::FLOW_SPEED_MIN/MAX</c>).</summary>
  internal static double ClampDustFlowSpeed(double k) =>
    double.IsNaN(k) ? DustFlowSpeedDefault : Math.Min(Math.Max(k, DustFlowSpeedMin), DustFlowSpeedMax);

  /// <summary>
  /// Native dust view flags (<c>dust::DUST_VIEW_*</c>): bit 0 tracers, bit 1 flow. The flow is
  /// always on: it runs while the sim runs and freezes in place on pause (no toggle).
  /// </summary>
  internal static uint DustViewFlags(bool tracers, bool flow) => (tracers ? 1u : 0u) | (flow ? 2u : 0u);

  private void PushDustViewFlags() =>
    _runtimeService?.SetDustViewFlags(DustViewFlags(DustTracers, DustFlow));

  partial void OnShowReferencePositionErrorChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.ShowReferencePositionError = value;
    // null while the base constructor raises CurrentSession
    _runtimeService?.SetReferenceErrorVisible(value);
  }

  /// <summary>
  /// Jet properties that only affect drawing (not emission or physics): editing them while paused
  /// must not restore the snapshot.
  /// </summary>
  internal static readonly System.Collections.Generic.HashSet<string?> VisualOnlyJetProperties =
    new() { nameof(JetViewModel.StreamColor) };

  private void SubscribeJetChanges(JetViewModel jet, ModelSession session)
  {
    Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => jet.PropertyChanged += h,
        h => jet.PropertyChanged -= h)
      .Where(e =>
      {
        var name = e.EventArgs.PropertyName;
        // computed outputs and the preview toggle (own subscription below) are not forwarded
        if (name == nameof(JetViewModel.Beta) ||
            name == nameof(JetViewModel.DustProductionRateAt1AuKgs) ||
            name == nameof(JetViewModel.IsPreviewVisible))
          return false;

        // Visual-only edits (draw parameters, read by the renderer every frame) are forwarded
        // without restoring the snapshot: a restore swaps in the scene cloned at Play, whose dust
        // ring only re-emits on simulation ticks, so the tail vanished while paused.
        if (!VisualOnlyJetProperties.Contains(name))
        {
            _dispatcher.Dispatch(() =>
            {
                if (!_timelineService.IsSimulationRunningValue && _snapshotExists)
                {
                    _timelineService.SnapshotRestore();
                    _snapshotExists = false;
                }
            });
        }
        return true;
      })
      .Throttle(TimeSpan.FromMilliseconds(250), _schedulerProvider.Background)
      .Subscribe(_ =>
      {
        if (jet.NativePsId == 0) return;
        var model = BuildModel(session);
        var psJet = BuildJet(jet);
        bool ok = _runtimeService.ModifyParticleSystem(
          jet.NativePsId, model, psJet,
          out ParticleSystemComputedProperties computed);
        if (ok)
        {
          jet.Beta = computed.Beta;
          jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
        }
      })
      .AddDisposableTo(_disposables);

    Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => jet.PropertyChanged += h,
        h => jet.PropertyChanged -= h)
      .Where(e => e.EventArgs.PropertyName == nameof(JetViewModel.IsPreviewVisible))
      .Throttle(TimeSpan.FromMilliseconds(250), _schedulerProvider.Background)
      .Subscribe(_ =>
      {
        if (jet.NativePsId != 0)
        {
          _runtimeService.SetJetPreviewVisibility(jet.NativePsId, jet.IsPreviewVisible);
        }
      })
      .AddDisposableTo(_disposables);
  }
}
