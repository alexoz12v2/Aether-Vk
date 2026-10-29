using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Models;
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
  private readonly ICameraServiceRegistry? _cameras;
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

  /// <summary>The comet's sphere gizmo (wireframe + axes), saved in the model session.</summary>
  [ObservableProperty]
  private bool _isGizmoVisible = true;

  partial void OnIsGizmoVisibleChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.SphereGizmoVisible = value;
    _runtimeService?.SetSphereGizmoVisibility(value);
  }

  private uint _gizmoParts = SphereGizmoPart.Default;

  /// <summary>Which sphere gizmo parts are drawn (<see cref="SphereGizmoPart"/> bits), saved in the model session.</summary>
  public uint GizmoParts
  {
    get => _gizmoParts;
    set
    {
      value &= SphereGizmoPart.All;
      if (!SetProperty(ref _gizmoParts, value))
        return;
      OnPropertyChanged(nameof(ShowGizmoGrid));
      OnPropertyChanged(nameof(ShowGizmoBodyAxes));
      OnPropertyChanged(nameof(ShowGizmoSpinAxis));
      OnPropertyChanged(nameof(ShowGizmoSun));
      OnPropertyChanged(nameof(ShowGizmoVelocity));
      if (CurrentSession is { } session)
        session.SphereGizmoParts = value;
      _runtimeService?.SetSphereGizmoParts(value);
    }
  }

  private void SetGizmoPart(uint part, bool on) => GizmoParts = on ? GizmoParts | part : GizmoParts & ~part;

  public bool ShowGizmoGrid { get => (GizmoParts & SphereGizmoPart.Grid) != 0; set => SetGizmoPart(SphereGizmoPart.Grid, value); }
  public bool ShowGizmoBodyAxes { get => (GizmoParts & SphereGizmoPart.BodyAxes) != 0; set => SetGizmoPart(SphereGizmoPart.BodyAxes, value); }
  public bool ShowGizmoSpinAxis { get => (GizmoParts & SphereGizmoPart.SpinAxis) != 0; set => SetGizmoPart(SphereGizmoPart.SpinAxis, value); }
  public bool ShowGizmoSun { get => (GizmoParts & SphereGizmoPart.Sun) != 0; set => SetGizmoPart(SphereGizmoPart.Sun, value); }
  public bool ShowGizmoVelocity { get => (GizmoParts & SphereGizmoPart.Velocity) != 0; set => SetGizmoPart(SphereGizmoPart.Velocity, value); }

  /// <summary>
  /// The selected jet's latitude/longitude arcs and degree labels in the viewport (the other jets
  /// show their direction line only), saved in the model session.
  /// </summary>
  [ObservableProperty]
  private bool _showJetAngles;

  partial void OnShowJetAnglesChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.ShowJetAngles = value;
    ApplyJetAngles();
  }

  // selecting a jet only moves the angle display: the camera stays where it is
  partial void OnSelectedJetChanged(JetViewModel? value) => ApplyJetAngles();

  private void ApplyJetAngles()
  {
    if (Jets is not { } jets || _runtimeService is null)
      return;
    foreach (var jet in jets)
      if (jet.NativePsId != 0)
        _runtimeService.SetJetAngleVisibility(jet.NativePsId, ShowJetAngles && ReferenceEquals(jet, SelectedJet));
  }

  // a commit rebuilds the comet with its gizmo shown (and a toggle made before any commit had no
  // comet to reach): apply the stored choice once the comet exists
  partial void OnIsCometCommittedChanged(bool value)
  {
    RefreshDustCurves();
    if (value)
    {
      _runtimeService?.SetSphereGizmoVisibility(IsGizmoVisible);
      _runtimeService?.SetSphereGizmoParts(GizmoParts);
    }
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
    TimelineService timelineService,
    ICameraServiceRegistry? cameras = null,
    CometPositionTrackerService? cometTracker = null)
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
    _cameras = cameras;
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

    // the comet's distance from the Sun, for the Afρ plot marker and the "Afρ now" read-out
    cometTracker?.CometPosition
      .Subscribe(p =>
      {
        double r = p is { } v ? v.Length() : double.NaN;
        double old = HeliocentricDistanceAu;
        // the sim ticks every frame: redraw the plot only when r_h moved visibly
        if (double.IsNaN(r) != double.IsNaN(old) || Math.Abs(r - old) > 1e-4)
          HeliocentricDistanceAu = r;
      })
      .AddDisposableTo(_disposables);

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
            jet.NativeError = null;
          }
        }
        else
        {
          // idempotent: a savefile open may have created them already
          JetNativeSync.EnsureNative(_runtimeService, session, EffectiveNucleusRadiusKm);
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
    if (CurrentSession is { } s)
      s.ManualNucleusRadiusKm = value;
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
    JetNativeSync.AddNative(_runtimeService, BuildModel(session, jet), BuildJet(jet), jet, isFirst);
    // a new native jet starts without angles: the selected one gets them back
    if (ShowJetAngles && ReferenceEquals(jet, SelectedJet) && jet.NativePsId != 0)
      _runtimeService.SetJetAngleVisibility(jet.NativePsId, true);
  }

  /// <summary>
  /// Asks the core again for every jet it refused (<see cref="JetViewModel.NativeError"/>); the
  /// jets already in the simulation are left alone.
  /// </summary>
  [RelayCommand]
  private void RetryNativeJets()
  {
    if (!IsCometCommitted || CurrentSession is not { } session) return;
    JetNativeSync.EnsureNative(_runtimeService, session, EffectiveNucleusRadiusKm);
    RefreshTotalActivityText();
  }

  public void SetCursorPosition(int x, int y)
  {
      _platformWindowService.SetCursorPosition(x, y);
  }

  /// <summary>
  /// Adds a new jet with a random site and ejection, the grain and activity defaults of the model
  /// session, and registers it with the native particle system runtime.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanAddJet))]
  private void AddJet()
  {
    var session = CurrentSession;
    if (session is null) return;

    // q is set per jet only (the model-wide box is gone): a new jet starts at the usual 3.5, whatever
    // an older save left in the session (its jets without a q of their own still inherit it)
    var jet = new JetViewModel(session.DefaultGrain with { SizeQ = Models.JetGrain.Default.SizeQ });
    jet.DisplayIndex = session.Jets.Count + 1;
    bool isFirst = session.Jets.Count == 0;

    session.Jets.Add(jet);
    SelectedJet = jet;

    if (IsCometCommitted)
    {
      AddJetNatively(jet, session, isFirst);
    }

    // Subscribe to this jet's property changes to push updates to native
    _subscribedJets.Add(jet);
    SubscribeJetChanges(jet, session);
    RefreshTotalActivityText();
  }

  /// <summary>
  /// Gives every other jet the selected jet's grain and activity parameters (diameter, density,
  /// Q_pr, albedo, q, F, Afρ₀); sites, speeds and colours stay as they are.
  /// </summary>
  [RelayCommand]
  private void CopyGrainToAllJets()
  {
    if (SelectedJet is not { } source || CurrentSession is not { } session)
      return;
    var grain = source.Grain;
    foreach (var jet in session.Jets)
      if (!ReferenceEquals(jet, source))
        jet.Grain = grain;
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
    RefreshTotalActivityText();
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

  private static ParticleSystemModel BuildModel(ModelSession s, JetViewModel j) => JetNativeSync.BuildModel(s, j);

  private ParticleSystemJet BuildJet(JetViewModel j) => JetNativeSync.BuildJet(j, EffectiveNucleusRadiusKm);

  // jets whose property changes are forwarded natively (each subscribed once)
  private readonly System.Collections.Generic.HashSet<JetViewModel> _subscribedJets = new();
  private ModelSession? _restoredHooked;

  /// <summary>Forwards edits of every session jet not yet subscribed (restored jets, a reopened tab).</summary>
  private void EnsureJetSubscriptions(ModelSession session)
  {
    // the base constructor wires the session before this class's services exist: the
    // constructor wires it again once they do
    if (_schedulerProvider is null || _dispatcher is null)
      return;
    for (int i = 0; i < session.Jets.Count; i++)
    {
      var jet = session.Jets[i];
      jet.DisplayIndex = i + 1;
      if (_subscribedJets.Add(jet))
        SubscribeJetChanges(jet, session);
    }
  }

  // ── Dust curves (Afρ(r_h) and the grain size law) ────────────────────────────

  /// <summary>Afρ(r_h) of the defaults for new jets (engine law), with a marker at the comet's r_h.</summary>
  [ObservableProperty]
  private CurvePlotModel? _afrhoPlot;

  /// <summary>Grain size PDFs by number and by cross-section, of the defaults for new jets.</summary>
  [ObservableProperty]
  private CurvePlotModel? _sizeDistributionPlot;

  /// <summary>Afρ(r_h) of the selected jet (its Afρ₀, the model-wide law); null without a selection.</summary>
  [ObservableProperty]
  private CurvePlotModel? _jetAfrhoPlot;

  /// <summary>Grain size PDFs of the selected jet; null without a selection.</summary>
  [ObservableProperty]
  private CurvePlotModel? _jetSizeDistributionPlot;

  /// <summary>"Afρ now: X cm → Q_d ≈ Y kg/s" for the comet's current r_h and the selected jet.</summary>
  [ObservableProperty]
  private string _afrhoNowText = "";

  /// <summary>"Σ Afρ → Σ Q_d" over every jet, at 1 AU and at the comet's current r_h.</summary>
  [ObservableProperty]
  private string _totalActivityText = "";

  /// <summary>The comet's heliocentric distance (AU); NaN when unknown.</summary>
  [ObservableProperty]
  private double _heliocentricDistanceAu = double.NaN;

  partial void OnHeliocentricDistanceAuChanged(double value) => RefreshDustCurves();

  // the session's defaults and the model-wide Afρ law
  private static readonly System.Collections.Generic.HashSet<string?> CurveInputs = new()
  {
    nameof(ModelSession.Afrho0Cm), nameof(ModelSession.AfrhoPower), nameof(ModelSession.AfrhoCutoffAu),
    nameof(ModelSession.AfrhoMaxValueCm), nameof(ModelSession.DiametreUm), nameof(ModelSession.SizeRange),
    nameof(ModelSession.SizeQ),
  };

  // a jet's own curve inputs
  private static readonly System.Collections.Generic.HashSet<string?> JetCurveInputs = new()
  {
    nameof(JetViewModel.Afrho0Cm), nameof(JetViewModel.DiametreUm), nameof(JetViewModel.SizeRange),
    nameof(JetViewModel.SizeQ),
  };

  /// <summary>
  /// The model-wide parameters every jet's native record carries (the Afρ law's shape): an edit is
  /// pushed to all jets. The session's other dust values are defaults for new jets only.
  /// </summary>
  internal static readonly System.Collections.Generic.HashSet<string?> ModelWideJetProperties = new()
  {
    nameof(ModelSession.AfrhoPower), nameof(ModelSession.AfrhoCutoffAu), nameof(ModelSession.AfrhoMaxValueCm),
  };

  private ModelSession? _curvesHooked;
  private JetViewModel? _rateHooked;

  private void HookCurveSession(ModelSession? session)
  {
    if (ReferenceEquals(_curvesHooked, session))
      return;
    if (_curvesHooked is not null)
    {
      _curvesHooked.PropertyChanged -= OnCurveInputChanged;
      _curvesHooked.Jets.CollectionChanged -= OnJetsCollectionChanged;
    }
    _curvesHooked = session;
    if (session is not null)
    {
      session.PropertyChanged += OnCurveInputChanged;
      session.Jets.CollectionChanged += OnJetsCollectionChanged;
    }
  }

  private void OnCurveInputChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is null || CurveInputs.Contains(e.PropertyName))
      RefreshDustCurves();
  }

  // a savefile open / New clears and refills the list
  private void OnJetsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e) =>
    RefreshTotalActivityText();

  partial void OnSelectedJetChanged(JetViewModel? oldValue, JetViewModel? newValue)
  {
    if (_rateHooked is not null)
      _rateHooked.PropertyChanged -= OnSelectedJetPropertyChanged;
    _rateHooked = newValue;
    if (newValue is not null)
      newValue.PropertyChanged += OnSelectedJetPropertyChanged;
    RefreshJetCurves();
    RefreshAfrhoNowText();
  }

  // the rate is written from the native-push thread: refresh on the UI thread
  private void OnSelectedJetPropertyChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName == nameof(JetViewModel.DustProductionRateAt1AuKgs))
      OnUiThread(RefreshAfrhoNowText);
    else if (e.PropertyName is null || JetCurveInputs.Contains(e.PropertyName))
      OnUiThread(() =>
      {
        RefreshJetCurves();
        RefreshAfrhoNowText();
      });
  }

  // any jet's Afρ₀ or rate: the Σ read-out (rates arrive from the native-push thread)
  private void OnAnyJetActivityChanged(string? propertyName)
  {
    if (propertyName is null || propertyName == nameof(JetViewModel.Afrho0Cm)
        || propertyName == nameof(JetViewModel.DustProductionRateAt1AuKgs))
      OnUiThread(RefreshTotalActivityText);
  }

  private void OnUiThread(Action action)
  {
    if (_dispatcher is null) action();
    else _dispatcher.Dispatch(action);
  }

  private double CurrentRhAu => IsCometCommitted ? HeliocentricDistanceAu : double.NaN;

  private void RefreshDustCurves()
  {
    if (CurrentSession is not { } s)
      return;
    AfrhoPlot = DustCurves.AfrhoPlot(s.Afrho0Cm, s.AfrhoPower, s.AfrhoCutoffAu, s.AfrhoMaxValueCm, CurrentRhAu);
    SizeDistributionPlot = DustCurves.SizeDistributionPlot(s.DiametreUm, s.SizeRange, Models.JetGrain.Default.SizeQ);
    RefreshJetCurves();
    RefreshAfrhoNowText();
    RefreshTotalActivityText();
  }

  private void RefreshJetCurves()
  {
    if (CurrentSession is not { } s || SelectedJet is not { } j)
    {
      JetAfrhoPlot = null;
      JetSizeDistributionPlot = null;
      return;
    }
    JetAfrhoPlot = DustCurves.AfrhoPlot(j.Afrho0Cm, s.AfrhoPower, s.AfrhoCutoffAu, s.AfrhoMaxValueCm, CurrentRhAu);
    JetSizeDistributionPlot = DustCurves.SizeDistributionPlot(j.DiametreUm, j.SizeRange, j.SizeQ);
  }

  private void RefreshAfrhoNowText()
  {
    if (CurrentSession is not { } s)
      return;
    // the selected jet's Afρ₀ and rate (each jet has its own), else the first jet, else the defaults
    var jet = SelectedJet ?? (s.Jets.Count > 0 ? s.Jets[0] : null);
    double rate = jet is { NativePsId: not 0 } ? jet.DustProductionRateAt1AuKgs : double.NaN;
    AfrhoNowText = DustCurves.AfrhoNowText(CurrentRhAu, jet?.Afrho0Cm ?? s.Afrho0Cm, s.AfrhoPower,
      s.AfrhoCutoffAu, s.AfrhoMaxValueCm, rate, jet?.DisplayIndex ?? 0);
  }

  private void RefreshTotalActivityText()
  {
    if (CurrentSession is not { } s)
      return;
    var jets = new System.Collections.Generic.List<(double Afrho0Cm, double RateAt1AuKgs)>(s.Jets.Count);
    foreach (var j in s.Jets)
      jets.Add((j.Afrho0Cm, j.NativePsId != 0 ? j.DustProductionRateAt1AuKgs : double.NaN));
    TotalActivityText = DustCurves.TotalActivityText(jets, s.AfrhoPower, s.AfrhoCutoffAu, s.AfrhoMaxValueCm,
      CurrentRhAu);
  }

  public override void Dispose()
  {
    HookCurveSession(null);
    if (_rateHooked is not null)
      _rateHooked.PropertyChanged -= OnSelectedJetPropertyChanged;
    _rateHooked = null;
    if (_restoredHooked is not null)
      _restoredHooked.Restored -= OnSessionRestored;
    _restoredHooked = null;
    _disposables.Dispose();
    base.Dispose();
  }

  private void OnSessionRestored()
  {
    // a savefile open / New replaced the scene: the Play snapshot belongs to the old one, and
    // restoring it on the next edit would undo the open (back to its pre-open epoch)
    _snapshotExists = false;
    _dispatcher.Dispatch(() =>
    {
      if (CurrentSession is { } s)
        WireModelSessionChanges(s);
    });
  }

  private void WireModelSessionChanges(ModelSession session)
  {
    if (!ReferenceEquals(_restoredHooked, session))
    {
      if (_restoredHooked is not null)
        _restoredHooked.Restored -= OnSessionRestored;
      session.Restored += OnSessionRestored;
      _restoredHooked = session;
    }
    ManualNucleusRadiusKm = session.ManualNucleusRadiusKm;
    HookCurveSession(session);
    RefreshDustCurves();
    EnsureJetSubscriptions(session);
    // tabs are scoped: a reopened tab gets a new VM over the same session (native still has it on)
    ShowReferencePositionError = session.ShowReferencePositionError;
    DustSoftening = session.DustSoftening;
    DustTracers = session.DustTracers;
    DustFlow = session.DustFlow;
    PushDustViewFlags();
    DustFlowSpeed = session.DustFlowSpeed;
    _runtimeService?.SetDustFlowSpeed((float)DustFlowSpeed);
    IsGizmoVisible = session.SphereGizmoVisible;
    GizmoParts = session.SphereGizmoParts;
    ShowJetAngles = session.ShowJetAngles;

    // only the model-wide Afρ law reaches the jets: the other values are defaults for new jets
    _modelChangeSub.Disposable = Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => session.PropertyChanged += h,
        h => session.PropertyChanged -= h)
      .Where(e => e.EventArgs.PropertyName is null || ModelWideJetProperties.Contains(e.EventArgs.PropertyName))
      .Throttle(TimeSpan.FromMilliseconds(250),
        _schedulerProvider?.Background ?? System.Reactive.Concurrency.DefaultScheduler.Instance)
      .Subscribe(_ => PushModelToAllJets(session));
  }

  /// <summary>
  /// Pushes the model-wide Afρ law (<see cref="ModelWideJetProperties"/>) to every live jet, each
  /// with its own grain and Afρ₀, by calling <see cref="INativeRuntimeService.ModifyParticleSystem"/>
  /// for each one.
  /// </summary>
  private void PushModelToAllJets(ModelSession session)
  {
    foreach (var jet in session.Jets)
    {
      if (jet.NativePsId == 0) continue;
      var psJet = BuildJet(jet);
      bool ok = _runtimeService.ModifyParticleSystem(
        jet.NativePsId, BuildModel(session, jet), psJet,
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
        var model = BuildModel(session, jet);
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

    // the Σ Afρ → Σ Q_d read-out follows every jet's Afρ₀ and rate
    Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => jet.PropertyChanged += h,
        h => jet.PropertyChanged -= h)
      .Subscribe(e => OnAnyJetActivityChanged(e.EventArgs.PropertyName))
      .AddDisposableTo(_disposables);
  }
}
