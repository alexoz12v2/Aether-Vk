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

  // Subject used to debounce the radius push (startup and session restore). The manual radius
  // is edited in the Comet tab (feedback item 10b); this tab only reads it from the session.
  private readonly System.Reactive.Subjects.Subject<float> _radiusChanges = new();
  private float _lastSeenManualRadiusKm = 2.0f;

  // ── Observable properties ───────────────────────────────────────────────────────

  /// <summary>The currently selected jet, or <c>null</c> when none is selected.</summary>
  [ObservableProperty]
  private JetViewModel? _selectedJet;

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

    // the comet's distance from the Sun, for the Q_d plot marker and the "Q_d now" read-out
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
            jet.LitWarning = null;
          }
        }
        else
        {
          // idempotent: a savefile open may have created them already
          JetNativeSync.EnsureNative(_runtimeService, session, EffectiveNucleusRadiusKm, _timelineService.IgnitionUs(),
            LitWarningFormat);
        }
      })
      .AddDisposableTo(_disposables);

    _cometConfigService.NucleusRadiusKm
      .Subscribe(_ =>
      {
        _dispatcher.Dispatch(() =>
        {
          // the Comet tab pushed its radius edit: already pushed, nothing to re-push on a rewire
          if (CurrentSession is { } cs) _lastSeenManualRadiusKm = cs.ManualNucleusRadiusKm;
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
    // null while the base constructor wires the first session
    if (_cometSessionService is null || _cometSessionService.ActiveSessionIds.Count == 0) return null;
    return _cometSessionService.GetSession(_cometSessionService.ActiveSessionIds[0]);
  }

  /// <summary>The model session's manual radius (edited in the Comet tab), else the catalogue one.</summary>
  private float EffectiveNucleusRadiusKm =>
    CurrentSession is { ManualNucleusRadiusKm: > 0f } s
      ? s.ManualNucleusRadiusKm
      : (GetCometSession()?.NucleusRadiusKm ?? 0f);

  private bool CanAddJet() => IsCometCommitted && EffectiveNucleusRadiusKm > 0f;

  /// <summary>Localized <see cref="JetViewModel.LitWarning"/> text ({0} / {1}: the committed range ends).</summary>
  private string LitWarningFormat => _translationService.GetString("Tabs_Model_JetNeverLit");

  public bool IsNucleusRadiusUnknown => EffectiveNucleusRadiusKm == 0f;



  /// <summary>
  /// Called whenever an observable property changes. Intercepts <see cref="CurrentSession"/>
  /// changes (raised by the base class when the user switches sessions) to re-wire the
  /// model-session property-change subscription so that edits to the shared grain/dust
  /// parameters are pushed to all live jets in the new session.
  /// </summary>
  protected override void OnPropertyChanged(System.ComponentModel.PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

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
    JetNativeSync.AddNative(_runtimeService, BuildModel(session, jet),
      JetNativeSync.BuildJet(jet, EffectiveNucleusRadiusKm, _timelineService.IgnitionUs()), jet, isFirst,
      LitWarningFormat);
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
    JetNativeSync.EnsureNative(_runtimeService, session, EffectiveNucleusRadiusKm, _timelineService.IgnitionUs(),
      LitWarningFormat);
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
  /// Q_pr, albedo, q, F, Q_d₀); sites, speeds and colours stay as they are.
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

  // ── Dust curves (Q_d(r_h) and the grain size law) ────────────────────────────

  /// <summary>Q_d(r_h) of the defaults for new jets (engine law), with a marker at the comet's r_h.</summary>
  [ObservableProperty]
  private CurvePlotModel? _dustRatePlot;

  /// <summary>Grain size PDFs by number and by cross-section, of the defaults for new jets.</summary>
  [ObservableProperty]
  private CurvePlotModel? _sizeDistributionPlot;

  /// <summary>Q_d(r_h) of the selected jet (its Q_d₀, the model-wide law); null without a selection.</summary>
  [ObservableProperty]
  private CurvePlotModel? _jetDustRatePlot;

  /// <summary>Grain size PDFs of the selected jet; null without a selection.</summary>
  [ObservableProperty]
  private CurvePlotModel? _jetSizeDistributionPlot;

  /// <summary>"Q_d now (r_h X AU): Y kg/s" for the comet's current r_h and the selected jet.</summary>
  [ObservableProperty]
  private string _dustRateNowText = "";

  /// <summary>"Σ Q_d" over every jet, at 1 AU and at the comet's current r_h.</summary>
  [ObservableProperty]
  private string _totalActivityText = "";

  /// <summary>The comet's heliocentric distance (AU); NaN when unknown.</summary>
  [ObservableProperty]
  private double _heliocentricDistanceAu = double.NaN;

  partial void OnHeliocentricDistanceAuChanged(double value) => RefreshDustCurves();

  // the session's defaults and the model-wide Q_d law
  private static readonly System.Collections.Generic.HashSet<string?> CurveInputs = new()
  {
    nameof(ModelSession.DustRate1AuKgs), nameof(ModelSession.DustRatePower), nameof(ModelSession.DustRateCutoffAu),
    nameof(ModelSession.DustRateMaxKgs), nameof(ModelSession.DiametreUm), nameof(ModelSession.SizeRange),
    nameof(ModelSession.SizeQ),
  };

  // a jet's own curve inputs
  private static readonly System.Collections.Generic.HashSet<string?> JetCurveInputs = new()
  {
    nameof(JetViewModel.DustRate1AuKgs), nameof(JetViewModel.DiametreUm), nameof(JetViewModel.SizeRange),
    nameof(JetViewModel.SizeQ),
  };

  /// <summary>
  /// The model-wide parameters every jet's native record carries (the Q_d law's shape): an edit is
  /// pushed to all jets. The session's other dust values are defaults for new jets only.
  /// </summary>
  internal static readonly System.Collections.Generic.HashSet<string?> ModelWideJetProperties = new()
  {
    nameof(ModelSession.DustRatePower), nameof(ModelSession.DustRateCutoffAu), nameof(ModelSession.DustRateMaxKgs),
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
    RefreshDustRateNowText();
  }

  private void OnSelectedJetPropertyChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is null || JetCurveInputs.Contains(e.PropertyName))
      OnUiThread(() =>
      {
        RefreshJetCurves();
        RefreshDustRateNowText();
      });
  }

  // any jet's Q_d₀: the Σ read-out
  private void OnAnyJetActivityChanged(string? propertyName)
  {
    if (propertyName is null || propertyName == nameof(JetViewModel.DustRate1AuKgs))
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
    DustRatePlot = DustCurves.DustRatePlot(s.DustRate1AuKgs, s.DustRatePower, s.DustRateCutoffAu, s.DustRateMaxKgs, CurrentRhAu);
    SizeDistributionPlot = DustCurves.SizeDistributionPlot(s.DiametreUm, s.SizeRange, Models.JetGrain.Default.SizeQ);
    RefreshJetCurves();
    RefreshDustRateNowText();
    RefreshTotalActivityText();
  }

  private void RefreshJetCurves()
  {
    if (CurrentSession is not { } s || SelectedJet is not { } j)
    {
      JetDustRatePlot = null;
      JetSizeDistributionPlot = null;
      return;
    }
    JetDustRatePlot = DustCurves.DustRatePlot(j.DustRate1AuKgs, s.DustRatePower, s.DustRateCutoffAu, s.DustRateMaxKgs, CurrentRhAu);
    JetSizeDistributionPlot = DustCurves.SizeDistributionPlot(j.DiametreUm, j.SizeRange, j.SizeQ);
  }

  private void RefreshDustRateNowText()
  {
    if (CurrentSession is not { } s)
      return;
    // the selected jet's Q_d₀ (each jet has its own), else the first jet, else the defaults
    var jet = SelectedJet ?? (s.Jets.Count > 0 ? s.Jets[0] : null);
    DustRateNowText = DustCurves.DustRateNowText(CurrentRhAu, jet?.DustRate1AuKgs ?? s.DustRate1AuKgs, s.DustRatePower,
      s.DustRateCutoffAu, s.DustRateMaxKgs, jet?.DisplayIndex ?? 0);
  }

  private void RefreshTotalActivityText()
  {
    if (CurrentSession is not { } s)
      return;
    var jets = new System.Collections.Generic.List<double>(s.Jets.Count);
    foreach (var j in s.Jets)
      jets.Add(j.DustRate1AuKgs);
    TotalActivityText = DustCurves.TotalActivityText(jets, s.DustRatePower, s.DustRateCutoffAu, s.DustRateMaxKgs,
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
    // the radius may have changed with the session (savefile open): push it (debounced) when it
    // did, as the tab's own radius property used to, and re-evaluate the jet button
    if (session.ManualNucleusRadiusKm != _lastSeenManualRadiusKm)
    {
      _lastSeenManualRadiusKm = session.ManualNucleusRadiusKm;
      _radiusChanges.OnNext(EffectiveNucleusRadiusKm);
    }
    AddJetCommand.NotifyCanExecuteChanged();
    OnPropertyChanged(nameof(IsNucleusRadiusUnknown));
    HookCurveSession(session);
    RefreshDustCurves();
    EnsureJetSubscriptions(session);
    // tabs are scoped: a reopened tab gets a new VM over the same session (native still has it on)
    ShowReferencePositionError = session.ShowReferencePositionError;
    DustSoftening = session.DustSoftening;
    DustTracers = session.DustTracers;
    DustFlow = session.DustFlow;
    DustTrajectories = session.DustTrajectories;
    DustRhoNorm = session.DustRhoNorm;
    PushDustViewFlags();
    DustExposureEv = session.DustExposureEv;
    _runtimeService?.SetDustExposureEv((float)DustExposureEv);
    DustFlowSpeed = session.DustFlowSpeed;
    _runtimeService?.SetDustFlowSpeed((float)DustFlowSpeed);
    IsGizmoVisible = session.SphereGizmoVisible;
    GizmoParts = session.SphereGizmoParts;
    ShowJetAngles = session.ShowJetAngles;

    // only the model-wide Q_d law reaches the jets: the other values are defaults for new jets
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
  /// Pushes the model-wide Q_d law (<see cref="ModelWideJetProperties"/>) to every live jet, each
  /// with its own grain and Q_d₀, by calling <see cref="INativeRuntimeService.ModifyParticleSystem"/>
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
  /// Trajectory lines: about 200 grains' paths since they left the nucleus, drawn as lines, one
  /// colour per jet (native view flag bit 2). Stored in the model session; visual only.
  /// </summary>
  [ObservableProperty]
  private bool _dustTrajectories = false;

  partial void OnDustTrajectoriesChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.DustTrajectories = value;
    PushDustViewFlags();
  }

  /// <summary>
  /// Coma 1/ρ normalisation: the dust brightness is divided by the distance from the nucleus so
  /// the inner coma and the outer jets share one exposure (native view flag bit 3). Stored in the
  /// model session; display only, the simulated dust is unchanged.
  /// </summary>
  [ObservableProperty]
  private bool _dustRhoNorm = false;

  partial void OnDustRhoNormChanged(bool value)
  {
    if (CurrentSession is { } session)
      session.DustRhoNorm = value;
    PushDustViewFlags();
  }

  /// <summary>
  /// Dust exposure compensation in photographic stops over the automatic exposure, in
  /// [<see cref="DustExposureEvMin"/>, <see cref="DustExposureEvMax"/>]. Stored in the model
  /// session; visual only.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(DustExposureEvText))]
  private double _dustExposureEv = 0.0;

  public const double DustExposureEvMin = -6.0;
  public const double DustExposureEvMax = 6.0;

  /// <summary>The exposure as shown next to its slider, e.g. "+1.25 EV".</summary>
  public string DustExposureEvText => FormatDustExposureEv(DustExposureEv);

  internal static string FormatDustExposureEv(double ev) =>
    (ev > 0 ? "+" : ev < 0 ? "-" : "") + Math.Abs(ev).ToString("0.00", System.Globalization.CultureInfo.InvariantCulture) + " EV";

  partial void OnDustExposureEvChanged(double value)
  {
    var clamped = ClampDustExposureEv(value);
    if (clamped != value)
    {
      DustExposureEv = clamped;
      return;
    }
    if (CurrentSession is { } session)
      session.DustExposureEv = value;
    _runtimeService?.SetDustExposureEv((float)value);
  }

  /// <summary>Native range of the dust exposure compensation (±6 EV); NaN is 0.</summary>
  internal static double ClampDustExposureEv(double ev) =>
    double.IsNaN(ev) ? 0.0 : Math.Min(Math.Max(ev, DustExposureEvMin), DustExposureEvMax);

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
  /// Native dust view flags (<c>dust::DUST_VIEW_*</c>): bit 0 tracers, bit 1 flow pulses, bit 2
  /// trajectory lines, bit 3 coma 1/ρ normalisation (display only).
  /// </summary>
  internal static uint DustViewFlags(bool tracers, bool flow, bool trajectories, bool rhoNorm) =>
    (tracers ? 1u : 0u) | (flow ? 2u : 0u) | (trajectories ? 4u : 0u) | (rhoNorm ? 8u : 0u);

  private void PushDustViewFlags() =>
    _runtimeService?.SetDustViewFlags(DustViewFlags(DustTracers, DustFlow, DustTrajectories, DustRhoNorm));

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
        // the site may have moved into (or out of) the polar night
        JetNativeSync.RefreshLitWarning(_runtimeService, jet, LitWarningFormat);
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

    // the Σ Q_d read-out follows every jet's Q_d₀
    Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => jet.PropertyChanged += h,
        h => jet.PropertyChanged -= h)
      .Subscribe(e => OnAnyJetActivityChanged(e.EventArgs.PropertyName))
      .AddDisposableTo(_disposables);
  }
}
