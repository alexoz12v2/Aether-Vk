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
    IModelTabViewModel,
    IRecipient<AetherVk.Logic.Messages.NucleusRadiusKnownMessage>
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
    IPlatformWindowService platformWindowService)
    : base("Model", sessionService, cometMessenger)
  {
    _translationService = translationService;
    _cometSessionService = cometSessionService;
    _cometConfigService = cometConfigService;
    _schedulerProvider = schedulerProvider;
    _runtimeService = runtimeService;
    _dispatcher = dispatcher;
    _platformWindowService = platformWindowService;
    Icon = "⬡"; // hexagon / 3D object — U+2B21
    SubscribeToStrings(schedulerProvider);

    // Track _modelChangeSub lifetime alongside all other subs.
    _disposables.Add(_modelChangeSub);

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

    IsActive = true;  // → OnActivated() → registers NucleusRadiusKnownMessage
  }

  protected override void OnActivated()
  {
    Messenger.Register<ModelTabViewModel, AetherVk.Logic.Messages.NucleusRadiusKnownMessage>(this, (r, m) => r.Receive(m));
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

  /// <inheritdoc />
  public void Receive(AetherVk.Logic.Messages.NucleusRadiusKnownMessage message)
  {
    // Re-evaluate AddJetCommand.CanExecute on the UI thread when Horizon radius arrives.
    _dispatcher.Dispatch(() =>
    {
      AddJetCommand.NotifyCanExecuteChanged();
      OnPropertyChanged(nameof(IsNucleusRadiusUnknown));
    });
  }

  /// <summary>
  /// Raised automatically by the MVVM toolkit when <see cref="ManualNucleusRadiusKm"/> changes.
  /// Keeps <see cref="ManualNucleusRadiusKmNullable"/> and <see cref="IsNucleusRadiusUnknown"/> in sync.
  /// </summary>
  partial void OnManualNucleusRadiusKmChanged(float value)
  {
    OnPropertyChanged(nameof(ManualNucleusRadiusKmNullable));
    OnPropertyChanged(nameof(IsNucleusRadiusUnknown));
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
  private void SubscribeJetChanges(JetViewModel jet, ModelSession session)
  {
    Observable
      .FromEventPattern<PropertyChangedEventHandler, PropertyChangedEventArgs>(
        h => jet.PropertyChanged += h,
        h => jet.PropertyChanged -= h)
      .Where(e =>
        e.EventArgs.PropertyName != nameof(JetViewModel.Beta) &&
        e.EventArgs.PropertyName != nameof(JetViewModel.DustProductionRateAt1AuKgs))
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
  }
}
