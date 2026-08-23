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
  private readonly IUiThreadDispatcher _dispatcher;
  private readonly CompositeDisposable _disposables = [];

  // ── Observable properties ───────────────────────────────────────────────────

  /// <summary>The currently selected jet, or <c>null</c> when none is selected.</summary>
  [ObservableProperty]
  private JetViewModel? _selectedJet;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(AddJetCommand))]
  private float _manualNucleusRadiusKm;

  // ── Construction ────────────────────────────────────────────────────────────

  public ModelTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<ModelSession> sessionService,
    ITabStateService<CometSession> cometSessionService,
    INativeRuntimeService runtimeService,
    IUiThreadDispatcher dispatcher)
    : base("Model", sessionService)
  {
    _translationService = translationService;
    _cometSessionService = cometSessionService;
    _runtimeService = runtimeService;
    _dispatcher = dispatcher;
    Icon = "⬡"; // hexagon / 3D object — U+2B21
    SubscribeToStrings(schedulerProvider);
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.NucleusRadiusKnownMessage>(this);
  }

  // ── Session passthrough ──────────────────────────────────────────────────────

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

  private bool CanAddJet() => EffectiveNucleusRadiusKm > 0f;

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

    var model = BuildModel(session);
    var psJet = BuildJet(jet);

    // Attempt to register with native runtime.
    // If this returns false (e.g. simulation is running or comet not yet spawned),
    // the jet is still added to the UI list — the user can retry once conditions are met.
    // TODO: surface error state in UI (breadcrumb is emitted on Rust side on failure).
    bool ok;
    if (session.Jets.Count == 0)
    {
      // First jet: request computed properties for immediate beta display
      var computed = _runtimeService.AddFirstParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
      if (computed is not null)
      {
        jet.Beta = computed.Beta;
        jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
      }
      ok = psId != 0;
    }
    else
    {
      ok = _runtimeService.AddParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
    }

    session.Jets.Add(jet);
    SelectedJet = jet;

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

  // ── Localization helper ──────────────────────────────────────────────────────

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

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
      .Throttle(TimeSpan.FromMilliseconds(250))
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
