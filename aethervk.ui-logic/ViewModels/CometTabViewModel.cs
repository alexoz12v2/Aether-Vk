using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

[GenerateLocalizedStrings(keyPrefix: "Tabs_Comet_", designTitle: "Comet", designIcon: "☄")]
public partial class CometTabViewModel : StatefulTabViewModelBase<CometSession>, ICometTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly HorizonJplService _jpl;
  private readonly CometConfigService _cometConfig;
  private readonly TimelineService _timeline;
  private readonly ILocalStorageService _storage;
  private readonly CompositeDisposable _disposables = [];
  private readonly BreadcrumbService _breadcrumbService;
  private readonly ICometMessenger _cometMessenger;

  // ISO strings of the proposed range at the moment the comet was committed.
  // Used to detect "proposed timeline changed after comet commit".
  // Stored as strings (not TAI TimeRange) to avoid nanosecond rounding
  // false-positives when comparing against the ProposedTimeRange stream.
  private (string Start, string End)? _lastCommittedProposedRange;

  // Pending rotational model debounce timer
  private IDisposable? _rotDebounceToken;

  // Debounces manual nucleus radius edits (250 ms) so a slider drag does not flood the FFI.
  private readonly System.Reactive.Subjects.Subject<float> _radiusChanges = new();

  // ── Nucleus radius (feedback item 10b: moved here from the Model tab) ──────

  /// <summary>
  /// Manual nucleus radius in km (0 = use the catalogue radius). Lives in
  /// <see cref="ModelSession.ManualNucleusRadiusKm"/>, so saves are unchanged; an edit is pushed
  /// (debounced) through <see cref="CometConfigService.SetNucleusRadiusKm"/>, and after Play it
  /// restores the snapshot like every other comet edit.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ManualNucleusRadiusKmNullable), nameof(IsNucleusRadiusUnknown))]
  private float _manualNucleusRadiusKm = 2.0f;

  partial void OnManualNucleusRadiusKmChanged(float value)
  {
    // restored values are what the session and the native runtime already have
    if (_restoringFromSession) return;
    if (GetModelSession() is { } model)
      model.ManualNucleusRadiusKm = value;
    _radiusChanges.OnNext(EffectiveNucleusRadiusKm);
  }

  /// <inheritdoc cref="ICometTabViewModel.ManualNucleusRadiusKmNullable"/>
  public float? ManualNucleusRadiusKmNullable
  {
    get => ManualNucleusRadiusKm > 0f ? ManualNucleusRadiusKm : null;
    set => ManualNucleusRadiusKm = value ?? 0f;
  }

  private float EffectiveNucleusRadiusKm =>
    ManualNucleusRadiusKm > 0f ? ManualNucleusRadiusKm : (CurrentSession?.NucleusRadiusKm ?? 0f);

  public bool IsNucleusRadiusUnknown => EffectiveNucleusRadiusKm == 0f;

  private ModelSession? GetModelSession() =>
    _modelSessionService is { ActiveSessionIds.Count: > 0 } svc ? svc.GetSession(svc.ActiveSessionIds[0]) : null;

  private ModelSession? _modelRestoredHooked;

  /// <summary>Reads the radius from the model session, without a push or a snapshot restore.</summary>
  private void LoadRadiusFromModelSession()
  {
    var model = GetModelSession();
    if (!ReferenceEquals(_modelRestoredHooked, model))
    {
      if (_modelRestoredHooked is not null) _modelRestoredHooked.Restored -= OnModelSessionRestored;
      if (model is not null) model.Restored += OnModelSessionRestored;
      _modelRestoredHooked = model;
    }
    if (model is null) return;
    bool was = _restoringFromSession;
    _restoringFromSession = true;
    try { ManualNucleusRadiusKm = model.ManualNucleusRadiusKm; }
    finally { _restoringFromSession = was; }
  }

  private void OnModelSessionRestored()
  {
    // a savefile open / New replaced the scene: the Play snapshot belongs to the old one
    _snapshotExists = false;
    LoadRadiusFromModelSession();
  }

  // ── Observable properties — Proposed Timeline (read-only) ─────────────────

  [ObservableProperty]
  private string _proposedStartEpoch = string.Empty;

  [ObservableProperty]
  private string _proposedEndEpoch = string.Empty;

  [ObservableProperty]
  private bool _hasProposedTimeline;

  /// <summary>
  /// Reference orbit choice for the commit, as a ComboBox index: -1 = nothing chosen yet (the
  /// commit stays disabled, so the user reads the options at least once), 0 = SBDB solution,
  /// 1 = re-osculated at the start epoch. See <see cref="Models.ReferenceOrbitMode"/>.
  /// </summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ReferenceOrbitMode))]
  private int _referenceOrbitIndex = -1;

  partial void OnReferenceOrbitIndexChanged(int value)
  {
    if (CurrentSession is { } session)
      session.ReferenceOrbitIndex = value;
  }

  /// <summary>The chosen reference orbit, or null while nothing is selected.</summary>
  public Models.ReferenceOrbitMode? ReferenceOrbitMode => ReferenceOrbitIndex switch
  {
    0 => Models.ReferenceOrbitMode.Sbdb,
    1 => Models.ReferenceOrbitMode.OsculatingAtStart,
    _ => null,
  };

  /// <summary>The reference orbit can only be chosen while no comet is committed.</summary>
  public bool IsReferenceOrbitSelectable => !IsAlmanacCommitted;

  // ── Observable properties — Comet Search ─────────────────────────────────

  [ObservableProperty]
  private string _searchQuery = string.Empty;

  [ObservableProperty]
  private bool _isSearching;

  [ObservableProperty]
  private CometSearchResult? _selectedComet;

  // ── Observable properties — SPK records ──────────────────────────────────

  [ObservableProperty]
  private bool _isLoadingSpkRecords;

  [ObservableProperty]
  private SpkRecordItem? _selectedSpkRecord;

  // ── Observable properties — Commit state ─────────────────────────────────

  [ObservableProperty]
  private bool _isAlmanacCommitted;

  [ObservableProperty]
  private string _committedCometName = string.Empty;

  [ObservableProperty]
  private string _downloadStatus = string.Empty;

  [ObservableProperty]
  private bool _isDownloading;

  [ObservableProperty]
  private bool _hasTimelineChangedAfterCommit;

  [ObservableProperty]
  private string _committedSpkRecordId = string.Empty;

  // ── Observable properties — Rotational model ─────────────────────────────

  [ObservableProperty]
  private double _poleRaDeg;

  [ObservableProperty]
  private double _poleDecDeg = 90.0;

  [ObservableProperty]
  private double _primeMeridianDeg;

  [ObservableProperty]
  private double _poleRaRateDegCen;

  [ObservableProperty]
  private double _poleDecRateDegCen;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(RotationPeriodHours))]
  private double _rotRateDegDay;

  /// <summary>
  /// Sidereal rotation period in hours, the same quantity as <see cref="RotRateDegDay"/> (period =
  /// 360° / rate × 24 h); setting it keeps the rate's sign (prograde / retrograde). 0 = no rotation.
  /// </summary>
  public double RotationPeriodHours
  {
    get => RotRateDegDay == 0 ? 0 : Math.Round(360.0 * 24.0 / Math.Abs(RotRateDegDay), 6);
    set
    {
      if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0)
        return;
      if (Math.Abs(value - RotationPeriodHours) < 1e-9)
        return;
      RotRateDegDay = (RotRateDegDay < 0 ? -1 : 1) * 360.0 * 24.0 / value;
    }
  }

  [ObservableProperty]
  private int _bodyFixedOrientationIndex = 0; // 0 = Yes, 1 = No

  // ── Collections from JPL service ─────────────────────────────────────────

  /// <summary>
  /// Filtered view of <see cref="HorizonJplService.CometsData"/> based on
  /// <see cref="SearchQuery"/>. Updated whenever the query or the source list changes.
  /// </summary>
  [ObservableProperty]
  private ObservableCollection<CometSearchResult> _filteredSearchResults = [];

  /// <summary>SPK records for the selected comet from the JPL service (bound directly).</summary>
  public ObservableCollection<SpkRecordItem> SpkRecords => _jpl.SpkRecordsData;

  // ── Debug Properties ─────────────────────────────────────────────────────

  public ObservableCollection<JetViewModel>? DebugJets
  {
    get
    {
#if DEBUG
      if (_modelSessionService.ActiveSessionIds.Count == 0)
        return null;
      return _modelSessionService.GetSession(_modelSessionService.ActiveSessionIds[0])?.Jets;
#else
      return null;
#endif
    }
  }

  // ── Preview Viewports ────────────────────────────────────────────────────
  public Viewport3DViewModel PreviewPerspective { get; }
  public Viewport3DViewModel PreviewOrthographic { get; }

  // ── Dependencies ─────────────────────────────────────────────────────────

  private readonly INativeRuntimeService _runtimeService;
  private readonly ITabStateService<ModelSession> _modelSessionService;

  // ── Construction ─────────────────────────────────────────────────────────

  public CometTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<CometSession> sessionService,
    HorizonJplService jpl,
    CometConfigService cometConfig,
    TimelineService timeline,
    BreadcrumbService breadcrumbService,
    ILocalStorageService storage,
    INativeRuntimeService runtimeService,
    ITabStateService<ModelSession> modelSessionService,
    ICometMessenger cometMessenger,
    ITabFactory tabFactory,
    ITabStateService<TimelineSession>? timelineSessions = null,
    TimeScaleService? timeScale = null
  )
    : base("Comet", sessionService)
  {
    _translationService = translationService;
    _jpl = jpl;
    _cometConfig = cometConfig;
    _timeline = timeline;
    _storage = storage;
    _breadcrumbService = breadcrumbService;
    _runtimeService = runtimeService;
    _modelSessionService = modelSessionService;
    _cometMessenger = cometMessenger;

    PreviewPerspective = (Viewport3DViewModel)tabFactory.CreateScopedTab(typeof(Viewport3DViewModel)).ViewModel!;
    PreviewOrthographic = (Viewport3DViewModel)tabFactory.CreateScopedTab(typeof(Viewport3DViewModel)).ViewModel!;

    Icon = "☄"; // comet — U+2604

    // the range is entered here (reviewer feedback item 7); the Timeline tab only shows it
    Range = new RangeEntryViewModel(timelineSessions, timeline, cometMessenger, () => RotRateDegDay, timeScale);
    Range.PropertyChanged += OnRangePropertyChanged;
    UpdateGuidedStep();

    SubscribeToStrings(schedulerProvider);
    WireReactiveSubscriptions(schedulerProvider);

    _disposables.Add(_radiusChanges);
    _radiusChanges
      .Throttle(TimeSpan.FromMilliseconds(250), schedulerProvider.MainThread)
      .Subscribe(r => _cometConfig.SetNucleusRadiusKm(r))
      .AddDisposableTo(_disposables);
    LoadRadiusFromModelSession();
    // the base constructor's restore ran before the services existed: redo it for the selection
    if (CurrentSession is { } restoredSession)
      RestoreFromSession(restoredSession);
    // the catalogue radius (fetched on commit) can clear the "radius unknown" hint
    _cometConfig.NucleusRadiusKm
      .Subscribe(_ => OnPropertyChanged(nameof(IsNucleusRadiusUnknown)))
      .AddDisposableTo(_disposables);
  }

  [ObservableProperty]
  private bool _isSimulationRunning;

  private bool _snapshotExists;

  // ── Reactive wiring ───────────────────────────────────────────────────────

  private TimeRange? _currentProposedTimeRange;

  /// <summary>The range Download &amp; Commit applies (RequestEpochRange, CommitCometAsync): the latest proposal.</summary>
  internal TimeRange? ProposedRangeForCommit => _currentProposedTimeRange;

  private void WireReactiveSubscriptions(ISchedulerProvider schedulerProvider)
  {
    _timeline.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(running =>
      {
          IsSimulationRunning = running;
          if (running) _snapshotExists = true;
      })
      .AddDisposableTo(_disposables);

    // 1. Proposed timeline display from TimelineService
    _timeline
      .ProposedTimeRange.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(range =>
      {
        _currentProposedTimeRange = range;
        if (range is null)
        {
          HasProposedTimeline = false;
          ProposedStartEpoch = ProposedEndEpoch = string.Empty;
        }
        else
        {
          HasProposedTimeline = true;
          ProposedStartEpoch = FormatProposedEpoch(range.StartCenturies, range.StartNs);
          ProposedEndEpoch = FormatProposedEpoch(range.EndCenturies, range.EndNs);

          // Detect change-after-commit by comparing ISO display strings — avoids
          // nanosecond rounding false-positives that occur with TAI record equality.
          if (
            IsAlmanacCommitted
            && _lastCommittedProposedRange is { } snap
            && (ProposedStartEpoch != snap.Start || ProposedEndEpoch != snap.End)
          )
            HasTimelineChangedAfterCommit = true;
        }
      })
      .AddDisposableTo(_disposables);

    // "Recommit with this range" (the range section here, while locked): the same Download & Commit, with the proposed range
    _timeline.RecommitRequests
      .Subscribe(async request =>
      {
        try
        {
          if (DownloadAndCommitCommand.CanExecute(null))
          {
            _lastCommitSucceeded = false;
            await DownloadAndCommitCommand.ExecuteAsync(null);
            // the commit's own outcome: IsAlmanacCommitted and the range lock follow it through
            // main-thread posts, so right after the await they can still show the decommit
            request.Complete(_lastCommitSucceeded);
          }
          else
          {
            // a request with nothing to commit (the same range again) counts as committed with it
            request.Complete(IsAlmanacCommitted && !Range.IsRangeLocked);
          }
        }
        catch
        {
          request.Complete(false);
        }
      })
      .AddDisposableTo(_disposables);

    // 2. Almanac committed state from CometConfigService
    _cometConfig
      .IsAlmanacCommitted.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(committed =>
      {
        IsAlmanacCommitted = committed;
        if (!committed)
        {
          CommittedCometName = string.Empty;
          HasTimelineChangedAfterCommit = false;
        }
        else
        {
          PushRotationalModel();
        }
      })
      .AddDisposableTo(_disposables);
  }

  // ── Session restore ───────────────────────────────────────────────────────

  private bool _restoringFromSession;
  private CometSession? _restoredHooked;

  /// <summary>A savefile open replaces the session's values: re-read them (see <see cref="CometSession.Restored"/>).</summary>
  private void HookRestored(CometSession? session)
  {
    if (ReferenceEquals(_restoredHooked, session)) return;
    if (_restoredHooked is not null) _restoredHooked.Restored -= OnSessionRestored;
    if (session is not null) session.Restored += OnSessionRestored;
    _restoredHooked = session;
  }

  private void OnSessionRestored()
  {
    // a savefile open / New replaced the scene: the Play snapshot belongs to the old one
    _snapshotExists = false;
    if (CurrentSession is { } session)
      RestoreFromSession(session);
  }

  public override void Dispose()
  {
    HookRestored(null);
    if (Range is not null)
    {
      Range.PropertyChanged -= OnRangePropertyChanged;
      Range.Dispose();
    }
    if (_modelRestoredHooked is not null) _modelRestoredHooked.Restored -= OnModelSessionRestored;
    _modelRestoredHooked = null;
    // a closed tab must stop reacting (it pushed its stale rotation on every later commit)
    _rotDebounceToken?.Dispose();
    _disposables.Dispose();
    base.Dispose();
  }

  /// <summary>
  /// Reloads the form from <paramref name="session"/>: tabs are scoped, so reopening the Comet tab
  /// builds a new VM over the same session. Runs from the base constructor too (fields such as
  /// <c>_timeline</c> are still null there), hence the side-effect guard in OnPropertyChanged.
  /// </summary>
  private void RestoreFromSession(CometSession session)
  {
    _restoringFromSession = true;
    try
    {
      ReferenceOrbitIndex = session.ReferenceOrbitIndex;
      PoleRaDeg = session.RotPoleRaDeg;
      PoleDecDeg = session.RotPoleDecDeg;
      PrimeMeridianDeg = session.RotPrimeMeridianDeg;
      PoleRaRateDegCen = session.RotPoleRaRateDegCen;
      PoleDecRateDegCen = session.RotPoleDecRateDegCen;
      RotRateDegDay = session.RotRateDegDay;
      BodyFixedOrientationIndex = session.RotBodyFixedOrientationIndex;
      if (session.IsAlmanacLoaded)
      {
        CommittedCometName = session.CommittedFullName;
        CommittedSpkRecordId = session.CommittedSpkRecordId;
        RestoreSelection(session);
      }
    }
    finally
    {
      _restoringFromSession = false;
    }
  }

  /// <summary>
  /// Rebuilds the comet / SPK record selection of a reopened save from what the commit stored, so
  /// Download &amp; Commit (and the recommit of a range change) work without a new search. The
  /// restored items are also put into the collections the combo boxes bind to. No network, no
  /// commit. Needs <c>_jpl</c>, so the base-constructor pass skips it (the constructor repeats the
  /// restore); an old save without the fields keeps the selection empty.
  /// </summary>
  private void RestoreSelection(CometSession session)
  {
    if (_jpl is null) return;
    string designation = session.CommittedDesignation;
    string recordId = session.CommittedSpkRecordId;
    string fullName = session.CommittedFullName;
    if (string.IsNullOrEmpty(designation) || string.IsNullOrEmpty(recordId) || string.IsNullOrEmpty(fullName))
      return;

    // the comet: the search result of the same comet when the list holds one, else an equivalent one
    CometSearchResult? comet = null;
    foreach (var c in _jpl.CometsData)
      if (c.PrimaryDesignation == designation && c.Name == fullName) { comet = c; break; }
    if (comet is null)
    {
      comet = new CometSearchResult { Name = fullName, PrimaryDesignation = designation };
      _jpl.CometsData.Add(comet);
    }
    ApplySearchFilter();
    if (!FilteredSearchResults.Contains(comet))
      FilteredSearchResults.Add(comet);

    // the record, as HorizonJplService.ParseSpkRecordsText builds it
    bool single = recordId.StartsWith(HorizonJplService.DesignationRecordPrefix, StringComparison.Ordinal);
    var record = new SpkRecordItem
    {
      RecordId = recordId,
      EpochYear = single ? "(only record)" : string.Empty,
      MatchDesig = designation,
      PrimaryDesig = designation,
      Name = single && session.SpkId is { } spk ? $"{designation} (SPK {spk})" : fullName,
    };
    _jpl.SpkRecordsData.Clear();
    _jpl.SpkRecordsData.Add(record);

    SelectedComet = comet;
    SelectedSpkRecord = record;
  }

  // ── Property change override for rotational model debounce ───────────────

  protected override void OnPropertyChanged(PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

    // tabs are scoped: a reopened tab gets a new VM over the same session
    if (e.PropertyName == nameof(CurrentSession))
    {
      HookRestored(CurrentSession);
      if (CurrentSession is { } session)
        RestoreFromSession(session);
    }

    bool isRotProp =
      e.PropertyName
      is nameof(PoleRaDeg)
        or nameof(PoleDecDeg)
        or nameof(PrimeMeridianDeg)
        or nameof(PoleRaRateDegCen)
        or nameof(PoleDecRateDegCen)
        or nameof(RotRateDegDay)
        or nameof(BodyFixedOrientationIndex);

    bool isStateChangingProp = isRotProp
      || e.PropertyName is nameof(SelectedComet) or nameof(SelectedSpkRecord) or nameof(ManualNucleusRadiusKm);

    // restored values are what the native runtime already has: no snapshot restore, no push
    if (_restoringFromSession)
      isRotProp = isStateChangingProp = false;

    if (isStateChangingProp && !_timeline.IsSimulationRunningValue && _snapshotExists)
    {
        _timeline.SnapshotRestore();
        _snapshotExists = false;
    }


    if (isRotProp && IsAlmanacCommitted)
    {
      // Debounce: cancel previous and schedule a new push after 250 ms
      _rotDebounceToken?.Dispose();
      _rotDebounceToken = Observable
        .Timer(TimeSpan.FromMilliseconds(250))
        .Subscribe(_ => PushRotationalModel());
    }

    // Re-filter whenever the search query changes
    if (e.PropertyName == nameof(SearchQuery))
      ApplySearchFilter();

    // Auto-load SPK records when a comet is selected
    if (e.PropertyName == nameof(SelectedComet) && SelectedComet is not null && !_restoringFromSession)
      _ = LoadSpkRecordsAsync();

    if (
      e.PropertyName
      is nameof(SelectedComet)
        or nameof(SelectedSpkRecord)
        or nameof(HasProposedTimeline)
        or nameof(IsAlmanacCommitted)
        or nameof(CommittedCometName)
        or nameof(CommittedSpkRecordId)
        or nameof(HasTimelineChangedAfterCommit)
        or nameof(ReferenceOrbitIndex)
    )
    {
      DownloadAndCommitCommand.NotifyCanExecuteChanged();
    }
    if (e.PropertyName == nameof(IsAlmanacCommitted))
      OnPropertyChanged(nameof(IsReferenceOrbitSelectable));

    if (e.PropertyName
        is nameof(FilteredSearchResults)
          or nameof(SelectedComet)
          or nameof(SelectedSpkRecord)
          or nameof(IsAlmanacCommitted))
      UpdateGuidedStep();
    // a culture change re-reads the step texts
    if (e.PropertyName is { } n && n.StartsWith("StrStep", StringComparison.Ordinal))
      OnPropertyChanged(nameof(CurrentStepHelp));
  }

  // ── Range entry (reviewer feedback item 7) ────────────────────────────────

  /// <summary>
  /// The simulated range, entered here: Start/End, the Model range, the dust lead time, and the
  /// lock + Recommit after a commit. The Timeline tab shows the committed range read-only.
  /// </summary>
  public RangeEntryViewModel Range { get; }

  private void OnRangePropertyChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is nameof(RangeEntryViewModel.HasValidRange) or nameof(RangeEntryViewModel.IsRangeLocked))
      UpdateGuidedStep();
    if (e.PropertyName is nameof(RangeEntryViewModel.IsRangeLocked))
      DownloadAndCommitCommand.NotifyCanExecuteChanged();
  }

  // ── Guided flow (reviewer feedback item 3): 1 Search → 2 SPK record → 3 Range → 4 Commit ──

  /// <summary>
  /// The step the user is on (1 to 4), 5 once committed with the range in the boxes. A committed
  /// comet counts steps 1 to 3 as done (a reopened save has no search results); an edited range
  /// after a commit makes step 4 (Recommit) current again, an invalid one step 3.
  /// </summary>
  internal static int GuidedStepFor(bool hasSearchResults, bool hasComet, bool hasRecord, bool hasValidRange,
                                    bool committed, bool rangeLocked)
  {
    if (committed)
      return !hasValidRange ? 3 : rangeLocked ? 4 : 5;
    if (!hasSearchResults && !hasComet) return 1;
    if (!hasComet || !hasRecord) return 2;
    if (!hasValidRange) return 3;
    return 4;
  }

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsStep1Done), nameof(IsStep2Done), nameof(IsStep3Done), nameof(IsStep4Done),
    nameof(IsStep1Current), nameof(IsStep2Current), nameof(IsStep3Current), nameof(IsStep4Current),
    nameof(CurrentStepHelp))]
  private int _currentGuidedStep = 1;

  public bool IsStep1Done => CurrentGuidedStep > 1;
  public bool IsStep2Done => CurrentGuidedStep > 2;
  public bool IsStep3Done => CurrentGuidedStep > 3;
  public bool IsStep4Done => CurrentGuidedStep > 4;
  public bool IsStep1Current => CurrentGuidedStep == 1;
  public bool IsStep2Current => CurrentGuidedStep == 2;
  public bool IsStep3Current => CurrentGuidedStep == 3;
  public bool IsStep4Current => CurrentGuidedStep == 4;

  /// <summary>The one-line help of the current step (all done: what to do next).</summary>
  public string CurrentStepHelp => CurrentGuidedStep switch
  {
    1 => StrStep1Help,
    2 => StrStep2Help,
    3 => StrStep3Help,
    4 => StrStep4Help,
    _ => StrStepsDoneHelp,
  };

  private void UpdateGuidedStep()
  {
    // runs from the base constructor too, before Range exists
    if (Range is null) return;
    CurrentGuidedStep = GuidedStepFor(
      FilteredSearchResults is { Count: > 0 },
      SelectedComet is not null,
      SelectedSpkRecord is not null,
      Range.HasValidRange,
      IsAlmanacCommitted,
      Range.IsRangeLocked);
  }

  /// <summary>
  /// Rebuilds <see cref="FilteredSearchResults"/> from <c>_jpl.CometsData</c>
  /// filtered by the current <see cref="SearchQuery"/> (case-insensitive contains
  /// on <c>Name</c> or <c>PrimaryDesignation</c>). An empty query shows all results.
  /// </summary>
  private void ApplySearchFilter()
  {
    var query = SearchQuery?.Trim() ?? string.Empty;

    var temp = new System.Collections.Generic.List<CometSearchResult>();
    foreach (var comet in _jpl.CometsData)
    {
      if (
        query.Length == 0
        || comet.Name.Contains(query, StringComparison.OrdinalIgnoreCase)
        || comet.PrimaryDesignation.Contains(query, StringComparison.OrdinalIgnoreCase)
      )
        temp.Add(comet);
    }
    FilteredSearchResults = new ObservableCollection<CometSearchResult>(temp);
  }

  // ── Commands ──────────────────────────────────────────────────────────────

  [RelayCommand]
  private async Task SearchCometsAsync()
  {
    IsSearching = true;
    try
    {
      await _jpl.FetchCometsAsync();
      ApplySearchFilter();
    }
    finally
    {
      IsSearching = false;
    }
  }

  /// <summary>
  /// The days (yyyy-MM-dd, UTC) of the SPK download for <paramref name="range"/>: the start's day
  /// and the end's day, so with <see cref="HorizonJplService.SpkRequestDays"/>'s margin the file
  /// always covers the range (feedback #19). The ±5 years around today in
  /// <see cref="LoadSpkRecordsAsync"/> only lists the comet's SPK records; it never sets the range.
  /// </summary>
  internal static (string Start, string Stop) SpkDownloadDays(TimeRange range)
  {
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    return (
      TimeUtils.FromTdbParts(range.StartCenturies, range.StartNs).UtcDateTime.ToString("yyyy-MM-dd", ci),
      TimeUtils.FromTdbParts(range.EndCenturies, range.EndNs).UtcDateTime.ToString("yyyy-MM-dd", ci));
  }

  [RelayCommand]
  private async Task LoadSpkRecordsAsync()
  {
    if (SelectedComet is null)
      return;

    IsLoadingSpkRecords = true;
    try
    {
      var start = DateTime.UtcNow.AddYears(-5).ToString("yyyy-MM-dd");
      var stop = DateTime.UtcNow.AddYears(5).ToString("yyyy-MM-dd");
      await _jpl.FetchSpkRecordsAsync(SelectedComet.PrimaryDesignation, start, stop);
    }
    finally
    {
      IsLoadingSpkRecords = false;
    }
  }

  // ── SPK coverage (reviewer slide 1: another date must not redo the download) ──

  /// <summary>
  /// The window the SPK on disk was requested for (<see cref="HorizonJplService.SpkRequestDays"/>:
  /// the range's days widened by the margin), per comet designation and record.
  /// </summary>
  internal sealed record SpkCoverage(string Designation, string RecordId, string FilePath, DateTimeOffset Start, DateTimeOffset Stop);

  private SpkCoverage? _spkCoverage;

  /// <summary>Test / diagnostics: the coverage the next commit checks.</summary>
  internal SpkCoverage? CurrentSpkCoverage => SpkCoverageNow();

  /// <summary>Tests: the coverage of an SPK already on disk (as a download of this run leaves it).</summary>
  internal void NoteSpkCoverage(SpkCoverage? coverage) => _spkCoverage = coverage;

  /// <summary>The coverage of a download made for the days <paramref name="startDay"/>..<paramref name="stopDay"/> (yyyy-MM-dd).</summary>
  internal static SpkCoverage? SpkCoverageOf(string designation, string recordId, string filePath, string startDay, string stopDay)
  {
    var (a, b) = HorizonJplService.SpkRequestDays(startDay, stopDay);
    var ci = System.Globalization.CultureInfo.InvariantCulture;
    const System.Globalization.DateTimeStyles Utc = System.Globalization.DateTimeStyles.AssumeUniversal | System.Globalization.DateTimeStyles.AdjustToUniversal;
    if (!DateTimeOffset.TryParseExact(a, "yyyy-MM-dd", ci, Utc, out var start)
        || !DateTimeOffset.TryParseExact(b, "yyyy-MM-dd", ci, Utc, out var stop))
      return null;
    return new SpkCoverage(designation, recordId, filePath, start, stop);
  }

  /// <summary>
  /// True when <paramref name="range"/> lies inside <paramref name="coverage"/> for the same
  /// comet and record and the file is still there. Compared as instants: the window's days start
  /// at midnight, so an end later on the window's last day is outside.
  /// </summary>
  internal static bool SpkCovers(SpkCoverage coverage, string designation, string recordId, TimeRange range)
  {
    if (coverage.Designation != designation || coverage.RecordId != recordId)
      return false;
    if (string.IsNullOrEmpty(coverage.FilePath) || !System.IO.File.Exists(coverage.FilePath))
      return false;
    var start = TimeUtils.FromTdbParts(range.StartCenturies, range.StartNs);
    var end = TimeUtils.FromTdbParts(range.EndCenturies, range.EndNs);
    // TDB parts read as UTC digits: a minute of slack covers the 69 s scale offset
    var slack = TimeSpan.FromMinutes(2);
    return start - slack >= coverage.Start && end + slack <= coverage.Stop;
  }

  /// <summary>
  /// The coverage known now: the last download of this run, else (a reopened save) the committed
  /// SPK, which was fetched for at least the committed range widened by the margin.
  /// </summary>
  private SpkCoverage? SpkCoverageNow()
  {
    if (_spkCoverage is not null)
      return _spkCoverage;
    var session = CurrentSession;
    if (!IsAlmanacCommitted || _cometConfig.CommittedInputs is not { } inputs || session is null
        || string.IsNullOrEmpty(session.CommittedSpkRecordId) || string.IsNullOrEmpty(inputs.SpkFilePath))
      return null;
    var (a, b) = SpkDownloadDays(inputs.Range);
    return SpkCoverageOf(session.CommittedDesignation, session.CommittedSpkRecordId, inputs.SpkFilePath, a, b);
  }

  partial void OnRotRateDegDayChanged(double value) => Range?.RefreshModelSpan();

  private bool CanDownloadAndCommit =>
    SelectedComet is not null
    && SelectedSpkRecord is not null
    && HasProposedTimeline
    && ReferenceOrbitMode is not null
    && (
      !IsAlmanacCommitted
      || CommittedCometName != SelectedComet.Name
      || CommittedSpkRecordId != SelectedSpkRecord.RecordId
      || HasTimelineChangedAfterCommit
      || (Range is not null && Range.IsRangeLocked)
    );

  /// <summary>Set by <see cref="DownloadAndCommitAsync"/> when the engine accepted the commit (the recommit request's result).</summary>
  private bool _lastCommitSucceeded;

  [RelayCommand(CanExecute = nameof(CanDownloadAndCommit))]
  private async Task DownloadAndCommitAsync()
  {
    if (!HasProposedTimeline || SelectedComet is null || SelectedSpkRecord is null)
    {
      EmitInvalidStateBreadcrumb();
      return;
    }

    string recordId = SelectedSpkRecord.RecordId;
    IsDownloading = true;
    DownloadStatus = "Fetching NAIF ID…";

    try
    {
      // Resolve NAIF SPK id from SBDB
      var sbData = await _jpl.FetchSmallBodyDataAsync(SelectedComet.PrimaryDesignation);
      if (sbData is null)
      {
        DownloadStatus = "Could not resolve NAIF ID.";
        return;
      }

      int naifId = sbData.SpkId;

      // Build download path (OS Downloads directory)
      string sanitized = SelectedComet.PrimaryDesignation.Replace("/", "_").Replace(" ", "_");
      // a single-record comet's id is "DES=<pdes>" (see HorizonJplService.ParseSpkRecordsText)
      string recordPart = SelectedSpkRecord.RecordId.Replace("=", "_").Replace("/", "_").Replace(" ", "_");
      string fileName = string.Concat("spk_", sanitized, "_", recordPart, ".bsp");
      string savePath = _storage.GetDownloadsPath(fileName);

      // The SPK covers the range the commit applies (the same proposal RequestEpochRange and
      // CommitCometAsync get below); HorizonJplService widens it by SpkMarginDays on each side.
      if (_currentProposedTimeRange is null)
      {
        EmitInvalidStateBreadcrumb();
        return;
      }
      var (startStr, endStr) = SpkDownloadDays(_currentProposedTimeRange);

      string? filePath;
      var coverage = SpkCoverageNow();
      if (coverage is not null && SpkCovers(coverage, SelectedComet.PrimaryDesignation, SelectedSpkRecord.RecordId, _currentProposedTimeRange))
      {
        // the SPK on disk was fetched for a window holding this range: no download (another
        // observation date over a campaign does not redo the fetch)
        DownloadStatus = "The downloaded SPK already covers this range: no download.";
        filePath = coverage.FilePath;
      }
      else
      {
        DownloadStatus = string.Concat("Downloading SPK for ", SelectedComet.Name, "…");
        filePath = await _jpl.DownloadSpkByIdAsync(
          SelectedComet.PrimaryDesignation,
          SelectedSpkRecord.RecordId,
          savePath,
          startStr,
          endStr
        );
        if (filePath is null)
        {
          DownloadStatus = "SPK download failed.";
          return;
        }
        _spkCoverage = SpkCoverageOf(SelectedComet.PrimaryDesignation, SelectedSpkRecord.RecordId, filePath, startStr, endStr);
      }

      DownloadStatus = "Committing to simulation…";

      // Decommit old almanac if any
      if (IsAlmanacCommitted)
        _cometConfig.DecommitComet();

      // Auto-commit timeline before loading SPK. A refused range would leave the engine on the
      // previous one (Run to End then landed on a stale end): never commit on top of it.
      if (_currentProposedTimeRange is null || !_timeline.RequestEpochRange(_currentProposedTimeRange))
      {
        DownloadStatus = "Timeline could not be applied: commit cancelled.";
        return;
      }

      // Commit the new SPK — sbData carries the SBDB Keplerian elements for orbit track generation
      bool committed = await _cometConfig.CommitCometAsync(
        filePath, naifId, _currentProposedTimeRange!, sbData, ReferenceOrbitMode ?? Models.ReferenceOrbitMode.Sbdb);

      if (committed)
      {
        _lastCommitSucceeded = true;
        // the jets now, before a caller seeks (Run at this date): the Model tab's commit
        // subscription posts to the main thread and could add them after the seek, and a jet
        // added while paused gets no dust history until Play. Idempotent with that subscription.
        if (GetModelSession() is { } jetsModel && JetNativeSync.EffectiveNucleusRadiusKm(jetsModel, CurrentSession) > 0f)
          JetNativeSync.EnsureNative(_runtimeService, jetsModel,
            JetNativeSync.EffectiveNucleusRadiusKm(jetsModel, CurrentSession), _timeline.IgnitionUs());
        CommittedSpkRecordId = SelectedSpkRecord.RecordId;
        // the committed range in the timeline session and the boxes, then everyone else
        Range.OnCommitted(_currentProposedTimeRange);
        _cometMessenger.Send(new Messages.CometCommittedMessage(_currentProposedTimeRange));

        // Update session
        var session = CurrentSession;
        if (session is not null)
        {
          session.SpkId = naifId;
          session.CommittedSpkRecordId = CommittedSpkRecordId;
          session.CommittedDesignation = SelectedComet.PrimaryDesignation;
          session.CommittedFullName = SelectedComet.Name;
          session.CommittedSpkFilePath = filePath;
          session.IsAlmanacLoaded = true;
        }

        // Fetch nucleus radius from Horizon constants (best-effort)
        DownloadStatus = "Fetching nucleus radius…";
        try
        {
          // Query by the committed apparition record: the bare designation ("67P;") makes
          // Horizons return the record index instead of data, which used to surface a
          // spurious "No ephemeris data" breadcrumb after a successful commit.
          var (radiusKm, _) = await _jpl.FetchObjectConstantsAsync(recordId);
          if (session is not null && radiusKm > 0.0)
          {
            session.NucleusRadiusKm = (float)radiusKm;
            _cometConfig.SetNucleusRadiusKm(session.NucleusRadiusKm);
          }
        }
        catch
        {
          // Best-effort — user can enter radius manually in the Nucleus section of this tab
          _ = _breadcrumbService.ShowMessageAsync(
            "Nucleus Radius",
            "SPK committed, but nucleus radius fetch failed. Please configure it manually in the Comet tab (Nucleus).",
            status: 3
          );
        }

        CommittedCometName = SelectedComet.Name;

        // Snapshot the proposed range ISO strings at commit time for change detection.
        // Using display strings (not TAI TimeRange) avoids nanosecond rounding false-positives.
        _lastCommittedProposedRange = (ProposedStartEpoch, ProposedEndEpoch);
        HasTimelineChangedAfterCommit = false;
        DownloadStatus = string.Concat("✓ Committed: ", SelectedComet.Name);
      }
      else
      {
        DownloadStatus = "Commit failed. Check logs.";
      }
    }
    catch (Exception ex)
    {
      DownloadStatus = string.Concat("Error: ", ex.Message);
    }
    finally
    {
      IsDownloading = false;
    }
  }

  [RelayCommand]
  private void DecommitComet()
  {
    if (!IsAlmanacCommitted)
      return;

    // Enforce pause invariant: Reset simulation before yanking the comet entity
    if (_timeline.IsSimulationRunningValue || _snapshotExists)
    {
        _timeline.Reset(); 
        _snapshotExists = false;
    }

    _cometConfig.DecommitComet();

    var session = CurrentSession;
    if (session is not null)
    {
      session.SpkId = null;
      session.CommittedSpkRecordId = string.Empty;
      session.CommittedDesignation = string.Empty;
      session.CommittedFullName = string.Empty;
      session.CommittedSpkFilePath = null;
      session.IsAlmanacLoaded = false;
    }

    _lastCommittedProposedRange = null;
    DownloadStatus = string.Empty;
    Range.OnDecommitted();
    _cometMessenger.Send(new Messages.CometDecommittedMessage());
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private void PushRotationalModel()
  {
    var dto = new BodyRotationalModelDto(
      PoleRaDeg,
      PoleDecDeg,
      PrimeMeridianDeg,
      PoleRaRateDegCen,
      PoleDecRateDegCen,
      RotRateDegDay,
      BodyFixedOrientationIndex == 0
    );
    _cometConfig.SetRotationalModel(dto);

    var session = CurrentSession;
    if (session is not null)
    {
      session.RotPoleRaDeg = PoleRaDeg;
      session.RotPoleDecDeg = PoleDecDeg;
      session.RotPrimeMeridianDeg = PrimeMeridianDeg;
      session.RotPoleRaRateDegCen = PoleRaRateDegCen;
      session.RotPoleDecRateDegCen = PoleDecRateDegCen;
      session.RotRateDegDay = RotRateDegDay;
      session.RotBodyFixedOrientationIndex = BodyFixedOrientationIndex;
    }
  }

  /// <summary>Formats an engine epoch (TDB centuries + ns) in the chosen scale (<see cref="RangeEntryViewModel.TimeScale"/>).</summary>
  private string FormatProposedEpoch(short centuries, ulong nanoseconds)
  {
    try
    {
      return Range.TimeScale.Format(centuries, nanoseconds, "yyyy-MM-dd HH:mm");
    }
    catch
    {
      return string.Concat(centuries.ToString(), "c+", nanoseconds.ToString(), "ns");
    }
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService
      .CultureChanged.Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }

  private void EmitInvalidStateBreadcrumb()
  {
    string errorContent = "Unknown Error";
    if (!HasProposedTimeline)
    {
      errorContent = "A Proposed timeline should have been chosen";
    }
    else if (SelectedComet is null)
    {
      errorContent = "A Comet Should have been selected";
    }
    else if (SelectedSpkRecord is null)
    {
      errorContent =
        $"An Observation record for comet {SelectedComet.Name} should have been chosen";
    }
    _ = _breadcrumbService.ShowMessageAsync(
      "Invalid State for Comet Commit",
      errorContent,
      default,
      1
    );
  }

  [RelayCommand]
  private void DebugQueryComet()
  {
#if DEBUG
    if (_runtimeService.CometEntityId.HasValue)
    {
      ulong cometId = _runtimeService.CometEntityId.Value;
      // Component IDs for Almanac Planet (26) and Rotational Body (24), and HighResTransform (1)
      Console.WriteLine($"[CometTabViewModel] Querying Comet Entity ID: {cometId}");
      Console.WriteLine($"[CometTabViewModel] Use this ID in gdb: print-ecs-entity {cometId}");
    }
    else
    {
      Console.WriteLine("[CometTabViewModel] Comet Entity ID is null. The comet might not be initialized.");
    }
#endif
  }

  [RelayCommand]
  private void DebugQueryJet(ulong jetId)
  {
#if DEBUG
    // Component ID for Particle System (22)
#endif
  }
}
