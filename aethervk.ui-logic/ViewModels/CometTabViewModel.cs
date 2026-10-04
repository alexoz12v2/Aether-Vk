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
  private double _rotRateDegDay;

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
    ITabFactory tabFactory
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
    SubscribeToStrings(schedulerProvider);
    WireReactiveSubscriptions(schedulerProvider);
  }

  [ObservableProperty]
  private bool _isSimulationRunning;

  private bool _snapshotExists;

  // ── Reactive wiring ───────────────────────────────────────────────────────

  private TimeRange? _currentProposedTimeRange;

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
          ProposedStartEpoch = FormatTaiEpoch(range.StartCenturies, range.StartNs);
          ProposedEndEpoch = FormatTaiEpoch(range.EndCenturies, range.EndNs);

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
      }
    }
    finally
    {
      _restoringFromSession = false;
    }
  }

  // ── Property change override for rotational model debounce ───────────────

  protected override void OnPropertyChanged(PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

    // tabs are scoped: a reopened tab gets a new VM over the same session
    if (e.PropertyName == nameof(CurrentSession) && CurrentSession is { } session)
      RestoreFromSession(session);

    bool isRotProp =
      e.PropertyName
      is nameof(PoleRaDeg)
        or nameof(PoleDecDeg)
        or nameof(PrimeMeridianDeg)
        or nameof(PoleRaRateDegCen)
        or nameof(PoleDecRateDegCen)
        or nameof(RotRateDegDay)
        or nameof(BodyFixedOrientationIndex);

    bool isStateChangingProp = isRotProp || e.PropertyName is nameof(SelectedComet) or nameof(SelectedSpkRecord);

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
    if (e.PropertyName == nameof(SelectedComet) && SelectedComet is not null)
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
    );

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
      string fileName = string.Concat("spk_", sanitized, "_", SelectedSpkRecord.RecordId, ".bsp");
      string savePath = _storage.GetDownloadsPath(fileName);

      // Use stored display strings for date parsing (ISO prefix)
      string startStr =
        ProposedStartEpoch.Length >= 10 ? ProposedStartEpoch.Substring(0, 10) : "2020-01-01";
      string endStr =
        ProposedEndEpoch.Length >= 10 ? ProposedEndEpoch.Substring(0, 10) : "2026-01-01";

      DownloadStatus = string.Concat("Downloading SPK for ", SelectedComet.Name, "…");

      string? filePath = await _jpl.DownloadSpkByIdAsync(
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

      DownloadStatus = "Committing to simulation…";

      // Decommit old almanac if any
      if (IsAlmanacCommitted)
        _cometConfig.DecommitComet();

      // Auto-commit timeline before loading SPK
      if (_currentProposedTimeRange is not null)
      {
        _timeline.RequestEpochRange(_currentProposedTimeRange);
      }

      // Commit the new SPK — sbData carries the SBDB Keplerian elements for orbit track generation
      bool committed = await _cometConfig.CommitCometAsync(
        filePath, naifId, _currentProposedTimeRange!, sbData, ReferenceOrbitMode ?? Models.ReferenceOrbitMode.Sbdb);

      if (committed)
      {
        CommittedSpkRecordId = SelectedSpkRecord.RecordId;
        _cometMessenger.Send(new Messages.CometCommittedMessage());

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
          // Best-effort — user can enter radius manually in Model tab
          _ = _breadcrumbService.ShowMessageAsync(
            "Nucleus Radius",
            "SPK committed, but nucleus radius fetch failed. Please configure it manually in the Model tab.",
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

  /// <summary>Formats a TAI epoch (centuries + ns) as a UTC display string.</summary>
  private static string FormatTaiEpoch(short centuries, ulong nanoseconds)
  {
    try
    {
      long ticksPerCentury = TimeSpan.TicksPerDay * 36525L;
      long ticks = (long)centuries * ticksPerCentury + (long)(nanoseconds / 100UL);
      var dt = new DateTimeOffset(2000, 1, 1, 12, 0, 0, TimeSpan.Zero).AddTicks(ticks);
      return dt.ToString("yyyy-MM-dd HH:mm");
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
