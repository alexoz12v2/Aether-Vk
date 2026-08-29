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

  // ── Collections from JPL service ─────────────────────────────────────────

  /// <summary>
  /// Filtered view of <see cref="HorizonJplService.CometsData"/> based on
  /// <see cref="SearchQuery"/>. Updated whenever the query or the source list changes.
  /// </summary>
  public ObservableCollection<CometSearchResult> FilteredSearchResults { get; } = [];

  /// <summary>SPK records for the selected comet from the JPL service (bound directly).</summary>
  public ObservableCollection<SpkRecordItem> SpkRecords => _jpl.SpkRecordsData;
  
  // ── Debug Properties ─────────────────────────────────────────────────────

  public ObservableCollection<JetViewModel>? DebugJets
  {
    get
    {
#if DEBUG
      if (_modelSessionService.ActiveSessionIds.Count == 0) return null;
      return _modelSessionService.GetSession(_modelSessionService.ActiveSessionIds[0])?.Jets;
#else
      return null;
#endif
    }
  }

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
    ITabStateService<ModelSession> modelSessionService
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

    Icon = "☄"; // comet — U+2604
    SubscribeToStrings(schedulerProvider);
    WireReactiveSubscriptions(schedulerProvider);

    // Keep FilteredSearchResults in sync with the source collection
    _jpl.CometsData.CollectionChanged += (_, _) => ApplySearchFilter();
  }

  // ── Reactive wiring ───────────────────────────────────────────────────────

  private void WireReactiveSubscriptions(ISchedulerProvider schedulerProvider)
  {
    // 1. Proposed timeline display from TimelineService
    _timeline
      .ProposedTimeRange.ObserveOn(schedulerProvider.MainThread)
      .Subscribe(range =>
      {
        if (range is null)
        {
          HasProposedTimeline = false;
          ProposedStartEpoch = ProposedEndEpoch = string.Empty;
        }
        else
        {
          HasProposedTimeline = true;
          ProposedStartEpoch = FormatTaiEpoch(range.StartCenturies, range.StartNs);
          ProposedEndEpoch   = FormatTaiEpoch(range.EndCenturies,   range.EndNs);

          // Detect change-after-commit by comparing ISO display strings — avoids
          // nanosecond rounding false-positives that occur with TAI record equality.
          if (IsAlmanacCommitted && _lastCommittedProposedRange is { } snap
              && (ProposedStartEpoch != snap.Start || ProposedEndEpoch != snap.End))
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
      })
      .AddDisposableTo(_disposables);
  }

  // ── Property change override for rotational model debounce ───────────────

  protected override void OnPropertyChanged(PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

    bool isRotProp =
      e.PropertyName
      is nameof(PoleRaDeg)
        or nameof(PoleDecDeg)
        or nameof(PrimeMeridianDeg)
        or nameof(PoleRaRateDegCen)
        or nameof(PoleDecRateDegCen)
        or nameof(RotRateDegDay);

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
  }

  /// <summary>
  /// Rebuilds <see cref="FilteredSearchResults"/> from <c>_jpl.CometsData</c>
  /// filtered by the current <see cref="SearchQuery"/> (case-insensitive contains
  /// on <c>Name</c> or <c>PrimaryDesignation</c>). An empty query shows all results.
  /// </summary>
  private void ApplySearchFilter()
  {
    var query = SearchQuery?.Trim() ?? string.Empty;

    FilteredSearchResults.Clear();
    foreach (var comet in _jpl.CometsData)
    {
      if (
        query.Length == 0
        || comet.Name.Contains(query, StringComparison.OrdinalIgnoreCase)
        || comet.PrimaryDesignation.Contains(query, StringComparison.OrdinalIgnoreCase)
      )
        FilteredSearchResults.Add(comet);
    }
  }

  // ── Commands ──────────────────────────────────────────────────────────────

  [RelayCommand]
  private async Task SearchCometsAsync()
  {
    IsSearching = true;
    try
    {
      await _jpl.FetchCometsAsync();
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

  [RelayCommand]
  private async Task DownloadAndCommitAsync()
  {
    if (!HasProposedTimeline || SelectedComet is null || SelectedSpkRecord is null)
    {
      EmitInvalidStateBreadcrumb();
      return;
    }

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

      // Commit the new SPK
      bool committed = await _cometConfig.CommitCometAsync(filePath, naifId);

      if (committed)
      {
        // Update session
        var session = CurrentSession;
        if (session is not null)
        {
          session.SpkId = naifId;
          session.CommittedDesignation = SelectedComet.PrimaryDesignation;
          session.CommittedFullName = SelectedComet.Name;
          session.CommittedSpkFilePath = filePath;
          session.IsAlmanacLoaded = true;
        }

        // Fetch nucleus radius from Horizon constants (best-effort)
        DownloadStatus = "Fetching nucleus radius…";
        try
        {
          var orbitData = await _jpl.GetPlanetDataAsync(
            SelectedComet.PrimaryDesignation,
            "@sun",
            DateTime.UtcNow,
            DateTime.UtcNow.AddDays(1),
            "1d"
          );
          if (session is not null && orbitData is not null && orbitData.CometRadiusKm > 0.0)
          {
            session.NucleusRadiusKm = (float)orbitData.CometRadiusKm;
            WeakReferenceMessenger.Default.Send(
              new Messages.NucleusRadiusKnownMessage { RadiusKm = session.NucleusRadiusKm }
            );
          }
        }
        catch
        {
          // Best-effort — user can enter radius manually in Model tab
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
    _cometConfig.DecommitComet();

    var session = CurrentSession;
    if (session is not null)
    {
      session.SpkId = null;
      session.CommittedDesignation = string.Empty;
      session.CommittedFullName = string.Empty;
      session.CommittedSpkFilePath = null;
      session.IsAlmanacLoaded = false;
    }

    _lastCommittedProposedRange = null;
    DownloadStatus = string.Empty;
    WeakReferenceMessenger.Default.Send(new Messages.CometDecommittedMessage());
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
      RotRateDegDay
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
    }
  }

  /// <summary>Formats a TAI epoch (centuries + ns) as a UTC display string.</summary>
  private static string FormatTaiEpoch(short centuries, ulong nanoseconds)
  {
    try
    {
      // TAI seconds since J2000: centuries * SecsPerCentury + ns / 1e9
      // J2000 = 2000-01-01T12:00:00 TAI = 2000-01-01T11:59:27.816 UTC (approx)
      const double SecsPerCentury = 3_155_760_000.0;
      const double J2000UnixSec = 946727967.816; // J2000 in Unix seconds (UTC)
      double taiSec = (double)centuries * SecsPerCentury + (double)nanoseconds / 1e9;
      double unixSec = taiSec + J2000UnixSec;
      var dt = DateTimeOffset.FromUnixTimeSeconds((long)unixSec).UtcDateTime;
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
      // Component IDs for Almanac Planet (26) and Rotational Body (24)
      _runtimeService.DebugECSPrint(1, [cometId], 2, [26, 24]);
    }
#endif
  }

  [RelayCommand]
  private void DebugQueryJet(ulong jetId)
  {
#if DEBUG
    // Component ID for Particle System (22)
    _runtimeService.DebugECSPrint(1, [jetId], 1, [22]);
#endif
  }
}
