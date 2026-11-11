using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reactive.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// The compact layout's panel: the whole workflow (comet, observation period, nucleus, jets, dust,
/// FITS image, view, playback, export) as one column of cards beside the 3D viewport.
/// <para>
/// It owns no model state. Every card binds to the classic tabs' view models, the very instances
/// the docking tree holds (<see cref="DockingManagerViewModel.EnsureTab(Type)"/>): a second
/// instance would run every commit twice, since they all subscribe to the same streams.
/// <see cref="Activate"/> reads those references again each time the compact layout is entered.
/// </para>
/// </summary>
[GenerateLocalizedStrings(keyPrefix: "Compact_", designTitle: "Compact")]
public partial class CompactPanelViewModel : ObservableObject, IDisposable
{
  private readonly ITranslationService _translationService;
  private readonly Func<Type, TabItemViewModel?> _ensureTab;
  private readonly IUiThreadDispatcher? _dispatcher;
  private readonly IDisposable? _cultureSubscription;
  private INotifyCollectionChanged? _billboardsHooked;

  /// <param name="ensureTab">the docking tree's <see cref="DockingManagerViewModel.EnsureTab(Type)"/></param>
  /// <param name="nucleusMini">the "Nucleus seen from Earth" view</param>
  /// <param name="modeBar">the Orbit / Nucleus / Model strip (null: none)</param>
  /// <param name="exportImageCommand">File › Export image</param>
  /// <param name="classicLayoutCommand">switches to the classic layout</param>
  /// <param name="translation">UI strings (null: English)</param>
  /// <param name="dispatcher">the UI thread (null: calls run where they arrive)</param>
  public CompactPanelViewModel(
    Func<Type, TabItemViewModel?> ensureTab,
    NucleusMiniViewModel nucleusMini,
    ViewModeBarViewModel? modeBar = null,
    ICommand? exportImageCommand = null,
    ICommand? classicLayoutCommand = null,
    ITranslationService? translation = null,
    IUiThreadDispatcher? dispatcher = null)
  {
    _ensureTab = ensureTab;
    _dispatcher = dispatcher;
    NucleusMini = nucleusMini;
    ModeBar = modeBar;
    ExportImageCommand = exportImageCommand;
    ClassicLayoutCommand = classicLayoutCommand;
    _translationService = translation ?? new TranslationService(
      new System.Resources.ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly),
      new CultureInfo("en"));
    RefreshStrings();
    if (translation?.CultureChanged is { } cultures)
    {
      _cultureSubscription = cultures.Skip(1).Subscribe(_ =>
      {
        if (_dispatcher is null || _dispatcher.CheckAccess())
          RefreshStrings();
        else
          _dispatcher.Dispatch(RefreshStrings);
      });
    }
  }

  // ── The classic tabs' view models (the docking tree's instances) ────────────

  [ObservableProperty]
  private CometTabViewModel? _comet;

  [ObservableProperty]
  private ModelTabViewModel? _model;

  [ObservableProperty]
  private TimelineTabViewModel? _timeline;

  [ObservableProperty]
  private ImportsTabViewModel? _imports;

  [ObservableProperty]
  private SettingsTabViewModel? _settings;

  /// <summary>The Orbit / Nucleus / Model strip's view model (the main window's own).</summary>
  public ViewModeBarViewModel? ModeBar { get; }

  /// <summary>The "Nucleus seen from Earth" view above the cards.</summary>
  public NucleusMiniViewModel NucleusMini { get; }

  /// <summary>File › Export image.</summary>
  public ICommand? ExportImageCommand { get; }

  /// <summary>"More settings": back to the classic layout.</summary>
  public ICommand? ClassicLayoutCommand { get; }

  /// <summary>The compact layout is the one shown.</summary>
  [ObservableProperty]
  private bool _isActive;

  /// <summary>The last export's outcome (the main window's status).</summary>
  [ObservableProperty]
  private string? _exportStatus;

  /// <summary>The billboard the FITS card's controls act on (the newest one by default).</summary>
  [ObservableProperty]
  private DesktopBillboardItemViewModel? _selectedBillboard;

  /// <summary>Outcome of the last "Import and overlay"; empty: none.</summary>
  [ObservableProperty]
  private string _fitsStatus = string.Empty;

  // ── Entering and leaving the compact layout ────────────────────────────────

  /// <summary>
  /// The compact layout is entered: takes the Comet, Model, Timeline, Imports and Settings view
  /// models from the docking tree (creating, parked and unselected, the ones the layout lacks) and
  /// shows the nucleus view. Never builds a view model of its own.
  /// </summary>
  public void Activate()
  {
    var comet = Ensure<CometTabViewModel>();
    var model = Ensure<ModelTabViewModel>();
    var timeline = Ensure<TimelineTabViewModel>();
    var imports = Ensure<ImportsTabViewModel>();
    var settings = Ensure<SettingsTabViewModel>();

    Rehook(Comet, comet, OnCometChanged);
    Rehook(Timeline, timeline, OnTimelineChanged);
    Comet = comet;
    Model = model;
    Timeline = timeline;
    Imports = imports;
    Settings = settings;
    HookBillboards(settings?.Billboards?.Items);

    IsActive = true;
    PushCometToMini();
    PushClockToMini();
    NucleusMini.IsLayoutCompact = true;
  }

  /// <summary>The classic layout is back: the nucleus view pauses; the references stay.</summary>
  public void Deactivate()
  {
    IsActive = false;
    NucleusMini.IsLayoutCompact = false;
  }

  private T? Ensure<T>() where T : TabItemViewModel => _ensureTab(typeof(T)) as T;

  private static void Rehook<T>(T? old, T? now, PropertyChangedEventHandler handler) where T : class, INotifyPropertyChanged
  {
    if (ReferenceEquals(old, now))
      return;
    if (old is not null)
      old.PropertyChanged -= handler;
    if (now is not null)
      now.PropertyChanged += handler;
  }

  private void OnCometChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is nameof(CometTabViewModel.IsAlmanacCommitted)
        or nameof(CometTabViewModel.ManualNucleusRadiusKm)
        or nameof(CometTabViewModel.CurrentSession))
      PushCometToMini();
  }

  private void OnTimelineChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is nameof(TimelineTabViewModel.CurrentEpochText))
      PushClockToMini();
  }

  private void PushCometToMini()
  {
    if (Comet is not { } comet)
      return;
    double radius = comet.ManualNucleusRadiusKm > 0f
      ? comet.ManualNucleusRadiusKm
      : comet.CurrentSession?.NucleusRadiusKm ?? 0f;
    NucleusMini.SetNucleusRadiusKm(radius);
    NucleusMini.SetCometCommitted(comet.IsAlmanacCommitted);
  }

  private void PushClockToMini()
  {
    if (Timeline is not { } timeline)
      return;
    string text = timeline.CurrentEpochText ?? string.Empty;
    NucleusMini.DateText = text.Length == 0 ? string.Empty : string.Concat(text, " ", timeline.ScaleName);
  }

  // ── Comet presets ──────────────────────────────────────────────────────────

  /// <summary>Preset: comet 67P/Churyumov-Gerasimenko.</summary>
  [RelayCommand]
  private Task Preset67PAsync() => SelectPresetAsync("67P");

  /// <summary>Preset: comet 10P/Tempel 2.</summary>
  [RelayCommand]
  private Task Preset10PAsync() => SelectPresetAsync("10P");

  /// <summary>How long a preset waits for the comet's ephemeris records.</summary>
  internal static readonly TimeSpan PresetRecordsTimeout = TimeSpan.FromSeconds(60);

  /// <summary>
  /// Searches <paramref name="designation"/> ("67P"), selects that comet and, once its ephemeris
  /// records are listed, the last usable one unless one is already chosen. The commit stays the
  /// user's click. False when the comet was not found.
  /// </summary>
  internal async Task<bool> SelectPresetAsync(string designation)
  {
    if (Comet is not { } comet)
      return false;
    comet.SearchQuery = designation;
    await comet.SearchCometsCommand.ExecuteAsync(null);
    var match = PickPreset(comet.FilteredSearchResults, designation);
    if (match is null)
      return false;
    if (!ReferenceEquals(comet.SelectedComet, match))
    {
      // selecting the comet starts the fetch of its records
      comet.SelectedComet = match;
      await WaitWhileAsync(comet, nameof(CometTabViewModel.IsLoadingSpkRecords), () => comet.IsLoadingSpkRecords, PresetRecordsTimeout);
    }
    if (comet.SelectedSpkRecord is null && comet.SpkRecords.LastOrDefault(r => r.IsValid) is { } record)
      comet.SelectedSpkRecord = record;
    return true;
  }

  /// <summary>The search result whose designation is <paramref name="designation"/> ("67P" matches
  /// "67P" and "67P/Churyumov-Gerasimenko", not "167P").</summary>
  internal static CometSearchResult? PickPreset(IEnumerable<CometSearchResult> results, string designation)
  {
    static bool Is(string text, string d) =>
      text.StartsWith(d, StringComparison.OrdinalIgnoreCase)
      && (text.Length == d.Length || !char.IsLetterOrDigit(text[d.Length]));
    var list = results.ToList();
    return list.FirstOrDefault(r => Is(r.PrimaryDesignation.Trim(), designation))
      ?? list.FirstOrDefault(r => Is(r.Name.Trim(), designation));
  }

  private static async Task WaitWhileAsync(INotifyPropertyChanged source, string property, Func<bool> busy, TimeSpan timeout)
  {
    if (!busy())
      return;
    var done = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    void Handler(object? s, PropertyChangedEventArgs e)
    {
      if (e.PropertyName == property && !busy())
        done.TrySetResult(true);
    }
    source.PropertyChanged += Handler;
    try
    {
      if (busy())
        await Task.WhenAny(done.Task, Task.Delay(timeout));
    }
    finally
    {
      source.PropertyChanged -= Handler;
    }
  }

  // ── FITS image: import and overlay ─────────────────────────────────────────

  private bool _importingOverlay;

  /// <summary>
  /// "Import and overlay": the Imports tab's file picker and import, then a billboard of the image
  /// that import added, shown over the viewport.
  /// </summary>
  [RelayCommand]
  private async Task ImportAndOverlayAsync()
  {
    if (_importingOverlay || Imports is not { } imports || Settings?.Billboards is not { } billboards)
      return;
    _importingOverlay = true;
    try
    {
      var texture = await ImportAndOverlayCoreAsync(imports, billboards);
      FitsStatus = texture is null
        ? (string.IsNullOrEmpty(imports.StatusMessage) ? StrFitsNothingImported : imports.StatusMessage)
        : string.Concat(StrFitsOverlaid, " ", texture.Label);
      if (texture is not null && billboards.Items.LastOrDefault(i => i.TextureKey == texture.Key) is { } row)
        SelectedBillboard = row;
    }
    finally
    {
      _importingOverlay = false;
    }
  }

  /// <summary>
  /// Runs <paramref name="imports"/>' import, then creates a billboard of the displayable texture
  /// it added (a FITS image first when it added several, e.g. a mesh with its textures). Returns
  /// that texture; null when the import added none (cancelled, failed, already imported, a mesh).
  /// </summary>
  /// <summary>How long "Import and overlay" waits for an import's texture to reach the list.</summary>
  internal static TimeSpan ImportedTextureWait { get; set; } = TimeSpan.FromSeconds(10);

  internal static async Task<ImportedAsset?> ImportAndOverlayCoreAsync(IImportsTabViewModel imports, DesktopBillboardsViewModel billboards)
  {
    var before = new HashSet<string>(imports.TextureAssets.Select(t => t.Key), StringComparer.Ordinal);
    string? statusBefore = imports.StatusMessage;
    await imports.ImportCommand.ExecuteAsync(null);
    List<ImportedAsset> Added() => imports.TextureAssets.Where(t => t.IsDisplayable && !before.Contains(t.Key)).ToList();
    var added = Added();
    // The asset library publishes its texture list on the UI thread (ObserveOn), a dispatcher pass
    // after the import's own continuation: the new texture is not in the list yet when the import
    // returns. An import that ran says so in the status line (a cancelled dialog leaves it as it
    // was); then its texture is waited for, the dispatcher running meanwhile.
    if (added.Count == 0 && !string.Equals(statusBefore, imports.StatusMessage, StringComparison.Ordinal))
    {
      var waited = System.Diagnostics.Stopwatch.StartNew();
      while (added.Count == 0 && waited.Elapsed < ImportedTextureWait)
      {
        await Task.Delay(25);
        added = Added();
      }
    }
    var texture = added.LastOrDefault(t => IsFitsKey(t.Key)) ?? added.LastOrDefault();
    if (texture is null)
      return null;
    billboards.SelectedTexture = texture;
    if (billboards.CreateCommand.CanExecute(null))
      billboards.CreateCommand.Execute(null);
    return texture;
  }

  private static bool IsFitsKey(string key) =>
    key.EndsWith(".fits", StringComparison.OrdinalIgnoreCase)
    || key.EndsWith(".fit", StringComparison.OrdinalIgnoreCase)
    || key.EndsWith(".fts", StringComparison.OrdinalIgnoreCase);

  private void HookBillboards(INotifyCollectionChanged? items)
  {
    if (!ReferenceEquals(_billboardsHooked, items))
    {
      if (_billboardsHooked is not null)
        _billboardsHooked.CollectionChanged -= OnBillboardsChanged;
      if (items is not null)
        items.CollectionChanged += OnBillboardsChanged;
      _billboardsHooked = items;
    }
    SyncSelectedBillboard(null);
  }

  private void OnBillboardsChanged(object? sender, NotifyCollectionChangedEventArgs e) =>
    SyncSelectedBillboard(e.Action == NotifyCollectionChangedAction.Add && e.NewItems is { Count: > 0 } added
      ? added[added.Count - 1] as DesktopBillboardItemViewModel
      : null);

  // a new row is selected; a selection whose row went away falls back to the last row
  private void SyncSelectedBillboard(DesktopBillboardItemViewModel? added)
  {
    var items = Settings?.Billboards?.Items;
    if (items is null)
    {
      SelectedBillboard = null;
      return;
    }
    if (added is not null)
      SelectedBillboard = added;
    else if (SelectedBillboard is null || !items.Contains(SelectedBillboard))
      SelectedBillboard = items.LastOrDefault();
  }

  public void Dispose()
  {
    _cultureSubscription?.Dispose();
    Rehook(Comet, null, OnCometChanged);
    Rehook(Timeline, null, OnTimelineChanged);
    if (_billboardsHooked is not null)
      _billboardsHooked.CollectionChanged -= OnBillboardsChanged;
    _billboardsHooked = null;
    NucleusMini.Dispose();
  }
}
