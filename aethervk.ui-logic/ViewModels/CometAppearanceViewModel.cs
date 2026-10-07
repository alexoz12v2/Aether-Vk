using System;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reactive.Concurrency;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>Entry of a texture-channel picker: an imported texture, or "none".</summary>
public sealed record TextureOption(ImportedAsset? Asset, string Label)
{
  public ulong Id => Asset?.Id ?? 0;
  public AssetThumbnail? Thumbnail => Asset?.Thumbnail;
}

/// <summary>
/// "Comet Appearance" section of the Settings tab.
///
/// <list type="bullet">
///   <item><b>Default</b> display mode keeps the procedural sphere.</item>
///   <item><b>Custom</b> shows an imported mesh, recentred on its bounding sphere and scaled so that
///   sphere matches the nucleus radius, with up to four texture channels (albedo, normal,
///   roughness, ambient occlusion — the <c>physical_mesh2</c> bindings) and a rotation /
///   translation offset in the comet body frame.</item>
/// </list>
///
/// The appearance is aesthetic only (simulation state is unaffected). Display mode, mesh and
/// texture wiring rebuild GPU resources, so they are locked while the simulation plays (and while
/// an import holds the asset library); the placement offset stays live and is throttled.
/// The native runtime is the source of truth: a refused change reverts the UI to the last
/// applied appearance.
/// </summary>
public partial class CometAppearanceViewModel : ObservableObject, IDisposable
{
  /// <summary>Minimum interval between two placement updates sent to the runtime.</summary>
  public static readonly TimeSpan PlacementThrottle = TimeSpan.FromMilliseconds(33);

  private readonly INativeRuntimeService _runtimeService;
  private readonly AssetLibraryService _assetLibrary;
  private readonly ITranslationService _translationService;
  private readonly CompositeDisposable _disposables = [];
  private readonly ISchedulerProvider _schedulerProvider;

  /// <summary>A placement dispatch is scheduled (one-shot timer, so ≤ ~30 updates/s while dragging).</summary>
  private bool _placementDispatchPending;
  private readonly SerialDisposable _placementTimer = new();

  /// <summary>Last appearance accepted by the runtime.</summary>
  private CometAppearanceDto _applied;

  /// <summary>Suppresses dispatch while the VM itself updates bound properties.</summary>
  private bool _suppressDispatch;

  public ObservableCollection<ImportedAsset> MeshOptions { get; } = [];
  public ObservableCollection<TextureOption> TextureOptions { get; } = [];

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsDefaultMode))]
  private bool _isCustomMode;

  [ObservableProperty]
  private ImportedAsset? _selectedMesh;

  [ObservableProperty]
  private TextureOption? _selectedAlbedo;

  [ObservableProperty]
  private TextureOption? _selectedNormal;

  [ObservableProperty]
  private TextureOption? _selectedRoughness;

  [ObservableProperty]
  private TextureOption? _selectedAo;

  [ObservableProperty]
  private double _yaw;

  [ObservableProperty]
  private double _pitch;

  [ObservableProperty]
  private double _roll;

  [ObservableProperty]
  private double _translationX;

  [ObservableProperty]
  private double _translationY;

  [ObservableProperty]
  private double _translationZ;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsWiringEnabled))]
  [NotifyPropertyChangedFor(nameof(IsWiringLocked))]
  [NotifyPropertyChangedFor(nameof(IsCustomModeAvailable))]
  private bool _isSimulationRunning;

  /// <summary>An import or an unload holds the native asset library.</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsWiringEnabled))]
  [NotifyPropertyChangedFor(nameof(IsWiringLocked))]
  [NotifyPropertyChangedFor(nameof(IsCustomModeAvailable))]
  private bool _isImporting;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(IsCustomModeAvailable))]
  private bool _hasMeshes;

  /// <summary>Reason the last change was refused (empty when none).</summary>
  [ObservableProperty]
  private string _statusMessage = string.Empty;

  public bool IsDefaultMode
  {
    get => !IsCustomMode;
    set => IsCustomMode = !value;
  }

  /// <summary>Display mode / mesh / texture pickers are editable.</summary>
  public bool IsWiringEnabled => !IsSimulationRunning && !IsImporting;

  /// <summary>Custom mode can be selected: at least one mesh is imported and wiring is editable.</summary>
  public bool IsCustomModeAvailable => HasMeshes && IsWiringEnabled;

  public bool IsWiringLocked => !IsWiringEnabled;

  public CometAppearanceViewModel(
    INativeRuntimeService runtimeService,
    AssetLibraryService assetLibrary,
    TimelineService timelineService,
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider)
  {
    _runtimeService = runtimeService;
    _assetLibrary = assetLibrary;
    _translationService = translationService;
    _schedulerProvider = schedulerProvider;

    if (_runtimeService.GetCometAppearance(out var current))
      _applied = current;
    RebuildOptions(_assetLibrary.CurrentMeshes, _assetLibrary.CurrentTextures);
    LoadFrom(_applied);

    timelineService.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(running => IsSimulationRunning = running)
      .AddDisposableTo(_disposables);

    _assetLibrary.IsBusy
      .Subscribe(v => IsImporting = v)
      .AddDisposableTo(_disposables);

    _assetLibrary.Meshes
      .CombineLatest(_assetLibrary.Textures)
      .Subscribe(t =>
      {
        // the runtime may have ejected / unwired the comet (asset unloaded): re-read it
        if (_runtimeService.GetCometAppearance(out var current))
          _applied = current;
        RebuildOptions(t.First, t.Second);
        LoadFrom(_applied);
      })
      .AddDisposableTo(_disposables);
  }

  /// <summary>Design-time constructor (no runtime): dispatch is a no-op.</summary>
  protected CometAppearanceViewModel()
  {
    _runtimeService = null!;
    _assetLibrary = null!;
    _translationService = null!;
    _schedulerProvider = null!;
  }

  /// <summary>Resets rotation and translation to identity.</summary>
  [RelayCommand]
  private void ResetPlacement()
  {
    _placementDispatchPending = false;
    _suppressDispatch = true;
    Yaw = Pitch = Roll = 0;
    TranslationX = TranslationY = TranslationZ = 0;
    _suppressDispatch = false;
    Dispatch();
  }

  // ── Change hooks ─────────────────────────────────────────────────────────

  partial void OnIsCustomModeChanged(bool value) => WiringChanged();

  partial void OnSelectedMeshChanged(ImportedAsset? value)
  {
    if (_suppressDispatch) return;
    // Pre-wire the textures bundled with the mesh into channels that are still empty.
    if (value is not null)
    {
      _suppressDispatch = true;
      SelectedAlbedo = PreWire(SelectedAlbedo, value, TextureChannel.Albedo);
      SelectedNormal = PreWire(SelectedNormal, value, TextureChannel.Normal);
      SelectedRoughness = PreWire(SelectedRoughness, value, TextureChannel.Roughness);
      SelectedAo = PreWire(SelectedAo, value, TextureChannel.Ao);
      _suppressDispatch = false;
    }
    WiringChanged();
  }

  partial void OnSelectedAlbedoChanged(TextureOption? value) => WiringChanged();
  partial void OnSelectedNormalChanged(TextureOption? value) => WiringChanged();
  partial void OnSelectedRoughnessChanged(TextureOption? value) => WiringChanged();
  partial void OnSelectedAoChanged(TextureOption? value) => WiringChanged();

  partial void OnYawChanged(double value) => PlacementChanged();
  partial void OnPitchChanged(double value) => PlacementChanged();
  partial void OnRollChanged(double value) => PlacementChanged();
  partial void OnTranslationXChanged(double value) => PlacementChanged();
  partial void OnTranslationYChanged(double value) => PlacementChanged();
  partial void OnTranslationZChanged(double value) => PlacementChanged();

  private void WiringChanged()
  {
    if (_suppressDispatch) return;
    Dispatch();
  }

  private void PlacementChanged()
  {
    if (_suppressDispatch || _placementDispatchPending || _schedulerProvider is null) return;
    // Trailing dispatch of the latest values; no periodic timer is kept alive while idle.
    _placementDispatchPending = true;
    _placementTimer.Disposable = _schedulerProvider.Background.Schedule(
      PlacementThrottle,
      () =>
        _schedulerProvider.MainThread.Schedule(() =>
        {
          _placementDispatchPending = false;
          Dispatch();
        }));
  }

  private TextureOption? PreWire(TextureOption? current, ImportedAsset mesh, TextureChannel channel)
  {
    if (current is not null && current.Id != 0) return current;
    ulong bundled = mesh.BundledTexture(channel);
    return bundled == 0 ? current : TextureOptions.FirstOrDefault(o => o.Id == bundled) ?? current;
  }

  // ── Runtime round trip ───────────────────────────────────────────────────

  /// <summary>Current UI state as a runtime DTO.</summary>
  public CometAppearanceDto BuildDto() =>
    new()
    {
      Mode = IsCustomMode ? CometDisplayMode.Custom : CometDisplayMode.Default,
      Mesh = SelectedMesh?.Id ?? 0,
      Albedo = SelectedAlbedo?.Id ?? 0,
      Normal = SelectedNormal?.Id ?? 0,
      Roughness = SelectedRoughness?.Id ?? 0,
      Ao = SelectedAo?.Id ?? 0,
      Yaw = (float)Yaw,
      Pitch = (float)Pitch,
      Roll = (float)Roll,
      TranslationX = (float)TranslationX,
      TranslationY = (float)TranslationY,
      TranslationZ = (float)TranslationZ,
    };

  private void Dispatch()
  {
    if (_runtimeService is null) return; // design time
    var dto = BuildDto();
    var status = _runtimeService.SetCometAppearance(dto);
    switch (status)
    {
      case CometAppearanceStatus.Queued:
        _applied = dto;
        StatusMessage = string.Empty;
        break;
      case CometAppearanceStatus.NotAvailable:
        // design / headless mode: keep the UI state
        _applied = dto;
        break;
      default:
        StatusMessage = status switch
        {
          CometAppearanceStatus.LockedWhileRunning =>
            _translationService.GetString("Tabs_Settings_AppearanceLockedHint"),
          CometAppearanceStatus.Busy => _translationService.GetString("Tabs_Imports_Importing"),
          _ => $"Appearance change refused ({status})",
        };
        // revert to what the runtime actually displays
        LoadFrom(_applied);
        break;
    }
  }

  /// <summary>Pushes <paramref name="dto"/> into the bound properties without dispatching.</summary>
  private void LoadFrom(CometAppearanceDto dto)
  {
    _suppressDispatch = true;
    try
    {
      IsCustomMode = dto.Mode == CometDisplayMode.Custom;
      SelectedMesh = dto.Mesh == 0 ? null : MeshOptions.FirstOrDefault(m => m.Id == dto.Mesh);
      SelectedAlbedo = OptionFor(dto.Albedo);
      SelectedNormal = OptionFor(dto.Normal);
      SelectedRoughness = OptionFor(dto.Roughness);
      SelectedAo = OptionFor(dto.Ao);
      Yaw = dto.Yaw;
      Pitch = dto.Pitch;
      Roll = dto.Roll;
      TranslationX = dto.TranslationX;
      TranslationY = dto.TranslationY;
      TranslationZ = dto.TranslationZ;
    }
    finally
    {
      _suppressDispatch = false;
    }
  }

  private TextureOption? OptionFor(ulong id) =>
    TextureOptions.FirstOrDefault(o => o.Id == id) ?? TextureOptions.FirstOrDefault();

  private void RebuildOptions(ImmutableArray<ImportedAsset> meshes, ImmutableArray<ImportedAsset> textures)
  {
    _suppressDispatch = true;
    try
    {
      MeshOptions.Clear();
      foreach (var m in meshes) MeshOptions.Add(m);
      TextureOptions.Clear();
      TextureOptions.Add(new TextureOption(null, _translationService.GetString("Tabs_Settings_TextureNone")));
      foreach (var t in textures) TextureOptions.Add(new TextureOption(t, t.Label));
      HasMeshes = MeshOptions.Count > 0;
    }
    finally
    {
      _suppressDispatch = false;
    }
  }

  public void Dispose()
  {
    _placementTimer.Dispose();
    _disposables.Dispose();
  }
}
