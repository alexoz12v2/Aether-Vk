using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.IO;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Imports tab: imports meshes (obj, ply, gltf, glb) and textures (png, jpg, jpeg, ktx2) into
/// the native asset library and lists them in two sections. Textures bundled in a mesh file
/// show up in the texture section as their own entries (no duplicates: a re-import, or the same
/// image reached through two files, maps to the existing asset).
/// </summary>
[GenerateLocalizedStrings(
  keyPrefix:    "Tabs_Imports_",
  designTitle:  "Imports",
  designIcon:   "⬇")]
public partial class ImportsTabViewModel : StatefulTabViewModelBase<ImportsSession>, IImportsTabViewModel
{
  private readonly ITranslationService _translationService;
  private readonly ILocalStorageService _localStorageService;
  private readonly AssetLibraryService _assetLibrary;
  private readonly IFileDialogService _fileDialogService;
  private readonly CompositeDisposable _disposables = [];

  [ObservableProperty]
  private ObservableCollection<string> _sessionFolders = [];

  public ObservableCollection<ImportedAsset> MeshAssets { get; } = [];
  public ObservableCollection<ImportedAsset> TextureAssets { get; } = [];

  [ObservableProperty]
  private bool _hasMeshes;

  [ObservableProperty]
  private bool _hasTextures;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(ImportCommand))]
  private bool _isImporting;

  /// <summary>An import or an unload is in flight.</summary>
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(RemoveAssetCommand))]
  [NotifyPropertyChangedFor(nameof(CanUnload))]
  private bool _isBusy;

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(RemoveAssetCommand))]
  [NotifyPropertyChangedFor(nameof(CanUnload))]
  private bool _isSimulationRunning;

  /// <summary>Unloading is locked while the simulation plays (like mesh / texture wiring).</summary>
  public bool CanUnload => !IsSimulationRunning && !IsBusy;

  /// <summary>Outcome of the last import (empty when none).</summary>
  [ObservableProperty]
  private string _statusMessage = string.Empty;

  public ImportsTabViewModel(
    ITranslationService translationService,
    ISchedulerProvider schedulerProvider,
    ITabStateService<ImportsSession> sessionService,
    ILocalStorageService localStorageService,
    AssetLibraryService assetLibrary,
    IFileDialogService fileDialogService,
    TimelineService timelineService)
    : base("Imports", sessionService)
  {
    _translationService = translationService;
    _localStorageService = localStorageService;
    _assetLibrary = assetLibrary;
    _fileDialogService = fileDialogService;
    Icon = "⬇"; // down arrow / import — U+2B07
    SubscribeToStrings(schedulerProvider);

    var sessionsDir = Path.GetDirectoryName(_localStorageService.SessionDirectory);
    if (sessionsDir is not null && Directory.Exists(sessionsDir))
    {
      var dirs = Directory.GetDirectories(sessionsDir);
      SessionFolders = new ObservableCollection<string>(dirs);
    }

    _assetLibrary.Meshes
      .Subscribe(list =>
      {
        Sync(MeshAssets, list);
        HasMeshes = MeshAssets.Count > 0;
      })
      .AddDisposableTo(_disposables);
    _assetLibrary.Textures
      .Subscribe(list =>
      {
        Sync(TextureAssets, list);
        HasTextures = TextureAssets.Count > 0;
      })
      .AddDisposableTo(_disposables);
    _assetLibrary.IsImporting
      .Subscribe(v => IsImporting = v)
      .AddDisposableTo(_disposables);
    _assetLibrary.IsBusy
      .Subscribe(v => IsBusy = v)
      .AddDisposableTo(_disposables);
    timelineService.IsSimulationRunning
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(v => IsSimulationRunning = v)
      .AddDisposableTo(_disposables);
  }

  /// <summary>
  /// Unloads an asset. A comet displaying the mesh goes back to the procedural sphere; a wired
  /// texture channel is cleared. Textures bundled with a mesh are kept.
  /// </summary>
  [RelayCommand(CanExecute = nameof(CanRemoveAsset))]
  private async System.Threading.Tasks.Task RemoveAssetAsync(ImportedAsset? asset)
  {
    if (asset is null) return;
    var result = await _assetLibrary.RemoveAsync(asset.Id);
    StatusMessage = result.Success
      ? $"{StrStatusUnloaded}: {asset.Label}"
      : $"{StrStatusUnloadFailed}: {asset.Label}";
  }

  private bool CanRemoveAsset(ImportedAsset? asset) => asset is not null && CanUnload;

  private bool CanImport() => !IsImporting;

  /// <summary>Opens a file picker and imports the chosen mesh or texture.</summary>
  [RelayCommand(CanExecute = nameof(CanImport))]
  private async Task ImportAsync()
  {
    var path = await _fileDialogService.ShowOpenFileDialogAsync(
      StrDialogTitle,
      AssetLibraryService.AllExtensions);
    if (string.IsNullOrEmpty(path)) return;
    await ImportPathsAsync([path!]);
  }

  /// <summary>Imports files sequentially (file picker, drag and drop).</summary>
  public async Task ImportPathsAsync(IEnumerable<string> paths)
  {
    foreach (var path in paths)
    {
      var name = Path.GetFileName(path);
      if (!AssetLibraryService.IsSupported(path))
      {
        StatusMessage = $"{StrStatusUnsupported}: {name}";
        continue;
      }
      var result = await _assetLibrary.ImportAsync(path);
      StatusMessage = !result.Success
        ? $"{StrStatusFailed}: {name}"
        : result.AddedCount == 0
          ? $"{StrStatusAlreadyImported}: {name}"
          : $"{StrStatusImported}: {name}";
    }
  }

  /// <summary>Replaces the collection content only when it actually changed.</summary>
  private static void Sync(ObservableCollection<ImportedAsset> target, ImmutableArray<ImportedAsset> source)
  {
    if (target.Count == source.Length)
    {
      bool same = true;
      for (int i = 0; i < source.Length && same; i++)
        same = Equals(target[i], source[i]);
      if (same) return;
    }
    target.Clear();
    foreach (var a in source) target.Add(a);
  }

  private void SubscribeToStrings(ISchedulerProvider schedulerProvider)
  {
    RefreshStrings();
    _translationService.CultureChanged
      .Skip(1)
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(_ => RefreshStrings())
      .AddDisposableTo(_disposables);
  }

  public override void Dispose()
  {
    _disposables.Dispose();
    base.Dispose();
  }
}

public partial interface IImportsTabViewModel
{
  ObservableCollection<string> SessionFolders { get; set; }
  ObservableCollection<ImportedAsset> MeshAssets { get; }
  ObservableCollection<ImportedAsset> TextureAssets { get; }
  bool HasMeshes { get; }
  bool HasTextures { get; }
  bool IsImporting { get; }
  string StatusMessage { get; }
  bool CanUnload { get; }
  IAsyncRelayCommand ImportCommand { get; }
  IAsyncRelayCommand<ImportedAsset?> RemoveAssetCommand { get; }
}

public partial class DesignImportsTabViewModel
{
  public ObservableCollection<string> SessionFolders { get; set; } = new([
    "/mock/session/folder_1",
    "/mock/session/folder_2"
  ]);

  public ObservableCollection<ImportedAsset> MeshAssets { get; } =
  [
    new(1, AssetKind.Mesh, "67P_churyumov", "/mock/67P_churyumov.glb", 24_000, 144_000, [3, 4, 0, 0], null, null),
    new(2, AssetKind.Mesh, "bennu", "/mock/bennu.obj", 9_800, 58_800, [0, 0, 0, 0], null, null),
  ];

  public ObservableCollection<ImportedAsset> TextureAssets { get; } =
  [
    new(3, AssetKind.Texture, "regolith_albedo", "/mock/67P_churyumov.glb#image0", 2048, 2048, [], TextureChannel.Albedo, null),
    new(4, AssetKind.Texture, "regolith_normal", "/mock/67P_churyumov.glb#image1", 2048, 2048, [], TextureChannel.Normal, null),
  ];

  public bool HasMeshes => true;
  public bool HasTextures => true;
  public bool IsImporting => false;
  public string StatusMessage => "Imported: 67P_churyumov.glb";
  public bool CanUnload => true;
  public IAsyncRelayCommand ImportCommand { get; } = new AsyncRelayCommand(() => System.Threading.Tasks.Task.CompletedTask);
  public IAsyncRelayCommand<ImportedAsset?> RemoveAssetCommand { get; } =
    new AsyncRelayCommand<ImportedAsset?>(_ => System.Threading.Tasks.Task.CompletedTask);
}
