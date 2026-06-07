using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class AlmanacExplorerViewModel : TabItemViewModel
{
  private readonly NativeRuntimeService _runtimeService;
  private readonly ILocalStorageService _localStorage;
  private readonly ConsoleService _console;
  private readonly BreadcrumbService _breadcrumb;

  [ObservableProperty]
  private ObservableCollection<SpkFileModel> _loadedFiles = new();

  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isLoading;

  [ObservableProperty]
  private bool _hasSelection;

  [ObservableProperty]
  private bool _isSelectionLoaded;

  public AlmanacExplorerViewModel(
    NativeRuntimeService runtimeService,
    IUiThreadDispatcher uiThreadDispatcher,
    ILocalStorageService localStorage,
    ConsoleService console,
    BreadcrumbService breadcrumb
  )
    : base("Almanac Explorer")
  {
    _runtimeService = runtimeService;
    _localStorage = localStorage;
    _console = console;
    _breadcrumb = breadcrumb;

    _runtimeService.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(NativeRuntimeService.IsInitialized))
      {
        uiThreadDispatcher.Dispatch(() =>
        {
          IsInitialized = _runtimeService.IsInitialized;
          RefreshLoadedFiles();
        });
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    RefreshLoadedFiles();
  }

  [RelayCommand]
  private void RefreshLoadedFiles()
  {
    LoadedFiles.Clear();
    HasSelection = false;

    // 1. Get all SPK files from persistent storage (spk-kernels)
    var spkDir = _localStorage.GetPersistentPath("spk-kernels");
    var localFiles = Directory.Exists(spkDir)
      ? Directory.GetFiles(spkDir, "*.spk").ToList()
      : new System.Collections.Generic.List<string>();

    // 2. Get files currently loaded in the native engine
    var activeNativeFiles = _runtimeService.IsInitialized
      ? _runtimeService.GetLoadedAlmanacFiles()
      : System.Array.Empty<string>();

    // 3. Merge default almanacs loaded natively into the list if they aren't in spk-kernels
    var allFiles = new System.Collections.Generic.HashSet<string>(localFiles);
    foreach (var nativeFile in activeNativeFiles)
    {
      allFiles.Add(nativeFile);
    }

    foreach (var file in allFiles)
    {
      var isLoaded = activeNativeFiles.Contains(file);
      var model = new SpkFileModel(file) { IsLoaded = isLoaded };
      model.PropertyChanged += OnFileModelPropertyChanged;
      LoadedFiles.Add(model);
    }

    UpdateSelectability();
  }

  private void OnFileModelPropertyChanged(
    object? sender,
    System.ComponentModel.PropertyChangedEventArgs e
  )
  {
    if (e.PropertyName == nameof(SpkFileModel.IsSelected))
    {
      UpdateSelectability();
    }
  }

  private void UpdateSelectability()
  {
    var selectedItems = LoadedFiles.Where(f => f.IsSelected).ToList();
    HasSelection = selectedItems.Count > 0;

    if (!HasSelection)
    {
      // Nothing selected, everything is selectable
      foreach (var file in LoadedFiles)
      {
        file.IsSelectable = true;
      }
      return;
    }

    // Determine the state of the current selection (either all loaded or all unloaded)
    IsSelectionLoaded = selectedItems.First().IsLoaded;

    // Only items matching the selection's loaded state are selectable
    foreach (var file in LoadedFiles)
    {
      if (file.IsSelected)
      {
        file.IsSelectable = true; // Selected items are always selectable (to unselect)
      }
      else
      {
        file.IsSelectable = file.IsLoaded == IsSelectionLoaded;
      }
    }
  }

  [RelayCommand]
  private async Task LoadDefaultAlmanacs()
  {
    if (!_runtimeService.IsInitialized)
      return;

    IsLoading = true;
    await Task.Run(() => _runtimeService.LoadDefaultAlmanacs());
    IsLoading = false;

    RefreshLoadedFiles();
  }

  [RelayCommand]
  private async Task LoadSelected()
  {
    var itemsToLoad = LoadedFiles.Where(f => f.IsSelected && !f.IsLoaded).ToList();
    if (itemsToLoad.Count == 0)
      return;

    IsLoading = true;

    foreach (var item in itemsToLoad)
    {
      _console.Log($"[AlmanacExplorer] Loading {item.FileName}...");
      bool success = await _runtimeService.LoadAlmanacFileAsync(item.FilePath);
      if (success)
      {
        _console.Log($"[AlmanacExplorer] Loaded {item.FileName} successfully.");
        var msg = new AetherVk.Logic.Messages.AlmanacUpdatedMessage
        {
          SceneId = 0,
          FilePath = item.FilePath,
          WasLoaded = true,
        };
        WeakReferenceMessenger.Default.Send(msg);
      }
      else
      {
        _console.Log($"[AlmanacExplorer] Failed to load {item.FileName}.");
      }
    }

    IsLoading = false;
    RefreshLoadedFiles();
  }

  [RelayCommand]
  private async Task UnloadSelected()
  {
    var itemsToUnload = LoadedFiles.Where(f => f.IsSelected && f.IsLoaded).ToList();
    if (itemsToUnload.Count == 0)
      return;

    IsLoading = true;

    foreach (var item in itemsToUnload)
    {
      _console.Log($"[AlmanacExplorer] Unloading {item.FileName}...");
      bool success = await _runtimeService.UnloadAlmanacFileAsync(item.FilePath);
      if (success)
      {
        _console.Log($"[AlmanacExplorer] Unloaded {item.FileName} successfully.");
        var msg = new AetherVk.Logic.Messages.AlmanacUpdatedMessage
        {
          SceneId = 0,
          FilePath = item.FilePath,
          WasLoaded = false,
        };
        WeakReferenceMessenger.Default.Send(msg);
      }
      else
      {
        _console.Log($"[AlmanacExplorer] Failed to unload {item.FileName}.");
      }
    }

    IsLoading = false;
    RefreshLoadedFiles();
  }
}
