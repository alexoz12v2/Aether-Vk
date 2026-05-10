using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public enum AppTheme
{
  System,
  Light,
  Dark,
}

public class ImportModelRequestMessage { }

public class ImportImageRequestMessage { }

public class OpenImportedModelsDialogMessage { }

public partial class ImportedModelItem : ObservableObject
{
  public ulong Id { get; }

  [ObservableProperty]
  private string _name = "";

  public string FullPath { get; }
  private readonly NativeRuntimeService _runtimeService;
  private readonly IWindowService _windowService;
  public System.Collections.Generic.List<ulong> SpawnedInstanceIds { get; } = new();

  public ImportedModelItem(
    ulong id,
    string name,
    string fullPath,
    NativeRuntimeService runtimeService,
    IWindowService windowService
  )
  {
    Id = id;
    Name = name;
    FullPath = fullPath;
    _runtimeService = runtimeService;
    _windowService = windowService;
  }

  [RelayCommand]
  private async Task SpawnAsync()
  {
    var instanceId = await _windowService.ShowSpawnMeshDialogAsync(Id.ToString(), Name);
    if (instanceId > 0)
    {
      SpawnedInstanceIds.Add(instanceId);
    }
  }

  [RelayCommand]
  private async Task ViewMeshAsync()
  {
    await _windowService.OpenMeshViewerAsync(Id.ToString());
  }

  [RelayCommand]
  private void Unload()
  {
    // Clean up instances across all known scenes
    // For now, we only spawn into Scene 1 via ShowSpawnMeshDialogAsync, but let's assume we remove them generally.
    foreach (var instanceId in SpawnedInstanceIds)
    {
      _runtimeService.RemoveEntity(1, instanceId); // Assuming Scene 1 since it's hardcoded in AvaloniaWindowService
    }
    SpawnedInstanceIds.Clear();

    _runtimeService.UnloadModel(Id);
    WeakReferenceMessenger.Default.Send(new ModelUnloadedMessage(this));
  }
}

public class OpenSpawnMeshDialogMessage
{
  public ImportedModelItem Model { get; }

  public OpenSpawnMeshDialogMessage(ImportedModelItem model) => Model = model;
}

public class OpenMeshViewerMessage
{
  public ImportedModelItem Model { get; }

  public OpenMeshViewerMessage(ImportedModelItem model) => Model = model;
}

public class ModelUnloadedMessage
{
  public ImportedModelItem Model { get; }

  public ModelUnloadedMessage(ImportedModelItem model) => Model = model;
}

public struct CameraActionParams
{
  public ulong SceneId { get; set; }
  public ulong CameraEntityId { get; set; }
}

public partial class MainWindowViewModel : ViewModelBase, IRecipient<ModelUnloadedMessage>
{
  private readonly NativeRuntimeService _runtimeService;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly IFileDialogService _fileDialogService;
  private readonly IWindowService _windowService;

  [ObservableProperty]
  private DockingManagerViewModel _dockingManager;

  [ObservableProperty]
  private AppTheme _currentTheme;

  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  public ObservableCollection<BreadcrumbMessage>? Breadcrumbs => _breadcrumbService.Messages;

  public MainWindowViewModel(
    NativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IFileDialogService fileDialogService,
    IWindowService windowService,
    DockingManagerViewModel dockingManager
  )
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _fileDialogService = fileDialogService;
    _windowService = windowService;
    _dockingManager = dockingManager;

    // Set initial theme to system default
    CurrentTheme = AppTheme.System;
    WeakReferenceMessenger.Default.Register<ModelUnloadedMessage>(this);
  }

  public void Receive(ModelUnloadedMessage message)
  {
    ImportedModels.Remove(message.Model);
  }

  [RelayCommand]
  private async Task ImportModelAsync()
  {
    var filters = new[] { "gltf", "glb" };
    var result = await _fileDialogService.ShowOpenFileDialogAsync("Import 3D Model", filters);

    if (!string.IsNullOrEmpty(result))
    {
      var modelId = await _runtimeService.ImportModelAsync(result);
      if (modelId > 0)
      {
        if (!System.Linq.Enumerable.Any(ImportedModels, m => m.Id == modelId))
        {
          var fileName = System.IO.Path.GetFileName(result);
          ImportedModels.Add(
            new ImportedModelItem(modelId, fileName, result, _runtimeService, _windowService)
          );
        }
      }
    }
  }

  [RelayCommand]
  private async Task ImportImageAsync()
  {
    var filters = new[] { "png", "jpg", "jpeg", "bmp", "tga" };
    var result = await _fileDialogService.ShowOpenFileDialogAsync("Import Image", filters);

    if (!string.IsNullOrEmpty(result))
    {
      try
      {
        // Actually, Avalonia isn't allowed here, but System.Drawing isn't either.
        // We might need an interface to get image dimensions.
        // For now, let's keep the actual logic from MainWindow.axaml.cs inside an ImageService, or use IWindowService.ShowSpawnImageDialogAsync(result)
        await _windowService.ShowSpawnImageDialogAsync(result);
      }
      catch (System.Exception ex)
      {
        _breadcrumbService.ShowMessageAsync(
          "Import Error",
          $"Failed to load image: {ex.Message}",
          default,
          3
        );
      }
    }
  }

  [RelayCommand]
  private async Task OpenImportedModelsDialogAsync()
  {
    await _windowService.ShowManageImportsDialogAsync();
  }

  [RelayCommand]
  private async Task OpenSettingsAsync()
  {
    await _windowService.ShowSettingsDialogAsync();
  }

  [RelayCommand]
  private void ToggleTheme()
  {
    // Simple toggle: if Dark -> Light, otherwise -> Dark (covers Light and System)
    CurrentTheme = CurrentTheme == AppTheme.Dark ? AppTheme.Light : AppTheme.Dark;
  }

  [RelayCommand]
  private void RotateCameraLeft(CameraActionParams p)
  {
    _runtimeService.RotateCamera(p.SceneId, p.CameraEntityId, 10.0f, 0.0f);
  }

  [RelayCommand]
  private void RotateCameraRight(CameraActionParams p)
  {
    _runtimeService.RotateCamera(p.SceneId, p.CameraEntityId, -10.0f, 0.0f);
  }

  [RelayCommand]
  private void ZoomIn(CameraActionParams p)
  {
    _runtimeService.ZoomCamera(p.SceneId, p.CameraEntityId, 2.0f);
  }

  [RelayCommand]
  private void ZoomOut(CameraActionParams p)
  {
    _runtimeService.ZoomCamera(p.SceneId, p.CameraEntityId, -2.0f);
  }

  [RelayCommand]
  private void ResetCamera(CameraActionParams p)
  {
    _runtimeService.ResetCamera(p.SceneId, p.CameraEntityId);
  }
}
