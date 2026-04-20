using System.Collections.ObjectModel;
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

  public ImportedModelItem(ulong id, string name, string fullPath)
  {
    Id = id;
    Name = name;
    FullPath = fullPath;
  }
  [RelayCommand]
  private void Spawn()
  {
    WeakReferenceMessenger.Default.Send(new OpenSpawnMeshDialogMessage(this));
  }

  [RelayCommand]
  private void ViewMesh()
  {
    WeakReferenceMessenger.Default.Send(new OpenMeshViewerMessage(this));
  }

  [RelayCommand]
  private void Unload()
  {
    var runtime = ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtime?.UnloadModel(Id);
    
    // Signal UI to remove from list
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

public partial class MainWindowViewModel : ViewModelBase, IRecipient<ModelUnloadedMessage>
{
  [ObservableProperty]
  private DockingManagerViewModel _dockingManager = new();

  [ObservableProperty]
  private AppTheme _currentTheme;

  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  public ObservableCollection<BreadcrumbMessage>? Breadcrumbs =>
    (ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService)?.Messages;

  public MainWindowViewModel()
  {
    // Set initial theme to system default
    CurrentTheme = AppTheme.System;
    WeakReferenceMessenger.Default.Register<ModelUnloadedMessage>(this);
  }

  public void Receive(ModelUnloadedMessage message)
  {
    ImportedModels.Remove(message.Model);
  }

  [RelayCommand]
  private void ImportModel()
  {
    WeakReferenceMessenger.Default.Send(new ImportModelRequestMessage());
  }

  [RelayCommand]
  private void ImportImage()
  {
    WeakReferenceMessenger.Default.Send(new ImportImageRequestMessage());
  }

  [RelayCommand]
  private void OpenImportedModelsDialog()
  {
    WeakReferenceMessenger.Default.Send(new OpenImportedModelsDialogMessage());
  }

  [RelayCommand]
  private void ToggleTheme()
  {
    // Simple toggle: if Dark -> Light, otherwise -> Dark (covers Light and System)
    CurrentTheme = CurrentTheme == AppTheme.Dark ? AppTheme.Light : AppTheme.Dark;
  }

  [RelayCommand]
  private void RotateCameraLeft()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.RotateCamera(10.0f, 0.0f);
  }

  [RelayCommand]
  private void RotateCameraRight()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.RotateCamera(-10.0f, 0.0f);
  }

  [RelayCommand]
  private void ZoomIn()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.ZoomCamera(2.0f);
  }

  [RelayCommand]
  private void ZoomOut()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.ZoomCamera(-2.0f);
  }

  [RelayCommand]
  private void ResetCamera()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.ResetCamera();
  }
}
