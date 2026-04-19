using System.Collections.ObjectModel;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public enum AppTheme
{
  System,
  Light,
  Dark,
}

public partial class MainWindowViewModel : ViewModelBase
{
  [ObservableProperty]
  private DockingManagerViewModel _dockingManager = new();

  [ObservableProperty]
  private AppTheme _currentTheme;

  public ObservableCollection<BreadcrumbMessage>? Breadcrumbs =>
    (ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService)?.Messages;

  public MainWindowViewModel()
  {
    // Set initial theme to system default
    CurrentTheme = AppTheme.System;
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
