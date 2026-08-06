using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
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
  private readonly INativeRuntimeService _runtimeService;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly IFileDialogService _fileDialogService;
  private readonly IWindowService _windowService;
  private readonly IUiThreadDispatcher _dispatcher;

  [ObservableProperty]
  private DockingManagerViewModel _dockingManager;

  [ObservableProperty]
  private AppTheme _currentTheme;

  [ObservableProperty]
  private Viewport3DViewModel? _activeViewport;

  [ObservableProperty]
  private bool _isViewportFocused;

  /// <summary>Avalonia window should call AttachToWindow on this in OnOpened can call Dispose on
  /// OnClosed</summary>
  public IWindowInputRouter InputRouter { get; }

  public ObservableCollection<BreadcrumbMessage>? Breadcrumbs => _breadcrumbService.Messages;

  public MainWindowViewModel(
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IFileDialogService fileDialogService,
    IWindowService windowService,
    IWindowInputRouter inputRouter,
    DockingManagerViewModel dockingManager,
    IUiThreadDispatcher dispatcher
  )
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _fileDialogService = fileDialogService;
    _windowService = windowService;
    _dockingManager = dockingManager;
    _dispatcher = dispatcher;
    InputRouter = inputRouter;

    // Set initial theme to system default
    CurrentTheme = AppTheme.System;
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
        await _windowService.ShowSpawnImageDialogAsync(result!);
      }
      catch (Exception ex)
      {
        _ = _breadcrumbService.ShowMessageAsync(
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
  private async Task OpenSpawnBillboardDialogAsync()
  {
    await _windowService.ShowSpawnBillboardDialogAsync();
  }

  [RelayCommand]
  private async Task OpenConsoleAsync()
  {
    await _windowService.ShowSettingsDialogAsync();
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
}
