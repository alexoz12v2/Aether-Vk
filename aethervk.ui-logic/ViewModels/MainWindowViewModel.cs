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

  /// <summary>The dedicated breadcrumb overlay ViewModel — sole UI recipient of BreadcrumbService events.</summary>
  public BreadcrumbWindowViewModel BreadcrumbViewModel { get; }

  /// <summary>Exposed so MainWindow code-behind can pass it to OverlaySynchronizer.</summary>
  public IPlatformWindowService PlatformWindowService { get; }

  public bool IsSystemThemeDark { get; set; }

  public ObservableCollection<MenuItemViewModel> MainMenu { get; } = new ObservableCollection<MenuItemViewModel>();


  public MainWindowViewModel(
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IFileDialogService fileDialogService,
    IWindowService windowService,
    IWindowInputRouter inputRouter,
    DockingManagerViewModel dockingManager,
    IUiThreadDispatcher dispatcher,
    BreadcrumbWindowViewModel breadcrumbWindowViewModel,
    IPlatformWindowService platformWindowService
  )
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _fileDialogService = fileDialogService;
    _windowService = windowService;
    _dockingManager = dockingManager;
    _dispatcher = dispatcher;
    InputRouter = inputRouter;
    BreadcrumbViewModel = breadcrumbWindowViewModel;
    PlatformWindowService = platformWindowService;

    // Set initial theme to system default
    CurrentTheme = AppTheme.System;

    BuildMenu();
  }


  private void BuildMenu()
  {
    var editMenu = new MenuItemViewModel { Header = "Edit" };
    editMenu.Items.Add(new MenuItemViewModel { Header = "Settings...", Gesture = "Cmd+OemComma", Command = OpenSettingsCommand });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Toggle Theme", Command = ToggleThemeCommand });

    MainMenu.Add(editMenu);
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
    if (CurrentTheme == AppTheme.System)
    {
        CurrentTheme = IsSystemThemeDark ? AppTheme.Light : AppTheme.Dark;
    }
    else
    {
        CurrentTheme = CurrentTheme == AppTheme.Dark ? AppTheme.Light : AppTheme.Dark;
    }
  }
}
