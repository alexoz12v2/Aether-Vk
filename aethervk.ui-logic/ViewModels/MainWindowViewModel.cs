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

  public bool IsSystemThemeDark { get; set; }

  public ObservableCollection<MenuItemViewModel> MainMenu { get; } = new ObservableCollection<MenuItemViewModel>();
  private MenuItemViewModel? _snapObserverMenuItem;

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

    BuildMenu();
  }

  private void BuildMenu()
  {
    var fileMenu = new MenuItemViewModel { Header = "File" };
    fileMenu.Items.Add(new MenuItemViewModel { Header = "Open" });
    fileMenu.Items.Add(new MenuItemViewModel { Header = "Save" });
    fileMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
    fileMenu.Items.Add(new MenuItemViewModel { Header = "Import Model" });
    fileMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
    fileMenu.Items.Add(new MenuItemViewModel { Header = "Exit" });

    var editMenu = new MenuItemViewModel { Header = "Edit" };
    editMenu.Items.Add(new MenuItemViewModel { Header = "Copy" });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Paste" });
    editMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });

    var sceneMenu = new MenuItemViewModel { Header = "Scene" };
    sceneMenu.Items.Add(new MenuItemViewModel { Header = "Spawn Comet" });
    editMenu.Items.Add(sceneMenu);

    _snapObserverMenuItem = new MenuItemViewModel { Header = "Snap Observer", IsVisible = false };
    editMenu.Items.Add(_snapObserverMenuItem);

    editMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Settings...", Gesture = "Cmd+OemComma", Command = OpenSettingsCommand });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Imported Models", Command = OpenImportedModelsDialogCommand });
    editMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Toggle Theme", Command = ToggleThemeCommand });

    MainMenu.Add(fileMenu);
    MainMenu.Add(editMenu);
  }

  partial void OnActiveViewportChanged(Viewport3DViewModel? oldValue, Viewport3DViewModel? newValue)
  {
      if (oldValue != null)
      {
          oldValue.PropertyChanged -= ActiveViewport_PropertyChanged;
      }
      if (newValue != null)
      {
          newValue.PropertyChanged += ActiveViewport_PropertyChanged;
      }
      UpdateSnapObserverVisibility();
  }

  private void ActiveViewport_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
  {
      if (e.PropertyName == nameof(Viewport3DViewModel.IsEarthObserverMode))
      {
          UpdateSnapObserverVisibility();
      }
  }

  private void UpdateSnapObserverVisibility()
  {
      if (_snapObserverMenuItem != null)
      {
          _snapObserverMenuItem.IsVisible = ActiveViewport?.IsEarthObserverMode ?? false;
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
