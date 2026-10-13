using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
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
  private readonly SaveFileService? _saveFiles;
  private readonly PreferencesService? _preferences;
  private readonly IUnsavedChangesPrompt? _unsavedPrompt;
  private readonly IDisposable? _titleTimer;

  public const string AppName = "AetherVk";
  public const int RecentMenuSize = PreferencesFile.MaxRecentSaves;

  /// <summary>"AetherVk — &lt;save&gt;", with " *" while there are unsaved changes.</summary>
  [ObservableProperty]
  private string _windowTitle = AppName;

  /// <summary>File › Open Recent entries (fixed slots: the menu is built once).</summary>
  private readonly MenuItemViewModel[] _recentItems = new MenuItemViewModel[RecentMenuSize];
  private MenuItemViewModel? _noRecentItem;

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
    IPlatformWindowService platformWindowService,
    SaveFileService? saveFiles = null,
    PreferencesService? preferences = null,
    IUnsavedChangesPrompt? unsavedPrompt = null
  )
  {
    _saveFiles = saveFiles;
    _preferences = preferences;
    _unsavedPrompt = unsavedPrompt;
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

    if (_saveFiles is not null)
    {
      _saveFiles.Changed += () => _dispatcher?.Dispatch(RefreshFileState);
      // the dirty marker follows edits made anywhere in the application
      _titleTimer = System.Reactive.Linq.Observable
        .Interval(TimeSpan.FromSeconds(1))
        .Subscribe(_ => _dispatcher?.Dispatch(RefreshTitle));
    }
    RefreshFileState();
  }

  /// <summary>Whether closing the application should ask first.</summary>
  public bool HasUnsavedChanges => _saveFiles?.IsDirty ?? false;

  /// <summary>Updates the title and the recent list after New, Open or Save.</summary>
  public void RefreshFileState()
  {
    RefreshTitle();
    var recent = _preferences?.RecentSaves ?? [];
    for (int i = 0; i < _recentItems.Length; i++)
    {
      var item = _recentItems[i];
      if (item is null) continue;
      item.IsVisible = i < recent.Count;
      item.Header = i < recent.Count ? $"{i + 1}  {recent[i]}" : string.Empty;
    }
    if (_noRecentItem is not null)
      _noRecentItem.IsVisible = recent.Count == 0;
  }

  public void RefreshTitle()
  {
    if (_saveFiles is null) return;
    var name = _saveFiles.CurrentSaveDir is { } dir ? System.IO.Path.GetFileName(dir) : "Untitled";
    WindowTitle = $"{AppName} — {name}{(_saveFiles.IsDirty ? " *" : string.Empty)}";
  }

  /// <summary>
  /// Before discarding the session (New, Open, closing): true when it may go, after saving it if
  /// the user chose so.
  /// </summary>
  public async Task<bool> ConfirmDiscardAsync()
  {
    if (_saveFiles is null || !_saveFiles.IsDirty || _unsavedPrompt is null) return true;
    var name = _saveFiles.CurrentSaveDir is { } dir ? System.IO.Path.GetFileName(dir) : "Untitled";
    return await _unsavedPrompt.AskAsync(name) switch
    {
      UnsavedChoice.Save => await SaveCoreAsync(),
      UnsavedChoice.Discard => true,
      _ => false,
    };
  }

  [RelayCommand]
  private async Task NewSaveAsync()
  {
    if (_saveFiles is null || !await ConfirmDiscardAsync()) return;
    await _saveFiles.NewAsync();
  }

  [RelayCommand]
  private async Task OpenSaveAsync()
  {
    if (_saveFiles is null) return;
    var path = await _fileDialogService.ShowOpenFileDialogAsync($"Open ({SaveIndex.FileName})", ["json"]);
    if (string.IsNullOrEmpty(path)) return;
    if (!string.Equals(System.IO.Path.GetFileName(path), SaveIndex.FileName, StringComparison.Ordinal))
    {
      _ = _breadcrumbService.ShowMessageAsync("Open", $"Pick the {SaveIndex.FileName} of a save folder.", default, 2);
      return;
    }
    await OpenPathAsync(path!);
  }

  /// <summary>Opens the save at <paramref name="path"/> (folder or index), asking about unsaved changes first.</summary>
  public async Task OpenPathAsync(string path)
  {
    if (_saveFiles is null || !await ConfirmDiscardAsync()) return;
    await _saveFiles.OpenAsync(path);
  }

  [RelayCommand]
  private async Task SaveAsync() => await SaveCoreAsync();

  [RelayCommand]
  private async Task SaveAsAsync() => await SaveAsCoreAsync();

  private async Task<bool> SaveCoreAsync()
  {
    if (_saveFiles is null) return false;
    if (_saveFiles.CurrentSaveDir is { } dir)
      return await _saveFiles.SaveAsync(dir);
    return await SaveAsCoreAsync();
  }

  private async Task<bool> SaveAsCoreAsync()
  {
    if (_saveFiles is null) return false;
    var dir = await _fileDialogService.PickFolderAsync("Save As (choose or create a folder)");
    if (string.IsNullOrEmpty(dir)) return false;
    return await _saveFiles.SaveAsync(dir!);
  }


  private void BuildMenu()
  {
    if (_saveFiles is not null)
    {
      var fileMenu = new MenuItemViewModel { Header = "File" };
      fileMenu.Items.Add(new MenuItemViewModel { Header = "New", Command = NewSaveCommand });
      fileMenu.Items.Add(new MenuItemViewModel { Header = "Open...", Gesture = "Ctrl+O", Command = OpenSaveCommand });
      var recentMenu = new MenuItemViewModel { Header = "Open Recent" };
      _noRecentItem = new MenuItemViewModel { Header = "(none)" };
      recentMenu.Items.Add(_noRecentItem);
      for (int i = 0; i < _recentItems.Length; i++)
      {
        int slot = i;
        _recentItems[i] = new MenuItemViewModel
        {
          IsVisible = false,
          Command = new AsyncRelayCommand(async () =>
          {
            var recent = _preferences?.RecentSaves ?? [];
            if (slot < recent.Count) await OpenPathAsync(recent[slot]);
          }),
        };
        recentMenu.Items.Add(_recentItems[i]);
      }
      fileMenu.Items.Add(recentMenu);
      fileMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
      fileMenu.Items.Add(new MenuItemViewModel { Header = "Save", Gesture = "Ctrl+S", Command = SaveCommand });
      fileMenu.Items.Add(new MenuItemViewModel { Header = "Save As...", Gesture = "Ctrl+Shift+S", Command = SaveAsCommand });
      MainMenu.Add(fileMenu);
    }

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
