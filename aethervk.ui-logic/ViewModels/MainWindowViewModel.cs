using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.ImageExport;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
namespace AetherVk.Logic.ViewModels;

public enum AppTheme
{
  System,
  Light,
  Dark,
}

public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly IFileDialogService _fileDialogService;
  private readonly IWindowService _windowService;
  private readonly IUiThreadDispatcher _dispatcher;
  private readonly SaveFileService? _saveFiles;
  private readonly PreferencesService? _preferences;
  private readonly IUnsavedChangesPrompt? _unsavedPrompt;
  private readonly ImageExportService? _imageExport;
  private readonly ITranslationService? _translation;
  private readonly UiLayoutService? _uiLayout;
  private readonly IDisposable? _titleTimer;
  private readonly IDisposable? _cultureSubscription;

  /// <summary>The radio group of the Layout menu's entries.</summary>
  public const string UiLayoutGroupName = "UiLayout";

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

  /// <summary>The top-bar Orbit / Nucleus / Model strip (null at design time); it acts on
  /// <see cref="ActiveViewport"/>'s camera.</summary>
  public ViewModeBarViewModel? ModeBar { get; }

  partial void OnActiveViewportChanged(Viewport3DViewModel? value)
  {
    if (ModeBar is { } bar)
      bar.ActiveCamera = value?.CameraService;
  }

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
    IUnsavedChangesPrompt? unsavedPrompt = null,
    ViewModeBarViewModel? modeBar = null,
    ImageExportService? imageExport = null,
    ITranslationService? translation = null,
    UiLayoutService? uiLayout = null,
    INativeInputHandlerFactory? inputHandlerFactory = null,
    CometPositionTrackerService? cometTracker = null
  )
  {
    _uiLayout = uiLayout;
    ModeBar = modeBar;
    _imageExport = imageExport;
    _translation = translation;
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

    // the compact layout's panel: its cards bind to the docking tree's own tab view models
    var nucleusMini = new NucleusMiniViewModel(
      runtimeService,
      // the mini-view's native view: an engine and a camera, no main-viewport model behind it
      pointer => new VulkanViewportControlViewModel(inputRouter, inputHandlerFactory!, runtimeService, null!)
      {
        Role = ViewportRole.NucleusMini,
        MiniPointer = pointer,
      },
      translation,
      preferences,
      dispatcher,
      cometTracker?.CometPosition);
    CompactPanel = new CompactPanelViewModel(
      EnsureLayoutTab,
      nucleusMini,
      modeBar,
      ExportImageCommand,
      new RelayCommand(() => SetLayout(UiLayout.Classic)),
      translation,
      dispatcher);

    if (_uiLayout is not null)
    {
      _uiLayout.Changed += OnUiLayoutChanged;
      if (_uiLayout.Current != UiLayout.Classic)
        ApplyUiLayout(_uiLayout.Current);
    }
    if (_translation is not null)
    {
      // the first emission is the current culture, already read by BuildMenu
      _cultureSubscription = System.Reactive.Linq.Observable
        .Skip(_translation.CultureChanged, 1)
        .Subscribe(_ => _dispatcher?.Dispatch(RefreshLayoutHeaders));
    }

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

  /// <summary>The compact layout's panel (shown left of the docking area while
  /// <see cref="IsCompactLayout"/>).</summary>
  public CompactPanelViewModel CompactPanel { get; }

  /// <summary>The docking tree's tab of <paramref name="tabType"/> (created parked when missing);
  /// null without a docking tree or when the tab cannot be made.</summary>
  private TabItemViewModel? EnsureLayoutTab(Type tabType)
  {
    if (DockingManager is not { } docking)
      return null;
    try
    {
      return docking.EnsureTab(tabType);
    }
    catch (InvalidOperationException e)
    {
      Console.WriteLine($"[MainWindowViewModel] compact panel: no {tabType.Name}: {e.Message}");
      return null;
    }
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

  private bool _disposed;

  /// <summary>Stops the title timer: the host disposes this before the services it reads.</summary>
  public void Dispose()
  {
    _disposed = true;
    _titleTimer?.Dispose();
    _cultureSubscription?.Dispose();
    if (_uiLayout is not null)
      _uiLayout.Changed -= OnUiLayoutChanged;
    CompactPanel.Dispose();
  }

  /// <summary>The open save's name: the <c>.avk</c> without its extension, else the folder's.</summary>
  private static string SaveName(SaveFileService saves) =>
    saves.CurrentArchivePath is { } avk ? System.IO.Path.GetFileNameWithoutExtension(avk)
    : saves.CurrentSaveDir is { } dir ? System.IO.Path.GetFileName(dir)
    : "Untitled";

  public void RefreshTitle()
  {
    // a tick dispatched before the host stopped must not read the disposed services
    if (_saveFiles is null || _disposed) return;
    var name = SaveName(_saveFiles);
    WindowTitle = $"{AppName} — {name}{(_saveFiles.IsDirty ? " *" : string.Empty)}";
  }

  /// <summary>
  /// Before discarding the session (New, Open, closing): true when it may go, after saving it if
  /// the user chose so.
  /// </summary>
  public async Task<bool> ConfirmDiscardAsync()
  {
    if (_saveFiles is null || !_saveFiles.IsDirty || _unsavedPrompt is null) return true;
    var name = SaveName(_saveFiles);
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
    var path = await _fileDialogService.ShowOpenFileDialogAsync($"Open (.avk or {SaveIndex.FileName})", ["avk", "json"]);
    if (string.IsNullOrEmpty(path)) return;
    if (!SaveArchive.IsArchive(path!)
        && !string.Equals(System.IO.Path.GetFileName(path), SaveIndex.FileName, StringComparison.Ordinal))
    {
      _ = _breadcrumbService.ShowMessageAsync("Open", $"Pick an .avk save, or the {SaveIndex.FileName} of a save folder.", default, 2);
      return;
    }
    await OpenPathAsync(path!);
  }

  /// <summary>Opens the save at <paramref name="path"/> (.avk, folder or index), asking about unsaved changes first.</summary>
  public async Task OpenPathAsync(string path)
  {
    if (_saveFiles is null || !await ConfirmDiscardAsync()) return;
    await _saveFiles.OpenAsync(path);
  }

  [RelayCommand]
  private async Task SaveAsync() => await SaveCoreAsync();

  [RelayCommand]
  private async Task SaveAsAsync() => await SaveAsCoreAsync();

  [RelayCommand]
  private async Task SaveAsFolderAsync() => await SaveAsFolderCoreAsync();

  private async Task<bool> SaveCoreAsync()
  {
    if (_saveFiles is null) return false;
    if (_saveFiles.CurrentSavePath is { } path)
      return await _saveFiles.SaveAsync(path);
    return await SaveAsCoreAsync();
  }

  /// <summary>Save As: one <c>.avk</c> file (the default since the reviewer report of 2026-10-29).</summary>
  private async Task<bool> SaveAsCoreAsync()
  {
    if (_saveFiles is null) return false;
    var path = await _fileDialogService.ShowSaveFileDialogAsync("Save As", "avk", ["avk"]);
    if (string.IsNullOrEmpty(path)) return false;
    if (!SaveArchive.IsArchive(path!)) path += SaveArchive.Extension;
    return await _saveFiles.SaveAsync(path!);
  }

  /// <summary>The last export's outcome, for the status line and tests.</summary>
  [ObservableProperty]
  private string? _exportStatus;

  partial void OnExportStatusChanged(string? value) => CompactPanel.ExportStatus = value;

  /// <summary>
  /// File › Export image...: asks for a .png name and writes <c>name_model.png</c> (the 3D viewport
  /// as the engine draws it, labels hidden) and, when a FITS billboard is visible,
  /// <c>name_with_fits.png</c> (the same image with the billboard composited at its on-screen
  /// geometry). The outcome goes to the status line.
  /// </summary>
  [RelayCommand]
  private async Task ExportImageAsync()
  {
    ulong pe = ActiveViewport?.PresentationEngineId ?? _runtimeService.PresentationEngineId ?? 0;
    if (_imageExport is null || pe == 0)
    {
      ReportExport(T("Export_NoViewport", "No 3D viewport to export."), 3);
      return;
    }
    var path = await _fileDialogService.ShowSaveFileDialogAsync(T("Export_DialogTitle", "Export image"), "png", ["png"]);
    if (string.IsNullOrEmpty(path)) return;
    ImageExportResult? result;
    try
    {
      result = await _imageExport.ExportAsync(pe, path!, ActiveViewport?.CameraService?.ViewportDesktopFrame);
    }
    catch (Exception e)
    {
      Console.WriteLine($"[MainWindowViewModel] export image failed: {e}");
      result = null;
    }
    if (result is null)
    {
      ReportExport(T("Export_Failed", "Export failed: the viewport could not be captured."), 3);
      return;
    }
    var (modelName, fitsName) = ImageExportService.FileNames(path!);
    string msg = result.FitsComposited
      ? T("Export_Written", "Written {model} and {fits}").Replace("{model}", System.IO.Path.GetFileName(modelName)).Replace("{fits}", System.IO.Path.GetFileName(fitsName))
      : T("Export_ModelOnly", "No FITS billboard visible: model only. Written {model}").Replace("{model}", System.IO.Path.GetFileName(modelName));
    ReportExport(msg, 2);
  }

  private void ReportExport(string message, int status)
  {
    ExportStatus = message;
    _ = _breadcrumbService.ShowMessageAsync(T("Export_DialogTitle", "Export image"), message, default, status);
  }

  /// <summary>A UI string by resx key, <paramref name="fallback"/> without a translation service or
  /// when the key is missing.</summary>
  private string T(string key, string fallback)
  {
    var s = _translation?.GetString(key);
    return string.IsNullOrEmpty(s) || s == $"[{key}]" ? fallback : s!;
  }

  /// <summary>Save As Folder: the folder layout (docs/savefile.md), for inspecting a save.</summary>
  private async Task<bool> SaveAsFolderCoreAsync()
  {
    if (_saveFiles is null) return false;
    var dir = await _fileDialogService.PickFolderAsync("Save As Folder (choose or create a folder)");
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
      fileMenu.Items.Add(new MenuItemViewModel { Header = "Save As Folder...", Command = SaveAsFolderCommand });
      fileMenu.Items.Add(new MenuItemViewModel { IsSeparator = true });
      fileMenu.Items.Add(new MenuItemViewModel { Header = "Export image...", Command = ExportImageCommand });
      MainMenu.Add(fileMenu);
    }

    var editMenu = new MenuItemViewModel { Header = "Edit" };
    editMenu.Items.Add(new MenuItemViewModel { Header = "Settings...", Gesture = "Cmd+OemComma", Command = OpenSettingsCommand });
    editMenu.Items.Add(new MenuItemViewModel { Header = "Toggle Theme", Command = ToggleThemeCommand });

    MainMenu.Add(editMenu);

    var current = _uiLayout?.Current ?? UiLayout.Classic;
    _layoutMenu = new MenuItemViewModel();
    _classicLayoutItem = new MenuItemViewModel
    {
      ToggleType = MenuToggle.Radio,
      GroupName = UiLayoutGroupName,
      IsChecked = current == UiLayout.Classic,
      Command = new RelayCommand(() => SetLayout(UiLayout.Classic)),
    };
    _compactLayoutItem = new MenuItemViewModel
    {
      ToggleType = MenuToggle.Radio,
      GroupName = UiLayoutGroupName,
      IsChecked = current == UiLayout.Compact,
      Command = new RelayCommand(() => SetLayout(UiLayout.Compact)),
    };
    _layoutMenu.Items.Add(_classicLayoutItem);
    _layoutMenu.Items.Add(_compactLayoutItem);
    RefreshLayoutHeaders();
    MainMenu.Add(_layoutMenu);
  }

  private MenuItemViewModel? _layoutMenu;
  private MenuItemViewModel? _classicLayoutItem;
  private MenuItemViewModel? _compactLayoutItem;

  /// <summary>Layout › Compact is shown: the docking area holds the 3D viewport only.</summary>
  [ObservableProperty]
  private bool _isCompactLayout;

  /// <summary>Why the last layout switch was refused (null after a switch that went through).</summary>
  [ObservableProperty]
  private string? _layoutStatus;

  /// <summary>The Layout menu's headers in the current language.</summary>
  private void RefreshLayoutHeaders()
  {
    if (_layoutMenu is null) return;
    _layoutMenu.Header = T("Menu_Layout", "Layout");
    _classicLayoutItem!.Header = T("Menu_Layout_Classic", "Classic");
    _compactLayoutItem!.Header = T("Menu_Layout_Compact", "Compact");
  }

  /// <summary>
  /// Layout › Classic / Compact. Compact is refused (the radio stays on Classic and the status line
  /// says why) when the docking area has no 3D viewport to show alone; otherwise the choice becomes
  /// the current layout and the preferred one for the next start.
  /// </summary>
  [RelayCommand]
  private void SetLayout(UiLayout layout)
  {
    if (layout == UiLayout.Compact && DockingManager is { } docking && !docking.CanEnterViewportOnly)
    {
      var why = T("Layout_CompactUnavailable", "Compact layout needs the 3D viewport tab to be visible.");
      LayoutStatus = why;
      _ = _breadcrumbService.ShowMessageAsync(T("Menu_Layout", "Layout"), why, default, 2);
      // the menu checked the clicked entry before this ran: put the current one back
      UpdateLayoutChecks(_uiLayout?.Current ?? (IsCompactLayout ? UiLayout.Compact : UiLayout.Classic));
      return;
    }
    LayoutStatus = null;
    if (_uiLayout is null)
      ApplyUiLayout(layout);
    else
    {
      _uiLayout.Set(layout);
      // also when Set saw no change and raised nothing: the menu's own check may be stale
      ApplyUiLayout(_uiLayout.Current);
    }
    _preferences?.SetLastUiLayout(layout);
  }

  private void OnUiLayoutChanged(object? sender, UiLayout layout)
  {
    if (_dispatcher is null || _dispatcher.CheckAccess())
      ApplyUiLayout(layout);
    else
      _dispatcher.Dispatch(() => ApplyUiLayout(layout));
  }

  /// <summary>Shows <paramref name="layout"/>: the docking area's viewport-only mode and the radio entries.</summary>
  private void ApplyUiLayout(UiLayout layout)
  {
    bool compact = layout == UiLayout.Compact;
    bool refused = false;
    // SetViewportOnly refuses (false) when there is no viewport to keep: stay Classic then
    if (compact != IsCompactLayout && DockingManager is { } docking && !docking.SetViewportOnly(compact) && compact)
    {
      compact = false;
      refused = true;
    }
    IsCompactLayout = compact;
    if (compact)
      CompactPanel.Activate();
    else
      CompactPanel.Deactivate();
    UpdateLayoutChecks(compact ? UiLayout.Compact : UiLayout.Classic);
    if (refused)
    {
      // a layout asked from elsewhere (an opened save, the preferences): say why, and leave the
      // service at the layout that is shown
      var why = T("Layout_CompactUnavailable", "Compact layout needs the 3D viewport tab to be visible.");
      LayoutStatus = why;
      _ = _breadcrumbService?.ShowMessageAsync(T("Menu_Layout", "Layout"), why, default, 2);
      _uiLayout?.Set(UiLayout.Classic);
    }
  }

  private void UpdateLayoutChecks(UiLayout layout)
  {
    if (_classicLayoutItem is not null) _classicLayoutItem.IsChecked = layout == UiLayout.Classic;
    if (_compactLayoutItem is not null) _compactLayoutItem.IsChecked = layout == UiLayout.Compact;
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
