using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace AetherVk;

public partial class MainWindow : Window
{
  private UI.BreadcrumbWindow?    _breadcrumbWindow;
  private UI.OverlaySynchronizer? _breadcrumbSynchronizer;
  private UI.DesktopBillboardWindowManager? _billboards;

  public MainWindow()
  {
    InitializeComponent();
#if DEBUG
    // Opens devtools window for this window
    this.AttachDevTools();
#endif

    KeyDown += OnKeyDown;
    TitleDragArea.PointerPressed += OnTitleDragPressed;
    AddHandler(
      GotFocusEvent,
      OnElementGotFocus,
      Avalonia.Interactivity.RoutingStrategies.Bubble
    );
    DataContextChanged += (_, _) => OnViewModelChanged();
  }

  // ── Classic / Compact: the widths of the three fixed columns ──────────────────

  /// <summary>The compact panel's least width, DIPs.</summary>
  public const double CompactPanelMinWidth = 300;

  private Logic.ViewModels.MainWindowViewModel? _layoutVm;
  // the user's splitter position, kept across a visit to the classic layout
  private GridLength _compactPanelWidth = new(3, GridUnitType.Star);
  private GridLength _compactDockingWidth = new(7, GridUnitType.Star);
  private bool _compactColumns;

  private void OnViewModelChanged()
  {
    if (_layoutVm is not null)
      _layoutVm.PropertyChanged -= OnViewModelPropertyChanged;
    _layoutVm = DataContext as Logic.ViewModels.MainWindowViewModel;
    if (_layoutVm is not null)
      _layoutVm.PropertyChanged += OnViewModelPropertyChanged;
    ApplyLayoutColumns();
  }

  private void OnViewModelPropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
  {
    if (e.PropertyName == nameof(Logic.ViewModels.MainWindowViewModel.IsCompactLayout))
      ApplyLayoutColumns();
  }

  /// <summary>
  /// Compact: panel 3* (at least <see cref="CompactPanelMinWidth"/>), splitter, docking area 7*, or
  /// where the user left the splitter. Classic: the docking area alone, exactly as without a panel.
  /// Only widths and visibility change; no control changes parent or column.
  /// </summary>
  private void ApplyLayoutColumns()
  {
    bool compact = _layoutVm?.IsCompactLayout ?? false;
    var columns = WorkArea.ColumnDefinitions;
    if (_compactColumns && !compact)
    {
      _compactPanelWidth = columns[0].Width;
      _compactDockingWidth = columns[2].Width;
    }
    _compactColumns = compact;
    CompactPanelHost.IsVisible = compact;
    CompactSplitter.IsVisible = compact;
    if (compact)
    {
      columns[0].MinWidth = CompactPanelMinWidth;
      columns[0].Width = _compactPanelWidth;
      columns[1].Width = GridLength.Auto;
      columns[2].Width = _compactDockingWidth;
    }
    else
    {
      columns[0].MinWidth = 0;
      columns[0].Width = new GridLength(0);
      columns[1].Width = new GridLength(0);
      columns[2].Width = new GridLength(1, GridUnitType.Star);
    }
  }

  protected override void OnOpened(EventArgs e)
  {
    base.OnOpened(e);
    // The visual tree is live at this point — safe to subscribe to tunneling events.
    if (DataContext is Logic.ViewModels.MainWindowViewModel vm)
    {
      vm.InputRouter.AttachToWindow(this);
      UI.MenuMapper.ApplyMenu(this, MenuContainer, vm.MainMenu);

      // Breadcrumb overlay — tracks the full MainWindow frame, not the NativeControlHost.
      _breadcrumbWindow = new UI.BreadcrumbWindow { DataContext = vm.BreadcrumbViewModel };
      _breadcrumbSynchronizer = new UI.OverlaySynchronizer(
        mainWindow: this,
        overlayWindow: _breadcrumbWindow,
        nativeHost: this,                     // follow full MainWindow, not the viewport
        platformWindowService: vm.PlatformWindowService
      );

      // Desktop billboards: one window per visible billboard of DesktopBillboardService.
      if (App.Host?.Services.GetService(typeof(UI.DesktopBillboardWindowManager)) is UI.DesktopBillboardWindowManager billboards)
      {
        _billboards = billboards;
        billboards.Attach(this);
      }
      // FITS billboards: re-applies the stretches the billboards ask for (savefile load, re-import)
      _ = App.Host?.Services.GetService(typeof(Logic.Services.FitsImageService));
    }
  }

  // set once the user answered the unsaved-changes prompt: the next Close goes through
  private bool _closeConfirmed;

  /// <summary>Unsaved changes: cancels this close, asks Save / Don't save / Cancel, then closes again.</summary>
  protected override void OnClosing(WindowClosingEventArgs e)
  {
    if (!_closeConfirmed && DataContext is Logic.ViewModels.MainWindowViewModel vm && vm.HasUnsavedChanges)
    {
      e.Cancel = true;
      _ = ConfirmAndCloseAsync(vm);
    }
    base.OnClosing(e);
  }

  private async System.Threading.Tasks.Task ConfirmAndCloseAsync(Logic.ViewModels.MainWindowViewModel vm)
  {
    if (!await vm.ConfirmDiscardAsync()) return;
    _closeConfirmed = true;
    Close();
  }

  protected override void OnClosed(EventArgs e)
  {
    _billboards?.Dispose();
    _billboards = null;

    // Dispose breadcrumb overlay first — OverlaySynchronizer.Dispose() closes the window.
    _breadcrumbSynchronizer?.Dispose();
    _breadcrumbSynchronizer = null;
    _breadcrumbWindow = null;

    // Dispose removes event handlers from all attached windows (main + all overlays).
    // Overlay windows are already closed by their OverlaySynchronizers at this point.
    if (DataContext is Logic.ViewModels.MainWindowViewModel vm)
      vm.InputRouter.Dispose();
    base.OnClosed(e);
  }


  private void OnElementGotFocus(object? sender, GotFocusEventArgs e)
  {
    if (DataContext is Logic.ViewModels.MainWindowViewModel vm)
    {
      var el = e.Source as Visual;
      Views.Viewport3DView? view = null;
      while (el != null)
      {
        // the nucleus mini-view is never the active viewport, wherever it is placed
        if (el is Controls.VulkanViewportControl { DataContext: Logic.ViewModels.VulkanViewportControlViewModel { Role: Logic.Services.ViewportRole.NucleusMini } })
          break;
        if (el is Views.Viewport3DView v)
        {
          view = v;
          break;
        }
        el = el.GetVisualParent();
      }

      if (view != null && view.DataContext is Logic.ViewModels.Viewport3DViewModel vvm)
      {
        vm.ActiveViewport = vvm;
        vm.IsViewportFocused = true;
      }
      else
      {
        vm.IsViewportFocused = false;
      }
    }
  }

  private void OnKeyDown(object? sender, KeyEventArgs e)
  {
    bool isMacOs = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
      System.Runtime.InteropServices.OSPlatform.OSX
    );

    if (isMacOs)
    {
      // Cmd + Ctrl + F
      if (e.KeyModifiers.HasFlag(KeyModifiers.Meta | KeyModifiers.Control) && e.Key == Key.F)
      {
        ToggleFullscreen();
        e.Handled = true;
      }
    }
    else
    {
      // Alt + Enter
      if (e.KeyModifiers.HasFlag(KeyModifiers.Alt) && e.Key == Key.Enter)
      {
        ToggleFullscreen();
        e.Handled = true;
      }
    }

    // Global: plain Enter clears focus so any focused input commits and unfocuses.
    // Child controls (behaviors, sliders) fire first via event bubbling; if they set
    // e.Handled we skip. Otherwise we clear focus here as a catch-all.
    if (e.Key == Key.Enter && e.KeyModifiers == KeyModifiers.None && !e.Handled)
    {
      FocusManager?.ClearFocus();
      e.Handled = true;
    }
  }

  private void OnTitleDragPressed(object? sender, PointerPressedEventArgs e)
  {
    if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
      return;
    if (e.ClickCount == 2)
      WindowState = WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;
    else if (WindowState != WindowState.FullScreen)
      BeginMoveDrag(e);
    e.Handled = true;
  }

  private void ToggleFullscreen()
  {
    if (WindowState == WindowState.FullScreen)
    {
      WindowState = WindowState.Normal;
    }
    else
    {
      WindowState = WindowState.FullScreen;
    }
  }
}
