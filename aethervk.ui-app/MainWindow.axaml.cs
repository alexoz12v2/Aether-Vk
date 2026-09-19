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

  public MainWindow()
  {
    InitializeComponent();
#if DEBUG
    // Opens devtools window for this window
    this.AttachDevTools();
#endif

    KeyDown += OnKeyDown;
    AddHandler(
      GotFocusEvent,
      OnElementGotFocus,
      Avalonia.Interactivity.RoutingStrategies.Bubble
    );
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
    }
  }

  protected override void OnClosed(EventArgs e)
  {
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
