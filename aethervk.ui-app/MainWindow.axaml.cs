using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.VisualTree;

namespace AetherVk;

public partial class MainWindow : Window
{
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
