using System;
using Avalonia.Controls;
using Avalonia.Input;

namespace AetherVk;

public partial class MainWindow : Window
{
  public MainWindow()
  {
    InitializeComponent();
    KeyDown += OnKeyDown;
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
