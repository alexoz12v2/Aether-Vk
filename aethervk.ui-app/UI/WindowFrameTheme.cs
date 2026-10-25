using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Styling;
using AetherVk.Logic.Services;

namespace AetherVk.UI;

/// <summary>
/// Keeps the OS-drawn frame of every decorated window (border, Alt+Tab caption, dialog title
/// bars) dark or light with the app theme, as Visual Studio does: on open and on every theme
/// change. Overlay windows without decorations are skipped.
/// </summary>
public static class WindowFrameTheme
{
  public static void Install(Application app, IPlatformWindowService platform)
  {
    Window.WindowOpenedEvent.AddClassHandler<Window>((w, _) => Apply(w, app, platform));
    app.ActualThemeVariantChanged += (_, _) =>
    {
      if (app.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        foreach (var w in desktop.Windows.ToList())
          Apply(w, app, platform);
    };
  }

  private static void Apply(Window w, Application app, IPlatformWindowService platform)
  {
    if (w.SystemDecorations == SystemDecorations.None)
      return;
    if (w.TryGetPlatformHandle()?.Handle is { } hwnd)
      platform.SetImmersiveDarkMode(hwnd, app.ActualThemeVariant == ThemeVariant.Dark);
  }
}
