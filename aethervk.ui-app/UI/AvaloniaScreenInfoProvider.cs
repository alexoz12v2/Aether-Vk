using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform;

namespace AetherVk.UI;

/// <summary>
/// <see cref="IScreenInfoProvider"/> from Avalonia's <see cref="Screens"/> of the main window.
/// Without a main window (headless tests, design time) it reports one 1920×1080 display.
/// </summary>
public sealed class AvaloniaScreenInfoProvider : IScreenInfoProvider
{
  private static readonly ScreenInfo Fallback = new(new RectI(0, 0, 1920, 1080), 1, IsPrimary: true);

  private static Window? MainWindow =>
    (Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.MainWindow;

  public IReadOnlyList<ScreenInfo> Screens
  {
    get
    {
      var screens = MainWindow?.Screens?.All;
      if (screens is null || screens.Count == 0)
        return [Fallback];
      return screens.Select(ToInfo).ToList();
    }
  }

  public ScreenInfo MainWindowScreen
  {
    get
    {
      var w = MainWindow;
      var s = w is null ? null : w.Screens?.ScreenFromWindow(w) ?? w.Screens?.Primary;
      return s is null ? Fallback : ToInfo(s);
    }
  }

  public static ScreenInfo ToInfo(Screen s) =>
    new(new RectI(s.Bounds.X, s.Bounds.Y, s.Bounds.Width, s.Bounds.Height), s.Scaling, s.IsPrimary);
}
