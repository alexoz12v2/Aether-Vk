#if DEBUG
using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The 3D viewport's native host must exist once and never be re-parented: detaching it destroys
/// its native window and disposes its view model for good. Viewport-only mode only resizes and
/// hides the siblings.
/// </summary>
public class DockingViewportOnlyViewTests
{
  private static (MainWindow Window, DockingManagerView Docking, DockingManagerViewModel Vm) ShowMainWindow()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    var vm = host.Services.GetRequiredService<MainWindowViewModel>();
    var window = new MainWindow { DataContext = vm, Width = 1400, Height = 900 };
    window.Show();
    Settle(window);
    var docking = window.GetVisualDescendants().OfType<DockingManagerView>().Single();
    var dockingVm = Assert.IsType<DockingManagerViewModel>(docking.DataContext);
    return (window, docking, dockingVm);
  }

  /// <summary>The main 3D viewport's native hosts (the compact panel's nucleus view has its own
  /// host, with another id, once it was shown in this session).</summary>
  private static IEnumerable<VulkanViewportControl> MainViewportHosts(Window window) =>
    window.GetVisualDescendants().OfType<VulkanViewportControl>().Where(c => c.Name == "ViewportHost");

  private static void Settle(Window window)
  {
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    Dispatcher.UIThread.RunJobs();
  }

  [AvaloniaFact]
  public void DockingDoubleTemplate_HasOneViewportHost()
  {
    var (window, docking, _) = ShowMainWindow();
    try
    {
      Assert.Single(window.GetVisualDescendants().OfType<Viewport3DView>());
      Assert.Single(MainViewportHosts(window));

      // each split builds a grid per orientation; the hidden one must never template its children
      // (a templated hidden grid would build a second view of every child, the viewport included)
      var hiddenGrids = docking.GetVisualDescendants().OfType<SplitNodeView>()
        .SelectMany(s => s.GetVisualDescendants().OfType<Grid>()
          .Where(g => g.Name is "HorizontalGrid" or "VerticalGrid" && !g.IsVisible && g.FindAncestorOfType<SplitNodeView>() == s))
        .ToList();
      Assert.NotEmpty(hiddenGrids);
      Assert.All(hiddenGrids, g => Assert.Empty(g.GetVisualDescendants().OfType<UserControl>()));
    }
    finally
    {
      window.Close();
    }
  }

  [AvaloniaFact]
  public void ViewportOnly_Toggle10x_KeepsTheSameViewportViewAndParentChain()
  {
    var (window, docking, dock) = ShowMainWindow();
    try
    {
      Assert.True(dock.CanEnterViewportOnly, $"viewport-only refused: {dock.ViewportOnlyBlockedBy()}");
      var view = Assert.Single(window.GetVisualDescendants().OfType<Viewport3DView>());
      var host = Assert.Single(MainViewportHosts(window));
      var chain = view.GetVisualAncestors().ToList();
      var widthDocked = view.Bounds.Width;
      Assert.True(widthDocked > 0, "the viewport was not laid out");
      Assert.True(widthDocked < 0.9 * docking.Bounds.Width, $"docked viewport {widthDocked} of {docking.Bounds.Width}");

      for (var i = 0; i < 10; ++i)
      {
        Assert.True(dock.SetViewportOnly(true));
        Settle(window);
        AssertSameViewport(window, view, host, chain);
        Assert.True(dock.IsViewportOnly);
        // the tab content keeps its small margin; everything else is gone
        Assert.InRange(view.Bounds.Width, docking.Bounds.Width - 16, docking.Bounds.Width);
        Assert.InRange(view.Bounds.Height, docking.Bounds.Height - 16, docking.Bounds.Height);

        Assert.True(dock.SetViewportOnly(false));
        Settle(window);
        AssertSameViewport(window, view, host, chain);
        Assert.False(dock.IsViewportOnly);
        Assert.InRange(view.Bounds.Width, widthDocked - 1, widthDocked + 1);
      }
    }
    finally
    {
      dock.SetViewportOnly(false);
      window.Close();
    }
  }

  private static void AssertSameViewport(Window window, Viewport3DView view, VulkanViewportControl host, List<Visual> chain)
  {
    Assert.Same(view, Assert.Single(window.GetVisualDescendants().OfType<Viewport3DView>()));
    Assert.Same(host, Assert.Single(MainViewportHosts(window)));
    Assert.Same(view, host.FindAncestorOfType<Viewport3DView>());
    var now = view.GetVisualAncestors().ToList();
    Assert.Equal(chain.Count, now.Count);
    for (var i = 0; i < chain.Count; ++i)
      Assert.Same(chain[i], now[i]);
  }
}
#endif
