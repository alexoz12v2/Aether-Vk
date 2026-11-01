#if DEBUG
using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.ViewModels;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using System.Numerics;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>Declutter of the viewport overlay (reviewer feedback): what is gone, and the legend.</summary>
public class OverlayDeclutterViewTests
{
  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static IEnumerable<string> Ids(Control root) =>
    All(root).Select(AutomationProperties.GetAutomationId).Where(i => !string.IsNullOrEmpty(i))!;

  [AvaloniaFact]
  public void Overlay_HasNoModeButtonsNorJetPreviewSize_AndHostsTheLegend()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    using var scope = host.Services.CreateScope();
    var viewportVm = scope.ServiceProvider.GetRequiredService<Viewport3DViewModel>();
    using var vm = scope.ServiceProvider.GetRequiredService<Func<Viewport3DViewModel, ViewportOverlayViewModel>>()(viewportVm);
    var window = new AetherVk.UI.OverlayWindow { DataContext = vm, Width = 800, Height = 600 };
    window.Show();
    Dispatcher.UIThread.RunJobs();

    var ids = Ids(window).ToList();
    Assert.DoesNotContain("ModeOrbitButton", ids);
    Assert.DoesNotContain("ModeNucleusButton", ids);
    Assert.DoesNotContain("ModeModelButton", ids);
    Assert.DoesNotContain(All(window).OfType<TextBlock>(), t => (t.Text ?? "").Contains("Jet Preview Size"));

    // static only: the overlay's own 100 ms timer refreshes the legend with the real (non-Nucleus)
    // state, so the panel's visibility is covered by GizmoLegendView_* on a timer-free view model
    Assert.Contains(All(window), c => AutomationProperties.GetAutomationId(c) == "GizmoLegendPanel");
    window.Close();
  }

  private static (AetherVk.UI.GizmoLegendView View, Window Window) ShowLegend(SphereGizmoLegendViewModel vm)
  {
    var view = new AetherVk.UI.GizmoLegendView { DataContext = vm };
    var window = new Window { Content = view, Width = 400, Height = 300 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    return (view, window);
  }

  [AvaloniaFact]
  public void GizmoLegendView_NucleusContext_ShowsTheGizmoRows()
  {
    using var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, AetherVk.Logic.Models.SphereGizmoPart.Default, Array.Empty<(int, Vector4)>()));
    var (view, window) = ShowLegend(vm);
    Assert.True(view.IsEffectivelyVisible);
    Assert.Contains("GizmoLegendRow_spin", Ids(window));
    Assert.DoesNotContain("GizmoLegendRow_velocity", Ids(window));
    Assert.DoesNotContain("GizmoLegendRow_orbit_earth", Ids(window));

    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, AetherVk.Logic.Models.SphereGizmoPart.All, Array.Empty<(int, Vector4)>()));
    Dispatcher.UIThread.RunJobs();
    Assert.Contains("GizmoLegendRow_velocity", Ids(window));
    window.Close();
  }

  [AvaloniaFact]
  public void GizmoLegendView_ModelContext_ShowsTheOrbitRows_AndOrbitLinesOffHidesIt()
  {
    using var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.EarthPositioning, true, AetherVk.Logic.Models.SphereGizmoPart.Default, Array.Empty<(int, Vector4)>(), true));
    var (view, window) = ShowLegend(vm);
    Assert.True(view.IsEffectivelyVisible);
    var ids = Ids(window).ToList();
    Assert.Contains("GizmoLegendRow_orbit_comet", ids);
    Assert.Contains("GizmoLegendRow_orbit_earth", ids);
    Assert.Contains("GizmoLegendRow_orbit_path", ids);
    Assert.DoesNotContain("GizmoLegendRow_spin", ids);

    vm.Refresh(new LegendContext(EarthObserverState.EarthPositioning, true, AetherVk.Logic.Models.SphereGizmoPart.Default, Array.Empty<(int, Vector4)>(), false));
    Dispatcher.UIThread.RunJobs();
    Assert.False(vm.IsShown);
    Assert.False(All(window).Single(c => AutomationProperties.GetAutomationId(c) == "GizmoLegendTitle").IsEffectivelyVisible);
    Assert.DoesNotContain("GizmoLegendRow_orbit_earth", Ids(window));
    window.Close();
  }
}
#endif
