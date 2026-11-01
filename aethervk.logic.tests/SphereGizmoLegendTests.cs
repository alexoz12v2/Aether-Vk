using System.Numerics;
using AetherVk.Logic.Models;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>The Nucleus view's gizmo legend: rows follow the parts mask, shown with the gizmo on.</summary>
public class SphereGizmoLegendTests
{
  private static readonly (int, Vector4)[] NoJets = [];

  private static string[] Keys(SphereGizmoLegendViewModel vm) =>
    vm.Rows.Select(r => r.Key).ToArray();

  [Fact]
  public void Rows_FollowTheGizmoPartsMask()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.Default, NoJets));
    Assert.Equal(new[] { "grid", "body_x", "body_y", "spin", "sun" }, Keys(vm));
    Assert.True(vm.IsShown);

    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.All, NoJets));
    Assert.Equal("velocity", Keys(vm)[^1]);
    Assert.Equal(6, vm.Rows.Count);

    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.SpinAxis | SphereGizmoPart.Sun, NoJets));
    Assert.Equal(new[] { "spin", "sun" }, Keys(vm));

    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, 0, NoJets));
    Assert.Empty(vm.Rows);
    Assert.False(vm.IsShown);
  }

  [Fact]
  public void Colours_AreTheShaderOnes()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.All, NoJets));
    string C(string key) => vm.Rows.Single(r => r.Key == key).ColorHex;
    Assert.Equal("#FF0000", C("body_x"));
    Assert.Equal("#00FF00", C("body_y"));
    Assert.Equal("#0000FF", C("spin"));
    Assert.Equal("#FFFF00", C("sun"));
    Assert.Equal("#FF59FF", C("velocity")); // (1, 0.35, 1)
  }

  [Fact]
  public void IsShown_NucleusViewNeedsTheGizmoOn()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.UpZenith, true, SphereGizmoPart.Default, NoJets));
    Assert.DoesNotContain("grid", Keys(vm)); // orbit rows, not gizmo rows
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, false, SphereGizmoPart.Default, NoJets));
    Assert.False(vm.IsShown);
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.Default, NoJets));
    Assert.True(vm.IsShown);
  }

  [Fact]
  public void Jets_AddARowInTheirOwnColour()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.Sun, [(1, new Vector4(0.4f, 0.8f, 1f, 1f)), (2, new Vector4(1f, 0f, 0f, 1f))]));
    Assert.Equal(new[] { "sun", "jet_1", "jet_2" }, Keys(vm));
    Assert.Equal("#66CCFF", vm.Rows[1].ColorHex);
    Assert.Equal("Jet 2 direction", vm.Rows[2].Label);
  }

  [Fact]
  public void ModelAndOrbitViews_ListTheOrbitLines()
  {
    foreach (var mode in new[] { EarthObserverState.EarthPositioning, EarthObserverState.UpZenith })
    {
      var vm = new SphereGizmoLegendViewModel();
      vm.Refresh(new LegendContext(mode, true, SphereGizmoPart.All, NoJets));
      Assert.Equal(new[] { "orbit_comet", "orbit_earth", "orbit_path" }, Keys(vm));
      Assert.True(vm.IsShown);
      // the gizmo state does not matter outside the Nucleus view
      vm.Refresh(new LegendContext(mode, false, 0, NoJets));
      Assert.True(vm.IsShown);
      // orbit lines off: nothing to explain
      vm.Refresh(new LegendContext(mode, true, SphereGizmoPart.All, NoJets, OrbitLinesShown: false));
      Assert.Empty(vm.Rows);
      Assert.False(vm.IsShown);
    }
  }

  [Fact]
  public void OrbitColours_AreTheEngineOnes()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.EarthPositioning, true, 0, NoJets));
    string C(string key) => vm.Rows.Single(r => r.Key == key).ColorHex;
    Assert.Equal("#FF3333", C("orbit_comet")); // logic_thread.rs [1, 0.2, 0.2]
    Assert.Equal("#4D99FF", C("orbit_earth")); // logic_thread.rs [0.3, 0.6, 1.0]
    Assert.Equal("#FFD91A", C("orbit_path")); // trajectory.rs [1, 0.85, 0.1]
    Assert.Equal("Earth orbit", vm.Rows.Single(r => r.Key == "orbit_earth").Label);
  }

  [Fact]
  public void NucleusView_ShowsNoOrbitRows_AndOrbitLinesOffDoesNotHideTheGizmo()
  {
    var vm = new SphereGizmoLegendViewModel();
    vm.Refresh(new LegendContext(EarthObserverState.CometOrbiting, true, SphereGizmoPart.Default, NoJets, OrbitLinesShown: false));
    Assert.DoesNotContain(Keys(vm), k => k.StartsWith("orbit_"));
    Assert.True(vm.IsShown);
  }

  [Fact]
  public void OverlayNoLongerOffersTheRemovedItems()
  {
    var t = typeof(ViewportOverlayViewModel);
    Assert.Null(t.GetProperty("JetPreviewSizeString"));
    Assert.Null(t.GetProperty("SelectOrbitViewCommand"));
    Assert.Null(t.GetProperty("SelectNucleusViewCommand"));
    Assert.Null(t.GetProperty("SelectModelViewCommand"));
    Assert.NotNull(t.GetProperty("GizmoLegend"));
  }
}
