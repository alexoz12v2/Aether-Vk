#if DEBUG
using System.Collections.Generic;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The long tooltips of the Settings tab's particle rendering section and of each viewport's
/// projection, lighting and Earth observer settings moved into "?" help flyouts: each ? is there
/// with a header and a text, the converted controls carry no tooltip any more, and the short
/// action tooltips (Restore Proportions, Restore preset projection, the snap buttons) stay.
/// </summary>
public class SettingsRenderingHelpViewTests
{
  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static T ById<T>(Control root, string id) where T : Control =>
    All(root).OfType<T>().Single(c => AutomationProperties.GetAutomationId(c) == id);

  private static void AssertHelp(Control root, string id, string? contains = null)
  {
    var help = ById<PropertyHelpButton>(root, id);
    Assert.False(string.IsNullOrWhiteSpace(help.HelpHeader), $"{id}: empty header");
    Assert.False(string.IsNullOrWhiteSpace(help.HelpContent), $"{id}: empty content");
    if (contains is not null)
      Assert.Contains(contains, help.HelpContent);
  }

  /// <summary>Controls outside any ? (the ? itself shows its header as a tooltip).</summary>
  private static IEnumerable<Control> Fields(Control root) =>
    All(root).Where(c => c is not PropertyHelpButton && c.FindAncestorOfType<PropertyHelpButton>() is null);

  private static (SettingsTabView View, Window Window) Show()
  {
    var view = new SettingsTabView { DataContext = new DesignSettingsTabViewModel() };
    var window = new Window { Content = view, Width = 1400, Height = 3000 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    foreach (var e in All(view).OfType<Expander>()
               .Where(e => e.Name == "ParticleRenderingSection"
                           || AutomationProperties.GetAutomationId(e) is "ViewportSettingsSection" or "ProjectionAdvancedExpander"))
      e.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    Dispatcher.UIThread.RunJobs();
    return (view, window);
  }

  [AvaloniaFact]
  public void ParticleRendering_HelpButtonsReplaceTheTooltips()
  {
    var (view, window) = Show();
    var section = All(view).OfType<Expander>().Single(e => e.Name == "ParticleRenderingSection");

    AssertHelp(section, "ParticleQualitySliderHelp", "Custom");
    AssertHelp(section, "VolumeDetailPickerHelp", "coarsely");
    AssertHelp(section, "PropagateDetailPickerHelp", "interpolated");
    AssertHelp(section, "HazeDetailPickerHelp", "coarsest level");
    AssertHelp(section, "ReuseTolerancePickerHelp", "smear");
    Assert.Equal(5, All(section).OfType<PropertyHelpButton>().Count());

    Assert.All(Fields(section), c => Assert.Null(ToolTip.GetTip(c)));
    window.Close();
  }

  [AvaloniaFact]
  public void Viewport_HelpButtonsReplaceTheTooltips_AndTheActionTipsStay()
  {
    var (view, window) = Show();
    var sections = All(view).OfType<Expander>()
      .Where(e => AutomationProperties.GetAutomationId(e) == "ViewportSettingsSection").ToList();
    Assert.NotEmpty(sections);

    foreach (var section in sections)
    {
      AssertHelp(section, "DiagnosticPanelsCheckBoxHelp", "hidden by default");
      AssertHelp(section, "ComparisonLockCheckBoxHelp", "telescope image");
      // one ? for the Perspective / Orthographic radio group
      AssertHelp(section, "ProjectionTypeHelp", "Perspective");
      AssertHelp(section, "ProjectionTypeHelp", "Orthographic");
      AssertHelp(section, "ProjectionAdvancedExpanderHelp", "Match camera to image");
      AssertHelp(section, "FovSliderHelp", "arcminutes");
      AssertHelp(section, "FovUnitPickerHelp", "1/3600");
      AssertHelp(section, "PerspNearHelp", "closer");
      AssertHelp(section, "PerspFarHelp", "farther");
      AssertHelp(section, "DistanceUnitsHelp", "AU");
      AssertHelp(section, "OrthoHalfWidthSliderHelp", "width");
      AssertHelp(section, "OrthoHalfHeightSliderHelp", "height");
      AssertHelp(section, "OrthoUnitPickerHelp", "AU");
      AssertHelp(section, "OrthoNearHelp", "Nearest");
      AssertHelp(section, "OrthoFarHelp", "Farthest");
      AssertHelp(section, "OrthoProportionsLockHelp", "stretched");
      AssertHelp(section, "ShowBehindCameraCheckBoxHelp", "behind the camera");
      AssertHelp(section, "PhysicalLightingToggleHelp", "indigo");
      AssertHelp(section, "ViewNucleusFromEarthButtonHelp", "Earth observer");
      AssertHelp(section, "NucleusExposureSliderHelp", "stops");
      AssertHelp(section, "ObserverTopocentricCheckBoxHelp", "diurnal parallax");
      AssertHelp(section, "ObserverNorthUpCheckBoxHelp", "J2000");
      AssertHelp(section, "ObserverLatSliderHelp", "north");
      AssertHelp(section, "ObserverLonSliderHelp", "Greenwich");
      // one ? for the Free / Comet / Sun look direction radio group
      AssertHelp(section, "LookDirectionHelp", "Free look");
      AssertHelp(section, "LookDirectionHelp", "Comet tracking");
      AssertHelp(section, "LookDirectionHelp", "Sun tracking");
      AssertHelp(section, "EarthObserverScaleTextHelp", "kilometres per pixel");

      var fields = Fields(section).ToList();
      foreach (var id in new[]
               {
                 "DiagnosticPanelsCheckBox", "ComparisonLockCheckBox", "ProjectionPerspective", "ProjectionOrthographic",
                 "ProjectionAdvancedExpander", "FovSlider", "FovUnitPicker", "OrthoHalfWidthSlider", "OrthoHalfHeightSlider",
                 "OrthoUnitPicker", "OrthoProportionsLock", "ShowBehindCameraCheckBox", "PhysicalLightingToggle",
                 "ViewNucleusFromEarthButton", "NucleusExposureSlider", "ObserverTopocentricCheckBox",
                 "ObserverNorthUpCheckBox", "ObserverLatSlider", "ObserverLonSlider", "OrientFree", "OrientCometTracking",
                 "OrientSunTracking", "EarthObserverScaleText",
               })
        Assert.Null(ToolTip.GetTip(fields.First(c => AutomationProperties.GetAutomationId(c) == id)));
      // the near / far sliders and the distance unit picker have no AutomationId: no slider, picker,
      // check box or radio of the viewport keeps a tooltip
      Assert.All(fields.Where(c => c is UnboundedSlider or Slider or ComboBox or CheckBox or RadioButton),
        c => Assert.Null(ToolTip.GetTip(c)));

      // KEEP: the short action tooltips stay
      Assert.NotNull(ToolTip.GetTip(ById<Button>(section, "RestoreOrthoProportionsButton")));
      Assert.NotNull(ToolTip.GetTip(ById<Button>(section, "RestorePresetButton")));
      foreach (var snap in new[] { "Sun", "Comet", "Earth" })
        Assert.NotNull(ToolTip.GetTip(fields.OfType<Button>().Single(b => Equals(b.Content, snap))));
    }
    window.Close();
  }
}
#endif
