#if DEBUG
using System;
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
/// The Settings tab's comet appearance, billboard and FITS rows explain their fields in "?" help
/// flyouts instead of long tooltips: each ? is there with a header and a text, the converted
/// controls carry no tooltip any more, and the short tooltips of the self-explaining controls stay.
/// </summary>
public class SettingsAssetsHelpViewTests
{
  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static PropertyHelpButton Help(Control root, string id) =>
    All(root).OfType<PropertyHelpButton>().Single(c => AutomationProperties.GetAutomationId(c) == id);

  private static void AssertHelp(Control root, string id, string? contains = null)
  {
    var help = Help(root, id);
    Assert.False(string.IsNullOrWhiteSpace(help.HelpHeader), $"{id}: empty header");
    Assert.False(string.IsNullOrWhiteSpace(help.HelpContent), $"{id}: empty content");
    if (contains is not null)
      Assert.Contains(contains, help.HelpContent);
  }

  /// <summary>The control with <paramref name="id"/> (not a ? and not inside one).</summary>
  private static Control Field(Control root, string id) =>
    All(root).First(c => AutomationProperties.GetAutomationId(c) == id && c is not PropertyHelpButton
      && c.FindAncestorOfType<PropertyHelpButton>() is null && c.TemplatedParent is null);

  private static void RunLayout(Window window)
  {
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    Dispatcher.UIThread.RunJobs();
  }

  /// A Settings view with every expander open, one billboard row and its FITS part.
  private static (SettingsTabView View, Window Window, DesignSettingsTabViewModel Vm) ShowWithFitsBillboard()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 4000 };
    window.Show();
    vm.Billboards.CreateCommand.Execute(null);
    var row = vm.Billboards.Items.Single();
    row.ShowFits(Logic.Models.FitsMetadata.Parse(DesignFits.SampleJson)!, 5);
    for (var pass = 0; pass < 3; pass++)
    {
      RunLayout(window);
      foreach (var e in All(view).OfType<Expander>()) e.IsExpanded = true;
    }
    RunLayout(window);
    return (view, window, vm);
  }

  [AvaloniaFact]
  public void Appearance_HelpButtonsReplaceTheLongTooltips()
  {
    var (view, window, vm) = ShowWithFitsBillboard();
    Assert.True(vm.CometAppearance.IsCustomMode);

    AssertHelp(view, "CometDisplayModeHelp", "procedural sphere");
    AssertHelp(view, "CometMeshPickerHelp", "Imports tab");
    AssertHelp(view, "CometTextureAlbedoPickerHelp", "reflectivity");
    AssertHelp(view, "CometTextureNormalPickerHelp", "relief");
    AssertHelp(view, "CometTextureRoughnessPickerHelp", "glossy");
    AssertHelp(view, "CometTextureAoPickerHelp", "crevices");
    Assert.Equal(vm.StrDisplayMode, Help(view, "CometDisplayModeHelp").HelpHeader);
    Assert.Equal(vm.StrChannelAlbedo, Help(view, "CometTextureAlbedoPickerHelp").HelpHeader);

    foreach (var id in new[]
             {
               "CometDisplayModeDefault", "CometMeshPicker", "CometTextureAlbedoPicker", "CometTextureNormalPicker",
               "CometTextureRoughnessPicker", "CometTextureAoPicker",
             })
      Assert.Null(ToolTip.GetTip(Field(view, id)));

    // the disabled-state hint on the custom radio and the placement sub-slider names stay
    Assert.NotNull(ToolTip.GetTip(Field(view, "CometDisplayModeCustom")));
    var custom = All(view).OfType<Expander>().Single(e => Equals(e.Header, vm.StrCustomMesh));
    var placement = All(custom).OfType<UnboundedSlider>().ToList();
    Assert.Equal(6, placement.Count);
    Assert.All(placement, s => Assert.NotNull(ToolTip.GetTip(s)));
    Assert.NotNull(ToolTip.GetTip(All(custom).OfType<Button>().Single(b => Equals(b.Content, vm.StrResetPlacement))));
    window.Close();
  }

  [AvaloniaFact]
  public void Billboards_HelpButtonsReplaceTheLongTooltips()
  {
    var (view, window, vm) = ShowWithFitsBillboard();

    AssertHelp(view, "BillboardTexturePickerHelp", "Imports tab");
    AssertHelp(view, "BillboardAspectLockHelp", "proportions");
    AssertHelp(view, "BillboardAlphaHelp", "fully opaque");
    AssertHelp(view, "BillboardRotationHelp", "rotation handle");
    AssertHelp(view, "BillboardImageHeightKmHelp", "wins over the plate scale");
    AssertHelp(view, "BillboardArcsecPerPxHelp", "WCS");
    // nucleus x / y and Detect share one ?
    AssertHelp(view, "BillboardNucleusPixelHelp", "from the left");
    AssertHelp(view, "BillboardNucleusPixelHelp", "from the top");
    AssertHelp(view, "BillboardNucleusPixelHelp", "9 × 9 px");
    // the three physical-size buttons share one ?
    AssertHelp(view, "BillboardPhysicalSizeHelp", "Apply physical size:");
    AssertHelp(view, "BillboardPhysicalSizeHelp", "Center on comet:");
    AssertHelp(view, "BillboardPhysicalSizeHelp", "Match camera to image:");
    // the ? inside the billboard row reach the tab's strings through the UserControl
    Assert.Equal(vm.StrBillboardAlpha, Help(view, "BillboardAlphaHelp").HelpHeader);
    Assert.Equal(vm.StrHelpBillboardAlpha, Help(view, "BillboardAlphaHelp").HelpContent);

    var picker = view.FindControl<ComboBox>("BillboardTexturePicker")!;
    Assert.Null(ToolTip.GetTip(picker));
    foreach (var id in new[]
             {
               "BillboardAspectLock", "BillboardAlpha", "BillboardRotation", "BillboardImageHeightKm", "BillboardArcsecPerPx",
               "BillboardNucleusX", "BillboardNucleusY", "BillboardDetectNucleus", "BillboardApplyPhysicalSize",
               "BillboardCenterOnComet", "BillboardMatchCamera",
             })
      Assert.Null(ToolTip.GetTip(Field(view, id)));

    // short, self-explaining ones stay
    Assert.NotNull(ToolTip.GetTip(view.FindControl<Button>("CreateBillboardButton")!));
    foreach (var id in new[] { "BillboardX", "BillboardY", "BillboardWidth", "BillboardHeight", "BillboardVisible" })
      Assert.NotNull(ToolTip.GetTip(Field(view, id)));
    Assert.NotNull(ToolTip.GetTip(All(view).OfType<Button>().Single(b => Equals(b.Content, vm.StrBillboardDelete))));
    window.Close();
  }

  [AvaloniaFact]
  public void Fits_HelpButtonsReplaceTheLongTooltips_AndThePixelRatioHelpIsBound()
  {
    var (view, window, vm) = ShowWithFitsBillboard();
    var fits = vm.Billboards.Items.Single().Fits!;
    var section = All(view).OfType<Expander>().Single(e => AutomationProperties.GetAutomationId(e) == "BillboardFitsSection");

    AssertHelp(section, "BillboardFitsFlipLonHelp", "LONGITUD");
    // mode, curve and the two values share the Stretch ?
    AssertHelp(section, "BillboardFitsStretchHelp", "IRAF zscale");
    AssertHelp(section, "BillboardFitsStretchHelp", "asinh");
    AssertHelp(section, "BillboardFitsStretchHelp", "low and high percentile");
    AssertHelp(section, "BillboardFitsCompareHelp", "rotations ending at");
    AssertHelp(section, "BillboardFitsCompareHelp", "Orient to image");
    AssertHelp(section, "BillboardFitsCompareHelp", "WCS plate scale");
    AssertHelp(section, "BillboardFitsCompareHelp", "locks the viewport");
    AssertHelp(section, "BillboardFitsPixelRatioHelp");
    Assert.Equal(fits.PixelRatioTip, Help(section, "BillboardFitsPixelRatioHelp").HelpContent);
    Assert.Equal(vm.StrHelpHeaderStretch, Help(section, "BillboardFitsStretchHelp").HelpHeader);

    foreach (var id in new[]
             {
               "BillboardFitsStretchMode", "BillboardFitsTransfer", "BillboardFitsStretchParam0", "BillboardFitsStretchParam1",
               "BillboardFitsPixelRatio", "BillboardFitsModelRange", "BillboardFitsOrientToImage", "BillboardFitsMatchCamera",
               "BillboardSetUpComparison",
             })
      Assert.Null(ToolTip.GetTip(Field(section, id)));
    // the flip checkbox only shows with a site longitude; when it is there it has no tooltip
    if (All(section).FirstOrDefault(c => AutomationProperties.GetAutomationId(c) == "BillboardFitsFlipLon") is { } flip)
      Assert.Null(ToolTip.GetTip(flip));

    // the short action tooltips stay
    Assert.NotNull(ToolTip.GetTip(Field(section, "BillboardFitsApplyStretch")));
    var buttons = All(section).OfType<Button>().ToList();
    Assert.NotNull(ToolTip.GetTip(buttons.Single(b => Equals(b.Content, "Seek to observation"))));
    Assert.NotNull(ToolTip.GetTip(buttons.Single(b => Equals(b.Content, "Use observatory site"))));
    window.Close();
  }
}
#endif
