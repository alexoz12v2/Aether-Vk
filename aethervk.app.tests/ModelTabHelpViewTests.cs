#if DEBUG
using System.Collections.Generic;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.Services;
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
/// The Model tab's explanations live in "?" help buttons (resx texts), not in tooltips.
/// Checked on the loaded view with every expander open and a jet selected.
/// </summary>
public class ModelTabHelpViewTests
{
  /// <summary>Every "?" of the Model tab, by AutomationId (the legacy Sun ? is hidden: see below).</summary>
  private static readonly string[] HelpIds =
  {
    // dust view aids
    "ShowReferenceErrorToggleHelp", "DustVisibilitySliderHelp", "DustFlowHelp", "DustRhoNormToggleHelp",
    "DustTrajectoriesToggleHelp", "DustExposureSliderHelp", "DustTracersToggleHelp", "DustFlowSpeedSliderHelp",
    "SphereGizmoVisibleCheckBoxHelp", "ShowJetAnglesCheckBoxHelp",
    // selected jet
    "JetLatitudeSliderHelp", "JetLongitudeSliderHelp", "JetApertureHelp", "JetVelocityHelp", "JetVelocityStdHelp",
    "JetStreamColorHelp", "JetBetaHelp", "JetQDustHelp",
    "JetGrainHelp", "JetDiametreUmHelp", "JetDensityHelp", "JetQprHelp", "JetAlbedoHelp", "JetSizeQHelp", "JetSizeRangeHelp",
    "JetActivityHelp", "JetDustRate0Help", "CopyGrainToAllJetsButtonHelp",
    // defaults for new jets
    "JetDefaultsHelp", "DustDensityHelp", "DustDiameterHelp", "DustQprHelp", "DustAlbedoHelp",
    "ActivityAdvancedHelp", "DustRate0BoxHelp", "DustRatePowerBoxHelp", "DustRateCutoffBoxHelp", "DustRateMaxBoxHelp",
    "SizeDistributionAdvancedHelp", "SizeRangeBoxHelp",
  };

  /// <summary>Controls whose tooltip moved into a "?".</summary>
  private static readonly string[] ConvertedIds =
  {
    "ShowReferenceErrorToggle", "DustVisibilitySlider", "DustRhoNormToggle", "DustTrajectoriesToggle",
    "DustExposureSlider", "DustTracersToggle", "DustFlowSpeedSlider",
    "SphereGizmoVisibleCheckBox", "GizmoGridCheckBox", "GizmoBodyAxesCheckBox", "GizmoSpinAxisCheckBox",
    "GizmoSunCheckBox", "GizmoVelocityCheckBox", "ShowJetAnglesCheckBox",
    "JetLatitudeSlider", "JetLongitudeSlider", "JetStreamColorPicker", "JetBetaBox", "JetQDustBox",
    "JetDiametreUm", "JetDensity", "JetQpr", "JetAlbedo", "JetSizeQ", "JetSizeRange", "JetDustRate0",
    "CopyGrainToAllJetsButton", "DustQprBox", "TotalActivityText",
    "DustRate0Box", "DustRatePowerBox", "DustRateCutoffBox", "DustRateMaxBox", "SizeRangeBox",
  };

  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static T ById<T>(Control root, string id) where T : Control =>
    All(root).OfType<T>().Single(c => AutomationProperties.GetAutomationId(c) == id);

  private static Window ShowExpanded(DesignModelTabViewModel vm, out ModelTabView view)
  {
    view = new ModelTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 3200 };
    window.Show();
    ById<Expander>(view, "DustSection").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    foreach (var id in new[] { "JetGrainExpander", "JetActivityExpander", "ActivityAdvancedExpander", "SizeDistributionAdvancedExpander" })
      ById<Expander>(view, id).IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    return window;
  }

  [AvaloniaFact]
  public void Model_EveryHelpButton_IsShownWithAHeaderAndAnExplanation()
  {
    var vm = new DesignModelTabViewModel();
    Assert.NotNull(vm.SelectedJet);
    var window = ShowExpanded(vm, out var view);

    foreach (var id in HelpIds)
    {
      var help = ById<PropertyHelpButton>(view, id);
      Assert.True(help.IsEffectivelyVisible, id);
      Assert.False(string.IsNullOrWhiteSpace(help.HelpHeader), $"{id} has no header");
      Assert.False(string.IsNullOrWhiteSpace(help.HelpContent), $"{id} has no content");
      Assert.NotEqual(help.HelpHeader, help.HelpContent);
    }

    // the jet panel's ?s read the tab's strings although their DataContext is the jet
    Assert.Equal(vm.StrHelpJetDiameter, ById<PropertyHelpButton>(view, "JetDiametreUmHelp").HelpContent);
    Assert.Equal(vm.StrLatitude, ById<PropertyHelpButton>(view, "JetLatitudeSliderHelp").HelpHeader);

    // the gizmo group has one ? that names every part
    var gizmo = ById<PropertyHelpButton>(view, "SphereGizmoVisibleCheckBoxHelp").HelpContent;
    foreach (var part in new[] { "Grid", "Body X/Y", "Spin axis", "Sun", "Velocity" })
      Assert.Contains(part, gizmo);

    // the Σ read-out's explanation is in the activity ?
    Assert.Contains("Σ", ById<PropertyHelpButton>(view, "ActivityAdvancedHelp").HelpContent);

    // the legacy Sun section (hidden behind the flag) explains itself instead of repeating its title
    var sun = ById<PropertyHelpButton>(view, "ModelSunSectionHelp");
    Assert.Equal(vm.StrSun, sun.HelpHeader);
    Assert.Equal(vm.StrHelpSun, sun.HelpContent);
    Assert.Contains("Horizons", sun.HelpContent);
    window.Close();
  }

  [AvaloniaFact]
  public void Model_ConvertedControls_HaveNoTooltip_OnlyTheKeptOnesDo()
  {
    var vm = new DesignModelTabViewModel();
    vm.Jets![0].NativeError = JetNativeSync.RefusedText;
    var window = ShowExpanded(vm, out var view);

    foreach (var id in ConvertedIds)
      Assert.Null(ToolTip.GetTip(ById<Control>(view, id)));
    var flow = All(view).OfType<CheckBox>().Single(c => Equals(c.Content, vm.StrDustFlow));
    Assert.Null(ToolTip.GetTip(flow));

    // the kept tooltips are still there
    Assert.NotNull(ToolTip.GetTip(ById<Button>(view, "AddJetButton")));
    var list = All(view).OfType<ListBox>().Single(l => l.Name == "JetListBox");
    var row = (Control)list.ContainerFromIndex(0)!;
    Assert.Equal(JetNativeSync.RefusedText, ToolTip.GetTip(ById<TextBlock>(row, "JetNativeErrorText")));
    Assert.NotNull(ToolTip.GetTip(ById<Button>(row, "JetRetryButton")));

    // no other control of the tab's markup carries a tooltip (logical tree: control templates are
    // not ours; a ? shows its header on hover: not counted)
    var inHelp = All(view).OfType<PropertyHelpButton>().SelectMany(h => All(h).Append(h)).ToHashSet();
    var kept = new HashSet<string?> { "AddJetButton", "JetNativeErrorText", "JetRetryButton" };
    var tipped = view.GetLogicalDescendants().OfType<Control>()
      .Where(c => !inHelp.Contains(c) && ToolTip.GetTip(c) is not null)
      .Where(c => !kept.Contains(AutomationProperties.GetAutomationId(c)))
      .ToList();
    // the per-row remove buttons ("−") keep theirs
    Assert.All(tipped, c => Assert.Equal("−", (c as Button)?.Content));
    window.Close();
  }
}
#endif
