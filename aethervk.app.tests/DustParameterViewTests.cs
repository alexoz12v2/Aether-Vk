#if DEBUG
using System.Collections.Generic;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The dust-parameter layout an astronomer asked for: physical grain parameters on top, the Q_d
/// production law and the grain size law in collapsed "Advanced" expanders with their plots and "?"
/// help flyouts; in the Comet tab the period (h) is the control and the °/day rate is advanced.
/// </summary>
public class DustParameterViewTests
{
  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static List<string?> Ids(Control root) => All(root).Select(AutomationProperties.GetAutomationId).ToList();

  private static Expander ExpanderById(Control root, string id) =>
    All(root).OfType<Expander>().Single(e => AutomationProperties.GetAutomationId(e) == id);

  [AvaloniaFact]
  public void Model_ActivityAndSizeLawAreAdvanced_WithPlotsAndHelp()
  {
    var vm = new DesignModelTabViewModel();
    var view = new ModelTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    foreach (var e in All(view).OfType<Expander>().Where(e => AutomationProperties.GetAutomationId(e) == "DustSection"))
      e.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var activity = ExpanderById(view, "ActivityAdvancedExpander");
    var size = ExpanderById(view, "SizeDistributionAdvancedExpander");
    Assert.False(activity.IsExpanded);
    Assert.False(size.IsExpanded);

    var inActivity = Ids(activity);
    Assert.Contains("DustRate0Box", inActivity);
    Assert.Contains("DustRatePowerBox", inActivity);
    Assert.Contains("DustRateCutoffBox", inActivity);
    Assert.Contains("DustRateMaxBox", inActivity);
    Assert.Contains("DustRateCurvePlot", inActivity);
    Assert.Contains("ActivityAdvancedHelp", inActivity);
    Assert.DoesNotContain("DustDiameterBox", inActivity);

    var inSize = Ids(size);
    Assert.DoesNotContain("SizeQBox", Ids(view)); // q is per jet only (JetSizeQ)
    Assert.Contains("SizeRangeBox", inSize);
    Assert.Contains("SizeDistributionPlot", inSize);
    Assert.Contains("SizeDistributionAdvancedHelp", inSize);

    // the physical grain parameters stay outside both expanders, each with its "?" flyout
    var outside = Ids(view).Except(inActivity).Except(inSize).ToList();
    foreach (var id in new[] { "DustDiameterBox", "DustDensityBox", "DustAlbedoBox", "DustDiameterHelp", "DustDensityHelp", "DustAlbedoHelp" })
      Assert.Contains(id, outside);

    var help = All(activity).OfType<PropertyHelpButton>().Single(h => AutomationProperties.GetAutomationId(h) == "ActivityAdvancedHelp");
    Assert.Contains("it does not depend on the grain or the speed", help.HelpContent);

    // the plots get their models and draw without throwing once expanded (the defaults' two and
    // the selected jet's two)
    activity.IsExpanded = true;
    size.IsExpanded = true;
    ExpanderById(view, "JetGrainExpander").IsExpanded = true;
    ExpanderById(view, "JetActivityExpander").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    var plots = All(view).OfType<CurvePlot>().ToList();
    Assert.Equal(
      new[] { "DustRateCurvePlot", "JetDustRateCurvePlot", "JetSizeDistributionPlot", "SizeDistributionPlot" },
      plots.Select(AutomationProperties.GetAutomationId).OrderBy(id => id, System.StringComparer.Ordinal));
    Assert.All(plots, p => Assert.NotNull(p.Model));
    window.UpdateLayout();
    using var frame = window.CaptureRenderedFrame();
    Assert.NotNull(frame);
    window.Close();
  }

  /// <summary>
  /// The selected jet's own grain and activity: collapsed "Grain" and "Advanced: activity (Q_d)"
  /// expanders in the jet panel with their boxes, plots and help, and the "Copy to all jets"
  /// button. The Σ read-out is in the model-wide advanced expander.
  /// </summary>
  [AvaloniaFact]
  public void Model_JetPanel_HasItsOwnGrainAndActivity()
  {
    var vm = new DesignModelTabViewModel();
    var view = new ModelTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    foreach (var e in All(view).OfType<Expander>().Where(e => AutomationProperties.GetAutomationId(e) == "DustSection"))
      e.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();

    var grain = ExpanderById(view, "JetGrainExpander");
    var activity = ExpanderById(view, "JetActivityExpander");
    Assert.False(grain.IsExpanded);
    Assert.False(activity.IsExpanded);

    var inGrain = Ids(grain);
    foreach (var id in new[] { "JetDiametreUm", "JetDensity", "JetQpr", "JetAlbedo", "JetSizeQ", "JetSizeRange",
               "JetSizeDistributionPlot", "JetGrainHelp" })
      Assert.Contains(id, inGrain);
    var inActivity = Ids(activity);
    foreach (var id in new[] { "JetDustRate0", "JetDustRateCurvePlot", "DustRateNowText", "JetActivityHelp" })
      Assert.Contains(id, inActivity);

    var outside = Ids(view).Except(inGrain).Except(inActivity).ToList();
    Assert.Contains("CopyGrainToAllJetsButton", outside);
    Assert.Contains("TotalActivityText", Ids(ExpanderById(view, "ActivityAdvancedExpander")));

    // the boxes are bound to the selected jet, not to the defaults
    grain.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    var diameter = All(grain).OfType<TextBox>().Single(t => AutomationProperties.GetAutomationId(t) == "JetDiametreUm");
    Assert.Equal("250", diameter.Text);
    var help = All(grain).OfType<PropertyHelpButton>().Single(h => AutomationProperties.GetAutomationId(h) == "JetGrainHelp");
    Assert.Contains("q = 3.5", help.HelpContent);
    window.Close();
  }

  [AvaloniaFact]
  public void Comet_RotationRateIsAdvanced_PeriodAndPoleHaveHelp()
  {
    var vm = new DesignCometTabViewModel();
    var view = new CometTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    Dispatcher.UIThread.RunJobs();

    var rates = ExpanderById(view, "PolePrecessionExpander");
    Assert.False(rates.IsExpanded);
    Assert.Equal("Advanced: rates (precession, °/day)",
      All(rates).OfType<TextBlock>().Select(t => t.Text).First(t => t?.StartsWith("Advanced") == true));
    var inside = Ids(rates);
    Assert.Contains("RotRateSlider", inside);
    Assert.Contains("RatesAdvancedHelp", inside);
    Assert.DoesNotContain("RotPeriodSlider", inside);

    var all = Ids(view);
    foreach (var id in new[] { "RotPeriodSlider", "RotPeriodHelp", "PoleRaHelp", "PoleDecHelp", "PrimeMeridianHelp" })
      Assert.Contains(id, all);
    Assert.DoesNotContain("PoleRaHelp", inside);
    window.Close();
  }
}
#endif

/// <summary>The reviewer asked for Afρ to go: the Model tab's markup names only the production rate Q_d.</summary>
public class ModelTabMarkupTests
{
  private static string ModelTabAxaml()
  {
    for (var d = new System.IO.DirectoryInfo(System.AppContext.BaseDirectory); d is not null; d = d.Parent)
    {
      var p = System.IO.Path.Combine(d.FullName, "aethervk.ui-app", "Views", "ModelTabView.axaml");
      if (System.IO.File.Exists(p)) return System.IO.File.ReadAllText(p);
    }
    throw new System.IO.FileNotFoundException("aethervk.ui-app/Views/ModelTabView.axaml");
  }

  [Fact]
  public void ModelTabView_MentionsNoAfrho()
  {
    var text = ModelTabAxaml();
    Assert.DoesNotContain("Afρ", text);
    Assert.DoesNotContain("afrho", text, System.StringComparison.OrdinalIgnoreCase);
    Assert.Contains("Q_d at 1 AU (kg/s)", text);
  }
}
