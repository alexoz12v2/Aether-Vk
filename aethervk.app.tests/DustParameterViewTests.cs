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
/// The dust-parameter layout an astronomer asked for: physical grain parameters on top, the Afρ
/// activity law and the grain size law in collapsed "Advanced" expanders with their plots and "?"
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
    Assert.Contains("Afrho0Box", inActivity);
    Assert.Contains("AfrhoPowerBox", inActivity);
    Assert.Contains("AfrhoCutoffBox", inActivity);
    Assert.Contains("AfrhoMaxBox", inActivity);
    Assert.Contains("AfrhoCurvePlot", inActivity);
    Assert.Contains("ActivityAdvancedHelp", inActivity);
    Assert.DoesNotContain("DustDiameterBox", inActivity);

    var inSize = Ids(size);
    Assert.Contains("SizeQBox", inSize);
    Assert.Contains("SizeRangeBox", inSize);
    Assert.Contains("SizeDistributionPlot", inSize);
    Assert.Contains("SizeDistributionAdvancedHelp", inSize);

    // the physical grain parameters stay outside both expanders, each with its "?" flyout
    var outside = Ids(view).Except(inActivity).Except(inSize).ToList();
    foreach (var id in new[] { "DustDiameterBox", "DustDensityBox", "DustAlbedoBox", "DustDiameterHelp", "DustDensityHelp", "DustAlbedoHelp" })
      Assert.Contains(id, outside);

    var help = All(activity).OfType<PropertyHelpButton>().Single(h => AutomationProperties.GetAutomationId(h) == "ActivityAdvancedHelp");
    Assert.Contains("not a grain distribution", help.HelpContent);

    // the plots get their models and draw without throwing once expanded
    activity.IsExpanded = true;
    size.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    var plots = All(view).OfType<CurvePlot>().ToList();
    Assert.Equal(2, plots.Count);
    Assert.All(plots, p => Assert.NotNull(p.Model));
    window.UpdateLayout();
    using var frame = window.CaptureRenderedFrame();
    Assert.NotNull(frame);
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
