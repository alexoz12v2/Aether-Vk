#if DEBUG
using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The reviewer's view items (the mail of 2026-10-29 and the third-round docx), checked on the
/// loaded views: what the user sees, not what the markup says.
/// </summary>
public class ReviewFeedbackViewTests
{
  private static List<Control> All(Control root) =>
    root.GetVisualDescendants().OfType<Control>()
      .Concat(root.GetLogicalDescendants().OfType<Control>()).Distinct().ToList();

  private static T ById<T>(Control root, string id) where T : Control =>
    All(root).OfType<T>().Single(c => AutomationProperties.GetAutomationId(c) == id);

  private static Window ShowModelTab(IModelTabViewModel vm, out ModelTabView view, bool expandDust)
  {
    view = new ModelTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    if (expandDust)
      ById<Expander>(view, "DustSection").IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    window.UpdateLayout();
    return window;
  }

  /// <summary>The text a TextBlock shows, whether it is set through Text or through Runs.</summary>
  private static string Shown(TextBlock t) =>
    t.Inlines is { Count: > 0 } inlines ? string.Concat(inlines.OfType<Run>().Select(r => r.Text)) : t.Text ?? "";

  /// <summary>
  /// R1: a refused jet's row says it is not in the simulation (the refusal in its tooltip) and
  /// offers a Retry bound to the tab's retry command; a jet the core took shows neither. The
  /// aperture shows two decimals (0.2° used to read "0°"). The lit warning line follows
  /// <see cref="JetViewModel.LitWarning"/>.
  /// </summary>
  [AvaloniaFact]
  public void Review_mail_R1_RefusedJetRow_ShowsTheWarningAndRetry_AndTheApertureHasTwoDecimals()
  {
    var vm = new DesignModelTabViewModel();
    var refused = vm.Jets![0];
    var taken = vm.Jets[1];
    refused.NativeError = JetNativeSync.RefusedText;
    refused.ApertureDeg = 0.2f;
    var window = ShowModelTab(vm, out var view, expandDust: true);

    var list = All(view).OfType<ListBox>().Single(l => l.Name == "JetListBox");
    var refusedRow = (Control)list.ContainerFromIndex(0)!;
    var takenRow = (Control)list.ContainerFromIndex(1)!;
    Assert.NotNull(refusedRow);
    Assert.NotNull(takenRow);

    var warning = ById<TextBlock>(refusedRow, "JetNativeErrorText");
    Assert.True(warning.IsEffectivelyVisible);
    Assert.Contains("Not in the simulation", warning.Text);
    Assert.Equal(JetNativeSync.RefusedText, ToolTip.GetTip(warning));
    Assert.StartsWith("Not in the simulation", JetNativeSync.RefusedText);

    var retry = ById<Button>(refusedRow, "JetRetryButton");
    Assert.True(retry.IsEffectivelyVisible);
    Assert.Equal("Retry", retry.Content);
    Assert.Same(vm.RetryNativeJetsCommand, retry.Command);

    Assert.False(ById<TextBlock>(takenRow, "JetNativeErrorText").IsEffectivelyVisible);
    Assert.False(ById<Button>(takenRow, "JetRetryButton").IsEffectivelyVisible);

    var aperture = All(refusedRow).OfType<TextBlock>().Select(Shown).Single(s => s.StartsWith("∠ ", StringComparison.Ordinal));
    Assert.Equal("∠ 0.2°", aperture);

    // the lit warning line: shown while set, hidden when cleared
    var lit = ById<TextBlock>(refusedRow, "JetLitWarning");
    Assert.False(lit.IsEffectivelyVisible);
    refused.LitWarning = "The site is never lit over the committed range";
    Dispatcher.UIThread.RunJobs();
    Assert.True(lit.IsEffectivelyVisible);
    Assert.Equal("⚠ The site is never lit over the committed range", lit.Text);
    refused.LitWarning = null;
    Dispatcher.UIThread.RunJobs();
    Assert.False(lit.IsEffectivelyVisible);
    window.Close();
  }

  /// <summary>
  /// docx 1 and 2: the General section (unbound placeholder) and the Orbital Elements section
  /// (never filled) of the Settings tab are hidden.
  /// </summary>
  [AvaloniaFact]
  public void Review_docx_1_2_GeneralAndOrbitalElementsSections_AreHidden()
  {
    var vm = new DesignSettingsTabViewModel();
    var view = new SettingsTabView { DataContext = vm };
    var window = new Window { Content = view, Width = 520, Height = 2400 };
    window.Show();
    Dispatcher.UIThread.RunJobs();

    foreach (var id in new[] { "GeneralSection", "OrbitalElementsSection" })
      Assert.False(ById<Expander>(view, id).IsEffectivelyVisible, id);
    // a section that is meant to show does (the check is not vacuous)
    Assert.True(ById<Expander>(view, "BillboardsSection").IsEffectivelyVisible);
    window.Close();
  }

  /// <summary>
  /// docx 9: the Sun and the Nucleus-and-Spin-Axis sections of the Model tab are hidden behind the
  /// legacy flag, which is off in the design view model and in the application's real one.
  /// </summary>
  [AvaloniaFact]
  public void Review_docx_9_SunAndNucleusSections_AreHiddenBehindTheLegacyFlag()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    using var scope = host.Services.CreateScope();
    var real = scope.ServiceProvider.GetRequiredService<ModelTabViewModel>();
    foreach (IModelTabViewModel vm in new IModelTabViewModel[] { new DesignModelTabViewModel(), real })
    {
      Assert.False(vm.EnableLegacyExpanders);
      var window = ShowModelTab(vm, out var view, expandDust: false);
      foreach (var id in new[] { "ModelSunSection", "NucleusSpinSection" })
        Assert.False(ById<Expander>(view, id).IsEffectivelyVisible, $"{id} ({vm.GetType().Name})");
      Assert.True(ById<Expander>(view, "DustSection").IsEffectivelyVisible);
      window.Close();
    }
  }

  /// <summary>
  /// docx 10: the reference position error, the dust visibility and the flow time-lapse controls
  /// each carry a "?" saying what they do (it replaced their tooltips).
  /// </summary>
  [AvaloniaFact]
  public void Review_docx_10_DustSettings_HaveHelpThatSaysWhatTheyDo()
  {
    var window = ShowModelTab(new DesignModelTabViewModel(), out var view, expandDust: true);

    string Help(string id)
    {
      var control = ById<Control>(view, id);
      Assert.True(control.IsEffectivelyVisible, id);
      Assert.Null(ToolTip.GetTip(control));
      var help = ById<AetherVk.Controls.PropertyHelpButton>(view, id + "Help");
      Assert.True(help.IsEffectivelyVisible, id + "Help");
      Assert.False(string.IsNullOrWhiteSpace(help.HelpHeader), $"{id}Help has no header");
      Assert.False(string.IsNullOrWhiteSpace(help.HelpContent), $"{id}Help has no content");
      return help.HelpContent;
    }

    var reference = Help("ShowReferenceErrorToggle");
    Assert.Contains("reference orbit", reference);
    Assert.Contains("10 nucleus radii", reference);

    var visibility = Help("DustVisibilitySlider");
    Assert.Contains("asinh", visibility);
    Assert.Contains("Visual only", visibility);

    var flow = Help("DustFlowSpeedSlider");
    Assert.Contains("K times faster than its dust", flow);
    Assert.Contains("Visual only", flow);
    window.Close();
  }
}
#endif
