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

  /// <summary>
  /// R5: the ruler's label (the overlay's scale bar) is a TextBlock with an AutomationId, so the
  /// end-to-end suite can read its unit; it shows the view model's text while the bar is shown.
  /// </summary>
  [AvaloniaFact]
  public void Review_mail_R5_TheRulerLabel_IsReadableByItsAutomationId()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    using var scope = host.Services.CreateScope();
    var viewportVm = scope.ServiceProvider.GetRequiredService<Viewport3DViewModel>();
    using var vm = scope.ServiceProvider.GetRequiredService<Func<Viewport3DViewModel, ViewportOverlayViewModel>>()(viewportVm);
    var window = new AetherVk.UI.OverlayWindow { DataContext = vm, Width = 800, Height = 600 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    vm.MeasurementIndicatorWidth = 80;
    vm.MeasurementIndicatorText = "20 arcsec ≈ 38,300 km at the comet";
    vm.ShowMeasurementIndicator = true;
    Dispatcher.UIThread.RunJobs();

    var ruler = ById<TextBlock>(window, "ViewportRulerText");
    Assert.Equal("20 arcsec ≈ 38,300 km at the comet", ruler.Text);
    Assert.True(ruler.IsEffectivelyVisible);

    vm.ShowMeasurementIndicator = false;
    Dispatcher.UIThread.RunJobs();
    Assert.False(ruler.IsEffectivelyVisible);
    window.Close();
  }

  /// <summary>The dust-history object the diagnostic panel binds to (a reflection binding).</summary>
  public sealed class DustHistorySource : System.ComponentModel.INotifyPropertyChanged
  {
    private string _dustHistory = string.Empty;

    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

    public string DustHistory
    {
      get => _dustHistory;
      set
      {
        _dustHistory = value;
        PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(nameof(DustHistory)));
      }
    }
  }

  /// <summary>
  /// The DUST HISTORY line of the diagnostic panel carries an AutomationId (the end-to-end suite
  /// waits for "building" to go away while it plays), shows the formatted tiers, and is hidden
  /// without dust.
  /// </summary>
  [AvaloniaFact]
  public void DustHistoryLine_IsReadableByItsAutomationId()
  {
    var source = new DustHistorySource();
    var panel = new AetherVk.UI.CameraMatrixDebugPanel { DataContext = source };
    var window = new Window { Content = panel, Width = 520, Height = 800 };
    window.Show();
    Dispatcher.UIThread.RunJobs();

    var line = ById<TextBlock>(panel, "DustHistoryText");
    Assert.False(line.IsEffectivelyVisible);

    var text = AetherVk.Logic.ViewModels.Debug.CameraMatrixDebugViewModel.FormatDustHistory(new[]
    {
      new AetherVk.Logic.Services.DustTierStats(LiveClusters: 1200, Capacity: 4096, YoungestAgeS: 0, OldestAgeS: 86400,
        BandMinS: 0, BandMaxS: 172800, CaughtUp: false),
    });
    Assert.EndsWith(" building", text);
    source.DustHistory = text;
    Dispatcher.UIThread.RunJobs();
    Assert.True(line.IsEffectivelyVisible);
    Assert.Equal(text, line.Text);
    window.Close();
  }

  /// <summary>The viewport-rectangle object the diagnostic panel binds to (a reflection binding).</summary>
  public sealed class ViewportRectSource : System.ComponentModel.INotifyPropertyChanged
  {
    private string _rect = string.Empty;

    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;

    public bool ShowDiagnostics => true;

    public string StrViewportRect => "Viewport on the desktop (x, y, w, h px)";

    public string ViewportDesktopRect
    {
      get => _rect;
      set
      {
        _rect = value;
        PropertyChanged?.Invoke(this, new System.ComponentModel.PropertyChangedEventArgs(nameof(ViewportDesktopRect)));
      }
    }
  }

  /// <summary>
  /// The viewport's desktop rectangle is a diagnostic: the camera HUD's diagnostic section shows it
  /// under its label (hidden until the viewport reports its place), and Settings › Billboards no
  /// longer does.
  /// </summary>
  [AvaloniaFact]
  public void ViewportDesktopRect_IsInTheDiagnostics_NotInSettingsBillboards()
  {
    var source = new ViewportRectSource();
    var panel = new AetherVk.UI.CameraMatrixDebugPanel { DataContext = source };
    var window = new Window { Content = panel, Width = 520, Height = 800 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    var value = ById<TextBlock>(panel, "DiagnosticsViewportDesktopRect");
    Assert.False(value.IsEffectivelyVisible);
    source.ViewportDesktopRect = "-1386.000,130.000,1386.000,950.500";
    Dispatcher.UIThread.RunJobs();
    Assert.True(value.IsEffectivelyVisible);
    Assert.Equal("-1386.000,130.000,1386.000,950.500", value.Text);
    Assert.Equal(source.StrViewportRect, ById<TextBlock>(panel, "DiagnosticsViewportDesktopRectLabel").Text);
    window.Close();

    var settings = new SettingsTabView { DataContext = new DesignSettingsTabViewModel() };
    var settingsWindow = new Window { Content = settings, Width = 520, Height = 1600 };
    settingsWindow.Show();
    foreach (var expander in All(settings).OfType<Expander>())
      expander.IsExpanded = true;
    Dispatcher.UIThread.RunJobs();
    Assert.DoesNotContain(All(settings), c => AutomationProperties.GetAutomationId(c) == "ViewportDesktopRect");
    settingsWindow.Close();
  }

  /// <summary>
  /// The end-to-end test reads the rectangle from the main window, whichever tab is shown and with
  /// the diagnostic panels off: the mode bar's transparent <c>ViewportDesktopRect</c> text, which
  /// must have an automation peer (a control with IsVisible = false has none) whose name is the text.
  /// </summary>
  [AvaloniaFact]
  public void ViewportDesktopRect_IsInTheMainWindowsAutomationTree_ButNotSeen()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    var vm = host.Services.GetRequiredService<MainWindowViewModel>();
    var bar = vm.ModeBar ?? throw new InvalidOperationException("no mode bar");
    var window = new MainWindow { DataContext = vm };
    window.Show();
    bar.ViewportDesktopRect = "-1386.000,130.000,1386.000,950.500";
    Dispatcher.UIThread.RunJobs();

    var text = ById<TextBlock>(window, "ViewportDesktopRect");
    Assert.Equal("-1386.000,130.000,1386.000,950.500", text.Text);
    Assert.Equal(0.0, text.Opacity);
    Assert.False(text.IsHitTestVisible);

    static IEnumerable<Avalonia.Automation.Peers.AutomationPeer> Peers(Avalonia.Automation.Peers.AutomationPeer p) =>
      new[] { p }.Concat(p.GetChildren().SelectMany(Peers));
    var root = Avalonia.Automation.Peers.ControlAutomationPeer.CreatePeerForElement(window);
    var peer = Assert.Single(Peers(root), p => p.GetAutomationId() == "ViewportDesktopRect");
    Assert.Equal("-1386.000,130.000,1386.000,950.500", peer.GetName());
    window.Close();
  }
}
#endif
