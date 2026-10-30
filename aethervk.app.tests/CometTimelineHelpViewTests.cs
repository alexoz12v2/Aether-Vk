#if DEBUG
using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.ViewModels.Debug;
using AetherVk.UI;
using AetherVk.Views;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.LogicalTree;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The long tooltips of the Comet and Timeline tabs, the mode bar and the small views moved into
/// "?" help flyouts: each ? is there with a header and a text, the converted
/// controls carry no tooltip any more, and the icon-only buttons keep a short one (their name).
/// </summary>
public class CometTimelineHelpViewTests
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

  private static void AssertNoTip(Control root, string id) =>
    Assert.Null(ToolTip.GetTip(All(root).First(c => AutomationProperties.GetAutomationId(c) == id && c is not PropertyHelpButton
      && c.FindAncestorOfType<PropertyHelpButton>() is null)));

  private static void AssertShortTip(Control root, string id, string expected)
  {
    var tip = ToolTip.GetTip(ById<Button>(root, id)) as string;
    Assert.Equal(expected, tip);
    Assert.True(tip!.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length <= 4, $"{id}: tip too long");
  }

  private static Window Show(Control view)
  {
    var window = new Window { Content = view, Width = 1400, Height = 2400 };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    return window;
  }

  [AvaloniaFact]
  public void Comet_HelpButtonsReplaceTheLongTooltips()
  {
    var view = new CometTabView { DataContext = new DesignCometTabViewModel() };
    var window = Show(view);

    AssertHelp(view, "SpkRecordsListHelp", "Download & Commit");
    AssertHelp(view, "EpochRangeHelp", "at least 1 day");
    AssertHelp(view, "ProposedRangeTextHelp", "running simulation");
    AssertHelp(view, "RecommitWithRangeButtonHelp", "dust restarts");
    AssertHelp(view, "ModelRangeHelp", "image epoch");
    AssertHelp(view, "DustLeadDaysBoxHelp", "one dust lifetime");
    AssertHelp(view, "ReferenceOrbitPickerHelp", "Required before the first commit");
    AssertHelp(view, "NucleusRadiusSliderHelp", "Horizons");
    AssertHelp(view, "BodyFixedOrientationPickerHelp", "stars");
    // merged tooltip + ? pairs: one ? carries both texts
    AssertHelp(view, "PoleRaHelp", "Right ascension");
    AssertHelp(view, "PoleRaHelp", "axis stays fixed");
    AssertHelp(view, "PoleDecHelp", "Declination");
    AssertHelp(view, "PrimeMeridianHelp", "jet spirals");
    AssertHelp(view, "RotPeriodHelp", "Model range");
    AssertHelp(view, "RotPeriodHelp", "718");
    AssertHelp(view, "RatesAdvancedHelp", "Julian century");
    AssertHelp(view, "RatesAdvancedHelp", "prograde");

    foreach (var id in new[]
             {
               "SpkRecordsList", "StartEpochBox", "EndEpochBox", "ProposedRangeText", "RecommitWithRangeButton",
               "ModelRotationsBox", "ModelEndEpochBox", "ApplyModelRangeButton", "DustLeadDaysBox",
               "ReferenceOrbitPicker", "NucleusRadiusSlider", "PoleRaSlider", "PoleDecSlider", "PrimeMeridianSlider",
               "RotPeriodSlider", "RotRateSlider", "BodyFixedOrientationPicker",
             })
      AssertNoTip(view, id);
    // the precession sliders have no AutomationId: none of the rotational-model sliders keeps a tooltip
    var section = ById<Expander>(view, "RotationalModelSection");
    Assert.All(All(section).OfType<UnboundedSlider>(), s => Assert.Null(ToolTip.GetTip(s)));

    // KEEP sites still have theirs
    foreach (var id in new[] { "GuidedStep1", "CometSearchBox", "CometSearchButton", "CometResultsList", "RestoreButton", "DownloadCommitButton" })
      Assert.NotNull(ToolTip.GetTip(All(view).First(c => AutomationProperties.GetAutomationId(c) == id)));
    window.Close();
  }

  [AvaloniaFact]
  public void Timeline_HelpButtons_AndShortTipsOnThePlaybackButtons()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    using var scope = host.Services.CreateScope();
    var vm = scope.ServiceProvider.GetRequiredService<TimelineTabViewModel>();
    var view = new TimelineTabView { DataContext = vm };
    var window = Show(view);

    AssertHelp(view, "TimelineEpochRangeHelp", "Comet tab");
    AssertHelp(view, "CommittedRangeTextHelp", "Run to End");
    AssertHelp(view, "SpeedPickerHelp", "per real second");
    AssertHelp(view, "PlaybackHelp", "Reset:");
    AssertHelp(view, "PlaybackHelp", "Pause keeps the current epoch");
    AssertHelp(view, "PlaybackHelp", "without animating");
    AssertHelp(view, "CurrentEpochBoxHelp", "Enter");

    foreach (var id in new[] { "TimelineStartEpochBox", "TimelineEndEpochBox", "CommittedRangeText", "SpeedPicker", "CurrentEpochBox" })
      AssertNoTip(view, id);
    AssertShortTip(view, "ResetButton", vm.StrTipReset);
    AssertShortTip(view, "PlayPauseButton", vm.StrTipPlayPause);
    AssertShortTip(view, "RunToEndButton", vm.StrTipRunToEnd);
    AssertShortTip(view, "SeekToEpochButton", vm.StrTipGo);
    // the time-scale tips on the labels stay
    Assert.NotNull(ToolTip.GetTip(ById<TextBlock>(view, "TimelineStartEpochLabel")));
    window.Close();
  }

  [AvaloniaFact]
  public void ModeBar_ModelButton_HasAShortTip_ExplainedByTheModelRangeHelp()
  {
    var host = AetherVk.App.Host ?? throw new InvalidOperationException("DI host not initialised");
    var vm = host.Services.GetRequiredService<MainWindowViewModel>();
    if (vm.ModeBar is not { } bar)
      return; // no mode bar in this configuration
    var window = new MainWindow { DataContext = vm };
    window.Show();
    Dispatcher.UIThread.RunJobs();
    AssertShortTip(window, "ModeBarModelButton", bar.StrModelShortTip);
    window.Close();

    var comet = new DesignCometTabViewModel();
    Assert.Contains("rotations earlier", comet.StrHelpModelRange);
  }

  [AvaloniaFact]
  public void SmallViews_HelpButtonsReplaceTheTooltips()
  {
    var ephemeris = new CameraMatrixDebugPanel { DataContext = new DesignCameraMatrixDebugViewModel() };
    var window = Show(ephemeris);
    AssertHelp(ephemeris, "EphemerisTableHelp", "Horizons");
    Assert.Null(ToolTip.GetTip(ById<Grid>(ephemeris, "EphemerisTable")));
    window.Close();

    var editor = new RotationalModelEditor();
    window = Show(editor);
    AssertHelp(editor, "RotationalModelEditorHelp", "J2000.0");
    Assert.DoesNotContain(All(editor).OfType<TextBlock>(), t => ToolTip.GetTip(t) is not null);
    window.Close();
  }

  /// <summary>The two reference windows need live view models; their markup is checked instead.</summary>
  [Theory]
  [InlineData("SpawnMeshDialogWindow.axaml", "KinematicModeHelp")]
  [InlineData("HorizonJplView.axaml", "HorizonJplHelp")]
  public void ReferenceViews_UseThePropertyHelpButton(string file, string id)
  {
    string? path = null;
    for (var d = new System.IO.DirectoryInfo(AppContext.BaseDirectory); d is not null && path is null; d = d.Parent)
    {
      var p = System.IO.Path.Combine(d.FullName, "aethervk.ui-app", "Views", "References", file);
      if (System.IO.File.Exists(p)) path = p;
    }
    Assert.NotNull(path);
    var xaml = System.IO.File.ReadAllText(path!);
    Assert.Contains($"AutomationProperties.AutomationId=\"{id}\"", xaml);
    Assert.Contains("controls:PropertyHelpButton", xaml);
    Assert.DoesNotContain("ToolTip.Tip", xaml);
  }
}
#endif
