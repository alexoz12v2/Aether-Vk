using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using AetherVk.Logic.Automation;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;

namespace AetherVk.AppTests;

/// <summary>
/// Every AutomationId of the main window, checked against the manifest file next to this test
/// (<c>automation_ids.manifest.txt</c>: sorted, one id per line). The window is walked in the
/// classic layout, in the compact layout and with each classic tab selected in turn.
/// <list type="bullet">
///   <item>No id is set on two controls of the same window state.</item>
///   <item>The ids found are exactly the manifest's: a new, renamed or lost id fails here, with the
///     ids added and removed, before a UI test that looks for it does.</item>
///   <item>A manifest line <c>?Id</c> is a conditional id: its control exists only in some states of
///     the shared test app (the Comet tab's Rotational Model section opens once a comet is
///     committed, which depends on the tests that ran before). It must still be unique and known,
///     but may be absent.</item>
/// </list>
/// To accept a change, run once with <c>AETHERVK_UPDATE_AUTOMATION_MANIFEST=1</c>: the test rewrites
/// the manifest (and still reports what changed); review the file's diff.
/// </summary>
public class AutomationIdManifestTests(ITestOutputHelper output)
{
  public const string ManifestFile = "automation_ids.manifest.txt";
  public const string UpdateEnvVar = "AETHERVK_UPDATE_AUTOMATION_MANIFEST";

  private static string ManifestPath()
  {
    for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
    {
      var p = Path.Combine(d.FullName, "aethervk.app.tests", ManifestFile);
      if (File.Exists(p)) return p;
    }
    throw new FileNotFoundException($"aethervk.app.tests/{ManifestFile}");
  }

  private static IEnumerable<LayoutNodeViewModelBase> Nodes(LayoutNodeViewModelBase node) =>
    node is SplitNodeViewModel s ? new[] { node }.Concat(Nodes(s.FirstChild)).Concat(Nodes(s.SecondChild)) : new[] { node };

  [AvaloniaFact]
  public void EveryAutomationId_IsUnique_AndInTheManifest()
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var vm = services.GetRequiredService<MainWindowViewModel>();
    var layout = services.GetRequiredService<UiLayoutService>();
    var translation = services.GetRequiredService<ITranslationService>();
    var culture = translation.CurrentCulture;
    layout.Set(UiLayout.Classic);
    translation.CurrentCulture = new CultureInfo("en");
    Dispatcher.UIThread.RunJobs();
    var window = new MainWindow { DataContext = vm, Width = 1400, Height = 900 };
    window.Show();
    var mini = vm.CompactPanel.NucleusMini;
    bool miniExpanded = mini.IsExpanded;
    var selections = new List<(TabGroupNodeViewModel Group, TabItemViewModel? Tab)>();

    void Settle()
    {
      Dispatcher.UIThread.RunJobs();
      window.UpdateLayout();
      Dispatcher.UIThread.RunJobs();
    }

    var found = new SortedSet<string>(StringComparer.Ordinal);
    var duplicates = new List<string>();

    // one state of the window: its ids, each on one control only
    void Collect(string state)
    {
      Settle();
      var ids = CompactPanelViewTests.AutomationIds(window);
      foreach (var g in ids.GroupBy(id => id, StringComparer.Ordinal).Where(g => g.Count() > 1))
        duplicates.Add($"{g.Key} ×{g.Count()} ({state})");
      found.UnionWith(ids);
    }

    try
    {
      Settle();
      Collect("classic");

      // each classic tab in turn. The viewport's group is left alone: selecting another tab
      // there would tear the native viewport down.
      var groups = Nodes(vm.DockingManager.RootNode).OfType<TabGroupNodeViewModel>()
        .Where(g => !g.Tabs.Any(t => t.IsPinned))
        .ToList();
      foreach (var group in groups)
      {
        selections.Add((group, group.SelectedTab));
        foreach (var tab in group.Tabs.ToList())
        {
          group.SelectedTab = tab;
          Collect($"classic, tab {tab.AutomationKey}");
        }
        group.SelectedTab = selections[selections.Count - 1].Tab;
        Settle();
      }

      Assert.True(vm.DockingManager.CanEnterViewportOnly, $"compact refused: {vm.DockingManager.ViewportOnlyBlockedBy()}");
      layout.Set(UiLayout.Compact);
      mini.IsExpanded = true;
      Collect("compact");
      // the tabs the compact layout parked in the tree, classic again
      layout.Set(UiLayout.Classic);
      foreach (var group in Nodes(vm.DockingManager.RootNode).OfType<TabGroupNodeViewModel>().Where(g => !g.Tabs.Any(t => t.IsPinned)))
      {
        var selected = group.SelectedTab;
        foreach (var tab in group.Tabs.ToList())
        {
          group.SelectedTab = tab;
          Collect($"classic after compact, tab {tab.AutomationKey}");
        }
        group.SelectedTab = selected;
        Settle();
      }
    }
    finally
    {
      mini.IsExpanded = miniExpanded;
      layout.Set(UiLayout.Classic);
      translation.CurrentCulture = culture;
      Dispatcher.UIThread.RunJobs();
      window.Close();
      Dispatcher.UIThread.RunJobs();
    }

    string path = ManifestPath();
    // A line "?Id" is a conditional id: its control exists only in some states of the shared app
    // (e.g. the Comet tab's Rotational Model section, an Expander that opens once a comet is
    // committed, which depends on the tests that ran before). It must be known and unique, but
    // may be absent from the window this test builds.
    var lines = File.ReadAllLines(path, Encoding.UTF8).Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
    static string Id(string line) => line.StartsWith('?') ? line.Substring(1) : line;
    var manifest = lines.Select(Id).ToList();
    var conditional = new HashSet<string>(lines.Where(l => l.StartsWith('?')).Select(Id), StringComparer.Ordinal);
    var sorted = manifest.OrderBy(l => l, StringComparer.Ordinal).ToList();
    var added = found.Except(manifest, StringComparer.Ordinal).ToList();
    var removed = manifest.Except(found, StringComparer.Ordinal).Where(id => !conditional.Contains(id))
      .OrderBy(l => l, StringComparer.Ordinal).ToList();

    var report = new StringBuilder();
    report.AppendLine($"{found.Count} AutomationIds in the window, {manifest.Count} in {ManifestFile}");
    report.AppendLine($"added (in the window, not in the manifest): {added.Count}");
    foreach (var id in added) report.AppendLine($"  + {id}");
    report.AppendLine($"removed (in the manifest, not in the window): {removed.Count}");
    foreach (var id in removed) report.AppendLine($"  - {id}");
    if (duplicates.Count > 0)
    {
      report.AppendLine($"duplicates: {duplicates.Count}");
      foreach (var d in duplicates) report.AppendLine($"  = {d}");
    }
    output.WriteLine(report.ToString());
    Console.Error.WriteLine($"[AutomationIdManifestTests] {report}");

    if (Environment.GetEnvironmentVariable(UpdateEnvVar) == "1" && (added.Count > 0 || removed.Count > 0))
    {
      // conditional ids keep their mark (and stay listed when absent from this window)
      var rewritten = new SortedSet<string>(found, StringComparer.Ordinal);
      rewritten.UnionWith(conditional);
      File.WriteAllText(path, string.Join("\n", rewritten.Select(id => conditional.Contains(id) ? "?" + id : id)) + "\n", new UTF8Encoding(false));
      output.WriteLine($"{ManifestFile} rewritten ({UpdateEnvVar}=1)");
      added.Clear();
      removed.Clear();
      manifest = found.ToList();
      sorted = manifest;
    }

    Assert.True(duplicates.Count == 0, $"AutomationIds set on more than one control:\n{report}");
    Assert.True(added.Count == 0 && removed.Count == 0, $"the window's AutomationIds differ from {ManifestFile}:\n{report}");
    Assert.True(manifest.SequenceEqual(sorted, StringComparer.Ordinal), $"{ManifestFile} is not sorted (ordinal)");
    Assert.Equal(manifest.Count, manifest.Distinct(StringComparer.Ordinal).Count());
    // the compact panel's contract is part of it
    var missing = CompactAutomationIds.All.Except(found, StringComparer.Ordinal).ToList();
    Assert.True(missing.Count == 0, $"ids of CompactAutomationIds not in the window: {string.Join(", ", missing)}");
  }
}
