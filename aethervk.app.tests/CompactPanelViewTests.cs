using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AetherVk.Controls;
using AetherVk.Logic.Automation;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace AetherVk.AppTests;

/// <summary>
/// The compact layout in the real window, headless: the panel takes its column beside the docking
/// area without the 3D viewport ever moving, the nucleus view is outside every ScrollViewer, every
/// AutomationId of the panel is there once, the cards are bound to the docking tree's own tab view
/// models, and the panel speaks Italian.
/// </summary>
public class CompactPanelViewTests
{
  private sealed class Session : IDisposable
  {
    public required IServiceProvider Services { get; init; }
    public required MainWindow Window { get; init; }
    public required MainWindowViewModel Vm { get; init; }
    public required DockingManagerViewModel Docking { get; init; }
    public required UiLayoutService Layout { get; init; }
    public required ITranslationService Translation { get; init; }
    public required CultureInfo Culture { get; init; }
    /// <summary>The profile's expander state, put back at the end (the view model saves it).</summary>
    public required bool MiniExpanded { get; init; }

    public void ExpandMini()
    {
      Vm.CompactPanel.NucleusMini.IsExpanded = true;
      Settle();
    }

    public CompactPanelView Panel => Window.GetVisualDescendants().OfType<CompactPanelView>().Single();

    public void Settle()
    {
      Dispatcher.UIThread.RunJobs();
      Window.UpdateLayout();
      Dispatcher.UIThread.RunJobs();
    }

    public void EnterCompact()
    {
      Assert.True(Docking.CanEnterViewportOnly, $"compact refused: {Docking.ViewportOnlyBlockedBy()}");
      Layout.Set(UiLayout.Compact);
      Settle();
      Assert.True(Vm.IsCompactLayout, $"the window did not enter the compact layout: {Vm.LayoutStatus}");
    }

    public void Dispose()
    {
      Vm.CompactPanel.NucleusMini.IsExpanded = MiniExpanded;
      Layout.Set(UiLayout.Classic);
      Translation.CurrentCulture = Culture;
      Dispatcher.UIThread.RunJobs();
      Window.Close();
      Dispatcher.UIThread.RunJobs();
    }
  }

  private static Session Open()
  {
    var services = (App.Host ?? throw new InvalidOperationException("DI host not initialized")).Services;
    var vm = services.GetRequiredService<MainWindowViewModel>();
    var layout = services.GetRequiredService<UiLayoutService>();
    var translation = services.GetRequiredService<ITranslationService>();
    var culture = translation.CurrentCulture;
    // the shared host may carry a layout or a language left by earlier tests
    layout.Set(UiLayout.Classic);
    translation.CurrentCulture = new CultureInfo("en");
    Dispatcher.UIThread.RunJobs();
    var window = new MainWindow { DataContext = vm, Width = 1400, Height = 900 };
    window.Show();
    var session = new Session
    {
      Services = services,
      Window = window,
      Vm = vm,
      Docking = vm.DockingManager,
      Layout = layout,
      Translation = translation,
      Culture = culture,
      MiniExpanded = vm.CompactPanel.NucleusMini.IsExpanded,
    };
    session.Settle();
    return session;
  }

  /// <summary>The id a control answers to in UI Automation: its AutomationId, else its name.</summary>
  private static string? EffectiveId(Control c) =>
    AutomationProperties.GetAutomationId(c) is { Length: > 0 } id ? id : c.Name;

  /// <summary>
  /// The AutomationIds set in <paramref name="root"/>'s visual tree, one entry per control. A control
  /// that only repeats the id of one of its ancestors is not counted: a NumericUpDown and a help
  /// button hand their id to the inner control UI Automation actually sees.
  /// </summary>
  internal static List<string> AutomationIds(Visual root)
  {
    var ids = new List<string>();
    foreach (var control in root.GetVisualDescendants().OfType<Control>())
    {
      var id = AutomationProperties.GetAutomationId(control);
      if (string.IsNullOrEmpty(id))
        continue;
      if (control.GetVisualAncestors().OfType<Control>().Any(a => AutomationProperties.GetAutomationId(a) == id))
        continue;
      ids.Add(id!);
    }
    return ids;
  }

  [AvaloniaFact]
  public void Classic_ThePanelTakesNoRoom_Compact_ItSharesTheRowWithTheDockingArea()
  {
    using var s = Open();
    var docking = s.Window.GetVisualDescendants().OfType<DockingManagerView>().Single();
    var viewport = Assert.Single(s.Window.GetVisualDescendants().OfType<Viewport3DView>());
    var chain = viewport.GetVisualAncestors().ToList();
    double classicWidth = docking.Bounds.Width;
    Assert.False(s.Panel.IsVisible);
    Assert.InRange(classicWidth, s.Window.ClientSize.Width - 1, s.Window.ClientSize.Width + 1);

    for (int i = 0; i < 3; i++)
    {
      s.EnterCompact();
      Assert.True(s.Panel.IsVisible);
      Assert.True(s.Panel.Bounds.Width >= MainWindow.CompactPanelMinWidth, $"panel {s.Panel.Bounds.Width} wide");
      // 3* / 7* of the row
      Assert.InRange(s.Panel.Bounds.Width / s.Window.ClientSize.Width, 0.25, 0.35);
      Assert.True(docking.Bounds.Width < classicWidth);
      // the viewport was resized, never re-parented
      Assert.Same(viewport, Assert.Single(s.Window.GetVisualDescendants().OfType<Viewport3DView>()));
      Assert.True(chain.SequenceEqual(viewport.GetVisualAncestors()), "the viewport's parent chain changed");

      s.Layout.Set(UiLayout.Classic);
      s.Settle();
      Assert.False(s.Panel.IsVisible);
      Assert.InRange(docking.Bounds.Width, classicWidth - 1, classicWidth + 1);
      Assert.True(chain.SequenceEqual(viewport.GetVisualAncestors()), "the viewport's parent chain changed");
    }
  }

  [AvaloniaFact]
  public void Compact_HasExactlyOneViewportHost_AndTheMiniHostIsNotIt()
  {
    using var s = Open();
    s.EnterCompact();

    var hosts = s.Window.GetVisualDescendants().OfType<Control>().Where(c => EffectiveId(c) == "ViewportHost").ToList();
    var main = Assert.IsType<VulkanViewportControl>(Assert.Single(hosts));
    Assert.NotNull(main.FindAncestorOfType<Viewport3DView>());
    Assert.Null(main.FindAncestorOfType<CompactPanelView>());

    // the mini-view is expanded by default in a fresh profile; show it whatever the profile says
    s.ExpandMini();
    var miniView = Assert.Single(s.Window.GetVisualDescendants().OfType<NucleusMiniView>());
    var mini = Assert.Single(s.Window.GetVisualDescendants().OfType<VulkanViewportControl>(),
      c => AutomationProperties.GetAutomationId(c) == CompactAutomationIds.NucleusMiniViewHost);
    Assert.Same(miniView.Host, mini);
    Assert.NotSame(main, mini);
    Assert.NotEqual("ViewportHost", EffectiveId(mini));
    var miniVm = Assert.IsType<VulkanViewportControlViewModel>(mini.DataContext);
    Assert.Equal(ViewportRole.NucleusMini, miniVm.Role);
    Assert.Same(s.Vm.CompactPanel.NucleusMini.Viewport, miniVm);
    Assert.Equal(2, s.Window.GetVisualDescendants().OfType<VulkanViewportControl>().Count());
  }

  [AvaloniaFact]
  public void TheMiniView_IsOutsideEveryScrollViewer_SquareAndAtMost320()
  {
    using var s = Open();
    s.EnterCompact();
    s.ExpandMini();

    var miniView = Assert.Single(s.Window.GetVisualDescendants().OfType<NucleusMiniView>());
    Assert.Null(miniView.FindAncestorOfType<ScrollViewer>());
    Assert.NotNull(miniView.FindAncestorOfType<CompactPanelView>());
    var host = Assert.IsType<VulkanViewportControl>(miniView.Host);
    Assert.Null(host.FindAncestorOfType<ScrollViewer>());
    // the cards do scroll
    var card = Assert.Single(s.Panel.GetVisualDescendants().OfType<Control>(),
      c => AutomationProperties.GetAutomationId(c) == CompactAutomationIds.CometCard);
    Assert.NotNull(card.FindAncestorOfType<ScrollViewer>());

    Assert.Equal(host.Width, host.Height);
    Assert.InRange(host.Width, 1, NucleusMiniViewModel.MaxSide);
    Assert.True(host.Width <= s.Panel.Bounds.Width - NucleusMiniViewModel.PanelGutter,
      $"host {host.Width} in a panel {s.Panel.Bounds.Width} wide");
    // 8 px of padding around the native area
    var frame = Assert.IsType<Border>(host.GetVisualAncestors().OfType<Border>().First());
    Assert.Equal(new Thickness(8), frame.Padding);

    // the same control through collapse, expand and both layouts
    for (int i = 0; i < 3; i++)
    {
      s.Vm.CompactPanel.NucleusMini.IsExpanded = false;
      s.Settle();
      s.Vm.CompactPanel.NucleusMini.IsExpanded = true;
      s.Settle();
      s.Layout.Set(UiLayout.Classic);
      s.Settle();
      s.EnterCompact();
      Assert.Same(host, miniView.Host);
      Assert.Same(host, Assert.Single(s.Window.GetVisualDescendants().OfType<VulkanViewportControl>(),
        c => AutomationProperties.GetAutomationId(c) == CompactAutomationIds.NucleusMiniViewHost));
    }
  }

  [AvaloniaFact]
  public void Compact_EveryAutomationIdIsThereOnce_AndNoneIsAClassicOne()
  {
    using var s = Open();
    // what the classic layout shows before the panel exists on screen
    var classicIds = AutomationIds(s.Window).Where(id => !CompactAutomationIds.All.Contains(id)).ToHashSet();
    s.EnterCompact();
    s.ExpandMini();

    var ids = AutomationIds(s.Window);
    var duplicates = ids.GroupBy(id => id).Where(g => g.Count() > 1).Select(g => $"{g.Key} ×{g.Count()}").ToList();
    Assert.True(duplicates.Count == 0, $"duplicate AutomationIds: {string.Join(", ", duplicates)}");

    var missing = CompactAutomationIds.All.Except(ids).ToList();
    Assert.True(missing.Count == 0, $"not in the compact window: {string.Join(", ", missing)}");
    Assert.Equal(CompactAutomationIds.All.Length, CompactAutomationIds.All.Distinct().Count());
    var reused = CompactAutomationIds.All.Intersect(classicIds).ToList();
    Assert.True(reused.Count == 0, $"ids of the classic layout reused: {string.Join(", ", reused)}");

    // everything of the panel is one of the contract's ids, and the panel is the control named so
    var inPanel = AutomationIds(s.Panel);
    var unknown = inPanel.Except(CompactAutomationIds.All).ToList();
    Assert.True(unknown.Count == 0, $"ids of the panel outside CompactAutomationIds: {string.Join(", ", unknown)}");
    Assert.Equal(CompactAutomationIds.CompactPanel, AutomationProperties.GetAutomationId(s.Panel));
  }

  [AvaloniaFact]
  public void TheCards_AreBoundToTheDockingTreesTabViewModels()
  {
    using var s = Open();
    s.EnterCompact();

    var panelVm = Assert.IsType<CompactPanelViewModel>(s.Panel.DataContext);
    Assert.Same(s.Vm.CompactPanel, panelVm);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<CometTabViewModel>()), panelVm.Comet);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<ModelTabViewModel>()), panelVm.Model);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<TimelineTabViewModel>()), panelVm.Timeline);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<ImportsTabViewModel>()), panelVm.Imports);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<SettingsTabViewModel>()), panelVm.Settings);
    Assert.Same(s.Vm.ModeBar, panelVm.ModeBar);

    // every control of the panel sees the panel's view model (or, in the mini-view, its own)
    var miniView = Assert.Single(s.Panel.GetVisualDescendants().OfType<NucleusMiniView>());
    Assert.Same(panelVm.NucleusMini, miniView.DataContext);
    var search = Assert.IsType<TextBox>(Assert.Single(s.Panel.GetVisualDescendants().OfType<Control>(),
      c => AutomationProperties.GetAutomationId(c) == CompactAutomationIds.CometSearch && c is TextBox));
    Assert.Same(panelVm, search.DataContext);

    // one state, two layouts: an edit in a card is the classic tab's value
    panelVm.Comet!.SearchQuery = "67P";
    s.Settle();
    Assert.Equal("67P", search.Text);
    search.Text = "10P";
    s.Settle();
    Assert.Equal("10P", panelVm.Comet.SearchQuery);
    panelVm.Comet.SearchQuery = string.Empty;

    // leaving and entering again keeps those instances
    s.Layout.Set(UiLayout.Classic);
    s.Settle();
    s.EnterCompact();
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<CometTabViewModel>()), panelVm.Comet);
    Assert.Same(Assert.Single(s.Docking.LayoutTabs().OfType<SettingsTabViewModel>()), panelVm.Settings);
  }

  [AvaloniaFact]
  public void ThePanel_SpeaksItalian()
  {
    using var s = Open();
    s.EnterCompact();
    s.ExpandMini();
    var expander = Assert.Single(s.Panel.GetVisualDescendants().OfType<Expander>(),
      e => AutomationProperties.GetAutomationId(e) == CompactAutomationIds.NucleusMiniViewExpander);
    Assert.Equal("Nucleus seen from Earth", expander.Header);

    s.Translation.CurrentCulture = new CultureInfo("it");
    s.Settle();

    Assert.Equal("Nucleo visto da Terra", expander.Header);
    var texts = s.Panel.GetVisualDescendants().OfType<TextBlock>().Select(t => t.Text).Where(t => !string.IsNullOrEmpty(t)).ToHashSet();
    foreach (var expected in new[]
    {
      "1 · Cometa", "2 · Periodo di osservazione", "3 · Nucleo", "4 · Getti", "5 · Polvere del getto selezionato",
      "6 · Immagine FITS", "7 · Vista", "8 · Riproduzione", "9 · Esporta",
      "Numero di rotazioni", "Periodo di rotazione (ore)", "Asse di rotazione: ascensione retta (°)",
      "Asse di rotazione: declinazione (°)",
    })
      Assert.True(texts.Contains(expected), $"\"{expected}\" is not shown");
    var buttons = s.Panel.GetVisualDescendants().OfType<Button>().Select(b => b.Content as string).ToHashSet();
    Assert.Contains("Aggiungi un getto", buttons);
    Assert.Contains("Importa e sovrapponi", buttons);
    Assert.Contains("Altre impostazioni → Layout classico", buttons);
    Assert.Contains("Torna alla vista da Terra", buttons);
    // nothing of the panel fell back to a missing-key marker
    Assert.DoesNotContain(texts, t => t!.StartsWith("[Compact_", StringComparison.Ordinal) || t.StartsWith("[NucleusMini_", StringComparison.Ordinal));
  }

  [AvaloniaFact]
  public void MoreSettings_IsTheWayBackToTheClassicLayout()
  {
    using var s = Open();
    s.EnterCompact();
    var more = Assert.IsType<Button>(Assert.Single(s.Panel.GetVisualDescendants().OfType<Control>(),
      c => AutomationProperties.GetAutomationId(c) == CompactAutomationIds.MoreSettings));
    // not executed here: it also saves the layout preference of whoever runs the tests
    Assert.Same(s.Vm.CompactPanel.ClassicLayoutCommand, more.Command);
    Assert.True(more.Command!.CanExecute(null));
    Assert.True(more.IsEffectivelyVisible);
    Assert.Null(more.FindAncestorOfType<ScrollViewer>()); // always reachable, whatever the scroll position
  }
}
