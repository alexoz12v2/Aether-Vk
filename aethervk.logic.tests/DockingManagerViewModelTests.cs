#if DEBUG
using System;
using System.Linq;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

// Put this in a non-parallel collection to prevent static state race conditions
[CollectionDefinition("Non-Parallel Collection", DisableParallelization = true)]
public class DockingManagerViewModelTests : IDisposable
{
  private readonly Mock<ITabFactory> _mockFactory;
  private readonly ConsoleService _consoleService;
  private readonly Mock<ILayoutMessenger> _mockLayoutMessenger;
  private readonly Mock<IConsoleMessenger> _mockConsoleMessenger;

  public DockingManagerViewModelTests()
  {
    _mockFactory = new Mock<ITabFactory>();
    _consoleService = new ConsoleService(new Mock<IUiThreadDispatcher>().Object);
    _mockLayoutMessenger = new Mock<ILayoutMessenger>();
    _mockConsoleMessenger = new Mock<IConsoleMessenger>();
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(UITestPanelViewModel))).Returns((new UITestPanelViewModel(), null));
    _mockFactory
      .Setup(f => f.CreateScopedTab(typeof(ConsoleViewModel)))
      .Returns(
        (new ConsoleViewModel(
          _consoleService,
          new Mock<IFileDialogService>().Object,
          _mockConsoleMessenger.Object
        ), null)
      );
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(Viewport3DViewModel))).Returns((new DebugUiViewModel(), null));
  }

  public void Dispose()
  {
    _consoleService.Dispose();
  }

  [Fact]
  public void DockingManager_InitializesWithDefaultTab()
  {
    var defaultTab = new DebugUiViewModel();
    var rootGroup = new TabGroupNodeViewModel(defaultTab, _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);

    Assert.NotNull(vm.RootNode);
    Assert.IsType<TabGroupNodeViewModel>(vm.RootNode);
    var currentRootGroup = (TabGroupNodeViewModel)vm.RootNode;
    Assert.Single(currentRootGroup.Tabs);
    Assert.Equal("Debug UI", currentRootGroup.Tabs[0].Title);
  }

  [Fact]
  public void AddNewTabCommand_AddsNewUITestPanelTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var initialTabCount = rootGroup.Tabs.Count;

    rootGroup.AddNewTabCommand.Execute(typeof(UITestPanelViewModel));

    Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
    Assert.IsType<UITestPanelViewModel>(rootGroup.SelectedTab);
    Assert.Equal("UI Test Panel", rootGroup.SelectedTab.Title);
  }

  [Fact]
  public void AddNewConsoleTabCommand_AddsNewConsoleTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var initialTabCount = rootGroup.Tabs.Count;

    rootGroup.AddNewTabCommand.Execute(typeof(ConsoleViewModel));

    Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
    Assert.IsType<ConsoleViewModel>(rootGroup.SelectedTab);
    Assert.Equal("Console", rootGroup.SelectedTab.Title);
  }

  [Fact]
  public void ReceiveTabDroppedMessage_CenterZone_AddsTabToTarget()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);

    var newTab = new TabItemViewModel("New Tab");
    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Center);

    vm.Receive(message);

    Assert.Contains(rootGroup.Tabs, t => t.Title == "New Tab");
  }

  [Fact]
  public void ReceiveTabDroppedMessage_LeftZone_SplitsHorizontally()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var originalTab = rootGroup.SelectedTab;
    var newTab = new TabItemViewModel("New Left Tab");

    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Left);
    vm.Receive(message);

    Assert.IsType<SplitNodeViewModel>(vm.RootNode);
    var splitNode = (SplitNodeViewModel)vm.RootNode;
    Assert.Equal(SplitOrientation.Horizontal, splitNode.Orientation);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.FirstChild);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.SecondChild);

    var leftGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
    var rightGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    Assert.Contains(newTab, leftGroup.Tabs);
    Assert.Contains(originalTab, rightGroup.Tabs);
    Assert.DoesNotContain(newTab, rightGroup.Tabs);
    Assert.DoesNotContain(originalTab, leftGroup.Tabs);
  }

  [Fact]
  public void ReceiveTabDroppedMessage_RightZone_SplitsHorizontally()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var originalTab = rootGroup.SelectedTab;
    var newTab = new TabItemViewModel("New Right Tab");

    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Right);
    vm.Receive(message);

    Assert.IsType<SplitNodeViewModel>(vm.RootNode);
    var splitNode = (SplitNodeViewModel)vm.RootNode;
    Assert.Equal(SplitOrientation.Horizontal, splitNode.Orientation);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.FirstChild);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.SecondChild);

    var leftGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
    var rightGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    Assert.Contains(originalTab, leftGroup.Tabs);
    Assert.Contains(newTab, rightGroup.Tabs);
    Assert.DoesNotContain(newTab, leftGroup.Tabs);
    Assert.DoesNotContain(originalTab, rightGroup.Tabs);
  }

  [Fact]
  public void ReceiveTabDroppedMessage_TopZone_SplitsVertically()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var originalTab = rootGroup.SelectedTab;
    var newTab = new TabItemViewModel("New Top Tab");

    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Top);
    vm.Receive(message);

    Assert.IsType<SplitNodeViewModel>(vm.RootNode);
    var splitNode = (SplitNodeViewModel)vm.RootNode;
    Assert.Equal(SplitOrientation.Vertical, splitNode.Orientation);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.FirstChild);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.SecondChild);

    var topGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
    var bottomGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    Assert.Contains(newTab, topGroup.Tabs);
    Assert.Contains(originalTab, bottomGroup.Tabs);
    Assert.DoesNotContain(newTab, bottomGroup.Tabs);
    Assert.DoesNotContain(originalTab, topGroup.Tabs);
  }

  [Fact]
  public void ReceiveTabDroppedMessage_BottomZone_SplitsVertically()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var originalTab = rootGroup.SelectedTab;
    var newTab = new TabItemViewModel("New Bottom Tab");

    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Bottom);
    vm.Receive(message);

    Assert.IsType<SplitNodeViewModel>(vm.RootNode);
    var splitNode = (SplitNodeViewModel)vm.RootNode;
    Assert.Equal(SplitOrientation.Vertical, splitNode.Orientation);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.FirstChild);
    Assert.IsType<TabGroupNodeViewModel>(splitNode.SecondChild);

    var topGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
    var bottomGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    Assert.Contains(originalTab, topGroup.Tabs);
    Assert.Contains(newTab, bottomGroup.Tabs);
    Assert.DoesNotContain(newTab, topGroup.Tabs);
    Assert.DoesNotContain(originalTab, bottomGroup.Tabs);
  }

  [Fact]
  public void RemoveTabAndCoalesce_RemovesTabAndCoalescesNode()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);

    // Split the root to create a more complex structure
    var newTab = new TabItemViewModel("New Tab");
    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Right);
    vm.Receive(message);

    var splitNode = (SplitNodeViewModel)vm.RootNode;
    var leftGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
    var rightGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    // Simulate dragging the ONLY tab in rightGroup into the center of leftGroup
    var tabToRemove = rightGroup.Tabs.First();
    var message2 = new TabDroppedMessage(tabToRemove, leftGroup, DockZone.Center);
    vm.Receive(message2);

    // The right group should now be empty, and the splitNode should coalesce
    // with the leftGroup becoming the new root.
    Assert.IsType<TabGroupNodeViewModel>(vm.RootNode);
    Assert.Null(vm.RootNode.Parent);
  }

  [Fact]
  public void RemoveTabAndCoalesce_DoesNotCoalesceIfTabsRemain()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    rootGroup.Tabs.Add(new UITestPanelViewModel());
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var tab1 = rootGroup.Tabs[0];

    // Remove one tab, but another remains
    rootGroup.Tabs.Remove(tab1);

    Assert.Single(rootGroup.Tabs);
    Assert.NotNull(vm.RootNode); // Root node should still be the same TabGroup
  }

  [Fact]
  public void RemoveTabAndCoalesce_DoesNotCoalesceRootIfOnlyOneTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var tab1 = rootGroup.Tabs[0];

    // Remove all tabs
    rootGroup.Tabs.Remove(tab1);

    // The root group should still exist, even if empty (to avoid null UI)
    Assert.Empty(rootGroup.Tabs);
    Assert.Equal(rootGroup, vm.RootNode);
  }

  [Fact]
  public void FindNodeContainingTab_FindsTabInRootGroup()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var tabToFind = rootGroup.Tabs[0];

    var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
    Assert.Equal(rootGroup, foundNode);
  }

  [Fact]
  public void FindNodeContainingTab_FindsTabInSplitNodeChild()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, rootGroup);
    var newTab = new TabItemViewModel("New Tab");

    // Split the root to create a split node
    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Right);
    vm.Receive(message);

    var splitNode = (SplitNodeViewModel)vm.RootNode;
    var rightGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

    var foundNode = vm.FindNodeContainingTab(vm.RootNode, newTab);
    Assert.Equal(rightGroup, foundNode);
  }

  [Fact]
  public void FindNodeContainingTab_ReturnsNullIfTabNotFound()
  {
    var vm = new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object);
    var tabToFind = new TabItemViewModel("Non Existent Tab");

    var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
    Assert.Null(foundNode);
  }

  // ── single, pinned viewport (a second viewport crashed the app; re-docking re-created it) ──

  private sealed class PinnedSingleTab() : TabItemViewModel("Pinned")
  {
    public override bool IsPinned => true;
    public override bool IsSingleInstance => true;
  }

  private (DockingManagerViewModel Vm, TabGroupNodeViewModel Left, TabGroupNodeViewModel Right, PinnedSingleTab Pinned)
    SplitLayoutWithPinnedTab()
  {
    var pinned = new PinnedSingleTab();
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(PinnedSingleTab))).Returns(() => (new PinnedSingleTab(), null));
    _mockFactory.Setup(f => f.AvailableTabs).Returns(
      [new TabDescriptor("Pinned", typeof(PinnedSingleTab), false), new TabDescriptor("UI Test Panel", typeof(UITestPanelViewModel), false)]);
    var left = new TabGroupNodeViewModel(pinned, _mockFactory.Object, _mockLayoutMessenger.Object);
    var right = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object, _mockLayoutMessenger.Object);
    var split = new SplitNodeViewModel(left, right, SplitOrientation.Horizontal);
    left.Parent = split;
    right.Parent = split;
    return (new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object, split), left, right, pinned);
  }

  [Fact]
  public void SingleInstanceTab_IsRefusedAnywhereInTheLayout_AndHiddenFromTheMenu()
  {
    var (_, _, right, _) = SplitLayoutWithPinnedTab();

    right.AddNewTabCommand.Execute(typeof(PinnedSingleTab));
    right.ChangeSelectedTabCommand.Execute(typeof(PinnedSingleTab));

    Assert.DoesNotContain(right.Tabs, t => t is PinnedSingleTab);
    Assert.False(right.CanCreateTab(typeof(PinnedSingleTab)));
    Assert.DoesNotContain(right.AvailableTabs, d => d.TabType == typeof(PinnedSingleTab));
    Assert.Contains(right.AvailableTabs, d => d.TabType == typeof(UITestPanelViewModel));
    _mockFactory.Verify(f => f.CreateScopedTab(typeof(PinnedSingleTab)), Times.Never);
  }

  [Fact]
  public void PinnedTab_CannotBeClosedReplacedOrDragged()
  {
    var (vm, left, right, pinned) = SplitLayoutWithPinnedTab();

    left.CloseTabCommand.Execute(pinned);
    left.CloseAllTabsCommand.Execute(null);
    left.ChangeSelectedTabCommand.Execute(typeof(UITestPanelViewModel));
    vm.Receive(new TabDroppedMessage(pinned, right, DockZone.Center));
    vm.Receive(new TabDroppedMessage(pinned, right, DockZone.Left));

    Assert.Same(pinned, left.SelectedTab);
    Assert.Contains(pinned, left.Tabs);
    Assert.DoesNotContain(pinned, right.Tabs);
    Assert.Same(left, vm.FindNodeContainingTab(vm.RootNode, pinned));
  }

  [Fact]
  public void CloseAllTabs_KeepsThePinnedTab_ClosesTheOthers()
  {
    var (_, left, _, pinned) = SplitLayoutWithPinnedTab();
    left.AddNewTabCommand.Execute(typeof(UITestPanelViewModel));

    left.CloseAllTabsCommand.Execute(null);

    Assert.Equal([pinned], left.Tabs);
  }

  // ── viewport-only presentation: only Collapsed / IsChromeHidden change, never the tree ──

  private sealed class ParkedTab() : TabItemViewModel("Parked");

  private TabGroupNodeViewModel Group(TabItemViewModel tab) =>
    new(tab, _mockFactory.Object, _mockLayoutMessenger.Object);

  private static SplitNodeViewModel Split(LayoutNodeViewModelBase first, LayoutNodeViewModelBase second, SplitOrientation orientation)
  {
    var split = new SplitNodeViewModel(first, second, orientation);
    first.Parent = split;
    second.Parent = split;
    return split;
  }

  private DockingManagerViewModel Manager(LayoutNodeViewModelBase root) =>
    new(_mockFactory.Object, _mockLayoutMessenger.Object, root);

  private static System.Collections.Generic.IEnumerable<LayoutNodeViewModelBase> Nodes(LayoutNodeViewModelBase node) =>
    node is SplitNodeViewModel s ? new[] { node }.Concat(Nodes(s.FirstChild)).Concat(Nodes(s.SecondChild)) : new[] { node };

  /// <summary>The default layout with a pinned single-instance tab standing in for the 3D viewport.</summary>
  private DockingManagerViewModel DefaultLayoutWithViewport(out PinnedSingleTab viewport)
  {
    var pinned = new PinnedSingleTab();
    viewport = pinned;
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(Viewport3DViewModel))).Returns(() => (pinned, null));
    foreach (var type in new[] { typeof(SettingsTabViewModel), typeof(CometTabViewModel), typeof(ModelTabViewModel), typeof(ImportsTabViewModel), typeof(TimelineTabViewModel) })
      _mockFactory.Setup(f => f.CreateScopedTab(type)).Returns(() => (new TabItemViewModel(type.Name), null));
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(ParkedTab))).Returns(() => (new ParkedTab(), null));
    return new DockingManagerViewModel(_mockFactory.Object, _mockLayoutMessenger.Object);
  }

  private static void AssertAllReset(DockingManagerViewModel vm)
  {
    Assert.All(Nodes(vm.RootNode).OfType<SplitNodeViewModel>(), s =>
    {
      Assert.Equal(CollapsedSide.None, s.Collapsed);
      Assert.True(s.IsFirstShown && s.IsSecondShown && s.IsSplitterShown);
    });
    Assert.All(Nodes(vm.RootNode).OfType<TabGroupNodeViewModel>(), g => Assert.False(g.IsChromeHidden));
  }

  [Fact]
  public void ViewportOnly_CollapsesOffPathSiblings_DefaultLayout()
  {
    var vm = DefaultLayoutWithViewport(out var viewport);
    // settings | ((viewport | imports) / timeline)
    var root = Assert.IsType<SplitNodeViewModel>(vm.RootNode);
    var allButSettings = Assert.IsType<SplitNodeViewModel>(root.SecondChild);
    var viewportAndImports = Assert.IsType<SplitNodeViewModel>(allButSettings.FirstChild);
    var center = Assert.IsType<TabGroupNodeViewModel>(viewportAndImports.FirstChild);
    Assert.Same(viewport, center.SelectedTab);

    Assert.True(vm.CanEnterViewportOnly);
    Assert.True(vm.SetViewportOnly(true));

    Assert.True(vm.IsViewportOnly);
    Assert.Equal(CollapsedSide.First, root.Collapsed);
    Assert.Equal(CollapsedSide.Second, allButSettings.Collapsed);
    Assert.Equal(CollapsedSide.Second, viewportAndImports.Collapsed);
    Assert.False(root.IsFirstShown);
    Assert.True(root.IsSecondShown);
    Assert.False(root.IsSplitterShown);
    Assert.True(viewportAndImports.IsFirstShown);
    Assert.False(viewportAndImports.IsSecondShown);
    Assert.True(center.IsChromeHidden);
    Assert.All(Nodes(vm.RootNode).OfType<TabGroupNodeViewModel>().Where(g => g != center), g => Assert.False(g.IsChromeHidden));
  }

  [Fact]
  public void ViewportOnly_CollapsedRaisesTheShownFlags()
  {
    var split = Split(Group(new PinnedSingleTab()), Group(new DebugUiViewModel()), SplitOrientation.Horizontal);
    var vm = Manager(split);
    var raised = new System.Collections.Generic.List<string?>();
    split.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

    vm.SetViewportOnly(true);

    Assert.Contains(nameof(SplitNodeViewModel.Collapsed), raised);
    Assert.Contains(nameof(SplitNodeViewModel.IsFirstShown), raised);
    Assert.Contains(nameof(SplitNodeViewModel.IsSecondShown), raised);
    Assert.Contains(nameof(SplitNodeViewModel.IsSplitterShown), raised);
  }

  [Fact]
  public void ViewportOnly_RearrangedTree()
  {
    // (a1 | a2) / (((b | viewport) / c) | d): the viewport sits three splits deep, on second sides
    var viewportGroup = Group(new PinnedSingleTab());
    var offPath = Split(Group(new DebugUiViewModel()), Group(new DebugUiViewModel()), SplitOrientation.Horizontal);
    var inner = Split(Group(new DebugUiViewModel()), viewportGroup, SplitOrientation.Horizontal);
    var middle = Split(inner, Group(new DebugUiViewModel()), SplitOrientation.Vertical);
    var outer = Split(middle, Group(new DebugUiViewModel()), SplitOrientation.Horizontal);
    var root = Split(offPath, outer, SplitOrientation.Vertical);
    var vm = Manager(root);

    Assert.True(vm.SetViewportOnly(true));

    Assert.Equal(CollapsedSide.First, root.Collapsed);
    Assert.Equal(CollapsedSide.Second, outer.Collapsed);
    Assert.Equal(CollapsedSide.Second, middle.Collapsed);
    Assert.Equal(CollapsedSide.First, inner.Collapsed);
    Assert.Equal(CollapsedSide.None, offPath.Collapsed);
    Assert.True(viewportGroup.IsChromeHidden);
    Assert.Single(Nodes(vm.RootNode).OfType<TabGroupNodeViewModel>(), g => g.IsChromeHidden);
  }

  [Fact]
  public void ViewportOnly_Off_RestoresNone()
  {
    var vm = DefaultLayoutWithViewport(out _);
    Assert.True(vm.SetViewportOnly(true));

    Assert.True(vm.SetViewportOnly(false));

    Assert.False(vm.IsViewportOnly);
    AssertAllReset(vm);
  }

  [Fact]
  public void ViewportOnly_DoesNotMutateTree()
  {
    var vm = DefaultLayoutWithViewport(out var viewport);
    var rootBefore = vm.RootNode;
    var before = Nodes(vm.RootNode).Select(Snapshot).ToList();

    for (var i = 0; i < 3; ++i)
    {
      Assert.True(vm.SetViewportOnly(true));
      Assert.Same(rootBefore, vm.RootNode);
      Assert.Equal(before, Nodes(vm.RootNode).Select(Snapshot).ToList());
      Assert.True(vm.SetViewportOnly(false));
      Assert.Same(rootBefore, vm.RootNode);
      Assert.Equal(before, Nodes(vm.RootNode).Select(Snapshot).ToList());
    }
    Assert.Same(viewport, vm.FindNodeContainingTab(vm.RootNode, viewport)!.SelectedTab);

    // reference identity of every node, its parent, its children / tabs and its selection
    static string Snapshot(LayoutNodeViewModelBase n) =>
      $"{Id(n)} p={Id(n.Parent)} " + n switch
      {
        SplitNodeViewModel s => $"split {Id(s.FirstChild)} {Id(s.SecondChild)} {s.Orientation} {s.SplitRatio}",
        TabGroupNodeViewModel g => $"group sel={Id(g.SelectedTab)} tabs={string.Join(",", g.Tabs.Select(Id))}",
        _ => "?",
      };
    static string Id(object? o) =>
      o is null ? "null" : System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(o).ToString();
  }

  [Fact]
  public void ViewportOnly_Refused_WhenViewportNotSelected()
  {
    var pinned = new PinnedSingleTab();
    var viewportGroup = Group(pinned);
    var other = new DebugUiViewModel();
    viewportGroup.Tabs.Add(other);
    viewportGroup.SelectedTab = other;
    var root = Split(viewportGroup, Group(new DebugUiViewModel()), SplitOrientation.Horizontal);
    var vm = Manager(root);

    Assert.False(vm.CanEnterViewportOnly);
    Assert.Equal(ViewportOnlyBlocker.ViewportNotSelected, vm.ViewportOnlyBlockedBy());
    Assert.False(vm.SetViewportOnly(true));

    Assert.False(vm.IsViewportOnly);
    Assert.Same(other, viewportGroup.SelectedTab);
    AssertAllReset(vm);
  }

  [Fact]
  public void ViewportOnly_Refused_WithoutViewport()
  {
    var root = Split(Group(new DebugUiViewModel()), Group(new DebugUiViewModel()), SplitOrientation.Horizontal);
    var vm = Manager(root);

    Assert.Equal(ViewportOnlyBlocker.NoViewport, vm.ViewportOnlyBlockedBy());
    Assert.False(vm.SetViewportOnly(true));
    AssertAllReset(vm);
  }

  [Fact]
  public void EnsureTab_ReturnsExisting()
  {
    var parked = new ParkedTab();
    var holder = Group(new DebugUiViewModel());
    holder.Tabs.Add(parked);
    var vm = Manager(Split(Group(new PinnedSingleTab()), holder, SplitOrientation.Horizontal));

    Assert.Same(parked, vm.EnsureTab<ParkedTab>());
    Assert.Same(parked, vm.EnsureTab(typeof(ParkedTab)));
    Assert.Equal(2, holder.Tabs.Count);
    _mockFactory.Verify(f => f.CreateScopedTab(typeof(ParkedTab)), Times.Never);
  }

  [Fact]
  public void EnsureTab_CreatesParked_NeverSelectsOverViewport()
  {
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(ParkedTab))).Returns(() => (new ParkedTab(), null));
    var pinned = new PinnedSingleTab();
    var viewportGroup = Group(pinned);
    var otherSelected = new DebugUiViewModel();
    var otherGroup = Group(otherSelected);
    var root = Split(viewportGroup, otherGroup, SplitOrientation.Horizontal);
    var vm = Manager(root);

    var tab = vm.EnsureTab<ParkedTab>();

    Assert.Contains(tab, otherGroup.Tabs);
    Assert.DoesNotContain(tab, viewportGroup.Tabs);
    Assert.False(tab.IsSelected);
    Assert.Same(otherSelected, otherGroup.SelectedTab);
    Assert.Same(pinned, viewportGroup.SelectedTab);
    Assert.Same(root, vm.RootNode);
    Assert.Same(tab, vm.EnsureTab<ParkedTab>());
    _mockFactory.Verify(f => f.CreateScopedTab(typeof(ParkedTab)), Times.Once);
  }

  [Fact]
  public void EnsureTab_NeverSplitsViewportGroup()
  {
    _mockFactory.Setup(f => f.CreateScopedTab(typeof(ParkedTab))).Returns(() => (new ParkedTab(), null));
    var pinned = new PinnedSingleTab();
    var viewportGroup = Group(pinned);
    var vm = Manager(viewportGroup);

    var tab = vm.EnsureTab<ParkedTab>();

    Assert.Same(viewportGroup, vm.RootNode);
    Assert.Null(viewportGroup.Parent);
    Assert.Equal(new TabItemViewModel[] { pinned, tab }, viewportGroup.Tabs);
    Assert.Same(pinned, viewportGroup.SelectedTab);
    Assert.False(tab.IsSelected);
    Assert.True(vm.SetViewportOnly(true));
    Assert.True(viewportGroup.IsChromeHidden);
  }
}
#endif
