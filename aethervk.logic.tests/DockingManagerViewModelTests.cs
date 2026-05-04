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

  public DockingManagerViewModelTests()
  {
    _mockFactory = new Mock<ITabFactory>();
    _mockFactory.Setup(f => f.CreateTab("UITestPanel")).Returns(new UITestPanelViewModel());
    _mockFactory
      .Setup(f => f.CreateTab("Console"))
      .Returns(new ConsoleViewModel(new ConsoleService(new Mock<IUiThreadDispatcher>().Object)));
    _mockFactory.Setup(f => f.CreateTab("Viewport3D")).Returns(new DebugUiViewModel());
    _mockFactory.Setup(f => f.CreateTab("Outline")).Returns(new DebugUiViewModel());
    _mockFactory.Setup(f => f.CreateTab("Properties")).Returns(new DebugUiViewModel());
  }

  public void Dispose() { }

  [Fact]
  public void DockingManager_InitializesWithDefaultTab()
  {
    var defaultTab = new DebugUiViewModel();
    var rootGroup = new TabGroupNodeViewModel(defaultTab, _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);

    Assert.NotNull(vm.RootNode);
    Assert.IsType<TabGroupNodeViewModel>(vm.RootNode);
    var currentRootGroup = (TabGroupNodeViewModel)vm.RootNode;
    Assert.Single(currentRootGroup.Tabs);
    Assert.Equal("Debug UI", currentRootGroup.Tabs[0].Title);
  }

  [Fact]
  public void AddNewTabCommand_AddsNewUITestPanelTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
    var initialTabCount = rootGroup.Tabs.Count;

    rootGroup.AddNewTabCommand.Execute("UITestPanel");

    Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
    Assert.IsType<UITestPanelViewModel>(rootGroup.SelectedTab);
    Assert.Equal("UI Test Panel", rootGroup.SelectedTab.Title);
  }

  [Fact]
  public void AddNewConsoleTabCommand_AddsNewConsoleTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
    var initialTabCount = rootGroup.Tabs.Count;

    rootGroup.AddNewTabCommand.Execute("Console");

    Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
    Assert.IsType<ConsoleViewModel>(rootGroup.SelectedTab);
    Assert.Equal("Console", rootGroup.SelectedTab.Title);
  }

  [Fact]
  public void ReceiveTabDroppedMessage_CenterZone_AddsTabToTarget()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);

    var newTab = new TabItemViewModel("New Tab");
    var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Center);

    vm.Receive(message);

    Assert.Contains(rootGroup.Tabs, t => t.Title == "New Tab");
  }

  [Fact]
  public void ReceiveTabDroppedMessage_LeftZone_SplitsHorizontally()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);

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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    rootGroup.Tabs.Add(new UITestPanelViewModel());
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
    var tab1 = rootGroup.Tabs[0];

    // Remove one tab, but another remains
    rootGroup.Tabs.Remove(tab1);

    Assert.Single(rootGroup.Tabs);
    Assert.NotNull(vm.RootNode); // Root node should still be the same TabGroup
  }

  [Fact]
  public void RemoveTabAndCoalesce_DoesNotCoalesceRootIfOnlyOneTab()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
    var tabToFind = rootGroup.Tabs[0];

    var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
    Assert.Equal(rootGroup, foundNode);
  }

  [Fact]
  public void FindNodeContainingTab_FindsTabInSplitNodeChild()
  {
    var rootGroup = new TabGroupNodeViewModel(new DebugUiViewModel(), _mockFactory.Object);
    var vm = new DockingManagerViewModel(_mockFactory.Object, rootGroup);
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
    var vm = new DockingManagerViewModel(_mockFactory.Object);
    var tabToFind = new TabItemViewModel("Non Existent Tab");

    var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
    Assert.Null(foundNode);
  }
}
