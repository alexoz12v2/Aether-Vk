using Xunit;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Messages;
using System.Linq;

namespace AetherVk.Logic.Tests;

public class DockingManagerViewModelTests
{
    [Fact]
    public void DockingManager_InitializesWithDefaultTab()
    {
        var vm = new DockingManagerViewModel();
        Assert.NotNull(vm.RootNode);
        Assert.IsType<TabGroupNodeViewModel>(vm.RootNode);
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        Assert.Equal(2, rootGroup.Tabs.Count); // Default View + UITestPanel
        Assert.Equal("Default View", rootGroup.Tabs[0].Title);
        Assert.Equal("UI Test Panel", rootGroup.Tabs[1].Title);
    }

    [Fact]
    public void AddNewTabCommand_AddsNewUITestPanelTab()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var initialTabCount = rootGroup.Tabs.Count;

        rootGroup.AddNewTabCommand.Execute(null);

        Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
        Assert.IsType<UITestPanelViewModel>(rootGroup.SelectedTab);
        Assert.Equal("UI Test Panel", rootGroup.SelectedTab.Title);
    }

    [Fact]
    public void AddNewConsoleTabCommand_AddsNewConsoleTab()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var initialTabCount = rootGroup.Tabs.Count;

        rootGroup.AddNewConsoleTabCommand.Execute(null);

        Assert.Equal(initialTabCount + 1, rootGroup.Tabs.Count);
        Assert.IsType<ConsoleViewModel>(rootGroup.SelectedTab);
        Assert.Equal("Console", rootGroup.SelectedTab.Title);
    }

    [Fact]
    public void ReceiveTabDroppedMessage_CenterZone_AddsTabToTarget()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var tab1 = rootGroup.Tabs[0];
        var tab2 = rootGroup.Tabs[1];

        // Simulate dragging tab1 from rootGroup to rootGroup (center zone)
        // This should effectively be a no-op if the tab is already there,
        // or reorder if reordering logic was implemented.
        // For now, we'll test adding a new tab to the same group.
        var newTab = new TabItemViewModel("New Tab");
        rootGroup.Tabs.Add(newTab); // Add a new tab to simulate a drag target

        var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Center);
        vm.Receive(message);

        Assert.Contains(newTab, rootGroup.Tabs);
        Assert.Equal(newTab, rootGroup.SelectedTab);
    }

    [Fact]
    public void ReceiveTabDroppedMessage_LeftZone_SplitsHorizontally()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
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
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
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
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
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
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
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
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var tab1 = rootGroup.Tabs[0];
        var tab2 = rootGroup.Tabs[1];

        // Split the root to create a more complex structure
        var newTab = new TabItemViewModel("New Tab");
        var message = new TabDroppedMessage(newTab, rootGroup, DockZone.Right);
        vm.Receive(message);

        var splitNode = (SplitNodeViewModel)vm.RootNode;
        var leftGroup = (TabGroupNodeViewModel)splitNode.FirstChild;
        var rightGroup = (TabGroupNodeViewModel)splitNode.SecondChild;

        // Remove the last tab from the right group
        var tabToRemove = rightGroup.Tabs.First(); // This should be originalTab
        rightGroup.Tabs.Remove(tabToRemove);

        // The right group should now be empty, and the splitNode should coalesce
        // with the leftGroup becoming the new root.
        Assert.Equal(leftGroup, vm.RootNode);
        Assert.Null(leftGroup.Parent);
    }

    [Fact]
    public void RemoveTabAndCoalesce_DoesNotCoalesceIfTabsRemain()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var tab1 = rootGroup.Tabs[0];
        var tab2 = rootGroup.Tabs[1];

        // Remove one tab, but another remains
        rootGroup.Tabs.Remove(tab1);

        Assert.Equal(1, rootGroup.Tabs.Count);
        Assert.NotNull(vm.RootNode); // Root node should still be the same TabGroup
    }

    [Fact]
    public void RemoveTabAndCoalesce_DoesNotCoalesceRootIfOnlyOneTab()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var tab1 = rootGroup.Tabs[0];
        var tab2 = rootGroup.Tabs[1];

        // Remove all but one tab
        rootGroup.Tabs.Remove(tab1);
        rootGroup.Tabs.Remove(tab2);

        // The root group should still exist, even if empty (to avoid null UI)
        Assert.Equal(0, rootGroup.Tabs.Count);
        Assert.Equal(rootGroup, vm.RootNode);
    }

    [Fact]
    public void FindNodeContainingTab_FindsTabInRootGroup()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var tabToFind = rootGroup.Tabs[0];

        var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
        Assert.Equal(rootGroup, foundNode);
    }

    [Fact]
    public void FindNodeContainingTab_FindsTabInSplitNodeChild()
    {
        var vm = new DockingManagerViewModel();
        var rootGroup = (TabGroupNodeViewModel)vm.RootNode;
        var originalTab = rootGroup.SelectedTab;
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
        var vm = new DockingManagerViewModel();
        var tabToFind = new TabItemViewModel("Non Existent Tab");

        var foundNode = vm.FindNodeContainingTab(vm.RootNode, tabToFind);
        Assert.Null(foundNode);
    }
}
