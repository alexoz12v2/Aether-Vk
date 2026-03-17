using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using AetherVk.Logic.Messages;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public enum SplitOrientation
{
  Horizontal,
  Vertical,
}

/// <summary>
/// Base class for all layout nodes
/// </summary>
public abstract class LayoutNodeViewModelBase(SplitNodeViewModel? parent) : ViewModelBase
{
  /// <summary>
  /// Parent reference for splitting/merging logic
  /// </summary>
  public SplitNodeViewModel? Parent { get; set; } = parent;

  public bool IsRoot() => Parent is null;
}

/// <summary>
/// Represents a split in the space`
/// </summary>
public partial class SplitNodeViewModel : LayoutNodeViewModelBase
{
  [ObservableProperty]
  private LayoutNodeViewModelBase _firstChild;

  [ObservableProperty]
  private LayoutNodeViewModelBase _secondChild;

  [ObservableProperty]
  private SplitOrientation _orientation;

  [ObservableProperty]
  private double _splitRatio;

#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
  public SplitNodeViewModel(
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider adding the 'required' modifier or declaring as nullable.
    LayoutNodeViewModelBase firstChild,
    LayoutNodeViewModelBase secondChild,
    SplitOrientation orientation,
    double splitRatio = 0.5,
    SplitNodeViewModel? parent = null
  )
    : base(parent)
  {
    FirstChild = firstChild;
    SecondChild = secondChild;
    Orientation = orientation;
    SplitRatio = splitRatio;
  }
}

/// <summary>
/// Represents a "box" or leaf node that actually holds tabs
/// </summary>
public partial class TabGroupNodeViewModel : LayoutNodeViewModelBase
{
  public ObservableCollection<TabItemViewModel> Tabs { get; } = [];

  [ObservableProperty]
  private TabItemViewModel _selectedTab;

  public TabGroupNodeViewModel(TabItemViewModel defaultTab, SplitNodeViewModel? parent = null)
    : base(parent)
  {
    Tabs.Add(defaultTab);
    // TODO remove
    Tabs.Add(new TabItemViewModel("The Other"));
    // Don't need to generate PropertyChanged on construction
    _selectedTab = defaultTab;
  }
}

/// <summary>
/// Represents the actual content of a single tab
/// </summary>
public partial class TabItemViewModel(string title) : ViewModelBase
{
  // TODO: add properties and stuff
  [ObservableProperty]
  private string _title = title;
}

/// <summary>
/// Root manager for docking layout
/// </summary>
public partial class DockingManagerViewModel : ViewModelBase, IRecipient<TabDroppedMessage>
{
  [ObservableProperty]
  private LayoutNodeViewModelBase _rootNode;

  public DockingManagerViewModel()
    : base()
  {
    WeakReferenceMessenger.Default.Register(this);
    // TODO establish default from configuration (then passed to VM)
    var defaultTab = new TabItemViewModel(title: "Default View");
    // Don't need to generate PropertyChanged on construction
    _rootNode = new TabGroupNodeViewModel(defaultTab);
  }

  public void Receive(TabDroppedMessage message)
  {
    var draggedTab = message.DraggedTab;
    var targetNode = message.TargetNode;
    var zone = message.Zone;

    // 1. Find the curent owner of this tab
    var sourceNode = FindNodeContainingTab(RootNode, draggedTab);
    if (sourceNode is null)
      return;

    // 2. Safety: Prevent dragging the last tab of the entire system.
    // If the root is a TabGroup and it has only one tab, cancel
    if (RootNode is TabGroupNodeViewModel rootGroup && rootGroup.Tabs.Count <= 1)
      return;

    // 3. No-op check: Don't split a group using it's only tab (you'd split with an empty box)
    if (sourceNode == targetNode && sourceNode.Tabs.Count <= 1 && zone != DockZone.Center)
      return;

    // 4. Center op check: Do nothing if you are returning to starting position
    // Possible TODO: reorder tabs
    if (zone == DockZone.Center && targetNode.Tabs.Contains(draggedTab))
      return;

    // 5. Perform Mutation: Remove First, Then split. Ensure we don't prune `targetNode` our of existence
    RemoveTabAndCoalesce(draggedTab, sourceNode);
    if (zone == DockZone.Center)
    {
      targetNode.Tabs.Add(draggedTab);
      targetNode.SelectedTab = draggedTab;
    }
    else
    {
      SplitNodeAndInsertTab(targetNode, draggedTab, zone);
    }
  }

  private void RemoveTabAndCoalesce(TabItemViewModel tabToRemove, TabGroupNodeViewModel sourceNode)
  {
    sourceNode.Tabs.Remove(tabToRemove);

    // If the node has still tabs, we don't need to coalesce anything
    if (sourceNode.Tabs.Count > 0)
      return;

    // If it's a root and it's empty, keep it alive to avoid a null UI
    var parent = sourceNode.Parent;
    if (parent == null)
      return;

    // Coalesce: Replace parent (SplitNode) with the sibling of the empty node
    // The parent is a SplitNode, It needs to be replaced by its other child
    var sibling = (parent.FirstChild == sourceNode) ? parent.SecondChild : parent.FirstChild;
    var grandParent = parent.Parent;
    if (grandParent == null)
    {
      // parent was root. Sibling is the new root
      RootNode = sibling;
      sibling.Parent = null;
    }
    else
    {
      // Replace the parent in the grandparent with sibling
      if (grandParent.FirstChild == parent)
        grandParent.FirstChild = sibling;
      else
        grandParent.SecondChild = sibling;

      sibling.Parent = grandParent;
    }
  }

  private void SplitNodeAndInsertTab(
    TabGroupNodeViewModel target,
    TabItemViewModel tab,
    DockZone zone
  )
  {
    var newGroup = new TabGroupNodeViewModel(tab);
    var orientation =
      (zone == DockZone.Left || zone == DockZone.Right)
        ? SplitOrientation.Horizontal
        : SplitOrientation.Vertical;
    var parent = target.Parent;
    var firstChild =
      (zone == DockZone.Left || zone == DockZone.Top) ? (LayoutNodeViewModelBase)newGroup : target;
    var secondChild =
      (zone == DockZone.Left || zone == DockZone.Top) ? target : (LayoutNodeViewModelBase)newGroup;
    var splitNode = new SplitNodeViewModel(firstChild, secondChild, orientation);

    firstChild.Parent = splitNode;
    secondChild.Parent = splitNode;

    // Swap out target node in the tree with the new split node
    if (parent == null)
    {
      RootNode = splitNode;
    }
    else
    {
      if (parent.FirstChild == target)
        parent.FirstChild = splitNode;
      else
        parent.SecondChild = splitNode;
      splitNode.Parent = parent;
    }
  }

  private TabGroupNodeViewModel? FindNodeContainingTab(
    LayoutNodeViewModelBase ancestor,
    TabItemViewModel tabItem
  )
  {
    if (ancestor is TabGroupNodeViewModel tabGroup)
    {
      return tabGroup.Tabs.FirstOrDefault(t => t == tabItem) is not null ? tabGroup : null;
    }
    else if (ancestor is SplitNodeViewModel splitNode)
    {
      if (FindNodeContainingTab(splitNode.FirstChild, tabItem) is TabGroupNodeViewModel tg)
        return tg;
      else
        return FindNodeContainingTab(splitNode.SecondChild, tabItem);
    }
    else
      return null;
  }
}
