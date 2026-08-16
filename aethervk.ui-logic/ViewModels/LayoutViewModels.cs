using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services; // Added for ConsoleService
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Used in <see cref="ITabFactory" />
/// </summary>
public record TabDescriptor(string Header, Type TabType, bool IsStateful);


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
public partial class TabGroupNodeViewModel
  : LayoutNodeViewModelBase
{
  public ObservableCollection<TabItemViewModel> Tabs { get; } = [];

  [ObservableProperty]
  private TabItemViewModel? _selectedTab;

  [ObservableProperty]
  private bool _hasTabs;

  private readonly ITabFactory _tabFactory;

  public IReadOnlyList<TabDescriptor> AvailableTabs => _tabFactory.AvailableTabs;

  public TabGroupNodeViewModel(
    TabItemViewModel defaultTab,
    ITabFactory tabFactory,
    SplitNodeViewModel? parent = null
  )
    : base(parent)
  {
    _tabFactory = tabFactory;
    Tabs.Add(defaultTab);
    SelectedTab = defaultTab;
    HasTabs = true;

    Tabs.CollectionChanged += (s, e) => HasTabs = Tabs.Count > 0;
  }

  [RelayCommand]
  private void CloseTab(TabItemViewModel tab)
  {
    Tabs.Remove(tab);
    if (tab is IDisposable disposable)
    {
      disposable.Dispose();
    }
    if (Tabs.Count > 0 && SelectedTab == tab)
    {
      SelectedTab = Tabs[Tabs.Count - 1];
    }
    else if (Tabs.Count == 0)
    {
      SelectedTab = null;
      WeakReferenceMessenger.Default.Send(new CoalesceGroupMessage(this));
    }
  }

  [RelayCommand]
  private void CloseAllTabs()
  {
    foreach (var tab in Tabs)
    {
      if (tab is IDisposable disposable)
      {
        disposable.Dispose();
      }
    }
    Tabs.Clear();
    SelectedTab = null;
    WeakReferenceMessenger.Default.Send(new CoalesceGroupMessage(this));
  }

  [RelayCommand]
  private void ChangeSelectedTab(Type tabType)
  {
    if (SelectedTab == null)
      return;
    var index = Tabs.IndexOf(SelectedTab);
    if (index == -1)
      return;

    TabItemViewModel? newTab = _tabFactory.CreateTab(tabType) as TabItemViewModel;

    if (newTab != null)
    {
      Tabs[index] = newTab;
      SelectedTab = newTab;
    }
  }

  [RelayCommand]
  private void AddNewTab(Type tabType)
  {
    var newTab = _tabFactory.CreateTab(tabType) as TabItemViewModel;
    if (newTab != null && !Tabs.Contains(newTab))
    {
      Tabs.Add(newTab);
      SelectedTab = newTab;
    }
  }
}

/// <summary>
/// Represents the actual content of a single tab
/// </summary>
public partial class TabItemViewModel(string title) : ViewModelBase
{
  [ObservableProperty]
  private string _title = title;

  [ObservableProperty]
  private string? _icon;

  [ObservableProperty]
  private bool _canClose = true;
}

/// <summary>
/// Root manager for docking layout
/// </summary>
public partial class DockingManagerViewModel
  : ViewModelBase,
    IRecipient<TabDroppedMessage>,
    IRecipient<TabDragTaskMessage>,
    IRecipient<CoalesceGroupMessage>
{
  [ObservableProperty]
  private LayoutNodeViewModelBase _rootNode;

  private readonly ITabFactory _tabFactory;

  public DockingManagerViewModel(
    ITabFactory tabFactory,
    LayoutNodeViewModelBase? rootNode = null
  )
    : base()
  {
    _tabFactory = tabFactory;
    WeakReferenceMessenger.Default.Register<TabDroppedMessage>(this);
    WeakReferenceMessenger.Default.Register<TabDragTaskMessage>(this);
    WeakReferenceMessenger.Default.Register<CoalesceGroupMessage>(this);

    _rootNode = rootNode ?? CreateDefaultLayout();
  }

  // admittedly, the fact that the starting layout is read from code in logic assembly sucks.
  // TODO: file based configuration at the app layer (like input registry, which now is code based
  // in App.axaml.cs)
  private LayoutNodeViewModelBase CreateDefaultLayout()
  {
    // -- Tab creation  --
    // center, most of space, 60% width centered, 80% height top
    var viewportTab = _tabFactory.CreateTab<Viewport3DViewModel>();
    var centerGroup = new TabGroupNodeViewModel(viewportTab!, _tabFactory);

    // left (27%)
    var settingsTab = _tabFactory.CreateTab<SettingsTabViewModel>();
    var cometTab = _tabFactory.CreateTab<CometTabViewModel>();
    var modelTab = _tabFactory.CreateTab<ModelTabViewModel>();

    var leftGroup = new TabGroupNodeViewModel(settingsTab!, _tabFactory);
    leftGroup.Tabs.Add(cometTab!);
    leftGroup.Tabs.Add(modelTab!);

    // right (13%) (17.8% when referring to viewport+imports = 73%)
    var importsTab = _tabFactory.CreateTab<ImportsTabViewModel>();
    var rightGroup = new TabGroupNodeViewModel(importsTab!, _tabFactory);

    // bottom (20% height)
    var timelineTab = _tabFactory.CreateTab<TimelineTabViewModel>();
    var bottomGroup = new TabGroupNodeViewModel(timelineTab!, _tabFactory);

    // viewport - imports horizontal split
    var viewportAndImportsGroup = new SplitNodeViewModel(
        centerGroup,
        rightGroup,
        SplitOrientation.Horizontal,
        0.822);
    centerGroup.Parent = viewportAndImportsGroup;
    rightGroup.Parent = viewportAndImportsGroup;

    // (...) - timeline vertical split
    var allButSettingsGroup = new SplitNodeViewModel(
        viewportAndImportsGroup,
        bottomGroup,
        SplitOrientation.Vertical,
        0.80);
    viewportAndImportsGroup.Parent = allButSettingsGroup;
    bottomGroup.Parent = allButSettingsGroup;

    // settings - (...) Horizontal split
    var settingsSplit = new SplitNodeViewModel(
        leftGroup,
        allButSettingsGroup,
        SplitOrientation.Horizontal,
        0.27);
    leftGroup.Parent = settingsSplit;
    allButSettingsGroup.Parent = settingsSplit;

    return settingsSplit;
  }

  // --- Track your Task safely from within the ViewModel ---
  public async void Receive(TabDragTaskMessage message)
  {
    // _audioService.PlayGrabAsync();
    string finalAction = await message.DragTask;

    if (finalAction == "None")
    {
      // The drag ended OUTSIDE the application window!
      // You can implement your float-to-new-window logic right here.
    }
  }

  // --- Fix the TabDroppedMessage ---
  public void Receive(TabDroppedMessage message)
  {
    // _audioService.PlayDropAsync();
    var draggedTab = message.DraggedTab;
    var targetNode = message.TargetNode;
    var zone = message.Zone;

    var sourceNode = FindNodeContainingTab(RootNode, draggedTab);

    // 5. Duplicate Crash Fix: When copying, we must create a NEW reference.
    TabItemViewModel tabToInsert = draggedTab;
    if (message.IsCopy)
    {
      // Generate a new reference. You may want to add a `.Clone()` method to
      // TabItemViewModel later to handle inner state deep-copying.
      tabToInsert = new TabItemViewModel(draggedTab.Title + " (Copy)");
    }

    if (sourceNode is null)
    {
      if (zone == DockZone.Center)
      {
        if (!targetNode.Tabs.Contains(tabToInsert))
          targetNode.Tabs.Add(tabToInsert);
        targetNode.SelectedTab = tabToInsert;
      }
      else
        SplitNodeAndInsertTab(targetNode, tabToInsert, zone);

      return;
    }

    // Return if it's the very last tab on screen (UNLESS we are explicitly copying it)
    if (!message.IsCopy && RootNode is TabGroupNodeViewModel rootGroup && rootGroup.Tabs.Count <= 1)
      return;

    // Don't split a 1-tab group with nothing (UNLESS we are copying it)
    if (
      !message.IsCopy
      && sourceNode == targetNode
      && sourceNode.Tabs.Count <= 1
      && zone != DockZone.Center
    )
      return;

    // Do nothing if returning to starting position
    if (!message.IsCopy && zone == DockZone.Center && targetNode.Tabs.Contains(draggedTab))
      return;

    // Mutate Old List
    if (!message.IsCopy)
    {
      sourceNode.Tabs.Remove(draggedTab);
      if (sourceNode.Tabs.Count > 0 && sourceNode.SelectedTab == draggedTab)
      {
        sourceNode.SelectedTab = sourceNode.Tabs[sourceNode.Tabs.Count - 1];
      }
      else if (sourceNode.Tabs.Count == 0)
      {
        sourceNode.SelectedTab = null;
        RemoveTabAndCoalesce(draggedTab, sourceNode);
      }
    }

    // Apply into New List
    if (zone == DockZone.Center)
    {
      if (!targetNode.Tabs.Contains(tabToInsert))
        targetNode.Tabs.Add(tabToInsert);
      targetNode.SelectedTab = tabToInsert;
    }
    else
      SplitNodeAndInsertTab(targetNode, tabToInsert, zone);
  }

  public void Receive(CoalesceGroupMessage message)
  {
    System.Diagnostics.Debug.WriteLine(
      $"[DockingManager] Coalesce requested for node with parent: {message.GroupNode.Parent != null}"
    );
    RemoveTabAndCoalesce(null, message.GroupNode);
  }

  private void RemoveTabAndCoalesce(TabItemViewModel? tabToRemove, TabGroupNodeViewModel sourceNode)
  {
    // If it's a root and it's empty, keep it alive to avoid a null UI
    var parent = sourceNode.Parent;
    if (parent == null)
    {
      System.Diagnostics.Debug.WriteLine("[DockingManager] Parent is null, keeping root alive");
      return;
    }

    System.Diagnostics.Debug.WriteLine("[DockingManager] Replacing parent with sibling");
    // Coalesce: Replace parent (SplitNode) with the sibling of the empty node
    // The parent is a SplitNode, It needs to be replaced by its other child
    var sibling = (parent.FirstChild == sourceNode) ? parent.SecondChild : parent.FirstChild;
    ReplaceNode(parent, sibling);
  }

  // --- Fix the Tree-Breaking Bug ---
  private void SplitNodeAndInsertTab(
    TabGroupNodeViewModel target,
    TabItemViewModel tab,
    DockZone zone
  )
  {
    var newGroup = new TabGroupNodeViewModel(tab, _tabFactory);
    var orientation =
      (zone == DockZone.Left || zone == DockZone.Right)
        ? SplitOrientation.Horizontal
        : SplitOrientation.Vertical;

    var firstChild =
      (zone == DockZone.Left || zone == DockZone.Top) ? (LayoutNodeViewModelBase)newGroup : target;
    var secondChild =
      (zone == DockZone.Left || zone == DockZone.Top) ? target : (LayoutNodeViewModelBase)newGroup;
    var splitNode = new SplitNodeViewModel(firstChild, secondChild, orientation);

    // CRITICAL FIX: You MUST execute ReplaceNode before setting the parent properties.
    // If you do it the other way around, ReplaceNode sees the new node as its own parent.
    ReplaceNode(target, splitNode);

    // Now it is safe to assign the relationships downward
    firstChild.Parent = splitNode;
    secondChild.Parent = splitNode;
  }

  private void ReplaceNode(LayoutNodeViewModelBase oldNode, LayoutNodeViewModelBase newNode)
  {
    var parent = oldNode.Parent;
    if (parent == null)
    {
      // oldNode was the root, newNode becomes the new root
      RootNode = newNode;
      newNode.Parent = null;
    }
    else
    {
      // Replace oldNode with newNode in the parent's children
      if (parent.FirstChild == oldNode)
        parent.FirstChild = newNode;
      else if (parent.SecondChild == oldNode)
        parent.SecondChild = newNode;

      newNode.Parent = parent;
    }
  }

  public TabGroupNodeViewModel? FindNodeContainingTab(
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
