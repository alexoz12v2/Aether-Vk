using System.Collections.ObjectModel;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services; // Added for ConsoleService
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
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
  [ObservableProperty] private LayoutNodeViewModelBase _firstChild;

  [ObservableProperty] private LayoutNodeViewModelBase _secondChild;

  [ObservableProperty] private SplitOrientation _orientation;

  [ObservableProperty] private double _splitRatio;

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
public partial class TabGroupNodeViewModel : LayoutNodeViewModelBase,
  IRecipient<EntitySelectedMessage>
{
  public ObservableCollection<TabItemViewModel> Tabs { get; } = [];

  [ObservableProperty] private TabItemViewModel? _selectedTab;

  [ObservableProperty] private bool _hasTabs;

  public TabGroupNodeViewModel(TabItemViewModel defaultTab, SplitNodeViewModel? parent = null)
    : base(parent)
  {
    Tabs.Add(defaultTab);
    SelectedTab = defaultTab;
    HasTabs = true;

    Tabs.CollectionChanged += (s, e) => HasTabs = Tabs.Count > 0;
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);
  }

  public void Receive(EntitySelectedMessage message)
  {
    if (message.SelectedEntity != null)
    {
      var propTab = Tabs.OfType<PropertiesViewModel>().FirstOrDefault();
      if (propTab != null)
      {
        SelectedTab = propTab;
      }
    }
  }

  [RelayCommand]
  private void CloseTab(TabItemViewModel tab)
  {
    Tabs.Remove(tab);
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
    Tabs.Clear();
    SelectedTab = null;
    WeakReferenceMessenger.Default.Send(new CoalesceGroupMessage(this));
  }

  [RelayCommand]
  private void ChangeSelectedTab(string tabType)
  {
    if (SelectedTab == null)
      return;
    var index = Tabs.IndexOf(SelectedTab);
    if (index == -1)
      return;

    TabItemViewModel? newTab = null;
    // TODO BAD. This inhibits persistence. Introduce a service which registers live tabs
    switch (tabType)
    {
      case "UITestPanel":
        newTab = new UITestPanelViewModel();
        break;
      case "Console":
        var consoleService =
          ServiceLocator.Provider?.GetService(typeof(ConsoleService)) as ConsoleService;
        newTab =
          consoleService != null
            ? new ConsoleViewModel(consoleService)
            : new ConsoleViewModel(new ConsoleService());
        break;
      case "HorizonJpl":
        var horizonService =
          ServiceLocator.Provider?.GetService(typeof(HorizonJplService)) as HorizonJplService;
        if (horizonService != null)
          newTab = new HorizonJplViewModel(horizonService);
        break;
      case "Outline":
        var outlineVm =
          new OutlineViewModel(1, ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService, ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager);
        if (outlineVm != null)
          newTab = outlineVm;
        break;
      case "Properties":
        var propertiesVm =
          new PropertiesViewModel(1, ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager);
        if (propertiesVm != null)
          newTab = propertiesVm;
        break;
      case "DebugUI":
        newTab = new DebugUiViewModel();
        break;
      case "Timeline":
        var timelineService =
          ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
        if (timelineService != null)
          newTab = new TimelineViewModel(timelineService);
        break;
      case "Almanac":
        var almanacService =
          ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
        if (almanacService != null)
          newTab = new AlmanacExplorerViewModel(almanacService);
        break;
      case "Viewport3D":
        var viewportRuntimeService =
          ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
        if (viewportRuntimeService != null)
          newTab = new Viewport3DViewModel(viewportRuntimeService);
        break;
    }

    if (newTab != null)
    {
      Tabs[index] = newTab;
      SelectedTab = newTab;
    }
  }

  // TODO Remove all these stupid commands and leave just one which receives the view model and does proper dependency injection
  [RelayCommand]
  private void AddNewTab()
  {
    var newTab = new UITestPanelViewModel(); // Default to UITestPanelViewModel
    Tabs.Add(newTab);
    SelectedTab = newTab;
  }

  [RelayCommand]
  private void AddNewConsoleTab()
  {
    // Resolve ConsoleService from ServiceLocator so it shares the same instance across tabs
    var consoleService =
      ServiceLocator.Provider?.GetService(typeof(ConsoleService)) as ConsoleService;
    var newTab =
      consoleService != null
        ? new ConsoleViewModel(consoleService)
        : new ConsoleViewModel(new ConsoleService());
    Tabs.Add(newTab);
    SelectedTab = newTab;
  }

  [RelayCommand]
  private void AddNewHorizonJplTab()
  {
    var horizonService =
      ServiceLocator.Provider?.GetService(typeof(HorizonJplService)) as HorizonJplService;
    if (horizonService != null)
    {
      var newTab = new HorizonJplViewModel(horizonService);
      Tabs.Add(newTab);
      SelectedTab = newTab;
    }
  }

  [RelayCommand]
  private void AddNewOutlineTab()
  {
    var newTab = new OutlineViewModel(1, ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService, ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager);
    if (newTab != null && !Tabs.Contains(newTab))
    {
      Tabs.Add(newTab);
    }

    if (newTab != null) SelectedTab = newTab;
  }

  [RelayCommand]
  private void AddNewPropertiesTab()
  {
    var newTab =
      new PropertiesViewModel(1, ServiceLocator.Provider?.GetService(typeof(SceneStateManager)) as SceneStateManager);
    if (newTab != null && !Tabs.Contains(newTab))
    {
      Tabs.Add(newTab);
    }

    if (newTab != null) SelectedTab = newTab;
  }

  [RelayCommand]
  private void AddNewTimelineTab()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    if (runtimeService != null)
    {
      var newTab = new TimelineViewModel(runtimeService);
      Tabs.Add(newTab);
      SelectedTab = newTab;
    }
  }

  [RelayCommand]
  private void AddNewAlmanacTab()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    if (runtimeService != null)
    {
      var newTab = new AlmanacExplorerViewModel(runtimeService);
      Tabs.Add(newTab);
      SelectedTab = newTab;
    }
  }

  [RelayCommand]
  private void AddNewViewport3DTab()
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    if (runtimeService != null)
    {
      var newTab = new Viewport3DViewModel(runtimeService);
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
  [ObservableProperty] private string _title = title;

  [ObservableProperty] private string? _icon;

  [ObservableProperty] private bool _canClose = true;
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
  [ObservableProperty] private LayoutNodeViewModelBase _rootNode;

  public DockingManagerViewModel(LayoutNodeViewModelBase? rootNode = null)
    : base()
  {
    WeakReferenceMessenger.Default.Register<TabDroppedMessage>(this);
    WeakReferenceMessenger.Default.Register<TabDragTaskMessage>(this);
    WeakReferenceMessenger.Default.Register<CoalesceGroupMessage>(this);

    _rootNode = rootNode ?? CreateDefaultLayout();
  }

  private LayoutNodeViewModelBase CreateDefaultLayout()
  {
    // Default layout: Viewport3D on top, Console and others on bottom
    var viewportTab = new Viewport3DViewModel(
      (ServiceLocator.Provider!.GetService(typeof(NativeRuntimeService))! as NativeRuntimeService)!
    );
    var viewportGroup = new TabGroupNodeViewModel(viewportTab);

    var consoleTab = new ConsoleViewModel(
      (ServiceLocator.Provider.GetService(typeof(ConsoleService))! as ConsoleService)!
    );
    var bottomGroup = new TabGroupNodeViewModel(consoleTab);

    var split = new SplitNodeViewModel(viewportGroup, bottomGroup, SplitOrientation.Vertical, 0.7);
    viewportGroup.Parent = split;
    bottomGroup.Parent = split;

    var outlineTab =
      new OutlineViewModel(1, ServiceLocator.Provider.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService, ServiceLocator.Provider.GetService(typeof(SceneStateManager)) as SceneStateManager);
    var propertiesTab = new PropertiesViewModel(1, ServiceLocator.Provider.GetService(typeof(SceneStateManager)) as SceneStateManager);
    var rightGroup = new TabGroupNodeViewModel(outlineTab);
    // TODO When I try to move this tab below, the left tab gets nuked. Fix that
    rightGroup.Tabs.Add(propertiesTab);
    // TODO split ratio doesn't work
    var halfSplit = new SplitNodeViewModel(split, rightGroup, SplitOrientation.Horizontal, 0.7);
    split.Parent = halfSplit;
    rightGroup.Parent = halfSplit;

    return halfSplit;
  }

  // --- Track your Task safely from within the ViewModel ---
  public async void Receive(TabDragTaskMessage message)
  {
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
    var newGroup = new TabGroupNodeViewModel(tab);
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
