using System.Threading.Tasks;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Messages;

public enum DockZone
{
  Center,
  Left,
  Right,
  Top,
  Bottom,
}

// Used to hand a pure .NET Task over to the ViewModels
public class TabDragTaskMessage
{
  public TabItemViewModel DraggedTab { get; }
  public Task<string> DragTask { get; }
  public IDragSourceView View { get; }

  public TabDragTaskMessage(
    TabItemViewModel draggedTab,
    Task<string> dragTask,
    IDragSourceView view
  )
  {
    DraggedTab = draggedTab;
    DragTask = dragTask;
    View = view;
  }
}

// Update this to carry an IsCopy flag
public class TabDroppedMessage
{
  public TabItemViewModel DraggedTab { get; }
  public TabGroupNodeViewModel TargetNode { get; }
  public DockZone Zone { get; }
  public bool IsCopy { get; }

  public TabDroppedMessage(
    TabItemViewModel draggedTab,
    TabGroupNodeViewModel targetNode,
    DockZone zone,
    bool isCopy = false
  )
  {
    DraggedTab = draggedTab;
    TargetNode = targetNode;
    Zone = zone;
    IsCopy = isCopy;
  }
}
