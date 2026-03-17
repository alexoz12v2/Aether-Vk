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

/// <summary>
/// Sent when a tab is dropped successfully
/// </summary>
public class TabDroppedMessage(
  TabItemViewModel draggedTab,
  TabGroupNodeViewModel targetNode,
  DockZone zone
)
{
  public TabItemViewModel DraggedTab { get; } = draggedTab;
  public TabGroupNodeViewModel TargetNode { get; } = targetNode;
  public DockZone Zone { get; } = zone;
}
