using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Messages;

public class CoalesceGroupMessage
{
  public TabGroupNodeViewModel GroupNode { get; }

  public CoalesceGroupMessage(TabGroupNodeViewModel groupNode)
  {
    GroupNode = groupNode;
  }
}
