namespace AetherVk.Logic.Messages;

using CommunityToolkit.Mvvm.Messaging.Messages;

/// <summary>
/// Sent by the radial context menu (Alt+S) to request that the main window open
/// the Spawn Comet dialog, regardless of which sub-panel triggered it.
/// </summary>
public sealed class OpenSpawnCometDialogMessage : AsyncRequestMessage<ulong>
{
  public ulong? PreselectedModelId { get; }

  public OpenSpawnCometDialogMessage(ulong? preselectedModelId = null)
  {
    PreselectedModelId = preselectedModelId;
  }
}
