namespace AetherVk.Logic.Messages;

using CommunityToolkit.Mvvm.Messaging.Messages;

public sealed class OpenSnapObserverDialogMessage
  : AsyncRequestMessage<(double X, double Y, double Z)?> { }
