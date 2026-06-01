using CommunityToolkit.Mvvm.Messaging.Messages;

namespace AetherVk.Logic.Messages;

public class EarthObserverModeChangedMessage : ValueChangedMessage<bool>
{
  public EarthObserverModeChangedMessage(bool isEnabled)
    : base(isEnabled) { }
}
