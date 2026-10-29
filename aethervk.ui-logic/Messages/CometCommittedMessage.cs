using AetherVk.Logic.Services;

namespace AetherVk.Logic.Messages;

/// <summary>
/// A comet was committed. <see cref="Range"/> is the epoch range it was committed with (what the
/// engine holds): the Timeline shows it, never its own text boxes, which may hold a later edit.
/// </summary>
public sealed class CometCommittedMessage
{
  public CometCommittedMessage(TimeRange? range = null) => Range = range;

  public TimeRange? Range { get; }
}
