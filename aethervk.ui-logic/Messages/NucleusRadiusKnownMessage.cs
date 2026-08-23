namespace AetherVk.Logic.Messages;

/// <summary>
/// Sent when the comet nucleus bounding-sphere radius becomes known,
/// either populated from Horizon small-body constants during commit
/// or entered manually in the Model tab.
/// Triggers <c>AddJetCommand.NotifyCanExecuteChanged</c> in <c>ModelTabViewModel</c>.
/// </summary>
public class NucleusRadiusKnownMessage
{
  /// <summary>The radius that was just set (km). Always &gt; 0.</summary>
  public float RadiusKm { get; init; }
}
