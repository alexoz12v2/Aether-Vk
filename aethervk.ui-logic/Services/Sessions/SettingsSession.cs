namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the application settings managed by the Settings tab.
/// Exclusive because application settings are global.
/// </summary>
[ExclusiveSession]
public sealed class SettingsSession : ITabSession
{
  // Placeholder — future fields might include:
  //   public string Theme { get; set; } = "Dark";
  //   public bool ShowGrid { get; set; } = true;
}
