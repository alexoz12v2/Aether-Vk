using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SettingsTabViewModel : TabItemViewModel
{
  public SettingsTabViewModel() : base("Settings")
  {
    Icon = "⚙"; // gear — U+2699
  }
}
