using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class ImportsTabViewModel : TabItemViewModel
{
  public ImportsTabViewModel() : base("Imports")
  {
    Icon = "⬇"; // down arrow / import — U+2B07
  }
}
