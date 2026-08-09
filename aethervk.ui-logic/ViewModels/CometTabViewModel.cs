using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class CometTabViewModel : TabItemViewModel
{
  public CometTabViewModel() : base("Comet")
  {
    Icon = "☄"; // comet — U+2604
  }
}
