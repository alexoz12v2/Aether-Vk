using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class ModelTabViewModel : TabItemViewModel
{
  public ModelTabViewModel() : base("Model")
  {
    Icon = "⬡"; // hexagon / 3D object — U+2B21
  }
}
