using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class UnknownComponentViewModel : ObservableObject
{
  [ObservableProperty]
  private string _componentName;

  public UnknownComponentViewModel(string componentName)
  {
    _componentName = componentName;
  }
}
