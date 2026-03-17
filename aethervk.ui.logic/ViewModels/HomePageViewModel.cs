using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class HomePageViewModel : ViewModelBase
{
  [ObservableProperty]
  private string? name;
}
