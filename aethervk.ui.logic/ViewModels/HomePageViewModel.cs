using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels
{
  public partial class HomePageViewModel : ObservableRecipient
  {
    [ObservableProperty]
    private string? name;
  }
}
