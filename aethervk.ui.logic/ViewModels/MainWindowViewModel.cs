using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
  [ObservableProperty]
  private DockingManagerViewModel _dockingManager = new();
}
