using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SpawnImageViewModel : ViewModelBase
{
  [ObservableProperty]
  private string _entityName = "";

  [ObservableProperty]
  private bool _isScreenSpace;

  [ObservableProperty]
  private float _width;

  [ObservableProperty]
  private float _height;

  public SpawnImageViewModel(string defaultName, float width, float height)
  {
    EntityName = defaultName;
    Width = width;
    Height = height;
  }
}
