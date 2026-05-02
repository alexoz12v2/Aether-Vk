using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SpawnMeshViewModel : ViewModelBase
{
  [ObservableProperty]
  private string _entityName = "";

  [ObservableProperty]
  private float _posX;

  [ObservableProperty]
  private float _posY;

  [ObservableProperty]
  private float _posZ;

  public SpawnMeshViewModel(string defaultName)
  {
    EntityName = defaultName;
  }
}
