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

  /// <summary>
  /// When true, the entity position is driven by almanac SPK data (AlmanacPlanet component).
  /// When false (default), the user sets an explicit position.
  /// </summary>
  [ObservableProperty]
  private bool _isKinematic;

  /// <summary>
  /// NAIF SPK ID for kinematic mode (e.g. 1000012 for a comet).
  /// Only used when IsKinematic is true.
  /// </summary>
  [ObservableProperty]
  private int _spkId;

  /// <summary>Position fields should only be editable when NOT in kinematic mode.</summary>
  public bool CanSetPosition => !IsKinematic;

  partial void OnIsKinematicChanged(bool value) => OnPropertyChanged(nameof(CanSetPosition));

  public SpawnMeshViewModel(string defaultName)
  {
    EntityName = defaultName;
  }
}
