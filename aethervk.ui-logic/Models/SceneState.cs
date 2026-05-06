using System.Collections.Generic;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

public partial class SceneState : ObservableObject
{
  public ulong SceneId { get; }
  public ObservableCollection<Entity> RootEntities { get; } = new();
  public Dictionary<ulong, Entity> EntityMap { get; } = new();

  [ObservableProperty]
  private Entity? _selectedEntity;

  public SceneState(ulong sceneId)
  {
    SceneId = sceneId;
  }

  public void Clear()
  {
    RootEntities.Clear();
    EntityMap.Clear();
    SelectedEntity = null;
  }
}
