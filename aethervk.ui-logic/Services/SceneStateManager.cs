using System.Collections.Generic;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Services;

public partial class SceneStateManager : ObservableObject
{
  private readonly Dictionary<ulong, SceneState> _scenes = new();

  public IEnumerable<SceneState> AllScenes => _scenes.Values;

  public SceneState GetOrCreateScene(ulong sceneId)
  {
    if (!_scenes.TryGetValue(sceneId, out var state))
    {
      state = new SceneState(sceneId);
      _scenes[sceneId] = state;
    }
    return state;
  }

  public void RemoveScene(ulong sceneId)
  {
    _scenes.Remove(sceneId);
  }

  public void Clear()
  {
    _scenes.Clear();
  }
}
