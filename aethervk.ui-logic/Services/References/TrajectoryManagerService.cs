using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Services;

public class TrajectoryManagerService
{
  private readonly NativeRuntimeService _runtimeService;
  private readonly SceneStateManager _sceneStateManager;
  private readonly Dictionary<int, ulong> _spkTrajectoryEntities = new();

  public TrajectoryManagerService(
    NativeRuntimeService runtimeService,
    SceneStateManager sceneStateManager
  )
  {
    _runtimeService = runtimeService;
    _sceneStateManager = sceneStateManager;
  }

  public async Task EnsureTrajectoryForSpkAsync(
    ulong sceneId,
    int spkId,
    double startTai,
    double endTai,
    double stepDays
  )
  {
    if (!_spkTrajectoryEntities.TryGetValue(spkId, out ulong entityId))
    {
      // Spawn new entity
      var entity = _runtimeService.SpawnEntity(sceneId, $"Trajectory_SPK_{spkId}");
      if (entity == null)
        return;
      entityId = entity.Id;

      var root = _runtimeService.GetEntityByName(sceneId, "root");
      if (root != null)
      {
        _runtimeService.SetParent(sceneId, entityId, root.Id);
      }

      // Identity transform
      _runtimeService.AddTransformComponent(sceneId, entityId, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1);
      _spkTrajectoryEntities[spkId] = entityId;
    }

    await _runtimeService.UpdateTrajectoryForSpkAsync(
      sceneId,
      entityId,
      spkId,
      startTai,
      endTai,
      stepDays
    );
  }

  public async Task UpdateAllTrajectoriesAsync(
    ulong sceneId,
    double startTai,
    double endTai,
    double stepDays
  )
  {
    foreach (var kvp in _spkTrajectoryEntities)
    {
      await _runtimeService.UpdateTrajectoryForSpkAsync(
        sceneId,
        kvp.Value,
        kvp.Key,
        startTai,
        endTai,
        stepDays
      );
    }
  }
}
