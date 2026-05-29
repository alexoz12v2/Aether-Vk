using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Defines a composable rule that mutates component UI states or reacts to the entity's composition.
/// </summary>
public interface IComponentRule
{
  void Apply(Entity entity);
}

/// <summary>
/// A rule that ensures the TransformComponent is only editable if the entity contains a Camera or Cursor.
/// </summary>
public class TransformEditableRule : IComponentRule
{
  public void Apply(Entity entity)
  {
    var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
    if (transform != null)
    {
      bool hasCameraOrCursor = entity.Components.Any(c =>
        c is CameraComponent || c is CursorComponent
      );
      transform.IsEditable = hasCameraOrCursor;
    }
  }
}

/// <summary>
/// A rule that triggers a BVH tree refresh for Comet components when selected.
/// </summary>
public class CometBvhRefreshRule : IComponentRule
{
  private readonly Services.NativeRuntimeService? _runtimeService;

  public CometBvhRefreshRule(Services.NativeRuntimeService? runtimeService)
  {
    _runtimeService = runtimeService;
  }

  public void Apply(Entity entity)
  {
    var comet = entity.Components.OfType<CometComponent>().FirstOrDefault();
    if (comet != null && _runtimeService != null)
    {
      _runtimeService.RefreshBvhNodes(entity.SceneId, entity.Id, comet);
    }
  }
}

/// <summary>
/// A rule that triggers EPA recomputation for Particle Emitters when selected or when properties change.
/// </summary>
public class EpaRefreshRule : IComponentRule
{
  private readonly Services.NativeRuntimeService? _runtimeService;

  public EpaRefreshRule(Services.NativeRuntimeService? runtimeService)
  {
    _runtimeService = runtimeService;
  }

  public void Apply(Entity entity)
  {
    var emitter = entity.Components.OfType<ParticleEmitterCirclesComponent>().FirstOrDefault();
    if (emitter != null && _runtimeService != null)
    {
      // Clean up previous event subscription to avoid duplicates if called multiple times
      emitter.PropertyChanged -= Emitter_PropertyChanged;
      emitter.PropertyChanged += Emitter_PropertyChanged;

      // Unsubscribe from inner circles as well
      foreach (var circle in emitter.Circles)
      {
        circle.PropertyChanged -= Emitter_PropertyChanged;
        circle.PropertyChanged += Emitter_PropertyChanged;
      }

      void Emitter_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
      {
        // Actually we only want to do this when paused.
        // But NativeRuntimeService doesn't know if it's paused here easily, so we just trigger it.
        _runtimeService.RecalculateJetPoints(entity.SceneId, entity.Id);
      }
    }
  }
}
