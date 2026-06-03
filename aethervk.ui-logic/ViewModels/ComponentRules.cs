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
    bool hasCameraOrCursor = entity.Components.Any(c =>
      c is CameraComponent || c is CursorComponent
    );
    bool hasPhysicalMesh = entity.Components.Any(c => c is PhysicalMeshComponent);
    bool hasComet = entity.Components.Any(c => c is CometComponent);

    // Default: only camera/cursor transforms are editable
    bool isPosEditable = hasCameraOrCursor;
    bool isRotEditable = hasCameraOrCursor;
    bool isScaleEditable = hasCameraOrCursor;
    string? posLockedReason = null;
    string? rotLockedReason = null;
    string? scaleLockedReason = null;

    if (hasPhysicalMesh)
    {
      // Rotation is always locked — governed by IAU Rotational Model
      isRotEditable = false;
      rotLockedReason = "Rotation is governed by the IAU Rotational Model under Physical Mesh.";

      if (hasComet)
      {
        // Comet position is kinematic (driven by SPK/ephemeris)
        isPosEditable = false;
        isScaleEditable = false;
        posLockedReason = "Position is driven by the ephemeris (SPK) data.";
        scaleLockedReason = "Scale is locked for comet entities.";
      }
      else
      {
        // Static mesh: position and scale are freely editable
        isPosEditable = true;
        isScaleEditable = true;
      }
    }

    var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
    if (transform != null)
    {
      transform.IsPositionEditable = isPosEditable;
      transform.IsRotationEditable = isRotEditable;
      transform.IsScaleEditable = isScaleEditable;
      transform.PositionLockedReason = posLockedReason;
      transform.RotationLockedReason = rotLockedReason;
      transform.ScaleLockedReason = scaleLockedReason;
    }

    var highRes = entity.Components.OfType<HighResTransformComponent>().FirstOrDefault();
    if (highRes != null)
    {
      highRes.IsEditable = isPosEditable;
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
  private readonly Services.BreadcrumbService? _breadcrumbService;

  public EpaRefreshRule(
    Services.NativeRuntimeService? runtimeService,
    Services.BreadcrumbService? breadcrumbService = null
  )
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
  }

  public void Apply(Entity entity)
  {
    var emitter = entity.Components.OfType<ParticleEmitterCirclesComponent>().FirstOrDefault();
    if (emitter != null && _runtimeService != null)
    {
      // Clean up previous event subscription to avoid duplicates if called multiple times
      emitter.PropertyChanged -= Emitter_PropertyChanged;
      emitter.PropertyChanged += Emitter_PropertyChanged;

      // Handle CollectionChanged for additions and removals
      if (
        emitter.Circles is System.Collections.Specialized.INotifyCollectionChanged notifyCollection
      )
      {
        notifyCollection.CollectionChanged -= Circles_CollectionChanged;
        notifyCollection.CollectionChanged += Circles_CollectionChanged;
      }

      // Unsubscribe from inner circles as well
      foreach (var circle in emitter.Circles)
      {
        circle.PropertyChanged -= Emitter_PropertyChanged;
        circle.PropertyChanged += Emitter_PropertyChanged;
      }

      // Perform initial sync
      _runtimeService.SyncEmissionCircleVisuals(entity.SceneId, entity.Id, emitter);

      void Emitter_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
      {
        _runtimeService.RecalculateJetPoints(entity.SceneId, entity.Id);
        _runtimeService.SyncEmissionCircleVisuals(entity.SceneId, entity.Id, emitter);
      }

      void Circles_CollectionChanged(
        object? sender,
        System.Collections.Specialized.NotifyCollectionChangedEventArgs e
      )
      {
        if (e.OldItems != null)
        {
          foreach (EmissionCircleItem oldItem in e.OldItems)
          {
            oldItem.PropertyChanged -= Emitter_PropertyChanged;
            if (oldItem.VisualEntityId != 0)
            {
              _runtimeService.RemoveEntity(entity.SceneId, oldItem.VisualEntityId);
              oldItem.VisualEntityId = 0;
            }
          }
        }
        if (e.NewItems != null)
        {
          foreach (EmissionCircleItem newItem in e.NewItems)
          {
            newItem.PropertyChanged += Emitter_PropertyChanged;
          }

          if (emitter.Circles.Count == 1)
          {
            _breadcrumbService?.ShowMessageAsync("Ready", "Simulation can be started now");
          }
        }

        _runtimeService.RecalculateJetPoints(entity.SceneId, entity.Id);
        _runtimeService.SyncEmissionCircleVisuals(entity.SceneId, entity.Id, emitter);
      }
    }
  }
}
