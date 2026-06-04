using System;
using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

public abstract class NativeComponent : ObservableObject, IComponent
{
  public abstract string Name { get; }

  public ulong SceneId { get; private set; }
  public ulong EntityId { get; private set; }
  protected IntPtr SimulationContext { get; private set; }

  protected bool IsSyncingFromNative { get; private set; }

  public void BindToNative(IntPtr context, ulong sceneId, ulong entityId)
  {
    SimulationContext = context;
    SceneId = sceneId;
    EntityId = entityId;
  }

  public void PullFromNative()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    IsSyncingFromNative = true;
    try
    {
      PullFromNativeImpl();
    }
    finally
    {
      IsSyncingFromNative = false;
    }
  }

  /// <summary>
  /// Explicitly pushes current state to native, bypassing property-change detection.
  /// Use when field values may match defaults (so OnPropertyChanged won't fire).
  /// </summary>
  public void ForceNativePush()
  {
    if (SimulationContext != IntPtr.Zero && SceneId != 0 && EntityId != 0)
      PushToNativeImpl();
  }

  protected abstract void PullFromNativeImpl();
  protected abstract void PushToNativeImpl();

  protected virtual bool ShouldPushToNative(string? propertyName) => true;

  protected override void OnPropertyChanged(PropertyChangedEventArgs e)
  {
    base.OnPropertyChanged(e);

    if (!IsSyncingFromNative && SimulationContext != IntPtr.Zero && SceneId != 0 && EntityId != 0)
    {
      if (ShouldPushToNative(e.PropertyName))
      {
        PushToNativeImpl();
      }
    }
  }
}
