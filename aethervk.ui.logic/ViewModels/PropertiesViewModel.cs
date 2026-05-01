using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class PropertiesViewModel : TabItemViewModel, IRecipient<EntitySelectedMessage>
{
  private readonly SceneStateManager _stateManager;

  [ObservableProperty]
  private Entity? _selectedEntity;

  [ObservableProperty]
  private bool _isFollowingEntity;

  [ObservableProperty]
  private ulong _currentSceneId;

  public PropertiesViewModel(ulong sceneId, SceneStateManager stateManager)
    : base("Properties")
  {
    _stateManager = stateManager;
    CurrentSceneId = sceneId;
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);

    // Initialize with current selection if any
    var state = _stateManager.GetOrCreateScene(CurrentSceneId);
    _selectedEntity = state.SelectedEntity;
  }

  public void Receive(EntitySelectedMessage message)
  {
    SelectedEntity = message.SelectedEntity;

    if (SelectedEntity != null)
    {
      var transform = SelectedEntity.Components.OfType<TransformComponent>().FirstOrDefault();
      if (transform != null)
      {
        bool hasCameraOrCursor = SelectedEntity.Components.Any(c =>
          c is CameraComponent || c is CursorComponent
        );
        transform.IsEditable = hasCameraOrCursor;
      }

      var comet = SelectedEntity.Components.OfType<CometComponent>().FirstOrDefault();
      if (comet != null)
      {
        var runtimeService =
          ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
        runtimeService?.RefreshBvhNodes(CurrentSceneId, SelectedEntity.Id, comet);
      }
    }
  }

  [RelayCommand]
  private void SnapToSelectedEntity(CameraActionParams p)
  {
    if (SelectedEntity != null)
    {
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      runtimeService?.SnapToEntity(p.SceneId, p.CameraEntityId, SelectedEntity.Id);
    }
  }

  [RelayCommand]
  private void FollowSelectedEntity(CameraActionParams p)
  {
    if (SelectedEntity != null)
    {
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      runtimeService?.FollowEntity(p.SceneId, p.CameraEntityId, SelectedEntity.Id);
    }
  }

  [RelayCommand]
  private void UnfollowSelectedEntity(CameraActionParams p)
  {
    var runtimeService =
      ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
    runtimeService?.UnfollowEntity(p.SceneId, p.CameraEntityId);
  }

  [RelayCommand]
  private void ToggleAddJetMode()
  {
    WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.ToggleAddJetModeMessage());

    var breadcrumb =
      ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
    breadcrumb?.ShowMessageAsync(
      "Add Jet Mode",
      "Hold Shift and Right Click on the comet to add a jet at the intersection point."
    );
  }

  [RelayCommand]
  private void DeleteSelectedEntity()
  {
    if (SelectedEntity != null && SelectedEntity.IsMeasurement)
    {
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      var name = SelectedEntity.Name;
      runtimeService?.RemoveEntity(CurrentSceneId, SelectedEntity.Id);

      var breadcrumb =
        ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
      breadcrumb?.ShowMessageAsync("Entity Deleted", $"Deleted measurement: {name}");

      // Deselect
      WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(null));
    }
  }
}
