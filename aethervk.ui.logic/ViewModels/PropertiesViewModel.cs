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
  private readonly NativeRuntimeService? _runtimeService;
  private readonly BreadcrumbService? _breadcrumbService;

  [ObservableProperty]
  private Entity? _selectedEntity;

  [ObservableProperty]
  private bool _isFollowingEntity;

  [ObservableProperty]
  private ulong _currentSceneId;

  public PropertiesViewModel(ulong sceneId, SceneStateManager stateManager, NativeRuntimeService? runtimeService = null, BreadcrumbService? breadcrumbService = null)
    : base("Properties")
  {
    System.Console.WriteLine($"[PropertiesViewModel] Constructor called for Scene {sceneId}");
    _stateManager = stateManager;
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    CurrentSceneId = sceneId;
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);

    // Initialize with current selection if any
    var state = _stateManager.GetOrCreateScene(CurrentSceneId);
    SelectedEntity = state.SelectedEntity;
  }

  public void Receive(EntitySelectedMessage message)
  {
    System.Console.WriteLine($"[PropertiesViewModel] Received selection: {message.SelectedEntity?.Name ?? "null"}");
    
    _breadcrumbService?.ShowMessageAsync("Properties", $"Received selection: {message.SelectedEntity?.Name ?? "null"}");
    SelectedEntity = message.SelectedEntity;
    IsFollowingEntity = false;

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
        _runtimeService?.RefreshBvhNodes(CurrentSceneId, SelectedEntity.Id, comet);
      }
    }
  }

  [RelayCommand]
  private void SnapToSelectedEntity(CameraActionParams p)
  {
    if (SelectedEntity != null)
    {
      _runtimeService?.SnapToEntity(p.SceneId, p.CameraEntityId, SelectedEntity.Id);
    }
  }

  [RelayCommand]
  private void FollowSelectedEntity(CameraActionParams p)
  {
    if (SelectedEntity != null)
    {
      _runtimeService?.FollowEntity(p.SceneId, p.CameraEntityId, SelectedEntity.Id);
    }
  }

  [RelayCommand]
  private void UnfollowSelectedEntity(CameraActionParams p)
  {
    _runtimeService?.UnfollowEntity(p.SceneId, p.CameraEntityId);
  }

  [RelayCommand]
  private void ToggleAddJetMode()
  {
    WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.ToggleAddJetModeMessage());

    _breadcrumbService?.ShowMessageAsync(
      "Add Jet Mode",
      "Hold Shift and Right Click on the comet to add a jet at the intersection point."
    );
  }

  [RelayCommand]
  private void DeleteSelectedEntity()
  {
    if (SelectedEntity != null && SelectedEntity.IsMeasurement)
    {
      var name = SelectedEntity.Name;
      _runtimeService?.RemoveEntity(CurrentSceneId, SelectedEntity.Id);

      _breadcrumbService?.ShowMessageAsync("Entity Deleted", $"Deleted measurement: {name}");

      // Deselect
      WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(null));
    }
  }
}
