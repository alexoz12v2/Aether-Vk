using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Input;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class PropertiesViewModel
  : TabItemViewModel,
    IRecipient<EntitySelectedMessage>,
    IActionHandler
{
  private readonly SceneStateManager _stateManager;
  private readonly NativeRuntimeService? _runtimeService;
  private readonly BreadcrumbService? _breadcrumbService;

  public OperatorStack OperatorStack { get; }

  public Entity? SelectedEntity => _stateManager.GetOrCreateScene(CurrentSceneId).SelectedEntity;

  [ObservableProperty]
  private bool _isFollowingEntity;

  [ObservableProperty]
  private ulong _currentSceneId;

  [ObservableProperty]
  private bool _areAllExpanded;

  [ObservableProperty]
  private bool _isFlyoutMenuOpen;

  public ObservableCollection<object> PropertiesExpanders { get; } = new();

  private readonly System.Collections.Generic.List<IComponentRule> _componentRules = new();

  public TimelineService Timeline { get; }

  public PropertiesViewModel(
    ulong sceneId,
    SceneStateManager stateManager,
    TimelineService timelineService,
    NativeRuntimeService? runtimeService = null,
    BreadcrumbService? breadcrumbService = null
  )
    : base("Properties")
  {
    System.Console.WriteLine($"[PropertiesViewModel] Constructor called for Scene {sceneId}");
    _stateManager = stateManager;
    Timeline = timelineService;
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    CurrentSceneId = sceneId;
    OperatorStack = new OperatorStack(new PropertiesBaseOperator(this));

    // Register composable rules
    _componentRules.Add(new TransformEditableRule());
    _componentRules.Add(new CometBvhRefreshRule(_runtimeService));
    _componentRules.Add(new EpaRefreshRule(_runtimeService));

    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);
  }

  public bool ProcessAction(AppAction action, bool isPressed)
  {
    return OperatorStack.ProcessAction(action, isPressed);
  }

  public bool ProcessPointerDelta(float dx, float dy) => OperatorStack.ProcessPointerDelta(dx, dy);

  public bool ProcessPointerWheel(float deltaY) => OperatorStack.ProcessPointerWheel(deltaY);

  public void Receive(EntitySelectedMessage message)
  {
    System.Console.WriteLine(
      $"[PropertiesViewModel] Received selection: {message.SelectedEntity?.Name ?? "null"}"
    );

    _breadcrumbService?.ShowMessageAsync(
      "Properties",
      $"Received selection: {message.SelectedEntity?.Name ?? "null"}"
    );
    OnPropertyChanged(nameof(SelectedEntity));
    IsFollowingEntity = false;

    PropertiesExpanders.Clear();

    if (SelectedEntity != null && _runtimeService != null)
    {
      var componentNames = _runtimeService.GetEntityComponentNames(
        CurrentSceneId,
        SelectedEntity.Id
      );
      
      // If it's a Screen Space Billboard, show the billboard-specific properties
      var ssBillboardComp = SelectedEntity.Components.OfType<ScreenSpaceBillboardComponent>().FirstOrDefault();
      if (ssBillboardComp != null)
      {
          PropertiesExpanders.Add(ssBillboardComp);
      }

      bool hasCamera = componentNames.Any(n => n.EndsWith("CameraComponent"));

      foreach (var name in componentNames)
      {
        if (name.EndsWith("TransformComponent"))
        {
          var comp = SelectedEntity.Components.OfType<TransformComponent>().FirstOrDefault();
          if (comp != null)
          {
            comp.IsEditable = hasCamera;
            PropertiesExpanders.Add(comp);
          }
        }
        else if (name.EndsWith("CameraComponent"))
        {
          var comp = SelectedEntity.Components.OfType<CameraComponent>().FirstOrDefault();
          if (comp != null)
          {
            PropertiesExpanders.Add(comp);
          }
        }
        else if (
          name.EndsWith("CometComponent")
          || name.EndsWith("PlanetComponent")
          || name.EndsWith("SunComponent")
          || name.EndsWith("CursorComponent")
          || name.EndsWith("GridComponent")
        )
        {
          // For existing complex components, try to find them in the old heuristic list
          var comp = SelectedEntity.Components.FirstOrDefault(c =>
            c.GetType().Name == name.Split(new[] { "::" }, System.StringSplitOptions.None).Last()
          );
          if (comp != null)
          {
            PropertiesExpanders.Add(comp);
          }
          else
          {
            // Provide fallback if not yet supported
            PropertiesExpanders.Add(
              new UnknownComponentViewModel(
                name.Split(new[] { "::" }, System.StringSplitOptions.None).Last()
              )
            );
          }
        }
        else if (name.Contains("ParticleEmitterCircles"))
        {
          // Build a fresh observable model for the circle-emitter component.
          // In a future sprint this could be populated from native data.
          var existing = SelectedEntity.Components
            .OfType<ParticleEmitterCirclesComponent>()
            .FirstOrDefault();
          if (existing == null)
          {
            existing = new ParticleEmitterCirclesComponent();
            SelectedEntity.Components.Add(existing);
          }
          PropertiesExpanders.Add(existing);
        }
        else if (name.Contains("SphereGizmo"))
        {
          var existing = SelectedEntity.Components
            .OfType<SphereGizmoComponent>()
            .FirstOrDefault();
          if (existing == null)
          {
            existing = new SphereGizmoComponent();
            SelectedEntity.Components.Add(existing);
          }
          PropertiesExpanders.Add(existing);
        }
        else
        {
          PropertiesExpanders.Add(
            new UnknownComponentViewModel(
              name.Split(new[] { "::" }, System.StringSplitOptions.None).Last()
            )
          );
        }
      }

      foreach (var rule in _componentRules)
      {
        rule.Apply(SelectedEntity);
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
