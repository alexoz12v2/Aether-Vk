using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public class EntitySelectedMessage
{
  public Entity? SelectedEntity { get; }

  public EntitySelectedMessage(Entity? selectedEntity)
  {
    SelectedEntity = selectedEntity;
  }
}

public partial class OutlineViewModel : TabItemViewModel, IRecipient<EntitySelectedMessage>
{
  private readonly NativeRuntimeService _runtimeService;
  private readonly BreadcrumbService? _breadcrumbService;
  private readonly ConsoleService? _consoleService;

  public SceneStateManager StateManager { get; }

  [ObservableProperty]
  private ulong _currentSceneId;

  public ObservableCollection<Entity>? RootEntities =>
    StateManager.GetOrCreateScene(CurrentSceneId).RootEntities;

  public Entity? SelectedEntity
  {
    get => StateManager.GetOrCreateScene(CurrentSceneId).SelectedEntity;
    set
    {
      var state = StateManager.GetOrCreateScene(CurrentSceneId);
      if (state.SelectedEntity != value)
      {
        state.SelectedEntity = value;
        _consoleService?.Log($"[Outline] SelectedEntity changed to: {value?.Name ?? "null"}");
        OnPropertyChanged();
        WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(value));
      }
    }
  }

  public OutlineViewModel(
    ulong sceneId,
    NativeRuntimeService runtimeService,
    SceneStateManager stateManager,
    BreadcrumbService? breadcrumbService = null,
    ConsoleService? consoleService = null
  )
    : base("Outline")
  {
    _runtimeService = runtimeService;
    StateManager = stateManager;
    CurrentSceneId = sceneId;
    _breadcrumbService = breadcrumbService;
    _consoleService = consoleService;

    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);

    // When the runtime initialises after tab creation, update the scene ID to the real
    // scene and notify bindings so the outline tree refreshes.
    runtimeService.PropertyChanged += (s, e) =>
    {
      if (
        e.PropertyName == nameof(NativeRuntimeService.IsInitialized)
        && runtimeService.IsInitialized
      )
        RefreshSceneId(stateManager);
    };

    // CreateScene sends SimulationStateUpdatedMessage after it finishes building the
    // entity tree — refresh here too in case we missed the IsInitialized event.
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.SimulationStateUpdatedMessage>(
      this,
      (r, m) => ((OutlineViewModel)r).RefreshSceneId(stateManager)
    );
  }

  private void RefreshSceneId(SceneStateManager stateManager)
  {
    var first = stateManager.AllScenes.FirstOrDefault();
    if (first != null && CurrentSceneId != first.SceneId)
    {
      CurrentSceneId = first.SceneId;
    }
    // Always re-raise RootEntities so the AXAML binding re-reads the live collection.
    OnPropertyChanged(nameof(RootEntities));
  }

  public void Receive(EntitySelectedMessage message)
  {
    OnPropertyChanged(nameof(SelectedEntity));
  }

  [RelayCommand]
  private void CopyNameToClipboard(string name)
  {
    WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.CopyToClipboardMessage(name));

    _breadcrumbService?.ShowMessageAsync("Copied", $"Copied entity name to clipboard: {name}");

    _consoleService?.Log($"Copied entity name to clipboard: {name}");
  }

  [RelayCommand]
  private void DeleteEntity(Entity entity)
  {
    if (entity.IsMeasurement)
    {
      _runtimeService.RemoveEntity(CurrentSceneId, entity.Id);

      _breadcrumbService?.ShowMessageAsync("Entity Deleted", $"Deleted measurement: {entity.Name}");
    }
  }
}
