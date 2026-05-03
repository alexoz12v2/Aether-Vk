using System.Collections.ObjectModel;
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

  [ObservableProperty] private ulong _currentSceneId;

  public ObservableCollection<Entity>? RootEntities =>
    StateManager.GetOrCreateScene(CurrentSceneId).RootEntities;

  private Entity? _selectedEntity;

  public Entity? SelectedEntity
  {
    get => _selectedEntity;
    set
    {
      if (SetProperty(ref _selectedEntity, value))
      {
        _consoleService?.Log($"[Outline] SelectedEntity changed to: {value?.Name ?? "null"}");
        var state = StateManager.GetOrCreateScene(CurrentSceneId);
        state.SelectedEntity = value;
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

    var state = StateManager.GetOrCreateScene(CurrentSceneId);
    SelectedEntity = state.SelectedEntity;
  }

  public void Receive(EntitySelectedMessage message)
  {
    if (_selectedEntity != message.SelectedEntity)
    {
      _selectedEntity = message.SelectedEntity;
      OnPropertyChanged(nameof(SelectedEntity));
    }
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
