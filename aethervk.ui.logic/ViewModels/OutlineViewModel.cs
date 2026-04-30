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

  public SceneStateManager StateManager { get; }

  [ObservableProperty]
  private ulong _currentSceneId;

  public ObservableCollection<Entity>? RootEntities => StateManager.GetOrCreateScene(CurrentSceneId).RootEntities;

  private Entity? _selectedEntity;
  public Entity? SelectedEntity
  {
    get => _selectedEntity;
    set
    {
      if (SetProperty(ref _selectedEntity, value))
      {
        var state = StateManager.GetOrCreateScene(CurrentSceneId);
        state.SelectedEntity = value;
        WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(value));
      }
    }
  }

  public OutlineViewModel(ulong sceneId, NativeRuntimeService runtimeService, SceneStateManager stateManager)
    : base("Outline")
  {
    _runtimeService = runtimeService;
    StateManager = stateManager;
    CurrentSceneId = sceneId;
    
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);
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

    var breadcrumb =
      ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
    breadcrumb?.ShowMessageAsync("Copied", $"Copied entity name to clipboard: {name}");

    var console = ServiceLocator.Provider?.GetService(typeof(ConsoleService)) as ConsoleService;
    console?.Log($"Copied entity name to clipboard: {name}");
  }

  [RelayCommand]
  private void DeleteEntity(Entity entity)
  {
    if (entity.IsMeasurement)
    {
      _runtimeService.RemoveEntity(CurrentSceneId, entity.Id);
      
      var breadcrumb = ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
      breadcrumb?.ShowMessageAsync("Entity Deleted", $"Deleted measurement: {entity.Name}");
    }
  }
}
