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

public partial class OutlineViewModel : TabItemViewModel
{
  private readonly NativeRuntimeService _runtimeService;

  public ObservableCollection<Entity> RootEntities => _runtimeService.RootEntities;

  [ObservableProperty]
  private Entity? _selectedEntity;

  public OutlineViewModel(NativeRuntimeService runtimeService)
    : base("Outline")
  {
    _runtimeService = runtimeService;
  }

  partial void OnSelectedEntityChanged(Entity? value)
  {
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(value));
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
      _runtimeService.RemoveEntity(entity.Id);
      
      var breadcrumb = ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
      breadcrumb?.ShowMessageAsync("Entity Deleted", $"Deleted measurement: {entity.Name}");
    }
  }
}
