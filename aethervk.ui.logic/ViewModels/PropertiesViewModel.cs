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
  [ObservableProperty]
  private Entity? _selectedEntity;

  public PropertiesViewModel()
    : base("Properties")
  {
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(this);
  }

  [ObservableProperty]
  private bool _isFollowingEntity;

  partial void OnIsFollowingEntityChanged(bool value)
  {
    if (SelectedEntity != null)
    {
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      
      if (value)
      {
        runtimeService?.FollowEntity(SelectedEntity.Id);
      }
      else
      {
        runtimeService?.UnfollowEntity();
      }
    }
  }

  public void Receive(EntitySelectedMessage message)
  {
    SelectedEntity = message.SelectedEntity;
    IsFollowingEntity = false; // Reset when selection changes


    if (SelectedEntity != null)
    {
      var transform = SelectedEntity.Components.OfType<TransformComponent>().FirstOrDefault();
      if (transform != null)
      {
        bool hasSunOrPlanet = SelectedEntity.Components.Any(c => c is SunComponent || c is PlanetComponent);
        transform.IsEditable = !hasSunOrPlanet;
      }
    }
  }

  [RelayCommand]
  private void SnapToSelectedEntity()
  {
    if (SelectedEntity != null)
    {
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      runtimeService?.SnapToEntity(SelectedEntity.Id);
    }
  }

  [RelayCommand]
  private void ToggleAddJetMode()
  {
    WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.ToggleAddJetModeMessage());
    
    var breadcrumb = ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
    breadcrumb?.ShowMessageAsync("Add Jet Mode", "Hold Shift and Right Click on the comet to add a jet at the intersection point.");
  }
}
