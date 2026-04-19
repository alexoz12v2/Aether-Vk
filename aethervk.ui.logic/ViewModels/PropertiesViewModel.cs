using System.Collections.ObjectModel;
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

  public void Receive(EntitySelectedMessage message)
  {
    SelectedEntity = message.SelectedEntity;
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
  }
}
