using System;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;

namespace AetherVk.Views;

public partial class SpawnCometWindow : Window
{
  public SpawnCometWindow()
  {
    InitializeComponent();
  }

  public void CancelCommand()
  {
    Close(null);
  }

  public void ImportMeshCommand()
  {
    CommunityToolkit.Mvvm.Messaging.IMessengerExtensions.Send(
        CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default,
        new AetherVk.Logic.ViewModels.ImportModelRequestMessage()
    );
    Close(null);
  }

  public void SpawnCommand()
  {
    if (DataContext is SpawnCometViewModel vm && vm.SelectedModel != null)
    {
      if (vm.PhysicsType != "Static" && vm.FetchedOrbitData == null)
        return;

      var result = new SpawnCometResult(
        vm.SelectedModel,
        vm.EntityName,
        vm.PhysicsType,
        vm.FetchedOrbitData,
        vm.PosX, vm.PosY, vm.PosZ,
        vm.ScaleX, vm.ScaleY, vm.ScaleZ,
        vm.Pitch, vm.Yaw, vm.Roll,
        vm.CometRadiusKm
      );
      Close(result);
    }
  }
}
