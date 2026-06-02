using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Input;

namespace AetherVk.Views;

public partial class SpawnCometWindow : Window
{
  public SpawnCometWindow()
  {
    InitializeComponent();
  }

  private void CometsAutoCompleteBox_DoubleTapped(object? sender, TappedEventArgs e)
  {
    if (sender is AutoCompleteBox autoCompleteBox && string.IsNullOrEmpty(autoCompleteBox.Text))
    {
      autoCompleteBox.IsDropDownOpen = true;
    }
  }

  public void CancelCommand()
  {
    Close(null);
  }

  public void SpawnCommand()
  {
    if (DataContext is not SpawnCometViewModel vm || vm.SelectedModel == null)
      return;

    // Timeline must be validated
    if (!vm.IsTimelineValidated)
      return;

    var result = new SpawnCometResult(
      vm.SelectedModel,
      vm.EntityName,
      vm.PhysicsType,
      vm.IsStaticMode ? vm.PosX : 0f,
      vm.IsStaticMode ? vm.PosY : 0f,
      vm.IsStaticMode ? vm.PosZ : 0f,
      vm.Pitch,
      vm.Yaw,
      vm.Roll,
      vm.CometRadiusKm,
      (float)vm.MassKg,
      vm.AngularVelX,
      vm.AngularVelY,
      vm.AngularVelZ,
      vm.SelectedSpkRecord?.RecordId,
      vm.SelectedComet?.PrimaryDesignation,
      vm.PoleRaDeg,
      vm.PoleDecDeg,
      vm.PrimeMeridianDeg,
      vm.PoleRaRateDeg,
      vm.PoleDecRateDeg,
      vm.RotationRateDeg,
      vm.WizardStartEpoch,
      vm.WizardEndEpoch
    );

    Close(result);
  }
}
