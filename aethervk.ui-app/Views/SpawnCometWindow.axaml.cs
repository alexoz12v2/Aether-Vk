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

  // TODO Command handling moved to view model
  public void SpawnCommand()
  {
    // TODO breadcrumg
    if (DataContext is not SpawnCometViewModel vm || vm.SelectedModel == null)
      return;

    // Orbit data is required for all modes (Step 3 gates CanGoNext already,
    // but enforce defensively here too).
    if (vm.FetchedOrbitData == null)
      return;

    // Resolve effective mass:
    //   - JPL GM-derived mass takes priority.
    //   - For Dynamic without JPL GM: use the user's slider value.
    //   - For Static/Kinematic: mass is cosmetic — density estimate is fine.
    float massKg;
    if (vm.FetchedOrbitData.MassKg.HasValue)
      massKg = (float)vm.FetchedOrbitData.MassKg.Value;
    else if (vm.PhysicsType == "Dynamic")
      massKg = (float)vm.DynamicMassKg;
    else
      massKg = (float)vm.FetchedOrbitData.EstimatedMassKg;

    var result = new SpawnCometResult(
      vm.SelectedModel,
      vm.EntityName,
      vm.PhysicsType,
      vm.FetchedOrbitData,
      vm.PosX,
      -vm.PosY,
      vm.PosZ,
      vm.ScaleX,
      vm.ScaleY,
      vm.ScaleZ,
      vm.Pitch,
      vm.Yaw,
      vm.Roll,
      vm.CometRadiusKm,
      massKg,
      vm.SelectedSpkRecord?.RecordId,
      vm.SelectedComet?.PrimaryDesignation
    );

    Close(result);
  }
}
