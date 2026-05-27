using System;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Services;
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

  public void SpawnCommand()
  {
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
      vm.PosX, -vm.PosY, vm.PosZ,
      vm.ScaleX, vm.ScaleY, vm.ScaleZ,
      vm.Pitch, vm.Yaw, vm.Roll,
      vm.CometRadiusKm,
      massKg,
      vm.SelectedSpkRecord?.RecordId,
      vm.SelectedComet?.PrimaryDesignation
    );

    Close(result);
  }
}
