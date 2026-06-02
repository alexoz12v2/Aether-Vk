using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

[Collection("Sequential")]
public class SpawnCometViewModelTests
{
  private static (SpawnCometViewModel vm, NativeRuntimeService runtime) CreateVm(
    List<ImportedModelItem>? models = null
  )
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var timelineService = new TimelineService();
    var stateManager = new SceneStateManager();
    var runtimeService = new NativeRuntimeService(
      stateManager,
      console,
      breadcrumb,
      new NativeBufferPoolService(),
      dispatcherMock.Object
    );

    models ??= new List<ImportedModelItem>();

    var vm = new SpawnCometViewModel(
      models,
      horizonService,
      runtimeService,
      timelineService,
      breadcrumb
    );

    return (vm, runtimeService);
  }

  [Fact]
  public void Constructor_InitializesDefaultValues()
  {
    var (vm, _) = CreateVm();

    Assert.Equal(1, vm.CurrentStep);
    Assert.True(vm.IsStep1);
    Assert.False(vm.CanGoBack);
    Assert.False(vm.CanGoNext); // Because no model is selected
  }

  [Fact]
  public void NextStep_PreviousStep_NavigateCorrectly()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var stateManager = new SceneStateManager();
    var runtimeService = new NativeRuntimeService(
      stateManager,
      console,
      breadcrumb,
      new NativeBufferPoolService(),
      dispatcherMock.Object
    );
    var models = new List<ImportedModelItem>
    {
      new ImportedModelItem(
        1,
        "TestModel",
        "path",
        runtimeService,
        new Moq.Mock<IWindowService>().Object
      ),
    };
    var timelineService = new TimelineService();

    var vm = new SpawnCometViewModel(
      models,
      horizonService,
      runtimeService,
      timelineService,
      breadcrumb
    );

    // Step 1: model is selected → can proceed
    Assert.True(vm.CanGoNext);

    vm.NextStepCommand.Execute(null);
    Assert.Equal(2, vm.CurrentStep);
    Assert.True(vm.CanGoBack);

    // Step 2: Static physics type → can proceed
    vm.PhysicsType = "Static";
    Assert.True(vm.CanGoNext);

    vm.NextStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);
    Assert.False(vm.IsFinalStep); // step 4 is the final step

    // Step 3: requires HasValidSpkRecord
    Assert.False(vm.CanGoNext);
    vm.SelectedSpkRecord = new SpkRecordItem
    {
      RecordId = "90000030",
      EpochYear = "1986",
      Name = "Halley",
    };
    Assert.True(vm.CanGoNext); // SPK record is sufficient now

    vm.NextStepCommand.Execute(null);
    Assert.Equal(4, vm.CurrentStep);
    Assert.True(vm.IsFinalStep); // step 4 is the final step

    // Step 4: requires IsTimelineValidated
    Assert.False(vm.CanGoNext); // Not validated yet

    vm.PreviousStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);
  }

  [Fact]
  public void GetRotationQuaternion_CalculatesCorrectly()
  {
    var (vm, _) = CreateVm();
    vm.Pitch = 90;
    vm.Yaw = 0;
    vm.Roll = 0;

    var (w, x, y, z) = vm.GetRotationQuaternion();

    // Check it returns a normalized quaternion
    var lengthSq = w * w + x * x + y * y + z * z;
    Assert.True(System.Math.Abs(lengthSq - 1.0f) < 0.0001f);
  }

  [Fact]
  public void EpochValidation_StartBeforeEnd()
  {
    var (vm, _) = CreateVm();

    // Default: start before end (from TimelineService defaults)
    Assert.True(vm.IsEpochRangeValid);

    // Set start after end
    vm.WizardStartEpoch = new System.DateTimeOffset(2030, 1, 1, 0, 0, 0, System.TimeSpan.Zero);
    vm.WizardEndEpoch = new System.DateTimeOffset(2020, 1, 1, 0, 0, 0, System.TimeSpan.Zero);
    Assert.False(vm.IsEpochRangeValid);

    // Fix it
    vm.WizardEndEpoch = new System.DateTimeOffset(2035, 1, 1, 0, 0, 0, System.TimeSpan.Zero);
    Assert.True(vm.IsEpochRangeValid);
  }

  [Fact]
  public void EpochChange_ResetsValidation()
  {
    var (vm, _) = CreateVm();

    // Simulate validation
    vm.IsTimelineValidated = true;
    Assert.True(vm.IsTimelineValidated);

    // Changing epoch should reset validation
    vm.WizardStartEpoch = new System.DateTimeOffset(2025, 6, 1, 0, 0, 0, System.TimeSpan.Zero);
    Assert.False(vm.IsTimelineValidated);
  }
}
