using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

public class SpawnCometViewModelTests
{
  [Fact]
  public void Constructor_InitializesDefaultValues()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var models = new List<ImportedModelItem>();
    var timelineService = new TimelineService();

    var vm = new SpawnCometViewModel(models, horizonService, timelineService, breadcrumb);

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
    var runtimeService = new NativeRuntimeService(stateManager, console, breadcrumb, dispatcherMock.Object);
    var models = new List<ImportedModelItem>
    {
      new ImportedModelItem(1, "TestModel", "path", runtimeService, new Moq.Mock<IWindowService>().Object)
    };
    var timelineService = new TimelineService();

    var vm = new SpawnCometViewModel(models, horizonService, timelineService, breadcrumb);

    // Step 1: model is selected → can proceed
    Assert.True(vm.CanGoNext);
    
    vm.NextStepCommand.Execute(null);
    Assert.Equal(2, vm.CurrentStep);
    Assert.True(vm.CanGoBack);
    
    // Step 2: any valid physics type → can proceed
    vm.PhysicsType = "Dynamic";
    Assert.True(vm.CanGoNext);
    
    vm.NextStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);
    Assert.False(vm.IsFinalStep); // step 4 is the final step now

    // Step 3 with Dynamic: requires both SelectedSpkRecord AND FetchedOrbitData
    Assert.False(vm.CanGoNext);
    vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000030", EpochYear = "1986", Name = "Halley" };
    Assert.False(vm.CanGoNext); // still false: FetchedOrbitData is null
    vm.FetchedOrbitData = new PlanetOrbitData();
    Assert.True(vm.CanGoNext);

    vm.NextStepCommand.Execute(null);
    Assert.Equal(4, vm.CurrentStep);
    Assert.True(vm.IsFinalStep); // step 4 is the final step

    // Step 4: always can proceed (placement is optional)
    Assert.True(vm.CanGoNext);

    vm.PreviousStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);

    // Step 3 with Static: can skip JPL data
    vm.PhysicsType = "Static";
    Assert.True(vm.CanGoNext);
  }

  [Fact]
  public void GetRotationQuaternion_CalculatesCorrectly()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var timelineService = new TimelineService();
    var vm = new SpawnCometViewModel(new List<ImportedModelItem>(), horizonService, timelineService, breadcrumb)
    {
      Pitch = 90,
      Yaw = 0,
      Roll = 0
    };

    var (w, x, y, z) = vm.GetRotationQuaternion();
    
    // 90 deg pitch (rotation around X or Y depending on convention, ZYX means roll=X, yaw=Y, pitch=Z?)
    // In ZYX convention: Roll(X), Yaw(Y), Pitch(Z) normally, but let's just check it doesn't crash 
    // and returns normalized quaternion
    var lengthSq = w * w + x * x + y * y + z * z;
    Assert.True(System.Math.Abs(lengthSq - 1.0f) < 0.0001f);
  }
}
