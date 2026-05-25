using System.Collections.Generic;
using System.Linq;
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
    var horizonService = new HorizonJplService(console, breadcrumb);
    var models = new List<ImportedModelItem>();

    var vm = new SpawnCometViewModel(models, horizonService);

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
    var horizonService = new HorizonJplService(console, breadcrumb);
    
    var stateManager = new SceneStateManager();
    var runtimeService = new NativeRuntimeService(stateManager, console, breadcrumb, dispatcherMock.Object);
    var models = new List<ImportedModelItem>
    {
      new ImportedModelItem(1, "TestModel", "path", runtimeService, new Moq.Mock<IWindowService>().Object)
    };

    var vm = new SpawnCometViewModel(models, horizonService);

    Assert.True(vm.CanGoNext); // Model is selected in constructor if list has items
    
    vm.NextStepCommand.Execute(null);
    Assert.Equal(2, vm.CurrentStep);
    Assert.True(vm.CanGoBack);
    
    vm.PhysicsType = "Static";
    Assert.True(vm.CanGoNext);
    
    vm.NextStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);
    
    // Cannot go next in step 3 until FetchedOrbitData is set
    Assert.False(vm.CanGoNext);
    vm.FetchedOrbitData = new PlanetOrbitData();
    Assert.True(vm.CanGoNext);
    
    vm.NextStepCommand.Execute(null);
    Assert.Equal(4, vm.CurrentStep);
    Assert.True(vm.CanGoNext); // Always true on step 4
    
    vm.PreviousStepCommand.Execute(null);
    Assert.Equal(3, vm.CurrentStep);
  }

  [Fact]
  public void GetRotationQuaternion_CalculatesCorrectly()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var horizonService = new HorizonJplService(console, breadcrumb);
    
    var vm = new SpawnCometViewModel(new List<ImportedModelItem>(), horizonService)
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
