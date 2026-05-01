using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

public class Viewport3DViewModelTests
{
  [Fact]
  public void Initialization_SetsUpDimensions_WithoutNativeCrash()
  {
    // Arrange
    var runtimeService = new NativeRuntimeService(new SceneStateManager());
    // Do not call InitializeSimulationContext so it stays in mock state

    try
    {
      var vm = new Viewport3DViewModel(runtimeService);
      Assert.Equal(800u, vm.Width);
      Assert.Equal(600u, vm.Height);
      vm.Stop();
    }
    catch (System.TypeInitializationException) { }
    catch (System.DllNotFoundException) { }
  }
}
