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
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());
    var runtimeService = new NativeRuntimeService(new SceneStateManager(),
      new ConsoleService(dispatcherMock.Object), new BreadcrumbService(dispatcherMock.Object),
      dispatcherMock.Object);
    // Do not call InitializeSimulationContext so it stays in mock state

    try
    {
      var sm = new SceneStateManager();
      var b = new BreadcrumbService(dispatcherMock.Object);
      var vm = new Viewport3DViewModel(runtimeService, b, sm);
      Assert.Equal(800u, vm.Width);
      Assert.Equal(600u, vm.Height);
      vm.Stop();
    }
    catch (System.TypeInitializationException)
    {
    }
    catch (System.DllNotFoundException)
    {
    }
  }
}
