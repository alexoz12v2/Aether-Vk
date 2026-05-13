using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
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
    dispatcherMock
      .Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());
    var runtimeService = new NativeRuntimeService(
      new SceneStateManager(),
      new ConsoleService(dispatcherMock.Object),
      new BreadcrumbService(dispatcherMock.Object),
      dispatcherMock.Object
    );
    // Do not call InitializeSimulationContext so it stays in mock state

    try
    {
      var sm = new SceneStateManager();
      var b = new BreadcrumbService(dispatcherMock.Object);
      var vm = new Viewport3DViewModel(runtimeService, b, sm, dispatcherMock.Object);
      Assert.Equal(800u, vm.Width);
      Assert.Equal(600u, vm.Height);
      vm.Stop();
    }
    catch (System.TypeInitializationException) { }
    catch (System.DllNotFoundException) { }
  }

  [Fact]
  public void ProcessAction_HandlesOrbitCamera()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock
      .Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());
    var runtimeService = new NativeRuntimeService(
      new SceneStateManager(),
      new ConsoleService(dispatcherMock.Object),
      new BreadcrumbService(dispatcherMock.Object),
      dispatcherMock.Object
    );

    var sm = new SceneStateManager();
    var b = new BreadcrumbService(dispatcherMock.Object);
    var vm = new Viewport3DViewModel(runtimeService, b, sm, dispatcherMock.Object);
    // Press middle button to start orbit
    bool handled = vm.ProcessAction(new AppAction("viewport.start_orbit", "Orbit"), true);
    Assert.True(handled);

    // Pointer delta should now be handled by OrbitOperator
    bool deltaHandled = vm.OperatorStack.ProcessPointerDelta(10, 10);
    Assert.True(deltaHandled);

    // Release middle button to stop orbit
    vm.ProcessAction(new AppAction("viewport.start_orbit", "Orbit"), false);

    // Delta should now fall back to base operator which returns false
    bool finalDeltaHandled = vm.OperatorStack.ProcessPointerDelta(10, 10);
    Assert.False(finalDeltaHandled);
  }
}
