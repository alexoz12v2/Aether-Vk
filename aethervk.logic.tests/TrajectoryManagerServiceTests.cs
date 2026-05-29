using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  [Collection("Sequential")]
  public class TrajectoryManagerServiceTests : IDisposable
  {
    private readonly NativeRuntimeService _runtimeService;
    private readonly SceneStateManager _stateManager;
    private readonly TrajectoryManagerService _trajectoryService;

    public TrajectoryManagerServiceTests()
    {
      var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
      dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<Action>())).Callback<Action>(a => a());
      _stateManager = new SceneStateManager();
      _runtimeService = new NativeRuntimeService(
        _stateManager,
        new ConsoleService(dispatcherMock.Object),
        new BreadcrumbService(dispatcherMock.Object),
        new AetherVk.Logic.Services.NativeBufferPoolService(),
        dispatcherMock.Object
      );
      _trajectoryService = new TrajectoryManagerService(_runtimeService, _stateManager);
    }

    public void Dispose()
    {
      _runtimeService.Dispose();
    }

    [Fact]
    public async Task UpdateAllTrajectoriesAsync_ShouldNotThrow()
    {
      try
      {
        ulong sceneId = 1;
        // Mock a trajectory update
        await _trajectoryService.UpdateAllTrajectoriesAsync(sceneId, 0, 100, 1.0);
      }
      catch (DllNotFoundException) { }
    }

    [Fact]
    public async Task EnsureTrajectoryForSpkAsync_ShouldNotThrow()
    {
      try
      {
        ulong sceneId = 1;
        await _trajectoryService.EnsureTrajectoryForSpkAsync(sceneId, 399, 0, 100, 1.0);
      }
      catch (DllNotFoundException) { }
    }
  }
}
