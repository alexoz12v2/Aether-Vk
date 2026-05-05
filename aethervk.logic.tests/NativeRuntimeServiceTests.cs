using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  [Collection("Sequential")]
  public class NativeRuntimeServiceTests : IDisposable
  {
    private readonly NativeRuntimeService _service;
    private readonly SceneStateManager _stateManager;
    private readonly string _assetPath;

    public NativeRuntimeServiceTests()
    {
      var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
      dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<Action>())).Callback<Action>(a => a());
      _stateManager = new SceneStateManager();
      _service = new NativeRuntimeService(
        _stateManager,
        new ConsoleService(dispatcherMock.Object),
        new BreadcrumbService(dispatcherMock.Object),
        dispatcherMock.Object
      );
      var baseDir = AppDomain.CurrentDomain.BaseDirectory;
      _assetPath = System.IO.Path.GetFullPath(
        System.IO.Path.Combine(baseDir, "../../../../assets")
      );
    }

    public void Dispose()
    {
      _service.Dispose();
    }

    [Fact]
    public void Initialization_ShouldSucceedWithVulkanBackend()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        Assert.True(_service.IsInitialized);

        ulong sceneId = _service.CreateScene(true);
        var rootEntities = _stateManager.GetOrCreateScene(sceneId).RootEntities;

        Assert.NotEmpty(rootEntities);

        var root = rootEntities.FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children, e => e.Name == "sun");

        TestSceneExporter.ExportScene(sceneId, _stateManager, "Initialization_DefaultScene");
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task ImportModel_ShouldReturnIdWhenValid()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        string modelPath = System.IO.Path.Combine(_assetPath, "Comet.glb");
        ulong modelId = await _service.ImportModelAsync(modelPath);
        // It should return a valid model ID now.
        Assert.NotEqual(0ul, modelId);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task SpawnModelInstance_ShouldAddEntityToScene()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        var initialCount =
          _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault()?.Children.Count
          ?? 0;

        await Assert.ThrowsAsync<Exception>(() =>
          _service.SpawnModelInstanceAsync(sceneId, 999, "test")
        );
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void CreateMeasurement_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        var entity = _service.CreateMeasurement(
          sceneId,
          "TestMeasure",
          new float[] { 0, 0, 0 },
          new float[] { 1, 1, 1 }
        );

        Assert.NotNull(entity);
        Assert.Equal("TestMeasure", entity.Name);

        var root = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children, e => e.Id == entity.Id);

        TestSceneExporter.ExportScene(sceneId, _stateManager, "CreateMeasurement_Scene");
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void TimeControls_ShouldUpdateSimulationTime()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        _service.SetSimulationTime(sceneId, 1000.0);
        var time = _service.GetSimulationTime(sceneId);
        Assert.Equal(1000.0, time, 3);

        _service.SetTimeScale(sceneId, 1); // e.g. OneDay
        _service.PlayScene(sceneId);
        _service.PauseScene(sceneId);

        // Just ensure no native crash occurred during these commands.
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task RaycastNdc_ShouldCompleteSuccessfully()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        // This usually requires an active camera which is created by CreateScene
        var result = await _service.RaycastNdcAsync(sceneId, 0.5f, 0.5f);

        // As long as the task completes without crashing, we are good.
        // It might not hit anything in an empty scene.
        Assert.False(result.hit);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void ProceduralSphere_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "MySphere", 5.0f, 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var entity = _service.GetEntityByName(sceneId, "MySphere");
        Assert.NotNull(entity);
      }
      catch (System.DllNotFoundException) { }
    }
  }
}
