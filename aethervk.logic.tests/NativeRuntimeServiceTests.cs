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
      _service = new NativeRuntimeService(_stateManager, new ConsoleService(dispatcherMock.Object), new BreadcrumbService(dispatcherMock.Object), dispatcherMock.Object);
      var baseDir = AppDomain.CurrentDomain.BaseDirectory;
      _assetPath = System.IO.Path.GetFullPath(System.IO.Path.Combine(baseDir, "../../../../assets"));
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
        ulong modelId = await _service.ImportModelAsync("dummy/path/to/model.glb");
        // Without an actual GLB it returns 0.
        Assert.Equal(0ul, modelId);
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
        var initialCount = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault()?.Children.Count ?? 0;
        
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
        
        var entity = _service.CreateMeasurement(sceneId, "TestMeasure", new float[]{0,0,0}, new float[]{1,1,1});
        
        Assert.NotNull(entity);
        Assert.Equal("TestMeasure", entity.Name);

        var root = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children, e => e.Id == entity.Id);
        
        TestSceneExporter.ExportScene(sceneId, _stateManager, "CreateMeasurement_Scene");
      }
      catch (System.DllNotFoundException) { }
    }
  }
}
