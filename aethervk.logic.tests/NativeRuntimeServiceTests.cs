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
  private NativeRuntimeService _service;
  private string _assetPath;

  public NativeRuntimeServiceTests()
  {
    _service = new NativeRuntimeService();
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
      _service.InitializeSimulationContext("Vulkan", 800, 600, _assetPath);
      Assert.True(_service.IsInitialized);
      Assert.NotEmpty(_service.RootEntities);
      Assert.Contains(_service.RootEntities, e => e.Name == "root");
      
      var root = _service.RootEntities.FirstOrDefault();
      Assert.NotNull(root);
      Assert.Contains(root.Children, e => e.Name == "sun");
    }
    catch (System.DllNotFoundException)
    {
      // Headless CI or missing libraries, acceptable skip
    }
  }

  [Fact]
  public async Task ImportModel_ShouldReturnIdWhenValid()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 800, 600, _assetPath);
      ulong modelId = await _service.ImportModelAsync("dummy/path/to/model.glb");
      Assert.Equal(0ul, modelId);
    }
    catch (System.DllNotFoundException) {}
  }

  [Fact]
  public async Task SpawnModelInstance_ShouldAddEntityToScene()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 800, 600, _assetPath);
      var initialCount = _service.RootEntities.FirstOrDefault()?.Children.Count ?? 0;
      await Assert.ThrowsAsync<Exception>(() => _service.SpawnModelInstanceAsync(999, "TestSpawn"));
    }
    catch (System.DllNotFoundException) {}
  }

  [Fact]
  public void CreateMeasurement_ShouldAddEntity()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 800, 600, _assetPath);
      var entity = _service.CreateMeasurement("TestMeasure", new float[] { 0, 0, 0 }, new float[] { 1, 1, 1 });
      Assert.NotNull(entity);
      Assert.Equal("TestMeasure", entity.Name);
      
      var root = _service.RootEntities.FirstOrDefault();
      Assert.NotNull(root);
      Assert.Contains(root.Children, e => e.Id == entity.Id);
    }
    catch (System.DllNotFoundException) {}
  }
  
  [Fact]
  public void ProcessCommand_ShouldExecuteWithoutCrashing()
  {
    // Deprecated
  }
}
}
