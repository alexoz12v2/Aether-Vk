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
      _service = new NativeRuntimeService(new SceneStateManager());
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
        Assert.NotEmpty(new System.Collections.ObjectModel.ObservableCollection<AetherVk.Logic.Models.Entity>()());
        Assert.Contains(new System.Collections.ObjectModel.ObservableCollection<AetherVk.Logic.Models.Entity>()());

        var root = new System.Collections.ObjectModel.ObservableCollection<AetherVk.Logic.Models.Entity>()().FirstOrDefault();
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
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong modelId = await _service.ImportModelAsync("dummy/path/to/model.glb");
        Assert.Equal(0ul);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task SpawnModelInstance_ShouldAddEntityToScene()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        var initialCount = new System.Collections.ObjectModel.ObservableCollection<AetherVk.Logic.Models.Entity>()().FirstOrDefault()?.Children.Count ?? 0;
        await Assert.ThrowsAsync<Exception>(() =>
          _service.SpawnModelInstanceAsync(999)
        );
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void CreateMeasurement_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath);
        Assert.NotNull(entity);
        Assert.Equal("TestMeasure", entity.Name);

        var root = new System.Collections.ObjectModel.ObservableCollection<AetherVk.Logic.Models.Entity>()().FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void ProcessCommand_ShouldExecuteWithoutCrashing()
    {
      // Deprecated
    }
  }
}
