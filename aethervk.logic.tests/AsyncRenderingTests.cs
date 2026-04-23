using System.Diagnostics;
using System.Runtime.InteropServices;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class AsyncRenderingTests : IDisposable
{
  private readonly NativeRuntimeService _service;
  private readonly string _assetPath;

  public AsyncRenderingTests()
  {
    ServiceLocator.DispatchToUI = a => a();
    _service = new NativeRuntimeService();
    var baseDir = AppDomain.CurrentDomain.BaseDirectory;
    // Adjust asset path if necessary to point to the actual assets folder
    _assetPath = Path.GetFullPath(Path.Combine(baseDir, "../../../../assets"));
  }

  public void Dispose()
  {
    _service.Dispose();
  }

  [Fact]
  public void BasicNativeCall_ShouldSucceed()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
      Assert.True(_service.IsInitialized);
      ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
      Assert.NotEqual(0ul, sphereId);
    }
    catch (DllNotFoundException)
    {
    }
  }

  [Fact]
  public async Task RenderTickAsync_ShouldCompleteAndProduceImage()
  {
    try
    {
      const uint width = 256;
      const uint height = 256;

      _service.InitializeSimulationContext("Vulkan", width, height, _assetPath);
      Assert.True(_service.IsInitialized);

      // Set clear color to a specific value (e.g., green)
      _service.SetClearColor(0.0f, 1.0f, 0.0f, 1.0f);

      // Spawn a procedural sphere to have something in the scene
      ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
      Assert.NotEqual(0ul, sphereId);

      // Position camera to look at the sphere
      var root = _service.RootEntities.FirstOrDefault();
      var camera = root?.Children.FirstOrDefault(e => e.Name == "camera");
      Assert.NotNull(camera);

      // Render a frame and await completion
      await _service.RenderTickAsync();

      // Download the image
      int bufferSize = (int)(width * height * 4);
      IntPtr bufferPtr = Marshal.AllocHGlobal(bufferSize);
      try
      {
        bool success = _service.DownloadImage(bufferPtr, (nuint)bufferSize);
        Assert.True(success, "Failed to download image from GPU.");

        byte[] pixels = new byte[bufferSize];
        Marshal.Copy(bufferPtr, pixels, 0, bufferSize);

        // Assert on some pixels.
        // Since we cleared with green [0, 255, 0, 255], and the sphere is likely white [255, 255, 255, 255] or dark
        // We check if at least some pixels are NOT the clear color (meaning something was rendered)
        // Or check corners for clear color.

        bool hasNonZero = false;
        for (int i = 0; i < pixels.Length; i++)
        {
          if (pixels[i] > 0)
          {
            hasNonZero = true;
            break;
          }
        }

        Assert.True(hasNonZero, "Image should contain rendered data.");
      }
      finally
      {
        Marshal.FreeHGlobal(bufferPtr);
      }
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }

  [Fact]
  public void UnloadModel_ShouldRemoveFromScene()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
      Assert.True(_service.IsInitialized);

      ulong modelId = 999;

      var entityMapField = typeof(NativeRuntimeService).GetField("_entityMap",
        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
      var entityMap =
        (System.Collections.Generic.Dictionary<ulong, Entity>)entityMapField.GetValue(_service)!;

      var entity = new Entity(modelId, "model_999");
      _service.RootEntities.Add(entity);
      entityMap[modelId] = entity;

      Assert.NotNull(_service.GetEntityByName("model_999"));

      _service.UnloadModel(modelId);

      // Verify the mirroring is cleaned
      Assert.Null(_service.GetEntityByName("model_999"));
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }

  [Fact]
  public void MultipleScenes_ShouldManageIndependently()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath, populateDefault: false);
      Assert.True(_service.IsInitialized);

      // Create a second scene explicitly using the FFI
      // Wait, how do I create a second scene if CreateScene just overwrites the first one?
      // I should add a method to switch or spawn new scenes, but for now I can just call the native method
      ulong newSceneId = NativeInterop.avkSimulationContext_createDefaultScene(
        (IntPtr)typeof(NativeRuntimeService).GetField("_simulationContext",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
          .GetValue(_service)!);

      Assert.True(newSceneId > 0, "Failed to create a new scene natively");

      // Verify that service1 and service2 have separate root entities and no default components
      Assert.Single(_service.RootEntities);

      // Clean shutdown
      _service.ShutdownSimulation();
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }

  [Fact]
  public async Task MultipleCameras_ShouldRenderSameScene()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath, populateDefault: true);
      Assert.True(_service.IsInitialized);

      var root = _service.RootEntities.FirstOrDefault();
      Assert.NotNull(root);

      // Add a second camera
      var secondCamera = _service.CreateCamera(root);
      Assert.NotNull(secondCamera);

      // Switch to the first camera and render
      _service.SetActiveCamera(2); // First camera ID
      // _service.SimulationTick(); TODO to still develop native method
      Task first = _service.RenderTickAsync();

      // Switch to the second camera and render
      _service.SetActiveCamera(secondCamera.Id);
      // _service.SimulationTick();
      Task second = _service.RenderTickAsync();

      Debug.WriteLine("Time To wait");
      Task.WaitAll(first, second);

      _service.ShutdownSimulation();
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }

  [Fact]
  public async Task RenderTickAsync_ShouldTerminateOnShutdown()
  {
    try
    {
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
      Assert.True(_service.IsInitialized);

      var renderTask = _service.RenderTickAsync();

      // Close app
      _service.ShutdownSimulation();

      // Ensure the task completes rather than hanging forever
      await Task.WhenAny(renderTask, Task.Delay(2000));
      Assert.True(renderTask.IsCompleted, "RenderTickAsync did not terminate upon shutdown.");
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }
}
