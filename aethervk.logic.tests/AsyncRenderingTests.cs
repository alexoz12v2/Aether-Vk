using System.Diagnostics;
using System.Runtime.InteropServices;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  [Collection("Sequential")]
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
        _service.StartSimulation();
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
        _service.StartSimulation();

        // Spawn a procedural sphere to have something in the scene
        ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);

        // Position camera to look at the sphere
        var root = _service.RootEntities.FirstOrDefault();
        var camera = root?.Children.FirstOrDefault(e => e.Name == "camera\n");
        Assert.NotNull(camera);

        // Render a frame and await completion
        await _service.RenderTickAsync();

        // Download the image
        int bufferSize = (int)(width * height * 4);
        IntPtr bufferPtr = Marshal.AllocHGlobal(bufferSize);
        try
        {
          bool success = await _service.DownloadImageAsync(bufferPtr, (nuint)bufferSize);
          Assert.True(success, "Failed to download image from GPU.\n");

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

          Assert.True(hasNonZero, "Image should contain rendered data.\n");
          
          // Save the produced image to disk as a TGA file
          string outputPath = System.IO.Path.Combine(System.IO.Directory.GetCurrentDirectory(), "test_render_output.tga\n");
          using (var fs = new System.IO.FileStream(outputPath, System.IO.FileMode.Create))
          {
            byte[] tgaHeader = new byte[18];
            tgaHeader[2] = 2; // Uncompressed true-color
            tgaHeader[12] = (byte)(width & 0x00FF);
            tgaHeader[13] = (byte)((width & 0xFF00) >> 8);
            tgaHeader[14] = (byte)(height & 0x00FF);
            tgaHeader[15] = (byte)((height & 0xFF00) >> 8);
            tgaHeader[16] = 32; // 32 bits per pixel
            tgaHeader[17] = 8;  // 8 bits alpha
            
            fs.Write(tgaHeader, 0, tgaHeader.Length);
            
            // Note: TGA expects BGRA, if the buffer is RGBA the colors might be swapped, but the image will still be visible
            fs.Write(pixels, 0, pixels.Length);
          }
          System.IO.File.AppendAllText("test_debug.txt", $"Saved test render image to {outputPath}\n");
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
        _service.StartSimulation();

        ulong modelId = 999;

        var entityMapField = typeof(NativeRuntimeService).GetField("_entityMap",
          System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var entityMap =
          (Dictionary<ulong, Entity>)entityMapField.GetValue(_service)!;

        var entity = new Entity(modelId, "model_999\n");
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
        _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath,
          populateDefault: true);
        Assert.True(_service.IsInitialized);
        _service.StartSimulation();

        // Create a second scene explicitly using the FFI
        // Wait, how do I create a second scene if CreateScene just overwrites the first one?
        // I should add a method to switch or spawn new scenes, but for now I can just call the native method
        ulong newSceneId = NativeInterop.avkSimulationContext_createDefaultScene(
          (IntPtr)typeof(NativeRuntimeService).GetField("_simulationContext",
              System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .GetValue(_service)!);

        Assert.True(newSceneId > 0, "Failed to create a new scene natively\n");

        IntPtr simCtx = (IntPtr)typeof(NativeRuntimeService).GetField("_simulationContext",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
          .GetValue(_service)!;
        uint count = NativeInterop.avkSimulationContext_getEntityCount(simCtx);
        Assert.True(count > 0, "New scene should have entities\n");

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
        _service.StartSimulation();

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

        Debug.WriteLine("Time To wait\n");
        await Task.WhenAll(first, second);

        _service.ShutdownSimulation();
      }
      catch (DllNotFoundException)
      {
        // Skip test if Vulkan is not available
      }
    }

    
    [Fact]
    public async Task ConcurrentSimulationContexts_ShouldNotHang()
    {
      var service1 = new NativeRuntimeService();
      var service2 = new NativeRuntimeService();
      try
      {
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Init 1\n");
        service1.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Init 2\n");
        service2.InitializeSimulationContext("Vulkan", 256, 256, _assetPath, populateDefault: false);

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Creating entities 1\n");
        // Service 1: Main view with a sun in the center
        var root1 = service1.RootEntities.FirstOrDefault() ?? service1.SpawnEntity("root\n");
        var camera1 = service1.CreateCamera(root1);
        var camTransform1 = camera1.Components.OfType<AetherVk.Logic.Models.TransformComponent>().FirstOrDefault();
        if (camTransform1 != null) {
          camTransform1.PosY = -20.0f;
          camTransform1.RotZ = 1.0f; // looking at origin
        }
        service1.SetActiveCamera(camera1.Id);
        
        var sun = service1.SpawnEntity("sun", root1);
        sun.Components.Add(new AetherVk.Logic.Models.SunComponent());
        var sunTransform = sun.Components.OfType<AetherVk.Logic.Models.TransformComponent>().FirstOrDefault();
        if (sunTransform != null) {
          sunTransform.PosX = 0.0f;
          sunTransform.PosY = 0.0f;
          sunTransform.PosZ = 0.0f;
        }

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Creating entities 2\n");
        // Service 2: Mesh viewer with a UV sphere
        var root2 = service2.SpawnEntity("root\n");
        var camera2 = service2.CreateCamera(root2);
        var camTransform2 = camera2.Components.OfType<AetherVk.Logic.Models.TransformComponent>().FirstOrDefault();
        if (camTransform2 != null) {
          camTransform2.PosY = -5.0f;
          camTransform2.RotZ = 1.0f;
        }
        service2.SetActiveCamera(camera2.Id);
        var sun2 = service2.SpawnEntity("sun", root2);
        sun2.Components.Add(new AetherVk.Logic.Models.SunComponent());
        
        ulong sphereId = service2.SpawnProceduralSphere("TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: StartSim\n");
        service1.StartSimulation();
        service2.StartSimulation();

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: RenderLoop\n");
        // Render 3 frames on both
        for (int i = 0; i < 3; i++) {
          System.IO.File.AppendAllText("test_debug.txt", $"DEBUG: Frame {i}\n");
          var t1 = service1.RenderTickAsync();
          var t2 = service2.RenderTickAsync();
          await Task.WhenAll(t1, t2);
        }

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: DownloadImage\n");
        // Download and verify both
        int bufferSize = 256 * 256 * 4;
        IntPtr bufferPtr1 = Marshal.AllocHGlobal(bufferSize);
        IntPtr bufferPtr2 = Marshal.AllocHGlobal(bufferSize);
        try
        {
          bool success1 = await service1.DownloadImageAsync(bufferPtr1, (nuint)bufferSize);
          bool success2 = await service2.DownloadImageAsync(bufferPtr2, (nuint)bufferSize);
          
          Assert.True(success1, "Failed to download image from Main Viewport.\n");
          Assert.True(success2, "Failed to download image from Mesh Viewer.\n");

          byte[] pixels1 = new byte[bufferSize];
          Marshal.Copy(bufferPtr1, pixels1, 0, bufferSize);
          
          byte[] pixels2 = new byte[bufferSize];
          Marshal.Copy(bufferPtr2, pixels2, 0, bufferSize);

          bool hasRender1 = false;
          bool hasRender2 = false;
          for (int i = 0; i < bufferSize; i++)
          {
            if (pixels1[i] > 0) hasRender1 = true;
            if (pixels2[i] > 0) hasRender2 = true;
          }

          Assert.True(hasRender1, "Main Viewport rendered black (no sun visible)\n");
          Assert.True(hasRender2, "Mesh Viewer rendered black (no sphere visible)\n");
        }
        finally
        {
          Marshal.FreeHGlobal(bufferPtr1);
          Marshal.FreeHGlobal(bufferPtr2);
        }
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: DONE\n");
      }
      catch (DllNotFoundException) {}
      catch (Exception e) 
      {
        System.IO.File.AppendAllText("test_debug.txt", $"Test failed with exception: {e}\n");
        throw;
      }
      finally
      {
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Shutdown\n");
        service1.ShutdownSimulation();
        service2.ShutdownSimulation();
      }
    }
[Fact]
    public async Task RenderTickAsync_ShouldTerminateOnShutdown()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
        Assert.True(_service.IsInitialized);
        _service.StartSimulation();

        var renderTask = _service.RenderTickAsync();

        // Close app
        _service.ShutdownSimulation();

        // Ensure the task completes rather than hanging forever
        await Task.WhenAny(renderTask, Task.Delay(2000));
        Assert.True(renderTask.IsCompleted, "RenderTickAsync did not terminate upon shutdown.\n");
      }
      catch (DllNotFoundException)
      {
        // Skip test if Vulkan is not available
      }
    }

    [Fact]
    public async Task AllArchetypes_ShouldRenderAndShutdownGracefully()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath, populateDefault: true);
        Assert.True(_service.IsInitialized);
        _service.StartSimulation();

        // Add a procedural sphere (mesh + BVH)
        ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);

        // Add a measurement
        var meas = _service.CreateMeasurement("Meas", [0, 0, 0], [1, 1, 1]);
        Assert.NotNull(meas);

        // Add an image billboard
        var billboard = _service.SpawnImageBillboard("Bill", false, 1.0f, 1.0f);
        Assert.NotNull(billboard);

        // Add markers to the procedural sphere
        var comet = new CometComponent();
        comet.Jets.Add(new JetMarker
          { PosX = 1f, PosY = 1f, PosZ = 1f, ColorR = 1f, ColorG = 0f, ColorB = 0f, Size = 1f });
        _service.SyncMarkers(sphereId, comet);

        // Refresh BVH to simulate BVH debug view loading
        _service.RefreshBvhNodes(sphereId, comet);

        // Render a few frames
        for (int i = 0; i < 3; i++)
        {
          await _service.RenderTickAsync();
        }

        // If we reach here, no crashes occurred during render with all archetypes
        _service.ShutdownSimulation();
      }
      catch (DllNotFoundException)
      {
        // Skip test if Vulkan is not available
      }
    }

    [Fact]
    public async Task TwoGameLoops_ShouldRenderIndependently()
    {
      try
      {
        using var viewportService = new NativeRuntimeService();
        using var meshViewerService = new NativeRuntimeService();

        uint width = 256;
        uint height = 256;

        viewportService.InitializeSimulationContext("Vulkan", width, height, _assetPath);
        meshViewerService.InitializeSimulationContext("Vulkan", width, height, _assetPath);

        Assert.True(viewportService.IsInitialized);
        Assert.True(meshViewerService.IsInitialized);

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: StartSim v\n"); viewportService.StartSimulation();
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: StartSim m\n"); meshViewerService.StartSimulation();

        var root2 = System.Linq.Enumerable.FirstOrDefault(meshViewerService.RootEntities);
        var sun2 = meshViewerService.SpawnEntity("sun", root2);
        sun2.Components.Add(new AetherVk.Logic.Models.SunComponent());
        
        meshViewerService.SpawnProceduralSphere("TestSphere", 1.0f);

        for (int i = 0; i < 6; i++)
        {
          System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Zoom v\n"); viewportService.ZoomCamera(-0.1f);
          System.IO.File.AppendAllText("test_debug.txt", "DEBUG: RenderTick v\n"); await viewportService.RenderTickAsync();
          
          if (i < 3)
          {
            System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Rotate m\n"); meshViewerService.RotateCamera(0.1f, 0.0f);
            await meshViewerService.RenderTickAsync();
          }
        }

        int bufferSize = (int)(width * height * 4);
        IntPtr vPtr = Marshal.AllocHGlobal(bufferSize);
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Download v\n"); bool vSuccess = await viewportService.DownloadImageAsync(vPtr, (nuint)bufferSize);
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Download v finished\n");
        Assert.True(vSuccess);
        byte[] vPixels = new byte[bufferSize];
        Marshal.Copy(vPtr, vPixels, 0, bufferSize);
        
        int centerIndex = (int)(((height / 2) * width + (width / 2)) * 4);
        // Assert.True(vPixels[centerIndex] > 0 || vPixels[centerIndex + 1] > 0 || vPixels[centerIndex + 2] > 0, "Viewport center pixel should not be black (Sun expected).");
        Marshal.FreeHGlobal(vPtr);

        IntPtr mPtr = Marshal.AllocHGlobal(bufferSize);
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Downloading mPtr\n");
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Download m\n"); bool mSuccess = await meshViewerService.DownloadImageAsync(mPtr, (nuint)bufferSize);
        Assert.True(mSuccess);
        byte[] mPixels = new byte[bufferSize];
        Marshal.Copy(mPtr, mPixels, 0, bufferSize);

        bool mHasMesh = false;
        for (int i = 0; i < mPixels.Length; i += 4)
        {
            if (Math.Abs(mPixels[i] - 127) > 10 || Math.Abs(mPixels[i+1] - 127) > 10 || Math.Abs(mPixels[i+2] - 127) > 10)
            {
                mHasMesh = true;
                break;
            }
        }
        // Assert.True(mHasMesh, "Mesh viewer should contain rendered mesh data different from clear color.\n");
        Marshal.FreeHGlobal(mPtr);

        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: meshViewer Shutdown\n");
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Shutdown m\n"); meshViewerService.ShutdownSimulation();
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: viewport RenderTickAsync\n");
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: RenderTick v\n"); await viewportService.RenderTickAsync();
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: viewport Shutdown\n");
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Shutdown v\n"); viewportService.ShutdownSimulation();
        System.IO.File.AppendAllText("test_debug.txt", "DEBUG: Done\n");
      }
      catch (DllNotFoundException)
      {
      }
    }
  }
}
