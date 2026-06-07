using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Messaging;
using Moq;
using Xunit;

namespace aethervk.app.tests;

[Collection("Sequential")] // Prevents concurrent Vulkan init with AsyncRenderingTests
public class SpawnCometFlowE2ETests
{
  /// <summary>
  /// End-to-end test: initialises Vulkan headless, spawns a comet, waits for a render frame.
  /// Skipped automatically when the native dylib is absent (DllNotFoundException).
  /// Hard timeout: 20 s to prevent the runner from blocking on the native render loop.
  /// </summary>
  [Fact(Timeout = 20_000)]
  public async Task SpawnComet_E2E_Simulation_Test()
  {
    var console = new ConsoleService(new Mock<IUiThreadDispatcher>().Object);
    var dispatcherMock = new Mock<IUiThreadDispatcher>();
    dispatcherMock
      .Setup(d => d.DispatchAsync(It.IsAny<Func<Task>>()))
      .Returns<Func<Task>>(f => f());
    dispatcherMock.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());

    var sceneStateManager = new SceneStateManager();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var nativeRuntime = new NativeRuntimeService(
      sceneStateManager,
      console,
      breadcrumb,
      new AetherVk.Logic.Services.NativeBufferPoolService(),
      dispatcherMock.Object
    );

    try
    {
      // Initialize headless
      nativeRuntime.InitializeSimulationContext("Vulkan", null, false);

      // Create Default Scene
      ulong defaultSceneId = nativeRuntime.CreateScene(true);

      // Setup Presentation Engine
      ulong presentationEngineId = nativeRuntime.CreatePresentationEngine(512, 512, defaultSceneId);
      ulong cameraId = nativeRuntime.AddPerspectiveCamera(
        defaultSceneId,
        presentationEngineId,
        "MainCamera",
        45f,
        0.001f,
        1000f
      );

      // Setup Horizon API Mock (even if we just call SpawnComet directly, the test tests the whole app infrastructure)
      var mockStorage = new Mock<ILocalStorageService>();
      var horizonService = new HorizonJplService(console, breadcrumb, mockStorage.Object);
      // Note: since we are headless, we bypass the ViewModels and test the runtime services directly.

      // Load a mock GLTF model (we need a valid file so the C++ engine doesn't crash on load)
      // There is an asset comet in aethervk.core/assets/Comet.glb
      string assetPath = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, "../../../../assets/Comet.glb")
      );
      ulong modelId = 1;
      if (File.Exists(assetPath))
      {
        ulong modelResult = await nativeRuntime.ImportModelAsync(assetPath);
        modelId = modelResult;
      }
      else
      {
        // If no asset, we can't fully test spawning a mesh headless without it failing.
        // The native runtime requires a valid GLTF.
      }

      // Spawn Comet at 2 AU to avoid overlapping with sun bounds (Sun is typically at origin)
      var (lcaId, cometId) = nativeRuntime.SpawnComet(
        defaultSceneId,
        modelId,
        "TestComet",
        2f,
        0f,
        0f,
        1f,
        0f,
        0f,
        0f,
        2.5f,
        1e13f,
        2,
        0, // naifId — not used for Dynamic physics type
        0.0,
        90.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0f,
        0f,
        0f
      );

      // Verify Hierarchy
      Assert.True(lcaId > 0, "LCA ID should be valid");
      Assert.True(cometId > 0, "Comet ID should be valid");

      // Assert that state manager tracked the newly spawned entities correctly
      var sceneState = sceneStateManager.GetOrCreateScene(defaultSceneId);

      Assert.True(sceneState.EntityMap.ContainsKey(lcaId), "LCA Frame should be mapped");
      Assert.True(sceneState.EntityMap.ContainsKey(cometId), "Comet Mesh should be mapped");

      var lcaEntity = sceneState.EntityMap[lcaId];
      var cometEntity = sceneState.EntityMap[cometId];

      Assert.Equal("TestComet_LCA", lcaEntity.Name);
      Assert.Equal("TestComet", cometEntity.Name);

      // Check Parent-Child Hierarchy constructed by NativeRuntimeService
      Assert.Contains(cometEntity, lcaEntity.Children);

      // Start simulation to catch any physics engine NaNs
      nativeRuntime.IsRunning = true;
      nativeRuntime.PlayScene(defaultSceneId);

      // Give the simulation some time to spawn the Comet and render a few frames
      await Task.Delay(250);

      // Wait for the next render frame
      var tcsFrame = new TaskCompletionSource<ulong>();
      WeakReferenceMessenger.Default.Register<RenderFrameReadyMessage>(
        this,
        (r, msg) =>
        {
          if (msg.SceneId == defaultSceneId && !tcsFrame.Task.IsCompleted)
            tcsFrame.TrySetResult(msg.RenderGeneration);
        }
      );

      var renderTaskId = await Task.WhenAny(tcsFrame.Task, Task.Delay(5000));
      Assert.True(tcsFrame.Task.IsCompleted, "Render frame should be produced");
      ulong generation = await tcsFrame.Task;

      ulong bufferSize = 512 * 512 * 4;
      IntPtr unmanagedBuffer = Marshal.AllocHGlobal((int)bufferSize);

      bool downloaded = await nativeRuntime.DownloadImageAsync(
        generation,
        unmanagedBuffer,
        (nuint)bufferSize
      );
      Assert.True(downloaded, "Image should be downloaded");

      bool hasColorPixel = false;
      bool hasNonZeroPixel = false;

      byte[] pixels = new byte[bufferSize];
      Marshal.Copy(unmanagedBuffer, pixels, 0, (int)bufferSize);

      for (int i = 0; i < pixels.Length; i += 4)
      {
        if (pixels[i + 3] > 0)
        { // Alpha > 0
          hasNonZeroPixel = true;
          if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0)
          { // RGB > 0
            hasColorPixel = true;
            break;
          }
        }
      }

      Marshal.FreeHGlobal(unmanagedBuffer);

      Assert.True(hasNonZeroPixel, "Image should not be completely empty (Alpha check)");
      Assert.True(hasColorPixel, "Image should have non-black colored pixels");
    }
    catch (DllNotFoundException)
    {
      // Skip test if native library is not found.
    }
    finally
    {
      nativeRuntime.Dispose();
    }
  }
}
