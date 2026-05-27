using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Xunit;
using Moq;
using AetherVk.Logic.Services;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.Messages;
using System.Collections.Generic;

namespace aethervk.app.tests;

public class SpawnCometFlowE2ETests
{
  [Fact]
  public async Task SpawnComet_E2E_Simulation_Test()
  {
    var console = new ConsoleService(new Mock<IUiThreadDispatcher>().Object);
    var dispatcherMock = new Mock<IUiThreadDispatcher>();
    dispatcherMock.Setup(d => d.DispatchAsync(It.IsAny<Func<Task>>())).Returns<Func<Task>>(f => f());
    dispatcherMock.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());

    var sceneStateManager = new SceneStateManager();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var nativeRuntime = new NativeRuntimeService(sceneStateManager, console, breadcrumb, dispatcherMock.Object);

    // Initialize headless
    nativeRuntime.InitializeSimulationContext("Vulkan", null, false);
    
    // Create Default Scene
    ulong defaultSceneId = nativeRuntime.CreateScene(true);
    
    // Setup Presentation Engine
    ulong presentationEngineId = nativeRuntime.CreatePresentationEngine(512, 512, defaultSceneId);
    ulong cameraId = nativeRuntime.AddPerspectiveCamera(defaultSceneId, presentationEngineId, "MainCamera", 45f, 0.001f, 1000f);
    
    // Setup Horizon API Mock (even if we just call SpawnComet directly, the test tests the whole app infrastructure)
    var mockStorage = new Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, mockStorage.Object);
    // Note: since we are headless, we bypass the ViewModels and test the runtime services directly.
    
    // Load a mock GLTF model (we need a valid file so the C++ engine doesn't crash on load)
    // There is an asset comet in aethervk.core/assets/Comet.glb
    string assetPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../aethervk.core/assets/Comet.glb"));
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

    // Spawn Comet
    var (lcaId, cometId) = nativeRuntime.SpawnComet(defaultSceneId, modelId, "TestComet", 0f, 0f, 0f, 1f, 0f, 0f, 0f, 2.5f, 1e13f, 2);

    // Start simulation to catch any physics engine NaNs
    nativeRuntime.IsRunning = true;
    nativeRuntime.PlayScene(defaultSceneId);

    // Wait for a render frame
    var tcsFrame = new TaskCompletionSource<ulong>();
    WeakReferenceMessenger.Default.Register<RenderFrameReadyMessage>(this, (r, msg) => 
    {
      if (msg.SceneId == defaultSceneId && !tcsFrame.Task.IsCompleted)
        tcsFrame.TrySetResult(msg.RenderGeneration);
    });

    // To ensure we get a render frame, we might need to tick the timeline or the native runtime automatically ticks?
    // The native runtime automatically dispatches render frames if there's a presentation engine.
    var renderTaskId = await Task.WhenAny(tcsFrame.Task, Task.Delay(5000));
    
    Assert.True(tcsFrame.Task.IsCompleted, "Render frame should be produced");
    ulong generation = tcsFrame.Task.Result;
    
    // Download image
    nuint bufferSize = (nuint)(512 * 512 * 4);
    IntPtr unmanagedBuffer = Marshal.AllocHGlobal((int)bufferSize);
    
    bool downloaded = await nativeRuntime.DownloadImageAsync(generation, unmanagedBuffer, bufferSize);
    Assert.True(downloaded, "Image should be downloaded");
    
    // Assert image is not completely empty (alpha = 0)
    byte[] pixels = new byte[bufferSize];
    Marshal.Copy(unmanagedBuffer, pixels, 0, (int)bufferSize);
    
    bool hasNonZeroPixel = false;
    bool hasColorPixel = false;
    for(int i = 0; i < pixels.Length; i += 4) {
        if (pixels[i+3] > 0) { // Alpha > 0
            hasNonZeroPixel = true;
            if (pixels[i] > 0 || pixels[i+1] > 0 || pixels[i+2] > 0) { // RGB > 0
                hasColorPixel = true;
                break;
            }
        }
    }
    Assert.True(hasNonZeroPixel, "Image should not be completely empty (Alpha check)");
    Assert.True(hasColorPixel, "Image should have non-black colored pixels");

    Marshal.FreeHGlobal(unmanagedBuffer);
  }
}
