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
    string assetPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../assets/Comet.glb"));
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
    var (lcaId, cometId) = nativeRuntime.SpawnComet(defaultSceneId, modelId, "TestComet", 2f, 0f, 0f, 1f, 0f, 0f, 0f, 2.5f, 1e13f, 2);

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

    // Save image to disk so user can verify
    try
    {
        int width = 512;
        int height = 512;
        byte[] bmp = new byte[54 + pixels.Length];
        bmp[0] = 0x42; bmp[1] = 0x4D; // BM
        int size = bmp.Length;
        bmp[2] = (byte)(size); bmp[3] = (byte)(size >> 8); bmp[4] = (byte)(size >> 16); bmp[5] = (byte)(size >> 24);
        bmp[10] = 54; // offset
        bmp[14] = 40; // header size
        bmp[18] = (byte)(width); bmp[19] = (byte)(width >> 8);
        int h = -height; // top-down
        bmp[22] = (byte)(h); bmp[23] = (byte)(h >> 8); bmp[24] = (byte)(h >> 16); bmp[25] = (byte)(h >> 24);
        bmp[26] = 1; // planes
        bmp[28] = 32; // bpp

        for(int i = 0; i < pixels.Length; i+=4) {
            bmp[54 + i] = pixels[i+2];     // B
            bmp[54 + i + 1] = pixels[i+1]; // G
            bmp[54 + i + 2] = pixels[i];   // R
            bmp[54 + i + 3] = pixels[i+3]; // A
        }

        string outPath = "/Volumes/ExtData/alessioext/.gemini/antigravity-cli/brain/ea0ff329-3e19-4520-8409-cec2fe2f95e3/test_output.bmp";
        File.WriteAllBytes(outPath, bmp);
    }
    catch (Exception ex)
    {
        File.WriteAllText("/Volumes/ExtData/alessioext/.gemini/antigravity-cli/brain/ea0ff329-3e19-4520-8409-cec2fe2f95e3/test_output.log", "Failed to save image: " + ex.ToString());
    }

    Marshal.FreeHGlobal(unmanagedBuffer);
  }
}
