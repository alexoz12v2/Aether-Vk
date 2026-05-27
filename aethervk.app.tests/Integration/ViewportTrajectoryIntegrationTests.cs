using System;
using System.Linq;
using System.Threading.Tasks;
using Xunit;
using Avalonia.Headless.XUnit;
using AetherVk.Logic.Services;
using AetherVk.Logic.Models;
using AetherVk.Services;
using CommunityToolkit.Mvvm.Messaging;

namespace aethervk.app.tests.Integration;

public class ViewportTrajectoryIntegrationTests : IRecipient<AetherVk.Logic.Messages.RenderFrameReadyMessage>
{
    private ulong _latestRenderGeneration = 0;

    void IRecipient<AetherVk.Logic.Messages.RenderFrameReadyMessage>.Receive(AetherVk.Logic.Messages.RenderFrameReadyMessage message)
    {
        _latestRenderGeneration = message.RenderGeneration;
    }

    [AvaloniaFact]
    public async Task OrthographicTrajectoryPixelTest()
    {
        // 1. Setup Services
        var dispatcher = new AvaloniaUiThreadDispatcher();
        var console = new ConsoleService(dispatcher);
        console.Messages.CollectionChanged += (s, e) => {
            if (e.NewItems != null) {
                foreach(var collection in e.NewItems) {
                    if (collection is System.Collections.IEnumerable list) {
                        foreach(var str in list) {
                            Console.WriteLine(str);
                        }
                    } else {
                        Console.WriteLine(collection.ToString());
                    }
                }
            }
        };
        var breadcrumb = new BreadcrumbService(dispatcher);
        var sceneStateManager = new SceneStateManager();
        var runtimeService = new NativeRuntimeService(sceneStateManager, console, breadcrumb, dispatcher);
        var storage = new LocalStorageService();
        var horizonService = new HorizonJplService(console, breadcrumb, storage);

        WeakReferenceMessenger.Default.Register(this);

        // We explicitly initialize the runtime
        runtimeService.InitializeSimulationContext("Vulkan", null, false);
        
        // Wait for init
        int timeout = 50;
        while (!runtimeService.IsInitialized && timeout-- > 0)
        {
            await Task.Delay(100);
        }
        Assert.True(runtimeService.IsInitialized, "NativeRuntimeService failed to initialize.");

        // Get the first active scene
        await Task.Delay(500); // Give scene creation a moment
        var scene = sceneStateManager.AllScenes.FirstOrDefault(s => s.SceneId != 0);
        Assert.NotNull(scene);
        ulong sceneId = scene.SceneId;

        // 2. Setup Camera and Presentation Engine
        ulong peId = runtimeService.CreatePresentationEngine(800, 600, sceneId);
        Assert.True(peId != 0, "Failed to create presentation engine.");

        var orthoCameraId = runtimeService.AddOrthographicCamera(
            sceneId, 
            peId, 
            "TestOrthoCamera", 
            -10.0f, // left
            10.0f,  // right
            -10.0f, // bottom
            10.0f   // top
        );
        Assert.True(orthoCameraId != 0, "Failed to add orthographic camera.");

        // Hide the sky (Entity 5) to prevent "sky image absent" crash
        if (scene.EntityMap.TryGetValue(5, out var skyEntity)) {
            skyEntity.IsVisible = false;
        }

        // 3. Fetch Encke data
        var enckeData = await horizonService.GetPlanetDataAsync(
            "90000033", // 2P/Encke
            "@10", // Sun
            DateTime.UtcNow,
            DateTime.UtcNow.AddYears(1),
            "1 d"
        );
        Assert.NotNull(enckeData);

        // 4. Spawn Trajectory
        var trajEntityId = runtimeService.SpawnTrajectoryFromElements(
            sceneId,
            "Encke_Trajectory",
            enckeData.SemiMajorAxisAu,
            enckeData.Eccentricity,
            enckeData.Inclination,
            enckeData.AscendingNodeLongitude,
            enckeData.ArgumentOfPerifocus,
            100.0f // THICK line
        );
        Assert.True(trajEntityId != 0, "Failed to spawn trajectory.");

        // 5. Wait for a frame to render
        int frameTimeout = 50;
        while (_latestRenderGeneration == 0 && frameTimeout-- > 0)
        {
            await Task.Delay(100);
        }
        Assert.True(_latestRenderGeneration != 0, "Never received a rendered frame.");

        // 6. Download pixels
        nuint bufferSize = 800 * 600 * 4;
        IntPtr buffer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)bufferSize);
        try
        {
            bool downloaded = await runtimeService.DownloadImageAsync(_latestRenderGeneration, buffer, bufferSize);
            Assert.True(downloaded, "DownloadImageAsync returned false");
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(buffer);
            WeakReferenceMessenger.Default.Unregister<AetherVk.Logic.Messages.RenderFrameReadyMessage>(this);
        }
    }
}
