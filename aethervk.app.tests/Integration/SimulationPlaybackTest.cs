using AetherVk.Logic.Services;
using Avalonia.Controls;
using Avalonia.Headless.XUnit;
using Avalonia.Headless;
using Avalonia.Media.Imaging;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Numerics;
using System.Threading.Tasks;
using Xunit;
using AetherVk.Views;
using AetherVk.Logic.ViewModels;

namespace AetherVk.AppTests.Integration;

public static class SimulationTestHarness
{
    public static async Task RunPlaybackScenarioAsync(
        int numParticleSystems,
        TimeSpan durationToWait,
        Action<INativeRuntimeService, WriteableBitmap> onTimeAdvancedAssert)
    {
        var host = AetherVk.App.Host
          ?? throw new InvalidOperationException("DI Host not initialized.");
        await using var scope = host.Services.CreateAsyncScope();
        
        var timelineService = scope.ServiceProvider.GetRequiredService<TimelineService>();
        var runtime = scope.ServiceProvider.GetRequiredService<INativeRuntimeService>();

        for (int i = 0; i < numParticleSystems; i++)
        {
            var model = new ParticleSystemModel(1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f);
            var jet = new ParticleSystemJet(0f, 0f, 0f, 10f, 1f, new Vector4(1,1,1,1), 1f, 123u);
            
            if (i == 0)
            {
                runtime.AddFirstParticleSystem(model, jet, out _);
            }
            else
            {
                runtime.AddParticleSystem(model, jet, out _);
            }
        }

        timelineService.Play(1);

        await Task.Delay(durationToWait);
        
        var viewportVm = scope.ServiceProvider.GetRequiredService<Viewport3DViewModel>();
        var view       = new Viewport3DView { DataContext = viewportVm };
        var window     = new Window { Content = view, Width = 800, Height = 600 };
        window.Show();

        await Task.Delay(200);

        using var bitmap = window.CaptureRenderedFrame();
        
        onTimeAdvancedAssert(runtime, bitmap);

        timelineService.SnapshotRestore();
        
        window.Close();

        Assert.False(timelineService.IsSimulationRunningValue);
    }

    public static bool HasNonBlackPixels(WriteableBitmap bitmap)
    {
        unsafe
        {
            using var buf = bitmap.Lock();
            int width = bitmap.PixelSize.Width;
            int height = bitmap.PixelSize.Height;
            int bytesPerPixel = 4;

            byte* ptr = (byte*)buf.Address;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int index = y * buf.RowBytes + x * bytesPerPixel;
                    byte b = ptr[index];
                    byte g = ptr[index + 1];
                    byte r = ptr[index + 2];
                    if (r > 0 || g > 0 || b > 0)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}

public class SimulationPlaybackTest
{
    [AvaloniaFact]
    public async Task SimulationPlay_WithOneParticleSystem_RendersAndRestores()
    {
        await SimulationTestHarness.RunPlaybackScenarioAsync(
            numParticleSystems: 1,
            durationToWait: TimeSpan.FromSeconds(2),
            onTimeAdvancedAssert: (runtime, bitmap) =>
            {
                Assert.True(SimulationTestHarness.HasNonBlackPixels(bitmap), "Should render particles.");
            });
    }

    [AvaloniaFact]
    public async Task SimulationPlay_WithZeroSystems_DoesNotCrash()
    {
        await SimulationTestHarness.RunPlaybackScenarioAsync(
            numParticleSystems: 0,
            durationToWait: TimeSpan.FromSeconds(1),
            onTimeAdvancedAssert: (runtime, bitmap) =>
            {
                Assert.True(true);
            });
    }
}
