using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels.Debug;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;
using Microsoft.Reactive.Testing;

namespace AetherVk.Logic.Tests;

public class CameraMatrixDebugViewModelTests
{
    [Fact]
    public void DebugCameraState_CalledWhenExpanded()
    {
        var runtimeServiceMock = new Mock<INativeRuntimeService>();
        var schedulerProvider = new TestSchedulerProvider();
        
        runtimeServiceMock.Setup(r => r.CameraEntityId).Returns(42);
        
        double px = 1, py = 2, pz = 3;
        double rx = 4, ry = 5, rz = 6, rw = 7;
        
        // This setup is needed if it uses `out` parameters
        // Depending on Moq version, this could be:
        // runtimeServiceMock.Setup(r => r.DebugCameraState(42, out px, out py, out pz, out rx, out ry, out rz, out rw)).Returns(true);
        // We'll write it like this:
        runtimeServiceMock.Setup(r => r.DebugCameraState(It.IsAny<ulong>(), out px, out py, out pz, out rx, out ry, out rz, out rw)).Returns(true);
        
        using var vm = new CameraMatrixDebugViewModel(runtimeServiceMock.Object, schedulerProvider);
        vm.GetType().GetProperty("IsExpanded")?.SetValue(vm, true);
        
        schedulerProvider.Background.AdvanceBy(TimeSpan.FromMilliseconds(500).Ticks);
        schedulerProvider.MainThread.AdvanceBy(1);
        
        double AuToKm = 149_597_870.7;
        
        // Using a 1mm (0.001 km) tolerance to prevent floating point assertion flakiness
        Assert.Equal(1 * AuToKm, vm.PosX, 0.001);
        Assert.Equal(2 * AuToKm, vm.PosY, 0.001);
        Assert.Equal(3 * AuToKm, vm.PosZ, 0.001);
        Assert.Equal(4.0, vm.RotX, 4);
        Assert.Equal(5.0, vm.RotY, 4);
        Assert.Equal(6.0, vm.RotZ, 4);
        Assert.Equal(7.0, vm.RotW, 4);
    }

    /// Dust history line per age tier: live/capacity, ages present, band, building flag.
    [Fact]
    public void FormatDustHistory_ShowsEachTier()
    {
        const double Day = 86400.0;
        var text = CameraMatrixDebugViewModel.FormatDustHistory(new[]
        {
            new DustTierStats(1391, 16384, 0, 0.2 * Day, 0, 30 * Day, false),
            new DustTierStats(8000, 8192, 30 * Day, 240 * Day, 30 * Day, 240 * Day, true),
        });
        Assert.Equal("T0 1,391/16,384 0.0–0.2 d [0–30 d] building\nT1 8,000/8,192 30.0–240.0 d [30–240 d]", text);
        Assert.Equal(string.Empty, CameraMatrixDebugViewModel.FormatDustHistory(null));
    }
}
