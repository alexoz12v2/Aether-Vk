#if DEBUG
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
        float rx = 4, ry = 5, rz = 6, rw = 7;
        
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
        Assert.Equal(4f, vm.RotX, 0.0001f);
        Assert.Equal(5f, vm.RotY, 0.0001f);
        Assert.Equal(6f, vm.RotZ, 0.0001f);
        Assert.Equal(7f, vm.RotW, 0.0001f);
    }
}
#endif
