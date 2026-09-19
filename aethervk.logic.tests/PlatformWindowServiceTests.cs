using System.Runtime.InteropServices;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class PlatformWindowServiceTests
{
    [Fact]
    public void SetCursorPosition_DoesNotThrow()
    {
        // Generic test to ensure it doesn't crash on any platform
        var service = new PlatformWindowService();
        service.SetCursorPosition(100, 100);
    }

    [SkippableFact]
    public void SetCursorPosition_Linux_ExecutesWithoutError()
    {
        Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Linux), "Linux only behavior");
        
        var service = new PlatformWindowService();
        service.SetCursorPosition(100, 100);
    }

    [Fact]
    public void SetWindowInputPassthrough_ZeroHandle_DoesNotThrow()
    {
        // A zero/null handle must be a safe no-op on all platforms (guard at the top of each branch).
        var service = new PlatformWindowService();
        service.SetWindowInputPassthrough(0);
    }
}
