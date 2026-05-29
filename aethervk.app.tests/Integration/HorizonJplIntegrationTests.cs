using System;
using System.IO;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Moq;
using Xunit;

namespace aethervk.app.tests.Integration;

public class HorizonJplIntegrationTests
{
  [Fact]
  public async Task DownloadSpkByIdAsync_Downloads_Real_SPK_Test()
  {
    var dispatcherMock = new Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new LocalStorageService();
    var service = new HorizonJplService(console, breadcrumb, storage);

    string tempDir = Path.Combine(Path.GetTempPath(), "aethervk_tests", Guid.NewGuid().ToString());
    Directory.CreateDirectory(tempDir);
    string spkPath = Path.Combine(tempDir, "halley.bsp");

    string? resultPath = await service.DownloadSpkByIdAsync(
      "1P",
      "90000033",
      spkPath,
      "2026-01-01",
      "2026-02-01"
    );
    Assert.NotNull(resultPath);
  }
}
