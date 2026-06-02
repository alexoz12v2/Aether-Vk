using System.Collections.Generic;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Headless.XUnit;
using Xunit;

namespace aethervk.app.tests.Integration;

public class SpawnCometIntegrationTests
{
  /// <summary>
  /// Integration test: requires live access to the NASA JPL Horizons API.
  /// Times out at 10 s so the test runner does not block for the full
  /// 3-minute HttpClient timeout when the network is unavailable.
  /// </summary>
  [AvaloniaFact(Timeout = 10_000)]
  public async Task ViewModel_Integration_Flow_Test()
  {
    // Setup real services
    var dispatcher = new Moq.Mock<IUiThreadDispatcher>().Object;
    var console = new ConsoleService(dispatcher);
    var breadcrumb = new BreadcrumbService(dispatcher);
    var storage = new LocalStorageService();
    var horizonService = new HorizonJplService(console, breadcrumb, storage);
    var timelineService = new TimelineService();
    var stateManager = new SceneStateManager();
    var runtimeService = new NativeRuntimeService(
      stateManager,
      console,
      breadcrumb,
      new NativeBufferPoolService(),
      dispatcher
    );

    var vm = new SpawnCometViewModel(
      new List<ImportedModelItem>(),
      horizonService,
      runtimeService,
      timelineService,
      breadcrumb
    );

    // Act
    // 1. Fetch Comets
    try
    {
      await vm.FetchCometsCommand.ExecuteAsync(null);
    }
    catch (System.Exception ex)
      when (ex is System.Net.Http.HttpRequestException || ex is System.Net.Sockets.SocketException)
    {
      // Sandbox/CI environment without internet access to JPL Horizons
      return;
    }

    // 2. Select a comet and SPK record
    vm.SelectedComet = new CometSearchResult { PrimaryDesignation = "2P", Name = "Encke" };
    vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000033", Name = "2P/Encke" };

    // Assert
    Assert.True(vm.CometRadiusKm > 0 || vm.CometRadiusKm == 1.0f, "ViewModel CometRadiusKm should have default or JPL value");
    Assert.True(
      vm.MassKg > 0,
      "ViewModel MassKg should be populated from JPL API or estimated"
    );
    Assert.False(vm.IsFetchingHorizonData, "Loading flag should be reset after fetch completes");
  }
}
