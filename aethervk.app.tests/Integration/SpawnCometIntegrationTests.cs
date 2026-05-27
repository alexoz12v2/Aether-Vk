using System.Collections.Generic;
using System.Threading.Tasks;
using Xunit;
using Avalonia.Headless.XUnit;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Models;

namespace aethervk.app.tests.Integration;

public class SpawnCometIntegrationTests
{
    [AvaloniaFact]
    public async Task ViewModel_Integration_Flow_Test()
    {
        // Setup real services
        var dispatcher = new Moq.Mock<IUiThreadDispatcher>().Object;
        var console = new ConsoleService(dispatcher);
        var breadcrumb = new BreadcrumbService(dispatcher);
        var storage = new LocalStorageService();
        var horizonService = new HorizonJplService(console, breadcrumb, storage);
        var timelineService = new TimelineService();

        var vm = new SpawnCometViewModel(new List<ImportedModelItem>(), horizonService, timelineService, breadcrumb);

        // Act
        // 1. Fetch Comets
        await vm.FetchCometsCommand.ExecuteAsync(null);
        
        // Let's assume we find Halley's comet or just manually set it
        // We can just construct a dummy SelectedComet with the real PrimaryDesignation so we don't have to search it.
        vm.SelectedComet = new CometSearchResult { PrimaryDesignation = "90000033", Name = "Halley" };

        // 2. Fetch Orbit Data
        await vm.FetchOrbitDataCommand.ExecuteAsync(null);

        // Assert
        Assert.NotNull(vm.FetchedOrbitData);
        Assert.True(vm.CometRadiusKm > 0, "ViewModel CometRadiusKm should be populated from JPL API");
        Assert.True(vm.FetchedOrbitData.MassKg > 0, "ViewModel FetchedOrbitData.MassKg should be populated from JPL API");
        Assert.False(vm.IsFetchingHorizonData, "Loading flag should be reset after fetch completes");
    }
}
