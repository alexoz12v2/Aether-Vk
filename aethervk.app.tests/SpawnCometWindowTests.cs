using System.Collections.Generic;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Headless.XUnit;
using Xunit;

namespace AetherVk.AppTests;

public class SpawnCometWindowTests
{
  [AvaloniaFact]
  public void SpawnCometWindow_Should_Render_And_Initialize()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var timelineService = new TimelineService();
    var vm = new SpawnCometViewModel(new List<ImportedModelItem>(), horizonService, timelineService, breadcrumb);
    var window = new SpawnCometWindow { DataContext = vm };

    window.Show();

    Assert.NotNull(window);
    Assert.True(window.IsVisible);
  }

  [AvaloniaFact]
  public void CancelCommand_Should_Close_Window()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var timelineService = new TimelineService();
    var vm = new SpawnCometViewModel(new List<ImportedModelItem>(), horizonService, timelineService, breadcrumb);
    var window = new SpawnCometWindow { DataContext = vm };
    
    window.Show();
    bool closed = false;
    window.Closed += (sender, args) => closed = true;

    window.CancelCommand();

    Assert.True(closed);
  }


}
