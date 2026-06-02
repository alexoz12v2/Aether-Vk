using System.Collections.Generic;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Headless.XUnit;
using Xunit;

namespace AetherVk.AppTests;

public class SpawnCometWindowTests
{
  private static SpawnCometViewModel CreateVm()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var console = new ConsoleService(dispatcherMock.Object);
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var storage = new Moq.Mock<ILocalStorageService>();
    var horizonService = new HorizonJplService(console, breadcrumb, storage.Object);
    var timelineService = new TimelineService();
    var stateManager = new SceneStateManager();
    var runtimeService = new NativeRuntimeService(
      stateManager,
      console,
      breadcrumb,
      new NativeBufferPoolService(),
      dispatcherMock.Object
    );
    return new SpawnCometViewModel(
      new List<ImportedModelItem>(),
      horizonService,
      runtimeService,
      timelineService,
      breadcrumb
    );
  }

  [AvaloniaFact]
  public void SpawnCometWindow_Should_Render_And_Initialize()
  {
    var vm = CreateVm();
    var window = new SpawnCometWindow { DataContext = vm };

    window.Show();

    Assert.NotNull(window);
    Assert.True(window.IsVisible);
  }

  [AvaloniaFact]
  public void CancelCommand_Should_Close_Window()
  {
    var vm = CreateVm();
    var window = new SpawnCometWindow { DataContext = vm };

    window.Show();
    bool closed = false;
    window.Closed += (sender, args) => closed = true;

    window.CancelCommand();

    Assert.True(closed);
  }
}
