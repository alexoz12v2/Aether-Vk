using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

public class HorizonJplViewModelTests
{
  [Fact]
  public void Initialization_SetsDefaultValues()
  {
    // Arrange
    var consoleService = new ConsoleService();
    var breadcrumbService = new BreadcrumbService();
    var service = new HorizonJplService(consoleService, breadcrumbService);

    // Act
    var vm = new HorizonJplViewModel(service);

    // Assert
    Assert.Equal("499", vm.Command);
    Assert.Equal("500@399", vm.Center);
    Assert.NotNull(vm.Data);
    Assert.Empty(vm.Data);
    Assert.Equal("Horizon JPL", vm.Title);
  }

  [Fact]
  public void SelectedCometChanged_UpdatesCommand()
  {
    // Arrange
    var consoleService = new ConsoleService();
    var breadcrumbService = new BreadcrumbService();
    var service = new HorizonJplService(consoleService, breadcrumbService);
    var vm = new HorizonJplViewModel(service);

    // Act: Select a comet with an SPK-ID (typically at index 3)
    string[] mockComet = new[] { "12P/Pons-Brooks", "1812-07-21", "2024-04-21", "90000033" };
    vm.SelectedComet = mockComet;

    // Assert
    Assert.Equal("DES=90000033; CAP", vm.Command);
  }
}
