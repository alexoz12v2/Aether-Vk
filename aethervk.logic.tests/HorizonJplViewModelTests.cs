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
}
