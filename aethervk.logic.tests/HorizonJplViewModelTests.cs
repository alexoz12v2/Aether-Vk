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
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock
      .Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());
    var localStorageMock = new Moq.Mock<ILocalStorageService>();
    var consoleService = new ConsoleService(dispatcherMock.Object);
    var breadcrumbService = new BreadcrumbService(dispatcherMock.Object);
    var service = new HorizonJplService(consoleService, breadcrumbService);

    // Act
    var vm = new HorizonJplViewModel(service, localStorageMock.Object, breadcrumbService);

    // Assert
    Assert.Equal("Horizon JPL", vm.Title);
  }
}
