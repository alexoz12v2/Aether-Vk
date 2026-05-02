using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class ConsoleServiceTests
{
  public ConsoleServiceTests()
  {
  }

  [Fact]
  public void Log_AddsMessageToCollection()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a());
    // Arrange
    var service = new ConsoleService(dispatcherMock.Object);

    // Act
    service.Log("Test Message");
    System.Threading.Thread.Sleep(200);

    // Assert
    Assert.Single(service.Messages);
    Assert.Contains("Test Message", service.Messages[0]);
  }

  [Fact]
  public void Clear_RemovesAllMessages()
  {
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a());
    // Arrange
    var service = new ConsoleService(dispatcherMock.Object);
    service.Log("Msg 1");
    service.Log("Msg 2");
    System.Threading.Thread.Sleep(200);

    // Act
    service.Clear();

    // Assert
    Assert.Empty(service.Messages);
  }
}
