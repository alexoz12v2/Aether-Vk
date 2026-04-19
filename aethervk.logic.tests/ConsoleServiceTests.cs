using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class ConsoleServiceTests
{
  [Fact]
  public void Log_AddsMessageToCollection()
  {
    // Arrange
    var service = new ConsoleService();

    // Act
    service.Log("Test Message");

    // Assert
    Assert.Single(service.Messages);
    Assert.Contains("Test Message", service.Messages[0]);
  }

  [Fact]
  public void Clear_RemovesAllMessages()
  {
    // Arrange
    var service = new ConsoleService();
    service.Log("Msg 1");
    service.Log("Msg 2");

    // Act
    service.Clear();

    // Assert
    Assert.Empty(service.Messages);
  }
}
