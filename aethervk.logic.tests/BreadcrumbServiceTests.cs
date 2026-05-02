using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class BreadcrumbServiceTests
{
  [Fact]
  public async Task ShowMessageAsync_AddsAndRemovesMessage()
  {
    // Arrange
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>())).Callback<System.Action>(a => a());
    var service = new BreadcrumbService(dispatcherMock.Object);

    // Act
    var task = service.ShowMessageAsync("Title", "Content", System.TimeSpan.FromMilliseconds(50));

    // Before delay finishes, it should be in the collection
    Assert.Single(service.Messages);
    Assert.Equal("Title", service.Messages[0].Title);
    Assert.Equal("Content", service.Messages[0].Content);

    await task;

    // Assert - After delay, it should be removed
    Assert.Empty(service.Messages);
  }
}
