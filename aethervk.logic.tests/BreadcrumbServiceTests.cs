using System.Collections.Generic;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class BreadcrumbServiceTests
{
  [Fact]
  public async Task ShowMessageAsync_EmitsAddedThenRemovedEvents()
  {
    // Arrange
    var service = new BreadcrumbService();
    var received = new List<BreadcrumbEvent>();
    using var _ = service.Events.Subscribe(ev => received.Add(ev));

    // Act
    var task = service.ShowMessageAsync("Title", "Content", System.TimeSpan.FromMilliseconds(50));

    // Before delay finishes, Added should have been emitted synchronously
    Assert.Single(received);
    Assert.IsType<BreadcrumbEvent.Added>(received[0]);
    Assert.Equal("Title", ((BreadcrumbEvent.Added)received[0]).Message.Title);

    await task;

    // After delay, Removed should also be emitted
    Assert.Equal(2, received.Count);
    Assert.IsType<BreadcrumbEvent.Removed>(received[1]);
  }
}
