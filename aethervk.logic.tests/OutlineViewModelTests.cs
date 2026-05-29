using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.Messaging;
using Xunit;

namespace AetherVk.Logic.Tests;

public class OutlineViewModelTests
{
  [Fact]
  public void SelectionChanged_SendsMessage()
  {
    // Arrange
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock
      .Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());
    var runtimeService = new NativeRuntimeService(
      stateManager,
      new ConsoleService(dispatcherMock.Object),
      new BreadcrumbService(dispatcherMock.Object),
      new AetherVk.Logic.Services.NativeBufferPoolService(),
      dispatcherMock.Object
    );
    var outlineVm = new OutlineViewModel(1, runtimeService, stateManager);
    var entity = new Entity(1, 100, "TestEntity");

    EntitySelectedMessage? receivedMessage = null;
    WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(
      this,
      (r, m) => receivedMessage = m
    );

    // Act
    outlineVm.SelectedEntity = entity;

    // Assert
    Assert.NotNull(receivedMessage);
    Assert.NotNull(receivedMessage.SelectedEntity);
    Assert.Equal(entity.Id, receivedMessage.SelectedEntity.Id);

    WeakReferenceMessenger.Default.Unregister<EntitySelectedMessage>(this);
  }
}
