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
    var runtimeService = new NativeRuntimeService();
    var outlineVm = new OutlineViewModel(runtimeService);
    var entity = new Entity(1, "TestEntity");

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
