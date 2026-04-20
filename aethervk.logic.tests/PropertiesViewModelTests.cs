using AetherVk.Logic.Models;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.Messaging;
using Xunit;

namespace AetherVk.Logic.Tests;

public class PropertiesViewModelTests
{
  [Fact]
  public void ReceivesMessage_SetsSelectedEntity()
  {
    // Arrange
    var propertiesVm = new PropertiesViewModel();
    var entity = new Entity(1, "TestEntity");

    // Act
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(entity));

    // Assert
    Assert.NotNull(propertiesVm.SelectedEntity);
    Assert.Equal(entity.Id, propertiesVm.SelectedEntity.Id);
  }

  [Fact]
  public void ReceivesMessage_ResetsFollowingState()
  {
    // Arrange
    var propertiesVm = new PropertiesViewModel();
    propertiesVm.IsFollowingEntity = true;
    var entity = new Entity(1, "TestEntity");

    // Act
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(entity));

    // Assert
    Assert.False(propertiesVm.IsFollowingEntity);
  }
}
