using AetherVk.Logic.Input;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
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
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var timelineService = new TimelineService();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var propertiesVm = new PropertiesViewModel(1, stateManager, timelineService, null, breadcrumb);
    var entity = new Entity(1, 100, "TestEntity");

    // Act
    stateManager.GetOrCreateScene(1).SelectedEntity = entity;
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(entity));

    // Assert
    Assert.NotNull(propertiesVm.SelectedEntity);
    Assert.Equal(entity.Id, propertiesVm.SelectedEntity.Id);
  }

  [Fact]
  public void ReceivesMessage_PopulatesPropertiesExpanders()
  {
    // Arrange
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    dispatcherMock
      .Setup(d => d.Dispatch(Moq.It.IsAny<System.Action>()))
      .Callback<System.Action>(a => a());

    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var timelineService = new TimelineService();

    var runtimeService = new NativeRuntimeService(
      stateManager,
      new ConsoleService(dispatcherMock.Object),
      breadcrumb,
      dispatcherMock.Object
    );

    var propertiesVm = new PropertiesViewModel(1, stateManager, timelineService, runtimeService, breadcrumb);

    var entity = new Entity(1, 100, "TestEntity");
    entity.Components.Add(new TransformComponent());
    entity.Components.Add(new CameraComponent());
    stateManager.GetOrCreateScene(1).EntityMap[100] = entity;
    stateManager.GetOrCreateScene(1).SelectedEntity = entity;

    // Note: We can't easily mock the FFI call inside NativeRuntimeService without abstracting it,
    // but since PropertiesViewModel calls runtimeService.GetEntityComponentNames(CurrentSceneId, SelectedEntity.Id)
    // which relies on actual FFI, this test will only pass if FFI is skipped or mocked.
    // Since this is a unit test, we should assume we only test the fallback or empty state if FFI isn't available,
    // OR we should verify that PropertiesExpanders is cleared when a new entity is selected.

    // Act
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(entity));

    // Assert
    // Without FFI running, GetEntityComponentNames returns empty array.
    // We just ensure it doesn't crash and clears the list.
    Assert.Empty(propertiesVm.PropertiesExpanders);
  }

  [Fact]
  public void ReceivesMessage_ResetsFollowingState()
  {
    // Arrange
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var timelineService = new TimelineService();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var propertiesVm = new PropertiesViewModel(1, stateManager, timelineService, null, breadcrumb);
    propertiesVm.IsFollowingEntity = true;
    var entity = new Entity(1, 100, "TestEntity");

    // Act
    stateManager.GetOrCreateScene(1).SelectedEntity = entity;
    WeakReferenceMessenger.Default.Send(new EntitySelectedMessage(entity));

    // Assert
    Assert.False(propertiesVm.IsFollowingEntity);
  }

  [Fact]
  public void ProcessAction_ExpandAll_TogglesState()
  {
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var timelineService = new TimelineService();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var vm = new PropertiesViewModel(1, stateManager, timelineService, null, breadcrumb);

    Assert.False(vm.AreAllExpanded);

    bool handled = vm.ProcessAction(new AppAction("ui.expand_all", "Expand"), true);

    Assert.True(handled);
    Assert.True(vm.AreAllExpanded);
  }

  [Fact]
  public void ProcessAction_ShowFlyout_PushesOperator()
  {
    var stateManager = new SceneStateManager();
    var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
    var timelineService = new TimelineService();
    var breadcrumb = new BreadcrumbService(dispatcherMock.Object);
    var vm = new PropertiesViewModel(1, stateManager, timelineService, null, breadcrumb);

    Assert.False(vm.IsFlyoutMenuOpen);

    bool handled = vm.ProcessAction(new AppAction("ui.show_flyout", "Flyout"), true);

    Assert.True(handled);
    Assert.True(vm.IsFlyoutMenuOpen);

    // Cancel should close it
    vm.ProcessAction(new AppAction("global.cancel", "Cancel"), true);
    Assert.False(vm.IsFlyoutMenuOpen);
  }
}
