using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class NativeRuntimeServiceTests
{
  [Fact]
  public void GetAvailableRenderDevices_ReturnsMockData()
  {
    var service = new NativeRuntimeService();
    var devices = service.GetAvailableRenderDevices();
    Assert.Contains("Vulkan", devices);
    Assert.Contains("Direct3D12", devices);
    Assert.Contains("Metal", devices);
  }

  [Fact]
  public void GetAvailableKernels_ReturnsMockData()
  {
    var service = new NativeRuntimeService();
    var kernels = service.GetAvailableKernels();
    Assert.Contains("CUDA", kernels);
    Assert.Contains("Vulkan Compute", kernels);
  }

  [Fact]
  public void CreateScene_PopulatesRootEntities()
  {
    // Arrange
    var service = new NativeRuntimeService();

    // Act
    service.CreateScene();

    // Assert
    Assert.NotEmpty(service.RootEntities);
    var root = service.GetEntityByName("root");
    Assert.NotNull(root);
    Assert.Equal("root", root.Name);

    var camera = service.GetEntityByName("camera");
    Assert.NotNull(camera);
    Assert.Contains(camera.Components, c => c is CameraComponent);
  }

  [Fact]
  public void SpawnEntity_AddsToParentOrRoot()
  {
    // Arrange
    var service = new NativeRuntimeService();
    service.RootEntities.Clear();
    service.SpawnEntity("dummy"); // uses ID 1

    // Act
    var root = service.SpawnEntity("my_root"); // uses ID 2, gets added to root
    var child = service.SpawnEntity("child", root);

    // Assert
    Assert.Contains(root, service.RootEntities);
    Assert.Contains(child, root.Children);
    Assert.Equal("child", child.Name);
  }

  [Fact]
  public void GetActiveCameraId_ReturnsFirstCameraIfNoneActive()
  {
    // Arrange
    var service = new NativeRuntimeService();
    service.CreateScene(); // Creates one camera by default, IsActiveCamera is false

    // Act
    var activeId = service.GetActiveCameraId();
    var cameraEntity = service.GetEntityByName("camera");

    // Assert
    Assert.NotNull(cameraEntity);
    Assert.Equal(cameraEntity.Id, activeId);
  }

  [Fact]
  public void GetActiveCameraId_ReturnsActiveCamera()
  {
    // Arrange
    var service = new NativeRuntimeService();
    var root = service.SpawnEntity("root");

    var cam1 = service.SpawnEntity("cam1", root);
    cam1.Components.Add(new CameraComponent { IsActiveCamera = false });

    var cam2 = service.SpawnEntity("cam2", root);
    cam2.Components.Add(new CameraComponent { IsActiveCamera = true });

    // Act
    var activeId = service.GetActiveCameraId();

    // Assert
    Assert.Equal(cam2.Id, activeId);
  }
}
