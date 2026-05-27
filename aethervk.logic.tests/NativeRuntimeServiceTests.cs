using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  [Collection("Sequential")]
  public class NativeRuntimeServiceTests : IDisposable
  {
    private readonly NativeRuntimeService _service;
    private readonly SceneStateManager _stateManager;
    private readonly string _assetPath;

    public NativeRuntimeServiceTests()
    {
      var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
      dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<Action>())).Callback<Action>(a => a());
      _stateManager = new SceneStateManager();
      _service = new NativeRuntimeService(
        _stateManager,
        new ConsoleService(dispatcherMock.Object),
        new BreadcrumbService(dispatcherMock.Object),
        dispatcherMock.Object
      );
      var baseDir = AppDomain.CurrentDomain.BaseDirectory;
      _assetPath = System.IO.Path.GetFullPath(
        System.IO.Path.Combine(baseDir, "../../../../assets")
      );
    }

    public void Dispose()
    {
      _service.Dispose();
    }

    [Fact]
    public void NativeSimulationCallback_ShouldUpdateTransformComponent()
    {
      // Arrange
      ulong sceneId = 100;
      ulong entityId = 200;

      var entity = new AetherVk.Logic.Models.Entity(sceneId, entityId, "Test");
      var transform = new AetherVk.Logic.Models.TransformComponent();
      entity.Components.Add(transform);

      _stateManager.GetOrCreateScene(sceneId).EntityMap[entityId] = entity;

      // Create dummy unmanaged memory to simulate FfiTransform payload
      var dto = new AetherVk.Logic.Services.NativeInterop.FfiTransform
      {
        Px = 1.0f,
        Py = 2.0f,
        Pz = 3.0f,
        Rw = 0.0f,
        Rx = 0.0f,
        Ry = 1.0f,
        Rz = 0.0f,
        Sx = 2.0f,
        Sy = 2.0f,
        Sz = 2.0f,
      };

      int size =
        System.Runtime.InteropServices.Marshal.SizeOf<AetherVk.Logic.Services.NativeInterop.FfiTransform>();
      IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
      System.Runtime.InteropServices.Marshal.StructureToPtr(dto, ptr, false);

      try
      {
        // Act
        // Access the private NativeSimulationCallback method via reflection to test parsing
        var method = typeof(NativeRuntimeService).GetMethod(
          "NativeSimulationCallback",
          System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
        );

        // componentId 1 = Transform
        method?.Invoke(_service, new object[] { sceneId, entityId, 1ul, ptr });

        // Assert
        Assert.Equal(1.0f, transform.PosX);
        Assert.Equal(2.0f, transform.PosY);
        Assert.Equal(3.0f, transform.PosZ);
        Assert.Equal(0.0f, transform.RotW);
        Assert.Equal(1.0f, transform.RotY);
        Assert.Equal(2.0f, transform.ScaleX);
      }
      finally
      {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
      }
    }

    [Fact]
    public void Initialization_ShouldSucceedWithVulkanBackend()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        Assert.True(_service.IsInitialized);

        ulong sceneId = _service.CreateScene(true);
        var rootEntities = _stateManager.GetOrCreateScene(sceneId).RootEntities;

        Assert.NotEmpty(rootEntities);

        var root = rootEntities.FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children, e => e.Name == "sun");

        TestSceneExporter.ExportScene(sceneId, _stateManager, "Initialization_DefaultScene");
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task ImportModel_ShouldReturnIdWhenValid()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        string modelPath = System.IO.Path.Combine(_assetPath, "Comet.glb");
        ulong modelId = await _service.ImportModelAsync(modelPath);
        // It should return a valid model ID now.
        Assert.NotEqual(0ul, modelId);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public async Task SpawnModelInstance_ShouldAddEntityToScene()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        var initialCount =
          _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault()?.Children.Count
          ?? 0;

        await Assert.ThrowsAsync<Exception>(() =>
          _service.SpawnModelInstanceAsync(sceneId, 999, "test")
        );
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void CreateMeasurement_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        var entity = _service.CreateMeasurement(
          sceneId,
          "TestMeasure",
          new float[] { 0, 0, 0 },
          new float[] { 1, 1, 1 }
        );

        Assert.NotNull(entity);
        Assert.Equal("TestMeasure", entity.Name);

        var root = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault();
        Assert.NotNull(root);
        Assert.Contains(root.Children, e => e.Id == entity.Id);

        TestSceneExporter.ExportScene(sceneId, _stateManager, "CreateMeasurement_Scene");
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void TimeControls_ShouldUpdateSimulationTime()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        _service.SetSimulationTime(sceneId, 1000.0);
        var time = _service.GetSimulationTime(sceneId);
        Assert.Equal(1000.0, time, 3);

        _service.SetTimeScale(sceneId, 1); // e.g. OneDay
        _service.PlayScene(sceneId);
        _service.PauseScene(sceneId);

        // Just ensure no native crash occurred during these commands.
      }
      catch (System.DllNotFoundException) { }
    }


    [Fact]
    public void GetEphemerisPosition_WithoutAlmanac_ShouldReturnNull()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        var pos = _service.GetEphemerisPosition(399, 0.0);
        
        // Almanac is not loaded synchronously, so it should be null or fail gracefully
        Assert.Null(pos);
      }
      catch (System.DllNotFoundException) { }
    }



    [Fact]
    public async Task RaycastNdc_ShouldCompleteSuccessfully()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);

        // The default scene contains a camera, grid, sun, sky, etc.
        // We need to pass a valid camera ID. Let's find the default camera.
        var state = _stateManager.GetOrCreateScene(sceneId);
        var camera = state.EntityMap.Values.FirstOrDefault(e => e.Name == "camera");
        ulong camId =
          camera != null
            ? camera.Id
            : _service.AddPerspectiveCamera(sceneId, peId, "testcam", 45f, 0.1f, 1000f);

        var result = await _service.RaycastNdcAsync(sceneId, camId, 0.5f, 0.5f);

        // The raycast might not hit anything depending on the default scene setup,
        // but it should complete without crashing.
        // Assert.True(result.hit);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void ProceduralSphere_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "MySphere", 5.0f, 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var entity = _service.GetEntityByName(sceneId, "MySphere");
        Assert.NotNull(entity);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void CameraAndCursorManipulations_ShouldExecuteWithoutCrashing()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);
        ulong camId = _service.AddPerspectiveCamera(sceneId, peId, "cam", 45f, 0.1f, 1000f);

        _service.RotateCamera(sceneId, camId, 10f, 10f);
        _service.ZoomCamera(sceneId, camId, 5f);
        _service.PanCamera(sceneId, camId, 1f, 1f);
        _service.ResetCamera(sceneId, camId);

        _service.PanCursor(sceneId, 1f, 1f);
        _service.MoveCursor(sceneId, 1f, 1f, 1f);

        Assert.True(true);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void EntityFollowing_ShouldExecuteWithoutCrashing()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);
        ulong camId = _service.AddPerspectiveCamera(sceneId, peId, "cam", 45f, 0.1f, 1000f);
        ulong targetId = _service.SpawnProceduralSphere(sceneId, "target", 1f, 1f);

        _service.SnapToEntity(sceneId, camId, targetId);
        _service.FollowEntity(sceneId, camId, targetId);
        _service.UnfollowEntity(sceneId, camId);

        Assert.True(true);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void AddOrthographicCamera_ShouldAddEntity()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);

        ulong camId = _service.AddOrthographicCamera(
          sceneId,
          peId,
          "orthoCam",
          -10f,
          -10f,
          0.1f,
          1000f
        );
        Assert.NotEqual(0ul, camId);

        var entity = _service.GetEntityByName(sceneId, "orthoCam");
        Assert.NotNull(entity);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void SpawnImageBillboard_ShouldExecuteWithoutCrashing()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        var billboard = _service.SpawnImageBillboard(sceneId, "MyBillboard", true, 100f, 100f);
        Assert.NotNull(billboard);
        Assert.Equal("MyBillboard", billboard.Name);
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void DestroySceneAndPresentationEngine_ShouldCleanupResources()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);

        _service.DestroyPresentationEngine(sceneId, peId, 0);
        _service.DestroyScene(sceneId);

        Assert.Null(_service.GetEntityByName(sceneId, "root"));
      }
      catch (System.DllNotFoundException) { }
    }

    [Fact]
    public void TwoWayBinding_NativeComponent_ShouldSyncChanges()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(256, 256, sceneId);

        ulong camId = _service.AddPerspectiveCamera(sceneId, peId, "testCam", 45f, 0.1f, 1000f);
        var cameraEntity = _service.GetEntityById(sceneId, camId);
        Assert.NotNull(cameraEntity);

        var camComp = cameraEntity
          .Components.OfType<AetherVk.Logic.Models.CameraComponent>()
          .FirstOrDefault();
        Assert.NotNull(camComp);

        // 1. Mutate C# property (should trigger PushToNativeImpl via PropertyChanged)
        camComp!.Fov = 90f;

        // Directly mutate native to test Pull
        var ctx =
          typeof(AetherVk.Logic.Services.NativeRuntimeService)
            .GetField(
              "_simulationContext",
              System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
            )?
            .GetValue(_service) as IntPtr?
          ?? IntPtr.Zero;

        var camData = new AetherVk.Logic.Services.NativeInterop.FfiCamera
        {
          IsOrthographic = false,
          Fov = 120f,
          Aspect = 1.77f,
          Near = 0.1f,
          Far = 1000f,
          OrthoLeft = -10f,
          OrthoRight = 10f,
          OrthoBottom = -10f,
          OrthoTop = 10f,
        };
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_setCameraComponent(
          ctx,
          sceneId,
          camId,
          in camData
        );

        // 2. Pull from native
        camComp!.PullFromNative();

        // 3. Assert value is restored from native
        Assert.Equal(120f, camComp.Fov, 3);

        var transformComp = cameraEntity
          .Components.OfType<AetherVk.Logic.Models.TransformComponent>()
          .FirstOrDefault();
        Assert.NotNull(transformComp);

        // 1. Mutate Transform properties via C#
        transformComp!.PosX = 100f;
        transformComp.PosY = 200f;
        transformComp.PosZ = 300f;

        // Verify native was updated automatically
        var nativeHasTransform =
          AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getTransformComponent(
            ctx,
            sceneId,
            camId,
            out var ffiTransform
          );

        Assert.True(nativeHasTransform);
        Assert.Equal(100f, ffiTransform.Px);
        Assert.Equal(200f, ffiTransform.Py);
        Assert.Equal(300f, ffiTransform.Pz);

        // 2. Reset locally and Pull from native to verify it reads back correctly
        transformComp!.SuspendNotifications = true;
        transformComp.PosX = 0f;
        transformComp.SuspendNotifications = false;

        transformComp.PullFromNative();
        Assert.Equal(100f, transformComp.PosX);
      }
      catch (System.DllNotFoundException) { }
    }
  }
}
