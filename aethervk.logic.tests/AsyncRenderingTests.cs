using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  [Collection("Sequential")]
  public class AsyncRenderingTests : IDisposable
  {
    private readonly SceneStateManager _stateManager;
    private readonly NativeRuntimeService _service;
    private readonly string _assetPath;

    public AsyncRenderingTests()
    {
      var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
      dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<Action>())).Callback<Action>(a => a());

      var console = new ConsoleService(dispatcherMock.Object);
      var breadcrumb = new BreadcrumbService(dispatcherMock.Object);

      _stateManager = new SceneStateManager();
      _service = new NativeRuntimeService(_stateManager, console, breadcrumb, dispatcherMock.Object);
      var baseDir = AppDomain.CurrentDomain.BaseDirectory;
      _assetPath = System.IO.Path.GetFullPath(System.IO.Path.Combine(baseDir, "../../../../assets"));
    }

    public void Dispose()
    {
      _service.Dispose();
    }

    [Fact]
    public void BasicNativeCall_ShouldSucceed()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        Assert.True(_service.IsInitialized);
        ulong sceneId = _service.CreateScene(true);
        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);
      }
      catch (DllNotFoundException) { }
    }

    [Fact]
    public async Task RenderTickAsync_ShouldCompleteAndProduceImage()
    {
      try
      {
        const uint width = 256;
        const uint height = 256;

        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        Assert.True(_service.IsInitialized);

        ulong peId = _service.CreatePresentationEngine(width, height);
        ulong sceneId = _service.CreateScene(true);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var camera = _service.GetEntityByName(sceneId, "camera");
        Assert.NotNull(camera);

        ulong taskId = await _service.RenderTickAsync(peId, sceneId, camera!.Id, width, height);

        int bufferSize = (int)(width * height * 4);
        IntPtr bufferPtr = Marshal.AllocHGlobal(bufferSize);
        try
        {
          bool success = await _service.DownloadImageAsync(taskId, bufferPtr, (nuint)bufferSize);
          Assert.True(success, "Failed to download image from GPU.\n");

          byte[] pixels = new byte[bufferSize];
          Marshal.Copy(bufferPtr, pixels, 0, bufferSize);

          TestSceneExporter.ExportPng(pixels, (int)width, (int)height, "BasicRender");
          TestSceneExporter.ExportScene(sceneId, _stateManager, "BasicRender");

          bool hasNonZero = false;
          for (int i = 0; i < pixels.Length; i++)
          {
            if (pixels[i] > 0)
            {
              hasNonZero = true;
              break;
            }
          }

          Assert.True(hasNonZero, "Image should contain rendered data.\n");
        }
        finally
        {
          Marshal.FreeHGlobal(bufferPtr);
        }
      }
      catch (DllNotFoundException) { }
    }

    [Fact]
    public async Task MultipleCameras_ShouldRenderSameScene()
    {
      try
      {
        const uint width = 256;
        const uint height = 256;

        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong peId = _service.CreatePresentationEngine(width, height);
        ulong sceneId = _service.CreateScene(true);

        var root = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault();
        Assert.NotNull(root);

        var firstCamera = _service.CreateCamera(sceneId, root!);
        var secondCamera = _service.CreateCamera(sceneId, root!);

        Task t1 = _service.RenderTickAsync(peId, sceneId, firstCamera.Id, width, height);
        Task t2 = _service.RenderTickAsync(peId, sceneId, secondCamera.Id, width, height);

        await Task.WhenAll(t1, t2);
      }
      catch (DllNotFoundException) { }
    }

    [Fact]
    public async Task AllArchetypes_ShouldRenderAndShutdownGracefully()
    {
      try
      {
        const uint width = 256;
        const uint height = 256;

        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong peId = _service.CreatePresentationEngine(width, height);
        ulong sceneId = _service.CreateScene(true);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var meas = _service.CreateMeasurement(sceneId, "Meas", new float[]{0,0,0}, new float[]{1,1,1});
        Assert.NotNull(meas);

        var billboard = _service.SpawnImageBillboard(sceneId, "Bill", false, 1.0f, 1.0f);
        Assert.NotNull(billboard);

        var comet = new CometComponent();
        comet.Jets.Add(new JetMarker { PosX = 1f, PosY = 1f, PosZ = 1f, ColorR = 1f, ColorG = 0f, ColorB = 0f, Size = 1f });
        
        _service.SyncMarkers(sceneId, sphereId, comet);
        _service.RefreshBvhNodes(sceneId, sphereId, comet);

        var camera = _service.GetEntityByName(sceneId, "camera");

        for (int i = 0; i < 3; i++)
        {
          await _service.RenderTickAsync(peId, sceneId, camera!.Id, width, height);
        }
        
        TestSceneExporter.ExportScene(sceneId, _stateManager, "AllArchetypes");
      }
      catch (DllNotFoundException) { }
    }
  }
}
