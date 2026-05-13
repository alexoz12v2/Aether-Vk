using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Messaging;
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
      _service = new NativeRuntimeService(
        _stateManager,
        console,
        breadcrumb,
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
    public void BasicNativeCall_ShouldSucceed()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        Assert.True(_service.IsInitialized);
        ulong sceneId = _service.CreateScene(true);
        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f, 1.0f);
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

        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(width, height, sceneId);
        _service.AddPerspectiveCamera(sceneId, peId, "camera", 45f, 0.1f, 1000f);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f, 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var camera = _service.GetEntityByName(sceneId, "camera");
        Assert.NotNull(camera);

        TaskCompletionSource<ulong> tcs = new();
        WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
          this,
          (r, m) =>
          {
            tcs.TrySetResult(m.RenderGeneration);
          }
        );
        _service.PlayScene(sceneId);

        ulong taskId;
        try
        {
          taskId = await tcs.Task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
          WeakReferenceMessenger.Default.Unregister<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
            this
          );
        }

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
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(width, height, sceneId);

        var root = _stateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault();
        Assert.NotNull(root);

        ulong peId2 = _service.CreatePresentationEngine(width, height, sceneId);
        var firstCamera = _service.AddPerspectiveCamera(sceneId, peId, "camera1", 45f, 0.1f, 1000f);
        var secondCamera = _service.AddPerspectiveCamera(sceneId, peId2, "camera2", 45f, 0.1f, 1000f);

        TaskCompletionSource<ulong> tcs = new();
        int msgCount = 0;
        WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
          this,
          (r, m) =>
          {
            msgCount++;
            if (msgCount >= 2)
              tcs.TrySetResult(m.RenderGeneration);
          }
        );

        _service.PlayScene(sceneId);

        try
        {
          await tcs.Task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
          WeakReferenceMessenger.Default.Unregister<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
            this
          );
        }
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
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(width, height, sceneId);

        ulong sphereId = _service.SpawnProceduralSphere(sceneId, "TestSphere", 1.0f, 1.0f);
        Assert.NotEqual(0ul, sphereId);

        var meas = _service.CreateMeasurement(
          sceneId,
          "Meas",
          new float[] { 0, 0, 0 },
          new float[] { 1, 1, 1 }
        );
        Assert.NotNull(meas);

        var billboard = _service.SpawnImageBillboard(sceneId, "Bill", false, 1.0f, 1.0f);
        Assert.NotNull(billboard);

        var comet = new CometComponent();
        comet.Jets.Add(
          new JetMarker
          {
            PosX = 1f,
            PosY = 1f,
            PosZ = 1f,
            ColorR = 1f,
            ColorG = 0f,
            ColorB = 0f,
            Size = 1f,
          }
        );

        _service.SyncMarkers(sceneId, sphereId, comet);
        _service.RefreshBvhNodes(sceneId, sphereId, comet);

        _service.AddPerspectiveCamera(sceneId, peId, "camera", 45f, 0.1f, 1000f);
        var camera = _service.GetEntityByName(sceneId, "camera");

        TaskCompletionSource<ulong> tcs = new();
        int msgCount = 0;
        WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
          this,
          (r, m) =>
          {
            msgCount++;
            if (msgCount >= 1)
              tcs.TrySetResult(m.RenderGeneration);
          }
        );

        _service.PlayScene(sceneId);

        try
        {
          await tcs.Task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
          WeakReferenceMessenger.Default.Unregister<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
            this
          );
        }

        TestSceneExporter.ExportScene(sceneId, _stateManager, "AllArchetypes");
      }
      catch (DllNotFoundException) { }
    }
  }
}
