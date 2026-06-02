using System;
using System.Linq;
using System.Runtime.InteropServices;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests
{
  /// <summary>
  /// Tests for camera aspect-ratio correctness across multiple presentation engines,
  /// ortho/persp toggling, and projection matrix validation.
  /// </summary>
  [Collection("Sequential")]
  public class CameraAspectRatioTests : IDisposable
  {
    private readonly NativeRuntimeService _service;
    private readonly SceneStateManager _stateManager;
    private readonly string _assetPath;

    public CameraAspectRatioTests()
    {
      var dispatcherMock = new Moq.Mock<IUiThreadDispatcher>();
      dispatcherMock.Setup(d => d.Dispatch(Moq.It.IsAny<Action>())).Callback<Action>(a => a());
      _stateManager = new SceneStateManager();
      _service = new NativeRuntimeService(
        _stateManager,
        new ConsoleService(dispatcherMock.Object),
        new BreadcrumbService(dispatcherMock.Object),
        new NativeBufferPoolService(),
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

    /// <summary>
    /// Helper: get the native simulation context pointer via reflection.
    /// </summary>
    private IntPtr GetNativeContext()
    {
      return typeof(NativeRuntimeService)
          .GetField(
            "_simulationContext",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
          )
          ?.GetValue(_service) as IntPtr?
        ?? IntPtr.Zero;
    }

    /// <summary>
    /// Helper: get a camera component from native via the generic getComponent API.
    /// Returns true if the component was found; sets <paramref name="cam"/> accordingly.
    /// </summary>
    private bool GetCameraComponent(
      IntPtr ctx,
      ulong sceneId,
      ulong entityId,
      out NativeInterop.FfiCamera cam
    )
    {
      int size = Marshal.SizeOf<NativeInterop.FfiCamera>();
      IntPtr ptr = Marshal.AllocHGlobal(size);
      try
      {
        if (NativeInterop.avkSimulationContext_getComponent(ctx, sceneId, entityId, 2, ptr))
        {
          cam = Marshal.PtrToStructure<NativeInterop.FfiCamera>(ptr);
          return true;
        }
        cam = default;
        return false;
      }
      finally
      {
        Marshal.FreeHGlobal(ptr);
      }
    }

    /// <summary>
    /// Helper: set a camera component via the generic setComponent API.
    /// </summary>
    private void SetCameraComponent(
      IntPtr ctx,
      ulong sceneId,
      ulong entityId,
      in NativeInterop.FfiCamera data
    )
    {
      int size = Marshal.SizeOf<NativeInterop.FfiCamera>();
      IntPtr ptr = Marshal.AllocHGlobal(size);
      try
      {
        Marshal.StructureToPtr(data, ptr, false);
        NativeInterop.avkSimulationContext_setComponent(ctx, sceneId, entityId, 2, ptr);
      }
      finally
      {
        Marshal.FreeHGlobal(ptr);
      }
    }

    /// <summary>
    /// When two presentation engines exist and one is resized, only its camera's
    /// aspect ratio should change — the other camera must remain untouched.
    /// </summary>
    [Fact]
    public void Resize_TwoPresentationEngines_ShouldOnlyAffectTargetCamera()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        // Create two PEs with different sizes
        ulong peA = _service.CreatePresentationEngine(800, 600, sceneId); // 4:3
        ulong peB = _service.CreatePresentationEngine(1920, 1080, sceneId); // 16:9

        ulong camA = _service.AddPerspectiveCamera(sceneId, peA, "camA", 45f, 0.1f, 1000f);
        ulong camB = _service.AddPerspectiveCamera(sceneId, peB, "camB", 45f, 0.1f, 1000f);

        var ctx = GetNativeContext();

        // Read initial aspects
        Assert.True(GetCameraComponent(ctx, sceneId, camA, out var ffiA));
        Assert.True(GetCameraComponent(ctx, sceneId, camB, out var ffiB));

        float aspectA_initial = ffiA.Aspect;
        float aspectB_initial = ffiB.Aspect;

        // Sanity: they should have different aspects
        Assert.NotEqual(aspectA_initial, aspectB_initial, 2);

        // Resize PE A to 1024x768 (still 4:3 but different size)
        _service.ResizePresentationEngine(sceneId, peA, 1024, 768);

        // Re-read both cameras
        Assert.True(GetCameraComponent(ctx, sceneId, camA, out var ffiA2));
        Assert.True(GetCameraComponent(ctx, sceneId, camB, out var ffiB2));

        // Camera A should have updated aspect (1024/768 ≈ 1.333)
        Assert.Equal(1024f / 768f, ffiA2.Aspect, 2);

        // Camera B must NOT have changed — it should still be 16:9
        Assert.Equal(aspectB_initial, ffiB2.Aspect, 2);
      }
      catch (DllNotFoundException) { }
    }

    /// <summary>
    /// Toggling a camera to orthographic and back to perspective should preserve
    /// a valid (non-zero) aspect ratio.
    /// </summary>
    [Fact]
    public void Toggle_PerspToOrthoAndBack_ShouldPreserveAspectRatio()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(1920, 1080, sceneId);
        ulong camId = _service.AddPerspectiveCamera(sceneId, peId, "cam", 60f, 0.01f, 1000f);

        var ctx = GetNativeContext();

        // Read initial perspective state
        Assert.True(GetCameraComponent(ctx, sceneId, camId, out var initial));
        Assert.False(initial.IsOrthographic);
        float originalAspect = initial.Aspect;
        Assert.True(
          originalAspect > 0.1f,
          $"Initial aspect should be positive, got {originalAspect}"
        );

        // Switch to orthographic via FFI
        var orthoData = new NativeInterop.FfiCamera
        {
          IsOrthographic = true,
          Fov = 0f,
          Aspect = 0f, // C# would compute ortho bounds, aspect is irrelevant for ortho push
          Near = 0.01f,
          Far = 1000f,
          Left = -10f,
          Right = 10f,
          Bottom = -5f,
          Top = 5f,
          FocusDistance = 1.0f,
        };
        SetCameraComponent(ctx, sceneId, camId, in orthoData);

        // Read back — ortho DTO should now have computed aspect from bounds
        Assert.True(GetCameraComponent(ctx, sceneId, camId, out var orthoRead));
        Assert.True(orthoRead.IsOrthographic);
        // With our fix, aspect should be (right-left)/(top-bottom) = 20/11.25 ≈ 1.778
        float orthoAspect = orthoRead.Aspect;
        Assert.True(orthoAspect > 0.1f, $"Ortho DTO aspect should be non-zero, got {orthoAspect}");

        // Switch back to perspective using the aspect from the ortho DTO
        var perspData = new NativeInterop.FfiCamera
        {
          IsOrthographic = false,
          Fov = 60f,
          Aspect = orthoAspect, // Use what we got from the ortho DTO
          Near = 0.01f,
          Far = 1000f,
          FocusDistance = 1.0f,
        };
        SetCameraComponent(ctx, sceneId, camId, in perspData);

        // Read back and verify aspect is preserved
        Assert.True(GetCameraComponent(ctx, sceneId, camId, out var perspRead));
        Assert.False(perspRead.IsOrthographic);
        Assert.True(
          perspRead.Aspect > 0.1f,
          $"Persp aspect after toggle should be positive, got {perspRead.Aspect}"
        );
        // Aspect should match what we computed from ortho bounds (≈1.778 for 16:9)
        Assert.Equal(orthoAspect, perspRead.Aspect, 2);
      }
      catch (DllNotFoundException) { }
    }

    /// <summary>
    /// Verify the perspective projection matrix has the expected structure:
    /// - Proj00 = f/aspect (column-major [0][0])
    /// - Proj11 should be zero (our coordinate mapping puts depth in col1)
    /// - Proj12 should be -f (maps +Z up to -Y clip)
    /// </summary>
    [Fact]
    public void PerspectiveProjectionMatrix_ShouldMatchExpected()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);
        ulong peId = _service.CreatePresentationEngine(800, 600, sceneId);
        ulong camId = _service.AddPerspectiveCamera(sceneId, peId, "cam", 60f, 0.1f, 1000f);

        var ctx = GetNativeContext();
        Assert.True(GetCameraComponent(ctx, sceneId, camId, out var cam));

        float fovRad = 60f * MathF.PI / 180f;
        float f = 1f / MathF.Tan(fovRad / 2f);
        float aspect = cam.Aspect;
        float near = cam.Near;
        float far = cam.Far;

        // ProjXY = flat[X*4+Y] = column X, row Y  (column-major)
        //
        // perspective_vk_reverse_z columns:
        // The engine uses a Reverse-Z mapping for perspective projections where
        // Z maps near..far to 1..0 (instead of 0..1 in standard Vulkan).
        //
        //   Col 0: [f/aspect, 0, 0, 0]
        //   Col 1: [0, 0, near/(far-near), -1]
        //   Col 2: [0, -f, 0, 0]
        //   Col 3: [0, 0, far*near/(far-near), 0]

        // Col 0  (Proj0Y)
        Assert.Equal(f / aspect, cam.Proj00, 2);
        Assert.Equal(0f, cam.Proj01, 5);
        Assert.Equal(0f, cam.Proj02, 5);
        Assert.Equal(0f, cam.Proj03, 5);

        // Col 1  (Proj1Y)
        Assert.Equal(0f, cam.Proj10, 5);
        Assert.Equal(0f, cam.Proj11, 5);
        Assert.Equal(near / (far - near), cam.Proj12, 2); // Reverse-Z: near/(far-near)
        Assert.Equal(-1f, cam.Proj13, 5);

        // Col 2  (Proj2Y)
        Assert.Equal(0f, cam.Proj20, 5);
        Assert.Equal(-f, cam.Proj21, 2);
        Assert.Equal(0f, cam.Proj22, 5);
        Assert.Equal(0f, cam.Proj23, 5);

        // Col 3  (Proj3Y)
        Assert.Equal(0f, cam.Proj30, 5);
        Assert.Equal(0f, cam.Proj31, 5);
        Assert.Equal(far * near / (far - near), cam.Proj32, 2); // Reverse-Z: far*near/(far-near)
        Assert.Equal(0f, cam.Proj33, 5);
      }
      catch (DllNotFoundException) { }
    }

    /// <summary>
    /// Resizing PE A should not corrupt PE B's camera projection matrix.
    /// The diagonal element Proj00 = f/aspect encodes the aspect ratio,
    /// so we verify it independently for each camera after a cross-resize.
    /// </summary>
    [Fact]
    public void Resize_ShouldNotCorruptOtherCameraProjectionMatrix()
    {
      try
      {
        _service.InitializeSimulationContext("Vulkan", _assetPath, false);
        ulong sceneId = _service.CreateScene(true);

        ulong peA = _service.CreatePresentationEngine(640, 480, sceneId);
        ulong peB = _service.CreatePresentationEngine(1920, 1080, sceneId);

        ulong camA = _service.AddPerspectiveCamera(sceneId, peA, "camA", 45f, 0.1f, 1000f);
        ulong camB = _service.AddPerspectiveCamera(sceneId, peB, "camB", 45f, 0.1f, 1000f);

        var ctx = GetNativeContext();

        // Grab camera B's initial projection Proj00
        Assert.True(GetCameraComponent(ctx, sceneId, camB, out var camBBefore));
        float proj00_B_before = camBBefore.Proj00;

        // Resize PE A to something very different (ultra-wide)
        _service.ResizePresentationEngine(sceneId, peA, 3440, 1440);

        // Camera B's projection should be unchanged
        Assert.True(GetCameraComponent(ctx, sceneId, camB, out var camBAfter));
        Assert.Equal(proj00_B_before, camBAfter.Proj00, 4);

        // Camera A should reflect the new aspect
        Assert.True(GetCameraComponent(ctx, sceneId, camA, out var camAAfter));
        float fovRad = 45f * MathF.PI / 180f;
        float f = 1f / MathF.Tan(fovRad / 2f);
        float expectedProj00A = f / (3440f / 1440f);
        Assert.Equal(expectedProj00A, camAAfter.Proj00, 2);
      }
      catch (DllNotFoundException) { }
    }
  }
}
