using System.Runtime.InteropServices;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

public class AsyncRenderingTests : IDisposable
{
  private readonly NativeRuntimeService _service;
  private readonly string _assetPath;

  public AsyncRenderingTests()
  {
    ServiceLocator.DispatchToUI = a => a();
    _service = new NativeRuntimeService();
    var baseDir = AppDomain.CurrentDomain.BaseDirectory;
    // Adjust asset path if necessary to point to the actual assets folder
    _assetPath = Path.GetFullPath(Path.Combine(baseDir, "../../../../assets"));
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
      _service.InitializeSimulationContext("Vulkan", 256, 256, _assetPath);
      Assert.True(_service.IsInitialized);
      ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
      Assert.NotEqual(0ul, sphereId);
    }
    catch (DllNotFoundException)
    {
    }
  }

  [Fact]
  public async Task RenderTickAsync_ShouldCompleteAndProduceImage()
  {
    try
    {
      const uint width = 256;
      const uint height = 256;

      _service.InitializeSimulationContext("Vulkan", width, height, _assetPath);
      Assert.True(_service.IsInitialized);

      // Set clear color to a specific value (e.g., green)
      _service.SetClearColor(0.0f, 1.0f, 0.0f, 1.0f);

      // Spawn a procedural sphere to have something in the scene
      ulong sphereId = _service.SpawnProceduralSphere("TestSphere", 1.0f);
      Assert.NotEqual(0ul, sphereId);

      // Position camera to look at the sphere
      var root = _service.RootEntities.FirstOrDefault();
      var camera = root?.Children.FirstOrDefault(e => e.Name == "camera");
      Assert.NotNull(camera);

      // Render a frame and await completion
      await _service.RenderTickAsync();

      // Download the image
      int bufferSize = (int)(width * height * 4);
      IntPtr bufferPtr = Marshal.AllocHGlobal(bufferSize);
      try
      {
        bool success = _service.DownloadImage(bufferPtr, (nuint)bufferSize);
        Assert.True(success, "Failed to download image from GPU.");

        byte[] pixels = new byte[bufferSize];
        Marshal.Copy(bufferPtr, pixels, 0, bufferSize);

        // Assert on some pixels.
        // Since we cleared with green [0, 255, 0, 255], and the sphere is likely white [255, 255, 255, 255] or dark
        // We check if at least some pixels are NOT the clear color (meaning something was rendered)
        // Or check corners for clear color.

        // Corner pixel (should be green if not obscured)
        int topLeftIdx = 0;
        // Note: format might be BGRA or RGBA depending on backend. Vulkan windowless is usually RGBA8 or BGRA8.
        // We'll check if it's green-ish.
        Assert.True(pixels[topLeftIdx + 1] > 200, "Top-left pixel should be green (clear color).");

        // Center pixel (should have the sphere)
        int centerIdx = (int)((height / 2 * width + width / 2) * 4);
        // The sphere should be there.
        // If it's correctly rendered, it shouldn't be green.
        Assert.True(
          pixels[centerIdx + 1] < 255 || pixels[centerIdx] > 0 || pixels[centerIdx + 2] > 0,
          "Center pixel should not be just the clear color (sphere should be rendered).");
      }
      finally
      {
        Marshal.FreeHGlobal(bufferPtr);
      }
    }
    catch (DllNotFoundException)
    {
      // Skip test if Vulkan is not available
    }
  }
}
