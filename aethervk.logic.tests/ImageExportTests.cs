using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.ImageExport;
using AetherVk.Logic.Utils;
using Moq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// File › Export image, the pure parts: the compositor (a billboard drawn at its on-screen
/// geometry: rotation, opacity, position), the two file names, the PNG codec, and the service's
/// "no FITS billboard visible" path (model PNG only).
/// </summary>
public class ImageExportTests
{
  private static byte[] SolidFrame(int w, int h, byte r, byte g, byte b)
  {
    var px = new byte[w * h * 4];
    for (int k = 0; k < w * h; k++)
    {
      px[4 * k] = r;
      px[4 * k + 1] = g;
      px[4 * k + 2] = b;
      px[4 * k + 3] = 255;
    }
    return px;
  }

  private static (int R, int G, int B) Px(byte[] rgba, int w, int x, int y)
  {
    int k = 4 * (y * w + x);
    return (rgba[k], rgba[k + 1], rgba[k + 2]);
  }

  /// A 4×4 billboard image: columns 0-1 red, 2-3 blue.
  private static byte[] RedBlueImage()
  {
    var img = new byte[4 * 4 * 4];
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++)
      {
        int k = 4 * (y * 4 + x);
        img[k] = (byte)(x < 2 ? 255 : 0);
        img[k + 2] = (byte)(x < 2 ? 0 : 255);
        img[k + 3] = 255;
      }
    return img;
  }

  [Fact]
  public void Composite_RotatedHalfTransparentBillboard_LandsWhereTheScreenShowsIt()
  {
    // black 8×8 frame of an 8×8 viewport at the desktop origin
    var frame = SolidFrame(8, 8, 0, 0, 0);
    var viewport = new ViewportFrame(new PointD(0, 0), 8, 8, 1);
    // a 4×4 billboard centred at (4, 4), turned 90° clockwise, half transparent: its red left
    // half (image columns 0-1) is now its top half (rows 2-3), the blue half the bottom (rows 4-5)
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 4, 4, 4, 4, Alpha: 0.5, RotationDeg: 90);
    var snapshot = new BillboardSnapshot(b, 4, 4, RedBlueImage());

    var output = ImageCompositor.Composite(frame, 8, 8, viewport, [snapshot]);

    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++)
      {
        var p = Px(output, 8, x, y);
        bool inside = x >= 2 && x <= 5 && y >= 2 && y <= 5;
        if (!inside)
        {
          Assert.Equal((0, 0, 0), p);
          continue;
        }
        var want = y <= 3 ? (128, 0, 0) : (0, 0, 128);
        Assert.True(Math.Abs(p.R - want.Item1) <= 1 && p.G == want.Item2 && Math.Abs(p.B - want.Item3) <= 1,
          $"pixel ({x},{y}) = {p}, wanted {want}");
      }
    // the input is untouched
    Assert.Equal(SolidFrame(8, 8, 0, 0, 0), frame);
  }

  [Fact]
  public void Composite_HiddenOrEmptyBillboardsLeaveTheFrame()
  {
    var frame = SolidFrame(4, 4, 10, 20, 30);
    var viewport = new ViewportFrame(new PointD(100, 50), 4, 4, 1);
    // a billboard far away from the viewport, and one with opacity 0
    var far = new BillboardSnapshot(new DesktopBillboard(Guid.NewGuid(), "k", 0, 0, 4, 4), 4, 4, RedBlueImage());
    var clear = new BillboardSnapshot(new DesktopBillboard(Guid.NewGuid(), "k", 102, 52, 4, 4, Alpha: 0), 4, 4, RedBlueImage());
    var output = ImageCompositor.Composite(frame, 4, 4, viewport, [far, clear]);
    Assert.Equal(frame, output);
    Assert.Equal(frame, ImageCompositor.Composite(frame, 4, 4, viewport, Array.Empty<BillboardSnapshot>()));
  }

  [Fact]
  public void Composite_ViewportScaling_MapsDesktopPixelsToFramePixels()
  {
    // the viewport's image is 4×4 but it covers 8×8 desktop px: a billboard over its right half
    var frame = SolidFrame(4, 4, 0, 0, 0);
    var viewport = new ViewportFrame(new PointD(0, 0), 8, 8, 2);
    var white = SolidFrame(2, 2, 255, 255, 255);
    var b = new DesktopBillboard(Guid.NewGuid(), "k", 6, 4, 4, 8, Alpha: 1);
    var output = ImageCompositor.Composite(frame, 4, 4, viewport, [new BillboardSnapshot(b, 2, 2, white)]);
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++)
        Assert.Equal(x >= 2 ? (255, 255, 255) : (0, 0, 0), Px(output, 4, x, y));
  }

  [Theory]
  [InlineData("/tmp/shot.png", "/tmp/shot_model.png", "/tmp/shot_with_fits.png")]
  [InlineData("/tmp/shot.PNG", "/tmp/shot_model.png", "/tmp/shot_with_fits.png")]
  [InlineData("/tmp/shot", "/tmp/shot_model.png", "/tmp/shot_with_fits.png")]
  [InlineData("C:\\out\\comet.2021.png", "C:\\out\\comet.2021_model.png", "C:\\out\\comet.2021_with_fits.png")]
  public void FileNames_ModelAndWithFitsNextToTheChosenName(string chosen, string model, string fits)
  {
    Assert.Equal((model, fits), ImageExportService.FileNames(chosen));
  }

  [Fact]
  public void PngCodec_RoundTripsAndIsAValidPng()
  {
    var dir = Path.Combine(Path.GetTempPath(), "aethervk-png-" + Guid.NewGuid().ToString("N").Substring(0, 8));
    var path = Path.Combine(dir, "sub", "t.png");
    try
    {
      var rgba = new byte[5 * 3 * 4];
      for (int k = 0; k < rgba.Length; k++) rgba[k] = (byte)(k * 7);
      PngCodec.WriteRgb(path, rgba, 5, 3);
      var back = PngCodec.ReadRgba(path, out int w, out int h);
      Assert.Equal((5, 3), (w, h));
      for (int k = 0; k < 5 * 3; k++)
      {
        Assert.Equal(rgba[4 * k], back[4 * k]);
        Assert.Equal(rgba[4 * k + 1], back[4 * k + 1]);
        Assert.Equal(rgba[4 * k + 2], back[4 * k + 2]);
        Assert.Equal(255, back[4 * k + 3]);
      }
      // an independent decoder agrees
      using var img = Image.Load<Rgb24>(path);
      Assert.Equal((5, 3), (img.Width, img.Height));
      var p = img[4, 2];
      int k0 = 4 * (2 * 5 + 4);
      Assert.Equal((rgba[k0], rgba[k0 + 1], rgba[k0 + 2]), (p.R, p.G, p.B));
    }
    finally
    {
      if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
  }

  [Fact]
  public void CapturedImage_BgraIsSwappedToRgba()
  {
    var bgra = new byte[] { 1, 2, 3, 255, 10, 20, 30, 255 };
    var image = CapturedImage.FromNative(2, 1, bgra, 0);
    Assert.True(image.IsBgra);
    Assert.Equal(new byte[] { 3, 2, 1, 255, 30, 20, 10, 255 }, image.ToRgba());
    var rgba = CapturedImage.FromNative(2, 1, bgra, 1);
    Assert.False(rgba.IsBgra);
    Assert.Equal(bgra, rgba.ToRgba());
  }

  private sealed class FixedBillboards(ViewportFrame? viewport, IReadOnlyList<BillboardSnapshot> visible) : IBillboardSnapshotProvider
  {
    public ViewportFrame? Viewport => viewport;
    public IReadOnlyList<BillboardSnapshot> VisibleBillboards() => visible;
  }

  private static Mock<INativeRuntimeService> RuntimeReturning(CapturedImage? image)
  {
    var runtime = new Mock<INativeRuntimeService>(MockBehavior.Strict);
    runtime.Setup(r => r.CaptureViewportAsync(7UL, true)).ReturnsAsync(image);
    return runtime;
  }

  [Fact]
  public async Task Export_NoVisibleBillboard_WritesTheModelOnly()
  {
    var dir = Path.Combine(Path.GetTempPath(), "aethervk-export-" + Guid.NewGuid().ToString("N").Substring(0, 8));
    try
    {
      // a 6×4 BGRA frame: pure blue in BGRA order is (255, 0, 0, 255)
      var bgra = new byte[6 * 4 * 4];
      for (int k = 0; k < 6 * 4; k++) { bgra[4 * k] = 255; bgra[4 * k + 3] = 255; }
      var runtime = RuntimeReturning(CapturedImage.FromNative(6, 4, bgra, 0));
      var service = new ImageExportService(runtime.Object, new FixedBillboards(new ViewportFrame(new PointD(0, 0), 6, 4, 1), []));

      var result = await service.ExportAsync(7, Path.Combine(dir, "shot.png"));

      Assert.NotNull(result);
      Assert.False(result!.FitsComposited);
      Assert.Null(result.CompositePath);
      Assert.Equal((6, 4), (result.Width, result.Height));
      Assert.Equal(Path.Combine(dir, "shot_model.png"), result.ModelPath);
      Assert.True(File.Exists(result.ModelPath));
      Assert.False(File.Exists(Path.Combine(dir, "shot_with_fits.png")));
      var px = PngCodec.ReadRgba(result.ModelPath, out int w, out int h);
      Assert.Equal((6, 4), (w, h));
      Assert.Equal((0, 0, 255), Px(px, 6, 5, 3));
      runtime.VerifyAll();
    }
    finally
    {
      if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
  }

  [Fact]
  public async Task Export_VisibleBillboard_WritesBothFiles()
  {
    var dir = Path.Combine(Path.GetTempPath(), "aethervk-export-" + Guid.NewGuid().ToString("N").Substring(0, 8));
    try
    {
      var runtime = RuntimeReturning(CapturedImage.FromNative(8, 8, SolidFrame(8, 8, 0, 0, 0), 1));
      var b = new DesktopBillboard(Guid.NewGuid(), "k", 4, 4, 4, 4, Alpha: 1);
      var provider = new FixedBillboards(null, [new BillboardSnapshot(b, 2, 2, SolidFrame(2, 2, 255, 255, 255))]);
      var service = new ImageExportService(runtime.Object, provider);

      // the provider has no viewport frame: the caller's wins
      var result = await service.ExportAsync(7, Path.Combine(dir, "shot"), new ViewportFrame(new PointD(0, 0), 8, 8, 1));

      Assert.NotNull(result);
      Assert.True(result!.FitsComposited);
      Assert.Equal(Path.Combine(dir, "shot_with_fits.png"), result.CompositePath);
      var model = PngCodec.ReadRgba(result.ModelPath, out _, out _);
      var fits = PngCodec.ReadRgba(result.CompositePath!, out int w, out int h);
      Assert.Equal((8, 8), (w, h));
      Assert.Equal((0, 0, 0), Px(model, 8, 4, 4));
      Assert.Equal((255, 255, 255), Px(fits, 8, 4, 4));
      Assert.Equal((0, 0, 0), Px(fits, 8, 0, 0));
    }
    finally
    {
      if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }
  }

  [Fact]
  public async Task Export_CaptureFailed_ReturnsNullAndWritesNothing()
  {
    var dir = Path.Combine(Path.GetTempPath(), "aethervk-export-" + Guid.NewGuid().ToString("N").Substring(0, 8));
    var service = new ImageExportService(RuntimeReturning(null).Object, new FixedBillboards(null, []));
    Assert.Null(await service.ExportAsync(7, Path.Combine(dir, "shot.png")));
    Assert.False(Directory.Exists(dir));
  }
}
