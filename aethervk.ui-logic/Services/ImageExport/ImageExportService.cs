using System;
using System.IO;
using System.Threading.Tasks;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services.ImageExport;

/// <summary>What one export wrote: the model PNG, the composite PNG (null without a visible FITS
/// billboard) and the image size.</summary>
public sealed record ImageExportResult(string ModelPath, string? CompositePath, int Width, int Height)
{
  /// <summary>A FITS billboard was visible and composited.</summary>
  public bool FitsComposited => CompositePath is not null;
}

/// <summary>
/// File › Export image: two PNGs per export, <c>&lt;name&gt;_model.png</c> (the engine's image of
/// the viewport with its labels hidden) and <c>&lt;name&gt;_with_fits.png</c> (the same image with
/// the visible FITS desktop billboards drawn at their on-screen geometry); the second only when a
/// billboard is visible. The Avalonia overlays (ephemeris table, dates, badges) are separate
/// windows and are in neither.
/// </summary>
public sealed class ImageExportService(INativeRuntimeService runtime, IBillboardSnapshotProvider billboards)
{
  public const string ModelSuffix = "_model.png";
  public const string WithFitsSuffix = "_with_fits.png";

  /// <summary>The two files of an export named by the save dialog's <paramref name="chosenPath"/>
  /// (its <c>.png</c>, if any, is dropped): <c>name_model.png</c> and <c>name_with_fits.png</c>.</summary>
  public static (string Model, string WithFits) FileNames(string chosenPath)
  {
    string basePath = chosenPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase)
      ? chosenPath.Substring(0, chosenPath.Length - 4)
      : chosenPath;
    if (basePath.Length == 0)
      basePath = "export";
    return (basePath + ModelSuffix, basePath + WithFitsSuffix);
  }

  /// <summary>
  /// Captures the viewport <paramref name="presentationEngineId"/> and writes the files. The
  /// viewport's desktop rectangle is <paramref name="viewport"/> when given, else the provider's.
  /// Null when the viewport could not be captured (no size, no frame, core down).
  /// </summary>
  public async Task<ImageExportResult?> ExportAsync(ulong presentationEngineId, string chosenPath, ViewportFrame? viewport = null)
  {
    var image = await runtime.CaptureViewportAsync(presentationEngineId, hideLabels: true).ConfigureAwait(false);
    if (image is null || image.Width <= 0 || image.Height <= 0)
      return null;
    var rgba = image.ToRgba();
    var (modelPath, fitsPath) = FileNames(chosenPath);
    var dir = Path.GetDirectoryName(modelPath);
    if (!string.IsNullOrEmpty(dir))
      Directory.CreateDirectory(dir);
    PngCodec.WriteRgb(modelPath, rgba, image.Width, image.Height);

    var frame = viewport ?? billboards.Viewport;
    var visible = billboards.VisibleBillboards();
    if (frame is null || visible.Count == 0)
      return new ImageExportResult(modelPath, null, image.Width, image.Height);
    var composite = ImageCompositor.Composite(rgba, image.Width, image.Height, frame.Value, visible);
    PngCodec.WriteRgb(fitsPath, composite, image.Width, image.Height);
    return new ImageExportResult(modelPath, fitsPath, image.Width, image.Height);
  }
}
