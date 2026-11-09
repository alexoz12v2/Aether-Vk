using System;

namespace AetherVk.Logic.Models;

/// <summary>
/// One captured frame of a 3D viewport (<c>INativeRuntimeService.CaptureViewportAsync</c>):
/// <see cref="Width"/> × <see cref="Height"/> pixels of 4 bytes each, rows from the top, in the
/// swapchain's byte order (<see cref="IsBgra"/>). The bytes are display-encoded (what the screen
/// showed), so a PNG takes them as they are.
/// </summary>
public sealed record CapturedImage(int Width, int Height, byte[] Pixels, bool IsBgra, bool IsSrgb)
{
  /// <summary>The native <c>format</c> code: bit 0 = RGBA (else BGRA), bit 8 = sRGB swapchain.</summary>
  public static CapturedImage FromNative(int width, int height, byte[] pixels, uint format) =>
    new(width, height, pixels, IsBgra: (format & 1) == 0, IsSrgb: (format & 0x100) != 0);

  /// <summary>The pixels as RGBA bytes (a copy when the capture is BGRA, the buffer itself otherwise).</summary>
  public byte[] ToRgba()
  {
    if (!IsBgra)
      return Pixels;
    var rgba = new byte[Pixels.Length];
    for (int k = 0; k + 3 < Pixels.Length; k += 4)
    {
      rgba[k] = Pixels[k + 2];
      rgba[k + 1] = Pixels[k + 1];
      rgba[k + 2] = Pixels[k];
      rgba[k + 3] = Pixels[k + 3];
    }
    return rgba;
  }
}
