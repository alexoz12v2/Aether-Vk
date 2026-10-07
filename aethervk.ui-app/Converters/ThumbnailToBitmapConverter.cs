using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AetherVk.Logic.Models;
using Avalonia;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace AetherVk.Converters
{
  /// <summary>
  /// Converts a native RGBA8 <see cref="AssetThumbnail"/> (Imports tab / appearance pickers) into
  /// a bitmap. One bitmap per thumbnail instance is cached, so re-templated list items do not
  /// re-upload pixels.
  /// </summary>
  public class ThumbnailToBitmapConverter : IValueConverter
  {
    public static readonly ThumbnailToBitmapConverter Instance = new();

    private static readonly ConditionalWeakTable<AssetThumbnail, Bitmap> Cache = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
      if (value is not AssetThumbnail thumb || thumb.Width <= 0 || thumb.Height <= 0)
        return null;
      if (thumb.Rgba.Length < thumb.Width * thumb.Height * 4)
        return null;
      return Cache.GetValue(thumb, Create);
    }

    private static Bitmap Create(AssetThumbnail thumb)
    {
      var handle = GCHandle.Alloc(thumb.Rgba, GCHandleType.Pinned);
      try
      {
        // Copies the pixels: the pinned array is released right after.
        return new Bitmap(
          PixelFormat.Rgba8888,
          AlphaFormat.Unpremul,
          handle.AddrOfPinnedObject(),
          new PixelSize(thumb.Width, thumb.Height),
          new Vector(96, 96),
          thumb.Width * 4
        );
      }
      finally
      {
        handle.Free();
      }
    }

    public object? ConvertBack(
      object? value,
      Type targetType,
      object? parameter,
      CultureInfo culture
    )
    {
      throw new NotSupportedException();
    }
  }
}
