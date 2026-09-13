using System;
using System.Globalization;
using System.Numerics;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace AetherVk.Converters;

/// <summary>
/// Converts between System.Numerics.Vector4 and Avalonia.Media.Color.
/// Vector4 uses [0,1] floating point range (X=R, Y=G, Z=B, W=A).
/// </summary>
public sealed class Vector4ToColorConverter : IValueConverter
{
  public static readonly Vector4ToColorConverter Instance = new();

  public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is Vector4 v)
    {
      byte a = (byte)Math.Clamp((int)(v.W * 255.0f), 0, 255);
      byte r = (byte)Math.Clamp((int)(v.X * 255.0f), 0, 255);
      byte g = (byte)Math.Clamp((int)(v.Y * 255.0f), 0, 255);
      byte b = (byte)Math.Clamp((int)(v.Z * 255.0f), 0, 255);
      return Color.FromArgb(a, r, g, b);
    }
    return Colors.White;
  }

  public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is Color c)
    {
      return new Vector4(
          c.R / 255.0f,
          c.G / 255.0f,
          c.B / 255.0f,
          c.A / 255.0f
      );
    }
    return new Vector4(1.0f);
  }
}
