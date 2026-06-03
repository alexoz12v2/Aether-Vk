using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media;

namespace AetherVk.Converters;

/// <summary>
/// Converts between a packed ARGB <see cref="uint"/> and <see cref="Color"/>.
/// Use with <see cref="ColorPicker"/> controls.
/// </summary>
public class UintToColorConverter : IValueConverter
{
  public static readonly UintToColorConverter Instance = new();

  public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is uint argb)
    {
      byte a = (byte)((argb >> 24) & 0xFF);
      byte r = (byte)((argb >> 16) & 0xFF);
      byte g = (byte)((argb >> 8) & 0xFF);
      byte b = (byte)(argb & 0xFF);
      return Color.FromArgb(a, r, g, b);
    }
    return Colors.White;
  }

  public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is Color c)
    {
      return ((uint)c.A << 24) | ((uint)c.R << 16) | ((uint)c.G << 8) | c.B;
    }
    return 0xFFFFFFFFu;
  }
}
