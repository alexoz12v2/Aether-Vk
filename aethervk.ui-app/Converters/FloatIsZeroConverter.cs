using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace AetherVk.Converters;

/// <summary>
/// Returns <c>true</c> when the bound <see cref="float"/> (or <see cref="double"/>) value is
/// equal to zero (or null). Useful for showing/hiding "required" hints.
/// </summary>
public class FloatIsZeroConverter : IValueConverter
{
  public static readonly FloatIsZeroConverter Instance = new();

  public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    return value switch
    {
      null => true,
      float f => f == 0f,
      double d => d == 0.0,
      _ => false,
    };
  }

  public object? ConvertBack(
    object? value,
    Type targetType,
    object? parameter,
    CultureInfo culture
  ) => throw new NotSupportedException();
}
