using System;
using System.Globalization;
using Avalonia.Data.Converters;
using Avalonia.Media.Imaging;

namespace AetherVk.Converters
{
  public class FilePathToBitmapConverter : IValueConverter
  {
    public static readonly FilePathToBitmapConverter Instance = new();

    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
      if (value is string path && System.IO.File.Exists(path))
      {
        try
        {
          return new Bitmap(path);
        }
        catch
        {
          return null;
        }
      }
      return null;
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
