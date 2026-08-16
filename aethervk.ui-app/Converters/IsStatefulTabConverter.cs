using System;
using System.Globalization;
using AetherVk.Logic.ViewModels;
using Avalonia.Data.Converters;

namespace AetherVk.Converters;

/// <summary>
/// Returns <c>true</c> when the bound value implements <see cref="IStatefulTabHeader"/>.
/// Used to show/hide <see cref="Controls.CommonTabHeader"/> in <c>TabGroupNodeView</c>.
/// </summary>
public sealed class IsStatefulTabConverter : IValueConverter
{
  public static readonly IsStatefulTabConverter Instance = new();

  public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
    value is IStatefulTabHeader;

  public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) =>
    throw new NotSupportedException();
}
