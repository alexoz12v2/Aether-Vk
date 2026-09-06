using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace AetherVk.Converters;

/// <summary>
/// Returns <c>true</c> when the bound enum value equals the string given in
/// <c>ConverterParameter</c>. Used to bind <see cref="Avalonia.Controls.RadioButton.IsChecked"/>
/// to enum properties without requiring separate bool properties per enum value.
/// </summary>
/// <example>
/// <code>
/// IsChecked="{Binding MyEnumProp,
///   Converter={x:Static converters:EnumToBoolConverter.Instance},
///   ConverterParameter=EnumValueName}"
/// </code>
/// </example>
public class EnumToBoolConverter : IValueConverter
{
  public static readonly EnumToBoolConverter Instance = new();

  public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is null || parameter is null)
      return false;
    // Compare the string representation of the enum value to the parameter string.
    return value.ToString() == parameter.ToString();
  }

  public object? ConvertBack(
    object? value,
    Type targetType,
    object? parameter,
    CultureInfo culture
  )
  {
    // When a RadioButton becomes checked (true), parse the parameter string back to the enum.
    if (value is true && parameter is not null)
      return Enum.Parse(targetType, parameter.ToString()!);
    // Unchecked → return UnsetValue so other RadioButtons can set the binding.
    return Avalonia.Data.BindingOperations.DoNothing;
  }
}
