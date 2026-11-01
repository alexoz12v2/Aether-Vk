using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Interactivity;
using Avalonia.Xaml.Interactivity;

namespace AetherVk.Behaviors;

/// <summary>
/// A plain <see cref="TextBox"/> editing a <c>double?</c>, without NumericUpDown's spin buttons
/// (they took half of a narrow box). Commits on Enter, on focus loss and when the box leaves the
/// tree (a tab switch). Parses invariant dot decimals, a comma is read as the decimal point.
/// An empty box commits null when <see cref="AllowEmpty"/> (e.g. the nucleus pixel: the image
/// centre); otherwise empty or unreadable text reverts to the value.
/// </summary>
public class DoubleTextBoxBehavior : Behavior<TextBox>
{
  public static readonly StyledProperty<double?> ValueProperty =
    AvaloniaProperty.Register<DoubleTextBoxBehavior, double?>(nameof(Value), defaultBindingMode: BindingMode.TwoWay);

  public double? Value
  {
    get => GetValue(ValueProperty);
    set => SetValue(ValueProperty, value);
  }

  public static readonly StyledProperty<double> MinimumProperty =
    AvaloniaProperty.Register<DoubleTextBoxBehavior, double>(nameof(Minimum), double.NegativeInfinity);

  /// <summary>Inclusive lower bound (clamped).</summary>
  public double Minimum
  {
    get => GetValue(MinimumProperty);
    set => SetValue(MinimumProperty, value);
  }

  public static readonly StyledProperty<string> FormatProperty =
    AvaloniaProperty.Register<DoubleTextBoxBehavior, string>(nameof(Format), "F0");

  /// <summary>Display format (invariant culture).</summary>
  public string Format
  {
    get => GetValue(FormatProperty);
    set => SetValue(FormatProperty, value);
  }

  public static readonly StyledProperty<bool> AllowEmptyProperty =
    AvaloniaProperty.Register<DoubleTextBoxBehavior, bool>(nameof(AllowEmpty));

  /// <summary>An empty box means null.</summary>
  public bool AllowEmpty
  {
    get => GetValue(AllowEmptyProperty);
    set => SetValue(AllowEmptyProperty, value);
  }

  private string Formatted(double? v) => v is { } d ? d.ToString(Format, CultureInfo.InvariantCulture) : string.Empty;

  protected override void OnAttached()
  {
    base.OnAttached();
    if (AssociatedObject is not { } box) return;
    box.LostFocus += OnLostFocus;
    box.KeyDown += OnKeyDown;
    box.DetachedFromVisualTree += OnDetached;
    box.Text = Formatted(Value);
  }

  protected override void OnDetaching()
  {
    base.OnDetaching();
    if (AssociatedObject is not { } box) return;
    box.LostFocus -= OnLostFocus;
    box.KeyDown -= OnKeyDown;
    box.DetachedFromVisualTree -= OnDetached;
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    // an external change (the view model) shows, unless the user is typing
    if ((change.Property == ValueProperty || change.Property == FormatProperty) && AssociatedObject is { IsFocused: false } box)
      box.Text = Formatted(Value);
  }

  private void OnKeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
  {
    if (e.Key == Avalonia.Input.Key.Enter)
      Commit();
  }

  private void OnLostFocus(object? sender, RoutedEventArgs e) => Commit();

  private void OnDetached(object? sender, VisualTreeAttachmentEventArgs e)
  {
    if ((AssociatedObject?.Text ?? string.Empty) != Formatted(Value))
      Commit();
  }

  /// <summary>Parses the box: invariant, a comma as the decimal point; null for empty text.</summary>
  internal static bool TryParse(string? text, out double? value)
  {
    value = null;
    var t = (text ?? string.Empty).Trim().Replace(',', '.');
    if (t.Length == 0)
      return true;
    if (double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && double.IsFinite(d))
    {
      value = d;
      return true;
    }
    return false;
  }

  private void Commit()
  {
    if (AssociatedObject is not { } box) return;
    if (TryParse(box.Text, out var parsed) && (parsed is not null || AllowEmpty))
    {
      if (parsed is { } d && d < Minimum)
        parsed = Minimum;
      Value = parsed;
    }
    box.Text = Formatted(Value);
  }
}
