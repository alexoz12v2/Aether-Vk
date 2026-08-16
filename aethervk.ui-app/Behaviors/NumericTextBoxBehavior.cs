using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Interactivity;
using Avalonia.Xaml.Interactivity;
using AetherVk.Controls;

namespace AetherVk.Behaviors;

/// <summary>
/// Base behavior for controls that accept a numeric float value.
/// Works with both <see cref="TextBox"/> and <see cref="UnboundedSlider"/>.
/// Provides commit-on-focus-loss / Enter, optional clamping, optional wrap-around,
/// and a configurable display format string.
/// </summary>
public class NumericTextBoxBehavior : Behavior<Control>, IHandlesCommit
{
  // ── Bindable value ────────────────────────────────────────────────────────

  public static readonly StyledProperty<float> ValueProperty =
    AvaloniaProperty.Register<NumericTextBoxBehavior, float>(
      nameof(Value),
      defaultBindingMode: BindingMode.TwoWay);

  /// <summary>Two-way bound numeric value.</summary>
  public float Value
  {
    get => GetValue(ValueProperty);
    set => SetValue(ValueProperty, value);
  }

  // ── Range ─────────────────────────────────────────────────────────────────

  public static readonly StyledProperty<float> MinimumProperty =
    AvaloniaProperty.Register<NumericTextBoxBehavior, float>(
      nameof(Minimum),
      defaultValue: float.NegativeInfinity);

  /// <summary>Inclusive lower bound. Default: <see cref="float.NegativeInfinity"/> (unclamped).</summary>
  public float Minimum
  {
    get => GetValue(MinimumProperty);
    set => SetValue(MinimumProperty, value);
  }

  public static readonly StyledProperty<float> MaximumProperty =
    AvaloniaProperty.Register<NumericTextBoxBehavior, float>(
      nameof(Maximum),
      defaultValue: float.PositiveInfinity);

  /// <summary>
  /// Exclusive upper bound when <see cref="Wrap"/> is <c>true</c>;
  /// inclusive upper bound when <see cref="Wrap"/> is <c>false</c>.
  /// Default: <see cref="float.PositiveInfinity"/> (unclamped).
  /// </summary>
  public float Maximum
  {
    get => GetValue(MaximumProperty);
    set => SetValue(MaximumProperty, value);
  }

  // ── Wrap-around ───────────────────────────────────────────────────────────

  public static readonly StyledProperty<bool> WrapProperty =
    AvaloniaProperty.Register<NumericTextBoxBehavior, bool>(
      nameof(Wrap),
      defaultValue: false);

  /// <summary>
  /// When <c>true</c>, values outside [<see cref="Minimum"/>, <see cref="Maximum"/>)
  /// wrap around instead of being clamped.
  /// </summary>
  public bool Wrap
  {
    get => GetValue(WrapProperty);
    set => SetValue(WrapProperty, value);
  }

  // ── Display format ────────────────────────────────────────────────────────

  public static readonly StyledProperty<string> FormatProperty =
    AvaloniaProperty.Register<NumericTextBoxBehavior, string>(
      nameof(Format),
      defaultValue: "0.##");

  /// <summary>
  /// Standard numeric format string used when rendering the value back into
  /// the control. Default: <c>"0.##"</c>.
  /// </summary>
  public string Format
  {
    get => GetValue(FormatProperty);
    set => SetValue(FormatProperty, value);
  }

  // ── Text access helpers ───────────────────────────────────────────────────

  /// <summary>Reads the displayed text from whichever control type is associated.</summary>
  protected string? GetText() => AssociatedObject switch
  {
    TextBox tb          => tb.Text,
    UnboundedSlider s   => s.InputText,
    _                   => null,
  };

  /// <summary>Writes text back to the displayed field.</summary>
  protected void SetText(string? text)
  {
    switch (AssociatedObject)
    {
      case TextBox tb:        tb.Text       = text; break;
      case UnboundedSlider s: s.InputText   = text; break;
    }
  }

  // ── Avalonia lifecycle ────────────────────────────────────────────────────

  protected override void OnAttached()
  {
    base.OnAttached();
    if (AssociatedObject is null) return;

    AssociatedObject.LostFocus += OnCommit;
    AssociatedObject.KeyDown   += OnKeyDown;
    SetText(Value.ToString(Format));
  }

  protected override void OnDetaching()
  {
    base.OnDetaching();
    if (AssociatedObject is null) return;

    AssociatedObject.LostFocus -= OnCommit;
    AssociatedObject.KeyDown   -= OnKeyDown;
  }

  // ── Input handling ────────────────────────────────────────────────────────

  private void OnKeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
  {
    if (e.Key == Avalonia.Input.Key.Enter)
      OnCommit(sender, e);
  }

  private void OnCommit(object? sender, RoutedEventArgs e)
  {
    if (AssociatedObject is null) return;

    if (float.TryParse(GetText(), out float parsed))
    {
      parsed = Constrain(parsed);
      Value  = parsed;
      SetText(parsed.ToString(Format));
    }
    else
    {
      // Revert to the last valid value
      SetText(Value.ToString(Format));
    }
  }

  // ── Range enforcement ─────────────────────────────────────────────────────

  /// <summary>
  /// Applies the configured range rule to <paramref name="value"/>:
  /// wrap-around when <see cref="Wrap"/> is <c>true</c>, clamping otherwise.
  /// Subclasses can override for custom logic.
  /// </summary>
  protected virtual float Constrain(float value)
    => NumericConstraint.Apply(value, Minimum, Maximum, Wrap);
}
