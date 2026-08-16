using Avalonia;
using Avalonia.Data;

namespace AetherVk.Behaviors;

/// <summary>
/// TextBox behavior for angle inputs.
/// <para>
/// Defaults:  wrap-around within [0°, 360°), degrees.
/// Set <see cref="UseRadians"/> = <c>true</c> to switch to radians
/// (wraps within [0, 2π) by default; change <see cref="NumericTextBoxBehavior.Minimum"/>
/// / <see cref="NumericTextBoxBehavior.Maximum"/> for a different radian range).
/// </para>
/// </summary>
public class AngleBehavior : NumericTextBoxBehavior
{
  public static readonly StyledProperty<bool> UseRadiansProperty =
    AvaloniaProperty.Register<AngleBehavior, bool>(
      nameof(UseRadians),
      defaultValue: false);

  /// <summary>
  /// When <c>true</c> the value is treated as radians and the default
  /// range becomes [0, 2π).  When <c>false</c> (default) the range is [0°, 360°).
  /// </summary>
  public bool UseRadians
  {
    get => GetValue(UseRadiansProperty);
    set => SetValue(UseRadiansProperty, value);
  }

  protected override void OnAttached()
  {
    // Apply sensible angle defaults before the base wires up events / sets the
    // initial text.  Only touch properties that are still at their default
    // values so an explicit XAML override is always respected.
    ApplyAngleDefaults();
    base.OnAttached();
  }

  private void ApplyAngleDefaults()
  {
    // Always wrap for angles
    if (!IsSet(WrapProperty))
      Wrap = true;

    if (UseRadians)
    {
      if (!IsSet(MinimumProperty))
        Minimum = 0f;
      if (!IsSet(MaximumProperty))
        Maximum = 2f * System.MathF.PI;
      if (!IsSet(FormatProperty))
        Format = "0.####";
    }
    else
    {
      if (!IsSet(MinimumProperty))
        Minimum = 0f;
      if (!IsSet(MaximumProperty))
        Maximum = 360f;
    }
  }
}
