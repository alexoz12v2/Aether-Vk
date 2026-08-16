namespace AetherVk.Behaviors;

/// <summary>
/// TextBox behavior for distance / non-negative float inputs.
/// <para>
/// Defaults: clamp to [0, ∞).
/// Override <see cref="NumericTextBoxBehavior.Minimum"/> /
/// <see cref="NumericTextBoxBehavior.Maximum"/> in XAML for custom ranges.
/// </para>
/// </summary>
public class DistanceBehavior : NumericTextBoxBehavior
{
  protected override void OnAttached()
  {
    // Distances are non-negative by default; respect explicit XAML overrides.
    if (!IsSet(MinimumProperty))
      Minimum = 0f;

    base.OnAttached();
  }
}
