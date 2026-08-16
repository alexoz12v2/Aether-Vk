using System.Numerics;

namespace AetherVk.Behaviors;

/// <summary>
/// Stateless helpers for applying numeric range constraints.
/// Both <see cref="NumericTextBoxBehavior"/> and
/// <c>UnboundedSlider</c> delegate here so the logic lives in exactly one place.
/// </summary>
public static class NumericConstraint
{
  /// <summary>
  /// Wraps <paramref name="value"/> into [<paramref name="min"/>, <paramref name="max"/>)
  /// using modular arithmetic that handles negative values correctly.
  /// </summary>
  public static T Wrap<T>(T value, T min, T max)
    where T : IFloatingPoint<T>
  {
    T span = max - min;
    if (span <= T.Zero) return min;

    value -= min;
    value  = value % span;
    if (value < T.Zero) value += span;
    return value + min;
  }

  /// <summary>
  /// Clamps <paramref name="value"/> to [<paramref name="min"/>, <paramref name="max"/>].
  /// Infinite bounds are treated as unconstrained on that side.
  /// </summary>
  public static T Clamp<T>(T value, T min, T max)
    where T : IFloatingPoint<T>
  {
    if (T.IsFinite(min) && value < min) return min;
    if (T.IsFinite(max) && value > max) return max;
    return value;
  }

  /// <summary>
  /// Applies wrap-around when <paramref name="wrap"/> is <c>true</c>,
  /// otherwise clamps to [<paramref name="min"/>, <paramref name="max"/>].
  /// </summary>
  public static T Apply<T>(T value, T min, T max, bool wrap)
    where T : IFloatingPoint<T>
    => wrap ? Wrap(value, min, max) : Clamp(value, min, max);
}
