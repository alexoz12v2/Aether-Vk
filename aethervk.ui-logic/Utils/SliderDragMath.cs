using System;

namespace AetherVk.Logic.Utils;

/// <summary>
/// Value change of an unbounded slider for a horizontal drag, independent of the display unit.
/// </summary>
public static class SliderDragMath
{
  /// <summary>Decades per pixel per unit of drag sensitivity on logarithmic sliders.</summary>
  public const double DecadesPerPixel = 0.005;

  /// <summary>Multiplier while Shift is held: fine control, like the camera drags.</summary>
  public const double FineFactor = 0.1;

  /// <summary>
  /// New value after dragging <paramref name="deltaPx"/> pixels.
  /// <para>Logarithmic: a fixed number of decades per pixel. It used to scale with
  /// <paramref name="step"/>, which depends on the unit: an AU extent (step 0.001) barely moved
  /// while the same extent in km (step 100) jumped 2.5 decades per pixel.</para>
  /// <para>Linear: <c>step</c> per 10 px per unit of sensitivity, so a finer unit gives finer
  /// control on purpose.</para>
  /// </summary>
  public static double Next(
    double value,
    double deltaPx,
    bool logarithmic,
    double step,
    double dragSensitivity,
    bool fine,
    double minPositive)
  {
    double mult = fine ? FineFactor : 1.0;
    if (logarithmic)
    {
      double floorLog = minPositive > 0 ? Math.Log10(minPositive) : -10.0;
      double currentLog = value > 0 ? Math.Log10(value) : floorLog;
      return Math.Pow(10, currentLog + deltaPx * DecadesPerPixel * dragSensitivity * mult);
    }
    return value + deltaPx * step * 0.1 * dragSensitivity * mult;
  }
}
