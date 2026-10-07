using System;

namespace AetherVk.Logic.Services;

/// <summary>Defaults and limits of Model tab settings shared by the session and the view model.</summary>
public static class ModelTabDefaults
{
  /// <summary>
  /// Dust display stretch softening (mirror of <c>dust::DUST_SOFTENING_DEFAULT</c>), relative to the
  /// view's white point (the brightest dust in view, measured every frame): dust 100× fainter than
  /// the brightest shows at ~17 % opacity.
  /// </summary>
  public const double DustSoftening = 1e-2;

  /// <summary>Accepted range (mirror of <c>dust::DUST_SOFTENING_MIN/MAX</c>); 1 is close to linear.</summary>
  public const double DustSofteningMin = 1e-5;
  public const double DustSofteningMax = 1.0;

  /// <summary>Clamps to [<see cref="DustSofteningMin"/>, <see cref="DustSofteningMax"/>],
  /// the default when not finite.</summary>
  public static double ClampDustSoftening(double s) =>
    double.IsNaN(s) || double.IsInfinity(s)
      ? DustSoftening
      : Math.Min(Math.Max(s, DustSofteningMin), DustSofteningMax);
}
