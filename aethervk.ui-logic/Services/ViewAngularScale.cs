using System;

namespace AetherVk.Logic.Services;

/// <summary>
/// How much of the world one viewport pixel covers, for frustum-aware camera drag rates: drags
/// stay "1:1 with the cursor" whatever the field of view, ortho extent or distance unit.
/// </summary>
public static class ViewAngularScale
{
  /// <summary>
  /// World size (projection units, AU) of one pixel at <paramref name="distance"/> from the
  /// camera. Orthographic: the full height divided by the pixel height (independent of distance).
  /// Perspective: the frustum height at that distance divided by the pixel height.
  /// </summary>
  public static double WorldPerPixel(CameraProjectionState proj, double distance, double heightPx)
  {
    double h = Math.Max(1.0, heightPx);
    if (proj.IsPerspective)
      return 2.0 * Math.Max(0.0, distance) * Math.Tan(proj.Fov * 0.5) / h;
    return Math.Abs(proj.Top - proj.Bottom) / h;
  }

  /// <summary>
  /// Angle (radians) one pixel subtends for an object at <paramref name="distance"/>:
  /// <c>fov / height</c> in perspective, <c>2·halfHeight / distance / height</c> in ortho.
  /// </summary>
  public static double RadPerPixel(CameraProjectionState proj, double distance, double heightPx)
  {
    if (proj.IsPerspective)
      return 2.0 * Math.Atan(Math.Tan(proj.Fov * 0.5)) / Math.Max(1.0, heightPx);
    return WorldPerPixel(proj, distance, heightPx) / Math.Max(distance, double.Epsilon);
  }
}
