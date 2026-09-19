using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

/// <summary>
/// PInvoke bindings for libXfixes — the X11 extension that provides server-side
/// region objects and per-window input/clip/bounding shape regions.
/// <para>
/// XFixes is available on every modern desktop Linux distribution (libXfixes.so.3).
/// </para>
/// </summary>
internal static class PInvokeXFixes
{
  private const string Lib = "libXfixes.so.3";

  /// <summary>
  /// Creates a new server-side region from an array of <paramref name="nrects"/> rectangles.
  /// Pass <c>rects = 0</c> and <c>nrects = 0</c> to create an <b>empty</b> region (no area).
  /// </summary>
  /// <returns>An opaque server-side XserverRegion handle. Must be freed with <see cref="XFixesDestroyRegion"/>.</returns>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern nint XFixesCreateRegion(nint display, nint rects, int nrects);

  /// <summary>
  /// Sets a window's shape region for the specified shape <paramref name="kind"/>.
  /// <list type="bullet">
  ///   <item><description><c>kind = 0</c> → ShapeBounding (outer border)</description></item>
  ///   <item><description><c>kind = 1</c> → ShapeClip     (drawing clip)</description></item>
  ///   <item><description><c>kind = 2</c> → ShapeInput    (pointer hit-test region)</description></item>
  /// </list>
  /// Passing an <b>empty</b> region for <c>ShapeInput</c> removes the window from the X server's
  /// hit-test map entirely → all pointer events fall through to the window beneath.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern void XFixesSetWindowShapeRegion(
    nint display, nint window, int kind, int x_off, int y_off, nint region);

  /// <summary>Destroys a server-side region previously allocated with <see cref="XFixesCreateRegion"/>.</summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern void XFixesDestroyRegion(nint display, nint region);
}
