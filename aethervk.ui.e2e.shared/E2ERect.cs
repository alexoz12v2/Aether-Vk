// Source-linked into aethervk.ui.e2e (Windows), aethervk.ui.e2e.mac and aethervk.ui.e2e.shared.tests.
using System;

namespace AetherVk.UI.E2E.Shared;

/// <summary>A rectangle of screen pixels (physical pixels, origin at the top left of the main display).</summary>
public readonly record struct E2ERect(int X, int Y, int Width, int Height)
{
  public int Right => X + Width;

  public int Bottom => Y + Height;

  public long Area => (long)Math.Max(0, Width) * Math.Max(0, Height);

  public double CentreX => X + Width / 2.0;

  public double CentreY => Y + Height / 2.0;

  public bool IsEmpty => Width <= 0 || Height <= 0;

  public bool Contains(double x, double y) => x >= X && x < Right && y >= Y && y < Bottom;

  /// <summary>Same place and size within <paramref name="tolerancePx"/> on every edge.</summary>
  public bool Near(E2ERect other, int tolerancePx) =>
    Math.Abs(X - other.X) <= tolerancePx && Math.Abs(Y - other.Y) <= tolerancePx
    && Math.Abs(Right - other.Right) <= tolerancePx && Math.Abs(Bottom - other.Bottom) <= tolerancePx;

  public override string ToString() => $"{X},{Y} {Width}×{Height}";
}

/// <summary>A rectangle in screen points (macOS global display coordinates, top-left origin).</summary>
public readonly record struct PointRect(double X, double Y, double Width, double Height)
{
  public override string ToString() => $"{X:0.##},{Y:0.##} {Width:0.##}×{Height:0.##} pt";
}
