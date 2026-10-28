using System;
using System.Collections.Generic;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Utils;

/// <summary>A point in desktop (or window-local) pixels, y down.</summary>
public readonly record struct PointD(double X, double Y)
{
  public static PointD operator +(PointD a, PointD b) => new(a.X + b.X, a.Y + b.Y);
  public static PointD operator -(PointD a, PointD b) => new(a.X - b.X, a.Y - b.Y);
  public static PointD operator *(PointD a, double k) => new(a.X * k, a.Y * k);
  public double Length => Math.Sqrt(X * X + Y * Y);
}

/// <summary>An integer pixel rectangle.</summary>
public readonly record struct RectI(int X, int Y, int Width, int Height)
{
  public bool Contains(double x, double y) => x >= X && y >= Y && x < X + Width && y < Y + Height;
}

/// <summary>A display: its bounds in desktop physical pixels and its scaling (physical px per DIP).</summary>
public readonly record struct ScreenInfo(RectI Bounds, double Scaling, bool IsPrimary = false);

/// <summary>What a gizmo handle does when dragged with the right button.</summary>
public enum BillboardHandle
{
  MoveFree,
  MoveAlongX,
  MoveAlongY,
  Rotate,
  ScaleTopLeft,
  ScaleTopRight,
  ScaleBottomRight,
  ScaleBottomLeft,
}

/// <summary>A gizmo handle: a disc of <see cref="Radius"/> px around <see cref="Center"/> (desktop px).</summary>
public readonly record struct HandleSpot(BillboardHandle Kind, PointD Center, double Radius);

/// <summary>
/// Pure geometry of desktop billboards: rotated corners, the window that holds the image and its
/// gizmo, gizmo handles and hit testing, the drags, display lookup and the X11 input region.
/// Everything in desktop physical pixels (y down); rotations are clockwise on screen.
/// </summary>
public static class DesktopBillboardGeometry
{
  /// <summary>Radians per arcsecond (π / 648000).</summary>
  public const double ArcsecToRad = Math.PI / 648000.0;

  private static bool Usable(double? v) => v is { } x && x > 0 && !double.IsInfinity(x);

  /// <summary>
  /// The angle (rad) the image's height subtends from the observer, the comet at
  /// <paramref name="cometDistanceKm"/>: from the image height at the comet
  /// (<see cref="DesktopBillboard.ImageHeightKm"/>, 2·atan(H / 2 / D)) when it and the distance are
  /// known (km wins), else from the plate scale (image height px × ″/px). Null when neither applies.
  /// </summary>
  public static double? ImageAngularHeightRad(DesktopBillboard b, double? cometDistanceKm)
  {
    if (Usable(b.ImageHeightKm) && Usable(cometDistanceKm))
      return 2.0 * Math.Atan(b.ImageHeightKm!.Value * 0.5 / cometDistanceKm!.Value);
    if (Usable(b.ImageArcsecPerPx) && b.ImageHeightPx > 0)
      return b.ImageHeightPx * b.ImageArcsecPerPx!.Value * ArcsecToRad;
    return null;
  }

  /// <summary>
  /// The image's height at the comet (km): <see cref="DesktopBillboard.ImageHeightKm"/> when set,
  /// else the plate-scale angle at <paramref name="cometDistanceKm"/> (2·D·tan(α / 2)). Null
  /// without either.
  /// </summary>
  public static double? ImageHeightAtCometKm(DesktopBillboard b, double? cometDistanceKm)
  {
    if (Usable(b.ImageHeightKm))
      return b.ImageHeightKm;
    if (!Usable(cometDistanceKm) || ImageAngularHeightRad(b, null) is not { } a)
      return null;
    return 2.0 * cometDistanceKm!.Value * Math.Tan(a * 0.5);
  }

  /// <summary>
  /// The on-screen size (desktop physical px) at which <paramref name="b"/> covers the sky its
  /// image covers, on a viewport of <paramref name="viewportArcsecPerPx"/> arcsec per physical px:
  /// the image's angular height (<see cref="ImageAngularHeightRad"/>) over the viewport's angle per
  /// px, the width from the image's aspect. With only a plate scale this is image px × image scale
  /// ÷ viewport scale. Null without an angular height, a native size or a scale.
  /// </summary>
  public static (double WidthPx, double HeightPx)? PhysicalSize(
    DesktopBillboard b, double viewportArcsecPerPx, double? cometDistanceKm = null)
  {
    if (b.ImageWidthPx <= 0 || b.ImageHeightPx <= 0 || !Usable(viewportArcsecPerPx)
        || ImageAngularHeightRad(b, cometDistanceKm) is not { } angle)
      return null;
    double h = angle / (viewportArcsecPerPx * ArcsecToRad);
    return (h * b.ImageWidthPx / b.ImageHeightPx, h);
  }

  /// <summary>Room around the rotated image for the gizmo handles, px.</summary>
  public const double GizmoMargin = 48;

  /// <summary>Distance of the translate arrow tips / rotate knob from the image, px.</summary>
  public const double HandleReach = 28;

  public const double HandleRadius = 8;

  /// <summary>Extra slack around a handle for hit testing, px.</summary>
  public const double HitSlack = 4;

  /// <summary>Smallest image side a scale drag can produce, px.</summary>
  public const double MinSidePx = 16;

  /// <summary>A right press that moves less than this before release is a click, px.</summary>
  public const double ClickThresholdPx = 4;

  public const double RotationSnapDeg = 15;

  public static PointD Center(DesktopBillboard b) => new(b.CenterX, b.CenterY);

  /// <summary>Rotates a vector clockwise on screen (y down) by <paramref name="deg"/>.</summary>
  public static PointD Rotate(PointD v, double deg)
  {
    double r = deg * Math.PI / 180.0;
    double c = Math.Cos(r), s = Math.Sin(r);
    return new PointD(v.X * c - v.Y * s, v.X * s + v.Y * c);
  }

  /// <summary>Desktop point → billboard-local (unrotated, centre at the origin).</summary>
  public static PointD ToLocal(DesktopBillboard b, PointD p) => Rotate(p - Center(b), -b.RotationDeg);

  /// <summary>Billboard-local → desktop point.</summary>
  public static PointD ToDesktop(DesktopBillboard b, PointD local) => Center(b) + Rotate(local, b.RotationDeg);

  /// <summary>Image corners on the desktop: top-left, top-right, bottom-right, bottom-left.</summary>
  public static PointD[] Corners(DesktopBillboard b)
  {
    double hw = b.WidthPx / 2, hh = b.HeightPx / 2;
    return
    [
      ToDesktop(b, new PointD(-hw, -hh)),
      ToDesktop(b, new PointD(hw, -hh)),
      ToDesktop(b, new PointD(hw, hh)),
      ToDesktop(b, new PointD(-hw, hh)),
    ];
  }

  /// <summary>
  /// The billboard window: the axis-aligned box of the rotated image grown by
  /// <see cref="GizmoMargin"/>, rounded outward to whole pixels.
  /// </summary>
  public static RectI WindowRect(DesktopBillboard b)
  {
    var cs = Corners(b);
    double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
    foreach (var c in cs)
    {
      minX = Math.Min(minX, c.X);
      minY = Math.Min(minY, c.Y);
      maxX = Math.Max(maxX, c.X);
      maxY = Math.Max(maxY, c.Y);
    }
    int x0 = (int)Math.Floor(minX - GizmoMargin);
    int y0 = (int)Math.Floor(minY - GizmoMargin);
    int x1 = (int)Math.Ceiling(maxX + GizmoMargin);
    int y1 = (int)Math.Ceiling(maxY + GizmoMargin);
    return new RectI(x0, y0, x1 - x0, y1 - y0);
  }

  /// <summary>The gizmo handles of <paramref name="mode"/>, desktop px.</summary>
  public static IReadOnlyList<HandleSpot> Handles(DesktopBillboard b, BillboardGizmoMode mode)
  {
    double hw = b.WidthPx / 2, hh = b.HeightPx / 2;
    switch (mode)
    {
      case BillboardGizmoMode.Translate:
        return
        [
          new(BillboardHandle.MoveFree, Center(b), HandleRadius + 2),
          new(BillboardHandle.MoveAlongX, ToDesktop(b, new PointD(hw + HandleReach, 0)), HandleRadius),
          new(BillboardHandle.MoveAlongY, ToDesktop(b, new PointD(0, hh + HandleReach)), HandleRadius),
        ];
      case BillboardGizmoMode.Rotate:
        return [new(BillboardHandle.Rotate, ToDesktop(b, new PointD(0, -hh - HandleReach)), HandleRadius)];
      case BillboardGizmoMode.Scale:
        return
        [
          new(BillboardHandle.ScaleTopLeft, ToDesktop(b, new PointD(-hw, -hh)), HandleRadius),
          new(BillboardHandle.ScaleTopRight, ToDesktop(b, new PointD(hw, -hh)), HandleRadius),
          new(BillboardHandle.ScaleBottomRight, ToDesktop(b, new PointD(hw, hh)), HandleRadius),
          new(BillboardHandle.ScaleBottomLeft, ToDesktop(b, new PointD(-hw, hh)), HandleRadius),
        ];
      default:
        return [];
    }
  }

  /// <summary>The handle under <paramref name="p"/> (desktop px), nearest first; null when none.</summary>
  public static BillboardHandle? HitTest(DesktopBillboard b, BillboardGizmoMode mode, PointD p)
  {
    BillboardHandle? best = null;
    double bestD = double.MaxValue;
    foreach (var h in Handles(b, mode))
    {
      double d = (p - h.Center).Length;
      if (d <= h.Radius + HitSlack && d < bestD)
      {
        best = h.Kind;
        bestD = d;
      }
    }
    return best;
  }

  /// <summary>Whether <paramref name="p"/> (desktop px) is on the image.</summary>
  public static bool IsOnImage(DesktopBillboard b, PointD p)
  {
    var l = ToLocal(b, p);
    return Math.Abs(l.X) <= b.WidthPx / 2 && Math.Abs(l.Y) <= b.HeightPx / 2;
  }

  /// <summary>Free translation by <paramref name="delta"/> px.</summary>
  public static DesktopBillboard Translate(DesktopBillboard b0, PointD delta) =>
    b0 with { CenterX = b0.CenterX + delta.X, CenterY = b0.CenterY + delta.Y };

  /// <summary>Translation along the billboard's own x (or y) axis: the drag projected on it.</summary>
  public static DesktopBillboard TranslateAlong(DesktopBillboard b0, PointD delta, bool alongX)
  {
    var axis = Rotate(alongX ? new PointD(1, 0) : new PointD(0, 1), b0.RotationDeg);
    double k = delta.X * axis.X + delta.Y * axis.Y;
    return Translate(b0, axis * k);
  }

  /// <summary>
  /// Rotation about the centre by the angle the pointer swept from <paramref name="start"/> to
  /// <paramref name="current"/>; <paramref name="snap"/> rounds the result to
  /// <see cref="RotationSnapDeg"/>. The result is kept in (−180, 180].
  /// </summary>
  public static DesktopBillboard RotateDrag(DesktopBillboard b0, PointD start, PointD current, bool snap)
  {
    var c = Center(b0);
    double a0 = Math.Atan2(start.Y - c.Y, start.X - c.X);
    double a1 = Math.Atan2(current.Y - c.Y, current.X - c.X);
    double deg = b0.RotationDeg + (a1 - a0) * 180.0 / Math.PI;
    if (snap)
      deg = Math.Round(deg / RotationSnapDeg) * RotationSnapDeg;
    return b0 with { RotationDeg = NormalizeDeg(deg) };
  }

  public static double NormalizeDeg(double deg)
  {
    double d = deg % 360.0;
    if (d <= -180) d += 360;
    if (d > 180) d -= 360;
    return d;
  }

  /// <summary>
  /// Corner drag: the corner follows the pointer, the opposite corner stays where it is.
  /// <paramref name="keepAspect"/> keeps the width / height ratio (the larger of the two relative
  /// changes wins). Sides never drop below <see cref="MinSidePx"/>.
  /// </summary>
  public static DesktopBillboard ScaleDrag(DesktopBillboard b0, BillboardHandle corner, PointD pointer, bool keepAspect)
  {
    (double sx, double sy) = corner switch
    {
      BillboardHandle.ScaleTopLeft => (-1.0, -1.0),
      BillboardHandle.ScaleTopRight => (1.0, -1.0),
      BillboardHandle.ScaleBottomRight => (1.0, 1.0),
      BillboardHandle.ScaleBottomLeft => (-1.0, 1.0),
      _ => throw new ArgumentOutOfRangeException(nameof(corner)),
    };
    var fixedLocal = new PointD(-sx * b0.WidthPx / 2, -sy * b0.HeightPx / 2);
    var p = ToLocal(b0, pointer);
    double w = Math.Max(MinSidePx, sx * (p.X - fixedLocal.X));
    double h = Math.Max(MinSidePx, sy * (p.Y - fixedLocal.Y));
    if (keepAspect && b0.WidthPx > 0 && b0.HeightPx > 0)
    {
      double k = Math.Max(w / b0.WidthPx, h / b0.HeightPx);
      k = Math.Max(k, MinSidePx / Math.Min(b0.WidthPx, b0.HeightPx));
      w = b0.WidthPx * k;
      h = b0.HeightPx * k;
    }
    var newCenterLocal = fixedLocal + new PointD(sx * w / 2, sy * h / 2);
    var c = ToDesktop(b0, newCenterLocal);
    return b0 with { CenterX = c.X, CenterY = c.Y, WidthPx = w, HeightPx = h };
  }

  /// <summary>The next gizmo mode of the right-click cycle None → Translate → Rotate → Scale → None.</summary>
  public static BillboardGizmoMode NextMode(BillboardGizmoMode m) => m switch
  {
    BillboardGizmoMode.None => BillboardGizmoMode.Translate,
    BillboardGizmoMode.Translate => BillboardGizmoMode.Rotate,
    BillboardGizmoMode.Rotate => BillboardGizmoMode.Scale,
    _ => BillboardGizmoMode.None,
  };

  /// <summary>Index of the display containing <paramref name="p"/>; −1 when none does.</summary>
  public static int DisplayOf(IReadOnlyList<ScreenInfo> screens, PointD p)
  {
    for (int i = 0; i < screens.Count; i++)
      if (screens[i].Bounds.Contains(p.X, p.Y))
        return i;
    return -1;
  }

  /// <summary>Desktop point → (display index, display-local px); index −1 when off every display.</summary>
  public static (int Display, PointD Local) ToDisplayLocal(IReadOnlyList<ScreenInfo> screens, PointD p)
  {
    int i = DisplayOf(screens, p);
    if (i < 0)
      return (-1, p);
    var b = screens[i].Bounds;
    return (i, new PointD(p.X - b.X, p.Y - b.Y));
  }

  /// <summary>Display-local px on display <paramref name="display"/> → desktop point.</summary>
  public static PointD FromDisplayLocal(IReadOnlyList<ScreenInfo> screens, int display, PointD local)
  {
    var b = screens[display].Bounds;
    return new PointD(b.X + local.X, b.Y + local.Y);
  }

  /// <summary>
  /// A billboard whose centre is on no display (a monitor was unplugged) moves to the centre of the
  /// primary display (the first one when none is flagged); otherwise it is returned unchanged.
  /// </summary>
  public static DesktopBillboard KeepOnScreen(DesktopBillboard b, IReadOnlyList<ScreenInfo> screens)
  {
    if (screens.Count == 0 || DisplayOf(screens, Center(b)) >= 0)
      return b;
    var primary = screens[0];
    foreach (var s in screens)
      if (s.IsPrimary)
        primary = s;
    return b with
    {
      CenterX = primary.Bounds.X + primary.Bounds.Width / 2.0,
      CenterY = primary.Bounds.Y + primary.Bounds.Height / 2.0,
    };
  }

  /// <summary>
  /// Size of a new billboard for a <paramref name="texW"/>×<paramref name="texH"/> texture: native,
  /// shrunk (aspect kept) to fit half the display.
  /// </summary>
  public static (double W, double H) InitialSize(int texW, int texH, ScreenInfo screen)
  {
    double w = Math.Max(1, texW), h = Math.Max(1, texH);
    double k = Math.Min(1.0, Math.Min(screen.Bounds.Width * 0.5 / w, screen.Bounds.Height * 0.5 / h));
    return (w * k, h * k);
  }

  /// <summary>
  /// The pointer-sensitive part of the billboard window, window-local px (origin = the window's
  /// top-left, <see cref="WindowRect"/>): the rotated image as horizontal strips of
  /// <paramref name="stripPx"/> rows (each covering the image's span over its rows), plus a square
  /// around every handle of <paramref name="mode"/>. Everything else lets clicks through to the
  /// windows behind (XFixes input shape).
  /// </summary>
  public static List<RectI> InputRegion(DesktopBillboard b, BillboardGizmoMode mode, int stripPx = 2)
  {
    var win = WindowRect(b);
    var rects = new List<RectI>();
    var cs = Corners(b);
    double minY = double.MaxValue, maxY = double.MinValue;
    foreach (var c in cs)
    {
      minY = Math.Min(minY, c.Y);
      maxY = Math.Max(maxY, c.Y);
    }
    for (int y = (int)Math.Floor(minY); y < (int)Math.Ceiling(maxY); y += stripPx)
    {
      // x span of the quad over [y, y + stripPx): union of the spans at the strip's two edges
      double x0 = double.MaxValue, x1 = double.MinValue;
      foreach (double yy in new[] { y + 0.0, Math.Min(y + stripPx, maxY), Math.Max(y, minY) })
      {
        if (SpanAt(cs, yy) is (double a, double c2))
        {
          x0 = Math.Min(x0, a);
          x1 = Math.Max(x1, c2);
        }
      }
      if (x1 < x0)
        continue;
      int rx0 = (int)Math.Floor(x0) - win.X, rx1 = (int)Math.Ceiling(x1) - win.X;
      rects.Add(new RectI(rx0, y - win.Y, rx1 - rx0, stripPx));
    }
    foreach (var h in Handles(b, mode))
    {
      int r = (int)Math.Ceiling(h.Radius + HitSlack);
      rects.Add(new RectI((int)Math.Floor(h.Center.X) - r - win.X, (int)Math.Floor(h.Center.Y) - r - win.Y, 2 * r + 1, 2 * r + 1));
    }
    return rects;
  }

  /// <summary>Horizontal span of the convex quad <paramref name="cs"/> at height <paramref name="y"/>.</summary>
  private static (double, double)? SpanAt(PointD[] cs, double y)
  {
    double x0 = double.MaxValue, x1 = double.MinValue;
    for (int i = 0; i < 4; i++)
    {
      var a = cs[i];
      var c = cs[(i + 1) % 4];
      if ((y < Math.Min(a.Y, c.Y)) || (y > Math.Max(a.Y, c.Y)))
        continue;
      double x = Math.Abs(c.Y - a.Y) < 1e-9 ? Math.Min(a.X, c.X) : a.X + (y - a.Y) * (c.X - a.X) / (c.Y - a.Y);
      double xb = Math.Abs(c.Y - a.Y) < 1e-9 ? Math.Max(a.X, c.X) : x;
      x0 = Math.Min(x0, Math.Min(x, xb));
      x1 = Math.Max(x1, Math.Max(x, xb));
    }
    return x1 >= x0 ? (x0, x1) : null;
  }
}
