using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Utils;

/// <summary>
/// The viewport on the desktop: its top-left corner (<see cref="Origin"/>, desktop physical px),
/// its size (physical px) and the scaling of the screen it is on (physical px per DIP).
/// </summary>
public readonly record struct ViewportFrame(PointD Origin, double WidthPx, double HeightPx, double Scaling)
{
  public PointD Center => new(Origin.X + WidthPx / 2.0, Origin.Y + HeightPx / 2.0);

  /// <summary>Whether <paramref name="p"/> (desktop px) is inside the viewport.</summary>
  public bool Contains(PointD p) =>
    p.X >= Origin.X && p.Y >= Origin.Y && p.X < Origin.X + WidthPx && p.Y < Origin.Y + HeightPx;
}

/// <summary>
/// A billboard's centre as saved: the offset from the viewport's top-left corner, in physical px
/// of the screen the viewport was on, with that screen's <see cref="Scaling"/> so a viewport on a
/// screen of another scaling can convert it.
/// </summary>
public readonly record struct ViewportOffset(double X, double Y, double Scaling);

/// <summary>
/// Where billboards go against the viewport: the saved offset of a centre, the centre a saved
/// offset gives on the viewport now (another monitor, another DPI), the re-placement of an older
/// save that stored only desktop positions, and the move that keeps billboards with a viewport
/// moving on the desktop. Pure; <see cref="Services.DesktopBillboardService"/> applies it.
/// </summary>
public static class BillboardViewportPlacement
{
  /// <summary>The billboard's centre as an offset from <paramref name="viewport"/>'s top-left.</summary>
  public static ViewportOffset OffsetOf(DesktopBillboard b, ViewportFrame viewport) =>
    new(b.CenterX - viewport.Origin.X, b.CenterY - viewport.Origin.Y, viewport.Scaling);

  /// <summary>
  /// The centre a saved <paramref name="offset"/> gives on <paramref name="viewport"/>: the offset
  /// is in physical px of the saving screen, so it is scaled by the ratio of the scalings (a save
  /// at 100 % opened at 150 % sits 1.5× as far from the corner, the same distance in DIPs).
  /// </summary>
  public static PointD CenterFor(ViewportOffset offset, ViewportFrame viewport)
  {
    double k = offset.Scaling > 0 && viewport.Scaling > 0 ? viewport.Scaling / offset.Scaling : 1.0;
    return new PointD(viewport.Origin.X + offset.X * k, viewport.Origin.Y + offset.Y * k);
  }

  /// <summary>
  /// Places restored billboards on <paramref name="viewport"/>: one with a saved offset goes to
  /// <see cref="CenterFor"/>. Those without (an older save, desktop positions only) keep their
  /// layout: unless every one of them is already over the viewport they move together so their
  /// centroid lands on the viewport's centre (the save was made with the app on another monitor,
  /// or the group straddles the viewport: never some on it and some off it).
  /// </summary>
  public static ImmutableArray<DesktopBillboard> Place(
    ImmutableArray<DesktopBillboard> billboards, Func<Guid, ViewportOffset?> savedOffsetOf, ViewportFrame viewport)
  {
    var next = billboards.ToBuilder();
    var legacy = new List<int>();
    for (int i = 0; i < next.Count; i++)
    {
      if (savedOffsetOf(next[i].Id) is { } offset)
      {
        var c = CenterFor(offset, viewport);
        next[i] = next[i] with { CenterX = c.X, CenterY = c.Y };
      }
      else
      {
        legacy.Add(i);
      }
    }
    if (legacy.Count > 0 && !legacy.All(i => viewport.Contains(DesktopBillboardGeometry.Center(next[i]))))
    {
      var centroid = new PointD(
        legacy.Average(i => next[i].CenterX),
        legacy.Average(i => next[i].CenterY));
      var delta = viewport.Center - centroid;
      foreach (int i in legacy)
        next[i] = DesktopBillboardGeometry.Translate(next[i], delta);
    }
    return next.ToImmutable();
  }

  /// <summary>
  /// Keeps billboards with a viewport that went from <paramref name="was"/> to
  /// <paramref name="now"/> (the window dragged, maximised on another monitor, resized): each
  /// centre keeps its offset from the viewport's top-left corner, scaled with the screen's
  /// scaling, like a view constrained to its parent's corner. A move is a translation by the
  /// move's vector; a resize leaves every billboard at the same distance from the corner (the
  /// ones on the comet are put back on it afterwards by their anchor).
  /// <paramref name="pinned"/> billboards (one the user is dragging) are left alone.
  /// </summary>
  public static ImmutableArray<DesktopBillboard> Follow(
    ImmutableArray<DesktopBillboard> billboards, ViewportFrame was, ViewportFrame now, Func<Guid, bool> pinned)
  {
    if (was.Origin == now.Origin && was.Scaling == now.Scaling)
      return billboards;
    double k = was.Scaling > 0 && now.Scaling > 0 ? now.Scaling / was.Scaling : 1.0;
    var o0 = was.Origin;
    var o1 = now.Origin;
    return billboards.Select(b => pinned(b.Id)
        ? b
        : b with { CenterX = o1.X + (b.CenterX - o0.X) * k, CenterY = o1.Y + (b.CenterY - o0.Y) * k })
      .ToImmutableArray();
  }
}
