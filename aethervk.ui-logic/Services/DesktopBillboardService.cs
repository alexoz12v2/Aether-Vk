using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

/// <summary>
/// The desktop billboards (imported textures shown in their own windows over the desktop): the
/// single source of truth that the Settings list and the billboard windows both observe and edit.
/// UI-framework free; the windows live in the app (<c>DesktopBillboardWindowManager</c>).
/// </summary>
public sealed class DesktopBillboardService : IDisposable
{
  private readonly BehaviorSubject<ImmutableArray<DesktopBillboard>> _billboards =
    new(ImmutableArray<DesktopBillboard>.Empty);
  private readonly object _gate = new();

  /// <summary>Every change of the list (creation, edit, removal), current value first.</summary>
  public IObservable<ImmutableArray<DesktopBillboard>> Billboards => _billboards;

  public ImmutableArray<DesktopBillboard> Current => _billboards.Value;

  public DesktopBillboard? Find(Guid id) => Current.FirstOrDefault(b => b.Id == id);

  /// <summary>
  /// Adds a billboard of <paramref name="texture"/> at the centre of <paramref name="screen"/>,
  /// native size shrunk to fit half the display (<see cref="DesktopBillboardGeometry.InitialSize"/>).
  /// </summary>
  public DesktopBillboard Create(ImportedAsset texture, ScreenInfo screen)
  {
    if (texture.Kind != AssetKind.Texture)
      throw new ArgumentException("billboards show textures", nameof(texture));
    var (w, h) = DesktopBillboardGeometry.InitialSize((int)texture.A, (int)texture.B, screen);
    var b = new DesktopBillboard(
      Guid.NewGuid(),
      texture.Key,
      screen.Bounds.X + screen.Bounds.Width / 2.0,
      screen.Bounds.Y + screen.Bounds.Height / 2.0,
      w,
      h)
    {
      ImageWidthPx = (int)Math.Min(int.MaxValue, texture.A),
      ImageHeightPx = (int)Math.Min(int.MaxValue, texture.B),
    };
    lock (_gate)
      _billboards.OnNext(Current.Add(b));
    return b;
  }

  /// <summary>Replaces the billboard with the same id (geometry, alpha, texture, gizmo …). A size
  /// change of a billboard bound to its physical size (a row edit, a scale drag) unbinds it.</summary>
  public bool Update(DesktopBillboard updated) => Replace(updated, keepBinding: false);

  private bool Replace(DesktopBillboard updated, bool keepBinding)
  {
    lock (_gate)
    {
      var list = Current;
      for (int i = 0; i < list.Length; i++)
      {
        if (list[i].Id != updated.Id)
          continue;
        var clean = Sanitize(updated);
        if (!keepBinding && list[i].PhysicalSizeBound && clean.PhysicalSizeBound
            && (clean.WidthPx != list[i].WidthPx || clean.HeightPx != list[i].HeightPx))
          clean = clean with { PhysicalSizeBound = false };
        if (list[i] != clean)
          _billboards.OnNext(list.SetItem(i, clean));
        return true;
      }
      return false;
    }
  }

  public bool Update(Guid id, Func<DesktopBillboard, DesktopBillboard> edit)
  {
    lock (_gate)
    {
      var b = Find(id);
      return b is not null && Update(edit(b));
    }
  }

  /// <summary>
  /// "Apply physical size": sizes the billboard to cover the sky its image covers on a viewport of
  /// <paramref name="viewportArcsecPerPx"/> (see <see cref="DesktopBillboardGeometry.PhysicalSize"/>)
  /// and binds it, so <see cref="RescaleBound"/> keeps it matched. False without a physical size.
  /// </summary>
  public bool ApplyPhysicalSize(Guid id, double viewportArcsecPerPx, double? cometDistanceKm = null)
  {
    lock (_gate)
    {
      if (Find(id) is not { } b
          || DesktopBillboardGeometry.PhysicalSize(b, viewportArcsecPerPx, cometDistanceKm) is not { } size)
        return false;
      return Replace(b with { WidthPx = size.WidthPx, HeightPx = size.HeightPx, PhysicalSizeBound = true }, keepBinding: true);
    }
  }

  /// <summary>
  /// "Match camera to image": the billboard fills <paramref name="viewportHeightPx"/> (the
  /// viewport's height, now the image's field), the width from the image's aspect, and is bound so
  /// <see cref="RescaleBound"/> keeps it matched. False without the billboard or a height.
  /// </summary>
  public bool ApplyViewportHeight(Guid id, double viewportHeightPx)
  {
    lock (_gate)
    {
      if (Find(id) is not { } b || !(viewportHeightPx > 0) || double.IsInfinity(viewportHeightPx))
        return false;
      double aspect = b.ImageWidthPx > 0 && b.ImageHeightPx > 0
        ? (double)b.ImageWidthPx / b.ImageHeightPx
        : b.HeightPx > 0 ? b.WidthPx / b.HeightPx : 1.0;
      return Replace(
        b with { WidthPx = viewportHeightPx * aspect, HeightPx = viewportHeightPx, PhysicalSizeBound = true },
        keepBinding: true);
    }
  }

  /// <summary>
  /// Re-sizes every bound billboard for a new viewport scale (zoom, resize) or comet distance:
  /// its height stays the image's angular height over the viewport's angle per px, so a billboard
  /// matched to the viewport's field keeps filling it.
  /// </summary>
  public void RescaleBound(double viewportArcsecPerPx, double? cometDistanceKm = null)
  {
    lock (_gate)
    {
      var list = Current;
      var next = list.Select(b =>
        b.PhysicalSizeBound && DesktopBillboardGeometry.PhysicalSize(b, viewportArcsecPerPx, cometDistanceKm) is { } size
          ? Sanitize(b with { WidthPx = size.WidthPx, HeightPx = size.HeightPx })
          : b).ToImmutableArray();
      if (!next.SequenceEqual(list))
        _billboards.OnNext(next);
    }
  }

  public bool Remove(Guid id)
  {
    lock (_gate)
    {
      var list = Current;
      var next = list.RemoveAll(b => b.Id == id);
      if (next.Length == list.Length)
        return false;
      _billboards.OnNext(next);
      return true;
    }
  }

  /// <summary>Replaces the whole list (savefile load / New).</summary>
  public void Restore(IEnumerable<DesktopBillboard> billboards)
  {
    lock (_gate)
      _billboards.OnNext(billboards.Select(b => Sanitize(b with { GizmoMode = BillboardGizmoMode.None })).ToImmutableArray());
  }

  /// <summary>Moves every billboard whose centre left all displays back onto the primary one.</summary>
  public void KeepOnScreen(IReadOnlyList<ScreenInfo> screens)
  {
    lock (_gate)
    {
      var list = Current;
      var next = list.Select(b => DesktopBillboardGeometry.KeepOnScreen(b, screens)).ToImmutableArray();
      if (!next.SequenceEqual(list))
        _billboards.OnNext(next);
    }
  }

  /// <summary>Clamps what a hand-edited value could break: alpha to [0, 1], sides ≥ the minimum,
  /// rotation into (−180, 180].</summary>
  private static DesktopBillboard Sanitize(DesktopBillboard b) =>
    b with
    {
      Alpha = double.IsNaN(b.Alpha) ? 1 : Math.Max(0, Math.Min(1, b.Alpha)),
      WidthPx = Math.Max(DesktopBillboardGeometry.MinSidePx, b.WidthPx),
      HeightPx = Math.Max(DesktopBillboardGeometry.MinSidePx, b.HeightPx),
      RotationDeg = DesktopBillboardGeometry.NormalizeDeg(b.RotationDeg),
      ImageArcsecPerPx = b.ImageArcsecPerPx is { } a && a > 0 && !double.IsInfinity(a) ? a : null,
      ImageHeightKm = b.ImageHeightKm is { } km && km > 0 && !double.IsInfinity(km) ? km : null,
      ImageWidthPx = Math.Max(0, b.ImageWidthPx),
      ImageHeightPx = Math.Max(0, b.ImageHeightPx),
    };

  public void Dispose() => _billboards.Dispose();
}
