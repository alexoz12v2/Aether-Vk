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
      h);
    lock (_gate)
      _billboards.OnNext(Current.Add(b));
    return b;
  }

  /// <summary>Replaces the billboard with the same id (geometry, alpha, texture, gizmo …).</summary>
  public bool Update(DesktopBillboard updated)
  {
    lock (_gate)
    {
      var list = Current;
      for (int i = 0; i < list.Length; i++)
      {
        if (list[i].Id != updated.Id)
          continue;
        var clean = Sanitize(updated);
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
    };

  public void Dispose() => _billboards.Dispose();
}
