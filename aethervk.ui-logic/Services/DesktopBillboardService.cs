using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
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
  private readonly Subject<Unit> _restored = new();
  private readonly object _gate = new();
  // the viewport as last followed (FollowViewport); null until the app reports it
  private ViewportFrame? _viewport;
  // a savefile open waiting for the viewport: each restored billboard's saved offset (null: an
  // older save, desktop position only); null when nothing waits
  private Dictionary<Guid, ViewportOffset?>? _awaitingViewport;
  // billboards the user is dragging: a viewport move does not touch them
  private readonly HashSet<Guid> _dragging = [];
  // the comet on the desktop as last reported (AnchorOnComet); null while it is out of view
  private PointD? _comet;
  // comet-anchored billboards the user moved off the comet: free until put back on it
  private readonly HashSet<Guid> _released = [];
  // where each billboard being dragged was when its drag started: a drop that moved it off the
  // comet releases it, a turn or a scale about its centre does not
  private readonly Dictionary<Guid, PointD> _dragStart = [];

  /// <summary>How far (desktop px) a comet-anchored billboard's nucleus may be dropped from the
  /// comet and still snap back onto it; farther releases the anchor.</summary>
  public const double AnchorReleasePx = 4.0;

  /// <summary>Every change of the list (creation, edit, removal), current value first.</summary>
  public IObservable<ImmutableArray<DesktopBillboard>> Billboards => _billboards;

  /// <summary>A savefile open replaced the list (<see cref="RestoreOverViewport"/>): the
  /// comparison billboards go back on the comet once the viewport shows the restored view.</summary>
  public IObservable<Unit> Restored => _restored.AsObservable();

  /// <summary>The viewport on the desktop as last reported (null before the first report).</summary>
  public ViewportFrame? ViewportFrame => _viewport;

  public ImmutableArray<DesktopBillboard> Current => _billboards.Value;

  public DesktopBillboard? Find(Guid id) => Current.FirstOrDefault(b => b.Id == id);

  /// <summary>A comparison billboard: bound to its physical size or with a nucleus pixel. Its
  /// nucleus sits on the comet while the comet is in view (<see cref="AnchorOnComet"/>), unless the
  /// user moved it off (<see cref="IsAnchored"/>).</summary>
  public static bool IsCometAnchored(DesktopBillboard b) => b.PhysicalSizeBound || b.ImageNucleusPx is not null;

  /// <summary>The billboard follows the comet now: comet-anchored and not moved off it.</summary>
  public bool IsAnchored(Guid id)
  {
    lock (_gate)
      return Find(id) is { } b && IsCometAnchored(b) && !_released.Contains(id);
  }

  /// <summary>The comet on the desktop as last reported to <see cref="AnchorOnComet"/>.</summary>
  public PointD? Comet => _comet;

  /// <summary>
  /// Adds a billboard of <paramref name="texture"/> at the centre of <paramref name="screen"/>, at
  /// its native size (<see cref="DesktopBillboardGeometry.InitialSize"/>).
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
        // the user moved it (a row edit; a drag is judged at its drop): off the comet releases
        // the anchor, back on it ("Center on comet") takes it again
        if (!_dragging.Contains(clean.Id)
            && (clean.CenterX != list[i].CenterX || clean.CenterY != list[i].CenterY))
          JudgeAnchor(clean);
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

  /// <summary>Releases or re-takes <paramref name="b"/>'s comet anchor by where its nucleus pixel
  /// now is against the comet as last reported (nothing to judge while the comet is out of view).</summary>
  private void JudgeAnchor(DesktopBillboard b)
  {
    if (!IsCometAnchored(b) || _comet is not { } comet)
      return;
    var d = DesktopBillboardGeometry.NucleusDesktop(b) - comet;
    if (Math.Sqrt(d.X * d.X + d.Y * d.Y) > AnchorReleasePx)
      _released.Add(b.Id);
    else
      _released.Remove(b.Id);
  }

  /// <summary>
  /// The comet is at <paramref name="comet"/> on the desktop (null: out of view; the viewport
  /// reports it every frame and every probe): each anchored billboard
  /// (<see cref="IsAnchored"/>, not being dragged) is moved so its nucleus pixel sits on it
  /// (<see cref="DesktopBillboardGeometry.CenterForNucleusAt"/>). Out of view, they stay where the
  /// viewport's moves take them, like free billboards.
  /// </summary>
  public void AnchorOnComet(PointD? comet)
  {
    lock (_gate)
    {
      _comet = comet;
      if (comet is not { } c)
        return;
      var list = Current;
      var next = list.Select(b =>
      {
        if (!IsCometAnchored(b) || _released.Contains(b.Id) || _dragging.Contains(b.Id))
          return b;
        var centre = DesktopBillboardGeometry.CenterForNucleusAt(b, c);
        return b with { CenterX = centre.X, CenterY = centre.Y };
      }).ToImmutableArray();
      if (!next.SequenceEqual(list))
        _billboards.OnNext(next);
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
      _dragging.Remove(id);
      _dragStart.Remove(id);
      _released.Remove(id);
      _billboards.OnNext(next);
      return true;
    }
  }

  /// <summary>Replaces the whole list (New; tests). The billboards stay where they are; every
  /// comparison billboard is anchored again.</summary>
  public void Restore(IEnumerable<DesktopBillboard> billboards)
  {
    lock (_gate)
    {
      _awaitingViewport = null;
      _released.Clear();
      _billboards.OnNext(Clean(billboards));
    }
  }

  /// <summary>
  /// Replaces the whole list from a savefile, each billboard with its saved offset from the
  /// viewport (null: an older save). They are placed on the viewport as soon as it is known
  /// (<see cref="BillboardViewportPlacement.Place"/>): now when it already is, else at the first
  /// <see cref="FollowViewport"/>. Every comparison billboard is anchored to the comet again. Then
  /// <see cref="Restored"/> fires.
  /// </summary>
  public void RestoreOverViewport(IEnumerable<(DesktopBillboard Billboard, ViewportOffset? SavedOffset)> billboards)
  {
    lock (_gate)
    {
      var list = billboards.ToList();
      var offsets = list.ToDictionary(x => x.Billboard.Id, x => x.SavedOffset);
      _released.Clear();
      var clean = Clean(list.Select(x => x.Billboard));
      if (_viewport is { } viewport)
      {
        _awaitingViewport = null;
        clean = BillboardViewportPlacement.Place(clean, id => offsets.TryGetValue(id, out var o) ? o : null, viewport);
      }
      else
      {
        _awaitingViewport = offsets;
      }
      _billboards.OnNext(clean);
    }
    _restored.OnNext(Unit.Default);
  }

  private static ImmutableArray<DesktopBillboard> Clean(IEnumerable<DesktopBillboard> billboards) =>
    billboards.Select(b => Sanitize(b with { GizmoMode = BillboardGizmoMode.None })).ToImmutableArray();

  /// <summary>
  /// The viewport is at <paramref name="viewport"/> on the desktop (the app reports it when laid
  /// out, moved or resized): billboards waiting for it since a savefile open are placed on it,
  /// the others follow its move (<see cref="BillboardViewportPlacement.Follow"/>), except one being
  /// dragged.
  /// </summary>
  public void FollowViewport(ViewportFrame viewport)
  {
    lock (_gate)
    {
      var list = Current;
      var next = list;
      if (_awaitingViewport is { } offsets)
      {
        next = BillboardViewportPlacement.Place(next, id => offsets.TryGetValue(id, out var o) ? o : null, viewport);
        _awaitingViewport = null;
      }
      else if (_viewport is { } was)
      {
        next = BillboardViewportPlacement.Follow(next, was, viewport, _dragging.Contains);
      }
      _viewport = viewport;
      if (!next.SequenceEqual(list))
        _billboards.OnNext(next);
    }
  }

  /// <summary>The billboard's centre against the viewport as last reported, for a save; null
  /// before the viewport is known (then a saved offset still waiting for it is kept).</summary>
  public ViewportOffset? ViewportOffsetOf(DesktopBillboard b)
  {
    lock (_gate)
    {
      if (_awaitingViewport is { } offsets && offsets.TryGetValue(b.Id, out var saved))
        return saved;
      return _viewport is { } viewport ? BillboardViewportPlacement.OffsetOf(b, viewport) : null;
    }
  }

  /// <summary>
  /// The user is dragging the billboard (or stopped): a viewport move and the comet anchor leave
  /// it alone meanwhile. A drop that moved a comet-anchored billboard's centre and left its nucleus
  /// off the comet releases the anchor (the user's placement is never overridden); a drop back on
  /// the comet, or a turn or scale about its centre, keeps it anchored and it snaps back.
  /// </summary>
  public void SetDragging(Guid id, bool dragging)
  {
    lock (_gate)
    {
      if (dragging)
      {
        if (_dragging.Add(id) && Find(id) is { } b)
          _dragStart[id] = DesktopBillboardGeometry.Center(b);
        return;
      }
      _dragging.Remove(id);
      if (_dragStart.TryGetValue(id, out var start) && Find(id) is { } dropped
          && DesktopBillboardGeometry.Center(dropped) != start)
        JudgeAnchor(dropped);
      _dragStart.Remove(id);
      if (IsAnchored(id) && _comet is not null)
        AnchorOnComet(_comet); // snaps back at once, not at the next frame
    }
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
  /// rotation into (−180, 180], a non-finite nucleus pixel dropped.</summary>
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
      ImageNucleusPx = b.ImageNucleusPx is { } n && IsFinite(n.X) && IsFinite(n.Y) ? n : null,
    };

  private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

  public void Dispose()
  {
    _billboards.Dispose();
    _restored.Dispose();
  }
}
