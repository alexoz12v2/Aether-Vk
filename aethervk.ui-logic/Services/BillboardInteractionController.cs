using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

public enum BillboardPointerButton
{
  Left,
  Right,
  Other,
}

/// <summary>
/// Pointer rules of a desktop billboard window, as a pure state machine over desktop pixels:
/// <list type="bullet">
///   <item>left press + drag moves the billboard, whatever gizmo is shown;</item>
///   <item>right press on a gizmo handle + drag edits through that handle (translate, rotate with
///   Ctrl = 15° snap, corner scale with Shift = keep aspect);</item>
///   <item>a right click (press and release within <see cref="DesktopBillboardGeometry.ClickThresholdPx"/>)
///   off every handle cycles the gizmo None → Translate → Rotate → Scale → None.</item>
/// </list>
/// Every call returns the billboard to apply (or null when nothing changes); the caller hands it
/// to <see cref="DesktopBillboardService.Update"/>.
/// </summary>
public sealed class BillboardInteractionController
{
  private enum State
  {
    Idle,
    Moving,
    RightOnHandle,
    RightOffHandle,
  }

  private State _state = State.Idle;
  private PointD _start;
  private DesktopBillboard? _b0;
  private BillboardHandle _handle;
  private bool _movedPastThreshold;

  public bool IsDragging => _state is State.Moving or State.RightOnHandle;

  public DesktopBillboard? PointerPressed(DesktopBillboard b, BillboardPointerButton button, PointD p)
  {
    if (_state != State.Idle || button == BillboardPointerButton.Other)
      return null;
    _start = p;
    _b0 = b;
    _movedPastThreshold = false;
    if (button == BillboardPointerButton.Left)
    {
      _state = State.Moving;
      return null;
    }
    var hit = DesktopBillboardGeometry.HitTest(b, b.GizmoMode, p);
    if (hit is { } h)
    {
      _handle = h;
      _state = State.RightOnHandle;
    }
    else
    {
      _state = State.RightOffHandle;
    }
    return null;
  }

  public DesktopBillboard? PointerMoved(PointD p, bool ctrl, bool shift)
  {
    if (_b0 is null)
      return null;
    var delta = p - _start;
    if (delta.Length >= DesktopBillboardGeometry.ClickThresholdPx)
      _movedPastThreshold = true;
    switch (_state)
    {
      case State.Moving:
        return DesktopBillboardGeometry.Translate(_b0, delta);
      case State.RightOnHandle:
        return _handle switch
        {
          BillboardHandle.MoveFree => DesktopBillboardGeometry.Translate(_b0, delta),
          BillboardHandle.MoveAlongX => DesktopBillboardGeometry.TranslateAlong(_b0, delta, alongX: true),
          BillboardHandle.MoveAlongY => DesktopBillboardGeometry.TranslateAlong(_b0, delta, alongX: false),
          BillboardHandle.Rotate => DesktopBillboardGeometry.RotateDrag(_b0, _start, p, snap: ctrl),
          _ => DesktopBillboardGeometry.ScaleDrag(_b0, _handle, p, keepAspect: shift),
        };
      default:
        return null;
    }
  }

  public DesktopBillboard? PointerReleased(BillboardPointerButton button, PointD p, bool ctrl, bool shift)
  {
    var b0 = _b0;
    var state = _state;
    DesktopBillboard? result = null;
    if (b0 is not null)
    {
      if (state == State.RightOffHandle && button == BillboardPointerButton.Right)
      {
        var moved = (p - _start).Length >= DesktopBillboardGeometry.ClickThresholdPx || _movedPastThreshold;
        if (!moved)
          result = b0 with { GizmoMode = DesktopBillboardGeometry.NextMode(b0.GizmoMode) };
      }
      else if (state is State.Moving or State.RightOnHandle)
      {
        result = PointerMoved(p, ctrl, shift);
      }
    }
    _state = State.Idle;
    _b0 = null;
    return result;
  }

  /// <summary>Drops any drag in progress (pointer capture lost).</summary>
  public void Cancel()
  {
    _state = State.Idle;
    _b0 = null;
  }
}
