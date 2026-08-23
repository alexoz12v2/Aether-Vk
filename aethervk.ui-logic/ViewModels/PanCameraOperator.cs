using System.Numerics;
using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Transient operator: Shift + Middle-Mouse drag.
/// Forwards pixel deltas to <see cref="Services.CameraService.RequestPan"/>.
/// Allowed in all three camera modes (pan is structurally always permitted).
/// </summary>
internal sealed class PanCameraOperator(Viewport3DViewModel vm, Vector2 startPos)
  : IActionOperator
{
  private Vector2 _lastPos = startPos;

  public void OnEnter() => vm.IsPanning = true;
  public void OnExit()  => vm.IsPanning = false;

  public bool ProcessAction(AppAction action, InputState state)
  {
    switch (action.Id)
    {
      case "viewport.pointer_delta":
        if (action.Payload is not Vector2 cur) return false;
        vm.CameraService.RequestPan(cur - _lastPos, state.Modifiers);
        _lastPos = cur;
        return true;

      case "viewport.pointer_end":
        vm.OperatorStack.PopSelf(this);
        return true;
    }
    return false;
  }
}
