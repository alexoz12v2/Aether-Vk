using System.Numerics;
using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Transient operator: Ctrl + Middle-Mouse drag.
/// Uses the vertical (Y-axis) component of the pointer delta as dolly input.
/// Forwards to <see cref="Services.CameraService.RequestZoom"/>.
///
/// In <c>CometOrbiting</c> mode, zoom adjusts the orbit-radius offset rather than
/// directly calling <c>TransformStaticCamera</c> (which is always blocked in that mode
/// by the continuous tracking animation).
/// </summary>
internal sealed class ZoomCameraOperator(Viewport3DViewModel vm, Vector2 startPos)
  : IActionOperator
{
  private Vector2 _lastPos = startPos;

  public void OnEnter() { }
  public void OnExit()  { }

  public bool ProcessAction(AppAction action, InputState state)
  {
    switch (action.Id)
    {
      case "viewport.pointer_delta":
        if (action.Payload is not Vector2 cur) return false;
        vm.CameraService.RequestZoom((cur - _lastPos).Y, state.Modifiers);
        _lastPos = cur;
        return true;

      case "viewport.pointer_end":
        vm.OperatorStack.PopSelf(this);
        return true;
    }
    return false;
  }
}
