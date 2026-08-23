using System.Numerics;
using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Transient operator: active while the user drags Middle-Mouse with no modifier key.
/// Forwards screen-space pixel deltas to <see cref="Services.CameraService.RequestOrbit"/>.
/// Sensitivity and Shift fine-control are handled inside CameraService.
/// Pops itself when the pointer is released (<c>viewport.pointer_end</c>).
/// </summary>
internal sealed class OrbitCameraOperator(Viewport3DViewModel vm, Vector2 startPos)
  : IActionOperator
{
  private Vector2 _lastPos = startPos;

  public void OnEnter() => vm.IsOrbiting = true;
  public void OnExit()  => vm.IsOrbiting = false;

  public bool ProcessAction(AppAction action, InputState state)
  {
    switch (action.Id)
    {
      case "viewport.pointer_delta":
        if (action.Payload is not Vector2 cur) return false;
        vm.CameraService.RequestOrbit(cur - _lastPos, state.Modifiers, Vector3.Zero);
        _lastPos = cur;
        return true;

      case "viewport.pointer_end":
        vm.OperatorStack.PopSelf(this);
        return true;
    }
    return false;
  }
}
