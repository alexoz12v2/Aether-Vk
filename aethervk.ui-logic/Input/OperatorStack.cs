using System.Collections.Generic;

namespace AetherVk.Logic.Input;

public class OperatorStack : IActionHandler
{
  private readonly Stack<IActionOperator> _stack = new();

  public OperatorStack(IActionOperator baseOperator)
  {
    Push(baseOperator);
  }

  public bool IsCameraControlEngaged => _stack.Count > 1;

  public bool IsCameraControlEnabled { get; set; } = true;

  public void Push(IActionOperator op)
  {
    if (!IsCameraControlEnabled && _stack.Count > 0)
      return;
    op.OnEnter();
    _stack.Push(op);
  }

  public void Pop()
  {
    if (_stack.Count > 1)
      _stack.Pop().OnExit();
  }

  public bool ProcessAction(AppAction action, bool isPressed)
  {
    if (
      !IsCameraControlEnabled
      && _stack.Count > 0
      && action.Id != "viewport.toggle_measuring"
      && action.Id != "viewport.open_radial_menu"
    )
      return false;
    if (_stack.Count > 0)
    {
      return _stack.Peek().ProcessAction(action, isPressed);
    }
    return false;
  }

  public bool ProcessPointerDelta(float dx, float dy)
  {
    if (!IsCameraControlEnabled && _stack.Count > 0)
      return false;
    if (_stack.Count > 0)
    {
      return _stack.Peek().ProcessPointerDelta(dx, dy);
    }
    return false;
  }

  public bool ProcessPointerWheel(float deltaY)
  {
    if (!IsCameraControlEnabled && _stack.Count > 0)
      return false;
    if (_stack.Count > 0)
    {
      return _stack.Peek().ProcessPointerWheel(deltaY);
    }
    return false;
  }
}
