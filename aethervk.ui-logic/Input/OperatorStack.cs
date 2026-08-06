using System;
using System.Collections.Generic;

namespace AetherVk.Logic.Input;

public class OperatorStack : IActionHandler
{
  private readonly Stack<IActionOperator> _stack = new();

  public OperatorStack(IActionOperator baseOperator)
  {
    Push(baseOperator);
  }

  public void Push(IActionOperator op)
  {
    if (_stack.Count > 0 && _stack.Peek() == op)
      throw new InvalidOperationException($"Operator {op.GetType().Name} is already active on this stack");

    op.OnEnter();
    _stack.Push(op);
  }

  public void PopSelf(IActionOperator self)
  {
    // Don't allow popping the base operator
    if (_stack.Count <= 1)
      return;

    // 1. Verify the top of the stack is actually 'self'
    if (_stack.Peek() != self)
    {
      throw new InvalidOperationException(
        $"Transient Action Operators should be the first to be popped. " +
        $"Expected to pop '{_stack.Peek().GetType().Name}', but '{self.GetType().Name}' tried to pop itself."
      );
    }

    // 2. We are safe to pop
    _stack.Pop().OnExit();
  }

  public bool Process(AppAction action, InputState state)
  {
    if (_stack.Count > 0)
    {
      return _stack.Peek().ProcessAction(action, state);
    }
    return false;
  }
}
