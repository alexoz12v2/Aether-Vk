using System;
using System.Collections.Generic;
using System.Diagnostics;

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
    // Silently ignore pushing the same instance or same type that is already on top.
    // ViewportBaseOperator always creates new instances, so reference equality alone
    // cannot guard against duplicate type pushes from rapid/multi-touch events.
    if (_stack.Count > 0 && (_stack.Peek() == op || _stack.Peek().GetType() == op.GetType()))
    {
      Console.WriteLine(
        $"[OperatorStack] Push SKIPPED (duplicate): {op.GetType().Name}, top={_stack.Peek().GetType().Name}"
      );
      return;
    }

    Console.WriteLine(
      $"[OperatorStack] Push: {op.GetType().Name} (depth {_stack.Count} → {_stack.Count + 1})"
    );
    op.OnEnter();
    _stack.Push(op);
  }

  public void PopSelf(IActionOperator self)
  {
    // Don't allow popping the base operator
    if (_stack.Count <= 1)
    {
      Console.WriteLine($"[OperatorStack] PopSelf SKIPPED (base operator): {self.GetType().Name}");
      return;
    }

    Console.WriteLine($"[OperatorStack] PopSelf: {self.GetType().Name} (depth {_stack.Count} → ?)");

    // Assert in debug builds; gracefully recover in release to avoid a field crash
    // from any out-of-order pointer events (e.g. multi-touch interleaving).
    Debug.Assert(
      _stack.Peek() == self,
      $"Out-of-order pop: expected '{_stack.Peek().GetType().Name}', got '{self.GetType().Name}'."
    );

    // Pop until we find 'self' or only the base operator remains.
    while (_stack.Count > 1 && _stack.Peek() != self)
      _stack.Pop().OnExit();
    if (_stack.Count > 1)
      _stack.Pop().OnExit(); // pop 'self'
    Console.WriteLine($"[OperatorStack] PopSelf done, depth now {_stack.Count}");
  }

  /// <summary>
  /// Pops all transient operators down to the base operator, calling <see cref="IActionOperator.OnExit"/>
  /// for each. Called when the viewport loses pointer capture to prevent stuck operators.
  /// </summary>
  public void ForceReset()
  {
    Console.WriteLine($"[OperatorStack] ForceReset (depth {_stack.Count})");
    while (_stack.Count > 1)
      _stack.Pop().OnExit();
  }

  public bool Process(AppAction action, InputState state)
  {
    if (_stack.Count > 0)
    {
      Console.WriteLine(
        $"[OperatorStack] Process: action={action.Id} isPressed={state.IsPressed} → top={_stack.Peek().GetType().Name}"
      );
      return _stack.Peek().ProcessAction(action, state);
    }
    Console.WriteLine($"[OperatorStack] Process: action={action.Id} — stack empty!");
    return false;
  }
}
