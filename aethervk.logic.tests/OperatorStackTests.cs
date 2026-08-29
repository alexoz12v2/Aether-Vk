using AetherVk.Logic.Input;
using Xunit;

namespace AetherVk.Logic.Tests;

public class OperatorStackTests
{
  /// <summary>Base-operator mock — represents ViewportBaseOperator.</summary>
  private class MockOperator : IActionOperator
  {
    public bool IsEntered { get; private set; }
    public bool IsExited { get; private set; }
    public bool HandledAction { get; set; } = false;

    public void OnEnter() => IsEntered = true;

    public void OnExit() => IsExited = true;

    public bool ProcessAction(AppAction action, InputState state) => HandledAction;
  }

  /// <summary>
  /// Distinct type for transient operators (mirrors OrbitCameraOperator, PanCameraOperator, etc.).
  /// Needed because the duplicate-type guard on <see cref="OperatorStack.Push"/> compares
  /// <c>GetType()</c> — using the same type for both base and transient would wrongly block
  /// the first legitimate transient push.
  /// </summary>
  private class MockTransientOperator : IActionOperator
  {
    public bool IsEntered { get; private set; }
    public bool IsExited { get; private set; }
    public bool HandledAction { get; set; } = true;

    public void OnEnter() => IsEntered = true;
    public void OnExit()  => IsExited  = true;
    public bool ProcessAction(AppAction action, InputState state) => HandledAction;
  }

  [Fact]
  public void PushAndPop_ManagesLifecycleMethods()
  {
    var baseOp = new MockOperator();
    var stack = new OperatorStack(baseOp);

    Assert.True(baseOp.IsEntered);
    Assert.False(baseOp.IsExited);

    var secondOp = new MockTransientOperator(); // distinct type — must be allowed through
    stack.Push(secondOp);

    Assert.True(secondOp.IsEntered);

    stack.PopSelf(secondOp);

    Assert.True(secondOp.IsExited);

    // Ensure base operator isn't popped
    stack.PopSelf(baseOp);
    Assert.False(baseOp.IsExited);
  }

  [Fact]
  public void ProcessAction_DelegatesToTopOperator()
  {
    var baseOp = new MockOperator { HandledAction = false };
    var stack = new OperatorStack(baseOp);

    var secondOp = new MockTransientOperator { HandledAction = true }; // distinct type
    stack.Push(secondOp);

    bool result = stack.Process(new AppAction("test"), new InputState(isPressed: true, InputModifiers.None));

    Assert.True(result);
  }

  [Fact]
  public void Push_DuplicateType_IsIgnoredSilently()
  {
    var baseOp = new MockOperator();
    var stack = new OperatorStack(baseOp);

    var first = new MockTransientOperator();
    stack.Push(first);
    Assert.True(first.IsEntered);

    var second = new MockTransientOperator(); // same type as first — must be blocked
    stack.Push(second);

    // Should be silently ignored — second operator must not have been entered
    Assert.False(second.IsEntered);
  }

  [Fact]
  public void PopSelf_OutOfOrder_DoesNotThrow()
  {
    var baseOp = new MockOperator();
    var stack = new OperatorStack(baseOp);
    var op = new MockTransientOperator();
    stack.Push(op);

    // Force reset so 'op' is no longer on top, then try to PopSelf — must not throw
    stack.ForceReset();
    var ex = Record.Exception(() => stack.PopSelf(op));
    Assert.Null(ex);
  }

  [Fact]
  public void ForceReset_ClearsAllTransientOperators()
  {
    var baseOp = new MockOperator();
    var stack = new OperatorStack(baseOp);

    var transient = new MockTransientOperator();
    stack.Push(transient);
    Assert.True(transient.IsEntered);

    stack.ForceReset();

    // Transient operator must have had OnExit called
    Assert.True(transient.IsExited);

    // After reset, pushing a new transient of the same type must succeed again
    var second = new MockTransientOperator();
    stack.Push(second);
    Assert.True(second.IsEntered);
  }
}
