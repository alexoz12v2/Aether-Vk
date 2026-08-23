using AetherVk.Logic.Input;
using Xunit;

namespace AetherVk.Logic.Tests;

public class OperatorStackTests
{
  private class MockOperator : IActionOperator
  {
    public bool IsEntered { get; private set; }
    public bool IsExited { get; private set; }
    public bool HandledAction { get; set; } = false;

    public void OnEnter() => IsEntered = true;

    public void OnExit() => IsExited = true;

    public bool ProcessAction(AppAction action, InputState state) => HandledAction;
  }

  [Fact]
  public void PushAndPop_ManagesLifecycleMethods()
  {
    var baseOp = new MockOperator();
    var stack = new OperatorStack(baseOp);

    Assert.True(baseOp.IsEntered);
    Assert.False(baseOp.IsExited);

    var secondOp = new MockOperator();
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

    var secondOp = new MockOperator { HandledAction = true };
    stack.Push(secondOp);

    bool result = stack.Process(new AppAction("test"), new InputState(isPressed: true, InputModifiers.None));

    Assert.True(result);
  }
}
