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
        if (_stack.Count > 0)
        {
            return _stack.Peek().ProcessAction(action, isPressed);
        }
        return false;
    }

    public bool ProcessPointerDelta(float dx, float dy)
    {
        if (_stack.Count > 0)
        {
            return _stack.Peek().ProcessPointerDelta(dx, dy);
        }
        return false;
    }

    public bool ProcessPointerWheel(float deltaY)
    {
        if (_stack.Count > 0)
        {
            return _stack.Peek().ProcessPointerWheel(deltaY);
        }
        return false;
    }
}
