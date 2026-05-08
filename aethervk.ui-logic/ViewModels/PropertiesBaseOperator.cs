using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

public class PropertiesBaseOperator : IActionOperator
{
    private readonly PropertiesViewModel _vm;

    public PropertiesBaseOperator(PropertiesViewModel vm)
    {
        _vm = vm;
    }

    public void OnEnter() { }
    public void OnExit() { }

    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (!isPressed) return false;

        if (action.Id == "ui.expand_all") 
        { 
            _vm.AreAllExpanded = !_vm.AreAllExpanded; 
            return true; 
        }
        if (action.Id == "ui.show_flyout") 
        { 
            _vm.OperatorStack.Push(new FlyoutMenuOperator(_vm));
            return true; 
        }
        return false;
    }

    public bool ProcessPointerDelta(float dx, float dy) => false;
    public bool ProcessPointerWheel(float deltaY) => false;
}

public class FlyoutMenuOperator : IActionOperator
{
    private readonly PropertiesViewModel _vm;

    public FlyoutMenuOperator(PropertiesViewModel vm)
    {
        _vm = vm;
    }

    public void OnEnter() => _vm.IsFlyoutMenuOpen = true;
    public void OnExit() => _vm.IsFlyoutMenuOpen = false;

    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (!isPressed) return false;
        
        if (action.Id == "ui.add_cube") 
        { 
            // Mock logic for adding a cube
            _vm.OperatorStack.Pop(); 
            return true; 
        }
        if (action.Id == "global.cancel" || action.Id == "ui.show_flyout") 
        { 
            _vm.OperatorStack.Pop(); 
            return true; 
        }

        return true; 
    }

    public bool ProcessPointerDelta(float dx, float dy) => false;
    public bool ProcessPointerWheel(float deltaY) => false;
}
