using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

public class MeshViewerBaseOperator : IActionOperator
{
    private readonly MeshViewerViewModel _vm;
    public MeshViewerBaseOperator(MeshViewerViewModel vm) => _vm = vm;
    public void OnEnter() { }
    public void OnExit() { }
    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (!isPressed) return false;
        switch (action.Id)
        {
            case "viewport.start_orbit":
                _vm.OperatorStack.Push(new MeshViewerOrbitOperator(_vm));
                return true;
            case "viewport.start_pan":
                _vm.OperatorStack.Push(new MeshViewerPanOperator(_vm));
                return true;
            case "viewport.start_zoom_drag":
                _vm.OperatorStack.Push(new MeshViewerZoomDragOperator(_vm));
                return true;
            case "viewport.reset_camera":
                _vm.RuntimeService.ResetCamera(_vm.SceneId, _vm.CameraId);
                return true;
        }
        return false;
    }
    public bool ProcessPointerWheel(float deltaY)
    {
        _vm.RuntimeService.ZoomCamera(_vm.SceneId, _vm.CameraId, deltaY);
        return true;
    }
    public bool ProcessPointerDelta(float dx, float dy) => false;
}

public class MeshViewerOrbitOperator : IActionOperator
{
    private readonly MeshViewerViewModel _vm;
    public MeshViewerOrbitOperator(MeshViewerViewModel vm) => _vm = vm;
    public void OnEnter() { }
    public void OnExit() { }
    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (action.Id == "viewport.start_orbit" && !isPressed)
        {
            _vm.OperatorStack.Pop();
            return true;
        }
        return true; 
    }
    public bool ProcessPointerDelta(float dx, float dy)
    {
        _vm.RuntimeService.RotateCamera(_vm.SceneId, _vm.CameraId, dx, dy);
        return true;
    }
    public bool ProcessPointerWheel(float deltaY) => false;
}

public class MeshViewerPanOperator : IActionOperator
{
    private readonly MeshViewerViewModel _vm;
    public MeshViewerPanOperator(MeshViewerViewModel vm) => _vm = vm;
    public void OnEnter() { }
    public void OnExit() { }
    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (action.Id == "viewport.start_pan" && !isPressed)
        {
            _vm.OperatorStack.Pop();
            return true;
        }
        return true; 
    }
    public bool ProcessPointerDelta(float dx, float dy)
    {
        _vm.RuntimeService.PanCursor(_vm.SceneId, dx, dy);
        return true;
    }
    public bool ProcessPointerWheel(float deltaY) => false;
}

public class MeshViewerZoomDragOperator : IActionOperator
{
    private readonly MeshViewerViewModel _vm;
    public MeshViewerZoomDragOperator(MeshViewerViewModel vm) => _vm = vm;
    public void OnEnter() { }
    public void OnExit() { }
    public bool ProcessAction(AppAction action, bool isPressed)
    {
        if (action.Id == "viewport.start_zoom_drag" && !isPressed)
        {
            _vm.OperatorStack.Pop();
            return true;
        }
        return true; 
    }
    public bool ProcessPointerDelta(float dx, float dy)
    {
        _vm.RuntimeService.ZoomCamera(_vm.SceneId, _vm.CameraId, dy * 2.0f);
        return true;
    }
    public bool ProcessPointerWheel(float deltaY) => false;
}
