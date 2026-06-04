using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

public class ViewportBaseOperator : IActionOperator
{
  private readonly Viewport3DViewModel _vm;

  public ViewportBaseOperator(Viewport3DViewModel vm)
  {
    _vm = vm;
  }

  public void OnEnter() { }

  public void OnExit() { }

  public bool ProcessAction(AppAction action, bool isPressed)
  {
    // Handle radial menu release ("hold to show, release to select")
    if (!isPressed)
    {
      if (action.Id == "viewport.open_radial_menu" && _vm.IsRadialMenuOpen)
      {
        _vm.CommitRadialMenuSelection();
        return true;
      }
      return false;
    }

    switch (action.Id)
    {
      case "viewport.cancel_add_jet":
        if (_vm.IsAddingJet)
        {
          _vm.IsAddingJet = false;
          return true;
        }
        if (_vm.IsRadialMenuOpen)
        {
          _vm.CloseRadialMenu();
          return true;
        }
        return false;
      case "viewport.toggle_measuring":
        _vm.ToggleEarthObserverModeCommand.Execute(null);
        return true;
      case "viewport.reset_camera":
      case "viewport.snap_to_selected":
      {
        var selected = _vm.SelectedEntity;
        if (selected != null)
          _vm.RuntimeService.SnapToEntity(_vm.SceneId, _vm.CameraId, selected.Id);
        else
          _vm.SnapCameraToSun();
        return true;
      }
      case "viewport.move_cursor_up":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, 0.0f, -0.5f, 0.0f);
        return true;
      case "viewport.move_cursor_down":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, 0.0f, 0.5f, 0.0f);
        return true;
      case "viewport.move_cursor_left":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, -0.5f, 0.0f, 0.0f);
        return true;
      case "viewport.move_cursor_right":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, 0.5f, 0.0f, 0.0f);
        return true;
      case "viewport.move_cursor_z_up":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, 0.0f, 0.0f, 0.5f);
        return true;
      case "viewport.move_cursor_z_down":
        _vm.RuntimeService.MoveCursor(_vm.SceneId, 0.0f, 0.0f, -0.5f);
        return true;
      case "viewport.start_orbit":
        if (!_vm.IsEarthObserverMode)
          _vm.OperatorStack.Push(new OrbitOperator(_vm));
        return true;
      case "viewport.start_pan":
        if (!_vm.IsEarthObserverMode)
          _vm.OperatorStack.Push(new PanOperator(_vm));
        return true;
      case "viewport.start_zoom_drag":
        _vm.OperatorStack.Push(new ZoomDragOperator(_vm));
        return true;
      case "viewport.open_radial_menu":
        if (!_vm.IsRadialMenuOpen)
          _vm.OpenRadialMenuAt(_vm.RadialMenuX, _vm.RadialMenuY);
        return true;
    }

    return false;
  }

  public bool ProcessPointerWheel(float deltaY)
  {
    if (_vm.IsEarthObserverMode)
      return true;
    _vm.RuntimeService.ZoomCamera(_vm.SceneId, _vm.CameraId, deltaY);
    return true;
  }

  public bool ProcessPointerDelta(float dx, float dy) => false;
}

public class OrbitOperator : IActionOperator
{
  private readonly Viewport3DViewModel _vm;

  public OrbitOperator(Viewport3DViewModel vm) => _vm = vm;

  public void OnEnter() { _vm.IsOrbiting = true; }

  public void OnExit() { _vm.IsOrbiting = false; }

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

public class PanOperator : IActionOperator
{
  private readonly Viewport3DViewModel _vm;

  public PanOperator(Viewport3DViewModel vm) => _vm = vm;

  public void OnEnter() { _vm.IsPanning = true; }

  public void OnExit() { _vm.IsPanning = false; }

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
    _vm.RuntimeService.PanCamera(_vm.SceneId, _vm.CameraId, dx, dy);
    return true;
  }

  public bool ProcessPointerWheel(float deltaY) => false;
}

public class ZoomDragOperator : IActionOperator
{
  private readonly Viewport3DViewModel _vm;

  public ZoomDragOperator(Viewport3DViewModel vm) => _vm = vm;

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
    _vm.RuntimeService.ZoomCamera(_vm.SceneId, _vm.CameraId, dy * 0.1f);
    return true;
  }

  public bool ProcessPointerWheel(float deltaY) => false;
}
