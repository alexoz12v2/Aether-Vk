using System;
using System.Numerics;
using AetherVk.Logic.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Possible actions to be performed when looking at a <see cref="Viewport3DViewModel"/> from the
/// <see cref="ViewportBaseOperator" />
///
/// All members named "Start*" push a new Operator on the base
/// </summary>
public enum ViewportAction
{
  /// <summary>Toggle between the 3 camera modes in Viewport view model</summary>
  SwitchCameraMode,
  /// <summary>Reset the camera towards the standard position, as defined by the current mode in the
  /// view model</summary>
  ResetPosition,
  /// <summary>Toggle between perspective and orthographic projection</summary>
  ToggleProjection,
  /// <summary>Rotate the camera, relative to its centre</summary>
  StartRotate,
  /// <summary>Translate the camera on the normal plane relative to its forward axis (-y)</summary>
  StartPan,
  /// <summary>Translate camera along its forward axis</summary>
  StartZoom,
  /// <summary>Rotate and translate camera such that it orbits around a point in space. Its
  /// resolution uses an IObservable of Vector3 position so that if orbit center rotates camera can
  /// can rotate and translate accordingly</summary>
  StartOrbit,
  /// <summary>Translate camera such that the relative distance between a point and camera's
  /// position is the same. both camera and object are expressed in world space</summary>
  StartTracking,

  CreateBillboard,
  StartBillboardManip,
  DeleteBillboard,

  /// <summary>Jump directly to the EarthPosition camera mode (Numpad 1).</summary>
  SwitchToEarthPosition,
  /// <summary>Jump directly to the UpZenith camera mode (Numpad 2).</summary>
  SwitchToUpZenith,
  /// <summary>Jump directly to the CometOrbiting camera mode (Numpad 3).</summary>
  SwitchToCometOrbiting,
}

public static class ViewportActionExtensions
{
  // C# 14 instance extensions
  extension(ViewportAction action)
  {
    public string ToCmdString() => action switch
    {
      ViewportAction.SwitchCameraMode       => "viewport.switch_camera_mode",
      ViewportAction.ResetPosition          => "viewport.reset_position",
      ViewportAction.ToggleProjection       => "viewport.toggle_projection",
      ViewportAction.StartRotate            => "viewport.start_rotate",
      ViewportAction.StartPan               => "viewport.start_pan",
      ViewportAction.StartZoom              => "viewport.start_zoom",
      ViewportAction.StartOrbit             => "viewport.start_orbit",
      ViewportAction.StartTracking          => "viewport.start_tracking",
      ViewportAction.CreateBillboard        => "viewport.create_billboard",
      ViewportAction.StartBillboardManip    => "viewport.start_billboard_manip",
      ViewportAction.DeleteBillboard        => "viewport.delete_billboard",
      ViewportAction.SwitchToEarthPosition  => "viewport.switch_to_earth_position",
      ViewportAction.SwitchToUpZenith       => "viewport.switch_to_up_zenith",
      ViewportAction.SwitchToCometOrbiting  => "viewport.switch_to_comet_orbiting",
      _                                     => "nothing"
    };
  }

  // C# 14 static extensions
  extension(ViewportAction)
  {
    public static ViewportAction FromCmdString(string value) => value switch
    {
      "viewport.switch_camera_mode"       => ViewportAction.SwitchCameraMode,
      "viewport.reset_position"           => ViewportAction.ResetPosition,
      "viewport.toggle_projection"        => ViewportAction.ToggleProjection,
      "viewport.start_rotate"             => ViewportAction.StartRotate,
      "viewport.start_pan"                => ViewportAction.StartPan,
      "viewport.start_zoom"               => ViewportAction.StartZoom,
      "viewport.start_orbit"              => ViewportAction.StartOrbit,
      "viewport.start_tracking"           => ViewportAction.StartTracking,
      "viewport.switch_to_earth_position" => ViewportAction.SwitchToEarthPosition,
      "viewport.switch_to_up_zenith"      => ViewportAction.SwitchToUpZenith,
      "viewport.switch_to_comet_orbiting" => ViewportAction.SwitchToCometOrbiting,
      _ => throw new FormatException($"invalid ViewportAction: {value}")
    };

    public static bool TryFromCmdString(string value, out ViewportAction result)
    {
      try
      {
        result = ViewportAction.FromCmdString(value);
        return true;
      }
      catch
      {
        result = default;
        return false;
      }
    }
  }
}

// internal and without DI, because the VM instantiates it directly
internal class ViewportBaseOperator(Viewport3DViewModel vm) : IActionOperator
{
  private readonly Viewport3DViewModel _vm = vm;

  public void OnEnter() { }

  public void OnExit() { }

  public bool ProcessAction(AppAction action, InputState state)
  {
    // Unrecognised IDs (e.g. "viewport.pointer_delta") are consumed by transient operators above us.
    if (!ViewportAction.TryFromCmdString(action.Id, out var act))
      return false;

    switch (act)
    {
      case ViewportAction.SwitchCameraMode when state.IsPressed:
        _vm.CameraService.CycleCameraMode();
        return true;

      case ViewportAction.ResetPosition when state.IsPressed:
        _vm.CameraService.ResetToModeDefault();
        return true;

      case ViewportAction.ToggleProjection when state.IsPressed:
        _vm.CameraService.ToggleProjection();
        return true;

      case ViewportAction.StartOrbit when state.IsPressed:
        if (action.Payload is Vector2 startOrbitPos && _vm.CameraService.IsOrbitAllowed())
          _vm.OperatorStack.Push(new OrbitCameraOperator(_vm, startOrbitPos));
        return true;

      case ViewportAction.StartPan when state.IsPressed:
        // Pan is always allowed — no gate check needed.
        if (action.Payload is Vector2 startPanPos)
          _vm.OperatorStack.Push(new PanCameraOperator(_vm, startPanPos));
        return true;

      case ViewportAction.StartZoom when state.IsPressed:
        if (action.Payload is Vector2 startZoomPos && _vm.CameraService.IsZoomAllowed())
          _vm.OperatorStack.Push(new ZoomCameraOperator(_vm, startZoomPos));
        return true;

      case ViewportAction.SwitchToEarthPosition when state.IsPressed:
        _vm.CameraService.SetCameraMode(Services.CameraMode.EarthPosition);
        return true;

      case ViewportAction.SwitchToUpZenith when state.IsPressed:
        _vm.CameraService.SetCameraMode(Services.CameraMode.UpZenith);
        return true;

      case ViewportAction.SwitchToCometOrbiting when state.IsPressed:
        _vm.CameraService.SetCameraMode(Services.CameraMode.CometOrbiting);
        return true;
    }
    return false;
  }
}
