using System;
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
  DeleteBillboard
}

public static class ViewportActionExtensions
{
  // C# 14 intstance extensions
  extension(ViewportAction action)
  {
    public string ToCmdString() => action switch
    {
      ViewportAction.SwitchCameraMode => "viewport.switch_camera_mode",
      ViewportAction.ResetPosition => "viewport.reset_position",
      ViewportAction.StartRotate => "viewport.start_rotate",
      ViewportAction.StartPan => "viewport.start_pan",
      ViewportAction.StartZoom => "viewport.start_zoom",
      ViewportAction.StartOrbit => "viewport.start_orbit",
      ViewportAction.StartTracking => "viewport.start_tracking",
      ViewportAction.CreateBillboard => "viewport.create_billboard",
      ViewportAction.StartBillboardManip => "viewport.start_billboard_manip",
      ViewportAction.DeleteBillboard => "viewport.delete_billboard",
      _ => "nothing"
    };
  }

  // C# 14 static extensions
  extension(ViewportAction)
  {
    public static ViewportAction FromCmdString(string value) => value switch
    {
      "viewport.switch_camera_mode" => ViewportAction.SwitchCameraMode,
      "viewport.reset_position" => ViewportAction.ResetPosition,
      "viewport.start_rotate" => ViewportAction.StartRotate,
      "viewport.start_pan" => ViewportAction.StartPan,
      "viewport.start_zoom" => ViewportAction.StartZoom,
      "viewport.start_orbit" => ViewportAction.StartOrbit,
      "viewport.start_tracking" => ViewportAction.StartTracking,
      _ => throw new FormatException($"invalid status: {value}")

    };
  }
}

// internal and without DI, cause the vm will instantiate it and that's it
internal class ViewportBaseOperator(Viewport3DViewModel vm) : IActionOperator
{
  private readonly Viewport3DViewModel _vm = vm;

  public void OnEnter() { }

  public void OnExit() { }

  public bool ProcessAction(AppAction action, InputState state)
  {
    throw new NotImplementedException();
  }
}

