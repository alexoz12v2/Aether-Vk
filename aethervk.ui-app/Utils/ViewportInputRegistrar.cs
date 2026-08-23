using AetherVk.Input;
using AetherVk.Logic.Input;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Utils;

/// <summary>
/// Registers the default Blender-style viewport keybindings into an <see cref="InputRegistry"/>.
/// Called once at DI host startup — hardcoded until a preferences system exists.
/// Pointer-based actions (orbit, pan, zoom) are dispatched from code-behind, not from this registry.
/// </summary>
public static class ViewportInputRegistrar
{
  public static void RegisterViewportDefaults(this InputRegistry registry)
  {
    var ctx = InputContext.Viewport.ToCtxString();

    // Numpad . (Decimal) — Reset camera to mode-default position
    registry.Register(ctx,
      new InputChord(Key: "NumPadDecimal"),
      new AppAction(ViewportAction.ResetPosition.ToCmdString()));

    // Numpad 5 — Toggle perspective / orthographic
    registry.Register(ctx,
      new InputChord(Key: "NumPad5"),
      new AppAction(ViewportAction.ToggleProjection.ToCmdString()));

    // V — Cycle camera mode (EarthPosition → UpZenith → CometOrbiting → …)
    registry.Register(ctx,
      new InputChord(Key: "V"),
      new AppAction(ViewportAction.SwitchCameraMode.ToCmdString()));

    // Numpad 1 — Jump directly to EarthPosition mode
    registry.Register(ctx,
      new InputChord(Key: "NumPad1"),
      new AppAction(ViewportAction.SwitchToEarthPosition.ToCmdString()));

    // Numpad 2 — Jump directly to UpZenith mode
    registry.Register(ctx,
      new InputChord(Key: "NumPad2"),
      new AppAction(ViewportAction.SwitchToUpZenith.ToCmdString()));

    // Numpad 3 — Jump directly to CometOrbiting mode
    registry.Register(ctx,
      new InputChord(Key: "NumPad3"),
      new AppAction(ViewportAction.SwitchToCometOrbiting.ToCmdString()));
  }
}
