using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Input;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.VisualTree;

namespace AetherVk.Input;

public class GlobalInputRouter
{
  private readonly InputRegistry _registry;
  private readonly Dictionary<IPointer, (Visual Target, IActionHandler Handler, AppAction Action)> _pressedVisuals = new();


  public GlobalInputRouter(TopLevel window, InputRegistry registry)
  {
    _registry = registry;

    window.AddHandler(
      InputElement.KeyDownEvent,
      OnKeyDown,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    window.AddHandler(
      InputElement.KeyUpEvent,
      OnKeyUp,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    window.AddHandler(
      InputElement.PointerPressedEvent,
      OnPointerPressed,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    window.AddHandler(
      InputElement.PointerReleasedEvent,
      OnPointerReleased,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
  }

  private void OnKeyDown(object? sender, KeyEventArgs e) => HandleInput(e, true);

  private void OnKeyUp(object? sender, KeyEventArgs e) => HandleInput(e, false);

  private void HandleInput(KeyEventArgs e, bool isPressed)
  {
    if (e.Handled)
      return;

    var focused =
      TopLevel.GetTopLevel(e.Source as Visual)?.FocusManager?.GetFocusedElement() as Visual;
    if (focused == null)
      return;

    if (
      focused is TextBox
      && e.KeyModifiers is KeyModifiers.None or KeyModifiers.Shift
      && e.Key is not Key.Escape
    )
      return;

    var chord = new InputChord(
      Key: e.Key.ToString(),
      Shift: e.KeyModifiers.HasFlag(KeyModifiers.Shift),
      Ctrl: e.KeyModifiers.HasFlag(KeyModifiers.Control),
      Alt: e.KeyModifiers.HasFlag(KeyModifiers.Alt)
    );

    var (handler, action) = RouteAction(focused, chord, isPressed);
    if (handler != null)
    {
      e.Handled = true;
    }
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e) =>
    HandlePointerInput(e, true);

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e) =>
    HandlePointerInput(e, false);

  private void HandlePointerInput(PointerEventArgs e, bool isPressed)
  {
    if (e.Handled)
      return;

    var visual = e.Source as Visual;
    if (visual == null)
      return;

    var target = visual;

    if (!isPressed && _pressedVisuals.TryGetValue(e.Pointer, out var info))
    {
      info.Handler.ProcessAction(info.Action, false);
      _pressedVisuals.Remove(e.Pointer);
      e.Pointer.Capture(null);
      e.Handled = true;
      return;
    }

    var point = e.GetCurrentPoint(visual);
    if (point.Properties.PointerUpdateKind != PointerUpdateKind.Other)
    {
      string pointerStr = point.Properties.PointerUpdateKind.ToString();
      if (pointerStr.EndsWith("Released"))
      {
        pointerStr = pointerStr.Replace("Released", "Pressed");
      }

      var chord = new InputChord(
        Key: null,
        Shift: e.KeyModifiers.HasFlag(KeyModifiers.Shift),
        Ctrl: e.KeyModifiers.HasFlag(KeyModifiers.Control),
        Alt: e.KeyModifiers.HasFlag(KeyModifiers.Alt),
        Pointer: pointerStr
      );

      var (handler, action) = RouteAction(target, chord, isPressed);
      if (handler != null && action != null)
      {
        e.Handled = true;
        if (isPressed)
        {
          _pressedVisuals[e.Pointer] = (target, handler, action);
          e.Pointer.Capture(target as IInputElement);
          
          if (target is InputElement ie && ie.Focusable)
          {
            ie.Focus();
          }
        }
      }
    }
  }

  private (IActionHandler? Handler, AppAction? Action) RouteAction(Visual focused, InputChord chord, bool isPressed)
  {
    var current = focused;
    while (current != null)
    {
      if (current is Control c)
      {
        var contextId = ActionContext.GetId(c);
        var handler = ActionContext.GetHandler(c);

        if (!string.IsNullOrEmpty(contextId) && handler != null)
        {
          if (_registry.Resolve(contextId, chord) is { } action)
          {
            if (handler.ProcessAction(action, isPressed))
            {
              return (handler, action);
            }
          }
        }
      }
      current = current.GetVisualParent() ?? (current as ILogical)?.LogicalParent as Visual;
    }
    return (null, null);
  }
}
