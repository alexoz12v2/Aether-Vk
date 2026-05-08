using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.VisualTree;
using AetherVk.Logic.Input;
using System.Linq;

namespace AetherVk.Input;

public class GlobalInputRouter
{
    private readonly InputRegistry _registry;

    public GlobalInputRouter(TopLevel window, InputRegistry registry)
    {
        _registry = registry;
        
        window.AddHandler(InputElement.KeyDownEvent, OnKeyDown, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        window.AddHandler(InputElement.KeyUpEvent, OnKeyUp, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        window.AddHandler(InputElement.PointerPressedEvent, OnPointerPressed, Avalonia.Interactivity.RoutingStrategies.Tunnel);
        window.AddHandler(InputElement.PointerReleasedEvent, OnPointerReleased, Avalonia.Interactivity.RoutingStrategies.Tunnel);
    }

    private void OnKeyDown(object? sender, KeyEventArgs e) => HandleInput(e, true);
    private void OnKeyUp(object? sender, KeyEventArgs e) => HandleInput(e, false);

    private void HandleInput(KeyEventArgs e, bool isPressed)
    {
        if (e.Handled) return;

        var focused = TopLevel.GetTopLevel(e.Source as Visual)?.FocusManager?.GetFocusedElement() as Visual;
        if (focused == null) return;

        if (focused is TextBox && e.KeyModifiers is KeyModifiers.None or KeyModifiers.Shift && e.Key is not Key.Escape) return;

        var chord = new InputChord(
            Key: e.Key.ToString(),
            Shift: e.KeyModifiers.HasFlag(KeyModifiers.Shift),
            Ctrl: e.KeyModifiers.HasFlag(KeyModifiers.Control),
            Alt: e.KeyModifiers.HasFlag(KeyModifiers.Alt)
        );

        if (RouteAction(focused, chord, isPressed))
        {
            e.Handled = true;
        }
    }

    private void OnPointerPressed(object? sender, PointerPressedEventArgs e) => HandlePointerInput(e, true);
    private void OnPointerReleased(object? sender, PointerReleasedEventArgs e) => HandlePointerInput(e, false);

    private void HandlePointerInput(PointerEventArgs e, bool isPressed)
    {
        if (e.Handled) return;

        var visual = e.Source as Visual;
        if (visual == null) return;
        
        // For pointer events, the target should be the element under the pointer
        // This ensures MiddleClick directly over the viewport resolves to the viewport's ActionContext, 
        // even if a TextBox somewhere else currently has keyboard focus.
        var target = visual;

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

            if (RouteAction(target, chord, isPressed))
            {
                e.Handled = true;
                // Auto-focus the visual if it successfully handled a pointer action
                if (isPressed && target is InputElement ie && ie.Focusable)
                {
                    ie.Focus();
                }
            }
        }
    }

    private bool RouteAction(Visual focused, InputChord chord, bool isPressed)
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
                            return true;
                        }
                    }
                }
            }
            current = current.GetVisualParent() ?? (current as ILogical)?.LogicalParent as Visual;
        }
        return false;
    }
}
