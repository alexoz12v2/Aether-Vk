using System.Collections.Generic;
using System;
using AetherVk.Logic.Input;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.VisualTree;
using AetherVk.Logic.Services;

namespace AetherVk.Input;

/// <summary>
/// The GlobalInputRouter doesn't keep a list of registered View models in memory. Instead, it uses
/// *Avalonia Attached Properties* to store the registration directoy inside the XAML UI Tree.
/// In fact, our <see cref="ActionContext" /> defines properties such as `ActionContext.Handler`
/// <pre>
///   <Border input:ActionContext.Id="MyViewportContext" input:ActionContext.Handler="{Binding}">
///     <!-- All children of this Border will trigger this handler -->
///   </Border>
/// </pre>
///
/// It routes with visual coordinates as it listens to the absolute Root of the window, meaning
/// <see cref="TopLevel" />, using "Tunneling" events, meaning that every single click is processed
/// by routers *Before* textboxes or buttons.
///
/// Example: when a user clicks the mouse:
/// - Avalonia communicates mouse was clicked for a specific Element, eg TextBlock (e.Source as Visual)
/// - Router takes the TextBlock and walks up the visual tree
/// - Whet it finds a parent with view model attached, asks the `InputRegistry` to convert the raw
///   click into a <see cref="AppAction" />
/// - inovke `ProcessAction` method
///
/// Important Note: Should Code-Behind dispatch an AppAction or should it do the GlobalInputRouter?
/// - Code-Behind: Catch raw mouse/pointers event, package them into an AppAction with a coordinate
///     payload, and push them directly to its *own* view model's Operator stack
/// - Global Router: Catch raw keyboard events globally, translate them into absract AppAction
///   strings using user's custom keybindings, and route them to whichever view model is currently
///   in the UI Tree
/// </summary>
public class GlobalInputRouter(InputRegistry registry) : IWindowInputRouter
{
  private readonly InputRegistry _registry = registry;
  private readonly Dictionary<
    IPointer,
    (Visual Target, IActionHandler Handler, AppAction Action)
  > _pressedVisuals = [];
  private TopLevel? _attachedWindow;

  public void AttachToWindow(object windowRoot)
  {
    if (windowRoot is not TopLevel window)
      throw new ArgumentException("windowRoot must be an Avalonia TopLevel", nameof(windowRoot));
    if (_attachedWindow != null)
      throw new InvalidOperationException("Router is already attached to a window");

    _attachedWindow = window;

    _attachedWindow.AddHandler(
      InputElement.KeyDownEvent,
      OnKeyDown,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    _attachedWindow.AddHandler(
      InputElement.KeyUpEvent,
      OnKeyUp,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    _attachedWindow.AddHandler(
      InputElement.PointerPressedEvent,
      OnPointerPressed,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
    _attachedWindow.AddHandler(
      InputElement.PointerReleasedEvent,
      OnPointerReleased,
      Avalonia.Interactivity.RoutingStrategies.Tunnel
    );
  }

  public void Dispose()
  {
    if (_attachedWindow != null)
    {
      _attachedWindow.RemoveHandler(InputElement.KeyDownEvent, OnKeyDown);
      _attachedWindow.RemoveHandler(InputElement.KeyUpEvent, OnKeyUp);
      _attachedWindow.RemoveHandler(InputElement.PointerPressedEvent, OnPointerPressed);
      _attachedWindow.RemoveHandler(InputElement.PointerReleasedEvent, OnPointerReleased);
    }
    _pressedVisuals.Clear();
    _attachedWindow = null;
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
    var state = new InputState(isPressed, GetModifiers(e.KeyModifiers));

    var (handler, action) = RouteAction(focused, chord, state);
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
    var state = new InputState(isPressed, GetModifiers(e.KeyModifiers));

    if (!isPressed && _pressedVisuals.TryGetValue(e.Pointer, out var info))
    {
      info.Handler.Process(info.Action, state);
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

      var (handler, action) = RouteAction(target, chord, state);
      if (handler != null && action != null)
      {
        e.Handled = true;
        if (isPressed)
        {
          _pressedVisuals[e.Pointer] = (target, handler, action.Value);
          e.Pointer.Capture(target as IInputElement);

          if (target is InputElement ie && ie.Focusable)
          {
            ie.Focus();
          }
        }
      }
    }
  }

  private (IActionHandler? Handler, AppAction? Action) RouteAction(Visual focused, InputChord chord, InputState state)
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
            if (handler.Process(action, state))
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

  public void RouteNativeComposed(string contextId, InputChord chord, InputState state)
  {
    if (_attachedWindow == null) return;
    FindAndDispatch(_attachedWindow, contextId, chord, state);
  }

  /// <summary>
  /// Depth-first walk of the visual tree to find the first <see cref="Control"/> tagged
  /// with <paramref name="contextId"/> via <see cref="ActionContext"/> and dispatch to it.
  /// Stops at the first match to avoid duplicate dispatches.
  /// </summary>
  private void FindAndDispatch(Visual root, string contextId, InputChord chord, InputState state)
  {
    if (root is Control c)
    {
      var id = ActionContext.GetId(c);
      var handler = ActionContext.GetHandler(c);
      if (id == contextId && handler != null)
      {
        if (_registry.Resolve(contextId, chord) is { } action)
          handler.Process(action, state);
        return; // found — stop walking
      }
    }
    foreach (var child in root.GetVisualChildren())
      FindAndDispatch(child, contextId, chord, state);
  }

  private InputModifiers GetModifiers(KeyModifiers avaloniaModifiers)
  {
    InputModifiers mods = InputModifiers.None;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Shift)) mods |= InputModifiers.Shift;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Control)) mods |= InputModifiers.Ctrl;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Alt)) mods |= InputModifiers.Alt;
    return mods;
  }
}
