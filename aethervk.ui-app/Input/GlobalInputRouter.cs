using System;
using System.Collections.Generic;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.LogicalTree;
using Avalonia.VisualTree;

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
///
/// Multi-window support: the router tracks a set of attached TopLevels (e.g. MainWindow + one
/// OverlayWindow per viewport). AttachToWindow / DetachFromWindow manage membership. Dispose
/// removes all at once when the application exits.
/// </summary>
public sealed class GlobalInputRouter(InputRegistry registry) : IWindowInputRouter
{
  private readonly InputRegistry _registry = registry;
  private readonly Dictionary<
    IPointer,
    (Visual Target, IActionHandler Handler, AppAction Action)
  > _pressedVisuals = [];

  // All attached TopLevels: main window + one overlay per active viewport.
  private readonly HashSet<TopLevel> _attachedWindows = [];

  public void AttachToWindow(object windowRoot)
  {
    if (windowRoot is not TopLevel window)
      throw new ArgumentException("windowRoot must be an Avalonia TopLevel", nameof(windowRoot));

    if (!_attachedWindows.Add(window))
      return; // idempotent — prevents double-attach

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

    Log($"[Router] Attached to window: {window.GetType().Name}");
  }

  public void DetachFromWindow(object windowRoot)
  {
    if (windowRoot is not TopLevel window)
      return;
    if (!_attachedWindows.Remove(window))
      return; // wasn't attached — no-op

    window.RemoveHandler(InputElement.KeyDownEvent, OnKeyDown);
    window.RemoveHandler(InputElement.KeyUpEvent, OnKeyUp);
    window.RemoveHandler(InputElement.PointerPressedEvent, OnPointerPressed);
    window.RemoveHandler(InputElement.PointerReleasedEvent, OnPointerReleased);

    // Discard any in-flight press state that was captured on this window's controls.
    _pressedVisuals.Clear();

    Log($"[Router] Detached from window: {window.GetType().Name}");
  }

  public void Dispose()
  {
    foreach (var window in _attachedWindows)
    {
      window.RemoveHandler(InputElement.KeyDownEvent, OnKeyDown);
      window.RemoveHandler(InputElement.KeyUpEvent, OnKeyUp);
      window.RemoveHandler(InputElement.PointerPressedEvent, OnPointerPressed);
      window.RemoveHandler(InputElement.PointerReleasedEvent, OnPointerReleased);
    }
    _attachedWindows.Clear();
    _pressedVisuals.Clear();
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

    // Let TextBoxes handle normal typing; only intercept Escape or modifier+key combos.
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

    Log($"[Router] Key {(isPressed ? "↓" : "↑")} chord={chord.DisplayText} focused={focused.GetType().Name}");

    var (handler, action) = RouteAction(focused, chord, state);
    if (handler != null && action != null)
    {
      Log($"[Router] → dispatching action={action.Value.Id} to {handler.GetType().Name}");
      handler.Process(action.Value, state);
      e.Handled = true;
    }
    else
    {
      Log($"[Router] → no binding found for chord={chord.DisplayText}");
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

    // ── Transparent Overlay Bypass ──────────────────────────────────────────────
    // If the pointer landed on the overlay's transparent void (the Window root itself
    // or the sentinel "TransparentRoot" Panel), this is not a UI action.
    // Returning without setting e.Handled lets the OS deliver the event natively to
    // the NativeControlHost XID below. The native input hooks pick it up and call
    // RouteNativeComposed, re-entering this router for viewport actions.
    if (visual is Window || (visual is Control ctrl && ctrl.Name == "TransparentRoot"))
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

      Log($"[Router] Pointer {(isPressed ? "↓" : "↑")} chord={chord.DisplayText} visual={visual.GetType().Name}");

      var (handler, action) = RouteAction(target, chord, state);
      if (handler != null && action != null)
      {
        Log($"[Router] → dispatching action={action.Value.Id} to {handler.GetType().Name}");
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
      else
      {
        Log($"[Router] → no binding found for chord={chord.DisplayText}");
      }
    }
  }

  private (IActionHandler? Handler, AppAction? Action) RouteAction(
    Visual focused,
    InputChord chord,
    InputState state
  )
  {
    Log($"[Router] RouteAction: start walk from {focused.GetType().Name} for chord={chord.DisplayText}");
    var current = focused;
    while (current != null)
    {
      if (current is Control c)
      {
        var contextId = ActionContext.GetId(c);
        var handler = ActionContext.GetHandler(c);

        string cName = c.Name ?? "(null)";
        string ctx = contextId ?? "(none)";
        string hName = handler != null ? handler.GetType().Name : "null";
        Log($"[Router]   walk node={c.GetType().Name} name={cName} contextId={ctx} handler={hName}");

        if (!string.IsNullOrEmpty(contextId) && handler != null)
        {
          var action = _registry.Resolve(contextId, chord);
          Log($"[Router]   resolved context={contextId} ({c.GetType().Name}) → {(action.HasValue ? $"action={action.Value.Id}" : "no binding")}");
          if (action is { } resolved)
          {
            return (handler, resolved);
          }
        }
      }
      current = current.GetVisualParent() ?? (current as ILogical)?.LogicalParent as Visual;
    }
    Log($"[Router] RouteAction: end walk, no binding found.");
    return (null, null);
  }

  public void RouteNativeComposed(string contextId, InputChord chord, InputState state)
  {
    if (_attachedWindows.Count == 0)
    {
      Log($"[Router] RouteNativeComposed called but no attached windows.");
      return;
    }
    // FindAndDispatch walks the Avalonia visual tree (GetVisualChildren / GetVisualParent).
    // Visual tree access is NOT thread-safe and must happen on the UI thread.
    // The Rx Buffer timer fires this callback on a TP worker — dispatch back to UI thread.
    Avalonia.Threading.Dispatcher.UIThread.Post(() =>
    {
      Log($"[Router] RouteNativeComposed contextId={contextId} chord={chord.DisplayText} state.IsPressed={state.IsPressed}");
      // Search across all attached windows (main + all overlay windows).
      // Stop as soon as the contextId is found in one window's visual tree.
      bool found = false;
      foreach (var window in _attachedWindows)
      {
        Log($"[Router]   searching in window: {window.GetType().Name}");
        if (FindAndDispatch(window, contextId, chord, state))
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        Log($"[Router]   RouteNativeComposed failed to find contextId={contextId} in any window.");
      }
    });
  }

  /// <summary>
  /// Depth-first walk of the visual tree to find the first <see cref="Control"/> tagged
  /// with <paramref name="contextId"/> via <see cref="ActionContext"/> and dispatch to it.
  /// Returns <c>true</c> on the first match so the caller can stop searching other windows.
  /// </summary>
  private bool FindAndDispatch(Visual root, string contextId, InputChord chord, InputState state)
  {
    if (root is Control c)
    {
      var id = ActionContext.GetId(c);
      var handler = ActionContext.GetHandler(c);
      Log($"[Router]   walk node={c.GetType().Name} name={c.Name ?? "(null)"} id={id ?? "(none)"} handler={(handler != null ? handler.GetType().Name : "null")}");
      if (id == contextId && handler != null)
      {
        if (_registry.Resolve(contextId, chord) is { } action)
        {
          Log($"[Router] FindAndDispatch context={contextId} → action={action.Id}");
          handler.Process(action, state);
        }
        else
        {
          Log($"[Router] FindAndDispatch context={contextId} chord={chord.DisplayText} → no binding in registry");
        }
        return true; // found — stop walking
      }
    }
    foreach (var child in root.GetVisualChildren())
      if (FindAndDispatch(child, contextId, chord, state))
        return true;
    return false;
  }

  private InputModifiers GetModifiers(KeyModifiers avaloniaModifiers)
  {
    InputModifiers mods = InputModifiers.None;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Shift))
      mods |= InputModifiers.Shift;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Control))
      mods |= InputModifiers.Ctrl;
    if (avaloniaModifiers.HasFlag(KeyModifiers.Alt))
      mods |= InputModifiers.Alt;
    return mods;
  }

  /// <summary>
  /// Structured logging for the input routing pipeline.
  /// In DEBUG builds: writes to both Console and Debug output.
  /// In Release builds: compiled out completely — zero overhead.
  /// </summary>
  [System.Diagnostics.Conditional("DEBUG")]
  private static void Log(string message)
  {
    var line = $"[{DateTime.Now:HH:mm:ss.fff}] {message}";
    Console.WriteLine(line);
    System.Diagnostics.Debug.WriteLine(line);
  }
}
