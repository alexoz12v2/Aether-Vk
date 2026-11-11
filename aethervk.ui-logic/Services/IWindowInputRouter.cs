
using System;
using AetherVk.Logic.Input;

namespace AetherVk.Logic.Services;

public interface IWindowInputRouter : IDisposable
{
  /// <summary>
  /// Attaches the router to a specific window root (eg Avalonia TopLevel or Window).
  /// Passed as object to keep logic layer UI-agnostic.
  /// </summary>
  void AttachToWindow(object windowRoot);

  /// <summary>
  /// Detaches the router from a previously attached window root and removes all its
  /// event subscriptions. Safe to call even if the window was never attached (no-op).
  /// </summary>
  void DetachFromWindow(object windowRoot);

  /// <summary>
  /// Routes a pre-composed <see cref="InputChord"/> (from native OS input, not an Avalonia
  /// keyboard event) through the <c>InputRegistry</c> and dispatches it to the matching
  /// <see cref="IActionHandler"/> registered in the visual tree under <paramref name="contextId"/>.
  /// </summary>
  /// <remarks>
  /// This may be called from a background scheduler thread. Implementors must marshal
  /// to the UI thread if the visual tree walk requires it.
  /// </remarks>
  void RouteNativeComposed(string contextId, InputChord chord, InputState state);

  /// <summary>
  /// A button went down in a native viewport (the main one or the mini-view). Until the next click
  /// on an Avalonia control, the keys that reach Avalonia with nothing focused belong to the
  /// viewport: on macOS the native view never gets keyDown: (Avalonia's window keeps the key
  /// events), so they arrive there instead of through <see cref="RouteNativeComposed"/>.
  /// </summary>
  /// <remarks>May be called from the input thread.</remarks>
  void NativeViewportPressed();
}
