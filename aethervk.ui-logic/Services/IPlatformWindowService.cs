namespace AetherVk.Logic.Services;

/// <summary>
/// Platform service for OS-level window management operations that cannot be expressed
/// through the cross-platform Avalonia API.
/// Registered as a singleton in DI; the implementation is platform-conditional.
/// </summary>
public interface IPlatformWindowService
{
  /// <summary>
  /// Applies OS-level hints so that the window manager (WM) cannot move the window
  /// through normal WM gestures (e.g. Super+LMB on X11 desktops).
  /// Must be called after the window has been shown / mapped.
  /// <para>
  /// On Linux/X11: sets <c>_NET_WM_WINDOW_TYPE_DOCK</c> and an empty
  /// <c>_NET_WM_ALLOWED_ACTIONS</c>, then lets the caller pulse Hide+Show so the WM re-reads.
  /// On other platforms: no-op.
  /// </para>
  /// </summary>
  /// <param name="windowXid">
  /// The OS window handle from <c>TopLevel.TryGetPlatformHandle().Handle</c>.
  /// On non-Linux platforms this value is ignored.
  /// </param>
  void SetWindowNonMoveable(nint windowXid);
}
