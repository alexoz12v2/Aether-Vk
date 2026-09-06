namespace AetherVk.Logic.Services;

/// <summary>
/// Platform service for OS-level window management operations that cannot be expressed
/// through the cross-platform Avalonia API.
/// Registered as a singleton in DI; the implementation is platform-conditional.
/// </summary>
public interface IPlatformWindowService
{
  /// <summary>
  /// Applies OS-level hints so that the window manager treats this window as a
  /// non-moveable companion panel (no title bar, no taskbar/dock entry, not user-moveable
  /// via WM gestures such as Super+LMB on X11).
  /// Must be called after the window has been shown/mapped.
  /// <para>
  /// On Linux/X11: sets <c>_NET_WM_WINDOW_TYPE_UTILITY</c> and an empty
  /// <c>_NET_WM_ALLOWED_ACTIONS</c>, then lets the caller pulse Hide+Show so the WM re-reads.
  /// On Windows/macOS: no-op — <c>ShowInTaskbar=False</c> + <c>SystemDecorations=None</c>
  /// in AXAML and <c>Show(ownerWindow)</c> ownership handle everything at the Avalonia level.
  /// </para>
  /// </summary>
  /// <param name="windowHandle">
  /// The OS window handle from <c>TopLevel.TryGetPlatformHandle().Handle</c>.
  /// On non-Linux platforms this value is ignored.
  /// </param>
  void SetWindowAsCompanionPanel(nint windowHandle);

  /// <summary>
  /// Returns the root/desktop window handle needed as the target for
  /// <see cref="SetOverlayAbove"/> on X11.
  /// <para>
  /// On Linux/X11: opens a temporary display connection, calls <c>XDefaultRootWindow</c>,
  /// closes the connection, and returns the XID.
  /// On Windows/macOS: returns <c>0</c> (ignored by the no-op <see cref="SetOverlayAbove"/>).
  /// </para>
  /// </summary>
  nint GetRootWindowHandle();

  /// <summary>
  /// Raises or lowers the overlay so it sits above (or no longer forces itself above)
  /// the owner window. Call with <paramref name="raise"/> = <c>true</c> when the owner
  /// gains focus and <c>false</c> when it loses focus.
  /// <para>
  /// On Linux/X11: sends a <c>_NET_WM_STATE</c> ClientMessage (add/remove
  /// <c>_NET_WM_STATE_ABOVE</c>) to the root window so the WM re-stacks the overlay.
  /// On Windows/macOS: no-op — Win32 owned-window Z-order / macOS child-window ordering
  /// handle this automatically without any explicit action on focus change.
  /// </para>
  /// </summary>
  /// <param name="overlayHandle">
  /// The OS handle of the overlay window (<c>TopLevel.TryGetPlatformHandle().Handle</c>).
  /// </param>
  /// <param name="rootHandle">
  /// The X11 root window handle (<c>XDefaultRootWindow</c>). Ignored on non-Linux platforms.
  /// </param>
  /// <param name="raise">
  /// <c>true</c> to add <c>_NET_WM_STATE_ABOVE</c>; <c>false</c> to remove it.
  /// </param>
  void SetOverlayAbove(nint overlayHandle, nint rootHandle, bool raise);

  /// <summary>
  /// Sets <c>override_redirect = True</c> on the overlay XID via <c>XChangeWindowAttributes</c>.
  /// This instructs Xwayland to export the overlay as an independent Wayland surface instead
  /// of blending it into the MainWindow's backing store, enabling correct GPU alpha-compositing
  /// of the transparent overlay over the Vulkan sub-surface.
  /// <para>
  /// Must be called after <c>Show()</c> allocates the XID and before the window is remapped.
  /// The caller is responsible for pulsing <c>Hide() → Show()</c> so Xwayland re-exports
  /// the window with override_redirect active.
  /// </para>
  /// <para>
  /// On Linux/X11: calls <c>XChangeWindowAttributes</c> with <c>CWOverrideRedirect = 0x200</c>.
  /// On Windows/macOS: no-op — <c>Show(ownerWindow)</c> already guarantees correct Z-order.
  /// </para>
  /// </summary>
  /// <param name="overlayHandle">
  /// The XID of the overlay window from <c>TopLevel.TryGetPlatformHandle().Handle</c>.
  /// </param>
  void SetOverlayOverrideRedirect(nint overlayHandle);

  /// <summary>
  /// Sets the global cursor position on the screen.
  /// </summary>
  void SetCursorPosition(int x, int y);
}
