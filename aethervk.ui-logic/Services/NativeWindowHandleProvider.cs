using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

/// <summary>
/// Platform-specific factories for <see cref="CNativeWindowHandle"/>,
/// used when calling <see cref="INativeRuntimeService.AddViewport"/> in windowed mode.
///
/// <para>
/// The handle type constants (<see cref="HandleType"/>) mirror Rust's
/// <c>gpu::NativeHandleType</c> enum in <c>gpu.rs</c> and must be kept in sync with it.
/// </para>
/// </summary>
public static class NativeWindowHandleProvider
{
  /// <summary>
  /// Linux X11 / XWayland — Avalonia 11 does not support native Wayland;
  /// it runs under XWayland when on a Wayland compositor.
  /// <para>
  /// Obtain <paramref name="display"/> and <paramref name="xid"/> from
  /// Avalonia's <c>IPlatformHandle</c> on the native control.
  /// </para>
  /// </summary>
  public static CNativeWindowHandle ForXlib(nint display, nint xid)
      => new() { Field0 = (ulong)display, Field1 = (ulong)xid };

  /// <summary>
  /// Windows Win32.
  /// <para>
  /// Obtain <paramref name="hInstance"/> and <paramref name="hwnd"/> from
  /// Avalonia's <c>IPlatformHandle</c>.
  /// </para>
  /// </summary>
  public static CNativeWindowHandle ForWin32(nint hInstance, nint hwnd)
      => new() { Field0 = (ulong)hInstance, Field1 = (ulong)hwnd };

#if TARGET_IS_OSX
  /// <summary>
  /// macOS Metal — passes the <c>CAMetalLayer*</c> obtained from
  /// <see cref="MacNativeInputHandler.MetalLayerPointer"/> as <see cref="CNativeWindowHandle.Field0"/>.
  ///
  /// <para><b>Requirements:</b>
  /// <list type="bullet">
  ///   <item>Must be called on the main thread (already enforced by the callback dispatch).</item>
  ///   <item><see cref="MacNativeInputHandler.HookEvents"/> must have been called first so
  ///     <c>setWantsLayer:YES</c> and ISA swizzle are in effect and the CAMetalLayer is live.</item>
  ///   <item>A <see cref="CocoaAutoreleasePool"/> must be active (injected by the thunk).</item>
  /// </list>
  /// </para>
  /// </summary>
  public static CNativeWindowHandle ForMetal(MacNativeInputHandler handler)
      => new() { Field0 = (ulong)handler.MetalLayerPointer, Field1 = 0 };
#endif

  /// <summary>
  /// <c>handle_type</c> constants for the <c>handleType</c> parameter of
  /// <see cref="INativeRuntimeService.AddViewport"/>.
  /// Values mirror Rust's <c>gpu::NativeHandleType</c> repr(u32) enum.
  /// </summary>
  public static class HandleType
  {
    /// <summary>Windowless / headless — no callback is fired.</summary>
    public const uint Windowless = 0;

    /// <summary>Windows Win32 (HINSTANCE + HWND).</summary>
    public const uint Win32 = 1;

    // Wayland = 2 is intentionally absent.
    // Avalonia 11 does not support native Wayland (runs via XWayland).

    /// <summary>Linux Xlib (Display* + Window/XID).</summary>
    public const uint Xlib = 3;

    /// <summary>Linux XCB (xcb_connection_t* + xcb_window_t).</summary>
    public const uint Xcb = 4;

    /// <summary>macOS Metal (CAMetalLayer*).</summary>
    public const uint Metal = 5;
  }
}
