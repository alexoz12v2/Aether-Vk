using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Services.NativeInput;

namespace AetherVk.Logic.Services;

// handles both X11 and Wayland

#if !TARGET_IS_LINUX

public unsafe class LinuxNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  protected override bool HookEvents() => false;
  protected override void UnhookEvents() { }
  protected override void DoSetSolidColor(byte r, byte g, byte b) { }
}

#else

/// <summary>
/// Linux X11 input handler. Operates a dedicated secondary X11 display connection on a background
/// thread to intercept raw input events for the native child window, without interfering with
/// Avalonia's primary X11 event loop.
///
/// Key design decisions:
/// - We do NOT call XSelectInput on the secondary connection. That would conflict with
///   Avalonia's ownership of the window's event mask on the primary connection.
/// - Instead, we use XGrabPointer + XGrabKeyboard (active grabs) which redirect server-side
///   event delivery to our secondary connection regardless of who set the event mask.
/// - Ungrab (instead of a dummy ClientMessage) is used to unblock XNextEvent cleanly,
///   avoiding the race condition where a synthetic event is processed by both loops.
/// - UnhookEvents must NOT block the UI thread. It posts cancellation and lets the background
///   thread drain; Dispose waits on a non-UI thread if needed.
///
/// Note: XInitThreads() must be called before any other Xlib call in the process (in Program.cs).
///
/// Wayland: Avalonia.Wayland (12.1+) does not export an IPlatformHandle, so we cannot hook
/// Wayland natively. Users must run with AVALONIA_BACKEND=X11 (XWayland) for this to work.
/// </summary>
public unsafe class LinuxNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  private nint _display = 0;
  private CancellationTokenSource? _cts;
  private Task? _eventLoopTask;
  // Written only from the background thread after grab succeeds; read from UI thread in Dispose.
  private volatile bool _grabbed = false;

  protected override bool HookEvents()
  {
    // Opens a secondary X11 connection dedicated to this handler.
    // Passing 0 → reads $DISPLAY environment variable.
    _display = PInvokeX11.XOpenDisplay(0);
    if (_display == 0)
    {
      Log(TraceLevel.Basic, "XOpenDisplay failed — no DISPLAY or XWayland not running.");
      return false;
    }

    _cts = new CancellationTokenSource();
    _eventLoopTask = Task.Factory.StartNew(
      X11EventLoop,
      _cts.Token,
      TaskCreationOptions.LongRunning,
      TaskScheduler.Default);

    if (_traceLevel >= TraceLevel.Basic)
      Log(TraceLevel.Basic, $"Started LinuxNativeInputHandler for handle 0x{_handle:X}, display 0x{_display:X}");

    return true;
  }

  protected override void UnhookEvents()
  {
    // Signal the XPending polling loop to exit. The background task checks
    // IsCancellationRequested every ~1ms and releases any active dynamic grab before exiting.
    // Do NOT touch _display here — this runs on the UI thread and would race with the
    // background event loop thread.
    _cts?.Cancel();
  }


  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
    // Open a transient connection strictly for thread-safe X11 drawing
    // 0 -> Read $DISPLAY environment variable
    nint disp = PInvokeX11.XOpenDisplay(0);
    if (disp == 0) return;

    nuint pixel = ((uint)r << 16) | ((uint)g << 8) | b;
    PInvokeX11.XSetWindowBackground(disp, (nint)_handle, pixel);
    PInvokeX11.XClearWindow(disp, (nint)_handle);
    PInvokeX11.XFlush(disp);
    PInvokeX11.XCloseDisplay(disp);
  }


  /// <summary>
  /// Called from Dispose (off UI thread). Waits for the event loop to exit, then closes the
  /// display connection.
  /// </summary>
  public override void Dispose()
  {
    base.Dispose(); // signals _rawInputSubject.OnCompleted and calls UnhookEvents via dispatcher

    // Now wait for the background task to finish — this is called from the ViewModel's Dispose
    // which is NOT on the UI thread.
    _eventLoopTask?.Wait(1000);
    _eventLoopTask = null;

    if (_display != 0)
    {
      PInvokeX11.XCloseDisplay(_display);
      _display = 0;
    }
  }

  private void X11EventLoop()
  {
    // --- Event interest ---
    // No permanent global grab: XGrabPointer(ownerEvents=0) would steal ALL pointer events
    // from the display, including title bar drags and close/minimize/maximize clicks that
    // the window manager needs. Instead we register with XSelectInput for events that arrive
    // inside the viewport window only.
    XEventMask mask =
      XEventMask.KeyPressMask    | XEventMask.KeyReleaseMask |
      XEventMask.ButtonPressMask | XEventMask.ButtonReleaseMask |
      XEventMask.PointerMotionMask | XEventMask.ButtonMotionMask;
    PInvokeX11.XSelectInput(_display, (nint)_handle, mask);
    // Do NOT call XSetInputFocus here: window may not be viewable yet, which generates
    // an X11 error event that corrupts the connection state before the event loop starts.
    // Focus is set on every ButtonPress (window is guaranteed viewable at that point).
    PInvokeX11.XFlush(_display);

    // No pre-warm needed: XLookupKeySym is no longer used. Key events are translated
    // via NormalizeX11Keycode (evdev keycode table), which is zero-allocation and never
    // makes any Xlib or X server calls.

    Log(TraceLevel.Basic, "X11EventLoop started (XSelectInput + dynamic drag grab).");

    // --- XPending event loop ---
    // XPending returns 0 when the queue is empty and never blocks, letting us check
    // IsCancellationRequested every ~1ms. This avoids the blocking-XNextEvent-after-ungrab
    // deadlock that plagued earlier iterations.
    PInvokeX11.XEvent xevent = default;
    while (!_cts!.IsCancellationRequested)
    {
      if (PInvokeX11.XPending(_display) == 0)
      {
        Thread.Sleep(1);
        continue;
      }
      PInvokeX11.XNextEvent(_display, &xevent);
      ManageDragGrab(ref xevent);   // update grab state BEFORE processing
      InterceptInputMessage(ref xevent);
    }

    // Release any lingering dynamic grab on clean exit
    if (_grabbed)
    {
      PInvokeX11.XUngrabPointer(_display, 0);
      PInvokeX11.XFlush(_display);
      _grabbed = false;
    }
  }

  /// <summary>
  /// Dynamic grab lifecycle for drag tracking.
  /// <list type="bullet">
  /// <item>ButtonPress: acquire XGrabPointer so drag events are delivered even when the pointer
  ///   moves outside the viewport window. Uses event timestamp (not CurrentTime) to avoid
  ///   the server rejecting the grab as "too old".</item>
  /// <item>ButtonRelease: release the grab once ALL buttons are released, so WM decorations
  ///   (title bar, resize handles, close/minimize/maximize) become interactive again.</item>
  /// </list>
  /// Called before <see cref="InterceptInputMessage"/> so the grab is active during processing.
  /// </summary>
  private void ManageDragGrab(ref PInvokeX11.XEvent ev)
  {
    const uint AllButtonMasks = (1u << 8) | (1u << 9) | (1u << 10); // B1 | B2 | B3 in state

    switch (ev.type)
    {
      case XEventName.ButtonPress:
        if (!_grabbed)
        {
          int r = PInvokeX11.XGrabPointer(
            _display, (nint)_handle,
            ownerEvents: 0,           // all events → us only, Avalonia sees nothing
            eventMask: (uint)(
              XEventMask.ButtonPressMask | XEventMask.ButtonReleaseMask |
              XEventMask.PointerMotionMask | XEventMask.ButtonMotionMask),
            pointerMode: GrabMode.GrabModeAsync,
            keyboardMode: GrabMode.GrabModeAsync,
            confineTo: 0, cursor: 0,
            time: ev.xbutton.time);   // use event timestamp
          if (r == 0)
          {
            _grabbed = true;
            if (_traceLevel >= TraceLevel.Verbose)
              Log(TraceLevel.Verbose, $"Drag grab acquired (button {ev.xbutton.button}).");
          }
        }
        // Restore keyboard focus on every click (Avalonia may have stolen it)
        PInvokeX11.XSetInputFocus(_display, (nint)_handle, revertTo: 2, time: 0);
        PInvokeX11.XFlush(_display);
        break;

      case XEventName.ButtonRelease when _grabbed:
        // state = button state BEFORE this event; the released button bit is still set.
        // Compute which buttons remain held after this release.
        uint releasedMask = ev.xbutton.button switch
        {
          1 => 1u << 8,
          2 => 1u << 9,
          3 => 1u << 10,
          _ => 0u
        };
        uint remaining = (ev.xbutton.state & AllButtonMasks) & ~releasedMask;
        if (remaining == 0)
        {
          PInvokeX11.XUngrabPointer(_display, 0);
          PInvokeX11.XFlush(_display);
          _grabbed = false;
          if (_traceLevel >= TraceLevel.Verbose)
            Log(TraceLevel.Verbose, "Drag grab released.");
        }
        break;
    }
  }

  private void InterceptInputMessage(ref PInvokeX11.XEvent ev)
  {
    const uint Button1Mask = 1 << 8;
    const uint Button2Mask = 1 << 9;
    const uint Button3Mask = 1 << 10;

    // Diagnostic: log every raw event type. Change to Basic for active debugging.
    if (_traceLevel >= TraceLevel.Verbose)
      Log(TraceLevel.Verbose, $"XEvent type={ev.type} ({(int)ev.type})");

    switch (ev.type)
    {
      // --- Keyboard ---
      case XEventName.KeyPress:
        // NormalizeX11Keycode maps evdev keycodes (X11 = evdev + 8) directly to Win32 VK codes.
        // No Xlib call — avoids XLookupKeySym's blocking XGetKeyboardMapping round-trip.
        PublishKeyEvent(NormalizeX11Keycode(ev.xkey.keycode), isDown: true,  GetModifiers(ev.xkey.state));
        break;

      case XEventName.KeyRelease:
        PublishKeyEvent(NormalizeX11Keycode(ev.xkey.keycode), isDown: false, GetModifiers(ev.xkey.state));
        break;

      // --- Mouse Movement ---
      // Only publish motion when a button is held (drag). Hover-only motion
      // (no button pressed) is not consumed by any camera mode and is consistent
      // with Windows (WM_MOUSEMOVE without MK_* flags) and macOS (no mouseMoved publish).
      case XEventName.MotionNotify:
        if ((ev.xmotion.state & Button1Mask) != 0)
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Left,   isDown: true, GetModifiers(ev.xmotion.state));
        if ((ev.xmotion.state & Button3Mask) != 0)
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Right,  isDown: true, GetModifiers(ev.xmotion.state));
        if ((ev.xmotion.state & Button2Mask) != 0)
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Middle, isDown: true, GetModifiers(ev.xmotion.state));
        break;

      // --- Mouse Clicks ---
      case XEventName.ButtonPress:
        PublishMouseEvent(ev.xbutton.x, ev.xbutton.y, GetMouseButton(ev.xbutton.button), isDown: true, GetModifiers(ev.xbutton.state));
        break;

      case XEventName.ButtonRelease:
        PublishMouseEvent(ev.xbutton.x, ev.xbutton.y, GetMouseButton(ev.xbutton.button), isDown: false, GetModifiers(ev.xbutton.state));
        break;
    }
  }

  private static MouseButton GetMouseButton(uint detail) => detail switch
  {
    1 => MouseButton.Left,
    2 => MouseButton.Middle,  // was incorrectly Right
    3 => MouseButton.Right,
    _ => MouseButton.None     // 4/5 = scroll wheel up/down — not handled yet
  };

  private static NativeModifierFlags GetModifiers(uint stateRaw)
  {
    NativeModifierFlags flags = NativeModifierFlags.None;
    XKeyMask state = (XKeyMask)stateRaw;
    if (state.HasFlag(XKeyMask.ShiftMask)) flags |= NativeModifierFlags.Shift;
    if (state.HasFlag(XKeyMask.ControlMask)) flags |= NativeModifierFlags.Control;
    if (state.HasFlag(XKeyMask.Mod1Mask)) flags |= NativeModifierFlags.Alt;
    if (state.HasFlag(XKeyMask.Mod4Mask)) flags |= NativeModifierFlags.Super;
    return flags;
  }

  /// <summary>
  /// Maps X11 evdev-based keycodes (keycode = evdev + 8) directly to Win32-style Virtual Key
  /// codes, matching the unified key standard used by the shared logic dictionary.
  /// No Xlib calls — evdev keycodes are stable across all standard Linux PC keyboards.
  /// </summary>
  private static uint NormalizeX11Keycode(uint keycode) => keycode switch
  {
    // --- Letters A-Z (evdev + 8 → Win32 VK_A-Z 0x41-0x5A) ---
    38 => 0x41, // A    57 => 0x4E, // N
    56 => 0x42, // B    32 => 0x4F, // O
    54 => 0x43, // C    33 => 0x50, // P
    40 => 0x44, // D    24 => 0x51, // Q
    26 => 0x45, // E    27 => 0x52, // R
    41 => 0x46, // F    39 => 0x53, // S
    42 => 0x47, // G    28 => 0x54, // T
    43 => 0x48, // H    30 => 0x55, // U
    31 => 0x49, // I    55 => 0x56, // V
    44 => 0x4A, // J    25 => 0x57, // W
    45 => 0x4B, // K    53 => 0x58, // X
    46 => 0x4C, // L    29 => 0x59, // Y
    58 => 0x4D, // M    52 => 0x5A, // Z

    // --- Digits 0-9 (Win32 0x30-0x39) ---
    19 => 0x30, // 0    10 => 0x31, // 1    11 => 0x32, // 2
    12 => 0x33, // 3    13 => 0x34, // 4    14 => 0x35, // 5
    15 => 0x36, // 6    16 => 0x37, // 7    17 => 0x38, // 8
    18 => 0x39, // 9

    // --- Control / Navigation ---
    9   => 0x1B, // Escape     → VK_ESCAPE
    36  => 0x0D, // Return     → VK_RETURN
    104 => 0x0D, // KP_Enter   → VK_RETURN
    65  => 0x20, // Space      → VK_SPACE
    23  => 0x09, // Tab        → VK_TAB
    22  => 0x08, // BackSpace  → VK_BACK
    119 => 0x2E, // Delete     → VK_DELETE
    118 => 0x2D, // Insert     → VK_INSERT
    110 => 0x24, // Home       → VK_HOME
    115 => 0x23, // End        → VK_END
    112 => 0x21, // Prior/PgUp → VK_PRIOR
    117 => 0x22, // Next/PgDn  → VK_NEXT

    // --- Arrow keys ---
    113 => 0x25, // Left       → VK_LEFT
    111 => 0x26, // Up         → VK_UP
    114 => 0x27, // Right      → VK_RIGHT
    116 => 0x28, // Down       → VK_DOWN

    // --- F-keys (Win32 VK_F1-F12 = 0x70-0x7B) ---
    67 => 0x70, 68 => 0x71, 69 => 0x72, 70 => 0x73,
    71 => 0x74, 72 => 0x75, 73 => 0x76, 74 => 0x77,
    75 => 0x78, 76 => 0x79, 95 => 0x7A, 96 => 0x7B,

    // --- Modifier keys ---
    50  => 0x10, // Shift_L    → VK_SHIFT
    62  => 0x10, // Shift_R    → VK_SHIFT
    37  => 0x11, // Control_L  → VK_CONTROL
    105 => 0x11, // Control_R  → VK_CONTROL
    64  => 0x12, // Alt_L      → VK_MENU
    108 => 0x12, // Alt_R/AltGr→ VK_MENU
    133 => 0x5B, // Super_L    → VK_LWIN
    134 => 0x5C, // Super_R    → VK_RWIN
    66  => 0x14, // CapsLock   → VK_CAPITAL

    _ => keycode  // unmapped — pass raw keycode through
  };
}

#endif


/// <summary>
/// PInvoke abstraction for Xlib. XInitThreads() must be called before any other call.
/// See note in the class: outside TARGET_IS_LINUX because XInitThreads must be called
/// unconditionally when the binary is built with multi-threaded Xlib support.
/// </summary>
/// <seealso href="https://tronche.com/gui/x/xlib/" />
public unsafe static class PInvokeX11
{
  private const string Lib = "libX11.so.6";

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern int XInitThreads();

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern nint XOpenDisplay(nint display);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XCloseDisplay(nint display);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XSelectInput(nint display, nint window, XEventMask eventMask);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XNextEvent(nint display, XEvent* eventReturn);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XSendEvent(nint display, nint window, int propagate, XEventMask eventMask, XEvent* eventSend);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XFlush(nint display);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XSetWindowBackground(nint display, nint window, nuint backgroundPixel);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XClearWindow(nint display, nint window);

  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern nint XLookupKeySym(XKeyEvent* key_event, int index);

  /// <summary>
  /// Active pointer grab: redirects all pointer events to our display connection.
  /// Returns 0 on success (GrabSuccess).
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XGrabPointer(
    nint display, nint grabWindow,
    int ownerEvents, uint eventMask,
    GrabMode pointerMode, GrabMode keyboardMode,
    nint confineTo, nint cursor, nuint time);

  /// <summary>
  /// Releases the active pointer grab.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XUngrabPointer(nint display, nuint time);

  /// <summary>
  /// Active keyboard grab: redirects all keyboard events to our display connection.
  /// Returns 0 on success (GrabSuccess).
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XGrabKeyboard(
    nint display, nint grabWindow,
    int ownerEvents,
    GrabMode pointerMode, GrabMode keyboardMode,
    nuint time);

  /// <summary>
  /// Releases the active keyboard grab.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XUngrabKeyboard(nint display, nuint time);

  /// <summary>
  /// Queries window geometry and attributes. Returns non-zero on success.
  /// We use this to check <see cref="XWindowAttributes.map_state"/> == 2 (IsViewable)
  /// before attempting a grab.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XGetWindowAttributes(nint display, nint window, XWindowAttributes* attributes_return);

  /// <summary>
  /// Sets input focus to the specified window. Used in the XSelectInput fallback so that
  /// KeyPress/KeyRelease events are delivered to the child window rather than Avalonia's root.
  /// revertTo=2 means RevertToParent: if the window unmaps, focus goes back to its parent.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XSetInputFocus(nint display, nint window, int revertTo, nuint time);

  /// <summary>
  /// Returns the number of events in the event queue for the connection.
  /// Non-blocking — returns immediately. Used to avoid blocking XNextEvent when polling
  /// with a cancellation token.
  /// </summary>
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern int XPending(nint display);

  // --- Structs ---

  /// <summary>
  /// Partial layout of XWindowAttributes — we only need map_state.
  /// Full struct is ~136 bytes; we declare the minimum safe size with explicit layout.
  /// map_state is at offset 92 on x86_64 and arm64 64-bit Linux (verified against X11/Xlib.h).
  /// Values: 0=IsUnmapped, 1=IsUnviewable, 2=IsViewable.
  /// Layout (64-bit):
  ///   int x,y,w,h,border,depth (0–23) | Visual* (24–31) | Window root (32–39)
  ///   int class,bitgrav,wingrav,backing_store (40–55) | ulong planes,pixel (56–71)
  ///   Bool save_under (72–75) | pad (76–79) | Colormap (80–87)
  ///   Bool map_installed (88–91) | int map_state (92–95)
  /// </summary>
  [StructLayout(LayoutKind.Explicit, Size = 136)]
  internal struct XWindowAttributes
  {
    [FieldOffset(92)] public int map_state;
  }


  [StructLayout(LayoutKind.Explicit, Size = 192)]
  internal struct XEvent
  {
    [FieldOffset(0)] public XEventName type;
    [FieldOffset(0)] public XKeyEvent xkey;
    [FieldOffset(0)] public XButtonEvent xbutton;
    [FieldOffset(0)] public XMotionEvent xmotion;
    [FieldOffset(0)] public XClientMessageEvent xclient;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct XKeyEvent
  {
    public XEventName type;
    public nuint serial;
    public int send_event;
    public nint display;
    public nint window;
    public nint root;
    public nint subwindow;
    public nuint time;
    public int x, y;
    public int x_root, y_root;
    public uint state;
    public uint keycode;
    public int same_screen;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct XButtonEvent
  {
    public XEventName type;
    public nuint serial;
    public int send_event;
    public nint display;
    public nint window;
    public nint root;
    public nint subwindow;
    public nuint time;
    public int x, y;
    public int x_root, y_root;
    public uint state;
    public uint button;
    public int same_screen;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct XMotionEvent
  {
    public XEventName type;
    public nuint serial;
    public int send_event;
    public nint display;
    public nint window;
    public nint root;
    public nint subwindow;
    public nuint time;
    public int x, y;
    public int x_root, y_root;
    public uint state;
    public byte is_hint;
    private readonly byte _pad1, _pad2, _pad3;
    public int same_screen;
  }

  [StructLayout(LayoutKind.Sequential)]
  internal struct XClientMessageEvent
  {
    public XEventName type;
    public nuint serial;
    public int send_event;
    public nint display;
    public nint window;
    public nint message_type;
    public int format;
    public fixed byte data[20];
  }
}

internal enum GrabMode : int
{
  GrabModeSync = 0,
  GrabModeAsync = 1,
}

[Flags]
internal enum XEventMask : long
{
  NoEventMask = 0L,
  KeyPressMask = 1L << 0,
  KeyReleaseMask = 1L << 1,
  ButtonPressMask = 1L << 2,
  ButtonReleaseMask = 1L << 3,
  EnterWindowMask = 1L << 4,
  LeaveWindowMask = 1L << 5,
  PointerMotionMask = 1L << 6,
  PointerMotionHintMask = 1L << 7,
  Button1MotionMask = 1L << 8,
  Button2MotionMask = 1L << 9,
  Button3MotionMask = 1L << 10,
  Button4MotionMask = 1L << 11,
  Button5MotionMask = 1L << 12,
  ButtonMotionMask = 1L << 13,
  KeymapStateMask = 1L << 14,
  ExposureMask = 1L << 15,
  VisibilityChangeMask = 1L << 16,
  StructureNotifyMask = 1L << 17,
  ResizeRedirectMask = 1L << 18,
  SubstructureNotifyMask = 1L << 19,
  SubstructureRedirectMask = 1L << 20,
  FocusChangeMask = 1L << 21,
  PropertyChangeMask = 1L << 22,
  ColormapChangeMask = 1L << 23,
  OwnerGrabButtonMask = 1L << 24,
}

[Flags]
internal enum XEventName : int
{
  KeyPress = 2,
  KeyRelease = 3,
  ButtonPress = 4,
  ButtonRelease = 5,
  MotionNotify = 6,
  EnterNotify = 7,
  LeaveNotify = 8,
  FocusIn = 9,
  FocusOut = 10,
  KeymapNotify = 11,
  Expose = 12,
  GraphicsExpose = 13,
  NoExpose = 14,
  VisibilityNotify = 15,
  CreateNotify = 16,
  DestroyNotify = 17,
  UnmapNotify = 18,
  MapNotify = 19,
  MapRequest = 20,
  ReparentNotify = 21,
  ConfigureNotify = 22,
  ConfigureRequest = 23,
  GravityNotify = 24,
  ResizeRequest = 25,
  CirculateNotify = 26,
  CirculateRequest = 27,
  PropertyNotify = 28,
  SelectionClear = 29,
  SelectionRequest = 30,
  SelectionNotify = 31,
  ColormapNotify = 32,
  ClientMessage = 33,
  MappingNotify = 34,
  GenericEvent = 35,
  LASTEvent = 36,
}

[Flags]
internal enum XKeyMask : uint
{
  ShiftMask = 1 << 0,
  LockMask = 1 << 1,
  ControlMask = 1 << 2,
  Mod1Mask = 1 << 3,
  Mod2Mask = 1 << 4,
  Mod3Mask = 1 << 5,
  Mod4Mask = 1 << 6,
  Mod5Mask = 1 << 7,
}
