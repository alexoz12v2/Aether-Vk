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
  protected override bool HookEvents()
  {
    return false;
  }

  protected override void UnhookEvents()
  {
  }

  protected override void DoSetSolidColor(byte r, byte g, byte b)
  {
  }
}

#else

/// <summary>
/// - Xlib vs XCB: we'll use Xlib, cause avalonia provides the <see cref="NativeInputHandlerBase" />
///   with an `XID` (32-bit Window ID), and that is the same in both X11 clients. Furthermore, Xlib,s
///   `XNextEvent` loop is simpler to implement in C# rather than XCB's asynchronous iteration.
///   on vulkan, whether we use `vkCreateXcbSurfaceKHR` or `vkCreateXlibSurfaceKHR` doesn't change
///   much, as the XID is the same
///
/// - Since X11 handler events over a network socket, we can open a secondary X11 display connection
///   dedicated solely to reading inputs for the child window, and run it on a background thread.
///   This avoids stalling Avalonia's primary UI thread entirely.
///
///   Furthermore, X11 natively mimicks the Win32 `SetCapture` logic out of the box, through a
///   feature called "Implicit Pointer Grabbing".
///
/// - A note about wayland: Experimental, opt in through `UseWayland()` from `Avalonia.Wayland`
///   support was introduced in Avalonia 12.1 (late July 2026), but it still doesn't export a
///   `IPlatformHandle` for us to use.
///   <see href="https://github.com/AvaloniaUI/Avalonia/blob/12.1.1/src/Avalonia.Wayland/WindowImplBase.cs" />, line 32
///   `public IPlatformHandle? Handle => null;`
/// </summary>
public unsafe class LinuxNativeInputHandler(IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  : NativeInputHandlerBase(handle, handleDescriptor, traceLevel, dispatcher, schedulerProvider)
{
  private nint _display = 0;
  private CancellationTokenSource? _cancellationTokenSource;
  private Task? _eventLoopTask;

  protected override bool HookEvents()
  {
    // --- X11 Background hooking ---
    // By opening our own connection to the X Server, we can block on XNextEvent
    // in a backgoroudn thread without stalling Avalonia's primary X11 UI Loop
    // Passing 0 -> Read environment variable $DISPLAY (eg :0 or :1)
    _display = PInvokeX11.XOpenDisplay(0);
    if (_display == 0) return false;
    // Subscribe to Mouse and Keyboard event for this specific XID (child window)
    // Other than PointerMotionMask, we also need ButtonMotionMask, otherwise dragging won't work
    // outside window
    XEventMask mask = XEventMask.KeyPressMask | XEventMask.KeyReleaseMask |
     XEventMask.ButtonPressMask | XEventMask.ButtonReleaseMask | XEventMask.PointerMotionMask | XEventMask.ButtonMotionMask;
    PInvokeX11.XSelectInput(_display, (nint)_handle, mask);
    PInvokeX11.XFlush(_display);

    _isHooked = true;
    _cancellationTokenSource = new CancellationTokenSource();

    _eventLoopTask = Task.Factory.StartNew(X11EventLoop, _cancellationTokenSource.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);

    return true;
  }

  protected override void UnhookEvents()
  {
    if (_display != 0)
    {
      _cancellationTokenSource?.Cancel();

      // XNextEvent is a blocking socket read. We must send a dummy event to the window to unblock
      // the background thread so it can evaluate the cancellation token and exit gracefully
      PInvokeX11.XEvent dummyEvent = default;
      dummyEvent.type = XEventName.ClientMessage;
      dummyEvent.xclient.window = _handle;
      dummyEvent.xclient.format = 32;

      PInvokeX11.XSendEvent(_display, _handle, 0, 0, &dummyEvent);
      PInvokeX11.XFlush(_display);

      _eventLoopTask?.Wait(500); // wait for thread to close safely

      PInvokeX11.XCloseDisplay(_display);
      _display = 0;
    }
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

  private void X11EventLoop()
  {
    PInvokeX11.XEvent xevent = default;

    while (!_cancellationTokenSource!.IsCancellationRequested)
    {
      // Blocks until the user interacts with the window. Using ~0% CPU when Idle
      PInvokeX11.XNextEvent(_display, &xevent);

      if (_cancellationTokenSource.IsCancellationRequested) break;

      InterceptInputMessage(ref xevent);
    }
  }

  private void InterceptInputMessage(ref PInvokeX11.XEvent ev)
  {
    const uint Button1Mask = 1 << 8;
    const uint Button2Mask = 1 << 9;
    const uint Button3Mask = 1 << 10;

    switch (ev.type)
    {
      // --- Keyboard ---
      case XEventName.KeyPress:
        // index 0 ignores Shift state, so 'Shift + A' still reports the base 'a' key, like Win32
        nint keysymDown = 0;
        fixed (PInvokeX11.XKeyEvent* pEv = &ev.xkey)
          keysymDown = PInvokeX11.XLookupKeySym(pEv, 0);

        PublishKeyEvent(NormalizeX11KeySym(keysymDown), isDown: true, GetModifiers(ev.xkey.state));
        break;
      case XEventName.KeyRelease:
        // index 0 ignores Shift state, so 'Shift + A' still reports the base 'a' key, like Win32
        nint keysymUp = 0;
        fixed (PInvokeX11.XKeyEvent* pEv = &ev.xkey)
          keysymUp = PInvokeX11.XLookupKeySym(pEv, 0);

        PublishKeyEvent(NormalizeX11KeySym(keysymUp), isDown: false, GetModifiers(ev.xkey.state));
        break;

      // --- Mouse Movement ---
      case XEventName.MotionNotify:
        bool isDragging = false;
        // ev.state holds the active button mask
        if ((ev.xmotion.state & Button1Mask) != 0)
        {
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Left, isDown: true, GetModifiers(ev.xmotion.state));
          isDragging = true;
        }
        if ((ev.xmotion.state & Button3Mask) != 0) // Button3 is right
        {
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Right, isDown: true, GetModifiers(ev.xmotion.state));
          isDragging = true;
        }
        if ((ev.xmotion.state & Button2Mask) != 0) // Button2 is middle
        {
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.Middle, isDown: true, GetModifiers(ev.xmotion.state));
          isDragging = true;
        }

        if (!isDragging)
          PublishMouseEvent(ev.xmotion.x, ev.xmotion.y, MouseButton.None, isDown: false, GetModifiers(ev.xmotion.state));

        break;

      // --- Mouse Clicks ---
      case XEventName.ButtonPress:
        // Note: X11 implicitly grabs the pointer on Button Press (Active Pointer Grabbing)
        // This natively mimics Win32 SetCapture and default AppKit behaviour for dragging outside the window
        PublishMouseEvent(ev.xbutton.x, ev.xbutton.y, GetMouseButton(ev.xbutton.button), isDown: true, GetModifiers(ev.xbutton.state));
        break;
      case XEventName.ButtonRelease:
        PublishMouseEvent(ev.xbutton.x, ev.xbutton.y, GetMouseButton(ev.xbutton.button), isDown: false, GetModifiers(ev.xbutton.state));
        // Note: X11 implicitly ungrabs the pointer when all buttons are released
        break;
    }
  }

  private static MouseButton GetMouseButton(uint detail) => detail switch
  {
    1 => MouseButton.Left,
    2 => MouseButton.Right,
    3 => MouseButton.Right,
    _ => MouseButton.None // Note: 4, 5 are scroll wheel up and down. Not handling for now
  };

  private static NativeModifierFlags GetModifiers(uint stateRaw)
  {
    NativeModifierFlags flags = NativeModifierFlags.None;
    XKeyMask state = (XKeyMask)stateRaw;
    if (state.HasFlag(XKeyMask.ShiftMask)) flags |= NativeModifierFlags.Shift;
    if (state.HasFlag(XKeyMask.ControlMask)) flags |= NativeModifierFlags.Control;
    if (state.HasFlag(XKeyMask.Mod1Mask)) flags |= NativeModifierFlags.Alt; // usually alt on linux?
    if (state.HasFlag(XKeyMask.Mod4Mask)) flags |= NativeModifierFlags.Super; // usually win on linux?
    return flags;
  }

  /// <summary>
  /// Normalizes X11 KeySyms into the unified Win32-style virtual key standard.
  /// </summary>
  private static uint NormalizeX11KeySym(nint keysym)
  {
    // 1. Letters: If it's a lowercase letter (0x61 'a' to 0x7A 'z'),
    // convert it to uppercase (0x41 'A' to 0x5A 'Z')
    if (keysym >= 0x61 && keysym <= 0x7A)
    {
      return (uint)(keysym - 0x20);
    }

    // 2. Control Keys: X11 prefixes control keys with 0xFF00.
    // E.g., XK_Escape is 0xFF1B. XK_Return is 0xFF0D.
    // We bitwise AND with 0x00FF to strip the prefix and perfectly match the Windows equivalents.
    if ((keysym & 0xFF00) == 0xFF00)
    {
      uint masked = (uint)(keysym & 0x00FF);

      // Only apply to Return and Escape, pass other control keys through unmodified
      if (masked == 0x0D || masked == 0x1B)
        return masked;
    }

    // Space (XK_Space) is 0x0020, which matches exactly without modification
    return (uint)keysym;
  }
}

#endif


/// <summary>
/// PInvoke Abstraction for X11 xlib client. You can locate it if ` sudo ldconfig -p | grep libX11`
/// or `find /usr/lib /lib -name "libX11.so.6" 2>/dev/null` to verify whether it's there.
/// DllImport on linux calls `dlopen`, which should resolve for the dynamic linker `ld-linux.so` to
///  - On x86_64 Ubuntu: /usr/lib/x86_64-linux-gnu/
///  - On ARM64 Ubuntu: /usr/lib/aarch64-linux-gnu/
///  - On Fedora/RHEL: /usr/lib64/
///  - On Arch / Alpine: /usr/lib/
/// Lince `libX11.so.6` is a standard SONAME, it should be portable, unless we are on a minimal
/// headless distro or Container (like alpine linux).
///
/// As other implementations, we avoid object marshalling and run under AOT-constraints
///
/// Note how this is outside the TARGET_IS_LINUX block. That's because, since we are opening
/// secondary `XOpenDisplay` on background thread, while avalonia is communicating with the X Server
/// on the main htread, we must call `XInitThreads` before any other Xlib calls in the entire
/// process, otherwise we risk random SIGSEGVs
/// </summary>
/// <seealso href="https://tronche.com/gui/x/xlib/" />
public unsafe static class PInvokeX11
{
  private const string Lib = "libX11.so.6";

  // --- P/Invokes ---

  /// <summary>Note how this is the only public method cause needs to be the first thing to be
  /// called before any other Xlib calls in Program.cs</summary>
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

  // maps hardware evdev key into a standardized KeySym. We need to invoke this and then handle
  // lowercase (converts to ASCII, but we want raw logical key), and extract control keys
  [DllImport(Lib, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  internal static extern nint XLookupKeySym(XKeyEvent* key_event, int index);

  // --- Structs (AOT Safe Zero-Marshaling) ---

  // union type
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
    // XClientMessage contains a union of either 20 8-bit, 10 16-bit, 5 32-bit values.
    public fixed byte data[20];
  }
}

/// <summary>
/// Generated by `grep -B 4 -A 49 "KeyPress" /usr/include/X11/X.h`
/// Input Event Masks. Used as event-mask window attribute and as arguments to Grab requests. Not to
/// be confused with event names
/// <summary>
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

/// <summary>
/// Generated by `grep -B 4 -A 49 "KeyPress" /usr/include/X11/X.h`
/// Event names. Used in "type" field in XEvent structures. Not to be
/// confused with event masks above. They start from 2 because 0 and 1
/// are reserved in the protocol for errors and replies.
/// </summary>
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
  LASTEvent = 36
}

/// <summary>
/// Generated by `grep -B 4 -A 49 "KeyPress" /usr/include/X11/X.h`
/// Key masks. Used as modifiers to GrabButton and GrabKey, results of QueryPointer,
/// state in various key-, mouse-, and button-related events.
/// </summary>
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
  Mod5Mask = 1 << 7
}

