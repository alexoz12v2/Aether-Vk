using System;
using System.Runtime.InteropServices;
using AetherVk.Logic.Input;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.NativeInput;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class VulkanViewportControlViewModel(
  IWindowInputRouter router,
  INativeInputHandlerFactory handlerFactory,
  INativeRuntimeService runtimeService,
  Viewport3DViewModel viewport3DViewModel
) : ObservableObject, IDisposable
{
  // The context ID used when forwarding native composed keystrokes to the router.
  // Must match the string registered via ActionContext.Id in the Viewport3DView XAML.
  private const string ViewportContextId = "Viewport";

  private readonly IWindowInputRouter _router = router;
  private readonly INativeInputHandlerFactory _handlerFactory = handlerFactory;
  private readonly INativeRuntimeService _runtimeService = runtimeService;
  private readonly Viewport3DViewModel _viewport3DViewModel = viewport3DViewModel;
  private INativeInputHandler? _handler;
  private IDisposable? _keystrokeSubscription;
  private IDisposable? _mouseSubscription;
  private MouseButton _lastPressedButton = MouseButton.None;

  private ulong _presentationEngineId;

  /// <summary>
  /// What this view is for. <see cref="ViewportRole.NucleusMini"/> owns a presentation engine and
  /// nothing else: no key or pointer routing, no camera registration (no
  /// <see cref="Viewport3DViewModel.OnViewportCreated"/>, so the camera registry, mode bar, settings,
  /// billboards and saves never see it), and its size and position never reach the main viewport's
  /// view model.
  /// </summary>
  public ViewportRole Role { get; init; } = ViewportRole.Main;

  private bool IsMini => Role == ViewportRole.NucleusMini;

  /// <summary>The attached presentation engine, 0 before <see cref="InitializeHandle"/> and after
  /// <see cref="Dispose"/>.</summary>
  internal ulong PresentationEngineId => _presentationEngineId;

  /// <summary>The camera the engine created with the presentation engine (0 without one).</summary>
  internal ulong CameraEntityId { get; private set; }

  /// <summary>The native window (XID / HWND) of the last <see cref="InitializeHandle"/>.</summary>
  internal IntPtr NativeHandle { get; private set; }

  /// <summary>How many times a native window was handed over (a recreated host counts again).</summary>
  internal int InitializeCount { get; private set; }

  private uint _miniWidth;
  private uint _miniHeight;

  /// <summary>The mini-view's size in physical pixels (0 before its first layout).</summary>
  internal (uint Width, uint Height) MiniPixelSize => (_miniWidth, _miniHeight);

  /// <summary>Raised once the engine attached a presentation engine and its camera to the native
  /// window (<see cref="PresentationEngineId"/> and <see cref="CameraEntityId"/> are set).</summary>
  public event Action? Attached;

  /// <summary>Raised when the mini-view's size changed (physical pixels).</summary>
  public event Action<uint, uint>? MiniResized;

  /// <summary>
  /// The mini-view's pointer over its native window: position (px, from its top-left corner),
  /// whether a button is held, and whether this event released it. Called on the input thread.
  /// Null (the default): the mini-view subscribes to no input at all.
  /// </summary>
  public Action<double, double, bool, bool>? MiniPointer { get; init; }

  /// <summary>
  /// Called by <see cref="AetherVk.Controls.VulkanViewportControl.CreateNativeControlCore"/>
  /// once the OS-level window handle is available.
  /// <para>
  /// Creates the platform-specific <see cref="INativeInputHandler"/> (input hooks),
  /// then calls <see cref="INativeRuntimeService.AddViewport"/> with the correct platform
  /// handle so Rust creates the Vulkan surface and swapchain against the real OS window.
  /// </para>
  /// <para>
  /// Platform mapping (Avalonia 11 — Linux always uses Xlib):
  /// <list type="bullet">
  ///   <item>Linux: <c>handleDescriptor == "XID"</c> — handle = XID (Window); the surface's Display* is
  ///     the process-wide <see cref="X11SurfaceDisplay"/> (never the input handler's)</item>
  ///   <item>Windows: <c>handleDescriptor == "HWND"</c> — handle = HWND, parentHandle = HINSTANCE</item>
  ///   <item>macOS: <c>handleDescriptor == "NSView"</c> — CAMetalLayer* from MacNativeInputHandler</item>
  /// </list>
  /// </para>
  /// </summary>
  /// <returns><c>false</c> if <paramref name="handle"/> is zero or factory creation fails.</returns>
  public bool InitializeHandle(
    IntPtr handle,
    string handleDescriptor,
    IntPtr parentHandle = default
  )
  {
    if (handle == IntPtr.Zero)
      return false;

#if DEBUG
    Console.WriteLine(
      $"[VulkanViewportControl] InitializeHandle handle={handle:X} desc=\"{handleDescriptor}\" parent={parentHandle:X}"
    );
#endif

    NativeHandle = handle;
    InitializeCount++;

    // TODO: from global preferences
    _handler = _handlerFactory.Create(handle, handleDescriptor, TraceLevel.Verbose);

    if (_handler is not INativeInputHandlerSubscribable subscribable)
      return false;

    // Keystrokes always go to the "Viewport" context, i.e. the main viewport's handler: the
    // mini-view has no action context of its own, so a key typed while its native window has the
    // X/Win32 focus (after a click or a drag on it) still drives the main camera instead of being
    // dropped. The mini's own input is only its drag (below).
    _keystrokeSubscription = subscribable.SubscribeComposedKeystrokes(stroke =>
    {
      var chord = NativeInputConverter.ToInputChord(stroke);
      if (chord is null)
        return;

      var last = stroke.Sequence[stroke.Sequence.Length - 1];
      var state = new InputState(
        isPressed: true,
        modifiers: NativeInputConverter.ToInputModifiers(last.Modifiers)
      );

      _router.RouteNativeComposed(ViewportContextId, chord, state);
    });

    if (IsMini && MiniPointer is { } miniPointer)
    {
      // the mini-view's own drag: never the main viewport's operators
      _mouseSubscription = subscribable.SubscribeNativeMouseEvents(ev =>
        miniPointer(ev.X, ev.Y, ev.IsDown && ev.Button != MouseButton.None, !ev.IsDown && ev.Button != MouseButton.None));
    }
    else _mouseSubscription = IsMini ? null : subscribable.SubscribeNativeMouseEvents(ev =>
    {
      var state = new InputState(
        isPressed: ev.IsDown,
        modifiers: NativeInputConverter.ToInputModifiers(ev.Modifiers)
      );

      var payload = new System.Numerics.Vector2((float)ev.X, (float)ev.Y);
      string actionId = "viewport.pointer_delta";

      if (ev.IsDown)
      {
        if (_lastPressedButton != ev.Button)
        {
          _lastPressedButton = ev.Button;
          if (ev.Button == MouseButton.Middle)
          {
            if (state.Modifiers.HasFlag(InputModifiers.Shift))
              actionId = ViewportAction.StartPan.ToCmdString();
            else if (state.Modifiers.HasFlag(InputModifiers.Ctrl))
              actionId = ViewportAction.StartZoom.ToCmdString();
            else
              actionId = ViewportAction.StartOrbit.ToCmdString();
          }
          else
          {
            actionId = "viewport.pointer_pressed";
          }
        }
        else
        {
          actionId = "viewport.pointer_delta";
        }
      }
      else if (ev.Button != MouseButton.None)
      {
        if (_lastPressedButton == ev.Button)
        {
          _lastPressedButton = MouseButton.None;
        }
        actionId = "viewport.pointer_end";
      }

      var action = new AppAction(actionId, payload);
      _viewport3DViewModel.OperatorStack.Process(action, state);
    });

    // ── Wire AddViewport ─────────────────────────────────────────────────────
    // a mini-view created before its first layout (inside a collapsed parent) has no size yet
    uint w = IsMini ? (_miniWidth > 0 ? _miniWidth : 320u) : _viewport3DViewModel.Width;
    uint h = IsMini ? (_miniHeight > 0 ? _miniHeight : 240u) : _viewport3DViewModel.Height;

    Func<CNativeWindowHandle>? provider = null;
    uint handleType;

    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      // Avalonia 11: Linux always uses Xlib. handle = XID (Window); parentHandle is the parent XID,
      // not a Display*. Every surface of the process (main and mini-view) is created on the one
      // X11SurfaceDisplay connection: surfaces on different connections crash the NVIDIA driver in
      // vkQueuePresentKHR. The input handler keeps its own connection for its event loop only.
      // Read inside the provider: the runtime calls it on the UI thread during AddViewport.
      var xid = handle;
      provider = () => NativeWindowHandleProvider.ForXlib(X11SurfaceDisplay.Get(), xid);
      handleType = NativeWindowHandleProvider.HandleType.Xlib;
    }
    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
      var hwnd = handle;
      var hInstance = parentHandle;
      provider = () => NativeWindowHandleProvider.ForWin32(hInstance, hwnd);
      handleType = NativeWindowHandleProvider.HandleType.Win32;
    }
#if TARGET_IS_OSX
    else if (_handler is MacNativeInputHandler macHandler)
    {
      provider = () => NativeWindowHandleProvider.ForMetal(macHandler);
      handleType = NativeWindowHandleProvider.HandleType.Metal;
    }
#endif
    else
    {
      handleType = NativeWindowHandleProvider.HandleType.Windowless;
    }

    bool ok = _runtimeService.AddViewport(
      w,
      h,
      "Viewport_",
      nativeHandleProvider: provider,
      handleType: handleType,
      out var presentationEngineId,
      out var cameraEntityId,
      Role
    );

    if (ok)
    {
      _presentationEngineId = presentationEngineId;
      CameraEntityId = cameraEntityId;
#if DEBUG
      Console.WriteLine(
        $"[VulkanViewportControl] AddViewport ok — PE={presentationEngineId} Cam={cameraEntityId}"
      );
#endif
      // the mini-view's camera never registers: single-viewport consumers see the main one only
      if (!IsMini)
        _viewport3DViewModel.OnViewportCreated(presentationEngineId, cameraEntityId);
      Attached?.Invoke();
    }
    else
    {
      Console.WriteLine($"[VulkanViewportControl] AddViewport ({Role}) failed.");
    }

    return ok || handleType == NativeWindowHandleProvider.HandleType.Windowless;
  }

  /// <summary>Broadcasts a fatal error via <see cref="WeakReferenceMessenger"/>.</summary>
  public void ReportFatalError(string message) =>
    WeakReferenceMessenger.Default.Send(new CriticalErrorMessage(message));

  /// <summary>
  /// Requests keyboard focus be moved to the native viewport window (XID on Linux, HWND on
  /// Windows). Call this whenever a pointer event passes through the transparent Avalonia overlay
  /// so that subsequent key events reach the native input handler's event loop.
  /// </summary>
  public void FocusViewport() => _handler?.FocusViewportWindow();

  /// <summary>The control's size in Avalonia DIPs at <paramref name="renderScaling"/>: the camera
  /// works in the swapchain's physical pixels.</summary>
  public void NotifyResized(double width, double height, double renderScaling = 1.0)
  {
    uint w = Viewport3DViewModel.DipToPixels(width, renderScaling);
    uint h = Viewport3DViewModel.DipToPixels(height, renderScaling);
    if (IsMini)
    {
      if (_miniWidth == w && _miniHeight == h)
        return;
      _miniWidth = w;
      _miniHeight = h;
      MiniResized?.Invoke(w, h);
      return;
    }
    _viewport3DViewModel.SetPixelSize(w, h);
  }

  /// <summary>The control's top-left corner on the desktop, physical px (billboard coordinates),
  /// and its screen's render scaling.</summary>
  public void NotifyScreenOrigin(double x, double y, double renderScaling = 1.0)
  {
    if (!IsMini)
      _viewport3DViewModel.CameraService.SetViewportScreenOrigin(x, y, renderScaling);
  }

  public void Dispose()
  {
    // The engine goes before the handler only so no frame is presented after the input hooks are
    // gone; the surface itself is on X11SurfaceDisplay (process lifetime), not on the handler's
    // connection, so closing that connection can no longer pull the display from under a surface.
    if (_presentationEngineId != 0)
    {
      _runtimeService.RemoveViewport(_presentationEngineId);
      _presentationEngineId = 0;
    }
    _keystrokeSubscription?.Dispose();
    _mouseSubscription?.Dispose();
    _handler?.Dispose();
  }
}
