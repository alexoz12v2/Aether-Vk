using System;
using System.Runtime.InteropServices;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.Input;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;

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
  ///   <item>Linux: <c>handleDescriptor == "XID"</c> — handle = XID (Window), parentHandle = Display*</item>
  ///   <item>Windows: <c>handleDescriptor == "HWND"</c> — handle = HWND, parentHandle = HINSTANCE</item>
  ///   <item>macOS: <c>handleDescriptor == "NSView"</c> — CAMetalLayer* from MacNativeInputHandler</item>
  /// </list>
  /// </para>
  /// </summary>
  /// <returns><c>false</c> if <paramref name="handle"/> is zero or factory creation fails.</returns>
  public bool InitializeHandle(IntPtr handle, string handleDescriptor, IntPtr parentHandle = default)
  {
    if (handle == IntPtr.Zero) return false;

#if DEBUG
    Console.WriteLine($"[VulkanViewportControl] InitializeHandle handle={handle:X} desc=\"{handleDescriptor}\" parent={parentHandle:X}");
#endif

    // TODO: from global preferences
    _handler = _handlerFactory.Create(handle, handleDescriptor, TraceLevel.Verbose);

    if (_handler is not INativeInputHandlerSubscribable subscribable)
      return false;

    _keystrokeSubscription = subscribable.SubscribeComposedKeystrokes(stroke =>
    {
      var chord = NativeInputConverter.ToInputChord(stroke);
      if (chord is null) return;

      var last = stroke.Sequence[stroke.Sequence.Length - 1];
      var state = new InputState(
        isPressed: true,
        modifiers: NativeInputConverter.ToInputModifiers(last.Modifiers));

      _router.RouteNativeComposed(ViewportContextId, chord, state);
    });

    // ── Wire AddViewport ─────────────────────────────────────────────────────
    uint w = _viewport3DViewModel.Width;
    uint h = _viewport3DViewModel.Height;

    Func<CNativeWindowHandle>? provider = null;
    uint handleType;

    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      // Avalonia 11: Linux always uses Xlib. handle = XID (Window).
      // We must get the Display* from the input handler, NOT parentHandle (which is the parent XID).
      var xid = handle;
      var display = (_handler as LinuxNativeInputHandler)?.DisplayPointer ?? IntPtr.Zero;
      provider = () => NativeWindowHandleProvider.ForXlib(display, xid);
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
      provider   = () => NativeWindowHandleProvider.ForMetal(macHandler);
      handleType = NativeWindowHandleProvider.HandleType.Metal;
    }
#endif
    else
    {
      handleType = NativeWindowHandleProvider.HandleType.Windowless;
    }

    bool ok = _runtimeService.AddViewport(
      w, h, "Viewport_",
      nativeHandleProvider: provider,
      handleType: handleType,
      out var presentationEngineId,
      out var cameraEntityId);

    if (ok)
    {
#if DEBUG
      Console.WriteLine($"[VulkanViewportControl] AddViewport ok — PE={presentationEngineId} Cam={cameraEntityId}");
#endif
      _viewport3DViewModel.OnViewportCreated(presentationEngineId, cameraEntityId);
    }
    else
    {
      Console.WriteLine("[VulkanViewportControl] AddViewport failed.");
    }

    return ok || handleType == NativeWindowHandleProvider.HandleType.Windowless;
  }

  /// <summary>Broadcasts a fatal error via <see cref="WeakReferenceMessenger"/>.</summary>
  public void ReportFatalError(string message)
    => WeakReferenceMessenger.Default.Send(new CriticalErrorMessage(message));

  /// <summary>
  /// Requests keyboard focus be moved to the native viewport window (XID on Linux, HWND on
  /// Windows). Call this whenever a pointer event passes through the transparent Avalonia overlay
  /// so that subsequent key events reach the native input handler's event loop.
  /// </summary>
  public void FocusViewport() => _handler?.FocusViewportWindow();

  public void Dispose()
  {
    _keystrokeSubscription?.Dispose();
    _handler?.Dispose();
  }
}
