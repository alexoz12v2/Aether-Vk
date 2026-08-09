using System;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.Input;
using AetherVk.Logic.Messages;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.ViewModels;

public partial class VulkanViewportControlViewModel(IWindowInputRouter router, INativeInputHandlerFactory handlerFactory) : ObservableObject, IDisposable
{
  // The context ID used when forwarding native composed keystrokes to the router.
  // Must match the string registered via ActionContext.Id in the Viewport3DView XAML.
  private const string ViewportContextId = "Viewport";

  private readonly IWindowInputRouter _router = router;
  private readonly INativeInputHandlerFactory _handlerFactory = handlerFactory;
  private INativeInputHandler? _handler;
  private IDisposable? _keystrokeSubscription;

  /// <summary>
  /// Called by <see cref="AetherVk.Controls.VulkanViewportControl.CreateNativeControlCore"/>
  /// once the OS-level window handle is available.
  /// Creates the platform-specific <see cref="INativeInputHandler"/> (Windows/Mac/Linux),
  /// subscribes to its composed keystroke stream, and routes matched chords to the
  /// <see cref="IWindowInputRouter"/>.
  /// </summary>
  /// <returns><c>false</c> if <paramref name="handle"/> is zero or factory creation fails.</returns>
  public bool InitializeHandle(IntPtr handle, string handleDescriptor)
  {
    if (handle == IntPtr.Zero) return false;

#if DEBUG
    Console.WriteLine($"Spawining Handle {handle:X} with handleDescriptor \"{handleDescriptor}\"");
#endif

    _handler = _handlerFactory.Create(handle, handleDescriptor, TraceLevel.Max);

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

    return true;
  }

  /// <summary>Broadcasts a fatal error via <see cref="WeakReferenceMessenger"/>.</summary>
  public void ReportFatalError(string message)
    => WeakReferenceMessenger.Default.Send(new CriticalErrorMessage(message));

  public void Dispose()
  {
    _keystrokeSubscription?.Dispose();
    _handler?.Dispose();
  }
}
