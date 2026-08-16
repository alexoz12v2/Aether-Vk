using System;
using System.Linq;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Services.NativeInput;

namespace AetherVk.Logic.Services;

public enum TraceLevel
{
  None,
  Basic,
  Verbose,
  Max
}

public abstract class NativeInputHandlerBase : INativeInputHandlerSubscribable
{
  protected readonly IntPtr _handle;
  protected readonly string _handleDescriptor;
  protected readonly TraceLevel _traceLevel;
  protected readonly IUiThreadDispatcher _dispatcher;
  protected readonly ISchedulerProvider _schedulerProvider;
  protected bool _isHooked = false;

  // reactive subjects
  private readonly Subject<NativeInputEvent> _rawInputSubject = new();
  // public observables (only to logic assembly)
  internal IObservable<NativeInputEvent> RawInput => _rawInputSubject.ObserveOn(_schedulerProvider.Background);
  internal IObservable<ComposedKeystroke> ComposedKeystrokes { get; }

  // INativeInputHandlerSubscribable implementation — subscribes to the internal Rx pipeline
  // without exposing Rx or ComposedKeystroke to ui-app (both are internal to ui-logic).
  IDisposable INativeInputHandlerSubscribable.SubscribeComposedKeystrokes(Action<ComposedKeystroke> onNext)
    => ComposedKeystrokes.Subscribe(onNext);

  // private protected = only derived classes in the same assembly can use this
  private protected NativeInputHandlerBase(
    IntPtr handle, string handleDescriptor, TraceLevel traceLevel, IUiThreadDispatcher dispatcher, ISchedulerProvider schedulerProvider)
  {
    _handle = handle;
    _handleDescriptor = handleDescriptor;
    _traceLevel = traceLevel;
    _dispatcher = dispatcher;
    _schedulerProvider = schedulerProvider;

    // Composition Pipeline
    ComposedKeystrokes = _rawInputSubject
      .OfType<NativeKeyEvent>() // select all key events
      .Where(k => k.IsDown) // compose key presses, not releases
      .Buffer(TimeSpan.FromMilliseconds(400), _schedulerProvider.Background) // accumulate on background for 400ms
      .Where(buffer => buffer.Count > 0) // if you accumulated something, then continue
      .Select(buffer => new ComposedKeystroke([.. buffer], DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()))
      .ObserveOn(_schedulerProvider.Background); // emit on the background thread

    // Hook only when we have a valid handle and haven't hooked yet
    if (!_isHooked && handle != IntPtr.Zero)
    {
      if (_dispatcher.CheckAccess())
      {
        if (HookEvents())
          _isHooked = true;
      }
      else
      {
        _dispatcher.Dispatch(() =>
        {
          if (HookEvents())
            _isHooked = true;
          // TODO: send a fatal error message to the UI on failure
        });
      }
    }
  }

  #region Publishing_API

  // private protected = only derived classes in the same assembly can use this
  private protected void PublishKeyEvent(uint keyCode, bool isDown, NativeModifierFlags modifiers)
  {
    var ev = new NativeKeyEvent(keyCode, isDown, modifiers, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
    _rawInputSubject.OnNext(ev);
    if (_traceLevel >= TraceLevel.Verbose) // avoid string interpolation
      Log(TraceLevel.Verbose, $"[Key] Code: {keyCode}, Down: {isDown}, Mods: {modifiers}");
  }

  // private protected = only derived classes in the same assembly can use this
  private protected void PublishMouseEvent(double x, double y, MouseButton button, bool isDown, NativeModifierFlags modifiers)
  {
    var ev = new NativeMouseEvent(x, y, button, isDown, modifiers, DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
    _rawInputSubject.OnNext(ev);
    if (button != MouseButton.None && _traceLevel >= TraceLevel.Verbose)
      Log(TraceLevel.Verbose, $"[Mouse] {button} Down: {isDown} at ({x}, {y}) | mods: {modifiers}");
  }

  #endregion

  /// <summary>Ensured to be called by Main Thread</summary>
  protected abstract bool HookEvents();

  /// <summary>Ensured to be called by Main Thread</summary>
  protected abstract void UnhookEvents();

  protected void Log(TraceLevel level, string message)
  {
    if (_traceLevel >= level)
    {
#if DEBUG
      System.Diagnostics.Debug.WriteLine($"[{DateTime.Now:O}] [{GetType().Name}] {message}");
#endif
      Console.WriteLine($"[{DateTime.Now:O}] [{GetType().Name}] {message}");
    }
  }

  // Note: public abstract is an antipattern. This is a temporary method to test native control
  // calls the abstract method from the main thread
  public void SetSolidColor(byte r, byte g, byte b)
  {
    _dispatcher.Dispatch(() => DoSetSolidColor(r, g, b));
  }

  protected abstract void DoSetSolidColor(byte r, byte g, byte b);

  public virtual void Dispose()
  {
    if (_handle != IntPtr.Zero && _isHooked)
    {
      _dispatcher.Dispatch(() =>
      {
        UnhookEvents();
        _isHooked = false;
      });
    }
    if (_isHooked && _handle == IntPtr.Zero)
    {
      // error hadling? maybe fatalerror message
      Console.WriteLine($"[{DateTime.Now:O}] [{GetType().Name}] Hooked but Null window handle!");
    }

    // events
    _rawInputSubject.OnCompleted();
    _rawInputSubject.Dispose();

    // see if this necessary
    GC.SuppressFinalize(this);
  }
}
