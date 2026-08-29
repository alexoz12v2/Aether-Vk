using System;
using AetherVk.Logic.Services.NativeInput;

namespace AetherVk.Logic.Services;

/// <summary>
/// Internal extension of <see cref="INativeInputHandler"/> that exposes the
/// composed keystroke subscription. Only visible within <c>aethervk.ui-logic</c>
/// — the <see cref="NativeInput.ComposedKeystroke"/> type is also internal.
/// </summary>
internal interface INativeInputHandlerSubscribable : INativeInputHandler
{
  /// <summary>
  /// Subscribes to composed keystroke events emitted by this handler.
  /// The returned <see cref="IDisposable"/> cancels the subscription when disposed.
  /// </summary>
  IDisposable SubscribeComposedKeystrokes(Action<ComposedKeystroke> onNext);
  IDisposable SubscribeNativeMouseEvents(Action<NativeMouseEvent> onNext);
}
