using System;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Tests;

/// <summary>An <see cref="ICometMessenger"/> that delivers (a private strong-reference messenger).</summary>
public sealed class ForwardingCometMessenger : ICometMessenger
{
  private readonly StrongReferenceMessenger _inner = new();

  public bool IsRegistered<TMessage, TToken>(object recipient, TToken token)
    where TMessage : class where TToken : IEquatable<TToken> => _inner.IsRegistered<TMessage, TToken>(recipient, token);

  public void Register<TRecipient, TMessage, TToken>(TRecipient recipient, TToken token, MessageHandler<TRecipient, TMessage> handler)
    where TRecipient : class where TMessage : class where TToken : IEquatable<TToken> => _inner.Register(recipient, token, handler);

  public void UnregisterAll(object recipient) => _inner.UnregisterAll(recipient);

  public void UnregisterAll<TToken>(object recipient, TToken token) where TToken : IEquatable<TToken> =>
    _inner.UnregisterAll(recipient, token);

  public void Unregister<TMessage, TToken>(object recipient, TToken token)
    where TMessage : class where TToken : IEquatable<TToken> => _inner.Unregister<TMessage, TToken>(recipient, token);

  public TMessage Send<TMessage, TToken>(TMessage message, TToken token)
    where TMessage : class where TToken : IEquatable<TToken> => _inner.Send(message, token);

  public void Cleanup() => ((IMessenger)_inner).Cleanup();

  public void Reset() => _inner.Reset();
}
