using System;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Services;

/// <summary>
/// Base class for domain-scoped messenger singletons.
/// Wraps a <see cref="StrongReferenceMessenger"/> and implements <see cref="IMessenger"/>
/// by delegation (StrongReferenceMessenger is sealed in CommunityToolkit.Mvvm 8.x).
/// </summary>
internal abstract class DomainMessenger : IMessenger
{
  protected readonly StrongReferenceMessenger Inner = new();

  public bool IsRegistered<TMessage, TToken>(object recipient, TToken token)
    where TMessage : class
    where TToken : IEquatable<TToken> => Inner.IsRegistered<TMessage, TToken>(recipient, token);

  public void Register<TRecipient, TMessage, TToken>(
    TRecipient recipient,
    TToken token,
    MessageHandler<TRecipient, TMessage> handler
  )
    where TRecipient : class
    where TMessage : class
    where TToken : IEquatable<TToken> => Inner.Register(recipient, token, handler);

  public void Unregister<TMessage, TToken>(object recipient, TToken token)
    where TMessage : class
    where TToken : IEquatable<TToken> => Inner.Unregister<TMessage, TToken>(recipient, token);

  public void UnregisterAll(object recipient) => Inner.UnregisterAll(recipient);

  public void UnregisterAll<TToken>(object recipient, TToken token)
    where TToken : IEquatable<TToken> => Inner.UnregisterAll(recipient, token);

  public TMessage Send<TMessage, TToken>(TMessage message, TToken token)
    where TMessage : class
    where TToken : IEquatable<TToken> => Inner.Send(message, token);

  public void Cleanup() { }

  public void Reset() => Inner.Reset();
}

internal sealed class CometMessenger : DomainMessenger, ICometMessenger { }

internal sealed class ConsoleMessenger : DomainMessenger, IConsoleMessenger { }

internal sealed class LayoutMessenger : DomainMessenger, ILayoutMessenger { }
