using System;

namespace AetherVk.Logic.Services;

/// <summary>
/// Uniquely identifies one session instance within a <see cref="TabStateService{TSession}"/>.
/// Session numbers start at 1 and auto-increment; they are never reused within an application lifetime.
/// </summary>
public readonly struct SessionId : IEquatable<SessionId>
{
  public Type   TabSessionType { get; }
  public int    Number         { get; }

  public SessionId(Type tabSessionType, int number)
  {
    TabSessionType = tabSessionType;
    Number         = number;
  }

  public bool Equals(SessionId other) =>
    TabSessionType == other.TabSessionType && Number == other.Number;

  public override bool Equals(object? obj) => obj is SessionId s && Equals(s);

  public override int GetHashCode()
  {
    unchecked
    {
      return (TabSessionType.GetHashCode() * 397) ^ Number;
    }
  }

  public static bool operator ==(SessionId left, SessionId right) => left.Equals(right);
  public static bool operator !=(SessionId left, SessionId right) => !left.Equals(right);

  public override string ToString() => $"Session #{Number}";
}
