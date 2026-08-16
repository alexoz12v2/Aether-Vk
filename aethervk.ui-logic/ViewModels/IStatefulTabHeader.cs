using System.Collections.Generic;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Non-generic interface implemented by <see cref="StatefulTabViewModelBase{TSession}"/>.
/// Used as the <c>x:DataType</c> for <c>CommonTabHeader</c> in Avalonia compiled bindings,
/// avoiding the open-generic type restriction.
/// Command types match the CommunityToolkit <c>[RelayCommand]</c> source-generator output:
/// parameterised methods generate <see cref="IRelayCommand{T}"/>, parameterless generate
/// <see cref="IRelayCommand"/>.
/// </summary>
public interface IStatefulTabHeader
{
  SessionId                SessionId             { get; }
  IReadOnlyList<SessionId> AvailableSessions     { get; }
  bool                     IsExclusiveSession    { get; }
  IRelayCommand<int>       SwitchSessionCommand  { get; }
  IRelayCommand<int>       DeleteSessionCommand  { get; }
  IRelayCommand            NewSessionCommand     { get; }
}
