using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Services;

/// <summary>
/// Messenger for ViewModel↔ViewModel messages in the comet configuration domain.
/// Carries: <see cref="AetherVk.Logic.Messages.CometCommittedMessage"/>,
///          <see cref="AetherVk.Logic.Messages.CometDecommittedMessage"/>,
///          <see cref="AetherVk.Logic.Messages.NucleusRadiusKnownMessage"/>.
/// </summary>
public interface ICometMessenger : IMessenger { }

/// <summary>
/// Messenger for ViewModel↔ViewModel messages in the console/logging domain.
/// Carries: <see cref="AetherVk.Logic.Messages.ConsoleMessage"/>.
/// </summary>
public interface IConsoleMessenger : IMessenger { }

/// <summary>
/// Messenger for ViewModel↔ViewModel messages in the docking layout domain.
/// Carries: <see cref="AetherVk.Logic.Messages.TabDroppedMessage"/>,
///          <see cref="AetherVk.Logic.Messages.TabDragTaskMessage"/>,
///          <see cref="AetherVk.Logic.Messages.CoalesceGroupMessage"/>,
///          <see cref="AetherVk.Logic.Messages.DragCompletedMessage"/>.
/// </summary>
public interface ILayoutMessenger : IMessenger { }
