using System;

namespace AetherVk.Logic.Services;

/// <summary>
/// Marker interface for all tab session POCOs.
/// Session types are plain data bags mutated exclusively through
/// <see cref="ITabStateService{TSession}.UpdateSession"/> — they do not implement
/// INotifyPropertyChanged themselves.
/// </summary>
public interface ITabSession { }

/// <summary>
/// Apply to an <see cref="ITabSession"/> implementation to declare that at most one session
/// of this type may ever exist. Enforced by <see cref="TabStateService{TSession}"/> at runtime.
/// Use this for tabs that interact with a singleton native resource (e.g. the ECS scene, the Vulkan swapchain).
/// </summary>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
public sealed class ExclusiveSessionAttribute : Attribute { }
