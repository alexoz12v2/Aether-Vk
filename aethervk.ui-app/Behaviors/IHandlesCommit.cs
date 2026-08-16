namespace AetherVk.Behaviors;

/// <summary>
/// Marker interface: signals that this behavior fully owns text-commit for the
/// associated control. When an <see cref="UnboundedSlider"/> detects a behavior
/// implementing this interface, it suppresses its own parse-and-set logic so the
/// behavior is the single source of truth for value commits.
/// </summary>
internal interface IHandlesCommit { }
