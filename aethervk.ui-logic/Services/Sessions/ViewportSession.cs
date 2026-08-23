namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the UI-side state for the 3-D Viewport tab.
/// Exclusive because the engine provides a single Vulkan swapchain / presentation engine per run.
/// The native swapchain handle itself lives inside <c>Viewport3DViewModel</c> — this session
/// only carries UI preferences that survive tab close/reopen.
/// </summary>
[ExclusiveSession]
public sealed class ViewportSession : ITabSession
{
  /// <summary>
  /// Camera mode to restore when the viewport tab reopens.
  /// Defaults to <see cref="CameraMode.EarthPosition"/> — the safest starting view.
  /// </summary>
  public CameraMode ActiveCameraMode { get; set; } = CameraMode.EarthPosition;

  /// <summary>Camera projection type to restore on tab reopen.</summary>
  public ViewModels.CameraProjectionType ProjectionType { get; set; } =
    ViewModels.CameraProjectionType.Perspective;
}
