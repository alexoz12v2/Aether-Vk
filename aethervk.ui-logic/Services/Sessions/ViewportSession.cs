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
  // Placeholder — future fields might include:
  //   public ViewModels.CameraProjectionType ProjectionType { get; set; }
  //   public bool ShowGrid { get; set; } = true;
  //   public bool ShowBillboards { get; set; } = true;
}
