using System;

namespace AetherVk.Logic.Models;

/// <summary>Gizmo shown on a desktop billboard; right click on the billboard cycles through them.</summary>
public enum BillboardGizmoMode
{
  None,
  Translate,
  Rotate,
  Scale,
}

/// <summary>
/// An imported texture shown in its own borderless, semi-transparent desktop window, to compare
/// the simulated dust trail with a real image. Geometry is in <b>desktop physical pixels</b> (the
/// single coordinate system all displays share, the one <c>Window.Position</c> uses).
/// </summary>
/// <param name="Id">Stable identity.</param>
/// <param name="TextureKey">Asset key of the texture (<c>ImportedAsset.Key</c>): stable across sessions and saves.</param>
/// <param name="CenterX">Centre of the image, desktop px.</param>
/// <param name="CenterY">Centre of the image, desktop px (y down).</param>
/// <param name="WidthPx">Image width before rotation, px.</param>
/// <param name="HeightPx">Image height before rotation, px.</param>
/// <param name="Alpha">Opacity 0..1.</param>
/// <param name="RotationDeg">Clockwise on screen, degrees.</param>
/// <param name="Visible">Shown (a window exists) or not.</param>
public sealed record DesktopBillboard(
  Guid Id,
  string TextureKey,
  double CenterX,
  double CenterY,
  double WidthPx,
  double HeightPx,
  double Alpha = 0.6,
  double RotationDeg = 0,
  bool Visible = true
)
{
  /// <summary>Gizmo currently shown (UI state, not persisted).</summary>
  public BillboardGizmoMode GizmoMode { get; init; } = BillboardGizmoMode.None;
}
