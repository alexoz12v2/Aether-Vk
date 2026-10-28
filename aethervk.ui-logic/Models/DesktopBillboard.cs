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

  /// <summary>Physical size: the image's plate scale, arcsec per image pixel (null: not set).</summary>
  public double? ImageArcsecPerPx { get; init; }

  /// <summary>
  /// Physical size: the image's vertical extent at the comet, km (null: not set). Typed by hand for
  /// an image without a WCS (a JPEG); it takes precedence over <see cref="ImageArcsecPerPx"/> once
  /// the comet's distance is known (<see cref="Utils.DesktopBillboardGeometry.ImageAngularHeightRad"/>).
  /// </summary>
  public double? ImageHeightKm { get; init; }

  /// <summary>The texture's native width, px (0: unknown until the texture is seen).</summary>
  public int ImageWidthPx { get; init; }

  /// <summary>The texture's native height, px (0: unknown).</summary>
  public int ImageHeightPx { get; init; }

  /// <summary>
  /// The size follows the Earth observer's viewport scale ("Apply physical size", "Match camera to
  /// image"): the billboard covers the sky the image covers (its height = the image's angular height
  /// over the viewport's angle per px), and re-scales on zoom and resize. A manual size edit unbinds.
  /// </summary>
  public bool PhysicalSizeBound { get; init; }

  /// <summary>
  /// FITS texture: the stretch the user applied (null: the import's zscale / linear). The engine
  /// does not persist it, so it is re-applied after a savefile load or a re-import. It belongs to
  /// the texture: applying it on one billboard sets it on every billboard of that texture.
  /// </summary>
  public FitsStretch? FitsStretch { get; init; }
}
