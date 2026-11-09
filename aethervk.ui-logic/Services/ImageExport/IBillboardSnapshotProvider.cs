using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services.ImageExport;

/// <summary>
/// One desktop billboard as shown: its geometry on the desktop (<see cref="DesktopBillboard"/>:
/// centre, unrotated size, rotation and opacity, desktop physical px) and the pixels it shows
/// (<see cref="ImageWidth"/> × <see cref="ImageHeight"/> RGBA, row 0 = the image's top, the same
/// bitmap the billboard window shows).
/// </summary>
public sealed record BillboardSnapshot(DesktopBillboard Billboard, int ImageWidth, int ImageHeight, byte[] Rgba);

/// <summary>
/// What the image export composites over the viewport's frame: the 3D viewport's rectangle on the
/// desktop and the visible billboards with their pixels. The app's billboard windows and the
/// viewport frame share these numbers (<see cref="DesktopBillboardGeometry"/>), so the composite
/// matches the screen.
/// </summary>
public interface IBillboardSnapshotProvider
{
  /// <summary>The 3D viewport on the desktop (physical px); null until the app reported it.</summary>
  ViewportFrame? Viewport { get; }

  /// <summary>The visible billboards whose texture is loaded, in stacking order (last on top).</summary>
  IReadOnlyList<BillboardSnapshot> VisibleBillboards();
}

/// <summary>
/// The provider over the application's services: the billboard list of
/// <see cref="DesktopBillboardService"/>, the textures of <see cref="AssetLibraryService"/> and
/// their texels from the core (<see cref="INativeRuntimeService.CopyAssetTexels"/>, the bitmap the
/// billboard windows show).
/// </summary>
public sealed class DesktopBillboardSnapshotProvider(
  DesktopBillboardService billboards,
  AssetLibraryService assets,
  INativeRuntimeService runtime) : IBillboardSnapshotProvider
{
  public ViewportFrame? Viewport => billboards.ViewportFrame;

  public IReadOnlyList<BillboardSnapshot> VisibleBillboards()
  {
    var result = new List<BillboardSnapshot>();
    var texels = new Dictionary<ulong, AssetThumbnail?>();
    foreach (var b in billboards.Current.Where(b => b.Visible))
    {
      var texture = assets.CurrentTextures.FirstOrDefault(t => t.Key == b.TextureKey && t.IsDisplayable);
      if (texture is null)
        continue;
      if (!texels.TryGetValue(texture.Id, out var image))
        texels[texture.Id] = image = runtime.CopyAssetTexels(texture.Id);
      if (image is null || image.Width <= 0 || image.Height <= 0)
        continue;
      result.Add(new BillboardSnapshot(b, image.Width, image.Height, image.Rgba));
    }
    return result;
  }
}
