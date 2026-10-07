using System.Collections.Immutable;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Models;

/// <summary>Kind of an imported asset. Mirrors Rust <c>asset_library::AssetKind</c>.</summary>
public enum AssetKind : uint
{
  Mesh = 1,
  Texture = 2,
}

/// <summary>
/// Texture channels supported by <c>physical_mesh2.frag</c> (bindings 0..3).
/// Mirrors Rust <c>asset_library::TextureChannel</c>.
/// </summary>
public enum TextureChannel
{
  Albedo = 0,
  Normal = 1,
  Roughness = 2,
  Ao = 3,
}

/// <summary>How the comet nucleus is displayed. Mirrors Rust <c>CometDisplayMode</c>.</summary>
public enum CometDisplayMode : uint
{
  /// <summary>Procedural UV sphere (pre-existing behaviour).</summary>
  Default = 0,

  /// <summary>Imported mesh, auto-scaled to the nucleus radius, with wired texture channels.</summary>
  Custom = 1,
}

/// <summary>
/// Winding repair applied natively when a mesh was imported (front faces must point outwards).
/// Mirrors Rust <c>asset_library::OrientationFix</c>.
/// </summary>
public enum OrientationFix : uint
{
  None = 0,

  /// <summary>Triangle winding disagreed with the vertex normals and was reversed.</summary>
  WindingFlipped = 1,

  /// <summary>Closed mesh was inside-out: winding reversed and normals negated.</summary>
  InsideOutFixed = 2,
}

/// <summary>Result of <c>SetCometAppearance</c>. Mirrors Rust <c>ffi_assets::AppearanceStatus</c>.</summary>
public enum CometAppearanceStatus
{
  /// <summary>Native library not loaded (design / headless mode).</summary>
  NotAvailable = -1,
  Queued = 0,

  /// <summary>Mode / mesh / texture wiring cannot change while the simulation plays.</summary>
  LockedWhileRunning = 1,
  UnknownAsset = 2,
  InvalidArgument = 3,
  QueueFull = 4,

  /// <summary>An import holds the asset library; retry after it completes.</summary>
  Busy = 5,
}

/// <summary>Tightly packed RGBA8 preview image produced natively at import.</summary>
public sealed record AssetThumbnail(int Width, int Height, byte[] Rgba);

/// <summary>
/// An asset of the native asset library (Imports tab). Meshes and textures are always separate
/// assets: textures bundled in an OBJ material or a glTF/GLB are imported as their own entries and
/// referenced through <see cref="BundledTextures"/>.
/// </summary>
/// <param name="Id">Native asset id (never 0).</param>
/// <param name="Key">Deduplication key: normalised source path, or <c>path#imageN</c> for images embedded in glTF.</param>
/// <param name="A">Texture: width. Mesh: vertex count.</param>
/// <param name="B">Texture: height. Mesh: index count.</param>
/// <param name="BundledTextures">Mesh only: texture id per <see cref="TextureChannel"/> (0 = none).</param>
public sealed record ImportedAsset(
  ulong Id,
  AssetKind Kind,
  string Label,
  string Key,
  ulong A,
  ulong B,
  ImmutableArray<ulong> BundledTextures,
  TextureChannel? ChannelHint,
  AssetThumbnail? Thumbnail
)
{
  /// <summary>Mesh only: winding repair applied at import.</summary>
  public OrientationFix OrientationFix { get; init; }

  /// <summary>Short secondary line for list items.</summary>
  public string Details =>
    Kind == AssetKind.Mesh
      ? $"{A:N0} vertices · {B / 3:N0} triangles"
        + (OrientationFix == OrientationFix.None ? "" : " · winding fixed")
      : $"{A} × {B}";

  public ulong BundledTexture(TextureChannel channel) =>
    BundledTextures.IsDefaultOrEmpty ? 0 : BundledTextures[(int)channel];
}

/// <summary>
/// Comet appearance exchanged with the native runtime. Blittable mirror of Rust
/// <c>ffi_assets::CCometAppearanceDTO</c> (72 bytes).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CometAppearanceDto
{
  /// <summary>Mesh asset id, 0 = none.</summary>
  public ulong Mesh;
  public ulong Albedo;
  public ulong Normal;
  public ulong Roughness;
  public ulong Ao;

  /// <summary>Intrinsic Z (yaw) → Y (pitch) → X (roll) rotation in degrees, comet body frame.</summary>
  public float Yaw;
  public float Pitch;
  public float Roll;

  /// <summary>Translation in units of nucleus radius, comet body frame.</summary>
  public float TranslationX;
  public float TranslationY;
  public float TranslationZ;
  public CometDisplayMode Mode;
  private uint _pad;

  public readonly ulong Texture(TextureChannel channel) =>
    channel switch
    {
      TextureChannel.Albedo => Albedo,
      TextureChannel.Normal => Normal,
      TextureChannel.Roughness => Roughness,
      _ => Ao,
    };

  /// <summary>Whether two appearances differ in mesh/texture wiring (not just placement).</summary>
  public readonly bool WiringDiffers(in CometAppearanceDto other) =>
    Mode != other.Mode
    || Mesh != other.Mesh
    || Albedo != other.Albedo
    || Normal != other.Normal
    || Roughness != other.Roughness
    || Ao != other.Ao;
}
