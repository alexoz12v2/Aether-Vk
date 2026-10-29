using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace AetherVk.Logic.SaveFile;

// The save index, format 1 (docs/savefile.md). Plain mutable DTOs: the JSON is the contract, the
// services map them to and from the live state. Epochs are TAI parts [centuries, nanoseconds].

/// <summary>A TAI epoch as engine parts, serialized as <c>[centuries, nanoseconds]</c>.</summary>
public sealed class EpochParts
{
  public short Centuries { get; set; }
  public ulong Nanoseconds { get; set; }

  public EpochParts() { }

  public EpochParts(short centuries, ulong nanoseconds)
  {
    Centuries = centuries;
    Nanoseconds = nanoseconds;
  }
}

public sealed class SaveIndex
{
  public const int CurrentFormatVersion = 1;
  public const string FileName = "aethervk_index.json";

  public int FormatVersion { get; set; } = CurrentFormatVersion;
  public string AppVersion { get; set; } = "";
  public DateTimeOffset CreatedAt { get; set; }
  public DateTimeOffset ModifiedAt { get; set; }
  public TimelineDto Timeline { get; set; } = new();
  public List<SpkDto> Spk { get; set; } = [];
  public List<StateDto> States { get; set; } = [];
  public ModelDto Model { get; set; } = new();
  public List<JetDto> Jets { get; set; } = [];
  public List<ViewportDto> Viewports { get; set; } = [];
  public List<AssetDto> Assets { get; set; } = [];
  public CometAppearanceIndexDto? CometAppearance { get; set; }
  public List<BillboardDto> Billboards { get; set; } = [];
}

public sealed class TimelineDto
{
  public EpochParts? CommittedStart { get; set; }
  public EpochParts? CommittedEnd { get; set; }
  public string? ProposedStart { get; set; }
  public string? ProposedEnd { get; set; }
  public EpochParts? CurrentEpoch { get; set; }
  /// <summary>Whether the clock is past the committed start. Informative: Open always restores
  /// paused at <see cref="CurrentEpoch"/>.</summary>
  public bool StartPaused { get; set; }
  /// <summary>The selected playback speed (<c>TimelineTabViewModel.SimulationSpeed</c> as int);
  /// absent in older saves and before one was chosen.</summary>
  public int? Speed { get; set; }
  /// <summary>Days the jets ignite before the committed start; absent in older saves (the default,
  /// one TTL: their jets ignited at the start, which left no tail until the end was reached).</summary>
  public double? DustLeadDays { get; set; }
}

public sealed class SpkDto
{
  public string File { get; set; } = "";
  public int NaifId { get; set; }
  public string Designation { get; set; } = "";
  public string FullName { get; set; } = "";
  public string RecordId { get; set; } = "";
  public CoverageDto? Coverage { get; set; }
  public SbdbDto? Sbdb { get; set; }
  public string ReferenceOrbitMode { get; set; } = "sbdb";
  public float NucleusRadiusKm { get; set; }
  public RotationDto? Rotation { get; set; }
  public DateTimeOffset CreatedAt { get; set; }
  public DateTimeOffset LastUsedAt { get; set; }
  public bool Committed { get; set; }
}

public sealed class CoverageDto
{
  public EpochParts Start { get; set; } = new();
  public EpochParts End { get; set; } = new();
}

public sealed class SbdbDto
{
  public int SpkId { get; set; }
  public string Designation { get; set; } = "";
  public string FullName { get; set; } = "";
  public string Kind { get; set; } = "";
  public string Prefix { get; set; } = "";
  public string OrbitClassName { get; set; } = "";
  public string OrbitClassCode { get; set; } = "";
  public bool IsNeo { get; set; }
  public bool IsPha { get; set; }
  public string OrbitId { get; set; } = "";
  public double? Tp { get; set; }
  public double? ElementsEpochJd { get; set; }
  public double? E { get; set; }
  public double? Q { get; set; }
  public double? I { get; set; }
  public double? Om { get; set; }
  public double? W { get; set; }
}

public sealed class RotationDto
{
  public double PoleRaDeg { get; set; }
  public double PoleDecDeg { get; set; } = 90;
  public double PrimeMeridianDeg { get; set; }
  public double PoleRaRateDegCen { get; set; }
  public double PoleDecRateDegCen { get; set; }
  public double RateDegDay { get; set; }
  public bool BodyFixedOrientation { get; set; } = true;
}

public sealed class StateDto
{
  public string CompatHash { get; set; } = "";
  public string CompatibilityJson { get; set; } = "";
  public string File { get; set; } = "";
  public uint DumpVersion { get; set; }
  public DateTimeOffset CreatedAt { get; set; }
  public DateTimeOffset ModifiedAt { get; set; }
  public DateTimeOffset LastUsedAt { get; set; }
  public bool Latest { get; set; }
}

/// <summary>
/// The Model tab. The grain and activity values (diameter … Q_d₀) are the defaults for new jets,
/// and what each jet of an older save (without its own, <see cref="JetDto"/>) inherits on load.
/// </summary>
public sealed class ModelDto
{
  public float DiametreUm { get; set; } = 100;
  public float DensityGCm3 { get; set; } = 0.533f;
  public float ScatteringEfficiency { get; set; } = 1;
  public float GeometricAlbedo { get; set; } = 0.04f;
  public float SizeQ { get; set; } = 3.5f;
  public float SizeRange { get; set; } = 10;
  // the dust production law: Q_d(r_h) = min(max, Q_d₀ · r_h^−k), 0 beyond the cutoff. Null only in
  // saves written before Q_d replaced Afρ: the legacy fields below are converted then
  [JsonPropertyName("dust_rate_1au_kgs")]
  public float? DustRate1AuKgs { get; set; }
  [JsonPropertyName("dust_rate_power")]
  public float? DustRatePower { get; set; }
  [JsonPropertyName("dust_rate_cutoff_au")]
  public float? DustRateCutoffAu { get; set; }
  [JsonPropertyName("dust_rate_max_kgs")]
  public float? DustRateMaxKgs { get; set; }

  // Obsolete: read only to convert pre-Q_d saves (Afρ at 1 AU in cm and its law); never written
  [JsonPropertyName("afrho0_cm"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
  public float? Afrho0Cm { get; set; }
  [JsonPropertyName("afrho_power"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
  public float? AfrhoPower { get; set; }
  [JsonPropertyName("afrho_cutoff_au"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
  public float? AfrhoCutoffAu { get; set; }
  [JsonPropertyName("afrho_max_value_cm"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
  public float? AfrhoMaxValueCm { get; set; }
  public double DustSoftening { get; set; } = 0.01;
  public bool DustTracers { get; set; }
  public bool DustFlow { get; set; }
  public double DustFlowSpeed { get; set; } = 1;
  public bool DustTrajectories { get; set; }
  public bool DustRhoNorm { get; set; }
  public double DustExposureEv { get; set; }
  public bool SphereGizmoVisible { get; set; } = true;
  public uint SphereGizmoParts { get; set; } = Models.SphereGizmoPart.Default;
  public bool ShowJetAngles { get; set; }
  public bool ShowReferencePositionError { get; set; }
  /// <summary>The Model tab's manual nucleus radius (used when Horizons has none).</summary>
  public float ManualNucleusRadiusKm { get; set; } = 2;
}

public sealed class JetDto
{
  public float LatitudeDeg { get; set; }
  public float LongitudeDeg { get; set; }
  public float ApertureDeg { get; set; }
  public float VelocityMeanMs { get; set; }
  public float VelocityStdMs { get; set; }
  public float[] StreamColor { get; set; } = [1, 1, 1, 1];
  public bool PreviewVisible { get; set; } = true;
  public uint Seed { get; set; }

  // the jet's own grain and activity; null in saves written before jets had their own: the jet
  // then inherits the model's value (ModelDto)
  public float? DiametreUm { get; set; }
  public float? DensityGCm3 { get; set; }
  public float? ScatteringEfficiency { get; set; }
  public float? GeometricAlbedo { get; set; }
  public float? SizeQ { get; set; }
  public float? SizeRange { get; set; }
  /// <summary>This jet's Q_d at 1 AU (kg/s); null in saves written before it (inherits or converts).</summary>
  [JsonPropertyName("dust_rate_1au_kgs")]
  public float? DustRate1AuKgs { get; set; }

  // Obsolete: read only to convert pre-Q_d saves (this jet's Afρ at 1 AU in cm); never written
  [JsonPropertyName("afrho0_cm"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
  public float? Afrho0Cm { get; set; }
}

/// <summary>
/// One viewport camera (CameraService's persisted state): the mode, the last confirmed pose and
/// projection, the comet-orbit offset and the Earth observer. Restoring lands on the saved pose.
/// </summary>
public sealed class ViewportDto
{
  /// <summary>earth_position | up_zenith | comet_orbiting.</summary>
  public string Mode { get; set; } = "up_zenith";
  public CameraSnapshotDto? Camera { get; set; }
  /// <summary>Camera offset from the comet in comet-orbiting mode, AU.</summary>
  public double[] OrbitOffsetAu { get; set; } = [0, 0, 0];
  public ObserverDto? Observer { get; set; }
  /// <summary>The settings panel's display choices; absent in older saves (defaults kept).</summary>
  public ViewportDisplayDto? Display { get; set; }
}

public sealed class ViewportDisplayDto
{
  /// <summary>au | km | m: the orthographic extents' and clip planes' unit.</summary>
  public string OrthoUnit { get; set; } = "au";
  /// <summary>deg | arcmin | arcsec: the perspective field of view's unit.</summary>
  public string FovUnit { get; set; } = "deg";
  public bool OrthoProportionsLocked { get; set; } = true;
  /// <summary>An orthographic view also draws what lies behind the camera (absent: on).</summary>
  public bool ShowBehindCamera { get; set; } = true;
}

public sealed class CameraSnapshotDto
{
  /// <summary>x, y, z (AU) then the rotation quaternion x, y, z, w.</summary>
  public double[] Transform { get; set; } = [0, 0, 0, 0, 0, 0, 1];
  public ProjectionDto? Projection { get; set; }
}

public sealed class ProjectionDto
{
  public bool IsPerspective { get; set; } = true;
  public float Fov { get; set; }
  public float Aspect { get; set; }
  public float Near { get; set; }
  public float Far { get; set; }
  public float Left { get; set; }
  public float Right { get; set; }
  public float Bottom { get; set; }
  public float Top { get; set; }
  public float FocusDistance { get; set; }
}

public sealed class ObserverDto
{
  public float LatDeg { get; set; }
  public float LonDeg { get; set; }
  public double ElevationM { get; set; }
  /// <summary>free | comet_tracking | sun_tracking (EarthObserverOrientationMode, snake_case). The
  /// retired comet_lock_in / sun_lock_in of older saves read as free, with the inertial look.</summary>
  public string Orientation { get; set; } = "free";
  /// <summary>Look quaternion x, y, z, w in the inertial frame.</summary>
  public double[] InertialLook { get; set; } = [0, 0, 0, 1];
  /// <summary>Retired lock-in look (Earth body-fixed): written as the inertial look, ignored on open.</summary>
  public double[] BodyFixedLook { get; set; } = [0, 0, 0, 1];
  /// <summary>Observer on its surface site (LatDeg / LonDeg / ElevationM) instead of the Earth's
  /// centre; false in older saves.</summary>
  public bool Topocentric { get; set; }
  /// <summary>The observer's projection remembered while the viewport is in another mode (absent:
  /// entering the observer starts from its defaults).</summary>
  public ProjectionDto? Projection { get; set; }
  /// <summary>The view's up: ecliptic | celestial_north (ObserverUp, snake_case). Older saves have
  /// none and read as ecliptic.</summary>
  public string Up { get; set; } = "ecliptic";
}

public sealed class AssetDto
{
  public string Key { get; set; } = "";
  public string Kind { get; set; } = "texture";
  public string Label { get; set; } = "";
  public string OriginalSource { get; set; } = "";
  public string StoredFile { get; set; } = "";
  public string ContentHash { get; set; } = "";
  public DateTimeOffset ImportedAt { get; set; }
  public string? ChannelHint { get; set; }
  public Dictionary<string, string?>? BundledTextures { get; set; }
}

public sealed class CometAppearanceIndexDto
{
  public string Mode { get; set; } = "default";
  public string? MeshKey { get; set; }
  public string? AlbedoKey { get; set; }
  public string? NormalKey { get; set; }
  public string? RoughnessKey { get; set; }
  public string? AoKey { get; set; }
  public float[] YawPitchRollDeg { get; set; } = [0, 0, 0];
  public float[] TranslationRadii { get; set; } = [0, 0, 0];
}

public sealed class BillboardDto
{
  public Guid Id { get; set; }
  public string TextureKey { get; set; } = "";
  public double CenterX { get; set; }
  public double CenterY { get; set; }
  public double WidthPx { get; set; }
  public double HeightPx { get; set; }
  public double Alpha { get; set; } = 0.6;
  public double RotationDeg { get; set; }
  public bool Visible { get; set; } = true;
  /// <summary>Physical size: arcsec per image pixel (absent: not set).</summary>
  public double? ImageArcsecPerPx { get; set; }
  /// <summary>Physical size: the image's height at the comet, km (absent / null: not set, older saves).</summary>
  public double? ImageHeightKm { get; set; }
  public int ImageWidthPx { get; set; }
  public int ImageHeightPx { get; set; }
  /// <summary>The size follows the Earth observer's viewport scale.</summary>
  public bool PhysicalSizeBound { get; set; }
  /// <summary>FITS texture stretch (absent / null: the import's zscale, linear).</summary>
  public FitsStretchDto? FitsStretch { get; set; }
  /// <summary>The nucleus pixel, image px (x right, y down, pixel centres at whole numbers);
  /// absent / null in either: the image centre (older saves).</summary>
  public double? ImageNucleusX { get; set; }
  public double? ImageNucleusY { get; set; }
}

/// <summary>A FITS billboard's stretch; enum values snake_case.</summary>
public sealed class FitsStretchDto
{
  /// <summary><c>z_scale</c> | <c>percentile</c> | <c>manual</c>.</summary>
  public string Mode { get; set; } = "z_scale";
  /// <summary><c>linear</c> | <c>log</c> | <c>asinh</c>.</summary>
  public string Transfer { get; set; } = "linear";
  /// <summary>Percentile: low %; manual: black (data units).</summary>
  public double P0 { get; set; }
  /// <summary>Percentile: high %; manual: white (data units).</summary>
  public double P1 { get; set; }
}

/// <summary>Source-generated (AOT-friendly) JSON metadata for the save index, snake_case keys.</summary>
[JsonSourceGenerationOptions(
  PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
  WriteIndented = true,
  DefaultIgnoreCondition = JsonIgnoreCondition.Never,
  Converters = [typeof(EpochPartsConverter)])]
[JsonSerializable(typeof(SaveIndex))]
public partial class SaveIndexJsonContext : JsonSerializerContext;
