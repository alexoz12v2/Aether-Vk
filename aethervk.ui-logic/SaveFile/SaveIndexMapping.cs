using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Text;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.SaveFile;

/// <summary>
/// Pure conversions between the save index DTOs and the live types (sessions, camera state, jets,
/// billboards). Every pair round-trips exactly; enum values are written snake_case.
/// </summary>
public static class SaveIndexMapping
{
  // ── enums ──────────────────────────────────────────────────────────────────

  /// <summary><c>CometOrbiting</c> → <c>comet_orbiting</c>.</summary>
  public static string Snake<T>(T value) where T : struct, Enum
  {
    var name = value.ToString();
    var sb = new StringBuilder(name.Length + 4);
    for (int i = 0; i < name.Length; i++)
    {
      char c = name[i];
      if (char.IsUpper(c))
      {
        if (i > 0) sb.Append('_');
        sb.Append(char.ToLowerInvariant(c));
      }
      else sb.Append(c);
    }
    return sb.ToString();
  }

  /// <summary>Inverse of <see cref="Snake{T}"/>; <paramref name="fallback"/> for an unknown name.</summary>
  public static T ParseSnake<T>(string? text, T fallback) where T : struct, Enum
  {
    foreach (T v in Enum.GetValues(typeof(T)))
      if (string.Equals(Snake(v), text, StringComparison.Ordinal))
        return v;
    return fallback;
  }

  /// <summary>A content hash as the index writes it: 16 lowercase hex digits.</summary>
  public static string HashHex(ulong hash) => hash.ToString("x16", CultureInfo.InvariantCulture);

  public static ulong? ParseHashHex(string? hex) =>
    ulong.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var h) ? h : null;

  // ── epochs ─────────────────────────────────────────────────────────────────

  public static (EpochParts Start, EpochParts End) ToParts(TimeRange r) =>
    (new EpochParts(r.StartCenturies, r.StartNs), new EpochParts(r.EndCenturies, r.EndNs));

  public static TimeRange ToRange(EpochParts start, EpochParts end) =>
    new(start.Centuries, start.Nanoseconds, end.Centuries, end.Nanoseconds);

  // ── cameras ────────────────────────────────────────────────────────────────

  public static ViewportDto ToDto(CameraPersistedState s) => new()
  {
    Mode = Snake(s.Mode),
    Camera = s.Transform is null && s.Projection is null
      ? null
      : new CameraSnapshotDto
      {
        Transform = s.Transform is { } t ? [t.PosX, t.PosY, t.PosZ, t.RotX, t.RotY, t.RotZ, t.RotW] : [0, 0, 0, 0, 0, 0, 1],
        Projection = s.Projection is { } p
          ? new ProjectionDto
          {
            IsPerspective = p.IsPerspective, Fov = p.Fov, Aspect = p.Aspect, Near = p.Near, Far = p.Far,
            Left = p.Left, Right = p.Right, Bottom = p.Bottom, Top = p.Top, FocusDistance = p.FocusDistance,
          }
          : null,
      },
    OrbitOffsetAu = [s.OrbitOffsetAu.X, s.OrbitOffsetAu.Y, s.OrbitOffsetAu.Z],
    Observer = new ObserverDto
    {
      LatDeg = s.ObserverLatDeg,
      LonDeg = s.ObserverLonDeg,
      ElevationM = s.ObserverElevationM,
      Orientation = Snake(s.ObserverOrientation),
      InertialLook = [.. s.ObserverInertialLook],
      BodyFixedLook = [.. s.ObserverBodyFixedLook],
    },
  };

  public static CameraPersistedState FromDto(ViewportDto d)
  {
    CameraTransformState? transform = null;
    CameraProjectionState? projection = null;
    if (d.Camera is { } c)
    {
      if (c.Transform is { Length: 7 } t)
        transform = new CameraTransformState(t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
      if (c.Projection is { } p)
        projection = new CameraProjectionState(p.IsPerspective, p.Fov, p.Aspect, p.Near, p.Far,
          p.Left, p.Right, p.Bottom, p.Top, p.FocusDistance);
    }
    var o = d.Observer ?? new ObserverDto();
    var off = d.OrbitOffsetAu is { Length: 3 } a ? a : [0, 0, 0];
    return new CameraPersistedState(
      ParseSnake(d.Mode, CameraMode.UpZenith),
      transform,
      projection,
      new Vector3((float)off[0], (float)off[1], (float)off[2]),
      o.LatDeg,
      o.LonDeg,
      o.ElevationM,
      ParseSnake(o.Orientation, EarthObserverOrientationMode.Free),
      Quat(o.InertialLook),
      Quat(o.BodyFixedLook));
  }

  private static double[] Quat(double[]? q) => q is { Length: 4 } ? [.. q] : [0, 0, 0, 1];

  // ── model and jets ─────────────────────────────────────────────────────────

  public static ModelDto ToDto(ModelSession m) => new()
  {
    DiametreUm = m.DiametreUm,
    DensityGCm3 = m.DensityGCm3,
    ScatteringEfficiency = m.ScatteringEfficiency,
    GeometricAlbedo = m.GeometricAlbedo,
    SizeQ = m.SizeQ,
    SizeRange = m.SizeRange,
    Afrho0Cm = m.Afrho0Cm,
    AfrhoPower = m.AfrhoPower,
    AfrhoCutoffAu = m.AfrhoCutoffAu,
    AfrhoMaxValueCm = m.AfrhoMaxValueCm,
    DustSoftening = m.DustSoftening,
    DustTracers = m.DustTracers,
    DustFlow = m.DustFlow,
    DustFlowSpeed = m.DustFlowSpeed,
    ShowReferencePositionError = m.ShowReferencePositionError,
    ManualNucleusRadiusKm = m.ManualNucleusRadiusKm,
  };

  /// <summary>Writes <paramref name="d"/> into <paramref name="m"/> (jets are separate).</summary>
  public static void Apply(ModelDto d, ModelSession m)
  {
    m.DiametreUm = d.DiametreUm;
    m.DensityGCm3 = d.DensityGCm3;
    m.ScatteringEfficiency = d.ScatteringEfficiency;
    m.GeometricAlbedo = d.GeometricAlbedo;
    m.SizeQ = d.SizeQ;
    m.SizeRange = d.SizeRange;
    m.Afrho0Cm = d.Afrho0Cm;
    m.AfrhoPower = d.AfrhoPower;
    m.AfrhoCutoffAu = d.AfrhoCutoffAu;
    m.AfrhoMaxValueCm = d.AfrhoMaxValueCm;
    m.DustSoftening = d.DustSoftening;
    m.DustTracers = d.DustTracers;
    m.DustFlow = d.DustFlow;
    m.DustFlowSpeed = d.DustFlowSpeed;
    m.ShowReferencePositionError = d.ShowReferencePositionError;
    m.ManualNucleusRadiusKm = d.ManualNucleusRadiusKm;
  }

  public static JetDto ToDto(JetViewModel j) => new()
  {
    LatitudeDeg = j.LatitudeDeg,
    LongitudeDeg = j.LongitudeDeg,
    ApertureDeg = j.ApertureDeg,
    VelocityMeanMs = j.StartVelocityMeanMs,
    VelocityStdMs = j.StartVelocityStdMs,
    StreamColor = [j.StreamColor.X, j.StreamColor.Y, j.StreamColor.Z, j.StreamColor.W],
    PreviewVisible = j.IsPreviewVisible,
    Seed = j.Seed,
  };

  public static JetViewModel FromDto(JetDto d)
  {
    var c = d.StreamColor is { Length: 4 } s ? s : [1, 1, 1, 1];
    return new JetViewModel(d.LatitudeDeg, d.LongitudeDeg, d.ApertureDeg, d.VelocityMeanMs, d.VelocityStdMs,
      new Vector4(c[0], c[1], c[2], c[3]), d.PreviewVisible, d.Seed);
  }

  // ── comet ──────────────────────────────────────────────────────────────────

  public static SbdbDto ToDto(SmallBodyDataComponent s) => new()
  {
    SpkId = s.SpkId,
    Designation = s.Designation,
    FullName = s.FullName,
    Kind = s.Kind,
    Prefix = s.Prefix,
    OrbitClassName = s.OrbitClassName,
    OrbitClassCode = s.OrbitClassCode,
    IsNeo = s.IsNeo,
    IsPha = s.IsPha,
    OrbitId = s.OrbitId,
    Tp = double.IsNaN(s.Tp) ? null : s.Tp,
    ElementsEpochJd = double.IsNaN(s.ElementsEpochJd) ? null : s.ElementsEpochJd,
    E = s.E,
    Q = s.Q,
    I = s.I,
    Om = s.Om,
    W = s.W,
  };

  public static SmallBodyDataComponent FromDto(SbdbDto d) => new()
  {
    SpkId = d.SpkId,
    Designation = d.Designation,
    FullName = d.FullName,
    Kind = d.Kind,
    Prefix = d.Prefix,
    OrbitClassName = d.OrbitClassName,
    OrbitClassCode = d.OrbitClassCode,
    IsNeo = d.IsNeo,
    IsPha = d.IsPha,
    OrbitId = d.OrbitId,
    Tp = d.Tp ?? double.NaN,
    ElementsEpochJd = d.ElementsEpochJd ?? double.NaN,
    E = d.E ?? 0,
    Q = d.Q ?? 0,
    I = d.I ?? 0,
    Om = d.Om ?? 0,
    W = d.W ?? 0,
  };

  public static RotationDto RotationOf(CometSession s) => new()
  {
    PoleRaDeg = s.RotPoleRaDeg,
    PoleDecDeg = s.RotPoleDecDeg,
    PrimeMeridianDeg = s.RotPrimeMeridianDeg,
    PoleRaRateDegCen = s.RotPoleRaRateDegCen,
    PoleDecRateDegCen = s.RotPoleDecRateDegCen,
    RateDegDay = s.RotRateDegDay,
    BodyFixedOrientation = s.RotBodyFixedOrientationIndex == 0,
  };

  public static void Apply(RotationDto r, CometSession s)
  {
    s.RotPoleRaDeg = r.PoleRaDeg;
    s.RotPoleDecDeg = r.PoleDecDeg;
    s.RotPrimeMeridianDeg = r.PrimeMeridianDeg;
    s.RotPoleRaRateDegCen = r.PoleRaRateDegCen;
    s.RotPoleDecRateDegCen = r.PoleDecRateDegCen;
    s.RotRateDegDay = r.RateDegDay;
    s.RotBodyFixedOrientationIndex = r.BodyFixedOrientation ? 0 : 1;
  }

  public static BodyRotationalModelDto ToNative(RotationDto r) =>
    new(r.PoleRaDeg, r.PoleDecDeg, r.PrimeMeridianDeg, r.PoleRaRateDegCen, r.PoleDecRateDegCen, r.RateDegDay,
      r.BodyFixedOrientation);

  // ── billboards ─────────────────────────────────────────────────────────────

  public static BillboardDto ToDto(DesktopBillboard b) => new()
  {
    Id = b.Id,
    TextureKey = b.TextureKey,
    CenterX = b.CenterX,
    CenterY = b.CenterY,
    WidthPx = b.WidthPx,
    HeightPx = b.HeightPx,
    Alpha = b.Alpha,
    RotationDeg = b.RotationDeg,
    Visible = b.Visible,
  };

  public static DesktopBillboard FromDto(BillboardDto d) =>
    new(d.Id, d.TextureKey, d.CenterX, d.CenterY, d.WidthPx, d.HeightPx, d.Alpha, d.RotationDeg, d.Visible);

  // ── appearance ─────────────────────────────────────────────────────────────

  /// <summary>The appearance with asset ids named by key (<paramref name="keyOf"/>: id → key, null when unknown).</summary>
  public static CometAppearanceIndexDto ToDto(CometAppearanceDto a, Func<ulong, string?> keyOf) => new()
  {
    Mode = Snake(a.Mode),
    MeshKey = keyOf(a.Mesh),
    AlbedoKey = keyOf(a.Albedo),
    NormalKey = keyOf(a.Normal),
    RoughnessKey = keyOf(a.Roughness),
    AoKey = keyOf(a.Ao),
    YawPitchRollDeg = [a.Yaw, a.Pitch, a.Roll],
    TranslationRadii = [a.TranslationX, a.TranslationY, a.TranslationZ],
  };

  /// <summary>
  /// The native appearance for <paramref name="d"/>, ids found by key (<paramref name="idOf"/>: key →
  /// id, 0 when not loaded). A custom appearance whose mesh is missing falls back to default.
  /// </summary>
  public static CometAppearanceDto FromDto(CometAppearanceIndexDto d, Func<string?, ulong> idOf, out bool fellBack)
  {
    var ypr = d.YawPitchRollDeg is { Length: 3 } r ? r : [0, 0, 0];
    var tr = d.TranslationRadii is { Length: 3 } t ? t : [0, 0, 0];
    var a = new CometAppearanceDto
    {
      Mesh = idOf(d.MeshKey),
      Albedo = idOf(d.AlbedoKey),
      Normal = idOf(d.NormalKey),
      Roughness = idOf(d.RoughnessKey),
      Ao = idOf(d.AoKey),
      Yaw = ypr[0],
      Pitch = ypr[1],
      Roll = ypr[2],
      TranslationX = tr[0],
      TranslationY = tr[1],
      TranslationZ = tr[2],
      Mode = ParseSnake(d.Mode, CometDisplayMode.Default),
    };
    fellBack = a.Mode == CometDisplayMode.Custom && a.Mesh == 0;
    if (fellBack) a.Mode = CometDisplayMode.Default;
    return a;
  }
}
