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
        Projection = s.Projection is { } p ? ToDto(p) : null,
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
      Topocentric = s.ObserverTopocentric,
      Projection = s.ObserverProjection is { } op ? ToDto(op) : null,
      Up = Snake(s.ObserverUp),
    },
    Display = s.Display is { } d
      ? new ViewportDisplayDto
      {
        OrthoUnit = UnitName(OrthoUnits, d.OrthoUnitIndex),
        FovUnit = UnitName(FovUnits, d.FovUnitIndex),
        OrthoProportionsLocked = d.OrthoProportionsLocked,
        ShowBehindCamera = d.ShowBehindCamera,
      }
      : null,
  };

  private static readonly string[] OrthoUnits = ["au", "km", "m"];
  private static readonly string[] FovUnits = ["deg", "arcmin", "arcsec"];

  private static string UnitName(string[] names, int index) =>
    index >= 0 && index < names.Length ? names[index] : names[0];

  private static int UnitIndex(string[] names, string? name) =>
    Math.Max(0, Array.IndexOf(names, name));

  public static CameraPersistedState FromDto(ViewportDto d)
  {
    CameraTransformState? transform = null;
    CameraProjectionState? projection = null;
    if (d.Camera is { } c)
    {
      if (c.Transform is { Length: 7 } t)
        transform = new CameraTransformState(t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
      if (c.Projection is { } p)
        projection = FromDto(p);
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
      Quat(o.BodyFixedLook),
      d.Display is { } disp
        ? new ViewportDisplayState(UnitIndex(OrthoUnits, disp.OrthoUnit), UnitIndex(FovUnits, disp.FovUnit),
          disp.OrthoProportionsLocked, disp.ShowBehindCamera)
        : null,
      o.Topocentric,
      o.Projection is { } op ? FromDto(op) : null,
      ParseSnake(o.Up, ObserverUp.Ecliptic));
  }

  private static ProjectionDto ToDto(CameraProjectionState p) => new()
  {
    IsPerspective = p.IsPerspective, Fov = p.Fov, Aspect = p.Aspect, Near = p.Near, Far = p.Far,
    Left = p.Left, Right = p.Right, Bottom = p.Bottom, Top = p.Top, FocusDistance = p.FocusDistance,
  };

  private static CameraProjectionState FromDto(ProjectionDto p) =>
    new(p.IsPerspective, p.Fov, p.Aspect, p.Near, p.Far, p.Left, p.Right, p.Bottom, p.Top, p.FocusDistance);

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
    DustRate1AuKgs = m.DustRate1AuKgs,
    DustRatePower = m.DustRatePower,
    DustRateCutoffAu = m.DustRateCutoffAu,
    DustRateMaxKgs = m.DustRateMaxKgs,
    DustSoftening = m.DustSoftening,
    DustTracers = m.DustTracers,
    DustFlow = m.DustFlow,
    DustFlowSpeed = m.DustFlowSpeed,
    DustTrajectories = m.DustTrajectories,
    DustRhoNorm = m.DustRhoNorm,
    DustExposureEv = m.DustExposureEv,
    SphereGizmoVisible = m.SphereGizmoVisible,
    SphereGizmoParts = m.SphereGizmoParts,
    ShowJetAngles = m.ShowJetAngles,
    ShowReferencePositionError = m.ShowReferencePositionError,
    ManualNucleusRadiusKm = m.ManualNucleusRadiusKm,
  };

  /// <summary>
  /// The new-jet speed a pre-Q_d save's model default Afρ₀ is converted with: the centre of the
  /// random draw a new jet starts from (2.5 ± 0.625 m/s).
  /// </summary>
  internal const float LegacyModelVelocityMeanMs = 2.5f;
  internal const float LegacyModelVelocityStdMs = 0.625f;

  /// <summary>
  /// The model's default Q_d₀ (kg/s): the save's own, else (a pre-Q_d save) its default Afρ₀
  /// converted with the model grain and the new-jet speed, else the built-in default.
  /// </summary>
  public static float DustRate1AuOf(ModelDto d)
  {
    if (d.DustRate1AuKgs is { } q)
      return q;
    if (d.Afrho0Cm is { } afrho)
      return (float)LegacyAfrho.DustRateKgs(afrho, d.DiametreUm, d.DensityGCm3,
        LegacyModelVelocityMeanMs, LegacyModelVelocityStdMs, d.GeometricAlbedo, d.SizeRange);
    return JetGrain.Default.DustRate1AuKgs;
  }

  /// <summary>
  /// The model's Q_d clamp (kg/s): the save's own, else (a pre-Q_d save) the Afρ clamp scaled as
  /// the default Q_d₀ to its Afρ₀ (Q ∝ Afρ for a given grain), else the built-in default.
  /// </summary>
  public static float DustRateMaxOf(ModelDto d)
  {
    if (d.DustRateMaxKgs is { } q)
      return q;
    if (d.AfrhoMaxValueCm is { } maxCm && d.Afrho0Cm is { } afrho && afrho > 0)
      return (float)(DustRate1AuOf(d) * ((double)maxCm / afrho));
    return new ModelSession().DustRateMaxKgs;
  }

  /// <summary>
  /// Writes <paramref name="d"/> into <paramref name="m"/> (jets are separate). A pre-Q_d save's
  /// Afρ law is converted (<see cref="DustRate1AuOf"/>, <see cref="DustRateMaxOf"/>).
  /// </summary>
  public static void Apply(ModelDto d, ModelSession m)
  {
    m.DiametreUm = d.DiametreUm;
    m.DensityGCm3 = d.DensityGCm3;
    m.ScatteringEfficiency = d.ScatteringEfficiency;
    m.GeometricAlbedo = d.GeometricAlbedo;
    m.SizeQ = d.SizeQ;
    m.SizeRange = d.SizeRange;
    m.DustRate1AuKgs = DustRate1AuOf(d);
    m.DustRatePower = d.DustRatePower ?? d.AfrhoPower ?? 2f;
    m.DustRateCutoffAu = d.DustRateCutoffAu ?? d.AfrhoCutoffAu ?? 15f;
    m.DustRateMaxKgs = DustRateMaxOf(d);
    m.DustSoftening = d.DustSoftening;
    m.DustTracers = d.DustTracers;
    m.DustFlow = d.DustFlow;
    m.DustFlowSpeed = d.DustFlowSpeed;
    m.DustTrajectories = d.DustTrajectories;
    m.DustRhoNorm = d.DustRhoNorm;
    m.DustExposureEv = d.DustExposureEv;
    m.SphereGizmoVisible = d.SphereGizmoVisible;
    m.SphereGizmoParts = d.SphereGizmoParts;
    m.ShowJetAngles = d.ShowJetAngles;
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
    DiametreUm = j.DiametreUm,
    DensityGCm3 = j.DensityGCm3,
    ScatteringEfficiency = j.ScatteringEfficiency,
    GeometricAlbedo = j.GeometricAlbedo,
    SizeQ = j.SizeQ,
    SizeRange = j.SizeRange,
    DustRate1AuKgs = j.DustRate1AuKgs,
  };

  /// <summary>
  /// The jet of <paramref name="d"/>; a grain or activity value the save lacks (written before
  /// jets had their own) comes from <paramref name="modelGrain"/>, the model's value of that save
  /// (the built-in defaults when null). A pre-Q_d jet (its own Afρ₀, no Q_d₀) gets the Q_d₀ the
  /// old engine derived from that Afρ₀ with the jet's grain and speeds (<see cref="LegacyAfrho"/>).
  /// </summary>
  public static JetViewModel FromDto(JetDto d, JetGrain? modelGrain = null)
  {
    var c = d.StreamColor is { Length: 4 } s ? s : [1, 1, 1, 1];
    var m = modelGrain ?? JetGrain.Default;
    float diametreUm = d.DiametreUm ?? m.DiametreUm;
    float densityGCm3 = d.DensityGCm3 ?? m.DensityGCm3;
    float albedo = d.GeometricAlbedo ?? m.GeometricAlbedo;
    float sizeRange = d.SizeRange ?? m.SizeRange;
    float rate = d.DustRate1AuKgs ?? (d.Afrho0Cm is { } afrho
      ? (float)LegacyAfrho.DustRateKgs(afrho, diametreUm, densityGCm3, d.VelocityMeanMs, d.VelocityStdMs, albedo, sizeRange)
      : m.DustRate1AuKgs);
    var grain = new JetGrain(
      diametreUm,
      densityGCm3,
      d.ScatteringEfficiency ?? m.ScatteringEfficiency,
      albedo,
      d.SizeQ ?? m.SizeQ,
      sizeRange,
      rate);
    return new JetViewModel(d.LatitudeDeg, d.LongitudeDeg, d.ApertureDeg, d.VelocityMeanMs, d.VelocityStdMs,
      new Vector4(c[0], c[1], c[2], c[3]), d.PreviewVisible, d.Seed, grain);
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
    ImageArcsecPerPx = b.ImageArcsecPerPx,
    ImageHeightKm = b.ImageHeightKm,
    ImageWidthPx = b.ImageWidthPx,
    ImageHeightPx = b.ImageHeightPx,
    PhysicalSizeBound = b.PhysicalSizeBound,
    FitsStretch = b.FitsStretch is { } s && !s.SameAs(FitsStretch.Default)
      ? new FitsStretchDto { Mode = Snake(s.Mode), Transfer = Snake(s.Transfer), P0 = s.P0, P1 = s.P1 }
      : null,
    ImageNucleusX = b.ImageNucleusPx?.X,
    ImageNucleusY = b.ImageNucleusPx?.Y,
  };

  public static DesktopBillboard FromDto(BillboardDto d) =>
    new(d.Id, d.TextureKey, d.CenterX, d.CenterY, d.WidthPx, d.HeightPx, d.Alpha, d.RotationDeg, d.Visible)
    {
      ImageArcsecPerPx = d.ImageArcsecPerPx,
      ImageHeightKm = d.ImageHeightKm,
      ImageWidthPx = d.ImageWidthPx,
      ImageHeightPx = d.ImageHeightPx,
      PhysicalSizeBound = d.PhysicalSizeBound,
      FitsStretch = d.FitsStretch is { } s
        ? new FitsStretch(
          ParseSnake(s.Mode, FitsStretchMode.ZScale), ParseSnake(s.Transfer, FitsTransfer.Linear), s.P0, s.P1)
        : null,
      ImageNucleusPx = d.ImageNucleusX is { } nx && d.ImageNucleusY is { } ny ? new Utils.PointD(nx, ny) : null,
    };

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
