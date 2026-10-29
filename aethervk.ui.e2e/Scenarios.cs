namespace AetherVk.UI.E2E;

/// <summary>
/// A jet site; <see cref="SpeedMs"/> and <see cref="ApertureDeg"/> are left at the application's
/// random defaults when null. With a speed, its σ is <see cref="SpeedStdFraction"/> of it.
/// <see cref="DiametreUm"/> (grain diameter, µm) and <see cref="DustRate1AuKgs"/> (Q_d at 1 AU, kg/s) are
/// the jet's own grain and activity; null keeps the Model tab's defaults for new jets.
/// </summary>
public sealed record Jet(
  double LatitudeDeg,
  double LongitudeDeg,
  double? SpeedMs = null,
  double? ApertureDeg = null,
  double SpeedStdFraction = 0.2,
  double? DiametreUm = null,
  double? DustRate1AuKgs = null
);

/// <summary>
/// IAU rotational model of the nucleus, as the Comet tab "Rotational Model" takes it: pole RA / Dec
/// (deg, J2000), prime meridian W0 (deg) and the sidereal rate (deg/day).
/// </summary>
public sealed record Rotation(double PoleRaDeg, double PoleDecDeg, double PrimeMeridianDeg, double RateDegPerDay)
{
  public static Rotation FromPeriodHours(double hours, double poleRaDeg = 0, double poleDecDeg = 90,
    double primeMeridianDeg = 0) => new(poleRaDeg, poleDecDeg, primeMeridianDeg, 360.0 * 24.0 / hours);
}

/// <summary>Ground site of the observing telescope (geodetic, degrees, east positive).</summary>
public sealed record Site(string Name, double LatitudeDeg, double LongitudeDeg);

/// <summary>
/// One experiment of the thesis "Experiments" chapter: the comet, the observation instant, the
/// telescope site and the modelled jets. <see cref="Search"/> is typed in the Comet tab (SBDB
/// spells Pan-STARRS "PANSTARRS", so designations are safer than names).
/// </summary>
public sealed record Scenario(
  string Id,
  string Search,
  string Designation,
  DateTime ObservationUtc,
  Site Site,
  IReadOnlyList<Jet> Jets,
  // Orthographic half-width of the viewport (km): the framing of the thesis chapter 5 overlay
  // figures, read off their scale bars (bar 148 px of a 898 px frame: half-width = bar * 449/148)
  double OrthoHalfWidthKm,
  // The nucleus spin: with the Comet tab's default rate 0 the jet sites never turn
  Rotation Spin,
  // Days of simulated activity before the observation (the timeline starts here); the
  // Timeline tab refuses ranges shorter than 1 day
  double LeadDays = 28.0,
  // Used only when Horizons publishes no radius (jet creation needs one)
  double FallbackNucleusRadiusKm = 2.0,
  // A mesh of assets/ shown as the nucleus (Comet Appearance → Custom) before the run
  string? NucleusMesh = null,
  // Grain size range F (radii s₀/F … s₀·F about the reference grain); null keeps the app's 10
  double? GrainSizeRange = null,
  // Expected fan ridges of the settled observer view (checked when set)
  FanCheck? Fan = null)
{
  public override string ToString() => Id;
}

/// <summary>
/// Expected ridge position angles (deg, north up, east left) of the settled observer view: in the
/// inner ring (km) and the outer ring (km), each found within <see cref="TolDeg"/>. The ridges are
/// measured with <see cref="FanRidges"/> (ring-median normalised, like the processed observation).
/// <see cref="ObservedInner"/> / <see cref="ObservedOuter"/> are the observation's ridges at the
/// same rings (fanpa.py on the JPEG), logged beside the render's for the juxtaposition.
/// </summary>
public sealed record FanCheck(
  (double LoKm, double HiKm) InnerRing, double[] Inner,
  (double LoKm, double HiKm) OuterRing, double[] Outer,
  double TolDeg,
  double[] ObservedInner, double[] ObservedOuter,
  // where a ring is clipped (a flat top) its ridges are noise: the fan's half-level edges are
  // compared instead with the tangents of the emission's outer cone (90° − |φ| + aperture)
  double[] ConeEdges);

public static class Scenarios
{
  // Stazione Astronomica di Sozzago (IAU MPC A12) and Asiago Mount Ekar (IAU MPC 098)
  public static readonly Site Sozzago = new("Sozzago (A12)", 45.398, 8.722);
  public static readonly Site AsiagoEkar = new("Asiago Mt Ekar (098)", 45.849, 11.569);
  // Liverpool Telescope, Roque de los Muchachos, La Palma (IAU MPC 950)
  public static readonly Site LiverpoolTelescope = new("Liverpool Telescope (950)", 28.7624, -17.8792);

  // ASSUMED: no published period for C/2025 A6 nor C/2022 N2, a typical 12 h about the default pole
  private static readonly Rotation Typical12h = Rotation.FromPeriodHours(12.0);

  // ASSUMED: the thesis gives only "four on the 0° meridian, one at 90°, latitudes −40° to +60°".
  // The sunward "U" of the thesis figure is the fountain turn-back of the sunward jets at
  // v²/(2βg): ~5–8·10³ km at r = 0.83 AU (βg ≈ 1.9·10⁻⁴ m/s²) needs ~50 m/s, in narrow cones (the
  // 1.5–3.5 m/s defaults turn back within ~15 km, a single trail)
  private const double LemmonSpeedMs = 50, LemmonApertureDeg = 15;
  private static readonly Jet[] LemmonJets =
  [
    new(-40, 0, LemmonSpeedMs, LemmonApertureDeg), new(0, 0, LemmonSpeedMs, LemmonApertureDeg),
    new(30, 0, LemmonSpeedMs, LemmonApertureDeg), new(60, 0, LemmonSpeedMs, LemmonApertureDeg),
    new(20, 90, LemmonSpeedMs, LemmonApertureDeg),
  ];

  // Observation times are ASSUMED (the thesis gives dates only): evening twilight for the evening
  // comets, pre-dawn for Lovejoy (a morning object in December 2013).
  public static readonly Scenario Lemmon20251008 = new(
    "lemmon-2025-10-08", "C/2025 A6", "C/2025 A6",
    new DateTime(2025, 10, 8, 18, 30, 0, DateTimeKind.Utc), Sozzago, LemmonJets, 143_500, Typical12h); // bar 47301 km

  public static readonly Scenario Lemmon20251011 = new(
    "lemmon-2025-10-11", "C/2025 A6", "C/2025 A6",
    new DateTime(2025, 10, 11, 18, 30, 0, DateTimeKind.Utc), AsiagoEkar, LemmonJets, 82_400, Typical12h); // bar 27160 km

  public static readonly Scenario PanStarrs20240815 = new(
    "n2-2024-08-15", "C/2022 N2", "C/2022 N2",
    new DateTime(2024, 8, 15, 21, 0, 0, DateTimeKind.Utc), AsiagoEkar,
    // the broad sunward envelope of the figure turns back ~2.2·10⁴ km out (725 km/px): at
    // r = 3.9 AU (βg ≈ 8.4·10⁻⁶ m/s²) that is ~20 m/s, in wide cones. A grain reaches the apex
    // only after v/(βg) ≈ 27 d and is back at the nucleus after ~55 d: the envelope is months old
    [new(-40, 0, 20, 30), new(40, 0, 20, 30)], 217_400, Typical12h, LeadDays: 120); // bar 71649 km

  // The thesis text says 2023, its figure and the perihelion (2013-12-22) say 2013.
  // Site ASSUMED: the image is by Federico Manzini (Sozzago).
  public static readonly Scenario Lovejoy20131203 = new(
    "lovejoy-2013-12-03", "C/2013 R1", "C/2013 R1",
    new DateTime(2013, 12, 3, 4, 30, 0, DateTimeKind.Utc), Sozzago,
    // the sunward lobes of the figure turn back ~8·10³ km out (261 km/px): at r = 0.89 AU
    // (βg ≈ 1.6·10⁻⁴ m/s²) that is ~50 m/s
    [new(-62, 0, 50, 15), new(-10, 0, 50, 15), new(30, 0, 50, 15), new(10, 3, 50, 15)], 78_400,
    // Manzini et al., "The slow-rotating nucleus of comet C/2013 R1 (Lovejoy)" (Astron. Nachr.):
    // 47.8 ± 1.2 h, spin axis RA 332°, Dec +47° (W0 unknown: 0)
    Rotation.FromPeriodHours(47.8, poleRaDeg: 332, poleDecDeg: 47)); // bar 25856 km

  // "four active emission regions", three listed. Site ASSUMED (the reviewer's image).
  public static readonly Scenario Churyumov20220111 = new(
    "67p-2022-01-11", "67P", "67P",
    new DateTime(2022, 1, 11, 20, 0, 0, DateTimeKind.Utc), Sozzago,
    // the whitish sunward fan of the thesis figure reaches ~6·10⁴ km (312 km/px at this half-width):
    // at r = 1.48 AU (βg ≈ 5.8·10⁻⁵ m/s²) that is ~60 m/s, not the 1.5–3.5 m/s defaults
    // The two southern jets drive the two fans the figure shows, each a narrow wedge: a 10° cone
    // and a 5 % speed σ (σ·age is the cluster's own spread: 20 % of 60 m/s over 28 d blurred each
    // by ~3·10⁴ km into one round haze)
    [new(-52, 0, 60, 10, 0.05), new(-68, 45, 60, 10, 0.05), new(40, 90, 60, 20)], 187_400,
    // pole RA 69.54°, Dec 64.11° (IAU WGCCRE / Rosetta). The period shortened at each perihelion
    // (12.40 h before 2015, 12.055 h after, Rosetta/OSIRIS 2016; 2021 unmeasured): 12.055 h. The
    // phase W0 (IAU, epoch J2000) is meaningless after the spin-ups: kept, it fixes only where the
    // jets point at the observation, not the shape of the spiral
    new Rotation(69.54, 64.11, 114.69, 360.0 * 24.0 / 12.055), NucleusMesh: "Comet.glb"); // bar 61775 km

  // The user's Liverpool Telescope stack (25–28 Dec 2021, g+r+i+z, comet-aligned): the first jet's
  // fan opens ~72° below the nucleus (north up). That angle is the target, not an input: the jet
  // sites are the thesis chapter 5 ones (p. 49), the free parameters those of the Linux analysis
  // that matched the JPEG's ridges at the true scale (handoff §12): jet 1 at 70 ± 6 m/s, the others
  // 60 m/s, narrow cones, 718°/d (12.03 h) about the IAU pole. DATE-OBS 2021-12-27T00:14:47
  // "UT, averaged"; 1200 px × 0.3037″ at Δ = 0.4664 AU is 123 300 km: half-width 61 650 km.
  public static readonly Scenario Churyumov20211227 = new(
    "67p-2021-12-27", "67P", "67P",
    new DateTime(2021, 12, 27, 0, 14, 47, DateTimeKind.Utc), LiverpoolTelescope,
    // the thesis sites. The +40° jet's cone sweeps through north (edges PA 41°/260°), a plume the
    // image lacks: at 15 m/s radiation pressure turns it back within ~2000 km, into the tail.
    // F 30 adds the small, high-β grains of the anti-sun tail (PA 271°) about the same s₀
    [new(-52, 0, 70, 10, 6.0 / 70.0), new(-68, 45, 60, 10, 0.05), new(40, 90, 15, 20)], 61_650,
    new Rotation(69.54, 64.11, 114.69, 718.0), NucleusMesh: "Comet.glb", GrainSizeRange: 30);

  // One jet, the southern source of the 2021-12-27 fan, as a single-jet comparison with the FITS
  // and its JPEG. Latitude −54°: the user's recorded source (the 2021/22 literature places the
  // post-perihelion southern sources at −58 ± 5° and −53 ± 10°). Polar day at the epoch (sub-solar
  // latitude −44.7°), so the rotating cone sweeps a shell of half-angle 36° about the pole; seen at
  // i = 54° from the pole (projected pole PA 330.6°) its edges are at 150.6° ± Δ with
  // sin Δ = cos φ / sin i: Δ = 46.6°, edges 104° / 197°. The ridges sit inside the edges and drift
  // to larger PA outwards by radiation pressure; Monte-Carlo truth (handoff §10) 112/190 in
  // 2–5·10³ km, 117/205 in 28–38·10³ km. JPEG ridges at the same rings: 118–128/190–194, 134/194.
  // Near the nucleus the render clips, so there the fan's outer edges are checked: a cone of
  // half-aperture a = 10° fills the shell out to 90° − 54° + 10° = 46° from the pole, whose
  // tangents are sin Δ = sin 46° / sin 54°: Δ = 62.8°, edges 88° / 213°.
  public static readonly Scenario Churyumov20211227South = new(
    "67p-2021-12-27-south", "67P", "67P",
    new DateTime(2021, 12, 27, 0, 14, 47, DateTimeKind.Utc), LiverpoolTelescope,
    [new(-54, 0, 70, 10, 6.0 / 70.0)], 61_650,
    new Rotation(69.54, 64.11, 114.69, 718.0), NucleusMesh: "Comet.glb",
    Fan: new FanCheck((3_000, 8_000), [112, 190], (25_000, 35_000), [117, 205], 8,
      ObservedInner: [123, 192], ObservedOuter: [134, 194], ConeEdges: [88, 213]));

  public static TheoryData<string> All => new()
  {
    Lemmon20251008.Id, Lemmon20251011.Id, PanStarrs20240815.Id, Lovejoy20131203.Id, Churyumov20220111.Id,
    Churyumov20211227.Id, Churyumov20211227South.Id,
  };

  public static Scenario ById(string id) =>
    new[] { Lemmon20251008, Lemmon20251011, PanStarrs20240815, Lovejoy20131203, Churyumov20220111, Churyumov20211227,
      Churyumov20211227South }
      .Single(s => s.Id == id);
}
