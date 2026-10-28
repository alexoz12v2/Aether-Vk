namespace AetherVk.UI.E2E;

/// <summary>
/// A jet site; <see cref="SpeedMs"/> and <see cref="ApertureDeg"/> are left at the application's
/// random defaults when null. With a speed, its σ is <see cref="SpeedStdFraction"/> of it.
/// </summary>
public sealed record Jet(
  double LatitudeDeg,
  double LongitudeDeg,
  double? SpeedMs = null,
  double? ApertureDeg = null,
  double SpeedStdFraction = 0.2
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
  // Timeline tab refuses ranges shorter than 28 days
  double LeadDays = 28.0,
  // Used only when Horizons publishes no radius (jet creation needs one)
  double FallbackNucleusRadiusKm = 2.0,
  // A mesh of assets/ shown as the nucleus (Comet Appearance → Custom) before the run
  string? NucleusMesh = null)
{
  public override string ToString() => Id;
}

public static class Scenarios
{
  // Stazione Astronomica di Sozzago (IAU MPC A12) and Asiago Mount Ekar (IAU MPC 098)
  public static readonly Site Sozzago = new("Sozzago (A12)", 45.398, 8.722);
  public static readonly Site AsiagoEkar = new("Asiago Mt Ekar (098)", 45.849, 11.569);

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

  // "four active emission regions", three listed. Site ASSUMED (image by Virginio Oldani).
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

  public static TheoryData<string> All => new()
  {
    Lemmon20251008.Id, Lemmon20251011.Id, PanStarrs20240815.Id, Lovejoy20131203.Id, Churyumov20220111.Id,
  };

  public static Scenario ById(string id) =>
    new[] { Lemmon20251008, Lemmon20251011, PanStarrs20240815, Lovejoy20131203, Churyumov20220111 }
      .Single(s => s.Id == id);
}
