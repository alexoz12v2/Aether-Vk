namespace AetherVk.Logic.Services;

/// <summary>
/// Holds the form data for the Comet tab.
/// Decorated with <see cref="ExclusiveSessionAttribute"/> because the engine scene
/// contains exactly one comet nucleus.
/// </summary>
[ExclusiveSession]
public sealed class CometSession : ITabSession
{
  // ── Committed (applied to native runtime) ─────────────────────────────────

  /// <summary>NAIF SPK id of the committed comet (e.g. 1000012 for 67P).</summary>
  public int? SpkId { get; set; }

  /// <summary>Short designation (e.g. "67P") of the committed comet.</summary>
  public string CommittedDesignation { get; set; } = string.Empty;

  /// <summary>Full name (e.g. "67P/Churyumov-Gerasimenko") of the committed comet.</summary>
  public string CommittedFullName { get; set; } = string.Empty;

  /// <summary>Absolute path to the committed SPK file on disk, or null if not yet loaded.</summary>
  public string? CommittedSpkFilePath { get; set; }

  /// <summary>Whether the native runtime has the SPK file loaded and AlmanacPlanet is attached.</summary>
  public bool IsAlmanacLoaded { get; set; }

  // ── Proposed (UI editing, not yet committed) ──────────────────────────────

  /// <summary>Short designation chosen in the UI but not yet committed.</summary>
  public string ProposedDesignation { get; set; } = string.Empty;

  /// <summary>Path to the downloaded but not-yet-loaded SPK file, or null.</summary>
  public string? ProposedSpkFilePath { get; set; }

  // ── Rotational model (live, reactive) ─────────────────────────────────────

  /// <summary>Right ascension of the rotation pole at J2000 (degrees).</summary>
  public double RotPoleRaDeg { get; set; }

  /// <summary>Declination of the rotation pole at J2000 (degrees). Defaults to north pole.</summary>
  public double RotPoleDecDeg { get; set; } = 90.0;

  /// <summary>Prime meridian angle at J2000 (degrees).</summary>
  public double RotPrimeMeridianDeg { get; set; }

  /// <summary>Rate of change of pole RA per century (degrees/century).</summary>
  public double RotPoleRaRateDegCen { get; set; }

  /// <summary>Rate of change of pole Dec per century (degrees/century).</summary>
  public double RotPoleDecRateDegCen { get; set; }

  /// <summary>Sidereal rotation rate (degrees/day).</summary>
  public double RotRateDegDay { get; set; }

  // ── Nucleus radius ────────────────────────────────────────────────────────

  /// <summary>
  /// Bounding-sphere radius of the committed comet nucleus in km.
  /// Populated from <c>PlanetOrbitData.CometRadiusKm</c> during <c>DownloadAndCommitAsync</c>.
  /// Can also be set manually via the Model tab when no Horizon data is available.
  /// Zero means "unknown" — <c>AddJetCommand</c> is disabled until this is &gt; 0.
  /// </summary>
  public float NucleusRadiusKm { get; set; } = 0f;
}

