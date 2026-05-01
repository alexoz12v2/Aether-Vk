using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Xunit;
using Xunit.Abstractions;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Educational tests for understanding the JPL (Jet Propulsion Laboratory) Horizons and SBDB APIs.
/// These tests explain the concepts of celestial mechanics, time scales, and coordinate systems
/// used by NASA to track small bodies like comets and asteroids.
/// </summary>
public class HorizonJplEducationalTests
{
  private readonly ITestOutputHelper _output;
  private readonly HorizonJplService _service;

  public HorizonJplEducationalTests(ITestOutputHelper output)
  {
    _output = output;
    // Mocking services for testing purposes
    var console = new ConsoleService();
    var breadcrumb = new BreadcrumbService();
    _service = new HorizonJplService(console, breadcrumb);
  }

  /// <summary>
  /// 0) Querying a list of comets using the Small-Body Database (SBDB) Query API.
  /// The SBDB is optimized for searching many objects at once based on filters.
  /// </summary>
  [Fact]
  public async Task Step0_QueryCometList()
  {
    _output.WriteLine("--- Step 0: Querying Comet List ---");

    // sb-kind=c filters for comets.
    // fields defines what data we want for each comet.
    // spkid is the unique identifier used by Horizons for ephemeris generation.
    // startTime and stopTime can filter by 'first_obs' (discovery date).

    // Example URL format:
    // https://ssd-api.jpl.nasa.gov/sbdb_query.api?sb-kind=c&fields=full_name,first_obs,spkid

    // Mocking the SBDB API JSON response
    string mockJsonResponse =
      @"
        {
            ""fields"": [""full_name"", ""first_obs"", ""spkid""],
            ""data"": [
                [""C/2023 A1 (Tsuchinshan-ATLAS)"", ""2023-01-09"", ""1000001""],
                [""P/2023 V3 (PanSTARRS)"", ""2023-11-03"", ""1000002""]
            ]
        }";

    _service.ParseCometsJson(mockJsonResponse);

    // Verify headers were parsed correctly
    Assert.Contains("spkid", _service.CometsHeaders);
    Assert.Equal(2, _service.CometsData.Count);

    foreach (var comet in _service.CometsData)
    {
      _output.WriteLine($"Comet: {comet[0]}, Discovered: {comet[1]}, SPK-ID: {comet[2]}");
    }

    /*
     * EDUCATIONAL NOTE:
     * The SPK-ID (Spacecraft and Planet Kernel ID) is CRITICAL.
     * Names like 'Tsuchinshan-ATLAS' can be ambiguous or change, but the SPK-ID
     * (e.g., '1000001') is the permanent numeric key for this object in the NASA system.
     */
  }

  /// <summary>
  /// 1) Querying 'static' data about a specific comet.
  /// This includes physical parameters (diameter, magnitude) and orbital elements (eccentricity, etc).
  /// </summary>
  [Fact]
  public void Step1_QueryStaticData()
  {
    _output.WriteLine("--- Step 1: Querying Static Data ---");

    /*
     * When you query the Horizons API (not SBDB), the text response contains
     * a 'Target body name' section with physical and orbital data.
     *
     * Common physical parameters:
     * - H: Absolute magnitude (intrinsic brightness).
     * - G: Slope parameter (how brightness changes with phase angle).
     * - radius: Radius of the object (often km).
     *
     * Common orbital elements (Keplerian):
     * - e: Eccentricity (0 = circle, <1 = ellipse, 1 = parabola, >1 = hyperbola).
     * - q: Perihelion distance (closest approach to Sun, in AU).
     * - i: Inclination (angle relative to the ecliptic plane).
     */

    string mockHorizonsText =
      @"
*******************************************************************************
JPL/HORIZONS                 12P/Pons-Brooks                 2026-Apr-21
...
Target body name: 12P/Pons-Brooks (ID: 90000033)
...
Physical Data:
  radius, km   =  17.0            magnitude, H =  5.0
  slope, G     =  0.15
...
Orbital Elements (ICRF):
  e = 0.9545,  q = 0.781 AU,  i = 74.19 deg
*******************************************************************************
$$SOE
... ephemeris data ...
$$EOE";

    _output.WriteLine("Static data is usually manually extracted from the header text block.");
    _output.WriteLine("Key parameters for rendering 3D orbits:");
    _output.WriteLine(
      "- Absolute Magnitude (H): Crucial for calculating apparent brightness in the shader."
    );
    _output.WriteLine("- Radius: Defines the 'point' size or the scale of the billboard.");
  }

  /// <summary>
  /// 2) Understanding Time Scales: UTC vs TDB (Ephemeris Time).
  /// 3) Concept of Observer and Origin (SSB).
  /// </summary>
  [Fact]
  public void Step2_3_TimeScalesAndOrigins()
  {
    _output.WriteLine("--- Step 2 & 3: Time and Origins ---");

    /*
     * TIME SCALES:
     * - UTC (Coordinated Universal Time): What we use on Earth. It has leap seconds to stay
     *   aligned with Earth's rotation.
     * - TDB (Barycentric Dynamical Time): A uniform time scale used for Solar System
     *   ephemeris calculations. It avoids the discontinuities of leap seconds.
     *
     * RELATIONSHIP:
     * TDB (or ET) = UTC + DeltaT
     * As of 2024, DeltaT is approximately 69 seconds. Horizons usually expects UTC
     * in the START/STOP parameters but performs internal calculations in TDB.
     *
     * ORIGINS (CENTER parameter):
     * The 'CENTER' parameter defines the (0,0,0) point of your coordinate system.
     * - '500@0'  -> Solar System Barycenter (SSB). This is the 'true' center of mass
     *               of the entire solar system. Preferred for global simulations.
     * - '500@10' -> Sun Center. Coordinates will be 'Heliocentric'.
     * - '500@399'-> Earth Center. Coordinates will be 'Geocentric'.
     * - 'coord@399' -> Topocentric. A specific location on Earth's surface.
     */

    _output.WriteLine("When building a 3D engine (like AetherVk):");
    _output.WriteLine("- Always use SSB ('500@0') as the world origin for maximum precision.");
    _output.WriteLine("- Results will be in ICRF (International Celestial Reference Frame),");
    _output.WriteLine("  which is the standard inertial frame for the Solar System.");
  }

  /// <summary>
  /// 4) Querying positional data for a range and understanding interpolation.
  /// </summary>
  [Fact]
  public void Step4_PositionalDataRange()
  {
    _output.WriteLine("--- Step 4: Positional Data Range ---");

    /*
     * IS IT CHEBYSHEV POLYNOMIALS?
     *
     * - Horizons API: The REST API returns a table of DISCRETE points (X, Y, Z at time T).
     * - SPK Kernels (.bsp files): These are binary files that NASA actually uses.
     *   They store orbital paths as coefficients for 'Chebyshev Polynomials'.
     *   This allows calculating the EXACT position at ANY microsecond without
     *   storing every single point.
     *
     * In a web-based or lightweight API client (like this service):
     * 1. You query discrete points for a range (e.g., every 1 day).
     * 2. You use interpolation (like Catmull-Rom or Splines) in your app
     *    to get smooth motion between those points.
     * 3. Alternatively, if you need extreme precision, you would use a library
     *    (like 'Anise' in this project) to evaluate the Chebyshev polynomials
     *    directly from an SPK kernel file.
     */

    // Example of a VECTORS response (CSV format) from Horizons
    string mockVectorsResponse =
      @"
Date__(UT)__HR:MN, , X, Y, Z, VX, VY, VZ,
*******************************************************************************
$$SOE
2024-Jan-01 00:00, A, 1.1E+08, 5.2E+07, -1.3E+07, 21.0, -5.2, 1.2
2024-Jan-02 00:00, A, 1.12E+08, 5.15E+07, -1.29E+07, 20.8, -5.3, 1.15
$$EOE
*******************************************************************************";

    _service.ParseText(mockVectorsResponse);

    _output.WriteLine($"Parsed {_service.SessionData.Count} state vectors.");
    if (_service.SessionData.Count > 0)
    {
      var firstRow = _service.SessionData[0];
      _output.WriteLine($"Time: {firstRow[0]} (UTC)");
      _output.WriteLine($"Position X: {firstRow[2]} km (SSB-relative)");
      _output.WriteLine($"Velocity VX: {firstRow[5]} km/s");
    }
  }
}
