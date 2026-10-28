using System;

namespace AetherVk.Logic.Utils;

/// <summary>
/// Sky maths of telescope images (FITS billboards): ICRS RA / Dec → the engine's world frame, the
/// TDB epoch parts, and the on-screen rotation of an image in the Earth observer view.
/// </summary>
public static class SkyMath
{
  /// <summary>
  /// J2000 equator → ecliptic J2000 angle (ANISE / SPICE <c>ECLIPJ2000</c>, IAU 1976 84381.448″),
  /// the rotation the engine's world axes use.
  /// </summary>
  public const double ObliquityRad = 0.40909280422232897;

  /// <summary>Nanoseconds per Julian century (the <c>hifitime</c> <c>Duration</c> parts).</summary>
  public const long NanosecondsPerCentury = 36525L * 86400L * 1_000_000_000L;

  private const double DegToRad = Math.PI / 180.0;

  /// <summary>World (ecliptic J2000) unit direction of an ICRS right ascension / declination.</summary>
  public static Vector3d EclipticDirection(double raDeg, double decDeg)
  {
    double ra = raDeg * DegToRad, dec = decDeg * DegToRad;
    double x = Math.Cos(dec) * Math.Cos(ra), y = Math.Cos(dec) * Math.Sin(ra), z = Math.Sin(dec);
    double s = Math.Sin(ObliquityRad), c = Math.Cos(ObliquityRad);
    return new Vector3d(x, c * y + s * z, -s * y + c * z);
  }

  /// <summary>
  /// TDB microseconds since J2000 TDB → the <c>(centuries, nanoseconds)</c> parts a seek takes
  /// (nanoseconds in [0, one century)).
  /// </summary>
  public static (short Centuries, ulong Nanoseconds) TdbMicrosToParts(long tdbUs)
  {
    long ns = checked(tdbUs * 1000L);
    long centuries = Math.DivRem(ns, NanosecondsPerCentury, out long rem);
    if (rem < 0)
    {
      centuries--;
      rem += NanosecondsPerCentury;
    }
    return ((short)centuries, (ulong)rem);
  }

  /// <summary>
  /// Position angle (degrees east of celestial north, (−180, 180]) of the world direction
  /// <paramref name="towards"/> seen from the sky point <paramref name="raDeg"/> /
  /// <paramref name="decDeg"/> (its projection on the tangent plane there).
  /// </summary>
  public static double PositionAngleDeg(double raDeg, double decDeg, Vector3d towards)
  {
    var d = EclipticDirection(raDeg, decDeg);
    var pole = EclipticDirection(0, 90);
    // local east / north at d; near the celestial pole "north" is taken along RA 180°
    var east = Vector3d.Cross(pole, d);
    east = east.Length() > 1e-12 ? Vector3d.Normalize(east) : Vector3d.Normalize(Vector3d.Cross(EclipticDirection(180, 0), d));
    var north = Vector3d.Cross(d, east);
    double pa = Math.Atan2(Vector3d.Dot(towards, east), Vector3d.Dot(towards, north)) / DegToRad;
    return DesktopBillboardGeometry.NormalizeDeg(pa);
  }

  /// <summary>
  /// The billboard rotation (clockwise on screen, degrees) that shows a telescope image the way the
  /// Earth observer looking at <paramref name="raDeg"/> / <paramref name="decDeg"/> sees that sky.
  /// <para>The billboard draws the texture with its +Y (the FITS "up", north up / east left for a
  /// standard WCS) up; <paramref name="imageRotationDeg"/> is the position angle of that +Y, east
  /// of north. The observer's view up is the ecliptic north pole (the south one when
  /// <paramref name="viewUpIsEclipticSouth"/>: a topocentric site south of the equator) projected
  /// on the sky, at position angle ψ. With east to the left of north on screen (the sky seen from
  /// inside), position angles grow counter-clockwise, so image +Y sits (θ − ψ) counter-clockwise from
  /// screen up and the clockwise rotation is ψ − θ.</para>
  /// A mirrored image (east right) cannot be matched by a rotation; its +Y still lines up.
  /// </summary>
  public static double BillboardRotationDeg(double raDeg, double decDeg, double imageRotationDeg, bool viewUpIsEclipticSouth)
  {
    var up = viewUpIsEclipticSouth ? -Vector3d.UnitZ : Vector3d.UnitZ;
    double psi = PositionAngleDeg(raDeg, decDeg, up);
    return DesktopBillboardGeometry.NormalizeDeg(psi - imageRotationDeg);
  }
}
