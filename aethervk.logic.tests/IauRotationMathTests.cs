using System;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Tests for <see cref="IauRotationMath"/>: validates that the IAU pole
/// (RA, Dec, W) → quaternion pipeline correctly produces orientations
/// in the ECLIPJ2000 frame (engine frame), including the obliquity
/// correction from ICRF to ECLIPJ2000.
///
/// Coordinate frames:
///   ICRF (equatorial J2000): X = vernal equinox, Z = celestial north pole
///   ECLIPJ2000 (ecliptic J2000): X = vernal equinox (shared), Z = ecliptic north pole
///   Rotation from ICRF → ECLIPJ2000: Rx(ε) where ε ≈ 23.44°
/// </summary>
public class IauRotationMathTests
{
  private const double Tol = 1e-6;
  private static readonly double EpsRad = IauRotationMath.ObliquityDeg * (Math.PI / 180.0);
  private static readonly double CosEps = Math.Cos(EpsRad);
  private static readonly double SinEps = Math.Sin(EpsRad);

  /// <summary>
  /// Helper: rotate body-axis (bx,by,bz) through the full pipeline:
  ///   IAU params → quaternion → Euler → BuildRotation → apply
  /// Returns the ECLIPJ2000-space direction of the body axis.
  /// </summary>
  private static (double x, double y, double z) PipelineTransform(
    double ra, double dec, double pm, double bx, double by, double bz)
  {
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(ra, dec, pm);
    var (p, yaw, r) = IauRotationMath.QuaternionToGizmoEuler(w, x, y, z);
    var m = IauRotationMath.BuildRotation(p, yaw, r);
    return IauRotationMath.Apply(m, bx, by, bz);
  }

  /// <summary>
  /// Helper: rotate body-axis directly using the quaternion rotation matrix.
  /// </summary>
  private static (double x, double y, double z) QuaternionTransform(
    double w, double qx, double qy, double qz, double bx, double by, double bz)
  {
    double r00 = 1 - 2 * (qy * qy + qz * qz);
    double r01 = 2 * (qx * qy - w * qz);
    double r02 = 2 * (qx * qz + w * qy);
    double r10 = 2 * (qx * qy + w * qz);
    double r11 = 1 - 2 * (qx * qx + qz * qz);
    double r12 = 2 * (qy * qz - w * qx);
    double r20 = 2 * (qx * qz - w * qy);
    double r21 = 2 * (qy * qz + w * qx);
    double r22 = 1 - 2 * (qx * qx + qy * qy);
    return (
      r00 * bx + r01 * by + r02 * bz,
      r10 * bx + r11 * by + r12 * bz,
      r20 * bx + r21 * by + r22 * bz
    );
  }

  // ─── Quaternion from IAU tests (ECLIPJ2000 output) ──────────────────────────

  [Fact]
  public void RA0_Dec0_PM0_PoleAlongEclipticX()
  {
    // IAU: RA=0, Dec=0 means the pole (body Z) points at the vernal equinox.
    // Vernal equinox = ICRF X = ECLIPJ2000 X (shared axis).
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(0, 0, 0);
    var (wx, wy, wz) = QuaternionTransform(w, x, y, z, 0, 0, 1);
    Assert.Equal(1.0, wx, Tol);
    Assert.Equal(0.0, wy, Tol);
    Assert.Equal(0.0, wz, Tol);
  }

  [Fact]
  public void RA0_Dec90_PM0_PoleAlongCelestialNorth_InEcliptic()
  {
    // Dec=90° means the pole points at the celestial north pole (ICRF Z).
    // In ECLIPJ2000: ICRF Z = (0, -sin(ε), cos(ε))
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(0, 90, 0);
    var (wx, wy, wz) = QuaternionTransform(w, x, y, z, 0, 0, 1);
    Assert.Equal(0.0, wx, Tol);
    Assert.Equal(-SinEps, wy, Tol);
    Assert.Equal(CosEps, wz, Tol);
  }

  [Fact]
  public void EclipticDefaults_IsIdentity()
  {
    // RA=90°, Dec=90°-ε, W₀=180°: exact identity in ECLIPJ2000.
    // Rz(180°)·Rx(ε)·Rz(180°) = Rx(-ε), then Rx(ε)·Rx(-ε) = I.
    double eclipticDec = 90.0 - IauRotationMath.ObliquityDeg;
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(90, eclipticDec, 180);
    Assert.Equal(1.0, Math.Abs(w), Tol);
    Assert.Equal(0.0, x, Tol);
    Assert.Equal(0.0, y, Tol);
    Assert.Equal(0.0, z, Tol);
  }

  [Fact]
  public void RA90_Dec0_PM0_PoleAlongIcrfY_InEcliptic()
  {
    // RA=90° means pole at ICRF (0,1,0).
    // In ECLIPJ2000: Rx(ε)·(0,1,0) = (0, cos(ε), sin(ε))
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(90, 0, 0);
    var (wx, wy, wz) = QuaternionTransform(w, x, y, z, 0, 0, 1);
    Assert.Equal(0.0, wx, Tol);
    Assert.Equal(CosEps, wy, Tol);
    Assert.Equal(SinEps, wz, Tol);
  }

  // ─── Euler decomposition round-trip tests ──────────────────────────────────

  [Theory]
  [InlineData(0, 0, 0)]
  [InlineData(0, 90, 0)]
  [InlineData(90, 0, 0)]
  [InlineData(90, 66.56, 0)]
  [InlineData(45, 45, 0)]
  [InlineData(0, 0, 180)]
  [InlineData(120, 30, 45)]
  [InlineData(200, -60, 90)]
  public void Pipeline_BodyZ_MatchesQuaternion(double ra, double dec, double pm)
  {
    // The Euler → BuildRotation pipeline should produce the SAME body-Z direction
    // as the quaternion rotation matrix directly.
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(ra, dec, pm);
    var (qx, qy, qz) = QuaternionTransform(w, x, y, z, 0, 0, 1);
    var (px, py, pz) = PipelineTransform(ra, dec, pm, 0, 0, 1);
    Assert.Equal(qx, px, Tol);
    Assert.Equal(qy, py, Tol);
    Assert.Equal(qz, pz, Tol);
  }

  [Theory]
  [InlineData(0, 0, 0)]
  [InlineData(0, 90, 0)]
  [InlineData(90, 0, 0)]
  [InlineData(90, 66.56, 0)]
  [InlineData(45, 45, 0)]
  [InlineData(120, 30, 45)]
  public void Pipeline_BodyX_MatchesQuaternion(double ra, double dec, double pm)
  {
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(ra, dec, pm);
    var (qx, qy, qz) = QuaternionTransform(w, x, y, z, 1, 0, 0);
    var (px, py, pz) = PipelineTransform(ra, dec, pm, 1, 0, 0);
    Assert.Equal(qx, px, Tol);
    Assert.Equal(qy, py, Tol);
    Assert.Equal(qz, pz, Tol);
  }

  [Theory]
  [InlineData(0, 0, 0)]
  [InlineData(0, 90, 0)]
  [InlineData(90, 0, 0)]
  [InlineData(90, 66.56, 0)]
  [InlineData(45, 45, 0)]
  [InlineData(120, 30, 45)]
  public void Pipeline_BodyY_MatchesQuaternion(double ra, double dec, double pm)
  {
    var (w, x, y, z) = IauRotationMath.IauToQuaternion(ra, dec, pm);
    var (qx, qy, qz) = QuaternionTransform(w, x, y, z, 0, 1, 0);
    var (px, py, pz) = PipelineTransform(ra, dec, pm, 0, 1, 0);
    Assert.Equal(qx, px, Tol);
    Assert.Equal(qy, py, Tol);
    Assert.Equal(qz, pz, Tol);
  }

  // ─── Specific gizmo display verification ───────────────────────────────────

  [Fact]
  public void RA0_Dec0_PM0_GizmoShowsBodyZ_AlongWorldX()
  {
    // Pole at vernal equinox → body Z should point along ECLIPJ2000 X
    var (wx, wy, wz) = PipelineTransform(0, 0, 0, 0, 0, 1);
    Assert.Equal(1.0, wx, Tol);
    Assert.Equal(0.0, wy, Tol);
    Assert.Equal(0.0, wz, Tol);
  }

  [Fact]
  public void EclipticDefaults_GizmoShowsIdentity()
  {
    // With default IAU params (RA=90°, Dec=66.56°, W₀=180°), all body axes
    // should align with ECLIPJ2000 axes via the gizmo pipeline.
    double eclipticDec = 90.0 - IauRotationMath.ObliquityDeg;

    var (xx, xy, xz) = PipelineTransform(90, eclipticDec, 180, 1, 0, 0);
    Assert.Equal(1.0, xx, Tol); Assert.Equal(0.0, xy, Tol); Assert.Equal(0.0, xz, Tol);

    var (yx, yy, yz) = PipelineTransform(90, eclipticDec, 180, 0, 1, 0);
    Assert.Equal(0.0, yx, Tol); Assert.Equal(1.0, yy, Tol); Assert.Equal(0.0, yz, Tol);

    var (zx, zy, zz) = PipelineTransform(90, eclipticDec, 180, 0, 0, 1);
    Assert.Equal(0.0, zx, Tol); Assert.Equal(0.0, zy, Tol); Assert.Equal(1.0, zz, Tol);
  }

  // ─── Orthogonality check ───────────────────────────────────────────────────

  [Theory]
  [InlineData(0, 0, 0)]
  [InlineData(30, 60, 45)]
  [InlineData(180, -45, 90)]
  [InlineData(350, 85, 270)]
  public void Pipeline_ProducesOrthogonalAxes(double ra, double dec, double pm)
  {
    var (x1, y1, z1) = PipelineTransform(ra, dec, pm, 1, 0, 0);
    var (x2, y2, z2) = PipelineTransform(ra, dec, pm, 0, 1, 0);
    var (x3, y3, z3) = PipelineTransform(ra, dec, pm, 0, 0, 1);

    // Dot products should be zero (orthogonal)
    double dot12 = x1 * x2 + y1 * y2 + z1 * z2;
    double dot13 = x1 * x3 + y1 * y3 + z1 * z3;
    double dot23 = x2 * x3 + y2 * y3 + z2 * z3;
    Assert.Equal(0.0, dot12, Tol);
    Assert.Equal(0.0, dot13, Tol);
    Assert.Equal(0.0, dot23, Tol);

    // Each axis should have unit length
    double len1 = Math.Sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    double len2 = Math.Sqrt(x2 * x2 + y2 * y2 + z2 * z2);
    double len3 = Math.Sqrt(x3 * x3 + y3 * y3 + z3 * z3);
    Assert.Equal(1.0, len1, Tol);
    Assert.Equal(1.0, len2, Tol);
    Assert.Equal(1.0, len3, Tol);
  }
}
