using System;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Utility methods for IAU rotational model → quaternion / Euler angle conversions.
/// These are extracted from RotationalModelEditor and SpawnCometResult for testability.
/// </summary>
public static class IauRotationMath
{
  /// <summary>
  /// Mean obliquity of the ecliptic at J2000 (IAU 2006), in degrees.
  /// This is the angle between the equatorial and ecliptic planes.
  /// </summary>
  public const double ObliquityDeg = 23.4392911;

  /// <summary>
  /// Computes the orientation quaternion from IAU pole (RA, Dec, W) at J2000.
  /// 
  /// Step 1: Build Q_body_to_ICRF = Rz(RA+90°) · Rx(90°-Dec) · Rz(W)
  /// Step 2: Convert to ECLIPJ2000 frame: Q_final = Rx(ε) · Q_body_to_ICRF
  /// 
  /// Returns (w, x, y, z) normalized, in ECLIPJ2000 frame.
  /// </summary>
  public static (double w, double x, double y, double z) IauToQuaternion(
    double poleRaDeg, double poleDecDeg, double primeMeridianDeg)
  {
    double raRad = poleRaDeg * (Math.PI / 180.0);
    double decRad = poleDecDeg * (Math.PI / 180.0);
    double wRad = primeMeridianDeg * (Math.PI / 180.0);

    // Rz(RA + 90°)
    double a1 = raRad + Math.PI / 2.0;
    double cz1 = Math.Cos(a1 * 0.5), sz1 = Math.Sin(a1 * 0.5);

    // Rx(90° - Dec)
    double a2 = Math.PI / 2.0 - decRad;
    double cx1 = Math.Cos(a2 * 0.5), sx1 = Math.Sin(a2 * 0.5);

    // Q_pole = Rz(a1) * Rx(a2)  (Hamilton product)
    double pw = cz1 * cx1;
    double px = cz1 * sx1;
    double py = sz1 * sx1;
    double pz = sz1 * cx1;

    // Rz(W)
    double cw = Math.Cos(wRad * 0.5), sw = Math.Sin(wRad * 0.5);

    // Q_body_to_ICRF = Q_pole * Rz(W)
    double tw = pw * cw - pz * sw;
    double tx = px * cw + py * sw;
    double ty = py * cw - px * sw;
    double tz = pz * cw + pw * sw;

    // Step 2: Convert ICRF → ECLIPJ2000 by pre-multiplying with Rx(ε)
    // Rx(ε) quaternion = (cos(ε/2), sin(ε/2), 0, 0)
    double epsRad = ObliquityDeg * (Math.PI / 180.0);
    double ce = Math.Cos(epsRad * 0.5);
    double se = Math.Sin(epsRad * 0.5);

    // Q_final = Rx(ε) * Q_body_to_ICRF  (Hamilton product)
    double fw = ce * tw - se * tx;
    double fx = ce * tx + se * tw;
    double fy = ce * ty - se * tz;
    double fz = ce * tz + se * ty;

    // Normalize
    double len = Math.Sqrt(fw * fw + fx * fx + fy * fy + fz * fz);
    if (len > 1e-12) { fw /= len; fx /= len; fy /= len; fz /= len; }

    return (fw, fx, fy, fz);
  }

  /// <summary>
  /// Decomposes an orientation quaternion to Euler angles matching
  /// DualRotationGizmo.BuildRotation's YXZ convention:
  ///   M = Ry(yaw) · Rx(pitch) · Rz(roll)
  /// Returns (pitchDeg, yawDeg, rollDeg).
  /// </summary>
  public static (double pitch, double yaw, double roll) QuaternionToGizmoEuler(
    double w, double x, double y, double z)
  {
    // Build quaternion rotation matrix R_q (body→world)
    double r00 = 1 - 2 * (y * y + z * z);
    double r01 = 2 * (x * y - w * z);
    double r02 = 2 * (x * z + w * y);
    double r10 = 2 * (x * y + w * z);
    double r11 = 1 - 2 * (x * x + z * z);
    double r12 = 2 * (y * z - w * x);
    double r22 = 1 - 2 * (x * x + y * y);

    // Decompose R_q = Ry(yaw) * Rx(pitch) * Rz(roll)
    // M[0,2] = -sin(yaw)
    double sinYaw = Math.Max(-1.0, Math.Min(1.0, -r02));
    double yawRad = Math.Asin(sinYaw);

    double pitchRad, rollRad;
    double cosYaw = Math.Cos(yawRad);

    if (Math.Abs(cosYaw) > 1e-6)
    {
      rollRad = Math.Atan2(r01, r00);
      pitchRad = Math.Atan2(r12, r22);
    }
    else
    {
      // Gimbal lock
      rollRad = 0;
      pitchRad = Math.Atan2(-r10, r11);
    }

    return (
      pitchRad * 180.0 / Math.PI,
      yawRad * 180.0 / Math.PI,
      rollRad * 180.0 / Math.PI
    );
  }

  /// <summary>
  /// Builds a 3×3 rotation matrix from Euler angles using the YXZ convention:
  ///   M = Ry(yaw) · Rx(pitch) · Rz(roll)
  /// This matches DualRotationGizmo.BuildRotation exactly.
  /// Returns the matrix as a flat [row*3+col] array of 9 elements.
  /// </summary>
  public static double[] BuildRotation(double pitchDeg, double yawDeg, double rollDeg)
  {
    double p = pitchDeg * Math.PI / 180.0;
    double y = yawDeg * Math.PI / 180.0;
    double r = rollDeg * Math.PI / 180.0;
    double cp = Math.Cos(p), sp = Math.Sin(p);
    double cy = Math.Cos(y), sy = Math.Sin(y);
    double cr = Math.Cos(r), sr = Math.Sin(r);
    return new[]
    {
      cy * cr,             cy * sr,             -sy,
      sp * sy * cr - cp * sr, sp * sy * sr + cp * cr, sp * cy,
      cp * sy * cr + sp * sr, cp * sy * sr - sp * cr, cp * cy,
    };
  }

  /// <summary>
  /// Applies a 3×3 rotation matrix (flat 9-element) to a 3D vector.
  /// </summary>
  public static (double x, double y, double z) Apply(double[] m, double vx, double vy, double vz)
  {
    return (
      m[0] * vx + m[1] * vy + m[2] * vz,
      m[3] * vx + m[4] * vy + m[5] * vz,
      m[6] * vx + m[7] * vy + m[8] * vz
    );
  }
}
