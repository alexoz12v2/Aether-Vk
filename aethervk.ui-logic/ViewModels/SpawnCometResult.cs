using AetherVk.Logic.Services;

namespace AetherVk.Logic.ViewModels;

public class SpawnCometResult
{
  public ImportedModelItem Model { get; }
  public string EntityName { get; }
  public string PhysicsType { get; }

  // ── Transform (Static: user position; Kinematic: zeroed, driven by almanac)
  public float PosX { get; }
  public float PosY { get; }
  public float PosZ { get; }

  // Orientation quaternion (computed from IAU pole RA/Dec/PM at J2000)
  public float RotW { get; }
  public float RotX { get; }
  public float RotY { get; }
  public float RotZ { get; }

  /// <summary>Nucleus radius in km.</summary>
  public float CometRadiusKm { get; }

  /// <summary>Mass in kg.</summary>
  public float MassKg { get; }

  // ── Angular velocity (rad/s) — computed from rotation rate along pole axis
  public float AngularVelX { get; }
  public float AngularVelY { get; }
  public float AngularVelZ { get; }

  // ── SPK / Kinematic linkage
  /// <summary>SPK record ID string (e.g. "90000030").</summary>
  public string? SpkRecordId { get; }

  /// <summary>SPK record ID parsed to int (NAIF id). 0 when not applicable.</summary>
  public int SpkNaifId { get; }

  /// <summary>Primary designation of the chosen comet (e.g. "1P").</summary>
  public string? CometDesignation { get; }

  // ── IAU Rotational Model
  public double PoleRaDeg { get; }
  public double PoleDecDeg { get; }
  public double PrimeMeridianDeg { get; }
  public double PoleRaRateDeg { get; }
  public double PoleDecRateDeg { get; }
  public double RotationRateDeg { get; }

  // ── Timeline (validated epoch interval to commit on spawn)
  public System.DateTimeOffset WizardStartEpoch { get; }
  public System.DateTimeOffset WizardEndEpoch { get; }

  public SpawnCometResult(
    ImportedModelItem model,
    string name,
    string physicsType,
    float px,
    float py,
    float pz,
    float pitchDeg,
    float yawDeg,
    float rollDeg,
    float cometRadiusKm,
    float massKg,
    float angularVelX,
    float angularVelY,
    float angularVelZ,
    string? spkRecordId,
    int spkNaifId,
    string? cometDesignation,
    double poleRaDeg,
    double poleDecDeg,
    double primeMeridianDeg,
    double poleRaRateDeg,
    double poleDecRateDeg,
    double rotationRateDeg,
    System.DateTimeOffset wizardStartEpoch,
    System.DateTimeOffset wizardEndEpoch
  )
  {
    Model = model;
    EntityName = name;
    PhysicsType = physicsType;
    PosX = px;
    PosY = py;
    PosZ = pz;
    CometRadiusKm = cometRadiusKm;
    MassKg = massKg;
    SpkRecordId = spkRecordId;
    SpkNaifId = spkNaifId;
    CometDesignation = cometDesignation;
    PoleRaDeg = poleRaDeg;
    PoleDecDeg = poleDecDeg;
    PrimeMeridianDeg = primeMeridianDeg;
    PoleRaRateDeg = poleRaRateDeg;
    PoleDecRateDeg = poleDecRateDeg;
    RotationRateDeg = rotationRateDeg;
    WizardStartEpoch = wizardStartEpoch;
    WizardEndEpoch = wizardEndEpoch;

    // ── Compute orientation from IAU pole (RA, Dec, W) at J2000 ──
    // IAU convention: body-fixed Z axis points along the pole (RA, Dec),
    // and the prime meridian W defines the rotation about the pole at epoch.
    double raRad = poleRaDeg * (System.Math.PI / 180.0);
    double decRad = poleDecDeg * (System.Math.PI / 180.0);
    double wRad = primeMeridianDeg * (System.Math.PI / 180.0);

    // Step 1: Rotation from J2000 inertial Z to pole direction
    // Achieved by: Rz(ra + 90°) * Rx(90° - dec)
    double rz1Angle = raRad + System.Math.PI / 2.0;
    double rx1Angle = System.Math.PI / 2.0 - decRad;

    // Rz(rz1Angle) quaternion: (0, 0, sin(a/2), cos(a/2))
    double cz1 = System.Math.Cos(rz1Angle * 0.5);
    double sz1 = System.Math.Sin(rz1Angle * 0.5);
    double qz1_x = 0, qz1_y = 0, qz1_z = sz1, qz1_w = cz1;

    // Rx(rx1Angle) quaternion: (sin(a/2), 0, 0, cos(a/2))
    double cx1 = System.Math.Cos(rx1Angle * 0.5);
    double sx1 = System.Math.Sin(rx1Angle * 0.5);
    double qx1_x = sx1, qx1_y = 0, qx1_z = 0, qx1_w = cx1;

    // Q_pole = Rz * Rx (Hamilton product)
    double qp_w = qz1_w * qx1_w - qz1_x * qx1_x - qz1_y * qx1_y - qz1_z * qx1_z;
    double qp_x = qz1_w * qx1_x + qz1_x * qx1_w + qz1_y * qx1_z - qz1_z * qx1_y;
    double qp_y = qz1_w * qx1_y - qz1_x * qx1_z + qz1_y * qx1_w + qz1_z * qx1_x;
    double qp_z = qz1_w * qx1_z + qz1_x * qx1_y - qz1_y * qx1_x + qz1_z * qx1_w;

    // Step 2: Rotation about pole by prime meridian angle W
    double cw = System.Math.Cos(wRad * 0.5);
    double sw = System.Math.Sin(wRad * 0.5);
    double qw_x = 0, qw_y = 0, qw_z = sw, qw_w = cw;

    // Q_total = Q_pole * Q_W
    double qt_w = qp_w * qw_w - qp_x * qw_x - qp_y * qw_y - qp_z * qw_z;
    double qt_x = qp_w * qw_x + qp_x * qw_w + qp_y * qw_z - qp_z * qw_y;
    double qt_y = qp_w * qw_y - qp_x * qw_z + qp_y * qw_w + qp_z * qw_x;
    double qt_z = qp_w * qw_z + qp_x * qw_y - qp_y * qw_x + qp_z * qw_w;

    // Normalize
    double qLen = System.Math.Sqrt(qt_w * qt_w + qt_x * qt_x + qt_y * qt_y + qt_z * qt_z);
    if (qLen > 1e-12)
    {
      qt_w /= qLen; qt_x /= qLen; qt_y /= qLen; qt_z /= qLen;
    }

    RotW = (float)qt_w;
    RotX = (float)qt_x;
    RotY = (float)qt_y;
    RotZ = (float)qt_z;

    // ── Compute angular velocity from rotation rate along pole axis ──
    double poleX = System.Math.Cos(decRad) * System.Math.Cos(raRad);
    double poleY = System.Math.Cos(decRad) * System.Math.Sin(raRad);
    double poleZ = System.Math.Sin(decRad);

    // rotationRateDeg is degrees/day → rad/s
    double omegaRadPerSec = rotationRateDeg * (System.Math.PI / 180.0) / 86400.0;

    AngularVelX = (float)(poleX * omegaRadPerSec);
    AngularVelY = (float)(poleY * omegaRadPerSec);
    AngularVelZ = (float)(poleZ * omegaRadPerSec);
  }
}
