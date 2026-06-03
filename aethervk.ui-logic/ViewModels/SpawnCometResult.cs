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
    // Uses IauToQuaternion which includes the ICRF → ECLIPJ2000 obliquity correction.
    var (qw, qx, qy, qz) = AetherVk.Logic.ViewModels.IauRotationMath.IauToQuaternion(
      poleRaDeg, poleDecDeg, primeMeridianDeg);

    RotW = (float)qw;
    RotX = (float)qx;
    RotY = (float)qy;
    RotZ = (float)qz;

    // ── Compute angular velocity from rotation rate along pole axis ──
    // Pole direction in ICRF Cartesian from RA/Dec
    double raRad = poleRaDeg * (System.Math.PI / 180.0);
    double decRad = poleDecDeg * (System.Math.PI / 180.0);
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
