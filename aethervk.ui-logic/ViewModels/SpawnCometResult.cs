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

  // Orientation quaternion (all modes — user sets initial body orientation)
  public float RotW { get; }
  public float RotX { get; }
  public float RotY { get; }
  public float RotZ { get; }

  /// <summary>Nucleus radius in km.</summary>
  public float CometRadiusKm { get; }

  /// <summary>Mass in kg.</summary>
  public float MassKg { get; }

  // ── Angular velocity (rad/s) — initial spin vector
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
    AngularVelX = angularVelX;
    AngularVelY = angularVelY;
    AngularVelZ = angularVelZ;
    SpkRecordId = spkRecordId;
    CometDesignation = cometDesignation;
    PoleRaDeg = poleRaDeg;
    PoleDecDeg = poleDecDeg;
    PrimeMeridianDeg = primeMeridianDeg;
    PoleRaRateDeg = poleRaRateDeg;
    PoleDecRateDeg = poleDecRateDeg;
    RotationRateDeg = rotationRateDeg;
    WizardStartEpoch = wizardStartEpoch;
    WizardEndEpoch = wizardEndEpoch;

    // Parse NAIF int id from the record string
    SpkNaifId = int.TryParse(spkRecordId, out int id) ? id : 0;

    // Convert Euler (degrees) → Quaternion (ZYX extrinsic)
    var pitch = pitchDeg * (float)(System.Math.PI / 180.0);
    var yaw = yawDeg * (float)(System.Math.PI / 180.0);
    var roll = rollDeg * (float)(System.Math.PI / 180.0);

    float cr = (float)System.Math.Cos(roll * 0.5);
    float sr = (float)System.Math.Sin(roll * 0.5);
    float cp = (float)System.Math.Cos(pitch * 0.5);
    float sp = (float)System.Math.Sin(pitch * 0.5);
    float cy = (float)System.Math.Cos(yaw * 0.5);
    float sy_yaw = (float)System.Math.Sin(yaw * 0.5);

    RotW = cr * cp * cy + sr * sp * sy_yaw;
    RotX = sr * cp * cy - cr * sp * sy_yaw;
    RotY = cr * sp * cy + sr * cp * sy_yaw;
    RotZ = cr * cp * sy_yaw - sr * sp * cy;
  }
}
