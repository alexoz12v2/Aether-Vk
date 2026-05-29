using AetherVk.Logic.Services;

namespace AetherVk.Logic.ViewModels;

public class SpawnCometResult
{
  public ImportedModelItem Model { get; }
  public string EntityName { get; }
  public string PhysicsType { get; }
  public PlanetOrbitData? OrbitData { get; }

  // ── Transform (Static only; zero for Kinematic/Dynamic)
  public float PosX { get; }
  public float PosY { get; }
  public float PosZ { get; }

  public float ScaleX { get; }
  public float ScaleY { get; }
  public float ScaleZ { get; }

  public float RotW { get; }
  public float RotX { get; }
  public float RotY { get; }
  public float RotZ { get; }

  /// <summary>Nucleus radius in km, from the Horizon JPL physical properties block.</summary>
  public float CometRadiusKm { get; }

  // ── Mass (kg)
  /// <summary>
  /// Mass in kg to pass to SpawnComet.
  /// For Static/Kinematic this is cosmetic; for Dynamic it drives the integrator.
  /// Priority: JPL GM-derived → user slider → density estimate.
  /// </summary>
  public float MassKg { get; }

  // ── SPK / Kinematic linkage
  /// <summary>SPK record ID string selected in Step 3 (e.g. "90000030").</summary>
  public string? SpkRecordId { get; }

  /// <summary>SPK record ID parsed to int (NAIF id). 0 when not applicable.</summary>
  public int SpkNaifId { get; }

  /// <summary>Primary designation of the chosen comet (e.g. "1P"). Needed for SPK download.</summary>
  public string? CometDesignation { get; }

  public SpawnCometResult(
    ImportedModelItem model,
    string name,
    string physicsType,
    PlanetOrbitData? orbitData,
    float px,
    float py,
    float pz,
    float sx,
    float sy,
    float sz,
    float pitchDeg,
    float yawDeg,
    float rollDeg,
    float cometRadiusKm,
    float massKg,
    string? spkRecordId,
    string? cometDesignation
  )
  {
    Model = model;
    EntityName = name;
    PhysicsType = physicsType;
    OrbitData = orbitData;
    PosX = px;
    PosY = py;
    PosZ = pz;
    ScaleX = sx;
    ScaleY = sy;
    ScaleZ = sz;
    CometRadiusKm = cometRadiusKm;
    MassKg = massKg;
    SpkRecordId = spkRecordId;
    CometDesignation = cometDesignation;

    // Parse NAIF int id from the record string (e.g. "90000030" → 90000030)
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
