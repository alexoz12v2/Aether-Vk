using AetherVk.Logic.Services;

namespace AetherVk.Logic.ViewModels;

public class SpawnCometResult
{
  public ImportedModelItem Model { get; }
  public string EntityName { get; }
  public string PhysicsType { get; }
  public PlanetOrbitData? OrbitData { get; }

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
    float cometRadiusKm = 1.0f
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

    // Convert Euler (degrees) to Quaternion
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
