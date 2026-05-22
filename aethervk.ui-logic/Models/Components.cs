using System.Collections.ObjectModel;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Models;

public partial class TransformComponent : NativeComponent
{
  public override string Name => "Transform";

  public bool SuspendNotifications { get; set; } = false;

  [ObservableProperty]
  private bool _isEditable = true;

  [ObservableProperty]
  private float _posX;

  [ObservableProperty]
  private float _posY;

  [ObservableProperty]
  private float _posZ;

  [ObservableProperty]
  private float _rotW = 1.0f;

  [ObservableProperty]
  private float _rotX;

  [ObservableProperty]
  private float _rotY;

  [ObservableProperty]
  private float _rotZ;

  [ObservableProperty]
  private float _scaleX = 1.0f;

  [ObservableProperty]
  private float _scaleY = 1.0f;

  [ObservableProperty]
  private float _scaleZ = 1.0f;

  protected override bool ShouldPushToNative(string? propertyName)
  {
    return propertyName != nameof(IsEditable) && propertyName != nameof(SuspendNotifications);
  }

  protected override void PushToNativeImpl()
  {
    if (SuspendNotifications)
      return;
    var data = new NativeInterop.FfiTransform
    {
      Px = PosX,
      Py = PosY,
      Pz = PosZ,
      Rw = RotW,
      Rx = RotX,
      Ry = RotY,
      Rz = RotZ,
      Sx = ScaleX,
      Sy = ScaleY,
      Sz = ScaleZ,
    };
    NativeInterop.avkSimulationContext_setTransformComponent(
      SimulationContext,
      SceneId,
      EntityId,
      in data
    );
  }

  protected override void PullFromNativeImpl()
  {
    if (
      NativeInterop.avkSimulationContext_getTransformComponent(
        SimulationContext,
        SceneId,
        EntityId,
        out var data
      )
    )
    {
      PosX = data.Px;
      PosY = data.Py;
      PosZ = data.Pz;
      RotW = data.Rw;
      RotX = data.Rx;
      RotY = data.Ry;
      RotZ = data.Rz;
      ScaleX = data.Sx;
      ScaleY = data.Sy;
      ScaleZ = data.Sz;
    }
  }
}

public partial class CameraComponent : NativeComponent
{
  public override string Name => "Camera";

  public bool SuspendNotifications { get; set; } = false;

  [ObservableProperty]
  private float _fov = 45.0f;

  [ObservableProperty]
  private float _aspectRatio = 1.77f;

  [ObservableProperty]
  private float _nearPlane = 0.1f;

  [ObservableProperty]
  private float _farPlane = 10000.0f;

  [ObservableProperty]
  private float _orthoLeft = -10.0f;

  [ObservableProperty]
  private float _orthoRight = 10.0f;

  [ObservableProperty]
  private float _orthoBottom = -10.0f;

  [ObservableProperty]
  private float _orthoTop = 10.0f;

  [ObservableProperty]
  private bool _isOrthographic;

  [ObservableProperty]
  private string _projectionMatrixPreview = "View / Projection Matrix Readonly Data";

  protected override bool ShouldPushToNative(string? propertyName)
  {
    return propertyName != nameof(ProjectionMatrixPreview);
  }

  protected override void PushToNativeImpl()
  {
    var data = new NativeInterop.FfiCamera
    {
      IsOrthographic = IsOrthographic,
      Fov = Fov,
      Aspect = AspectRatio,
      Near = NearPlane,
      Far = FarPlane,
      OrthoLeft = OrthoLeft,
      OrthoRight = OrthoRight,
      OrthoBottom = OrthoBottom,
      OrthoTop = OrthoTop,
      // proj array doesn't matter for pushing
    };
    NativeInterop.avkSimulationContext_setCameraComponent(
      SimulationContext,
      SceneId,
      EntityId,
      in data
    );
  }

  protected override void PullFromNativeImpl()
  {
    if (
      NativeInterop.avkSimulationContext_getCameraComponent(
        SimulationContext,
        SceneId,
        EntityId,
        out var data
      )
    )
    {
      IsOrthographic = data.IsOrthographic;
      Fov = data.Fov;
      AspectRatio = data.Aspect;
      NearPlane = data.Near;
      FarPlane = data.Far;
      OrthoLeft = data.OrthoLeft;
      OrthoRight = data.OrthoRight;
      OrthoBottom = data.OrthoBottom;
      OrthoTop = data.OrthoTop;

      // Safe unpacking of the projection matrix
      ProjectionMatrixPreview =
        $"[{data.Proj[0]:F2}, {data.Proj[4]:F2}, {data.Proj[8]:F2}, {data.Proj[12]:F2}]\n"
        + $"[{data.Proj[1]:F2}, {data.Proj[5]:F2}, {data.Proj[9]:F2}, {data.Proj[13]:F2}]\n"
        + $"[{data.Proj[2]:F2}, {data.Proj[6]:F2}, {data.Proj[10]:F2}, {data.Proj[14]:F2}]\n"
        + $"[{data.Proj[3]:F2}, {data.Proj[7]:F2}, {data.Proj[11]:F2}, {data.Proj[15]:F2}]";
    }
  }
}

public partial class CursorComponent : ObservableObject, IComponent
{
  public string Name => "Cursor";

  [ObservableProperty]
  private bool _isVisible = true;
}

public partial class GridComponent : ObservableObject, IComponent
{
  public string Name => "Grid";

  [ObservableProperty]
  private bool _isVisible = true;
}

public partial class SunComponent : ObservableObject, IComponent
{
  public string Name => "Sun";

  [ObservableProperty]
  private float _positionX;

  [ObservableProperty]
  private float _positionY;

  [ObservableProperty]
  private float _positionZ;

  [ObservableProperty]
  private float _temperature = 5778.0f; // K

  [ObservableProperty]
  private bool _showBoundingBox = false;
}

public partial class MeasurementComponent : ObservableObject, IComponent
{
  public string Name => "Measurement";
}

public partial class PlanetComponent : ObservableObject, IComponent
{
  public string Name => "Planet (Ephemeris)";

  [ObservableProperty]
  private float _positionX = 150000000.0f;

  [ObservableProperty]
  private float _positionY;

  [ObservableProperty]
  private float _positionZ;

  [ObservableProperty]
  private float _velocityX;

  [ObservableProperty]
  private float _velocityY = 30.0f;

  [ObservableProperty]
  private float _velocityZ;

  [ObservableProperty]
  private bool _showBoundingBox;
}

public enum BvhNodeType
{
  BoundingSphere,
  AABB,
  OBB,
}

public class BvhNodeVisibilityChangedMessage
{
  public ulong SceneId { get; }
  public ulong EntityId { get; }
  public uint NodeIndex { get; }
  public bool IsVisible { get; }

  public BvhNodeVisibilityChangedMessage(
    ulong sceneId,
    ulong entityId,
    uint nodeIndex,
    bool isVisible
  )
  {
    SceneId = sceneId;
    EntityId = entityId;
    NodeIndex = nodeIndex;
    IsVisible = isVisible;
  }
}

public partial class BvhNode : ObservableObject
{
  public ulong SceneId { get; set; }
  public ulong EntityId { get; set; }
  public uint Index { get; set; }

  public string Name { get; set; } = "Node";
  public BvhNodeType Type { get; set; }

  [ObservableProperty]
  private bool _isVisible;

  [ObservableProperty]
  private string _details = "";

  public ObservableCollection<BvhNode> Children { get; } = new();

  partial void OnIsVisibleChanged(bool value)
  {
    if (EntityId != 0)
    {
      WeakReferenceMessenger.Default.Send(
        new BvhNodeVisibilityChangedMessage(SceneId, EntityId, Index, value)
      );
    }
  }
}

public partial class JetMarker : ObservableObject
{
  [ObservableProperty]
  private string _name = "New Jet";

  [ObservableProperty]
  private float _colorR = 1.0f;

  [ObservableProperty]
  private float _colorG = 0.0f;

  [ObservableProperty]
  private float _colorB = 0.0f;

  [ObservableProperty]
  private float _posX;

  [ObservableProperty]
  private float _posY;

  [ObservableProperty]
  private float _posZ;

  [ObservableProperty]
  private float _size = 5.0f;
}

public partial class CometComponent : ObservableObject, IComponent
{
  public string Name => "Comet";

  [ObservableProperty]
  private float _positionX;

  [ObservableProperty]
  private float _positionY;

  [ObservableProperty]
  private float _positionZ;

  [ObservableProperty]
  private float _velocityX;

  [ObservableProperty]
  private float _velocityY;

  [ObservableProperty]
  private float _velocityZ;

  [ObservableProperty]
  private float _angularVelocityX = 0.1f;

  [ObservableProperty]
  private float _angularVelocityY = 0.2f;

  [ObservableProperty]
  private float _angularVelocityZ = 0.05f;

  [ObservableProperty]
  private string _inertiaTensor = "[[1.0, 0.0, 0.0],\n [0.0, 1.0, 0.0],\n [0.0, 0.0, 1.0]]";

  public ObservableCollection<BvhNode> BvhTree { get; } = new();

  public ObservableCollection<JetMarker> Jets { get; } = new();

  public CometComponent() { }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical Gizmo Component
// ─────────────────────────────────────────────────────────────────────────────

public partial class SphericalGizmoComponent : ObservableObject, IComponent
{
  public string Name => "Spherical Gizmo";

  [ObservableProperty]
  private bool _isVisible = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Particle Emitter Circles
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// C# UI model for a single circular emission zone on a comet surface.
/// Angles are stored in degrees; the FFI layer converts to radians.
/// </summary>
public partial class EmissionCircleItem : ObservableObject
{
  /// <summary>Latitude of the emission circle centre, in degrees (−90 south … +90 north).</summary>
  [ObservableProperty]
  private float _latitudeDeg;

  /// <summary>Longitude of the emission circle centre, in degrees (0 … 360).</summary>
  [ObservableProperty]
  private float _longitudeDeg;

  /// <summary>
  /// Radius of the emission disc as a fraction of the mesh bounding-sphere radius (0.001 … 1.0).
  /// </summary>
  [ObservableProperty]
  private float _circleRadius = 0.1f;

  /// <summary>Mass of particles emitted from this circle.</summary>
  [ObservableProperty]
  private float _mass = 1.0f;

  // ── Colour ──────────────────────────────────────────────────────────────────
  [ObservableProperty] private float _colorR = 1.0f;
  [ObservableProperty] private float _colorG = 0.6f;
  [ObservableProperty] private float _colorB = 0.2f;
  [ObservableProperty] private float _colorA = 1.0f;
}

/// <summary>
/// Attaches a set of discrete circular particle-emission zones to a comet mesh entity.
/// </summary>
public partial class ParticleEmitterCirclesComponent : ObservableObject, IComponent
{
  public string Name => "Particle Emitter Circles";

  public ObservableCollection<EmissionCircleItem> Circles { get; } = new();

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void AddCircle() => Circles.Add(new EmissionCircleItem());

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void RemoveCircle(EmissionCircleItem item) => Circles.Remove(item);
}
