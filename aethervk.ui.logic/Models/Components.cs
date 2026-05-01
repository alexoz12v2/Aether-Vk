using System.Collections.ObjectModel;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

public partial class TransformComponent : ObservableObject, IComponent
{
  public string Name => "Transform";

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
}

public partial class CameraComponent : ObservableObject, IComponent
{
  public string Name => "Camera";

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
  private bool _isActiveCamera;

  [ObservableProperty]
  private string _projectionMatrixPreview = "View / Projection Matrix Readonly Data";
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
      var runtimeService =
        ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      runtimeService?.SetBvhNodeVisibility(SceneId, EntityId, Index, value);
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
