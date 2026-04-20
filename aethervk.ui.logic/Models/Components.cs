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

  public float PositionX { get; } = 0.0f;
  public float PositionY { get; } = 0.0f;
  public float PositionZ { get; } = 0.0f;
  public float Temperature { get; } = 5778.0f;

  [ObservableProperty]
  private bool _showBoundingBox;
}

public partial class PlanetComponent : ObservableObject, IComponent
{
  public string Name => "Planet (Ephemeris)";

  public float PositionX { get; } = 150000000.0f;
  public float PositionY { get; } = 0.0f;
  public float PositionZ { get; } = 0.0f;

  public float VelocityX { get; } = 0.0f;
  public float VelocityY { get; } = 30.0f;
  public float VelocityZ { get; } = 0.0f;

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
  public ulong EntityId { get; set; }
  public uint Index { get; set; }

  public string Name { get; set; } = "Node";
  public BvhNodeType Type { get; set; }

  [ObservableProperty]
  private bool _isVisible;

  // Readonly details
  public string Details =>
    Type switch
    {
      BvhNodeType.BoundingSphere => "Radius: 15.2, Center: (0,0,0)",
      BvhNodeType.AABB => "Min: (-10, -10, -10), Max: (10, 10, 10)",
      BvhNodeType.OBB => "Center: (0,0,0), Extents: (10,5,2)",
      _ => "",
    };

  public ObservableCollection<BvhNode> Children { get; } = new();

  partial void OnIsVisibleChanged(bool value)
  {
    if (EntityId != 0)
    {
      var runtimeService = ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      runtimeService?.SetBvhNodeVisibility(EntityId, Index, value);
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

  public float PositionX { get; } = 0.0f;
  public float PositionY { get; } = 0.0f;
  public float PositionZ { get; } = 0.0f;

  public float VelocityX { get; } = 0.0f;
  public float VelocityY { get; } = 0.0f;
  public float VelocityZ { get; } = 0.0f;

  public float AngularVelocityX { get; } = 0.1f;
  public float AngularVelocityY { get; } = 0.2f;
  public float AngularVelocityZ { get; } = 0.05f;

  public string InertiaTensor { get; } = "[[1.0, 0.0, 0.0],\n [0.0, 1.0, 0.0],\n [0.0, 0.0, 1.0]]";

  public ObservableCollection<BvhNode> BvhTree { get; } = new();

  public ObservableCollection<JetMarker> Jets { get; } = new();

  public CometComponent()
  {
    // Mock data
    var root = new BvhNode { Name = "Root BS", Type = BvhNodeType.BoundingSphere };
    var child1 = new BvhNode { Name = "Left AABB", Type = BvhNodeType.AABB };
    var child2 = new BvhNode { Name = "Right AABB", Type = BvhNodeType.AABB };
    root.Children.Add(child1);
    root.Children.Add(child2);

    child1.Children.Add(new BvhNode { Name = "Leaf OBB 1", Type = BvhNodeType.OBB });
    child2.Children.Add(new BvhNode { Name = "Leaf OBB 2", Type = BvhNodeType.OBB });

    BvhTree.Add(root);
  }
}
