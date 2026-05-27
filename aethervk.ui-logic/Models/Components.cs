using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
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

    if (SimulationContext == IntPtr.Zero)
      return;

    System.Console.WriteLine($"[TransformComponent] PushToNativeImpl called! X={PosX}, Y={PosY}, Z={PosZ}. StackTrace:\n{new System.Diagnostics.StackTrace()}");

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

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyPosition()
  {
    string json = $"{{ \"x\": {PosX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {PosY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {PosZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.CopyToClipboardMessage(json));
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyRotation()
  {
    string json = $"{{ \"x\": {RotX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {RotY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {RotZ.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"w\": {RotW.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.CopyToClipboardMessage(json));
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyScale()
  {
    string json = $"{{ \"x\": {ScaleX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {ScaleY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {ScaleZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(new AetherVk.Logic.Messages.CopyToClipboardMessage(json));
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
  private float _nearPlane = 0.01f;

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

  // Clamp FOV to a valid perspective range [0.1°, 179.0°].
  partial void OnFovChanged(float value)
  {
    if (value < 0.1f || value > 179.0f)
      Fov = System.Math.Min(System.Math.Max(value, 0.1f), 179.0f);
  }

  // Near must be positive and strictly less than Far.
  partial void OnNearPlaneChanged(float value)
  {
    const float minNear = 0.01f;
    float clamped = System.Math.Max(value, minNear);
    if (clamped >= FarPlane)
      clamped = System.Math.Max(FarPlane - 0.0001f, minNear);
    if (clamped != value)
      NearPlane = clamped;
  }

  // Far must be strictly greater than Near and at most 10 000 AU.
  partial void OnFarPlaneChanged(float value)
  {
    float clamped = System.Math.Min(System.Math.Max(value, NearPlane + 0.0001f), 10_000.0f);
    if (clamped != value)
      FarPlane = clamped;
  }

  protected override void PushToNativeImpl()
  {
    if (SuspendNotifications)
      return;

    // Additional safety clamp: never send invalid values to native even if
    // the OnXxxChanged correction hasn't fired yet.
    float safeFov = System.Math.Min(System.Math.Max(Fov, 0.1f), 179.0f);
    float safeNear = System.Math.Max(NearPlane, 0.01f);
    float safeFar = System.Math.Max(FarPlane, safeNear + 0.0001f);

    var data = new NativeInterop.FfiCamera
    {
      IsOrthographic = IsOrthographic,
      Fov = safeFov,
      Aspect = AspectRatio,
      Near = safeNear,
      Far = safeFar,
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

    // Pull from native immediately to update the projection matrix preview
    PullFromNative();
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

      ProjectionMatrixPreview =
        $"[ {data.Proj00,7:F2} {data.Proj10,7:F2} {data.Proj20,7:F2} {data.Proj30,7:F2} ]\n"
        + $"[ {data.Proj01,7:F2} {data.Proj11,7:F2} {data.Proj21,7:F2} {data.Proj31,7:F2} ]\n"
        + $"[ {data.Proj02,7:F2} {data.Proj12,7:F2} {data.Proj22,7:F2} {data.Proj32,7:F2} ]\n"
        + $"[ {data.Proj03,7:F2} {data.Proj13,7:F2} {data.Proj23,7:F2} {data.Proj33,7:F2} ]";
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

  [ObservableProperty]
  private float _radius = 0.1f;

  public float RadiusKm
  {
      get => Radius / 1000f;
      set => Radius = value * 1000f;
  }

  partial void OnRadiusChanged(float value)
  {
      OnPropertyChanged(nameof(RadiusKm));
  }

  [ObservableProperty]
  private float _latitude;

  [ObservableProperty]
  private float _longitude;

  [ObservableProperty]
  private float _mass = 1.0f;

  public float MassGrams
  {
      get => Mass * 1000f;
      set => Mass = value / 1000f;
  }

  partial void OnMassChanged(float value)
  {
      OnPropertyChanged(nameof(MassGrams));
  }

  [ObservableProperty]
  private int _particlesPerTick = 100;

  [ObservableProperty]
  private float _tTL = 1000.0f;

  [ObservableProperty]
  private float _meanVelocity = 10.0f;
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

  /// <summary>Mass of particles emitted from this circle (in kg).</summary>
  [ObservableProperty]
  private float _mass = 1.0f;

  public float MassGrams
  {
      get => Mass * 1000f;
      set => Mass = value / 1000f;
  }

  partial void OnMassChanged(float value)
  {
      OnPropertyChanged(nameof(MassGrams));
  }

  // ── Colour ──────────────────────────────────────────────────────────────────
  [ObservableProperty] private float _colorR = 1.0f;
  [ObservableProperty] private float _colorG = 0.6f;
  [ObservableProperty] private float _colorB = 0.2f;
  [ObservableProperty] private float _colorA = 1.0f;

  // ── Emission Params ─────────────────────────────────────────────────────────
  [ObservableProperty]
  private uint _particlesPerTick = 100;

  [ObservableProperty]
  private ulong _tTL = 1000;

  [ObservableProperty]
  private float _meanVelocity = 10.0f;

  public float MeanVelocityKms
  {
      get => MeanVelocity / 1000f;
      set => MeanVelocity = value * 1000f;
  }

  partial void OnMeanVelocityChanged(float value)
  {
      OnPropertyChanged(nameof(MeanVelocityKms));
  }
}

/// <summary>
/// Attaches a set of discrete circular particle-emission zones to a comet mesh entity.
/// </summary>
public partial class ParticleEmitterCirclesComponent : NativeComponent
{
  public override string Name => "Particle Emitter Circles";

  public ObservableCollection<EmissionCircleItem> Circles { get; } = new();

  private bool _isSyncing = false;

  public ParticleEmitterCirclesComponent()
  {
    Circles.CollectionChanged += (s, e) =>
    {
      if (_isSyncing) return;
      if (e.NewItems != null)
      {
        foreach (EmissionCircleItem item in e.NewItems)
          item.PropertyChanged += Item_PropertyChanged;
      }
      if (e.OldItems != null)
      {
        foreach (EmissionCircleItem item in e.OldItems)
          item.PropertyChanged -= Item_PropertyChanged;
      }
      PushToNativeImpl();
    };
  }

  private void Item_PropertyChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
  {
    if (_isSyncing) return;
    PushToNativeImpl();
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void AddCircle() => Circles.Add(new EmissionCircleItem());

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void RemoveCircle(EmissionCircleItem item) => Circles.Remove(item);

  protected override bool ShouldPushToNative(string? propertyName) => true;

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero) return;

    var arr = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle[Circles.Count];
    for (int i = 0; i < Circles.Count; i++)
    {
      arr[i] = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle
      {
        LatitudeRad = Circles[i].LatitudeDeg * (float)Math.PI / 180f,
        LongitudeRad = Circles[i].LongitudeDeg * (float)Math.PI / 180f,
        // Assuming CircleRadius is in meters, converting to km:
        CircleRadiusFrac = Circles[i].CircleRadius / 1000f,
        Mass = Circles[i].Mass / 1000f,
        ColorR = Circles[i].ColorR,
        ColorG = Circles[i].ColorG,
        ColorB = Circles[i].ColorB,
        ColorA = Circles[i].ColorA,
        ParticlesPerTick = Circles[i].ParticlesPerTick,
        TTL = Circles[i].TTL,
        MeanVelocity = Circles[i].MeanVelocity,
      };
    }
    AetherVk.Logic.Services.NativeInterop.avkSimulationContext_setParticleEmitterCirclesComponent(
      SimulationContext, SceneId, EntityId, arr, (uint)arr.Length);
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero) return;

    uint maxCount = 64;
    var arr = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle[maxCount];
    if (AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getParticleEmitterCirclesComponent(
      SimulationContext, SceneId, EntityId, arr, maxCount, out uint actualCount))
    {
      _isSyncing = true;
      foreach (var c in Circles)
        c.PropertyChanged -= Item_PropertyChanged;
      Circles.Clear();

      for (int i = 0; i < actualCount; i++)
      {
        var item = new EmissionCircleItem
        {
          LatitudeDeg = arr[i].LatitudeRad * 180f / (float)Math.PI,
          LongitudeDeg = arr[i].LongitudeRad * 180f / (float)Math.PI,
          CircleRadius = arr[i].CircleRadiusFrac * 1000f,
          Mass = arr[i].Mass * 1000f,
          ColorR = arr[i].ColorR,
          ColorG = arr[i].ColorG,
          ColorB = arr[i].ColorB,
          ColorA = arr[i].ColorA,
          ParticlesPerTick = arr[i].ParticlesPerTick,
          TTL = arr[i].TTL,
          MeanVelocity = arr[i].MeanVelocity,
        };
        item.PropertyChanged += Item_PropertyChanged;
        Circles.Add(item);
      }
      _isSyncing = false;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere Gizmo Component
// ─────────────────────────────────────────────────────────────────────────────

public partial class SphericalGizmoComponent : NativeComponent
{
  public override string Name => "Sphere Gizmo";

  [ObservableProperty]
  private bool _isVisible = true;

  protected override bool ShouldPushToNative(string? propertyName) => true;

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero) return;
    AetherVk.Logic.Services.NativeInterop.avkSimulationContext_setSphereGizmoVisibility(
      SimulationContext, SceneId, EntityId, IsVisible);
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero) return;
    if (AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getSphereGizmoVisibility(
      SimulationContext, SceneId, EntityId, out bool isVisible))
    {
      IsVisible = isVisible;
    }
  }
}

public partial class BillboardComponent : ObservableObject, IComponent
{
    public string Name => "UI Billboard";

    public ViewModels.BillboardViewModel ViewModel { get; }

    public BillboardComponent(ViewModels.BillboardViewModel viewModel)
    {
        ViewModel = viewModel;
        ViewModel.PropertyChanged += (s, e) => {
            if (e.PropertyName == nameof(ViewModels.BillboardViewModel.Opacity))
            {
                OnPropertyChanged(nameof(Opacity));
            }
        };
    }

    public double Opacity
    {
        get => ViewModel.Opacity;
        set => ViewModel.Opacity = value;
    }
}
