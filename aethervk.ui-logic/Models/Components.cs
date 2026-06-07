using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Models;

public partial class TransformComponent
  : NativeComponent,
    CommunityToolkit.Mvvm.Messaging.IRecipient<AetherVk.Logic.Messages.TransformUpdatedFromNativeMessage>
{
  public TransformComponent()
  {
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register(this);
  }

  public void Receive(AetherVk.Logic.Messages.TransformUpdatedFromNativeMessage message)
  {
    if (message.SceneId == SceneId && message.EntityId == EntityId)
    {
      PullFromNative();
    }
  }

  [ObservableProperty]
  private string _unitLabel = "AU";

  public override string Name => $"Transform ({UnitLabel})";

  partial void OnUnitLabelChanged(string value)
  {
    OnPropertyChanged(nameof(Name));
  }

  public bool SuspendNotifications { get; set; } = false;

  [ObservableProperty]
  private bool _isPositionEditable = true;

  [ObservableProperty]
  private bool _isRotationEditable = true;

  [ObservableProperty]
  private bool _isScaleEditable = true;

  /// <summary>Tooltip reason shown when position is locked. Null if editable.</summary>
  [ObservableProperty]
  private string? _positionLockedReason;

  /// <summary>Tooltip reason shown when rotation is locked. Null if editable.</summary>
  [ObservableProperty]
  private string? _rotationLockedReason;

  /// <summary>Tooltip reason shown when scale is locked. Null if editable.</summary>
  [ObservableProperty]
  private string? _scaleLockedReason;

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

  private static readonly HashSet<string> _uiOnlyFields = new()
  {
    nameof(IsPositionEditable),
    nameof(IsRotationEditable),
    nameof(IsScaleEditable),
    nameof(PositionLockedReason),
    nameof(RotationLockedReason),
    nameof(ScaleLockedReason),
    nameof(SuspendNotifications),
  };

  protected override bool ShouldPushToNative(string? propertyName)
  {
    return propertyName != null && !_uiOnlyFields.Contains(propertyName);
  }

  protected override void PushToNativeImpl()
  {
    if (SuspendNotifications)
      return;

    if (SimulationContext == IntPtr.Zero)
      return;

#if DEBUG
    System.Console.WriteLine(
      $"[TransformComponent] PushToNativeImpl called! X={PosX}, Y={PosY}, Z={PosZ}. StackTrace:\n{new System.Diagnostics.StackTrace()}"
    );
#endif

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
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiTransform>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
      NativeInterop.avkSimulationContext_setComponent(SimulationContext, SceneId, EntityId, 1, ptr);
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }

  protected override void PullFromNativeImpl()
  {
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiTransform>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          1,
          ptr
        )
      )
      {
        var data =
          System.Runtime.InteropServices.Marshal.PtrToStructure<NativeInterop.FfiTransform>(ptr);
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

        uint frameType = NativeInterop.avkSimulationContext_getEntityReferenceFrameType(
          SimulationContext,
          SceneId,
          EntityId
        );
        UnitLabel = frameType == 1 ? "km" : "AU";
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyPosition()
  {
    string json =
      $"{{ \"x\": {PosX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {PosY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {PosZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyRotation()
  {
    string json =
      $"{{ \"x\": {RotX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {RotY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {RotZ.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"w\": {RotW.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyScale()
  {
    string json =
      $"{{ \"x\": {ScaleX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {ScaleY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {ScaleZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
  }
}

public partial class HighResTransformComponent
  : NativeComponent,
    CommunityToolkit.Mvvm.Messaging.IRecipient<AetherVk.Logic.Messages.EarthObserverModeChangedMessage>
{
  [ObservableProperty]
  private string _unitLabel = "AU";

  public override string Name => $"HighRes Transform ({UnitLabel})";

  public HighResTransformComponent()
  {
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Register(this);
  }

  public void Receive(AetherVk.Logic.Messages.EarthObserverModeChangedMessage message)
  {
    IsEditable = !message.Value;
  }

  partial void OnUnitLabelChanged(string value)
  {
    OnPropertyChanged(nameof(Name));
  }

  public bool SuspendNotifications { get; set; } = false;

  [ObservableProperty]
  private bool _isEditable = true;

  [ObservableProperty]
  private double _posX;

  [ObservableProperty]
  private double _posY;

  [ObservableProperty]
  private double _posZ;

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

    var data = new NativeInterop.FfiHighResTransform
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
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiHighResTransform>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
      NativeInterop.avkSimulationContext_setComponent(SimulationContext, SceneId, EntityId, 3, ptr);
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }

  protected override void PullFromNativeImpl()
  {
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiHighResTransform>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          3,
          ptr
        )
      )
      {
        var data =
          System.Runtime.InteropServices.Marshal.PtrToStructure<NativeInterop.FfiHighResTransform>(
            ptr
          );
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

        uint frameType = NativeInterop.avkSimulationContext_getEntityReferenceFrameType(
          SimulationContext,
          SceneId,
          EntityId
        );
        UnitLabel = frameType == 1 ? "km" : "AU";
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyPosition()
  {
    string json =
      $"{{ \"x\": {PosX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {PosY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {PosZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyRotation()
  {
    string json =
      $"{{ \"x\": {RotX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {RotY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {RotZ.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"w\": {RotW.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void CopyScale()
  {
    string json =
      $"{{ \"x\": {ScaleX.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"y\": {ScaleY.ToString(System.Globalization.CultureInfo.InvariantCulture)}, \"z\": {ScaleZ.ToString(System.Globalization.CultureInfo.InvariantCulture)} }}";
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CopyToClipboardMessage(json)
    );
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
  private float _nearPlane = 0.00001f;

  [ObservableProperty]
  private float _farPlane = 10000.0f;

  [ObservableProperty]
  private float _orthoScaleFactor = 0.01f;

  [ObservableProperty]
  private bool _isOrthographic;

  [ObservableProperty]
  private float _focusDistance = 10.0f;

  private float _nativeLeft;
  private float _nativeRight;
  private float _nativeBottom;
  private float _nativeTop;

  partial void OnIsOrthographicChanged(bool value)
  {
    if (value && SuspendNotifications == false && !IsSyncingFromNative)
    {
      SuspendNotifications = true;
      try
      {
        // bounds logic was removed
      }
      finally
      {
        SuspendNotifications = false;
        PushToNativeImpl();
      }
    }
  }

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

  private bool _isClamping;

  // Near must be positive and strictly less than Far.
  partial void OnNearPlaneChanged(float value)
  {
    if (_isClamping)
      return;
    try
    {
      _isClamping = true;
      const float minNear = 0.00001f;
      float clamped = System.Math.Max(value, minNear);
      if (clamped >= FarPlane)
        clamped = System.Math.Max(FarPlane - 0.0001f, minNear);
      if (clamped != value)
        NearPlane = clamped;
    }
    finally
    {
      _isClamping = false;
    }
  }

  // Far must be strictly greater than Near and at most 10 000 AU.
  partial void OnFarPlaneChanged(float value)
  {
    if (_isClamping)
      return;
    try
    {
      _isClamping = true;
      float clamped = System.Math.Min(System.Math.Max(value, NearPlane + 0.0001f), 10_000.0f);
      if (clamped != value)
        FarPlane = clamped;
    }
    finally
    {
      _isClamping = false;
    }
  }

  protected override void PushToNativeImpl()
  {
    if (SuspendNotifications)
      return;

    // Additional safety clamp: never send invalid values to native even if
    // the OnXxxChanged correction hasn't fired yet.
    float safeFov = System.Math.Min(System.Math.Max(Fov, 0.1f), 179.0f);
    float safeNear = System.Math.Max(NearPlane, 0.00001f);
    float safeFar = System.Math.Max(FarPlane, safeNear + 0.0001f);

    var data = new NativeInterop.FfiCamera
    {
      IsOrthographic = IsOrthographic,
      Fov = safeFov,
      Aspect = AspectRatio,
      Near = safeNear,
      Far = safeFar,
      Left = _nativeLeft,
      Right = _nativeRight,
      Bottom = _nativeBottom,
      Top = _nativeTop,
      FocusDistance = FocusDistance,
      // proj array doesn't matter for pushing
    };
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiCamera>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
      NativeInterop.avkSimulationContext_setComponent(SimulationContext, SceneId, EntityId, 2, ptr);
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }

    // Pull from native immediately to update the projection matrix preview
    PullFromNative();
  }

  protected override void PullFromNativeImpl()
  {
    int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiCamera>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          2,
          ptr
        )
      )
      {
        var data = System.Runtime.InteropServices.Marshal.PtrToStructure<NativeInterop.FfiCamera>(
          ptr
        );
        IsOrthographic = data.IsOrthographic;
        Fov = data.Fov;
        AspectRatio = data.Aspect;
        NearPlane = data.Near;
        FarPlane = data.Far;
        FocusDistance = data.FocusDistance;
        _nativeLeft = data.Left;
        _nativeRight = data.Right;
        _nativeBottom = data.Bottom;
        _nativeTop = data.Top;

        if (data.IsOrthographic)
        {
          // Preserve C# OrthoScaleFactor for UI zooming
        }

        ProjectionMatrixPreview =
          $"[ {data.Proj00, 7:F2} {data.Proj10, 7:F2} {data.Proj20, 7:F2} {data.Proj30, 7:F2} ]\n"
          + $"[ {data.Proj01, 7:F2} {data.Proj11, 7:F2} {data.Proj21, 7:F2} {data.Proj31, 7:F2} ]\n"
          + $"[ {data.Proj02, 7:F2} {data.Proj12, 7:F2} {data.Proj22, 7:F2} {data.Proj32, 7:F2} ]\n"
          + $"[ {data.Proj03, 7:F2} {data.Proj13, 7:F2} {data.Proj23, 7:F2} {data.Proj33, 7:F2} ]";
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
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

  public ObservableCollection<EmissionCircleItem> Jets { get; } = new();

  public CometComponent() { }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical Gizmo Component
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// C# UI model for a single circular emission zone on a comet surface.
/// Angles are stored in degrees; the FFI layer converts to radians.
/// </summary>
public partial class EmissionCircleItem : ObservableObject
{
  [ObservableProperty]
  private ulong _visualEntityId;

  /// <summary>Latitude of the emission circle centre, in degrees (−90 south … +90 north).</summary>
  [ObservableProperty]
  private float _latitudeDeg;

  /// <summary>Longitude of the emission circle centre, in degrees (0 … 360).</summary>
  [ObservableProperty]
  private float _longitudeDeg;

  /// <summary>
  /// Radius of the emission disc in km.
  /// </summary>
  [ObservableProperty]
  private float _circleRadiusKm = 0.1f;

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
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ColorByteR))]
  [NotifyPropertyChangedFor(nameof(ColorArgbUint))]
  private float _colorR = 1.0f;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ColorByteG))]
  [NotifyPropertyChangedFor(nameof(ColorArgbUint))]
  private float _colorG = 0.6f;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ColorByteB))]
  [NotifyPropertyChangedFor(nameof(ColorArgbUint))]
  private float _colorB = 0.2f;

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ColorByteA))]
  [NotifyPropertyChangedFor(nameof(ColorArgbUint))]
  private float _colorA = 1.0f;

  /// <summary>Packed ARGB uint for easy Color conversion in the UI layer.</summary>
  public uint ColorArgbUint
  {
    get
    {
      byte a = (byte)Math.Max(0, Math.Min(255, (int)(ColorA * 255f)));
      byte r = (byte)Math.Max(0, Math.Min(255, (int)(ColorR * 255f)));
      byte g = (byte)Math.Max(0, Math.Min(255, (int)(ColorG * 255f)));
      byte b = (byte)Math.Max(0, Math.Min(255, (int)(ColorB * 255f)));
      return ((uint)a << 24) | ((uint)r << 16) | ((uint)g << 8) | b;
    }
    set
    {
      ColorA = ((value >> 24) & 0xFF) / 255f;
      ColorR = ((value >> 16) & 0xFF) / 255f;
      ColorG = ((value >> 8) & 0xFF) / 255f;
      ColorB = (value & 0xFF) / 255f;
    }
  }

  /// <summary>Byte-range accessors for ColorPicker binding.</summary>
  public byte ColorByteR
  {
    get => (byte)Math.Max(0, Math.Min(255, (int)(ColorR * 255f)));
    set => ColorR = value / 255f;
  }
  public byte ColorByteG
  {
    get => (byte)Math.Max(0, Math.Min(255, (int)(ColorG * 255f)));
    set => ColorG = value / 255f;
  }
  public byte ColorByteB
  {
    get => (byte)Math.Max(0, Math.Min(255, (int)(ColorB * 255f)));
    set => ColorB = value / 255f;
  }
  public byte ColorByteA
  {
    get => (byte)Math.Max(0, Math.Min(255, (int)(ColorA * 255f)));
    set => ColorA = value / 255f;
  }

  // ── Emission Params ─────────────────────────────────────────────────────────
  [ObservableProperty]
  private uint _particlesPerTick = 100;

  /// <summary>Double proxy for UnboundedSlider binding.</summary>
  public double ParticlesPerTickDouble
  {
    get => ParticlesPerTick;
    set => ParticlesPerTick = (uint)Math.Max(0, Math.Round(value));
  }

  partial void OnParticlesPerTickChanged(uint value)
  {
    OnPropertyChanged(nameof(ParticlesPerTickDouble));
  }

  [ObservableProperty]
  private ulong _tTL = 1000;

  /// <summary>Double proxy for UnboundedSlider binding.</summary>
  public double TTLDouble
  {
    get => TTL;
    set => TTL = (ulong)Math.Max(1, Math.Round(value));
  }

  partial void OnTTLChanged(ulong value)
  {
    OnPropertyChanged(nameof(TTLDouble));
  }

  [ObservableProperty]
  private float _meanVelocity = 10.0f;

  [ObservableProperty]
  private float _velocityDirStdDevDeg = 5.0f;

  /// <summary>
  /// Radiation pressure coefficient (dimensionless). ~1.0 for a perfect absorber;
  /// ~2.0 for a perfect reflector. Used by the Barnes-Hut radiation pressure kernel.
  /// </summary>
  [ObservableProperty]
  private float _beta = 1.0f;

  /// <summary>
  /// Maximum number of particles this jet can have alive simultaneously.
  /// Higher values consume more GPU memory. Default: 4096.
  /// </summary>
  [ObservableProperty]
  private uint _maxParticles = 4096;

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
      if (_isSyncing)
        return;
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

  private void Item_PropertyChanged(
    object? sender,
    System.ComponentModel.PropertyChangedEventArgs e
  )
  {
    if (_isSyncing)
      return;
    PushToNativeImpl();
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void AddCircle()
  {
    Circles.Add(new EmissionCircleItem());
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.JetConfigChangedMessage { SceneId = SceneId }
    );
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void RemoveCircle(EmissionCircleItem item)
  {
    if (item.VisualEntityId != 0 && item.VisualEntityId != ulong.MaxValue)
    {
      if (SimulationContext != IntPtr.Zero)
      {
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_removeEntity(
          SimulationContext,
          SceneId,
          item.VisualEntityId
        );
      }
    }
    Circles.Remove(item);
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.JetConfigChangedMessage { SceneId = SceneId }
    );
  }

  protected override bool ShouldPushToNative(string? propertyName) => true;

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    var arr = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle[Circles.Count];
    for (int i = 0; i < Circles.Count; i++)
    {
      arr[i] = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle
      {
        LatitudeRad = Circles[i].LatitudeDeg * (float)Math.PI / 180f,
        LongitudeRad = Circles[i].LongitudeDeg * (float)Math.PI / 180f,
        CircleRadiusKm = Circles[i].CircleRadiusKm,
        Mass = Circles[i].Mass / 1000f,
        ColorR = Circles[i].ColorR,
        ColorG = Circles[i].ColorG,
        ColorB = Circles[i].ColorB,
        ColorA = Circles[i].ColorA,
        ParticlesPerTick = Circles[i].ParticlesPerTick,
        TTL = Circles[i].TTL,
        MeanVelocity = Circles[i].MeanVelocity,
        VelocityDirStdDevRad = Circles[i].VelocityDirStdDevDeg * (float)Math.PI / 180f,
        ChildEntity = Circles[i].VisualEntityId == 0 ? ulong.MaxValue : Circles[i].VisualEntityId,
        Beta = Circles[i].Beta,
        MaxParticles = Circles[i].MaxParticles,
      };
    }
    AetherVk.Logic.Services.NativeInterop.avkSimulationContext_setParticleEmitterCirclesComponent(
      SimulationContext,
      SceneId,
      EntityId,
      arr,
      (uint)arr.Length
    );

    // After updating circles, recalculate jet surface points so child entities are spawned
    AetherVk.Logic.Services.NativeInterop.avkSimulationContext_recalculateJetPoints(
      SimulationContext,
      SceneId,
      EntityId
    );
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    uint maxCount = 64;
    var arr = new AetherVk.Logic.Services.NativeInterop.FfiEmissionCircle[maxCount];
    if (
      AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getParticleEmitterCirclesComponent(
        SimulationContext,
        SceneId,
        EntityId,
        arr,
        maxCount,
        out uint actualCount
      )
    )
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
          LongitudeDeg = (float)(arr[i].LongitudeRad * 180.0 / Math.PI),
          CircleRadiusKm = arr[i].CircleRadiusKm,
          Mass = arr[i].Mass * 1000f,
          ColorR = arr[i].ColorR,
          ColorG = arr[i].ColorG,
          ColorB = arr[i].ColorB,
          ColorA = arr[i].ColorA,
          ParticlesPerTick = arr[i].ParticlesPerTick,
          TTL = arr[i].TTL,
          MeanVelocity = arr[i].MeanVelocity,
          VelocityDirStdDevDeg = arr[i].VelocityDirStdDevRad * 180f / (float)Math.PI,
          VisualEntityId = arr[i].ChildEntity == ulong.MaxValue ? 0 : arr[i].ChildEntity,
          Beta = arr[i].Beta,
          MaxParticles = arr[i].MaxParticles,
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

public partial class SphereGizmoComponent : NativeComponent
{
  public override string Name => "Sphere Gizmo";

  [ObservableProperty]
  private bool _isVisible = true;

  protected override bool ShouldPushToNative(string? propertyName) => true;

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    // Read-modify-write: read the full DTO, update IsVisible, write back
    int size =
      System.Runtime.InteropServices.Marshal.SizeOf<AetherVk.Logic.Models.FfiSphereGizmo>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      var data = new AetherVk.Logic.Models.FfiSphereGizmo();
      // Try to read existing state first to preserve Radius/Subdivisions/LocalFrame
      if (
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          4,
          ptr
        )
      )
      {
        data =
          System.Runtime.InteropServices.Marshal.PtrToStructure<AetherVk.Logic.Models.FfiSphereGizmo>(
            ptr
          );
      }
      data.IsVisible = IsVisible ? (byte)1 : (byte)0;
      System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
      AetherVk.Logic.Services.NativeInterop.avkSimulationContext_setComponent(
        SimulationContext,
        SceneId,
        EntityId,
        4,
        ptr
      );
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    int size =
      System.Runtime.InteropServices.Marshal.SizeOf<AetherVk.Logic.Models.FfiSphereGizmo>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          4,
          ptr
        )
      )
      {
        var data =
          System.Runtime.InteropServices.Marshal.PtrToStructure<AetherVk.Logic.Models.FfiSphereGizmo>(
            ptr
          );
        IsVisible = data.IsVisible != 0;
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }
}

public partial class ScreenSpaceBillboardComponent : NativeComponent
{
  public override string Name => "Screen Space Billboard";

  [ObservableProperty]
  private string _imagePath = "";

  [ObservableProperty]
  private float _ndcX;

  [ObservableProperty]
  private float _ndcY;

  [ObservableProperty]
  private float _scale = 1.0f;

  [ObservableProperty]
  private float _rotationDeg;

  [ObservableProperty]
  private float _opacity = 1.0f;

  [ObservableProperty]
  private int _zIndex = 1;

  [ObservableProperty]
  private ulong _viewportId;

  /// <summary>
  /// When non-null, property changes are synced bidirectionally with this BillboardViewModel
  /// (the Avalonia overlay). A reentrant guard prevents infinite ping-pong.
  /// </summary>
  private AetherVk.Logic.ViewModels.BillboardViewModel? _linkedBillboard;
  private bool _isSyncingToBillboard;

  /// <summary>
  /// Links this component model to its Avalonia overlay ViewModel for bidirectional sync.
  /// Call this right after creating both objects (in InsertBillboard).
  /// </summary>
  public void LinkBillboard(AetherVk.Logic.ViewModels.BillboardViewModel billboard)
  {
    // Unsubscribe from previous
    if (_linkedBillboard != null)
      _linkedBillboard.PropertyChanged -= OnLinkedBillboardPropertyChanged;

    _linkedBillboard = billboard;

    if (_linkedBillboard != null)
      _linkedBillboard.PropertyChanged += OnLinkedBillboardPropertyChanged;
  }

  /// <summary>
  /// Handles property changes from the linked BillboardViewModel (e.g., from Ctrl+Wheel in viewport).
  /// Syncs values back to this NativeComponent so they push to Rust.
  /// </summary>
  private void OnLinkedBillboardPropertyChanged(
    object? sender,
    System.ComponentModel.PropertyChangedEventArgs e
  )
  {
    if (_isSyncingToBillboard || _linkedBillboard == null)
      return;

    _isSyncingToBillboard = true;
    try
    {
      switch (e.PropertyName)
      {
        case nameof(AetherVk.Logic.ViewModels.BillboardViewModel.Opacity):
          Opacity = (float)_linkedBillboard.Opacity;
          break;
        case nameof(AetherVk.Logic.ViewModels.BillboardViewModel.Scale):
          Scale = (float)_linkedBillboard.Scale;
          break;
        case nameof(AetherVk.Logic.ViewModels.BillboardViewModel.Rotation):
          RotationDeg = (float)_linkedBillboard.Rotation;
          break;
      }
    }
    finally
    {
      _isSyncingToBillboard = false;
    }
  }

  protected override bool ShouldPushToNative(string? propertyName)
  {
    // ImagePath and ViewportId are set at creation only, not pushed on change
    return propertyName != nameof(ImagePath) && propertyName != nameof(ViewportId);
  }

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    var data = new NativeInterop.FfiScreenSpaceBillboard
    {
      NdcX = NdcX,
      NdcY = NdcY,
      Scale = Scale,
      RotationDeg = RotationDeg,
      Opacity = Opacity,
      ZIndex = ZIndex,
      ViewportId = ViewportId,
    };

    int size =
      System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiScreenSpaceBillboard>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
      NativeInterop.avkSimulationContext_setComponent(SimulationContext, SceneId, EntityId, 5, ptr);
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }

    // Sync to linked BillboardViewModel (the Avalonia overlay) so the visual updates
    if (!_isSyncingToBillboard && _linkedBillboard != null)
    {
      _isSyncingToBillboard = true;
      try
      {
        _linkedBillboard.Opacity = Opacity;
        _linkedBillboard.Scale = Scale;
        _linkedBillboard.Rotation = RotationDeg;
      }
      finally
      {
        _isSyncingToBillboard = false;
      }
    }
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    int size =
      System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiScreenSpaceBillboard>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          5,
          ptr
        )
      )
      {
        var data =
          System.Runtime.InteropServices.Marshal.PtrToStructure<NativeInterop.FfiScreenSpaceBillboard>(
            ptr
          );
        NdcX = data.NdcX;
        NdcY = data.NdcY;
        Scale = data.Scale;
        RotationDeg = data.RotationDeg;
        Opacity = data.Opacity;
        ZIndex = data.ZIndex;
        ViewportId = data.ViewportId;
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Physical Mesh Component
// ─────────────────────────────────────────────────────────────────────────────

public partial class PhysicalMeshComponent : NativeComponent
{
  public override string Name => "Physical Mesh";

  public const ulong ComponentId = 20;

  [ObservableProperty]
  private bool _isProcedural;

  [ObservableProperty]
  private string _assetPath = string.Empty;

  // ── Nucleus Physical Parameters ──────────────────────────────────────────

  /// <summary>Mass of the nucleus (kg). Used for gravitational interaction and inertia.</summary>
  [ObservableProperty]
  private double _massKg = 1.0;

  /// <summary>Effective radius of the nucleus (km). Used for spherical inertia approximation.</summary>
  [ObservableProperty]
  private double _radiusKm = 1.0;

  /// <summary>Mesh bounding sphere in vertex units (cached from native). Used to compute scale = radius / boundingSphere.</summary>
  private float _boundingSphere = 1.0f;

  /// <summary>Bulk density (kg/m³). Computed from mass and radius, read-only display.</summary>
  public double DensityKgM3
  {
    get
    {
      double r_m = RadiusKm * 1000.0;
      if (r_m <= 0)
        return 0;
      double volume = (4.0 / 3.0) * Math.PI * r_m * r_m * r_m;
      return MassKg / volume;
    }
  }

  partial void OnMassKgChanged(double value) => OnPropertyChanged(nameof(DensityKgM3));

  partial void OnRadiusKmChanged(double value) => OnPropertyChanged(nameof(DensityKgM3));

  // ── IAU Rotational Model ─────────────────────────────────────────────────

  /// <summary>Right ascension of pole at epoch (degrees).</summary>
  [ObservableProperty]
  private double _poleRaDeg = 90.0;

  /// <summary>Declination of pole at epoch (degrees).</summary>
  [ObservableProperty]
  private double _poleDecDeg = 90.0 - AetherVk.Logic.ViewModels.IauRotationMath.ObliquityDeg;

  /// <summary>Prime meridian angle at epoch (degrees).</summary>
  [ObservableProperty]
  private double _primeMeridianDeg = 180.0;

  /// <summary>RA rate of change (degrees/century).</summary>
  [ObservableProperty]
  private double _poleRaRateDeg;

  /// <summary>Dec rate of change (degrees/century).</summary>
  [ObservableProperty]
  private double _poleDecRateDeg;

  /// <summary>Spin rate (degrees/day).</summary>
  [ObservableProperty]
  private double _rotationRateDeg;

  private static readonly HashSet<string> _iauFields = new()
  {
    nameof(PoleRaDeg),
    nameof(PoleDecDeg),
    nameof(PrimeMeridianDeg),
    nameof(PoleRaRateDeg),
    nameof(PoleDecRateDeg),
    nameof(RotationRateDeg),
  };

  protected override bool ShouldPushToNative(string? propertyName) =>
    propertyName != null && (_iauFields.Contains(propertyName) || propertyName == nameof(RadiusKm));

  protected override void PushToNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    // 1. Get current simulation time as Julian Date for the rotational model evaluation
    double currentSimTimeTai = NativeInterop.avkSimulationContext_getSimulationTime(
      SimulationContext,
      SceneId
    );
    // Convert TAI seconds since J1900 to Julian Date
    // J1900 = JD 2415020.0, TAI seconds -> days
    double currentJd = 2415020.0 + (currentSimTimeTai / 86400.0);

    // 2. Push rotational model + recompute rotation via native function
    NativeInterop.avkSimulationContext_setRotationalModel(
      SimulationContext,
      SceneId,
      EntityId,
      PoleRaDeg,
      PoleDecDeg,
      PrimeMeridianDeg,
      PoleRaRateDeg,
      PoleDecRateDeg,
      RotationRateDeg,
      2451545.0, // J2000.0 reference epoch
      currentJd
    );

    // 3. If radius changed, also update the TransformComponent scale
    if (_boundingSphere > 0.0f)
    {
      int size = System.Runtime.InteropServices.Marshal.SizeOf<NativeInterop.FfiTransform>();
      IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
      try
      {
        if (
          NativeInterop.avkSimulationContext_getComponent(
            SimulationContext,
            SceneId,
            EntityId,
            1,
            ptr
          )
        )
        {
          var data =
            System.Runtime.InteropServices.Marshal.PtrToStructure<NativeInterop.FfiTransform>(ptr);
          float meshScale = (float)(RadiusKm / _boundingSphere);
          data.Sx = meshScale;
          data.Sy = meshScale;
          data.Sz = meshScale;
          System.Runtime.InteropServices.Marshal.StructureToPtr(data, ptr, false);
          NativeInterop.avkSimulationContext_setComponent(
            SimulationContext,
            SceneId,
            EntityId,
            1,
            ptr
          );
        }
      }
      finally
      {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
      }
    }

    // 3b. Sync ColliderComponent and SphereGizmoComponent radius
    NativeInterop.avkSimulationContext_syncColliderRadius(
      SimulationContext,
      SceneId,
      EntityId,
      (float)RadiusKm
    );

    // 4. Notify C# TransformComponent to re-pull values
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.TransformUpdatedFromNativeMessage(SceneId, EntityId)
    );

    // 5. Trigger jet recomputation since the mesh orientation changed
    NativeInterop.avkSimulationContext_recalculateJetPoints(SimulationContext, SceneId, EntityId);
  }

  protected override void PullFromNativeImpl()
  {
    if (SimulationContext == IntPtr.Zero)
      return;

    int size =
      System.Runtime.InteropServices.Marshal.SizeOf<AetherVk.Logic.Models.FfiPhysicalMesh>();
    IntPtr ptr = System.Runtime.InteropServices.Marshal.AllocHGlobal(size);
    try
    {
      if (
        AetherVk.Logic.Services.NativeInterop.avkSimulationContext_getComponent(
          SimulationContext,
          SceneId,
          EntityId,
          ComponentId,
          ptr
        )
      )
      {
        var dto =
          System.Runtime.InteropServices.Marshal.PtrToStructure<AetherVk.Logic.Models.FfiPhysicalMesh>(
            ptr
          );
        IsProcedural = dto.IsProcedural != 0;
        // Cache bounding sphere and sphere_radius from native
        if (dto.BoundingSphere > 0.0f)
          _boundingSphere = dto.BoundingSphere;
        if (dto.SphereRadius > 0.0f)
          RadiusKm = dto.SphereRadius;

        if (!IsProcedural)
        {
          AssetPath = dto.AssetPath ?? string.Empty;
        }
        else
        {
          AssetPath = "Procedural UV Sphere";
        }
      }
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(ptr);
    }
  }
}

public partial class SkyComponent : NativeComponent
{
  public override string Name => "Sky";

  protected override void PullFromNativeImpl() { }

  protected override void PushToNativeImpl() { }
}
