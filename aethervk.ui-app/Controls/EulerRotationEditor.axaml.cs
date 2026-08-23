using System;
using System.Numerics;
using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Controls;

/// <summary>
/// Control with custom properties to manipulate a quaternion either by manually setting its
/// rotation property, or by using its own sliders to maninpulate the angles, and convert it to
/// quaternion inside
///
/// Note: In comparison to its previous version, this class is purely math/UI, without any knowledge
/// of the simulation engine underneath
/// </summary>
public partial class EulerRotationEditor : UserControl
{
  public static readonly StyledProperty<Quaternion?> RotationProperty = AvaloniaProperty.Register<
    EulerRotationEditor,
    Quaternion?
  >(nameof(Quaternion));

  public Quaternion? Rotation
  {
    get => GetValue(RotationProperty);
    set => SetValue(RotationProperty, value);
  }

  public static readonly StyledProperty<double> RotXDegProperty = AvaloniaProperty.Register<
    EulerRotationEditor,
    double
  >(nameof(RotXDeg));

  public double RotXDeg
  {
    get => GetValue(RotXDegProperty);
    set => SetValue(RotXDegProperty, value);
  }

  public static readonly StyledProperty<double> RotYDegProperty = AvaloniaProperty.Register<
    EulerRotationEditor,
    double
  >(nameof(RotYDeg));

  public double RotYDeg
  {
    get => GetValue(RotYDegProperty);
    set => SetValue(RotYDegProperty, value);
  }

  public static readonly StyledProperty<double> RotZDegProperty = AvaloniaProperty.Register<
    EulerRotationEditor,
    double
  >(nameof(RotZDeg));

  public double RotZDeg
  {
    get => GetValue(RotZDegProperty);
    set => SetValue(RotZDegProperty, value);
  }

  private bool _isUpdatingFromComponent;
  private bool _isUpdatingFromSliders;

  public EulerRotationEditor()
  {
    InitializeComponent();
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);

    if (change.Property == RotationProperty)
    {
      SyncFromQuaternion();
    }
    else if (
      change.Property == RotXDegProperty
      || change.Property == RotYDegProperty
      || change.Property == RotZDegProperty
    )
    {
      SyncToQuaternion();
    }
  }

  private void SyncFromQuaternion()
  {
    if (_isUpdatingFromSliders || Rotation == null)
      return;
    _isUpdatingFromComponent = true;

    try
    {
      var q = Rotation.Value;
      var (rxDeg, ryDeg, rzDeg) = FromQuatToDegRot(q);
      RotXDeg = rxDeg;
      RotYDeg = ryDeg;
      RotZDeg = rzDeg;
    }
    finally
    {
      _isUpdatingFromComponent = false;
    }
  }

  private void SyncToQuaternion()
  {
    if (_isUpdatingFromComponent)
      return;
    _isUpdatingFromSliders = true;

    try
    {
      Rotation = FromDegRotToQuat(RotXDeg, RotYDeg, RotZDeg);
    }
    finally
    {
      _isUpdatingFromSliders = false;
    }
  }

  private static (double, double, double) FromQuatToDegRot(Quaternion quat)
  {
    double x = quat.X;
    double y = quat.Y;
    double z = quat.Z;
    double w = quat.W;

    // Quat to Euler (X, Y, Z standard ZYX extrinsic)
    double sinr_cosp = 2.0 * (w * x + y * z);
    double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    double rx = Math.Atan2(sinr_cosp, cosr_cosp);

    double sinp = 2.0 * (w * y - z * x);
    double ry;
    if (Math.Abs(sinp) >= 1)
      ry = Math.CopySign(Math.PI / 2.0, sinp);
    else
      ry = Math.Asin(sinp);

    double siny_cosp = 2.0 * (w * z + x * y);
    double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    double rz = Math.Atan2(siny_cosp, cosy_cosp);

    // Convert radians to degrees
    double rxDeg = rx * 180.0 / Math.PI % 360.0;
    double ryDeg = ry * 180.0 / Math.PI % 360.0;
    double rzDeg = rz * 180.0 / Math.PI % 360.0;

    return (rxDeg, ryDeg, rzDeg);
  }

  // Note: not using `Quaternion.FromYawPitchRoll ` to preserve original behaviour. Try it if needed
  private static Quaternion FromDegRotToQuat(double rotDegX, double rotDegY, double rotDegZ)
  {
    double rx = rotDegX * Math.PI / 180.0;
    double ry = rotDegY * Math.PI / 180.0;
    double rz = rotDegZ * Math.PI / 180.0;

    float cr = (float)Math.Cos(rx * 0.5);
    float sr = (float)Math.Sin(rx * 0.5);
    float cp = (float)Math.Cos(ry * 0.5);
    float sp = (float)Math.Sin(ry * 0.5);
    float cy = (float)Math.Cos(rz * 0.5);
    float sy = (float)Math.Sin(rz * 0.5);

    float w = cr * cp * cy + sr * sp * sy;
    float x = sr * cp * cy - cr * sp * sy;
    float y = cr * sp * cy + sr * cp * sy;
    float z = cr * cp * sy - sr * sp * cy;

    return new Quaternion
    {
      X = x,
      Y = y,
      Z = z,
      W = w,
    };
  }
}
