using System;
using AetherVk.Logic.Models;
using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Controls;

public partial class EulerRotationEditor : UserControl
{
  public static readonly StyledProperty<NativeComponent?> ComponentProperty =
    AvaloniaProperty.Register<EulerRotationEditor, NativeComponent?>(nameof(Component));

  public NativeComponent? Component
  {
    get => GetValue(ComponentProperty);
    set => SetValue(ComponentProperty, value);
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

    if (change.Property == ComponentProperty)
    {
      if (change.OldValue is NativeComponent oldComp)
        oldComp.PropertyChanged -= OnComponentPropertyChanged;

      if (change.NewValue is NativeComponent newComp)
      {
        newComp.PropertyChanged += OnComponentPropertyChanged;
        SyncFromComponent();
      }
    }
    else if (
      change.Property == RotXDegProperty
      || change.Property == RotYDegProperty
      || change.Property == RotZDegProperty
    )
    {
      SyncToComponent();
    }
  }

  private void OnComponentPropertyChanged(
    object? sender,
    System.ComponentModel.PropertyChangedEventArgs e
  )
  {
    if (
      e.PropertyName == "RotW"
      || e.PropertyName == "RotX"
      || e.PropertyName == "RotY"
      || e.PropertyName == "RotZ"
    )
    {
      SyncFromComponent();
    }
  }

  private void SyncFromComponent()
  {
    if (_isUpdatingFromSliders || Component == null)
      return;
    _isUpdatingFromComponent = true;

    try
    {
      double w = 1.0,
        x = 0.0,
        y = 0.0,
        z = 0.0;
      if (Component is TransformComponent tc)
      {
        w = tc.RotW;
        x = tc.RotX;
        y = tc.RotY;
        z = tc.RotZ;
      }
      else if (Component is HighResTransformComponent htc)
      {
        w = htc.RotW;
        x = htc.RotX;
        y = htc.RotY;
        z = htc.RotZ;
      }

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
      double rxDeg = (rx * 180.0 / Math.PI) % 360.0;
      double ryDeg = (ry * 180.0 / Math.PI) % 360.0;
      double rzDeg = (rz * 180.0 / Math.PI) % 360.0;

      if (rxDeg < 0)
        rxDeg += 360.0;
      if (ryDeg < 0)
        ryDeg += 360.0;
      if (rzDeg < 0)
        rzDeg += 360.0;

      RotXDeg = rxDeg;
      RotYDeg = ryDeg;
      RotZDeg = rzDeg;
    }
    finally
    {
      _isUpdatingFromComponent = false;
    }
  }

  private void SyncToComponent()
  {
    if (_isUpdatingFromComponent || Component == null)
      return;
    _isUpdatingFromSliders = true;

    try
    {
      double rx = RotXDeg * Math.PI / 180.0;
      double ry = RotYDeg * Math.PI / 180.0;
      double rz = RotZDeg * Math.PI / 180.0;

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

      if (Component is TransformComponent tc)
      {
        tc.RotW = w;
        tc.RotX = x;
        tc.RotY = y;
        tc.RotZ = z;
      }
      else if (Component is HighResTransformComponent htc)
      {
        htc.RotW = w;
        htc.RotX = x;
        htc.RotY = y;
        htc.RotZ = z;
      }
    }
    finally
    {
      _isUpdatingFromSliders = false;
    }
  }
}
