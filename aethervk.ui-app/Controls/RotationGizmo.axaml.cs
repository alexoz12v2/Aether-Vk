using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace AetherVk.Controls;

public partial class RotationGizmo : UserControl
{
  public static readonly StyledProperty<float> PitchProperty = AvaloniaProperty.Register<
    RotationGizmo,
    float
  >(nameof(Pitch));

  public static readonly StyledProperty<float> YawProperty = AvaloniaProperty.Register<
    RotationGizmo,
    float
  >(nameof(Yaw));

  public static readonly StyledProperty<float> RollProperty = AvaloniaProperty.Register<
    RotationGizmo,
    float
  >(nameof(Roll));

  public float Pitch
  {
    get => GetValue(PitchProperty);
    set => SetValue(PitchProperty, value);
  }

  public float Yaw
  {
    get => GetValue(YawProperty);
    set => SetValue(YawProperty, value);
  }

  public float Roll
  {
    get => GetValue(RollProperty);
    set => SetValue(RollProperty, value);
  }

  public RotationGizmo()
  {
    InitializeComponent();
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (change.Property == YawProperty)
    {
      YawScaleValue = Math.Cos(Yaw * Math.PI / 180.0);
    }
  }

  public static readonly StyledProperty<double> YawScaleValueProperty = AvaloniaProperty.Register<
    RotationGizmo,
    double
  >(nameof(YawScaleValue), 1.0);

  public double YawScaleValue
  {
    get => GetValue(YawScaleValueProperty);
    set => SetValue(YawScaleValueProperty, value);
  }

  private bool _isDragging = false;
  private Point _lastPos;

  private void OnPointerPressed(object? sender, Avalonia.Input.PointerPressedEventArgs e)
  {
      if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
      {
          _isDragging = true;
          _lastPos = e.GetPosition(this);
          e.Handled = true;
      }
  }

  private void OnPointerMoved(object? sender, Avalonia.Input.PointerEventArgs e)
  {
      if (_isDragging)
      {
          var pos = e.GetPosition(this);
          var dx = pos.X - _lastPos.X;
          var dy = pos.Y - _lastPos.Y;
          _lastPos = pos;

          Yaw += (float)(dx * 1.5);
          Pitch += (float)(dy * 1.5); 

          if (Yaw > 180) Yaw -= 360;
          if (Yaw < -180) Yaw += 360;
          if (Pitch > 180) Pitch -= 360;
          if (Pitch < -180) Pitch += 360;

          e.Handled = true;
      }
  }

  private void OnPointerReleased(object? sender, Avalonia.Input.PointerReleasedEventArgs e)
  {
      if (_isDragging)
      {
          _isDragging = false;
          e.Handled = true;
      }
  }
}
