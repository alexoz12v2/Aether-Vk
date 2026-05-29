using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Media;

namespace AetherVk.Controls;

/// <summary>
/// A 3-axis rotation gizmo.
/// Axes in local 3D space:
///   Red   = +X (Right)
///   Green = -Y (Forward — foreshortened 45°, always visible)
///   Blue  = +Z (Up)
/// Mouse drag: horizontal → Yaw, vertical → Pitch.
/// </summary>
public partial class RotationGizmo : UserControl
{
  // ──────────────────────────────────────────────────────── Avalonia Properties

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

  // ──────────────────────────────────────────────────────── Constructor

  private Canvas? _canvas;

  public RotationGizmo()
  {
    InitializeComponent();
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    _canvas = this.FindControl<Canvas>("GizmoCanvas");
    RebuildLines();
  }

  // ──────────────────────────────────────────────────────── Property change

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (
      change.Property == PitchProperty
      || change.Property == YawProperty
      || change.Property == RollProperty
    )
    {
      RebuildLines();
    }
  }

  // ──────────────────────────────────────────────────────── 3-D projection

  // Returns a rotation matrix from Pitch/Yaw/Roll (degrees, ZYX extrinsic order).
  private static double[,] BuildRotation(double pitchDeg, double yawDeg, double rollDeg)
  {
    double p = pitchDeg * Math.PI / 180.0;
    double y = yawDeg * Math.PI / 180.0;
    double r = rollDeg * Math.PI / 180.0;

    // Individual rotation matrices: Rx(p) * Ry(y) * Rz(r)
    double cp = Math.Cos(p),
      sp = Math.Sin(p);
    double cy = Math.Cos(y),
      sy = Math.Sin(y);
    double cr = Math.Cos(r),
      sr = Math.Sin(r);

    // Combined: R = Rx * Ry * Rz
    var m = new double[3, 3];
    m[0, 0] = cy * cr;
    m[0, 1] = cy * sr;
    m[0, 2] = -sy;

    m[1, 0] = sp * sy * cr - cp * sr;
    m[1, 1] = sp * sy * sr + cp * cr;
    m[1, 2] = sp * cy;

    m[2, 0] = cp * sy * cr + sp * sr;
    m[2, 1] = cp * sy * sr - sp * cr;
    m[2, 2] = cp * cy;
    return m;
  }

  // Project a 3-D unit vector to canvas 2-D using cabinet (oblique) projection:
  //   X → screen right, Z → screen up (−canvas Y),
  //   Y → foreshortened at 45°, blended into both axes so all three
  //       axes remain visible regardless of orientation.
  private static (double px, double py) Project(
    double[,] rot,
    double x,
    double y,
    double z,
    double r
  )
  {
    // Transform local axis vector by the current rotation matrix.
    double wx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z;
    double wy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z;
    double wz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z;

    // Cabinet oblique: Y depth projected at 45°, scaled to 50% length.
    const double cosA = 0.35355339; // cos(45°) × 0.5
    const double sinA = 0.35355339; // sin(45°) × 0.5

    double px = (wx + wy * cosA) * r;
    double py = (-wz - wy * sinA) * r; // Z is "up" → negate for canvas
    return (px, py);
  }

  // ──────────────────────────────────────────────────────── Draw

  private void RebuildLines()
  {
    if (_canvas == null)
      return;
    _canvas.Children.Clear();

    const double cx = 50,
      cy = 50; // centre of canvas
    const double arm = 38; // axis arm length in canvas units

    var rot = BuildRotation(Pitch, Yaw, Roll);

    // Axis definitions: (localX, localY, localZ, color, label)
    // Red  = +X Right, Green = -Y Forward, Blue = +Z Up
    DrawAxis(rot, cx, cy, arm, 1, 0, 0, Colors.OrangeRed, "X");
    DrawAxis(rot, cx, cy, arm, 0, -1, 0, Colors.LimeGreen, "-Y");
    DrawAxis(rot, cx, cy, arm, 0, 0, 1, Colors.DodgerBlue, "Z");

    // Centre dot
    var dot = new Ellipse
    {
      Width = 5,
      Height = 5,
      Fill = new SolidColorBrush(Colors.White),
    };
    Canvas.SetLeft(dot, cx - 2.5);
    Canvas.SetTop(dot, cy - 2.5);
    _canvas.Children.Add(dot);
  }

  private void DrawAxis(
    double[,] rot,
    double cx,
    double cy,
    double arm,
    double lx,
    double ly,
    double lz,
    Color color,
    string label
  )
  {
    var (ex, ey) = Project(rot, lx, ly, lz, arm);

    // Depth value (W-component along original Y, used for dot/opacity fade)
    double wx = rot[0, 0] * lx + rot[0, 1] * ly + rot[0, 2] * lz;
    double wy = rot[1, 0] * lx + rot[1, 1] * ly + rot[1, 2] * lz;
    // Depth ≈ wy (Y-depth in world, range -1..1); behind = dimmer
    double opacity = 0.35 + 0.65 * ((wy + 1.0) * 0.5);

    var brush = new SolidColorBrush(color) { Opacity = opacity };

    var line = new Line
    {
      StartPoint = new Point(cx, cy),
      EndPoint = new Point(cx + ex, cy + ey),
      Stroke = brush,
      StrokeThickness = 2.5,
    };
    _canvas.Children.Add(line);

    // Arrowhead dot at tip
    var tip = new Ellipse
    {
      Width = 6,
      Height = 6,
      Fill = brush,
    };
    Canvas.SetLeft(tip, cx + ex - 3);
    Canvas.SetTop(tip, cy + ey - 3);
    _canvas.Children.Add(tip);

    // Axis label
    var tb = new TextBlock
    {
      Text = label,
      Foreground = brush,
      FontSize = 10,
      FontWeight = Avalonia.Media.FontWeight.Bold,
    };
    Canvas.SetLeft(tb, cx + ex + (ex >= 0 ? 3 : -13));
    Canvas.SetTop(tb, cy + ey - 6);
    _canvas.Children.Add(tb);
  }

  // ──────────────────────────────────────────────────────── Mouse drag

  private bool _isDragging;
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
    if (!_isDragging)
      return;
    var pos = e.GetPosition(this);
    var dx = pos.X - _lastPos.X;
    var dy = pos.Y - _lastPos.Y;
    _lastPos = pos;
    Roll -= (float)(dx * 1.2);
    Pitch -= (float)(dy * 1.2);

    if (Roll > 180)
      Roll -= 360;
    if (Roll < -180)
      Roll += 360;
    if (Pitch > 180)
      Pitch -= 360;
    if (Pitch < -180)
      Pitch += 360;

    e.Handled = true;
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
