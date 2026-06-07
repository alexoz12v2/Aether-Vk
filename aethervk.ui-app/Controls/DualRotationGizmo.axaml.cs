using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Markup.Xaml;
using Avalonia.Media;

namespace AetherVk.Controls;

public partial class DualRotationGizmo : UserControl
{
  public static readonly StyledProperty<float> QuatWProperty = AvaloniaProperty.Register<
    DualRotationGizmo,
    float
  >(nameof(QuatW), 1.0f);
  public static readonly StyledProperty<float> QuatXProperty = AvaloniaProperty.Register<
    DualRotationGizmo,
    float
  >(nameof(QuatX), 0.0f);
  public static readonly StyledProperty<float> QuatYProperty = AvaloniaProperty.Register<
    DualRotationGizmo,
    float
  >(nameof(QuatY), 0.0f);
  public static readonly StyledProperty<float> QuatZProperty = AvaloniaProperty.Register<
    DualRotationGizmo,
    float
  >(nameof(QuatZ), 0.0f);

  public float QuatW
  {
    get => GetValue(QuatWProperty);
    set => SetValue(QuatWProperty, value);
  }
  public float QuatX
  {
    get => GetValue(QuatXProperty);
    set => SetValue(QuatXProperty, value);
  }
  public float QuatY
  {
    get => GetValue(QuatYProperty);
    set => SetValue(QuatYProperty, value);
  }
  public float QuatZ
  {
    get => GetValue(QuatZProperty);
    set => SetValue(QuatZProperty, value);
  }

  private Canvas? _canvas;

  public DualRotationGizmo()
  {
    AvaloniaXamlLoader.Load(this);
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    _canvas = this.FindControl<Canvas>("GizmoCanvas");
    ToolTip.SetTip(
      this,
      "Fixed axes (UVW in Cyan, Yellow, Magenta) represent the Ecliptic J2000 axes.\nThe colored axes (X, -Y, Z) represent the local frame."
    );
    RebuildLines();
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (
      change.Property == QuatWProperty
      || change.Property == QuatXProperty
      || change.Property == QuatYProperty
      || change.Property == QuatZProperty
    )
    {
      RebuildLines();
    }
  }

  private static double[,] BuildRotationFromQuat(double w, double x, double y, double z)
  {
    var m = new double[3, 3];
    m[0, 0] = 1 - 2 * (y * y + z * z);
    m[0, 1] = 2 * (x * y - w * z);
    m[0, 2] = 2 * (x * z + w * y);
    m[1, 0] = 2 * (x * y + w * z);
    m[1, 1] = 1 - 2 * (x * x + z * z);
    m[1, 2] = 2 * (y * z - w * x);
    m[2, 0] = 2 * (x * z - w * y);
    m[2, 1] = 2 * (y * z + w * x);
    m[2, 2] = 1 - 2 * (x * x + y * y);
    return m;
  }

  private static (double px, double py) Project(
    double[,] rot,
    double x,
    double y,
    double z,
    double r
  )
  {
    double wx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z;
    double wy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z;
    double wz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z;
    const double cosA = 0.35355339;
    const double sinA = 0.35355339;
    double px = (wx - wy * cosA) * r;
    double py = (-wz + wy * sinA) * r;
    return (px, py);
  }

  private static (double wx, double wy, double wz) GetWorldVec(
    double[,] rot,
    double x,
    double y,
    double z
  )
  {
    double wx = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z;
    double wy = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z;
    double wz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z;
    return (wx, wy, wz);
  }

  private void RebuildLines()
  {
    if (_canvas == null)
      return;
    _canvas.Children.Clear();

    const double cx = 50,
      cy = 50;
    const double arm = 38;

    var rot = BuildRotationFromQuat(QuatW, QuatX, QuatY, QuatZ);
    var ident = new double[,]
    {
      { 1, 0, 0 },
      { 0, 1, 0 },
      { 0, 0, 1 },
    };

    // Draw fixed UVW axes (Cyan, Yellow, Magenta)
    DrawAxis(ident, cx, cy, arm, 1, 0, 0, Colors.Cyan, "U");
    DrawAxis(ident, cx, cy, arm, 0, -1, 0, Colors.Yellow, "-V");
    DrawAxis(ident, cx, cy, arm, 0, 0, 1, Colors.Magenta, "W");

    // Draw local axes
    DrawAxis(rot, cx, cy, arm, 1, 0, 0, Colors.OrangeRed, "X");
    DrawAxis(rot, cx, cy, arm, 0, -1, 0, Colors.LimeGreen, "-Y");
    DrawAxis(rot, cx, cy, arm, 0, 0, 1, Colors.DodgerBlue, "Z");

    // Draw segmented arcs
    DrawArc(ident, rot, cx, cy, arm, 1, 0, 0, Colors.White); // X-U
    DrawArc(ident, rot, cx, cy, arm, 0, -1, 0, Colors.White); // Y-V
    DrawArc(ident, rot, cx, cy, arm, 0, 0, 1, Colors.White); // Z-W

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

  private void DrawArc(
    double[,] ident,
    double[,] rot,
    double cx,
    double cy,
    double arm,
    double lx,
    double ly,
    double lz,
    Color color
  )
  {
    var (w1x, w1y, w1z) = GetWorldVec(ident, lx, ly, lz);
    var (w2x, w2y, w2z) = GetWorldVec(rot, lx, ly, lz);

    double dot = w1x * w2x + w1y * w2y + w1z * w2z;
    dot = Math.Max(-1.0, Math.Min(1.0, dot));
    double angleDeg = Math.Acos(dot) * 180.0 / Math.PI;

    if (angleDeg < 2.0)
      return; // Too small

    var line = new Polyline
    {
      Stroke = new SolidColorBrush(color) { Opacity = 0.5 },
      StrokeThickness = 1,
      StrokeDashArray = new Avalonia.Collections.AvaloniaList<double>(2, 2),
    };

    int segments = 6;
    for (int i = 0; i <= segments; i++)
    {
      double t = (double)i / segments;
      // Slerp roughly by linear interpolation + normalize
      double ix = w1x * (1 - t) + w2x * t;
      double iy = w1y * (1 - t) + w2y * t;
      double iz = w1z * (1 - t) + w2z * t;
      double len = Math.Sqrt(ix * ix + iy * iy + iz * iz);
      ix /= len;
      iy /= len;
      iz /= len;

      var (px, py) = Project(ident, ix, iy, iz, arm * 0.8); // Arc slightly inward
      line.Points.Add(new Point(cx + px, cy + py));
    }
    _canvas.Children.Add(line);

    // Text block at mid point
    double mx = w1x * 0.5 + w2x * 0.5;
    double my = w1y * 0.5 + w2y * 0.5;
    double mz = w1z * 0.5 + w2z * 0.5;
    double mlen = Math.Sqrt(mx * mx + my * my + mz * mz);
    mx /= mlen;
    my /= mlen;
    mz /= mlen;

    var (mpx, mpy) = Project(ident, mx, my, mz, arm * 0.9);
    var tb = new TextBlock
    {
      Text = $"{angleDeg:F0}°",
      Foreground = new SolidColorBrush(color) { Opacity = 0.8 },
      FontSize = 9,
    };
    Canvas.SetLeft(tb, cx + mpx - 10);
    Canvas.SetTop(tb, cy + mpy - 6);
    _canvas.Children.Add(tb);
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
    var (wx, wy, wz) = GetWorldVec(rot, lx, ly, lz);
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

    var tip = new Ellipse
    {
      Width = 6,
      Height = 6,
      Fill = brush,
    };
    Canvas.SetLeft(tip, cx + ex - 3);
    Canvas.SetTop(tip, cy + ey - 3);
    _canvas.Children.Add(tip);

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
}
