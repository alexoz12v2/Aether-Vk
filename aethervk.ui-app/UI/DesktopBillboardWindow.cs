using System;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace AetherVk.UI;

/// <summary>
/// One desktop billboard: a borderless, transparent window a little larger than the (rotated)
/// image, holding the image and its gizmo. Pointer rules live in
/// <see cref="BillboardInteractionController"/>; every edit goes to
/// <see cref="DesktopBillboardService"/> and comes back through <see cref="Apply"/>.
/// </summary>
public sealed class DesktopBillboardWindow : Window
{
  private readonly BillboardInteractionController _controller = new();
  private readonly Action<DesktopBillboard> _edit;
  private readonly Canvas _canvas = new() { Background = Brushes.Transparent };
  private readonly Image _image = new() { Stretch = Stretch.Fill };
  private readonly Border _placeholder = new()
  {
    Background = new SolidColorBrush(Color.FromArgb(160, 60, 60, 60)),
    BorderBrush = Brushes.OrangeRed,
    BorderThickness = new Thickness(2),
    Child = new TextBlock
    {
      Text = "texture not loaded",
      Foreground = Brushes.White,
      HorizontalAlignment = HorizontalAlignment.Center,
      VerticalAlignment = VerticalAlignment.Center,
    },
  };
  private readonly BillboardGizmoLayer _gizmo = new();
  private DesktopBillboard _billboard;
  private RectI _windowRect;

  public Guid BillboardId => _billboard.Id;

  public DesktopBillboard Billboard => _billboard;

  /// <summary>Window-local physical-pixel rectangles that take pointer input (X11 input shape).</summary>
  public event Action<DesktopBillboardWindow>? InputRegionChanged;

  public DesktopBillboardWindow(DesktopBillboard billboard, Action<DesktopBillboard> edit)
  {
    _billboard = billboard;
    _edit = edit;
    Title = "AetherVk billboard";
    SystemDecorations = SystemDecorations.None;
    Background = Brushes.Transparent;
    TransparencyLevelHint = [WindowTransparencyLevel.Transparent];
    ShowInTaskbar = false;
    ShowActivated = false;
    Focusable = false;
    CanResize = false;
    SizeToContent = SizeToContent.Manual;
    _canvas.Children.Add(_image);
    _canvas.Children.Add(_placeholder);
    _canvas.Children.Add(_gizmo);
    Content = _canvas;
    _canvas.PointerPressed += OnPointerPressed;
    _canvas.PointerMoved += OnPointerMoved;
    _canvas.PointerReleased += OnPointerReleased;
    _canvas.PointerCaptureLost += (_, _) => _controller.Cancel();
    ScalingChanged += (_, _) => Apply(_billboard);
  }

  /// <summary>The image to show; null shows the "texture not loaded" placeholder.</summary>
  public void SetBitmap(Bitmap? bitmap)
  {
    _image.Source = bitmap;
    _image.IsVisible = bitmap is not null;
    _placeholder.IsVisible = bitmap is null;
  }

  private double Scale => RenderScaling > 0 ? RenderScaling : 1;

  /// <summary>Places the window and its content for <paramref name="b"/>.</summary>
  public void Apply(DesktopBillboard b)
  {
    var oldMode = _billboard.GizmoMode;
    var oldRect = _windowRect;
    _billboard = b;
    _windowRect = DesktopBillboardGeometry.WindowRect(b);
    double s = Scale;
    Position = new PixelPoint(_windowRect.X, _windowRect.Y);
    Width = _windowRect.Width / s;
    Height = _windowRect.Height / s;

    double w = b.WidthPx / s, h = b.HeightPx / s;
    double left = (b.CenterX - _windowRect.X) / s - w / 2;
    double top = (b.CenterY - _windowRect.Y) / s - h / 2;
    foreach (Control c in new Control[] { _image, _placeholder })
    {
      c.Width = w;
      c.Height = h;
      Canvas.SetLeft(c, left);
      Canvas.SetTop(c, top);
      c.RenderTransformOrigin = RelativePoint.Center;
      c.RenderTransform = new RotateTransform(b.RotationDeg);
      c.Opacity = b.Alpha;
    }
    _gizmo.Width = _windowRect.Width / s;
    _gizmo.Height = _windowRect.Height / s;
    _gizmo.Update(b, _windowRect, s);
    if (oldMode != b.GizmoMode || oldRect != _windowRect)
      InputRegionChanged?.Invoke(this);
  }

  /// <summary>Desktop physical pixel of a pointer event.</summary>
  private PointD Desktop(PointerEventArgs e)
  {
    var p = e.GetPosition(this);
    double s = Scale;
    return new PointD(Position.X + p.X * s, Position.Y + p.Y * s);
  }

  private static BillboardPointerButton Button(PointerUpdateKind k) => k switch
  {
    PointerUpdateKind.LeftButtonPressed or PointerUpdateKind.LeftButtonReleased => BillboardPointerButton.Left,
    PointerUpdateKind.RightButtonPressed or PointerUpdateKind.RightButtonReleased => BillboardPointerButton.Right,
    _ => BillboardPointerButton.Other,
  };

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    var p = Desktop(e);
    var kind = e.GetCurrentPoint(this).Properties.PointerUpdateKind;
    // presses on the transparent margin that is not a handle are not ours (X11 lets them through
    // already; this covers platforms without an input shape)
    if (!DesktopBillboardGeometry.IsOnImage(_billboard, p)
        && DesktopBillboardGeometry.HitTest(_billboard, _billboard.GizmoMode, p) is null)
      return;
    Push(_controller.PointerPressed(_billboard, Button(kind), p));
    e.Pointer.Capture(_canvas);
    e.Handled = true;
  }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    if (!_controller.IsDragging)
      return;
    var m = e.KeyModifiers;
    Push(_controller.PointerMoved(Desktop(e), m.HasFlag(KeyModifiers.Control), m.HasFlag(KeyModifiers.Shift)));
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    var m = e.KeyModifiers;
    Push(_controller.PointerReleased(
      Button(e.InitialPressMouseButton == MouseButton.Right ? PointerUpdateKind.RightButtonReleased
        : e.InitialPressMouseButton == MouseButton.Left ? PointerUpdateKind.LeftButtonReleased
        : PointerUpdateKind.Other),
      Desktop(e),
      m.HasFlag(KeyModifiers.Control),
      m.HasFlag(KeyModifiers.Shift)));
    e.Pointer.Capture(null);
  }

  private void Push(DesktopBillboard? b)
  {
    if (b is null)
      return;
    // show it at once (the service echo arrives a dispatcher tick later)
    Apply(b);
    _edit(b);
  }
}

/// <summary>Draws the gizmo of a billboard in its window (window-local DIPs).</summary>
public sealed class BillboardGizmoLayer : Control
{
  private DesktopBillboard? _b;
  private RectI _win;
  private double _scale = 1;

  private static readonly IPen Outline = new Pen(new SolidColorBrush(Color.FromArgb(220, 255, 200, 60)), 1.5);
  private static readonly IBrush HandleFill = new SolidColorBrush(Color.FromArgb(230, 255, 200, 60));
  private static readonly IPen HandleEdge = new Pen(Brushes.Black, 1);

  public BillboardGizmoLayer()
  {
    IsHitTestVisible = false;
  }

  public void Update(DesktopBillboard b, RectI win, double scale)
  {
    _b = b;
    _win = win;
    _scale = scale;
    InvalidateVisual();
  }

  private Point Local(PointD desktop) => new((desktop.X - _win.X) / _scale, (desktop.Y - _win.Y) / _scale);

  public override void Render(DrawingContext ctx)
  {
    if (_b is not { } b || b.GizmoMode == BillboardGizmoMode.None)
      return;
    var cs = DesktopBillboardGeometry.Corners(b);
    for (int i = 0; i < 4; i++)
      ctx.DrawLine(Outline, Local(cs[i]), Local(cs[(i + 1) % 4]));
    var centre = Local(DesktopBillboardGeometry.Center(b));
    foreach (var h in DesktopBillboardGeometry.Handles(b, b.GizmoMode))
    {
      var c = Local(h.Center);
      double r = h.Radius / _scale;
      switch (h.Kind)
      {
        case BillboardHandle.MoveAlongX:
        case BillboardHandle.MoveAlongY:
          ctx.DrawLine(Outline, centre, c);
          ctx.DrawEllipse(HandleFill, HandleEdge, c, r, r);
          break;
        case BillboardHandle.Rotate:
          double ring = Math.Sqrt(Math.Pow(c.X - centre.X, 2) + Math.Pow(c.Y - centre.Y, 2));
          ctx.DrawEllipse(null, Outline, centre, ring, ring);
          ctx.DrawEllipse(HandleFill, HandleEdge, c, r, r);
          break;
        default:
          ctx.DrawRectangle(HandleFill, HandleEdge, new Rect(c.X - r, c.Y - r, 2 * r, 2 * r));
          break;
      }
    }
  }
}
