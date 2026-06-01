using System;
using AetherVk.Interop;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class Viewport3DView : UserControl, IViewportRenderer
{
  private Viewport3DViewModel? _viewModel;
  private WriteableBitmap? _bitmap;
  private DispatcherTimer? _livelinessTimer;
  private DateTime _lastFrameTime;

  private bool _isMiddleDragging = false;
  private bool _isRightDragging = false;
  private bool _isZoomDragging = false;
  private Avalonia.Point _lastPointerPos;

  private DispatcherTimer? _resizeTimer;
  private Avalonia.Size _pendingSize;

  public Viewport3DView()
  {
    InitializeComponent();

    _livelinessTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
    _livelinessTimer.Tick += OnLivelinessTick;
    _livelinessTimer.Start();

    _resizeTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(150) };
    _resizeTimer.Tick += OnResizeTimerTick;

    SizeChanged += OnSizeChanged;
    KeyUp += OnKeyUp;
  }

  private void OnSizeChanged(object? sender, SizeChangedEventArgs e)
  {
    _pendingSize = e.NewSize;
    _resizeTimer?.Stop();
    _resizeTimer?.Start();
  }

  private void OnResizeTimerTick(object? sender, EventArgs e)
  {
    _resizeTimer?.Stop();

    if (_viewModel == null || _pendingSize.Width <= 0 || _pendingSize.Height <= 0)
      return;

    uint newWidth = (uint)_pendingSize.Width;
    uint newHeight = (uint)_pendingSize.Height;

    if (newWidth == _viewModel.Width && newHeight == _viewModel.Height)
      return;

    _viewModel.Width = newWidth;
    _viewModel.Height = newHeight;

    _bitmap = new WriteableBitmap(
      new Avalonia.PixelSize((int)newWidth, (int)newHeight),
      new Avalonia.Vector(96, 96),
      PixelFormat.Bgra8888,
      AlphaFormat.Opaque
    );

    RenderTargetImage.Source = _bitmap;

    _viewModel.RuntimeService.ResizePresentationEngine(
      _viewModel.SceneId,
      _viewModel.PresentationEngineId,
      newWidth,
      newHeight
    );
  }

  private void OnLivelinessTick(object? sender, EventArgs e)
  {
    if (_viewModel == null || !_viewModel.IsInitialized)
      return;

    var age = (DateTime.Now - _lastFrameTime).TotalMilliseconds;
    // Interpolate from Green (0) to Red (1000ms+)
    var t = Math.Clamp(age / 1000.0, 0.0, 1.0);
    byte r = (byte)(255 * t);
    byte g = (byte)(255 * (1.0 - t));
    LivelinessIndicator.Fill = new SolidColorBrush(Color.FromRgb(r, g, 0));
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    // Unregister previous message listener
    if (_viewModel != null)
    {
      _viewModel.Renderer = null;
      WeakReferenceMessenger.Default.Unregister<EntitySelectedMessage>(this);
      WeakReferenceMessenger.Default.Unregister<AetherVk.Logic.Models.EntityVisibilityChangedMessage>(
        this
      );
    }

    _viewModel = DataContext as Viewport3DViewModel;

    if (_viewModel != null)
    {
      _viewModel.Renderer = this;
      _bitmap = new WriteableBitmap(
        new Avalonia.PixelSize((int)_viewModel.Width, (int)_viewModel.Height),
        new Avalonia.Vector(96, 96),
        PixelFormat.Bgra8888,
        AlphaFormat.Opaque
      );

      RenderTargetImage.Source = _bitmap;

      // Listen for entity selection from Outline panel to sync billboard IsSelected
      WeakReferenceMessenger.Default.Register<EntitySelectedMessage>(
        this,
        (recipient, msg) =>
        {
          if (_viewModel == null)
            return;
          ulong selectedId = msg.SelectedEntity?.Id ?? 0;
          foreach (var b in _viewModel.Billboards)
          {
            b.IsSelected = (b.EntityId != 0 && b.EntityId == selectedId);
          }
        }
      );

      // Listen for entity visibility toggle from Outline panel
      WeakReferenceMessenger.Default.Register<AetherVk.Logic.Models.EntityVisibilityChangedMessage>(
        this,
        (recipient, msg) =>
        {
          if (_viewModel == null)
            return;
          foreach (var b in _viewModel.Billboards)
          {
            if (b.EntityId != 0 && b.EntityId == msg.Entity.Id)
            {
              b.IsEntityVisible = msg.Entity.IsVisible;
            }
          }
        }
      );
    }
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    if (_viewModel != null)
    {
      _viewModel.Renderer = this;
    }
    _livelinessTimer?.Start();
  }

  protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnDetachedFromVisualTree(e);
    if (_viewModel != null)
    {
      _viewModel.Renderer = null;
    }
    _livelinessTimer?.Stop();
    _resizeTimer?.Stop();
  }

  public void UpdateFrame(IntPtr buffer, nuint bufferSize)
  {
    if (_bitmap == null || _viewModel == null)
      return;

    _lastFrameTime = DateTime.Now;

    using (var frame = _bitmap.Lock())
    {
      unsafe
      {
        System.Buffer.MemoryCopy(
          buffer.ToPointer(),
          frame.Address.ToPointer(),
          bufferSize,
          bufferSize
        );
      }
    }
    RenderTargetImage.InvalidateVisual();
  }

  /// <summary>
  /// Catches Alt key release so the radial menu doesn't get stuck when the user
  /// releases Alt before S (the GlobalInputRouter chord won't match in that case).
  /// </summary>
  private void OnKeyUp(object? sender, KeyEventArgs e)
  {
    if (_viewModel?.IsRadialMenuOpen == true && (e.Key == Key.LeftAlt || e.Key == Key.RightAlt))
    {
      _viewModel.CommitRadialMenuSelection();
      e.Handled = true;
    }
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    _lastPointerPos = point.Position;

    if (point.Properties.IsRightButtonPressed && e.KeyModifiers.HasFlag(KeyModifiers.Shift))
    {
      _viewModel?.PerformRaycast(
        point.Position.X,
        point.Position.Y,
        RenderTargetImage.Bounds.Width,
        RenderTargetImage.Bounds.Height
      );
      e.Handled = true;
      return;
    }

    // Suppress native right-click context menu (radial menu is opened via Alt+S instead).
    if (point.Properties.IsRightButtonPressed)
    {
      e.Handled = true;
      return;
    }

    RenderTargetImage.Focus();
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e) { }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    var currentPos = point.Position;

    var deltaX = (float)(currentPos.X - _lastPointerPos.X);
    var deltaY = (float)(currentPos.Y - _lastPointerPos.Y);

    var bounds = RenderTargetImage.Bounds;
    // Filter out huge artificial jumps caused by wrapping
    if (Math.Abs(deltaX) > bounds.Width / 2)
      deltaX = 0;
    if (Math.Abs(deltaY) > bounds.Height / 2)
      deltaY = 0;

    _viewModel?.OperatorStack.ProcessPointerDelta(deltaX, deltaY);

    if (_viewModel != null)
    {
      if (_viewModel.IsRadialMenuOpen)
      {
        // When radial menu is open, track hover instead of updating anchor position
        _viewModel.UpdateRadialMenuHover(currentPos.X, currentPos.Y);
      }
      else
      {
        // Keep last pointer position so the radial menu opens at the right spot
        _viewModel.RadialMenuX = currentPos.X;
        _viewModel.RadialMenuY = currentPos.Y;
      }

      // Check if we need to wrap the cursor (like Blender)
      if (!_viewModel.IsEarthObserverMode && _viewModel.OperatorStack.IsCameraControlEngaged)
      {
        bool wrapped = false;
        double wrapX = currentPos.X;
        double wrapY = currentPos.Y;

        double margin = 2.0;

        if (currentPos.X <= 0)
        {
          wrapX = bounds.Width - margin;
          wrapped = true;
        }
        else if (currentPos.X >= bounds.Width - 1)
        {
          wrapX = margin;
          wrapped = true;
        }

        if (currentPos.Y <= 0)
        {
          wrapY = bounds.Height - margin;
          wrapped = true;
        }
        else if (currentPos.Y >= bounds.Height - 1)
        {
          wrapY = margin;
          wrapped = true;
        }

        if (wrapped)
        {
          var topLevel = TopLevel.GetTopLevel(this);
          if (topLevel != null)
          {
            var screenPt = RenderTargetImage.PointToScreen(new Avalonia.Point(wrapX, wrapY));
            MouseUtils.SetCursorPosition(screenPt, topLevel.RenderScaling);
          }
        }
      }
    }

    _lastPointerPos = currentPos;
  }

  private void OnPointerWheelChanged(object? sender, PointerWheelEventArgs e)
  {
    if (_viewModel != null)
    {
      _viewModel.ProcessPointerWheel((float)e.Delta.Y);
      e.Handled = true;
    }
  }

  // ── Billboard drag-to-translate (only when already selected from Outline) ──
  private bool _isDraggingBillboard;
  private BillboardViewModel? _dragBillboard;
  private Avalonia.Point _dragStartPos;
  private double _dragStartBillboardX;
  private double _dragStartBillboardY;

  /// <summary>
  /// Called when the billboard image is pressed. Only starts drag if the billboard is
  /// already selected (selection is driven by the Outline panel, not by clicking here).
  /// </summary>
  private void OnBillboardDragPressed(object? sender, PointerPressedEventArgs e)
  {
    if (sender is Image image && image.DataContext is BillboardViewModel bvm && bvm.IsSelected)
    {
      _isDraggingBillboard = true;
      _dragBillboard = bvm;
      _dragStartPos = e.GetPosition(this);
      _dragStartBillboardX = bvm.X;
      _dragStartBillboardY = bvm.Y;
      e.Pointer.Capture((IInputElement)sender);
      e.Handled = true;
    }
  }

  private void OnBillboardPointerMoved(object? sender, PointerEventArgs e)
  {
    if (!_isDraggingBillboard || _dragBillboard == null || _viewModel == null)
      return;

    var pos = e.GetPosition(this);
    var dx = pos.X - _dragStartPos.X;
    var dy = pos.Y - _dragStartPos.Y;

    _dragBillboard.X = _dragStartBillboardX + dx;
    _dragBillboard.Y = _dragStartBillboardY + dy;
  }

  private void OnBillboardPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (_isDraggingBillboard && _dragBillboard != null && _viewModel != null)
    {
      SyncBillboardToRust(_dragBillboard);
    }

    _isDraggingBillboard = false;
    _dragBillboard = null;
    e.Pointer.Capture(null);
  }

  // ── Rotation handle state ────────────────────────────────────────────────
  private bool _isRotatingBillboard;
  private BillboardViewModel? _rotateBillboard;

  private void OnRotateHandlePressed(object? sender, PointerPressedEventArgs e)
  {
    if (
      sender is Avalonia.Controls.Shapes.Ellipse ellipse
      && ellipse.DataContext is BillboardViewModel bvm
    )
    {
      _isRotatingBillboard = true;
      _rotateBillboard = bvm;
      e.Pointer.Capture((IInputElement)sender);
      ShowGoniometer(bvm);
      e.Handled = true;
    }
  }

  private void OnRotateHandleMoved(object? sender, PointerEventArgs e)
  {
    if (!_isRotatingBillboard || _rotateBillboard == null)
      return;

    var pos = e.GetPosition(this);
    double cx = _rotateBillboard.X + _rotateBillboard.ScaledWidth / 2.0;
    double cy = _rotateBillboard.Y + _rotateBillboard.ScaledHeight / 2.0;
    double angle = Math.Atan2(pos.X - cx, -(pos.Y - cy)) * (180.0 / Math.PI);

    // Ctrl key snaps to 15° increments
    if (e.KeyModifiers.HasFlag(KeyModifiers.Control))
    {
      angle = Math.Round(angle / 15.0) * 15.0;
    }

    _rotateBillboard.Rotation = angle;
    UpdateGoniometer(_rotateBillboard, angle);
    e.Handled = true;
  }

  private void OnRotateHandleReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (_isRotatingBillboard && _rotateBillboard != null && _viewModel != null)
    {
      SyncBillboardToRust(_rotateBillboard);
    }

    _isRotatingBillboard = false;
    _rotateBillboard = null;
    e.Pointer.Capture(null);
    HideGoniometer();
  }

  // ── Corner scale handles ─────────────────────────────────────────────────
  private bool _isScalingBillboard;
  private BillboardViewModel? _scaleBillboard;
  private double _scaleStartDist;
  private double _scaleStartValue;

  private void OnScaleHandlePressed(object? sender, PointerPressedEventArgs e)
  {
    if (
      sender is Avalonia.Controls.Shapes.Ellipse ellipse
      && ellipse.DataContext is BillboardViewModel bvm
    )
    {
      _isScalingBillboard = true;
      _scaleBillboard = bvm;
      _scaleStartValue = bvm.Scale;

      var pos = e.GetPosition(this);
      double cx = bvm.X + bvm.ScaledWidth / 2.0;
      double cy = bvm.Y + bvm.ScaledHeight / 2.0;
      _scaleStartDist = Math.Max(
        10.0,
        Math.Sqrt((pos.X - cx) * (pos.X - cx) + (pos.Y - cy) * (pos.Y - cy))
      );

      e.Pointer.Capture((IInputElement)sender);
      e.Handled = true;
    }
  }

  private void OnScaleHandleMoved(object? sender, PointerEventArgs e)
  {
    if (!_isScalingBillboard || _scaleBillboard == null)
      return;

    var pos = e.GetPosition(this);
    double cx = _scaleBillboard.X + _scaleBillboard.ScaledWidth / 2.0;
    double cy = _scaleBillboard.Y + _scaleBillboard.ScaledHeight / 2.0;
    double dist = Math.Sqrt((pos.X - cx) * (pos.X - cx) + (pos.Y - cy) * (pos.Y - cy));

    double newScale = _scaleStartValue * (dist / _scaleStartDist);
    _scaleBillboard.Scale = Math.Max(0.1, newScale);
    e.Handled = true;
  }

  private void OnScaleHandleReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (_isScalingBillboard && _scaleBillboard != null && _viewModel != null)
    {
      SyncBillboardToRust(_scaleBillboard);
    }

    _isScalingBillboard = false;
    _scaleBillboard = null;
    e.Pointer.Capture(null);
  }

  // ── Goniometer overlay ───────────────────────────────────────────────────
  private void ShowGoniometer(BillboardViewModel bvm)
  {
    GoniometerOverlay.Children.Clear();
    GoniometerOverlay.IsVisible = true;
    BuildGoniometerVisuals(bvm, bvm.Rotation);
  }

  private void UpdateGoniometer(BillboardViewModel bvm, double angleDeg)
  {
    GoniometerOverlay.Children.Clear();
    BuildGoniometerVisuals(bvm, angleDeg);
  }

  private void HideGoniometer()
  {
    GoniometerOverlay.IsVisible = false;
    GoniometerOverlay.Children.Clear();
  }

  private void BuildGoniometerVisuals(BillboardViewModel bvm, double angleDeg)
  {
    double cx = bvm.X + bvm.ScaledWidth / 2.0;
    double cy = bvm.Y + bvm.ScaledHeight / 2.0;
    double radius = Math.Max(bvm.ScaledWidth, bvm.ScaledHeight) * 0.7;
    if (radius < 40)
      radius = 40;

    var accentBrush = new SolidColorBrush(Color.FromArgb(180, 100, 180, 255));
    var tickBrush = new SolidColorBrush(Color.FromArgb(120, 160, 200, 255));

    // Outer ring
    var ring = new Avalonia.Controls.Shapes.Ellipse
    {
      Width = radius * 2,
      Height = radius * 2,
      Stroke = accentBrush,
      StrokeThickness = 1.5,
      StrokeDashArray = new Avalonia.Collections.AvaloniaList<double> { 4, 3 },
      Fill = null,
    };
    Canvas.SetLeft(ring, cx - radius);
    Canvas.SetTop(ring, cy - radius);
    GoniometerOverlay.Children.Add(ring);

    // 15° tick marks
    for (int deg = 0; deg < 360; deg += 15)
    {
      double rad = deg * Math.PI / 180.0;
      bool major = deg % 90 == 0;
      double innerR = major ? radius * 0.8 : radius * 0.9;
      double outerR = radius;

      var line = new Avalonia.Controls.Shapes.Line
      {
        StartPoint = new Avalonia.Point(cx + innerR * Math.Sin(rad), cy - innerR * Math.Cos(rad)),
        EndPoint = new Avalonia.Point(cx + outerR * Math.Sin(rad), cy - outerR * Math.Cos(rad)),
        Stroke = major ? accentBrush : tickBrush,
        StrokeThickness = major ? 2.0 : 1.0,
      };
      GoniometerOverlay.Children.Add(line);
    }

    // Current angle indicator line (from center to rim)
    double angleRad = angleDeg * Math.PI / 180.0;
    var indicator = new Avalonia.Controls.Shapes.Line
    {
      StartPoint = new Avalonia.Point(cx, cy),
      EndPoint = new Avalonia.Point(
        cx + radius * Math.Sin(angleRad),
        cy - radius * Math.Cos(angleRad)
      ),
      Stroke = accentBrush,
      StrokeThickness = 2.0,
    };
    GoniometerOverlay.Children.Add(indicator);

    // Angle readout text
    var text = new TextBlock
    {
      Text = $"{angleDeg:F1}°",
      FontSize = 12,
      FontWeight = Avalonia.Media.FontWeight.SemiBold,
      Foreground = Brushes.White,
      Background = new SolidColorBrush(Color.FromArgb(180, 30, 30, 40)),
    };
    Canvas.SetLeft(text, cx + radius + 8);
    Canvas.SetTop(text, cy - 8);
    GoniometerOverlay.Children.Add(text);
  }

  /// <summary>
  /// Pushes the BillboardViewModel's current state to Rust via SetScreenSpaceBillboard.
  /// </summary>
  private void SyncBillboardToRust(BillboardViewModel bvm)
  {
    if (bvm.EntityId == 0 || _viewModel == null)
      return;

    float ndcX = _viewModel.Width > 0 ? (float)(bvm.X / _viewModel.Width) : 0f;
    float ndcY = _viewModel.Height > 0 ? (float)(bvm.Y / _viewModel.Height) : 0f;

    _viewModel.RuntimeService.SetScreenSpaceBillboard(
      _viewModel.SceneId,
      bvm.EntityId,
      ndcX,
      ndcY,
      (float)bvm.Scale,
      (float)bvm.Rotation,
      (float)bvm.Opacity,
      bvm.ZIndex
    );
  }

  private void OnBillboardWheelChanged(object? sender, PointerWheelEventArgs e)
  {
    if (sender is Image image && image.DataContext is BillboardViewModel bvm && bvm.IsSelected)
    {
      if (e.KeyModifiers.HasFlag(KeyModifiers.Control))
      {
        // Ctrl + Wheel = uniform scale
        double delta = e.Delta.Y > 0 ? 0.1 : -0.1;
        bvm.Scale = Math.Max(0.1, bvm.Scale + delta);
        e.Handled = true;
      }
      else if (e.KeyModifiers.HasFlag(KeyModifiers.Shift))
      {
        // Shift + Wheel = opacity
        double delta = e.Delta.Y > 0 ? 0.05 : -0.05;
        bvm.Opacity = Math.Clamp(bvm.Opacity + delta, 0.0, 1.0);
        e.Handled = true;
      }

      // Sync back to Rust if ECS-linked
      if (e.Handled && bvm.EntityId != 0 && _viewModel != null)
      {
        SyncBillboardToRust(bvm);
      }
    }
  }
}
