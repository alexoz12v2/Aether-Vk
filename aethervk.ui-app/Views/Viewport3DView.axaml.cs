using AetherVk.Logic.Models;
using System;
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

    if (_viewModel != null)
    {
      _viewModel.Renderer = null;
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
    if (_viewModel?.IsRadialMenuOpen == true &&
        (e.Key == Key.LeftAlt || e.Key == Key.RightAlt))
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

  private void OnBillboardPointerPressed(object? sender, PointerPressedEventArgs e)
  {
      if (sender is Image image && image.DataContext is BillboardViewModel bvm)
      {
          if (_viewModel != null)
          {
              foreach (var b in _viewModel.Billboards)
              {
                  b.IsSelected = false;
              }
          }
          bvm.IsSelected = true;

          // Select this billboard
          var entity = new Entity(_viewModel?.SceneId ?? 0, (ulong)bvm.GetHashCode(), "UI Billboard");
          entity.Components.Add(new BillboardComponent(bvm));
          
          if (_viewModel != null)
          {
              // _viewModel.SceneStateManager.GetOrCreateScene(_viewModel.SceneId).SelectedEntity = entity;
          }
          
          CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
              new AetherVk.Logic.ViewModels.EntitySelectedMessage(entity)
          );

          e.Handled = true;
      }
  }
}
